"""Drive a quantization search to its deadline through a Humanize flow.

The loop resends the task after idle or recoverable failed turns and stops on fatal failures or the GPU
deadline. While the search runs, Humanize's watchdog gives every CLI a window of at least
`WATCHDOG_FLOOR_S` (a launcher's `HUMANIZE_WATCHDOG` stands), so a tool call of a few minutes is not taken
for a silent CLI; a turn the watchdog does stop is a failed turn, resent like any other, and only
`MAX_WATCHDOG_STOPS` of them in a row end the run. Humanize handles retries and fallbacks. At shutdown, the
host drains pending checks, selects the submission and writes the final result and report.

`qda_search` sends `/goal`; `qda_search_plain` sends the same instruction as a plain prompt.

Run dir layout::

    run_dir/
      work/          the agent's workdir (TASK.md, CLOCK.md, notes/, scripts/, bin/ = the four commands,
                     .qda/rpc.sock = the host's socket; a private short directory bound there when
                     the run dir is too long for a unix socket, `rpc.private_dir`)
      flow.jsonl     what the flow did (goals sent, turns, failures, cut-off, final)
      tools.jsonl    every tool call the agent made (supervisor log, outside the sandbox); a Codex agent's calls
                     are read from its session log when the run ends (`source: codex-session-log`)
      FINAL.json     the selected submission (written by the host's judge)
      result.json    the run in one object (how the agent phase ended, the pick, counts, the epic)
      END-REPORT.md  the end report
      epic/          Humanize's trace of the run (every turn and tool call), moved here when the run ends
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence, Sized
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from hmz.coganchor.agents import Failed
from hmz.coganchor.agents.watchdog import WATCHDOG, silence
from hmz.flows import (
    Agent,
    AgentCollection,
    Budget,
    BudgetExceeded,
    Env,
    EnvCollection,
    FlowContext,
    FlowParams,
    GoalCommandAgentMixin,
    HarnessError,
    HarnessMissing,
    HarnessNotInstalled,
    HarnessRefused,
    HarnessSandboxed,
    HarnessUnrecoverable,
    ModelUnavailable,
    PreToolUseHookParams,
    PreToolUseHookResult,
    SessionError,
    UnsupportedOperation,
    flow,
)
from pydantic import Field

from qda_agent import agents, isolation, prompt, rpc, workspace
from qda_agent.config import RunConfig
from qda_agent.host import (
    CLOSING,
    STOPPING,
    Final,
    HandIn,
    Host,
    label,
    rotation_hint,
    unmeasured,
    with_serve,
)
from qda_agent.isolation.api import Isolation
from qda_agent.isolation.api import make as make_isolation
from qda_agent.isolation.trusted import TrustChanged
from qda_agent.judge import final_rules
from qda_agent.judge.contamination import ContaminationCheck
from qda_agent.sched.snapshot import remove_tree
from qda_agent.settings import load_object

__all__ = [
    "CalibTemplateRefused",
    "ContaminationFound",
    "FINALIZE_AFTER_S",
    "StartRefused",
    "TrialRunRefused",
    "check_inputs",
    "completed",
    "EndKind",
    "finalize",
    "readable_inputs",
    "SearchParams",
    "prepare",
    "qda_search",
    "qda_search_plain",
    "search",
]


class EndKind(StrEnum):
    """How the agent phase ended; `result.json` records it, and `completed` reads it for the exit status."""

    DEADLINE = "deadline"  # the GPU deadline (or the budget that is it) ended the phase: the run is done
    STOPPED = "stopped"  # stopped from outside (Ctrl-C, SIGTERM, SIGHUP): `qda-agent run` exits 130
    ERROR = "error"  # a bug: the exception is left to the caller, so the command fails as it did before
    FATAL = "fatal"  # a harness failure no retry would change: the run failed
    REFUSED = "refused"  # the wall refused a turn: the run failed
    FINALIZE = "finalize"  # the run was finalized early (`finalize`); the command exits 3


#: Failures no retry would change: the agent phase ends and the final checkpoint is picked at once.
FATAL = (
    HarnessRefused,
    ModelUnavailable,
    HarnessNotInstalled,
    HarnessMissing,
    HarnessSandboxed,
    HarnessUnrecoverable,
    UnsupportedOperation,
)


#: How many turns in a row Humanize's watchdog may stop before the run ends as fatal. One stop is a CLI that
#: went quiet (a single long tool call outlasts the window); the same again with no turn answered between is
#: a CLI that does not come back. A turn that returns resets the count.
MAX_WATCHDOG_STOPS = 3
#: How Humanize's watchdog words the failure it ends a stopped turn with (`coganchor.agents.watchdog`).
WATCHDOG_STOP = "the watchdog stopped this turn:"
#: The least window, in seconds, Humanize's watchdog gives a CLI that says nothing during a search: the one
#: it gives every CLI but `dsh` (360 s). A dsh tool call runs in silence, and an agent waiting on its judge
#: jobs makes calls of several minutes (`sleep 420; kl list`, declared 480 s): at 360 s the watchdog took
#: that for a wedged CLI and stopped the turn.
WATCHDOG_FLOOR_S = 900.0
#: How long the agent's API may keep failing, or the disk stay below the run's free-space floor, before the
#: search is finalized: ten minutes with no recovery (`finalize`).
FINALIZE_AFTER_S = 600.0
#: How often the flow reads the judge's live disk state while the agent works, in seconds. The judge updates
#: it every scheduler pass; this is only the watch's own resolution.
FLOOR_POLL_S = 5.0
#: How often the flow probes the route to the agent's API while the agent works, in seconds (`_watch_api`).
API_POLL_S = 60.0


class StartRefused(RuntimeError):
    """An input of the search is refused before the clock; the search does not start (`qda-agent run` exits 2).

    The message is made one line here, once, however many lines the refusal it wraps has: `cli.execute` and
    `qda doctor --search` print this sentence as it is.
    """

    def __init__(self, message: object) -> None:
        super().__init__(" ".join(str(message).split()))


class ContaminationFound(StartRefused):
    """The workdir holds a test item; the search does not start."""


class CalibTemplateRefused(StartRefused):
    """The default calibration file holds rows in another model's chat template; the search does not start."""


class GencheckDomainsRefused(StartRefused):
    """The long-generation prompt set leaves out an evaluated domain; the search does not start."""


class AgentTurnRefused(StartRefused):
    """The configured agent backend cannot take the run's first turn; the search does not start."""


class TrialRunRefused(StartRefused):
    """The launch's trial run could not build, serve or read candidate 0 (`qda-agent run` exits 3).

    The machine, not the input, is what the trial run judges: the run is finalized with candidate 0
    (`finalize`) and the command exits `qda_quant.exit.ENVIRONMENT`.
    """


#: The evaluated domains TASK.md names, and the `domain` values of a prompt set that cover each one.
EVALUATED_DOMAINS: dict[str, tuple[str, ...]] = {
    "maths": ("math", "maths"),
    "code": ("code",),
    "science": ("science",),
    "instruction following": ("instruction", "instruction following"),
}


class GoalWorker(Agent, GoalCommandAgentMixin):
    """A worker whose CLI has its own `/goal`."""


class GoalAgents(AgentCollection):
    worker: GoalWorker


class PlainAgents(AgentCollection):
    worker: Agent


class Envs(EnvCollection):
    workspace: Env


class SearchParams(FlowParams):
    """Select the run configuration passed to the flow."""

    config: Path = Field(description="the run YAML (qda_agent.config.RunConfig)")


class _Log:
    """Append-only JSON lines, one file per stream, outside the sandbox."""

    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        self.path, self.clock = path, clock
        self.counts: dict[str, int] = {}
        self.last: dict[str, dict[str, Any]] = {}  # each event's latest fields
        path.write_text("")  # one run per run dir: a rerun starts its logs afresh

    def __call__(self, ev: str, **fields: Any) -> None:
        self.counts[ev] = self.counts.get(ev, 0) + 1
        self.last[ev] = fields
        with self.path.open("a") as f:
            f.write(json.dumps({"t": round(self.clock(), 3), "ev": ev, **fields}, default=str) + "\n")

    def extend(self, records: list[dict[str, Any]]) -> None:
        """Append records that carry their own time and event (made elsewhere, e.g. read from a log)."""
        for r in records:
            self.counts[r["ev"]] = self.counts.get(r["ev"], 0) + 1
        with self.path.open("a") as f:
            f.writelines(json.dumps(r, default=str) + "\n" for r in records)


def _utc(t: float) -> datetime:
    return datetime.fromtimestamp(t, UTC)


def _stamp(t: float) -> str:
    """`t` as the end report prints it: UTC to the second, `YYYY-MM-DD HH:MM:SSZ`."""
    return _utc(t).strftime("%Y-%m-%d %H:%M:%SZ")


def _recycle(session: Any) -> dict[str, str]:
    # hmz 0.2.2 only: SessionView._handle -> coganchor._id / _moved_to / _lets_go. Recycle the transport,
    # preserving the landed conversation; fork() would give it a different id.
    unavailable = {"outcome": "new session", "reason": "driver cannot recycle"}
    try:
        if version("hmz") != "0.2.2":
            return unavailable
        from hmz.coganchor.agents.claude import ClaudeCodeSession

        conversation = session._handle.coganchor
        if (
            not isinstance(conversation, ClaudeCodeSession)
            or conversation._moved_to is not None
            or not callable(conversation._lets_go)
        ):
            return unavailable
        if conversation._id is None:
            return {"outcome": "new session", "reason": "no landed turn to resume"}
        session_id = conversation.id
        conversation._lets_go()
        return {"outcome": "resumed", "session_id": session_id}
    except Exception:  # noqa: BLE001 -- a changed private driver must leave a usable new-session path
        return unavailable


def _resume_failure(error: BaseException, session_id: str) -> bool:
    # hmz 0.2.2 has no missing-session fault. Claude's --resume failure is this stderr line;
    # use the driver's process error and requested id, rather than other unrecoverable failures' prose.
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, Failed) and not current.fault and current.returncode == 1:
            command = current.cmd
            requested = any(
                flag == "--resume" and value == session_id
                for flag, value in zip(command, command[1:], strict=False)
            )
            if (
                requested
                and isinstance(current.stderr, str)
                and f"No conversation found with session ID: {session_id}" in current.stderr.splitlines()
            ):
                return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def install_toolset(cfg: RunConfig) -> None:
    """Install toolset files into the overlay, or copy them into an unrestricted workdir.

    It copies trees (the wiki, the library's source), so `search` runs it before the clock starts. The
    read-only overlay of an earlier start is replaced with `remove_tree`: its trees are sealed (0444/0555)
    and a plain `rmtree` could not enter them.
    """
    work = cfg.workdir
    work.mkdir(parents=True, exist_ok=True)
    run = cfg.run
    if run.overlay is not None and run.overlay.exists():
        remove_tree(run.overlay)
    cfg.toolset.install(run)
    if not run.walled:  # no wall binds the overlay over the workdir: the agent gets a copy to read
        for name in cfg.toolset.protected(run):
            _replace(run.ro(name), work / name, work)


def _replace(src: Path, dst: Path, work: Path) -> None:
    """Replace `dst` with a read-only copy of `src`, inside the run's own workdir only.

    A refresh after the host reorders the baselines must leave no file of the previous start behind, so the
    old tree goes before the new one is copied (through `remove_tree`, which returns the write permission a
    sealed tree no longer has). A link at `dst` is removed as a link, never through it. The copy is then made
    read-only (`workspace.read_only`): it is a host input, not disk the agent may reclaim.
    """
    if dst == work or not dst.is_relative_to(work):
        raise ValueError(f"refusing to replace {dst}: not inside the run's workdir {work}")
    remove_tree(dst)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
        workspace.read_only(dst)
    elif src.is_file():
        shutil.copyfile(src, dst)
        dst.chmod(0o444)


def prepare(
    cfg: RunConfig,
    start: datetime,
    deadline: datetime,
    installed: bool = False,
    candidate0_kl: float | None = None,
    baseline_readings: list[HandIn] | None = None,
) -> workspace.Manifest:
    """Build the agent's workdir: TASK.md, CLOCK.md, notes/, scripts/, bin/ and the components' files.

    TASK.md's tools and inputs are rendered from the workspace manifest, and the manifest is then written to
    `run_dir/manifest.json` (outside the workdir), recording every file and bind with its digest.

    Args:
        cfg: The run config.
        start: Start of the search.
        deadline: The GPU deadline.
        installed: `install_toolset` already ran (before the clock started); else it runs here.
        candidate0_kl: The host's startup reading, when automatic candidate-0 selection is enabled.
        baseline_readings: The host's readings of the baselines, when it read them at start: with
            `judge.starts` TASK.md lists each labelled start with the directory the agent sees and its KL.

    Returns:
        The manifest: every file the agent can read in its workdir (the read-only overlay's included), which
        is what the contamination check scans.
    """
    work = cfg.workdir
    for sub in ("notes", "scripts", "bin"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    start_rows = None
    if cfg.judge.starts:
        # The starts are listed in the baselines' order (KL order after the host's reorder): each labelled
        # start, the directory the agent's jobs read it at (`baselines/<N>`), and its measured KL once read.
        labels = {os.path.normpath(s.path): s.name for s in cfg.judge.starts}
        read = {os.path.normpath(r.dir): r.metrics.get("kl") for r in baseline_readings or []}
        shown = cfg.toolset.baseline_names(cfg.run) or [f"baselines/{i}" for i in range(len(cfg.baselines))]
        start_rows = [
            (labels[os.path.normpath(path)], shown[i], read.get(os.path.normpath(path)))
            for i, path in enumerate(cfg.baselines)
        ]
    from qda_agent.judge.checks.budget import lease_formats
    from qda_agent.prepare import calib_thinking_note  # the calib artifact's own metadata says what it lacks

    plan = workspace.build_manifest(cfg, cfg.toolset)
    task = prompt.task_statement(
        model=cfg.model,
        fmt=cfg.format,
        n_cards=len(cfg.cards),
        start=start,
        deadline=deadline,
        baselines=plan.inputs["baselines"] or cfg.baselines,
        calib=cfg.calib,
        calib_note=calib_thinking_note(cfg.calib),
        metric=cfg.judge.metric,
        candidate0_kl=cfg.judge.candidate0_kl if candidate0_kl is None else candidate0_kl,
        eval_max_tokens=cfg.judge.eval_max_tokens,
        kl_noise=cfg.judge.kl_noise,
        starts=start_rows,
        budget_bpw=cfg.judge.budget_bpw,
        engine=cfg.judge.engine,
        head=cfg.judge.head,
        head_lease=lease_formats(cfg.format) if cfg.judge.head_lease else (),
        budget_bytes=cfg.judge.budget_bytes,
        budget_basis=cfg.judge.budget_basis,
        engine_floor=cfg.judge.engine_floor,
        format_menu=cfg.judge.format_menu,
        require_build_receipt=cfg.judge.require_build_receipt,
        max_kl=cfg.judge.max_kl,
        max_run_bytes=cfg.judge.max_run_bytes,
        min_free_bytes=cfg.judge.min_free_bytes,
        final_rules=final_rules.statement(cfg.judge.final_rules, cfg.judge.metric),
        handin_rule=final_rules.final_handin_statement(cfg.judge.final_rules),
        gen_max_tokens=cfg.judge.gencheck_max_tokens,
        full_card_moe=cfg.judge.full_card_moe,
        commands=tuple(plan.commands),
        subcommands=plan.subcommands,
        tools=plan.tools,
        template=cfg.task,
    )
    (work / "TASK.md").write_text(task)
    (work / "CLOCK.md").write_text(prompt.clock(start, deadline, cfg.cards))
    if not installed:
        install_toolset(cfg)
    if cfg.commands:
        cfg.socket.parent.mkdir(exist_ok=True)
        load_object(cfg.commands)(work / "bin", cfg.agent_socket, names=tuple(plan.commands))
    texts = {
        "work/TASK.md": str(work / "TASK.md"),
        "work/CLOCK.md": str(work / "CLOCK.md"),
    }
    if cfg.run.overlay is not None and (cfg.run.overlay / "DATA.md").is_file():
        texts["overlay/DATA.md"] = str(cfg.run.overlay / "DATA.md")
    manifest = workspace.build_manifest(cfg, cfg.toolset, texts=texts, snapshot=True)
    manifest.write(cfg.run_dir / workspace.MANIFEST)
    cfg._manifest = manifest  # the wall and the scheduler binds read the same record
    return manifest


#: Files skipped by the pre-start scan: weights and other binaries, which hold no text.
BINARY = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".npy",
    ".npz",
    ".so",
    ".pyc",
    ".whl",
    ".zip",
    ".gz",
    ".tar",
)


def readable_inputs(cfg: RunConfig) -> list[Path]:
    """The files beyond the workdir the agent can read (its jobs, read-only).

    The calibration file the task names, and every text file under the paths `RunConfig.read_paths` binds
    (the baselines included, wherever they are bound);
    an entry that is itself a symlink is scanned as its target, which is what the job's bind shows.
    The installed trees `RunConfig.contamination_skips` names (the run's real site-packages directories and
    qda_kl's vendored IFBench checkers) are left out by exact path; the run's configured data -- the
    calibration file and the prompt set -- is scanned even when it sits in one of those trees, and so is the
    workdir (`prepare`).
    """
    readable, binds = cfg.read_paths()
    data = [Path(p) for p in (cfg.calib, cfg.judge.prompts_path()) if p]
    roots = [*readable, *binds.values(), *data]
    skip = {p.resolve() for p in cfg.contamination_skips()}
    keep = {p.resolve() for p in data}
    out: list[Path] = []
    for r in roots:
        r = r.resolve()  # an entry that is a link is mounted as its target: scan what the job sees
        found = [r] if r.is_file() else _walk_files(r, skip, keep) if r.is_dir() else []
        for f in found:
            p = f.resolve()
            if f.is_symlink() or f.suffix in BINARY:
                continue
            if any(p.is_relative_to(s) for s in skip) and not any(p.is_relative_to(k) for k in keep):
                continue
            out.append(f)
    return list(dict.fromkeys(out))


def _walk_files(top: Path, skip: set[Path], keep: set[Path]) -> list[Path]:
    """The files under the real directory `top`, sorted, links to directories not followed.

    A directory inside a skipped tree is not entered unless a kept path lies in it: a venv's site-packages
    is tens of thousands of files, which would take minutes to list on a network file system only to be
    dropped afterwards.
    """
    found: list[Path] = []
    for root, dirs, files in os.walk(top, followlinks=False):
        here = Path(root)
        dirs[:] = [
            d
            for d in dirs
            if not any((here / d).is_relative_to(s) for s in skip)
            or any(k.is_relative_to(here / d) for k in keep)
        ]
        found += [here / name for name in files if (here / name).is_file()]
    return sorted(found)


async def search(
    worker: Agent,
    workspace: Env,
    cfg: RunConfig,
    *,
    goal: bool,
    ctx: FlowContext | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Prepare and scan a search, drive it to the deadline, then select its submission.

    Args:
        worker: The worker role.
        workspace: The agent's workdir as a Humanize environment.
        cfg: The run config.
        goal: Send the instruction as `/goal` (the CLI's own goal feature) or as a plain prompt.
        ctx: The flow context, for the spending in the end report.
        clock: Wall clock (epoch seconds).

    Returns:
        A summary: the final pick, goals sent, failures, paths.

    Raises:
        ContaminationFound: The contamination check found test items; nothing was started.
    """
    isolation.check(cfg)
    with _watchdog_window(cfg), _endpoint_env(cfg) as dropped:
        if not cfg.needs_private_socket:
            return await _search(worker, workspace, cfg, goal=goal, ctx=ctx, clock=clock, dropped=dropped)
        with rpc.private_dir(cfg.run_dir):  # removed once the host has stopped, however the run ends
            return await _search(worker, workspace, cfg, goal=goal, ctx=ctx, clock=clock, dropped=dropped)


def check_calib_template(cfg: RunConfig) -> None:
    """Refuse a default calibration file (`RunConfig.calib`) whose text rows use another model's chat template.

    The same check a build runs on its text rows (`qda_quant.calib.template`), run once before the clock starts:
    TASK.md hands this file to the agent as its calibration data, so a file in the wrong template would reach
    every build that mixes it in. The model's tokenizer is loaded only when a row holds a chat marker, from the
    model resolved as a build resolves it (`qda_quant.engine.planning.local_source`, metadata files only).

    Raises:
        CalibTemplateRefused: a row is in another model's chat template, the file is not calibration JSONL, or
            the model's tokenizer cannot be found.
    """
    if not cfg.calib:
        return
    from qda_quant.calib.sources import read_file
    from qda_quant.calib.template import check_rows, has_markers
    from qda_quant.step import Refused

    try:
        rows = read_file(cfg.calib).rows
    except (Refused, ValueError) as e:
        raise CalibTemplateRefused(f"calibration {cfg.calib}: not a calibration JSONL file: {e}") from e
    tokenizer = None
    if has_markers(rows):
        from qda_quant.engine.planning import local_source
        from qda_quant.intake.tokenizer import load_tokenizer

        # the model as a build resolves it: a directory, or the hub snapshot (`org/name@rev` keeps its revision)
        # of its configs and tokenizer files only
        try:
            source = local_source(cfg.model, weights=False)
        except Refused as e:
            raise CalibTemplateRefused(
                f"calibration {cfg.calib}: cannot read the model's chat template: {e}"
            ) from e
        tokenizer = load_tokenizer(source)
    if (why := check_rows(rows, tokenizer).refusal(cfg.calib)) is not None:
        raise CalibTemplateRefused(why)


def check_budget(cfg: RunConfig) -> None:
    """Refuse a candidate 0 outside `judge.budget_bpw`, with the check the judge's start makes.

    The host prices `baselines[0]` with `bpw_problem` against the anchor's `denominator` (and, with
    `judge.head_lease`, its head). Run from here, the same functions refuse before the clock, so a budget below
    candidate 0's own bpw cannot pass `prepare` and `qda doctor --search` only to fail the run's start.

    Raises:
        StartRefused: candidate 0 is over the budget, or the anchor cannot be priced.
    """
    judge = cfg.judge
    if judge.budget_bpw is None or judge.engine != "vllm" or judge.anchor is None or not cfg.baselines:
        return
    from qda_agent.judge.checks.budget import bpw_problem, denominator, head_weights
    from qda_agent.judge.checks.tensors import CheckpointError

    try:
        weights = denominator(judge.anchor)
        lease = head_weights(judge.anchor) if judge.head_lease else None
    except CheckpointError as e:
        raise StartRefused(f"judge.anchor {judge.anchor}: {e}") from None
    if why := bpw_problem(cfg.baselines[0], weights, judge.budget_bpw, lease=lease):
        raise StartRefused(f"baselines[0] {cfg.baselines[0]}: {why}")


def check_inputs(cfg: RunConfig) -> None:
    """The data checks a search makes on its inputs before the clock; `qda doctor --search` runs the same.

    The default calibration file, the long-generation prompt set, the agent's data sets, and candidate 0's
    own bits per quantized weight against `judge.budget_bpw`.

    Raises:
        StartRefused: the first input refused, with what to fix.
    """
    check_calib_template(cfg)
    check_gencheck_domains(cfg)
    cfg.toolset.check_shipped_data(cfg.model)
    check_budget(cfg)


def check_agent_turn(cfg: RunConfig) -> str | None:
    """The check a search makes on its agent before the clock; `qda doctor --search` runs the same.

    The backend's own answer (`agents.Backend.turn_problem`): the SDK Humanize would drive it through is
    installed here, and the endpoint this run's key goes to takes the model it names. A backend that cannot
    be asked without spending a turn of it passes, and the run's own first turn is the check for those.

    A problem a retry or a fallback can take the turn after is *reported*, not refused: Humanize retries the
    same place and then walks the configured fallbacks, which is what an endpoint that did not answer, or one
    that is throttled, needs (`TurnProblem.transient`), and a fallback covers the endpoint that answered with
    a refusal besides. Without either there is no place left for the run to take a turn, which is what makes
    the refusal the run's own.

    Returns:
        What to record about the first turn, or None when there was nothing to say about it.

    Raises:
        AgentTurnRefused: neither the configured backend nor a fallback can take the first turn.
    """
    found = cfg.agent.backend.turn_problem()
    if found is None:
        return None
    if found.transient or cfg.agent.fallbacks:
        takes = (
            "a retry takes the turn" if found.transient else f"{cfg.agent.fallbacks[0].cli} takes the turn"
        )
        return f"{cfg.agent.backend.cli}: {found.said} ({takes})"
    raise AgentTurnRefused(f"{cfg.agent.backend.cli}: {found.said}")


def check_gencheck_domains(cfg: RunConfig) -> None:
    """Refuse a run whose long-generation prompt set covers fewer than the four evaluated domains.

    TASK.md tells the agent it is evaluated on maths, code, science and instruction following, and the
    long-generation check is the search's only in-search reading of them: a set that leaves one out would
    report a clean check for a domain nothing read. Each row names its domain in its `domain` field, and the
    set's names map to the four in `EVALUATED_DOMAINS`. The shipped set (`gencheck_prompts` unset) has all
    four; a set this check cannot read is refused, never passed.

    A run whose toolset has no `gencheck` command never reads the set, so it is not refused.

    Raises:
        GencheckDomainsRefused: an evaluated domain has no prompt in the set, or the set cannot be read.
    """
    if "gencheck" not in cfg.toolset.commands(cfg.run):
        return
    path = _gencheck_prompt_path(cfg)
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as e:
        raise GencheckDomainsRefused(
            f"judge.gencheck_prompts {path}: cannot read the prompt set as JSONL: {e}"
        ) from e
    domains = {str(row.get("domain", "")) for row in rows if isinstance(row, dict)}
    missing = [name for name, names in EVALUATED_DOMAINS.items() if not domains.intersection(names)]
    if not missing:
        return
    fields = ", ".join(f"`{n}`" for name in missing for n in EVALUATED_DOMAINS[name])
    raise GencheckDomainsRefused(
        f"judge.gencheck_prompts {path}: the set has no prompt for {', '.join(missing)}. The search is "
        "evaluated on maths, code, science and instruction following, and the long-generation check is the "
        f"only reading of them inside the search; add rows whose `domain` field is {fields}."
    )


def _gencheck_prompt_path(cfg: RunConfig) -> Path:
    """The local file of the run's long-generation prompt set: the shipped one when none is configured.

    Raises:
        GencheckDomainsRefused: the set cannot be made available (a dataset that refuses its settings).
    """
    from qda_kl.datasets import PROMPTS, parse_dataset
    from qda_quant.step import Refused

    try:
        dataset = cfg.judge.prompt_set()
        if dataset is None:
            dataset = parse_dataset({"gencheck_builtin": {}}, "judge.gencheck_prompts", PROMPTS)
        return dataset.local()
    except Refused as e:
        raise GencheckDomainsRefused(f"judge.gencheck_prompts: cannot read the prompt set: {e}") from e


def announce_disk_cap(cfg: RunConfig) -> None:
    """Print the run's resolved disk cap and the rule behind it, before the start phase."""
    if line := cfg.judge.disk_cap_line():
        print(line, file=sys.stderr)


def log_scan(
    cfg: RunConfig,
    log: _Log,
    manifest: workspace.Manifest,
    *,
    inputs: int,
    findings: list[str],
    after: str = "",
) -> None:
    """Log one contamination scan and the manifest it read (`flow.jsonl`, event `contamination`).

    The record names the manifest file and its digest, so the log says exactly which workspace the check
    passed; `inputs` counts the files outside the workdir this scan also read.
    """
    path = cfg.run_dir / workspace.MANIFEST
    fields: dict[str, Any] = {
        "check": cfg.contamination.name,
        "files": inputs + len(manifest.files),
        "findings": findings,
        "skipped": [str(p) for p in cfg.contamination_skips()],
        "manifest": str(path),
        "manifest_sha256": workspace.file_sha256(path),
    }
    if after:
        fields["after"] = after
    log("contamination", **fields)


async def _open(
    cfg: RunConfig, log: _Log, clock: Callable[[], float]
) -> tuple[Host, Isolation, float, float]:
    """Everything a search does before the agent's first turn, on the search's own clock.

    Scans the inputs, starts the clock, writes the workdir, starts the judge host (its build-receipt and
    reading smokes, and candidate 0's reading) and returns it started, with the clock's start and the deadline.
    The time from the clock's start to the return is the search's start overhead W (`gate`).

    Raises:
        ContaminationFound: The contamination check found test items; nothing was started.
        CalibTemplateRefused: The default calibration file holds rows in another model's chat template;
            nothing was started.
        StartRefused: The host refused a condition of this run before it wrote anything; nothing was
            started.
        TrialRunRefused: The launch's trial run failed; the caller finalizes the run with candidate 0.
        AgentTurnRefused: neither the configured agent backend nor a fallback can take the run's first turn;
            nothing was started.
    """
    # Before the clock: a statement whose values the config lacks refuses here, not mid-run.
    cfg.require_task_values()
    # Before the smokes and the agent's first turn: the operator sees the cap the run will hold.
    announce_disk_cap(cfg)
    # Before the scan and the smokes: an agent that cannot take a turn at all refuses now rather than after
    # them, and one a retry or a fallback takes the turn after is recorded and the run goes on.
    if (why := check_agent_turn(cfg)) is not None:
        log("agent_turn", why=why)
    scan = cast(ContaminationCheck, cfg.contamination).scanner()  # the run's check, built once
    # The read-only inputs are scanned before the clock starts, so a large input tree never comes out of the
    # agent's GPU budget. The workdir is scanned after `prepare`, since TASK.md names the deadline.
    inputs = readable_inputs(cfg)
    findings = scan(inputs)
    if not findings:  # a test item refuses first, below; these checks read their files as data rows
        check_inputs(cfg)
    install_toolset(cfg)  # copies trees: not out of the agent's GPU budget
    t0 = clock()
    deadline = t0 + cfg.wall_s
    # The wall binds the refreshed sign-in file each turn (a named dir); a default dir has no refresh loop.
    isolation.check_sign_in(cfg, _utc(deadline))
    manifest = prepare(cfg, _utc(t0), _utc(deadline), installed=True)
    findings += scan(manifest.paths())
    log_scan(cfg, log, manifest, inputs=len(inputs), findings=findings)
    if findings:
        raise ContaminationFound(f"{len(findings)} finding(s) in {cfg.workdir}: {findings[:5]}")

    # One isolation for the run, made before the clock's first job: it is what the queue, the judge and the
    # agent's turns go through, and it refuses a machine that cannot hold the configured level.
    iso = make_isolation(cfg, on_skip=lambda path, why: log("mount_skipped", path=path, why=why))
    host = await _start_host(cfg, iso, deadline)
    try:
        if cfg.judge.select_candidate0 or cfg.judge.starts:
            readings = host.final().baseline_readings
            if readings:  # a host that read no baseline (a CPU stand-in) leaves the prepared TASK.md
                cfg.baselines[:] = [reading.dir for reading in readings]
                install_toolset(cfg)
                # Candidate 0's reading rewrites TASK.md: the new manifest is scanned again, so a text the
                # check has not seen cannot reach the agent.
                manifest = prepare(
                    cfg,
                    _utc(t0),
                    _utc(deadline),
                    installed=True,
                    candidate0_kl=readings[0].metrics["kl"],
                    baseline_readings=readings,
                )
                after = scan(manifest.paths())
                log_scan(cfg, log, manifest, inputs=0, findings=after, after="candidate0")
                if after:
                    raise ContaminationFound(
                        f"{len(after)} finding(s) after candidate 0 rewrote TASK.md: {after[:5]}"
                    )
    except BaseException:
        await asyncio.to_thread(host.stop)
        raise
    return host, iso, t0, deadline


async def _start_host(cfg: RunConfig, iso: Isolation, deadline: float) -> Host:
    """The run's host, built and started; a condition either call refuses is `StartRefused`.

    Building it is where the host reads the conditions it refuses before it writes anything
    (`JudgeHost.__init__` -> `judge.startup.static_checks`), and `start` asks its own (`qda`, the smokes).
    Only those two calls are turned into a refusal: the candidate 0 block `_open` runs next also reads and
    builds, and a `ValueError` from there is a failure, not a start condition.

    Raises:
        TrialRunRefused: The launch's trial run could not build, serve or read candidate 0 on this machine.
        StartRefused: The host refused a condition of this run before its trial run.
    """
    try:
        host: Host = load_object(cfg.host)(
            cfg.run_dir, cfg.sched_config(_utc(deadline)), cfg.judge_config(), isolation=iso, run_config=cfg
        )
    except ValueError as why:
        raise StartRefused(why) from why
    from qda_agent.judge.startup import TrialRunFailed  # the host's module, imported with the host

    try:
        await asyncio.to_thread(host.start)
    except TrialRunFailed as why:  # the machine, not the input: the run is finalized with candidate 0
        await asyncio.to_thread(host.stop)
        raise TrialRunRefused(why) from why
    except ValueError as why:
        await asyncio.to_thread(host.stop)
        raise StartRefused(why) from why
    except BaseException:
        await asyncio.to_thread(host.stop)
        raise
    return host


async def gate(cfg: RunConfig, *, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Run a search's start phase alone and measure its W; no agent is started (`qda-agent gate`).

    The start phase is `_open`, the same code a search runs: W is what a search on this config, tree and
    node spends of its clock before the agent's first turn. The host is stopped at once; `gate.json` in the
    run dir records W, candidate 0's readings, and how the jobs ran: `isolation.level` and the job sandbox it
    resolves to (`RunConfig.job_sandbox`, the one `resolved.yaml` records for a search).
    """
    isolation.check(cfg)
    with rpc.private_dir(cfg.run_dir) if cfg.needs_private_socket else contextlib.nullcontext():
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        log = _Log(cfg.run_dir / "flow.jsonl", clock)
        host, _iso, t0, _ = await _open(cfg, log, clock)
        w = clock() - t0
        try:
            readings = [
                {"dir": str(r.dir), "metrics": dict(r.metrics)} for r in host.final().baseline_readings
            ]
        finally:
            await asyncio.to_thread(host.stop)
    out = {
        "w_s": w,
        "start": _utc(t0).isoformat(),
        "baseline_readings": readings,
        "isolation_level": cfg.isolation.level,
        "job_sandbox": cfg.job_sandbox(),
    }
    (cfg.run_dir / "gate.json").write_text(json.dumps(out, indent=2, default=str))
    log("gate", w_s=w)
    return out


async def _search(
    worker: Agent,
    workspace: Env,
    cfg: RunConfig,
    *,
    goal: bool,
    ctx: FlowContext | None,
    clock: Callable[[], float],
    dropped: Sequence[str] = (),
) -> dict[str, Any]:
    """`search` once the socket's directory is settled; `dropped` names the variables `_endpoint_env` removed."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log, tools = _Log(cfg.run_dir / "flow.jsonl", clock), _Log(cfg.run_dir / "tools.jsonl", clock)
    if dropped:  # names only: a value may be a credential's half
        log(
            "env_dropped",
            names=list(dropped),
            why="a CLI signed in by a file or an account keeps its own endpoint",
        )
    try:
        host, iso, t0, deadline = await _open(cfg, log, clock)
    except TrialRunRefused as why:
        # The trial run did on this machine what the search would do, and it failed: the run ends with what
        # it has (candidate 0) and a FINAL.json saying why, instead of leaving the operator a bare refusal.
        await finalize(cfg, None, reason=str(why), log=log, never_ran="the launch's trial run failed")
        raise
    # The operator's acceptance and what was on the node when the run started (None without an acceptance).
    shared_node = isolation.shared_node_record(cfg)
    log(
        "start",
        deadline=_utc(deadline).isoformat(),
        spec=cfg.agent.spec,
        sandbox=cfg.sandbox.name,
        isolation=iso.level,
    )

    async def supervise(p: PreToolUseHookParams) -> PreToolUseHookResult:
        tools("tool", tool=p.tool, input=dict(p.input))
        return PreToolUseHookResult()

    worker = worker.derive(permission=iso.agent_permission())
    worker.on_pre_tool_use(supervise)
    instruction = prompt.goal(cfg.format, _utc(deadline), cfg.toolset.commands(cfg.run))
    text = f"/goal {instruction}" if goal else instruction
    kind, ended = EndKind.DEADLINE, str(EndKind.DEADLINE)
    sign_in_restarts: list[dict[str, str]] = []
    resume_first = False
    # The run ends early (`finalize`) when the API has failed for `FINALIZE_AFTER_S`, its route has failed every
    # probe that long, or the disk has stayed below its floor that long; `finalize_watch` is the flow's only view
    # of the route and the disk while a turn is in flight.
    finalize_reason: str | None = None
    api_failed_since: float | None = None
    finalize_watch = asyncio.ensure_future(_first_reason(_watch_floor(host, clock), _watch_api(cfg, clock)))
    # The seconds of turns the flow saw running and not failing: ones that returned, and the one the deadline,
    # the budget or a stop cut while it ran (`flying`: when the turn in flight started, None between turns).
    seen_s: float = 0.0
    flying: float | None = None
    try:
        session = await worker.spawn()
        sent = -math.inf
        held: dict[str, tuple[int, int] | None] | None = None  # the sign-in the live process holds
        retried: dict[str, tuple[int, int] | None] | None = None  # the file a retry was already spent on
        fresh = False  # whether the current process was recycled for a change no refusal has answered yet
        stops = 0  # turns Humanize's watchdog stopped since the last turn that returned
        # A backend whose session takes one instruction (or a fallback of that kind, which may hold the turn).
        one_turn = not all(b.resumes for b in (cfg.agent.backend, *cfg.agent.fallbacks))
        used = False  # whether `session` has been sent an instruction
        while True:
            # One fixed tick between any two sends (idle returns and failed turns alike); none past the deadline.
            wait = sent + cfg.agent.resend_wait_s - clock()
            if wait > 0:
                await asyncio.sleep(min(wait, max(0.0, deadline - clock())))
            if (left := deadline - clock()) <= 0:
                break
            if STOPPING.is_set():  # a stop signal: no new turn, whatever the last one's end was
                kind, ended = EndKind.STOPPED, str(EndKind.STOPPED)
                break
            if finalize_watch.done():  # the disk stayed below its floor, or the API out of reach: no new turn
                finalize_reason = finalize_watch.result()
                kind, ended = EndKind.FINALIZE, f"{EndKind.FINALIZE}: {finalize_reason}"
                break
            sent = clock()
            now = isolation.sign_in_witness(cfg)  # the sign-in file the host has now (stat only)
            if (mounted := isolation.sign_in_mounted(cfg)) is not None:
                held = mounted  # what the running wall records it bound, fresher than any goal
            if held is None:
                held = now
            elif held != now:
                restart = (
                    {"outcome": "new session", "reason": "the agent's session takes one instruction"}
                    if one_turn
                    else await asyncio.to_thread(_recycle, session)
                )
                if deadline - clock() <= 0:
                    log("restart_cancelled", why="deadline reached while recycling the sign-in process")
                    break
                sign_in_restarts.append(restart)
                resume_first = restart["outcome"] == "resumed"
                log("restart", why="sign-in refreshed since the wall was built", **restart)
                if not resume_first:
                    session = await worker.spawn()
                held, fresh, used = now, now != retried, False
            if used and one_turn:
                # The session's one turn is spent (an idle return or a failed turn alike): a second
                # instruction to it is refused as unrecoverable, which would end the agent phase at the first
                # idle return. As above, the search does not need the conversation: the goal is re-sent to a
                # new session in the same workspace, under the flow's deadline and remaining budget.
                log("restart", why="the agent's session takes one instruction; a new session for this goal")
                session = await worker.spawn()
            if (left := deadline - clock()) <= 0:
                break  # recycling the process can itself use up the remaining time
            used = True
            log("goal", left_s=round(left, 1))
            flying = clock()
            try:
                # The hard turn budget is Humanize's cut-off; the timeout holds it on every driver
                # (cancelling a turn interrupts the CLI).
                async with asyncio.timeout(left):
                    budget = Budget(duration=timedelta(seconds=left), graceful=False)
                    first, resume_first = resume_first, False
                    try:
                        said = await _unless_stopped(
                            worker.run(text, session=session, env=workspace, budget=budget), finalize_watch
                        )
                    except (HarnessUnrecoverable, SessionError) as e:
                        # Only a missing/rejected conversation on the first resumed turn permits a new
                        # conversation. Sign-in refusals and the wall keep their existing retry rules.
                        if (
                            not first
                            or not _resume_failure(e, sign_in_restarts[-1]["session_id"])
                            or _wall_refusal(e) is not None
                        ):
                            raise
                        restart = sign_in_restarts[-1]
                        restart.clear()
                        restart.update(outcome="new session", reason=f"resume failed ({type(e).__name__})")
                        log("restart", why="sign-in resume failed", **restart)
                        session = await worker.spawn()
                        log("goal", left_s=round(deadline - clock(), 1))
                        said = await _unless_stopped(
                            worker.run(text, session=session, env=workspace, budget=budget), finalize_watch
                        )
            except _Finalized:
                finalize_reason = finalize_watch.result()
                kind, ended = EndKind.FINALIZE, f"{EndKind.FINALIZE}: {finalize_reason}"
                break
            except _Stopped:  # the turn in flight was ended by a stop signal
                kind, ended = EndKind.STOPPED, str(EndKind.STOPPED)
                break
            except (TimeoutError, BudgetExceeded):  # the GPU deadline, or the budget that is it, is spent
                break
            except HarnessRefused as e:
                flying = None  # a failed turn: not the agent at work
                # The provider refused the credential this process holds. If the host refreshed the sign-in
                # file since (mid-turn, or before a recycle the new wall has not bound yet), send again: the
                # next go-round recycles the warm process and its wall binds the new file. Once per change:
                # a refusal with no change stays fatal.
                current = isolation.sign_in_witness(cfg)
                if (current != held and current != retried) or fresh:
                    log(
                        "failed",
                        error=type(e).__name__,
                        message=str(e),
                        why="sign-in refreshed; retrying with the new file",
                    )
                    retried, fresh = current, False
                    continue
                log("fatal", error=type(e).__name__, message=str(e))
                kind, ended = EndKind.FATAL, f"{EndKind.FATAL}: {type(e).__name__}"
                break
            except FATAL as e:
                flying = None
                # Humanize's watchdog fails a turn whose CLI went silent past its window, and Humanize names
                # that failure unrecoverable; the watchdog means it as a turn worth another go. Send the goal
                # again, unless the watchdog has stopped every turn since the last one that returned.
                if _watchdog_stop(e) is not None and (stops := stops + 1) < MAX_WATCHDOG_STOPS:
                    log(
                        "failed",
                        error=type(e).__name__,
                        message=str(e),
                        why=f"the watchdog stopped the turn ({stops} of {MAX_WATCHDOG_STOPS} in a row); resending",
                    )
                    continue
                log("fatal", error=type(e).__name__, message=str(e), cause=_fatal_cause(e, goal=goal))
                kind, ended = EndKind.FATAL, f"{EndKind.FATAL}: {type(e).__name__}"
                break
            except HarnessError as e:  # Humanize's retries and fallbacks are spent: send the goal again
                flying = None
                if (wall := _wall_refusal(e)) is not None:  # unless the wall refused the turn: no retry helps
                    log("fatal", error=type(wall).__name__, message=str(wall))
                    kind, ended = EndKind.REFUSED, f"{EndKind.REFUSED}: {wall}"
                    break
                log("failed", error=type(e).__name__, message=str(e))
                # An API that keeps failing is the machine's, not the agent's: after `FINALIZE_AFTER_S` with
                # no turn that returned, end the run with what it has instead of spending its clock on retries.
                now_failed = clock()
                api_failed_since = now_failed if api_failed_since is None else api_failed_since
                if now_failed - api_failed_since >= FINALIZE_AFTER_S:
                    # The class and fixed words only: the CLI's own words can echo a key, so they stay in
                    # flow.jsonl (the `failed` line above), as a fatal end's do.
                    finalize_reason = (
                        f"the agent API failed for {FINALIZE_AFTER_S / 60:.0f} minutes without a turn that "
                        f"returned (last: {type(e).__name__}; details in {log.path})"
                    )
                    log("finalize_trigger", why=finalize_reason)
                    kind, ended = EndKind.FINALIZE, f"{EndKind.FINALIZE}: {finalize_reason}"
                    break
                continue
            if flying is not None:  # a turn that returned
                seen_s += clock() - flying
            flying = None
            fresh = False  # the recycled process took its turn: its wall bound a credential that works
            stops = 0  # a turn returned: the CLI is back
            api_failed_since = None  # consecutive API failures are counted from the last turn that returned
            log("idle", said=str(said)[-500:])
    except asyncio.CancelledError:  # stopped from outside (Ctrl-C, run budget): still pick a checkpoint
        kind, ended = EndKind.STOPPED, str(EndKind.STOPPED)
        raise
    except BaseException as e:  # a bug: still pick a checkpoint
        kind, ended = EndKind.ERROR, f"{EndKind.ERROR}: {type(e).__name__}"
        raise
    finally:
        CLOSING.set()
        finalize_watch.cancel()
        searched_s = clock() - t0  # the search's own wall time: the drain after it is the judge's
        log("cutoff", ended=ended, late_s=round(clock() - deadline, 1))
        if flying is not None:  # the turn in flight when the deadline, the budget or a stop ended it
            seen_s += clock() - flying
        # Before the decision: Codex's calls are read from its own log, not the hook.
        agents.log_codex_calls(cfg, tools, log)
        jobs, written = _agent_jobs(host), int(getattr(getattr(ctx, "usage", None), "output_tokens", 0) or 0)
        ran, warning = _agent_ran(log, tools, jobs=jobs, seen_s=seen_s, written=written)
        never_ran = None if ran else _why_not_run(log, ended)
        if finalize_reason is not None:
            # The run cannot go on: the same close as the deadline's, with the reason and no further turns.
            final = await finalize(cfg, host, reason=finalize_reason, log=log, never_ran=never_ran)
            stopped = False
        else:
            final, stopped = await _finish(
                host,
                cfg,
                log,
                sign_in_restarts=sign_in_restarts,
                never_ran=never_ran,
                warning=warning,
            )
        result = {
            "ended": ended,
            "end_kind": kind,
            "rule": final.rule,
            "choice": str(final.choice.dir) if final.choice else None,
            "baseline": str(final.baseline) if final.baseline else None,
            "agent_ran": ran,
            "goals": log.counts.get("goal", 0),
            "failed": log.counts.get("failed", 0),
            "host_errors": log.counts.get("host_error", 0),
            "tools": tools.counts.get("tool", 0),
            "time_ratio": cfg.time_ratio(searched_s),
            "final": str(cfg.run_dir / "FINAL.json"),
        }
        if kind == EndKind.FATAL:  # the class only: the CLI's words can echo a key (flow.jsonl has them)
            result["fatal"] = log.last["fatal"]["error"]
            if cause := log.last["fatal"].get("cause"):
                result["fatal_cause"] = cause
        if final.export:
            result["export"] = final.export
        if final.serve:  # how the user serves the checkpoint this run picked
            result["serve"] = final.serve
        if final.generation_check:
            result["generation_check"] = final.generation_check
        if shared_node is not None:  # a result reader sees the acceptance and the node it was accepted on
            result["shared_node"] = shared_node
        if finalize_reason is not None:  # why the run ended before its deadline
            result["finalize_reason"] = finalize_reason
        # Written here, not by the caller: a stop re-raises, and its run still gets every file.
        if sign_in_restarts:
            result["sign_in_restarts"] = sign_in_restarts
        try:
            (cfg.run_dir / "result.json").write_text(json.dumps(result, indent=2))
            _end_report(
                cfg,
                final,
                log,
                tools,
                kind,
                ended,
                ctx,
                t0,
                deadline,
                searched_s,
                shared_node=shared_node,
                sign_in_restarts=sign_in_restarts,
            )
        except OSError as e:
            log("report_error", error=type(e).__name__, message=str(e))
        if stopped:  # a stop during the close was held off until the files were written: honour it now
            raise asyncio.CancelledError
    return result


class _Stopped(Exception):
    """A stop signal ended the agent's turn in flight."""


class _Finalized(Exception):
    """The disk stayed below its floor long enough that the run finalized instead of starting another turn."""


#: How often a turn in flight looks for a stop signal, in seconds.
STOP_POLL_S = 0.5


def _host_now(host: Host, clock: Callable[[], float]) -> float:
    """The host's own clock when it has one (its scheduler pass stamped `below_floor_since` with it)."""
    host_clock = getattr(host, "clock", None)
    return host_clock.now() if host_clock is not None else clock()


async def _watch_floor(host: Host, clock: Callable[[], float]) -> str:
    """Wait until the disk has stayed below the run's free-space floor for `FINALIZE_AFTER_S`.

    The judge marks when it first went below (`JudgeHost.below_floor_since`) and clears the mark on the
    first pass above it again: a continuous stretch below the floor is one the run's own prune pass could
    not end. Returns:
        Why the run is finalized, for `finalize`.
    """
    while True:
        await asyncio.sleep(FLOOR_POLL_S)
        since = getattr(host, "below_floor_since", None)
        if since is not None and _host_now(host, clock) - since >= FINALIZE_AFTER_S:
            return (
                f"the disk stayed below the run's free-space floor for "
                f"{FINALIZE_AFTER_S / 60:.0f} minutes with nothing to free"
            )


async def _watch_api(cfg: RunConfig, clock: Callable[[], float]) -> str:
    """Wait until the route to the agent's API has failed every probe for `FINALIZE_AFTER_S`.

    A CLI that loses its API mid-turn need not fail the turn or fall silent: Codex reports `Reconnecting...
    waiting for network` for as long as the network is gone, which keeps Humanize's watchdog fed and the turn
    in flight. So the route the pre-start check asks about (`api_route`) is probed once a minute; a probe that
    passes starts the ten minutes again. Returns:
        Why the run is finalized, for `finalize`.
    """
    from qda_agent.api_route import api_endpoint, api_route_problem, walled

    found = api_endpoint(cfg)
    if found is None:  # no CLI with a known API host: nothing to probe
        await asyncio.Future()
    assert found is not None
    host, port = found
    down_since: float | None = None
    while True:
        await asyncio.sleep(API_POLL_S)
        why = await asyncio.to_thread(api_route_problem, host, port, direct=walled(cfg))
        if why is None:
            down_since = None
            continue
        now = clock()
        down_since = now if down_since is None else down_since
        if now - down_since >= FINALIZE_AFTER_S:
            return f"the agent API could not be reached for {FINALIZE_AFTER_S / 60:.0f} minutes ({why})"


async def _first_reason(*watches: Coroutine[Any, Any, str]) -> str:
    """The reason of the first watch that decides the run cannot go on; the others are cancelled."""
    tasks = [asyncio.ensure_future(w) for w in watches]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return next(iter(done)).result()
    finally:
        for task in tasks:
            task.cancel()


async def _unless_stopped(turn: Any, finalize_watch: asyncio.Future[str] | None = None) -> Any:
    """Await one agent turn, cancelling it (which interrupts its CLI) as soon as a stop signal comes (`STOPPING`).

    A signal's own exception may never reach this task (a driver that catches the interrupt and reports a failed
    turn, which the loop would answer with a new one), so the flag is polled. `finalize_watch` is polled the same
    way: the turn ends as soon as it decides the run cannot go on.

    Raises:
        _Stopped: The turn was ended by a stop signal.
        _Finalized: The turn was ended by `finalize_watch`.
    """
    task = asyncio.ensure_future(turn)
    try:
        while not task.done():
            if STOPPING.is_set():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                raise _Stopped
            if finalize_watch is not None and finalize_watch.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                raise _Finalized
            await asyncio.wait({task}, timeout=STOP_POLL_S)
        return task.result()
    finally:
        if not task.done():  # this await was cancelled (the deadline, or a stop from outside)
            task.cancel()


def completed(result: Mapping[str, Any]) -> bool:
    """Whether a run did what it was asked (it reached its GPU deadline): `qda-agent run` exits 0 only then."""
    return result.get("end_kind") == EndKind.DEADLINE


@contextlib.contextmanager
def _endpoint_env(cfg: RunConfig) -> Iterator[list[str]]:
    """Remove, while the search runs, each run CLI's inherited endpoint variables it must not follow.

    A claude or codex CLI (`isolation.CLI_CONFIG`; every other CLI is left alone) that signs in by its sign-in file
    or by an account sends that sign-in where `endpoint_vars` point; the
    run configures no endpoint, so any such variable is the launching shell's (a relay its own session uses).
    One that signs in by a token variable (`isolation.CLI_CONFIG`) keeps them: the token and its endpoint are
    one pair. The flow's environment is what the wall passes on; the variables are restored when the search
    ends, however it ends.

    Yields:
        The names removed (never their values).

    Raises:
        SignInRefused: One CLI signs in by an account in one of the run's agents and by a token variable in
            another while one of its endpoint variables is set: one environment cannot keep that endpoint for
            one and drop it for the other. With none set there is nothing to choose, and the run starts.
    """
    # Only the CLIs whose sign-in qda knows (`isolation.CLI_CONFIG`: its file, or its token variables): any other
    # CLI may take its endpoint from its account (a `dsh@<account>` relay), so its environment is left alone.
    specs = [
        (name, account) for name, account in isolation.cli_dirs.clis(cfg) if name in isolation.CLI_CONFIG
    ]

    def by_token(name: str) -> bool:
        return any(os.environ.get(t) for t in isolation.token_vars(name))

    # Decided before anything is removed: a refusal leaves the environment exactly as it found it.
    for name, account in specs:
        endpoint = [v for v in agents.endpoint_vars(name) if v in os.environ]
        if account and by_token(name) and (name, False) in specs and endpoint:
            # An alias spelling of the CLI (`claude-code`) counts as the CLI.
            mine = [spec for spec in cfg.agent.specs if agents.canonical_cli(spec) == name]
            by_account = next(spec for spec in mine if "@" in spec.split("/", 1)[0])
            by_variable = next(spec for spec in mine if "@" not in spec.split("/", 1)[0])
            raise isolation.SignInRefused(
                f"agent: {by_account} signs in by an account and {by_variable} by a token variable, and "
                f"{', '.join(endpoint)} is set: one environment cannot keep that endpoint for {by_variable} and "
                f"drop it for {by_account}. Sign both in the same way (both by the account, or both by the "
                f"token), or give {by_variable} its endpoint in the row and unset {', '.join(endpoint)} where the "
                "run is launched"
            )
    saved: dict[str, str] = {}
    for name, account in specs:
        if not account and by_token(name):
            continue  # signed in by a token variable: its endpoint goes with it
        for var in agents.endpoint_vars(name):
            if var in os.environ and var not in saved:
                saved[var] = os.environ.pop(var)
    try:
        yield sorted(saved)
    finally:
        os.environ.update(saved)


def _wall_refusal(e: BaseException) -> isolation.WallRefused | None:
    """The wall's refusal of a turn, when it is what failed it (Humanize wraps it in a `HarnessError`)."""
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, isolation.WallRefused):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


@contextlib.contextmanager
def _watchdog_window(cfg: RunConfig) -> Iterator[None]:
    """Give Humanize's watchdog a window of at least `WATCHDOG_FLOOR_S` for every CLI of the run, while it runs.

    The window is Humanize's own override (`HUMANIZE_WATCHDOG`, one value for every CLI), read at each turn:
    set to the longest of the floor and the run's CLIs' own windows, so no CLI's window gets shorter. A value
    the launcher set (`0` turns the watchdog off) stands. The variable is restored when the search ends.
    """
    if os.environ.get(WATCHDOG, "").strip():
        yield
        return
    clis = [spec.split("/", 1)[0].split("@", 1)[0] for spec in cfg.agent.specs]
    os.environ[WATCHDOG] = f"{max([WATCHDOG_FLOOR_S, *(silence(c) for c in clis)]):g}"
    try:
        yield
    finally:
        os.environ.pop(WATCHDOG, None)


def _watchdog_stop(e: BaseException) -> Failed | None:
    """The failure Humanize's watchdog ended the turn with, when it is what failed it, else None.

    The watchdog's own `Failed` is in the cause chain of the harness error the turn raised, unclassified (no
    `fault`) and worded `WATCHDOG_STOP ...`: a CLI's own failure keeps its own words, never these.
    """
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, Failed) and not cur.fault and str(cur.stderr or "").startswith(WATCHDOG_STOP):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _agent_jobs(host: Host) -> int:
    """How many jobs the agent submitted, as the scheduler records them (`origin: agent`); 0 when unknown."""
    listed = getattr(getattr(host, "sched", None), "list", None)
    if not callable(listed):
        return 0
    try:
        got = listed(origin="agent")
        return len(got) if isinstance(got, Sized) else 0
    except Exception:  # noqa: BLE001 -- a pluggable host: the count is evidence for the report, never a failure
        return 0


def _agent_ran(log: _Log, tools: _Log, *, jobs: int, seen_s: float, written: int) -> tuple[bool, str | None]:
    """Whether the agent took a turn, from facts the flow and the judge own, and a warning for a lost tool log.

    The tool log (`tools.jsonl`) is best effort: the CLI reports each call through a hook, and a hook that
    cannot reach the judge loses them silently. So any of these says the agent ran: a turn that returned (an
    idle return), a logged tool call, a job the scheduler holds from the agent (`origin: agent`), a turn the
    flow saw running without failing (`seen_s`: one the deadline, the budget or a stop cut), or output tokens
    the agent wrote (`written`, the API spend the end report prints).

    Args:
        log: The flow's events.
        tools: The tool log.
        jobs: The agent's jobs in the scheduler.
        seen_s: Seconds of turns the flow saw running and not failing.
        written: Output tokens of the agent's turns.

    Returns:
        Whether the agent ran, and a warning when it demonstrably worked (it submitted jobs, or wrote in a turn
        that never returned) while the tool log recorded nothing; None otherwise.
    """
    calls, idle = tools.counts.get("tool", 0), log.counts.get("idle", 0)
    ran = idle > 0 or calls > 0 or jobs > 0 or seen_s > 0 or written > 0
    if calls or not (jobs or (written and not idle)):
        return ran, None
    return ran, (
        f"tool log empty: the agent worked ({jobs} agent job(s), {seen_s:.0f} s of turns, {written} output "
        "tokens) but tools.jsonl recorded no tool call; the CLI's hook could not reach the judge"
    )


#: What a fatal end's error class means, in fixed words: stderr, FINAL.json and END-REPORT.md carry these, and
#: the agent CLI's own words go to flow.jsonl only, since a CLI's refusal can echo a key or an auth header.
FATAL_SAID = {
    "HarnessRefused": "the agent CLI refused the run (sign-in or permission)",
    "HarnessNotInstalled": "the agent CLI is not installed where the agent runs",
    "HarnessMissing": "the agent CLI would not start",
    "HarnessSandboxed": "this machine cannot hold the agent's sandbox (`sandbox: {name: bwrap}`)",
    "HarnessUnrecoverable": "the agent CLI failed in a way no resend changes",
    "HarnessUnrecoverable/goal_flow": "the endpoint does not take codex's goal flow",
}


def _fatal_cause(e: BaseException, *, goal: bool) -> str | None:
    """The fixed tag of a fatal end whose fix differs from its class's, read from the CLI's words (never shown).

    A Responses endpoint that keeps no state (`store=false`) answers the goal flow's follow-up with
    "function_call_output requires call_id or previous_response_id"; the same endpoint takes the plain flow
    (`agent.goal: false`). Other errors that name `call_id` keep their class's fix.
    """
    if (
        goal
        and isinstance(e, HarnessUnrecoverable)
        and re.search(r"requires (call_id|previous_response_id)", str(e))
    ):
        return "goal_flow"
    return None


def fatal_key(error: str, cause: str | None) -> str:
    """The key of `FATAL_SAID` (and the command's fix table) for a fatal end's class and cause."""
    return f"{error}/{cause}" if cause else error


def fatal_said(key: str, run_dir: Path) -> str:
    """A fatal end's line for the user: the class, its fixed words and where the CLI's own words are."""
    said = FATAL_SAID.get(key, "the agent CLI failed")
    return f"{key.split('/')[0]}: {said}; details in {run_dir / 'flow.jsonl'}"


def _why_not_run(log: _Log, ended: str) -> str:
    """Why the agent took no turn: a fatal end's fixed line, the wall's refusal, else how the agent phase ended."""
    fatal = log.last.get("fatal")
    if fatal is None:
        return ended
    if ended.startswith(EndKind.FATAL):
        return fatal_said(fatal_key(fatal["error"], fatal.get("cause")), log.path.parent)
    first = (str(fatal["message"]).strip().splitlines() or [""])[0]
    return f"{fatal['error']}: {first}" if first else fatal["error"]


def _write_final(cfg: RunConfig, final: Final, log: _Log) -> None:
    """Write `FINAL.json` atomically; a failure is logged, never raised (the run has nothing better to do)."""
    try:
        tmp = cfg.run_dir / "FINAL.json.tmp"
        tmp.write_text(final.model_dump_json(indent=1) + "\n")
        tmp.replace(cfg.run_dir / "FINAL.json")
    except OSError as e:
        log("final_write_error", error=type(e).__name__, message=str(e))


async def _finish(
    host: Host,
    cfg: RunConfig,
    log: _Log,
    *,
    sign_in_restarts: list[dict[str, str]] | None = None,
    never_ran: str | None = None,
    warning: str | None = None,
) -> tuple[Final, bool]:
    """Finalize the search, logging failures and continuing through each shutdown step.

    A failed generation guard selects only an already checked model, or nothing. Other runs fall back to
    the baseline. A cancellation (Ctrl-C) during the close does not cut it short: the step it lands in
    finishes, the rest follow, and the caller re-raises it once the files are written. With `never_ran`
    (the agent took no turn), FINAL.json says so (`agent_ran: false`) instead of the rule's own reason. A
    `warning` (the flow's own, such as a tool log that recorded nothing) is added to FINAL.json's `warnings`.

    Returns:
        The final pick, and whether the close was asked to stop.
    """
    stopped = False
    errors: dict[str, str] = {}

    async def step(fn: Callable[..., Any], *args: Any) -> tuple[bool, Any]:
        nonlocal stopped
        name = getattr(fn, "__name__", str(fn))
        work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        while True:
            try:
                return True, await asyncio.shield(work)
            except asyncio.CancelledError:
                if work.cancelled():  # the step itself was cancelled: there is no result
                    return False, None
                stopped = True
                log("stop_deferred", step=name)
            except Exception as e:  # noqa: BLE001 -- a pluggable host step may fail with any type
                log("host_error", step=name, error=type(e).__name__, message=str(e))
                errors[name] = f"{type(e).__name__}: {e}"
                return False, None

    try:
        await step(host.cutoff)
        log("drained", complete=(await step(host.drain, cfg.drain_window_s()))[1])
        checked, refused = await step(_trust_refusal, host)  # the digest can read a tree
        refused = refused if checked else ""
        if refused:
            # P2 at `none`: nothing keeps the agent out of the judge's files, so a changed trusted path
            # refuses the reading the run would have picked; the run records that instead of a result
            final = Final(
                rule="trust",  # no choice, no baseline: nothing was picked, the reading is refused
                reason=f"a trusted file changed while the agent could write it: {refused}",
                ledger=host.handins(),
            )
            log("trust_refused", reason=refused)
            final.sign_in_restarts = sign_in_restarts or []
            try:
                (cfg.run_dir / "FINAL.json").write_text(final.model_dump_json(indent=2))
            except OSError as e:
                log("final_write_error", error=type(e).__name__, message=str(e))
            log("final", rule=final.rule, choice=None, reason=final.reason)
            return final, stopped
        picked, final = await step(host.final)
        if not picked:
            configured = cfg.judge.final_rules
            guards = [rule for rule in configured if rule.reserve_s() > 0]
            if guards:
                reason = errors.get("final", "judge finalization failed")
                final = Final(
                    rule=guards[0].name,
                    reason="generation check failed",
                    generation_check={"outcome": "generation check failed", "reason": reason},
                )
                actual = getattr(getattr(host, "cfg", None), "final_rules", configured)
                for rule in actual:
                    if rule.reserve_s() <= 0:
                        continue
                    recovered, pick = await step(rule.failsafe, reason)
                    if recovered and pick is not None:
                        final = Final(
                            rule=rule.name,
                            choice=pick.choice,
                            baseline=pick.baseline,
                            reason=pick.reason,
                            generation_check=pick.generation_check or {},
                        ).with_delivery(cfg.judge.metric)
                        break
            else:
                final = Final(
                    rule="baseline",
                    baseline=cfg.baselines[0] if cfg.baselines else None,
                    reason="the judge made no pick (see host_error in flow.jsonl); the first baseline as-is",
                ).with_delivery(cfg.judge.metric)
        final = with_serve(final)  # the pick's kernel plan, from the same one renderer everywhere
        final.sign_in_restarts = sign_in_restarts or []
        write_args = (
            (cfg.run_dir / "FINAL.json", final) if sign_in_restarts else (cfg.run_dir / "FINAL.json",)
        )
        if not (picked and (await step(host.write_final, *write_args))[0]):
            try:  # the judge's pick (or the baseline) is kept; only the write is redone here
                (cfg.run_dir / "FINAL.json").write_text(final.model_dump_json(indent=2))
            except OSError as e:
                log("final_write_error", error=type(e).__name__, message=str(e))
        elif cfg.judge.custody_dir is not None:
            refreshed, exported = await step(host.final)
            if refreshed:
                final = with_serve(exported)
        final.sign_in_restarts = sign_in_restarts or []
        for (
            said
        ) in final.warnings:  # what the judge warns of in FINAL.json (the pick's reading) is on stderr too
            print(f"qda-agent: warning: {said}", file=sys.stderr)
        if never_ran is not None:
            fell_back = "; candidate 0 as-is" if final.baseline is not None and final.choice is None else ""
            final = final.model_copy(
                update={"agent_ran": False, "reason": f"the agent never ran ({never_ran}){fell_back}"}
            )
        if warning is not None:
            final = final.model_copy(update={"warnings": [*final.warnings, warning]})
        if never_ran is not None or warning is not None:
            _write_final(cfg, final, log)
        log(
            "final",
            rule=final.rule,
            choice=str(final.choice.dir) if final.choice else None,
            reason=final.reason,
        )
    finally:
        await step(host.stop)
    return final, stopped  # after the stop: a stop asked during it counts too


async def finalize(
    cfg: RunConfig,
    host: Host | None,
    *,
    reason: str,
    log: _Log | None = None,
    never_ran: str | None = None,
) -> Final:
    """End a search early: keep what it produced, write FINAL.json, and say why in one line.

    The one close for the three conditions that end a run before its deadline: the agent's API failing for
    `FINALIZE_AFTER_S`, the disk staying below `judge.min_free_bytes` for `FINALIZE_AFTER_S` with nothing to
    free, and a launch smoke that fails before the agent's first turn (called with `host=None`: nothing was
    built, so candidate 0 is the result). What the run already produced is kept: the judge cancels the
    agent's queued work, finishes the checks already queued within the drain window, prunes nothing, and
    applies the run's final rules (by default the agent's best-first hand-in list, else candidate 0 as-is).

    Args:
        cfg: The run config.
        host: The run's started host, or None when the run never started one (or already stopped it).
        reason: Why the run is ending early; printed as one line and recorded in FINAL.json's warnings.
        log: The flow's event log; a new one in the run dir when None.
        never_ran: Why the agent took no turn, when it took none (`Final.agent_ran`).

    Returns:
        The final pick, also written to `FINAL.json`. The run ends with `qda_quant.exit.ENVIRONMENT` (3):
        the machine (the API, the disk, the launch check) has to change before the same command can work.
    """
    log = log if log is not None else _Log(cfg.run_dir / "flow.jsonl", time.time)
    if host is None:
        # Nothing was built: candidate 0 is the result, as a run with no valid hand-in ends.
        baseline = cfg.baselines[0] if cfg.baselines else None
        final = Final(
            rule="baseline",
            baseline=baseline,
            reason=(
                f"the agent never ran ({reason}); candidate 0 as-is"
                if baseline is not None
                else f"the agent never ran ({reason}), and no baseline is configured"
            ),
            agent_ran=False,
            warnings=[f"the search was finalized early: {reason}"],
        )
        final = with_serve(final)
        _write_final(cfg, final, log)
    else:
        final, _stopped = await _finish(
            host,
            cfg,
            log,
            never_ran=never_ran,
            warning=f"the search was finalized early: {reason}",
        )
    pick = final.choice.dir if final.choice else final.baseline
    log("finalize", reason=reason, rule=final.rule, choice=str(pick) if pick else None)
    print(f"qda-agent: finalize: {reason}; FINAL.json: {pick or 'no checkpoint'}", file=sys.stderr)
    return final


def _trust_refusal(host: Host) -> str:
    """Why the final pick is refused, or "": a trusted path changed while the agent could write it.

    P2 at `isolation.level: none`: the level digests the judge's state, the reference and the scorer at
    the start (the judge's start, right after it installs) and compares them at every reading and here. A
    level that holds the wall (`sandbox`, `strict`) reads nothing and never refuses.
    """
    iso = getattr(getattr(host, "sched", None), "iso", None)
    if iso is None:
        return ""
    try:
        iso.verify_trusted(())
    except TrustChanged as why:
        return str(why)
    return ""


def _end_report(
    cfg: RunConfig,
    final: Final,
    log: _Log,
    tools: _Log,
    kind: EndKind,
    ended: str,
    ctx: FlowContext | None,
    t0: float,
    deadline: float,
    searched_s: float,
    *,
    shared_node: dict[str, str | None] | None = None,
    sign_in_restarts: Sequence[Mapping[str, str]] = (),
) -> None:
    usage = ctx.usage if ctx else None
    ratio = cfg.time_ratio(searched_s)
    # The run's own start-time record when it has one (the flow passes it); a direct caller gets the same
    # fields read here. A run with no acceptance adds no line.
    shared_node = shared_node if shared_node is not None else isolation.shared_node_record(cfg)
    pick = final.choice.dir if final.choice else final.baseline
    metric = cfg.judge.metric
    # the pick's reading when it came from a scored entry of the same files (`Final.metrics_from`)
    inherited = final.choice if final.choice and final.metrics_from else None

    def reading(h: HandIn) -> str:
        if metric in h.metrics:
            return f"{h.metrics[metric]:.5f}"
        if inherited is not None and final.is_pick(h) and metric in inherited.metrics:
            return f"{inherited.metrics[metric]:.5f} (from `{final.metrics_from}`: same files)"
        return ""

    rows = [
        f"| {h.kind} | `{h.dir}` | {h.valid}{'' if h.eligible else ', ineligible'} | {reading(h)} | {h.reason} |"
        for h in final.ledger
    ]
    # How a failed run names its end in the report; a completed or stopped run has no such line.
    failed = {
        EndKind.REFUSED: f"The wall {ended}",
        EndKind.FATAL: f"The agent phase ended {ended}",
    }.get(kind)
    head = f"**The run failed.** {failed}. No agent turn ran after that." if failed else None
    if final.agent_ran is False:  # FINAL.json's reason names why
        head = f"**The agent never ran.** {final.reason}."
    lines = [
        "# End report",
        "",
        *([head, ""] if head else []),
        *[f"**Warning:** {w}.\n" for w in final.warnings],
    ]
    # The result and what to do with it first; the run's own facts after.
    if pick is not None:  # a run that selected nothing has no submission to show (a trust refusal)
        lines += [f"## Submission: `{pick}` (rule `{final.rule}`)", ""]
        # FINAL.json's reading/gencheck decide, so the report and `evaluate` say the same thing
        if gaps := unmeasured(final.reading, final.gencheck, metric):
            lines += [f"**{gaps}**", ""]
        if rotate := rotation_hint(pick):  # a speculative draft must be in the checkpoint's basis
            lines += [f"**Rotation:** {rotate}.", ""]
        if final.serve:
            lines += ["Serve it with:", "", "```sh", final.serve, "```", ""]
        if cfg.baselines:  # `qda eval RUN_DIR` reads candidate 0 from baselines[0]
            lines += ["Score it and candidate 0 on held-out prompts:", "", "```sh"]
            lines += [f"qda eval {shlex.quote(str(cfg.run_dir))}", "```", ""]
    lines += [*([final.reason] if final.reason else []), *[f"- not {w}" for w in final.why_not], ""]
    lines += [
        "## Run",
        "",
        f"- model `{cfg.model}`, format `{cfg.format}`, {len(cfg.cards)} card(s), "
        f"{cfg.budget_gpu_hours:g} GPU-h",
        f"- start {_stamp(t0)}, GPU deadline {_stamp(deadline)}; agent phase ended: {ended}",
        *(
            [
                f"- search wall time {searched_s:.0f} s; time ratio {ratio:g} "
                f"(against candidate 0's {cfg.judge.candidate0_build_s:.0f} s build)"
            ]
            if ratio is not None
            else []
        ),
        f"- agent `{cfg.agent.spec}` (fallbacks: {', '.join(cfg.agent.specs[1:]) or 'none'}), "
        f"isolation `{cfg.isolation.level}`, jobs `{cfg.job_sandbox()}`",
        *(
            [
                f"- API spend: ${usage.cost:.2f}, {usage.output_tokens} output tokens, "
                f"{usage.duration.total_seconds():.0f} s of turns"
            ]
            if usage is not None
            else []
        ),
        f"- agent turns: {log.counts.get('goal', 0)} instructions sent, {log.counts.get('idle', 0)} idle returns, "
        f"{log.counts.get('failed', 0)} failed; {tools.counts.get('tool', 0)} tool calls, "
        f"{log.counts.get('host_error', 0)} host errors",
        *[
            f"- sign-in restart: {r['outcome']}: "
            + (f"`{r['session_id']}`" if r["outcome"] == "resumed" else r["reason"])
            for r in sign_in_restarts
        ],
        *(
            [
                f"- shared node, accepted by {shared_node['accepted_by']}: {shared_node['reason']}; other work on "
                f"the node at the start: {shared_node['other_work'] or 'none found'}"
            ]
            if shared_node
            else []
        ),
        "",
        "## Readings",
        "",
        f"| kind | dir | valid | {label(metric)} | reason |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## Files",
        "",
        "- `FINAL.json`, `flow.jsonl` (the flow's events), `tools.jsonl` (every tool call), `work/notes/`",
    ]
    (cfg.run_dir / "END-REPORT.md").write_text("\n".join(lines) + "\n")


@flow(agents=GoalAgents, envs=Envs, params=SearchParams)
async def qda_search(
    task: str, *, agents: GoalAgents, envs: Envs, params: SearchParams, ctx: FlowContext
) -> dict[str, Any]:
    """Quantization search: the agent pursues one /goal until the GPU deadline."""
    del task  # the instruction is fixed; the task statement comes from the run config
    cfg = RunConfig.load(params.config)
    return await search(agents["worker"], envs["workspace"], cfg, goal=True, ctx=ctx)


@flow(agents=PlainAgents, envs=Envs, params=SearchParams)
async def qda_search_plain(
    task: str, *, agents: PlainAgents, envs: Envs, params: SearchParams, ctx: FlowContext
) -> dict[str, Any]:
    """Quantization search for a CLI without /goal: the instruction is re-sent whenever it goes idle."""
    del task
    cfg = RunConfig.load(params.config)
    return await search(agents["worker"], envs["workspace"], cfg, goal=False, ctx=ctx)
