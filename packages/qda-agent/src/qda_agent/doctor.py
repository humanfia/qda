"""The search prerequisites of `qda doctor --search CONFIG.yaml`, one check line each.

The plugin `search` is qda-agent's entry in the ``qda.checks`` registry: `qda doctor` runs it when qda-agent
is installed and a config is given. It walks the start's own guards, the ones `qda-agent run` refuses on before
it creates anything (`cli.START_GUARDS`, `judge.startup.STATIC_GUARDS`), so each of those lines is the run's
refusal word for word, and lists what only the run's launch decides (`judge.startup.LAUNCH_GUARDS`). Its other
lines inspect job launch support and open one job-readable weight, evaluator imports, anchor files and socket
path length; what a card, a compiler and an engine really do is left to the launch's trial run
(`judge.startup.launch_smoke`), which does it on this machine. The plugin `disk room` reports the
cache and temporary disks (`room.cache_room`), with or without a config. With ``--report`` both list facts
only: the card numbering, the filesystems, a config's unconfigured components.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError
from qda_quant.checks import HEALTH_FIELDS, card_fault
from qda_quant.doctor import Check, Context, Info
from qda_quant.exit import USAGE

from .isolation.api import job_sandbox
from .isolation.bwrap import (
    BWRAP_FIX,
    BwrapTooOld,
    bwrap_argv,
    pid_form,
    preflight,
    proc_form,
    require_bind_fd,
)
from .isolation.devices import (
    NV_PROC,
    CardError,
    cards_by_bus,
    job_visible,
    resolve_cards,
    smi_bus,
)
from .room import cache_room, room, room_report
from .sched.config import SchedConfig

if TYPE_CHECKING:
    from .config import RunConfig

#: The sandboxes whose turns run inside Humanize's fence as a *network* fence, which needs `pidfd_getfd` for
#: the listener it keeps on loopback. The bwrap wall cuts the network with a namespace of its own instead and
#: puts only the filesystem fence inside (`isolation.wall`), so it needs no `pidfd_getfd` -- which a DGX
#: node refuses.
NET_FENCED = ()


def _out(
    argv: list[str], env: dict[str, str] | None = None, *, pass_fds: tuple[int, ...] = ()
) -> tuple[bool, str]:
    """Return command success and stdout, or the last output line or exception name on failure."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=180, env=env, pass_fds=pass_fds)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, type(e).__name__
    return p.returncode == 0, p.stdout if p.returncode == 0 else (
        (p.stdout + p.stderr).strip() or " "
    ).splitlines()[-1]


def pidfd_problem() -> str | None:
    """Explain a missing `os.pidfd_open`, or return None when it is available.

    Humanize 0.1's fence probe requires this call. Use a Python build that supplies it for fenced runs.
    """
    if hasattr(os, "pidfd_open"):
        return None
    return (
        f"this Python ({sys.executable}, {sys.version.split()[0]}) has no os.pidfd_open, and Humanize 0.1's "
        "fence crashes without it (isolation bwrap and fence); use a Python that has it, such as the "
        "repository's pinned 3.12 (`uv sync` picks it from .python-version)"
    )


def _job_binds(cfg: SchedConfig, workdir: Path, empty: Path) -> dict[Path, Path]:
    """`cfg.binds`, plus an empty read-only directory at each missing parent under the workdir.

    A run binds the toolset's overlay first (`work/baselines` -> its copy), so a component's deeper bind
    (`work/baselines/0` -> a baseline) mounts under one. Before the first run the overlay does not exist,
    so those parent binds are absent from `cfg.binds` and `mount_points` would drop every deeper bind. An
    empty parent keeps their destinations and mount relationship, so they are read as a job reads them.
    Each mount made under an empty parent gets its directory in that shared source, or bwrap would have to
    make it inside the parent's read-only mount and fail.
    """
    out: dict[Path, Path] = {}
    for dst in cfg.binds:
        parent = Path(dst).parent
        while parent != workdir and parent.is_relative_to(workdir) and parent not in cfg.binds:
            out.setdefault(parent, empty)
            parent = parent.parent
    for dst, _ in (*out.items(), *cfg.binds.items()):
        if Path(dst).parent in out:
            (empty / Path(dst).name).mkdir(exist_ok=True)
    out.update(cfg.binds)
    return out


def _work(cfg: SchedConfig, workdir: Path | None, root: Path) -> Path:
    """The run's real workdir (`run_dir/work` by default), or `root` when no run is configured."""
    if workdir is not None:
        return Path(workdir)
    return Path(cfg.run_dir) / "work" if cfg.run_dir is not None else root


def _views(cfg: SchedConfig, workdir: Path | None, root: Path) -> dict[Path, Path]:
    """`cfg.binds` with each destination under the run's workdir moved under the throwaway `root`.

    A probe mounts what a job mounts, at the same relative paths, but never under the real workdir: bwrap
    makes every missing mount point inside its writable root, and a run refuses a run dir that is not empty.
    """
    work = _work(cfg, workdir, root)
    return {
        root / Path(dst).relative_to(work) if Path(dst).is_relative_to(work) else Path(dst): src
        for dst, src in cfg.binds.items()
    }


def check_bwrap(cfg: SchedConfig, workdir: Path | None = None, *, level: str = "strict") -> Check:
    """Check that bwrap runs `true`, and opens job-readable weights, in the job's sandbox.

    `bwrap true` only proves the sandbox starts. When `scheduler.readable` or a `scheduler.binds` source
    holds a file, the check reads its first bytes through exactly the mounts a job gets
    (`sandbox.preflight`: every link hop of a snapshot, and every configured bind at the workdir it is
    mounted under), so a model or a baseline a job could not read is a MISS here
    instead of jobs that cannot start. The job's workdir is a throwaway directory with the same layout:
    nothing is written under `run_dir`, so `qda-agent run` can start in it afterwards.

    Args:
        cfg: The run's scheduler settings.
        workdir: The agent's workdir (`run_dir/work` by default); its binds are mounted at the same paths
            under the throwaway one.
        level: The run's `isolation.level`: `none` runs plain jobs, so there is no job sandbox to check;
            the sandboxed levels keep `scheduler.job_sandbox`, whose `none` they honour (`isolation.api.job_sandbox`).
            Only `strict` requires bubblewrap, so only there is a missing one a MISS.
    """
    if job_sandbox(level, cfg.job_sandbox) == "none":
        why = (
            "isolation.level: none (jobs run as plain processes)"
            if level == "none"
            else "scheduler.job_sandbox: none (jobs run unisolated)"
        )
        return Check("sandbox", True, why, skipped=True)
    if shutil.which("bwrap") is None:
        if level != "strict":
            why = f"bwrap: not found; isolation.level: {level} does not require it"
            return Check("sandbox", True, why, skipped=True)
        return Check("sandbox", False, "bwrap: not found", BWRAP_FIX)
    try:  # one probe for doctor and the run: the job sandbox builder asks it too (`bwrap.require_bind_fd`)
        require_bind_fd()
    except BwrapTooOld as e:
        return Check("sandbox", False, e.why, e.fix)
    run = cfg.run_dir is not None
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "work"
        root.mkdir()
        ok, out = _out(bwrap_argv(["true"], str(root), writable=[root]))
        weight: Path | None = None
        if ok:
            binds = _views(cfg, workdir, root)
            if run:
                empty = Path(d) / "empty"
                empty.mkdir()
                binds = _job_binds(cfg.model_copy(update={"binds": binds}), root, empty)
            jobs = Path(cfg.run_dir) / "jobs" if cfg.run_dir is not None else None
            try:
                weight = preflight(
                    cfg.readable,
                    binds,
                    writable=[root],
                    run_dir=cfg.run_dir,
                    host_written=[*cfg.host_written, *([jobs] if jobs is not None else [])],
                    hide=cfg.hide,
                )
            except ValueError as why:  # named at the path a job sees, not the probe's throwaway copy
                detail = str(why).replace(str(root), str(_work(cfg, workdir, root)))
                return Check("sandbox", False, detail, "fix scheduler.readable or the job sandbox's binds")
    proc = "a fresh /proc" if proc_form() == "fresh" else "the host's /proc, read-only"
    pid = pid_form()
    detail = f"{proc}; pid: {pid}" + (" (a dedicated node is required)" if pid == "shared" else "")
    if ok and weight is not None:
        try:
            size = f" ({weight.stat().st_size} bytes)"
        except OSError:  # gone since the read: report the line without its size
            size = ""
        detail += f"; reads {weight.name}{size} through the job's binds"
    return Check(
        "sandbox",
        ok,
        f"bwrap runs a sandboxed job ({detail})" if ok else f"bwrap: {out}",
        BWRAP_FIX,
    )


def card_health(cards: list[int], proc: Path | None = None) -> tuple[str | None, str | None]:
    """The run's cards nvidia-smi reports faulty (`qda_quant.checks.card_fault`): (refusal, why not judged).

    Cards are matched by PCI bus, so a fault is pinned to the run's card even when nvidia-smi numbers cards
    differently -- including a confined job, where `resolve_cards` maps the job-local number to the bus the
    job's own nvidia-smi reports. The search's start refuses on it (`cli.START_GUARDS`' ``card health``). The
    second value says why nothing was judged (no card to name, or no nvidia-smi listing), which another line
    reports.
    """
    try:
        found = resolve_cards(cards, proc)
    except CardError:
        return None, "the cards line names the problem"
    argv = ["nvidia-smi", f"--query-gpu=pci.bus_id,{','.join(HEALTH_FIELDS)}", "--format=csv,noheader"]
    try:  # its rows, whatever its status: a node with one broken card can still list the others
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, "no nvidia-smi listing"
    faults: dict[str, str | None] = {}
    for row in (line.split(",") for line in out.splitlines()):
        if len(row) == 1 + len(HEALTH_FIELDS) and ":" in row[0]:
            faults[smi_bus(row[0])] = card_fault(row[1:])
    if not found or not faults:
        return None, "no nvidia-smi listing of the run's cards"
    bad = [
        f"card {n}: {faults[c.bus] if c.bus in faults else 'nvidia-smi gives no row for it'}"
        for n, c in zip(cards, found, strict=False)
        if faults.get(c.bus, "missing")
    ]
    return ("; ".join(bad) if bad else None), None


#: What a search's start tells the user to do about a faulty card.
CARD_HEALTH_FIX = "remove the card from cards, or have it reset or drained"


def card_order_info(cfg: SchedConfig, proc: Path | None = None) -> Info | None:
    """How `cards` and nvidia-smi number this machine's GPUs: ``--report``'s fact, with no verdict.

    On a whole node `cards` are PCI-bus-order ordinals, and nvidia-smi's index is normally the same number.
    In a job that sees only part of the node (Slurm's device cgroup, a container) nvidia-smi numbers the
    job's cards from 0, and so do `cards`: the line reports that job-local numbering. Nothing refuses on it;
    it is here because a bug report about the wrong card is read with it.
    """
    proc = NV_PROC if proc is None else proc
    if not cfg.gpus:
        return Info("card order", "no GPU (scheduler.gpus: false)")
    if cfg.fake_gpus or not proc.is_dir():
        return Info("card order", "no NVIDIA driver")
    ok, out = _out(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
    if not ok:
        return Info("card order", f"nvidia-smi: {out}")
    smi = {}
    for i, bus in (line.split(",") for line in out.splitlines() if "," in line):
        smi[int(i)] = smi_bus(bus)
    hidden = job_visible(proc)
    if hidden is not None:
        try:
            cards = resolve_cards(cfg.cards, proc, hidden)
        except CardError:
            return None  # the cards this machine has is the GPU line's fact, not this one's
        bad = [n for n, c in zip(cfg.cards, cards, strict=False) if smi.get(n) != c.bus]
        if bad:
            return Info("card order", f"cards {bad}: nvidia-smi's job-local index names another card")
        return Info(
            "card order",
            f"job-local cards {cfg.cards}: nvidia-smi lists {len(hidden)} of the node's "
            f"{len(cards_by_bus(proc))} GPU(s)",
        )
    by_bus = {c.ordinal: c.bus for c in cards_by_bus(proc)}
    bad = [c for c in cfg.cards if smi.get(c, "?") != by_bus.get(c)]
    if bad:
        return Info("card order", f"cards {bad}: nvidia-smi index != PCI-bus-order ordinal")
    return Info("card order", f"cards {cfg.cards}: nvidia-smi index == PCI-bus-order ordinal")


def check_components(components: list) -> list[Info]:
    """Note a serving or data component with no configured input (``--report``'s lines for a config).

    Empty components may be intentional, so they are facts, not checks.
    """
    out = []
    for c in components:
        if c.name == "serve" and getattr(c, "python", None) is None:
            out.append(
                Info(
                    "serve",
                    "not configured (serve.python or judge.serve_python), the agent's jobs get no vLLM",
                )
            )
        elif c.name == "data" and not getattr(c, "sets", None):
            out.append(Info("data", "no datasets (sets), DATA.md is not written"))
    return out


def disk_room(ctx: Context) -> list[Check] | list[Info]:
    """Free bytes and inodes on the cache and temporary disks: `room.cache_room`, which prepare refuses on.

    The run directory's disk is not measured here: that is the disk-cap line's (`check_disk_cap`). With
    ``--report``, the same disks by variable and filesystem, and the quota notes (`room.room_report`).
    """
    return room_report() if ctx.report else cache_room()


def check_disk_cap(cfg: RunConfig) -> Check:
    """MISS when the run's own start check refuses its disk: the free-space floor, and a cap set by hand.

    The check is `custody.check_space`, the one `JudgeHost` refuses to start on (its `ValueError` is this
    line's detail, so a MISS and a refusal cannot disagree): free space below the floor (`free_floor`:
    `judge.min_free_bytes` when set by hand, else twice the original model's bf16 weights), and with
    a cap, a cap below what the run must hold or above the free disk. Without a cap (the
    default) there is no estimate to check: the line states the floor and the free space it is read against,
    the same floor value the host computes.
    """
    from qda_agent.judge.custody import DiskRefused, build_bytes, check_space, free_floor
    from qda_agent.judge.gguf_template import run_template_bytes

    judge = cfg.judge
    cap = judge.max_run_bytes
    where = judge.custody_dir or cfg.run_dir
    try:
        check_space(where, judge, len(cfg.cards))
    except DiskRefused as refusal:  # the start's own `disk space` guard, which this line stands for
        return Check("disk cap", False, " ".join(str(refusal).split()), refusal.problem.fix)
    if cap is None:
        free = room(where)
        floor = free_floor(judge)
        return Check(
            "disk cap",
            True,
            f"none (judge.max_run_bytes unset): {free.free_bytes / 1e9:.1f} GB free on {free.path}; a build is "
            f"stopped when it drops below the judge.min_free_bytes floor of {floor / 1e9:.1f} GB",
        )
    # A llama.cpp search that converts its bf16 template at start holds it in the run, once, inside the cap.
    template = run_template_bytes(judge)
    build = judge.budget_bytes or 0
    if template and template + build > cap:
        return Check(
            "disk cap",
            False,
            f"judge.max_run_bytes {cap / 1e9:.1f} GB cannot hold the run's bf16 GGUF template "
            f"(about {template / 1e9:.1f} GB, made once at start) and one {build / 1e9:.1f} GB build: its first "
            "build after the template is over the cap (doctor's estimate; the start does not check it)",
            "name a prepared template with judge.llamacpp.template (`qda-agent prepare` makes one), or raise "
            "judge.max_run_bytes",
        )
    free = room(where)
    held = f", the bf16 GGUF template about {template / 1e9:.1f} GB of it once" if template else ""
    # `check_space` checks the cap against the run's builds only when the config names one build's size; say
    # when it did not, so an OK line never stands for a check that was not made.
    sized = (
        ""
        if build_bytes(judge) is not None
        else "; not checked against the run's builds: no build size is known for this config"
    )
    return Check(
        "disk cap",
        True,
        f"{cap / 1e9:.1f} GB of {free.free_bytes / 1e9:.1f} GB free on {free.path}{held}{sized}",
    )


def check_pid_rule(cfg: RunConfig) -> Check | None:
    """MISS when the search's start refuses a `pid: shared` turn on a node it does not hold (`isolation.pid_refusal`).

    None where the rule does not apply (no bwrap agent wall); an OK line where the node is the run's or the
    config names `isolation.shared_node`.
    """
    from qda_agent import isolation

    if cfg.isolation.level == "none" or cfg.sandbox.name != "bwrap":
        return None
    if (why := isolation.pid_refusal(cfg)) is not None:
        return Check("pid rule", False, why, isolation.PID_FIX)
    pid = isolation.pid_form(cfg.sandbox.pid)  # the resolver the rule itself asks (`shared_pid_refusal`)
    accepted = cfg.isolation.shared_node
    if pid == "shared" and accepted is not None:
        return Check(
            "pid rule", True, f"pid: shared; isolation.shared_node accepted by {accepted.accepted_by}"
        )
    return Check(
        "pid rule",
        True,
        f"pid: {pid}" + (" on a node no other process of yours uses" if pid == "shared" else ""),
    )


def check_run_dir_place(cfg: RunConfig) -> Check:
    """MISS when the search's start refuses where its run directory is (`isolation.run_dir_in_tmp`, word for word)."""
    from qda_agent import isolation

    if (why := isolation.run_dir_in_tmp(cfg)) is not None:
        return Check("run dir", False, why, isolation.RUN_DIR_FIX, code=USAGE)
    where = "" if cfg.isolation.level == "none" else ", outside the temporary directories the agent can write"
    return Check("run dir", True, f"{cfg.run_dir.resolve()}{where}")


def search(ctx: Context) -> list[Check] | list[Info]:
    """What a search needs of this machine that a config decides: `qda doctor --search CONFIG.yaml`.

    Two things are read from the config and nothing else: the isolation level, which says whether the run
    needs bubblewrap, and the run directory, whose disk the run writes to. The search's own conditions --
    its evaluator, its references, its sockets, its cards, the programs its jobs run -- are refused by the
    search itself when it starts, and the launch's trial run (`judge.startup.launch_smoke`) builds, serves
    and reads candidate 0 on this machine before the agent gets a turn. Doctor repeats none of it. A config
    that does not load is the input (exit 2). With ``--report``, the facts only: card numbering and the
    config's unconfigured components.
    """
    if ctx.search is None:
        return []  # nothing in scope: the tail line says what was checked
    from datetime import UTC, datetime

    from qda_agent.config import RunConfig
    from qda_agent.configfile import ConfigError

    try:
        cfg = RunConfig.load(ctx.search, list(ctx.overrides))
    except ConfigError as e:
        if ctx.report:
            return [Info("run config", f"does not load: {e}")]
        return [Check("run config", False, str(e), e.fix or f"edit {ctx.search}", code=USAGE)]
    sched = cfg.sched_config(datetime.now(UTC))
    if ctx.report:
        found: list[Info] = check_components(list(cfg.toolset.components))
        with contextlib.suppress(ValidationError):
            if (order := card_order_info(SchedConfig.model_validate(sched))) is not None:
                found.append(order)
        return found
    checks = [
        Check(
            "run config",
            True,
            f"{cfg.model} {cfg.format}, {cfg.wall_s / 3600:.2f} h on {len(cfg.cards)} card(s)"
            + ("" if cfg.work_dir is None else f"; the agent's workdir is work_dir {cfg.workdir}"),
        )
    ]
    try:
        s = SchedConfig.model_validate(sched)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(map(str, first["loc"])) or "section"
        return [*checks, Check("config", False, f"{where}: {first['msg']}", "fix the scheduler section")]
    checks.append(check_run_dir_place(cfg))
    checks.append(check_bwrap(s, cfg.workdir, level=cfg.isolation.level))
    if (pid := check_pid_rule(cfg)) is not None:
        checks.append(pid)
    checks.append(check_disk_cap(cfg))
    checks.append(check_agent_cli(cfg))
    checks.append(check_sign_in(cfg))
    return checks


def check_agent_cli(cfg: RunConfig) -> Check:
    """MISS when an agent the run starts cannot be: a CLI not on `PATH`, or an SDK not in this interpreter.

    The run would find out only when it starts the agent, after candidate 0's trial run (minutes in). The
    primary backend and each fallback are looked up as the run looks them up: the executable `cli` on this
    shell's `PATH`, or the backend's `modules` (`agents.sdk_missing`, the rule its own turn check asks).
    """
    from .agents import sdk_install, sdk_missing

    found: list[str] = []
    for backend in (cfg.agent.backend, *cfg.agent.fallbacks):
        if (cli := backend.cli) is None:
            continue
        if backend.modules:  # driven through an SDK: no executable to look for on PATH
            if missing := sdk_missing(backend.modules):
                return Check(
                    "agent CLI",
                    False,
                    f"{cli}: {', '.join(missing)} not installed in {sys.prefix}",
                    f"{sdk_install(cli)} (into the interpreter that runs qda-agent), or change agent.backend",
                )
            found.append(f"{cli}: SDK in {sys.prefix}")
        elif (path := shutil.which(cli)) is None:
            return Check(
                "agent CLI",
                False,
                f"{cli} is not on PATH: the search would stop when it starts the agent",
                f"install {cli} and put it on PATH, or change agent.backend",
            )
        else:
            found.append(path)
    return Check("agent CLI", True, ", ".join(found) or f"none: {cfg.agent.spec} runs in this process")


def check_sign_in(cfg: RunConfig) -> Check:
    """Report the agent's sign-in kind, source and coverage, and refuse where the run's own check refuses.

    The check is `isolation.check_sign_in`, the one `flow._open` runs at the same deadline, so a sign-in the
    run stops on is a MISS here and never the `opt` line `sign_in_report` alone would give. Its two kinds of
    refusal carry their own fix: a configuration naming two dirs for one CLI is `sign_in_dirs`', and asks for
    one dir; a sign-in that cannot last the run is `check_sign_in`'s, and asks for the CLI's long-run sign-in
    (`isolation.long_run_fix`: for claude, `claude setup-token` and `CLAUDE_CODE_OAUTH_TOKEN`).
    """
    from datetime import UTC, datetime, timedelta

    from qda_agent import isolation

    deadline = datetime.now(UTC) + timedelta(seconds=cfg.wall_s)
    try:
        isolation.sign_in_dirs(cfg)  # two agents of one CLI naming different dirs is this one's refusal
    except isolation.SignInRefused as why:
        return Check("agent sign-in", False, " ".join(str(why).split()), "name one agent.backend.sign_in_dir")
    try:
        isolation.check_sign_in(cfg, deadline)
        ok, optional, lines = isolation.sign_in_report(cfg, deadline)
    except isolation.SignInRefused as why:
        fixes = [  # the CLIs whose sign-in file records an expiry: the ones this refusal can be about
            isolation.long_run_fix(name, isolation.sign_in_dirs(cfg).get(name))
            for name, src in isolation.sign_in_accounts(cfg).items()
            if src is not None and isolation.CLI_CONFIG[name][4] is not None
        ]
        return Check(
            "agent sign-in",
            False,
            " ".join(str(why).split()),
            "; ".join(dict.fromkeys(fixes)) or "sign in again",
        )
    # a sign-in the report finds short but the run's own check lets start is said, not judged: the run
    # refreshes it where it can, and only `check_sign_in`'s refusal stops a run
    return Check("agent sign-in", ok or optional, "; ".join(lines), "sign in using the source named above")
