"""`qda-agent run CONFIG.yaml`: start a search; `qda doctor --search CONFIG.yaml` checks the machine first.

`qda-agent prepare --model M --out DIR` makes every input of a search and DIR/search.yaml;
`qda-agent search DIR` checks the machine and runs it; `qda eval RUN_DIR` scores the result on the
holdout prompts. `qda-agent status RUN_DIR` prints a run's progress now, from the files the run writes.
`qda-agent gate CONFIG.yaml` runs the search's start phase alone and prints its W.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from qda_quant.exit import ENVIRONMENT, USAGE, refuse
from qda_quant.guards import SEARCH_START, Guard, Problem, first_problem
from qda_quant.io import line_buffered_stdout, show_only
from qda_quant.output.record import write_resolved

from qda_agent import summary
from qda_agent.config import RunConfig
from qda_agent.configfile import ConfigError
from qda_agent.isolation.api import make as make_isolation

__all__ = ["START_GUARDS", "close", "execute", "main", "pidfd_refusal", "start_refusal"]


def pidfd_refusal(cfg: RunConfig) -> str | None:
    """Why this run cannot start on this Python (`os.pidfd_open`), or None.

    Only `sandbox: fence` needs it: the bwrap wall cuts the network with a namespace of its own and puts only
    the filesystem fence inside, which needs no descriptor taken out of another process.
    """
    from qda_agent.doctor import NET_FENCED, pidfd_problem

    return pidfd_problem() if cfg.sandbox.name in NET_FENCED else None


def work_dir_refusal(cfg: RunConfig) -> str:
    """The refusal for a `work_dir` that exists already: the agent's work starts in a fresh directory."""
    return f"work_dir {cfg.workdir} exists; give a fresh one per run"


def run_dir_refusal(cfg: RunConfig) -> str | None:
    """Why the run's own directory cannot be used; None: it can (or does not exist yet).

    `qda doctor --search` reports this through `check_run_start`, the other half of what `run_dir` promises.
    """
    if cfg.run_dir.exists() and not cfg.run_dir.is_dir():  # `iterdir` would raise, and `mkdir` after it too
        return f"run_dir {cfg.run_dir} is not a directory; give a directory"
    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        return f"{cfg.run_dir} is not empty; give a new run_dir"
    return None


def _no_gpus(cfg: RunConfig) -> str | None:
    """Why the run has no cards to check, or None (the scheduler plugin's own settings type both fields)."""
    if not getattr(cfg.scheduler, "gpus", True):
        return "no GPU (scheduler.gpus: false)"
    if getattr(cfg.scheduler, "fake_gpus", False):
        return "fake GPUs"
    return None


def _run_dir(cfg: RunConfig) -> Problem | None:
    if (why := run_dir_refusal(cfg)) is None:
        return None
    return Problem(why, f"rerun the same command with --set run_dir=NEW_DIR, or delete {cfg.run_dir}", USAGE)


def _run_dir_in_tmp(cfg: RunConfig) -> Problem | None:
    from qda_agent import isolation

    if (why := isolation.run_dir_in_tmp(cfg)) is None:
        return None
    return Problem(why, isolation.RUN_DIR_FIX, USAGE)


def _cards_visible(cfg: RunConfig) -> Problem | None:
    from qda_agent.isolation.devices import invisible_cards

    return None if (why := invisible_cards(cfg.cards)) is None else Problem(why)  # cards: physical numbers


def _python_fence(cfg: RunConfig) -> Problem | None:
    # this Python cannot hold the fence: a different interpreter can
    return None if (why := pidfd_refusal(cfg)) is None else Problem(why)


def _sandbox_hold(cfg: RunConfig) -> Problem | None:
    from qda_agent import isolation

    # the wall, or this machine's own fence
    return None if (why := isolation.hold_problem(cfg)) is None else Problem(why)


def _pid_rule(cfg: RunConfig) -> Problem | None:
    from qda_agent import isolation

    if (why := isolation.pid_refusal(cfg)) is None:
        return None
    return Problem(why, isolation.PID_FIX)


def _card_health(cfg: RunConfig) -> Problem | None:
    from qda_agent.doctor import CARD_HEALTH_FIX, card_health

    bad, _not_judged = card_health(list(cfg.cards))
    return None if bad is None else Problem(bad, CARD_HEALTH_FIX)


def _cards_not_judged(cfg: RunConfig) -> str | None:
    from qda_agent.doctor import card_health

    if (why := _no_gpus(cfg)) is not None:
        return why
    return card_health(list(cfg.cards))[1]


def _healthy(cfg: RunConfig) -> str:
    return f"no fault on cards {list(cfg.cards)}"


def _work_dir(cfg: RunConfig) -> Problem | None:
    if cfg.work_dir is None or not cfg.workdir.exists():
        return None
    return Problem(work_dir_refusal(cfg), "give a fresh work_dir per run (--set work_dir=...)", USAGE)


_START = frozenset({SEARCH_START})
_EXECUTE = "qda-agent run: before it creates anything (cli.execute)"

#: What a search refuses before it creates anything, in `execute`'s order: the run directory (in use, or under a
#: temporary directory the sandboxed agent can write), cards this
#: process cannot see, a Python that cannot hold the fence, a machine that can hold the agent with neither the
#: wall nor the native fence, the wall's pid rule, a `work_dir` that exists already, and a card nvidia-smi
#: reports faulty. `start_problem` walks
#: it, and `qda doctor --search` walks the same tuple (`qda_quant.guards`).
START_GUARDS: tuple[Guard[RunConfig], ...] = (
    Guard("run dir", _START, _run_dir, _EXECUTE),
    Guard("run dir place", _START, _run_dir_in_tmp, _EXECUTE),
    Guard("cards visible", _START, _cards_visible, _EXECUTE, applies=_no_gpus),
    Guard("python fence", _START, _python_fence, _EXECUTE),
    Guard("sandbox hold", _START, _sandbox_hold, _EXECUTE),
    Guard("pid rule", _START, _pid_rule, _EXECUTE),
    Guard("work dir", _START, _work_dir, _EXECUTE),
    Guard("card health", _START, _card_health, _EXECUTE, applies=_cards_not_judged, held=_healthy),
)


def start_fix(refused: BaseException) -> str | None:
    """The fix a start refusal's cause names (`judge.startup.ReferenceUnreadable`'s), or None."""
    fix = getattr(refused.__cause__, "fix", None)
    return fix if isinstance(fix, str) and fix else None


def start_problem(cfg: RunConfig) -> tuple[str, str | None, int] | None:
    """Why this run cannot start here, as (problem, fix, exit status); None when it can.

    `execute`'s pre-mkdir block: the first problem of `START_GUARDS`. The run directory and the work dir are
    the input (status 2); the others are this machine (3, `qda_quant.exit`).
    """
    found = first_problem(START_GUARDS, SEARCH_START, cfg)
    return None if found is None else (found.text, found.fix, found.code)


def start_refusal(cfg: RunConfig) -> str | None:
    """`start_problem`'s sentence alone: what `qda doctor --search` prints instead of passing a search that
    would stop here."""
    found = start_problem(cfg)
    return None if found is None else found[0]


def execute(cfg: RunConfig, *, worker: Any = None) -> int:
    """Run the search `cfg` describes, optionally with a supplied agent driver.

    Without `worker`, the backend's driver fills the role, or Humanize starts the backend's CLI.

    The run writes `run_dir/resolved.yaml` first (`RunConfig.resolved`). However the run ends (done, stopped with
    Ctrl-C, SIGTERM or SIGHUP, or failed), it is closed the same way (`close`).

    Returns:
        The exit status (`qda_quant.exit`): 0 done (the run reached its GPU deadline), 2 for the run's input (a
        run dir that is not empty, a `work_dir` that exists already, a config whose sign-ins conflict, Humanize
        refusing the run, a start this run's own checks refuse, or an end no run reached), 3 for the machine
        (cards this process cannot see, a Python that cannot hold the fence, a host that holds neither the wall
        nor the native fence), 130 stopped from outside.
    """
    from hmz.sdk import Refused

    from qda_agent import backends, flow, isolation, progress
    from qda_agent.host import CLOSING, STOPPING
    from qda_agent.sched import kill_every_group

    if (found := start_problem(cfg)) is not None:
        problem, fix, code = found
        return refuse(problem, fix, code=code)
    if worker is None:  # a real agent CLI: its model API must be reachable by the route its turns take
        from qda_quant.guards import SEARCH_LAUNCH

        from qda_agent.api_route import AGENT_GUARDS

        if (blocked := first_problem(AGENT_GUARDS, SEARCH_LAUNCH, cfg)) is not None:
            return refuse(blocked.text, blocked.fix, code=blocked.code)
    if worker is None:
        from datetime import UTC, datetime, timedelta

        try:
            if conflicts := isolation.sign_in_conflicts(cfg):
                return refuse("; ".join(conflicts.values()), "name one sign-in in the config")
            lines = isolation.sign_in_report(cfg, datetime.now(UTC) + timedelta(seconds=cfg.wall_s))[2]
        except isolation.SignInRefused:
            lines = ["conflicting sign-in dirs"]
        for line in lines:
            print(f"agent sign-in: {line}", file=sys.stderr)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.work_dir is not None:
        try:  # fresh, made here once: no earlier run's files and no link planted in its place
            cfg.workdir.mkdir(mode=0o700)
        except FileExistsError:  # `start_problem` says it too; the mkdir is what makes the check race-free
            return refuse(work_dir_refusal(cfg), "give a fresh work_dir per run (--set work_dir=...)")
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    data = cfg.resolved()
    # The pid probe left config loading: the level's own record fills the same key, in the same place, so the
    # resolved config a run writes is what it always was (None at none, the bwrap form at sandbox/strict).
    held = data["provenance"]
    data["provenance"] = {
        "commit": held["commit"],
        "plugins": held["plugins"],
        "pid": make_isolation(cfg).provenance()["pid_form"],
        "contamination_skipped": held["contamination_skipped"],
    }
    resolved = write_resolved(cfg.run_dir, data, data["provenance"])
    worker = worker if worker is not None else cfg.agent.backend.driver(cfg)
    try:
        run = backends.make_run(cfg, resolved, worker=worker)
    except Refused as why:
        return refuse(str(why))
    scoped = backends.run_fallbacks(cfg) if worker is None else contextlib.nullcontext()
    status = 130
    CLOSING.clear()  # a new run (tests run several in one process)
    STOPPING.clear()
    with _StopSignals():
        try:
            with progress.watch(cfg.run_dir), isolation.gpu_wall(cfg), scoped:
                run.run()
            status = 0
        except isolation.SignInRefused as why:  # the run's own refusal: the close below still runs
            status = refuse(" ".join(str(why).split()), code=why.code)
        except flow.TrialRunRefused as why:
            # the machine, not the input: the flow has already written FINAL.json with candidate 0
            status = refuse(" ".join(str(why).split()), code=ENVIRONMENT)
        except (isolation.WallRefused, flow.StartRefused) as why:
            status = refuse(" ".join(str(why).split()), start_fix(why))
        except KeyboardInterrupt:  # the flow, if it started, has picked the submission and written its files
            pass
        finally:
            CLOSING.set()
            try:
                if left := kill_every_group():  # the flow's close was cut short (a second Ctrl-C)
                    print(f"qda-agent: killed {left} job(s) the close had not stopped", file=sys.stderr)
            except OSError as e:  # never at the cost of `close`
                print(f"qda-agent: could not kill every job group: {e}", file=sys.stderr)
            result = close(cfg, run.epic)
    if status == 0 and STOPPING.is_set():
        status = 130  # a stop during the drain or close need not raise an exception
    if status == 130:
        picked = (cfg.run_dir / "FINAL.json").is_file()
        print(
            "qda-agent: stopped; "
            + ("the submission was picked and the run closed" if picked else "no submission was picked"),
            file=sys.stderr,
        )
    print(json.dumps(result, indent=2))
    if text := summary.of_run(cfg.run_dir):  # the checkpoint the run picked, and the KL it read for it
        print(text)
    if status == 0 and result.get("end_kind") == flow.EndKind.FINALIZE:
        # finalized early: the machine has to change before the same command works
        reason = str(result.get("finalize_reason") or "the run was finalized early")
        fix = _FINALIZE_DISK_FIX if reason.startswith("the disk") else _FINALIZE_API_FIX
        fresh = "--set run_dir=NEW_DIR" + (" --set work_dir=NEW_WORK_DIR" if cfg.work_dir is not None else "")
        return refuse(f"the run finalized early: {reason}", fix.replace("FRESH", fresh), code=ENVIRONMENT)
    if status == 0 and not flow.completed(result):
        if (error := result.get("fatal")) is None:
            return refuse(f"the run failed: {result.get('ended', 'unknown')}")
        key = flow.fatal_key(error, result.get("fatal_cause"))
        fix, code = _FATAL_FIX.get(key, (None, USAGE))
        fresh = "--set run_dir=NEW_DIR" + (" --set work_dir=NEW_WORK_DIR" if cfg.work_dir is not None else "")
        return refuse(
            f"the run failed: {flow.fatal_said(key, cfg.run_dir)}",
            fix and fix.replace("FRESH", fresh).replace("LEVEL", cfg.isolation.level),
            code=code,
        )
    return status


#: An early finalize's fix lines (FRESH: the new dirs a rerun needs): the agent API kept failing or was out of
#: reach, or the disk stayed below its floor. The exit status is 3.
_FINALIZE_API_FIX = (
    "check the agent's sign-in and endpoint (`qda doctor --search CONFIG.yaml`), then rerun with FRESH"
)
_FINALIZE_DISK_FIX = "free disk space (`qda doctor --search CONFIG.yaml` checks), then rerun with FRESH"

#: A fatal end's error class: its fix line (FRESH: the new dirs a rerun needs; LEVEL: the run's isolation.level)
#: and exit status. Others: 2.
_FATAL_FIX = {
    "HarnessRefused": (
        "sign in to the agent CLI again, or check the agent CLI's permissions, then rerun with FRESH",
        ENVIRONMENT,
    ),
    "HarnessNotInstalled": (
        "install the agent CLI (`qda doctor --search CONFIG.yaml` checks), then rerun with FRESH",
        ENVIRONMENT,
    ),
    "HarnessMissing": (
        "start the agent CLI by hand on this machine, fix what it says, then rerun with FRESH",
        ENVIRONMENT,
    ),
    "HarnessUnrecoverable": (
        "fix the cause the last `fatal` line of flow.jsonl names, then rerun with FRESH",
        ENVIRONMENT,
    ),
    "HarnessUnrecoverable/goal_flow": (
        "the endpoint does not take codex's goal flow: rerun with FRESH --set agent.goal=false",
        ENVIRONMENT,
    ),
    "HarnessSandboxed": (
        'rerun with FRESH --set "sandbox={name: none}" --set isolation.level=LEVEL (the agent\'s turns run '
        "without their sandbox, the jobs keep LEVEL)",
        ENVIRONMENT,
    ),
}


class _StopSignals:
    """SIGTERM and SIGHUP stop the run as Ctrl-C does: every job stopped, the submission picked, files written.

    Python's default for both ends the process at once, skipping all of that. The first one stops the agent
    phase; once the run closes
    (`host.CLOSING`: the deadline passed, Ctrl-C, or a first signal) more are ignored. The first two Ctrl-C
    presses during the close wait; the third abandons it. An inherited ignored SIGTERM or SIGHUP stays ignored. SIGINT
    does not: a shell starts a background command (`qda-agent run CONFIG &` in a script) with it ignored, and
    such a run would then outlive every SIGINT, its agent CLI with it. Only in the main thread (where signals
    are delivered).
    """

    SIGNALS = (signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self._old: dict[int, Any] = {}
        self._close_interrupts = 0

    def __enter__(self) -> _StopSignals:
        if threading.current_thread() is threading.main_thread():
            for s in self.SIGNALS:
                if signal.getsignal(s) is not signal.SIG_IGN:
                    self._old[s] = signal.signal(s, self._on)
            if signal.getsignal(signal.SIGINT) in (signal.default_int_handler, signal.SIG_IGN):
                self._old[signal.SIGINT] = signal.signal(signal.SIGINT, self._interrupt)
        return self

    def _interrupt(self, sig: int, frame: Any) -> None:
        """Ctrl-C stops the agent phase; the first two during the close wait, the third abandons it."""
        from qda_agent.host import CLOSING, STOPPING

        STOPPING.set()
        if CLOSING.is_set():
            self._close_interrupts += 1
            if self._close_interrupts < 3:
                return
            print("qda-agent: Ctrl-C: abandoning the close; pending readings may be lost", file=sys.stderr)
        signal.default_int_handler(sig, frame)

    def __exit__(self, *exc: object) -> None:
        for s, handler in self._old.items():
            signal.signal(s, handler)

    def _on(self, sig: int, frame: object) -> None:
        del frame
        name = signal.Signals(sig).name
        from qda_agent.host import CLOSING, STOPPING

        if CLOSING.is_set():
            print(
                f"qda-agent: {name}: already closing the run; it ends once every job is stopped",
                file=sys.stderr,
            )
            return
        STOPPING.set()
        CLOSING.set()
        print(f"qda-agent: {name}: stopping the run as for Ctrl-C", file=sys.stderr)
        # Forwarding to our handler would count SIGTERM/SIGHUP as a Ctrl-C during the close.
        if signal.getsignal(signal.SIGINT) != self._interrupt:
            signal.raise_signal(signal.SIGINT)


def close(cfg: RunConfig, epic: Path | None) -> dict[str, Any]:
    """Leave the run dir holding the whole run, and nothing of it elsewhere.

    Humanize writes its trace of the run (the epic) under its home (`~/.hmz/epics/`), where the agent wall
    binds it while the run is live. Here it moves into `run_dir/epic/`, `result.json` and the end report
    name it, and the run dir's sealed directories become writable again, so that `rm -rf` of the run dir
    works. Sealed files stay read-only.

    Args:
        cfg: The run config.
        epic: Humanize's epic of the run, or None when Humanize never opened one.

    Returns:
        The contents of `result.json` (the flow's summary, plus `epic`).
    """
    from qda_agent.judge.freeze import deletable

    kept = _move_epic(epic, cfg.run_dir / "epic")
    path = cfg.run_dir / "result.json"
    result: dict[str, Any] = (
        json.loads(path.read_text()) if path.is_file() else {"ended": "error: the flow never finished"}
    )
    result["epic"] = str(kept) if kept else None
    path.write_text(json.dumps(result, indent=2))
    with (cfg.run_dir / "END-REPORT.md").open("a") as f:
        f.write(f"- Humanize trace (every turn and tool call): `{result['epic']}`\n")
    deletable(cfg.run_dir)
    return result


def _move_epic(epic: Path | None, dst: Path) -> Path | None:
    """Move the epic to `dst` and drop its workspace dir if that is now empty; where it ended up."""
    if epic is None or not epic.is_dir():
        return None
    try:
        shutil.move(epic, dst)
    except OSError as e:  # left (whole or in part) where Humanize wrote it
        print(f"qda-agent: could not move the epic {epic} to {dst}: {e}", file=sys.stderr)
        return dst if dst.is_dir() and not epic.exists() else epic
    with contextlib.suppress(OSError):
        epic.parent.rmdir()  # Humanize's per-workspace dir, when no other epic is in it
    return dst


def search(args: argparse.Namespace) -> int:
    """`qda-agent search`: `qda doctor --search` on the config for its report, then `qda-agent run` on it.

    Doctor's exit status no longer gates the run. Every check the run refuses on is a guard that the
    run walks itself and doctor walks too (`qda_quant.guards`), and the contrast test holds both to one sentence,
    so where doctor's MISS is one the run refuses on, the run refuses with it (exit 2 or 3) before it creates
    anything. A MISS the run does not refuse on is doctor's report, and the run goes ahead.
    """
    from qda_quant import cli as qda

    config = args.config / "search.yaml" if args.config.is_dir() else args.config
    if not config.exists():  # no prepare made it: a directory to edit is not the fix
        out = args.config if args.config.suffix not in (".yaml", ".yml") else args.config.parent
        return refuse(
            f"{config}: no such file",
            f"run `qda-agent prepare --model M --out {out}` first; it makes {out}/search.yaml",
        )
    sets = [a for item in args.overrides for a in ("--set", item)]
    if qda.main(["doctor", "--search", str(config), *sets]):
        print(
            "qda-agent search: qda doctor reported a miss above; the run checks its own start and refuses "
            "if it cannot run",
            file=sys.stderr,
        )
    return main(["run", str(config), *sets])


#: The commands `qda-agent --help` lists and 0.1.x keeps stable: every one the user docs tell you to run.
#: The rest still run when named.
PUBLIC = ("prepare", "search", "status", "run", "gate", "cand0")


def main(argv: list[str] | None = None) -> int:
    """Entry point of `qda-agent`."""
    line_buffered_stdout()
    assert __doc__ is not None  # the module's docstring is the CLI's description
    ap = argparse.ArgumentParser(prog="qda-agent", description=__doc__.splitlines()[0].replace("`", ""))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run a search")
    p.add_argument("config", type=Path, help="the run YAML, or a run's resolved.yaml to repeat it")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one key of the config (the value is YAML); repeatable",
    )
    g = sub.add_parser("gate", help="run a search's start phase alone and print its W (no agent)")
    g.add_argument("config", type=Path, help="the run YAML the search will launch with")
    g.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    from qda_agent import cand0, evaluate, prepare

    cand0.add_parser(sub)
    prepare.add_parser(sub)
    s = sub.add_parser("search", help="check the machine (qda doctor --search), then run the search")
    s.add_argument("config", type=Path, help="a prepare directory (its search.yaml) or a run YAML")
    s.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    evaluate.add_parser(sub)
    s = sub.add_parser("status", help="print a run's progress now (the view `run` streams to stderr)")
    s.add_argument("run_dir", type=Path, help="a run's run directory, while it works or after it ends")
    show_only(sub, PUBLIC)
    args = ap.parse_args(argv)
    if args.cmd == "cand0":  # reads only what the build needs: the config's other inputs may not exist yet
        return cand0.run(args)
    if args.cmd == "prepare":
        return prepare.main(args)
    if args.cmd == "evaluate":
        return evaluate.main(args)
    if args.cmd == "search":
        return search(args)
    if args.cmd == "status":
        from qda_agent import progress

        lines = progress.status(args.run_dir)
        if lines is None:
            print(f"qda-agent: {args.run_dir} is not a run directory", file=sys.stderr)
            return 2
        print("\n".join(lines))
        return 0
    try:
        cfg = RunConfig.load(args.config, args.overrides)
    except ConfigError as e:  # nothing is made yet: name what to fix, no traceback
        return refuse(str(e), e.fix)
    if args.cmd == "gate":
        import asyncio

        from qda_agent.flow import StartRefused, gate

        try:
            out = asyncio.run(gate(cfg))
        except StartRefused as why:  # the run's own start refused it: one line, as `run` prints it
            return refuse(" ".join(str(why).split()), start_fix(why))
        print(f"W {out['w_s']:.1f} s (search start to the agent's first turn); {cfg.run_dir / 'gate.json'}")
        return 0
    return cfg.launcher.launch(cfg, lambda: execute(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
