"""Launch, monitor and stop jobs on the cards assigned by the scheduler.

Backends return process handles; the queue owns scheduling and exclusive card allocation.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .clock import Clock

if TYPE_CHECKING:
    from qda_agent.isolation.api import Isolation

if TYPE_CHECKING:
    from types import ModuleType


class Backend(Protocol):
    """Define how to start, monitor and stop a job's process."""

    #: Whether the job's processes see exactly the `cards` given (False: the backend leases its own).
    exact_cards: bool

    def launch(
        self,
        argv: list[str],
        cwd: str,
        cards: list[int],
        log: Path,
        name: str,
        env: dict[str, str] | None = None,
        pass_fds: Iterable[int] = (),
    ) -> object:
        """Start `argv` in `cwd` on `cards`, its output to `log`, in `env` (None: the host's); return a handle.

        `pass_fds` are descriptors of the calling process (the sandbox's opened sources) that must stay open
        in the started process; the caller closes them once `launch` returns.
        """
        ...

    def poll(self, handle: object) -> int | None:
        """Return the exit code once the process ended, else None."""
        ...

    def stop(self, handle: object) -> None:
        """End the process (and everything it started)."""
        ...


@dataclass
class FakeRun:
    """Describe a fake job's duration, result and launch effect for scheduler tests.

    Attributes:
        seconds: Fake run time.
        rc: Exit code.
        output: Text written to the job's log.
        effect: Called with the job's argv at launch (e.g. to write the result file a command would).
    """

    seconds: float
    rc: int = 0
    output: str = ""
    effect: Callable[[list[str]], None] | None = None


def _expanded_argv(argv: list[str]) -> list[str]:
    """`argv` with bwrap's `--args` settings read back from their descriptor, as the real sandbox would see them.

    A sandboxed job's options (mounts, environment) travel in a private descriptor so no secret is on a
    readable command line; the fake backend records the effective command the sandbox would run, so tests can
    read it the way they read any other argv. The descriptor is still open when the launch is recorded.
    """
    if "--args" not in argv:
        return list(argv)
    i = argv.index("--args")
    try:
        fd = int(argv[i + 1])
        settings = b""
        while chunk := os.pread(fd, 1 << 16, len(settings)):
            settings += chunk
    except (OSError, ValueError):
        return list(argv)
    options = [a.decode(errors="replace") for a in settings.rstrip(b"\0").split(b"\0")]
    return [argv[0], *options, *argv[i + 2 :]]


@dataclass
class _FakeHandle:
    name: str
    cards: list[int]
    start: float
    run: FakeRun
    stopped: bool = False


@dataclass
class FakeBackend:
    """Simulate GPU jobs using an injectable clock for scheduler tests.

    Args:
        clock: The clock that decides when a job ends.
        runs: What each job name does; a name not listed takes `default`.
        default: The run of an unlisted job.

    Attributes:
        launched: Every launch, in order, as (name, cards, start time).
        argvs: The argv of each launch, by name.
        envs: The environment of each launch, by name.
    """

    clock: Clock
    runs: dict[str, FakeRun] = field(default_factory=dict)
    default: FakeRun = field(default_factory=lambda: FakeRun(60.0))
    launched: list[tuple[str, list[int], float]] = field(default_factory=list)
    argvs: dict[str, list[str]] = field(default_factory=dict)
    envs: dict[str, dict[str, str] | None] = field(default_factory=dict)
    exact_cards: bool = True

    def launch(  # noqa: ARG002
        self,
        argv: list[str],
        cwd: str,  # noqa: ARG002
        cards: list[int],
        log: Path,
        name: str,
        env: dict[str, str] | None = None,
        pass_fds: Iterable[int] = (),  # noqa: ARG002
    ) -> _FakeHandle:
        """Record the launch and write the fake output to `log`."""
        self.argvs[name], self.envs[name] = _expanded_argv(argv), env
        run = self.runs.get(name, self.default)
        log.write_text(run.output)
        if run.effect is not None:
            run.effect(argv)
        self.launched.append((name, list(cards), self.clock.now()))
        return _FakeHandle(name, list(cards), self.clock.now(), run)

    def poll(self, handle: object) -> int | None:
        """The run's rc once its time has passed; -15 once stopped."""
        assert isinstance(handle, _FakeHandle)
        if handle.stopped:
            return -signal.SIGTERM
        return handle.run.rc if self.clock.now() >= handle.start + handle.run.seconds else None

    def stop(self, handle: object) -> None:
        """Mark the fake process stopped."""
        assert isinstance(handle, _FakeHandle)
        handle.stopped = True


@dataclass(eq=False)
class _Proc:
    popen: subprocess.Popen
    grace_s: float
    pgid_file: Path | None = None


#: Every job this process started under a reaper and has not seen end (`kill_every_group`).
_LIVE: set[_Proc] = set()
_LIVE_LOCK = threading.Lock()


def _reaper() -> ModuleType:
    """The reaper module (`isolation.reaper`), loaded on first use.

    Loaded here, not at the top: `qda_agent.isolation` imports this package's `config` at load, and the
    scheduler config is loaded from `qda_agent.config`, so a top-level import would be a cycle.
    """
    from ..isolation import reaper

    return reaper


def _forget(h: _Proc) -> None:
    """The job has ended: drop it from `_LIVE` and its `pgid` file (a stale one could name a reused pid)."""
    with _LIVE_LOCK:
        _LIVE.discard(h)
    if h.pgid_file is not None:
        with contextlib.suppress(OSError):
            h.pgid_file.unlink()


def kill_every_group() -> int:
    """SIGKILL the process group of every job still running; how many were killed.

    The last line of defence when a run's close was cut short (a second Ctrl-C): no reaper outlives the run.
    A reaper dies with a hard kill and cannot sweep, so its job dies with it only as far as the kernel takes it
    (the job's own process asked for SIGKILL when its parent died); a process that left the job's tree may
    survive, which is why `stop` is the way to end a job.
    """
    with _LIVE_LOCK:
        left = list(_LIVE)
        _LIVE.clear()
    for h in left:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(h.popen.pid, signal.SIGKILL)
        h.popen.wait()
        _forget(h)
    return len(left)


class LocalBackend:
    """Run each job on this host under a reaper, with its assigned cards.

    The reaper (`isolation.reaper`) is the outer process of every job: a subreaper in a session of its own,
    so `stop` reaches the job's whole tree even where bubblewrap gave the job a session of its own.

    Args:
        kill_grace_s: Seconds between SIGTERM and SIGKILL in `stop`.
    """

    exact_cards = True

    def __init__(self, kill_grace_s: float = 10.0, isolation: Isolation | None = None) -> None:
        self.kill_grace_s = kill_grace_s
        self.iso = isolation

    def command(self, argv: list[str], cards: list[int], name: str) -> list[str]:  # noqa: ARG002
        """The argv actually started (a wrapper backend prefixes its tool)."""
        return argv

    def launch(
        self,
        argv: list[str],
        cwd: str,
        cards: list[int],
        log: Path,
        name: str,
        env: dict[str, str] | None = None,
        pass_fds: Iterable[int] = (),
    ) -> _Proc:
        """Start the command under a reaper so `stop` can end its whole tree; its pgid is in `pgid` by `log`.

        The scheduler names the cards in `env` (by UUID) and that is kept; a caller whose `env` has no
        `CUDA_VISIBLE_DEVICES` (or that gives none) gets the card numbers, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
        so that they mean what the scheduler meant.
        """
        if env is None or "CUDA_VISIBLE_DEVICES" not in env:  # (the host's own value names other cards)
            env = {
                **(os.environ if env is None else env),
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards)),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        with log.open("ab") as out:
            p = _reaper().spawn(
                self.command(argv, cards, name),
                cwd=cwd,
                env=env,
                out=out,
                pass_fds=tuple(pass_fds),
                grace_s=self.kill_grace_s,
            )
        # for the user who must clean up after a host killed outright (see `qda_agent.flow`): the reaper's pid,
        # and a SIGTERM to it ends the job's whole tree, as `stop` does
        pgid: Path | None = log.parent / "pgid"
        try:
            pgid.write_text(f"{p.pid}\n")
        except OSError:
            pgid = None
        h = _Proc(p, self.kill_grace_s, pgid)
        with _LIVE_LOCK:
            _LIVE.add(h)
        return h

    def poll(self, handle: object) -> int | None:
        """The process's exit code, or None while it runs."""
        assert isinstance(handle, _Proc)
        rc = handle.popen.poll()
        if rc is not None:
            _forget(handle)
        return rc

    def stop(self, handle: object) -> None:
        """SIGTERM the job's reaper, which ends the whole tree (TERM, the grace, then KILL), and wait for it.

        The grace is the tree's, not the outer process's: a wrapper that exits at once still leaves its job
        `kill_grace_s` to shut down.
        """
        assert isinstance(handle, _Proc)
        _reaper().stop(handle.popen, handle.grace_s)
        _forget(handle)
