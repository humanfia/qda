"""Run a job under a reaper: the outermost process of every job, at every isolation level.

A job's own end does not end its processes. A sandboxed job runs bubblewrap with a session of its own
(`--new-session`), so a signal to the job's process group reaches only bubblewrap, which does not pass it on;
where the sandbox shares the host's pid namespace (`pid: shared`) the job's grandchildren are reparented to init
when bubblewrap dies and outlive the job. A plain command that calls `setsid` or double-forks does the same.

So every job starts under this program, at the outermost position: a child subreaper (`prctl`) in a session of its
own. The kernel reparents the job's orphans to it rather than to init; when the job ends it sweeps `/proc` for the
job's session and for every descendant it holds, sends them SIGTERM, waits the grace, then SIGKILLs what is left.
Its parent signals this program (never the job's process group) and waits for it to exit.

The grace is the stopping caller's: `--grace 0` leaves the pace to `stop`, which SIGTERMs this program, waits its
own grace, and then sends `_KILL_NOW` for the SIGKILL of what is left. A signal to one process goes out only to the
process the scan saw (`_signal`): through a pid descriptor where the kernel has one, so a number it has handed to
another process cannot be signalled, and by start time elsewhere. The group signal is checked the same way, against
the leader's start time.

`turn_guard` in the wall is the same mechanism for one agent turn; the wall passes that program by value into the
sandbox, while this one runs outside every sandbox, so the file itself is the program. Both need the standard
library only.

Usage: ``python -I -S reaper.py [--grace S] [--pass-fds N[,N...]] -- COMMAND...``

`LocalBackend` starts every job this way; `judge.build_receipt` and `probe` use `run` for the commands they start
themselves, so a parent that is interrupted does not leave the engine it started behind.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    from .api import Handle, Launch

__all__ = ["DEFAULT_GRACE_S", "REAPER", "Reaper", "program", "run", "spawn", "stop"]

#: This file, run by an interpreter with the standard library only.
REAPER = Path(__file__).resolve()
DEFAULT_GRACE_S = 10.0

#: `prctl(PR_SET_PDEATHSIG, ...)`: the signal a process gets when its parent dies.
_PR_SET_PDEATHSIG = 1

#: `prctl(PR_SET_CHILD_SUBREAPER, 1)`: the job's orphans are reparented to this program, not to init.
_PR_SET_CHILD_SUBREAPER = 36

#: `/proc/<pid>/stat` after the command name (which may hold spaces and parentheses): state, ppid, process
#: group, session, and the start time (field 22 of the file) that names the process together with its pid.
_STATE = 0
_PPID = 1
_PGRP = 2
_AFTER_COMM = 3
_START = 19

#: The signals that mean "stop this job": what the parent's `stop` sends.
_STOP = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

#: The signal the parent's `stop` sends once its grace is over: SIGKILL what the stop left, now.
_KILL_NOW = signal.SIGUSR1

#: How long a tree gets when the caller named no grace of its own (`--grace 0`): the fallback if that caller
#: disappears before it sends `_KILL_NOW`.
_ORPHAN_GRACE = 30.0

#: Set once a stop signal is being acted on: a second one must not cut the cleanup short.
_stopping = False

#: Set by `_KILL_NOW`: the grace is over, kill what is left at once.
_kill_now = False


# -------------------------------------------------------------------------------------------- the parent's side
def program(
    command: Sequence[str], *, grace_s: float = DEFAULT_GRACE_S, pass_fds: Iterable[int] = ()
) -> list[str]:
    """The command line that runs `command` under a reaper.

    `pass_fds` are descriptors the caller has open and the job needs (bwrap's private settings, a pinned
    source): the reaper keeps them open for the job and closes nothing else.
    """
    argv = [sys.executable, "-I", "-S", str(REAPER), "--grace", f"{grace_s:g}"]
    if fds := [str(fd) for fd in pass_fds]:
        argv += ["--pass-fds", ",".join(fds)]
    return [*argv, "--", *command]


def spawn(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    out: Any = None,
    pass_fds: Iterable[int] = (),
    grace_s: float = DEFAULT_GRACE_S,
) -> subprocess.Popen:
    """Start `command` under a reaper, the reaper in a session of its own.

    `out` is the file the job's output goes to (None: the caller's own); `stop` then ends the job and its tree.
    """
    fds = tuple(pass_fds)  # once: the command line and the child must name the same descriptors
    return subprocess.Popen(
        program(command, grace_s=grace_s, pass_fds=fds),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT if out is not None else None,
        start_new_session=True,
        pass_fds=fds,
    )


def stop(proc: subprocess.Popen, grace_s: float = DEFAULT_GRACE_S) -> int | None:
    """End the job's whole tree; the status the reaper left (negative: a signal), once it is gone.

    The reaper does the killing: SIGTERM to it sends SIGTERM to the tree, and `grace_s` is how long that tree
    gets to go by itself -- the caller's number, not the reaper's. When it is over, `_KILL_NOW` tells the reaper
    to SIGKILL what is left, and its two-second kill pass and exit follow. A reaper still there after that is
    wedged; its process group is killed as the last resort.
    """
    if proc.poll() is not None:
        return proc.returncode
    with contextlib.suppress(ProcessLookupError):
        os.kill(proc.pid, signal.SIGTERM)
    if _wait_for(proc, grace_s):
        return proc.returncode
    with contextlib.suppress(ProcessLookupError):
        os.kill(proc.pid, _KILL_NOW)
    if _wait_for(proc, 3.0):
        return proc.returncode
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)  # the reaper leads the group it alone is in
    return proc.wait()


def _wait_for(proc: subprocess.Popen, seconds: float) -> bool:
    """Wait up to `seconds` for `proc` to end; whether it did."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if proc.poll() is not None:
            return True
        time.sleep(0.02)
    return proc.poll() is not None


def run(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    out: Any = None,
    pass_fds: Iterable[int] = (),
    grace_s: float = DEFAULT_GRACE_S,
) -> int:
    """Run `command` under a reaper to its end, and leave none of its processes behind; its exit code."""
    proc = spawn(command, cwd=cwd, env=env, out=out, pass_fds=pass_fds, grace_s=grace_s)
    try:
        return proc.wait()
    finally:
        stop(proc, grace_s)


class Reaper:
    """Start, poll and stop jobs under the reaper: the `KillDomain` of `isolation.api`.

    The grace belongs to `stop`, not to the launch: a job is started with none of its own (`--grace 0`), so the
    caller that stops it decides how long it gets, and the reaper's `_ORPHAN_GRACE` is only the fallback for a
    caller that vanishes before it can ask.
    """

    def spawn(self, launch: Launch, *, cwd: Path, log: Path) -> Handle:
        """Start a launch (`.argv`, `.env`, `.pass_fds`), its output appended to `log`."""
        env = launch.env
        with log.open("ab") as out:
            return spawn(
                launch.argv,
                cwd=cwd,
                env=None if env is None else dict(env),  # {} is an empty environment; None inherits this one
                out=out,
                pass_fds=launch.pass_fds,
                grace_s=0.0,
            )

    def stop(self, handle: Handle, grace_s: float) -> None:
        """End the job and its tree, giving it `grace_s` seconds to go by itself."""
        stop(cast(subprocess.Popen, handle), grace_s)

    def poll(self, handle: Handle) -> int | None:
        """The job's status once it ended, else None."""
        return cast(subprocess.Popen, handle).poll()


# -------------------------------------------------------------------------------------------- the program itself
class _Stopped(Exception):
    """Raised by the signal handler so the wait ends and the tree is swept before exit."""

    signum: int

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _own_session() -> None:
    """Leave the caller's process group, and hear the caller's death as SIGTERM.

    This program is started as a session leader, so `setsid` is already done and fails harmlessly. Asking for
    SIGTERM (rather than the SIGKILL bwrap's child gets) when the parent dies is what lets a crashed or killed
    caller still leave nothing behind.
    """
    with contextlib.suppress(OSError):
        os.setsid()
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)


def _die_with_parent() -> None:
    """In the job's own process: die with this program, so the last resort still ends the job."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)


def _adopt_orphans() -> None:
    """Have the kernel reparent the job's orphans to this program rather than to init.

    A process the job starts that double-forks is reparented to the nearest living subreaper of its pid
    namespace; without this that is init (or bubblewrap), and it would be neither in the job's session nor this
    program's descendant when the job ends. As a subreaper this program keeps it. The setting is not inherited
    across `fork`, so the job and its children do not become subreapers themselves.
    """
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)


def _on_stop(signum: int, _frame: object) -> None:
    """Stop the job now; later stop signals are ignored so the cleanup can finish."""
    global _stopping  # noqa: PLW0603 (one flag for the one handler)
    if not _stopping:
        _stopping = True
        raise _Stopped(signum)


def _on_kill_now(_signum: int, _frame: object) -> None:
    """The caller's grace is over: the sweep stops waiting for the tree to go and kills what is left."""
    global _kill_now  # noqa: PLW0603 (one flag for the one handler)
    _kill_now = True


def _fields(pid: int) -> list[bytes] | None:
    """`/proc/<pid>/stat` after the command name, split; None if it is gone (or not ours to read)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:  # noqa: PTH123 (a pid, not a path)
            return f.read().rsplit(b")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _stat(pid: int) -> tuple[bytes, int, int] | None:
    """The state, parent pid and session id of `pid`, or None if it is gone (or not ours to read)."""
    fields = _fields(pid)
    if fields is None:
        return None
    return fields[_STATE], int(fields[_PPID]), int(fields[_AFTER_COMM])


def _started(pid: int) -> int | None:
    """When `pid` started (`/proc/<pid>/stat` field 22), or None if no such process is there.

    A pid is a name, not a process: the kernel hands it to a new process once this one is reaped. The start
    time is not handed on with it, so the pair (pid, start time) names one process for as long as it lives.
    """
    fields = _fields(pid)
    return None if fields is None else int(fields[_START])


def _pgrp(pid: int) -> int | None:
    """The process group of `pid`, or None if it is gone (or not ours to read)."""
    fields = _fields(pid)
    return None if fields is None else int(fields[_PGRP])


class _Row(NamedTuple):
    """One observation of a process: what `/proc/<pid>/stat` said in a single read."""

    state: bytes
    ppid: int
    session: int
    started: int


def _row(pid: int) -> _Row | None:
    """The state, parent, session and start time of `pid` from one read; None if it is gone."""
    fields = _fields(pid)
    if fields is None:
        return None
    return _Row(fields[_STATE], int(fields[_PPID]), int(fields[_AFTER_COMM]), int(fields[_START]))


def _members(session: int, guard: int) -> dict[int, int]:
    """Every live process the job holds, by pid, with the start time of the same observation.

    A process that started a session of its own has left `session`; a process that double-forked has left the
    job's tree and come back as this program's child (it is a subreaper). Both are found here. A zombie is not
    alive (it holds no file and writes nowhere); this program's own children are reaped by init once it exits.

    Each pid's membership and its start time come from one `/proc/<pid>/stat` read (`_row`): a member's identity
    is never paired with the start time of a process that took the number after it was reaped.
    """
    table: dict[int, _Row] = {}
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if (row := _row(pid)) is None:
            continue
        table[pid] = row
        children.setdefault(row.ppid, []).append(pid)
    found = {
        pid for pid, row in table.items() if pid != guard and row.session == session and row.state != b"Z"
    }
    stack = list(children.get(guard, ()))
    while stack:
        pid = stack.pop()
        if pid == guard:
            continue
        if table[pid].state != b"Z":
            found.add(pid)
        stack.extend(children.get(pid, ()))
    found.discard(guard)
    return {pid: table[pid].started for pid in found}


def _signal(pid: int, started: int, sig: int) -> None:
    """Signal `pid` only if it is still the process this scan saw.

    A process can end and have its number handed to another process between the scan and the signal. Where the
    kernel has pid descriptors (`os.pidfd_open`), one is opened first: from then on it names that one process,
    and the signal goes through it, so a number reused under it is never the one signalled. Elsewhere the start
    time is read again just before the signal, and a pid that names nothing, or a newer process, is skipped.
    """
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if opener is not None and sender is not None:
        try:
            fd = opener(pid)
        except OSError:
            return
        try:
            if _started(pid) == started:
                sender(fd, sig)
        except OSError:
            pass
        finally:
            os.close(fd)
        return
    if _started(pid) != started:
        return
    with contextlib.suppress(OSError):
        os.kill(pid, sig)


def _kill(session: int, guard: int, grace: float, leader: int | None = None) -> None:
    """Terminate the job, escalating to SIGKILL, and wait for its processes to go.

    Signalled by id as well as by pid: `session` is the job's leader, which is its process group's leader, so
    `killpg` reaches the job and every process it started that kept its group -- including one whose `/proc`
    entry this scan missed. The group signal goes out only while the leader's start time is still `leader` (the
    number of a group is the leader's pid, and that can be reused too). A process in that group is not signalled
    again by pid (one signal per process: a handler that forks must not run twice), so the per-pid signals are
    for the processes that left the group, each to a process this scan just saw (`_signal`). `grace` is how long
    the tree gets between SIGTERM and SIGKILL; 0 belongs to a caller that sets the pace itself with `_KILL_NOW`,
    and leaves `_ORPHAN_GRACE` as the fallback should that caller disappear.
    """
    for sig, wait in ((signal.SIGTERM, grace or _ORPHAN_GRACE), (signal.SIGKILL, 2.0)):
        members = _members(session, guard)
        if not members:
            return
        grouped = bool(session) and leader is not None and _started(session) == leader
        if grouped:
            with contextlib.suppress(OSError):
                os.killpg(session, sig)
        for pid, started in members.items():
            if grouped and _pgrp(pid) == session:
                continue  # the group signal already reached it
            _signal(pid, started, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not _kill_now and _members(session, guard):
            time.sleep(0.02)


def _die_as(signum: int) -> None:
    """Die as the job did, so the caller reads a signal and not an exit code.

    A process killed by a signal has no exit status of its own; reported as the exit code 128 + n, a job the
    operator killed (SIGTERM) or the kernel's OOM killer ended (SIGKILL) would look like a plain failure.
    Restoring the default action and re-raising the signal keeps the job's death visible to the caller.
    """
    with contextlib.suppress(OSError, ValueError):
        signal.signal(signum, signal.SIG_DFL)
    with contextlib.suppress(OSError):
        os.kill(os.getpid(), signum)


def _args(argv: list[str]) -> tuple[float, tuple[int, ...], list[str]]:
    """The grace, the descriptors to keep for the job, and the command, from this program's argv."""
    grace, fds = DEFAULT_GRACE_S, ()
    while argv and argv[0].startswith("--") and argv[0] != "--":
        flag = argv.pop(0)
        value = argv.pop(0) if argv else ""
        if flag == "--grace" and value:
            grace = float(value)
        elif flag == "--pass-fds":
            fds = tuple(int(fd) for fd in value.split(",") if fd)
        else:
            raise SystemExit(f"reaper: {flag!r} takes a value, or is not an option")
    if argv and argv[0] == "--":
        argv = argv[1:]
    return grace, fds, argv


def _child_setup(old: set[int]) -> None:
    """In the job, before it execs: the caller's signal mask back, and death with this program."""
    signal.pthread_sigmask(signal.SIG_SETMASK, old)
    _die_with_parent()


def main(argv: list[str]) -> int:
    """Run `argv` as a session leader, return its exit status, and leave none of its processes behind."""
    grace, pass_fds, command = _args(list(argv))
    if not command:
        print("reaper: no command", file=sys.stderr)
        return 2
    _own_session()
    _adopt_orphans()
    # Blocked while the job is started and the handlers are put in place: a stop signal delivered in between
    # would otherwise kill this program before it had a session to clean. The job gets the mask back before it
    # execs, or it would inherit the block and a later SIGTERM could never end it.
    child: subprocess.Popen[bytes] | None = None
    session = 0
    leader: int | None = None
    old = signal.pthread_sigmask(signal.SIG_BLOCK, (*_STOP, _KILL_NOW))
    try:
        try:
            child = subprocess.Popen(  # noqa: S603 (the caller built argv)
                command,
                start_new_session=True,
                pass_fds=pass_fds,
                preexec_fn=partial(_child_setup, old),
            )
        except OSError as exc:  # no such command, no permission: report it as a shell does
            print(f"reaper: {command[0]}: {exc.strerror or exc}", file=sys.stderr)
        else:
            session = child.pid  # a session leader's session id is its pid
            leader = _started(session)  # its identity, for the group signal: that number is reused too
            for sig in _STOP:
                signal.signal(sig, _on_stop)
            signal.signal(_KILL_NOW, _on_kill_now)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        if child is None:
            return 127
        # Waited for, not reaped: the job's own process stays a zombie, so its pid -- the job's process group id
        # -- is not handed to another process while the sweep below sends the group its signals. It is reaped
        # only after the last of them.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    except _Stopped as stopped:
        _kill(session, os.getpid(), grace, leader)
        if child is not None:
            with contextlib.suppress(OSError):
                child.wait()
        return 128 + stopped.signum
    global _stopping  # noqa: PLW0603 -- past the wait the job has ended; a stop now must not cut cleanup
    _stopping = True
    _kill(session, os.getpid(), grace, leader)
    child.wait()  # now reap: the last group signal is behind us
    rc = child.returncode if child.returncode is not None else 1
    if rc < 0:  # the job was killed by a signal: die by it, so the caller's poll() reports a signal too
        _die_as(-rc)
        return 128 - rc
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
