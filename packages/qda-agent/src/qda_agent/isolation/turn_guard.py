"""Run one turn's command in a session of its own and kill what it leaves behind when the turn ends.

The wall runs every turn through this program (it is passed by value, like `live_state`, so the sandbox needs
no package on its path). With a private pid namespace the namespace's death killed every process the turn
started; with `pid: shared` it does not, and a process the agent leaves running outlives its turn -- it could
keep writing into the workdir after the turn was judged. So the command runs as the leader of a fresh session;
when it exits it terminates what remains of the session (SIGTERM, then SIGKILL) and waits for it.

A cancelled turn must leave nothing behind either. The run's stop sequence signals the process group it
started (bubblewrap's) and then the group again with SIGKILL, not the session this program made, so a SIGTERM
would end this program before its cleanup ran and the turn would survive it. Bubblewrap also has its child
killed outright (`PR_SET_PDEATHSIG` = SIGKILL) when it dies, which would take this program before it could
act. So at start this program leaves bubblewrap's process group (`setsid`) and asks the kernel for SIGTERM
rather than SIGKILL when its parent dies; TERM, INT and HUP are caught: the session goes as it does on a
normal end (TERM by session id, waited out, then KILL), and only then does this program exit. The turn and
its children are signalled by the session id, which is the turn's own pid, so the first TERM reaches all of
them, and the group's SIGKILL no longer reaches this program to cut that short.

A process that calls `setsid` itself leaves the session, and one that double-forks is reparented out of the
turn. The session alone cannot hold either. This program is therefore a child subreaper (`prctl`): the
kernel reparents the turn's orphans to it rather than to init, and at the turn's end its sweep takes the
session's members and every descendant it can reach through `/proc`, so a child that left the session is
still a descendant and goes with it. A per-turn cgroup would hold it too, but this account has none to use:
the node's Slurm cgroup (`.../job_<id>/step_<n>/user/task_<t>`) is root-owned and not writable by the run,
and there is no user systemd bus for `systemd-run --user`.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: `prctl(PR_SET_PDEATHSIG, ...)`: the signal this process gets when its parent dies.
_PR_SET_PDEATHSIG = 1

#: `prctl(PR_SET_CHILD_SUBREAPER, 1)`: the turn's orphans are reparented to this process, not to init.
_PR_SET_CHILD_SUBREAPER = 36

#: `/proc/<pid>/stat` after the command name (which may hold spaces and parentheses): state, ppid, pgrp, session.
_STATE = 0
_PPID = 1
_AFTER_COMM = 3

#: The signals that mean "stop this turn": what the run's stop sequence sends to bubblewrap's process group.
_STOP = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

#: Set once a stop signal is being acted on: a second one must not cut the cleanup short.
_stopping = False


class _Stopped(Exception):
    """Raised by the signal handler so the wait ends and the session is cleaned up before exit."""

    signum: int

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _leave_the_group() -> None:
    """Stop being killable with bubblewrap's process group, and hear its death as SIGTERM.

    Bubblewrap's child dies with it by `PR_SET_PDEATHSIG` = SIGKILL, and the run's stop sequence sends the
    group a TERM and then a KILL. Both would end this program before it could end the turn. A session of its
    own takes this program out of the group, and asking for a SIGTERM (instead of the SIGKILL) when the parent
    dies tells it to clean up rather than to vanish. Not every platform has `prctl`; without it the group
    signals are all there is.
    """
    with contextlib.suppress(OSError):
        os.setsid()
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)


def _on_stop(signum: int, _frame: object) -> None:
    """Stop the turn now; later stop signals are ignored so the cleanup can finish."""
    global _stopping  # noqa: PLW0603 (one flag for the one handler)
    if not _stopping:
        _stopping = True
        raise _Stopped(signum)


def _adopt_orphans() -> None:
    """Have the kernel reparent the turn's orphans to this program rather than to init.

    A process the turn starts that double-forks is reparented to the nearest living subreaper of its pid
    namespace; without this that is init (or bubblewrap), and the process would be neither in the session nor
    this program's descendant when the turn ends. As a subreaper this program keeps it. The setting is not
    inherited across `fork`, so the turn and its own children do not become subreapers themselves. Not every
    platform has `prctl`; without it the session and the ancestry that stays under it are all there is.
    """
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)


#: The one `/proc` scan of a process table. `isolation.allocation` (the dedicated-node check) imports
#: `rows`/`descendants` too: it is not a second walker. This file is passed by value into the sandbox
#: (`python -I -S -c`), so it may not import the package that holds the allocation check; the shared code
#: has to live here.
def rows(proc: Path = Path("/proc")) -> dict[int, tuple[bytes, int, int]]:
    """Every process `proc` lists, as `{pid: (state, parent pid, session id)}`.

    Read from `/proc/<pid>/stat`, after the command name (which may hold spaces and parentheses), so the
    fields are state, ppid, pgrp, session. A process whose `stat` is gone or unreadable but whose `status` is
    readable is kept as well, with `status`'s state and parent and no session (-1): the allocation check used
    to read `status` alone, so the union of both readers must not lose a parent link either of them saw.
    """
    table: dict[int, tuple[bytes, int, int]] = {}
    for entry in Path(proc).iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            fields = (entry / "stat").read_bytes().rsplit(b")", 1)[1].split()
            table[pid] = (fields[_STATE], int(fields[_PPID]), int(fields[_AFTER_COMM]))
            continue
        except (OSError, IndexError):
            pass
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        state, ppid = b"Z", None  # no State line: treated as not alive, like a zombie
        for line in status.splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()[:1].encode() or b"?"
            elif line.startswith("PPid:"):
                ppid = int(line.split()[1])
        if ppid is not None:
            table[pid] = (state, ppid, -1)
    return table


def descendants(root: int, parents: dict[int, int]) -> set[int]:
    """Every process whose parent links lead back to `root` (not `root` itself), from `{pid: ppid}`."""
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    found: set[int] = set()
    stack = list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        if pid == root:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


def _members(session: int, guard: int, proc: Path = Path("/proc")) -> set[int]:
    """Every live process the turn holds: the session's, and every descendant of this program.

    A process that started a session of its own has left `session`; a process that double-forked has left the
    turn's tree and come back as this program's child (it is a subreaper). Both are found here. A zombie is not
    alive (it holds no file and writes nowhere); this program's own children are reaped by init once it exits.
    """
    table = rows(proc)
    parents = {pid: ppid for pid, (_state, ppid, _session) in table.items()}
    found = {
        pid
        for pid, (state, _ppid, sess) in table.items()
        if pid != guard and sess == session and state != b"Z"
    }
    found |= {pid for pid in descendants(guard, parents) if table[pid][_STATE] != b"Z"}
    found.discard(guard)
    return found


def _kill(session: int, guard: int) -> None:
    """Terminate the turn, escalating to SIGKILL, and wait for its processes to go.

    Signalled by id as well as by pid: `session` is the leader's pid, and the leader is its process group's
    leader, so `killpg` reaches the turn and every process it started that kept its group -- including one
    whose `/proc` entry this scan missed. Processes that changed group but stayed in the session, and the ones
    that left it but remain descendants of this program, get the per-pid signal.
    """
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        pids = _members(session, guard)
        if not pids:
            return
        if session:  # never `killpg(0, ...)`: that is this program's own group
            with contextlib.suppress(OSError):
                os.killpg(session, sig)
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and _members(session, guard):
            time.sleep(0.02)


def main() -> int:
    """Run `argv` as a session leader, return its exit status, and leave none of its processes behind."""
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("turn: no command", file=sys.stderr)
        return 2
    _leave_the_group()
    _adopt_orphans()
    # Blocked while the turn is started and the handlers are put in place: a stop signal delivered in between
    # would otherwise kill this program before it had a session to clean. The child gets the mask back before
    # it execs (`preexec_fn`), or it would inherit the block and a later SIGTERM could never end it.
    child: subprocess.Popen[bytes] | None = None
    session = 0
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP)
    try:
        try:
            child = subprocess.Popen(  # noqa: S603 (the wall built argv)
                argv,
                start_new_session=True,
                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old),
            )
            session = child.pid  # a session leader's session id is its pid
            for sig in _STOP:
                signal.signal(sig, _on_stop)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        child.wait()
    except _Stopped as stop:
        _kill(session, os.getpid())
        if child is not None:
            with contextlib.suppress(OSError):
                child.wait()
        return 128 + stop.signum
    global _stopping  # noqa: PLW0603 -- past the wait the turn has ended; a stop now must not cut cleanup
    _stopping = True
    _kill(session, os.getpid())
    return child.returncode if child.returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
