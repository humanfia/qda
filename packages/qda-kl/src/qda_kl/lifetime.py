"""Every qda-kl command owns the engine processes it starts, and leaves none of them behind.

A vLLM engine is a tree of processes: ``vllm.LLM`` starts an engine core (``VLLM::EngineCore``), which starts one
worker per card above one. The engine core holds the card's memory. Nothing in the command stops it: vLLM shuts
it down from a finalizer of the ``LLM`` object, and Python's default for SIGTERM and SIGHUP ends the process at
once, running no ``finally``, no finalizer and no ``atexit``. A ``kill`` of ``qda-kl ref`` left the engine core
running as an orphan of init, holding 70 GB of its card until someone found it. (Ctrl-C in a terminal reached
the engine too, because the terminal signals the whole foreground group; a ``kill`` reaches one process.)

:func:`owned` is the one owner, entered by every command that opens an engine (:func:`command`), and it makes
the engine's lifetime the command's:

* SIGTERM, SIGHUP and SIGINT raise :class:`Stopped` in the main thread, so the command unwinds like any error.
  A second signal while the command unwinds raises again (a ``finally`` that talks to a wedged engine must not
  hold the stop); once the sweep runs, one more only cuts its grace short.
* The command's process is a child subreaper (``PR_SET_CHILD_SUBREAPER``) while it owns: a process whose
  parent dies (an engine core whose launcher ended, a worker that double-forked) is reparented to it, not to
  init, so it is still found below.
* On the way out -- a return, an exception or a stop -- the sweep finds every live descendant that started
  while the owner held (``/proc``: each one's children, the subreaper's orphans among them), sends SIGTERM to
  each process group one of them leads and to each of them by pid, waits up to the grace, then sends SIGKILL to
  what is left. Only descendants: a process is named by its pid and start time, read in one observation, and a
  signal goes out only while that pair still holds. A child that was already running when the owner started is
  not this command's, and neither is its tree.
* A child forked from the main thread while the owner holds gets ``PR_SET_PDEATHSIG`` = SIGKILL (an
  ``os.register_at_fork`` hook; vLLM's default start method forks, unless CUDA is already up in the parent,
  which a qda-kl command does not do): when the command dies by a signal no handler sees (SIGKILL, the OOM
  killer), the kernel kills the engine core, and its workers follow it by vLLM's own death pipe.
  A child started by ``exec`` (``subprocess``, multiprocessing's spawn) runs no fork hook, so a command killed
  with SIGKILL can still leave it; every other ending is the sweep's.

Linux only (``/proc``, ``prctl``); elsewhere the signals still unwind the command, and the sweep finds nothing.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from qda_quant.exit import INTERRUPTED

__all__ = ["DEFAULT_GRACE_S", "Stopped", "command", "descendants", "owned"]

#: How long the engine's processes get to go by themselves after SIGTERM (an engine core releases its card and
#: stops its workers in a few seconds), before SIGKILL.
DEFAULT_GRACE_S = 10.0

#: How long after SIGKILL the sweep waits for the last of them to be gone before it names what is left.
KILL_WAIT_S = 5.0

#: The signals that mean "stop this command".
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)

_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37

#: `/proc/<pid>/stat` fields after the command name (which may hold spaces and parentheses).
_STATE, _PPID, _PGRP, _START = 0, 1, 2, 19

F = TypeVar("F", bound=Callable[..., int])


class Stopped(KeyboardInterrupt):
    """The command was told to stop (SIGTERM, SIGHUP or SIGINT); its engine's processes are being ended.

    A ``KeyboardInterrupt``, so an ``except Exception`` that turns a crash into an error line does not catch
    it, and code that treats Ctrl-C as a cancel treats every stop signal the same way.
    """

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.name = signal.Signals(signum).name
        super().__init__(self.name)


class _State:
    """The owner of this process, if one holds; one per process (an inner `owned` is the outer one's)."""

    pid: int | None = None  # the process the owner belongs to (a forked child is not it)
    sweeping = False  # the sweep runs: a stop signal only cuts its grace short
    hurry = False  # a stop signal came during the sweep: SIGKILL what is left now
    fork_parent: int | None = None  # the pid a fork in flight starts from, when its child gets a death signal
    handlers: dict[int, Any] = {}  # noqa: RUF012 (one per process) the handlers the owner replaced


def _libc() -> Any:
    return ctypes.CDLL(None, use_errno=True)


def _prctl(option: int, value: int) -> int:
    """``prctl(option, value)``; -1 where there is no ``prctl`` or it refused."""
    if not sys.platform.startswith("linux"):
        return -1
    try:
        return int(_libc().prctl(option, ctypes.c_ulong(value), 0, 0, 0))
    except (OSError, AttributeError):
        return -1


def _subreaper() -> int | None:
    """Whether this process is a child subreaper now (None where it cannot tell)."""
    if not sys.platform.startswith("linux"):
        return None
    flag = ctypes.c_int(0)
    try:
        if _libc().prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return flag.value


# ------------------------------------------------------------------------------------------------ signals
def _on_stop(signum: int, frame: object) -> None:
    del frame
    if _State.sweeping:
        _State.hurry = True
        return
    raise Stopped(signum)


def _install() -> None:
    """Route the stop signals to :func:`_on_stop`, in the main thread only (where signals are delivered).

    A SIGTERM or SIGHUP this process inherited as ignored (``nohup``) stays ignored: the caller chose that.
    SIGINT does not: a shell starts a background command with it ignored, and the command would outlive every
    Ctrl-C its script gets (as ``qda-agent``'s ``_StopSignals``).
    """
    if threading.current_thread() is not threading.main_thread():
        return
    for s in STOP_SIGNALS:
        old = signal.getsignal(s)
        if old is signal.SIG_IGN and s != signal.SIGINT:
            continue
        _State.handlers[s] = signal.signal(s, _on_stop)


def _restore() -> None:
    for s, old in _State.handlers.items():
        with contextlib.suppress(ValueError, OSError, TypeError):
            if signal.getsignal(s) is _on_stop:
                signal.signal(s, old)
    _State.handlers = {}


# --------------------------------------------------------------------------------------------- fork hooks
def _before_fork() -> None:
    owner = _State.pid == os.getpid()
    _State.fork_parent = (
        os.getpid() if owner and threading.current_thread() is threading.main_thread() else None
    )


def _after_fork_in_child() -> None:
    """A child forked from the owner's main thread dies with it; it owns nothing and takes no stop handler.

    ``PR_SET_PDEATHSIG`` fires when the thread that forked ends, hence main-thread forks only. A parent that
    died before the call is caught by the parent pid check.
    """
    parent, _State.fork_parent = _State.fork_parent, None
    if _State.pid is None:
        return
    _State.pid = None  # the owner's exit is the owner's, not this child's
    _State.sweeping = _State.hurry = False
    try:
        if parent is not None:
            _prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
            if os.getppid() != parent:  # the owner died between the fork and the prctl
                os._exit(128 + signal.SIGKILL)
        _restore()
    except Exception:  # noqa: BLE001 (a fork hook must not fail the child)
        pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_child=_after_fork_in_child)


# ----------------------------------------------------------------------------------------------- the sweep
class Proc(NamedTuple):
    """One process as one ``/proc/<pid>/stat`` read saw it: its pid, start time, parent, group and state."""

    pid: int
    started: int
    ppid: int
    pgrp: int
    state: bytes


def _proc(pid: int) -> Proc | None:
    """`pid` from one read; None if it is gone (or not ours to read)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:  # noqa: PTH123 (a pid, not a path)
            fields = f.read().rsplit(b")", 1)[1].split()
        return Proc(pid, int(fields[_START]), int(fields[_PPID]), int(fields[_PGRP]), fields[_STATE])
    except (OSError, IndexError, ValueError):
        return None


def _children(pid: int) -> list[int] | None:
    """The children of `pid` from ``/proc/<pid>/task/*/children`` (every thread's); None without that file."""
    tasks = Path(f"/proc/{pid}/task")
    found: list[int] = []
    try:
        for task in tasks.iterdir():
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                found += [int(c) for c in (task / "children").read_text().split()]
    except FileNotFoundError:
        return [] if not Path(f"/proc/{pid}").exists() else None
    except OSError:
        return None
    if not found and not (tasks / str(pid) / "children").exists():
        return None  # a kernel without CONFIG_PROC_CHILDREN
    return found


def _by_scan() -> dict[int, list[int]]:
    """Every process's children from one pass over ``/proc`` (the fallback of :func:`_children`)."""
    tree: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and (got := _proc(int(entry.name))) is not None:
            tree.setdefault(got.ppid, []).append(got.pid)
    return tree


def descendants(root: int | None = None, *, skip: frozenset[tuple[int, int]] = frozenset()) -> list[Proc]:
    """Every live process below `root` (this process), except the trees of the children in `skip`.

    `skip` holds (pid, start time) pairs. A zombie is not alive (it holds no memory and no card), so it is not
    returned, but its children are.
    """
    if not Path("/proc/self/stat").exists():
        return []
    root = os.getpid() if root is None else root
    tree: dict[int, list[int]] | None = None
    found: list[Proc] = []
    seen = {root}
    stack = [root]
    while stack:
        pid = stack.pop()
        kids = None if tree is not None else _children(pid)
        if kids is None:
            tree = tree if tree is not None else _by_scan()
            kids = tree.get(pid, [])
        for kid in kids:
            if kid in seen or (proc := _proc(kid)) is None:
                continue
            seen.add(kid)
            if proc.ppid != pid or (pid == root and (proc.pid, proc.started) in skip):
                continue
            if proc.state not in (b"Z", b"X"):
                found.append(proc)
            stack.append(kid)
    return found


def _alive(proc: Proc) -> bool:
    now = _proc(proc.pid)
    return now is not None and now.started == proc.started and now.state not in (b"Z", b"X")


def _signal(proc: Proc, sig: int) -> None:
    """Signal `proc` only while it is still the process the scan saw (its pid and start time)."""
    if _alive(proc):
        with contextlib.suppress(OSError):
            os.kill(proc.pid, sig)


def sweep(
    grace_s: float = DEFAULT_GRACE_S,
    *,
    skip: frozenset[tuple[int, int]] = frozenset(),
    kill_wait_s: float = KILL_WAIT_S,
) -> list[Proc]:
    """End every live descendant (:func:`descendants`); what is still alive after SIGKILL (none, normally).

    SIGTERM first, to each process group a descendant leads (so a member the scan missed is reached too) and
    to each descendant by pid; SIGKILL to what is left after `grace_s`, or at once after a stop signal in the
    sweep. A process is signalled at most once per signal: a handler that forks must not run twice.
    """
    own = os.getpgrp()
    termed: set[tuple[int, int]] = set()
    killed: set[tuple[int, int]] = set()
    term_end = time.monotonic() + grace_s
    kill_end: float | None = None
    while True:
        live = descendants(skip=skip)
        if not live:
            return []
        now = time.monotonic()
        if kill_end is None and (now >= term_end or _State.hurry):
            kill_end = now + kill_wait_s
        if kill_end is not None and now >= kill_end:
            return live
        sig, done = (signal.SIGTERM, termed) if kill_end is None else (signal.SIGKILL, killed)
        fresh = [p for p in live if (p.pid, p.started) not in done]
        for p in fresh:
            if p.pgrp == p.pid and p.pgrp != own and _alive(p):
                with contextlib.suppress(OSError):
                    os.killpg(p.pgrp, sig)
        for p in fresh:
            _signal(p, sig)
            done.add((p.pid, p.started))
        time.sleep(0.05)


# ----------------------------------------------------------------------------------------------- the owner
@contextlib.contextmanager
def owned(grace_s: float | None = None) -> Iterator[None]:
    """Own every process started inside: stop signals unwind, and none of those processes outlive the block.

    An `owned` inside another one in the same process is the outer one's (it does nothing of its own).
    """
    if _State.pid == os.getpid():
        yield
        return
    me = os.getpid()
    # Every descendant alive now, not only the direct children: a grandchild forked before `owned` (its parent
    # may exit and reparent it to the subreaper) is not this command's to sweep.
    before = frozenset((p.pid, p.started) for p in descendants(me))
    was_subreaper = _subreaper()
    _prctl(_PR_SET_CHILD_SUBREAPER, 1)
    _State.pid, _State.sweeping, _State.hurry = me, False, False
    _install()
    try:
        yield
    finally:
        if os.getpid() == me:  # a child forked inside leaves through here too, and owns nothing
            _State.sweeping = True
            try:
                left = sweep(DEFAULT_GRACE_S if grace_s is None else grace_s, skip=before)
                if left:
                    named = ", ".join(f"{p.pid} ({p.state.decode()})" for p in left)
                    print(
                        f"qda-kl: {len(left)} process(es) did not exit after SIGKILL: {named}",
                        file=sys.stderr,
                    )
            finally:
                _restore()
                if was_subreaper == 0:
                    _prctl(_PR_SET_CHILD_SUBREAPER, 0)
                _State.pid, _State.sweeping, _State.hurry = None, False, False


def command(prog: str, grace_s: float | None = None) -> Callable[[F], F]:
    """Run a CLI command under :func:`owned`; a stop signal ends it with `INTERRUPTED` (130).

    One status for every stop signal, Ctrl-C included (`qda_quant.exit`): a script that watches for an
    interruption has one code to handle, whatever the sender chose. The line on stderr still names the signal.
    """

    def wrap(fn: F) -> F:
        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> int:
            try:
                with owned(grace_s):
                    return fn(*args, **kwargs)
            except Stopped as stop:
                print(f"{prog}: stopped by {stop.name}", file=sys.stderr)
                return INTERRUPTED

        return run  # type: ignore[return-value]

    return wrap
