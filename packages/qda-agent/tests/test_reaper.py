"""Every process a job starts ends with the job: the reaper is the outermost process of every job.

The shapes here are the ones a process group misses: a child that starts a session of its own (`setsid`), a
double-forked daemon that reparents away from the job, and the job sandbox's own `bwrap --new-session` with the
host's pid namespace (`pid: shared`), where a signal to bwrap's group reaches only bwrap.
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest
from qda_agent.sched import LocalBackend, sandbox

#: A job that leaves two processes behind, each writing its pid: a grandchild in a session of its own, and a
#: daemon (fork, setsid, fork, middle exits) that is reparented away from the job.
ESCAPES = """import os, sys, time
out = sys.argv[1]
grandchild = os.fork()
if grandchild == 0:
    os.setsid()
    with open(out, "a") as f:
        f.write(f"{os.getpid()}\\n")
    time.sleep(600)
    os._exit(0)
middle = os.fork()
if middle == 0:
    os.setsid()
    if os.fork():
        os._exit(0)
    with open(out, "a") as f:
        f.write(f"{os.getpid()}\\n")
    time.sleep(600)
    os._exit(0)
os.waitpid(middle, 0)
with open(out, "a") as f:
    f.write(f"{os.getpid()}\\n")
time.sleep(600)
"""

#: A job that ends at once, leaving a double-forked daemon behind: the job's end must end it too. The job waits
#: for the daemon to be up first, so the pid the test reads is the daemon's.
LEAVES = """import os, sys, time
middle = os.fork()
if middle == 0:
    os.setsid()
    if os.fork():
        os._exit(0)
    with open(sys.argv[1], "w") as f:
        f.write(str(os.getpid()))
    time.sleep(600)
    os._exit(0)
os.waitpid(middle, 0)
for _ in range(6000):
    if os.path.exists(sys.argv[1]) and os.path.getsize(sys.argv[1]):
        break
    time.sleep(0.01)
"""

#: A job that writes its own pid, leaves a double-forked daemon that ignores SIGTERM, and exits: through the
#: sweep that daemon holds up, the job's own process must stay unreaped (a zombie), so its pid cannot be reused.
ZOMBIE_LEADER = """import os, signal, sys, time
with open(sys.argv[1], "a") as f:
    f.write(f"{os.getpid()}\\n")
middle = os.fork()
if middle == 0:
    os.setsid()
    if os.fork():
        os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(sys.argv[1], "a") as f:
        f.write(f"{os.getpid()}\\n")
    time.sleep(600)
    os._exit(0)
os.waitpid(middle, 0)
for _ in range(6000):
    if len(open(sys.argv[1]).read().split()) > 1:
        break
    time.sleep(0.01)
"""

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


def sleepers(marker: Path, count: int, limit: float = 30.0) -> list[int]:
    """The `count` pids the job wrote to `marker`."""
    end = time.monotonic() + limit
    while True:
        pids = [int(line) for line in marker.read_text().split()] if marker.exists() else []
        if len(pids) >= count:
            return pids
        assert time.monotonic() < end, f"the job wrote {pids!r}, waited for {count}"
        time.sleep(0.05)


def state_of(pid: int) -> str | None:
    """The state of `pid` from /proc, or None if it is not there."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def survivors(pids: list[int]) -> list[int]:
    """The pids still in /proc and not a zombie (a zombie holds nothing; init reaps it)."""
    return [pid for pid in pids if (state := state_of(pid)) is not None and state != "Z"]


def wait_gone(pids: list[int], limit: float = 5.0) -> None:
    """Fail unless every pid left /proc (or is a zombie) within `limit` seconds."""
    end = time.monotonic() + limit
    while (left := survivors(pids)) and time.monotonic() < end:
        time.sleep(0.05)
    assert not left, f"processes of the job survived it: {left}"


def test_stop_ends_a_session_of_its_own_and_a_double_forked_daemon(tmp_path):
    """`LocalBackend.stop` reaches a setsid grandchild and a daemon that reparented away from the job."""
    script = tmp_path / "job.py"
    script.write_text(ESCAPES)
    marker = tmp_path / "pids"
    be = LocalBackend(kill_grace_s=2)
    handle = be.launch([sys.executable, str(script), str(marker)], str(tmp_path), [0], tmp_path / "log", "t")
    pids = sleepers(marker, 3)
    be.stop(handle)
    assert be.poll(handle) is not None
    wait_gone(pids)


def test_a_job_that_ends_alone_leaves_nothing_behind(tmp_path):
    """The reaper sweeps what a job leaves when the job itself exits successfully."""
    script = tmp_path / "job.py"
    script.write_text(LEAVES)
    marker = tmp_path / "pid"
    be = LocalBackend(kill_grace_s=2)
    handle = be.launch([sys.executable, str(script), str(marker)], str(tmp_path), [0], tmp_path / "log", "t")
    pids = sleepers(marker, 1)
    end = time.monotonic() + 10
    while (rc := be.poll(handle)) is None and time.monotonic() < end:
        time.sleep(0.02)
    assert rc == 0
    wait_gone(pids)


def test_the_jobs_own_process_is_not_reaped_before_the_sweep_ends(tmp_path):
    """Its pid is its process group's id: it stays a zombie until the last group signal, so it cannot be reused."""
    script = tmp_path / "job.py"
    script.write_text(ZOMBIE_LEADER)
    marker = tmp_path / "pids"
    be = LocalBackend(kill_grace_s=2)  # the sweep waits this long for the TERM-ignoring daemon
    handle = be.launch([sys.executable, str(script), str(marker)], str(tmp_path), [0], tmp_path / "log", "t")
    leader, daemon = sleepers(marker, 2)
    seen = False
    end = time.monotonic() + 10
    while time.monotonic() < end and be.poll(handle) is None:
        seen = seen or state_of(leader) == "Z"
        time.sleep(0.02)
    assert seen, "the sweep ended without the job's own process waiting as a zombie"
    assert be.poll(handle) is not None
    wait_gone([daemon])  # the sweep killed what the job left
    assert state_of(leader) is None  # and only then reaped the job's own process


@needs_bwrap
def test_stop_ends_the_tree_of_a_job_sandbox_with_a_shared_pid(tmp_path):
    """The orphan shape of a GPU job: bwrap `--new-session` in the host's pid namespace, so the job's session
    and group are bwrap's own and a signal to bwrap's group never reaches the processes it started."""
    script = tmp_path / "job.py"
    script.write_text(ESCAPES)
    marker = tmp_path / "pids"
    inner = [sys.executable, str(script), str(marker)]
    argv = sandbox.bwrap_argv(inner, str(tmp_path), writable=[tmp_path], pid="shared")
    be = LocalBackend(kill_grace_s=2)
    handle = be.launch(argv, str(tmp_path), [0], tmp_path / "log", "t")
    pids = sleepers(marker, 3)
    assert len([pid for pid in pids if Path(f"/proc/{pid}").exists()]) == 3
    be.stop(handle)
    assert be.poll(handle) is not None
    wait_gone(pids)


def test_the_kill_domain_starts_polls_and_stops_a_job(tmp_path):
    """`Reaper` is the `KillDomain` of `isolation.api`: a launch in, a handle out, the job's own output."""
    from qda_agent.isolation.api import Launch
    from qda_agent.isolation.reaper import Reaper

    launch = Launch(argv=[sys.executable, "-c", "print('hi')"], env={"KEEP": "1"})
    domain = Reaper()
    log = tmp_path / "log"
    handle = domain.spawn(launch, cwd=tmp_path, log=log)
    end = time.monotonic() + 30
    while (rc := domain.poll(handle)) is None and time.monotonic() < end:
        time.sleep(0.02)
    assert rc == 0 and log.read_text() == "hi\n"
    domain.stop(handle, 2.0)  # already ended: nothing is signalled


def test_the_launch_environment_is_kept_exactly(tmp_path, monkeypatch):
    """`env={}` stays an empty environment; it is not this process's, and `env=None` is what inherits."""
    from qda_agent.isolation.api import Launch
    from qda_agent.isolation.reaper import Reaper

    monkeypatch.setenv("QDA_ENV_CANARY", "1")  # a variable of this process the job must not see
    domain = Reaper()
    log = tmp_path / "log"
    read = "import os; print(os.environ.get('QDA_ENV_CANARY', 'absent'))"
    handle = domain.spawn(Launch(argv=[sys.executable, "-c", read], env={}), cwd=tmp_path, log=log)
    end = time.monotonic() + 30
    while (rc := domain.poll(handle)) is None and time.monotonic() < end:
        time.sleep(0.02)
    assert rc == 0 and log.read_text().strip() == "absent"
    log.unlink()
    handle = domain.spawn(Launch(argv=[sys.executable, "-c", read], env=None), cwd=tmp_path, log=log)
    end = time.monotonic() + 30
    while (rc := domain.poll(handle)) is None and time.monotonic() < end:
        time.sleep(0.02)
    assert rc == 0 and log.read_text().strip() == "1"  # env=None inherits this process's environment


def test_stop_with_a_short_grace_ends_a_detached_child(tmp_path):
    """The caller's grace is the tree's: a detached, TERM-ignoring child is gone when `stop` returns."""
    from qda_agent.isolation.api import Launch
    from qda_agent.isolation.reaper import Reaper

    marker = tmp_path / "pid"
    leaf = (
        "import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(5)"
    )
    command = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{leaf!r},sys.argv[1]],start_new_session=True); time.sleep(5)"
    )
    domain = Reaper()
    launch = Launch(argv=[sys.executable, "-c", command, str(marker)], env=dict(os.environ))
    handle = domain.spawn(launch, cwd=tmp_path, log=tmp_path / "log")
    pids = sleepers(marker, 1)
    domain.stop(handle, 0.05)
    assert not survivors(pids), "stop returned while a detached child of the job was still alive"


def fields(*, state=b"S", ppid=1, pgrp=1, session=1, started=1):
    """One `/proc/<pid>/stat` read, split after the command name as the reaper splits it."""
    row = [b"0"] * 20
    row[0], row[1], row[2], row[3], row[19] = (
        state,
        str(ppid).encode(),
        str(pgrp).encode(),
        str(session).encode(),
        str(started).encode(),
    )
    return row


def test_membership_and_identity_come_from_one_read(monkeypatch):
    """A member's start time is the one read that made it a member, never a newer process's."""
    from qda_agent.isolation import reaper

    reads = iter([fields(ppid=71000, session=71001, started=1), fields(ppid=72000, session=72000, started=2)])
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("/proc/71002")]))
    monkeypatch.setattr(reaper, "_fields", lambda pid: next(reads))
    assert reaper._members(71001, 71000) == {71002: 1}


def test_a_signal_goes_only_to_the_process_the_pidfd_names(monkeypatch):
    """The signal goes through a descriptor opened before the identity is read again."""
    from qda_agent.isolation import reaper

    sent = []
    monkeypatch.setattr(reaper.os, "pidfd_open", lambda pid: 9, raising=False)
    monkeypatch.setattr(reaper.os, "close", lambda fd: None, raising=False)
    monkeypatch.setattr(
        reaper.signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd, sig)), raising=False
    )
    monkeypatch.setattr(reaper, "_started", lambda pid: 2)  # the number now names a newer process
    reaper._signal(71001, 1, signal.SIGTERM)
    assert not sent
    monkeypatch.setattr(reaper, "_started", lambda pid: 1)
    reaper._signal(71001, 1, signal.SIGTERM)
    assert sent == [(9, signal.SIGTERM)]


def test_the_group_signal_needs_the_leader_it_started(monkeypatch):
    """A group id that a reuse has given to another process is not signalled."""
    from qda_agent.isolation import reaper

    groups = []
    monkeypatch.setattr(reaper, "_members", lambda session, guard: {71002: 1})
    monkeypatch.setattr(reaper, "_signal", lambda *args: None)
    monkeypatch.setattr(reaper, "_kill_now", True)
    monkeypatch.setattr(reaper.os, "killpg", lambda group, sig: groups.append((group, sig)))
    monkeypatch.setattr(reaper, "_started", lambda pid: 999)
    reaper._kill(71001, 71000, 0.05, leader=123)
    assert not groups
    monkeypatch.setattr(reaper, "_started", lambda pid: 123)
    reaper._kill(71001, 71000, 0.05, leader=123)
    # the leader this program started: its group gets both passes (TERM, then KILL)
    assert groups == [(71001, signal.SIGTERM), (71001, signal.SIGKILL)]


def test_a_reaper_keeps_the_descriptors_the_job_needs(tmp_path):
    """The job reads a descriptor the caller opened; the reaper passes it through, and nothing else."""
    log = tmp_path / "log"
    read, write = os.pipe()
    try:
        os.write(write, b"carried")
        os.close(write)
        be = LocalBackend(kill_grace_s=2)
        command = [sys.executable, "-c", f"import os; print(os.read({read}, 7).decode())"]
        handle = be.launch(command, str(tmp_path), [0], log, "t", pass_fds=[read])
        end = time.monotonic() + 30
        while (rc := be.poll(handle)) is None and time.monotonic() < end:
            time.sleep(0.02)
        assert rc == 0
        assert log.read_text().strip() == "carried"
    finally:
        os.close(read)
