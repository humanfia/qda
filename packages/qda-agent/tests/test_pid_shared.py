"""`pid: shared`: the pid namespace the wall needs where a fresh procfs is refused."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from qda_agent.isolation import allocation, wall
from qda_agent.sched import sandbox
from tests_support import assert_job_group, assert_job_process


def test_auto_keeps_a_private_pid_where_a_fresh_procfs_mounts(monkeypatch):
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))
    assert sandbox.pid_form() == "private"
    assert sandbox.pid_args() == ["--unshare-pid"]


def test_auto_shares_the_pid_where_a_fresh_procfs_needs_the_hosts(monkeypatch):
    """The DGX hosts: a fresh procfs there comes only with the host's pid namespace."""
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "shared"))
    assert sandbox.pid_form() == "shared"
    assert sandbox.pid_args() == []


def test_auto_shares_the_pid_where_the_host_proc_is_bound(monkeypatch):
    monkeypatch.setattr(sandbox, "_ns", ("host", "shared"))
    assert sandbox.pid_form() == "shared"
    assert sandbox.pid_args() == []


def test_an_explicit_form_wins_and_an_unknown_one_is_refused(monkeypatch):
    monkeypatch.setattr(sandbox, "_ns", ("host", "shared"))
    assert sandbox.pid_form("private") == "private"
    assert sandbox.pid_form("shared") == "shared"
    with pytest.raises(ValueError, match="pid"):
        sandbox.pid_form("host")


def test_the_walls_argv_follows_the_pid_form(tmp_path):
    private = wall.bwrap_argv([], [str(tmp_path)], [], keep=[tmp_path], pid="private")
    assert "--unshare-pid" in private
    shared = wall.bwrap_argv([], [str(tmp_path)], [], keep=[tmp_path], pid="shared")
    assert "--unshare-pid" not in shared


def test_shared_pid_refuses_a_node_with_other_work(monkeypatch):
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: "other running job(s) of this user: 9")
    cfg = SimpleNamespace(sandbox=SimpleNamespace(name="bwrap", pid="auto"))
    with pytest.raises(wall.WallRefused, match="dedicated"), wall.gpu_wall(cfg):
        pass


def _fake_proc(tmp_path: Path, parents: dict[int, int], names: dict[int, str]) -> Path:
    """A `/proc` holding only these `{pid: ppid}` entries (this process among them), with their names.

    `stat` carries the fields `rows` reads (state, ppid, pgrp, session) and `status` the name and parent a
    real `/proc` shows; `comm` is what the stranger listing prints.
    """
    proc = tmp_path / "proc"
    for pid, ppid in parents.items():
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "stat").write_text(f"{pid} ({names.get(pid, 'x')}) S {ppid} {pid} {pid} 0 -1 0")
        (entry / "status").write_text(f"Name:\t{names.get(pid, 'x')}\nPPid:\t{ppid}\n")
        (entry / "comm").write_text(f"{names.get(pid, 'x')}\n")
    return proc


def test_the_allocations_own_keepers_do_not_refuse_a_dedicated_node(tmp_path, monkeypatch):
    """The batch step's script and its idle child are the allocation's own; Slurm names them as its batch step."""
    run, step, batch, keeper = os.getpid(), 424241, 424242, 424243
    proc = _fake_proc(
        tmp_path,
        {run: step, step: 1, 1: 0, batch: 1, keeper: batch},
        {step: "bash", batch: "slurm_script", keeper: "sleep"},
    )
    monkeypatch.setattr(allocation, "_other_jobs", lambda: [])
    monkeypatch.setattr(allocation, "_batch_step_pids", lambda: set())  # Slurm cannot name the step
    assert {p.pid for p in allocation._other_processes(proc)} == {batch, keeper}
    monkeypatch.setattr(allocation, "_batch_step_pids", lambda: {batch, keeper})
    assert allocation._other_processes(proc) == []
    assert allocation.why_not_dedicated(proc) is None


def test_a_process_beside_the_keepers_still_refuses(tmp_path, monkeypatch):
    """Another step's payload (or any other process of ours) is not the batch step's, so it still refuses."""
    run, step, batch, keeper, extra = os.getpid(), 424241, 424242, 424243, 424244
    proc = _fake_proc(
        tmp_path,
        {run: step, step: 1, 1: 0, batch: 1, keeper: batch, extra: 1},
        {step: "bash", batch: "slurm_script", keeper: "sleep", extra: "bash"},
    )
    monkeypatch.setattr(allocation, "_other_jobs", lambda: [])
    monkeypatch.setattr(allocation, "_batch_step_pids", lambda: {batch, keeper})
    why = allocation.why_not_dedicated(proc)
    assert why is not None
    assert f"{extra} (bash)" in why
    assert str(keeper) not in why


def test_a_process_is_excused_by_its_slurm_step_not_by_its_name(tmp_path, monkeypatch):
    """A process named like the batch script is a stranger unless Slurm lists it in the batch step."""
    run, step, batch, impostor = os.getpid(), 424241, 424242, 424243
    proc = _fake_proc(
        tmp_path,
        {run: step, step: 1, 1: 0, batch: 1, impostor: 1},
        {step: "bash", batch: "slurm_script", impostor: "slurm_script"},
    )
    monkeypatch.setattr(allocation, "_other_jobs", lambda: [])
    monkeypatch.setattr(allocation, "_batch_step_pids", lambda: {batch})
    why = allocation.why_not_dedicated(proc)
    assert why is not None and f"{impostor} (slurm_script)" in why


def test_the_batch_step_pids_are_read_from_slurm(monkeypatch):
    """`scontrol listpids <job>.batch` names the keepers; no Slurm answer leaves the check strict."""
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    calls = []

    def fake(argv):
        calls.append(argv)
        return (
            "PID      JOBID    STEPID   LOCALID GLOBALID\n"
            "2611442  77 batch    0       0\n"
            "2611446  77 batch    -       -\n"
        )

    monkeypatch.setattr(allocation, "_slurm", fake)
    assert allocation._batch_step_pids() == {2611442, 2611446}
    assert calls == [["scontrol", "listpids", "77.batch"]]
    monkeypatch.setattr(allocation, "_slurm", lambda argv: None)
    assert allocation._batch_step_pids() == set()
    monkeypatch.delenv("SLURM_JOB_ID")
    assert allocation._batch_step_pids() == set()
    assert calls == [["scontrol", "listpids", "77.batch"]]  # no job, no query


def test_a_process_outside_the_run_is_named(tmp_path):
    """A process of this user that is not this run's tree (a detached grandchild) is what `shared` must refuse."""
    marker = tmp_path / "pid"
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork():\n"
        "        os._exit(0)\n"
        f"    open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(120)\n"
        "os._exit(0)\n"
    )
    launcher = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, timeout=30)
    assert launcher.returncode == 0
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(marker.read_text())
    try:
        assert pid in {p.pid for p in allocation._other_processes(Path("/proc"))}
        assert allocation.why_not_dedicated() is not None
    finally:
        assert_job_process(pid, tmp_path)
        os.kill(pid, 9)


def test_a_stopped_turn_leaves_no_process_writing(tmp_path):
    """A cancelled turn's child stops writing: the stop sequence signals the outer group (bubblewrap's), which
    the turn left when the guard put it in a session of its own -- the guard must end that session on TERM
    before it can itself be killed by the group's SIGKILL."""
    marker, pulse = tmp_path / "pids", tmp_path / "pulse"
    child = f"import time\nwhile True:\n    open({str(pulse)!r}, 'a').write('x')\n    time.sleep(0.02)\n"
    turn = (
        "import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ' ' + str(p.pid))\n"
        "time.sleep(120)\n"
    )
    # The guard is started by a parent of its own, as bubblewrap starts it in a turn: the parent leads the
    # group the stop sequence signals, and the guard is not its leader (so it can leave the group).
    parent = "import subprocess, sys\nsys.exit(subprocess.Popen(sys.argv[1:]).wait())\n"
    guard = subprocess.Popen(
        [
            sys.executable,
            "-S",
            "-c",
            parent,
            sys.executable,
            "-I",
            "-S",
            "-c",
            wall.TURN_SOURCE,
            "--",
            sys.executable,
            "-S",
            "-c",
            turn,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        start_new_session=True,  # the process group the stop sequence signals (bubblewrap's, in a turn)
    )
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 30
        while (not marker.exists() or not pulse.exists()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists() and pulse.exists(), guard.poll()
        pids = [int(one) for one in marker.read_text().split()]
        # What the run's stop sequence does (Humanize's CodexServer.stop): TERM the group, wait for its leader,
        # then KILL whatever is left of it. The turn is not in that group: only the guard hears the stop.
        assert_job_group(guard.pid, tmp_path)
        os.killpg(guard.pid, signal.SIGTERM)
        assert guard.wait(timeout=30) == -signal.SIGTERM, guard.returncode
        with contextlib.suppress(ProcessLookupError, PermissionError):
            assert_job_group(guard.pid, tmp_path)
            os.killpg(guard.pid, signal.SIGKILL)
        time.sleep(0.3)
        before = pulse.stat().st_size
        time.sleep(0.3)
        assert pulse.stat().st_size == before, "a child of the cancelled turn kept writing"
        for pid in pids:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                continue
            assert state == "Z", f"process {pid} outlived the cancelled turn"
    finally:
        if guard.poll() is None:
            guard.kill()
            guard.wait()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                assert_job_process(pid, tmp_path)
                os.kill(pid, signal.SIGKILL)


def test_a_turn_child_that_changes_session_is_still_held(tmp_path):
    """A child that starts a session of its own and double-forks escapes the session, but not the turn: the
    guard is a subreaper, so the orphan is reparented to it, and the turn's end takes it with the rest."""
    marker, pulse = tmp_path / "escaped", tmp_path / "pulse"
    escape = (
        "import os, time\n"
        "os.setsid()\n"
        "if os.fork():\n"
        "    os._exit(0)\n"
        f"open({str(pulse)!r}, 'a').write('x')\n"
        f"open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n"
        f"    open({str(pulse)!r}, 'a').write('x')\n"
        "    time.sleep(0.02)\n"
    )
    turn = (
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {escape!r}], stdin=subprocess.DEVNULL,\n"
        "                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "deadline = time.monotonic() + 30\n"
        f"while not pathlib.Path({str(marker)!r}).exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "sys.exit(0)\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", wall.TURN_SOURCE, "--", sys.executable, "-S", "-c", turn],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    pid = int(marker.read_text())
    before = pulse.stat().st_size
    time.sleep(0.3)
    assert pulse.stat().st_size == before, "the child that left the session kept writing after the turn"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the child that left the session ({pid}) outlived the turn")


def test_a_turn_leaves_no_process_behind(tmp_path):
    """The guard ends the turn's session: a sleep the turn started is gone when the turn is."""
    marker = tmp_path / "pid"
    turn = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
        "sys.exit(3)\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", wall.TURN_SOURCE, "--", sys.executable, "-S", "-c", turn],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 3, done.stderr
    pid = int(marker.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the turn's sleep {pid} outlived the turn")
