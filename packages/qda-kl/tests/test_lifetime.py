"""A qda-kl command leaves none of its engine's processes behind, however it ends.

The engine is fake (``fake_engine.py``): a forked child and a session of its own whose leader ignores SIGTERM,
started through ``open_engine`` from ``qda-kl ref``. What is checked is the processes, not the exit code.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from qda_kl import lifetime

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc and prctl are Linux's")

DRIVER = Path(__file__).with_name("fake_engine.py")


def _start(pid: int) -> int:
    """The process's start time (`/proc/<pid>/stat` field 22), which with its pid names it."""
    return int(Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()[19])


def _alive(pid: int, start: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
    except OSError:
        return False
    return int(fields[19]) == start and fields[0] not in (b"Z", b"X")


def _wait(predicate, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@contextlib.contextmanager
def _command(tmp_path: Path, mode: str, which: str = "main"):
    """The fake ``qda-kl ref``, its engine's (pid, start) pairs once it generates; whatever is left is killed."""
    (tmp_path / "p.jsonl").write_text("")
    proc = subprocess.Popen(
        [sys.executable, str(DRIVER), str(tmp_path), mode, which], stderr=subprocess.PIPE, text=True
    )
    tree: list[tuple[int, int]] = []
    try:
        assert _wait(lambda: (tmp_path / "ready").exists() or proc.poll() is not None, 60), "no engine"
        tree += [tuple(map(int, line.split())) for line in (tmp_path / "pids").read_text().splitlines()]
        assert len(tree) == 3
        assert mode == "raise" or all(_alive(*p) for p in tree)  # a raise may already have ended it
        yield proc, tree
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        proc.wait()
        for pid, start in tree:  # by pid, only what this test started
            if _alive(pid, start):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize(
    "sig,mode",
    [(signal.SIGTERM, "hang"), (signal.SIGINT, "hang"), (signal.SIGHUP, "hang"), (signal.SIGTERM, "start")],
)
def test_a_stop_signal_ends_the_whole_engine(tmp_path, sig, mode):
    """Every stop signal leaves the contract's one status for an interruption: 130, whatever was sent."""
    with _command(tmp_path, mode) as (proc, tree):
        proc.send_signal(sig)
        assert proc.wait(timeout=30) == 130
        assert f"stopped by {signal.Signals(sig).name}" in proc.stderr.read()
        assert not [p for p in tree if _alive(*p)]


def test_an_exception_ends_the_whole_engine(tmp_path):
    with _command(tmp_path, "raise") as (proc, tree):
        assert proc.wait(timeout=30) == 1
        assert "the fake engine failed" in proc.stderr.read()
        assert not [p for p in tree if _alive(*p)]


def test_without_the_owner_sigterm_orphans_the_engine(tmp_path):
    """The defect itself: Python's default SIGTERM ends the command and nothing else."""
    with _command(tmp_path, "hang", which="unowned") as (proc, tree):
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == -signal.SIGTERM
        assert [p for p in tree if _alive(*p)]


def test_sigkill_takes_the_forked_engine_with_it(tmp_path):
    """No handler sees SIGKILL; the forked engine core dies by its parent-death signal."""
    with _command(tmp_path, "hang") as (proc, tree):
        proc.kill()
        proc.wait(timeout=30)
        assert _wait(lambda: not _alive(*tree[0]), 10)


def test_a_child_older_than_the_owner_is_not_its_own():
    older = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with lifetime.owned(grace_s=1.0):
            newer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        assert newer.wait(timeout=5) == -signal.SIGTERM
        assert older.poll() is None
    finally:
        older.kill()
        older.wait()


def test_a_grandchild_that_predates_the_owner_is_not_its_own():
    """A child forked before `owned` forks a grandchild and exits: the grandchild reparents to the subreaper,
    and only recording every descendant the owner already had keeps it out of the sweep."""
    spawn = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", spawn], stdout=subprocess.PIPE, text=True)
    grandchild = None
    try:
        grandchild = int(parent.stdout.readline())
        started = _start(grandchild)
        with lifetime.owned(grace_s=0.5):
            parent.terminate()  # the orphan reparents to this process, now the subreaper
            parent.wait(timeout=10)
        assert _alive(grandchild, started)
    finally:
        parent.kill()
        parent.wait()
        if grandchild is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild, 0)
