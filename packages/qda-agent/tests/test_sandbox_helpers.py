"""A completed or interrupted pytest run leaves none of its live-state helper processes alive."""

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"),
    reason="checks helper lifetimes with Linux pidfds",
)


@pytest.mark.parametrize("interrupted", [False, True], ids=["normal", "keyboard-interrupt"])
def test_pytest_reaps_its_sandbox_helpers(tmp_path, interrupted):
    """End a real pytest process with a live CLI and relay, then check each recorded pidfd for exit."""
    marker, ready, release = (tmp_path / name for name in ("pids", "ready", "release"))
    cli = (
        "import json, os, time\n"
        "from pathlib import Path\n"
        "pid = os.getpid()\n"
        "children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()\n"
        f"Path({str(ready)!r}).write_text(json.dumps([pid, *map(int, children)]))\n"
        "time.sleep(120)\n"
    )
    test = tmp_path / "test_run.py"
    test.write_text(
        "import json, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from qda_agent.isolation import wall, hostnet\n"
        "def test_run(sandbox_helpers, request):\n"
        "    bridge = hostnet.NetBridge()\n"
        "    request.addfinalizer(bridge.stop)\n"
        "    sock = bridge.socket([])\n"
        f"    cli = {cli!r}\n"
        "    command = [*wall.LIVE_PYTHON, wall.LIVE_SOURCE, '{}', str(sock), '--',\n"
        "               sys.executable, '-I', '-S', '-c', cli]\n"
        "    guard = sandbox_helpers(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    deadline = time.monotonic() + 15\n"
        f"    ready = Path({str(ready)!r})\n"
        "    while not ready.exists() and time.monotonic() < deadline:\n"
        "        time.sleep(0.02)\n"
        "    pids = json.loads(ready.read_text())\n"
        "    assert len(pids) == 2, 'the CLI must have one live relay'\n"
        f"    Path({str(marker)!r}).write_text(json.dumps([guard.pid, *pids]))\n"
        f"    while not Path({str(release)!r}).exists():\n"
        "        time.sleep(0.02)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]),
    }
    runner = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", "-p", "testkit", str(test)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    pidfds = []
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and runner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), runner.communicate(timeout=5)[0]
        pids = json.loads(marker.read_text())
        for pid in pids:
            pidfd = os.pidfd_open(pid)
            pidfds.append(pidfd)
            assert not select.select([pidfd], [], [], 0)[0], f"helper {pid} exited before teardown"
        if interrupted:
            runner.send_signal(signal.SIGINT)  # pytest alone: its finalizer must stop the separate guard
        else:
            release.touch()  # return from the test without waiting for its helpers
        out, _ = runner.communicate(timeout=30)
        assert runner.returncode == (2 if interrupted else 0), out
        for pid, pidfd in zip(pids, pidfds, strict=True):
            assert select.select([pidfd], [], [], 5)[0], f"helper {pid} outlived pytest:\n{out}"
    finally:
        if runner.poll() is None:
            runner.send_signal(signal.SIGINT)
            try:
                runner.wait(timeout=10)
            except subprocess.TimeoutExpired:
                runner.kill()
                runner.wait()
        for pidfd in reversed(pidfds):
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            os.close(pidfd)
