"""Engine children keep their stderr destination after launch-time capture ends."""

import os
import subprocess
import sys
import threading

from qda_quant.output import vllm_env


def test_late_child_stderr_cannot_corrupt_a_reused_descriptor(tmp_path, monkeypatch, capfd):
    """A live engine child must keep forwarding to stderr when the launcher's saved FD is reused."""
    saved = []
    dup = os.dup

    def track_dup(fd):
        result = dup(fd)
        if fd == 2 and not saved:
            saved.append(result)
        return result

    monkeypatch.setattr(vllm_env.os, "dup", track_dup)
    record = tmp_path / "unrelated-record.json"
    original = b'{"untouched": true}\n'
    record.write_bytes(original)
    before = set(threading.enumerate())
    child = None
    record_fd = None
    try:
        with vllm_env.engine_output():
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; print('READY'); sys.stdin.buffer.read(1); "
                    "sys.stderr.write('LATE_ENGINE_LINE\\n'); sys.stderr.flush()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            assert child.stdout.readline() == b"READY\n"
        pumps = [thread for thread in threading.enumerate() if thread not in before]
        record_fd = os.open(record, os.O_WRONLY | os.O_APPEND)
        assert record_fd == saved[0]  # force reuse of the launcher's closed descriptor
        child.stdin.write(b"x")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        for thread in pumps:
            thread.join(timeout=2)
            assert not thread.is_alive()
        assert record.read_bytes() == original
        assert "LATE_ENGINE_LINE" in capfd.readouterr().err
    finally:
        if record_fd is not None:
            os.close(record_fd)
        if child is not None:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()


def test_a_live_engine_is_forwarded_but_no_longer_recorded_after_the_block(capfd):
    """A long-lived engine keeps writing to stderr after its start; that output must reach it, not a list."""
    child = None
    try:
        with vllm_env.engine_output() as output:
            os.write(2, b"WHILE_OPEN\n")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; print('READY'); sys.stdin.buffer.read(1); "
                    "sys.stderr.write('LATE_ENGINE_LINE\\n'); sys.stderr.flush()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            assert child.stdout.readline() == b"READY\n"
            assert "WHILE_OPEN" in output()
        # the block is closed: fd 2 is the caller's again, and the child still holds the pump's pipe
        os.write(2, b"AFTER_CLOSE\n")
        child.stdin.write(b"x")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        recorded = output()  # joins the pump, which ends when the child's last write end closes
        assert "WHILE_OPEN" in recorded
        assert "AFTER_CLOSE" not in recorded and "LATE_ENGINE_LINE" not in recorded
        err = capfd.readouterr().err
        assert "AFTER_CLOSE" in err and "LATE_ENGINE_LINE" in err
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
