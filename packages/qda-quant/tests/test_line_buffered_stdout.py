"""A CLI's progress line reaches a redirected stdout when it is printed, not when 8 KB have piled up."""

import subprocess
import sys
import time

CHILD = """
import sys, time
{setup}
print("step 1", flush=False)
time.sleep(30)
"""


def _first_line_seen(tmp_path, setup: str) -> bool:
    out = tmp_path / "log"
    with out.open("w") as f:
        child = subprocess.Popen([sys.executable, "-c", CHILD.format(setup=setup)], stdout=f)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if out.read_text():
                return True
            time.sleep(0.1)
        return False
    finally:
        child.kill()
        child.wait()


def test_a_redirected_stdout_shows_each_line_as_it_is_printed(tmp_path):
    assert _first_line_seen(tmp_path, "from qda_quant.io import line_buffered_stdout; line_buffered_stdout()")


def test_without_it_python_holds_the_line_in_its_buffer(tmp_path):
    # the fault being fixed: a redirected stdout is block-buffered
    assert not _first_line_seen(tmp_path, "")
