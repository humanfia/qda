"""The paper's LiveCodeBench case runner (QDA's, not upstream), run inside the sandbox by ``grade_one.py``.

Ported from the paper's grader, judging as described in the paper's Appendix B. It differs from the official
runner (``testing_util.run_test``) in two ways:

* A stdin problem runs each case as a real child interpreter, with the input on fd 0 and fd 1 captured. The
  official runner executes the program in its own interpreter with ``sys.stdin``/``sys.stdout`` swapped for
  mock objects, where much code that reads ``sys.stdin.buffer`` or writes ``sys.stdout.buffer`` fails, as does
  ``global`` in a nested function (the program is wrapped in one). The output is compared by the official rule
  (stripped lines, exact or Decimal-equal). The child sees the sandbox's standard library and NumPy; the
  paper's saw its eval environment's site-packages.
* The per-case limit is CPU time, with a wall-clock backstop of ``backstop`` times the limit (the host's
  ``livecodebench_sandbox.WALL_BACKSTOP``), so a verdict depends on the work a program does rather than on
  node load. A functional problem still runs through the official ``run_test``, with its ``signal.alarm``
  counted on CPU time.
"""

from __future__ import annotations

import json
import resource
import signal
import subprocess
import sys

import testing_util


class _CpuAlarm:
    """The ``signal`` module as the official runner sees it, with ``alarm`` on CPU time.

    Through this object ``SIGALRM`` is SIGPROF, and ``alarm(t)`` arms ITIMER_PROF (CPU time of this process,
    which runs the program) for ``t`` seconds plus ITIMER_REAL for ``t * backstop`` on the real SIGALRM;
    ``signal(SIGALRM, h)`` installs ``h`` on both. Everything else is the real module.
    """

    def __init__(self, real, backstop: int):
        self._real = real
        self._backstop = backstop
        self.SIGALRM = real.SIGPROF

    def __getattr__(self, name):
        return getattr(self._real, name)

    def signal(self, signum, handler):
        if signum == self._real.SIGPROF:
            self._real.signal(self._real.SIGALRM, handler)
        return self._real.signal(signum, handler)

    def alarm(self, seconds):
        self._real.setitimer(self._real.ITIMER_PROF, float(seconds))
        self._real.setitimer(self._real.ITIMER_REAL, float(seconds) * self._backstop)
        return 0


def _same_output(pred: str, gold: str) -> bool:
    """The official stdin comparison: stripped lines, each exact or Decimal-equal."""
    got, want = testing_util.get_stripped_lines(pred), testing_util.get_stripped_lines(gold)
    if len(got) != len(want):
        return False
    for x, y in zip(got, want, strict=True):
        if x == y:
            continue
        ok_x, dx = testing_util.convert_line_to_decimals(x)
        ok_y, dy = testing_util.convert_line_to_decimals(y)
        if not (ok_x and ok_y and dx == dy):
            return False
    return True


def _run_stdin(code: str, inputs: list[str], outputs: list[str], timeout: int, backstop: int) -> list:
    """Each case as a child interpreter; the first failing case ends the run, as the official runner's does.

    The official ``import_string`` is prepended (after any ``from __future__`` line), so a program that leans
    on its implicit imports grades as it does officially. Codes as the official runner's: True, -2 wrong
    answer, -3 timeout, -4 runtime error with no output.
    """
    lines = code.split("\n")
    future = [one for one in lines if one.lstrip().startswith("from __future__")]
    body = "\n".join(one for one in lines if not one.lstrip().startswith("from __future__"))
    program = (
        "\n".join(future)
        + "\nimport sys as _qs\n_qs.set_int_max_str_digits(50000)\n"
        + testing_util.import_string
        + "\n"
        + body
    )

    def cpu_limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))

    results: list = []
    for given, want in zip(inputs, outputs, strict=True):
        try:
            done = subprocess.run(
                [sys.executable, "-s", "-S", "-c", program],
                input=given.encode(),
                capture_output=True,
                timeout=timeout * backstop,
                preexec_fn=cpu_limit,
                check=False,
            )
        except subprocess.TimeoutExpired:
            results.append(-3)
            break
        if done.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            results.append(-3)
            break
        pred = done.stdout.decode(errors="replace")
        if done.returncode != 0 and not pred.strip():
            results.append(-4)
            break
        if not _same_output(pred, want):
            results.append(-2)
            break
        results.append(True)
    return results


def run_test(sample: dict, code: str, timeout: int, backstop: int) -> list:
    """The paper's verdicts for one program: a stdin problem in child processes, a functional one officially.

    Args:
        sample: The official ``input_output`` sample.
        code: The program.
        timeout: CPU seconds per case.
        backstop: The wall-clock limit per case, as a multiple of ``timeout``.
    """
    cases = json.loads(sample["input_output"])
    if cases["fn_name"] is None:
        return _run_stdin(code, cases["inputs"], cases["outputs"], timeout, backstop)
    testing_util.signal = _CpuAlarm(signal, backstop)
    results, _ = testing_util.run_test(sample, test=code, timeout=timeout)
    return results
