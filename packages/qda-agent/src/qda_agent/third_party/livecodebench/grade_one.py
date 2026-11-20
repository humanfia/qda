"""Run one LiveCodeBench program inside the sandbox (QDA's entry point, not upstream code).

The payload arrives on stdin as JSON: ``code``, ``cases``, ``fn_name`` (None for a stdin problem),
``timeout``, ``protocol``, ``wall_backstop`` and a ``nonce``. Under ``official`` all cases run in one official
``testing_util.run_test`` call, preserving the compiled program's state between cases and the official
per-case timer; under ``paper`` they run through :mod:`paper`. The result is written to stdout after a
nonce-marked prefix; the parent reads the last marked line, so the program's own prints cannot be mistaken for
it.

The arguments are the readiness descriptor and, outside the sandbox (``sandbox: none``), ``--no-process-cap``.
"""

from __future__ import annotations

import json
import os
import resource
import sys

MARK = "@@qda-lcb-result@@"

#: Asks for the process cap to be left alone; the host names it when it runs this grader unsandboxed.
NO_PROCESS_CAP = "--no-process-cap"

#: One case may address at most 4 GiB, write at most 64 MiB per file, and hold at most 64 processes.
MEMORY_CAP = 4 * 1024**3
FILE_CAP = 64 * 1024**2
PROCESS_CAP = 64


def limit_resources(processes: bool = True) -> None:
    """Cap the process before any model-written code runs: address space, file size, processes, core dumps.

    The hard limit equals the soft one, so the sandboxed code cannot raise it. Processes are counted per
    sandbox: bubblewrap's user namespace gives the case its own process count, so the cap bounds a fork bomb
    without limiting the host session or keeping the sandbox from starting. Outside the sandbox that count is
    the whole login's, where the cap would refuse the program's first fork, so ``processes`` turns it off.
    """
    caps = [(resource.RLIMIT_AS, MEMORY_CAP), (resource.RLIMIT_FSIZE, FILE_CAP), (resource.RLIMIT_CORE, 0)]
    if processes:
        caps.append((resource.RLIMIT_NPROC, PROCESS_CAP))
    for what, value in caps:
        resource.setrlimit(what, (value, value))


def _passed(one: object) -> bool:
    """Whether one case result is LiveCodeBench's pass (``True``, or NumPy's bool)."""
    if isinstance(one, bool):
        return one
    item = getattr(one, "item", None)
    try:
        value = item() if item is not None else None
    except Exception:  # noqa: BLE001 - any object that refuses item() is not a pass
        return False
    return isinstance(value, bool) and value


def _jsonable(one: object) -> object:
    """One case result as JSON: bools stay bools, NumPy bools become bools, codes stay ints."""
    if isinstance(one, bool):
        return one
    item = getattr(one, "item", None)
    try:
        value = item() if item is not None else one
    except Exception:  # noqa: BLE001
        return repr(one)
    return value if isinstance(value, (bool, int, float, str)) else repr(value)


def main() -> int:
    """Read the payload, run the cases through the runner its protocol names, write the result line."""
    args = sys.argv[1:]
    limit_resources(processes=NO_PROCESS_CAP not in args)  # before NumPy and the code are imported or run
    import paper
    from testing_util import run_test

    payload = json.load(sys.stdin)
    ready = [arg for arg in args if arg != NO_PROCESS_CAP]
    if ready:
        ready_fd = int(ready[0])
        os.write(ready_fd, b"1")
        os.close(ready_fd)  # generated code cannot alter the trusted startup signal
    sample = {
        "input_output": json.dumps(
            {
                "inputs": [case.get("input", "") for case in payload["cases"]],
                "outputs": [case.get("output", "") for case in payload["cases"]],
                "fn_name": payload["fn_name"],
            }
        )
    }
    try:
        if payload["protocol"] == "paper":
            results, metadata = (
                paper.run_test(sample, payload["code"], int(payload["timeout"]), payload["wall_backstop"]),
                {},
            )
        else:
            results, metadata = run_test(sample, test=payload["code"], timeout=int(payload["timeout"]))
    except BaseException as why:  # noqa: BLE001 - the generated program may raise anything
        results, metadata = [-4], {"error_code": -4, "error_message": f"Error during testing: {why}"}
    results = list(results or [])
    out = {
        "passed": bool(results) and all(_passed(one) for one in results),
        "timed_out": -3 in results or (isinstance(metadata, dict) and metadata.get("error_code") == -3),
        "results": [_jsonable(one) for one in results],
        "metadata": metadata if isinstance(metadata, dict) else {"value": repr(metadata)},
    }
    sys.stdout.write(f"\n{MARK}{payload['nonce']}{json.dumps(out)}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
