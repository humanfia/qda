"""The ``qda inspect`` subcommand."""

from __future__ import annotations

import argparse
import json

from ..exit import ERROR, USAGE, refuse
from ..output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION
from .probe import IntakeRefused
from .report import _text, inspect
from .serve import serve_check


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``inspect`` subcommand."""
    p = sub.add_parser("inspect", help="say what a checkpoint is, what it lacks and whether it serves as is")
    p.add_argument("path", help="checkpoint directory, .gguf file or HF repo id (org/name[@rev])")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument(
        "--target",
        default=None,
        help=(
            "serving target, e.g. sm90@0.33.0 (default: the card this process sees, else the target the "
            "checkpoint's build.json records, else the library's)"
        ),
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="also load it in stock vLLM and name every tensor vLLM never read (needs a card and vLLM)",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"the share of the card --serve lets vLLM use (default {DEFAULT_GPU_MEMORY_UTILIZATION})",
    )
    p.set_defaults(run=_run_inspect)


def _run_inspect(a: argparse.Namespace) -> int:
    from ..step import Refused  # here, not at the top (as `plan` does): the target tables load early

    try:
        rep = inspect(a.path, a.target)
    except IntakeRefused as e:
        return refuse(str(e))
    except Refused as e:  # a target name no table serves, from --target or the checkpoint's build.json
        return refuse(str(e), e.fix or None, code=e.code)
    if a.serve:
        try:
            rep["serve"] = serve_check(a.path, a.gpu_memory_utilization)
        except RuntimeError as e:  # the child that loads the checkpoint died before it reported
            return refuse(f"inspect --serve: {e}", code=ERROR)
    print(json.dumps(rep, indent=1) if a.json else _text(rep))
    if problems := _problems(rep):  # the report above has the detail; the refusal is the verdict
        return refuse(f"inspect: {rep['path']}: {'; '.join(problems)}", code=USAGE)
    return 0


def _problems(rep: dict) -> list[str]:
    """Why the inspected checkpoint does not serve as it is: one entry per reason, none when it does."""
    problems = (
        [f"not servable as is on {rep['target']}: {'; '.join(rep['reasons'])}"] if rep["reasons"] else []
    )
    serve = rep.get("serve")
    if serve is not None and not serve["ok"]:
        if serve["error"]:
            problems.append(f"vLLM could not load it: {serve['error']}")
        if serve["unconsumed"]:
            problems.append(
                f"vLLM never read {len(serve['unconsumed'])} tensor(s), e.g. {serve['unconsumed'][0]}"
            )
    return problems
