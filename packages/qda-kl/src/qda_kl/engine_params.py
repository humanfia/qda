"""The engine parameters one reference is built with, defined once.

`qda-kl ref`, `qda-kl kl` and `qda-agent prepare` build references of the same original, and the reference cache
keys an entry by those settings exactly (:func:`qda_kl.refcache.key_payload`). Two consequences, both of them
one fact with several owners:

- a caller that names a setting another one does not builds a second copy of the same reference
  (`qda_kl.refcache` now fills every setting a caller leaves out from the build's own signature, so this module
  carries only the four parameters the engine flags set);
- a caller with no flag for a setting leaves the user no lever for it (`qda-kl kl` could not lower vLLM's share
  of a shared card, and the engine died at start).

This module is the one place that adds the flags and reads them into the four values (`engine_parameters`), so
`qda-kl ref` and `qda-kl kl` offer and mean the same parameters; the default they fall back on is every engine's
(`qda_quant.output.vllm_env.DEFAULT_GPU_MEMORY_UTILIZATION`). `vllm.LLM` reads `gpu_memory_utilization` as a
fraction of the card's *total* memory, so the default asks for memory
that is not free on a card another job partly holds: it has its own flag rather than riding in `--serve-args`,
and a `--serve-args` that names it is refused instead of silently winning.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from qda_quant.output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_quant.step import Refused


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the engine parameters every command that builds a reference offers, defined here once."""
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        metavar="FRACTION",
        help=f"vLLM's share of the card (default {DEFAULT_GPU_MEMORY_UTILIZATION}); lower it when other jobs "
        "share the card",
    )
    parser.add_argument(
        "--serve-env",
        type=json.loads,
        default=None,
        metavar="JSON",
        help='engine environment as JSON, e.g. \'{"TOKENIZERS_PARALLELISM": "false"}\'',
    )
    parser.add_argument(
        "--serve-args",
        type=json.loads,
        default=None,
        metavar="JSON",
        help="extra vllm.LLM arguments as JSON, e.g. '{\"enable_expert_parallel\": true}'",
    )
    parser.add_argument(
        "--nccl-nvls",
        action="store_true",
        help="let the engine's ranks join NCCL's NVLS collectives (default off)",
    )


def engine_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """The engine parameters of a parsed command line, checked against each other in one place.

    Raises:
        Refused: ``--serve-args`` names ``gpu_memory_utilization``; the flag is its one owner.
    """
    serve_args = getattr(args, "serve_args", None)
    if isinstance(serve_args, dict) and "gpu_memory_utilization" in serve_args:
        raise Refused(
            "--serve-args names gpu_memory_utilization; pass --gpu-memory-utilization FRACTION instead"
        )
    return {
        "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION),
        "serve_env": getattr(args, "serve_env", None),
        "serve_args": serve_args,
        "nccl_nvls": bool(getattr(args, "nccl_nvls", False)),
    }


def topk_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The ``scorer.topk`` entries of the same parameters: a reading starts its own engine on the same card.

    Only what the command line names is returned, so a score config keeps the checkpoint's own serving settings
    for everything the user did not set.
    """
    got: dict[str, Any] = {
        "tp": getattr(args, "tp", DEFAULT_TP),
        "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION),
    }
    for name in ("serve_env", "serve_args"):
        if (value := getattr(args, name, None)) is not None:
            got[name] = value
    if getattr(args, "nccl_nvls", False):
        got["nccl_nvls"] = True
    return got
