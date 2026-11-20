"""qda-quant: composable post-training quantization, built for speed, served by stock vLLM.

    import qda_quant as qda
    recipe = qda.Recipe.from_yaml(open("qwen3-8b-w4a16.yaml").read())
    out = qda.build("Qwen/Qwen3-8B", recipe, format="w4a16", target="h200", out_dir="ckpt/")

    # or everything from one config file (what `qda build CONFIG.yaml` runs)
    out = qda.BuildConfig.from_file("build.yaml", overrides=["out=ckpt2/"]).build()

Five concepts: `Format` (what is stored), `Target` (where it is served), `Step` (one method),
`Recipe` (steps + calibration data) and the engine (the only code that runs the model).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .arch import KEEP_ROLES

if TYPE_CHECKING:
    import torch

    from .calib.sources import CalibSource, FileSource
    from .config import BuildConfig
    from .engine.stats import HessianSetting
    from .format import PRESETS, Format
    from .output.exporters import Exporter
    from .recipe import Recipe
    from .step import REGISTRY, LayerContext, Refused, Step
    from .target import KernelPlan, Target

__version__ = "0.1.0"

#: Public names and the module that defines each. Imported on first use, so ``import qda_quant`` (and so any
#: ``qda_quant.<module>`` import, e.g. the torch-free plugin registry other packages use) does not import torch.
#: The built-in methods register themselves on the first `REGISTRY` lookup (`step.REGISTRY`'s ``builtins``).
_LAZY = {
    "BuildConfig": "config",
    "PRESETS": "format",
    "Format": "format",
    "CalibSource": "calib.sources",
    "FileSource": "calib.sources",
    "Recipe": "recipe",
    "REGISTRY": "step",
    "LayerContext": "step",
    "Refused": "step",
    "Step": "step",
    "KernelPlan": "target",
    "Target": "target",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in _LAZY:
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "PRESETS",
    "REGISTRY",
    "BuildConfig",
    "CalibSource",
    "FileSource",
    "Format",
    "KernelPlan",
    "LayerContext",
    "Recipe",
    "Refused",
    "Step",
    "Target",
    "build",
]


def build(  # noqa: A002
    model: str | Path,
    recipe: Recipe | str,
    *,
    format: Format | str | None = None,  # noqa: A002
    target: Target | str | None = None,
    out_dir: str | Path,
    devices: Sequence[int | str | torch.device] | None = None,
    keep_intermediate: bool = False,
    batch_size: int = 1,
    residency: str | None = None,
    nccl_nvls: bool = False,
    dtype: torch.dtype | None = None,
    hessian: HessianSetting | dict | str | None = None,
    ignore: Sequence[str] = KEEP_ROLES,
    format_overrides: Sequence[tuple[str, str]] = (),
    exporter: Exporter | str | dict | None = None,
    layer_cache: dict | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build a quantized checkpoint (compressed-tensors, loadable by stock vLLM) into `out_dir`.

    Args:
        model: HF repo id or local path of the original (BF16 / FP8) model.
        recipe: a `Recipe`, or a path / YAML text of one. A relative path in a YAML file is relative to that
            file's directory.
        format: a `Format` or its name (``"w4a16"``, ``"w4a8"``, ``"w3a16"``, ``"w4a16-asym-g128"``); a packed
            checkpoint as `model` has its own (then `format`, if given, must agree).
        target: a `Target` or its name (``"h200"``, ``"sm90@0.33.0"``); default sm90 + vLLM 0.33.0.
        out_dir: where the checkpoint, ``build.json`` (timings, versions) and ``resolved.yaml`` (the config that
            repeats this build, see `BuildConfig`) go. It must be new or empty, and appears when the build is
            complete.
        devices: the cards to use, any number this process can see (default: every visible card), or ``"cpu"``
            to build without one. A job that holds cards (a GPU lease, or visible cards) refuses a CPU-only
            list when the recipe has a step that runs on a card (GPTQ, AWQ, ...): name the job's cards, or
            leave `devices` unset to use them.
        keep_intermediate: keep a staged build's intermediate checkpoints in ``out_dir/.stages``.
        batch_size: calibration rows per forward (1 = llm-compressor's numerics).
        residency: a weight store (`qda_quant.engine.stores`: ``"resident"``, ``"stream"``, ``"disk"``, ...)
            instead of the planner's choice.
        nccl_nvls: let the card processes join NCCL's NVLS (multicast) collectives; off by default (see
            `qda_quant.engine.build`).
        dtype: dtype the model is loaded in (default: bf16, the published rotated bases' form; ``fp32`` keeps
            a `rotate` step's full precision into the quantizer -- a config key, see `BuildConfig`).
        hessian: Hessian arithmetic and accumulation order: a `HessianSetting`, ``{"dtype", "buffer_rows"}``, or
            ``"fast"`` (default: bf16 per sample).
        ignore: linears that stay bf16: roles (`qda_quant.arch.KEEP_ROLES`, all by default), exact module
            names and ``re:`` patterns (`qda_quant.config.BuildConfig`); ``[]`` quantizes them too.
        format_overrides: ``(pattern, format)`` rules that store chosen layer units in another format; a
            pattern is an exact module name or ``re:`` and a regex matched from the start, and must name a whole
            solve unit (see `qda_quant.config.BuildConfig`). The body `format` stores every unit no rule names.
        exporter: what the build writes (`qda_quant.output.exporters`): an `Exporter`, a name or
            ``{name: settings}``; default ``compressed_tensors``.
        layer_cache: ``{"dir": ..., "match": "exact" | "formats"}``: reuse and keep solved layers
            (`qda_quant.engine.layer_cache`); one card only. Default: solve every layer.
        log: progress sink (rank 0 only on multi-card builds).

    Returns:
        `out_dir`, as an absolute path.
    """
    import torch

    from .engine import build as _build

    return _build(
        model,
        recipe,
        format=format,
        target=target,
        out_dir=out_dir,
        devices=devices,
        keep_intermediate=keep_intermediate,
        batch_size=batch_size,
        residency=residency,
        nccl_nvls=nccl_nvls,
        dtype=torch.bfloat16 if dtype is None else dtype,
        hessian=hessian,
        ignore=ignore,
        format_overrides=format_overrides,
        exporter=exporter,
        layer_cache=layer_cache,
        log=log,
    )
