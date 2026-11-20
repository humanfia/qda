"""Packed-weight checks: which quantized modules lack the tensors a loader needs."""

from __future__ import annotations

import re
from typing import Any

from .probe import Source, _tied


def _ignored(ignore: list[str], module: str) -> bool:
    """Whether a compressed-tensors ``ignore`` list names `module`.

    Exactly, by regex, or under another root prefix (``lm_head`` and ``model.language_model.lm_head`` name one module).
    """
    return any(_matches(e, module) or module.endswith("." + e) or e.endswith("." + module) for e in ignore)


def _matches(target: str, module: str) -> bool:
    return re.match(target[3:], module) is not None if target.startswith("re:") else target == module


#: fused expert banks store ``<bank>.experts.<param>_{packed,scale,shape}`` (export.pack_module), not
#: ``.weight_*``; ``up_proj`` is the in projection of a non-gated bank (Nemotron-3-Nano)
_FUSED = (".experts.gate_up_proj", ".experts.up_proj", ".experts.down_proj")


def _expected(m: str, sep: str, f: Any) -> tuple[str, ...]:
    """The stored tensors `f`'s layout needs for module `m`, whose stored parts are named with `sep`."""
    if f.weight_type == "fp8":
        return (f"{m}.weight", f"{m}.weight_scale")
    if f.weight_type == "e2m1":
        return tuple(f"{m}{sep}{s}" for s in ("packed", "scale", "global_scale"))
    return tuple(f"{m}{sep}{s}" for s in ("packed", "scale", "shape"))


def unpacked(src: Source, rep: dict[str, Any]) -> list[str]:
    """Modules a quantized config claims whose stored tensors the loader cannot find.

    Each module is checked against the layout of the groups claiming it: ``weight`` and ``weight_scale`` for
    FP8 W8A8, ``weight_packed``, ``weight_scale`` and ``weight_global_scale`` for NVFP4, and the packed codes,
    scale and shape for an integer format. A group of a layout this library does not build (or no group at
    all) claims nothing.

    The candidates include the tied head, which stores no tensor of its own; embeddings are no Linear, and a
    module listed here makes the loader fail.
    """
    from .formats import group_format  # here, not at the top: formats imports this module

    cfg = src.jsons.get("config.json") or {}
    tied = _tied(cfg)
    mods = {
        k.rpartition(".")[0]: ".weight_"
        for k, (_, s) in src.tensors.items()
        if k.endswith(".weight") and len(s) == 2
    }
    mods |= {k[: -len(".weight_packed")]: ".weight_" for k in src.tensors if k.endswith(".weight_packed")}
    mods |= {
        k.rpartition("_")[0]: "_"
        for k in src.tensors
        if k.rpartition("_")[0].endswith(_FUSED) and k.rpartition("_")[2] in ("packed", "scale", "shape")
    }
    if tied and not any(m.rpartition(".")[2] == "lm_head" for m in mods):
        mods["lm_head"] = ".weight_"
    targets = [(t, group_format(g)) for g in rep["groups"].values() for t in g["targets"]]
    out = []
    for m, sep in sorted(mods.items()):
        if "embed" in m.rpartition(".")[2] or _ignored(rep["kept_bf16"], m):
            continue
        claimed = [f for t, f in targets if f is not None and (t == "Linear" or _matches(t, m))]
        if claimed and not any(all(k in src.tensors for k in _expected(m, sep, f)) for f in claimed):
            out.append(m)
    return out
