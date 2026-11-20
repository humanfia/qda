"""fp64 checks of every relation `derive` found, on one copy of each distinct kind of decoder layer.

* norm -> group: scaling the norm's output by ``s`` (its weight, and its bias when it has one) and every
  reader's input columns by ``1/s`` leaves the layer output unchanged (tried for ``x * w`` and zero-centred
  ``x * (1 + w)`` norms; the offset is recorded in ``extra["norm_offset"]``).
* prev -> group: scaling ``prev``'s output rows (and bias) by ``s`` and the group's columns by ``1/s`` leaves
  it unchanged.

A norm relation that fails is dropped and listed in ``extra["dropped"]``; a ``prev`` hint that fails is
cleared. Either way a method that needs the relation is refused instead of computing something else.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, cast

import torch
from torch import nn

from . import LayerGraph, LinearGroup, ModelGraph
from .adapters.fused_experts import in_param
from .derive import decoder_layers

#: relative tolerance: some modules cast to fp32 inside (Qwen3.5 norms do ``1 + w.float()``), so an exact
#: fold still differs by ~1e-7; a wrong relation is off by O(1)
_TOL = 1e-5


def _signature(g: LayerGraph) -> tuple:
    cut = len(g.module) + 1
    rel = lambda n: n[cut:] if n else n  # noqa: E731
    return tuple((tuple(map(rel, x.modules)), x.kind, rel(x.norm), rel(x.prev)) for x in g.groups) + tuple(
        (rel(b.router), b.fused) for b in g.banks
    )


class _InputsCaptured(Exception):
    """The representative layers' inputs are ready for fold verification."""


@torch.no_grad()
def _capture(
    model: nn.Module, layers: nn.ModuleList, want: set[int], *, stop_early: bool = True
) -> dict[int, tuple[tuple, dict]]:
    """Capture representative inputs, stopping once all are ready; retain a full-forward reference."""
    got: dict[int, tuple[tuple, dict]] = {}
    if not want:
        return got

    def mk(i: int):  # noqa: ANN202
        def pre(_m: nn.Module, args: tuple, kwargs: dict) -> None:
            got[i] = (args, kwargs)
            if stop_early and got.keys() == want:
                raise _InputsCaptured

        return pre

    hs = [layers[i].register_forward_pre_hook(mk(i), with_kwargs=True) for i in want]
    try:
        dev = next(model.parameters()).device
        ids = torch.arange(1, 9, device=dev)[None]
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    except _InputsCaptured:
        pass
    finally:
        for h in hs:
            h.remove()
    return got


def _f64(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.float64) if x.is_floating_point() else x.detach().cpu()
    if isinstance(x, tuple | list):
        return type(x)(_f64(v) for v in x)
    if isinstance(x, dict):
        return {k: _f64(v) for k, v in x.items()}
    return x


def _f64_copy(layer: nn.Module) -> nn.Module:
    """A fp64 host copy of `layer`; fused grouped-mm expert kernels have no fp64, so it runs the reference loop."""
    layer = copy.deepcopy(layer).to("cpu", torch.float64)
    for m in layer.modules():
        cfg = cast(Any, getattr(m, "config", None))  # HF config: attributes are dynamic
        if hasattr(cfg, "_experts_implementation"):
            cfg._experts_implementation = "eager"
    return layer


def _out(layer: nn.Module, args: tuple, kwargs: dict) -> torch.Tensor:
    y = layer(*args, **kwargs)
    return y[0] if isinstance(y, tuple | list) else y


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool((a - b).abs().max() <= _TOL * (b.abs().max() + 1e-30))


@torch.no_grad()
def _fold_ok(layer: nn.Module, args: tuple, kwargs: dict, base: torch.Tensor, fold, *fargs) -> bool:  # noqa: ANN001
    """Whether `fold(*fargs)` (an in-place rescaling of `layer`) leaves its output unchanged; undone after."""
    saved = {k: v.clone() for k, v in layer.state_dict().items()}
    try:
        fold(*fargs)
        return _close(_out(layer, args, kwargs), base)
    finally:
        layer.load_state_dict(saved)


def _scale(t: torch.Tensor, s: torch.Tensor, dim: int) -> None:
    shape = [1] * t.dim()
    shape[dim] = -1
    t.mul_(s.reshape(shape))


def _check_layer(
    layer: nn.Module, g: LayerGraph, io: tuple[tuple, dict], readers: dict, gen: torch.Generator
) -> tuple[dict[str, LinearGroup], dict[str, float], list[str]]:
    """Fold checks of one fp64 layer copy: (groups keyed by relative name, norm offsets, dropped norms)."""
    args, kwargs = io
    base = _out(layer, args, kwargs)
    cut = len(g.module) + 1

    def sub(n: str) -> nn.Module:
        return layer.get_submodule(n[cut:])

    def fold_norm(grp: LinearGroup, s: torch.Tensor, off: float) -> None:
        assert grp.norm is not None  # computed only when grp.norm is set
        n = sub(grp.norm)
        w = cast(torch.Tensor, n.weight)  # nn.Module attribute access is dynamic
        w.copy_((w + off) * s - off)
        # y = x_hat * (w + off) + b, so scaling the norm's output by s scales the bias by s too; RMSNorms
        # (the common case) have no bias attribute at all
        b = cast(torch.Tensor | None, getattr(n, "bias", None))
        if b is not None:
            b.mul_(s)
        for r in readers.get(grp.modules[0], grp.modules):
            _scale(getattr(sub(r), in_param(sub(r))), 1 / s, -1)

    def fold_prev(grp: LinearGroup, s: torch.Tensor) -> None:
        assert grp.prev is not None  # computed only when grp.prev is set
        p = sub(grp.prev)
        _scale(cast(torch.Tensor, p.weight), s, 0)
        bias = cast(torch.Tensor | None, p.bias)
        if bias is not None:
            bias.mul_(s)
        for m in grp.modules:
            _scale(cast(torch.Tensor, sub(m).weight), 1 / s, -1)

    fixed, offsets, dropped = {}, {}, []
    for grp in g.groups:
        s = torch.empty(cast(nn.Linear, sub(grp.modules[0])).in_features, dtype=torch.float64).uniform_(
            0.5, 2.0, generator=gen
        )
        new = grp
        if grp.norm:
            ok = next(
                (o for o in (0.0, 1.0) if _fold_ok(layer, args, kwargs, base, fold_norm, grp, s, o)),
                None,
            )
            if ok is None:
                dropped.append(f"{grp.norm} -> {grp.modules[0]}")
                new = dataclasses.replace(new, norm=None)
            elif ok:
                offsets[grp.norm[cut:]] = ok
        if grp.prev and not _fold_ok(layer, args, kwargs, base, fold_prev, grp, s):
            new = dataclasses.replace(new, prev=None)  # a hint, not a derived fact
        fixed[grp.modules[0][cut:]] = new
    return fixed, offsets, dropped


def verify(model: nn.Module, graph: ModelGraph) -> ModelGraph:
    """Check `graph` numerically; return it with failed relations dropped and ``verified=True``."""
    _, layers = decoder_layers(model)
    reps: dict[tuple, int] = {}
    for g in graph.layers:
        reps.setdefault(_signature(g), g.index)
    io = _capture(model, layers, set(reps.values()))
    readers = graph.extra.get("readers", {})
    offsets: dict[str, float] = {}
    dropped: list[str] = []
    checked: dict[tuple, dict[str, LinearGroup]] = {}
    gen = torch.Generator().manual_seed(0)
    for sig, i in reps.items():
        layer = _f64_copy(layers[i])
        checked[sig], off, drop = _check_layer(layer, graph.layers[i], _f64(io[i]), readers, gen)
        offsets.update(off)
        dropped += drop
        del layer
    out_layers = []
    for g in graph.layers:
        fixed = checked[_signature(g)]
        cut = len(g.module) + 1
        groups = tuple(
            dataclasses.replace(
                x,
                norm=x.norm if fixed[x.modules[0][cut:]].norm else None,
                prev=x.prev if fixed[x.modules[0][cut:]].prev else None,
            )
            for x in g.groups
        )
        out_layers.append(dataclasses.replace(g, groups=groups))
    extra = {
        **graph.extra,
        "norm_offset": offsets,
        "dropped": dropped,
        "verified_layers": sorted(reps.values()),
    }
    return dataclasses.replace(graph, layers=tuple(out_layers), verified=True, extra=extra)
