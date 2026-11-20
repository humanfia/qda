"""Fused routed experts (transformers 5): one module, two 3-D weight parameters per bank.

The stored layout is declared by the class, not implied by the names. ``@use_experts_implementation`` gives
every bank ``is_transposed`` (weights ``[E, in, out]`` rather than ``[E, out, in]``), ``has_bias``
(``gate_up_proj_bias`` / ``down_proj_bias``) and a ``_apply_gate`` that turns the ``[.., 2I]`` projection
output into the down input (contiguous halves, or interleaved as gpt-oss stores it). The helpers below read
that declaration: `in_weight` / `down_weight` return the canonical ``[E, out, in]`` form, `gate` the module's
own gated activation, `bias` its bias. Steps addressing a bank use them instead of the stored shape.

Canonically, ``gate_up_proj`` [E, 2I, D] holds gate rows ``[:I]`` then up rows ``[I:]``; ``down_proj`` [E, D, I].
The two linear inputs per expert are the routed tokens ``x`` (gate/up) and, by default, ``act(gate) * up``
(down). A bank whose model forms the down input otherwise declares how in its own ``down_input``
(`down_input`).

A non-gated bank (``has_gate`` false, e.g. Nemotron-3-Nano) stores ``up_proj`` [E, I, D] instead of
``gate_up_proj`` and forms the down input as ``act(up)``. `in_param` names the parameter that reads the routed
tokens, so steps address either layout the same way.

Generic derivation sees such a module only as a stack of experts it cannot open, so it leaves it out of the
graph. `FUSED_EXPERTS` adds it back: as an `ExpertBank`, and as a reader of the tensor its router reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import torch
from torch import nn

from ...step import Refused
from .. import ExpertBank, ExportLayout, LayerGraph, ModelGraph
from ..derive import router_of, routes_itself, text_config

#: parameter of a gated bank that reads the routed tokens, and the one that reads act(gate) * up
IN, DOWN = "gate_up_proj", "down_proj"
#: parameter of a non-gated bank that reads the routed tokens
IN_UP = "up_proj"


def has_gate(m: nn.Module) -> bool:
    """Whether `m`'s experts multiply a gate into their up projection (transformers' ``has_gate``)."""
    return bool(getattr(m, "has_gate", True))


def in_param(m: nn.Module) -> str:
    """The parameter of `m` that multiplies its input: ``weight`` for a Linear, the bank's input projection.

    A gated fused bank stores ``gate_up_proj``; a non-gated one (``up_proj`` + ``down_proj``) stores ``up_proj``.
    """
    if hasattr(m, IN):
        return IN
    if isinstance(getattr(m, IN_UP, None), nn.Parameter):
        return IN_UP
    return "weight"


def is_bank(m: nn.Module) -> bool:
    """True for a module that stores its experts as fused 3-D parameters, gated or not."""
    p, d = getattr(m, in_param(m), None), getattr(m, DOWN, None)
    return (
        in_param(m) != "weight"
        and isinstance(p, nn.Parameter)
        and isinstance(d, nn.Parameter)
        and p.dim() == 3
        and d.dim() == 3
    )


def transposed(m: nn.Module) -> bool:
    """Whether `m` stores a bank weight ``[E, in, out]`` (transformers' ``is_transposed``)."""
    return bool(getattr(m, "is_transposed", False))


def in_weight(m: nn.Module) -> torch.Tensor:
    """The input-projection parameter of `m` as ``[E, R, D]``: ``gate_up_proj`` [E, 2I, D], or ``up_proj`` [E, I, D]."""
    return _out_first(m, in_param(m))


def down_weight(m: nn.Module) -> torch.Tensor:
    """``down_proj`` of `m` as ``[E, D, I]``."""
    return _out_first(m, DOWN)


def _out_first(m: nn.Module, param: str) -> torch.Tensor:
    w = cast(torch.Tensor, getattr(m, param))
    return w.transpose(1, 2) if transposed(m) else w


def gate(m: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """`m`'s gated activation, ``[.., 2I] -> [.., I]``: halves or interleaved, as the module declares."""
    fn = getattr(m, "_apply_gate", None)
    if not callable(fn):
        raise Refused(f"fused expert bank {type(m).__name__}: no _apply_gate to turn its gate/up into I")
    return cast(Callable[[torch.Tensor], torch.Tensor], fn)


def bias(m: nn.Module, param: str) -> torch.Tensor | None:
    """``<param>_bias`` ``[E, out]`` of `m`, when it has one."""
    return cast(torch.Tensor | None, getattr(m, f"{param}_bias", None))


def down_input(m: nn.Module, gate_up: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """The input of bank `m`'s down projection, from its input projection's output, as the model forms it.

    A non-gated bank returns ``act_fn(up)``. A bank whose model does it otherwise (an fp32 activation, a clamp,
    router weights applied before the down projection) defines ``down_input(gate_up, weights)`` with the same
    signature. Otherwise the module's own ``_apply_gate`` (contiguous halves, or interleaved as gpt-oss stores
    it) is used; without one, the default is ``act_fn(gate) * up``.

    Args:
        m: the bank module.
        gate_up: ``[..., 2I]`` gate/up outputs, gate first (``[..., I]`` for a non-gated bank).
        weights: ``[...]`` each row's router weight, or None for rows no router chose (AWQ's every-token pass).
    """
    own = getattr(m, "down_input", None)
    if own is not None:
        return own(gate_up, weights)
    if not has_gate(m):
        return cast(Callable[[torch.Tensor], torch.Tensor], m.act_fn)(gate_up)
    apply_gate = getattr(m, "_apply_gate", None)
    if callable(apply_gate):
        return cast(torch.Tensor, apply_gate(gate_up))
    gate, up = gate_up.chunk(2, dim=-1)
    act_fn = cast(Callable[[torch.Tensor], torch.Tensor], m.act_fn)
    return act_fn(gate) * up


def split(m: nn.Module) -> tuple[int, int]:
    """(rows of gate, rows of up) in the input projection: ``(0, I)`` for a non-gated bank."""
    rows = in_weight(m).shape[1]
    return (0, rows) if not has_gate(m) else (rows // 2, rows - rows // 2)


def per_expert_layout(name: str, shape: torch.Size) -> ExportLayout | None:
    """Standard serving fields for canonical gate/up halves, a non-gated up, and down projections.

    Both stock engines load per-expert compressed-tensors names. A fused packed
    name can reach SGLang's float-bank path and be skipped without a load error.
    The shape is the canonical codes shape after shared transpose handling.
    """
    for param, projections in (
        (IN, ("gate_proj", "up_proj")),
        (IN_UP, ("up_proj",)),
        (DOWN, ("down_proj",)),
    ):
        if not name.endswith(f".experts.{param}"):
            continue
        base = name[: -len(param)]
        rows = shape[1] // len(projections)
        pieces = tuple(
            (
                f"{base}{expert}.{projection}",
                (expert, slice(index * rows, (index + 1) * rows)),
                {
                    field: f"{base}{expert}.{projection}.weight_{field}"
                    for field in ("packed", "scale", "shape", "zero_point")
                },
            )
            for expert in range(shape[0])
            for index, projection in enumerate(projections)
        )
        return ExportLayout(pieces, (name,))
    return None


class FusedExperts:
    """The ``fused_experts`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True if `model` has a fused expert bank and no other registered adapter fixes the model's graph.

        This is the fallback: an architecture whose banks need more (float weights, other readers) adds them
        from its own adapter, which may reuse `adjust`. An adapter that only declares the residual stream
        (`StreamDeclaration`, Qwen3.5 MoE's) fixes nothing, so the bank still needs this one.
        """
        from . import ADAPTERS
        from .stream import StreamDeclaration

        if not any(is_bank(m) for m in model.modules()):
            return False
        return not any(
            a is not self and not isinstance(a, StreamDeclaration) and a.matches(model)
            for _, a in ADAPTERS.items()
        )

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Add every fused bank to its layer's banks and to the readers of its router's input.

        Raises:
            Refused: a bank has no router beside it (its inputs, and so its readers, are unknown).
        """
        cfg = text_config(model)
        top_k = int(getattr(cfg, "num_experts_per_tok", 0) or getattr(cfg, "top_k", 0))
        readers = dict(graph.extra["readers"])
        layers = tuple(_layer(model.get_submodule(g.module), g, top_k, readers) for g in graph.layers)
        return replace(graph, layers=layers, extra={**graph.extra, "readers": readers})

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """Write canonical fused banks under the expert names both stock engines load."""
        return per_expert_layout(name, shape)

    @staticmethod
    def awq_skip_aliases(names: set[str]) -> set[str]:
        """Literal standard fused linear names when every checkpoint shard remains float."""
        aliases = set()
        for name in names:
            if name.endswith(".gate_proj"):
                base = name.removesuffix(".gate_proj")
                if base + ".up_proj" in names:
                    aliases.add(base + ".gate_up_proj")
            if name.endswith(".q_proj"):
                base = name.removesuffix(".q_proj")
                if {base + ".k_proj", base + ".v_proj"} <= names:
                    aliases.add(base + ".qkv_proj")
        return aliases


def _layer(layer: nn.Module, g: LayerGraph, top_k: int, readers: dict[str, tuple[str, ...]]) -> LayerGraph:
    """`g` with the fused banks of decoder layer `layer`; `readers` gets each bank right after its router."""
    mods = dict(layer.named_modules(prefix=g.module))
    banks = []
    for k, m in mods.items():
        if not is_bank(m):
            continue
        gated = has_gate(m)
        # the bank must declare how its gate/up becomes the down input: its own gate (`_apply_gate`, as
        # transformers declares it), its own `down_input` (a bank that forms it otherwise), or, non-gated,
        # its activation (the down input is `act_fn(up)`; `down_input` falls back to it)
        if (
            gated
            and not callable(getattr(m, "_apply_gate", None))
            and not callable(getattr(m, "down_input", None))
        ):
            raise Refused(
                f"fused expert bank {k}: neither _apply_gate nor down_input, so its gate/up layout is unknown"
            )
        if (
            not gated
            and not callable(getattr(m, "act_fn", None))
            and not callable(getattr(m, "down_input", None))
        ):
            raise Refused(
                f"fused expert bank {k}: a non-gated bank with neither act_fn nor down_input cannot form its "
                "down input"
            )
        if getattr(m, "has_post_expert_norm", False):
            raise Refused(f"fused expert bank {k}: a post expert norm is not supported")
        router = router_of(mods, k, in_weight(m).shape[0])
        if router is None:
            raise Refused(
                f"fused expert bank {k}: no router beside it (a sibling with one weight row per expert)"
            )
        banks.append(
            ExpertBank(
                k,
                router,
                top_k,
                fused=True,
                transposed=transposed(m),
                gated=gated,
                custom_router=routes_itself(mods[router]),
            )
        )
        for key, rs in readers.items():  # the block hands the router and the bank the same tensor
            if router in rs and k not in rs:
                at = rs.index(router) + 1
                readers[key] = (*rs[:at], k, *rs[at:])
    return replace(g, banks=g.banks + tuple(banks)) if banks else g


#: The adapter object the registry and the ``qda.arch_adapters`` entry point hold.
FUSED_EXPERTS = FusedExperts()
