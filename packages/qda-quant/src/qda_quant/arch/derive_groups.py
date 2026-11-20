"""Layer groups: linears that share an input, the norm that made it, and the ``ignore`` entries that keep them float."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, TypeGuard, cast

import torch
from torch import nn

from ..step import Refused
from . import (
    GDN_GATES,
    KEEP_ROLES,
    MIN_QUANT_ROWS,
    ExpertBank,
    GroupKind,
    LayerGraph,
    LinearGroup,
    ModelGraph,
)
from .derive_experts import _banks
from .derive_model import _num_experts
from .derive_trace import _Trace


def _kind(rel: str, is_in: bool) -> GroupKind:
    top = rel.split(".")[0].lower()
    if "linear_attn" in top:
        return "linear_attn_in" if is_in else "linear_attn_out"
    if "attn" in top or "attention" in top:
        return "attn_in" if is_in else "attn_out"
    if any(t in top for t in ("mlp", "ffn", "feed_forward", "moe")):
        return "mlp_in" if is_in else "mlp_out"
    return "other"


def _readers(mods: dict[str, nn.Module], calls: list, tid: int, width: int) -> tuple[str, ...]:
    """Every module whose input is tensor `tid` and that owns a weight over it (linears, routers).

    Stacked expert banks are not among them: their adapter adds them, as it knows which parameter reads `tid`.
    """
    out = []
    for k, i, _o in calls:
        if i != tid or k in out:
            continue
        p = getattr(mods.get(k), "weight", None)
        if isinstance(p, nn.Parameter) and p.dim() >= 2 and p.shape[-1] == width:
            out.append(k)
    return tuple(out)


def _layer(
    idx: int,
    prefix: str,
    layer: nn.Module,
    tr: _Trace,
    cfg: Any,
    readers: dict,
    keep: frozenset,
    kept: dict,
    ignored: dict,
) -> LayerGraph:
    pre = prefix + "."
    mods = dict(layer.named_modules(prefix=prefix))
    n_exp = _num_experts(cfg)
    banks, bank_owned = _banks(mods, cfg, n_exp)
    calls = [(k, i, o) for k, i, o in tr.calls if k.startswith(pre)]
    out_of: dict[int, str] = {}
    for k, _i, o in calls:
        out_of.setdefault(o, k)  # first producer of a tensor
    seen: dict[int, list[str]] = {}
    order: list[int] = []
    patterns = tuple(sorted(k for k in keep if k not in KEEP_ROLES))
    for k, i, _o in calls:
        m = mods.get(k)
        if not isinstance(m, nn.Linear) or k in bank_owned or (n_exp and m.out_features == n_exp):
            continue
        if m.out_features < MIN_QUANT_ROWS:  # scalar gates (e.g. shared_expert_gate) stay in bf16
            continue
        if GDN_GATES in keep and _head_gate(k, m, mods):
            kept.setdefault(GDN_GATES, []).append(k)
            if hits := ignore_hits(patterns, k):  # a pattern may name what a role keeps too
                ignored[k] = (hits, "linear_attn_in")
            continue
        if i not in seen:
            seen[i] = []
            order.append(i)
        if k not in seen[i]:
            seen[i].append(k)
    groups = []
    for i in order:
        names = tuple(seen[i])
        src = out_of.get(i)
        width = cast(nn.Linear, mods[names[0]]).in_features
        nm = mods.get(src) if src else None
        # a fold anchor scales channels one-to-one: per-head norms (GDN's gated norm) are not
        norm = src if _is_norm(nm) and cast(torch.Tensor, nm.weight).shape[-1] == width else None
        is_in = norm is not None and norm.rpartition(".")[0] == prefix
        # A group fed by a norm inside a submodule is internal to that block (MLA's kv_b_proj reads
        # kv_a_layernorm): it neither reads nor writes the residual stream, so it is not an output of it.
        internal = norm is not None and not is_in
        prev = None
        if not is_in:
            prev = _prev_candidate(names, mods, calls)
        kind = "other" if internal else _kind(names[0][len(pre) :], is_in)
        hits = {k: h for k in names if (h := ignore_hits(patterns, k))}
        if hits and kind in ("attn_in", "mlp_in") and len(hits) < len(names):
            k, h = next(iter(hits.items()))
            raise Refused(
                f"ignore: {h[0]!r} keeps {k} float but not the rest of {list(names)}: vLLM serves them as "
                "one fused linear with one scheme; ignore all of them or none"
            )
        for k, h in hits.items():
            ignored[k] = (h, kind)
        names = tuple(k for k in names if k not in hits)
        if not names:
            continue
        groups.append(LinearGroup(names, kind, norm, prev))
        readers[names[0]] = _readers(mods, calls, i, width)
    for k, _i, o in calls:  # every decoder-layer norm, also those feeding only a router and expert bank
        m = mods.get(k)
        if _is_norm(m) and k.rpartition(".")[0] == prefix and k not in readers:
            readers[k] = _readers(mods, calls, o, cast(torch.Tensor, m.weight).shape[0])
    return LayerGraph(idx, prefix, tuple(groups), banks)


def _head_gate(name: str, m: nn.Linear, mods: dict[str, nn.Module]) -> bool:
    """Whether linear `name` is a Gated DeltaNet per-head projection: at most two outputs per value head.

    Only a block that declares ``num_v_heads`` (the Gated DeltaNet modules of transformers) has such projections.
    """
    n = getattr(mods.get(name.rpartition(".")[0]), "num_v_heads", None)
    return isinstance(n, int) and m.out_features <= 2 * n


def _is_norm(m: nn.Module | None) -> TypeGuard[nn.Module]:
    return m is not None and "norm" in type(m).__name__.lower() and hasattr(m, "weight")


def _prev_candidate(names: tuple[str, ...], mods: dict[str, nn.Module], calls: list) -> str | None:
    """The last linear called before the group in the same parent, if its output width is the group's input.

    A hint only: `verify` keeps it only if scaling its output rows by s and the group's input columns by 1/s
    leaves the layer output unchanged.
    """
    parent = names[0].rpartition(".")[0]
    first = next(n for n, (k, _i, _o) in enumerate(calls) if k == names[0])
    for k, _i, _o in reversed(calls[:first]):
        m = mods.get(k)
        if isinstance(m, nn.Linear) and k.rpartition(".")[0] == parent:
            return k if m.out_features == mods[names[0]].in_features else None
    return None


def ignore_hits(patterns: tuple[str, ...], name: str) -> tuple[str, ...]:
    """The ``ignore`` patterns that match module `name`.

    A pattern is an exact name, or ``re:`` and a regex that `re.match`es the name from its start (the rule of
    compressed-tensors, so the exported ``ignore`` reads the same way).
    """
    return tuple(
        p for p in patterns if (re.match(p[3:], name) is not None if p.startswith("re:") else p == name)
    )


def _check_ignored(
    graph: ModelGraph,
    keep: frozenset[str],
    ignored: dict[str, tuple[tuple[str, ...], GroupKind]],
) -> tuple[dict[str, tuple[str, GroupKind]], ModelGraph]:
    """`ignore` entries that fall on routed experts, and the graph without the banks they keep float.

    Returns ``(graph.extra["ignored"], graph)``: the map (module -> (the first pattern that keeps it float, the
    kind of its group)) and the graph whose layers no longer hold a bank that ``ignore`` keeps float in full —
    such a bank is no solve unit, gets no statistics and no plan, and its weights are copied as they are.
    The kept-float banks' roots are listed in ``graph.extra["float_banks"]``.

    An ``ignore`` may keep a whole routed bank float (the stock engines serve a float bank beside quantized
    ones) but never part of one: engines serve a bank with one scheme, so a pattern that names some of its
    experts and not the rest is refused. A pattern that names the output head is accepted: no layer step
    quantizes it, so the entry asks for what the build already does. `engine.planning.compile_plan` refuses a
    head pattern beside a head step, which would store it instead.

    Raises:
        Refused: a pattern matches part of a routed bank, or matches no linear the build would quantize (and
            no head).
    """
    from ..solve.units import bank_module

    patterns = tuple(sorted(k for k in keep if k not in KEEP_ROLES))
    layers = []
    float_banks: list[str] = []
    for lg in graph.layers:
        banks = []
        for bank in lg.banks:
            root = bank_module(bank)
            parts = _bank_parts(bank)  # linear -> True if it reads the residual stream (a gate or up)
            hits = {n: h for n in parts if (h := ignore_hits(patterns, n))}
            # a whole bank is named by its module, or by one pattern that names every one of its linears: a bank
            # is served with one quantization scheme, so there is no half-float bank (several patterns may each
            # cover it; every one of them is a cover, none of them is part of it)
            full = {
                p for p in patterns if ignore_hits((p,), root) or all(p in hits.get(n, ()) for n in parts)
            }
            cover = min(full) if full else None
            if not hits and cover is None:
                banks.append(bank)
                continue
            if cover is None:
                n, h = next(iter(hits.items()))
                missing = sorted(set(parts) - set(hits))
                raise Refused(
                    f"ignore: {h[0]!r} matches the routed expert {n}"
                    + (f" but not {missing[0]}" if missing else " but not the whole bank")
                    + "; a bank of experts is served with one quantization scheme, so ignore keeps a whole "
                    "bank float or none of it; name every linear of the bank"
                )
            if partial := sorted({p for n in hits for p in hits[n]} - full):
                raise Refused(
                    f"ignore: {partial[0]!r} matches part of the routed bank {root} but not all of it; a bank "
                    "of experts is served with one quantization scheme, so ignore keeps a whole bank float or "
                    "none of it; name every linear of the bank"
                )
            for n, is_in in parts.items():
                ignored[n] = (tuple(sorted(full)), "mlp_in" if is_in else "mlp_out")
            float_banks.append(root)
        layers.append(replace(lg, banks=tuple(banks)))
    graph = replace(graph, layers=tuple(layers))
    graph.extra["float_banks"] = tuple(float_banks)
    matched = {p for hits, _kind in ignored.values() for p in hits}
    for p in patterns:
        if p not in matched and not (graph.lm_head and ignore_hits((p,), graph.lm_head)):
            raise Refused(
                f"ignore: {p!r} matches no linear the build would quantize; an entry is a role "
                f"({', '.join(KEEP_ROLES)}), an exact module name as `qda plan` prints it, or 're:' and a regex"
            )
    return {k: (hits[0], kind) for k, (hits, kind) in ignored.items()}, graph


def _bank_parts(bank: ExpertBank) -> dict[str, bool]:
    """Every linear of `bank` -> True if it reads the residual stream (a gate or up), False if it writes it.

    A fused bank stores its experts as stacked parameters named after the module, so those are its parts.
    """
    from .adapters.fused_experts import DOWN, IN, IN_UP

    if isinstance(bank.experts, str):
        inp = IN if bank.gated else IN_UP
        return {f"{bank.experts}.{inp}": True, f"{bank.experts}.{DOWN}": False}
    return {
        **{i: True for ins, _out in bank.experts for i in ins},
        **{out: False for _ins, out in bank.experts},
    }
