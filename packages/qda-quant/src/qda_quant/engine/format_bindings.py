"""Resolve per-unit formats and validate serving-kernel bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch

from ..arch import ModelGraph
from ..format import Format
from ..step import Refused
from ..target import KernelPlan, LayerKind, Target

if TYPE_CHECKING:
    from .packed import Packed

#: The pattern `resolve_formats` records for a unit the planner, not a config rule, gave a format.
_DEFAULTED = "<default>"

#: The one pair of activation schemes a mixed build may run in one layer: NVFP4's block cut and FP8's per-token
#: cut. vLLM serves each module with its own quant method, so a layer may have both; any other pair would need
#: two cuts over the same layer inputs, which the calibration forward does not do.
_MIXED_ACTS = frozenset({"nvfp4_block", "fp8_token"})


def expert_defaults(graph: ModelGraph, fmt: Format) -> dict[str, Format]:
    """The format the planner gives expert units the body format leaves to its default (unit key -> format).

    A spelling that names no symmetry inherits it (`Format.symmetric_defaulted`), and the released MoE
    checkpoints are symmetric: those routed expert banks, and the shared experts that live in them, default
    to symmetric at the body format's group size. A spelling that names a symmetry keeps it for every unit, so
    a config can still pin asymmetric experts explicitly, and ``format_overrides`` still gives a bank its own
    scheme. Dense layers outside those banks and dense models are untouched: their formats do not change.
    """
    sym = fmt.expert_format
    if sym == fmt:
        return {}
    shared = {name for lg in graph.layers for bank in lg.banks for name in bank.shared}
    out: dict[str, Format] = {}
    for lg in graph.layers:
        for bank in lg.banks:
            out[bank.router] = sym
        for grp in lg.groups:
            if any(name in shared for name in grp.modules):
                out[grp.modules[0]] = sym
    return out


def _default_record(graph: ModelGraph, applied: Mapping[str, Format]) -> dict[str, str]:
    """``Plan.format_defaults``: the unit class a default reached -> the format, for the records."""
    routers = {bank.router for lg in graph.layers for bank in lg.banks}
    out: dict[str, str] = {}
    for key, f in sorted(applied.items()):
        out["routed_experts" if key in routers else "shared_experts"] = f.name
    return out


def resolve_formats(
    model: torch.nn.Module,
    graph: ModelGraph,
    rules: Sequence[tuple[str, str]],
    fmt: Format,
    target: Target,
    plan_dense: KernelPlan,
    plan_moe: KernelPlan | None,
    defaults: Mapping[str, Format] | None = None,
) -> tuple[dict[str, Format], dict[tuple[str, str], KernelPlan], dict[str, Format]]:
    """Formats the config gives each unit (unit key -> format), the kernel plans they need, and the defaults.

    A rule names a whole solve unit: a dense `LinearGroup` (keyed by its first module) or an `ExpertBank`
    (keyed by its router), or a fused bank's down projections (keyed by `down_key`, the down parameter), which
    then leave the bank's format; only an exporter that `splits_banks` writes those. A pattern naming the bank
    names its down projections too. Patterns match as `ignore` matches them; the body `fmt` stores unclaimed
    units. Refused before a weight loads: no unit matched, part of a fused group, a unit stored two ways, a
    format the target cannot serve for that kind, bank rules that leave a bank out or split the banks
    (transformers resolves one expert scheme), and a dense binding the serve settings would not give the
    planned kernel.

    `defaults` (`expert_defaults`) gives units no rule names their class default; the third result is the
    part of it that reached a unit (nothing when a rule claimed every expert bank).
    """
    from ..arch.derive import ignore_hits
    from ..output.export import merge_serve
    from ..solve.units import bank_module, down_key

    units: dict[str, tuple[tuple[str, ...], LayerKind]] = {}
    downs: dict[str, str] = {}  # down_key -> its bank's router
    custom = {bank.router: bank.custom_router for lg in graph.layers for bank in lg.banks}
    for lg in graph.layers:
        for grp in lg.groups:
            units[grp.modules[0]] = (tuple(grp.modules), "dense")
        for bank in lg.banks:
            units[bank.router] = ((bank_module(bank),), "moe")
            if (down := down_key(bank)) is not None:
                units[down], downs[down] = ((down,), "moe"), bank.router
    if not units:
        raise Refused(
            "format_overrides: this build quantizes no layer unit to override (every linear is kept bf16)"
        )
    chosen: dict[str, tuple[Format, LayerKind, str]] = {}
    for pattern, name in rules:
        f = Format.parse(name)
        hits = []
        for key, (names, kind) in units.items():
            matched = [n for n in names if ignore_hits((pattern,), n)]
            if not matched:
                continue
            if kind == "dense" and len(matched) < len(names):
                raise Refused(
                    f"format_overrides: {pattern!r} matches {len(matched)} of the {len(names)} linears of "
                    f"{key}; they are solved and served as one fused linear with one scheme: match them all"
                )
            hits.append(key)
        hits = [k for k in hits if downs.get(k) not in hits]  # a rule naming a bank names all of it
        if not hits:
            raise Refused(_no_unit(pattern, graph, model))
        for key in hits:
            if key in chosen and chosen[key][0] != f:
                raise Refused(
                    f"format_overrides: {key} is {chosen[key][0].name} under {chosen[key][2]!r} and "
                    f"{f.name} under {pattern!r}; one unit is stored in one format"
                )
            chosen[key] = (f, units[key][1], pattern)
    claimed = set(chosen)  # every unit a rule names, even the one it stores in the body format
    chosen = {k: v for k, v in chosen.items() if v[0] != fmt or k in downs}
    banks = {bank.router for lg in graph.layers for bank in lg.banks}
    expert = {k: v for k, v in chosen.items() if k in banks}
    if expert:
        shared = {v[0] for v in expert.values()}
        missing = sorted(banks - set(expert))
        if len(shared) > 1 or missing:
            what = (
                f"banks it leaves unset: {missing[:3]}"
                if missing
                else f"formats it gives: {sorted(f.name for f in shared)}"
            )
            raise Refused(
                "format_overrides: transformers loads every routed-expert bank with one scheme (the first "
                f"`re:` group naming experts), so an expert rule must name every bank in one format; {what}"
            )
    applied = {k: f for k, f in (defaults or {}).items() if k not in claimed}
    for key, f in applied.items():
        chosen[key] = (f, units[key][1], _DEFAULTED)
    for key in downs.keys() & chosen.keys():
        if chosen[key][0] == chosen.get(downs[key], (fmt,))[0]:
            del chosen[key]  # stored as its bank is: the bank is not split
    plans: dict[tuple[str, str], KernelPlan] = {}
    for key, (f, kind, pattern) in sorted(chosen.items()):
        if f.act != fmt.act and frozenset((f.act, fmt.act)) != _MIXED_ACTS:
            raise Refused(
                f"{_unit_why(key, pattern, f)} quantizes activations as {f.act}, the body's {fmt.act}; "
                "the build runs one activation scheme over every linear of a layer"
            )
        if (f.name, kind) in plans:
            continue
        try:
            plans[(f.name, kind)] = target.plan(f, kind, decomposed=kind == "moe" and custom.get(key, False))
        except Refused as e:
            raise Refused(f"{_unit_why(key, pattern, f)}: {e}") from None
    used = {f"{fmt.name}:dense": plan_dense}
    if plan_moe is not None:
        used[f"{fmt.name}:moe"] = plan_moe
    used |= {f"{name}:{kind}": p for (name, kind), p in plans.items()}
    try:
        serve = merge_serve(used, target.vllm)
    except ValueError as e:
        raise Refused(f"format_overrides: {e}; the formats' kernels need different serve settings") from None
    _check_bindings(model, graph, chosen, fmt, target, plan_dense, plans, serve)
    return {k: v[0] for k, v in chosen.items()}, plans, applied


def _unit_why(key: str, pattern: str, f: Format) -> str:
    """How `key` came to store `f`, for a refusal: a config rule, or the planner's expert default."""
    if pattern == _DEFAULTED:
        return f"{key} defaults to {f.name} (the body format names no symmetry)"
    return f"format_overrides: {key} in {f.name} ({pattern!r})"


def _no_unit(pattern: str, graph: ModelGraph, model: torch.nn.Module) -> str:
    """Why `pattern` names no quantized unit, naming the near miss when the graph has one."""
    from ..arch.derive import ignore_hits
    from ..solve.units import bank_module

    kept = graph.extra.get("ignored") or {}
    hit = next((m for m in kept if ignore_hits((pattern,), m)), None)
    parts = (
        (bank_module(b), f"{bank_module(b)}.{n}")
        for lg in graph.layers
        for b in lg.banks
        for n, _ in model.get_submodule(bank_module(b)).named_parameters()
    )
    part = next((pair for pair in parts if ignore_hits((pattern,), pair[1])), None)
    if hit is not None:
        extra = f"; {hit} is kept bf16 by `ignore` ({kept[hit][0]})"
    elif part is not None:
        extra = (
            f"; {part[1]} is part of the routed bank {part[0]}, and a rule names the whole bank or (a fused "
            "bank, written by an exporter that stores its projections apart) its down projections"
        )
    elif graph.lm_head and ignore_hits((pattern,), graph.lm_head):
        extra = "; the head is stored in its own format by steps.head (format: ...)"
    else:
        extra = ""
    return (
        f"format_overrides: {pattern!r} matches no unit this build quantizes{extra}; a pattern is an exact "
        "module name as `qda plan` prints it, or re: and a regex matched from its start"
    )


def _check_bindings(
    model: torch.nn.Module,
    graph: ModelGraph,
    chosen: dict[str, tuple[Format, LayerKind, str]],
    fmt: Format,
    target: Target,
    plan_dense: KernelPlan,
    plans: dict[tuple[str, str], KernelPlan],
    serve: dict,
) -> None:
    """Refuse a dense unit whose format would not bind to the kernel planned for it under `serve`."""
    for lg in graph.layers:
        for grp in lg.groups:
            f, _, pattern = chosen.get(grp.modules[0], (fmt, "dense", ""))
            if not ((f.weight_type == "int" and f.act == "bf16") or f.weight_type == "fp8"):
                continue  # the dense binding rule covers weight-only int formats and FP8 W8A8
            plan = plans.get((f.name, "dense"), plan_dense)
            for name in grp.modules:
                w = cast(torch.nn.Linear, model.get_submodule(name)).weight
                shape = (int(w.shape[0]), int(w.shape[1]))
                try:
                    bound = target.binds(f, shape, serve)
                except Refused as e:
                    raise Refused(
                        f"{_unit_why(name, pattern, f)} is stored in {f.name}, and the checkpoint's serve "
                        f"settings do not say which kernel binds it: {e}"
                    ) from None
                if bound != plan.kernel:
                    raise Refused(
                        f"{_unit_why(name, pattern, f)} is planned for {plan.kernel}, but under the "
                        f"checkpoint's serve settings vLLM would bind {bound}"
                    )


def _check_moe_bindings(
    model: torch.nn.Module,
    graph: ModelGraph,
    fmt: Format,
    target: Target,
    plan_dense: KernelPlan,
    plan_moe: KernelPlan,
    formats: dict[str, Format],
    plans: dict[tuple[str, str], KernelPlan],
    packed: Packed | None,
) -> None:
    """Check each quantized bank's shape and format under the settings the checkpoint will record or retain."""
    from ..output.export import merge_serve
    from ..solve.units import bank_module, stored_format, units_of

    custom = {bank.router: bank.custom_router for lg in graph.layers for bank in lg.banks}
    if packed is not None:
        target = packed.target
        plan_dense = target.plan(fmt, "dense")
        plan_moe = target.plan(fmt, "moe", decomposed=any(custom.values()))
    used = {"dense": plan_dense, "moe": plan_moe, **{str(k): v for k, v in plans.items()}}
    serve = (packed.record.get("serve") if packed is not None else None) or merge_serve(used, target.vllm)
    for lg in graph.layers:
        names = {bank.router: bank_module(bank) for bank in lg.banks}
        for unit in units_of(model.get_submodule(lg.module), lg):
            if unit.part != "down":
                continue
            f = stored_format(formats, unit, fmt)
            plan = plans.get((f.name, "moe"), plan_moe)
            if packed is not None:
                prefix = names[unit.key] + "."
                if not any(n.startswith(prefix) and n.endswith("_packed") for n in packed.where):
                    continue  # a packed head does not make the untouched experts quantized
                f = packed.format_of(prefix + "0.down_proj")  # the projection vLLM probes for the bank
                plan = target.plan(f, "moe", decomposed=custom.get(unit.key, False))
            try:
                bound = target.binds_moe(f, unit.shape(), serve, decomposed=custom.get(unit.key, False))
            except Refused as e:
                raise Refused(f"expert bank {names[unit.key]} ({unit.key}): {e}") from None
            if bound != plan.kernel:
                raise Refused(
                    f"expert bank {names[unit.key]} is planned for {plan.kernel} in {f.name}, "
                    f"but the checkpoint's serving settings bind {bound}"
                )
