"""Routed expert banks, the module lists and routers that hold them, and their shared experts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from torch import nn

from ..step import Refused
from . import SHARED_EXPERTS, ExpertBank, ModelGraph

if TYPE_CHECKING:
    from .derive_trace import _Trace


def _expert_stack(m: nn.Module) -> bool:
    """True for a module that holds routed experts as stacked parameters: two or more ``[E, ., .]`` tensors."""
    lead = {p.shape[0] for p in m.parameters(recurse=False) if p.dim() == 3}
    return len(lead) == 1 and sum(p.dim() == 3 for p in m.parameters(recurse=False)) >= 2


def router_of(mods: dict[str, nn.Module], bank: str, n_exp: int) -> str | None:
    """The sibling of module `bank` (among `mods`, by name) whose 2-D weight has one row per expert."""
    parent = bank.rpartition(".")[0]
    for k, m in mods.items():
        w = getattr(m, "weight", None)
        sibling = k != bank and k.rpartition(".")[0] == parent
        if sibling and w is not None and w.dim() == 2 and w.shape[0] == n_exp:
            return k
    return None


def routes_itself(router: nn.Module) -> bool:
    """Whether `router` scales the weight of each selected expert by a parameter of its own.

    vLLM's stock routers express the router logits, a top-k and a renormalization; a per-expert multiplier
    applied after selection is a custom routing function (``RoutingMethodType.Custom``), which its monolithic
    fused-expert kernels do not take. Gemma 4 stores it as ``per_expert_scale``.
    """
    return any(name == "per_expert_scale" for name, _ in router.named_parameters())


def _banks(
    mods: dict[str, nn.Module], cfg: Any, n_exp: int | None
) -> tuple[tuple[ExpertBank, ...], set[str]]:
    """Banks of one layer whose experts are modules (a ModuleList of per-expert linears), and the names they own.

    Banks stored as stacked parameters have no linears to read; their adapter adds them.
    """
    if not n_exp:
        return (), set()
    banks, owned = [], set()
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or getattr(cfg, "top_k", 0))
    for k, m in mods.items():
        router = router_of(mods, k, n_exp) if isinstance(m, nn.ModuleList) and len(m) == n_exp else None
        if not router:
            continue
        experts = []
        for e in range(n_exp):
            lin = [
                f"{k}.{e}.{c}"
                for c, cm in cast(nn.ModuleList, m)[e].named_modules()
                if isinstance(cm, nn.Linear)
            ]
            ins = tuple(n for n in lin if not n.endswith(("down_proj", "w2")))
            outs = [n for n in lin if n not in ins]
            if len(outs) != 1:
                raise Refused(
                    f"expert {k}.{e}: cannot tell its output linear among {lin}; "
                    "add an architecture adapter for this model"
                )
            experts.append((ins, outs[0]))
            owned.update(lin)
        banks.append(ExpertBank(tuple(experts), router, top_k, custom_router=routes_itself(mods[router])))
    return tuple(banks), owned


def _expert_linears(bank: ExpertBank) -> set[str]:
    """Every linear of a bank stored as a ModuleList of per-expert modules (a fused bank has none)."""
    if isinstance(bank.experts, str):
        return set()
    return {n for ins, out in bank.experts for n in (*ins, out)}


def _with_shared(graph: ModelGraph, mods: dict[str, nn.Module], tr: _Trace) -> ModelGraph:
    """Every bank with the shared experts the traced forward shows: the dense MLP beside the bank.

    A MoE block feeds its router and its shared expert the same tensor; the shared expert is the other child of
    the router's parent that owns linears reading it (the routed experts are the bank's). Adding its output to
    the bank's is the block's forward, which the trace cannot follow through the addition. No name decides it:
    Qwen spells the path ``shared_expert`` and DeepSeek ``shared_experts``.
    """
    layers = []
    for lg in graph.layers:
        calls = [c for c in tr.calls if c[0].startswith(lg.module + ".")]
        banks = tuple(
            replace(bank, shared=found) if (found := _shared(bank, mods, calls)) else bank
            for bank in lg.banks
        )
        layers.append(replace(lg, banks=banks))
    return replace(graph, layers=tuple(layers))


def _shared(bank: ExpertBank, mods: dict[str, nn.Module], calls: list) -> tuple[str, ...]:
    """Linears of the dense MLP that reads the same input as `bank`'s router."""
    parent = bank.router.rpartition(".")[0]
    tid = next((i for k, i, _o in calls if k == bank.router), None)
    owned = _expert_linears(bank)
    shared: list[str] = []
    seen: set[str] = set()
    for k, i, _o in calls:
        if i != tid or k == bank.router or k.rpartition(".")[0] != parent or k not in mods or k in seen:
            continue
        seen.add(k)
        if any(n.startswith(k + ".") for n in owned):  # the expert bank itself
            continue
        linears = [n for n, m in mods[k].named_modules(prefix=k) if isinstance(m, nn.Linear)]
        if len(linears) >= 2:  # a dense MLP, not a single projection or a scalar gate
            shared += [n for n in linears if n not in shared]
    return tuple(shared)


def _keep_shared(graph: ModelGraph, kept: dict) -> ModelGraph:
    """`graph` without its banks' shared experts among the quantized groups (they stay bf16; readers still list them)."""
    readers = graph.extra["readers"]
    layers = []
    for lg in graph.layers:
        shared = {k for b in lg.banks for k in b.shared}
        groups = []
        for grp in lg.groups:
            mods = tuple(k for k in grp.modules if k not in shared)
            if len(mods) < len(grp.modules):
                kept.setdefault(SHARED_EXPERTS, []).extend(k for k in grp.modules if k in shared)
                if mods and mods[0] != grp.modules[0]:
                    readers[mods[0]] = readers.pop(grp.modules[0])
                elif not mods:
                    readers.pop(grp.modules[0], None)
            if mods:
                groups.append(replace(grp, modules=mods))
        layers.append(replace(lg, groups=tuple(groups)))
    return replace(graph, layers=tuple(layers))
