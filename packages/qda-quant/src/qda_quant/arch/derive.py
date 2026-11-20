"""Derive a `ModelGraph` from a model's own forward: which linears read the same tensor, which norm made it.

One tiny forward (a few tokens; meta tensors work too) runs with a hook on every module of the decoder
layers. Linears that receive the very same tensor object share an input; the norm whose output *is* that
object feeds them. Nothing here reads a module name to decide a relation: names only label group kinds.
`verify.py` then checks every relation numerically in fp64.

What the forward cannot show (the layout of fused parameters) is added afterwards by the one registered
architecture adapter that matches the model (`qda_quant.arch.adapters`).
"""

from __future__ import annotations

from typing import Any, cast

from torch import nn

from ..step import Refused, probe_refusal
from . import KEEP_ROLES, SHARED_EXPERTS, GroupKind, ModelGraph
from .derive_adapt import (  # noqa: F401  (re-exports: the pre-split module's names)
    _adapt,
    _checked_stream,
    _dotted,
    _serving_names,
    served_name,
)
from .derive_experts import (  # noqa: F401
    _banks,
    _expert_linears,
    _expert_stack,
    _keep_shared,
    _shared,
    _with_shared,
    router_of,
    routes_itself,
)
from .derive_groups import (  # noqa: F401
    _check_ignored,
    _head_gate,
    _is_norm,
    _kind,
    _layer,
    _prev_candidate,
    _readers,
    ignore_hits,
)
from .derive_model import _name_of, _num_experts, _outside, decoder_layers, text_config  # noqa: F401
from .derive_trace import (  # noqa: F401
    _GDN_KERNELS,
    _ROUTED,
    _TRACE_TOKENS,
    _by_device,
    _meta_nonzero,
    _meta_scalar,
    _root,
    _Trace,
    route_gdn,
    trace,
)


def derive(model: nn.Module, keep: frozenset[str] = frozenset(KEEP_ROLES)) -> ModelGraph:
    """Structural graph of `model` (no numeric checks; works on a meta-device model).

    `keep` holds the config's ``ignore`` entries: roles (`KEEP_ROLES`), exact module names and ``re:`` patterns
    (`ignore_hits`). Linears they match are left out of the groups, so no step quantizes them; roles are listed
    in ``graph.extra["kept_bf16"]``, names in ``extra["ignored"]`` (module -> (pattern, kind of its group)), and
    ``extra["readers"]`` still lists them all, so a norm fold scales them too. A pattern that names every
    linear of a routed bank keeps that whole bank float: the bank leaves the graph (no unit, no statistics,
    no plan) and its root joins ``extra["float_banks"]``.

    Raises:
        Refused: a module holds stacked expert parameters that no adapter turned into a bank (its experts would
            stay unquantized); an ``ignore`` pattern matches nothing the build would quantize, part of a routed
            bank, or part of a group vLLM fuses into one linear (q/k/v, gate/up).
    """
    cfg = text_config(model)
    prefix, layers = decoder_layers(model)
    route_gdn(model)  # the trace runs on meta or CPU; a card must keep fla's kernels
    tr = trace(model)
    readers: dict[str, tuple[str, ...]] = {}
    kept: dict[str, list[str]] = {}
    ignored: dict[str, tuple[tuple[str, ...], GroupKind]] = {}
    graphs = []
    for i, layer in enumerate(layers):
        where = f"{prefix}.{i}"
        try:
            graphs.append(_layer(i, where, layer, tr, cfg, readers, keep, kept, ignored))
        except Refused:
            raise
        except Exception as e:  # noqa: BLE001 - a layer QDA cannot read; refuse it, do not traceback
            raise probe_refusal(model, "deriving its graph", e, module=where) from e
    graphs = tuple(graphs)
    embed = _name_of(model, cast(Any, model).get_input_embeddings())
    head = cast(Any, model).get_output_embeddings()  # HF model method, absent from nn.Module
    lm_head = _name_of(model, head)
    # the last norm called outside the decoder layers (the head may read a slice of its output)
    final_norm = next(
        (
            k
            for k, _i, _o in reversed(tr.calls)
            if not k.startswith(prefix + ".") and _is_norm(model.get_submodule(k))
        ),
        "",
    )
    graph = ModelGraph(
        model_type=getattr(cfg, "model_type", type(model).__name__),
        layers=graphs,
        embed=embed,
        lm_head=lm_head,
        final_norm=final_norm,
        extra={
            "tied": head is not None and head.weight is cast(Any, model).get_input_embeddings().weight,
            # group key or decoder-layer norm -> every module reading that input (a norm fold scales them all)
            "readers": readers,
            # outermost modules outside the text model (a vision tower, a projector, an MTP head)
            "outside": _outside(model, prefix.rpartition(".")[0], {embed, lm_head, final_norm} - {"", None}),
        },
    )
    graph = _adapt(model, graph)
    graph = _with_shared(graph, dict(model.named_modules()), tr)
    if SHARED_EXPERTS in keep:
        graph = _keep_shared(graph, kept)
    # an adapter may also report weights its layout cannot quantize; keep those roles beside the config's
    graph.extra["kept_bf16"] = {
        **(graph.extra.get("kept_bf16") or {}),
        **{r: tuple(kept[r]) for r in KEEP_ROLES if r in kept},
    }
    # before `ignore` drops a bank it keeps float: its stacked parameters are still explained by that bank
    banked = {b.experts for g in graph.layers for b in g.banks}
    graph.extra["ignored"], graph = _check_ignored(graph, keep, ignored)
    stray = [k for k, m in model.named_modules() if _expert_stack(m) and k not in banked]
    if stray:
        raise Refused(
            f"no architecture adapter handles the stacked expert parameters of {stray}; add one to the "
            "'qda.arch_adapters' entry-point group (see qda_quant.arch.adapters)"
        )
    return graph


__all__ = [
    "decoder_layers",
    "derive",
    "ignore_hits",
    "route_gdn",
    "router_of",
    "served_name",
    "text_config",
    "trace",
]
