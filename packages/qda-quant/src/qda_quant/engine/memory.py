"""Price and fit per-rank model, statistics and checkpoint-stage memory."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch

from ..arch import ModelGraph
from ..format import Format
from ..recipe import Recipe
from ..solve.solve import MIN_FACTOR_BATCH
from ..step import QuantStep, Refused, StagePeak, Step, keeps_float_stream
from . import residency as _res
from .stats import input_rows_of

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .planning import Plan


def _residency(
    recipe: Recipe,
    meta: PreTrainedModel,
    graph: ModelGraph,
    steps: Sequence[Step],
    needs: frozenset[str],
    fmt: Format,
    devices: list[torch.device],
    dtype: torch.dtype,
    residency: str | None,
    batch_size: int,
    formats: dict[str, Format],
    least_memory: bool = False,
) -> tuple[_res.Residency, int, list[int], dict[str, int], Callable[[int, str | None, bool], _res.Residency]]:
    esz = torch.finfo(dtype).bits // 8
    model_bytes = sum(p.numel() for p in meta.parameters()) * esz
    loaded = [sum(p.numel() for p in meta.get_submodule(lg.module).parameters()) * esz for lg in graph.layers]
    host_tensors = frozenset(graph.extra.get("host_tensors", ()))
    host_parameter_bytes = (
        {
            lg.module: sum(
                p.numel() * esz
                for name, p in meta.get_submodule(lg.module).named_parameters(prefix=lg.module)
                if name in host_tensors
            )
            for lg in graph.layers
        }
        if host_tensors
        else {}
    )
    layer_bytes = max(
        n - host_parameter_bytes.get(lg.module, 0) for lg, n in zip(graph.layers, loaded, strict=True)
    )
    text_config = cast(Any, meta.config).get_text_config()
    hidden = text_config.hidden_size
    # a rank streams the batches `engine.dist.shard` gives it: with an uneven batch count that is a whole
    # batch more than `samples // world`, so price the largest share, not the average
    n_batches = -(-recipe.calib.samples // batch_size)
    rank_samples = min(-(-n_batches // len(devices)) * batch_size, recipe.calib.samples)
    per_stream = rank_samples * recipe.calib.seqlen * hidden * esz if needs else 0
    streams = 2 if keeps_float_stream(needs) else 1
    act_bytes = per_stream * streams
    # a step needing layer_io or outputs makes the engine keep each batch's layer output on the card, whether
    # or not the hidden states are offloaded, so plan it apart from the streams that follow activation residency
    retained_bytes = per_stream if needs & {"outputs", "layer_io"} else 0

    model_step_bytes = max((s.model_step_bytes(meta, graph) for s in steps), default=0)

    def at(
        chunks: int, layer_bytes: int, force: str | None, prefer_least: bool = least_memory
    ) -> _res.Residency:
        terms, solves = memory_terms(
            meta,
            graph,
            steps,
            samples=recipe.calib.samples,
            seqlen=recipe.calib.seqlen,
            batch_size=batch_size,
            world=len(devices),
            fmt=fmt,
            esz=esz,
            chunks=chunks,
            formats=formats,
        )
        res = _res.plan(
            model_bytes,
            layer_bytes,
            act_bytes,
            devices,
            terms,
            solves,
            force=force,
            retained_bytes=retained_bytes,
            least_memory=prefer_least and force is None,
            model_step_bytes=model_step_bytes,
        )
        return replace(res, expert_chunks=chunks)

    def fit(layer_bytes: int, force: str | None, prefer_least: bool) -> _res.Residency:
        """The plan at the fewest chunks of experts that fit one card: the whole layer, else up to one expert each."""
        chunked = [lg for lg in graph.layers if len(devices) == 1 and _chunks(meta, lg, steps, 2)]
        most = max((_bank_dims(meta, b)[0] for lg in chunked for b in lg.banks), default=1)
        # the solve's int32 offsets cap a chunk of experts as well: price the fewest chunks that fit
        from ..solve.solve import experts_per_solve
        from ..solve.units import units_of

        need = 1
        for lg in chunked:
            for u in units_of(meta.get_submodule(lg.module), lg):
                if u.part is not None:
                    rows, cols = u.shape()
                    need = max(need, -(-u.n_experts() // experts_per_solve(rows * cols)))
        most = max(most, need)
        for chunks in range(need, most):
            with contextlib.suppress(Refused):
                return at(chunks, layer_bytes, force, prefer_least)
        return at(most, layer_bytes, force, prefer_least)

    return fit(layer_bytes, residency, least_memory), layer_bytes, loaded, host_parameter_bytes, fit


def _peaks(peaks: StagePeak | tuple[StagePeak, ...] | None) -> tuple[StagePeak, ...]:
    if peaks is None:
        return ()
    return (peaks,) if isinstance(peaks, StagePeak) else peaks


def _store_peak(
    plan: Plan,
    graph: ModelGraph,
    loaded: list[int],
    layer_bytes: int,
    fit: Callable[[int, str | None, bool], _res.Residency],
    forced: str | None,
    least_memory: bool,
) -> Plan:
    while True:  # at most twice: the second turn is the default choice's, which never hands over again
        kept = max(plan.store.layer_bytes(plan, lg, n) for lg, n in zip(graph.layers, loaded, strict=True))
        if kept != layer_bytes:  # the store keeps a layer smaller than it loads: the peak it holds is that
            plan = replace(plan, residency=fit(kept, plan.residency.weights, least_memory))
            layer_bytes = kept
        try:
            plan.store.check(plan)
        except Refused:
            if forced is not None or not least_memory:
                raise
            least_memory = False
            plan = replace(plan, residency=fit(kept, None, False))
        else:
            return plan


CALL_ARGS_BYTES = 5 * 2**20


def per_layer_inputs_bytes(meta: torch.nn.Module, layers: int, seqlen: int, esz: int) -> int:
    """One calibration sample's per-layer inputs, which a model with per-layer embeddings keeps on the card."""
    text = cast("Any", meta.config).get_text_config()
    width = getattr(text, "hidden_size_per_layer_input", 0) or 0
    return seqlen * layers * width * esz


def memory_terms(
    meta: torch.nn.Module,
    graph: Any,
    steps: Sequence[Step],
    *,
    samples: int,
    seqlen: int,
    batch_size: int,
    world: int,
    fmt: Format,
    esz: int = 2,
    chunks: int = 1,
    formats: dict[str, Format] | None = None,
) -> tuple[list[_res.Term], list[_res.Solve]]:
    """What each rank holds besides its weights and hidden states, at the peak of the largest layer."""
    needs = frozenset().union(*(s.needs for s in steps))
    if not needs:
        return [], []
    tokens = batch_size * seqlen
    cap = input_rows_of(steps)
    rows = samples * seqlen
    per_layer = [
        _layer_terms(meta, lg, steps, needs, tokens, world, fmt, chunks, formats, esz, cap, rows)
        for lg in graph.layers
    ]
    terms, solves = max(per_layer, key=lambda pair: sum(t.bytes for t in pair[0]))
    n_batches = -(-samples // batch_size)
    share = -(-n_batches // world)  # this rank's batches
    # samples, not whole batches: a partial last batch must not be rounded up
    retained = min(share * batch_size, samples)
    terms = [
        *terms,
        _res.Term("call arguments", share * CALL_ARGS_BYTES),
        _res.Term(
            "per-layer inputs", retained * per_layer_inputs_bytes(meta, len(graph.layers), seqlen, esz)
        ),
    ]
    return [t for t in terms if t.bytes], solves


def _layer_terms(
    meta: torch.nn.Module,
    lg: Any,
    steps: Sequence[Step],
    needs: frozenset[str],
    tokens: int,
    world: int,
    fmt: Format,
    chunks: int,
    formats: dict[str, Format] | None,
    esz: int,
    cap: int | None,
    rows: int,
) -> tuple[list[_res.Term], list[_res.Solve]]:
    chunks = chunks if world == 1 and _chunks(meta, lg, steps, chunks) else 1
    dense, bank_in, bank_down = _hessian_bytes(meta, lg, needs, chunks)
    widest = _widest_hessian_bytes(meta, lg, needs)
    share = world if _splits(meta, lg, steps, world) else chunks
    # a row-sharing step keeps only this rank's calibration batches, so nothing is gathered on rank 0 and every
    # card holds its own share of the rows (`dist.shard`: contiguous batches, the largest share what is priced).
    # An expert-parallel step on a split bank layer keeps its own batches too: the card trains its own experts on
    # them and the other ranks' tokens reach it through `engine.ep`'s all-to-all, never as a gathered batch list
    ep = (
        world > 1
        and bool(lg.banks)
        and any(getattr(s, "expert_parallel", False) for s in steps)
        and _splits(meta, lg, steps, world)
    )
    shared = world > 1 and (ep or (not lg.banks and any(getattr(s, "shares_rows", False) for s in steps)))
    if shared:  # `dist.shard` gives a rank contiguous batches: price the largest share
        batched = -(-rows // tokens)
        mine = min(-(-batched // world) * tokens, rows)
    else:
        mine = rows
    terms = [
        _res.Term("dense Hessians", dense),
        _res.Term("expert Hessians", bank_in + bank_down),
        _res.Term("routing transient", _routing_bytes(meta, lg, needs, tokens, chunks)),
        _res.Term("retained inputs", _retained_inputs_bytes(meta, lg, needs, cap, mine, esz)),
        _res.Term("search copy", _search_copy_bytes(meta, lg, steps, needs, tokens)),
        _res.Term("search weights", _search_weights_bytes(meta, lg, steps, needs, esz)),
        _res.Term("search outputs", _search_outputs_bytes(meta, lg, steps, needs, cap, mine, esz)),
        # The rank-order fold reuses local statistics, but receives a whole bank before a split solve.
        _res.Term("fold buffer", max(widest, bank_in, bank_down) if world > 1 else 0),
        *([] if shared else _gather_terms(meta, lg, steps, needs, cap, rows, tokens, world, esz)),
        _res.Term("linear-attention transients", linear_attention_bytes(meta, lg, tokens, esz)),
    ]
    return terms, _solves(meta, lg, steps, share, fmt, formats, joint=ep)


GDN_CHUNK = 64


def linear_attention_bytes(meta: torch.nn.Module, lg: Any, tokens: int, esz: int) -> int:
    """One calibration forward's transients through a gated-delta (linear-attention) layer, at `tokens` rows."""
    layer = meta.get_submodule(lg.module)
    gdn = next(
        (m for m in layer.modules() if {"A_log", "dt_bias"} <= set(dict(m.named_parameters(recurse=False)))),
        None,
    )
    if gdn is None:
        return 0
    from ..arch.adapters import ADAPTERS

    for _, adapter in ADAPTERS.items():
        own = getattr(adapter, "linear_attention_bytes", None)
        if own is not None and adapter.matches(meta):
            return own(gdn, tokens, esz, GDN_CHUNK)
    attrs = cast("Any", gdn)
    if not hasattr(attrs, "num_v_heads"):  # a Mamba2 mixer (NemotronH) owns A_log / dt_bias too
        heads, head_dim = attrs.num_heads, attrs.head_dim
        inter, conv = heads * head_dim, attrs.conv_dim  # conv_dim = intermediate + 2 * groups * state
        projected = tokens * esz * (inter + conv + heads)  # in_proj's gate / B-C / dt output
        conv_io = tokens * esz * 2 * conv  # the convolution's input and output
        scan = (
            tokens * 4 * (inter + 2 * attrs.n_groups * attrs.ssm_state_size + heads)
        )  # fp32 x, B, C, dt, out
        states = tokens * 4 * heads * head_dim * attrs.ssm_state_size // attrs.chunk_size  # h, every chunk
        return projected + conv_io + scan + states
    heads = attrs.num_v_heads  # q and k are repeated to the value heads before the rule runs
    key, value = heads * attrs.head_k_dim, attrs.value_dim
    conv = attrs.conv_dim  # 2 x key_dim + value_dim, the convolution's channels
    model = tokens * esz * (2 * conv + value)  # the convolution's input and output, and `z`
    rule = tokens * 4 * (2 * key + 5 * value)  # fp32 q, k; v, w, u, the new values, the output
    state = tokens * 4 * heads * attrs.head_k_dim * attrs.head_v_dim // GDN_CHUNK  # h, every chunk
    intra = tokens * 4 * heads * GDN_CHUNK  # the pairwise-decay matrix of one chunk
    gates = tokens * 4 * 3 * heads  # g, beta, the cumulative decay
    return model + rule + state + intra + gates


def _dense_groups(lg: Any, *, searched: bool = False) -> list[Any]:
    fused = {b.experts for b in lg.banks if isinstance(b.experts, str)}
    return [
        g
        for g in lg.groups
        if not any(m == f or m.startswith(f + ".") for m in g.modules for f in fused)
        and not (searched and g.norm is None and g.prev is None)
    ]


def _gather_terms(
    meta: torch.nn.Module,
    lg: Any,
    steps: Sequence[Step],
    needs: frozenset[str],
    cap: int | None,
    rows: int,
    tokens: int,
    world: int,
    esz: int,
) -> list[_res.Term]:
    if world == 1:
        return []
    hidden = cast("Any", meta.config).get_text_config().hidden_size
    io = "layer_io" in needs and any(getattr(s, "multi_card_layer_io", False) for s in steps)
    # `engine.dist.shard` gives a rank contiguous *batches*, not contiguous rows, so an uneven batch count
    # leaves one rank with a whole batch more than `rows // world`; that rank's rows are what is padded
    n_batches = -(-rows // tokens)  # `engine.data` cuts the calibration into batches of `tokens` rows
    share = -(-n_batches // world)
    shard = min(share * tokens, rows)  # samples, not whole batches: a partial last batch is not rounded up
    terms: list[_res.Term] = []
    if "inputs" in needs:
        widths = [_features(meta.get_submodule(g.modules[0])) for g in _dense_groups(lg)]
        banks = [_bank_dims(meta, b)[1] for b in lg.banks if isinstance(b.experts, str)]
        kept = shard if cap is None else min(shard, cap)  # rows a rank sends, one tensor per group
        widest = max([*widths, *banks], default=0)
        if widest:
            terms.append(_res.Term("gather pads", kept * widest * esz))
            # the receive holds every rank's shard: one sample at that key's width, beside rank 0's rows
            terms.append(_res.Term("gathered inputs", world * kept * widest * esz))
        if cap is not None and widths:  # a capped group comes back as one tensor: rank 0 copies the receive
            terms.append(_res.Term("compacted inputs", world * kept * max(widths) * esz))
        if banks:  # a bank's all-token inputs are one tensor on every rank: its copy is always made
            terms.append(_res.Term("compacted bank inputs", rows * max(banks) * esz))
    if io:  # the layer's hidden states are never capped: every batch's row travels
        terms.append(_res.Term("gathered layer_io", world * shard * hidden * esz))
        terms.append(_res.Term("layer_io pad", shard * hidden * esz))
        # a step that fits the float target (GSQ) gathers the targets beside the hidden states
        terms.append(_res.Term("gathered layer_io targets", world * shard * hidden * esz))
    return terms


def _retained_inputs_bytes(
    meta: torch.nn.Module, lg: Any, needs: frozenset[str], cap: int | None, rows: int, esz: int
) -> int:
    if "inputs" not in needs:
        return 0
    kept = rows if cap is None else min(rows, cap)
    dense = sum(kept * _features(meta.get_submodule(g.modules[0])) for g in _dense_groups(lg))
    fused = sum(rows * _bank_dims(meta, b)[1] for b in lg.banks if isinstance(b.experts, str))
    return (dense + fused) * esz


def _search_copy_bytes(
    meta: torch.nn.Module, lg: Any, steps: Sequence[Step], needs: frozenset[str], tokens: int
) -> int:
    if "inputs" not in needs or not any(getattr(s, "search_input_copy", False) for s in steps):
        return 0
    widths = [_features(meta.get_submodule(g.modules[0])) for g in _dense_groups(lg)]
    return tokens * max(widths, default=0) * 4


def _search_weights_bytes(
    meta: torch.nn.Module, lg: Any, steps: Sequence[Step], needs: frozenset[str], esz: int
) -> int:
    if "inputs" not in needs or not any(getattr(s, "search_weight_copies", False) for s in steps):
        return 0
    widest = widest_linear = 0
    for g in _dense_groups(lg, searched=True):
        elems = [_weight_elements(meta.get_submodule(m)) for m in g.modules]
        widest = max(widest, sum(elems))
        widest_linear = max(widest_linear, max(elems, default=0))
    # w0 + wcat + wn per group weight; ws + its two float32 and dtype results per linear weight
    return widest * (esz + 8) + widest_linear * 14


def _weight_elements(mod: torch.nn.Module) -> int:
    weight = getattr(mod, "weight", None)
    return int(weight.numel()) if weight is not None else 0


def _search_outputs_bytes(
    meta: torch.nn.Module,
    lg: Any,
    steps: Sequence[Step],
    needs: frozenset[str],
    cap: int | None,
    rows: int,
    esz: int,
) -> int:
    lists = max((int(getattr(s, "search_output_lists", 0)) for s in steps), default=0)
    if "inputs" not in needs or not lists:
        return 0
    kept = rows if cap is None else min(rows, cap)
    parent_attention = "layer_io" in needs and any(
        getattr(s, "search_attention_parent", False) for s in steps
    )
    width = 0
    for g in _dense_groups(lg, searched=True):
        if g.kind == "mlp_in" or (g.kind == "attn_in" and parent_attention and g.norm is not None):
            width = max(width, _features(meta.get_submodule(g.modules[0])))
        else:
            width = max(width, sum(_out_features(meta.get_submodule(m)) for m in g.modules))
    return lists * kept * width * esz


def _out_features(mod: torch.nn.Module) -> int:
    return int(getattr(mod, "out_features", 0) or cast(torch.Tensor, mod.weight).shape[0])


def _solves(
    meta: torch.nn.Module,
    lg: Any,
    steps: Sequence[Step],
    share: int,
    fmt: Format,
    formats: dict[str, Format] | None,
    *,
    joint: bool = False,
) -> list[_res.Solve]:
    from ..solve.units import stored_format, units_of

    quant = [s for s in steps if isinstance(s, QuantStep)]
    joint = joint and any(getattr(s, "joint_units", False) for s in quant)
    out = []
    for u in units_of(meta.get_submodule(lg.module), lg):
        f = stored_format(formats, u, fmt)
        group = f.group_size
        rows, cols = u.shape()
        workspace = max((s.solver_workspace(rows, cols) for s in quant), default=0)
        if not workspace:
            continue
        if u.part is None:
            dense = max((s.dense_workspace(rows, cols) for s in quant), default=0)
            out.append(_res.Solve("solver workspace", 1, dense, 1, joint=joint))
            continue
        experts = -(-u.n_experts() // share)
        codes = experts * rows * cols
        scales = experts * rows * -(-cols // (cols if group == -1 else group)) * 4
        buffers = codes + (1 if f.symmetric else 2) * scales
        out.append(_res.Solve("solver workspace", experts, workspace, MIN_FACTOR_BATCH, buffers, joint=joint))
    return out


def _hessian_bytes(
    meta: torch.nn.Module, lg: Any, needs: frozenset[str], chunks: int
) -> tuple[int, int, int]:
    dense = 0
    if "hessian" in needs:
        # every dense group's Hessian is gathered in the same calibration pass, so all of them are held at once
        dense = sum(_features(meta.get_submodule(g.modules[0])) ** 2 * 4 for g in _dense_groups(lg))
    bank_in = bank_down = 0
    if "expert_hessian" in needs:
        for b in lg.banks:
            e, d, i = _bank_dims(meta, b)
            held = -(-e // chunks)
            bank_in += held * d * d * 4
            bank_down += held * i * i * 4
    return dense, bank_in, bank_down


def _widest_hessian_bytes(meta: torch.nn.Module, lg: Any, needs: frozenset[str]) -> int:
    """The widest dense group's Hessian: what the rank-order fold receives at once."""
    if "hessian" not in needs:
        return 0
    return max((_features(meta.get_submodule(g.modules[0])) for g in _dense_groups(lg)), default=0) ** 2 * 4


def _routing_bytes(meta: torch.nn.Module, lg: Any, needs: frozenset[str], tokens: int, chunks: int) -> int:
    if "expert_hessian" not in needs:
        return 0
    total = 0
    for b in lg.banks:
        e, d, i = _bank_dims(meta, b)
        cmax = 2 * tokens * b.top_k // e + 1  # twice the mean rows per expert covers routing skew ([M7])
        total += -(-e // chunks) * cmax * (2 * d + 4 * i) * 2
    return total


def _bank_dims(meta: torch.nn.Module, bank: Any) -> tuple[int, int, int]:
    if isinstance(bank.experts, str):
        from ..arch.adapters import fused_experts

        ex = meta.get_submodule(bank.experts)
        experts = fused_experts.in_weight(ex).shape[0]
        return (
            int(experts),
            int(fused_experts.down_weight(ex).shape[1]),
            int(fused_experts.down_weight(ex).shape[-1]),
        )
    ins, out = bank.experts[0]
    return len(bank.experts), _features(meta.get_submodule(ins[0])), _features(meta.get_submodule(out))


def _features(mod: torch.nn.Module) -> int:
    from .stats import _input_features

    return _input_features(mod)


def _splits(meta: torch.nn.Module, lg: Any, steps: Sequence[Step], world: int) -> bool:
    from .runner import splits_experts

    return splits_experts(meta.get_submodule(lg.module), lg, steps, world)


def _chunks(meta: torch.nn.Module, lg: Any, steps: Sequence[Step], n: int) -> bool:
    from .runner import chunks_experts

    return chunks_experts(meta.get_submodule(lg.module), lg, steps, n)


def _stage_devices(
    step: Step, ctx: Any, devices: list[torch.device]
) -> tuple[list[torch.device], tuple[StagePeak, ...]]:
    from dataclasses import replace

    from .residency import free_bytes

    cap = getattr(step, "max_devices", None)
    if cap is None or len(devices) <= cap:
        return devices, _peaks(step.checkpoint_peak(ctx))

    def free(device: torch.device) -> int:
        # a reading, not a gate: a card another job has filled raises from the driver rather than reporting 0,
        # and the plan is made before the build holds anything, so rank an unreadable card last and let the
        # step's own peak refuse it (with its numbers) if the cards chosen cannot hold the stage
        try:
            return free_bytes(device)
        except RuntimeError:  # torch's AcceleratorError is a RuntimeError
            return -1

    order = sorted(range(len(devices)), key=lambda i: free(devices[i]), reverse=True)
    ranked = [devices[i] for i in order]
    priced: tuple[StagePeak, ...] = ()
    for n in range(1, cap + 1):
        pick = ranked[:n]
        priced = _peaks(step.checkpoint_peak(replace(ctx, devices=pick)))
        if priced and not any(p.over_free for p in priced):
            return pick, priced
    return ranked[:cap], priced
