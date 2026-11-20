"""AWQ (arXiv:2306.00978): per-input-channel scales folded into the producer of a group's input.

Search ``s = x_mean^r / w_mean^(1-r)``, normalized by ``sqrt(max(s) min(s))``, using output MSE with
``Q(W s) / s`` weights; divide the producer by the chosen scale, multiply its readers by it, and rescale
the input statistics so subsequent steps see the folded input.

The default ``parent`` objective evaluates q/k/v through the attention module on every calibration batch
and gate/up through the MLP; ``linear_out`` evaluates q/k/v directly on sampled inputs, while both use
linear outputs for down and o projections, and both require a later ``rtn`` or ``gptq`` step to quantize.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, ClassVar, Literal, cast

import torch
from torch import nn

from ..arch import ExpertBank, LinearGroup, ModelGraph, submodule
from ..arch.adapters.fused_experts import in_param
from ..engine.stats import input_chunks
from ..format import Format
from ..solve.quant import groups
from ..solve.units import down_key, unit_format
from ..step import LayerContext, QuantStep, Refused
from ..target import KernelPlan
from ._model import layer_without_fold_anchor


class AWQ(QuantStep):
    """Activation-aware channel scaling.

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Fused expert searches collect all-token means, down-input Hessians and parent inputs in the engine.

    Attributes:
        n_grid: grid points for the exponent ``r`` in ``[0, 1)``.
        duo_scaling: divide by the weight-magnitude term ``w_mean^(1-r)`` (llm-compressor default).
        objective: ``parent`` (llm-compressor 0.15: q/k/v priced by the attention module's
            output over every calibration batch, re-run through `layer_io`; x_mean and every other loss
            over every calibration token, ``input_rows = None``) or ``linear_out`` (the group's own
            linear outputs on the engine's 8192-row input sample; no layer re-run).
    """

    name = "awq"
    all_token_experts: ClassVar[bool] = True
    #: it folds scales into the float weights; a later step stores the codes (see `AWQ`'s module docstring)
    quantizes: ClassVar[bool] = False
    #: it folds activation scales into a norm and every reader of it (`check_float_banks` refuses it beside a
    #: routed bank `ignore` keeps float, whose exported weights are the source's)
    folds_readers: ClassVar[bool] = True
    #: the parent objective re-runs attention on every batch; a multi-card build gathers the ranks' `layer_io`
    #: batches in rank order, so the search prices the rows a one-card build holds. On a dense layer the ranks
    #: instead share the rows (``shares_rows``): each prices its own batches and the per-batch sums are all
    #: gathered in calibration order, so the search is the one-card arithmetic with the cards' work divided.
    #: A layer with routed expert banks gathers, as its expert statistics are global.
    multi_card_layer_io: ClassVar[bool] = True
    #: each rank prices the groups over its own calibration batches and combines the per-batch sums itself
    #: (`_search`, `Comm.all_rows`); false only where a bank's search reads gathered global statistics
    shares_rows: ClassVar[bool] = True
    #: its searches price unweighted inputs and all-token Hessians; it only rescales the token-weighted
    #: ``expert_hessian`` for a later step (`planning.check_token_weight_readers`)
    reads_token_weights: ClassVar[bool] = False
    #: the search copies one retained batch of a group's input to float32 (`_search`), which the planner counts
    search_input_copy: ClassVar[bool] = True
    #: the search keeps every retained chunk's output for the reference and builds the list again for each
    #: candidate (`_mse(run(), ref)`), so the planner counts both (`planning._search_outputs_bytes`)
    search_output_lists: ClassVar[int] = 2
    #: the search copies the group's weights (``w0``, ``wcat``, ``wn``) and, per candidate, one linear again
    #: (``ws``, quantize-dequantized in float32); the planner counts them (`planning._search_weights_bytes`)
    search_weight_copies: ClassVar[bool] = True
    needs_verified_graph: ClassVar[bool] = True
    n_grid: int = 20
    duo_scaling: bool = True
    objective: Literal["parent", "linear_out"] = "parent"

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse a block with no norm at layer scope, or a fused bank the search does not model.

        The search scales the stored tensor along its last axis and splits gate/up at ``[:I]``, i.e. it
        assumes ``[E, out, in]`` with gate rows first. For a bank stored ``[E, in, out]`` that last axis
        is the output (the hidden axis for ``down_proj``), so the scale would be applied to the wrong
        dimension.
        """
        if module := layer_without_fold_anchor(graph):
            raise Refused(
                f"awq: {module} has no norm at layer scope (its input is normalized inside a submodule), "
                "so the scales cannot be folded; quantize without awq (e.g. gptq alone)"
            )
        bad = next((b for g in graph.layers for b in g.banks if b.transposed), None)
        if bad is not None:
            raise Refused(
                f"awq: routed experts {bad.experts} are stored [E, in, out]; the expert search assumes "
                "[E, out, in] with gate rows before up rows — quantize without awq"
            )
        non_gated = next((b for g in graph.layers for b in g.banks if b.fused and not b.gated), None)
        if non_gated is not None:
            raise Refused(
                f"awq: routed experts {non_gated.experts} are a non-gated bank (up_proj, act_fn(up) into "
                "down_proj); the expert search scales the up projection and the down input together, which "
                "only commutes with the activation for a gated bank (act(gate) * up) — quantize without awq"
            )

    def candidates(self) -> list[tuple[float, bool]]:
        """Return (ratio r, duo-scaled) candidates for the scale search.

        ``parent`` (default): exactly llm-compressor 0.15's ``AWQModifier._get_grid_search_params`` (identity
        ``(0, False)`` plus ``r = i / (n_grid - 2)``, i = 0 .. n_grid - 2, duo-scaled; r = 1 included).
        ``linear_out``: ``r = i / n_grid``, i = 0 .. n_grid - 1.
        """
        # Parent-objective candidates follow llm-compressor 0.15 AWQModifier._get_grid_search_params.
        if self.objective == "linear_out":
            return [(i / self.n_grid, self.duo_scaling) for i in range(self.n_grid)]
        if not self.duo_scaling:
            return [(i / (self.n_grid - 1), False) for i in range(self.n_grid)]
        return [(0.0, False)] + [(i / (self.n_grid - 2), True) for i in range(self.n_grid - 1)]

    @property
    def input_rows(self) -> int | None:
        """Rows of each group's input the engine keeps (None = every calibration token).

        Keep every token for the parent objective and an 8192-row sample for ``linear_out``.
        """
        return None if self.objective == "parent" else 8192

    @property
    def search_attention_parent(self) -> bool:
        """True when q/k/v are priced through the attention module (the parent objective, `_attention_batches`)."""
        return self.objective == "parent"

    @property
    def needs(self) -> frozenset[str]:  # type: ignore[override]
        """Group inputs and expert statistics; the parent objective also re-runs attention on `layer_io`."""
        needs = {"inputs", "expert_hessian"}
        if self.objective == "parent":
            needs.add("layer_io")
        return frozenset(needs)

    @torch.no_grad()
    def apply_layer(self, ctx: LayerContext) -> None:
        """Search and fold one scale vector per foldable group."""
        for grp in ctx.graph.groups:
            if grp.norm is None and grp.prev is None:
                continue
            x = ctx.stats.get(("inputs", grp.modules[0]))
            if x is None or not sum(c.shape[0] for c in input_chunks(x)):
                continue
            s = self._search(ctx, grp, x)
            self._fold(ctx, grp, s)
        for bank in ctx.graph.banks:
            if isinstance(bank.experts, str):
                self._search_expert_bank(ctx, bank)

    @torch.no_grad()
    def _search_expert_bank(self, ctx: LayerContext, bank: ExpertBank) -> None:
        """Scale a fused bank from its norm and each expert's up projection."""
        from ..arch.adapters.fused_experts import DOWN, in_param, split

        assert isinstance(bank.experts, str)
        module = submodule(ctx.layer, ctx.graph, bank.experts)
        if not hasattr(module, in_param(module)) or not hasattr(module, DOWN):
            return
        readers = ctx.model_graph.extra.get("readers", {})
        norm_name = next(
            (
                norm
                for norm, names in readers.items()
                if bank.router in names
                and bank.experts in names
                and getattr(submodule(ctx.layer, ctx.graph, norm), "weight", torch.empty(0)).ndim == 1
            ),
            None,
        )
        if norm_name is None:
            return
        sum_x = ctx.stats.get(("input_abs_sum", bank.router))
        count_x = ctx.stats.get(("all_tokens_seen", bank.router), 0)
        if sum_x is None or not count_x:
            return
        quantizer = self.quantizer(*unit_format(ctx, bank.router, "moe"))
        weight = getattr(module, in_param(module))
        original = weight.data.clone()
        x_mean = (sum_x.float() / count_x).to(weight.device).unsqueeze(0)
        w_mean = _weight_mean(original, quantizer.group(original.shape[-1]), shared=True)
        inputs = ctx.stats.get(("inputs", bank.router))
        if inputs is None:
            raise ValueError("awq expert search needs bank inputs")
        parent_name = bank.experts.rpartition(".")[0]
        parent = submodule(ctx.layer, ctx.graph, parent_name)
        batches = inputs.to(weight.device, weight.dtype).split(2048)
        refs = [_parent_output(parent, rows) for rows in batches]
        best_scale, best_loss = None, float("inf")
        try:
            for ratio, duo in self.candidates():
                scale = _candidate(x_mean, w_mean, ratio, duo)
                scaled = original.float() * scale[:, None, :]
                qscale, zero = quantizer.params(scaled)
                effective = quantizer.fake(scaled, qscale, zero) / scale[:, None, :]
                weight.data.copy_(effective.to(weight.dtype))
                squared_error = sum(
                    (
                        (_parent_output(parent, rows).float() - ref.float()).square().sum()
                        for rows, ref in zip(batches, refs, strict=True)
                    ),
                    start=torch.zeros((), device=weight.device),
                )
                loss = float(squared_error.item() / inputs.numel())
                if loss < best_loss:
                    best_scale, best_loss = scale, loss
        finally:
            weight.data.copy_(original)
        if best_scale is None:
            return
        self._fold(ctx, LinearGroup((norm_name,), "other", norm=norm_name), best_scale.squeeze(0))
        gate_rows, _ = split(module)
        self._search_expert_up_down(ctx, bank, module, gate_rows)

    def _search_expert_up_down(
        self, ctx: LayerContext, bank: ExpertBank, module: nn.Module, gate_rows: int
    ) -> None:
        """Search one up-to-down scale per expert, priced by each expert's all-token Hessian."""
        from ..arch.adapters.fused_experts import DOWN, in_param

        stats = ctx.stats
        x_sum = stats.get(("down_abs_sum", bank.router))
        hessian = stats.get(("all_down", bank.router))
        count = int(stats.get(("all_tokens_seen", bank.router), 0))
        if x_sum is None or hessian is None or not count:
            return
        weight = getattr(module, in_param(module))
        down = getattr(module, DOWN)
        up = weight.data[:, gate_rows:, :].clone()
        down0 = down.data.clone()
        q = self.quantizer(*unit_format(ctx, bank.router, "moe", down_key(bank)))
        x_mean = x_sum.float().to(up.device) / count
        w_mean = _weight_mean(down0, q.group(down0.shape[-1]))
        hessian = hessian.to(down0.device).float()
        best = torch.ones_like(x_mean)
        best_loss = torch.full((up.shape[0],), float("inf"), device=up.device)
        try:
            for ratio, duo in self.candidates():
                scale = _candidate(x_mean, w_mean, ratio, duo)
                scaled = down0.float() * scale[:, None, :]
                qscale, zero = q.params(scaled)
                effective = q.fake(scaled, qscale, zero) / scale[:, None, :]
                difference = down0.float() - effective.to(down.dtype).float()
                loss = torch.einsum("eoi,eij,eoj->e", difference, hessian, difference)
                loss /= count * down0.shape[1]
                better = loss < best_loss
                best_loss = torch.where(better, loss, best_loss)
                best = torch.where(better[:, None], scale, best)
        finally:
            down.data.copy_(down0)
        weight.data[:, gate_rows:, :].copy_((up.float() / best[:, :, None]).to(weight.dtype))
        down.data.copy_((down0.float() * best[:, None, :]).to(down.dtype))
        routed = stats.get(("expert_hessian", bank.router))
        if isinstance(routed, dict) and "down" in routed:
            h = routed["down"]
            scale = best.to(h.device)
            h.div_(scale[:, :, None] * scale[:, None, :])
        all_down = stats[("all_down", bank.router)]
        scale = best.to(all_down.device)
        all_down.div_(scale[:, :, None] * scale[:, None, :])
        x_sum.div_(best.to(x_sum.device))

    def _search(
        self, ctx: LayerContext, grp: LinearGroup, x: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        """Search one scale vector; `x` is the group's retained inputs, in chunks when every token is kept.

        The parent objective keeps every calibration token (`input_rows = None`), so the engine hands the rows
        over as the batches it collected them in: the all-token mean accumulates over the chunks and the
        objective's MSE is a per-chunk sum divided once (`_mse`), never one concatenated tensor.

        On a multi-card build whose step shares its rows (`Step.shares_rows`), `x` and the attention batches are
        this rank's calibration batches only: the per-batch sums are all gathered in calibration order
        (`_batches`, `_shared_mse`) and every rank reaches the same winner on the one-card arithmetic.
        """
        lin = [cast(nn.Linear, submodule(ctx.layer, ctx.graph, n)) for n in grp.modules]
        dev, dt = lin[0].weight.device, lin[0].weight.dtype
        comm = ctx.comm  # a multi-card build whose step shares the calibration rows (`Step.shares_rows`)
        chunks = [c.to(dev, dt) for c in input_chunks(x)]
        q = self.quantizer(*unit_format(ctx, grp.modules[0], "dense"))
        g = q.group(chunks[0].shape[-1])
        rows = sum(c.shape[0] for c in chunks)
        x_mean = (
            sum((c.float().abs().sum(0) for c in chunks), torch.zeros(chunks[0].shape[-1], device=dev)) / rows
        )
        attn = self._attention_batches(ctx, grp)
        if attn is not None:  # llm-compressor's parent objective for q/k/v
            parent, batches = attn
            parts = torch.stack([b[0].float().abs().reshape(-1, b[0].shape[-1]).sum(0) for b in batches])
            rows = sum(b[0].numel() // b[0].shape[-1] for b in batches)
            if comm is not None:  # every rank's batches, in the order a one-card build holds them
                parts, rows = comm.all_rows(parts), _total(comm, rows, dev)
            x_mean = parts.sum(0) / rows
        elif comm is not None:
            parts, rows = _batches(comm, chunks, dev)
            acc = torch.zeros(parts.shape[1], dtype=parts.dtype, device=dev)
            for part in parts:  # left to right, as the one-card `sum` above
                acc = acc + part
            x_mean = acc / rows
        w0 = [m.weight.data.clone() for m in lin]
        wcat = torch.cat([w.float() for w in w0])
        w_mean = _weight_mean(wcat, g)
        attn_out, linear_run, se = _fast(_attn_out, dev), _fast(_run, dev), _fast(_se, dev)
        if attn is not None:
            ref = [attn_out(parent, xb, kw) for xb, kw in batches]
            run = lambda: [attn_out(parent, xb, kw) for xb, kw in batches]  # noqa: E731
        else:
            parent = _parent(ctx, grp)
            ref = [linear_run(parent, lin, c) for c in chunks]
            run = lambda: [linear_run(parent, lin, c) for c in chunks]  # noqa: E731
        best, best_err = None, float("inf")
        try:
            for r, duo in self.candidates():
                s = _candidate(x_mean[None], w_mean[None], r, duo).squeeze(0)
                for m, w in zip(lin, w0, strict=True):
                    ws = w.float() * s
                    sc, z = q.params(ws)
                    m.weight.data.copy_((q.fake(ws, sc, z) / s).to(dt))
                outs = run()
                err = _mse(outs, ref, se) if comm is None else _shared_mse(comm, outs, ref, se)
                if err < best_err:
                    best, best_err = s, err
        finally:
            for m, w in zip(lin, w0, strict=True):
                m.weight.data.copy_(w)
        assert best is not None
        return best

    def _fold(self, ctx: LayerContext, grp: LinearGroup, s: torch.Tensor) -> None:
        """Producer / s, readers * s, statistics of the input rescaled: the layer output is unchanged."""
        from ..engine.stats import scale_input_stats

        if grp.norm is not None:
            norm = submodule(ctx.layer, ctx.graph, grp.norm)
            p = cast(torch.Tensor, norm.weight)
            off = ctx.model_graph.extra.get("norm_offset", {}).get(grp.norm[len(ctx.graph.module) + 1 :], 0.0)
            p.data.copy_(((p.float() + off) / s.to(p.device) - off).to(p.dtype))
            # the norm emits y = x_hat * (w + off) + b: dividing its output by s divides the bias too, and
            # RMSNorms have no bias attribute
            b = cast(torch.Tensor | None, getattr(norm, "bias", None))
            if b is not None:
                b.data.copy_((b.float() / s.to(b.device)).to(b.dtype))
            readers = ctx.model_graph.extra.get("readers", {}).get(grp.modules[0], grp.modules)
        else:
            assert grp.prev is not None
            prev = submodule(ctx.layer, ctx.graph, grp.prev)
            _scale_rows(prev, 1 / s)
            readers = grp.modules
        for r in readers:
            m = submodule(ctx.layer, ctx.graph, r)
            p = getattr(m, in_param(m))
            p.data.copy_((p.float() * s.to(p.device)).to(p.dtype))
        scale_input_stats(ctx, grp, s)
        for other in ctx.graph.groups:
            if other.modules[0] != grp.modules[0] and other.modules[0] in readers:
                scale_input_stats(ctx, other, s)
        for bank in ctx.graph.banks:  # a norm fold also changes what the experts read
            if isinstance(bank.experts, str) and bank.experts in readers:
                h = ctx.stats.get(("expert_hessian", bank.router))
                if isinstance(h, dict) and "in" in h:
                    sd = s.to(h["in"].device)
                    h["in"].div_(sd[:, None] * sd[None, :])
                all_in = ctx.stats.get(("all_in", bank.router))  # a later GPTQ's all-token blend
                if isinstance(all_in, torch.Tensor):
                    sd = s.to(all_in.device)
                    all_in.div_(sd[:, None] * sd[None, :])
                x_sum = ctx.stats.get(("input_abs_sum", bank.router))
                if isinstance(x_sum, torch.Tensor):
                    x_sum.div_(s.to(x_sum.device))
                inputs = ctx.stats.get(("inputs", bank.router))
                if isinstance(inputs, torch.Tensor):
                    inputs.div_(s.to(inputs.device, inputs.dtype))

    def _attention_batches(self, ctx: LayerContext, grp: LinearGroup) -> tuple[nn.Module, list] | None:
        """(attention module, [(its input, layer kwargs)] per calibration batch) for the parent objective."""
        io = ctx.stats.get(("layer_io", ctx.graph.module))
        if self.objective != "parent" or grp.kind != "attn_in" or grp.norm is None or io is None:
            return None
        parent = submodule(ctx.layer, ctx.graph, grp.modules[0].rpartition(".")[0])
        norm = submodule(ctx.layer, ctx.graph, grp.norm)
        batches = []
        for h, args, kw, _ in io:
            if args:
                raise ValueError("awq parent objective: the layer was called with positional args")
            batches.append((norm(h), {k: v for k, v in kw.items() if k != "hidden_states"}))
        return parent, batches


def _fast(fn: Callable, dev: torch.device) -> Callable:
    """`fn` under torch.compile on a CUDA card, else `fn` itself (the eager reference path).

    The search runs the same forward 21 times per group; eager, its elementwise chain (norms, rotary, activation,
    the squared error) is one kernel per operation and costs about as much as the GEMMs. Compiled, the chain fuses
    into a few kernels. Fusion reorders floating-point arithmetic, so losses differ in the last bits from the eager
    path and a near-tie between candidates can resolve the other way: the build is not byte-identical, and its
    KL is checked against the eager build's instead.
    """
    return _compiled(fn) if dev.type == "cuda" else fn


@cache
def _compiled(fn: Callable) -> Callable:
    """One compiled wrapper per function, so every layer of the same shape reuses its graph."""
    return torch.compile(fn, dynamic=False)


def _se(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Sum of squared differences, in float32."""
    return (out - ref).float().pow(2).sum()


def _attn_out(attn: nn.Module, x: torch.Tensor, kw: dict) -> torch.Tensor:
    out = attn(hidden_states=x, **kw)
    return out[0] if isinstance(out, tuple) else out


def _batches(comm: Any, chunks: list[torch.Tensor], dev: torch.device) -> tuple[torch.Tensor, int]:
    """Every rank's per-batch sums over the group's input, in calibration order, and the rows they cover.

    The rows rank `r` holds are `engine.dist.shard`'s contiguous span of the batches, so the gathered order is
    the one-card batch order and adding the parts left to right is the one-card sum (the search's
    arithmetic must not depend on the card count).
    """
    parts = torch.stack([c.float().abs().sum(0) for c in chunks])
    return comm.all_rows(parts), _total(comm, sum(c.shape[0] for c in chunks), dev)


def _total(comm: Any, n: int, dev: torch.device) -> int:
    """The sum of `n` over every rank (an exact integer count: rows, elements)."""
    box = torch.tensor([[n]], dtype=torch.int64, device=dev)
    return int(comm.all_rows(box).sum())


def _shared_mse(comm: Any, out: list, ref: list, se: Callable = _se) -> float:
    """`_mse`'s list form over every rank's batches: their per-batch sums added in calibration order, once.

    `_search` always prices candidates as lists (one entry per retained batch), so this is the same reduction
    the one-card build makes, term for term — not an average of per-rank means.
    """
    assert isinstance(ref, list)
    sums = torch.stack([se(o, r) for o, r in zip(out, ref, strict=True)])
    parts = comm.all_rows(sums.reshape(-1, 1)).reshape(-1)
    acc = torch.zeros((), dtype=parts.dtype, device=parts.device)
    for part in parts:
        acc = acc + part
    return (acc / _total(comm, sum(r.numel() for r in ref), parts.device)).item()


def _mse(out: torch.Tensor | list, ref: torch.Tensor | list, se: Callable = _se) -> float:
    """Mean squared error over every element (llm-compressor sums per batch, then divides once)."""
    if isinstance(out, torch.Tensor):
        assert isinstance(ref, torch.Tensor)
        return (out - ref).float().pow(2).mean().item()
    assert isinstance(ref, list)
    total = sum(
        (se(o, r) for o, r in zip(out, ref, strict=True)), start=torch.zeros((), device=out[0].device)
    )
    return (total / sum(r.numel() for r in ref)).item()


def _scale_rows(m: nn.Module, s: torch.Tensor) -> None:
    weight = cast(torch.Tensor, m.weight)
    weight.data.copy_((weight.float() * s.to(weight.device)[:, None]).to(weight.dtype))
    bias = cast(torch.Tensor | None, getattr(m, "bias", None))
    if bias is not None:
        bias.data.copy_((bias.float() * s.to(bias.device)).to(bias.dtype))


def _parent(ctx: LayerContext, grp: LinearGroup) -> nn.Module | None:
    """The module whose output prices a candidate, when it takes the input tensor alone (the MLP)."""
    if grp.kind != "mlp_in":
        return None
    name = grp.modules[0].rpartition(".")[0]
    return submodule(ctx.layer, ctx.graph, name)


def _run(parent: nn.Module | None, lin: list[nn.Linear], x: torch.Tensor) -> torch.Tensor:
    if parent is not None:
        return parent(x)
    return torch.cat([m(x) for m in lin], -1)


def _candidate(x_mean: torch.Tensor, w_mean: torch.Tensor, ratio: float, duo: bool) -> torch.Tensor:
    """AWQ candidate scales for one shared mapping or one scale per expert."""
    if w_mean.shape[0] > 1 and x_mean.shape[0] == 1:
        x_mean = x_mean.expand_as(w_mean)
    scale = x_mean.pow(ratio)
    if duo:
        scale = scale / (w_mean.pow(1 - ratio) + 1e-4)
    scale = scale.clamp(min=1e-4)
    scale = scale / (scale.amax(-1, keepdim=True) * scale.amin(-1, keepdim=True)).sqrt()
    return torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))


def _weight_mean(weight: torch.Tensor, group: int, *, shared: bool = False) -> torch.Tensor:
    """Mean normalized weight magnitude per input channel, optionally shared over all experts."""
    normalized = groups(weight.float().abs(), group)
    normalized = normalized / (normalized.amax(-1, keepdim=True) + 1e-6)
    mean = normalized.reshape(weight.shape).mean(-2)
    return mean.mean(0, keepdim=True) if shared else mean


def _parent_output(parent: nn.Module, rows: torch.Tensor) -> torch.Tensor:
    """The MoE block's router-weighted output on cached bank-input rows."""
    out = parent(rows.unsqueeze(0))
    return (out[0] if isinstance(out, tuple) else out).detach()
