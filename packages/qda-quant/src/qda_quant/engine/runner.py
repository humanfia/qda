"""Capture model inputs and run quantization passes over layers."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from ..arch import ExpertBank as ExpertBank
from ..arch import LayerGraph, ModelGraph
from ..arch.adapters import layer_carry
from ..solve.units import QPARAM_TENSORS, SHARE, ExpertShare
from ..step import LayerContext, Step, keeps_float_stream, statistics_of
from .execute import Timers
from .layer_cache import LayerCache
from .stats import (
    ALL_TOKEN_HESSIAN,
    DEFAULT_HESSIAN,
    DENSE_STATS,
    HessianSetting,
    LayerStats,
    input_rows_of,
    input_stride,
    rel,
)
from .stores import RESIDENT, WeightStore
from .token_weights import BatchWeights

_PRIM = (bool, int, float, str, type(None))

#: Statistics a row-sharing step combines itself (`Step.shares_rows`): each rank keeps its own rows of them
#: and the engine neither gathers nor frees them.
LOCAL_STATS = frozenset({"inputs", "layer_io"})


def tree(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply `fn` to every tensor in nested tuples / lists / dicts."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple | list):
        return type(obj)(tree(v, fn) for v in obj)
    if isinstance(obj, dict):
        return {k: tree(v, fn) for k, v in obj.items()}
    return obj


def to_device(obj: Any, device: torch.device, memo: dict[int, torch.Tensor] | None = None) -> Any:
    """Move tensors in `obj` to `device`, moving a tensor shared by several entries once."""
    memo = {} if memo is None else memo

    def move(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) not in memo:
            memo[id(tensor)] = tensor.to(device, non_blocking=True)
        return memo[id(tensor)]

    return tree(obj, move)


class _Stop(Exception):
    pass


class _Recorder(torch.nn.Module):
    """Stands in for decoder layer `i`: records its call, passes the hidden states through."""

    def __init__(
        self, i: int, real: torch.nn.Module, sink: dict, last: int, carry: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self.i, self.sink, self.last = i, sink, last
        self.carry = carry
        for k, v in vars(real).items():  # primitive attributes some models read (attention type, ...)
            if not k.startswith("_") and k != "training" and isinstance(v, _PRIM):
                with contextlib.suppress(Exception):
                    setattr(self, k, v)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor | tuple[Any, ...]:  # noqa: D102
        h, rest = (args[0], args[1:]) if args else (kwargs.pop("hidden_states"), ())
        if self.i == 0:
            self.sink["h"] = h
        self.sink[self.i] = (rest, kwargs)
        if self.i == self.last:
            raise _Stop
        return (h, *(kwargs.get(k) for k in self.carry)) if self.carry else h


@contextlib.contextmanager
def recorders(model: torch.nn.Module, graph: ModelGraph, sink: dict) -> Iterator[None]:
    """Temporarily replace every decoder layer of `model` by a `_Recorder`."""
    saved = []
    n = len(graph.layers)
    for i, lg in enumerate(graph.layers):
        owner, _, attr = lg.module.rpartition(".")
        parent = model.get_submodule(owner)
        parent_dyn = cast(Any, parent)
        real = cast(
            torch.nn.Module,
            parent_dyn[int(attr)] if attr.isdigit() else getattr(parent, attr),
        )
        saved.append((parent, attr, real))
        stub = _Recorder(i, real, sink, n - 1, layer_carry(graph))
        if attr.isdigit():
            parent_dyn[int(attr)] = stub
        else:
            setattr(parent, attr, stub)
    try:
        yield
    finally:
        for parent, attr, real in saved:
            if attr.isdigit():
                cast(Any, parent)[int(attr)] = real
            else:
                setattr(parent, attr, real)


@dataclass
class Slot:
    """One calibration batch: its current hidden states and every layer's call arguments."""

    h: torch.Tensor
    calls: list[tuple[tuple, dict]]
    device: torch.device
    out: torch.Tensor | None = None  # the float layer output, when a step needs it
    weights: BatchWeights | None = None
    float_h: torch.Tensor | None = None
    carry: dict[str, Any] = field(default_factory=dict)
    float_carry: dict[str, Any] = field(default_factory=dict)

    def inp(self) -> torch.Tensor:
        """The hidden states on the slot's card."""
        return self.h.to(self.device, non_blocking=True)


def capture(
    model: torch.nn.Module,
    graph: ModelGraph,
    batches: Sequence[torch.Tensor],
    device: torch.device,
    host_acts: bool = False,
    keep_float: bool = False,
    inputs: Sequence[dict[str, torch.Tensor]] | None = None,
) -> list[Slot]:
    """Run the model's forward on each batch with the layers as recorders.

    The model's non-layer modules (embedding, rotary, masks, a vision tower) must already be on `device`.
    `inputs`, one per batch, are passed to the forward with the ids (an image sample's ``pixel_values``).

    Returns:
        One `Slot` per batch, in batch order.
    """
    sink: dict = {}
    slots: list[Slot] = []
    n = len(graph.layers)
    with recorders(model, graph, sink), torch.no_grad():
        for b, ids in enumerate(batches):
            sink.clear()
            extra = {} if inputs is None else to_device(inputs[b], device)
            with contextlib.suppress(_Stop):
                model(input_ids=ids.to(device), use_cache=False, **extra)
            if sorted(k for k in sink if k != "h") != list(range(n)):
                raise RuntimeError(f"the model's forward called {len(sink) - 1} of {n} decoder layers")
            memo: dict[int, torch.Tensor] = {}
            calls = [to_device(sink[i], device, memo) for i in range(n)]
            h = sink["h"].detach()
            h = h.to("cpu").pin_memory() if host_acts and h.is_cuda else h.clone()
            float_h = h.clone() if keep_float else None
            if float_h is not None and h.is_pinned():
                float_h = float_h.pin_memory()
            carried = {k: calls[0][1][k] for k in layer_carry(graph) if k in calls[0][1]}
            slots.append(
                Slot(
                    h=h,
                    calls=calls,
                    device=device,
                    float_h=float_h,
                    carry=carried,
                    float_carry=tree(carried, lambda t: t.detach().clone()) if keep_float else {},
                )
            )
    return slots


def merge(parts: Sequence[dict[tuple[str, str], Any]], device: torch.device) -> dict[tuple[str, str], Any]:
    """Combine per-card statistics in card order.

    Sums for Hessians and counts, max for absmax, concatenation for input samples.
    """
    out = dict(parts[0])
    for p in parts[1:]:
        for k, v in p.items():
            out[k] = _merge_statistic(k[0], out[k], v, device)
    return {k: tree(v, lambda t: t.to(device)) for k, v in out.items()}


def _merge_statistic(name: str, left: Any, right: Any, device: torch.device) -> Any:
    if isinstance(left, dict):
        return {key: _merge_statistic(name, left[key], right[key], device) for key in left}
    if left is None or right is None:
        return left if right is None else right
    if name == "act_absmax":
        return torch.maximum(left.to(device), right.to(device))
    if name == "inputs":
        return torch.cat([left.to(device), right.to(device)])
    if isinstance(left, torch.Tensor):
        return left.to(device) + right.to(device)
    return left + right


@dataclass
class LayerIO:
    """Per calibration batch ``(hidden_states, args, kwargs, float_output)`` of one layer, on `device`.

    For steps that must re-run the layer (AutoRound, GSQ, loss-based AWQ). The engine owns the batching;
    the hidden states are the layer's (quantized-stream) inputs. `float_output` is the float layer on those
    inputs or, with `float_stream`, on the float stream.
    """

    slots: list[Slot]
    index: int
    device: torch.device
    float_stream: bool = False

    def __iter__(self) -> Iterator[tuple[torch.Tensor, tuple, dict, torch.Tensor | None]]:
        for s in self.slots:
            args, kw = to_device(s.calls[self.index], self.device)
            out = s.float_h if self.float_stream else s.out
            yield s.h.to(self.device), args, kw, None if out is None else out.to(self.device)

    def __len__(self) -> int:
        return len(self.slots)


@dataclass
class GatheredLayerIO:
    """Every rank's calibration batches of one layer, merged on rank 0 in rank order.

    `engine.dist.Comm.gather_layer_io` builds it for a step that declares `multi_card_layer_io`, so the step
    re-runs the layer over the rows a one-card build holds; the batches share the one call the gather checked.
    `targets` is each batch's float output when every rank kept one (GSQ fits it), else None.
    """

    batches: list[torch.Tensor]
    args: tuple
    kwargs: dict
    device: torch.device
    targets: list[torch.Tensor] | None = None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, tuple, dict, torch.Tensor | None]]:
        for i, h in enumerate(self.batches):
            yield (
                h.to(self.device),
                self.args,
                self.kwargs,
                (None if self.targets is None else self.targets[i].to(self.device)),
            )

    def __len__(self) -> int:
        return len(self.batches)


def call(layer: torch.nn.Module, h: torch.Tensor, args: tuple, kwargs: dict) -> torch.Tensor:
    """One decoder layer forward; returns the hidden states."""
    out = layer(h, *args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _carried_call(layer, h, args, kwargs, names):
    """Call a layer and label only the auxiliary outputs its adapter declares."""
    out = layer(h, *args, **kwargs)
    if not names:
        return (out[0] if isinstance(out, tuple) else out), {}
    if not isinstance(out, tuple) or len(out) != len(names) + 1:
        raise RuntimeError(f"layer carry {names}: expected hidden states plus {len(names)} outputs")
    return out[0], dict(zip(names, out[1:], strict=True))


@dataclass
class Pass:
    """One pass over the layers for consecutive layer steps, on one card (one rank).

    Attributes:
        model: the model, its weights where `store` keeps them.
        graph: its `ModelGraph`.
        steps: the layer steps of this pass, in recipe order.
        device: this rank's card.
        make_ctx: builds the `LayerContext` for (layer module, layer graph, stats).
        timers: the build's timers.
        on_layer: called with the context after the steps (the writer), or None.
        store: where the weights live (`engine.stores`); it places each layer for its turn and releases it after.
        act_cuts: per unit key (a group's first module), the input round trip its format's kernel plan names
            (`solve.quant.ACT_CUTS`), installed on that unit's linears for both forwards; a dense group absent
            from the map reads its input as it is.
        bank_cut: that round trip when the MoE plan reads a cut bank (`engine.execute.bank_cut`: NVFP4 only),
            installed on the expert banks; a fused bank's down input is formed inside its forward, so its
            statistics cut it (`LayerStats.cut`) while the stream keeps it uncut. None leaves banks untouched.
        extra_needs: statistics the build's format asks for beside the steps' (NVFP4's input ranges).
        after_steps: called with the context once a layer's steps have written its entries, before the entries
            are cached and written; it completes them from the layer's statistics (NVFP4's input scales).
        hessian: Hessian arithmetic and accumulation order (`HessianSetting`).
        comm: the process group when the build runs one process per card (`engine.dist.Comm`), or None.
        total_rows: calibration rows over all ranks (sets the input-sample stride); default: this rank's.
        expert_chunks: on one card, gather and solve each layer's expert statistics in this many chunks of
            experts (`engine.residency.Residency.expert_chunks`); 1 gathers them whole.
    """

    model: torch.nn.Module
    graph: ModelGraph
    steps: Sequence[Step]
    device: torch.device
    make_ctx: Callable[[torch.nn.Module, LayerGraph, dict], LayerContext]
    timers: Timers
    on_layer: Callable[[LayerContext], None] | None = None
    layer_context: Callable[[torch.nn.Module, LayerGraph], contextlib.AbstractContextManager] | None = None
    store: WeightStore = RESIDENT
    act_cuts: dict[str, Callable[[torch.Tensor], torch.Tensor]] = field(default_factory=dict)
    bank_cut: Callable[[torch.Tensor], torch.Tensor] | None = None
    extra_needs: frozenset[str] = frozenset()
    after_steps: Callable[[LayerContext], None] | None = None
    comm: Any = None
    total_rows: int | None = None
    token_weights: list[BatchWeights] | None = None
    hessian: HessianSetting = DEFAULT_HESSIAN
    expert_chunks: int = 1
    #: solved layers to reuse and keep (`engine.layer_cache`); None solves every layer
    cache: LayerCache | None = None
    log: Callable[[str], None] = field(default=lambda _: None)

    def needs(self) -> frozenset[str]:
        """Union of the steps' statistics (read on the instances: settings may change them) and the format's."""
        out: set[str] = set(self.extra_needs)
        for s in self.steps:
            out |= set(s.needs)
        return frozenset(out)

    def run(self, slots: list[Slot]) -> None:
        """Quantize every layer in order; `slots` carry this rank's hidden states from layer to layer."""
        needs = self.needs()
        rw = any(getattr(s, "router_weighted", False) for s in self.steps)
        rows = self.total_rows or sum(s.h.shape[0] * s.h.shape[1] for s in slots)
        cap = input_rows_of(self.steps)
        stride = input_stride(rows, cap)
        keep = bool(needs & {"outputs", "layer_io"})
        provided = {name: stat.provide for name, stat in statistics_of(needs).items() if stat.provide}
        root = self.comm is None or self.comm.root
        # a step that shares its rows keeps every rank's own calibration batches and combines its sums itself
        shares = any(getattr(s, "shares_rows", False) for s in self.steps)
        # a step that re-runs layers over every calibration batch gets the ranks' `layer_io` batches gathered
        gathers_io = (
            self.comm is not None and "layer_io" in needs and any(s.multi_card_layer_io for s in self.steps)
        )
        # a step that trains across the cards (`engine.ep`) keeps each rank's own batches on that rank: its
        # experts see the other ranks' tokens through the all-to-all, not through a merged batch list
        ep_io = any(getattr(s, "expert_parallel", False) for s in self.steps)
        # the steps a rank runs when no share of the experts splits the layer: the row-sharing ones. A step that
        # refines what an earlier one wrote (`gsq` after `gptq`) starts from the parameters that step wrote on
        # rank 0, which it asks every rank for itself (`Comm.broadcast_qparams`); the earlier step must not run
        # here, where the statistics it reads were not gathered
        local_steps = [s for s in self.steps if getattr(s, "shares_rows", False)]
        d = self.device
        for i, lg in enumerate(self.graph.layers):
            layer = self.model.get_submodule(lg.module)
            with self.timers.phase("place"):
                self.store.place(layer, lg, d)
            context = self.layer_context(layer, lg) if self.layer_context else contextlib.nullcontext()
            if self.cache is not None and (entry := self.cache.find(i, lg.module)) is not None:
                with context:
                    self._reuse(layer, lg, i, slots, entry)
                self.store.release(layer)
                self.log(f"layer {i} reused")
                continue
            with context:
                stats = {}
                share = self._chunk(layer, lg)
                if slots or needs:
                    with self.timers.phase("calib_forward"):
                        for slot, weights in zip(
                            slots, self.token_weights or [None] * len(slots), strict=True
                        ):
                            slot.weights = weights
                        layer_stats = LayerStats(
                            layer,
                            lg,
                            needs,
                            d,
                            stride,
                            rw,
                            self.hessian,
                            cap,
                            all_token_experts=any(getattr(s, "all_token_experts", False) for s in self.steps),
                            share=share,
                            all_token_hessian=any(getattr(s, "all_token_hessian", False) for s in self.steps),
                            cut=self.bank_cut,
                        )
                        stats = self._forward(layer, lg, i, slots, layer_stats, keep)
                split = self._splits(lg)
                # a row-shared layer keeps this rank's own rows; a layer with routed experts gathers its
                # statistics as the other steps do (the expert search reads global sums)
                share_rows = self.comm is not None and shares and not lg.banks
                gather_io = gathers_io and not share_rows and not ep_io
                if gather_io:
                    stats[("layer_io", lg.module)] = LayerIO(slots, i, d)
                if self.comm is not None:
                    with self.timers.phase("reduce"):
                        merged = self.comm.reduce_stats(
                            stats,
                            cap,
                            gather_layer_io=gather_io,
                            keep_local=LOCAL_STATS if share_rows else frozenset(),
                        )
                        banks = {b.router for b in lg.banks}
                        shared_stats = [
                            k
                            for k in stats
                            if k[1] in banks and k[0] in ("expert_hessian", "n_tokens", *ALL_TOKEN_HESSIAN)
                        ]
                        if split:  # every card solves a share of the experts: all need the reduced bank stats
                            self.comm.broadcast_stats(merged if root else stats, shared_stats)
                        # other ranks: a row-sharing step keeps its own statistics; a share solve reads only
                        # the broadcast bank statistics; anything else runs on rank 0 alone and drops them
                        if share_rows:
                            stats = merged if root else stats
                        else:
                            stats = merged if root else {k: stats[k] for k in shared_stats if split}
                if split:
                    assert self.comm is not None  # a split solve means several cards
                    stats[(SHARE, lg.module)] = ExpertShare(self.comm.rank, self.comm.world)
                    if not root:
                        if ep_io:  # a card trains on its own batches; its experts see the others' through ep
                            if "outputs" in needs:
                                stats[("outputs", lg.module)] = [s.out for s in slots]
                            stats[("layer_io", lg.module)] = LayerIO(slots, i, d)
                            for name, provide in provided.items():
                                stats[(name, lg.module)] = provide(slots, i, d)
                        ctx = self.make_ctx(layer, lg, stats)
                        ctx.comm = self.comm
                        with self.timers.phase("steps"):
                            for step in self.steps:
                                step.apply_layer(ctx)
                        with self.timers.phase("gather"):
                            self.comm.gather_experts(ctx.qparams, layer, lg.module)
                        del ctx
                elif share_rows and not root:
                    # a row-sharing step: every rank prices its own share of the calibration rows and combines
                    # the per-rank sums through `LayerContext.comm`; rank 0's fold is broadcast after the layer
                    if "outputs" in needs:
                        stats[("outputs", lg.module)] = [s.out for s in slots]
                    stats[("layer_io", lg.module)] = LayerIO(slots, i, d)
                    for name, provide in provided.items():
                        stats[(name, lg.module)] = provide(slots, i, d)
                    ctx = self.make_ctx(layer, lg, stats)
                    ctx.comm = self.comm
                    with self.timers.phase("steps"):
                        for step in local_steps:
                            step.apply_layer(ctx)
                    del ctx
                if root:
                    if "outputs" in needs:
                        stats[("outputs", lg.module)] = [s.out for s in slots]
                    if not gather_io:  # a gathered merge already holds every rank's batches
                        stats[("layer_io", lg.module)] = LayerIO(slots, i, d)
                    for name, provide in provided.items():
                        stats[(name, lg.module)] = provide(slots, i, d)
                    if share is not None:
                        stats[(SHARE, lg.module)] = share
                    ctx = self.make_ctx(layer, lg, stats)
                    if share_rows or split:
                        ctx.comm = self.comm
                    with self.timers.phase("steps"):
                        for step in self.steps:
                            step.apply_layer(ctx)
                    if share is not None:
                        ctx = self._later_chunks(ctx, layer_stats, slots, i)
                    if split:
                        assert self.comm is not None  # a split solve means several cards
                        with self.timers.phase("gather"):
                            self.comm.gather_experts(ctx.qparams, layer, lg.module)
                    if self.after_steps is not None:
                        self.after_steps(ctx)
                    if self.cache is not None:
                        with self.timers.phase("cache"):
                            self.cache.save(i, layer, lg.module, ctx.qparams)
                    if self.on_layer is not None:
                        with self.timers.phase("write"):
                            self.on_layer(ctx)
                    settle(ctx.qparams, lg.module, written=self.on_layer is not None)
                    del ctx
                del stats
                if self.comm is not None:
                    with self.timers.phase("broadcast"):
                        self.comm.broadcast_module(layer)
                if slots:
                    with self.timers.phase("propagate"):
                        self._forward(
                            layer,
                            lg,
                            i,
                            slots,
                            LayerStats(layer, lg, frozenset(), d, cut=self.bank_cut),
                            propagate=True,
                        )
            self.store.release(layer)
            self.log(f"layer {i} done")

    def _reuse(self, layer: torch.nn.Module, lg: LayerGraph, i: int, slots: list[Slot], entry: Any) -> None:
        """Layer `i` from a cache entry: its parameters and codes back, written, and run on the slots."""
        assert self.cache is not None
        float_needs = frozenset(
            name for name, stat in statistics_of(self.needs()).items() if stat.float_stream
        )
        if slots and float_needs:
            # Advance the original stream before restoring the solved weights; later solves read its targets.
            with self.timers.phase("calib_forward"):
                self._forward(
                    layer,
                    lg,
                    i,
                    slots,
                    LayerStats(layer, lg, float_needs, self.device, cut=self.bank_cut),
                )
        ctx = self.make_ctx(layer, lg, {})
        with self.timers.phase("cache"):
            self.cache.restore(entry, layer, lg.module, ctx.qparams)
        if self.on_layer is not None:
            with self.timers.phase("write"):
                self.on_layer(ctx)
        settle(ctx.qparams, lg.module, written=self.on_layer is not None)
        if slots:
            with self.timers.phase("propagate"):
                self._forward(
                    layer,
                    lg,
                    i,
                    slots,
                    LayerStats(layer, lg, frozenset(), self.device, cut=self.bank_cut),
                    propagate=True,
                )

    def _chunk(self, layer: torch.nn.Module, lg: LayerGraph) -> ExpertShare | None:
        """The first chunk (the last share) of a layer whose expert statistics are gathered in chunks, or None."""
        n = self.expert_chunks
        if n == 1 or self.comm is not None or not chunks_experts(layer, lg, self.steps, n):
            return None
        return ExpertShare(n - 1, n, local=True)

    def _later_chunks(
        self, ctx: LayerContext, layer_stats: LayerStats, slots: list[Slot], i: int
    ) -> LayerContext:
        """Gather and solve the layer's other chunks of experts, after `ctx` solved the last one.

        Chunks go from last to first: chunk 0's solve also quantizes the dense groups, from the statistics the
        first forward gathered, so every other chunk reads its inputs from the float layer. Fused banks replay
        the bank calls that forward recorded; unfused ones run the layer again. A chunk's codes move to the
        host once it is solved; the returned context holds every bank whole, as a one-chunk solve leaves it.
        """
        layer, lg, module = ctx.layer, ctx.graph, ctx.graph.module
        n = ctx.stats[(SHARE, module)].world
        routers = {b.router for b in ctx.graph.banks}
        kept = {k: v for k, v in ctx.stats.items() if k[1] not in routers and k[0] != SHARE}
        parts: dict[str, list[dict[str, Any]]] = {}
        for c in range(n - 2, -1, -1):
            _stash(ctx.qparams, module, parts, solved=True)
            ctx.stats.clear()  # the previous chunk's statistics go before the next chunk's are allocated
            share = ExpertShare(c, n, local=True)
            with self.timers.phase("replay"):
                stats = layer_stats.replay(share, lambda st: self._forward(layer, lg, i, slots, st))
            stats.update(kept)
            stats[(SHARE, module)] = share
            ctx = self.make_ctx(layer, lg, stats)
            with self.timers.phase("steps"):
                for step in self.steps:
                    step.apply_layer(ctx)
        _stash(ctx.qparams, module, parts, solved=False)
        for name, chunk in parts.items():
            chunk.sort(key=lambda qp: qp["experts"][0])
            ctx.qparams[name] = _joined(chunk, layer.get_parameter(rel(name, module)))
        return ctx

    def _splits(self, lg: LayerGraph) -> bool:
        """Whether this layer's solve is split across the cards (`splits_experts`)."""
        world = 1 if self.comm is None else self.comm.world
        return splits_experts(self.model.get_submodule(lg.module), lg, self.steps, world)

    def _forward(
        self,
        layer: torch.nn.Module,
        lg: LayerGraph,
        i: int,
        slots: list[Slot],
        st: LayerStats,
        keep: bool = False,
        propagate: bool = False,
    ) -> dict:
        """Run the layer over this rank's slots.

        Calibration feeds the statistics in `st` (and keeps the float outputs if `keep`); propagation
        makes the output the slot's hidden states.
        """
        d = self.device
        float_stream = keeps_float_stream(st.needs)
        names = layer_carry(self.graph)
        t = time.perf_counter()
        with torch.no_grad(), _on(d), self._acts(layer, lg), st:
            for s in slots:
                args, kwargs = s.calls[i]
                kwargs = {**kwargs, **to_device(s.carry, d)}
                s.calls[i] = (args, kwargs)
                st.weights = s.weights
                st.float_inputs = {}
                if float_stream and s.float_h is not None:
                    float_kwargs = {**kwargs, **to_device(s.float_carry, d)}
                    next_float = {}

                    def float_call(
                        call_args=args,
                        call_kwargs=float_kwargs,
                        sink=next_float,
                        float_h=s.float_h,
                    ):
                        out, carried = _carried_call(layer, float_h.to(d), call_args, call_kwargs, names)
                        sink.update(carried)
                        return out

                    float_out = st.capture_float(float_call)
                    s.float_h = (
                        s.float_h.copy_(float_out, non_blocking=True)
                        if s.float_h.is_pinned()
                        else float_out.detach()
                    )
                    s.float_carry = next_float
                    if s.float_h.is_pinned():
                        s.float_carry = tree(s.float_carry, lambda t: t.detach().to("cpu").pin_memory())
                o, next_carry = _carried_call(layer, s.inp(), args, kwargs, names)
                if propagate:
                    s.h = s.h.copy_(o, non_blocking=True) if s.h.is_pinned() else o
                    if s.h.is_pinned():
                        s.carry = tree(next_carry, lambda t: t.detach().to("cpu").pin_memory())
                    else:
                        s.carry = next_carry
                    s.out = None
                elif keep:
                    s.out = o
            if d.type == "cuda":
                torch.cuda.synchronize(d)
        self.timers.add(f"busy:{'propagate' if propagate else 'calib'}", time.perf_counter() - t)
        return st.result()

    def _acts(self, layer: torch.nn.Module, lg: LayerGraph) -> contextlib.AbstractContextManager:
        if not self.act_cuts and self.bank_cut is None:
            return contextlib.nullcontext()
        from .execute import act_inputs

        return act_inputs(self.act_cuts, layer, lg, bank_qdq=self.bank_cut)


def splits_experts(layer: torch.nn.Module, lg: LayerGraph, steps: Sequence[Step], world: int) -> bool:
    """Whether a layer's GPTQ-style solve is split across `world` cards.

    Only with several cards, fused expert banks of at least `world` experts each (every card gets one or
    more), and steps that all solve a share (``splits_experts``). Otherwise rank 0 solves the whole layer.
    """
    from ..arch import submodule
    from ..arch.adapters import fused_experts

    return (
        world > 1
        and bool(lg.banks)
        and all(
            isinstance(b.experts, str)
            and fused_experts.in_weight(submodule(layer, lg, b.experts)).shape[0] >= world
            for b in lg.banks
        )
        and all(getattr(s, "splits_experts", False) for s in steps)
    )


def _stash(qparams: dict[str, dict[str, Any]], module: str, parts: dict[str, list], solved: bool) -> None:
    """Move a chunk's fused-bank entries of layer `module` into `parts`, on the host.

    With `solved`, every other entry of the layer (an unfused bank's experts; the dense groups solve last) moves to
    the host too.
    """
    for name in [k for k in qparams if k.startswith(module + ".")]:
        if "experts" in qparams[name]:
            parts.setdefault(name, []).append(tree(qparams.pop(name), lambda t: t.to("cpu")))
        elif solved:
            qparams[name] = tree(qparams[name], lambda t: t.to("cpu"))


def _joined(chunks: list[dict[str, Any]], p: torch.nn.Parameter) -> dict[str, Any]:
    """One bank's qparams entry from its chunks' entries (each with its ``"experts"`` span), in expert order."""
    from ..intake.fingerprint import tensor_digest

    fields = [f for f in QPARAM_TENSORS if f in chunks[0]]
    whole = {f: torch.cat([qp[f] for qp in chunks]) for f in fields}
    return {**whole, "format": chunks[0]["format"], "weight_check": (id(p), tensor_digest(p))}


def chunks_experts(layer: torch.nn.Module, lg: LayerGraph, steps: Sequence[Step], n: int) -> bool:
    """Whether one card can gather and solve this layer's expert statistics in `n` chunks of experts.

    Only with a bank of at least `n` experts in every bank (each chunk gets one or more), and steps that all
    solve a share (``splits_experts``).
    """
    from ..arch import submodule
    from ..arch.adapters import fused_experts

    def experts(b: ExpertBank) -> int:
        if isinstance(b.experts, str):
            return fused_experts.in_weight(submodule(layer, lg, b.experts)).shape[0]
        return len(b.experts)

    return (
        n > 1
        and bool(lg.banks)
        and all(experts(b) >= n for b in lg.banks)
        and all(getattr(s, "splits_experts", False) for s in steps)
    )


def settle(qparams: dict[str, dict[str, Any]], layer: str, written: bool) -> None:
    """After a layer's turn: drop its qparams entries once written, else keep them on the host for the writer.

    A layer pass without the writer (an earlier pass of a multi-pass recipe) leaves its entries for the last
    pass; a later step on the same module replaces them.
    """
    for k in [k for k in qparams if k.startswith(layer + ".")]:
        if written:
            del qparams[k]
        else:
            qparams[k] = {f: v.to("cpu") if isinstance(v, torch.Tensor) else v for f, v in qparams[k].items()}


def _on(d: torch.device) -> contextlib.AbstractContextManager:
    return torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext()


__all__ = [
    "DENSE_STATS",
    "ExpertBank",
    "LayerIO",
    "LayerStats",
    "Pass",
    "Slot",
    "capture",
    "merge",
    "recorders",
    "rel",
    "chunks_experts",
    "splits_experts",
]
