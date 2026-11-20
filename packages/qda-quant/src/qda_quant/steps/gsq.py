"""GSQ: Gumbel-Softmax rounding refinement of an earlier quantizing step's codes.

Use ``gptq`` or ``rtn`` before ``gsq`` in the recipe. The engine captures float layer targets once;
GSQ trains logits over five code shifts and group scales with Lion. Zero points stay fixed.

The algorithm is derived from IST-DASLab/GSQ, commit
``03fc16484c369e3127225615d5e03e8d3a6043e3``, by Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan,
Michael Helcig, Eldar Kurtić, Max Kleinegger, and Dan Alistarh, as described in "GSQ: Highly-Accurate
Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling" (arXiv:2604.18556).
The upstream code is licensed under Apache-2.0. Lion follows ``lion_pytorch`` 0.2.5, licensed under MIT.
See THIRD_PARTY_NOTICES.md for source links and the Lion license notice.
"""

from __future__ import annotations

import contextlib
import logging
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator
from torch.func import functional_call

from ..arch import submodule
from ..format import Format
from ..solve.units import ALL, SHARE, BoundUnit
from ..step import LayerContext, Refused
from ._attention import attention_kernel
from ._gumbel import GumbelRounding, Lion, cosine_lr
from ._layer_fit import LayerFit
from .autoround import _split


def _loss(layer: torch.nn.Module, replacements: dict, item: tuple) -> torch.Tensor:
    hidden, args, kwargs, reference = item
    output = functional_call(layer, replacements, (hidden, *args), kwargs, strict=False)
    output = output[0] if isinstance(output, tuple) else output
    return (output.float() - reference.float()).square().mean()


#: bytes per weight a `GumbelRounding` holds: five code shifts and their centring temporary in float32 (60,
#: measured: 38.3 GB for a 638M-parameter Nemotron bank unit) plus the state kept for the rest of the layer.
BYTES_PER_WEIGHT = 80


def _spans(n: int, count: int) -> list[slice]:
    """The expert spans of a unit that one `GumbelRounding` can hold at a time."""
    if count >= n:
        return [slice(0, n)]
    return [slice(start, min(start + count, n)) for start in range(0, n, count)]


def _whole(
    unit: BoundUnit, weights: list[tuple[slice, torch.Tensor]], wholes: dict[int, torch.Tensor]
) -> None:
    """Write a fused bank's served pieces into a copy of its parameter, which `functional_call` replaces whole.

    A bank is one parameter (``experts.up_proj`` / ``experts.down_proj``); a piece of its experts, or a share of
    them on a card, is a slice of its rows, and `_split` would hand `functional_call` a short tensor for a
    whole-sized parameter. The pieces go into the rows they cover, in the parameter's own expert order.
    """
    p = unit.params[0]
    whole = wholes.get(id(p))
    if whole is None:
        whole = wholes[id(p)] = p.data.clone()
    base = 0 if unit.experts == ALL else unit.experts.start
    for span, weight in weights:
        whole[base + span.start : base + span.stop] = unit.stored(weight).to(p.dtype)


#: Levels of upstream's int path (``GumbelQuantizerInt``, ``bits=3``): ``qmax - qmin`` at 3 bits symmetric.
#: ``scales_lr`` is upstream's learning rate for that width, and `GSQ._scales_lr` holds its RELATIVE size
#: at any other width (see there).
UPSTREAM_LEVELS = 7


def _project_scales(optimizer: Lion) -> None:
    """Put every trained group scale back in its grid's domain, after the step that may have left it.

    Lion moves a scale by exactly ``+-lr`` whatever its size, so a scale can cross zero. The absmax grid's
    scales are positive (the codes were solved against a positive scale, and `Grid.usable` is what the rest
    of the engine writes), so a negative one is not a worse fit, it is outside the format: it flips the sign
    of every weight in its group while the codes stay put.
    """
    with torch.no_grad():
        for group in optimizer.param_groups:
            quantizer = group.get("grid")
            if quantizer is None:
                continue
            for parameter in group["params"]:
                parameter.copy_(quantizer.usable(parameter))


def _initial(ctx: LayerContext, unit: BoundUnit, key: str) -> torch.Tensor:
    def read(name: str) -> torch.Tensor:
        if name not in ctx.qparams:
            raise Refused("gsq refines existing codes; put gptq or rtn before gsq in the recipe")
        return ctx.qparams[name][key].float()

    if unit.part is None:
        return torch.cat([read(name) for name in unit.names])[None]
    if isinstance(unit.names, str):
        return read(unit.names)
    return torch.stack([torch.cat([read(name) for name in names]) for names in unit.names])


class GSQ(LayerFit):
    """Refine a quantized layer with Gumbel-softmax rounding and trainable group scales.

    Reproducibility: noise band by default (docs/reproduce.md): quote a 3-seed mean and band. The cuDNN attention
    backward varies in its last bits and Lion takes signs of the result. ``reproducible`` removes that source.

    Attributes:
        chunk: experts of a fused bank one rounding unit holds, default 0 (as many as half the free card holds).
            A bank too big for one rounding is trained in pieces, one piece at a time, the other pieces frozen at
            their codes: the loss is still the layer's own output, but the pieces no longer share one optimizer
            state, so say so in the build record.
        epochs: training epochs, default 10.
        batches: engine calibration batches accumulated per Lion step, default 16; capped to train size.
            The GLOBAL batch, as upstream's ``data.batch_size`` is: on `world` cards each takes
            ``batches // world`` of it, so the step count, the schedules and the result do not depend on
            how many cards the fit runs on.
        validation: engine batches held out for soft/hard validation, default 16; 0 trains on all batches.
        logits_lr: initial logit learning rate, default 2e-4.
        scales_lr: initial group-scale learning rate, default 1e-4.
        weight_decay: logit weight decay, default 1.0; scales have none.
        betas: Lion momentum coefficients, default (0.9, 0.95).
        temperature: initial and final Gumbel temperatures, default (2.0, 0.05).
        logit_scale: initial and final logit multipliers, default (100.0, 500.0).
        std: initial logit noise standard deviation, default 0.01.
        strength: initial preference for the starting codes, default 6.0.
        minimum_lr: cosine schedule's final learning-rate fraction, default 0.1.
        warmup: warmup optimizer steps, default 0.
        logits_dtype: logit and momentum dtype, default bfloat16.
        seed: per-layer random seed, default 0.
        target: the output to reproduce; inputs are the quantized stream either way. ``quant_stream``
            (default): the float layer on the quantized stream, the same stream upstream GSQ fits.
            ``float_stream``: the float layer on the float stream, so each layer also corrects the error it
            inherits; the engine keeps the float stream.
        reproducible: train with torch's math attention instead of cuDNN's, so a repeat build writes the same
            bytes. The measured build took 1.1x as long at one epoch, at the same peak card memory; more epochs
            were not measured (docs/reproduce.md).
        expert_parallel: when a multi-card build splits a fused bank's experts across the cards, train each
            card's own whole experts and run the layer through `engine.ep`, so every card's tokens reach every
            expert and each expert sees every calibration row (upstream GSQ's `AllToAllTokens` dispatch); the
            cards' losses sum, no gradient is averaged, and each card trains its block as ONE joint unit (no
            `chunk` cut). This is the mode a bank layer uses across cards: the data parallelism below cannot
            run there, because the layer loss needs every expert of the layer differentiable. False keeps each
            card on its own rows only, which a bank layer cannot do either (it then needs `chunk` under what
            the card holds).
    """

    name = "gsq"
    experimental = True
    #: A multi-card build gathers every card's calibration batches (with their float targets) for this step,
    #: so rank 0 fits the layer over the rows a one-card build fits, in the same order and with the same seed.
    multi_card_layer_io: ClassVar[bool] = True
    #: A fused bank is solved in the engine's chunks of experts (``expert_chunks``) like gptq's: one unit of a
    #: 128-expert bank holds five code-shift tensors plus a centring temporary per expert, ~38 GB at 638M
    #: parameters, which one card cannot hold at once (a Nemotron-3-Nano 2-layer smoke build).
    splits_experts: ClassVar[bool] = True
    #: Data parallelism, as upstream GSQ runs it (`src/trainer.py` at 03fc16484c): every rank trains on its own
    #: contiguous shard of the calibration batches and the Lion step takes the mean of the ranks' gradients
    #: (`Comm.average_grads`), so a card holds its own rows, not every rank's. The codes then match a one-card
    #: build within fp tolerance rather than bitwise: the gradient sum takes a different order.
    shares_rows: ClassVar[bool] = True
    #: GSQ refines codes an earlier step wrote, and that step runs on rank 0 alone under `shares_rows`, so the
    #: fit starts from rank 0's quantization parameters sent to every rank (`Comm.broadcast_qparams`).
    starts_from_root: ClassVar[bool] = True
    #: `_train` builds every unit of the layer before it trains any and keeps all of their rounding state to the
    #: end of the layer: only a bank it cuts into pieces trains them one set at a time, and an expert-parallel
    #: layer is never cut (`count = 1 << 40`). `engine.memory._solves` marks such a layer's solves `joint`, so
    #: the plan holds every unit's workspace at once instead of the largest one's (xy-2140).
    joint_units: ClassVar[bool] = True
    chunk: int = Field(default=0, ge=0)
    epochs: int = Field(default=10, ge=1)
    batches: int = Field(default=16, ge=1)
    validation: int = Field(default=16, ge=0)
    logits_lr: float = Field(default=2e-4, gt=0)
    scales_lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    temperature: tuple[float, float] = (2.0, 0.05)
    logit_scale: tuple[float, float] = (100.0, 500.0)
    std: float = Field(default=0.01, gt=0)
    strength: float = Field(default=6.0, ge=0)
    minimum_lr: float = Field(default=0.1, ge=0, le=1)
    warmup: int = Field(default=0, ge=0)
    logits_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    seed: int = 0
    reproducible: bool = False
    expert_parallel: bool = True

    def _scales_lr(self, fmt: Format) -> float:
        """`scales_lr` at `fmt`'s bit width, holding upstream's RELATIVE step.

        Lion's update is the sign of its momentum, so it moves a scale by exactly ``+-lr`` however large that
        scale is. A symmetric group scale is ``absmax / (levels / 2)``, so it shrinks as the width grows: at
        8 bits it is 36x smaller than the 3-bit scale upstream tuned ``scales_lr`` on, and the same absolute
        step becomes a ~20 % move per step that walks the scale through zero within a few steps (measured on
        w8 groups: 4-12 negative scales per tensor after 4 steps, while w3 banks keep cosine 0.9999).

        Dividing by the level count keeps the step the same FRACTION of the scale at every width, and leaves
        upstream's own path (``GumbelQuantizerInt``, 3-bit) at exactly ``scales_lr``.
        """
        return self.scales_lr * UPSTREAM_LEVELS / (fmt.qmax - fmt.qmin)

    @property
    def dense_on_every_rank(self) -> bool:
        """Whether a split layer's dense groups train on every card (`solve.units.bound` skips them otherwise).

        Under expert parallel every card holds its own calibration batches, so a dense group fitted on rank 0
        alone would see a quarter (or an eighth) of the pool. Every card trains them instead and the Lion step
        takes the mean of the ranks' gradients (`Comm.average_grads`), which is what the data-parallel fit
        does over the same rows; only rank 0's codes are kept, and `broadcast_module` gives them to the rest.
        """
        return self.expert_parallel

    @model_validator(mode="after")
    def _validate_schedules(self) -> GSQ:
        if any(value <= 0 for value in (*self.temperature, *self.logit_scale)):
            raise ValueError("gsq temperature and logit_scale endpoints must be positive")
        if any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("gsq betas must lie in [0, 1)")
        return self

    def solver_workspace(self, rows: int, cols: int) -> int:
        """Bytes per expert that this step holds on the card while it trains one of a unit's experts.

        `GumbelRounding` keeps the logits, their gradient, the valid mask, the starting codes and scales for the
        rest of the layer (~31 B per weight) and builds its five code shifts in place, so 80 B per weight also
        covers the largest single allocation, one five-shift float32 array (20 B per weight).

        A layer's units are all held at once (`joint_units`), and `engine.residency._peak` sums the solves a
        layer marks `joint`: this must return ONE unit's state, or a bank's `in` and `down` pair counts twice.
        """
        return 80 * rows * cols

    def apply_layer(self, ctx: LayerContext) -> None:
        """Train this layer from its existing codes and write the final hard served weights."""
        io = self.layer_io(ctx)
        if ctx.comm is None and self.validation >= len(io):
            raise Refused(f"gsq validation={self.validation} leaves no training batches out of {len(io)}")
        # with more than one card the same check runs in `_train`, after the first collective: `dist.shard`
        # can leave one rank a batch short, and a refusal on that rank alone would leave the others waiting
        device = ctx.device or next(ctx.layer.parameters()).device
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad(), attention_kernel(self.reproducible):
            torch.manual_seed(self.seed * 100003 + ctx.graph.index * 16 + 1)
            with self._parallel(ctx, io) as (io, ep):
                self._train(ctx, io, device, ep=ep)

    def _parallel(self, ctx: LayerContext, io: list):  # noqa: ANN201
        """Run the layer expert-parallel: this card's experts on every card's tokens, or a no-op context.

        Every card must call the all-to-all the same number of times, in the same order, so the cards first
        agree on the smallest batch count they hold and each keeps its first that many; the extra batch an
        uneven `engine.dist.shard` gives one card is dropped (upstream drops it too, `nsamples // world`).
        """
        comm, share = ctx.comm, ctx.stats.get((SHARE, ctx.graph.module))
        if comm is None or comm.world < 2 or share is None or not self.expert_parallel:
            return contextlib.nullcontext((io, False))
        import torch.distributed as dist

        from ..engine import ep

        n = torch.tensor([len(io)], device=comm.device)
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
        io = io[: int(n)]
        cuts = set()
        for bank in ctx.graph.banks:
            if not isinstance(bank.experts, str):
                raise Refused(
                    "gsq expert-parallel needs a fused bank; per-expert linears have no owner table"
                )
            module = submodule(ctx.layer, ctx.graph, bank.experts)
            span = share.experts(module.num_experts)
            cuts.add((bank.experts, span.start, span.stop))
        if len(cuts) > 1:
            raise Refused(f"gsq expert-parallel: the layer's banks disagree on this card's experts ({cuts})")
        name, first, stop = cuts.pop()

        @contextlib.contextmanager
        def run():  # noqa: ANN202
            with ep.expert_parallel(
                submodule(ctx.layer, ctx.graph, name), first, stop, comm.world, dist.group.WORLD
            ):
                yield io, True

        return run()

    def _experts_at_a_time(self, device: torch.device, units: list[BoundUnit]) -> int:
        """Experts of a fused bank one `GumbelRounding` holds: `chunk`, else as many as half the free card holds.

        The other units of the layer keep their served weights on the card beside it (`_frozen`), so the budget
        is half the free memory. The count is rounded down to a power of two so a bank splits evenly and the same
        piece sizes recur layer after layer. `engine.residency._peak` prices a solve in slices, so a step that
        did not cut its unit itself would run out of memory under a plan that fits.
        """
        if self.chunk:
            return self.chunk
        if device.type != "cuda":
            return 1 << 40
        from ..solve.solve import free_bytes

        per = sum(BYTES_PER_WEIGHT * u.shape()[0] * u.shape()[1] for u in units if u.part is not None)
        if not per:
            return 1 << 40
        count = max(1, free_bytes(device) // (2 * per))
        while count > 1 and count & (count - 1):
            count &= count - 1
        return count

    def _train(self, ctx: LayerContext, io: list, device: torch.device, *, ep: bool = False) -> None:
        comm = ctx.comm
        if comm is not None:
            # every rank starts from the codes rank 0 wrote (the earlier step runs on rank 0 alone). Expert
            # parallel keeps its own bank entries -- this card's experts, quantized here, merged by the gather
            # after the layer -- but takes rank 0's DENSE codes, which every card trains (`dense_on_every_rank`)
            comm.broadcast_qparams(ctx.qparams, skip_experts=ep)
        name_of = {id(parameter): name for name, parameter in ctx.layer.named_parameters()}
        units = list(self.units(ctx))
        # expert parallel never cuts: one unit is this card's whole block of experts, trained jointly
        count = 1 << 40 if ep else self._experts_at_a_time(device, units)
        if comm is not None and not ep:
            # `count` comes from this card's free memory, so two ranks can cut a bank into a different number
            # of pieces and wait for each other in the gradients' collective: the world takes the fewest, which
            # is also what the tightest card holds
            count = comm.min_across(count)
        initial: list[tuple[BoundUnit, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        groups: list[tuple[int, slice]] = []
        for unit in units:
            codes = _initial(ctx, unit, "codes")
            scales = _initial(ctx, unit, "scale")
            zero = None if unit.fmt.symmetric else _initial(ctx, unit, "zero_point")
            fused = unit.part is not None and isinstance(unit.names, str)
            groups.extend(
                (len(initial), span) for span in (_spans(codes.shape[0], count) if fused else [slice(None)])
            )
            initial.append((unit, codes, scales, zero))
        logging.getLogger(__name__).info(
            "gsq layer=%s experts_at_a_time=%s roundings=%s units=%s",
            ctx.graph.index,
            count,
            len(groups),
            len(units),
        )
        roundings: list[GumbelRounding | None] = [None] * len(groups)
        frozen: list[torch.Tensor | None] = [None] * len(groups)
        done: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(groups)

        def piece(j: int) -> tuple[BoundUnit, slice, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            unit, codes, scales, zero = initial[groups[j][0]]
            return unit, groups[j][1], codes, scales, zero

        def build(j: int) -> GumbelRounding:
            unit, span, codes, scales, zero = piece(j)
            parameter = (
                unit.params[0] if unit.part is None or isinstance(unit.names, str) else unit.params[0][0]
            )
            return GumbelRounding(
                codes[span].to(device).contiguous(),
                scales[span].to(device).contiguous(),
                None if zero is None else zero[span].to(device).contiguous(),
                unit.quantizer.group(codes.shape[-1]),
                unit.fmt.qmin,
                unit.fmt.qmax,
                self.std,
                self.strength,
                parameter.dtype,
                getattr(torch, self.logits_dtype),
            )

        def freeze(j: int) -> None:
            """Keep group `j`'s served weights, which every other group's forward reads while it trains."""
            unit, span, codes, scales, zero = piece(j)
            frozen[j] = unit.quantizer.dequant(
                codes[span].to(device),
                scales[span].to(device),
                None if zero is None else zero[span].to(device),
            )

        def replacements(active: frozenset[int], soft: bool, temperature: float, logit_scale: float) -> dict:
            per_unit: dict[int, list[tuple[slice, torch.Tensor]]] = {}
            for j, (k, span) in enumerate(groups):
                if j in active:
                    rounding = roundings[j]
                    assert rounding is not None
                    zero = initial[k][3]
                    weight = (
                        rounding(temperature, logit_scale)
                        if soft
                        else self._hard(
                            initial[k][0].quantizer, rounding, zero[span] if zero is not None else None
                        )
                    )
                else:
                    weight = frozen[j]
                    assert weight is not None
                per_unit.setdefault(k, []).append((span, weight))
            out: dict = {}
            whole: dict[int, torch.Tensor] = {}
            for k, pieces in per_unit.items():
                unit = initial[k][0]
                if unit.part is None or not isinstance(unit.names, str):
                    out.update(_split(unit, pieces[0][1], name_of))
                else:
                    _whole(unit, pieces, whole)
            for p, tensor in whole.items():
                out[name_of[p]] = tensor
            return out

        def validate(active: frozenset[int], items: list, temperature: float, logit_scale: float):
            if not items:
                return float("nan"), float("nan")
            losses = []
            with torch.no_grad():
                for start in range(0, len(items), self.batches):
                    batch_items = items[start : start + self.batches]
                    soft = replacements(active, True, temperature, logit_scale)
                    soft_loss = sum(_loss(ctx.layer, soft, item) for item in batch_items) / len(batch_items)
                    hard = replacements(active, False, temperature, logit_scale)
                    hard_loss = sum(_loss(ctx.layer, hard, item) for item in batch_items) / len(batch_items)
                    losses.append((float(soft_loss), float(hard_loss)))
            return (
                sum(soft for soft, _ in losses) / len(losses),
                sum(hard for _, hard in losses) / len(losses),
            )

        # `validation` is global too, and for the same reason as `batches` below: it is held out of the WHOLE
        # pool, so a card holds its share of it. Subtracting the whole count from each shard instead would
        # leave `train_count = n/W - V`, which still shrinks faster than the batch and keeps the step count
        # world-dependent (at n=512, V=32, batches=16: 30 steps on one card but 16 on eight).
        world = 1 if comm is None else comm.world
        held = self.validation // world
        train_count = len(io) - held
        if comm is not None:
            # every rank decides on the smallest shard, for the same reason as the step count below: a refusal
            # raised on one rank alone would leave the others waiting in the next collective
            shortest = comm.min_across(train_count)
            if shortest <= 0:
                raise Refused(
                    f"gsq validation={self.validation} holds out {held} of each of {world} shards and leaves "
                    f"no training batches out of the smallest calibration shard's {shortest + held} batches"
                )
        elif train_count <= 0:
            raise Refused(f"gsq validation={self.validation} leaves no training batches out of {len(io)}")
        # `batches` is the GLOBAL batch, as upstream's `data.batch_size` is: a card takes its own share of it
        # (`src/trainer.py:87` `batch_size = config.data.batch_size // world_size`). With the shard already
        # 1/W of the pool, dividing the batch too keeps `per_epoch = ceil(n/B)` and every schedule it drives
        # the SAME on one card and on eight -- the world changes who does the work, not what is done.
        if self.batches // world <= 0:
            raise Refused(
                f"gsq batches={self.batches} is the global batch and splits to 0 over {world} cards; "
                f"raise gsq batches to at least {world}, or use fewer cards"
            )
        batch = min(self.batches // world, train_count)
        per_epoch = train_count // batch
        if comm is not None:
            # a data-parallel fit takes the same number of collective steps on every rank; `engine.dist.shard`
            # gives one rank a batch more, so the world fits what the smallest shard holds (upstream truncates
            # to whole batches the same way)
            batch = comm.min_across(batch)
            per_epoch = comm.min_across(per_epoch)
            if per_epoch <= 0:
                raise Refused(
                    f"gsq: the smallest calibration shard leaves no full batch of {batch} to fit; "
                    "lower gsq batches, add calibration batches, or use fewer cards"
                )
        total = per_epoch * self.epochs
        if self.warmup >= total:
            raise Refused(f"gsq warmup={self.warmup} must be smaller than {total} optimizer steps")
        # Everything fits whole: the step's own path, every unit of the layer in one optimizer over the layer's
        # loss (upstream GSQ). Only a unit cut into pieces trains them one set at a time.
        joint = len(groups) == len(units)
        sets = [frozenset(range(len(groups)))] if joint else [frozenset({j}) for j in range(len(groups))]
        for active in sets:
            for j in active:
                roundings[j] = build(j)
            for j in range(len(groups)):
                if j not in active:
                    freeze(j)
            parameter_groups = []
            for j in sorted(active):
                rounding = roundings[j]
                assert rounding is not None
                parameter_groups.extend(
                    [
                        {
                            "params": [rounding.logits],
                            "lr": self.logits_lr,
                            "base_lr": self.logits_lr,
                            "weight_decay": self.weight_decay,
                        },
                        {
                            "params": [rounding.scales],
                            "lr": self._scales_lr(initial[groups[j][0]][0].fmt),
                            "base_lr": self._scales_lr(initial[groups[j][0]][0].fmt),
                            "grid": initial[groups[j][0]][0].quantizer,
                        },
                    ]
                )
            optimizer = Lion(parameter_groups, self.betas)
            dense = {
                id(p)
                for j in sorted(active)
                if initial[groups[j][0]][0].part is None
                for p in (roundings[j].logits, roundings[j].scales)  # type: ignore[union-attr]
            }
            label = ",".join(str(j) for j in sorted(active))
            validate(active, io[train_count:], self.temperature[0], self.logit_scale[0])
            step = 0
            for epoch in range(self.epochs):
                for batch_index in torch.randperm(per_epoch).tolist():
                    progress = step / (total - 1) if total > 1 else 0
                    temperature = self.temperature[0] + (self.temperature[1] - self.temperature[0]) * progress
                    logit_scale = self.logit_scale[0] + (self.logit_scale[1] - self.logit_scale[0]) * progress
                    optimizer.zero_grad(set_to_none=True)
                    for offset in range(batch):
                        loss = (
                            _loss(
                                ctx.layer,
                                replacements(active, True, temperature, logit_scale),
                                io[batch_index * batch + offset],
                            )
                            / batch
                        )
                        loss.backward()
                    if comm is not None:
                        # upstream's Lion step over the ranks: sum the gradients, divide by the world. An
                        # expert-parallel bank is the card's own (its loss sums its own tokens' terms, and the
                        # gather merges the shares), but the layer's DENSE groups train on every card's rows:
                        # averaged, every card takes the step a one-card fit takes over all of them
                        comm.average_grads(optimizer, only=dense if ep else None)
                    for group in optimizer.param_groups:
                        group["lr"] = cosine_lr(group["base_lr"], step, total, self.warmup, self.minimum_lr)
                    optimizer.step()
                    _project_scales(optimizer)
                    step += 1
                soft_loss, hard_loss = validate(active, io[train_count:], temperature, logit_scale)
                logging.getLogger(__name__).info(
                    "gsq layer=%s rounding=%s epoch=%s val_soft=%s val_hard=%s",
                    ctx.graph.index,
                    label,
                    epoch + 1,
                    soft_loss,
                    hard_loss,
                )
            with torch.no_grad():
                for j in sorted(active):
                    rounding = roundings[j]
                    _, span, _, _, zero = piece(j)
                    quantizer = initial[groups[j][0]][0].quantizer
                    codes_j = rounding.hard_codes()  # type: ignore[union-attr]
                    scales_j = quantizer.stored_scale(
                        rounding.scales.reshape(*codes_j.shape[:-1], -1)  # type: ignore[union-attr]
                    ).float()
                    done[j] = (codes_j.to("cpu"), scales_j.to("cpu"))
                    if not joint:
                        frozen[j] = quantizer.dequant(
                            codes_j, scales_j, None if zero is None else zero[span].to(device)
                        )
                    roundings[j] = None
            del optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with torch.no_grad():
            for k, (unit, _codes, _scales, zero) in enumerate(initial):
                js = [j for j, (kk, _) in enumerate(groups) if kk == k]
                assert all(done[j] is not None for j in js)
                codes_k = torch.cat([done[j][0] for j in js])  # type: ignore[index]
                scales_k = torch.cat([done[j][1] for j in js])  # type: ignore[index]
                unit.write(codes_k.to(device), scales_k.to(device), None if zero is None else zero.to(device))

    @staticmethod
    def _hard(quantizer, rounding: GumbelRounding, zero: torch.Tensor | None) -> torch.Tensor:
        """The served weights of a rounding's hard codes, which a frozen piece keeps while another trains."""
        codes = rounding.hard_codes()
        scales = quantizer.stored_scale(rounding.scales.reshape(*codes.shape[:-1], -1)).float()
        return quantizer.dequant(codes, scales, zero)
