"""What a quantizing step iterates over: stacks of weights that share one Hessian.

A dense `LinearGroup` is one unit, its modules' rows stacked (``E = 1``): q/k/v share one solve. An
`ExpertBank` gives two units, the experts' input projections and their down projections (``E =
experts``), read from the fused 3-D parameters or stacked from per-expert linears.

A step gets them from `QuantStep.units` as `BoundUnit`: each knows its layer, the step's quantizer and the
experts this card solves, so ``u.weight()``, ``u.hessian()`` and ``u.write(codes, scale, zero)`` need nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

from ..arch import ExpertBank, LinearGroup, submodule
from ..arch.adapters import fused_experts
from ..format import Format
from ..intake.fingerprint import tensor_digest
from ..step import LayerContext, QuantStep
from ..target import KernelPlan
from .quant import Quantizer

ALL = slice(None)
#: The tensors of a qparams entry that have one leading row per expert of a bank: what an entry solved in shares
#: or chunks is cut into and joined from. ``global_scale`` is a layout's second scale level (`Grid.global_scale`).
QPARAM_TENSORS = ("codes", "scale", "zero_point", "global_scale")
#: ``ctx.stats`` key name of the engine's `ExpertShare` (keyed by the layer's module name)
SHARE = "expert_share"
#: ``ctx.stats`` key name of a unit's format and kernel plan (keyed by the unit key, or by `down_key` for a bank's
#: down projections stored apart), set for the units a config's ``format_overrides`` name; a unit without it uses
#: the layer's body format and plan
FMT = "unit_format"


@dataclass(frozen=True)
class ExpertShare:
    """This card's part of a layer's solve when the solve is split across cards, or into chunks on one card.

    Card `rank` of `world` solves a contiguous block of every fused expert bank's experts; card 0 also
    solves the dense groups. On one card whose expert statistics do not fit at once, chunk `rank` of
    `world` is solved in turn, and its bank statistics hold that chunk's experts only (`local`). The engine
    sets it only for steps that declare ``splits_experts``.
    """

    rank: int
    world: int
    local: bool = False

    def experts(self, n: int) -> slice:
        """This card's experts of a bank of `n`."""
        return slice(n * self.rank // self.world, n * (self.rank + 1) // self.world)


@dataclass
class Unit:
    """Weights ``[E, R, C]`` that share Hessians ``[E, C, C]``, and where each row block goes back to.

    Attributes:
        key: stats key (group's first module or bank router).
        part: ``None`` for a dense group; ``"in"`` / ``"down"`` for a bank's two inputs.
        names: per stored tensor, its qparams key; dense: module names, fused bank: the 3-D parameter name,
            unfused bank: per-expert module names ``[E][k]``.
        rows: row counts of the stacked pieces (dense: per module; bank: per expert piece).
        kind: ``"dense"`` or ``"moe"``.
        transpose: the fused parameter stores ``[E, C, R]`` (transformers' ``is_transposed``): `weight`
            returns the canonical ``[E, R, C]`` and `write` transposes the served weights back.
    """

    key: str
    part: str | None
    names: tuple | str
    rows: tuple[int, ...]
    kind: str
    params: tuple  # the nn.Parameter(s) written back, aligned with `names`
    transpose: bool = field(default=False, kw_only=True)  # keyword: BoundUnit adds required fields after

    def weight(self, experts: slice = ALL) -> torch.Tensor:
        """The stacked weights ``[len(experts), R, C]`` of `experts`.

        A new tensor for a dense group or an unfused bank; a view of the parameter for a fused one.
        """
        if self.part is None:
            return torch.cat([p.data for p in self.params], 0)[None]
        if isinstance(self.names, str):
            w = self.params[0].data[experts]
            return w.transpose(1, 2) if self.transpose else w
        ps = self.params if experts == ALL else self.params[experts]
        return torch.stack([torch.cat([p.data for p in ps_e], 0) for ps_e in ps])

    def n_experts(self) -> int:
        """Experts of the whole unit: 1 for a dense group."""
        if self.part is None:
            return 1
        return self.params[0].shape[0] if isinstance(self.names, str) else len(self.params)

    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of one expert."""
        if self.part is None:
            return sum(self.rows), self.params[0].shape[1]
        if isinstance(self.names, str):
            _, a, b = self.params[0].shape
            return (b, a) if self.transpose else (a, b)
        return sum(self.rows), self.params[0][0].shape[1]

    @property
    def down_key(self) -> str | None:
        """A fused bank's down projections: the `formats` key that may store them apart (`down_key`)."""
        return self.names if self.part == "down" and isinstance(self.names, str) else None

    def stored(self, w: torch.Tensor) -> torch.Tensor:
        """Canonical ``[E, R, C]`` weights as the module stores them."""
        return w.transpose(1, 2) if self.transpose else w

    def hessian_stat(self) -> str:
        """Which statistic holds this unit's Hessian."""
        return "hessian" if self.part is None else "expert_hessian"

    def put_weights(self, served: torch.Tensor, experts: slice = ALL) -> None:
        """Copy served weights ``[len(experts), R, C]`` into the parameters' rows for `experts`."""
        if self.part is None:
            r0 = 0
            for p, r in zip(self.params, self.rows, strict=True):
                p.data.copy_(served[0, r0 : r0 + r].to(p.dtype))
                r0 += r
            return
        if isinstance(self.names, str):
            p = self.params[0]
            p.data[experts].copy_(self.stored(served).to(p.dtype))
            return
        ps = self.params if experts == ALL else self.params[experts]
        for row, ps_e in zip(served, ps, strict=True):
            r0 = 0
            for p in ps_e:
                p.data.copy_(row[r0 : r0 + p.shape[0]].to(p.dtype))
                r0 += p.shape[0]


@dataclass
class BoundUnit(Unit):
    """A `Unit` as `QuantStep.units` gives it to a step: with the layer, the step's quantizer and this card's experts.

    Attributes:
        ctx: the layer being quantized.
        quantizer: the step's quantizer for this unit (its grid; the dense or the MoE kernel plan).
        experts: the experts this card solves (`ALL` unless the engine split the solve into shares).
        fmt: the format this unit is stored in: the body one, or the one a ``format_overrides`` rule gives it.
    """

    ctx: LayerContext
    quantizer: Quantizer
    experts: slice
    fmt: Format

    def weight(self, experts: slice | None = None) -> torch.Tensor:
        """The stacked weights ``[E, R, C]`` of this card's experts, or of `experts`."""
        return super().weight(self.experts if experts is None else experts)

    def hessian(self, experts: slice | None = None) -> tuple[torch.Tensor, torch.Tensor | int | None]:
        """(raw Hessian ``[E, C, C]``, token count(s)) of this card's experts, or of `experts`."""
        experts = self.experts if experts is None else experts
        h = self.ctx.stats[(self.hessian_stat(), self.key)]
        n = self.ctx.stats.get(("n_tokens", self.key))
        if self.part is not None:
            h = h[self.part] if isinstance(h, dict) else h
            share: ExpertShare | None = self.ctx.stats.get((SHARE, self.ctx.graph.module))
            if share is not None and share.local:  # the statistics start at the share's first expert
                first = share.experts(self.n_experts()).start
                experts = slice(experts.start - first, experts.stop - first)
            if experts != ALL:
                h = h[experts]
                n = n[experts] if isinstance(n, torch.Tensor) and n.dim() > 0 else n
        return h, n

    def all_token_hessian(self, experts: slice | None = None) -> tuple[torch.Tensor, int]:
        """(raw all-token Hessian ``[E, C, C]``, or ``[1, C, C]`` for gate/up, token count) of a bank part."""
        assert self.part is not None  # only expert banks have all-token statistics
        n = int(self.ctx.stats[("all_tokens_seen", self.key)])
        if self.part == "in":  # every expert reads the same tokens
            return self.ctx.stats[("all_in", self.key)][None], n
        experts = self.experts if experts is None else experts
        h = self.ctx.stats[("all_down", self.key)]
        share: ExpertShare | None = self.ctx.stats.get((SHARE, self.ctx.graph.module))
        if share is not None and share.local:  # the statistics start at the share's first expert
            first = share.experts(self.n_experts()).start
            experts = slice(experts.start - first, experts.stop - first)
        return (h if experts == ALL else h[experts]), n

    def cross_hessian(self) -> torch.Tensor:
        """GPTAQ's cross term ``[E, C, C]`` (``cross_hessian`` in `needs`)."""
        c = self.ctx.stats[("cross_hessian", self.key)]
        if self.part is not None and isinstance(c, dict):
            c = c[self.part]
        return c

    def write(self, codes: torch.Tensor, scale, zero) -> None:  # noqa: ANN001
        """Store the weights the kernel serves for `codes` and (scale, zero), and the codes and scales.

        With an expert share of a fused bank, the tensors hold only those experts: they go into those
        rows of the parameter, and the qparams entry records the span (``"experts": (start, stop)``) until
        the engine gathers the whole bank (`engine.dist.Comm.gather_experts`).
        """
        self.put_weights(self.quantizer.dequant(codes, scale, zero), self.experts)
        self.record(codes, scale, zero)

    def record(self, codes: torch.Tensor, scale, zero) -> None:  # noqa: ANN001
        """Store the codes, scales and served-weight digest of this card's experts in ``ctx.qparams``.

        A grid with a second scale level adds ``global_scale``: ``[1, 1]`` for a module, ``[E, 1, 1]`` for a bank.
        """
        ctx, q, experts = self.ctx, self.quantizer, self.experts
        sc = q.stored_scale(scale)
        zp = q.stored_zero(zero) if zero is not None else None
        gs = q.grid.global_scale(sc)  # of the whole unit, so the modules of a fused group store one value
        cq = q.stored_codes(codes)
        if experts != ALL and isinstance(self.names, str):
            d = {"codes": cq, "scale": sc, "format": self.fmt.name, "experts": (experts.start, experts.stop)}
            if zp is not None:
                d["zero_point"] = zp
            if gs is not None:
                d["global_scale"] = gs
            ctx.qparams[self.names] = d
            return

        def put(name: str, p: nn.Parameter, sl: tuple) -> None:
            d = {
                "codes": cq[sl],
                "scale": sc[sl],
                "format": self.fmt.name,
                "weight_check": (id(p), tensor_digest(p)),
            }
            if zp is not None:
                d["zero_point"] = zp[sl]
            if gs is not None:
                d["global_scale"] = gs[sl[:1]]
            ctx.qparams[name] = d

        if self.part is None:
            r0 = 0
            for name, p, r in zip(self.names, self.params, self.rows, strict=True):
                put(name, p, (0, slice(r0, r0 + r)))
                r0 += r
        elif isinstance(self.names, str):
            put(self.names, self.params[0], (slice(None),))
        else:  # one entry per expert Linear, for `experts` only when the bank is solved in chunks
            names_ps = zip(self.names[experts], self.params[experts], strict=True)
            for e, (names, ps) in enumerate(names_ps):
                r0 = 0
                for name, p in zip(names, ps, strict=True):
                    put(name, p, (e, slice(r0, r0 + p.shape[0])))
                    r0 += p.shape[0]


def unit_format(ctx: LayerContext, key: str, kind: str, down: str | None = None) -> tuple[Format, KernelPlan]:
    """The (format, kernel plan) unit `key` of `kind` is stored in; `down`: a bank's down projections (`down_key`).

    The engine puts a config's ``format_overrides`` decisions in ``ctx.stats``; a unit no rule names keeps the
    layer's body format and plan, and a bank's down projections keep the bank's unless a rule stores them apart.
    Steps that read a group's or bank's format directly (AWQ's search) ask here.
    """
    body = ctx.plan_dense if kind == "dense" or ctx.plan_moe is None else ctx.plan_moe
    got = ctx.stats.get((FMT, key), (ctx.fmt, body))
    return ctx.stats.get((FMT, down), got) if down is not None else got


def stored_format(formats: dict[str, Format] | None, u: Unit, fmt: Format) -> Format:
    """`u`'s format in a plan's `formats`: its own, its bank's, or the body `fmt`."""
    formats = formats or {}
    return formats.get(u.down_key or "", formats.get(u.key, fmt))


def down_key(bank: ExpertBank) -> str | None:
    """The `formats` key of a fused bank's down projections, which a GGUF may store apart from its gate/up.

    llama.cpp reads a bank's gate, up and down projections as three tensors, each with its own type; vLLM and
    SGLang load the bank as one fused module with one scheme (`Exporter.splits_banks`). The key is the down
    parameter, as the build's qparams name it. A bank stored one module per expert has none.
    """
    return f"{bank.experts}.{fused_experts.DOWN}" if isinstance(bank.experts, str) else None


def bank_module(bank: ExpertBank) -> str:
    """The module an expert bank is served as, however the checkpoint names its experts."""
    if isinstance(bank.experts, str):
        return bank.experts
    return bank.experts[0][0][0].rsplit(".", 2)[0]


def bound(step: QuantStep, ctx: LayerContext) -> Iterator[BoundUnit]:
    """The units of `ctx`'s layer that this card solves, bound to `step`'s quantizer (`QuantStep.units`)."""
    share: ExpertShare | None = ctx.stats.get((SHARE, ctx.graph.module))
    for u in units_of(ctx.layer, ctx.graph):
        experts = ALL
        if share is not None:
            if u.part is None:
                # a split layer's dense groups are rank 0's, unless the step trains them on every card
                # (`GSQ.dense_on_every_rank` under expert parallel: each card's own rows, gradients averaged)
                if share.rank != 0 and not getattr(step, "dense_on_every_rank", False):
                    continue
            elif isinstance(u.names, str) or share.local:
                experts = share.experts(u.n_experts())
            else:
                raise ValueError(f"{u.key}: per-expert linears are solved in shares on one card only")
        fmt, plan = unit_format(ctx, u.key, u.kind, u.down_key)
        q = step.quantizer(fmt, plan)
        yield BoundUnit(
            u.key, u.part, u.names, u.rows, u.kind, u.params, ctx, q, experts, fmt,
            transpose=u.transpose,
        )  # fmt: skip


def units_of(layer: nn.Module, g) -> Iterator[Unit]:  # noqa: ANN001
    """Every quantizable unit of `layer`, described by graph `g`, groups first, then banks."""
    for grp in g.groups:
        yield _dense(layer, g, grp)
    for bank in g.banks:
        yield from _bank(layer, g, bank)


def _dense(layer: nn.Module, g, grp: LinearGroup) -> Unit:  # noqa: ANN001
    weights = [cast(torch.Tensor, submodule(layer, g, n).weight) for n in grp.modules]
    return Unit(grp.modules[0], None, grp.modules, tuple(w.shape[0] for w in weights), "dense",
                tuple(weights))  # fmt: skip


def _bank(layer: nn.Module, g, bank: ExpertBank) -> Iterator[Unit]:  # noqa: ANN001
    if isinstance(bank.experts, str):
        m = submodule(layer, g, bank.experts)
        flip = fused_experts.transposed(m)
        for part, attr, w in (
            ("in", fused_experts.in_param(m), fused_experts.in_weight(m)),
            ("down", fused_experts.DOWN, fused_experts.down_weight(m)),
        ):
            p = getattr(m, attr)
            yield Unit(
                bank.router, part, f"{bank.experts}.{attr}", (w.shape[1],), "moe", (p,), transpose=flip
            )
        return
    for part, pick in (("in", lambda ex: ex[0]), ("down", lambda ex: (ex[1],))):
        names = tuple(tuple(pick(ex)) for ex in bank.experts)
        params = tuple(tuple(cast(torch.Tensor, submodule(layer, g, n).weight) for n in ns) for ns in names)
        yield Unit(bank.router, part, names, tuple(p.shape[0] for p in params[0]), "moe", params)
