"""Run trainable quantized modules using stored serving numerics."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from torch import nn

from ..format import Format
from ..solve.quant import ACT_CUTS, ActParts, Quantizer, widen
from .packed import stored_scale

#: elements of one recomputed row block in backward (bounds the fp32 transient of a large head)
ROW_BLOCK_ELEMS = 1 << 27

#: NVFP4 stores its group scales as E4M3
FP8_MAX = 448.0


def decodes_once(fmt: Format) -> bool:
    """True when `scale_kd` decodes a module's served weight once per step (`ServedLinear.materialize`).

    NVFP4 and FP8 weights: their served forward reads the decode. Every other format decodes per call and
    recomputes the weight in row blocks in backward, so the card never holds a decoded copy of the model.
    """
    return fmt.weight_type == "fp8" or fmt.act == "nvfp4_block"


@contextlib.contextmanager
def _tf32() -> Iterator[None]:
    """fp32 matmuls on TF32 tensor cores (fp32 sums): exact products when both inputs hold E4M3 or int8 values."""
    b = torch.backends.cuda.matmul
    old = b.fp32_precision
    b.fp32_precision = "tf32"
    try:
        yield
    finally:
        b.fp32_precision = old


def ste_bf16(s: torch.Tensor) -> torch.Tensor:
    """``s`` rounded to bf16 in the forward, identity in the backward (fp32 result)."""
    return s + (s.to(torch.bfloat16).float() - s).detach()


def ste_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """``scale`` rounded to E4M3 in the forward, identity in the backward (fp32 result).

    NVFP4 stores its group scales as E4M3: a training state's ``s0 * exp(delta)`` is only served through the code
    the checkpoint can hold. The cast is straight-through, so `delta` trains through that code.
    """
    r = torch.clamp(scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return scale + (r - scale).detach()


def _gscale_rows(g: torch.Tensor, s0: torch.Tensor, rows: slice) -> torch.Tensor:
    """The weight global scale of rows `rows`: a module's ``[1]`` covers all of them, a flat bank's one per row."""
    return g[rows] if g.shape[0] == s0.shape[0] > 1 else g


@dataclass(frozen=True)
class Stored:
    """The frozen stored tensors of one quantized weight ``[out, in]``: a Linear, or one expert of a bank.

    Attributes:
        codes: codes ``[out, in]`` as stored: int8 (signed; NVFP4: ladder positions -7 .. 7), or the E4M3
            values themselves for FP8 weights.
        zero: zero points ``[out, groups]`` (int8, same offset as `codes`) or None.
        s0: stored scale ``[out, groups]`` (bf16; for NVFP4 the stored E4M3 ``weight_scale``; for FP8 the fp32
            per-channel scale the checkpoint holds).
        q: the format and kernel plan.
        bias: frozen bias or None.
        gscale: NVFP4 ``weight_global_scale`` ``[1]`` (fp32) or None; a bank's flattened rows carry one per row
            (``[rows, 1]``), since its experts have their own.
        igscale: NVFP4 static ``input_global_scale`` ``[1]`` (fp32) or None.
        served: the weight `Stored.weight` / `weight_bf16` gives for the state a step works on, decoded once
            (`ServedLinear.materialize`) and used by every forward and backward of that step; None decodes it
            per call. A `split` module's holds the E4M3 part only (`weight_parts`).
        chan: with `served`, a `split` module's per-row factor ``[out, 1]`` (fp32).
    """

    codes: torch.Tensor
    zero: torch.Tensor | None
    s0: torch.Tensor
    q: Quantizer
    bias: torch.Tensor | None = None
    gscale: torch.Tensor | None = None
    igscale: torch.Tensor | None = None
    served: torch.Tensor | None = None
    chan: torch.Tensor | None = None

    @property
    def nvfp4(self) -> bool:
        """True when the kernel decodes E2M1 codes under an E4M3 scale and a per-module global scale."""
        return self.q.plan.scale_dtype == "nvfp4"

    @property
    def act_parts(self) -> ActParts | None:
        """How the kernel cuts its input (`solve.quant.ACT_CUTS`), or None when it reads the input as it is."""
        cut = ACT_CUTS[self.q.plan.act_dtype]
        if cut is None:
            return None
        if self.q.plan.act_dtype == "nvfp4":
            assert self.igscale is not None, "an NVFP4 input cut needs input_global_scale"
            # the NVFP4 cut is block_cut_parts(x, gscale=...); ActParts names its one-argument form
            parts = cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], cut.parts)
            return partial(parts, gscale=self.igscale)
        return cut.parts

    def nvfp4_step(self, delta: torch.Tensor, rows: slice = slice(None)) -> torch.Tensor:
        """The NVFP4 fp32 step per column: ``e4m3(s0 * exp(delta)) / weight_global_scale`` of `rows`."""
        g = _gscale_rows(cast(torch.Tensor, self.gscale), self.s0, rows)
        return ste_e4m3(self.s0[rows].float() * delta.exp()) / g.float()

    def weight(self, delta: torch.Tensor, rows: slice = slice(None)) -> torch.Tensor:
        """Served weight rows ``[rows, in]`` in fp32 (bf16-exact for WNA16), differentiable in `delta`."""
        q = self.codes[rows]
        if self.q.fp8_weights:
            # the kernel multiplies the stored E4M3 value by the fp32 scale of its output channel; nothing is
            # rounded again, so the forward and the checkpoint hold the same scale
            return q.float() * (self.s0[rows].float() * delta.exp())
        if self.nvfp4:
            return self.q.dequant(q, self.nvfp4_step(delta, rows), None)
        s = ste_bf16(self.s0[rows].float() * delta.exp())
        if self.q.fp8_scales:
            return self.q.dequant(q, s, None)
        g = self.q.group(q.shape[-1])
        x = q.float() if self.zero is None else q.float() - widen(self.zero[rows].float(), g)
        return x * widen(s.to(torch.bfloat16).float(), g)

    @property
    def split(self) -> bool:
        """True when the matmul runs on the E4M3 part of an FP8-scale weight, its per-row factor after (W4A8).

        Both inputs are then exact in TF32, so the TF32 product is the fp32 one with the kernel's fp32 sums, as
        CUTLASS's FP8 mainloop multiplies ``e4m3(q * f8)`` and applies ``chan`` after the MMA.
        """
        return self.q.fp8_scales and self.act_parts is not None

    @torch.no_grad()
    def weight_parts(
        self, delta: torch.Tensor, rows: slice = slice(None)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A `split` module's ``weight(delta, rows)`` as (E4M3 part ``[rows, in]``, ``chan`` ``[rows, 1]``)."""
        s = (self.s0[rows].float() * delta.exp()).to(torch.bfloat16).float()  # `weight`'s ste_bf16, forward
        return self.q.dequant_parts(self.codes[rows], s)

    def decode(self, delta: torch.Tensor) -> torch.Tensor:
        """The served weight for `delta`, in the dtype the forward's matmul reads (see `matmul`)."""
        return self.weight_bf16(delta) if self.act_parts is None else self.weight(delta)

    @torch.no_grad()
    def weight_bf16(self, delta: torch.Tensor, rows: slice = slice(None)) -> torch.Tensor:
        """``weight(delta, rows)`` rounded to bf16, the forward's weight, without fp32 transients or gradient.

        Same bits as `weight` (the reference): ``(q - z) * s`` is exact in fp32 and rounded once to bf16 here too,
        with the group scale broadcast instead of repeated per column.
        """
        if self.q.fp8_scales or self.q.fp8_weights or self.nvfp4:
            return self.weight(delta, rows).to(torch.bfloat16)
        q = self.codes[rows]
        g = self.q.group(q.shape[-1])
        s = _scale(self.s0[rows], delta).float().unsqueeze(-1)
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        qg, og = q.view(*q.shape[:-1], -1, g), out.view(*q.shape[:-1], -1, g)
        if self.zero is None:
            torch.mul(qg, s, out=og)
        else:  # fp32 q*s + (-z*s): both products exact, so one rounding of the exact (q - z) * s
            torch.addcmul(-self.zero[rows].float().unsqueeze(-1) * s, qg, s, out=og)
        return out

    @torch.no_grad()
    def delta_grad(self, delta: torch.Tensor, gw: torch.Tensor, rows: slice = slice(None)) -> torch.Tensor:
        """d(delta) of rows `rows` given ``gw = dL/dW`` ``[rows, in]`` (fp32): autograd of `weight`, in closed form.

        Same bits as ``autograd.grad(weight(delta, rows), delta, gw)`` on the CPU: the group sum of ``gw * (q - z)``
        passes the bf16 cast of the scale (rounded to bf16, as the cast's backward does), then ``* s0 * exp(delta)``.
        NVFP4's scale cast is straight-through, so its factor is ``s0 * exp(delta) / weight_global_scale``.
        An FP8 weight's code is its E4M3 value and its scale fp32, so its factor is ``s0 * exp(delta)``.
        """
        if self.q.fp8_weights:
            # weight row r is code * s0[r] * exp(delta[r]): the sum of gw over the row's columns, times that scale
            q = self.codes[rows]
            return (gw * q.float()).sum(-1, keepdim=True) * self.s0[rows].float() * delta.exp()
        if self.q.fp8_scales:
            with torch.enable_grad():
                d = delta.detach().requires_grad_(True)
                return torch.autograd.grad(self.weight(d, rows), d, gw)[0]
        if self.nvfp4:
            return self._nvfp4_delta_grad(delta, gw, rows)
        q = self.codes[rows]
        g = self.q.group(q.shape[-1])
        x = q.to(torch.bfloat16).view(*q.shape[:-1], -1, g)
        if self.zero is not None:
            x = x - self.zero[rows].to(torch.bfloat16).unsqueeze(-1)
        gs = (gw.view_as(x) * x).sum(-1)
        return gs.to(torch.bfloat16).float() * self.s0[rows].float() * delta.exp()

    @torch.no_grad()
    def _nvfp4_delta_grad(self, delta: torch.Tensor, gw: torch.Tensor, rows: slice) -> torch.Tensor:
        """`delta_grad` for NVFP4: ``sum_i gw * value`` per group, times ``s0 * exp(delta) / weight_global_scale``.

        ``value`` is the code's E2M1 value (the multiplier `weight` applies at step 1); `ste_e4m3` passes the
        gradient straight through, so the scale's factor is the state's own ``s0 * exp(delta)`` over the module's
        global scale.
        """
        from ..solve import nvfp4

        q = self.codes[rows]
        group = self.q.group(q.shape[-1])
        x = nvfp4.e2m1_values(nvfp4.codes_of_positions(q)).view(*q.shape[:-1], -1, group)
        gs = (gw.view_as(x) * x).sum(-1)
        g = _gscale_rows(cast(torch.Tensor, self.gscale), self.s0, rows)
        return gs * (self.s0[rows].float() * delta.exp() / g.float())

    def matmul(self, x: torch.Tensor, w: torch.Tensor, chan: torch.Tensor | None = None) -> torch.Tensor:
        """The served matmul of input `x` with served weight `w` (fp32), or its E4M3 part and `chan` (`split`)."""
        parts = self.act_parts
        if parts is None:
            b = None if self.bias is None else self.bias.to(torch.bfloat16)
            return nn.functional.linear(x, w.to(torch.bfloat16), b).to(x.dtype)
        codes, s = parts(x)
        if chan is None:
            y = nn.functional.linear(codes, w.float()) * s
        else:
            with _tf32():
                y = nn.functional.linear(codes, w)
            y = y * chan.t() * s
        return (y if self.bias is None else y + self.bias.float()).to(x.dtype)

    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """``matmul(x, weight(delta))``, differentiable in `x` and `delta`."""
        return _ServedFn.apply(x, delta, self)


class ServedLinear(nn.Module):
    """A quantized Linear ``[out, in]`` of a packed checkpoint.

    Args:
        stem: the module's name in the checkpoint (``<stem>.weight_packed``, or ``<stem>.weight`` in FP8).
        codes: codes ``[out, in]`` as stored (int8 signed, or FP8's E4M3 values).
        zero: zero points ``[out, groups]`` (int8, same offset as `codes`) or None.
        s0: stored scale ``[out, groups]`` (bf16, or fp32 for FP8 weights).
        delta: initial log-scale offset ``[out, groups]`` (fp32).
        quantizer: the module's format and kernel plan.
        bias: frozen bias or None.
        gscale: NVFP4 ``weight_global_scale`` ``[1]`` (fp32) or None.
        igscale: NVFP4 static ``input_global_scale`` ``[1]`` (fp32) or None.
    """

    codes: torch.Tensor
    zero: torch.Tensor | None
    s0: torch.Tensor
    bias: torch.Tensor | None
    gscale: torch.Tensor | None
    igscale: torch.Tensor | None

    def __init__(
        self,
        stem: str,
        codes: torch.Tensor,
        zero: torch.Tensor | None,
        s0: torch.Tensor,
        delta: torch.Tensor,
        quantizer: Quantizer,
        bias: torch.Tensor | None = None,
        *,
        gscale: torch.Tensor | None = None,
        igscale: torch.Tensor | None = None,
    ):
        super().__init__()
        if quantizer.plan.act_dtype == "nvfp4" and (gscale is None or igscale is None):
            raise ValueError(f"{stem}: an NVFP4 module needs its weight and input global scales")
        self.stem, self.q = stem, quantizer
        self.out_features, self.in_features = codes.shape
        self._w: torch.Tensor | None = None
        self._chan: torch.Tensor | None = None
        # an FP8 weight keeps the E4M3 values and the fp32 scale the checkpoint stores; an integer or NVFP4 one
        # keeps int8 codes and a bf16 scale
        self.register_buffer(
            "codes", codes if quantizer.fp8_weights else codes.to(torch.int8), persistent=False
        )
        self.register_buffer("zero", None if zero is None else zero.to(torch.int8), persistent=False)
        self.register_buffer(
            "s0", s0.float() if quantizer.fp8_weights else s0.to(torch.bfloat16), persistent=False
        )
        self.delta = nn.Parameter(delta.float().to(s0.device))
        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("gscale", None if gscale is None else gscale.float(), persistent=False)
        self.register_buffer("igscale", None if igscale is None else igscale.float(), persistent=False)

    @property
    def stored(self) -> Stored:
        """The frozen tensors (on the module's current device), with the materialized weight if there is one."""
        return Stored(
            self.codes, self.zero, self.s0, self.q, self.bias, self.gscale, self.igscale, self._w, self._chan
        )

    @property
    def decodes_once(self) -> bool:
        """True when `scale_kd` materializes this module (`decodes_once` of its format)."""
        return decodes_once(self.q.fmt)

    def materialize(self) -> None:
        """Decode the weight `delta` serves now, once, for every forward and backward until the next call.

        A training step re-decodes its weights when the optimizer changes `delta`, and a stage with frozen scales
        (no ``scales`` in ``trainable``) decodes once for the whole stage. Outside a step every call decodes.
        """
        with torch.no_grad():
            st = Stored(self.codes, self.zero, self.s0, self.q, self.bias, self.gscale, self.igscale)
            self._w, self._chan = st.weight_parts(self.delta) if st.split else (st.decode(self.delta), None)

    def weight(self, delta: torch.Tensor | None = None, rows: slice = slice(None)) -> torch.Tensor:
        """Served weight rows ``[rows, in]`` in fp32, differentiable in `delta` (default: the module's)."""
        return self.stored.weight(self.delta if delta is None else delta, rows)

    def scale(self) -> torch.Tensor:
        """The scale the forward serves and the checkpoint stores.

        FP8: the fp32 per-channel scale itself; NVFP4: its E4M3 ``weight_scale``, in bf16; else the bf16 scale.
        """
        if self.q.fp8_weights:
            return self.s0.float() * self.delta.detach().exp()
        if self.q.plan.scale_dtype != "nvfp4":
            return _scale(self.s0, self.delta)
        x = self.s0.float() * self.delta.detach().exp()
        return torch.clamp(x, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float().to(torch.bfloat16)

    def state(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{stem: (s0, delta)}`` on the CPU: what the checkpoint writer and the next stage read."""
        return {self.stem: (self.s0.cpu(), self.delta.detach().float().cpu())}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``matmul(x, weight())``, differentiable in `x` and `delta`."""
        return self.stored(x, self.delta)

    def extra_repr(self) -> str:
        """Name, shape and format."""
        return f"{self.stem}: {self.in_features} -> {self.out_features}, {self.q.fmt.name}"


class ServedBank(nn.Module):
    """One fused 3-D parameter of a routed-experts module, ``[E, out, in]``, from per-expert stored modules.

    Rows of expert ``e`` are its stored parts in order (``gate_proj`` then ``up_proj`` for ``gate_up_proj``),
    each a quantized module with its own scales. One `delta` ``[E, out, groups]`` covers the bank.

    Args:
        parts: per expert, ``((stem, rows), ...)`` of its stored modules, in row order.
        codes: ``[E, out, in]`` int8.
        zero: ``[E, out, groups]`` int8 or None.
        s0: ``[E, out, groups]`` bf16.
        delta: ``[E, out, groups]`` fp32.
        quantizer: the format and (MoE) kernel plan.
        gscale: NVFP4 ``weight_global_scale`` per expert ``[E, 1, 1]`` (fp32) or None.
        igscale: NVFP4 ``input_global_scale`` of the bank's input ``[1]`` (fp32) or None; every expert of one
            bank projection cuts its input under the same scale.
    """

    codes: torch.Tensor
    zero: torch.Tensor | None
    s0: torch.Tensor
    gscale: torch.Tensor | None
    igscale: torch.Tensor | None

    def __init__(
        self,
        parts: list[tuple[tuple[str, int], ...]],
        codes: torch.Tensor,
        zero: torch.Tensor | None,
        s0: torch.Tensor,
        delta: torch.Tensor,
        quantizer: Quantizer,
        *,
        gscale: torch.Tensor | None = None,
        igscale: torch.Tensor | None = None,
    ):
        super().__init__()
        if quantizer.plan.act_dtype == "nvfp4" and (gscale is None or igscale is None):
            raise ValueError("an NVFP4 bank needs its weight and input global scales")
        if quantizer.fp8_weights:
            raise ValueError("an FP8 expert bank is not served: no kernel row plans FP8 experts")
        self.parts, self.q = parts, quantizer
        self.num_experts, self.out_features, self.in_features = codes.shape
        if gscale is not None:
            gscale = (
                gscale.float().view(self.num_experts, 1, 1).expand(self.num_experts, self.out_features, 1)
            )
        self._w: torch.Tensor | None = None
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("zero", None if zero is None else zero.to(torch.int8), persistent=False)
        self.register_buffer("s0", s0.to(torch.bfloat16), persistent=False)
        self.register_buffer("gscale", None if gscale is None else gscale.contiguous(), persistent=False)
        self.register_buffer("igscale", None if igscale is None else igscale.float(), persistent=False)
        self.delta = nn.Parameter(delta.float().to(s0.device))

    def flat(self) -> Stored:
        """The bank as one ``[E * out, in]`` module (experts are independent rows)."""
        n = self.num_experts * self.out_features
        z = None if self.zero is None else self.zero.reshape(n, -1)
        g = None if self.gscale is None else self.gscale.reshape(n, -1)
        w = None if self._w is None else self._w.reshape(n, self.in_features)
        return Stored(self.codes.reshape(n, -1), z, self.s0.reshape(n, -1), self.q, gscale=g,
                      igscale=self.igscale, served=w)  # fmt: skip

    @property
    def decodes_once(self) -> bool:
        """True when `scale_kd` materializes this bank (`decodes_once` of its format)."""
        return decodes_once(self.q.fmt)

    def materialize(self) -> None:
        """Decode the bank's served weights ``[E, out, in]`` once (`Stored.decode`), as `ServedLinear` does.

        A cut input (NVFP4) decodes in fp32; the bank keeps it in bf16, the dtype of the matmul the grouped path
        runs (`_GroupedServedFn`: torch's grouped_mm takes bf16), which halves a large bank's footprint.
        """
        n = self.num_experts * self.out_features
        z = None if self.zero is None else self.zero.reshape(n, -1)
        g = None if self.gscale is None else self.gscale.reshape(n, -1)
        with torch.no_grad():
            st = Stored(self.codes.reshape(n, -1), z, self.s0.reshape(n, -1), self.q, gscale=g,
                        igscale=self.igscale)  # fmt: skip
            w = st.decode(self.delta.reshape(-1, self.delta.shape[-1]))
            self._w = w.to(torch.bfloat16).view(self.num_experts, self.out_features, self.in_features)

    def weights_bf16(self, delta: torch.Tensor) -> torch.Tensor:
        """``weights(delta)`` in bf16 ``[E, out, in]``, without gradient (`Stored.weight_bf16`).

        The step's materialized bank (`materialize`) when there is one, else decoded now.
        """
        if self._w is not None:
            return self._w
        w = self.flat().weight_bf16(delta.reshape(-1, delta.shape[-1]))
        return w.view(self.num_experts, self.out_features, self.in_features)

    def delta_grad(self, delta: torch.Tensor, gw: torch.Tensor, experts: slice) -> torch.Tensor:
        """d(delta) ``[E', out, G]`` of `experts` given their ``dL/dW`` ``[E', out, in]`` (`Stored.delta_grad`)."""
        e0, e1, _ = experts.indices(self.num_experts)
        rows = slice(e0 * self.out_features, e1 * self.out_features)
        d = delta[experts].reshape(-1, delta.shape[-1])
        return self.flat().delta_grad(d, gw.reshape(-1, self.in_features), rows).view_as(delta[experts])

    def grouped(self, x: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """Rows `x` ``[S, in]`` sorted by expert (`offs`: cumulative row ends per expert) times their expert.

        With a cut input (NVFP4) the rows are cut per 16 values under the bank's `igscale` before the grouped
        matmul; the cut's scale of ones multiplies the result, and dx passes the cut straight through.
        """
        return _GroupedServedFn.apply(x, offs, self.delta, self)

    def expert(self, e: int) -> Stored:
        """Expert `e`'s frozen tensors (views into the bank), with its materialized weight if there is one."""
        z = None if self.zero is None else self.zero[e]
        g = None if self.gscale is None else self.gscale[e]
        return Stored(self.codes[e], z, self.s0[e], self.q, gscale=g, igscale=self.igscale,
                      served=None if self._w is None else self._w[e])  # fmt: skip

    def linear(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """``x @ W_e^T`` through the served operator, differentiable in `x` and the bank's `delta`."""
        return self.expert(e)(x, self.delta[e])

    def scale(self) -> torch.Tensor:
        """The bf16 scales ``[E, out, groups]`` the forward serves (NVFP4: its E4M3 ``weight_scale``)."""
        if self.q.plan.scale_dtype != "nvfp4":
            return _scale(self.s0, self.delta)
        x = self.s0.float() * self.delta.detach().exp()
        return torch.clamp(x, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float().to(torch.bfloat16)

    def state(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{stem: (s0, delta)}`` of every stored module of the bank, on the CPU."""
        s0, d = self.s0.cpu(), self.delta.detach().float().cpu()
        out = {}
        for e, parts in enumerate(self.parts):
            r0 = 0
            for stem, n in parts:
                out[stem] = (s0[e, r0 : r0 + n].clone(), d[e, r0 : r0 + n].clone())
                r0 += n
        return out

    def extra_repr(self) -> str:
        """Shape and format."""
        return f"E={self.num_experts}: {self.in_features} -> {self.out_features}, {self.q.fmt.name}"


class ServedExperts(nn.Module):
    """A fused routed-experts module (transformers 5 Qwen3-MoE / Qwen3.5-MoE layout) on served banks.

    Same call as the module it replaces: ``(hidden_states [T, D], top_k_index [T, k], top_k_weights [T, k])``
    -> ``[T, D]``. It runs per expert (the eager reference) or, with `grouped`, on torch's grouped matmul.

    Args:
        gate_up: bank of ``gate_up_proj`` (gate rows, then up rows).
        down: bank of ``down_proj``.
        act_fn: the activation of the gate.
        grouped: run the grouped fast path instead of the per-expert loop.
    """

    def __init__(self, gate_up: ServedBank, down: ServedBank, act_fn: nn.Module, grouped: bool = False):
        super().__init__()
        self.gate_up, self.down, self.act_fn = gate_up, down, act_fn
        self.num_experts = gate_up.num_experts
        self.grouped = grouped

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Sum over each token's chosen experts of ``weight * down(act(gate) * up)``."""
        if self.grouped:
            return self._grouped(hidden_states, top_k_index, top_k_weights)
        return self._loop(hidden_states, top_k_index, top_k_weights)

    def _loop(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """The eager experts forward of transformers, each expert's matmuls served."""
        out = torch.zeros_like(x)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            gate, up = self.gate_up.linear(x[tok], e).chunk(2, dim=-1)
            h = self.down.linear(self.act_fn(gate) * up, e) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out

    def _grouped(
        self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """The ``grouped_mm_experts_forward`` of transformers (no expert parallelism, no biases), banks served."""
        k = top_k_index.size(-1)
        ids, perm = torch.sort(top_k_index.reshape(-1))
        xs, w = x[perm // k], top_k_weights.reshape(-1)[perm]
        hist = ids.float() if x.device.type in ("cpu", "mps") else ids.int()
        offs = torch.cumsum(
            torch.histc(hist, bins=self.num_experts, min=0, max=self.num_experts - 1),
            dim=0,
            dtype=torch.int32,
        )
        gate, up = self.gate_up.grouped(xs, offs).chunk(2, dim=-1)
        y = self.down.grouped(self.act_fn(gate) * up, offs) * w.unsqueeze(-1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=x.device)
        return y[inv].view(x.size(0), k, -1).sum(dim=1).to(x.dtype)


def _scale(s0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The stored scale of a training state (`packed.stored_scale`, the one owner of the rule)."""
    return stored_scale(s0, delta.detach())


class _ServedFn(torch.autograd.Function):
    """``y = matmul(x, W(delta))``; saves only `x`, and the weight decoded once when the step materialized it.

    Without `Stored.served`, backward recomputes `W` in row blocks (a stage that never materializes, e.g. a test
    or a module used outside `scale_kd`); with it the weight is already the step's, so only the products run.
    """

    @staticmethod
    def forward(ctx, x, delta, op):  # noqa: ANN001, ANN205
        with torch.no_grad():
            if op.served is not None:
                y = op.matmul(x, op.served, op.chan)
            elif op.split:
                y = op.matmul(x, *op.weight_parts(delta))
            else:
                y = op.matmul(x, op.decode(delta))
        ctx.save_for_backward(x, delta)
        ctx.op = op
        return y

    @staticmethod
    def backward(ctx, gy):  # noqa: ANN001, ANN205
        x, delta = ctx.saved_tensors
        m: Stored = ctx.op
        n_out, n_in = m.codes.shape
        x2, gy2 = x.reshape(-1, n_in), gy.reshape(-1, n_out)
        parts = m.act_parts
        if parts is None:
            xs = x2
        else:  # what the kernel multiplies; dx passes the cut straight through
            codes, s = parts(x2)
            xs = codes * s
        need_x, need_d = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gx = torch.zeros(x2.shape, dtype=torch.float32, device=x.device) if need_x else None
        gd = torch.zeros_like(delta) if need_d else None
        rb = max(1, min(n_out, ROW_BLOCK_ELEMS // max(1, n_in)))
        for r0 in range(0, n_out, rb):
            sl = slice(r0, min(n_out, r0 + rb))
            if need_d:
                assert gd is not None  # allocated when need_d is true
                gd[sl] = m.delta_grad(delta[sl], gy2[:, sl].t().float() @ xs.float(), sl)
            if need_x:
                assert gx is not None  # allocated when need_x is true
                if m.chan is not None or (m.split and m.served is None):
                    # (gy * chan) @ part: TF32 rounds gy * chan (W4A16's dx rounds to bf16); a bank's served
                    # weight is whole (bf16) and takes the path below
                    if m.chan is not None:
                        pb, cb = cast(torch.Tensor, m.served)[sl], m.chan[sl]
                    else:
                        pb, cb = m.weight_parts(delta[sl], sl)
                    with _tf32():
                        gx += (gy2[:, sl].float() * cb.t()) @ pb
                    del pb
                    continue
                if m.served is not None:  # the forward's weight, decoded once this step
                    wb = m.served[sl]
                else:
                    wb = m.weight_bf16(delta[sl], sl) if parts is None else m.weight(delta[sl], sl)
                gx += (gy2[:, sl].to(wb.dtype) @ wb).float()
                del wb
        if need_x:
            assert gx is not None  # allocated when need_x is true
            return gx.to(x.dtype).view_as(x), gd, None
        return None, gd, None


def _gmm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Rows of `a` ``[S, K]`` sorted by expert times ``b[e]`` ``[K, N]`` -> ``[S, N]``.

    torch's grouped_mm on a card, per expert elsewhere (rows past ``offs[-1]`` are 0 on the slow path).
    """
    if a.is_cuda:
        return nn.functional.grouped_mm(a, b, offs=offs)
    out = a.new_zeros((a.shape[0], b.shape[-1]))
    r0 = 0
    for e, r1 in enumerate(offs.tolist()):
        if r1 > r0:
            out[r0:r1] = a[r0:r1] @ b[e]
        r0 = r1
    return out


def _gmm_weight_grad(gy: torch.Tensor, x: torch.Tensor, offs: torch.Tensor, n_exp: int) -> torch.Tensor:
    """``gW[e] = gy_e^T x_e`` ``[E, out, in]`` in fp32 (the per-expert backward's ``gy.t().float() @ x.float()``).

    torch's grouped_mm takes bf16 on a card (fp32 is refused), so its products are bf16 and the sum is fp32.
    """
    if gy.is_cuda:
        g, xb = gy.to(torch.bfloat16), x.to(torch.bfloat16)
        return nn.functional.grouped_mm(g.t(), xb, offs=offs).float()
    out = torch.zeros((n_exp, gy.shape[1], x.shape[1]), dtype=torch.float32, device=gy.device)
    r0 = 0
    for e, r1 in enumerate(offs.tolist()):
        if r1 > r0:
            out[e] = gy[r0:r1].float().t() @ x[r0:r1].float()
        r0 = r1
    return out


class _GroupedServedFn(torch.autograd.Function):
    """``y[s] = x[s] @ W_e(s)^T`` in bf16 over rows sorted by expert, the bank dequantized once.

    With a cut input -- NVFP4's `TrtLlmNvFp4ExpertsMonolithic` runs the bank cut, and FP8/INT8 cut per token --
    the rows are cut first, as `Stored.act_parts` says (NVFP4 per 16 values under the bank's `igscale`); the cut
    values are exact in bf16, and the bank's weights are kept in bf16 too (`weights_bf16`), the grouped matmul's
    dtype. Backward recomputes the bank; d(delta) is `ServedBank.delta_grad` in blocks of ``ROW_BLOCK_ELEMS``.
    """

    @staticmethod
    def forward(ctx, x, offs, delta, bank):  # noqa: ANN001, ANN205
        with torch.no_grad():
            w = bank.weights_bf16(delta)
            parts = bank.flat().act_parts
            if parts is None:
                xs = x.to(torch.bfloat16)
                s = None
            else:  # what the kernel multiplies: the cut already carries each block's scale
                xs, s = parts(x)
                xs = xs.to(torch.bfloat16)
            y = _gmm(xs, w.transpose(-2, -1), offs)
            y = y if s is None else y * s
            y = y.to(x.dtype)
        ctx.save_for_backward(x, offs, delta)
        ctx.bank = bank
        return y

    @staticmethod
    def backward(ctx, gy):  # noqa: ANN001, ANN205
        x, offs, delta = ctx.saved_tensors
        bank: ServedBank = ctx.bank
        parts = bank.flat().act_parts
        need_x, need_d = ctx.needs_input_grad[0], ctx.needs_input_grad[2]
        gx = gd = None
        if need_x:
            with torch.no_grad():
                w = bank.weights_bf16(delta)  # the forward's weight
                gx = _gmm(gy.to(torch.bfloat16), w, offs).to(x.dtype)
            del w
        if need_d:
            if parts is None:
                xs = x
            else:  # the kernel's rows: the cut values, as the per-expert backward uses them
                xs = parts(x)[0]
            gw = _gmm_weight_grad(gy, xs, offs, bank.num_experts)
            gd = torch.zeros_like(delta)
            eb = max(1, ROW_BLOCK_ELEMS // max(1, bank.out_features * bank.in_features))
            for e0 in range(0, bank.num_experts, eb):
                sl = slice(e0, min(bank.num_experts, e0 + eb))
                gd[sl] = bank.delta_grad(delta, gw[sl], sl)
            del gw
        return gx, None, gd, None
