"""Quantizers: pick integer codes against the scales the serving kernel really multiplies with.

A `Quantizer` binds a `Format` and a `KernelPlan` to a `Grid` (`qda_quant.solve.grid`). Every method works on stacks:
weights are ``[..., out, in]`` (a dense Linear is ``[out, in]``, an expert bank ``[E, out, in]``), scales and zero
points are ``[..., out, groups]`` in fp32.

Conventions (internal, all fp32):

* codes ``q`` are integers in ``[fmt.qmin, fmt.qmax]``; value = ``(q - z) * s`` (``z = 0`` symmetric).
  Asymmetric codes are unsigned (``0 .. 2^b - 1``); export shifts them to compressed-tensors' signed form.
  FP8 weights (``w8a8-fp8``) are the exception: a code is the E4M3 value the checkpoint holds, and
  value = ``q * s`` under the per-channel fp32 scale.
* ``scale`` is the *stored* scale. `served_scale` is what the kernel multiplies with: the grid's rounding
  for WNA16 kernels, the FP8 E4M3 x per-row fp32 grid for CUTLASS W4A8 (``plan.scale_dtype``), the stored
  fp32 scale for FP8 weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..format import Format
from ..step import Refused
from ..target import KernelPlan
from .grid import GRIDS, Grid
from .nvfp4 import block_cut, block_cut_parts

FP8_MAX = 448.0
#: vLLM's floor on a dynamic per-token FP8 scale (``_FP8_MIN_SCALING_FACTOR``)
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def groups(x: torch.Tensor, g: int) -> torch.Tensor:
    """``[..., C]`` -> ``[..., C // g, g]`` (C must be divisible by g)."""
    return x.reshape(*x.shape[:-1], x.shape[-1] // g, g)


def widen(t: torch.Tensor, g: int) -> torch.Tensor:
    """Per-group ``[..., G]`` -> per-column ``[..., G * g]``."""
    return t.repeat_interleave(g, dim=-1)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """WNA16 kernels dequantize in bf16: ``(q - z) * s`` is exact in fp32, then rounded once to bf16."""
    return x.to(torch.bfloat16).float()


def fp8_scale_parts(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VLLM ``convert_bf16_scales_to_fp8`` of a stored scale matrix ``[..., R, G]``.

    Returns:
        (f8, chan): the e4m3 group factor (as fp32) ``[..., R, G]`` and the per-row fp32 factor ``[..., R, 1]``;
        the kernel's weight is ``e4m3(q * f8) * chan``.
    """
    s = scale.to(torch.bfloat16)
    c = (s.abs().amax(-1, keepdim=True).float() / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    codes = (s.float() * c.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (codes / 8.0).to(torch.float8_e4m3fn).float(), c * 8.0


def fp8_token_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token FP8 E4M3 quantization of activations: (e4m3 codes as fp32, per-token fp32 scale)."""
    s = (x.abs().amax(-1, keepdim=True).float() / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    codes = (x.float() * s.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return codes, s


#: A kernel's cut of its input: ``x -> (codes as fp32, per-token fp32 scale)``. A cut whose scale is finer than a
#: token (NVFP4's blocks of 16) returns the scaled values and a scale of ones.
ActParts = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _round_trip(x: torch.Tensor, parts: ActParts) -> torch.Tensor:
    """``codes * scale`` of a cut, in the dtype of `x`; empty when the last dimension has no channel to scale."""
    if x.shape[-1] == 0:
        return torch.empty_like(x)
    codes, s = parts(x)
    return (codes * s).to(x.dtype)


def fp8_token_reference(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-token FP8 E4M3 quant-dequant of activations, as vLLM's CUTLASS W4A8 path computes it.

    The elementwise reference: `fp8_token_parts` materializes the fp32 codes and scales, then the round trip
    multiplies and casts back. The fused kernel (`qda_quant.solve._triton.fp8_token`) must give these bits.

    A zero-size last dimension has no channel to scale, so the result is an empty tensor of the input's shape
    and dtype, exactly as the fused kernel returns it.
    """
    return _round_trip(x, fp8_token_parts)


def triton_fp8(device: torch.device) -> bool:
    """Whether Triton kernels can cast to E4M3 on `device`.

    Triton's ``fp8e4nv`` needs compute capability 8.9 or newer (Ada, Hopper, Blackwell); on an A100 (8.0) a
    kernel that casts to it does not compile. Building there for a target served elsewhere is legitimate
    (`Target.on_cards`), so the fused FP8 paths fall back to their torch references there (the references whose
    bits the kernels reproduce where they compile).
    """
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)


def fp8_token(x: torch.Tensor, impl: str = "auto") -> torch.Tensor:
    """Dynamic per-token FP8 E4M3 quant-dequant of activations, one fused pass on CUDA.

    Args:
        x: activations ``[..., C]``, the last dimension is the token's channels.
        impl: ``"auto"`` (the fused Triton kernel on CUDA for bf16 or fp16 on a card `triton_fp8` admits, the
            torch reference otherwise), ``"triton"`` (require the fused kernel) or ``"torch"`` (the reference).

    Raises:
        ValueError: an unknown ``impl``, or ``"triton"`` for an input or a card the kernel does not take.
    """
    if impl not in ("auto", "torch", "triton"):
        raise ValueError(f"impl {impl!r} is not 'auto', 'torch' or 'triton'")
    fused = None
    if impl != "torch" and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        if not triton_fp8(x.device):
            if impl == "triton":
                raise ValueError("impl 'triton' casts to E4M3, which needs compute capability 8.9 or newer")
            return fp8_token_reference(x)
        try:
            from ._triton import fp8_token as fused  # noqa: PLC0415  (Triton stays an optional import)
        except ImportError:
            if impl == "triton":
                raise
    if fused is None:
        if impl == "triton":
            raise ValueError("impl 'triton' needs a CUDA bf16 or fp16 input")
        return fp8_token_reference(x)
    return fused(x)


def int8_token_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token symmetric INT8 quantization of activations: (int8 codes as fp32, per-token fp32 scale).

    vLLM's ``scaled_int8_quant``: ``s = absmax / 127`` with no floor, and the codes multiply by ``127 / absmax``
    (dividing by ``s`` rounds differently), round half to even, and saturate. A NaN product becomes code 0, as in the
    kernel's saturating cast; that is every code of a zero row, whose scale is 0, and the zeros of a row so small
    that ``127 / absmax`` overflows.

    Both divisions are true divisions, as in the kernel, hence the tensor 127: ``127.0 / t`` is torch's
    ``t.reciprocal() * 127.0``, and ``t / 127.0`` on CUDA also gives scales that differ in the last bit.
    """
    amax = x.abs().amax(-1, keepdim=True).float()
    c = amax.new_tensor(127.0)
    return (x.float() * (c / amax)).nan_to_num(0.0).round().clamp(-128, 127), amax / c


def int8_token(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-token INT8 quant-dequant of activations, as vLLM's ``scaled_int8_quant`` cuts them."""
    return _round_trip(x, int8_token_parts)


class ActCut(NamedTuple):
    """How a kernel cuts its input, in the two forms the engine uses.

    Attributes:
        parts: the codes and scale; the served matmul multiplies the codes, then scales the result.
        qdq: the quant-dequant, in the input's dtype; what a calibration forward reads.
    """

    parts: ActParts
    qdq: Callable[[torch.Tensor], torch.Tensor]


#: The cut of each `KernelPlan.act_dtype`. None: the kernel reads its input as it is.
ACT_CUTS: dict[str, ActCut | None] = {
    "bf16": None,
    "fp8_e4m3": ActCut(fp8_token_parts, fp8_token),
    "int8": ActCut(int8_token_parts, int8_token),
    "nvfp4": ActCut(block_cut_parts, block_cut),
}


@dataclass(frozen=True)
class Quantizer:
    """Round-to-grid for one (format, kernel plan).

    Attributes:
        fmt: the stored format.
        plan: how the serving kernel uses it (``scale_dtype`` decides the scale grid).
        grid: the rule for group scales.
    """

    fmt: Format
    plan: KernelPlan
    grid: Grid

    @classmethod
    def of(cls, fmt: Format, plan: KernelPlan, grid: str) -> Quantizer:
        """The quantizer for a grid named by the config: the one place a name becomes a `Grid`."""
        return cls(fmt, plan, GRIDS[grid])

    @property
    def fp8_scales(self) -> bool:
        """True when the kernel multiplies with FP8-cast group scales (CUTLASS W4A8)."""
        return self.plan.scale_dtype == "fp8_e4m3"

    @property
    def fp8_weights(self) -> bool:
        """True when the weights are FP8 E4M3 codes under a per-channel fp32 scale (FP8 W8A8).

        The checkpoint holds the E4M3 values themselves, and the kernel multiplies each row by its fp32
        ``weight_scale``; there is no second rounding level, so none of the FP8 group-scale machinery
        (`fp8_scales`) applies.
        """
        return self.fmt.weight_type == "fp8"

    def group(self, cols: int) -> int:
        """Group size for a matrix with `cols` input columns."""
        return cols if self.fmt.group_size == -1 else self.fmt.group_size

    def range_params(self, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The grid's qparams of group ranges ``lo <= 0 <= hi``: (scale, zero | None).

        Raises:
            Refused: the grid fits on values and has no range rule.
        """
        if not self.grid.range_rule:
            raise Refused(f"grid {self.grid.name!r} fits its scales on the weights; it takes no group range")
        s = self.usable(self.grid.scale(self.fmt, lo, hi))
        return s, None if self.fmt.symmetric else self.grid.zero(self.fmt, lo, s)

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """A raw scale as the grid keeps it, with the kernel's floor on FP8 group factors."""
        return self._fp8_floor(self.grid.usable(s))

    def stored_scale(self, s: torch.Tensor) -> torch.Tensor:
        """The scale in the dtype the checkpoint holds: fp32 for FP8 weights, else the grid's."""
        if self.fp8_weights:
            return s.float()
        return self.grid.stored(s)

    def stored_codes(self, q: torch.Tensor) -> torch.Tensor:
        """Dequantized-grid codes in the dtype ``ctx.qparams`` records: E4M3 for FP8 weights, else int8."""
        return q.to(torch.float8_e4m3fn) if self.fp8_weights else q.to(torch.int8)

    def stored_zero(self, z: torch.Tensor) -> torch.Tensor:
        """Zero points in the dtype the checkpoint holds."""
        return self.grid.stored_zero(z)

    def _fp8_floor(self, s: torch.Tensor) -> torch.Tensor:
        """With FP8 group scales, keep every group's e4m3 factor nonzero.

        The kernel stores ``e4m3(e4m3(s / chan) / 8)`` per group, with ``chan`` = row max / 448. A group far below its
        row's max (an all-zero group next to a large one) would get a factor of 0 and NaN codes. Scales below
        ``chan * 2^-5`` (with ``chan`` clamped like the kernel's, twice the e4m3 subnormal limit) are raised to
        it; other groups are unchanged.
        """
        if not self.fp8_scales:
            return s
        chan = (s.amax(-1, keepdim=True) / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        return torch.maximum(s, chan * 2.0**-5)

    def params(
        self, w: torch.Tensor, shrink: float | torch.Tensor = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Min/max qparams of ``w [..., R, C]`` per group, the range scaled by `shrink`.

        `shrink` is a clip ratio; a tensor broadcasts over ``[..., R, G]``. A grid without a range rule fits on
        `w` itself (`Grid.fit`) and takes no clip ratio.

        Raises:
            Refused: a clip ratio for a grid without a range rule.
        """
        if not self.grid.range_rule:
            if not (isinstance(shrink, float) and shrink == 1.0):
                raise Refused(
                    f"grid {self.grid.name!r} fits its scales on the weights; clip searches do not apply"
                )
            return self.grid.fit(self.fmt, w)
        wg = groups(w.float(), self.group(w.shape[-1]))
        lo, hi = wg.amin(-1).clamp(max=0), wg.amax(-1).clamp(min=0)
        return self.range_params(lo * shrink, hi * shrink)

    def fit_codes(
        self, w: torch.Tensor, clip: str = "minmax"
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None:
        """(scale, zero, codes) of ``w`` in one pass when the grid can make both, else None.

        Only a grid without a range rule fits on the weights at all, and only the ``minmax`` clip leaves the fitted
        ranges alone; anything else needs `params` and then `quantize`.
        """
        if clip != "minmax" or self.grid.range_rule:
            return None
        return self.grid.fit_codes(self.fmt, w)

    def served_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """Per-group scale the kernel multiplies with (fp32): the grid's rounding, ``f8 * chan`` for FP8 scales,
        or the stored fp32 scale for FP8 weights."""
        if self.fp8_scales:
            f8, chan = fp8_scale_parts(scale)
            return f8 * chan
        if self.fp8_weights:
            return scale.float()
        return self.grid.served(scale)

    def quantize(self, w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None) -> torch.Tensor:
        """Codes of ``w [..., R, C]`` on the served grid of (scale, zero), as fp32 values.

        Integer formats give integer-valued codes; FP8 weights give the E4M3 values themselves, exactly as a
        checkpoint stores them (``e4m3(w / s)`` clamped to 448). For FP8 scales each code is the one among
        ``round(w / s)`` and its neighbours whose served value ``e4m3(q * f8) * chan`` is nearest ``w``.
        """
        g = self.group(w.shape[-1])
        f = self.fmt
        if self.fp8_scales:
            f8, chan = fp8_scale_parts(scale)
            return round_fp8(w.float(), widen(f8, g), chan, f.qmin, f.qmax)[0]
        if self.fp8_weights:
            s = widen(self.served_scale(scale), g)
            return torch.clamp(w.float() / s, f.qmin, f.qmax).to(torch.float8_e4m3fn).float()
        s = widen(self.served_scale(scale), g)
        if not self.grid.range_rule:
            return self.grid.codes(w.float(), s, None if zero is None else widen(zero, g), f.qmin, f.qmax)
        x = w.float() / s
        if zero is not None:
            x = x + widen(zero, g)
        return torch.clamp(torch.round(x), f.qmin, f.qmax)

    def dequant_parts(self, q: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 scales: the kernel's weight as ``(e4m3(q * f8), chan)``, ``dequant = part * chan``.

        The part is an E4M3 value (exact in bf16 and TF32); `chan` ``[..., R, 1]`` multiplies after the MMA.
        """
        f8, chan = fp8_scale_parts(scale)
        return (q.float() * widen(f8, self.group(q.shape[-1]))).to(torch.float8_e4m3fn).float(), chan

    def dequant(self, q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None) -> torch.Tensor:
        """The weight the kernel multiplies with (fp32) for codes `q` and stored (scale, zero)."""
        g = self.group(q.shape[-1])
        if self.fp8_scales:
            part, chan = self.dequant_parts(q, scale)
            return part * chan
        if self.fp8_weights:
            return q.float() * widen(self.served_scale(scale), g)
        if not self.grid.range_rule:
            return self.grid.value(
                q.float(), widen(self.served_scale(scale), g), None if zero is None else widen(zero, g)
            )
        x = q.float() if zero is None else q.float() - widen(zero, g)
        return _bf16(x * widen(self.served_scale(scale), g))

    def fake(self, w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None) -> torch.Tensor:
        """Quantize-dequantize: what the served model computes with, fp32."""
        return self.dequant(self.quantize(w, scale, zero), scale, zero)


def round_fp8(
    w: torch.Tensor, f8: torch.Tensor, chan: torch.Tensor, qmin: int, qmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest code under the CUTLASS W4A8 weight ``e4m3(q * f8) * chan``; ties keep ``round(w / s)``.

    Args:
        w: weights ``[..., R, C]`` (or one column ``[..., R]``).
        f8: e4m3 group factor broadcast to `w`'s shape.
        chan: per-row factor ``[..., R, 1]`` (``[..., R]`` for a single column).
        qmin: smallest code.
        qmax: largest code.

    Returns:
        (codes, served values), both fp32.
    """
    q0 = torch.clamp(torch.round(w / (f8 * chan)), qmin, qmax)
    best_q = q0
    best_v = (q0 * f8).to(torch.float8_e4m3fn).float() * chan
    best_e = (w - best_v).abs()
    for d in (-1.0, 1.0):
        q = torch.clamp(q0 + d, qmin, qmax)
        v = (q * f8).to(torch.float8_e4m3fn).float() * chan
        e = (w - v).abs()
        better = e < best_e
        best_q, best_v, best_e = (
            torch.where(better, q, best_q),
            torch.where(better, v, best_v),
            torch.minimum(e, best_e),
        )
    return best_q, best_v
