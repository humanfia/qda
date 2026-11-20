"""NVFP4 weight numerics, the layout of compressed-tensors ``nvfp4-pack-quantized``.

A weight is ``e2m1(code) * scale / global_scale``: 4-bit E2M1 codes (sign in bit 3; magnitudes 0, 0.5, 1, 1.5,
2, 3, 4, 6), one FP8 E4M3 ``scale`` per 16 values along the input dimension, and one fp32 ``global_scale`` per
stored tensor that brings the group scales into E4M3's range. Two codes share a byte, column ``2j`` in the low
nibble.

The functions on stored tensors follow compressed-tensors' own arithmetic (``generate_gparam``,
``calculate_qparams``, ``cast_to_fp4``, ``pack_fp4_to_uint8``), which is what vLLM loads; ``tests/test_nvfp4.py``
checks them against the library value for value.

The ``nvfp4`` grid is how a build solves in this layout. Its scale is the fp32 step of one E2M1 unit,
``scale / global_scale``, and its codes are signed positions on the E2M1 ladder (-7 .. 7). The grid keeps every
step on the lattice a checkpoint can hold: an E4M3 value over a global scale that is a power of two. The kernel
uses only the quotient, so any positive global scale serves; a power of two makes the quotient exact in fp32 and
makes the lattice the same whichever rows share the global scale, so a step may fit one module of a fused group,
or one block of a search, and land on the steps the whole unit stores. compressed-tensors' own rule
(``448 * 6 / absmax``, `global_scale`) gives another lattice of the same density; its round-to-nearest scales
differ from this grid's by E4M3 rounding only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..format import NVFP4_FORMAT
from .grid import Grid

if TYPE_CHECKING:
    from ..format import Format
    from ..target import KernelPlan

__all__ = [
    "E2M1",
    "GROUP",
    "NVFP4_GRID",
    "Nvfp4Grid",
    "block_cut",
    "block_cut_parts",
    "block_scales",
    "codes_of_positions",
    "dequant",
    "e2m1_codes",
    "e2m1_round",
    "e2m1_values",
    "global_scale",
    "input_global_scale",
    "lattice",
    "lattice_global",
    "pack_nibbles",
    "positions_of_codes",
    "quantize",
    "unpack_nibbles",
]

#: E2M1 magnitudes by the low three bits of a code.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: Values sharing one E4M3 scale.
GROUP = 16
_FP4_MAX = 6.0
_FP8_MAX = 448.0
#: compressed-tensors' replacement for a zero group scale, the smallest E4M3 value it stores.
_SCALE_FLOOR = 0.125


def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (int8, sign in bit 3) of `x` clamped to +-6: nearest value, a midpoint to the even code.

    A negative `x` that rounds to zero keeps its sign (code 8), as the library's packer stores it.
    """
    a = x.abs()
    # Thresholds are the midpoints; `>` or `>=` sends each tie to the code whose mantissa bit is 0.
    idx = (
        (a > 0.25).to(torch.int8)
        + (a >= 0.75)
        + (a > 1.25)
        + (a >= 1.75)
        + (a > 2.5)
        + (a >= 3.5)
        + (a > 5.0)
    ).to(torch.int8)
    return idx | ((x < 0).to(torch.int8) << 3)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """The fp32 value of E2M1 `codes`."""
    c = codes.to(torch.int64)
    v = torch.tensor(E2M1, dtype=torch.float32, device=codes.device)[c & 7]
    return torch.where((c & 8) != 0, -v, v)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Codes ``[..., C]`` as ``weight_packed`` uint8 ``[..., C / 2]``.

    Raises:
        ValueError: `C` is odd.
    """
    if codes.shape[-1] % 2:
        raise ValueError(f"an NVFP4 row packs two codes a byte; {codes.shape[-1]} columns is odd")
    c = codes.to(torch.uint8) & 15
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """``weight_packed`` uint8 ``[..., C / 2]`` as codes int8 ``[..., C]``."""
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int8)


def global_scale(amax: torch.Tensor) -> torch.Tensor:
    """The fp32 global scale of tensors whose largest magnitudes are `amax`: ``448 * 6 / amax``.

    `amax` keeps the dtype of the tensor it was measured on. The library divides in that dtype, so a bf16 weight's
    global scale is a bf16 value; a zero or non-finite result is 1.
    """
    dt = amax.dtype
    a = torch.clamp(amax, min=torch.finfo(dt).tiny)
    # torch evaluates 2688 / a on a one-element tensor as reciprocal, then product, each rounded to `dt`.
    gs = ((1.0 / a.float()).to(dt).float() * (_FP8_MAX * _FP4_MAX)).to(dt)
    return torch.nan_to_num(gs, nan=1.0, posinf=1.0, neginf=1.0).to(torch.float32)


def block_scales(w: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """Min/max E4M3 scales ``[..., R, C / 16]`` (as fp32) of ``w [..., R, C]`` under `gscale`.

    `gscale` broadcasts against ``[..., R, 1]``. A zero group stores 0.125.

    Raises:
        ValueError: `C` is not a whole number of 16-value groups.
    """
    if w.shape[-1] % GROUP:
        raise ValueError(
            f"NVFP4 scales cover 16 values; {w.shape[-1]} columns is not a whole number of groups"
        )
    amax = w.unflatten(-1, (-1, GROUP)).abs().amax(-1)
    # The quotient is taken in the weight's dtype and the product in fp32, as calculate_qparams does.
    s = gscale.to(torch.float32) * (amax / _FP4_MAX).to(torch.float32)
    s = torch.clamp(s, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return torch.where(s == 0, torch.full_like(s, _SCALE_FLOOR), s)


def _steps(scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """The fp32 value of one E2M1 unit per column: ``scale / gscale``, widened to the columns."""
    return (scale.to(torch.float32) / gscale.to(torch.float32)).repeat_interleave(GROUP, dim=-1)


def quantize(w: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest codes of ``w [..., R, C]`` under stored `scale` ``[..., R, C / 16]`` and `gscale`."""
    return e2m1_codes(torch.clamp(w.float() / _steps(scale, gscale), -_FP4_MAX, _FP4_MAX))


def dequant(codes: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """The fp32 weight `codes` decode to under stored `scale` and `gscale`."""
    return e2m1_values(codes) * _steps(scale, gscale)


def positions_of_codes(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 `codes` as signed ladder positions (int8, -7 .. 7); a negative zero becomes position 0."""
    c = codes.to(torch.int8)
    mag = c & 7
    return torch.where((c & 8) != 0, -mag, mag)


def codes_of_positions(q: torch.Tensor) -> torch.Tensor:
    """Signed ladder positions (-7 .. 7) as E2M1 codes (int8, sign in bit 3)."""
    p = q.to(torch.int8)
    return p.abs() | ((p < 0).to(torch.int8) << 3)


def lattice_global(steps: torch.Tensor) -> torch.Tensor:
    """The global scale ``[..., 1, 1]`` of steps ``[..., R, G]``: the largest power of two under ``448 / max``.

    Every leading index is one stored tensor (an expert of a bank); a dense unit's rows share one. A unit of
    zeros gets 1.
    """
    top = steps.float().amax((-2, -1), keepdim=True)
    # frexp: 448 / top = m * 2^e with m in [0.5, 1), so the largest power of two not above it is 2^(e - 1).
    exponent = torch.frexp(_FP8_MAX / top.clamp(min=torch.finfo(torch.float32).tiny))[1] - 1
    return torch.where(top > 0, torch.ldexp(torch.ones_like(top), exponent), torch.ones_like(top))


def lattice(raw: torch.Tensor) -> torch.Tensor:
    """Raw fp32 steps ``[..., R, G]`` on the lattice a checkpoint holds: ``e4m3(raw * G) / G``, `G` their global.

    A step E4M3 rounds to zero (a zero group, or one about 2^17 below the unit's largest) becomes ``0.125 / G``,
    as compressed-tensors stores it; its weights code to zero. The result is a fixed point of this function.
    """
    g = lattice_global(raw)
    f8 = torch.clamp(raw.float() * g, max=_FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return torch.where(f8 == 0, torch.full_like(f8, _SCALE_FLOOR), f8) / g


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """The E2M1 magnitude nearest each ``0 <= a <= 6``, a midpoint to the even code: `e2m1_codes`, as values."""
    return torch.where(
        a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2)
    )


def block_cut_parts(x: torch.Tensor, gscale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's cut of an input ``[..., C]``: (values the matmul reads, fp32; a scale of ones ``[..., 1]``).

    Each 16 values get the scale ``e4m3(gscale * absmax / 6) / gscale`` and round to the nearest E2M1 multiple
    of it; a block of zeros stays zero. With `gscale` (the module's ``input_global_scale``) a block above
    ``448 * 6 / gscale`` saturates and one about 2^15 below it loses its scale's mantissa, as served. Without
    it the scale is rounded to E4M3's four significant bits at any magnitude, which is the served cut whenever
    the global scale is a power of two and the block is inside that range.

    The block scale differs by block, so the values come back already scaled, with a scale of ones.
    """
    xf = x.float()
    raw = xf.unflatten(-1, (-1, GROUP)).abs().amax(-1) * (1.0 / _FP4_MAX)
    if gscale is None:
        mantissa, exponent = torch.frexp(raw)
        step = torch.ldexp(mantissa.to(torch.float8_e4m3fn).to(torch.float32), exponent)
    else:
        g = gscale.to(torch.float32)
        step = torch.clamp(raw * g, max=_FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32) / g
    step = step.repeat_interleave(GROUP, dim=-1)
    live = step > 0
    y = torch.where(live, xf / torch.where(live, step, torch.ones_like(step)), torch.zeros_like(xf))
    y = torch.copysign(e2m1_round(y.abs().clamp(max=_FP4_MAX)), y)
    return y * step, torch.ones_like(xf[..., :1])


def block_cut(x: torch.Tensor, gscale: torch.Tensor | None = None) -> torch.Tensor:
    """`block_cut_parts` as a round trip in the dtype of `x`; every value it returns is exact in bf16."""
    if x.shape[-1] == 0:
        return torch.empty_like(x)
    return block_cut_parts(x, gscale)[0].to(x.dtype)


def input_global_scale(absmax: torch.Tensor) -> torch.Tensor:
    """The ``input_global_scale`` (fp32 scalar) for inputs up to `absmax`: the largest power of two whose
    range ``448 * 6 / scale`` holds them. A power of two keeps `block_cut`'s scales the ones the kernel makes;
    the range is one to two times `absmax`. Inputs that were never seen (`absmax` 0) get 1.
    """
    top = absmax.float().amax()
    exponent = torch.frexp(_FP8_MAX * _FP4_MAX / top.clamp(min=torch.finfo(torch.float32).tiny))[1] - 1
    return torch.where(top > 0, torch.ldexp(torch.ones_like(top), exponent), torch.ones_like(top))


class Nvfp4Grid(Grid):
    """NVFP4's two-level scales: an E4M3 value per 16 columns over one power-of-two global scale per unit."""

    name = "nvfp4"
    range_rule = False
    stores = NVFP4_FORMAT
    gptq_fused = False

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:
        """Only the format NVFP4 is, on a kernel that reads its scales."""
        if fmt.name != NVFP4_FORMAT:
            return f"NVFP4 stores E2M1 codes in groups of 16: format {NVFP4_FORMAT}, not {fmt.name}"
        if plan.scale_dtype != "nvfp4":
            return f"the kernel reads {plan.scale_dtype} group scales, not NVFP4's E4M3 under a global scale"
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa: ARG002
        """Each group's largest magnitude on the ladder's 6, snapped to the lattice."""
        return lattice(w.float().unflatten(-1, (-1, GROUP)).abs().amax(-1) / _FP4_MAX), None

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """On the lattice (`lattice`); a step a fit or a search already put there is unchanged."""
        return lattice(s)

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """The fp32 step; the packer splits it into the E4M3 scale and the global scale (`global_scale`)."""
        return s.float()

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """The step itself: the kernel multiplies with ``scale / global_scale``, which is exact here."""
        return s.float()

    def global_scale(self, s: torch.Tensor) -> torch.Tensor:
        """The global scale ``[..., 1, 1]`` a unit's stored steps share.

        Raises:
            ValueError: a step is off the lattice, so the checkpoint could not hold what the build solved.
        """
        g = lattice_global(s)
        f8 = s.float() * g
        if not torch.equal(f8, f8.to(torch.float8_e4m3fn).to(torch.float32)) or bool((f8 == 0).any()):
            raise ValueError("an NVFP4 step is not an E4M3 value over the unit's global scale")
        return g

    def codes(
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """The position of the E2M1 value nearest ``w / s``, a midpoint to the even code as the library rounds."""
        assert z is None, "NVFP4 is symmetric"
        near = positions_of_codes(e2m1_codes(torch.clamp(w / s, -_FP4_MAX, _FP4_MAX)))
        return near.clamp(qmin, qmax).float()

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """``e2m1(q) * s`` in fp32."""
        assert z is None, "NVFP4 is symmetric"
        return e2m1_values(codes_of_positions(q)) * s


NVFP4_GRID = Nvfp4Grid()
