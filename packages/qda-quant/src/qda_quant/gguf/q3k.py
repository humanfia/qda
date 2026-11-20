"""The ``q3k`` grid: llama.cpp's Q3_K, solved rather than fitted by ggml's search.

Q3_K stores a row in super-blocks of 256 values, 16 groups of 16. A group's value is ``d * s * q`` with signed
3-bit codes ``q`` in ``-4 .. 3``, a signed 6-bit ``s`` per group and one fp16 ``d`` per super-block (``110``
bytes a super-block). The code's high bit is stored inverted in the block's ``hmask``.

The grid fits ``s`` from the group's largest magnitude, on the side whose code has four steps (``-4`` for a
negative extreme, ``+3`` for a positive one, with the sign of ``s`` putting the four-step side there), so the
largest value lands on the four-step code before ``s`` is rounded to six bits; that rounding can still leave a
group's reconstruction short of its extreme (measured on random weights: 123 of 256 groups, worst 4.9 %). The
scale is negative for a group whose extreme is positive. GPTQ then rounds each column to the nearest of the
eight codes against that scale, with error feedback. The format is ``w3a16-q3k`` (``qda_quant.format``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..format import Q3K_FORMAT, Format
from ..solve.grid import GRIDS, Grid
from ..step import Refused
from .rule import Q3_K
from .scales import pack_q3k_scales, split_6bit, unpack_q3k_scales, weighted_records

if TYPE_CHECKING:
    from ..target import KernelPlan

#: Values per group, and per super-block.
GROUP = 16
SUPER = 256


class Q3K(Grid):
    """llama.cpp's Q3_K: signed 3-bit codes, 6-bit group scales, one fp16 ``d`` per super-block."""

    name = "q3k"
    gptq_fused = False
    range_rule = False
    stores = Q3K_FORMAT

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Only the format Q3_K is."""
        if fmt.name != Q3K_FORMAT:
            return f"Q3_K stores 3-bit signed codes in groups of 16: format {Q3K_FORMAT}, not {fmt.name}"
        return None

    def divides(self, fmt: Format, cols: int) -> bool:  # noqa: ARG002
        """Whether a row of `cols` values is what this format stores: whole 256-value super-blocks."""
        return cols % SUPER == 0

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa: ARG002
        """``fp16(d) * s`` per 16 values, ``s`` signed and six bits, its sign putting four steps on the extreme.

        The product is exact in fp32 (an 11-bit significand times a 6-bit integer), which is how the writer
        recovers the pair.

        Raises:
            Refused: `C` is not a whole number of 16-value groups.
        """
        c = w.shape[-1]
        if c % GROUP:
            raise Refused(f"grid {self.name!r}: a row of {c} values is not a whole number of 16-value groups")
        rows = w.reshape(-1, c).float()
        pad = -c % SUPER
        if pad:
            rows = torch.nn.functional.pad(rows, (0, pad))
        groups = rows.reshape(rows.shape[0], -1, GROUP)
        amax = groups.abs().amax(-1)
        signed = groups.gather(-1, groups.abs().argmax(-1, keepdim=True)).squeeze(-1)
        # A positive extreme can only use the three positive codes, so its scale is negative and the code is
        # -4 on the negative side, which then has four steps; a negative extreme takes the positive scale.
        real = torch.where(signed < 0, amax / 4, -amax / 3)
        blocks = real.reshape(rows.shape[0], -1, SUPER // GROUP)  # [rows, super-blocks, 16]
        # One fp16 `d` per 256 values, the super-block's largest group over 31: a super-block far below the
        # row's maximum keeps its own scale instead of sharing one sized for another block.
        d = (blocks.abs().amax(-1, keepdim=True) / 31).to(torch.float16).float()
        live = (amax > 0).reshape(rows.shape[0], -1, SUPER // GROUP)
        safe = torch.where(d > 0, d, torch.ones_like(d))
        s6 = torch.where(live, torch.clamp(torch.round(blocks / safe), -32, 31), torch.zeros_like(blocks))
        s6 = torch.where(live & (s6 == 0), torch.sign(blocks), s6)  # a live group keeps a nonzero scale
        scale = (d * s6).reshape(rows.shape[0], -1)[:, : c // GROUP]
        return scale.reshape(*w.shape[:-1], c // GROUP).to(w.device), None

    def fit_weighted(self, w: torch.Tensor, importance: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Native imatrix fit's exact fp16 times signed 6-bit scales, keeping the original row width."""
        records = weighted_records(w, importance, Q3_K, GROUP, SUPER, self.name)
        d, factors, _ = q3k_fields(records)
        scale = d.view("<f2").astype(np.float32)[:, None] * factors.astype(np.float32)
        c = w.shape[-1]
        scale = scale.reshape(-1, (c + -c % SUPER) // GROUP)[:, : c // GROUP].copy()
        return torch.from_numpy(scale).reshape(*w.shape[:-1], c // GROUP).to(w.device), None

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """As fitted."""
        return s

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Fp32: ``fp16(d) * 6-bit`` is exact there."""
        return s.float()

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """llama.cpp multiplies with the fp32 product itself."""
        return s.float()

    def codes(  # noqa: ARG002
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """``round(w / s)`` clamped to ``-4 .. 3``; a group with ``s == 0`` gets code 0."""
        assert z is None, "Q3_K stores a signed scale and no zero point"
        live = s != 0
        safe = torch.where(live, s, torch.ones_like(s))
        q = torch.clamp(torch.round(w / safe), qmin, qmax)
        return torch.where(live, q, torch.zeros_like(q))

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """``s * q`` in fp32, as ``dequantize_row_q3_K``."""
        assert z is None, "Q3_K stores a signed scale and no zero point"
        return s * q


def q3k_fields(records: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fp16 d bits [n], signed scale [n, 16], codes [n, 256])`` of Q3_K records, codes in ``-4 .. 3``."""
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, 110)
    hmask, qs, scales, d = rec[:, :32], rec[:, 32:96], rec[:, 96:108], rec[:, 108:]
    low, high = scales[:, :8], scales[:, 8:]
    sc = unpack_q3k_scales(low, high)
    q = qs.reshape(-1, 2, 32)
    qs_codes = np.stack([(q >> 2 * k) & 0x03 for k in range(4)], axis=2).reshape(-1, 16, 16)
    bits = np.stack([(hmask >> k) & 0x01 for k in range(8)], axis=1).reshape(-1, 16, 16)
    codes = (qs_codes.astype(np.int16) - (1 - bits.astype(np.int16)) * 4).astype(np.int8)
    return d.copy().view("<u2").reshape(-1), sc, codes.reshape(-1, SUPER)


def q3k_records(d: np.ndarray, scales: np.ndarray, codes: np.ndarray) -> bytes:
    """Q3_K records from ``fp16 d bits [n]``, signed scales ``[n, 16]`` and codes ``[n, 256]`` in ``-4 .. 3``."""
    q = np.asarray(codes, dtype=np.int16).reshape(-1, 16, 16)
    if (q < -4).any() or (q > 3).any():
        raise Refused("Q3_K codes are integers in -4 .. 3")
    ql = np.where(q >= 0, q, q + 4).astype(np.uint8)
    qs = np.zeros((q.shape[0], 2, 32), dtype=np.uint8)
    hmask = np.zeros((q.shape[0], 2, 16), dtype=np.uint8)
    for g in range(16):
        half, parity, k = g // 8, g % 2, (g % 8) // 2
        qs[:, half, parity * 16 : parity * 16 + 16] |= ql[:, g, :] << (2 * k)
        hmask[:, parity, :] |= (q[:, g, :] >= 0).astype(np.uint8) << (g // 2)
    low, high = pack_q3k_scales(np.asarray(scales, dtype=np.int16).reshape(-1, 16))
    return np.concatenate(
        (
            hmask.reshape(-1, 32),
            qs.reshape(-1, 64),
            low,
            high,
            np.asarray(d, dtype="<u2").view(np.uint8).reshape(-1, 2),
        ),
        axis=1,
    ).tobytes()


def q3k_bytes(
    qp: dict[str, Any],
    rows: torch.Tensor,
    values: torch.Tensor,  # noqa: ARG001
    name: str,
    workers: int | None = None,  # noqa: ARG001
) -> bytes:
    """Q3_K records of one qparams entry: signed codes and exact ``fp16(d) * s`` region scales."""
    if qp["format"] != Q3K_FORMAT or qp.get("zero_point") is not None:
        raise Refused(f"gguf: {name}: Q3_K requires {Q3K_FORMAT} on grid 'q3k', with no zero point")
    codes = qp["codes"].cpu()[rows]
    if codes.ndim != 2 or codes.shape[1] % SUPER:
        raise Refused(f"gguf: {name}: Q3_K requires whole 256-value super-blocks")
    if ((codes < -4) | (codes > 3) | (codes != codes.round())).any():
        raise Refused(f"gguf: {name}: Q3_K codes are integers in -4 .. 3")
    products = qp["scale"].cpu().float()[rows].reshape(-1, SUPER // GROUP).numpy()
    d, scales = split_6bit(products, name)
    named = codes.reshape(-1, SUPER // GROUP, GROUP).to(torch.int64).numpy()
    named[products == 0] = 0  # a zero scale decodes to zero whatever the codes are
    return q3k_records(d, scales, named.reshape(-1, SUPER))


def quantize_q3k(x: torch.Tensor) -> bytes:
    """Round-to-nearest Q3_K records of ``x [rows, C]`` with ``C`` a multiple of 256."""
    fmt = Format.parse(Q3K_FORMAT)
    grid = GRIDS["q3k"]
    scale, _ = grid.fit(fmt, x)
    wide = scale.reshape(x.shape[0], -1).repeat_interleave(GROUP, -1)
    codes = grid.codes(x.float(), wide, None, fmt.qmin, fmt.qmax)
    d, scales = split_6bit(scale.reshape(-1, SUPER // GROUP).numpy(), "rtn")
    return q3k_records(d, scales, codes.to(torch.int64).numpy())


Q3K_GRID = Q3K()
