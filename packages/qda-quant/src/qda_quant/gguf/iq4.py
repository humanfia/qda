"""The ``iq4nl`` and ``iq4xs`` grids: llama.cpp's nonlinear 4-bit codebooks.

Both formats store a 4-bit code that indexes one fixed table of sixteen int8 values, ``kvalues_iq4nl``
(``-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113``), scaled per group:

* IQ4_NL: one fp16 ``d`` per 32 values, ``18`` bytes a block.
* IQ4_XS: 8 groups of 32 in a 256-value super-block, each with a signed 6-bit scale ``s`` and the super-block's
  fp16 ``d``; a value is ``d * s * kvalues[q]``, ``136`` bytes a super-block.

A group's scale is the table's extreme on the side of the group's largest value over that value's magnitude, so
the largest value lands on a table entry (``113`` when it is positive, ``-127`` when it is negative) *before*
the scale is rounded. IQ4_NL's ``d`` is the fp16 scale itself, so the extreme is exact there; IQ4_XS rounds its
6-bit factor after ``d`` is fixed, so a group's reconstruction can fall short of its largest magnitude (measured
on random weights: 47 of 128 groups, worst 2.4 %). GPTQ then codes every column as the nearest table entry to
``w / scale``, with the usual error feedback: this format's rounding step is a search over sixteen entries, not
``round()``.

The formats are ``w4a16-iq4nl`` and ``w4a16-iq4xs`` (``qda_quant.format``): a code is a table index, so the
format's weight type is the table's, and its ``qmin``/``qmax`` are 0 and 15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from ..format import IQ4NL_FORMAT, IQ4XS_FORMAT, Format
from ..solve.grid import GRIDS, Grid
from ..step import Refused
from .rule import IQ4_NL, IQ4_XS
from .scales import split_6bit, weighted_records

if TYPE_CHECKING:
    from ..target import KernelPlan

#: ``kvalues_iq4nl``: the sixteen values a code can mean, in code order.
KVALUES = np.array((-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113), dtype=np.int8)
#: The table's largest value on each side: a group's fit maps its extreme onto one of these.
POS_MAX, NEG_MAX = 113, 127
#: Values per group, and per IQ4_XS super-block.
GROUP = 32
SUPER = 256


def _table(device: torch.device | None = None) -> torch.Tensor:
    """`KVALUES` as an fp32 tensor on `device`."""
    return torch.from_numpy(KVALUES.astype(np.float32)).to(device)


def _midpoints(device: torch.device | None = None) -> torch.Tensor:
    """The fifteen boundaries between the table's entries, for `torch.bucketize`."""
    t = _table(device)
    return (t[:-1] + t[1:]) / 2


def _nearest(x: torch.Tensor) -> torch.Tensor:
    """Index of the table entry nearest each value of `x`, as a long tensor."""
    return torch.bucketize(x, _midpoints(x.device))


def _extreme(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(largest magnitude, its signed value)`` per group of ``values [..., n, GROUP]``."""
    amax = values.abs().amax(-1)
    signed = values.gather(-1, values.abs().argmax(-1, keepdim=True)).squeeze(-1)
    return amax, signed


def _fit_iq4(values: torch.Tensor) -> torch.Tensor:
    """Per-group scale of ``values [rows, blocks, GROUP]``: the extreme over the table's extreme on its side."""
    amax, signed = _extreme(values)
    side = torch.where(
        signed >= 0, torch.full_like(amax, float(POS_MAX)), torch.full_like(amax, float(NEG_MAX))
    )
    return torch.where(amax > 0, amax / side, torch.zeros_like(amax))


def _pad_to(rows: torch.Tensor, block: int) -> torch.Tensor:
    """`rows [n, C]` zero-padded on the right to a whole number of `block` values."""
    pad = -rows.shape[-1] % block
    return torch.nn.functional.pad(rows, (0, pad)) if pad else rows


class _CodebookGrid(Grid):
    """Shared fit and decode of the two IQ4 formats; the subclasses fix the block and the record layout."""

    gptq_fused = False
    range_rule = False
    group = GROUP
    #: Values one stored block holds: 32 for IQ4_NL, 256 for IQ4_XS.
    block: ClassVar[int] = GROUP

    def divides(self, fmt: Format, cols: int) -> bool:  # noqa: ARG002
        """Whether a row of `cols` values is what this format stores: whole super-blocks."""
        return cols % self.block == 0

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """As fitted: every value is an exact fp16 scale (IQ4_NL) or an exact fp16 times 6-bit product."""
        return s

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Fp32: IQ4_NL's ``d`` is fp16 and IQ4_XS's product is exact there."""
        return s.float()

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """llama.cpp multiplies with the fp32 scale itself; it does not round it to bf16."""
        return s.float()

    def codes(
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """Nearest table entry to ``w / s``; a zero scale (a zero group) decodes to code 0, which is 0 * value."""
        assert z is None, "the IQ4 formats are symmetric"
        live = s != 0
        safe = torch.where(live, s, torch.ones_like(s))
        idx = _nearest(torch.where(live, w / safe, torch.zeros_like(w))).clamp(qmin, qmax)
        return torch.where(live, idx, torch.zeros_like(idx)).float()

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """``s * kvalues[q]`` in fp32, as ``dequantize_row_iq4_nl`` / ``dequantize_row_iq4_xs``."""
        assert z is None, "the IQ4 formats are symmetric"
        return s * _table(q.device)[q.long()]


class IQ4NL(_CodebookGrid):
    """llama.cpp's IQ4_NL: one fp16 ``d`` per 32 values, codes indexing `KVALUES`."""

    name = "iq4nl"
    stores = IQ4NL_FORMAT
    block = GROUP

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Only the format IQ4_NL is."""
        if fmt.name != IQ4NL_FORMAT:
            return f"IQ4_NL stores 4-bit table indices in groups of 32: format {IQ4NL_FORMAT}, not {fmt.name}"
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa: ARG002
        """One fp16 scale per 32 values, the extreme over the table's extreme on its side.

        Raises:
            Refused: `C` is not a whole number of 32-value groups.
        """
        c = w.shape[-1]
        if c % GROUP:
            raise Refused(f"grid {self.name!r}: a row of {c} values is not a whole number of 32-value groups")
        rows = _pad_to(w.reshape(-1, c).float(), GROUP)
        blocks = rows.reshape(rows.shape[0], -1, GROUP)
        scale = _fit_iq4(blocks).to(torch.float16).float()
        scale = torch.where(blocks.abs().amax(-1) > 0, scale, torch.zeros_like(scale))
        return scale[:, : c // GROUP].reshape(*w.shape[:-1], c // GROUP).to(w.device), None

    def fit_weighted(self, w: torch.Tensor, importance: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Native imatrix fit's signed fp16 scale per 32 values, with GPTQ codes fitted separately."""
        records = weighted_records(w, importance, IQ4_NL, GROUP, GROUP, self.name)
        d, _ = iq4nl_fields(records)
        scale = torch.from_numpy(d.view("<f2").astype(np.float32))
        return scale.reshape(*w.shape[:-1], w.shape[-1] // GROUP).to(w.device), None


class IQ4XS(_CodebookGrid):
    """llama.cpp's IQ4_XS: 8 groups of 32 per 256 values, each with an fp16 ``d`` times a signed 6-bit scale."""

    name = "iq4xs"
    stores = IQ4XS_FORMAT
    block = SUPER

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Only the format IQ4_XS is."""
        if fmt.name != IQ4XS_FORMAT:
            return f"IQ4_XS stores 4-bit table indices in groups of 32: format {IQ4XS_FORMAT}, not {fmt.name}"
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa: ARG002
        """``fp16(d) * s`` per 32 values: ``d`` is the super-block's largest group over 31, so ``s`` fits six bits.

        The product is exact in fp32 (an 11-bit significand times a 6-bit integer), which is how the writer
        recovers the pair.

        Raises:
            Refused: `C` is not a whole number of 32-value groups.
        """
        c = w.shape[-1]
        if c % GROUP:
            raise Refused(f"grid {self.name!r}: a row of {c} values is not a whole number of 32-value groups")
        rows = _pad_to(w.reshape(-1, c).float(), SUPER)
        blocks = rows.reshape(rows.shape[0], -1, GROUP)
        groups = _fit_iq4(blocks).reshape(rows.shape[0], -1, SUPER // GROUP)  # [rows, super-blocks, 8]
        # One fp16 `d` per 256 values, the super-block's largest group over 31: a super-block far below the
        # row's maximum keeps its own scale instead of sharing one sized for another block.
        d = (groups.amax(-1, keepdim=True) / 31).to(torch.float16).float()
        live = blocks.abs().amax(-1).reshape(rows.shape[0], -1, SUPER // GROUP) > 0
        safe = torch.where(d > 0, d, torch.ones_like(d))
        s6 = torch.where(live, torch.clamp(torch.round(groups / safe), 1, 31), torch.zeros_like(groups))
        scale = (d * s6).reshape(rows.shape[0], -1)[:, : c // GROUP]
        return scale.reshape(*w.shape[:-1], c // GROUP).to(w.device), None

    def fit_weighted(self, w: torch.Tensor, importance: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Native imatrix fit's exact fp16 times signed 6-bit scales, keeping the original row width."""
        records = weighted_records(w, importance, IQ4_XS, GROUP, SUPER, self.name)
        d, factors, _ = iq4xs_fields(records)
        scale = d.view("<f2").astype(np.float32)[:, None] * factors.astype(np.float32)
        c = w.shape[-1]
        scale = scale.reshape(-1, (c + -c % SUPER) // GROUP)[:, : c // GROUP].copy()
        return torch.from_numpy(scale).reshape(*w.shape[:-1], c // GROUP).to(w.device), None


def iq4nl_fields(records: bytes) -> tuple[np.ndarray, np.ndarray]:
    """``(fp16 d bits [n], codes [n, 32])`` of IQ4_NL records."""
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, 18)
    d = rec[:, :2].copy().view("<u2").reshape(-1)
    half = rec[:, 2:]
    codes = np.concatenate((half & 0x0F, half >> 4), axis=1).astype(np.uint8)
    return d, codes


def iq4nl_records(d: np.ndarray, codes: np.ndarray) -> bytes:
    """IQ4_NL records from ``fp16 d bits [n]`` and codes ``[n, 32]`` in ``0 .. 15``."""
    c = np.asarray(codes, dtype=np.uint8).reshape(-1, GROUP)
    half = (c[:, :16] | (c[:, 16:] << 4)).astype(np.uint8)
    return np.concatenate((np.asarray(d, dtype="<u2").view(np.uint8).reshape(-1, 2), half), axis=1).tobytes()


def iq4xs_fields(records: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fp16 d bits [n], signed scale [n, 8], codes [n, 256])`` of IQ4_XS records."""
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, 136)
    d = rec[:, :2].copy().view("<u2").reshape(-1)
    high = rec[:, 2:4].copy().view("<u2").reshape(-1)
    low = rec[:, 4:8]
    st = np.empty((rec.shape[0], 8), dtype=np.uint8)
    for j in range(4):
        st[:, 2 * j] = low[:, j] & 0x0F
        st[:, 2 * j + 1] = low[:, j] >> 4
    for i in range(8):
        st[:, i] |= ((high >> (2 * i)) & 0x03).astype(np.uint8) << 4
    scales = st.astype(np.int16) - 32
    q = rec[:, 8:].reshape(-1, 8, 16)
    codes = np.concatenate((q & 0x0F, q >> 4), axis=2).reshape(-1, SUPER).astype(np.uint8)
    return d, scales, codes


def iq4xs_records(d: np.ndarray, scales: np.ndarray, codes: np.ndarray) -> bytes:
    """IQ4_XS records from ``fp16 d bits [n]``, signed scales ``[n, 8]`` and codes ``[n, 256]``."""
    st = (np.asarray(scales, dtype=np.int16) + 32).astype(np.uint8)
    low = np.empty((st.shape[0], 4), dtype=np.uint8)
    for j in range(4):
        low[:, j] = (st[:, 2 * j] & 0x0F) | ((st[:, 2 * j + 1] & 0x0F) << 4)
    high = np.zeros(st.shape[0], dtype=np.uint16)
    for i in range(8):
        high |= (st[:, i] >> 4).astype(np.uint16) << (2 * i)
    c = np.asarray(codes, dtype=np.uint8).reshape(-1, 8, GROUP)
    q = (c[:, :, :16] | (c[:, :, 16:] << 4)).astype(np.uint8).reshape(-1, SUPER // 2)
    return np.concatenate(
        (
            np.asarray(d, dtype="<u2").view(np.uint8).reshape(-1, 2),
            high.view(np.uint8).reshape(-1, 2),
            low,
            q,
        ),
        axis=1,
    ).tobytes()


def _codes_int(qp: dict[str, Any], rows: torch.Tensor, block: int, name: str, top: int) -> np.ndarray:
    """The entry's codes ``[n, block]`` as ``uint8``, in template row order, checked against the format."""
    codes = qp["codes"].cpu()[rows]
    if codes.ndim != 2 or codes.shape[1] % block:
        raise Refused(f"gguf: {name}: {qp['format']} requires whole {block}-value blocks")
    if ((codes < 0) | (codes > top) | (codes != codes.round())).any():
        raise Refused(f"gguf: {name}: {qp['format']} codes are table indices 0 .. {top}")
    return codes.reshape(-1, block).to(torch.int64).numpy().astype(np.uint8)


def iq4nl_bytes(
    qp: dict[str, Any],
    rows: torch.Tensor,
    values: torch.Tensor,  # noqa: ARG001
    name: str,
    workers: int | None = None,  # noqa: ARG001
) -> bytes:
    """IQ4_NL records of one qparams entry: table indices and one exact fp16 scale per 32 values."""
    if qp["format"] != IQ4NL_FORMAT or qp.get("zero_point") is not None:
        raise Refused(f"gguf: {name}: IQ4_NL requires {IQ4NL_FORMAT} on grid 'iq4nl', with no zero point")
    codes = _codes_int(qp, rows, GROUP, name, 15)
    scale = qp["scale"].cpu().float()[rows].reshape(-1, 1).numpy()
    d = scale.astype(np.float16)
    if not np.array_equal(d.astype(np.float32), scale):
        raise Refused(f"gguf: {name}: IQ4_NL requires an exact fp16 scale per 32 values")
    return iq4nl_records(d.view("<u2"), codes)


def iq4xs_bytes(
    qp: dict[str, Any],
    rows: torch.Tensor,
    values: torch.Tensor,  # noqa: ARG001
    name: str,
    workers: int | None = None,  # noqa: ARG001
) -> bytes:
    """IQ4_XS records of one qparams entry: table indices and exact ``fp16(d) * s`` region scales."""
    if qp["format"] != IQ4XS_FORMAT or qp.get("zero_point") is not None:
        raise Refused(f"gguf: {name}: IQ4_XS requires {IQ4XS_FORMAT} on grid 'iq4xs', with no zero point")
    codes = _codes_int(qp, rows, SUPER, name, 15)
    products = qp["scale"].cpu().float()[rows].reshape(-1, 8).numpy()
    d, scales = split_6bit(products, name)
    named = codes.reshape(-1, 8, GROUP)
    named[products.reshape(-1, 8) == 0] = 0  # a zero scale decodes to zero whatever the codes are
    return iq4xs_records(d, scales, codes)


def quantize_iq4nl(x: torch.Tensor) -> bytes:
    """Round-to-nearest IQ4_NL records of ``x [rows, C]`` (the template-fit path and a test oracle)."""
    grid = GRIDS["iq4nl"]
    fmt = Format.parse(IQ4NL_FORMAT)
    scale, _ = grid.fit(fmt, x)
    wide = scale.reshape(x.shape[0], -1).repeat_interleave(GROUP, -1)
    codes = grid.codes(x.float(), wide, None, fmt.qmin, fmt.qmax)
    return iq4nl_records(scale.reshape(-1, 1).to(torch.float16).numpy().view("<u2"), codes.numpy())


def quantize_iq4xs(x: torch.Tensor) -> bytes:
    """Round-to-nearest IQ4_XS records of ``x [rows, C]`` with ``C`` a multiple of 256."""
    grid = GRIDS["iq4xs"]
    fmt = Format.parse(IQ4XS_FORMAT)
    scale, _ = grid.fit(fmt, x)
    wide = scale.reshape(x.shape[0], -1).repeat_interleave(GROUP, -1)
    codes = grid.codes(x.float(), wide, None, fmt.qmin, fmt.qmax)
    d, scales = split_6bit(scale.reshape(-1, 8).numpy(), "rtn")
    return iq4xs_records(d, scales, codes.numpy().astype(np.uint8))


IQ4NL_GRID = IQ4NL()
IQ4XS_GRID = IQ4XS()
