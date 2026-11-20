"""Recover ggml's fp16 ``d`` and signed 6-bit scale from the exact products a grid fits.

IQ4_XS and Q3_K store a group's scale as ``fp16(d) * s`` with ``s`` a signed 6-bit integer, and the product is
exact in fp32 (an 11-bit significand times a 6-bit integer needs 16 bits). A solved grid keeps the product and
not the pair, so the writer has to recover a pair the codes were solved against. Several pairs can give the same
products (``d, s`` and ``2d, 2s``): every one of them decodes to the same weights, so any exact one is the
writer's answer, taken in a fixed order so the bytes do not depend on the search's mood.
"""

from __future__ import annotations

import numpy as np
import torch

from ..step import Refused
from .file import row_bytes
from .kernel import quantize_weighted_chunk

#: The 6-bit scale's range, in the stored form ``s`` (bits 0 .. 63 hold ``s + 32``).
SCALE_MIN, SCALE_MAX = -32, 31


def weighted_records(
    w: torch.Tensor, importance: torch.Tensor, kind: int, group: int, block: int, name: str
) -> bytes:
    """Native Hessian-weighted records; zero padding only stands in for partial blocks during calibration."""
    c = w.shape[-1]
    if c % group:
        raise Refused(f"grid {name!r}: a row of {c} values is not a whole number of {group}-value groups")
    rows = w.reshape(-1, c).float()
    pad = -c % block
    rows = torch.nn.functional.pad(rows, (0, pad))
    imp = torch.nn.functional.pad(importance.reshape(-1, c), (0, pad))
    n = rows.shape[0] // imp.shape[0]
    try:
        return b"".join(
            quantize_weighted_chunk(kind, rows[i * n : (i + 1) * n], row_bytes(kind, rows.shape[-1]), imp[i])
            for i in range(imp.shape[0])
        )
    except ValueError as e:
        raise Refused(
            f"grid {name!r} needs native ggml_quantize_chunk; set exporter.gguf.libggml: {e}"
        ) from e


def split_6bit(products: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(fp16 d bits [n], signed scale [n, k])`` of exact products ``fp16(d) * s``.

    Args:
        products: ``[n, k]`` fp32 group products, each exactly ``fp16(d) * s`` for one block's ``d``.
        name: tensor name, for the refusal.

    Returns:
        ``d`` as ``uint16`` fp16 bits per block (0 for a block of zeros) and ``s`` as ``int16`` per group.

    Raises:
        Refused: a block's products are not one fp16 scale times integers in the 6-bit range.
    """
    p = np.ascontiguousarray(products, dtype=np.float32)
    n, k = p.shape
    d = np.zeros(n, dtype="<u2")
    s = np.zeros((n, k), dtype=np.int16)
    live = np.flatnonzero(p.any(axis=1))
    if not len(live):
        return d, s
    p = p[live]
    # The product with the largest magnitude is the block's reference: it carries the group the fit made
    # largest, so its scale is the one the fp16 rounding of d was chosen against.
    ref = p[np.arange(len(p)), np.abs(p).argmax(axis=1)]
    chosen = np.zeros(len(p), dtype=bool)
    for cand in range(SCALE_MIN, SCALE_MAX + 1):
        if cand == 0:
            continue
        d16 = (ref / np.float32(cand)).astype(np.float16)
        d32 = d16.astype(np.float32)
        todo = np.flatnonzero(~chosen & np.isfinite(d32) & (d32 != 0))
        if not len(todo):
            continue
        q = p[todo] / d32[todo][:, None]
        qr = np.round(q)
        fits = (q == qr) & (qr >= SCALE_MIN) & (qr <= SCALE_MAX)
        allfit = fits.all(axis=1)
        good = todo[allfit]
        d[live[good]] = d16[good].view("<u2")
        s[live[good]] = qr[allfit].astype(np.int16)
        chosen[good] = True
    if not chosen.all():
        bad = int(np.flatnonzero(~chosen)[0])
        raise Refused(
            f"gguf: {name}: a block's group products are not one fp16 scale times 6-bit integers "
            f"(first product {float(products[live[bad], 0]):.9g}); use the grid that fitted them"
        )
    return d, s


def pack_q3k_scales(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(low nibbles [n, 8], high 2 bits [n, 4])`` of Q3_K's signed 6-bit scales ``[n, 16]``.

    Byte ``j`` of the low part holds scale ``j`` in its low nibble and scale ``j + 8`` in its high nibble
    (``IIIIAAAA``); byte ``j`` of the high part holds the top two bits of scales ``j, j + 4, j + 8, j + 12``,
    two bits each (``MMIIEEAA``).
    """
    st = (np.asarray(scales, dtype=np.int16) + 32).astype(np.uint8)
    low = (st[:, :8] & 0x0F) | ((st[:, 8:] & 0x0F) << 4)
    high = np.zeros((st.shape[0], 4), dtype=np.uint8)
    for j in range(4):
        for h in range(4):
            high[:, j] |= (st[:, 4 * h + j] >> 4) << (2 * h)
    return low, high


def unpack_q3k_scales(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Q3_K's signed 6-bit scales ``[n, 16]`` from `pack_q3k_scales`' two parts."""
    st = np.empty((low.shape[0], 16), dtype=np.uint8)
    st[:, :8] = low & 0x0F
    st[:, 8:] = low >> 4
    for j in range(4):
        for h in range(4):
            st[:, 4 * h + j] |= (high[:, j] >> (2 * h) & 0x03) << 4
    return st.astype(np.int16) - 32
