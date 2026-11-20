"""Q5_0 and Q8_0 as ``llama-quantize`` writes them: the types its Q4_K_M rule falls back to.

llama.cpp stores a tensor whose rows are not a whole number of 256-value super-blocks in a 32-value block type
instead (`rule`): Q5_0 where the rule wanted Q4_K, Q8_0 where it wanted Q6_K. These are ports of
``quantize_row_q5_0_ref`` and ``quantize_row_q8_0_ref`` (``ggml/src/ggml-quants.c``), the paths ``quantize_q5_0``
and ``quantize_q8_0`` take without an importance matrix. All arithmetic is fp32 in the C order; the codes use the
fp32 scale ``d``, while the record stores ``fp16(d)``.
"""

from __future__ import annotations

import numpy as np
import torch

QK = 32
Q5_0_BLOCK = 22  #: fp16 d, 4 bytes of high bits, 16 bytes of nibbles
Q8_0_BLOCK = 34  #: fp16 d, 32 int8 codes
_F32 = np.float32


def _blocks(x: torch.Tensor) -> np.ndarray:
    """``[rows, width]`` (width a multiple of 32) as fp32 ``(n, 32)`` blocks."""
    if x.shape[-1] % QK:
        raise ValueError(f"width must be a multiple of {QK}, got {x.shape[-1]}")
    return x.detach().to("cpu", torch.float32).contiguous().numpy().reshape(-1, QK)


def _first_max_abs(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per block: the absolute maximum and the value that first exceeds the running one.

    This is C's ``if (amax < fabsf(v))`` starting from ``amax = max = 0.0f``.

    A block of zeros never updates ``max``, so it stays ``+0.0`` even when the first value is ``-0.0``.
    """
    a = np.abs(b)
    i = np.argmax(a, axis=-1)  # the first index of the maximum, as the strict ``<`` keeps
    amax = np.take_along_axis(a, i[:, None], -1)[:, 0]
    mx = np.take_along_axis(b, i[:, None], -1)[:, 0]
    return amax, np.where(amax > 0, mx, _F32(0.0))


def _inv(d: np.ndarray) -> np.ndarray:
    """``d ? 1.0f / d : 0.0f``."""
    return np.where(d != 0, _F32(1.0) / d, _F32(0.0)).astype(_F32)


def _to_int8(v: np.ndarray) -> np.ndarray:
    """C's ``(int8_t)`` of fp32 ``v`` as x86 does it: truncate, and 0 where ``v`` is not finite.

    ``1.0f / d`` overflows to inf when ``|d|`` is below about 2.94e-39, i.e. a block's largest magnitude is at most
    3.732e-37 (Q8_0) or 4.702e-38 (Q5_0), so a block of tiny weights gives ``±inf`` or ``nan`` here.
    C leaves that cast undefined: x86 writes 0, aarch64 writes 0xff for ``+inf``. We write 0 on every arch.
    """
    return np.trunc(np.where(np.isfinite(v), v, _F32(0))).astype(np.int8)


def _roundf(v: np.ndarray) -> np.ndarray:
    """C's ``roundf``: half away from zero, exactly. ``floor(|v| + 0.5)`` in fp32 rounds 0.49999997 up to 1."""
    t = np.trunc(v)
    return t + np.where(np.abs(v - t) >= _F32(0.5), np.sign(v), _F32(0)).astype(_F32)


def _f16_bytes(d: np.ndarray) -> np.ndarray:
    """``GGML_FP32_TO_FP16`` as two little-endian bytes per value."""
    return d.astype("<f2").view(np.uint8).reshape(-1, 2)


@np.errstate(all="ignore")
def quantize_q8_0(x: torch.Tensor) -> bytes:
    """Q8_0 records of ``x [rows, width]``: ``d = amax / 127``, codes ``roundf(x / d)``."""
    b = _blocks(x)
    amax, _ = _first_max_abs(b)
    d = (amax / _F32(127)).astype(_F32)
    v = b * _inv(d)[:, None]
    q = _to_int8(_roundf(v))
    out = np.empty((b.shape[0], Q8_0_BLOCK), np.uint8)
    out[:, :2] = _f16_bytes(d)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


@np.errstate(all="ignore")
def quantize_q5_0(x: torch.Tensor) -> bytes:
    """Q5_0 records of ``x [rows, width]``.

    ``d = max / -16`` with ``max`` the signed value of largest magnitude, codes ``min(31, (int8)(x / d + 16.5))``,
    the fifth bits gathered into ``qh``.
    """
    b = _blocks(x)
    _, mx = _first_max_abs(b)
    d = (mx / _F32(-16)).astype(_F32)
    v = b * _inv(d)[:, None] + _F32(16.5)
    q = np.minimum(31, _to_int8(v).view(np.uint8)).astype(np.uint32)
    lo, hi = q[:, :16], q[:, 16:]
    qs = ((lo & 0xF) | ((hi & 0xF) << 4)).astype(np.uint8)
    bit = np.arange(16, dtype=np.uint32)
    qh = (((lo >> 4) & 1) << bit).sum(-1) | (((hi >> 4) & 1) << (bit + 16)).sum(-1)
    out = np.empty((b.shape[0], Q5_0_BLOCK), np.uint8)
    out[:, :2] = _f16_bytes(d)
    out[:, 2:6] = qh.astype("<u4").view(np.uint8).reshape(-1, 4)
    out[:, 6:] = qs
    return out.tobytes()
