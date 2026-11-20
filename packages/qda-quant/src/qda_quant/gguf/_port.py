"""The numpy port of llama.cpp's Q4_K and Q6_K reference quantizers.

``quantize_row_q4_K_ref`` / ``quantize_row_q6_K_ref`` (``ggml/src/ggml-quants.c``) ported to numpy, with their
helpers ``make_qkx2_quants`` and ``make_qx_quants``. Every intermediate is fp32 in C's order, because the last bit
of a code depends on it; super-blocks are independent, so batching them changes nothing. Byte identity was
checked against llama.cpp ``9e2a04a``; for non-finite inputs (undefined in C) the port reproduces what x86
produced in our parity checks.
"""

from __future__ import annotations

import numpy as np

QK_K = 256

_Q4K_BLOCK = 144  # 2 + 2 + 12 + 128 bytes

_Q6K_BLOCK = 210  # 128 + 64 + 16 + 2 bytes

_EPS = np.float32(1e-15)  # GROUP_MAX_EPS

_ROUND_MAGIC = np.float32(12582912.0)  # 0x1.8p23: nearest_int's rounding bias

_NAN_BITS = 0x7E00  # what ggml's fp32_to_fp16 returns for NaN

_F32 = np.float32

# ---------------------------------------------------------------------------------------------------------
# Shared primitives


def _nearest_int(v: np.ndarray | np.float32) -> np.ndarray | np.int32:
    """Port of ggml's ``nearest_int``: round half to even through the fp32 magic constant.

    The addition rounds to the nearest representable fp32, whose low mantissa bits are then the integer.
    """
    bits = (v + _ROUND_MAGIC).view(np.int32)
    return (bits & 0x007FFFFF) - 0x00400000


def _fp16_bits(v: np.ndarray | np.float32) -> np.ndarray:
    """fp32 -> fp16 bits, as ``GGML_FP32_TO_FP16``: round to nearest even, NaN canonicalized (sign kept)."""
    arr = np.asarray(v, dtype=np.float32)
    bits = arr.astype(np.float16).view(np.uint16)
    nan = np.isnan(arr)
    if np.any(nan):
        bits = np.where(nan, (bits & np.uint16(0x8000)) | np.uint16(_NAN_BITS), bits).astype(np.uint16)
    return bits


def _fp16_to_fp32(bits: np.ndarray) -> np.ndarray:
    """fp16 bits -> fp32, as ``GGML_FP16_TO_FP32``."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# Fast paths: the same arithmetic, batched across super-blocks


def _seq_sum(a: np.ndarray) -> np.ndarray:
    """Sum over the last axis left to right, as a C ``for`` loop does (no pairwise tree).

    ``np.cumsum`` is a sequential running sum (not a pairwise reduction), so its last entry is the C sum.
    """
    return np.cumsum(a, axis=-1, dtype=np.float32)[..., -1]


def _seq_min(a: np.ndarray) -> np.ndarray:
    """Elementwise minimum of the last axis with C's ``if (x < min) min = x`` (NaN never wins)."""
    acc = a[..., 0].copy()
    for i in range(1, a.shape[-1]):
        acc = np.where(a[..., i] < acc, a[..., i], acc)
    return acc


def _seq_max(a: np.ndarray) -> np.ndarray:
    """Elementwise maximum of the last axis with C's ``if (x > max) max = x`` (NaN never wins)."""
    acc = a[..., 0].copy()
    for i in range(1, a.shape[-1]):
        acc = np.where(a[..., i] > acc, a[..., i], acc)
    return acc


def _scan_max_zero(a: np.ndarray) -> np.ndarray:
    """Max over the last axis starting from ``0``, as ``float max = 0; if (x > max) ...``."""
    acc = np.zeros(a.shape[:-1], dtype=np.float32)
    for i in range(a.shape[-1]):
        acc = np.where(a[..., i] > acc, a[..., i], acc)
    return acc


def _make_qkx2_fast(
    x: np.ndarray, weights: np.ndarray, nmax: int, rmin: np.float32, rdelta: np.float32, nstep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``make_qkx2_quants`` over any leading axes; returns ``(scale, -min, codes)``."""
    mn = _seq_min(x)
    mx = _seq_max(x)
    sum_w = _seq_sum(weights)
    sum_x = _seq_sum(weights * x)
    mn = np.where(mn > 0, _F32(0.0), mn)
    degenerate = mx == mn
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iscale = _F32(nmax) / (mx - mn)
        scale = _F32(1.0) / iscale
        L = np.clip(_nearest_int(iscale[..., None] * (x - mn[..., None])), 0, nmax)
        diff = scale[..., None] * L.astype(np.float32) + mn[..., None] - x
        best_error = _seq_sum(weights * (diff * diff))
        if nstep >= 1:
            for is_ in range(nstep + 1):
                isca = (rmin + rdelta * _F32(is_) + _F32(nmax)) / (mx - mn)
                la = np.clip(_nearest_int(isca[..., None] * (x - mn[..., None])), 0, nmax)
                laf = la.astype(np.float32)
                sum_l = _seq_sum(weights * laf)
                sum_l2 = _seq_sum(weights * laf * laf)
                sum_xl = _seq_sum(weights * laf * x)
                D = sum_w * sum_l2 - sum_l * sum_l
                valid = D > 0
                this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
                this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
                fix = this_min > 0
                this_min = np.where(fix, _F32(0.0), this_min)
                this_scale = np.where(fix, sum_xl / sum_l2, this_scale)
                diff = this_scale[..., None] * laf + this_min[..., None] - x
                cur_error = _seq_sum(weights * (diff * diff))
                better = valid & (cur_error < best_error)
                L = np.where(better[..., None], la, L)
                best_error = np.where(better, cur_error, best_error)
                scale = np.where(better, this_scale, scale)
                mn = np.where(better, this_min, mn)
    L = np.where(degenerate[..., None], 0, L)
    scale = np.where(degenerate, _F32(0.0), scale)
    return scale, -mn, L


def _make_qx_fast(x: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``make_qx_quants`` (``rmse_type=1``, ``qw=NULL``); returns ``(scale, codes)``."""
    mx = np.zeros(x.shape[:-1], dtype=np.float32)
    amax = np.zeros(x.shape[:-1], dtype=np.float32)
    for i in range(x.shape[-1]):
        ax = np.abs(x[..., i])
        better = ax > amax
        amax = np.where(better, ax, amax)
        mx = np.where(better, x[..., i], mx)
    zero = amax < _EPS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iscale = -_F32(nmax) / mx
        code = np.clip(_nearest_int(iscale[..., None] * x), -nmax, nmax - 1)
        L = code + nmax
        lf = code.astype(np.float32)
        w = x * x
        sumlx = _seq_sum(w * x * lf)
        suml2 = _seq_sum(w * lf * lf)
        scale = np.where(suml2 != 0, sumlx / suml2, _F32(0.0))
        best = scale * sumlx
        for is_ in range(-9, 10):
            if is_ == 0:
                continue
            isca = -(_F32(nmax) + _F32(0.1) * _F32(is_)) / mx
            la = np.clip(_nearest_int(isca[..., None] * x), -nmax, nmax - 1)
            laf = la.astype(np.float32)
            sumlx = _seq_sum(w * x * laf)
            suml2 = _seq_sum(w * laf * laf)
            better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
            L = np.where(better[..., None], la + nmax, L)
            scale = np.where(better, sumlx / suml2, scale)
            best = np.where(better, scale * sumlx, best)
    L = np.where(zero[..., None], 0, L)
    scale = np.where(zero, _F32(0.0), scale)
    return scale, L


def _q4k_scale_bytes(ls: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Pack ``(n, 8)`` 6-bit scales and minimums into ``(n, 12)`` Q4_K scale bytes."""
    n = ls.shape[0]
    sb = np.zeros((n, 12), dtype=np.uint8)
    for j in range(4):
        sb[:, j] = ls[:, j]
        sb[:, j + 4] = lm[:, j]
    for j in range(4, 8):
        sb[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).astype(np.uint8)
        sb[:, j - 4] |= (((ls[:, j] >> 4) << 6) & 0xFF).astype(np.uint8)
        sb[:, j] |= (((lm[:, j] >> 4) << 6) & 0xFF).astype(np.uint8)
    return sb


def _q4k_scale_values(sb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 12)`` Q4_K scale bytes -> 6-bit scales and minimums ``(n, 8)``, as ``get_scale_min_k4``."""
    n = sb.shape[0]
    sc = np.empty((n, 8), dtype=np.uint8)
    mm = np.empty((n, 8), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = sb[:, j] & 63
            mm[:, j] = sb[:, j + 4] & 63
        else:
            sc[:, j] = (sb[:, j + 4] & 0xF) | ((sb[:, j - 4] >> 6) << 4)
            mm[:, j] = (sb[:, j + 4] >> 4) | ((sb[:, j] >> 6) << 4)
    return sc, mm


def _q4k_nibbles(L: np.ndarray) -> np.ndarray:
    """Codes ``(n, 256)`` (any shape with 256 per super-block) -> ``(n, 128)`` packed nibbles."""
    q = L.reshape(-1, 4, 2, 32)
    return (q[:, :, 0, :] | (q[:, :, 1, :] << 4)).astype(np.uint8).reshape(-1, 128)


def _pack_q4k(d: np.ndarray, dmin: np.ndarray, sb: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Lay out Q4_K super-block fields as ``n`` contiguous 144-byte records."""
    n = d.shape[0]
    out = np.empty((n, _Q4K_BLOCK), dtype=np.uint8)
    out[:, 0:2] = d.astype("<u2").view(np.uint8).reshape(n, 2)
    out[:, 2:4] = dmin.astype("<u2").view(np.uint8).reshape(n, 2)
    out[:, 4:16] = sb
    out[:, 16:144] = qs
    return out


def _q4k_records(blocks: np.ndarray) -> np.ndarray:
    """Quantize an ``(n, 256)`` fp32 array to ``(n, 144)`` Q4_K records."""
    n = blocks.shape[0]
    x = blocks.reshape(n, 8, 32)
    sum_x2 = _seq_sum(x * x)
    av_x = np.sqrt(sum_x2 / _F32(32.0))
    weights = av_x[..., None] + np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scales, mins, L = _make_qkx2_fast(x, weights, 15, _F32(-1.0), _F32(0.1), 20)
        max_scale = _scan_max_zero(scales)
        max_min = _scan_max_zero(mins)
        inv_scale = np.where(max_scale > 0, _F32(63.0) / max_scale, _F32(0.0))
        inv_min = np.where(max_min > 0, _F32(63.0) / max_min, _F32(0.0))
        ls = np.minimum(63, _nearest_int(inv_scale[:, None] * scales) & 0xFF)
        lm = np.minimum(63, _nearest_int(inv_min[:, None] * mins) & 0xFF)
        d = _fp16_bits(max_scale / _F32(63.0))
        dmin = _fp16_bits(max_min / _F32(63.0))

        sb = _q4k_scale_bytes(ls, lm)
        sc, mm = _q4k_scale_values(sb)

        dd = _fp16_to_fp32(d)[:, None] * sc.astype(np.float32)
        dm = _fp16_to_fp32(dmin)[:, None] * mm.astype(np.float32)
        live = dd != 0
        code = np.clip(_nearest_int((x + dm[..., None]) / dd[..., None]), 0, 15)
        L = np.where(live[..., None], code, L)

    return _pack_q4k(d, dmin, sb, _q4k_nibbles(L))


def _pack_q6k(ql: np.ndarray, qh: np.ndarray, sbytes: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Lay out Q6_K super-block fields as ``n`` contiguous 210-byte records."""
    n = d.shape[0]
    out = np.empty((n, _Q6K_BLOCK), dtype=np.uint8)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = sbytes
    out[:, 208:210] = d.astype("<u2").view(np.uint8).reshape(n, 2)
    return out


def _q6k_code_bytes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleave stored ``0 .. 63`` codes into Q6_K's low nibbles and high bit pairs."""
    n = codes.shape[0]
    quarters = codes.reshape(n, 2, 4, 32)
    q1, q2, q3, q4 = (quarters[:, :, i] for i in range(4))
    ql = np.concatenate([(q1 & 15) | ((q3 & 15) << 4), (q2 & 15) | ((q4 & 15) << 4)], axis=2)
    qh = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    return ql.astype(np.uint8).reshape(n, 128), qh.astype(np.uint8).reshape(n, 64)


def _q6k_records(blocks: np.ndarray) -> np.ndarray:
    """Quantize an ``(n, 256)`` fp32 array to ``(n, 210)`` Q6_K records."""
    n = blocks.shape[0]
    x = blocks.reshape(n, 16, 16)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scales, L = _make_qx_fast(x, 32)
        max_scale = np.zeros(n, dtype=np.float32)
        max_abs_scale = np.zeros(n, dtype=np.float32)
        for ib in range(16):
            s = scales[:, ib]
            a = np.abs(s)
            better = a > max_abs_scale
            max_abs_scale = np.where(better, a, max_abs_scale)
            max_scale = np.where(better, s, max_scale)
        zero_block = max_abs_scale < _EPS

        iscale = -_F32(128.0) / max_scale
        d = _fp16_bits(_F32(1.0) / iscale)
        sb = np.minimum(127, _nearest_int(iscale[:, None] * scales)) & 0xFF
        sv = np.where(sb >= 128, sb - 256, sb).astype(np.float32)
        dd = _fp16_to_fp32(d)[:, None] * sv
        live = dd != 0
        code = np.clip(_nearest_int(x / dd[..., None]), -32, 31) + 32
        L = np.where(live[..., None], code, L)

    ql, qh = _q6k_code_bytes(L)
    out = _pack_q6k(ql, qh, sb.astype(np.uint8), d)
    out[zero_block] = 0
    return out
