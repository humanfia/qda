"""GPTQ's column loop inside one block as one Triton kernel (fast path of `solve._block_torch`).

Rows are independent inside a block (they share only the factor U), so a program owns a tile of rows
for the whole block and walks its columns in registers: one launch per block instead of ~10 per column.
The arithmetic is the torch loop's, operation for operation, with IEEE round-to-nearest and no fused
multiply-add (``div_rn``, ``mul_rn``, ``sub_rn``; ``rint`` is torch.round's half-to-even), and the served value is
rounded to bf16 for WNA16 kernels. Offset grids (Q4_K/Q6_K) retain their fp32 served values instead;
both paths preserve the reference's codes (``tests/test_solve.py``, ``tests/gguf/test_fused_gptq.py``, GPU).

``fp8_token`` is the other hot loop's kernel: the W4A8 activation round trip (`solve.quant.fp8_token_reference`)
was a chain of elementwise passes over fp32 intermediates; here one program owns a row, reads it once for its
amax and once to quantize-dequantize it, and stores it in its own dtype.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdevice

# libdevice's stubs end in ``...`` (so they read as returning None) and ``static_range.__iter__`` raises
# outside a ``@triton.jit`` body: the Triton compiler, not a Python type checker, resolves both. The casts
# are no-ops at runtime and keep the names the kernel body reads from its module globals.
libdevice = cast(Any, _libdevice)
_static_range = cast(Any, tl.static_range)


@triton.jit
def _block(
    W, S, Z, F8, CH, U, Q, ERR, R, i1, nb,
    sWe, sWr, sUe, sUr, sQe, sQr, sEe, sEr, sCe,
    qmin, qmax, HAS_Z: tl.constexpr, FP8: tl.constexpr, OFFSET_GRID: tl.constexpr,
    BR: tl.constexpr, BC: tl.constexpr,
):  # fmt: skip
    # int64: a bank's `e * R * C` elements reach 2^31 at 256 experts of 2048 x 4096 weights, and an int32
    # offset wraps onto another expert's rows without raising. Every product that can grow
    # is widened before it is computed: a cast added after the multiply cannot undo a wrapped product.
    e = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1).to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    cols = tl.arange(0, BC)
    m = (rows < R)[:, None] & (cols < nb)[None, :]
    off = e * sWe + rows[:, None] * sWr + (i1 + cols)[None, :]
    wt = tl.load(W + off, mask=m, other=0.0)
    st = tl.load(S + off, mask=m, other=1.0)
    if HAS_Z:
        zt = tl.load(Z + off, mask=m, other=0.0)
    if FP8:  # CUTLASS W4A8: per-column e4m3 group factor, per-row fp32 channel factor
        ft = tl.load(F8 + off, mask=m, other=1.0)
        ch = tl.load(CH + e * sCe + rows, mask=rows < R, other=1.0)
    qt = tl.zeros((BR, BC), dtype=tl.float32)
    et = tl.zeros((BR, BC), dtype=tl.float32)
    ub = U + e * sUe
    # Column selection follows llm-compressor 0.15's GPTQ kernel; attributed conservatively (origin uncertain).
    for i in range(0, nb):
        # the factor's row index reaches C^2 (the last live offset of `U [E, C, C]`): widen both operands
        # before the product, or C > 46340 wraps it before it ever reaches the int64 expert base
        row_u = (i1 + i).to(tl.int64) * sUr.to(tl.int64)
        sel = cols[None, :] == i
        w = tl.sum(tl.where(sel, wt, 0.0), axis=1)
        s = tl.sum(tl.where(sel, st, 0.0), axis=1)
        if OFFSET_GRID:
            numerator = w
            if HAS_Z:
                z = tl.sum(tl.where(sel, zt, 0.0), axis=1)
                numerator = libdevice.add_rn(w, z)
            q = libdevice.rint(libdevice.div_rn(numerator, s))
            # torch.clamp keeps a rounded -0.0 at lower bound 0; maximum(-0.0, 0.0) does not.
            q = tl.where(q < qmin, qmin, q)
            q = tl.where(q > qmax, qmax, q)
            q = tl.where(s != 0.0, q, 0.0)
            wq = libdevice.mul_rn(q, s)
            if HAS_Z:
                wq = libdevice.sub_rn(wq, z)
        else:
            q = libdevice.rint(libdevice.div_rn(w, s))
        if not OFFSET_GRID and FP8:  # nearest served e4m3(q*f8)*chan among q0-1, q0, q0+1
            f = tl.sum(tl.where(sel, ft, 0.0), axis=1)
            q = tl.minimum(tl.maximum(q, qmin), qmax)
            wq = libdevice.mul_rn(
                libdevice.mul_rn(q, f).to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.float32), ch
            )
            be = tl.abs(libdevice.sub_rn(w, wq))
            for dlt in _static_range(2):
                qc = tl.minimum(tl.maximum(libdevice.add_rn(q, 2.0 * dlt - 1.0), qmin), qmax)
                vc = libdevice.mul_rn(
                    libdevice.mul_rn(qc, f).to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.float32), ch
                )
                ec = tl.abs(libdevice.sub_rn(w, vc))
                better = ec < be
                q = tl.where(better, qc, q)
                wq = tl.where(better, vc, wq)
                be = tl.where(better, ec, be)
        elif not OFFSET_GRID and HAS_Z:
            z = tl.sum(tl.where(sel, zt, 0.0), axis=1)
            q = tl.minimum(tl.maximum(libdevice.add_rn(q, z), qmin), qmax)
            wq = libdevice.mul_rn(libdevice.sub_rn(q, z), s).to(tl.bfloat16).to(tl.float32)
        elif not OFFSET_GRID:
            q = tl.minimum(tl.maximum(q, qmin), qmax)
            wq = libdevice.mul_rn(q, s).to(tl.bfloat16).to(tl.float32)
        d = tl.load(ub + row_u + (i1 + i))
        err = libdevice.div_rn(libdevice.sub_rn(w, wq), d)
        u = tl.load(ub + row_u + (i1 + cols), mask=(cols < nb) & (cols >= i), other=0.0)
        wt = tl.where(
            (cols >= i)[None, :], libdevice.sub_rn(wt, libdevice.mul_rn(err[:, None], u[None, :])), wt
        )
        qt = tl.where(sel, q[:, None], qt)
        et = tl.where(sel, err[:, None], et)
    tl.store(Q + e * sQe + rows[:, None] * sQr + (i1 + cols)[None, :], qt, mask=m)
    tl.store(ERR + e * sEe + rows[:, None] * sEr + cols[None, :], et, mask=m)


def gptq_block(
    w: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor | None,
    fp8: tuple[torch.Tensor, torch.Tensor] | None,
    u: torch.Tensor,
    codes: torch.Tensor,
    i1: int,
    i2: int,
    qmin: int,
    qmax: int,
    br: int = 16,
    offset_grid: bool = False,
) -> torch.Tensor:
    """Columns ``i1:i2`` for every ``[E, R]`` row: writes ``codes[..., i1:i2]``, returns the scaled errors."""
    e, r, _ = w.shape
    nb = i2 - i1
    err = torch.empty(e, r, nb, dtype=torch.float32, device=w.device)
    if fp8 is not None:
        f8, ch = fp8[0], fp8[1].reshape(e, r).contiguous()
    else:  # unused by the kernel (FP8=False): any valid pointer
        f8 = ch = w
    _block[(e, triton.cdiv(r, br))](
        w, s, z if z is not None else w, f8, ch, u, codes, err, r, i1, nb,
        w.stride(0), w.stride(1), u.stride(0), u.stride(1),
        codes.stride(0), codes.stride(1), err.stride(0), err.stride(1), ch.stride(0),
        float(qmin), float(qmax), HAS_Z=z is not None, FP8=fp8 is not None, OFFSET_GRID=offset_grid,
        BR=br, BC=triton.next_power_of_2(max(nb, 16)),
        num_warps=4,  # pyright: ignore[reportCallIssue] triton launcher kwarg
    )  # fmt: skip
    return err


@triton.jit
def _fp8_token(
    X, Y, N, sXr, sXc, sYr, sYc,
    FP8_MAX: tl.constexpr, MIN_SCALE: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    # int64: a batch's rows can pass 2^31 elements, and an int32 `row * sXr` wraps onto an earlier row
    # without raising, reading and writing the wrong memory. `cols * sXc` stays in range.
    row = tl.program_id(0).to(tl.int64)
    amax = 0.0
    for c0 in range(0, N, BN):
        cols = c0 + tl.arange(0, BN)
        m = cols < N
        x = tl.load(X + row * sXr + cols * sXc, mask=m, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    # The reference's ``amax.float() / FP8_MAX`` divides by a Python scalar, which torch's CUDA kernel
    # implements as a multiply by the fp32 reciprocal of 448 (``div_true_kernel_cuda``); that product is
    # one ulp above the IEEE quotient for some amaxes, and the reference's bits are that product. So this
    # is ``mul_rn``, not ``div_rn``: a build must not change when its activation path is fused.
    s = tl.maximum(libdevice.mul_rn(amax, 1.0 / FP8_MAX), MIN_SCALE)
    r = libdevice.div_rn(1.0, s)
    for c0 in range(0, N, BN):
        cols = c0 + tl.arange(0, BN)
        m = cols < N
        x = tl.load(X + row * sXr + cols * sXc, mask=m, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(libdevice.mul_rn(x, r), -FP8_MAX), FP8_MAX)
        q8 = q.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.float32)
        tl.store(Y + row * sYr + cols * sYc, libdevice.mul_rn(q8, s), mask=m)


def fp8_token(x: torch.Tensor) -> torch.Tensor:
    """The W4A8 activation round trip of `solve.quant.fp8_token_reference`, one pass per row.

    ``x`` is ``[..., C]``; each ``[C]`` row is read once for its amax and once to quantize and dequantize it,
    and the result is stored in x's dtype. The arithmetic is the reference's operation for operation, with IEEE
    round-to-nearest division and multiply and the same RTNE casts, so values are bit-identical
    (``tests/test_fp8_token.py``, GPU). The input must be CUDA bf16 or fp16; `solve.quant.fp8_token` dispatches.
    """
    from .quant import FP8_MAX, FP8_MIN_SCALE  # the constants live with the reference

    n = x.shape[-1]
    rows = x.numel() // n if n else 0
    x2 = x.reshape(rows, n)
    x2 = x2 if x2.is_contiguous() else x2.contiguous()
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if rows:
        bn = min(triton.next_power_of_2(n), 4096)
        _fp8_token[(rows,)](
            x2, out, n, x2.stride(0), x2.stride(1), out.stride(0), out.stride(1),
            FP8_MAX=FP8_MAX, MIN_SCALE=FP8_MIN_SCALE, BN=bn,
            num_warps=8 if bn >= 2048 else 4,  # pyright: ignore[reportCallIssue]
        )  # fmt: skip
    return out.view(x.shape)
