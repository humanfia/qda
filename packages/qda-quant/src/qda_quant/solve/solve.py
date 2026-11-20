"""The GPTQ solve over a stack of independent problems: ``W [E, R, C]`` with Hessians ``H [E, C, C]``.

A dense group is ``E = 1`` with the rows of every module that reads the same input stacked (q/k/v share
one Cholesky); a routed-expert bank is ``E = experts``. Group scales are fixed before the solve (static
groups, as llm-compressor 0.15 does), so each code is rounded against the scale the kernel serves.

Two implementations of the column loop inside a block, with the same arithmetic: `_block_torch` (the
reference) and `_block_triton` (one program per row tile, columns walked in registers, IEEE ``*_rn``
ops without FMA). `tests/test_solve.py` checks they give identical codes.

Every expert's solve is independent, so a bank too large for one call is solved in chunks of experts
(`chunk_experts`); the codes do not depend on where the chunks fall. The one batch-size effect is in
`_factor`, which pads a short expert-bank batch so that every factorization is the whole bank's.

Both implementations address ``[E, R, C]`` and the factor ``U [E, C, C]`` with int64 offsets. `chunk_experts`
and `engine.planning` still keep one call inside `INT32_ELEMENTS` elements as a deliberate safety bound: it costs
chunks, it guards no live access. `solve_offsets` reports every solve's geometry for `build.json`.
"""

from __future__ import annotations

from typing import Any

import torch

from .quant import Quantizer, fp8_scale_parts, round_fp8, triton_fp8, widen

#: The element count one solve stays inside. The Triton kernel's offsets are int64, so this is a
#: belt-and-braces bound, not a wrap point: at exactly `2**31` elements the last live offset is `2**31 - 1`,
#: which fits, and only `e * r * c > 2**31` moves an expert past it (`256 * 4096 * 2048 == 2**31`). A bank is
#: solved in chunks that stay inside it; a dense group (`e = 1`) or a factor with more than `2**31` elements
#: (`cols > 46340`) is refused by the planner.
INT32_ELEMENTS = 2**31

#: Every solve this process ran since `solve_offsets()` last cleared it; `execute` keeps the largest offsets
#: in `build.json`, so an offset audit is answered from the artifact instead of a re-run.
_SOLVES: list[dict[str, Any]] = []


def solve_offsets() -> list[dict[str, Any]]:
    """Take the solves recorded since the last call.

    Each entry is ``label``, ``impl``, ``experts``, ``rows``, ``cols``, ``strides`` and the largest element
    offset the kernel computes over the weight stack (`max_offset`), and ``u_strides`` with the largest
    per-expert offset inside the factor (`u_max_offset`): the two paths a solve addresses.
    """
    out, _SOLVES[:] = list(_SOLVES), []
    return out


def experts_per_solve(elements_per_expert: int) -> int:
    """The most experts one solve may stack and stay inside `INT32_ELEMENTS` elements.

    A deliberate safety bound, not the point where an offset wraps: `INT32_ELEMENTS` experts fit
    exactly (their last live offset is `INT32_ELEMENTS - 1`), the next one starts at the limit.
    """
    return max(1, INT32_ELEMENTS // max(elements_per_expert, 1))


def prepare_hessian(h: torch.Tensor, n_tokens: torch.Tensor | int | None = None) -> torch.Tensor:
    """fp32 ``[E, C, C]`` copy of a raw Hessian sum, averaged per token; an expert that saw no token gets I."""
    h = h.float().clone()
    if h.dim() == 2:
        h = h[None]
    if n_tokens is not None:
        n = torch.as_tensor(n_tokens, device=h.device).reshape(-1).float()
        h /= n.clamp(min=1).reshape(-1, 1, 1)
        empty = n == 0
        if bool(empty.any()):
            h[empty] = torch.eye(h.shape[-1], device=h.device)
    return h


@torch.no_grad()
def gptq(
    w: torch.Tensor,
    h: torch.Tensor,
    q: Quantizer,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
    *,
    damp: float = 0.01,
    actorder: str = "static",
    block: int = 128,
    impl: str = "auto",
    fused_grid: bool = True,
    min_batch: int | None = None,
    label: str | None = None,
) -> torch.Tensor:
    """GPTQ codes of ``w [E, R, C]`` against fixed (scale, zero) ``[E, R, G]``.

    Args:
        w: weights (any float dtype; solved in fp32).
        h: Hessians ``[E, C, C]`` fp32, consumed (damped and factorized in place).
        q: the quantizer (format + kernel plan).
        scale: stored group scales.
        zero: zero points (asymmetric) or None.
        damp: added to the diagonal as a fraction of its mean (llm-compressor ``dampening_frac``).
        actorder: ``"none"`` or ``"static"`` (columns in decreasing Hessian diagonal; groups unchanged,
            so nothing extra is stored).
        block: columns per lazy-update block.
        impl: ``"torch"``, ``"triton"`` or ``"auto"`` (Triton on CUDA when the grid allows it).
        fused_grid: enable a grid's declared fused block formula; False keeps its torch reference.
        min_batch: factor ``h`` in a batch of at least this many, padding with identities. An expert-bank
            solve passes `MIN_FACTOR_BATCH`, so a short chunk is factored as the whole bank would be; a
            dense group's batch is always 1 and is left alone (None).
        label: which layer and unit this solve is, for `solve_offsets` (``build.json``).

    Returns:
        Codes ``[E, R, C]`` (fp32), in the original column order: integer-valued for integer formats, E4M3
        values for FP8 weights (which the torch path solves; the Triton block has no FP8-weight formula).
    """
    e, r, c = w.shape
    g = q.group(c)
    w = w.float().clone()
    diag = torch.diagonal(h, dim1=-2, dim2=-1)
    dead = diag == 0
    diag.masked_fill_(dead, 1.0)
    w.masked_fill_(dead[:, None, :], 0.0)
    perm = None
    if actorder == "static":
        perm = torch.argsort(diag, dim=-1, descending=True)
        w = w.gather(2, perm[:, None, :].expand(e, r, c))
        h = h.gather(1, perm[:, :, None].expand(e, c, c)).gather(2, perm[:, None, :].expand(e, c, c))
    elif actorder != "none":
        raise ValueError(f"actorder {actorder!r}: 'none' or 'static'")
    u = _factor(h, damp, min_batch)

    # per-column served scale / zero / FP8 factor, in solve order
    cols = torch.arange(c, device=w.device) // g
    cols = cols.expand(e, c) if perm is None else cols.expand(e, c).gather(1, perm)
    idx = cols[:, None, :].expand(e, r, c)
    s = q.served_scale(scale).gather(2, idx)
    z = zero.float().gather(2, idx) if zero is not None else None
    fp8 = None
    if q.fp8_scales:
        f8, chan = fp8_scale_parts(scale)
        fp8 = (f8.gather(2, idx), chan)
    grid = None if q.grid.range_rule else q.grid
    grid_fast = grid is not None and grid.gptq_fused and fused_grid
    # the kernel's FP8-scale formula (no grid) casts to E4M3, which Triton compiles on 8.9 and newer only
    fp8_ok = fp8 is None or grid is not None or not w.is_cuda or triton_fp8(w.device)
    if impl == "auto":
        impl = (
            "triton"
            if w.is_cuda and not q.fp8_weights and (grid is None or grid_fast) and fp8_ok
            else "torch"
        )
    elif impl == "triton" and not fp8_ok:
        raise ValueError(
            "impl 'triton' casts FP8 scales to E4M3, which needs compute capability 8.9 or newer"
        )
    elif impl == "triton" and q.fp8_weights:
        raise ValueError("impl 'triton' has no FP8-weight formula; the torch path casts through e4m3")
    elif impl == "triton" and grid is not None and not grid_fast:
        raise ValueError(f"impl 'triton' has no enabled fused formula for grid {grid.name!r}")
    codes = torch.empty_like(w)
    uc = u.contiguous() if impl == "triton" else u
    for i1 in range(0, c, block):
        i2 = min(i1 + block, c)
        if impl == "triton":
            err = _block_triton(w, s, z, fp8, uc, codes, i1, i2, q.fmt.qmin, q.fmt.qmax, grid is not None)
        else:
            err = _block_torch(w, s, z, fp8, u, codes, i1, i2, q.fmt.qmin, q.fmt.qmax, grid, q.fp8_weights)
        if i2 < c:
            w[:, :, i2:] -= torch.bmm(err, u[:, i1:i2, i2:])
    if perm is not None:
        codes = codes.gather(2, torch.argsort(perm, dim=-1)[:, None, :].expand(e, r, c))
    s0, s1 = int(w.stride(0)), int(w.stride(1))  # the launch's own geometry, in solve order
    fu = uc if impl == "triton" else u  # the factor tensor that path indexes
    u0, u1, u2 = (int(fu.stride(i)) for i in range(3))
    _SOLVES.append(
        {
            "label": label,
            "impl": impl,
            "fused_grid": impl == "triton" and grid is not None,
            "experts": e,
            "rows": r,
            "cols": c,
            "strides": [s0, s1],
            "max_offset": (e - 1) * s0 + (r - 1) * s1 + (c - 1),
            # per expert: the int64 expert base is added outside this product (the U path)
            "u_strides": [u0, u1, u2],
            "u_max_offset": (c - 1) * u1 + (c - 1) * u2,
        }
    )
    return codes


#: `torch.cholesky_inverse` factors a short batch through a different path: on an H200, torch 2.13.0+cu130,
#: batches of 1, 2, 4 and 8 differ from a whole bank's in the last bits (at 768, 2048 and 6144 columns), a
#: batch of 12 still differs on Qwen3-30B-A3B's real Hessians, and 16 matches. An expert-bank solve pads
#: its Hessian batch to this many (`gptq(..., min_batch=MIN_FACTOR_BATCH)`).
MIN_FACTOR_BATCH = 16


def _factor(h: torch.Tensor, damp: float, min_batch: int | None = None) -> torch.Tensor:
    """Upper Cholesky factor of the damped inverse Hessian; damping grows until every member factors.

    With `min_batch`, a shorter batch is padded with identities, which factor independently and are
    dropped, so a short chunk's factor is the one the whole bank would have got.
    """
    n = h.shape[0]
    pad = 0 if min_batch is None else max(0, min_batch - n)
    if pad:
        eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
        h = torch.cat([h, eye.expand(pad, -1, -1)])
    diag = torch.diagonal(h, dim1=-2, dim2=-1)
    lam = damp * diag.mean(-1, keepdim=True)
    diag += lam
    for attempt in range(8):
        lower, info = torch.linalg.cholesky_ex(h)
        if int(info.max()) == 0:
            break
        diag[info > 0] += lam[info > 0] * 2**attempt
    else:
        raise RuntimeError("GPTQ: Hessian not positive definite after extra damping")
    u = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    return u[:n] if pad else u


def _block_torch(w, s, z, fp8, u, codes, i1, i2, qmin, qmax, grid=None, fp8_weights=False) -> torch.Tensor:  # noqa: ANN001
    w1 = w[:, :, i1:i2].clone()
    err = torch.zeros_like(w1)
    u1 = u[:, i1:i2, i1:i2]
    for i in range(i2 - i1):
        col = w1[:, :, i]
        si = s[:, :, i1 + i]
        if grid is not None:
            zi = None if z is None else z[:, :, i1 + i]
            qi = grid.codes(col, si, zi, qmin, qmax)
            wq = grid.value(qi, si, zi)
        elif fp8 is not None:
            qi, wq = round_fp8(col, fp8[0][:, :, i1 + i], fp8[1][..., 0], qmin, qmax)
        elif fp8_weights:  # the checkpoint holds the E4M3 value; the kernel multiplies by the fp32 scale
            qi = torch.clamp(col / si, qmin, qmax).to(torch.float8_e4m3fn).float()
            wq = qi * si
        elif z is None:
            qi = torch.clamp(torch.round(col / si), qmin, qmax)
            wq = (qi * si).bfloat16().float()  # the kernel's bf16 dequant
        else:
            zi = z[:, :, i1 + i]
            qi = torch.clamp(torch.round(col / si) + zi, qmin, qmax)
            wq = ((qi - zi) * si).bfloat16().float()
        codes[:, :, i1 + i] = qi
        e = (col - wq) / u1[:, i, i][:, None]
        w1[:, :, i:] -= e[:, :, None] * u1[:, None, i, i:]
        err[:, :, i] = e
    return err


def _block_triton(w, s, z, fp8, u, codes, i1, i2, qmin, qmax, offset_grid=False) -> torch.Tensor:  # noqa: ANN001
    from ._triton import gptq_block

    return gptq_block(w, s, z, fp8, u, codes, i1, i2, qmin, qmax, offset_grid=offset_grid)


def expert_workspace(rows: int, cols: int) -> int:
    """The most bytes solving one expert of a ``[E, rows, cols]`` bank costs, in fp32.

    The Cholesky factorization holds four Hessian-sized ``[cols, cols]`` tensors and two weight-sized
    ``[rows, cols]`` ones; the column loop and the group-scale search hold two Hessian-sized and eight
    weight-sized ones. Calibrated on a real GLM layer, where a 32-expert share of the bank cost 33.4 GiB,
    1.04 GiB per expert, 6.7 x the expert's 0.156 GiB Hessian.
    """
    return 4 * max(4 * cols * cols + 2 * rows * cols, 2 * cols * cols + 8 * rows * cols)


def chunk_experts(experts: int, rows: int, cols: int, free: int) -> int:
    """How many experts of a bank one `gptq` call solves in `free` bytes: as many as fit, at least one.

    The count also stays inside `experts_per_solve`; one expert past the limit is the planner's refusal, not a
    smaller chunk.
    """
    return max(1, min(experts, free // expert_workspace(rows, cols), experts_per_solve(rows * cols)))


def free_bytes(device: torch.device) -> int:
    """Bytes `device` can still hand this process a chunk's tensors, less the headroom a plan leaves unused.

    That is the driver's free memory, not the driver's free memory plus the blocks torch already holds: a held
    block serves a request up to its own size, and a calibration forward leaves blocks that a solve's
    Hessian-sized tensors do not fit, so counting them sizes a chunk the allocator cannot place. The headroom a
    residency plan keeps unused (`engine.residency.plan_headroom`) is not the chunk's either: spending it breaks
    the plan's peak.

    A resident build whose forward fills the card leaves the driver little, so the chunk falls to one expert
    (its factor still pads to `MIN_FACTOR_BATCH`) and the build completes. Releasing the cache first
    (`empty_cache`) would size larger chunks and a faster solve, but the next forward then regrows its segments
    at the card's edge and the build is slower overall. The CPU reports no ceiling.
    """
    if device.type != "cuda":
        return 1 << 50
    from ..engine.residency import plan_headroom

    free, _ = torch.cuda.mem_get_info(device)
    return max(0, int(free) - plan_headroom(device))


def rtn(w: torch.Tensor, q: Quantizer, scale: torch.Tensor, zero: torch.Tensor | None) -> torch.Tensor:
    """Round-to-nearest codes on the served grid (any leading dims)."""
    return q.quantize(w, scale, zero)


def dequant_cols(
    q: Quantizer, codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None
) -> torch.Tensor:
    """Served weight of codes (fp32); same as `Quantizer.dequant`, re-exported for steps."""
    return q.dequant(codes, scale, zero)


__all__ = [
    "MIN_FACTOR_BATCH",
    "chunk_experts",
    "dequant_cols",
    "expert_workspace",
    "free_bytes",
    "gptq",
    "prepare_hessian",
    "rtn",
    "widen",
]
