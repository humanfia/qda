"""Pack solved qparams into GGUF records in the template's row order."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..step import Refused
from . import kernel as kern
from .kernel import q4k_fields, q4k_records, q6k_fields, q6k_records, quantize_q4k, quantize_q6k
from .q4k import Q4K_FORMAT
from .q5k import Q5K_FORMAT
from .q6k import Q6K_FORMAT


def row_order(template: torch.Tensor, source: torch.Tensor, name: str) -> torch.Tensor:
    """Index ``i`` such that ``source[i]`` is ``template`` row by row (both bf16-exact fp32 ``[R, C]``).

    Raises:
        Refused: the template is not a row permutation of the source.
    """
    if template.shape == source.shape and torch.equal(template, source):
        return torch.arange(source.shape[0])
    rows: dict[bytes, list[int]] = defaultdict(list)
    for i, r in enumerate(source.numpy()):
        rows[r.tobytes()].append(i)
    for v in rows.values():
        v.reverse()  # pop() then hands out the first occurrence first
    try:
        return torch.tensor([rows[r.tobytes()].pop() for r in template.numpy()])
    except (KeyError, IndexError):
        raise Refused(f"gguf: {name}: the template is not a row permutation of the source weight") from None


def _split(x: torch.Tensor, what: str, name: str, kind: str = "Q4_K") -> tuple[np.ndarray, np.ndarray]:
    """Group products ``fp16(D) * k`` ``[n, 8]`` -> (``D`` fp16 bits ``[n]``, ``k`` 6-bit ``[n, 8]``).

    Ggml's plain fit sets ``D = max / 63``, so the largest group has ``k = 63``. Its weighted fit searches for ``D``, so
    there the largest ``k`` may be lower. Each block takes the highest that gives every product exactly: its bytes are
    ggml's when ``k = 63``, and any other answer holds the same products.
    """
    top = x.amax(-1)
    d = torch.zeros_like(top, dtype=torch.float16)
    k = torch.zeros_like(x)
    todo = (top > 0).nonzero()[:, 0]
    for largest in range(63, 0, -1):
        if not len(todo):
            break
        df = (top[todo] / largest).to(torch.float16)
        kk = torch.round(x[todo] / df.float()[:, None])
        fits = ((df.float()[:, None] * kk == x[todo]) & (kk >= 0) & (kk <= 63)).all(-1)
        d[todo[fits]], k[todo[fits]] = df[fits], kk[fits]
        todo = todo[~fits]
    if not torch.equal(d.float()[:, None] * k, x):
        raise ValueError(f"{name}: the {what} are not {kind}'s fp16 x 6-bit products")
    return d.view(torch.int16).numpy().view("<u2"), k.to(torch.uint8).numpy()


def q4k_bytes(
    qp: dict[str, Any], rows: torch.Tensor, values: torch.Tensor, name: str, workers: int | None = None
) -> bytes:
    """Q4_K records of one qparams entry, rows in template order.

    Args:
        qp: the entry: codes ``[R, C]``, scale and zero_point ``[R, C / 32]`` on the ``q4k`` grid.
        rows: `row_order` of the template against the source.
        values: the template's values ``[R, C]``, for groups whose scale is 0.
        name: tensor name, for messages.
        workers: threads of the kernel, for groups whose scale is 0.
    """
    if qp["format"] != Q4K_FORMAT:
        raise ValueError(
            f"{name}: stored as {qp['format']}; the gguf exporter writes {Q4K_FORMAT} (grid 'q4k')"
        )
    return _asym_bytes(qp, rows, values, name, workers, Q4K_FORMAT)


def q5k_bytes(qp, rows, values, name, workers=None) -> bytes:
    """Q5_K records of solved unsigned codes and exact Q4_K-style group products."""
    codes = qp["codes"]
    if qp["format"] != Q5K_FORMAT or codes.ndim != 2 or codes.shape[1] % 256 or codes.shape != values.shape:
        raise Refused(f"gguf: {name}: Q5_K requires {Q5K_FORMAT} and whole 256-value blocks")
    if ((codes < 0) | (codes > 31) | (codes != codes.round())).any():
        raise Refused(f"gguf: {name}: Q5_K codes must be integers in 0..31")
    try:
        return _asym_bytes(qp, rows, values, name, workers, Q5K_FORMAT)
    except ValueError as e:
        raise Refused(f"gguf: {e}; use grid 'q5k' and keep its fitted scales unchanged") from e


def _asym_bytes(qp, rows, values, name, workers, fmt) -> bytes:
    """The shared eight-group scale/minimum layout, with the format's code packing."""
    kind, read, pack, fit = {
        Q4K_FORMAT: ("Q4_K", q4k_fields, q4k_records, lambda x: quantize_q4k(x, workers=workers)),
        Q5K_FORMAT: ("Q5_K", kern.q5k_fields, kern.q5k_records, kern.quantize_q5k),
    }[fmt]
    codes = qp["codes"].cpu()[rows].to(torch.uint8)
    r, c = codes.shape
    d, sc = _split(qp["scale"].cpu().float()[rows].reshape(-1, 8), "scales", name, kind)
    dmin, m = _split(qp["zero_point"].cpu().float()[rows].reshape(-1, 8), "minimums", name, kind)
    codes = codes.reshape(-1, 256).numpy().copy()
    # A group with scale 0 keeps ggml's first-pass codes, which no scale can reproduce; its value is -dmin * m
    # whatever the code, so take ggml's codes for it.
    zero = (sc == 0) | (d.view("<f2") == 0)[:, None]
    if zero.any():
        blocks = np.flatnonzero(zero.any(-1))
        ggml = read(fit(values.reshape(-1, 256)[torch.from_numpy(blocks)]))[4].reshape(-1, 8, 32)
        sel = codes[blocks].reshape(-1, 8, 32)
        sel[zero[blocks]] = ggml[zero[blocks]]
        codes[blocks] = sel.reshape(-1, 256)
    assert codes.shape[0] * 256 == r * c
    return pack(d, dmin, sc, m, codes)


def _split_q6k(x: torch.Tensor, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Recover ggml's fp16 scale and signed int8 factors from sixteen exact fp32 products.

    Ggml assigns -128 to the extreme group, fixing the super-block scale. Every other
    product must divide it exactly; moving fitted scales cannot be silently rounded away.
    """
    extreme = x.gather(1, x.abs().argmax(-1, keepdim=True))
    d = (extreme[:, 0] / -128).to(torch.float16)
    d = torch.where(extreme[:, 0] == 0, torch.zeros_like(d), d)
    factors = torch.where(d[:, None] != 0, torch.round(x / d.float()[:, None]), torch.zeros_like(x))
    fits = (d.float()[:, None] * factors == x) & (factors >= -128) & (factors <= 127)
    if not fits.all():
        raise Refused(
            f"gguf: {name}: scales are not Q6_K's exact fp16 x int8 products; "
            "use grid 'q6k' and keep its fitted scales unchanged"
        )
    return d.view(torch.int16).numpy().view("<u2"), factors.to(torch.int16).numpy()


def q6k_bytes(
    qp: dict[str, Any], rows: torch.Tensor, values: torch.Tensor, name: str, workers: int | None = None
) -> bytes:
    """Q6_K records of one solved qparams entry, rows in template order.

    Args:
        qp: signed codes ``[R, C]`` and exact fitted scales ``[R, C / 16]`` on grid ``q6k``.
        rows: `row_order` of the template against the source.
        values: template values ``[R, C]``, used only for zero-scale groups.
        name: tensor name, for refusal messages.
        workers: kernel threads for zero-scale groups.

    Raises:
        Refused: format, shape, codes or scales cannot be represented as Q6_K.
    """
    if qp["format"] != Q6K_FORMAT or qp.get("zero_point") is not None:
        raise Refused(f"gguf: {name}: Q6_K requires {Q6K_FORMAT} on grid 'q6k', with no zero point")
    signed = qp["codes"].cpu()[rows]
    if signed.ndim != 2 or signed.shape[1] % 256 or values.shape != signed.shape:
        raise Refused(f"gguf: {name}: Q6_K requires matrix rows containing whole 256-value super-blocks")
    r, c = signed.shape
    products = qp["scale"].cpu().float()[rows]
    if products.shape != (r, c // 16):
        raise Refused(f"gguf: {name}: Q6_K requires one fitted scale per 16 values")
    if not ((signed >= -32) & (signed <= 31) & (signed == signed.round())).all():
        raise Refused(f"gguf: {name}: Q6_K requires integer codes in -32 .. 31")
    d, scales = _split_q6k(products.reshape(-1, 16), name)
    codes = (signed.to(torch.int16) + 32).to(torch.uint8).reshape(-1, 256).numpy().copy()
    # Zero products do not identify ggml's first-pass codes (or an underflowed block's
    # scale bytes). Preserve those unobservable fields from the original fitted values.
    zero = products.reshape(-1, 16).numpy() == 0
    if zero.any():
        blocks = np.flatnonzero(zero.any(-1))
        gd, gs, gc = q6k_fields(
            quantize_q6k(values.reshape(-1, 256)[torch.from_numpy(blocks)], workers=workers)
        )
        # The reference is an UNWEIGHTED fit of the template. It is the fit the build itself made only when
        # the build did not weigh its own, and there its unobservable fields are the ones llama-quantize
        # would write, so taking them keeps that path byte-identical. A weighted fit can zero a group the
        # unweighted one does not: an importance of zero across a group's sixteen columns is valid and
        # llama-imatrix produces it. There the reference says nothing about this build's bytes, and a zero
        # scale makes the group decode to zero whatever its codes are, so the build's own codes stand.
        agrees = zero[blocks] & (gd.view("<f2").astype(np.float32)[:, None] * gs == 0)
        selected = codes[blocks].reshape(-1, 16, 16)
        selected[agrees] = gc.reshape(-1, 16, 16)[agrees]
        codes[blocks] = selected.reshape(-1, 256)
        # A block ggml underflows is written out as zeros, which `_split_q6k` has already produced. Take the
        # reference's scale bytes only for a block it underflowed too.
        both = agrees.all(-1)
        d[blocks[both]], scales[blocks[both]] = gd[both], gs[both]
    return q6k_records(d, scales, codes)
