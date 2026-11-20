"""The quantizer backend: the loaded C library, the worker pool, and the public entry points.

``_BACKEND`` is the process-wide choice made by `use_ggml`: None runs the numpy port (`_port`), a `_Ggml` runs
``libggml-base``'s own ``quantize_row_q{4,6}_K_ref`` (or the shipped ``_quant.so``) through ctypes. The public
``quantize_q4k`` / ``quantize_q6k`` dispatch on it; the worker helpers split independent super-blocks across
threads without changing the bytes.
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._port import (
    _Q4K_BLOCK,
    _Q6K_BLOCK,
    QK_K,
    _pack_q4k,
    _pack_q6k,
    _q4k_nibbles,
    _q4k_records,
    _q4k_scale_bytes,
    _q4k_scale_values,
    _q6k_code_bytes,
    _q6k_records,
)

_CHUNK_BLOCKS = 16384  # super-blocks per vectorized batch: 16 MB of input

_WORKER_CAP = (
    32  # the Python port's default upper bound on worker threads; past this the GIL costs more than it buys
)

_MIN_SHARD_BLOCKS = 1024  # per worker; smaller slices make thread startup dominate


def default_workers() -> int:
    """The thread count ``workers=None`` means: every core in C, as ``llama-quantize``; the Python port stops at a cap."""
    cores = os.cpu_count() or 1
    return cores if _BACKEND is not None else min(_WORKER_CAP, cores)


def _resolve_workers(workers: int | None, n_blocks: int) -> int:
    """Turn the public ``workers`` argument into a thread count the input is large enough to keep busy."""
    if workers is None:
        workers = default_workers()
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
        raise TypeError(f"workers must be an int or None, got {type(workers).__name__}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    return min(int(workers), max(1, n_blocks // _MIN_SHARD_BLOCKS))


def _shard_bytes(blocks: np.ndarray, records) -> bytes:
    """Quantize one contiguous shard in bounded chunks; the bytes are chunk-independent."""
    with np.errstate(all="ignore"):  # the error state is per thread; workers start with the default
        return b"".join(
            records(blocks[i : i + _CHUNK_BLOCKS]).tobytes() for i in range(0, blocks.shape[0], _CHUNK_BLOCKS)
        )


def _fast_bytes(blocks: np.ndarray, records, workers: int | None) -> bytes:
    """Quantize ``(n, 256)`` super-blocks on ``workers`` threads, concatenating the shards in order."""
    n = blocks.shape[0]
    if n == 0:
        return b""
    workers = _resolve_workers(workers, n)
    if workers == 1:
        return _shard_bytes(blocks, records)
    bounds = np.linspace(0, n, workers + 1).astype(np.int64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = [
            pool.submit(_shard_bytes, blocks[bounds[i] : bounds[i + 1]], records) for i in range(workers)
        ]
    return b"".join(part.result() for part in parts)


def _quantize_q4k_fast(blocks: np.ndarray, workers: int | None = 1) -> bytes:
    """Quantize an ``(n, 256)`` fp32 array; super-blocks are independent, so any split gives the same bytes."""
    return _fast_bytes(blocks, _q4k_records, workers)


def _quantize_q6k_fast(blocks: np.ndarray, workers: int | None = 1) -> bytes:
    """Quantize an ``(n, 256)`` fp32 array; super-blocks are independent, so any split gives the same bytes."""
    return _fast_bytes(blocks, _q6k_records, workers)


# ---------------------------------------------------------------------------------------------------------
# llama.cpp's own kernel, loaded from its build


#: ggml functions by grid name: the symbol and the bytes one super-block takes.
_REF = {"q4k": ("quantize_row_q4_K_ref", _Q4K_BLOCK), "q6k": ("quantize_row_q6_K_ref", _Q6K_BLOCK)}


class _Ggml:
    """``libggml-base``'s ``quantize_row_q{4,6}_K_ref`` through ctypes: the C the port follows, in C time.

    ctypes releases the GIL in the call, so slices of super-blocks run on threads and the bytes are still the C
    ones: each super-block is quantized for itself, in order.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()  # a bare name would be searched on the loader's path, not the cwd
        self.lib = ctypes.CDLL(str(self.path))
        self.fns: dict[str, Any] = {}
        for name, (symbol, _) in _REF.items():
            fn = getattr(self.lib, symbol)  # AttributeError: not a llama.cpp quantizer library
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
            self.fns[name] = fn
        # llama.cpp's weighted fit; the shipped extension (`_quant.c`) has only the plain quantizers.
        self.weighted = getattr(self.lib, "quantize_q4_K", None)
        if self.weighted is not None:
            self.weighted.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            self.weighted.restype = ctypes.c_size_t
        # llama.cpp's own type-dispatched entry: it takes the importance vector for every type it encodes,
        # and with a null one it is the plain fit (measured, not assumed: q4_K/q6_K/q5_0/q8_0 all identical).
        self.chunk = getattr(self.lib, "ggml_quantize_chunk", None)
        if self.chunk is not None:
            self.chunk.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            self.chunk.restype = ctypes.c_size_t
        self.product = self._text("qda_product") or "libggml"  # the shipped extension names itself
        self.version = self._text("ggml_version")
        self.commit = self._text("ggml_commit")

    def _text(self, symbol: str) -> str:
        """A ``const char *`` function's value, or '' when this build does not export it."""
        if not hasattr(self.lib, symbol):
            return ""
        fn = getattr(self.lib, symbol)
        fn.argtypes = []
        fn.restype = ctypes.c_char_p
        return (fn() or b"").decode()

    def describe(self) -> str:
        """``libggml 0.29.0 (9e2a04a)`` or ``qda-quant-gguf 0.1.0 (9e2a04a)``, as the plan and log name it."""
        parts = [f"{self.product} {self.version}" if self.version else self.product]
        if self.commit:
            parts.append(f"({self.commit})")
        return " ".join(parts)

    def quantize(self, name: str, blocks: np.ndarray, workers: int | None) -> bytes:
        """Packed records of ``(n, 256)`` super-blocks, quantized on ``workers`` threads."""
        fn = self.fns[name]
        block = _REF[name][1]
        n = blocks.shape[0]
        workers = _resolve_workers(workers, n)
        if n == 0:
            return b""
        if workers == 1:
            return self._call(fn, blocks, block)
        bounds = np.linspace(0, n, workers + 1).astype(np.int64)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            shards = pool.map(
                lambda i: self._call(fn, blocks[bounds[i] : bounds[i + 1]], block), range(workers)
            )
            return b"".join(shards)

    def quantize_weighted(self, rows: np.ndarray, importance: np.ndarray, workers: int | None) -> bytes:
        """Packed Q4_K records of ``(r, c)`` rows with one importance per column, on ``workers`` threads."""
        weighted = self.weighted
        if weighted is None:
            raise ValueError(
                f"{self.path}: no quantize_q4_K; the importance-weighted fit needs llama.cpp's libggml-base"
            )
        r, c = rows.shape
        row_bytes = c // QK_K * _Q4K_BLOCK
        out = np.empty(r * row_bytes, dtype=np.uint8)
        bounds = np.linspace(0, r, _resolve_workers(workers, r * c // QK_K) + 1).astype(np.int64)

        def one(a: int, b: int) -> None:
            weighted(rows[a:b].ctypes.data, out.ctypes.data + a * row_bytes, b - a, c, importance.ctypes.data)

        with ThreadPoolExecutor(max_workers=len(bounds) - 1) as pool:
            list(pool.map(one, bounds[:-1].tolist(), bounds[1:].tolist()))
        return out.tobytes()

    def quantize_chunk(
        self, ggml_type: int, rows: np.ndarray, row_bytes: int, importance: np.ndarray | None
    ) -> bytes:
        """``ggml_quantize_chunk`` over ``(r, c)`` rows, weighted by one importance per column when given."""
        if self.chunk is None:
            raise ValueError(f"{self.path}: no ggml_quantize_chunk; use a built libggml-base")
        r, c = rows.shape
        out = np.empty(r * row_bytes, dtype=np.uint8)
        weights = None if importance is None else np.ascontiguousarray(importance, dtype=np.float32)
        wrote = self.chunk(
            ggml_type,
            rows.ctypes.data,
            out.ctypes.data,
            0,
            r,
            c,
            None if weights is None else weights.ctypes.data,
        )
        if wrote != out.size:
            raise ValueError(f"ggml_quantize_chunk wrote {wrote} bytes, expected {out.size}")
        return out.tobytes()

    def _call(self, fn, blocks: np.ndarray, block: int) -> bytes:  # noqa: ANN001
        """One call over a contiguous slice: its rows must be whole super-blocks (they are)."""
        out = np.empty(blocks.shape[0] * block, dtype=np.uint8)
        fn(blocks.ctypes.data, out.ctypes.data, np.int64(blocks.shape[0] * QK_K))
        return out.tobytes()


#: The backend the public functions use: None is the numpy port, which is the reference and the fallback.
_BACKEND: _Ggml | None = None


def use_ggml(lib_path: str | Path | None) -> str:
    """Quantize through llama.cpp's own C reference, or (with None) through this module's numpy port.

    The setting is process-wide: the engine's fit and the exporter's other types both go through
    `quantize_q4k` / `quantize_q6k`.

    Args:
        lib_path: a built ``libggml-base`` (``cmake --build ... --target ggml-base``), or None for numpy.

    Returns:
        One line naming the backend, e.g. ``libggml 0.29.0 (9e2a04a)`` or ``numpy``.

    Raises:
        ValueError: the library does not load or lacks a quantizer symbol.
    """
    global _BACKEND
    if lib_path is None:
        _BACKEND = None
        return "numpy"
    try:
        lib = _Ggml(lib_path)
    except (OSError, AttributeError) as e:
        _BACKEND = None
        raise ValueError(f"{lib_path}: not a llama.cpp quantizer library ({e})") from e
    _BACKEND = lib
    return lib.describe()


def ggml_backend() -> str | None:
    """`use_ggml`'s description when the C reference is loaded, else None."""
    return None if _BACKEND is None else _BACKEND.describe()


def has_weighted() -> bool:
    """True when the loaded library exports llama.cpp's importance-weighted Q4_K fit (``quantize_q4_K``).

    The shipped ``_quant.so`` has only the plain quantizers; a built ``libggml-base`` has both.
    """
    return _BACKEND is not None and _BACKEND.weighted is not None


def ggml_version() -> str:
    """The loaded library's ``ggml_version()``, or '' (the numpy port has none)."""
    return "" if _BACKEND is None else _BACKEND.version


def ggml_commit() -> str:
    """The loaded library's ``ggml_commit()``, or '' (the numpy port has none)."""
    return "" if _BACKEND is None else _BACKEND.commit


# ---------------------------------------------------------------------------------------------------------
# Public API


def _as_blocks(x: torch.Tensor) -> np.ndarray:
    """Validate a 1-D or 2-D fp32 tensor and view it as ``(n, 256)`` super-blocks."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(x).__name__}")
    if x.dim() not in (1, 2):
        raise ValueError(f"expected a 1-D or 2-D tensor, got {x.dim()} dimensions")
    width = x.shape[-1] if x.dim() == 2 else x.shape[0]
    if width % QK_K:
        raise ValueError(f"width must be a multiple of {QK_K}, got {width}")
    arr = x.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return arr.reshape(-1, QK_K)


def quantize_q4k(x: torch.Tensor, workers: int | None = None) -> bytes:
    """Quantize ``x`` to Q4_K exactly as ``llama-quantize`` writes it.

    Args:
        x: fp32 values, 1-D or 2-D ``(rows, width)``; every width must be a multiple of 256. Other dtypes are
            cast to fp32 first.
        workers: threads; None uses `default_workers`, at least 1024 super-blocks each (`_resolve_workers`).

    Returns:
        The packed records: ``n_rows * width / 256`` super-blocks of 144 bytes each, in row-major order.
    """
    blocks = _as_blocks(x)
    if _BACKEND is not None:
        return _BACKEND.quantize("q4k", blocks, workers)
    return _quantize_q4k_fast(blocks, workers)


def quantize_q4k_weighted(x: torch.Tensor, importance: torch.Tensor, workers: int | None = None) -> bytes:
    """Quantize ``x`` to Q4_K as ``llama-quantize --imatrix`` writes it.

    This is llama.cpp's own ``quantize_q4_K``; it has no numpy port, so it needs `use_ggml`.

    Args:
        x: ``(rows, width)``; the width must be a multiple of 256.
        importance: ``(width,)``, the mean squared input of each column, which is what an importance matrix holds.
        workers: threads; None as in `quantize_q4k`.

    Returns:
        The packed records, as `quantize_q4k` returns them.

    Raises:
        ValueError: no library is loaded.
    """
    if _BACKEND is None:
        raise ValueError("the importance-weighted Q4_K fit needs llama.cpp's libggml-base; none is loaded")
    rows = _as_blocks(x).reshape(x.shape[0], -1)
    return _BACKEND.quantize_weighted(
        rows, importance.detach().to("cpu", torch.float32).contiguous().numpy(), workers
    )


def quantize_weighted_chunk(
    ggml_type: int, x: torch.Tensor, row_bytes: int, importance: torch.Tensor | None
) -> bytes:
    """Quantize ``x`` to `ggml_type` through llama.cpp's ``ggml_quantize_chunk``, weighted when given.

    The entry ``llama-quantize`` itself uses, so a weighted fit of any type is the one it would write. With no
    importance it is the plain fit of that type, byte for byte.

    Args:
        ggml_type: the target type's ``ggml_type`` value.
        x: ``(rows, width)`` fp32 values.
        row_bytes: encoded bytes per row, which fixes the output size.
        importance: ``(width,)`` mean squared input per column, or None for the plain fit.

    Raises:
        ValueError: no library is loaded, or it has no ``ggml_quantize_chunk``.
    """
    if _BACKEND is None:
        raise ValueError("an importance-weighted fit needs llama.cpp's libggml-base; none is loaded")
    rows = x.detach().to("cpu", torch.float32).contiguous().numpy().reshape(x.shape[0], -1)
    weights = None if importance is None else importance.detach().to("cpu", torch.float32).numpy()
    return _BACKEND.quantize_chunk(ggml_type, rows, row_bytes, weights)


def quantize_q6k(x: torch.Tensor, workers: int | None = None) -> bytes:
    """Quantize ``x`` to Q6_K exactly as ``llama-quantize`` writes it.

    Args:
        x: fp32 values, 1-D or 2-D ``(rows, width)``; every width must be a multiple of 256. Other dtypes are
            cast to fp32 first.
        workers: threads; None uses `default_workers`, at least 1024 super-blocks each (`_resolve_workers`).

    Returns:
        The packed records: ``n_rows * width / 256`` super-blocks of 210 bytes each, in row-major order.
    """
    blocks = _as_blocks(x)
    if _BACKEND is not None:
        return _BACKEND.quantize("q6k", blocks, workers)
    return _quantize_q6k_fast(blocks, workers)


def q4k_fields(records: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode Q4_K records into their fields.

    Returns:
        ``(d, dmin, sc, m, codes)``: fp16 bits of the super-block scale and minimum ``(n,)`` uint16, 6-bit group
        scales and minimums ``(n, 8)`` uint8, codes ``(n, 256)`` uint8. A value is
        ``fp16(d) * sc * code - fp16(dmin) * m``.
    """
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, _Q4K_BLOCK)
    n = rec.shape[0]
    d = rec[:, 0:2].copy().view("<u2").reshape(n)
    dmin = rec[:, 2:4].copy().view("<u2").reshape(n)
    sc, m = _q4k_scale_values(rec[:, 4:16])
    qs = rec[:, 16:144].reshape(n, 4, 1, 32)
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(n, 256)
    return d, dmin, sc, m, codes


def q4k_records(d: np.ndarray, dmin: np.ndarray, sc: np.ndarray, m: np.ndarray, codes: np.ndarray) -> bytes:
    """Encode the fields `q4k_fields` returns back into Q4_K records."""
    return _pack_q4k(d, dmin, _q4k_scale_bytes(sc, m), _q4k_nibbles(codes)).tobytes()


def q6k_fields(records: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Q6_K records into their fields.

    A Q6_K super-block is 16 groups of 16: an fp16 ``d`` and a signed 8-bit scale per group, and 6-bit codes
    split across ``ql`` (low nibble) and ``qh`` (high two bits). A value is ``fp16(d) * s * (code - 32)``.

    Returns:
        ``(d, scales, codes)``: fp16 bits ``(n,)`` uint16, group scales ``(n, 16)`` int16 holding the stored
        signed bytes, and codes ``(n, 256)`` uint8 in ggml's stored ``0 .. 63`` form, not the signed value.
    """
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, _Q6K_BLOCK)
    n = rec.shape[0]
    ql = rec[:, 0:128].reshape(n, 2, 64)
    qh = rec[:, 128:192].reshape(n, 2, 32)
    scales = rec[:, 192:208].copy().view(np.int8).astype(np.int16).reshape(n, 16)
    d = rec[:, 208:210].copy().view("<u2").reshape(n)
    lo, hi = ql[:, :, 0:32], ql[:, :, 32:64]
    quarters = [
        (lo & 0xF) | ((qh & 3) << 4),
        (hi & 0xF) | (((qh >> 2) & 3) << 4),
        (lo >> 4) | (((qh >> 4) & 3) << 4),
        (hi >> 4) | (((qh >> 6) & 3) << 4),
    ]
    codes = np.stack(quarters, axis=2).reshape(n, 256)
    return d, scales, codes.astype(np.uint8)


def q6k_records(d: np.ndarray, scales: np.ndarray, codes: np.ndarray) -> bytes:
    """Encode the fields `q6k_fields` returns back into Q6_K records.

    Codes use ggml's stored ``0 .. 63`` form; group scales hold signed int8 values.
    The low nibbles and high bit pairs interleave four 32-value quarters per half-block.
    """
    ql, qh = _q6k_code_bytes(codes)
    return _pack_q6k(ql, qh, scales.astype(np.int8).view(np.uint8), d).tobytes()
