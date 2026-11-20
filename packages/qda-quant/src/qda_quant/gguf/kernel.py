"""Q4_K and Q6_K quantizers that write the exact bytes llama.cpp writes.

Q4_K stores super-blocks of 256 values: 8 groups of 32 with 6-bit scales and minimums, plus an fp16 scale and
minimum. Q6_K stores 16 groups of 16 with int8 scales and an fp16 scale. ``llama-quantize`` without an importance
matrix writes them with ``quantize_row_q4_K_ref`` / ``quantize_row_q6_K_ref`` (``ggml/src/ggml-quants.c``); this
module ports those and their helpers ``make_qkx2_quants`` and ``make_qx_quants``. Widths must be a multiple of 256.

Two paths, one result:

* ``use_ggml`` loads a C library that exports ``quantize_row_q4_K_ref`` / ``quantize_row_q6_K_ref`` and calls
  them itself — the C this module ports, at C speed (many times faster on a whole model). The library is a built
  ``libggml-base``, or the one shipped in qda-quant's linux wheel (``_quant.so``, ``use_bundled``); the bytes
  come from the same source as ``llama-quantize``.
* ``_quantize_q4k_fast`` / ``_quantize_q6k_fast`` are the numpy port: they batch the independent super-blocks;
  loops C runs over a group stay Python loops, so every sum is still added in order. They run on ``workers``
  threads over contiguous slices, so the bytes do not depend on the worker count. They are the reference and the
  fallback: the tests pin their bytes to llama.cpp's, and compare them with the library when one is available.

Every intermediate is fp32 in C's order, because the last bit of a code depends on it. Byte identity was checked
against llama.cpp ``9e2a04a``; ``ggml-quants.c`` is in ``libggml-base``, which ``GGML_NATIVE`` does not change.
For non-finite inputs (undefined in C) the port reproduces what x86 produced in our parity checks.
Q5_K uses the native ``ggml_quantize_chunk`` fit and has no NumPy fallback.
"""


# This module is the public face of the package: it re-exports the implementation, so an import here is
# not an unused import.
# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ._backend import (
    _CHUNK_BLOCKS,
    _MIN_SHARD_BLOCKS,
    _REF,
    _WORKER_CAP,
    _as_blocks,
    _fast_bytes,
    _Ggml,
    _quantize_q4k_fast,
    _quantize_q6k_fast,
    _resolve_workers,
    _shard_bytes,
    default_workers,
    ggml_backend,
    ggml_commit,
    ggml_version,
    has_weighted,
    q4k_fields,
    q4k_records,
    q6k_fields,
    q6k_records,
    quantize_q4k,
    quantize_q4k_weighted,
    quantize_q6k,
    quantize_weighted_chunk,
    use_ggml,
)
from ._port import (
    _EPS,
    _F32,
    _NAN_BITS,
    _Q4K_BLOCK,
    _Q6K_BLOCK,
    _ROUND_MAGIC,
    QK_K,
    _fp16_bits,
    _fp16_to_fp32,
    _make_qkx2_fast,
    _make_qx_fast,
    _nearest_int,
    _pack_q4k,
    _pack_q6k,
    _q4k_nibbles,
    _q4k_records,
    _q4k_scale_bytes,
    _q4k_scale_values,
    _q6k_records,
    _scan_max_zero,
    _seq_max,
    _seq_min,
    _seq_sum,
)

__all__ = [
    "bundled_path",
    "ggml_backend",
    "ggml_commit",
    "ggml_version",
    "has_weighted",
    "quantize_weighted_chunk",
    "q4k_fields",
    "q4k_records",
    "q5k_fields",
    "q5k_records",
    "q6k_fields",
    "q6k_records",
    "quantize_q4k",
    "quantize_q5k",
    "quantize_q6k",
    "use_bundled",
    "use_ggml",
]


def quantize_q5k(x: torch.Tensor, importance: torch.Tensor | None = None) -> bytes:
    """Q5_K's native plain or imatrix fit; a configured libggml-base is required."""
    from ..step import Refused
    from .file import row_bytes
    from .rule import Q5_K

    if x.ndim != 2:
        raise ValueError("Q5_K expects a matrix of whole 256-value blocks")
    size = row_bytes(Q5_K, x.shape[1])
    try:
        return quantize_weighted_chunk(Q5_K, x, size, importance)
    except ValueError as e:
        raise Refused(f"Q5_K needs native ggml_quantize_chunk; set exporter.gguf.libggml: {e}") from e


def q5k_fields(records: bytes):
    """Read Q5_K's Q4_K scale/minimum fields and unsigned five-bit codes."""
    rec = np.frombuffer(records, dtype=np.uint8).reshape(-1, 176)
    d, dmin, sc, m, codes = q4k_fields(np.concatenate((rec[:, :16], rec[:, 48:]), axis=1).tobytes())
    high = (rec[:, None, 16:48] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
    return d, dmin, sc, m, codes | (high.reshape(-1, 256) << 4)


def q5k_records(d, dmin, sc, m, codes) -> bytes:
    """Pack Q5_K fields: Q4_K's header, 32 high-bit bytes, then 128 low-nibble bytes."""
    low = np.frombuffer(q4k_records(d, dmin, sc, m, codes & 15), dtype=np.uint8).reshape(-1, 144)
    high = ((codes.reshape(-1, 8, 32) >> 4) & 1) << np.arange(8, dtype=np.uint8)[None, :, None]
    return np.concatenate((low[:, :16], np.bitwise_or.reduce(high, axis=1), low[:, 16:]), axis=1).tobytes()


def bundled_path() -> Path | None:
    """The C quantizer shipped in this wheel, or None (a source checkout, or a platform without it)."""
    path = Path(__file__).with_name("_quant.so")
    return path if path.is_file() else None


def use_bundled() -> str:
    """Quantize through the shipped C quantizer when there is one, else through the numpy port.

    The library is compiled into linux wheels from `_quant.c` (llama.cpp's own reference functions); a source
    checkout without one, or a library that will not load, is not an error: the numpy port writes the same bytes,
    only slower.

    Returns:
        One line naming the backend, e.g. ``qda-quant-gguf 0.1.0 (9e2a04a)`` or ``numpy``.
    """
    path = bundled_path()
    if path is None:
        return use_ggml(None)
    try:
        return use_ggml(path)
    except ValueError:  # a broken or foreign file: the fallback is the same bytes, not a failed build
        return use_ggml(None)
