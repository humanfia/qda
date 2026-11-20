"""A GGUF candidate 0's type map, from one measured class sweep per (model, template, reference).

`qda sensitivity read` prices one tensor class at one ggml type: it fake-quantizes that class of a bf16
checkpoint with the pinned quantizer, scores the variant against a sealed reference, and records the KL
increment over bf16. A sweep reads every class the template has at every type it can store, and `qda
sensitivity table` turns those rows into a type per class inside a file-size budget. The sweep's rows are a
pure function of their inputs, so they are measured once per machine into

    $QDA_CACHE/gguf-sensitivity/<model fingerprint[:16]>/<key>/rows.json   (the table's own manifest)
    $QDA_CACHE/gguf-sensitivity/<model fingerprint[:16]>/<key>.json        (what the key covers)

The key covers the model's content fingerprint, the template's sha256, the reference the reads scored against
(its content hash and the subset), the pinned quantizer's decoder, and the arm set (the classes and the types
tried). A budget, a floor and an allocation quantum are not part of it: the rows say what each cell costs for
every budget, so the allocation is re-derived without another read.

The layout, lock and atomic write follow the reference cache (`qda_kl.refcache`): two callers of one
key wait on a lock instead of measuring twice, and rows without their record are an unfinished measurement,
never a hit. The allocation and its `tensor_types` rules are cheap, so only the rows are cached.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qda_quant.gguf import classes as grid
from qda_quant.intake.fingerprint import model_fingerprint
from qda_quant.io import qda_cache, sha256_file

__all__ = [
    "ROWS",
    "VERSION",
    "cache_root",
    "cells",
    "entry",
    "key_payload",
    "lookup",
    "measuring",
    "record_path",
    "rows_path",
    "store",
]

#: The rows file in an entry directory (the shape `qda sensitivity table` reads).
ROWS = "rows.json"
#: The arm set's shape this module writes and reads; a cache of another version is not reinterpreted.
VERSION = 1


def cache_root() -> Path:
    """``$QDA_CACHE/gguf-sensitivity``; without QDA_CACHE ``~/.cache/qda/gguf-sensitivity``."""
    return qda_cache("gguf-sensitivity")


def cells(plan: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The arms: every class of a `qda sensitivity plan --json` at every type it can store, in plan order."""
    out: list[tuple[str, str]] = []
    for row in plan:
        if "class" not in row or "payload_bytes" not in row:
            continue
        for name, _kind in grid.CELL_TYPES:
            if (row.get("payload_bytes") or {}).get(name) is not None:
                out.append((row["class"], name))
    return out


def key_payload(
    model: str | Path,
    template: str | Path,
    *,
    reference: dict[str, Any],
    arms: list[tuple[str, str]],
) -> dict[str, Any]:
    """Everything the rows depend on, as one JSON-able mapping.

    ``reference`` is the reference manifest's own ``{"content_sha256", "subset"}``; a read against another
    reference, or another subset of it, is another sweep.
    """
    import gguf
    from qda_quant.gguf.llama_cpp import PINNED

    return {
        "version": VERSION,
        "model": model_fingerprint(model),
        "template": {
            "sha256": sha256_file(Path(template)),
            "bytes": Path(template).stat().st_size,
        },
        "reference": {"content_sha256": reference.get("content_sha256"), "subset": reference.get("subset")},
        "quantizer": {
            "pinned": PINNED,
            "gguf_py_quants_sha256": sha256_file(Path(gguf.quants.__file__)),
        },
        "arms": [list(cell) for cell in arms],
    }


def entry(payload: dict[str, Any]) -> Path:
    """The cache directory of the sweep ``payload`` describes."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_root() / payload["model"][:16] / digest[:24]


def rows_path(where: Path) -> Path:
    """The rows file in an entry directory."""
    return where / ROWS


def record_path(where: Path) -> Path:
    """The record beside an entry directory: ``<key>.json``."""
    return where.with_name(where.name + ".json")


def lookup(payload: dict[str, Any]) -> Path | None:
    """The cached rows of ``payload``, or None; a hit is the record's own sha256 of the rows."""
    where = entry(payload)
    found, record = rows_path(where), record_path(where)
    if not found.is_file() or not record.is_file():
        return None
    try:
        got = json.loads(record.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or got.get("payload") != payload or got.get("sha256") != sha256_file(found):
        return None
    return found


@contextmanager
def measuring(target: Path, log: Any = print) -> Iterator[None]:
    """Hold the sweep's lock (``target``'s entry), waiting for another process measuring the same key."""
    lock = target.with_name(f".{target.name}.cache-lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"sensitivity: another process is measuring this sweep; waiting for it ({lock})")
            started = time.monotonic()
            fcntl.flock(handle, fcntl.LOCK_EX)
            log(f"sensitivity: waited {time.monotonic() - started:.0f} s for the other sweep")
        yield


def store(rows: str | Path, payload: dict[str, Any], *, baseline_kl: float) -> Path:
    """Write measured rows into the cache under ``payload``'s entry, atomically, and return the entry.

    The rows are written first under a partial name and renamed; the record (the payload and the rows'
    sha256) is written after it, so a reader never takes a half-written sweep for a finished one. The caller
    holds `measuring(entry(payload))`, so two prepares of one key measure once.
    """
    target = entry(payload)
    target.mkdir(parents=True, exist_ok=True)
    partial = target / f".{ROWS}.partial"
    partial.write_bytes(Path(rows).read_bytes())
    partial.replace(target / ROWS)
    record = {
        "payload": payload,
        "file": ROWS,
        "sha256": sha256_file(target / ROWS),
        "bytes": (target / ROWS).stat().st_size,
        "baseline_kl": baseline_kl,
    }
    tmp = target.with_name(f".{target.name}.json.partial")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    tmp.replace(record_path(target))
    return target
