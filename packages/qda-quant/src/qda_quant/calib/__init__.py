"""Calibration data: the original model's own answers (:mod:`.selfgen`) and weighted mixing of sources (:mod:`.mix`).

Both write a calibration JSONL that a recipe reads through ``calib: {file: {path: ...}}``, or through the ``mix``
and ``selfgen`` sources (:mod:`.sources`), which call them (docs/recipe.md, "Calibration data"),
plus a ``<file>.meta.json`` sidecar recording how it was made. Self-generated data is cached per model under
``$QDA_CACHE/calib/<model fingerprint>/``.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from ..intake.fingerprint import model_fingerprint
from ..io import qda_cache, sha256_file

__all__ = [
    "cache_root",
    "meta_path",
    "model_fingerprint",
    "read_rows",
    "row_text",
    "sha256_file",
    "write_jsonl",
]


def cache_root() -> Path:
    """``$QDA_CACHE/calib``; without QDA_CACHE ``~/.cache/qda/calib``."""
    return qda_cache("calib")


def write_jsonl(out: Path, rows: list[dict], meta: dict) -> Path:
    """Write a calibration file and its sidecars, each atomically.

    ``rows`` go to ``out``; when ``meta`` names a tokenizer, its digest goes to ``out.tokenizer`` (the engine's
    sidecar); ``meta`` goes to ``out.meta.json`` last, so a present meta sidecar means a complete file. Each is a
    unique temp file (pid + random), renamed into place.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))]
    if meta.get("tokenizer"):
        files.append((out.with_name(out.name + ".tokenizer"), meta["tokenizer"] + "\n"))
    files.append((meta_path(out), json.dumps(meta, indent=1, sort_keys=True)))
    for path, text in files:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_text(text)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return out


def meta_path(path: str | Path) -> Path:
    """The sidecar of a calibration file."""
    p = Path(path)
    return p.with_name(p.name + ".meta.json")


def read_rows(path: str | Path) -> list[dict]:
    """The JSON rows of a calibration JSONL."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_text(row: dict) -> str:
    """The text a row carries: ``text``, or the ``messages`` contents joined; ``""`` for token-id rows."""
    if "text" in row:
        return str(row["text"])
    if "messages" in row:
        return "\n".join(str(m.get("content", "")) for m in row["messages"])
    if "prompt" in row:
        return str(row["prompt"])
    return ""
