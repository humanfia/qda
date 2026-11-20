"""Compressed generation records for regrading without another model run."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_samples(
    work: Path, checkpoint: str, suite: str, settings: dict[str, Any], records: Iterable[dict[str, Any]]
) -> Path:
    """Write gzip JSONL under ``work``, separated by checkpoint, suite and resolved settings."""
    key = json.dumps([checkpoint, suite, settings], sort_keys=True).encode()
    path = Path(work) / "generations" / f"{hashlib.sha256(key).hexdigest()}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
