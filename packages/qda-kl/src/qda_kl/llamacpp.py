"""The strict load check of a `.gguf` the pinned llama-server loaded (the judge's admission check)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from qda_quant.step import Refused


def _gguf_reader(llama_cpp: Path) -> Any:
    path = str(llama_cpp / "gguf-py")
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        import gguf
    finally:
        if added:
            sys.path.remove(path)
    where = getattr(gguf, "__file__", None) or ""
    if not where or not Path(where).resolve().is_relative_to(llama_cpp.resolve()):
        raise Refused(f"{llama_cpp}: the imported gguf module is {where or 'unknown'}, not this checkout's")
    return gguf


def strict_check(model: str | Path, log: str, llama_cpp: str | Path) -> dict:
    """Compare the file inventory with every tensor the bridge observed after the actual load."""
    reader = _gguf_reader(Path(llama_cpp)).GGUFReader(model)
    expected = {
        t.name: (int(t.tensor_type), tuple(map(int, t.shape)) + (1,) * (4 - len(t.shape)))
        for t in reader.tensors
    }
    actual = {}
    for line in log.splitlines():
        if line.startswith("QDA_TENSOR\t"):
            _, name, dtype, *shape = line.split("\t")
            value = (int(dtype), tuple(map(int, shape)))
            if name in actual and actual[name] != value:
                raise Refused(f"conflicting loaded copies: {name}")
            actual[name] = value
    arch = str(reader.get_field("general.architecture").contents())
    nextn = reader.get_field(f"{arch}.nextn_predict_layers")
    n_nextn = 0 if nextn is None else int(nextn.contents())
    n_layers = int(reader.get_field(f"{arch}.block_count").contents())
    skipped = {}
    for name in expected.keys() - actual.keys():
        match = re.fullmatch(r"blk\.(\d+)\..+", name)
        if match and n_layers - n_nextn <= int(match[1]) < n_layers:
            skipped[name] = {"classification": "MTP-off", "file_type_shape": expected[name]}
    active = {name: value for name, value in expected.items() if name not in skipped}
    if actual != active:
        raise Refused(
            f"loaded inventory differs: missing={set(expected) - set(actual)}, "
            f"extra={set(actual) - set(expected)}, "
            f"changed={[k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]}"
        )
    ignored = re.findall(r"model has unused tensor ([^ ]+) .*-- ignoring", log)
    if set(ignored) - skipped.keys():
        raise Refused(f"llama.cpp ignored non-MTP tensors: {set(ignored) - skipped.keys()}")
    return {
        "verdict": "PASS-WITH-MTP-OFF" if skipped else "PASS",
        "tensors": len(expected),
        "types": dict(Counter(v[0] for v in expected.values())),
        "tensor_payload_bytes": sum(t.n_bytes for t in reader.tensors),
        "inventory": actual,
        "skipped": skipped,
        "ignored_log_names": ignored,
    }


__all__ = ["strict_check"]
