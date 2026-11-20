"""One sensitivity read: fake-quant one class at one ggml type, score it, and record the row.

A read is the unit an allocation is built from. It writes a variant of a prepared base in which exactly one
tensor class carries the values the pinned quantizer stores and decodes (`variant.build_variant`), scores that
variant against a sealed reference on the same named subset the search judge uses (`qda-kl`'s `Scorer`), and
writes the reading to `OUT/result.json` (the `qda-kl` result, unchanged) plus the row to `OUT/row.json`.

``kind == "bf16"`` scores the prepared base itself and records its ``kl`` as the baseline; it is not a cell,
because bf16 is the origin every increment is measured from.

Everything a row claims about its own inputs is recorded: the variant's rewritten shards and index, the base
manifest's digest, the quantizer pin and backend, the reference's content hash and subset, and the scorer's own
version and check fields. A reading the scorer marks non-deterministic is refused. `scorer` is injectable so a
CPU test can drive the write path without a server; by default it is `qda-kl`'s `score`, imported there only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..io import sha256_file as _sha256
from ..output.vllm_env import DEFAULT_TP

BF16 = "bf16"
DEFAULT_SCORER: Callable[[], Callable[..., Any]] | None = None


def _quantizer() -> dict[str, Any]:
    """The quantizer's provenance: the pin, the loaded backend, and the gguf-py reader that decodes it."""
    import gguf

    from . import kernel
    from .llama_cpp import PINNED

    return {
        "pinned": PINNED,
        "backend": kernel.ggml_backend() or "numpy",
        "ggml_commit": kernel.ggml_commit(),
        "gguf_py_quants_sha256": _sha256(Path(gguf.quants.__file__)),
    }


def _default_scorer() -> Callable[..., Any]:
    from ..registry import load_plugins

    load_plugins()
    if DEFAULT_SCORER is None:
        raise ModuleNotFoundError("No module named 'qda_kl'", name="qda_kl")
    return DEFAULT_SCORER()


def _result(got: Any) -> dict[str, Any]:
    """The scorer's result as JSON-ready data, through the serializer the score itself uses.

    `qda-kl`'s `Score` is a pydantic model: it writes its own `result.json` with `model_dump_json`, and
    `dataclasses.asdict` raises on it. A plain dataclass (a CPU test's scorer) still goes through `asdict`.
    """
    if hasattr(got, "model_dump"):
        return got.model_dump(mode="json")
    return asdict(got)


def read(
    base: str | Path,
    klass: str,
    kind: str,
    ref: str | Path,
    *,
    out: str | Path,
    subset: str | None = None,
    tp: int = DEFAULT_TP,
    libggml: str | Path | None = None,
    check: bool = True,
    scorer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Score `base` (or a one-class variant of it) against `ref` and write `OUT/result.json`, `OUT/row.json`."""
    from . import variant

    base, out = Path(base), Path(out)
    if out.exists():
        raise ValueError(f"{out}: a read is written into a fresh directory")
    manifest_path = base / "sensitivity-base.json"
    if not manifest_path.is_file():
        raise ValueError(f"{base}: not a prepared base (no sensitivity-base.json; prepare one first)")
    manifest = json.loads(manifest_path.read_text())
    if kind == BF16:
        if klass not in ("", BF16):
            raise ValueError("gguf: a bf16 read is the baseline; it replaces no class")
        checkpoint, written = base, {"baseline": True, "base_anchor": manifest["anchor"]}
    else:
        if klass not in manifest["classes"]:
            raise ValueError(f"gguf: {base}: no class {klass!r}; the base has {sorted(manifest['classes'])}")
        out.mkdir(parents=True)
        checkpoint = out / "ckpt"
        written = variant.build_variant(base, checkpoint, klass, kind, libggml=libggml)
    scorer = scorer or _default_scorer()
    got = scorer(str(checkpoint), str(ref), subset=subset, tp=tp, check=check)
    if check and getattr(got, "deterministic", None) is False:
        raise ValueError(f"gguf: {klass} {kind}: the scorer reported a non-deterministic reading")
    kl = float(got.kl)
    if not math.isfinite(kl):
        raise ValueError(f"gguf: {klass} {kind}: the reading is not finite ({got.kl})")
    out.mkdir(parents=True, exist_ok=True)
    result = _result(got)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    row = {
        "class": klass,
        "type": kind,
        "out": str(out),
        "kl": kl,
        "subset": subset,
        "reference": result.get("reference") or {},
        "written": written,
        "base": str(base),
        "base_sha256": _sha256(manifest_path),
        "quantizer": _quantizer(),
        "server": {"vllm": got.vllm, "tp": tp, "serve": result.get("serve")},
        "tokenizer_sha256": got.tokenizer_sha256,
        "check": {"deterministic": got.deterministic, "max_abs_diff": got.validate_max_abs_diff},
    }
    (out / "row.json").write_text(json.dumps(row, indent=1, default=str))
    return row
