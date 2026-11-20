"""A checkpoint with one tensor class replaced by the exact values a ggml type decodes to.

A sensitivity read serves the same bf16 checkpoint over and over, one class at a time, at one ggml type at a
time. Rewriting the whole checkpoint per read is wasteful and rewriting whole source shards is worse: a class
spans every shard. So the base is laid out once, **one class per shard file** (`prepare_base`), and a read
(`build_variant`) hardlinks every shard it does not touch and rewrites only the class's own shard.

The replacement values are `dequantize(quantize(x, type))` from the pinned ggml quantizer and gguf-py reader --
the same pair a GGUF file stores and decodes -- in fp32, stored bf16, everything else untouched. A class whose
tensor is stored in the converter's order is quantized in that order: `ssm_out` is the GDN output projection, whose
columns the converter tiles by value head. Its permutation is `columns.value_head_columns`, taken once and
inverted after quantization; row reorders are inert because the quantizer is per row.

``prepare_base`` holds the checkpoint in memory while grouping it (a 27B bf16 checkpoint is ~55 GB). It is run
once per scoring node; every read after it writes only the class's shard.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import torch

from ..io import sha256_file as _sha256
from . import classes as sensitivity
from . import kernel
from .columns import value_head_columns

#: a class's name is its GGUF tensor class (the same vocabulary `sensitivity.CLASSES` uses)
_HF_SUFFIX: dict[str, str] = {
    ".self_attn.q_proj.weight": "attn_q",
    ".self_attn.k_proj.weight": "attn_k",
    ".self_attn.v_proj.weight": "attn_v",
    ".self_attn.o_proj.weight": "attn_output",
    ".linear_attn.in_proj_qkv.weight": "attn_qkv",
    ".linear_attn.in_proj_z.weight": "attn_gate",
    ".linear_attn.in_proj_a.weight": "ssm_alpha",
    ".linear_attn.in_proj_b.weight": "ssm_beta",
    ".linear_attn.out_proj.weight": "ssm_out",
    ".mlp.gate_proj.weight": "ffn_gate",
    ".mlp.up_proj.weight": "ffn_up",
    ".mlp.down_proj.weight": "ffn_down",
}
_EXACT = {"lm_head.weight": "output"}
_QUANT = {"Q4_K": kernel.quantize_q4k, "Q5_K": kernel.quantize_q5k, "Q6_K": kernel.quantize_q6k}
_GDN_OUT = ".linear_attn.out_proj.weight"


def class_of(name: str) -> str:
    """The class of an HF tensor name: a grid class, ``mtp``, ``mtp_eh``, ``visual`` or ``keeps``."""
    if name.startswith("model.visual."):
        return "visual"
    if name.startswith("mtp."):
        return "mtp_eh" if name == "mtp.fc.weight" else "mtp"
    if name.endswith("embed_tokens.weight"):
        return "token_embd"
    if name in _EXACT:
        return _EXACT[name]
    return next((c for s, c in _HF_SUFFIX.items() if name.endswith(s)), "keeps")


def _shards(anchor: Path) -> list[Path]:
    """The checkpoint's shard files, in index order (or the single file of an unsharded checkpoint)."""
    index = anchor / "model.safetensors.index.json"
    if index.is_file():
        files = json.loads(index.read_text())["weight_map"].values()
        return [anchor / f for f in sorted(set(files))]
    files = sorted(anchor.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{anchor}: no *.safetensors and no model.safetensors.index.json")
    return files


def _gdn(anchor: Path) -> dict[str, int] | None:
    """The text model's value-head constants, or None for a model without a GDN output projection."""
    cfg = json.loads((anchor / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    try:
        nk, nv = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
        return {"nk": nk, "nv": nv, "head_v": int(text["linear_value_head_dim"])}
    except KeyError:
        return None


def _split(tensors: Mapping[str, torch.Tensor], max_shard: int) -> Iterator[dict[str, torch.Tensor]]:
    """Greedily pack `tensors` (sorted by name) into shards of at most `max_shard` bytes."""
    part: dict[str, torch.Tensor] = {}
    used = 0
    for name in sorted(tensors):
        t = tensors[name]
        size = t.numel() * t.element_size()
        if part and used + size > max_shard:
            yield part
            part, used = {}, 0
        part[name] = t
        used += size
    if part:
        yield part


def prepare_base(anchor: str | Path, out: str | Path, *, max_shard: int = 4 << 30) -> dict:
    """Write `anchor` as a checkpoint whose shard files each hold one tensor class, and return its manifest."""
    from safetensors.torch import load_file, save_file

    anchor, out = Path(anchor), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: not empty; a base is prepared into a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for shard in _shards(anchor):
        for name, t in load_file(str(shard)).items():
            groups[class_of(name)][name] = t
    manifest: dict = {
        "anchor": str(anchor),
        "gdn": _gdn(anchor),
        "max_shard": max_shard,
        "classes": {},
        "weight_map": {},
    }
    for klass in sorted(groups):
        for i, part in enumerate(_split(groups[klass], max_shard)):
            name = f"model-{klass}-{i:05d}.safetensors"
            save_file(part, str(out / name))
            got = manifest["classes"].setdefault(klass, {"shards": [], "tensors": 0, "bytes": 0})
            got["shards"].append(name)
            got["tensors"] += len(part)
            got["bytes"] += sum(t.numel() * t.element_size() for t in part.values())
            manifest["weight_map"].update({n: name for n in part})
    for f in anchor.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    total = sum(t.numel() * t.element_size() for part in groups.values() for t in part.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": manifest.pop("weight_map")}, indent=1)
    )
    (out / "sensitivity-base.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def fake_quant(
    x: torch.Tensor,
    kind: str,
    *,
    columns: torch.Tensor | None = None,
    chunk: int = 8192,
) -> torch.Tensor:
    """`x` after quantizing to `kind` and decoding exactly, in bf16, in the same layout it came in.

    `columns` is a column permutation to quantize in (the converter's stored order); values come back in `x`'s
    order. Row chunks bound the working set; a block depends only on its own values, so the chunk size is inert.
    """
    import gguf

    from .file import row_bytes
    from .rule import NATIVE

    width = int(x.shape[-1])
    flat = x.float().reshape(-1, width)
    out = torch.empty_like(flat)
    inverse = columns.argsort() if columns is not None else None
    ggml = getattr(gguf.GGMLQuantizationType, kind)
    for start in range(0, flat.shape[0], chunk):
        part = flat[start : start + chunk]
        if columns is not None:
            part = part[:, columns]
        if kind == "Q8_0":
            raw = gguf.quants.quantize(part.contiguous().numpy(), ggml)
        elif int(ggml) in NATIVE:
            raw = kernel.quantize_weighted_chunk(
                int(ggml), part.contiguous(), row_bytes(int(ggml), width), None
            )
        else:
            raw = _QUANT[kind](part.contiguous())
        raw = raw.tobytes() if isinstance(raw, np.ndarray) else raw
        vals = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), ggml)
        got = torch.from_numpy(np.ascontiguousarray(vals, dtype=np.float32)).view(part.shape)
        if inverse is not None:
            got = got[:, inverse]
        out[start : start + chunk] = got
    return out.view(x.shape).to(torch.bfloat16)


def build_variant(
    base: str | Path, out: str | Path, klass: str, kind: str, *, libggml: str | Path | None = None
) -> dict:
    """Write a checkpoint equal to `base` except `klass`, whose tensors are `kind`'s exact decode.

    The variant hardlinks every shard it does not rewrite, so it costs the class's own bytes. `out` must not
    exist: a read owns a fresh directory. `libggml` selects llama.cpp's quantizer (Q5_K needs it); without it the
    numpy port runs for Q4_K/Q6_K and Q5_K is refused. Returns the provenance a reading records.
    """
    from safetensors.torch import load_file, save_file

    base, out = Path(base), Path(out)
    manifest = json.loads((base / "sensitivity-base.json").read_text())
    if klass not in sensitivity.CLASSES:
        # the base also holds the fixed buckets (mtp, mtp_eh, visual, keeps): a read measures an allocation's
        # unit, so it replaces only a grid class, never a bucket the plan never assigns
        if klass in manifest["classes"]:
            raise ValueError(
                f"gguf: {klass}: a fixed bucket, not an allocatable class; a read replaces one of the "
                f"{len(sensitivity.CLASSES)} grid classes"
            )
        raise ValueError(f"{base}: no class {klass!r}; the base has {sorted(manifest['classes'])}")
    if out.exists():
        raise ValueError(f"{out}: a variant is written into a fresh directory")
    if libggml is not None:
        kernel.use_ggml(libggml)
    gdn = manifest["gdn"]
    shards = set(manifest["classes"][klass]["shards"])
    out.mkdir(parents=True)
    for f in base.iterdir():
        if f.name not in shards:
            os.link(f, out / f.name)
    columns = None
    if gdn is not None and gdn["nv"] != gdn["nk"]:
        columns = value_head_columns(gdn["nk"], gdn["nv"], gdn["head_v"])
    for shard in sorted(shards):
        tensors = load_file(str(base / shard))
        new = {
            name: fake_quant(t, kind, columns=columns if name.endswith(_GDN_OUT) else None)
            for name, t in tensors.items()
        }
        save_file(new, str(out / shard))
    return {
        "class": klass,
        "type": kind,
        "base_anchor": manifest["anchor"],
        "gdn": gdn,
        "rewritten_shards": {s: _sha256(out / s) for s in sorted(shards)},
        "index_sha256": _sha256(out / "model.safetensors.index.json"),
    }
