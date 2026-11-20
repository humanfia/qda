"""Tiny real safetensors checkpoints for the judge's tests (no torch: the format is a JSON header + bytes)."""

import json
import struct
from pathlib import Path

ITEM = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


def write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """tensors: name -> (dtype, shape, data bytes or None for zeros)."""
    header, blobs, at = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        n = ITEM[dtype]
        for s in shape:
            n *= s
        data = data if data is not None else bytes(n)
        data = (data * (n // len(data) + 1))[:n] if len(data) != n else data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + n]}
        blobs.append(data)
        at += n
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


CONFIG = {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1}
QCONFIG = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {"weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "int"}}
    },
}


def packed_tensors(weights: bytes = b"w1", zero_points: bool = False) -> dict:
    """A one-layer W4 group-128 checkpoint's tensors: packed q_proj + scales, a bf16 embedding."""
    t = {
        "model.embed_tokens.weight": ("BF16", (16, 8), b"\x01\x02"),
        "model.layers.0.self_attn.q_proj.weight_packed": ("I32", (8, 1), weights),
        "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (8, 1), b"\x03\x04"),
        "model.layers.0.self_attn.q_proj.weight_shape": ("I32", (2,), b"\x08\x00\x00\x00"),
    }
    if zero_points:
        t["model.layers.0.self_attn.q_proj.weight_zero_point"] = ("I32", (1, 1), b"\x00\x00\x00\x00")
    return t


def anchor_dir(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(CONFIG))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(
        d / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (16, 8), b"\x01\x02"),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", (8, 8), b"\x05\x06"),
        },
    )
    return d


def quant_dir(
    d: Path, weights: bytes = b"w1", zero_points: bool = False, tensors: dict | None = None
) -> Path:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({**CONFIG, "quantization_config": QCONFIG}))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors or packed_tensors(weights, zero_points))
    return d
