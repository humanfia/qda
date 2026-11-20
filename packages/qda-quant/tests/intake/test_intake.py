"""qda inspect: format detection on tiny directories.

The old layout is built here with compressed-tensors' own packer (`pack_to_int32`, what the legacy research codec
called), so detection is checked against an independent implementation, not against our packer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# The legacy rows were packed by compressed-tensors 0.18's layout; 0.17 packs 3-bit codes differently.
pytest.importorskip("compressed_tensors", minversion="0.18")
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from qda_quant.intake.formats import CheckpointFormat
from qda_quant.intake.report import inspect
from qda_quant.output.export import pack_module
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER = {"tokenizer.json": "{}", "tokenizer_config.json": '{"chat_template": "x"}'}
MODULES = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj")


def _group(bits: int, sym: bool, group: int, targets: list[str], **extra) -> dict:
    w = {"num_bits": bits, "type": "int", "symmetric": sym, "strategy": "group", "group_size": group, **extra}
    return {"targets": targets, "weights": w, "input_activations": None, "output_activations": None}


def _old(path: Path, bits: int, sym: bool, head: bool = False, g_idx: bool = False) -> None:
    """A legacy research row: per-module CT tensors (out 20 and in 100 leave padding bits), records, tokenizer."""
    gen = torch.Generator().manual_seed(bits)
    t = {"model.embed_tokens.weight": torch.randn(32, 100, generator=gen).bfloat16()}
    mods = [(m, bits, 128, sym) for m in MODULES] + ([("lm_head", 4, 64, True)] if head else [])
    for m, b, g, s in mods:
        lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
        q = torch.randint(lo, hi + 1, (20, 100), generator=gen, dtype=torch.int8)
        groups = -(-100 // g)
        t[m + ".weight_packed"] = pack_to_int32(q, b, packed_dim=1)
        t[m + ".weight_scale"] = torch.rand(20, groups, generator=gen).bfloat16()
        t[m + ".weight_shape"] = torch.tensor([20, 100], dtype=torch.int64)
        if not s:
            z = torch.randint(lo, hi + 1, (20, groups), generator=gen, dtype=torch.int8)
            t[m + ".weight_zero_point"] = pack_to_int32(z, b, packed_dim=0)
        if g_idx:
            t[m + ".weight_g_idx"] = torch.arange(100, dtype=torch.int32) // g
    path.mkdir()
    save_file(t, str(path / "model.safetensors"))
    groups = {"group_0": _group(bits, sym, 128, ["Linear"], actorder="static")}
    if head:
        groups["group_head"] = _group(4, True, 64, ["re:.*lm_head$"])
    qc = {"quant_method": "compressed-tensors", "format": "pack-quantized", "version": "0.18.0",
          "config_groups": groups, "ignore": ["model.embed_tokens"]}  # fmt: skip
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "quantization_config": qc})
    )
    for name, text in {**TOKENIZER, "quant-record.json": "{}", "generation_config.json": "{}"}.items():
        (path / name).write_text(text)


@pytest.fixture
def fresh_formats(monkeypatch):
    """`CHECKPOINT_FORMATS` with its built-ins but entry points not yet read; restored after the test."""
    from qda_quant.intake.formats import CHECKPOINT_FORMATS

    monkeypatch.setattr(CHECKPOINT_FORMATS, "_entries", dict(CHECKPOINT_FORMATS._entries))
    return CHECKPOINT_FORMATS


def test_a_third_party_format_comes_from_an_entry_point(monkeypatch, fresh_formats, tmp_path: Path):
    """A format package adds detection and serving with an entry point alone; outranking the catch-all, it is asked."""
    from ..test_plugins import fake_plugins

    class FakeFormat(CheckpointFormat):
        name = "fake"

        def matches(self, src):
            return "fake.flag" in src.files

        def describe(self, src, out):
            out.update(format="fake w2", layout="fake", kept_bf16=["lm_head"])

        def serve(self, src, rep, target):
            rep["kernels"]["dense"] = "FakeLinearKernel"

    d = tmp_path / "fake"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "not-compressed"}}))
    (d / "fake.flag").write_text("")
    fake_plugins(monkeypatch, lambda: fresh_formats.register("fake", FakeFormat()))
    rep = inspect(str(d))
    assert rep["format"] == "fake w2" and rep["layout"] == "fake" and rep["kept_bf16"] == ["lm_head"]
    assert rep["kernels"] == {"dense": "FakeLinearKernel"}


def test_inspect_detects_formats_and_what_blocks_serving(tmp_path: Path):
    """bf16, the old layout, g_idx (no act-order 'group' on vLLM 0.33), missing tokenizer, VLM without processor."""
    plain = tmp_path / "bf16"
    plain.mkdir()
    save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16)},
              str(plain / "model.safetensors"))  # fmt: skip
    (plain / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    rep = inspect(str(plain))
    assert rep["format"] == "bf16" and rep["reasons"] == ["missing tokenizer (tokenizer.json, "
                                                          "tokenizer.model or vocab.json + merges.txt)"]  # fmt: skip
    assert any(m.startswith("generation_config.json") for m in rep["missing_optional"])
    for name, text in TOKENIZER.items():
        (plain / name).write_text(text)
    assert inspect(str(plain))["reasons"] == []
    # only a checkpoint whose record plans a vLLM server names the model a request sends
    assert "serve_name" not in inspect(str(plain))
    (plain / "build.json").write_text(json.dumps({"serve": {"args": {}}}))
    assert inspect(str(plain))["serve_name"] == "bf16"  # a record naming no source: the directory's name
    (plain / "config.json").write_text(json.dumps({"architectures": ["X"], "vision_config": {}}))
    assert inspect(str(plain))["missing"][0].startswith("preprocessor_config.json")

    _old(tmp_path / "w3", 3, True, head=True)
    rep = inspect(str(tmp_path / "w3"))
    assert rep["layout"] == "legacy" and rep["format"] == "compressed-tensors pack-quantized w3a16+w4a16"
    assert rep["kernels"]["group_0/dense"] == "HummingLinearKernel" and not rep["reasons"]

    fused = (
        tmp_path / "fused"
    )  # fused expert banks name their tensors <bank>_packed, not <bank>.weight_packed
    fused.mkdir()
    qp = {"format": "w4a16-sym-g128", "codes": torch.zeros(2, 8, 128, dtype=torch.int8),
          "scale": torch.ones(2, 8, 1, dtype=torch.bfloat16)}  # fmt: skip
    t = {
        k: v
        for p in ("gate_up_proj", "down_proj")
        for k, v in pack_module(f"model.layers.0.mlp.experts.{p}", qp).items()
    }
    del t["model.layers.0.mlp.experts.down_proj_shape"]
    save_file({k: v.clone() for k, v in t.items()}, str(fused / "model.safetensors"))
    qc = {"quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": [],
          "config_groups": {"group_0": _group(4, True, 128, ["Linear"])}}  # fmt: skip
    (fused / "config.json").write_text(json.dumps({"architectures": ["X"], "quantization_config": qc}))
    assert inspect(str(fused))["unpacked"] == ["model.layers.0.mlp.experts.down_proj"]

    _old(tmp_path / "gidx", 4, True, g_idx=True)
    rep = inspect(str(tmp_path / "gidx"))
    assert rep["g_idx"] and "weight_g_idx" in rep["reasons"][0]


def test_inspect_a_gguf_file_needs_no_sidecar_files(tmp_path: Path):
    """A GGUF file carries its own config, tokenizer and weights; llama.cpp serves it as it is."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")
    rep = inspect(str(f))
    assert rep["format"] == "gguf"
    assert rep["missing"] == [] and rep["missing_optional"] == []


def test_inspect_refuses_a_tied_head_the_body_group_would_quantize(tmp_path: Path):
    """A tied head stores no tensor of its own: without its ``ignore`` entry the body group would quantize it
    and loading fails, so inspect reports such a config as unservable."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    src = tmp_path / "old"
    cfg = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=8,
                      num_key_value_heads=4, vocab_size=256, max_position_embeddings=128, tie_word_embeddings=True)  # fmt: skip
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(src)
    with safe_open(src / "model.safetensors", "pt") as h:
        t = {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
    assert "lm_head.weight" not in t
    gen = torch.Generator().manual_seed(0)
    for k in [k for k, v in t.items() if k.endswith(".weight") and v.dim() == 2 and "embed" not in k]:
        m, w = k[: -len(".weight")], t.pop(k)
        t[m + ".weight_packed"] = pack_to_int32(
            torch.randint(-8, 8, w.shape, dtype=torch.int8, generator=gen), 4
        )
        t[m + ".weight_scale"] = torch.rand(w.shape[0], w.shape[1] // 128, generator=gen).bfloat16()
        t[m + ".weight_shape"] = torch.tensor(w.shape, dtype=torch.int64)
    save_file(t, str(src / "model.safetensors"))
    raw = json.loads((src / "config.json").read_text())
    raw["quantization_config"] = {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                  "version": "0.18.0", "ignore": ["model.embed_tokens"],  # no `lm_head`
                                  "config_groups": {"group_0": _group(4, True, 128, ["Linear"])}}  # fmt: skip
    (src / "config.json").write_text(json.dumps(raw))
    for name, text in {**TOKENIZER, "quant-record.json": "{}"}.items():
        (src / name).write_text(text)
    rep = inspect(str(src))
    assert rep["unpacked"] == ["lm_head"] and "lm_head" in rep["reasons"][0]
