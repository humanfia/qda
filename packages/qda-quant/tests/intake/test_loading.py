"""Model selection and VLM checkpoints with legacy and current source prefixes."""

import json

import pytest
import torch
import transformers
from qda_quant import FileSource, Recipe, build
from qda_quant.arch import describe
from qda_quant.engine.build_validation import _skeleton
from qda_quant.engine.packed import Packed, _cpu_buffers, load_float, load_student
from qda_quant.intake.fisher import load_bf16
from qda_quant.intake.loading import load_model
from qda_quant.intake.probe import _weight_map
from qda_quant.intake.source_names import SourceNames
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ
from qda_quant.target import Target
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..engine_util import save_tiny_tokenizer, write_ids


@pytest.mark.parametrize("family", ["qwen3", "qwen2_5"])
def test_vlm_build_keeps_source_names_and_vision_bytes(tmp_path, family):
    """Legacy Qwen2.5 prefixes survive a full GPTQ build and decompressed text forward.

    Unlike a tiny save_pretrained checkpoint, the real Qwen2.5 VL source uses `visual.*` and
    `model.layers.*`; this exercises transformers' renamings and the exporter's inverse map together.
    """
    cls = transformers.Qwen3VLConfig if family == "qwen3" else transformers.Qwen2_5_VLConfig
    cfg = cls(
        text_config={
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [4, 6, 6]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "embed_dim": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 128,
            "patch_size": 14,
            "deepstack_visual_indexes": [],
        },
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=127,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    source = tmp_path / "source"
    transformers.AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16).save_pretrained(source)
    if family == "qwen2_5":
        tensors = load_file(str(source / "model.safetensors"))
        tensors = {
            k.replace("model.visual.", "visual.").replace("model.language_model.", "model."): v
            for k, v in tensors.items()
        }
        save_file(tensors, str(source / "model.safetensors"), metadata={"format": "pt"})
    sha = save_tiny_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[3, 4, 5, 6] * 8] * 4, sha)
    out = build(
        source,
        Recipe(steps=[GPTQ(impl="torch")], calib=FileSource(path=str(calib), samples=4, seqlen=32)),
        format="w4a16-sym-g128",
        devices=["cpu"],
        out_dir=tmp_path / "out",
    )
    original, written = _weight_map(source), _weight_map(out)
    text_prefix = "model.language_model." if family == "qwen3" else "model."
    assert text_prefix + "layers.0.self_attn.q_proj.weight_packed" in written
    vision_prefix = "model.visual." if family == "qwen3" else "visual."
    vision = [k for k in original if k.startswith(vision_prefix)]
    assert vision and set(vision) <= set(written)
    for key in original:
        assert key in written or key.removesuffix(".weight") + ".weight_packed" in written
    for key in vision:
        with safe_open(source / original[key], "pt") as a, safe_open(out / written[key], "pt") as b:
            assert torch.equal(a.get_tensor(key), b.get_tensor(key)), key
    back = load_bf16(out, "cpu")
    assert type(back).__name__ == type(load_model(source)).__name__
    with torch.device("meta"):
        assert len(describe(load_model(out, weights=False), verify=False).layers) == 2
    names = SourceNames(out)
    for name, value in back.state_dict().items():
        key = names.tensor(name)
        if key in vision:
            with safe_open(out / written[key], "pt") as saved:
                assert torch.equal(value, saved.get_tensor(key)), name
    packed = Packed(out)
    scales, floats = packed.state(log=lambda _: None)
    student = load_student(packed, scales, floats, torch.device("cpu"), "eager")
    with torch.no_grad():
        ids = torch.tensor([[3, 4, 5, 6]])
        logits = back(input_ids=ids).logits
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(student(input_ids=ids).logits, logits, rtol=0.01, atol=0.01)
    quant = json.loads((out / "config.json").read_text())["quantization_config"]
    assert all(not k.startswith(vision_prefix) for k in written if "weight_packed" in k)
    assert quant["format"] == "pack-quantized"


def test_unmapped_config_is_refused_during_planning(tmp_path):
    transformers.ViTConfig().save_pretrained(tmp_path)
    with pytest.raises(Refused, match="no causal-LM or image-text-to-text mapping"):
        _skeleton(str(tmp_path), torch.bfloat16, Target())


@pytest.mark.parametrize("family", ["gemma4", "qwen3"])
def test_packed_readers_restore_stored_buffers(tmp_path, monkeypatch, family):
    """Stored layer scalars and arbitrary float/int buffers survive both packed readers; RoPE is derived."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    small = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    if family == "gemma4":
        cfg = transformers.Gemma4TextConfig(
            **small,
            global_head_dim=32,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            layer_types=["sliding_attention", "full_attention"],
        )
    else:
        # A non-Gemma model with config-owned persistent buffers exercises the general reader contract.
        original_init = Qwen3RotaryEmbedding.__init__

        def with_buffers(self, config):
            original_init(self, config)
            self.register_buffer("stored_float", torch.tensor(0.0))
            self.register_buffer("stored_int", torch.tensor(0, dtype=torch.int64))

        monkeypatch.setattr(Qwen3RotaryEmbedding, "__init__", with_buffers)
        cfg = transformers.Qwen3Config(**small)
    model = load_model(config=cfg, dtype=torch.bfloat16, attn_implementation="eager")
    if family == "gemma4":
        for i, layer in enumerate(model.model.layers):
            layer.layer_scalar.fill_(2 + i)
    else:
        model.model.rotary_emb.stored_float.fill_(37.5)
        model.model.rotary_emb.stored_int.fill_(17)
    expected = {name: value.clone() for name, value in model.named_buffers()}
    source = tmp_path / "source"
    model.save_pretrained(source)
    sha = save_tiny_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[3, 4, 5, 6] * 4] * 2, sha)
    out = build(
        source,
        Recipe(steps=[GPTQ(impl="torch")], calib=FileSource(path=str(calib), samples=2, seqlen=16)),
        format="w4a16",
        devices=["cpu"],
        out_dir=tmp_path / "out",
        log=lambda _: None,
    )
    packed = Packed(out)
    scales, floats = packed.state(lambda _: None)
    readers = [
        load_student(packed, scales, floats, torch.device("cpu"), "eager"),
        load_float(packed, torch.device("cpu"), "eager"),
    ]
    for reader in readers:
        actual = dict(reader.named_buffers())
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)
    with torch.no_grad():
        ids = torch.tensor([[3, 4, 5, 6]])
        torch.testing.assert_close(readers[0](input_ids=ids).logits, readers[1](input_ids=ids).logits)


def test_cpu_buffers_skips_indexed_layers_but_refreshes_rotary():
    """A decoder with an indexed constructor must not prevent config-only rotary refreshes."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer,
        Gemma4TextRotaryEmbedding,
    )

    cfg = transformers.Gemma4TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        layer_types=["sliding_attention"],
    )
    layer = Gemma4TextDecoderLayer(cfg, layer_idx=0)
    layer.layer_scalar.fill_(2)  # Unsupported constructors keep their existing buffers.
    rotary = Gemma4TextRotaryEmbedding(cfg)
    expected = {n: b.clone() for n, b in rotary.named_buffers()}
    assert expected
    for b in rotary.buffers():
        b.zero_()
    _cpu_buffers(torch.nn.ModuleList([layer, rotary]))
    assert layer.layer_scalar.item() == 2
    for name, b in rotary.named_buffers():
        torch.testing.assert_close(b, expected[name], rtol=0, atol=0)
