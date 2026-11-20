"""Expert projection rows and numbers must survive fused/native layouts and the template writer shortcut."""

from __future__ import annotations

import json

import pytest
import torch
import transformers
from qda_quant.config import BuildConfig
from qda_quant.gguf import rule

from ..engine_util import save_tiny_tokenizer, write_ids


def _source(tmp_path, per_expert=False):
    """A tiny real Qwen3-MoE checkpoint and BF16 template, with a different row permutation per projection."""
    import gguf
    from safetensors.torch import save_file

    cfg = transformers.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.experts.gate_up_proj.normal_(0, 0.1)
            layer.mlp.experts.down_proj.normal_(0, 0.1)
    src = tmp_path / "src"
    model.save_pretrained(src)
    fingerprint = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], fingerprint)
    weights = {key: value.contiguous() for key, value in model.state_dict().items()}
    if not per_expert:
        save_file(weights, str(src / "model.safetensors"))
    template = tmp_path / "template.gguf"
    writer = gguf.GGUFWriter(str(template), "qwen3moe")
    writer.add_block_count(1)
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_expert_feed_forward_length(cfg.moe_intermediate_size)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(cfg.num_key_value_heads)
    writer.add_key_length(cfg.head_dim)
    writer.add_value_length(cfg.head_dim)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    writer.add_expert_count(cfg.num_experts)
    writer.add_expert_used_count(cfg.num_experts_per_tok)
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3MOE, 1)
    expanded = {}
    for key, value in weights.items():
        if key.endswith(".experts.gate_up_proj"):
            root = key.removesuffix("gate_up_proj")
            for projection, part in zip(("gate_proj", "up_proj"), value.chunk(2, dim=1), strict=True):
                expanded[root + projection + ".weight"] = part
        elif key.endswith(".experts.down_proj"):
            expanded[key + ".weight"] = value
        else:
            expanded[key] = value
    for key, value in expanded.items():
        name = names.get_name(key, try_suffixes=(".weight",))
        assert name is not None, key
        if value.dim() == 3:
            value = value.roll(3 if "gate_exps" in name else 7, dims=1)
        if "norm" in name:
            writer.add_tensor(name, value.float().reshape(1, -1).numpy())
        else:
            writer.add_tensor(
                name, value.contiguous().view(torch.int16).numpy(), raw_dtype=gguf.GGMLQuantizationType.BF16
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if per_expert:
        for key in list(weights):
            if key.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                del weights[key]
        for key, value in expanded.items():
            if value.dim() == 3:
                root, projection = key.removesuffix(".weight").rsplit(".", 1)
                for expert, part in enumerate(value):
                    weights[f"{root}.{expert}.{projection}.weight"] = part.contiguous()
        save_file(weights, str(src / "model.safetensors"))
    return src, calib, template


@pytest.mark.parametrize("per_expert", [False, True], ids=["native-fused", "native-per-expert"])
def test_expert_projection_order_matches_template_shortcut(tmp_path, monkeypatch, per_expert):
    """No gate/up split or expert may move when the source layout or template row order changes."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template
    from qda_quant.gguf.kernel import quantize_q4k

    src, calib, template = _source(tmp_path, per_expert)
    # Isolate writer behavior from the separately ported architecture mixing policy.
    monkeypatch.setattr(
        rule, "q4_k_m_types", lambda arch, n, tensors: {name: kind for name, _, kind in tensors}
    )
    cfg = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "tensor_types": [
                    {"pattern": "re:.*_exps\\.weight$", "type": "Q4_K"},
                ],
            }
        },
        out=str(tmp_path / "fast"),
    )
    fast = cfg.build(log=lambda _: None)
    slow = BuildConfig.from_file(
        fast / "resolved.yaml", overrides=[f"out={tmp_path / 'slow'}", "exporter.gguf.layer_pass=true"]
    ).build(log=lambda _: None)
    assert (fast / "model-Q4_K_M.gguf").read_bytes() == (slow / "model-Q4_K_M.gguf").read_bytes()
    assert json.loads((fast / "build.json").read_text())["stages"][0]["kind"] == "export"
    assert json.loads((slow / "build.json").read_text())["stages"][0]["kind"] == "layers"
    solved = BuildConfig.from_file(
        fast / "resolved.yaml", overrides=[f"out={tmp_path / 'gptq'}", "steps.0={gptq: {impl: torch}}"]
    ).build(log=lambda _: None)
    output, original = Template(solved / "model-Q4_K_M.gguf"), Template(template)
    expert_names = [name for name in output.tensors if "_exps.weight" in name]
    assert len(expert_names) == 3
    assert all(output.tensors[name].shape[2] == 4 for name in expert_names)
    assert all(len(t.shape) == 1 for name, t in output.tensors.items() if "norm" in name)
    assert any(output.raw(name) != quantize_q4k(original.values(name)) for name in expert_names)
