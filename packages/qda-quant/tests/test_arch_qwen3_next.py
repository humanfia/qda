"""Qwen3-Next adds the full-attention interval stock SGLang reads, derived from the source ``layer_types``."""

import pytest
import torch
import transformers
from qda_quant.arch import describe
from qda_quant.arch.adapters.qwen3_next import QWEN3_NEXT
from qda_quant.step import Refused

FULL = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _model(layer_types=FULL):
    config = transformers.Qwen3NextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=128,
        max_position_embeddings=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=list(layer_types),
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return transformers.AutoModelForCausalLM.from_config(config).eval()


def test_adapter_adds_the_fused_banks_and_the_interval_stock_sglang_reads():
    model = _model()
    graph = describe(model, verify=False)
    assert graph.adapter == "qwen3_next"
    for layer in graph.layers:
        (bank,) = layer.banks
        assert (bank.experts, bank.router, bank.top_k, bank.fused) == (
            f"{layer.module}.mlp.experts",
            f"{layer.module}.mlp.gate",
            2,
            True,
        )
        assert bank.shared == tuple(
            f"{layer.module}.mlp.shared_expert.{p}" for p in ("gate_proj", "up_proj", "down_proj")
        )
    source = model.config.to_dict()  # what a transformers 5.21 checkpoint saves: layer_types, no interval
    assert source["layer_types"] == FULL and "full_attention_interval" not in source
    served = QWEN3_NEXT.export_config(source)
    assert served["full_attention_interval"] == 4
    assert {k: v for k, v in served.items() if k != "full_attention_interval"} == source


def test_export_config_refuses_a_schedule_one_interval_cannot_express():
    assert QWEN3_NEXT.export_config({"layer_types": ["full_attention"] * 2})["full_attention_interval"] == 1
    assert QWEN3_NEXT.export_config({"num_hidden_layers": 2}) == {"num_hidden_layers": 2}
    with pytest.raises(Refused, match="cannot express"):
        QWEN3_NEXT.export_config({"layer_types": ["full_attention", "linear_attention"]})
    with pytest.raises(Refused, match="requires a full-attention layer"):
        QWEN3_NEXT.export_config({"layer_types": ["linear_attention"] * 2})
    with pytest.raises(Refused, match="declares full_attention_interval=2"):
        QWEN3_NEXT.export_config({"layer_types": FULL, "full_attention_interval": 2})


def test_export_layout_keeps_the_per_expert_layout_of_a_fused_source():
    layout = QWEN3_NEXT.export_layout("model.layers.0.mlp.experts.gate_up_proj", torch.Size([8, 512, 256]))
    names = [piece[0] for piece in layout.pieces]
    assert names[:2] == [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
    ]
    assert len(names) == 16  # 8 experts, gate and up
    assert layout.pieces[0][2]["packed"] == "model.layers.0.mlp.experts.0.gate_proj.weight_packed"
    assert QWEN3_NEXT.export_layout("model.layers.0.linear_attn.in_proj_qkvz", torch.Size([1])) is None


def test_describe_refuses_at_plan_time():
    with pytest.raises(Refused, match="cannot express"):
        describe(_model(["full_attention", "linear_attention"]), verify=False)
