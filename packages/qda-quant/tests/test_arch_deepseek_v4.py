import pytest
from qda_quant.arch import SHARED_EXPERTS, describe
from qda_quant.arch.adapters.deepseek_v4 import (
    ATTENTION_INPUT,
    COMPRESSOR,
    GROUPED_OUTPUT,
    HYPER_CONNECTIONS,
    INDEXER,
)

from .tiny import model


def _rel(g, n):
    return n[len(g.module) + 1 :] if n else n


@pytest.mark.parametrize("meta", [False, True])
def test_deepseek_v4_groups_banks_and_float_weights(meta):
    graph = describe(model("deepseek_v4", meta=meta), verify=not meta)
    assert graph.adapter == "deepseek_v4"
    if not meta:
        assert graph.verified and graph.extra["dropped"] == []
    for g in graph.layers:
        # the attention inputs q_a_proj / kv_proj stay float, so no group is left on the block's input
        assert [([_rel(g, m) for m in x.modules], x.kind) for x in g.groups] == [
            # q_b_proj reads q_norm inside self_attn, so it is not a residual-stream writer.
            (["self_attn.q_b_proj"], "other"),
            (["self_attn.o_b_proj"], "attn_out"),
        ]
        (bank,) = g.banks
        assert bank.experts == f"{g.module}.mlp.experts" and bank.router == f"{g.module}.mlp.gate"
        assert bank.fused and bank.top_k == 2
        assert bank.shared == (
            f"{g.module}.mlp.shared_experts.gate_proj",
            f"{g.module}.mlp.shared_experts.up_proj",
            f"{g.module}.mlp.shared_experts.down_proj",
        )
    kept = graph.extra["kept_bf16"]
    assert kept[GROUPED_OUTPUT] == tuple(f"model.layers.{i}.self_attn.o_a_proj" for i in range(4))
    assert kept[ATTENTION_INPUT] == tuple(
        f"model.layers.{i}.self_attn.{p}" for i in range(4) for p in ("q_a_proj", "kv_proj")
    )
    assert "model.layers.1.self_attn.compressor.gate_proj" in kept[COMPRESSOR]
    assert "model.layers.2.self_attn.compressor.position_bias" in kept[COMPRESSOR]
    assert "model.layers.1.self_attn.compressor.indexer.q_b_proj" in kept[INDEXER]
    assert "model.layers.0.attn_hc.fn" in kept[HYPER_CONNECTIONS]
    assert "model.hc_head.hc_fn" in kept[HYPER_CONNECTIONS]
    assert len(kept[SHARED_EXPERTS]) == 12  # three linears in each of the four layers
    # a norm fold must still scale the linears taken out of the groups
    readers = graph.extra["readers"]["model.layers.1.input_layernorm"]
    assert "model.layers.1.self_attn.compressor.kv_proj" in readers
    assert "model.layers.1.self_attn.compressor.indexer.gate_proj" in readers
    assert {"model.layers.1.self_attn.q_a_proj", "model.layers.1.self_attn.kv_proj"} <= set(readers)


def test_deepseek_v4_draft_ignore_derives_the_mtp_layer_names():
    """A vLLM MTP draft builds MTP layer i at ``model.layers.{num_hidden_layers+i}``; the adapter says so."""
    m = model("deepseek_v4")
    first = m.config.num_hidden_layers
    assert m.config.num_nextn_predict_layers == 1
    assert f"model.layers.{first}.e_proj" in describe(m, verify=False).extra["draft_ignore"]

    m.config.num_nextn_predict_layers = 2
    names = describe(m, verify=False).extra["draft_ignore"]
    assert len(names) == 18
    for suffix in (
        "e_proj",
        "h_proj",
        "attn.fused_wqa_wkv",  # the draft fuses the checkpoint's wq_a and wkv
        "attn.wq_b",
        "attn.wo_a",
        "attn.wo_b",
        "ffn.shared_experts.gate_up_proj",
        "ffn.shared_experts.down_proj",
    ):
        assert f"model.layers.{first}.{suffix}" in names
        assert f"model.layers.{first + 1}.{suffix}" in names
    # vLLM checks the MoE's ignore entry against the unfused per-expert names
    for i in range(2):
        assert rf"re:model\.layers\.{first + i}\.ffn\.experts\.\d+\.w[123]" in names


def test_deepseek_v4_serving_normalization_preserves_rope_and_refuses_hash_reordering():
    """Legacy aliases preserve RoPE frequencies; counting scattered hash layers would change routing."""
    import torch
    from qda_quant.arch.adapters import check_served
    from qda_quant.arch.adapters.deepseek_v4 import DEEPSEEK_V4
    from qda_quant.format import Format
    from qda_quant.step import Refused
    from transformers import DeepseekV4Config
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    m = model("deepseek_v4", meta=True)
    graph = describe(m, verify=False)
    with pytest.raises(Refused, match="hash MoE layers to form a prefix"):
        check_served(graph, Format.parse("w4a16-sym-g128"))

    cfg = m.config.to_dict()
    cfg["mlp_layer_types"] = ["hash_moe", "hash_moe", "moe", "moe"]
    original = DeepseekV4Config(**cfg)
    served = DeepseekV4Config(**DEEPSEEK_V4.export_config(cfg))
    assert served.layer_types == original.layer_types
    assert served.mlp_layer_types == original.mlp_layer_types
    expected, expected_scale = DeepseekV4RotaryEmbedding.compute_default_rope_parameters(
        original, torch.device("cpu"), layer_type="compress"
    )
    actual, actual_scale = _compute_yarn_parameters(served, torch.device("cpu"), layer_type="compress")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_scale == expected_scale
