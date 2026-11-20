"""GLM-5.3-Flash: what its adapter adds to the graph, and a whole build through the layer carry."""

import torch
from qda_quant.arch import describe
from qda_quant.arch.adapters.deepseek_v4 import HYPER_CONNECTIONS
from qda_quant.arch.adapters.glm5_next import ATTENTION, GLM5_NEXT
from qda_quant.config import BuildConfig
from qda_quant.engine.stats import LayerStats
from qda_quant.intake.loading import load_model
from safetensors.torch import load_file
from transformers.models.glm5_next import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig

from .engine_util import save_tiny_tokenizer, write_ids


def _model():
    config = Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            vocab_size=128,
            pad_token_id=0,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            index_kpool=2,
            hc_mult=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            num_nextn_predict_layers=0,
            max_position_embeddings=64,
            layer_types=["linear_attention", "linear_attention", "indexed_attention", "linear_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            linear_attn_config={"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 2},
        ),
        vision_config=Glm5NextVisionConfig(depth=1, hidden_size=16, out_hidden_size=128, num_heads=2),
    )
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = load_model(config=config, dtype=torch.float32).eval()
    GLM5_NEXT.configure_model(m, None)
    return m


def test_only_the_mlps_and_experts_are_quantized():
    m = _model()
    graph = describe(m, verify=False)
    assert graph.adapter == "glm5_next"
    root = "model.language_model.layers"
    assert [list(x.modules) for x in graph.layers[0].groups] == [
        [f"{root}.0.mlp.gate_proj", f"{root}.0.mlp.up_proj"],
        [f"{root}.0.mlp.down_proj"],
    ]
    for g in graph.layers[1:]:
        (bank,) = g.banks
        assert bank.experts == f"{g.module}.mlp.experts" and bank.router == f"{g.module}.mlp.gate"
    kept = graph.extra["kept_bf16"]
    assert f"{root}.0.self_attn.forget_gate.f_a_proj" in kept[ATTENTION]
    assert f"{root}.2.self_attn.indexer.wq_b" in kept[ATTENTION]
    assert not [k for k in kept[ATTENTION] if "visual" in k]
    assert f"{root}.0.ffn_hc.fn" in kept[HYPER_CONNECTIONS]


def test_expert_statistics_see_the_clamped_gate():
    """A bank's down input is what the model feeds its down projection: gate and up clamped."""
    m = _model()
    graph = describe(m, verify=False)
    layer = m.model.language_model.layers[1]
    experts = layer.mlp.experts
    experts.swiglu_limit = 0.01  # small enough that the clamp changes every token
    inputs = []
    handle = experts.register_forward_pre_hook(lambda _, args: inputs.append(args))
    with (
        torch.no_grad(),
        LayerStats(layer, graph.layers[1], frozenset({"expert_hessian"}), torch.device("cpu")) as stats,
    ):
        m(input_ids=torch.randint(0, 128, (2, 8)))
    handle.remove()
    down = stats.banks[graph.layers[1].banks[0].router].result()["hessian"]["down"]
    x, index, _ = inputs[0]
    for e in range(index.max() + 1):
        rows = x[torch.where(index == e)[0]].float()
        gate, up = torch.nn.functional.linear(rows, experts.gate_up_proj[e]).chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate.clamp(max=0.01)) * up.clamp(-0.01, 0.01)
        torch.testing.assert_close(down[e], h.T @ h, rtol=1e-4, atol=1e-5)


def test_a_build_carries_the_indexer_and_writes_merged_weights_as_stored(tmp_path):
    """The carry reaches the indexed layer, and the merged ``conv1d`` goes back to its three stored tensors."""
    src = tmp_path / "src"
    _model().to(torch.bfloat16).save_pretrained(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[i % 128] * 16 for i in range(4)], save_tiny_tokenizer(src))
    out = tmp_path / "out"
    BuildConfig(
        model=str(src),
        format="w4a16-sym-g128",
        steps=[{"gptq": {}}],
        calib={"file": {"path": str(calib), "samples": 4, "seqlen": 16}},
        out=str(out),
    ).build()
    stored = load_file(src / "model.safetensors")
    written = load_file(out / "model-00001.safetensors")
    root = "model.language_model.layers"
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        key = f"{root}.0.self_attn.{name}.weight"
        assert torch.equal(written[key], stored[key])
    assert f"{root}.0.mlp.gate_proj.weight_packed" in written
    assert f"{root}.1.mlp.experts.0.down_proj.weight_packed" in written
    assert f"{root}.0.self_attn.q_proj.weight" in written
    assert f"{root}.2.self_attn.q_a_proj.weight" in written
