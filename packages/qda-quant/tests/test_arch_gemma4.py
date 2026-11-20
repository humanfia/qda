import pytest
import torch
from qda_quant.arch import describe
from qda_quant.engine.stats import LayerStats, reference_expert_stats
from transformers import AutoModelForCausalLM, Gemma4TextConfig


@pytest.mark.parametrize("meta", [False, True])
@pytest.mark.parametrize("enable_moe", [False, True])
def test_gemma4_nested_router_and_separate_expert_input(meta, enable_moe):
    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size=128,
        max_position_embeddings=64,
        enable_moe_block=enable_moe,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        hidden_size_per_layer_input=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    with torch.device("meta" if meta else "cpu"):
        model = AutoModelForCausalLM.from_config(config).eval()
    graph = describe(model, verify=not meta)
    if not meta:
        assert graph.verified and graph.extra["dropped"] == []
    if not enable_moe:
        assert graph.adapter is None and all(not layer.banks for layer in graph.layers)
        return
    assert graph.adapter == "gemma4_experts"
    for layer in graph.layers:
        (bank,) = layer.banks
        assert bank.experts == f"{layer.module}.experts"
        assert bank.router == f"{layer.module}.router"
        assert bank.fused and bank.top_k == 2 and bank.shared == ()
        # the router multiplies each selected expert's weight by its own per-expert scale after top-k, so
        # vLLM routes outside the fused expert kernel (RoutingMethodType.Custom)
        assert bank.custom_router
        assert graph.extra["readers"][f"{layer.module}.pre_feedforward_layernorm_2"] == (bank.experts,)
        assert graph.extra["readers"][f"{layer.module}.pre_feedforward_layernorm"] == (
            f"{layer.module}.mlp.gate_proj",
            f"{layer.module}.mlp.up_proj",
        )
        assert all(f"{layer.module}.router.proj" not in group.modules for group in layer.groups)
    if not meta:
        layer = model.model.layers[0]
        inputs = []
        handle = layer.experts.register_forward_pre_hook(lambda module, args: inputs.append(args))
        with (
            torch.no_grad(),
            LayerStats(layer, graph.layers[0], frozenset({"expert_hessian"}), torch.device("cpu")) as stats,
        ):
            model(torch.randint(0, config.vocab_size, (2, 8)))
        handle.remove()
        result = stats.banks[graph.layers[0].banks[0].router].result()
        hidden, indices, weights = inputs[0]
        reference = reference_expert_stats(layer.experts, hidden, indices, weights, False)
        torch.testing.assert_close(result["hessian"]["in"], reference["in"])
        torch.testing.assert_close(result["hessian"]["down"], reference["down"])
        assert int(result["n_tokens"].sum()) == 32
