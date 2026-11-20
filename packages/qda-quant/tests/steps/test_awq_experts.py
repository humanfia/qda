"""Fused AWQ against independent per-expert linears and all-token calibration."""

import copy

import pytest
import torch
from qda_quant import Format, LayerContext, Target
from qda_quant.arch import describe
from qda_quant.engine.execute import Timers
from qda_quant.engine.runner import Pass, capture
from qda_quant.engine.stats import LayerStats
from qda_quant.solve.quant import Quantizer
from qda_quant.steps.awq import AWQ
from torch import nn
from torch.nn import functional as functional

from ..tiny import model


def expert_context(dtype=torch.float32, token_rows=33, kind="qwen3_moe", format_name="w4a16"):
    source = model(kind, dtype=dtype)
    graph = describe(source, verify=False)
    layer_graph = graph.layers[0]
    layer = source.get_submodule(layer_graph.module)
    rows = torch.randn(1, token_rows, 128, generator=torch.Generator().manual_seed(7), dtype=dtype)
    inputs = layer.post_attention_layernorm(rows)
    with (
        torch.no_grad(),
        LayerStats(
            layer,
            layer_graph,
            frozenset({"inputs", "expert_hessian"}),
            torch.device("cpu"),
            all_token_experts=True,
        ) as collector,
    ):
        layer.mlp(inputs)
    fmt = Format.parse(format_name)
    target = Target()
    return (
        LayerContext(
            layer,
            layer_graph,
            graph,
            fmt,
            target.plan(fmt, "dense"),
            target.plan(fmt, "moe"),
            collector.result(),
        ),
        rows,
        inputs,
    )


def unfused_output(parent, inputs, gate_up=None):
    experts = parent.experts
    logits = functional.linear(inputs, parent.gate.weight)
    probabilities = logits.float().softmax(-1)
    probabilities, indices = probabilities.topk(parent.gate.top_k, dim=-1)
    if parent.gate.norm_topk_prob:
        probabilities /= probabilities.sum(-1, keepdim=True)
    probabilities = probabilities.to(inputs.dtype)
    output = torch.zeros_like(inputs)
    weights = experts.gate_up_proj if gate_up is None else gate_up
    for expert in range(weights.shape[0]):
        gate, up = functional.linear(inputs, weights[expert]).chunk(2, dim=-1)
        hidden = experts.act_fn(gate) * up
        result = functional.linear(hidden, experts.down_proj[expert])
        importance = (probabilities * (indices == expert)).sum(-1)
        output += result * importance[:, None]
    return output


def reference_search(weight, inputs, quantizer, grid, objective):
    """Slow scale search on ordinary 2-D linears, with direct output MSE."""
    grouped = weight.float().abs().reshape(weight.shape[0], -1, quantizer.group(weight.shape[-1]))
    normalized = grouped / (grouped.amax(-1, keepdim=True) + 1e-6)
    mean_weight = normalized.reshape(weight.shape).mean(0)
    mean_input = inputs.float().abs().mean(0)
    best_loss, best = float("inf"), None
    for ratio, duo in grid:
        scale = mean_input.pow(ratio)
        if duo:
            scale /= mean_weight.pow(1 - ratio) + 1e-4
        scale = scale.clamp(min=1e-4)
        scale /= (scale.max() * scale.min()).sqrt()
        scaled = weight.float() * scale
        qscale, zero = quantizer.params(scaled)
        effective = quantizer.fake(scaled, qscale, zero) / scale
        loss = objective(effective.to(weight.dtype))
        if loss < best_loss:
            best_loss, best = loss, scale
    return best


@pytest.mark.parametrize("token_rows", [33, 257])
def test_all_token_statistics_match_unfused_experts(token_rows):
    ctx, _, inputs = expert_context(token_rows=token_rows)
    router = ctx.graph.banks[0].router
    rows = inputs.squeeze(0)
    experts = ctx.layer.mlp.experts
    torch.testing.assert_close(ctx.stats[("input_abs_sum", router)], rows.abs().sum(0))
    assert ctx.stats[("all_tokens_seen", router)] == rows.shape[0]
    for expert in range(experts.gate_up_proj.shape[0]):
        gate, up = functional.linear(rows, experts.gate_up_proj[expert]).chunk(2, dim=-1)
        hidden = experts.act_fn(gate) * up
        torch.testing.assert_close(ctx.stats[("down_abs_sum", router)][expert], hidden.abs().sum(0))
        torch.testing.assert_close(ctx.stats[("all_down", router)][expert], hidden.T @ hidden)


@torch.no_grad()
def test_all_token_statistics_include_experts_without_routed_tokens():
    ctx, _, inputs = expert_context()
    ctx.layer.mlp.gate.weight.zero_()
    with LayerStats(
        ctx.layer,
        ctx.graph,
        frozenset({"expert_hessian"}),
        torch.device("cpu"),
        all_token_experts=True,
    ) as collector:
        ctx.layer.mlp(inputs)
    stats = collector.result()
    router = ctx.graph.banks[0].router
    cold = stats[("n_tokens", router)] == 0
    assert cold.any()
    assert torch.count_nonzero(stats[("expert_hessian", router)]["down"][cold]) == 0
    assert torch.all(stats[("down_abs_sum", router)][cold] > 0)
    assert torch.all(torch.diagonal(stats[("all_down", router)][cold], dim1=-2, dim2=-1) > 0)


@torch.no_grad()
def test_fused_search_and_fold_match_unfused_reference():
    ctx, rows, inputs = expert_context(torch.float64)
    original = copy.deepcopy(ctx.layer)
    parent = original.mlp
    inputs = inputs.squeeze(0)
    awq = AWQ(n_grid=6)
    quantizer = Quantizer.of(ctx.fmt, ctx.plan_moe, "absmax")
    weight = parent.experts.gate_up_proj
    reference = unfused_output(parent, inputs)
    scale = reference_search(
        weight.flatten(0, 1),
        inputs,
        quantizer,
        awq.candidates(),
        lambda effective: (
            (unfused_output(parent, inputs, effective.reshape_as(weight)).float() - reference.float())
            .square()
            .mean()
            .item()
        ),
    )
    expected_gate_up = weight.float() * scale
    expected_down = parent.experts.down_proj.float().clone()
    for expert in range(weight.shape[0]):
        gate, up = functional.linear(inputs, weight[expert]).chunk(2, dim=-1)
        hidden = parent.experts.act_fn(gate) * up
        down = parent.experts.down_proj[expert]
        ref = functional.linear(hidden, down)
        down_scale = reference_search(
            down,
            hidden,
            quantizer,
            awq.candidates(),
            lambda effective, hidden=hidden, ref=ref: (
                (functional.linear(hidden, effective).float() - ref.float()).square().mean().item()
            ),
        )
        expected_gate_up[expert, weight.shape[1] // 2 :] /= down_scale[:, None]
        expected_down[expert] *= down_scale
    awq._search_expert_bank(ctx, ctx.graph.banks[0])
    torch.testing.assert_close(
        ctx.layer.mlp.experts.gate_up_proj.float(), expected_gate_up, rtol=2e-6, atol=1e-8
    )
    torch.testing.assert_close(ctx.layer.mlp.experts.down_proj.float(), expected_down, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(ctx.layer.mlp.gate.weight, parent.gate.weight * scale, rtol=2e-6, atol=1e-8)
    before_logits = functional.linear(inputs, parent.gate.weight)
    new_inputs = ctx.layer.post_attention_layernorm(rows).squeeze(0)
    after_logits = functional.linear(new_inputs, ctx.layer.mlp.gate.weight)
    torch.testing.assert_close(after_logits, before_logits, rtol=2e-6, atol=1e-7)
    assert torch.equal(before_logits.topk(2).indices, after_logits.topk(2).indices)
    torch.testing.assert_close(
        ctx.layer.mlp(new_inputs.unsqueeze(0)).squeeze(0), reference, rtol=3e-6, atol=1e-8
    )
    router = ctx.graph.banks[0].router
    with LayerStats(ctx.layer, ctx.graph, frozenset({"expert_hessian"}), torch.device("cpu")) as fresh:
        ctx.layer.mlp(new_inputs.unsqueeze(0))
    for key in ("in", "down"):
        measured = fresh.result()[("expert_hessian", router)][key]
        assert (
            ctx.stats[("expert_hessian", router)][key] - measured
        ).abs().max() < 3e-6 * measured.abs().max()


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_norm_fold_compensates_kept_router_and_shared_expert(dtype, monkeypatch):
    ctx, rows, inputs = expert_context(dtype=dtype, kind="qwen2_moe", format_name="w4a16-sym-g64")
    bank = ctx.graph.banks[0]
    quantized = {name for group in ctx.graph.groups for name in group.modules}
    assert not {bank.router, *bank.shared} & quantized
    parent = ctx.layer.mlp
    original = copy.deepcopy(parent)
    reference = original(inputs)
    scales = []
    fold = AWQ._fold

    def record_fold(step, context, group, scale):
        scales.append(scale.clone())
        fold(step, context, group, scale)

    monkeypatch.setattr(AWQ, "_fold", record_fold)
    AWQ(n_grid=6)._search_expert_bank(ctx, bank)
    (scale,) = scales
    assert not torch.equal(scale, torch.ones_like(scale))
    for name in ("gate", "shared_expert_gate", "shared_expert.gate_proj", "shared_expert.up_proj"):
        before = original.get_submodule(name).weight
        after = parent.get_submodule(name).weight
        assert torch.equal(after, (before.float() * scale).to(dtype))
    assert torch.equal(parent.shared_expert.down_proj.weight, original.shared_expert.down_proj.weight)
    folded_inputs = ctx.layer.post_attention_layernorm(rows)
    result = parent(folded_inputs)
    if dtype == torch.float64:
        torch.testing.assert_close(result, reference, rtol=3e-6, atol=1e-8)
        before_logits = functional.linear(inputs, original.gate.weight)
        after_logits = functional.linear(folded_inputs, parent.gate.weight)
        assert torch.equal(before_logits.topk(2).indices, after_logits.topk(2).indices)
    else:
        before_shared = original.shared_expert(inputs) * original.shared_expert_gate(inputs).sigmoid()
        after_shared = (
            parent.shared_expert(folded_inputs) * parent.shared_expert_gate(folded_inputs).sigmoid()
        )
        assert (after_shared.float() - before_shared.float()).norm() / before_shared.float().norm() < 0.01


@torch.no_grad()
def test_bf16_down_search_matches_unfused_rounded_weight_reference():
    ctx, _, inputs = expert_context(torch.bfloat16)
    experts = ctx.layer.mlp.experts
    original = experts.gate_up_proj.clone()
    down = experts.down_proj.clone()
    expected_up = original[:, 128:].clone()
    expected_down = down.clone()
    awq = AWQ()
    quantizer = Quantizer.of(ctx.fmt, ctx.plan_moe, "absmax")
    for expert in range(original.shape[0]):
        gate, up = functional.linear(inputs.squeeze(0), original[expert]).chunk(2, dim=-1)
        hidden = experts.act_fn(gate) * up
        ref = functional.linear(hidden.float(), down[expert].float())
        scale = reference_search(
            down[expert],
            hidden,
            quantizer,
            awq.candidates(),
            lambda effective, hidden=hidden, ref=ref: (
                (functional.linear(hidden.float(), effective.float()) - ref).square().mean().item()
            ),
        )
        expected_up[expert] = (original[expert, 128:].float() / scale[:, None]).to(torch.bfloat16)
        expected_down[expert] = (down[expert].float() * scale).to(torch.bfloat16)
    awq._search_expert_up_down(ctx, ctx.graph.banks[0], experts, 128)
    assert torch.equal(experts.gate_up_proj[:, :128], original[:, :128])
    assert torch.equal(experts.gate_up_proj[:, 128:], expected_up)
    assert torch.equal(experts.down_proj, expected_down)


@torch.no_grad()
def test_searched_mappings_equal_llm_compressor(monkeypatch):
    reference_module = pytest.importorskip("llmcompressor.modifiers.transform.awq.base")
    mappings_module = pytest.importorskip("llmcompressor.modifiers.transform.awq.mappings")
    ctx, _, inputs = expert_context()
    unfused = nn.Module()
    unfused.model = nn.Module()
    unfused.model.layers = nn.ModuleList([copy.deepcopy(ctx.layer)])
    bank = unfused.model.layers[0].mlp.experts
    unfused_experts = nn.ModuleList()
    for expert in range(bank.gate_up_proj.shape[0]):
        item = nn.Module()
        item.gate_proj = nn.Linear(128, 128, bias=False)
        item.up_proj = nn.Linear(128, 128, bias=False)
        item.down_proj = nn.Linear(128, 128, bias=False)
        item.gate_proj.weight.copy_(bank.gate_up_proj[expert, :128])
        item.up_proj.weight.copy_(bank.gate_up_proj[expert, 128:])
        item.down_proj.weight.copy_(bank.down_proj[expert])
        unfused_experts.append(item)
    unfused.model.layers[0].mlp.experts = unfused_experts
    for module in unfused.modules():
        if isinstance(module, nn.Linear):
            module.quantization_scheme = object()
    reference = reference_module.AWQModifier(mappings=mappings_module._moe_default_mappings)
    reference._set_resolved_mappings(unfused)
    expected = {
        (mapping.smooth_name, frozenset(mapping.balance_names)) for mapping in reference._resolved_mappings
    }
    actual = set()

    def record_fold(self, context, group, scale):
        readers = context.model_graph.extra["readers"].get(group.modules[0], group.modules)
        expanded = []
        for reader in readers:
            if reader == ctx.graph.banks[0].experts:
                expanded.extend(
                    f"{reader}.{expert}.{projection}"
                    for expert in range(8)
                    for projection in ("gate_proj", "up_proj")
                )
            else:
                expanded.append(reader)
        actual.add((group.norm or group.prev, frozenset(expanded)))

    def record_up_down(self, context, bank, module, gate_rows):
        for expert in range(module.gate_up_proj.shape[0]):
            prefix = f"{bank.experts}.{expert}"
            actual.add((prefix + ".up_proj", frozenset({prefix + ".down_proj"})))

    monkeypatch.setattr(AWQ, "_fold", record_fold)
    monkeypatch.setattr(AWQ, "_search_expert_up_down", record_up_down)
    monkeypatch.setattr(AWQ, "_search", lambda *args: torch.ones(128))
    for group in ctx.graph.groups:
        ctx.stats[("inputs", group.modules[0])] = inputs.squeeze(0)
    AWQ(n_grid=3).apply_layer(ctx)
    assert actual == expected


def test_search_failure_restores_bank_weights(monkeypatch):
    ctx, _, _ = expert_context()
    before = ctx.layer.mlp.experts.gate_up_proj.clone()
    own = ctx.layer.mlp.forward
    calls = 0

    def fail_after_reference(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("candidate failure")
        return own(*args)

    monkeypatch.setattr(ctx.layer.mlp, "forward", fail_after_reference)
    with pytest.raises(RuntimeError, match="candidate failure"):
        AWQ()._search_expert_bank(ctx, ctx.graph.banks[0])
    assert torch.equal(before, ctx.layer.mlp.experts.gate_up_proj)


@pytest.mark.parametrize("objective", ["parent", "linear_out"])
@torch.no_grad()
def test_engine_collects_expert_awq_statistics_and_preserves_float_model(objective):
    source = model("qwen3_moe", dtype=torch.float64)
    graph = describe(source, verify=False)
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(8))
    reference = source(input_ids=ids).logits
    original = source.model.layers[0].mlp.experts.gate_up_proj.clone()
    device = torch.device("cpu")
    fmt = Format.parse("w4a16")
    target = Target()
    seen = []

    def context(layer, layer_graph, statistics):
        bank = layer_graph.banks[0]
        assert statistics[("all_tokens_seen", bank.router)] == 32
        assert statistics[("inputs", bank.router)].shape == (32, 128)
        seen.append(layer_graph.index)
        return LayerContext(
            layer, layer_graph, graph, fmt, target.plan(fmt, "dense"), target.plan(fmt, "moe"), statistics
        )

    slots = capture(source, graph, [ids], device)
    runner = Pass(
        source, graph, [AWQ(n_grid=6, objective=objective)], device, make_ctx=context, timers=Timers()
    )
    runner.run(slots)
    assert seen == [0, 1]
    assert not torch.equal(source.model.layers[0].mlp.experts.gate_up_proj, original)
    actual = source(input_ids=ids).logits
    assert (actual - reference).abs().max() < 1e-5 * reference.abs().max()
