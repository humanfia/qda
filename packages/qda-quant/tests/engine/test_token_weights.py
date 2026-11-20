"""Token-weight equivalence, determinism, routing and residency tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.arch import describe
from qda_quant.engine.runner import LayerStats, capture
from qda_quant.engine.stats import HessianSetting
from qda_quant.engine.stores import STREAM
from qda_quant.engine.token_weights import anchor_residency, fisher_trace, guided
from qda_quant.step import Refused
from transformers import AutoModelForCausalLM, Qwen2MoeConfig

from ..engine_util import tiny_qwen3, tiny_qwen3_moe


def test_token_weights_are_deterministic() -> None:
    """Repeated anchor passes produce bit-identical normalized weights."""
    model = tiny_qwen3(layers=1, hidden=32, inter=48)
    generator = torch.Generator().manual_seed(3)
    batches = [torch.randint(0, 128, (1, length), generator=generator) for length in (8, 12)]
    fisher_a = fisher_trace(model, batches, torch.device("cpu"))
    fisher_b = fisher_trace(model, batches, torch.device("cpu"))
    guided_a = guided(model, batches, torch.device("cpu"))
    guided_b = guided(model, batches, torch.device("cpu"))
    for first, second in zip(fisher_a, fisher_b, strict=True):
        assert torch.equal(first, second)
    for first, second in zip(guided_a, guided_b, strict=True):
        assert first.keys() == second.keys()
        for key in first:
            assert torch.equal(first[key], second[key])
            assert first[key][0, -1] == 0
    assert torch.isclose(torch.cat([weight.flatten() for weight in fisher_a]).mean(), torch.ones(()))
    for key in guided_a[0]:
        assert torch.isclose(torch.cat([weight[key].flatten() for weight in guided_a]).mean(), torch.ones(()))


def test_guided_matches_old_one_layer_reference() -> None:
    """Module-output gradients give the old per-group weights and one-layer Hessians."""
    # Reference math: qda/quant/guided.py, gradient_norms and normalise.
    model = tiny_qwen3(layers=1, hidden=32, inter=48)
    graph = describe(model, verify=False)
    generator = torch.Generator().manual_seed(4)
    batches = [torch.randint(0, 128, (1, length), generator=generator) for length in (8, 12)]
    raw = {group.modules[0]: [] for group in graph.layers[0].groups}
    model.requires_grad_(False)
    for ids in batches:
        outputs = {}
        names = [name for group in graph.layers[0].groups for name in group.modules]
        handles = [
            model.get_submodule(name).register_forward_hook(
                lambda _module, _args, output, name=name, outputs=outputs: outputs.__setitem__(name, output)
            )
            for name in names
        ]
        handles.append(
            model.get_input_embeddings().register_forward_hook(
                lambda _module, _args, output: output.requires_grad_(True)
            )
        )
        logits = model(input_ids=ids, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="sum")
        gradients = dict(
            zip(names, torch.autograd.grad(loss, tuple(outputs[name] for name in names)), strict=True)
        )
        for handle in handles:
            handle.remove()
        for group in graph.layers[0].groups:
            score = sum(gradients[name].float().square().sum(-1).flatten() for name in group.modules)
            raw[group.modules[0]].append(score)
    expected = [{} for _ in batches]
    for key, scores in raw.items():
        score = torch.cat(scores).clamp_min(0)
        weights = (score / score.double().mean()).float().clamp_max(50)
        for batch, weight, ids in zip(
            expected, weights.split([ids.numel() for ids in batches]), batches, strict=True
        ):
            batch[key] = weight.reshape_as(ids)
    got = guided(model, batches, torch.device("cpu"), graph=graph)
    for actual, wanted in zip(got, expected, strict=True):
        for key in wanted:
            torch.testing.assert_close(actual[key], wanted[key])
    group_keys = list(raw)
    assert not torch.equal(got[0][group_keys[0]], got[0][group_keys[-1]])
    _check_weighted_hessians(model, batches, got, expected)


def test_fisher_matches_old_one_layer_reference() -> None:
    """Fisher weights and one layer's Hessians match the old stream-wide formula."""
    # Reference math: qda/quant/tokweights.py, fisher_trace, causal_window_mean and weights_from_s.
    model = tiny_qwen3(layers=1, hidden=32, inter=48)
    generator = torch.Generator().manual_seed(5)
    batches = [torch.randint(0, 128, (1, length), generator=generator) for length in (8, 23)]
    scores = []
    for ids in batches:
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
        raw = 1.0 - torch.softmax(logits, -1).pow(2).sum(-1)
        cumulative = torch.cat([raw.new_zeros(1), torch.cumsum(raw.double(), 0)])
        positions = torch.arange(len(raw))
        high = torch.clamp(positions + 16, max=len(raw))
        scores.append(
            ((cumulative[high] - cumulative[positions]) / (high - positions).double()).to(raw.dtype)
        )
    mean = torch.cat(scores).mean()
    expected = [0.25 + 0.75 * score / mean.clamp(min=1e-20) for score in scores]
    got = fisher_trace(model, batches, torch.device("cpu"))
    for actual, wanted in zip(got, expected, strict=True):
        torch.testing.assert_close(actual.flatten(), wanted, rtol=0, atol=0)
    _check_weighted_hessians(model, batches, got, expected)


@pytest.mark.parametrize("method", [guided, fisher_trace])
@pytest.mark.parametrize("router_weighted", [False, True])
@pytest.mark.parametrize("buffer_rows", [0, 64])
def test_shared_expert_weights_and_hessians_match_reference(method, router_weighted, buffer_rows) -> None:
    """Shared norm groups weight bank inputs; only routed-output gradients weight bank down inputs."""
    torch.manual_seed(4)
    config = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "embed" not in name:
                parameter.normal_(0, parameter.shape[-1] ** -0.5)
    model.requires_grad_(False)
    graph = describe(model, verify=False, keep=frozenset())  # the shared expert quantized
    layer_graph = graph.layers[0]
    bank = layer_graph.banks[0]
    assert bank.shared
    experts = model.get_submodule(bank.experts)
    shared = [name for name in bank.shared if name.endswith(("gate_proj", "up_proj"))]
    names = [*shared, bank.experts]
    generator = torch.Generator().manual_seed(15)
    batches = [torch.randint(0, 128, (2, 8), generator=generator) for _ in range(2)]
    raw_input, raw_down, routed_batches = [], [], []
    for ids in batches:
        outputs = {}
        handles = [
            model.get_submodule(name).register_forward_hook(
                lambda _module, _args, output, name=name, outputs=outputs: outputs.__setitem__(name, output)
            )
            for name in names
        ]
        handles.append(
            model.get_input_embeddings().register_forward_hook(
                lambda _module, _args, output: output.requires_grad_(True)
            )
        )
        handles.append(
            experts.register_forward_pre_hook(
                lambda _module, args: routed_batches.append(tuple(value.detach().clone() for value in args))
            )
        )
        try:
            logits = model(input_ids=ids, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, 128), ids[:, 1:].reshape(-1), reduction="sum"
            )
            gradients = torch.autograd.grad(loss, tuple(outputs[name] for name in names))
        finally:
            for handle in handles:
                handle.remove()
        scores = [gradient.float().square().sum(-1).flatten() for gradient in gradients]
        raw_input.append(sum(scores))
        raw_down.append(scores[-1])
    if method is guided:
        expected = []
        for parts in (raw_input, raw_down):
            score = torch.cat(parts).clamp_min(0)
            expected.append((score / score.double().mean()).float().clamp_max(50).split(16))
        expected_input, expected_down = expected
    else:
        scores = []
        with torch.no_grad():
            for ids in batches:
                logits = model(input_ids=ids, use_cache=False).logits.float()
                raw = 1 - torch.softmax(logits, -1).square().sum(-1)
                scores.append(
                    torch.stack([raw[:, pos : pos + 16].double().mean(-1) for pos in range(8)], -1).float()
                )
        mean = torch.cat([score.flatten() for score in scores]).mean()
        expected_input = [(0.25 + 0.75 * score / mean).flatten() for score in scores]
        expected_down = expected_input
    kw = {"graph": graph} if method is guided else {}  # guided reads the groups: the shared expert's included
    actual = method(model, batches, torch.device("cpu"), **kw)
    repeated = method(model, batches, torch.device("cpu"), **kw)
    expected_hessian = {
        "in": torch.zeros(4, 32, 32),
        "down": torch.zeros(4, 16, 16),
    }
    layer = model.get_submodule(layer_graph.module)
    current = [None]
    with (
        torch.no_grad(),
        LayerStats(
            layer,
            layer_graph,
            frozenset({"expert_hessian"}),
            torch.device("cpu"),
            router_weighted=router_weighted,
            hessian=HessianSetting(buffer_rows=buffer_rows),
            weights=lambda: current[0],
        ) as stats,
    ):
        for index, (values, weights) in enumerate(zip(routed_batches, actual, strict=True)):
            routed_inputs, indices, router_weights = values
            current[0] = weights
            if isinstance(weights, dict):
                input_weights = weights[bank.router].flatten()
                down_weights = weights[bank.experts + ".down_proj"].flatten()
                for key in weights:
                    assert torch.equal(weights[key], repeated[index][key])
                assert not torch.equal(input_weights, down_weights)
            else:
                input_weights = down_weights = weights.flatten()
                assert torch.equal(weights, repeated[index])
            torch.testing.assert_close(input_weights, expected_input[index], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(down_weights, expected_down[index], rtol=1e-6, atol=1e-6)
            experts(routed_inputs, indices, router_weights)
            for expert_id in range(4):
                token, position = torch.where(indices == expert_id)
                routed = routed_inputs[token]
                gate, up = (routed @ experts.gate_up_proj[expert_id].T).chunk(2, -1)
                hidden = experts.act_fn(gate) * up
                router = router_weights[token, position] if router_weighted else 1.0
                expected_hessian["in"][expert_id] += routed.T @ (
                    routed * (expected_input[index][token] * router)[:, None]
                )
                expected_hessian["down"][expert_id] += hidden.T @ (
                    hidden * (expected_down[index][token] * router)[:, None]
                )
    result = stats.result()[("expert_hessian", bank.router)]
    for key in expected_hessian:
        torch.testing.assert_close(result[key], expected_hessian[key], rtol=1e-5, atol=1e-5)


def _check_weighted_hessians(model, batches, weights, expected) -> None:
    """Compare the runner's hooks with weighted Gram matrices at every group in a layer."""
    graph = describe(model, verify=False)
    layer_graph = graph.layers[0]
    layer = model.get_submodule(layer_graph.module)
    inputs: dict[str, list[torch.Tensor]] = {group.modules[0]: [] for group in layer_graph.groups}
    handles = [
        model.get_submodule(key).register_forward_pre_hook(
            lambda _module, args, key=key: inputs[key].append(args[0].detach().reshape(-1, args[0].shape[-1]))
        )
        for key in inputs
    ]
    with torch.no_grad():
        for ids in batches:
            model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()
    current = [None]
    slots = capture(model, graph, batches, torch.device("cpu"))
    with (
        torch.no_grad(),
        LayerStats(
            layer, layer_graph, frozenset({"hessian"}), torch.device("cpu"), weights=lambda: current[0]
        ) as stats,
    ):
        for slot, weight in zip(slots, weights, strict=True):
            current[0] = weight
            layer(slot.h, *slot.calls[0][0], **slot.calls[0][1])
    for key, samples in inputs.items():
        reference = sum(
            sample.T @ (sample * (weight[key] if isinstance(weight, dict) else weight).reshape(-1, 1))
            for sample, weight in zip(samples, expected, strict=True)
        )
        torch.testing.assert_close(stats.result()[("hessian", key)], reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["guided", "fisher_trace"])
def test_unfused_expert_token_weights_are_refused(method) -> None:
    """Unsupported expert routing fails at plan time instead of dropping token weights."""
    from qda_quant.format import Format
    from qda_quant.step import Refused
    from qda_quant.target import Target

    graph = describe(tiny_qwen3_moe(layers=1), verify=False)
    layer = graph.layers[0]
    bank = replace(layer.banks[0], experts=((("gate_proj", "up_proj"), "down_proj"),))
    graph = replace(graph, layers=(replace(layer, banks=(bank,)),))
    fmt = Format.parse("w4a16-sym-g32")
    with pytest.raises(Refused, match="token weights currently require fused expert banks"):
        REGISTRY[method]().check(graph, fmt, Target().plan(fmt, "dense"))


def test_guided_restores_model_flags_and_hooks_after_failure(monkeypatch) -> None:
    """A failed anchor calculation leaves no hooks or changed training/gradient flags."""
    model = tiny_qwen3(layers=1, hidden=32, inter=48).train()
    model.model.layers[0].eval()
    first_parameter = next(model.parameters())
    first_parameter.requires_grad_(False)
    gradient_flags = [parameter.requires_grad for parameter in model.parameters()]
    training_flags = [module.training for module in model.modules()]
    hook_counts = [len(module._forward_hooks) for module in model.modules()]

    def fail(*args, **kwargs):
        raise RuntimeError("anchor forward failed")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(Refused, match="anchor forward failed"):  # the probe failure is one refusal
        guided(model, [torch.ones((1, 8), dtype=torch.long)], torch.device("cpu"))
    assert [parameter.requires_grad for parameter in model.parameters()] == gradient_flags
    assert [module.training for module in model.modules()] == training_flags
    assert [len(module._forward_hooks) for module in model.modules()] == hook_counts


@pytest.mark.gpu
@pytest.mark.parametrize("method", [guided, fisher_trace])
def test_streamed_anchor_matches_resident_weights(method) -> None:
    """Host-offloaded autograd and ordinary resident anchor passes compute the same weights."""
    resident = tiny_qwen3(layers=2, hidden=32, inter=48).to("cuda")
    streamed = tiny_qwen3(layers=2, hidden=32, inter=48)
    graph = describe(streamed, verify=False)
    generator = torch.Generator().manual_seed(10)
    batches = [torch.randint(0, 128, (1, 12), generator=generator) for _ in range(2)]
    expected = method(resident, batches, torch.device("cuda"))
    with anchor_residency(streamed, graph, torch.device("cuda"), store=STREAM):
        actual = method(streamed, batches, torch.device("cuda"))
    for first, second in zip(actual, expected, strict=True):
        if isinstance(first, dict):
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)
        else:
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    assert all(parameter.device.type == "cpu" for parameter in streamed.parameters())


@pytest.mark.gpu
@pytest.mark.parametrize("method", [guided, fisher_trace])
def test_gpu_token_weights_are_bit_identical(method) -> None:
    """Repeated CUDA anchor passes produce bit-identical weights for every token and group."""
    model = tiny_qwen3(layers=2, hidden=64, inter=96).to("cuda")
    generator = torch.Generator().manual_seed(11)
    batches = [torch.randint(0, 128, (1, 64), generator=generator) for _ in range(2)]
    first = method(model, batches, torch.device("cuda"))
    second = method(model, batches, torch.device("cuda"))
    for first_batch, second_batch in zip(first, second, strict=True):
        if isinstance(first_batch, dict):
            for key in first_batch:
                assert torch.equal(first_batch[key], second_batch[key])
        else:
            assert torch.equal(first_batch, second_batch)


@pytest.mark.parametrize("router_weighted", [False, True])
def test_token_weights_reach_routed_expert_hessians(router_weighted) -> None:
    """Each expert receives its routed tokens' weights, optionally multiplied by router weights."""
    model = tiny_qwen3_moe(layers=1)
    graph = describe(model, verify=False)
    lg = graph.layers[0]
    bank = lg.banks[0]
    layer = model.get_submodule(lg.module)
    experts = model.get_submodule(bank.experts)
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(12, model.config.hidden_size, generator=generator)
    _, router_weights, indices = layer.mlp.gate(x)
    token_weights = torch.linspace(0.25, 2.0, len(x))
    expected_in, expected_down = [], []
    for expert_id in range(experts.gate_up_proj.shape[0]):
        token, position = torch.where(indices == expert_id)
        routed = x[token]
        gate, up = (routed @ experts.gate_up_proj[expert_id].T).chunk(2, -1)
        hidden = experts.act_fn(gate) * up
        weights = token_weights[token]
        if router_weighted:
            weights = weights * router_weights[token, position]
        expected_in.append(routed.T @ (routed * weights[:, None]))
        expected_down.append(hidden.T @ (hidden * weights[:, None]))
    with (
        torch.no_grad(),
        LayerStats(
            layer,
            lg,
            frozenset({"expert_hessian"}),
            torch.device("cpu"),
            router_weighted=router_weighted,
            weights=lambda: token_weights,
        ) as stats,
    ):
        experts(x, indices, router_weights)
    result = stats.result()[("expert_hessian", bank.router)]
    torch.testing.assert_close(result["in"], torch.stack(expected_in))
    torch.testing.assert_close(result["down"], torch.stack(expected_down))
