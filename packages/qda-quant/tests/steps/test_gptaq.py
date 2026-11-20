"""Cross-Hessian, paired-stream and GPTAQ reference tests."""

from __future__ import annotations

import copy

import pytest
import torch
from qda_quant.arch import describe
from qda_quant.engine.execute import Timers
from qda_quant.engine.runner import LayerStats, Pass, capture, rel
from qda_quant.engine.stats import GroupStats
from qda_quant.format import Format
from qda_quant.solve import solve
from qda_quant.step import LayerContext, Step
from qda_quant.steps.awq import AWQ
from qda_quant.steps.gptaq import GPTAQ, refit
from qda_quant.steps.gptq import GPTQ
from qda_quant.target import Target

from ..engine_util import rtn_, tiny_qwen3


def layer_context(
    fmt_name: str = "w4a16-sym-g32", hidden: int = 64, inter: int = 96, tokens: int = 128
) -> LayerContext:
    """A bf16 layer with full-rank paired calibration statistics."""
    model = tiny_qwen3(layers=1, hidden=hidden, inter=inter).to(torch.bfloat16)
    graph = describe(model, verify=False)
    layer = graph.layers[0]
    fmt = Format.parse(fmt_name)
    plan = Target().plan(fmt, "dense")
    generator = torch.Generator().manual_seed(13)
    stats = {}
    for group in layer.groups:
        width = model.get_submodule(group.modules[0]).weight.shape[-1]
        quant_inputs = torch.randn(tokens, width, generator=generator)
        float_inputs = quant_inputs + 0.125 * torch.randn(tokens, width, generator=generator)
        stats[("hessian", group.modules[0])] = quant_inputs.T @ quant_inputs
        stats[("cross_hessian", group.modules[0])] = (float_inputs - quant_inputs).T @ quant_inputs
        stats[("n_tokens", group.modules[0])] = len(quant_inputs)
    return LayerContext(
        model.model.layers[0], layer, graph, fmt, plan, None, stats, device=torch.device("cpu")
    )


def test_gptaq_rounds_refit_before_scale_fitting() -> None:
    """The old bf16 write occurs before scales and codes are chosen on every group in a layer."""
    # Reference math: qda/quant/algos/gptaq.py, gptaq_refit and GPTAQModifier._refit_modules.
    ctx = layer_context()
    expected = {}
    for unit in GPTAQ().units(ctx):
        weight = unit.weight().float()
        old_hessian = ctx.stats[("hessian", unit.key)] * 2
        old_cross = ctx.stats[("cross_hessian", unit.key)] * -2
        damped = old_hessian.clone()
        diagonal = torch.arange(damped.shape[0])
        damped[diagonal, diagonal] += 0.01 * torch.mean(torch.diag(damped))
        factor = torch.linalg.cholesky(damped)
        correction = torch.cholesky_solve((weight @ old_cross).transpose(-1, -2), factor).transpose(-1, -2)
        refitted = (weight - 0.3 * correction).to(torch.bfloat16).float()
        quantizer = unit.quantizer
        scales, zero = quantizer.params(refitted)
        hessian = solve.prepare_hessian(ctx.stats[("hessian", unit.key)], 128)
        codes = solve.gptq(refitted, hessian, quantizer, scales, zero, impl="torch")
        expected[unit.key] = quantizer.dequant(codes, scales, zero).to(torch.bfloat16)
    GPTAQ(impl="torch").apply_layer(ctx)
    for unit in GPTAQ().units(ctx):
        assert torch.equal(unit.weight(), expected[unit.key])


@pytest.mark.parametrize(
    "settings",
    [{"clip": "grid3", "grid_k": 0}, {"clip": "grid3", "grid_k": 2}, {"clip": "hmse", "actorder": "none"}],
)
def test_gptaq_solves_the_refit_with_its_gptq_settings(settings) -> None:
    """A nonzero refit followed by GPTQ with the same settings: none falls back to a default."""
    refitted = layer_context("w4a8-sym-g128", hidden=128, inter=256, tokens=512)
    reference = layer_context("w4a8-sym-g128", hidden=128, inter=256, tokens=512)
    for unit in GPTQ().units(reference):
        raw, count = unit.hessian()
        cross = reference.stats[("cross_hessian", unit.key)] / count
        unit.put_weights(refit(unit.weight().float(), solve.prepare_hessian(raw, count), cross, 0.3, 0.01))
    GPTQ(impl="torch", **settings).apply_layer(reference)
    GPTAQ(alpha=0.3, impl="torch", **settings).apply_layer(refitted)
    for name, qparams in reference.qparams.items():
        assert torch.equal(qparams["codes"], refitted.qparams[name]["codes"])
        assert torch.equal(qparams["scale"], refitted.qparams[name]["scale"])


def test_gptaq_alpha_zero_is_bit_identical_to_gptq() -> None:
    """A disabled refit never reads C and produces exactly the ordinary GPTQ checkpoint tensors."""
    plain, asymmetric = layer_context(), layer_context()
    for key in asymmetric.stats:
        if key[0] == "cross_hessian":
            asymmetric.stats[key].fill_(torch.nan)
    GPTQ(impl="torch").apply_layer(plain)
    step = GPTAQ(alpha=0, impl="torch")
    assert step.needs == GPTQ.needs
    step.apply_layer(asymmetric)
    for name, parameter in plain.layer.named_parameters():
        assert torch.equal(parameter, asymmetric.layer.get_parameter(name))
    for name, qparams in plain.qparams.items():
        assert torch.equal(qparams["codes"], asymmetric.qparams[name]["codes"])
        assert torch.equal(qparams["scale"], asymmetric.qparams[name]["scale"])


def test_gptaq_refit_matches_direct_solve() -> None:
    """The helper applies the documented damped least-squares correction."""
    weight = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    hessian = torch.tensor([[[2.0, 0.25], [0.25, 3.0]]])
    cross = torch.tensor([[[0.5, 0.0], [0.0, 0.25]]])
    damp = 0.1
    got = refit(weight, hessian, cross, 1.0, damp)
    damped = (
        hessian + torch.eye(2).unsqueeze(0) * hessian.diagonal(dim1=-2, dim2=-1).mean(-1).view(1, 1, 1) * damp
    )
    expected = weight + torch.linalg.solve(damped, torch.matmul(weight, cross).transpose(-1, -2)).transpose(
        -1, -2
    )
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("kind", ["attn_in", "mlp_in", "mlp_out"])
def test_awq_cross_hessian_matches_separate_reforwards(kind) -> None:
    """A norm or producer fold rescales C into the basis of both new activation streams."""
    model = tiny_qwen3(layers=2).double()
    reference = copy.deepcopy(model)
    graph = describe(model, verify=False)
    layer_graph = graph.layers[1]
    group = next(group for group in layer_graph.groups if group.kind == kind)
    key = group.modules[0]
    ids = torch.randint(0, 128, (1, 128), generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        for parameter in model.model.layers[0].parameters():
            if parameter.ndim == 2:
                rtn_(parameter, bits=3)

    def inputs(current):
        values = []
        handle = current.get_submodule(key).register_forward_pre_hook(
            lambda _module, args: values.append(args[0].detach().clone().flatten(0, -2))
        )
        try:
            with torch.no_grad():
                current(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
        return values[0]

    float_inputs, quant_inputs = inputs(reference), inputs(model)
    accumulator = GroupStats(
        quant_inputs.shape[-1],
        torch.device("cpu"),
        frozenset({"hessian", "cross_hessian", "inputs"}),
        dtype="fp32",
    )
    accumulator.add(quant_inputs, xf=float_inputs)
    stats = {(name, key): value for name, value in accumulator.result().items()}
    original_cross = stats[("cross_hessian", key)].clone()
    fmt = Format.parse("w4a16-sym-g32")
    layer = model.get_submodule(layer_graph.module)
    context = LayerContext(layer, layer_graph, graph, fmt, Target().plan(fmt, "dense"), None, stats)
    scales = torch.linspace(0.5, 2.0, quant_inputs.shape[-1])
    AWQ()._fold(context, group, scales)
    reference.get_submodule(layer_graph.module).load_state_dict(layer.state_dict())
    float_after, quant_after = inputs(reference).float(), inputs(model).float()
    expected_cross = (float_after - quant_after).T @ quant_after
    expected_hessian = quant_after.T @ quant_after
    assert torch.count_nonzero(expected_cross) > 0
    assert not torch.equal(stats[("cross_hessian", key)], original_cross)
    torch.testing.assert_close(stats[("hessian", key)], expected_hessian, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats[("cross_hessian", key)], expected_cross, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("statistic", ["hessian", "cross"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_gptaq_refit_refuses_nonfinite_statistics(statistic, invalid) -> None:
    """Non-finite calibration statistics stop the build, as in the old qda (it raised NotFinite)."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hessian, cross = torch.eye(2), torch.eye(2)
    (hessian if statistic == "hessian" else cross)[0, 0] = invalid
    with pytest.raises(RuntimeError, match="non-finite"):
        refit(weight, hessian, cross, 0.3, 0.01)


def test_gptaq_refit_skips_cholesky_failure(caplog) -> None:
    """An indefinite damped Hessian skips only the refit, as in the old modifier."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hessian = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    result = refit(weight, hessian, torch.eye(2), 0.3, 0.01)
    assert torch.equal(result, weight)
    assert "refit skipped: damped Hessian" in caplog.text


def test_gptaq_nonfinite_cross_stops_the_layer() -> None:
    """An invalid cross term is an error, never a silent fall back to plain GPTQ."""
    asymmetric = layer_context()
    for key, value in asymmetric.stats.items():
        if key[0] == "cross_hessian":
            value.fill_(torch.nan)
    with pytest.raises(RuntimeError, match="non-finite"):
        GPTAQ(impl="torch").apply_layer(asymmetric)


def test_cross_hessian_matches_tiny_model_direct_computation() -> None:
    """The cross statistic equals the direct float-minus-quantized Gram matrix."""
    model = tiny_qwen3(layers=1, hidden=32, inter=48)
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        x_float = model.model.embed_tokens(ids).reshape(-1, model.config.hidden_size)
    x_quant = x_float + 0.125
    stats = GroupStats(
        x_quant.shape[-1], torch.device("cpu"), frozenset({"hessian", "cross_hessian"}), dtype="fp32"
    )
    stats.add(x_quant, xf=x_float)
    expected = (x_float - x_quant).T @ x_quant
    torch.testing.assert_close(stats.result()["cross_hessian"], expected)


def test_paired_stream_matches_old_gptaq_one_layer_reference() -> None:
    """Real quantized-prefix inputs give the old cross term, Hessian and refit on one layer."""
    # Reference math: qda/quant/algos/gptaq.py, accumulate_cross and gptaq_refit.
    model = tiny_qwen3(layers=2, hidden=32, inter=48)
    graph = describe(model, verify=False)
    batches = [torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(8))]
    keys = [group.modules[0] for group in graph.layers[1].groups]

    def collect_inputs() -> dict[str, torch.Tensor]:
        inputs = {}
        handles = [
            model.get_submodule(key).register_forward_pre_hook(
                lambda _module, args, key=key: inputs.__setitem__(
                    key, args[0].detach().clone().flatten(0, -2)
                )
            )
            for key in keys
        ]
        with torch.no_grad():
            model(input_ids=batches[0], use_cache=False)
        for handle in handles:
            handle.remove()
        return inputs

    float_inputs = collect_inputs()
    seen = []

    class Probe(Step):
        needs = frozenset({"hessian", "cross_hessian"})

        def apply_layer(self, ctx: LayerContext) -> None:
            if ctx.graph.index == 0:
                for group in ctx.graph.groups:
                    assert torch.count_nonzero(ctx.stats[("cross_hessian", group.modules[0])]) == 0
                    for name in group.modules:
                        rtn_(ctx.layer.get_submodule(rel(name, ctx.graph.module)).weight.data)
                return
            quant_inputs = collect_inputs()
            for key in keys:
                xq, xf = quant_inputs[key], float_inputs[key]
                old_hessian = 2 * xq.T @ xq
                old_cross = 2 * (xq - xf).T @ xq
                hessian = ctx.stats[("hessian", key)]
                cross = ctx.stats[("cross_hessian", key)]
                torch.testing.assert_close(hessian * 2, old_hessian)
                torch.testing.assert_close(cross * -2, old_cross)
                torch.testing.assert_close(cross, (xf - xq).T @ xq)
                assert torch.count_nonzero(cross) > 0
                weight = model.get_submodule(key).weight.detach().float()
                damped = old_hessian + torch.eye(xq.shape[-1]) * old_hessian.diagonal().mean() * 0.01
                factor = torch.linalg.cholesky(damped)
                reference = weight - 0.3 * torch.cholesky_solve((weight @ old_cross).T, factor).T
                actual = refit(weight, hessian, cross, 0.3, 0.01)
                torch.testing.assert_close(actual, reference)
                seen.append(key)

    slots = capture(model, graph, batches, torch.device("cpu"), keep_float=True)
    runner = Pass(
        model,
        graph,
        [Probe()],
        torch.device("cpu"),
        lambda layer, lg, stats: LayerContext(layer, lg, graph, None, None, None, stats),
        Timers(),
    )
    runner.run(slots)
    assert seen == keys


def test_capture_does_not_allocate_float_buffer_without_cross_hessian() -> None:
    """Ordinary layer passes keep no float activation stream."""
    model = tiny_qwen3(layers=1, hidden=32, inter=48)
    graph = describe(model, verify=False)
    ids = [torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(7))]
    slots = capture(model, graph, ids, torch.device("cpu"), keep_float=False)
    assert all(slot.float_h is None for slot in slots)
    seen = []

    class Probe(Step):
        needs = frozenset({"hessian"})

        def apply_layer(self, ctx: LayerContext) -> None:
            assert all(slot.float_h is None for slot in slots)
            assert all(name != "cross_hessian" for name, _ in ctx.stats)
            seen.append(ctx.graph.index)

    Pass(
        model,
        graph,
        [Probe()],
        torch.device("cpu"),
        lambda layer, lg, stats: LayerContext(layer, lg, graph, None, None, None, stats),
        Timers(),
    ).run(slots)
    assert seen == [0]
    assert all(slot.float_h is None for slot in slots)


def test_float_capture_restores_hooks_and_flags_after_failure() -> None:
    """A failed reference forward leaves no capture hooks or disabled accumulators behind."""
    model = tiny_qwen3(layers=1)
    graph = describe(model, verify=False)
    layer = model.model.layers[0]
    stats = LayerStats(layer, graph.layers[0], frozenset({"hessian", "cross_hessian"}), torch.device("cpu"))
    original = [len(module._forward_pre_hooks) for module in layer.modules()]

    def fail():
        raise RuntimeError("reference forward failed")

    with pytest.raises(RuntimeError, match="reference forward failed"):
        stats.capture_float(fail)
    assert [len(module._forward_pre_hooks) for module in layer.modules()] == original
    assert all(accumulator.enabled for accumulator in stats.groups.values())


@pytest.mark.gpu
def test_paired_streams_stay_on_host_when_planned() -> None:
    """Both activation streams remain pinned on the host throughout a GPU layer pass."""
    model = tiny_qwen3(layers=2).to("cuda")
    graph = describe(model, verify=False)
    device = torch.device("cuda")
    batches = [torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(14))]
    slots = capture(model, graph, batches, device, host_acts=True, keep_float=True)
    seen = []

    class Probe(Step):
        needs = frozenset({"hessian", "cross_hessian"})

        def apply_layer(self, ctx: LayerContext) -> None:
            for slot in slots:
                assert slot.h.device.type == slot.float_h.device.type == "cpu"
                assert slot.h.is_pinned() and slot.float_h.is_pinned()
            for group in ctx.graph.groups:
                assert torch.count_nonzero(ctx.stats[("cross_hessian", group.modules[0])]) == 0
            seen.append(ctx.graph.index)

    Pass(
        model,
        graph,
        [Probe()],
        device,
        lambda layer, lg, stats: LayerContext(layer, lg, graph, None, None, None, stats, device=device),
        Timers(),
    ).run(slots)
    assert seen == [0, 1]


def test_a_missing_float_stream_raises_instead_of_giving_zero_cross_hessian():
    """Rows without their float-model inputs would give C = 0 and silently turn GPTAQ into GPTQ."""
    st = GroupStats(4, torch.device("cpu"), frozenset({"hessian", "cross_hessian"}))
    st.add(torch.randn(3, 4))  # quantized-stream rows, no xf
    with pytest.raises(RuntimeError, match="paired float stream is missing"):
        st.result()
    empty = GroupStats(4, torch.device("cpu"), frozenset({"hessian", "cross_hessian"}))
    assert torch.count_nonzero(empty.result()["cross_hessian"]) == 0  # nothing seen: zeros, as before
