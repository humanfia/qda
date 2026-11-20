"""AutoRound trains and exports the shared quantizer's served grid."""

import json

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from qda_quant.arch import describe
from qda_quant.format import Format
from qda_quant.output.export import Writer
from qda_quant.solve.quant import Quantizer
from qda_quant.step import LayerContext
from qda_quant.steps import autoround
from qda_quant.target import Target
from safetensors import safe_open

from ..tiny import model


@pytest.mark.parametrize("name", ["w4a16", "w4a16-sym-g128", "w4a8"])
def test_training_forward_uses_shared_fake_quant(name):
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(1, 8, 128, generator=generator)
    weight[0, 0] = 0
    fmt = Format.parse(name)
    quantizer = Quantizer.of(fmt, Target().plan(fmt, "dense"), "absmax")
    lower = weight.amin(-1, keepdim=True).clamp(max=0).requires_grad_()
    upper = weight.amax(-1, keepdim=True).clamp(min=0).requires_grad_()
    offset = (torch.randn(weight.shape, generator=generator) * 0.1).requires_grad_()
    scale, zero = quantizer.range_params(lower, upper)
    shifted = weight + offset * quantizer.served_scale(scale).repeat_interleave(128, -1)
    expected = quantizer.fake(shifted, scale, zero)
    actual = autoround._fake(quantizer, weight, 128, lower, upper, offset)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad((actual - weight).square().mean() * 1000, (offset, lower, upper))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("name", ["w4a16", "w4a16-sym-g128", "w4a8"])
def test_exporter_packs_the_codes_evaluated_at_best_iterate(name, monkeypatch, tmp_path):
    tiny = model("qwen3", dtype=torch.bfloat16)
    source = tmp_path / "source"
    tiny.save_pretrained(source)
    graph = describe(tiny, verify=False)
    layer_graph = graph.layers[0]
    layer = tiny.model.layers[0]
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(2, 8, 128, generator=generator).to(torch.bfloat16)
    keyword = {
        "position_embeddings": (
            torch.ones(2, 8, 32, dtype=torch.bfloat16),
            torch.zeros(2, 8, 32, dtype=torch.bfloat16),
        )
    }
    with torch.no_grad():
        reference = layer(hidden, **keyword)
        reference = reference[0] if isinstance(reference, tuple) else reference
    fmt = Format.parse(name)
    context = LayerContext(
        layer,
        layer_graph,
        graph,
        fmt,
        Target().plan(fmt, "dense"),
        None,
        {("layer_io", layer_graph.module): [(hidden, (), keyword, reference)]},
    )
    units = list(autoround.AutoRound().units(context))
    evaluated = []
    best = []
    best_loss = float("inf")
    original_fake = Quantizer.fake
    original_call = autoround.functional_call

    def trace_fake(quantizer, weight, scale, zero):
        evaluated.append(
            (
                quantizer.quantize(weight, scale, zero).detach().clone(),
                quantizer.served_scale(scale).detach().clone(),
            )
        )
        return original_fake(quantizer, weight, scale, zero)

    def trace_call(*args, **kwargs):
        nonlocal best_loss, best
        output = original_call(*args, **kwargs)
        actual = output[0] if isinstance(output, tuple) else output
        loss = float((actual.float() - reference.float()).square().mean().detach())
        if loss < best_loss:
            best_loss = loss
            best = evaluated[-len(units) :]
        return output

    monkeypatch.setattr(Quantizer, "fake", trace_fake)
    monkeypatch.setattr(autoround, "functional_call", trace_call)
    autoround.AutoRound(iters=8, lr=0.005, target="quant_stream").apply_layer(context)  # the stat built above
    assert len(evaluated) == 8 * len(units)
    writer = Writer(tmp_path / "out", source, graph, fmt, Target())
    writer.add_layer(context)
    output = writer.finish(tiny.config, None, {})
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    shift = 0 if fmt.symmetric else -(1 << (fmt.weight_bits - 1))
    for unit, (expected_codes, expected_scale) in zip(units, best, strict=True):
        row_start = 0
        for module_name, rows in zip(unit.names, unit.rows, strict=True):
            with safe_open(output / index[module_name + ".weight_packed"], "pt") as tensors:
                shape = torch.Size(tensors.get_tensor(module_name + ".weight_shape").tolist())
                packed = tensors.get_tensor(module_name + ".weight_packed")
                actual_codes = unpack_from_int32(packed, fmt.weight_bits, shape).float() - shift
                stored_scale = tensors.get_tensor(module_name + ".weight_scale")
            row_slice = slice(row_start, row_start + rows)
            torch.testing.assert_close(actual_codes, expected_codes[0, row_slice], rtol=0, atol=0)
            torch.testing.assert_close(
                unit.quantizer.served_scale(stored_scale), expected_scale[0, row_slice], rtol=0, atol=0
            )
            row_start += rows
