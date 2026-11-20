"""HMSE parity with the old qda codec, before bf16 scale storage."""

import pytest
import torch
from qda_quant import Format, Target
from qda_quant.clips import CLIPS
from qda_quant.solve.quant import Quantizer


def old_grid3(weights, diagonal, grid=True, reach=1):
    experts, rows, columns = weights.shape
    group = 128
    grouped = weights.reshape(experts, rows, columns // group, group)
    positive = grouped.clamp(min=0).amax(-1)
    negative = (-grouped).clamp(min=0).amax(-1)
    start = torch.maximum(positive / 7, negative / 8).clamp(min=1e-10)
    channel = (start.amax(-1, keepdim=True) / 448).clamp(min=1e-12)
    normalized = (start / channel).clamp(max=448).to(torch.float8_e4m3fn).float()
    spacing = torch.exp2(torch.floor(torch.log2(normalized.clamp(min=2**-6))) - 3)
    importance = diagonal.reshape(experts, 1, columns // group, group)
    best = best_error = at_start = None
    for offset in range(-reach, reach + 1):
        candidate = (normalized + offset * spacing).clamp(min=2**-9, max=448)
        scale = (candidate / 8).to(torch.float8_e4m3fn).float() * (channel * 8)
        codes = torch.round(grouped / scale[..., None]).clamp(-8, 7)
        error = ((grouped - codes * scale[..., None]).square() * importance).sum(-1)
        if offset == 0:
            at_start = scale
        if best_error is None:
            best, best_error = scale, error
        else:
            better = error < best_error
            best = torch.where(better, scale, best)
            best_error = torch.where(better, error, best_error)
    best = torch.where(start >= start.amax(-1, keepdim=True), at_start, best)
    if grid:
        rows = best.reshape(experts * rows, columns // group)
        channel = (rows.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-12)
        group_scale = (rows / channel).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        group_scale = (group_scale / 8).to(torch.float8_e4m3fn).float()
        best = (group_scale * (channel * 8)).reshape(experts, rows.shape[0] // experts, columns // group)
    return best.to(torch.bfloat16)


def old_hmse(weights, diagonal, bits, symmetric, group):
    grouped = weights.reshape(*weights.shape[:-1], -1, group)
    importance = diagonal.clamp_min(0).reshape(diagonal.shape[0], 1, -1, group)
    lower, upper = grouped.amin(-1).clamp_max(0), grouped.amax(-1).clamp_min(0)
    signed_min, signed_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    best_scale = best_zero = best_error = None
    for ratio in torch.linspace(0.70, 1.05, 36).tolist():
        if symmetric:
            scale = grouped.abs().amax(-1).clamp_min(1e-10) / ((2**bits - 1) / 2) * ratio
            zero = None
        else:
            scale = (upper * ratio - lower * ratio) / (2**bits - 1)
            scale = torch.where(scale == 0, torch.finfo(torch.float32).eps, scale)
            zero = (signed_min - lower * ratio / scale).clamp(signed_min, signed_max).round()
        offset = 0 if zero is None else zero[..., None]
        codes = (grouped / scale[..., None] + offset).round().clamp(signed_min, signed_max)
        error = (((codes - offset) * scale[..., None] - grouped).square() * importance).sum(-1)
        if best_error is None:
            best_scale, best_zero, best_error = scale, zero, error
        else:
            better = error < best_error
            best_scale = torch.where(better, scale, best_scale)
            best_zero = torch.where(better, zero, best_zero) if zero is not None else None
            best_error = torch.minimum(error, best_error)
    return best_scale, best_zero


@pytest.mark.parametrize("fmt", ["w4a16", "w4a16-sym-g128", "w3a16"])
@pytest.mark.parametrize("experts", [1, 3])
def test_hmse_matches_old_codec(fmt, experts):
    generator = torch.Generator().manual_seed(19)
    weights = torch.randn(experts, 7, 256, generator=generator)
    weights[:, 0] = 0
    diagonal = torch.rand(experts, 256, generator=generator).pow(4)
    diagonal[0, :128] = 0
    format_spec = Format.parse(fmt)
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")
    scale, zero = CLIPS["hmse"].choose(quantizer, weights, diagonal)
    expected_scale, expected_zero = old_hmse(
        weights, diagonal, format_spec.weight_bits, format_spec.symmetric, 128
    )
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    if zero is not None:
        torch.testing.assert_close(zero - 2 ** (format_spec.weight_bits - 1), expected_zero, rtol=0, atol=0)


def test_hmse_requires_hessian():
    format_spec = Format.parse("w3a16")
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")
    with pytest.raises(ValueError, match="Hessian diagonal"):
        CLIPS["hmse"].choose(quantizer, torch.zeros(1, 2, 128))


@pytest.mark.parametrize("grid_k", [0, 1, 2])
def test_grid3_matches_old_three_point_scale_search(grid_k):
    generator = torch.Generator().manual_seed(23)
    weights = torch.randn(2, 5, 256, generator=generator)
    diagonal = torch.rand(2, 256, generator=generator).pow(4)
    format_spec = Format.parse("w4a8")
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")

    scale, zero = CLIPS["grid3"].choose(quantizer, weights, diagonal, grid_k=grid_k)

    torch.testing.assert_close(scale, old_grid3(weights, diagonal, reach=grid_k), rtol=0, atol=0)
    assert zero is None


def test_grid3_requires_hessian_and_the_w4a8_scale_grid():
    format_spec = Format.parse("w4a8")
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")
    with pytest.raises(ValueError, match="Hessian diagonal"):
        CLIPS["grid3"].choose(quantizer, torch.zeros(1, 2, 128))

    format_spec = Format.parse("w4a16")
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")
    with pytest.raises(ValueError, match="symmetric W4A8 g128"):
        CLIPS["grid3"].choose(quantizer, torch.zeros(1, 2, 128), torch.ones(1, 128))


@pytest.mark.parametrize("magnitude", [0.0, 1e-20, 1e-10, 1e8])
def test_grid3_scales_survive_the_serving_split(magnitude):
    format_spec = Format.parse("w4a8")
    quantizer = Quantizer.of(format_spec, Target().plan(format_spec, "dense"), "absmax")
    weights = torch.full((1, 2, 256), magnitude)
    weights[:, :, 128:] = 0
    scale, zero = CLIPS["grid3"].choose(quantizer, weights, torch.ones(1, 256))

    assert (quantizer.served_scale(scale) > 0).all()
    codes = quantizer.quantize(weights, scale, zero)
    assert torch.isfinite(codes).all()
    assert torch.equal(codes, codes.round().clamp(format_spec.qmin, format_spec.qmax))
    assert torch.isfinite(quantizer.dequant(codes, scale, zero)).all()
    assert torch.count_nonzero(codes[:, :, 128:]) == 0
