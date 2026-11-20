"""qda_quant.solve.nvfp4 against compressed-tensors' NVFP4 arithmetic, value for value."""

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8, unpack_fp4_from_uint8
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from qda_quant.clips import CLIPS
from qda_quant.format import NVFP4, NVFP4_FORMAT, Format
from qda_quant.output.pack import pack_module
from qda_quant.solve import nvfp4
from qda_quant.solve.grid import GRIDS
from qda_quant.solve.quant import Quantizer
from qda_quant.target import KernelPlan

_SCHEME = preset_name_to_scheme("NVFP4", ["Linear"])
_PLAN = KernelPlan("K", scale_dtype="nvfp4", act_dtype="nvfp4", zero_point=False)


def _weights(*shape, seed=0):
    """Heavy-tailed bf16 weights with a zero group and a group small enough for a subnormal E4M3 scale."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g) * 0.02 * torch.exp(torch.randn(*shape, generator=g) * 0.7)
    w[..., 0, :16] = 0.0
    w[..., 1, 16:32] *= 1e-4
    return w.to(torch.bfloat16)


def _their_global(amax):
    return generate_gparam(-amax.reshape(1), amax.reshape(1))


def _their_scales(w, gs):
    wg = w.unflatten(-1, (-1, 16))
    return calculate_qparams(wg.amin(-1), wg.amax(-1), _SCHEME.weights, global_scale=gs)[0]


def test_global_scale_is_generate_gparam():
    edge = [0.0, 1e-30, 1e-8, 3e-3, 0.0172, 0.5, 1.0, 7.3, 448.0, 3e4, 3e38]
    amax = torch.tensor(edge + (torch.rand(200, generator=torch.Generator().manual_seed(1)) * 5).tolist())
    amax = amax.to(torch.bfloat16)
    ours = nvfp4.global_scale(amax)
    assert ours.dtype == torch.float32
    assert torch.equal(ours, torch.cat([_their_global(a) for a in amax]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_block_scales_are_calculate_qparams(seed):
    w = _weights(96, 256, seed=seed)
    gs = _their_global(w.abs().amax())
    ours = nvfp4.block_scales(w, gs)
    assert torch.equal(ours, _their_scales(w, gs).float())
    assert torch.equal(ours, ours.to(torch.float8_e4m3fn).float())  # every scale is an E4M3 value
    assert (ours[0, 0] == 0.125).all()  # the zero group


def test_codes_and_bytes_are_the_librarys():
    ties = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 1e-9]
    noise = (torch.randn(4000, generator=torch.Generator().manual_seed(3)) * 3).tolist()
    x = torch.tensor(ties + [-t for t in ties] + noise)
    x = torch.clamp(x[: x.numel() // 64 * 64].reshape(-1, 64), -6, 6)
    codes = nvfp4.e2m1_codes(x)
    ref = FP4_E2M1_DATA.cast_to_fp4(x.clone())
    ours = nvfp4.e2m1_values(codes)
    assert torch.equal(ours, ref)
    assert torch.equal(torch.signbit(ours), torch.signbit(ref))  # -0 where the library has it
    packed = nvfp4.pack_nibbles(codes)
    assert torch.equal(packed, pack_fp4_to_uint8(ref))
    assert torch.equal(nvfp4.unpack_nibbles(packed), codes)
    assert torch.equal(unpack_fp4_from_uint8(packed, *x.shape, dtype=torch.float32), ours)


@pytest.mark.parametrize("seed", [0, 1])
def test_round_to_nearest_is_the_nvfp4_compressor(seed):
    w = _weights(128, 512, seed=seed)
    gs = _their_global(w.abs().amax())
    scale = nvfp4.block_scales(w, gs)
    codes = nvfp4.quantize(w, scale, gs)
    theirs = NVFP4PackedCompressor.compress(
        {"weight": w, "weight_scale": _their_scales(w, gs), "weight_global_scale": gs}, _SCHEME
    )
    assert torch.equal(nvfp4.pack_nibbles(codes), theirs["weight_packed"])
    assert torch.equal(scale, theirs["weight_scale"].float())
    back = NVFP4PackedCompressor.decompress({**theirs, "weight_global_scale": gs}, _SCHEME)["weight"]
    assert torch.equal(nvfp4.dequant(codes, scale, gs).to(back.dtype), back)


def test_odd_and_partial_widths_are_refused():
    with pytest.raises(ValueError, match="odd"):
        nvfp4.pack_nibbles(torch.zeros(2, 3, dtype=torch.int8))
    with pytest.raises(ValueError, match="whole number of groups"):
        nvfp4.block_scales(torch.zeros(2, 24), torch.ones(1))


def _e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def test_format_has_one_spelling_and_signed_ladder_codes():
    fmt = Format.parse(" W4A4-NVFP4 ")
    assert fmt == NVFP4 and fmt.name == fmt.spelling == NVFP4_FORMAT
    assert (fmt.weight_bits, fmt.group_size, fmt.symmetric, fmt.weight_type) == (4, 16, True, "e2m1")
    assert (fmt.qmin, fmt.qmax, fmt.act) == (-7, 7, "nvfp4_block") and not fmt.symmetric_defaulted
    assert Format.parse("w4a4") is NVFP4  # the preset name


def test_grid_serves_its_format_only_and_absmax_leaves_it_alone():
    grid = GRIDS["nvfp4"]
    assert grid.refusal(NVFP4, _PLAN) is None and not grid.range_rule and grid.stores == NVFP4_FORMAT
    assert "not w4a16-asym-g128" in grid.refusal(Format.parse("w4a16"), _PLAN)
    assert "bf16 group scales" in grid.refusal(NVFP4, KernelPlan("K"))
    assert "nvfp4 grid" in GRIDS["absmax"].refusal(NVFP4, _PLAN)
    assert grid.divides(NVFP4, 4096) and not grid.divides(NVFP4, 4104)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lattice_is_e4m3_over_a_power_of_two_and_a_fixed_point(seed):
    w = _weights(3, 96, 256, seed=seed).float()
    w[2] *= 37.0  # every expert its own global scale
    steps, zero = GRIDS["nvfp4"].fit(NVFP4, w)
    assert zero is None and steps.shape == (3, 96, 16)
    g = GRIDS["nvfp4"].global_scale(steps)
    assert g.shape == (3, 1, 1) and torch.equal(g, torch.exp2(torch.log2(g).round()))
    f8 = steps * g
    assert torch.equal(f8, _e4m3(f8)) and (f8 > 0).all()
    top = f8.amax((-2, -1))
    assert ((top > 224) & (top <= 448)).all()
    assert torch.equal(nvfp4.lattice(steps), steps)
    assert torch.equal(GRIDS["nvfp4"].usable(steps), steps) and torch.equal(
        GRIDS["nvfp4"].served(steps), steps
    )
    # a zero group and a group below E4M3's range both take the floor, and code to zero
    assert (f8[:, 0, 0] == 0.125).all()


def test_lattice_does_not_depend_on_which_rows_share_the_global_scale():
    w = _weights(96, 256, seed=4).float()[2:]  # without the two rows that reach E4M3's floor
    whole, _ = GRIDS["nvfp4"].fit(NVFP4, w)
    for part in (slice(0, 40), slice(40, 94)):
        assert torch.equal(GRIDS["nvfp4"].fit(NVFP4, w[part])[0], whole[part])


def test_a_searched_step_matrix_keeps_its_steps():
    steps, _ = GRIDS["nvfp4"].fit(NVFP4, _weights(64, 256, seed=5).float())
    g = GRIDS["nvfp4"].global_scale(steps)
    f8 = steps * g
    grown = torch.where(f8 < 400, _e4m3(f8 * 1.3), _e4m3(f8 * 0.5)) / g  # the largest group shrinks
    assert torch.equal(nvfp4.lattice(grown), grown)
    assert (GRIDS["nvfp4"].global_scale(grown) >= g).all()
    with pytest.raises(ValueError, match="not an E4M3 value"):
        GRIDS["nvfp4"].global_scale(steps * 1.01)


@pytest.mark.parametrize("seed", [0, 1])
def test_packed_entry_decodes_in_the_library_to_the_served_weight(seed):
    w = _weights(2, 64, 256, seed=seed).float()
    q = Quantizer.of(NVFP4, _PLAN, "nvfp4")
    steps, zero = q.params(w)
    codes = q.quantize(w, steps, zero)
    assert codes.min() >= -7 and codes.max() <= 7 and torch.equal(codes, codes.round())
    served = q.dequant(codes, steps, zero)
    assert torch.equal(served, q.fake(w, steps, zero))
    g = q.grid.global_scale(q.stored_scale(steps))
    for e in range(2):
        qp = {
            "codes": codes[e].to(torch.int8),
            "scale": steps[e],
            "global_scale": g[e],
            "format": NVFP4_FORMAT,
        }
        qp["input_global_scale"] = torch.tensor(3.5)
        t = pack_module("m.proj", qp)
        assert sorted(t) == [f"m.proj.{k}" for k in sorted(
            ("input_global_scale", "weight_global_scale", "weight_packed", "weight_scale")
        )]  # fmt: skip
        packed, scale, gs = (t[f"m.proj.weight_{k}"] for k in ("packed", "scale", "global_scale"))
        assert (packed.dtype, scale.dtype, gs.dtype) == (torch.uint8, torch.float8_e4m3fn, torch.float32)
        assert packed.shape == (64, 128) and scale.shape == (64, 16) and gs.shape == (1,)
        assert torch.equal(t["m.proj.input_global_scale"], torch.tensor([3.5]))
        back = NVFP4PackedCompressor.decompress(
            {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": gs}, _SCHEME
        )["weight"]
        assert torch.equal(back.float(), served[e])
        # a slice of the entry, as a per-expert layout cuts a bank's rows
        half = pack_module("m.proj", qp, (slice(0, 32),))
        assert torch.equal(half["m.proj.weight_packed"], packed[:32])
        assert torch.equal(half["m.proj.weight_global_scale"], gs)


def test_a_shared_global_scale_is_stored_apart_for_each_module():
    """q/k/v and gate/up share one scale tensor; safetensors refuses two names over one storage."""
    q = Quantizer.of(NVFP4, _PLAN, "nvfp4")
    w = _weights(1, 16, 16, seed=3)
    steps, zero = q.params(w.float())
    codes = q.quantize(w.float(), steps, zero)
    g = q.grid.global_scale(q.stored_scale(steps)).reshape(1).to(torch.float32)  # an fp32 scale: `.to` no-op
    qp = {
        "codes": codes[0].to(torch.int8),
        "scale": steps[0],
        "global_scale": g,
        "format": NVFP4_FORMAT,
        "input_global_scale": torch.tensor(2.0),
    }
    a, b = (pack_module(m, qp) for m in ("m.q_proj", "m.k_proj"))
    for part in ("weight_global_scale", "input_global_scale"):
        x, y = a[f"m.q_proj.{part}"], b[f"m.k_proj.{part}"]
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()


def test_source_codec_reads_a_packed_entry_back():
    """`intake.source`'s nvfp4 codec decodes the three stored parts to the served weight (`Packed`'s view)."""
    from qda_quant.intake.source import codec_for

    w = _weights(64, 256).float()
    q = Quantizer.of(NVFP4, _PLAN, "nvfp4")
    steps, zero = q.params(w)
    codes = q.quantize(w, steps, zero)
    served = q.dequant(codes, steps, zero)
    g = q.grid.global_scale(q.stored_scale(steps))
    qp = {
        "codes": codes.to(torch.int8),
        "scale": steps,
        "global_scale": g,
        "format": NVFP4_FORMAT,
        "input_global_scale": torch.tensor(3.5),
    }
    t = pack_module("m.proj", qp)
    dtypes = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}
    headers = {k: (dtypes[v.dtype], list(v.shape)) for k, v in t.items()}
    config = {"quantization_config": {"quant_method": "compressed-tensors"}}
    codec = codec_for(config, headers, "m.proj.weight")
    assert codec.name == "nvfp4"
    assert codec.parts("m.proj.weight") == (
        "m.proj.weight_packed",
        "m.proj.weight_scale",
        "m.proj.weight_global_scale",
    )
    assert torch.equal(codec.decode(t, torch.float32), served)


def test_round_to_nearest_is_as_close_as_the_librarys():
    w = _weights(256, 1024, seed=6)
    gs = _their_global(w.abs().amax())
    theirs = nvfp4.dequant(
        nvfp4.quantize(w, _their_scales(w, gs).float(), gs), _their_scales(w, gs).float(), gs
    )
    q = Quantizer.of(NVFP4, _PLAN, "nvfp4")
    steps, _ = q.params(w.float())
    ours = q.fake(w.float(), steps, None)
    mse_ours, mse_theirs = (ours - w.float()).pow(2).mean(), (theirs - w.float()).pow(2).mean()
    assert abs(mse_ours / mse_theirs - 1) < 0.02


def test_packer_refuses_what_no_engine_loads():
    qp = {
        "codes": torch.zeros(2, 4, 16, dtype=torch.int8),
        "scale": torch.ones(2, 4, 1),
        "format": NVFP4_FORMAT,
    }
    with pytest.raises(ValueError, match="one Linear per expert"):
        pack_module("layers.0.mlp.experts.down_proj", {**qp, "global_scale": torch.ones(2, 1, 1)})
    one = {**qp, "codes": qp["codes"][0], "scale": qp["scale"][0]}
    with pytest.raises(ValueError, match="no global scale"):
        pack_module("m.proj", one)
    with pytest.raises(ValueError, match="no input scale"):
        pack_module("m.proj", {**one, "global_scale": torch.ones(1, 1)})


def _their_cut(x, gscale):
    """compressed-tensors' dynamic-local input quantization of `x` under `gscale`, block by block."""
    args = _SCHEME.input_activations
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = calculate_qparams(blocks.amin(-1), blocks.amax(-1), args, global_scale=gscale)[0].float()
    step = (scale / gscale).unsqueeze(-1)
    return (FP4_E2M1_DATA.cast_to_fp4(torch.clamp(blocks / step, -6, 6)) * step).flatten(-2)


@pytest.mark.parametrize("seed", [0, 1])
def test_block_cut_is_the_librarys_dynamic_local_quantization(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(5, 9, 64, generator=g) * torch.exp(torch.randn(5, 9, 64, generator=g))).to(
        torch.bfloat16
    )
    x[0, 0, :16] = 0.0
    gscale = nvfp4.input_global_scale(x.abs().amax())
    assert (
        gscale == torch.exp2(torch.log2(gscale).round())
        and x.abs().amax() <= 2688 / gscale < 2 * x.abs().amax()
    )
    ours = nvfp4.block_cut(x, gscale)
    assert ours.dtype == torch.bfloat16 and (ours[0, 0, :16] == 0).all()
    assert torch.equal(ours.float(), _their_cut(x, gscale))
    assert torch.equal(nvfp4.block_cut_parts(x, gscale)[0], ours.float())  # every value is exact in bf16
    # inside the range a power-of-two global scale leaves, the cut does not depend on it
    assert torch.equal(nvfp4.block_cut(x), ours)
    # a block above the range saturates as the library clamps its scale, and only with the scale given
    loud = x.clone()
    loud[1, 2, 16:32] *= 64.0
    assert torch.equal(nvfp4.block_cut(loud, gscale).float(), _their_cut(loud, gscale))
    assert not torch.equal(nvfp4.block_cut(loud), nvfp4.block_cut(loud, gscale))


def test_e2m1_round_is_the_code_table():
    a = torch.linspace(0, 6, 4801)
    assert torch.equal(nvfp4.e2m1_round(a), nvfp4.e2m1_values(nvfp4.e2m1_codes(a)))


@pytest.mark.parametrize("clip", ["fp4mse", "fp4hmse"])
def test_scale_search_stays_on_the_lattice_and_never_loses_to_the_fit(clip):
    w = _weights(2, 48, 256, seed=7).float()
    hdiag = torch.rand(2, 256, generator=torch.Generator().manual_seed(8)) + 0.01
    q = Quantizer.of(NVFP4, _PLAN, "nvfp4")
    fitted, _ = q.params(w)
    steps, zero = CLIPS[clip].choose(q, w, hdiag)
    assert zero is None and torch.equal(nvfp4.lattice(steps), steps)
    q.grid.global_scale(steps)  # every step is storable
    weight = hdiag[:, None, :] if clip == "fp4hmse" else torch.ones(2, 1, 256)

    def loss(s):
        return ((q.fake(w, s, None) - w).square() * weight).unflatten(-1, (-1, 16)).sum(-1)

    assert (loss(steps) <= loss(fitted)).all() and (loss(steps) < loss(fitted)).float().mean() > 0.2
    moved = torch.log2(steps / fitted)
    assert moved.min() >= -0.34 and moved.max() <= 1.51 and (moved > 0).any() and (moved < 0).any()
    assert CLIPS[clip].refusal(NVFP4, _PLAN) is None and CLIPS[clip].only_grid == "nvfp4"
    assert "bf16 scales" in CLIPS[clip].refusal(Format.parse("w4a16"), KernelPlan("K"))


def test_a_grid_named_clip_falls_back_per_unit_and_others_do_not():
    """In a mixed-format recipe the NVFP4 searches run on the NVFP4 units; another format takes the default.

    A clip that names no grid keeps today's behaviour: it is the step's clip on every unit, and its refusal
    is the one the plan raises (`test_mixed_formats.py` pins the texts).
    """
    from qda_quant.steps.gptq import GPTQ
    from qda_quant.steps.rtn import RTN

    step = GPTQ(clip="fp4hmse")
    assert step.default_clip == "minmax"
    assert step.clip_for("nvfp4").name == "fp4hmse"
    assert step.clip_for("absmax").name == "minmax"
    assert RTN(clip="fp4hmse").clip_for("absmax").name == "minmax"
    for clip in ("hmse", "mse", "imse", "grid3"):
        assert GPTQ(clip=clip).clip_for("absmax").name == clip, clip
    # the check-time copy for another format carries the fallback; the recipe's step keeps its setting
    fmt, plan = Format.parse("w8a8-fp8"), KernelPlan("K", scale_dtype="fp8_e4m3")
    copy = step.for_format(fmt, plan)
    assert (copy.clip, step.clip) == ("minmax", "fp4hmse")
    assert step.for_format(NVFP4, _PLAN) is step
