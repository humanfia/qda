"""The group-size check as a plan can call it: widths in, usable group sizes out.

No weights and no GPU: these are the widths a model's safetensors headers hold.
"""

import pytest
from qda_quant import PRESETS
from qda_quant.engine.planning import QuantWidth, check_widths, working_group_sizes
from qda_quant.format import Format
from qda_quant.step import Refused
from qda_quant.target import Target

#: Hopper's table, as a build on an H100 plans.
TARGET = Target.parse("sm90")
W4 = Format.parse("w4a16")

#: NVIDIA-Nemotron-Nano-9B-v2: `intermediate_size` 15680 is not a multiple of 128 (25 mixer.down_proj).
NEMOTRON = [
    QuantWidth("model.layers.1.mixer.down_proj", 4480, 15680),
    QuantWidth("model.layers.0.mixer.in_proj", 22656, 4480),
]
#: DeepSeek-V2-Lite-Chat: `moe_intermediate_size` 10944 hits the same group size, by the kernel-binding path.
DEEPSEEK = [
    QuantWidth("model.layers.0.mlp.down_proj", 2048, 10944),
    QuantWidth("model.layers.1.mlp.up_proj", 10944, 2048),
]
QWEN = [
    QuantWidth("model.layers.0.self_attn.q_proj", 4096, 4096),
    QuantWidth("model.layers.0.mlp.down_proj", 4096, 14336),
]


def test_nemotrons_mixer_down_proj_takes_group_64():
    sizes = working_group_sizes(W4, NEMOTRON, TARGET)
    assert 128 not in sizes and 64 in sizes
    problem = check_widths(W4, NEMOTRON, TARGET)
    assert problem is not None
    assert (problem.name, problem.cols, problem.reason) == (
        "model.layers.1.mixer.down_proj",
        15680,
        "is not a multiple of 128",
    )


def test_deepseek_v2_lite_takes_group_64_by_the_same_check():
    """10944 was reported through a kernel-binding refusal; the width check must reach it too."""
    sizes = working_group_sizes(W4, DEEPSEEK, TARGET)
    assert 128 not in sizes and 64 in sizes
    problem = check_widths(W4, DEEPSEEK, TARGET)
    assert problem is not None and problem.cols == 10944
    assert check_widths(Format.parse("w4a16-g64"), DEEPSEEK, TARGET) is None


def test_a_model_whose_widths_all_divide_128_keeps_group_128():
    assert 128 in working_group_sizes(W4, QWEN, TARGET)
    assert check_widths(W4, QWEN, TARGET) is None


def test_widths_that_only_per_channel_stores_offer_just_per_channel():
    modules = [QuantWidth("model.layers.0.mlp.down_proj", 1024, 48)]
    assert working_group_sizes(W4, modules, TARGET) == [-1]
    assert check_widths(W4, modules, TARGET) is not None


def test_an_expert_tensor_keeps_its_width_check_without_a_dense_binding():
    """A routed expert is stored in the build's own fused layout; only its width is checked here."""
    expert = QuantWidth("model.layers.0.mlp.experts.0.down_proj", 2048, 10944, expert=True, bind=False)
    assert check_widths(W4, [expert], TARGET) is not None
    assert check_widths(Format.parse("w4a16-g64"), [expert], TARGET) is None


def test_hopper_binds_w4a8_g128_on_the_r1_down_proj_shape_and_refuses_the_others():
    """W4A8 on Hopper is one fact, not three: the plan names `CutlassW4A8LinearKernel` (group 128, symmetric),
    the width check binds it on the shape R1-Distill-Llama-8B's down_proj has (4096 x 14336), a shape CUTLASS
    cannot take is one sentence, and group 64 keeps the plan's own refusal."""
    w4a8 = PRESETS["w4a8"]
    down = [QuantWidth("model.layers.0.mlp.down_proj", 4096, 14336)]
    assert check_widths(w4a8, down, TARGET) is None
    assert TARGET.binds(w4a8, (4096, 14336), {}) == "CutlassW4A8LinearKernel"
    with pytest.raises(Refused, match=r"not multiples of 128"):
        TARGET.binds(w4a8, (4097, 14336), {})
    problem = check_widths(Format.parse("w4a8-sym-g64"), down, TARGET)
    assert problem is not None and problem.cols == 14336
    assert "takes int4, group 128, symmetric weights only" in problem.detail
    assert working_group_sizes(w4a8, down, TARGET) == [128]
    # the weight-only rows are untouched: W4A16 binds Humming on the same shape
    assert check_widths(W4, down, TARGET) is None
    assert TARGET.binds(W4, (4096, 14336), {}) == "HummingLinearKernel"
