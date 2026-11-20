"""An FP8 W8A8 ``lm_head``: the head step stores it as FP8 body groups store a weight, and inspect reads it.

An FP8 head is priced at +1.2-2.0 % KL of candidate 0; this pins what `qda plan` and `qda build` do with it
on a tiny CPU model: one ``float-quantized`` config group targeting the head, E4M3 codes and an fp32
per-output-channel scale, the body FP8 groups' dynamic per-token FP8 input scheme, and the two refusals (an
NVFP4 head by name, a tied head).
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from qda_quant import REGISTRY, Recipe, build
from qda_quant.arch import describe
from qda_quant.config import BuildConfig
from qda_quant.engine.checkpoint_size import packed_bits
from qda_quant.format import Format
from qda_quant.intake.report import inspect
from qda_quant.step import Refused
from qda_quant.target import Target
from safetensors import safe_open
from safetensors.torch import load_file

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids
from ..tiny import model

FP8 = Format.parse("w8a8-fp8")
NVFP4 = Format.parse("w4a4-nvfp4")


@pytest.fixture
def tiny(tmp_path):
    m = tmp_path / "m"
    m.mkdir()
    tiny_qwen3(layers=2).save_pretrained(m)
    sha = save_tiny_tokenizer(m)
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha
    )
    return tmp_path


def recipe(tmp, *steps):
    return Recipe.from_dict(
        {
            "steps": list(steps),
            "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )


def built(tmp, *steps, name="out", fmt="w4a4-nvfp4", target="sm100", **kw):
    return build(
        str(tmp / "m"),
        recipe(tmp, *steps),
        format=fmt,
        target=target,
        out_dir=tmp / name,
        devices=["cpu"],
        dtype=torch.float32,
        log=lambda s: None,
        **kw,
    )


def config(tmp, *steps, name="plan", **kw):
    return BuildConfig.from_dict(
        {
            "model": str(tmp / "m"),
            "format": "w4a4-nvfp4",
            "target": "sm100",
            "steps": list(steps),
            "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 4, "seqlen": 16}},
            "out": str(tmp / name),
            "devices": ["cpu"],
            **kw,
        }
    )


def test_the_fp8_head_is_stored_in_a_float_quantized_group_with_per_token_fp8_inputs(tiny):
    """The config group targets the head and declares the body FP8 groups' dynamic token FP8 activations."""
    out = built(tiny, {"rtn": {}}, {"head": {"format": "w8a8-fp8"}})
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "mixed-precision"  # NVFP4 body beside the FP8 head, one file set
    g = qc["config_groups"]["group_head"]
    assert g["targets"] == ["re:.*lm_head$"]
    assert g["format"] == "float-quantized"
    assert (g["weights"]["type"], g["weights"]["num_bits"], g["weights"]["strategy"]) == (
        "float",
        8,
        "channel",
    )
    assert g["weights"]["dynamic"] is False
    acts = g["input_activations"]
    assert (acts["type"], acts["num_bits"], acts["strategy"], acts["dynamic"]) == ("float", 8, "token", True)
    assert not any("lm_head" in x for x in qc["ignore"])
    # the body group is NVFP4; the head's group is the one the FP8 body groups declare (see the test below)
    assert qc["config_groups"]["group_0"]["format"] == "nvfp4-pack-quantized"


def test_the_head_stores_e4m3_codes_and_one_fp32_scale_per_output_channel(tiny):
    """The stored tensors are the FP8 body layout, and they are the AbsMax round trip of the source head."""
    out = built(tiny, {"rtn": {}}, {"head": {"format": "w8a8-fp8"}})
    stored = load_file(str(out / "model-00001.safetensors"))
    w = stored["lm_head.weight"]
    s = stored["lm_head.weight_scale"]
    assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (128, 64)
    assert s.dtype == torch.float32 and tuple(s.shape) == (128, 1)
    source = load_file(str(tiny / "m" / "model.safetensors"))["lm_head.weight"].float()
    scale = source.abs().amax(dim=1, keepdim=True) / 448  # the study's rule: row absmax / 448
    torch.testing.assert_close(s, scale, rtol=0, atol=0)
    assert torch.equal(w, (source / scale).clamp(-448, 448).to(torch.float8_e4m3fn))


def test_an_fp8_head_declares_the_same_activation_scheme_as_an_fp8_body(tiny):
    """A body already in FP8 W8A8 keeps one top-level format: the head's group repeats its activation args."""
    out = built(tiny, {"rtn": {}}, {"head": {"format": "w8a8-fp8"}}, fmt="w8a8-fp8")
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "float-quantized"
    assert (
        qc["config_groups"]["group_head"]["input_activations"]
        == qc["config_groups"]["group_0"]["input_activations"]
    )
    assert qc["config_groups"]["group_head"]["weights"] == qc["config_groups"]["group_0"]["weights"]


def test_inspect_names_the_head_group_and_its_kernel(tiny):
    """`qda inspect` reads the head's group from the config and binds CUTLASS FP8 to it."""
    out = built(tiny, {"rtn": {}}, {"head": {"format": "w8a8-fp8"}})
    rep = inspect(str(out), target="sm100")
    assert rep["reasons"] == []
    assert rep["groups"]["group_head"]["targets"] == ["re:.*lm_head$"]
    assert rep["kernels"] == {
        "group_0/dense": "FlashInferCuteDslNvFp4LinearKernel",
        "group_head/dense": "CutlassFP8ScaledMMLinearKernel",
    }


def test_the_plan_accepts_an_fp8_head_and_prices_its_bytes(tiny):
    """`qda plan` takes the recipe and its predicted checkpoint size drops by the head's stored difference."""
    base = config(tiny, {"rtn": {}}, name="base").plan()
    fp8 = config(tiny, {"rtn": {}}, {"head": {"format": "w8a8-fp8"}}, name="fp8").plan()
    with safe_open(str(tiny / "m" / "model.safetensors"), "pt") as h:
        source = h.get_tensor("lm_head.weight")
    shape, width = tuple(source.shape), source.element_size()
    # one bf16/float source tensor replaced by weight + weight_scale: one more header entry, 1 byte a weight
    assert base.output_bytes - fp8.output_bytes == (
        math.prod(shape) * width + 190 - (packed_bits(shape, FP8) // 8 + 2 * 190)
    )


def test_an_nvfp4_head_is_refused_by_name():
    """The search offers bf16 or w8a8-fp8; an NVFP4 head keeps its own refusal."""
    graph = describe(model("qwen3"), verify=False)
    with pytest.raises(Refused, match="NVFP4 head is not offered in this release"):
        REGISTRY["head"](format="w4a4-nvfp4").check(graph, NVFP4, Target.parse("sm100").plan(NVFP4, "dense"))


@pytest.mark.parametrize("name", ["w8a8-sym-pc", "w4a8-sym-g128"])
def test_an_integer_head_with_fp8_inputs_is_still_refused(name):
    """Only the FP8 weight format brings FP8 inputs to the head; an integer head stays weight-only."""
    graph = describe(model("qwen3"), verify=False)
    body = Format.parse("w4a8-sym-g128")
    with pytest.raises(Refused, match="the head runs weight-only"):
        REGISTRY["head"](format=name).check(graph, body, Target.parse("sm90").plan(body, "dense"))


def test_a_tied_head_is_still_refused():
    """A tied head would quantize the embedding lookup too."""
    graph = describe(model("qwen3"), verify=False)
    import dataclasses

    tied = dataclasses.replace(graph, extra={**graph.extra, "tied": True})
    with pytest.raises(Refused, match="tied input/output embeddings"):
        REGISTRY["head"](format="w8a8-fp8").check(tied, FP8, Target.parse("sm100").plan(FP8, "dense"))
