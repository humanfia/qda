"""A checkpoint whose modules are stored in two schemes at once: NVFP4 body, FP8 W8A8 attention.

vLLM 0.33 serves such a checkpoint from its top-level ``format: mixed-precision``: each module is loaded by
the quant method of the config group that claims it, so an NVFP4 MLP and an FP8 attention live in one file.
"""

from __future__ import annotations

import json
import math
import re
import shutil

import pytest
import torch
import yaml
from qda_quant import REGISTRY, build
from qda_quant.calib.sources import FileSource
from qda_quant.engine import packed as P
from qda_quant.recipe import Recipe
from qda_quant.solve.nvfp4 import dequant, unpack_nibbles
from qda_quant.step import Refused
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .engine_util import save_tiny_tokenizer, tiny_qwen3, tiny_qwen3_moe, write_ids

#: Every attention projection of every layer, in FP8 W8A8; a fused q/k/v name takes the same rule.
ATTENTION = r"re:^model\.layers\.\d+\.self_attn\."
ATTN = "model.layers.0.self_attn.q_proj"
MLP = "model.layers.0.mlp.down_proj"


@pytest.fixture
def tiny(tmp_path):
    m = tiny_qwen3(layers=2)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha
    )
    return tmp_path


def recipe(tmp):
    return Recipe(
        steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=16)
    )


def built(tmp):
    return build(
        str(tmp / "m"),
        recipe(tmp),
        format="w4a4-nvfp4",
        target="sm100",
        out_dir=tmp / "out",
        devices=["cpu"],
        dtype=torch.float32,
        format_overrides=[(ATTENTION, "w8a8-fp8")],
        log=lambda s: None,
    )


def test_the_config_declares_a_mixed_precision_checkpoint_with_one_group_per_scheme(tiny):
    out = built(tiny)
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "mixed-precision"
    assert qc["config_groups"]["group_0"]["format"] == "nvfp4-pack-quantized"
    assert qc["config_groups"]["group_0"]["targets"] == ["Linear"]
    groups = [g for g in qc["config_groups"].values() if g["format"] == "float-quantized"]
    assert len(groups) == 1
    g = groups[0]
    assert (g["weights"]["type"], g["weights"]["num_bits"]) == ("float", 8)
    assert (g["weights"]["strategy"], g["weights"]["symmetric"]) == ("channel", True)
    assert (g["input_activations"]["type"], g["input_activations"]["strategy"]) == ("float", "token")
    assert g["input_activations"]["dynamic"] is True
    for layer in (0, 1):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"model.layers.{layer}.self_attn.{proj}" in g["targets"]
    assert yaml.safe_load((out / "resolved.yaml").read_text())["format_overrides"] == [
        {"pattern": ATTENTION, "format": "w8a8-fp8"}
    ]


def test_each_scheme_stores_its_own_tensors(tiny):
    out = built(tiny)
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert f"{ATTN}.weight_packed" not in idx
    with safe_open(out / idx[f"{ATTN}.weight"], "pt") as h:
        w = h.get_tensor(f"{ATTN}.weight")
        s = h.get_tensor(f"{ATTN}.weight_scale")
    assert w.dtype == torch.float8_e4m3fn and w.shape == (64, 64)
    assert s.dtype == torch.float32 and s.shape == (64, 1)
    assert f"{MLP}.weight" not in idx
    with safe_open(out / idx[f"{MLP}.weight_packed"], "pt") as h:
        packed = h.get_tensor(f"{MLP}.weight_packed")
        scale = h.get_tensor(f"{MLP}.weight_scale")
        global_scale = h.get_tensor(f"{MLP}.weight_global_scale")
        input_scale = h.get_tensor(f"{MLP}.input_global_scale")
    assert packed.dtype == torch.uint8 and packed.shape == (64, 96 // 2)
    assert scale.dtype == torch.float8_e4m3fn
    assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
    assert input_scale.dtype == torch.float32 and input_scale.numel() == 1


def test_each_stored_weight_decodes_to_its_served_weight(tiny):
    """The stored tensors alone decode to the matrices the kernels multiply.

    FP8: ``weight * weight_scale`` (the E4M3 values are the stored weight). NVFP4: the two codes a byte, their
    E4M3 group scales and the fp32 global scale, through the library's own dequantizer.
    """
    out = built(tiny)
    stored = {}
    for shard in sorted(out.glob("*.safetensors")):
        stored.update(load_file(str(shard)))
    w8 = stored[f"{ATTN}.weight"].float() * stored[f"{ATTN}.weight_scale"].float()
    assert w8.shape == (64, 64) and torch.isfinite(w8).all()
    # the source module's weight: both schemes' dequantized values are close to what the build quantized
    source = tiny_qwen3(layers=2).state_dict()
    assert (w8 - source[f"{ATTN}.weight"]).abs().max() < 0.2
    w4 = dequant(
        unpack_nibbles(stored[f"{MLP}.weight_packed"]),
        stored[f"{MLP}.weight_scale"].float(),
        stored[f"{MLP}.weight_global_scale"],
    )
    assert w4.shape == (64, 96) and torch.isfinite(w4).all()
    assert (w4 - source[f"{MLP}.weight"]).abs().max() < 0.2


def test_stock_transformers_serves_both_kinds(tiny):
    """The CPU proxy for vLLM's loader: every module takes the scheme its config group names, and answers."""
    import transformers

    out = built(tiny)
    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).eval()
    attn = back.model.layers[0].self_attn.q_proj.quantization_scheme
    mlp = back.model.layers[0].mlp.down_proj.quantization_scheme
    assert attn.weights.num_bits == 8
    assert attn.input_activations.dynamic is True
    assert mlp.weights.num_bits == 4
    assert mlp.input_activations.dynamic == "local"
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(0))
    source = tiny_qwen3(layers=2).eval()
    with torch.no_grad():
        logits = back(input_ids=ids).logits.float()
        want = source(input_ids=ids).logits.float()
    assert torch.isfinite(logits).all()
    assert (logits - want).abs().max() < 0.5


def test_inspect_names_both_kernels(tiny):
    from qda_quant.intake.report import inspect

    out = built(tiny)
    rep = inspect(str(out), "sm100")
    assert rep["reasons"] == []
    assert rep["kernels"]["group_0/dense"] == "FlashInferCuteDslNvFp4LinearKernel"
    assert rep["kernels"]["group_1/dense"] == "CutlassFP8ScaledMMLinearKernel"


MOE_EXPERT = "model.layers.0.mlp.experts.0"
MOE_ROUTER = "model.layers.0.mlp.gate"


@pytest.fixture(scope="module")
def built_moe(tmp_path_factory):
    """The tiny MoE model built four ways -- rtn and gptq, with and without the NVFP4 scale search.

    NVFP4 experts and MLP, FP8 attention. Both steps run: NVFP4 banks need the calibration forward for
    their input scales, which rtn itself does not ask for, and GPTQ also solves the bank against its
    Hessian. The ``*_h`` builds give the steps ``clip: fp4hmse``, which the NVFP4 units take and the FP8
    attention must not (``test_a_mixed_moe_build_keeps_the_search_on_its_nvfp4_banks``).
    """
    tmp = tmp_path_factory.mktemp("mixed-moe")
    tiny_qwen3_moe(layers=2).save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(tmp / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha)
    outs = {}
    for name, settings in (
        ("rtn", {}),
        ("gptq", {"actorder": "static", "damp": 0.01}),
        ("rtn_h", {"clip": "fp4hmse"}),
        ("gptq_h", {"actorder": "static", "damp": 0.01, "clip": "fp4hmse"}),
    ):
        r = Recipe.from_dict(
            {
                "steps": [{name.removesuffix("_h"): settings}],
                "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 4, "seqlen": 16}},
            }
        )
        outs[name] = build(
            str(tmp / "m"), r, format="w4a4-nvfp4", target="sm100", out_dir=tmp / name,
            devices=["cpu"], dtype=torch.float32, format_overrides=[(ATTENTION, "w8a8-fp8")],
            log=lambda s: None,
        )  # fmt: skip
    return outs


@pytest.mark.parametrize("step", ["rtn", "gptq"])
def test_a_mixed_moe_build_groups_its_experts_and_attention(built_moe, step):
    """The NVFP4 body keeps the class target that claims the per-expert Linears; attention gets a group."""
    qc = json.loads((built_moe[step] / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "mixed-precision"
    assert qc["config_groups"]["group_0"]["format"] == "nvfp4-pack-quantized"
    assert qc["config_groups"]["group_0"]["targets"] == ["Linear"]
    (fp8,) = [g for g in qc["config_groups"].values() if g["format"] == "float-quantized"]
    assert set(fp8["targets"]) == {
        f"model.layers.{layer}.self_attn.{proj}"
        for layer in (0, 1)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    # the router is a float Linear the body group would otherwise claim: it stays bf16
    assert {MOE_ROUTER, "model.layers.1.mlp.gate"} <= set(qc["ignore"])


@pytest.mark.parametrize("step", ["rtn", "gptq"])
def test_a_mixed_moe_build_stores_an_expert_and_attention_in_their_schemes(built_moe, step):
    out = built_moe[step]
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(out / idx[f"{MOE_EXPERT}.gate_proj.weight_packed"], "pt") as h:
        packed = h.get_tensor(f"{MOE_EXPERT}.gate_proj.weight_packed")
        scale = h.get_tensor(f"{MOE_EXPERT}.gate_proj.weight_scale")
        global_scale = h.get_tensor(f"{MOE_EXPERT}.gate_proj.weight_global_scale")
        input_scale = h.get_tensor(f"{MOE_EXPERT}.gate_proj.input_global_scale")
    assert packed.dtype == torch.uint8 and packed.shape == (32, 64)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (32, 8)
    assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
    assert input_scale.dtype == torch.float32 and input_scale.numel() == 1
    with safe_open(out / idx[f"{ATTN}.weight"], "pt") as h:
        w = h.get_tensor(f"{ATTN}.weight")
        s = h.get_tensor(f"{ATTN}.weight_scale")
    assert w.dtype == torch.float8_e4m3fn and w.shape == (128, 128)
    assert s.dtype == torch.float32 and s.shape == (128, 1)


def test_a_mixed_moe_build_names_the_three_kernels(built_moe):
    """`qda inspect` expects the TRT-LLM NVFP4 experts beside the FP8 CUTLASS and NVFP4 dense kernels."""
    from qda_quant.intake.report import inspect

    rep = inspect(str(built_moe["rtn"]), "sm100")
    assert rep["reasons"] == []
    assert rep["kernels"]["group_0/moe"] == "TrtLlmNvFp4ExpertsMonolithic"
    assert set(rep["kernels"].values()) >= {
        "TrtLlmNvFp4ExpertsMonolithic",
        "CutlassFP8ScaledMMLinearKernel",
    }


@pytest.mark.parametrize("step", ["scale_kd", "soup"])
def test_a_checkpoint_step_plans_on_a_mixed_stage(tiny, step):
    """A checkpoint step reads the FP8 group back too: the plan accepts `scale_kd` and `soup` on a mixed stage.

    The stage stores an FP8 attention group beside the NVFP4 body; `engine/packed.py` decodes each module in its
    own scheme (FP8 W8A8 as its E4M3 weight times its fp32 per-channel scale), so nothing is half-read: each
    stage plans with the one checkpoint step it names.
    """
    from qda_quant.engine.planning import compile_plan
    from qda_quant.steps.scale_kd import ScaleKD
    from qda_quant.steps.soup import Soup
    from qda_quant.target import Target

    out = built(tiny)
    s = ScaleKD() if step == "scale_kd" else Soup(**{"with": [str(out)]})
    r = Recipe(steps=[s], calib=FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16))
    plan = compile_plan(str(out), r, None, Target.parse("sm100"), [torch.device("cpu")])
    assert [c.name for c in plan.checkpoint_steps] == [step] and plan.segments == []


def _kd_recipe(tiny, out=None, *, overrides=(), head=None):
    """The mixed NVFP4 + FP8 build's recipe shape: a data-free NVFP4 pass, then `scale_kd` on the stage it wrote."""
    steps = [{"rtn": {"clip": "fp4mse"}}]
    if head is not None:
        steps.append({"head": {"format": head}})
    steps.append({"scale_kd": {"steps": 1, "batch_tokens": 64, "head_chunk": 16, "trainable": ["scales"]}})
    cfg = {
        "model": str(tiny / "m"),
        "format": "w4a4-nvfp4",
        "target": "sm100",
        "out": None if out is None else str(out),
        "devices": ["cpu"],
        "steps": steps,
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
    }
    if overrides:
        cfg["format_overrides"] = [{"pattern": p, "format": f} for p, f in overrides]
    return cfg


@pytest.mark.parametrize(
    ("overrides", "head"),
    [
        ([(ATTENTION, "w8a8-fp8")], "w8a8-fp8"),
        ([(ATTENTION, "w8a8-fp8")], None),
        ((), "w8a8-fp8"),
    ],
    ids=["mixed-shape", "override-only", "head-only"],
)
def test_a_checkpoint_step_on_an_fp8_stage_plans_and_builds(tiny, capsys, overrides, head):
    """`scale_kd` reads an FP8 attention group and an FP8 head back, wherever the recipe puts them.

    The mixed NVFP4 + FP8 build's shape: an NVFP4 body, an FP8 attention override from the config and/or an FP8 head from a
    `head` step. `qda plan` reaches the checkpoint step, and the build trains scales and writes a checkpoint
    whose FP8 modules hold the trained fp32 scale; one step is enough to see the read-back is exact.
    """
    from qda_quant.cli import main

    path = tiny / "kd.yaml"
    path.write_text(yaml.safe_dump(_kd_recipe(tiny, tiny / "out", overrides=overrides, head=head)))
    assert main(["plan", str(path)]) == 0
    assert "then checkpoint step scale_kd" in capsys.readouterr().out
    assert main(["build", str(path)]) == 0
    out = tiny / "out"
    ck = P.Packed(out)
    fp8 = [s for s in ck.stems if ck.format_of(s).weight_type == "fp8"]
    assert fp8 and len(fp8) < len(ck.stems)
    scales, floats = ck.state(lambda _: None)
    assert not floats
    for s in fp8:
        stored = ck.tensor(s + ".weight_scale")
        assert stored.dtype == torch.float32 and scales[s][0].dtype == torch.float32
        assert torch.equal(ck.stored_scale(s, *scales[s]), stored)
    assert json.loads((out / "build.json").read_text())["history"][-1]["scales_changed"][0] > 0


def test_a_checkpoint_step_on_a_pure_nvfp4_stage_with_a_bf16_head_still_builds(tiny):
    """The check names FP8 groups only: an all-NVFP4 stage whose head stays bf16 plans and builds as before."""
    from qda_quant.cli import main

    path = tiny / "kd.yaml"
    path.write_text(yaml.safe_dump(_kd_recipe(tiny, tiny / "out")))
    assert main(["plan", str(path)]) == 0
    assert main(["build", str(path)]) == 0
    qc = json.loads((tiny / "out" / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "nvfp4-pack-quantized"
    assert not any(g["format"] == "float-quantized" for g in qc["config_groups"].values())


def test_the_fp8_weight_and_its_closed_form_gradient_are_autograds():
    """An FP8 served weight is ``E4M3 value * fp32 scale`` (straight-through in `delta`), and `delta_grad` is
    autograd's gradient of that weight sum over the row's columns."""
    from qda_quant.engine import served_modules as S
    from qda_quant.format import Format
    from qda_quant.solve.quant import Quantizer
    from qda_quant.target import Target

    torch.manual_seed(0)
    f = Format.parse("w8a8-fp8")
    q = Quantizer.of(f, Target.parse("sm100").plan(f, "dense"), "absmax")
    w = torch.randn(48, 256)
    codes = w.to(torch.float8_e4m3fn)
    s0 = (codes.float().abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-4)
    lin = S.ServedLinear("m", codes, None, s0, torch.zeros_like(s0), q)
    delta = 0.05 * torch.randn(s0.shape)
    rows = slice(5, 37)
    assert torch.equal(
        lin.stored.weight_bf16(delta[rows], rows), lin.stored.weight(delta[rows], rows).to(torch.bfloat16)
    )
    gw = torch.randn(32, 256)
    d = delta[rows].clone().requires_grad_(True)
    want = torch.autograd.grad(lin.stored.weight(d, rows), d, gw)[0]
    assert torch.allclose(lin.stored.delta_grad(delta[rows], gw, rows), want, rtol=1e-6, atol=1e-9)


def _mixed(tmp, name="mix"):
    """A mixed NVFP4 + FP8 build's shape: an NVFP4 body, the FP8 attention override and an FP8 head."""
    return build(
        str(tmp / "m"),
        Recipe(
            steps=[REGISTRY["rtn"](), REGISTRY["head"](format="w8a8-fp8")],
            calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=16),
        ),
        format="w4a4-nvfp4",
        target="sm100",
        out_dir=tmp / name,
        devices=["cpu"],
        dtype=torch.float32,
        format_overrides=[(ATTENTION, "w8a8-fp8")],
        log=lambda s: None,
    )


def _tensors(d):
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out |= {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
    return out


def _same_tensors(a, b):
    ta, tb = _tensors(a), _tensors(b)
    return set(ta) == set(tb) and all(ta[k].dtype == tb[k].dtype and torch.equal(ta[k], tb[k]) for k in ta)


def _same_files(a, b):
    fa = {p.name: p.read_bytes() for p in sorted(a.glob("*.safetensors"))}
    fb = {p.name: p.read_bytes() for p in sorted(b.glob("*.safetensors"))}
    return fa == fb and (a / P.STATE).read_bytes() == (b / P.STATE).read_bytes()


def test_a_mixed_stage_reads_back_and_writes_itself_byte_for_byte(tiny):
    """Every stem reads back in its own scheme, the served FP8 weight is the kernel's arithmetic, and writing
    the read-back state is the checkpoint again, byte for byte, so exporting it again reproduces the files.

    FP8 W8A8: the stored E4M3 ``weight`` times the fp32 per-channel ``weight_scale``, with the dynamic per-token
    E4M3 input cut. NVFP4: E2M1 ladder codes under their E4M3 group scales and the fp32 global scale.
    """
    from qda_quant.solve.quant import fp8_token_parts

    out = _mixed(tiny)
    ck = P.Packed(out)
    kinds = {s: ck.format_of(s).weight_type for s in ck.stems}
    assert set(kinds.values()) == {"fp8", "e2m1"}
    assert kinds[ATTN] == kinds["lm_head"] == "fp8" and kinds[MLP] == "e2m1"
    scales, floats = ck.state(lambda _: None)
    assert not floats and set(scales) == set(ck.stems)
    for s, kind in kinds.items():
        assert torch.equal(ck.stored_scale(s, *scales[s]), ck.tensor(s + ".weight_scale")), s
        if kind == "fp8":
            assert torch.equal(ck.codes(s)[0], ck.tensor(s + ".weight"))
            assert ck.codes(s)[0].dtype == torch.float8_e4m3fn
            assert ck.tensor(s + ".weight_scale").dtype == torch.float32
            assert ck.codes(s)[1] is None  # no zero point: symmetric
        else:
            assert ck.codes(s)[0].dtype == torch.int8
    # the served weight and forward are vLLM's: E4M3 value times the fp32 scale, input cut per token
    student = P.load_student(ck, scales, floats, torch.device("cpu"), "eager")
    mod = student.get_submodule(ATTN)
    assert mod.q.fp8_weights and mod.s0.dtype == torch.float32
    w = ck.tensor(f"{ATTN}.weight").float() * ck.tensor(f"{ATTN}.weight_scale").float()
    assert torch.equal(mod.weight(mod.delta.detach()), w)
    x = torch.randn(3, mod.in_features, dtype=torch.bfloat16)
    cut, sx = fp8_token_parts(x)
    assert torch.equal(mod(x), (torch.nn.functional.linear(cut, w) * sx).to(x.dtype))
    # the read-back state reproduces the checkpoint; writing it again is its files, byte for byte
    again, third = tiny / "again", tiny / "third"
    P.write(ck, again, scales, floats, {"step": "roundtrip"}, ck.teacher(), [torch.device("cpu")])
    assert _same_tensors(out, again)
    sc2, fl2 = P.Packed(again).state(lambda _: None)
    assert not fl2
    assert all(torch.equal(sc2[s][0], scales[s][0]) and torch.equal(sc2[s][1], scales[s][1]) for s in scales)
    P.write(P.Packed(again), third, sc2, fl2, {"step": "roundtrip"}, ck.teacher(), [torch.device("cpu")])
    assert _same_files(again, third)


def test_scale_kd_two_steps_on_a_mixed_stage_train_both_schemes_scales(tiny):
    """A `scale_kd` stage on a mixed checkpoint trains the FP8 per-channel scales and the NVFP4 group scales.

    Every other stored tensor stays byte-identical, each stored scale is the state's ``s0 * exp(delta)`` in the
    scheme's own dtype, and the result reads back: the state matches the stored scales and the student serves
    the epsilon-step weight ``code * s0``.
    """
    src = _mixed(tiny)
    # two full steps (warmup 1) at lr 0.1: an NVFP4 group scale is stored as E4M3, so a smaller move rounds
    # away; this is a mechanism test on a tiny torch model, not a claim about the stage's KL (a scale_kd stage
    # on a real model is the mixed build's check in the release notes)
    kd = REGISTRY["scale_kd"](
        steps=2, lr=0.1, warmup=1, batch_tokens=64, head_chunk=16, trainable=["scales"], attn="eager"
    )
    out = build(
        str(src),
        Recipe(steps=[kd], calib=FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16)),
        target="sm100",
        out_dir=tiny / "kd",
        devices=["cpu"],
        log=lambda _: None,
    )
    before, after = _tensors(src), _tensors(out)
    assert set(before) == set(after)
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.endswith(".weight_scale") for k in changed)
    ck = P.Packed(out)
    scales, floats = ck.state(lambda _: None)
    assert not floats
    fp8 = [s for s in ck.stems if ck.format_of(s).weight_type == "fp8"]
    nvfp4 = [s for s in ck.stems if ck.format_of(s).weight_type == "e2m1"]
    assert {k.removesuffix(".weight_scale") for k in changed} & set(fp8)  # an FP8 scale moved...
    assert {k.removesuffix(".weight_scale") for k in changed} & set(nvfp4)  # ...and an NVFP4 one
    assert all((scales[s][1] != 0).any() for s in fp8 + nvfp4)  # every scheme's training state moved
    assert all(after[s + ".weight_scale"].dtype == torch.float32 for s in fp8)
    assert all(after[s + ".weight_scale"].dtype == torch.float8_e4m3fn for s in nvfp4)
    assert all(torch.equal(ck.stored_scale(s, *scales[s]), after[s + ".weight_scale"]) for s in ck.stems)
    for s in fp8:
        assert torch.equal(before[s + ".weight"], after[s + ".weight"])  # codes untouched
    stage = json.loads((out / "build.json").read_text())["history"][-1]
    assert stage["scales_changed"][0] > 0 and len(stage["log"]) == 2
    assert all(math.isfinite(entry["kl"]) for entry in stage["log"])
    # the trained FP8 scale loads back: the student serves code * s0 * exp(delta), the stored scale
    student = P.load_student(ck, scales, floats, torch.device("cpu"), "eager")
    mod = student.get_submodule(ATTN)
    assert torch.equal(mod.delta, scales[ATTN][1])
    assert torch.equal(
        mod.weight(), ck.tensor(f"{ATTN}.weight").float() * after[f"{ATTN}.weight_scale"].float()
    )


def _fake_stage(src, out, delta_fn):
    """Write `src` to `out` with a hand-made training state (delta per stem) and its stored scales."""
    ck = P.Packed(src)
    scales, _ = ck.state(lambda _: None)
    P.write(
        ck, out, {s: (s0, delta_fn(s0)) for s, (s0, _) in scales.items()}, {},
        {"step": "fake"}, ck.teacher(), [torch.device("cpu")],
    )  # fmt: skip


def test_soup_of_two_mixed_stages_averages_both_schemes_scales(tiny):
    """A soup of two mixed stages averages FP8 and NVFP4 scales in log space and keeps both schemes' code bytes.

    An input whose FP8 ``weight`` (its codes) differs is refused by name: an FP8 module shares its weight bytes
    as an integer or NVFP4 module shares ``weight_packed``.
    """
    src = _mixed(tiny)
    _fake_stage(src, tiny / "A", lambda s0: torch.full(s0.shape, 0.30))
    _fake_stage(src, tiny / "B", lambda s0: torch.full(s0.shape, -0.10))
    soup = REGISTRY["soup"](**{"with": [str(tiny / "B")], "weights": [1, 3]})
    r = Recipe(steps=[soup], calib=FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16))
    out = build(str(tiny / "A"), r, target="sm100", out_dir=tiny / "S", devices=["cpu"], log=lambda _: None)
    result = P.Packed(out)
    scales, _ = result.state(lambda _: None)
    for s, (s0, d) in scales.items():
        assert torch.allclose(d, torch.full(s0.shape, 0.25 * 0.30 + 0.75 * -0.10), atol=1e-6), s
        assert torch.equal(result.stored_scale(s, s0, d), result.tensor(s + ".weight_scale"))
    a, b = _tensors(src), _tensors(out)
    assert all(torch.equal(a[k], b[k]) for k in a if k.endswith(("weight_packed", "weight_zero_point")))
    fp8 = [s for s in result.stems if result.format_of(s).weight_type == "fp8"]
    assert fp8 and all(torch.equal(a[s + ".weight"], b[s + ".weight"]) for s in fp8)
    # one input's FP8 weight bytes differ: refused by name, as an integer module's codes would be
    c = tiny / "C"
    shutil.copytree(tiny / "B", c)
    t = _tensors(c)
    k = fp8[0] + ".weight"
    t[k] = (t[k].float() + 1).to(torch.float8_e4m3fn)
    for f in c.glob("*.safetensors"):
        f.unlink()
    save_file(t, str(c / "model.safetensors"))
    (c / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(t, "model.safetensors")})
    )
    with pytest.raises(Refused, match=rf"{re.escape(k)} differs"):
        build(
            str(tiny / "A"), Recipe(steps=[REGISTRY["soup"](**{"with": [str(c)]})], calib=r.calib),
            target="sm100", out_dir=tiny / "S2", devices=["cpu"], log=lambda _: None,
        )  # fmt: skip


def test_an_override_may_not_split_a_fused_group(tiny):
    """vLLM fuses q/k/v into one linear with one scheme, so half of it cannot take the FP8 format.

    The refusal runs at plan time (before a weight loads) and says why: the rule matched one of the group's
    three linears, which are solved and served together.
    """
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.step import Refused
    from qda_quant.target import Target

    with pytest.raises(Refused, match=r"matches 1 of the 3 linears.*one fused linear with one scheme"):
        compile_plan(
            str(tiny / "m"), recipe(tiny), Format.parse("w4a4-nvfp4"), Target.parse("sm100"),
            [torch.device("cpu")],
            overrides=[("model.layers.0.self_attn.q_proj", "w8a8-fp8")],
        )  # fmt: skip


# --- the NVFP4 clip search in a recipe that mixes NVFP4 and FP8 ---------------------------------------------


#: The mixed build's settings: GPTQ with the NVFP4 scale search, attention stored in FP8.
def _search_recipe(tmp, clip):
    settings = {"actorder": "static", "damp": 0.01}
    if clip is not None:
        settings["clip"] = clip
    return Recipe.from_dict(
        {
            "steps": [{"gptq": settings}],
            "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )


def _search_build(tmp, out, clip, overrides=((ATTENTION, "w8a8-fp8"),)):
    return build(
        str(tmp / "m"), _search_recipe(tmp, clip), format="w4a4-nvfp4", target="sm100",
        out_dir=tmp / out, devices=["cpu"], dtype=torch.float32, format_overrides=list(overrides),
        log=lambda s: None,
    )  # fmt: skip


@pytest.fixture(scope="module")
def clip_builds(tmp_path_factory):
    """One mixed recipe built three ways: the NVFP4 search, it overridden to minmax, and no FP8 units."""
    tmp = tmp_path_factory.mktemp("mixed-clip")
    tiny_qwen3(layers=2).save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(tmp / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha)
    return {
        "search": _search_build(tmp, "search", "fp4hmse"),
        "minmax": _search_build(tmp, "minmax", None),
        "pure": _search_build(tmp, "pure", "fp4hmse", overrides=[]),
    }


def _stored(out):
    """Every tensor of a build's shards, by name."""
    got = {}
    for shard in sorted(out.glob("*.safetensors")):
        got.update(load_file(str(shard)))
    return got


def test_a_grid_named_clip_searches_the_nvfp4_units_of_a_mixed_build(clip_builds):
    """``fp4hmse`` runs on the NVFP4 units and the FP8 attention takes ``minmax``, unit by unit.

    The first layer's FP8 projections have the same inputs in every build (the embedding), so they compare:
    the mixed build with the search stores exactly the build with no clip. Its NVFP4 projections cannot be
    compared with a pure-NVFP4 build's: the calibration forward cuts each unit's input with its own kernel,
    so layer 0's MLP sees an FP8-cut attention output in the mixed build, and its input scale and Hessian
    differ. `test_each_clip_is_asked_only_for_the_grid_it_serves` tests the search at the call site instead.
    ``build.json`` records the clip each format got; a recipe with one format records nothing new.
    """
    search, minmax = (_stored(clip_builds[k]) for k in ("search", "minmax"))
    attn = [
        f"model.layers.0.self_attn.{p}.{t}"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        for t in ("weight", "weight_scale")
    ]
    mlp = [
        f"model.layers.0.mlp.{p}.{t}"
        for p in ("gate_proj", "up_proj", "down_proj")
        for t in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
    ]
    for name in attn:
        assert torch.equal(search[name], minmax[name]), name
    assert any(not torch.equal(search[name], minmax[name]) for name in mlp)
    got = json.loads((clip_builds["search"] / "build.json").read_text())["clip_by_format"]
    assert got == {"gptq": {"w4a4-nvfp4": "fp4hmse", "w8a8-fp8": "minmax"}}
    for one in ("minmax", "pure"):
        assert "clip_by_format" not in json.loads((clip_builds[one] / "build.json").read_text()), one


@pytest.mark.parametrize("step", ["rtn", "gptq"])
def test_each_clip_is_asked_only_for_the_grid_it_serves(tiny, step, monkeypatch):
    """The per-unit choice is what runs: the NVFP4 search is never called on an FP8 unit, nor minmax on NVFP4.

    The build's calibration and solves call `Clip.choose` with the unit's quantizer; this records the grid of
    every call, so the test reads the choice at the call site rather than the plan's admission of it.
    """
    from qda_quant.clips import CLIPS

    seen: dict[str, set[str]] = {}
    for name in ("fp4hmse", "minmax"):
        clip, original = CLIPS[name], CLIPS[name].choose

        def record(q, w, hdiag=None, grid_k=1, _name=name, _original=original):
            seen.setdefault(_name, set()).add(q.grid.name)
            return _original(q, w, hdiag, grid_k)

        monkeypatch.setattr(clip, "choose", record)
    r = Recipe.from_dict(
        {
            "steps": [{step: {"clip": "fp4hmse"}}],
            "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )
    build(
        str(tiny / "m"), r, format="w4a4-nvfp4", target="sm100", out_dir=tiny / f"calls-{step}",
        devices=["cpu"], dtype=torch.float32, format_overrides=[(ATTENTION, "w8a8-fp8")],
        log=lambda s: None,
    )  # fmt: skip
    assert seen["fp4hmse"] == {"nvfp4"}
    assert seen["minmax"] == {"absmax"}  # the FP8 attention's grid


@pytest.mark.parametrize("step", ["gptq", "rtn"])
def test_the_recipes_format_still_decides_whether_a_grid_named_clip_is_admitted(tiny, step):
    """Only an *override* format falls back: with the FP8 format as the recipe's own, the search is refused.

    The text is the one before the per-unit choice existed: the body check keeps the clip it was given.
    """
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.step import Refused
    from qda_quant.target import Target

    r = Recipe.from_dict(
        {
            "steps": [{step: {"clip": "fp4hmse"}}],
            "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )
    with pytest.raises(Refused) as e:
        compile_plan(
            str(tiny / "m"), r, Format.parse("w8a8-fp8"), Target.parse("sm100"), [torch.device("cpu")]
        )
    assert str(e.value) == (
        f"{step} clip='fp4hmse': fp4hmse steps along NVFP4's E4M3 group scales; w8a8-fp8 here has fp32 scales"
    )


def test_a_clip_that_names_no_grid_keeps_todays_refusal_in_a_mixed_recipe(tiny):
    """``grid3`` names no grid: it is not handed to some units and refused for the rest, but refused as before.

    Hopper stores the two W4A16 spellings in one file (asymmetric body, symmetric attention); the text is
    the body step's own grid3 refusal and the override changes nothing about it.
    """
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.step import Refused
    from qda_quant.target import Target

    def refused(overrides):
        r = Recipe.from_dict(
            {
                "steps": [{"gptq": {"clip": "grid3"}}],
                "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
            }
        )
        with pytest.raises(Refused) as e:
            compile_plan(
                str(tiny / "m"), r, Format.parse("w4a16-asym-g128"), Target.parse("sm90"),
                [torch.device("cpu")], overrides=overrides,
            )  # fmt: skip
        return str(e.value)

    text = "gptq clip='grid3': grid3 requires symmetric W4A8 g128 and a target with FP8 group scales"
    assert refused([(ATTENTION, "w4a16-sym-g128")]) == text
    assert refused([]) == text


def test_a_mixed_moe_build_keeps_the_search_on_its_nvfp4_banks(built_moe):
    """q36's shape: NVFP4 expert banks beside FP8 attention. The bank solve keeps the clip; attention does not."""
    for step in ("rtn", "gptq"):
        searched, plain = _stored(built_moe[f"{step}_h"]), _stored(built_moe[step])
        attn = [
            f"model.layers.0.self_attn.{p}.{t}"
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            for t in ("weight", "weight_scale")
        ]
        experts = [
            f"{MOE_EXPERT}.{p}.{t}"
            for p in ("gate_proj", "up_proj", "down_proj")
            for t in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
        ]
        for name in attn:
            assert torch.equal(searched[name], plain[name]), (step, name)
        assert any(not torch.equal(searched[name], plain[name]) for name in experts), step


def test_plan_says_which_formats_take_the_default_clip(tiny, capsys):
    """One line beside the format lines names the formats each clip runs on and those that take the default."""
    from qda_quant.cli import main

    cfg = {
        "model": str(tiny / "m"),
        "format": "w4a4-nvfp4",
        "target": "sm100",
        "out": None,
        "devices": ["cpu"],
        "format_overrides": [{"pattern": ATTENTION, "format": "w8a8-fp8"}],
        "steps": [{"gptq": {"actorder": "static", "damp": 0.01, "clip": "fp4hmse"}}],
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
    }
    path = tiny / "clip.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["plan", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "clip fp4hmse: w4a4-nvfp4 units; w8a8-fp8 units take minmax" in lines
