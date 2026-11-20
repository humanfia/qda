"""Quantization grids: selection in the config, plan refusals, and the checkpoint-equals-kernel invariant (CPU)."""

from __future__ import annotations

import shutil

import pytest
import torch
from qda_quant import REGISTRY, Target
from qda_quant.arch import describe
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from qda_quant.output.export import Writer
from qda_quant.solve.packing import unpack
from qda_quant.solve.quant import Quantizer
from qda_quant.step import Refused
from safetensors import safe_open
from safetensors.torch import save_file

from .ctx import contexts
from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids


@pytest.fixture
def cfg(tmp_path):
    """A plannable config dict for a saved tiny Qwen3."""
    tiny_qwen3(layers=1, hidden=128, inter=256).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [list(range(32))] * 2, sha)
    return {
        "model": str(tmp_path / "m"),
        "format": "w4a16-sym-g128",
        "devices": ["cpu"],
        "steps": [{"gptq": {}}],
        "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 2, "seqlen": 32}},
    }


def test_signed8_scales_are_auto_rounds():
    """Same scale bits as auto-round's own rule, whose sign a reader easily gets backwards; all-zero groups too."""
    int_rule = pytest.importorskip("auto_round.data_type.int")
    fmt = Format.parse("w4a16-sym-g128")
    weights = torch.randn(40, 256, generator=torch.Generator().manual_seed(0))
    weights = weights * torch.logspace(-7, 1, 40)[:, None]
    weights[3] = 0
    scale, zero = Quantizer.of(fmt, Target().plan(fmt, "dense"), "signed8").params(weights)
    _, expected, _ = int_rule.quant_tensor_rtn_sym(weights.clone(), bits=4, group_size=128)
    assert zero is None and (scale < 0).any() and (scale > 0).any()
    assert torch.equal(scale.to(torch.float16), expected.reshape(scale.shape))


@pytest.mark.parametrize("name", ["absmax", "signed8"])
def test_checkpoint_holds_what_the_kernel_computes(name, tmp_path):
    """From the bytes on disk alone, as vLLM loads them (bf16 scales), the served weight is the one the build kept.

    Clipped on purpose: at min/max, ``absmax / 8`` of bf16 weights is already on the bf16 grid, so fp16 and bf16 agree.
    """
    model = tiny_qwen3(layers=1, hidden=128, inter=256).to(torch.bfloat16).eval()
    model.save_pretrained(tmp_path / "src")
    fmt = "w4a16-sym-g128"
    writer = Writer(tmp_path / "out", tmp_path / "src", describe(model, verify=False), fmt, Target())
    for ctx in contexts(model, fmt, tokens=256):
        REGISTRY["gptq"](impl="torch", grid=name, clip="mse").apply_layer(ctx)
        writer.add_layer(ctx)
    writer.add_model(model, {})
    out = writer.finish(model.config, None, {})

    stored = {}
    for path in out.glob("*.safetensors"):
        with safe_open(path, "pt") as h:
            stored.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118 (safe_open has no __iter__)
    stems = [k.removesuffix(".weight_packed") for k in stored if k.endswith(".weight_packed")]
    assert stems
    rerounded = 0
    for stem in stems:
        scale = stored[f"{stem}.weight_scale"]
        assert scale.dtype == {"absmax": torch.bfloat16, "signed8": torch.float16}[name]
        shape = torch.Size(stored[f"{stem}.weight_shape"].tolist())
        codes = unpack(stored[f"{stem}.weight_packed"], 4, shape, packed_dim=1).float()
        kernel_scale = scale.to(torch.bfloat16).float()
        rerounded += int((kernel_scale != scale.float()).sum())
        served = codes * kernel_scale.repeat_interleave(128, -1)
        assert torch.equal(served.to(torch.bfloat16), model.get_submodule(stem).weight.data)
    assert (rerounded > 0) == (name == "signed8")


def test_grid_selection_and_its_record(cfg):
    """The recipe's grid fills the steps that name none, and the record still lets ``--set grid`` reach them."""
    built = BuildConfig.from_dict(
        cfg | {"grid": "signed8", "steps": [{"awq": {"grid": "absmax"}}, {"gptq": {}}]}
    )
    assert [s.grid for s in built.recipe.steps] == ["absmax", "signed8"]
    assert built.grid == "signed8" and "grid" not in built.steps[1]["gptq"]
    assert built.steps[0]["awq"]["grid"] == "absmax"
    again = BuildConfig.from_dict(
        built.model_dump(exclude={"provenance"}), ["grid=absmax", "steps.0.grid=signed8"]
    )
    assert [s.grid for s in again.recipe.steps] == ["signed8", "absmax"]

    with pytest.raises(Refused, match=r"steps\.0\.gptq\.grid: unknown grid 'x'; known: .*absmax"):
        BuildConfig.from_dict(cfg | {"steps": [{"gptq": {"grid": "x"}}]})
    with pytest.raises(Refused, match=r"grid: unknown grid 'x'"):
        BuildConfig.from_dict(cfg | {"grid": "x"})
    with pytest.raises(
        Refused, match=r"steps\.0\.rotate\.grid: unknown key"
    ):  # a step with no grid has no knob
        BuildConfig.from_dict(cfg | {"steps": [{"rotate": {"grid": "absmax"}}]})


@pytest.mark.parametrize(
    ("over", "refused"),
    [
        ({"format": "w4a16", "grid": "signed8"}, "symmetric"),
        ({"format": "w4a8", "grid": "signed8"}, "fp8_e4m3"),
        ({"format": "w4a8", "steps": [{"awq": {"grid": "signed8"}}, {"gptq": {}}]}, "awq.*fp8_e4m3"),
        # the head is checked in its own format: the body's FP8 kernel does not matter, its asymmetry does
        ({"format": "w4a8", "steps": [{"head": {"grid": "signed8"}}, {"gptq": {}}]}, None),
        ({"steps": [{"head": {"format": "w4a16", "grid": "signed8"}}, {"gptq": {}}]}, "head.*symmetric"),
        ({"format": "w4a8", "grid": "absmax"}, None),
    ],
)
def test_plan_refuses_a_grid_the_kernel_cannot_serve(cfg, over, refused):
    plan = BuildConfig.from_dict(cfg | over).plan
    if refused is None:
        plan()
    else:
        with pytest.raises(Refused, match=refused):
            plan()


@pytest.mark.parametrize(
    ("over", "grids"),
    [
        ({}, ["signed8", "absmax"]),  # autoround's own default; gptq takes the recipe's
        ({"format": "w4a16"}, ["absmax", "absmax"]),  # signed8 cannot serve it: the default
        ({"grid": "absmax"}, ["absmax", "absmax"]),  # a named grid wins, even the default
        ({"grid": "signed8"}, ["signed8", "signed8"]),
        ({"steps": [{"autoround": {"iters": 1, "grid": "absmax"}}, {"gptq": {}}]}, ["absmax", "absmax"]),
    ],
)
def test_a_step_default_grid_yields_to_a_named_one(cfg, over, grids):
    """`autoround` defaults to `signed8` where it serves, and any grid the config names wins, at either level."""
    built = BuildConfig.from_dict(cfg | {"steps": [{"autoround": {"iters": 1}}, {"gptq": {}}]} | over)
    built.plan()
    assert [s.grid for s in built.recipe.steps] == grids


def test_an_unnamed_grid_survives_serialization(cfg, tmp_path):
    """A recipe that names no grid builds the same bytes from the object, its YAML and the build's record."""
    from qda_quant import Recipe, build

    recipe = Recipe.from_dict({"steps": [{"autoround": {"iters": 2}}], "calib": cfg["calib"]})
    text = recipe.to_yaml()  # before any build: planning names the recipe's grids
    kw = {"format": "w4a16-sym-g128", "devices": ["cpu"]}
    outs = [build(cfg["model"], recipe, out_dir=tmp_path / "a", **kw)]
    outs.append(build(cfg["model"], text, out_dir=tmp_path / "b", **kw))
    again = BuildConfig.from_file(outs[0] / "resolved.yaml", [f"out={tmp_path / 'c'}"])
    assert [s.grid for s in again.recipe.steps] == ["signed8"]
    outs.append(again.build())
    shards = [b"".join(p.read_bytes() for p in sorted(o.glob("*.safetensors"))) for o in outs]
    assert shards[0] == shards[1] == shards[2]

    planned = BuildConfig.from_dict(BuildConfig.from_dict(cfg | {"steps": [{"autoround": {}}]}).model_dump())
    planned.plan()
    assert planned.grid == "absmax" and planned.steps[0]["autoround"]["grid"] == "signed8"
    reread = BuildConfig.from_dict(planned.resolved())
    assert [s.grid for s in reread.recipe.steps] == ["signed8"]


def test_signed8_refuses_a_scale_fp16_cannot_hold():
    """Fp16 overflows to infinity, and infinite scales give NaN weights; bf16 (`absmax`) holds the same range."""
    fmt = Format.parse("w4a16-sym-g128")
    plan = Target().plan(fmt, "dense")
    weights = torch.full((1, 128), 1048576.0)
    with pytest.raises(Refused, match="overflows fp16"):
        Quantizer.of(fmt, plan, "signed8").params(weights)
    with pytest.raises(Refused, match="overflows fp16"):  # a scale no range rule made: GSQ trains its own
        Quantizer.of(fmt, plan, "signed8").stored_scale(torch.full((1, 1), 1e6))
    absmax = Quantizer.of(fmt, plan, "absmax")
    scale, zero = absmax.params(weights)
    assert absmax.dequant(absmax.quantize(weights, scale, zero), scale, zero).isfinite().all()


def test_soup_refuses_scales_that_differ_in_sign(cfg, tmp_path):
    """Identical codes do not imply identical signs: a mean of magnitudes would silently keep the first input's."""
    base = BuildConfig.from_dict(
        cfg | {"grid": "signed8", "steps": [{"rtn": {}}], "out": str(tmp_path / "a")}
    ).build()
    flipped = tmp_path / "b"
    shutil.copytree(base, flipped)
    shard = next(flipped.glob("*.safetensors"))
    with safe_open(shard, "pt") as h:
        tensors, metadata = {k: h.get_tensor(k) for k in h.keys()}, h.metadata()  # noqa: SIM118
    name = next(k for k in tensors if k.endswith(".weight_scale"))
    original = tensors[name].clone()
    tensors[name][0, 0] = -tensors[name][0, 0]
    save_file(tensors, shard, metadata=metadata)

    def soup(other, out):
        steps = [{"soup": {"with": [str(other)], "weights": [0, 1]}}]
        source = {k: v for k, v in cfg.items() if k != "format"} | {"model": str(base)}
        return BuildConfig.from_dict(source | {"steps": steps, "out": str(tmp_path / out)}).build()

    with pytest.raises(Refused, match="differ in sign"):
        soup(flipped, "refused")
    same = soup(base, "same")  # shared signs still average, keeping the negative scales
    with safe_open(next(same.glob("*.safetensors")), "pt") as h:
        assert torch.equal(h.get_tensor(name) < 0, original.to(torch.bfloat16) < 0)


def test_the_q4k_fit_codes_match_a_separate_coding_pass():
    """The q4k grid's fit makes the codes anyway; RTN's one-pass path must keep the same three tensors."""
    import qda_quant.gguf.q4k  # noqa: F401  (registers the grid)

    fmt = Format.parse("w4a16-asym-g32")
    q = Quantizer.of(fmt, Target().plan(fmt, "dense"), "q4k")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(4, 256, generator=g).mul(0.1).to(torch.bfloat16).float()
    s, z = q.params(w)
    made = q.fit_codes(w)
    assert made is not None
    s2, z2, codes = made
    assert torch.equal(s2.cpu(), s.cpu()) and torch.equal(z2.cpu(), z.cpu())
    assert torch.equal(codes.float().cpu(), q.quantize(w, s, z).cpu())
