"""A group-divisible expert bank can still violate the selected serving kernel's hidden tile."""

import pytest
import torch
from qda_quant import REGISTRY, Format, Refused, Target, build
from qda_quant.calib.sources import FileSource
from qda_quant.engine.planning import compile_plan
from qda_quant.recipe import Recipe
from qda_quant.target import BINDINGS, TARGETS, KernelPlan

from .engine_util import save_tiny_tokenizer, tiny_gpt_oss, tiny_qwen3_moe, write_ids


def _plan(tmp, model, target=None, fmt="w4a16-sym-g64", overrides=(), steps=None):
    model.save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    write_ids(tmp / "c.jsonl", [[3, 4, 5, 6]], sha)
    return compile_plan(
        str(tmp / "m"),
        Recipe(
            steps=steps or [REGISTRY["rtn"]()],
            calib=FileSource(path=str(tmp / "c.jsonl"), samples=1, seqlen=4),
        ),
        Format.parse(fmt),
        target or Target(),
        [torch.device("cpu")],
        overrides=overrides,
    )


@pytest.mark.parametrize("make", [tiny_qwen3_moe, tiny_gpt_oss])
@pytest.mark.parametrize("hidden", [192, 256])
def test_expert_hidden_tile_is_checked_before_cuda(tmp_path, monkeypatch, make, hidden):
    """Canonical D, not transposed storage or group divisibility, controls Marlin's unpadded hidden tile."""

    def no_cuda(*args, **kwargs):
        pytest.fail("planning must not query CUDA for this CPU source")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_cuda)
    model = make(layers=1, hidden=hidden, inter=64)
    marlin = Target.parse("sm90@0.33.0:marlin")
    if hidden == 192:
        with pytest.raises(Refused, match=r"expert bank model.layers.0.mlp.experts.*hidden 192 % 128") as exc:
            _plan(tmp_path, model, marlin)
        assert "hidden % 128 == 0" in str(exc.value)
        assert "target: sm90@0.33.0:humming" in str(exc.value)
    else:
        assert _plan(tmp_path, model, marlin).plan_moe.kernel == "MarlinExperts"
    plan = _plan(tmp_path, model, Target.parse("sm90@0.33.0:humming"))
    assert plan.plan_moe.kernel == "HummingIndexedExperts"
    assert plan.plan_moe.serve_args == {"moe_backend": "humming"}


def test_expert_override_cannot_bypass_hidden_rule(tmp_path):
    """An INT3 body fits Humming; overriding every bank to INT4 on `:marlin` selects Marlin and must check its shape."""
    model = tiny_qwen3_moe(layers=2, hidden=192, inter=64)
    rules = [(r"re:^model\.layers\.\d+\.mlp\.experts$", "w4a16-sym-g64")]
    with pytest.raises(Refused, match=r"expert bank .*hidden 192 % 128"):
        _plan(tmp_path, model, Target.parse("sm90@0.33.0:marlin"), fmt="w3a16-sym-g64", overrides=rules)
    plan = _plan(tmp_path, model, Target.parse("sm90@0.33.0:humming"), "w3a16-sym-g64", rules)
    assert plan.plans[("w4a16-sym-g64", "moe")].kernel == "HummingIndexedExperts"


def test_head_only_recipe_keeps_float_experts(tmp_path):
    """A model QuantStep that only packs the head must not validate untouched float banks against Marlin."""
    _plan(
        tmp_path,
        tiny_gpt_oss(hidden=192, inter=64),
        steps=[REGISTRY["head"](format="w4a16-sym-g64")],
    )


def test_packed_source_still_checks_expert_shapes(tmp_path):
    """Packed stages retain source settings: Humming fits; a recorded Marlin pin must still be refused."""
    import json

    target = Target.parse("sm90@0.33.0:humming")
    _plan(tmp_path, tiny_gpt_oss(hidden=192, inter=64), target)
    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=1, seqlen=4)
    packed = build(
        str(tmp_path / "m"),
        Recipe(steps=[REGISTRY["rtn"]()], calib=calib),
        format="w4a16-sym-g64",
        target=target,
        devices=["cpu"],
        out_dir=tmp_path / "packed",
        log=lambda _: None,
    )
    recipe = Recipe(steps=[], calib=calib)
    compile_plan(str(packed), recipe, None, Target(), [torch.device("cpu")])
    record = json.loads((packed / "build.json").read_text())
    record["target"] = "sm90@0.33.0"
    record["serve"]["args"]["moe_backend"] = "marlin"
    (packed / "build.json").write_text(json.dumps(record))
    with pytest.raises(Refused, match=r"expert bank .*hidden 192 % 128"):
        compile_plan(str(packed), recipe, None, Target(), [torch.device("cpu")])


def test_merged_marlin_pin_cannot_serve_int3_override(tmp_path):
    """The body's recorded `:marlin` pin also applies to an overridden expert format that cannot run that kernel."""
    with pytest.raises(Refused, match=r"expert bank .*Marlin has no INT3"):
        _plan(
            tmp_path,
            tiny_qwen3_moe(hidden=256, inter=64),
            Target.parse("sm90@0.33.0:marlin"),
            overrides=[(r"re:^model\.layers\.\d+\.mlp\.experts$", "w3a16-sym-g64")],
        )


def test_target_plugin_validates_every_bank_through_existing_registry(tmp_path, monkeypatch):
    """Plugins receive canonical expert dimensions and the merged settings, without a new registry group."""
    calls = []

    def table(target, fmt, kind):
        return KernelPlan(f"plugin_{kind}", serve_args={"moe_backend": "plugin"})

    def binds(target, fmt, shape, args, env):
        calls.append((shape, args))
        return "plugin_moe"

    monkeypatch.setitem(TARGETS._entries, "testmoe", table)
    monkeypatch.setitem(BINDINGS._entries, "testmoe_moe", binds)
    _plan(tmp_path, tiny_gpt_oss(layers=2, hidden=192, inter=64), Target("testmoe"))
    assert calls == [((192, 64), {"moe_backend": "plugin"})] * 2
