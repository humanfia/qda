"""The ``formats:`` recipe: one mapping from a class pattern to the format a build stores it in.

It compiles to the keys a build already takes (``format``, ``format_overrides``, ``ignore``), so every
plan-time refusal of those keys applies unchanged -- a partial fused unit, a unit two keys name, uncounted
experts and a second activation scheme are refused before a weight loads.
"""

from __future__ import annotations

import json
import re

import pytest
import torch
import yaml
from qda_quant.arch import KEEP_ROLES
from qda_quant.arch.derive import ignore_hits
from qda_quant.config import BuildConfig
from qda_quant.format_recipe import compile_formats
from qda_quant.step import Refused

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids
from ..goldens import check_golden, refusal

MLP_0_1 = [f"model.layers.{n}.mlp.{m}" for n in (0, 1) for m in ("gate_proj", "down_proj")]


@pytest.fixture
def tiny(tmp_path):
    tiny_qwen3(layers=2, hidden=64, inter=96).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(2)], sha
    )
    return tmp_path


def config(root, formats, **extra):
    return {
        "model": str(root / "m"),
        "devices": ["cpu"],
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(root / "c.jsonl"), "samples": 2, "seqlen": 16}},
        **extra,
        "formats": formats,
    }


def test_the_recipe_compiles_to_the_three_keys(tiny):
    cfg = BuildConfig.from_dict(
        config(tiny, {"default": "w4a16", "layers.0-1.mlp": "w8a16", "lm_head": "bf16"})
    )
    assert cfg.format == "w4a16-asym-g128"
    assert cfg.format_overrides == [
        {"pattern": r"re:(.*\.)?layers\.(0|1)\.mlp(\.|$)", "format": "w8a16-sym-g128"}
    ]
    assert cfg.ignore == sorted({*KEEP_ROLES, "lm_head"})  # bf16 entries add to the always-bf16 roles


def test_a_bare_default_keeps_its_inherited_symmetry(tiny):
    assert BuildConfig.from_dict(config(tiny, {"default": "w4a16"})).body_format.symmetric_defaulted
    assert BuildConfig.from_dict(config(tiny, {"default": "w4a16-sym"})).format == "w4a16-sym-g128"
    assert not BuildConfig.from_dict(config(tiny, {"default": "w4a16-sym"})).body_format.symmetric_defaulted


def test_a_layer_range_matches_whole_layers_and_a_suffix_under_them():
    _, rules, _ = compile_formats({"layers.0-3.mlp": "w8a16-g32"})
    pattern = rules[0]["pattern"][3:]  # the rule `ignore` applies: re.match from the start of the name
    assert re.match(pattern, "model.layers.2.mlp.down_proj")
    assert not re.match(pattern, "model.layers.4.mlp.down_proj")
    assert not re.match(pattern, "model.layers.2.mlp_x.down_proj")
    _, bare, _ = compile_formats({"layers.0-1": "w8a16-g32"})
    assert re.match(bare[0]["pattern"][3:], "model.layers.1.self_attn.q_proj")


def test_the_example_recipe_plans_the_class_it_names(tiny):
    plan = BuildConfig.from_dict(config(tiny, {"default": "w4a16-g32", "layers.0-1.mlp": "w8a16-g32"})).plan()
    assert set(plan.formats) == set(MLP_0_1)  # the fused gate/up unit is keyed by its first linear
    assert {f.name for f in plan.formats.values()} == {"w8a16-sym-g32"}
    assert plan.output_bytes > 0


def test_a_bf16_entry_keeps_that_unit_as_it_is(tiny):
    cfg = BuildConfig.from_dict(
        config(tiny, {"default": "w4a16-g32", "model.layers.1.self_attn.o_proj": "bf16"})
    )
    assert "model.layers.1.self_attn.o_proj" in cfg.ignore
    cfg.plan()  # the entry names a unit the build would otherwise quantize


def test_a_bf16_entry_wins_over_a_format_entry_that_also_names_the_unit(tiny):
    """``ignore`` removes a unit before overrides resolve, so a class format and a bf16 exception compose.

    This is "the mlp in W8 except down_proj"; the opposite request still refuses when the format rule reaches
    nothing quantized, naming the bf16 entry that keeps it.
    """
    plan = BuildConfig.from_dict(
        config(
            tiny,
            {
                "default": "w4a16-sym-g32",
                "lm_head": "bf16",
                "layers.0-0.mlp.down_proj": "bf16",
                "layers.0-0.mlp": "w8a16-g32",
            },
        )
    ).plan()
    assert ignore_hits(plan.keep, "model.layers.0.mlp.down_proj")  # keep holds the compiled patterns
    assert set(plan.formats) == {"model.layers.0.mlp.gate_proj"}  # the fused gate/up unit is the only match
    assert {f.name for f in plan.formats.values()} == {"w8a16-sym-g32"}
    assert not ignore_hits(plan.keep, "model.layers.1.mlp.down_proj")  # the range names layer 0 only
    with pytest.raises(Refused, match="kept bf16 by `ignore`"):
        BuildConfig.from_dict(
            config(
                tiny,
                {
                    "default": "w4a16-sym-g32",
                    "layers.0-0.mlp": "bf16",
                    "layers.0-0.mlp.down_proj": "w8a16-g32",
                },
            )
        ).plan()


def test_a_bf16_head_is_a_no_op_and_a_head_step_beside_it_is_refused(tiny):
    recipe = {"default": "w4a16-g32", "lm_head": "bf16", "layers.0-1.mlp": "w8a16-g32"}
    plan = BuildConfig.from_dict(config(tiny, recipe)).plan()
    assert "lm_head" in plan.keep  # no layer step quantizes the head, so the entry asks for the status quo
    steps = [{"head": {"format": "w4a16-sym-g64"}}, {"rtn": {}}]
    with pytest.raises(Refused, match="stores lm_head in w4a16-sym-g64, but `ignore` keeps it bf16"):
        BuildConfig.from_dict(config(tiny, recipe, steps=steps)).plan()
    BuildConfig.from_dict(
        config(tiny, {k: v for k, v in recipe.items() if k != "lm_head"}, steps=steps)
    ).plan()


def test_qda_plan_reads_the_recipe_from_a_yaml_file(tiny, capsys):
    path = tiny / "plan.yaml"
    path.write_text(
        yaml.safe_dump(
            config(tiny, {"default": "w4a16-g32", "lm_head": "bf16", "layers.0-1.mlp": "w8a16-g32"})
        )
    )
    from qda_quant.cli import main

    assert main(["plan", str(path)]) == 0
    assert "format_overrides: 4 units in ['w8a16-sym-g32']" in capsys.readouterr().out


def test_a_build_records_the_compiled_form_and_a_rebuild_reads_it(tiny):
    cfg = BuildConfig.from_dict(
        config(tiny, {"default": "w4a16-g32", "layers.0-1.mlp": "w8a16-g32"}, out=str(tiny / "out"))
    )
    out = cfg.build(dtype=torch.float32, log=lambda s: None)
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert "formats" not in resolved
    assert resolved["format"] == "w4a16-asym-g32"  # w4a16's inherited default, as a bare `format:` reads
    assert resolved["format_overrides"] == [
        {"pattern": r"re:(.*\.)?layers\.(0|1)\.mlp(\.|$)", "format": "w8a16-sym-g32"}
    ]
    BuildConfig.from_dict(resolved)  # resolved.yaml is the compiled form: it reads back as itself
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    assert groups["group_0"]["weights"]["num_bits"] == 4
    assert sorted(groups, key=lambda g: int(g.split("_")[1]))[-1] == "group_1"
    assert sorted(groups["group_1"]["targets"]) == sorted(
        f"model.layers.{n}.mlp.{m}" for n in (0, 1) for m in ("gate_proj", "up_proj", "down_proj")
    )


def test_the_public_recipe_with_bf16_head_plans(tiny):
    cfg = BuildConfig.from_dict(
        config(tiny, {"default": "w4a16-g32", "lm_head": "bf16", "layers.0-1.mlp": "w8a16-g32"})
    )
    plan = cfg.plan()
    assert set(plan.formats) == set(MLP_0_1)
    assert {f.name for f in plan.formats.values()} == {"w8a16-sym-g32"}


# ---------------------------------------------------------------- refusal wording: golden/errors_formats_recipe.txt
W4 = {"default": "w4a16-g32"}

# name -> (formats, extra config keys, plan too); the planner refuses what it refuses for an override list
REFUSALS = {
    "part of fused q/k/v": ({**W4, "layers.0-0.self_attn.q_proj": "w8a16-g32"}, {}, True),
    "part of fused gate/up": ({**W4, "layers.0-0.mlp.gate_proj": "w8a16-g32"}, {}, True),
    "second activation scheme": ({**W4, "layers.0-0.mlp.down_proj": "w8a8-g32"}, {}, True),
    "default bf16": ({"default": "bf16"}, {}, False),
    "empty recipe": ({}, {}, False),
    "range written high to low": ({"layers.3-1.mlp": "w4a16-g32"}, {}, False),
    "unknown format": ({"layers.0-1.mlp": "w4a99"}, {}, False),
    "format not a string": ({"layers.0-1.mlp": 4}, {}, False),
    "recipe beside format": (W4, {"format": "w4a16-g128"}, False),
    "recipe beside format_overrides": (W4, {"format_overrides": []}, False),
    "recipe beside ignore": (W4, {"ignore": []}, False),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusals_keep_their_wording(tiny, name):
    """Each ``formats:`` refusal fires at load or plan, with the exact text in the golden file."""
    formats, extra, planned = REFUSALS[name]

    def load():
        cfg = BuildConfig.from_dict(config(tiny, formats, **extra))
        return cfg.plan() if planned else cfg

    check_golden("formats_recipe", name, refusal(load).replace(str(tiny), "<tmp>"), list(REFUSALS))
