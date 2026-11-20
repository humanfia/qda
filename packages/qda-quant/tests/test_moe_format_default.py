"""MoE experts default to symmetric: a spelling that names no symmetry inherits it per unit class.

A bare ``w4a16`` still stores dense layers asymmetric, but the routed expert banks (and the shared experts
that live in them) resolve symmetric, at the body format's group size. A spelling that names the symmetry
(``w4a16-asym-g128``, the canonical name a record carries) pins every unit; ``format_overrides`` still gives
a bank its own scheme. Dense models are untouched.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml
from qda_quant import REGISTRY, build
from qda_quant.calib.sources import FileSource
from qda_quant.config import BuildConfig
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.target import Target

from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids
from .tiny import model


def config(tmp, **over):
    return {
        "model": str(tmp / "m"),
        "format": "w4a16",
        "devices": ["cpu"],
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 2, "seqlen": 16}},
        **over,
    }


def _save(tmp, m):
    m.save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(tmp / "c.jsonl", [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(2)], sha)
    return tmp


@pytest.fixture
def dense(tmp_path):
    return _save(tmp_path, tiny_qwen3(layers=2, hidden=128, inter=256))


@pytest.fixture
def moe(tmp_path):
    return _save(tmp_path, model("qwen3_moe"))


@pytest.fixture
def shared_moe(tmp_path):
    return _save(tmp_path, model("qwen2_moe"))


@pytest.fixture
def mla_moe(tmp_path):
    """MLA attention with fused routed experts beside shared ones; every quantized width is a multiple of 64."""
    return _save(tmp_path, model("deepseek_v2"))


def recipe(tmp):
    return Recipe(
        steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(tmp / "c.jsonl"), samples=2, seqlen=16)
    )


def plan(tmp, fmt, **kw):
    return compile_plan(str(tmp / "m"), recipe(tmp), Format.parse(fmt), Target(), [torch.device("cpu")], **kw)


def routers(p):
    return {bank.router for lg in p.graph.layers for bank in lg.banks}


def test_only_a_spelling_without_a_symmetry_inherits_one():
    """`symmetric_defaulted` is the provenance of the symmetry, not the symmetry itself."""
    bare = Format.parse("w4a16")
    assert bare.symmetric is False and bare.symmetric_defaulted
    assert bare.expert_format.symmetric and bare.expert_format.name == "w4a16-sym-g128"
    for named in ("w4a16-asym-g128", "w4a16-sym-g128"):
        f = Format.parse(named)
        assert not f.symmetric_defaulted and f.expert_format == f
    w3 = Format.parse("w3a16")  # its family defaults symmetric: marked, but the experts already are
    assert w3.symmetric_defaulted and w3.expert_format == w3


def test_the_config_keeps_the_spelling_it_was_given(dense):
    """`format` is canonicalized (`w4a16` -> `w4a16-asym-g128`); the plan still knows it was a bare spelling."""
    bare = BuildConfig.from_dict(config(dense))
    assert bare.format == "w4a16-asym-g128"
    assert bare.body_format.symmetric_defaulted
    named = BuildConfig.from_dict(config(dense, format="w4a16-asym-g128"))
    assert not named.body_format.symmetric_defaulted


def test_a_dense_model_is_unchanged(dense):
    p = plan(dense, "w4a16")
    assert p.fmt.name == "w4a16-asym-g128"
    assert p.format_defaults == {} and not p.defaulted
    assert p.formats == {}  # every unit keeps the body format


def test_a_bare_spelling_stores_the_routed_experts_symmetric(moe):
    p = plan(moe, "w4a16")
    assert p.fmt.name == "w4a16-asym-g128"  # dense layers and attention stay asymmetric
    assert p.format_defaults == {"routed_experts": "w4a16-sym-g128"}
    assert p.defaulted == frozenset(routers(p))
    assert {f.name for key, f in p.formats.items() if key in routers(p)} == {"w4a16-sym-g128"}
    assert {kind for _, kind in p.plans} == {"moe"}
    assert {name for name, _ in p.plans} == {"w4a16-sym-g128"}


def test_shared_experts_that_live_in_the_bank_default_symmetric_too(shared_moe):
    """Shared experts are kept bf16 by default; when a config quantizes them, they follow the bank's default."""
    p = plan(shared_moe, "w4a16-g64", keep=frozenset())
    assert p.format_defaults == {"routed_experts": "w4a16-sym-g64", "shared_experts": "w4a16-sym-g64"}
    shared = {name for lg in p.graph.layers for bank in lg.banks for name in bank.shared}
    keys = {g.modules[0] for lg in p.graph.layers for g in lg.groups if set(g.modules) & shared}
    assert keys and keys <= p.defaulted  # a group is keyed by its first module (gate/up are one fused group)


def test_the_group_size_is_the_body_formats(moe):
    assert plan(moe, "w4a16-g64").format_defaults == {"routed_experts": "w4a16-sym-g64"}


def test_an_explicit_asym_pins_every_unit(moe):
    p = plan(moe, "w4a16-asym-g128")
    assert p.format_defaults == {} and not p.defaulted and p.formats == {}


def test_an_expert_rule_pins_every_bank_over_the_default(moe):
    """A rule claims the units it names: a config that names the expert banks' symmetry keeps it."""
    p0 = plan(moe, "w4a16")
    banks = {bank.router for lg in p0.graph.layers for bank in lg.banks}
    p = plan(moe, "w4a16", overrides=[(r"re:^model\.layers\.\d+\.mlp\.experts$", "w4a16-asym-g64")])
    assert p.format_defaults == {} and not p.defaulted
    assert set(p.formats) == banks
    assert {f.name for f in p.formats.values()} == {"w4a16-asym-g64"}
    assert {name for name, kind in p.plans if kind == "moe"} == {"w4a16-asym-g64"}


def test_a_plan_records_the_spelling_so_a_replay_stores_the_same_experts(moe):
    out = build(
        str(moe / "m"), recipe(moe), format="w4a16", out_dir=moe / "out", devices=["cpu"],
        dtype=torch.float32, log=lambda s: None,
    )  # fmt: skip
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["format"] == "w4a16"  # the bare spelling: reading it back re-applies the default
    assert resolved["provenance"]["format_defaults"] == {"routed_experts": "w4a16-sym-g128"}
    assert json.loads((out / "build.json").read_text())["format_defaults"] == {
        "routed_experts": "w4a16-sym-g128"
    }
    assert BuildConfig.from_dict(resolved).body_format.symmetric_defaulted
    # the bare spelling parses to a marked copy; its provenance still names the registered plugin
    plugin = resolved["provenance"]["plugins"]["w4a16"]
    assert plugin["group"] == "qda.formats" and plugin["package"]
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    schemes = {(g["weights"]["symmetric"], tuple(g["targets"])) for g in groups.values()}
    assert any(symmetric and any(t.startswith("re:") for t in targets) for symmetric, targets in schemes)


def test_a_dense_build_record_is_unchanged(dense):
    """Nothing changed for a model without a routed bank: the record keeps the canonical name and no default."""
    out = build(
        str(dense / "m"), recipe(dense), format="w4a16", out_dir=dense / "out", devices=["cpu"],
        dtype=torch.float32, log=lambda s: None,
    )  # fmt: skip
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["format"] == "w4a16-asym-g128"
    assert "format_defaults" not in resolved["provenance"]
    assert "format_defaults" not in json.loads((out / "build.json").read_text())


def test_plan_prints_which_class_went_symmetric(moe, capsys):
    from qda_quant.cli import main

    path = moe / "p.yaml"
    path.write_text(yaml.safe_dump(config(moe, out=None), sort_keys=False))
    assert main(["plan", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "format_defaults: routed experts stored w4a16-sym-g128 (the format names no symmetry)" in lines
    assert not any(line.startswith("format_overrides:") for line in lines)


def test_plan_does_not_warn_about_zero_points_the_default_removed(moe, capsys):
    """The SGLang zero-point warning describes asymmetric experts; the default stores them symmetric."""
    from qda_quant.cli import main

    asym = moe / "asym.yaml"
    asym.write_text(yaml.safe_dump(config(moe, format="w4a16-asym-g128", out=None), sort_keys=False))
    assert main(["plan", str(asym)]) == 0
    assert "asymmetric per-expert INT4" in capsys.readouterr().out  # an explicit -asym still warns
    bare = moe / "bare.yaml"
    bare.write_text(yaml.safe_dump(config(moe, out=None), sort_keys=False))
    assert main(["plan", str(bare)]) == 0
    out = capsys.readouterr().out
    assert "asymmetric per-expert INT4" not in out
    assert "format_defaults: routed experts stored w4a16-sym-g128" in out


def test_an_mla_moe_defaults_its_fused_banks_at_the_spellings_group(mla_moe):
    """DeepSeek's fused routed banks follow the default; group 128 cannot divide this fixture's widths."""
    from qda_quant.step import Refused

    with pytest.raises(Refused, match="not a whole number of groups"):
        plan(mla_moe, "w4a16")
    p = plan(mla_moe, "w4a16-g64")
    assert p.format_defaults == {"routed_experts": "w4a16-sym-g64"}
    assert set(p.formats) == routers(p)  # the router takes the default with the bank; no user rule


@pytest.mark.parametrize(
    "body,override,expected",
    [
        ("w4a16", None, ()),
        ("w4a16-asym-g128", None, (0, 1)),
        ("w4a16", (r"re:^model\.layers\.\d+\.mlp\.experts$", "w4a16-asym-g64"), (0, 1)),
        ("w4a16", ("model.layers.0.mlp.experts", "w4a16-asym-g128"), (0,)),
        ("w4a16-asym-g128", (r"re:^model\.layers\.\d+\.mlp\.experts$", "w4a16-sym-g128"), ()),
    ],
)
def test_plan_zero_point_warning_names_only_resolved_asymmetric_banks(moe, capsys, body, override, expected):
    """Expert rules can restore zero points over a bare body or remove them over an explicit asym body."""
    from qda_quant.cli import main

    settings = config(moe, format=body, out=None)
    if override:
        settings["format_overrides"] = [{"pattern": override[0], "format": override[1]}]
    path = moe / "resolved-warning.yaml"
    path.write_text(yaml.safe_dump(settings))
    assert main(["plan", str(path)]) == 0
    warnings = [line for line in capsys.readouterr().out.splitlines() if "asymmetric per-expert INT4" in line]
    assert len(warnings) == bool(expected)
    for i in range(2):
        assert any(f"model.layers.{i}.mlp.experts" in warning for warning in warnings) == (i in expected)
    if warnings:
        assert f"INT4 g{64 if override and override[1].endswith('g64') else 128}" in warnings[0]


def test_dense_plan_with_asymmetric_override_keeps_moe_warning_silent(dense, capsys):
    """The same format spelling and dense rule write no routed zero points."""
    from qda_quant.cli import main

    path = dense / "dense-warning.yaml"
    path.write_text(
        yaml.safe_dump(
            config(
                dense,
                format_overrides=[
                    {"pattern": "model.layers.0.self_attn.o_proj", "format": "w4a16-asym-g64"},
                ],
            )
        )
    )
    assert main(["plan", str(path)]) == 0
    assert "asymmetric per-expert INT4" not in capsys.readouterr().out
