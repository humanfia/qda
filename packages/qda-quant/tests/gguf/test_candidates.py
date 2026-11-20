"""A candidate's per-class type map expands to exact rules, and its mistakes are refused before a build."""

from __future__ import annotations

import re

import pytest
import transformers
from qda_quant.arch import describe
from qda_quant.gguf import file
from qda_quant.gguf.candidates import (
    SOLVED_FORMATS,
    class_key,
    classes,
    expand,
    expand_recipe,
    solved_formats,
    unit_key,
    units,
)
from qda_quant.step import Refused

from .test_export import WIDE, _source

#: WIDE with 4 heads of 64 values: every q/k/v projection is 256 values wide, so a 256-value K-quant fits the
#: whole fused unit (WIDE's own 4 x 32 = 128-value attention cannot hold one).
WIDE_MHA = {**WIDE, "head_dim": 64, "num_key_value_heads": 4}


def _template(tmp_path, config=WIDE):  # noqa: ANN001, ANN202
    pytest.importorskip("gguf")
    _src, _calib, template, _checkout, _layers = _source(tmp_path, config)
    return file.Template(template)


def test_class_key_removes_only_the_layer_index() -> None:
    assert class_key("blk.12.ffn_down.weight") == "blk.*.ffn_down.weight"
    assert class_key("blk.0.attn_norm.weight") == "blk.*.attn_norm.weight"
    assert class_key("output.weight") == "output.weight"


def test_a_class_expands_to_one_exact_rule_per_tensor(tmp_path) -> None:  # noqa: ANN001
    """A class map names every layer; the rules are exact names, in sorted order, and resolve cleanly."""
    template = _template(tmp_path)
    assert set(classes(template)["blk.*.ffn_down.weight"]) == {
        "blk.0.ffn_down.weight",
        "blk.1.ffn_down.weight",
    }
    rules = expand(
        template,
        {"blk.*.ffn_down.weight": "Q4_K", "blk.*.attn_q.weight": "Q4_K", "output.weight": "Q8_0"},
    )
    assert [(r.pattern, r.type) for r in rules] == [
        ("blk.0.attn_q.weight", "Q4_K"),
        ("blk.0.ffn_down.weight", "Q4_K"),
        ("blk.1.attn_q.weight", "Q4_K"),
        ("blk.1.ffn_down.weight", "Q4_K"),
        ("output.weight", "Q8_0"),
    ]
    # the expansion is what the build's own resolver accepts: widths, norms and matches all checked there
    from qda_quant.gguf.tensor_types import resolve_types

    types = resolve_types(template, rules)
    assert types["blk.1.ffn_down.weight"] == file.type_id("Q4_K")


def test_an_exact_tensor_overrides_its_class(tmp_path) -> None:  # noqa: ANN001
    rules = expand(
        _template(tmp_path),
        {"blk.*.ffn_down.weight": "Q4_K"},
        overrides={"blk.1.ffn_down.weight": "Q6_K"},
    )
    assert {r.pattern: r.type for r in rules} == {
        "blk.0.ffn_down.weight": "Q4_K",
        "blk.1.ffn_down.weight": "Q6_K",
    }


def test_a_fused_unit_that_disagrees_is_refused_with_its_members_named(tmp_path) -> None:  # noqa: ANN001
    """The writer would refuse this after the solve; the candidate map refuses it before the build."""
    with pytest.raises(Refused, match=r"fused unit .*ffn_gate.*ffn_up.*members name 2"):
        expand(
            _template(tmp_path),
            {"blk.*.ffn_gate.weight": "Q6_K", "blk.*.ffn_up.weight": "Q4_K"},
            fused=[("blk.*.ffn_gate.weight", "blk.*.ffn_up.weight")],
        )


def test_a_fused_unit_whose_member_is_unnamed_is_refused(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(Refused, match=r"has no type in the map; name every member"):
        expand(
            _template(tmp_path),
            {"blk.*.ffn_gate.weight": "Q6_K"},
            fused=[("blk.*.ffn_gate.weight", "blk.*.ffn_up.weight")],
        )


def test_a_row_the_type_cannot_hold_is_refused_with_both_named(tmp_path) -> None:  # noqa: ANN001
    """WIDE's attention output is 4 heads x 32 = 128 values: Q8_0 fits it, a 256-value K-quant does not."""
    template = _template(tmp_path)
    assert template.tensors["blk.0.attn_output.weight"].shape[0] == 128
    with pytest.raises(Refused, match=r"blk\.0\.attn_output\.weight: type Q6_K: rows of 128 values need a"):
        expand(template, {"blk.*.attn_output.weight": "Q6_K"})
    rules = expand(template, {"blk.*.attn_output.weight": "Q8_0"})
    assert {r.type for r in rules} == {"Q8_0"}


def test_a_key_that_names_nothing_is_refused(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(Refused, match=r"names no two-dimensional tensor"):
        expand(_template(tmp_path), {"blk.*.no_such_projection.weight": "Q4_K"})


def test_solved_formats_name_the_pair_a_type_needs(tmp_path) -> None:  # noqa: ANN001
    """Q6_K/Q5_K codes can only be written as that type; Q8_0 and Q4_K need no override."""
    rules = expand(
        _template(tmp_path),
        {
            "blk.*.ffn_down.weight": "Q6_K",
            "blk.*.ffn_up.weight": "Q5_K",
            "blk.*.ffn_gate.weight": "Q8_0",
            "blk.0.attn_q.weight": "Q4_K",
            "blk.1.attn_q.weight": "Q4_K",
        },
    )
    pairs = solved_formats(rules)
    assert pairs["blk.0.ffn_down.weight"] == "w6a16-sym-g16"
    assert pairs["blk.1.ffn_up.weight"] == "w5a16-asym-g32"
    assert "blk.0.ffn_gate.weight" not in pairs and "blk.0.attn_q.weight" not in pairs
    assert SOLVED_FORMATS == {"Q6_K": "w6a16-sym-g16", "Q5_K": "w5a16-asym-g32"}


def _graph(tmp_path, config=WIDE_MHA):  # noqa: ANN001, ANN202
    src, _calib, template, _checkout, _layers = _source(tmp_path, config)
    return file.Template(template), describe(transformers.AutoModelForCausalLM.from_pretrained(src))


def test_units_are_the_graphs_fused_groups(tmp_path) -> None:  # noqa: ANN001
    _template_obj, graph = _graph(tmp_path)
    ids = units(graph)
    qkv = next(key for key in ids if "layers.0.self_attn.q_proj" in key)
    assert qkv.split("|") == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert unit_key(("a", "b")) == "a|b"
    # every layer's units are present, singles included (an output or down projection is a unit of one)
    assert sum("layers.0." in key for key in ids) == sum("layers.1." in key for key in ids)
    assert any("layers.0.mlp.down_proj" in key for key in ids)


def test_expand_recipe_writes_the_unit_type_and_its_solve_pair(tmp_path) -> None:  # noqa: ANN001
    template, graph = _graph(tmp_path)
    qkv = next(key for key in units(graph) if "layers.0.self_attn.q_proj" in key)
    tables = expand_recipe(template, graph, {qkv: "Q6_K"})
    stored = {rule["pattern"]: rule["type"] for rule in tables["tensor_types"]}
    assert stored["blk.0.attn_q.weight"] == "Q6_K"
    assert stored["blk.0.attn_k.weight"] == "Q6_K"
    assert stored["blk.0.attn_v.weight"] == "Q6_K"
    (pair,) = tables["format_overrides"]
    assert pair["format"] == "w6a16-sym-g16"
    assert pair["pattern"].startswith("re:^(?:")
    assert re.match(pair["pattern"][3:], "model.layers.0.self_attn.q_proj")
    assert not re.match(pair["pattern"][3:], "model.layers.1.self_attn.q_proj")


def test_expand_recipe_pairs_only_the_types_that_solve_codes(tmp_path) -> None:  # noqa: ANN001
    template, graph = _graph(tmp_path)
    qkv = next(key for key in units(graph) if "layers.0.self_attn.q_proj" in key)
    down = next(key for key in units(graph) if "layers.0.mlp.down_proj" in key)
    tables = expand_recipe(template, graph, {qkv: "Q8_0", down: "Q4_K"})
    assert {rule["type"] for rule in tables["tensor_types"]} == {"Q8_0", "Q4_K"}
    assert tables["format_overrides"] == []


def test_expand_recipe_refuses_a_unit_and_a_gguf_rule_that_disagree(tmp_path) -> None:  # noqa: ANN001
    template, graph = _graph(tmp_path)
    qkv = next(key for key in units(graph) if "layers.0.self_attn.q_proj" in key)
    with pytest.raises(Refused, match="one type per tensor"):
        expand_recipe(template, graph, {qkv: "Q6_K"}, types={"blk.0.attn_q.weight": "Q4_K"})


def test_expand_recipe_refuses_an_id_that_names_no_unit(tmp_path) -> None:  # noqa: ANN001
    template, graph = _graph(tmp_path)
    with pytest.raises(Refused, match="names no solve unit"):
        expand_recipe(template, graph, {"model.layers.0.mlp.up_proj": "Q6_K"})


def test_expand_recipe_refuses_a_unit_type_the_rows_cannot_hold(tmp_path) -> None:  # noqa: ANN001
    """A GGUF row width is ne[0]: o_proj's input is 4 heads x 32 = 128 values, so a 256-value K-quant fails."""
    template, graph = _graph(tmp_path, WIDE)
    out = next(key for key in units(graph) if "layers.0.self_attn.o_proj" in key)
    assert template.tensors["blk.0.attn_output.weight"].shape[0] == 128
    with pytest.raises(Refused, match="rows of 128 values need a multiple of 256"):
        expand_recipe(template, graph, {out: "Q6_K"})
