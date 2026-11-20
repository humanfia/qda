"""The derived-group cache: a candidate's unit map expands without the model, to the same tables a graph gives."""

from __future__ import annotations

import re

import pytest
import yaml
from qda_quant.config import BuildConfig
from qda_quant.gguf import file, groups
from qda_quant.gguf.candidates import expand_recipe, units
from qda_quant.step import Refused

from .test_export import WIDE, _source


def _config(src, calib, template, checkout, **gguf):  # noqa: ANN001, ANN202
    return {
        "model": str(src),
        "format": "w4a16-asym-g32",
        "devices": ["cpu"],
        "steps": [{"rtn": {"grid": "q4k"}}],
        "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        "out": str(src.parent / "out"),
        "exporter": {"gguf": {"llama_cpp": str(checkout), "template": str(template), **gguf}},
    }


def _setup(tmp_path):  # noqa: ANN001, ANN202
    """A tiny model, its template, and the config the candidates will run; the graph comes from its own plan."""
    pytest.importorskip("gguf")
    src, calib, template, checkout, _layers = _source(tmp_path, WIDE)
    base = BuildConfig.from_dict(_config(src, calib, template, checkout))
    return base, file.Template(template), src, calib, template, checkout


def test_the_cache_expands_exactly_what_the_live_graph_does(tmp_path) -> None:  # noqa: ANN001
    """The cache is not an interpretation: `expand` equals `expand_recipe` on the graph it was derived from."""
    base, template, *_ = _setup(tmp_path)
    graph = base.plan().graph
    qkv = next(uid for uid in units(graph) if "layers.0.self_attn.q_proj" in uid)
    down = next(uid for uid in units(graph) if "layers.0.mlp.down_proj" in uid)
    data = groups.derive(graph, template, model=base.model)
    assert qkv in data["units"]
    spec = {qkv: "Q6_K", down: "Q8_0"}
    assert groups.expand(data, spec) == expand_recipe(template, graph, spec)


def _moe_template(tmp_path):  # noqa: ANN001, ANN202
    """A one-block ``qwen35moe`` template: its expert banks are stored three-dimensional, as llama.cpp does."""
    gguf = pytest.importorskip("gguf")
    import torch

    from .test_rule import _qwen35moe_tensors

    rows = [
        (name, ne, kind)
        for name, ne, kind in _qwen35moe_tensors(experts=4)
        if name.startswith(("blk.0.", "token_embd", "output"))
    ]
    path = tmp_path / "moe.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen35moe")
    writer.add_block_count(1)
    for name, ne, _kind in rows:
        dtype = gguf.GGMLQuantizationType.F32 if len(ne) == 1 else gguf.GGMLQuantizationType.BF16
        stored = torch.zeros(tuple(reversed(ne)), dtype=torch.float32 if len(ne) == 1 else torch.bfloat16)
        if len(ne) != 1:
            stored = stored.view(torch.int16)
        writer.add_tensor(name, stored.numpy(), raw_shape=tuple(reversed(ne)), raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return file.Template(path)


def test_a_three_dimensional_bank_unit_expands_as_the_live_graph_does(tmp_path) -> None:  # noqa: ANN001
    """A bank's stored projections are 3-D; the unit map moves them in the type the unit solved.

    The live derivation reads every tensor the template stores of a unit's modules, whatever its rank. The
    cache must replay that derivation; sending its pairs back through the class map's two-dimensional rule
    refused the same unit, and the unknown-unit refusal differed between the two paths.
    """
    from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph

    template = _moe_template(tmp_path)
    down = "model.layers.0.mlp.experts.down_proj"
    graph = ModelGraph(
        model_type="qwen3_5_moe_text",
        layers=(LayerGraph(0, "model.layers.0", (LinearGroup((down,), "mlp_out"),)),),
        embed="model.embed_tokens",
        lm_head="lm_head",
        final_norm="model.norm",
    )
    assert template.tensors["blk.0.ffn_down_exps.weight"].shape == (512, 256, 4)
    live = expand_recipe(template, graph, {down: "Q6_K"})
    assert [rule["pattern"] for rule in live["tensor_types"]] == ["blk.0.ffn_down_exps.weight"]
    data = groups.derive(graph, template)
    assert groups.expand(data, {down: "Q6_K"}) == live
    # one owner: an unknown unit is refused with one message, whichever path read the table
    with pytest.raises(Refused) as from_live:
        expand_recipe(template, graph, {down + "_x": "Q6_K"})
    with pytest.raises(Refused) as from_cache:
        groups.expand(data, {down + "_x": "Q6_K"})
    assert str(from_live.value) == str(from_cache.value)


def test_the_cache_records_the_template_it_belongs_to(tmp_path) -> None:  # noqa: ANN001
    base, template, *_ = _setup(tmp_path)
    data = groups.derive(base.plan().graph, template)
    assert data["version"] == groups.VERSION
    assert data["template"]["bytes"] == template.path.stat().st_size
    assert re.fullmatch(r"[0-9a-f]{64}", data["template"]["sha256"])
    # every unit is kept only with the tensors the default mix stores of it
    assert all(entry["tensors"] for entry in data["units"].values())


def test_a_type_map_expands_into_the_config_beside_its_cache(tmp_path) -> None:  # noqa: ANN001
    """The search's shape: `groups` + `type_map`, no model and no template read by the expansion."""
    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    graph = base.plan().graph
    qkv = next(uid for uid in units(graph) if "layers.0.self_attn.q_proj" in uid)
    down = next(uid for uid in units(graph) if "layers.0.mlp.down_proj" in uid)
    where = groups.write(groups.derive(graph, template), tmp_path / groups.NAME)
    path = tmp_path / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(
            _config(src, calib, tpl, checkout, groups=groups.NAME, type_map={qkv: "Q6_K", down: "Q8_0"})
        )
    )
    got = BuildConfig.from_file(path)
    stored = {r["pattern"]: r["type"] for r in got.exporter["gguf"]["tensor_types"]}
    assert stored["blk.0.attn_q.weight"] == "Q6_K"
    assert stored["blk.0.ffn_down.weight"] == "Q8_0"
    assert got.exporter["gguf"]["groups"] == str(where)
    (pair,) = got.format_overrides
    assert pair["format"] == "w6a16-sym-g16"
    assert re.match(pair["pattern"][3:], "model.layers.0.self_attn.q_proj")
    # the expanded config plans exactly as a hand-written one would
    plan = got.plan()
    assert plan.output_bytes_exact and plan.output_bytes > 0


def test_a_type_map_without_its_cache_is_refused(tmp_path) -> None:  # noqa: ANN001
    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    qkv = next(uid for uid in units(base.plan().graph) if "layers.0.self_attn.q_proj" in uid)
    with pytest.raises(Refused, match="needs exporter.gguf.groups"):
        BuildConfig.from_dict(_config(src, calib, tpl, checkout, type_map={qkv: "Q8_0"}))


def test_a_type_map_and_tensor_types_together_are_refused(tmp_path) -> None:  # noqa: ANN001
    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    qkv = next(uid for uid in units(base.plan().graph) if "layers.0.self_attn.q_proj" in uid)
    where = groups.write(groups.derive(base.plan().graph, template), tmp_path / groups.NAME)
    with pytest.raises(Refused, match="both name the stored types"):
        BuildConfig.from_dict(
            _config(
                src,
                calib,
                tpl,
                checkout,
                groups=str(where),
                type_map={qkv: "Q8_0"},
                tensor_types=[{"pattern": "blk.0.attn_q.weight", "type": "Q4_K"}],
            )
        )


def test_an_unknown_unit_is_refused_with_the_ids_this_cache_has(tmp_path) -> None:  # noqa: ANN001
    base, template, *_ = _setup(tmp_path)
    data = groups.derive(base.plan().graph, template)
    with pytest.raises(Refused, match="names no solve unit"):
        groups.expand(data, {"model.layers.0.mlp.up_proj": "Q6_K"})


def test_a_type_the_rows_cannot_hold_is_refused_at_config_time(tmp_path) -> None:  # noqa: ANN001
    """WIDE's attention output is 4 heads x 32 = 128 values: a 256-value K-quant does not fit it."""
    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    graph = base.plan().graph
    out = next(uid for uid in units(graph) if "layers.0.self_attn.o_proj" in uid)
    where = groups.write(groups.derive(graph, template), tmp_path / groups.NAME)
    with pytest.raises(Refused, match=r"rows of 128 values need a multiple of 256"):
        BuildConfig.from_dict(_config(src, calib, tpl, checkout, groups=str(where), type_map={out: "Q6_K"}))


def test_a_build_writes_the_types_the_map_named(tmp_path) -> None:  # noqa: ANN001
    """Not just the plan: the file the writer produces stores the mapped types, layer by layer."""
    from qda_quant.gguf.writer import OUT_NAME

    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    graph = base.plan().graph
    down = next(uid for uid in units(graph) if "layers.0.mlp.down_proj" in uid)
    where = groups.write(groups.derive(graph, template), tmp_path / groups.NAME)
    cfg = BuildConfig.from_dict(
        _config(src, calib, tpl, checkout, groups=str(where), type_map={down: "Q8_0"})
        | {"out": str(tmp_path / "built")}
    )
    cfg.build()
    written = file.Template(tmp_path / "built" / OUT_NAME)
    names = {13: "Q5_K", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K", 0: "F32"}
    assert names[written.tensors["blk.0.ffn_down.weight"].tensor_type] == "Q8_0"
    # the other layer keeps the default mix: the map moved exactly the unit it named
    assert names[written.tensors["blk.1.ffn_down.weight"].tensor_type] != "Q8_0"


def test_a_cache_of_another_version_is_refused_not_reinterpreted(tmp_path) -> None:  # noqa: ANN001
    base, template, *_ = _setup(tmp_path)
    data = groups.derive(base.plan().graph, template)
    where = groups.write({**data, "version": groups.VERSION + 1}, tmp_path / groups.NAME)
    with pytest.raises(Refused, match="version"):
        groups.load(where)


def test_a_format_the_recipe_already_states_is_not_stated_twice(tmp_path) -> None:  # noqa: ANN001
    """The arm's own `format_overrides` and the type map name the same unit: one entry, not two."""
    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    graph = base.plan().graph
    down = next(uid for uid in units(graph) if "layers.0.mlp.down_proj" in uid)
    where = groups.write(groups.derive(graph, template), tmp_path / groups.NAME)
    stated = groups.expand(groups.load(where), {down: "Q6_K"})["format_overrides"]
    assert stated, "Q6_K must need a pair for this test to mean anything"
    for written in (stated, [(r["pattern"], r["format"]) for r in stated]):
        got = BuildConfig.from_dict(
            _config(src, calib, tpl, checkout, groups=str(where), type_map={down: "Q6_K"})
            | {"format_overrides": written}
        )
        assert got.format_overrides == stated


def test_an_engine_caller_that_skips_the_config_is_refused_not_silently_defaulted(tmp_path) -> None:  # noqa: ANN001
    """`type_map` is expanded by the config; an exporter dict passed straight to the engine says so."""
    from qda_quant.gguf.export import GGUF

    base, template, src, calib, tpl, checkout = _setup(tmp_path)
    exp = GGUF(llama_cpp=str(checkout), template=str(tpl), type_map={"x": "Q6_K"})
    assert "type_map was not expanded" in (exp.refusal(base.plan(), base.recipe) or "")


def test_the_cli_writes_the_cache_from_the_configs_own_plan(tmp_path, capsys) -> None:  # noqa: ANN001
    """`qda sensitivity groups CONFIG --template T --out F`: the derive step a search host runs once."""
    from qda_quant import cli

    _base, _template, src, calib, tpl, checkout = _setup(tmp_path)
    conf = tmp_path / "base.yaml"
    conf.write_text(yaml.safe_dump(_config(src, calib, tpl, checkout)))
    out = tmp_path / groups.NAME
    assert cli.main(["sensitivity", "groups", str(conf), "--template", str(tpl), "--out", str(out)]) == 0
    data = groups.load(out)
    assert data["units"] and data["arch"]
    assert groups.NAME in capsys.readouterr().out


def test_a_missing_cache_names_the_command_that_writes_it(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(Refused, match="qda sensitivity groups"):
        groups.load(tmp_path / "absent.json")
