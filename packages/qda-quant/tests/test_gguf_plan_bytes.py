"""A plan must state the size of the file a GGUF build actually writes, not of a checkpoint it never writes."""

from __future__ import annotations

import pytest
from qda_quant.config import BuildConfig

from .gguf.test_export import WIDE, _source

Q6K_RULE = [{"pattern": "blk.0.attn_q.weight", "type": "Q6_K"}]


def _config(tmp_path, out, template, src, calib, checkout, rules=()):
    return BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                **({"template": str(template)} if template is not None else {}),
                "llama_cpp": str(checkout),
                "tensor_types": list(rules),
            }
        },
        out=str(tmp_path / out),
    )


@pytest.mark.parametrize("rules", [(), tuple(Q6K_RULE)], ids=["default_mix", "q6k_override"])
def test_planned_bytes_are_the_bytes_the_build_writes(tmp_path, rules):
    """Both figures must match the finished GGUF exactly, with and without a Q6_K type rule.

    The payload is read back with upstream's own reader, not this package's layout, so the test cannot pass
    by repeating the same arithmetic twice.
    """
    gguf = pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    cfg = _config(tmp_path, "out", template, src, calib, checkout, rules)
    plan = cfg.plan()
    assert plan.output_bytes_exact is True
    assert plan.exporter_builds == ()
    out = cfg.build(log=lambda _: None)
    written = out / "model-Q4_K_M.gguf"
    assert plan.output_bytes == written.stat().st_size
    payload = sum(int(t.n_bytes) for t in gguf.GGUFReader(written).tensors)
    assert plan.weight_bytes == payload
    # the container is not free, and a weight budget must not be compared with the file
    assert 0 < plan.weight_bytes < plan.output_bytes


def test_a_q6k_rule_moves_the_planned_bytes(tmp_path):
    """The number must follow the mix: a tensor promoted to Q6_K makes the file bigger, before it is written.

    Without this, `output_bytes` could be any constant that happens to match one build.
    """
    pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    plain = _config(tmp_path, "plain", template, src, calib, checkout).plan()
    promoted = _config(tmp_path, "promoted", template, src, calib, checkout, Q6K_RULE).plan()
    assert promoted.output_bytes > plain.output_bytes
    assert promoted.weight_bytes > plain.weight_bytes


def test_the_estimate_a_gguf_plan_replaces_is_not_the_gguf_size(tmp_path):
    """Guard the reason this exists: the engine's checkpoint estimate is the wrong artefact for a GGUF.

    The same model and recipe planned for the default exporter keeps that estimate, and it is not the size of
    the GGUF. If some future estimate happens to match the file, this test stops being evidence and should be
    deleted rather than loosened -- `test_planned_bytes_are_the_bytes_the_build_writes` must still hold.
    """
    pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    cfg = _config(tmp_path, "out", template, src, calib, checkout)
    real = (cfg.build(log=lambda _: None) / "model-Q4_K_M.gguf").stat().st_size
    estimated = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(tmp_path / "ct"),
    ).plan()
    assert estimated.output_bytes_exact is False
    assert estimated.output_bytes != real
    assert estimated.weight_bytes == 0  # unknown, and must not read as a budget figure


def test_a_build_without_a_template_says_it_would_convert_one(tmp_path):
    """A candidate that names no template reconverts the whole source; a search must be able to see that."""
    pytest.importorskip("gguf")
    src, calib, _, checkout, _ = _source(tmp_path, WIDE)
    plan = _config(tmp_path, "own", None, src, calib, checkout).plan()
    assert plan.exporter_builds == ("template",)
    assert plan.output_bytes_exact is False  # no template to read a layout from
    assert plan.weight_bytes == 0
    assert plan.output_bytes > 0  # the engine's estimate still reserves something


def test_plan_prints_the_exact_byte_figure_and_the_converter_warning(tmp_path, capsys):
    """`qda plan` is what the agent reads: a rounded GiB cannot be compared with a byte cap."""
    pytest.importorskip("gguf")
    import yaml
    from qda_quant.cli import main

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)

    def written(name, cfg):
        path = tmp_path / name
        # only what the test set: a defaulted `target` in the file is refused for a gguf output
        spec = cfg.model_dump(mode="json", exclude_unset=True)
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        return str(path)

    cfg = _config(tmp_path, "out", template, src, calib, checkout)
    assert main(["plan", written("plan.yaml", cfg)]) == 0
    printed = capsys.readouterr().out
    plan = cfg.plan()
    assert f"({plan.output_bytes} bytes, {plan.weight_bytes} in weights)" in printed
    assert "exporter builds its own" not in printed

    own = _config(tmp_path, "own", None, src, calib, checkout)
    assert main(["plan", written("own.yaml", own)]) == 0
    printed = capsys.readouterr().out
    assert "exporter builds its own template first" in printed
    assert "checkpoint" in printed and "bytes)" not in printed  # an estimate must not look exact


def test_other_exporters_keep_the_engine_estimate(tmp_path):
    """Only an exporter that answers replaces the estimate; the default must not claim exactness."""
    src, calib, _, _, _ = _source(tmp_path, WIDE)
    cfg = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(tmp_path / "ct"),
    )
    plan = cfg.plan()
    assert plan.output_bytes_exact is False
    assert plan.weight_bytes == 0
    assert plan.exporter_builds == ()
    assert plan.output_bytes > 0
