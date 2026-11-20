"""Tensor mixes must survive config replay and both writer paths without silently changing a requested type."""

from __future__ import annotations

import json

import pytest
from qda_quant.config import BuildConfig
from qda_quant.step import Refused

from .test_export import WIDE, _source


def _config(src, calib, template, out, tensor_types, steps=None, **gguf) -> BuildConfig:
    """The one-sample CPU Q4_K_M build of every test here: `steps` (default rtn) and `tensor_types` vary."""
    return BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=steps or [{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={"gguf": {"template": str(template), "tensor_types": tensor_types, **gguf}},
        out=str(out),
    )


@pytest.mark.parametrize("native", [False, True], ids=["portable-mix", "native-menu"])
def test_mixed_types_replay_and_match_the_layer_pass(tmp_path, monkeypatch, request, native):
    """Changing both solved Q4_K and template-only tensors must survive the fast path and resolved YAML."""
    gguf = pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    rules = [
        {"pattern": "re:blk\\.\\d+\\.attn_[qkv]\\.weight$", "type": "Q8_0"},
        {"pattern": "blk.0.attn_q.weight", "type": "Q6_K"},
        {"pattern": "blk.0.ffn_down.weight", "type": "Q4_K"},
        {"pattern": "output.weight", "type": "F16"},
        {"pattern": "blk.0.attn_norm.weight", "type": "F32"},
    ]
    extra = {}
    if native:
        extra["libggml"] = str(request.getfixturevalue("ggml_lib"))
        rules += [
            {"pattern": "blk.0.attn_v.weight", "type": "Q5_K"},
            {"pattern": "blk.0.ffn_gate.weight", "type": "IQ4_XS"},
            {"pattern": "blk.0.ffn_up.weight", "type": "Q8_0"},
            {"pattern": "blk.1.attn_q.weight", "type": "IQ4_XS"},
            {"pattern": "blk.1.ffn_down.weight", "type": "Q5_K"},
        ]
    cfg = _config(src, calib, template, tmp_path / "fast", rules, llama_cpp=str(checkout), **extra)
    # Whole tensors cross part boundaries; float and 32/256-value encodings have different row sizes.
    monkeypatch.setattr("qda_quant.gguf.writer._PART_VALUES", 700)
    fast = cfg.build(log=lambda _: None)
    plan = cfg.model_copy(update={"out": str(tmp_path / "planned")}).plan()
    replay = BuildConfig.from_file(fast / "resolved.yaml", overrides=[f"out={tmp_path / 'replay'}"])
    assert replay.exporter["gguf"]["tensor_types"] == rules
    slow = BuildConfig.from_file(
        fast / "resolved.yaml", overrides=[f"out={tmp_path / 'slow'}", "exporter.gguf.layer_pass=true"]
    ).build(log=lambda _: None)
    replay_out = replay.build(log=lambda _: None)
    data = (fast / "model-Q4_K_M.gguf").read_bytes()
    assert data == (slow / "model-Q4_K_M.gguf").read_bytes()
    assert data == (replay_out / "model-Q4_K_M.gguf").read_bytes()
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(fast / "model-Q4_K_M.gguf").tensors}
    assert types["blk.0.attn_q.weight"] == "Q6_K"
    assert types["blk.0.attn_k.weight"] == "Q8_0"
    assert types["blk.0.attn_v.weight"] == ("Q5_K" if native else "Q8_0")
    assert types["blk.0.ffn_down.weight"] == "Q4_K"
    assert types["output.weight"] == "F16" and types["blk.0.attn_norm.weight"] == "F32"
    assert json.loads((fast / "build.json").read_text())["gguf"]["mix"] == "custom"
    assert plan.output_bytes_exact and plan.output_bytes == len(data)
    assert plan.weight_bytes == sum(
        int(t.n_bytes) for t in gguf.GGUFReader(fast / "model-Q4_K_M.gguf").tensors
    )
    if native:
        assert types["blk.0.ffn_gate.weight"] == types["blk.1.attn_q.weight"] == "IQ4_XS"
        assert types["blk.0.ffn_up.weight"] == "Q8_0"
        assert types["blk.1.ffn_down.weight"] == "Q5_K"


def test_q4_override_keeps_gptq_codes_and_other_types_use_the_source(tmp_path):
    """Overriding a normally Q6_K tensor to Q4_K must store the solve rather than silently rounding again."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template
    from qda_quant.gguf.kernel import quantize_q4k
    from qda_quant.gguf.legacy import quantize_q8_0

    src, calib, template, _, _ = _source(tmp_path, WIDE)
    rules = [
        {"pattern": "blk.0.ffn_down.weight", "type": "Q4_K"},
        {"pattern": "blk.0.attn_q.weight", "type": "Q8_0"},
    ]
    cfg = _config(src, calib, template, tmp_path / "out", rules, steps=[{"gptq": {"impl": "torch"}}])
    out = Template(cfg.build(log=lambda _: None) / "model-Q4_K_M.gguf")
    source = Template(template)
    assert out.raw("blk.0.ffn_down.weight") != quantize_q4k(source.values("blk.0.ffn_down.weight"))
    assert out.raw("blk.0.attn_q.weight") == quantize_q8_0(source.values("blk.0.attn_q.weight"))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"pattern": "output.weight", "type": "Q4_K"}, "output.weight.*Q4_K.*128.*256"),
        ({"pattern": "re:missing", "type": "Q8_0"}, "missing.*matches no tensor"),
        ({"pattern": "blk.0.attn_norm.weight", "type": "Q6_K"}, "attn_norm.weight.*Q6_K.*norms require"),
    ],
)
def test_invalid_shape_or_unmatched_pattern_refuses_at_plan(tmp_path, override, match):
    """An explicit K type on a narrow row must fail rather than using the default 32-value fallback."""
    pytest.importorskip("gguf")
    src, calib, template, _, _ = _source(tmp_path, {**WIDE, "hidden_size": 128})
    cfg = _config(src, calib, template, tmp_path / "out", [override])
    with pytest.raises(Refused, match=match):
        cfg.plan()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["Q4_K", "Q6_K", "Q5_0", "Q8_0", "F16", "BF16"])
def test_aligned_non_f32_norm_refuses_at_plan(tmp_path, kind):
    """Encoding a block-aligned norm is possible, but stock llama.cpp requires F32 to multiply it."""
    pytest.importorskip("gguf")
    src, calib, template, _, _ = _source(tmp_path, WIDE)
    rule = {"pattern": r"re:blk\.0\.attn_norm\.weight$", "type": kind}
    cfg = _config(src, calib, template, tmp_path / "out", [rule])
    with pytest.raises(Refused, match=rf"blk\.0\.attn_norm\.weight.*{kind}.*norms require F32.*type: F32"):
        cfg.plan()
    assert not (tmp_path / "out").exists()


def test_f32_norm_override_supersedes_quantized_rule(tmp_path):
    """Only the final selected type is validated, so a later F32 rule may rescue a broad quantized rule."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template, type_id
    from qda_quant.gguf.tensor_types import TensorTypeRule, resolve_types

    _, _, template, _, _ = _source(tmp_path, WIDE)
    norm = "blk.0.attn_norm.weight"
    types = resolve_types(
        Template(template),
        [TensorTypeRule(pattern=norm, type="Q4_K"), TensorTypeRule(pattern=norm, type="F32")],
    )
    assert types[norm] == type_id("F32")
