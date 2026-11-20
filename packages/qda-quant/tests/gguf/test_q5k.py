"""Q5_K packing against native ggml, and solved dense/expert tensors through a real build."""

import json

import numpy as np
import pytest
import torch
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from qda_quant.gguf import file, kernel, rule
from qda_quant.gguf.q5k import Q5K_FORMAT
from qda_quant.solve.grid import GRIDS
from qda_quant.step import Refused


@pytest.mark.parametrize("kind", ["Q5_K", "IQ4_XS", "IQ4_NL", "Q3_K", "IQ3_S"])
def test_plan_refuses_plain_only_library_before_native_build(tmp_path, kind):
    from .test_export import WIDE, _source
    from .test_kernel import compile_quantizer

    src, calib, template, _, _ = _source(tmp_path, WIDE)
    lib = compile_quantizer(tmp_path)
    cfg = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={"gguf": {"template": str(template), "libggml": str(lib)}},
        out=str(tmp_path / "out"),
    )
    cfg.plan()  # The plain-only library remains valid for the existing Q4_K/Q6_K mix.
    cfg.exporter["gguf"]["tensor_types"] = [{"pattern": "blk.0.ffn_down.weight", "type": kind}]
    with pytest.raises(Refused, match=f"{kind} requires native ggml_quantize_chunk"):
        cfg.plan()


def test_plan_reads_the_checkouts_own_libggml_when_the_setting_is_empty(tmp_path, monkeypatch):
    """With `libggml` unset, `qda plan` checks the library the build would load (the checkout's
    build*/bin/libggml-base), not the setting alone: it names that library instead of asking for the setting."""
    import shutil

    from .test_export import WIDE, _source
    from .test_kernel import compile_quantizer

    src, calib, template, _, _ = _source(tmp_path, WIDE)
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    found = checkout / "build" / "bin" / "libggml-base.so"
    shutil.copy(compile_quantizer(tmp_path), found)  # plain-only: no ggml_quantize_chunk
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    cfg = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "llama_cpp": str(checkout),
                "tensor_types": [{"pattern": "blk.0.ffn_down.weight", "type": "Q5_K"}],
            }
        },
        out=str(tmp_path / "out"),
    )
    with pytest.raises(Refused, match=f"Q5_K requires native ggml_quantize_chunk in {found}"):
        cfg.plan()


@pytest.mark.parametrize("weighted", [False, True])
def test_native_codes_round_trip_and_edited_high_bits_decode(ggml_lib, weighted):
    gguf = pytest.importorskip("gguf")
    kind = gguf.GGMLQuantizationType.Q5_K
    kernel.use_ggml(ggml_lib)
    try:
        torch.manual_seed(838)
        w = torch.randn(2, 512) * 0.05
        w[:, 64:96] = -0.03
        importance = torch.rand(512) if weighted else None
        raw = kernel.quantize_q5k(w, importance)
        d, dmin, sc, m, codes = kernel.q5k_fields(raw)
        assert kernel.q5k_records(d, dmin, sc, m, codes) == raw
        # Exercise every fifth-bit position with codes chosen independently of the fit.
        codes[:] = (np.arange(256, dtype=np.uint16) * 7 + np.arange(256) // 32) % 32
        edited = kernel.q5k_records(d, dmin, sc, m, codes)
        decoded = gguf.quants.dequantize(np.frombuffer(edited, dtype=np.uint8), kind)
        scale = d.view("<f2").astype(np.float32)[:, None] * sc
        zero = dmin.view("<f2").astype(np.float32)[:, None] * m
        assert np.array_equal(
            decoded, (scale[:, :, None] * codes.reshape(-1, 8, 32) - zero[:, :, None]).reshape(-1)
        )
        grid = GRIDS["q5k"]
        scale, zero = (
            grid.fit_weighted(w[None], importance[None])
            if weighted
            else grid.fit(Format.parse(Q5K_FORMAT), w)
        )
        wide = scale.reshape(2, -1).repeat_interleave(32, -1)
        zwide = zero.reshape(2, -1).repeat_interleave(32, -1)
        values = grid.value(grid.codes(w, wide, zwide, 0, 31), wide, zwide)
        assert np.array_equal(
            values.numpy().reshape(-1),
            gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), kind),
        )
    finally:
        kernel.use_ggml(None)


@pytest.mark.parametrize("bank", [False, True])
def test_gptq_q5k_codes_reach_dense_and_expert_files(tmp_path, monkeypatch, ggml_lib, bank):
    gguf = pytest.importorskip("gguf")
    if bank:
        from .test_experts import _source

        src, calib, template = _source(tmp_path)
        monkeypatch.setattr(
            rule, "q4_k_m_types", lambda arch, n, tensors: {name: kind for name, _, kind in tensors}
        )
        module, tensor = "model.layers.0.mlp.experts", "blk.0.ffn_down_exps.weight"
    else:
        from .test_export import WIDE, _source

        src, calib, template, _, _ = _source(tmp_path, WIDE)
        module, tensor = "model.layers.0.mlp.down_proj", "blk.0.ffn_down.weight"
    pattern = "re:blk\\.0\\.ffn_.*_exps\\.weight$" if bank else tensor
    cfg = BuildConfig(
        model=str(src),
        formats={"default": "w4a16-asym-g32", module: Q5K_FORMAT},
        grid="q4k",
        devices=["cpu"],
        steps=[{"gptq": {"impl": "torch", "clip": "imatrix"}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "libggml": str(ggml_lib),
                "tensor_types": [{"pattern": pattern, "type": "Q5_K"}],
            }
        },
        out=str(tmp_path / "out"),
    )
    plan = cfg.plan()
    out = cfg.build(log=lambda _: None)
    written = out / "model-Q4_K_M.gguf"
    got, original = file.Template(written), file.Template(template)
    assert got.tensors[tensor].tensor_type == rule.Q5_K
    assert tensor not in json.loads((out / "build.json").read_text())["gguf"]["rounded_from_template"]
    assert got.raw(tensor) != kernel.quantize_q5k(original.values(tensor))
    assert plan.output_bytes_exact and plan.output_bytes == written.stat().st_size
    assert plan.weight_bytes == sum(int(t.n_bytes) for t in gguf.GGUFReader(written).tensors)
