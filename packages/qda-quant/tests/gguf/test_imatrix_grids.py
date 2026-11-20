"""Native weighted scales and GPTQ codes must survive IQ4/Q3 GGUF encoding exactly."""

import ctypes
import json

import numpy as np
import pytest
import torch
from qda_quant import Target
from qda_quant.clips import CLIPS
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from qda_quant.gguf import file, kernel
from qda_quant.gguf.iq4 import KVALUES
from qda_quant.gguf.writer import GGUFWriter
from qda_quant.solve.quant import Quantizer
from qda_quant.solve.solve import gptq
from qda_quant.step import Refused

from .test_export import WIDE, _source
from .test_native_types import _native, _reference_decode


@pytest.mark.parametrize(
    ("grid_name", "kind_name", "format_name"),
    [
        ("iq4xs", "IQ4_XS", "w4a16-iq4xs"),
        ("iq4nl", "IQ4_NL", "w4a16-iq4nl"),
        ("q3k", "Q3_K", "w3a16-q3k"),
    ],
)
def test_weighted_scales_and_solved_codes_reach_the_native_reader(
    tmp_path, monkeypatch, ggml_lib, grid_name, kind_name, format_name
):
    gguf = pytest.importorskip("gguf")
    kind = int(gguf.GGMLQuantizationType[kind_name])
    fmt = Format.parse(format_name)
    q = Quantizer.of(fmt, Target(arch="llama_cpp").plan(fmt, "dense"), grid_name)
    clip = CLIPS["imatrix"]
    lib = ctypes.CDLL(str(ggml_lib))
    kernel.use_ggml(ggml_lib)
    try:
        # Different importances per expert, signed scales, zero rows and a partial super-block.
        w = torch.randn(2, 3, 512, generator=torch.Generator().manual_seed(17)) * 0.05
        w[:, 0] = 0
        w[:, 1, :32] *= 1e-9
        imp = torch.logspace(-3, 3, 512).repeat(2, 1)
        imp[1] = imp[1].flip(0)
        scale, zero = clip.choose(q, w, imp)
        assert zero is None
        assert (scale < 0).any() and (scale == 0).any()
        live_groups = w.reshape(2, 3, -1, fmt.group_size).abs().amax(-1) > 0
        assert ((scale == 0) & live_groups).any()
        codes = gptq(w, torch.diag_embed(imp), q, scale, zero, impl="torch")
        assert codes.isfinite().all()
        assert (codes.reshape(2, 3, -1, fmt.group_size)[scale == 0] == 0).all()
        for e in range(2):
            raw = _native(lib, kind, w[e].numpy(), imp[e].numpy())
            # Decode the native bytes independently of QDA's field unpacking.
            decoded = _reference_decode(lib, kind, raw, w[e].numel()).reshape(w[e].shape)
            wide = scale[e].repeat_interleave(fmt.group_size, -1)
            # Native fitted codes need not match GPTQ, but their values must use these exact scales.
            if kind_name == "Q3_K":
                ratios = np.divide(decoded, wide.numpy(), out=np.zeros_like(decoded), where=wide.numpy() != 0)
                assert np.array_equal(ratios, ratios.round()) and (abs(ratios) <= 4).all()
            else:
                table_values = wide.numpy()[..., None] * KVALUES.astype(np.float32)
                assert (decoded[..., None] == table_values).any(-1).all()
            c = fmt.group_size * 9
            padded = torch.nn.functional.pad(w[e, :, :c], (0, -c % (32 if grid_name == "iq4nl" else 256)))
            weights = torch.nn.functional.pad(imp[e, :c], (0, padded.shape[-1] - c))
            full, _ = q.grid.fit_weighted(padded, weights[None])
            partial, _ = q.grid.fit_weighted(w[e, :, :c], imp[e : e + 1, :c])
            assert torch.equal(partial, full[:, : c // fmt.group_size])
        with pytest.raises(ValueError, match="weights its fit by the Hessian diagonal; none was given"):
            clip.choose(q, w)
        with pytest.raises(Refused, match=f"whole number of {fmt.group_size}-value groups"):
            q.grid.fit_weighted(w[..., :-1], imp[..., :-1])
    finally:
        kernel.use_ggml(None)
    with pytest.raises(Refused, match="set exporter.gguf.libggml"):
        q.grid.fit_weighted(w, imp)

    src, calib, template, _, _ = _source(tmp_path, {**WIDE, "num_hidden_layers": 1})
    module, tensor = "model.layers.0.mlp.down_proj", "blk.0.ffn_down.weight"
    captured = {}
    add_layer = GGUFWriter.add_layer

    def capture(self, ctx):
        captured.update({k: v for k, v in ctx.qparams.items() if k == module})
        return add_layer(self, ctx)

    monkeypatch.setattr(GGUFWriter, "add_layer", capture)
    cfg = BuildConfig(
        model=str(src),
        formats={"default": "w4a16-asym-g32", module: format_name},
        grid="q4k",
        devices=["cpu"],
        steps=[{"gptq": {"impl": "torch", "clip": "imatrix"}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "libggml": str(ggml_lib),
                "tensor_types": [{"pattern": tensor, "type": kind_name}],
            }
        },
        out=str(tmp_path / "out"),
    )
    cfg.plan()
    out = cfg.build(log=lambda _: None)
    qp = captured[module]
    codes = qp["codes"]
    assert (codes == codes.round()).all() and codes.min() >= fmt.qmin and codes.max() <= fmt.qmax
    wide = qp["scale"].repeat_interleave(fmt.group_size, -1)
    expected = q.grid.value(codes.float(), wide, None).numpy()
    stored = file.Template(out / "model-Q4_K_M.gguf")
    assert stored.tensors[tensor].tensor_type == kind
    decoded = _reference_decode(lib, kind, stored.raw(tensor), codes.numel())
    assert np.array_equal(decoded, expected.reshape(-1))
    assert tensor not in json.loads((out / "build.json").read_text())["gguf"]["rounded_from_template"]
