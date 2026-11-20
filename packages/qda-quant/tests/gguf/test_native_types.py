"""Native template fitting of the types with no Python port (IQ4_XS, IQ4_NL, Q3_K, IQ3_S).

Each is checked against llama.cpp's own type-specific fit (the function ``llama-quantize`` reaches through
``ggml_quantize_chunk``) and decoded by gguf-py, whose values must be the reference decoder's.
"""

import ctypes
import json

import numpy as np
import pytest
from qda_quant.config import BuildConfig
from qda_quant.gguf import file, imatrix, rule

from .test_export import WIDE, _imatrix_file, _source

#: ggml type -> (upstream name, the suffix of its ``quantize_*`` / ``dequantize_row_*`` symbols, bytes, values)
NATIVE = {
    rule.IQ4_XS: ("IQ4_XS", "iq4_xs", 136, 256),
    rule.IQ4_NL: ("IQ4_NL", "iq4_nl", 18, 32),
    rule.Q3_K: ("Q3_K", "q3_K", 110, 256),
    rule.IQ3_S: ("IQ3_S", "iq3_s", 110, 256),
}


def test_the_native_types_are_named_and_laid_out_as_gguf_py_lays_them_out():
    gguf = pytest.importorskip("gguf")
    assert set(NATIVE) == set(rule.NATIVE)
    for kind, (name, _, size, block) in NATIVE.items():
        assert file.type_id(name) == kind == int(gguf.GGMLQuantizationType[name])
        assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[name]] == (block, size)
        assert file.row_bytes(kind, 512) == 512 // block * size
        with pytest.raises(ValueError, match=f"multiple of {block}"):
            file.row_bytes(kind, block + 16)


def _native(lib: ctypes.CDLL, kind: int, values: np.ndarray, weights: np.ndarray | None) -> bytes:
    """llama.cpp's own fit of `values` as `kind`, called by its type-specific symbol."""
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init(kind)  # IQ3_S's grid tables; a no-op for the others
    fn = getattr(lib, f"quantize_{NATIVE[kind][1]}")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_size_t
    out = np.empty(values.shape[0] * file.row_bytes(kind, values.shape[1]), dtype=np.uint8)
    wrote = fn(
        values.ctypes.data, out.ctypes.data, *values.shape, None if weights is None else weights.ctypes.data
    )
    assert wrote == out.size
    return out.tobytes()


def _reference_decode(lib: ctypes.CDLL, kind: int, raw: bytes, n: int) -> np.ndarray:
    """llama.cpp's ``dequantize_row_*`` of `raw`: the decoder llama.cpp itself runs."""
    fn = getattr(lib, f"dequantize_row_{NATIVE[kind][1]}")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    src = np.frombuffer(raw, dtype=np.uint8).copy()
    out = np.empty(n, dtype=np.float32)
    fn(src.ctypes.data, out.ctypes.data, n)
    return out


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", list(NATIVE))
def test_native_types_write_llama_cpps_fit_and_decode_as_llama_cpp_does(tmp_path, ggml_lib, kind, weighted):
    """An override to a native type stores llama.cpp's fit of the template (weighted when a matrix is given),
    gguf-py decodes it to exactly llama.cpp's values, and the plan's sizes are the file's."""
    gguf = pytest.importorskip("gguf")
    name = NATIVE[kind][0]
    src, calib, template, _, _ = _source(tmp_path, WIDE)
    tensor = "blk.0.attn_q.weight"
    matrix = _imatrix_file(tmp_path / "m.imatrix.gguf", template, skip=set())
    cfg = BuildConfig(
        model=str(src),
        format="w4a16-asym-g32",
        grid="q4k",
        devices=["cpu"],
        steps=[{"gptq": {"impl": "torch"}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "libggml": str(ggml_lib),
                "tensor_types": [{"pattern": tensor, "type": name}],
                **({"imatrix": str(matrix)} if weighted else {}),
            }
        },
        out=str(tmp_path / "out"),
    )
    plan = cfg.plan()
    out = cfg.build(log=lambda _: None)
    path = out / "model-Q4_K_M.gguf"
    result = file.Template(path)
    values = file.Template(template).values(tensor).float().numpy()
    weights = imatrix.load(matrix)[tensor] if weighted else None
    lib = ctypes.CDLL(str(ggml_lib))
    # The upstream type-specific fit, independently of QDA's chunk dispatch and layout arithmetic.
    expected = _native(lib, kind, values, weights)
    assert result.raw(tensor) == expected
    assert result.tensors[tensor].tensor_type == kind
    decoded = gguf.quants.dequantize(np.frombuffer(expected, dtype=np.uint8), gguf.GGMLQuantizationType[name])
    assert np.array_equal(decoded.reshape(-1), _reference_decode(lib, kind, expected, values.size))
    err = np.linalg.norm(decoded.reshape(values.shape) - values) / np.linalg.norm(values)
    assert err < 0.25, f"{name} round trip relative error {err:.3f}"
    assert plan.output_bytes_exact and plan.output_bytes == path.stat().st_size
    assert plan.weight_bytes == sum(int(t.n_bytes) for t in gguf.GGUFReader(path).tensors)
    record = json.loads((out / "build.json").read_text())["gguf"]
    assert "codes dropped" in record["rounded_from_template"][tensor]
    assert ("imatrix" in record) is weighted
