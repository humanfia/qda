"""Q5_0 and Q8_0 records against ggml's ``quantize_row_q5_0_ref`` / ``quantize_row_q8_0_ref`` (llama.cpp 9e2a04a).

Golden bytes come from calling the C functions on these blocks; no llama.cpp is needed here. The blocks are the
ones a port gets wrong: an all-zero block led by ``-0.0`` (C's signed maximum stays ``+0.0``, so ``d`` is ``-0.0``),
a tie in magnitude (the first wins), a negative maximum, values on and just below half steps (rounding direction) and blocks so
small that ``1.0f / d`` is inf. C's ``(int8_t)`` of inf or nan is undefined; the golden bytes there are x86's (0),
which aarch64's C does not give (0xff for +inf), so those blocks are pinned here, not compared with a local build.
"""

import ctypes

import numpy as np
import pytest
import torch
from qda_quant.gguf.legacy import quantize_q5_0, quantize_q8_0


def _block(**values: float) -> np.ndarray:
    b = np.zeros(32, np.float32)
    for k, v in values.items():
        b[int(k[1:])] = v
    return b


BLOCKS = {
    "neg0_first": _block(i0=-0.0),
    "tie_neg_first": _block(i0=-3.0, i1=3.0),
    "neg_max": _block(i3=-5.0, i4=2.0),
    "half_steps": np.arange(32, dtype=np.float32) - 15.5,
    "below_half": _block(i0=127.0, i1=0.49999997, i2=-0.49999997, i3=126.5),  # fp32 floor(v + 0.5) gives 1
    "subnormal": np.full(32, 1e-40, np.float32),
    "subnormal_mixed": np.array([1e-40, -1e-40, 0.0, 3e-41] * 8, np.float32),
    "tiny_normal": np.full(32, 1e-37, np.float32),  # normal, but 1/d still overflows
    "normal_and_subnormal": np.array([1.0, 1e-40, -1e-40, 0.0] * 8, np.float32),
}
GOLDEN = {
    "q5_0": {
        "neg0_first": "0080ffffffff00000000000000000000000000000000",
        "tie_neg_first": "0032feffffff000f0000000000000000000000000000",
        "neg_max": "0035f7ffffff00000000060000000000000000000000",
        "half_steps": "c03b0000ffff102132435465768798a9bacbdcedfeff",
        "below_half": "f0c7f6ffffff" + "00" * 16,
        "subnormal": "00800000000000000000000000000000000000000000",
        "subnormal_mixed": "0080" + "00" * 20,
        "tiny_normal": "0080" + "00" * 20,
        "normal_and_subnormal": "00aceeeeeeee" + "00" * 16,
    },
    "q8_0": {
        "neg0_first": "00" * 34,
        "tie_neg_first": "0c26817f" + "00" * 30,
        "neg_max": "0a290000008133" + "00" * 27,
        "half_steps": "d02f8189919aa2aab2bac3cbd3dbe3ecf4fc040c141d252d353d464e565e666f777f",
        "below_half": "003c7f00007f" + "00" * 28,
        "subnormal": "00" * 34,
        "subnormal_mixed": "00" * 34,
        "tiny_normal": "00" * 34,
        "normal_and_subnormal": "0820" + "7f000000" * 8,
    },
}


@pytest.mark.parametrize(("kind", "quantize"), [("q5_0", quantize_q5_0), ("q8_0", quantize_q8_0)])
@pytest.mark.parametrize("name", list(BLOCKS))
def test_matches_ggml_reference(kind, quantize, name):
    assert quantize(torch.from_numpy(BLOCKS[name][None])).hex() == GOLDEN[kind][name]


@pytest.mark.parametrize(
    ("kind", "quantize", "size"), [("q5_0", quantize_q5_0, 22), ("q8_0", quantize_q8_0, 34)]
)
def test_matches_local_ggml(ggml_lib, kind, quantize, size):
    """Random blocks and the blocks above against this machine's ``quantize_row_{kind}_ref``, byte for byte."""
    lib = ctypes.CDLL(str(ggml_lib))
    rng = np.random.default_rng(0)
    finite = [b for n, b in BLOCKS.items() if n not in ("subnormal", "subnormal_mixed", "tiny_normal")]
    x = np.concatenate([*finite, rng.standard_normal(32 * 256, np.float32) * 0.02]).astype(np.float32)
    out = ctypes.create_string_buffer(size * (x.size // 32))
    getattr(lib, f"quantize_row_{kind}_ref")(x.ctypes.data_as(ctypes.c_void_p), out, ctypes.c_int64(x.size))
    assert quantize(torch.from_numpy(x[None])) == out.raw


@pytest.mark.parametrize(("kind", "weighted"), [("q8_0", True), ("q8_0", False), ("q5_0", False)])
def test_an_imatrix_build_writes_these_formats_without_libggml(monkeypatch, kind, weighted):
    """With ``imatrix`` set, Q8_0 and unweighted Q5_0 tensors get the bytes above, not this machine's libggml's.

    ``subnormal_mixed`` holds ``-1e-40``, which gives ``+inf`` before the cast for both formats: aarch64's libggml
    writes 0xff there.
    """
    from qda_quant.gguf import rule, writer

    def libggml(*_):
        raise AssertionError("routed to libggml")

    monkeypatch.setattr(writer.kern, "quantize_weighted_chunk", libggml)
    names = ["subnormal_mixed", "tiny_normal", "half_steps"]
    x = torch.from_numpy(np.concatenate([BLOCKS[n] for n in names])[None])
    weights = np.linspace(0.5, 2.0, x.shape[1], dtype=np.float32) if weighted else None
    got = writer._weighted(getattr(rule, kind.upper()), x, 0, weights)
    assert got.hex() == "".join(GOLDEN[kind][n] for n in names)


def test_a_weighted_q5_0_still_takes_llama_cpps_weighted_fit(monkeypatch):
    """llama.cpp's Q5_0 with an importance matrix is its own fit (``nearest_int``, no undefined cast)."""
    from qda_quant.gguf import rule, writer

    calls = []
    monkeypatch.setattr(writer.kern, "quantize_weighted_chunk", lambda *a: calls.append(a) or b"")
    writer._weighted(rule.Q5_0, torch.zeros(1, 32), 22, np.ones(32, np.float32))
    assert len(calls) == 1
