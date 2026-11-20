"""Byte parity of the GGUF Q4_K and Q6_K kernels.

The golden bytes below were written by llama.cpp 9e2a04a's ``quantize_row_q4_K_ref`` and
``quantize_row_q6_K_ref`` (``GGML_NATIVE=OFF``) on the tensors built by ``_cases``. They pin the hard cases: a
constant row (degenerate min), a negative constant, all zeros, sub-normal magnitude, magnitude whose super-block
scale overflows fp16, and a wide dynamic range. No llama.cpp is needed to run the tests.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from qda_quant.gguf import kernel
from qda_quant.gguf.export import q4k_bytes, q6k_bytes
from qda_quant.gguf.kernel import (
    _as_blocks,
    _quantize_q4k_fast,
    _quantize_q6k_fast,
    q4k_fields,
    q6k_fields,
    q6k_records,
    quantize_q4k,
    quantize_q4k_weighted,
    quantize_q6k,
    use_ggml,
)
from qda_quant.gguf.packing import _split, _split_q6k
from qda_quant.gguf.q4k import Q4K_FORMAT, Q4K_GRID
from qda_quant.gguf.q6k import Q6K_FORMAT, Q6K_GRID
from qda_quant.step import Refused

GOLDEN = {
    "constant": {
        "q4k": "0b1b0000ffffffff000000000f0f0f0fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "q6k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000808080808080808080808080808080808012",
    },
    "negative_constant": {
        "q4k": "00009e2f00000000fffffffff0f0f0f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "q6k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000808080808080808080808080808080808097",
    },
    "zeros": {
        "q4k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "q6k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    },
    "tiny": {
        "q4k": "00000000ffffffffffffffffffffffffc8febe490230a5fcdf6b131083ebef8c250062d9feae370141b7fdcf5a032094feae370141b7fdcf59022095ecef7b141073daff9d260151c8febe480230a6fd9d260151c8febe480230a6fddf6a031084ebef8c150062d9ffae370141b7fdcf150062d9ffad370141b7fecf59022095fcdf7b141073eaef9d260052c8febe48",
        "q6k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    },
    "large": {
        "q4k": "007c007cffffffffffffffffffffffff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "q6k": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    },
    "dynamic_range": {
        "q4k": "89621170c034fff4000fffbff400f40f3c3c3c2c1c0c9cfc3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3d3d3e393630cacacacacacacacacacacac9cbcccfc0cacacadaea9a6a0acacacacacacacaca23232323232323232323232323232323232323232323232323232212015089ef5c5c5c5d5e5f56505c5c5c6c4c3c0cfc5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c",
        "q6k": "000000000000000000f0f1e24488000000000000000000000000f1e24488000000000f02040800000000f0204080000000f010204080000000000f0e0408000000f0f0204080000000000f0204080000000000000000000000001122c488000000000000000000000000f1e24488000000000000000000000000f1e244880000aaaaa6aaaaaaa6a2aa9a5a9aaaa97908aa6aaaaaaa6a6a2aaaaa9696aaa9bd80aa9a9aaaaaaa9a8aaaaa696aaaa6e520aaaaaaaaaaaaaaaaaaaa6a6a9a96d500000080007f00007f008000007f0080000058",
    },
}


def _cases() -> dict[str, np.ndarray]:
    idx = np.arange(256, dtype=np.float32)
    dynamic = np.sign(np.sin(idx)) * np.float32(2.0) ** ((np.arange(256) % 40) - 20)
    return {
        "constant": np.full(256, 3.25, np.float32),
        "negative_constant": np.full(256, -7.5, np.float32),
        "zeros": np.zeros(256, np.float32),
        "tiny": np.sin(idx) * np.float32(1e-30),
        "large": (np.cos(idx) * np.float32(1e19)).astype(np.float32),
        "dynamic_range": dynamic.astype(np.float32),
    }


CASES = _cases()


def compile_quantizer(tmp_path: Path) -> Path:
    """Build the shipped `_quant.c` the way `hatch_build.py` does; skip where no C compiler can."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None or sys.platform != "linux":
        pytest.skip("no C compiler for the shipped quantizer")
    source = Path(kernel.__file__).with_name("_quant.c")
    out = tmp_path / "_quant.so"
    subprocess.run(
        [
            cc,
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-fno-fast-math",
            '-DQDA_VERSION="test"',
            "-o",
            str(out),
            str(source),
        ],  # fmt: skip
        check=True,
    )
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("kind", ["q4k", "q6k"])
def test_golden_bytes(name: str, kind: str) -> None:
    quantize = quantize_q4k if kind == "q4k" else quantize_q6k
    assert quantize(torch.from_numpy(CASES[name])) == bytes.fromhex(GOLDEN[name][kind])


def test_shapes_and_refusals() -> None:
    x = np.stack([CASES["constant"], CASES["dynamic_range"]])
    rows = quantize_q4k(torch.from_numpy(x))
    assert rows == quantize_q4k(torch.from_numpy(x.reshape(-1)))  # rows are independent super-blocks
    assert len(rows) == 2 * 144
    assert len(quantize_q6k(torch.from_numpy(x))) == 2 * 210
    assert quantize_q4k(torch.zeros(0)) == b""

    with pytest.raises(ValueError, match="multiple of 256"):
        quantize_q4k(torch.zeros(255))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        quantize_q4k(torch.zeros(1, 1, 256))
    with pytest.raises(TypeError, match="torch.Tensor"):
        quantize_q6k(np.zeros(256, np.float32))  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["q4k", "q6k"])
def test_worker_count_does_not_change_bytes(kind: str) -> None:
    quantize = quantize_q4k if kind == "q4k" else quantize_q6k
    rng = np.random.default_rng(7)
    # 4096 super-blocks lets four workers run; larger counts are clamped by _MIN_SHARD_BLOCKS.
    x = torch.from_numpy((rng.standard_normal(4096 * 256) * 0.05).astype(np.float32))
    want = quantize(x, workers=1)
    for workers in (2, 3, 4, 32):
        assert quantize(x, workers=workers) == want

    with pytest.raises(ValueError, match="at least 1"):
        quantize(x, workers=0)
    with pytest.raises(TypeError, match="int or None"):
        quantize(x, workers=2.5)  # type: ignore[arg-type]


def test_ggml_library_writes_the_same_bytes(ggml_lib: Path) -> None:
    """``libggml-base``'s ``quantize_row_q{4,6}_K_ref`` and the port agree on every case, at any worker count."""
    rng = np.random.default_rng(11)
    wide = (rng.standard_normal((64, 3 * 256)) * 0.07).astype(np.float32)
    tensors = [*CASES.values(), wide.reshape(-1)]
    try:
        assert use_ggml(ggml_lib).startswith("libggml")
        assert kernel.ggml_version() and kernel.ggml_commit()
        for quantize, slow in ((quantize_q4k, _quantize_q4k_fast), (quantize_q6k, _quantize_q6k_fast)):
            for t in tensors:
                with np.errstate(all="ignore"):  # the large case overflows fp32 on purpose, as in C
                    want = slow(_as_blocks(torch.from_numpy(t)))
                assert quantize(torch.from_numpy(t)) == want
                assert quantize(torch.from_numpy(t), workers=4) == want
        assert kernel.ggml_backend().startswith("libggml")  # type: ignore[union-attr]
    finally:
        assert use_ggml(None) == "numpy"
    assert kernel.ggml_backend() is None


def test_use_ggml_refuses_a_file_that_is_not_a_library(tmp_path: Path) -> None:
    use_ggml(None)
    not_a_library = tmp_path / "libggml-base.so"
    not_a_library.write_bytes(b"not a shared library")
    with pytest.raises(ValueError, match="not a llama.cpp quantizer library"):
        use_ggml(not_a_library)
    assert kernel.ggml_backend() is None  # a failed load leaves the port in place


@pytest.mark.parametrize("case", GOLDEN)
def test_q6k_fields_repack_the_pinned_reference_bytes(case: str) -> None:
    """The constructor must reproduce C's records, including zero and non-finite fit cases."""
    raw = bytes.fromhex(GOLDEN[case]["q6k"])
    assert q6k_records(*q6k_fields(raw)) == raw


def test_q6k_solved_codes_keep_fitted_scales_and_template_row_order() -> None:
    """Pack grid qparams without re-rounding codes, including ggml's zero-scale fields."""
    w = torch.randn(5, 512, generator=torch.Generator().manual_seed(73)) * 0.07
    w[0, :256] = 0
    w[1, 16:32] = 0
    w[2, :256] *= 1e-7  # d underflows fp16, while ggml's int8 scales remain nonzero
    rows = torch.tensor([3, 0, 4, 2, 1])
    scale, zero, codes = Q6K_GRID.fit_codes(None, w)  # type: ignore[arg-type]
    qp = {"format": Q6K_FORMAT, "codes": codes, "scale": scale, "zero_point": zero}
    want = quantize_q6k(w)
    records = np.frombuffer(want, dtype=np.uint8).reshape(5, -1)[rows.numpy()].tobytes()
    assert q6k_bytes(qp, rows, w[rows], "blk.0.ffn_down.weight", workers=1) == records
    # Move a live solved code. The writer must preserve it, rather than quantize the template again.
    codes[3, 7] = 0 if codes[3, 7] != 0 else 1
    got = q6k_bytes(qp, rows, w[rows], "blk.0.ffn_down.weight", workers=1)
    assert got != records
    d, sc, stored = q6k_fields(got)
    expected = (codes[rows].to(torch.int16) + 32).numpy().reshape(-1, 256)
    live = (d.view("<f2").astype(np.float32)[:, None] * sc != 0).repeat(16, axis=1)
    assert np.array_equal(stored[live], expected[live])
    moved = scale.clone()
    moved[3, 0] *= 1.001
    with pytest.raises(Refused, match="blk.0.ffn_down.weight: scales are not Q6_K"):
        q6k_bytes({**qp, "scale": moved}, rows, w[rows], "blk.0.ffn_down.weight")


def test_q6k_factorisation_refuses_nonfinite_or_out_of_range_products() -> None:
    """Exact signed factors matter: +128, NaN and infinity cannot become stored scale bytes."""
    valid = torch.tensor([[-128.0, 127, 0, -1] * 4]) * torch.tensor(0.003, dtype=torch.float16).float()
    d, s = _split_q6k(valid, "t")
    assert torch.equal(
        torch.from_numpy(d.view("<f2").astype(np.float32))[:, None] * torch.from_numpy(s), valid
    )
    for bad in (float("nan"), float("inf"), float("-inf"), 128.0):
        x = torch.tensor([[-128.0, bad] + [0] * 14])
        with pytest.raises(Refused, match="t: scales are not Q6_K"):
            _split_q6k(x, "t")


def _decoded(records: bytes) -> torch.Tensor:
    """The values Q4_K records hold, as ``dequantize_row_q4_K`` computes them."""
    d, dmin, sc, m, codes = q4k_fields(records)
    f16 = lambda bits: torch.from_numpy(bits.view("<f2").astype("<f4"))[:, None]  # noqa: E731
    k = torch.from_numpy(sc.astype("<f4")).repeat_interleave(32, -1)
    mins = torch.from_numpy(m.astype("<f4")).repeat_interleave(32, -1)
    return f16(d) * k * torch.from_numpy(codes.astype("<f4")) - f16(dmin) * mins


def test_the_weighted_fit_holds_what_llama_quantize_imatrix_holds(ggml_lib: Path) -> None:
    """Scales recovered from the weighted fit, coded as a build codes them, decode to ``quantize_q4_K``'s values.

    Its ``make_qp_quants`` leaves some blocks' largest group below 63, which the plain fit never does.
    """
    rng = np.random.default_rng(12)
    w = torch.from_numpy((rng.standard_normal((512, 3 * 256)) * 0.07).astype(np.float32))
    w[0, 32:64] = 0  # an all-zero group
    importance = torch.from_numpy(rng.lognormal(size=3 * 256, sigma=2).astype(np.float32))
    importance[:32] = 0  # a group that weighs nothing
    try:
        use_ggml(ggml_lib)
        want = quantize_q4k_weighted(w, importance)
        assert (q4k_fields(want)[2].max(-1) < 63).any()
        s, z = Q4K_GRID.fit_weighted(w[None], importance[None])
        codes = Q4K_GRID.codes(w, s[0].repeat_interleave(32, -1), z[0].repeat_interleave(32, -1), 0, 15)
        qp = {"format": Q4K_FORMAT, "codes": codes.to(torch.int8), "scale": s[0], "zero_point": z[0]}
        got = q4k_bytes(qp, torch.arange(w.shape[0]), w, "t")
    finally:
        use_ggml(None)
    assert torch.equal(_decoded(got), _decoded(want))


def test_split_takes_a_block_whose_largest_group_is_below_63() -> None:
    """``D`` and ``k`` come back from their products alone, and a value that is no such product is refused."""
    d = torch.tensor([0.25, 3e-4, 0.0078125], dtype=torch.float16)
    k = torch.tensor(
        [[63.0, 1, 2, 3, 4, 5, 6, 7], [40, 39, 0, 1, 2, 3, 4, 5], [17, 17, 17, 17, 17, 17, 17, 17]]
    )
    x = d.float()[:, None] * k
    got_d, got_k = _split(x, "scales", "t")
    assert torch.equal(
        torch.from_numpy(got_d.view("<f2").astype("<f4"))[:, None] * torch.from_numpy(got_k).float(), x
    )
    assert got_d[0] == d[:1].view(torch.int16).numpy().view("<u2")[0]  # k = 63 is ggml's own split
    with pytest.raises(ValueError, match="not Q4_K's"):
        _split(x * 1.0001, "scales", "t")
    with pytest.raises(ValueError, match="not Q4_K's"):
        _split(-x, "scales", "t")


def test_a_named_directory_selects_the_library_in_it(ggml_lib: Path, tmp_path: Path) -> None:
    """``libggml`` as the library's own directory works as the file does, and the notice names the kernel."""
    from qda_quant.gguf.export import GGUF

    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text("")
    ident = ["-c", "user.email=t@e", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(checkout), *ident, "commit", "-qm", "x", "--allow-empty"], check=True)
    try:
        note = GGUF(llama_cpp=str(checkout), libggml=str(ggml_lib.parent)).prepare()
        assert note.startswith("gguf: quantizing with llama.cpp's own kernel")
        assert kernel.ggml_backend() is not None
    finally:
        use_ggml(None)


def test_prepare_without_a_library_names_the_build_command(tmp_path: Path) -> None:
    """The fallback is the Python port and the notice says so, with the one command that builds the library."""
    from qda_quant.gguf.export import GGUF, libggml_base

    use_ggml(None)
    assert libggml_base(tmp_path) is None
    note = GGUF(llama_cpp=str(tmp_path)).prepare()
    assert note is not None
    assert "no libggml-base found" in note and "Python port" in note
    assert "cmake -S" in note and "--target ggml-base" in note
    assert kernel.ggml_backend() is None


def test_library_discovery_reads_the_build_directories(tmp_path: Path) -> None:
    """llama.cpp's CMake leaves ``libggml-base.so`` in ``build/bin``; the newest build tree's is found."""
    from qda_quant.gguf.export import libggml_base

    lib = tmp_path / "build-portable" / "bin" / "libggml-base.so.0.29.0"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    assert libggml_base(tmp_path) == lib


def test_library_discovery_reads_the_directory_itself(tmp_path: Path) -> None:
    """A directory that holds the library is read as it is, not only as a checkout with a ``build*`` tree."""
    from qda_quant.gguf.export import libggml_base

    lib = tmp_path / "bin" / "libggml-base.so.0.29.0"
    lib.parent.mkdir()
    lib.write_bytes(b"")
    assert libggml_base(lib.parent) == lib
    assert libggml_base(tmp_path) is None  # only the named directory and its own build trees


def test_a_named_directory_without_a_library_is_refused(tmp_path: Path) -> None:
    from qda_quant.gguf.export import GGUF
    from qda_quant.step import Refused

    (tmp_path / "bin").mkdir()
    with pytest.raises(Refused, match="no libggml-base"):
        GGUF(llama_cpp=str(tmp_path), libggml=str(tmp_path / "bin")).prepare()


def test_a_named_library_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    from qda_quant.gguf.export import GGUF
    from qda_quant.step import Refused

    with pytest.raises(Refused, match="not a file"):
        GGUF(llama_cpp=str(tmp_path), libggml=str(tmp_path / "libggml-base.so")).prepare()


def test_the_shipped_quantizer_writes_the_reference_bytes(tmp_path: Path) -> None:
    """`_quant.c` (llama.cpp's own functions) writes the golden bytes and the port's, at any worker count."""
    lib = compile_quantizer(tmp_path)
    rng = np.random.default_rng(13)
    wide = (rng.standard_normal((64, 3 * 256)) * 0.07).astype(np.float32)
    try:
        assert use_ggml(lib).startswith("qda-quant-gguf")
        assert kernel.ggml_version() == "test" and kernel.ggml_commit() == "9e2a04a"
        for name, tensor in CASES.items():
            for quantize, kind in ((quantize_q4k, "q4k"), (quantize_q6k, "q6k")):
                want = bytes.fromhex(GOLDEN[name][kind])
                assert quantize(torch.from_numpy(tensor)) == want
                assert quantize(torch.from_numpy(tensor), workers=4) == want
        for quantize, fast in ((quantize_q4k, _quantize_q4k_fast), (quantize_q6k, _quantize_q6k_fast)):
            with np.errstate(all="ignore"):
                want = fast(_as_blocks(torch.from_numpy(wide.reshape(-1))))
            assert quantize(torch.from_numpy(wide.reshape(-1))) == want
            assert quantize(torch.from_numpy(wide.reshape(-1)), workers=3) == want
    finally:
        use_ggml(None)


def test_use_bundled_falls_back_to_numpy(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """No library (a source install without one) or a broken one is a slower build, never a failed one."""
    monkeypatch.setattr(kernel, "bundled_path", lambda: None)
    assert kernel.use_bundled() == "numpy"
    broken = tmp_path / "_quant.so"
    broken.write_bytes(b"not a shared library")
    monkeypatch.setattr(kernel, "bundled_path", lambda: broken)
    assert kernel.use_bundled() == "numpy"
    assert kernel.ggml_backend() is None


def test_prepare_uses_the_shipped_kernel_before_the_python_port(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from qda_quant.gguf.export import GGUF

    lib = compile_quantizer(tmp_path)
    monkeypatch.setattr(kernel, "bundled_path", lambda: lib)
    try:
        note = GGUF(llama_cpp=str(tmp_path)).prepare()
        assert note == "gguf: quantizing with the shipped kernel, qda-quant-gguf test (9e2a04a)"
        assert kernel.ggml_backend() is not None
    finally:
        use_ggml(None)


def test_q6k_fields_decodes_what_the_quantizer_wrote(ggml_lib: Path) -> None:
    """The reader inverts llama.cpp's Q6_K packing: decoding its own records reproduces the values.

    The codes are split across `ql` (low nibble) and `qh` (two bits), in an order that is easy to get subtly
    wrong -- a wrong quarter still yields codes in range and plausible-looking bytes. Reconstructing the
    values from the decoded fields catches that, where checking shapes and ranges does not.
    """
    kernel.use_ggml(str(ggml_lib))
    try:
        rng = np.random.default_rng(0)
        x = torch.from_numpy((rng.standard_normal((8, 1024)) * 0.05).astype(np.float32))
        d, scales, codes = kernel.q6k_fields(kernel.quantize_q6k(x, workers=1))

        assert d.shape == (32,) and scales.shape == (32, 16) and codes.shape == (32, 256)
        assert codes.min() >= 0 and codes.max() <= 63, "codes are ggml's stored 0..63, not signed"
        assert scales.min() < 0, "group scales are signed"

        fp16 = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)
        value = (
            fp16[:, None, None]
            * scales.astype(np.float32)[:, :, None]
            * (codes.reshape(-1, 16, 16).astype(np.float32) - 32)
        )
        assert np.abs(value.reshape(x.shape) - x.numpy()).max() < 0.01
    finally:
        kernel.use_ggml(None)
