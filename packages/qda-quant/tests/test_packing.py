"""qda_quant.solve.packing against compressed-tensors' packer, byte for byte, and qda without compressed-tensors."""

import importlib.metadata
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from qda_quant.solve.packing import pack, unpack

#: Shapes export passes: [out, in] weights (packed_dim 1), [out, groups] zero points (packed_dim 0), fused
#: [experts, rows, cols]; widths on and off multiples of 32, one column, one row.
#: compressed-tensors before 0.20 packs 3, 5, 6 and 7 bits ``32 // bits`` codes per word, no code crossing a word.
#: vLLM 0.33 reads the dense layout (``pack_factor = Fraction(32, bits)``), which is what qda writes.
_CT_DENSE = tuple(int(x) for x in importlib.metadata.version("compressed-tensors").split(".")[:2]) >= (0, 20)

SHAPES = [(64, 256), (48, 100), (17, 33), (8, 1), (1, 40), (5, 3), (3, 16, 70)]


def _codes(shape, bits, seed):
    g = torch.Generator().manual_seed(seed)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = torch.randint(lo, hi + 1, shape, generator=g, dtype=torch.int16)
    flat = q.view(-1)
    flat[0], flat[-1] = lo, hi  # both extremes in every case
    if flat.numel() > 3:
        flat[1], flat[2] = hi, lo
    return q.to(torch.int8)


def _theirs_pack(q, bits, dim):
    if q.dim() > 2:
        return torch.stack([pack_to_int32(s, bits, packed_dim=dim) for s in q])
    return pack_to_int32(q, bits, packed_dim=dim)


def _theirs_unpack(p, bits, shape, dim):
    if p.dim() > 2:
        return torch.stack([unpack_from_int32(s, bits, torch.Size(shape[1:]), packed_dim=dim) for s in p])
    return unpack_from_int32(p, bits, torch.Size(shape), packed_dim=dim)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("bits", range(1, 9))
def test_pack_matches_compressed_tensors(bits, shape, dim, request):
    if not _CT_DENSE and 32 % bits:
        request.applymarker(pytest.mark.xfail(reason="padded layout; short rows coincide"))
    q = _codes(shape, bits, seed=bits * 100 + dim)
    ours, theirs = pack(q, bits, packed_dim=dim), _theirs_pack(q, bits, dim)
    assert ours.dtype == torch.int32 and ours.is_contiguous()
    assert torch.equal(ours, theirs)
    assert ours.numpy().tobytes() == theirs.contiguous().numpy().tobytes()
    assert torch.equal(_theirs_unpack(ours, bits, shape, dim).to(torch.int8), q)
    back = unpack(theirs.contiguous(), bits, shape, packed_dim=dim)
    assert back.dtype == torch.int8 and tuple(back.shape) == shape
    assert torch.equal(back, q)


@pytest.mark.skipif(not _CT_DENSE, reason="compressed-tensors < 0.20 does not split codes across words")
def test_large_matrix_spans_word_boundaries():
    q = _codes((32, 4096 + 7), 3, seed=1)
    assert torch.equal(pack(q, 3), pack_to_int32(q, 3))
    assert torch.equal(unpack(pack(q, 3), 3, q.shape), q)


@pytest.mark.parametrize("bad", [0, 9, 4.0, True])
def test_refuses_bad_bits(bad):
    with pytest.raises(ValueError):
        pack(torch.zeros(2, 8, dtype=torch.int8), bad)
    with pytest.raises(ValueError):
        unpack(torch.zeros(2, 1, dtype=torch.int32), bad, (2, 8))


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.uint8, torch.float32])
def test_pack_refuses_other_dtypes(dtype):
    with pytest.raises(ValueError):
        pack(torch.zeros(2, 8, dtype=dtype), 4)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.uint32])
def test_unpack_refuses_other_dtypes(dtype):
    with pytest.raises(ValueError):
        unpack(torch.zeros(2, 1, dtype=dtype), 4, (2, 8))


def test_refuses_bad_packed_dim():
    with pytest.raises(ValueError):
        pack(torch.zeros(2, 8, dtype=torch.int8), 4, packed_dim=2)


_WITHOUT_CT = textwrap.dedent(
    """
    import sys
    sys.modules["compressed_tensors"] = None  # any import of it raises ImportError
    import torch, transformers
    import qda_quant
    from qda_quant import REGISTRY, Target
    from qda_quant.arch import describe
    from qda_quant.engine.packed import Packed
    from qda_quant.output.export import Writer
    from tests.ctx import contexts
    from tests.tiny import model

    tmp = sys.argv[1]
    for fmt in ("w4a16", "w3a16"):  # asymmetric (zero points) and symmetric
        src, out = f"{tmp}/{fmt}-src", f"{tmp}/{fmt}-out"
        model("qwen3", dtype=torch.bfloat16).save_pretrained(src)
        m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
        w = Writer(out, src, describe(m, verify=False), fmt, Target())
        for ctx in contexts(m, fmt):
            REGISTRY["gptq"](impl="torch").apply_layer(ctx)
            w.add_layer(ctx)
        w.finish(m.config, None, {"ok": True})
        p = Packed(out)
        assert p.stems
        for stem in p.stems:
            q, z = p.codes(stem)
            assert q.dtype == torch.int8 and (z is None) == (fmt == "w3a16")
    assert "compressed_tensors" not in {k for k, v in sys.modules.items() if v is not None}
    print("ok")
    """
)


def test_build_and_read_back_without_compressed_tensors(tmp_path):
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-c", _WITHOUT_CT, str(tmp_path)], cwd=root, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().endswith("ok")
