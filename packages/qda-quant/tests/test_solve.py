import pytest
import torch
from qda_quant import Format, Target
from qda_quant.solve.quant import Quantizer, triton_fp8
from qda_quant.solve.solve import MIN_FACTOR_BATCH, _factor, gptq, prepare_hessian


def _problem(e=2, r=32, c=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(e, r, c, generator=g) * 0.02
    mix = torch.randn(e, 48, c, generator=g)  # correlated channels, as real activations are
    x = (
        torch.randn(e, 512, 48, generator=g) @ mix + 0.1 * torch.randn(e, 512, c, generator=g)
    ) * torch.linspace(0.1, 3, c)
    return w, x.transpose(1, 2) @ x


def _q(fmt):
    fmt = Format.parse(fmt)
    return Quantizer.of(fmt, Target().plan(fmt, "moe"), "absmax")


def _loss(w, wq, h):
    d = wq - w
    return (torch.bmm(d, h) * d).sum((1, 2))


def test_factor_pads_a_short_batch_and_returns_only_the_real_matrices():
    """A batch under `MIN_FACTOR_BATCH` is padded with identities for the factorization, then cut back."""
    _, h = _problem(e=2, c=256)
    u = _factor(h.clone(), 0.01, MIN_FACTOR_BATCH)
    assert u.shape == h.shape
    padded = torch.cat([h, torch.eye(256).expand(MIN_FACTOR_BATCH - 2, -1, -1)])
    assert torch.equal(u, _factor(padded, 0.01, MIN_FACTOR_BATCH)[:2])


@pytest.mark.parametrize("fmt", ["w4a16", "w4a16-sym-g128", "w3a16", "w4a8"])
@pytest.mark.parametrize("actorder", ["none", "static"])
def test_gptq_beats_rtn(fmt, actorder):
    q = _q(fmt)
    w, h = _problem()
    s, z = q.params(w)
    codes = gptq(w, h.clone(), q, s, z, actorder=actorder, impl="torch")
    assert codes.min() >= q.fmt.qmin and codes.max() <= q.fmt.qmax
    assert (_loss(w, q.dequant(codes, s, z), h) < 0.8 * _loss(w, q.fake(w, s, z), h)).all()


def test_unrouted_expert_is_rtn():
    q = _q("w4a16")
    w, h = _problem()
    h[1] = 0
    h = prepare_hessian(h, torch.tensor([512, 0]))
    s, z = q.params(w)
    codes = gptq(w, h, q, s, z, impl="torch")
    assert torch.equal(codes[1], q.quantize(w[1], s[1], z[1]))


@pytest.mark.parametrize("sym", [True, False])
def test_matches_llm_compressor(sym):
    lq = pytest.importorskip("llmcompressor.modifiers.gptq.gptq_quantize")
    from compressed_tensors.quantization import QuantizationArgs

    fmt = Format(4, 128, symmetric=sym)
    q = Quantizer.of(fmt, Target().plan(fmt, "dense"), "absmax")
    w, h = _problem(e=1)
    s, z = q.params(w)
    s = s.to(torch.bfloat16)
    ours = q.dequant(gptq(w, h.clone(), q, s, z, damp=0.01, actorder="static", impl="torch"), s, z)
    args = QuantizationArgs(num_bits=4, symmetric=sym, strategy="group", group_size=128, actorder="weight")
    zp = torch.zeros_like(s) if z is None else (z + fmt.qmin - (-8)).to(torch.int8)  # CT stores signed zero
    ref, *_ = lq.quantize_weight(w.clone(), h.clone(), s, zp if sym else z - 8, None, args, 128, 0.01)
    assert (ours - ref.float()).abs().max() < 1e-6


@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["w4a16", "w4a16-sym-g128", "w3a16", "w4a8"])
def test_triton_is_bit_identical(fmt):
    q = _q(fmt)
    w, h = (t.cuda() for t in _problem(e=3, r=200, c=512))
    s, z = q.params(w)
    a = gptq(w, h.clone(), q, s, z, impl="torch")
    if q.fp8_scales and not triton_fp8(w.device):  # an A100: no E4M3 cast in Triton, so auto is the reference
        with pytest.raises(ValueError, match="compute capability 8.9"):
            gptq(w, h.clone(), q, s, z, impl="triton")
        assert torch.equal(a, gptq(w, h.clone(), q, s, z))
        return
    b = gptq(w, h.clone(), q, s, z, impl="triton")
    assert torch.equal(a, b)


def test_a_bank_chunk_stops_at_the_int32_offset_limit():
    """256 experts of 4096 x 2048 is exactly 2^31 elements, and its last live offset is 2^31 - 1, which
    fits; a 257th expert starts at 2^31 and is the first element past the bound.
    """
    from qda_quant.solve.solve import INT32_ELEMENTS, chunk_experts, experts_per_solve

    assert INT32_ELEMENTS == 2**31
    assert INT32_ELEMENTS == 256 * 4096 * 2048
    assert experts_per_solve(4096 * 2048) == 256
    assert chunk_experts(257, 4096, 2048, 1 << 62) == 256  # memory takes all 257; the offsets do not
    assert chunk_experts(256, 4096, 2048, 1 << 62) == 256
    assert chunk_experts(257, 32, 256, 1 << 62) == 257  # a small bank is one solve


def test_every_solve_records_its_geometry_and_largest_offset():
    """`build.json` keeps each solve's E/R/C, strides and largest offset on both paths (W and U)."""
    from qda_quant.solve.solve import solve_offsets

    solve_offsets()  # drop any earlier solve in this process
    q = _q("w4a16")
    w, h = _problem(e=2, r=32, c=256)
    s, z = q.params(w)
    gptq(w, h.clone(), q, s, z, impl="torch", label="layers.0.mlp.down_proj")
    (only,) = solve_offsets()
    assert only["label"] == "layers.0.mlp.down_proj" and only["impl"] == "torch"
    assert (only["experts"], only["rows"], only["cols"]) == (2, 32, 256)
    s0, s1 = only["strides"]
    assert only["max_offset"] == (2 - 1) * s0 + (32 - 1) * s1 + (256 - 1) == 2 * 32 * 256 - 1
    u0, u1, u2 = only["u_strides"]
    assert only["u_max_offset"] == (256 - 1) * u1 + (256 - 1) * u2 == 256 * 256 - 1


@pytest.mark.gpu
def test_triton_offsets_past_the_int32_boundary_are_int64(monkeypatch):
    """257 experts of 65536 x 128 is 2^31 + 2^23 elements, so expert 256 starts exactly at 2^31 and an
    int32 offset makes the kernel read and write other memory. Both impls run on the same 257-batch here (a
    128-wide block means no lazy-update matmul, whose batch kernel would differ from a one-expert solve), so
    every code must match bit for bit — expert 256's past the boundary, expert 0's below it as the control.
    """
    from qda_quant.solve import solve

    q = _q("w4a16")
    w, h = (t.cuda() for t in _problem(e=257, r=65536, c=128))
    s, z = q.params(w)
    launches: list[int] = []
    triton_block = solve._block_triton

    def counting(*a, **k):
        launches.append(1)
        return triton_block(*a, **k)

    monkeypatch.setattr(solve, "_block_triton", counting)
    codes = gptq(w, h.clone(), q, s, z, impl="triton", min_batch=solve.MIN_FACTOR_BATCH)
    rec = [solved for solved in solve.solve_offsets() if solved["experts"] == 257][-1]
    # the pass is worthless if the stack silently solved on the torch path
    assert rec["impl"] == "triton" and len(launches) == 1
    assert rec["max_offset"] == 257 * 65536 * 128 - 1 > 2**31  # the last expert's last element
    print(f"impl={rec['impl']} launches={len(launches)} max_offset={rec['max_offset']}")
    ref = gptq(w, h.clone(), q, s, z, impl="torch", min_batch=solve.MIN_FACTOR_BATCH)
    for i in (0, 256):
        assert torch.equal(codes[i : i + 1], ref[i : i + 1]), f"expert {i}"


@pytest.mark.gpu
def test_triton_factor_rows_past_the_int32_boundary_are_int64():
    """The factor's row index is `row * stride` and reaches C^2; at C = 46464 (the smallest multiple of
    the 128-wide group past the crossing) the product 46463 * 46464 = 2,158,856,832 passes 2^31, so left as
    int32 scalars it wraps before the expert base is added. A 1 x 1 x C stack is a dense group (refusing it
    is the planner's job, not this solver's), and both impls solve the same batch. C^2 fp32 is 8.6 GiB.
    """
    q = _q("w4a16")
    c = 46464
    assert (c - 1) * c == 2_158_856_832 > 2**31
    w, h = (t.cuda() for t in _problem(e=1, r=1, c=c))
    s, z = q.params(w)
    codes = gptq(w, h.clone(), q, s, z, impl="triton")
    ref = gptq(w, h.clone(), q, s, z, impl="torch")
    assert torch.equal(codes, ref)
