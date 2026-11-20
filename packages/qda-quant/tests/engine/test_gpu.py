"""GPU checks of the engine: bf16 fast paths and card-count parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from qda_quant.arch import describe
from qda_quant.engine.runner import LayerStats
from qda_quant.engine.stats import accumulate, input_chunks, observe_experts, reference_expert_stats
from qda_quant.step import Step
from qda_quant.steps.gptq import GPTQ

from ..engine_util import (
    StatDump,
    TensorWriter,
    save_tiny_tokenizer,
    tiny_qwen3,
    tiny_qwen3_moe,
    write_ids,
    writes,
)

pytestmark = pytest.mark.gpu
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")


@cuda
def test_accumulate_matches_fp64():
    x = torch.randn(3, 512, 256, device="cuda", dtype=torch.bfloat16)
    H = None
    for t in x:
        H = accumulate(H, t)
    ref = sum(t.double().T @ t.double() for t in x)
    assert ((H.double() - ref).norm() / ref.norm()) < 1e-5
    Hb = accumulate(None, x)
    Hb = accumulate(Hb, x)
    refb = 2 * torch.einsum("erc,erd->ecd", x.double(), x.double())
    assert ((Hb.double() - refb).norm() / refb.norm()) < 1e-5


@cuda
@pytest.mark.parametrize("hd", ["bf16", "tf32", "fp32"])
def test_hessian_dtypes_vs_fp64(hd):
    from qda_quant.engine.stats import GroupStats

    torch.manual_seed(0)
    x = torch.randn(6, 2048, 512, device="cuda", dtype=torch.bfloat16)
    st = GroupStats(512, x.device, frozenset({"hessian"}), dtype=hd)
    for t in x:
        st.add(t)
    H = st.result()["hessian"].double()
    ref = sum(t.double().T @ t.double() for t in x)
    err = float((H - ref).norm() / ref.norm())
    print(f"hessian_dtype={hd}: rel Frobenius vs fp64 {err:.2e}")
    # this test (6 x 2048 rows, C=512) measured on H200: bf16 1.02e-5, tf32 5.7e-7, fp32 2.6e-7 (two runs, same
    # values); on A100 tf32 3.4e-6 (cuBLAS picks another TF32 reduction there). Bars ~2x the measured value;
    # a wider probe of the same reduction puts bf16 at 0.8-2.3e-5 and tf32 at 0.5-3.0e-5 over C=512..12288,
    # so the bf16 tensor-core Hessian is about as accurate as tf32. On GB300 (sm103) that probe and three seeds
    # read bf16 3.9-5.1e-6 and tf32 5.5-6.9e-6: Blackwell's tf32 tensor cores round like bf16, so its tf32 bar
    # is the bf16 one. fp32 stays exact.
    cap = torch.cuda.get_device_capability()
    tf32 = 7e-6 if cap < (9, 0) else 2e-5 if cap >= (10, 0) else 2e-6
    assert err < {"bf16": 2e-5, "tf32": tf32, "fp32": 1e-6}[hd]


@cuda
def test_weighted_accumulate_no_bf16_round_trip():
    x = torch.randn(4, 300, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4, 300, device="cuda")
    from qda_quant.engine.stats import symmetrize_

    H = symmetrize_(accumulate(accumulate(None, x[:, :150], w[:, :150]), x[:, 150:], w[:, 150:]))
    ref = torch.einsum("erc,er,erd->ecd", x.double(), w.double(), x.double())
    err = float((H.double() - ref).norm() / ref.norm())
    print(f"weighted accumulate rel Frobenius {err:.2e}")
    assert err < 1e-5


@cuda
@pytest.mark.parametrize("rw", [False, True])
def test_fused_experts_bf16_cuda(rw):
    model = tiny_qwen3_moe(layers=1).to("cuda", torch.bfloat16)
    moe = model.model.layers[0].mlp
    x = torch.randn(300, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    _, w, idx = moe.gate(x)
    lg = describe(model, verify=False).layers[0]
    st = LayerStats(model.model.layers[0], lg, frozenset({"expert_hessian"}), x.device, router_weighted=rw)
    bank = st.banks[lg.banks[0].router]
    observe_experts(moe.experts, x, idx, w, bank)
    ref = reference_expert_stats(moe.experts, x, idx, w, rw)
    r = bank.result()
    tol = {"in": 1e-5, "down": 1e-5}  # measured 1e-7 .. 3e-7 on an H200
    for k in ("in", "down"):
        err = float((r["hessian"][k].double() - ref[k].double()).norm() / ref[k].double().norm())
        print(f"fused vs reference rw={rw} {k}: rel Frobenius {err:.2e}")
        assert err < tol[k], (k, err)


def _save_tiny(tmp, make):
    m = make()
    m.save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(tmp / "c.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(8)], sha)
    return tmp / "m"


def _build(tmp, src, steps, n_cards, tag, residency=None):
    """Build into ``tmp/tag``; `steps` gets its own directory under ``tmp`` for the files it dumps.

    The output must stay empty until the build publishes (it is renamed onto), so a step that writes files
    must not write there. Returns the output directory and the dump directory.
    """
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine import build
    from qda_quant.recipe import Recipe

    out, dump = tmp / tag, tmp / f"{tag}-stats"
    out.mkdir()
    dump.mkdir()
    r = Recipe(steps=steps(dump), calib=FileSource(path=str(tmp / "c.jsonl"), samples=8, seqlen=32))
    build(
        str(src),
        r,
        format="w4a16-sym-g128",
        out_dir=out,
        devices=list(range(n_cards)),
        exporter=writes(TensorWriter),
        residency=residency,
        log=lambda s: None,
    )
    if residency is not None:  # the test must exercise what it says
        assert json.loads((out / "build.json").read_text())["residency"]["weights"] == residency
    return out, dump


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
@pytest.mark.parametrize("rows", [64, None])
@pytest.mark.parametrize("make", [tiny_qwen3, tiny_qwen3_moe])
def test_stats_independent_of_card_count(tmp_path, make, rows):
    """Every statistic (Hessians, expert Hessians, absmax, input sample, token counts) on 2 cards equals
    the 1-card build up to fp32 summation order; the input sample picks the same rows."""
    src = _save_tiny(tmp_path, make)
    _, dump_one = _build(tmp_path, src, lambda d: [StatDump(dump=str(d), input_rows=rows)], 1, "c1")
    _, dump_two = _build(tmp_path, src, lambda d: [StatDump(dump=str(d), input_rows=rows)], 2, "c2")
    files = sorted(dump_one.glob("stats-*.pt"))
    assert files, "the 1-card build dumped no statistics, so there is nothing to compare"
    assert {f.name for f in files} == {f.name for f in dump_two.glob("stats-*.pt")}
    for f in files:
        a, b = torch.load(f), torch.load(dump_two / f.name)
        assert a.keys() == b.keys()
        assert a, f"{f.name}: no statistics"
        for k in a:
            va, vb = a[k], b[k]
            if isinstance(va, dict):
                assert va.keys() == vb.keys(), k
                pairs = [(f"{k}[{j}]", va[j], vb[j]) for j in va]
            else:
                pairs = [(k, va, vb)]
            for name, x, y in pairs:
                if name.startswith("inputs"):  # a list of chunks or one tensor (`stats.input_chunks`)
                    x, y = torch.cat(input_chunks(x)).cpu(), torch.cat(input_chunks(y)).cpu()
                if isinstance(x, torch.Tensor):
                    exact = name.startswith(("act_absmax", "inputs", "n_tokens"))
                    err = float(
                        (x.double() - y.double()).abs().max() / x.double().abs().max().clamp_min(1e-30)
                    )
                    print(f"1 vs 2 cards {name}: max rel {err:.2e}")
                    assert err <= (0 if exact else 1e-5), (name, err)
                else:
                    assert x == y, name


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
def test_gptq_weights_independent_of_card_count(tmp_path):
    """2 cards quantize as well as 1. Their statistics differ only in fp32 summation order (see
    test_stats_independent_of_card_count), so layer 0, which sees the same inputs on both, agrees to a code
    step. GPTQ amplifies that rounding through later layers (31% of codes differ by layer 2 on this model, at most
    2.006 steps apart), so deeper layers get a per-entry bound of 2.5 code steps, which still catches one wrong
    code (a sign flip is 5.5-11.5 steps here), plus a quality bound: the error against the float weight within 1%."""
    from qda_quant import REGISTRY
    from safetensors.torch import load_file

    src = _save_tiny(tmp_path, lambda: tiny_qwen3(hidden=128, inter=256))
    one, _ = _build(tmp_path, src, lambda _: [REGISTRY["gptq"]()], 1, "g1")
    two, _ = _build(tmp_path, src, lambda _: [REGISTRY["gptq"]()], 2, "g2")
    ref = load_file(str(src / "model.safetensors"))
    for i, f in enumerate(sorted(one.glob("layer-*.pt"))):
        a, b = torch.load(f), torch.load(two / f.name)
        for k in a:
            x, y = a[k].float(), b[k].float()
            step = x.abs().max().item() / 7
            assert (x - y).abs().max().item() <= (1 if i == 0 else 2.5) * step, (k, "codes too far apart")
            if i == 0:
                assert ((x - y).abs() > 0).float().mean().item() <= 0.01, (
                    k,
                    "layer 0: over 1% of codes differ",
                )
            w = ref[f"model.layers.{i}.{k}"].float()
            e1, e2 = (x - w).norm().item(), (y - w).norm().item()
            print(f"{f.name} {k}: error vs float, 1 card {e1:.5f}, 2 cards {e2:.5f}")
            assert e2 <= 1.01 * e1 + 1e-6, (k, e1, e2)
    info = json.loads((two / "build.json").read_text())
    assert info["card_count"] == 2 and len(info["rank_timings"]) == 2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
@pytest.mark.parametrize("residency", ["resident", "stream"])
def test_model_step_then_gptq_two_cards_match_one(tmp_path, residency):
    """rotate (a model step) + gptq on 2 ranks equals 1 rank up to fp32 summation order of the Hessians:
    the model step runs once before the ranks start, so every rank quantizes the same rotated model."""
    from qda_quant import REGISTRY

    src = _save_tiny(tmp_path, lambda: tiny_qwen3(hidden=128, inter=256))
    steps = lambda o: [REGISTRY["rotate"](), REGISTRY["gptq"]()]  # noqa: E731
    one, _ = _build(tmp_path, src, steps, 1, "r1", residency)
    two, _ = _build(tmp_path, src, steps, 2, "r2", residency)
    same = total = 0
    for f in sorted(one.glob("layer-*.pt")):
        a, b = torch.load(f), torch.load(two / f.name)
        for k in a:
            total += 1
            same += torch.equal(a[k], b[k])
            d = (a[k].float() - b[k].float()).abs().max().item()
            assert d <= 2 * a[k].float().abs().max().item() / 7, (k, d)  # at most a code flip
    print(f"rotate+gptq {residency}: {same}/{total} layer tensors bit-identical 1 vs 2 cards")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
def test_head_format_two_cards_bit_identical_to_one(tmp_path):
    """head (own format, embedding on its grid) + rtn with the real Writer: every stored tensor and the
    config on 2 cards equal 1 card bit for bit. The head's codes reach rank 0's Writer through the qparams
    dict the fan-out ships next to the model steps' weights."""
    from qda_quant import REGISTRY
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine import build
    from qda_quant.recipe import Recipe
    from safetensors.torch import load_file

    src = _save_tiny(tmp_path, lambda: tiny_qwen3(hidden=128, inter=256))
    r = Recipe(
        steps=[REGISTRY["head"](format="w4a16-sym-g64", embed=True), REGISTRY["rtn"]()],
        calib=FileSource(path=str(tmp_path / "c.jsonl"), samples=8, seqlen=32),
    )
    outs = []
    for n in (1, 2):
        out = build(str(src), r, format="w4a16", out_dir=tmp_path / f"h{n}", devices=list(range(n)),
                    log=lambda s: None)  # fmt: skip
        outs.append((out, {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}))
    (o1, a), (o2, b) = outs
    assert a.keys() == b.keys() and "lm_head.weight_packed" in a
    for k in a:
        assert torch.equal(a[k], b[k]), k
    qc1, qc2 = (json.loads((o / "config.json").read_text())["quantization_config"] for o in (o1, o2))
    assert qc1 == qc2 and "group_head" in qc1["config_groups"]
    assert json.loads((o2 / "build.json").read_text())["card_count"] == 2


class WholeGPTQ(GPTQ):
    """GPTQ with the routed-expert split off: rank 0 solves every bank (the reference for the split's bytes).
    Module level: spawn pickles the recipe."""

    name = "gptq_whole_test"
    splits_experts: ClassVar[bool] = False


def _tensor_sha(out) -> tuple[dict, str]:
    """Every stored tensor of a checkpoint, and one sha256 over their names, dtypes, shapes and bytes."""
    import hashlib

    from safetensors.torch import load_file

    t = {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}
    h = hashlib.sha256()
    for k in sorted(t):
        v = t[k].contiguous()
        h.update(f"{k}|{v.dtype}|{tuple(v.shape)}|".encode())
        h.update(v.view(torch.uint8).numpy().tobytes() if v.numel() else b"")
    return t, h.hexdigest()


@pytest.mark.parametrize("cards", [2, 4])
def test_expert_split_bytes_equal_rank0_solve(tmp_path, cards):
    """The routed-expert solve split across cards stores the same checkpoint, byte for byte, as rank 0 solving
    every bank on the same cards (same statistics, so only the split differs). Real Writer, tiny Qwen3-MoE with
    8 experts per bank, so every card gets a share at 2 and 4 cards."""
    if torch.cuda.device_count() < cards:
        pytest.skip(f"needs {cards} cards")
    from qda_quant import REGISTRY
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine import build
    from qda_quant.recipe import Recipe

    src = _save_tiny(tmp_path, tiny_qwen3_moe)
    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=8, seqlen=32)
    got = {}
    for tag, step in (("split", REGISTRY["gptq"]()), ("whole", WholeGPTQ())):
        out = build(str(src), Recipe(steps=[step], calib=calib), format="w4a16-sym-g32",
                    out_dir=tmp_path / tag, devices=list(range(cards)), log=lambda s: None)  # fmt: skip
        phases = json.loads((out / "build.json").read_text())["timings"]["phases"]
        assert ("gather" in phases) is (tag == "split"), (
            tag,
            sorted(phases),
        )  # the split ran only where meant
        got[tag] = _tensor_sha(out)
    (a, ha), (b, hb) = got["split"], got["whole"]
    assert a.keys() == b.keys() and any("experts" in k for k in a)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    print(f"{cards} cards: split {ha[:12]} = rank-0 solve {hb[:12]} over {len(a)} tensors")
    assert ha == hb


class Boom(Step):
    """Fails on rank 0 (where layer steps run) at layer 1. Module level: spawn pickles the recipe."""

    name = "boom_test"

    def apply_layer(self, ctx):
        if ctx.graph.index == 1:
            raise RuntimeError("boom")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
def test_dead_rank_fails_the_build(tmp_path):
    import time

    src = _save_tiny(tmp_path, tiny_qwen3)
    t = time.perf_counter()
    with pytest.raises(Exception, match="boom"):
        _build(tmp_path, src, lambda o: [Boom()], 2, "boom")
    took = time.perf_counter() - t
    print(f"dead rank: build failed after {took:.1f} s")
    assert took < 120  # the other rank is stopped, the build does not hang


def _from_root_rank(rank: int, port: int, fail: bool, out: str) -> None:
    import os

    import torch.distributed as dist
    from qda_quant.engine.dist import Comm

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=dev)

    def fn():
        if fail:
            raise ValueError("verify failed on rank 0")
        return {"graph": "from rank 0"}

    try:
        got = Comm(rank, 2, dev).from_root(fn)
        Path(out, f"rank{rank}").write_text(repr(got))
    except RuntimeError as e:
        Path(out, f"rank{rank}").write_text(f"raised: {e}")
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 cards")
@pytest.mark.parametrize("fail", [False, True])
def test_from_root_shares_result_or_fails_everywhere(tmp_path, fail):
    """Rank 0's describe(verify=True) result reaches every rank; if it raises, every rank raises (no hang)."""
    import time

    import torch.multiprocessing as mp
    from qda_quant.engine.dist import free_port

    t = time.perf_counter()
    mp.spawn(_from_root_rank, args=(free_port(), fail, str(tmp_path)), nprocs=2, join=True)
    got = [(tmp_path / f"rank{r}").read_text() for r in (0, 1)]
    print(f"from_root fail={fail}: {got} in {time.perf_counter() - t:.1f} s")
    if fail:
        assert all("raised: rank 0 failed: ValueError: verify failed on rank 0" in g for g in got)
    else:
        assert all(g == repr({"graph": "from rank 0"}) for g in got)
