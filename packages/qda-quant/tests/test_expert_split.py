"""The routed-expert solve in parts: across cards (partitioning, per-share solves, the rank-order gather) and
in chunks of experts (a memory bound that must not change the bytes). CPU."""

import copy
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qda_quant import REGISTRY
from qda_quant.solve import solve
from qda_quant.solve.units import SHARE, ExpertShare

from .ctx import contexts
from .tiny import model


@pytest.mark.parametrize("n", [1, 5, 8, 128])
@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_shares_cover_the_bank_in_rank_order(n, world):
    spans = [ExpertShare(r, world).experts(n) for r in range(world)]
    assert spans[0].start == 0 and spans[-1].stop == n
    assert all(a.stop == b.start for a, b in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize(("world", "want"), [(1, False), (2, True), (8, True), (9, False)])
def test_split_needs_one_expert_per_card(world, want):
    """The tiny MoE has 8 experts per bank: 9 cards would leave one card without experts (rank-0 solve)."""
    from qda_quant.arch import describe
    from qda_quant.engine.runner import splits_experts

    m = model("qwen3_moe", dtype=torch.bfloat16)
    lg = describe(m, verify=False).layers[0]
    layer = m.get_submodule(lg.module)
    assert splits_experts(layer, lg, [REGISTRY["gptq"]()], world) is want
    assert splits_experts(layer, lg, [REGISTRY["gptaq"]()], world) is False  # GPTAQ does not split


def _solve(m, fmt, share=None):
    """GPTQ every layer of `m` (torch impl); with `share`, only that card's part. Returns the qparams."""
    qparams = {}
    for ctx in contexts(m, fmt):
        ctx.qparams = qparams
        if share is not None:
            ctx.stats[(SHARE, ctx.graph.module)] = share
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
    return qparams


@pytest.mark.parametrize(("fmt", "world"), [("w4a16-sym-g128", 2), ("w4a16", 3)])
def test_share_solves_equal_the_whole_bank_solve(fmt, world):
    base = model("qwen3_moe", dtype=torch.bfloat16)
    whole_m = copy.deepcopy(base)
    whole = _solve(whole_m, fmt)
    banks = [k for k, v in whole.items() if v["codes"].dim() == 3]
    assert banks
    for r in range(world):
        m = copy.deepcopy(base)
        part = _solve(m, fmt, ExpertShare(r, world))
        assert (set(part) == set(whole)) if r == 0 else (set(part) == set(banks))  # dense: card 0 only
        for name in banks:
            qp, ex = part[name], ExpertShare(r, world).experts(whole[name]["codes"].shape[0])
            assert qp["experts"] == (ex.start, ex.stop)
            for f in ("codes", "scale", "zero_point"):
                if f in whole[name]:
                    assert torch.equal(qp[f], whole[name][f][ex]), (name, f)
            p, wp = m.get_parameter(name), whole_m.get_parameter(name)
            assert torch.equal(p.data[ex], wp.data[ex]), name


def _weights(m, name):
    """The parameter a qparams name belongs to: a module's weight, or a fused bank's stacked parameter."""
    try:
        return m.get_parameter(name)
    except AttributeError:
        return m.get_submodule(name).weight


@pytest.mark.parametrize("fmt", ["w4a16", "w4a8", "w3a16"])  # bf16 group scales, FP8 ones, 3-bit codes
@pytest.mark.parametrize("chunk", [1, 3, 5, 8])  # 8 is the whole bank; 3 and 5 leave a shorter last chunk
def test_chunked_bank_solves_equal_the_whole_bank(chunk, fmt, monkeypatch):
    base = model("qwen3_moe", dtype=torch.bfloat16)
    whole_m = copy.deepcopy(base)
    whole = _solve(whole_m, fmt)
    m = copy.deepcopy(base)
    monkeypatch.setattr(solve, "chunk_experts", lambda *_: chunk)
    got = _solve(m, fmt)
    assert set(got) == set(whole)
    for name, qp in whole.items():
        for f in ("codes", "scale", "zero_point"):
            if f in qp:
                assert torch.equal(got[name][f], qp[f]), (name, f)
        assert torch.equal(_weights(m, name).data, _weights(whole_m, name).data), name


@pytest.mark.parametrize(
    ("rank", "every_rank", "want_dense"), [(0, False, True), (1, False, False), (1, True, True)]
)
def test_dense_groups_of_a_split_follow_the_step(rank, every_rank, want_dense):
    """A split layer's dense groups are rank 0's unless the step trains them on every card (GSQ under EP)."""
    from qda_quant.solve.units import bound

    ctx = next(iter(contexts(model("qwen3_moe", dtype=torch.bfloat16), "w4a16")))
    ctx.stats[(SHARE, ctx.graph.module)] = ExpertShare(rank, 2)
    got = {u.part for u in bound(REGISTRY["gsq"](expert_parallel=every_rank), ctx)}
    assert (None in got) is want_dense
    assert any(part is not None for part in got)  # this card's bank is always bound


def test_banks_pad_their_factor_batch_and_dense_groups_do_not(monkeypatch):
    """A short bank chunk is factored as the whole bank would be; a dense group keeps its batch of 1."""
    seen = []
    real = solve.gptq

    def spy(w, *a, **kw):
        seen.append((w.shape[0], kw.get("min_batch")))
        return real(w, *a, **kw)

    monkeypatch.setattr(solve, "gptq", spy)
    monkeypatch.setattr(solve, "chunk_experts", lambda *_: 3)
    _solve(model("qwen3_moe", dtype=torch.bfloat16), "w4a16")
    assert (1, None) in seen  # a dense group
    assert seen and all(mb == solve.MIN_FACTOR_BATCH for n, mb in seen if n != 1)


def test_chunks_inside_a_card_share_use_absolute_expert_bounds(monkeypatch):
    base = model("qwen3_moe", dtype=torch.bfloat16)
    whole_m = copy.deepcopy(base)
    whole = _solve(whole_m, "w4a16")
    m = copy.deepcopy(base)
    monkeypatch.setattr(solve, "chunk_experts", lambda *_: 3)
    share = ExpertShare(1, 2)
    got = _solve(m, "w4a16", share)
    for name, qp in got.items():
        ex = share.experts(whole[name]["codes"].shape[0])
        assert qp["experts"] == (ex.start, ex.stop)
        for f in ("codes", "scale", "zero_point"):
            if f in whole[name]:
                assert torch.equal(qp[f], whole[name][f][ex]), (name, f)
        assert torch.equal(_weights(m, name).data[ex], _weights(whole_m, name).data[ex]), name


def _gather_rank(rank, world, port, parts, out):
    from qda_quant.engine.dist import Comm

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        layer = torch.nn.Module()
        layer.experts = torch.nn.Module()
        layer.experts.w = torch.nn.Parameter(parts["base"].clone(), requires_grad=False)
        lo, hi = parts["spans"][rank]
        layer.experts.w.data[lo:hi] = parts["served"][lo:hi]
        qp = {
            "L.experts.w": {
                "codes": parts["codes"][lo:hi].clone(),
                "scale": parts["scale"][lo:hi].clone(),
                "format": "w4a16",
                "experts": (lo, hi),
            },
            "L.dense": {"codes": torch.zeros(1), "format": "w4a16"},  # not a share: left alone
        }
        comm = Comm(rank, world, torch.device("cpu"))
        comm.gather_experts(qp, layer, "L")
        mine = {  # rank 0's values replace every rank's own
            ("expert_hessian", "r"): {
                "in": torch.full((2, 3, 3), float(rank + 1)),
                "down": torch.full((2, 2, 2), 7.0 * rank),
            },
            ("n_tokens", "r"): torch.tensor([rank, 10 + rank]),
            ("n_tokens", "g"): 5 + rank,
        }
        comm.broadcast_stats(mine, list(mine))
        assert torch.equal(mine[("expert_hessian", "r")]["in"], torch.ones(2, 3, 3))
        assert torch.equal(mine[("expert_hessian", "r")]["down"], torch.zeros(2, 2, 2))
        assert torch.equal(mine[("n_tokens", "r")], torch.tensor([0, 10])) and mine[("n_tokens", "g")] == 5
        if rank == 0:
            torch.save({"qp": qp, "w": layer.experts.w.data}, out)
        else:
            assert list(qp) == ["L.dense"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_gather_experts_and_broadcast_stats(world, tmp_path):
    from qda_quant.engine.dist import free_port

    n = 7
    g = torch.Generator().manual_seed(0)
    parts = {
        "base": torch.zeros(n, 4, 8, dtype=torch.bfloat16),
        "served": torch.randn(n, 4, 8, generator=g).to(torch.bfloat16),
        "codes": torch.randint(0, 16, (n, 4, 8), generator=g, dtype=torch.int8),
        "scale": torch.rand(n, 4, 1, generator=g).to(torch.bfloat16),
    }
    parts["spans"] = [(s.start, s.stop) for s in (ExpertShare(r, world).experts(n) for r in range(world))]
    out = tmp_path / "rank0.pt"
    mp.spawn(_gather_rank, args=(world, free_port(), parts, str(out)), nprocs=world, join=True)
    got = torch.load(out, weights_only=False)
    qp = got["qp"]["L.experts.w"]
    assert torch.equal(got["w"], parts["served"])
    assert torch.equal(qp["codes"], parts["codes"]) and torch.equal(qp["scale"], parts["scale"])
    assert "experts" not in qp and qp["format"] == "w4a16" and "weight_check" in qp


@pytest.fixture
def tiny_moe(tmp_path):
    """A saved tiny Qwen3-MoE (fused expert banks) with calibration ids."""
    from .engine_util import save_tiny_tokenizer, tiny_qwen3_moe, write_ids

    tiny_qwen3_moe(layers=2).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(5)], sha
    )
    return tmp_path


def _tiny_v41(tmp_path):
    """A saved tiny DeepSeek-V4.1 (unfused expert banks: one Linear per expert) with calibration text."""
    from .test_arch_deepseek_v41 import save_tokenizer, tiny_v41

    tiny_v41().save_pretrained(tmp_path / "m", save_original_format=False)
    save_tokenizer(tmp_path / "m")
    text = [json.dumps({"text": " ".join(f"w{i + j}" for i in range(20))}) for j in range(5)]
    (tmp_path / "c.jsonl").write_text("\n".join(text) + "\n")
    return tmp_path


@pytest.mark.parametrize(
    ("bank", "hessian"),
    # fast: one flush over several samples; blend: GPTQ mixes in all-token Hessians, kept per chunk
    [("fused", "default"), ("fused", "fast"), ("fused", "blend"), ("unfused", "default")],
)
def test_expert_statistics_in_chunks_build_the_whole_layers_bytes(tiny_moe, monkeypatch, bank, hessian):
    """Chunks of 3 (a short last chunk) and of 1 expert write the one-chunk build's files.

    Fused banks replay the recorded bank calls; unfused ones run the layer again for each chunk.
    """
    import dataclasses
    import hashlib
    import importlib

    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.build import build
    from qda_quant.recipe import Recipe

    src = tiny_moe if bank == "fused" else _tiny_v41(tiny_moe / "v41")
    engine_build = importlib.import_module("qda_quant.engine.build")
    plan = engine_build.compile_plan

    def files(chunks, blend=0.25 if hessian == "blend" else 0.0):
        def chunked(*args, **kwargs):
            p = plan(*args, **kwargs)
            return dataclasses.replace(p, residency=dataclasses.replace(p.residency, expert_chunks=chunks))

        monkeypatch.setattr(engine_build, "compile_plan", chunked)
        out = src / f"out-{chunks}-{blend}"
        calib = FileSource(path=str(src / "c.jsonl"), samples=4, seqlen=16)
        rec = Recipe(steps=[REGISTRY["gptq"](impl="torch", all_token_blend=blend)], calib=calib)
        build(
            str(src / "m"),
            rec,
            format="w4a16-sym-g32",
            out_dir=out,
            devices=["cpu"],
            hessian="default" if hessian == "blend" else hessian,
            log=print,
        )
        info = json.loads((out / "build.json").read_text())
        assert ("replay" in info["timings"]["phases"]) is (chunks > 1)
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.name not in ("build.json", "resolved.yaml")
        }

    whole = files(1)
    assert files(3) == whole
    assert files(8) == whole
    if hessian == "blend":  # the blend reached the solve
        assert files(1, 0.0) != whole


def test_plan_gathers_a_bank_larger_than_the_card_in_the_fewest_chunks(tiny_moe, monkeypatch):
    """Kimi K3's bank (896 experts top-16, widths 3584 and 3072: 74.4 GiB of Hessians) on a 94 GiB card."""
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine import residency
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.recipe import Recipe
    from qda_quant.step import Refused
    from qda_quant.target import Target
    from transformers import Qwen3MoeConfig

    from .engine_util import save_tiny_tokenizer

    src = tiny_moe / "k3-bank"
    Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=3584,
        moe_intermediate_size=3072,
        num_hidden_layers=1,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=896,
        num_experts_per_tok=16,
        architectures=["Qwen3MoeForCausalLM"],
    ).save_pretrained(src)
    save_tiny_tokenizer(src)
    rec = lambda: Recipe(  # noqa: E731
        steps=[REGISTRY["gptq"]()], calib=FileSource(path=str(tiny_moe / "c.jsonl"), samples=4, seqlen=16)
    )
    card = [torch.device("cuda", 0)]

    def plan(gib):
        monkeypatch.setattr(residency, "free_bytes", lambda _: int(gib * 2**30))
        return compile_plan(str(src), rec(), Format.parse("w4a16"), Target(), card).residency

    per_expert = (3584**2 + 3072**2) * 4
    res = plan(94.0)
    hessians = dict(res.peak_terms)["expert Hessians"]
    assert res.expert_chunks > 1 and res.peak_gib <= 94.0
    assert hessians == round(-(-896 // res.expert_chunks) * per_expert / 2**30, 2)
    # even a 500 GiB card solves this bank in 10 chunks: an expert's gate_up is 6144 x 3584 = 22.0M
    # elements, and 2^31 elements (the solve's int32 offsets) hold 97 experts, 896 / 97 rounded up
    assert plan(500.0).expert_chunks == 10
    with pytest.raises(Refused, match="solver workspace"):
        plan(60.0)


def test_solves_of_an_expert_parallel_layer_are_joint(tiny_moe):
    """xy-2140: gsq builds and trains every unit of an expert-parallel layer in one optimizer and never cuts
    one, so the plan must hold that layer's units together, not the largest alone (`residency._peak`)."""
    from qda_quant.arch import describe
    from qda_quant.engine import memory
    from qda_quant.format import Format
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny_moe / "m", dtype=torch.bfloat16)
    lg = describe(m, verify=False).layers[0]
    steps = [REGISTRY["gptq"](), REGISTRY["gsq"]()]
    fmt = Format.parse("w4a16")
    joint = memory._solves(m, lg, steps, 2, fmt, None, joint=True)
    apart = memory._solves(m, lg, steps, 2, fmt, None)
    assert joint and all(s.joint for s in joint) and not any(s.joint for s in apart)
    # the same workspaces either way: only how many of them the plan holds at once changes
    assert [(s.experts, s.workspace) for s in joint] == [(s.experts, s.workspace) for s in apart]
    # 8 routed experts over 2 cards: each rank's share is 4; a dense group is whole, one
    bank = [s.experts for s in joint if s.experts > 1]
    assert bank and all(e == 4 for e in bank) and any(s.experts == 1 for s in joint)
    # gptq alone completes one unit before the next, so the flag must not reach it from the layer alone
    assert not any(s.joint for s in memory._solves(m, lg, steps[:1], 2, fmt, None, joint=True))
