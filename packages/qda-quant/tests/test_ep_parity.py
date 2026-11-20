"""Expert-parallel bank forward (engine/ep.py, upstream GSQ 03fc1648 qwen35_moe_dist) against the one-card bank.

Upstream's arithmetic: every rank routes its own rows, the tokens go to the rank that owns each WHOLE expert and come
back, each rank's loss is its own rows' sum (loss SUM, no gradient averaging). So on CPU gloo:
  * 1 rank is `NemotronHExperts.forward`, bit for bit;
  * N ranks give each rank's rows the 1-rank output and input gradient, and each owner the 1-rank gradient of its
    experts (summed over every rank's tokens), within fp32 rounding (only the batching of an expert's tokens changes);
  * a rank's gradient of the experts it does not own is zero;
  * rank r owns `ExpertShare(r, world).experts(n)`, the blocks the gsq solve and `Comm.gather_experts` use.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qda_quant.solve.units import ExpertShare

from .tiny import config

ep = pytest.importorskip("qda_quant.engine.ep")


@pytest.mark.parametrize(("n", "world"), [(8, 1), (8, 2), (10, 4), (64, 2), (64, 3), (128, 4)])
def test_owners_are_the_expert_share_blocks(n, world):
    got = ep.owners(n, world, torch.device("cpu"))
    assert got.dtype == torch.long and got.shape == (n,)
    for r in range(world):
        assert (got == r).nonzero().flatten().tolist() == list(range(n))[ExpertShare(r, world).experts(n)]


def _bank(n: int, seed: int = 0):
    """A real `NemotronHExperts` (relu^2, non-gated) of `n` experts with random fp32 weights."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    cfg = config("nemotron_h")
    cfg.n_routed_experts = n
    bank = NemotronHExperts(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        bank.up_proj.copy_(torch.randn(bank.up_proj.shape, generator=g) * 0.1)
        bank.down_proj.copy_(torch.randn(bank.down_proj.shape, generator=g) * 0.1)
    return bank


def _routes(n_tok: int, n: int, k: int, seed: int, skew: bool):
    """Hidden states, distinct top-k experts per token and their weights. `skew` sends the first half of the rows only
    to the first experts and the rest only to the last ones, so each rank's rows reach one owner (empty splits)."""
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tok, 128, generator=g)
    scores = torch.rand(n_tok, n, generator=g)
    if skew:
        half = n_tok // 2
        scores[:half, n // 2 :] -= 2
        scores[half:, : n // 2] -= 2
    idx = scores.topk(k, dim=1).indices
    w = torch.softmax(torch.randn(n_tok, k, generator=g), dim=1)
    # loss = sum(out * probe): every output element gets a gradient
    probe = torch.randn(n_tok, 128, generator=g)
    return hidden, idx, w, probe


def _run(bank, hidden, idx, w, probe, fn):
    bank.zero_grad(set_to_none=True)
    h = hidden.clone().requires_grad_(True)
    out = fn(bank, h, idx, w)
    (out * probe).sum().backward()
    up = bank.up_proj.grad if bank.up_proj.grad is not None else torch.zeros_like(bank.up_proj)
    down = bank.down_proj.grad if bank.down_proj.grad is not None else torch.zeros_like(bank.down_proj)
    return {"out": out.detach(), "dx": h.grad, "up": up.clone(), "down": down.clone()}


def _rank(rank, world, port, case, out):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        torch.manual_seed(1234 + rank)  # nothing in the forward may depend on a rank's RNG
        bank = _bank(case["n"])
        # a contiguous row shard per rank, as the engine splits calibration rows
        rows = ExpertShare(rank, world).experts(len(case["hidden"]))
        args = [case[k][rows] for k in ("hidden", "idx", "w", "probe")]
        mine = ExpertShare(rank, world).experts(case["n"])
        got = _run(bank, *args, lambda b, *a: ep.forward(b, *a, mine.start, mine.stop, world, None))
        torch.save({"rows": (rows.start, rows.stop), **got}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _spawn(world, case, tmp_path):
    from qda_quant.engine.dist import free_port

    out = str(tmp_path / "rank")
    mp.spawn(_rank, args=(world, free_port(), case, out), nprocs=world, join=True)
    return [torch.load(f"{out}.{r}", weights_only=False) for r in range(world)]


def _case(n, n_tok=37, k=6, seed=0, skew=False):
    hidden, idx, w, probe = _routes(n_tok, n, min(k, n), seed, skew)
    return {"n": n, "hidden": hidden, "idx": idx, "w": w, "probe": probe}


def _reference(case):
    return _run(_bank(case["n"]), case["hidden"], case["idx"], case["w"], case["probe"], lambda b, *a: b(*a))


@pytest.mark.parametrize("n", [8, 64])
def test_one_rank_is_the_bank_forward_bit_for_bit(n, tmp_path):
    case = _case(n)
    ref = _reference(case)
    (got,) = _spawn(1, case, tmp_path)
    assert got["rows"] == (0, len(case["hidden"]))
    assert torch.equal(got["out"], ref["out"])
    for key in ("dx", "up", "down"):
        torch.testing.assert_close(got[key], ref[key], rtol=1e-5, atol=1e-6, msg=key)


@pytest.mark.parametrize(
    ("n", "world", "skew"),
    [(8, 2, False), (64, 2, False), (64, 2, True), (10, 4, False), (64, 3, False)],
)
def test_ranks_match_one_rank_and_train_only_their_own_experts(n, world, skew, tmp_path):
    case = _case(n, skew=skew)
    ref = _reference(case)
    got = _spawn(world, case, tmp_path)
    # the shards cover every row once
    assert [g["rows"][0] for g in got[1:]] == [g["rows"][1] for g in got[:-1]]
    for g in got:  # each rank's rows: the one-rank output and input gradient
        lo, hi = g["rows"]
        torch.testing.assert_close(g["out"], ref["out"][lo:hi], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g["dx"], ref["dx"][lo:hi], rtol=1e-5, atol=1e-6)
    for r, g in enumerate(got):
        mine = ExpertShare(r, world).experts(n)
        others = torch.ones(n, dtype=torch.bool)
        others[mine] = False
        for key in ("up", "down"):
            # loss SUM, no averaging: the owner's gradient is the whole one-rank gradient of its experts
            close = dict(rtol=1e-5, atol=1e-6, msg=f"rank {r} {key}")
            torch.testing.assert_close(g[key][mine], ref[key][mine], **close)
            assert not g[key][others].any(), f"rank {r} has a {key} gradient on an expert it does not own"
    hit = torch.zeros(n, dtype=torch.bool)
    hit[case["idx"].flatten()] = True
    assert ref["up"][hit].abs().sum(dim=(1, 2)).gt(0).all()  # the test reaches every routed expert's gradient
