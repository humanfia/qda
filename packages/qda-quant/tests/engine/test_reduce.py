"""Rank-order reduction keeps one receive buffer, including before an expert-parallel solve."""

import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qda_quant import REGISTRY
from qda_quant.arch import describe
from qda_quant.engine.dist import Comm, free_port
from qda_quant.engine.planning import memory_terms
from qda_quant.engine.runner import merge
from qda_quant.format import Format
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ..tiny import model


def _statistics(rank):
    # Cancellation distinguishes the sequential fold from a reassociated/tree reduction.
    value = [1e20, 1.0, -1e20, 1.0][rank % 4]
    return {
        ("expert_hessian", "router"): {
            "in": torch.full((8, 256, 256), value),
            "down": torch.full((8, 64, 64), rank + 0.25, dtype=torch.float64),
        },
        ("hessian", "dense"): torch.arange(256, dtype=torch.float32).reshape(16, 16).T + value,
        ("act_absmax", "router"): {
            "in": torch.tensor([float(rank), -float(rank), float("nan")]),
            "down": torch.tensor([rank / 8], dtype=torch.bfloat16),
        },
        ("n_tokens", "router"): torch.arange(8) + rank,
        ("n_tokens", "dense"): 2**62 + rank,
    }


class _LiveBytes(TorchDispatchMode):
    """Track live tensor storage beyond the local statistics, including all intermediate sums."""

    def __init__(self, stats):
        super().__init__()
        self.local = {
            t.untyped_storage().data_ptr() for t in tree_flatten(stats)[0] if isinstance(t, torch.Tensor)
        }
        self.refs = []
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.refs.extend(weakref.ref(t) for t in tree_flatten(result)[0] if isinstance(t, torch.Tensor))
        self.refs = [r for r in self.refs if r() is not None]
        storage = {}
        for ref in self.refs:
            t = ref()
            if t is not None:
                s = t.untyped_storage()
                if s.data_ptr() not in self.local:
                    storage[s.data_ptr()] = s.nbytes()
        self.peak = max(self.peak, sum(storage.values()))
        return result


def _reduce_rank(rank, world, port, out):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        stats = _statistics(rank)
        track = _LiveBytes(stats)
        with track:
            reduced = Comm(rank, world, torch.device("cpu")).reduce_stats(stats)
        if rank == 0:
            expected = merge([_statistics(r) for r in range(world)], torch.device("cpu"))
            assert reduced is not None and reduced.keys() == expected.keys()
            torch.testing.assert_close(reduced, expected, rtol=0, atol=0, equal_nan=True)
            largest = stats[("expert_hessian", "router")]["in"].numel() * 4
            # Object-gather metadata is small; no rank count may add a second Hessian-sized temporary.
            assert largest <= track.peak <= largest + 4096, (world, track.peak, largest)
            torch.save({"world": world, "receive_peak": track.peak, "largest_leaf": largest}, out)
        else:
            assert reduced is None
            torch.testing.assert_close(stats, _statistics(rank), rtol=0, atol=0, equal_nan=True)
        comm = Comm(rank, world, torch.device("cpu"))
        shared = [("expert_hessian", "router"), ("n_tokens", "router")]
        actual = reduced if rank == 0 else stats
        assert actual is not None
        comm.broadcast_stats(actual, shared)
        expected = merge([_statistics(r) for r in range(world)], torch.device("cpu"))
        torch.testing.assert_close(
            {k: actual[k] for k in shared}, {k: expected[k] for k in shared}, rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_rank_fold_matches_reference_with_one_leaf_of_temporary_storage(world, tmp_path):
    out = tmp_path / "peak.pt"
    mp.spawn(_reduce_rank, args=(world, free_port(), out), nprocs=world, join=True)
    measured = torch.load(out, weights_only=True)
    assert measured["world"] == world


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_plan_counts_a_full_expert_receive_before_splitting_the_solve(world):
    meta = model("qwen3_moe", meta=True)
    terms, _ = memory_terms(
        meta,
        describe(meta, verify=False),
        [REGISTRY["gptq"]()],
        samples=8,
        seqlen=16,
        batch_size=1,
        world=world,
        fmt=Format.parse("w4a16-sym-pc"),
    )
    sized = {term.name: term.bytes for term in terms}
    cfg = meta.config
    receive = cfg.num_experts * max(cfg.hidden_size, cfg.moe_intermediate_size) ** 2 * 4
    assert sized.get("fold buffer", 0) == (receive if world > 1 else 0)
    assert (
        sized["expert Hessians"] == cfg.num_experts * (cfg.hidden_size**2 + cfg.moe_intermediate_size**2) * 4
    )


def test_large_bank_receive_reaches_the_peak_and_refuses_an_undersized_budget(monkeypatch):
    from qda_quant.engine import residency
    from qda_quant.step import Refused
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=4096,
        moe_intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=288,
        num_experts_per_tok=8,
    )
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(cfg)
    terms, _ = memory_terms(
        meta,
        describe(meta, verify=False),
        [REGISTRY["gptq"]()],
        samples=8,
        seqlen=16,
        batch_size=1,
        world=4,
        fmt=Format.parse("w4a16-sym-pc"),
    )
    devices = [torch.device("cuda", rank) for rank in range(4)]
    monkeypatch.setattr(residency, "free_bytes", lambda _: 50 * 2**30)
    result = residency.plan(0, 0, 0, devices, terms)
    assert dict(result.peak_terms)["expert Hessians"] == 22.5
    assert dict(result.peak_terms)["fold buffer"] == 18.0
    # The old split-sized buffer was 9 GiB and incorrectly admitted this 40 GiB budget.
    monkeypatch.setattr(residency, "free_bytes", lambda _: 40 * 2**30)
    with pytest.raises(Refused, match="the build needs .* at its peak"):
        residency.plan(0, 0, 0, devices, terms)
