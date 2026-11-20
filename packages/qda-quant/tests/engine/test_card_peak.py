"""The card peak `qda plan` prints: allocated tensors, the CUDA context and the allocator's reserve (11-14)."""

from __future__ import annotations

import pytest
import torch
from qda_quant.engine import residency
from qda_quant.solve import solve
from qda_quant.steps.gptq import GPTQ

GIB = 2**30


@pytest.fixture
def no_alloc_env(monkeypatch):
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)


def test_card_peak_adds_context_and_reserve_rounded_up(no_alloc_env):
    """8 GiB of tensors: + 0.8 context + 8 % reserve (expandable, the default) = 9.44 -> 9.5 GiB."""
    card, terms = residency.card_peak(8 * GIB)
    assert card == 9.5
    assert dict(terms) == {"tensors": 8.0, "CUDA context": 0.8, "allocator reserve": 0.64}
    fixed, _ = residency.card_peak(8 * GIB, expandable=False)
    assert fixed == 12.0  # 8 + 0.8 + 40 % of 8 = 12.0 exactly: no extra tenth


@pytest.mark.parametrize(
    ("value", "expandable"),
    [
        (None, True),
        ("expandable_segments:True", True),
        ("max_split_size_mb:128, expandable_segments: True", True),
        ("expandable_segments:False", False),
        ("max_split_size_mb:128", False),
    ],
)
def test_reserve_follows_the_allocator_setting(monkeypatch, no_alloc_env, value, expandable):
    """The environment's own setting wins, and the reserve share follows it."""
    if value is not None:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", value)
    assert residency.expandable_segments() is expandable
    share = residency.RESERVE_SHARE[expandable]
    assert dict(residency.card_peak(10 * GIB)[1])["allocator reserve"] == round(10 * share, 2)


def test_build_sets_expandable_segments_unless_the_environment_names_one(monkeypatch, no_alloc_env):
    calls: list[str] = []
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", calls.append, raising=False)
    residency.use_expandable_segments()
    assert calls == [residency.ALLOC_CONF]
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    residency.use_expandable_segments()
    assert calls == [residency.ALLOC_CONF]  # the user's setting is left alone
    assert "PYTORCH_CUDA_ALLOC_CONF" not in __import__("os").environ  # nothing a child process inherits


def test_printed_card_peak_is_the_recorded_one(monkeypatch, no_alloc_env):
    """`qda plan` prints the number build.json records (the 1.5-vs-2.47 report): one field, rounded once."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80 * GIB, 80 * GIB))
    got = residency.plan(
        20 * GIB,
        2 * GIB,
        4 * GIB,
        [torch.device("cuda:0")],
        terms=[residency.Term("call arguments", GIB)],
        solves=[residency.Solve("solver workspace", 8, GIB, 1)],
    )
    assert got.peak_gib == 33.0
    assert got.card_gib == residency.card_peak(33 * GIB)[0] == 36.5  # 33 + 0.8 + 2.64 = 36.44
    assert f"card peak (nvidia-smi) {got.card_gib:.1f} GiB" in str(got)
    record = got.as_dict()
    assert record["card_peak_gib"] == got.card_gib
    assert f"{record['card_peak_gib']:.1f}" == str(record["card_peak_gib"])  # no hidden digits
    assert record["peak_gib"] == 33.0  # the tensor peak keeps its meaning


def test_gptq_prices_a_wide_dense_unit_as_measured():
    """Llama-3.1-8B down_proj [4096, 14336], act-order static: the solve held 5.19 GiB at the build's peak."""
    dense = GPTQ().dense_workspace(4096, 14336)
    assert dense >= 5.19 * GIB
    assert dense <= 5.19 * GIB * 1.05
    assert dense > solve.expert_workspace(4096, 14336)  # the per-expert bound (3.5 GiB) was the 1.6 GiB gap
    assert GPTQ().solver_workspace(4096, 14336) == solve.expert_workspace(4096, 14336)  # banks unchanged


def test_a_resident_build_prices_its_model_steps_with_the_weights(monkeypatch, no_alloc_env):
    """Rotate runs with the whole model on the card when it is resident: that stage can be the build's peak."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80 * GIB, 80 * GIB))

    def at(store, step):
        return residency.plan(
            17 * GIB,
            GIB,
            GIB,
            [torch.device("cuda:0")],
            force=store,
            model_step_bytes=step,
            host_free=1024 * GIB,
        )

    assert at("resident", 0).peak_gib == 18.0  # weights 17, activations 1
    got = at("resident", 15 * GIB)
    assert got.peak_gib == 32.0
    assert dict(got.peak_terms) == {"weights": 17.0, "model steps": 15.0}
    assert at("stream", 15 * GIB).peak_gib < 18.0  # a streamed build rotates on the host


def test_rotate_prices_two_float64_embeddings_and_the_rotation():
    from qda_quant.steps.rotate import Rotate

    class Graph:
        embed, lm_head = "embed", "head"

    meta = torch.nn.ModuleDict(
        {
            "embed": torch.nn.Embedding(1000, 64, device="meta"),
            "head": torch.nn.Linear(64, 1000, device="meta"),
        }
    )
    assert Rotate(residual=True).model_step_bytes(meta, Graph()) == 16 * 1000 * 64 + 8 * 64 * 64
    assert Rotate(residual=False, value_heads=True).model_step_bytes(meta, Graph()) == 0


def test_the_cuda_specific_alias_wins_as_in_torch(monkeypatch, no_alloc_env):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    assert residency.expandable_segments() is False
    assert residency.card_peak(8 * GIB)[0] == 12.0  # the fixed-segment reserve


def test_a_failed_allocator_switch_warns_and_prices_the_fixed_reserve(monkeypatch, no_alloc_env):
    def refuse(_setting):
        raise RuntimeError("no")

    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", refuse, raising=False)
    monkeypatch.setattr(residency, "_SET_FAILED", [])
    with pytest.warns(UserWarning, match="could not set the CUDA allocator"):
        residency.use_expandable_segments()
    assert residency.expandable_segments() is False


def test_the_model_stage_takes_part_in_the_fit(monkeypatch, no_alloc_env):
    """24 GiB free: 17 GiB resident weights + 15 GiB of rotation do not fit, so the plan streams; forced, it refuses."""
    from qda_quant.step import Refused

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 * GIB, 24 * GIB))
    got = residency.plan(
        17 * GIB, GIB, GIB, [torch.device("cuda:0")], model_step_bytes=15 * GIB, host_free=1024 * GIB
    )
    assert got.weights != "resident"
    assert got.card_gib <= 24.0
    with pytest.raises(Refused, match="model steps"):
        residency.plan(
            17 * GIB,
            GIB,
            GIB,
            [torch.device("cuda:0")],
            force="resident",
            model_step_bytes=15 * GIB,
            host_free=1024 * GIB,
        )


def test_the_card_overhead_takes_part_in_the_fit(monkeypatch, no_alloc_env):
    """48 GiB free, 44 GiB model: resident tensors (46 GiB) fit the headroom rule, but its card peak (50.5) does not."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (48 * GIB, 48 * GIB))
    got = residency.plan(44 * GIB, GIB, GIB, [torch.device("cuda:0")], host_free=1024 * GIB)
    assert got.weights != "resident"
    assert got.card_gib <= 48.0


def test_an_empty_cuda_alias_still_wins_as_in_torch(monkeypatch, no_alloc_env):
    """torch goes by presence: an empty PYTORCH_CUDA_ALLOC_CONF overrides the generic alias (fixed segments)."""
    calls: list[str] = []
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", calls.append, raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    assert residency.expandable_segments() is False
    residency.use_expandable_segments()
    assert calls == []  # the environment named a setting: left alone


def test_the_card_fit_does_not_refuse_a_placement_whose_own_card_peak_fits(monkeypatch, no_alloc_env):
    """64 GiB free, one 57.5 GiB dense solve: streamed it needs 58.5 GiB of tensors, a 64.0 GiB card peak: it fits."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * GIB, 64 * GIB))
    solve_ = residency.Solve("solver workspace", 1, int(57.5 * GIB), 1)
    got = residency.plan(
        int(4.5 * GIB), GIB, 0, [torch.device("cuda:0")], solves=[solve_], host_free=1024 * GIB
    )
    assert got.peak_gib == 58.5
    assert got.card_gib <= 64.0
