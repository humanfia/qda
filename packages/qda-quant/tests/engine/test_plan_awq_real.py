"""Parent-objective AWQ on a real 27B config: the search's live output lists are planned.

Needs ``--model-dir qwen3-8b=PATH`` and ``qwen3.8-27b=PATH`` (see conftest.py); each case skips when its model is
not given. The plan reads no weights: the skeleton is on the meta device and the card's free memory is stubbed.
"""

from __future__ import annotations

import json

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.calib.sources import FileSource
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.target import Target
from transformers import AutoConfig

GIB = 2**30
CPU = [torch.device("cpu")]
TOKENS = 128 * 2048  # the one-card calibration of the issue: every token is retained, and re-priced


def _recipe(tmp_path) -> Recipe:
    path = tmp_path / "calib.jsonl"
    line = json.dumps({"text": "The quick brown fox jumps over the lazy dog. " * 400}) + "\n"
    path.write_text(line * 128)
    return Recipe(
        steps=[REGISTRY["awq"](), REGISTRY["rtn"]()],
        calib=FileSource(path=str(path), samples=128, seqlen=2048),
    )


def _card(monkeypatch, cards: int = 1, gib: int = 80) -> None:
    """`cards` 80 GiB cards' free memory, without a GPU: the variable names them, the plan reads the stub."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(cards)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (gib * GIB, gib * GIB))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cards)


def test_the_27b_search_outputs_are_counted(real_model, tmp_path):
    """Two lists of the widest group's output for every retained token, and the plan is no longer 72 GiB."""
    src = str(real_model("qwen3.8-27b"))
    plan = compile_plan(src, _recipe(tmp_path), Format.parse("w4a16"), Target(), CPU)
    terms = dict(plan.residency.peak_terms)
    # the widest group is a linear-attention layer's in_proj_qkv + in_proj_z: 10240 + 6144 = 16384 wide
    assert terms["search outputs"] == pytest.approx(2 * TOKENS * 16384 * 2 / GIB, abs=0.01)
    # the search's weight copies: gate+up (2 x 17408 x 5120) and the widest single linear's candidate copies
    assert terms["search weights"] == pytest.approx((178257920 * 10 + 89128960 * 14) / GIB, abs=0.01)
    assert plan.residency.peak_gib >= 77.85  # the allocated peak the failing one-card builds reached


@pytest.mark.parametrize("cards", [2, 4])
def test_the_27b_multicard_gather_is_planned(real_model, tmp_path, monkeypatch, cards):
    """Rank 0 gathers every card's retained rows and `layer_io` batches; the plan holds a pad for each
    rank's longest shard, the gathered inputs and hidden states with their pads on top of the retained rows,
    and refuses up front a card that cannot hold the peak instead of OOMing in the merge."""
    src = str(real_model("qwen3.8-27b"))
    text = AutoConfig.from_pretrained(src).get_text_config()
    hidden, widest = text.hidden_size, text.intermediate_size  # 27B: 5120, 17408 (the down projection)
    _card(monkeypatch, cards)
    devs = [torch.device(f"cuda:{i}") for i in range(cards)]
    plan = compile_plan(src, _recipe(tmp_path), Format.parse("w4a16"), Target(), devs)
    terms = dict(plan.residency.peak_terms)
    shard = TOKENS // cards  # 128 x 2048 rows split contiguously, so every rank has the same row count
    assert terms["gather pads"] == pytest.approx(shard * widest * 2 / GIB, abs=0.01)
    # the receive: every rank's shard at the widest retained input, held beside the rows rank 0 keeps
    assert terms["gathered inputs"] == pytest.approx(cards * shard * widest * 2 / GIB, abs=0.01)
    # AWQ's parent objective keeps the batches as views into the receive, so there is no compaction copy
    assert "compacted inputs" not in terms and "compacted bank inputs" not in terms
    assert terms["gathered layer_io"] == pytest.approx(cards * shard * hidden * 2 / GIB, abs=0.01)
    assert terms["layer_io pad"] == pytest.approx(shard * hidden * 2 / GIB, abs=0.01)
    assert plan.residency.peak_gib < 80  # it fits the card it was planned against
    # the measured one-card oom high-water was ~78.3 GiB with the merge's third copy; the planned peak now
    # holds the gather, and a card too small for it is refused before any rank starts
    _card(monkeypatch, cards, gib=int(plan.residency.peak_gib) - 5)
    with pytest.raises(Refused, match="the build needs"):
        compile_plan(src, _recipe(tmp_path), Format.parse("w4a16"), Target(), devs)


def test_the_27b_one_card_build_no_longer_plans_resident(real_model, tmp_path, monkeypatch):
    """The resident commitment is over the card, so the planner streams it; a forced resident plan refuses."""
    src = str(real_model("qwen3.8-27b"))
    rec = _recipe(tmp_path)
    _card(monkeypatch)
    plan = compile_plan(src, rec, Format.parse("w4a16"), Target(), [torch.device("cuda:0")])
    assert plan.residency.weights == "stream"  # one layer, not 88 GiB, and that fits
    assert plan.residency.peak_gib < 77.85
    with pytest.raises(
        Refused, match=r"cuda:0: the build needs 9[01]\.\d GiB .* largest term is the weights"
    ):
        compile_plan(
            src, rec, Format.parse("w4a16"), Target(), [torch.device("cuda:0")], residency="resident"
        )


def test_the_8b_one_card_build_still_plans_resident(real_model, tmp_path, monkeypatch):
    """The term is real there too (the hidden width, twice, over every token), and the build still fits."""
    src = str(real_model("qwen3-8b"))
    _card(monkeypatch)
    plan = compile_plan(src, _recipe(tmp_path), Format.parse("w4a16"), Target(), [torch.device("cuda:0")])
    terms = dict(plan.residency.peak_terms)
    assert terms["search outputs"] == pytest.approx(2 * TOKENS * 4096 * 2 / GIB, abs=0.01)  # 4096 hidden
    # gate+up (2 x 12288 x 4096) and the widest single linear's candidate copies (12288 x 4096)
    assert terms["search weights"] == pytest.approx((100663296 * 10 + 50331648 * 14) / GIB, abs=0.01)
    assert plan.residency.weights == "resident"
    assert plan.residency.peak_gib < 77.85
