"""The Qwen3.6-35B-A3B GPTQ plan on the card that failed: the GDN forward's transients are counted.

Needs ``--model-dir qwen3.6-35b-a3b=PATH`` (and ``qwen3-8b=PATH`` for the control); each case skips when its
model is not given. The plan reads no weights: the skeleton is on the meta device and the card's free memory is
stubbed with the 78.7 GiB the failing one had.
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
from qda_quant.target import Target

GIB = 2**30
CARD = [torch.device("cuda:0")]
FREE = int(78.7 * GIB)  # what the failing one-card cell had free when it planned
TOTAL = int(79.18 * GIB)


def _recipe(tmp_path) -> Recipe:
    path = tmp_path / "calib.jsonl"
    line = json.dumps({"text": "The quick brown fox jumps over the lazy dog. " * 400}) + "\n"
    path.write_text(line * 128)
    return Recipe(
        steps=[REGISTRY["gptq"]()],
        calib=FileSource(path=str(path), samples=128, seqlen=2048),
    )


def _card(monkeypatch) -> None:
    """The failing cell's card, without a GPU: ``resolve_devices`` reads the variable, the plan the stub."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (FREE, TOTAL))


def test_the_35b_gptq_plan_counts_the_gdn_transients_and_keeps_room(real_model, tmp_path, monkeypatch):
    """One forward of 2,048 tokens through a 16-key/32-value-head delta rule; the 2.4 GiB headroom stays unused.

    The term counts the rule's own sets, which transformers' reference and fla's kernels each allocate; at this
    size the measured forward allocates 72.2 GiB peak, inside the 76.3 GiB this plan leaves the card, and the
    same with either implementation bound. It does not move the plan off `resident`: what the failing build
    needed was a solve chunk sized from the memory the driver can hand (`solve.free_bytes`); this pins the
    accounting the two now share.
    """
    src = str(real_model("qwen3.6-35b-a3b"))
    _card(monkeypatch)
    plan = compile_plan(src, _recipe(tmp_path), Format.parse("w4a16-sym-g128"), Target(), CARD)
    residency = plan.residency
    terms = dict(residency.peak_terms)
    assert terms["linear-attention transients"] == pytest.approx(0.38, abs=0.01)
    assert residency.weights == "resident"
    assert residency.activations == "device"
    # the plan leaves the headroom (3 % of 78.7 GiB = 2.4 GiB) unused, and the solver's share is what is left
    reserve = max(2 * GIB, int(FREE * 0.03))
    assert residency.peak_gib == pytest.approx((FREE - reserve) / GIB, abs=0.01)
    assert residency.peak_gib == pytest.approx(76.3, abs=0.05)  # the number the failing cell planned
    # the solver term is the room the plan leaves it, and the chunk it names is one the card holds
    (name,) = [n for n in terms if n.startswith("solver workspace")]
    assert " of 256 at a time" in name
    assert terms[name] >= 4.0


def test_the_8b_plan_counts_no_gdn_transients(real_model, tmp_path, monkeypatch):
    """Qwen3-8B has no delta rule: the term is absent and its plan is the same as before the term."""
    src = str(real_model("qwen3-8b"))
    _card(monkeypatch)
    plan = compile_plan(src, _recipe(tmp_path), Format.parse("w4a16-sym-g128"), Target(), CARD)
    assert "linear-attention transients" not in dict(plan.residency.peak_terms)
    assert plan.residency.weights == "resident"
