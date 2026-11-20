"""qda-kl refuses a format the card cannot run, and the tables are where the fact lives."""

from __future__ import annotations

import pytest
from qda_quant import PRESETS, Format
from qda_quant.target import capabilities_serving, serving_refusal


def test_only_sm90_plans_w4a8_and_the_serving_refusal_names_the_alternative():
    """The tables own the fact; qda-kl prints their sentence before an engine starts."""
    assert capabilities_serving(PRESETS["w4a8"]) == frozenset({(9, 0)})
    for capability in ((8, 0), (8, 9), (12, 0)):
        why = serving_refusal(PRESETS["w4a8"], [("cuda:0", capability)])
        assert why is not None and "compute capability 9.0 only" in why and "w4a16" in why
        assert f"is compute capability {capability[0]}.{capability[1]}" in why
    assert serving_refusal(PRESETS["w4a8"], [("cuda:0", (9, 0))]) is None
    assert serving_refusal(PRESETS["w4a16"], [("cuda:0", (8, 9))]) is None, "weight-only is not refused here"


def test_fp8_w8a8_is_planned_from_ada_and_an_ampere_card_is_refused():
    """The dense row spans 8.9 and up; below 8.9 vLLM serves the checkpoint as W8A16, so an A100 is refused."""
    served = frozenset({(8, 9), (9, 0), (10, 0), (10, 3), (12, 0)})
    assert capabilities_serving(Format.parse("w8a8-fp8")) == served
    why = serving_refusal(Format.parse("w8a8-fp8"), [("cuda:0", (8, 0))])
    assert why is not None and "FP8 W8A8 kernel" in why and "w4a16" in why
    assert "cuda:0 is compute capability 8.0" in why
    assert serving_refusal(Format.parse("w8a8-fp8"), [("cuda:0", (8, 9))]) is None


@pytest.mark.parametrize(
    ("name", "served"), [("w3a16", True), ("w3a16-sym-g128", True), ("w3a16-sym-g32", False)]
)
def test_an_a100_plans_w3a16_through_humming_in_its_groups(name, served):
    """Ampere gained the int3 row (Humming, groups 64 and 128); a weight-only format is never refused at serve."""
    fmt = Format.parse(name)
    assert ((8, 0) in capabilities_serving(fmt)) is served
    assert ((8, 0) in capabilities_serving(fmt, "moe")) is served
    assert serving_refusal(fmt, [("cuda:0", (8, 0))]) is None
