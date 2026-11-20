"""The vendored IFBench checkers are the pinned files, and the task scores them as the paper does.

The digests in ``VENDOR.json`` were taken from github.com/allenai/IFBench at the pinned commit; the test checks
this copy is upstream plus exactly the recorded import rewrite. Scoring is tested with a stand-in library, so no
NLTK data or checkers' dependencies are needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qda_kl.eval.tasks import ifbench

VENDOR = Path(__file__).parents[1] / "src/qda_kl/third_party/ifbench"


def test_vendored_files_match_their_pins() -> None:
    said = json.loads((VENDOR / "VENDOR.json").read_text(encoding="utf-8"))
    for name, digests in said["files"].items():
        got = "sha256:" + hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        assert got == digests["vendored_sha256"], name


def test_the_import_rewrite_is_the_only_edit() -> None:
    for path in VENDOR.glob("*.py"):
        assert not any(line.startswith("from ifbench") for line in path.read_text().splitlines()), path.name


def test_hub_float_kwargs_become_the_official_ints() -> None:
    assert ifbench._official({"N": 3.0, "min_words": 40.0, "percentage": 0.5, "keys": [2.0, "x"]}) == {
        "N": 3,
        "min_words": 40,
        "percentage": 0.5,
        "keys": [2, "x"],
    }


class _Library:
    """Stand-in for the vendored library: a summary says, per response, what passed."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def InputExample(self, **kwargs):  # noqa: N802 -- upstream's name
        self.seen.append(kwargs)
        return SimpleNamespace(**kwargs)

    def _out(self, example, answers, flags_by_text):
        text = answers[example.prompt]
        if text == "boom":
            raise ValueError("checker could not parse the answer")
        return SimpleNamespace(follow_instruction_list=list(flags_by_text[text]))

    def test_instruction_following_strict(self, example, answers):
        return self._out(example, answers, {"good": [True, True], "half": [True, False]})

    def test_instruction_following_loose(self, example, answers):
        return self._out(example, answers, {"good": [True, True], "half": [True, True]})


@pytest.fixture
def library(monkeypatch) -> _Library:
    # The scorer seeds langdetect's detector; langdetect comes with the `eval` extra.
    pytest.importorskip("langdetect")
    fake = _Library()
    monkeypatch.setattr(ifbench, "_LIBRARY", fake)
    return fake


DOC = {
    "key": "7",
    "prompt": "Do the thing.",
    "instruction_id_list": ["length_constraints:number_paragraphs", "words:keywords_specific_position"],
    "kwargs": [{"N": 3.0}, {"keyword": "giggle"}],
}


def test_metrics_average_over_responses_and_count_checker_errors(library: _Library) -> None:
    got = ifbench.process_results(DOC, ["half", "boom"])
    # "half" follows one instruction strictly and both loosely; "boom" errors and follows nothing. The
    # instruction-level metrics are (passed, checked) so the harness can weight prompts by instruction count.
    assert got == {
        "prompt_level_strict_acc": 0.0,
        "prompt_level_loose_acc": 0.5,
        "instruction_level_strict_acc": (1, 4),
        "instruction_level_loose_acc": (2, 4),
        "checker_errors": 0.5,
    }
    assert library.seen[0]["kwargs"] == [{"N": 3}, {"keyword": "giggle"}]
    assert library.seen[0]["instruction_id_list"] == DOC["instruction_id_list"]


def test_instruction_level_metrics_weight_by_instruction_count() -> None:
    # A prompt with two instructions half passed, and a one-instruction prompt passed: 2/3, not mean(0.5, 1).
    assert ifbench.weighted_ratio([(1, 2), (1, 1)]) == pytest.approx(2 / 3)
    assert ifbench.weighted_ratio([]) == 0.0
    assert "aggregation: !function ifbench.weighted_ratio" in (
        Path(ifbench.__file__).with_suffix(".yaml").read_text()
    )


def test_an_empty_answer_follows_nothing_without_an_error(library: _Library) -> None:
    got = ifbench.process_results(DOC, ["  "])
    assert got["prompt_level_loose_acc"] == 0.0
    assert got["checker_errors"] == 0.0
    assert library.seen == []  # the checkers were never asked
