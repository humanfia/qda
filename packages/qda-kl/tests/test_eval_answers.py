"""The paper's graders and GPQA answer key: the readings a simpler extractor gets wrong, and the paper's shuffle."""

from __future__ import annotations

import pytest

pytest.importorskip("sympy")

from qda_kl.eval import answers  # noqa: E402
from qda_kl.eval.tasks import paper  # noqa: E402


@pytest.mark.parametrize(
    ("region", "gold", "right"),
    [
        (r"so $AB \cdot AC = 468$: \boxed{AB \cdot AC = 468}", "468", True),
        # An unreadable trailing box falls through to the one before it; a fraction is not its numerator.
        (r"\boxed{70} which is \boxed{\dfrac{1200}{7}} per unit", "70", True),
        (r"\boxed{\dfrac{1200}{7}}", "1200", False),
        # The lenient patterns run only when no box reads, and the last candidate wins.
        ("The answer is 12. On reflection, the final answer is 70.", "70", True),
        ("\x08oxed{70}", "70", True),
        ("", "70", False),
    ],
)
def test_aime_reads_the_last_readable_box_then_the_lenient_patterns(
    region: str, gold: str, right: bool
) -> None:
    assert answers.aime(region, gold) is right


@pytest.mark.parametrize(
    ("region", "gold", "right"),
    [
        (r"\boxed{\frac{4-\pi}{2}}", r"2 - \frac{\pi}{2}", True),
        (r"\boxed{74^\circ}", "74", True),
        (r"\boxed{(3+\sqrt{6})^{-1/3}}", r"\frac{1}{\sqrt[3]{3+\sqrt{6}}}", True),
        # Equal at float precision, not exactly: HMMT Feb 2026 #9.
        (r"\boxed{50}", r"50(1 - \frac{1}{2^{101} - 1})", False),
        (r"\boxed{1, 2}", "1", False),
        (r"\boxed{9^{9^{9}}}", "1", False),
    ],
)
def test_hmmt_compares_closed_forms_exactly(region: str, gold: str, right: bool) -> None:
    assert answers.hmmt(region, gold) is right


def test_hmmt_refuses_a_gold_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        answers.hmmt(r"\boxed{1}", "x + 1")


@pytest.mark.parametrize(
    ("region", "gold", "right"),
    [
        # The harness's extractor reads the first "(X)" and scores this (A).
        ("(A) is too small and (B) too large.\nAnswer: (D)", "D", True),
        (r"\boxed{\text{C}} then \boxed{\frac{1}{2}}", "C", True),
        (r"\boxed{\frac{1+nv}{n+v}}", "A", False),
        ("Comparing (A) and (C), only one holds.\n\n**C**", "C", True),
    ],
)
def test_gpqa_reads_the_final_letter(region: str, gold: str, right: bool) -> None:
    assert answers.gpqa(region, gold) is right


class _Rows(list):
    """The slice of ``datasets.Dataset`` the task hooks use."""

    def map(self, fn, with_indices=False):
        return _Rows({**row, **(fn(row, n) if with_indices else fn(row))} for n, row in enumerate(self))


def test_gpqa_options_are_shuffled_as_the_paper_shuffled_them() -> None:
    # Item n's order comes from random.Random(17 + n) over [correct, wrong 1, wrong 2, wrong 3]; these are the
    # paper's gold letters of GPQA-Diamond items 0-3.
    rows = _Rows(
        {
            "Question": f" Q{n} ",
            "Correct Answer": " right ",
            "Incorrect Answer 1": "w1",
            "Incorrect Answer 2": "w2",
            "Incorrect Answer 3": "w3",
        }
        for n in range(4)
    )
    docs = paper.gpqa_docs(rows)
    assert [doc["gold"] for doc in docs] == ["A", "C", "D", "B"]
    assert docs[0]["prompt"] == (
        "Q0\n\nA) right\nB) w2\nC) w1\nD) w3\n\n"
        "Reason step by step, then give the final answer as a single letter (A, B, C or D) inside \\boxed{}."
    )


def test_the_simple_evals_prompt_asks_for_the_same_items_and_gold() -> None:
    rows = _Rows(
        {
            "Question": f" Q{n} ",
            "Correct Answer": " right ",
            "Incorrect Answer 1": "w1",
            "Incorrect Answer 2": "w2",
            "Incorrect Answer 3": "w3",
        }
        for n in range(4)
    )
    plain = paper.gpqa_docs(rows)
    simple = paper.gpqa_simple_evals_docs(rows)
    # Same items, shuffle and gold: only the prompt text differs.
    assert [doc["gold"] for doc in simple] == [doc["gold"] for doc in plain]
    assert simple[0]["prompt"] == (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\nQ0\n\nA) right\nB) w2\nC) w1\nD) w3"
    )
    # The last-line format the prompt asks for is read by the lenient pattern the grader already has.
    assert answers.gpqa("So the correct option is C.\nAnswer: C", "C") is True
