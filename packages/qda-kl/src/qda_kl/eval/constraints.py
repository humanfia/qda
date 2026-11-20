"""Rule verifiers for the constraints the generation-check instruction prompts ask for.

The shipped instruction prompts (:mod:`qda_kl.gencheck`) ask for exactly five numbered sections, a descriptive
heading and a concrete example in each, and a one-sentence summary at the end. IFBench has no checker for
numbered prose sections, and importing it needs NLTK, emoji and syllapy, which a serving environment does not
carry; these four rules are the fast equivalents, with IFEval's scoring shape: one boolean per constraint per
answer, and an instruction-level pass rate. Like IFEval's, the checks are shallow: they test the requested shape
(a heading is a non-empty line after the number, an example is the word ``example`` or ``e.g.`` in the section, a
summary is a one-sentence final paragraph), not the quality of the content. The rules belong to the shipped
instruction prompts; the sanity suite scores them for that set only and refuses a custom set with instruction rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

#: A numbered section marker: ``1.``, ``**2.**``, ``### 3.``, ``Section 4:``, at the start of a line.
MARKER = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*(?:section[ \t]*)?(\d{1,2})[ \t]*[.):][ \t]*(?:\*\*)?[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def sections(text: str) -> list[tuple[int, str, str]]:
    """The numbered sections of ``text``: ``(number, heading, body)`` in the order they appear."""
    marks = list(MARKER.finditer(text))
    return [
        (
            int(mark.group(1)),
            mark.group(2).strip(" \t*#_"),
            text[mark.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)],
        )
        for i, mark in enumerate(marks)
    ]


def _five(text: str) -> list[tuple[int, str, str]] | None:
    """The sections when the answer has exactly the five the prompt asked for, else None."""
    got = sections(text)
    return got if [number for number, _, _ in got] == [1, 2, 3, 4, 5] else None


def _ok(text: str, test: Callable[[str, str], object]) -> bool:
    """Whether every one of the five sections passes ``test(heading, body)``."""
    found = _five(text)
    return found is not None and all(test(heading, body) for _, heading, body in found)


def numbered(text: str) -> bool:
    """Exactly five numbered sections, numbered 1 to 5 in order."""
    return _five(text) is not None


def headings(text: str) -> bool:
    """Every numbered section carries a non-empty heading."""
    return _ok(text, lambda heading, _: re.search(r"\w", heading))


def examples(text: str) -> bool:
    """Every numbered section names an example."""
    return _ok(text, lambda _, body: re.search(r"\bexample\b|\be\.g\.", body, re.I))


def summary(text: str) -> bool:
    """The fifth section ends with a one-sentence paragraph."""
    found = _five(text)
    if found is None:
        return False
    last = found[4][2].strip().split("\n\n")[-1].strip()
    if not last or last[-1] not in ".!?":
        return False
    return len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", last) if part.strip()]) == 1


#: One rule per constraint, keyed by the name the result reports.
CHECKS = {"sections": numbered, "headings": headings, "examples": examples, "summary": summary}


def score(answers: Sequence[str]) -> dict[str, Any] | None:
    """The pass rates over ``answers``; None when there is no instruction answer to check.

    Returns:
        ``instruction_rate`` (passed constraints over checked ones), ``prompt_rate`` (answers that pass all
        four), ``prompts``, and each rule's pass rate.
    """
    if not answers:
        return None
    passed = {rule: [check(answer) for answer in answers] for rule, check in CHECKS.items()}
    return {
        "prompts": len(answers),
        "instruction_rate": sum(map(sum, passed.values())) / (len(answers) * len(CHECKS)),
        "prompt_rate": sum(all(values[i] for values in passed.values()) for i in range(len(answers)))
        / len(answers),
        "rules": {rule: sum(values) / len(answers) for rule, values in passed.items()},
    }
