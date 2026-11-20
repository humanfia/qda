"""Task helpers for the benchmarks lm-evaluation-harness does not ship in the paper's form.

Only the GPQA answer extraction differs from the harness: the paper reads the strict "The answer is (X)" and
falls back to a lenient "(X)" only when the strict extractor finds nothing. The harness reports the two rules
as two metrics, and neither falls back.
"""

from __future__ import annotations

import hashlib
import random
import re

STRICT = re.compile(r"[Tt]he answer is\s*\(?([A-D])\)?")
LOOSE = re.compile(r"\(([A-D])\)")


def _question_seed(question: str) -> int:
    """A stable seed for one question: sha256, not the per-process-salted built-in ``hash``."""
    return int.from_bytes(hashlib.sha256(question.encode("utf-8")).digest()[:8], "big")


def process_docs(dataset):
    """Shuffle each question's choices the same way everywhere and record the correct choice as ``"(A)"``.

    The permutation is a pure function of the question text, so every row and every process shows the same
    options in the same order; ambient random state and a cached dataset map cannot change a prompt.
    """
    options = ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer"]

    def process(doc):
        answers = sorted(doc[key] for key in options)
        random.Random(_question_seed(doc["Question"])).shuffle(answers)
        return {
            **doc,
            "choice1": answers[0],
            "choice2": answers[1],
            "choice3": answers[2],
            "choice4": answers[3],
            "answer": f"({chr(65 + answers.index(doc['Correct Answer']))})",
        }

    return dataset.map(process)


def strict_then_loose(resps, _docs):
    """Map each instance's responses to one extracted choice letter, strict first and lenient second."""
    out = []
    for responses in resps:
        text = responses if isinstance(responses, str) else responses[0]
        match = STRICT.search(text) or LOOSE.search(text)
        out.append(f"({match.group(1)})" if match else "")
    return out
