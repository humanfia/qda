r"""The paper's AIME 2025, HMMT February 2026 and GPQA-Diamond tasks: its prompts, items, answer keys and graders.

The harness's own ``aime25`` and ``gpqa_diamond`` (and the ``hmmt_feb_2026`` shipped beside them) ask
"Question: ...\nAnswer:" and read the first "(X)" or the harness's AIME normaliser; the paper asked for a
``\boxed{}`` answer and read it with :mod:`qda_kl.eval.answers`. These tasks render the paper's prompt text as the
user message, keep its item order and gold, and shuffle each GPQA question's options as the paper did: item ``n``
of the dataset with ``random.Random(17 + n)``, so every gold letter is the paper's. The answer the grader reads is
the suite's answer region (``think_end_token``), and the tasks stop only at EOS, as the paper's generations did.

``paper_gpqa_diamond_simple_evals`` is the same task with openai/simple-evals' multiple-choice prompt, which App. B
uses on DeepSeek-R1-Distill-Llama-8B because that model's answers to the default prompt fail to parse in 4.9% of
samples: the same items, shuffle and gold, and the same strict-then-lenient grader.
"""

from __future__ import annotations

import random
import re

from qda_kl.eval import answers

AIME_PROMPT = "{problem}\n\nReason step by step, then give the final answer as an integer inside \\boxed{{}}."
#: MathArena's prompt for this competition (eth-sri/matharena ``hmmt_feb_2026.yaml``), instruction first.
HMMT_PROMPT = "Put your final answer within \\boxed{{}}.\n\n{problem}"
GPQA_PROMPT = (
    "{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
    "Reason step by step, then give the final answer as a single letter (A, B, C or D) inside \\boxed{{}}."
)
#: openai/simple-evals ``QUERY_TEMPLATE_MULTICHOICE``: the prompt DeepSeek-R1 reports for GPQA-Diamond, which
#: App. B asks for on that model because its answers to the default prompt fail to parse in 4.9% of samples.
GPQA_SIMPLE_EVALS_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)
#: The seed of the paper's GPQA option shuffle; part of the answer key, not of the sampling.
GPQA_SEED = 17


def aime_docs(dataset):
    """Each problem with the paper's prompt and its gold: the answer's first integer ("" when it has none)."""

    def item(doc):
        gold = re.search(r"-?\d+", str(doc["answer"]))
        return {"prompt": AIME_PROMPT.format(problem=doc["problem"]), "gold": gold.group(0) if gold else ""}

    return dataset.map(item)


def hmmt_docs(dataset):
    """Each problem with MathArena's prompt and its closed-form gold."""
    return dataset.map(
        lambda doc: {
            "prompt": HMMT_PROMPT.format(problem=str(doc["problem"]).strip()),
            "gold": str(doc["answer"]),
        }
    )


def gpqa_docs(dataset, template: str = GPQA_PROMPT):
    """Each question with its options in the paper's shuffled order and the letter of the correct one.

    ``template`` is the prompt with ``{question}``, ``{a}``-``{d}``; the item order, the shuffle and the gold are
    the same for every template, so the paper's prompt and the simple-evals prompt are the same questions.
    """

    def shuffled(doc, n):
        options = [
            str(doc[key]).strip()
            for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
        ]
        order = list(range(4))
        random.Random(GPQA_SEED + n).shuffle(order)
        a, b, c, d = (options[i] for i in order)
        return {
            "prompt": template.format(question=str(doc["Question"]).strip(), a=a, b=b, c=c, d=d),
            "gold": "ABCD"[order.index(0)],
        }

    return dataset.map(shuffled, with_indices=True)


def gpqa_simple_evals_docs(dataset):
    """The same items and gold as :func:`gpqa_docs`, with the simple-evals prompt App. B asks for on R1."""
    return gpqa_docs(dataset, GPQA_SIMPLE_EVALS_PROMPT)


def aime_results(doc, results):
    """The paper's AIME verdict on one answer region."""
    return {"exact_match": float(answers.aime(results[0], doc["gold"]))}


def hmmt_results(doc, results):
    """The paper's HMMT verdict (exact symbolic equivalence) on one answer region."""
    return {"exact_match": float(answers.hmmt(results[0], doc["gold"]))}


def gpqa_results(doc, results):
    """The paper's GPQA verdict on one answer region."""
    return {"exact_match": float(answers.gpqa(results[0], doc["gold"]))}
