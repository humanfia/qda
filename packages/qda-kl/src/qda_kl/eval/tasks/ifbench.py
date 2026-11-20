"""IFBench scoring for the ``ifbench`` task: AllenAI's official checkers on each answer.

The checkers are vendored verbatim (``qda_kl/third_party/ifbench``, Apache-2.0). This module supplies only what
upstream leaves to the caller:

* the answer text, and per-check seeding of ``random`` and ``langdetect`` (both are drawn inside the checkers,
  and an unseeded scorer disagreed with itself);
* the int coercion the Hub release needs: its parquet stores integer kwargs as float64 (``N: 3.0``), while the
  official JSONL ships ints, and checkers such as ``range(kwargs["N"])`` would receive them;
* a checker exception on one answer counted as "not followed" (``checker_errors``) rather than ending the run;
  a broken environment (missing module or NLTK resource) still raises.

Metrics are upstream's four: strict and loose, at prompt and instruction level. The paper reports prompt-level
loose, and the suite's ``metric`` selects the headline; the other metrics remain the harness's results. The
instruction-level metrics return passed and total instructions per prompt, so the harness's weighted aggregation
(:func:`weighted_ratio`) gives upstream's count-weighted mean rather than a mean of per-prompt fractions.
"""

from __future__ import annotations

import copy
import random

from qda_kl.eval import INSTALL_EVAL

#: Every NLTK resource the vendored checkers load.
NLTK_RESOURCES = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)

#: What a checker legitimately raises on an answer it cannot parse. Anything else (``LookupError``,
#: ``NameError``, ``ImportError``) is the environment and must end the run instead of lowering the score.
CHECKER_ERRORS = (ValueError, TypeError, AttributeError, KeyError, IndexError, ZeroDivisionError)

_LIBRARY = None


def _library():
    """The vendored evaluation library, imported once its dependencies and NLTK data are present.

    Raises:
        RuntimeError: a checker dependency is not installed, or an NLTK resource is missing.
    """
    global _LIBRARY  # noqa: PLW0603 -- the library and its check are built once per process
    if _LIBRARY is not None:
        return _LIBRARY
    try:
        import nltk
        from qda_kl.third_party.ifbench import evaluation_lib
    except ModuleNotFoundError as why:
        raise RuntimeError(f"ifbench: {why.name} is not installed; {INSTALL_EVAL}") from why
    missing = []
    for name in NLTK_RESOURCES:
        try:
            nltk.data.find(name)
        except LookupError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"ifbench: NLTK data {missing} not found: run `python -m nltk.downloader "
            "punkt punkt_tab stopwords averaged_perceptron_tagger_eng` into a private directory and point "
            "NLTK_DATA at it"
        )
    _LIBRARY = evaluation_lib
    return _LIBRARY


def _official(value):
    """The official JSONL's ints, from the Hub parquet's float64 columns (recursively)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _official(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_official(item) for item in value]
    return value


def _seed() -> None:
    """Seed what the checkers draw: ``random`` and ``langdetect``'s detector factory."""
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    random.seed(0)


def _check(spec: dict, response: str, lib) -> tuple[list[bool], list[bool], bool]:
    """One answer through upstream's strict then loose test; ``(strict, loose, erred)``."""
    none = [False] * len(spec["instruction_id_list"])
    if not response.strip():
        return none, none, False
    example = lib.InputExample(
        key=spec["key"],
        prompt=spec["prompt"],
        instruction_id_list=list(spec["instruction_id_list"]),
        kwargs=copy.deepcopy(list(spec["kwargs"])),
    )
    _seed()
    try:
        answers = {example.prompt: response}
        strict = lib.test_instruction_following_strict(example, answers)
        loose = lib.test_instruction_following_loose(example, answers)
    except CHECKER_ERRORS:
        return none, none, True
    return (
        list(strict.follow_instruction_list),
        list(loose.follow_instruction_list),
        False,
    )


def weighted_ratio(items: list[tuple[int, int]]) -> float:
    """Upstream's instruction-level mean: passed instructions over all instructions.

    The harness's ``aggregation`` hook receives one ``(passed, total)`` pair per prompt, so prompts with more
    instructions weigh more, as :func:`evaluation_lib.print_report` does.

    Args:
        items: one ``(passed, total)`` pair per prompt.

    Returns:
        The weighted mean, 0.0 when no instruction was checked.
    """
    passed = sum(item[0] for item in items)
    total = sum(item[1] for item in items)
    return passed / total if total else 0.0


def process_results(doc, results):  # noqa: ANN001, ANN201 -- lm-eval's task interface
    """One IFBench prompt: the four metrics, averaged over the prompt's responses."""
    lib = _library()
    spec = {
        "key": str(doc["key"]),
        "prompt": str(doc["prompt"]),
        "instruction_id_list": list(doc["instruction_id_list"]),
        "kwargs": _official(list(doc["kwargs"])),
    }
    responses = [results] if isinstance(results, str) else [str(text) for text in results]
    responses = responses or [""]
    tally = dict.fromkeys(("prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose"), 0)
    errors = 0
    for response in responses:
        strict, loose, erred = _check(spec, response, lib)
        errors += int(erred)
        tally["prompt_strict"] += int(all(strict))
        tally["prompt_loose"] += int(all(loose))
        tally["instruction_strict"] += sum(strict)
        tally["instruction_loose"] += sum(loose)
    n = len(responses)
    checks = n * len(spec["instruction_id_list"])
    return {
        "prompt_level_strict_acc": tally["prompt_strict"] / n,
        "prompt_level_loose_acc": tally["prompt_loose"] / n,
        # (passed, total): the harness weights these by instruction count (:func:`weighted_ratio`).
        "instruction_level_strict_acc": (tally["instruction_strict"], checks),
        "instruction_level_loose_acc": (tally["instruction_loose"], checks),
        "checker_errors": errors / n,
    }
