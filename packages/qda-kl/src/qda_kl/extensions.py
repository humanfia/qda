"""Register the CLI commands this package adds to QDA."""

from collections.abc import Callable
from typing import Any


def register() -> None:
    """Add lazy cross-package registrations before the first registry read."""
    from qda_quant.cli import COMMANDS
    from qda_quant.gguf import read
    from qda_quant.intake.corpus import CORPORA

    COMMANDS.register("eval", "qda_kl.eval.cli:add_parser")
    COMMANDS.register("traces", "qda_kl.traces:add_parser")
    CORPORA.register("kl_holdout", "qda_kl.holdout:KLHoldoutCorpus")
    read.DEFAULT_SCORER = default_scorer


def default_scorer() -> Callable[..., Any]:
    """Resolve the sensitivity scorer without importing its engine during registration."""
    from .score import score

    return score
