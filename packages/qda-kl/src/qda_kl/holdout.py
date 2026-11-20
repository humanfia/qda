"""The prompts a KL reading defaults to: the KL v2 corpus's holdout split.

`qda-kl kl` reads it when given no ``--prompts``, and the ``kl`` eval suite reads it when its config names no
corpus, so both score the same prompts. The split is the corpus cache's (:func:`qda_kl.corpus.cached_splits`),
built once on the first read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from qda_quant.intake.corpus import Corpus, read_rows

from qda_kl import corpus

__all__ = ["KLHoldoutCorpus", "default_prompts"]


def default_prompts() -> Path:
    """The KL v2 corpus's holdout split, from the corpus cache."""
    return corpus.cached_splits()["holdout"]


class KLHoldoutCorpus(Corpus):
    """The KL v2 corpus's holdout split (``qda.corpora.kl_holdout``), the prompts a reading defaults to.

    The file is the corpus cache's (:func:`default_prompts`), built on the first read. Its rows are already the
    holdout, so every one is on the eval side.
    """

    name: ClassVar[str] = "kl_holdout"
    holdout: ClassVar[float] = 1.0

    def rows(self) -> list[dict[str, Any]]:
        """The holdout split's rows, in file order."""
        return read_rows(default_prompts(), f"corpus {self.name}")
