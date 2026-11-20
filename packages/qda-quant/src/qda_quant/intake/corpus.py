"""Evaluation and calibration corpora: prompt sets registered in ``qda.corpora`` and chosen by name.

`split` gives evaluation and calibration disjoint rows, placed by ``sha256(id)`` so the split never moves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ..registry import Registry
from ..step import Refused

#: Corpora by key (entry-point group ``qda.corpora``).
CORPORA = Registry(
    "corpora",
    {
        "file": "qda_quant.intake.corpus:FileCorpus",
        "reasoning": "qda_quant.intake.corpus:ReasoningCorpus",
    },
    group="qda.corpora",
)

#: The content keys a row may carry, exactly one of them.
_CONTENT = ("messages", "prompt", "text")

#: The ``split`` values a row may carry, and whether each is on the evaluation side.
_SIDES = {"eval": True, "holdout": True, "calib": False}

#: sha256 of the paper's bundled reference prompt file, ``data/reasoning-kl3.jsonl`` (the 48 kl3 prompts).
REASONING_SHA256 = "535f7848b9227880482104dba6c1f50d0dc371b0be0ea23755795b7b91359ea2"

#: Where the bundled paper reference prompts live, when the file is present.
REASONING_PATH = Path(__file__).with_name("data") / "reasoning-kl3.jsonl"

#: The one refusal a corpus that is not bundled gives.
_NOT_BUNDLED = "the paper's reasoning corpus is not bundled yet; use corpus: {file: {path: ...}}"


class Corpus(BaseModel):
    """Base class of a corpus: its settings and its rows.

    Attributes:
        holdout: the share of rows only evaluation may read, in [0, 1]; fixed per class, not per config, so two
            workflows can never split one corpus differently.
    """

    model_config = ConfigDict(extra="forbid")

    name: ClassVar[str] = ""
    holdout: ClassVar[float] = 0.5
    #: Settings that hold a local path. A config file resolves them relative to its own directory
    #: (`qda_quant.recipe.resolve_paths`).
    paths: ClassVar[tuple[str, ...]] = ()

    def rows(self) -> list[dict[str, Any]]:
        """The corpus's rows, in the order the source gives them."""
        raise NotImplementedError

    def side(self, row: dict[str, Any]) -> Literal["eval", "calib"]:
        """The side `row` belongs to: its own ``split`` tag, else its ``sha256(id)`` against :attr:`holdout`."""
        return "eval" if _on_eval_side(row, self.holdout) else "calib"


def split(corpus: Corpus, part: Literal["eval", "calib"]) -> list[dict[str, Any]]:
    """The rows of `corpus` that `part` (``"eval"`` or ``"calib"``) may read, in order; sides are disjoint.

    Raises:
        Refused: `part` is neither name, a row has no ``id`` (a row that cannot be placed on a side), or its
            ``split`` is not one of :data:`_SIDES`.
    """
    if part not in ("eval", "calib"):
        raise Refused(f"corpus: part must be 'eval' or 'calib', not {part!r}")
    rows = corpus.rows()
    for n, row in enumerate(rows):
        _check_row(row, f"corpus {corpus.name} row {n}")
    return [row for row in rows if corpus.side(row) == part]


def _on_eval_side(row: dict[str, Any], holdout: float) -> bool:
    """Whether `row` is on the evaluation side: its own ``split`` tag, else its id's hash share."""
    if "split" in row:
        return _SIDES[row["split"]]
    return _eval_side(row["id"], holdout)


def _eval_side(ident: Any, holdout: float) -> bool:
    """Whether the row with `ident` is on the eval side of a corpus with `holdout`."""
    digest = hashlib.sha256(str(ident).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") < holdout * (1 << 256)


def _check_row(row: Any, where: str) -> None:
    """Refuse a row without a usable ``id`` or with not exactly one of :data:`_CONTENT`."""
    if not isinstance(row, dict):
        raise Refused(f"{where}: expected a JSON object, got {type(row).__name__}")
    if not isinstance(row.get("id"), (str, int)) or str(row.get("id")) == "":
        raise Refused(f'{where}: needs a non-empty "id"')
    have = [k for k in _CONTENT if k in row]
    if len(have) != 1:
        raise Refused(f"{where}: needs exactly one of {list(_CONTENT)}, got {have or 'none'}")
    if "split" in row and row["split"] not in _SIDES:
        raise Refused(f"{where}: split must be one of {sorted(_SIDES)}, got {row['split']!r}")


def read_rows(path: str | Path, where: str) -> list[dict[str, Any]]:
    """The JSONL rows at `path`, checked one by one; `where` names the file in a refusal."""
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise Refused(f"{where}: cannot read {p}: {e}") from e
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise Refused(f"{where}:{n}: not JSON: {e}") from e
        _check_row(row, f"{where}:{n}")
        out.append(row)
    return out


def parse_corpus(d: dict[str, Any]) -> Corpus:
    """The corpus a config's ``corpus`` mapping names: exactly one key, the corpus's, with its settings.

    Raises:
        Refused: not one key, or an unknown key (the message lists the known ones). A key beside the
            corpus's name (``corpus.path``) is named as that dotted key.
    """
    named = [k for k in d if k in CORPORA] if isinstance(d, dict) else []
    if len(named) == 1 and len(d) > 1:
        raise Refused(
            "; ".join(
                f"corpus.{k}: unknown key; a corpus's settings go under its key (corpus.{named[0]}.{k})"
                for k in sorted(set(d) - set(named))
            )
        )
    if not isinstance(d, dict) or len(d) != 1:
        raise Refused(f"corpus: expected one corpus {{key: settings}}, known: {sorted(CORPORA)}; got {d!r}")
    ((key, cfg),) = d.items()
    if key not in CORPORA:
        raise Refused(f"corpus.{key}: unknown corpus; known: {sorted(CORPORA)}")
    return CORPORA[key](**(cfg or {}))


class FileCorpus(Corpus):
    """A corpus file on disk: JSONL rows of ``{"id", "domain", "prompt" | "messages" | "text"}``.

    A row may carry ``split`` (``"eval"``/``"holdout"`` or ``"calib"``) to pin its side.

    Attributes:
        path: the JSONL file.
        split: how the file is placed on the sides: ``"row"`` (default) places each row by its own ``split``
            tag, else its ``sha256(id)`` (see :class:`Corpus`); ``"eval"`` or ``"calib"`` takes the whole file
            as that side, a file that is already one split.
    """

    name: ClassVar[str] = "file"
    paths: ClassVar[tuple[str, ...]] = ("path",)
    path: str
    split: Literal["row", "eval", "calib"] = "row"

    def rows(self) -> list[dict[str, Any]]:
        """The file's rows, in file order."""
        return read_rows(self.path, f"corpus file {self.path}")

    def side(self, row: dict[str, Any]) -> Literal["eval", "calib"]:
        """The whole file's side when the corpus names one, else the row's own tag or hash."""
        if self.split == "row":
            return super().side(row)
        return self.split


class ReasoningCorpus(Corpus):
    """The paper's reference prompts: 48 prompts from AIME 2000--2020, SuperGPQA, OpenThoughts and OpenHermes.

    Evaluation-only (``holdout`` 1.0), since the paper marks them "do not calibrate"; :meth:`rows` refuses
    until the pinned file (:data:`REASONING_SHA256`) is bundled.
    """

    name: ClassVar[str] = "reasoning"
    holdout: ClassVar[float] = 1.0

    def rows(self) -> list[dict[str, Any]]:
        """The bundled prompts, after checking the file's bytes against the pinned sha256.

        Raises:
            Refused: the file is missing or is not the pinned one.
        """
        try:
            data = REASONING_PATH.read_bytes()
        except OSError as e:
            raise Refused(_NOT_BUNDLED) from e
        if hashlib.sha256(data).hexdigest() != REASONING_SHA256:
            raise Refused(_NOT_BUNDLED)
        return read_rows(REASONING_PATH, "corpus reasoning")
