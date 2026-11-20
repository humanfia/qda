"""Whether calibration rows are written in the model's own chat template (own, plain or foreign).

A row in another model's template calibrates the model on role markers it never sees when served.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Chat-template marker families: name -> markers. A model's template produces a family when its rendered
#: sentinel chat contains any of that family's markers.
FAMILIES: dict[str, tuple[str, ...]] = {
    "chatml": ("<|im_start|>", "<|im_end|>"),
    "llama3": ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
    "gemma": ("<start_of_turn>", "<end_of_turn>"),
    "mistral": ("[INST]", "[/INST]"),
    "deepseek": ("<｜User｜>", "<｜Assistant｜>"),
    "harmony": ("<|start|>", "<|message|>", "<|channel|>"),
    "user-tag": ("<|user|>", "<|assistant|>"),
}

ADVICE = "write the rows with this model's chat template, or as messages rows"
_SENTINEL = [
    {"role": "user", "content": "qda sentinel question"},
    {"role": "assistant", "content": "qda sentinel answer"},
]


@dataclass
class Finding:
    """The classification of a calibration file's rows against one model's chat template.

    Attributes:
        produced: the marker families the model's template produces, sorted.
        own, plain, foreign: row counts of each class.
        first: ``(row index, marker)`` of the first foreign row, None when there is none.
    """

    produced: list[str] = field(default_factory=list)
    own: int = 0
    plain: int = 0
    foreign: int = 0
    first: tuple[int, str] | None = None

    def refusal(self, label: str) -> str | None:
        """The refusal message for the calibration data named `label`, None when no row is foreign."""
        if self.first is None:
            return None
        row, marker = self.first
        family = next(name for name, marks in FAMILIES.items() if marker in marks)
        own = ", ".join(self.produced) or "none of the known families"
        return (
            f"calibration {label}: {self.foreign} of {self.own + self.plain + self.foreign} text rows are written "
            f"in another model's chat template (first: row {row}, counted from 0, marker {marker!r} of {family}; this model's "
            f"template produces {own}); {ADVICE}"
        )

    def record(self) -> dict[str, Any]:
        """What ``build.json`` records (``calib_template``)."""
        return {
            "check": "calibration rows vs the model's chat template",
            "produced": list(self.produced),
            "own": self.own,
            "plain": self.plain,
            "foreign": self.foreign,
        }


def _markers(text: str) -> list[str]:
    return [m for marks in FAMILIES.values() for m in marks if m in text]


def _kind(row: Any) -> str | None:
    """``messages``, ``text`` or None; a token-id row is never text: the engine calibrates on its ids."""
    if not isinstance(row, dict) or "input_ids" in row or "prompt_ids" in row:
        return None
    return "messages" if "messages" in row else "text" if "text" in row else None


def has_markers(rows: list[dict[str, Any]]) -> bool:
    """Whether any text row holds a known chat marker (only then does the check need the model's tokenizer)."""
    return any(_markers(str(r["text"])) for r in rows if _kind(r) == "text")


def produced(tokenizer: Any) -> list[str]:
    """The marker families the tokenizer's chat template produces on a sentinel chat (none without a template)."""
    try:
        text = tokenizer.apply_chat_template(_SENTINEL, tokenize=False)
    except Exception:  # noqa: BLE001 - a template refusing a final assistant turn (Mistral's) takes the continuation
        try:
            text = tokenizer.apply_chat_template(_SENTINEL, tokenize=False, continue_final_message=True)
        except Exception:  # noqa: BLE001 - no chat template: the model produces no family
            return []
    text = str(text)
    return sorted(name for name, marks in FAMILIES.items() if any(m in text for m in marks))


def check_rows(rows: list[dict[str, Any]], tokenizer: Any) -> Finding:
    """Classify every row as own, plain or foreign for the model whose tokenizer is `tokenizer`.

    `tokenizer` may be None when no text row holds a marker (:func:`has_markers`); it is then never used.
    """
    out = Finding()
    if not has_markers(rows):
        out.plain = sum(1 for r in rows if _kind(r) == "text")
        out.own = sum(1 for r in rows if _kind(r) == "messages")
        return out
    if tokenizer is None:
        raise ValueError("the rows hold chat markers: the check needs the model's tokenizer")
    out.produced = produced(tokenizer)
    own = {m for name in out.produced for m in FAMILIES[name]}
    for i, r in enumerate(rows):
        kind = _kind(r)
        if kind == "messages":
            out.own += 1
            continue
        if kind is None:
            continue
        found = _markers(str(r["text"]))
        alien = [m for m in found if m not in own]
        if alien:
            out.foreign += 1
            if out.first is None:
                out.first = (i, alien[0])
        elif found:
            out.own += 1
        else:
            out.plain += 1
    return out


# ---------------------------------------------------------------------------------- the build's data lint (per file)

#: The thinking markers the lint counts by (R1, Qwen3 and the models that copied their format).
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_EMPTY_THINK = re.compile(re.escape(THINK_OPEN) + r"\s*" + re.escape(THINK_CLOSE))
#: The defect counts of :class:`Lint`, in the order the build log prints them, with their log words.
DEFECTS = {
    "empty_think": "empty think",
    "unclosed_think": "unclosed think",
    "no_eos": "no end-of-turn/EOS at the end",
    "no_bos": "no BOS at the start",
    "appended_eos": "appended_eos flag",
}


@dataclass
class Expect:
    """What the model's own chat format puts around a chat: read once from its tokenizer.

    Attributes:
        produced: the marker families its template produces (:func:`produced`).
        bos: the BOS a rendered chat starts with; None when the template starts with none.
        ends: the strings a finished chat ends with: the EOS, and the end of a rendered assistant turn.
    """

    produced: list[str]
    bos: str | None
    ends: tuple[str, ...]


def expect(tokenizer: Any) -> Expect:
    """The model's own chat format, from its tokenizer's chat template and special tokens."""
    try:
        text = str(tokenizer.apply_chat_template(_SENTINEL, tokenize=False))
    except Exception:  # noqa: BLE001 - no chat template, or one refusing a final assistant turn: no turn end
        text = ""
    bos = getattr(tokenizer, "bos_token", None)
    eos = getattr(tokenizer, "eos_token", None)
    answer = _SENTINEL[-1]["content"]
    tail = text[text.rfind(answer) + len(answer) :].strip() if answer in text else ""
    ends = tuple(dict.fromkeys(e for e in (eos, tail) if isinstance(e, str) and e))
    return Expect(
        produced(tokenizer), bos if isinstance(bos, str) and bos and text.startswith(bos) else None, ends
    )


@dataclass
class Lint(Finding):
    """:class:`Finding` of one data file, with token-id rows decoded and rendered rows' `DEFECTS` counted.

    Attributes:
        tokens: how many rows are token-id rows.
        bad_ids: token-id rows whose ids are not the tokenizer's; left to the engine's own id checks.
        empty_think, unclosed_think, no_eos, no_bos, appended_eos: rendered rows with that defect (``appended_eos``:
            the row's own flag that its EOS was appended after a cut).
    """

    file: str = ""
    rows: int = 0
    tokens: int = 0
    empty_think: int = 0
    unclosed_think: int = 0
    no_eos: int = 0
    no_bos: int = 0
    appended_eos: int = 0
    bad_ids: int = 0

    def refusal(self, label: str) -> str | None:
        """The refusal message for the file named `label`, None when no row is foreign."""
        why = super().refusal(label)
        if why is None or not self.tokens:
            return why
        return why.replace(" text rows ", " rows (token ids decoded) ", 1)

    def lint_record(self) -> dict[str, Any]:
        """What ``build.json`` records of this file (``data_lint``)."""
        out: dict[str, Any] = {
            "file": self.file,
            "rows": self.rows,
            "token_rows": self.tokens,
            "bad_ids": self.bad_ids,
        }
        out |= {"own": self.own, "plain": self.plain, "foreign": self.foreign}
        return out | {k: getattr(self, k) for k in DEFECTS}


def lint_rows(rows: list[dict[str, Any]], tokenizer: Any, file: str) -> Lint:
    """Classify and count every row of the data file `file` for the model whose tokenizer is `tokenizer`.

    Text rows are classified as :func:`check_rows` does; token-id rows on their decoded ids. `tokenizer` may be
    None only when no row holds token ids or a chat marker.
    """
    out = Lint(file=file)
    known: Expect | None = None
    own: set[str] = set()
    for i, r in enumerate(rows):
        out.rows += 1
        if not isinstance(r, dict):
            continue
        if "input_ids" in r or "prompt_ids" in r:
            if tokenizer is None:
                raise ValueError("the rows hold token ids: the lint needs the model's tokenizer")
            out.tokens += 1
            try:
                ids = _ids(r, tokenizer)  # checked first: a fast decode drops an unknown id silently
            except LookupError:
                out.bad_ids += (
                    1  # not decoded; the engine's own id checks refuse ids beyond the model's vocabulary
                )
                continue
            text = str(tokenizer.decode(ids, skip_special_tokens=False))
        elif (kind := _kind(r)) == "messages":
            out.own += 1
            continue
        elif kind == "text":
            text = str(r["text"])
        else:
            continue
        found = _markers(text)
        if not found:
            out.plain += 1
            continue
        if known is None:
            if tokenizer is None:
                raise ValueError("the rows hold chat markers: the lint needs the model's tokenizer")
            known = expect(tokenizer)
            out.produced = known.produced
            own = {m for name in known.produced for m in FAMILIES[name]}
        alien = [m for m in found if m not in own]
        if alien:
            out.foreign += 1
            if out.first is None:
                out.first = (i, alien[0])
        else:
            out.own += 1
        out.empty_think += bool(_EMPTY_THINK.search(text))
        out.unclosed_think += text.rfind(THINK_OPEN) > text.rfind(THINK_CLOSE)
        out.no_eos += bool(known.ends) and not text.rstrip().endswith(known.ends)
        out.no_bos += known.bos is not None and not text.startswith(known.bos)
        out.appended_eos += r.get("appended_eos") in (True, 1, "True", "true")
    return out


def lint_receipt(records: list[dict[str, Any]]) -> dict[str, Any]:
    """build.json's ``data_lint``: the records of every data file the build's plan read, or that none was read."""
    if not records:
        return {"check": "not run (no data rows read at plan time)"}
    return {"check": "data rows vs the model's chat format", "files": list(records)}


def lint_block(records: list[dict[str, Any]]) -> str:
    """The build log's one block of the data lint: one line per file read."""
    lines = ["data lint (defects counted on rows with chat markup; plain and messages rows are not):"]
    for r in records:
        if "rows" not in r:  # not linted: the record says why
            lines.append(f"  {r['file']}: {r['check']}")
            continue
        counts = ", ".join(f"{DEFECTS[k]} {r[k]}" for k in DEFECTS)
        if r.get("bad_ids"):
            counts += f", ids not the tokenizer's (not decoded) {r['bad_ids']}"
        lines.append(
            f"  {r['file']}: {r['rows']} rows ({r['token_rows']} token-id): own {r['own']}, plain {r['plain']}, "
            f"foreign {r['foreign']}; {counts}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- data pools shipped to an agent

#: Token-id fields of a row; a row with any of them is checked on its decoded ids, whatever else it holds.
TOKEN_KEYS = ("input_ids", "prompt_ids", "completion_ids")
#: How many defective lines a refusal names.
SHOWN = 3
POOL_ADVICE = "a pool shipped to the agent holds un-rendered text: messages rows, prompt fields or plain text"


@dataclass
class FileFinding:
    """One JSONL data pool's rows classified; a row with any family's chat marker is ``rendered``.

    Attributes:
        messages: ``messages`` rows (un-rendered: the agent renders them).
        rendered, undecodable: row counts of each defect.
        other: lines that are no row of any kind (not a JSON object, or no text, ids or messages).
        rendered_at: ``(line, marker)`` of the first :data:`SHOWN` rendered rows, lines counted from 1.
        undecodable_at: the lines of the first :data:`SHOWN` undecodable rows.
    """

    path: str
    plain: int = 0
    messages: int = 0
    rendered: int = 0
    undecodable: int = 0
    other: int = 0
    rendered_at: list[tuple[int, str]] = field(default_factory=list)
    undecodable_at: list[int] = field(default_factory=list)

    @property
    def rows(self) -> int:
        """The rows classified (every class but `other`)."""
        return self.plain + self.messages + self.rendered + self.undecodable

    def refusal(self, label: str) -> str | None:
        """The refusal message for this file, shipped as `label`; None when every row passes."""
        why = []
        if self.rendered:
            line, marker = self.rendered_at[0]
            family = next(name for name, marks in FAMILIES.items() if marker in marks)
            lines = ", ".join(str(n) for n, _ in self.rendered_at)
            why.append(
                f"{self.rendered} rows hold pre-rendered chat markup (first at lines {lines}; "
                f"line {line} holds {marker!r} of {family})"
            )
        if self.undecodable:
            lines = ", ".join(str(n) for n in self.undecodable_at)
            why.append(
                f"{self.undecodable} token-id rows are not ids of this model's tokenizer (first at lines {lines})"
            )
        if not why:
            return None
        return f"{label} {self.path}: of {self.rows} rows, " + "; ".join(why) + f"; {POOL_ADVICE}"


def _ids(row: dict[str, Any], tokenizer: Any) -> list[int]:
    """The token ids of `row`, each checked to be an id of `tokenizer` (a fast decode drops unknown ids silently).

    Raises:
        LookupError: a token-id field that is not a list of integers, or an id outside the tokenizer's vocabulary.
    """
    ids: list[int] = []
    for key in TOKEN_KEYS:
        if key not in row:
            continue
        value = row[key]
        if not isinstance(value, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in value
        ):
            raise LookupError(f"{key} is not a list of integers")
        ids += value
    size = (
        len(tokenizer) if hasattr(tokenizer, "__len__") else None
    )  # a tokenizer without a size: ids unchecked
    if size is not None and (bad := [i for i in ids if not 0 <= i < size]):
        raise LookupError(f"ids {bad[:3]} are outside the vocabulary of {size}")
    return ids


def _row_text(row: dict[str, Any], tokenizer: Callable[[], Any]) -> str | None:
    """The text the markup check reads in `row`, None for a row with none (``messages``, or no text field).

    Raises:
        LookupError: a token-id row the tokenizer cannot decode.
    """
    if any(k in row for k in TOKEN_KEYS):
        return str(tokenizer().decode(_ids(row, tokenizer()), skip_special_tokens=False))
    if "messages" in row:
        return None
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("prompt"), str):
        completion = row.get("completion")
        return row["prompt"] + (completion if isinstance(completion, str) else "")
    return None


def check_file(path: str | Path, tokenizer: Callable[[], Any]) -> FileFinding:
    """Classify every row of the JSONL pool at `path`; `tokenizer()` is called only for a token-id row.

    Raises:
        OSError, UnicodeDecodeError: the file cannot be read as text.
    """
    out = FileFinding(str(path))
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out.other += 1
                continue
            if not isinstance(row, dict):
                out.other += 1
                continue
            try:
                text = _row_text(row, tokenizer)
            except LookupError:
                out.undecodable += 1
                if len(out.undecodable_at) < SHOWN:
                    out.undecodable_at.append(n)
                continue
            if text is None:
                if "messages" in row:
                    out.messages += 1
                else:
                    out.other += 1
                continue
            if not (found := _markers(text)):
                out.plain += 1
                continue
            out.rendered += 1
            if len(out.rendered_at) < SHOWN:
                out.rendered_at.append((n, found[0]))
    return out
