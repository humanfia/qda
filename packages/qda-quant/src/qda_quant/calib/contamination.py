"""Drop calibration rows whose PROMPT shares text with an evaluation reference's prompts.

A calibration row is dropped when it matches a reference prompt by ``id``, by normalized-text hash, or by sharing
a ``WINDOW``-word window under either normalization: the contamination guard's (lower-cased, whitespace-split
words, plus ``CHARS``-character windows for text written without spaces) and the punctuation-insensitive one
(lower-cased ASCII alphanumerics). A row hit under either normalization is dropped, as the post-launch audit
flags it, so a file that passes here also passes that audit.

Only PROMPTS are compared: a ``prompt`` row's prompt, a chat row's user turns, a text row's text. Generated
answer bodies are never indexed (the reference side holds prompts only) and never read from a chat row, so
self-generated calibration (``calib.selfgen``), whose answers may echo the model, is kept unless its prompt
itself matches. That is deliberate: an answer that resembles a reference answer is not contamination. A
``render="text"`` selfgen row (prompt and answer in one string) is never screened by this row rule either:
``qda_quant.calib.sources.SelfgenSource`` screens the prompt file it generated from and drops each
contaminated prompt's answer rows with it.

The reference side is an index of hashes only (``PromptIndex``): the prompts never need to reach the build, so a
search's hidden KL reference can be indexed on the host and only the index passed to the agent's ``qda build``.
``python -m qda_quant.calib.contamination --reference PROMPTS --out INDEX`` writes one; ``--reference`` takes a
prompt JSONL (``{id, prompt}`` or ``{id, text}`` rows) or a ``qda-kl ref`` directory (its manifest's prompt file).

Dropped rows are recorded (count and ids) in ``build.json``; they are never a refusal of the build by the judge.

A window match is not by itself contamination, so a shared window is exempt only when ALL of it is structural
scaffolding: nothing but numbers (an example's input sequence), or an input/output window whose every word is
I/O format vocabulary. A window that merely CONTAINS an I/O clause (``the input contains n integers``) beside
substantive task words is still the task, so it drops. Frequency buys nothing: a window that recurs in several
reference prompts is still reference content, and recurrence in the calibration batch cannot prove a row clean
(the batch may hold the same copied passage with different prefixes), so neither makes a window generic. A row
whose every shared window is exempt is KEPT, and ``build.json`` records it as exempted with the rule and id. An
``id`` or exact-text hit is never exempted: those are the prompt itself, not a window of it.

Known limit: a reference prompt shorter than ``WINDOW`` words has no word window, so a copy of it embedded in a
LONGER calibration prompt meets neither the window arms nor the normalized-text hash (which matches only the same
text in full). The threshold stays at ``WINDOW``: lowering it would flag ordinary short spans everywhere. A
reference prompt written in a no-space script is still covered by the ``CHARS``-character arm at any length.

A reference may hold only the subset of a corpus it was scored on (a sealed eval or holdout split).
Its own selection record beside it names the parent corpus with its row count and digest, and the index is then
built from that corpus's full split instead of the subset (:func:`parent_corpus`). A record whose corpus is
missing, changed, or not this reference's corpus is a refusal, never a silent fall back to the subset: the builds
would then be filtered against a fraction of the hidden reference. The index records the corpus path, its file
digest and selected-row digest per role, and its ``kind``/``window``/``chars`` fields name the scanner the
exclusions were computed with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..step import Refused

__all__ = [
    "CHARS",
    "KIND",
    "NOSPACE",
    "NOSPACE_SHARE",
    "SELECTION_RECORDS",
    "WINDOW",
    "PromptIndex",
    "filter_rows",
    "parent_corpus",
    "prompt_texts",
    "windows",
]

#: Words in a window: the contamination guard's window (the audit imports the same number).
WINDOW = 20
#: Character windows added for text written without spaces between words. 32 characters cover every MGSM
#: ja/zh/th item with no false hit on 11,892 aya rows in those languages; 24 characters did hit two Thai rows.
CHARS = 32
#: Unicode ranges of scripts normally written without spaces: Thai, Hiragana, Katakana, Han.
NOSPACE = (
    (0x0E00, 0x0E7F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
)
#: A text takes character windows when this share of its non-space characters is from such a script.
NOSPACE_SHARE = 0.2

_WORDS = re.compile(r"[a-z0-9]+")
#: The index file's name for itself and the layout, so a stale index is refused rather than read wrongly.
KIND = "qda-calibration-contamination"

#: Exemptions are structural only. A shared window that recurs in several reference prompts is still reference
#: content, and the calibration batch cannot certify a row clean (the same copied passage may arrive in many
#: rows behind different prefixes), so no frequency test can make a window generic.

#: The prompt's section headings, by kind: a shared window in an input/output/example section is a problem's
#: I/O format, not its statement. Any other heading ends that section.
_SECTIONS = {
    "input": "format",
    "inputs": "format",
    "output": "format",
    "outputs": "format",
    "example": "format",
    "examples": "format",
    "sample": "format",
    "note": "other",
    "notes": "other",
    "constraint": "other",
    "constraints": "other",
    "problem": "other",
    "statement": "other",
}


def windows(text: str, n: int = WINDOW) -> set[str]:
    """sha256 of every `n`-word window of `text` (lower case, whitespace-split) and of its character windows.

    Every text gets its word windows. A text written without spaces between words gets no word
    windows at all, so it also yields every `CHARS`-character window of its whitespace-stripped, NFKC-normalized
    lower-case body, and a copy of a test item is found even though it has no 20-word window. In a mixed text,
    each whitespace-separated run of no-space script yields its own windows, so English padding in the same
    string does not hide the run.
    """
    w = text.lower().split()
    out = {hashlib.sha256(" ".join(w[i : i + n]).encode()).hexdigest() for i in range(len(w) - n + 1)}
    return out | _char_windows(text)


def _char_windows(text: str) -> set[str]:
    """Every `CHARS`-character window over the no-space bodies of a text (none otherwise).

    A text mostly written in a no-space script gets the windows of its whole whitespace-stripped body. A mixed
    text (foreign text inside an English instruction) stays below that share, so every whitespace-separated
    run that contains a no-space character gets the windows of its own body instead: the text around the run
    cannot hide it.
    """
    body = _body(text)
    if not body:
        return set()
    if _unspaced(body) >= NOSPACE_SHARE * len(body):
        return _windows(body)
    out: set[str] = set()
    for run in text.split():
        run = _body(run)
        if _unspaced(run):
            out |= _windows(run)
    return out


def _body(text: str) -> str:
    """The text NFKC-normalized and lower-cased, with whitespace removed."""
    return "".join(unicodedata.normalize("NFKC", text).lower().split())


def _unspaced(body: str) -> int:
    """How many characters of `body` belong to a script normally written without spaces."""
    return sum(1 for c in body if any(lo <= ord(c) <= hi for lo, hi in NOSPACE))


def _windows(body: str) -> set[str]:
    """The sha256 of every `CHARS`-character window of `body`."""
    return {hashlib.sha256(body[i : i + CHARS].encode()).hexdigest() for i in range(len(body) - CHARS + 1)}


def regex_windows(text: str, n: int = WINDOW) -> set[str]:
    """sha256 of the punctuation-insensitive ``n``-word windows (lower-cased ASCII alphanumerics).

    Hashed like :func:`windows`, so an index of them never holds a prompt passage.
    """
    words = _WORDS.findall(text.lower())
    return {_sha(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def _sha(text: str) -> str:
    """sha256 hex of a UTF-8 text, or of the normalized text for a text hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_windows(text: str, n: int = WINDOW) -> dict[str, str]:
    """The guard normalization's word windows, by sha256, with the window text (the character windows excluded)."""
    w = text.lower().split()
    return {_sha(" ".join(w[i : i + n])): " ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def _regex_texts(text: str, n: int = WINDOW) -> dict[str, str]:
    """The regex normalization's word windows, by sha256, with the window text."""
    words = _WORDS.findall(text.lower())
    return {_sha(" ".join(words[i : i + n])): " ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def exempt_rule(text: str, source: str | None = None) -> str | None:
    """The structural exemption a shared window falls under, or None when it may carry a problem.

    Two shapes cannot distinguish a copied prompt from a clean one and are exempt on their own: a window of
    nothing but numbers (an example's input sequence), and an input/output window inside a prompt's
    input/output/example section whose EVERY word is I/O format vocabulary. A window that merely CONTAINS an
    I/O clause (``the input contains n integers``) beside substantive task words is still the task, so it
    stays a hit. Recurrence exempts nothing: a window shared by other reference prompts or other calibration
    rows is still content, so this function never looks at another prompt.
    """
    words = _WORDS.findall(text.lower())
    if not words:
        return None
    if all(w.isdigit() for w in words):
        return "numeric"
    if source is not None and _in_io_section(words, source) and _io_wording(words):
        return "code-io"
    return None


#: The wording a problem's input/output format uses ("contains n integers", "space-separated integers",
#: bounds "(1 <= a_i <= 10^9)", "the i-th of them contains ..."). A task instruction has none of these.
_IO_WORDING = (
    re.compile(r"\bcontains?\b.{0,40}\b(?:integers?|integer|numbers?|number|strings?|string|characters?)\b"),
    re.compile(r"\bwhere\b.{0,80}\bis the\b"),
    re.compile(r"\b(?:space|whitespace)[- ]separated\b"),
    re.compile(r"\bwhere\b.{0,60}\bis the\b.{0,30}\b(?:element|value|character)\b"),
    re.compile(r"\d+\s*(?:<=|\u2264|=<|<)\s*[a-z_][a-z_0-9]*\s*(?:<=|\u2264|=<|<)\s*\d+"),
)

#: The closed vocabulary of a problem's I/O format: function words, counting and labeling words, and the
#: nouns a statement uses for its input. A window with any other word carries content and is not format.
_FORMAT_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "as",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "not",
        "no",
        "each",
        "every",
        "all",
        "any",
        "some",
        "such",
        "then",
        "than",
        "there",
        "here",
        "where",
        "when",
        "if",
        "which",
        "who",
        "whose",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "first",
        "second",
        "third",
        "fourth",
        "last",
        "next",
        "following",
        "given",
        "line",
        "lines",
        "test",
        "case",
        "cases",
        "example",
        "examples",
        "sample",
        "samples",
        "input",
        "inputs",
        "output",
        "outputs",
        "number",
        "numbers",
        "integer",
        "integers",
        "digit",
        "digits",
        "character",
        "characters",
        "string",
        "strings",
        "word",
        "words",
        "array",
        "arrays",
        "sequence",
        "sequences",
        "list",
        "lists",
        "element",
        "elements",
        "value",
        "values",
        "item",
        "items",
        "length",
        "lengths",
        "size",
        "sizes",
        "index",
        "indices",
        "position",
        "positions",
        "space",
        "spaces",
        "separated",
        "whitespace",
        "contains",
        "contain",
        "holds",
        "hold",
        "consists",
        "consist",
        "format",
        "formats",
        "left",
        "right",
        "above",
        "below",
        "same",
        "different",
        "total",
        "sum",
        "count",
        "independent",
        "others",
    }
)


def _io_wording(words: list[str]) -> bool:
    """Whether the WHOLE window reads as input/output format, not as a task carrying an I/O clause.

    Every word must be format vocabulary (or a number or a one/two-letter variable), and at least one I/O
    clause must match the window itself. The task ``... using uranium lead measurements the input contains n
    integers`` has substantive words (uranium, measurements, ...), so its window is not format.
    """
    if not any(p.search(" ".join(words)) for p in _IO_WORDING):
        return False
    return all(w.isdigit() or len(w) <= 2 or w in _FORMAT_WORDS for w in words)


def _in_io_section(words: list[str], source: str) -> bool:
    """Whether `words` occur in `source` after an input/output/example heading and before the next heading.

    A heading is a line of its own holding exactly one heading word ("Input", "### Output", "Examples:") -
    the shape every problem statement uses. A heading word inside a sentence is not one, so a problem's
    statement sentence is never read as an I/O section.
    """
    tokens = _WORDS.findall(source.lower())
    at = _find(tokens, words)
    if at is None:
        return False
    kind, seen = None, 0
    for line in source.splitlines():
        line_words = _WORDS.findall(line.lower())
        if len(line_words) == 1 and line_words[0] in _SECTIONS and seen <= at:
            kind = _SECTIONS[line_words[0]]
        seen += len(line_words)
    return kind == "format"


def _find(tokens: list[str], words: list[str]) -> int | None:
    """Where the token sequence `words` starts in `tokens` (None: not there)."""
    for i in range(len(tokens) - len(words) + 1):
        if tokens[i : i + len(words)] == words:
            return i
    return None


def _exempt_rules(shared: dict[str, tuple[str, str] | None]) -> frozenset[str]:
    """The structural rules that cover every shared window (empty when any window may carry a problem)."""
    rules: set[str] = set()
    for w in shared.values():
        rule = None if w is None else exempt_rule(*w)
        if rule is None:
            return frozenset()
        rules.add(rule)
    return frozenset(rules)


#: A chat template's user turns, as they appear in a rendered ``text`` row (Qwen, ChatML, Llama).
_USER_REGIONS = (
    re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S),
    re.compile(r"<\|user\|>(.*?)<\|end\|>", re.S),
    re.compile(r"\[INST\](.*?)\[/INST\]", re.S),
)


def _prompt_regions(text: str) -> list[str]:
    """The user turns of a rendered chat `text` row; the whole text when it is no rendered chat.

    A rendered selfgen row holds its prompt and its generated answer in one string; reading the whole string
    as the prompt would drop it for an answer that echoes a reference, which this filter never does.
    """
    found = [m.group(1) for pattern in _USER_REGIONS for m in pattern.finditer(text)]
    return found or [text]


def _as_text(value: Any) -> str:
    """A field's text, with JSON null and a missing field both reading as no text (never ``"None"``)."""
    return "" if value is None else str(value)


def prompt_texts(row: dict[str, Any], tokenizer: Any = None) -> list[str]:
    """Every PROMPT-side text of a calibration row: never an answer body.

    A ``prompt`` row gives its prompt; a chat gives its user turns only (assistant turns are answer bodies); a
    plain ``text`` row gives its text, which is all the prompt it has. A row may carry more than one prompt-side
    field - a trace's ``prompt`` string beside the ``prompt_ids`` a step actually trains on, or an ``input_ids``
    file with a ``text`` column - and every one of them is returned: a clean string beside copied token ids
    must not hide the tokens the build reads. Token-id fields are decoded with `tokenizer` (a trace's
    ``prompt_ids`` only: the completion is an answer body); with no tokenizer they give no text. A rendered
    selfgen ``text`` row (prompt + answer in one string) is not one of these: its source screens the prompt file
    before generation instead (see the module docstring).
    """
    texts: list[str] = []
    if "prompt" in row:
        texts.append(_as_text(row["prompt"]))
    if "messages" in row:
        texts += [
            _as_text(m.get("content"))
            for m in row["messages"]
            if isinstance(m, dict) and m.get("role") == "user"
        ]
    if "text" in row:
        texts += _prompt_regions(_as_text(row["text"]))
    if tokenizer is not None:
        for key in ("prompt_ids", "input_ids"):
            if key in row:
                texts.append(tokenizer.decode(list(row[key]), skip_special_tokens=True))
    return texts


#: The roles an index's source sets carry: the reference's evaluation set, and the set it holds out.
ROLES = ("eval", "holdout")

#: The bases a source's prompt path resolved from: the path itself, a manifest's absolute record, the
#: reference directory, its parent, its group directory (a shared corpus beside a reference root), or the
#: reference's own selection record naming the full corpus a subset manifest was drawn from.
BASES = ("file", "absolute", "reference", "parent", "group", "selection")

#: The files a reference's own record of its selection is written to, beside the reference root or its group.
SELECTION_RECORDS = ("selection.json", "ref-record.json")


@dataclass(frozen=True)
class Source:
    """One prompt set an index was built from: its role, file digest, selected-row digest and path base.

    The digests are what a host compares against a prebuilt index: metadata alone never proves the content an
    index holds. ``rows_sha256`` covers exactly the rows this source contributed, so two roles partitioned out
    of one prompt file still have distinct row digests while sharing ``file_sha256``. ``path`` is the prompt
    file actually read and ``base`` says where its recorded name resolved from.
    """

    role: str
    path: str
    file_sha256: str
    rows: int
    rows_sha256: str
    base: str = "file"

    def to_dict(self) -> dict[str, Any]:
        """The source as JSON data."""
        return {
            "role": self.role,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "rows": self.rows,
            "rows_sha256": self.rows_sha256,
            "base": self.base,
        }

    @classmethod
    def from_dict(cls, d: Any) -> Source:
        """One written source.

        Raises:
            Refused: not an object, an unknown role, a missing digest, or a non-positive row count.
        """
        if not isinstance(d, dict):
            raise Refused("contamination index: a source is not a JSON object")
        unknown = set(d) - _SOURCE_FIELDS
        if unknown:
            raise Refused(f"contamination index: a source has unexpected field(s) {sorted(unknown)}")
        role = d.get("role")
        if not isinstance(role, str) or role not in ROLES:
            raise Refused(f"contamination index: a source role is {role!r}; expected one of {list(ROLES)}")
        path = d.get("path")
        if not isinstance(path, str) or not path:
            raise Refused("contamination index: a source has no path")
        file_sha, rows_sha = d.get("file_sha256"), d.get("rows_sha256")
        if not isinstance(file_sha, str) or not _DIGEST.fullmatch(file_sha):
            raise Refused("contamination index: a source file digest is not a sha256 digest")
        if not isinstance(rows_sha, str) or not _DIGEST.fullmatch(rows_sha):
            raise Refused("contamination index: a source row digest is not a sha256 digest")
        rows = d.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise Refused("contamination index: a source row count is not a positive integer")
        base = d.get("base", "file")
        if not isinstance(base, str) or base not in BASES:
            raise Refused(f"contamination index: a source base is {base!r}; expected one of {list(BASES)}")
        return cls(role, path, file_sha, rows, rows_sha, base)

    def identity(self) -> tuple[str, int, str, str]:
        """What a prebuilt source must match: role, rows and both digests (the path may be relocated)."""
        return (self.role, self.rows, self.file_sha256, self.rows_sha256)


#: Exactly the fields a written source holds.
_SOURCE_FIELDS = frozenset({"role", "path", "file_sha256", "rows", "rows_sha256", "base"})


@dataclass(frozen=True)
class PromptIndex:
    """The reference prompts as hashes: ids, normalized texts and the two window sets (no prompt text)."""

    ids: frozenset[str]
    texts: frozenset[str]
    guard: frozenset[str]
    regex: frozenset[str]
    prompts: int = 0
    sources: tuple[Source, ...] = ()

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> PromptIndex:
        """The index of reference-prompt rows (``{id, prompt}``, ``{id, text}`` or a chat's user turns).

        Every row is validated here, not only in :meth:`from_dict`: a prompt file that is empty, holds no
        prompt text, or carries only ids would produce an index that filters nothing while looking present.
        Windows are not required as such: a prompt shorter than ``WINDOW`` words has none, and its text hash
        still covers it under the exact-text arm.

        Raises:
            Refused: a row is not an object, a row has no prompt text, or no row has prompt text at all (an
                ids-only reference cannot match a prompt).
        """
        ids: set[str] = set()
        texts: set[str] = set()
        guard: set[str] = set()
        regex: set[str] = set()
        n = 0
        for row in rows:
            if not isinstance(row, dict):
                raise Refused(
                    "contamination reference: a prompt row is not a JSON object "
                    f"({type(row).__name__}); a reference is JSONL of {{id, prompt}} rows"
                )
            n += 1
            if "id" in row:
                ids.add(_sha(str(row["id"])))
            found = [_body(t) for t in prompt_texts(row)]
            found = [body for body in found if any(c.isalnum() for c in body)]
            if not found:
                raise Refused(
                    f"contamination reference: prompt row {row.get('id', n - 1)!r} carries no prompt text "
                    "(its normalized text is empty: punctuation or whitespace only); an index of ids alone "
                    "cannot match a calibration prompt"
                )
            texts |= {_sha(t) for t in found}
            for text in prompt_texts(row):
                guard |= windows(text)
                regex |= regex_windows(text)
        index = cls(frozenset(ids), frozenset(texts), frozenset(guard), frozenset(regex), n)
        if not index.texts:
            raise Refused(
                "contamination reference: no prompt text to match; an ids-only index filters nothing "
                "(write one from prompt rows)"
            )
        return index

    @classmethod
    def read(cls, path: str | Path) -> PromptIndex:
        """The index of a prompt JSONL, or of a ``qda-kl ref`` directory's prompt file.

        Raises:
            Refused: the path cannot be read, or a reference directory's manifest names no prompt file.
        """
        return cls.from_rows(reference_rows(path)[0])

    @classmethod
    def read_reference(cls, path: str | Path) -> PromptIndex:
        """The index of a reference's EVAL set, plus the rows its own manifest labels ``split: holdout``.

        A ``qda-kl ref`` directory's manifest names the prompt file in ``build.prompts`` (verified against
        ``build.prompts_sha256`` when recorded). Rows whose ``split`` is ``"holdout"`` are the old-shape holdout
        set carried inside the same file; the rest, including unlabeled rows, are the eval set. A reference whose
        own rows are a subset of a corpus its selection record names (a sealed 96-prompt eval file) is indexed
        from that corpus's full eval split instead (:func:`parent_corpus`); the holdout split is the holdout
        reference's role. ``sources`` records each set's role, file digest and selected-row digest, so a prebuilt
        index can be compared with this canonical derivation.

        Raises:
            Refused: no prompt file is recorded or readable, its recorded digest does not match, the record's
                parent corpus cannot be resolved, or it holds no prompt text.
        """
        prompts, base, rows, sha, build = _reference_rows(path, "reference")
        if build is not None:
            _verify_source_sha(build.get("prompts_sha256"), sha, "prompts", prompts)
        own_rows = rows
        holdout_rows = [r for r in rows if _held_out(r)]
        if (parent := parent_corpus(path, rows, "reference")) is not None:
            prompts, rows, sha, record = parent
            base, holdout_rows = "selection", []
            eval_rows = [r for r in rows if not _held_out(r)]
            root = Path(path)
            _require_own_prompts(
                own_rows,
                eval_rows,
                prompts,
                record,
                root / "manifest.json" if root.is_dir() else root,
                "eval",
            )
        else:
            eval_rows = [r for r in rows if not _held_out(r)]
        return _index_from_rows(prompts, base, sha, eval_rows, holdout_rows)

    @classmethod
    def read_holdout(cls, path: str | Path) -> PromptIndex:
        """The index of a holdout reference: every prompt row it holds is holdout content.

        The holdout reference is a separate set the judge holds out (a ``qda-kl ref`` directory or a prompt
        JSONL). Its rows are all indexed under the ``holdout`` role, whatever ``split`` they carry. A holdout
        reference that is itself a subset (a sealed 96-prompt holdout file) is indexed from its selection
        record's full corpus holdout split instead (:func:`parent_corpus`).

        Raises:
            Refused: no prompt file is recorded or readable, its recorded digest does not match, the record's
                parent corpus cannot be resolved or holds no holdout rows, or it holds no prompt rows.
        """
        prompts, base, rows, sha, build = _reference_rows(path, "holdout reference")
        if build is not None:
            _verify_source_sha(build.get("prompts_sha256"), sha, "prompts", prompts)
        if not rows:
            raise Refused(
                f"contamination holdout reference {prompts}: no prompt rows; the union would miss it"
            )
        own_rows = rows
        if (parent := parent_corpus(path, rows, "holdout reference")) is not None:
            prompts, rows, sha, record = parent
            base = "selection"
            held_rows = [r for r in rows if _held_out(r)]
            if not held_rows:
                raise Refused(
                    f"contamination holdout reference {path}: the parent corpus {prompts} holds no "
                    "`split: holdout` rows; an index may not fall back to the subset"
                )
            root = Path(path)
            _require_own_prompts(
                own_rows,
                held_rows,
                prompts,
                record,
                root / "manifest.json" if root.is_dir() else root,
                "holdout",
            )
            rows = held_rows
        return _index_from_rows(prompts, base, sha, [], rows)

    @classmethod
    def union(cls, parts: Iterable[PromptIndex]) -> PromptIndex:
        """One index over the union of several prompt sets (an evaluation set and its holdout set).

        The hash arrays are sets, so repeated prompts collapse: ``prompts`` is the row count summed over the
        parts and need not equal ``len(texts)``.

        Raises:
            Refused: no part is given, or a part carries no prompt text.
        """
        parts = list(parts)
        if not parts:
            raise Refused("contamination index: no reference prompt sets to union")
        if any(not p.texts for p in parts):
            raise Refused("contamination index: a prompt set in the union carries no prompt text")
        return cls(
            frozenset().union(*(p.ids for p in parts)),
            frozenset().union(*(p.texts for p in parts)),
            frozenset().union(*(p.guard for p in parts)),
            frozenset().union(*(p.regex for p in parts)),
            sum(p.prompts for p in parts),
            tuple(s for p in parts for s in p.sources),
        )

    def same_content(self, other: PromptIndex) -> bool:
        """Whether both indexes hold the same content: every hash array, row count and source identity."""
        return (
            self.prompts,
            self.ids,
            self.texts,
            self.guard,
            self.regex,
            sorted(s.identity() for s in self.sources),
        ) == (
            other.prompts,
            other.ids,
            other.texts,
            other.guard,
            other.regex,
            sorted(s.identity() for s in other.sources),
        )

    def to_dict(self) -> dict[str, Any]:
        """The index as JSON data: hashes only, sorted so the same prompts give the same bytes."""
        return {
            "kind": KIND,
            "window": WINDOW,
            "chars": CHARS,
            "prompts": self.prompts,
            "ids": sorted(self.ids),
            "texts": sorted(self.texts),
            "guard_windows": sorted(self.guard),
            "regex_windows": sorted(self.regex),
            "sources": [s.to_dict() for s in sorted(self.sources, key=lambda s: s.role)],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromptIndex:
        """Read an index written by :meth:`to_dict`.

        Raises:
            Refused: not this kind of index, made with another window (never read under a different rule),
                or empty / id-only (an index that matches nothing must not start a search).
        """
        if not isinstance(d, dict) or d.get("kind") != KIND:
            raise Refused(f"contamination index: not a {KIND} file")
        unknown = set(d) - _INDEX_FIELDS
        if unknown:
            raise Refused(
                f"contamination index: unexpected field(s) {sorted(unknown)}; an index is hashes only "
                "(rewrite it with `python -m qda_quant.calib.contamination`)"
            )
        if d.get("window") != WINDOW or d.get("chars") != CHARS:
            raise Refused(
                f"contamination index: window {d.get('window')}/{d.get('chars')}, this qda uses {WINDOW}/{CHARS}"
            )
        sources = d.get("sources", [])
        if not isinstance(sources, list):
            raise Refused("contamination index: sources is not an array")
        index = cls(
            _digests(d, "ids"),
            _digests(d, "texts"),
            _digests(d, "guard_windows"),
            _digests(d, "regex_windows"),
            _count(d, "prompts"),
            tuple(Source.from_dict(s) for s in sources),
        )
        if index.prompts < 1:
            raise Refused(
                "contamination index: no reference rows; it would filter no row while reporting success "
                "(write one with `python -m qda_quant.calib.contamination`)"
            )
        if index.sources and sum(s.rows for s in index.sources) != index.prompts:
            raise Refused(
                "contamination index: prompts does not equal the sum of its source rows; the manifest and "
                "the union it describes disagree (rewrite it from the references)"
            )
        if _sha("") in index.texts:
            raise Refused(
                "contamination index: a text hash is the empty text; every reference row must carry non-empty "
                "normalized prompt text (rewrite it from prompt rows)"
            )
        if not index.texts:
            raise Refused(
                "contamination index: no prompt-text hashes; an ids-only index matches nothing but a copied id "
                "while reporting success (rewrite it from prompt rows)"
            )
        return index

    def save(self, path: str | Path) -> Path:
        """Write the index to `path`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> PromptIndex:
        """Read an index file (see :meth:`from_dict`).

        Raises:
            Refused: the file cannot be read or is not an index.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            raise Refused(f"contamination index {p}: cannot read ({type(e).__name__})") from None
        return cls.from_dict(data)

    def _shared(
        self, texts: list[str]
    ) -> tuple[dict[str, tuple[str, str] | None], dict[str, tuple[str, str] | None]]:
        """The shared guard and regex windows, by hash, with their text and the prompt they came from.

        A character window (a script written without spaces) has no text and is never exempted: None.
        """
        guard: dict[str, tuple[str, str] | None] = {}
        regex: dict[str, tuple[str, str] | None] = {}
        for t in texts:
            word = _word_windows(t)
            for h, w in word.items():
                if h in self.guard:
                    guard[h] = (w, t)
            for h in windows(t):
                if h not in word and h in self.guard:
                    guard[h] = None  # a character window: a match in a script written without spaces
            for h, w in _regex_texts(t).items():
                if h in self.regex:
                    regex[h] = (w, t)
        return guard, regex

    def hit(self, row: dict[str, Any], tokenizer: Any = None) -> dict[str, Any] | None:
        """Why `row` matches a reference prompt (None: it does not).

        A row is a hit under either normalization, matching the audit's OR. The returned record says which
        test hit: ``id``, ``text``, and ``guard`` / ``regex`` for the window normalizations.
        """
        texts = prompt_texts(row, tokenizer)
        hit: dict[str, Any] = {}
        if "id" in row and _sha(str(row["id"])) in self.ids:
            hit["id"] = True
        if any(_sha(_body(t)) in self.texts for t in texts):
            hit["text"] = True
        guard, regex = self._shared(texts)
        if guard:
            hit["guard"] = True
        if regex:
            hit["regex"] = True
        if not hit:
            return None
        if "id" in hit or "text" in hit:
            return hit
        hit["shared"] = {**guard, **regex}
        return hit


#: Exactly the fields an index file holds: a valid digest array plus an extra raw-text field is refused.
_INDEX_FIELDS = frozenset(
    {
        "kind",
        "window",
        "chars",
        "prompts",
        "ids",
        "texts",
        "guard_windows",
        "regex_windows",
        "sources",
    }
)
#: A stored digest: sha256 hex only, so an index can never carry a prompt passage.
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digests(d: dict[str, Any], key: str) -> frozenset[str]:
    """One index array: every entry a sha256 digest (never a raw string or an object).

    Raises:
        Refused: the key is missing, not an array of strings, or holds a value that is not a sha256 digest.
    """
    value = d.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) and _DIGEST.fullmatch(v) for v in value):
        raise Refused(f"contamination index: {key} is not an array of sha256 digests")
    return frozenset(value)


def _count(d: dict[str, Any], key: str) -> int:
    """One index count: a positive integer (a zero-prompt reference would filter nothing).

    Raises:
        Refused: the value is missing, not an integer, or not positive.
    """
    value = d.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Refused(f"contamination index: {key} is not a positive integer")
    return value


def reference_rows(path: str | Path, what: str = "reference") -> tuple[list[Any], str]:
    """A prompt JSONL's rows and its file digest.

    Raises:
        Refused: the file cannot be read or a line is not JSON.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise Refused(f"contamination {what} {p}: cannot read ({type(e).__name__})") from None
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, ValueError) as e:
        raise Refused(f"contamination {what} {p}: cannot read prompt rows ({type(e).__name__})") from None
    return rows, hashlib.sha256(data).hexdigest()


def reference_prompt_file(
    path: str | Path, what: str = "reference"
) -> tuple[Path, str, dict[str, Any] | None]:
    """The prompt file a reference names, the base it resolved from, and its manifest's ``build``.

    An absolute recorded path is used as written (base ``absolute``). A relative one is resolved against the
    reference (split) directory first (base ``reference``), then against its parent, the reference root (base
    ``parent``), then against the root's parent, the reference group directory (base ``group``), accepting the
    first candidate whose sha256 is the manifest's recorded ``prompts_sha256``: several references under one
    group may deliberately share a corpus beside them rather than keeping a copy each. Beyond the split
    directory that digest is mandatory; no other base is tried, and a relative record may not escape the
    group directory. A plain prompt file (not a directory) is its own base (``file``).

    Raises:
        Refused: the manifest names no prompt file, a relative record escapes the group directory, or it
            matches no candidate's digest.
    """
    p = Path(path)
    if not p.is_dir():
        return p, "file", None
    root = Path(p).resolve()
    build = _manifest_build(root)
    recorded = Path(build["prompts"])
    if recorded.is_absolute():
        return recorded, "absolute", build
    expected = build.get("prompts_sha256")
    beyond = [(root.parent, "parent"), (root.parent.parent, "group")] if expected is not None else []
    group, tried = root.parent.parent, []
    from . import sha256_file

    for base_dir, base in [(root, "reference"), *beyond]:
        candidate = base_dir / recorded
        tried.append(candidate)
        lexical = Path(os.path.normpath(candidate))
        if not lexical.is_relative_to(group):
            raise Refused(
                f"contamination {what} {p}: the recorded prompt file {recorded} leaves {group}; "
                "a relative record may not escape the reference's group directory"
            )
        if expected is None:
            if not lexical.is_relative_to(root):
                raise Refused(
                    f"contamination {what} {p}: the recorded prompt file {recorded} leaves {root} and the "
                    "manifest records no prompts_sha256; a relative record outside the reference directory "
                    "must record the file digest"
                )
            return candidate, base, build
        try:
            if sha256_file(candidate) == expected:
                return candidate, base, build
        except OSError:
            continue
    raise Refused(
        f"contamination {what} {p}: the recorded prompt file {recorded} matches no candidate "
        f"({', '.join(str(t) for t in tried)}); none has the recorded prompts_sha256 {expected}"
    )


def _reference_rows(path: str | Path, what: str) -> tuple[Path, str, list[Any], str, dict[str, Any] | None]:
    """A reference's prompt file, its base, its rows, its file digest, and its manifest's ``build`` (None: file)."""
    prompts, base, build = reference_prompt_file(path, what)
    rows, sha = reference_rows(prompts, what)
    return prompts, base, rows, sha, build


def _selection_record_dirs(reference: Path) -> list[Path]:
    """The directories a reference's selection record is looked in: its root, its parent, its group.

    Each is also tried with a ``corpus/`` subdirectory: a group that shares one corpus beside its references
    keeps the record with the corpus.
    """
    root = reference if reference.is_dir() else reference.parent
    bases = (root, root.parent, root.parent.parent)
    return [d for base in bases for d in (base, base / "corpus")]


def _record_corpus(record: Path) -> dict[str, Any] | None:
    """The ``corpus`` a selection record names (``{path, rows, sha256}``), or None when it names none.

    Raises:
        Refused: the record cannot be read as JSON, or it has a ``corpus`` that is not an object naming a
            path. A record beside a reference is that reference's claim about its parent corpus, so an
            unreadable or malformed claim is refused rather than skipped in favour of the subset it holds.
    """
    try:
        data = json.loads(record.read_text())
    except (OSError, ValueError) as e:
        raise Refused(
            f"contamination selection record {record}: cannot read it as JSON ({type(e).__name__}); an "
            "unreadable record may not be skipped in favour of the subset the reference holds"
        ) from None
    nested = data.get("selection") if isinstance(data, dict) else None
    for node in (data, nested):
        if isinstance(node, dict) and "corpus" in node:
            corpus = node["corpus"]
            if not isinstance(corpus, dict):
                raise Refused(
                    f"contamination selection record {record}: its corpus is {type(corpus).__name__}, not "
                    "an object naming a path, a row count and a sha256"
                )
            if not isinstance(corpus.get("path"), str) or not corpus["path"]:
                raise Refused(f"contamination selection record {record}: its corpus names no path")
            return corpus
    return None


def _prompt_bodies(row: Any) -> set[str]:
    """The digests of a row's prompt-side texts, normalized as the index normalizes them."""
    if not isinstance(row, dict):
        return set()
    bodies = (_body(t) for t in prompt_texts(row))
    return {_sha(body) for body in bodies if any(c.isalnum() for c in body)}


def _require_own_prompts(
    own: list[Any], indexed: list[Any], corpus: Path, record: Path, manifest: Path, role: str
) -> None:
    """Refuse unless every row the reference itself holds is in the indexed set, by id or by prompt text.

    A subset reference's own rows must survive whatever ``split`` labels the parent corpus carries: a corpus
    that files one of them under another split would otherwise leave a prompt the run itself scores out of the
    index.

    Raises:
        Refused: a row of the reference is in neither the indexed ids nor the indexed prompt texts.
    """
    held_ids = {str(r["id"]) for r in indexed if isinstance(r, dict) and "id" in r}
    held_texts: set[str] = set()
    for r in indexed:
        held_texts |= _prompt_bodies(r)
    for row in own:
        if not isinstance(row, dict):
            continue
        rid = str(row["id"]) if "id" in row else None
        if rid is not None and rid in held_ids:
            continue
        if _prompt_bodies(row) & held_texts:
            continue
        raise Refused(
            f"contamination {role} reference {manifest}: its prompt {rid!r} is not in the {role} rows indexed "
            f"from the parent corpus {corpus} (record {record}); that corpus's split labels would leave a "
            "prompt the run itself scores out of the index, so it is refused rather than narrowed"
        )


def parent_corpus(
    path: str | Path, rows: list[Any], what: str = "reference"
) -> tuple[Path, list[Any], str, Path] | None:
    """The full corpus a subset reference was drawn from: its file, rows, digest and the record naming it.

    A reference may hold only the subset of a corpus it was scored on (a sealed eval or holdout split):
    indexing just those rows would leave the rest of the hidden reference unfiltered. So the reference's
    own selection record (``selection.json`` or ``ref-record.json`` in the reference root, its parent, or its
    group, or in a ``corpus/`` directory beside any of them) is read for the corpus it names, and that corpus's
    file must exist at its recorded size and match its recorded digest. Every id the reference holds must be in
    that corpus with the SAME prompt text, and the corpus must hold more rows than the reference: only then is it
    the reference's parent.

    Args:
        path: the reference directory (or prompt file) an index was asked for.
        rows: the prompt rows that reference's manifest names.
        what: the role the reference plays in the message (``reference``, ``holdout reference``).

    Returns:
        (corpus file, its rows, its file digest, the record naming it), or None when no record names a corpus or
        the record's corpus is not a superset of this reference.

    Raises:
        Refused: a record beside the reference cannot be read or names no corpus path; the corpus it names is
            missing or cannot be read; it has not its recorded row count or digest; it does not hold every id
            the reference holds; a shared id carries a different prompt text; or the reference's rows carry no
            ids, so membership cannot be proven. Never fall back to the subset: the builds would then be
            filtered against a fraction of the hidden reference.
    """
    root = Path(path)
    manifest = root / "manifest.json" if root.is_dir() else root
    records = [d / name for d in _selection_record_dirs(root) for name in SELECTION_RECORDS]
    claims = [
        (record, corpus) for record in records if record.is_file() and (corpus := _record_corpus(record))
    ]
    if not claims:
        return None
    ids: dict[str, Any] = {}
    idless = 0
    for r in rows:
        if isinstance(r, dict) and "id" in r:
            ids[str(r["id"])] = r
        else:
            idless += 1
    if idless:
        record, corpus = claims[0]
        named = Path(corpus["path"])
        corpus_path = (named if named.is_absolute() else record.parent / named).resolve()
        raise Refused(
            f"contamination {what} {manifest}: {idless} of its {len(rows)} prompt rows carry no id, and the "
            f"selection record {record} names the parent corpus {corpus_path}; without ids the reference "
            "cannot be shown to be a subset of that corpus, so the index is refused rather than narrowed to "
            "the subset"
        )
    for record, corpus in claims:
        named = Path(corpus["path"])
        corpus_path = (named if named.is_absolute() else record.parent / named).resolve()
        why = (
            f"contamination {what} {manifest}: the selection record {record} names the parent corpus "
            f"{corpus_path}"
        )
        if not corpus_path.is_file():
            raise Refused(f"{why}, which is missing; an index may not fall back to the subset it holds")
        try:
            corpus_rows, digest = reference_rows(corpus_path, f"parent corpus of {what}")
        except Refused as e:
            raise Refused(f"{why}, which cannot be read ({e})") from None
        want = corpus.get("sha256")
        count = corpus.get("rows")
        if not isinstance(want, str) or want != digest:
            raise Refused(f"{why}: its sha256 is {digest}, the record says {want!r}; the corpus changed")
        if not isinstance(count, int) or isinstance(count, bool) or count != len(corpus_rows):
            raise Refused(f"{why}: it holds {len(corpus_rows)} rows, the record says {count!r}")
        held: dict[str, Any] = {str(r["id"]): r for r in corpus_rows if isinstance(r, dict) and "id" in r}
        if missing := set(ids) - set(held):
            raise Refused(
                f"{why}: {len(missing)} of the reference's ids are not in it (first: {sorted(missing)[0]!r}); "
                "the record does not describe this reference"
            )
        for rid, row in ids.items():
            ref_bodies, corpus_bodies = _prompt_bodies(row), _prompt_bodies(held[rid])
            if ref_bodies != corpus_bodies:
                raise Refused(
                    f"{why}: the row with id {rid!r} carries different prompt text (reference "
                    f"{sorted(ref_bodies)[0][:12] if ref_bodies else 'none'}, corpus "
                    f"{sorted(corpus_bodies)[0][:12] if corpus_bodies else 'none'}); the id alone does not "
                    "make it the same prompt"
                )
        if len(held) > len(ids):
            return corpus_path, corpus_rows, digest, record
    return None


def _manifest_build(p: Path) -> dict[str, Any]:
    """A reference directory's recorded ``build`` section.

    Raises:
        Refused: the manifest is missing or malformed, or it names no prompt file.
    """
    try:
        build = json.loads((p / "manifest.json").read_text())["build"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused(f"contamination reference {p}: no prompt file recorded ({type(e).__name__})") from None
    if not isinstance(build, dict) or not isinstance(build.get("prompts"), str):
        raise Refused(f"contamination reference {p}: no prompt file recorded")
    return build


def _verify_source_sha(declared: Any, actual: str, key: str, path: Path) -> None:
    """Refuse when a reference's recorded digest is not the file's (the file changed after it was recorded).

    Raises:
        Refused: a recorded digest is malformed or no longer matches the file.
    """
    if declared is None:
        return
    if not isinstance(declared, str) or declared != actual:
        raise Refused(
            f"contamination reference {path}: {key}_sha256 does not match the file ({declared!r} != {actual})"
        )


def _held_out(row: Any) -> bool:
    """Whether a prompt row is labeled ``split: holdout``."""
    return isinstance(row, dict) and row.get("split") == "holdout"


def _rows_digest(rows: list[Any]) -> str:
    """The digest of exactly these rows, in order, as canonical JSON lines."""
    data = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for r in rows)
    return hashlib.sha256(data.encode()).hexdigest()


def _index_from_rows(
    path: Path, base: str, file_sha256: str, eval_rows: list[Any], holdout_rows: list[Any]
) -> PromptIndex:
    """The union index of two row sets from one file, with a source record for each non-empty set."""
    sources: list[Source] = []
    if eval_rows:
        sources.append(Source("eval", str(path), file_sha256, len(eval_rows), _rows_digest(eval_rows), base))
    if holdout_rows:
        sources.append(
            Source("holdout", str(path), file_sha256, len(holdout_rows), _rows_digest(holdout_rows), base)
        )
    index = PromptIndex.from_rows([*eval_rows, *holdout_rows])
    return PromptIndex(index.ids, index.texts, index.guard, index.regex, index.prompts, tuple(sources))


def filter_rows(
    rows: Iterable[dict[str, Any]], index: PromptIndex, tokenizer: Any = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows the index keeps (exempted ones included), one record per dropped row, one per exempted row.

    A record is ``{"row": position, "id": the row's id, "match": ["id"|"text"|"guard"|"regex", ...]}``; an
    exempted record adds ``"rules"``. The id is never one of the match keys, so a record always names its row.

    A row hit only by windows is exempt when every shared window is structural scaffolding
    (:func:`exempt_rule`); frequency never exempts a window.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    exempted: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        why = index.hit(row, tokenizer)
        if why is None:
            kept.append(row)
            continue
        rec = {
            "row": i,
            "id": row.get("id", f"row {i}"),
            "match": sorted(k for k in ("id", "text", "guard", "regex") if why.get(k)),
        }
        rules = _exempt_rules(why.pop("shared", {}))
        if rules:
            rec["rules"] = sorted(rules)
            exempted.append(rec)
            kept.append(row)  # scaffolding: the row is kept, and its record says why
        else:
            dropped.append(rec)
    return kept, dropped, exempted


def main(argv: list[str] | None = None) -> int:
    """``python -m qda_quant.calib.contamination``: write a reference index of hashes only."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--reference", type=Path, required=True, help="prompt JSONL or a qda-kl reference directory"
    )
    ap.add_argument("--out", type=Path, required=True, help="the index file to write")
    a = ap.parse_args(argv)
    index = PromptIndex.read_reference(a.reference)
    print(index.save(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
