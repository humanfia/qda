"""Contamination checks (`qda_agent.contamination_checks`): no test item is readable where it must not be.

A check is a plugin whose settings build a scanner, `(paths) -> findings`, one finding per contaminated
file; an empty list is clean. The flow runs the run config's check over the agent's files before it starts,
and the judge runs the same check on every frozen checkpoint:

- `none`: finds nothing.
- `ngram`: finds files that contain a `WINDOW`-word window of a test item. Its guard file is text, one
  sha256 per line, each over `WINDOW` whitespace-separated lower-cased words of a test item (built from the
  benchmarks before the run, so the test items themselves never sit where the agent reads). Text written
  without spaces between words (Han, Kana, Thai) additionally yields `CHARS`-character windows, so those
  test items are covered too, even when English text sits beside them in the same string. A file is read as
  its raw text and as each JSON row's strings on their own, so an English row cannot dilute the script share
  that would uncover a Japanese row beside it.

    findings = scan([workdir], Ngram(guard=guard))    # the flow's signature: (paths, check)
    check = Ngram(guard=guard).scanner(); check([ckpt])   # build once, run many times (the judge)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from qda_quant.registry import Registry

from qda_agent.settings import Plugin, validate_plugin

from .freeze import walk

CHECKS = Registry(
    "contamination_checks",
    {"none": "qda_agent.judge.contamination:NoCheck", "ngram": "qda_agent.judge.contamination:Ngram"},
    group="qda_agent.contamination_checks",
    on_add=partial(validate_plugin, "qda_agent.contamination_checks"),
)

__all__ = ["ContaminationCheck", "Ngram", "NoCheck", "scan", "windows"]

Scanner = Callable[[list[Path]], list[str]]
WINDOW = 20
MAX_FILE = 64 << 20
#: One guard line: the lowercase sha256 hex digest `windows` writes.
GUARD_HASH = re.compile(r"[0-9a-f]{64}")
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


class ContaminationCheck(Plugin):
    """Settings of a contamination check.

    Attributes:
        guard: The check's guard file (what it looks for); None for a check that needs none.
    """

    guard: Path | None = None

    def scanner(self) -> Scanner:
        """The scanner, built once (it may read the guard file).

        Raises:
            ValueError: The guard file is missing, unreadable or unusable.
        """
        raise NotImplementedError


class NoCheck(ContaminationCheck):
    """Finds nothing."""

    name: Literal["none"] = "none"

    def scanner(self) -> Scanner:
        """A scanner with no findings."""
        return lambda paths: []  # noqa: ARG005


class Ngram(ContaminationCheck):
    """Files holding a `WINDOW`-word window of a test item (the guard: one window sha256 per line).

    It is the run's default check, so a config that names no guard is refused when it is loaded, naming
    `contamination.guard`, rather than when the host builds the scanner at start; `contamination: none`
    written out runs without a check.
    """

    name: Literal["ngram"] = "ngram"

    @model_validator(mode="after")
    def _needs_a_guard(self) -> Ngram:
        if self.guard is None:
            raise ValueError(
                "contamination.guard is not set: the contamination check (ngram, the default) needs its guard "
                "file, one sha256 per line built from the test items; set contamination.guard, or write "
                "`contamination: {name: none}` to run without the check"
            )
        return self

    def scanner(self) -> Scanner:
        """The scanner against the guard's hashes.

        Raises:
            ValueError: The guard file is missing or unreadable, holds no hashes, or holds a line that is not
                a sha256 hex digest. A guard with no usable hash would report every file clean.
        """
        if self.guard is None:
            raise ValueError("the ngram contamination check needs a guard file (contamination.guard)")
        try:
            lines = [line.strip() for line in Path(self.guard).read_text().splitlines() if line.strip()]
        except OSError as e:
            raise ValueError(f"contamination guard {self.guard}: {type(e).__name__}") from None
        if not lines:
            raise ValueError(
                f"contamination guard {self.guard}: no window hashes (contamination.guard lists one sha256 "
                "per line, built from the test items)"
            )
        if wrong := next((line for line in lines if not GUARD_HASH.fullmatch(line)), None):
            raise ValueError(
                f"contamination guard {self.guard}: {wrong[:16]!r} is not a sha256 hex digest "
                "(contamination.guard lists one 64-character sha256 per line)"
            )
        bad = set(lines)

        def check(paths: list[Path]) -> list[str]:
            found = []
            for f in _files(paths):
                if f.suffix == ".safetensors" or f.stat().st_size > MAX_FILE:
                    continue
                if any(windows(text) & bad for text in _texts(f.read_text(errors="ignore"), f.suffix)):
                    found.append(f"{f}: contains a test item")
            return found

        return check


def scan(paths: list[Path], check: ContaminationCheck) -> list[str]:
    """Run `check` over every file under `paths` (what the flow runs before the start).

    Returns:
        One finding per contaminated file; empty means clean.
    """
    return check.scanner()(list(paths))


def windows(text: str, n: int = WINDOW) -> set[str]:
    """sha256 of every `n`-word window of `text` (lower case, whitespace-split) and of its character windows.

    The word windows are unchanged for every text. A text written without spaces between words gets no word
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

    A text mostly written in a no-space script gets the windows of its whole whitespace-stripped body, as
    before. A mixed text (foreign text inside an English instruction) stays below that share, so every
    whitespace-separated run that contains a no-space character gets the windows of its own body instead:
    the text around the run can no longer hide it.
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


def _texts(text: str, suffix: str) -> Iterator[str]:
    """Every text item of a file on its own: the file, each line, and each JSON row's strings.

    The script share that turns on character windows is a property of a text, not of the file it sits in, so
    a Japanese benchmark row is found with or without an English row beside it.
    """
    yield text
    if suffix == ".json":
        with contextlib.suppress(json.JSONDecodeError):
            yield from _strings(json.loads(text))
    for line in text.splitlines():
        yield line
        with contextlib.suppress(json.JSONDecodeError):
            yield from (_strings(json.loads(line)) if suffix == ".jsonl" else ())


def _strings(value) -> Iterator[str]:
    """Every string in a decoded JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend([p] if p.is_file() else [x for x in walk(p) if x.is_file() and not x.is_symlink()])
    return out
