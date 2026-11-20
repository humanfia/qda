"""File-byte hashing shared by calibration, intake and checkpoint records, and the CLIs' shared stdout setting."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of a file's full bytes without loading it into memory."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def qda_cache(*parts: str) -> Path:
    """``$QDA_CACHE/<parts>``; without QDA_CACHE ``~/.cache/qda/<parts>``: the one root of every QDA cache."""
    return Path(os.environ.get("QDA_CACHE") or Path.home() / ".cache" / "qda", *parts)


def show_only(sub: argparse._SubParsersAction, public: tuple[str, ...]) -> None:
    """List only `public` in a program's ``--help``; every other subcommand still runs when named.

    Drops the others from the ``{a,b}`` list of the usage line and from the help rows, and records the public
    names on the action (``public``) for the generated CLI page.
    """
    sub.metavar = "{" + ",".join(public) + "}"
    sub._choices_actions = [a for a in sub._choices_actions if a.dest in public]
    sub.public = public  # type: ignore[attr-defined]


def line_buffered_stdout() -> None:
    """Flush stdout at every newline, also when it is a file or a pipe; each CLI's ``main`` calls this first.

    Python block-buffers a redirected stdout (8 KB), so a long step's progress lines (``qda build``'s per-step
    line, a reference's progress) stayed invisible in ``> log`` for tens of minutes. stderr is line-buffered
    already. A stdout that is not a text stream (a test's capture) is left as it is.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
