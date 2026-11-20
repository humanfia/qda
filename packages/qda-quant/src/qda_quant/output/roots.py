"""Where the writing commands may put their outputs: ``QDA_OUTPUT_ROOTS``, a default and a refusal.

A site whose shared filesystems fill up can confine every large output (checkpoints, traces, release copies,
data files) to its own disks by setting ``QDA_OUTPUT_ROOTS`` to one or more directories, ``os.pathsep``
separated (``/var/tmp`` for node-local disk, say). Then for ``qda build``, ``qda plan``, ``qda traces``,
``qda export`` and ``qda data public``:

- the default: a relative output path is taken inside the first root, not the current directory;
- the refusal: an output that resolves outside every root is refused before anything is written.

Unset, outputs go where they are named. The rule lives in the commands, so it holds for whoever runs them.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..step import Refused

ENV = "QDA_OUTPUT_ROOTS"


def roots() -> list[Path]:
    """The configured output roots, resolved, in order (empty when ``QDA_OUTPUT_ROOTS`` is unset)."""
    return [Path(r).expanduser().resolve() for r in os.environ.get(ENV, "").split(os.pathsep) if r]


def output_path(path: str | Path) -> Path:
    """``path`` resolved under the output roots: relative paths go inside the first root.

    Raises:
        Refused: the path resolves outside every root in ``QDA_OUTPUT_ROOTS``.
    """
    allowed = roots()
    p = Path(path).expanduser()
    if not allowed:
        return p.resolve()
    if not p.is_absolute():
        p = allowed[0] / p
    p = p.resolve()
    if not any(p.is_relative_to(r) for r in allowed):
        raise Refused(
            f"output {p} is outside {ENV}={os.environ[ENV]}; write it under one of those roots "
            "(a relative path goes under the first)"
        )
    return p
