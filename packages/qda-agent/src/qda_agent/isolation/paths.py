"""Path facts every isolation level and the run's own checks need, with no level behind them.

These are pure functions and constants: what the system mounts are, what the running Python needs, whether a
path resolves into or holds the run directory, the read-only entries a job published (the judge registers the
reader), a readable source's snapshot links, the interpreter's own path, and the clean environment a sandboxed
job starts from. Nothing here starts a process, so any module may import it.
"""

from __future__ import annotations

from .bwrap import (
    SYSTEM,
    Expanded,
    Interpreter,
    expand_links,
    holds_run,
    in_run,
    interpreter,
    interpreter_roots,
    published_entries,
    python,
    python_roots,
    sandbox_env,
)

#: The clean environment of a sandboxed job (the old name `sandbox_env` is kept for one release).
job_env = sandbox_env

__all__ = [
    "SYSTEM",
    "Expanded",
    "Interpreter",
    "expand_links",
    "holds_run",
    "in_run",
    "interpreter",
    "interpreter_roots",
    "job_env",
    "published_entries",
    "python",
    "python_roots",
    "sandbox_env",
]
