"""Exit statuses of every ``qda`` command, and the ``error:``/``fix:`` lines a refusal prints instead of a traceback.

A refusal carries its status in `qda_quant.step.Refused.code`, so a script can tell bad input from a machine limit.
"""

from __future__ import annotations

import sys

__all__ = [
    "ENVIRONMENT",
    "ERROR",
    "ERROR_LINE",
    "FIX_LINE",
    "INTERRUPTED",
    "OK",
    "USAGE",
    "crash",
    "refusal",
    "refuse",
]

#: The two lines a refusal prints, aligned; job-log readers grep for `ERROR_LINE` to quote a child's failure.
ERROR_LINE = "error: "
FIX_LINE = "fix:   "

#: The command did what it was asked.
OK = 0
#: An unexpected error (a bug or a crashed helper), the same status an uncaught Python exception leaves.
ERROR = 1
#: The input: a config, an argument, a path, a checkpoint the command cannot use. Another command fixes it.
USAGE = 2
#: The machine: no such card, too little memory or disk, a missing package, no sign-in.
ENVIRONMENT = 3
#: Interrupted: Ctrl-C, or SIGTERM/SIGHUP in a command that owns engine processes.
INTERRUPTED = 130


def refusal(problem: str, fix: str | None = None) -> str:
    """The lines a refusal prints: what is wrong, and the one change that fixes it when there is one."""
    return f"{ERROR_LINE}{problem}\n" + (f"{FIX_LINE}{fix}\n" if fix else "")


def refuse(problem: str, fix: str | None = None, *, code: int = USAGE) -> int:
    """Print a refusal on stderr and return its exit status, for a command's ``return refuse(...)``."""
    print(refusal(problem, fix), end="", file=sys.stderr)
    return code


def crash(error: BaseException) -> tuple[str, str | None]:
    """What a failure that is no refusal prints: its type and message with the lines noted on it, and its fix.

    An engine start notes the upstream line that names a failure it knows and a `FIX_LINE` after it
    (`qda_quant.output.vllm_env.add_fix`); the fix is returned on its own, without the prefix, for `refuse`.
    """
    notes = [str(note) for note in getattr(error, "__notes__", ())]
    fix = next((note.removeprefix(FIX_LINE) for note in notes if note.startswith(FIX_LINE)), None)
    said = [note for note in notes if not note.startswith(FIX_LINE)]
    return "; ".join([f"{type(error).__name__}: {error}", *said]), fix
