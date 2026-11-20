"""Guards: the checks a command refuses on, in one ordered list that the command and `qda doctor` both walk.

A command that can be refused (``qda build``, an engine start, ``qda-agent prepare``, a search's start) keeps
its preconditions as a tuple of `Guard` entries. The command walks it with `first_problem` and refuses on the
first `Problem`, in the contract's two lines (`qda_quant.exit`); ``qda doctor`` walks the same tuple with
`all_problems`, so a line doctor prints is the refusal the command would print, word for word. There is no
second copy of a check to drift: a guard is its check function, named once.

A guard that cannot be decided before the command runs (a kernel compiled at load, two engines taking one
lock, a reading of candidate 0) is listed with ``advance=False`` and no check: doctor names it and the step
that really checks it, and never prints it as passed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .exit import ENVIRONMENT, refusal

__all__ = [
    "BUILD",
    "EXPORT_GGUF",
    "PREPARE",
    "SEARCH_LAUNCH",
    "SEARCH_START",
    "SERVE",
    "Guard",
    "Problem",
    "Verdict",
    "all_problems",
    "first_problem",
]

#: The commands a guard can belong to. ``search-start`` is what a search refuses before it creates anything
#: (`qda_agent.cli.start_problem`, `qda_agent.judge.startup.static_checks`); ``search-launch`` is the host's
#: start, after the run dir exists and before the agent's first turn (`qda_agent.judge.startup.launch_smoke`).
BUILD = "build"
SERVE = "serve"
PREPARE = "prepare"
SEARCH_START = "search-start"
SEARCH_LAUNCH = "search-launch"
EXPORT_GGUF = "export-gguf"

C = TypeVar("C")


@dataclass(frozen=True)
class Problem:
    """Why a command cannot run here: the refusal's two lines and its exit status.

    Attributes:
        text: What is wrong, as the command prints it after ``error:``.
        fix: The one change or command that unblocks it, printed after ``fix:``; None when there is none.
        code: The exit status (`qda_quant.exit`): 2 for the input, 3 for this machine.
        error: The exception the command's own code raises for it, when that code raises rather than returns
            (`qda_agent.judge.startup.static_checks` re-raises it, so its callers see the type they always did).
    """

    text: str
    fix: str | None = None
    code: int = ENVIRONMENT
    error: BaseException | None = field(default=None, compare=False, repr=False)

    def refusal(self) -> str:
        """The two lines the command prints (`qda_quant.exit.refusal`)."""
        return refusal(self.text, self.fix)


def _applies(ctx: object) -> str | None:  # noqa: ARG001  (the default: a guard applies wherever it is listed)
    return None


@dataclass(frozen=True)
class Guard(Generic[C]):
    """One precondition of one or more commands.

    Attributes:
        name: The label doctor prints the guard under (``run dir``, ``judge qda``).
        commands: The commands that refuse on it (`BUILD`, `SERVE`, ...).
        check: The check itself: the `Problem` the command refuses with, or None when it holds. None for a
            guard that cannot be checked in advance.
        where: Where the command really checks it, for doctor's line about a guard it cannot run
            (``qda build: Triton's driver build``).
        advance: Whether it can be decided before the command runs, without a GPU job, an engine or a model
            load. Doctor runs every advance guard in scope and lists the others with `where`.
        applies: Why the guard does not apply to this context (a CPU run has no cards to check), or None when
            it does. A guard that does not apply is a skip in doctor, never a pass, and the command does not
            run it.
        held: What doctor's OK line says was checked, when the name alone does not (``no fault on cards [0, 1]``).
    """

    name: str
    commands: frozenset[str]
    check: Callable[[C], Problem | None] | None
    where: str
    advance: bool = True
    applies: Callable[[C], str | None] = _applies
    held: Callable[[C], str] | None = None

    def __post_init__(self) -> None:
        if self.advance and self.check is None:
            raise ValueError(f"guard {self.name!r} can be checked in advance, so it needs a check")


@dataclass(frozen=True)
class Verdict:
    """What doctor found for one guard: a problem, a skip with its reason, or that it cannot run in advance."""

    guard: Guard
    problem: Problem | None = None
    skipped: str | None = None
    held: str = ""

    @property
    def advance(self) -> bool:
        """Whether the guard was decided here (False: it is checked only when the command runs)."""
        return self.guard.advance


def _in(guard: Guard, commands: Iterable[str]) -> bool:
    return not guard.commands.isdisjoint(commands)


def first_problem(guards: Sequence[Guard[C]], command: str, ctx: C) -> Problem | None:
    """The first problem `command` refuses with, walking `guards` in order; None when every guard holds.

    Only the advance guards of `command` that apply to `ctx` are run; a check's own exception propagates, as
    it does in the command that calls this.
    """
    for guard in guards:
        if command not in guard.commands or not guard.advance or guard.applies(ctx) is not None:
            continue
        assert guard.check is not None  # an advance guard has one (`Guard.__post_init__`)
        if (problem := guard.check(ctx)) is not None:
            return problem
    return None


def all_problems(guards: Sequence[Guard[C]], commands: Iterable[str], ctx: C) -> list[Verdict]:
    """Every guard of `commands` in order, each with what it found: `qda doctor`'s walk of `first_problem`'s list.

    Unlike `first_problem` it does not stop at a problem, and a check that raises is that guard's problem
    (named, with the exception) rather than the end of the report.
    """
    commands = frozenset(commands)
    found: list[Verdict] = []
    for guard in guards:
        if not _in(guard, commands):
            continue
        if not guard.advance:
            found.append(Verdict(guard))
            continue
        if (why := guard.applies(ctx)) is not None:
            found.append(Verdict(guard, skipped=why))
            continue
        assert guard.check is not None
        try:
            problem = guard.check(ctx)
        except Exception as e:  # noqa: BLE001  (one guard's crash is its line, not the end of the report)
            problem = Problem(f"the check raised {type(e).__name__}: {e}", error=e)
        held = guard.held(ctx) if problem is None and guard.held is not None else ""
        found.append(Verdict(guard, problem, held=held))
    return found
