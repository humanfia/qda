"""``qda doctor``: walk the guards real commands refuse on, before running them, and report what was checked.

Checks are ``qda.checks`` plugins; a miss exits 3, and a search config that does not load exits 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .exit import ENVIRONMENT, OK, USAGE
from .guards import Verdict
from .registry import Registry

__all__ = ["CHECKS", "Check", "Context", "Info", "add_parser", "guard_lines", "main", "report", "run"]


@dataclass(frozen=True)
class Context:
    """The command's options, for the checks that depend on a config.

    Attributes:
        search: a qda-agent run config to check the search prerequisites of, when one is given.
        overrides: ``KEY.PATH=VALUE`` overrides for that config.
        report: ``--report``: list the environment (`Info` lines), judge nothing.
    """

    search: Path | None = None
    overrides: tuple[str, ...] = ()
    report: bool = False


@dataclass(frozen=True)
class Check:
    """One line: what was checked and what was found, and for a miss the one change that fixes it.

    Attributes:
        name: the line's label (``GPU``), which `report` also lists a failed line by.
        fix: the command or the one change that fixes a miss; left out of an OK line.
        skipped: the check does not apply here (a CPU run has no cards), so it neither passes nor fails.
        unchecked: it cannot be decided in advance; the line names `where` it is checked, never a pass.
        code: the exit status a miss stands for: 3 (the machine), or 2 when the input itself is wrong.
        timed_out: the check hit its time limit before a verdict (`checks.check_triton_build`).
    """

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    skipped: bool = False
    unchecked: bool = False
    where: str = ""
    code: int = ENVIRONMENT
    timed_out: bool = False

    @property
    def missed(self) -> bool:
        """Whether this line is a MISS: checked here, and it does not hold."""
        return not self.ok and not self.skipped and not self.unchecked

    def line(self) -> str:
        """The line as printed: a status tag, the name, the detail, and a miss's fix or a step's name."""
        tag = "skip" if self.skipped else "----" if self.unchecked else "OK" if self.ok else "MISS"
        line = f"{tag:<4} {self.name}" + (f": {self.detail}" if self.detail else "")
        if self.fix and (self.missed or self.unchecked):
            line += f" (fix: {self.fix})"
        if self.unchecked and self.where:
            line += ("; " if self.detail or self.fix else ": ") + f"checked by {self.where}"
        return line


@dataclass(frozen=True)
class Info:
    """One line of ``qda doctor --report``: a fact about this machine, with no verdict."""

    name: str
    detail: str

    def line(self) -> str:
        """The line as printed."""
        return f"{self.name}: {self.detail}"


def guard_lines(verdicts: Iterable[Verdict]) -> list[Check]:
    """One doctor line per walked guard (`qda_quant.guards.all_problems`), with its refusal verbatim."""
    lines = []
    for v in verdicts:
        g = v.guard
        if not v.advance:
            lines.append(Check(g.name, False, "cannot be checked in advance", unchecked=True, where=g.where))
        elif v.skipped is not None:
            lines.append(Check(g.name, True, v.skipped, skipped=True))
        elif v.problem is None:
            lines.append(Check(g.name, True, v.held))
        else:
            # the command's own status is its refusal's; doctor's is 3 for any miss but a config that does not load
            lines.append(Check(g.name, False, v.problem.text, v.problem.fix or ""))
    return lines


def _adopt(key: str, check: Callable[[Context], Iterable[Check | Info]]) -> None:
    """Refuse a ``qda.checks`` entry that is not a function taking a `Context`."""
    if not callable(check):
        raise TypeError(f"checks plugin {key!r} is {check!r}, not a callable taking a Context")


#: Checks by plugin name: built-ins and package registrations (see `qda_quant.registry`).
CHECKS: Registry[Callable[[Context], Iterable[Check | Info]]] = Registry(
    "checks",
    {
        "environment": "qda_quant.checks:environment",
    },
    group="qda.checks",
    on_add=_adopt,
)


def run(ctx: Context) -> list[Check | Info]:
    """Every check's lines in plugin-name order; a raising or missing plugin is one MISS."""
    lines: list[Check | Info] = []
    asked = set()
    for key, check in CHECKS.items():
        asked.add(key)
        try:
            lines += list(check(ctx))
        except Exception as e:  # noqa: BLE001  (a broken third-party check is a line, not a traceback)
            why = f"the check raised {type(e).__name__}: {e}"
            lines.append(Info(key, why) if ctx.report else Check(key, False, why))
    if ctx.search is not None and "search" not in asked and not ctx.report:
        why = "qda-agent is not installed"
        lines.append(
            Check(
                "search",
                False,
                f"{ctx.search}: {why}",
                "uv pip install -c constraints.txt ./packages/qda-agent",
            )
        )
    return lines


def report(lines: Sequence[Check | Info]) -> str:
    """Print the lines, then the tail ``checked N, M MISS, K cannot be checked in advance``.

    A skip counts in none of N, M and K; `Info` lines (``--report``) are printed with no tail.
    """
    checks = [c for c in lines if isinstance(c, Check)]
    body = [c.line() for c in lines]
    if not checks:
        return "\n".join(body) + "\n" if body else ""
    checked = sum(1 for c in checks if not c.skipped and not c.unchecked)
    missed = sum(1 for c in checks if c.missed)
    later = sum(1 for c in checks if c.unchecked)
    tail = f"checked {checked}, {missed} MISS, {later} cannot be checked in advance"
    return "\n".join([*body, tail]) + "\n"


def status(lines: Sequence[Check | Info]) -> int:
    """The exit status: 2 when a miss is the input (a config that does not load), 3 for any other miss, else 0."""
    misses = [c for c in lines if isinstance(c, Check) and c.missed]
    if not misses:
        return OK
    return USAGE if any(c.code == USAGE for c in misses) else ENVIRONMENT


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Add ``doctor`` to a subparser set."""
    p = sub.add_parser(
        "doctor", help="run the checks the commands refuse on, early; --report lists the environment"
    )
    p.add_argument(
        "--search",
        type=Path,
        metavar="CONFIG.yaml",
        help="also check the prerequisites of a qda-agent search config (needs qda-agent)",
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY.PATH=VALUE", help="override one key of --search"
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="list the environment for a bug report (versions, driver, filesystems): no verdict, no paths",
    )
    p.set_defaults(run=main)


def main(a: argparse.Namespace) -> int:
    """Run every check and print the report; with ``--report``, list the environment and exit 0.

    Exits 3 (`ENVIRONMENT`) when a check misses, 2 (`USAGE`) when the search config does not load, else 0.
    """
    lines = run(Context(a.search, tuple(a.set), report=getattr(a, "report", False)))
    print(report(lines), end="")
    return OK if getattr(a, "report", False) else status(lines)
