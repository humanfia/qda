"""One isolation interface: nothing outside `isolation/` imports the modules behind it.

`qda_agent.isolation.api` is what the flow, the CLI, the scheduler and the judge use; `bwrap`, `wall`,
`allocation` and `hostnet` are its implementations, and `sched.sandbox` is the old path the implementations
moved from. An import of one of those from outside `isolation/` bypasses the level a run chose, which is
exactly what the refactor removes. The named exemptions are independent sandboxes, the diagnostic tool for
the sandbox itself, and the deprecated old path, not the run's level.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "qda_agent"

#: The implementation modules no code outside `isolation/` may import.
BANNED = (
    "qda_agent.isolation.bwrap",
    "qda_agent.isolation.wall",
    "qda_agent.isolation.allocation",
    "qda_agent.isolation.hostnet",
    "qda_agent.sched.sandbox",
)

#: Modules allowed to import a banned one, and why.
EXEMPT = {
    # The LiveCodeBench grader's allowlist sandbox runs untrusted generated code; it is deliberately
    # independent of the run's isolation level.
    "livecodebench_sandbox.py": {"qda_agent.isolation.bwrap"},
    # `qda doctor --search` builds probe sandboxes; it is the diagnostic tool for the sandbox itself.
    "doctor.py": {"qda_agent.isolation.bwrap"},
    # The deprecated import path itself: forwarding every name to the implementation is its whole job.
    "sched/sandbox.py": {"qda_agent.isolation.bwrap"},
}


def _module_of(path: Path) -> str:
    """The dotted name of the package holding `path` (its `__package__`)."""
    parts = list(path.relative_to(SRC).with_suffix("").parts[:-1])  # drop the file
    return ".".join(["qda_agent", *parts])


def _resolve(module: str | None, level: int, package: str) -> str:
    """The absolute module an `ImportFrom` names, resolving `level` dots against `package`."""
    if level == 0:
        return module or ""
    base = package.split(".") if package else []
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, *(module.split(".") if module else [])])


def _imports(path: Path) -> set[str]:
    """Every module path the file imports, including `from X import Y` as `X.Y`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = _module_of(path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            if base:
                found.add(base)
            found.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return found


def _offends(found: set[str], banned: tuple[str, ...]) -> set[str]:
    return {imported for imported in found for b in banned if imported == b or imported.startswith(b + ".")}


def _allowed(imported: str, allowed: set[str]) -> bool:
    """Whether `imported` (a module or a name under it) is one of the exempted module paths."""
    return any(imported == a or imported.startswith(a + ".") for a in allowed)


def test_nothing_outside_isolation_imports_an_isolation_implementation():
    """Every source file outside `isolation/` reaches isolation only through its public modules."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[0] == "isolation":
            continue
        allowed = EXEMPT.get(rel.as_posix(), set())
        bad = sorted(name for name in _offends(_imports(path), BANNED) if not _allowed(name, allowed))
        if bad:
            offenders[rel.as_posix()] = bad
    assert not offenders, offenders


def test_the_exemptions_are_still_needed():
    """An exemption that stops importing a banned module is removed, so the list stays honest."""
    unused = []
    for rel, allowed in EXEMPT.items():
        path = SRC / rel
        assert path.is_file(), rel
        if not any(_allowed(name, allowed) for name in _offends(_imports(path), BANNED)):
            unused.append(rel)
    assert not unused, unused
