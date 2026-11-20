"""Reject internal names and machine paths in files intended for public release."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PUBLIC_DIRECTORIES = ("packages", "docs", "bench", "recipes", "configs", "examples")
EXCLUDED_PATHS = {
    "uv.lock",
    "tools/check_public_tree.py",
    # This test embeds path prefixes in a regex that rejects machine-specific config values.
    "packages/qda-agent/tests/test_cli.py",
}
FORBIDDEN = (
    # `/Users/<name>` is the macOS spelling of the same thing; matching only `/home/` covered one platform.
    ("home path", re.compile(r"/(?:home|Users)/(?!<)")),
    ("workspace name", re.compile(r"\bqda-publish\b")),
    ("old repository name", re.compile(r"\bqda-dev\b")),
    ("old repository name", re.compile(r"\bqda-flow\b")),
    ("manager role", re.compile(r"\bmgr-[a-z0-9]+\b")),
    ("team role", re.compile(r"\b(?:[wv]-m[1-5]-|s-(?:objective|allocation)\b)")),
    ("legacy team role", re.compile(r"\b[wv]-[a-l][0-9]*\b")),
    ("reviewer role", re.compile(r"\bv-[12]\b")),
    ("review identifier", re.compile(r"\b(?:res|req)-[a-z0-9]+\b")),
    (
        "round-one role",
        re.compile(
            r"\b(?:w-engine|w-methods|w-humming|w-kl|w-flow|w-sched|v-quant|v-kl|v-flow|"
            r"mgr-quant|mgr-kl|mgr-flow)\b"
        ),
    ),
    ("team directory", re.compile(r"\bteam/(?:tools|status|logs|reviews|data|venvs)\b")),
    # Team run trees are internal wherever they sit; before this, a scratch path was caught only when a role name
    # happened to appear inside it, so a scratch path that ended in `qda-team-runs/calib.jsonl` passed.
    ("team run tree", re.compile(r"\bqda-team-runs\b")),
    # A clock stamp (`HH:MMZ`, or the tail of `HH:MM:SSZ`) says when something was noted, not what it was.
    ("clock stamp", re.compile(r"[0-9]{2}:[0-9]{2}Z")),
    # A tracker number names an issue or pull request a reader outside the project cannot open; the sentence
    # around it says the same thing without it. A colour hex (`#1f77b4`, `#fff`, `#123abc`), a Markdown anchor
    # (`(#section)`), a heading (`# 123` is a hash, a space, then a number) and an HTML entity (`&#123;`, whose
    # `#` follows `&`) are not references. A `#` that opens a quoted attribute (`fill="#000"`, `stroke='#fff'`)
    # is a colour too, which is why one lookbehind steps over the `=` and its quote; a number right after a
    # docstring's `"""` is still a reference, and the trailing `\b` keeps `#123abc` out.
    ("issue reference", re.compile(r"(?<![\w#&/])(?<!=['\"])#\d{3,4}\b")),
)

Rules = tuple[tuple[str, re.Pattern[str]], ...]


def load_patterns(path: Path) -> Rules:
    """Read extra rules from ``path``: a ``rule name<TAB>regex`` per line, matched in any letter case.

    Blank lines and lines that start with ``#`` are skipped. Names that must not appear in this repository
    (so cannot be spelled here either) are checked this way, from a file kept outside it.

    Raises:
        ValueError: a line has no tab or an empty regex, which would match every line.
    """
    rules = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, _, pattern = line.partition("\t")
        if not name.strip() or not pattern:
            raise ValueError(f"{path}:{number}: expected `rule name<TAB>regex`")
        rules.append((name.strip(), re.compile(pattern, re.IGNORECASE)))
    return tuple(rules)


def public_path(path: str) -> bool:
    """Return whether a tracked path belongs to the public source tree."""
    if path in EXCLUDED_PATHS:
        return False
    return "/" not in path or path.startswith(tuple(f"{directory}/" for directory in PUBLIC_DIRECTORIES))


def find_matches(text: str, extra: Rules = ()) -> list[tuple[int, str, str]]:
    """Find forbidden terms (the built-in rules, then ``extra``) and return their line, rule and source text."""
    matches = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in (*FORBIDDEN, *extra):
            if pattern.search(line):
                matches.append((line_number, rule, line.strip()))
    return matches


def tracked_public_paths(root: Path) -> list[str]:
    """List tracked files covered by the release check."""
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    paths = result.stdout.decode().split("\0")
    return sorted(path for path in paths if path and public_path(path))


def check_tree(root: Path, extra: Rules = ()) -> list[str]:
    """Return formatted violations of the built-in and ``extra`` rules in tracked public text files."""
    violations = []
    for relative_path in tracked_public_paths(root):
        contents = (root / relative_path).read_bytes()
        if b"\0" in contents:
            continue
        text = contents.decode("utf-8", errors="replace")
        for line_number, rule, line in find_matches(text, extra):
            violations.append(f"{relative_path}:{line_number}: {rule}: {line}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the public-tree check from this checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-patterns",
        type=Path,
        metavar="FILE",
        help="more rules to apply, one `rule name<TAB>regex` per line (case-insensitive; # starts a comment)",
    )
    args = parser.parse_args(argv)
    extra = load_patterns(args.extra_patterns) if args.extra_patterns else ()
    root = Path(__file__).resolve().parents[1]
    violations = check_tree(root, extra)
    if violations:
        print("Public-tree check failed:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
