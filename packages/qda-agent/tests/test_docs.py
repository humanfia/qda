"""docs/search-config.md: its doctests pass, its configs validate, its commands and flags exist, its table is the code's."""

import argparse
import doctest
import re
import shlex
from pathlib import Path

import pytest
import yaml
from qda_agent import cli
from qda_agent.agents import BACKENDS
from qda_agent.components import BUILTINS, COMPONENTS
from qda_agent.config import RunConfig
from qda_agent.judge.contamination import CHECKS
from qda_agent.judge.final_rules import FINAL_RULES
from qda_agent.launchers import LAUNCHERS
from qda_agent.sandbox_settings import SANDBOXES
from qda_agent.sched.config import SCHEDULERS
from qda_agent.toolsets import TOOLSETS
from qda_quant.registry import load_plugins

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "docs" / "search-config.md"


def blocks(lang: str) -> list[str]:
    """The bodies of the page's fenced ``lang`` blocks."""
    return re.findall(rf"^```{lang}\n(.*?)^```", PAGE.read_text(), re.S | re.M)


def test_doctests_pass():
    failed, tried = doctest.testfile(
        str(PAGE), module_relative=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert tried > 0 and failed == 0


#: The paths a user page's YAML names, and how to stand in for each: a virtualenv, a file, a directory.
INTERPRETERS = ("toolset.python", "judge.serve_python")
FILES = ("calib", "judge.kl_holdout_ref", "contamination.guard")
DIRS = (
    "run_dir",
    "baselines",
    "scheduler.readable",
    "judge.kl_ref",
    "judge.anchor",
    "judge.format_reference",
)


def _stand_in(tmp_path: Path, raw: dict, key: str) -> None:
    node: object = raw
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    for value in node if isinstance(node, list) else [node]:
        if not isinstance(value, str):
            continue
        path = tmp_path / value
        if key in INTERPRETERS:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/usr/bin/env python3\n")
            (path.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
        elif key in FILES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        else:
            path.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("text", blocks("yaml"))
def test_yaml_blocks_validate(text, tmp_path):
    raw = yaml.safe_load(text)
    for key in (*INTERPRETERS, *FILES, *DIRS):
        _stand_in(tmp_path, raw, key)
    if ref := raw.get("judge", {}).get("format_reference"):  # a checkpoint, not an empty dir
        (tmp_path / ref / "config.json").write_text("{}")
    (tmp_path / "run.yaml").write_text(text)
    RunConfig.load(tmp_path / "run.yaml")


def test_shell_commands_exist(capsys):
    """Every documented command runs here, takes the page's flags, and names an existing config."""
    from qda_quant import cli as qda_cli

    lines = [
        shlex.split(x)
        for b in blocks("sh")
        for x in b.splitlines()
        if x.startswith(("qda-agent ", "qda doctor ", "qda eval "))
    ]
    assert {argv[1] for argv in lines if argv[0] == "qda-agent"} == {"run", "prepare", "search"}
    assert ["qda", "doctor"] in [argv[:2] for argv in lines]  # `qda doctor --search` is the one entry
    assert ["qda", "eval"] in [argv[:2] for argv in lines]  # `qda eval RUN_DIR` scores the result
    for argv in lines:
        main = cli.main if argv[0] == "qda-agent" else qda_cli.main
        with pytest.raises(SystemExit):
            main([argv[1], "--help"])
        usage = capsys.readouterr().out
        for flag in (a.split("=")[0] for a in argv if a.startswith("--")):
            assert flag in usage, f"{flag} is not an option of {argv[0]} {argv[1]}"
        if argv[1] in ("run", "doctor"):  # prepare, search and eval name the user's own directories
            assert (ROOT / next(a for a in argv[2:] if not a.startswith("-"))).is_file()


def test_plugin_table_is_the_registries():
    rows = re.findall(r"^\| [^|]+ \| `[\w.]+` \| `([\w.]+)` \| ([^|]+) \|$", PAGE.read_text(), re.M)
    table = {group: names.strip().split(", ") for group, names in rows}
    regs = (
        BACKENDS,
        SCHEDULERS,
        LAUNCHERS,
        TOOLSETS,
        COMPONENTS,
        CHECKS,
        FINAL_RULES,
        SANDBOXES,
    )
    load_plugins()  # this package's own entry point adds the `fisher` and `probe` components
    declared = {
        r.group: {
            key
            for key, obj in ({**BUILTINS, **COMPONENTS._entries} if r is COMPONENTS else r._entries).items()
            if (obj if isinstance(obj, str) else obj.__module__).startswith("qda_agent.")
        }
        for r in regs
    }
    assert table == {group: sorted(names - UNDOCUMENTED) for group, names in declared.items()}


#: Registered toolsets the user docs do not offer: `public` (a slot that refuses until it is built).
UNDOCUMENTED = {"public"}


#: The pages a user copies commands from: the README and the site (docs/*.md).
SITE = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _site_commands() -> list[tuple[str, list[str]]]:
    """Every ``qda`` / ``qda-kl`` / ``qda-agent`` command line in a ``bash`` block of a site page, split."""
    out = []
    for page in SITE:
        if page.name == "cli.md":  # generated from --help; it holds usage lines, not commands
            continue
        for body in re.findall(r"^```bash\n(.*?)^```", page.read_text(), re.S | re.M):
            for line in body.splitlines():
                line = re.sub(r"\s+#.*$", "", line).strip()
                if re.match(r"(qda|qda-kl|qda-agent) ", line):
                    out.append((page.name, shlex.split(line)))
    return out


def _subparsers(main) -> argparse.ArgumentParser:
    """The parser `main` builds, caught at its ``parse_args`` before it runs anything."""
    caught: list[argparse.ArgumentParser] = []

    def grab(parser: argparse.ArgumentParser, *_: object, **__: object) -> None:
        caught.append(parser)
        raise SystemExit(0)

    with pytest.MonkeyPatch.context() as mp, pytest.raises(SystemExit):
        mp.setattr(argparse.ArgumentParser, "parse_args", grab)
        main([])
    return caught[0]


def _usage(argv: list[str], capsys) -> tuple[list[str], str]:
    """The subcommand path of `argv` and the --help text of its deepest subcommand.

    Subcommands are read from each parser's subparsers action, not from the ``{a,b}`` of a usage line, which
    also spells an option's choices."""
    from qda_kl.cli import main as kl
    from qda_quant.cli import main as quant

    main = {"qda": quant, "qda-kl": kl, "qda-agent": cli.main}[argv[0]]
    parser, path = _subparsers(main), []
    capsys.readouterr()
    for token in argv[1:]:
        subs = next((a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction)), {})
        if token.startswith("-") or token not in subs:
            break
        parser = subs[token]
        path.append(token)
    with pytest.raises(SystemExit):
        main([*path, "--help"])
    return path, capsys.readouterr().out


@pytest.mark.parametrize(
    ("page", "argv"), _site_commands(), ids=lambda x: " ".join(x) if isinstance(x, list) else x
)
def test_site_commands_exist(page, argv, capsys):
    """Every command a user is told to paste names a real subcommand and real options: the copy-paste promise
    checked without a GPU. `cli.md` is generated from the same parsers."""
    path, usage = _usage(argv, capsys)
    assert path, f"{page}: {' '.join(argv)}: no such subcommand of {argv[0]}"
    for flag in (a.split("=")[0] for a in argv[1 + len(path) :] if a.startswith("--")):
        assert flag in usage, f"{page}: {flag} is not an option of {argv[0]} {' '.join(path)}"


def test_no_page_or_message_installs_a_qda_package_from_pypi():
    """QDA is not on PyPI, so `pip install qda-...` fails: a page, and a refusal's fix line in the packages' code,
    installs from the checkout, as the README does."""
    # `pip install qda-...`, or `install qda-...[extra]` with no command at all
    bare = re.compile(r"""pip install +(?:-\S+ +)*['"]?qda-|(?<![\w/])install +['"]?qda-[a-z]+\[""")
    sources = sorted(ROOT.glob("packages/*/src/**/*.py"))
    hits = [
        f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
        for path in [*SITE, *sources]
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if bare.search(line)
    ]
    assert not hits, "\n".join(hits)


def test_user_docs_name_no_job_sandbox_knob():
    """Isolation has one user switch, `isolation.level`; `scheduler.job_sandbox` stays an undocumented override."""
    root = Path(__file__).resolve().parents[3]
    pages = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    named = [p.relative_to(root) for p in pages if p.name != "cli.md" and "job_sandbox" in p.read_text()]
    assert named == [], f"user docs name scheduler.job_sandbox: {named}"
