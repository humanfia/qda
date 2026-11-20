"""One table of the config settings that hold a path or a command, and one check over it.

The agent's jobs write the run dir: an input that lies in it (or follows a link into it) could be a link the
agent made to anything, and one that holds the run dir would bind the judge's dir (frozen copies, texts) into
the jobs. A sandbox may also let the agent write more than the run dir (`Sandbox.writable`): a judge-side path
or command there could be swapped between the start of the run and its use in a job. `SOURCES` names every
such setting, its class and how it is used; `check` walks it and refuses what cannot be trusted.

Classes (`Kind`):

- **agent-readable** (`agent`): bound read-only into the agent's jobs or shell. Inside the run dir it is
  refused unless it lies under the run's overlay or jobs dir, which the host writes and the jobs read by
  design; one that holds the run dir is refused. It must exist.
- **judge-read** (`judge`): the host reads it for the judgement. It may not lie in or hold the run dir, nor
  lie under a directory the agent or its jobs can write. It must exist.
- **judge-exec** (`host`): the host runs it (a command, a script or its environment). As `judge`, but a path
  need not exist yet (a command may be found on `PATH`), and nothing can hold the run dir through a file.
  A module reference must not name a path at all.
- **wall** (`wall`): a path the wall hides. It grants nothing, so it is not checked.
- **no path** (`no path`): recorded so the table names every string setting; not a path, not checked.

`Row.field` is a key path from the run config; `[]` stands for every item of a list (or value of a mapping).
A plugin's path field that the table does not name is checked as agent-readable, unless its class declares
it in `Plugin.source_kinds`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel
from qda_kl.datasets import Dataset, parse_dataset

from qda_agent.configfile import PLACEHOLDER
from qda_agent.isolation.paths import SYSTEM, holds_run, in_run, python_roots

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

__all__ = ["SOURCES", "How", "Kind", "Row", "check"]


class Kind(StrEnum):
    """Who uses a setting, and so what it may not be."""

    AGENT = "agent-readable"
    JUDGE = "judge-read"
    HOST = "judge-exec"
    WALL = "wall"
    DERIVED = "no path"


class How(StrEnum):
    """What a setting's value is."""

    PATH = "path"
    PATHS = "paths"
    HUB = "hub"
    PYTHON = "python"
    PREFIX = "prefix"
    VENV = "venv"
    ARGV = "argv"
    ENV = "env"
    MODULE = "module"
    ROOTS = "roots"
    SUITE = "suite"
    DATASET = "dataset"
    NONE = "none"


A = frozenset({Kind.AGENT})
J = frozenset({Kind.JUDGE})
X = frozenset({Kind.HOST})
W = frozenset({Kind.WALL})
D = frozenset({Kind.DERIVED})
AJ = frozenset({Kind.AGENT, Kind.JUDGE})
JX = frozenset({Kind.JUDGE, Kind.HOST})


@dataclass(frozen=True)
class Row:
    """One config setting that holds a path or a command.

    Attributes:
        field: Its key path from the run config; `[]` stands for every item of a list (or value of a mapping).
        kind: Who uses it (`Kind`; `AJ` and `JX` when both do).
        how: What its value is.
        who: Who uses it, for a reader of this table.
    """

    field: str
    kind: frozenset[Kind]
    how: How
    who: str


SOURCES: tuple[Row, ...] = (
    Row("model", AJ, How.HUB, "the original model: the jobs read it, the judge generates with it"),
    Row("format", D, How.NONE, "a preset name (w4a16, ...), not a path"),
    Row("run_dir", D, How.ROOTS, "where the run writes; checked against the roots every job binds"),
    Row(
        "work_dir",
        D,
        How.ROOTS,
        "the agent's workdir outside the run dir, checked apart from inputs and binds",
    ),
    Row("task", J, How.PATH, "the task template the host reads"),
    Row("calib", A, How.PATH, "the calibration file TASK.md names"),
    Row("baselines", AJ, How.PATHS, "fallback checkpoints: the jobs and the judge read them"),
    Row("commands", X, How.MODULE, "the module:function that installs the agent's commands"),
    Row("host", X, How.MODULE, "the module:factory of the host"),
    Row("judge.anchor", J, How.HUB, "the original model a checkpoint's files are compared with"),
    Row("judge.custody_dir", D, How.ROOTS, "host checkpoint output, checked apart from inputs and binds"),
    Row("judge.serve_python", JX, How.PREFIX, "the interpreter the judge runs its jobs with"),
    Row("judge.job_env", X, How.ENV, "the environment of the judge's jobs"),
    Row("judge.metric", D, How.NONE, "a metric's name, not a path"),
    Row("judge.final_rules[].metric", D, How.NONE, "a metric's name, not a path"),
    Row("judge.suite", J, How.SUITE, "the eval suite a reading runs: the job reads what its settings name"),
    Row("judge.kl_ref", J, How.DATASET, "the KL reference"),
    Row("judge.kl_holdout_ref", J, How.PATH, "the prompt set held out from the KL reference"),
    Row("judge.kl_cmd", X, How.ARGV, "the scoring command"),
    Row("judge.kl_moe_args", X, How.ARGV, "more scorer arguments on a checkpoint with routed experts"),
    Row("judge.gencheck_cmd", X, How.ARGV, "the long-generation check command"),
    Row("judge.check_cmd", X, How.ARGV, "the serving check command"),
    Row("judge.llamacpp.server", JX, How.PATH, "the pinned llama-server the llama.cpp admission starts"),
    Row("judge.llama_cmd", X, How.ARGV, "the llama.cpp admission command"),
    Row("judge.gencheck_prompts", J, How.DATASET, "the prompt set the long-generation check reads"),
    Row("judge.format_reference", J, How.PATH, "the checkpoint of the task's format"),
    Row(
        "judge.format_map",
        J,
        How.PATH,
        "the checkpoint whose per-module formats are the task's fixed allocation",
    ),
    Row("judge.format_menu", D, How.NONE, "the task's formats by name (bf16, w8a16, w4a16), not paths"),
    Row("judge.head", D, How.NONE, "the output head's rule by name (16-bit, Q8_0, w4a16-sym-g64), not paths"),
    Row("judge.unscored", D, How.NONE, "tensor-name globs, not paths"),
    Row("judge.evaluator", JX, How.PATH, "the evaluator tree the judge copies; its jobs import it"),
    Row("contamination.guard", J, How.PATH, "what the contamination check looks for"),
    Row("agent.retry_policy", D, How.NONE, "Humanize's wait policy name"),
    Row("agent.backend.model", D, How.NONE, "the CLI's model name"),
    Row("agent.backend.effort", D, How.NONE, "the CLI's reasoning effort"),
    Row("agent.backend.account", D, How.NONE, "the CLI's Humanize account"),
    Row("agent.backend.sign_in_dir", J, How.PATH, "the CLI config dir whose sign-in the host copies"),
    Row("agent.fallbacks[].model", D, How.NONE, "a fallback CLI's model name"),
    Row("agent.fallbacks[].effort", D, How.NONE, "a fallback CLI's reasoning effort"),
    Row("agent.fallbacks[].account", D, How.NONE, "a fallback CLI's Humanize account"),
    Row("agent.fallbacks[].sign_in_dir", J, How.PATH, "a fallback's sign-in dir, copied by the host"),
    Row("sandbox.hide", W, How.PATHS, "more directories the bwrap wall hides"),
    Row("scheduler.readable", A, How.PATHS, "more read-only paths of the job sandbox"),
    Row("toolset.python", A, How.PREFIX, "the agent's job interpreter (the host runs it for `qda plan`)"),
    Row("toolset.qda", A, How.ARGV, "the agent's qda command"),
    Row("toolset.env", A, How.ENV, "more environment of every agent job"),
    Row("toolset.components[].sets", A, How.PATHS, "named read-only datasets of the jobs"),
    Row("toolset.components[].python", A, How.VENV, "a component's job interpreter (a virtualenv)"),
    Row("toolset.components[].package", A, How.PATH, "a package tree a component copies"),
    Row("python_roots()", A, How.ROOTS, "the host interpreter's roots, which every job binds"),
    Row("SYSTEM", A, How.ROOTS, "the system directories every job reads"),
)

#: The fields the table names, for the plugin walk below.
_TABLE = frozenset(row.field for row in SOURCES)


def check(cfg: RunConfig) -> None:
    """Refuse an input, command or environment the agent's run dir or sandbox makes untrustworthy.

    Walks `SOURCES`, then the path fields of every plugin instance that the table does not name.

    Raises:
        ValueError: A setting is of a class and location the table does not allow (see the module docstring).
    """
    run = Path(os.path.realpath(cfg.run_dir))
    _roots(cfg, run)
    writable = _writable(cfg)
    jobs = Path(os.path.realpath(cfg.run_dir / "jobs"))
    overlay = cfg.run.overlay
    allow = (jobs,) if overlay is None else (Path(os.path.realpath(overlay)), jobs)
    for row in SOURCES:
        for indices, value in _walk(cfg, row.field.split(".")):
            _one(row, _label(row, indices), value, cfg, run, writable, allow)
    _plugin_fields(cfg, run, writable, allow)
    _bound(cfg, run, writable, allow)
    _apart(cfg)
    if cfg.judge.custody_dir is not None:
        from .judge.custody import validate

        inputs = [
            p
            for row in SOURCES
            if Kind.JUDGE in row.kind
            for _, v in _walk(cfg, row.field.split("."))
            for p in _judge_paths(row, v)
        ]
        _, binds = cfg.read_paths()
        validate(cfg.judge.custody_dir, [run, *_agent_paths(cfg), *binds, *inputs, *(writable or [])])
    if cfg.work_dir is not None:
        _work_dir(cfg, run, writable)


def _work_dir(cfg: RunConfig, run: Path, writable: list[Path] | None) -> None:
    """Place the agent's `work_dir` by `judge.custody_dir`'s rules: trusted ancestors, apart from the rest.

    Its jobs write there, so it may hold no input of any kind and lie in none (the judge's private ones
    included), nor overlap the run dir, the judge's custody or a bind. That it is fresh is checked when the
    run is launched (`qda_agent.cli`), not here: the run reloads its own config after making it.
    """
    from .judge.custody import validate

    inputs = [
        p
        for row in SOURCES
        if row.how is not How.ROOTS
        for _, v in _walk(cfg, row.field.split("."))
        for p in _judge_paths(row, v)
    ]
    others = [run, *([cfg.judge.custody_dir] if cfg.judge.custody_dir is not None else [])]
    # `_agent_paths` holds the binds' sources; their destinations lie in the workdir by design
    forbidden = [*others, *_agent_paths(cfg), *inputs, *(writable or [])]
    validate(cfg.workdir, forbidden, setting="work_dir", fresh=False)


#: Where a value sits in a list or mapping the walk crossed: `(field name, index, item)`.
_Index = tuple[tuple[str, int, Any], ...]


def _walk(node: Any, parts: list[str], indices: _Index = ()) -> Iterator[tuple[_Index, Any]]:
    """Yield `(indices, value)` for every value `parts` selects from `node`; `[]` selects every item."""
    if not parts:
        yield indices, node
        return
    head, *rest = parts
    many = head.endswith("[]")
    name = head.removesuffix("[]")
    child = getattr(node, name, None)
    if not many:
        if child is not None:
            yield from _walk(child, rest, indices)
        return
    items = list(child.values()) if isinstance(child, dict) else list(child or [])
    for i, item in enumerate(items):
        yield from _walk(item, rest, (*indices, (name, i, item)))


def _label(row: Row, indices: _Index) -> str:
    """Name the setting a value came from, for the messages: a component by its key, a list by its index."""
    for name, _, item in indices:
        if name == "components":
            return f"toolset ({getattr(item, 'name', '?')})"
    out: list[str] = []
    for part in row.field.split("."):
        if part.endswith("[]"):
            name = part.removesuffix("[]")
            hit = next((i for n, i, _ in indices if n == name), None)
            out.append(name if hit is None else f"{name}[{hit}]")
        else:
            out.append(part)
    return ".".join(out)


def _one(
    row: Row,
    label: str,
    value: Any,
    cfg: RunConfig,
    run: Path,
    writable: list[Path] | None,
    allow: tuple[Path, ...],
) -> None:
    """Check one value of one setting by the setting's class and shape."""
    if value is None or row.how in (How.NONE, How.ROOTS):
        return
    if row.how is How.MODULE:
        _module(label, value)
    elif row.how is How.ENV:
        for entry in _values(value):
            for token in str(entry).split(os.pathsep):
                if token:
                    _token(row, label, token, cfg, run, writable)
    elif row.how is How.SUITE:
        _suite(row, label, value, cfg, run, writable)
    elif row.how is How.DATASET:  # by where its data is: a plugin's settings need not name that
        _path(row, label, parse_dataset(value, label).local(), cfg, run, writable, allow, how=How.PATH)
    elif row.how is How.ARGV:
        program = True
        for token in value:
            _token(row, label, token, cfg, run, writable, program=program)
            program = program and _wrapper(token)
    elif row.how is How.HUB:
        _hub(row, label, value, cfg, run, writable)
    elif row.how is How.PATHS:
        for item in _values(value):
            _path(row, label, item, cfg, run, writable, allow, how=How.PATH)
    else:
        _path(row, label, value, cfg, run, writable, allow, how=row.how)


def _where(
    row: Row,
    label: str,
    path: Path,
    cfg: RunConfig,
    run: Path,
    writable: list[Path] | None,
    allow: tuple[Path, ...],
) -> None:
    """Refuse a resolved path that is inside the run dir, holds it, or lies where the agent can write it."""
    if row.kind in (W, D):
        return
    named = f"{label} ({path})"
    if in_run(path, run, allow=() if Kind.JUDGE in row.kind else allow):
        why = (
            "the host reads or runs it, and the agent's jobs write there"
            if Kind.JUDGE in row.kind
            else "which the agent's jobs write"
        )
        raise ValueError(
            f"{named}: inside run_dir {cfg.run_dir} (or a link into it), {why}; give a path outside the run dir"
        )
    if (Kind.HOST not in row.kind or Kind.JUDGE in row.kind) and path.exists() and holds_run(path, run):
        why = (
            "so the host would read the run dir's files through it"
            if Kind.JUDGE in row.kind
            else "so the agent's jobs would read the judge's files"
        )
        raise ValueError(
            f"{named}: holds run_dir {cfg.run_dir}, {why}; give a path that does not contain the run dir"
        )
    if writable is not None and (Kind.JUDGE in row.kind or Kind.HOST in row.kind):
        for box in writable:
            if path.is_relative_to(box):
                raise ValueError(
                    f"{named}: under {box}, which the agent or its jobs can write; keep it where they cannot "
                    "change it"
                )


def _path(
    row: Row,
    label: str,
    value: Any,
    cfg: RunConfig,
    run: Path,
    writable: list[Path] | None,
    allow: tuple[Path, ...],
    *,
    how: How,
) -> None:
    """Check one path setting: where it is, that it is there, and for an interpreter what it is."""
    if _skip(value) or Kind.WALL in row.kind:
        return
    if how in (How.PYTHON, How.PREFIX, How.VENV):
        _interpreter(row, label, value, cfg, run, writable, allow, how)
        return
    path = _real(value)
    _where(row, label, path, cfg, run, writable, allow)
    if not path.exists():
        raise ValueError(f"{label} ({value}): does not exist")
    _links(row, label, path, run, writable)


#: The trees the host copies into the agent's workdir: every link in one must stay inside it.
_COPIED = frozenset({"toolset.components[].package"})


def _links(row: Row, label: str, path: Path, run: Path, writable: list[Path] | None) -> None:
    """Refuse a link inside a tree the host copies or reads that leads to agent-made or host files.

    A judge-read tree is copied (`evaluator`) or read (`format_reference`) by the host while the tree stays
    where it is; a link in it that leads into the run dir or a writable directory would carry the agent's
    file into the judgement. A tree the toolset copies into the workdir must be self-contained: there the
    link itself would carry any host file into the agent's reach. A hub snapshot's links into its cache's
    blobs point at host files outside the run dir, so they are left alone.
    """
    copied = row.field in _COPIED
    judged = Kind.JUDGE in row.kind and Kind.AGENT not in row.kind
    if not (copied or judged) or not path.is_dir():
        return
    top = Path(os.path.realpath(path))
    for root, dirs, files in os.walk(top, followlinks=False):
        for name in (*dirs, *files):
            link = Path(root, name)
            if not link.is_symlink():
                continue
            real = Path(os.path.realpath(link))
            if real.is_relative_to(top):
                continue
            if copied:  # a tree the toolset copies: every outside link is a host file in the agent's reach
                raise ValueError(
                    f"{label} ({path}): holds a link {link} that leads outside it ({real}); the toolset copies "
                    "the tree into the agent's workdir, so the link would carry host files there; keep the tree "
                    "self-contained"
                )
            # a tree the judge reads (the two cases are disjoint, see the guard above)
            box = next((b for b in writable or () if real.is_relative_to(b)), None)
            if in_run(real, run, allow=()) or box is not None:
                where = f"the run dir {run}" if box is None else f"{box}, which the agent can write"
                raise ValueError(
                    f"{label} ({path}): holds a link {link} that leads into {where}; the host reads or copies "
                    "the tree, so the link would carry the agent's own file into the judgement; keep the "
                    "tree self-contained"
                )


def _interpreter(
    row: Row,
    label: str,
    value: Any,
    cfg: RunConfig,
    run: Path,
    writable: list[Path] | None,
    allow: tuple[Path, ...],
    how: How,
) -> None:
    """Check an interpreter by its installation, not its link: a venv's `bin/python` points at its base."""
    given = Path(Path(str(value)).expanduser())  # as configured: its `bin/` knows the venv
    home = Path(os.path.realpath(given.parent.parent))  # the installation a job would load code from
    _where(row, label, home, cfg, run, writable, allow)
    # what the link points at as well as where it lives: a `bin/python` the agent can replace would run its
    # file while its installation stays where the check looked
    target = Path(os.path.realpath(given))
    _where(row, label, target, cfg, run, writable, allow)
    if not given.exists():
        raise ValueError(f"{label} ({value}): does not exist")
    if how is How.PYTHON:
        return
    if not given.is_file():
        raise ValueError(f"{label} ({value}): not an interpreter (not a file)")
    if not home.is_dir():
        raise ValueError(f"{label} ({value}): its installation {home} is not a directory")
    if how is How.VENV and not (home / "pyvenv.cfg").is_file():
        raise ValueError(
            f"{label} ({value}): not a virtualenv's interpreter (no pyvenv.cfg), and its whole installation "
            "would be bound into the agent's jobs; give a virtualenv's interpreter"
        )


def _hub(row: Row, label: str, value: Any, cfg: RunConfig, run: Path, writable: list[Path] | None) -> None:
    """Check a path or hub id: a path is resolved like any other; a bare id is left to the hub client."""
    if _skip(value):
        return
    text = str(value)
    path = _real(text)
    _where(row, label, path, cfg, run, writable, ())
    if _path_like(text) and not path.exists():
        raise ValueError(f"{label} ({text}): does not exist (a path, not a hub id)")
    if path.is_dir():
        _links(row, label, path, run, writable)


def _suite(row: Row, label: str, value: Any, cfg: RunConfig, run: Path, writable: list[Path] | None) -> None:
    """Check every string of a parsed suite as a judge input, by its value after `KEY=` or between `:`.

    Settings may be free-form, so a string that names no path (`</think>`, `mtp`) passes as a command token does.
    The job runs in the judge's dir, inside the run dir, so a relative path that climbs reaches the agent's files.
    """
    for where, text in _strings(label, _parsed_suite(value)):
        for token in text.split(os.pathsep):
            named = _value(token)
            if ".." in Path(named).parts and not Path(named).expanduser().is_absolute():
                raise ValueError(
                    f"{where} ({named}): a relative path out of the job's own directory, the judge's dir in the "
                    "run dir; give an absolute path outside the run dir"
                )
            _token(row, where, token, cfg, run, writable)
            if _path_like(named) and _real(named).is_dir():
                _links(row, where, _real(named), run, writable)


def _parsed_suite(value: dict[str, Any]) -> BaseModel:
    """`judge.suite` as the job will build it: its nested plugins are models, with all their defaults."""
    from qda_kl.eval.suites import parse_suite  # the evaluator's own parser: one reading of a suite

    return parse_suite(value, "judge.suite")


def _strings(label: str, value: Any, *, public: bool = True) -> Iterator[tuple[str, str]]:
    """Every string or path a model holds, nested models and containers included, with where it sits.

    A dataset also counts by where its data is: a plugin's settings need not name that.

    Args:
        label: Where `value` sits.
        value: The model, container or string.
        public: Whether a public dataset counts. It does for what the agent could change, and does not for what
            the agent may read.
    """
    if isinstance(value, Dataset):
        if value.public and not public:
            return
        yield label, str(value.local())
    if isinstance(value, BaseModel):
        items: Iterable[tuple[Any, Any]] = value  # its fields and extras, as the objects they are
    elif isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        if isinstance(value, (str, PurePath)):
            yield label, str(value)
        return
    for key, item in items:
        yield from _strings(f"{label}.{key}", item, public=public)


#: The slots that hold the agent's own content: a frozen copy of its submission, and its results.
_AGENT_SLOTS = frozenset({"{ckpt}", "{out}"})

#: A token that only sets up the program after it: an environment variable or a `key=value` argument.
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _token(
    row: Row,
    label: str,
    token: Any,
    cfg: RunConfig,
    run: Path,
    writable: list[Path] | None,
    *,
    program: bool = False,
) -> None:
    """Check one argv token or environment segment: a path-like one is resolved, a slot-prefixed one is not.

    `program` marks the token the host (or the job) runs: the agent's own content there is refused. A token
    that is `KEY=value` (an `env` wrapper, `--opt=value`, `--set key=value`) is checked by its value.
    """
    if not isinstance(token, str) or _skip(token):
        return
    head, sep, _ = token.partition("=")
    if sep and _is_key(head):
        token = _value(token)
        program = False  # an assignment's value is an argument, never the program
        if not token:
            return
        if head.isupper() and os.pathsep in token:  # a list of paths, as `PYTHONPATH` holds
            for part in token.split(os.pathsep):
                if part:
                    _token(row, label, part, cfg, run, writable)
            return
    if program and token in _AGENT_SLOTS:
        raise ValueError(
            f"{label} ({token}): the slot holds the agent's own content (a frozen copy or its results), and the "
            "host would run it; name a program the agent cannot change"
        )
    if token.startswith("{"):
        name, closed, rest = token.partition("}")
        slot = name + closed
        if closed and slot in _AGENT_SLOTS and rest:
            raise ValueError(
                f"{label} ({token}): a path under {slot}, which holds the agent's own content; the host would "
                "run or read the agent's file through it; pass the slot itself to a judge tool that checks it"
            )
        if closed and ".." in Path(rest).parts:
            raise ValueError(
                f"{label} ({token}): climbs out of its slot; a slot is a path the judge makes, and the agent "
                "must not be able to redirect it"
            )
        return
    if _path_like(token):
        if not Path(token).expanduser().is_absolute():
            raise ValueError(
                f"{label} ({token}): a relative path; the command runs from the job's own directory, not where "
                "the config was read, so it would name whatever is there; give an absolute path"
            )
        _where(row, label, _real(token), cfg, run, writable, ())


def _is_key(text: str) -> bool:
    """Whether `text` names an option (`--flag`), an environment variable, or a `key` of a `key=value`."""
    return text.startswith("-") or _KEY.fullmatch(text) is not None


def _value(token: str) -> str:
    """The value of `token`: `--a=b=path` and `--set key=path` yield `path`."""
    for _ in range(4):
        key, sep, tail = token.partition("=")
        if not sep or not _is_key(key):
            return token
        token = tail
    return token


def _wrapper(token: Any) -> bool:
    """Whether an argv token only sets up the program that follows it (`env`, its flags, `KEY=value`)."""
    if not isinstance(token, str) or not token:
        return False
    if token == "env" or token.startswith("-"):
        return True
    key, sep, _ = token.partition("=")
    return bool(sep) and _is_key(key)


def _module(label: str, value: Any) -> None:
    """Refuse a module reference that names a path: the host imports it, so it must be code the agent cannot change."""
    if not isinstance(value, str) or _skip(value):
        return
    if os.sep in value or (os.altsep is not None and os.altsep in value):
        raise ValueError(
            f"{label} ({value}): a path, not a module:function reference; the host imports it, so it must "
            "name code the agent cannot change"
        )


def _roots(cfg: RunConfig, run: Path) -> None:
    """Refuse a run dir inside a Python root or a system directory, and a root inside the run dir."""
    for root in python_roots():
        path = Path(os.path.realpath(root))
        if run.is_relative_to(path):
            raise ValueError(
                f"python root {path}: holds run_dir {cfg.run_dir}, and every job binds the host's Python roots; "
                "use a run dir outside the installation the host runs from"
            )
        if path.is_relative_to(run):
            raise ValueError(
                f"python root {path}: inside run_dir {cfg.run_dir}, and every job binds it; use a run dir "
                "outside the installation the host runs from"
            )
    for name in SYSTEM:
        path = Path(os.path.realpath(name))
        if run.is_relative_to(path):
            raise ValueError(
                f"run_dir {cfg.run_dir} is under {path}, a system directory every job reads; use a run dir of "
                "the user's own"
            )


def _writable(cfg: RunConfig) -> list[Path] | None:
    """The directories outside the run dir the agent or its jobs can write; None when that cannot be named."""
    if getattr(cfg.scheduler, "job_sandbox", "bwrap") == "none":  # jobs on the host: anything the user can
        return None
    dirs = cfg.sandbox.writable()
    return None if dirs is None else [Path(os.path.realpath(d)) for d in dirs]


def _plugin_fields(cfg: RunConfig, run: Path, writable: list[Path] | None, allow: tuple[Path, ...]) -> None:
    """Check the path fields of every plugin instance that `SOURCES` does not name (agent-readable by default)."""
    instances = [
        ("sandbox", cfg.sandbox),
        ("scheduler", cfg.scheduler),
        ("launcher", cfg.launcher),
        ("toolset", cfg.toolset),
        *(("toolset.components[]", c) for c in cfg.toolset.components),
        ("contamination", cfg.contamination),
        *(("judge.final_rules[]", rule) for rule in cfg.judge.final_rules),
        ("agent.backend", cfg.agent.backend),
        *(("agent.fallbacks[]", fallback) for fallback in cfg.agent.fallbacks),
    ]
    for prefix, plugin in instances:
        declared = getattr(type(plugin), "source_kinds", {})
        for name, info in type(plugin).model_fields.items():
            how = _path_how(info.annotation)
            if how is None or f"{prefix}.{name}" in _TABLE:
                continue
            kind = _declared(type(plugin).__name__, name, declared.get(name))
            row = Row(f"{prefix}.{name}", kind, how, "a plugin field the table does not name")
            label = f"toolset ({plugin.name})" if prefix == "toolset.components[]" else f"{prefix}.{name}"
            loose = not _mentions_any(info.annotation, (Path, PurePath))  # a `str` that may be a path
            for value in _values(getattr(plugin, name, None)):
                if loose and not _loose(value):
                    continue
                _path(row, label, value, cfg, run, writable, allow, how=How.PATH if loose else how)


#: The row the paths the jobs bind are checked as (`RunConfig.read_paths`).
_BOUND = Row("read_paths()", A, How.PATHS, "a path the agent's jobs bind read-only")

#: The judge's private inputs: data the agent must never read. The model, the baselines, the format
#: reference and the evaluator are public (the jobs read them by design); an interpreter may be shared too.
_JUDGE_DATA = frozenset(
    {
        "judge.kl_ref",
        "judge.kl_holdout_ref",
        "judge.gencheck_prompts",
        "judge.suite",
        "contamination.guard",
    }
)


def _bound(cfg: RunConfig, run: Path, writable: list[Path] | None, allow: tuple[Path, ...]) -> None:
    """Check the paths the jobs actually bind (`read_paths()`), not only the settings the table names.

    A component's `readable()` entry, a bind source or a baseline the table does not name arrives here: the
    scheduler binds exactly this list, so a path in it must be as trustworthy as any the table knows.
    """
    readable, binds = cfg.read_paths()
    for i, path in enumerate(readable):
        _path(_BOUND, f"read_paths().readable[{i}]", path, cfg, run, writable, allow, how=How.PATH)
    for dst, src in binds.items():
        _path(_BOUND, f"read_paths().binds[{dst}]", src, cfg, run, writable, allow, how=How.PATH)


def _apart(cfg: RunConfig) -> None:
    """Refuse one of the judge's private inputs under a path the agent's jobs read.

    Every readable path is bound into the agent's jobs, so a private file inside one is the agent's to read
    (and, where it can write, to replace before the judgement). The model, the baselines, the format
    reference and the evaluator are public (the jobs read them by design), and an interpreter may be shared
    with the serving component, so none of them is checked here.
    """
    check_private_reads(cfg.judge, cfg.contamination, _agent_paths(cfg))


def private_paths(judge: Any, contamination: Any) -> Iterator[tuple[str, Path]]:
    """Private inputs from the source table, shared by config validation and the host's final bind checks."""
    for row in SOURCES:
        if row.field not in _JUDGE_DATA:
            continue
        owner, _, field = row.field.partition(".")
        for indices, value in _walk(judge if owner == "judge" else contamination, field.split(".")):
            label = _label(row, indices)
            for path in _judge_paths(row, value):
                yield label, path


def check_private_reads(judge: Any, contamination: Any, readable: Iterable[Path]) -> None:
    """Refuse any private input exposed by the completed set of agent read binds, resolving aliases."""
    roots = [Path(os.path.realpath(path)) for path in readable]
    for label, path in private_paths(judge, contamination):
        if box := next((root for root in roots if path.is_relative_to(root)), None):
            where = (
                "the agent's jobs read it" if path == box else f"inside {box}, which the agent's jobs read"
            )
            raise ValueError(
                f"{label} ({path}): {where}; a judge input there is the agent's to read (and to replace "
                "where it can write); keep the judge's files apart from what the agent reads"
            )


def _judge_paths(row: Row, value: Any) -> list[Path]:
    """The paths one value of a judge-read row names, without placeholders or a bare hub id."""
    if value is None or _skip(value):
        return []
    if row.how is How.SUITE:
        strings = _strings("", _parsed_suite(value), public=False)
        tokens = (_value(t) for _, text in strings for t in text.split(os.pathsep))
        return [_real(token) for token in tokens if _path_like(token)]
    if row.how is How.DATASET:
        dataset = parse_dataset(value, row.field)
        return [] if dataset.public or _skip(dataset.local()) else [_real(dataset.local())]
    if row.how is How.HUB:
        text = str(value)
        path = _real(text)
        return [path] if _path_like(text) or path.exists() else []
    return [_real(item) for item in _values(value)]


def _agent_paths(cfg: RunConfig) -> list[Path]:
    """Every path the agent's jobs read: `read_paths()`, the interpreter's roots, and its readable settings."""
    found: list[Path] = []

    def add(value: Any) -> None:
        if value is None or _skip(value):
            return
        path = _real(value)
        if path not in found:
            found.append(path)

    readable, binds = cfg.read_paths()
    for path in (*readable, *binds.values()):
        add(path)
    if Path(cfg.model).expanduser().exists():  # the jobs read the model; `model` is not an A row (it is AJ)
        add(cfg.model)
    if cfg.toolset.python is not None:
        add(cfg.toolset.python.parent.parent)  # the interpreter's installation, as `isolation` counts it
    for root in (*python_roots(), *SYSTEM):
        add(root)
    for row in SOURCES:
        if Kind.AGENT not in row.kind or Kind.JUDGE in row.kind or row.how not in (How.PATH, How.PATHS):
            continue
        for _, value in _walk(cfg, row.field.split(".")):
            for item in _values(value):
                add(item)
    return found


_DECLARED_KINDS = {"agent": A, "judge": J, "host": X}


def _declared(who: str, field: str, value: str | None) -> frozenset[Kind]:
    """The class a plugin declared for one of its path fields, or agent-readable when it declared none."""
    if value is None:
        return A
    if value not in _DECLARED_KINDS:
        raise ValueError(
            f"{who}.source_kinds[{field!r}] = {value!r}: expected one of {', '.join(_DECLARED_KINDS)}"
        )
    return _DECLARED_KINDS[value]


def _path_how(annotation: Any) -> How | None:
    """`How.PATHS` for a list or mapping of paths, `How.PATH` for one, None for anything else.

    A `str` field is taken as holding a path too: a plugin may hand its path back as a string, and
    `read_paths()` returns what its `readable()` gives. Such a field's values are checked only when they
    read like a path (`_loose`), so a name or a format is not mistaken for one.
    """
    if _mentions_any(annotation, (Path, PurePath, str)):
        return How.PATHS if _container(annotation) else How.PATH
    return None


def _mentions(node: Any, cls: type) -> bool:
    return node is cls or any(_mentions(arg, cls) for arg in get_args(node))


def _mentions_any(node: Any, classes: tuple[type, ...]) -> bool:
    return any(_mentions(node, cls) for cls in classes)


def _loose(value: Any) -> bool:
    """Whether a `str` field's value names a path: it reads like one, or one exists where it resolves."""
    text = str(value)
    return _path_like(text) or (os.sep in text and _real(text).exists())


def _container(node: Any) -> bool:
    if get_origin(node) in (list, dict):
        return True
    return any(_container(arg) for arg in get_args(node))


def _values(value: Any) -> list[Any]:
    """The values a path setting holds: a mapping's values, a container's items, or the value itself.

    `None` and `<...>` placeholders yield nothing, so every caller checks only what it can open.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        items: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:  # a `str`, a `Path` or any other single value
        items = [value]
    return [item for item in items if item is not None and not _skip(item)]


def _real(value: Any) -> Path:
    """A path setting as the host would open it: `~` expanded and every link followed."""
    return Path(os.path.realpath(Path(str(value)).expanduser()))


def _skip(value: Any) -> bool:
    """Whether a value is a `<...>` placeholder, which the loader refuses by key before a run starts."""
    return not str(value) or PLACEHOLDER.fullmatch(str(value)) is not None


def _path_like(text: str) -> bool:
    """Whether a string names a path (so its absence is an error); a bare hub id like `org/model` is not."""
    return text.startswith(("/", "~", "./", "../")) or text in (".", "..")
