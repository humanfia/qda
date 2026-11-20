"""`qda_agent.run_sources`: one table, one check, and nothing it does not name.

`test_every_row_is_checked_where_its_class_says` walks `SOURCES` and plants each setting inside the run dir,
over it, nowhere (and, as a control, elsewhere), expecting what the row's class and shape say. A row this test
cannot plant fails in `_put`: the table and the test grow together.

`test_every_string_or_path_setting_has_a_row` walks the config's models and every registered plugin (the
scripted test backend included) so that a new string or path setting fails until the table names it, and the
reverse test fails a row that names no setting. The rest pin the shapes the check reads (argv and env tokens,
slots) and what each sandbox says the agent can write outside the run dir.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePath
from typing import Any, Literal, get_args

import pytest
import testkit  # noqa: F401  (registers the scripted backend the reflection walks)
from qda_agent.agents import BACKENDS, AgentConfig
from qda_agent.components import COMPONENTS, Component
from qda_agent.config import RunConfig
from qda_agent.judge.config import JudgeSettings, LlamaCpp
from qda_agent.judge.contamination import CHECKS
from qda_agent.judge.final_rules import FINAL_RULES
from qda_agent.launchers import LAUNCHERS
from qda_agent.run_sources import SOURCES, How, Kind, Row, check
from qda_agent.sandbox_settings import SANDBOXES, Bwrap, NoSandbox, Sandbox
from qda_agent.sched.config import SCHEDULERS
from qda_agent.toolsets import TOOLSETS

#: Rows that name no config field: a run dir is checked against them apart.
SYNTHETIC = {"python_roots()", "SYSTEM"}
WALK = tuple(
    row for row in SOURCES if row.field not in SYNTHETIC | {"run_dir", "work_dir", "judge.custody_dir"}
)
INTERPRETERS = (How.PREFIX, How.VENV, How.PYTHON)
#: The shapes whose value must be there: a token may name a command found later on `PATH`.
EXISTS = frozenset({How.PATH, How.PATHS, How.DATASET, How.PREFIX, How.VENV, How.PYTHON, How.HUB})


def _base(tmp_path: Path, **extra: Any) -> RunConfig:
    """A validated run with every plugin that has a table row (the run dir is `tmp_path`'s `run`)."""
    guard = tmp_path / "judge-private" / "guard.txt"  # the ngram check needs one at load
    guard.parent.mkdir(exist_ok=True)
    guard.write_text("0" * 64 + "\n")
    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "sandbox": extra.pop("sandbox", "bwrap"),
        "toolset": {
            "name": "qda",
            "components": [{"name": "data"}, {"name": "serve"}, {"name": "source"}, {"name": "wiki"}],
        },
        "contamination": {"name": "ngram", "guard": str(guard)},
        "agent": {"backend": {"name": "claude"}, "fallbacks": [{"name": "claude"}]},
        "judge": {
            "final_rules": ["best_metric"],
        },
        **extra,
    }
    return RunConfig.model_validate(raw)


def _put(cfg: RunConfig, field: str, value: Any) -> None:
    """Set one table row's value on `cfg`; `[]` sets the field of every item the base holds.

    Asserts the base has the setting: a row the test cannot plant must fail here, not go unchecked.
    """
    if field.startswith("judge.llamacpp.") and cfg.judge.llamacpp is None:
        cfg.judge.llamacpp = LlamaCpp.model_construct()  # a vLLM base has none: plant the row's one field
    nodes: list[Any] = [cfg]
    parts = field.split(".")
    for i, part in enumerate(parts):
        many = part.endswith("[]")
        name = part.removesuffix("[]")
        last = i == len(parts) - 1
        if many:
            held = [getattr(node, name) for node in nodes]
            nodes = [item for h in held for item in (h.values() if isinstance(h, dict) else h)]
        elif last:
            set_: list[Any] = []
            for node in nodes:
                if name in type(node).model_fields:
                    setattr(node, name, value)
                    set_.append(node)
            assert set_, f"the base has no {field!r} to plant: extend this test"
        else:
            nodes = [getattr(node, name) for node in nodes]


def _venv(home: Path) -> Path:
    """A virtualenv's interpreter at `home/bin/python`, with the `pyvenv.cfg` the check looks for."""
    given = home / "bin" / "python"
    given.parent.mkdir(parents=True, exist_ok=True)
    given.write_text("#!/usr/bin/env python3\n")
    (home / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return given


def _value(row: Row, where: str, tmp_path: Path) -> Any:
    """One plant of a row's setting: inside the run dir, over it, nowhere, elsewhere, or under a writable dir."""
    run = tmp_path / "run"
    if row.how in INTERPRETERS:
        home = {
            "inside": run / "v",
            "holding": tmp_path,
            "absent": tmp_path / "gone",
            "elsewhere": tmp_path / "ok",
            "writable": tmp_path / "shared" / "v",
        }[where]
        path = home / "bin" / "python"
        if where != "absent":
            _venv(home)
    else:
        path = {
            "inside": run / "planted" / "x",
            "holding": tmp_path,
            "absent": tmp_path / "gone" / "x",
            "elsewhere": tmp_path / "ok" / "x",
            "writable": tmp_path / "shared" / "x",
        }[where]
        if where in ("elsewhere", "writable"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    if row.how is How.MODULE:
        return "qda_agent.commands:install_bin" if where == "elsewhere" else f"{path}:install"
    if row.how is How.HUB:
        return "org/model" if where == "elsewhere" else str(path)
    if row.how is How.ENV:
        return {"K": "value"} if where == "elsewhere" else {"K": str(path)}
    if row.how is How.ARGV:
        return ["job"] if where == "elsewhere" else [f"--flag={path}"]
    if row.how is How.SUITE:
        return {"kl": {"corpus": {"file": {"path": str(path)}}}}
    if row.how in INTERPRETERS:
        return path  # a `Path`: `read_paths()` reads its installation from it
    if row.how is How.PATHS:
        return {"k": path} if row.field.endswith(".sets") else [path]
    return str(path)


def _refused(row: Row, where: str) -> bool:
    """Whether the row's class and shape refuse the plant (see the module docstring of `run_sources`)."""
    if row.kind in (frozenset({Kind.WALL}), frozenset({Kind.DERIVED})):
        return False
    if row.how is How.MODULE:
        return where != "elsewhere"  # a path in a module reference is refused wherever it is
    if where == "elsewhere":
        return False
    if where == "inside":
        return True  # the agent's jobs write the run dir, for every class but wall and no-path
    if where == "holding":
        return Kind.HOST not in row.kind or Kind.JUDGE in row.kind  # a judge-exec command holds nothing
    return row.how in EXISTS  # absent: a command may be found later on `PATH`


@pytest.mark.parametrize("row", WALK, ids=[row.field for row in WALK])
@pytest.mark.parametrize("where", ["inside", "holding", "absent", "elsewhere"])
def test_every_row_is_checked_where_its_class_says(tmp_path, row, where):
    cfg = _base(tmp_path)
    _put(cfg, row.field, _value(row, where, tmp_path))
    if _refused(row, where):
        with pytest.raises(ValueError):
            check(cfg)
    else:
        check(cfg)


WRITABLE = tuple(
    row
    for row in WALK
    if row.kind not in (frozenset({Kind.WALL}), frozenset({Kind.DERIVED}))
    and (Kind.JUDGE in row.kind or Kind.HOST in row.kind)
    and row.how is not How.MODULE  # refused for its shape, not its place
)


@pytest.mark.parametrize("row", WRITABLE, ids=[row.field for row in WRITABLE])
def test_a_judge_or_host_setting_the_agent_can_write_is_refused(tmp_path, monkeypatch, row):
    """A bwrap TMPDIR grants host writes; unconfined jobs leave no finite writable set."""
    cfg = _base(tmp_path)
    with tempfile.TemporaryDirectory(prefix="qda-writable-", dir=Path.home()) as shared:
        monkeypatch.setenv("TMPDIR", shared)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        _put(cfg, row.field, _value(row, "writable", Path(shared)))
        with pytest.raises(ValueError, match="which the agent or its jobs can write"):
            check(cfg)
        cfg.scheduler.job_sandbox = "none"  # jobs on the host: the agent could write anything the user can
        check(cfg)


def test_a_suites_nested_settings_are_judge_inputs(tmp_path):
    """Whatever depth a suite holds a path at; a string that only looks path-ish stays legal."""
    work = tmp_path / "run" / "work"

    def scorer(serve: dict[str, Any]) -> dict[str, Any]:
        return {"kl": {"scorer": {"topk": serve}}}

    def checked(suite: dict[str, Any], **extra: Any) -> None:
        cfg = _base(tmp_path, **extra)
        cfg.judge.suite = suite
        check(cfg)

    for suite, said in (
        ({"kl": {"corpus": {"file": {"path": str(work / "eval.jsonl")}}}}, "inside run_dir"),
        ({"kl": {"corpus": {"file": {"path": "results/../../work/eval.jsonl"}}}}, "relative path"),
        ({"kl": {"corpus": {"file": {"path": "../work/eval.jsonl"}}}}, "relative path"),
        (scorer({"serve_env": {"PYTHONPATH": f"/opt/a:{work}/b"}}), "inside run_dir"),
        (scorer({"serve_args": {"download_dir": str(work)}}), "inside run_dir"),
        (scorer({"serve_args": {"nested": [{"dir": f"--dir={work}"}]}}), "inside run_dir"),
    ):
        with pytest.raises(ValueError, match=said):
            checked(suite)
    ok = {
        "kl": {
            "sampling": {"stop": ["</think>", "a/b"]},
            "corpus": {"file": {"path": str(tmp_path / "e.jsonl")}},
        }
    }
    checked(ok)

    public = tmp_path / "baseline"
    public.mkdir()
    with pytest.raises(ValueError, match="inside .*which the agent's jobs read"):
        checked({"kl": {"corpus": {"file": {"path": str(public / "eval.jsonl")}}}}, baselines=[str(public)])

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "link").symlink_to(work)
    with pytest.raises(ValueError, match="leads into"):
        checked(scorer({"serve_args": {"data": str(tree)}}))


def test_each_sandbox_says_what_the_agent_can_write_outside_the_run_dir(monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    assert Bwrap().writable() == []
    assert NoSandbox().writable() is None and Sandbox(name="base").writable() is None


def test_bwrap_names_the_tmpdir_the_wall_shows_and_not_a_private_one(tmp_path, monkeypatch):
    """The wall binds the fence's `$TMPDIR` writable; `/tmp`, `/var/tmp` and the home are replaced or hidden."""
    # `tmp_path` may be under the real home with another `--basetemp`, which `bwrap` hides
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("TMPDIR", str(shared))
    assert Bwrap().writable() == [Path(os.path.realpath(shared))]
    for private in ("/tmp", "/var/tmp", str(Path.home()), "relative/tmp"):
        monkeypatch.setenv("TMPDIR", private)
        assert Bwrap().writable() == []
    monkeypatch.delenv("TMPDIR")
    assert Bwrap().writable() == []


def test_a_judge_command_under_the_shown_tmpdir_is_refused(tmp_path, monkeypatch):
    # `tmp_path` may be under the real home with another `--basetemp`, which `bwrap` hides
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    shared = tmp_path / "shared"
    shared.mkdir()
    score = shared / "score"
    score.write_text("#!/bin/sh\n")
    monkeypatch.setenv("TMPDIR", str(shared))
    cfg = _base(tmp_path)
    cfg.judge.kl_cmd = [str(score), "{ckpt}", "{out}"]
    with pytest.raises(ValueError, match="which the agent or its jobs can write"):
        check(cfg)


class _StringPath(Component):
    """A component that hands its one path back as a string (`readable()` may be third-party code)."""

    name: Literal["strings"] = "strings"
    where: str = ""

    def readable(self) -> list[Path]:
        return [Path(self.where)] if self.where else []


class _Shapes(Component):
    """A component whose path settings are not typed `Path`."""

    name: Literal["shapes"] = "shapes"
    where_map: dict[str, str] = {}
    root: PurePath | None = None


class _Computed(Component):
    """A component whose `readable()` is not a settings field at all."""

    name: Literal["computed"] = "computed"

    def readable(self) -> list[Path]:
        return [Path(os.environ["QDA_TEST_READABLE"])]


def _secret(tmp_path: Path) -> Path:
    """A file in the judge's dir, as the host would read it."""
    path = tmp_path / "run" / "judge" / "secret.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("judge secret\n")
    return path


def test_a_string_path_field_is_checked(tmp_path):
    cfg = _base(tmp_path)
    cfg.toolset.components.append(_StringPath(where=str(_secret(tmp_path))))
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


@pytest.mark.parametrize("field", ["where_map", "root"])
def test_a_mapping_or_a_purepath_field_is_checked(tmp_path, field):
    inside = str(_secret(tmp_path))
    component = _Shapes()
    if field == "where_map":
        component.where_map = {"k": inside}
    else:
        component.root = PurePath(inside)
    cfg = _base(tmp_path)
    cfg.toolset.components.append(component)
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_the_paths_the_jobs_bind_are_checked(tmp_path, monkeypatch):
    """A component's `readable()` is what `read_paths()` binds, whether or not the table names its field."""
    monkeypatch.setenv("QDA_TEST_READABLE", str(_secret(tmp_path)))
    cfg = _base(tmp_path)
    cfg.toolset.components.append(_Computed())
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_a_tree_the_judge_copies_may_not_link_into_the_run_dir(tmp_path):
    """A link in the evaluator tree stayed a link in the frozen copy, so the host ran the agent's file."""
    cfg = _base(tmp_path)
    work = tmp_path / "run" / "work"
    work.mkdir(parents=True)
    (work / "evil.py").write_text("print('agent code')\n")
    tree = tmp_path / "eval"
    tree.mkdir()
    (tree / "run.py").symlink_to(work / "evil.py")
    cfg.judge.evaluator = tree
    with pytest.raises(ValueError, match="holds a link"):
        check(cfg)


def test_a_format_reference_may_not_link_into_the_run_dir(tmp_path):
    cfg = _base(tmp_path)
    work = tmp_path / "run" / "work"
    work.mkdir(parents=True)
    (work / "config.json").write_text("{}\n")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "config.json").symlink_to(work / "config.json")
    cfg.judge.format_reference = ref
    with pytest.raises(ValueError, match="holds a link"):
        check(cfg)


def test_a_package_the_toolset_copies_may_not_link_out_of_its_tree(tmp_path):
    """`copytree` follows a link: one out of the package tree would carry a host file to the agent."""
    secret = tmp_path / "host-secret.txt"
    secret.write_text("host secret\n")
    tree = tmp_path / "pkg"
    tree.mkdir()
    (tree / "leak").symlink_to(secret)
    cfg = _base(tmp_path)
    source = next(c for c in cfg.toolset.components if c.name == "source")
    source.package = tree
    with pytest.raises(ValueError, match="leads outside it"):
        check(cfg)


def test_a_private_judge_input_under_a_readable_path_is_refused(tmp_path):
    cfg = _base(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    ref = shared / "kl.json"
    ref.write_text("{}\n")
    cfg.scheduler.readable = [shared]
    cfg.judge.kl_ref = ref
    with pytest.raises(ValueError, match="which the agent's jobs read"):
        check(cfg)


def test_a_guard_inside_a_dataset_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    guard = shared / "guard.txt"
    guard.write_text("x\n")
    cfg = _base(
        tmp_path,
        toolset={"name": "qda", "components": [{"name": "data", "sets": {"c": str(shared)}}]},
    )
    cfg.contamination.guard = guard
    with pytest.raises(ValueError, match="which the agent's jobs read"):
        check(cfg)


@pytest.mark.parametrize("token", ["./score.py", "--out=../x", "-o=./x"])
def test_a_relative_command_path_is_refused(tmp_path, token):
    """The command runs from the job's directory, not where the config was read: a relative path is not the
    file the check resolved."""
    cfg = _base(tmp_path)
    cfg.judge.kl_cmd = ["{python}", token, "{ckpt}"]
    with pytest.raises(ValueError, match="a relative path"):
        check(cfg)


def test_a_relative_path_in_an_environment_value_is_refused(tmp_path):
    cfg = _base(tmp_path)
    cfg.judge.job_env = {"PYTHONPATH": "/safe:./work"}
    with pytest.raises(ValueError, match="a relative path"):
        check(cfg)


def test_an_env_wrapper_path_list_is_checked(tmp_path):
    cfg = _base(tmp_path)
    cfg.judge.kl_cmd = ["env", f"PYTHONPATH=/safe:{tmp_path / 'run' / 'work'}", "{python}"]
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_a_second_equals_in_an_option_is_checked(tmp_path):
    cfg = _base(tmp_path)
    cfg.judge.kl_cmd = ["{python}", f"--opt=x={tmp_path / 'run' / 'work'}/y"]
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_a_set_assignment_is_checked(tmp_path):
    cfg = _base(tmp_path)
    cfg.judge.kl_cmd = ["{python}", "--set", f"k={tmp_path / 'run' / 'work'}/y"]
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_an_interpreter_link_into_the_run_dir_is_refused(tmp_path):
    """The host ran a `bin/python` link the agent could replace; the installation alone looked safe."""
    cfg = _base(tmp_path)
    evil = tmp_path / "run" / "work" / "evil"
    evil.parent.mkdir(parents=True)
    evil.write_text("#!/bin/sh\n")
    home = tmp_path / "venv"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "python").symlink_to(evil)
    (home / "pyvenv.cfg").write_text("home = /usr/bin\n")
    cfg.judge.serve_python = home / "bin" / "python"
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


def test_a_placeholder_inside_a_path_does_not_hide_it(tmp_path):
    """`_skip` must match a whole `<...>`, not any part: `judge/<x>/../secret.txt` is the judge's file."""
    cfg = _base(tmp_path)
    _secret(tmp_path)
    cfg.scheduler.readable = [str(tmp_path / "run" / "judge" / "<x>" / ".." / "secret.txt")]
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


@pytest.mark.parametrize("field", ["judge.kl_cmd", "toolset.qda", "toolset.env"])
@pytest.mark.parametrize(
    "token,ok",
    [
        ("--flag={run}/x", False),  # a flag split on `=`: the tail is a path
        ("KEY={run}/x", False),  # an environment split on `=`
        ("-o={run}/x", False),
        ("{out}/../x", False),  # a slot is a path the judge makes; the agent must not climb out of it
        ("{out}/..", False),
        ("--flag={out}/../x", False),
        ("{out}/x", False),  # the agent's results: a path under the slot is its own content
        ("{ckpt}/run.py", False),  # a frozen copy of the agent's submission: the same
        ("{tools}/sub/x", True),  # a judge-side slot: the rest after it is the judge's to place
        ("--flag={out}", True),  # the bare slot as an argument is the judge's to place
    ],
)
def test_a_slot_is_a_path_the_judge_makes(tmp_path, field, token, ok):
    cfg = _base(tmp_path)
    token = token.replace("{run}", str(tmp_path / "run"))
    _put(cfg, field, {"K": token} if field == "toolset.env" else [token])
    if ok:
        check(cfg)
    else:
        with pytest.raises(ValueError, match="inside run_dir|climbs out of its slot|agent's own content"):
            check(cfg)


@pytest.mark.parametrize("field", ["judge.kl_cmd", "judge.gencheck_cmd", "toolset.qda"])
@pytest.mark.parametrize("token", ["{ckpt}", "{out}"])
def test_an_agent_slot_is_not_a_program(tmp_path, field, token):
    """A command whose program is the agent's own content would run what the agent chose, on the host."""
    cfg = _base(tmp_path)
    _put(cfg, field, [token])
    with pytest.raises(ValueError, match="the host would run it"):
        check(cfg)


def test_a_run_dir_inside_a_python_root_is_refused(tmp_path, monkeypatch):
    """Every job binds the host's Python roots: one inside the run dir would be a place the agent controls."""
    from qda_agent import run_sources

    run = tmp_path / "run"
    monkeypatch.setattr(run_sources, "python_roots", lambda: [run / "venv"])
    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": str(run),
    }
    with pytest.raises(ValueError, match="python root .*inside run_dir"):
        RunConfig.model_validate(raw)


def test_a_run_dir_under_a_system_directory_is_refused():
    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": "/etc/qda-test-run-never-created",
    }
    with pytest.raises(ValueError, match="a system directory every job reads"):
        RunConfig.model_validate(raw)


# ----------------------------------------------------------------------------------------------- the reflection


def _mentions(node: Any, cls: type) -> bool:
    return node is cls or any(_mentions(arg, cls) for arg in get_args(node))


def _settings() -> dict[str, Any]:
    """Every config setting that holds a string or a path, by its key path, from the models and plugins."""
    models: list[tuple[str, Any]] = [("", RunConfig), ("judge", JudgeSettings), ("agent", AgentConfig)]
    for registry, prefix in (
        (BACKENDS, "agent.backend"),
        (BACKENDS, "agent.fallbacks[]"),
        (SCHEDULERS, "scheduler"),
        (LAUNCHERS, "launcher"),
        (TOOLSETS, "toolset"),
        (CHECKS, "contamination"),
        (FINAL_RULES, "judge.final_rules[]"),
        (SANDBOXES, "sandbox"),
        (COMPONENTS, "toolset.components[]"),
    ):
        models += [(prefix, registry[key]) for key in registry]
    out: dict[str, Any] = {}
    for prefix, model in models:
        for name, info in model.model_fields.items():
            if name == "name":  # the plugin's key, not a setting
                continue
            if _mentions(info.annotation, str) or _mentions(info.annotation, Path):
                out[f"{prefix}.{name}" if prefix else name] = info.annotation
    # the one llama.cpp program the table walks; the bridge and checkout are guarded by `isolation.trusted`
    out["judge.llamacpp.server"] = LlamaCpp.model_fields["server"].annotation
    return out


def test_every_string_or_path_setting_has_a_row():
    missing = sorted(set(_settings()) - {row.field for row in SOURCES})
    assert not missing, (
        "these settings hold a string or a path and have no row in run_sources.SOURCES; add one, "
        f"with Kind.DERIVED and a reason if it is not a path: {', '.join(missing)}"
    )


def test_every_row_names_a_setting():
    extra = sorted({row.field for row in SOURCES} - set(_settings()) - SYNTHETIC)
    assert not extra, f"run_sources.SOURCES names settings the config does not have: {', '.join(extra)}"


def test_a_holdout_reference_under_a_readable_path_is_refused(tmp_path):
    """The holdout prompts are a private judge input too: the agent's jobs must not read their directory."""
    cfg = _base(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    ref = shared / "holdout.jsonl"
    ref.write_text('{"id": "h1", "prompt": "held out"}\n')
    cfg.scheduler.readable = [shared]
    cfg.judge.kl_holdout_ref = ref
    with pytest.raises(ValueError, match="which the agent's jobs read"):
        check(cfg)
