"""Values a refactor must not change: a cache key, `resolved.yaml`'s key order, plugin names and the CLI."""

import argparse
import importlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from qda_agent import cli
from qda_agent.config import RunConfig
from qda_agent.probe import _digest
from tests_support import filled

PACKAGE = Path(__file__).parents[1]
GOLDEN = Path(__file__).parent / "golden"
#: Entry-point group -> plugin names, in the order `pyproject.toml` declares them.
ENTRY_POINTS = json.loads((GOLDEN / "entry_points.json").read_text())


def registered_names(group: str) -> list[str]:
    """The names this package registers under `group` (the one place that knows how they are registered)."""
    refs = {
        "qda_agent.backends": "qda_agent.agents:BACKENDS",
        "qda_agent.schedulers": "qda_agent.sched.config:SCHEDULERS",
        "qda_agent.launchers": "qda_agent.launchers:LAUNCHERS",
        "qda_agent.toolsets": "qda_agent.toolsets:TOOLSETS",
        "qda_agent.components": "qda_agent.components:COMPONENTS",
        "qda_agent.contamination_checks": "qda_agent.judge.contamination:CHECKS",
        "qda.final_rules": "qda_agent.judge.final_rules:FINAL_RULES",
        "qda_agent.sandboxes": "qda_agent.sandbox_settings:SANDBOXES",
        "qda.eval_suites": "qda_kl.eval.suites:SUITES",
        "qda.checks": "qda_quant.doctor:CHECKS",
        "qda.commands": "qda_quant.cli:COMMANDS",
    }
    module, attr = refs[group].split(":")
    owner = importlib.import_module(module)
    registry = getattr(owner, attr)
    list(registry)
    entries = owner.BUILTINS if group == "qda_agent.components" else registry._entries
    return [
        name
        for name, obj in entries.items()
        if (obj if isinstance(obj, str) else obj.__module__).startswith("qda_agent.")
    ]


def test_the_probe_digest_is_frozen():
    assert _digest({"reference": "ref", "rows": 8, "seed": "qda-probe"}) == "686f75df9f3b"


def test_the_resolved_config_key_order_is_frozen(tmp_path):
    """The top-level keys (under "") and each section's keys of the search preset's `resolved.yaml`, in order."""
    resolved = RunConfig.load(filled(PACKAGE / "configs" / "search.yaml", tmp_path)).resolved()
    keys = {"": list(resolved), **{k: list(v) for k, v in resolved.items() if isinstance(v, dict)}}
    golden = json.loads((GOLDEN / "resolved_keys.json").read_text())
    assert list(keys.items()) == list(golden.items())


def test_registry_names_and_their_declared_order_are_frozen():
    declared = {group: registered_names(group) for group in ENTRY_POINTS}
    assert {group: list(names) for group, names in declared.items()} == ENTRY_POINTS
    # Installed metadata does not keep the declared order, so the registered names are compared as a set.
    registered = {group: sorted(registered_names(group)) for group in ENTRY_POINTS}
    assert registered == {group: sorted(names) for group, names in ENTRY_POINTS.items()}


def test_the_cli_commands_are_frozen(monkeypatch):
    parsers: list[argparse.ArgumentParser] = []

    def grab(parser, *_):
        parsers.append(parser)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        cli.main([])
    (sub,) = [a for a in parsers[0]._actions if isinstance(a, argparse._SubParsersAction)]
    assert list(sub.choices) == ["run", "gate", "cand0", "prepare", "search", "evaluate", "status"]


def test_the_wheel_ships_the_vendored_livecodebench_license_and_record(tmp_path):
    """The built wheel holds one LICENSE and one VENDOR.json beside the vendored code, wherever that sits."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv builds the wheel")
    subprocess.run(
        [uv, "build", "--wheel", str(PACKAGE), "-o", str(tmp_path)], check=True, capture_output=True
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as built:
        names = built.namelist()
        records = [
            n
            for n in names
            if n.endswith("/VENDOR.json") and "LiveCodeBench" in json.loads(built.read(n))["repo"]
        ]
    (record,) = records
    home = record.rsplit("/", 1)[0]
    assert f"{home}/__init__.py" in names  # the record sits in the vendored package
    assert [n for n in names if n.startswith(f"{home}/") and "LICENSE" in n] == [f"{home}/LICENSE"]
