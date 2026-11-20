"""Values a refactor must not change: a cache key, the registered plugin names and the CLI's commands."""

import argparse
import importlib
import json
from pathlib import Path

import pytest
from qda_quant import cli
from qda_quant.calib.selfgen import cache_key

PACKAGE = Path(__file__).parents[1]
#: Entry-point group -> plugin names, in the order ``pyproject.toml`` declares them.
GOLDEN = json.loads((Path(__file__).parent / "golden" / "entry_points.json").read_text())


def registered_names(group: str) -> list[str]:
    """The names this package registers under ``group`` (the one place that knows how they are registered)."""
    refs = {
        "qda.steps": "qda_quant.step:REGISTRY",
        "qda.commands": "qda_quant.cli:COMMANDS",
        "qda.stats": "qda_quant.step:STATISTICS",
        "qda.arch_adapters": "qda_quant.arch.adapters:ADAPTERS",
        "qda.prefill_numerics": "qda_quant.arch.adapters.prefill:PREFILL_NUMERICS",
        "qda.targets": "qda_quant.target:TARGETS",
        "qda.formats": "qda_quant.format:FORMATS",
        "qda.bindings": "qda_quant.target:BINDINGS",
        "qda.grids": "qda_quant.solve.grid:GRIDS",
        "qda.clips": "qda_quant.clips:CLIPS",
        "qda.weight_stores": "qda_quant.engine.stores:WEIGHT_STORES",
        "qda.corpora": "qda_quant.intake.corpus:CORPORA",
        "qda.source_codecs": "qda_quant.intake.source:SOURCE_CODECS",
        "qda.checkpoint_formats": "qda_quant.intake.formats:CHECKPOINT_FORMATS",
        "qda.exporters": "qda_quant.output.exporters:EXPORTERS",
        "qda.checks": "qda_quant.doctor:CHECKS",
        "qda.calib": "qda_quant.calib.sources:SOURCES",
    }
    module, attr = refs[group].split(":")
    registry = getattr(importlib.import_module(module), attr)
    list(registry)
    return [
        name
        for name, obj in registry._entries.items()
        if (obj if isinstance(obj, str) else obj.__module__).startswith("qda_quant.")
    ]


def test_the_selfgen_cache_key_is_frozen() -> None:
    settings = {"n": 4, "max_tokens": 16, "seed": 0, "sampling": {"temperature": 0.7}}
    assert cache_key(settings) == "77e4510ed9f29376972a10cd"


def test_registry_names_and_their_declared_order_are_frozen() -> None:
    declared = {group: registered_names(group) for group in GOLDEN}
    assert {group: list(names) for group, names in declared.items()} == GOLDEN
    # Installed metadata does not keep the declared order, so the registered names are compared as a set.
    assert {group: sorted(registered_names(group)) for group in GOLDEN} == {
        g: sorted(n) for g, n in GOLDEN.items()
    }


def test_the_cli_commands_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """``qda``'s subcommands in listed order, with the ones qda-kl and qda-agent add (``qda.commands``)."""
    parsers: list[argparse.ArgumentParser] = []

    def grab(parser: argparse.ArgumentParser, *_: object) -> None:
        parsers.append(parser)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        cli.main([])
    (sub,) = [a for a in parsers[0]._actions if isinstance(a, argparse._SubParsersAction)]
    assert list(sub.choices) == [
        *(
            "build",
            "plan",
            "doctor",
            "fisher",
            "inspect",
            "data",
            "calib",
            "export",
        ),
        *("pipeline", "sensitivity", "decode-gguf", "eval", "probe", "rotate-draft", "traces"),
    ]
