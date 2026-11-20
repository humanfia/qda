"""Values a refactor must not change: cache keys, the registered plugin names and the CLI's commands."""

import argparse
import ast
import importlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from qda_kl import cli, engine, gencheck, score, vllm_hooks
from qda_kl.eval import sanity
from qda_kl.eval.run import digest
from qda_kl.eval.suites import parse_suite
from qda_kl.gencheck import GencheckConfig, original_cache_key, read_prompts

PACKAGE = Path(__file__).parents[1]
#: Entry-point group -> plugin names, in the order ``pyproject.toml`` declares them.
GOLDEN = json.loads((Path(__file__).parent / "golden" / "entry_points.json").read_text())
DOMAINS = ("math", "code", "science", "instruction")
RPC_METHODS = {"qda_kl_install", "qda_kl_begin", "qda_kl_end", "qda_kl_kernels"}


def test_worker_extension_import_and_tag_are_frozen() -> None:
    assert vllm_hooks.TAG == "qda_kl"
    assert vllm_hooks.EXTENSION == "qda_kl.vllm_hooks.KLWorker"
    module, _, name = vllm_hooks.EXTENSION.rpartition(".")
    worker = getattr(importlib.import_module(module), name)
    assert isinstance(worker, type)
    assert {name for name in vars(worker) if name.startswith("qda_kl_")} == RPC_METHODS
    assert all(callable(getattr(worker, name)) for name in RPC_METHODS)


@pytest.mark.parametrize(
    "module,expected",
    [(engine, RPC_METHODS - {"qda_kl_kernels"}), (score, {"qda_kl_kernels"}), (sanity, {"qda_kl_kernels"})],
)
def test_collective_rpc_names_used_by_callers_are_frozen(module, expected) -> None:
    calls = ast.walk(ast.parse(Path(module.__file__).read_text()))
    names = {
        call.args[0].value
        for call in calls
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "collective_rpc"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert names == expected


def registered_names(group: str) -> list[str]:
    """The names this package registers under ``group`` (the one place that knows how they are registered)."""
    refs = {
        "qda_kl.scorers": "qda_kl.scorers:SCORERS",
        "qda.eval_suites": "qda_kl.eval.suites:SUITES",
        "qda.datasets": "qda_kl.datasets:DATASETS",
        "qda.commands": "qda_quant.cli:COMMANDS",
    }
    module, attr = refs[group].split(":")
    registry = getattr(importlib.import_module(module), attr)
    list(registry)
    return [
        name
        for name, obj in registry._entries.items()
        if (obj if isinstance(obj, str) else obj.__module__).startswith("qda_kl.")
    ]


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Two fixed rows, named by a relative path so no key holds the temporary directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rows.jsonl").write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    return {"file": {"path": "rows.jsonl"}}


@pytest.mark.parametrize(("key", "name"), [("kl", "2eabed49d39f394d"), ("specdec", "ac035c2a1fd347c0")])
def test_a_corpus_suite_reference_dir_is_frozen(corpus: dict, tmp_path: Path, key: str, name: str) -> None:
    suite = parse_suite({key: {"corpus": corpus}}, "suites.0")
    assert suite.reference_dir("original", tmp_path) == tmp_path / "refs" / name


def test_the_sanity_suite_reference_dir_and_digest_are_frozen(tmp_path: Path, monkeypatch) -> None:
    """Both hold the shipped prompts' hash, so a change to the prompt file moves them too."""
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)  # no model to read a config from
    suite = parse_suite({"sanity": {"domains": ["math"], "per_domain": 1}}, "suites.0")
    # The key holds the suite's name and the original's engine settings (no_jit, as gencheck's), so answers
    # cached under other settings are not reused.
    # Moved on purpose when no_jit turned FlashInfer autotune off. Was f65472bec4770cb5.
    assert suite.reference_dir("original", tmp_path) == tmp_path / "refs" / "13c9f896d2e50472"
    expected = "b9637df3fd14a909ac03a7e36e6505247e77861ace699e4a66698d5977d09907"
    assert digest(suite, "original", "checkpoint") == expected


def test_the_result_digest_is_frozen(corpus: dict) -> None:
    suite = parse_suite({"kl": {"corpus": corpus}}, "suites.0")
    # moved once, by adding the scorer's gpu_memory_utilization key: the resolved settings changed
    expected = "51286302209075f1231663b87cd39293987e54d4709295010d7a8ee678f00f58"
    assert digest(suite, "original", "checkpoint", {"config.json": "0" * 64}) == expected


def test_the_gencheck_original_cache_key_is_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model identity and library versions are fixed here: they name this machine's files and installs."""
    monkeypatch.setattr(gencheck, "_original_identity", lambda model: {"model": model})
    monkeypatch.setattr(gencheck, "version", lambda name: "0")
    rows = [{"id": f"{d}-{i}", "domain": d, "prompt": "Explain a proof."} for i in range(2) for d in DOMAINS]
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = GencheckConfig(
        checkpoint="checkpoint",
        original="original",
        prompts=str(tmp_path / "prompts.jsonl"),
        out=str(tmp_path / "out"),
        per_domain=1,
        max_tokens=300,
    )
    # Changed on purpose (F1, 2026-11-13): the key holds the serve settings and both engines now start with
    # gencheck.no_jit's (vLLM's own sampler, no FlashInfer all-reduce), whose answers differ from FlashInfer's
    # sampler at the same seed, so answers cached before must not be reused. Was ba5f0d25...bd5dd.
    # Changed again on purpose (F1b, 2026-11-13): no_jit also starts both engines with the Triton GDN prefill
    # (gdn_prefill_backend: triton), a serve setting the key holds; a Qwen hybrid original answers with a
    # different prefill kernel than FlashInfer's, so answers cached before must not be reused. Was eeaa70ae...cc08.
    # Changed again on purpose: no_jit starts both engines with FlashInfer autotune off
    # (enable_flashinfer_autotune: false), a serve setting the key holds. Was 88696752...3049.
    expected = "fe751333d2885a3f49f3f3b109f46b9fe6f739783c081fb37dbd0dce93bc47e4"
    assert original_cache_key(config, read_prompts(config)) == expected


def test_registry_names_and_their_declared_order_are_frozen() -> None:
    declared = {group: registered_names(group) for group in GOLDEN}
    assert {group: list(names) for group, names in declared.items()} == GOLDEN
    # Installed metadata does not keep the declared order, so the registered names are compared as a set.
    assert {group: sorted(registered_names(group)) for group in GOLDEN} == {
        g: sorted(n) for g, n in GOLDEN.items()
    }


def test_the_cli_commands_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    parsers: list[argparse.ArgumentParser] = []

    def grab(parser: argparse.ArgumentParser, *_: object) -> None:
        parsers.append(parser)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        cli.main([])
    (sub,) = [a for a in parsers[0]._actions if isinstance(a, argparse._SubParsersAction)]
    assert list(sub.choices) == ["ref", "score", "gencheck", "corpus", "kl"]


def test_the_wheel_ships_the_vendored_ifbench_license_and_record(tmp_path):
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
            n for n in names if n.endswith("/VENDOR.json") and "IFBench" in json.loads(built.read(n))["repo"]
        ]
    (record,) = records
    home = record.rsplit("/", 1)[0]
    assert f"{home}/__init__.py" in names  # the record sits in the vendored package
    assert [n for n in names if n.startswith(f"{home}/") and "LICENSE" in n] == [f"{home}/LICENSE"]
