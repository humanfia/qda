"""One definition of the engine parameters: `qda-kl ref`, `qda-kl kl` and `qda-agent prepare` read it.

The reference cache keys an entry by these settings, so two commands that name the same settings must ask for the
same reference; and a setting one command cannot name is a setting the user cannot use there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from qda_kl import cli, refcache, score
from qda_kl.engine_params import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    engine_parameters,
    topk_settings,
)
from qda_kl.scorers import TopKScorer
from qda_kl.store import Manifest
from qda_quant import cli as qda
from qda_quant.calib import selfgen
from qda_quant.intake import serve
from qda_quant.output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION as SHARED
from qda_quant.output.vllm_env import DEFAULT_TP


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}")
    (model / "generation_config.json").write_text(json.dumps({"temperature": 0.7}))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "domain": "web", "text": "hello"}) + "\n")
    return model, prompts


def _entry(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(Manifest(k=128, vocab=10).model_dump_json())
    return directory


def ref_settings(tmp_path, monkeypatch, extra=()) -> dict:
    """The settings `qda-kl ref --cache` hands the reference cache."""
    model, prompts = _inputs(tmp_path)
    seen = {}

    def cached_many(got_model, groups, settings, log=print):
        seen.update(settings)
        entry = _entry(tmp_path / "entry")
        for group in groups:
            if group.out is not None:
                refcache.link(Path(group.out), entry)
        return [entry for _ in groups], [True for _ in groups]

    monkeypatch.setattr(refcache, "cached_many", cached_many)
    argv = ["ref", "--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "r"), "--cache"]
    assert cli.main([*argv, *extra]) == 0
    return seen


def kl_settings(tmp_path, monkeypatch, extra=()) -> tuple[dict, dict]:
    """The settings `qda-kl kl` hands the reference cache, and the score config it writes for the reading."""
    model, prompts = _inputs(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "resolved.yaml").write_text(f"model: {model}\n")
    seen = {}

    def cached(got_model, got_prompts, out, settings, log=print):
        seen.update(settings)
        return _entry(tmp_path / "entry"), False

    def scored(config, overrides):
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.json").write_text(
            json.dumps({"kl": 0.1, "domains": {}, "vllm": "0.33.0", "tp": 1, "tokens": 3})
        )
        return 0

    monkeypatch.setattr(refcache, "cached", cached)
    monkeypatch.setattr(cli, "_score_config", scored)
    argv = ["kl", str(checkpoint), "--prompts", str(prompts), "--out", str(tmp_path / "out")]
    assert cli.main([*argv, *extra]) == 0
    config = yaml.safe_load((tmp_path / "out.yaml").read_text())
    return seen, config


def test_ref_and_kl_ask_the_cache_for_one_reference(tmp_path, monkeypatch):
    """A setting one command names and the other leaves at its default is the same reference."""
    model, prompts = _inputs(tmp_path)
    ref = ref_settings(tmp_path, monkeypatch)
    got_kl, _ = kl_settings(tmp_path, monkeypatch)
    assert refcache.key_payload(model, prompts, ref) == refcache.key_payload(model, prompts, got_kl)


def test_kl_carries_the_memory_share_it_is_given(tmp_path, monkeypatch):
    """`--gpu-memory-utilization` reaches the reference build and the reading's own engine."""
    built, config = kl_settings(tmp_path, monkeypatch, ("--gpu-memory-utilization", "0.3"))
    assert built["gpu_memory_utilization"] == 0.3
    assert config["scorer"]["topk"]["gpu_memory_utilization"] == 0.3


def test_defaults_are_one_value_in_one_place(tmp_path, monkeypatch):
    built, config = kl_settings(tmp_path, monkeypatch)
    assert built["gpu_memory_utilization"] == DEFAULT_GPU_MEMORY_UTILIZATION
    assert config["scorer"]["topk"]["gpu_memory_utilization"] == DEFAULT_GPU_MEMORY_UTILIZATION


@pytest.mark.parametrize("verb", ["ref", "kl"])
def test_serve_args_may_not_own_the_memory_share(tmp_path, monkeypatch, capsys, verb):
    """The flag is the one owner: a `--serve-args` that names the same setting is refused, never silently won."""
    extra = ("--serve-args", '{"gpu_memory_utilization": 0.2}')
    if verb == "ref":
        model, prompts = _inputs(tmp_path)
        argv = ["ref", "--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "r")]
        assert cli.main([*argv, *extra]) == 2
    else:
        model, prompts = _inputs(tmp_path)
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / "resolved.yaml").write_text(f"model: {model}\n")
        argv = ["kl", str(checkpoint), "--prompts", str(prompts), "--out", str(tmp_path / "out")]
        assert cli.main([*argv, *extra]) == 2
    assert "pass --gpu-memory-utilization" in capsys.readouterr().err


def test_topk_scorer_gives_its_engine_the_memory_share(monkeypatch):
    """The reading's engine starts where the reference did: the scorer hands its share to `score.Scorer`."""
    seen = {}

    class FakeScorer:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def score(self, **kwargs):
            return SimpleNamespace(deterministic=True)

    monkeypatch.setattr(score, "Scorer", FakeScorer)
    outcome = TopKScorer(gpu_memory_utilization=0.3).run("ckpt", Path("ref"))
    assert seen["gpu_memory_utilization"] == 0.3
    assert outcome.result.deterministic is True


def test_a_score_config_cannot_name_the_memory_share_twice():
    with pytest.raises(ValidationError, match="gpu_memory_utilization"):
        TopKScorer(serve_args={"gpu_memory_utilization": 0.2})


def test_the_engine_parameters_and_the_score_settings_agree(tmp_path):
    args = SimpleNamespace(tp=2, gpu_memory_utilization=0.4, serve_env=None, serve_args=None, nccl_nvls=False)
    assert engine_parameters(args)["gpu_memory_utilization"] == 0.4
    assert topk_settings(args) == {"tp": 2, "gpu_memory_utilization": 0.4}


def _parsers(main, argv: list[str], monkeypatch) -> list[argparse.ArgumentParser]:
    """Every parser and subparser `main` builds, without running a command."""
    grabbed: list[argparse.ArgumentParser] = []

    def grab(parser: argparse.ArgumentParser, *_: object) -> None:
        grabbed.append(parser)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        main(argv)
    out, todo = [], list(grabbed)
    while todo:
        parser = todo.pop()
        out.append(parser)
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                todo.extend(action.choices.values())
    return out


def test_every_command_reads_the_one_memory_share_and_tp(monkeypatch):
    """`--gpu-memory-utilization` and `--tp` default to one definition in every command."""
    assert DEFAULT_GPU_MEMORY_UTILIZATION is SHARED and (SHARED, DEFAULT_TP) == (0.85, 1)
    parsers = [
        *_parsers(qda.main, [], monkeypatch),
        *_parsers(cli.main, [], monkeypatch),
        *_parsers(selfgen.main, [], monkeypatch),
        *_parsers(serve._serve_main, [], monkeypatch),
    ]
    defaults = {
        (parser.prog, action.dest): action.default
        for parser in parsers
        for action in parser._actions
        if action.dest in ("gpu_memory_utilization", "tp")
    }
    wanted = {"gpu_memory_utilization": SHARED, "tp": DEFAULT_TP}
    assert {key: wanted[key[1]] for key in defaults} == defaults
    names = {prog for prog, _ in defaults}
    assert {"qda inspect", "qda traces", "qda sensitivity read", "qda-kl ref", "qda-kl kl"} <= names
