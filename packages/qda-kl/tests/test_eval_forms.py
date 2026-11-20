"""`qda eval`'s three forms: a config file, a checkpoint directory (`build.json`) and a run directory
(`FINAL.json`), told apart by the first argument; and the four `--suite` presets, owned by qda-kl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from qda_kl.eval import cli as eval_cli
from qda_kl.eval import presets, rundir
from qda_kl.eval.suites import parse_suite
from qda_quant.cli import main as qda


def _checkpoint(path: Path, source: str = "Qwen/Qwen3-0.6B", **extra: str) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "build.json").write_text(json.dumps({"source": source, **extra}))
    return path


def _run(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "resolved.yaml").write_text("model: m\n")
    (path / "FINAL.json").write_text("{}")
    return path


def test_the_first_argument_decides_the_form(tmp_path: Path) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text("reference: m\n")
    assert eval_cli.form(config) == "config"
    assert eval_cli.form(_checkpoint(tmp_path / "ckpt")) == "checkpoint"
    assert eval_cli.form(_run(tmp_path / "run")) == "run"


@pytest.mark.parametrize("case", ["both", "neither", "missing", "unfinished"])
def test_an_unclear_first_argument_is_refused_in_two_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    target = tmp_path / "x"
    if case == "both":
        _run(target)
        (target / "build.json").write_text("{}")
    elif case == "neither":
        target.mkdir()
    elif case == "unfinished":
        target.mkdir()
        (target / "resolved.yaml").write_text("model: m\n")
    assert qda(["eval", str(target), "--suite", "kl"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 2
    assert eval_cli.FORMS in err
    said = {
        "both": "holds both FINAL.json",
        "neither": "neither FINAL.json",
        "missing": "no such file or directory",
        "unfinished": "no FINAL.json; evaluate a run that has finished",
    }
    assert said[case] in err


def test_the_help_states_the_rule() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    eval_cli.add_parser(parser.add_subparsers())
    sub = parser._subparsers._group_actions[0].choices["eval"]  # type: ignore[union-attr]
    assert "FINAL.json is a search run" in (sub.description or "")
    assert "build.json is a checkpoint" in (sub.description or "")


def test_the_four_presets_resolve_from_qda_kl() -> None:
    assert list(presets.PRESETS) == ["kl", "sanity", "full", "specdec"]
    for name in presets.PRESETS:
        got, groups = rundir.suite_groups(name)
        assert got == name and groups == presets.PRESETS[name]
        for group in groups:
            for entry in group.suites:
                assert parse_suite(entry, name).name in entry


def test_the_kl_preset_reads_the_holdout_qda_kl_defaults_to(tmp_path: Path, monkeypatch) -> None:
    """`qda eval --suite kl` with no corpus scores the KL v2 holdout: the file `qda-kl kl` reads by default."""
    from qda_kl import corpus, kl

    holdout = tmp_path / "corpus.holdout.jsonl"
    holdout.write_text(json.dumps({"id": "a", "domain": "math", "split": "holdout", "prompt": "2+2?"}) + "\n")
    splits = {"eval": tmp_path / "e.jsonl", "holdout": holdout, "search96": tmp_path / "s.jsonl"}
    monkeypatch.setattr(corpus, "cached_splits", lambda *a, **k: splits)
    _name, groups = rundir.suite_groups("kl")
    suite = parse_suite(groups[0].suites[0], "suites.0")
    assert kl.default_prompts() == holdout
    assert suite.corpus.rows() == [json.loads(holdout.read_text())]
    assert suite.to_dict()["kl"]["corpus"] == {"kl_holdout": {}}


@pytest.mark.parametrize("preset", ["kl", "specdec"])
@pytest.mark.parametrize("form", ["checkpoint", "run"])
def test_a_preset_given_no_corpus_reads_the_kl_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, form: str, preset: str
) -> None:
    """`qda eval CKPT|RUN_DIR --suite PRESET` with no corpus reads the KL v2 holdout, not an unbundled corpus."""
    from qda_kl import corpus
    from qda_kl.eval.config import EvalConfig

    holdout = tmp_path / "corpus.holdout.jsonl"
    holdout.write_text(json.dumps({"id": "a", "domain": "math", "split": "holdout", "prompt": "2+2?"}) + "\n")
    splits = {"eval": tmp_path / "e.jsonl", "holdout": holdout, "search96": tmp_path / "s.jsonl"}
    monkeypatch.setattr(corpus, "cached_splits", lambda *a, **k: splits)
    ckpt = _checkpoint(tmp_path / "ckpt")
    if form == "checkpoint":
        (config,) = eval_cli.checkpoint_configs([ckpt], preset)
    else:
        _name, (group,) = rundir.suite_groups(preset)
        out = tmp_path / "run" / "evaluate" / preset
        config = rundir.benchmark_config(
            tmp_path / "run", {"model": "m", "judge": {}}, {"c": str(ckpt)}, out, group.suites
        )
    (suite,) = EvalConfig(**config).suites
    assert suite.corpus.name == "kl_holdout"
    assert suite.corpus.rows() == [json.loads(holdout.read_text())]


def test_the_kl_preset_uses_the_corpus_it_is_given(tmp_path: Path, monkeypatch) -> None:
    """A corpus named beside the suite wins: the default holdout is not read at all."""
    from qda_kl import holdout

    def no_holdout() -> Path:
        raise AssertionError("the default holdout was read")

    monkeypatch.setattr(holdout, "default_prompts", no_holdout)
    given = tmp_path / "prompts.jsonl"
    given.write_text(json.dumps({"id": "b", "split": "eval", "prompt": "3+3?"}) + "\n")
    one = json.dumps({"kl": {"corpus": {"file": {"path": str(given)}}}})
    _name, groups = rundir.suite_groups(one)
    suite = parse_suite(groups[0].suites[0], "suites.0")
    assert suite.corpus.rows() == [{"id": "b", "split": "eval", "prompt": "3+3?"}]


def test_checkpoints_are_scored_against_the_original_their_build_names(tmp_path: Path) -> None:
    a = _checkpoint(tmp_path / "q w4a16")
    b = _checkpoint(tmp_path / "other" / "q w4a16", source="/models/q", source_repo="Qwen/Qwen3-0.6B")
    (config,) = eval_cli.checkpoint_configs([a, b], "sanity")
    assert config["reference"] == "Qwen/Qwen3-0.6B"  # source_repo wins over a local source
    assert config["checkpoints"] == {"q-w4a16": str(a.resolve()), "q-w4a16-2": str(b.resolve())}
    assert config["out"] == str(tmp_path / "q w4a16-eval" / "sanity")
    assert config["suites"] == [{"sanity": {}}]


def test_checkpoints_of_different_originals_are_refused(tmp_path: Path) -> None:
    a = _checkpoint(tmp_path / "a")
    b = _checkpoint(tmp_path / "b", source="Qwen/Qwen3-8B")
    with pytest.raises(ValueError, match="different originals"):
        eval_cli.checkpoint_configs([a, b], "kl")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "build.json").write_text("{}")
    with pytest.raises(ValueError, match="names no source model"):
        eval_cli.checkpoint_configs([tmp_path / "c"], "kl")


def test_a_checkpoint_runs_the_config_it_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _checkpoint(tmp_path / "ckpt")
    seen: list[dict] = []

    def score(text: str, a: object, where: str) -> int:
        seen.append(yaml.safe_load(text))
        return 0

    monkeypatch.setattr(eval_cli, "_score", score)
    assert qda(["eval", str(ckpt), "--suite", "kl", "--dry-run"]) == 0
    assert seen == [eval_cli.checkpoint_configs([ckpt], "kl")[0]]
    assert qda(["eval", str(ckpt)]) == 2
    assert "needs --suite" in capsys.readouterr().err


def test_a_config_takes_no_suite_and_a_run_takes_no_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text("reference: m\n")
    assert qda(["eval", str(config), "--suite", "kl"]) == 2
    assert "a config names its suites" in capsys.readouterr().err
    run = _run(tmp_path / "run")
    assert qda(["eval", str(run), "--set", "out=x"]) == 2
    assert "a run directory takes --suite" in capsys.readouterr().err


def test_a_run_directory_goes_to_the_run_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path / "run")
    calls: list[tuple] = []
    monkeypatch.setattr(rundir, "run", lambda *a, **kw: calls.append((a, kw)) or 0)
    assert qda(["eval", str(run), "--suite", "sanity", "--dry-run", "--lcb-sandbox", "none"]) == 0
    assert calls == [((run.resolve(), "sanity"), {"dry_run": True, "lcb_sandbox": "none"})]
