"""`qda-kl kl`: the original is resolved from the checkpoint's own files, or the command refuses naming --original."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import pytest
from qda_kl import kl


def test_the_original_is_the_build_record_first(tmp_path):
    (tmp_path / "resolved.yaml").write_text("model: Qwen/Qwen3-8B@abc\nout: x\n")
    (tmp_path / "config.json").write_text(json.dumps({"base_model": "other/m"}))
    assert kl.original_of(tmp_path) == "Qwen/Qwen3-8B@abc"


SHA = "c1899de289a04d12100db370d81485cdf75e47ca"


def _built(path, model, revision):
    path.mkdir(parents=True, exist_ok=True)
    prov = f"provenance:\n  source_revision: {revision}\n" if revision else ""
    (path / "resolved.yaml").write_text(f"model: {model}\nout: {path.name}\n{prov}")
    return path


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B@main"])
def test_the_original_defaults_to_the_revision_the_build_recorded(tmp_path, model):
    assert kl.original_of(_built(tmp_path / "ckpt", model, SHA)) == f"Qwen/Qwen3-0.6B@{SHA}"


def test_a_later_stage_follows_back_to_the_first_stages_recorded_revision(tmp_path):
    _built(tmp_path / "start", "Qwen/Qwen3-0.6B", SHA)
    assert kl.original_of(_built(tmp_path / "final", "./start", None)) == f"Qwen/Qwen3-0.6B@{SHA}"


def test_a_record_without_a_revision_keeps_the_model_as_recorded(tmp_path):
    assert kl.original_of(_built(tmp_path / "ckpt", "Qwen/Qwen3-0.6B", None)) == "Qwen/Qwen3-0.6B"
    (tmp_path / "ckpt" / "resolved.yaml").write_text("model: Qwen/Qwen3-0.6B\nprovenance: odd\n")
    assert kl.original_of(tmp_path / "ckpt") == "Qwen/Qwen3-0.6B"


def _built_json(path, source_repo=None, source=None):
    path.mkdir(parents=True)
    (path / "build.json").write_text(json.dumps({"source": source, "source_repo": source_repo}))
    return path


def test_a_published_checkpoint_takes_the_commit_its_build_json_pins(tmp_path):
    """A Hub checkpoint ships build.json and no resolved.yaml; the pin is build.json's source_repo."""
    ckpt = _built_json(tmp_path / "ckpt", source_repo=f"Qwen/Qwen3.6-35B-A3B@{SHA}")
    assert kl.original_of(ckpt) == f"Qwen/Qwen3.6-35B-A3B@{SHA}"


def test_build_json_without_a_source_repo_uses_the_source_it_read(tmp_path):
    assert kl.original_of(_built_json(tmp_path / "ckpt", source="/builds/qwen3-0.6b")) == "/builds/qwen3-0.6b"
    here = tmp_path / "here"
    here.mkdir()
    assert kl.original_of(_built_json(tmp_path / "ckpt2", source=str(here))) == str(here)


def test_a_build_json_source_not_on_this_machine_gives_way_to_the_card_base_model(tmp_path):
    ckpt = _built_json(tmp_path / "ckpt", source="/builds/qwen3-0.6b")
    (ckpt / "README.md").write_text("---\nbase_model: Qwen/Qwen3-0.6B\n---\n")
    assert kl.original_of(ckpt) == "Qwen/Qwen3-0.6B"


def test_a_resolved_record_without_a_commit_takes_the_one_build_json_pins(tmp_path):
    ckpt = _built(tmp_path / "ckpt", "Qwen/Qwen3-0.6B", None)
    (ckpt / "build.json").write_text(json.dumps({"source_repo": f"Qwen/Qwen3-0.6B@{SHA}"}))
    assert kl.original_of(ckpt) == f"Qwen/Qwen3-0.6B@{SHA}"


def test_a_resolved_record_wins_over_build_json(tmp_path):
    ckpt = _built(tmp_path / "ckpt", "./start", None)
    _built(tmp_path / "start", "Qwen/Qwen3-0.6B", SHA)
    (ckpt / "build.json").write_text(json.dumps({"source_repo": f"Other/Model@{'b' * 40}"}))
    assert kl.original_of(ckpt) == f"Qwen/Qwen3-0.6B@{SHA}"
    (ckpt / "resolved.yaml").write_text("model: Qwen/Qwen3-0.6B\n")
    assert kl.original_of(ckpt) == "Qwen/Qwen3-0.6B"  # not the other repo's pin


def test_build_json_wins_over_the_config_and_the_model_card(tmp_path):
    ckpt = _built_json(tmp_path / "ckpt", source_repo=f"Qwen/Qwen3.6-35B-A3B@{SHA}")
    (ckpt / "config.json").write_text(json.dumps({"base_model": "org/base"}))
    (ckpt / "README.md").write_text("---\nbase_model: org/card\n---\n")
    assert kl.original_of(ckpt) == f"Qwen/Qwen3.6-35B-A3B@{SHA}"


def _run_with(tmp_path, monkeypatch, original, given, build_json=None):
    """`kl.run` up to the reference, with the Hub snapshot faked; returns the stderr line naming the revision."""
    import qda_kl.refcache as refcache
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    seen = {}

    def snapshot(model):
        if Path(model).is_dir():
            return Path(model)
        name, _, rev = str(model).partition("@")
        seen["model"] = model
        d = tmp_path / "hub" / "models--x" / "snapshots" / (rev if len(rev) == 40 else "f" * 40)
        d.mkdir(parents=True, exist_ok=True)
        return d

    class Stop(Exception):
        pass

    def cached(*a, **k):
        raise Stop

    monkeypatch.setattr(refcache, "local_model", snapshot)
    monkeypatch.setattr(refcache, "cached", cached)
    monkeypatch.setattr("qda_kl.reference.default_sampling", lambda source: {})
    if build_json is None:
        ckpt = _built(tmp_path / "ckpt", "Qwen/Qwen3-0.6B", original)
    else:  # a published checkpoint: build.json alone
        ckpt = _built_json(tmp_path / "ckpt", source_repo=build_json)
    args = argparse.Namespace(
        checkpoint=str(ckpt), original=given, prompts=tmp_path / "p.jsonl", tp=1, out=None
    )
    kl.run(args)
    return seen["model"]


def test_run_reads_the_recorded_revision_and_says_so(tmp_path, monkeypatch, capsys):
    assert _run_with(tmp_path, monkeypatch, SHA, None) == f"Qwen/Qwen3-0.6B@{SHA}"
    assert f"revision {SHA}, from resolved.yaml" in capsys.readouterr().err


def test_run_reads_the_pinned_commit_build_json_records_and_names_the_file(tmp_path, monkeypatch, capsys):
    """The published checkpoint's line says where its commit came from: build.json."""
    pinned = f"Qwen/Qwen3-0.6B@{SHA}"
    assert _run_with(tmp_path, monkeypatch, None, None, build_json=pinned) == pinned
    assert f"revision {SHA}, from build.json" in capsys.readouterr().err


def test_a_hub_checkpoint_reads_the_original_its_build_json_pins(tmp_path, monkeypatch, capsys):
    """`qda-kl kl qda-lab/ckpt` downloads the checkpoint first; its build.json then names the original."""
    import huggingface_hub
    import qda_kl.refcache as refcache
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    snapshot = tmp_path / "hub" / "models--qda-lab--ckpt" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "build.json").write_text(json.dumps({"source_repo": f"Qwen/Qwen3-0.6B@{SHA}"}))
    seen = {}

    def local(model):
        if Path(model).is_dir():
            return Path(model)
        name, _, revision = str(model).partition("@")
        if name == "qda-lab/ckpt":
            return snapshot
        seen["original"] = str(model)
        return tmp_path / "hub" / "models--x" / "snapshots" / revision

    class Stop(Exception):
        pass

    def offline(*args, **kwargs):
        raise OSError("offline")

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(huggingface_hub, "list_repo_files", offline)
    monkeypatch.setattr(refcache, "local_model", local)
    monkeypatch.setattr(refcache, "cached", stop)
    monkeypatch.setattr("qda_kl.reference.default_sampling", lambda source: {})
    args = argparse.Namespace(
        checkpoint="qda-lab/ckpt", original=None, prompts=tmp_path / "p.jsonl", tp=1, out=None
    )
    kl.run(args)
    assert seen["original"] == f"Qwen/Qwen3-0.6B@{SHA}"
    assert f"revision {SHA}, from build.json" in capsys.readouterr().err


def test_the_commit_is_read_through_a_symlinked_cache(tmp_path, monkeypatch):
    """`snapshot_download` resolves the cache dir's symlinks; HF_HUB_CACHE keeps the configured form."""
    from huggingface_hub import constants

    real = tmp_path / "real"
    (real / "models--x" / "snapshots" / SHA).mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(link))
    source = real / "models--x" / "snapshots" / SHA
    assert (
        kl.revision_note(f"Qwen/Qwen3-0.6B@{SHA}", source, None, "build.json")
        == f"revision {SHA}, from build.json"
    )


def test_an_explicit_original_overrides_the_recorded_revision(tmp_path, monkeypatch, capsys):
    other = "a" * 40
    assert _run_with(tmp_path, monkeypatch, SHA, f"Qwen/Qwen3-0.6B@{other}") == f"Qwen/Qwen3-0.6B@{other}"
    assert f"revision {other}, from --original" in capsys.readouterr().err


def test_with_no_recorded_revision_run_reads_main_and_names_the_commit(tmp_path, monkeypatch, capsys):
    assert _run_with(tmp_path, monkeypatch, None, None) == "Qwen/Qwen3-0.6B"
    assert (
        f"revision {'f' * 40}: the checkpoint records no commit, so the Hub's main" in capsys.readouterr().err
    )


def _stage(path, model):
    path.mkdir(parents=True)
    (path / "resolved.yaml").write_text(f"model: {model}\nout: {path.name}\n")
    return path


def test_a_later_stage_of_a_recipe_resolves_to_the_model_the_first_stage_read(tmp_path):
    """A later stage records the earlier stage's local output as its model (`./start`, relative to where the
    recipe ran); that output is a QDA checkpoint, so the chain is followed back to the original."""
    _stage(tmp_path / "start", "Qwen/Qwen3.5-9B@c0ffee")
    final = _stage(tmp_path / "final", "./start")
    assert kl.original_of(final) == "Qwen/Qwen3.5-9B@c0ffee"
    third = _stage(tmp_path / "third", str(final))  # an absolute path, two stages deep
    assert kl.original_of(third) == "Qwen/Qwen3.5-9B@c0ffee"
    plain = tmp_path / "bf16"  # a local original: a model directory, not a QDA checkpoint
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    assert kl.original_of(_stage(tmp_path / "other", str(plain))) == str(plain.resolve())


def test_a_broken_chain_is_refused_naming_original_with_a_revision(tmp_path, capsys):
    """An earlier stage whose own model is not on this machine leaves the original unknown: refused, not guessed."""
    _stage(tmp_path / "start", str(tmp_path / "gone"))
    final = _stage(tmp_path / "final", "./start")
    with pytest.raises(kl.Refused) as refused:
        kl.original_of(final)
    assert "the original model is unknown" in str(refused.value)
    assert refused.value.fix == "pass --original ID@REVISION (the model the recipe's first stage read)"
    args = argparse.Namespace(checkpoint=str(final), original=None, prompts=None, tp=1, out=None)
    assert kl.run(args) == 2
    err = capsys.readouterr().err
    assert "the original model is unknown" in err and "fix:" in err and "--original ID@REVISION" in err


def test_then_the_config_and_the_model_card(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"base_model": "org/base"}))
    assert kl.original_of(tmp_path) == "org/base"
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "README.md").write_text("---\nbase_model:\n- org/card\n---\n# card\n")
    assert kl.original_of(tmp_path) == "org/card"


def test_with_nothing_to_go_on_it_refuses_naming_original(tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}")
    args = argparse.Namespace(checkpoint=str(tmp_path), original=None, prompts=None, tp=1, out=None)
    assert kl.run(args) == 2
    assert "pass --original M" in capsys.readouterr().err


@pytest.mark.parametrize("card", ["---\nbase_model: [a/x, b/y]\n---\n", "no front matter\n"])
def test_an_ambiguous_or_missing_card_names_no_original(tmp_path, card):
    (tmp_path / "README.md").write_text(card)
    assert kl.original_of(tmp_path) is None


def test_a_recorded_local_original_that_is_not_here_is_refused_naming_original(tmp_path, capsys):
    (tmp_path / "resolved.yaml").write_text(f"model: {tmp_path / 'gone'}\n")
    args = argparse.Namespace(checkpoint=str(tmp_path), original=None, prompts=None, tp=1, out=None)
    assert kl.run(args) == 2
    err = capsys.readouterr().err
    assert "not on this machine" in err and "pass --original M" in err


@pytest.mark.parametrize(
    ("built", "table", "kernels"),
    [
        (None, None, "kernels: dense MarlinLinearKernel;"),
        # a checkpoint built for another card is checked against this card's plan, and the line says so
        (
            "sm90@0.33.0",
            "sm120",
            "kernels: dense MarlinLinearKernel (this card's sm120 plan; the checkpoint was built for sm90);",
        ),
        # on the build's card, when a recorded kernel is one this card's table no longer plans
        (
            "sm90@0.33.0",
            "sm90",
            "kernels: dense MarlinLinearKernel (this card's sm90 plan; a recorded kernel is no longer planned on this card);",
        ),
    ],
)
def test_the_reading_names_the_original_the_domains_and_the_engine(
    tmp_path, monkeypatch, capsys, built, table, kernels
):
    """One reading per call: KL, per-domain KL, the engine and kernels, and whether the reference was cached."""
    import qda_kl.cli as cli
    import qda_kl.refcache as refcache

    ckpt, orig = tmp_path / "ckpt", tmp_path / "orig"
    for d in (ckpt, orig):
        d.mkdir()
        (d / "config.json").write_text("{}")
    (ckpt / "resolved.yaml").write_text(f"model: {orig}\n")
    if built:
        (ckpt / "build.json").write_text(json.dumps({"target": built}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    monkeypatch.setattr(refcache, "cached", lambda *a, **k: (tmp_path / "ref", False))

    def score(config, overrides):
        out = tmp_path / "out"
        (out / "score.json").write_text(
            json.dumps(
                {
                    "kl": 0.0251,
                    "domains": {"math": 0.02, "code": 0.03},
                    "vllm": "0.33.0",
                    "tp": 1,
                    "kernels": {"dense": ["MarlinLinearKernel"]},
                    "kernels_from": table,
                    "tokens": 1000,
                    "flip_rate": 0.05,
                }
            )
        )
        return 0

    monkeypatch.setattr(cli, "_score_config", score)
    args = argparse.Namespace(
        checkpoint=str(ckpt), original=None, prompts=prompts, tp=1, out=tmp_path / "out"
    )
    assert kl.run(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("KL 0.025100") and f"original {orig}" in out[0] and "reference cached" in out[0]
    assert out[1].split() == ["code", "0.030000"] and out[2].split() == ["math", "0.020000"]
    assert f"vLLM 0.33.0, tp 1, {kernels}" in out[3]


def test_the_default_prompts_are_the_cached_corpus_holdout(tmp_path, monkeypatch):
    """The corpus module owns where its cached splits live; kl only asks it for the holdout."""
    from qda_kl import corpus

    holdout = tmp_path / "corpus.holdout.jsonl"
    seen = []

    def cached_splits():
        seen.append(True)
        return {"eval": tmp_path / "e.jsonl", "holdout": holdout, "search96": tmp_path / "s.jsonl"}

    monkeypatch.setattr(corpus, "cached_splits", cached_splits)
    assert kl.default_prompts() == holdout and seen == [True]


@pytest.mark.parametrize("where", ["file", "directory", "hub"])
def test_a_gguf_is_refused_before_any_reference_is_built(tmp_path, monkeypatch, capsys, where):
    """A `.gguf`, a directory of one, or a Hub repo of one: two lines pointing at verify.md, no reference."""
    import huggingface_hub
    from qda_kl import refcache

    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    (tmp_path / "README.md").write_text("card\n")
    given = {
        "file": str(tmp_path / "model-Q4_K_M.gguf"),
        "directory": str(tmp_path),
        "hub": "org/model-GGUF@abc",
    }
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo, revision=None: ["README.md", "model-Q4_K_M.gguf"]
    )
    monkeypatch.setattr(refcache, "cached", lambda *a, **k: pytest.fail("a reference was built for a GGUF"))
    args = argparse.Namespace(checkpoint=given[where], original=None, prompts=None, tp=1, out=None)
    assert kl.run(args) == 2
    err = capsys.readouterr().err
    assert "a GGUF (model-Q4_K_M.gguf)" in err and '"A GGUF file"' in err and "fix:" in err


def test_a_checkpoint_with_safetensors_beside_a_gguf_is_not_refused_as_one(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert kl.gguf_files(str(tmp_path)) == []


def _race(tmp_path, monkeypatch, first_builds: bool, progress_s: float = 300.0):
    """Two `qda-kl kl` processes on one original: the first holds the reference lock while its (fake) build
    waits for a go-ahead, the second starts once the first is building and waits on the lock. Returns each
    process's exit code, stdout and stderr, and the builds that ran."""
    import multiprocessing
    import sys
    import time

    import qda_kl.cli as cli
    import qda_kl.reference as reference
    import yaml
    from qda_kl import refcache

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(refcache, "PROGRESS_S", progress_s)
    orig = tmp_path / "orig"
    orig.mkdir()
    (orig / "config.json").write_text("{}")
    (orig / "model.safetensors").write_bytes(b"w" * 64)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "prompt": "x"}\n')
    building, go, calls = tmp_path / "building", tmp_path / "go", tmp_path / "calls"

    @functools.wraps(reference.build)  # the key reads the build's own defaults from its signature
    def build(model, prompts, out, **settings):
        with calls.open("a") as f:
            f.write(f"{os.getpid()}\n")
        building.touch()
        while not go.exists():
            time.sleep(0.05)
        if not first_builds:
            raise RuntimeError("the engine died")
        Path(out).mkdir(parents=True)
        (Path(out) / "manifest.json").write_text(json.dumps({"k": 1, "vocab": 1, "build": {}}))

    def score(config, overrides):
        body = yaml.safe_load(Path(config).read_text())
        assert (Path(body["reference"]) / "manifest.json").is_file()
        (Path(body["out"]) / "score.json").write_text(json.dumps({"kl": 0.01, "tp": 1}))
        return 0

    monkeypatch.setattr(reference, "build", build)
    monkeypatch.setattr(cli, "_score_config", score)

    def one(name):
        ckpt = tmp_path / name
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        (ckpt / "resolved.yaml").write_text(f"model: {orig}\n")
        args = argparse.Namespace(
            checkpoint=str(ckpt), original=None, prompts=prompts, tp=1, out=tmp_path / f"{name}.out"
        )

        def main():
            sys.stdout = open(tmp_path / f"{name}.stdout", "w", buffering=1)  # noqa: SIM115
            sys.stderr = open(tmp_path / f"{name}.stderr", "w", buffering=1)  # noqa: SIM115
            sys.exit(kl.run(args))

        return multiprocessing.get_context("fork").Process(target=main)

    def until(cond):
        end = time.monotonic() + 30
        while not cond():
            assert time.monotonic() < end
            time.sleep(0.05)

    first, second = one("first"), one("second")
    first.start()
    until(building.exists)
    second.start()
    err2 = tmp_path / "second.stderr"
    until(lambda: err2.exists() and "waiting for it" in err2.read_text())
    if progress_s < 60:  # let the waiter pass several progress intervals while the first still builds
        until(lambda: err2.read_text().count("still waiting for ") >= 2)
    go.touch()
    for p in (first, second):
        p.join(30)
    read = {
        f: (tmp_path / f).read_text()
        for f in ("first.stdout", "second.stdout", "first.stderr", "second.stderr")
    }
    return first.exitcode, second.exitcode, read, calls.read_text().split()


def test_a_second_run_waiting_on_the_reference_scores_with_the_one_the_first_built(tmp_path, monkeypatch):
    rc1, rc2, read, calls = _race(tmp_path, monkeypatch, first_builds=True)
    assert (rc1, rc2) == (0, 0), read
    assert len(calls) == 1  # the second built nothing: it read the first one's reference
    assert "reference built now" in read["first.stdout"] and "reference cached" in read["second.stdout"]


def test_a_second_run_whose_first_released_without_building_refuses_in_two_lines(tmp_path, monkeypatch):
    rc1, rc2, read, calls = _race(tmp_path, monkeypatch, first_builds=False)
    assert rc1 != 0 and rc2 == 2 and len(calls) == 1
    lines = read["second.stderr"].splitlines()[-2:]
    assert lines[0].startswith("error: reference cache: the other run building ")
    assert "released it without building it" in lines[0] and f"pid {calls[0]} " in lines[0]
    assert lines[1].startswith("fix:   ") and "log" in lines[1]
    assert not read["second.stdout"]


def test_a_waiting_run_says_whom_it_waits_for_and_then_scores(tmp_path, monkeypatch):
    """No timeout: a healthy build of a large reference runs for an hour. The waiter says, every interval, the
    holder's pid and the time waited, and scores with the reference once the holder has built it."""
    rc1, rc2, read, calls = _race(tmp_path, monkeypatch, first_builds=True, progress_s=0.2)
    assert (rc1, rc2) == (0, 0) and len(calls) == 1, read
    still = [line for line in read["second.stderr"].splitlines() if "still waiting for " in line]
    assert len(still) >= 2 and all(f"pid {calls[0]} " in line and " s so far)" in line for line in still)
    assert "reference cached" in read["second.stdout"]
