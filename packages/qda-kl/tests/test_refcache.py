"""The reference cache: one build per (original's content, prompts, settings, engine), links for every caller."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import qda_kl.reference as reference
from qda_kl import refcache
from qda_quant.step import Refused

SETTINGS = {"answers": 2, "seed": 17, "max_tokens": 64, "tp": 1, "sampling": {"temperature": 1.0}}


def _model(root: Path, weights: bytes = b"w" * 64) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(weights)
    return root


@pytest.fixture
def builds(tmp_path, monkeypatch):
    """A fake `reference.build` that writes a manifest the way the writer does (into a new directory)."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    calls = []

    @functools.wraps(reference.build)  # the key reads the build's own defaults from its signature
    def build(model, prompts, out, **settings):
        out = Path(out)
        assert not out.exists()  # the writer never overwrites
        calls.append((model, settings))
        out.mkdir(parents=True)
        build_record = {
            **settings,
            "prompts": str(prompts),
            "prompts_sha256": hashlib.sha256(Path(prompts).read_bytes()).hexdigest(),
        }
        (out / "manifest.json").write_text(
            json.dumps({"k": 1, "vocab": 1, "model": {"path": model}, "build": build_record})
        )

    monkeypatch.setattr(reference, "build", build)
    return calls


def test_a_second_call_on_the_same_inputs_reuses_the_build(tmp_path, builds):
    model, prompts = _model(tmp_path / "m"), tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "prompt": "x"}\n')
    first, built = refcache.cached(model, prompts, tmp_path / "run1" / "ref", SETTINGS, log=lambda _: None)
    assert built and len(builds) == 1
    second, built = refcache.cached(model, prompts, tmp_path / "run2" / "ref", SETTINGS, log=lambda _: None)
    assert not built and second == first and len(builds) == 1
    for run in ("run1", "run2"):
        link = tmp_path / run / "ref"
        assert link.is_symlink() and link.resolve() == first.resolve()
        assert (link / "manifest.json").is_file()


def test_a_cache_entry_keeps_its_own_prompts_after_its_builder_is_deleted(tmp_path, builds):
    """A cached reference outlives the run whose data directory built it.

    The entry records its prompt file relative to itself and keeps a copy, so a later run that links it reads
    the cache's copy; before, it recorded the first run's absolute path and a deleted run broke the start.
    """
    model = _model(tmp_path / "m")
    run_a = tmp_path / "runA"
    (run_a / "data").mkdir(parents=True)
    prompts = run_a / "data" / "corpus.search96.jsonl"
    prompts.write_text('{"id": "a", "prompt": "x"}\n')
    first, built = refcache.cached(model, prompts, run_a / "ref", SETTINGS, log=lambda _: None)
    assert built
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["build"]["prompts"] == refcache.PROMPTS_COPY
    copy = first / refcache.PROMPTS_COPY
    assert copy.read_bytes() == prompts.read_bytes()
    shutil.rmtree(run_a)  # the user deletes a finished run, its data directory included

    run_b = tmp_path / "runB"
    (run_b / "data").mkdir(parents=True)
    prompts_b = run_b / "data" / "corpus.search96.jsonl"
    prompts_b.write_text('{"id": "a", "prompt": "x"}\n')
    second, built = refcache.cached(model, prompts_b, run_b / "ref", SETTINGS, log=lambda _: None)
    assert second == first and not built
    from qda_quant.calib.contamination import reference_prompt_file

    found, base, build = reference_prompt_file(second)
    assert base == "reference" and found == copy and found.is_file()
    assert not Path(build["prompts"]).is_absolute()
    # the reader that probes the build's first prompt resolves the same copy, not a path from the cwd
    from types import SimpleNamespace

    from qda_kl import score

    assert score._build_prompt(SimpleNamespace(path=second)) == "x"


def test_an_entry_that_recorded_an_absolute_path_is_migrated_on_a_hit(tmp_path, builds):
    """A cache entry built before this rule (an absolute record, no copy) is given the caller's copy on a hit."""
    model = _model(tmp_path / "m")
    run_a = tmp_path / "runA"
    (run_a / "data").mkdir(parents=True)
    prompts = run_a / "data" / "corpus.search96.jsonl"
    prompts.write_text('{"id": "a", "prompt": "x"}\n')
    first, _ = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    manifest_path = first / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build"]["prompts"] = str(prompts)  # the old shape
    manifest["content_sha256"] = ""
    manifest_path.write_text(json.dumps(manifest))
    (first / refcache.PROMPTS_COPY).unlink()

    run_b = tmp_path / "runB"
    (run_b / "data").mkdir(parents=True)
    prompts_b = run_b / "data" / "corpus.search96.jsonl"
    prompts_b.write_text('{"id": "a", "prompt": "x"}\n')
    refcache.cached(model, prompts_b, run_b / "ref", SETTINGS, log=lambda _: None)
    assert json.loads(manifest_path.read_text())["build"]["prompts"] == refcache.PROMPTS_COPY
    assert (first / refcache.PROMPTS_COPY).read_bytes() == prompts_b.read_bytes()


def test_moving_an_entrys_prompt_record_keeps_its_identity(tmp_path, builds):
    """The identity covers the data, not the build record (a hit rewrites the record).

    `score.json`, `prepare.json` and `qda-agent evaluate` compare the reference's `content_sha256`; migrating an
    entry's prompt record into the entry must not make every reading of it look like one of another reference.
    """
    from qda_kl.store import Manifest, _content_sha256

    model = _model(tmp_path / "m")
    run_a = tmp_path / "runA"
    (run_a / "data").mkdir(parents=True)
    prompts = run_a / "data" / "corpus.search96.jsonl"
    prompts.write_text('{"id": "a", "prompt": "x"}\n')
    first, _ = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    manifest_path = first / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build"]["prompts"] = str(prompts)  # the old shape: a record pointing into the builder's run
    identity = _content_sha256(Manifest.model_validate(manifest))  # what the writer recorded for it
    manifest["content_sha256"] = identity
    manifest_path.write_text(json.dumps(manifest))
    (first / refcache.PROMPTS_COPY).unlink()

    refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    migrated = json.loads(manifest_path.read_text())
    assert migrated["build"]["prompts"] == refcache.PROMPTS_COPY
    assert migrated["content_sha256"] == identity == _content_sha256(Manifest.model_validate(migrated))


def test_the_same_weights_in_another_directory_share_one_reference(tmp_path, builds):
    """The key is the original's content, not its path: a second download of the same model reuses it."""
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    a, _ = refcache.cached(_model(tmp_path / "a"), prompts, None, SETTINGS, log=lambda _: None)
    b, built = refcache.cached(_model(tmp_path / "b"), prompts, None, SETTINGS, log=lambda _: None)
    assert a == b and not built


@pytest.mark.parametrize(
    "change",
    ["weights", "config", "prompts", "settings"],
)
def test_anything_the_reference_depends_on_gives_another_reference(tmp_path, builds, change):
    model, prompts = _model(tmp_path / "m"), tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    first, _ = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    settings = dict(SETTINGS)
    if change == "weights":
        (model / "model.safetensors").write_bytes(b"v" * 64)  # same size: only the content differs
        os.utime(model / "model.safetensors", ns=(1, 1))
    elif change == "config":
        (model / "config.json").write_text('{"a": 1}')
    elif change == "prompts":
        prompts.write_text("{}\n{}\n")
    else:
        settings["seed"] = 18
    second, built = refcache.cached(model, prompts, None, settings, log=lambda _: None)
    assert built and second != first


def test_a_link_to_another_reference_is_replaced_and_a_directory_is_refused(tmp_path, builds):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    out = tmp_path / "ref"
    a, _ = refcache.cached(_model(tmp_path / "a"), prompts, out, SETTINGS, log=lambda _: None)
    b, _ = refcache.cached(_model(tmp_path / "b", b"z" * 64), prompts, out, SETTINGS, log=lambda _: None)
    assert a != b and out.resolve() == b.resolve()
    built_in_place = tmp_path / "in-place"
    built_in_place.mkdir()
    with pytest.raises(Refused, match="not a link to the reference cache"):
        refcache.cached(_model(tmp_path / "c"), prompts, built_in_place, SETTINGS, log=lambda _: None)


def test_an_interrupted_build_is_built_again(tmp_path, builds, monkeypatch):
    """The key file is written first; without the finished directory beside it, the entry is not a hit."""
    model, prompts = _model(tmp_path / "m"), tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    payload = refcache.key_payload(model.resolve(), prompts, SETTINGS)
    target = refcache.entry(payload)
    target.parent.mkdir(parents=True)
    target.with_suffix(".json").write_text("{}")  # a build that died before its rename
    assert refcache.lookup(payload) is None
    got, built = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    assert built and got == target


def test_a_hub_id_is_read_from_its_snapshot(tmp_path, builds, monkeypatch):
    snap = _model(tmp_path / "hub" / "snapshots" / "abc")
    seen = []

    def snapshot_download(repo, revision=None):
        seen.append((repo, revision))
        return str(snap)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    refcache.cached("org/m@abc", prompts, None, SETTINGS, log=lambda _: None)
    assert seen == [("org/m", "abc")] and builds[0][0] == str(snap)


def test_an_omitted_setting_and_the_value_it_derives_to_are_one_reference(tmp_path, monkeypatch):
    """`qda-agent prepare` omits `--sampling`/`--serve-args` (the engine derives them), while
    `qda-agent evaluate` spells out the values the run reference recorded. The key holds the effective settings,
    so both callers ask for, and get, one entry."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    model = _model(tmp_path / "m")
    (model / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_k": 20, "top_p": 0.95})
    )
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    common = {"answers": 2, "seed": 17, "max_tokens": 8192, "k": 128, "tp": 2, "gpu_memory_utilization": 0.85}
    prepared = {**common, "chat_template_kwargs": None, "nccl_nvls": False, "sampling": None}
    prepared |= {"serve_args": None, "serve_env": None, "subsets": None}
    evaluated = {
        **common,
        "chat_template_kwargs": None,
        "nccl_nvls": False,
        "serve_env": None,
        "subsets": None,
    }
    evaluated |= {"sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}
    evaluated |= {"serve_args": {"compilation_config": {"inductor_compile_config": {"deterministic": True}}}}
    first = refcache.key_payload(model.resolve(), prompts, prepared)
    second = refcache.key_payload(model.resolve(), prompts, evaluated)
    assert first["settings"] == second["settings"]
    assert refcache.entry(first) == refcache.entry(second)


def test_a_really_different_sampling_is_another_reference(tmp_path, monkeypatch):
    """The effective value is what is keyed: a sampling that changes a token's distribution is a new entry."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    model = _model(tmp_path / "m")
    (model / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_k": 20, "top_p": 0.95})
    )
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    settings = {"answers": 2, "seed": 17, "max_tokens": 64, "sampling": {"temperature": 1.0}}
    other = {**settings, "sampling": {"temperature": 0.6}}
    first = refcache.key_payload(model.resolve(), prompts, settings)
    second = refcache.key_payload(model.resolve(), prompts, other)
    assert refcache.entry(first) != refcache.entry(second)


def test_a_changed_engine_default_in_code_is_a_new_entry(tmp_path, builds, monkeypatch):
    """The key holds the engine settings the build derives from the flags, so a default changed in code (for
    example the no-JIT settings) never reuses a reference built under the old one."""
    model, prompts = _model(tmp_path / "m"), tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    first, _ = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    real = reference.engine_settings
    monkeypatch.setattr(
        reference,
        "engine_settings",
        lambda env, args, nvls: real({**(env or {}), "A_NEW_ENGINE_DEFAULT": "1"}, args, nvls),
    )
    second, built = refcache.cached(model, prompts, None, SETTINGS, log=lambda _: None)
    assert built and second != first


def test_offline_the_original_without_refs_main_is_refused_before_a_download(tmp_path, monkeypatch):
    """`qda-kl kl` resolves the original through `local_model`: offline, a pinned-only cache names its revision."""
    import huggingface_hub
    from huggingface_hub import constants
    from qda_kl.refcache import local_model
    from qda_quant.step import Refused

    sha = "c" * 40
    (tmp_path / "models--org--model" / "snapshots" / sha).mkdir(parents=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **k: str(tmp_path / repo / k["revision"])
    )

    with pytest.raises(Refused, match="no `main` ref") as got:
        local_model("org/model")
    assert f"org/model@{sha}" in got.value.fix
    assert local_model(f"org/model@{sha}") == tmp_path / "org/model" / sha
