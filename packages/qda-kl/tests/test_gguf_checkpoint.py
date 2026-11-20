"""`qda-kl score` on a `.gguf`: decoded against the snapshot, read with vLLM's top-k scorer, the decode removed."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from qda_kl import cli, gguf_checkpoint
from qda_kl.reference import model_identity
from qda_kl.scorers import Outcome, TopKScorer
from qda_quant.step import Refused
from test_cli import fake_result


def _setup(tmp_path: Path, monkeypatch, *, language_only: bool = False, snapshot: str | None = None):
    """A .gguf, a reference naming its model, a fake decoder and a fake scorer that records what it read."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(json.dumps({"model": {"path": str(model)}}))
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    seen: dict = {}

    def decode(path, out, snap):
        seen["decode"] = (Path(path), Path(out), Path(snap))
        Path(out).mkdir(parents=True)
        (Path(out) / "config.json").write_text("{}")
        note = {"note": "language model only"} if language_only else {}
        return {"gguf": {"sha256": "ab" * 32}, "config": note}

    def run(self, checkpoint, reference, *, subset=None):
        seen["read"] = (checkpoint, Path(checkpoint).is_dir(), dict(self.serve_args))
        return Outcome([fake_result()], fake_result())

    monkeypatch.setattr("qda_quant.gguf.decode.decode", decode)
    monkeypatch.setattr(TopKScorer, "run", run)
    body = {"checkpoint": str(gguf), "reference": str(ref), "out": str(tmp_path / "out")}
    if snapshot:
        body["snapshot"] = snapshot
    config = tmp_path / "score.yaml"
    config.write_text(yaml.safe_dump(body))
    return config, gguf, model, seen


def test_a_gguf_is_decoded_against_the_references_model_and_the_reading_names_the_file(tmp_path, monkeypatch):
    config, gguf, model, seen = _setup(tmp_path, monkeypatch)
    assert cli.main(["score", str(config)]) == 0
    out = tmp_path / "out"
    assert seen["decode"] == (gguf, out / "decoded", model)
    assert seen["read"] == (str(out / "decoded"), True, {})  # the decode existed while it was read
    assert not (out / "decoded").exists()  # and is gone after
    got = json.loads((out / "score.json").read_text())
    assert got["kl"] == 0.02 and got["checkpoint"]["path"] == str(gguf)
    assert json.loads((out / "gguf-decode.json").read_text())["gguf"]["sha256"] == "ab" * 32


def test_a_language_model_only_decode_is_served_language_only_against_a_named_snapshot(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}")
    config, _, _, seen = _setup(tmp_path, monkeypatch, language_only=True, snapshot=str(other))
    assert cli.main(["score", str(config)]) == 0
    assert seen["decode"][2] == other
    assert seen["read"][2] == {"language_model_only": True}
    assert yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())["snapshot"] == str(other)


OLD, NEW = "0" * 40, "1" * 40


def _hub(tmp_path: Path, monkeypatch, model: dict) -> dict[str, Path]:
    """A Hugging Face cache holding two snapshots of Org/M (refs/main = the newer), a reference built from Org/M."""
    cache = tmp_path / "hub"
    snaps = {}
    for commit, config in ((OLD, '{"v": 1}'), (NEW, '{"v": 2}')):
        snaps[commit] = cache / "models--Org--M" / "snapshots" / commit
        snaps[commit].mkdir(parents=True)
        (snaps[commit] / "config.json").write_text(config)
    (cache / "models--Org--M" / "refs").mkdir()
    (cache / "models--Org--M" / "refs" / "main").write_text(NEW)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ref" / "manifest.json").write_text(json.dumps({"model": {"path": "Org/M", **model}}))
    return snaps


def _sha(path: Path) -> str:
    return hashlib.sha256((path / "config.json").read_bytes()).hexdigest()


def test_the_recorded_revision_wins_over_the_newest_cached_snapshot(tmp_path, monkeypatch):
    config, gguf, _, seen = _setup(tmp_path, monkeypatch)
    snaps = _hub(tmp_path, monkeypatch, {"revision": OLD})
    assert cli.main(["score", str(config)]) == 0
    assert seen["decode"] == (gguf, tmp_path / "out" / "decoded", snaps[OLD])


def test_an_unpinned_reference_decodes_against_the_cached_main_whose_config_it_hashed(tmp_path, monkeypatch):
    config, gguf, _, seen = _setup(tmp_path, monkeypatch)
    snaps = _hub(tmp_path, monkeypatch, {})
    recorded = model_identity("Org/M", hub=True)  # what `qda-kl ref` records
    assert recorded["config_sha256"] == _sha(snaps[NEW]) and recorded["commit"] == NEW
    del recorded["commit"]  # a reference built before `commit` was recorded
    (tmp_path / "ref" / "manifest.json").write_text(json.dumps({"model": recorded}))
    assert cli.main(["score", str(config)]) == 0
    assert seen["decode"] == (gguf, tmp_path / "out" / "decoded", snaps[NEW])


def test_an_unpinned_reference_decodes_against_the_commit_it_recorded_after_main_moves(tmp_path, monkeypatch):
    config, gguf, _, seen = _setup(tmp_path, monkeypatch)
    snaps = _hub(tmp_path, monkeypatch, {})
    (snaps[NEW] / "config.json").write_text(
        (snaps[OLD] / "config.json").read_text()
    )  # a tokenizer-only update
    (tmp_path / "hub" / "models--Org--M" / "refs" / "main").write_text(OLD)
    recorded = model_identity("Org/M", hub=True)
    (tmp_path / "hub" / "models--Org--M" / "refs" / "main").write_text(NEW)
    (tmp_path / "ref" / "manifest.json").write_text(json.dumps({"model": recorded}))
    assert cli.main(["score", str(config)]) == 0
    assert seen["decode"] == (gguf, tmp_path / "out" / "decoded", snaps[OLD])
    assert model_identity("Org/M")["config_sha256"] is None  # a score's checkpoint record reads no cache


@pytest.mark.parametrize("recorded", ["old", None])
def test_an_unpinned_reference_whose_cached_main_moved_is_refused(tmp_path, monkeypatch, capsys, recorded):
    config, _, _, seen = _setup(tmp_path, monkeypatch)
    snaps = _hub(tmp_path, monkeypatch, {})
    model = {"path": "Org/M", "config_sha256": _sha(snaps[OLD]) if recorded else None}
    (tmp_path / "ref" / "manifest.json").write_text(json.dumps({"model": model}))
    assert cli.main(["score", str(config)]) == 2
    err = capsys.readouterr().err
    assert ("not the config.json" if recorded else "neither a revision nor") in err
    assert "set `snapshot:" in err and "decode" not in seen


@pytest.mark.parametrize("offline", [True, False])
def test_an_uncached_hub_id_is_refused_with_its_download_command(tmp_path, monkeypatch, capsys, offline):
    config, _, _, seen = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", offline)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ref" / "manifest.json").write_text(json.dumps({"model": {"path": "Org/M", "revision": OLD}}))
    assert cli.main(["score", str(config)]) != 0
    err = capsys.readouterr().err
    assert f"hf download Org/M --revision {OLD}" in err and "decode" not in seen


def test_a_gguf_with_no_snapshot_is_refused_before_decoding(tmp_path, monkeypatch, capsys):
    config, _, model, seen = _setup(tmp_path, monkeypatch)
    (model / "config.json").unlink()
    assert cli.main(["score", str(config)]) == 2
    assert "set `snapshot:" in capsys.readouterr().err and "decode" not in seen


def test_a_decoder_refusal_is_a_refusal_with_no_reading(tmp_path, monkeypatch, capsys):
    config, *_ = _setup(tmp_path, monkeypatch)

    def refuse(path, out, snap):
        Path(out).mkdir(parents=True)
        raise ValueError(f"{path}: architecture 'llama'; this decoder is the qwen35/qwen35moe inverse")

    monkeypatch.setattr("qda_quant.gguf.decode.decode", refuse)
    assert cli.main(["score", str(config)]) == 2
    err = capsys.readouterr().err
    assert "architecture 'llama'" in err and "fix:" in err
    assert not (tmp_path / "out" / "score.json").exists() and not (tmp_path / "out" / "decoded").exists()


def test_which_checkpoints_are_a_gguf(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    (hf / "extra.gguf").write_bytes(b"GGUF")
    assert gguf_checkpoint.gguf_file(str(hf)) is None  # a model directory vLLM loads
    one = tmp_path / "one"
    one.mkdir()
    (one / "m.gguf").write_bytes(b"GGUF")
    assert gguf_checkpoint.gguf_file(str(one)) == one / "m.gguf"
    (one / "n.gguf").write_bytes(b"GGUF")
    with pytest.raises(Refused, match="2 .gguf files"):
        gguf_checkpoint.gguf_file(str(one))


def test_a_run_dirs_gguf_reading_is_redone_when_the_file_changes(tmp_path):
    """`evaluate` reuses a GGUF score only for the same file bytes: a .gguf has no config.json to tell it apart."""
    import hashlib

    from qda_kl.eval.rundir import _scored, score_config

    gguf = tmp_path / "pick" / "model.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF one")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(json.dumps({"content_sha256": "c"}))
    resolved = {"model": "/m", "judge": {"engine": "llama.cpp", "anchor": "/anchor"}}
    out = tmp_path / "score"
    config = score_config(resolved, gguf.parent, ref, out)
    assert config["checkpoint"] == str(gguf) and config["snapshot"] == "/anchor"
    assert config["scorer"] == {"topk": {"tp": 1, "serve_args": {}}}
    out.mkdir()
    out.with_suffix(".yaml").write_text(yaml.safe_dump(config))
    identity = {"name": "model.gguf", "path": str(gguf), "config_sha256": None}
    score = {"reference": {"content_sha256": "c"}}
    sha = hashlib.sha256(b"GGUF one").hexdigest()
    (out / "score.json").write_text(json.dumps({**score, "checkpoint": {**identity, "sha256": sha}}))
    assert _scored(out, config)
    gguf.write_bytes(b"GGUF two")
    assert not _scored(out, config)
