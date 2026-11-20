"""A build carries the source's ``generation_config.json`` over unchanged, and writes none when the source has none.

Engines read their default sampling and stop ids from that file, so a model built from a recipe must serve with the
same defaults as its source. Both a local directory and a Hub id (fetched through ``snapshot_download``) are covered.
"""

import fnmatch
import json
import shutil
from typing import cast

import huggingface_hub
import pytest
import torch
from qda_quant import BuildConfig
from qda_quant.output.generation import CARD_SAMPLING
from transformers import PreTrainedModel

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids

GENERATION = {
    "bos_token_id": 1,
    "eos_token_id": [3, 5],
    "pad_token_id": 0,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "transformers_version": "5.20.2",
}


def _source(root, *, generation: bool):
    """A tiny Qwen3 with a tokenizer, and a ``generation_config.json`` only when ``generation`` is set."""
    src = root / "m"
    cast(PreTrainedModel, tiny_qwen3(layers=2)).save_pretrained(src)
    fp = save_tiny_tokenizer(src)
    (src / "generation_config.json").unlink(missing_ok=True)  # save_pretrained writes one of its own
    if generation:
        (src / "generation_config.json").write_text(json.dumps(GENERATION, indent=2))
    g = torch.Generator().manual_seed(0)
    calib = root / "c.jsonl"
    write_ids(calib, [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(2)], fp)
    return src, calib


def _build(model, calib, out):
    BuildConfig.from_dict(
        {
            "model": model,
            "format": "w4a16-sym-g32",
            "devices": ["cpu"],
            "steps": [{"rtn": {}}],
            "calib": {"file": {"path": str(calib), "samples": 2, "seqlen": 32}},
            "out": str(out),
        }
    ).build(log=lambda s: None)
    return out


def _hub(monkeypatch, repo_dir, snapshot):
    """Serve ``repo_dir`` as Hub repo ``org/tiny``: a download copies only the files its patterns allow."""

    def fake_download(repo, revision=None, allow_patterns=None):
        assert repo == "org/tiny"
        snapshot.mkdir(parents=True, exist_ok=True)
        for f in repo_dir.iterdir():
            if allow_patterns is None or any(fnmatch.fnmatch(f.name, p) for p in allow_patterns):
                shutil.copy2(f, snapshot / f.name)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)


@pytest.mark.parametrize("via", ["local", "hub"])
def test_the_source_generation_config_is_copied_field_for_field(tmp_path, monkeypatch, via):
    src, calib = _source(tmp_path, generation=True)
    model = str(src)
    if via == "hub":
        _hub(monkeypatch, src, tmp_path / "hub" / "models--org--tiny" / "snapshots" / "0123abcd")
        model = "org/tiny"
    out = _build(model, calib, tmp_path / "out")
    assert json.loads((out / "generation_config.json").read_text()) == GENERATION
    assert (out / "generation_config.json").read_bytes() == (src / "generation_config.json").read_bytes()


@pytest.mark.parametrize("via", ["local", "hub"])
def test_no_generation_config_is_written_when_the_source_has_none(tmp_path, monkeypatch, via):
    src, calib = _source(tmp_path, generation=False)
    model = str(src)
    if via == "hub":
        _hub(monkeypatch, src, tmp_path / "hub" / "models--org--tiny" / "snapshots" / "0123abcd")
        model = "org/tiny"
    out = _build(model, calib, tmp_path / "out")
    assert (out / "config.json").exists()
    assert not (out / "generation_config.json").exists()


def test_a_hub_original_without_one_gets_its_released_cards_sampling(tmp_path, monkeypatch):
    """``Qwen/Qwen3.5-4B`` ships no generation_config.json: the build writes its card's row and both stop ids."""
    src, calib = _source(tmp_path, generation=False)
    cfg = json.loads((src / "config.json").read_text())
    (src / "config.json").write_text(json.dumps({**cfg, "eos_token_id": 3}))
    tok = json.loads((src / "tokenizer_config.json").read_text())
    (src / "tokenizer_config.json").write_text(json.dumps({**tok, "eos_token": "w5"}))  # end of turn: id 5
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "0123abcd"
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision=None, allow_patterns=None: str(
            shutil.copytree(src, snapshot, dirs_exist_ok=True)
        ),
    )
    out = _build("Qwen/Qwen3.5-4B", calib, tmp_path / "out")
    written = json.loads((out / "generation_config.json").read_text())
    assert written == {"do_sample": True, **CARD_SAMPLING["Qwen/Qwen3.5-4B"], "eos_token_id": [3, 5]}
    assert json.loads((out / "build.json").read_text())["generation_config"] == {
        "from_card": "Qwen/Qwen3.5-4B"
    }


def test_an_original_s_own_generation_config_wins_over_its_card(tmp_path, monkeypatch):
    src, calib = _source(tmp_path, generation=True)
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "0123abcd"
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision=None, allow_patterns=None: str(
            shutil.copytree(src, snapshot, dirs_exist_ok=True)
        ),
    )
    out = _build("Qwen/Qwen3.5-4B", calib, tmp_path / "out")
    assert json.loads((out / "generation_config.json").read_text()) == GENERATION
    assert "generation_config" not in json.loads((out / "build.json").read_text())
