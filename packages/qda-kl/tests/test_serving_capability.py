"""A format the card cannot run is refused before the engine starts, in one sentence."""

from __future__ import annotations

import json

import pytest
import torch
from qda_kl.engine import open_engine
from qda_kl.serve import check_servable, checkpoint_format
from qda_quant.step import Refused


def checkpoint(tmp_path, fmt: str, name: str = "ckpt"):
    """A directory shaped like a qda checkpoint: ``build.json`` names its format."""
    path = tmp_path / name
    path.mkdir()
    (path / "build.json").write_text(json.dumps({"format": fmt, "serve": {"vllm": "0.33.0"}}))
    return path


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
def test_w4a8_on_a_non_hopper_card_names_the_format_the_capability_and_w4a16(
    tmp_path, monkeypatch, capability
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: capability)
    with pytest.raises(Refused) as e:
        check_servable(checkpoint(tmp_path, "w4a8-sym-g128"))
    text = str(e.value)
    assert "w4a8-sym-g128" in text
    assert "compute capability 9.0 only" in text
    assert f"is compute capability {capability[0]}.{capability[1]}" in text
    assert "Use a w4a16 checkpoint" in text


def test_w4a8_on_hopper_passes_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (9, 0))
    path = checkpoint(tmp_path, "w4a8-sym-g128")
    assert checkpoint_format(path).name == "w4a8-sym-g128"
    check_servable(path)


def test_a_source_model_has_no_build_json_and_no_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 9))
    (tmp_path / "model").mkdir()
    assert checkpoint_format(tmp_path / "model") is None
    check_servable(tmp_path / "model")


def test_a_weight_only_format_is_never_refused_here(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 9))
    check_servable(checkpoint(tmp_path, "w4a16-asym-g128"))


def test_open_engine_refuses_before_it_imports_vllm(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (12, 0))
    path = checkpoint(tmp_path, "w4a8-sym-g128")
    with pytest.raises(Refused, match="Use a w4a16 checkpoint"):
        open_engine(str(path), max_model_len=16)
