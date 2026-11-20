"""A build on a job that holds cards refuses a config that puts its card steps on CPU (CPU only).

The configs an agent writes during a run inherit a start's `devices: [cpu]`; on a job with cards, a card
step such as GPTQ then runs on the CPU for hours. The rule is `qda_quant.checks.cpu_step_names` /
`cpu_step_refusal`, shared with the agent's launch check of shipped start configs. It reads the resolved
`devices` and `steps` before any weight is loaded: a refused build writes no `build.json` and mints no
receipt. `devices` unset or naming a card, a mixed list, a job without cards and a CPU-only step list (RTN
alone) all build as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from qda_quant.checks import GPU_STEPS, cpu_step_config, cpu_step_names, cpu_step_refusal
from qda_quant.cli import main as cli
from qda_quant.config import BuildConfig
from qda_quant.step import Refused
from transformers import PreTrainedModel

from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    """A saved tiny Qwen3 and a fingerprinted token-id calibration file."""
    # `engine_util.tiny_qwen3` is annotated as `torch.nn.Module`; the save helper lives on the concrete class
    cast(PreTrainedModel, tiny_qwen3(layers=2, hidden=128, inter=256)).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(4)], sha
    )
    return tmp_path


def config(tmp: Path, **over) -> dict:
    """A config for the tiny model; the default is the cand0 shape: RTN alone, `devices: [cpu]`."""
    return {
        "model": str(tmp / "m"),
        "format": "w4a16",
        "devices": ["cpu"],
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 4, "seqlen": 16}},
        "out": str(tmp / "out"),
        **over,
    }


def card_job(monkeypatch) -> None:
    """A job that holds cards, without needing a GPU: the lease names them."""
    monkeypatch.setenv("GPU_RUN_LEASE", "0")


def no_card_job(monkeypatch) -> None:
    monkeypatch.delenv("GPU_RUN_LEASE", raising=False)


def test_an_rtn_config_builds_on_a_cpu_job(tiny, monkeypatch):
    no_card_job(monkeypatch)
    out = BuildConfig.from_dict(config(tiny)).build()
    assert (out / "build.json").is_file()


def test_an_rtn_config_builds_on_a_card_job(tiny, monkeypatch):
    """RTN is CPU work: a card job serves it as it is, even with `devices: [cpu]`."""
    card_job(monkeypatch)
    out = BuildConfig.from_dict(config(tiny)).build()
    assert (out / "build.json").is_file()


def test_a_card_step_on_cpu_is_refused_before_any_model_read(tiny, monkeypatch):
    card_job(monkeypatch)
    gptq = config(tiny, steps=[{"gptq": {"damp": 0.02}}])

    def touched(*_args, **_kwargs):
        raise AssertionError("the refused build read the model")

    import importlib

    engine = importlib.import_module("qda_quant.engine.build")
    monkeypatch.setattr(engine, "local_source", touched)
    with pytest.raises(Refused) as refused:
        BuildConfig.from_dict(gptq).build()
    message = str(refused.value)
    assert "devices: cpu puts gptq on CPU" in message, message
    assert "the run has cards: remove `devices` or name a card" in message, message
    # a refused build leaves no output directory, build.json or receipt behind
    assert not (tiny / "out").exists()


def test_the_refusal_names_the_config_file_and_its_devices_line(tiny, monkeypatch):
    card_job(monkeypatch)
    path = tiny / "build.yaml"
    path.write_text(yaml.safe_dump(config(tiny, steps=[{"gptq": {"damp": 0.02}}]), sort_keys=False))
    with pytest.raises(Refused) as refused:
        BuildConfig.from_file(path).build()
    message = str(refused.value)
    assert f"{path}:" in message, message
    assert "devices: cpu puts gptq on CPU" in message, message
    assert not (tiny / "out").exists()


def test_qda_build_refuses_and_reports_the_line(tiny, monkeypatch, capsys):
    card_job(monkeypatch)
    path = tiny / "build.yaml"
    path.write_text(yaml.safe_dump(config(tiny, steps=[{"gptq": {"damp": 0.02}}]), sort_keys=False))
    assert cli(["build", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "devices: cpu puts gptq on CPU" in err, err
    assert not (tiny / "out").exists()


def test_a_job_without_cards_builds_a_card_step_on_cpu(tiny, monkeypatch):
    no_card_job(monkeypatch)
    out = BuildConfig.from_dict(config(tiny, steps=[{"gptq": {"damp": 0.02}}])).build()
    assert (out / "build.json").is_file()


def test_devices_unset_builds_on_a_cpu_job(tiny, monkeypatch):
    no_card_job(monkeypatch)
    cfg = config(tiny, steps=[{"gptq": {"damp": 0.02}}])
    del cfg["devices"]
    assert (BuildConfig.from_dict(cfg).build() / "build.json").is_file()


def test_devices_unset_is_never_refused_even_on_a_card_job(monkeypatch):
    """Unset means the job's own cards: the config itself puts nothing on CPU."""
    card_job(monkeypatch)
    assert cpu_step_refusal(None, [{"gptq": {}}]) is None


def test_a_mixed_device_list_is_not_refused():
    """Only an entirely CPU device list puts the step on CPU: a card in the list is card work."""
    assert cpu_step_names(["cpu", 0], [{"gptq": {}}]) == []
    assert cpu_step_names([0, "cpu"], [{"gptq": {}}]) == []
    assert cpu_step_refusal(["cpu", 0], [{"gptq": {}}]) is None


def test_alternate_cpu_spellings_are_cpu_devices(monkeypatch):
    """`CPU`, `cpu:0` and `torch.device("cpu")` resolve to the same CPU device, so they are refused too."""
    card_job(monkeypatch)
    for devices in (["cpu"], ["CPU"], ["cpu:0"], ["cpu", "cpu:0"], [torch.device("cpu")]):
        assert "gptq" in (cpu_step_refusal(devices, [{"gptq": {}}]) or ""), devices


def test_explicit_cpu_index_is_also_a_cpu_device(tiny, monkeypatch):
    """`cpu:0` is a CPU device to `resolve_devices`, so a config naming it is refused like `cpu`."""
    card_job(monkeypatch)
    from qda_quant.engine.planning import resolve_devices

    assert resolve_devices(["cpu:0"])[0].type == "cpu"
    path = tiny / "build.yaml"
    path.write_text(
        yaml.safe_dump(config(tiny, devices=["cpu:0"], steps=[{"gptq": {"damp": 0.02}}]), sort_keys=False)
    )
    import importlib

    engine = importlib.import_module("qda_quant.engine.build")

    def reached(*_args, **_kwargs):
        raise AssertionError("the refused build resolved the model")

    monkeypatch.setattr(engine, "local_source", reached)
    with pytest.raises(Refused) as refused:
        BuildConfig.from_file(path).build()
    assert "puts gptq on CPU" in str(refused.value)


def test_a_cpu_only_step_list_is_not_refused(monkeypatch):
    card_job(monkeypatch)
    assert cpu_step_names(["cpu"], [{"rtn": {}}]) == []
    assert cpu_step_refusal(["cpu"], [{"rtn": {}}]) is None


def test_the_moved_rule_judges_a_config_file_like_the_launch_check(tmp_path):
    """`cpu_step_config` is the single owner of A5's wording: `<path>:<line> devices: cpu puts <step> on CPU`."""
    path = tmp_path / "start.yaml"
    path.write_text("model: m\nformat: w4a16\nsteps:\n  - gptq: {}\ndevices:\n  - cpu\n")
    assert cpu_step_config(path) == f"{path}:5 devices: cpu puts gptq on CPU"
    assert cpu_step_names(["cpu"], [{"rtn": {}}]) == []
    assert "gptq" in GPU_STEPS


def test_a_pathologically_nested_yaml_is_not_a_refusal(tmp_path):
    """PyYAML's composer recurses; that unreadable file is skipped like any other, never fatal."""
    path = tmp_path / "nested.yaml"
    path.write_text("a: " + "[" * 1000 + "]" * 1000 + "\n")
    assert cpu_step_config(path) is None
