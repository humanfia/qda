"""The disk cap is off unless set by hand, custody or not, and the launch preflight's line for one set."""

from __future__ import annotations

from pathlib import Path

import yaml
from qda_agent.config import RunConfig
from qda_agent.flow import announce_disk_cap
from qda_agent.judge.config import MIN_FREE_BYTES
from test_judge import llama_paths

BUDGET = 22_000_000_000


def _config(tmp_path: Path, monkeypatch, **judge) -> RunConfig:
    """Load a llama.cpp row whose judge section is `judge`, with a budget and a custody dir."""
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    (tmp_path / "llama-server").write_bytes(b"")
    raw = {
        "model": "tiny/model",
        "format": "w4a16",
        "budget_gpu_hours": 0.1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "sandbox": "bwrap",
        "contamination": "none",
        "baselines": [str(tmp_path / "base" / "gptq")],
        "commands": None,
        "judge": {
            "engine": "llama.cpp",
            "budget_bytes": BUDGET,
            "engine_floor": 0.000273,
            "llamacpp": llama_paths(tmp_path, monkeypatch),
            "custody_dir": str(tmp_path / "custody"),
            **judge,
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return RunConfig.load(path)


def test_an_unset_cap_stays_unset_with_custody_and_the_preflight_prints_nothing(
    tmp_path, monkeypatch, capsys
):
    """No default cap: 10 x `budget_bytes` was an estimate of what custody holds. The run's disk is guarded
    by its live free-space floor (`judge.min_free_bytes`) instead."""
    cfg = _config(tmp_path, monkeypatch)
    assert cfg.judge.max_run_bytes is None and cfg.resolved()["judge"]["max_run_bytes"] is None
    assert cfg.judge.min_free_bytes == MIN_FREE_BYTES
    assert cfg.judge.disk_cap_line() is None
    announce_disk_cap(cfg)
    assert capsys.readouterr().err == ""


def test_a_vllm_row_with_custody_needs_no_cap(tmp_path):
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    raw = {
        "model": "tiny/model",
        "format": "w4a16",
        "budget_gpu_hours": 0.1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "sandbox": "bwrap",
        "contamination": "none",
        "baselines": [str(tmp_path / "base" / "gptq")],
        "commands": None,
        "judge": {"custody_dir": str(tmp_path / "custody")},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert RunConfig.load(path).judge.max_run_bytes is None


def test_an_explicit_cap_is_kept_and_warned_when_below_the_rule(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, monkeypatch, budget_bytes=21_700_000_000, max_run_bytes=150_000_000_000)
    assert cfg.judge.max_run_bytes == 150_000_000_000
    expected = (
        "disk cap: 150.0 GB (judge.max_run_bytes, explicit) WARNING: below 10 x budget_bytes "
        "(217.0 GB); about 6 builds of 21.7 GB fit"
    )
    assert cfg.judge.disk_cap_line() == expected
    announce_disk_cap(cfg)
    assert expected in capsys.readouterr().err


def test_an_explicit_cap_at_or_above_the_rule_prints_no_warning(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, monkeypatch, max_run_bytes=220_000_000_000)
    assert cfg.judge.disk_cap_line() == "disk cap: 220.0 GB (judge.max_run_bytes, explicit)"
    announce_disk_cap(cfg)
    assert "WARNING" not in capsys.readouterr().err


def test_a_row_without_a_cap_prints_nothing(tmp_path, monkeypatch, capsys):
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "tiny/model",
                "format": "w4a16",
                "budget_gpu_hours": 0.1,
                "cards": [0],
                "run_dir": str(tmp_path / "run"),
                "contamination": "none",
                "baselines": [str(tmp_path / "base" / "gptq")],
                "commands": None,
            }
        )
    )
    cfg = RunConfig.load(path)
    assert cfg.judge.max_run_bytes is None
    announce_disk_cap(cfg)
    assert capsys.readouterr().err == ""
