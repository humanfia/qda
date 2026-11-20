"""A W4A8 run's default judge.format_menu admits the row's own family.

The default menu is the W4 tier's (bf16, w8a16, w4a16); a W3 row adds w3a16, and a W4A8 row (FP8 activations)
must add w4a8 the same way, or prepare writes a config that search's host refuses at start.
"""

from pathlib import Path

from qda_agent.config import RunConfig


def config(tmp_path: Path, fmt: str, judge: dict | None = None) -> RunConfig:
    (tmp_path / "run").mkdir()
    (tmp_path / "base").mkdir()
    (tmp_path / "model").mkdir()
    raw = {
        "model": str(tmp_path / "model"),
        "format": fmt,
        "budget_gpu_hours": 1.0,
        "run_dir": str(tmp_path / "run"),
        "cards": [0],
        "sandbox": "none",
        "baselines": [str(tmp_path / "base")],
        "contamination": "none",
    }
    return RunConfig.model_validate({**raw, "judge": judge} if judge is not None else raw)


def test_a_w4a8_run_admits_its_own_family_by_default(tmp_path):
    cfg = config(tmp_path, "w4a8")
    assert cfg.judge.format_menu == ["bf16", "w8a16-sym-g128", "w4a16-sym-g128", "w4a8"]


def test_a_w3a16_run_keeps_its_menu(tmp_path):
    cfg = config(tmp_path, "w3a16")
    assert "w3a16" in cfg.judge.format_menu


def test_a_w4a16_run_is_unchanged(tmp_path):
    cfg = config(tmp_path, "w4a16")
    assert cfg.judge.format_menu == [
        "bf16",
        "w8a16-sym-g128",
        "w4a16",
    ]  # the row's own family keeps its spelling


def test_a_named_menu_is_not_extended(tmp_path):
    cfg = config(tmp_path, "w4a8", judge={"format_menu": ["bf16", "w4a8"]})
    assert cfg.judge.format_menu == ["bf16", "w4a8"]
