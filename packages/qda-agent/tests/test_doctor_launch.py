"""`qda doctor --search` OK lines that say what was not checked."""

from pathlib import Path


def _shipped(tmp_path):
    """The shipped search config, filled under `tmp_path`, with a baseline that is a checkpoint."""
    from tests_support import filled

    path = filled(Path(__file__).parents[1] / "configs" / "search.yaml", tmp_path / "cfg")
    (tmp_path / "cfg" / "path" / "to" / "baseline-ckpt" / "config.json").write_text("{}")
    return path


def _launch_cfg(tmp_path, *sets):
    from qda_agent.config import RunConfig

    return RunConfig.load(_shipped(tmp_path), [f"run_dir={tmp_path / 'run'}", "cards=[0]", *sets])


def test_a_disk_cap_line_says_when_it_did_not_check_the_builds(tmp_path, monkeypatch):
    """Without a build size the cap is checked against the disk only; the OK line must say so."""
    from qda_agent import doctor as d
    from qda_agent.judge import custody

    monkeypatch.setattr(custody, "build_bytes", lambda judge: None)
    line = d.check_disk_cap(_launch_cfg(tmp_path, "judge.max_run_bytes=1000000000"))
    assert line.ok and "not checked against the run's builds" in line.detail
