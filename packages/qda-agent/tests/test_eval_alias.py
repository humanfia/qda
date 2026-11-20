"""`qda-agent evaluate RUN_DIR` is an alias of `qda eval RUN_DIR`, and the run directory qda-agent writes is
the one qda-kl reads (the contract between the two, which share no import in that direction)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from qda_agent import cli, evaluate
from qda_agent.host import Final, HandIn
from qda_agent.judge.config import JudgeSettings
from qda_agent.run_config import RunConfig
from qda_kl.eval import rundir
from qda_quant.cli import main as qda


def test_final_json_as_qda_agent_writes_it_is_read_by_qda_kl(tmp_path: Path) -> None:
    """The keys `qda_kl.eval.rundir` reads are fields of the records qda-agent writes."""
    win, c0 = tmp_path / "win", tmp_path / "c0"
    final = Final(
        rule="best",
        choice=HandIn(dir=win, submitted_at=datetime.now(UTC)),
        baseline=c0,
        reason="r",
        export={"winner": str(tmp_path / "exported")},
        chosen_by="handin",
        reading=True,
        gencheck=False,
    )
    (tmp_path / "FINAL.json").write_text(final.model_dump_json(indent=1) + "\n")  # as flow._write_final does
    assert rundir.winner(tmp_path) == tmp_path / "exported"
    final.export = {}
    (tmp_path / "FINAL.json").write_text(final.model_dump_json(indent=1) + "\n")
    assert rundir.winner(tmp_path) == win
    for key in ("choice", "baseline", "reason", "export", "chosen_by", "reading", "gencheck"):
        assert key in Final.model_fields
    for key in (
        "anchor",
        "serve_python",
        "metric",
        "kl_ref",
        "kl_holdout_ref",
        "engine",
        "kl_layout",
        "llamacpp",
    ):
        assert key in JudgeSettings.model_fields
    for key in ("model", "cards", "baselines", "judge"):
        assert key in RunConfig.model_fields


def _finished(tmp_path: Path) -> Path:
    run, c0, win, ref = (tmp_path / n for n in ("run", "c0", "win", "ref"))
    for path in (run, c0, win, ref):
        path.mkdir()
    judge = {"kl_ref": str(ref), "kl_holdout_ref": str(ref), "anchor": "m", "serve_python": sys.executable}
    resolved = {"model": "m", "cards": [0], "baselines": [str(c0)], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    (run / "FINAL.json").write_text(json.dumps({"baseline": str(c0), "choice": {"dir": str(win)}}))
    return run


def test_the_alias_notes_itself_and_prints_what_qda_eval_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _finished(tmp_path)

    def fake(argv, **kw):
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            kl = 0.01 if body["checkpoint"].endswith("win") else 0.02
            Path(body["out"], "score.json").write_text(json.dumps({"kl": kl}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run)]) == 0
    alias = capsys.readouterr()
    assert alias.err.splitlines()[0] == evaluate.NOTE
    assert qda(["eval", str(run)]) == 0
    direct = capsys.readouterr()
    assert evaluate.NOTE not in direct.err
    assert alias.out == direct.out and "verdict: the winner is better than candidate 0" in direct.out


def test_qda_agent_adds_its_report_to_qda_eval() -> None:
    assert "qda-agent" in rundir.RUN_REPORTS
    assert rundir.RUN_REPORTS["qda-agent"] is evaluate.report
