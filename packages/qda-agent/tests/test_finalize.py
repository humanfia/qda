"""`flow.finalize`: the one early close keeps what a run produced, writes FINAL.json and says why."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qda_agent.config import RunConfig
from qda_agent.flow import FINALIZE_AFTER_S, _Log, finalize
from qda_agent.host import Final, HandIn


def run_config(tmp_path: Path, monkeypatch) -> RunConfig:
    # The default sandbox names this process's $TMPDIR as agent-writable whenever it is not /tmp or
    # /var/tmp; pytest's tmp_path is under it then, and the baselines below would be refused as a place the
    # agent can change. That check is not this test's subject, and a fenced mode may point its writable set
    # at any root, so it adds none here.
    from qda_agent.sandbox_settings import Bwrap

    monkeypatch.setattr(Bwrap, "writable", lambda self: [])
    run, baseline = tmp_path / "run", tmp_path / "baseline"
    run.mkdir()
    baseline.mkdir()
    return RunConfig(
        contamination="none",
        model="fake/model",
        format="w4a16",
        cards=[0],
        budget_gpu_hours=1,
        run_dir=run,
        baselines=[baseline],
    )


@pytest.mark.asyncio
async def test_finalize_without_a_host_writes_candidate_0(tmp_path, capsys, monkeypatch):
    """A run whose launch failed before any build ends on candidate 0, with the agent recorded as never run."""
    cfg = run_config(tmp_path, monkeypatch)
    log = _Log(cfg.run_dir / "flow.jsonl", time.time)

    final = await finalize(cfg, None, reason="the launch check failed", log=log)

    saved = json.loads((cfg.run_dir / "FINAL.json").read_text())
    assert saved["baseline"] == str(cfg.baselines[0])
    assert saved["rule"] == "baseline"
    assert saved["agent_ran"] is False
    assert "the agent never ran (the launch check failed)" in saved["reason"]
    assert saved["warnings"] == ["the search was finalized early: the launch check failed"]
    assert final.baseline == cfg.baselines[0]
    events = [json.loads(line)["ev"] for line in (cfg.run_dir / "flow.jsonl").read_text().splitlines()]
    assert "finalize" in events
    lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("qda-agent: finalize: ")
    ]
    assert len(lines) == 1
    assert "the launch check failed" in lines[0] and str(cfg.baselines[0]) in lines[0]


class WorkingHost:
    """A started host whose judge has one valid hand-in and stops when the close does."""

    def __init__(self, pick: HandIn) -> None:
        self.calls: list[str] = []
        self.pick = pick

    def cutoff(self) -> None:
        self.calls.append("cutoff")

    def drain(self, timeout_s: float) -> bool:
        self.calls.append("drain")
        return True

    def final(self) -> Final:
        self.calls.append("final")
        return Final(
            rule="handin",
            choice=self.pick,
            reason=f"the first valid entry of the latest hand-in list ({self.pick.dir.name}, entry 1)",
            chosen_by="handin",
        )

    def write_final(self, path: Path, final: Final | None = None) -> None:
        self.calls.append("write_final")
        path.write_text((final or self.final()).model_dump_json(indent=1) + "\n")

    def stop(self) -> None:
        self.calls.append("stop")


@pytest.mark.asyncio
async def test_finalize_with_a_host_picks_the_best_handin_and_keeps_it(tmp_path, capsys, monkeypatch):
    """An agent turn that cannot go on still delivers the judge's pick, with the reason on FINAL and stderr."""
    cfg = run_config(tmp_path, monkeypatch)
    pick = HandIn(
        id="h1",
        dir=tmp_path / "build-1",
        submitted_at=datetime.now(UTC),
        valid=True,
        metrics={"kl": 0.01},
        higher_is_better={"kl": False},
    )
    host = WorkingHost(pick)

    final = await finalize(
        cfg,
        host,
        reason="the agent API failed for 10 minutes",
        log=_Log(cfg.run_dir / "flow.jsonl", time.time),
    )

    saved = json.loads((cfg.run_dir / "FINAL.json").read_text())
    assert saved["choice"]["dir"] == str(pick.dir)
    assert saved["rule"] == "handin"
    assert saved["warnings"] == ["the search was finalized early: the agent API failed for 10 minutes"]
    assert final.choice is not None and final.choice.dir == pick.dir
    # the close is the deadline's: judge-only queue, the queued checks, the pick, FINAL, stop
    assert host.calls[:4] == ["cutoff", "drain", "final", "write_final"]
    assert host.calls[-1] == "stop" and host.calls.count("stop") == 1
    lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("qda-agent: finalize: ")
    ]
    assert len(lines) == 1 and "the agent API failed for 10 minutes" in lines[0]
    assert str(pick.dir) in lines[0]


@pytest.mark.asyncio
async def test_the_judges_warning_about_the_pick_is_on_stderr_and_in_final_json(
    tmp_path, capsys, monkeypatch
):
    """A hand-in with no KL reading is said once where the run's operator reads: FINAL.json and stderr."""
    cfg = run_config(tmp_path, monkeypatch)
    pick = HandIn(id="h1", dir=tmp_path / "build-1", submitted_at=datetime.now(UTC), valid=True)
    said = "the delivered build build-1 has no KL reading (candidate 0: 0.1929, best scored build: 0.2000)"

    class Warns(WorkingHost):
        def final(self) -> Final:
            return super().final().model_copy(update={"warnings": [said]})

    await finalize(
        cfg, Warns(pick), reason="the disk stayed full", log=_Log(cfg.run_dir / "flow.jsonl", time.time)
    )

    assert said in json.loads((cfg.run_dir / "FINAL.json").read_text())["warnings"]
    assert f"qda-agent: warning: {said}" in capsys.readouterr().err.splitlines()


def test_ten_minute_window_is_the_finalize_cutoff():
    """Both conditions the run watches (API failure, disk below its floor) give up after ten minutes."""
    assert FINALIZE_AFTER_S == 600.0
