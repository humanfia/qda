"""A run's progress: the lines `qda-agent run` streams, and what `qda-agent status` prints on demand."""

import json
from datetime import UTC, datetime
from pathlib import Path

from qda_agent import cli, progress
from qda_agent.judge.ledger import Entry

#: a fixed epoch in mid-January 2027: the search's clock in these tests.
T0 = 1_800_000_000.0


def _run_dir(tmp_path: Path) -> Path:
    """A run directory whose resolved config gives the search a one hour wall clock."""
    run = tmp_path / "run"
    (run / "judge").mkdir(parents=True)
    (run / "resolved.yaml").write_text(json.dumps({"budget_gpu_hours": 2, "cards": [0, 1]}))
    return run


def _flow(run: Path, when: float, ev: str, **fields: object) -> None:
    with (run / "flow.jsonl").open("a") as f:
        f.write(json.dumps({"t": when, "ev": ev, **fields}) + "\n")


def _handin(
    run: Path,
    submitted: float,
    *,
    valid: bool | None = None,
    ended: float | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    entry = Entry(
        id="h1",
        kind="handin",
        src="ckpts/a",
        dir=run / "a",
        digest="d",
        submitted=submitted,
        valid=valid,
        ended=ended,
        metrics=metrics or {},
    )
    with (run / "judge" / "ledger.jsonl").open("a") as f:
        f.write(entry.model_dump_json() + "\n")


def test_the_stream_prints_one_line_per_transition_with_elapsed_and_remaining(tmp_path):
    """A hand-in's lines arrive as they change, stamped with the search's own clock; a rewrite and the judge's
    prune record are not transitions."""
    run = _run_dir(tmp_path)
    _flow(run, T0, "contamination", check="none", files=1, findings=[], skipped=[])
    _flow(run, T0 + 900, "start", deadline=datetime.fromtimestamp(T0 + 3600, UTC).isoformat())
    _handin(run, T0 + 1200)
    _handin(run, T0 + 1200, valid=True, ended=T0 + 1500, metrics={"kl": 0.0216})
    _handin(run, T0 + 1200, valid=True, ended=T0 + 1500, metrics={"kl": 0.0216})  # the same line again
    with (run / "judge" / "ledger.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "prune", "destination": "", "bytes": 1}) + "\n")

    view = progress.Progress(run, now=T0 + 120)
    assert view.stream(T0 + 1600) == [
        "27 min elapsed, ~33 min remaining: the agent phase begins; the deadline is 09:00 UTC",
        "27 min elapsed, ~33 min remaining: h1 ckpts/a: checking",
        "27 min elapsed, ~33 min remaining: h1 ckpts/a: valid: KL 0.0216",
    ]


def test_a_quiet_judge_is_reported_once_the_usual_gap_is_past(tmp_path):
    """Silence longer than the usual gap is a line of its own, at most one per gap, naming the last transition."""
    run = _run_dir(tmp_path)
    _flow(run, T0, "start", deadline=datetime.fromtimestamp(T0 + 3600, UTC).isoformat())
    _handin(run, T0 + 300)
    view = progress.Progress(run, now=T0 + 300)
    assert view.stream(T0 + 840) == [
        "14 min elapsed, ~46 min remaining: the agent phase begins; the deadline is 09:00 UTC",
        "14 min elapsed, ~46 min remaining: h1 ckpts/a: checking",
    ]
    assert view.stream(T0 + 960) == [
        "16 min elapsed, ~44 min remaining: no judge activity for 11 min (last: h1 ckpts/a: checking)"
    ]
    assert view.stream(T0 + 1260) == []  # the notice is not repeated inside the gap
    assert view.stream(T0 + 1560) == [
        "26 min elapsed, ~34 min remaining: no judge activity for 21 min (last: h1 ckpts/a: checking)"
    ]


def test_status_prints_the_live_view_and_the_finished_one(tmp_path):
    """On demand: the clock with the deadline, each entry's newest line, and the end of a closed run."""
    run = _run_dir(tmp_path)
    _flow(run, T0, "start", deadline=datetime.fromtimestamp(T0 + 3600, UTC).isoformat())
    _handin(run, T0 + 1200)
    assert progress.status(run, clock=lambda: T0 + 1740) == [
        "29 min elapsed, ~31 min remaining (the deadline is 09:00 UTC)",
        "h1 ckpts/a: checking",
    ]
    result = {"ended": "deadline", "rule": "handin", "choice": str(run / "work" / "a")}
    (run / "result.json").write_text(json.dumps(result))
    assert progress.status(run, clock=lambda: T0 + 1740) == [
        f"the run finished: deadline; the rule handin picked {run / 'work' / 'a'}",
        "h1 ckpts/a: checking",
    ]
    # No submission beat candidate 0: the rule keeps the baseline, and that is the pick to report.
    result.pop("choice")
    result["baseline"] = str(run / "baselines" / "gptq")
    (run / "result.json").write_text(json.dumps(result))
    assert progress.status(run, clock=lambda: T0 + 1740) == [
        f"the run finished: deadline; the rule handin picked {run / 'baselines' / 'gptq'}",
        "h1 ckpts/a: checking",
    ]
    assert progress.status(tmp_path / "nothing") is None


def test_the_status_command_prints_the_view_and_refuses_a_directory_that_is_no_run(tmp_path, capsys):
    run = _run_dir(tmp_path)  # resolved.yaml only: the flow has not started
    assert cli.main(["status", str(run)]) == 0
    assert capsys.readouterr().out.strip() == "start phase: 0 min so far"
    assert cli.main(["status", str(tmp_path)]) == 2
    assert "is not a run directory" in capsys.readouterr().err
