"""Codex's tool calls at `isolation.level: none`: read in place from the run's Humanize epic (no wall, no bwrap)."""

import json
import os

from qda_agent import agents, codexlog, flow
from qda_agent.codexlog import SOURCE, epic_calls
from qda_agent.config import RunConfig
from test_codexlog import TWO_CALLS, rollout


def config(tmp_path, backend="codex"):
    models = {"codex": "gpt-5.6-sol", "claude": "claude-opus-5-5"}
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        agent={"backend": {"name": backend, "model": models[backend], "effort": "low"}},
        isolation={"level": "none"},
    )
    cfg.run_dir.mkdir()
    return cfg


def test_the_epic_is_read_in_place_with_the_same_entries(tmp_path):
    sessions = tmp_path / "epic" / "sessions" / "codex"
    rollout(sessions, lines=TWO_CALLS)
    entries, refused = epic_calls(sessions / "sessions")
    assert refused == []
    assert [e["tool"] for e in entries] == ["clock.sleep", "exec"]  # oldest first
    assert {e["source"] for e in entries} == {SOURCE}


def test_a_link_in_the_epic_is_not_followed(tmp_path):
    sessions = tmp_path / "epic" / "sessions" / "codex"
    rollout(sessions, lines=TWO_CALLS)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("\n".join(TWO_CALLS) + "\n")
    os.symlink(outside, sessions / "sessions" / "rollout-2026-11-03T08-00-00-b.jsonl")
    entries, refused = epic_calls(sessions / "sessions")
    assert len(entries) == 2 and refused == ["rollout-2026-11-03T08-00-00-b.jsonl: a link; not followed"]
    assert epic_calls(tmp_path / "missing")[0] == []


def test_a_codex_run_at_level_none_logs_its_calls(tmp_path, monkeypatch):
    from hmz.runtime import epic

    cfg = config(tmp_path)
    run_epic = tmp_path / "hmz" / "20261103T075600.000Z-1"
    rollout(run_epic / "sessions" / "codex", lines=TWO_CALLS)
    monkeypatch.setattr(epic, "epics", lambda workspace=None: [tmp_path / "hmz" / "older", run_epic])
    log = flow._Log(cfg.run_dir / "flow.jsonl", lambda: 1.0)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", lambda: 1.0)
    agents.log_codex_calls(cfg, tools, log)
    got = [json.loads(x) for x in (cfg.run_dir / "tools.jsonl").read_text().splitlines()]
    assert [g["source"] for g in got] == [SOURCE, SOURCE] and tools.counts["tool"] == 2
    said = [json.loads(x) for x in (cfg.run_dir / "flow.jsonl").read_text().splitlines()]
    assert said == [{"t": 1.0, "ev": "codex_log", "entries": 2, "refused": []}]


def test_a_run_without_codex_reads_no_session_log(tmp_path, monkeypatch):
    cfg = config(tmp_path, backend="claude")
    monkeypatch.setattr(codexlog, "epic_calls", lambda sessions: (_ for _ in ()).throw(AssertionError))
    log = flow._Log(cfg.run_dir / "flow.jsonl", lambda: 1.0)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", lambda: 1.0)
    agents.log_codex_calls(cfg, tools, log)
    assert (cfg.run_dir / "flow.jsonl").read_text() == ""
