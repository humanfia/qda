"""Codex's tool calls from its session log (`qda_agent.codexlog`): read only from a sandboxed copy, bounded, tolerant."""

import json
import os
import shutil

import pytest
from qda_agent import agents, codexlog, flow
from qda_agent.codexlog import SOURCE, session_calls
from qda_agent.config import RunConfig

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="the copy helper needs bwrap")

SECRET = "never-read-this-outside-the-run"


def rec(kind: str, ts: str = "2026-11-03T07:56:21.039Z", **payload) -> str:
    return json.dumps({"timestamp": ts, "type": "response_item", "payload": {"type": kind, **payload}})


def rollout(live, name="rollout-2026-11-03T07-56-20-a.jsonl", lines=()):
    d = live / "sessions" / "2026" / "11" / "03"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(lines) + "\n")
    return d / name


TWO_CALLS = [
    json.dumps({"timestamp": "2026-11-03T07:56:20.000Z", "type": "session_meta", "payload": {"id": "x"}}),
    rec("reasoning", summary=[]),
    rec(
        "custom_tool_call",
        "2026-11-03T07:56:24.500Z",
        name="exec",
        call_id="c1",
        input="tools.exec_command({cmd:'ls'})",
    ),
    rec("custom_tool_call_output", call_id="c1", output=[{"type": "input_text", "text": "TASK.md"}]),
    rec(
        "function_call",
        "2026-11-03T07:56:22.000Z",
        name="sleep",
        namespace="clock",
        arguments='{"duration_ms": 30}',
    ),
    rec("message", role="assistant", content=[]),
]


def test_two_tool_calls_become_hook_shaped_entries(tmp_path):
    live, scratch = tmp_path / "live" / "codex", tmp_path / "scratch"
    scratch.mkdir()
    rollout(live, lines=TWO_CALLS)
    entries, refused = session_calls(live, scratch)
    assert refused == []
    assert entries == [  # oldest first, the hook's schema plus the source
        {
            "t": 1793692582.0,
            "ev": "tool",
            "tool": "clock.sleep",
            "input": {"duration_ms": 30},
            "source": SOURCE,
        },
        {
            "t": 1793692584.5,
            "ev": "tool",
            "tool": "exec",
            "input": {"input": "tools.exec_command({cmd:'ls'})"},
            "source": SOURCE,
        },
    ]
    assert list(scratch.iterdir()) == []  # the copy is gone


def test_links_never_lead_outside_the_run(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "rollout-x.jsonl").write_text(rec("custom_tool_call", name="exec", input=SECRET) + "\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    live = tmp_path / "a" / "codex"  # a link in place of a rollout file
    good = rollout(live, lines=TWO_CALLS[2:3])
    (good.parent / "rollout-b.jsonl").symlink_to(outside / "rollout-x.jsonl")
    entries, refused = session_calls(live, scratch)
    assert len(entries) == 1 and SECRET not in json.dumps(entries)
    assert any("rollout-b.jsonl: a link; not followed" in r for r in refused)

    live = tmp_path / "b" / "codex"  # a link in place of the whole sessions dir
    live.mkdir(parents=True)
    (live / "sessions").symlink_to(outside)
    entries, refused = session_calls(live, scratch)
    assert entries == [] and refused and refused[0].startswith("sessions:")

    live = tmp_path / "c" / "codex"  # a link to a directory of rollouts, inside sessions
    (live / "sessions").mkdir(parents=True)
    (live / "sessions" / "2026").symlink_to(outside)
    entries, refused = session_calls(live, scratch)
    assert entries == [] and refused == ["2026: a link; not followed"]

    link = tmp_path / "d"  # the live dir itself a link
    link.symlink_to(outside)
    assert session_calls(link, scratch) == ([], ["d: no live dir (or not a directory)"])
    assert list(scratch.iterdir()) == []


def test_garbled_huge_and_special_files_are_refused_not_fatal(tmp_path, monkeypatch):
    live, scratch = tmp_path / "live" / "codex", tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(codexlog, "MAX_LINE", 4096)
    monkeypatch.setattr(codexlog, "MAX_FILE", 256 << 10)
    monkeypatch.setattr(codexlog, "MAX_INPUT", 1024)
    garbled = [
        "\x00\xff not json",
        "[" * 100_000,  # nesting too deep for json (RecursionError), and over MAX_LINE anyway
        "[" * 2000 + "]" * 2000,  # deep but short: a RecursionError or a non-record, never a record
        "42",
        json.dumps({"type": "response_item", "payload": "not a dict"}),
        rec("custom_tool_call", name="exec", input="x" * 3000),  # a big input is truncated, not dropped
        rec("function_call", name={"not": "a str"}, arguments="not json", timestamp_is="ignored"),
        rec("function_call", ts="garbage", name="f", arguments="[1, 2]"),
        json.dumps({"type": "response_item", "payload": {"type": []}}),  # unhashable: must not escape
        json.dumps({"type": "response_item", "payload": {"type": {"a": 1}}}),
        '{"type": "response_item", "payload": {"type": "function_call", "name": "n", "arguments": NaN}}',
    ]
    rollout(live, "rollout-garbled.jsonl", garbled)
    big = rollout(live, "rollout-big.jsonl", [rec("custom_tool_call", name="exec", input="y" * 100)])
    big.write_bytes(big.read_bytes() * 3000)  # over MAX_FILE
    os.mkfifo(big.parent / "rollout-fifo.jsonl")  # a FIFO would block a reader
    entries, refused = session_calls(live, scratch)

    assert [e["tool"] for e in entries] == ["exec", "function_call", "f"], refused
    assert set(entries[0]["input"]) == {"truncated"} and len(entries[0]["input"]["truncated"]) == 1024
    assert entries[1]["input"] == {"arguments": "not json"}
    assert entries[2]["input"] == {"arguments": "[1, 2]"} and entries[2]["t"] is None
    assert any("rollout-big.jsonl: larger than" in r for r in refused)
    assert any("rollout-fifo.jsonl: not a regular file" in r for r in refused)
    assert any("rollout-garbled.jsonl: 7 record(s) not read" in r for r in refused), refused
    assert list(scratch.iterdir()) == []


def test_a_codex_run_appends_its_calls_to_tools_jsonl(tmp_path):
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        agent={"backend": {"name": "codex", "model": "gpt-5.6-sol", "effort": "low"}},
        sandbox={"name": "bwrap"},
    )
    cfg.run_dir.mkdir()
    rollout(cfg.run_dir / "live" / "codex", lines=TWO_CALLS)
    log = flow._Log(cfg.run_dir / "flow.jsonl", lambda: 1.0)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", lambda: 1.0)
    tools("tool", tool="Bash", input={"command": "ls"})  # a hook entry, written live
    agents.log_codex_calls(cfg, tools, log)
    got = [json.loads(x) for x in (cfg.run_dir / "tools.jsonl").read_text().splitlines()]
    assert [g.get("source") for g in got] == [None, SOURCE, SOURCE] and tools.counts["tool"] == 3
    said = [json.loads(x) for x in (cfg.run_dir / "flow.jsonl").read_text().splitlines()]
    assert said == [{"t": 1.0, "ev": "codex_log", "entries": 2, "refused": []}]
    assert not [p for p in cfg.run_dir.iterdir() if p.name.startswith("codex-log-")]


def test_the_entry_count_is_capped(tmp_path, monkeypatch):
    live, scratch = tmp_path / "live" / "codex", tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(codexlog, "MAX_ENTRIES", 1)
    rollout(live, lines=TWO_CALLS)
    entries, refused = session_calls(live, scratch)
    assert len(entries) == 1 and any("over 1 tool calls in all" in r for r in refused)


def test_no_failure_of_the_codex_log_costs_the_run_its_result(tmp_path, monkeypatch):
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        agent={"backend": {"name": "codex", "model": "gpt-5.6-sol", "effort": "low"}},
        sandbox={"name": "bwrap"},
    )
    cfg.run_dir.mkdir()

    def boom(live, scratch):
        raise TypeError("unhashable type: 'list'")

    monkeypatch.setattr(codexlog, "session_calls", boom)
    log = flow._Log(cfg.run_dir / "flow.jsonl", lambda: 1.0)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", lambda: 1.0)
    agents.log_codex_calls(cfg, tools, log)  # does not raise
    said = [json.loads(x) for x in (cfg.run_dir / "flow.jsonl").read_text().splitlines()]
    assert said == [
        {
            "t": 1.0,
            "ev": "host_error",
            "step": "codex_log",
            "error": "TypeError",
            "message": "unhashable type: 'list'",
        }
    ]


def test_read_only_dirs_in_the_log_are_copied_read_and_removed(tmp_path):
    live, scratch = tmp_path / "live" / "codex", tmp_path / "scratch"
    scratch.mkdir()
    f = rollout(live, lines=TWO_CALLS)
    for d in (f.parent, f.parent.parent):  # cp -a keeps these modes in the copy
        d.chmod(0o555)
    try:
        entries, refused = session_calls(live, scratch)
    finally:
        for d in (f.parent.parent, f.parent):
            d.chmod(0o755)
    assert len(entries) == 2 and refused == [] and list(scratch.iterdir()) == []
