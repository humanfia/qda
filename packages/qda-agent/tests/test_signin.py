import asyncio
import json
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hmz.coganchor.agents import Failed
from hmz.coganchor.agents.claude import ClaudeCodeAgent, ClaudeCodeAgentConfig, ClaudeCodeSession
from hmz.flows import HarnessRefused, HarnessUnrecoverable
from hmz.runtime.flowing.harnesses import HarnessSession
from hmz.runtime.flowing.harnessing import harness_error
from hmz.runtime.flowing.viewing import SessionView
from hmz.sdk import fakes
from qda_agent import flow, isolation
from qda_agent.agents import AgentBackend
from qda_agent.config import RunConfig
from qda_agent.doctor import check_sign_in
from qda_quant import doctor
from test_flow import write_config
from tests_support import agent


@pytest.mark.parametrize("missing", [None, "_handle", "coganchor", "_id", "_lets_go", "_moved_to"])
async def test_installed_driver_recycles_the_process_and_keeps_the_landed_id(tmp_path, monkeypatch, missing):
    agent = ClaudeCodeAgent(ClaudeCodeAgentConfig(model="test", effort="high"))
    conversation = agent.new(tmp_path)
    assert isinstance(conversation, ClaudeCodeSession)
    assert conversation._id is None and conversation._moved_to is None
    assert callable(conversation._lets_go)
    conversation._id = "landed-session"
    handle = HarnessSession.__new__(HarnessSession)
    handle._session = conversation
    view = SessionView.__new__(SessionView)
    view._handle = handle
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    conversation._proc = process
    try:
        if missing is not None:
            if missing == "_handle":
                monkeypatch.delattr(view, "_handle")
            elif missing == "coganchor":
                monkeypatch.delattr(HarnessSession, "coganchor")
            elif missing == "_lets_go":
                monkeypatch.setattr(conversation, "_lets_go", None)
            else:
                monkeypatch.delattr(conversation, missing)
            assert await asyncio.to_thread(flow._recycle, view) == {
                "outcome": "new session",
                "reason": "driver cannot recycle",
            }
            return
        assert await asyncio.to_thread(flow._recycle, view) == {
            "outcome": "resumed",
            "session_id": "landed-session",
        }
        assert process.poll() is not None and conversation._proc is None
        assert conversation._holding() == ["--resume", "landed-session"]
        conversation._id, conversation._named = None, "init-only"
        assert (await asyncio.to_thread(flow._recycle, view))["reason"] == "no landed turn to resume"
        monkeypatch.setattr(flow, "version", lambda _: "0.3.0")
        assert (await asyncio.to_thread(flow._recycle, view))["reason"] == "driver cannot recycle"
    finally:
        monkeypatch.undo()
        if process.poll() is None:
            process.kill()
        process.wait()
        conversation.close()


def missing_session(session_id="previous-session"):
    failed = Failed(
        1,
        ["claude", "--resume", session_id, "-p", "hi"],
        stderr=f"No conversation found with session ID: {session_id}\n",
    )
    error = harness_error(failed, "claude")
    assert isinstance(error, HarnessUnrecoverable) and not failed.fault
    error.__cause__ = failed
    return error


def test_missing_session_uses_the_unsigned_cli_stderr_and_has_no_distinct_hmz_fault():
    session_id = "00000000-0000-0000-0000-000000000000"
    error = missing_session(session_id)
    assert isinstance(error.__cause__, Failed)
    assert (
        error.__cause__.stderr
        == "No conversation found with session ID: 00000000-0000-0000-0000-000000000000\n"
    )
    assert flow._resume_failure(error, session_id)
    assert not flow._resume_failure(error, "different-session")
    assert not flow._resume_failure(HarnessUnrecoverable(str(error)), session_id)


@pytest.mark.parametrize("failure", [None, "resume", "twice", "late", "auth", "other", "wall"])
async def test_sign_in_restart_resumes_or_falls_back_once(tmp_path, monkeypatch, failure):
    state = {"turns": 0, "now": 0}
    calls = []

    def recycle(session):
        calls.append(session)
        return {"outcome": "resumed", "session_id": "previous-session"}

    async def reply(prompt, session, **_):
        state["turns"] += 1
        if state["turns"] == 1:
            state["now"] = 1
        if state["turns"] == 2:
            if failure in ("resume", "twice"):
                raise missing_session()
            if failure == "auth":
                raise HarnessRefused("Not logged in")
            if failure == "other":
                raise HarnessUnrecoverable("model is unavailable")
            if failure == "wall":
                error = missing_session()
                assert error.__cause__ is not None
                error.__cause__.__cause__ = isolation.WallRefused("wall refused")
                raise error
        if state["turns"] == 3 and failure in ("twice", "late"):
            raise missing_session()
        return "ok"

    monkeypatch.setattr(flow, "_recycle", recycle)
    monkeypatch.setattr(isolation, "sign_in_witness", lambda _: {"claude": (state["now"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    config = write_config(tmp_path, seconds=0.3)
    out = await fakes.run_fake(flow.qda_search, agents={"worker": worker}, params={"config": config})
    assert len(calls) == 1
    fallback = failure in ("resume", "twice")
    assert len(worker.sessions) == (2 if fallback else 1)
    (restart,) = out["sign_in_restarts"]
    assert restart["outcome"] == ("new session" if fallback else "resumed")
    expected = "resume failed (HarnessUnrecoverable)" if fallback else "previous-session"
    assert expected in (tmp_path / "run" / "END-REPORT.md").read_text()
    final = json.loads((tmp_path / "run" / "FINAL.json").read_text())
    assert final["sign_in_restarts"] == [restart]
    events = [json.loads(s) for s in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert [e for e in events if e["ev"] == "restart"][-1]["outcome"] == restart["outcome"]
    if failure in ("twice", "late", "other", "wall"):
        assert out["ended"] == "fatal: HarnessUnrecoverable"
        assert len(worker.prompts) == (2 if failure in ("other", "wall") else 3)
    else:
        assert out["ended"] == "deadline"
    if failure == "auth":
        assert out["failed"] == 1


async def test_a_one_turn_backend_still_gets_one_instruction_per_session(tmp_path, monkeypatch, endpoint):
    state = {"now": 0}

    def no_recycle(_):
        pytest.fail("a one-turn backend must not resume")

    async def reply(prompt, session, **_):
        state["now"] = 1
        assert len(session.prompts) == 1
        return "ok"

    monkeypatch.setattr(flow, "_recycle", no_recycle)
    monkeypatch.setattr(isolation, "sign_in_witness", lambda _: {"dsh": (state["now"], 0)})
    config = write_config(
        tmp_path,
        seconds=0.3,
        agent={
            "backend": {"name": "dsh", "model": "test"},
            "goal": False,
            "resend_wait_s": 0.02,
        },
    )
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(flow.qda_search_plain, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and len(worker.sessions) >= 2
    assert out["sign_in_restarts"][0]["reason"] == "the agent's session takes one instruction"


@pytest.mark.parametrize("reason", ["driver cannot recycle", "no landed turn to resume"])
async def test_an_unresumable_restart_records_a_new_session(tmp_path, monkeypatch, reason):
    state = {"now": 0}

    async def reply(prompt, **_):
        state["now"] = 1
        return "ok"

    monkeypatch.setattr(flow, "_recycle", lambda _: {"outcome": "new session", "reason": reason})
    monkeypatch.setattr(isolation, "sign_in_witness", lambda _: {"claude": (state["now"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(
        flow.qda_search, agents={"worker": worker}, params={"config": write_config(tmp_path, seconds=0.3)}
    )
    assert len(worker.sessions) == 2
    assert out["sign_in_restarts"] == [{"outcome": "new session", "reason": reason}]
    assert reason in (tmp_path / "run" / "END-REPORT.md").read_text()


async def test_recycling_cannot_send_a_goal_after_the_deadline(tmp_path, monkeypatch):
    state = {"now": 0}

    def recycle(_):
        time.sleep(0.35)
        return {"outcome": "resumed", "session_id": "previous-session"}

    async def reply(prompt, **_):
        state["now"] = 1
        return "ok"

    monkeypatch.setattr(flow, "_recycle", recycle)
    monkeypatch.setattr(isolation, "sign_in_witness", lambda _: {"claude": (state["now"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(
        flow.qda_search, agents={"worker": worker}, params={"config": write_config(tmp_path, seconds=0.3)}
    )
    assert out["ended"] == "deadline" and len(worker.prompts) == 1


@pytest.mark.parametrize("ending", ["deadline", "stop", "fatal"])
async def test_a_restart_keeps_one_final_writer_on_every_exit(tmp_path, monkeypatch, ending):
    state = {"turns": 0, "now": 0}
    writes = []
    write_text = Path.write_text

    def record_write(path, *args, **kwargs):
        if path in (tmp_path / "run" / "FINAL.json", tmp_path / "run" / "FINAL.tmp"):
            writes.append(path)
        return write_text(path, *args, **kwargs)

    async def reply(prompt, **_):
        state["turns"] += 1
        state["now"] = 1
        if state["turns"] == 2:
            if ending == "stop":
                raise asyncio.CancelledError
            if ending == "fatal":
                raise HarnessUnrecoverable("model is unavailable")
        return "ok"

    monkeypatch.setattr(Path, "write_text", record_write)
    monkeypatch.setattr(flow, "_recycle", lambda _: {"outcome": "resumed", "session_id": "previous-session"})
    monkeypatch.setattr(isolation, "sign_in_witness", lambda _: {"claude": (state["now"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    config = write_config(tmp_path, seconds=0.3)
    if ending == "stop":
        with pytest.raises(asyncio.CancelledError):
            await fakes.run_fake(flow.qda_search, agents={"worker": worker}, params={"config": config})
    else:
        out = await fakes.run_fake(flow.qda_search, agents={"worker": worker}, params={"config": config})
        assert out["ended"] == ("deadline" if ending == "deadline" else "fatal: HarnessUnrecoverable")
    assert len(writes) == 1
    final = json.loads((tmp_path / "run" / "FINAL.json").read_text())
    assert final["sign_in_restarts"] == [{"outcome": "resumed", "session_id": "previous-session"}]


@pytest.fixture
def unsigned(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.delenv("CODEX_HOME", raising=False)  # the user's codex config is the fake home's
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    for tokens in isolation.CLI_CONFIG.values():
        for var in tokens[3]:
            monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "name,var,kind",
    [
        ("claude", "ANTHROPIC_API_KEY", "API key"),
        ("claude", "ANTHROPIC_AUTH_TOKEN", "gateway token"),
        ("claude", "CLAUDE_CODE_OAUTH_TOKEN", "setup-token OAuth token"),
        ("codex", "OPENAI_API_KEY", "API key"),
        ("codex", "CODEX_API_KEY", "API key"),
        ("dsh", "DEEPSEEK_API_KEY", "API key"),
    ],
)
def test_doctor_names_token_kind_and_variable_without_its_value(
    tmp_path, monkeypatch, unsigned, name, var, kind
):
    monkeypatch.setenv(var, "planted-secret-never-print")
    cfg = RunConfig.load(write_config(tmp_path, agent={"backend": {"name": name, "model": "test"}}))
    line = check_sign_in(cfg)
    assert line.ok and kind in line.detail and var in line.detail
    assert "planted-secret-never-print" not in str(line)


@pytest.mark.parametrize("hours,covered", [(3, True), (0, False)])
def test_doctor_reports_file_expiry_and_long_run_advice(tmp_path, monkeypatch, unsigned, hours, covered):
    chosen = tmp_path / "account"
    chosen.mkdir()
    when = datetime.now(UTC) + timedelta(hours=hours)
    (chosen / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "expiresAt": int(when.timestamp() * 1000),
                    "accessToken": "planted-secret-never-print",
                }
            }
        )
    )
    cfg = RunConfig.load(
        write_config(
            tmp_path,
            seconds=3600,
            agent={
                "backend": {
                    "name": "claude",
                    "sign_in_dir": str(chosen),
                }
            },
        )
    )
    line = check_sign_in(cfg)
    # a sign-in the run's own check lets start is said, not judged: the run refreshes it where it can
    assert line.ok and line.line().split()[0] == "OK"
    monkeypatch.setattr(doctor, "run", lambda _: [line])
    assert doctor.main(Namespace(search=None, set=(), run_dir=None, report=False)) == 0
    assert "credential file in agent.backend.sign_in_dir" in line.detail
    assert str(chosen) not in line.detail
    assert f"covers deadline + 30 min: {'yes' if covered else 'no'}" in line.detail
    assert "claude setup-token" in line.detail and "planted-secret-never-print" not in str(line)


@pytest.mark.parametrize(
    "contents", ["planted-secret-never-print", '{"claudeAiOauth":{"expiresAt":"planted-secret-never-print"}}']
)
def test_expiry_errors_are_fixed_strings(tmp_path, unsigned, contents):
    chosen = tmp_path / "account"
    chosen.mkdir()
    (chosen / ".credentials.json").write_text(contents)
    cfg = RunConfig.load(
        write_config(tmp_path, agent={"backend": {"name": "claude", "sign_in_dir": str(chosen)}})
    )
    line = check_sign_in(cfg)
    assert not line.ok and "expiry unavailable" in line.detail
    assert "planted-secret-never-print" not in str(line)


def test_doctor_reports_account_dir_and_missing_sign_in(tmp_path, monkeypatch, unsigned):
    cfg = RunConfig.load(write_config(tmp_path))
    assert not check_sign_in(cfg).ok
    assert isinstance(cfg.agent.backend, AgentBackend)
    cfg.agent.backend.account = "work"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "planted-secret-never-print")
    line = check_sign_in(cfg)
    assert line.ok and "account dir of account work" in line.detail
    assert "ANTHROPIC_API_KEY" not in line.detail and "planted-secret-never-print" not in str(line)


def test_doctor_names_the_default_source_without_its_path(tmp_path, unsigned):
    chosen = tmp_path / "home" / ".claude"
    chosen.mkdir(parents=True)
    (chosen / ".credentials.json").write_text('{"claudeAiOauth":{"expiresAt":0}}')
    line = check_sign_in(RunConfig.load(write_config(tmp_path)))
    assert "credential file in the default CLI config dir" in line.detail
    assert str(chosen) not in line.detail


def test_run_start_prints_the_sign_in_without_starting_an_agent(tmp_path, monkeypatch, unsigned, capsys):
    from hmz.sdk import Refused
    from qda_agent import backends, cli

    def refuse(*args, **kwargs):
        raise Refused("synthetic stop")

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "planted-secret-never-print")
    monkeypatch.setattr(cli, "pidfd_refusal", lambda _: None)
    monkeypatch.setattr(isolation, "hold_problem", lambda _: None)
    monkeypatch.setattr(backends, "make_run", refuse)
    assert cli.execute(RunConfig.load(write_config(tmp_path))) == 2
    err = capsys.readouterr().err
    assert "setup-token OAuth token via CLAUDE_CODE_OAUTH_TOKEN" in err
    assert "planted-secret-never-print" not in err


@pytest.mark.parametrize(
    "name,source,fallback",
    [
        ("claude", "ANTHROPIC_API_KEY", False),
        ("claude", "ANTHROPIC_AUTH_TOKEN", False),
        ("claude", "CLAUDE_CODE_OAUTH_TOKEN", False),
        ("codex", "OPENAI_API_KEY", False),
        ("codex", "CODEX_API_KEY", False),
        ("claude", "account", False),
        ("claude", "account", True),
    ],
)
def test_conflicting_sign_in_is_a_doctor_miss_and_refuses_before_making_files(
    tmp_path, monkeypatch, unsigned, capsys, name, source, fallback
):
    from qda_agent import backends, cli

    chosen = tmp_path / "account"
    chosen.mkdir()
    backend = {"name": name, "model": "test", "sign_in_dir": str(chosen)}
    agent = {"backend": dict(backend)}
    if fallback:
        agent["fallbacks"] = [backend]
    if source == "account":
        backend["account"] = "work"
        if not fallback:
            agent["backend"] = backend
    else:
        monkeypatch.setenv(source, "planted-secret-never-print")
    config = write_config(tmp_path, agent=agent)
    cfg = RunConfig.load(config)
    message = f"{name}: agent.backend.sign_in_dir and {source} both set; name one sign-in"
    line = check_sign_in(cfg)
    assert line.missed and line.line().startswith("MISS ")
    assert line.detail == message
    monkeypatch.setattr(doctor, "run", lambda _: [line])
    assert (
        doctor.main(Namespace(search=None, set=(), run_dir=None, report=False)) == 3
    )  # a miss is the machine
    capsys.readouterr()

    def unexpected(*args, **kwargs):
        raise AssertionError("the start check must refuse before constructing a run")

    monkeypatch.setattr(cli, "pidfd_refusal", lambda _: None)
    monkeypatch.setattr(isolation, "hold_problem", lambda _: None)
    monkeypatch.setattr(backends, "make_run", unexpected)
    assert cli.main(["run", str(config)]) == 2  # the config names two sign-ins: the input, not the machine
    err = capsys.readouterr().err
    assert err == f"error: {message}\nfix:   name one sign-in in the config\n"
    assert str(chosen) not in err and "planted-secret-never-print" not in err
    assert not cfg.run_dir.exists()
    with pytest.raises(isolation.SignInRefused, match="name one sign-in") as caught:
        with isolation.private_cli_config(cfg):
            unexpected()
    assert str(caught.value) == message
    assert not cfg.run_dir.exists()


@pytest.mark.parametrize("source", ["missing", "unreadable", "uncovered", "conflicting"])
def test_start_sign_in_reporting_preserves_the_run_exit_status(
    tmp_path, monkeypatch, unsigned, capsys, source
):
    from qda_agent import backends, cli, sched

    chosen = tmp_path / "account"
    chosen.mkdir()
    if source == "unreadable":
        (chosen / ".credentials.json").write_text("planted-secret-never-print")
    elif source == "uncovered":
        (chosen / ".credentials.json").write_text('{"claudeAiOauth":{"expiresAt":0}}')
    agent: dict[str, object] = {"backend": {"name": "claude", "sign_in_dir": str(chosen)}}
    if source == "conflicting":
        other = tmp_path / "other-account"
        other.mkdir()
        agent["fallbacks"] = [{"name": "claude", "sign_in_dir": str(other)}]
    cfg = RunConfig.load(write_config(tmp_path, agent=agent))

    class Run:
        epic = None

        def run(self):
            (cfg.run_dir / "result.json").write_text('{"end_kind":"deadline","ended":"deadline"}')

    monkeypatch.setattr(cli, "pidfd_refusal", lambda _: None)
    monkeypatch.setattr(isolation, "hold_problem", lambda _: None)
    monkeypatch.setattr(isolation, "gpu_wall", lambda _: nullcontext())
    monkeypatch.setattr(backends, "run_fallbacks", lambda _: nullcontext())
    monkeypatch.setattr(backends, "make_run", lambda *args, **kwargs: Run())
    monkeypatch.setattr(sched, "kill_every_group", lambda: [])
    assert cli.execute(cfg) == 0
    err = capsys.readouterr().err
    assert "agent sign-in:" in err
    assert "planted-secret-never-print" not in err
    if source == "conflicting":
        assert "conflicting sign-in dirs" in err
    elif source == "uncovered":
        assert "covers deadline + 30 min: no" in err and "claude setup-token" in err


def test_search_doctor_includes_the_sign_in_line(tmp_path, monkeypatch, unsigned):
    from qda_agent import doctor
    from qda_quant.doctor import Context

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "planted-secret-never-print")
    checks = doctor.search(Context(search=write_config(tmp_path)))
    (line,) = [c for c in checks if c.name == "agent sign-in"]
    assert line.ok and "gateway token via ANTHROPIC_AUTH_TOKEN" in line.detail
    assert "planted-secret-never-print" not in str(checks)


def _codex_cfg(tmp_path, spec="codex/m", **extra):
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        agent=agent(spec),
        **extra,
    )


@pytest.mark.parametrize(
    "var,base,expected",
    [
        ("OPENAI_API_KEY", None, "https://api.openai.com/v1"),
        ("OPENAI_API_KEY", "https://relay.invalid/v1", "https://relay.invalid/v1"),
        ("CODEX_API_KEY", None, "https://api.openai.com/v1"),
    ],
)
@pytest.mark.parametrize("level", ["none", "sandbox"])
def test_codex_token_signs_in_through_a_provider_reading_the_variable(
    tmp_path, monkeypatch, unsigned, var, base, expected, level
):
    """`codex app-server` reads no key variable and no OPENAI_BASE_URL (401 Missing bearer): the run's codex
    config names a provider whose `env_key` is the variable, and whose base URL is OPENAI_BASE_URL; the key's
    value is never written. The wall's live dir is copied from this dir (`live_state.sync`), so a walled turn
    reads the same config."""
    import tomllib

    from qda_agent import live_state

    monkeypatch.setenv(var, "planted-secret-never-write")
    if base is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base)
    walled = {"sandbox": "bwrap"} if level != "none" else {}
    cfg = _codex_cfg(tmp_path, isolation={"level": level}, **walled)
    with isolation.private_cli_config(cfg) as dirs:
        text = (dirs["codex"] / "config.toml").read_text()
        assert [p.name for p in dirs["codex"].iterdir()] == ["config.toml"]  # no auth.json: no file sign-in
        live = tmp_path / "live"
        live_state.sync(dirs["codex"], live)
        assert (live / "config.toml").read_text() == text
    assert "planted-secret-never-write" not in text
    conf = tomllib.loads(text)
    provider = conf["model_providers"][conf["model_provider"]]
    assert provider["env_key"] == var and provider["base_url"] == expected
    assert provider["wire_api"] == "responses"
    line = check_sign_in(cfg)
    assert line.ok and f"via {var}" in line.detail
    assert ("OPENAI_BASE_URL" if base else expected) in line.detail
    assert "planted-secret-never-write" not in str(line)


def test_codex_without_a_token_keeps_an_empty_config(tmp_path, monkeypatch, unsigned):
    """An account's provider (or the sign-in file) signs codex in: the run's config names no provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "planted")
    cfg = _codex_cfg(tmp_path, "codex@acct/m")
    with isolation.private_cli_config(cfg) as dirs:
        assert (dirs["codex"] / "config.toml").read_text() == ""
    assert isolation.cli_signin.codex_token_provider({}) is None


def test_codex_reports_only_the_key_it_uses(tmp_path, monkeypatch, unsigned):
    """With both key variables set, the provider reads OPENAI_API_KEY: the doctor row names that one alone."""
    monkeypatch.setenv("OPENAI_API_KEY", "a")
    monkeypatch.setenv("CODEX_API_KEY", "b")
    line = check_sign_in(_codex_cfg(tmp_path))
    assert line.ok and "via OPENAI_API_KEY" in line.detail and "CODEX_API_KEY" not in line.detail


def _own_provider_config(tmp_path, home, text=None):
    """A signed-in codex dir `home` whose config.toml names the provider `relay`."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text("{}")
    (home / "config.toml").write_text(
        text
        or 'model_provider = "relay"\n\n[model_providers.relay]\nbase_url = "https://relay.invalid/v1"\n'
        'env_key = "RELAY_KEY"\n'
    )


def test_codex_with_its_own_provider_is_a_said_row_naming_the_variables(
    tmp_path, monkeypatch, unsigned, capsys
):
    """The run's codex config is written afresh, so a provider of the user's codex config never reaches it. The
    sign-in file still signs the run in, so the doctor's row is said and not a MISS (`qda doctor --search` exits
    0 on it); it and the run's start name the two variables, and with them set the run's own provider signs
    codex in."""
    from qda_agent import backends, cli

    _own_provider_config(tmp_path, tmp_path / "home" / ".codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))  # not where the sign-in is read from
    line = check_sign_in(_codex_cfg(tmp_path))
    assert line.ok and not line.missed and line.detail == f"codex: {isolation.CODEX_PROVIDER_FIX}"
    assert "OPENAI_API_KEY and OPENAI_BASE_URL" in line.detail

    config = write_config(tmp_path, agent={"backend": {"name": "codex", "model": "m"}})
    monkeypatch.setattr(cli, "pidfd_refusal", lambda _: None)
    monkeypatch.setattr(isolation, "hold_problem", lambda _: None)
    monkeypatch.setattr(backends, "make_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        cli.execute(RunConfig.load(config))
    assert f"agent sign-in: codex: {isolation.CODEX_PROVIDER_FIX}\n" in capsys.readouterr().err

    monkeypatch.setenv("OPENAI_API_KEY", "planted-secret-never-print")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.invalid/v1")
    line = check_sign_in(_codex_cfg(tmp_path))
    assert line.ok and "API key via OPENAI_API_KEY; endpoint OPENAI_BASE_URL" in line.detail


def test_codex_with_its_own_provider_and_no_sign_in_file_stays_a_miss(tmp_path, unsigned):
    home = tmp_path / "home" / ".codex"
    _own_provider_config(tmp_path, home)
    (home / "auth.json").unlink()
    assert check_sign_in(_codex_cfg(tmp_path)).missed


@pytest.mark.parametrize(
    ("text", "warned"),
    [
        ('profile = "x"\n\n[profiles.x]\nmodel_provider = "relay"\n', True),
        ('model_provider = "relay"\nprofile = "x"\n\n[profiles.x]\nmodel_provider = "openai"\n', False),
        ('model_provider = "relay"\nprofile = "x"\n\n[profiles.x]\nmodel = "m"\n', True),
        ('model_provider = "relay"\nprofile = "gone"\n\n[profiles.y]\nmodel_provider = "openai"\n', True),
        ('model_provider = "relay"\n\n[profiles.x]\nmodel_provider = "openai"\n', True),
        ('profile = "x"\n\n[profiles.y]\nmodel_provider = "relay"\n', False),
    ],
)
def test_codex_provider_is_the_active_profiles_as_codex_resolves_it(tmp_path, unsigned, text, warned):
    """The active profile's `model_provider` wins over the top level's; without one the top level's counts."""
    _own_provider_config(tmp_path, tmp_path / "home" / ".codex", text)
    line = check_sign_in(_codex_cfg(tmp_path))
    assert line.ok
    assert (isolation.CODEX_PROVIDER_FIX in line.detail) is warned


def test_codex_provider_is_read_from_the_sign_in_dir_the_run_names(tmp_path, unsigned):
    """With `agent.backend.sign_in_dir` the config.toml read is that dir's, not the default dir's."""
    named = tmp_path / "named"
    _own_provider_config(tmp_path, named)
    default = tmp_path / "home" / ".codex"
    _own_provider_config(tmp_path, default, 'model = "m"\n')
    cfg = RunConfig.load(
        write_config(tmp_path, agent={"backend": {"name": "codex", "model": "m", "sign_in_dir": str(named)}})
    )
    assert isolation.CODEX_PROVIDER_FIX in check_sign_in(cfg).detail
    _own_provider_config(tmp_path, named, 'model = "m"\n')
    _own_provider_config(tmp_path, default)  # a provider in the default dir is not the named dir's
    assert isolation.CODEX_PROVIDER_FIX not in check_sign_in(cfg).detail


@pytest.mark.parametrize("text", ['model_provider = "openai"\n', 'model = "m"\n', "not toml ["])
def test_codex_with_the_default_provider_keeps_its_file_sign_in(tmp_path, unsigned, text):
    """Codex's own provider, a config naming none and one that does not parse change nothing."""
    _own_provider_config(tmp_path, tmp_path / "home" / ".codex", text)
    line = check_sign_in(_codex_cfg(tmp_path))
    assert line.ok and "codex: credential file in the default CLI config dir" in line.detail


def test_codex_account_and_key_in_one_run_are_refused(tmp_path, monkeypatch, unsigned):
    """Both agents would read the run's one codex config: the key's provider would reach the account's agent,
    or the key's agent would get no provider and fail with 401."""
    monkeypatch.setenv("OPENAI_API_KEY", "a")
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        agent=agent("codex@acct/m", "codex/m"),
    )
    assert not check_sign_in(cfg).ok
    with pytest.raises(isolation.SignInRefused, match="sign both in the same way"):
        with isolation.private_cli_config(cfg):
            pass
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "codex" not in isolation.cli_signin.sign_in_conflicts(cfg)
