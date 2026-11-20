"""A configured sign-in that cannot last the run reaches the public CLI as the contract's two lines.

The status tells the user which answer it is: a config that names two sign-ins is the input (2), while a
credential that is missing, unusable or expiring before the deadline is the machine (3, `qda_quant.exit`).
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from qda_agent import backends, cli, isolation, sched
from qda_agent.config import RunConfig


@pytest.mark.parametrize(
    ("problem", "code"),
    [("missing_file", 3), ("token_conflict", 2), ("no_expiry", 3)],  # the machine, the input, the machine
)
def test_sign_in_refusal_is_a_cli_error_before_a_turn(tmp_path, monkeypatch, capsys, problem, code):
    # its subject is a later refusal: the run dir rule (`isolation.run_dir_in_tmp`) holds wherever tmp_path is
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: None)
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    if problem != "missing_file":
        oauth = {"accessToken": "FAKE_SECRET"}
        if problem != "no_expiry":  # a sign-in under the limit only warns when the run names its own dir
            oauth["expiresAt"] = 1
        (chosen / ".credentials.json").write_text(json.dumps({"claudeAiOauth": oauth}))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    if problem == "token_conflict":
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "FAKE_CONFLICT_SECRET")
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        isolation={"level": "strict"},  # the strict wall's sign-in refusal: `none` skips it by design
        agent={"backend": {"name": "claude", "model": "m", "sign_in_dir": str(chosen)}},
    )
    turns = []

    def run():
        isolation.check_sign_in(cfg, datetime.now(UTC) + timedelta(hours=1))
        turns.append("started")

    monkeypatch.setattr(cli, "pidfd_refusal", lambda cfg: None)
    monkeypatch.setattr(isolation, "hold_problem", lambda cfg: None)
    monkeypatch.setattr(isolation, "gpu_wall", isolation.private_cli_config)
    monkeypatch.setattr(backends, "make_run", lambda *args, **kwargs: SimpleNamespace(run=run, epic=tmp_path))
    monkeypatch.setattr(cli, "close", lambda *args: {})
    monkeypatch.setattr(sched, "kill_every_group", lambda: [])
    assert cli.execute(cfg, worker=object()) == code
    captured = capsys.readouterr()
    lines = captured.err.splitlines()  # the progress line comes first; the refusal is the last one
    assert lines[-1].startswith("error: ") and "Traceback" not in captured.err
    assert "FAKE_SECRET" not in captured.err + captured.out
    assert "FAKE_CONFLICT_SECRET" not in captured.err + captured.out
    assert ".credentials.json" not in captured.err
    assert turns == []
    assert not (cfg.run_dir / "cli" / "claude" / ".credentials.json").exists()
