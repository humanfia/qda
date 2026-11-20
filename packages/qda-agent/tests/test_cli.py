import json
import os
import re
import shutil
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from hmz.flows import PermissionKind
from hmz.sdk import fakes
from qda_agent import backends, cli, isolation
from qda_agent.config import IsolationSettings, RunConfig
from qda_agent.configfile import ConfigError
from qda_agent.isolation import SANDBOX, env_spec, permission
from qda_agent.sandbox_settings import Bwrap, NoSandbox
from testkit import HOSTS, ScriptedWorker, build_ckpt, scripted, shell_worker
from tests_support import agent, filled
from tests_support import search_config as _search_config

EXAMPLES = Path(__file__).parents[1] / "examples"
CONFIGS = Path(__file__).parents[1] / "configs"
FIXTURES = Path(__file__).parent / "fixtures"


def test_scripted_flow_only(tmp_path):
    assert scripted(FIXTURES / "scripted-flow.yaml", tmp_path / "run", 3) == 0
    run = tmp_path / "run"
    result = json.loads((run / "result.json").read_text())
    assert result["rule"] == "handin" and result["choice"].endswith("work/out/b")
    assert result["goals"] >= 4 and result["epic"]
    report = (run / "END-REPORT.md").read_text()
    assert "rule `handin`" in report and "Humanize trace" in report
    assert (run / "work" / "TASK.md").exists() and (run / "work" / "CLOCK.md").exists()
    # B was read and never generation-checked, so only the check is named as missing, and FINAL.json
    # records the two facts both readers use
    assert "**The delivered checkpoint has no long-generation check (`gencheck`).**" in report
    final = json.loads((run / "FINAL.json").read_text())
    assert final["reading"] is True and final["gencheck"] is False
    assert "Rotation:" not in report  # an unrotated build says nothing about drafts


def test_the_end_report_leads_with_the_submission_and_the_next_command(tmp_path):
    """END-REPORT puts the pick and the commands to serve and score it before the run's own facts, and prints
    times to the second."""
    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 3) == 0
    report = (run / "END-REPORT.md").read_text()
    assert report.index("## Submission:") < report.index("## Run") < report.index("## Readings")
    assert f"```sh\nqda eval {run}\n```" in report
    stamp = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ"
    assert re.search(rf"- start {stamp}, GPU deadline {stamp};", report)
    assert "instructions sent" in report and " tool calls" in report


def test_the_end_report_quotes_the_eval_command_and_drops_it_without_candidate_0(tmp_path):
    """The printed `qda eval` line pastes into a shell as is; a run with no baselines prints none (it would fail)."""
    run = tmp_path / "a run"
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 3) == 0
    assert f"qda eval '{run}'\n" in (run / "END-REPORT.md").read_text()
    bare = tmp_path / "bare"
    assert scripted(FIXTURES / "scripted-flow.yaml", bare, 3, overrides=["baselines=[]"]) == 0
    assert "qda eval" not in (bare / "END-REPORT.md").read_text()


def test_an_unscored_handin_is_named_in_the_end_report(tmp_path):
    """A hand-in the judge never read or checked is said loudly where the result is reported."""

    def script(cfg: RunConfig) -> ScriptedWorker:
        def handin() -> str:
            d = build_ckpt(cfg.workdir, "k2")
            HOSTS[cfg.run_dir].submit(d)
            return "k2: handed in without a reading"

        return ScriptedWorker([handin])

    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 3, script=script) == 0
    report = (run / "END-REPORT.md").read_text()
    assert (
        "**The delivered checkpoint has no KL reading and no long-generation check (`gencheck`).**" in report
    )
    # the audit record survives the run directory's weights: FINAL.json says both facts itself
    final = json.loads((run / "FINAL.json").read_text())
    assert final["reading"] is False and final["gencheck"] is False


def test_a_rotated_handin_names_rotate_draft_in_the_end_report(tmp_path):
    """A checkpoint whose record ran a residual rotation tells the reader to rotate a DFlash/DSpark draft."""

    def script(cfg: RunConfig) -> ScriptedWorker:
        def handin() -> str:
            d = build_ckpt(cfg.workdir, "rot")
            (d / "resolved.yaml").write_text(
                json.dumps({"steps": [{"rotate": {"residual": True, "seed": 3}}]})
            )
            HOSTS[cfg.run_dir].submit(d, kl=0.02)
            return "rot: handed in"

        return ScriptedWorker([handin])

    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 3, script=script) == 0
    report = (run / "END-REPORT.md").read_text()
    assert "**Rotation:** this checkpoint runs in a rotated residual basis" in report
    assert f"qda rotate-draft --checkpoint {run / 'work' / 'out' / 'rot'}" in report
    assert "DFlash / DSpark" in report


def test_scripted_worker_against_the_real_host(tmp_path, humanize_home):
    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-search.yaml", run, 8) == 0
    result = json.loads((run / "result.json").read_text())
    final = json.loads((run / "FINAL.json").read_text())
    # the fixture names no final rule, so the default `handin` decides: b is the last valid hand-in (c is refused)
    assert result["rule"] == final["rule"] == "handin" and result["host_errors"] == 0
    assert 4 <= result["goals"] <= 17  # 8 s at one send per 0.5 s at most (was 57 sends in 6 s)
    assert (
        (Path(final["choice"]["dir"]) / "model.safetensors").read_bytes().endswith(b"b\0\0\0\0\0\0\0")
    )  # b's copy
    assert [h["kind"] for h in final["ledger"]].count("scored") == 3
    said = [json.loads(line) for line in (run / "flow.jsonl").read_text().splitlines()]
    said = "".join(e.get("said", "") for e in said)
    assert "submitted h1 out/b: valid" in said and "refused: out/c" in said
    assert not (run / "rpc.sock").exists()  # the host stopped
    assert cli.main(["run", str(FIXTURES / "scripted-search.yaml"), "--set", f"run_dir={run}"]) == 2
    # The run leaves nothing outside its run dir, and a user can delete that dir.
    assert result["epic"] == str(run / "epic") and (run / "epic" / "epic.jsonl").is_file()
    assert f"`{run / 'epic'}`" in (run / "END-REPORT.md").read_text()
    assert not any((humanize_home / "epics").iterdir())
    sealed = run / "judge" / "eval" / "standin_kl"  # the frozen evaluator: its files stay read-only
    assert not os.access(sealed / "kl.py", os.W_OK) and os.access(sealed, os.W_OK)
    shutil.rmtree(run)


def test_a_stopped_run_still_writes_every_file(tmp_path):
    def stop() -> str:  # Ctrl-C in the middle of the agent's turn
        os.kill(os.getpid(), signal.SIGINT)
        return "stopping"

    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 30, script=lambda _: ScriptedWorker([stop])) == 130
    result = json.loads((run / "result.json").read_text())
    assert result["ended"] == "stopped" and result["rule"] == "handin"
    assert result["epic"] == str(run / "epic") and (run / "FINAL.json").is_file()
    report = (run / "END-REPORT.md").read_text()
    assert "agent phase ended: stopped" in report and "Humanize trace" in report


def test_a_sign_in_refused_at_the_start_exits_three_without_a_traceback(tmp_path, monkeypatch, capsys):
    """The public `qda-agent run` path: a configured sign-in dir that holds no sign-in by the time the wall
    starts is one line on stderr and exit 3 (the machine has no sign-in), not a traceback."""
    # its subject is a later refusal: the run dir rule (`isolation.run_dir_in_tmp`) holds wherever tmp_path is
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in (
        "CLAUDE_CONFIG_DIR",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    chosen, run = tmp_path / "chosen", tmp_path / "run"
    chosen.mkdir()  # the dir the config names exists; its sign-in file does not
    raw = yaml.safe_load((FIXTURES / "scripted-flow.yaml").read_text())
    raw.update(run_dir=str(run), agent=agent("claude/claude-opus-5-5:high"), isolation={"level": "strict"})
    raw["agent"]["backend"]["sign_in_dir"] = str(chosen)
    raw["baselines"] = [str(FIXTURES / "baselines" / name) for name in ("gptq", "awq")]
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw))
    assert RunConfig.load(config).agent.backend.sign_in_dir == chosen  # the config loads
    monkeypatch.setattr(cli, "pidfd_refusal", lambda cfg: None)  # the wall is this test's own
    monkeypatch.setattr(isolation, "hold_problem", lambda cfg: None)
    monkeypatch.setattr(isolation, "gpu_wall", isolation.private_cli_config)
    assert cli.main(["run", str(config)]) == 3
    err = capsys.readouterr().err
    assert "error: agent.backend.sign_in_dir" in err and "no sign-in file" in err
    assert "Traceback" not in err and ".credentials.json" not in err  # no path, no file name


def test_config_refuses_unknown_keys_and_bad_cards(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: m\nformat: w4a16\nbudget_gpu_hours: 1\ncards: [1, 1]\nrun_dir: r\n")
    with pytest.raises(ValueError, match="twice"):
        RunConfig.load(bad)
    bad.write_text("model: m\nformat: w4a16\nbudget_gpu_hours: 1\ncards: [1]\nrun_dir: r\nteacher: on\n")
    with pytest.raises(ValueError, match="teacher"):
        RunConfig.load(bad)


def test_sandbox_permission_and_env():
    assert SANDBOX.online is PermissionKind.NONE and SANDBOX.user is PermissionKind.NONE
    assert SANDBOX.system is PermissionKind.NONE and SANDBOX.local is PermissionKind.ALL
    assert permission(Bwrap()) == SANDBOX
    assert env_spec(Bwrap(), Path("/x/work")) == "local/x/work"


def test_cli_without_goal_gets_the_plain_flow(tmp_path):
    cfg = RunConfig.load(FIXTURES / "scripted-flow.yaml", {"run_dir": tmp_path, "sandbox": "none"})
    cfg.workdir.mkdir(parents=True)
    run = backends.make_run(cfg, FIXTURES / "scripted-flow.yaml", worker=fakes.FakeAgentDriver("pi"))
    assert run.ref.endswith("qda_search_plain")
    run = backends.make_run(cfg, FIXTURES / "scripted-flow.yaml", worker=fakes.FakeAgentDriver("codex"))
    assert run.ref.endswith(":qda_search")


@pytest.mark.parametrize("goal", ["no", "yes", 1, "Read TASK.md in full and pre-register"])
def test_agent_goal_is_a_flag_not_text(tmp_path, goal):
    """Lax bool coercion would load "no" as False and 1 as True: a silent change of prompting mode."""
    base = {"run_dir": tmp_path, "sandbox": "none"}
    with pytest.raises(ConfigError, match="agent.goal is a mode flag"):
        RunConfig.load(FIXTURES / "scripted-flow.yaml", {**base, "agent.goal": goal})
    for flag in ["agent.goal=true", "agent.goal=false", "agent.goal=null"]:  # `--set` parses YAML
        RunConfig.load(FIXTURES / "scripted-flow.yaml", [flag, "sandbox=none", f"run_dir={tmp_path}"])


def test_fenced_run_dir_not_under_tmp(tmp_path):
    from qda_agent import isolation

    cfg = RunConfig.load(
        FIXTURES / "scripted-search.yaml",
        {"run_dir": tmp_path, "sandbox": "bwrap", "isolation.level": "strict"},
    )
    with pytest.raises(ValueError, match="which the agent can read and write at isolation.level strict"):
        isolation.check(cfg)
    isolation.check(
        cfg.model_copy(update={"sandbox": NoSandbox(), "isolation": IsolationSettings(level="none")})
    )


@pytest.mark.parametrize("level", ["sandbox", "strict"])
def test_a_sandboxed_run_dir_under_var_tmp_is_two_lines_at_the_start_and_in_doctor(tmp_path, level):
    """The search's start refuses it before it creates anything (error + fix, exit 2), and `qda doctor
    --search` prints the same line as a MISS; at isolation.level none the same place is fine."""
    from qda_agent import cli, isolation
    from qda_agent.doctor import check_run_dir_place
    from qda_quant.exit import USAGE

    run = Path("/var/tmp") / f"qda-test-run-{tmp_path.name}"
    cfg = RunConfig.load(FIXTURES / "scripted-search.yaml", {"run_dir": run, "isolation.level": level})
    problem, fix, code = cli.start_problem(cfg)
    assert problem == isolation.run_dir_in_tmp(cfg) and "\n" not in problem
    assert f"is under /var/tmp, which the agent can read and write at isolation.level {level}" in problem
    assert fix == isolation.RUN_DIR_FIX and "outside /tmp and /var/tmp" in fix and "work_dir" in fix
    assert code == USAGE and not run.exists()
    line = check_run_dir_place(cfg)
    assert not line.ok and line.detail == problem and line.fix == fix
    plain = cfg.model_copy(update={"sandbox": NoSandbox(), "isolation": IsolationSettings(level="none")})
    assert isolation.run_dir_in_tmp(plain) is None and check_run_dir_place(plain).ok


def test_doctor_passes_a_sandboxed_run_dir_outside_the_temporary_directories(tmp_path, monkeypatch):
    """An OK line names the place; the rule is the start's own, so it passes where the start passes."""
    from qda_agent import isolation
    from qda_agent.doctor import check_run_dir_place

    monkeypatch.setattr(isolation.tempfile, "gettempdir", lambda: "/nonexistent-tmp")
    run = Path.home() / "qda-test-not-created"
    cfg = RunConfig.load(FIXTURES / "scripted-search.yaml", {"run_dir": run, "isolation.level": "sandbox"})
    assert isolation.run_dir_in_tmp(cfg) is None
    assert check_run_dir_place(cfg).ok and not run.exists()


def test_an_invalid_card_selection_never_reaches_a_gpu_probe(tmp_path):
    """The cards skip is decided before a check reads the cards; no card is put in their place."""
    from unittest.mock import patch

    from qda_agent import doctor as d
    from qda_agent import isolation
    from qda_quant.doctor import Context, report

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "remote/model",
                "format": "w4a16",
                "budget_gpu_hours": 1,
                "contamination": "none",
                "cards": [],
                "run_dir": "run",
                "sandbox": "none",
                "toolset": {"name": "qda", "components": []},
                "judge": {"allow_any_format": True, "kl_cmd": ["custom-score"]},
            }
        )
    )

    def never(*args, **kwargs):
        raise AssertionError("a card selection that did not validate must not reach a machine check")

    with (
        patch.object(d, "check_bwrap", side_effect=never) as sandbox,
        patch.object(d, "check_disk_cap", side_effect=never),
        patch.object(d, "check_sign_in", side_effect=never),
        patch.object(d, "_out", return_value=(True, "")),
        patch.object(d.shutil, "which", side_effect=lambda tool: f"/usr/bin/{tool}"),
        patch.object(isolation, "check"),
        patch.object(isolation, "hold_problem", return_value=None),
    ):
        out = report(d.search(Context(search=path)))
    assert not sandbox.called, "a card selection that did not validate must not reach the machine's checks"
    assert out.splitlines()[0].startswith("MISS run config:")
    assert out.endswith("checked 1, 1 MISS, 0 cannot be checked in advance\n"), "no other line runs"


def test_a_config_with_no_miss_skips_no_line(tmp_path):
    """An intact config prints the same lines as before, none of them a skip for a setting."""
    from qda_agent.doctor import search
    from qda_quant.doctor import Context, report

    out = report(search(Context(search=_search_config(tmp_path))))
    assert "\nMISS " not in out and not out.startswith("MISS ") and "needs judge." not in out
    assert out.startswith("OK   run config: Qwen/Qwen3-8B w4a16")
    assert "OK   disk cap" in out


def test_a_host_that_refuses_at_start_leaves_one_line(tmp_path, capsys):
    """The index `ValueError` left `qda-agent run` as 23 frames of traceback and exit 1.

    Humanize imports the flow from its file under the name the file gives it, so a second copy of
    `qda_agent.flow` held the refusal classes and nothing `cli.execute` catches ever matched them.
    """
    from qda_agent import cli

    assert cli.main(["run", str(_search_config(tmp_path, kl_holdout_ref=None))]) == 2
    err = capsys.readouterr().err
    assert "error: contamination-index:" in err and "kl_holdout_ref is unset" in err
    assert "Traceback" not in err


def test_a_wall_refusal_the_run_meets_before_a_turn_is_one_line(tmp_path, monkeypatch, capsys):
    """A `WallRefused` escaping `gpu_wall` left the run as a traceback and exit 1."""
    from types import SimpleNamespace

    from qda_agent import backends, cli, isolation, sched
    from qda_agent.isolation.wall import WallRefused

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
    )

    def raise_wall(cfg):
        raise WallRefused("pid: shared gives the agent's process the host's pid namespace")

    monkeypatch.setattr(isolation, "gpu_wall", raise_wall)
    monkeypatch.setattr(backends, "make_run", lambda *a, **k: SimpleNamespace(run=lambda: None, epic=None))
    monkeypatch.setattr(cli, "close", lambda *a: {})
    monkeypatch.setattr(sched, "kill_every_group", lambda: [])
    assert cli.execute(cfg, worker=object()) == 2
    err = capsys.readouterr().err
    assert "error: pid: shared gives the agent's process" in err and "Traceback" not in err


def test_a_run_finalized_before_its_deadline_exits_3(tmp_path, monkeypatch):
    """An early close (the API or the disk) is the machine's, not the input's: exit 3, not 0 or 2."""
    import contextlib
    from types import SimpleNamespace

    from qda_agent import sched

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
    )
    monkeypatch.setattr(backends, "make_run", lambda *a, **k: SimpleNamespace(run=lambda: None, epic=None))
    monkeypatch.setattr(cli, "close", lambda *a: {"end_kind": "finalize"})
    monkeypatch.setattr(isolation, "gpu_wall", lambda cfg: contextlib.nullcontext())
    monkeypatch.setattr(sched, "kill_every_group", lambda: [])
    assert cli.execute(cfg, worker=object()) == 3


def test_a_fatal_end_the_machine_cannot_fix_exits_2_and_points_to_the_log(tmp_path, monkeypatch, capsys):
    """A fatal end that is no machine problem (here: a model the account is not served) keeps status 2, and
    its last line names the run log, where the agent CLI's own words are."""
    import contextlib
    from types import SimpleNamespace

    from qda_agent import sched

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
    )
    result = {"end_kind": "fatal", "ended": "fatal: ModelUnavailable", "fatal": "ModelUnavailable"}
    monkeypatch.setattr(backends, "make_run", lambda *a, **k: SimpleNamespace(run=lambda: None, epic=None))
    monkeypatch.setattr(cli, "close", lambda *a: result)
    monkeypatch.setattr(isolation, "gpu_wall", lambda cfg: contextlib.nullcontext())
    monkeypatch.setattr(sched, "kill_every_group", lambda: [])
    assert cli.execute(cfg, worker=object()) == 2
    err = capsys.readouterr().err
    assert (
        f"error: the run failed: ModelUnavailable: the agent CLI failed; details in {tmp_path / 'run' / 'flow.jsonl'}\n"
        in err
    )
    assert "fix:" not in err


def test_the_start_names_the_way_out_for_a_reference_whose_prompt_file_is_gone(tmp_path):
    """`judge start` is the start's own refusal (`ReferenceUnreadable`) and its fix is the prepare step."""
    from qda_agent.judge import make_host
    from qda_agent.judge.startup import ReferenceUnreadable

    ref = tmp_path / "ref"
    shutil.copytree(FIXTURES / "ref", ref)
    (ref / "prompts.jsonl").unlink()
    gone = tmp_path / "deleted-run" / "data" / "corpus.search96.jsonl"
    (ref / "manifest.json").write_text(json.dumps({"build": {"prompts": str(gone)}}))
    path = _search_config(tmp_path, kl_ref=str(ref))
    cfg = RunConfig.load(path)
    with pytest.raises(ReferenceUnreadable) as refused:
        make_host(cfg.run_dir, cfg.sched_config(datetime.now(UTC)), cfg.judge_config(), autotick=False)
    assert "cannot read" in " ".join(str(refused.value).split())
    assert "run the `qda-agent prepare` that made it again" in str(refused.value) + str(
        getattr(refused.value, "fix", "")
    )
    # `qda-agent run` prints it: the flow wraps the start's refusal, and the CLI reads the fix off its cause
    from qda_agent.cli import start_fix
    from qda_agent.flow import StartRefused

    try:
        raise StartRefused(refused.value) from refused.value
    except StartRefused as wrapped:
        assert start_fix(wrapped) == refused.value.fix


def test_the_start_says_a_qda_program_the_run_cannot_find(tmp_path):
    """`toolset.qda: [no-such-qda-xyz]` is the start's own refusal, before it creates anything.

    The program is resolved by `judge.startup.qda_resolution` (main's), which `static_checks` calls before the
    run creates anything: one resolution, one severity.
    """
    from qda_agent.judge import make_host

    path = _search_config(tmp_path)
    overrides = ["toolset.qda=[no-such-qda-xyz]"]
    cfg = RunConfig.load(path, overrides)
    with pytest.raises(ValueError) as refused:
        make_host(cfg.run_dir, cfg.sched_config(datetime.now(UTC)), cfg.judge_config(), autotick=False)
    assert "no-such-qda-xyz" in " ".join(str(refused.value).split())


def test_the_start_refuses_a_run_dir_that_already_holds_something(tmp_path):
    """A run dir that is not empty is the start's own refusal, so the run exits 2 in one line, not later."""
    from qda_agent.cli import start_refusal

    path = _search_config(tmp_path)
    cfg = RunConfig.load(path)
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "FINAL.json").write_text("{}\n")
    assert start_refusal(cfg) == f"{cfg.run_dir} is not empty; give a new run_dir"


def test_a_run_dir_that_is_a_file_is_one_line(tmp_path, capsys):
    """A `run_dir` naming a file must not make `run` trace back with `NotADirectoryError` (`iterdir`)."""
    from qda_agent import cli
    from qda_agent.cli import start_refusal

    path = _search_config(tmp_path)
    cfg = RunConfig.load(path)
    cfg.run_dir.write_text("not a directory\n")
    assert start_refusal(cfg) == f"run_dir {cfg.run_dir} is not a directory; give a directory"
    assert cli.execute(cfg, worker=object()) == 2
    err = capsys.readouterr().err
    assert f"run_dir {cfg.run_dir} is not a directory" in err and "Traceback" not in err


def test_the_start_refuses_a_shared_pid_turn(tmp_path, monkeypatch):
    """The wall's dedicated-node rule is a start refusal, not an uncaught `WallRefused` from `qda-agent run`."""
    # its subject is a later refusal: the run dir rule (`isolation.run_dir_in_tmp`) holds wherever tmp_path is
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: None)
    from qda_agent.cli import start_refusal

    shared = "other process(es) of this user on this node: 4242"
    monkeypatch.setattr("qda_agent.isolation.wall.why_not_dedicated", lambda: shared)
    path = _search_config(tmp_path)
    overrides = ("sandbox={name: bwrap, pid: shared}", "isolation.level=sandbox")
    why = start_refusal(RunConfig.load(path, list(overrides)))
    assert why is not None and why.startswith("pid: shared gives the agent's process") and shared in why


def test_doctor_refuses_a_sign_in_that_cannot_last_the_run(tmp_path, monkeypatch):
    """The audit's `agent sign-in` row - `opt` in doctor, a refusal in the run's own check."""
    import json

    from qda_agent import isolation
    from qda_agent.doctor import search
    from qda_quant.doctor import Context, report

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "FAKE_SECRET", "expiresAt": 1}})
    )
    monkeypatch.setenv("HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # a named account dir is where the default-dir refusal does not apply, and the run's config dir holds the
    # link to it only once the wall is up: doctor reads the named dir's own file, the same credential
    named = tmp_path / "account"
    named.mkdir()
    (named / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "FAKE_SECRET", "expiresAt": 1}})
    )
    path = _search_config(tmp_path)
    overrides = ["sandbox=bwrap", "isolation.level=sandbox", "agent.backend={name: claude}"]
    out = report(search(Context(search=path, overrides=tuple(overrides))))
    cfg = RunConfig.load(path, overrides)
    # the run's sentence names the deadline each call computes from `now`, so the line is compared up to it
    with pytest.raises(isolation.SignInRefused) as refused:
        isolation.check_sign_in(cfg, datetime.now(UTC) + timedelta(seconds=cfg.wall_s))
    head = " ".join(str(refused.value).split()).split(" before the run")[0]
    assert f"MISS agent sign-in: {head}" in out and "claude setup-token" in out
    assert "CLAUDE_CODE_OAUTH_TOKEN" in out and "refresh loop" not in out
    assert "FAKE_SECRET" not in out
    # A run with no wall reads the copy the run's own config dir links to (`<run_dir>/cli/<name>`, made at the
    # start): the named dir's file is the same credential, so doctor can answer before the run has made it.
    named_overrides = [
        "sandbox=none",
        "isolation.level=sandbox",
        f"agent.backend={{name: claude, sign_in_dir: {named}}}",
    ]
    named_out = report(search(Context(search=path, overrides=tuple(named_overrides))))
    named_cfg = RunConfig.load(path, named_overrides)
    with pytest.raises(isolation.SignInRefused) as refused_named:
        isolation.check_sign_in(named_cfg, datetime.now(UTC) + timedelta(seconds=named_cfg.wall_s))
    named_head = " ".join(str(refused_named.value).split()).split(" before the run")[0]
    assert named_head.startswith(f"agent.backend.sign_in_dir {named}: the sign-in expires at")
    assert f"MISS agent sign-in: {named_head}" in named_out


def test_doctor_says_agents_that_name_different_sign_in_dirs(tmp_path):
    """This configuration aborted the whole `search` check ("the check raised SignInRefused").

    `sign_in_report` refuses it as well as `check_sign_in` (two agents of one CLI naming different dirs), so
    both are asked inside the one handler: doctor reports the sentence the run refuses with.
    """
    from qda_agent import isolation
    from qda_agent.doctor import search
    from qda_quant.doctor import Context, report

    named = tmp_path / "acct"
    named.mkdir()
    path = _search_config(tmp_path)
    overrides = [
        "sandbox=none",
        "isolation.level=none",
        f"agent.backend={{name: claude, sign_in_dir: {named}}}",
        "agent.fallbacks=[{name: claude}]",
    ]
    out = report(search(Context(search=path, overrides=tuple(overrides))))
    cfg = RunConfig.load(path, overrides)
    with pytest.raises(isolation.SignInRefused) as refused:
        isolation.sign_in_dirs(cfg)  # the run's own refusal for this configuration
    assert f"MISS agent sign-in: {' '.join(str(refused.value).split())}" in out
    assert "raised" not in out


def test_run_refuses_only_where_neither_the_wall_nor_the_native_fence_holds(tmp_path, monkeypatch, capsys):
    """A `fence` run on such a machine is refused before anything is made, naming the wall."""
    # its subject is a later refusal: the run dir rule (`isolation.run_dir_in_tmp`) holds wherever tmp_path is
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: None)
    import hmz.coganchor.fence as hmz_fence

    monkeypatch.setattr(hmz_fence, "enforceable", lambda *, net: False)
    run = tmp_path / "run"
    argv = [
        "run",
        str(FIXTURES / "scripted-search.yaml"),
        "--set",
        f"run_dir={run}",
        "--set",
        "sandbox=bwrap",
        "--set",
        "isolation.level=strict",
        "--set",
        "agent.backend={name: claude}",
    ]
    assert cli.main(argv) == 3  # this host holds neither: the same command works on one that does
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "the bwrap wall cannot start" in err and not run.exists()


def test_qda_doctor_finds_the_search_plugin_and_runs_it(tmp_path, capsys):
    """`qda doctor --search` is the one entry: qda-agent adds its checks to `qda.checks` as an entry point."""
    from qda_quant import cli as qda_cli

    qda_cli.main(
        ["doctor", "--search", str(FIXTURES / "scripted-search.yaml"), "--set", f"run_dir={tmp_path / 'run'}"]
    )
    out = capsys.readouterr().out
    assert "OK   run config" in out
    # the cache and temp disks are the `disk room` plugin's lines: --search adds no second pass over them
    rooms = [line for line in out.splitlines() if " disk room: " in line and not line.startswith("opt")]
    assert rooms and sum("TMPDIR" in line for line in rooms) == 1


def test_doctor_notes_empty_input_components(tmp_path):
    """An empty `serve` or `data` is a fact for ``--report`` (the user may want them off), not a check."""
    from qda_agent.doctor import check_components, search
    from qda_agent.toolsets import Qda
    from qda_quant.doctor import Context, report

    comps = Qda(components=["kl", {"name": "serve"}, {"name": "data", "sets": {}}]).components
    lines = [c.line() for c in check_components(comps)]
    assert lines == [
        "serve: not configured (serve.python or judge.serve_python), the agent's jobs get no vLLM",
        "data: no datasets (sets), DATA.md is not written",
    ]
    full = Qda(
        components=[
            {"name": "serve", "python": sys.executable},
            {"name": "data", "sets": {"c": str(tmp_path)}},
        ]
    ).components
    assert check_components(full) == []
    asked = Context(search=FIXTURES / "scripted-search.yaml", overrides=(f"run_dir={tmp_path / 'run'}",))
    out = report(search(asked))
    assert "serve: not configured" not in out and "data: no datasets" not in out
    facts = report(search(Context(asked.search, overrides=asked.overrides, report=True)))
    assert "serve: not configured" in facts and "data: no datasets" in facts


def test_fallbacks_are_scoped_to_the_run(tmp_path, monkeypatch):
    import hashlib

    from hmz.coganchor import fallbacks

    monkeypatch.setenv("HUMANIZE_HOME", str(tmp_path / "hmz-home"))
    (tmp_path / "hmz-home").mkdir()
    fallbacks.points(
        "codex/gpt-5.6-sol", ["claude/claude-haiku-5-5"]
    )  # somebody's own rule, kept in the file
    settings = tmp_path / "hmz-home" / "settings.yaml"
    before = hashlib.sha256(settings.read_bytes()).hexdigest()
    cfg = RunConfig.load(
        FIXTURES / "scripted-flow.yaml",
        {
            "run_dir": tmp_path / "run",
            "agent": agent("claude/claude-opus-5-5:high", "codex/gpt-5.6-sol:high", retries=2),
        },
    )
    with backends.run_fallbacks(cfg):
        assert fallbacks.chain("claude/claude-opus-5-5") == ["claude/claude-opus-5-5", "codex/gpt-5.6-sol"]
        assert fallbacks.tried("claude/claude-opus-5-5").tries == 2
        assert fallbacks.chain("codex/gpt-5.6-sol") == ["codex/gpt-5.6-sol", "claude/claude-haiku-5-5"]
        assert hashlib.sha256(settings.read_bytes()).hexdigest() == before  # nothing written
    assert fallbacks.chain("claude/claude-opus-5-5") == ["claude/claude-opus-5-5"]  # gone with the run
    assert hashlib.sha256(settings.read_bytes()).hexdigest() == before
    with pytest.raises(KeyError), backends.run_fallbacks(cfg):
        raise KeyError("a run that fails")
    assert fallbacks.chain("claude/claude-opus-5-5") == ["claude/claude-opus-5-5"]


def test_run_refuses_a_non_empty_run_dir(tmp_path, capsys):
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "precious.txt").write_text("not a run's")
    assert cli.main(["run", str(FIXTURES / "scripted-flow.yaml"), "--set", f"run_dir={mine}"]) == 2
    err = capsys.readouterr().err
    assert (mine / "precious.txt").read_text() == "not a run's" and "not empty" in err
    # the fix works for `search DIR` too: it names no command the user did not type
    assert "fix:   rerun the same command with --set run_dir=NEW_DIR" in err


def test_resolved_config_repeats_the_run(tmp_path):
    """`run` writes resolved.yaml; loading it (as `qda-agent run OUT/resolved.yaml` does) gives the same config."""
    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-search.yaml", run, 3) == 0
    resolved = yaml.safe_load((run / "resolved.yaml").read_text())
    prov = resolved["provenance"]
    assert set(prov["plugins"]) >= {
        "agent.backend",
        "sandbox",
        "scheduler",
        "launcher",
        "toolset",
        "contamination",
    }
    assert prov["plugins"]["scheduler"]["package"] in ("qda-agent", "qda_agent")
    assert prov["pid"] is None  # the level reported it (none probes nothing); the key and place are base's
    again = tmp_path / "again"
    assert scripted(run / "resolved.yaml", again, 3) == 0
    first = RunConfig.load(run / "resolved.yaml").model_dump(mode="json")
    second = RunConfig.load(again / "resolved.yaml").model_dump(mode="json")
    assert {**first, "run_dir": None} == {**second, "run_dir": None}


def test_relative_paths_are_relative_to_the_config_file(tmp_path, monkeypatch):
    """Every path setting is made absolute against the config's dir, whatever the cwd; resolved.yaml keeps them."""
    cfgdir, elsewhere = tmp_path / "cfgdir", tmp_path / "elsewhere"
    cfgdir.mkdir()
    elsewhere.mkdir()
    (cfgdir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "format": "w4a16",
                "budget_gpu_hours": 1,
                "cards": [0],
                "run_dir": "runs/x",
                "baselines": ["base"],
                "sandbox": {"name": "bwrap", "hide": ["secret-tree"]},
                "scheduler": {"name": "qsched", "readable": ["model"]},
                "toolset": {"name": "qda", "python": "venvs/quant/bin/python"},
                "contamination": {"name": "ngram", "guard": "guard.txt"},
                "judge": {
                    "serve_python": "venvs/vllm/bin/python",
                    "kl_ref": "ref",
                    "kl_holdout_ref": "holdout",
                    "evaluator": "ev",
                },
            }
        )
    )
    venv_bin = cfgdir / "venvs" / "quant" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)  # as in every venv: a link to the base interpreter
    (venv_bin.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
    for name in ("base", "model", "ref", "holdout", "ev"):  # the other settings name inputs that must exist
        (cfgdir / name).mkdir()
    (cfgdir / "guard.txt").write_text("")
    vllm_bin = cfgdir / "venvs" / "vllm" / "bin"
    vllm_bin.mkdir(parents=True)
    (vllm_bin / "python").touch()
    (vllm_bin.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
    monkeypatch.chdir(elsewhere)
    cfg = RunConfig.load(cfgdir / "run.yaml")
    assert cfg.toolset.python == venv_bin / "python"  # the link, not its target: the venv is kept
    assert cfg.toolset.job_env()["PATH"].startswith(f"{venv_bin}:")
    assert cfg.toolset.qda_cmd()[0] == str(venv_bin / "python")
    absolute = RunConfig.load(cfgdir / "run.yaml", [f"toolset.python={venv_bin / 'python'}"])
    assert absolute.toolset.python == venv_bin / "python"
    assert cfg.judge.serve_python == cfgdir / "venvs/vllm/bin/python"
    assert cfg.sandbox.hide == [cfgdir / "secret-tree"]
    again = RunConfig.load(_write(tmp_path / "resolved.yaml", cfg.resolved()))
    assert again.resolved() == cfg.resolved()  # the record repeats itself, derived values included
    paths = [
        cfg.run_dir,
        *cfg.baselines,
        *cfg.sandbox.hide,
        *cfg.scheduler.readable,
        cfg.toolset.python,
        cfg.contamination.guard,
        cfg.judge.serve_python,
        cfg.judge.kl_ref,
        cfg.judge.kl_holdout_ref,
        cfg.judge.evaluator,
    ]
    assert all(Path(p).is_absolute() for p in paths), paths


def test_every_llamacpp_path_is_relative_to_the_config_file(tmp_path, monkeypatch):
    """The imatrix resolves beside the config file like the other `judge.llamacpp` paths."""
    from qda_agent import configfile

    cfgdir, elsewhere = tmp_path / "cfgdir", tmp_path / "elsewhere"
    cfgdir.mkdir()
    elsewhere.mkdir()
    names = {
        "bridge": "llama/build/probe.so",
        "llama_cpp": "llama",
        "server": "llama/build/bin/llama-server",
        "template": "gguf/bf16.gguf",
        "imatrix": "gguf/imatrix.gguf",
    }
    _write(cfgdir / "run.yaml", {"judge": {"llamacpp": names}})
    monkeypatch.chdir(elsewhere)
    raw = configfile.read(cfgdir / "run.yaml")
    assert raw["judge"]["llamacpp"] == {k: str((cfgdir / v).resolve()) for k, v in names.items()}


def test_a_set_path_is_relative_to_the_current_directory(tmp_path, monkeypatch):
    """`--set run_dir=rel` lands at ./rel; a relative path written in the file still resolves beside the file."""
    from qda_agent import cand0, configfile

    cfgdir, elsewhere = tmp_path / "cfgdir", tmp_path / "elsewhere"
    cfgdir.mkdir()
    elsewhere.mkdir()
    (elsewhere / "Qwen" / "Qwen3-8B").mkdir(parents=True)  # a hub name in the file, also a directory here
    path = _write(
        cfgdir / "run.yaml",
        {"model": "Qwen/Qwen3-8B", "run_dir": "runs/x", "judge": {"llamacpp": {"imatrix": "gguf/i.gguf"}}},
    )
    monkeypatch.chdir(elsewhere)
    raw = configfile.read(path)
    assert raw["run_dir"] == str(cfgdir / "runs" / "x")
    assert raw["judge"]["llamacpp"]["imatrix"] == str(cfgdir / "gguf" / "i.gguf")
    assert raw["model"] == "Qwen/Qwen3-8B"
    raw = configfile.read(path, ["run_dir=rel", "judge.llamacpp.imatrix=gguf/j.gguf"])
    assert raw["run_dir"] == str(elsewhere / "rel")
    assert raw["judge"]["llamacpp"]["imatrix"] == str(elsewhere / "gguf" / "j.gguf")
    assert raw["model"] == "Qwen/Qwen3-8B"  # not set: still the hub name, not ./Qwen/Qwen3-8B
    assert configfile.read(path, {"run_dir": "rel"})["run_dir"] == str(elsewhere / "rel")
    assert configfile.read(path, ["model=Qwen/Qwen3-8B"])["model"] == str(elsewhere / "Qwen" / "Qwen3-8B")
    # Each command that takes `--set` reads the config through `configfile.read`: run, gate, search, doctor, cand0.
    fixture = FIXTURES / "scripted-search.yaml"
    assert RunConfig.load(fixture, ["run_dir=rel"]).run_dir == elsewhere / "rel"
    assert RunConfig.load(fixture).run_dir == (FIXTURES / "../../runs/scripted-search").resolve()
    assert cand0.Inputs.read(fixture, ["run_dir=rel"]).raw["run_dir"] == str(elsewhere / "rel")


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_shipped_configs_load_and_hold_no_machine_path(tmp_path):
    """configs/search.yaml, filled in, loads, and no shipped config names a machine path or a site tool."""
    cfg = RunConfig.load(filled(CONFIGS / "search.yaml", tmp_path))
    assert cfg.agent.backend.name == "claude" and cfg.scheduler.name == "qsched"
    machine = re.compile(r"/home/|/data\d*/|/Users/|/mnt/|/scratch/|gpu-run|gpu_run")
    for path in [*CONFIGS.rglob("*.yaml"), *EXAMPLES.rglob("*.yaml")]:
        assert not machine.search(path.read_text()), path


def test_every_shipped_config_ends_on_the_agents_hand_in_list(tmp_path):
    """A shipped run ends on the agent's own hand-in list, not on a KL-ranked build it never handed in."""
    for i, path in enumerate(CONFIGS.rglob("*.yaml")):
        cfg = RunConfig.load(filled(path, tmp_path / str(i)))
        assert [rule.name for rule in cfg.judge.final_rules] == ["handin"], path
    for name in ("scripted-search.yaml", "scripted-flow.yaml"):  # the launch fixtures
        cfg = RunConfig.load(FIXTURES / name)
        assert [rule.name for rule in cfg.judge.final_rules] == ["handin"], name


def test_run_refuses_placeholders_by_key_and_makes_nothing(capsys):
    """A shipped config as it ships: exit 2 naming every key to fill, no traceback, no run dir made."""
    path = CONFIGS / "search.yaml"
    before = sorted(path.parent.iterdir())
    assert cli.main(["run", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"error: {path}: fill in model, run_dir, calib, baselines[0], ")
    assert (
        "scheduler.readable[0], toolset.python, toolset.components[9].sets.corpus, contamination.guard" in err
        and "judge.format_reference" in err
        and len(err.splitlines()) == 1
    )
    assert sorted(path.parent.iterdir()) == before
    from qda_agent.doctor import search
    from qda_quant.doctor import Context, report

    out = report(search(Context(search=path)))
    assert "MISS run config" in out and "fill in model" in out


def test_a_bad_config_is_one_line_and_exit_2(tmp_path, capsys):
    config = _write(tmp_path / "run.yaml", {"model": "m", "format": "w4a16", "cards": [0], "run_dir": "r"})
    assert cli.main(["run", str(config)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "budget_gpu_hours: Field required" in err and "Traceback" not in err


def test_test_backend_needs_no_agent_cli_and_no_gpu():
    """The test-only scripted backend has no CLI and uses cards as queue slots."""
    cfg = RunConfig.load(FIXTURES / "scripted-search.yaml")
    assert cfg.agent.backend.cli is None and cfg.agent.spec == "script/test_scripted"
    assert isinstance(cfg.agent.backend.driver(cfg), fakes.FakeAgentDriver)
    assert cfg.scheduler.gpus is False


def test_set_overrides_keys_only(tmp_path):
    cfg = RunConfig.load(
        FIXTURES / "scripted-search.yaml",
        ["scheduler.tick_s=0.5", "sandbox=bwrap", "sandbox.pid=shared"],
    )
    assert cfg.scheduler.tick_s == 0.5 and cfg.sandbox.name == "bwrap" and cfg.sandbox.pid == "shared"
    with pytest.raises(ValueError, match="KEY.PATH=VALUE"):
        RunConfig.load(FIXTURES / "scripted-search.yaml", ["no-equals-sign"])
    with pytest.raises(ValueError, match="teacher"):
        RunConfig.load(FIXTURES / "scripted-search.yaml", ["scheduler.teacher=1"])


def test_a_toolset_without_qda_gets_three_commands(tmp_path):
    """A toolset with fewer components: no `qda` in bin/ or TASK.md, the host refuses it, the rest work."""
    from qda_agent.commands import request

    run = tmp_path / "run"
    asked: list[dict] = []

    def script(cfg: RunConfig) -> ScriptedWorker:
        worker = shell_worker(cfg)
        ask = lambda: asked.append(request(str(cfg.host_socket), "qda", ["--help"], ".")) or "asked qda"  # noqa: E731
        worker.steps.insert(0, ask)
        return worker

    over = ["toolset.components=[kl, qsched, handin]"]
    assert scripted(FIXTURES / "scripted-search.yaml", run, 8, script=script, overrides=over) == 0
    work = run / "work"
    assert sorted(p.name for p in (work / "bin").iterdir() if p.is_symlink()) == ["handin", "kl", "qsched"]
    task = (work / "TASK.md").read_text()
    assert "Three commands in `bin/`" in task and "`qda build CONFIG.yaml" not in task
    assert "quant-methods" not in task and not (work / ".claude").exists()  # no wiki component: no word of it
    assert asked[0]["exit"] != 0 and "no such command" in asked[0]["stderr"]
    assert json.loads((run / "result.json").read_text())["rule"] == "handin"  # kl, qsched, handin worked


def test_a_set_value_that_is_not_yaml_is_one_line_and_exit_2(tmp_path, capsys):
    assert cli.main(["run", str(FIXTURES / "scripted-search.yaml"), "--set", "cards=["]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: --set 'cards=[': the value is not YAML") and len(err.splitlines()) == 1


@pytest.mark.parametrize("made", [False, True])
def test_search_without_a_prepared_config_names_prepare(tmp_path, capsys, made):
    """A directory prepare never made, or one without its search.yaml, refuses in two lines naming prepare."""
    out = tmp_path / "q06b"
    if made:
        out.mkdir()
    assert cli.main(["search", str(out)]) == 2
    err = capsys.readouterr().err.splitlines()
    config = out / "search.yaml" if made else out
    assert err == [
        f"error: {config}: no such file",
        f"fix:   run `qda-agent prepare --model M --out {out}` first; it makes {out}/search.yaml",
    ], err


def test_help_lists_exactly_the_public_commands(capsys):
    """Each program's `--help` names its public set and no other command; a hidden one still runs by name."""
    import re

    from qda_agent import cli as agent
    from qda_kl import cli as kl
    from qda_quant import cli as qda

    for module, public in (
        (
            qda,
            (
                "build",
                "plan",
                "doctor",
                "inspect",
                "eval",
                "export",
                "decode-gguf",
                "fisher",
                "calib",
                "rotate-draft",
                "sensitivity",
                "traces",
            ),
        ),
        (kl, ("ref", "score", "kl", "gencheck")),
        (agent, ("prepare", "search", "status", "run", "gate", "cand0")),
    ):
        assert module.PUBLIC == public
        with pytest.raises(SystemExit):
            module.main(["--help"])
        out = capsys.readouterr().out
        choices = re.search(r"\{([^}]*)\}", out).group(1).split(",")
        assert choices == list(public)
        rows = re.findall(r"^    ([a-z][a-z0-9-]*)\s", out, re.M)
        assert sorted(rows) == sorted(public)
    assert agent.main(["status", "/nonexistent-run-dir"]) == 2
    with pytest.raises(SystemExit):  # hidden, not removed: argparse parses it and asks for its argument
        agent.main(["evaluate"])
    assert "run_dir" in capsys.readouterr().err
