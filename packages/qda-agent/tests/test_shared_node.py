"""`isolation.shared_node`: an operator's acceptance lifts the wall's dedication refusal and nothing else.

`strict`/`sandbox` refuse a `pid: shared` turn on a node this user shares. The acceptance is the product way
to run anyway on a watched node: it is a run config value (never an environment variable), it is recorded in
`resolved.yaml`, and the wall writes who accepted it, why, and what else is on the node to the launch log.
Everything else the walls hold (file wall, netns, host proxy, user namespace) is untouched, and a config
without the acceptance refuses exactly as before.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from qda_agent import isolation
from qda_agent.config import RunConfig
from qda_agent.isolation import allocation, wall
from qda_agent.isolation import bwrap as bwrap_mod
from qda_agent.isolation.api import make
from qda_agent.isolation.levels import strict
from qda_agent.isolation.levels.legacy import LegacyIsolation
from testkit import scripted

FIXTURES = Path(__file__).parent / "fixtures"
WHY = "other process(es) of this user on this node: 4242 (sleep)"
ACCEPT = {"accepted_by": "the owner", "reason": "a watched run on a node that is shared with other work"}
REFUSAL = (
    "pid: shared gives the agent's process the host's pid namespace (this node refuses a fresh procfs), "
    f"so the run needs the node to itself, and it does not have it: {WHY}. Ask for a dedicated allocation "
    "(an exclusive node or container for this run)."
)


def config(tmp_path: Path, *, shared_node: dict | None = None, level: str = "strict") -> RunConfig:
    """A run config at `level` whose sandbox names bwrap; `shared_node` names the acceptance."""
    isolation_settings: dict = {"level": level}
    if shared_node is not None:
        isolation_settings["shared_node"] = shared_node
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        sandbox="bwrap",
        isolation=isolation_settings,
    )


def pid_resolves(monkeypatch, form: str) -> None:
    """`auto` resolves to `form` for the wall and for doctor (`isolation.pid_form`), whatever this machine probes."""
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": form)
    monkeypatch.setattr(isolation, "pid_form", lambda requested="auto": form)


@pytest.fixture
def busy_node(monkeypatch):
    """A node with other work of this user, and a pid form that resolves `shared`."""
    pid_resolves(monkeypatch, "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: WHY)


@pytest.fixture
def today_passes(monkeypatch):
    """Today's preflight passes: these tests are about the wall's one dedication rule."""
    monkeypatch.setattr(LegacyIsolation, "preflight", lambda self, cfg: None)


def test_without_an_acceptance_the_refusal_is_unchanged_at_both_call_sites(tmp_path, busy_node, today_passes):
    """No acceptance: the same sentence from `shared_pid_refusal` itself, the wall and the strict preflight."""
    assert wall.shared_pid_refusal("shared") == REFUSAL
    cfg = config(tmp_path)
    with pytest.raises(wall.WallRefused) as from_wall, wall.gpu_wall(cfg):
        pass
    with pytest.raises(wall.WallRefused) as from_preflight:
        strict.StrictIsolation.for_config(cfg).preflight(cfg)
    assert str(from_wall.value) == str(from_preflight.value) == REFUSAL


def test_an_acceptance_lifts_only_the_dedication_refusal(tmp_path, busy_node, today_passes, caplog):
    """Named: None instead of the sentence, one warning naming who, why and what else is on the node."""
    cfg = config(tmp_path, shared_node=ACCEPT)
    accepted = cfg.isolation.shared_node
    assert accepted is not None
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="qda_agent.isolation.wall"):
        assert wall.shared_pid_refusal("shared", shared_node=accepted) is None
    (line,) = [r.getMessage() for r in caplog.records]
    assert WHY in line and ACCEPT["accepted_by"] in line and ACCEPT["reason"] in line
    assert "file and network walls unchanged" in line
    with caplog.at_level(logging.WARNING, logger="qda_agent.isolation.wall"):
        strict.StrictIsolation.for_config(cfg).preflight(cfg)  # passes
    with caplog.at_level(logging.WARNING, logger="qda_agent.isolation.wall"):
        with wall.gpu_wall(cfg):
            pass  # the wall goes up: file wall, netns, host proxy, userns, as without an acceptance
    assert any(WHY in r.getMessage() for r in caplog.records)


def test_the_acceptance_is_recorded_in_the_resolved_config(tmp_path, monkeypatch):
    """`resolved.yaml` carries who accepted and why, so the record shows what the run was allowed."""
    monkeypatch.setattr(bwrap_mod, "pid_form", lambda requested="auto": "private")  # no probe in a unit test
    cfg = config(tmp_path, shared_node=ACCEPT)
    resolved = cfg.resolved()
    assert resolved["isolation"]["shared_node"] == ACCEPT
    assert resolved["isolation"]["level"] == "strict"


def test_without_the_key_resolved_yaml_records_nothing_about_a_shared_node(tmp_path, monkeypatch):
    monkeypatch.setattr(bwrap_mod, "pid_form", lambda requested="auto": "private")
    assert "shared_node" not in config(tmp_path).resolved()["isolation"]  # old records unchanged


@pytest.mark.parametrize(
    "bad",
    [
        {"accepted_by": "the owner"},  # no reason
        {"reason": "a watched run"},  # no name
        {"accepted_by": "", "reason": "x"},
        {"accepted_by": "x", "reason": "  "},
        {"accepted_by": "x", "reason": "y", "accepted_at": "2026-11-15"},  # extra key
    ],
)
def test_an_incomplete_or_misspelled_acceptance_is_refused(tmp_path, bad):
    with pytest.raises(ValidationError):
        config(tmp_path, shared_node=bad)


def test_the_environment_is_not_an_input(tmp_path, busy_node, today_passes, monkeypatch):
    """The variable the operator patch used is inert: only the config value can lift the refusal."""
    monkeypatch.setenv("QDA_SHARED_NODE_ACCEPTED", "accepted by the environment")
    assert wall.shared_pid_refusal("shared") == REFUSAL
    cfg = config(tmp_path)
    with pytest.raises(wall.WallRefused):
        strict.StrictIsolation.for_config(cfg).preflight(cfg)


def test_the_warning_names_the_config_not_the_environment(tmp_path, busy_node, monkeypatch, caplog):
    monkeypatch.setenv("QDA_SHARED_NODE_ACCEPTED", "accepted by the environment")
    cfg = config(tmp_path, shared_node=ACCEPT)
    with caplog.at_level(logging.WARNING, logger="qda_agent.isolation.wall"):
        assert wall.shared_pid_refusal("shared", shared_node=cfg.isolation.shared_node) is None
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert ACCEPT["accepted_by"] in text and "accepted by the environment" not in text


def test_a_dedicated_node_is_unchanged_by_naming_an_acceptance(tmp_path, busy_node, monkeypatch):
    """The acceptance is read only after the refusal is reached: a free node logs and decides as before."""
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: None)
    cfg = config(tmp_path, shared_node=ACCEPT)
    assert wall.shared_pid_refusal("shared", shared_node=cfg.isolation.shared_node) is None
    assert wall.shared_pid_refusal("shared") is None  # nothing to lift: same answer with or without it


def test_a_private_pid_form_never_reads_the_acceptance(tmp_path, busy_node, monkeypatch):
    """Only `pid: shared` asks the question; a private pid form is decided before either value is read."""
    pid_resolves(monkeypatch, "private")
    cfg = config(tmp_path, shared_node=ACCEPT)
    assert wall.shared_pid_refusal("auto", shared_node=cfg.isolation.shared_node) is None
    assert wall.shared_pid_refusal("auto") is None


def test_level_none_is_unaffected_by_the_acceptance(tmp_path, busy_node):
    """At `none` nothing is walled and nothing is asked: the key is recorded and changes no decision."""
    cfg = config(tmp_path, shared_node=ACCEPT, level="none")
    iso = make(cfg)
    assert iso.level == "none"
    with isolation.gpu_wall(cfg):
        pass  # no wall, no probe, no refusal
    iso.preflight(cfg)


SLEEPER = """
import os, sys, time

if os.fork():  # the parent returns at once, so the child is reparented away from the test's tree
    os._exit(0)
os.setsid()
if os.fork():  # the intermediate exits: the grandchild is adopted outside the test's process tree
    os._exit(0)
with open(sys.argv[1], "w") as pid_file:
    pid_file.write(str(os.getpid()))
time.sleep(120)
"""


def test_a_none_search_runs_while_the_user_has_other_work_on_the_node(tmp_path, monkeypatch):
    """Item 10, first half: a real sleeper of this user, outside the run's tree, does not block `none`."""
    marker = tmp_path / "sleeper.pid"
    subprocess.run([sys.executable, "-c", SLEEPER, str(marker)], check=True)
    pid = None
    for _ in range(200):
        if marker.is_file() and marker.read_text().strip():
            pid = int(marker.read_text())
            break
        time.sleep(0.05)
    assert pid is not None
    try:
        others = {p.pid for p in allocation._other_processes(Path("/proc"))}
        assert pid in others, "the sleeper is not seen as another process of this user"
        assert allocation.why_not_dedicated() is not None  # the node really is shared while the run starts
        real, asked = allocation.why_not_dedicated, []

        def asked_once() -> str | None:
            asked.append(1)
            return real()

        monkeypatch.setattr(wall, "why_not_dedicated", asked_once)
        run = tmp_path / "run"
        overrides = ["isolation.level=none", "scheduler.job_sandbox=none"]
        assert scripted(FIXTURES / "scripted-search.yaml", run, 25, overrides=overrides) == 0
        final = json.loads((run / "FINAL.json").read_text())
        assert final["rule"] == "handin" and final["choice"] is not None
        assert asked == []  # the dedication check is never asked at `none`
    finally:
        os.kill(pid, signal.SIGTERM)  # the pid this test started, captured above
        for _ in range(200):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)


def search_config(tmp_path: Path, *, accepted: bool) -> Path:
    """The scripted search (a CPU stand-in) at `none`, with or without the acceptance, as a config file."""
    from test_components import _abs_refs

    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "run")
    raw["baselines"] = [str(FIXTURES / "baselines" / "gptq")]
    raw["judge"]["evaluator"] = str(FIXTURES.parents[1] / "examples" / "standin_kl")
    _abs_refs(raw)  # the reference paths, as the fixture's own directory would resolve them
    raw["isolation"] = {"level": "none"}
    if accepted:
        raw["isolation"]["shared_node"] = ACCEPT
    config_file = tmp_path / "search.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    return config_file


def test_the_acceptance_is_recorded_beside_the_result(tmp_path):
    """A reader of the result sees who accepted the node, why, and what else was on it — no launch log needed."""
    config_file = search_config(tmp_path, accepted=True)
    run = tmp_path / "run"
    assert scripted(config_file, run, 25) == 0
    recorded = json.loads((run / "result.json").read_text())["shared_node"]
    assert recorded["accepted_by"] == ACCEPT["accepted_by"]
    assert recorded["reason"] == ACCEPT["reason"]
    assert isinstance(recorded["other_work"], str | type(None))  # the node's own sentence, or none found
    report = (run / "END-REPORT.md").read_text()
    assert ACCEPT["accepted_by"] in report and ACCEPT["reason"] in report
    assert f"- shared node, accepted by {ACCEPT['accepted_by']}: {ACCEPT['reason']}" in report


def test_without_an_acceptance_the_result_files_are_unchanged(tmp_path):
    """No acceptance: neither `result.json` nor `END-REPORT.md` gains a key or a line (today's bytes)."""
    config_file = search_config(tmp_path, accepted=False)
    run = tmp_path / "run"
    assert scripted(config_file, run, 25) == 0
    assert "shared_node" not in json.loads((run / "result.json").read_text())
    assert "shared node" not in (run / "END-REPORT.md").read_text()


def test_the_wall_is_built_the_same_with_and_without_the_acceptance(tmp_path, monkeypatch):
    """Accepted-on-a-busy-node and unaccepted-on-a-free-node build the same bwrap argv (as L1c's proof)."""
    fence = pytest.importorskip("hmz.coganchor.fence")
    from test_isolation import _fence

    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)

    def argv_for(cfg: RunConfig, why: str | None) -> list[str]:
        monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
        monkeypatch.setattr(wall, "why_not_dedicated", lambda: why)
        f = _fence(work, hosts=["api.anthropic.com"])
        with wall.gpu_wall(cfg):
            argv = fence.wrapper(f)
        return json.loads(Path(argv[4]).read_text()) + argv[5:]

    accepted = argv_for(config(tmp_path, shared_node=ACCEPT), WHY)  # set + busy: the wall is built
    dedicated = argv_for(config(tmp_path), None)  # unset + dedicated: the same wall, today's case
    # The proxy's socket directory is made fresh per wall (`qda-net-<random>`): everything else is identical.
    normalise = lambda argv: [re.sub(r"qda-net-[A-Za-z0-9_]+", "qda-net-X", a) for a in argv]  # noqa: E731
    assert normalise(accepted) == normalise(dedicated)


def test_doctor_and_the_start_name_the_shared_node_fix_at_sandbox(tmp_path, busy_node):
    """At `sandbox` too, doctor shows the start's refusal as a MISS, and both fixes name `isolation.shared_node`."""
    from qda_agent import cli, doctor

    cfg = config(tmp_path, level="sandbox")
    line = doctor.check_pid_rule(cfg)
    assert line is not None and line.ok is False
    assert line.detail == REFUSAL and line.fix == isolation.PID_FIX
    assert "isolation.shared_node" in isolation.PID_FIX and "sandbox.pid" not in isolation.PID_FIX
    problem = cli._pid_rule(cfg)
    assert problem is not None and problem.fix == isolation.PID_FIX
    accepted = doctor.check_pid_rule(config(tmp_path, level="sandbox", shared_node=ACCEPT))
    assert accepted is not None and accepted.ok and ACCEPT["accepted_by"] in accepted.detail
    assert doctor.check_pid_rule(config(tmp_path, level="none")) is None


@pytest.mark.parametrize("form", ["shared", "private"])
def test_doctor_names_the_pid_form_the_wall_resolves(tmp_path, busy_node, monkeypatch, form):
    """On a busy node, doctor's row is a MISS exactly when the wall resolves `shared`, and names that form."""
    from qda_agent import doctor

    pid_resolves(monkeypatch, form)
    refused = doctor.check_pid_rule(config(tmp_path, level="sandbox"))
    assert refused is not None and refused.ok is (form == "private")
    line = doctor.check_pid_rule(config(tmp_path, level="sandbox", shared_node=ACCEPT))
    assert line is not None and line.ok and line.detail.startswith(f"pid: {form}")
