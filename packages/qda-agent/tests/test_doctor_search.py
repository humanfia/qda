"""What `qda doctor --search` reports, and what it deliberately leaves to the search itself.

Doctor reads two things out of a run config -- the isolation level and the run directory's disk -- and the
agent's sign-in with them. The conditions it no longer repeats are refused by the search when it starts, and
the launch's trial run (`test_trial_run.py`) does the rest on this machine.
"""

from pathlib import Path

import pytest
from qda_quant.doctor import Context


def _shipped(tmp_path):
    """The shipped search config, filled under `tmp_path`, with a baseline that is a checkpoint."""
    from tests_support import filled

    path = filled(Path(__file__).parents[1] / "configs" / "search.yaml", tmp_path / "cfg")
    (tmp_path / "cfg" / "path" / "to" / "baseline-ckpt" / "config.json").write_text("{}")
    return path


def test_the_search_lines_are_the_isolation_level_the_disk_and_the_sign_in(tmp_path):
    """One line per thing doctor can still decide; nothing that the search refuses on by itself."""
    from qda_agent import doctor as d

    ctx = Context(search=_shipped(tmp_path), overrides=(f"run_dir={tmp_path / 'run'}", "cards=[0]"))
    names = [line.name for line in d.search(ctx)]
    assert names[0] == "run config"
    # the lines doctor can still decide; a config that does not load stops after the first one
    assert set(names) <= {
        "run config",
        "config",
        "run dir",
        "sandbox",
        "disk cap",
        "agent CLI",
        "agent sign-in",
    }
    gone = {
        "gpu sandbox",
        "cards",
        "card order",
        "judge fast kernels",
        "build job compiler",
        "judge qda",
        "judge FlashInfer cache",
        "evaluator",
        "anchor",
        "socket",
        "judge ipc",
        "start configs",
        "inputs",
    }
    assert not gone & set(names), "a search refuses on these itself; doctor no longer repeats them"


def test_an_agent_cli_not_on_path_is_a_miss_before_the_run_starts(tmp_path, monkeypatch):
    """The run starts the agent only after candidate 0's trial run, minutes in; doctor looks the CLI up on this
    shell's PATH first, the primary backend's and each fallback's."""
    from types import SimpleNamespace

    from qda_agent import doctor as d
    from qda_agent.agents import AgentConfig

    monkeypatch.setenv("PATH", str(tmp_path))
    cfg = SimpleNamespace(agent=AgentConfig.model_validate({"backend": {"name": "claude"}}))
    line = d.check_agent_cli(cfg)
    assert (
        line.missed and line.detail == "claude is not on PATH: the search would stop when it starts the agent"
    )
    assert line.fix == "install claude and put it on PATH, or change agent.backend"

    (tmp_path / "claude").write_text("#!/bin/sh\n")
    (tmp_path / "claude").chmod(0o755)
    line = d.check_agent_cli(cfg)
    assert line.ok and line.detail == str(tmp_path / "claude")
    both = {"backend": {"name": "claude"}, "fallbacks": [{"name": "codex"}]}
    line = d.check_agent_cli(SimpleNamespace(agent=AgentConfig.model_validate(both)))
    assert line.missed and line.detail.startswith("codex is not on PATH")


def test_the_run_refuses_an_agent_cli_not_on_path_before_its_clock(tmp_path, monkeypatch):
    """The run's own pre-clock check (`flow.check_agent_turn`) says what doctor's line says: no CLI on PATH is
    refused, unless a fallback takes the turn, which is recorded instead."""
    from types import SimpleNamespace

    from qda_agent.agents import AgentConfig
    from qda_agent.flow import AgentTurnRefused, check_agent_turn

    monkeypatch.setenv("PATH", str(tmp_path))
    alone = SimpleNamespace(agent=AgentConfig.model_validate({"backend": {"name": "claude"}}))
    with pytest.raises(AgentTurnRefused, match=r"^claude: claude is not on PATH \(fix: install claude"):
        check_agent_turn(alone)
    both = {"backend": {"name": "claude"}, "fallbacks": [{"name": "codex"}]}
    said = check_agent_turn(SimpleNamespace(agent=AgentConfig.model_validate(both)))
    assert said == (
        "claude: claude is not on PATH (fix: install claude and put it on PATH, or change agent.backend) "
        "(codex takes the turn)"
    )
    (tmp_path / "claude").write_text("#!/bin/sh\n")
    (tmp_path / "claude").chmod(0o755)
    assert check_agent_turn(alone) is None


def test_a_config_that_does_not_load_is_the_input(tmp_path):
    from qda_agent import doctor as d
    from qda_quant.exit import USAGE

    (tmp_path / "broken.yaml").write_text("model: [unclosed\n")
    (line,) = d.search(Context(search=tmp_path / "broken.yaml"))
    assert line.missed and line.code == USAGE


def test_bubblewrap_is_a_miss_only_where_the_run_requires_it(tmp_path, monkeypatch):
    """`isolation.level: strict` is the level that needs bwrap; the others say it is missing, and go on."""
    from qda_agent import doctor as d
    from qda_agent.sched.config import SchedConfig

    monkeypatch.setattr(d.shutil, "which", lambda name, **kw: None)
    cfg = SchedConfig(cards=[0], run_dir=tmp_path / "run")
    strict = d.check_bwrap(cfg, level="strict")
    assert strict.missed and "bwrap: not found" in strict.detail and strict.fix

    sandboxed = d.check_bwrap(cfg, level="sandbox")
    assert sandboxed.skipped and not sandboxed.missed
    assert "bwrap: not found" in sandboxed.detail and "sandbox" in sandboxed.detail

    # `isolation.level: none` runs plain jobs: there is no job sandbox to look for at all
    assert d.check_bwrap(cfg, level="none").skipped


def test_the_build_jobs_of_a_search_carry_the_compiler_doctor_probed(monkeypatch):
    """A job's PATH is the run's, not this shell's: the compiler is passed by absolute path, or every
    build of the search refuses on Triton's compiler check after doctor said the machine was fine."""
    from qda_quant import checks

    monkeypatch.setenv("CC", "/opt/gcc-13/bin/gcc")
    monkeypatch.setenv("CXX", "/opt/gcc-13/bin/g++")
    assert checks.compiler_env() == {"CC": "/opt/gcc-13/bin/gcc", "CXX": "/opt/gcc-13/bin/g++"}

    monkeypatch.delenv("CC")
    monkeypatch.delenv("CXX")
    found = checks.compiler_env()
    for name, program in found.items():
        assert Path(program).is_absolute(), f"{name} must name the program, not a PATH lookup"
