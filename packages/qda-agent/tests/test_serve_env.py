"""The engine environment the judge's own vLLM engines start with reaches the agent's vLLM jobs too.

The agent writes its own serving scripts; they start from the job environment, not from the judge's engine
settings. One definition (`qda_kl.serve.NO_JIT_ENV`, `qda_quant.output.vllm_env.cache_env`), one job
environment (the queue's `_command`), both levels.
"""

import time
from pathlib import Path

from qda_agent.components import Serve
from qda_agent.isolation.api import Launch
from qda_agent.sched import FakeClock, Kind, LocalBackend, SchedConfig, Scheduler
from qda_kl.serve import NO_JIT_ENV
from qda_quant.output import vllm_env


def test_the_serve_component_adds_the_engine_env(monkeypatch):
    """`Serve.env` reads the judge's one definition; it does not state its own copy."""
    for name in vllm_env.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    caches = vllm_env.cache_env()
    engine = {**NO_JIT_ENV, "VLLM_DISABLED_KERNELS": "MacheteLinearKernel", **caches}  # Machete off here too
    assert Serve().env(None) == engine
    with_python = Serve(python=Path("/serving/bin/python")).env(None)
    assert with_python == {**engine, "QDA_VLLM_PYTHON": "/serving/bin/python"}


def test_the_engine_caches_are_node_local_unless_the_user_names_them(tmp_path, monkeypatch):
    """The `$HOME` defaults (often a network disk) are never the jobs' or the engines' compile caches."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in vllm_env.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    env = Serve().env(None)
    for name, leaf in vllm_env.CACHE_DIRS.items():
        assert env[name] == str(vllm_env.cache_root() / leaf)
        assert not Path(env[name]).is_relative_to(tmp_path / "home")
    # the judge's own engines resolve the same caches (the in-process resolver)
    assert {name: vllm_env.cache_env()[name] for name in vllm_env.CACHE_ENV} == {
        name: env[name] for name in vllm_env.CACHE_ENV
    }
    # a directory the user named is kept, for both
    named = tmp_path / "named"
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(named))
    assert Serve().env(None)["FLASHINFER_WORKSPACE_BASE"] == str(named)
    assert vllm_env.cache_env()["FLASHINFER_WORKSPACE_BASE"] == str(named)


class Capture:
    """An isolation level that records each job's environment and starts nothing."""

    level = "none"

    def __init__(self):
        self.envs: dict[str, dict[str, str]] = {}

    def job(self, spec) -> Launch:
        self.envs[spec.argv[-1]] = dict(spec.env)
        return Launch(tuple(spec.argv), dict(spec.env), ())


def test_every_agent_job_gets_the_toolsets_environment(tmp_path):
    """Sandboxed or plain, an agent job starts with the toolset's env; the judge's own jobs do not."""
    run, work = tmp_path / "run", tmp_path / "work"
    work.mkdir(parents=True)
    iso = Capture()
    agent_env = {**NO_JIT_ENV, "PATH": "/job/bin:/usr/bin:/bin"}
    s = Scheduler(
        run,
        SchedConfig(cards=[0], gpus=False, agent_job_env=dict(agent_env)),
        LocalBackend(1),
        FakeClock(time.time()),
        isolation=iso,
    )
    s.submit(["/bin/true", "agent-job"], str(work), name="a", expected_s=60, root=work, kind=Kind.build)
    s.submit(
        ["/bin/true", "judge-job"],
        str(work),
        name="j",
        expected_s=60,
        root=work,
        kind=Kind.build,
        origin="judge",
    )
    for _ in range(50):
        s.tick()
        if "agent-job" in iso.envs:
            break
        time.sleep(0.02)
    assert iso.envs["agent-job"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert iso.envs["agent-job"]["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
    assert iso.envs["agent-job"]["PATH"] == agent_env["PATH"]
    # the judge's own jobs run on the host with their own (trusted) env: the toolset's is not mixed in
    judge = s._command(s.jobs["j"], [])
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in (judge.env or {})
