"""The launch's trial run: candidate 0 is really built, then really served, before the agent's first turn.

`startup._serve_trial` runs the run's own hand-in check (`judge.check_cmd`) on the tree candidate 0's build
published, as a job of the run. These tests pin the job's result and assert what the launch does with it; the
engine itself is the check command's business, and a real one is exercised on a card, not here.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from qda_agent.judge import startup

if TYPE_CHECKING:
    from qda_agent.judge.host import JudgeHost


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class _Job:
    def __init__(self, name: str, state) -> None:
        self.name, self.state = name, state


class _Host:
    """The fields `_serve_trial` reads, with a scheduler that ends the job as `state` says."""

    def __init__(self, tmp_path: Path, state, *, said: dict | None = None, template=None) -> None:
        self.jdir = tmp_path / "judge"
        self.jdir.mkdir(parents=True, exist_ok=True)
        self.cfg = SimpleNamespace(
            check_cmd=["{python}", "{tools}/serve_check.py", "{ckpt}", "{out}"]
            if template is None
            else template,
            check_cards=1,
            check_expected_s=30.0,
            serve_python=None,
            job_environ=lambda parent, tmp_dir=None: {},
        )
        self.clock = _Clock()
        self.eval_dir = None
        self.temps = SimpleNamespace(make=lambda key, name: tmp_path / "tmp", release=lambda name: None)
        self.submitted: list[list[str]] = []
        self.said = said
        self.state = state
        self.sched = SimpleNamespace(
            cfg=SimpleNamespace(tick_s=1.0),
            submit=self._submit,
            cancel=lambda name: None,
            log=lambda name, lines, origin: "ImportError: libcuda.so.1: cannot open shared object file",
        )

    def _submit(self, argv, cwd, **kwargs):
        self.submitted.append(argv)
        if self.said is not None:
            Path(argv[-1]).write_text(__import__("json").dumps(self.said))
        return _Job(kwargs["name"], self.state)

    def tick(self) -> None:
        return None


def _trial(host: _Host, published: Path) -> None:
    startup._serve_trial(cast("JudgeHost", host), published)


def test_the_trial_run_serves_candidate_0_with_the_hand_in_check(tmp_path):
    """The command the trial run runs is `judge.check_cmd`, on the tree the build published."""
    from qda_agent.sched.job import State

    host = _Host(tmp_path, State.done, said={"ok": True, "reason": ""})
    _trial(host, tmp_path / "published")
    argv = host.submitted[0]
    assert argv[1].endswith("serve_check.py"), "the hand-in check, not another command"
    assert argv[2] == str(tmp_path / "published"), "candidate 0's published tree"
    assert Path(argv[3]).name == "serve_check.json"


def test_an_engine_that_refuses_candidate_0_fails_the_launch_with_its_own_words(tmp_path):
    """The check's own reason is the launch's, so the operator reads the engine's refusal, not ours."""
    from qda_agent.sched.job import State

    host = _Host(tmp_path, State.done, said={"ok": False, "reason": "no kernel image is available"})
    with pytest.raises(ValueError, match="no kernel image is available"):
        _trial(host, tmp_path / "published")


def test_a_check_job_that_fails_carries_the_job_log(tmp_path):
    """A job that dies before it writes a result is reported with the cause from its log."""
    from qda_agent.sched.job import State

    host = _Host(tmp_path, State.failed)
    with pytest.raises(ValueError, match="libcuda.so.1"):
        _trial(host, tmp_path / "published")


def test_a_run_that_configures_no_check_command_serves_nothing(tmp_path):
    """`judge.check_cmd: null` is the operator's choice: the trial run submits no job."""
    from qda_agent.sched.job import State

    host = _Host(tmp_path, State.done, template=None)
    host.cfg.check_cmd = None
    _trial(host, tmp_path / "published")
    assert host.submitted == []


def test_the_trial_runs_failure_is_the_launchs_own_error_type(tmp_path, monkeypatch):
    """Whatever step fails, `launch_smoke` raises `TrialRunFailed`: the flow finalizes the run on it."""
    host = SimpleNamespace(
        cfg=SimpleNamespace(
            candidate0_smoke=True, engine="vllm", model="m", format="w4a16", require_build_receipt=False
        ),
        sched=SimpleNamespace(cfg=SimpleNamespace(gpus=True, fake_gpus=False, cards=[0])),
    )
    monkeypatch.setattr(startup, "_start_configs", lambda h: None)

    def refuse(h):
        raise ValueError("candidate 0 failed to build (failed): no space left on device")

    monkeypatch.setattr(startup, "_candidate0", refuse)
    with pytest.raises(startup.TrialRunFailed, match="no space left on device"):
        startup.launch_smoke(cast("JudgeHost", host))


def test_a_cpu_run_and_a_test_s_fake_gpus_do_no_trial_run(tmp_path, monkeypatch):
    """Nothing is built where there is no card to build on, and a run naming no model has nothing to build."""
    called = []
    monkeypatch.setattr(startup, "_start_configs", lambda h: None)
    monkeypatch.setattr(startup, "_candidate0", lambda h: called.append(h))
    for sched, model, fmt in (
        ({"gpus": False, "fake_gpus": False, "cards": [0]}, "m", "w4a16"),
        ({"gpus": True, "fake_gpus": True, "cards": [0]}, "m", "w4a16"),
        ({"gpus": True, "fake_gpus": False, "cards": []}, "m", "w4a16"),
        ({"gpus": True, "fake_gpus": False, "cards": [0]}, None, "w4a16"),
        ({"gpus": True, "fake_gpus": False, "cards": [0]}, "m", None),
    ):
        host = SimpleNamespace(
            cfg=SimpleNamespace(
                candidate0_smoke=True, engine="vllm", model=model, format=fmt, require_build_receipt=False
            ),
            sched=SimpleNamespace(cfg=SimpleNamespace(**sched)),
        )
        startup.launch_smoke(cast("JudgeHost", host))
    assert called == []


def test_the_serving_check_reports_the_engines_cause_and_fix(tmp_path, monkeypatch):
    """vLLM's parent error is bare; the hand-in check's reason carries the line and fix the engine start noted."""
    import json

    serve_check = _serve_check()

    def crash(*args, **kwargs):
        error = RuntimeError("Engine core initialization failed. See root cause above.")
        error.add_note("RuntimeError: Failed to find C compiler.")
        error.add_note("fix:   install gcc, or set CC to a C compiler")
        raise error

    monkeypatch.setattr(serve_check, "check", crash)
    out = tmp_path / "serve_check.json"
    assert serve_check.main([str(tmp_path), str(out)]) == 0
    said = json.loads(out.read_text())
    assert not said["ok"] and said["reason"].startswith("vLLM could not serve it: RuntimeError: Engine core")
    assert said["reason"].endswith(
        "\nRuntimeError: Failed to find C compiler.\nfix:   install gcc, or set CC to a C compiler"
    )


def _serve_check():
    """The judge's `serve_check.py`, loaded as the judge runs it (a file, not a module of qda_agent)."""
    import importlib.util

    from qda_agent.judge.jobs import TOOLS

    spec = importlib.util.spec_from_file_location("serve_check_t", Path(TOOLS) / "serve_check.py")
    assert spec is not None and spec.loader is not None
    serve_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve_check)
    return serve_check


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """The real `engine_start` with its environment only (no guards, compiler probes or cache directories), and
    a stand-in for vLLM whose engine records `os.environ` and its arguments as it starts."""
    import os
    import sys
    import types

    from qda_quant.output import vllm_env

    for name in vllm_env.CACHE_DIRS:
        monkeypatch.setenv(name, str(tmp_path / name))
    for name in vllm_env.NO_JIT_ENV:
        monkeypatch.delenv(name, raising=False)  # restored after the check's own update
    monkeypatch.setattr(vllm_env, "vllm_environ", vllm_env.environ)
    started: dict = {}
    done = SimpleNamespace(outputs=[SimpleNamespace(text="x", finish_reason="stop")])

    class LLM:
        def __init__(self, **kwargs):
            started.update(environ=dict(os.environ), kwargs=kwargs)

        def generate(self, prompts, params):
            return [done] * len(prompts)

    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = LLM, lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return started


def test_the_serving_check_starts_vllm_with_qda_kls_no_jit_settings(tmp_path, engine):
    """The hand-in check on a host without nvcc died in FlashInfer's JIT (`Could not find nvcc`) where `qda-kl kl`
    started: its engine started without qda-kl's no-JIT environment and arguments."""
    from qda_kl.serve import NO_AUTOTUNE_ARGS, ServeSettings, no_jit

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    assert _serve_check().check(ckpt, 64, 4096) == (True, "")
    kl = no_jit(ServeSettings(args={"max_model_len": 4096, "enforce_eager": True, "tensor_parallel_size": 1}))
    assert {name: engine["environ"].get(name) for name in kl.env} == kl.env
    # FlashInfer autotune stays vLLM's default here: only reference, scoring and gencheck engines turn it off
    args = {k: v for k, v in kl.args.items() if k not in NO_AUTOTUNE_ARGS}
    assert engine["kwargs"] == {"model": str(ckpt), **args}


def test_the_serving_check_keeps_the_sampler_the_checkpoint_is_served_with(tmp_path, engine):
    """A `VLLM_USE_FLASHINFER_SAMPLER` in the checkpoint's serve section is what the hand-in is served with."""
    import json

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "build.json").write_text(json.dumps({"serve": {"env": {"VLLM_USE_FLASHINFER_SAMPLER": "1"}}}))
    assert _serve_check().check(ckpt, 64, 4096) == (True, "")
    assert engine["environ"]["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    assert engine["environ"]["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"


def test_a_serving_venv_without_qda_quant_is_refused_in_one_line(tmp_path, engine, monkeypatch):
    """Without qda-quant the check would start vLLM without the no-JIT settings; it starts nothing."""
    import json
    import sys

    # the import fails as it does in a serving venv without qda-quant
    monkeypatch.setitem(sys.modules, "qda_quant.output.vllm_env", None)
    out = tmp_path / "serve_check.json"
    assert _serve_check().main([str(tmp_path), str(out)]) == 0
    said = json.loads(out.read_text())
    assert not said["ok"] and said["reason"].startswith("judge.serve_python needs qda-quant: ")
    assert "\n" not in said["reason"] and engine == {}


@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        (25.2, 60.0),
        (300.0, 600.0),
        (2400.0, startup.CANDIDATE0_TIMEOUT_S),
        (None, startup.CANDIDATE0_TIMEOUT_S),
    ],
)
def test_the_trial_build_is_queued_at_candidate_0s_measured_time(measured, expected):
    """A 15-minute budget's 29.8-minute drain window refused a trial queued at the hour-long timeout."""
    cfg = SimpleNamespace(candidate0_build_s=measured)
    assert startup.candidate0_expected_s(cast("startup.JudgeConfig", cfg)) == expected


def test_the_trial_builds_calibration_fills_its_one_sample(tmp_path):
    """A W4A4 rtn reads the trial build's calibration (its activation scales). The one-sentence row it had was 11
    tokens, 0 samples of 16: every W4A4 search stopped at the trial build, before the agent's first turn."""
    import json

    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.calib import load_ids

    model = startup.tiny_model(tmp_path / "m")
    rows = tmp_path / "calib.jsonl"
    calib = FileSource(path=str(rows), samples=1, seqlen=startup.CANDIDATE0_SEQLEN)
    rows.write_text(json.dumps({"text": startup.CANDIDATE0_TEXT}) + "\n")
    assert [ids.numel() for ids in load_ids(calib, model)] == [startup.CANDIDATE0_SEQLEN]
    rows.write_text(json.dumps({"text": "The launch check builds the run's own model once."}) + "\n")
    with pytest.raises(Exception, match="0 samples of 16"):
        load_ids(calib, model)
