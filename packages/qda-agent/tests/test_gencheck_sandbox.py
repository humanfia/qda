"""What the agent can reach of a long-generation check: nothing but the counts.

The judge keeps the frozen copy, the original model's answers (`original_cache`), the prompt set and the texts
in its own dir under the run dir. These tests run a probe in the agent's wall (the real bwrap wall around a
fenced turn) and in an agent job (the job sandbox) and check that it reads none of them, and that `gencheck`
refuses a path outside the workdir or a link out like `kl`.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from qda_agent.sandbox_settings import Bwrap, NoSandbox
from qda_agent.sched import FakeClock, Kind, LocalBackend, SchedConfig, Scheduler, State
from qda_agent.sched.sandbox import python_roots
from test_gencheck import GenRig
from test_judge import ckpt

fence = pytest.importorskip("hmz.coganchor.fence")

from qda_agent import isolation  # noqa: E402
from test_isolation import _fence, needs_wall  # noqa: E402

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")

PROBE = r"""
import json, os, sys
out = {}
for p in sys.argv[1:]:
    try:
        if os.path.isdir(p):
            os.listdir(p)
            for root, _, files in os.walk(p):
                for f in files:
                    open(os.path.join(root, f), "rb").read(1)
        else:
            open(p, "rb").read(1)
        out[p] = "read"
    except OSError as e:
        out[p] = type(e).__name__
print("PROBE " + json.dumps(out))
"""


@pytest.fixture
def checked(monkeypatch):
    """A run (outside the temp dirs, which the wall replaces) whose judge has run one long-generation check."""
    base = Path(__file__).parents[1] / "runs" / f"gencheck-{uuid.uuid4().hex[:8]}"
    home = base / "home"  # the wall hides the home; a fake one keeps the real one out of the test
    home.mkdir(parents=True)
    cli = base / "cli"  # the same for the CLI's config dir, which `live_state` copies: never the launcher's
    cli.mkdir()
    (cli / "settings.json").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cli))
    prompts = home / "prompts.jsonl"
    prompts.write_text('{"id": "1", "domain": "math", "prompt": "PROMPT-SECRET"}\n')
    r = GenRig(base, gencheck_prompts=prompts)
    r.host.install_prompts()
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert "done" in r.call("gencheck", "show", "g1").stdout
    ckpt(r.work / "b", weights=b"w2")
    r.call("gencheck", "b")  # still generating: its frozen copy is there
    jdir = r.run / "judge"
    secret = [
        jdir / "frozen" / "g2",
        jdir / "gencheck-original",
        jdir / "gencheck-prompts.jsonl",
        jdir / "results" / "g1" / "texts",
        jdir / "results" / "g1.yaml",
    ]
    for p in secret:  # control: each is there and readable on the host
        assert p.exists(), p
    yield r, secret
    for d in [p for p in base.rglob("*") if p.is_dir() and not p.is_symlink()]:
        d.chmod(0o755)
    shutil.rmtree(base, ignore_errors=True)


def _probe(argv, cwd, paths):
    p = subprocess.run(
        [*argv, "/usr/bin/python3", "-c", PROBE, *map(str, paths)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    return json.loads(next(x for x in p.stdout.splitlines() if x.startswith("PROBE "))[6:])


@needs_bwrap
def test_the_probe_reads_them_on_the_host(checked):
    r, secret = checked
    assert set(_probe([], r.work, secret).values()) == {"read"}  # the probe below is not vacuous


@needs_wall
def test_the_agents_shell_reads_nothing_of_a_check(checked):
    from qda_agent.config import RunConfig

    r, secret = checked
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=r.run,
        sandbox="bwrap",  # this test is about the wall: it names one (the product default walls nothing)
    )
    with isolation.gpu_wall(cfg):  # the turn's way out is the wall's bridge: probe while it is up
        argv = fence.wrapper(_fence(r.work))
        options = json.loads(Path(argv[4]).read_text())  # bwrap's own arguments, carried in a private file
        assert options[0] == "bwrap"
        seen = _probe(argv, r.work, secret)
    assert "read" not in seen.values(), seen


@needs_bwrap
def test_an_agent_job_reads_nothing_of_a_check(checked):
    r, secret = checked
    probe = r.work / "probe.py"
    probe.write_text(PROBE)
    venv = r.run.parent / "serve-venv"  # what the defaults bind: a serving venv and a dataset dir
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    data = r.run.parent / "data"
    data.mkdir()
    (data / "corpus.jsonl").write_text("{}\n")
    s = Scheduler(
        r.run.parent / "sched",
        SchedConfig(cards=[5], budget_s=600, job_sandbox="bwrap", readable=[str(venv), str(data)]),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    py = os.path.normpath(Path(sys.executable).absolute())
    s.submit(
        [py, "probe.py", *map(str, secret)],
        str(r.work),
        name="p",
        expected_s=60,
        root=r.work,
        kind=Kind.build,
    )
    end = time.time() + 30
    while s.pending():
        assert time.time() < end
        s.tick()
        time.sleep(0.05)
    assert s.get("p").state is State.done, s.log("p")
    line = next(x for x in s.log("p").splitlines() if x.startswith("PROBE "))
    seen = json.loads(line[6:])
    assert "read" not in seen.values(), seen


@pytest.mark.parametrize("how", ["absolute", "parent", "dir link", "file link", "link into the judge"])
def test_gencheck_refuses_a_path_outside_the_workdir_or_a_link_out(tmp_path, how):
    r = GenRig(tmp_path)
    outside = ckpt(tmp_path / "outside")
    if how == "absolute":
        arg = str(outside)
    elif how == "parent":
        arg = "../../outside"
    elif how == "dir link":
        (r.work / "a").symlink_to(outside)
        arg = "a"
    elif how == "file link":
        d = ckpt(r.work / "a")
        (d / "model.safetensors").unlink()
        (d / "model.safetensors").symlink_to(outside / "model.safetensors")
        arg = "a"
    else:  # a link to the original's answers, to have them frozen and "checked"
        (r.run / "judge" / "gencheck-original").mkdir()
        d = ckpt(r.work / "a")
        (d / "answers").symlink_to(r.run / "judge" / "gencheck-original")
        arg = "a"
    rep = r.call("gencheck", arg)
    assert rep.exit == 2 and "refused" in rep.stderr, rep
    assert r.host.sched.list(origin="judge") == [] and not (r.run / "judge" / "frozen" / "g1").exists()
    assert how == "absolute" or str(tmp_path) not in rep.stderr  # only the agent's own argument is echoed


def _cfg(tmp_path, **kw):
    from qda_agent.config import RunConfig

    base = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": tmp_path / "run",
    }
    return RunConfig.model_validate({**base, **kw})


def test_a_prompt_set_the_wall_would_show_is_refused(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "p").mkdir(parents=True)
    (home / "p" / "x.jsonl").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    shown = tmp_path / "data" / "prompts.jsonl"
    shown.parent.mkdir()
    shown.write_text("{}\n")
    with pytest.raises(ValueError, match="readable from the agent's shell"):
        isolation._prompts_hidden(_cfg(tmp_path, judge={"gencheck_prompts": shown}))
    isolation._prompts_hidden(_cfg(tmp_path, judge={"gencheck_prompts": home / "p" / "x.jsonl"}))
    hidden = {"name": "bwrap", "hide": [str(tmp_path / "data")]}
    isolation._prompts_hidden(_cfg(tmp_path, sandbox=hidden, judge={"gencheck_prompts": shown}))
    no_cmd = {"name": "qda", "components": ["kl"]}
    isolation._prompts_hidden(_cfg(tmp_path, toolset=no_cmd, judge={"gencheck_prompts": shown}))


def test_judge_paths_in_the_run_dir_or_holding_it_are_refused(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="judge.gencheck_prompts .*inside run_dir"):
        _cfg(tmp_path, judge={"gencheck_prompts": run / "work" / "p.jsonl"})
    with pytest.raises(ValueError, match="judge.serve_python .*inside run_dir"):
        _cfg(
            tmp_path,
            toolset={"name": "qda", "components": ["kl"]},
            judge={"serve_python": run / "work/v/bin/python"},
        )
    with pytest.raises(ValueError, match="holds run_dir"):  # it would bind the judge's dir into the jobs
        _cfg(tmp_path, scheduler={"name": "qsched", "readable": [str(tmp_path)]})


def test_serve_must_be_a_virtualenv(tmp_path):
    py = tmp_path / "srv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    with pytest.raises(ValueError, match="not a virtualenv's interpreter"):
        _cfg(tmp_path, judge={"serve_python": py})
    (tmp_path / "srv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (serve,) = [c for c in _cfg(tmp_path, judge={"serve_python": py}).toolset.components if c.name == "serve"]
    assert serve.python == py


@needs_bwrap
def test_the_jobs_still_read_what_is_bound(checked):
    """Control for the job probe: a bound dataset is read, so a refusal above is the sandbox, not the probe."""
    r, _ = checked
    data = r.run.parent / "data2"
    data.mkdir()
    (data / "corpus.jsonl").write_text("{}\n")
    (r.work / "probe.py").write_text(PROBE)
    s = Scheduler(
        r.run.parent / "sched2",
        SchedConfig(cards=[5], budget_s=600, job_sandbox="bwrap", readable=[str(data)]),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    py = os.path.normpath(Path(sys.executable).absolute())
    s.submit([py, "probe.py", str(data / "corpus.jsonl")], str(r.work), name="p", expected_s=60, root=r.work)
    end = time.time() + 30
    while s.pending():
        assert time.time() < end
        s.tick()
        time.sleep(0.05)
    line = next(x for x in s.log("p").splitlines() if x.startswith("PROBE "))
    assert json.loads(line[6:]) == {str(data / "corpus.jsonl"): "read"}


@pytest.mark.parametrize("where", ["data", "readable", "serve", "python", "host python"])
def test_a_prompt_set_the_jobs_read_is_refused_in_every_sandbox_mode(tmp_path, monkeypatch, where):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    # a venv whose interpreter is there is bound; a missing one binds nothing
    (venv / "bin" / "python").touch()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    kw = {
        "data": {
            "toolset": {
                "name": "qda",
                "components": ["gencheck", {"name": "data", "sets": {"c": str(shared)}}],
            }
        },
        "readable": {"scheduler": {"name": "qsched", "readable": [str(shared)]}},
        "serve": {
            "toolset": {
                "name": "qda",
                "components": ["gencheck", {"name": "serve", "python": str(venv / "bin/python")}],
            }
        },
        "python": {
            "toolset": {"name": "qda", "components": ["gencheck"], "python": str(venv / "bin/python")}
        },
        "host python": {},
    }[where]
    if where == "host python":
        # A file under a root the running interpreter's jobs bind, taken from `python_roots()` itself: an
        # editable install's `.pth` may point into another tree than the one `isolation` was imported from,
        # so `isolation.__file__` is not always under one.
        src = Path(sys.executable)
        roots = python_roots()
        assert any(src.resolve().is_relative_to(r) for r in roots), (src, roots)
    else:
        src = (venv if where in ("serve", "python") else shared) / "prompts.jsonl"
        src.write_text("{}\n")  # a set path must be there (and this one must be readable)
    ok = tmp_path / "home" / "p.jsonl"
    ok.parent.mkdir(parents=True, exist_ok=True)
    ok.write_text("{}\n")
    with pytest.raises(ValueError, match="which the agent's jobs read"):  # the config check refuses it
        _cfg(tmp_path, sandbox={"name": "none"}, judge={"gencheck_prompts": str(src)}, **kw)
    for mode in ("none", "bwrap"):
        # The prompt check is the same in every mode; validate the config under `none` because the fenced
        # modes' writable sets (see `Sandbox.writable`) include /tmp, where these prompt files live. Set the
        # prompt set after validation: the isolation check is the second line, for a config changed in memory.
        cfg = _cfg(tmp_path, sandbox={"name": "none"}, **kw)
        cfg.sandbox = {"none": NoSandbox, "bwrap": Bwrap}[mode]()
        cfg.judge.gencheck_prompts = src
        with pytest.raises(ValueError, match="which the agent's jobs read"):
            isolation._prompts_unread(cfg)
        if mode == "none":  # and through the sandbox check itself
            with pytest.raises(ValueError, match="which the agent's jobs read"):
                isolation.check(cfg)
    isolation._prompts_unread(_cfg(tmp_path, judge={"gencheck_prompts": str(ok)}, **kw))
