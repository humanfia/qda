"""`work_dir`: the agent's workdir in a directory of its own, off the run dir's storage.

The agent's own builds and every job's writes land there, as the judge's builds land in `judge.custody_dir`;
the run's records, the jobs' logs and the export stay in `run_dir`.
"""

import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml
from qda_agent import cli
from qda_agent.config import RunConfig
from qda_agent.configfile import ConfigError
from qda_agent.judge import prune
from qda_agent.sched import FakeClock, Kind, LocalBackend, SchedConfig, Scheduler, State, sandbox
from test_judge import Rig, ckpt
from testkit import scripted

FIXTURES = Path(__file__).parent / "fixtures"
needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


def config(tmp_path: Path, **extra) -> Path:
    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "toolset": {"name": "qda", "components": []},
        "contamination": {"name": "none"},  # no guard file here; the default ngram check needs one
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(raw))
    return path


def local(tmp_path: Path) -> Path:
    """A private parent on 'local storage' for the work dir (the trusted-ancestor rule needs one)."""
    parent = tmp_path / "local"
    parent.mkdir(mode=0o700)
    return parent / "work"


def test_work_dir_is_the_agents_workdir_and_is_recorded(tmp_path):
    work = local(tmp_path)
    cfg = RunConfig.load(config(tmp_path, work_dir=str(work)))
    assert cfg.workdir == work and not work.exists()  # made at launch, not by loading the config
    assert cfg.judge_config()["workdir"] == work  # the judge takes kl and handin paths in it
    assert cfg.socket == work / ".qda" / "rpc.sock"
    assert cfg.resolved()["work_dir"] == str(work)
    assert "work_dir" not in RunConfig.load(config(tmp_path)).resolved()  # unset: records as before
    assert RunConfig.load(config(tmp_path)).workdir == tmp_path / "run" / "work"


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ("in-run", "overlaps run input or bind"),
        ("holds-run", "overlaps run input or bind"),
        ("input", "overlaps run input or bind"),
        ("writable", "writable ancestor"),
        ("file", "already exists"),
        ("custody", "overlaps run input or bind"),
    ],
)
def test_a_work_dir_outside_custodys_rules_is_refused(tmp_path, bad, why):
    work = local(tmp_path)
    extra: dict = {"work_dir": str(work)}
    if bad == "in-run":
        (tmp_path / "run").mkdir()
        extra["work_dir"] = str(tmp_path / "run" / "elsewhere")
    elif bad == "holds-run":
        extra["work_dir"] = str(tmp_path)
    elif bad == "input":
        (work / "baseline").mkdir(parents=True)  # a baseline inside it: the agent's jobs could rewrite it
        extra["baselines"] = [str(work / "baseline")]
    elif bad == "writable":
        work.parent.chmod(0o777)
    elif bad == "file":
        work.write_text("not a directory")
    else:
        work.mkdir(mode=0o700)  # an existing work dir passes until launch; the custody inside it does not
        extra["judge"] = {"custody_dir": str(work / "custody"), "max_run_bytes": 1 << 20}
    with pytest.raises(ConfigError, match=rf"work_dir \(.*\): {why}"):
        RunConfig.load(config(tmp_path, **extra))


def test_a_launch_refuses_a_work_dir_that_exists(tmp_path, capsys):
    work = local(tmp_path)
    work.mkdir()
    (work / "precious.txt").write_text("an earlier run's")
    path = FIXTURES / "scripted-search.yaml"
    run = tmp_path / "run"
    rc = cli.main(
        ["run", str(path), "--set", f"run_dir={run}", "--set", f"work_dir={work}", "--set", "sandbox=none"]
    )
    assert rc == 2 and "work_dir" in capsys.readouterr().err
    assert (work / "precious.txt").read_text() == "an earlier run's"


def test_a_run_with_a_work_dir_writes_the_agents_files_there_and_its_records_in_run_dir(tmp_path):
    work = local(tmp_path)
    run = tmp_path / "run"
    assert scripted(FIXTURES / "scripted-search.yaml", run, 3, overrides=[f"work_dir={work}"]) == 0
    assert not (run / "work").exists()
    assert (work / "TASK.md").is_file() and (work / ".qda").is_dir()
    assert work.stat().st_mode & 0o777 == 0o700
    assert yaml.safe_load((run / "resolved.yaml").read_text())["work_dir"] == str(work)
    assert (run / "FINAL.json").is_file() and (run / "flow.jsonl").is_file()


def test_the_judge_reads_and_takes_a_checkpoint_from_the_work_dir_and_counts_it(tmp_path):
    work = local(tmp_path)
    work.mkdir()
    r = Rig(tmp_path, workdir=work, max_run_bytes=1 << 30, max_scratch_bytes=1 << 30)
    r.work = work
    on_run_storage = prune.run_bytes(r.host, scratch=True)
    ckpt(work / "cand", b"agent-built")  # what an agent job's `qda build` wrote in its workdir
    size = (work / "cand" / "model.safetensors").stat().st_size
    assert prune.run_bytes(r.host) >= on_run_storage + size  # the run's cap counts the work dir
    assert prune.run_bytes(r.host, scratch=True) == on_run_storage  # the run dir's storage does not hold it
    assert r.call("kl", "cand").exit == 0
    assert r.call("handin", "cand").exit == 0
    r.settle()
    kinds = {e.kind: e for e in r.host.entries}
    assert kinds["scored"].valid is True and kinds["handin"].valid is True
    assert r.call("kl", "../run/judge").exit == 2  # still nothing outside the workdir


PROBE = r"""
import json, sys
from pathlib import Path
work, run, beside = (Path(a) for a in sys.argv[1:4])
out = {}
def attempt(name, fn):
    try:
        fn(); out[name] = "ok"
    except OSError as e:
        out[name] = type(e).__name__
attempt("work", lambda: (work / "ckpt").mkdir() or (work / "ckpt" / "model.gguf").write_bytes(b"x"))
attempt("run", lambda: (run / "planted").write_text("x"))
attempt("beside", lambda: (beside / "planted").write_text("x"))
print("PROBE " + json.dumps(out))
"""


@needs_bwrap
def test_an_agent_jobs_write_lands_in_the_work_dir_and_nowhere_else(tmp_path):
    work = local(tmp_path)
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    s = Scheduler(
        run, SchedConfig(cards=[0], budget_s=600, gpus=False), LocalBackend(1), FakeClock(time.time())
    )
    s.submit(
        [sandbox.python(), "-c", PROBE, str(work), str(run), str(work.parent)],
        str(work),
        name="h",
        expected_s=60,
        root=work,  # the scheduler's writable root is the run's workdir (`judge.commands`: `host.workdir`)
        kind=Kind.build,
    )
    deadline = time.time() + 30
    while s.pending():
        assert time.time() < deadline, s.log("h")
        s.tick()
        time.sleep(0.05)
    log = s.log("h")
    assert s.get("h").state is State.done, log
    res = json.loads(next(x for x in log.splitlines() if x.startswith("PROBE ")).split(" ", 1)[1])
    assert res["work"] == "ok", res
    assert (work / "ckpt" / "model.gguf").read_bytes() == b"x"  # on the host, in the work dir
    # the run dir and the work dir's parent are not the job's: a write there fails or lands in a throwaway
    # tmpfs of the sandbox, never on the host
    assert not (run / "planted").exists() and not (work.parent / "planted").exists(), res


def test_the_walls_writable_bind_follows_the_work_dir(tmp_path):
    """The agent's own turn writes its workdir, wherever `work_dir` puts it, and nothing of the run dir."""
    fence = pytest.importorskip("hmz.coganchor.fence")
    from qda_agent import isolation
    from qda_agent.isolation import SANDBOX
    from qda_agent.isolation.cli_dirs import live_dirs

    work = local(tmp_path)
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    cfg = RunConfig(
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        work_dir=work,
        contamination={"name": "none"},
        sandbox="bwrap",  # this test is about the wall: it names one (the product default walls nothing)
    )
    f = fence.Fence.of(
        local=str(SANDBOX.local),
        user=str(SANDBOX.user),
        system=str(SANDBOX.system),
        online=False,
        workdir=work,
        home=Path.home(),
        hosts=["api.anthropic.com"],
    )
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(f)
    argv = json.loads(Path(argv[4]).read_text()) + argv[5:]
    binds = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--bind"]
    assert (str(work), str(work)) in binds, binds  # the agent's workdir, writable, at its own path
    assert ["--tmpfs", str(run)] == argv[
        argv.index(str(run)) - 1 : argv.index(str(run)) + 1
    ]  # run dir hidden
    # of the run dir, the turn writes only its CLI's live config dir, as without `work_dir`
    lives = {os.path.realpath(d) for d in live_dirs(cfg).values()}
    inside = {
        os.path.realpath(d) for _, d in binds if Path(os.path.realpath(d)).is_relative_to(run.resolve())
    }
    assert inside <= lives, (inside, lives)
