"""A `qda-agent run` stopped by SIGTERM or SIGHUP closes as for Ctrl-C and leaves no job's process behind."""

import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from tests_support import agent, assert_job_group, assert_job_process

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
FIXTURES = ROOT / "tests" / "fixtures"

#: `qda-agent` with the test-only backend registered (it lives in testkit, next to this file).
CLI = [
    sys.executable,
    "-c",
    "import sys, testkit; from qda_agent import cli; sys.exit(cli.main(sys.argv[1:]))",
]


def _env() -> dict[str, str]:
    """The environment of a run started with `CLI`: testkit on the import path."""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "tests"), os.environ.get("PYTHONPATH")])),
    }


def _scripted_config(tmp_path: Path, steps: int) -> dict:
    """The scripted-search fixture, its worker playing the first `steps` of the command script (each waits on
    its build without holding the event loop)."""
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw.update(run_dir=str(tmp_path / "run"), baselines=[str(_baseline(tmp_path))])
    raw["agent"]["backend"] = {"name": "test_scripted", "steps": steps}
    raw["judge"]["evaluator"] = str(EXAMPLES / "standin_kl")
    # these tests are about how a signal ends a run, not about the final rule: they keep the chain that was the
    # default before `handin` (a pending reading is waited for, then the best scored one or the baseline is picked)
    raw["judge"]["final_rules"] = ["last_valid", "best_metric", "baseline"]
    _abs_refs(raw)
    return raw


def _abs_refs(raw: dict) -> None:
    """The fixture's reference paths made absolute: its config is copied outside the fixtures directory."""
    raw["judge"]["kl_ref"] = str(FIXTURES / "ref")
    raw["judge"]["kl_holdout_ref"] = str(FIXTURES / "holdout")


#: A local job with a child: signals readiness without exporting namespace-local process ids.
BLOCKING_JOB = """import os, subprocess, sys, time
child = subprocess.Popen(["sleep", "600"])
with open(os.path.join(sys.argv[1], sys.argv[2]), "w") as f:
    f.write("ready")
time.sleep(600)
"""


def _group(pgid: int) -> list[int]:
    """The live (not zombie) processes of process group `pgid`."""
    pids = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            pids.append(int(stat.parent.name))
    return pids


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads process groups from /proc")
def test_cleanup_refuses_special_caller_and_foreign_groups(tmp_path):
    for pgid in {0, 1, os.getpgid(0), os.getpgid(os.getppid())}:
        with pytest.raises(AssertionError):
            assert_job_group(pgid, tmp_path)
    own = tmp_path / "own"
    own.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             cwd=tmp_path, start_new_session=True)  # fmt: skip
    try:
        with pytest.raises(AssertionError):
            assert_job_group(child.pid, own)
        assert_job_group(child.pid, tmp_path)
    finally:
        child.kill()
        child.wait()


def _groups_in(cwd: Path) -> set[int]:
    """The process groups of the live processes working in `cwd`."""
    out = set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "cwd").readlink() == cwd:
                out.add(os.getpgid(int(proc.name)))
        except OSError:
            continue
    return out


def _baseline(tmp_path: Path) -> Path:
    """A baseline the judge can read: a real (empty) safetensors file."""
    base = tmp_path / "baseline"
    base.mkdir()
    (base / "config.json").write_text("{}")
    header = b'{"__metadata__": {"tag": "b0"}}'.ljust(32)
    (base / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    return base


def _rm(run: Path) -> None:
    """Remove a run dir, its sealed (read-only) directories included."""
    if run.exists():
        for p in [run, *run.rglob("*")]:
            if p.is_dir() and not p.is_symlink():
                p.chmod(0o755)
        shutil.rmtree(run)


def _kill(agent: subprocess.Popen, tmp_path: Path, pgids: set[int]) -> None:
    """Stop the agent and only host-observed, verified groups belonging to this test."""
    agent.kill()
    agent.wait()
    for g in pgids:
        assert_job_group(g, tmp_path)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(g, signal.SIGKILL)


def _start(tmp_path: Path) -> tuple[subprocess.Popen, set[int], Path]:
    """Start a scripted search with a blocking local build and wait for its process group."""
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (tmp_path / "job.py").write_text(BLOCKING_JOB)
    raw = _scripted_config(tmp_path, steps=1)
    raw["budget_gpu_hours"] = 300 / 3600 * 2  # far past the test: only the signal ends the run
    raw["scheduler"].update(gpus=False, job_sandbox="none")
    build = tmp_path / "blocking-build.sh"
    build.write_text(
        "#!/bin/sh\nexec " + shlex.join([sys.executable, str(tmp_path / "job.py"), str(jobs), "build"]) + "\n"
    )
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, testkit; testkit.BUILD_SH = open(sys.argv.pop(1)).read(); "
            "from qda_agent import cli; sys.exit(cli.main(sys.argv[1:]))",
            str(build),
            "run",
            str(config),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_env(),
        cwd=tmp_path,
        start_new_session=True,
    )
    end = time.monotonic() + 60
    while not any(jobs.iterdir()):
        if proc.poll() is not None or time.monotonic() > end:
            proc.kill()
            pytest.fail(f"the blocking local job never started:\n{proc.communicate()[0]}")
        time.sleep(0.1)
    run = tmp_path / "run"
    pgids = _groups_in(run / "work") - {os.getpgid(proc.pid)}
    try:
        assert pgids, "found no host-side process group for the blocking job"
        for g in pgids:
            assert_job_group(g, tmp_path)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    return proc, pgids, run


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads process groups from /proc")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_closes_the_run_and_leaves_no_job_behind(tmp_path, sig):
    """Stopping a real run reaps its job and child, writes final records, and exits 130."""
    agent, pgids, run = _start(tmp_path)
    try:
        assert all(_group(g) for g in pgids)  # the job's tree is up
        agent.send_signal(sig)
        out = agent.communicate(timeout=60)[0]
        end = time.monotonic() + 5  # the init process reaps what it inherited
        while any(_group(g) for g in pgids) and time.monotonic() < end:
            time.sleep(0.1)
        assert not [_group(g) for g in pgids if _group(g)], out
        assert agent.returncode == 130, out
        result = json.loads((run / "result.json").read_text())
        assert result["ended"] == "stopped" and result["rule"] == "baseline", result
        assert (run / "FINAL.json").is_file()
        report = (run / "END-REPORT.md").read_text()
        assert "agent phase ended: stopped" in report and "Humanize trace" in report
        assert not (run / "rpc.sock").exists()
    finally:
        _kill(agent, tmp_path, pgids)


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads process groups from /proc")
def test_a_second_signal_does_not_cut_the_close_short(tmp_path):
    """A second SIGTERM while the run closes is ignored: the close still stops every job and writes the files."""
    agent, pgids, run = _start(tmp_path)
    try:
        agent.send_signal(signal.SIGTERM)
        time.sleep(0.2)
        agent.send_signal(signal.SIGTERM)
        out = agent.communicate(timeout=60)[0]
        assert agent.returncode == 130 and "already closing" in out, out
        assert not [g for g in pgids if _group(g)], out
        assert json.loads((run / "result.json").read_text())["ended"] == "stopped"
    finally:
        _kill(agent, tmp_path, pgids)


#: A stand-in claude whose turn never ends: it signals readiness and waits.
BLOCKING_CLAUDE = """#!/usr/bin/python3
import os, sys, time
if "--version" in sys.argv:
    print("2.1.317 (Claude Code)")
    sys.exit(0)
with open("CLI-READY", "w") as f:
    f.write("ready")
time.sleep(600)
"""


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
def test_a_signal_ends_the_agent_cli_inside_the_wall(tmp_path):
    """SIGTERM while the agent's turn runs in the bwrap wall: the CLI's process group is gone, the run closed."""
    fence = pytest.importorskip("hmz.coganchor.fence")
    if not fence.enforceable(net=False):  # the wall holds the filesystem; the netns holds the network
        pytest.skip("this machine cannot hold the wall's filesystem fence")
    run = tmp_path / "run"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "claude").write_text(BLOCKING_CLAUDE)
    (bindir / "claude").chmod(0o755)
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw.update(
        run_dir=str(run),
        baselines=[str(_baseline(tmp_path))],
        budget_gpu_hours=300 / 3600 * 2,
        agent=agent("claude/claude-haiku-5-5:low", resend_wait_s=1),
        sandbox="bwrap",
        drain_timeout_s=5,
    )
    raw["judge"]["evaluator"] = str(EXAMPLES / "standin_kl")
    _abs_refs(raw)
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw))
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
    # The CLI runs in a process of its own (the test signals it), so it does not see the conftest fixture that
    # stands in for a dedicated node; this launcher does the same for it. A node that refuses a fresh procfs
    # needs `pid: shared`, which the wall refuses unless the node is the run's alone -- and the test node is
    # one Slurm job shared by every track. `tests/test_pid_shared.py` exercises the real check.
    launcher = tmp_path / "cli.py"
    launcher.write_text(
        "import sys\n"
        "from qda_agent.isolation import wall\n"
        "wall.why_not_dedicated = lambda: None  # this test stands in for the dedicated node pid: shared needs\n"
        "from qda_agent.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(launcher), "run", str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=tmp_path,
        start_new_session=True,
    )
    marker = run / "work" / "CLI-READY"
    pgids: set[int] = set()
    try:
        end = time.monotonic() + 60
        while not (marker.is_file() and marker.read_text()):
            if proc.poll() is not None or time.monotonic() > end:
                proc.kill()
                pytest.fail(f"the CLI never started:\n{proc.communicate()[0]}")
            time.sleep(0.1)
        # the pgid inside the wall's pid namespace differs: find the stand-in by its cwd on the host
        pgids = {g for g in _groups_in(run / "work") if g != os.getpgid(proc.pid)}
        assert pgids, "found no process of the agent's turn"
        for g in pgids:
            assert_job_group(g, tmp_path, bindir / "claude")
        proc.send_signal(signal.SIGTERM)
        out = proc.communicate(timeout=60)[0]
        end = time.monotonic() + 5
        while any(_group(g) for g in pgids) and time.monotonic() < end:
            time.sleep(0.1)
        assert not [g for g in pgids if _group(g)], out
        assert proc.returncode == 130, out
        assert json.loads((run / "result.json").read_text())["ended"] == "stopped"
    finally:
        proc.kill()
        proc.wait()
        for g in pgids:
            assert_job_group(g, tmp_path, bindir / "claude")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(g, signal.SIGKILL)
        _rm(run)


#: A KL reading that waits for `RELEASE`: one is still pending when the deadline passes, so the drain waits on
#: it for as long as the machine needs to queue it.
SLOW_KL = """import os, sys, time
while not os.path.exists({release!r}):
    time.sleep(0.05)
os.execv(sys.executable, [sys.executable, {kl!r}, *sys.argv[1:]])
"""

#: Written by a test once its signals are in: lets the pending KL reading finish.
RELEASE = "release"


def _closing_run(tmp_path: Path, **popen) -> tuple[subprocess.Popen, Path]:
    """A scripted search (one build, its KL reading held); returns once its close has begun.

    The build says it expects 5 s of GPU time, and the queue admits no agent job that would end past the
    deadline: the run's wall time leaves a loaded machine room to start the first turn before that.
    """
    release = tmp_path / RELEASE
    (tmp_path / "slow_kl.py").write_text(
        SLOW_KL.format(kl=str(EXAMPLES / "standin_kl" / "kl.py"), release=str(release))
    )
    raw = _scripted_config(tmp_path, steps=1)
    raw.update(budget_gpu_hours=15 / 3600 * 2, cards=[0, 1], drain_timeout_s=30)
    raw["scheduler"]["readable"] = [str(EXAMPLES), sys.prefix]
    raw["judge"].update(
        evaluator=None, kl_cmd=[sys.executable, str(tmp_path / "slow_kl.py"), "{ckpt}", "{out}"]
    )
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw))
    proc = subprocess.Popen(
        [*CLI, "run", str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_env(),
        **popen,
    )
    run, flow = tmp_path / "run", tmp_path / "run" / "flow.jsonl"
    end = time.monotonic() + 60
    # Wait for the reading itself, not the clock: the deadline it must beat comes 15 s in.
    while not any((run / "jobs").glob("*-judge-*")):
        if flow.is_file() and '"cutoff"' in flow.read_text():
            proc.kill()
            pytest.fail(f"no KL reading was queued before the deadline:\n{proc.communicate()[0]}")
        if proc.poll() is not None or time.monotonic() > end:
            proc.kill()
            pytest.fail(f"the run never queued its KL reading:\n{proc.communicate()[0]}")
        time.sleep(0.05)
    while not (flow.is_file() and '"cutoff"' in flow.read_text()):
        if proc.poll() is not None or time.monotonic() > end:
            proc.kill()
            pytest.fail(f"the run never reached its close:\n{proc.communicate()[0]}")
        time.sleep(0.05)
    return proc, run


@pytest.mark.parametrize(
    "signals",
    [
        [signal.SIGTERM],
        [signal.SIGHUP],
        [signal.SIGINT, signal.SIGTERM],
        [signal.SIGINT, signal.SIGINT, signal.SIGTERM],
    ],
)
def test_a_signal_during_the_drain_still_writes_the_pick(tmp_path, signals):
    """A stop after the deadline, while the drain waits on a reading: FINAL.json and the end report are written
    as for a run left alone. SIGTERM and SIGHUP are ignored then (the run is closing); a Ctrl-C waits for the
    files and then exits 130, and a SIGTERM after it is ignored too. Before, the first signal cancelled the
    drain's step: no FINAL.json, and result.json said the flow never finished."""
    proc, run = _closing_run(tmp_path)
    try:
        for sig in signals:
            proc.send_signal(sig)
            time.sleep(0.3)
        (tmp_path / RELEASE).touch()  # let the reading the drain waits on finish
        out = proc.communicate(timeout=90)[0]
        assert proc.returncode == (130 if signal.SIGINT in signals else 0), out
        result = json.loads((run / "result.json").read_text())
        # the pending reading was waited for: the pick is a scored checkpoint, not the baseline
        assert result["ended"] == "deadline" and result["rule"] == "best_metric", (result, out)
        assert json.loads((run / "FINAL.json").read_text())["rule"] == "best_metric"
        assert "## Submission:" in (run / "END-REPORT.md").read_text()
        if signal.SIGINT in signals:
            assert "the submission was picked" in out and "already closing" in out, out
        else:
            assert "already closing" in out, out
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads the run's live processes from /proc")
def test_a_third_ctrl_c_abandons_the_drain(tmp_path):
    """Two Ctrl-C presses keep waiting; a third abandons the pending reading and leaves no process behind."""
    proc, run = _closing_run(tmp_path)
    pids = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            if (path / "cwd").readlink().is_relative_to(tmp_path):
                pid = int(path.name)
                assert_job_process(pid, tmp_path)
                pids[pid] = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except OSError:
            continue

    def live():
        left = []
        for pid, started in pids.items():
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if fields[19] == started and fields[0] != "Z":
                left.append(pid)
        return left

    try:
        assert pids, "found no process for the pending reading"
        for _ in range(2):
            proc.send_signal(signal.SIGINT)
            time.sleep(0.3)
            assert proc.poll() is None, proc.communicate()[0]
        proc.send_signal(signal.SIGINT)
        out = proc.communicate(timeout=60)[0]
        assert proc.returncode == 130, out
        assert "abandoning the close" in out, out
        assert not (tmp_path / RELEASE).exists()
        end = time.monotonic() + 5
        while live() and time.monotonic() < end:
            time.sleep(0.05)
        assert not live(), (live(), out)
    finally:
        proc.kill()
        proc.wait()
        for pid in live():
            assert_job_process(pid, tmp_path)
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_sighup_ignored_by_nohup_stays_ignored(tmp_path):
    """Under `nohup` (SIGHUP inherited as ignored) a hangup does not stop the run."""
    proc, run = _closing_run(tmp_path, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    try:
        proc.send_signal(signal.SIGHUP)
        (tmp_path / RELEASE).touch()  # let the reading the drain waits on finish
        out = proc.communicate(timeout=90)[0]
        assert proc.returncode == 0 and "SIGHUP" not in out, out
        assert json.loads((run / "FINAL.json").read_text())["rule"] == "best_metric"
    finally:
        proc.kill()
        proc.wait()


#: An agent CLI that never answers and ignores SIGINT, as a busy turn's CLI may.
SILENT_CLI = "#!/bin/sh\ntrap '' INT\nexec sleep 600\n"


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads process groups from /proc")
@pytest.mark.parametrize("inherited", ["default", "ignored"])
def test_sigint_stops_qda_agent_run_with_a_real_agent_cli(tmp_path, inherited):
    """Live: SIGINT to `qda-agent run`'s group left it running with Humanize's fence under it. A shell
    starts `qda-agent run CONFIG &` with SIGINT ignored; the run takes it anyway and closes, exit 130, nothing left.
    The agent is the claude backend driven by Humanize, on a CLI that never answers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(SILENT_CLI)
    (bin_dir / "claude").chmod(0o755)
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    run = tmp_path / "run"
    config = tmp_path / "search.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": "tiny/model",
                "contamination": "none",
                "format": "w4a16",
                "budget_gpu_hours": 600 / 3600,
                "cards": [0],
                "run_dir": str(run),
                "sandbox": "none",
                "isolation": {"level": "none"},
                "host": "testkit:fake_host",
                "baselines": [str(tmp_path / "base" / "gptq")],
                "drain_timeout_s": 5,
                "judge": {"eval_max_tokens": 32768},
                "commands": None,
                "scheduler": {"gpus": False},
                "agent": {"resend_wait_s": 0.5, "backend": {"name": "claude", "model": "m", "effort": None}},
            }
        )
    )
    launcher = (
        "import signal, sys, testkit\n"
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN)\n" if inherited == "ignored" else "")
        + "from qda_agent import cli\n"
        "sys.exit(cli.main(sys.argv[1:]))\n"
    )
    env = {**_env(), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CLAUDE_CODE_OAUTH_TOKEN": "fake"}
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher, "run", str(config)],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        flow = run / "flow.jsonl"
        for _ in range(600):  # the first agent turn is under way
            if flow.is_file() and '"ev": "goal"' in flow.read_text():
                break
            assert proc.poll() is None, proc.communicate()[1][-3000:]
            time.sleep(0.1)
        else:
            pytest.fail("the agent's first turn never started")
        time.sleep(1)
        os.killpg(proc.pid, signal.SIGINT)
        try:
            code = proc.wait(timeout=5 + 25)  # drain_timeout_s and a margin
        except subprocess.TimeoutExpired:
            pytest.fail(f"still running 30 s after SIGINT; group: {_group(proc.pid)}")
        err = proc.stderr.read() if proc.stderr else ""
        assert code == 130, err[-3000:]
        assert "the submission was picked and the run closed" in err
        assert json.loads((run / "result.json").read_text())["ended"] == "stopped"
        for _ in range(50):
            if not _group(proc.pid):
                break
            time.sleep(0.1)
        assert _group(proc.pid) == []  # Humanize's fence and the CLI are gone too
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
