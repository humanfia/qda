import contextlib
import json
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from qda_agent.sched import (
    FakeBackend,
    FakeClock,
    FakeRun,
    Kind,
    LocalBackend,
    Refused,
    SchedConfig,
    Scheduler,
    State,
    kill_every_group,
)
from qda_agent.sched import backend as backend_module
from qda_agent.sched.queue import exit_reason
from qda_agent.sched.sandbox import card_uuids


def make(tmp_path, cards=(0, 1, 2, 3), budget_s=3600.0, runs=None):
    clock = FakeClock()
    be = FakeBackend(clock, runs=runs or {})
    cfg = SchedConfig(cards=list(cards), budget_s=budget_s, fake_gpus=True)
    return Scheduler(tmp_path, cfg, be, clock), be, clock


def run_until_idle(s, clock, step=10.0, limit=10_000):
    for _ in range(limit):
        s.tick()
        busy = [c for j in s.jobs.values() if j.state is State.running for c in j.cards]
        assert len(busy) == len(set(busy)), "two running jobs share a card"
        if not s.pending():
            return
        clock.advance(step)
    raise AssertionError("queue never drained")


def test_jobs_get_disjoint_cards_and_queue_when_full(tmp_path):
    s, be, clock = make(tmp_path, runs={f"000{i}-b{i}": FakeRun(100) for i in range(1, 4)})
    for i in range(1, 4):
        s.submit(["true"], str(tmp_path), name=f"b{i}", ncards=2, expected_s=100)
    s.tick()
    st = {j.name: j.state for j in s.list()}
    assert st == {"b1": State.running, "b2": State.running, "b3": State.queued}
    assert set(s.get("b1").cards).isdisjoint(s.get("b2").cards)
    clock.advance(100)
    s.tick()
    assert s.get("b1").state is State.done and s.get("b3").state is State.running
    run_until_idle(s, clock)
    assert all(j.state is State.done for j in s.list())


def test_score_jumps_queued_builds_but_never_preempts(tmp_path):
    s, be, clock = make(tmp_path, cards=(0, 1))
    s.submit(["true"], "/", name="b1", ncards=2, expected_s=60)
    s.tick()
    s.submit(["true"], "/", name="b2", ncards=1, expected_s=60)
    s.submit(["true"], "/", name="kl1", ncards=1, expected_s=60, kind=Kind.score, origin="judge")
    s.tick()
    assert s.get("b1").state is State.running  # not stopped for the score
    assert s.get("kl1").state is State.queued
    clock.advance(60)
    s.tick()
    assert s.get("kl1").state is State.running
    assert [n for n, _, _ in be.launched] == ["0001-b1", "0003-kl1", "0002-b2"]


def test_eight_card_score_uses_the_build_cards_before_the_next_build(tmp_path):
    s, be, clock = make(tmp_path, cards=tuple(range(8)))
    s.submit(["true"], "/", name="b1", ncards=8, expected_s=60)
    s.tick()
    s.submit(["true"], "/", name="b2", ncards=8, expected_s=60)
    s.submit(["true"], "/", name="kl1", ncards=8, expected_s=60, kind=Kind.score, origin="judge")
    assert s.get("b1").state is State.running
    assert s.get("b2").state is State.queued and s.get("kl1").state is State.queued
    clock.advance(60)
    s.tick()
    assert s.get("kl1").state is State.running and s.get("b2").state is State.queued
    clock.advance(60)
    s.tick()
    assert s.get("b2").state is State.running
    assert [len(cards) for _, cards, _ in be.launched] == [8, 8, 8]


@pytest.mark.parametrize("judge_first, order", [(True, ["b1", "kl1", "b2"]), (False, ["b1", "b2", "kl1"])])
def test_scheduler_priority_orders_the_judge_lane(tmp_path, judge_first, order):
    """Turning judge priority off keeps FIFO order for queued readings and builds."""
    clock = FakeClock()
    be = FakeBackend(clock)
    cfg = SchedConfig(cards=[0], budget_s=3600.0, fake_gpus=True, judge_first=judge_first)
    s = Scheduler(tmp_path, cfg, be, clock)
    s.submit(["true"], "/", name="b1", expected_s=60)
    s.tick()
    s.submit(["true"], "/", name="b2", expected_s=60)
    s.submit(["true"], "/", name="kl1", expected_s=60, kind=Kind.score, origin="judge")
    run_until_idle(s, clock, step=60.0)
    assert [n.split("-", 1)[1] for n, _, _ in be.launched] == order


def test_blocked_multicard_job_holds_free_cards(tmp_path):
    s, be, clock = make(tmp_path, cards=(0, 1, 2))
    s.submit(["true"], "/", name="small", ncards=1, expected_s=100)
    s.tick()  # small on card 0
    s.submit(["true"], "/", name="kl", ncards=3, expected_s=50, kind=Kind.score, origin="judge")
    s.submit(["true"], "/", name="b", ncards=1, expected_s=10)
    s.tick()
    assert s.get("b").state is State.queued, "a build must not take the cards the score waits for"
    assert "behind kl" in s.get("b").reason
    clock.advance(100)
    s.tick()
    assert s.get("kl").state is State.running and s.get("kl").cards == [0, 1, 2]


def test_deadline_admission(tmp_path):
    s, be, clock = make(tmp_path, cards=(0,), budget_s=1000)
    with pytest.raises(Refused, match="deadline"):
        s.submit(["true"], "/", name="long", expected_s=2000)
    s.submit(["true"], "/", name="a", expected_s=600)
    s.submit(["true"], "/", name="b", expected_s=600)  # fits now, but only after a ends
    be.runs["0001-a"] = FakeRun(600)
    s.tick()
    clock.advance(600)
    s.tick()
    b = s.get("b")
    assert b.state is State.failed and "deadline" in b.reason and b.started is None


def test_dependencies(tmp_path):
    s, be, clock = make(tmp_path, runs={"0001-a": FakeRun(10, rc=3)})
    s.submit(["true"], "/", name="a", expected_s=10)
    s.submit(["true"], "/", name="b", expected_s=10, after=["a"])
    s.submit(["true"], "/", name="c", expected_s=10)
    s.submit(["true"], "/", name="d", expected_s=10, after=["c"])
    with pytest.raises(Refused, match="unknown"):
        s.submit(["true"], "/", name="e", expected_s=10, after=["zz"])
    s.tick()
    assert s.get("b").state is State.queued and "after a" in s.get("b").reason
    assert s.get("d").state is State.queued
    run_until_idle(s, clock)
    assert s.get("a").state is State.failed and s.get("a").rc == 3
    assert s.get("b").state is State.failed and s.get("b").started is None
    assert s.get("d").state is State.done
    assert s.get("d").started >= s.get("c").ended


def test_script_snapshot_at_submit(tmp_path):
    s, be, clock = make(tmp_path)
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    script = work / "scripts" / "x.sh"
    script.write_text("echo v1\n")
    job = s.submit([], str(work), name="x", expected_s=10, root=work, freeze="scripts/x.sh")
    script.write_text("echo v2\n")
    assert job.argv[0] == "bash" and len(job.argv) == 2
    snap = Path(job.argv[1])
    assert snap.read_text() == "echo v1\n" and snap.is_relative_to(tmp_path / "jobs")
    job = s.submit(
        ["tool", "build", "{frozen}", "--x"], str(work), name="y", expected_s=10, freeze="scripts/x.sh"
    )
    assert job.argv[:2] == ["tool", "build"] and Path(job.argv[2]).read_text() == "echo v2\n"


def test_cancel_list_log_and_origin(tmp_path):
    s, be, clock = make(tmp_path, runs={"0001-a": FakeRun(100, output="l1\nl2\nl3\n")})
    s.submit(["true"], "/", name="a", expected_s=100)
    s.submit(["true"], "/", name="j", expected_s=100, kind=Kind.check, origin="judge")
    s.submit(["true"], "/", name="q", expected_s=100, ncards=4)
    s.tick()
    assert s.log("a", lines=2) == "l2\nl3"
    assert [j.name for j in s.list(origin="agent")] == ["a", "q"]
    with pytest.raises(Refused):
        s.cancel("j", origin="agent")
    with pytest.raises(Refused):
        s.log("j", origin="agent")
    s.cancel("a", origin="agent")
    s.cancel("q", origin="agent")
    assert s.get("a").state is State.cancelled and s.get("q").state is State.cancelled


def test_cutoff_keeps_judge_jobs(tmp_path):
    s, be, clock = make(tmp_path)
    s.submit(["true"], "/", name="a", expected_s=100)
    s.submit(["true"], "/", name="aq", expected_s=100, ncards=4)
    s.submit(["true"], "/", name="j", expected_s=100, kind=Kind.check, origin="judge")
    s.tick()
    s.cutoff()
    assert s.get("a").state is State.cancelled and s.get("aq").state is State.cancelled
    assert s.get("j").state is State.running
    with pytest.raises(Refused, match="judge"):
        s.submit(["true"], "/", name="late", expected_s=1)
    s.submit(["true"], "/", name="j2", expected_s=1, kind=Kind.score, origin="judge")
    assert s.pending("judge") and not s.pending("agent")


def test_state_is_written(tmp_path):
    s, be, clock = make(tmp_path)
    job = s.submit(["true"], "/", name="a", expected_s=10)
    s.tick()
    rec = (tmp_path / "jobs" / job.id / "job.json").read_text()
    assert '"state": "running"' in rec


def test_config():
    with pytest.raises(ValueError):
        SchedConfig(cards=[1, 1])
    assert SchedConfig(cards=[0], budget_s=10).resolve_deadline(5.0) == 15.0
    assert SchedConfig(cards=[0]).resolve_deadline(5.0) == float("inf")


# ---------------------------------------------------------------------- real processes, no GPU
def wait_rc(be, h, limit=10.0):
    t = time.time() + limit
    while (rc := be.poll(h)) is None:
        assert time.time() < t, "process did not end"
        time.sleep(0.02)
    return rc


def test_local_backend_sets_cards_and_rc(tmp_path):
    be = LocalBackend(kill_grace_s=1)
    log = tmp_path / "log"
    h = be.launch(["bash", "-c", "echo cards=$CUDA_VISIBLE_DEVICES; exit 4"], str(tmp_path), [2, 5], log, "t")
    assert wait_rc(be, h) == 4
    assert log.read_text().strip() == "cards=2,5"


@pytest.mark.parametrize("file_wrapper", [False, True])
def test_sandbox_environment_is_carried_by_private_descriptor_not_command_line(tmp_path, file_wrapper):
    """A secret set in bwrap options must reach the child without appearing in readable host argv."""
    wrapper = tmp_path / "bwrap"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, stat, sys\n"
        "fd = int(sys.argv[2])\n"
        "assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600\n"
        "with os.fdopen(fd, 'rb') as f: options = f.read().decode().rstrip('\\0').split('\\0')\n"
        "print(json.dumps({'argv': sys.argv, 'options': options}))\n"
    )
    wrapper.chmod(0o755)
    secret = "dummy-api-key-for-argv-check"
    backend = LocalBackend()
    log = tmp_path / "log"
    argv = [str(wrapper), "--clearenv", "--setenv", "OPENAI_API_KEY", secret, "--", "true"]
    if file_wrapper:
        from qda_agent.sched.sandbox import private_bwrap_argv

        argv = private_bwrap_argv(argv, tmp_path)
        assert secret not in " ".join(argv)
        assert Path(argv[4]).stat().st_mode & 0o777 == 0o600
        handle = backend.launch(argv, str(tmp_path), [0], log, "private-settings")
    else:  # the level hands the backend a private-args descriptor, as `Isolation.job` builds it
        from qda_agent.sched.sandbox import private_args

        with private_args(argv) as (command, fds):
            assert secret not in " ".join(command)
            handle = backend.launch(command, str(tmp_path), [0], log, "private-settings", pass_fds=fds)
    assert wait_rc(backend, handle) == 0
    got = json.loads(log.read_text())
    assert secret not in " ".join(got["argv"])
    assert got["options"] == ["--clearenv", "--setenv", "OPENAI_API_KEY", secret]
    if file_wrapper:
        assert not Path(argv[4]).exists()


def test_local_backend_stop_kills_the_group(tmp_path):
    be = LocalBackend(kill_grace_s=1)
    pidf = tmp_path / "child.pid"
    h = be.launch(
        ["bash", "-c", f"sleep 60 & echo $! > {pidf}; wait"], str(tmp_path), [0], tmp_path / "log", "t"
    )
    t = time.time() + 5
    while not pidf.exists() or not pidf.read_text().strip():
        assert time.time() < t
        time.sleep(0.02)
    child = int(pidf.read_text())
    be.stop(h)
    assert be.poll(h) is not None
    time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_local_backend_grace_is_the_groups_not_the_leaders(tmp_path):
    """A wrapper that ends at once on SIGTERM (a lease tool) still leaves its job the grace to shut down."""
    be = LocalBackend(kill_grace_s=5)
    done, pidf = tmp_path / "clean", tmp_path / "child.pid"
    child = f"trap 'sleep 0.3; echo ok > {done}; exit 0' TERM; while :; do sleep 0.05; done"
    h = be.launch(
        ["bash", "-c", f'bash -c "{child}" & echo $! > {pidf}; wait'],
        str(tmp_path),
        [0],
        tmp_path / "log",
        "t",
    )
    t = time.time() + 5
    while not pidf.exists() or not pidf.read_text().strip():
        assert time.time() < t
        time.sleep(0.02)
    time.sleep(0.2)  # the child's trap is set
    t0 = time.monotonic()
    be.stop(h)
    assert (
        done.read_text() == "ok\n" and time.monotonic() - t0 < 4
    )  # the child finished; no wait for the rest
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidf.read_text()), 0)


def test_kill_every_group_ends_the_jobs_a_cut_close_left(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend_module, "_LIVE", set()
    )  # the registry is per process: other tests' jobs are not ours
    be = LocalBackend(kill_grace_s=1)
    h = be.launch(["sleep", "60"], str(tmp_path), [0], tmp_path / "log", "t")
    done = be.launch(["true"], str(tmp_path), [0], tmp_path / "log2", "u")
    wait_rc(be, done)  # seen to end: no longer counted
    assert kill_every_group() == 1 and be.poll(h) == -signal.SIGKILL
    assert kill_every_group() == 0


def test_a_stopped_queue_takes_and_starts_nothing(tmp_path):
    """After `stop_all` (the host's close), a request still in flight can neither queue nor start a job."""
    s, be, clock = make(tmp_path)
    s.submit(["true"], "/", name="a", expected_s=1)
    s.stop_all()
    assert s.get("a").state is State.cancelled
    with pytest.raises(Refused, match="closing"):
        s.submit(["true"], "/", name="b", expected_s=1)
    s.tick()
    assert be.launched == []


def test_scheduler_on_local_backend(tmp_path):
    clock = FakeClock(time.time())
    cfg = SchedConfig(cards=[0, 1], budget_s=3600, job_sandbox="none")  # plain local backend; no device bound
    s = Scheduler(tmp_path, cfg, LocalBackend(kill_grace_s=1), clock)
    (tmp_path / "x.sh").write_text("echo hi $CUDA_VISIBLE_DEVICES\n")
    s.submit([], str(tmp_path), name="x", expected_s=5, freeze="x.sh")
    t = time.time() + 10
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.02)
    assert s.get("x").state is State.done and s.log("x") == "hi " + ",".join(card_uuids([0]))  # by UUID


KILLED = "killed by signal 9 (SIGKILL), not by the job itself; often out of memory or an operator kill"


@pytest.mark.parametrize(
    ("script", "reason"),
    [
        ("kill -KILL $$", KILLED),
        ("kill -TERM $$", "killed by signal 15 (SIGTERM)"),
        ("bash -c 'kill -KILL $$'", f"exit 137: {KILLED}"),  # a kill inside the job, as bwrap reports it
    ],
)
def test_a_job_killed_by_a_signal_says_so(tmp_path, script, reason):
    """The agent reads why (`qsched show`): a signal by name, not `exit -9`."""
    clock = FakeClock(time.time())
    cfg = SchedConfig(cards=[0], budget_s=3600, job_sandbox="none")
    s = Scheduler(tmp_path, cfg, LocalBackend(kill_grace_s=1), clock)
    (tmp_path / "x.sh").write_text(script + "\n")
    s.submit([], str(tmp_path), name="x", expected_s=5, freeze="x.sh")
    t = time.time() + 10
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.02)
    assert s.get("x").state is State.failed and s.get("x").reason == reason
    assert not (tmp_path / "jobs" / s.get("x").id / "pgid").exists()  # an ended job's pgid is not left
    assert exit_reason(3) == "exit 3" and exit_reason(-200) == "killed by signal 200"
    assert exit_reason(-11) == "killed by signal 11 (SIGSEGV)"


def test_zero_cards_refused(tmp_path):
    s, be, clock = make(tmp_path)
    with pytest.raises(Refused, match="at least one card"):
        s.submit(["true"], "/", name="z", ncards=0, expected_s=1)


@pytest.mark.parametrize("case", ["dotdot", "nested-dotdot", "absolute", "symlink-out", "dir", "missing"])
def test_script_must_be_a_file_in_the_workdir(tmp_path, case):
    s, be, clock = make(tmp_path)
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    secret = tmp_path / "secret.sh"
    secret.write_text("echo secret\n")
    (work / "link.sh").symlink_to(secret)
    path = {
        "dotdot": "../secret.sh",
        "nested-dotdot": "sub/../../secret.sh",
        "absolute": str(secret),
        "symlink-out": "link.sh",
        "dir": "sub",
        "missing": "nope.sh",
    }[case]
    with pytest.raises(Refused, match="outside|relative|not a file|not readable|without"):
        s.submit([], str(work), name="x", expected_s=1, root=work, freeze=path)
    assert not any((tmp_path / "jobs").iterdir())


def test_edit_after_submit_runs_the_old_bytes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.sh").write_text("echo v1\n")
    s = Scheduler(
        tmp_path,
        SchedConfig(cards=[0], budget_s=600, job_sandbox="none"),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    s.submit([], str(work), name="x", expected_s=5, root=work, freeze="x.sh")
    (work / "x.sh").write_text("echo v2\n")
    t = time.time() + 10
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.02)
    assert s.log("x") == "v1"


def test_a_failed_dependency_fails_at_once(tmp_path):
    s, be, clock = make(tmp_path, runs={"0001-slow": FakeRun(1000), "0002-bad": FakeRun(10, rc=1)})
    s.submit(["true"], "/", name="slow", expected_s=1000)
    s.submit(["true"], "/", name="bad", expected_s=10)
    s.submit(["true"], "/", name="dep", expected_s=10, after=["slow", "bad"])
    s.tick()
    clock.advance(10)
    s.tick()
    dep = s.get("dep")
    assert s.get("slow").state is State.running
    assert dep.state is State.failed and dep.reason == "after bad: failed" and dep.started is None


def test_judge_deadline_is_the_gpu_deadline_plus_grace(tmp_path):
    clock = FakeClock()
    cfg = SchedConfig(cards=[0], budget_s=100, fake_gpus=True, judge_grace_s=50)
    s = Scheduler(tmp_path, cfg, FakeBackend(clock), clock)
    clock.advance(100)
    s.cutoff()
    s.submit(["true"], "/", name="j", expected_s=50, kind=Kind.score, origin="judge")
    with pytest.raises(Refused, match="drain window"):
        s.submit(["true"], "/", name="j2", expected_s=51, kind=Kind.score, origin="judge")
    s.tick()
    assert s.get("j").state is State.running


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_snapshot_survives_swaps_of_file_and_parent(tmp_path):
    """Race the snapshot: swap the script and its parent dir for links out of the workdir while it copies."""
    s, be, clock = make(tmp_path)
    work = tmp_path / "work"
    (work / "d").mkdir(parents=True)
    secret_dir = tmp_path / "judge"
    secret_dir.mkdir()
    (secret_dir / "job.sh").write_text("echo SYNTHETIC_JUDGE_SECRET\n")
    good = "echo fine\n"
    stop = threading.Event()

    def swapper():
        while not stop.is_set():
            for path, target in ((work / "d" / "job.sh", secret_dir / "job.sh"), (work / "d", secret_dir)):
                tmp = path.with_name(path.name + ".tmp")
                with contextlib.suppress(OSError):
                    if path.is_dir() and not path.is_symlink():
                        (path / "job.sh").write_text(good)
                    tmp.unlink(missing_ok=True)
                    tmp.symlink_to(target)
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    tmp.replace(path)  # now a link out
                with contextlib.suppress(OSError):
                    path.unlink()  # back to real
                    if path.name == "d":
                        path.mkdir()
                    (work / "d" / "job.sh").write_text(good)

    (work / "d" / "job.sh").write_text(good)
    t = threading.Thread(target=swapper, daemon=True)
    t.start()
    ok = refused = 0
    try:
        for i in range(60):
            try:
                job = s.submit([], str(work), name=f"x{i}", expected_s=1, root=work, freeze="d/job.sh")
            except Refused:
                refused += 1
                continue
            text = Path(job.argv[1]).read_text()
            assert "SECRET" not in text, f"attempt {i} copied the outside file"
            assert text in (good, "")  # "" : the swapper's write_text truncates before it writes
            ok += 1
    finally:
        stop.set()
        t.join()
    assert ok + refused == 60


@pytest.mark.parametrize("tree", [False, True])
def test_snapshot_refusal_names_the_sandbox_error_and_doctor(tmp_path, monkeypatch, tree):
    """A sandbox startup failure must give its cause, rather than blame a readable source file."""
    from qda_agent.sched.snapshot import SnapshotError, snapshot_file, snapshot_tree

    work, into = tmp_path / "work", tmp_path / "into"
    work.mkdir()
    into.mkdir()
    (work / "job.sh").write_text("echo yes\n")
    (work / "checkpoint").mkdir()
    launcher = tmp_path / "bwrap"
    launcher.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"bwrap: Can't mount proc: Operation not permitted\" 'extra diagnostics' >&2\n"
        "exit 1\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with pytest.raises(SnapshotError) as refused:
        if tree:
            snapshot_tree(work, "checkpoint", into / "frozen")
        else:
            snapshot_file(work, "job.sh", into)
    message = str(refused.value)
    assert "bwrap: Can't mount proc: Operation not permitted" in message
    assert "qda doctor --search CONFIG.yaml" in message
    assert "extra diagnostics" not in message


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_snapshot_refuses_links_fifos_and_big_files(tmp_path):
    from qda_agent.sched.snapshot import SnapshotError, snapshot_file

    work, into = tmp_path / "work", tmp_path / "into"
    work.mkdir()
    into.mkdir()
    (tmp_path / "secret").write_text("SECRET")
    (work / "out.sh").symlink_to(tmp_path / "secret")
    (work / "in.sh").write_text("echo in\n")
    (work / "rel.sh").symlink_to("in.sh")  # a link inside the workdir is copied as a link: refused too
    (work / "big.sh").write_bytes(b"x" * 2048)
    os.mkfifo(work / "fifo.sh")
    for name in ("out.sh", "rel.sh", "fifo.sh"):
        with pytest.raises(SnapshotError):
            snapshot_file(work, name, into, timeout_s=2)
    with pytest.raises(SnapshotError, match="larger"):
        snapshot_file(work, "big.sh", into, max_bytes=1024)
    with pytest.raises(SnapshotError, match="without"):
        snapshot_file(work, "../secret", into)
    snap = snapshot_file(work, "in.sh", into)
    assert snap.read_text() == "echo in\n" and not os.access(snap, os.W_OK) or os.geteuid() == 0
    assert sorted(p.name for p in into.iterdir()) == ["in.sh"]  # refused copies are removed


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_a_snapshot_is_independent_of_the_source(tmp_path):
    """A clone, or a copy where the filesystem cannot clone: no write the agent can still make reaches it."""
    import hashlib

    from qda_agent.sched.snapshot import snapshot_tree

    work, run = tmp_path / "work", tmp_path / "run"
    (work / "ckpt").mkdir(parents=True)
    run.mkdir()
    for name in ("a.bin", "b.bin"):
        (work / "ckpt" / name).write_bytes(name.encode() * 512)
    held = os.open(work / "ckpt" / "a.bin", os.O_WRONLY)  # opened before the snapshot, written after it
    try:
        frozen = snapshot_tree(work, "ckpt", run / "frozen")
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()}
        (work / "ckpt" / "a.bin").write_bytes(b"tampered")  # its own path
        os.link(work / "ckpt" / "b.bin", work / "b-link.bin")  # a second name for the same inode
        (work / "b-link.bin").write_bytes(b"tampered")
        os.pwrite(held, b"tampered", 0)  # the fd it still holds
        (work / "ckpt" / "b.bin").rename(work / "ckpt" / "b.old")  # renamed away and replaced
        (work / "ckpt" / "b.bin").write_bytes(b"replacement")
    finally:
        os.close(held)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()} == before
    for p in frozen.iterdir():
        assert p.stat().st_ino != (work / "ckpt" / p.name).stat().st_ino  # a copy, never the same inode


def test_chain_of_failed_and_cancelled_deps(tmp_path):
    s, be, clock = make(tmp_path, runs={"0001-a": FakeRun(10, rc=3)})
    s.submit(["true"], "/", name="a", expected_s=10)
    s.submit(["true"], "/", name="b", expected_s=10, after=["a"])
    s.submit(["true"], "/", name="c", expected_s=10, after=["b"])
    s.submit(["true"], "/", name="x", expected_s=10)
    s.submit(["true"], "/", name="y", expected_s=10, after=["x"])
    s.submit(["true"], "/", name="z", expected_s=10, after=["y", "c"])
    s.cancel("x")
    run_until_idle(s, clock)
    assert [s.get(n).state for n in "abcxyz"] == [
        State.failed,
        State.failed,
        State.failed,
        State.cancelled,
        State.failed,
        State.failed,
    ]
    assert all(s.get(n).started is None for n in "bcyz")
    assert s.get("y").reason == "after x: cancelled"


def test_snapshot_oserror_frees_the_name(tmp_path, monkeypatch):

    s, be, clock = make(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.sh").write_text("true\n")

    def boom(*a, **k):
        raise PermissionError("disk")

    monkeypatch.setattr(s.iso, "copy_out", boom)  # the level's copy-out is what a submission uses
    with pytest.raises(Refused, match="could not be copied"):
        s.submit([], str(work), name="x", expected_s=1, root=work, freeze="x.sh")
    assert not any((tmp_path / "jobs").iterdir())
    monkeypatch.undo()
    if shutil.which("bwrap"):
        assert s.submit([], str(work), name="x", expected_s=1, root=work, freeze="x.sh").name == "x"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_a_blocking_script_copy_does_not_hold_the_queue(tmp_path, monkeypatch):
    """A FIFO named as the script blocks `cp` until its timeout; the queue must keep working."""
    from qda_agent.sched import snapshot

    monkeypatch.setattr(snapshot, "SNAPSHOT_TIMEOUT_S", 3.0)
    monkeypatch.setattr(snapshot.snapshot_file, "__defaults__", (snapshot.SNAPSHOT_MAX, 3.0))
    s, be, clock = make(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    os.mkfifo(work / "fifo.sh")
    (work / "ok.sh").write_text("true\n")
    errs = []

    def slow():
        try:
            s.submit([], str(work), name="slow", expected_s=1, root=work, freeze="fifo.sh")
        except Refused as e:
            errs.append(str(e))

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.3)  # the FIFO copy is now blocked in cp
    t0 = time.time()
    s.submit([], str(work), name="fast", expected_s=1, root=work, freeze="ok.sh")
    s.tick()
    assert time.time() - t0 < 2.0, "the queue waited for another submission's copy"
    t.join()
    assert errs and "slow" not in s.jobs
    assert [p.name for p in (tmp_path / "jobs").iterdir()] == ["0001-fast"]  # no staging dir left behind


def test_auto_name_never_replaces_a_named_job(tmp_path):
    s, be, clock = make(tmp_path)
    s.submit(["true"], "/", name="job2", expected_s=10)
    j = s.submit(["true"], "/", expected_s=10)  # seq 2: job2 is taken
    assert j.name != "job2" and s.get("job2").seq == 1 and len(s.jobs) == 2
    assert s.submit(["true"], "/", expected_s=10).name not in ("job2", j.name)
    with pytest.raises(Refused, match="exists"):
        s.submit(["true"], "/", name="job2", expected_s=10)


def test_unknown_config_keys_are_refused():
    from qda_agent.judge import JudgeConfig

    with pytest.raises(ValueError, match="carsd"):
        SchedConfig(cards=[0], carsd=[1])
    with pytest.raises(ValueError, match="qda_envv"):
        JudgeConfig(contamination="none", kl_ref="r", qda_envv={"PATH": "/x"})
    assert JudgeConfig(contamination="none", kl_ref="r", qda_env={"PATH": "/x"}).qda_env == {"PATH": "/x"}


def test_a_confined_job_scheduler_names_the_card_nvidia_smi_shows(tmp_path, monkeypatch):
    """Slurm's one-card job: the queue's UUID and the job's device node are the GPU the job got, not PCI card 0.

    Both the scheduler's launch environment (`Queue.uuids` -> `CUDA_VISIBLE_DEVICES=<uuid>`) and the sandbox's
    device binds resolve through the one `resolve_cards`, so they agree with the job's own nvidia-smi.
    """
    import subprocess

    from qda_agent.isolation import devices
    from test_doctor import confined_smi, gpu_table

    proc, dev = gpu_table(tmp_path)  # PCI ordinal 3 is minor 0 at bus 0000:3a:00.0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(subprocess, "run", confined_smi({"visible": "00000000:3A:00.0"}))
    clock = FakeClock()
    s = Scheduler(tmp_path, SchedConfig(cards=[0], budget_s=60.0), FakeBackend(clock), clock, proc=proc)
    assert s.uuids == {0: "GPU-0"}  # the visible card's UUID, not "GPU-7" of PCI ordinal 0
    nodes = devices.device_nodes([0], proc, dev)
    assert dev / "nvidia0" in nodes and dev / "nvidia7" not in nodes
