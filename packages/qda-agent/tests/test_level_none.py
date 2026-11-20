"""`isolation.level: none`: a search with no sandbox tooling, and P2 held by a digest.

The machine these tests describe has no bubblewrap and no Slurm (`no_sandbox_tools`): every sandbox tool is
off PATH and no `SLURM_*` variable is set. A search at `none` must run there, reach FINAL.json, and never
start bwrap, squeue or scontrol.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from qda_agent.config import RunConfig
from qda_agent.isolation.api import JobSpec, make
from qda_agent.isolation.levels.none import NoneIsolation
from qda_agent.isolation.trusted import TrustChanged, Trusted, trusted_paths
from testkit import scripted

FIXTURES = Path(__file__).parent / "fixtures"
#: Programs a `none` run must never start (and that this machine does not have).
TOOLS = ("bwrap", "squeue", "scontrol", "sbatch", "srun")


@pytest.fixture
def no_sandbox_tools(monkeypatch):
    """Bubblewrap off PATH, no Slurm environment: a workstation, a cloud VM or a container."""
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name in TOOLS else real(name, *a, **k))
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURMD_NODENAME", "SLURM_JOB_NAME"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _fresh_bwrap_probe(monkeypatch):
    """Probe this host's bwrap forms again per test: the probe caches its answer per process.

    A test that runs with bwrap off PATH must not leave that answer behind for a later test that has it.
    """
    from qda_agent.sched import sandbox

    monkeypatch.setattr(sandbox, "_ns", None)


@pytest.fixture
def bwrap_shim(tmp_path, monkeypatch):
    """A `bwrap` on PATH that logs its argv and exits 127, as a host whose bwrap cannot start.

    A none run must not start it for anything: not a job, not a snapshot copy, not the pid-form probe. The
    returned path exists only if something did.
    """
    from qda_agent.isolation import bwrap as bwrap_mod

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    log = tmp_path / "bwrap-exec.log"
    shim = shim_dir / "bwrap"
    shim.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\nexit 127\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(bwrap_mod, "_ns", None)
    return log


@pytest.fixture
def execs(monkeypatch):
    """Every argv started through `subprocess.Popen` during the test."""
    seen: list[list[str]] = []
    real = subprocess.Popen

    class Recorded(real):  # type: ignore[misc, valid-type]
        def __init__(self, args, *rest, **kwargs):
            seen.append([str(a) for a in (args if isinstance(args, (list, tuple)) else [args])])
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", Recorded)
    return seen


def config(tmp_path: Path, *overrides: str) -> RunConfig:
    """The scripted search config, run in `tmp_path`, with `overrides` applied."""
    return RunConfig.load(FIXTURES / "scripted-search.yaml", [f"run_dir={tmp_path / 'run'}", *overrides])


def a_job(tmp_path: Path) -> JobSpec:
    return JobSpec(argv=["true"], cwd=tmp_path, root=tmp_path, env={"A": "1"})


def test_the_strict_level_hashes_nothing(tmp_path, monkeypatch):
    """P2 at strict is the wall's alone: the level reads no trusted file (zero digests taken)."""
    from qda_agent.isolation import trusted as trust

    seen: list[Path] = []
    monkeypatch.setattr(trust, "path_digest", lambda p: seen.append(Path(p)) or "x")
    monkeypatch.setattr(trust, "_sha256", lambda p: seen.append(Path(p)) or "x")
    cfg = config(tmp_path, "isolation.level=strict")
    iso = make(cfg)
    assert iso.level == "strict"
    iso.verify_trusted(())
    assert seen == []


def test_a_product_run_gets_the_none_level(tmp_path, no_sandbox_tools):
    """A config that names no sandbox knob is the product default: what a workstation, VM or container runs."""
    cfg = config(tmp_path)
    assert cfg.isolation.level == "none"
    assert isinstance(make(cfg), NoneIsolation)


def test_the_none_level_is_selected_and_runs_a_job_plainly(tmp_path, no_sandbox_tools):
    """A run that names the level gets it, and it needs nothing this host lacks."""
    cfg = config(tmp_path, "isolation.level=none")
    assert cfg.isolation.level == "none"
    iso = make(cfg)
    assert isinstance(iso, NoneIsolation) and iso.level == "none"
    iso.preflight(cfg)
    launch = iso.job(a_job(tmp_path))
    assert launch.argv == ("true",)
    assert launch.env["A"] == "1" and launch.env["PATH"] == os.environ["PATH"]  # the host env plus the job's
    assert launch.pass_fds == ()
    assert iso.provenance()["level"] == "none"


def test_the_kill_domain_is_the_one_it_was_given(tmp_path):
    """The scheduler starts every job through `kill`; the level keeps the domain it was given."""

    class FakeKill:
        def spawn(self, launch, *, cwd, log):  # noqa: ANN001, ANN201
            raise NotImplementedError

        def stop(self, handle, grace_s):  # noqa: ANN001, ANN201
            raise NotImplementedError

        def poll(self, handle):  # noqa: ANN001, ANN201
            return None

    fake = FakeKill()
    iso = NoneIsolation(config(tmp_path, "isolation.level=none"), kill=fake)
    assert iso.kill is fake


def test_none_never_asks_for_a_dedicated_node(tmp_path, no_sandbox_tools, monkeypatch):
    """The Slurm/allocation check is the experiment's; `none` never reaches it."""
    monkeypatch.setattr(
        "qda_agent.isolation.allocation.why_not_dedicated",
        lambda: pytest.fail("a none run asked whether the node is its own"),
    )
    cfg = config(tmp_path, "isolation.level=none")
    iso = make(cfg)
    iso.preflight(cfg)
    iso.job(a_job(tmp_path))


def test_a_none_search_reaches_final_without_sandbox_tools(tmp_path, no_sandbox_tools, execs):
    """The scripted search runs to FINAL.json, and no process it starts is bwrap, squeue or scontrol."""
    run = tmp_path / "run"
    overrides = ["isolation.level=none", "scheduler.job_sandbox=none"]
    assert scripted(FIXTURES / "scripted-search.yaml", run, 25, overrides=overrides) == 0
    final = json.loads((run / "FINAL.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert final["rule"] == result["rule"] == "handin" and final["choice"] is not None
    assert result["host_errors"] == 0
    assert yaml.safe_load((run / "resolved.yaml").read_text())["isolation"]["level"] == "none"
    assert execs  # the search really started jobs
    started = [Path(argv[0]).name for argv in execs]
    assert not [name for name in started if name in TOOLS]


def test_a_none_run_records_no_pid_form_and_walls_nothing(tmp_path, bwrap_shim):
    """A1's probe: a none run keeps the product sandbox setting (`bwrap`) yet resolves and walls nothing.

    The level reports `pid_form: null` (the value a non-bwrap sandbox records, and what `resolved.yaml`
    keeps as `provenance.pid`), and the bwrap on PATH is never started -- not even to ask which pid form
    it could give.
    """
    from qda_agent import isolation

    # the product sandbox (bwrap) as the setting, and the level saying nothing is walled: A1's shape
    cfg = config(tmp_path, "isolation.level=none", "sandbox=bwrap")
    assert cfg.sandbox.name == "bwrap"
    assert make(cfg).provenance()["pid_form"] is None
    assert "pid" not in cfg.resolved()["provenance"]  # the probe left config loading
    with isolation.gpu_wall(cfg):
        pass
    assert not bwrap_shim.exists(), bwrap_shim.read_text()


def test_a_none_search_starts_no_bwrap_where_bwrap_exists(tmp_path, bwrap_shim):
    """The scripted search at none reaches FINAL.json, and a bwrap that exists and fails is never started."""
    run = tmp_path / "run"
    overrides = ["isolation.level=none", "scheduler.job_sandbox=none"]
    assert scripted(FIXTURES / "scripted-search.yaml", run, 25, overrides=overrides) == 0
    final = json.loads((run / "FINAL.json").read_text())
    assert final["rule"] == "handin" and final["choice"] is not None
    assert not bwrap_shim.exists(), bwrap_shim.read_text()


def test_the_build_estimate_copies_the_file_through_the_none_level(tmp_path, bwrap_shim, monkeypatch):
    """The judge's local build estimate copies the plan file through `NoneIsolation.copy_file`: no bwrap.

    The site keeps `snapshot_file`'s file-only contract, so at none the level's single-file copy is what
    runs (a tree copy would not refuse a directory or link the way the walled snapshot does).
    """
    from types import SimpleNamespace

    from qda_agent.judge.build_receipt import estimate

    iso = make(config(tmp_path, "isolation.level=none"))
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    (work / "scripts" / "build.yaml").write_text("model: tiny\nout: candidate\n")
    judge = tmp_path / "judge"
    judge.mkdir()
    seen: list[tuple[Path, str]] = []
    plain = iso.copy_file

    def copy_file(root, rel, into):  # noqa: ANN001, ANN202 -- the copy's own signature
        seen.append((Path(root), rel))
        return plain(root, rel, into)

    monkeypatch.setattr(iso, "copy_file", copy_file)
    host = SimpleNamespace(
        workdir=work,
        jdir=judge,
        sched=SimpleNamespace(iso=iso, cfg=SimpleNamespace(cards=[])),
        cfg=SimpleNamespace(engine="qda", budget_bytes=None, budget_basis="file", anchor_bytes=lambda: None),
        max_bytes=4096,
    )
    assert estimate(host, work, "scripts/build.yaml", []) == 4096  # the fallback: the plan is unreadable
    assert seen == [(work, "scripts/build.yaml")]
    assert not bwrap_shim.exists(), bwrap_shim.read_text()


def test_copy_file_at_none_refuses_a_directory_and_a_link_like_the_walled_snapshot(tmp_path):
    """`NoneIsolation.copy_file` is file-only: one regular file is copied, a directory or link is refused."""
    from qda_agent.sched.snapshot import SnapshotError

    iso = make(config(tmp_path, "isolation.level=none"))
    work = tmp_path / "work"
    work.mkdir()
    (work / "build.yaml").write_text("model: tiny\nout: candidate\n")
    (work / "a-directory").mkdir()
    (work / "a-link").symlink_to("build.yaml")
    into = tmp_path / "into"
    into.mkdir()
    assert iso.copy_file(work, "build.yaml", into).read_text() == "model: tiny\nout: candidate\n"
    with pytest.raises(SnapshotError, match="not a regular file"):
        iso.copy_file(work, "a-directory", into)
    with pytest.raises(SnapshotError, match="not a file"):
        iso.copy_file(work, "a-link", into)
    assert [p.name for p in into.iterdir()] == ["build.yaml"]  # a refused copy leaves nothing behind


def test_a_none_close_never_snapshots_a_codex_session(tmp_path, monkeypatch):
    """A none run never copies a session through bwrap, and never reads a planted live dir.

    The live dir is planted anyway (`none` does not build one), so the guard is what is tested: the close
    must not reach `codexlog.session_calls` and its bwrap tree copy; it reads the run's epic, here none.
    """
    from hmz.runtime import epic
    from qda_agent import agents, codexlog, flow

    monkeypatch.setattr(epic, "epics", lambda workspace=None: [])

    cfg = RunConfig(
        contamination="none",
        model="tiny/model",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        isolation={"level": "none"},
        agent={"backend": {"name": "codex", "model": "gpt-5.6-sol", "effort": "low"}},
    )
    sessions = cfg.run_dir / "live" / "codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout-synthetic.jsonl").write_text(
        json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "synthetic"}})
        + "\n"
    )

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 -- the copy's own signature
        raise AssertionError("a none close reached the bwrap snapshot")

    monkeypatch.setattr(codexlog, "snapshot_tree", forbidden)
    events = flow._Log(cfg.run_dir / "flow.jsonl", lambda: 1.0)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", lambda: 1.0)
    agents.log_codex_calls(cfg, tools, events)
    assert (cfg.run_dir / "flow.jsonl").read_text() == ""
    assert (cfg.run_dir / "tools.jsonl").read_text() == ""


def test_a_changed_judge_file_refuses_the_next_job(tmp_path):
    """P2 at none: the agent can write the judge's files, so a digest refuses the reading and is recorded."""
    cfg = config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none")
    iso = make(cfg)
    assert isinstance(iso, NoneIsolation)
    judge = cfg.run_dir / "judge"
    judge.mkdir(parents=True)
    record = judge / "judge.json"
    record.write_text('{"evaluator": "qda_kl"}\n')
    with iso.session(cfg):
        assert iso.job(a_job(tmp_path)).argv == ("true",)
        record.write_text('{"evaluator": "not the judge"}\n')  # the agent edits a judge file
        with pytest.raises(TrustChanged, match="judge-config"):
            iso.job(a_job(tmp_path))
    events = [json.loads(line) for line in (cfg.run_dir / "trust.jsonl").read_text().splitlines()]
    refused = [e for e in events if e["event"] == "refused"]
    assert refused and "judge-config" in refused[-1]["changed"]


def test_tamper_check_none_runs_without_the_digest(tmp_path):
    """D-a behind its flag: `none` runs plain processes and records no digest."""
    cfg = config(
        tmp_path, "isolation.level=none", "isolation.tamper_check=none", "scheduler.job_sandbox=none"
    )
    iso = make(cfg)
    assert isinstance(iso, NoneIsolation) and iso.trusted is None
    judge = cfg.run_dir / "judge"
    judge.mkdir(parents=True)
    record = judge / "judge.json"
    record.write_text("{}\n")
    with iso.session(cfg):
        iso.job(a_job(tmp_path))
        record.write_text('{"changed": true}\n')
        iso.job(a_job(tmp_path))  # no trust domain: nothing refuses
        iso.verify_trusted(())
    assert not (cfg.run_dir / "trust.jsonl").exists()
    assert iso.provenance()["tamper_check"] == "none"


def test_a_caller_naming_the_judge_directory_adds_nothing(tmp_path):
    """A caller's directory adds no digest of its own: the host's own writes under it are not a tamper."""
    cfg = config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none")
    iso = make(cfg)
    assert isinstance(iso, NoneIsolation)
    judge = cfg.run_dir / "judge"
    judge.mkdir(parents=True)
    (judge / "judge.json").write_text("{}\n")
    with iso.session(cfg):
        iso.job(a_job(tmp_path))
        (judge / "ledger.jsonl").write_text('{"id": 1}\n')  # the host's record, written while the run goes
        iso.verify_trusted([judge])


def test_the_scorer_installed_during_start_is_adopted_then_guarded(tmp_path):
    """The judge installs `judge/eval` after the run's session starts: seen once, then a change refuses."""
    cfg = config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none")
    iso = make(cfg)
    assert isinstance(iso, NoneIsolation)
    eval_dir = cfg.run_dir / "judge" / "eval"
    with iso.session(cfg):
        eval_dir.mkdir(parents=True)
        (eval_dir / "kl.py").write_text("score = 1\n")
        iso.job(a_job(tmp_path))  # the judge's scorer is installed by now: adopted
        (eval_dir / "kl.py").write_text("score = 0\n")
        with pytest.raises(TrustChanged, match="judge-eval"):
            iso.verify_trusted(())


def test_the_trusted_set_names_the_judge_the_reference_and_the_scorer(tmp_path):
    names = trusted_paths(config(tmp_path, "isolation.level=none"))
    assert {"judge-config", "judge-eval", "reference", "holdout"} <= set(names)
    assert names["reference"].is_dir()


def test_a_changed_reference_refuses(tmp_path):
    """The reference is digested like the judge's state: an edited prompt set is a changed digest."""
    reference = tmp_path / "reference"
    manifest = reference / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"k": 8}\n')
    record = tmp_path / "trust.jsonl"
    trusted = Trusted({"reference": reference}, record=record)
    trusted.take()
    manifest.write_text('{"k": 4}\n')
    with pytest.raises(TrustChanged, match="reference"):
        trusted.verify("reading")
    assert json.loads(record.read_text().splitlines()[-1])["event"] == "refused"


def test_copy_out_follows_no_link_the_agent_planted(tmp_path):
    """A file goes into the directory, a tree to a new path, and a link at the top is refused."""
    from qda_agent.sched.snapshot import SnapshotError

    cfg = config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none")
    iso = make(cfg)
    work = tmp_path / "agent-work"
    work.mkdir()
    (work / "notes.txt").write_text("hi\n")
    files = tmp_path / "files"
    files.mkdir()
    assert iso.copy_out(work, "notes.txt", files) == files / "notes.txt"
    out = iso.copy_out(work, ".", tmp_path / "copied")
    assert (out / "notes.txt").read_text() == "hi\n"
    (work / "link").symlink_to("/etc/passwd")
    with pytest.raises(SnapshotError, match="link"):
        iso.copy_out(work, "link", files)
    copied = iso.copy_out(work, ".", tmp_path / "copied-again")
    assert (copied / "link").is_symlink()  # inside a tree it travels as a link; the caller refuses it


def test_the_default_kill_domain_is_the_reaper(tmp_path):
    """Given none, the level starts jobs through `isolation.reaper.Reaper`, the kernel subreaper."""
    from qda_agent.isolation.reaper import Reaper

    iso = make(config(tmp_path, "isolation.level=none"))
    assert isinstance(iso.kill, Reaper)
    assert iso.kill is iso.kill  # built once, not per job


@pytest.mark.parametrize("rel", ["/etc/hostname", "../outside.txt"])
def test_copy_out_refuses_a_path_outside_the_work_directory(tmp_path, rel):
    """`rel` is a path inside the work directory, as `snapshot_tree` holds: absolute and `..` are refused."""
    from qda_agent.sched.snapshot import SnapshotError

    iso = make(config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none"))
    work = tmp_path / "agent-work"
    work.mkdir()
    (tmp_path / "outside.txt").write_text("secret\n")
    files = tmp_path / "files"
    files.mkdir()
    with pytest.raises(SnapshotError, match="without"):
        iso.copy_out(work, rel, files)


def test_copy_out_keeps_the_source_mode_bits_like_cp_a(tmp_path):
    """A script the agent wrote stays runnable in the judge's copy."""
    iso = make(config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none"))
    work = tmp_path / "agent-work"
    work.mkdir()
    script = work / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    files = tmp_path / "files"
    files.mkdir()
    out = iso.copy_out(work, "run.sh", files)
    assert out.stat().st_mode & 0o777 == 0o755


def test_copy_out_streams_a_large_file(tmp_path):
    """A checkpoint shard is not read whole into memory: the copy streams it in chunks."""
    import tracemalloc

    iso = make(config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none"))
    work = tmp_path / "agent-work"
    work.mkdir()
    shard = work / "shard.bin"
    with shard.open("wb") as f:
        f.truncate(64 << 20)
    files = tmp_path / "files"
    files.mkdir()
    tracemalloc.start()
    iso.copy_out(work, "shard.bin", files)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # the 16 MiB chunking bound, plus the buffers the source's and the copy's files hold: a block each (8 MiB on GPFS)
    assert peak < (16 << 20) + 2 * max(shard.stat().st_blksize, files.stat().st_blksize), peak


def test_copy_out_an_unopenable_source_is_a_snapshot_error(tmp_path):
    """`os.open`'s refusal is the error the caller sees: no chmod in a finally masks it."""
    from qda_agent.sched.snapshot import SnapshotError

    iso = make(config(tmp_path, "isolation.level=none", "scheduler.job_sandbox=none"))
    work = tmp_path / "agent-work"
    work.mkdir()
    locked = work / "locked.bin"
    locked.write_text("x")
    locked.chmod(0)
    files = tmp_path / "files"
    files.mkdir()
    with pytest.raises(SnapshotError, match="not readable"):
        iso.copy_out(work, "locked.bin", files)


def test_a_change_behind_a_link_inside_the_tree_refuses(tmp_path):
    """A link that stays inside the trusted tree is walked: content it reaches is what the tree depends on."""
    eval_dir = tmp_path / "eval"
    (eval_dir / "sub").mkdir(parents=True)
    (eval_dir / "sub" / "m.py").write_text("a\n")
    (eval_dir / "lib").symlink_to(eval_dir / "sub")
    record = tmp_path / "trust.jsonl"
    trusted = Trusted({"judge-eval": eval_dir}, record=record)
    trusted.take()
    (eval_dir / "sub" / "m.py").write_text("b\n")
    with pytest.raises(TrustChanged, match="judge-eval"):
        trusted.verify("reading")


def test_a_link_out_of_the_tree_is_hashed_not_walked(tmp_path, monkeypatch):
    """A link to `/` (or any tree outside) costs its target string, no walk; retargeting still changes it."""
    import os

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "m.py").write_text("a\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "lib").symlink_to(outside)

    from qda_agent.isolation import trusted as trust

    listed: list[Path] = []
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p=".": listed.append(Path(p)) or real(p))
    first = trust.path_digest(eval_dir)
    assert outside not in listed  # the target tree was never read
    assert first == trust.path_digest(eval_dir)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (eval_dir / "lib").unlink()
    (eval_dir / "lib").symlink_to(elsewhere)
    assert trust.path_digest(eval_dir) != first  # retargeting a link out is still a change


def test_a_forged_trust_record_does_not_clear_a_change(tmp_path):
    """`trust.jsonl` is evidence: the digest in memory decides, so an appended line cannot clear a change."""
    f = tmp_path / "judge.json"
    f.write_text("a\n")
    record = tmp_path / "trust.jsonl"
    trusted = Trusted({"judge-config": f}, record=record)
    trusted.take()
    f.write_text("b\n")
    forged = {"event": "taken", "digests": {"judge-config": "sha256:forged"}}
    record.write_text(json.dumps(forged) + "\n")
    with pytest.raises(TrustChanged, match="judge-config"):
        trusted.verify("reading")


def test_a_hub_snapshot_file_link_is_digested(tmp_path):
    """A hub snapshot is links to blobs: rewriting the blob behind a small file is a change."""
    hub = tmp_path / "hub"
    (hub / "blobs").mkdir(parents=True)
    (hub / "snapshots" / "abc").mkdir(parents=True)
    blob = hub / "blobs" / "feed"
    blob.write_text("{}\n")
    (hub / "snapshots" / "abc" / "config.json").symlink_to("../../blobs/feed")
    record = tmp_path / "trust.jsonl"
    trusted = Trusted({"anchor": hub / "snapshots" / "abc"}, record=record)
    trusted.take()
    blob.write_text('{"tampered": true}\n')  # the link is unchanged; the file behind it is not
    with pytest.raises(TrustChanged, match="anchor"):
        trusted.verify("reading")


def test_size_out_counts_a_hardlink_once_and_follows_no_link(tmp_path, no_sandbox_tools):
    """`size_out` walks with `lstat` at `none`: hardlinks count once, links are not followed, `..` is refused."""
    import os

    cfg = config(tmp_path, "isolation.level=none")
    iso = make(cfg)
    source = tmp_path / "work" / "candidate"
    source.mkdir(parents=True)
    (source / "a").write_bytes(b"12345")
    os.link(source / "a", source / "hardlink")
    (source / "distinct").write_bytes(b"12345")
    (tmp_path / "outside").write_bytes(b"x" * 100)
    (source / "symlink").symlink_to(tmp_path / "outside")
    assert iso.size_out(tmp_path / "work", "candidate") == 10
    with pytest.raises(Exception, match="without '\\.\\.'"):
        iso.size_out(tmp_path / "work", "../outside")


def test_a_none_scheduler_job_keeps_the_host_environment(tmp_path, monkeypatch):
    """At `none` a job's environment is the host's with its own on top (and None means inherit), as today.

    Routing `none` to `NoneIsolation` must not narrow the queued job's environment to the spec's own: the
    host canary, the host PATH and the scheduler's CUDA override all have to be there.
    """
    from qda_agent.sched import SchedConfig, Scheduler
    from qda_agent.sched.backend import FakeBackend
    from qda_agent.sched.clock import FakeClock

    monkeypatch.setenv("REVIEW_HOST_ENV", "synthetic-host-value")
    cfg = config(tmp_path, "isolation.level=none")
    clock = FakeClock()
    scheduler = Scheduler(
        tmp_path / "run",
        SchedConfig(cards=[0], gpus=False, job_sandbox="none", budget_s=60),
        FakeBackend(clock),
        clock,
        isolation=make(cfg),
    )
    job = scheduler.submit(["true"], str(tmp_path), name="environment", expected_s=1)
    launch = scheduler._command(job, [])
    env = os.environ if launch.env is None else launch.env
    assert env.get("REVIEW_HOST_ENV") == "synthetic-host-value"
    assert env.get("PATH") == os.environ["PATH"]
    assert env.get("CUDA_VISIBLE_DEVICES") == ""  # the scheduler's override, on top of the host's env
    # a spec that names no environment inherits the caller's
    assert make(cfg).job(JobSpec(argv=["true"], cwd=tmp_path, root=tmp_path)).env is None


def test_a_tampered_judge_refuses_the_final_reading(tmp_path, no_sandbox_tools):
    """The end-of-run check: a judge file changed while the agent ran leaves a FINAL.json that says so."""
    from testkit import ScriptedWorker

    run = tmp_path / "run"

    def tamper(cfg: RunConfig) -> ScriptedWorker:
        def edit() -> str:
            (cfg.run_dir / "judge" / "judge.json").write_text('{"evaluator": "not the judge"}\n')
            return "the judge's file says something else now"

        return ScriptedWorker([edit])

    overrides = ["isolation.level=none", "scheduler.job_sandbox=none"]
    assert scripted(FIXTURES / "scripted-search.yaml", run, 15, script=tamper, overrides=overrides) == 0
    final = json.loads((run / "FINAL.json").read_text())
    assert final["rule"] == "trust", final
    assert "changed" in final["reason"] and "judge-config" in final["reason"]
    # nothing was picked: no choice and no baseline, so no submission in result.json or END-REPORT.md
    assert final["choice"] is None and final["baseline"] is None, final
    result = json.loads((run / "result.json").read_text())
    assert result["rule"] == "trust" and result["choice"] is None and result["baseline"] is None, result
    report = (run / "END-REPORT.md").read_text()
    assert "## Submission:" not in report, (
        report
    )  # nothing selected: no submission line, not even candidate 0
    assert "baselines/" not in report, report
    events = [json.loads(line) for line in (run / "flow.jsonl").read_text().splitlines()]
    assert any(e["ev"] == "trust_refused" for e in events), events[-3:]
