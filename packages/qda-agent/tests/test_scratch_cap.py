"""A scratch cap limits work and job growth independently of local custody weights."""

import json

import pytest
from qda_agent.judge import build_receipt, prune
from qda_agent.sched import State
from test_custody_space import local_build
from test_judge import Rig


def test_local_build_reserves_records_on_scratch_and_weights_locally(tmp_path):
    r, _ = local_build(tmp_path)
    r.host.cfg.job_reserve_bytes = 1024
    held = prune.run_bytes(r.host, scratch=True)
    r.host.cfg.max_scratch_bytes = held + 8192
    (r.work / "next.yaml").write_text(f"model: {r.host.cfg.model}\nout: next\n")
    private, spec = build_receipt.setup(r.host, r.work, ["next.yaml"])
    (private / "request.json").write_text(json.dumps(spec))
    assert spec["expected_bytes"] > 8192  # weight reservation exceeds the separate scratch cap
    assert prune.reserved_bytes(r.host, scratch=True) == 1024
    assert prune.reserved_bytes(r.host) == spec["expected_bytes"]
    (r.work / "copy.bin").write_bytes(b"x" * 16384)
    assert prune.over_cap(r.host).startswith("scratch disk budget:")
    with pytest.raises(prune.Refused, match="scratch disk budget:"):
        prune.check_room(r.host)


def test_scratch_only_cap_refuses_admission_and_late_job_start(tmp_path):
    r = Rig(tmp_path, max_scratch_bytes=1 << 20, job_reserve_bytes=1024)
    (r.work / "job.sh").write_text("true\n")
    assert r.call("qsched", "add", "job.sh", "--expected", "10s").exit == 0
    (r.work / "overflow.bin").write_bytes(b"x" * (2 << 20))
    r.host.tick()
    job = next(iter(r.host.sched.jobs.values()))
    assert job.state is State.failed and "scratch disk budget:" in job.reason
    assert r.call("qsched", "add", "job.sh", "--expected", "10s").exit == 2


def test_scratch_bytes_include_records_and_winner_but_exclude_local_weights(tmp_path):
    r, private = local_build(tmp_path)
    before = prune.run_bytes(r.host, scratch=True)
    (private / "payload").write_bytes(b"x" * 8192)
    assert prune.run_bytes(r.host, scratch=True) == before
    for name in ("judge/records/record", "winner/model.safetensors", "jobs/log"):
        target = r.run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 4096)
    assert prune.run_bytes(r.host, scratch=True) == before + 3 * 4096
    r.host.cfg.max_scratch_bytes = before + 3 * 4096
    with pytest.raises(prune.Refused, match="scratch disk budget:"):
        prune.check_room(r.host, 1)


def test_a_trusted_build_request_with_jobs_on_the_host_needs_no_sandbox_unless_receipted(tmp_path):
    """The trial run's candidate 0 goes through `setup` at every level; only receipt custody needs bwrap jobs."""
    r, _ = local_build(tmp_path)
    r.host.sched.cfg = r.host.sched.cfg.model_copy(update={"job_sandbox": "none"})  # isolation.level: none
    (r.work / "next.yaml").write_text(f"model: {r.host.cfg.model}\nout: next\n")
    r.host.cfg.require_build_receipt = False
    private, _ = build_receipt.setup(r.host, r.work, ["next.yaml"])
    assert private.is_dir()
    r.host.cfg.require_build_receipt = True
    with pytest.raises(ValueError, match="fix: set isolation.level: sandbox or strict"):
        build_receipt.setup(r.host, r.work, ["next.yaml"])
