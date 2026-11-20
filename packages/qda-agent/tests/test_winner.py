"""Winner export is verified and durable; custody failures retain their records and sources."""

import json
import shutil
from pathlib import Path

import pytest
from qda_agent.host import Final
from qda_agent.judge import build_receipt
from qda_agent.judge.final_rules import export
from qda_agent.judge.freeze import digest
from test_build_receipt import _trusted, editable
from test_judge import Rig, ckpt


def local_build(tmp_path):
    original = ckpt(tmp_path / "original")
    r = Rig(
        tmp_path,
        custody_dir=tmp_path / "local",
        max_run_bytes=1 << 30,
        max_copy_bytes=1 << 20,
        require_build_receipt=True,
        model=str(original),
        final_rules=["best_metric"],  # a scored build is the pick here (the default rule takes hand-ins)
    )
    r.host.qda_argv = ["/usr/bin/python3", "-m", "qda_quant.cli"]
    job, private = _trusted(r, "candidate")
    build_receipt.publish(private)
    job.state, job.rc = type(job.state).done, 0
    assert r.call("kl", "candidate").exit == 0
    r.settle()
    return r, private


def test_node_loss_keeps_unscored_build_and_entry_records(tmp_path):
    r, private = local_build(tmp_path)
    # A build never handed in must have records too, before any finalization.
    _, unscored = _trusted(r, "unscored")
    records = r.host.jdir / "records"
    for path in (private, unscored):
        durable = records / "builds" / path.name
        assert json.loads((durable / "receipt.json").read_text())["tree_sha256"]
        assert (durable / "request.json").is_file()
        assert (durable / "build.json").is_file()
        assert list(durable.glob("*.yaml"))
    assert (records / "frozen" / "k1" / "build.json").is_file()
    before = (r.host.jdir / "ledger.jsonl").read_bytes()
    editable(tmp_path / "local")
    shutil.rmtree(tmp_path / "local")
    assert (r.host.jdir / "ledger.jsonl").read_bytes() == before
    assert (records / "builds" / unscored.name / "receipt.json").is_file()


@pytest.mark.parametrize("failure", [None, "copy", "tampered", "link", "exists"])
def test_export_verifies_winner_and_keeps_custody_on_failure(tmp_path, monkeypatch, failure):
    r, private = local_build(tmp_path)
    final = r.host.final(generation_checks=False)
    expected = r.host.entries[0].digest
    if failure == "copy":

        def no_space(*args, **kwargs):
            raise OSError("no space left")

        monkeypatch.setattr(shutil, "copytree", no_space)
    elif failure == "tampered":
        (editable(final.choice.dir) / "build.json").write_text("{}")
    elif failure == "link":
        (editable(final.choice.dir) / "foreign").symlink_to(tmp_path / "original", target_is_directory=True)
    elif failure == "exists":
        (r.run / "winner").mkdir()
        (r.run / "winner" / "keep").write_text("untouched")
    export(r.host, final)
    assert (private / "published" / "model.safetensors").is_file()
    assert not list(r.run.glob(".winner-*"))
    if failure:
        assert final.export["winner"] is None and final.export["error"]
        assert "local custody retained" in final.reason
    else:
        assert final.choice.dir == r.run / "winner"
        assert digest(final.choice.dir) == expected == final.export["sha256"]
        assert final.choice.dir.stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    ("cap", "refusal"), [("max_run_bytes", "^disk budget"), ("max_scratch_bytes", "^scratch disk")]
)
def test_export_ignores_run_caps_already_exceeded(tmp_path, cap, refusal):
    from qda_agent.judge import prune

    r, _ = local_build(tmp_path)
    final = r.host.final(generation_checks=False)
    r.host.cfg.max_run_bytes = r.host.cfg.max_scratch_bytes = None
    setattr(r.host.cfg, cap, 1)  # one cap alone, already passed before the export starts
    assert prune.over_cap(r.host)
    export(r.host, final)
    assert final.export["winner"] == str(r.run / "winner"), final.export
    assert digest(final.choice.dir) == r.host.entries[0].digest
    with pytest.raises(prune.Refused, match=refusal):
        prune.check_room(r.host)  # build and job admission still refuse at that cap


@pytest.mark.parametrize("tampered", [False, True])
def test_baseline_export_materializes_trusted_cache_links_and_checks_metadata(tmp_path, tampered):
    baseline = ckpt(tmp_path / "cache" / "snapshots" / "release")
    blobs = tmp_path / "cache" / "blobs"
    blobs.mkdir()
    shard = baseline / "model.safetensors"
    shard.rename(blobs / "weights")
    shard.symlink_to(blobs / "weights")
    r = Rig(tmp_path, custody_dir=tmp_path / "local", max_run_bytes=1 << 30, baselines=[baseline])
    final = Final(rule="baseline", baseline=baseline)
    if tampered:
        (baseline / "config.json").write_text("{}")
    export(r.host, final)
    if tampered:
        assert final.export["winner"] is None
        return
    assert final.baseline == r.run / "winner", final.export
    assert final.export["source"] == str(baseline)  # the copy is not read: the pick's reading is the source's
    assert not (final.baseline / "model.safetensors").is_symlink()


def test_final_records_share_the_durable_pick_after_shutdown(tmp_path):
    r, _ = local_build(tmp_path)
    r.host.cutoff()
    r.host.write_final(r.run / "FINAL.json")
    saved = json.loads((r.run / "FINAL.json").read_text())
    assert saved["choice"]["dir"] == str(r.run / "winner")
    assert r.host.final().choice.dir == Path(saved["export"]["winner"])
    assert r.host.sched.closed


@pytest.mark.asyncio
async def test_flow_returns_the_exported_pick(tmp_path):
    from types import SimpleNamespace

    from qda_agent.flow import _finish

    r, _ = local_build(tmp_path)
    cfg = SimpleNamespace(judge=r.host.cfg, run_dir=r.run, drain_window_s=lambda: 900)
    final, stopped = await _finish(r.host, cfg, lambda *args, **kwargs: None)
    assert final.choice.dir == r.run / "winner" and final.export["winner"] == str(r.run / "winner")
    assert not stopped
