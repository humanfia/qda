"""Build custody: a receipted build's outputs are eligible for the final rules; a hand-edited or mixed set is
scored and reported but never wins."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from qda_agent.judge import build_receipt
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.freeze import digest
from qda_agent.sched import FakeClock, Kind, LocalBackend, SchedConfig, Scheduler, State, sandbox
from qda_agent.sched.sandbox import bwrap_argv, private_args
from test_judge import Rig, ckpt

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


def _trusted(r, name, weights=b"w1", serve=None, source=None, overrides=()):
    """Run `qda build NAME` in the agent route and capture its output as the trusted runner would."""
    before = set(r.host.sched.jobs)
    model = source or r.host.cfg.model
    (r.work / f"{name}.yaml").write_text(f"model: {model}\nout: {name}\n")
    arguments = ["build", f"{name}.yaml"]
    for override in overrides:
        arguments += ["--set", override]
    reply = r.call("qda", *arguments)
    assert reply.exit == 0, reply.stderr
    job = r.host.sched.jobs[(set(r.host.sched.jobs) - before).pop()]
    private = Path(job.build_private)
    output = ckpt(private / "output", weights)
    (output / "build.json").write_text(json.dumps({"serve": serve or {}}))
    spec = json.loads((private / "request.json").read_text()) | {"job_id": job.id}
    build_receipt.capture(private, spec)
    return job, private


def receipted(tmp_path, *, publish=True):
    original = ckpt(tmp_path / "original")
    r = Rig(
        tmp_path,
        model=str(original),
        require_build_receipt=True,
        qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
    )
    r.host.qda_argv = [sys.executable, "-m", "qda_quant.cli"]
    job, private = _trusted(r, "candidate")
    spec = json.loads((private / "request.json").read_text()) | {"job_id": job.id}
    if publish:
        build_receipt.publish(private)
        job.state, job.rc = State.done, 0
    return r, private, spec


def build_again(r, name, weights=b"w2", serve=None):
    """A second receipted build on the same run; return its private dir."""
    job, private = _trusted(r, name, weights, serve)
    build_receipt.publish(private)
    job.state, job.rc = State.done, 0
    return private


def editable(tree: Path) -> Path:
    """`tree` with writable modes, so a test can simulate a hand-edited output."""
    for p in [tree, *tree.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return tree


@pytest.mark.parametrize("change", ["kernels", "delete", "weights", "config"])
def test_an_edited_output_is_scored_but_ineligible_and_never_final(tmp_path, change):
    r, private, _ = receipted(tmp_path)
    copy = editable(shutil.copytree(private / "published", r.work / "edited"))
    if change == "kernels":
        (copy / "build.json").write_text('{"serve":{"kernels":{}}}')
    elif change == "delete":
        (copy / "build.json").unlink()
    elif change == "weights":
        shard = copy / "model.safetensors"
        data = shard.read_bytes()
        shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    else:
        (copy / "config.json").write_text((copy / "config.json").read_text() + "\n")
    assert r.call("kl", "edited").exit == 0  # scored, not refused
    r.settle()
    (e,) = r.host.handins()
    assert e.valid is True and e.eligible is False and e.build_receipt == {}
    assert "not eligible" in e.reason
    final = r.host.final(generation_checks=False)
    assert final.choice is None  # no final rule may pick it
    assert "not eligible" in final.model_dump_json()


def test_a_copy_at_another_path_is_the_builds_outputs(tmp_path):
    """Eligibility is the file set, not the path: a copy or a moved build is the same build."""
    r, private, _ = receipted(tmp_path)
    shutil.copytree(private / "published", r.work / "copy")
    assert r.call("kl", "copy").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid and e.eligible and e.build_receipt["tree_sha256"] == e.digest


def test_a_file_set_mixed_from_two_builds_is_scored_but_ineligible(tmp_path):
    r, first, _ = receipted(tmp_path)
    second = build_again(r, "other", serve={"args": {"dtype": "bfloat16"}})
    mix = r.work / "mix"
    mix.mkdir()
    for name in ("config.json", "build.json"):
        shutil.copyfile(second / "published" / name, mix / name)
    shutil.copyfile(first / "published" / "model.safetensors", mix / "model.safetensors")
    assert r.call("kl", "mix").exit == 0
    r.settle()
    (e,) = r.host.handins()
    assert e.valid is True and e.eligible is False and "not eligible" in e.reason
    assert r.host.final(generation_checks=False).choice is None


def test_a_workdir_shadow_never_replaces_the_receipted_build(tmp_path):
    """What the agent leaves in the workdir at a published destination is ignored: the custody tree is
    scored, so an edit (or a planted copy) in the writable tree changes nothing."""
    r, private, _ = receipted(tmp_path)
    shadow = ckpt(r.work / "candidate", b"evil")
    (shadow / "build.json").write_text('{"serve": {}}')
    assert r.call("kl", "candidate").exit == 0
    e = r.host.entries[0]
    assert e.build_receipt["tree_sha256"] == e.digest
    assert (e.dir / "model.safetensors").read_bytes() == (
        private / "published" / "model.safetensors"
    ).read_bytes()


def test_a_build_from_an_empty_published_destination_refuses_with_the_custody_tree(tmp_path):
    """A follow-on build naming a build's workdir destination (empty under custody) is refused up front:
    the receipted tree is host-only, so the refusal names it instead of failing on a transformers config."""
    r, private, _ = receipted(tmp_path)
    destination = r.work / "candidate"
    destination.mkdir()
    (r.work / "kd.yaml").write_text("model: candidate\nout: kd\n")
    before = set(r.host.sched.jobs)
    reply = r.call("qda", "build", "kd.yaml")
    assert reply.exit != 0 and set(r.host.sched.jobs) == before  # no job is admitted
    assert "candidate is a custody placeholder" in reply.stderr
    assert str(private / "published") in reply.stderr


def test_unreceipted_and_receipted_candidates_have_distinct_admission(tmp_path):
    r, _, _ = receipted(tmp_path)
    ckpt(r.work / "manual")
    assert r.call("kl", "manual").exit == 0  # scored: only the final rules refuse it
    assert r.call("kl", "candidate").exit == 0
    r.settle()
    manual, built = r.host.entries
    assert manual.valid is True and manual.eligible is False and "not eligible" in manual.reason
    assert built.build_receipt["tree_sha256"] == built.digest and built.eligible
    assert r.host.handins()[1].build_receipt == built.build_receipt
    final = r.host.final(generation_checks=False)
    ledger = json.loads(final.model_dump_json())["ledger"]
    assert ledger[0]["build_receipt"] == {} and ledger[1]["build_receipt"] == built.build_receipt


@pytest.mark.parametrize("kind", ["original", "baseline"])
def test_a_build_from_the_original_or_a_baseline_is_eligible(tmp_path, kind):
    """The source is recorded by resolved path and content fingerprint; the original and every configured
    baseline are eligible sources, whatever the output path."""
    r, _, _ = receipted(tmp_path)
    original = Path(r.host.cfg.model)
    source = original
    if kind == "baseline":
        source = ckpt(tmp_path / "baseline", b"base")
        r.host.cfg.baselines = [source]
    job, private = _trusted(r, kind, b"w9", source=source)
    build_receipt.publish(private)
    job.state, job.rc = State.done, 0
    receipt = json.loads((private / "receipt.json").read_text())
    assert receipt["source"] == str(source.resolve())
    assert receipt["source_sha256"] == build_receipt.signature(str(source.resolve()))
    assert r.call("kl", kind).exit == 0
    r.settle()
    e = r.host.entries[-1]
    assert e.valid is True and e.eligible is True, e
    r.host.cfg.final_rules = JudgeSettings(final_rules=["best_metric"]).final_rules  # picks a scored build
    assert r.host.final(generation_checks=False).choice.dir == e.dir


@pytest.mark.parametrize("kind", ["patched", "copy", "override"])
def test_a_build_from_a_non_original_source_is_scored_but_ineligible(tmp_path, kind):
    """An agent-written copy of the original (byte-equal, patched, or named by `--set model=`) is scored and
    reported, but no final rule may pick its output."""
    r, _, _ = receipted(tmp_path)
    original = Path(r.host.cfg.model)
    source, overrides = editable(shutil.copytree(original, tmp_path / "patched")), ()
    if kind != "copy":
        shard = source / "model.safetensors"
        data = shard.read_bytes()
        shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    if kind == "override":
        overrides, source = (f"model={source}",), original
    job, private = _trusted(r, "non-original", b"w9", source=source, overrides=overrides)
    build_receipt.publish(private)
    job.state, job.rc = State.done, 0
    assert r.call("kl", "non-original").exit == 0
    r.settle()
    e = r.host.entries[-1]
    assert e.valid is True and e.eligible is False, e
    assert "non-original source" in e.reason and "eligible" in e.reason
    assert r.host.final(generation_checks=False).choice is None


def test_a_configured_baseline_read_through_a_bind_is_eligible(tmp_path):
    """A model named at a judge bind's mount path is the bind's backing directory: that is the receipt's
    source and fingerprint, and the build is handed the mount path it can read inside its sandbox.

    The stock run binds the overlay parent (`baselines`) before the baseline (`baselines/0`), and the
    deeper bind is the one whose bytes are read: the source must not be the overlay's mount-point directory.
    """
    r, _, _ = receipted(tmp_path)
    baseline = ckpt(tmp_path / "baseline", b"base")
    r.host.cfg.baselines = [baseline]
    overlay = tmp_path / "overlay" / "baselines"
    (overlay / "0").mkdir(parents=True)  # the installed mount point, not the baseline
    r.host.sched.cfg.binds[r.work / "baselines"] = overlay
    mounted = r.work / "baselines" / "0"
    r.host.sched.cfg.binds[mounted] = baseline
    job, private = _trusted(r, "bound", b"w7", source=mounted)
    build_receipt.publish(private)
    job.state, job.rc = State.done, 0
    receipt = json.loads((private / "receipt.json").read_text())
    assert receipt["source"] == str(baseline.resolve())
    assert receipt["source_sha256"] == build_receipt.signature(str(baseline.resolve()))
    assert r.call("kl", "bound").exit == 0
    r.settle()
    e = r.host.entries[-1]
    assert e.valid is True and e.eligible is True, e
    r.host.cfg.final_rules = JudgeSettings(final_rules=["best_metric"]).final_rules  # picks a scored build
    assert r.host.final(generation_checks=False).choice.dir == e.dir


def test_the_recorded_source_fingerprint_covers_weight_values(tmp_path):
    """A payload byte changed between two builds of the same path is a different source, though its
    safetensors header (the cheap file signature) is unchanged."""
    r, _, _ = receipted(tmp_path)
    source = editable(Path(r.host.cfg.model))
    first = json.loads((_trusted(r, "first", b"w9", source=source)[1] / "receipt.json").read_text())
    shard = source / "model.safetensors"
    data = shard.read_bytes()
    shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    second = json.loads((_trusted(r, "second", b"w9", source=source)[1] / "receipt.json").read_text())
    assert first["source_sha256"] != second["source_sha256"]
    assert first["source"] == second["source"] == str(source.resolve())


def test_capture_renames_and_refuses_a_hardlinked_output(tmp_path):
    r, private, spec = receipted(tmp_path, publish=False)
    build_receipt.remove_tree(private / "frozen")
    output = ckpt(private / "output")
    (output / "build.json").write_text(json.dumps({"serve": {}}))
    # Another writer could change these bytes after the receipt: refuse a hardlink before capture.
    link = r.work / "retained"
    os.link(output / "model.safetensors", link)
    with pytest.raises(ValueError, match="hardlink"):
        build_receipt.capture(private, spec)
    assert not (private / "frozen").exists()
    link.unlink()
    before = (output / "model.safetensors").stat()
    build_receipt.capture(private, spec)
    after = (private / "frozen" / "model.safetensors").stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)  # one physical copy: renamed
    assert not (private / "output").exists()
    assert not (private / "published").exists()
    assert not (r.work / "candidate").exists()  # nothing was ever written into the agent's workdir
    build_receipt.publish(private)
    assert (private / "published" / "build.json").is_file()
    assert not (r.work / "candidate").exists()
    job = r.host.sched.get("job1")
    job.state, job.rc = State.done, 0
    assert r.call("kl", "candidate").exit == 0


def test_a_planted_link_at_a_published_destination_is_refused_not_followed(tmp_path):
    """A link the agent leaves where a bind would go is refused: the mount is dropped, and the host neither
    follows nor removes the link."""
    from qda_agent.sched.sandbox import mount_points, published_entries

    r, private, _ = receipted(tmp_path)
    planted = r.work / "candidate"
    planted.symlink_to(private, target_is_directory=True)
    entries = dict(published_entries(r.run, r.work))
    assert entries == {r.work / "candidate": private / "published"}
    assert mount_points(entries, [r.work], deep=entries) == {}
    assert planted.is_symlink() and planted.readlink() == private
    assert not (private / "candidate").exists()  # nothing was created through the link


def test_a_nested_published_destination_gets_real_ancestors(tmp_path):
    """A build at `out/deep/candidate` is mounted through parents the host makes: missing directories only,
    never through a link."""
    from qda_agent.sched.sandbox import mount_points, published_entries

    r, private, _ = receipted(tmp_path)
    receipt = json.loads((private / "receipt.json").read_text())
    receipt["destination"] = "out/deep/candidate"
    (private / "receipt.json").write_text(json.dumps(receipt))
    entries = dict(published_entries(r.run, r.work))
    assert entries == {r.work / "out" / "deep" / "candidate": private / "published"}
    assert mount_points(entries, [r.work], deep=entries) == entries
    for made in (r.work / "out", r.work / "out" / "deep", r.work / "out" / "deep" / "candidate"):
        assert made.is_dir() and not made.is_symlink(), made
    assert not (private / "deep").exists()


def test_a_link_under_a_nested_published_destination_refuses_the_bind(tmp_path):
    """A link in the ancestor chain is refused too: the host drops the bind without touching the chain."""
    from qda_agent.sched.sandbox import mount_points, published_entries

    r, private, _ = receipted(tmp_path)
    receipt = json.loads((private / "receipt.json").read_text())
    receipt["destination"] = "out/deep/candidate"
    (private / "receipt.json").write_text(json.dumps(receipt))
    (r.work / "out").symlink_to(private, target_is_directory=True)
    entries = dict(published_entries(r.run, r.work))
    assert entries == {r.work / "out" / "deep" / "candidate": private / "published"}
    assert mount_points(entries, [r.work], deep=entries) == {}
    assert (r.work / "out").is_symlink() and (r.work / "out").readlink() == private
    assert not (private / "deep").exists()


def test_an_ancestor_swapped_after_the_walk_refuses_the_bind_and_deletes_nothing(tmp_path, monkeypatch):
    """The host prepares mount points with directory descriptors, so a link planted between the ancestor walk
    and the final step can make it neither delete a host file nor follow the link: the bind is refused."""
    from qda_agent.sched import sandbox

    work = tmp_path / "work"
    work.mkdir()
    host_only = tmp_path / "host-only"
    host_only.mkdir()
    (host_only / "keep.txt").write_text("host data\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_text("weights\n")
    real_open = os.open
    swapped = []

    def open_then_swap(path, flags, mode=0o777, *, dir_fd=None):
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and path == "nested" and flags & os.O_DIRECTORY and not swapped:
            # the agent wins the race right after the host opened the ancestor it had just made
            (work / "nested").rename(work / "nested-old")
            (work / "nested").symlink_to(host_only, target_is_directory=True)
            swapped.append(True)
        return fd

    monkeypatch.setattr(sandbox.os, "open", open_then_swap)
    binds = {work / "nested" / "model": source}
    assert sandbox.mount_points(binds, [work], deep=binds) == {}
    assert swapped == [True]
    assert (host_only / "keep.txt").read_text() == "host data\n"  # the host file survives
    assert sorted(p.name for p in host_only.iterdir()) == ["keep.txt"]  # not replaced by a directory
    assert not (host_only / "model").exists()  # nothing was created through the link
    assert (work / "nested").is_symlink() and (work / "nested").readlink() == host_only
    assert (work / "nested-old" / "model").is_dir()  # the host made a point only in what it created


def test_no_agent_writable_manifest_can_redirect_a_publication(tmp_path):
    """There is no manifest: the destinations are re-derived from the receipts under the host-only judge
    directory at every launch, so a list the agent writes anywhere changes nothing."""
    from qda_agent.sched.sandbox import published_entries

    r, private, _ = receipted(tmp_path)
    evil = ckpt(tmp_path / "evil")
    (r.run / "builds-readonly.json").write_text(json.dumps({"candidate": str(evil)}))
    (r.run / "published.json").write_text(json.dumps({str(r.work / "candidate"): str(evil)}))
    (r.host.jdir / "builds-readonly.json").write_text(json.dumps({"candidate": str(evil)}))
    assert dict(published_entries(r.run, r.work)) == {r.work / "candidate": private / "published"}
    assert r.call("kl", "candidate").exit == 0
    assert r.host.entries[0].build_receipt["tree_sha256"] == r.host.entries[0].digest


@pytest.mark.parametrize("destination", ["../escape", "/etc", ""])
def test_a_receipt_with_a_destination_outside_the_workdir_is_ignored(tmp_path, destination):
    from qda_agent.sched.sandbox import published_entries

    r, private, _ = receipted(tmp_path)
    receipt = json.loads((private / "receipt.json").read_text())
    receipt["destination"] = destination
    (private / "receipt.json").write_text(json.dumps(receipt))
    assert published_entries(r.run, r.work) == []


def test_trusted_request_confines_output_and_does_not_follow_config_links(tmp_path):
    r, _, _ = receipted(tmp_path)
    (r.work / "escape.yaml").write_text("out: ../../outside\n")
    assert r.call("qda", "build", "escape.yaml").exit == 2
    (r.work / "link.yaml").symlink_to(r.work / "recipe.yaml")
    assert r.call("qda", "build", "link.yaml").exit == 2


def test_custody_is_unreadable_in_an_actual_ordinary_job_sandbox(tmp_path):
    r, private, _ = receipted(tmp_path)
    script = "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())"
    argv = bwrap_argv(
        ["/usr/bin/python3", "-I", "-c", script, str(private / "receipt.json")],
        str(r.work),
        writable=[r.work],
        hide=r.host.sched.cfg.hide,
        pid="private",
    )
    with private_args(argv) as (argv, fds):
        result = subprocess.run(argv, capture_output=True, text=True, pass_fds=fds)
    assert result.returncode != 0
    assert "tree_sha256" not in result.stdout


def _ref_dir(root):
    """An eval+holdout reference: the judge host derives its calibration index from configured references."""
    d = Path(root) / "receipt-ref"
    d.mkdir(parents=True, exist_ok=True)
    prompts = d / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "r1", "prompt": "a reference prompt used by the judge tests only"})
        + "\n"
        + json.dumps(
            {"id": "h1", "prompt": "a holdout prompt used by the judge tests only", "split": "holdout"}
        )
        + "\n"
    )
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return d


def _index_file(root):
    """A prebuilt index equal to the canonical union of `_ref_dir`'s eval and holdout sets."""
    from qda_quant.calib.contamination import PromptIndex

    return PromptIndex.read_reference(_ref_dir(root)).save(Path(root) / "contamination-index.json")


OPS = r"""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / "candidate" / "model.safetensors"

def attempt(fn):
    try:
        fn()
    except OSError as e:
        return type(e).__name__
    return "ok"

def copy():
    (root / "copied").mkdir(exist_ok=True)
    (root / "copied" / "model.safetensors").write_bytes(target.read_bytes())

res = {}
res["read"] = attempt(lambda: target.read_bytes())
res["copy"] = attempt(copy)
res["alias_read"] = attempt(lambda: (root / "alias" / "candidate" / "model.safetensors").read_bytes())
res["write"] = attempt(lambda: target.write_bytes(b"x"))
res["append"] = attempt(lambda: target.open("ab").write(b"x"))
res["truncate"] = attempt(lambda: os.truncate(target, 0))
res["chmod"] = attempt(lambda: os.chmod(target, 0o644))
res["rename"] = attempt(lambda: os.rename(target, target.with_name("moved")))
res["unlink"] = attempt(lambda: os.unlink(target))
res["replace"] = attempt(lambda: os.replace(root / "mine" / "model.safetensors", target))
res["alias_write"] = attempt(lambda: (root / "alias" / "candidate" / "model.safetensors").write_bytes(b"x"))
print("PROBE " + json.dumps(res))
"""

WRITES = ("write", "append", "truncate", "chmod", "rename", "unlink", "replace", "alias_write")


def custody(run: Path, work: Path, rel: str = "candidate") -> Path:
    """The host-only published tree a receipted build leaves, with its receipt."""
    private = run / "judge" / "builds" / "job1"
    tree = private / "published"
    tree.mkdir(parents=True)
    (tree / "model.safetensors").write_bytes(b"w1")
    (tree / "config.json").write_text("{}")
    (tree / "build.json").write_text('{"serve": {}}')
    (private / "receipt.json").write_text(
        json.dumps({"job_id": "job1", "destination": rel, "workdir": str(work), "tree_sha256": "0" * 64})
    )
    for p in tree.rglob("*"):
        p.chmod(0o444)
    tree.chmod(0o555)
    return private


def prepare_work(work: Path) -> None:
    """The writable fixtures an agent may have: a copy of its own and an alias to the workdir root."""
    ckpt(work / "mine")
    (work / "alias").symlink_to(".", target_is_directory=True)


def check_reads_and_denied_writes(res: dict[str, str]) -> None:
    assert res["read"] == "ok", res  # reads (and a copy elsewhere) work before every denied write
    assert res["copy"] == "ok", res
    assert res["alias_read"] == "ok", res
    for op in WRITES:
        assert res[op] != "ok", (op, res)


def run_job(run: Path, work: Path, code: str) -> tuple[dict[str, str], str]:
    s = Scheduler(
        run, SchedConfig(cards=[0], budget_s=600, gpus=False), LocalBackend(1), FakeClock(time.time())
    )
    s.submit(
        [sandbox.python(), "-c", code, str(work)],
        str(work),
        name="h",
        expected_s=60,
        root=work,
        kind=Kind.build,
    )
    deadline = time.time() + 30
    while s.pending():
        assert time.time() < deadline, s.log("h")
        s.tick()
        time.sleep(0.05)
    log = s.log("h")
    assert s.get("h").state is State.done, log
    line = next(x for x in log.splitlines() if x.startswith("PROBE "))
    return json.loads(line.split(" ", 1)[1]), log


@needs_bwrap
def test_a_receipted_output_is_read_only_in_an_agent_job(tmp_path):
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    private = custody(run, work)
    prepare_work(work)
    res, log = run_job(run, work, OPS)
    check_reads_and_denied_writes(res)
    assert (private / "published" / "model.safetensors").read_bytes() == b"w1", log


@needs_bwrap
def test_a_receipted_output_is_read_only_in_the_qda_verb(tmp_path):
    r, private, _ = receipted(tmp_path)
    prepare_work(r.work)
    before = (private / "published" / "model.safetensors").read_bytes()
    r.host.qda_argv = [sys.executable, "-c", OPS]
    reply = r.call("qda", str(r.work))
    assert reply.exit == 0, reply.stderr
    line = next(x for x in reply.stdout.splitlines() if x.startswith("PROBE "))
    check_reads_and_denied_writes(json.loads(line.split(" ", 1)[1]))
    assert (private / "published" / "model.safetensors").read_bytes() == before


@needs_bwrap
def test_a_job_started_before_publication_never_sees_the_build(tmp_path):
    """A process already running in its old namespace cannot see the build published after it started:
    every path below the destination is absent there, so even a write fails."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    prepare_work(work)
    started, go = work / "started", work / "go"
    code = (
        f"import pathlib,time\npathlib.Path({str(started)!r}).write_text('x')\n"
        f"while not pathlib.Path({str(go)!r}).exists(): time.sleep(0.02)\n" + OPS
    )
    s = Scheduler(
        run, SchedConfig(cards=[0], budget_s=600, gpus=False), LocalBackend(1), FakeClock(time.time())
    )
    s.submit(
        [sandbox.python(), "-c", code, str(work)],
        str(work),
        name="h",
        expected_s=60,
        root=work,
        kind=Kind.build,
    )
    deadline = time.time() + 30
    while not started.exists():
        assert time.time() < deadline, s.log("h")
        s.tick()
        time.sleep(0.02)
    custody(run, work)  # published while the job's sandbox is already running
    go.write_text("x")
    while s.pending():
        assert time.time() < deadline, s.log("h")
        s.tick()
        time.sleep(0.05)
    log = s.log("h")
    assert s.get("h").state is State.done, log
    res = json.loads(next(x for x in log.splitlines() if x.startswith("PROBE ")).split(" ", 1)[1])
    for op, outcome in res.items():
        assert outcome != "ok", (op, res)
    assert not (work / "candidate").exists()


def test_full_card_moe_requires_build_completion_then_scoring_before_next_build(tmp_path):
    r, private, _ = receipted(tmp_path)
    r.host.sched.cfg.cards = list(range(8))
    build = r.host.sched.get("job1")
    r.host.cfg.full_card_moe = True
    r.host.full_card_build = build.name
    build.state = State.queued
    waiting = r.call("kl", "candidate")
    assert waiting.exit == 2 and "wait for build" in waiting.stderr
    build.state, build.rc = State.done, 0
    unrelated = r.call("kl", "other")
    assert unrelated.exit == 2 and "score candidate" in unrelated.stderr
    assert r.host.full_card_build == build.name and not r.host.entries
    unscored = r.call("qda", "build", "candidate.yaml")
    assert unscored.exit == 2 and "before another build" in unscored.stderr
    narrow = r.call("qda", "--cards", "1", "build", "candidate.yaml")
    assert narrow.exit == 2 and "all 8 cards" in narrow.stderr
    assert r.call("kl", "candidate").exit == 0
    assert r.host.full_card_build is None
    next_build = r.call("qda", "build", "candidate.yaml")
    assert next_build.exit == 2 and "score" in next_build.stderr
    r.settle()
    (r.work / "next.yaml").write_text(f"model: {r.host.cfg.model}\nout: next\n")
    assert r.call("qda", "build", "next.yaml").exit == 0
    assert r.host.sched.get("job3").ncards == 8
    assert r.host.full_card_build == "job3"
    assert (private / "published").is_dir()


def test_trusted_runner_really_builds_before_receipting_and_publishing(tmp_path):
    from datetime import UTC, datetime, timedelta

    import torch
    import yaml
    from qda_agent.judge import JudgeConfig, JudgeHost
    from qda_agent.rpc import Request
    from qda_agent.sched import SchedConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

    model = tmp_path / "model"
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
    )
    AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"a": 0, "<unk>": 1}, unk_token="<unk>"))
    )
    tokenizer.save_pretrained(model)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "a a a a a a a a a a a a a a a a"}) + "\n")
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    recipe = {
        "model": str(model),
        "format": "w4a16",
        "target": "h100",
        "devices": ["cpu"],
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 16}},
        "out": "candidate",
    }
    (work / "build.yaml").write_text(yaml.safe_dump(recipe))
    from qda_agent import __file__ as agent_file
    from qda_kl import __file__ as kl_file
    from qda_quant import __file__ as quant_file

    paths = [Path(p).parents[1] for p in (quant_file, kl_file, agent_file)]
    host = JudgeHost(
        run,
        SchedConfig(
            cards=[0],
            gpus=False,
            deadline=datetime.now(UTC) + timedelta(minutes=2),
            readable=[model, calib, *paths],
        ),
        JudgeConfig(
            contamination="none",
            kl_ref=str(_ref_dir(run.parent)),
            kl_holdout_ref=str(_ref_dir(run.parent)),
            workdir=work,
            model=str(model),
            kl_cmd=["true"],
            allow_any_format=True,
            require_build_receipt=True,
            qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
            qda_env={"PYTHONPATH": os.pathsep.join(map(str, paths))},
        ),
        autotick=False,
    )
    host.qda_argv = [sys.executable, "-m", "qda_quant.cli"]
    reply = host.handle(Request("qda", ["--expected", "1m", "build", "build.yaml"], str(work)))
    assert reply.exit == 0, reply.stderr
    job = host.sched.get("job1")
    command = build_receipt.command(host.sched, job, [0])
    subprocess.run(command, check=True, timeout=90)
    job.state, job.rc = State.done, 0
    published = Path(job.build_private) / "published"
    receipt, why = build_receipt.match(host, published, digest(published))
    assert why == "" and receipt is not None
    assert receipt["job_id"] == job.id
    assert receipt["workdir"] == str(work)
    assert receipt["source"] == str(model.resolve())  # the real build recorded its own resolved source
    assert (published / "build.json").is_file()
    assert not (work / "candidate").exists()  # the build never lands in the agent's workdir
    assert not (Path(job.build_private) / "output").exists()


@pytest.mark.parametrize("devices", [None, [5]])
def test_the_judges_build_plan_touches_no_card(tmp_path, monkeypatch, devices):
    """The judge prices a build in its own process, which holds no card: planning must call no CUDA API.

    A plan on the cards read each one's free memory (`torch.cuda.mem_get_info`), which opened a ~0.5 GiB
    CUDA context in the judge on every card it could see, other runs' cards included. The machine here
    claims eight cards, and every per-card query fails the test.
    """
    import torch
    from qda_agent.judge.startup import tiny_model

    model = tiny_model(tmp_path / "model")
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": " ".join(f"w{i}" for i in range(16))}) + "\n")
    raw = {
        "model": str(model),
        "format": "w4a16",
        "out": "candidate",
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 16}},
    }
    if devices is not None:
        raw["devices"] = devices
    touched = []

    def card_query(name):
        return lambda *args, **kwargs: touched.append((name, args)) or pytest.fail(f"torch.cuda.{name}{args}")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    for name in ("mem_get_info", "get_device_capability", "get_device_properties", "init", "set_device"):
        monkeypatch.setattr(torch.cuda, name, card_query(name))
    assert build_receipt._plan_bytes(raw, str(tmp_path))  # the plan is read, and it prices the output
    assert touched == []


def test_the_judges_build_plan_plans_the_runs_cards_when_the_recipe_names_no_target(tmp_path):
    """A W4A4 recipe with no `target:` prices on the run's cards' table, and a recipe's own target still wins."""
    from qda_agent.judge.startup import tiny_model
    from qda_quant.target import Target

    model = tiny_model(tmp_path / "model")
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": " ".join(f"w{i}" for i in range(16))}) + "\n")
    raw = {
        "model": str(model),
        "format": "w4a4-nvfp4",
        "out": "candidate",
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 16}},
    }
    assert build_receipt._build_plan(raw, str(tmp_path)) is None, "the default Target() is sm90, no W4A4"
    plan = build_receipt._build_plan(raw, str(tmp_path), Target.parse("sm103"))
    assert plan is not None and plan.plan_dense.kernel == "FlashInferCuteDslNvFp4LinearKernel"
    named = build_receipt._build_plan({**raw, "target": "sm100"}, str(tmp_path), Target.parse("sm103"))
    assert named is not None, "a recipe's own target wins over the run's"


def test_the_run_target_is_the_schedulers_cards(monkeypatch):
    from types import SimpleNamespace

    from qda_agent import group_size
    from qda_quant.target import Target

    seen = []
    monkeypatch.setattr(group_size, "run_target", lambda cards: seen.append(cards) or Target.parse("sm103"))
    host = SimpleNamespace(sched=SimpleNamespace(cfg=SimpleNamespace(cards=[2, 3])))
    assert build_receipt._run_target(host).arch == "sm103"
    assert seen == [[2, 3]]
