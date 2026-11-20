"""Both storage roots are charged once, and local launch bounds fail before custody is created."""

import json
import os
from types import SimpleNamespace

import pytest
from qda_agent.judge import build_receipt, prune
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.custody import check_space, required_bytes
from qda_agent.sched import State
from test_build_receipt import _trusted
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
    )
    r.host.qda_argv = ["/usr/bin/python3", "-m", "qda_quant.cli"]
    job, private = _trusted(r, "candidate")
    build_receipt.publish(private)
    job.state, job.rc = type(job.state).done, 0
    assert r.call("kl", "candidate").exit == 0
    r.settle()
    return r, private


@pytest.mark.parametrize("resource", ["bytes", "inodes", "enough"])
def test_local_launch_checks_bytes_and_inode_peak(tmp_path, monkeypatch, resource):
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=99 if resource == "bytes" else 100,
            f_frsize=1,
            f_favail=49 if resource == "inodes" else 60,
            f_blocks=100,
            f_files=60,
        ),
    )
    # the cap and the inode peak alone: the free-space floor has its own tests below
    judge = JudgeSettings(custody_dir=tmp_path / "local", max_run_bytes=100, max_files=10, min_free_bytes=0)
    if resource == "enough":
        check_space(tmp_path / "local", judge, 2)
    elif resource == "bytes":  # the cap is above the 99 bytes free: one check, one message
        with pytest.raises(ValueError, match=r"above the .* free on"):
            check_space(tmp_path / "local", judge, 2)
    else:
        with pytest.raises(ValueError, match=r"49 free inodes.*50 the run needs"):
            check_space(tmp_path / "local", judge, 2)
    assert JudgeSettings(custody_dir=tmp_path / "local").max_run_bytes is None  # no default cap
    if resource != "enough":
        with pytest.raises(ValueError, match="judge.custody_dir"):
            Rig(
                tmp_path,
                custody_dir=tmp_path / "local",
                max_run_bytes=100,
                max_files=10,
                min_free_bytes=0,
            )
        assert not (tmp_path / "local").exists()


def test_local_weights_cap_and_hardlinks_are_counted_once(tmp_path):
    r, private = local_build(tmp_path)
    before = prune.run_bytes(r.host)
    weight = private / "published" / "model.safetensors"
    os.link(weight, r.work / "another-name")
    assert prune.run_bytes(r.host) == before
    # Custody is reached by directory links from judge; its bytes still count exactly once.
    payload = tmp_path / "local" / "payload"
    payload.write_bytes(b"x" * 8192)
    assert prune.run_bytes(r.host) == before + 8192
    r.host.cfg.max_run_bytes = before + 8191
    assert "disk budget:" in prune.over_cap(r.host)
    payload.unlink()
    r.host.cfg.max_run_bytes = before + 1
    with pytest.raises(prune.Refused, match="disk budget:"):
        prune.check_room(r.host, 2)


def test_local_pending_build_reservation_counts_with_both_roots(tmp_path):
    r, _ = local_build(tmp_path)
    job, _ = _trusted(r, "pending")
    job.state = type(job.state).queued
    reserved = prune.reserved_bytes(r.host)
    assert reserved > 0
    (tmp_path / "local" / "payload").write_bytes(b"x" * 8192)
    held = prune.run_bytes(r.host)
    r.host.cfg.max_run_bytes = held + reserved - 1
    with pytest.raises(prune.Refused, match="queued jobs"):
        prune.check_room(r.host)


def test_required_bytes_is_the_live_copies_and_working_trees_summed():
    """One owner computes what the run must hold at once. A 6.11 GB build with 1 card needs the
    hand-in's own frozen copy + 1 card working copy + 1 in flight = 3 builds. A build handed in stays whole
    for the rest of the run, so this is a floor, not a promise of how many hand-ins fit."""
    build = 6_110_000_000
    assert required_bytes(build, cards=1) == 3 * build == 18_330_000_000
    assert required_bytes(build, cards=2) == 4 * build  # one working copy per scheduler card
    # a cutoff rule that reads the top 3 by KL adds their frozen copies (prune._frozen_keep)
    assert required_bytes(build, cards=1, handin_slots=1 + 3) == 6 * build


def test_one_build_s_size_comes_from_the_configs_own_bound(tmp_path):
    from qda_agent.judge.custody import build_bytes, handin_slots

    cand0 = tmp_path / "cand0"
    cand0.mkdir()
    (cand0 / "model.safetensors").write_bytes(b"x" * 4096)
    (cand0 / "config.json").write_text("{}")
    assert build_bytes(SimpleNamespace(engine="vllm", budget_bytes=None, format_reference=cand0)) == 4098
    assert build_bytes(SimpleNamespace(engine="llama.cpp", budget_bytes=1234, format_reference=None)) == 1234
    assert build_bytes(SimpleNamespace(engine="vllm", budget_bytes=None, format_reference=None)) is None
    rule = SimpleNamespace(frozen_keep=lambda: 3)
    # the hand-in's own copy is always counted; a cutoff rule's frozen copies are added, as _frozen_keep keeps them
    assert handin_slots(SimpleNamespace(final_rules=[SimpleNamespace(frozen_keep=lambda: 0), rule])) == 4
    assert handin_slots(SimpleNamespace(final_rules=[])) == 1


def test_the_runs_own_check_names_a_cap_below_what_it_must_hold(tmp_path, monkeypatch):
    """The run's start check (not a second opinion in doctor) names the required cap, pasteable."""
    from qda_agent.judge.custody import space_problem

    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=10**12, f_frsize=1, f_favail=10**9, f_blocks=10**12, f_files=10**9
        ),
    )
    build = 6_110_000_000
    need = required_bytes(build, cards=1)
    problem = space_problem(tmp_path / "local", cap=need - 1, build=build, cards=1, handin_slots=1)
    assert problem is not None and "18.3 GB" in problem.detail  # 3 builds of 6.11 GB
    assert problem.fix == f"--set judge.max_run_bytes={need}"
    assert space_problem(tmp_path / "local", cap=need, build=build, cards=1) is None


def test_the_runs_own_check_refuses_a_cap_above_the_free_disk(tmp_path, monkeypatch):
    """Prepare wrote a 36.7 GB cap on a disk with 4.3 GB free and doctor only warned."""
    from qda_agent.judge.custody import space_problem

    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=4_300_000_000, f_frsize=1, f_favail=10**9, f_blocks=10**12, f_files=10**9
        ),
    )
    problem = space_problem(tmp_path / "run", cap=36_700_000_000, build=6_110_000_000, cards=1)
    assert problem is not None and "above the 4.3 GB free on" in problem.detail
    assert "--out on a larger disk" in problem.fix


def test_local_handins_share_inodes_and_pruning_frees_last_links(tmp_path, monkeypatch):
    from test_build_receipt import build_again

    r, first = local_build(tmp_path)
    frozen = r.host.entries[0].dir
    a = (first / "published" / "model.safetensors").stat()
    b = (frozen / "model.safetensors").stat()
    assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    other = build_again(r, "better", weights=b"w2")
    assert r.call("kl", "better").exit == 0
    r.settle()
    r.host.entries[0].metrics["kl"] = 0.5
    r.host.entries[1].metrics["kl"] = 0.01
    state = {"left": 1}  # room for the next build once one build's weights are gone
    original = prune._prune_one

    def counted(*args, **kwargs):
        record = original(*args, **kwargs)
        if record is not None:
            state["left"] -= 1
        return record

    monkeypatch.setattr(prune, "_prune_one", counted)
    monkeypatch.setattr(prune, "_short_of_room", lambda host: state["left"] > 0)
    removed = prune.prune(r.host)  # the worse-KL build goes first
    assert any(rec["destination"] == "candidate" and rec["bytes"] > 0 for rec in removed)
    assert not (frozen / "model.safetensors").exists()
    assert not (first / "published").exists()
    assert (other / "published" / "model.safetensors").is_file()


def _disk(monkeypatch, free_bytes: int, total_bytes: int = 100 * 10**9) -> dict[str, int]:
    """Every `statvfs` reads `free_bytes` free (plenty of inodes) on a `total_bytes` disk; set the dict's
    "free" to change it later. Without `judge.anchor` the default floor is 10 GB, whatever `total_bytes`."""
    disk = {"free": free_bytes}
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=disk["free"], f_frsize=1, f_favail=10**9, f_blocks=total_bytes, f_files=10**9
        ),
    )
    return disk


def test_without_a_cap_the_floor_is_the_one_disk_check_of_doctor_and_the_start(tmp_path, monkeypatch):
    """No cap by default, so no estimate to check. The start refuses a disk already below the free-space
    floor, and `qda doctor --search` prints that same refusal (one `check_space`), not an estimate-based MISS."""
    from qda_agent.doctor import check_disk_cap

    judge = JudgeSettings()
    assert judge.max_run_bytes is None and judge.min_free_bytes == 10_000_000_000
    cfg = SimpleNamespace(judge=judge, run_dir=tmp_path / "run", cards=[0, 1])
    disk = _disk(monkeypatch, 4_300_000_000)
    with pytest.raises(ValueError) as refused:
        Rig(tmp_path)  # the Rig's run_dir is tmp_path / "run", as `cfg`'s
    assert "DISK FLOOR: needs at least 10.0 GB free (the default floor)" in str(refused.value)
    assert "has only 4.3 GB; use --out on a larger disk" in str(refused.value)
    check = check_disk_cap(cfg)
    assert check.missed and check.detail == " ".join(str(refused.value).split())
    assert check.fix.endswith("or set judge.min_free_bytes by hand")

    disk["free"] = 20_000_000_000  # above the floor: no inode peak, no cap, no estimate refuses it
    check = check_disk_cap(cfg)
    assert check.ok, check.detail
    assert check.detail.startswith("none (judge.max_run_bytes unset): 20.0 GB free on ")
    assert "estimate" not in check.detail and "prepare" not in check.detail
    Rig(tmp_path / "second")  # and the start passes


def _original(tmp_path, weight_bytes: int):
    """An original model directory whose safetensors take `weight_bytes` (a sparse file: nothing is written)."""
    m = tmp_path / "original"
    m.mkdir()
    (m / "config.json").write_text("{}")
    with open(m / "model.safetensors", "wb") as f:
        f.truncate(weight_bytes)
    return m


@pytest.mark.parametrize(
    ("free", "total"),
    [(50_000_000_000, 1_800_000_000_000), (2_000_000_000, 20_000_000_000)],
    ids=["large-disk", "small-disk"],
)
def test_the_default_floor_is_two_copies_of_the_original_model_whatever_the_disk(
    tmp_path, monkeypatch, free, total
):
    """Left at its default, the floor is twice the original model's bf16 weights (`judge.anchor`'s safetensors),
    not a share of the disk: a 1.5 GB model needs 3 GB free on a 1.8 TB disk and on a 20 GB one. The start and
    `qda doctor --search` read the same value (`custody.free_floor`); a value set by hand wins."""
    from qda_agent.doctor import check_disk_cap

    judge = JudgeSettings(anchor=_original(tmp_path, 1_500_000_000))
    cfg = SimpleNamespace(judge=judge, run_dir=tmp_path / "run", cards=[0])
    disk = _disk(monkeypatch, free, total_bytes=total)
    if free >= 3_000_000_000:
        check_space(tmp_path / "run", judge, 1)
        check = check_disk_cap(cfg)
        assert check.ok and "judge.min_free_bytes floor of 3.0 GB" in check.detail
    else:
        with pytest.raises(ValueError) as refused:
            check_space(tmp_path / "run", judge, 1)
        assert (
            "DISK FLOOR: needs at least 3.0 GB free (2 copies of the bf16 model), "
            f"{tmp_path} has only 2.0 GB; use --out on a larger disk"  # the run's nearest existing directory
        ) in str(refused.value)
        check = check_disk_cap(cfg)
        assert check.missed and check.detail == " ".join(str(refused.value).split())  # the same line

    hand = judge.model_copy(update={"min_free_bytes": 1})  # set by hand: kept as is on either disk
    check_space(tmp_path / "run", hand, 1)
    hand = judge.model_copy(update={"min_free_bytes": 10**13})
    disk["free"] = free
    with pytest.raises(ValueError, match=r"below the judge.min_free_bytes floor of 10000.0 GB.*lower judge"):
        check_space(tmp_path / "run", hand, 1)


def test_an_explicit_cap_is_still_checked_beside_the_floor(tmp_path, monkeypatch):
    """A cap set by hand keeps today's checks: above the free disk it is a MISS, and a build past it is
    refused at admission (`disk budget:`)."""
    from qda_agent.doctor import check_disk_cap

    _disk(monkeypatch, 20_000_000_000)
    cfg = SimpleNamespace(
        judge=JudgeSettings(max_run_bytes=30_000_000_000), run_dir=tmp_path / "x", cards=[0]
    )
    check = check_disk_cap(cfg)
    assert not check.ok and "above the 20.0 GB free on" in check.detail
    r = Rig(tmp_path, max_run_bytes=1 << 30)
    r.host.cfg.max_run_bytes = 1
    with pytest.raises(prune.SpaceRefused, match="^disk budget"):
        prune.check_room(r.host)


def test_below_the_floor_a_running_build_is_stopped_and_the_run_falls_back_saying_so(
    tmp_path, monkeypatch, capsys
):
    """The floor is read live each scheduler pass. Below it the agent's running build is stopped, the
    stop is recorded as a space refusal so the `handin` rule submits the best existing entry (the
    space fallback), and the terminal gets two loud lines; nothing new is admitted until the disk is above it."""
    from qda_agent.judge.ledger import Entry

    disk = _disk(monkeypatch, 10**12)
    # no cap: the floor is the only disk guard; set by hand, so the lines below are a hand-set floor's
    r = Rig(tmp_path, min_free_bytes=10_000_000_000)
    assert r.host.cfg.max_run_bytes is None

    def entry(name, kind, digest, kl=None):
        read = {} if kl is None else {"kl": kl}
        frozen = ckpt(r.host.jdir / "frozen" / name, digest.encode())  # a whole copy, as a run holds
        return Entry(
            id=name,
            kind=kind,
            src=f"out/{name}",
            dir=frozen,
            digest=digest,
            submitted=r.clock.now(),
            ended=r.clock.now(),
            valid=True,
            metrics=read,
            higher_is_better=dict.fromkeys(read, False),
        )

    # k8 read best; the agent's hand-in h1 is k1's files, which read far worse
    r.host.entries += [
        entry("k1", "scored", "d1", 0.9),
        entry("k8", "scored", "d8", 0.05),
        entry("h1", "handin", "d1"),
    ]
    (r.work / "s.sh").write_text("echo 1\n")
    assert r.call("qsched", "add", "--expected", "5", "--name", "build", "s.sh").exit == 0
    r.host.tick()
    assert r.host.sched.get("build").state is State.running
    assert r.host.final().choice.id == "h1"  # above the floor: the agent's own hand-in counts
    capsys.readouterr()

    disk["free"] = 4_300_000_000  # the build wrote the disk below the 10 GB floor
    r.host.tick()
    assert r.host.sched.get("build").state is State.cancelled
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        f"judge: DISK FLOOR: stopped build: 4.3 GB free on {lines[0].split(' free on ')[1].split(',')[0]}, "
        "below the judge.min_free_bytes floor of 10.0 GB; free space on that disk or use --out on a larger "
        "disk (move run_dir, judge.custody_dir or the agent's work_dir), or lower judge.min_free_bytes",
        "judge: DISK FLOOR: falling back to the best existing entry: the run submits the best scored eligible "
        "build (or candidate 0 as-is) instead",
    ]
    records = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
    (record,) = [rec for rec in records if rec.get("kind") == "space_refusal"]
    assert record["what"] == "build" and record["build"] == "build"
    final = r.host.final()
    assert final.choice is not None and final.choice.id == "k8" and final.chosen_by == "handin"
    assert final.reason == (
        "build build stopped for disk space; submitted the best scored eligible build k8 (KL 0.05) instead"
    )

    reply = r.call("qsched", "add", "--expected", "5", "--name", "again", "s.sh")
    assert reply.exit == 2 and "DISK FLOOR" in reply.stderr  # admission reads the same floor
    r.host.tick()
    assert capsys.readouterr().err == ""  # said once per crossing, not every pass
    disk["free"] = 10**12
    r.host.tick()
    assert r.call("qsched", "add", "--expected", "5", "--name", "again", "s.sh").exit == 0


def test_below_the_floor_a_handin_is_refused_for_space_in_two_lines(tmp_path, monkeypatch, capsys):
    """A hand-in refused at the floor is a space refusal: recorded, said in two lines, and the run
    submits the best scored entry instead."""
    r = Rig(tmp_path, min_free_bytes=10_000_000_000)  # set by hand, as the lines below word it
    ckpt(r.work / "out" / "k8", b"w8")
    _disk(monkeypatch, 4_300_000_000)
    capsys.readouterr()
    reply = r.call("handin", "out/k8")
    assert reply.exit == 2 and "DISK FLOOR" in reply.stderr
    first, second = capsys.readouterr().err.splitlines()
    assert first.startswith("judge: DISK FLOOR: hand-in of out/k8 refused: 4.3 GB free on ")
    assert second.startswith("judge: DISK FLOOR: falling back to the best existing entry")
    (record,) = r.host.space_refusals
    assert record["build"] == "out/k8" and record["cap"] is None and "what" not in record
    assert not (r.host.jdir / "frozen" / "h1").exists()  # refused before any copy
