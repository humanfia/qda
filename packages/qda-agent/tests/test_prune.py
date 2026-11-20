"""Build pruning: near the floor the judge frees the worst-KL builds' weights, safely; hand-ins stay."""

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading

import pytest
from qda_agent.judge import prune
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.freeze import file_sha256, seal
from qda_agent.judge.ledger import Entry
from qda_agent.sched import Job, State
from safetensors_kit import anchor_dir, write_safetensors
from test_gencheck import GenRig
from test_judge import Rig, ckpt, kl_effect


def custody(root, count=5, handins=False, readings=True, rules=("best_metric",), **judge):
    """A rig whose custody holds `count` published builds (KL 0.1, 0.2, ...), each with a scored reading."""
    judge.setdefault("min_free_bytes", 0)  # the floor is off unless a test names it: the host's disk is not ours
    r = Rig(
        root,
        require_build_receipt=True,
        final_rules=list(rules),
        qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
        **judge,
    )
    r.host.qda_argv = [sys.executable, "-m", "qda_quant.cli"]
    trees = []
    for i in range(count):
        name = f"b{i}"
        private = r.host.jdir / "builds" / name
        tree = ckpt(private / "published", bytes([i + 1]))
        (tree / "build.json").write_text(json.dumps({"serve": {}}))
        digest = seal(tree)
        receipt = {
            "job_id": name,
            "destination": name,
            "workdir": str(r.work),
            "tree_sha256": digest,
            "build_sha256": file_sha256(tree / "build.json"),
        }
        (private / "receipt.json").write_text(json.dumps(receipt))
        r.host.sched.jobs[name] = Job(
            id=name,
            name=name,
            seq=i,
            argv=["true"],
            cwd=str(r.work),
            expected_s=1,
            submitted=r.clock.now(),
            state=State.done,
            rc=0,
            build_private=str(private),
        )
        if not readings:  # a published build nothing has read yet
            trees.append(tree)
            continue
        frozen = r.host.jdir / "frozen" / f"k{i}"
        shutil.copytree(tree, frozen, copy_function=os.link)
        seal(frozen)
        entry = Entry(
            id=f"k{i}",
            kind="scored",
            src=name,
            dir=frozen,
            digest=digest,
            submitted=r.clock.now(),
            ended=r.clock.now(),
            valid=True,
            metrics={"kl": (i + 1) / 10},
            higher_is_better={"kl": False},
            build_receipt=receipt,
        )
        r.host.entries.append(entry)
        if handins:
            handed = r.host.jdir / "frozen" / f"h{i}"
            shutil.copytree(tree, handed, copy_function=os.link)
            seal(handed)
            r.host.entries.append(
                entry.model_copy(
                    update={
                        "id": f"h{i}",
                        "kind": "handin",
                        "dir": handed,
                        "metrics": {},
                        "higher_is_better": {},
                    }
                )
            )
        trees.append(tree)
    return r, trees


def floor_short(monkeypatch, n: int = 1):
    """Make the pass see no room until it has pruned `n` builds (the loop's own stop condition).

    The real floor is a `statvfs` of the test host, which these rigs do not control; the builds' bytes are
    too small to move it. Counting the prunes is the condition the pass itself tests after each one.
    """
    state = {"left": n}
    original = prune._prune_one

    def counted(*args, **kwargs):
        record = original(*args, **kwargs)
        if record is not None:
            state["left"] -= 1
        return record

    monkeypatch.setattr(prune, "_prune_one", counted)
    monkeypatch.setattr(prune, "_short_of_room", lambda host: state["left"] > 0)


def always_short(monkeypatch):
    """Make every prune pass run as if the disk were at its floor."""
    monkeypatch.setattr(prune, "_short_of_room", lambda host: True)


def test_the_floor_trigger_asks_for_the_next_builds_estimate(tmp_path, monkeypatch):
    """`_short_of_room` is true only when one more build's output would not fit above the floor."""
    from types import SimpleNamespace

    r, _trees = custody(tmp_path, 1)
    r.host.cfg.anchor = tmp_path / "anchor"  # only tests that the estimate comes from the anchor
    r.host.cfg.min_free_bytes = 10_000  # set by hand, so `free_floor` returns it as is
    monkeypatch.setattr(prune, "expected_bytes", lambda host: 1_000)
    free = {"bytes": 10_500}
    monkeypatch.setattr(
        prune,
        "room",
        lambda path: SimpleNamespace(free_bytes=free["bytes"], path=path, total_bytes=10**12, free_inodes=10**9),
    )
    assert prune._short_of_room(r.host) is True  # 10.5 GB free is less than the floor + one build
    free["bytes"] = 11_000
    assert prune._short_of_room(r.host) is False  # exactly the floor + one build: it fits
    free["bytes"] = 9_000
    r.host.cfg.min_free_bytes = 0
    assert prune._short_of_room(r.host) is False  # the floor is off: only the disk cap frees anything


def test_nothing_is_pruned_while_the_next_build_fits(tmp_path):
    """Room above the floor is not a reason to delete a scored build's weights."""
    r, trees = custody(tmp_path, 3)
    r.host.cfg.min_free_bytes = 0  # the floor is off, and no cap is set
    assert prune.prune(r.host) == []
    assert all((tree / "model.safetensors").is_file() for tree in trees)
    assert not (r.host.jdir / "ledger.jsonl").exists()  # no prune record


def floor_at(monkeypatch, free_bytes: int = 1 << 20) -> None:
    """Make every disk read report `free_bytes` free, so the real floor trips on any path."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=free_bytes, f_frsize=1, f_favail=10**9, f_blocks=10**12, f_files=10**9
        ),
    )


def agent_floor_text(refused: Exception) -> str:
    """A floor refusal as the agent reads it; the operator's fix (disks, --out, the setting) is not there."""
    text = " ".join(str(refused).split())
    for operator in ("free space on that disk", "--out", "min_free_bytes"):
        assert operator not in text, (operator, text)
    return text


def test_the_floor_refusal_names_only_this_runs_own_builds(tmp_path, tmp_path_factory, monkeypatch):
    """A refused agent reads the refusal: it names this run's builds, never another path on the disk."""
    r, _trees = custody(tmp_path, 2, handins=True)
    elsewhere = tmp_path_factory.mktemp("not-the-run")
    (elsewhere / "someone-elses-cache.bin").write_bytes(b"x" * 4096)
    r.host.cfg.min_free_bytes = 10_000_000_000
    floor_at(monkeypatch)
    with pytest.raises(prune.SpaceRefused) as refused:
        prune.check_room(r.host)
    text = agent_floor_text(refused.value)
    assert "held by this run's own builds and frozen copies: build b0" in text
    assert "build b1" in text and "(handed in)" in text
    assert str(elsewhere) not in text
    assert "someone-elses-cache" not in text
    assert "outside the run's own files" not in text
    assert "never delete a file outside it" in text


def test_the_floor_refusal_names_a_frozen_copy_when_no_build_holds_the_weights(
    tmp_path, tmp_path_factory, monkeypatch
):
    """Without build receipts the run's frozen copies are its only weights: the refusal names them."""
    r = Rig(tmp_path, final_rules=["best_metric"])
    ckpt(r.work / "a", b"a")
    kl_effect(r, 0.3)
    assert r.call("kl", "a").exit == 0
    r.settle()
    elsewhere = tmp_path_factory.mktemp("not-the-run")
    (elsewhere / "someone-elses-cache.bin").write_bytes(b"x" * 4096)
    r.host.cfg.min_free_bytes = 10_000_000_000
    floor_at(monkeypatch)
    with pytest.raises(prune.SpaceRefused) as refused:
        prune.check_room(r.host)
    text = agent_floor_text(refused.value)
    assert "held by this run's own builds and frozen copies: frozen copy " in text
    assert str(elsewhere) not in text
    assert "outside the run's own files" not in text
    assert "never delete a file outside it" in text


def test_the_floor_refusal_names_a_copy_by_what_needs_it_most(tmp_path, monkeypatch):
    """One frozen copy can back a scored entry and a hand-in: the refusal says why it must stay."""
    r = Rig(tmp_path, final_rules=["best_metric"])
    ckpt(r.work / "a", b"a")
    kl_effect(r, 0.3)
    assert r.call("kl", "a").exit == 0
    r.settle()
    ckpt(r.work / "b", b"b")
    kl_effect(r, 0.1)  # the final pick, so the hand-in's copy has to say what else keeps it
    assert r.call("kl", "b").exit == 0
    r.settle()
    assert r.call("handin", "a").exit == 0  # the same digest: one frozen copy, now also handed in
    r.settle()
    r.host.cfg.min_free_bytes = 10_000_000_000
    floor_at(monkeypatch)
    with pytest.raises(prune.SpaceRefused) as refused:
        prune.check_room(r.host)
    text = agent_floor_text(refused.value)
    assert "frozen copy " in text and "(handed in)" in text and "(the final pick)" in text


def test_the_floor_refusal_says_so_when_the_run_holds_nothing(tmp_path, monkeypatch):
    """An empty run names no path at all and still tells the agent where it may free space."""
    r = Rig(tmp_path, final_rules=["best_metric"])
    r.host.cfg.min_free_bytes = 10_000_000_000
    floor_at(monkeypatch)
    with pytest.raises(prune.SpaceRefused) as refused:
        prune.check_room(r.host)
    text = agent_floor_text(refused.value)
    assert "no build or frozen copy of this run holds weights" in text
    assert "outside the run's own files" not in text
    assert "never delete a file outside it" in text


def test_removes_the_worst_kl_builds_first_until_the_next_build_fits(tmp_path, monkeypatch):
    r, trees = custody(tmp_path, 5)
    floor_short(monkeypatch, 2)  # room for one more build once two are gone
    records = prune.prune(r.host)
    assert [rec["destination"] for rec in records] == ["b4", "b3"]
    for i, tree in enumerate(trees):
        assert (tree / "model.safetensors").exists() == (i < 3)
        assert (tree.parent / "receipt.json").is_file()
        if i >= 3:  # the publication is renamed out of the mount list; metadata and logs stay
            assert (tree.parent / "pruned" / "build.json").is_file()
    assert sum(rec["bytes"] for rec in records) > 0
    lines = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
    assert all(rec in lines for rec in records)
    assert prune.prune(r.host) == []  # the next build fits now: a second pass frees nothing


def test_near_the_floor_copies_nothing_needs_are_reclaimed_whole_and_the_ledger_stays(tmp_path, monkeypatch):
    """Above the floor nothing is deleted; near it, a copy whose reading landed and that nothing needs is
    deleted whole, and its ledger line, with the score, stays."""
    r = Rig(tmp_path, final_rules=["best_metric"])  # `judge.require_build_receipt: false`
    for name, kl in (("a", 0.3), ("b", 0.1), ("c", 0.2)):
        ckpt(r.work / name, name.encode())
        kl_effect(r, kl)
        assert r.call("kl", name).exit == 0
        r.settle()
    copies = {e.src: e.dir for e in r.host.entries}
    assert all(copy.is_dir() for copy in copies.values())  # room above the floor: every copy stays
    always_short(monkeypatch)
    r.host.tick()  # near the floor, only the current pick's copy is needed
    assert not copies["a"].exists() and not copies["c"].exists()
    assert copies["b"].is_dir()  # the `best_metric` rule's pick (KL 0.1)
    records = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
    reclaimed = [rec for rec in records if rec.get("kind") == "reclaim"]
    assert sorted(rec["frozen"] for rec in reclaimed) == ["k1", "k3"]
    assert sum(rec["bytes"] for rec in reclaimed) > 0
    # the score records stay: the ledger still holds every entry, with its reading
    entries = {e.id: e for e in r.host.entries}
    assert entries["k1"].valid and entries["k1"].metrics["kl"] == 0.3
    assert entries["k3"].valid and entries["k3"].metrics["kl"] == 0.2


def test_one_digest_is_frozen_once_across_kl_handin_and_gencheck(tmp_path):
    """#...: a checkpoint read, handed in and generation-checked is frozen once. `intake.frozen_for` is the
    one owner of "this digest is frozen at"; each call still gets its own ledger entry, all pointing there."""
    r = GenRig(tmp_path)
    ckpt(r.work / "out" / "a", b"w1")
    assert r.call("kl", "out/a").exit == 0
    r.settle()
    assert r.call("handin", "out/a").exit == 0
    r.settle()
    assert r.call("gencheck", "out/a").exit == 0
    entries = {e.id: e for e in r.host.entries}
    assert {"k1", "h1", "g1"} <= set(entries)
    assert len({entries[i].dir for i in ("k1", "h1", "g1")}) == 1
    assert [p.name for p in (r.host.jdir / "frozen").iterdir()] == ["k1"]  # one copy, not three
    assert entries["k1"].digest == entries["h1"].digest == entries["g1"].digest


def test_a_copy_a_queued_or_running_check_reads_is_not_reclaimed(tmp_path, monkeypatch):
    """A copy stays while any entry of it has no verdict yet: the check has not read it out."""
    r = Rig(tmp_path)
    ckpt(r.work / "out" / "a", b"w1")
    kl_effect(r, 0.3)
    assert r.call("kl", "out/a").exit == 0
    r.settle()
    first = next(e.dir for e in r.host.entries if e.id == "k1")
    assert r.call("handin", "out/a").exit == 0  # the hand-in reuses k1's copy
    ckpt(r.work / "out" / "b", b"w2")
    kl_effect(r, 0.1)
    assert r.call("kl", "out/b").exit == 0
    r.host.tick()  # b's reading started; the hand-in's check is queued or running
    always_short(monkeypatch)
    prune.prune(r.host)
    assert first.is_dir()  # an entry of the copy has no verdict: this pass frees neither copy
    r.settle()  # both jobs end: the hand-in is a valid submission, so its copy stays whole
    entries = {e.id: e for e in r.host.entries}
    assert entries["h1"].valid is True and entries["h1"].dir.is_dir()


def test_a_refused_handins_copy_is_reclaimed_and_its_record_stays(tmp_path, monkeypatch):
    """A refused hand-in is no result: near the floor its copy goes, its entry line stays."""
    r = Rig(tmp_path)
    ckpt(r.work / "out" / "a", b"w1")
    kl_effect(r, 0.3)
    assert r.call("kl", "out/a").exit == 0
    r.settle()
    r.check_result = {"ok": False}
    assert r.call("handin", "out/a").exit == 0  # same files as k1: one more entry of one copy
    r.settle()
    ckpt(r.work / "out" / "b", b"w2")
    kl_effect(r, 0.1)
    assert r.call("kl", "out/b").exit == 0
    r.settle()
    always_short(monkeypatch)
    prune.prune(r.host)
    entries = {e.id: e for e in r.host.entries}
    assert entries["h1"].valid is False and not entries["h1"].dir.exists()
    assert entries["k1"].metrics["kl"] == 0.3  # the score record is kept with the entry
    assert any(
        json.loads(line).get("kind") == "reclaim"
        for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()
    )


def _handed_in(r, build: int, eid: str, list_id: int, slot: int = 0) -> Entry:
    """A valid hand-in of custody build `build` in hand-in list `list_id`: its copy hardlinks the build."""
    scored = next(e for e in r.host.entries if e.id == f"k{build}")
    handed = r.host.jdir / "frozen" / eid
    shutil.copytree(r.host.jdir / "builds" / f"b{build}" / "published", handed, copy_function=os.link)
    seal(handed)
    entry = scored.model_copy(
        update={
            "id": eid,
            "kind": "handin",
            "dir": handed,
            "metrics": {},
            "higher_is_better": {},
            "list_id": list_id,
            "slot": slot,
        }
    )
    r.host.entries.append(entry)
    return entry


def _has_weights(tree) -> bool:
    return any(p.suffix == ".safetensors" for p in tree.rglob("*"))


def _list_of_three(tmp_path):
    """A handin-rule custody run: one hand-in list of three valid entries."""
    r, trees = custody(tmp_path, 4, rules=("handin",))
    handed = [_handed_in(r, 1 + slot, f"h{slot + 1}", list_id=1, slot=slot) for slot in range(3)]
    return r, trees, handed


def _metadata_only(r) -> list[str]:
    """The frozen copies a pass left behind without their weights, as measured on a real run."""
    root = r.host.jdir / "frozen"
    return [p.name for p in root.iterdir() if p.is_dir() and not _has_weights(p)] if root.is_dir() else []


def test_near_the_floor_every_handins_copy_stays_whole(tmp_path, monkeypatch):
    """Every valid hand-in of the list keeps its copy whole, whether or not a rule can pick it;
    the pass leaves no copy as metadata, and every ledger line stays."""
    r, _trees, (first, second, third) = _list_of_three(tmp_path)
    always_short(monkeypatch)
    prune.prune(r.host)
    for handed in (first, second, third):
        assert handed.dir.is_dir() and _has_weights(handed.dir)
    assert _metadata_only(r) == []
    assert {e.id for e in r.host.entries} >= {"h1", "h2", "h3"}  # the entries stay in the ledger


def _end_report(tmp_path, final) -> str:
    """The END-REPORT a run with `final` writes (`flow._end_report`, the run's own writer)."""
    import time

    from qda_agent import flow
    from qda_agent.config import RunConfig
    from tests_support import search_config

    (tmp_path / "report").mkdir()
    cfg = RunConfig.load(search_config(tmp_path / "report"))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log = flow._Log(cfg.run_dir / "flow.jsonl", time.time)
    tools = flow._Log(cfg.run_dir / "tools.jsonl", time.time)
    now = time.time()
    flow._end_report(cfg, final, log, tools, flow.EndKind.DEADLINE, "at the deadline", None, now, now, 1.0)
    return (cfg.run_dir / "END-REPORT.md").read_text()


def test_a_pick_that_stops_counting_never_names_a_stripped_or_missing_copy(tmp_path):
    """A valid entry whose copy no longer holds its weights is skipped: FINAL names no copy that
    is gone, takes the documented fallback (candidate 0 as-is), and says which hand-ins it skipped and why,
    in its reason and in `why_not`, as END-REPORT prints them. A prune pass never leaves a valid hand-in
    stripped any more; the rule still has to read such a copy (an operator may have removed it)."""
    r, _trees, (first, second, third) = _list_of_three(tmp_path)
    for handed in (second, third):  # a copy an older pass could have left without its weights
        for path in [p for p in handed.dir.rglob("*") if p.suffix == ".safetensors"]:
            prune._unlink(path, {})
    r.host.cfg.baselines = [tmp_path / "candidate0"]  # the documented fallback needs a candidate 0
    r.host.update(first, eligible=False, reason="not eligible: this build's KL reading was refused")
    final = r.host.final(generation_checks=False)
    assert final.choice is None and final.chosen_by == "fallback_candidate0"
    assert final.reason == (
        "valid hand-ins h2, h3 skipped: their frozen copies no longer hold weights; candidate 0 as-is"
    )
    assert final.why_not == [
        "h2: its frozen copy no longer holds its weights",
        "h3: its frozen copy no longer holds its weights",
    ]
    report = _end_report(tmp_path, final)
    assert (
        "valid hand-ins h2, h3 skipped: their frozen copies no longer hold weights; candidate 0 as-is"
        in report
    )
    assert "- not h2: its frozen copy no longer holds its weights" in report
    assert "- not h3: its frozen copy no longer holds its weights" in report


def test_a_whole_first_entry_is_picked_and_nothing_is_said_skipped(tmp_path):
    """Slot 0 is whole and is the pick while slots 1-2 are still whole too: nothing was passed over for
    the pick, so neither the reason nor `why_not` names them."""
    r, _trees, (first, second, third) = _list_of_three(tmp_path)
    assert first.dir.is_dir() and second.dir.is_dir() and third.dir.is_dir()
    final = r.host.final(generation_checks=False)
    assert final.choice is not None and final.choice.id == "h1" and final.chosen_by == "handin"
    assert final.reason == "the first valid entry of the latest hand-in list (h1, entry 1)"
    assert final.why_not == []


def test_the_disk_refusal_path_skips_a_gone_copy_and_says_so():
    """After a disk refusal the rule ranks only whole copies; the better reading whose copy is gone is
    named in the reason and in `why_not`; with every copy whole the reason is unchanged."""
    from datetime import UTC, datetime
    from pathlib import Path

    from qda_agent.host import HandIn
    from qda_agent.judge.final_rules import Handin, apply

    def h(name, kind, kl=None, **kw):
        read = {} if kl is None else {"kl": kl}
        record = {"id": name, "dir": name, "submitted_at": datetime.fromtimestamp(0, UTC), "kind": kind}
        return HandIn.model_validate(
            {**record, "valid": True, "metrics": read, "higher_is_better": dict.fromkeys(read, False), **kw}
        )

    ledger = [h("k1", "scored", 0.9), h("k8", "scored", 0.05), h("h1", "handin", handin_list=1)]
    refusal = [{"build": "out/k8"}]
    final = apply(
        [Handin()], ledger, [Path("c0")], "kl", disk_refusals=refusal, whole=lambda d: d != Path("k8")
    )
    assert final.choice is not None and final.choice.id == "k1"
    assert final.reason == (
        "hand-in of out/k8 refused for disk space; submitted the best scored eligible build k1 (KL 0.9) instead; "
        "valid entry k8 skipped: its frozen copy no longer holds weights"
    )
    assert final.why_not == ["k8: its frozen copy no longer holds its weights"]
    whole = apply([Handin()], ledger, [Path("c0")], "kl", disk_refusals=refusal)
    assert whole.choice is not None and whole.choice.id == "k8" and whole.why_not == []
    assert whole.reason.endswith("build k8 (KL 0.05) instead")


def test_a_superseded_lists_handin_copy_stays_whole(tmp_path, monkeypatch):
    """A later list supersedes an older one for the pick, but the older list's copy is still a valid
    hand-in: near the floor it stays whole, never left as metadata without weights."""
    r, _trees = custody(tmp_path, 4, rules=("handin",))
    old = _handed_in(r, 2, "h1", list_id=1)
    new = _handed_in(r, 3, "h2", list_id=2)
    always_short(monkeypatch)
    prune.prune(r.host)
    assert old.dir.is_dir() and _has_weights(old.dir)
    assert new.dir.is_dir() and _has_weights(new.dir)
    assert _metadata_only(r) == []


def test_a_rehandin_of_a_stripped_copy_is_frozen_whole_and_passes(tmp_path):
    """A copy stripped by an earlier pass is not reused; the new entry keeps its own copy."""
    r = Rig(tmp_path)
    ckpt(r.work / "out" / "a", b"w1")
    assert r.call("handin", "out/a").exit == 0
    r.settle()
    stripped = next(e.dir for e in r.host.entries if e.id == "h1")
    for path in [p for p in stripped.rglob("*") if p.suffix == ".safetensors"]:
        prune._unlink(path, {})  # what a pass of the older rule did to a valid hand-in that was not the pick
    assert stripped.is_dir() and not _has_weights(stripped)
    assert r.call("handin", "out/a").exit == 0
    r.settle()
    again = next(e for e in r.host.entries if e.id == "h2")
    assert again.dir != stripped and _has_weights(again.dir)
    assert again.valid is True


def test_a_handin_of_a_cleaned_build_without_the_agents_copy_is_refused(tmp_path, monkeypatch):
    """The weights of a build the floor pass freed are gone; an agent copy that still hashes to the receipt
    would be re-frozen (`intake.take`), but here there is none: the hand-in says so and asks for a rebuild."""
    r, _ = custody(tmp_path, 3)
    always_short(monkeypatch)
    prune.prune(r.host)
    reply = r.call("handin", "b2")
    assert reply.exit == 2
    assert "weights were cleaned" in reply.stderr and "build it again" in reply.stderr


def test_a_rehandin_of_a_cleaned_build_from_an_intact_agent_copy_is_frozen_again(tmp_path, monkeypatch):
    """A build the floor pass cleaned can be handed in again: the agent's unchanged copy is the build."""
    from qda_agent.judge import build_receipt

    r, trees = custody(tmp_path, 3)
    shutil.copytree(trees[2], r.work / "b2")  # the agent's own copy, untouched
    always_short(monkeypatch)
    prune.prune(r.host)
    private = r.host.jdir / "builds" / "b2"
    assert (private / "pruned").is_dir() and not (trees[2] / "model.safetensors").exists()
    assert r.call("handin", "b2").exit == 0
    r.settle()
    entry = next(e for e in r.host.entries if e.kind == "handin")
    assert entry.valid is True and _has_weights(entry.dir)
    matched, why = build_receipt.match(r.host, entry.dir, entry.digest)
    assert matched is not None and matched["destination"] == "b2", why  # the receipt in `pruned/` still speaks


def test_a_rehandin_of_a_cleaned_build_with_a_changed_agent_copy_is_refused(tmp_path, monkeypatch):
    """A copy the agent changed after the build is not the recorded build: the hand-in asks for a rebuild."""
    r, trees = custody(tmp_path, 3)
    shutil.copytree(trees[2], r.work / "b2")
    weights = r.work / "b2" / "model.safetensors"
    os.chmod(weights, 0o644)  # the build's copy is read-only; the agent changed its own
    weights.write_bytes(weights.read_bytes() + b"x")
    always_short(monkeypatch)
    prune.prune(r.host)
    reply = r.call("handin", "b2")
    assert reply.exit == 2
    assert "weights were cleaned" in reply.stderr and "build it again" in reply.stderr


def test_a_refused_reading_does_not_keep_its_weights(tmp_path, monkeypatch):
    """A refused reading is a verdict: near the floor that build goes first (it has no KL, so it is worst)."""
    r, trees = custody(tmp_path, 3)
    r.host.entries[2].valid, r.host.entries[2].metrics = False, {}
    r.host.entries[2].reason = "KL too high"
    floor_short(monkeypatch, 1)  # room for the next build once one scored build is gone
    prune.prune(r.host)
    assert not (trees[2] / "model.safetensors").exists()
    assert (trees[0] / "model.safetensors").is_file()
    assert (trees[1] / "model.safetensors").is_file()


def test_an_open_handin_and_a_pending_job_keep_their_builds(tmp_path, monkeypatch):
    r, trees = custody(tmp_path, 4)
    scored = next(e for e in r.host.entries if e.id == "k3")
    r.host.entries.append(  # its check is queued: the build is not a result yet, but nothing may free it
        scored.model_copy(update={"id": "h1", "kind": "handin", "valid": None, "ended": None})
    )
    private = r.host.jdir / "builds" / "next"
    private.mkdir()
    config = private / "config.yaml"
    config.write_text(f"model: {r.work / 'b2'}\nout: next\n")  # a queued job reads b2 via its snapshot
    (private / "request.json").write_text(
        json.dumps({"destination": "next", "config": str(config), "cwd": str(r.work)})
    )
    r.host.sched.jobs["consumer"] = Job(
        id="consumer",
        name="consumer",
        seq=9,
        argv=["qda", "build", str(config)],
        cwd=str(r.work),
        expected_s=1,
        submitted=r.clock.now(),
        state=State.queued,
        build_private=str(private),
    )
    always_short(monkeypatch)
    prune.prune(r.host)
    assert (trees[3] / "model.safetensors").is_file()  # the open hand-in
    assert (trees[2] / "model.safetensors").is_file()  # the queued job reads it
    assert (trees[0] / "model.safetensors").is_file()  # the final pick (KL 0.1)
    assert not (trees[1] / "model.safetensors").exists()


def test_candidate_zero_and_the_current_final_pick_keep_their_weights(tmp_path, monkeypatch):
    r, trees = custody(tmp_path, 4)  # `best_metric` picks b0, the best KL reading
    r.host.cfg.baselines = [r.work / "b3"]
    always_short(monkeypatch)
    prune.prune(r.host)
    assert (trees[3] / "model.safetensors").is_file()  # candidate 0
    assert (trees[0] / "model.safetensors").is_file()  # the final pick
    assert not (trees[1] / "model.safetensors").exists()
    assert not (trees[2] / "model.safetensors").exists()


def test_the_pick_rule_waits_quietly_for_candidate_zeros_reading(tmp_path, caplog):
    """With `select_candidate0`, candidate 0's KL is known only once the judge's own reading of it is folded.
    Before that, a prune pass logs no traceback and names no pick; from that reading on, the pick is among what
    pruning keeps."""
    import logging

    r, _ = custody(tmp_path, 4)
    rules = JudgeSettings(final_rules=["best_metric"])
    r.host.cfg.final_rules, r.host.cfg.select_candidate0 = rules.final_rules, True
    r.host.cfg.candidate0_kl = None  # the start phase: candidate 0 not read yet
    found = prune.builds(r.host)
    with caplog.at_level(logging.ERROR, logger=prune.__name__):
        before = prune._in_use(r.host, found)
        prune.prune(r.host)
    assert not [rec for rec in caplog.records if rec.exc_info], "no traceback before candidate 0 is read"

    r.host.cfg.candidate0_kl = 0.5  # what `select_baselines` records once candidate 0 is read
    pick = r.host.final(generation_checks=False).choice
    assert pick is not None
    picked = {prune._entry_identity(e) for e in r.host.entries if e.dir == pick.dir} - {None}
    assert picked and not picked & before  # the start phase named no pick
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=prune.__name__):
        assert picked <= prune._in_use(r.host, prune.builds(r.host))  # the pick is kept from here on
    assert not caplog.records


def test_a_custody_handin_freezes_one_physical_copy(tmp_path):
    """One digest is frozen once: the hand-in points at the scored reading's copy, not a second one."""
    r, trees = custody(tmp_path, 3)
    assert r.call("handin", "b0").exit == 0
    e = r.host.entries[-1]
    reading = next(x for x in r.host.entries if x.id == "k0")
    assert e.dir == reading.dir  # the same frozen directory, one entry each
    published = (trees[0] / "model.safetensors").stat()
    frozen = (e.dir / "model.safetensors").stat()
    assert (frozen.st_dev, frozen.st_ino) == (published.st_dev, published.st_ino)
    assert frozen.st_nlink == 2  # the publication and the one frozen copy of both entries
    assert len({p for p in (r.host.jdir / "frozen").iterdir()}) == 3  # k0..k2: the hand-in added none


@pytest.mark.parametrize("dtype, width", [("F8_E4M3", 1), ("BF16", 2), ("F32", 4)])
def test_disk_reservation_subtracts_the_originals_actual_stored_width(tmp_path, dtype, width):
    original = anchor_dir(tmp_path / "original")
    write_safetensors(
        original / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (16, 8), None),
            "model.layers.0.self_attn.q_proj.weight": (dtype, (8, 8), None),
        },
    )
    r = Rig(tmp_path, anchor=original, budget_bpw=4)
    assert r.host.bpw_denominator == 64
    assert prune.expected_bytes(r.host) == r.host.cfg.anchor_bytes() - 64 * width + 32


def test_max_run_bytes_refuses_a_new_build(tmp_path):
    r, _ = custody(tmp_path, 1)
    r.host.cfg.max_run_bytes = 1
    (r.work / "recipe.yaml").write_text("out: another\n")
    reply = r.call("qda", "build", "recipe.yaml")
    assert reply.exit == 2
    assert "disk budget:" in reply.stderr
    assert "prune finished builds" in reply.stderr


def test_run_bytes_counts_the_whole_run_and_each_inode_once(tmp_path):
    r, _ = custody(tmp_path, 2, handins=True)  # the readings' copies hardlink the publications
    expected = {}
    for path in (p for p in r.run.rglob("*") if p.is_file() and not p.is_symlink()):
        info = path.lstat()
        expected[(info.st_dev, info.st_ino)] = info.st_size
    assert prune.run_bytes(r.host) == sum(expected.values())
    # a second name for one publication's weights adds no bytes: the cap sees the disk, not the paths
    before = prune.run_bytes(r.host)
    os.link(r.host.entries[0].dir / "model.safetensors", r.work / "again.safetensors")
    assert prune.run_bytes(r.host) == before


def test_a_local_build_that_would_pass_the_cap_is_refused_before_it_runs(tmp_path):
    r = Rig(tmp_path, max_run_bytes=2 << 20, max_copy_bytes=1 << 20)
    (r.work / "big.bin").write_bytes(b"x" * (1500 << 10))  # under the cap; its estimate is not
    (r.work / "recipe.yaml").write_text("model: m\nout: out\n")
    reply = r.call("qda", "build", "recipe.yaml", "--expected", "1m")
    assert reply.exit == 2
    assert "disk budget:" in reply.stderr
    assert "prune finished builds" in reply.stderr
    assert not r.host.sched.list()  # refused before it was queued


def test_a_job_is_refused_while_over_the_cap_and_admitted_once_back_under(tmp_path):
    r = Rig(tmp_path, max_run_bytes=2 << 20, max_copy_bytes=1 << 20)
    r.host.cfg.job_reserve_bytes = 1 << 20
    (r.work / "job.sh").write_text("true\n")
    big = r.work / "big.bin"
    big.write_bytes(b"x" * (3 << 20))  # over the cap on its own
    reply = r.call("qsched", "add", "job.sh", "--expected", "1m")
    assert reply.exit == 2
    assert "disk budget:" in reply.stderr
    assert not r.host.sched.list()
    big.unlink()  # the agent pruned its own work/
    reply = r.call("qsched", "add", "job.sh", "--expected", "1m")
    assert reply.exit == 0, reply.stderr
    assert len(r.host.sched.list()) == 1


def test_a_handin_copy_that_would_pass_the_cap_is_refused_and_removed(tmp_path):
    r = Rig(tmp_path)
    ckpt(r.work / "cand", b"weights")
    used = prune.run_bytes(r.host)
    r.host.cfg.max_run_bytes = used + 16  # room for a few bytes, not for the copy
    reply = r.call("handin", "cand")
    assert reply.exit == 2
    assert "disk budget:" in reply.stderr
    assert "prune finished builds" in reply.stderr
    assert not r.host.entries
    assert not (r.host.jdir / "frozen" / "h1").exists()  # the refused copy is gone again


@pytest.mark.parametrize("pending", [False, True])
def test_handin_reserves_copy_bytes_before_allocating(tmp_path, monkeypatch, pending):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    source = ckpt(r.work / "cand")
    if pending:
        (r.work / "job.sh").write_text("true\n")
        assert r.call("qsched", "add", "job.sh", "--expected", "1m").exit == 0
    used = prune.run_bytes(r.host)
    size = sum(p.stat().st_size for p in source.iterdir())
    r.host.cfg.max_run_bytes = used + prune.reserved_bytes(r.host) + size // 2

    def no_copy(*args, **kwargs):
        pytest.fail("a hand-in beyond held bytes plus reservations must refuse before copying")

    monkeypatch.setattr(r.host.sched.iso, "copy_out", no_copy)  # the level does the copy-out
    reply = r.call("handin", "cand")
    assert reply.exit == 2 and "disk budget:" in reply.stderr
    assert not r.host.entries and not (r.host.jdir / "frozen" / "h1").exists()


@pytest.mark.parametrize("build", [False, True])
def test_concurrent_jobs_cannot_spend_one_reservation_twice(tmp_path, monkeypatch, build):
    from qda_agent.sched import queue

    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    (r.work / "job.sh").write_text("true\n")
    (r.work / "recipe.yaml").write_text("model: m\nout: out\n")
    command = ("qda", "build", "recipe.yaml") if build else ("qsched", "add", "job.sh")
    r.host.cfg.job_reserve_bytes = 1 << 20  # the declared checkpoint's reservation for the qsched job
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) + (1 << 20) + (64 << 10)
    entered, release, checked_second = threading.Event(), threading.Event(), threading.Event()
    snapshot = queue.snapshot
    check_room = prune.check_room
    checks = []

    def checked(*args, **kwargs):
        check_room(*args, **kwargs)
        checks.append(True)
        if len(checks) == 2:
            checked_second.set()

    def slow_snapshot(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return snapshot(*args, **kwargs)

    monkeypatch.setattr(queue, "snapshot", slow_snapshot)
    monkeypatch.setattr(prune, "check_room", checked)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(r.call, *command, "--expected", "1m")
        assert entered.wait(10)
        second = pool.submit(r.call, *command, "--expected", "1m")
        # Without atomic admission both checks pass while the first snapshot is still blocked.
        checked_second.wait(0.2)
        release.set()
        replies = [first.result(), second.result()]
    assert sorted(reply.exit for reply in replies) == [0, 2]
    assert not r.be.launched
    assert prune.run_bytes(r.host) + prune.reserved_bytes(r.host) <= r.host.cfg.max_run_bytes
    r.settle()
    assert prune.reserved_bytes(r.host) == 0


def test_queued_agent_jobs_share_the_cap_and_release_it(tmp_path):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    r.host.cfg.job_reserve_bytes = 1 << 20
    (r.work / "job.sh").write_text("true\n")
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) + (2 << 20) + (64 << 10)
    replies = [r.call("qsched", "add", "job.sh", "--expected", "1m") for _ in range(4)]
    assert [reply.exit for reply in replies] == [0, 0, 2, 2]
    assert not r.be.launched
    r.settle()
    assert prune.reserved_bytes(r.host) == 0


def test_the_disk_refusal_states_each_part_in_decimal_gb():
    msg = prune.disk_message(16_000_000_000, 20_500_000_000, 2_000_000_000, 38_000_000_000)
    assert msg == (
        "disk budget: 16.0 GB on disk + 20.5 GB reserved for queued/running jobs + 2.0 GB for this job "
        "= 38.5 GB of 38.0 GB (on-disk use alone is under the cap); prune finished builds, hand in, or "
        "wait for queued jobs to end"
    )
    over = prune.disk_message(36_000_000_000, 0, 0, 35_000_000_000)
    assert over == (
        "disk budget: 36.0 GB on disk = 36.0 GB of 35.0 GB; prune finished builds, hand in, or wait for "
        "queued jobs to end"
    )


def test_a_refusal_caused_only_by_reservations_says_so(tmp_path):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    r.host.cfg.job_reserve_bytes = 1 << 20
    (r.work / "job.sh").write_text("true\n")
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) + (3 << 19)  # room for 1.5 jobs' reservations
    assert r.call("qsched", "add", "job.sh", "--expected", "1m").exit == 0
    reply = r.call("qsched", "add", "job.sh", "--expected", "1m")
    assert reply.exit == 2
    assert "reserved for queued/running jobs" in reply.stderr
    assert "on-disk use alone is under the cap" in reply.stderr
    assert "GiB" not in reply.stderr and "GB" in reply.stderr


def test_an_undeclared_job_reserves_the_job_default_and_a_declared_one_a_checkpoint(tmp_path):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    (r.work / "job.sh").write_text("true\n")
    assert r.call("qsched", "add", "job.sh", "--expected", "1m", "--name", "inspect").exit == 0
    assert r.host.sched.jobs["inspect"].expected_bytes == r.host.cfg.job_reserve_bytes
    assert (
        r.call("qsched", "add", "job.sh", "--expected", "1m", "--name", "build", "--checkpoint", "out").exit
        == 0
    )
    assert r.host.sched.jobs["build"].expected_bytes == prune.expected_bytes(r.host)


def test_four_inspect_jobs_fit_where_four_checkpoint_reservations_did_not(tmp_path):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    (r.work / "job.sh").write_text("true\n")
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) + 1_000_000
    r.host.cfg.job_reserve_bytes = 200_000
    replies = [r.call("qsched", "add", "job.sh", "--expected", "1m") for _ in range(4)]
    assert [reply.exit for reply in replies] == [0, 0, 0, 0]
    assert prune.reserved_bytes(r.host) == 800_000
    assert 4 * prune.expected_bytes(r.host) > r.host.cfg.max_run_bytes - prune.run_bytes(r.host)


def test_confined_copy_estimate_counts_hardlinks_once_without_following_symlinks(tmp_path):
    r = Rig(tmp_path)
    source = r.work / "candidate"
    source.mkdir()
    (source / "a").write_bytes(b"12345")
    os.link(source / "a", source / "hardlink")
    (source / "distinct").write_bytes(b"12345")
    (tmp_path / "outside").write_bytes(b"x" * 100)
    (source / "symlink").symlink_to(tmp_path / "outside")
    assert r.host.sched.iso.size_out(r.work, "candidate") == 10


def test_disk_admission_lock_and_reservations_are_visible_across_processes(tmp_path):
    r = Rig(tmp_path, max_copy_bytes=1 << 20)
    r.host.cfg.job_reserve_bytes = 1 << 20
    (r.work / "job.sh").write_text("true\n")
    r.host.cfg.max_run_bytes = 4 << 20
    attempt = [
        sys.executable,
        "-c",
        "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",
        str(r.host.jdir / "disk.lock"),
    ]
    with prune.disk_lock(r.host):
        assert subprocess.run(attempt, capture_output=True).returncode != 0
    assert subprocess.run(attempt, capture_output=True).returncode == 0
    assert r.call("qsched", "add", "job.sh", "--expected", "1m").exit == 0
    r.host.sched.jobs.clear()  # another process has only the published job records
    assert prune.reserved_bytes(r.host) == 1 << 20


def test_queued_builds_reserve_their_estimates_and_a_failed_one_releases_them(tmp_path):
    r, _ = custody(tmp_path, 1)
    private = []
    for name, state in (("one", State.queued), ("two", State.running)):
        p = r.host.jdir / "builds" / name
        p.mkdir()
        (p / "request.json").write_text(json.dumps({"destination": name, "expected_bytes": 100}))
        r.host.sched.jobs[name] = Job(
            id=name,
            name=name,
            seq=1,
            argv=["true"],
            cwd=str(r.work),
            expected_s=1,
            submitted=r.clock.now(),
            state=state,
            build_private=str(p),
        )
        private.append(p)
    assert prune.reserved_bytes(r.host) == 200  # two queued builds, before either wrote a byte
    private[0].joinpath("request.json").unlink()  # a failed build: nothing of its output is published
    (private[0] / "output").mkdir()
    (private[0] / "output" / "model.safetensors").write_bytes(b"partial")
    job = next(j for j in r.host.sched.jobs.values() if j.name == "one")
    job.state, job.rc = State.failed, 1
    r.host.tick()
    assert not private[0].exists()  # the partial output is gone and its reservation with it
    assert prune.reserved_bytes(r.host) == 100


def test_nothing_outside_the_run_custody_is_touched(tmp_path, monkeypatch):
    r, _ = custody(tmp_path, 3)
    external = ckpt(tmp_path / "outside", b"safe")
    before = (external / "model.safetensors").read_bytes()
    (r.host.jdir / "builds" / "linked").symlink_to(external, target_is_directory=True)
    always_short(monkeypatch)
    prune.prune(r.host)
    assert (external / "model.safetensors").read_bytes() == before


def publish(r, name, extra=0, at=0.0):
    """Publish an unscored build `name` in custody, `extra` weight bytes above a tiny checkpoint, written `at`."""
    private = r.host.jdir / "builds" / name
    tree = ckpt(private / "published", name.encode())
    if (
        extra
    ):  # an install that landed above its estimate: the header padded with spaces, as the format allows
        shard = tree / "model.safetensors"
        data = shard.read_bytes()
        n = int.from_bytes(data[:8], "little")
        header = data[8 : 8 + n] + b" " * extra
        shard.write_bytes(len(header).to_bytes(8, "little") + header + data[8 + n :])
    (tree / "build.json").write_text(json.dumps({"serve": {}}))
    receipt = {"job_id": name, "destination": name, "workdir": str(r.work), "tree_sha256": seal(tree)}
    (private / "receipt.json").write_text(json.dumps(receipt))
    os.utime(private / "receipt.json", (at, at))
    return tree, receipt


def over_by(r, n):
    """Set `judge.max_run_bytes` so the run holds `n` bytes more than it allows."""
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) - n


def test_a_run_pushed_over_the_cap_frees_an_old_unscored_build_and_scores_its_next(tmp_path):
    r, trees = custody(tmp_path, 2)
    prune.prune(r.host)  # room above the floor: nothing is freed
    old, _ = publish(r, "u0", extra=12000, at=1)  # room for the overage and the reading's own records
    mid, _ = publish(r, "u1", extra=4000, at=2)
    new, _ = publish(r, "u2", extra=8000, at=3)
    over_by(r, 500)  # u2's install pushed the run over; u0 alone gets it back under, with room to read
    r.host.issued["scored"] = 9  # past the rig's own frozen copies k0, k1
    # disk bytes, as the record counts them: 4000 where blocks are allocated, less where the filesystem holds a
    # small file in its inode (GPFS)
    floor = min(4000, (old / "model.safetensors").stat().st_blocks * 512)
    reply = r.call("kl", "u2")
    assert reply.exit == 0, reply.stderr  # the request's own pass freed u0, then admitted the reading
    assert (
        not (old / "model.safetensors").exists()
        and not (old.parent / "pruned" / "model.safetensors").exists()
    )
    record = json.loads((old.parent / prune.RECORD).read_text())
    assert record["reason"] == prune.AT_CAP and record["bytes"] >= floor
    assert (mid / "model.safetensors").is_file()  # freeing stopped once back under the cap
    assert (trees[0] / "model.safetensors").is_file() and (new / "model.safetensors").is_file()
    assert prune.over_cap(r.host) is None
    r.settle()
    assert (mid / "model.safetensors").is_file() and prune.over_cap(r.host) is None
    scored = next(e for e in r.host.entries if e.src == "u2")
    assert scored.valid is True and scored.metrics["kl"] == pytest.approx(0.0712)
    reply = r.call("handin", "u0")
    assert reply.exit == 2 and "disk cap" in reply.stderr and "not scored yet" in reply.stderr


def test_at_the_cap_a_build_with_a_reading_in_flight_is_not_freed(tmp_path):
    r, _ = custody(tmp_path, 1)
    publish(r, "u0", extra=1000, at=1)
    publish(r, "u1", extra=1000, at=2)
    publish(r, "u2", at=3)
    queued = Entry(
        id="k9",
        kind="scored",
        src="u0",
        dir=r.host.jdir / "frozen" / "k9",
        digest="d",
        submitted=r.clock.now(),
        build_receipt=json.loads((r.host.jdir / "builds" / "u0" / "receipt.json").read_text()),
    )
    r.host.entries.append(queued)  # a reading of u0 is queued
    over_by(r, 500)
    records = prune.prune(r.host)
    assert [rec["destination"] for rec in records] == ["u1"]  # the next oldest goes instead, in the same pass
    assert (r.host.jdir / "builds" / "u0" / "published" / "model.safetensors").is_file()


def test_at_the_cap_nothing_protected_is_freed_and_the_refusal_names_the_holders(tmp_path, caplog):
    r, trees = custody(tmp_path, 1)  # b0: the best
    publish(r, "base", extra=1000, at=1)  # candidate 0
    r.host.cfg.baselines = [r.work / "base"]
    _, receipt = publish(r, "handed", extra=1000, at=2)
    handed = r.host.jdir / "frozen" / "h9"
    handed.mkdir()
    r.host.entries.append(
        Entry(
            id="h9",
            kind="handin",
            src="handed",
            dir=handed,
            digest="d",
            submitted=r.clock.now(),
            build_receipt=receipt,
        )
    )  # its check is queued
    publish(r, "spare", extra=100, at=3)
    newest, _ = publish(r, "newest", extra=1000, at=4)
    over_by(r, 2000)  # freeing `spare` alone cannot get the run under: nothing goes
    before = prune.run_bytes(r.host)
    with caplog.at_level("WARNING", logger=prune.__name__):
        for _ in range(3):  # every tick prunes
            assert prune.prune(r.host) == []
    assert prune.run_bytes(r.host) == before
    assert (
        sum("would not get the run under" in rec.message for rec in caplog.records) == 1
    )  # once, not per tick
    why = prune.over_cap(r.host)
    assert why is not None and why.startswith("disk budget:")
    assert f"of {r.host.cfg.max_run_bytes / 1e9:.1f} GB" in why
    for named in (
        "b0 ",
        "(the best scored build)",
        "base ",
        "(candidate 0 or a baseline)",
        "handed ",
        "(a reading in flight)",
        "newest ",
        "spare ",
        "outside the builds",
        "judge.max_run_bytes",
    ):
        assert named in why, (named, why)
    assert (trees[0] / "model.safetensors").is_file() and (newest / "model.safetensors").is_file()


@pytest.mark.parametrize("basis", ["file", "payload"])
def test_a_llama_cpp_search_reserves_its_byte_budget_not_the_whole_anchor(tmp_path, monkeypatch, basis):
    """A 72 GB anchor with a 22 GB budget: an agent's `qsched add --checkpoint` reserves ~22 GB, not 72."""
    from test_judge_gguf import llama_rig

    original = anchor_dir(tmp_path / "original")
    (original / "tokenizer.json").write_bytes(b"t" * 4096)  # metadata a .gguf carries in its header
    budget = 22_000_000_000
    r = llama_rig(tmp_path, monkeypatch, budget=budget, anchor=original, max_copy_bytes=80_000_000_000)
    r.host.cfg.budget_basis = basis
    monkeypatch.setattr(type(r.host.cfg), "anchor_bytes", lambda self: 72_000_000_000)
    header = (
        0
        if basis == "file"
        else prune.GGUF_HEADER_MARGIN
        + sum(p.stat().st_size for p in original.iterdir() if p.suffix != ".safetensors")
    )
    assert prune.expected_bytes(r.host) == budget + header
    assert prune.expected_bytes(r.host) < 23_000_000_000
    (r.work / "quantize.sh").write_text("true\n")
    reply = r.call("qsched", "add", "quantize.sh", "--checkpoint", "out.gguf", "--expected", "1m")
    assert reply.exit == 0, reply.stderr
    (job,) = (j for j in r.host.sched.jobs.values() if j.origin == "agent")
    assert job.expected_bytes == budget + header
    r.host.cfg.max_copy_bytes = r.host.max_bytes = 10_000_000_000  # never above the largest checkpoint admitted
    assert prune.expected_bytes(r.host) == 10_000_000_000


def test_a_build_refused_at_intake_is_prunable(tmp_path, monkeypatch):
    """A kl refused at intake is a verdict on the build: its weights are rankable, not "not read yet"."""
    r, trees = custody(tmp_path, 1, readings=False, format_menu=["bf16"])
    reply = r.call("kl", "b0")
    assert reply.exit == 2 and "outside judge.format_menu" in reply.stderr, reply.stderr
    assert r.host.entries == [] and (trees[0] / "model.safetensors").is_file()
    always_short(monkeypatch)
    records = prune.prune(r.host)
    assert [rec["destination"] for rec in records] == ["b0"]
    assert not (trees[0] / "model.safetensors").exists()
    assert (trees[0].parent / "pruned" / "build.json").is_file()
    assert prune.prune(r.host) == []


def test_the_disk_refusal_names_the_builds_no_prune_pass_frees(tmp_path):
    """The bytes may sit in a build the judge never scored: name it and its size, so a jam is not a puzzle."""
    msg = prune.disk_message(1_000_000_000, 0, 0, 2_000_000_000, stuck=[("ckpt/C_realloc", 21_679_413_568)])
    assert "no prune frees them: ckpt/C_realloc (21.7 GB)" in msg
    more = prune.disk_message(1, 0, 0, 2, stuck=[(f"b{i}", 1_000_000_000) for i in range(10)])
    assert "b0 (1.0 GB)" in more and "2 more" in more and "b8 " not in more
    assert "no prune frees them" not in prune.disk_message(1, 0, 0, 2)

    r, _ = custody(tmp_path, 1)  # b0 is scored
    stuck, _ = publish(r, "stuck", extra=4000, at=1)  # nothing ever read it
    ckpt(r.work / "cand", b"weights")
    r.host.cfg.max_run_bytes = prune.run_bytes(r.host) + 16  # room for a few bytes only
    reply = r.call("kl", "cand")  # under the cap: the copy would pass it
    assert reply.exit == 2 and "disk budget:" in reply.stderr, reply.stderr
    assert "no prune frees them: stuck (" in reply.stderr
    assert (stuck / "model.safetensors").is_file()  # a refusal frees nothing

    r.host.cfg.max_run_bytes = max(1, prune.run_bytes(r.host) - 10_000)  # over it, and it cannot help
    over = prune.over_cap(r.host)
    assert over is not None and "(unscored: no prune pass frees it)" in over
