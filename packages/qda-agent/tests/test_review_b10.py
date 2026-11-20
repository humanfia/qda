import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from qda_agent.host import Final, HandIn
from qda_agent.judge import commands
from qda_agent.judge.final_rules import Handin, apply
from test_judge import Rig, ckpt


def test_review_real_handin_list_preserves_priority_after_intake_refusal(tmp_path):
    rig = Rig(tmp_path, check=False)
    ckpt(rig.work / "a", b"a")
    ckpt(rig.work / "b", b"b")
    reply = rig.call("handin", "missing", "a", "b")
    assert reply.exit == 2
    final = rig.host.final(generation_checks=False)
    assert final.choice is not None
    assert final.choice.dir == rig.host.entries[0].dir
    assert [entry.list_id for entry in rig.host.entries] == [1, 1]
    assert [entry.slot for entry in rig.host.entries] == [1, 2]
    assert final.chosen_by == "handin"


def test_review_latest_call_wins_when_an_older_list_freezes_last(tmp_path, monkeypatch):
    rig = Rig(tmp_path, check=False)
    ckpt(rig.work / "older", b"a")
    ckpt(rig.work / "newer", b"b")
    entered = threading.Event()
    resume = threading.Event()
    original_take = commands.take

    def delayed_take(host, kind, arg, cwd, **kwargs):
        if arg == "older":
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("the newer hand-in did not finish")
        return original_take(host, kind, arg, cwd, **kwargs)

    monkeypatch.setattr(commands, "take", delayed_take)
    with ThreadPoolExecutor(max_workers=1) as pool:
        older = pool.submit(rig.call, "handin", "older")
        try:
            assert entered.wait(10)
            newer = rig.call("handin", "newer")
            assert newer.exit == 0
        finally:
            resume.set()
        assert older.result(timeout=10).exit == 0
    final = rig.host.final(generation_checks=False)
    assert final.choice is not None
    selected = next(entry for entry in rig.host.entries if entry.dir == final.choice.dir)
    assert selected.list_id == 2, [(entry.src, entry.list_id) for entry in rig.host.entries]


def test_review_legacy_final_and_ledger_without_list_ids_resolve():
    entries = [
        HandIn.model_validate(
            {
                "dir": name,
                "submitted_at": datetime.fromtimestamp(0, UTC),
                "kind": "handin",
                "valid": valid,
            }
        )
        for name, valid in [("older", True), ("newer", True), ("refused", False)]
    ]
    final = apply([Handin()], entries, [Path("baseline")], "kl")
    assert final.choice is not None and final.choice.dir == Path("newer")
    legacy = final.model_dump(mode="json", exclude={"chosen_by"})
    replayed = Final.model_validate(legacy).choice
    assert replayed is not None and replayed.dir == Path("newer")
    assert all("handin_list" not in entry.model_dump() for entry in entries)
