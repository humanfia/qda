"""A reference named by ``refs`` itself is scored in place, never rebuilt, and guarded by the corpus."""

import json
from pathlib import Path
from typing import Any

import pytest
from qda_kl.cli import main
from qda_kl.eval.suites import KLSuite
from test_reference_merge import built, shard


def _suite(tmp_path: Path, ids: list[str], **cfg: Any) -> KLSuite:
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"id": i, "split": "eval", "prompt": "q"}) + "\n" for i in ids))
    return KLSuite(corpus={"file": {"path": str(rows)}}, refs=str(tmp_path / "frozen"), **cfg)


def test_a_frozen_reference_is_scored_in_place(tmp_path: Path, monkeypatch) -> None:
    frozen = shard(tmp_path / "frozen", "a")
    before = (frozen / "manifest.json").read_bytes()
    built: list[Any] = []
    monkeypatch.setattr("qda_kl.reference.build", lambda *a, **k: built.append(a))
    suite = _suite(tmp_path, ["a"])

    assert suite.reference_dir("model", tmp_path) == frozen
    suite.prepare("model", tmp_path)  # no lock, no build, no prompt file beside the sealed directory

    assert built == [] and (frozen / "manifest.json").read_bytes() == before
    assert not (tmp_path / "frozen.lock").exists() and not (tmp_path / "frozen.jsonl").exists()


def test_a_frozen_reference_refuses_a_different_corpus(tmp_path: Path) -> None:
    shard(tmp_path / "frozen", "a")
    with pytest.raises(ValueError, match="a frozen reference is scored on the prompts it holds"):
        _suite(tmp_path, ["b"]).prepare("model", tmp_path)


def test_a_frozen_reference_covers_a_prompt_the_builder_dropped(tmp_path: Path) -> None:
    """A prompt whose every draw the builder dropped is covered by the corpus, not scored, and not "extra"."""
    built(tmp_path / "frozen", ["kept"], dropped=["short#0", "short#1"])
    _suite(tmp_path, ["kept", "short"]).prepare("model", tmp_path)  # no refusal, nothing built
    with pytest.raises(ValueError, match="0 missing, 1 extra"):
        _suite(tmp_path, ["kept", "short", "gone"]).prepare("model", tmp_path)


def test_a_frozen_reference_scores_its_subset_from_a_wider_corpus(tmp_path: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("b\n")
    merged = [
        "ref",
        "merge",
        str(tmp_path / "frozen"),
        str(shard(tmp_path / "b", "b")),
        "--subset",
        f"s={ids}",
    ]
    assert main(merged) == 0

    _suite(tmp_path, ["a", "b"], subset="s").prepare("model", tmp_path)  # extra corpus rows are not read
    with pytest.raises(ValueError, match=r"subset 's' holds 1 prompts"):
        _suite(tmp_path, ["a"], subset="s").prepare("model", tmp_path)
