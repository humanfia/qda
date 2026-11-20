"""Corpora: plugins chosen by name, and the split that keeps an eval row out of calibration."""

import json

import pytest
from qda_quant.calib.sources import parse_source
from qda_quant.intake.corpus import CORPORA, Corpus, split
from qda_quant.step import Refused


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def prompts(n, domain="math"):
    return [{"id": f"row-{i}", "domain": domain, "prompt": f"question {i}"} for i in range(n)]


def test_the_split_is_disjoint_stable_and_pinned(tmp_path):
    rows = prompts(40) + [{"id": i, "domain": "chat", "prompt": f"question {i}"} for i in "aby"]
    path = write_jsonl(tmp_path / "c.jsonl", rows)
    corpus = CORPORA["file"](path=path)
    ev, ca = split(corpus, "eval"), split(corpus, "calib")

    assert {r["id"] for r in ev}.isdisjoint({r["id"] for r in ca})
    assert {r["id"] for r in ev} | {r["id"] for r in ca} == {r["id"] for r in rows}
    sides = {r["id"]: "eval" for r in ev} | {r["id"]: "calib" for r in ca}
    # pinned: sha256(id) alone decides the side, so these ids must not move
    assert [sides[i] for i in ("a", "b", "y")] == ["calib", "eval", "calib"]
    assert sides["row-2"] == "eval"

    shuffled = CORPORA["file"](path=write_jsonl(tmp_path / "rev.jsonl", rows[::-1]))
    assert {r["id"] for r in split(shuffled, "eval")} == {r["id"] for r in ev}


def test_holdout_is_the_eval_share_and_not_a_setting():
    class Fixed(Corpus):
        name = "fixed"
        data: list[dict]

        def rows(self):
            return self.data

    class Mostly(Corpus):
        name = "mostly"
        holdout = 0.9
        data: list[dict]

        def rows(self):
            return self.data

    class EvalOnly(Corpus):
        name = "only"
        holdout = 1.0
        data: list[dict]

        def rows(self):
            return self.data

    class CalibOnly(EvalOnly):
        holdout = 0.0

    rows = prompts(2000)
    assert abs(len(split(Fixed(data=rows), "eval")) / len(rows) - 0.5) < 0.03
    assert abs(len(split(Mostly(data=rows), "eval")) / len(rows) - 0.9) < 0.03
    assert split(EvalOnly(data=rows), "calib") == []
    assert split(CalibOnly(data=rows), "eval") == []
    with pytest.raises(ValueError, match="holdout"):  # the corpus fixes the share; a config cannot change it
        Fixed(data=rows, holdout=0.25)


def test_corpus_calib_source_reads_only_the_calib_side(tmp_path):
    path = write_jsonl(tmp_path / "c.jsonl", prompts(30))
    source = parse_source({"corpus": {"file": {"path": path}}})
    calib = source.read(tmp_path)
    ev = {r["id"] for r in split(CORPORA["file"](path=path), "eval")}

    assert {r["id"] for r in calib.rows}.isdisjoint(ev)
    assert all(r["text"] == r["prompt"] for r in calib.rows)  # a prompt is calibrated as its text
    assert "holdout" not in source.to_dict()["corpus"]["file"]
    with pytest.raises(ValueError, match="holdout"):
        parse_source({"corpus": {"file": {"path": path, "holdout": 0.25}}})


def test_a_row_split_tag_decides_its_side(tmp_path):
    rows = prompts(4) + [
        {"id": "h", "domain": "math", "split": "holdout", "prompt": "q"},
        {"id": "e", "domain": "math", "split": "eval", "prompt": "q"},
        {"id": "c", "domain": "math", "split": "calib", "prompt": "q"},
    ]
    corpus = CORPORA["file"](path=write_jsonl(tmp_path / "tagged.jsonl", rows))
    ev = {r["id"] for r in split(corpus, "eval")}
    ca = {r["id"] for r in split(corpus, "calib")}

    assert {"h", "e"} <= ev and "c" not in ev  # holdout rows stay evaluation-only
    assert "c" in ca and {"h", "e"}.isdisjoint(ca)

    bad = write_jsonl(tmp_path / "bad.jsonl", [{"id": "x", "domain": "d", "split": "test", "prompt": "q"}])
    with pytest.raises(Refused, match="split must be one of"):
        split(CORPORA["file"](path=bad), "eval")


def test_a_row_without_an_id_cannot_be_split():
    class Bad(Corpus):
        name = "bad"

        def rows(self):
            return [{"domain": "math", "prompt": "no id"}]

    with pytest.raises(Refused, match='needs a non-empty "id"'):
        split(Bad(), "eval")


@pytest.mark.parametrize("side", ["eval", "calib"])
def test_a_file_can_be_one_side_whole(tmp_path, side):
    rows = [{"id": f"row-{i}", "prompt": "q"} for i in range(3)]
    rows.append({"id": "tagged", "split": "calib" if side == "eval" else "eval", "prompt": "q"})
    corpus = CORPORA["file"](path=write_jsonl(tmp_path / "c.jsonl", rows), split=side)

    assert [r["id"] for r in split(corpus, side)] == ["row-0", "row-1", "row-2", "tagged"]
    assert split(corpus, "calib" if side == "eval" else "eval") == []
