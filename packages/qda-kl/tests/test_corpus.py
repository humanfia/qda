"""`qda-kl corpus`: an empty, unreadable or malformed guard is refused, and a corpus built without one says so."""

import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import pytest
from qda_kl import cli
from qda_kl import corpus as build
from qda_quant.step import Refused

SHIM = Path(__file__).resolve().parents[3] / "tools" / "build_kl_corpus.py"
A, B = hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()


def test_an_empty_guard_file_is_refused(tmp_path):
    empty = tmp_path / "bench.txt"
    empty.write_text("\n  \n")
    with pytest.raises(Refused) as refused:
        build.Guard(tmp_path)
    assert str(refused.value) == f"the guard file {empty} holds no windows"
    assert refused.value.fix == "give the file its window hashes"


def test_an_unreadable_guard_file_is_refused(tmp_path, monkeypatch):
    broken = tmp_path / "bench.txt"
    broken.write_text(f"{A}\n")
    real = Path.read_bytes

    def read_bytes(self):
        if self == broken:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(Refused) as refused:
        build.Guard(tmp_path)
    assert str(refused.value).startswith(f"cannot read the guard file {broken}: PermissionError")


def test_a_guard_line_that_is_no_window_hash_is_refused(tmp_path):
    """A placeholder guard (any text) would let the build claim a clean corpus it never checked."""
    (tmp_path / "bench.txt").write_text(f"{A}\n<path/to/guard>\n")
    with pytest.raises(Refused, match="is no window sha256"):
        build.Guard(tmp_path)


def test_the_builder_and_its_shim_exit_2_before_they_build(tmp_path, monkeypatch, capsys):
    """An empty guard stops the build with the contract's status for the input (2), from both entry points."""
    empty = tmp_path / "bench.txt"
    empty.write_text("")
    argv = ["--out", str(tmp_path / "out.jsonl"), "--guard", str(tmp_path)]
    assert cli.main(["corpus", *argv]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and str(empty) in err
    assert err.endswith("fix:   give the file its window hashes\n")
    monkeypatch.setattr(sys, "argv", [str(SHIM), *argv])
    with pytest.raises(SystemExit) as exit:  # the shim's `raise SystemExit(main())`
        runpy.run_path(str(SHIM), run_name="__main__")
    assert exit.value.code == 2
    assert not (tmp_path / "out.jsonl").exists()


def test_a_guard_with_windows_still_loads(tmp_path):
    (tmp_path / "bench.txt").write_text(f"{A}\n{B}\n")
    guard = build.Guard(tmp_path)
    assert guard.union == {A, B}
    assert set(guard.per_bench) == {"bench"}


ROWS = [
    {
        "id": "klv2-chat-chat-00000",
        "group": "easy",
        "domain": "chat",
        "split": "eval",
        "prompt": "hello there",
    },
    {
        "id": "klv2-chat-chat-00001",
        "group": "easy",
        "domain": "chat",
        "split": "holdout",
        "prompt": "bye now",
    },
]


def test_without_a_guard_the_manifest_says_none_and_claims_nothing(tmp_path):
    out = tmp_path / "corpus.jsonl"
    summary = build.finish(out, ROWS, None, tokenizer="tok", tokenizer_revision="r")
    manifest = json.loads((tmp_path / "corpus.manifest.json").read_text())
    assert manifest["guard"] == "none" and summary["guard"] == "none"
    assert "guard_clean" not in summary and "guard_clean" not in json.dumps(manifest)
    held = [json.loads(line) for line in (tmp_path / "corpus.holdout.jsonl").read_text().splitlines()]
    evals = [json.loads(line) for line in (tmp_path / "corpus.eval.jsonl").read_text().splitlines()]
    assert [r["id"] for r in held] == [ROWS[1]["id"]] and [r["id"] for r in evals] == [ROWS[0]["id"]]
    assert manifest["splits"]["holdout"]["rows"] == 1 and manifest["splits"]["eval"]["rows"] == 1


def test_with_a_guard_the_finished_file_is_scanned_and_recorded(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "bench.txt").write_text(f"{A}\n")
    summary = build.finish(
        tmp_path / "c.jsonl",
        ROWS,
        build.Guard(tmp_path / "g"),
        tokenizer="t",
        tokenizer_revision="r",
    )
    manifest = json.loads((tmp_path / "c.manifest.json").read_text())
    assert summary["guard_clean"] is True and manifest["guard"]["rows_dropped"] == 0
    assert manifest["guard"]["per_bench_sha256"]["bench"]


def test_search96_is_fixed_stratified_and_has_no_long_row(tmp_path):
    """The search set: 96 eval rows, no `long`, each domain by its eval share, chosen by sha256(id)."""
    rows = [
        {"id": f"klv2-{d}-x-{i:05d}", "group": "hard", "domain": d, "split": s, "prompt": f"{d} {i}"}
        for d, n in (("math", 200), ("code", 100), ("long", 40), ("chat", 30))
        for i in range(n)
        for s in ["holdout" if i % 6 == 0 else "eval"]
    ]
    picked = build.search_rows(rows)
    assert len(picked) == 96 and all(r["domain"] != "long" for r in picked)
    assert all(r["split"] == "eval" for r in picked)
    assert Counter(r["domain"] for r in picked) == {"math": 58, "code": 29, "chat": 9}
    assert build.search_rows(list(reversed(rows))) == list(
        reversed(picked)
    )  # the data decides, not the order
    build.finish(tmp_path / "c.jsonl", rows, None, tokenizer="t", tokenizer_revision="r")
    ids = (tmp_path / "c.search96.ids").read_text().split()
    assert ids == [r["id"] for r in picked]
    manifest = json.loads((tmp_path / "c.manifest.json").read_text())
    assert manifest["splits"]["search96"]["rows"] == 96 and manifest["splits"]["search96"]["ids"].endswith(
        ".ids"
    )


def test_an_unguarded_corpus_is_selected_once_and_rebuilt_byte_identical(tmp_path, monkeypatch):
    """The selection is the slow part and the same for everyone; a second build reads it from the cache
    and writes the same bytes (corpus, splits, ids and manifest), the filter counts included."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    selections = []

    def select(args, guard):
        selections.append(guard)
        build.FILTERED["math: image"] += 3
        return [dict(r) for r in ROWS]

    monkeypatch.setattr(build, "_select", select)
    files = {}
    for run in ("a", "b"):
        build.FILTERED.clear()
        out = tmp_path / run / "corpus.jsonl"
        out.parent.mkdir()
        args = build.argparse.Namespace(out=out, tokenizer="tok", tokenizer_revision="r", guard=None)
        assert build.run(args) == 0
        files[run] = {
            p.name: p.read_bytes().replace(str(out.parent).encode(), b"DIR")
            for p in sorted(out.parent.iterdir())
        }
    assert selections == [None]  # the second build read the cache
    assert files["a"] == files["b"] and "corpus.manifest.json" in files["a"]
    assert json.loads(files["b"]["corpus.manifest.json"])["other_filters"] == {"math: image": 3}


def test_the_corpus_cache_key_follows_the_code_and_the_tokenizer():
    key = build.cache_key("tok", "r")
    assert build.cache_dir(key) != build.cache_dir(build.cache_key("tok", "other"))
    assert key["code_sha256"] == hashlib.sha256(Path(build.__file__).read_bytes()).hexdigest()


def test_cached_splits_builds_once_and_refuses_a_failed_build(tmp_path, monkeypatch):
    """`cached_splits` is the one owner of a cached corpus's split paths: one build, then the cache; a failed
    build is refused instead of returning a path that does not exist."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    calls = []

    def run(args):
        calls.append(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for name in build.SPLITS:
            (args.out.parent / f"corpus.{name}.jsonl").write_text(name)
        return 0

    monkeypatch.setattr(build, "run", run)
    first = build.cached_splits(tokenizer="tok", tokenizer_revision="r")
    assert set(first) == set(build.SPLITS) and all(path.is_file() for path in first.values())
    assert build.cached_splits(tokenizer="tok", tokenizer_revision="r") == first
    assert len(calls) == 1  # the second call reads the cache
    monkeypatch.setattr(build, "run", lambda args: 1)
    from qda_quant.step import Refused

    with pytest.raises(Refused, match="corpus build failed \(exit 1\)"):
        build.cached_splits(tokenizer="other", tokenizer_revision="r")
