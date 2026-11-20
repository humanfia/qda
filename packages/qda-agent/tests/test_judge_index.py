"""The host's calibration index: the union of the reference's eval set and its explicit holdout set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from qda_agent.judge import JudgeConfig, JudgeHost
from qda_agent.sched import FakeBackend, FakeClock, SchedConfig
from qda_quant.calib.contamination import PromptIndex, filter_rows

EVAL = "Which city is the capital of France and how many people live in it"
HELD = "Explain how photosynthesis converts light energy into chemical energy in glucose"


def _ref(root: Path, name: str, rows: list[dict]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    prompts = d / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return d


def _host(run: Path, readable=(), hide=(), **judge) -> JudgeHost:
    clock = FakeClock()
    return JudgeHost(
        run,
        SchedConfig(cards=[4], budget_s=3600, fake_gpus=True, readable=list(readable), hide=list(hide)),
        JudgeConfig(contamination="none", kl_cmd=["score"], check_cmd=None, **judge),
        clock,
        FakeBackend(clock),
        autotick=False,
    )


def _eval_ref(tmp_path: Path) -> Path:
    return _ref(tmp_path, "eval", [{"id": "r1", "prompt": EVAL}, {"id": "r2", "prompt": "A second eval row"}])


def _holdout_ref(tmp_path: Path) -> Path:
    return _ref(tmp_path, "holdout", [{"id": "h1", "prompt": HELD}])


def test_a_search_never_starts_without_an_explicit_holdout_set(tmp_path):
    with pytest.raises(ValueError, match="kl_holdout_ref is unset"):
        _host(tmp_path / "run", kl_ref=str(_eval_ref(tmp_path)))


def test_the_eval_and_holdout_sets_are_unioned_into_one_index(tmp_path):
    host = _host(
        tmp_path / "run", kl_ref=str(_eval_ref(tmp_path)), kl_holdout_ref=str(_holdout_ref(tmp_path))
    )
    index = PromptIndex.load(host.index)
    assert sorted((s.role, s.rows) for s in index.sources) == [("eval", 2), ("holdout", 1)]
    assert index.prompts == 3
    kept, dropped, _exempted = filter_rows(
        [{"id": "a", "prompt": EVAL}, {"id": "b", "prompt": HELD}, {"id": "c", "prompt": "unrelated text"}],
        index,
    )
    assert [r["id"] for r in kept] == ["c"]
    assert {d["id"] for d in dropped} == {"a", "b"}  # the holdout prompt is reference content too


def test_the_old_one_file_shape_is_partitioned_never_counted_twice(tmp_path):
    ref = _ref(
        tmp_path,
        "both",
        [{"id": "r1", "prompt": EVAL}, {"id": "h1", "prompt": HELD, "split": "holdout"}],
    )
    host = _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(ref))
    index = PromptIndex.load(host.index)
    assert sorted((s.role, s.rows) for s in index.sources) == [("eval", 1), ("holdout", 1)]
    assert index.prompts == 2


def test_an_old_shape_reference_with_no_holdout_rows_refuses(tmp_path):
    ref = _ref(tmp_path, "flat", [{"id": "r1", "prompt": EVAL}])
    with pytest.raises(ValueError, match="none of its rows carries split=holdout"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(ref))


def test_a_prebuilt_index_must_equal_the_canonical_union(tmp_path):
    """Metadata that lists a holdout source is not enough: erased holdout hashes must refuse."""
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    canonical = PromptIndex.read_holdout(holdout_ref)
    erased = PromptIndex(
        PromptIndex.read_reference(eval_ref).ids,
        PromptIndex.read_reference(eval_ref).texts,
        PromptIndex.read_reference(eval_ref).guard,
        PromptIndex.read_reference(eval_ref).regex,
        3,
        (
            *PromptIndex.read_reference(eval_ref).sources,
            *canonical.sources,  # the holdout is listed, but its hashes are not in the union
        ),
    )
    p = erased.save(tmp_path / "prebuilt.json")
    with pytest.raises(ValueError, match="not the canonical union"):
        _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref), contamination_index=p)


def test_a_prebuilt_union_that_matches_the_references_starts(tmp_path):
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    union = PromptIndex.union([PromptIndex.read_reference(eval_ref), PromptIndex.read_holdout(holdout_ref)])
    p = union.save(tmp_path / "prebuilt.json")
    host = _host(
        tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref), contamination_index=p
    )
    assert PromptIndex.load(host.index).same_content(union)


def test_a_reference_record_declaring_another_holdout_refuses(tmp_path):
    eval_ref = _eval_ref(tmp_path)
    other = _ref(tmp_path, "other", [{"id": "h9", "prompt": "a different held-out prompt entirely"}])
    (eval_ref / "selection.json").write_text(json.dumps({"holdout_dir": str(other)}))
    with pytest.raises(ValueError, match="declares the holdout set"):
        _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(_holdout_ref(tmp_path)))


def test_declared_holdout_ids_must_be_held_by_the_named_set(tmp_path):
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    (eval_ref / "selection.json").write_text(json.dumps({"ids": {"holdout": ["h1"]}}))
    _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref))  # h1 is held
    (eval_ref / "selection.json").write_text(json.dumps({"ids": {"holdout": ["not-held"]}}))
    with pytest.raises(ValueError, match="holds a different set"):
        _host(tmp_path / "run-2", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref))


def test_a_declared_holdout_digest_must_match_the_named_set(tmp_path):
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    (eval_ref / "ref-record.json").write_text(
        json.dumps({"holdout": {"path": str(holdout_ref), "sha256": hashlib.sha256(b"other").hexdigest()}})
    )
    with pytest.raises(ValueError, match="does not match the digest"):
        _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref))


def test_a_split_label_or_a_count_map_in_the_record_declares_no_path(tmp_path):
    """``split_of_ids.holdout`` is a label and ``holdout_source`` counts: neither is a holdout path."""
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    (eval_ref / "selection.json").write_text(
        json.dumps(
            {
                "split_of_ids": {"holdout": "holdout"},
                "holdout_source": {"math": 2, "code": 1},
                "ids": {"holdout": ["h1"]},
            }
        )
    )
    _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref))  # starts, no false refusal


def test_a_prebuilt_index_without_configured_references_refuses(tmp_path):
    """A prebuilt index is not provenance: with no references to derive from, forged arrays cannot be checked."""
    from qda_quant.calib.contamination import Source

    index = PromptIndex(
        frozenset(),
        frozenset(hashlib.sha256(b"decoy").hexdigest() for _ in range(2)),
        frozenset(),
        frozenset(),
        2,
        (
            Source(
                "eval", "elsewhere", hashlib.sha256(b"e").hexdigest(), 1, hashlib.sha256(b"r").hexdigest()
            ),
            Source(
                "holdout", "elsewhere", hashlib.sha256(b"h").hexdigest(), 1, hashlib.sha256(b"s").hexdigest()
            ),
        ),
    )
    p = index.save(tmp_path / "forged.json")
    with pytest.raises(ValueError, match="prebuilt index is not provenance"):
        _host(tmp_path / "run", contamination_index=p)


def test_declared_holdout_ids_must_equal_the_named_set(tmp_path):
    """A declared id the set lacks or a held id the record does not declare both mean the wrong set."""
    eval_ref, holdout_ref = _eval_ref(tmp_path), _holdout_ref(tmp_path)
    (eval_ref / "selection.json").write_text(json.dumps({"ids": {"holdout": ["h1"]}}))
    holdout_file = holdout_ref / "prompts.jsonl"
    holdout_file.write_text(
        holdout_file.read_text() + json.dumps({"id": "h2", "prompt": "an extra held-out prompt"}) + "\n"
    )
    with pytest.raises(ValueError, match="holds a different set"):
        _host(tmp_path / "run", kl_ref=str(eval_ref), kl_holdout_ref=str(holdout_ref))


def test_a_reference_row_with_punctuation_only_text_refuses_at_start(tmp_path):
    """Every reference row must carry non-empty normalized text; a punctuation-only prompt is refused."""
    ref = _ref(tmp_path, "punctuation", [{"id": "r1", "prompt": "!!! ... ???"}])
    with pytest.raises(ValueError, match="normalized text is empty"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(_holdout_ref(tmp_path)))


def _shared(tmp_path: Path, *, digest: bool = True) -> tuple[Path, Path, Path]:
    """A shared-corpus layout: two references beside one corpus, their prompt paths recorded relatively."""
    root = tmp_path / "c2refs"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    refs = []
    for name, rows in (("eval", [{"id": "r1", "prompt": EVAL}]), ("holdout", [{"id": "h1", "prompt": HELD}])):
        d = root / f"{name}-ref"
        d.mkdir()
        prompts = corpus / f"klv2-{name}.jsonl"
        prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
        build = {"prompts": f"corpus/{prompts.name}"}
        if digest:
            build["prompts_sha256"] = hashlib.sha256(prompts.read_bytes()).hexdigest()
        (d / "manifest.json").write_text(json.dumps({"build": build}))
        refs.append(d)
    return refs[0], refs[1], corpus


def test_a_shared_corpus_beside_the_references_resolves_from_the_parent(tmp_path):
    ref, hold, corpus = _shared(tmp_path)
    host = _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold))
    index = PromptIndex.load(host.index)
    assert sorted((s.role, s.rows) for s in index.sources) == [("eval", 1), ("holdout", 1)]
    # The manifest records the file actually used, and the base the recorded path resolved from.
    assert {Path(s.path).parent for s in index.sources} == {corpus}
    assert {s.base for s in index.sources} == {"parent"}


def test_the_base_a_prompt_path_resolved_from_is_recorded(tmp_path):
    absolute = _ref(tmp_path, "absolute", [{"id": "r1", "prompt": EVAL}])
    assert {s.base for s in PromptIndex.read_reference(absolute).sources} == {"absolute"}
    own = tmp_path / "own"
    own.mkdir()
    prompts = own / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "r1", "prompt": EVAL}) + "\n")
    (own / "manifest.json").write_text(
        json.dumps(
            {
                "build": {
                    "prompts": prompts.name,
                    "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                }
            }
        )
    )
    assert {s.base for s in PromptIndex.read_reference(own).sources} == {"reference"}
    assert {s.base for s in PromptIndex.read_holdout(prompts).sources} == {"file"}


def test_a_relative_record_may_not_escape_the_reference_group(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text(json.dumps({"id": "r1", "prompt": EVAL}) + "\n")
    ref = tmp_path / "c2refs" / "ref-MODEL" / "eval"
    ref.mkdir(parents=True)
    _record(ref, "../../../outside.jsonl", hashlib.sha256(outside.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="may not escape"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(_holdout_ref(tmp_path)))


def test_a_relative_record_without_a_digest_may_not_leave_the_reference_dir(tmp_path):
    """Beyond the split directory the recorded sha256 is mandatory: without it the record must stay inside."""
    ref = tmp_path / "c2refs" / "eval"
    ref.mkdir(parents=True)
    _record(ref, "../corpus.jsonl")
    with pytest.raises(ValueError, match="must record the file digest"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(_holdout_ref(tmp_path)))


def test_a_parent_candidate_with_the_wrong_digest_refuses(tmp_path):
    ref, hold, corpus = _shared(tmp_path)
    (ref / "manifest.json").write_text(
        json.dumps(
            {
                "build": {
                    "prompts": "corpus/klv2-eval.jsonl",
                    "prompts_sha256": hashlib.sha256(b"x").hexdigest(),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="matches no candidate"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold))


def test_a_relative_record_without_a_digest_never_tries_the_parent(tmp_path):
    ref, hold, _corpus = _shared(tmp_path, digest=False)
    with pytest.raises(ValueError, match="cannot read"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold))


def test_a_selected_prompt_file_the_agent_can_read_refuses(tmp_path):
    ref, hold, corpus = _shared(tmp_path)
    with pytest.raises(ValueError, match="prompt file"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold), readable=[corpus])


def test_a_selected_prompt_file_reached_by_a_bind_refuses(tmp_path):
    ref, hold, corpus = _shared(tmp_path)
    clock = FakeClock()
    with pytest.raises(ValueError, match="prompt file"):
        JudgeHost(
            tmp_path / "run",
            SchedConfig(cards=[4], budget_s=3600, fake_gpus=True, binds={Path("/data"): corpus}),
            JudgeConfig(
                contamination="none",
                kl_cmd=["score"],
                check_cmd=None,
                kl_ref=str(ref),
                kl_holdout_ref=str(hold),
            ),
            clock,
            FakeBackend(clock),
            autotick=False,
        )


def test_a_hidden_shared_corpus_starts(tmp_path):
    ref, hold, corpus = _shared(tmp_path)
    host = _host(
        tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold), readable=[corpus], hide=[corpus]
    )
    assert PromptIndex.load(host.index).prompts == 2


def test_distinct_parent_corpora_are_bound_to_each_references_hash(tmp_path):
    """The shared filename never identifies a corpus: each reference must match its recorded hash."""
    refs = []
    paths = []
    bodies = [
        json.dumps({"id": "eval", "prompt": "Explain ocean circulation in six words."}) + "\n",
        json.dumps({"id": "eval", "prompt": "Describe winter snowfall on mountain slopes."}) + "\n",
    ]
    for name, body in zip(("one", "two"), bodies, strict=True):
        parent = tmp_path / name
        ref = parent / "eval"
        corpus = parent / "corpus" / "klv2-eval.jsonl"
        ref.mkdir(parents=True)
        corpus.parent.mkdir()
        corpus.write_text(body)
        (ref / "manifest.json").write_text(
            json.dumps(
                {
                    "build": {
                        "prompts": "corpus/klv2-eval.jsonl",
                        "prompts_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    }
                }
            )
        )
        refs.append(ref)
        paths.append(corpus)
    for i, ref in enumerate(refs):
        index = PromptIndex.read_reference(ref)
        assert index.sources[0].base == "parent" and Path(index.sources[0].path) == paths[i]
        assert index.hit({"prompt": json.loads(bodies[i])["prompt"]})
        assert index.hit({"prompt": json.loads(bodies[1 - i])["prompt"]}) is None
    # Each matching file still exists under the other parent after the swap. Neither may be cross-picked.
    for i, corpus in enumerate(paths):
        corpus.write_text(bodies[1 - i])
    from qda_quant.step import Refused

    for ref in refs:
        with pytest.raises(Refused, match="matches no candidate"):
            PromptIndex.read_reference(ref)


def _record(ref: Path, prompts: str, digest: str | None = None) -> None:
    """Write a reference manifest recording `prompts` relatively, with the digest when given."""
    build = {"prompts": prompts} | ({"prompts_sha256": digest} if digest else {})
    (ref / "manifest.json").write_text(json.dumps({"build": build}))


def _c2refs(tmp_path: Path, *, group: str = "c2refs", rows=(EVAL, HELD)) -> tuple[Path, Path, Path]:
    """The real layout: one corpus in a group dir, a model dir beside it holding the eval and holdout refs."""
    root = tmp_path / group
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    for name, prompt in zip(("eval", "holdout"), rows, strict=True):
        p = corpus / f"klv2-{name}.jsonl"
        p.write_text(json.dumps({"id": name[0], "prompt": prompt}) + "\n")
        d = root / "ref-MODEL" / name
        d.mkdir(parents=True)
        _record(d, f"corpus/{p.name}", hashlib.sha256(p.read_bytes()).hexdigest())
    return root / "ref-MODEL" / "eval", root / "ref-MODEL" / "holdout", corpus


def test_the_c2refs_layout_resolves_from_the_group_dir(tmp_path):
    ref, hold, corpus = _c2refs(tmp_path)
    index = PromptIndex.load(_host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold)).index)
    assert {Path(s.path) for s in index.sources} == {
        corpus / "klv2-eval.jsonl",
        corpus / "klv2-holdout.jsonl",
    }
    assert {s.base for s in index.sources} == {"group"} and index.hit({"prompt": EVAL})
    _record(
        ref, "corpus/klv2-eval.jsonl", hashlib.sha256(b"x").hexdigest()
    )  # a group-only candidate needs the hash
    with pytest.raises(ValueError, match="matches no candidate"):
        _host(tmp_path / "run-2", kl_ref=str(ref), kl_holdout_ref=str(hold))


def test_two_sibling_groups_never_cross_pick_their_corpora(tmp_path):
    one, two = "Explain ocean circulation in six words.", "Describe winter snowfall on mountain slopes."
    ra, _, ca = _c2refs(tmp_path / "a", group="g", rows=(one, "first held prompt"))
    rb, _, cb = _c2refs(tmp_path / "b", group="g", rows=(two, "second held prompt"))
    for ref, corpus, own, other in ((ra, ca, one, two), (rb, cb, two, one)):
        index = PromptIndex.read_reference(ref)
        assert Path(index.sources[0].path) == corpus / "klv2-eval.jsonl" and index.hit({"prompt": own})
        assert index.hit({"prompt": other}) is None
    (ca / "klv2-eval.jsonl").write_text(json.dumps({"id": "e", "prompt": two}) + "\n")
    (cb / "klv2-eval.jsonl").write_text(json.dumps({"id": "e", "prompt": one}) + "\n")
    from qda_quant.step import Refused

    for ref in (ra, rb):  # neither group's record may be satisfied by the other group's corpus
        with pytest.raises(Refused, match="matches no candidate"):
            PromptIndex.read_reference(ref)


def test_one_reference_dir_beside_a_shared_corpus_resolves_from_its_parent(tmp_path):
    """The other real shape: one reference dir at the top of the group, its corpus beside it (several is tested above)."""
    group = tmp_path / "group"
    p = group / "corpus" / "klv2-eval.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"id": "r1", "prompt": EVAL}) + "\n")
    (ref := group / "ref").mkdir()
    _record(ref, "corpus/klv2-eval.jsonl", hashlib.sha256(p.read_bytes()).hexdigest())
    assert (s := PromptIndex.read_reference(ref).sources[0]).base == "parent" and Path(s.path) == p


def test_a_group_resolved_corpus_the_agent_can_read_refuses_unless_hidden(tmp_path):
    ref, hold, corpus = _c2refs(tmp_path)
    with pytest.raises(ValueError, match="prompt file"):
        _host(tmp_path / "run", kl_ref=str(ref), kl_holdout_ref=str(hold), readable=[corpus])
    host = _host(
        tmp_path / "run-2", kl_ref=str(ref), kl_holdout_ref=str(hold), readable=[corpus], hide=[corpus]
    )
    assert PromptIndex.load(host.index).prompts == 2


def _corpus(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "corpus" / "klv2-corpus-v2.jsonl"
    p.parent.mkdir()
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _subset(tmp_path: Path, name: str, row: dict, corpus: Path) -> Path:
    """A reference holding one row of a corpus its selection record names, as M5's 96-prompt files do."""
    d = _ref(tmp_path, name, [row])
    (d / "selection.json").write_text(
        json.dumps(
            {
                "corpus": {
                    "path": str(corpus),
                    "rows": len(corpus.read_text().splitlines()),
                    "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                }
            }
        )
    )
    return d


def test_a_subset_reference_pair_is_indexed_from_the_full_parent_corpus(tmp_path):
    """A 96-prompt eval and holdout file leave the rest of the corpus unfiltered: its full splits are indexed."""
    extra_eval = "A different eval prompt about ocean currents and the tides"
    extra_held = "Another held-out prompt about mountain snowfall"
    corpus = _corpus(
        tmp_path,
        [
            {"id": "r1", "prompt": EVAL, "split": "eval"},
            {"id": "r2", "prompt": extra_eval, "split": "eval"},
            {"id": "h1", "prompt": HELD, "split": "holdout"},
            {"id": "h2", "prompt": extra_held, "split": "holdout"},
        ],
    )
    ev = _subset(tmp_path, "eval", {"id": "r1", "prompt": EVAL}, corpus)
    ho = _subset(tmp_path, "holdout", {"id": "h1", "prompt": HELD}, corpus)
    index = PromptIndex.load(_host(tmp_path / "run", kl_ref=str(ev), kl_holdout_ref=str(ho)).index)
    assert sorted((s.role, s.rows, s.base) for s in index.sources) == [
        ("eval", 2, "selection"),
        ("holdout", 2, "selection"),
    ]
    assert {Path(s.path) for s in index.sources} == {corpus} and index.prompts == 4
    kept, dropped, _ = filter_rows(
        [{"id": "x", "prompt": extra_eval}, {"id": "y", "prompt": "unrelated"}], index
    )
    assert [r["id"] for r in kept] == ["y"] and dropped[0]["id"] == "x"


def test_a_subset_record_with_a_missing_parent_corpus_refuses_at_start(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": EVAL, "split": "eval"}])
    ev = _subset(tmp_path, "eval", {"id": "r1", "prompt": EVAL}, corpus)
    hold = _holdout_ref(tmp_path)
    corpus.unlink()
    with pytest.raises(ValueError, match="which is missing") as refused:
        _host(tmp_path / "run", kl_ref=str(ev), kl_holdout_ref=str(hold))
    assert "qda-calibration-contamination" in str(refused.value)  # the refusal names the scanner


def test_a_subset_record_whose_corpus_changed_refuses_at_start(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": EVAL, "split": "eval"}])
    ev = _subset(tmp_path, "eval", {"id": "r1", "prompt": EVAL}, corpus)
    corpus.write_text(corpus.read_text() + json.dumps({"id": "r2", "prompt": "a later row"}) + "\n")
    with pytest.raises(ValueError, match="the corpus changed"):
        _host(tmp_path / "run", kl_ref=str(ev), kl_holdout_ref=str(_holdout_ref(tmp_path)))


def test_a_record_that_does_not_hold_the_references_ids_refuses_at_start(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": EVAL, "split": "eval"}])
    ev = _subset(tmp_path, "eval", {"id": "r1", "prompt": EVAL}, corpus)
    body = json.dumps({"id": "other", "prompt": "a prompt from another corpus", "split": "eval"}) + "\n"
    corpus.write_text(body)
    (ev / "selection.json").write_text(
        json.dumps(
            {
                "corpus": {
                    "path": str(corpus),
                    "rows": 1,
                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="does not describe this reference"):
        _host(tmp_path / "run", kl_ref=str(ev), kl_holdout_ref=str(_holdout_ref(tmp_path)))
