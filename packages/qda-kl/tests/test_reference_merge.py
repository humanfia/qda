"""Shard merge preserves scored data and refuses incompatible or incomplete inputs."""

import json

import pytest
import torch
from qda_kl.cli import main
from qda_kl.reference_merge import merge
from qda_kl.store import Reference, ReferenceWriter, _sha256
from qda_kl.topk import anchor_topk


def shard(path, prompt, seed=17, **extra):
    facts = {"seed": seed, "prompts": str(path), "dropped_short": [str(path)], **extra}
    writer = ReferenceWriter(path, 2, 4, model={"name": "tiny"}, build=facts)
    logits = torch.arange(12.0).reshape(3, 4) * (1 if prompt == "a" else 2)
    anchor = anchor_topk(torch.log_softmax(logits, -1), 2)
    tokens = [0, 1, 2, 3] if prompt == "a" else [3, 2, 1, 0]
    for draw in range(2):
        writer.add(f"{prompt}#{draw}", "math", tokens, 1, anchor, group="hard", split="eval", finish="length")
    writer.close()
    return path


def test_merge_verified_data_and_corpus_order(tmp_path):
    a, b = shard(tmp_path / "a", "a"), shard(tmp_path / "b", "b")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"id": p, "prompt": "question"}) + "\n" for p in ("b", "a")))
    out = tmp_path / "merged"
    assert main(["ref", "merge", str(out), str(a), str(b), "--corpus", str(corpus)]) == 0
    merged = Reference(out, verify=True)
    assert [s.id for s in merged.sequences] == ["b#0", "b#1", "a#0", "a#1"]
    assert merged.manifest.positions == 12
    assert merged.manifest.build["prompts_sha256"] == _sha256(corpus)
    for path in (a, b):
        ref = Reference(path)
        assert merged.manifest.build["shards"][ref.content_sha256]["dropped_short"] == [str(path)]
        for s in ref.sequences:
            got = next(t for t in merged.sequences if t.id == s.id)
            assert merged.tokens(got) == ref.tokens(s)
            assert (got.domain, got.group, got.split, got.finish) == (s.domain, s.group, s.split, s.finish)
            for actual, expected in zip(merged.topk(got), ref.topk(s), strict=True):
                assert torch.equal(actual, expected)
    means = [Reference(path).manifest.tail_mass["mean"] for path in (a, b)]
    assert merged.manifest.tail_mass["mean"] == pytest.approx(sum(means) / 2)


def test_shards_of_different_sizes_merge_with_their_own_derived_batch(tmp_path):
    """A shard's derived ``max_num_seqs`` follows its answer count, so it is kept per shard, not compared."""
    a = shard(tmp_path / "a", "a", max_num_seqs=8)
    b = shard(tmp_path / "b", "b", max_num_seqs=2)
    merged = merge(tmp_path / "merged", [a, b])
    assert "max_num_seqs" not in merged.build
    assert {s["max_num_seqs"] for s in merged.build["shards"].values()} == {8, 2}


def built(path, ids, dropped=(), answers=2):
    """A tiny reference of `ids`, every draw scored, with the build's dropped draws recorded."""
    writer = ReferenceWriter(
        path, 2, 4, model={"name": "tiny"}, build={"answers": answers, "dropped_short": list(dropped)}
    )
    logits = torch.arange(12.0).reshape(3, 4)
    anchor = anchor_topk(torch.log_softmax(logits, -1), 2)
    for i in ids:
        for draw in range(answers):
            writer.add(f"{i}#{draw}", "math", [0, 1, 2, 3], 1, anchor)
    writer.close()
    return path


def without_merge_provenance(manifest):
    """The manifest with the facts a merge adds or a build stamps: what a merge must not change."""
    build = {
        k: v
        for k, v in manifest.build.items()
        if k not in ("shards", "prompts", "prompts_sha256", "written_utc")
    }
    return manifest.model_copy(update={"build": build, "content_sha256": ""})


def test_a_prompt_the_builder_dropped_is_covered_and_listed(tmp_path):
    """All draws of a prompt dropped as too short: covered, listed in corpus order, as a whole build records it."""
    a = built(tmp_path / "a", ["a"])
    b = built(tmp_path / "b", ["b"], dropped=["gone#0", "gone#1"])
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"id": p, "prompt": "q"}) + "\n" for p in ("a", "gone", "b")))

    merged = merge(tmp_path / "merged", [a, b], corpus=corpus)

    assert merged.build["dropped_short"] == ["gone#0", "gone#1"]
    assert merged.build["prompts_sha256"] == _sha256(corpus)
    whole = Reference(built(tmp_path / "whole", ["a", "b"], dropped=["gone#0", "gone#1"])).manifest
    assert without_merge_provenance(merged) == without_merge_provenance(whole)

    # one draw of a prompt dropped and the other absent is still a missing prompt
    half = built(tmp_path / "half", ["b"], dropped=["gone#0"])
    with pytest.raises(ValueError, match="corpus coverage differs: missing=\\['gone'\\]"):
        merge(tmp_path / "never", [a, half], corpus=corpus)
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("case", ["settings", "duplicate", "missing"])
def test_merge_refusals(tmp_path, case):
    error = {"settings": "build.seed", "duplicate": "duplicate prompt", "missing": "missing=.*c"}[case]
    a = shard(tmp_path / "a", "a")
    b = shard(tmp_path / "b", "a" if case == "duplicate" else "b", seed=18 if case == "settings" else 17)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"c","prompt":"question"}\n')
    with pytest.raises(ValueError, match=error):
        merge(tmp_path / "merged", [a, b], corpus=corpus if case == "missing" else None)
    assert not (tmp_path / "merged").exists()


def test_merge_takes_every_shards_answer_caps(tmp_path):
    """An answer's window cap is per shard: capped and uncapped shards merge, and the merged build records the union."""
    anchor = anchor_topk(torch.log_softmax(torch.arange(12.0).reshape(3, 4), -1), 2)

    def capped_shard(path, prompt_id, cap=None):
        build = {"answers": 2, "seed": 17, "sampling": {"max_tokens": 24576}, "dropped_short": []}
        if cap is not None:
            build["capped_answers"] = {f"{prompt_id}#{d}": cap for d in range(2)}
        writer = ReferenceWriter(path, 2, 4, model={"name": "tiny"}, build=build)
        for draw in range(2):
            writer.add(f"{prompt_id}#{draw}", "math", [0, 1, 2, 3], 1, anchor)
        writer.close()
        return path

    merged = merge(
        tmp_path / "merged", [capped_shard(tmp_path / "a", "a", 18202), capped_shard(tmp_path / "b", "b")]
    )
    assert merged.build["capped_answers"] == {"a#0": 18202, "a#1": 18202}
    assert len(Reference(tmp_path / "merged", verify=True).sequences) == 4

    plain = merge(tmp_path / "plain", [capped_shard(tmp_path / "c", "c"), capped_shard(tmp_path / "d", "d")])
    assert "capped_answers" not in plain.build


PRIVATE = {"FLASHINFER_WORKSPACE_BASE": "/tmp/ref/a", "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache"}


def test_shards_differing_only_in_private_cache_paths_merge(tmp_path):
    """A launcher's per-shard cache paths are not settings; the merged build keeps the shared keys."""
    shared = {"NCCL_NVLS_ENABLE": "0"}
    a = shard(tmp_path / "a", "a", serve_env={**PRIVATE, **shared})
    b = shard(tmp_path / "b", "b", serve_env={**PRIVATE, **shared, "TRITON_CACHE_DIR": "/tmp/t"})
    assert merge(tmp_path / "merged", [a, b]).build["serve_env"] == shared


def test_shards_differing_in_a_shared_env_key_are_refused(tmp_path):
    a = shard(tmp_path / "a", "a", serve_env={"NCCL_NVLS_ENABLE": "0"})
    b = shard(tmp_path / "b", "b", serve_env={"NCCL_NVLS_ENABLE": "1"})
    with pytest.raises(ValueError, match=r"build\.serve_env\.NCCL_NVLS_ENABLE"):
        merge(tmp_path / "merged", [a, b])
