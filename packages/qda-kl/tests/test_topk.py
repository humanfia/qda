import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qda_kl.store import (
    Manifest,
    Reference,
    ReferenceWriter,
    Sequence,
    _content_sha256,
    _legacy_content_sha256,
    write_reduced,
)
from qda_kl.tally import SeqSum, reading
from qda_kl.topk import TopK, anchor_topk, kl_exact, kl_topk, log_softmax
from qda_quant.step import Refused


def pair(t=64, v=1000, noise=0.3, seed=0, temp=3.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(t, v, generator=g) * temp
    c = a + noise * torch.randn(t, v, generator=g)
    return log_softmax(a), log_softmax(c)


@pytest.mark.parametrize("k", [1, 8, 128, 999])
def test_topk_is_lower_bound(k):
    a, c = pair()
    est, exact = kl_topk(anchor_topk(a, k), c), kl_exact(a, c)
    assert (est <= exact + 1e-5).all()
    assert (est >= -1e-6).all()


def test_topk_equals_exact_at_full_vocab():
    a, c = pair(v=300)
    est, exact = kl_topk(anchor_topk(a, 300), c), kl_exact(a, c)
    torch.testing.assert_close(est, exact, rtol=1e-5, atol=1e-6)
    assert torch.isneginf(anchor_topk(a, 300).log_rest).all()


def test_topk_close_on_peaked_distributions():
    # LLM-like: most mass in the top-128, the estimate within 1% of the exact KL
    a, c = pair(t=256, v=4000, temp=6.0, noise=0.2)
    est, exact = kl_topk(anchor_topk(a, 128), c).sum(), kl_exact(a, c).sum()
    assert 0.99 * exact <= est <= exact


def test_identical_is_zero():
    a, _ = pair()
    assert kl_topk(anchor_topk(a, 16), a).abs().max() < 1e-6


DOMAINS = [
    ("math", "answer", 20, "hard", "eval"),
    ("code", "answer", 31, "hard", "eval"),
    ("math", "answer", 12, "hard", "holdout"),
    ("web", "text", 40, "easy", "eval"),
    ("web", "text", 9, "easy", "eval"),
    ("books", "text", 25, "easy", "holdout"),
]


def write_ref(path, k=16, v=200, seqs=DOMAINS, tokenizer=None):
    """A reference of random anchors; returns its manifest and, per id, (tokens, anchor log-probs, candidate)."""
    w = ReferenceWriter(path, k=k, vocab=v, model={"name": "tiny"}, tokenizer=tokenizer)
    got, rng = {}, np.random.default_rng(0)
    for i, (dom, kind, n, group, split) in enumerate(seqs):
        start = 5
        toks = rng.integers(0, v, n).tolist()
        a, c = pair(t=n - start, v=v, seed=i)
        got[f"s{i}"] = (toks, a, c)
        w.add(f"s{i}", dom, toks, start, anchor_topk(a, k), group=group, split=split, kind=kind)
    return w.close(), got


def test_store_round_trip(tmp_path):
    m, got = write_ref(tmp_path / "ref")
    ref = Reference(tmp_path / "ref", verify=True)
    assert ref.manifest == m
    assert [s.id for s in ref.sequences] == list(got)
    for s in ref.sequences:
        toks, a, _ = got[s.id]
        top, stored = anchor_topk(a, 16), ref.topk(s)
        assert ref.tokens(s) == toks
        assert stored.topv.dtype == torch.float32 and stored.topi.dtype == torch.int32
        for x, y in zip(stored, top, strict=True):
            assert torch.equal(x, y)
    assert m.positions == sum(s.n_scored for s in m.sequences)
    assert not list(tmp_path.glob(".*tmp*"))


@pytest.mark.parametrize("k", [16, 200])
def test_estimator_through_the_store(tmp_path, k):
    # identity -> 0, lower bound of the exact KL, == exact at k = V: on what a score actually reads
    write_ref(tmp_path / "ref", k=k)
    ref = Reference(tmp_path / "ref")
    for s in ref.sequences:
        stored = ref.topk(s)
        a, c = pair(t=s.n_scored, v=200, seed=int(s.id[1:]))
        assert kl_topk(stored, a).abs().max() < 1e-6
        est, exact = kl_topk(stored, c), kl_exact(a, c)
        assert (est <= exact + 1e-5).all()
        if k == 200:
            torch.testing.assert_close(est, exact, rtol=1e-5, atol=1e-6)


def test_an_old_reference_without_a_recorded_hash_is_read(tmp_path):
    """Legacy bytes keep their identity when a sequence schema gains a default; non-default labels matter."""
    manifest, _ = write_ref(tmp_path / "ref")
    path = tmp_path / "ref" / "manifest.json"
    old = json.loads(path.read_text())
    del old["content_sha256"]
    path.write_text(json.dumps(old, indent=1))
    before = path.read_bytes()
    ref = Reference(tmp_path / "ref", verify=True)
    assert ref.content_sha256 == manifest.content_sha256
    assert path.read_bytes() == before

    class LabeledSequence(Sequence):
        label: str = ""

    class LabeledManifest(Manifest):
        sequences: list[LabeledSequence]

    upgraded = LabeledManifest.model_validate(ref.manifest.model_dump())
    assert _content_sha256(upgraded) == ref.content_sha256
    upgraded.sequences[0].label = "hard"
    assert _content_sha256(upgraded) != ref.content_sha256


def test_the_identity_hashes_the_data_not_the_clock(tmp_path, monkeypatch):
    """Two builds of the same arrays are one reference, whenever each ran and however long it took."""
    stamps = iter(("2000-01-01T00:00:00", "2000-01-02T03:04:05"))
    monkeypatch.setattr("qda_kl.store.time.strftime", lambda *_: next(stamps))
    first, _ = write_ref(tmp_path / "a")
    second, _ = write_ref(tmp_path / "b")
    assert first.build["written_utc"] != second.build["written_utc"]
    assert first.content_sha256 == second.content_sha256
    assert Reference(tmp_path / "b", verify=True).content_sha256 == first.content_sha256
    timed = first.model_copy(update={"build": {**first.build, "seconds": {"generate": 1.0}}})
    assert _content_sha256(timed) == first.content_sha256


def test_a_reference_recorded_before_the_data_identity_still_opens(tmp_path):
    """The old identity hashed the build record too: a manifest recording it is read, and still checked."""
    manifest, _ = write_ref(tmp_path / "ref")
    path = tmp_path / "ref" / "manifest.json"
    legacy = _legacy_content_sha256(manifest)
    recorded = json.loads(path.read_text())
    recorded["content_sha256"] = legacy
    path.write_text(json.dumps(recorded, indent=1))
    assert Reference(tmp_path / "ref", verify=True).content_sha256 == legacy

    recorded["positions"] += 1
    path.write_text(json.dumps(recorded, indent=1))
    with pytest.raises(ValueError, match="content sha256"):
        Reference(tmp_path / "ref")


def test_store_refuses_a_manifest_whose_content_changed(tmp_path):
    write_ref(tmp_path / "ref")
    path = tmp_path / "ref" / "manifest.json"
    changed = json.loads(path.read_text())
    changed["positions"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="content sha256") as refused:
        Reference(tmp_path / "ref")
    assert changed["content_sha256"] in str(refused.value)
    assert _content_sha256(Manifest.model_validate(changed)) in str(refused.value)
    assert "re-hash or rebuild" in str(refused.value)


def test_store_detects_tampering(tmp_path):
    write_ref(tmp_path / "ref")
    arr = np.load(tmp_path / "ref" / "topi.npy")
    arr[0, 0] += 1
    np.save(tmp_path / "ref" / "topi.npy", arr)
    with pytest.raises(ValueError, match="sha256"):
        Reference(tmp_path / "ref", verify=True)


def tokenizer_files(path):
    """A tokenizer directory of the shape a reference copies: files, and one named-template directory."""
    (path / "additional_chat_templates").mkdir(parents=True)
    (path / "tokenizer.json").write_text('{"model": {}}')
    (path / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')
    (path / "additional_chat_templates" / "tool.jinja").write_text("{{ messages }}")
    return path


def test_a_reference_holds_and_verifies_its_tokenizer(tmp_path):
    """The copy is inside the reference, hashed in the manifest, and part of its identity."""
    source = tokenizer_files(tmp_path / "src")
    m, _ = write_ref(tmp_path / "ref", tokenizer=source)
    ref = Reference(tmp_path / "ref", verify=True)
    assert ref.tokenizer == tmp_path / "ref" / "tokenizer"
    assert (ref.tokenizer / "tokenizer.json").read_text() == '{"model": {}}'
    assert (ref.tokenizer / "additional_chat_templates" / "tool.jinja").is_file()
    assert set(m.files) > {"tokenizer/tokenizer.json", "tokenizer/additional_chat_templates/tool.jinja"}
    plain, _ = write_ref(tmp_path / "bare")
    assert Reference(tmp_path / "bare").tokenizer is None
    assert plain.content_sha256 != m.content_sha256  # the files it holds are its identity
    (ref.tokenizer / "tokenizer.json").write_text('{"model": {"changed": 1}}')
    with pytest.raises(ValueError, match="sha256"):
        Reference(tmp_path / "ref", verify=True)


def test_a_reduced_reference_reads_the_tokenizer_of_the_one_it_narrows(tmp_path):
    """`qda-agent probe` narrows a reference to a few sequences; the narrowed one is self-contained too."""
    write_ref(tmp_path / "ref", tokenizer=tokenizer_files(tmp_path / "src"))
    ref = Reference(tmp_path / "ref")
    out = write_reduced(ref, ref.sequences[:2], tmp_path / "few")
    assert Reference(out, verify=True).tokenizer == out / "tokenizer"


def test_store_never_replaces_a_reference_with_pinned_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("qda_kl.store.time.strftime", lambda *_: "2000-01-01T00:00:00")
    writer = ReferenceWriter(tmp_path / "ref", k=2, vocab=4)
    zeros = torch.zeros((1, 2))
    writer.add("golden", "math", [0, 1], 1, TopK(zeros, zeros.int(), zeros[0, :1]))
    m = writer.close()
    assert m.content_sha256 == "88e9653560e6ad06c96a8e9d76235575cc5d3ca686fec113e969dce21208f561"
    with pytest.raises(Refused, match="writes only a new directory"):
        write_ref(tmp_path / "ref", seqs=DOMAINS[:2])
    assert Reference(tmp_path / "ref", verify=True).manifest == m
    assert not list(tmp_path.glob(".*tmp*"))


def test_store_failed_publish_leaves_nothing(tmp_path, monkeypatch):
    def fail(self, target):
        raise OSError("injected rename failure")

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="injected"):
        write_ref(tmp_path / "ref")
    assert not (tmp_path / "ref").exists()
    assert not list(tmp_path.glob(".*tmp*"))


def test_store_failed_array_write_leaves_nothing(tmp_path, monkeypatch):
    real, calls = np.save, []

    def save(path, arr):
        calls.append(path)
        if len(calls) == 2:  # tokens.npy written, then the disk fills
            raise OSError("injected write failure")
        real(path, arr)

    monkeypatch.setattr(np, "save", save)
    with pytest.raises(OSError, match="injected"):
        write_ref(tmp_path / "ref")
    assert len(calls) == 2
    assert not (tmp_path / "ref").exists()
    assert not list(tmp_path.glob(".*tmp*"))


def test_reading_weights_tokens_and_splits_text(tmp_path):
    m, _ = write_ref(tmp_path / "ref")
    s = {x.id: x for x in m.sequences}
    rate = {x.id: 0.1 * (i + 1) for i, x in enumerate(m.sequences)}
    sums = {i: SeqSum(kl=rate[i] * x.n_scored, tokens=x.n_scored) for i, x in s.items()}
    r = reading(m.sequences, sums)

    def mean(ids):
        return sum(rate[i] * s[i].n_scored for i in ids) / sum(s[i].n_scored for i in ids)

    assert r.kl == pytest.approx(mean(["s0", "s1", "s2"]))
    assert r.tokens == sum(s[i].n_scored for i in ("s0", "s1", "s2"))
    assert r.domains == pytest.approx({"code": 0.2, "math": mean(["s0", "s2"])})
    assert r.kl_text == pytest.approx(mean(["s3", "s4", "s5"]))
    assert r.domains_text == pytest.approx({"books": 0.6, "web": mean(["s3", "s4"])})
    assert r.domain_text_tokens == {"books": s["s5"].n_scored, "web": s["s3"].n_scored + s["s4"].n_scored}
    assert r.per_seq["s0"].kl == pytest.approx(rate["s0"])
    assert (r.per_seq["s0"].tokens, r.per_seq["s0"].group, r.per_seq["s2"].split) == (
        s["s0"].n_scored,
        "hard",
        "holdout",
    )
    assert (r.per_seq["s3"].kind, r.per_seq["s3"].split) == ("text", "eval")
    with pytest.raises(ValueError, match="not scored"):
        reading(m.sequences, {k: v for k, v in sums.items() if k != "s1"})
