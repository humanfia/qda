"""`qda probe`'s own reference: rows read from the caller's data, a seeded fixed subset, built once."""

import hashlib
import json

import pytest
from qda_agent import probe


def _data(tmp_path, rows):
    p = tmp_path / "own.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_rows_take_text_prompt_or_the_first_user_message(tmp_path):
    data = _data(
        tmp_path,
        [
            {"text": "plain text"},
            {"id": "p", "domain": "math", "prompt": "2+2?"},
            {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]},
        ],
    )
    assert probe._rows(data) == [
        {"id": "row-0", "domain": "own", "text": "plain text"},
        {"id": "p", "domain": "math", "prompt": "2+2?"},
        {"id": "row-2", "domain": "own", "prompt": "hi"},
    ]
    with pytest.raises(ValueError, match="no text, prompt or user message"):
        probe._rows(_data(tmp_path, [{"messages": [{"role": "assistant", "content": "x"}]}]))
    with pytest.raises(ValueError, match="row ids repeat"):
        probe._rows(_data(tmp_path, [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]))


def test_own_reference_is_a_seeded_subset_built_once(tmp_path, monkeypatch):
    data = _data(tmp_path, [{"id": f"r{i}", "text": f"t{i}"} for i in range(10)])
    calls = []

    def run(argv, log):
        calls.append(argv)
        out = argv[argv.index("--out") + 1]
        (tmp_path / "work" / out.rsplit("/", 1)[1]).mkdir(parents=True)
        (tmp_path / "work" / out.rsplit("/", 1)[1] / "manifest.json").write_text("{}")
        return 0.0

    monkeypatch.setattr(probe, "_run", run)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    (tmp_path / "model" / "generation_config.json").write_text('{"temperature": 0.6}')
    work = tmp_path / "work"
    a = probe.own_reference(tmp_path / "model", data, 4, work, 2048)
    assert probe.own_reference(tmp_path / "model", data, 4, work, 2048) == a and len(calls) == 1
    chosen = [json.loads(x)["id"] for x in (work / f"{a.name}.prompts.jsonl").read_text().splitlines()]
    seeded = sorted(
        (f"r{i}" for i in range(10)), key=lambda r: hashlib.sha256(f"{probe.SEED}:{r}".encode()).hexdigest()
    )
    assert chosen == seeded[:4]
    assert calls[0][2:4] == ["qda_kl.cli", "ref"] and calls[0][calls[0].index("--answers") + 1] == "1"
    assert (
        probe.own_reference(tmp_path / "model", data, 5, work, 2048) != a
    )  # another size, another reference
    with pytest.raises(ValueError, match="holds 10 rows"):
        probe.own_reference(tmp_path / "model", data, 11, work, 2048)


def test_probe_takes_exactly_one_reference_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one of --ref and --data"):
        probe.probe({"format": "w4a16-sym-g128"}, tmp_path, None, 4, tmp_path / "w")


def _fake_reference(path, sha):
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"content_sha256": sha}))
    return path


@pytest.fixture
def fake_store(monkeypatch):
    """qda_kl.store without arrays: a reference is its manifest's content hash; a reduced one is written as such."""
    import qda_kl.store as store

    class Ref:
        def __init__(self, path):
            self.manifest = type("M", (), {})()
            self.manifest.content_sha256 = json.loads((path / "manifest.json").read_text())["content_sha256"]
            self.manifest.sequences = [type("S", (), {"id": f"s{i}"})() for i in range(8)]

    def write_reduced(r, seqs, out):
        return _fake_reference(out, f"{r.manifest.content_sha256}/{len(seqs)}")

    monkeypatch.setattr(store, "Reference", Ref)
    monkeypatch.setattr(store, "write_reduced", write_reduced)


def test_reduced_reference_is_bound_to_its_source_and_size(tmp_path, fake_store):
    """Another reference or another N in the same --work never reuses a reduced reference."""
    work = tmp_path / "work"
    a = probe.reduced_reference(_fake_reference(tmp_path / "A", "sha-A"), 4, work)
    b = probe.reduced_reference(_fake_reference(tmp_path / "B", "sha-B"), 4, work)
    c = probe.reduced_reference(tmp_path / "A", 5, work)
    assert len({a, b, c}) == 3
    assert probe.reduced_reference(tmp_path / "A", 4, work) == a  # the same inputs reuse it
    (work / f"{a.name}.inputs.json").write_text(json.dumps({"source": "sha-B", "n": 4, "seed": probe.SEED}))
    with pytest.raises(ValueError, match="made from other inputs"):
        probe.reduced_reference(tmp_path / "A", 4, work)


def test_a_probe_record_answers_its_reference_anchor_and_build(tmp_path, fake_store, monkeypatch):
    """A cached record is returned only for the same build, anchor and reference content; otherwise it is made."""
    made = []

    def unlocked(anchor, ref, work, keep, repeat, scheme, build, identity, key):
        made.append(identity)
        rec = {"identity": identity, "kl": len(made)}
        (work / f"probe-{key}.json").write_text(json.dumps(rec))
        return rec

    monkeypatch.setattr(probe, "_probe_unlocked", unlocked)
    for name in ("anchor1", "anchor2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps({"name": name}))
    ref_a, ref_b = _fake_reference(tmp_path / "A", "sha-A"), _fake_reference(tmp_path / "B", "sha-B")
    cfg, work = {"model": "m", "format": "w4a16-sym-g128"}, tmp_path / "work"
    first = probe.probe_one(cfg, tmp_path / "anchor1", ref_a, work)
    assert probe.probe_one(cfg, tmp_path / "anchor1", ref_a, work) == first and len(made) == 1
    assert probe.probe_one(cfg, tmp_path / "anchor2", ref_a, work)["kl"] == 2  # another anchor
    assert probe.probe_one(cfg, tmp_path / "anchor1", ref_b, work)["kl"] == 3  # another reference
    assert (
        probe.probe_one({**cfg, "calib": {"x": 1}}, tmp_path / "anchor1", ref_a, work)["kl"] == 4
    )  # another build
    assert probe.probe_one(cfg, tmp_path / "anchor1", ref_a, work, repeat=3)["kl"] == 5  # other scoring


def test_an_own_reference_resolves_its_sampling_as_the_reference_build_does(tmp_path, monkeypatch):
    """The probe asks `qda_kl.reference.reference_sampling`: a model the cards name takes the card's row."""
    data = _data(tmp_path, [{"id": "r0", "text": "t0"}])
    calls = []

    def run(argv, log):
        calls.append(argv)
        out = tmp_path / "work" / argv[argv.index("--out") + 1].rsplit("/", 1)[1]
        out.mkdir(parents=True)
        (out / "manifest.json").write_text("{}")
        return 0.0

    monkeypatch.setattr(probe, "_run", run)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-9B"}))
    (model / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    work = tmp_path / "work"

    ref = probe.own_reference(model, data, 1, work, 128)

    inputs = json.loads((work / f"{ref.name}.inputs.json").read_text())
    assert inputs["effective_sampling"] == {
        "temperature": 1.0,  # the released cards' row, not the file's 0.6
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    assert inputs["sampling"] is None  # no flag: the build resolves the same settings by itself


def test_an_own_reference_is_bound_to_the_generation_config_it_sampled_from(tmp_path, monkeypatch):
    """Without --sampling qda-kl samples from generation_config.json: changing or removing it builds afresh."""
    data = _data(tmp_path, [{"id": f"r{i}", "text": f"t{i}"} for i in range(4)])
    calls = []

    def run(argv, log):
        calls.append(argv)
        out = tmp_path / "work" / argv[argv.index("--out") + 1].rsplit("/", 1)[1]
        out.mkdir(parents=True)
        (out / "manifest.json").write_text("{}")
        return 0.0

    monkeypatch.setattr(probe, "_run", run)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    gen = model / "generation_config.json"
    gen.write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    work = tmp_path / "work"
    a = probe.own_reference(model, data, 2, work, 256)
    assert probe.own_reference(model, data, 2, work, 256) == a and len(calls) == 1
    gen.write_text(json.dumps({"temperature": 1.0, "top_p": 0.95}))
    b = probe.own_reference(model, data, 2, work, 256)
    assert b != a and len(calls) == 2  # another temperature: qda-kl runs again
    gen.write_text(json.dumps({"eos_token_id": 2}))  # no sampling left: refused before any cache hit
    with pytest.raises(ValueError, match="no sampling settings"):
        probe.own_reference(model, data, 2, work, 256)
    assert len(calls) == 2
    c = probe.own_reference(model, data, 2, work, 256, {"temperature": 0.6, "top_p": 0.95})
    assert (
        c not in (a, b) and len(calls) == 3
    )  # the same values given explicitly: another input, built afresh
