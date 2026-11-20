import json
from collections import Counter

import pytest
from qda_quant.calib import meta_path, model_fingerprint, read_rows
from qda_quant.calib.mix import NGramGuard, allocate, mix
from qda_quant.calib.selfgen import resolve_sampling, sampling_layers, sampling_params, selfgen
from qda_quant.step import Refused

from ..goldens import check_golden, refusal


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def model_dir(tmp_path, name="m", temperature=0.6, vocab="v1"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 8}))
    (d / "tokenizer.json").write_text(json.dumps({"vocab": vocab}))
    (d / "generation_config.json").write_text(
        json.dumps(
            {"temperature": temperature, "top_p": 0.95, "top_k": 20, "do_sample": True, "eos_token_id": [1]}
        )
    )
    return d


class FakeGen:
    def __init__(self):
        self.calls = []

    def __call__(self, model, prompts, n, max_tokens, seed, sampling, ctk, gpu_memory_utilization):
        self.calls.append(
            {
                "model": model,
                "prompts": prompts,
                "n": n,
                "seed": seed,
                "sampling": sampling,
                "gpu_memory_utilization": gpu_memory_utilization,
            }
        )
        return [[f"answer {d} to {p}" for d in range(n)] for p in prompts]


# ---------------------------------------------------------------- selfgen


def sg(*a, **k):
    return selfgen(*a, **{"render": "messages", **k})


class FakeTok:
    """Like Qwen3's template: an assistant turn without </think> gets an empty think block prepended."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **k):
        out = ""
        for m in messages:
            c = m["content"]
            if m["role"] == "assistant" and "</think>" not in c:
                c = "<think>\n\n</think>\n\n" + c
            out += f"<|{m['role']}|>{c}<|end|>"
        return out + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split()]}


def test_selfgen_writes_messages_and_sidecar(tmp_path, cache):
    m, prompts = (
        model_dir(tmp_path),
        jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}, {"id": "b", "prompt": "yo"}]),
    )
    gen = FakeGen()
    out = sg(m, prompts, n=2, max_tokens=64, seed=17, generate=gen)
    assert out.parent == cache / "calib" / model_fingerprint(m)
    rows = read_rows(out)
    assert len(rows) == 4
    assert rows[0] == {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "answer 0 to hi"}]
    }
    meta = json.loads(meta_path(out).read_text())
    assert meta["model"]["fingerprint"] == model_fingerprint(m)
    s = meta["settings"]
    assert s["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20} == gen.calls[0]["sampling"]
    assert s["stopping"] == {"eos_token_id": [1]}
    assert s["n"] == 2 and s["seed"] == 17 and s["max_tokens"] == 64 and meta["tokenizer"]
    assert out.with_name(out.name + ".tokenizer").read_text().strip() == meta["tokenizer"]


def test_hub_sampling_reads_the_raw_fields_and_refuses_an_eos_only_config(tmp_path, monkeypatch):
    """A hub generation_config.json is read raw: no None fields; an EOS-only file is refused (golden row)."""
    import huggingface_hub

    cfg = tmp_path / "generation_config.json"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(cfg))
    cfg.write_text(json.dumps({"temperature": 0.6, "top_p": 0.95, "top_k": 20}))
    assert resolve_sampling("org/model", None, "sampling") == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def _hub_cache(root, files: dict, no_exist: tuple = ()) -> None:
    """A Hugging Face cache entry for org/model at one commit: ``refs/main``, a snapshot and ``.no_exist``."""
    repo, commit = root / "models--org--model", "0" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "snapshots" / commit).mkdir(parents=True)
    for name, content in files.items():
        (repo / "snapshots" / commit / name).write_text(json.dumps(content))
    for name in no_exist:
        (repo / ".no_exist" / commit).mkdir(parents=True, exist_ok=True)
        (repo / ".no_exist" / commit / name).touch()


@pytest.mark.parametrize(
    ("files", "no_exist", "expected"),
    [
        (
            {"generation_config.json": {"temperature": 0.6, "top_p": 0.95}},
            (),
            {"temperature": 0.6, "top_p": 0.95},
        ),
        ({"config.json": {}}, ("generation_config.json",), {}),  # the hub said the repo ships none
    ],
)
def test_offline_hub_sampling_reads_the_cache(tmp_path, monkeypatch, files, no_exist, expected):
    """HF_HUB_OFFLINE=1: a hub id is answered from the local cache, the file or the cache's record of its absence."""
    from huggingface_hub import constants
    from qda_quant.calib.selfgen import sampling_of

    _hub_cache(tmp_path, files, no_exist)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    assert sampling_of("org/model") == expected


def test_offline_hub_sampling_not_in_the_cache_is_refused_not_read_as_none(tmp_path, monkeypatch):
    """Offline, a generation_config.json the cache does not hold is unknown, not "names no sampling".

    Before, the offline cache miss was taken for a repo with no file, and a reference build was refused with
    ``no sampling settings``, which names the wrong cause and the wrong fix.
    """
    from huggingface_hub import constants
    from qda_quant.calib.selfgen import sampling_of
    from qda_quant.exit import ENVIRONMENT

    _hub_cache(
        tmp_path, {"config.json": {}}
    )  # weights and config cached, generation_config.json never fetched
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    with pytest.raises(Refused) as caught:
        sampling_of("org/model")
    assert "org/model: generation_config.json is not in the local Hugging Face cache" in str(caught.value)
    assert "\n" not in str(caught.value)
    assert caught.value.fix.startswith("download it once with network access (hf download org/model ")
    assert caught.value.code == ENVIRONMENT
    with pytest.raises(Refused, match="not in the local Hugging Face cache"):
        resolve_sampling("org/model", None, "--sampling")


def test_a_fallback_answers_only_for_a_file_that_names_no_sampling(tmp_path):
    """The file is read whole: any sampling in it keeps the caller's fallback out; naming none lets it in.

    The fallback is what a reference build passes (the settings the model's released cards record); the layer
    each field came from is the merged mapping's own record, not a second merge.
    """
    model = tmp_path / "m"
    model.mkdir()
    (model / "generation_config.json").write_text(json.dumps({"temperature": 0.6}))
    fallback = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

    got = resolve_sampling(model, {"top_p": 0.8}, "sampling", fallback=fallback)

    assert got == {"temperature": 0.6, "top_p": 0.8}  # the file's own settings; the fallback stayed out
    assert sampling_layers(model, {"top_p": 0.8}, fallback) == {
        "temperature": "generation_config",
        "top_p": "config",
    }
    (model / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))

    got = resolve_sampling(model, {"temperature": 0.7}, "sampling", fallback=fallback)

    assert got == {"temperature": 0.7, "top_p": 0.95, "top_k": 20}  # the flag over the cards, field by field
    layers = sampling_layers(model, {"temperature": 0.7}, fallback)
    assert layers["temperature"] == "config" and layers["top_p"] == "fallback"


def test_selfgen_sampling_setting_wins_field_by_field(tmp_path, cache):
    """A setting that names one field keeps the model file's other fields, as the reference build does."""
    m, p = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}])
    gen = FakeGen()
    selfgen(m, p, generate=gen, tokenizer=FakeTok(), sampling={"temperature": 0.3})
    assert gen.calls[0]["sampling"] == {"temperature": 0.3, "top_p": 0.95, "top_k": 20}


def test_sampling_params_keep_vllms_default_except_where_asked(monkeypatch):
    """LiveCodeBench asks for skip_special_tokens=False; every other caller keeps vLLM's default."""
    import sys
    import types

    calls: list[dict] = []

    class Fake:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake = types.ModuleType("vllm")
    fake.SamplingParams = Fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)

    sampling_params({"temperature": 0.6}, 64, 7)
    sampling_params({"temperature": 0.6}, 64, 8, skip_special_tokens=False)

    assert calls[0] == {"temperature": 0.6, "max_tokens": 64, "seed": 7, "skip_special_tokens": True}
    assert calls[1] == {"temperature": 0.6, "max_tokens": 64, "seed": 8, "skip_special_tokens": False}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """The real `engine_start` with its environment only (no guards, compiler probes or cache directories), and
    a stand-in for vLLM whose engine records `os.environ` and its arguments as it starts."""
    import os
    import sys
    import types

    from qda_quant.output import vllm_env

    for name in vllm_env.CACHE_DIRS:
        monkeypatch.setenv(name, str(tmp_path / name))
    for name in vllm_env.NO_JIT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vllm_env, "vllm_environ", vllm_env.environ)
    started: dict = {}

    class LLM:
        def __init__(self, **kwargs):
            started.update(environ=dict(os.environ), kwargs=kwargs)

        def chat(self, *args, **kwargs):
            return []

    fake = types.ModuleType("vllm")
    fake.LLM = LLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return started


def test_where_flashinfer_cannot_build_selfgen_starts_vllm_with_qda_kls_no_jit_settings(engine, monkeypatch):
    """A selfgen build on an RTX PRO 6000 without nvcc or FlashInfer's prebuilt kernels died in FlashInfer's JIT
    (`Could not find nvcc`) where `qda-kl kl` started: its engine started without qda-kl's no-JIT settings."""
    from qda_kl.serve import NO_AUTOTUNE_ARGS, ServeSettings, no_jit
    from qda_quant.calib import selfgen as S

    monkeypatch.setattr(S, "flashinfer_cannot_build", lambda: True)
    S.vllm_generate("m", [], 1, 64, 17, {"temperature": 0.6}, {}, 0.5)
    kl = no_jit(ServeSettings(args={"seed": 17, "gpu_memory_utilization": 0.5}))
    assert {name: engine["environ"].get(name) for name in kl.env} == kl.env
    # FlashInfer autotune stays vLLM's default here: only reference, scoring and gencheck engines turn it off
    args = {k: v for k, v in kl.args.items() if k not in NO_AUTOTUNE_ARGS}
    assert engine["kwargs"] == {"model": "m", **args}


def test_where_flashinfer_builds_selfgen_keeps_its_engine_and_cache_key(engine, tmp_path, cache, monkeypatch):
    """Where FlashInfer builds or ships its kernels, the engine starts as before the no-JIT settings and the cache
    key is the same, so a published recipe's calibration is generated and served as it was."""
    from qda_quant.calib import selfgen as S
    from qda_quant.output.vllm_env import NO_JIT_ENV

    monkeypatch.setattr(S, "flashinfer_cannot_build", lambda: False)
    S.vllm_generate("m", [], 1, 64, 17, {"temperature": 0.6}, {}, 0.5)
    assert not set(NO_JIT_ENV) & set(engine["environ"])
    assert engine["kwargs"] == {"model": "m", "seed": 17, "gpu_memory_utilization": 0.5}
    m, prompts = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}])
    assert S.cache_file(m, prompts)[0].name == "1c258c961d27a27570880cfd.jsonl"  # the key before this change


def test_where_flashinfer_cannot_build_the_cache_key_has_the_no_jit_settings(tmp_path, cache, monkeypatch):
    """The no-JIT settings choose other kernels (the Triton GDN prefill on H100), so neither host is served the
    other's file; a host where FlashInfer builds keeps the key it had."""
    from qda_quant.calib import selfgen as S

    gen = FakeGen()
    m, prompts = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}])
    monkeypatch.setattr(S, "flashinfer_cannot_build", lambda: False)
    builds = sg(m, prompts, generate=gen)
    monkeypatch.setattr(S, "flashinfer_cannot_build", lambda: True)
    cannot = sg(m, prompts, generate=gen)
    assert cannot != builds and len(gen.calls) == 2
    assert "engine" not in json.loads(meta_path(builds).read_text())["settings"]
    assert json.loads(meta_path(cannot).read_text())["settings"]["engine"] == S.NO_JIT_ENGINE


def test_cache_hit_does_not_call_the_generator(tmp_path, cache):
    m, prompts = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}])
    gen = FakeGen()
    first = sg(m, prompts, generate=gen)
    assert len(gen.calls) == 1

    def boom(*a):
        raise AssertionError("generator called on a cache hit")

    assert sg(m, prompts, generate=boom) == first


def test_cache_key_changes_with_model_prompts_and_sampling(tmp_path, cache):
    gen = FakeGen()
    p1 = jsonl(tmp_path / "p1.jsonl", [{"id": "a", "prompt": "hi"}])
    p2 = jsonl(tmp_path / "p2.jsonl", [{"id": "a", "prompt": "hello"}])
    base = sg(model_dir(tmp_path, "m"), p1, generate=gen)
    outs = {
        "model (tokenizer)": sg(model_dir(tmp_path, "m2", vocab="v2"), p1, generate=gen),
        "sampling": sg(model_dir(tmp_path, "m3", temperature=0.7), p1, generate=gen),
        "prompts": sg(tmp_path / "m", p2, generate=gen),
        "n": sg(tmp_path / "m", p1, n=2, generate=gen),
        "seed": sg(tmp_path / "m", p1, seed=18, generate=gen),
        "max_tokens": sg(tmp_path / "m", p1, max_tokens=99, generate=gen),
        "gpu_memory_utilization": sg(tmp_path / "m", p1, gpu_memory_utilization=0.2, generate=gen),
    }
    assert len({base, *outs.values()}) == 1 + len(outs)
    assert len(gen.calls) == 1 + len(outs)
    # the model's fingerprint is the directory; same model, same directory
    assert outs["prompts"].parent == base.parent != outs["model (tokenizer)"].parent
    meta = json.loads(meta_path(outs["gpu_memory_utilization"]).read_text())
    assert meta["settings"]["gpu_memory_utilization"] == gen.calls[-1]["gpu_memory_utilization"] == 0.2


def test_selfgen_refuses_short_generator_output(tmp_path, cache):
    m, prompts = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "a", "prompt": "hi"}])
    with pytest.raises(ValueError, match="answers"):
        sg(m, prompts, n=2, generate=lambda *a: [["only one"]])
    assert not list((cache / "calib").rglob("*.jsonl"))


# ---------------------------------------------------------------- mix


def src(tmp_path, name, n):
    return jsonl(tmp_path / f"{name}.jsonl", [{"text": f"{name} row {i} with some words"} for i in range(n)])


def test_allocate():
    assert allocate([3, 1], 128) == [96, 32]
    assert allocate([1, 1, 1], 10) == [4, 3, 3]  # remainder ties go to the earlier source
    assert sum(allocate([0.2, 0.5, 0.3], 7)) == 7


def test_mix_proportions_and_determinism(tmp_path, cache):
    a, b = src(tmp_path, "a", 200), src(tmp_path, "b", 200)
    out1 = mix([(a, 3), (b, 1)], 128, seed=5, out=tmp_path / "m1.jsonl")
    out2 = mix([(a, 3), (b, 1)], 128, seed=5, out=tmp_path / "m2.jsonl")
    assert out1.read_text() == out2.read_text()
    rows = read_rows(out1)
    assert Counter(r["text"].split()[0] for r in rows) == {"a": 96, "b": 32}
    assert len({r["text"] for r in rows}) == 128
    assert mix([(a, 3), (b, 1)], 128, seed=6, out=tmp_path / "m3.jsonl").read_text() != out1.read_text()
    meta = json.loads(meta_path(out1).read_text())
    assert [s["rows"] for s in meta["sources"]] == [96, 32] and meta["guard"] is None
    # default output under the cache, same inputs -> same path
    assert mix([(a, 3), (b, 1)], 128, seed=5) == mix([(a, 3), (b, 1)], 128, seed=5)


def test_ngram_guard_drops_heldout_overlap(tmp_path, cache):
    held = jsonl(
        tmp_path / "held.jsonl",
        [
            {
                "id": "q1",
                "prompt": "Find all real x such that x to the fourth minus five x squared plus four is zero",
            },
            {"id": "q2", "prompt": "short eval prompt"},
        ],
    )
    rows = [
        {"text": f"clean calibration row number {i} about cooking pasta and the weather today"}
        for i in range(20)
    ]
    rows += [
        {"text": "Please: find all real x such that x to the fourth minus five x squared plus four is zero!"},
        {
            "messages": [
                {"role": "user", "content": "here is a SHORT eval prompt indeed"},
                {"role": "assistant", "content": "ok"},
            ]
        },
    ]
    s = jsonl(tmp_path / "s.jsonl", rows)
    g = NGramGuard(held, n=13)
    out = mix([(s, 1)], 20, seed=0, guard=g, out=tmp_path / "m.jsonl")
    texts = [r.get("text", "") for r in read_rows(out)]
    assert len(texts) == 20 and not any("fourth" in t for t in texts)
    meta = json.loads(meta_path(out).read_text())
    assert meta["guard"]["kind"] == "ngram" and meta["guard"]["dropped"] == 2
    assert meta["sources"][0]["guard_rejected"] == 2
    # a guard that leaves too few rows is an error, never a smaller file
    with pytest.raises(Refused, match="rejected by the guard"):
        mix([(s, 1)], 21, guard=NGramGuard(held), out=tmp_path / "m2.jsonl")


def test_custom_guard_is_any_callable(tmp_path, cache):
    s = src(tmp_path, "a", 50)
    out = mix([(s, 1)], 10, guard=lambda r: r["text"].split()[2] < "3", out=tmp_path / "m.jsonl")
    assert all(r["text"].split()[2] < "3" for r in read_rows(out))


def test_token_id_sources_need_one_tokenizer(tmp_path, cache):
    a = jsonl(tmp_path / "a.jsonl", [{"input_ids": [1, 2, i]} for i in range(10)])
    b = jsonl(tmp_path / "b.jsonl", [{"input_ids": [3, 4, i]} for i in range(10)])
    meta_path(a).write_text(json.dumps({"tokenizer": "t1"}))
    meta_path(b).write_text(json.dumps({"tokenizer": "t1"}))
    out = mix([(a, 1), (b, 1)], 10, out=tmp_path / "m.jsonl")
    assert json.loads(meta_path(out).read_text())["tokenizer"] == "t1"


def _weights(d, payload: bytes):
    (d / "model.safetensors").write_bytes(payload)
    return d


def test_fingerprint_sees_weight_values(tmp_path, cache):
    a = _weights(model_dir(tmp_path, "a"), b"weights-A" * 100)
    b = _weights(model_dir(tmp_path, "b"), b"weights-B" * 100)  # same config, tokenizer, shard layout
    assert model_fingerprint(a) != model_fingerprint(b)
    c = _weights(model_dir(tmp_path, "c"), b"weights-A" * 100)
    assert model_fingerprint(a) == model_fingerprint(c)  # content, not path


def test_selfgen_refuses_another_models_cache(tmp_path, cache):
    a = _weights(model_dir(tmp_path, "a"), b"A" * 64)
    b = _weights(model_dir(tmp_path, "b"), b"B" * 64)
    p = jsonl(tmp_path / "p.jsonl", [{"id": "x", "prompt": "hi"}])
    gen = FakeGen()
    out_a = sg(a, p, generate=gen)
    out_b = sg(b, p, generate=gen)
    assert len(gen.calls) == 2 and gen.calls[1]["model"] == str(b) and out_a != out_b


def test_stale_or_foreign_sidecar_regenerates(tmp_path, cache):
    m, p = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "x", "prompt": "hi"}])
    gen = FakeGen()
    out = sg(m, p, generate=gen)
    meta = json.loads(meta_path(out).read_text())
    meta["model"]["fingerprint"] = "0" * 64  # e.g. a file copied in from another model's directory
    meta_path(out).write_text(json.dumps(meta))
    assert sg(m, p, generate=gen) == out and len(gen.calls) == 2
    assert json.loads(meta_path(out).read_text())["model"]["fingerprint"] == model_fingerprint(m)


def test_stopping_settings_change_the_key(tmp_path, cache):
    p = jsonl(tmp_path / "p.jsonl", [{"id": "x", "prompt": "hi"}])
    m1, m2 = model_dir(tmp_path, "m1"), model_dir(tmp_path, "m2")
    g = json.loads((m2 / "generation_config.json").read_text())
    (m2 / "generation_config.json").write_text(json.dumps({**g, "eos_token_id": [1, 2]}))
    assert sg(m1, p, generate=FakeGen()).name != sg(m2, p, generate=FakeGen()).name


def test_weight_digest_memo_hit_and_invalidation(tmp_path, cache, monkeypatch):
    import os

    from qda_quant.intake import fingerprint as F

    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x" * 1000)
    calls = []
    real = F._sha256
    monkeypatch.setattr(F, "_sha256", lambda p: calls.append(p) or real(p))
    d1 = F.file_digest(f)
    assert F.file_digest(f) == d1 and len(calls) == 1  # memo hit: no re-read
    f.write_bytes(b"y" * 1000)
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    d2 = F.file_digest(f)
    assert len(calls) == 2 and d2 != d1


def test_mix_header_row_is_read_stripped_and_kept(tmp_path, cache):
    tok = "a" * 64
    a = jsonl(tmp_path / "a.jsonl", [{"tokenizer": tok}] + [{"input_ids": [1, i]} for i in range(10)])
    out = mix([(a, 1)], 10, out=tmp_path / "m.jsonl")
    rows = read_rows(out)
    assert len(rows) == 10 and all("input_ids" in r for r in rows)
    assert out.with_name("m.jsonl.tokenizer").read_text().strip() == tok
    assert json.loads(meta_path(out).read_text())["tokenizer"] == tok


def test_allocate_exact_ties():
    # 0.1 + 0.2 style float noise must not break the tie rule: equal remainders go to the earlier source
    assert allocate([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], 5) == allocate([1, 2, 3, 4, 1, 2, 3, 4], 5)
    assert allocate([1, 1], 3) == [2, 1]


def test_selfgen_text_output_loads_in_the_engine(tmp_path, cache):
    """selfgen (fake generator, render text) feeds the engine's loader, old or new signature."""
    import inspect

    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.calib import load_ids

    m = model_dir(tmp_path)
    p = jsonl(tmp_path / "p.jsonl", [{"id": str(i), "prompt": f"question {i} " * 3} for i in range(8)])
    out = selfgen(m, p, n=1, generate=FakeGen(), tokenizer=FakeTok())
    rows = read_rows(out)
    assert all(set(r) == {"text"} and r["text"].startswith("<|user|>") for r in rows)
    # the prompt as rendered for generation, then the answer exactly as generated (no template rewrite)
    assert (
        rows[0]["text"]
        == "<|user|>" + "question 0 " * 3 + "<|end|><|assistant|>answer 0 to " + "question 0 " * 3
    )
    assert "<think>" not in rows[0]["text"]
    calib = FileSource(path=str(out), samples=4, seqlen=4)
    if "model_dir" in inspect.signature(load_ids).parameters:  # the per-model loader
        ids = load_ids(calib, m, FakeTok())
    else:
        ids = load_ids(calib, FakeTok())
    assert len(ids) == 4 and all(len(x) == 4 for x in ids)


def test_render_version_invalidates_old_rows(tmp_path, cache, monkeypatch):
    from qda_quant.calib import selfgen as S

    m, p = model_dir(tmp_path), jsonl(tmp_path / "p.jsonl", [{"id": "x", "prompt": "hi"}])
    gen = FakeGen()
    first = selfgen(m, p, generate=gen, tokenizer=FakeTok())
    monkeypatch.setattr(S, "RENDER_VERSION", S.RENDER_VERSION + 1)
    assert selfgen(m, p, generate=gen, tokenizer=FakeTok()) != first and len(gen.calls) == 2


def _tok_model(tmp_path, name, vocab, merges="a b\n"):
    """A model with no tokenizer.json: its vocabulary lives in vocab.json / merges.txt (GPT-2 style)."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer"}))
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text(merges)
    (d / "model.safetensors").write_bytes(b"same weights")
    return d


def test_fingerprint_covers_every_file(tmp_path, cache):
    # equal config and tokenizer_config, no tokenizer.json, different vocab.json
    a = _tok_model(tmp_path, "a", {"x": 0, "y": 1})
    b = _tok_model(tmp_path, "b", {"x": 0, "y": 2})
    assert model_fingerprint(a) != model_fingerprint(b)
    c = _tok_model(tmp_path, "c", {"x": 0, "y": 1}, merges="a c\n")  # merges.txt only
    assert model_fingerprint(a) != model_fingerprint(c)
    d = _tok_model(tmp_path, "d", {"x": 0, "y": 1})
    assert model_fingerprint(a) == model_fingerprint(d)
    (d / "added_tokens.json").write_text("{}")  # any new file counts
    assert model_fingerprint(a) != model_fingerprint(d)


def test_fingerprint_ignores_dotfiles_and_readmes(tmp_path, cache):
    a = _tok_model(tmp_path, "a", {"x": 0})
    before = model_fingerprint(a)
    (a / ".git").mkdir()
    (a / ".git" / "HEAD").write_text("ref")
    (a / ".cache").mkdir()
    (a / ".cache" / "x.lock").write_text("")
    (a / "README.md").write_text("# model card")
    (a / "README_zh.md").write_text("# card")
    assert model_fingerprint(a) == before


def test_fingerprint_symlinks(tmp_path, cache):
    a = _tok_model(tmp_path, "a", {"x": 0})
    b = _tok_model(tmp_path, "b", {"x": 0})
    (b / "merges.txt").unlink()
    (b / "merges.txt").symlink_to(b / "vocab.json")  # inside: hashed as the target's content
    (a / "merges.txt").write_text((a / "vocab.json").read_text())
    assert model_fingerprint(a) == model_fingerprint(b)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("secret")
    (b / "extra.txt").symlink_to(outside)
    with pytest.raises(Refused, match="outside"):
        model_fingerprint(b)


def _hub_snapshot(tmp_path, src):
    """``src``'s files laid out like the hub cache: snapshots/<rev>/<file> -> ../../blobs/<n> (relative links)."""
    import os
    import shutil

    repo = tmp_path / "hub" / "models--org--tiny"
    snap, blobs = repo / "snapshots" / ("a" * 40), repo / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for i, f in enumerate(sorted(p for p in src.rglob("*") if p.is_file())):
        shutil.copy(f, blobs / f"b{i}")
        link = snap / f.relative_to(src)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blobs / f"b{i}", link.parent))
    return repo, snap


def test_fingerprint_of_a_hub_snapshot_is_its_files(tmp_path, cache):
    a = _tok_model(tmp_path, "a", {"x": 0})
    _, snap = _hub_snapshot(tmp_path, a)
    assert model_fingerprint(snap) == model_fingerprint(a)


def test_a_hub_snapshot_on_the_shared_blob_store_is_its_files(tmp_path, cache):
    """huggingface_hub's shared store: <repo>/blobs/<sha> -> <hub>/blobs/<xx>/<hash>, one more link."""
    import os

    a = _tok_model(tmp_path, "a", {"x": 0})
    repo, snap = _hub_snapshot(tmp_path, a)
    store = repo.parent / "blobs" / "d5"
    store.mkdir(parents=True)
    for blob in (repo / "blobs").iterdir():
        target = store / f"d5{blob.name}"
        blob.rename(target)
        blob.symlink_to(os.path.relpath(target, blob.parent))
    assert model_fingerprint(snap) == model_fingerprint(a)
    # the shared store holds every repo's blobs: a direct link past this repo's blobs reads another repo's
    (snap / "extra.txt").symlink_to(os.path.relpath(next(store.iterdir()), snap))
    with pytest.raises(ValueError, match="outside"):
        model_fingerprint(snap)


def test_a_hub_snapshot_link_outside_its_blobs_is_still_refused(tmp_path, cache):
    a = _tok_model(tmp_path, "a", {"x": 0})
    repo, snap = _hub_snapshot(tmp_path, a)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    (snap / "extra.txt").symlink_to(secret)
    with pytest.raises(Refused, match="outside"):
        model_fingerprint(snap)
    (snap / "extra.txt").unlink()
    other = repo / "refs"
    other.mkdir()
    (other / "main").write_text("x")
    (snap / "extra.txt").symlink_to(other / "main")  # inside the repo, but not a blob
    with pytest.raises(Refused, match="outside"):
        model_fingerprint(snap)


# ---------------------------------------------------------------- refusal wording: golden/errors_calib.txt


def _ids(t, name, tokenizer=None):
    """A token-id file with `tokenizer` in its sidecar (none: no fingerprint at all)."""
    f = jsonl(t / f"{name}.jsonl", [{"input_ids": [1, 2, i]} for i in range(10)])
    if tokenizer:
        meta_path(f).write_text(json.dumps({"tokenizer": tokenizer}))
    return f


def _no_sampling(t):
    m = t / "none"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (m / "tokenizer.json").write_text(json.dumps({"vocab": "v1"}))
    selfgen(m, jsonl(t / "p.jsonl", [{"id": "a", "prompt": "hi"}]), generate=FakeGen(), tokenizer=FakeTok())


def _eos_only_hub_config(t, mp):
    import huggingface_hub

    cfg = jsonl(t / "generation_config.json", [])
    cfg.write_text(json.dumps({"eos_token_id": [1]}))
    mp.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(cfg))
    resolve_sampling("org/model", None, "sampling")


def _header(t, sidecar=None, header="a" * 64):
    a = jsonl(t / "a.jsonl", [{"tokenizer": header}] + [{"input_ids": [1, i]} for i in range(10)])
    if sidecar:
        (t / "a.jsonl.tokenizer").write_text(sidecar)
    mix([(a, 1)], 1, out=t / "m.jsonl")


# name -> render(tmp_path, monkeypatch); each one must refuse
REFUSALS = {
    # no silent vLLM temperature 1.0: the refusal names the config key to set
    "selfgen without sampling": lambda t, mp: _no_sampling(t),
    "hub generation config with only eos": lambda t, mp: _eos_only_hub_config(t, mp),
    "fingerprint of a non-model dir": lambda t, mp: model_fingerprint(t),
    "fingerprint of a hub id": lambda t, mp: model_fingerprint("Qwen/Qwen3-8B"),
    "allocate all-zero weights": lambda t, mp: allocate([0, 0], 4),
    "mix source shorter than its share": lambda t, mp: mix(
        [(src(t, "a", 10), 1), (src(t, "b", 10), 1)], 40, out=t / "m.jsonl"
    ),
    "mix token ids without fingerprint": lambda t, mp: mix([(_ids(t, "a"), 1)], 5, out=t / "m.jsonl"),
    "mix token ids of two tokenizers": lambda t, mp: mix(
        [(_ids(t, "a", "t1"), 1), (_ids(t, "b", "t2"), 1)], 10, out=t / "m.jsonl"
    ),
    "mix header and sidecar disagree": lambda t, mp: _header(t, sidecar="b" * 64),
    "mix header not a sha256": lambda t, mp: _header(t, header="not-a-digest"),
    "ngram guard on token rows": lambda t, mp: mix(
        [(_ids(t, "a", "a" * 64), 1)],
        2,
        guard=NGramGuard(jsonl(t / "held.jsonl", [{"prompt": "an evaluation prompt"}])),
        out=t / "m.jsonl",
    ),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusals_keep_their_wording(tmp_path, cache, monkeypatch, name):
    """Each calibration-data refusal fires, with the exact text in ``golden/errors_calib.txt``."""
    text = refusal(lambda: REFUSALS[name](tmp_path, monkeypatch), ValueError)
    check_golden("calib", name, text.replace(str(tmp_path), "<tmp>"), list(REFUSALS))


def test_offline_hub_sampling_reads_the_revision_after_the_at(tmp_path, monkeypatch):
    """``org/model@SHA`` reads the pinned snapshot; the bare id, with no ``refs/main``, names that revision."""
    from huggingface_hub import constants
    from qda_quant.calib.selfgen import sampling_of
    from qda_quant.exit import ENVIRONMENT

    _hub_cache(tmp_path, {"generation_config.json": {"temperature": 0.6, "top_p": 0.95}})
    (tmp_path / "models--org--model" / "refs" / "main").unlink()  # a download pinned to the commit
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    assert sampling_of("org/model@" + "0" * 40) == {"temperature": 0.6, "top_p": 0.95}
    with pytest.raises(Refused, match="no `main` ref") as got:
        sampling_of("org/model")
    assert got.value.code == ENVIRONMENT and f"org/model@{'0' * 40}" in got.value.fix
