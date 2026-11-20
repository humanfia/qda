"""selfgen as a build stage: plan time never generates; a cache miss is the build's first stage, run in a child.

vLLM never runs in a build's process: it changes process-wide torch state, and a build after it wrote other
weights. These tests compare the bytes of a build after a generate stage with a build from the same cache.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from qda_quant.calib import selfgen as S
from qda_quant.calib import sha256_file
from qda_quant.calib.sources import SelfgenSource
from qda_quant.engine.build import build
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ
from qda_quant.steps.token_weights import Guided
from qda_quant.target import Target

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, writes
from .test_build import FakeWriter, NeverHere, TinyRTN

CPU = [torch.device("cpu")]


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    tiny_qwen3(layers=2, hidden=128, inter=256).save_pretrained(tmp_path / "m")  # g128 groups for GPTQ
    save_tiny_tokenizer(tmp_path / "m")
    # a sampling setting, so selfgen needs no explicit one (a model without it is refused)
    (tmp_path / "m" / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    (tmp_path / "p.jsonl").write_text(
        "".join(json.dumps({"id": str(i), "prompt": f"w{i + 3}"}) + "\n" for i in range(4))
    )
    return tmp_path


@pytest.fixture
def recorder(monkeypatch):
    """The default generator, replaced: records each call and answers with `words` tiny-vocab words."""
    calls = []

    def generate(model, prompts, n, max_tokens, seed, sampling, ctk, gpu_memory_utilization):
        calls.append(prompts)
        return [
            [" ".join(f"w{(i + d + k) % 100 + 3}" for k in range(recorder.words)) for d in range(n)]
            for i, _ in enumerate(prompts)
        ]

    recorder = type("R", (), {"calls": calls, "words": 20})
    monkeypatch.setattr(S, "vllm_generate", generate)
    # a build generates in a child process; here the "child" is this process, so the recorder sees the call
    monkeypatch.setattr(
        S, "generate_isolated", lambda *a, **k: S.selfgen(*a, **k, generate=generate, tokenizer=None)
    )
    return recorder


def recipe(tmp, *steps, samples=4):
    src = SelfgenSource(prompts=str(tmp / "p.jsonl"), samples=samples, seqlen=8, max_tokens=32)
    return Recipe(steps=list(steps), calib=src)


def plan(tmp, r):
    return compile_plan(str(tmp / "m"), r, Format.parse("w4a16"), Target(), CPU)


def run(tmp, r, out="out", **engine):
    engine = {"exporter": writes(FakeWriter), **engine}
    return build(
        str(tmp / "m"), r, format="w4a16-sym-g128", out_dir=tmp / out, devices=["cpu"],
        dtype=torch.float32, log=lambda s: None, **engine,
    )  # fmt: skip


def weights(ckpt: Path) -> dict[str, str]:
    """sha256 of each safetensors file of a checkpoint."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ckpt.glob("*.safetensors"))}


def stage_kinds(ckpt: Path) -> list[str]:
    return [st["kind"] for st in json.loads((ckpt / "build.json").read_text())["stages"]]


def test_plan_never_generates_and_marks_the_stage(tiny, recorder):
    p = plan(tiny, recipe(tiny, TinyRTN()))
    assert recorder.calls == []
    assert p.generate.startswith("selfgen: 1 answer(s) per prompt") and p.calib_ids is None


def test_reading_a_missing_cache_is_refused_not_generated(tiny, recorder):
    with pytest.raises(Refused, match="no cached generation"):
        recipe(tiny).calib.read(tiny / "m")
    assert recorder.calls == []


def test_a_missing_prompt_file_is_refused_at_plan(tiny, recorder):
    """The cache key hashes the prompt file; a missing one is a refusal, not an OSError traceback."""
    r = recipe(tiny, TinyRTN())
    (tiny / "p.jsonl").unlink()
    with pytest.raises(Refused, match="cannot read the prompt file .*p.jsonl"):
        plan(tiny, r)
    assert recorder.calls == []


def test_a_refused_plan_never_generates(tiny, recorder):
    with pytest.raises(Refused, match="not on this target"):
        run(tiny, recipe(tiny, NeverHere()))
    assert recorder.calls == []


def test_a_miss_is_the_first_stage_and_records_the_cache(tiny, recorder):
    r = recipe(tiny, TinyRTN())
    info = json.loads((run(tiny, r) / "build.json").read_text())
    assert len(recorder.calls) == 1
    stage = info["stages"][0]
    assert [st["kind"] for st in info["stages"]] == ["generate", "layers"]
    assert stage["steps"] == ["selfgen"] and stage["seconds"] >= 0
    assert stage["sha256"] == sha256_file(stage["path"])
    path, hit = r.calib._cache(tiny / "m")
    assert hit and stage["path"] == str(path)
    # a hit: no stage, no generation
    info = json.loads((run(tiny, r, "out2") / "build.json").read_text())
    assert len(recorder.calls) == 1 and [st["kind"] for st in info["stages"]] == ["layers"]
    assert plan(tiny, r).generate is None


def test_row_checks_run_after_generation_before_layer_work(tiny, recorder):
    recorder.words = 1  # too few tokens for 4 samples of 8
    FakeWriter.last = None
    with pytest.raises(Refused, match="samples of 8, recipe asks for 4"):
        run(tiny, recipe(tiny, TinyRTN()))
    assert len(recorder.calls) == 1
    assert FakeWriter.last is None  # the run never started: no model load, no layer


def test_a_build_after_generation_equals_a_build_from_the_cache(tiny, recorder):
    r = recipe(tiny, GPTQ())  # a real step and the real writer: the bytes a user gets
    miss = run(tiny, r, "miss", exporter=None)
    hit = run(tiny, r, "hit", exporter=None)
    assert len(recorder.calls) == 1
    assert stage_kinds(miss) == ["generate", "layers"] and stage_kinds(hit) == ["layers"]
    assert weights(miss) and weights(miss) == weights(hit)


def test_the_plan_after_generation_uses_the_build_batch_size(tiny, recorder):
    """The planner sizes the calibration calls in its peak by the batch count, so the plan on the generated rows
    must be made with the build's batch size, like the first plan."""
    out = run(tiny, recipe(tiny, TinyRTN()), batch_size=2)
    after = json.loads((out / "build.json").read_text())["residency"]

    def plan_at(batch_size):
        return compile_plan(
            str(tiny / "m"),
            recipe(tiny, TinyRTN()),
            Format.parse("w4a16-sym-g128"),
            Target(),
            CPU,
            batch_size=batch_size,
        ).residency.as_dict()

    assert plan_at(1) != plan_at(2) and after == plan_at(2)


#: A stand-in for vLLM in the child process: answers each prompt with tiny-vocabulary words.
FAKE_VLLM = """
import json
from pathlib import Path

class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw

class _Out:
    def __init__(self, text):
        self.outputs = [type("C", (), {"text": text})()]

class LLM:
    def __init__(self, model, seed=0, gpu_memory_utilization=None, **kwargs):
        (Path(model).parent / "llm-budget.json").write_text(json.dumps(gpu_memory_utilization))

    def chat(self, convs, params, use_tqdm=False, chat_template_kwargs=None):
        return [_Out(" ".join(f"w{(i + k) % 100 + 3}" for k in range(20))) for i, _ in enumerate(convs)]
"""


@pytest.mark.parametrize("budget", [None, 0.2])
def test_generation_runs_in_a_child_never_in_the_build_process(tiny, monkeypatch, budget):
    """Fails if generation comes back into the build's process: there vLLM raises, and must not be imported."""
    (tiny / "fake" / "vllm").mkdir(parents=True)
    (tiny / "fake" / "vllm" / "__init__.py").write_text(FAKE_VLLM)
    monkeypatch.setenv("PYTHONPATH", str(tiny / "fake"))  # read by the child only

    def in_process(*a, **k):
        raise AssertionError("vLLM generation ran in the build's process")

    monkeypatch.setattr(S, "vllm_generate", in_process)
    had_vllm = "vllm" in sys.modules
    r = recipe(tiny, TinyRTN())
    if budget is not None:
        r.calib.gpu_memory_utilization = budget
    out = run(tiny, r)
    expected_budget = DEFAULT_GPU_MEMORY_UTILIZATION if budget is None else budget
    assert json.loads((tiny / "llm-budget.json").read_text()) == expected_budget
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["calib"]["selfgen"]["gpu_memory_utilization"] == expected_budget
    assert stage_kinds(out) == ["generate", "layers"]
    assert r.calib._cache(tiny / "m")[1]  # the child wrote the cache
    path, _ = r.calib._cache(tiny / "m")
    meta = json.loads(S.meta_path(path).read_text())
    assert meta["settings"]["gpu_memory_utilization"] == expected_budget
    assert ("vllm" in sys.modules) == had_vllm
    # the plain Python API takes the same route
    p2 = tiny / "p2.jsonl"
    p2.write_text(json.dumps({"id": "0", "prompt": "w9"}) + "\n")
    assert S.selfgen(tiny / "m", p2, max_tokens=32).is_file()
    assert json.loads((tiny / "llm-budget.json").read_text()) == DEFAULT_GPU_MEMORY_UTILIZATION
    assert ("vllm" in sys.modules) == had_vllm


@pytest.mark.gpu
def test_real_generation_then_build_equals_a_build_from_the_cache(real_model, tmp_path, monkeypatch):
    """Qwen3-0.6B, real vLLM in the child: a miss build and a hit build write the same bytes."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("needs vLLM in this venv (the child runs this interpreter)")
    model = real_model("qwen3-0.6b")
    had_vllm = "vllm" in sys.modules  # an earlier GPU test in this process may have imported it
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    prompts = ["Explain why the sky is blue.", "Write a haiku about rain.", "What is 17 times 23?",
               "Name three prime numbers.", "Describe a cat in one sentence.", "What is the capital of France?",
               "Give one tip for sleeping well.", "Translate 'thank you' into Spanish."]  # fmt: skip
    rows = (json.dumps({"id": str(i), "prompt": p}) + "\n" for i, p in enumerate(prompts))
    (tmp_path / "p.jsonl").write_text("".join(rows))
    src = SelfgenSource(prompts=str(tmp_path / "p.jsonl"), n=2, max_tokens=512, samples=8, seqlen=512)
    r = Recipe(steps=[GPTQ()], calib=src)
    outs = {
        name: build(str(model), r, format="w4a16-sym-g128", out_dir=tmp_path / name, devices=[0],
                    log=lambda s: None)
        for name in ("miss", "hit")
    }  # fmt: skip
    assert stage_kinds(outs["miss"]) == ["generate", "layers"] and stage_kinds(outs["hit"]) == ["layers"]
    assert ("vllm" in sys.modules) == had_vllm  # generation ran in the child
    assert weights(outs["miss"]) and weights(outs["miss"]) == weights(outs["hit"])


def test_too_few_batches_for_the_cards_is_refused_before_generation(tiny, recorder):
    """4 samples in batches of 4 are one batch: two cards cannot both get one, known from the config alone."""
    with pytest.raises(Refused, match="give 1 batches for 2 cards"):
        build(
            str(tiny / "m"), recipe(tiny, TinyRTN()), format="w4a16-sym-g128", out_dir=tiny / "out",
            devices=["cpu", "cpu"], batch_size=4, dtype=torch.float32, exporter=writes(FakeWriter), log=lambda s: None,
        )  # fmt: skip
    assert recorder.calls == []


def test_a_mix_forwards_pending_and_generation_to_its_selfgen_part(tiny, recorder):
    """A mix of sources plans a generate stage for the part that needs one, and records its cache."""
    from qda_quant.calib.sources import MixSource

    text = tiny / "t.jsonl"
    text.write_text(json.dumps({"text": "w3 w4 w5 w6 w7 w8 w9 w10"}) + "\n")
    calib = MixSource(
        parts=[
            {  # chunks, as the fixed recipe: pieces, the last of a row shorter
                "selfgen": {"prompts": str(tiny / "p.jsonl"), "max_tokens": 32, "seqlen": 8, "chunks": True},
                "weight": 3,
            },
            {"file": {"path": str(text), "seqlen": 8}, "weight": 1},
        ],
        samples=4,
    )
    r = Recipe(steps=[TinyRTN()], calib=calib)

    p = plan(tiny, r)
    assert recorder.calls == [] and p.calib_ids is None
    assert p.generate.startswith("selfgen:") and "p.jsonl" in p.generate

    out = run(tiny, r)
    info = json.loads((out / "build.json").read_text())
    stage = info["stages"][0]
    assert stage["kind"] == "generate" and stage["steps"] == ["mix"] and len(recorder.calls) == 1
    (part,) = stage["parts"]
    assert part["source"] == "selfgen" and part["cache"]["sha256"] == sha256_file(part["cache"]["path"])

    run(tiny, r, "out2")  # the cache is re-read: no second generation
    assert len(recorder.calls) == 1 and plan(tiny, r).generate is None


@pytest.mark.parametrize(
    ("steps", "devices", "match"),
    [
        ([TinyRTN()], [torch.device("cpu")] * 2, "image calibration builds on 1 card only"),
        ([Guided(), GPTQ()], CPU, "without the images"),
    ],
)
def test_an_image_mix_refuses_its_static_limits_at_plan(tiny, recorder, steps, devices, match):
    """Both limits come from the source and the step, so a pending generation must not defer them."""
    from qda_quant.calib.sources import MixSource

    calib = MixSource(
        parts=[
            {"selfgen": {"prompts": str(tiny / "p.jsonl"), "max_tokens": 32, "seqlen": 8}, "weight": 1},
            {"vlm": {}, "weight": 1},
        ],
        samples=4,
    )
    r = Recipe(steps=steps, calib=calib)
    assert calib.pending(str(tiny / "m")) is not None  # a cache miss: the build would generate first
    with pytest.raises(Refused, match=match):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16-sym-g128"), Target(), devices)
    assert recorder.calls == []


def test_with_chunks_the_batch_count_is_checked_after_generation(tiny, recorder):
    """With chunks, samples may differ in length (4 samples in batches of 4 can be 2+ batches): no early refusal.

    Here the rows happen to give 4 pieces of one length, so the check after generation, from the rows, refuses.
    """
    recorder.words = 40  # the first row alone gives the 4 pieces of 8
    r = Recipe(
        steps=[TinyRTN()],
        calib=SelfgenSource(prompts=str(tiny / "p.jsonl"), samples=4, seqlen=8, max_tokens=32, chunks=True),
    )
    with pytest.raises(Refused, match="1 calibration batches for 2 cards"):
        build(
            str(tiny / "m"), r, format="w4a16-sym-g128", out_dir=tiny / "out", devices=["cpu", "cpu"],
            batch_size=4, dtype=torch.float32, exporter=writes(FakeWriter), log=lambda s: None,
        )  # fmt: skip
    assert len(recorder.calls) == 1  # not refused before generation


def test_a_build_names_the_generated_calibration_samples_it_used(tiny, recorder, monkeypatch):
    """build.json's `calib_sha256` digests the samples the build fed its model; generated rows have no file a
    config could name, so the build is the only place that can record them. A cache hit gives the same digest,
    and other generated rows give another."""
    r = recipe(tiny, GPTQ())
    first = json.loads((run(tiny, r, "a", exporter=None) / "build.json").read_text())["calib_sha256"][
        "sha256"
    ]
    again = json.loads((run(tiny, r, "b", exporter=None) / "build.json").read_text())["calib_sha256"][
        "sha256"
    ]
    assert len(recorder.calls) == 1  # the second build read the cache
    assert len(first) == 64 and int(first, 16) >= 0 and again == first

    def shifted(model, prompts, n, max_tokens, seed, sampling, ctk, gpu_memory_utilization):
        return [
            [" ".join(f"w{(i + d + k) % 50 + 40}" for k in range(20)) for d in range(n)]
            for i in range(len(prompts))
        ]

    monkeypatch.setenv("QDA_CACHE", str(tiny / "cache2"))  # a miss: generate again, other rows this time
    monkeypatch.setattr(
        S, "generate_isolated", lambda *a, **k: S.selfgen(*a, **k, generate=shifted, tokenizer=None)
    )
    other = json.loads((run(tiny, r, "c", exporter=None) / "build.json").read_text())["calib_sha256"][
        "sha256"
    ]
    assert len(other) == 64 and other != first


def test_a_build_that_read_no_samples_says_its_digest_is_not_recorded(tiny, recorder):
    """Missing reads as missing: an explicit "not recorded: <why>", never an absent key or ""."""
    from qda_quant.steps.rtn import RTN

    info = json.loads(
        (run(tiny, recipe(tiny, RTN())) / "build.json").read_text()
    )  # minmax reads weights only
    rec = info["calib_sha256"]
    assert rec["sha256"].startswith("not recorded: ") and not recorder.calls
    # the field says what it covers and what it does not, so it cannot read as "all data recorded"
    assert "calibration forward" in rec["covers"] and "scale_kd" in rec["not_covered"]


def test_the_samples_digest_follows_the_samples_and_says_when_it_has_none():
    """Same samples, same digest; other ids or other image inputs, another; none or unhashable: "not recorded"."""
    from qda_quant.engine.calib import samples_sha256

    ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    assert samples_sha256(ids) == samples_sha256([x.clone() for x in ids])
    assert samples_sha256(ids) != samples_sha256([ids[0], torch.tensor([4, 5, 7])])
    assert samples_sha256(ids) != samples_sha256(ids[::-1])  # order is part of the data
    pix = [{"pixel_values": torch.zeros(2, 2)}, {}]
    other = [{"pixel_values": torch.ones(2, 2)}, {}]
    assert (
        samples_sha256(ids, pix)
        != samples_sha256(ids)
        != samples_sha256(ids, other)
        != samples_sha256(ids, pix)
    )
    for none in (None, []):
        assert samples_sha256(none).startswith("not recorded: the build read no")
    broken = samples_sha256([torch.tensor([1]), "not a tensor"])  # type: ignore[list-item]
    assert broken.startswith("not recorded: the samples could not be hashed")
