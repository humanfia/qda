"""The sanity suite: its registered settings, prompt selection, and the rule verifiers it scores with."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from qda_kl import serve as qda_serve
from qda_kl.eval import constraints, sanity
from qda_kl.eval.sanity import SanitySuite
from qda_kl.eval.suites import parse_suite
from qda_kl.gencheck import GencheckConfig, Generation, read_prompts
from qda_kl.serve import NO_JIT_ENV, ServeSettings, no_jit

GOOD = """Five sections, as asked.

1. **First heading**
This section has an example: a concrete case.

2. **Second heading**
Another example follows here.

3. **Third heading**
For example, this one.

4. **Fourth heading**
An example.

5. **Fifth heading**
For example, this section ends the answer with a summary sentence.
"""


def test_the_sanity_suite_is_registered_and_selects_eight_prompts_per_domain() -> None:
    suite = parse_suite({"sanity": {"chat_template_kwargs": {"enable_thinking": False}}}, "suites.0")
    assert isinstance(suite, SanitySuite)
    config = GencheckConfig(checkpoint="model", original="model", out=".", per_domain=suite.per_domain)
    prompts = read_prompts(config)
    assert len(prompts) == 4 * suite.per_domain
    assert {row.domain for row in prompts} == {"math", "code", "science", "instruction"}


def test_the_sanity_suite_refuses_its_own_draft() -> None:
    with pytest.raises(ValueError, match="speculative_config"):
        parse_suite({"sanity": {"serve_args": {"speculative_config": {"method": "mtp"}}}}, "suites.0")


def test_the_default_budget_and_the_whole_batch_reach_the_engine(monkeypatch) -> None:
    """16384 tokens and max_num_seqs = the prompt count, unless serve_args names one."""
    engines: list[dict[str, Any]] = []

    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)

        def collective_rpc(self, name):
            assert name == "qda_kl_kernels"
            return [{"dense": ["MacheteLinearKernel"]}]

        def get_tokenizer(self):
            return SimpleNamespace(
                chat_template="template", apply_chat_template=lambda *args, **kwargs: {"input_ids": [1]}
            )

        def generate(self, jobs, params, **kwargs):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text="a", token_ids=[1], finish_reason="stop")])
                for _ in jobs
            ]

    fake = SimpleNamespace(LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(qda_serve, "start_engine", lambda settings, start: start())
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    suite = parse_suite({"sanity": {}}, "suites.0")
    assert suite.max_tokens == 16384
    prompts = suite._prompts()
    rows, _ = suite._measure("model", suite._settings(), prompts, {"temperature": 0.0})
    assert len(rows) == len(prompts) == 32
    assert engines[0]["max_num_seqs"] == 32
    assert engines[0]["max_model_len"] >= suite.max_tokens
    explicit = SanitySuite(serve_args={"max_num_seqs": 4})
    explicit._measure("model", explicit._settings(), prompts, {"temperature": 0.0})
    assert engines[1]["max_num_seqs"] == 4


def test_the_selected_kernels_are_recorded_for_both_halves(tmp_path, monkeypatch) -> None:
    """sanity.json names each half's bound kernels and where a forced backend came from."""
    engines: list[dict[str, Any]] = []
    bound = [{"dense": ["MacheteLinearKernel"]}, {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]}]

    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)

        def collective_rpc(self, name):
            assert name == "qda_kl_kernels"
            rank = len(engines) - 1  # the first engine answers the original, the second the candidate
            return [bound[rank]]

        def get_tokenizer(self):
            return SimpleNamespace(
                chat_template="template", apply_chat_template=lambda *args, **kwargs: {"input_ids": [1]}
            )

        def generate(self, jobs, params, **kwargs):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text="a", token_ids=[1], finish_reason="stop")])
                for _ in jobs
            ]

    fake = SimpleNamespace(LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(qda_serve, "start_engine", lambda settings, start: start())
    monkeypatch.setattr(SanitySuite, "_sampling", lambda self, model: {"temperature": 0.0})
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    monkeypatch.setattr(sanity, "versions", lambda: {})
    monkeypatch.setattr(sanity, "vllm_version", lambda: "test")
    suite = SanitySuite(domains=["math"], per_domain=2, serve_args={"linear_backend": "marlin"})
    suite.prepare("original", tmp_path)
    result = suite.run("candidate", "original", tmp_path)
    serve = result.detail["serve"]
    assert engines[0]["worker_extension_cls"] == sanity.EXTENSION
    assert serve["original"]["kernels"] == {"dense": ["MacheteLinearKernel"]}
    assert serve["checkpoint"]["kernels"] == {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]}
    assert serve["checkpoint"]["backends"]["linear"] == {"value": "marlin", "source": "serve_args"}
    assert serve["checkpoint"]["backends"]["moe"] == {"value": None, "source": "default"}
    meta = json.loads((suite.reference_dir("original", tmp_path) / "meta.json").read_text())
    assert meta["kernels"] == {"dense": ["MacheteLinearKernel"]}


class _TextOf:
    """A config that exposes its text part only through ``get_text_config()``, as a composite config does."""

    def __init__(self, text: Any) -> None:
        self.text = text

    def get_text_config(self) -> Any:
        return self.text


def test_the_mtp_head_is_read_under_either_family_name() -> None:
    """DeepSeek names the head ``num_nextn_predict_layers``, Qwen3.5/3.8 ``mtp_num_hidden_layers``."""
    assert sanity.mtp_layers(SimpleNamespace(num_nextn_predict_layers=1)) == 1
    assert sanity.mtp_layers(SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=1))) == 1
    assert sanity.mtp_layers(SimpleNamespace()) == 0
    assert sanity.mtp_layers(SimpleNamespace(num_nextn_predict_layers=None)) == 0


def test_the_mtp_head_is_read_through_the_composite_config_interface() -> None:
    """The head lives on the language model's config; an outer config that has no such attribute still reports it."""
    outer = _TextOf(SimpleNamespace(mtp_num_hidden_layers=1))
    assert not hasattr(outer, "text_config")
    assert sanity.mtp_layers(outer) == 1


def _answers(prompts, *, text="4", token_ids=(1,), specdec=None):
    """One Generation per prompt and the run record a real measurement returns."""
    rows = [
        Generation(
            id=row.id,
            domain=row.domain,
            text=text if isinstance(text, str) else text(row),
            token_ids=list(token_ids),
            token_count=len(token_ids),
            finish_reason="stop",
        )
        for row in prompts
    ]
    return rows, {
        "chat_template": True,
        "generation_config": True,
        "load_s": 0.0,
        "kernels": None,
        "specdec": specdec,
    }


def _measure_with(*, text="4", token_ids=(1,), on_call=None, on_rows=None, mtp_specdec=False):
    """A fake ``_measure`` answering every prompt, reporting the call and the rows it returns."""

    def measure(self, model, settings, prompts, sampling):
        if on_call is not None:
            on_call(model, settings)
        specdec = {"acceptance_rate": 0.5} if mtp_specdec and sanity._mtp_layers(model) else None
        rows, record = _answers(prompts, text=text, token_ids=token_ids, specdec=specdec)
        if on_rows is not None:
            on_rows(model, rows)
        return rows, record

    return measure


def _offline(monkeypatch, measure, *, mtp=0) -> None:
    """Replace the engine and the run metadata a suite reads, so a test needs no GPU or model directory."""
    monkeypatch.setattr(SanitySuite, "_measure", measure)
    monkeypatch.setattr(SanitySuite, "_sampling", lambda self, model: {"temperature": 0.0})
    if mtp is not None:
        monkeypatch.setattr(sanity, "_mtp_layers", lambda model: mtp)
    monkeypatch.setattr(sanity, "versions", lambda: {})
    monkeypatch.setattr(sanity, "vllm_version", lambda: "test")


def test_a_new_mtp_detection_does_not_reuse_an_old_original_cache(tmp_path, monkeypatch) -> None:
    """A record written before the detector knew a family's head name is answered again after the fix."""
    _offline(monkeypatch, _measure_with(mtp_specdec=True), mtp=None)
    suite = SanitySuite(domains=["math"], per_domain=1)
    # The old detector returns 0 for a Qwen MTP config, so the original was answered without the draft.
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    suite.prepare("original", tmp_path)
    old = suite.reference_dir("original", tmp_path)
    assert json.loads((old / sanity.META).read_text())["specdec"] is None
    # The corrected detector finds the head: the same run directory must not serve the old answers.
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 1)
    suite.prepare("original", tmp_path)
    assert suite.reference_dir("original", tmp_path) != old
    result = suite.run("candidate", "original", tmp_path)
    assert result.metrics["mtp_acceptance_rate"] == 0.5
    assert result.metrics["mtp_acceptance_rate_ratio"] == 1.0


def test_an_original_answered_at_the_old_cap_is_not_reused(tmp_path, monkeypatch) -> None:
    """``max_tokens`` is in the original's cache key, so a 512-token record answers again at the default."""
    measured: list[str] = []
    _offline(monkeypatch, _measure_with(on_call=lambda model, _settings: measured.append(model)))
    old = SanitySuite(domains=["math"], per_domain=1, max_tokens=512)
    old.prepare("original", tmp_path)
    suite = SanitySuite(domains=["math"], per_domain=1)
    assert suite.reference_dir("original", tmp_path) != old.reference_dir("original", tmp_path)
    suite.prepare("original", tmp_path)
    assert measured == ["original", "original"]


def test_a_cache_answered_before_the_batch_was_recorded_is_not_reused(tmp_path, monkeypatch) -> None:
    """The effective ``max_num_seqs`` is in both identities, so an old-batching original answers again."""
    measured: list[tuple[str, dict[str, Any]]] = []
    _offline(
        monkeypatch,
        _measure_with(on_call=lambda model, settings: measured.append((model, settings.args))),
    )
    suite = SanitySuite(max_tokens=16384)
    # A record written before the derived batch entered the identity: its key carried the raw serve_args.
    with monkeypatch.context() as before:
        before.setattr(SanitySuite, "_engine_args", lambda self: dict(self.serve_args))
        suite.prepare("original", tmp_path)
        stale = suite.reference_dir("original", tmp_path)
    assert suite.reference_dir("original", tmp_path) != stale
    suite.prepare("original", tmp_path)
    assert [model for model, _ in measured] == ["original", "original"]
    assert measured[0][1] == no_jit(ServeSettings()).args  # the stale record: the old key's engine settings
    assert measured[1][1] == no_jit(ServeSettings(args={"max_num_seqs": 32})).args
    result = suite.run("candidate", "original", tmp_path)
    assert measured[2][1] == no_jit(ServeSettings(args={"max_num_seqs": 32})).args
    # Both records and the resolved settings name the batch that ran.
    meta = json.loads((suite.reference_dir("original", tmp_path) / sanity.META).read_text())
    assert meta["serve"] == no_jit(ServeSettings(args={"max_num_seqs": 32})).args
    assert suite.to_dict()["sanity"]["serve_args"] == {"max_num_seqs": 32}
    assert result.detail["serve"]["args"] == no_jit(ServeSettings(args={"max_num_seqs": 32})).args


def test_every_row_is_measured_with_the_suite_engine_settings(tmp_path, monkeypatch) -> None:
    """A candidate carrying its own serve section still runs the suite's engine setting, as the original does."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "build.json").write_text(json.dumps({"serve": {"args": {"moe_backend": "marlin"}}}))
    seen = []
    _offline(monkeypatch, _measure_with(on_call=lambda model, settings: seen.append((model, settings))))
    suite = SanitySuite(domains=["math"], per_domain=1, serve_args={"max_num_seqs": 4})
    suite.prepare("original", tmp_path)
    result = suite.run(str(candidate), "original", tmp_path)
    assert [model for model, _ in seen] == ["original", str(candidate)]
    assert seen[0][1] == seen[1][1] == no_jit(ServeSettings(args={"max_num_seqs": 4}))
    assert result.detail["serve"]["args"] == no_jit(ServeSettings(args={"max_num_seqs": 4})).args


def test_the_engines_compile_no_flashinfer_kernel(tmp_path, monkeypatch) -> None:
    """Both engines start without FlashInfer's JIT sampler, as scoring and gencheck engines do.

    The sampler is built at the first request; on a card without prebuilt kernels the build needs nvcc and a
    C++ compiler, and the suite refused with "Engine core initialization failed" where scoring ran fine.
    """
    seen = []
    _offline(monkeypatch, _measure_with(on_call=lambda model, settings: seen.append(settings)))
    suite = SanitySuite(domains=["math"], per_domain=1)
    suite.prepare("original", tmp_path)
    suite.run("candidate", "original", tmp_path)
    assert len(seen) == 2
    for settings in seen:
        assert settings.env.items() >= NO_JIT_ENV.items()
        assert settings.args["gdn_prefill_backend"] == "triton"


def test_the_candidate_answers_are_kept_beside_its_result(tmp_path, monkeypatch) -> None:
    """A candidate's answers are written next to its sanity.json, in the original's format, with path and sha256."""
    measured: list[list[Generation]] = []

    def keep(model, rows):
        if model == "candidate":
            measured.append(rows)

    _offline(
        monkeypatch,
        _measure_with(text=lambda row: f"answer to {row.id}", token_ids=(1, 2), on_rows=keep),
    )
    suite = SanitySuite(domains=["math"], per_domain=2)
    suite.prepare("original", tmp_path)
    suite.result_path = tmp_path / "results" / "candidate" / "sanity.json"  # the runner sets this before run
    result = suite.run("candidate", "original", tmp_path)
    stored = tmp_path / "results" / "candidate" / sanity.ANSWERS
    rows = [Generation.model_validate_json(line) for line in stored.read_text().splitlines()]
    assert [(row.id, row.text) for row in rows] == [(row.id, row.text) for row in measured[0]]
    assert len(rows) == result.detail["gencheck"]["total"]["checkpoint"]["n"]
    assert sum(row["n"] for row in result.detail["gencheck"]["checkpoint"].values()) == len(rows)
    assert result.detail["answers"] == {
        "path": "results/candidate/answers.jsonl",
        "sha256": hashlib.sha256(stored.read_bytes()).hexdigest(),
    }


def _fake_engine(cuts: set[tuple[str, int, int]], loops: set[tuple[str, int, int]]):
    """A fake ``_measure``: one answer per (model, seed, prompt), cut or looping where the sets say.

    The index is the prompt's position in the suite's selection; rows come back seed-major, as the engine
    orders them.
    """

    def measure(self, model, settings, prompts, sampling):
        rows = []
        for seed in self.seeds:
            for index, prompt in enumerate(prompts):
                token_ids = [7] * 300 if (model, seed, index) in loops else [1]
                rows.append(
                    Generation(
                        id=prompt.id,
                        domain=prompt.domain,
                        text="answer",
                        token_ids=token_ids,
                        token_count=len(token_ids),
                        finish_reason="length" if (model, seed, index) in cuts else "stop",
                    )
                )
        return rows, {
            "chat_template": True,
            "generation_config": True,
            "load_s": 0.0,
            "kernels": None,
            "specdec": None,
        }

    return measure


def test_one_seed_reports_todays_keys(tmp_path, monkeypatch) -> None:
    """Without ``seeds`` a run answers seed 17 once per prompt and its result keeps today's shape."""
    _offline(monkeypatch, _fake_engine(cuts={("candidate", 17, 0)}, loops={("candidate", 17, 1)}))
    suite = SanitySuite(domains=["math"], per_domain=2)
    assert suite.seeds == [17]
    suite.prepare("original", tmp_path)
    result = suite.run("candidate", "original", tmp_path)
    assert result.metrics == {
        "cut_rate": 0.5,
        "loop_rate": 0.5,
        "chat_template": 1.0,
        "generation_config": 1.0,
    }
    assert set(result.detail) == {
        "gencheck",
        "sampling",
        "seed",
        "prompts",
        "load_s",
        "vllm",
        "versions",
        "serve",
    }
    # Naming the one seed explicitly is the same run, byte for byte.
    explicit = SanitySuite(domains=["math"], per_domain=2, seeds=[17])
    assert suite.run("candidate", "original", tmp_path).model_dump_json() == (
        explicit.run("candidate", "original", tmp_path).model_dump_json()
    )


def test_two_seeds_pool_their_counts(tmp_path, monkeypatch) -> None:
    """The pooled block counts every answer of every seed; the per-seed blocks add up to it."""
    _offline(
        monkeypatch,
        _fake_engine(
            cuts={("candidate", 17, 0), ("candidate", 18, 1), ("original", 18, 0)},
            loops={("candidate", 18, 1), ("original", 17, 0)},
        ),
    )
    suite = SanitySuite(domains=["math"], per_domain=2, seeds=[17, 18])
    suite.prepare("original", tmp_path)
    result = suite.run("candidate", "original", tmp_path)
    per_seed = result.detail["per_seed"]
    pooled = result.detail["pooled"]
    assert [entry["seed"] for entry in per_seed] == [17, 18]
    for side in ("checkpoint", "original"):
        assert pooled[side]["n"] == 4
        assert pooled[side]["cuts"] == sum(entry[side]["cuts"] for entry in per_seed)
        assert pooled[side]["loops"] == sum(entry[side]["loops"] for entry in per_seed)
    assert (pooled["checkpoint"]["cuts"], pooled["checkpoint"]["loops"]) == (2, 1)
    assert (pooled["original"]["cuts"], pooled["original"]["loops"]) == (1, 1)
    assert pooled["checkpoint"]["cut_rate"] == 0.5
    assert pooled["checkpoint"]["loop_rate"] == 0.25
    assert len(pooled["checkpoint"]["loop_ids"]) == 1
    assert len(pooled["original"]["loop_ids"]) == 1
    assert pooled["checkpoint"]["loop_ids"] != pooled["original"]["loop_ids"]


def test_each_seed_is_part_of_the_original_cache_key(tmp_path, monkeypatch) -> None:
    """A reference record holds exactly the seeds it was measured with; no other seed set reuses it."""
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    suites = (
        SanitySuite(domains=["math"], per_domain=1),
        SanitySuite(domains=["math"], per_domain=1, seed=18, seeds=[18]),
        SanitySuite(domains=["math"], per_domain=1, seeds=[17, 18]),
    )
    assert len({suite.reference_dir("original", tmp_path) for suite in suites}) == 3


def test_seeds_must_begin_with_seed_and_not_repeat() -> None:
    with pytest.raises(ValueError, match=r"seeds\[0\]"):
        SanitySuite(seed=17, seeds=[18, 19])
    with pytest.raises(ValueError, match="repeats"):
        SanitySuite(seeds=[17, 17])


def test_several_seeds_refuse_greedy_decoding(tmp_path, monkeypatch) -> None:
    """Over greedy decoding every seed answers identically, so the pooled counts would be copies."""
    measured: list[str] = []

    def measure(self, model, settings, prompts, sampling):
        measured.append(model)
        return [], {"chat_template": True, "generation_config": True, "load_s": 0.0, "specdec": None}

    monkeypatch.setattr(SanitySuite, "_measure", measure)
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    model = tmp_path / "model"  # no generation_config.json: neither side names a setting
    model.mkdir()
    with pytest.raises(ValueError, match="set sanity.sampling"):
        SanitySuite(domains=["math"], per_domain=1, seeds=[17, 18]).prepare(str(model), tmp_path)
    assert measured == [], "refused before the engine answers"
    with pytest.raises(ValueError, match="set sanity.sampling"):
        SanitySuite(seeds=[17, 18], sampling={"temperature": 0.0})._sampling(str(model))
    assert SanitySuite(seeds=[17])._sampling(str(model)) == {"temperature": 0.0}
    sampled = SanitySuite(seeds=[17, 18], sampling={"temperature": 1.0})
    assert sampled._sampling(str(model)) == {"temperature": 1.0}


def test_a_conforming_answer_passes_every_constraint() -> None:
    got = constraints.score([GOOD])
    assert got is not None
    assert got["instruction_rate"] == 1.0
    assert got["prompt_rate"] == 1.0


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        (GOOD.replace("4. **Fourth heading**\nAn example.\n\n", ""), "sections"),
        (GOOD.replace("**Third heading**", ""), "headings"),
        (GOOD.replace("An example.", "Nothing else here."), "examples"),
        (GOOD.replace("a summary sentence.", "a summary. It has two sentences."), "summary"),
    ],
)
def test_each_rule_catches_its_own_violation(change: str, rule: str) -> None:
    got = constraints.score([change])
    assert got is not None
    assert got["rules"][rule] == 0.0
    assert got["prompt_rate"] == 0.0
