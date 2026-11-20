"""Long-generation checks, independent of vLLM and GPU availability."""

import hashlib
import json
import os
import random
import re
import sys
from types import SimpleNamespace

import pytest
import yaml
from qda_kl import cli, gencheck
from qda_kl.eval import constraints
from qda_kl.eval.suites import sha256_rows
from qda_kl.gencheck import GencheckConfig, Generation, build_report, is_loop, read_prompts
from qda_kl.serve import ServeSettings
from qda_quant.step import Refused


@pytest.mark.parametrize("period", [1, 7, 37, 512])
def test_periodic_tail(period):
    tokens = list(range(period)) * max(4, 2048 // period)
    assert is_loop(list(range(10000, 12048)) + tokens)


def test_short_answer_is_not_a_loop():
    assert not is_loop([1] * 255)


def test_natural_text_is_not_a_loop():
    text = (
        "A useful solution starts by stating the assumptions and deriving each intermediate result. "
        "Next we check the boundary conditions, compare alternatives, and explain why the proposed "
        "construction works. Several examples illustrate the distinction between necessity and sufficiency. "
        "The final argument follows from induction, together with a separate verification of the base case."
    )
    assert not is_loop(list(text.encode()))


def test_long_natural_text_is_not_a_loop():
    rng = random.Random(17)
    locations = ("northern wetlands", "rocky shoreline", "sheltered valley", "coastal grassland")
    teams = ("hydrologists", "field biologists", "research assistants", "ecologists")
    subjects = ("stream temperature", "plant growth", "insect activity", "soil moisture", "water quality")
    methods = (
        "recorded readings from several independent stations",
        "checked the measurements against their field notes",
        "compared sheltered and exposed sampling sites",
        "repeated each observation with a second instrument",
    )
    findings = (
        "a gradual response to recent rainfall",
        "a stronger effect near the edge of the habitat",
        "little change in the shaded plots",
        "a brief recovery after cooler weather",
    )
    limitations = (
        "the role of wind",
        "the influence of earlier flooding",
        "the effect of nearby construction",
        "the contribution of measurement error",
    )
    paragraphs = []
    for _ in range(64):
        paragraphs.append(
            f"In the {rng.choice(locations)}, the {rng.choice(teams)} monitored "
            f"{rng.choice(subjects)} over {rng.randrange(12, 90)} days. "
            f"They {rng.choice(methods)} to separate seasonal changes from local effects. "
            f"The observations suggested {rng.choice(findings)}, although "
            f"{rng.choice(limitations)} remained uncertain. "
            f"The next visit will compare {rng.choice(subjects)} at "
            f"{rng.choice(('dawn', 'midday', 'dusk'))} and repeat the measurements under similar conditions."
        )
    assert len(set(paragraphs)) == len(paragraphs)
    words = re.findall(r"\w+|[^\w\s]", "\n\n".join(paragraphs))
    vocabulary = {word: index for index, word in enumerate(sorted(set(words)))}
    token_ids = [vocabulary[word] for word in words]
    assert len(token_ids) >= 3000
    assert not is_loop(token_ids)


def test_noise_in_periodic_tail():
    tokens = list(range(37)) * 55
    for index in random.Random(17).sample(range(len(tokens)), len(tokens) // 100):
        tokens[index] = 10000 + index
    assert is_loop(tokens)


def test_only_first_half_loops():
    tokens = [1] * 2048 + list(range(2048))
    assert not is_loop(tokens)


def test_loop_must_cover_half_the_window():
    assert not is_loop(list(range(1100)) + [1] * 948)
    assert is_loop(list(range(1024)) + [1] * 1024)


def test_four_full_periods_are_required():
    tokens = list(range(80)) * 3
    assert not is_loop(tokens, min_tokens=1)
    assert is_loop(tokens + list(range(80)))


@pytest.mark.parametrize(
    "settings",
    [{"window": 0}, {"max_period": 0}, {"min_tokens": 0}, {"min_repeats": 1}, {"mismatch_rate": 1}],
)
def test_bad_detector_settings(settings):
    with pytest.raises(ValueError):
        is_loop([], **settings)


@pytest.fixture
def config(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps({"id": f"{domain}-{index}", "domain": domain, "prompt": "Explain a proof."}) + "\n"
            for index in range(3)
            for domain in ("math", "code", "science", "instruction")
        )
    )
    return GencheckConfig(
        checkpoint="checkpoint",
        original="original",
        prompts=str(prompts),
        out=str(tmp_path / "out"),
        per_domain=2,
        max_tokens=300,
    )


def answer(prompt, *, loop=False, cut=False, length=300):
    return Generation(
        id=prompt.id,
        domain=prompt.domain,
        text="private generated text" if length else "",
        token_ids=[7] * length if loop else list(range(length)),
        token_count=length,
        finish_reason="length" if cut else "stop",
    )


def recording_answers(calls, length=300):
    """Stand in for gencheck._generate: record the model that ran, answer every prompt."""

    def generate(model, prompts, cfg):
        calls.append(model)
        return [answer(prompt, length=length) for prompt in prompts]

    return generate


def fake_engine(monkeypatch, record):
    """The smallest vLLM engine the runner accepts; ``record`` sees each construction's kwargs."""
    from types import SimpleNamespace as NS

    class Engine:
        def __init__(self, **kwargs):
            record(kwargs)

        def get_tokenizer(self):
            return NS(apply_chat_template=lambda *args, **kwargs: {"input_ids": [1]})

        def get_default_sampling_params(self):
            return NS(clone=lambda: NS())

        def generate(self, jobs, params, **kwargs):
            output = NS(outputs=[NS(text="answer", token_ids=[9], finish_reason="stop")])
            return [output for _ in jobs]

    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=Engine, TokensPrompt=dict))
    monkeypatch.setattr(gencheck, "_context_tokenizer", lambda model, kwargs: fake_tokenizer())


def fake_tokenizer(lengths=None):
    """A chat tokenizer whose prompt i has ``lengths[i]`` tokens (3 for every prompt by default), in call order."""
    calls = iter(lengths or [])

    def template(messages, **kwargs):
        return {"input_ids": [1] * next(calls, 3)}

    return SimpleNamespace(apply_chat_template=template)


@pytest.fixture
def cache_config(config, monkeypatch):
    """A config with an original cache and a fixed identity, which every cache test needs."""
    config.original_cache = str(gencheck.Path(config.out).parent / "cache")
    monkeypatch.setattr(gencheck, "_original_identity", lambda model: {"model": model})
    return config


def builtin_config(**kwargs):
    """The shipped prompt set at its full 24 rows per domain, optionally narrowed."""
    return GencheckConfig(checkpoint="checkpoint", original="original", out="out", per_domain=24, **kwargs)


def test_report_counts_deltas_and_totals(config):
    prompts = read_prompts(config)
    original = [answer(prompt, length=200) for prompt in prompts]
    checkpoint = [
        answer(prompt, loop=index == 0, cut=index in (0, 4), length=300 if index < 4 else 200)
        for index, prompt in enumerate(prompts)
    ]
    report = build_report(checkpoint, original, config)
    assert report["checkpoint"]["math"] == {"n": 2, "cut": 2, "loops": 1, "mean_tokens": 250.0}
    assert report["original"]["math"] == {"n": 2, "cut": 0, "loops": 0, "mean_tokens": 200.0}
    assert report["delta"]["math"] == {"cut": 2, "loops": 1, "mean_tokens": 50.0}
    assert report["total"] == {
        "checkpoint": {"n": 8, "cut": 2, "loops": 1},
        "original": {"n": 8, "cut": 0, "loops": 0},
    }
    assert report["max_tokens"] == 300 and report["per_domain"] == 2


def test_report_requires_matching_answers(config):
    original = [answer(prompt) for prompt in read_prompts(config)]
    with pytest.raises(ValueError, match="matching, unique"):
        build_report(original[:-1], original, config)
    with pytest.raises(ValueError, match="matching, unique"):
        build_report(original + [original[0]], original + [original[0]], config)


def test_selects_first_n_in_stable_file_order(config):
    selected = read_prompts(config.model_copy(update={"domains": ["science", "math"]}))
    assert [prompt.id for prompt in selected] == ["math-0", "science-0", "math-1", "science-1"]


def test_invalid_prompt_sets_refused(config):
    with pytest.raises(ValueError, match="need 4 prompts, found 3"):
        read_prompts(config.model_copy(update={"per_domain": 4}))
    path = gencheck.Path(config.prompts)
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="unique"):
        read_prompts(config)


def test_config_defaults_and_overrides(config):
    raw = config.model_dump()
    minimal = {key: raw[key] for key in ("checkpoint", "original", "out")}
    parsed = GencheckConfig.from_yaml(yaml.safe_dump(minimal), ["max_tokens=4096", "per_domain=8"])
    assert parsed.max_tokens == 4096 and parsed.per_domain == 8
    assert parsed.prompts == "builtin" and parsed.domains == ["math", "code", "science", "instruction"]
    assert parsed.seed == 17 and parsed.temperature == 0.6 and parsed.tp == 1
    defaults = GencheckConfig.from_yaml(yaml.safe_dump(minimal))
    assert defaults.per_domain == 4 and defaults.max_tokens == 8192


def test_serve_args_reject_tensor_parallel_size(config):
    # The card count belongs to tp; serve_args may not grow a second one that the runner ignores.
    raw = config.model_dump()
    raw["serve_args"] = {"tensor_parallel_size": 2}
    with pytest.raises(ValueError, match="set tp instead"):
        GencheckConfig(**raw)


def test_prompt_set_is_the_source_of_domains(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    rows = [
        {"id": f"{domain}-{index}", "domain": domain, "prompt": "Explain."}
        for domain in ("chinese", "tool-use")
        for index in range(2)
    ]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    base = {
        "checkpoint": "checkpoint",
        "original": "original",
        "prompts": str(prompts),
        "out": "out",
        "per_domain": 2,
    }
    config = GencheckConfig(**base)
    assert config.domains == ["chinese", "tool-use"]
    assert [prompt.domain for prompt in read_prompts(config)] == [
        "chinese",
        "chinese",
        "tool-use",
        "tool-use",
    ]
    with pytest.raises(ValueError, match=r"available: chinese, tool-use"):
        GencheckConfig(**base, domains=["math"])


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": 1},
        {"per_domain": 0},
        {"max_tokens": 0},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"tp": 0},
        {"domains": []},
        {"domains": ["math", "math"]},
        {"domains": ["other"]},
    ],
)
def test_bad_config_refused(config, change):
    with pytest.raises(ValueError):
        GencheckConfig.from_yaml(yaml.safe_dump(config.model_dump() | change))


def test_cli_success_writes_counts_only_and_private_texts(config, tmp_path, monkeypatch, capsys):
    calls = []

    def generate(model, prompts, cfg):
        calls.append(model)
        assert cfg.max_tokens == 512
        return [answer(prompt) for prompt in prompts]

    monkeypatch.setattr(gencheck, "_generate", generate)
    path = tmp_path / "gencheck.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    assert cli.main(["gencheck", str(path), "--set", "max_tokens=512"]) == 0
    assert calls == ["checkpoint", "original"]
    out = gencheck.Path(config.out)
    report_text = (out / "gencheck.json").read_text()
    assert json.loads(report_text)["total"]["checkpoint"]["n"] == 8
    for secret in ("private generated text", "token_ids", "prompt", '"text"'):
        assert secret not in report_text and secret not in capsys.readouterr().out
    for name in ("checkpoint", "original"):
        assert "private generated text" in (out / "texts" / f"{name}.jsonl").read_text()
    resolved = GencheckConfig.from_yaml((out / "resolved.yaml").read_text())
    assert resolved.max_tokens == 512


@pytest.mark.parametrize("language_only", [False, True])
def test_a_gguf_checkpoint_is_decoded_against_the_original_and_answered_by_vllm(
    config, tmp_path, monkeypatch, language_only
):
    """A .gguf is decoded once into <out>/decoded, generated from there, and the decode removed after."""
    from qda_kl import gguf_checkpoint

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF fixture")
    original = tmp_path / "original"
    original.mkdir()
    (original / "config.json").write_text("{}")
    config.checkpoint, config.original = str(gguf), str(original)
    decodes, calls = [], []

    def decode(file, snapshot, out):
        decodes.append((file, snapshot))
        target = out / gguf_checkpoint.DECODED
        target.mkdir(parents=True)
        return target, {"gguf": {"sha256": "f" * 64}, "config": {"note": "x"} if language_only else {}}

    def generate(model, prompts, cfg):
        calls.append((model, cfg.serve_args, gencheck.Path(model).is_dir()))
        return [answer(prompt) for prompt in prompts]

    monkeypatch.setattr(gguf_checkpoint, "decode", decode)
    monkeypatch.setattr(gencheck, "_generate", generate)
    report = gencheck.run_gencheck(config)
    out = gencheck.Path(config.out)
    assert decodes == [(gguf, original)]
    assert calls[0] == (str(out / "decoded"), {"language_model_only": True} if language_only else {}, True)
    assert calls[1] == (str(original), {}, True)
    assert not (out / "decoded").exists()
    assert report["gguf"] == {"sha256": "f" * 64} and "engine" not in report
    assert json.loads((out / "gencheck.json").read_text()) == report


def test_a_gguf_against_a_hub_original_is_refused_before_any_generation(config, tmp_path, monkeypatch):
    """The decoder needs the snapshot on disk: a Hub id as `original` is refused with the fix."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF fixture")
    config.checkpoint, config.original = str(gguf), "Qwen/Qwen3-0.6B"
    monkeypatch.setattr(gencheck, "_generate", lambda *a: pytest.fail("generated"))
    with pytest.raises(Refused, match="HF snapshot") as got:
        gencheck.run_gencheck(config)
    assert "set `original:` to the local HF snapshot" in got.value.fix
    # what a qda-agent user sets: the judge writes `original:` itself
    assert "`judge.anchor`" in got.value.fix


def test_a_llamacpp_section_is_refused(config):
    """The llama-server generator is gone: a config naming one is refused, not silently read with vLLM."""
    with pytest.raises(Refused, match="llamacpp"):
        GencheckConfig.from_yaml(yaml.safe_dump(config.model_dump() | {"llamacpp": {"server": "s"}}))


@pytest.mark.parametrize("error, expected", [(Refused("bad prompts"), 2), (RuntimeError("engine"), 1)])
def test_cli_failure_removes_old_result(config, tmp_path, monkeypatch, error, expected):
    def fail(*args):
        raise error

    monkeypatch.setattr(gencheck, "_generate", fail)
    out = gencheck.Path(config.out)
    out.mkdir()
    (out / "gencheck.json").write_text("old")
    path = tmp_path / "check.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    assert cli.main(["gencheck", str(path)]) == expected
    assert not (out / "gencheck.json").exists()


def test_generation_failure_names_the_engine_process_own_error(config, monkeypatch):
    """The caller's error carries the child's own failure line, not only its exit code (11-16)."""

    def run(argv, check=False):
        gencheck.Path(argv[5]).write_text(
            "Refused: max_num_seqs=1024 exceeds available mamba cache blocks (653)\n"
        )
        return gencheck.subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(gencheck.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exceeds available mamba cache blocks"):
        gencheck._generate("checkpoint", [answer(prompt) for prompt in read_prompts(config)][:1], config)


def test_generation_failure_without_a_reported_cause_keeps_the_exit_code(config, monkeypatch):
    def run(argv, check=False):
        return gencheck.subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(gencheck.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=r"engine process exited 1\)$"):
        gencheck._generate("checkpoint", [answer(prompt) for prompt in read_prompts(config)][:1], config)


def test_cli_bad_override_removes_old_result(config, tmp_path):
    out = gencheck.Path(config.out)
    out.mkdir()
    (out / "gencheck.json").write_text("old")
    path = tmp_path / "check.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    assert cli.main(["gencheck", str(path), "--set", "unknown=1"]) == 2
    assert not (out / "gencheck.json").exists()


def test_cli_missing_config_is_a_refusal(tmp_path, capsys):
    """The config path the user gave is the input, not a crash: the contract's first line and status 2."""
    path = tmp_path / "missing.yaml"
    assert cli.main(["gencheck", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"error: {path}: ") and "[Errno 2]" in err and "Traceback" not in err


def test_builtin_prompts_are_complete_unique_and_documented():
    from importlib.resources import files

    config = builtin_config()
    prompts = read_prompts(config)
    assert len(prompts) == 96
    assert len({prompt.id for prompt in prompts}) == len(prompts)
    assert {prompt.domain for prompt in prompts} == {"math", "code", "science", "instruction"}
    assert all(prompt.id and prompt.domain and prompt.prompt.strip() for prompt in prompts)
    readme = files("qda_kl").joinpath("data/README.md").read_text()
    for revision in (
        "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
        "802411c3010cb00d1b05bad57ca77365a3c699d6",
        "210d026faf9955653af8916fad021475a3f00453",
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    ):
        assert revision in readme


def test_math_prompts_exclude_amc_aime_and_named_contests():
    config = builtin_config(domains=["math"])
    prompts = read_prompts(config)
    for prompt in prompts:
        assert prompt.source != "amc_aime"
        assert prompt.source in {"olympiads", "aops_forum"}
        assert prompt.id.startswith("numina-train-00000-")
        assert "aime" not in prompt.prompt.casefold()
        assert "hmmt" not in prompt.prompt.casefold()


def test_code_prompts_use_competition_training_descriptions():
    config = builtin_config(domains=["code"])
    prompts = read_prompts(config)
    assert all(prompt.id.startswith("code-contests-train-00014-") for prompt in prompts)
    assert all(prompt.source == "code_contests" for prompt in prompts)


def test_builtin_prompt_set_digest_changed_with_the_new_rows():
    """The shipped rows no longer repeat the evaluation reference prompts.

    The sanity suite keys its cached originals on this digest, so replacing the set must change it. The
    second value is the superseded set's digest; a regression to those rows fails here.
    """
    config = builtin_config()
    prompts = read_prompts(config)
    digest = sha256_rows([prompt.model_dump() for prompt in prompts])
    assert digest == "7b37112865cf7869554850460d56883b8ea19e838f7aebca2990c901714c75ca"
    assert digest != "bfca822118ca1961e17ed4dc17abfef69b8e83a9f1ee2465af1f9ec27dd63474"


def test_builtin_instruction_prompts_ask_for_every_checked_constraint():
    config = builtin_config(domains=["instruction"])
    prompts = read_prompts(config)
    assert len(prompts) == 24
    for prompt in prompts:
        text = prompt.prompt.casefold()
        for request in (
            "exactly five numbered sections",
            "descriptive heading",
            "concrete example",
            "one-sentence summary",
        ):
            assert request in text
    assert set(constraints.CHECKS) == {"sections", "headings", "examples", "summary"}


def test_original_cache_reuses_answers_across_checkpoints(cache_config, monkeypatch, capsys):
    config = cache_config
    calls = []
    monkeypatch.setattr(gencheck, "_generate", recording_answers(calls, config.max_tokens))
    first = gencheck.run_gencheck(config)
    assert "reusing cached original generations" not in capsys.readouterr().err
    config.checkpoint = "another-checkpoint"
    second = gencheck.run_gencheck(config)
    assert "qda-kl: reusing cached original generations" in capsys.readouterr().err
    assert first == second
    assert calls == ["checkpoint", "original", "another-checkpoint"]
    entries = list(gencheck.Path(config.original_cache).glob("*/generations.jsonl"))
    assert len(entries) == 1
    manifest = json.loads((entries[0].parent / "complete.json").read_text())
    assert manifest == {
        "cache_key": entries[0].parent.name,
        "record_count": len(read_prompts(config)),
        "sha256": hashlib.sha256(entries[0].read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": 512},
        {"seed": 18},
        {"temperature": 0.7},
        {"original": "another-original"},
        {"enable_thinking": False},
        {"tp": 2},
        {"serve_args": {"max_model_len": 32768}},
        {"nccl_nvls": True},
    ],
)
def test_original_cache_key_includes_sampling_inputs(cache_config, change):
    config = cache_config
    prompts = read_prompts(config)
    first = gencheck.original_cache_key(config, prompts)
    assert gencheck.original_cache_key(config.model_copy(update=change), prompts) != first


def test_original_cache_key_includes_prompt_contents_and_model_defaults(config, monkeypatch):
    identity = {"model": "original", "generation_config": {"top_p": 0.9}}
    monkeypatch.setattr(gencheck, "_original_identity", lambda model: identity)
    prompts = read_prompts(config)
    first = gencheck.original_cache_key(config, prompts)
    assert gencheck.original_cache_key(config, list(reversed(prompts))) != first
    prompts[0].prompt += " More detail, please."
    assert gencheck.original_cache_key(config, prompts) != first
    prompts = read_prompts(config)
    identity["generation_config"]["top_p"] = 0.95
    assert gencheck.original_cache_key(config, prompts) != first


def test_domain_order_reuses_identical_selected_prompts(cache_config):
    config = cache_config
    reordered = config.model_copy(update={"domains": list(reversed(config.domains))})
    prompts = read_prompts(config)
    assert read_prompts(reordered) == prompts
    assert gencheck.original_cache_key(config, prompts) == gencheck.original_cache_key(reordered, prompts)


def test_corrupt_original_cache_is_regenerated(cache_config, monkeypatch):
    config = cache_config
    key = gencheck.original_cache_key(config, read_prompts(config))
    cache = gencheck.Path(config.original_cache) / key / "generations.jsonl"
    cache.parent.mkdir(parents=True)
    cache.write_text("incomplete json")
    calls = []
    monkeypatch.setattr(gencheck, "_generate", recording_answers(calls))
    gencheck.run_gencheck(config)
    assert calls == ["checkpoint", "original"]
    assert gencheck._cached_original(cache, read_prompts(config)) is not None


def test_empty_token_record_without_completion_manifest_is_regenerated(cache_config, monkeypatch):
    config = cache_config
    config.domains = ["math"]
    config.per_domain = 1
    prompts = read_prompts(config)
    key = gencheck.original_cache_key(config, prompts)
    cache = gencheck.Path(config.original_cache) / key / "generations.jsonl"
    cache.parent.mkdir(parents=True)
    partial = {
        "id": prompts[0].id,
        "domain": "math",
        "text": "",
        "token_ids": [],
        "finish_reason": "stop",
    }
    cache.write_text(json.dumps(partial) + "\n")
    assert gencheck._cached_original(cache, prompts) is None
    calls = []
    monkeypatch.setattr(gencheck, "_generate", recording_answers(calls, 10))
    gencheck.run_gencheck(config)
    assert calls == ["checkpoint", "original"]
    cached = gencheck._cached_original(cache, prompts)
    assert cached is not None and cached[0].token_count == 10


def test_truncated_original_cache_with_stale_manifest_is_regenerated(cache_config, monkeypatch):
    config = cache_config
    prompts = read_prompts(config)
    key = gencheck.original_cache_key(config, prompts)
    cache = gencheck.Path(config.original_cache) / key / "generations.jsonl"
    calls = []
    monkeypatch.setattr(gencheck, "_generate", recording_answers(calls, 10))
    gencheck.run_gencheck(config)
    completion = cache.parent / "complete.json"
    stale_manifest = completion.read_bytes()
    cache.write_bytes(b"\n".join(cache.read_bytes().splitlines()[:-1]) + b"\n")
    assert completion.read_bytes() == stale_manifest
    assert gencheck._cached_original(cache, prompts) is None
    gencheck.run_gencheck(config)
    assert calls == ["checkpoint", "original", "checkpoint", "original"]
    assert gencheck._cached_original(cache, prompts) is not None


def test_completed_immediate_eos_answer_is_reused(cache_config, monkeypatch):
    config = cache_config
    calls = []

    def generate(model, prompts, cfg):
        calls.append(model)
        return [answer(prompt, length=0 if model == "original" else 10) for prompt in prompts]

    monkeypatch.setattr(gencheck, "_generate", generate)
    first = gencheck.run_gencheck(config)
    second = gencheck.run_gencheck(config)
    assert calls == ["checkpoint", "original", "checkpoint"]
    assert first == second
    assert all(group["mean_tokens"] == 0 for group in second["original"].values())


@pytest.mark.parametrize("change", [{"cache_key": "wrong-key"}, {"record_count": 999}, {"sha256": "0" * 64}])
def test_invalid_completion_manifest_is_refused(config, tmp_path, change):
    prompts = read_prompts(config)
    cache = tmp_path / "key" / "generations.jsonl"
    gencheck._write_original_cache(cache, [answer(prompt) for prompt in prompts])
    completion = cache.parent / "complete.json"
    manifest = json.loads(completion.read_text()) | change
    completion.write_text(json.dumps(manifest))
    assert gencheck._cached_original(cache, prompts) is None


@pytest.mark.parametrize("change", [{"finish_reason": "abort"}, {"token_count": 0}, {"token_count": None}])
def test_completed_cache_still_validates_each_record(config, tmp_path, change):
    prompts = read_prompts(config)
    cache = tmp_path / "key" / "generations.jsonl"
    gencheck._write_original_cache(cache, [answer(prompt) for prompt in prompts])
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    rows[0].update(change)
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
    completion = cache.parent / "complete.json"
    manifest = json.loads(completion.read_text())
    manifest["sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    completion.write_text(json.dumps(manifest))
    assert gencheck._cached_original(cache, prompts) is None


def test_failed_completion_publish_leaves_no_reusable_cache(config, tmp_path, monkeypatch):
    prompts = read_prompts(config)
    rows = [answer(prompt) for prompt in prompts]
    cache = tmp_path / "key" / "generations.jsonl"
    gencheck._write_original_cache(cache, rows)
    publish = cli.publish
    calls = []

    def fail_completion(path, text):
        calls.append(path.name)
        if path.name == "complete.json":
            assert cache.is_file()
            raise OSError("injected completion publication failure")
        assert not (cache.parent / "complete.json").exists()
        publish(path, text)

    monkeypatch.setattr(cli, "publish", fail_completion)
    with pytest.raises(OSError, match="completion publication failure"):
        gencheck._write_original_cache(cache, rows)
    assert calls == ["generations.jsonl", "complete.json"]
    assert gencheck._cached_original(cache, prompts) is None
    assert not list(cache.parent.glob("*.tmp"))


def test_vllm_runner_uses_original_sampling_and_generated_token_ids(config, monkeypatch):
    calls = {}

    class Params(SimpleNamespace):
        def clone(self):
            return Params(**vars(self))

    class Engine:
        def __init__(self, **kwargs):
            calls["engine"] = kwargs

        def get_tokenizer(self):
            def template(messages, **kwargs):
                calls.setdefault("templates", []).append((messages, kwargs))
                return {"input_ids": [1, 2, 3]}

            return SimpleNamespace(apply_chat_template=template)

        def get_default_sampling_params(self):
            return Params(top_p=0.95, top_k=20, temperature=0.9)

        def generate(self, jobs, params, **kwargs):
            calls["jobs"], calls["params"], calls["generate_kwargs"] = jobs, params, kwargs
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="answer", token_ids=[9, 7], finish_reason="length")]
                )
                for _ in jobs
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, TokensPrompt=dict))
    monkeypatch.setattr(gencheck, "_context_tokenizer", lambda model, kwargs: fake_tokenizer())
    config.enable_thinking = False
    prompts = read_prompts(config)
    rows = gencheck._generate_in_process("checkpoint", prompts, config)
    assert calls["engine"]["generation_config"] == "original"
    assert calls["engine"]["tensor_parallel_size"] == 1
    assert all(template[1]["enable_thinking"] is False for template in calls["templates"])
    assert all(job == {"prompt_token_ids": [1, 2, 3]} for job in calls["jobs"])
    assert calls["generate_kwargs"] == {"use_tqdm": False}
    assert all(
        params.seed == 17
        and params.max_tokens == 300
        and params.temperature == 0.6
        and params.top_p == 0.95
        and params.top_k == 20
        and params.n == 1
        for params in calls["params"]
    )
    assert len({id(params) for params in calls["params"]}) == len(prompts)
    assert all(row.token_ids == [9, 7] and row.finish_reason == "length" for row in rows)


def test_vllm_runner_applies_checkpoint_serve_settings(config, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    serve = {"args": {"linear_backend": "marlin"}, "env": {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}}
    (checkpoint / "build.json").write_text(json.dumps({"serve": serve}))
    calls = {}

    def record(kwargs):
        calls["engines"] = calls.get("engines", 0) + 1
        calls["engine"] = kwargs
        calls["env"] = os.environ.get("VLLM_DISABLED_KERNELS")

    fake_engine(monkeypatch, record)
    config.checkpoint = str(checkpoint)
    gencheck._generate_in_process(str(checkpoint), read_prompts(config), config)
    assert calls["engine"]["linear_backend"] == "marlin"
    assert calls["env"] == "MacheteLinearKernel"
    assert "VLLM_DISABLED_KERNELS" not in os.environ
    gencheck._generate_in_process("original", read_prompts(config), config)
    assert "linear_backend" not in calls["engine"]
    assert calls["env"] == "MacheteLinearKernel"  # the original has no build.json; Machete is still off
    # A serve argument named like one of this runner's own defaults replaces it instead of raising a duplicate
    # keyword: build.json's policy allows only plan arguments today, so the settings are injected here.
    monkeypatch.setattr(
        gencheck,
        "settings_for",
        lambda model, args=None, nccl_nvls=None: ServeSettings(
            args={"gpu_memory_utilization": 0.9, **(args or {})}, nccl_nvls=nccl_nvls
        ),
    )
    calls.clear()
    tp2 = config.model_copy(update={"tp": 2})
    gencheck._generate_in_process(str(checkpoint), read_prompts(tp2), tp2)
    assert calls["engines"] == 1
    assert calls["engine"]["gpu_memory_utilization"] == 0.9
    assert calls["engine"]["tensor_parallel_size"] == 2  # generation supports tp>1, unlike scoring
    # serve_args reach the original too, so checkpoint and original serve on the same engine settings.
    calls.clear()
    sized = config.model_copy(update={"serve_args": {"max_model_len": 65536}})
    gencheck._generate_in_process("original", read_prompts(sized), sized)
    assert calls["engine"]["max_model_len"] == 65536


def test_vllm_runner_caps_the_context_at_the_longest_prompt_plus_max_tokens(config, monkeypatch):
    """With no max_model_len, an engine would start at the model's own context (262144 for Qwen3.8-27B), and its
    weights plus that cache do not fit one card (11-13: the 27B original OOMed at engine start). Both engines
    get the longest templated prompt plus max_tokens, from the tokenizer the engine will load."""
    seen, loaded = [], []
    fake_engine(monkeypatch, lambda kwargs: seen.append(kwargs.get("max_model_len")))
    prompts = read_prompts(config)
    lengths = [40 + index for index in range(len(prompts))]

    def tokenizer(model, kwargs):
        loaded.append((model, kwargs.get("tokenizer"), kwargs.get("trust_remote_code")))
        return fake_tokenizer(lengths)

    monkeypatch.setattr(gencheck, "_context_tokenizer", tokenizer)
    gencheck._generate_in_process(config.checkpoint, prompts, config)
    gencheck._generate_in_process(config.original, prompts, config)
    assert seen == [max(lengths) + config.max_tokens] * 2
    assert [model for model, _, _ in loaded] == [config.checkpoint, config.original]


def test_vllm_runner_keeps_a_configured_max_model_len_without_loading_a_tokenizer(
    config, tmp_path, monkeypatch
):
    """serve_args or the checkpoint's own build.json serve args that name max_model_len win; nothing is measured."""
    seen = []
    fake_engine(monkeypatch, lambda kwargs: seen.append(kwargs.get("max_model_len")))

    def refuse(model, kwargs):
        raise AssertionError("a configured max_model_len must not load a tokenizer")

    monkeypatch.setattr(gencheck, "_context_tokenizer", refuse)
    sized = config.model_copy(update={"serve_args": {"max_model_len": 65536}})
    gencheck._generate_in_process(sized.original, read_prompts(sized), sized)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "build.json").write_text(json.dumps({"serve": {"args": {"max_model_len": 4096}}}))
    monkeypatch.setattr(
        gencheck,
        "settings_for",
        lambda model, args=None, nccl_nvls=None: ServeSettings(
            args={**({"max_model_len": 4096} if str(model) == str(checkpoint) else {}), **(args or {})},
            nccl_nvls=nccl_nvls,
        ),
    )
    gencheck._generate_in_process(str(checkpoint), read_prompts(config), config)
    assert seen == [65536, 4096]


def test_vllm_runner_sizes_the_engine_for_its_prompts_unless_configured(config, monkeypatch):
    """The engine is sized for the check's own sequences (all its prompts in one batch), not vLLM's default 256+:
    the 27B original's memory profile at the default did not fit one card (11-13). A configured value wins."""
    seen = []
    fake_engine(monkeypatch, lambda kwargs: seen.append(kwargs.get("max_num_seqs")))
    prompts = read_prompts(config)
    gencheck._generate_in_process(config.original, prompts, config)
    sized = config.model_copy(update={"serve_args": {"max_num_seqs": 64}})
    gencheck._generate_in_process(sized.original, read_prompts(sized), sized)
    assert seen == [len(prompts), 64]


def test_context_tokenizer_loads_the_engine_tokenizer(monkeypatch):
    """The measuring tokenizer is the one vLLM would load: its ``tokenizer`` argument, else the model's."""
    got = []
    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda name, **kwargs: got.append((name, kwargs)) or name
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert gencheck._context_tokenizer("model", {}) == "model"
    assert gencheck._context_tokenizer("model", {"tokenizer": "tok", "trust_remote_code": True}) == "tok"
    assert got == [("model", {"trust_remote_code": False}), ("tok", {"trust_remote_code": True})]


@pytest.mark.parametrize(("nvls", "want"), [(False, "0"), (True, "1")])
def test_vllm_runner_sets_nvls_for_both_engines(config, monkeypatch, nvls, want):
    """``nccl_nvls`` reaches the checkpoint's engine and the original's, and is restored after each."""
    seen = []
    fake_engine(monkeypatch, lambda kwargs: seen.append(os.environ.get("NCCL_NVLS_ENABLE")))
    config.nccl_nvls = nvls
    gencheck._generate_in_process(config.checkpoint, read_prompts(config), config)
    gencheck._generate_in_process(config.original, read_prompts(config), config)
    assert seen == [want, want]
    assert "NCCL_NVLS_ENABLE" not in os.environ


@pytest.mark.parametrize("tp", [1, 2])
def test_vllm_runner_starts_both_engines_with_no_flashinfer_jit(config, monkeypatch, tp):
    """Both engines start with vLLM's own sampler, no FlashInfer all-reduce, no all-reduce + RMSNorm fusion and
    the Triton GDN prefill.

    Those are the paths that JIT-built FlashInfer's ``sampling`` and ``trtllm_mnnvl_comm`` with nvcc in every
    2026-11-13 judge gencheck, and FlashInfer's GDN prefill that a Qwen hybrid original JIT-builds at its first
    prefill (vLLM 0.33.0 warns "FlashInfer GDN prefill is JIT-compiled"); the variables are set only while an
    engine starts.
    """
    seen = []

    def record(kwargs):
        seen.append(({k: os.environ.get(k) for k in gencheck.NO_JIT_ENV}, kwargs["compilation_config"]))
        gdn.append(kwargs.get("gdn_prefill_backend"))

    gdn = []

    fake_engine(monkeypatch, record)
    for key in gencheck.NO_JIT_ENV:
        monkeypatch.delenv(key, raising=False)
    config.tp = tp
    gencheck._generate_in_process(config.checkpoint, read_prompts(config), config)
    gencheck._generate_in_process(config.original, read_prompts(config), config)
    want = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ALLREDUCE_USE_FLASHINFER": "0"}
    assert [env for env, _ in seen] == [want, want]
    assert all(c["pass_config"] == {"fuse_allreduce_rms": False} for _, c in seen)
    assert all(c["inductor_compile_config"] == {"deterministic": True} for _, c in seen)  # the pin stays
    assert gdn == ["triton", "triton"]
    assert not any(key in os.environ for key in gencheck.NO_JIT_ENV)


def test_no_jit_keeps_an_explicit_fusion_setting_and_compilation_off():
    """A ``serve_args`` naming the fusion keeps it; compilation level 0 compiles nothing and stays 0."""
    on = ServeSettings(args={"compilation_config": {"pass_config": {"fuse_allreduce_rms": True}, "mode": 3}})
    assert gencheck.no_jit(on).args["compilation_config"] == {
        "pass_config": {"fuse_allreduce_rms": True},
        "mode": 3,
    }
    assert gencheck.no_jit(ServeSettings(args={"compilation_config": 0})).args["compilation_config"] == 0
    kept = gencheck.no_jit(ServeSettings(env={"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}))
    assert kept.env == {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel", **gencheck.NO_JIT_ENV}


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ({}, {"gdn_prefill_backend": "triton"}),
        ({"gdn_prefill_backend": "flashinfer"}, {"gdn_prefill_backend": "flashinfer"}),
        (
            {"additional_config": {"gdn_prefill_backend": "flashinfer"}},
            {"additional_config": {"gdn_prefill_backend": "flashinfer"}},
        ),
        (
            {"additional_config": {"other": 1}},
            {"additional_config": {"other": 1}, "gdn_prefill_backend": "triton"},
        ),
    ],
)
def test_no_jit_sets_the_triton_gdn_prefill_unless_serve_args_name_one(args, want):
    """No FlashInfer GDN prefill JIT build: ``triton`` unless ``serve_args`` names a backend (either spelling)."""
    got = gencheck.no_jit(ServeSettings(args=args)).args
    assert {
        k: v for k, v in got.items() if k not in ("compilation_config", "enable_flashinfer_autotune")
    } == want


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ({}, {"enable_flashinfer_autotune": False}),
        ({"enable_flashinfer_autotune": True}, {"enable_flashinfer_autotune": True}),
        (
            {"kernel_config": {"enable_flashinfer_autotune": True}},
            {"kernel_config": {"enable_flashinfer_autotune": True}},
        ),
    ],
)
def test_no_jit_turns_flashinfer_autotune_off_unless_serve_args_name_it(args, want):
    """Autotune off unless ``serve_args`` names it in either spelling: vLLM refuses both at once."""
    got = gencheck.no_jit(ServeSettings(args=args)).args
    assert {k: v for k, v in got.items() if k in ("enable_flashinfer_autotune", "kernel_config")} == want


def test_original_cache_key_includes_the_no_jit_settings(cache_config, monkeypatch):
    """Answers generated with FlashInfer's sampler are not reused for a check that runs without it."""
    prompts = read_prompts(cache_config)
    with_no_jit = gencheck.original_cache_key(cache_config, prompts)
    monkeypatch.setattr(gencheck, "no_jit", lambda settings: settings)
    assert gencheck.original_cache_key(cache_config, prompts) != with_no_jit


def test_report_records_the_checkpoints_serve_settings(config, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    serve = {"args": {"linear_backend": "marlin"}, "kernels": {"dense": ["MarlinLinearKernel"]}}
    (checkpoint / "build.json").write_text(json.dumps({"serve": serve}))
    config.checkpoint = str(checkpoint)
    monkeypatch.setattr(gencheck, "_generate", lambda model, prompts, cfg: [answer(p) for p in prompts])
    report = gencheck.run_gencheck(config)
    # the report records what both engines ran with: build.json's arguments, the deterministic pin and no_jit's
    assert report["serve"]["args"] == {
        "linear_backend": "marlin",
        "compilation_config": {
            "inductor_compile_config": {"deterministic": True},
            "pass_config": {"fuse_allreduce_rms": False},
        },
        "gdn_prefill_backend": "triton",
        "enable_flashinfer_autotune": False,
    }
    assert {k: report["serve"]["env"][k] for k in gencheck.NO_JIT_ENV} == gencheck.NO_JIT_ENV
    assert report["serve"]["kernels"] == {"dense": ["MarlinLinearKernel"]}
    assert report["serve"]["source"] == "build.json"


def test_child_process_failure_is_runtime_error(config, monkeypatch):
    monkeypatch.setattr(gencheck.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="engine process exited 1"):
        gencheck._generate("checkpoint", read_prompts(config), config)


@pytest.mark.gpu
def test_real_vllm_tiny_random_model(tmp_path, monkeypatch):
    import vllm
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    model_path = tmp_path / "model"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model.save_pretrained(model_path)
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "Explain": 3}
    vocabulary.update({f"token{index}": index for index in range(4, 128)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(model_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "tiny", "domain": "instruction", "prompt": "Explain"}) + "\n")
    config = GencheckConfig(
        checkpoint=str(model_path),
        original=str(model_path),
        prompts=str(prompts),
        domains=["instruction"],
        per_domain=1,
        max_tokens=16,
        temperature=0.0,
        out=str(tmp_path / "out"),
    )
    engine = vllm.LLM

    def tiny_engine(**kwargs):
        return engine(**kwargs, kv_cache_memory_bytes=16 * 1024**2, enforce_eager=True)

    monkeypatch.setattr(vllm, "LLM", tiny_engine)
    rows = gencheck._generate_in_process(str(model_path), read_prompts(config), config)
    assert len(rows) == 1 and rows[0].id == "tiny" and rows[0].domain == "instruction"
    assert 0 < len(rows[0].token_ids) <= 16
    assert rows[0].finish_reason in ("stop", "length")
