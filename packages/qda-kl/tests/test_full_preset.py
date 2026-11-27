"""The `full` preset's one table: every released family's settings, the match, the defaults and the overrides."""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml
from qda_kl.eval.full_preset import FAMILIES, applied
from qda_quant.step import Refused

QWEN35 = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
QWEN38 = {**QWEN35, "presence_penalty": 0.0}
R1 = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
GEMMA = {**R1, "temperature": 1.0, "top_k": 64}
NEMOTRON = {**R1, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
CODER = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.05,
}
METRIC = {
    "ifbench": "prompt_level_loose_acc",
    "ifeval": "prompt_level_strict_acc",
    "mmlu_pro": "exact_match",
}


def _caps(high: int, gpqa: str = "paper_gpqa_diamond") -> dict[str, int]:
    """The six `lm_eval` caps: every task at ``high`` except MMLU-Pro's 32768 (all cards)."""
    return {
        "paper_aime25": high,
        "paper_hmmt_feb_2026": high,
        gpqa: high,
        "ifbench": high,
        "ifeval": high,
        "mmlu_pro": 32768,
    }


_CAPS_81920 = _caps(81920)
_CAPS_27B = {**_CAPS_81920, "paper_gpqa_diamond": 32768, "ifbench": 32768, "ifeval": 32768}
_CAPS_R1 = _caps(32768, "paper_gpqa_diamond_simple_evals")

#: The expected expansion of every table row, restated from `EVAL-SETTINGS.md` §0 (the released cards).
EXPECTED: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3.5-4B": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": QWEN35,
        "think": "</think>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 1,
        "tokenizer": False,
    },
    "Qwen/Qwen3.5-9B": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": QWEN35,
        "think": "</think>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 1,
        "tokenizer": False,
    },
    "Qwen/Qwen3.6-35B-A3B": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": QWEN35,
        "think": "</think>",
        "ctk": {},
        "thinking": True,
        "util": 0.9,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 2,
        "tokenizer": False,
    },
    "Qwen/Qwen3.8-27B": {
        "caps": _CAPS_27B,
        "lcb": 81920,
        "sampling": QWEN38,
        "think": "</think>",
        "ctk": {"reasoning_effort": "medium"},
        "thinking": True,
        "util": 0.9,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 1,
        "tokenizer": False,
    },
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": CODER,
        "think": None,
        "ctk": {},
        "thinking": False,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 2,
        "tokenizer": False,
    },
    "google/gemma-4-26B-A4B-it": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": GEMMA,
        "think": "<channel|>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 1,
        "tokenizer": False,
    },
    "google/gemma-4-31B-it": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": GEMMA,
        "think": "<channel|>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 2,
        "tokenizer": False,
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {
        "caps": _CAPS_81920,
        "lcb": 81920,
        "sampling": NEMOTRON,
        "think": "</think>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 114688,
        "gpqa": "paper_gpqa_diamond",
        "lcb_tp": 1,
        "tokenizer": False,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "caps": _CAPS_R1,
        "lcb": 32768,
        "sampling": R1,
        "think": "</think>",
        "ctk": {},
        "thinking": True,
        "util": 0.95,
        "window": 49152,
        "gpqa": "paper_gpqa_diamond_simple_evals",
        "lcb_tp": 1,
        "tokenizer": True,
    },
}


def test_the_table_holds_exactly_the_nine_released_ids():
    assert list(FAMILIES) == list(EXPECTED)


@pytest.mark.parametrize("model", list(EXPECTED))
def test_a_family_expands_to_its_released_card_settings(model: str) -> None:
    """Every table row expands to the caps, sampling, thinking, window, repeats and seed its cards recorded."""
    want = EXPECTED[model]
    got = applied(model)
    lm = got.entries[0]["lm_eval"]
    lcb = got.entries[1]["livecodebench"]
    assert got.note == f"full: settings of {model} (the released cards')"
    assert lm["tasks"] == [
        "paper_aime25",
        "paper_hmmt_feb_2026",
        want["gpqa"],
        "ifbench",
        "ifeval",
        "mmlu_pro",
    ]
    assert lm["max_tokens"] == want["caps"]
    assert lm["repeats"] == {
        "paper_aime25": 32,
        "paper_hmmt_feb_2026": 32,
        want["gpqa"]: 16,
        "ifbench": 16,
        "ifeval": 16,
        "mmlu_pro": 1,
    }
    assert lm["sampling"] == want["sampling"]
    assert lm["think_end_token"] == want["think"]
    assert lm["chat_template_kwargs"] == want["ctk"]
    assert lm["enable_thinking"] is want["thinking"]
    assert lm["metric"] == METRIC
    assert lm["seed"] == 17
    serve = {
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "enable_prefix_caching": False,
        "max_num_seqs": 128,
        "gpu_memory_utilization": want["util"],
        "max_model_len": want["window"],
    }
    if want["tokenizer"]:
        serve["tokenizer"] = model
    assert lm["serve_args"] == serve
    assert lcb["protocol"] == "paper" and lcb["timeout_s"] == 8
    assert lcb["repeats"] == 4 and lcb["max_tokens"] == want["lcb"]
    assert lcb["sampling"] == want["sampling"]
    assert lcb["think_end_token"] == want["think"]
    lcb_serve = dict(serve)
    if want["lcb_tp"] > 1:
        lcb_serve["tensor_parallel_size"] = want["lcb_tp"]
    assert lcb["serve_args"] == lcb_serve


def test_mmlu_pro_uses_the_models_sampling_not_greedy() -> None:
    """The table's MMLU-Pro is sampled like every other task; the old greedy group is gone."""
    lm = applied("Qwen/Qwen3.8-27B").entries[0]["lm_eval"]
    assert lm["sampling"]["temperature"] == 1.0
    assert lm["sampling"]["presence_penalty"] == 0.0
    assert lm["max_tokens"]["mmlu_pro"] == 32768
    assert lm["repeats"]["mmlu_pro"] == 1


def test_the_row_is_found_by_the_reference_or_a_checkpoints_base_model(tmp_path) -> None:
    """The original's HF id comes from the reference (an id, or a Hub-cache path), else a checkpoint's files."""
    assert applied("Qwen/Qwen3.5-9B@abc123").note == "full: settings of Qwen/Qwen3.5-9B (the released cards')"
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "851bf6e"
    snapshot.mkdir(parents=True)
    assert applied(str(snapshot)).note == "full: settings of Qwen/Qwen3.5-4B (the released cards')"
    reference = tmp_path / "reference"
    reference.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "README.md").write_text("---\nbase_model: Qwen/Qwen3.8-27B\n---\n\n# ckpt\n")
    assert applied(str(reference), [str(checkpoint)]).note == (
        "full: settings of Qwen/Qwen3.8-27B (the released cards')"
    )
    (checkpoint / "resolved.yaml").write_text(yaml.safe_dump({"model": "google/gemma-4-31B-it"}))
    assert applied(str(reference), [str(checkpoint)]).note == (
        "full: settings of google/gemma-4-31B-it (the released cards')"
    )


def test_an_unknown_model_gets_default_caps_and_its_generation_config_sampling(tmp_path) -> None:
    """No row: 32768-token caps (capped by the context) and the sampling the model's own file names."""
    model = tmp_path / "unknown"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 16384}))
    (model / "generation_config.json").write_text(
        json.dumps({"temperature": 0.7, "top_p": 0.9, "do_sample": True})
    )
    got = applied(str(model))
    lm = got.entries[0]["lm_eval"]
    assert got.note == (
        f"full: no released card for {model}; 32768-token caps, its generation_config.json's sampling, and "
        "whole answers scored (no closing thinking tag in its chat template)"
    )
    assert set(lm["max_tokens"].values()) == {16384}
    assert lm["sampling"] == {"temperature": 0.7, "top_p": 0.9}
    assert lm["serve_args"]["max_model_len"] == 16384
    assert got.entries[1]["livecodebench"]["max_tokens"] == 16384


#: A thinking template's generation prompt: the block opens in the prompt, so an answer holds only its close.
THINKING_TEMPLATE = (
    "{%- if enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}"
)
#: A thinking family's non-thinking model: the tag is in its vocabulary, never in its template.
TAG_AS_TOKEN = {"added_tokens_decoder": {"151668": {"content": "</think>"}}, "chat_template": "{{ m }}"}
GEMMA_TEMPLATE = {"chat_template": "<|channel>thought{{ x }}<channel|>"}


@pytest.mark.parametrize(
    ("name", "text", "tag"),
    [
        ("chat_template.jinja", THINKING_TEMPLATE, "</think>"),
        ("tokenizer_config.json", json.dumps(GEMMA_TEMPLATE), "<channel|>"),
        ("tokenizer_config.json", json.dumps(TAG_AS_TOKEN), None),
    ],
)
def test_a_model_without_a_card_is_cut_at_its_chat_templates_thinking_tag(tmp_path, name, text, tag) -> None:
    """A local fine-tune whose config names no original has no row; its template says where answers start."""
    model = tmp_path / "fine-tune"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
    (model / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    (model / name).write_text(text)
    got = applied(str(model))
    assert got.entries[0]["lm_eval"]["think_end_token"] == tag
    assert got.entries[1]["livecodebench"]["think_end_token"] == tag
    assert got.note.endswith(
        f"and only the text after {tag!r} scored"
        if tag
        else "and whole answers scored (no closing thinking tag in its chat template)"
    )
    override = applied(str(model), [], {"think_end_token": "</x>"})
    assert override.entries[0]["lm_eval"]["think_end_token"] == "</x>"
    assert override.note.endswith("and only the text after '</x>' scored (your overrides applied)")


def test_an_unreadable_generation_config_is_refused_in_two_lines(tmp_path) -> None:
    """The reason, then the fix: never a silent vLLM default."""
    model = tmp_path / "unknown"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    with pytest.raises(Refused) as refused:
        applied(str(model))
    reason, fix = str(refused.value).splitlines()
    assert reason == f"full: {model}: its generation_config.json is missing or names no sampling"
    assert fix.startswith("fix: ")


def test_the_users_overrides_win_field_by_field() -> None:
    """`--set suites.0.full.…` (or an entry's settings) wins over the row; every other value stays the card's."""
    got = applied(
        "Qwen/Qwen3.8-27B",
        [],
        {
            "max_tokens": 4096,
            "repeats": {"mmlu_pro": 2, "livecodebench": 1},
            "sampling": {"temperature": 0.7},
            "think_end_token": "</x>",
            "chat_template_kwargs": {"reasoning_effort": "high"},
            "enable_thinking": False,
            "metric": {"ifbench": "prompt_level_strict_acc"},
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.8,
            "serve_args": {"max_num_seqs": 8},
            "seed": 5,
        },
    )
    assert got.note == "full: settings of Qwen/Qwen3.8-27B (the released cards') (your overrides applied)"
    lm = got.entries[0]["lm_eval"]
    lcb = got.entries[1]["livecodebench"]
    assert set(lm["max_tokens"].values()) == {4096} and lcb["max_tokens"] == 4096
    assert lm["repeats"]["mmlu_pro"] == 2 and lm["repeats"]["paper_aime25"] == 32
    assert lcb["repeats"] == 1
    assert lm["sampling"] == {**QWEN38, "temperature": 0.7}
    assert lm["think_end_token"] == "</x>" and lm["enable_thinking"] is False
    assert lm["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert lm["metric"] == {**METRIC, "ifbench": "prompt_level_strict_acc"}
    assert lm["serve_args"]["max_model_len"] == 32768
    assert lm["serve_args"]["gpu_memory_utilization"] == 0.8
    assert lm["serve_args"]["max_num_seqs"] == 8
    assert lm["seed"] == 5


def test_a_partial_per_task_override_keeps_the_other_caps() -> None:
    got = applied("Qwen/Qwen3.8-27B", [], {"max_tokens": {"livecodebench": 2048}, "repeats": {}})
    assert got.entries[1]["livecodebench"]["max_tokens"] == 2048
    assert got.entries[0]["lm_eval"]["max_tokens"]["paper_aime25"] == 81920


def test_an_unknown_override_is_refused() -> None:
    with pytest.raises(Refused, match="temprature"):
        applied("Qwen/Qwen3.8-27B", [], {"temprature": 0.3})
    with pytest.raises(Refused, match="unknown task"):
        applied("Qwen/Qwen3.8-27B", [], {"max_tokens": {"aime": 1}})
    with pytest.raises(Refused, match="sampling"):
        applied("Qwen/Qwen3.8-27B", [], {"sampling": {"seed": 1}})
