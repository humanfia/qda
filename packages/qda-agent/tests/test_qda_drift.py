"""The judge's file check against what a real `qda build` writes, so `checks.INERT` cannot drift from qda (CPU).

A tiny random Qwen3 (built from a transformers config) is the original model; `qda build CONFIG.yaml` quantizes it
on CPU; the hand-in check must accept the output as it is, with every file qda writes next to the weights.
"""

from __future__ import annotations

import json

import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from qda_agent.judge.checks import files_problem, inventory  # noqa: E402


def tiny_model(d) -> str:
    """Save a tiny Qwen3 with a 128-token tokenizer into `d`; return the tokenizer's fingerprint."""
    from qda_quant.engine.calib import fingerprint
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=128,  # the format quantizes in groups of 128 columns
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
    )
    transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(d)
    t = Tokenizer(models.WordLevel({f"w{i}": i for i in range(127)} | {"[UNK]": 127}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    tok.chat_template = "{% for m in messages %}w1 {{ m['content'] }} w2 {% endfor %}"
    tok.save_pretrained(d)
    return fingerprint(d)


def test_a_real_qda_build_passes_the_hand_in_file_check(tmp_path):
    from qda_quant.cli import main

    model = tmp_path / "m"
    sha = tiny_model(model)
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(6)]
    calib = tmp_path / "c.jsonl"
    calib.write_text(
        "\n".join([json.dumps({"tokenizer": sha}), *(json.dumps({"input_ids": r}) for r in rows)])
    )
    out = tmp_path / "out"
    config = {
        "model": str(model),
        "format": "w4a16-sym-g128",
        "devices": ["cpu"],
        "steps": [{"gptq": {}}],
        "calib": {"file": {"path": str(calib), "samples": 6, "seqlen": 32}},
        "out": str(out),
    }
    (tmp_path / "build.yaml").write_text(yaml.safe_dump(config))
    assert main(["build", str(tmp_path / "build.yaml")]) == 0

    assert (out / "resolved.yaml").is_file()  # what qda writes now, so the check below covers it
    assert inventory(model)["lm_head.weight"][0] == inventory(out)["lm_head.weight"][0] == "F32"
    assert files_problem(out, anchor=model) is None, sorted(p.name for p in out.iterdir())
