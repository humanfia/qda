"""Activation dtype changes require explicit permission before a KL result is published."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from qda_kl import cli, score
from qda_kl.store import ReferenceWriter
from qda_kl.topk import anchor_topk
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize("requested", ["float16", "bfloat16"])
@pytest.mark.parametrize("allow", [False, True])
def test_requested_and_loaded_dtype_guard(tmp_path, monkeypatch, requested, allow):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"a": 0, "b": 1, "[UNK]": 2, "[EOS]": 3}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tmp_path)
    ref = tmp_path / "ref"
    writer = ReferenceWriter(ref, 2, 4, model={"dtype": "bfloat16", "path": str(tmp_path)})
    writer.add("a", "math", [0, 1], 1, anchor_topk(torch.zeros((1, 4)).log_softmax(-1), 2))
    writer.close()
    before = (ref / "manifest.json").read_bytes()
    model = SimpleNamespace(dtype="float16", get_vocab_size=lambda: 4)
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=model),
        collective_rpc=lambda _: [{}],
        get_tokenizer=lambda: tokenizer,
    )
    opened = []
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    monkeypatch.setattr(
        score, "teacher_force", lambda *a, **k: {"a": dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))}
    )
    out, config = tmp_path / "out", tmp_path / "score.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "checkpoint": str(tmp_path),
                "reference": str(ref),
                "out": str(out),
                "scorer": {"topk": {"serve_args": {"dtype": requested}, "allow_dtype_mismatch": allow}},
            }
        )
    )
    assert cli.main(["score", str(config)]) == (0 if allow else 2)
    if allow:
        result = json.loads((out / "score.json").read_text())
        assert result["dtype"] == "float16" and result["allow_dtype_mismatch"] is True
        resolved = yaml.safe_load((out / "resolved.yaml").read_text())
        assert resolved["scorer"]["topk"]["allow_dtype_mismatch"] is True
    else:
        assert not (out / "score.json").exists()
        assert len(opened) == (0 if requested == "float16" else 1)
    assert (ref / "manifest.json").read_bytes() == before
