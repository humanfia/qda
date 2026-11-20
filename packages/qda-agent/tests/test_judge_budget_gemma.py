"""Gemma's per-layer embedding is outside the decoder-Linear bpw budget."""

import pytest
from qda_agent.judge.checks.budget import bpw, denominator
from test_judge_budget import K, Q, model_dir, quantized


def test_per_layer_embedding_is_not_charged_to_the_linear_budget(tmp_path):
    # Gemma4TextModel stores a second nn.Embedding under this name, in addition to embed_tokens.
    embedding = {"model.language_model.embed_tokens_per_layer.weight": ("BF16", (256, 128), None)}
    original = model_dir(
        tmp_path / "anchor",
        {**embedding, f"{Q}.weight": ("BF16", (128, 128), None), f"{K}.weight": ("BF16", (128, 128), None)},
    )
    candidate = model_dir(
        tmp_path / "candidate", {**embedding, **quantized(Q, 4), **quantized(K, 4)}, quantized=True
    )
    assert denominator(original) == 2 * 128 * 128
    assert bpw(candidate, original) == pytest.approx(4 + 16 / 128)
