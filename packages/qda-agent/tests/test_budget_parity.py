"""The bpw denominator's two copies: `qda_quant.intake.fisher.decoder_weights` (the model side) and
`qda_agent.judge.checks.budget.denominator` (the checkpoint side). qda_quant may not import qda_agent, so
this file owns their equality."""

import torch
from qda_agent.judge.checks.budget import denominator
from qda_quant.arch.derive import derive
from qda_quant.intake.fisher import decoder_weights
from transformers import AutoModelForCausalLM, Qwen3Config


def test_the_model_side_and_checkpoint_side_denominators_agree(tmp_path):
    """The head is in neither sum, tied or not (the judge prices an untied one by its lease), so both sides
    agree and the two checkpoints count the same weights."""
    counts = {}
    for tied in (False, True):
        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            tie_word_embeddings=tied,
        )
        cfg._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
        model.tie_weights()
        d = tmp_path / ("tied" if tied else "untied")
        model.save_pretrained(d)
        graph = derive(model)
        counts[tied] = decoder_weights(model, graph)
        assert counts[tied] == denominator(d)
    assert counts[False] == counts[True]
