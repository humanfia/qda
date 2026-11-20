"""Without generation_config.json the original stops where the engine stops: config.json's EOS, plus its own."""

import json

import pytest
from qda_kl.tokenizer import check_tokenizer
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5TextConfig

CONFIG_EOS = 99


def tiny(path, config=True):
    """A tiny local model dir whose config.json EOS differs from its tokenizer EOS, or has no model config."""
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "a": 3}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)
    if config:
        (path / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": CONFIG_EOS}))
    return tokenizer


def test_missing_generation_config_adds_the_config_eos(tmp_path):
    """The original identity is config.json's EOS plus the tokenizer EOS, so a candidate reporting both passes."""
    tiny(tmp_path / "original")
    candidate = tiny(tmp_path / "candidate")
    assert check_tokenizer({"path": str(tmp_path / "original")}, candidate, stop_ids=[CONFIG_EOS, 2])


def test_no_model_config_keeps_the_tokenizer_eos(tmp_path):
    """A directory with a tokenizer but no model config falls back to the tokenizer EOS, as it did before."""
    tiny(tmp_path / "original", config=False)
    candidate = tiny(tmp_path / "candidate", config=False)
    assert check_tokenizer({"path": str(tmp_path / "original")}, candidate, stop_ids=[2])


def test_a_qwen35_text_config_eos_is_a_stop(tmp_path):
    """A Qwen3.5 checkpoint keeps its EOS in text_config, and without generation_config.json it is a stop."""
    tiny(tmp_path / "original", config=False)
    Qwen3_5Config(text_config=Qwen3_5TextConfig(eos_token_id=CONFIG_EOS)).save_pretrained(
        tmp_path / "original"
    )
    candidate = tiny(tmp_path / "candidate", config=False)
    assert check_tokenizer({"path": str(tmp_path / "original")}, candidate, stop_ids=[CONFIG_EOS, 2])


@pytest.mark.parametrize("stops", [[2], [CONFIG_EOS, 2, 3]])
def test_missing_generation_config_still_refuses_other_stops(tmp_path, stops):
    """The tokenizer EOS alone, or any extra stop, is a mismatch and refused."""
    tiny(tmp_path / "original")
    candidate = tiny(tmp_path / "candidate")
    with pytest.raises(ValueError, match="mismatch in stop_ids"):
        check_tokenizer({"path": str(tmp_path / "original")}, candidate, stop_ids=stops)
