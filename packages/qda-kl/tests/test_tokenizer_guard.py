"""Token meanings, roles and effective stops are checked before teacher-forcing a frozen reference."""

from types import SimpleNamespace

import pytest
from qda_kl import score
from qda_kl.serve import ServeSettings
from qda_kl.store import Reference
from qda_kl.tokenizer import tokenizer_sha256
from test_reference_merge import shard
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def tiny(swapped=False):
    vocab = {"[UNK]": 0, "[EOS]": 1, "a": 3 if swapped else 2, "b": 2 if swapped else 3}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")


@pytest.mark.parametrize(
    "change,part", [("same", ""), ("swapped", "tokens"), ("roles", "roles"), ("stop", "stop_ids")]
)
def test_token_identity_checked_before_scoring(tmp_path, monkeypatch, change, part):
    original, candidate = tiny(), tiny(change == "swapped")
    if change == "roles":
        candidate.eos_token = "a"
    stops = [1, 2] if change == "stop" else [1]
    ref = shard(tmp_path / "ref", "a")
    before = (ref / "manifest.json").read_bytes(), Reference(ref).content_sha256
    model = SimpleNamespace(dtype="bfloat16", get_vocab_size=lambda: 4)
    processor = SimpleNamespace(generation_config_fields={"eos_token_id": stops})
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=model, input_processor=processor),
        collective_rpc=lambda _: [{}],
        get_tokenizer=lambda: candidate,
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: original)
    monkeypatch.setattr(score, "open_engine", lambda *a, **kw: engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")

    checkpoint = tmp_path / "candidate"  # a directory: the engine tokenizer guard skips an unaffected class
    checkpoint.mkdir()

    def make():
        return score.Scorer(str(checkpoint), ref, settings=ServeSettings(args={"dtype": "bfloat16"}))

    if change == "same":
        assert make().tokenizer_sha256 == tokenizer_sha256(original)
    else:
        with pytest.raises(ValueError, match=f"mismatch in .*{part}") as error:
            make()
        assert tokenizer_sha256(original) in str(error.value)
        assert tokenizer_sha256(candidate, stop_ids=stops) in str(error.value)
    assert before == ((ref / "manifest.json").read_bytes(), Reference(ref, verify=True).content_sha256)


@pytest.mark.parametrize("candidate_eos,refused", [(2, False), (3, True)])
def test_a_generation_config_without_eos_takes_the_config_stops(
    tmp_path, monkeypatch, candidate_eos, refused
):
    """A published checkpoint against its own original: same tokenizer and config EOS, extra sampling file.

    The original ships no generation_config.json, so its stops are config.json's EOS plus the tokenizer's. The
    checkpoint ships one that names only sampling settings, so the engine reports no EOS for it. Its stops
    come from its config.json as the original's do; a config.json naming another EOS is still refused.
    """
    import json

    original, candidate = tiny(), tiny()
    source = tmp_path / "original"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": 2}))
    original.name_or_path = str(source)
    ref = shard(tmp_path / "ref", "a")
    model = SimpleNamespace(dtype="bfloat16", get_vocab_size=lambda: 4)
    processor = SimpleNamespace(generation_config_fields={"temperature": 1.0})  # no eos_token_id
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=model, input_processor=processor),
        collective_rpc=lambda _: [{}],
        get_tokenizer=lambda: candidate,
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: original)
    monkeypatch.setattr(score, "open_engine", lambda *a, **kw: engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    checkpoint = tmp_path / "candidate"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "llama", "eos_token_id": candidate_eos})
    )
    (checkpoint / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 0.95}))

    def make():
        return score.Scorer(str(checkpoint), ref, settings=ServeSettings(args={"dtype": "bfloat16"}))

    if refused:
        with pytest.raises(ValueError, match="mismatch in stop_ids:"):
            make()
    else:
        assert make().tokenizer_sha256 == tokenizer_sha256(original)  # stops [1, 2] on both sides
