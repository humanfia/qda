"""The engine-tokenizer check: the R1-Distill case is 8 ids where the checkpoint's tokenizer.json gives 11."""

from functools import lru_cache
from pathlib import Path

import pytest
from qda_kl.tokenizer_check import PROBE, tokenizer_refusal
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

# The checkpoint's tokenizer.json: ByteLevel BPE, which splits the probe into 11 ids.
FULL = ["Hello", "Ġis", "Ġa", "ĊNext", "Ġline."]
PARTIAL = ["Ġworld,", "Ġthis", "Ġtest."]  # one merge short, so each keeps two tokens


@lru_cache(maxsize=1)
def _correct_tokenizer() -> Tokenizer:
    """A ByteLevel BPE like the one the checkpoint ships: 11 ids for the probe."""
    vocab = {char: i for i, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    merges = []
    for piece in FULL + PARTIAL:
        parts = list(piece)[: len(piece) if piece in FULL else len(piece) - 1]
        merged = parts[0]
        for nxt in parts[1:]:
            if merged + nxt not in vocab:
                vocab[merged + nxt] = len(vocab)
            merges.append((merged, nxt))
            merged += nxt
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def _broken_tokenizer() -> Tokenizer:
    """The class transformers 5 builds for those repos: one id per whitespace word, 8 for the probe."""
    words = ["Hello", "world,", "this", "is", "a", "test.", "Next", "line."]
    vocab = {word: i for i, word in enumerate(words)} | {"<unk>": len(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


@pytest.fixture
def checkpoint(tmp_path):
    """A snapshot whose tokenizer.json is the correct one, and the two tokenizers to compare."""
    correct = _correct_tokenizer()
    correct.save(str(tmp_path / "tokenizer.json"))
    return tmp_path, correct, _broken_tokenizer()


def test_the_case_is_eight_ids_against_eleven(checkpoint):
    """The fixtures really are the issue's case: 8 ids from the engine, 11 from the checkpoint."""
    _, correct, broken = checkpoint
    assert len(broken.encode(PROBE, add_special_tokens=False).ids) == 8
    assert len(correct.encode(PROBE, add_special_tokens=False).ids) == 11


def test_a_different_tokenization_is_refused_with_both_id_lists(checkpoint):
    _, _, broken = checkpoint
    message = tokenizer_refusal(broken, checkpoint[0])
    assert message is not None
    assert "as 8 ids" in message and "gives 11" in message
    assert "LlamaTokenizerFast" in message and "#" not in message  # stands on its own, no issue number
    assert "--serve-args" in message and '"tokenizer"' in message  # the remedy: a flag `qda-kl ref` has


def test_a_tokenizer_that_agrees_passes(checkpoint):
    assert tokenizer_refusal(checkpoint[1], checkpoint[0]) is None


def test_a_snapshot_without_tokenizer_json_is_skipped(tmp_path):
    """A repository id or a tokenizer in another format has nothing to compare against."""
    assert tokenizer_refusal(_broken_tokenizer(), tmp_path) is None


def test_a_transformers_tokenizer_that_returns_ids_is_read(checkpoint):
    """vLLM's tokenizer returns a list; a `tokenizers` one returns an Encoding. Both are read."""

    class ListEncoding:
        def encode(self, text, add_special_tokens=False):  # noqa: ARG002 - the HF signature
            return list(range(11))

    message = tokenizer_refusal(ListEncoding(), checkpoint[0])
    assert message is not None and "as 11 ids" in message  # the list was read, not an Encoding
    assert Path(checkpoint[0], "tokenizer.json").is_file()


REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SHA = "6a6f4aa4d1a1e7c3b8a6f1d2c3b4a5968778695a"


@pytest.fixture
def hub_cache(monkeypatch, tmp_path):
    """A local Hugging Face cache with the checkpoint's tokenizer.json at the pinned revision."""
    hub = tmp_path / "hub"
    repo = hub / f"models--{REPO.replace('/', '--')}"
    snapshot = repo / "snapshots" / SHA
    snapshot.mkdir(parents=True)
    _correct_tokenizer().save(str(snapshot / "tokenizer.json"))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    return tmp_path


def test_a_hub_id_reads_the_cached_snapshot(hub_cache):
    """The reference is the cached tokenizer.json at the pinned revision, not a skipped check."""
    message = tokenizer_refusal(_broken_tokenizer(), REPO)
    assert message is not None and "as 8 ids" in message and "gives 11" in message
    assert tokenizer_refusal(_correct_tokenizer(), REPO) is None


def test_a_hub_id_reads_the_named_revision(hub_cache, tmp_path):
    """A revision that has the file among its snapshots is read even when the main ref does not."""
    repo = tmp_path / "hub" / f"models--{REPO.replace('/', '--')}"
    other = repo / "snapshots" / ("0" * 40)
    other.mkdir()
    _correct_tokenizer().save(str(other / "tokenizer.json"))
    assert tokenizer_refusal(_correct_tokenizer(), REPO, revision="0" * 40) is None


def test_a_hub_id_whose_tokenizer_json_is_not_cached_is_refused(hub_cache):
    """What cannot be compared is not scored: a missing cache entry is a refusal, not a silent skip."""
    message = tokenizer_refusal(_correct_tokenizer(), "org/not-downloaded")
    assert message is not None and "not in the local Hugging Face cache" in message
    assert "Download the checkpoint" in message
