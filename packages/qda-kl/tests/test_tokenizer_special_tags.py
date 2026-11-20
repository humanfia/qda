"""Vendor tokenizers can tag fewer special tokens while retaining identical teacher-forced ids and roles."""

from qda_kl.tokenizer import check_tokenizer
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def test_special_tags_do_not_refuse_identical_token_ids_and_stopping_roles(monkeypatch):
    """RedHat INT4 keeps the anchor's map and EOS/PAD roles but tags fewer tokens as special."""
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "a": 3, "[SECTION]": 4}

    def tiny(additional):
        return PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            additional_special_tokens=additional,
        )

    anchor, candidate = tiny(["[SECTION]"]), tiny([])
    assert anchor.get_vocab() == candidate.get_vocab()
    assert anchor.eos_token_id == candidate.eos_token_id
    assert anchor.pad_token_id == candidate.pad_token_id
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: anchor)
    assert check_tokenizer({"name": "original", "revision": "pinned"}, candidate)
