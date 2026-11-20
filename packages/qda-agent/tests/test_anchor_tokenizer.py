"""The anchor check accepts the tokenizer-class correction ``qda build`` records, and no other tokenizer change.

The anchor is a fine-tune's original: a ByteLevel ``tokenizer.json`` under a ``tokenizer_config.json`` declaring
``LlamaTokenizerFast``, which transformers rebuilds as another pipeline. Candidate 0 is a build of it, so its
config declares ``PreTrainedTokenizerFast`` (``build.json``: ``declared -> written``). Before, every baseline
built from the anchor, and the original beside a corrected anchor, was refused as "tokenizer_config.json differs".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from qda_agent.judge.checks import anchor_problem
from qda_quant.intake.tokenizer import tokenizer_config_record
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

TEXT = ["Hello world, this is a test.\nNext line.", "  two  spaces\n\ttab What is 2 + 2?"]


def _model(root: Path, tokenizer_class: str = "LlamaTokenizerFast", model_type: str = "qwen2") -> Path:
    """A model directory's config and tokenizer files (the anchor check reads nothing else)."""
    root.mkdir(parents=True)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(TEXT, trainers.BpeTrainer(vocab_size=300))
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>").save_pretrained(root)
    conf = json.loads((root / "tokenizer_config.json").read_text())
    conf.update(tokenizer_class=tokenizer_class, chat_template="{{ messages }}")
    (root / "tokenizer_config.json").write_text(json.dumps(conf))
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    return root


def _built(src: Path, out: Path) -> Path:
    """`out` as ``qda build`` leaves `src`'s files: the declared class corrected and recorded, as the export does."""
    shutil.copytree(src, out)
    record = tokenizer_config_record(src)
    if record["written"] != "unchanged":
        declared = json.loads((src / "tokenizer_config.json").read_text())
        declared["tokenizer_class"] = record["written"]
        (out / "tokenizer_config.json").write_text(json.dumps(declared, indent=2) + "\n")
    (out / "build.json").write_text(json.dumps({"tokenizer_config": record}))
    return out


def _edit(d: Path, **fields) -> None:
    conf = json.loads((d / "tokenizer_config.json").read_text())
    conf.update(fields)
    (d / "tokenizer_config.json").write_text(json.dumps(conf, indent=2) + "\n")


@pytest.fixture
def anchor(tmp_path):
    return _model(tmp_path / "anchor")


def test_the_fixture_is_corrected_by_the_build(anchor, tmp_path):
    built = _built(anchor, tmp_path / "cand0")
    assert json.loads((built / "build.json").read_text())["tokenizer_config"]["serialized_ok"] is False
    assert (built / "tokenizer_config.json").read_bytes() != (anchor / "tokenizer_config.json").read_bytes()


def test_a_build_of_the_anchor_passes(anchor, tmp_path):
    assert anchor_problem(_built(anchor, tmp_path / "cand0"), anchor) is None


def test_the_original_passes_beside_a_corrected_anchor(anchor, tmp_path):
    """``baselines[1]``: the original fine-tune listed beside an anchor that carries the correction."""
    corrected = _built(anchor, tmp_path / "corrected")
    assert anchor_problem(anchor, corrected) is None


def test_a_correction_that_is_a_no_op_under_this_transformers_passes(tmp_path):
    """A declaration transformers already loads as the stored pipeline: the build's class is the same tokenizer."""
    # Under the transformers this tree pins, LlamaTokenizerFast is rebuilt as the slow LlamaTokenizer
    # (a Metaspace pipeline), not the stored ByteLevel one; TokenizersBackend is the class that loads it as is.
    anchor = _model(tmp_path / "anchor", tokenizer_class="TokenizersBackend", model_type="llama")
    assert tokenizer_config_record(anchor)["serialized_ok"] is True
    candidate = shutil.copytree(anchor, tmp_path / "cand")
    _edit(candidate, tokenizer_class="PreTrainedTokenizerFast")
    assert anchor_problem(candidate, anchor) is None


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"chat_template": "{{ other }}"}, "chat_template"),
        ({"eos_token": "<bos>"}, "eos_token"),
        ({"model_max_length": 7}, "model_max_length"),
        ({"added_tokens_decoder": {}}, "added_tokens_decoder"),
    ],
)
def test_any_other_tokenizer_config_difference_is_refused_by_name(anchor, tmp_path, fields, named):
    """The mutation test: a change the build did not record still fails the anchor check, naming its field."""
    candidate = _built(anchor, tmp_path / "cand0")
    _edit(candidate, **fields)
    why = anchor_problem(candidate, anchor)
    assert why is not None and why.startswith("tokenizer_config.json differs from the anchor's: ")
    assert named in why


def test_a_class_other_than_the_builds_correction_is_refused(anchor, tmp_path):
    candidate = shutil.copytree(anchor, tmp_path / "cand")
    _edit(candidate, tokenizer_class="Qwen2TokenizerFast")
    why = anchor_problem(candidate, anchor)
    assert why is not None and "tokenizer_class 'Qwen2TokenizerFast'" in why and "LlamaTokenizerFast" in why


def test_the_correction_over_a_pipeline_the_build_does_not_correct_is_refused(tmp_path):
    """Outside the build's rule (no ByteLevel pipeline) a class change is not accounted for by any record."""
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    word = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
    word.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=word, unk_token="<unk>").save_pretrained(anchor)
    _edit(anchor, tokenizer_class="BertTokenizerFast")
    (anchor / "config.json").write_text(json.dumps({"model_type": "llama"}))
    candidate = shutil.copytree(anchor, tmp_path / "cand")
    _edit(candidate, tokenizer_class="PreTrainedTokenizerFast")
    why = anchor_problem(candidate, anchor)
    assert why is not None and "does not correct" in why and "not a ByteLevel pipeline" in why


def test_tokenizer_json_itself_stays_byte_for_byte(anchor, tmp_path):
    candidate = _built(anchor, tmp_path / "cand0")
    doc = json.loads((candidate / "tokenizer.json").read_text())
    (candidate / "tokenizer.json").write_text(json.dumps(doc))  # same content, other bytes
    assert anchor_problem(candidate, anchor) == "tokenizer.json differs from the anchor's"
