"""qda-kl applies the tokenizer-class correction ``qda build`` applies, instead of refusing the checkpoint.

The fixture is the shape of the R1 fine-tunes: a ByteLevel ``tokenizer.json`` under a ``tokenizer_config.json``
that declares ``LlamaTokenizerFast``. With a Qwen ``config.json`` this transformers builds another pipeline for
that declaration, so ``qda build`` writes ``PreTrainedTokenizerFast`` and records it; ``qda-kl ref`` must hand the
engine that same tokenizer, check it before the engine loads, and record the correction in the manifest.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import qda_kl.reference as reference
import qda_quant.output.vllm_env as vllm_env
from qda_kl.store import Reference, ReferenceWriter
from qda_kl.tokenizer import check_tokenizer
from qda_kl.tokenizer_check import (
    SERIALIZED_CLASS,
    correct_tokenizer,
    engine_tokenizer,
    preload_refusal,
    read_tokenizer_config,
)
from qda_quant.intake.tokenizer import TOKENIZER_PROBES, tokenizer_config_record
from qda_quant.step import Refused
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TEXT = ["Hello world, this is a test.\nNext line.", "  two  spaces\n\ttab What is 2 + 2?"]
TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}<|assistant|>"


def bytelevel_model(
    root: Path, tokenizer_class: str = "LlamaTokenizerFast", model_type: str = "qwen2"
) -> Path:
    """A checkpoint's tokenizer files: ByteLevel BPE, a chat template, and the declared class.

    transformers picks its own class for a ``qwen2`` config whatever is declared, so a declaration meant to be
    kept as it is goes with ``model_type="llama"``.
    """
    root.mkdir(parents=True)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(TEXT, trainers.BpeTrainer(vocab_size=300))
    PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>").save_pretrained(
        root
    )
    conf = json.loads((root / "tokenizer_config.json").read_text())
    conf.update(tokenizer_class=tokenizer_class, legacy=True, chat_template=TEMPLATE)
    (root / "tokenizer_config.json").write_text(json.dumps(conf))
    (root / "config.json").write_text(json.dumps({"model_type": model_type, "max_position_embeddings": 4096}))
    (root / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    return root


def _ids(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


@pytest.fixture
def model(tmp_path):
    return bytelevel_model(tmp_path / "m")


def test_the_fixture_is_the_case_the_build_corrects(model):
    """The declared class really rebuilds the pipeline here: otherwise every test below proves nothing."""
    declared, serialized = (
        AutoTokenizer.from_pretrained(model),
        PreTrainedTokenizerFast.from_pretrained(model),
    )
    assert any(_ids(declared, p) != _ids(serialized, p) for p in TOKENIZER_PROBES)
    assert tokenizer_config_record(model) == {
        "declared": "LlamaTokenizerFast",
        "serialized_ok": False,
        "written": SERIALIZED_CLASS,
    }


def test_the_decision_is_the_builds_own(model):
    config = read_tokenizer_config(model)
    assert config.record == tokenizer_config_record(model)
    assert config.corrected
    source = json.loads((model / "tokenizer_config.json").read_text())
    assert config.effective == {**source, "tokenizer_class": SERIALIZED_CLASS}


def test_the_corrected_tokenizer_is_the_checkpoints_own_but_the_class(model, tmp_path):
    into = tmp_path / "tok"
    record = correct_tokenizer(model, into)
    assert record == {"declared": "LlamaTokenizerFast", "serialized_ok": False, "written": SERIALIZED_CLASS}
    assert (into / "tokenizer.json").read_bytes() == (model / "tokenizer.json").read_bytes()
    assert (into / "generation_config.json").read_bytes() == (model / "generation_config.json").read_bytes()
    # written exactly as the build writes its corrected copy
    source = json.loads((model / "tokenizer_config.json").read_text())
    expected = json.dumps({**source, "tokenizer_class": SERIALIZED_CLASS}, indent=2) + "\n"
    assert (into / "tokenizer_config.json").read_text() == expected
    built = AutoTokenizer.from_pretrained(into)  # what the engine builds from it
    serialized = Tokenizer.from_file(str(model / "tokenizer.json"))
    for text in TOKENIZER_PROBES:
        assert _ids(built, text) == serialized.encode(text, add_special_tokens=False).ids
    assert built.chat_template == TEMPLATE  # the template is the checkpoint's, untouched


def test_a_declaration_that_already_reproduces_the_pipeline_is_left_alone(tmp_path):
    model = bytelevel_model(tmp_path / "m", tokenizer_class=SERIALIZED_CLASS, model_type="llama")
    into = tmp_path / "tok"
    args, record = engine_tokenizer(model, {"seed": 1}, into)
    assert (args, record) == ({"seed": 1}, {})
    assert not into.exists()


def test_an_explicit_tokenizer_argument_is_the_callers(model, tmp_path):
    into = tmp_path / "tok"
    args, record = engine_tokenizer(model, {"tokenizer": "/mine"}, into)
    assert (args, record) == ({"tokenizer": "/mine"}, {})
    assert not into.exists()


def test_the_engine_is_given_the_corrected_tokenizer(model, tmp_path):
    into = tmp_path / "tok"
    args, record = engine_tokenizer(model, {"seed": 1}, into)
    assert args == {"seed": 1, "tokenizer": str(into)}
    assert record["written"] == SERIALIZED_CLASS


def test_the_encode_check_runs_before_the_engine_and_names_a_real_remedy(model, tmp_path):
    """Uncorrected, the declared class is refused from the files alone; corrected, it passes."""
    message = preload_refusal(model, probes=(TEXT[1],))
    assert message is not None and "gives" in message
    assert "--serve-args" in message and '"tokenizer"' in message  # a flag `qda-kl ref` has
    assert "--tokenizer " not in message and "#" not in message  # no vLLM-only flag, no issue number
    into = tmp_path / "tok"
    correct_tokenizer(model, into)
    assert preload_refusal(into, probes=(TEXT[1],)) is None


def test_the_original_tokenizer_is_fingerprinted_as_the_engine_uses_it(model, tmp_path):
    """``qda-kl score``'s fingerprint check loads the reference's original through the same correction."""
    into = tmp_path / "tok"
    correct_tokenizer(model, into)
    engine = AutoTokenizer.from_pretrained(into)
    assert check_tokenizer({"path": str(model)}, engine)


def test_a_hub_id_gets_the_decision_its_snapshot_gets(model, tmp_path, monkeypatch):
    """By Hub id (with or without ``@rev``) and by local path: one record, one corrected tokenizer."""
    sha = "a" * 40
    repo = tmp_path / "hub" / "models--org--ft"
    snapshot = repo / "snapshots" / sha
    snapshot.mkdir(parents=True)
    for f in model.iterdir():
        (snapshot / f.name).write_bytes(f.read_bytes())
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    local = correct_tokenizer(model, tmp_path / "local")
    for i, name in enumerate(("org/ft", f"org/ft@{sha}")):
        into = tmp_path / f"hub{i}"
        assert correct_tokenizer(name, into, fetch=False) == local
        assert (into / "tokenizer_config.json").read_bytes() == (
            tmp_path / "local/tokenizer_config.json"
        ).read_bytes()


def test_a_class_the_build_cannot_compare_is_left_to_the_engine_check(tmp_path, monkeypatch):
    """qda-kl does not refuse what it cannot correct: the engine's own encode check still runs."""
    model = bytelevel_model(tmp_path / "m")

    def cannot(_src):
        raise Refused("tokenizer_config: cannot compare")

    monkeypatch.setattr("qda_quant.intake.tokenizer.tokenizer_config_record", cannot)
    assert engine_tokenizer(model, {}, tmp_path / "tok") == ({}, {})


def _install(monkeypatch, model):
    """A fake engine that builds its tokenizer as vLLM does: ``AutoTokenizer`` over ``tokenizer`` or the model."""
    opened = []

    class Engine:
        def __init__(self, settings):
            self.tok = AutoTokenizer.from_pretrained(settings.args.get("tokenizer") or model)
            self.llm_engine = SimpleNamespace(
                model_config=SimpleNamespace(get_vocab_size=lambda: 300, dtype="float32", max_model_len=4096)
            )

        def get_tokenizer(self):
            return self.tok

        def generate(self, prompts, params, **kwargs):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], finish_reason="stop")]) for _ in params
            ]

    def open_engine(model, **kwargs):
        opened.append(kwargs["settings"])
        return Engine(kwargs["settings"])

    class Writer:
        def __init__(self, out, **kwargs):
            self.build = kwargs["build"]

        def add(self, *args, **kwargs):
            pass

        def close(self):
            return self.build

    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    fake.TokensPrompt = lambda prompt_token_ids: SimpleNamespace(prompt_token_ids=prompt_token_ids)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(vllm_env, "vllm_environ", nullcontext)
    monkeypatch.setattr(reference, "open_engine", open_engine)
    monkeypatch.setattr(reference, "teacher_force", lambda *a, **k: {})
    monkeypatch.setattr(reference, "ReferenceWriter", Writer)
    return opened


def test_ref_builds_with_the_correction_and_records_it(model, tmp_path, monkeypatch):
    """The refusal this replaces: the engine's tokenizer encoded differently, found after the weights loaded."""
    opened = _install(monkeypatch, model)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": TEXT[1]}) + "\n")
    out = tmp_path / "ref"
    build = reference.build(str(model), prompts, out, answers=1, max_tokens=64)
    assert build["tokenizer_config"] == {
        "declared": "LlamaTokenizerFast",
        "serialized_ok": False,
        "written": SERIALIZED_CLASS,
    }
    (settings,) = opened
    assert settings.args["tokenizer"] == str(out.with_name(".ref.tokenizer"))
    assert "tokenizer" not in build["serve_args"]  # the recorded settings are the caller's, no staging path
    assert not out.with_name(".ref.tokenizer").exists()  # removed with the build


def test_ref_refuses_before_the_engine_when_an_explicit_tokenizer_rebuilds_the_pipeline(
    model, tmp_path, monkeypatch
):
    opened = _install(monkeypatch, model)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": TEXT[1]}) + "\n")
    with pytest.raises(Refused, match="--serve-args"):
        reference.build(
            str(model),
            prompts,
            tmp_path / "ref",
            answers=1,
            max_tokens=64,
            serve_args={"tokenizer": str(model)},
        )
    assert opened == []  # no engine was started for a check the files could answer


def test_ref_of_a_checkpoint_that_needs_no_correction_records_none(tmp_path, monkeypatch):
    model = bytelevel_model(tmp_path / "m", tokenizer_class=SERIALIZED_CLASS, model_type="llama")
    opened = _install(monkeypatch, model)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": TEXT[1]}) + "\n")
    build = reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)
    assert "tokenizer_config" not in build
    assert "tokenizer" not in opened[0].args


def _install_real(monkeypatch, model):
    """As :func:`_install`, but the real writer: the engine answers 20 tokens and the hook returns zeros."""
    opened = _install(monkeypatch, model)
    inner = reference.open_engine

    def open_engine(name, **kwargs):
        engine = inner(name, **kwargs)
        engine.generate = lambda prompts, params, **kw: [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(20)), finish_reason="stop")])
            for _ in params
        ]
        return engine

    def teacher_force(llm, mode, items, *, k=2, **kw):
        return {
            i.key: {
                "topv": np.zeros((len(i.tokens) - i.start, k), np.float32),
                "topi": np.zeros((len(i.tokens) - i.start, k), np.int32),
                "log_rest": np.zeros(len(i.tokens) - i.start, np.float32),
            }
            for i in items
        }

    monkeypatch.setattr(reference, "open_engine", open_engine)
    monkeypatch.setattr(reference, "teacher_force", teacher_force)
    monkeypatch.setattr(reference, "ReferenceWriter", ReferenceWriter)
    return opened


def engine_tokenizer_of(checkpoint, into):
    """The tokenizer an engine builds for `checkpoint`: its own files, with the build's class correction."""
    correct_tokenizer(checkpoint, into)
    return AutoTokenizer.from_pretrained(into)


def tiny_reference(monkeypatch, model, tmp_path, out):
    """Build a reference of one prompt from `model`, written as `qda-kl ref` writes it."""
    _install_real(monkeypatch, model)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": TEXT[1]}) + "\n")
    reference.build(str(model), prompts, out, answers=1, max_tokens=64, k=2)
    return Reference(out, verify=True)


def test_a_reference_carries_the_anchor_tokenizer_it_was_built_with(model, tmp_path, monkeypatch):
    """The corrected tokenizer the engine read is in the reference, with where it came from."""
    ref = tiny_reference(monkeypatch, model, tmp_path, tmp_path / "ref")
    assert ref.tokenizer == tmp_path / "ref" / "tokenizer"
    assert (ref.tokenizer / "tokenizer.json").read_bytes() == (model / "tokenizer.json").read_bytes()
    written = json.loads((ref.tokenizer / "tokenizer_config.json").read_text())
    assert written["tokenizer_class"] == SERIALIZED_CLASS  # the corrected form, as qda build writes it
    assert written["chat_template"] == TEMPLATE
    assert not (ref.tokenizer / "model.safetensors").exists()  # the tokenizer, never the weights
    recorded = ref.manifest.model["tokenizer"]
    assert recorded["path"] == str(model.resolve()) and recorded["dir"] == "tokenizer"
    assert recorded["record"] == tokenizer_config_record(model)
    assert ref.manifest.files["tokenizer/tokenizer.json"]  # hashed, so it is part of content_sha256


def test_a_reference_scores_after_its_model_directory_is_deleted(model, tmp_path, monkeypatch):
    """The live failure: the base model's weights were deleted to free disk, the reference intact."""
    checkpoint = bytelevel_model(tmp_path / "ckpt")  # the candidate: the same tokenizer, another build
    ref = tiny_reference(monkeypatch, model, tmp_path, tmp_path / "ref")
    shutil.rmtree(model)
    candidate = engine_tokenizer_of(checkpoint, tmp_path / "tok")
    assert check_tokenizer(ref.manifest.model, candidate, copy=ref.tokenizer)


def test_a_reference_without_a_copy_falls_back_to_the_path_and_names_it_when_it_is_gone(model, tmp_path):
    """A reference built before this change: the recorded path while it is there, one refusal when it is not."""
    original = {"path": str(model)}
    candidate = engine_tokenizer_of(model, tmp_path / "tok")
    assert check_tokenizer(original, candidate)
    shutil.rmtree(model)
    with pytest.raises(Refused) as refused:
        check_tokenizer(original, candidate)
    assert str(model) in str(refused.value) and "rebuild the reference" in str(refused.value)


def test_a_candidate_whose_tokenizer_differs_is_refused_by_name(model, tmp_path, monkeypatch):
    """The copy is the identity: a candidate with other token meanings is refused, the model dir gone."""
    other = bytelevel_model(tmp_path / "other")
    (other / "tokenizer.json").write_bytes((model / "tokenizer.json").read_bytes())
    ref = tiny_reference(monkeypatch, model, tmp_path, tmp_path / "ref")
    shutil.rmtree(model)
    tok = engine_tokenizer_of(other, tmp_path / "tok")
    tok.add_tokens(["<extra-id-0>"])
    with pytest.raises(ValueError, match="fingerprint mismatch in tokens"):
        check_tokenizer(ref.manifest.model, tok, copy=ref.tokenizer)
