"""KL scoring starts its engine without FlashInfer's JIT-built sampler, GDN prefill and all-reduce.

``qda-kl score`` (and the judge's readings through it) died in nvcc while FlashInfer built ``sampling``, its GDN
prefill and, at tp > 1, ``trtllm_mnnvl_comm``. The engine now starts with the same settings a gencheck uses, and
with ``max_num_seqs`` at the reference's sequence count (vLLM's default did not fit a hybrid model's cache), and
with the kernel settings the checkpoint's card serve line shows. It times no FlashInfer tactic either: autotune
took most of a W4A4 engine's start on an RTX PRO 6000.
"""

import itertools
import json
import shlex
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from qda_kl import score
from qda_kl.scorers import TopKScorer
from qda_kl.store import ReferenceWriter
from qda_kl.topk import anchor_topk
from qda_quant.format import Format
from qda_quant.output.serve_command import NVIDIA_LIBS, serve_command_line
from qda_quant.target import Target, merge_serve
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

NO_JIT = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ALLREDUCE_USE_FLASHINFER": "0"}


def tiny_reference(tmp_path, keys=("a",)):
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
    for key in keys:
        writer.add(key, "math", [0, 1], 1, anchor_topk(torch.zeros((1, 4)).log_softmax(-1), 2))
    writer.close()
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(dtype="bfloat16", get_vocab_size=lambda: 4)),
        collective_rpc=lambda _: [{}],
        get_tokenizer=lambda: tokenizer,
    )
    return ref, engine


@pytest.mark.parametrize(
    ("serve_args", "fusion", "gdn"),
    [
        ({}, False, "triton"),
        ({"compilation_config": {"pass_config": {"fuse_allreduce_rms": True}}}, True, "triton"),
        (
            {"additional_config": {"gdn_prefill_backend": "flashinfer"}},
            False,
            "flashinfer",
        ),
    ],
)
def test_topk_scorer_starts_without_runtime_jit(tmp_path, monkeypatch, serve_args, fusion, gdn):
    """The engine gets NO_JIT's settings unless ``serve_args`` names the fusion or the GDN prefill backend."""
    ref, engine = tiny_reference(tmp_path)
    opened = []
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    monkeypatch.setattr(
        score, "teacher_force", lambda *a, **k: {"a": dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))}
    )

    outcome = TopKScorer(serve_args=serve_args).run(str(tmp_path), ref)

    settings = opened[0]["settings"]
    assert {key: settings.env.get(key) for key in NO_JIT} == NO_JIT
    assert settings.args["compilation_config"]["pass_config"]["fuse_allreduce_rms"] is fusion
    nested = settings.args.get("additional_config", {}).get("gdn_prefill_backend")
    assert settings.args.get("gdn_prefill_backend", nested) == gdn
    assert {key: outcome.result.serve.env.get(key) for key in NO_JIT} == NO_JIT  # the reading records them


@pytest.mark.parametrize(("serve_args", "expected"), [({}, 3), ({"max_num_seqs": 2}, 2)])
def test_the_score_engine_serves_the_references_sequences(tmp_path, monkeypatch, serve_args, expected):
    """``max_num_seqs`` is the sequence count unless ``serve_args`` names it, and the reading records it."""
    keys = ("a", "b", "c")
    ref, engine = tiny_reference(tmp_path, keys)
    opened = []
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    zeros = dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))
    monkeypatch.setattr(score, "teacher_force", lambda *a, **k: dict.fromkeys(keys, zeros))

    outcome = TopKScorer(serve_args=serve_args).run(str(tmp_path), ref)

    assert opened[0]["settings"].args["max_num_seqs"] == expected
    assert outcome.result.serve.args["max_num_seqs"] == expected


def test_the_score_window_is_capped_by_the_checkpoints_context(tmp_path, monkeypatch):
    """A model whose answers fill its context (a 4k model) was refused at engine start, because the
    scorer asked for the longest sequence + 16 without the cap the reference's own window has."""
    import json

    ref, engine = tiny_reference(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 10}))
    opened = []
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    monkeypatch.setattr(
        score, "teacher_force", lambda *a, **k: {"a": dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))}
    )
    TopKScorer().run(str(tmp_path), ref)
    assert opened[0]["max_model_len"] == 10  # 2 tokens + 16 would be 18, over the model's 10
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    TopKScorer().run(str(tmp_path), ref)
    assert opened[1]["max_model_len"] == 18  # under the cap: the sequence and its margin, as before


def _kl_engine_settings(tmp_path, monkeypatch, vllm="0.33.0", kernels=None):
    """The settings `qda-kl kl`'s scorer starts the engine on `tmp_path` with."""
    ref, engine = tiny_reference(tmp_path)
    engine.collective_rpc = lambda _: [kernels or {}]
    opened = []
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(score, "vllm_version", lambda: vllm)
    monkeypatch.setattr(
        score, "teacher_force", lambda *a, **k: {"a": dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))}
    )
    TopKScorer().run(str(tmp_path), ref)
    return opened[0]["settings"]


@pytest.mark.parametrize("target", ["sm90@0.33.0", "sm90@0.33.0:marlin"])
def test_kl_serves_a_built_checkpoint_on_its_card_lines_kernels(tmp_path, monkeypatch, target):
    """The engine `qda-kl kl` starts has the kernel env and backend flags of the checkpoint's card serve line.

    Settings only: the engine is a stub that reports the planned kernels, so `check_kernels` cannot fail here.
    `test_kl_binds_the_card_lines_kernels` (gpu) serves a checkpoint both ways and compares the bound kernels.
    """
    t, fmt = Target.parse(target), Format.parse("w4a16")
    serve = merge_serve({"dense": t.plan(fmt, "dense"), "moe": t.plan(fmt, "moe")}, t.vllm)
    (tmp_path / "build.json").write_text(json.dumps({"serve": serve}))
    settings = _kl_engine_settings(tmp_path, monkeypatch, t.vllm, serve["kernels"])

    line = serve_command_line(tmp_path).removeprefix(NVIDIA_LIBS + " ")  # the loader prefix is not a setting
    words = shlex.split(line)
    env = dict(w.split("=", 1) for w in itertools.takewhile(lambda w: "=" in w, words))
    argv = words[len(env) :]
    flags = {k: argv[argv.index(f"--{k.replace('_', '-')}") + 1] for k in ("linear_backend", "moe_backend")
             if f"--{k.replace('_', '-')}" in argv}  # fmt: skip
    for key in ("VLLM_DISABLED_KERNELS", "VLLM_PLUGINS"):
        assert settings.env.get(key) == env.get(key), key
    assert {k: v for k, v in settings.args.items() if k in ("linear_backend", "moe_backend")} == flags
    assert "MacheteLinearKernel" in env["VLLM_DISABLED_KERNELS"].split(",")


def test_kl_disables_machete_on_a_checkpoint_qda_did_not_build(tmp_path, monkeypatch):
    """A rival's checkpoint has no build.json; its engine still starts with Machete off."""
    assert _kl_engine_settings(tmp_path, monkeypatch).env["VLLM_DISABLED_KERNELS"] == "MacheteLinearKernel"


def test_kl_keeps_the_kernels_a_user_disabled(tmp_path, monkeypatch):
    """`VLLM_DISABLED_KERNELS` the user exported stays off beside the checkpoint's own list and Machete."""
    t = Target.parse("sm90@0.32.0:marlin")  # its plan disables Humming
    serve = merge_serve({"dense": t.plan(Format.parse("w4a16"), "dense")}, t.vllm)
    (tmp_path / "build.json").write_text(json.dumps({"serve": serve}))
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "ExllamaLinearKernel")
    off = _kl_engine_settings(tmp_path, monkeypatch, t.vllm, serve["kernels"]).env["VLLM_DISABLED_KERNELS"]
    assert off.split(",") == ["ExllamaLinearKernel", "HummingLinearKernel", "MacheteLinearKernel"]
    assert "ExllamaLinearKernel" not in serve_command_line(tmp_path)  # a card line never carries the builder's own


def test_kl_scores_with_flashinfer_autotune_off_and_the_serve_line_keeps_vllms_default(tmp_path, monkeypatch):
    """The scoring engine starts with ``enable_flashinfer_autotune: False``; the printed serve line names no autotune."""
    t = Target.parse("sm120@0.33.0")
    serve = merge_serve({"dense": t.plan(Format.parse("w4a4"), "dense")}, t.vllm)
    (tmp_path / "build.json").write_text(json.dumps({"serve": serve}))
    settings = _kl_engine_settings(tmp_path, monkeypatch, t.vllm, serve["kernels"])
    assert settings.args["enable_flashinfer_autotune"] is False
    assert "autotune" not in serve_command_line(tmp_path)
