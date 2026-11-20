"""The reference build and self-generation sample from one resolved mapping (fake engine, CPU)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import qda_kl.reference as reference
import qda_quant.output.vllm_env as vllm_env
from qda_quant.calib import selfgen as S
from qda_quant.step import Refused

_UNSET = object()  # _install's sentinel: "write the usual generation_config.json"


class _Engine:
    """vLLM's ``LLM`` for both paths: records the ``SamplingParams`` and writes no scored token."""

    def __init__(self, window=1 << 20, tokens=False):
        self.params = []
        self.tokens = tokens  # True: answer with exactly max_tokens tokens, as vLLM does at its cap
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 10, dtype="float32", max_model_len=window)
        )

    def get_tokenizer(self):
        return SimpleNamespace(apply_chat_template=lambda *a, **k: {"input_ids": [1, 2]})

    def generate(self, prompts, params, **kwargs):
        self.params = params
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[0] * p.kwargs["max_tokens"] if self.tokens else [],
                        finish_reason="length",
                    )
                ]
            )
            for p in params
        ]

    def chat(self, prompts, params, **kwargs):
        self.params = params
        return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")]) for _ in prompts]


class _Writer:
    def __init__(self, out, **kwargs):
        self.build = kwargs["build"]

    def add(self, *args, **kwargs):
        pass

    def close(self):
        return self.build


def _install(monkeypatch, tmp_path, window=1 << 20, tokens=False, named=None, generation=_UNSET):
    """Install a fake ``vllm``, engine and writer; return the model, the prompt file and the captured params.

    ``named`` becomes the model's ``_name_or_path`` (the original's Hub id, as a downloaded snapshot writes it);
    ``generation`` the ``generation_config.json`` mapping, the file being left out when it is None.
    """
    captured = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured.append(kwargs)

    engine = _Engine(window=window, tokens=tokens)
    fake = types.ModuleType("vllm")
    fake.SamplingParams = SamplingParams
    fake.TokensPrompt = lambda prompt_token_ids: SimpleNamespace(prompt_token_ids=prompt_token_ids)
    fake.LLM = lambda *a, **k: engine
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(vllm_env, "vllm_environ", nullcontext)
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: engine)
    monkeypatch.setattr(reference, "teacher_force", lambda *a, **k: {})
    monkeypatch.setattr(reference, "ReferenceWriter", _Writer)
    # the fake engine's prompts, [1, 2]: two tokens each
    monkeypatch.setattr(reference, "prompt_token_counts", lambda _m, items, **_k: [(p.id, 2) for p in items])

    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"_name_or_path": named} if named else {}))
    if generation is _UNSET:
        generation = {
            "do_sample": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.5,
        }
    if generation is not None:
        (model / "generation_config.json").write_text(json.dumps(generation))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "prompt": "hi"}\n')
    return model, prompts, captured


def test_greedy_sampling_is_the_same_in_the_reference_and_in_selfgen(tmp_path, monkeypatch):
    """``do_sample: false`` makes temperature 0 but keeps ``repetition_penalty``: both paths get one mapping."""
    model, prompts, captured = _install(monkeypatch, tmp_path)

    S.vllm_generate(str(model), ["hi"], 1, 64, 17, S.resolve_sampling(model, None, "sampling"), {})
    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[0] == captured[1]
    assert captured[0]["temperature"] == 0.0 and captured[0]["repetition_penalty"] == 1.5


def test_a_model_whose_file_names_no_sampling_is_sampled_as_its_released_cards_were(tmp_path, monkeypatch):
    """Qwen3.5-9B's generation_config.json names no sampling: the reference takes the released cards' row.

    The row is the `full` preset's one table, so the reference and a `full` eval of the model are sampled
    alike instead of the build being refused.
    """
    model, prompts, captured = _install(
        monkeypatch, tmp_path, named="Qwen/Qwen3.5-9B", generation={"eos_token_id": 1}
    )

    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[0]["temperature"] == 1.0 and captured[0]["top_p"] == 0.95
    assert captured[0]["top_k"] == 20 and captured[0]["presence_penalty"] == 1.5
    assert reference.sampling_note(str(model)) == "ref: sampling of Qwen/Qwen3.5-9B's released cards"


def test_the_released_cards_row_wins_over_the_models_own_generation_config(tmp_path, monkeypatch):
    """A model the table names is sampled as its released cards were, whatever its file names."""
    model, prompts, captured = _install(
        monkeypatch, tmp_path, named="Qwen/Qwen3.5-9B", generation={"temperature": 0.0, "top_p": 0.5}
    )

    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[0]["temperature"] == 1.0 and captured[0]["top_p"] == 0.95  # the row's
    assert captured[0]["top_k"] == 20 and captured[0]["presence_penalty"] == 1.5  # the row's
    assert reference.sampling_note(str(model)) == "ref: sampling of Qwen/Qwen3.5-9B's released cards"


def test_a_model_the_cards_name_never_reads_its_generation_config(tmp_path, monkeypatch):
    """The row replaces the file: an unreadable file is not a failure and cannot top the row up."""
    model, prompts, captured = _install(monkeypatch, tmp_path, named="Qwen/Qwen3.5-9B", generation={})
    (model / "generation_config.json").write_text("{ not json")  # a file no reader could use

    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[0]["presence_penalty"] == 1.5 and captured[0]["temperature"] == 1.0  # the row's


def test_an_explicit_setting_wins_field_by_field_over_the_row(tmp_path, monkeypatch):
    """``--sampling`` changes only the fields it names; the cards still fill the rest."""
    model, prompts, captured = _install(
        monkeypatch, tmp_path, named="Qwen/Qwen3.5-9B", generation={"temperature": 0.0, "top_p": 0.5}
    )

    reference.build(
        str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64, sampling={"temperature": 0.7}
    )

    assert captured[0]["temperature"] == 0.7  # the setting
    assert captured[0]["top_k"] == 20 and captured[0]["presence_penalty"] == 1.5  # the cards
    assert reference.sampling_note(str(model), {"temperature": 0.7}) == (
        "ref: sampling of Qwen/Qwen3.5-9B's released cards"
    )


def test_a_setting_naming_every_field_of_the_row_needs_no_note(tmp_path, monkeypatch):
    """The note explains a sampling the user did not spell out; a full override consulted the rows for nothing."""
    from qda_kl.eval.full_preset import FAMILIES

    model, prompts, captured = _install(monkeypatch, tmp_path, named="Qwen/Qwen3.5-9B")
    row = dict(FAMILIES["Qwen/Qwen3.5-9B"].sampling)
    row["temperature"] = 0.7

    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64, sampling=row)

    assert captured[0]["temperature"] == 0.7 and captured[0]["presence_penalty"] == 1.5
    assert reference.sampling_note(str(model), row) is None


def test_default_sampling_agrees_with_the_build_the_prepare_hands_it_to(tmp_path):
    """The flag `qda-agent prepare` / `qda-kl kl` pass and the build's own resolution are one rule.

    A mismatch is two references (two cache keys) of one model: the flag must be None exactly where the build
    resolves the settings by itself, and the flag, when given, must be what the build resolves.
    """
    from qda_kl.eval.full_preset import FAMILIES

    row = FAMILIES["Qwen/Qwen3.5-9B"].sampling
    nine = tmp_path / "nine"
    nine.mkdir()
    (nine / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-9B"}))
    (nine / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    assert reference.default_sampling(nine) is None  # the build takes the row itself
    assert reference.reference_sampling(str(nine)) == row

    own = tmp_path / "own"
    own.mkdir()
    (own / "generation_config.json").write_text(json.dumps({"temperature": 0.6}))
    assert reference.default_sampling(own) is None  # the build reads the file itself
    assert reference.reference_sampling(str(own)) == {"temperature": 0.6}  # and the cards stay out of it

    named = tmp_path / "named"  # a row and a file: the build takes the row, so the flag stays out too
    named.mkdir()
    (named / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-9B"}))
    (named / "generation_config.json").write_text(json.dumps({"temperature": 0.6}))
    assert reference.default_sampling(named) is None
    assert reference.reference_sampling(str(named)) == row

    tune = tmp_path / "tune"
    tune.mkdir()
    (tune / "config.json").write_text("{}")
    assert reference.default_sampling(tune) == dict(reference.DEFAULT_SAMPLING)  # prepare never refuses
    with pytest.raises(Refused):
        reference.reference_sampling(str(tune))


def test_a_model_in_neither_the_file_nor_the_cards_refuses_with_a_copyable_example(tmp_path, monkeypatch):
    """Out of the table and with no sampling in its file: refused before the engine, in two lines."""
    model, prompts, _ = _install(monkeypatch, tmp_path, generation={"eos_token_id": 1})

    with pytest.raises(Refused) as why:
        reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    lines = str(why.value).splitlines()
    assert len(lines) == 2 and "no sampling settings" in lines[0]
    assert lines[1] == 'fix: pass --sampling \'{"temperature": 1.0, "top_p": 0.95}\''
    assert not (tmp_path / "ref").exists()


def test_a_model_out_of_the_table_still_takes_its_own_file_and_no_note(tmp_path, monkeypatch):
    """An arbitrary fine-tuned model is not given the cards' settings: its file names its own."""
    model, prompts, captured = _install(monkeypatch, tmp_path, named="acme/finetune-9B")

    reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[0]["temperature"] == 0.0 and "presence_penalty" not in captured[0]
    assert reference.sampling_note(str(model)) is None


def test_the_manifest_records_the_effective_engine_environment(tmp_path, monkeypatch):
    """The manifest holds the NVLS default plus the overrides, and refuses a second value for the key."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    got = reference.build(
        str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64, serve_env={"A": "1"}
    )
    from qda_kl.serve import NO_JIT_ENV
    from qda_quant.output.vllm_env import CACHE_ENV

    # the engine's compile caches are resolved elsewhere (`cache_env`): not this test's subject
    env = {k: v for k, v in got["serve_env"].items() if k not in CACHE_ENV}
    # the no-JIT settings too, and Machete off as on every engine
    assert env == {"NCCL_NVLS_ENABLE": "0", "A": "1", **NO_JIT_ENV, "VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
    with pytest.raises(ValueError, match="set nccl_nvls instead"):
        reference.build(
            str(model),
            prompts,
            tmp_path / "ref2",
            answers=1,
            max_tokens=64,
            serve_env={"NCCL_NVLS_ENABLE": "1"},
        )


def test_the_manifest_pins_a_deterministic_reduction_config(tmp_path, monkeypatch):
    """A cold engine must not autotune onto another bf16 variant; serve_args may still override it."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    got = reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)
    pin = got["serve_args"]["compilation_config"]["inductor_compile_config"]
    assert pin["deterministic"] is True
    got = reference.build(
        str(model),
        prompts,
        tmp_path / "ref2",
        answers=1,
        max_tokens=64,
        serve_args={"compilation_config": {"inductor_compile_config": {"deterministic": False}}},
    )
    assert got["serve_args"]["compilation_config"]["inductor_compile_config"]["deterministic"] is False


def test_the_engine_serves_the_answers_it_generates_at_once(tmp_path, monkeypatch):
    """``max_num_seqs`` is answers x prompts, not vLLM's default, unless serve_args names it; recorded."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    prompts.write_text('{"id": "a", "prompt": "hi"}\n{"id": "b", "prompt": "ho"}\n')
    engine, opened = reference.open_engine(), []  # the fake's engine, whatever it is called with
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: opened.append(k["settings"].args) or engine)
    got = reference.build(str(model), prompts, tmp_path / "ref", answers=3, max_tokens=64)
    assert opened[0]["max_num_seqs"] == got["max_num_seqs"] == 6 and "max_num_seqs" not in got["serve_args"]
    got = reference.build(
        str(model), prompts, tmp_path / "ref2", answers=3, max_tokens=64, serve_args={"max_num_seqs": 2}
    )
    assert opened[1]["max_num_seqs"] == got["max_num_seqs"] == got["serve_args"]["max_num_seqs"] == 2


def test_dropped_prompts_are_the_prompts_whose_every_draw_was_dropped():
    """A build covers a prompt it dropped only when no draw survived; one scored draw means it is not covered."""
    assert reference.dropped_prompts({"answers": 2, "dropped_short": ["gone#0", "gone#1", "half#1"]}) == {
        "gone"
    }
    assert reference.dropped_prompts({"answers": 1}) == set()
    assert reference.dropped_prompts({"answers": 1, "dropped_short": ["x#0"]}) == {"x"}


def test_an_answer_is_capped_to_leave_room_for_teacher_forcing(tmp_path, monkeypatch):
    """The window, not ``max_tokens``, caps an answer when it is short: 25 - 1 - 2 prompt tokens = 22."""
    model, prompts, captured = _install(monkeypatch, tmp_path, window=25, tokens=True)
    forced = []

    def teacher_force(llm, mode, items, **k):
        forced.extend(len(i.tokens) for i in items)
        arrays = {"topv": np.zeros((1, 2), np.float32), "topi": np.zeros((1, 2), np.int32)}
        return {i.key: {**arrays, "log_rest": np.zeros(1, np.float32)} for i in items}

    monkeypatch.setattr(reference, "teacher_force", teacher_force)

    got = reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert captured[-1]["max_tokens"] == 22
    assert forced == [24]  # 2 prompt tokens + the 22-token answer: one token short of the 25-token window
    assert got["capped_answers"] == {"a#0": 22} and got["sampling"]["max_tokens"] == 64


def test_a_window_with_room_keeps_the_requested_cap_and_the_manifest(tmp_path, monkeypatch):
    """Where the cap never binds, the sampling params and the build record are what they were."""
    _, prompts, captured = _install(monkeypatch, tmp_path)

    got = reference.build(str(tmp_path / "m"), prompts, tmp_path / "ref", answers=2, max_tokens=64)

    assert [g["max_tokens"] for g in captured] == [64, 64]
    assert "capped_answers" not in got


def test_a_prompt_with_no_room_is_refused_by_name(tmp_path, monkeypatch):
    """A prompt that fills the window is refused before anything is generated."""
    model, prompts, captured = _install(monkeypatch, tmp_path, window=2)  # the prompt is [1, 2]
    with pytest.raises(ValueError, match="a: 2 prompt tokens leave no answer room in the 2-token window"):
        reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)
    assert captured == []


def test_the_window_holds_the_longest_prompt_and_the_answer(tmp_path):
    """A 21k-token document fits: the window is the longest prompt + 1 + max_tokens, never past the context."""
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 32768}}))
    assert reference.max_model_len(str(model), 8192) == 16384  # short prompts: the window it always was
    assert reference.max_model_len(str(model), 8192, 21487) == 21487 + 1 + 8192
    assert reference.max_model_len(str(model), 8192, 30000) == 32768  # the context caps it


def test_the_longest_prompt_loads_its_tokenizer_with_the_runs_options(monkeypatch):
    """A `--serve-args` option (`trust_remote_code`) reaches the tokenizer `longest_prompt` builds."""
    from transformers import AutoTokenizer

    seen = []

    class Tok:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1, 2, 3]}

        def apply_chat_template(self, messages, **kw):
            return [1, 2, 3]

    def load(model, **kwargs):
        seen.append((model, kwargs))
        return Tok()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    item = reference.Prompt(id="a", prompt="hi")
    assert reference.longest_prompt("m", [item], max_text_tokens=100, trust_remote_code=True) == (3, "a")
    assert seen == [("m", {"trust_remote_code": True})]


def test_a_trust_remote_code_serve_arg_reaches_the_window_tokenizer(tmp_path, monkeypatch):
    """The build hands `longest_prompt` the same options it hands `preload_refusal`."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    seen = {}

    def counts(*a, **kw):
        seen.update(kw)
        return [("a", 2)]

    monkeypatch.setattr(reference, "prompt_token_counts", counts)
    reference.build(
        str(model),
        prompts,
        tmp_path / "ref",
        answers=1,
        max_tokens=64,
        serve_args={"trust_remote_code": True},
    )
    assert seen.get("trust_remote_code") is True


def test_a_prompt_past_the_context_is_refused_before_the_engine_loads(tmp_path, monkeypatch):
    model, prompts, captured = _install(monkeypatch, tmp_path)
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 16}))
    monkeypatch.setattr(reference, "prompt_token_counts", lambda *a, **k: [("a", 15)])
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: pytest.fail("the engine loaded"))
    with pytest.raises(
        Refused, match="a: 1 of 1 prompts \\(100.0%\\) leave no answer room in the model's 16-token context"
    ):
        reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)
    assert captured == []


def test_a_second_build_of_the_same_output_is_refused_by_name(tmp_path, monkeypatch):
    """A duplicate launch is refused at once and names the build that holds the lock."""
    model, prompts, _ = _install(monkeypatch, tmp_path)

    with (
        reference.output_lock(tmp_path / "ref"),
        pytest.raises(Refused, match=rf"pid {os.getpid()} on {re.escape(socket.gethostname())} since "),
    ):
        reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64)


def test_builds_of_different_outputs_do_not_block_each_other(tmp_path, monkeypatch):
    """The lock is per output directory: another build's output is not this build's business."""
    model, prompts, _ = _install(monkeypatch, tmp_path)

    with reference.output_lock(tmp_path / "ref"):
        got = reference.build(str(model), prompts, tmp_path / "other", answers=1, max_tokens=64)
    assert got["answers"] == 1


def test_the_lock_is_released_after_a_build_and_after_an_exception(tmp_path, monkeypatch):
    """The lock is held only while the build runs, on the error path too: the next build takes it."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    reference.build(str(model), prompts, tmp_path / "once", answers=1, max_tokens=64)

    with pytest.raises(RuntimeError, match="boom"), reference.output_lock(tmp_path / "ref"):
        raise RuntimeError("boom")
    with reference.output_lock(tmp_path / "ref"):  # neither the build above nor the error left it taken
        pass


def test_a_named_subset_is_recorded_and_its_ids_must_be_prompts(tmp_path, monkeypatch):
    model, prompts, _ = _install(monkeypatch, tmp_path)
    recorded = {}
    monkeypatch.setattr(
        reference, "ReferenceWriter", lambda out, **kw: recorded.update(kw) or _Writer(out, **kw)
    )
    reference.build(
        str(model), prompts, tmp_path / "ref", answers=1, max_tokens=64, subsets={"search96": ["a"]}
    )
    assert recorded["subsets"] == {"search96": ["a"]}
    with pytest.raises(Refused, match="subset search96: 1 ids are not prompts"):
        reference.build(
            str(model), prompts, tmp_path / "r2", answers=1, max_tokens=64, subsets={"search96": ["b"]}
        )


def test_a_reference_engine_starts_with_the_no_jit_settings_of_scoring_and_gencheck():
    """The reference engine was the one engine QDA opens without `no_jit`, so it JIT-built FlashInfer's
    sampler (and at tp > 1 its all-reduce) that score and gencheck are guarded against."""
    from qda_kl.serve import NO_JIT_ENV

    got = reference.engine_settings({"A": "1"}, {"max_num_seqs": 8}, False)
    assert {**got.env} == {"A": "1", **NO_JIT_ENV}
    assert got.args["compilation_config"]["pass_config"]["fuse_allreduce_rms"] is False
    assert got.args["compilation_config"]["inductor_compile_config"]["deterministic"] is True
    assert got.args["max_num_seqs"] == 8 and got.args["gdn_prefill_backend"] == "triton"
    assert got.args["enable_flashinfer_autotune"] is False


def test_one_engine_load_builds_every_group_and_each_keeps_its_own_manifest(tmp_path, monkeypatch):
    """Prompt groups share one engine load; the window and the batch cover the largest group, and each
    group writes its own directory and records its own prompts and subsets."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    prompts.write_text('{"id": "a", "prompt": "hi"}\n{"id": "b", "prompt": "ho"}\n')
    other = tmp_path / "q.jsonl"
    other.write_text('{"id": "c", "prompt": "he"}\n')
    engine = reference.open_engine()
    opened, written = [], []

    def counts(_model, items, **kw):
        assert [p.id for p in items] == ["a", "b", "c"]  # every group, one tokenizer load
        return [(p.id, 5) for p in items]

    monkeypatch.setattr(reference, "prompt_token_counts", counts)
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: opened.append(k) or engine)
    monkeypatch.setattr(
        reference, "ReferenceWriter", lambda out, **kw: written.append((out, kw)) or _Writer(out, **kw)
    )

    got = reference.build_many(
        str(model),
        [
            reference.Group(prompts, tmp_path / "one", {"search96": ["a"]}),
            reference.Group(other, tmp_path / "two"),
        ],
        answers=2,
        max_tokens=64,
    )

    assert len(opened) == 1  # one engine load for both references
    assert opened[0]["max_model_len"] == reference.max_model_len(str(model), 64, 5)
    assert opened[0]["settings"].args["max_num_seqs"] == 4  # the largest group: 2 prompts x 2 answers
    assert [m["prompts"] for m in got] == [str(prompts), str(other)]
    assert [Path(out) for out, _ in written] == [tmp_path / "one", tmp_path / "two"]
    assert written[0][1]["subsets"] == {"search96": ["a"]} and written[1][1]["subsets"] == {}
    assert [m["prompts_sha256"] for m in got] == [
        hashlib.sha256(prompts.read_bytes()).hexdigest(),
        hashlib.sha256(other.read_bytes()).hexdigest(),
    ]


def test_an_existing_group_output_is_refused_before_the_engine_loads(tmp_path, monkeypatch):
    """A multi-group build never writes one group and then refuses the other."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    other = tmp_path / "q.jsonl"
    other.write_text('{"id": "b", "prompt": "ho"}\n')
    (tmp_path / "two").mkdir()
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: pytest.fail("the engine loaded"))
    with pytest.raises(Refused, match="exists and holds no reference"):
        reference.build_many(
            str(model),
            [reference.Group(prompts, tmp_path / "one"), reference.Group(other, tmp_path / "two")],
            answers=1,
            max_tokens=64,
        )


def test_ref_into_an_existing_reference_is_refused_in_two_lines(tmp_path, monkeypatch, capsys):
    """`ref --out` on a reference already there refuses before the engine loads and names the one to score."""
    from qda_kl import cli
    from qda_quant.exit import USAGE

    model, prompts, _ = _install(monkeypatch, tmp_path)
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "manifest.json").write_text("{}")
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: pytest.fail("the engine loaded"))

    rc = cli.main(["ref", "--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "ref")])

    assert rc == USAGE
    assert capsys.readouterr().err.splitlines() == [
        f"error: {tmp_path / 'ref'} already holds a reference; qda-kl ref never overwrites one",
        f"fix:   score against it with `qda-kl score` (`reference: {tmp_path / 'ref'}`), or pass another --out",
    ]


def test_two_groups_on_one_directory_are_refused(tmp_path, monkeypatch):
    model, prompts, _ = _install(monkeypatch, tmp_path)
    with pytest.raises(Refused, match="same reference directory"):
        reference.build_many(
            str(model),
            [reference.Group(prompts, tmp_path / "ref"), reference.Group(prompts, tmp_path / "ref")],
            answers=1,
            max_tokens=64,
        )


class _CacheWriter(_Writer):
    """`ReferenceWriter` for the cache tests: writes the manifest, which is what makes an entry complete."""

    def __init__(self, out, **kwargs):
        super().__init__(out, **kwargs)
        self.out = Path(out)

    def close(self):
        out = Path(self.out)
        out.mkdir(parents=True)
        # a manifest a reader accepts: the record refcache rewrites when it gives the entry its prompt copy
        (out / "manifest.json").write_text(json.dumps({"k": 1, "vocab": 1, "build": self.build}))
        return self.build


def _cache_writer(monkeypatch):
    monkeypatch.setattr(reference, "ReferenceWriter", _CacheWriter)


def test_one_engine_load_builds_every_cached_group(tmp_path, monkeypatch):
    """On the reference cache, two groups build in one `build_many` call inside `refcache.cached_many`, and
    each lands in its own cache entry; a second call links them without an engine."""
    from qda_kl import refcache

    model, prompts, _ = _install(monkeypatch, tmp_path)
    other = tmp_path / "q.jsonl"
    other.write_text('{"id": "c", "prompt": "he"}\n')
    cache = tmp_path / "cache"
    monkeypatch.setenv("QDA_CACHE", str(cache))
    _cache_writer(monkeypatch)
    engine = reference.open_engine()
    opened = []
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: opened.append(k) or engine)
    settings = {"answers": 1, "max_tokens": 64, "k": 128, "tp": 1}
    one, two = tmp_path / "one", tmp_path / "two"
    targets, built = refcache.cached_many(
        str(model),
        [reference.Group(prompts, one, {"search96": ["a"]}), reference.Group(other, two)],
        settings,
    )
    assert opened and len(opened) == 1  # one engine load for both entries
    assert built == [True, True]
    assert len({str(t) for t in targets}) == 2  # each group has its own cache entry
    assert one.is_symlink() and two.is_symlink() and Path(one).resolve() == targets[0]
    opened.clear()
    again, built_again = refcache.cached_many(
        str(model),
        [reference.Group(prompts, one, {"search96": ["a"]}), reference.Group(other, two)],
        settings,
    )
    assert again == targets and built_again == [False, False] and opened == []


def test_a_cached_group_without_an_output_only_warms_the_cache(tmp_path, monkeypatch):
    """Prepare warms the holdout entry with no `out`; evaluate links it where it needs it."""
    from qda_kl import refcache

    model, prompts, _ = _install(monkeypatch, tmp_path)
    _cache_writer(monkeypatch)
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    targets, built = refcache.cached_many(
        str(model), [reference.Group(prompts, None)], {"answers": 1, "max_tokens": 64}
    )
    assert built == [True] and (targets[0] / "manifest.json").is_file()
    assert not (tmp_path / "any-out").exists()


def _rows(path, n):
    """Write ``n`` prompts named r0..r{n-1}."""
    ids = [f"r{i}" for i in range(n)]
    path.write_text("".join(json.dumps({"id": pid, "prompt": "hi"}) + "\n" for pid in ids))
    return ids


def test_a_row_past_the_context_is_dropped_and_recorded_not_fatal(tmp_path, monkeypatch):
    """One long holdout row no longer refuses the whole reading: it is skipped and the manifest says so."""
    model, prompts, _ = _install(monkeypatch, tmp_path)
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 16}))
    ids = _rows(prompts, 40)
    long_id = ids[7]
    monkeypatch.setattr(
        reference,
        "prompt_token_counts",
        lambda _m, items, **k: [(p.id, 15 if p.id == long_id else 10) for p in items],
    )

    got = reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=2)

    assert got["dropped_over_context"] == [long_id]
    assert (got["prompt_rows"], got["context"]) == (40, 16)
    assert got["dropped_short"] != [f"{long_id}#0"]  # it was never generated from: it does not fit
    line = reference.over_context_line(got)
    assert line == f"{(tmp_path / 'p.jsonl').name}: 39 of 40 rows kept, 1 over the model's 16-token context"


def test_too_many_rows_past_the_context_refuse_the_build(tmp_path, monkeypatch):
    """More than one row in twenty past the context is a corpus for another model: refused in one line."""
    model, prompts, captured = _install(monkeypatch, tmp_path)
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 16}))
    ids = _rows(prompts, 40)
    long_ids = set(ids[:3])  # 7.5% over the model's context
    monkeypatch.setattr(
        reference,
        "prompt_token_counts",
        lambda _m, items, **k: [(p.id, 15 if p.id in long_ids else 10) for p in items],
    )
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: pytest.fail("the engine loaded"))

    with pytest.raises(Refused, match=r"3 of 40 prompts \(7.5%\) leave no answer room in the model's 16"):
        reference.build(str(model), prompts, tmp_path / "ref", answers=1, max_tokens=2)
    assert captured == []


def test_the_line_names_no_drops_when_there_were_none():
    assert reference.over_context_line({"prompts": "p.jsonl"}) is None


SHA = "c" * 40


def _offline_pinned_cache(tmp_path, monkeypatch) -> None:
    """``hf download org/model --revision SHA`` offline: the snapshot and its generation_config, no refs/main."""
    from huggingface_hub import constants

    snap = tmp_path / "hub" / "models--org--model" / "snapshots" / SHA
    snap.mkdir(parents=True)
    (snap / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


def test_a_hub_id_at_a_revision_reaches_the_reads_the_engine_and_the_manifest(tmp_path, monkeypatch):
    """``org/model@SHA`` is ``org/model`` with ``revision=SHA`` everywhere, as ``qda build`` reads it."""
    _, prompts, captured = _install(monkeypatch, tmp_path)
    _offline_pinned_cache(tmp_path, monkeypatch)
    seen, opened, written = {}, [], []
    engine = reference.open_engine()

    def tokenizer(model, args, into):
        seen["tokenizer"] = (model, args.get("revision"))
        return dict(args), {}

    class Writer(_Writer):
        def __init__(self, out, **kwargs):
            super().__init__(out, **kwargs)
            written.append(kwargs["model"])

    monkeypatch.setattr(reference, "engine_tokenizer", tokenizer)
    monkeypatch.setattr(
        reference, "preload_refusal", lambda *a, **k: seen.update(preload=(a[0], k["revision"]))
    )
    monkeypatch.setattr(reference, "tokenizer_refusal", lambda *a, **k: None)
    monkeypatch.setattr(
        reference, "tokenizer_identity", lambda m, into, revision=None: {"dir": None, "revision": revision}
    )
    monkeypatch.setattr(reference, "ReferenceWriter", Writer)
    monkeypatch.setattr(
        reference, "open_engine", lambda m, **k: opened.append((m, k["settings"].args)) or engine
    )

    got = reference.build(f"org/model@{SHA}", prompts, tmp_path / "ref", answers=1, max_tokens=64)

    assert opened[0][0] == "org/model" and opened[0][1]["revision"] == SHA  # vllm.LLM(revision=SHA)
    assert seen == {"tokenizer": ("org/model", SHA), "preload": ("org/model", SHA)}
    assert captured[0]["temperature"] == 0.6  # the pinned snapshot's generation_config.json, read offline
    assert got["serve_args"]["revision"] == SHA
    assert written[0]["name"] == "model" and written[0]["revision"] == SHA
    assert written[0]["tokenizer"]["revision"] == SHA


def test_offline_a_bare_id_without_refs_main_is_refused_in_two_lines(tmp_path, monkeypatch, capsys):
    """The documented download pins a commit: the bare id names no snapshot offline, and `ref` says which."""
    from qda_kl import cli
    from qda_quant.exit import ENVIRONMENT

    _, prompts, _ = _install(monkeypatch, tmp_path)
    _offline_pinned_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(reference, "open_engine", lambda *a, **k: pytest.fail("the engine loaded"))

    rc = cli.main(["ref", "--model", "org/model", "--prompts", str(prompts), "--out", str(tmp_path / "ref")])

    assert rc == ENVIRONMENT
    assert capsys.readouterr().err.splitlines() == [
        f"error: org/model: the local Hugging Face cache has no `main` ref for it (it holds {SHA}) "
        "and the hub cannot be reached (HF_HUB_OFFLINE is set)",
        f"fix:   name the cached revision, org/model@{SHA}, or download it once with network access: "
        "hf download org/model",
    ]
