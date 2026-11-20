"""The `lm_eval` suite passes the config's settings to the harness and reports the mean over samples.

The harness is not a test dependency: the test installs a fake ``lm_eval`` module whose ``simple_evaluate``
records the settings it receives and returns one metric per task.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import statistics
import sys
import types
from pathlib import Path

import pytest
from qda_kl.eval import INSTALL_EVAL
from qda_kl.eval.lm_eval import LmEvalSuite, answer_region
from qda_kl.eval.tasks.utils import process_docs, strict_then_loose
from qda_quant.exit import ENVIRONMENT
from qda_quant.step import Refused

CALLS: list[dict] = []
MODEL_ARGS: list[dict] = []
TEMPLATES: list[dict] = []
ANSWERS: list[str] = []
SERVED: list[list[str]] = []
STOPPED: list[list[str]] = []
FINISHES: list[str] = []
#: Per-question scores for the fake harness; empty means every question scores ``seed / 100``.
SCORES: list[float] = []
#: The harness's filtered response per question, when a test needs what its filter made of the answer
#: (e.g. mmlu_pro's regex fallback ``[invalid]``); empty means the served answer itself.
FILTERED: list[str] = []
#: Group name -> leaf task names, for the group tests.
GROUPS: dict[str, list[str]] = {}
MANAGERS: list[_TaskManager] = []
UNTIL: list[list[str]] = []

#: The harness's EOS string and a task stop string, as the served model would see them.
EOS = "</s>"
TASK_STOPS = ["Question:", EOS]


class _Tokenizer:
    """Records the chat-template call the harness makes, so the thinking mode can be checked."""

    @staticmethod
    def apply_chat_template(messages, **kwargs):
        TEMPLATES.append({"messages": messages, **kwargs})
        return "<prompt>"


class _LM:
    """A model that serves whatever ``ANSWERS`` holds, cutting each at the stops it was sent (like vLLM)."""

    def __init__(self, **kwargs):
        MODEL_ARGS.append(kwargs)
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """The harness's own render: it always names ``enable_thinking``, which this suite sets."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
            **self.chat_template_args,
        )

    @staticmethod
    def modify_gen_kwargs(gen_kwargs, eos=None, default_max_gen_toks=256):
        until = [stop for stop in gen_kwargs.get("until", [])]
        if eos and eos not in until:
            until.append(eos)
        return {"seed": gen_kwargs.get("seed")}, until, gen_kwargs.get("max_gen_toks", default_max_gen_toks)

    def generate_until(self, requests):
        out = []
        for i, request in enumerate(requests):
            # Like vLLM: the engine applies the stops ``modify_gen_kwargs`` sends it, per request.
            _, until, _ = self.modify_gen_kwargs(dict(request.args[1]), eos=EOS)
            STOPPED.append(until)
            text = ANSWERS[i % len(ANSWERS)]
            cuts = [text.find(stop) for stop in until if stop and stop in text]
            out.append(text[: min(cuts)] if cuts else text)
        self._model_generate(requests=requests, generate=True, texts=out)
        return out

    @staticmethod
    def tok_encode(contexts):
        return [list(text) for text in contexts]

    @staticmethod
    def _model_generate(requests, generate, texts):
        order = sorted(range(len(requests)), key=lambda i: (-len(requests[i].args[0]), requests[i].args[0]))
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(
                        text=texts[i],
                        finish_reason=FINISHES[i],
                        token_ids=range(32768) if FINISHES[i] == "length" else [1],
                    )
                ]
            )
            for i in order
        ]


class _TaskManager:
    def __init__(self, include_path=None, metadata=None):
        self.include_path = include_path
        self.metadata = metadata
        MANAGERS.append(self)

    def load(self, tasks):
        leaves = []
        groups = {}
        for name in tasks:
            if name in GROUPS:
                groups[name] = object()
                leaves.extend(GROUPS[name])
            elif name not in leaves:
                leaves.append(name)
        return {"tasks": dict.fromkeys(leaves, object()), "groups": groups, "group_map": {}}


def _simple_evaluate(**kwargs):
    CALLS.append(kwargs)
    if any(task in GROUPS for task in kwargs["tasks"]):
        return _group_result(**kwargs)
    task = kwargs["tasks"][0]
    kwargs["model"].apply_chat_template([{"role": "user", "content": task}])
    requests = [
        types.SimpleNamespace(
            args=(q * (i + 1), {"until": UNTIL[i] if i < len(UNTIL) else TASK_STOPS, "max_gen_toks": 2048}),
            task_name=task,
            doc_id=i,
        )
        for i, q in enumerate(["q1", "q2"])
    ]
    SERVED.append(kwargs["model"].generate_until(requests))
    seed = kwargs["gen_kwargs"]["seed"]
    tasks = kwargs["tasks"]
    base = SCORES or [seed / 100] * 2

    def score(task: str, i: int) -> float:
        return base[i] + (0.01 if task == "t2" else 0.0)

    return {
        "results": {
            task: {
                "name": task,
                "alias": task,
                "score,none": statistics.fmean(score(task, i) for i in range(2)),
                "score_stderr,none": "N/A",
            }
            for task in tasks
        },
        "higher_is_better": {task: {"score": True} for task in tasks},
        "n-samples": {task: {"original": 5, "effective": 5} for task in tasks},
        "samples": {
            task: [
                {
                    "doc_id": i,
                    "doc_hash": q,
                    "doc": {"question_id": i},
                    "prompt_hash": hashlib.sha256(q.encode()).hexdigest(),
                    "filter": "none",
                    "score": score(task, i),
                    "filtered_resps": [FILTERED[i] if FILTERED else text],
                }
                for i, (q, text) in enumerate(zip(["q1", "q2"], SERVED[-1], strict=True))
            ]
            for task in tasks
        },
    }


def _group_result(**kwargs):
    """A harness result for a group: the group's aggregate plus one sample list per leaf, no group sample count."""
    seed = kwargs["gen_kwargs"]["seed"]
    leaves = [leaf for task in kwargs["tasks"] for leaf in GROUPS.get(task, [task])]
    results: dict[str, dict] = {}
    samples: dict[str, list[dict]] = {}
    for task in leaves:
        requests = [
            types.SimpleNamespace(
                args=(
                    q * (i + 1),
                    {"until": UNTIL[i] if i < len(UNTIL) else TASK_STOPS, "max_gen_toks": 2048},
                ),
                task_name=task,
                doc_id=i,
            )
            for i, q in enumerate(["q1", "q2"])
        ]
        served = kwargs["model"].generate_until(requests)
        score = seed / 100 + (0.01 if task == "t2" else 0.0)
        results[task] = {"name": task, "alias": task, "score,none": score, "score_stderr,none": "N/A"}
        samples[task] = [
            {
                "doc_id": i,
                "doc_hash": f"{task}:{q}",
                "prompt_hash": hashlib.sha256(f"{task}:{q}".encode()).hexdigest(),
                "filter": "none",
                "score": score,
                "filtered_resps": [text],
            }
            for i, (q, text) in enumerate(zip(["q1", "q2"], served, strict=True))
        ]
    group_subtasks = {task: GROUPS[task] for task in kwargs["tasks"] if task in GROUPS}
    for group, children in group_subtasks.items():
        results[group] = {
            "name": group,
            "alias": group,
            "score,none": statistics.fmean(results[child]["score,none"] for child in children),
            "score_stderr,none": "N/A",
        }
    return {
        "results": results,
        "higher_is_better": {task: {"score": True} for task in results},
        "n-samples": {task: {"original": 5, "effective": 5} for task in leaves},
        "samples": samples,
        "group_subtasks": group_subtasks,
    }


@pytest.fixture
def fake_lm_eval(monkeypatch):
    """Install a fake ``lm_eval`` with the parts the suite imports; clear the call log."""
    lm_eval = types.ModuleType("lm_eval")
    api = types.ModuleType("lm_eval.api")
    registry = types.ModuleType("lm_eval.api.registry")
    evaluator = types.ModuleType("lm_eval.evaluator")
    tasks = types.ModuleType("lm_eval.tasks")
    registry.get_model = lambda name: _LM
    evaluator.simple_evaluate = _simple_evaluate
    tasks.TaskManager = _TaskManager
    api.registry = registry
    lm_eval.api = api
    lm_eval.evaluator = evaluator
    lm_eval.tasks = tasks
    lm_eval.simple_evaluate = _simple_evaluate  # the harness re-exports it; the suite imports the definition
    for name, module in {
        "lm_eval": lm_eval,
        "lm_eval.api": api,
        "lm_eval.api.registry": registry,
        "lm_eval.evaluator": evaluator,
        "lm_eval.tasks": tasks,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    CALLS.clear()
    MODEL_ARGS.clear()
    TEMPLATES.clear()
    SERVED.clear()
    STOPPED.clear()
    GROUPS.clear()
    MANAGERS.clear()
    ANSWERS[:] = ["plain answer"]
    FINISHES[:] = ["stop", "stop"]
    SCORES.clear()
    FILTERED[:] = []
    UNTIL.clear()
    return lm_eval


def _checkpoint(tmp: Path, generation_config: dict | None) -> str:
    tmp.mkdir()
    if generation_config is not None:
        (tmp / "generation_config.json").write_text(json.dumps(generation_config))
    return str(tmp)


def test_a_missing_harness_is_the_machine(monkeypatch, tmp_path: Path) -> None:
    """lm-evaluation-harness is the `eval` extra: without it the suite refuses as the machine (3), not a crash."""
    for name in ("lm_eval", "lm_eval.api", "lm_eval.api.registry", "lm_eval.evaluator", "lm_eval.tasks"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(Refused) as refused:
        LmEvalSuite(tasks=["t1"]).run("ckpt", "ref", tmp_path)
    assert refused.value.code == ENVIRONMENT and refused.value.fix == INSTALL_EVAL


def test_settings_and_mean_over_repeats(fake_lm_eval, tmp_path: Path) -> None:
    ckpt = _checkpoint(
        tmp_path / "ckpt",
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.1,
            "min_p": 0.01,
        },
    )
    suite = LmEvalSuite(
        tasks=["t1", "t2"],
        repeats={"t1": 2, "t2": 1},
        max_tokens={"t1": 81920, "t2": 32768},
        limit=5,
    )

    got = suite.run(ckpt, "reference", tmp_path / "out")

    # Sample j is one harness run per answer cap, with the checkpoint's sampling and seed 17 + j.
    assert [(c["gen_kwargs"]["seed"], c["gen_kwargs"]["max_gen_toks"]) for c in CALLS] == [
        (17, 81920),
        (17, 32768),
        (18, 81920),
    ]
    assert [c["tasks"] for c in CALLS] == [["t1"], ["t2"], ["t1"]]
    for call in CALLS:
        # Every field the shared resolver keeps reaches the harness, not just the first three.
        assert call["gen_kwargs"] == {
            "temperature": 0.6,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.1,
            "min_p": 0.01,
            "max_gen_toks": call["gen_kwargs"]["max_gen_toks"],
            "seed": call["gen_kwargs"]["seed"],
        }
        assert call["limit"] == 5
        assert call["batch_size"] == "auto"
        assert call["apply_chat_template"] is True
    assert [
        {
            "pretrained": ckpt,
            "tensor_parallel_size": 1,
            "compilation_config": {"inductor_compile_config": {"deterministic": True}},
            "batch_size": "auto",
            "seed": 17,
        }
    ] == MODEL_ARGS

    assert got.metrics == pytest.approx({"t1": 0.175, "t2": 0.18})
    assert got.higher_is_better == {"t1": True, "t2": True}
    assert got.detail["t1"]["samples"] == 10
    assert got.detail["t2"]["samples"] == 5
    assert got.detail["t1"]["per_repeat"] == [0.17, 0.18]
    assert got.detail["t1"]["per_question"] == {"0:q1": [0.17, 0.18], "1:q2": [0.17, 0.18]}
    assert got.detail["t1"]["prompt_hashes"] == {
        "0:q1": hashlib.sha256(b"q1").hexdigest(),
        "1:q2": hashlib.sha256(b"q2").hexdigest(),
    }
    assert got.detail["t1"]["stderr"] == pytest.approx(statistics.stdev([0.17, 0.18]) / 2**0.5)
    assert got.detail["t2"]["stderr"] is None
    assert got.detail["sampling"]["presence_penalty"] == 1.5


def test_caps_and_raw_thinking_keep_the_question_association(fake_lm_eval, tmp_path: Path) -> None:
    """Length-sorted completions retain their ids and raw thinking; answered caps are not failures."""
    ANSWERS[:] = ["<think>unfinished", "<think>r</think>answer"]
    FINISHES[:] = ["length", "length"]
    suite = LmEvalSuite(tasks=["t1"], repeats=2, think_end_token="</think>")
    first = suite.run(_checkpoint(tmp_path / "first", {"temperature": 0.6}), "ref", tmp_path)
    second = suite.run(_checkpoint(tmp_path / "second", {"temperature": 0.6}), "ref", tmp_path)
    assert first.detail["t1"]["per_question"] == second.detail["t1"]["per_question"]
    assert first.detail["t1"]["cap_hits"] == [1, 1]
    with gzip.open(tmp_path / first.detail["t1"]["raw_generations"][0], "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert [(r["question_id"], r["raw"][0]["text"]) for r in records] == [
        ("0:q1", "<think>unfinished"),
        ("1:q2", "<think>r</think>answer"),
    ]
    assert first.detail["t1"]["raw_generations"] != second.detail["t1"]["raw_generations"]


def test_a_filter_fallback_does_not_hide_a_capped_generation(fake_lm_eval, tmp_path: Path) -> None:
    """mmlu_pro's filter turns an unclosed answer into ``[invalid]``; the cap count follows the raw finish."""
    ANSWERS[:] = ["<think>unfinished", "<think>r</think>The answer is (B)"]
    FINISHES[:] = ["length", "length"]
    FILTERED[:] = ["[invalid]", "The answer is (B)"]
    suite = LmEvalSuite(tasks=["t1"], repeats=2, max_tokens={"t1": 1}, think_end_token="</think>")

    got = suite.run(_checkpoint(tmp_path / "ckpt", {"temperature": 0.0}), "ref", tmp_path)

    assert got.detail["t1"]["cap_hits"] == [1, 1]


def test_tp_is_the_layout_and_reaches_the_harness(fake_lm_eval, tmp_path: Path) -> None:
    """The suite's ``tp`` splits the engine, and ``cards()`` -- the job's card count -- follows it."""
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.0})
    suite = LmEvalSuite(tasks=["t1"], tp=2)

    suite.run(ckpt, "reference", tmp_path / "out")

    assert MODEL_ARGS[0]["tensor_parallel_size"] == 2
    assert suite.cards() == 2
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        LmEvalSuite(tasks=["t1"], serve_args={"tensor_parallel_size": 2})


def test_eval_starts_every_task_with_the_users_cache_directory(fake_lm_eval, tmp_path, monkeypatch):
    """A cache variable the user set reaches every harness engine as it is, and is what the run records."""
    import os

    import yaml
    from qda_kl.eval.config import EvalConfig
    from qda_kl.eval.run import run

    requested = tmp_path / "shared"
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(requested))
    seen = []

    def model(**kwargs):
        seen.append(os.environ["FLASHINFER_WORKSPACE_BASE"])
        return _LM(**kwargs)

    fake_lm_eval.api.registry.get_model = lambda name: model
    checkpoint = _checkpoint(tmp_path / "ckpt", {"temperature": 0.0})
    cfg = EvalConfig.from_yaml(
        yaml.safe_dump(
            {
                "reference": checkpoint,
                "checkpoints": {"same": checkpoint},
                "out": str(tmp_path / "out"),
                "suites": [{"lm_eval": {"tasks": ["t1", "t2"]}}],
            }
        )
    )
    assert run(cfg, log=lambda _: None) == 0
    assert seen == [str(requested), str(requested)]
    resolved = yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())
    assert resolved["suites"][0]["lm_eval"]["serve_env"]["FLASHINFER_WORKSPACE_BASE"] == str(requested)
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(requested)


def test_config_sampling_merges_over_the_checkpoint(fake_lm_eval, tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})

    # The explicit setting wins field by field: the file's top_p/top_k survive, top_k -1 is legal.
    suite = LmEvalSuite(tasks=["t1"], sampling={"temperature": 0.3, "top_k": -1}, metric={"t1": "score"})
    assert suite.run(ckpt, "reference", tmp_path / "out").metrics == {"t1": 0.17}
    assert CALLS[0]["gen_kwargs"] == {
        "temperature": 0.3,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": -1,
        "max_gen_toks": 32768,
        "seed": 17,
    }


def test_sampling_needs_the_config_or_generation_config(fake_lm_eval, tmp_path: Path) -> None:
    bare = _checkpoint(tmp_path / "bare", None)
    with pytest.raises(ValueError, match="generation_config.json"):
        LmEvalSuite(tasks=["t1"]).run(bare, "reference", tmp_path / "out")
    assert CALLS == []  # refused before the engine started

    # A zero temperature is greedy, and it needs no file.
    suite = LmEvalSuite(tasks=["t1"], sampling={"temperature": 0.0}, metric={"t1": "score"})
    assert suite.run(bare, "reference", tmp_path / "out").metrics == {"t1": 0.17}
    assert CALLS[0]["gen_kwargs"] == {
        "temperature": 0.0,
        "do_sample": False,
        "max_gen_toks": 32768,
        "seed": 17,
    }


def test_answer_region_takes_the_first_close_tag() -> None:
    # Unfinished thinking has no answer, and a later close tag does not move the region.
    assert answer_region("<think>The answer is (B)", "</think>") == ""
    assert (
        answer_region("<think>reasoning</think>The answer is (B)</think>(C)", "</think>")
        == "The answer is (B)</think>(C)"
    )
    assert answer_region("<think>r</think>  The answer is (B)", "</think>") == "The answer is (B)"
    assert answer_region("no tag at all", None) == "no tag at all"
    # Task stops apply to the region, not to the thinking before it.
    assert (
        answer_region(
            "<think>Question: hmm?</think>The answer is (B)\nQuestion: done", "</think>", ["Question:"]
        )
        == "The answer is (B)\n"
    )
    # With gold B, the unfinished sample scores nothing and the two-tag sample extracts (B), not (C).
    assert strict_then_loose([answer_region("<think>The answer is (B)", "</think>")], [{}]) == [""]
    assert strict_then_loose(
        [answer_region("<think>r</think>The answer is (B)</think>(C)", "</think>")], [{}]
    ) == ["(B)"]


def test_thinking_answers_are_cut_before_the_task_scores(fake_lm_eval, tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})
    ANSWERS[:] = ["<think>r</think>The answer is (B)</think>(C)", "<think>The answer is (B)"]

    LmEvalSuite(tasks=["t1"], think_end_token="</think>").run(ckpt, "reference", tmp_path / "out")

    assert SERVED == [["The answer is (B)</think>(C)", ""]]
    assert "think_end_token" not in MODEL_ARGS[0]  # the harness's own (last-tag) split is not used


def test_the_checkpoint_sizes_the_synthetic_tasks(fake_lm_eval, tmp_path: Path) -> None:
    """The harness gets the served checkpoint's tokenizer, which the synthetic long-context tasks size data with."""
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})

    LmEvalSuite(tasks=["t1"]).run(ckpt, "reference", tmp_path / "out")

    assert MANAGERS[-1].metadata == {"tokenizer": ckpt}


def test_a_group_headline_is_its_aggregate_and_its_leaves_are_the_questions(
    fake_lm_eval, tmp_path: Path
) -> None:
    """The harness's ``mmlu_pro`` is a group: its weighted mean headlines, its leaf samples give the ids."""
    GROUPS["g1"] = ["t1", "t2"]
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})

    got = LmEvalSuite(tasks=["g1"]).run(ckpt, "reference", tmp_path / "out")

    assert got.metrics == {"g1": pytest.approx(0.175)}
    assert got.detail["g1"]["samples"] == 10  # the harness reports no group count; the leaves' add up
    assert set(got.detail["g1"]["per_question"]) == {"0:t1:q1", "1:t1:q2", "0:t2:q1", "1:t2:q2"}
    assert got.detail["g1"]["cap_hits"] == [0]


def test_clean_subset_reports_both_numbers_from_one_run(fake_lm_eval, tmp_path: Path) -> None:
    """Full and clean means share one run; the dataset-id mapping is recorded and an unscored id refuses."""
    SCORES[:] = [0.9, 0.1]
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.0})
    ids = tmp_path / "private.json"
    ids.write_text(json.dumps({"clean_subset_excludes_question_ids": [0]}))
    suite = LmEvalSuite(tasks=["t1"], clean_subset=str(ids))

    got = suite.run(ckpt, "reference", tmp_path / "out")

    assert got.metrics == {"t1": pytest.approx(0.5), "t1_clean": pytest.approx(0.1)}
    assert got.higher_is_better == {"t1": True, "t1_clean": True}
    assert got.detail["t1"]["clean_subset"] == {
        "score": 0.1,
        "questions": 1,
        "excluded": 1,
        "per_repeat": [0.1],
        "question_ids": {"0:q1": "0", "1:q2": "1"},
    }
    first = suite.inputs()["clean_subset_sha256"]
    ids.write_text(json.dumps({"clean_subset_excludes_question_ids": [0, 99]}))
    assert suite.inputs()["clean_subset_sha256"] != first  # an edited list reruns the job
    with pytest.raises(ValueError, match="1 excluded question ids"):
        suite.run(ckpt, "reference", tmp_path / "out2")


def test_the_suite_names_the_thinking_mode_in_every_prompt(fake_lm_eval, tmp_path: Path) -> None:
    """The harness's unset mode (None) renders another prompt; the suite names the model's default instead."""
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})
    LmEvalSuite(tasks=["t1"]).run(ckpt, "reference", tmp_path / "out")
    assert TEMPLATES[-1]["enable_thinking"] is True

    LmEvalSuite(tasks=["t1"], enable_thinking=False).run(ckpt, "reference", tmp_path / "out")
    assert TEMPLATES[-1]["enable_thinking"] is False

    kwargs = {"reasoning_effort": "medium"}
    suite = LmEvalSuite(tasks=["t1"], chat_template_kwargs=kwargs)
    got = suite.run(ckpt, "reference", tmp_path / "out")
    assert TEMPLATES[-1]["reasoning_effort"] == "medium"
    assert TEMPLATES[-1]["enable_thinking"] is True
    assert got.detail["chat_template_kwargs"] == kwargs
    assert suite.model_dump()["chat_template_kwargs"] == kwargs
    with pytest.raises(ValueError, match="set enable_thinking on the suite"):
        LmEvalSuite(tasks=["t1"], chat_template_kwargs={"enable_thinking": False})


def test_parts_are_one_task_each(fake_lm_eval, tmp_path: Path) -> None:
    """`qda eval` runs one job per task: a part carries its task's settings and runs only that task."""
    suite = LmEvalSuite(
        tasks=["t1", "t2"],
        repeats={"t1": 2, "t2": 1},
        max_tokens={"t1": 81920, "t2": 32768},
        metric={"t2": "score"},
    )
    parts = suite.parts()

    assert [p.part_label() for p in parts] == ["t1", "t2"]
    assert [(p.tasks, p.repeats, p.max_tokens, p.metric) for p in parts] == [
        (["t1"], {"t1": 2}, {"t1": 81920}, {}),
        (["t2"], {"t2": 1}, {"t2": 32768}, {"t2": "score"}),
    ]
    assert [LmEvalSuite(**p.model_dump()) for p in parts] == parts  # a part is itself a valid suite

    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})
    got = parts[1].run(ckpt, "reference", tmp_path / "out")
    assert got.metrics == {"t2": pytest.approx(0.18)}
    assert [call["tasks"] for call in CALLS] == [["t2"]]


def test_a_task_stop_inside_the_thinking_does_not_end_the_answer(fake_lm_eval, tmp_path: Path) -> None:
    # The reviewer's case: "Question:" occurs while thinking, the real answer follows the close tag.
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})
    thinking = "<think>Restating it. Question: which is right? I pick B.</think>The answer is (B)"
    ANSWERS[:] = [thinking]

    LmEvalSuite(tasks=["t1"], think_end_token="</think>").run(ckpt, "reference", tmp_path / "out")

    assert STOPPED == [[EOS], [EOS]]  # the engine stops on EOS only while thinking
    assert SERVED == [["The answer is (B)", "The answer is (B)"]]
    assert strict_then_loose([SERVED[0][0]], [{}]) == ["(B)"]

    # Without thinking, the task's stops are the engine's, and the same text is cut inside the thinking.
    STOPPED.clear()
    ANSWERS[:] = [thinking]
    LmEvalSuite(tasks=["t1"]).run(ckpt, "reference", tmp_path / "out")
    assert STOPPED == [TASK_STOPS, TASK_STOPS]
    assert strict_then_loose([SERVED[-1][0]], [{}]) == [""]


def test_a_batch_with_two_tasks_keeps_each_requests_stops(fake_lm_eval, tmp_path: Path) -> None:
    """The stops are per request: the last task's stops must not cut the other task's answer."""
    ckpt = _checkpoint(tmp_path / "ckpt", {"temperature": 0.6})
    ANSWERS[:] = ["r</think>A Question: tail", "r</think>B STOP tail"]
    UNTIL[:] = [["Question:"], ["STOP"]]

    LmEvalSuite(tasks=["t1"], think_end_token="</think>").run(ckpt, "reference", tmp_path / "out")

    assert SERVED[-1] == ["A ", "B "]


def test_gpqa_option_order_does_not_depend_on_ambient_random_state() -> None:
    """The same question shuffles the same way in every process, so its prompt hash is stable."""
    keys = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")
    docs = [{"Question": f"question {i}", **{key: f"{key} {i}" for key in keys}} for i in range(4)]

    class _Dataset:
        def map(self, function):
            return [function(doc) for doc in docs]

    def order(processed):
        return [(d["choice1"], d["choice2"], d["choice3"], d["choice4"], d["answer"]) for d in processed]

    random.seed(0)
    first = order(process_docs(_Dataset()))
    random.seed(12345)
    random.random()

    assert order(process_docs(_Dataset())) == first
