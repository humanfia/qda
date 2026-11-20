"""Official benchmarks through lm-evaluation-harness on vLLM: the ``lm_eval`` eval suite.

    suites:
      - lm_eval: {tasks: [gpqa_diamond, ifeval], repeats: {gpqa_diamond: 16, ifeval: 16}}

The suite serves ``checkpoint`` on the kernels ``build.json`` plans (:func:`qda_kl.serve.settings_for`), runs each
task through ``lm_eval.simple_evaluate`` on a vLLM engine, and returns the headline metric of each task with its
stderr and sample count.

Each task also reports ``per_question`` (document id plus hash -> scores in seed order), ``prompt_hashes`` (the
same question key -> the hash of the exact prompt the model received, so two rows can be proven to have asked
the same question) and ``cap_hits`` (generations reaching the configured cap with an empty answer region, per
seed). Raw completions and harness sample records are retained as gzip JSONL under the run's work directory for
regrading.

Sampling is the shared resolver's (:func:`qda_quant.calib.selfgen.resolve_sampling`) merge of the config's
``sampling`` over the checkpoint's ``generation_config.json``; a checkpoint with neither is refused. Repeats follow
the paper's protocol: sample ``j`` is one harness run with seed ``seed + j``, and a task's score is the mean.

A configured ``think_end_token`` cuts every answer to the paper's answer region (:func:`answer_region`) before any
task extracts it; LiveCodeBench uses the same function and setting. ``enable_thinking`` names the chat template's
thinking mode (default: on, as the benchmark models' cards show). The harness always passes
``enable_thinking=None`` -- which renders a prompt without the model's default reasoning instruction -- so the
suite sets the mode on the harness instance instead. Tasks the harness does not ship (the paper's HMMT Feb 2026
and its GPQA extraction) come from the suite's ``datasets`` (:mod:`qda_kl.datasets`), found through the harness's
``TaskManager.include_path``.

A ``tasks`` entry may name a harness group (e.g. ``mmlu_pro``): its headline metric is the group's own weighted
aggregate, and its per-question scores and cap counts are collected from the group's leaf tasks. The checkpoint is
passed to the task manager as the ``tokenizer`` metadata, which the synthetic long-context tasks use to size their
data with the model's own tokenizer.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_serializer, field_validator, model_validator
from qda_quant.calib import sha256_file
from qda_quant.calib.selfgen import resolve_sampling
from qda_quant.exit import ENVIRONMENT
from qda_quant.step import Refused

from qda_kl.datasets import TASKS, Dataset, content_sha256, parse_dataset
from qda_kl.eval import INSTALL_EVAL
from qda_kl.eval.samples import write_samples
from qda_kl.eval.suites import Layout, SuiteResult
from qda_kl.serve import check_tp, engine_env, llm, settings_for


def answer_region(text: str, close: str | None, stops: Sequence[str] = ()) -> str:
    """The part of ``text`` a task may score: after the first ``close`` tag and before the first stop string.

    The paper scores the text after the first closing thinking tag, and an answer that never closes its thinking
    scores nothing; a missing tag therefore leaves an empty answer. Task stop strings apply to that region only,
    so a quoted "Question:" inside the thinking does not end the answer. With ``close`` unset the text is whole.

    Args:
        text: the model's generated answer.
        close: the model's closing thinking tag, when it thinks first.
        stops: the task's stop strings.

    Returns:
        The answer region.
    """
    if close is not None:
        _, found, rest = text.partition(close)
        text = rest.lstrip() if found else ""
    for stop in stops:
        if stop:
            text = text.split(stop)[0]
    return text


def task_stops(request: Any) -> tuple[str, ...]:
    """The stop strings that request declared, so each answer is cut with its own task's stops."""
    gen_kwargs = request.args[1] if len(request.args) > 1 else {}
    until = gen_kwargs.get("until") or ()
    if isinstance(until, str):
        until = (until,)
    return tuple(stop for stop in until if stop)


def _answer_region_model(lm: Any, close: str) -> Any:
    """``lm`` itself, whose answers are cut to the paper's answer region before any task extracts them.

    The harness's own option splits at the last tag and keeps the whole text when there is none;
    :func:`answer_region` needs the first tag, and an unfinished sample must score nothing. While thinking, only
    the EOS stop reaches vLLM (a task stop quoted in the thinking would truncate the answer); the task stops are
    applied after the region is cut, per request, because a batch may mix tasks with different stops. The class
    hierarchy, and the harness's ``isinstance`` check, are kept.

    Args:
        lm: the harness's served model instance.
        close: the model's closing thinking tag.

    Returns:
        The same object, with ``generate_until`` returning answer regions.
    """
    base = lm.__class__

    class AnswerRegion(base):  # type: ignore[misc]
        """The served model; every generated answer is cut to its answer region."""

        def modify_gen_kwargs(  # type: ignore[override]
            self, gen_kwargs: dict[str, Any], eos: Any = None, default_max_gen_toks: int = 256
        ) -> tuple[dict[str, Any], list[str], int]:
            """Keep the EOS stop for the engine; hand the task's stops to the region cut instead."""
            kwargs, until, cap = base.modify_gen_kwargs(
                gen_kwargs, eos=eos, default_max_gen_toks=default_max_gen_toks
            )
            eoses = [eos] if isinstance(eos, str) else list(eos or [])
            return kwargs, [stop for stop in until if stop in eoses], cap

        def generate_until(self, requests: list[Any], *args: Any, **kwargs: Any) -> list[str]:
            stops = [task_stops(request) for request in requests]
            return [
                answer_region(text, close, stop)
                for text, stop in zip(super().generate_until(requests, *args, **kwargs), stops, strict=True)
            ]

    lm.__class__ = AnswerRegion
    return lm


def _sample_model(lm: Any) -> Any:
    """Retain raw vLLM completions, undoing the harness's length sort to associate them with documents."""
    base = lm.__class__

    class Samples(base):  # type: ignore[misc]
        """The served model, with raw completions indexed by task and document id."""

        def _model_generate(self, *args: Any, **kwargs: Any) -> Any:
            outputs = super()._model_generate(*args, **kwargs)
            if kwargs.get("generate"):
                self._qda_outputs.extend(outputs)
            return outputs

        def generate_until(self, requests: list[Any], **kwargs: Any) -> list[str]:
            self._qda_outputs = []
            result = super().generate_until(requests, **kwargs)
            contexts = [req.args[0] for req in requests]
            tokens = self.tok_encode(contexts)
            # lm-eval 0.4.13's generate_until sorts by (-token count, context), across its batches.
            order = sorted(range(len(requests)), key=lambda i: (-len(tokens[i]), contexts[i]))
            for i, output in zip(order, self._qda_outputs, strict=True):
                req = requests[i]
                completion = output.outputs[0]
                self._qda_samples.setdefault((req.task_name, str(req.doc_id)), []).append(
                    {
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "tokens": len(completion.token_ids),
                    }
                )
            return result

    lm.__class__ = Samples
    lm._qda_samples = {}
    return lm


def _excluded_question_ids(path: str) -> set[str]:
    """The dataset ``question_id``s a clean-subset file excludes: a list or ``clean_subset_excludes_question_ids``.

    Raises:
        ValueError: the file is missing or is not JSON of one of those shapes.
    """
    try:
        data = json.loads(Path(path).read_text())
    except OSError as why:
        raise ValueError(f"clean_subset: {path}: {why.strerror or why}") from None
    except json.JSONDecodeError as why:
        raise ValueError(f"clean_subset: {path}: not valid JSON: {why}") from None
    if isinstance(data, dict):
        data = data.get("clean_subset_excludes_question_ids")
    if not isinstance(data, list) or not all(
        isinstance(qid, (int, str)) and not isinstance(qid, bool) for qid in data
    ):
        raise ValueError(
            f"clean_subset: {path}: expected a list of question ids, or a mapping with "
            "clean_subset_excludes_question_ids"
        )
    return {str(qid) for qid in data}


@dataclass
class _TaskScores:
    """One task's numbers over its repeats, as `LmEvalSuite.run` reads them from each harness result."""

    metric: str = ""
    higher: bool = True
    samples: int = 0
    per_repeat: list[float] = field(default_factory=list)
    questions: dict[str, list[float]] = field(default_factory=dict)
    question_ids: dict[str, str] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    cap_hits: list[int] = field(default_factory=list)
    raw_files: list[str] = field(default_factory=list)


class LmEvalSuite(Layout):
    """The paper's official benchmarks, measured by lm-evaluation-harness on vLLM.

    Attributes:
        tasks: harness task or group names, e.g. ``aime25``, ``gpqa_diamond`` or the group ``mmlu_pro``.
        datasets: the directories of task YAMLs the harness reads besides its own, each a path or one
            ``{key: settings}`` from ``qda.datasets``; default: the tasks this package ships. Editing one reruns
            the suite.
        repeats: samples per problem, one per task or for every task (default 1).
        max_tokens: cap of one answer, one per task or for every task (default 32768).
        sampling: temperature/top_p/top_k/min_p/presence_penalty/repetition_penalty; merged field by field
            over the checkpoint's ``generation_config.json`` by the shared resolver, and a checkpoint with
            neither is refused.
        batch_size: harness batch size, or ``"auto"``.
        tp: tensor-parallel size of the engine (the layout's ``tp``); a job needs ``tp`` times its
            ``serve_args``' ``pipeline_parallel_size`` cards.
        serve_env: environment variables for the engine, over ``build.json``'s ``serve.env`` (:class:`Layout`).
        serve_args: vLLM keyword arguments over ``build.json``'s ``serve.args`` (:func:`settings_for`); they
            carry the pipeline stages and expert parallelism, and may not name ``tensor_parallel_size``.
        nccl_nvls: let the engine's ranks join NCCL's NVLS collectives (:class:`Layout`).
        limit: problems per task, for a smoke run; None evaluates every problem.
        metric: headline metric per task when the task reports more than one, e.g.
            ``{ifeval: prompt_level_strict_acc}``.
        clean_subset: path to a private JSON file of dataset ``question_id``s to leave out of a second headline
            metric, ``<task>_clean`` (a list, or a mapping with ``clean_subset_excludes_question_ids``). Each
            question's dataset id is recorded beside its ``per_question`` key, and an excluded id the run did
            not score refuses. The file's digest joins the job's ``config_digest``; neither the ids nor this
            path belong in a committed config.
        apply_chat_template: apply the model's chat template to the prompt (the paper's models are chat models).
        enable_thinking: the chat template's thinking mode, passed to every prompt; the benchmark models'
            cards show thinking on by default, and the harness's own unset value would render a different
            prompt (it drops the default reasoning instruction). Set false for a non-thinking model.
        chat_template_kwargs: extra template arguments, e.g. ``{"reasoning_effort": "medium"}``;
            thinking mode is selected by ``enable_thinking``.
        think_end_token: the model's closing thinking tag; a task scores only the text after its first
            occurrence, and an answer that never closes its thinking scores nothing.
        seed: seed of sample 0; sample ``j`` uses ``seed + j`` (the paper's 17 + j).
        server: base URL of a running vLLM completions server to generate on, instead of an engine in this
            process (:mod:`qda_kl.eval.served`). The URL is transport and stays out of the job's
            ``config_digest``, but the engine path does not: a served job's digest names the engine, so a
            served run neither invalidates nor silently reuses the in-process result at the same path. The
            result's ``detail`` records the server's own vLLM version and model, read from the server.
    """

    name: ClassVar[str] = "lm_eval"

    tasks: list[str] = Field(min_length=1)
    datasets: list[Dataset] = Field(
        default_factory=lambda: [parse_dataset({"lm_eval_builtin": {}}, "datasets")]
    )
    repeats: int | dict[str, int] = 1
    max_tokens: int | dict[str, int] = 32768
    sampling: dict[str, Any] | None = None
    batch_size: int | str = "auto"
    limit: int | None = Field(default=None, ge=1)
    metric: dict[str, str] = Field(default_factory=dict)
    clean_subset: str | None = None
    apply_chat_template: bool = True
    enable_thinking: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    think_end_token: str | None = None
    seed: int = 17
    server: str | None = Field(default=None, exclude=True)

    @field_validator("datasets", mode="before")
    @classmethod
    def _named(cls, v: Any) -> Any:
        if not isinstance(v, list):
            raise Refused(f"expected a list of task datasets; got {v!r}")
        return [parse_dataset(d, f"datasets.{i}", TASKS) for i, d in enumerate(v)]

    @field_serializer("datasets")
    def _keyed(self, datasets: list[Dataset]) -> list[dict[str, Any]]:
        return [d.to_dict() for d in datasets]

    @model_validator(mode="after")
    def _check(self) -> LmEvalSuite:
        """Refuse duplicate tasks, a per-task setting that misses or invents a task, and non-positive counts."""
        if "enable_thinking" in self.chat_template_kwargs:
            raise ValueError("chat_template_kwargs.enable_thinking: set enable_thinking on the suite instead")
        if len(set(self.tasks)) != len(self.tasks):
            raise Refused(f"tasks {self.tasks}: each task once")
        for name in ("repeats", "max_tokens"):
            value = getattr(self, name)
            if isinstance(value, dict):
                bad = sorted(set(value) - set(self.tasks))
                missing = sorted(set(self.tasks) - set(value))
                if bad or missing:
                    raise Refused(f"{name}: unknown {bad}, missing {missing}; tasks are {self.tasks}")
                if any(v < 1 for v in value.values()):
                    raise Refused(f"{name}: every task needs a positive value, got {value}")
            elif value < 1:
                raise Refused(f"{name}: {value} is not positive")
        bad = sorted(set(self.metric) - set(self.tasks))
        if bad:
            raise Refused(f"metric: unknown tasks {bad}; tasks are {self.tasks}")
        return self

    def run(self, checkpoint: str, reference: str, _work: Path) -> SuiteResult:
        """Run every task on ``checkpoint`` and return each task's mean metric over its samples.

        Raises:
            Refused: lm-evaluation-harness is not installed (``ENVIRONMENT``: install it), or the
                checkpoint's sampling is undetermined, a task name, a per-task setting or the headline
                metric is wrong (``USAGE``).
            RuntimeError: the harness refused the run.
        """
        try:
            from lm_eval.api.registry import get_model
            from lm_eval.evaluator import simple_evaluate
            from lm_eval.tasks import TaskManager
        except ModuleNotFoundError as why:  # the harness is the ``eval`` extra
            raise Refused(
                "lm_eval: lm-evaluation-harness is not installed",
                INSTALL_EVAL,
                code=ENVIRONMENT,
            ) from why

        gen = self._gen_kwargs(checkpoint)
        # Task loading runs before the harness seeds itself (inside simple_evaluate); seed it here so a task's
        # document processing cannot depend on ambient state.
        random.seed(self.seed)
        include = [str(dataset.local()) for dataset in self.datasets]
        # The checkpoint's tokenizer sizes the synthetic long-context tasks; other tasks ignore the metadata.
        task_manager = TaskManager(include_path=include, metadata={"tokenizer": checkpoint})
        loaded = task_manager.load(list(self.tasks))
        missing = [t for t in self.tasks if t not in loaded["tasks"] and t not in loaded["groups"]]
        if missing:
            raise Refused(f"tasks {missing}: not a lm-eval task or group (read the harness's and {include})")

        repeats = self._per_task(self.repeats)
        caps = self._per_task(self.max_tokens)
        excluded = _excluded_question_ids(self.clean_subset) if self.clean_subset else set()
        scores = {t: _TaskScores() for t in self.tasks}
        higher: dict[str, bool] = {}

        settings = settings_for(checkpoint, self.serve_env, self.serve_args, self.nccl_nvls)
        model_args: dict[str, Any] = {
            "pretrained": checkpoint,
            "tensor_parallel_size": self.tp,
            **settings.args,
        }
        check_tp(model_args, self.tp)
        model_args.setdefault("batch_size", self.batch_size)
        model_args.setdefault("seed", self.seed)

        served = self.server is not None
        identity: dict[str, Any] | None = None
        if served:
            from qda_kl.eval.served import ServedModel, check_server_identity, server_identity

            if "max_model_len" not in model_args:
                raise Refused("server: serve_args must name max_model_len (the client fits prompts to it)")
            # Read the engine that will generate, and refuse a server that is not the checkpoint's: the
            # request's settings say what was asked for, not what answered. No engine in this process: the
            # prompts, the sampling and the answer cuts are the in-process path's, and only generation
            # happens on the server, which every cell of the card shares.
            identity = server_identity(str(self.server))
            check_server_identity(identity, checkpoint, settings.vllm)
            lm: Any = ServedModel(
                base_url=str(self.server),
                checkpoint=checkpoint,
                max_model_len=int(model_args["max_model_len"]),
                seed=self.seed,
                trust_remote_code=bool(model_args.get("trust_remote_code", False)),
                # the same override the server was started with: the client is what encodes the prompts
                tokenizer=(str(t) if (t := model_args.get("tokenizer")) else None),
            )
        else:
            try:
                # through start_engine, so ``settings.env`` holds the caches the engine really used,
                # which the block below keeps for the whole evaluation
                lm = llm(settings, model_args, constructor=lambda **args: get_model("vllm")(**args))
            except ModuleNotFoundError as why:
                raise RuntimeError(f"lm_eval: {why.name} is not installed; {INSTALL_EVAL}") from why
        with contextlib.nullcontext() if served else engine_env(settings):
            # The harness hands the chat template enable_thinking=None unless configured, and None renders a
            # prompt without the model's default reasoning instruction. Name the mode on the instance: the
            # constructor argument would also require the harness's think_end_token, whose last-tag cut would
            # run before this suite's first-tag cut.
            lm.enable_thinking = self.enable_thinking
            lm.chat_template_args = dict(self.chat_template_kwargs)
            if served:
                lm.think_end_token = self.think_end_token
                samples: dict[tuple[str, str], list[dict[str, Any]]] = lm.qda_samples
            else:
                if self.think_end_token:
                    lm = _answer_region_model(lm, self.think_end_token)
                lm = _sample_model(lm)
                samples = lm._qda_samples
            for j in range(max(repeats.values())):
                for cap, tasks in self._runs(repeats, caps, j).items():
                    samples.clear()
                    result = simple_evaluate(
                        model=lm,
                        tasks=tasks,
                        batch_size=self.batch_size,
                        limit=self.limit,
                        bootstrap_iters=0,
                        log_samples=True,
                        apply_chat_template=self.apply_chat_template,
                        gen_kwargs={**gen, "max_gen_toks": cap, "seed": self.seed + j},
                        task_manager=task_manager,
                        random_seed=self.seed,
                    )
                    if result is None:
                        raise RuntimeError(f"lm_eval: task(s) {tasks} produced no result")
                    for task in tasks:
                        records = self._score(task, result, scores[task], samples, cap, j)
                        higher[task] = scores[task].higher
                        raw_path = write_samples(
                            _work,
                            checkpoint,
                            self.name,
                            {**self.model_dump(mode="json"), "task": task, "draw": j},
                            records,
                        )
                        scores[task].raw_files.append(str(raw_path.relative_to(_work)))

        metrics: dict[str, float] = {}
        detail: dict[str, Any] = {
            "reference": reference,
            "vllm": settings.vllm,
            "serve": settings.model_dump(mode="json"),
            "sampling": gen,
            "chat_template_kwargs": self.chat_template_kwargs,
            "seed": self.seed,
            "engine": "server" if self.server else "in-process",
        }
        if identity is not None:
            # The engine that generated, read from the server: the client process need not have vLLM at all,
            # so the record must not take its version from the request's settings or from this process.
            detail["server"] = identity
            detail["vllm"] = identity["vllm"]
        for task in self.tasks:
            got = scores[task]
            metrics[task] = statistics.fmean(got.per_repeat)
            detail[task] = {
                "metric": got.metric,
                "stderr": _stderr(got.per_repeat),
                "repeats": repeats[task],
                "samples": got.samples * repeats[task],
                "per_repeat": got.per_repeat,
                "per_question": got.questions,
                "prompt_hashes": got.prompt_hashes,
                "cap_hits": got.cap_hits,
                "raw_generations": got.raw_files,
            }
            if self.clean_subset:
                clean = self._clean(task, got.question_ids, got.questions, excluded)
                metrics[f"{task}_clean"] = clean["score"]
                higher[f"{task}_clean"] = higher[task]
                detail[task]["clean_subset"] = clean
        try:
            detail["lm_eval"] = version("lm-eval")
        except PackageNotFoundError:
            detail["lm_eval"] = "unknown"
        return SuiteResult(suite=self.name, metrics=metrics, higher_is_better=higher, detail=detail)

    def _score(
        self,
        task: str,
        result: dict[str, Any],
        got: _TaskScores,
        raw_by_doc: dict[tuple[str, str], list[dict[str, Any]]],
        cap: int,
        j: int,
    ) -> list[dict[str, Any]]:
        """Add sample ``j`` of ``task`` from one harness result to ``got``; return its records for regrading.

        Raises:
            RuntimeError: a question repeats within the sample, or the questions differ from sample 0's.
            ValueError: ``clean_subset`` is set and a question has no dataset ``question_id``.
        """
        key = self._metric_key(task, result["results"][task])
        got.metric = key
        got.per_repeat.append(float(result["results"][task][key]))
        got.higher = bool(result.get("higher_is_better", {}).get(task, {}).get(key.split(",")[0], True))
        leaves = _leaves(result, task)
        counts = result.get("n-samples", {})
        got.samples = int(
            counts.get(task, {}).get("effective")
            or sum(int(counts.get(leaf, {}).get("effective", 0)) for leaf in leaves)
        )
        metric, filter_name = key.split(",", 1)
        records = []
        hits = 0
        seen = set()
        previous = set(got.questions)
        for leaf in leaves:
            for row in result.get("samples", {}).get(leaf, []):
                if row["filter"] != filter_name:
                    continue
                ident = f"{row['doc_id']}:{row['doc_hash']}"
                if ident in seen:
                    raise RuntimeError(f"lm_eval: duplicate question {ident} in {task}")
                seen.add(ident)
                got.questions.setdefault(ident, []).append(float(row[metric]))
                if self.clean_subset:
                    doc = row.get("doc")
                    qid = doc.get("question_id") if isinstance(doc, dict) else None
                    if qid is None:
                        raise ValueError(
                            f"clean_subset: {task} question {ident} has no dataset question_id; the clean "
                            "subset cannot be mapped"
                        )
                    got.question_ids[ident] = str(qid)
                got.prompt_hashes.setdefault(ident, row["prompt_hash"])
                raw = raw_by_doc.get((leaf, str(row["doc_id"])), [])
                # A cap hit is a generation the cap cut before any answer. Read it from the raw completion's
                # finish reason and answer region: a filter's stand-in for an unparsed answer (``[invalid]``)
                # is not an answer.
                hits += sum(
                    r["finish_reason"] == "length"
                    and r["tokens"] >= cap
                    and not answer_region(r["text"], self.think_end_token).strip()
                    for r in raw
                )
                records.append({**row, "question_id": ident, "seed": self.seed + j, "raw": raw})
        if not seen or (j and seen != previous):
            raise RuntimeError(f"lm_eval: questions changed between repeats for {task}")
        got.cap_hits.append(hits)
        return records

    @staticmethod
    def _clean(
        task: str, ids: dict[str, str], questions: dict[str, list[float]], excluded: set[str]
    ) -> dict[str, Any]:
        """The clean-subset headline: the mean per-question score over the questions whose dataset id is kept.

        Raises:
            ValueError: two scored questions share a dataset id, an excluded id was not scored, or nothing
                remains after the exclusion.
        """
        if len(set(ids.values())) != len(ids):
            raise ValueError(
                f"clean_subset: {task} repeats a dataset question_id; the subset cannot be mapped"
            )
        missing = excluded - set(ids.values())
        if missing:
            raise ValueError(
                f"clean_subset: {len(missing)} excluded question ids are not among {task}'s scored questions"
            )
        kept = [ident for ident, qid in ids.items() if qid not in excluded]
        if not kept:
            raise ValueError(f"clean_subset: no {task} question remains after excluding {len(excluded)} ids")
        per_repeat = [
            statistics.fmean(questions[ident][j] for ident in kept) for j in range(len(questions[kept[0]]))
        ]
        return {
            "score": statistics.fmean(per_repeat),
            "questions": len(kept),
            "excluded": len(excluded),
            "per_repeat": per_repeat,
            "question_ids": ids,
        }

    def inputs(self) -> dict[str, str | None]:
        """SHA-256 of each task dataset's files and of the private exclusion file, so editing one reruns the job."""
        found: dict[str, str | None] = {}
        for i, dataset in enumerate(self.datasets):
            try:
                found[f"datasets.{i}"] = content_sha256(dataset.local())
            except (OSError, ValueError):
                found[f"datasets.{i}"] = None
        if self.clean_subset is not None:
            try:
                found["clean_subset_sha256"] = sha256_file(Path(self.clean_subset))
            except OSError:
                found["clean_subset_sha256"] = None
        return found

    def parts(self) -> list[Layout]:
        """One part per task, each with that task's repeats, answer cap and metric, so each task is its own job."""
        repeats, caps = self._per_task(self.repeats), self._per_task(self.max_tokens)
        return [
            self.model_copy(
                update={
                    "tasks": [task],
                    "repeats": {task: repeats[task]},
                    "max_tokens": {task: caps[task]},
                    "metric": {task: self.metric[task]} if task in self.metric else {},
                }
            )
            for task in self.tasks
        ]

    def part_label(self) -> str | None:
        """The task's name, naming the result file ``lm_eval.<task>.json``; None while the suite holds several."""
        return self.tasks[0] if len(self.tasks) == 1 else None

    def _gen_kwargs(self, checkpoint: str) -> dict[str, Any]:
        """The sampling the harness generates with: the shared resolver's settings plus ``do_sample``.

        Raises:
            ValueError: the resolver finds no sampling setting, or one it does not know.
        """
        sampling = resolve_sampling(checkpoint, self.sampling, "lm_eval.sampling")
        # vLLM starts at temperature 1.0, but the harness reads a missing setting as 0 (greedy); name it.
        gen: dict[str, Any] = {"temperature": 1.0, **sampling}
        gen["do_sample"] = bool(gen["temperature"] > 0.0)
        return gen

    def _per_task(self, value: int | dict[str, int]) -> dict[str, int]:
        """Expand a setting given once or per task into a mapping over every task."""
        return dict.fromkeys(self.tasks, value) if isinstance(value, int) else dict(value)

    @staticmethod
    def _runs(repeats: dict[str, int], caps: dict[str, int], j: int) -> dict[int, list[str]]:
        """The tasks that still need sample ``j``, grouped by their answer cap (one harness call per cap)."""
        runs: dict[int, list[str]] = {}
        for task, count in repeats.items():
            if count > j:
                runs.setdefault(caps[task], []).append(task)
        return runs

    def _metric_key(self, task: str, result: dict[str, Any]) -> str:
        """The one headline metric of ``task``, refusing an ambiguous result.

        ``metric[task]`` names the metric (with or without its filter suffix) when the task reports several.

        Raises:
            Refused: the task reports no metric, or more than one and the config names none.
        """
        keys = [
            key
            for key in result
            if key not in ("name", "alias", "sample_len") and not key.split(",")[0].endswith("_stderr")
        ]
        wanted = self.metric.get(task)
        if wanted:
            keys = [key for key in keys if key == wanted or key.split(",")[0] == wanted]
        if len(keys) != 1:
            raise Refused(f"metric: task {task} reports {sorted(keys)}; set metric: {{{task}: <name>}}")
        return keys[0]


def _leaves(result: dict[str, Any], task: str) -> list[str]:
    """The leaf tasks whose per-question scores make up ``task``: itself, unless it is a harness group.

    A group's result holds one weighted aggregate (the headline) and per-leaf ``samples``; the mapping is the
    harness's ``group_subtasks``. A nested group is walked down to its leaves.

    Args:
        result: one ``simple_evaluate`` result.
        task: a requested task or group name.

    Returns:
        The leaf task names, in the group's order.
    """
    children = result.get("group_subtasks", {}).get(task)
    return [leaf for child in children for leaf in _leaves(result, child)] if children else [task]


def _stderr(readings: list[float]) -> float | None:
    """The standard error of the mean over one reading per sample, or None with a single sample."""
    return statistics.stdev(readings) / math.sqrt(len(readings)) if len(readings) > 1 else None
