"""Eval suites: the plugins behind ``suites`` in a ``qda eval`` config.

A config names suites by key, each with its settings::

    suites:
      - kl: {corpus: {file: {path: prompts.jsonl}}, scorer: {topk: {}}}

``SUITES`` holds built-ins and extensions added through ``qda.plugins`` hooks. A suite measures one checkpoint and
returns a :class:`SuiteResult`; it raises :class:`qda_quant.step.Refused` to refuse, and the runner records the
refusal in that job's result file (its ``error``) instead of a measurement. A suite of independent measurements
splits itself into parts (:meth:`Suite.parts`), one job and one result file each.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from qda_quant.intake.corpus import Corpus, parse_corpus, split
from qda_quant.recipe import error_lines
from qda_quant.registry import Registry
from qda_quant.step import Refused

from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.holdout import KLHoldoutCorpus
from qda_kl.scorers import KLScorer, parse_scorer
from qda_kl.serve import check_serve_env

#: Eval suites by config key; records keep the group ``qda.eval_suites``.
SUITES = Registry(
    "eval_suites",
    {
        "full": "qda_kl.eval.full_preset:FullSuite",
        "kl": "qda_kl.eval.suites:KLSuite",
        "lm_eval": "qda_kl.eval.lm_eval:LmEvalSuite",
        "sanity": "qda_kl.eval.sanity:SanitySuite",
        "specdec": "qda_kl.eval.specdec:SpecDecSuite",
    },
    group="qda.eval_suites",
)


class SuiteResult(BaseModel):
    """What one part of a suite measured on one checkpoint: a ``results/<checkpoint>/<suite>.json`` file.

    Attributes:
        suite: the suite's key.
        checkpoint: the checkpoint's key in the config (filled in by the runner).
        metrics: headline numbers, e.g. ``{"kl": 0.0098}``; empty when the job failed.
        higher_is_better: per metric, its direction (the table's business, recorded here).
        percent: per metric, whether the table shows it as a percent (a fraction); a metric not named here is.
        detail: anything else worth keeping (per-domain numbers, versions, kernels).
        wall_s: seconds the suite's ``run`` took (filled in by the runner).
        config_digest: SHA-256 of the suite's settings, inputs, reference and checkpoint (filled in by the
            runner); a result whose digest differs from the config's is stale and is run again.
        error: the exact error that refused this job (an exception's text), or None for a measurement. A failed
            job is never cached: it is retried, and a successful rerun replaces this record.
    """

    suite: str
    checkpoint: str = ""
    metrics: dict[str, float]
    higher_is_better: dict[str, bool]
    percent: dict[str, bool] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)
    wall_s: float = 0.0
    config_digest: str = ""
    error: str | None = None


class Suite(BaseModel):
    """A way of measuring a checkpoint. Settings are pydantic fields with documented defaults."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: ClassVar[str] = ""

    #: The result file this job writes (``results/<key>/<suite>.json``), set by the runner before ``run`` so a
    #: suite that keeps an artifact can put it beside its result. Not a setting: it is left out of ``to_dict``.
    result_path: Path | None = Field(default=None, exclude=True)

    def prepare(self, reference: str, work: Path) -> None:
        """Do the once-per-process work for ``reference`` under ``work`` (the run's ``out``); default: nothing.

        It runs before the suite's first job in each process, so separate ``--only`` jobs sharing one ``out``
        call it once each; it must therefore be idempotent and locked against a concurrent build (the ``kl``
        suite locks its reference directory). Once a process has prepared the suite, it prepares it no more.
        """

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        """Measure ``checkpoint`` (a local directory) against ``reference`` (a local directory)."""
        raise NotImplementedError

    def parts(self) -> list[Suite]:
        """The independent pieces of this suite, each a narrower copy of it; default: the suite itself.

        Each part is its own job with its own result file, so a suite whose measurements are independent (the
        tasks of a benchmark suite) returns one part per measurement and each becomes resumable on its own.
        """
        return [self]

    def cards(self) -> int:
        """The cards one job needs (``qda eval --dry-run`` states it); default: one."""
        return 1

    def part_label(self) -> str | None:
        """The part's label, naming its result file ``<suite>.<label>.json``; None for a suite that is one job.

        A label is one name segment, ``[A-Za-z0-9][A-Za-z0-9_-]*``: no ``.`` (which separates it from the suite
        in the file name) and no path separator. Two parts of one suite cannot share a label.
        """
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings as ``{key: settings}`` with every default filled in."""
        return {self.name: self.model_dump(mode="json")}

    def inputs(self) -> dict[str, str | None]:
        """Digests of the data the settings name (a corpus's rows), so editing it reruns the suite.

        The runner adds them to each job's ``config_digest``. Default: none.
        """
        return {}


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold ``path``'s exclusive lock (:mod:`fcntl`) for the block, creating its directory.

    The suites that build a reference under ``refs/<digest>`` take ``<digest>.lock`` this way, so parallel
    ``--only`` jobs, and runs sharing one ``refs``, build it once and the others reuse it.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class Layout(Suite):
    """The layout a suite's engines run on: suite-owned ``tp``, ``serve_env``, ``serve_args`` and ``nccl_nvls``.

    Attributes:
        tp: tensor-parallel size for every engine the suite starts. The layout is part of the test: it is one
            per suite, so a reading taken on one layout is never reused for another.
        serve_env: environment variables for those engines, such as a setting another copy of a library reads.
            The reference build and every checkpoint engine the suite starts read it. It may not name another
            ``NCCL_NVLS_ENABLE`` than ``nccl_nvls`` does.
        serve_args: other ``vllm.LLM`` arguments for those engines, such as ``enable_expert_parallel`` and
            ``pipeline_parallel_size`` (a positive integer; a job needs ``tp`` times that many cards). The card
            count is ``tp``, which nothing else names: ``tensor_parallel_size`` here is refused.
        nccl_nvls: let those engines' ranks join NCCL's NVLS (multicast) collectives
            (:attr:`qda_kl.serve.ServeSettings.nccl_nvls`). Off by default, as for a build: NVLS binds shared
            memory that some nodes cannot map, so an engine at ``tp`` above 1 would stop at group init there.
    """

    tp: int = Field(default=DEFAULT_TP, ge=1)
    serve_env: dict[str, str] = Field(default_factory=dict)
    serve_args: dict[str, Any] = Field(default_factory=dict)
    nccl_nvls: bool = False

    @model_validator(mode="after")
    def _one_layout(self) -> Layout:
        """Refuse a serve_args that names its own card count or a split that is not a positive integer."""
        if "tensor_parallel_size" in self.serve_args:
            raise ValueError(
                f"serve_args names tensor_parallel_size={self.serve_args['tensor_parallel_size']}; "
                f"set tp={self.tp} instead"
            )
        self._pipeline_parallel_size()
        check_serve_env(self.serve_env, self.nccl_nvls)
        return self

    def _pipeline_parallel_size(self) -> int:
        """``serve_args.pipeline_parallel_size``: a positive integer (default 1), refused otherwise.

        Raises:
            ValueError: the value is not a positive integer; the message names the key.
        """
        stages = self.serve_args.get("pipeline_parallel_size", 1)
        if type(stages) is not int or stages < 1:
            raise ValueError(f"serve_args.pipeline_parallel_size={stages!r}: expected a positive integer")
        return stages

    def cards(self) -> int:
        """The engines run this suite's ``tp`` across its serve_args' pipeline stages."""
        return self.tp * self._pipeline_parallel_size()

    def _layout_fields(self) -> dict[str, Any]:
        """The layout settings, for a subclass's :meth:`LayoutSuite._digest_fields`."""
        return self.model_dump(mode="json", include={"tp", "serve_env", "serve_args", "nccl_nvls"})


class LayoutSuite(Layout):
    """A suite measured on a corpus's eval side, whose engines run on one card layout.

    Attributes:
        corpus: the rows the suite reads, one ``{key: settings}`` from ``qda.corpora``; default: the KL v2
            corpus's holdout split (``qda.corpora.kl_holdout``). Only its eval side is read, the rows calibration
            never sees.
        refs: shared directory for ``<digest>/`` readings; default the run's ``out/refs``. Use one directory
            across runs to build the original's reference once and reuse identical bytes.
    """

    #: What the eval side must hold, for the refusal message ("rows", "prompt rows").
    kind: ClassVar[str] = "rows"

    corpus: Corpus = Field(default_factory=KLHoldoutCorpus)
    refs: str | None = Field(default=None, min_length=1)

    def refs_dir(self, work: Path) -> Path:
        """The directory that holds the ``<digest>/`` references: ``refs``, or the run's ``out/refs``."""
        return Path(self.refs) if self.refs else work / "refs"

    @field_validator("corpus", mode="before")
    @classmethod
    def _corpus(cls, v: Any) -> Any:
        if isinstance(v, Corpus):
            return v
        try:
            return parse_corpus(v)
        except ValidationError as e:
            ((key, _),) = v.items()  # parse_corpus validated the shape before the settings
            raise Refused("; ".join(error_lines(e, f"corpus.{key}"))) from None

    def _keep(self, _row: dict[str, Any]) -> bool:
        """Whether ``_row`` is one the suite measures; default: every row."""
        return True

    def _rows(self) -> list[dict[str, Any]]:
        """The corpus's eval-side rows the suite reads, ids as strings.

        Raises:
            Refused: the corpus refuses (missing file, a bad row, nothing of :attr:`kind` on the eval side).
        """
        rows = [{**r, "id": str(r["id"])} for r in split(self.corpus, "eval") if self._keep(r)]
        if not rows:
            raise Refused(
                f"corpus {self.corpus.name}: no {self.kind} on the eval side (holdout {self.corpus.holdout})"
            )
        return rows

    def inputs(self) -> dict[str, str | None]:
        """SHA-256 of the rows the suite reads (None when the corpus refuses; ``prepare`` then refuses)."""
        try:
            return {"corpus_sha256": sha256_rows(self._rows())}
        except ValueError:
            return {"corpus_sha256": None}

    def _own(self) -> dict[str, Any]:
        """The suite's settings but the corpus, as JSON, in the order :meth:`to_dict` lists them."""
        own = self.model_dump(mode="json", exclude={"corpus"})
        return {"refs": own.pop("refs"), **own}  # refs before the layout, as before the CorpusSuite fold

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings with the corpus as ``{key: settings}``."""
        corpus = {self.corpus.name: self.corpus.model_dump(mode="json")}
        return {self.name: {"corpus": corpus, **self._own()}}

    def _digest_fields(self) -> dict[str, Any]:
        """The settings that can change the reference; each suite names its own (:meth:`reference_dir`)."""
        raise NotImplementedError

    def reference_dir(self, reference: str, work: Path) -> Path:
        """``refs/<digest>``: the reading for these rows, settings and this model.

        Raises:
            Refused: the corpus refuses.
        """
        key = {"reference": reference, "corpus_sha256": sha256_rows(self._rows()), **self._digest_fields()}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return self.refs_dir(work) / digest


def parse_suite(d: Any, where: str) -> Suite:
    """Parse one ``{key: settings}`` entry of ``suites``; ``where`` is its config key (``suites.0``).

    Raises:
        Refused: not one key, an unknown key (the message lists the known ones), or invalid settings.
    """
    if isinstance(d, Suite):
        return d
    if not isinstance(d, dict) or len(d) != 1:
        raise Refused(f"{where}: expected one suite {{key: settings}}, known: {sorted(SUITES)}; got {d!r}")
    ((key, cfg),) = d.items()
    if "." in key:
        raise Refused(
            f"{where}.{key}: a suite name cannot contain '.' (it separates a suite from its labels)"
        )
    if key not in SUITES:
        raise Refused(f"{where}.{key}: unknown suite; known: {sorted(SUITES)}")
    try:
        return SUITES[key](**(cfg or {}))
    except ValidationError as e:
        raise Refused("; ".join(error_lines(e, f"{where}.{key}"))) from None


#: ``vllm.LLM`` arguments that split one model over cards: the suite's layout, never a scorer's own.
PARALLEL_ARGS = ("tensor_parallel_size", "pipeline_parallel_size", "enable_expert_parallel")


class KLSuite(LayoutSuite):
    """Mean per-token KL(original || checkpoint) on the reference's own answers (:mod:`qda_kl.score`).

    The reference (the original model's answers and its top-k at every token) is built once in ``prepare`` from
    the corpus's eval side (:func:`qda_quant.intake.corpus.split`), and cached under ``refs/<digest>`` (see
    :attr:`LayoutSuite.refs`), where the digest covers the reference model, those rows and the settings below.
    A ``refs`` that is itself a reference is a frozen one: it is scored in place, never rebuilt, and the corpus
    must select the prompts it holds.

    Attributes:
        corpus: the prompts the reference answers, one ``{key: settings}`` from ``qda.corpora``; default: the
            KL v2 corpus's holdout split (``qda.corpora.kl_holdout``), the prompts ``qda-kl kl`` reads. Rows are
            ``prompt`` (answered) or ``text`` (teacher-forced) rows.
        answers: answers per prompt.
        seed: seed of draw 0; draw ``d`` uses ``seed + d``.
        max_tokens: cap of one answer.
        k: top-k kept per position.
        chat_template_kwargs: extra chat-template arguments (e.g. ``{"enable_thinking": false}``).
        sampling: the reference's sampling (``temperature``, ``top_p``, ...); each field wins over the model's
            ``generation_config.json``. None: that file alone; the build refuses when neither names one.
        subset: named prompt subset stored in the reference; None scores the full reference.
        scorer: the KL scorer (:mod:`qda_kl.scorers`); defaults to the top-k scorer. Its ``tp`` must be the
            suite's, and its own ``serve_args`` may name no parallel argument, which would reach only the
            checkpoint.
        gpu_memory_utilization: vLLM's share of the card for the reference build. It cannot change the
            reference, so it is not part of the reference digest; lower it when other jobs share the card
            (the checkpoint side takes it through ``scorer.topk.serve_args``).
        serve_env: environment variables for both engines, over the scorer's own (:class:`Layout`).
        nccl_nvls: let both engines' ranks join NCCL's NVLS collectives (:class:`qda_kl.scorers.TopKScorer`). Like
            ``tp`` it is one per suite: a scorer naming another value is refused.
    """

    name: ClassVar[str] = "kl"
    corpus: Corpus = Field(default_factory=KLHoldoutCorpus)
    answers: int = Field(default=2, ge=1)
    seed: int = 17
    max_tokens: int = Field(default=8192, ge=1)
    k: int = Field(default=128, ge=1)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling: dict[str, Any] | None = None
    subset: str | None = Field(default=None, min_length=1)
    scorer: KLScorer = Field(default_factory=lambda: parse_scorer(None))
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, ge=0.05, le=0.95)

    @model_validator(mode="after")
    def _scorer_layout(self) -> KLSuite:
        """Refuse a scorer's own card count, NVLS or parallel split: the suite's reach both engines."""
        for key in ("tp", "nccl_nvls"):
            if key in self.scorer.model_fields_set:
                named, ours = self.scorer.model_dump(include={key})[key], getattr(self, key)
                if named != ours:
                    raise Refused(f"{key}: the scorer names {key}={named}, the suite {key}={ours}")
        own = self.scorer.model_dump(include={"serve_args"})
        held = sorted(set(own.get("serve_args", {})) & set(PARALLEL_ARGS))
        if held:
            raise Refused(
                f"serve_args: the scorer names {', '.join(held)}; the suite owns the layout "
                "(its tp, serve_env and serve_args), which both engines read"
            )
        type(self.scorer).model_validate(self._scorer().model_dump())
        return self

    def _owned(self) -> dict[str, Any]:
        """The suite's values of the settings its scorer has and the suite owns."""
        return {k: getattr(self, k) for k in self.scorer.model_dump(include={"tp", "nccl_nvls"})}

    @field_validator("scorer", mode="before")
    @classmethod
    def _named(cls, v: Any) -> Any:
        return v if isinstance(v, KLScorer) else parse_scorer(v)

    def _own(self) -> dict[str, Any]:
        """Serialize settings with the scorer at the suite's layout: a resolved config reloads as it stands."""
        scorer = self.scorer.model_copy(update=self._owned())
        d = self.model_dump(mode="json", exclude={"corpus", "scorer"})
        return {"refs": d.pop("refs"), **d, "scorer": scorer.to_dict()}

    def _digest_fields(self) -> dict[str, Any]:
        """The settings that can change the reference; the card share cannot, the layout can."""
        own = self.model_dump(
            mode="json",
            include={"answers", "seed", "max_tokens", "k", "chat_template_kwargs", "sampling"},
        )
        return {**own, **self._layout_fields()}

    def frozen(self) -> Path | None:
        """``refs`` itself when it is a reference (holds ``manifest.json``), else None."""
        if self.refs is None:
            return None
        path = Path(self.refs)
        return path if (path / "manifest.json").is_file() else None

    def reference_dir(self, reference: str, work: Path) -> Path:
        """The frozen ``refs`` itself, else ``refs/<digest>`` (:meth:`LayoutSuite.reference_dir`)."""
        frozen = self.frozen()
        return frozen if frozen is not None else super().reference_dir(reference, work)

    def prepare(self, reference: str, work: Path) -> None:
        """Build the reference unless ``refs/<digest>`` already holds it.

        Holds ``refs/<digest>.lock`` while it checks and builds, so parallel ``--only`` jobs and runs sharing
        one ``refs`` build the reference once and the others reuse it. The eval-side rows go to
        ``refs/<digest>.jsonl``, the prompt file the reference records. A ``refs`` that is itself a reference
        is scored in place, never built (:meth:`_frozen_prompts`).
        """
        if (frozen := self.frozen()) is not None:
            self._frozen_prompts(frozen)
            return
        out = self.reference_dir(reference, work)
        with locked(out.with_suffix(".lock")):
            if (out / "manifest.json").is_file():
                return
            from qda_kl.reference import build

            prompts = out.with_suffix(".jsonl")
            prompts.write_text("".join(json.dumps(r) + "\n" for r in self._rows()))
            build(
                reference,
                prompts,
                out,
                answers=self.answers,
                seed=self.seed,
                max_tokens=self.max_tokens,
                k=self.k,
                tp=self.tp,
                serve_env=self.serve_env,
                serve_args=self.serve_args,
                nccl_nvls=self.nccl_nvls,
                chat_template_kwargs=self.chat_template_kwargs or None,
                sampling=self.sampling,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )

    def _frozen_prompts(self, path: Path) -> None:
        """Refuse (``ValueError``) unless the corpus selects the prompts the reference at ``path`` covers."""
        from qda_kl.reference import dropped_prompts
        from qda_kl.store import Reference

        ref = Reference(path, subset=self.subset)
        want = {s.prompt_id for s in ref.sequences}
        covered = want | dropped_prompts(ref.manifest.build)
        got = {str(r["id"]) for r in self._rows()}
        missing = sorted(want - got)
        extra = sorted(got - covered) if self.subset is None else []
        if missing or extra:
            where = f"{path}: the reference" + (f" subset {self.subset!r}" if self.subset else "")
            raise ValueError(
                f"{where} holds {len(want)} prompts, the corpus selects {len(got)} ({len(missing)} missing,"
                f" {len(extra)} extra; e.g. {missing[:3]} / {extra[:3]}); a frozen reference is scored on"
                " the prompts it holds"
            )

    def _scorer(self) -> KLScorer:
        """This suite's scorer at the suite's layout: the layout fields it has, over its own values."""
        own = self.scorer.model_dump(include={"serve_env", "serve_args"})
        gains = self._owned()
        for key, ours in (("serve_env", self.serve_env), ("serve_args", self.serve_args)):
            if key in own:
                gains[key] = {**own[key], **ours}
        return self.scorer.model_copy(update=gains) if gains else self.scorer

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        """Score ``checkpoint`` against the reference ``prepare`` built.

        Raises:
            Refused: the scorer refused the result (e.g. repeats differ).
        """
        from qda_kl.store import Reference

        path = self.reference_dir(reference, work)
        ref = Reference(path, subset=self.subset)
        selection = {"subset": self.subset} if self.subset is not None else {}
        got = self._scorer().run(checkpoint, path, **selection)
        if got.refused:
            raise Refused(got.refused)
        result = kl_result(got.result.model_dump(mode="json"), self.name)
        result.detail["reference_sha256"], result.detail["subset"] = ref.content_sha256, self.subset
        return result


def kl_result(reading: dict[str, Any], suite: str = KLSuite.name) -> SuiteResult:
    """A KL reading (what ``qda-kl score`` writes to ``score.json``) as ``suite``'s result.

    ``kl`` and ``kl_text`` are its metrics when they are numbers; everything else is its detail.
    """
    metrics = {k: reading[k] for k in ("kl", "kl_text") if type(reading.get(k)) in (int, float)}
    return SuiteResult(
        suite=suite,
        metrics=metrics,
        higher_is_better=dict.fromkeys(metrics, False),
        percent=dict.fromkeys(metrics, False),
        detail={k: v for k, v in reading.items() if k not in metrics},
    )


def sha256_rows(rows: list[dict[str, Any]]) -> str:
    """SHA-256 of rows, independent of key order."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
