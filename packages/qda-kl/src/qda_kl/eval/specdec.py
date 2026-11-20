"""Speculative decoding still works on a checkpoint: the ``specdec`` eval suite.

A checkpoint keeps its speculative decoder only if the draft is still served and the checkpoint accepts its
tokens. The draft is either the model's own multi-token-prediction layers (``method: mtp``) or an external
draft model (``method:`` the draft's hub id or directory); either way vLLM proposes, the checkpoint verifies, and
the acceptance rate says how often it agrees::

    suites:
      - specdec: {method: mtp, num_speculative_tokens: 1, sampling: {temperature: 1.0, top_p: 0.95}}

A build with ``rotate`` runs the model in a rotated residual stream, and its external draft reads that basis
only after ``qda rotate-draft``. The original model's reading keeps the original draft; ``checkpoint_method``
names the checkpoint's own (the rotated output), so one suite measures both sides correctly::

    suites:
      - specdec: {method: z-lab/Qwen3.5-9B-DFlash, checkpoint_method: work/qwen3.5-9b-dflash-rot}

Without ``checkpoint_method`` both sides use ``method``, as before.

The suite answers the corpus's eval-side prompts twice, through the original model and through the checkpoint,
with the same sampling and (unless ``checkpoint_method`` names another) the same draft, and returns the
checkpoint's acceptance rate, compared against the original's reading in ``detail.reference``. vLLM counts accepted and drafted tokens per decoding step and
publishes them as Prometheus counters (``vllm:spec_decode_num_*``), which
:func:`read_spec_metrics` reads from the engine's own snapshot (``LLM.get_metrics``), never from a log line.

Sampling is the shared resolver's merge of the config's ``sampling`` over the original model's
``generation_config.json`` (:func:`qda_quant.calib.selfgen.resolve_sampling`); the checkpoint is measured with
those settings, so the ratio compares decoding, not sampling. The original's reading is built once per ``refs``
directory in ``prepare`` and cached under ``refs/<digest>/specdec.json`` with its sha256 beside it, under the
same lock the ``kl`` suite takes (:class:`~qda_kl.eval.suites.LayoutSuite`).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator
from qda_quant.calib.selfgen import resolve_sampling, sampling_params
from qda_quant.step import Refused

from qda_kl.cli import publish
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION
from qda_kl.eval.suites import LayoutSuite, SuiteResult, locked, sha256_rows
from qda_kl.reference import max_model_len, versions
from qda_kl.serve import ServeSettings, settings_for, start_engine, vllm_version

#: The vLLM counters of speculative decoding (:mod:`vllm.v1.spec_decode.metrics`).
DRAFTS = "vllm:spec_decode_num_drafts"
DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"
ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

#: The reading ``prepare`` writes and ``run`` compares against, in the order the result reports it.
READING = (
    "acceptance_rate",
    "mean_acceptance_length",
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "per_position_acceptance",
)

#: The file under ``refs/<digest>`` that holds the original model's reading.
REFERENCE = "specdec.json"

#: The file beside it that records the reading's sha256, so a changed reading is refused.
DIGEST = "specdec.sha256"


def read_spec_metrics(metrics: Iterable[Any]) -> dict[str, Any]:
    """The acceptance figures in one ``LLM.get_metrics()`` snapshot, summed over every engine.

    Args:
        metrics: the snapshot: objects with a ``name``; a counter also has an integer ``value``, the
            per-position vector a list ``values``.

    Returns:
        ``acceptance_rate`` (accepted draft tokens / drafted tokens), ``mean_acceptance_length`` (vLLM's
        ``1 + accepted / drafts``: the tokens one decoding step emits, counting the target's own token), the
        three counts, and ``per_position_acceptance`` (the share of drafts accepted at each draft position).

    Raises:
        RuntimeError: a counter is missing, so the engine ran without speculative decoding, or it drafted no
            token.
    """
    counts = {DRAFTS: 0, DRAFT_TOKENS: 0, ACCEPTED_TOKENS: 0}
    seen = set()
    per_pos: list[int] = []
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name in counts:
            counts[name] += int(metric.value)
            seen.add(name)
        elif name == ACCEPTED_PER_POS:
            values = [int(v) for v in metric.values]
            per_pos += [0] * (len(values) - len(per_pos))
            for i, v in enumerate(values):
                per_pos[i] += v
            seen.add(name)
    missing = sorted(set(counts) - seen)
    if missing:
        raise RuntimeError(
            f"vLLM published no speculative-decoding counters ({', '.join(missing)}); the engine did not run "
            "with speculative decoding"
        )
    drafts, drafted, accepted = counts[DRAFTS], counts[DRAFT_TOKENS], counts[ACCEPTED_TOKENS]
    if not drafts or not drafted:
        raise RuntimeError("the engine drafted no tokens, so there is no acceptance rate to read")
    return {
        "acceptance_rate": accepted / drafted,
        "mean_acceptance_length": 1 + accepted / drafts,
        "num_drafts": drafts,
        "num_draft_tokens": drafted,
        "num_accepted_tokens": accepted,
        "per_position_acceptance": [n / drafts for n in per_pos],
    }


class SpecDecSuite(LayoutSuite):
    """How often a checkpoint accepts the tokens of its speculative decoder, against the original model's.

    Both models answer the corpus's eval-side ``prompt`` rows with the same draft and the same sampling; the
    original is read once in ``prepare`` and every checkpoint is compared with that reading.

    Attributes:
        method: ``mtp`` for the model's own multi-token-prediction layers, otherwise the draft model's hub id or
            directory (vLLM detects the draft's method, e.g. DFlash, from the draft's config). The original
            model's reading uses it, and the checkpoint's unless ``checkpoint_method`` names another.
        checkpoint_method: the checkpoint's own draft: ``qda rotate-draft``'s output for a rotated build, the
            same forms as ``method``. None: ``method``, so a run that does not rotate is one draft for both.
        num_speculative_tokens: draft tokens proposed per decoding step.
        corpus: the prompts, one ``{key: settings}`` from ``qda.corpora``; default: the KL v2 corpus's
            holdout split (``qda.corpora.kl_holdout``). Only its eval side is read, and only its ``prompt`` rows
            (a ``text`` or ``messages`` row has no answer to accept or reject).
        max_tokens: cap of one answer.
        sampling: the original model's sampling (``temperature``, ``top_p``, ...); each field wins over its
            ``generation_config.json``. None: that file alone; the suite refuses when neither names one.
        serve_env: the layout's engine environment (see :class:`LayoutSuite`); it goes over the checkpoint's
            ``build.json`` ``serve.env`` (:func:`qda_kl.serve.settings_for`) and to the reference as it is.
        serve_args: the layout's other ``vllm.LLM`` arguments (see :class:`LayoutSuite`); they go over the
            checkpoint's ``build.json`` (:func:`qda_kl.serve.settings_for`) and to the reference as they are.
            The draft is not set here: ``speculative_config`` is refused.
        nccl_nvls: let both engines' ranks join NCCL's NVLS collectives (:class:`Layout`).
        gpu_memory_utilization: vLLM's share of the card, for both engines; lower it when other jobs share the
            card.
        seed: sampling seed, the same for both models.
    """

    name: ClassVar[str] = "specdec"
    kind: ClassVar[str] = "prompt rows"

    method: str = Field(default="mtp", min_length=1)
    checkpoint_method: str | None = Field(default=None, min_length=1)
    num_speculative_tokens: int = Field(default=1, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    sampling: dict[str, Any] | None = None
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, ge=0.05, le=0.95)
    seed: int = 17

    @model_validator(mode="after")
    def _draft(self) -> SpecDecSuite:
        """Refuse a serve_args that competes with the suite's own draft."""
        if "speculative_config" in self.serve_args:
            raise Refused(
                "serve_args: the draft is set by method and num_speculative_tokens; drop speculative_config"
            )
        return self

    def _keep(self, row: dict[str, Any]) -> bool:
        """Only a ``prompt`` row has an answer to accept or reject."""
        return "prompt" in row

    def _digest_fields(self) -> dict[str, Any]:
        """The settings that can change the reading; the card share cannot, the layout can.

        ``checkpoint_method`` is not here: the reference is the original model's reading, and its draft is
        ``method``. A different checkpoint draft changes the checkpoint's job, not the reference.
        """
        own = {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "max_tokens": self.max_tokens,
            "sampling": self.sampling,
            "seed": self.seed,
        }
        return {**own, **self._layout_fields()}

    def draft(self, *, checkpoint: bool) -> str:
        """The draft one measurement uses: the suite's ``method``, or the checkpoint's ``checkpoint_method``.

        The one place that says who uses which draft; :meth:`speculative_config` renders it and nothing else
        decides it.
        """
        if checkpoint and self.checkpoint_method:
            return self.checkpoint_method
        return self.method

    def speculative_config(self, *, checkpoint: bool = False) -> dict[str, Any]:
        """The ``speculative_config`` vLLM starts with: the model's own MTP layers, or a named draft model."""
        k = self.num_speculative_tokens
        draft = self.draft(checkpoint=checkpoint)
        if draft == "mtp":
            return {"method": "mtp", "num_speculative_tokens": k}
        return {"model": draft, "num_speculative_tokens": k}

    def prepare(self, reference: str, work: Path) -> None:
        """Measure the original model once and cache the reading under :meth:`reference_dir`.

        Holds ``refs/<digest>.lock`` while it checks and builds, so parallel ``--only`` jobs and runs sharing
        one ``refs`` measure the original once and the others reuse it. The reading's sha256 goes beside it, so
        a later run can tell whether its bytes changed.
        """
        out = self.reference_dir(reference, work)
        with locked(out.with_suffix(".lock")):
            if (out / REFERENCE).is_file():
                self._reading(out)
                return
            rows = self._rows()
            sampling = resolve_sampling(reference, self.sampling, "specdec.sampling")
            got = self._measure(
                reference,
                ServeSettings(env=dict(self.serve_env), args=dict(self.serve_args), nccl_nvls=self.nccl_nvls),
                rows,
                sampling,
                checkpoint=False,
            )
            got["sampling"] = sampling
            got["prompts"] = {"rows": len(rows), "sha256": sha256_rows(rows)}
            out.mkdir(parents=True, exist_ok=True)
            publish(out / REFERENCE, json.dumps(got, indent=1))
            publish(out / DIGEST, hashlib.sha256((out / REFERENCE).read_bytes()).hexdigest() + "\n")

    def _reading(self, out: Path) -> tuple[dict[str, Any], str]:
        """The original model's reading and its sha256, refused unless they still match.

        Raises:
            RuntimeError: the reading or its recorded sha256 is missing, or the reading changed since it was
                written.
        """
        stored = out / REFERENCE
        try:
            raw = stored.read_bytes()
        except OSError as why:
            raise RuntimeError(f"the original model's specdec reading is missing ({stored}): {why}") from None
        try:
            recorded = (out / DIGEST).read_text().strip()
        except OSError:
            raise RuntimeError(f"{stored}: no recorded sha256 beside it; rebuild the reference") from None
        sha = hashlib.sha256(raw).hexdigest()
        if sha != recorded:
            raise RuntimeError(
                f"{stored}: sha256 differs from the one recorded beside it; the reference changed since it "
                "was built: rebuild it"
            )
        return json.loads(raw), sha

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        """Measure ``checkpoint`` with the original's draft and sampling, and compare with the original.

        Raises:
            Refused: the original's reading is missing, or it accepted no draft token (the ratio would be
                undefined).
            RuntimeError: the reading changed since it was built, or the checkpoint's engine did not run
                speculative decoding.
        """
        original, sha = self._reading(self.reference_dir(reference, work))
        if original["acceptance_rate"] <= 0:
            raise Refused(
                "the original model accepted no draft token, so the checkpoint's acceptance has no ratio"
            )
        rows = self._rows()
        settings = settings_for(checkpoint, self.serve_env, self.serve_args, self.nccl_nvls)
        got = self._measure(checkpoint, settings, rows, original["sampling"], checkpoint=True)
        metrics = {"acceptance_rate": got["acceptance_rate"]}
        detail = {
            "reference_sha256": sha,
            "method": self.method,
            "checkpoint_method": self.draft(checkpoint=True),
            "num_speculative_tokens": self.num_speculative_tokens,
            "max_tokens": self.max_tokens,
            "sampling": original["sampling"],
            "seed": self.seed,
            "prompts": {"rows": len(rows), "sha256": sha256_rows(rows)},
            "checkpoint": {k: got[k] for k in READING} | {"seconds": got["seconds"]},
            "reference": {k: original[k] for k in READING} | {"seconds": original["seconds"]},
            "serve": got["serve"],
            "vllm": got["vllm"],
            "versions": got["versions"],
        }
        return SuiteResult(
            suite=self.name,
            metrics=metrics,
            higher_is_better=dict.fromkeys(metrics, True),
            detail=detail,
        )

    def engine_kwargs(
        self, model: str, settings: ServeSettings, *, checkpoint: bool = False
    ) -> dict[str, Any]:
        """The ``vllm.LLM`` keyword arguments of one engine, on the suite's layout.

        ``settings.args`` (the checkpoint's ``build.json`` under ``serve_args``) may set ``max_model_len``,
        ``seed`` and ``gpu_memory_utilization``, and carries the pipeline stages and expert parallelism; the
        card count is ``tp``, which nothing overrides. The draft and stat logging are passed last: vLLM records
        the spec-decode counters only with ``disable_log_stats=False`` (``LLM`` defaults to off), and neither
        key may come from a checkpoint's file.

        Raises:
            ValueError: ``settings.args`` names another ``tensor_parallel_size`` than ``tp``.
        """
        kwargs = {
            "tensor_parallel_size": self.tp,
            "max_model_len": max_model_len(model, self.max_tokens),
            "seed": self.seed,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            **settings.args,
            "speculative_config": self.speculative_config(checkpoint=checkpoint),
            "disable_log_stats": False,
        }
        if int(kwargs["tensor_parallel_size"]) != self.tp:  # tp is the card count: nothing else may name one
            raise Refused(
                f"serve_args names tensor_parallel_size={kwargs['tensor_parallel_size']}; set tp={self.tp} instead"
            )
        return kwargs

    def _measure(
        self,
        model: str,
        settings: ServeSettings,
        rows: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        checkpoint: bool,
    ) -> dict[str, Any]:
        """Answer ``rows`` on ``model`` with speculative decoding on, and read the acceptance figures."""
        from vllm import LLM

        t0 = time.time()
        kwargs = self.engine_kwargs(model, settings, checkpoint=checkpoint)
        llm = start_engine(settings, lambda: LLM(model=model, **kwargs))
        loaded = time.time()
        llm.chat(
            [[{"role": "user", "content": row["prompt"]}] for row in rows],
            sampling_params(sampling, self.max_tokens, self.seed),
            use_tqdm=False,
        )
        got = read_spec_metrics(llm.get_metrics())
        got.update(
            {
                "vllm": vllm_version(),
                "versions": versions(),
                "serve": settings.model_dump(mode="json"),
                "seconds": {"load": round(loaded - t0, 1), "generate": round(time.time() - loaded, 1)},
            }
        )
        return got
