"""KL scorers: the plugins a ``qda-kl score`` config names as ``scorer: {key: settings}`` (default ``topk``).

A package adds one to ``SCORERS`` from its ``qda.plugins`` hook; :meth:`~KLScorer.run` raises `Refused` to refuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from qda_quant.recipe import error_lines
from qda_quant.registry import Registry
from qda_quant.step import Refused

from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.serve import check_serve_env

#: KL scorers by config key; records keep the group ``qda_kl.scorers``.
SCORERS = Registry(
    "scorers",
    {"topk": "qda_kl.scorers:TopKScorer"},
    group="qda_kl.scorers",
)

#: The scorer a config gets when it names none.
DEFAULT = "topk"

#: Scorer keys that no longer exist, and what replaced them.
REMOVED = {
    "llamacpp": "drop it: `qda-kl score` decodes a .gguf with `qda decode-gguf` and reads the decode with vLLM "
    '(docs/verify.md, "A GGUF file")',
}
REMOVED["llama.cpp"] = REMOVED["llamacpp"]


@dataclass(frozen=True)
class Outcome:
    """Collect repeated readings, their combined result and any publication refusal.

    Attributes:
        readings: each reading, in order (one per repeat).
        result: the reading the result file holds (for several repeats, their combination).
        refused: why the result must not be published, after the readings were shown (None: publish).
    """

    readings: Sequence[BaseModel]
    result: BaseModel
    refused: str | None = None


class KLScorer(BaseModel):
    """Configure a scorer that compares a checkpoint against a reference."""

    model_config = ConfigDict(extra="forbid")

    name: ClassVar[str] = ""

    def run(self, checkpoint: str, reference: Path, *, subset: str | None = None) -> Outcome:
        """Score ``checkpoint`` against ``reference``; a named subset must be honored or refused."""
        raise NotImplementedError

    def summary(self, reading: BaseModel) -> dict[str, Any]:
        """Return the fields ``qda-kl score`` prints for a reading, defaulting to all fields."""
        return reading.model_dump(mode="json")

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings as ``{key: settings}`` with every default filled in."""
        return {self.name: self.model_dump(mode="json")}


def parse_scorer(d: dict[str, Any] | None) -> KLScorer:
    """Parse a single-key scorer mapping, using the default scorer for None.

    Raises:
        Refused: not one key, or an unknown key (the message lists the known ones).
    """
    if d is None:
        return SCORERS[DEFAULT]()
    if not isinstance(d, dict) or len(d) != 1:
        raise Refused(f"scorer: expected one scorer {{key: settings}}, known: {sorted(SCORERS)}; got {d!r}")
    ((key, cfg),) = d.items()
    if key in REMOVED:
        raise Refused(
            f"scorer.{key}: removed; {REMOVED[key]}"
        )  # one line: config load keeps the message only
    if key not in SCORERS:
        raise Refused(f"scorer.{key}: unknown scorer; known: {sorted(SCORERS)}")
    try:
        return SCORERS[key](**(cfg or {}))
    except ValidationError as e:  # name the settings as config keys
        raise Refused("; ".join(error_lines(e, f"scorer.{key}"))) from None


class TopKScorer(KLScorer):
    """Score teacher-forced KL against stored top-k probabilities inside vLLM's workers.

    Attributes:
        repeat: score this many times on one engine; differing readings are refused unless
            ``allow_nondeterministic``.
        allow_nondeterministic: with ``repeat`` > 1, publish the mean and spread of differing readings (MoE).
        check: validate the hook against the engine's own prompt log-probs.
        group: sequences per engine call (1: each read alone, batch-independent).
        serve_env: engine environment, over ``build.json``'s ``serve.env``.
        serve_args: ``vllm.LLM`` keyword arguments, over ``serve.args``.
        gpu_memory_utilization: vLLM's share of the card (``0.85``); lower it when another job shares the card.
            ``serve_args`` may not name it: this field is where it is set.
        expect_kernels: kernels vLLM must bind (``{kind: class or [classes]}``), over ``serve.kernels``.
        allow_vllm_mismatch: score on a vLLM other than the checkpoint's.
        allow_dtype_mismatch: permit an activation dtype different from the reference anchor's; default false.
        tp: tensor-parallel size; the job must run on that many cards.
        nccl_nvls: let the engine's ranks join NCCL's NVLS (multicast) collectives. Off by default: NVLS binds
            shared memory that some nodes cannot map, and an engine on more than one card then fails at start.
            ``serve_env`` may not name ``NCCL_NVLS_ENABLE`` with another value.
    """

    name: ClassVar[str] = "topk"
    repeat: int = Field(default=1, ge=1)
    allow_nondeterministic: bool = False
    check: bool = True
    group: int = Field(default=1, ge=1)
    serve_env: dict[str, str] = Field(default_factory=dict)
    serve_args: dict[str, Any] = Field(default_factory=dict)
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, ge=0.05, le=0.95)
    expect_kernels: dict[str, str | list[str]] = Field(default_factory=dict)
    allow_vllm_mismatch: bool = False
    allow_dtype_mismatch: bool = False
    tp: int = Field(default=DEFAULT_TP, ge=1)
    nccl_nvls: bool = False

    @model_validator(mode="after")
    def _one_nvls(self) -> TopKScorer:
        """Refuse a ``serve_env`` that says otherwise than ``nccl_nvls``: the key is the one place it is set."""
        check_serve_env(self.serve_env, self.nccl_nvls)
        return self

    @model_validator(mode="after")
    def _one_gpu_memory(self) -> TopKScorer:
        """Refuse a ``serve_args`` that says otherwise than ``gpu_memory_utilization``: one field owns it."""
        if "gpu_memory_utilization" in self.serve_args:
            raise ValueError(
                "serve_args names gpu_memory_utilization; set the scorer's gpu_memory_utilization instead"
            )
        return self

    def run(self, checkpoint: str, reference: Path, *, subset: str | None = None) -> Outcome:
        """Open one engine on ``checkpoint`` and score it ``repeat`` times."""
        from qda_kl import score
        from qda_kl.serve import engine_environment, settings_for

        settings = settings_for(checkpoint, self.serve_env, self.serve_args, self.nccl_nvls)
        settings.env = engine_environment(settings)
        self.serve_env = dict(settings.env)
        if self.expect_kernels:
            settings = settings.model_copy(update={"kernels": {**settings.kernels, **self.expect_kernels}})
        sc = score.Scorer(
            checkpoint,
            reference,
            settings=settings,
            allow_vllm_mismatch=self.allow_vllm_mismatch,
            allow_dtype_mismatch=self.allow_dtype_mismatch,
            tp=self.tp,
            gpu_memory_utilization=self.gpu_memory_utilization,
            subset=subset,
        )
        got = [sc.score(check=self.check, group=self.group) for _ in range(self.repeat)]
        result = score.combine(got) if len(got) > 1 else got[0]
        refused = None
        if result.deterministic is False and not self.allow_nondeterministic:
            assert result.repeats is not None  # combine() records them whenever the readings differ
            refused = (
                f"repeats differ: kl {result.repeats['kl']}, kl_text {result.repeats['kl_text']} "
                "(allow_nondeterministic reports their mean and spread)"
            )
        return Outcome(got, result, refused)

    def summary(self, reading: BaseModel) -> dict[str, Any]:
        """Return a reading's KL, domains, token count, flip rate and serving diagnostics."""
        keys = (
            "kl",
            "domains",
            "kl_text",
            "tokens",
            "flip_rate",
            "vllm",
            "kernels",
            "dtype",
            "tp",
            "wall_s",
            "validate_max_abs_diff",
        )
        d = reading.model_dump(mode="json")
        return {k: d[k] for k in keys}
