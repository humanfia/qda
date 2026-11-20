"""Score a checkpoint by mean per-token KL(original || candidate) against a stored reference.

The candidate uses the checkpoint's serve settings or explicit overrides, with its kernels and activation
dtype checked before scoring. Workers reduce teacher-forced logits to per-position KL, top-1 flips and
target log-probabilities; the client sums these in fp64 in reference order.

The flat :class:`Score` combines :class:`~qda_kl.tally.Reading` fields with serving settings, timings and
provenance for the judge.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from pydantic import Field
from qda_quant.step import Refused

from qda_kl.config import check_checkpoint
from qda_kl.engine import Item, open_engine, teacher_force
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.reference import model_identity, read_prompts, versions, window
from qda_kl.serve import (
    ServeSettings,
    batch_args,
    card_kernels,
    cards,
    check_dtype,
    check_kernels,
    check_plugins,
    check_vllm,
    no_jit,
    resolve_dtype,
    settings_for,
    vllm_version,
)
from qda_kl.store import Reference
from qda_kl.tally import Reading, SeqSum, reading
from qda_kl.tally import combine as combine_readings
from qda_kl.tally import readings_of as readings_of
from qda_kl.tokenizer import check_tokenizer, declared_stops
from qda_kl.tokenizer_check import engine_tokenizer, preload_refusal, tokenizer_refusal
from qda_kl.vllm_hooks import merge_kernels


class Score(Reading):
    """Record a KL reading with its serving settings, timings and provenance."""

    kernels: dict[str, list[str]]
    #: The target table whose plan the bound kernels were checked against when it is not ``build.json``'s: this
    #: card's (`qda_kl.serve.card_kernels`); ``serve.kernels`` holds that plan.
    kernels_from: str | None = None
    vllm: str
    vllm_mismatch: str | None = None
    wall_s: float
    dtype: str
    dtype_from: str
    allow_dtype_mismatch: bool = False
    serve: ServeSettings
    tp: int
    checkpoint: dict
    reference: dict
    tokenizer_sha256: str = ""
    tokenizer_config: dict = Field(
        default_factory=dict,
        description="qda build's tokenizer record when the engine got the corrected class (build.json's shape)",
    )
    validate_max_abs_diff: float
    seconds: dict[str, float]
    versions: dict
    deterministic: bool | None = None
    repeats: dict[str, list[float | None]] | None = None
    spread: dict[str, float | None] | None = None


def combine(scores: list[Score]) -> Score:
    """Combine vLLM readings with their original domain and spread conventions."""
    return combine_readings(scores, native=False)


def sums_of(per_pos: dict[str, dict[str, np.ndarray]]) -> dict[str, SeqSum]:
    """Sum each sequence's per-position values in fp64, preserving their array order."""
    return {
        key: SeqSum(
            kl=float(r["kl"].astype(np.float64).sum()),
            tokens=int(r["kl"].shape[0]),
            flips=int(r["flip"].sum()),
            cand_logp=float(r["tgt"].astype(np.float64).sum()),
        )
        for key, r in per_pos.items()
    }


def _build_prompt(ref: Reference) -> str | None:
    """The first prompt text the reference was built from, when its prompt file is still readable.

    The prompt file is resolved by `qda_quant.calib.contamination.reference_prompt_file`, the one reader of
    "the prompt file a reference names": a cache entry records its own copy relative to itself, so a
    recorded path is not a path to resolve against the current directory.
    """
    from qda_quant.calib.contamination import reference_prompt_file

    try:
        first = read_prompts(reference_prompt_file(ref.path)[0])[:1]
    except (OSError, ValueError, KeyError, TypeError, Refused):
        return None
    return (first[0].prompt or first[0].text) if first else None


class Scorer:
    """Reuse one checked vLLM engine to score a checkpoint against a reference."""

    def __init__(
        self,
        checkpoint: str,
        ref_dir: str | Path,
        *,
        settings: ServeSettings | None = None,
        allow_vllm_mismatch: bool = False,
        allow_dtype_mismatch: bool = False,
        tp: int = DEFAULT_TP,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        subset: str | None = None,
    ):
        """Open ``checkpoint`` against ``ref_dir`` and validate its serving requirements.

        Raises:
            Refused: ``checkpoint`` is a missing local path (a failed build leaves none); the vLLM version
                differs without permission, the kernels differ, the vocabulary is too small, ``tp`` is
                invalid, ``settings.args`` names another ``tensor_parallel_size``, or the engine's tokenizer
                does not split text like the checkpoint's ``tokenizer.json``.
            RuntimeError: A required vLLM plugin is not installed (an environment error, not a verdict).
        """
        check_checkpoint(checkpoint)  # before the reference is read and an engine reserves a card
        t0 = time.time()
        self.tp = tp
        self.allow_dtype_mismatch = allow_dtype_mismatch
        self.checkpoint, self.ref = checkpoint, Reference(ref_dir, verify=True, subset=subset)
        s = settings if settings is not None else settings_for(checkpoint)
        s = no_jit(s)  # no FlashInfer kernel is JIT-built while scoring, as for a gencheck
        # the reference's sequences, not vLLM's default: a hybrid model holds one Mamba block per sequence
        s = s.model_copy(update={"args": batch_args(s.args, len(self.ref.sequences))})
        anchor_dtype = self.ref.manifest.model.get("dtype")
        s, self.dtype_from = resolve_dtype(s, anchor_dtype, allow_mismatch=allow_dtype_mismatch)
        # the kernels of this card's plan, which vLLM binds here, rather than the build target's
        kernels, self.kernels_from = card_kernels(checkpoint, s.kernels, cards(tp))
        s = s.model_copy(update={"kernels": kernels})
        self.settings, self.vllm = s, vllm_version()
        self.vllm_mismatch = check_vllm(s, self.vllm, allow_vllm_mismatch)
        check_plugins(s)
        longest = max(x.n_tokens for x in self.ref.sequences)
        # the build's class correction (a source checkpoint scored as it is, e.g. a full-precision baseline); the
        # directory lives as long as the scorer, so the engine can read it whenever it loads its tokenizer
        self._tokenizer_stage = tempfile.TemporaryDirectory(prefix="qda-kl-tokenizer-")
        engine_args, self.tokenizer_config = engine_tokenizer(
            checkpoint, s.args, Path(self._tokenizer_stage.name) / "tokenizer"
        )
        source = engine_args.get("tokenizer") or checkpoint
        revision = s.args.get("tokenizer_revision") or s.args.get("revision")
        prompt = _build_prompt(self.ref)
        options = {"trust_remote_code": True} if s.args.get("trust_remote_code") else {}
        if refusal := preload_refusal(source, revision=revision, probes=(prompt,), **options):
            raise Refused(refusal)  # before the weights load
        self.llm = open_engine(
            checkpoint,
            tp=tp,
            max_model_len=window(checkpoint, longest + 16),  # a 4k model's answers can fill its window
            settings=s.model_copy(update={"args": engine_args}),
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.dtype = str(self.llm.llm_engine.model_config.dtype).removeprefix("torch.")
        check_dtype(self.dtype, anchor_dtype, allow_dtype_mismatch)
        self.kernels = merge_kernels(self.llm.collective_rpc("qda_kl_kernels"))
        check_kernels(s.kernels, self.kernels)
        vocab = int(self.llm.llm_engine.model_config.get_vocab_size())
        if vocab < self.ref.manifest.vocab:
            raise Refused(f"{checkpoint}: vocabulary {vocab} < the reference's {self.ref.manifest.vocab}")
        processor = getattr(self.llm.llm_engine, "input_processor", None)
        stops = getattr(processor, "generation_config_fields", {}).get("eos_token_id")
        if not stops:  # its generation_config.json names no EOS (a published file may hold only sampling)
            stops = declared_stops(checkpoint, s.args.get("revision")) or []
        recorded = self.ref.manifest.build.get("sampling", {}).get("stop_token_ids")
        self.tokenizer_sha256 = check_tokenizer(
            self.ref.manifest.model,
            self.llm.get_tokenizer(),
            copy=self.ref.tokenizer,  # the reference's own anchor tokenizer: the model directory may be gone
            original_stop_ids=recorded,
            stop_ids=stops,
        )
        refusal = tokenizer_refusal(self.llm.get_tokenizer(), source, revision=revision)
        if not refusal and prompt:
            refusal = tokenizer_refusal(self.llm.get_tokenizer(), source, revision=revision, probe=prompt)
        if refusal:
            raise Refused(refusal)
        self.load_s = time.time() - t0

    def score(self, *, check: bool = True, group: int = 1) -> Score:
        """Score the checkpoint against the reference.

        Args:
            check: validate the hook against the engine's own prompt log-probs.
            group: Sequences per engine call (:func:`qda_kl.engine.teacher_force`); 1 keeps each sequence's
                batch shapes fixed. Larger groups can change floating-point results.
        """
        t0 = time.time()
        ref, m = self.ref, self.ref.manifest
        items = [Item(s.id, ref.tokens(s), s.start) for s in ref.sequences]
        per_pos = teacher_force(self.llm, "score", items, ref_path=str(ref.path), check=check, group=group)
        rd = reading(ref.sequences, sums_of(per_pos))
        if rd.tokens == 0:
            raise Refused(f"{ref.path}: no answer tokens scored; a reading needs the anchor's answers")
        wall = time.time() - t0
        diffs = [float(r.get("validate_max_abs_diff", 0.0)) for r in per_pos.values()]
        return Score(
            **rd.model_dump(),
            kernels=self.kernels,
            kernels_from=self.kernels_from,
            vllm=self.vllm,
            vllm_mismatch=self.vllm_mismatch,
            wall_s=round(wall, 2),
            dtype=self.dtype,
            dtype_from=self.dtype_from,
            allow_dtype_mismatch=self.allow_dtype_mismatch,
            serve=self.settings,
            tp=self.tp,
            checkpoint=model_identity(self.checkpoint),
            tokenizer_sha256=self.tokenizer_sha256,
            tokenizer_config=self.tokenizer_config,
            reference={
                "subset": ref.subset,
                "path": str(ref.path),
                "content_sha256": ref.content_sha256,
                "k": m.k,
                "positions": m.positions,
                "model": m.model,
                "tail_mass": m.tail_mass,
            },
            validate_max_abs_diff=max(diffs, default=0.0),
            seconds={"load": round(self.load_s, 1), "score": round(wall, 1)},
            versions=versions(),
        )


def score(checkpoint: str, ref_dir: str | Path, *, check: bool = True, group: int = 1, **kw) -> Score:
    """Score ``checkpoint`` against ``ref_dir`` once, passing ``kw`` to :class:`Scorer`."""
    return Scorer(checkpoint, ref_dir, **kw).score(check=check, group=group)
