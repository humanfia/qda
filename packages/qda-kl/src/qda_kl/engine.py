"""Run teacher-forced sequences through a vLLM engine with the KL worker extension.

Sequences run one at a time by default, because batching them changes the kernels' floating-point results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qda_quant.step import Refused

from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.serve import ServeSettings, check_servable, check_tp, llm, pinned
from qda_kl.vllm_hooks import EXTENSION, TAG, unpack

#: Parallel settings scoring refuses unless 1 (vllm.LLM keyword -> refusal name): they split sequences over
#: ranks, and the hook's one reading rank must see each sequence whole (tensor and pipeline parallel do).
PARALLEL = {
    "data_parallel_size": "data-parallel",
    "prefill_context_parallel_size": "prefill-context-parallel",
    "decode_context_parallel_size": "decode-context-parallel",
}


def check_parallel(kwargs: dict) -> None:
    """Refuse the effective engine arguments when any of :data:`PARALLEL` is not 1."""
    for key, name in PARALLEL.items():
        if int(kwargs.get(key, 1)) != 1:
            raise Refused(f"{name} scoring is not supported ({key}={kwargs[key]})")


#: Maximum absolute difference between hook and engine target log-probabilities at each position.
VALIDATE_TOL = 1e-3


@dataclass(frozen=True)
class Item:
    """One sequence to teacher-force: its key, tokens and the index of its first scored token."""

    key: str
    tokens: list[int]
    start: int


def open_engine(
    model: str,
    *,
    tp: int = DEFAULT_TP,
    max_model_len: int,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    seed: int = 0,
    settings: ServeSettings | None = None,
):
    """Open ``model`` (a directory, or ``org/name[@rev]``) in a ``vllm.LLM`` with the qda-kl worker extension.

    ``settings.env`` is the engine's start-up environment; ``settings.args`` override the defaults here except ``tp``.

    Raises:
        Refused: ``tp`` is below 1, ``settings.args`` names another ``tensor_parallel_size`` or another
            ``revision`` than ``@rev``, or another parallel size (:data:`PARALLEL`) spreads the engine over ranks.
    """
    if tp < 1:
        raise Refused(f"tp must be at least 1, got {tp}")
    s = settings or ServeSettings()
    model, args = pinned(model, s.args)  # ``org/name@rev`` loads ``org/name`` at ``revision=rev``
    check_servable(model, tp)  # the table's verdict, before an engine reserves a card and dies in vLLM
    kwargs = {
        "tensor_parallel_size": tp,
        "max_model_len": max_model_len,
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        **args,
    }
    check_tp(kwargs, tp)
    check_parallel(kwargs)  # after every setting is merged: serve.args may name a parallel size too
    engine = llm(s, kwargs, model=model, worker_extension_cls=EXTENSION)
    engine.collective_rpc("qda_kl_install")
    return engine


def _reader(results: list) -> dict:
    got = [r for r in results if r is not None]
    if len(got) != 1:
        raise Refused(f"{len(got)} reading ranks answered, expected 1")
    return got[0]


def teacher_force(
    llm,
    mode: str,
    items: Sequence[Item],
    *,
    k: int = 0,
    ref_path: str | None = None,
    group: int = 1,
    check: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Run teacher-forced sequences and return the hook's arrays keyed by sequence id.

    Args:
        llm: an engine from :func:`open_engine`.
        mode: ``ref`` (keep the top-``k``) or ``score`` (eq. kltopk against the reference at ``ref_path``).
        items: the sequences, read in this order.
        k: top-k kept in ``ref`` mode.
        ref_path: the reference directory in ``score`` mode.
        group: sequences per ``generate`` call (1: each sequence read alone, batch-independent).
        check: compare the hook's target log-probs with the engine's ``prompt_logprobs`` (costs one more
            lm-head pass over the scored rows).

    Raises:
        RuntimeError: a sequence not fully read, or a hook / engine mismatch above :data:`VALIDATE_TOL`.
    """
    from vllm import SamplingParams, TokensPrompt

    out: dict[str, dict[str, np.ndarray]] = {}
    for g in range(0, len(items), group):
        part = items[g : g + group]
        llm.collective_rpc(
            "qda_kl_begin", args=(mode, {i.key: (i.start, i.tokens) for i in part}, k, ref_path)
        )
        try:
            params = [
                SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    prompt_logprobs=0 if check else None,
                    skip_reading_prefix_cache=True,
                    detokenize=False,
                    extra_args={TAG: {"key": i.key, "start": i.start}},
                )
                for i in part
            ]
            said = llm.generate(
                [TokensPrompt(prompt_token_ids=i.tokens) for i in part], params, use_tqdm=False
            )
        finally:
            got = _reader(llm.collective_rpc("qda_kl_end"))
        for i, o in zip(part, said, strict=True):
            r = unpack(got[i.key]) if i.key in got else None
            if r is None or "missing" in r:
                raise RuntimeError(f"{i.key}: the hook did not see every scored position ({r})")
            if check:
                engine = np.array(
                    [_logprob(o.prompt_logprobs[p], i.tokens[p]) for p in range(i.start, len(i.tokens))]
                )
                diff = np.abs(engine - r["tgt"]).max()
                if not diff <= VALIDATE_TOL:
                    raise RuntimeError(
                        f"{i.key}: hook target log-probs differ from the engine's by {diff:.3g}"
                    )
                r["validate_max_abs_diff"] = np.array(diff)
            out[i.key] = r
    return out


def _logprob(entry: dict, token: int) -> float:
    one = entry.get(token) if entry else None
    return math.nan if one is None else float(one.logprob)
