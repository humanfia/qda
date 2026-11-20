"""Reduce teacher-forced logits inside vLLM workers without transferring full-vocabulary rows.

Prompts contain the full token sequence and a ``SamplingParams.extra_args["qda_kl"]`` tag with its key and
first scored token index. Prefix caching is skipped so every position is recomputed.
The extension wraps the V2 runner's ``compute_prompt_logprobs`` or V1 runner's
``_get_prompt_logprobs_dict``, locates each tagged request's hidden-state rows and calls the served model's
``compute_logits`` on the engine's own prompt-logprob blocks when the runner exposes them -- :func:`engine_chunk`
rows counted from row 0 of the step's batch buffer, the last block cut at the step's row count -- and otherwise in
chunks of :data:`CHUNK` rows from the first scored row. The same rows at the same shape matter because a quantized
lm-head's fp32 accumulation order follows the block shape, so a different blocking moves rare log-probabilities by
more than the client's misalignment tolerance. It reduces the logits on the card:

* ``ref`` mode (the anchor): the top-k log-probabilities, their ids and the log of the remaining mass;
* ``score`` mode (a candidate): eq. (kltopk) against the reference's stored top-k, plus top-1 flips and the
  candidate's log-probability of the target token, kept per position.

``compute_logits`` gathers the sharded lm-head across the tensor-parallel group, so every rank makes the same
extra calls for the same chunk shapes; on CUDA that gather is an all-gather, so every rank holds the full row.
Only tensor-parallel rank 0 of the last pipeline stage (with pipeline parallelism 1, rank 0) keeps the results and
leaves them through :meth:`KLWorker.qda_kl_end`; the others return None. The client checks target
log-probabilities against vLLM's own ``prompt_logprobs`` to detect row misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from qda_kl.topk import CHUNK, TopK, anchor_topk, kl_topk


def engine_chunk() -> int:
    """The engine's prompt-logprob block: the hook asks the lm-head for the same rows at the same shape.

    Read in the worker, not at import: importing vLLM's GPU worker loads the CUDA stack, and every `qda-agent`
    command imports this module through `qda_kl.engine`.
    """
    try:
        from vllm.v1.worker.gpu.sample.prompt_logprob import CHUNK_SIZE
    except ImportError:  # a vLLM without that module, or a test environment without vLLM
        return 1024
    return CHUNK_SIZE


#: The worker extension's import path (``LLM(worker_extension_cls=EXTENSION)``).
EXTENSION = "qda_kl.vllm_hooks.KLWorker"
#: The key of a tagged request's ``SamplingParams.extra_args``.
TAG = "qda_kl"


@dataclass
class _Track:
    """One tagged sequence's per-position results, filled as its rows go by."""

    start: int
    tokens: torch.Tensor  # [n] int64, on the card
    anchor: TopK | None = None  # score mode: the reference's top-k at its scored positions, on the card
    out: dict[str, torch.Tensor] = field(default_factory=dict)
    seen: torch.Tensor | None = None  # [n_scored] bool


@dataclass
class _State:
    mode: str = ""
    k: int = 0
    ref: Any = None  # score mode: qda_kl.store.Reference
    tracks: dict[str, _Track] = field(default_factory=dict)
    tags: dict[str, dict] = field(default_factory=dict)  # V2: req_id -> tag (from add_request)
    installed: str = ""


def _segments(
    hook: _State, kind: str, owner: Any, args: tuple, kwargs: dict
) -> list[tuple[str, int, int, int, int]]:
    """Locate each tagged request as (key, first row, end row, first position, step rows) in this step."""
    out = []
    if kind == "v2":
        batch = kwargs.get("input_batch", args[2] if len(args) > 2 else None)
        assert batch is not None  # the v2 entry point receives the step's input batch
        q, done = batch.query_start_loc_np, batch.num_computed_prefill_tokens_np
        n_rows = int(batch.num_tokens)  # the engine's blocks are counted from this buffer's row 0
        for i, req_id in enumerate(batch.req_ids):
            tag = hook.tags.get(req_id)
            if tag is not None:
                out.append((tag["key"], int(q[i]), int(q[i + 1]), int(done[i]), n_rows))
    else:
        scheduled = kwargs.get("num_scheduled_tokens", args[1] if len(args) > 1 else None)
        assert scheduled is not None  # both entry points receive the step's scheduled tokens
        for req_id, n in scheduled.items():
            req = owner.requests.get(req_id)
            if req is None:
                continue
            extra = req.sampling_params.extra_args if req.sampling_params else None
            tag = (extra or {}).get(TAG)
            if tag is not None:
                lo = int(owner.query_start_loc.np[owner.input_batch.req_id_to_index[req_id]])
                # v1 keeps the hook's own reduction chunks: there is no step buffer to align blocks to
                out.append((tag["key"], lo, lo + int(n), int(req.num_computed_tokens), 0))
    return out


class KLWorker:
    """Extend vLLM's GPU worker with KL methods called through ``collective_rpc``."""

    #: Set by vLLM's worker before it calls any extension RPC method.
    model_runner: Any

    def _qda_kl(self) -> _State:
        st = getattr(self, "_qda_kl_state", None)
        if st is None:
            st = self._qda_kl_state = _State()
        return st

    def _qda_kl_reader(self) -> bool:
        """Return whether this is tensor-parallel rank 0 of the last pipeline stage: the rank that keeps results."""
        from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0 and get_pp_group().is_last_rank

    def _qda_kl_last_pp(self) -> bool:
        """Return whether this rank is in the last pipeline stage, the only one that computes logits."""
        from vllm.distributed import get_pp_group

        return get_pp_group().is_last_rank

    def qda_kl_install(self) -> str:
        """Wrap the runner's prompt-logprob entry point (once); returns the runner kind, ``v1`` or ``v2``.

        A pipeline stage before the last has no prompt-logprob entry point (and computes no logits); it returns
        ``""`` so pipeline-parallel scoring can still start.
        """
        st, runner = self._qda_kl(), self.model_runner
        if st.installed:
            return st.installed
        if getattr(runner, "prompt_logprobs_worker", None) is not None:
            kind, owner, name = "v2", runner.prompt_logprobs_worker, "compute_prompt_logprobs"
            add, remove = owner.add_request, owner.remove_request

            def add_request(req_id, req_idx, sampling_params):
                tag = (sampling_params.extra_args or {}).get(TAG)
                if tag is not None:
                    st.tags[req_id] = tag
                return add(req_id, req_idx, sampling_params)

            def remove_request(req_id):
                st.tags.pop(req_id, None)
                return remove(req_id)

            owner.add_request, owner.remove_request = add_request, remove_request
        elif hasattr(runner, "_get_prompt_logprobs_dict"):
            kind, owner, name = "v1", runner, "_get_prompt_logprobs_dict"
        elif not self._qda_kl_last_pp():
            return ""  # a pipeline stage before the last computes no logits
        else:
            raise RuntimeError("no prompt-logprob entry point on this vLLM model runner")
        original = getattr(owner, name)
        hidden_at = 1 if kind == "v2" else 0

        def entry(*args, **kwargs):
            if st.tracks:
                hidden = kwargs.get("hidden_states", args[hidden_at] if len(args) > hidden_at else None)
                assert hidden is not None  # both entry points receive the step's hidden states
                read = self._qda_kl_reader()
                for key, lo, hi, pos0, n_rows in _segments(st, kind, owner, args, kwargs):
                    if key in st.tracks:
                        self._qda_kl_take(st, st.tracks[key], hidden, lo, hi, pos0, n_rows, read)
            return original(*args, **kwargs)

        setattr(owner, name, entry)
        st.installed = kind
        return kind

    def qda_kl_kernels(self) -> dict[str, list[str]]:
        """Return bound kernel classes by kind: ``dense`` for quantized linears, ``moe`` for fused experts."""
        return bound_kernels(self.model_runner.model)

    def qda_kl_begin(
        self, mode: str, seqs: dict[str, tuple[int, list[int]]], k: int = 0, ref_path: str | None = None
    ) -> bool:
        """Start reading ``seqs`` ({key: (start, tokens)}); returns whether this rank reads.

        ``ref`` keeps the top-``k``; ``score`` compares with the reference at ``ref_path`` (keys are its sequence
        ids). Every rank tracks the sequences, because the lm-head gather makes every rank run the extra calls;
        only the reader reduces them.
        """
        if mode not in ("ref", "score"):
            raise ValueError(f"mode {mode!r}")
        st = self._qda_kl()
        st.mode, st.k, st.tracks = mode, k, {}
        dev = self.model_runner.device
        if mode == "score":
            from qda_kl.store import Reference

            assert ref_path is not None  # score mode always names the reference directory
            if st.ref is None or str(st.ref.path) != ref_path:
                st.ref = Reference(ref_path)
            by_id = {s.id: s for s in st.ref.sequences}
        for key, (start, tokens) in seqs.items():
            n = len(tokens) - start
            t = _Track(start=start, tokens=torch.tensor(tokens, dtype=torch.long, device=dev))
            t.seen = torch.zeros(n, dtype=torch.bool, device=dev)
            t.out["tgt"] = torch.full((n,), float("nan"), device=dev)
            if mode == "ref":
                t.out.update(
                    topv=torch.empty(n, k, device=dev),
                    topi=torch.empty(n, k, dtype=torch.int32, device=dev),
                    log_rest=torch.empty(n, device=dev),
                )
            else:
                t.anchor = st.ref.topk(by_id[key], dev)
                t.out.update(
                    kl=torch.full((n,), float("nan"), device=dev),
                    flip=torch.zeros(n, dtype=torch.bool, device=dev),
                )
            st.tracks[key] = t
        return self._qda_kl_reader()

    def _qda_kl_take(
        self,
        st: _State,
        t: _Track,
        hidden: torch.Tensor,
        lo: int,
        hi: int,
        pos0: int,
        n_rows: int,
        read: bool,
    ) -> None:
        """Read rows lo..hi of this step (positions pos0..) of one tracked sequence.

        Every rank makes the same lm-head calls, whose gather is collective; only ``read`` reduces them.
        ``n_rows`` is the step's row count (0 on v1, which has no step buffer): with it the calls are the engine's
        own prompt-logprob blocks -- :func:`engine_chunk` rows counted from row 0 of the step, the last cut at
        ``n_rows`` -- so the same rows meet the same lm-head shape and a quantized lm-head accumulates them the same
        way. Blocks with no scored row of this sequence are read only if the engine would compute them anyway; the
        engine computes every block of the step, so the selection below skips only the reduction.
        """
        n_tok = t.tokens.shape[0]
        assert t.seen is not None  # qda_kl_begin allocates it with the track
        # position p predicts token p + 1; scored positions are start - 1 .. n_tok - 2, i.e. rows a..b-1 here
        a, b = max(pos0, t.start - 1), min(pos0 + hi - lo, n_tok - 1)
        compute_logits = self.model_runner.model.compute_logits
        if n_rows:
            first, last = lo + a - pos0, lo + b - pos0  # this sequence's scored rows in the step buffer
            block = engine_chunk()
            spans = [(o, min(o + block, n_rows)) for o in range(first - first % block, last, block)]
        else:  # v1 has no step buffer: keep the reduction chunks from the first scored row
            spans = [(lo + s - pos0, lo + min(s + CHUNK, b) - pos0) for s in range(a, b, CHUNK)]
        for start, end in spans:
            logits = compute_logits(hidden[start:end])
            if logits is None or not read:  # the gather is collective; only the reader reduces it
                continue
            lp = logits.float().log_softmax(-1)
            s, e = max(a, pos0 + start - lo), min(b, pos0 + end - lo)  # the scored positions in this block
            if e <= s:  # a block that holds no scored row of this sequence
                del logits, lp
                continue
            lp = lp[s - (pos0 + start - lo) : e - (pos0 + start - lo)]
            r = slice(s - (t.start - 1), e - (t.start - 1))
            tgt = t.tokens[s + 1 : e + 1]
            t.out["tgt"][r] = lp.gather(1, tgt[:, None])[:, 0]
            if st.mode == "ref":
                top = anchor_topk(lp, st.k)
                t.out["topv"][r], t.out["topi"][r], t.out["log_rest"][r] = top
            else:
                assert t.anchor is not None  # qda_kl_begin loads it in score mode
                anchor = TopK(*(x[r] for x in t.anchor))
                t.out["kl"][r] = kl_topk(anchor, lp)
                t.out["flip"][r] = anchor.topi[:, 0].long() != lp.argmax(-1)
            t.seen[r] = True
            del logits, lp

    def qda_kl_end(self) -> dict[str, dict[str, tuple[str, list[int], bytes]]] | None:
        """Pack tracked sequences' per-position results, returning None on non-reading ranks.

        ``ref``: topv, topi, log_rest, tgt. ``score``: kl, flip, tgt. A sequence some of whose positions were not
        seen is returned with ``missing`` (their count) instead.
        """
        st = self._qda_kl()
        tracks, st.tracks = st.tracks, {}
        if not self._qda_kl_reader():
            return None
        out = {}
        for key, t in tracks.items():
            assert t.seen is not None  # qda_kl_begin allocates it with the track
            missing = int((~t.seen).sum())
            got = (
                {"missing": np.array(missing)} if missing else {n: v.cpu().numpy() for n, v in t.out.items()}
            )
            out[key] = {n: (a.dtype.str, list(a.shape), a.tobytes()) for n, a in got.items()}
        return out


def _kernel_name(module: Any) -> tuple[str, str] | None:
    """Return a layer's quantization kind and kernel class, or None for non-GEMM methods."""
    qm = getattr(module, "quant_method", None)
    # unquantized layers, and attention's KV-cache scales (CompressedTensorsKVCacheMethod, ...): not a GEMM kernel
    if qm is None or type(qm).__name__.startswith("Unquantized") or "KVCache" in type(qm).__name__:
        return None
    if hasattr(module, "w13_weight") or "MoE" in type(module).__name__:
        mk = getattr(qm, "moe_kernel", None)
        experts = getattr(mk, "fused_experts", None) or getattr(qm, "fused_experts", None)
        return "moe", type(experts or qm).__name__
    kernel = _dense_kernel(qm, getattr(module, "scheme", None))
    return "dense", type(kernel or qm).__name__


def _dense_kernel(qm: Any, scheme: Any) -> Any:
    """The kernel a dense method or scheme holds: vLLM 0.33's FP8 schemes name it `fp8_linear` (W8A8) or
    `linear_kernel` (W8A16), the others `kernel`."""
    for holder in (qm, scheme):
        for name in ("kernel", "fp8_linear", "linear_kernel"):
            if (k := getattr(holder, name, None)) is not None:
                return k
    return None


def bound_kernels(model: Any) -> dict[str, list[str]]:
    """Collect sorted, distinct kernel classes by quantization kind from a torch model."""
    got: dict[str, set[str]] = {}
    for _, m in model.named_modules():
        kn = _kernel_name(m)
        if kn is not None:
            got.setdefault(kn[0], set()).add(kn[1])
    return {k: sorted(v) for k, v in sorted(got.items())}


def merge_kernels(per_rank: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Union each kind's kernel classes over the engine's ranks: pipeline stages hold different layers."""
    got: dict[str, set[str]] = {}
    for rank in per_rank:
        for kind, names in rank.items():
            got.setdefault(kind, set()).update(names)
    return {k: sorted(v) for k, v in sorted(got.items())}


def unpack(packed: dict[str, tuple[str, list[int], bytes]]) -> dict[str, np.ndarray]:
    """Reconstruct a sequence's arrays from the raw bytes returned by :meth:`KLWorker.qda_kl_end`."""
    return {n: np.frombuffer(b, dtype=np.dtype(d)).reshape(s).copy() for n, (d, s, b) in packed.items()}
