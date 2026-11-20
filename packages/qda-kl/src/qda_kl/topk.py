"""The top-k KL estimator (paper eq. kltopk, Prop. 1) and its exact full-vocabulary reference.

At one position the anchor (the original model) has log-probabilities ``a`` and the candidate ``c`` over a
vocabulary of V tokens. The reference keeps only the anchor's top-k: their log-probabilities ``topv``, their ids
``topi`` and the log of the anchor's mass outside them ``log_rest``. The estimate is exact over those k tokens and
merges the rest of the vocabulary into one bucket::

    KL_k = sum_{i in top-k} p_a(i) (a_i - c_i)  +  m_a (log m_a - log m_c)

with ``m`` each side's mass outside the anchor's top-k, computed as a log-sum-exp over the masked row (never as
``1 - sum``, which cancels). Merging outcomes never increases KL (log-sum inequality), so ``KL_k <= KL`` and the
two are equal when k = V (Prop. 1). :func:`kl_exact` is the slow reference path the tests hold the estimator to.

All functions take fp32 log-probabilities with positions on the first axis and work on any device.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

#: The paper's k: the anchor's 128 most likely tokens per position, the rest one bucket.
TOPK = 128
#: Rows of a [T, V] block processed at a time (bounds the fp32 temporaries to CHUNK x V).
CHUNK = 512


class TopK(NamedTuple):
    """The anchor side of the estimate for T positions.

    Attributes:
        topv: [T, k] the anchor's k largest log-probabilities, descending.
        topi: [T, k] their token ids (int32).
        log_rest: [T] log of the anchor's mass outside its top-k (-inf when k covers the vocabulary).
    """

    topv: torch.Tensor
    topi: torch.Tensor
    log_rest: torch.Tensor


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute fp32 log-softmax over the last axis, matching vLLM's raw prompt log-probabilities."""
    return logits.float().log_softmax(dim=-1)


def log_outside(logp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per row, the log of the probability mass of ``logp`` outside the columns ``ids``.

    Args:
        logp: [T, V] log-probabilities.
        ids: [T, k] column ids to leave out.

    Returns:
        [T] log-sum-exp of the remaining columns (-inf when nothing remains).
    """
    return torch.logsumexp(logp.scatter(1, ids.long(), float("-inf")), dim=-1)


def anchor_topk(logp: torch.Tensor, k: int = TOPK) -> TopK:
    """Extract the anchor's top-k and remaining mass from each row of fp32 ``logp`` [T, V]."""
    v, i = logp.topk(min(k, logp.shape[-1]), dim=-1)
    return TopK(v, i.to(torch.int32), log_outside(logp, i))


def kl_topk(anchor: TopK, cand_logp: torch.Tensor) -> torch.Tensor:
    """Per-position KL_k(anchor || candidate): exact over the anchor's top-k, the rest one bucket.

    Args:
        anchor: the anchor's top-k.
        cand_logp: [T, V] the candidate's fp32 log-probabilities at the same positions.

    Returns:
        [T] fp32 KL in nats, a lower bound of :func:`kl_exact` (equal when k = V).
    """
    topv, topi = anchor.topv.float(), anchor.topi.long()
    log_rest = anchor.log_rest.float()
    kl = (topv.exp() * (topv - cand_logp.gather(1, topi))).sum(-1)
    cand_rest = log_outside(cand_logp, topi)
    # an anchor whose top-k holds all the mass: 0 log 0 = 0 (a NaN or +inf passes through, the caller refuses it)
    bucket = torch.where(
        torch.isneginf(log_rest), torch.zeros_like(kl), log_rest.exp() * (log_rest - cand_rest)
    )
    return kl + bucket


def kl_exact(anchor_logp: torch.Tensor, cand_logp: torch.Tensor) -> torch.Tensor:
    """Per-position full-vocabulary KL(anchor || candidate): the slow reference path.

    Args:
        anchor_logp: [T, V] the anchor's fp32 log-probabilities.
        cand_logp: [T, V] the candidate's.

    Returns:
        [T] fp32 KL in nats (terms with zero anchor probability contribute 0).
    """
    p = anchor_logp.exp()
    return torch.where(p > 0, p * (anchor_logp - cand_logp), torch.zeros_like(p)).sum(-1)


def chunks(n: int, size: int = CHUNK) -> list[slice]:
    """Return consecutive row slices of at most ``size`` covering ``range(n)``."""
    return [slice(s, min(n, s + size)) for s in range(0, n, size)]
