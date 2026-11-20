"""Aggregate per-sequence KL sums into token-weighted readings.

Each :class:`SeqSum` holds summed per-position KL, token count, top-1 flips and the candidate's summed
target log-likelihood. The reading (:func:`reading`) reports the paper's mean per-token KL:

* ``kl``: over every scored token of the anchor's own answers;
* ``domains``: the same, per prompt domain;
* ``kl_text``: over the human-written texts (``None`` when the reference has none), and ``domains_text`` per
  text domain.

Sums use fp64 and reference sequence order, so aggregation of the same per-position values is independent
of request batching.

Each sequence's own reading (:attr:`Reading.per_seq`) keeps its token count and the reference row's ``group``,
``domain`` and ``split``, because the eval table's bootstrap resamples sequences and needs those weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:  # only an annotation: importing store would pull torch into the table's reader
    from qda_kl.store import Sequence


class SeqSum(BaseModel):
    """One sequence's sums over its scored positions."""

    kl: float
    tokens: int
    flips: int = 0
    cand_logp: float = 0.0


class SeqKL(BaseModel):
    """One sequence's mean per-token KL and the reference row's labels the report groups it by."""

    kl: float
    tokens: int
    domain: str = ""
    group: str = ""
    split: str = ""
    kind: Literal["answer", "text"] = "answer"


class Reading(BaseModel):
    """A score: mean per-token KL(anchor || candidate) in nats."""

    kl: float
    tokens: int
    domains: dict[str, float]
    domain_tokens: dict[str, int]
    kl_text: float | None = None
    text_tokens: int = 0
    domains_text: dict[str, float] = {}
    domain_text_tokens: dict[str, int] = {}
    flip_rate: float = math.nan
    per_seq: dict[str, SeqKL] = {}


T = TypeVar("T", bound=Reading)


def readings_of(s: Reading) -> dict[str, float | None]:
    """Return the aggregate KL values compared across repeats, including answer and text domains."""
    return {
        "kl": s.kl,
        "kl_text": s.kl_text,
        **{f"domains.{k}": v for k, v in s.domains.items()},
        **{f"domains_text.{k}": v for k, v in s.domains_text.items()},
    }


def combine(scores: list[T], *, native: bool) -> T:
    """Combine repeats using the engine's domain, precision and zero-spread conventions."""
    first = scores[0]
    vals = [{"kl": s.kl, "kl_text": s.kl_text} if native else readings_of(s) for s in scores]
    keys = list(vals[0])
    if any(list(v) != keys for v in vals):
        raise ValueError("repeats scored different domains")
    same = len({(s.kl, s.kl_text) for s in scores}) == 1 if native else all(v == vals[0] for v in vals)

    def values(k: str) -> list[float] | None:
        xs = [v[k] for v in vals]
        if any(x is None for x in xs):
            return None
        return [float(x) if native else x for x in xs if x is not None]

    def mean(k: str) -> float | None:
        xs = values(k)
        return None if xs is None else float(np.mean(xs))

    def spread(k: str) -> float | None:
        xs = values(k)
        if xs is None:
            return None
        m, width = abs(float(np.mean(xs))), max(xs) - min(xs)
        if m == 0:
            return None if native or width else 0.0
        return width / m

    update = {
        "deterministic": same,
        "repeats": {k: [v[k] for v in vals] for k in ("kl", "kl_text")},
        "spread": {k: spread(k) for k in ("kl", "kl_text")},
    }
    if native:
        update["repeat"] = len(scores)
    if not same:
        update |= {"kl": mean("kl"), "kl_text": mean("kl_text")}
        if not native:
            update |= {
                "domains": {k: mean(f"domains.{k}") for k in first.domains},
                "domains_text": {k: mean(f"domains_text.{k}") for k in first.domains_text},
            }
    return first.model_copy(update=update)


def reading(seqs: Iterable[Sequence], sums: dict[str, SeqSum]) -> Reading:
    """Aggregate per-sequence sums (keyed by sequence id) into a :class:`Reading`.

    Raises:
        ValueError: a sequence has no sum, or a sum is not finite (a reading is never made of part of the data).
    """
    kl = tok = flips = 0.0
    dom: dict[str, list[float]] = {}
    text: dict[str, list[float]] = {}
    per = {}
    for s in seqs:
        g = sums.get(s.id)
        if g is None:
            raise ValueError(f"{s.id}: not scored")
        if not math.isfinite(g.kl) or g.tokens != s.n_scored:
            raise ValueError(f"{s.id}: KL sum {g.kl} over {g.tokens} tokens ({s.n_scored} expected)")
        per[s.id] = SeqKL(
            kl=g.kl / g.tokens,
            tokens=g.tokens,
            domain=s.domain,
            group=s.group,
            split=s.split,
            kind=s.kind,
        )
        if s.kind == "answer":
            kl += g.kl
            tok += g.tokens
            flips += g.flips
        d = (dom if s.kind == "answer" else text).setdefault(s.domain, [0.0, 0.0])
        d[0] += g.kl
        d[1] += g.tokens
    text_kl, text_tok = sum(v[0] for v in text.values()), sum(v[1] for v in text.values())
    return Reading(
        kl=kl / tok if tok else math.nan,
        tokens=int(tok),
        domains={k: v[0] / v[1] for k, v in sorted(dom.items())},
        domain_tokens={k: int(v[1]) for k, v in sorted(dom.items())},
        kl_text=text_kl / text_tok if text_tok else None,
        text_tokens=int(text_tok),
        domains_text={k: v[0] / v[1] for k, v in sorted(text.items())},
        domain_text_tokens={k: int(v[1]) for k, v in sorted(text.items())},
        flip_rate=flips / tok if tok else math.nan,
        per_seq=per,
    )
