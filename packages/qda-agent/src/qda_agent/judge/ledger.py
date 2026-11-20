"""The judge's ledger: one `Entry` per frozen checkpoint, and what the agent may read of it.

Every entry is appended to `judge/ledger.jsonl` on each change. The agent sees only the lines built here: an
entry's status, its reading or long-generation counts, and a reason with host paths cut.

The file holds plain records beside the entries (a prune, a hand-in refused for disk space), so a reader
tells them apart by `kind`: an entry states `handin`, `scored` or `gencheck`, a record other names, and a
resumed host reads both back without parsing a record as an `Entry`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..host import HandIn, label

#: Retained for readings in older ledgers.
SCREENING_ONLY = "a screening reading only: not a reading of the full reference"
#: The `kind`s an entry line states (`Entry.kind`); every other `kind` is a plain record beside the entries.
ENTRY_KINDS = frozenset({"handin", "scored", "gencheck"})

HOST_PATH = re.compile(r"(?<![\w.~-])/[^\s'\"`,;:)\]]+")
#: A domain name the agent may read in a long-generation result (anything else is dropped).
DOMAIN = re.compile(r"[A-Za-z0-9][\w.-]{0,31}", re.ASCII)
#: What a long-generation result says of one domain, the only fields the agent sees.
GEN_FIELDS = ("n", "cut", "loops", "mean_tokens")
#: How `Entry.gen_line` names each count.
GEN_LABELS = (("cut", "cut"), ("loops", "loops"), ("tokens", "mean_tokens"))
#: The reason of a long-generation result that is not a full report of both models.
INCOMPLETE = "an incomplete long-generation report"

Kind = Literal["handin", "scored", "gencheck"]


def utc(t: float) -> str:
    """`t` (epoch seconds) as `HH:MM:SS UTC`."""
    return datetime.fromtimestamp(t, UTC).strftime("%H:%M:%S UTC")


def neutral(text: str) -> str:
    """`text` with absolute paths cut (what the agent may read of a judge's message)."""
    return HOST_PATH.sub("<path>", text)[:300]


def _docs(path: Path) -> Iterator[dict[str, Any]]:
    """The JSON objects a ledger holds, in file order; a malformed line (a torn append) is skipped."""
    try:
        stream = Path(path).open()
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            if isinstance(doc, dict):
                yield doc


def entries(path: Path) -> list[Entry]:
    """The latest snapshot of every entry a ledger holds, oldest first.

    A record line (a prune, a hand-in refused for disk space) is skipped, not parsed as an `Entry`: its
    `kind` is none of `Entry.kind`'s, so a resumed judge reads both from one file.
    """
    latest: dict[str, Entry] = {}
    for doc in _docs(path):
        if doc.get("kind") in ENTRY_KINDS:
            entry = Entry.model_validate(doc)
            latest[entry.id] = entry  # the newest snapshot, at the position the entry first appeared
    return list(latest.values())


def space_refusals(path: Path) -> list[dict[str, Any]]:
    """The `space_refusal` records of a ledger, oldest first (what a resumed host rebuilds its list from)."""
    return [doc for doc in _docs(path) if doc.get("kind") == "space_refusal"]


class Entry(BaseModel):
    """One frozen checkpoint the judge scored, checked or generated with (host-side record)."""

    id: str
    kind: Kind
    src: str  # as the agent named it, relative to the workdir
    dir: Path  # the frozen copy
    digest: str
    submitted: float
    ended: float | None = None  # when the judge recorded its verdict or result
    job: str | None = None
    out: Path | None = None
    valid: bool | None = None
    #: A valid entry a final rule may still not pick (`judge.require_build_receipt`: no build matches its files).
    eligible: bool = True
    metrics: dict[str, float] = {}  # a reading's metrics, empty unless it has a finite `judge.metric`
    higher_is_better: dict[str, bool] = {}
    domains: dict[str, float] = {}  # `kl` per domain
    domains_text: dict[str, float] = {}  # `kl_text` per domain
    gen: dict[str, dict[str, dict[str, float]]] = {}  # {checkpoint|original: {domain: GEN_FIELDS}}
    max_tokens: int | None = None  # a long-generation check's answer cap, in tokens
    per_domain: int | None = Field(default=None, exclude_if=lambda v: v is None)  # its prompts per domain
    reason: str = ""  # why it was refused, or why it is valid but ineligible
    build_receipt: dict = {}
    #: `screen` while this entry's reading is a screening subset, `full` once it reads all of it; `None` (every
    #: current search) is one reading of the full reference. Only an older search, under a screening setting
    #: since removed, wrote the other two; a ledger it left is still read.
    stage: Literal["screen", "full"] | None = Field(default=None, exclude_if=lambda v: v is None)
    #: The screening reading's metrics, kept after promotion so the rank is re-read as candidates arrive.
    screen_metrics: dict[str, float] = Field(default_factory=dict, exclude_if=lambda v: not v)
    #: A llama.cpp candidate's measured `.gguf` bytes: `file` (the frozen file's `os.stat` size) and
    #: `payload` (its stored tensors, without the header). Both recorded; `judge.budget_basis` screened one.
    sizes: dict[str, int] = {}
    #: The `handin` call (its list) this entry was submitted in, and its rank in it (0 = best first).
    list_id: int | None = Field(default=None, exclude_if=lambda v: v is None)
    slot: int = Field(default=0, exclude_if=lambda v: v == 0)
    #: A judge job the drain window cut before it could start: no verdict on the checkpoint (its reason says so).
    cut: bool = Field(default=False, exclude_if=lambda v: not v)
    #: A baseline reading the run ranks as a pre-built start (`judge.starts`): its label and origin.
    name: str = Field(default="", exclude_if=lambda v: not v)
    origin: Literal["own", "gsq"] | None = Field(default=None, exclude_if=lambda v: v is None)
    #: The start a candidate is compared with (lowest valid KL); its name and reading are in this record.
    bar: bool = Field(default=False, exclude_if=lambda v: not v)

    def record(self) -> HandIn:
        """The entry as the flow sees it (a `gencheck` entry is no submitted checkpoint).

        A screening reading (in a ledger an older, screening search wrote) is reported as such and is never
        eligible: only full readings decide, and a candidate awaiting its full reading has not been read in full.
        """
        kind = self.kind
        assert kind != "gencheck", "a long-generation check is not a submitted checkpoint"
        screen = self.stage == "screen"
        return HandIn(
            id=self.id,
            dir=self.dir,
            submitted_at=datetime.fromtimestamp(self.submitted, UTC),
            kind=kind,
            valid=self.valid,
            eligible=self.eligible and not screen,
            metrics=self.metrics,
            higher_is_better=self.higher_is_better,
            reason=self.reason or (SCREENING_ONLY if screen and self.valid else ""),
            build_receipt=self.build_receipt,
            stage=self.stage,
            sizes=self.sizes,
            handin_list=self.list_id,
            slot=self.slot,
            name=self.name,
            origin=self.origin,
            bar=self.bar,
        )

    def status(self) -> str:
        """`checking` / `screening` / `reading` / `generating`, then `valid` / `ineligible` / `refused` / `cut`."""
        screen = self.stage == "screen"
        if self.valid is None:
            if screen:
                return "screening"
            return {"handin": "checking", "scored": "reading", "gencheck": "generating"}[self.kind]
        if self.cut:  # the drain window cut the job before it ran: no verdict on the checkpoint
            return "cut"
        if self.kind == "gencheck":
            return "done" if self.valid else "failed"
        if not self.valid:
            return "refused"
        if screen:
            return "screening"
        return "valid" if self.eligible else "ineligible"

    def metric_line(self) -> str:
        """Each metric of the reading, its per-domain numbers and a direction that is not lower-is-better."""
        per_domain = {"kl": self.domains, "kl_text": self.domains_text}
        parts = []
        for name, value in self.metrics.items():
            doms = ", ".join(f"{k} {v:.4g}" for k, v in sorted(per_domain.get(name, {}).items()))
            parts.append(
                f"{label(name)} {value:.4g}"
                + (" (higher is better)" if self.higher_is_better.get(name) else "")
                + (f" (per domain: {doms})" if doms else "")
            )
        text = "; ".join(parts)
        return f"screening reading: {text}" if self.stage == "screen" and text else text

    def gen_line(self) -> str:
        """Per-domain counts of the checkpoint against the original (cut, loops, mean tokens); nothing else."""
        mine, orig = self.gen.get("checkpoint", {}), self.gen.get("original", {})
        if not mine:
            return ""
        parts = [
            f"{d} ({mine[d].get('n', 0):.0f}): "
            + ", ".join(f"{label} {_vs(mine[d], orig.get(d, {}), k)}" for label, k in GEN_LABELS)
            for d in sorted(mine)
        ]
        head = "checkpoint vs original" + (f", max {self.max_tokens} tokens" if self.max_tokens else "")
        return f"{head}; " + "; ".join(parts)

    def said(self) -> str:
        """`ID SRC: status (reason)`, the reply to the command that made the entry."""
        return f"{self.id} {self.src}: {self.status()}" + (f" ({self.reason})" if self.reason else "")

    def line(self) -> str:
        """The entry's line in `kl list`, `gencheck list` and `handin status`."""
        tail = self.gen_line() or self.metric_line()
        if self.valid and not self.eligible and self.reason:
            tail = f"{tail}; {self.reason}" if tail else self.reason
        elif not tail:
            tail = self.reason
        when = ""
        # checks take minutes and run all through a search: when each was asked and ended
        if self.kind == "gencheck":
            when = f" (asked {utc(self.submitted)}" + (
                f", finished {utc(self.ended)})" if self.ended else ")"
            )
        return f"{self.id} {self.src}: {self.status()}{when}" + (f": {tail}" if tail else "") + "\n"


def _vs(mine: dict[str, float], orig: dict[str, float], k: str) -> str:
    """`5 vs 3`: the checkpoint's count against the original's (only the first when the original has none)."""
    return f"{mine.get(k, 0):.0f}" + (f" vs {orig[k]:.0f}" if k in orig else "")


def finite(x: object) -> float | None:
    """`x` as a float when it is a finite number (not a bool); else None."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def floats(d: object) -> dict[str, float]:
    """The finite float values of a {name: number} mapping (anything else dropped)."""
    if not isinstance(d, dict):
        return {}
    return {str(k): f for k, v in d.items() if (f := finite(v)) is not None}


def _gen_side(side: object) -> tuple[dict[str, dict[str, float]] | None, str]:
    """One side of a `gencheck.json` (`{domain: counts}`) as plain-domain `GEN_FIELDS`, and why it is not one.

    None and the reason when the side is incomplete ("" when it is). A domain whose name is not plain is
    dropped (it is never shown); any shown domain must have every field finite, integer `n`, `cut` and
    `loops`, `n >= 1`, `0 <= cut, loops <= n` and `mean_tokens >= 0`.
    """
    if not isinstance(side, dict):
        return None, INCOMPLETE
    out: dict[str, dict[str, float]] = {}
    for d, v in side.items():
        if not (isinstance(d, str) and DOMAIN.fullmatch(d)):
            continue
        if not isinstance(v, dict):
            return None, INCOMPLETE
        row: dict[str, float] = {}
        for k in GEN_FIELDS:
            if (f := finite(v.get(k))) is None:
                return None, INCOMPLETE
            row[k] = f
        n, cut, loops, tokens = (row[k] for k in GEN_FIELDS)
        if n < 1:
            return None, f"{d}: n {n:g} is not a positive count"
        for k in ("cut", "loops"):
            if not 0 <= row[k] <= n:
                return None, f"{d}: {k} {row[k]:g} is not between 0 and n {n:g}"
        if tokens < 0:
            return None, f"{d}: mean_tokens {tokens:g} is negative"
        for k in ("n", "cut", "loops"):
            if row[k] != int(row[k]):
                return None, f"{d}: {k} {row[k]:g} is not an integer"
        out[d] = row
    return (out, "") if out else (None, INCOMPLETE)


def gen_report(
    got: dict, per_domain: int | None = None
) -> tuple[dict[str, dict[str, dict[str, float]]] | None, str]:
    """Both sides of a `gencheck.json`, and why it is not a complete paired report ("" when it is).

    Both sides cover the same domains, with integer counts, the same `n` in each domain on both sides and,
    when `per_domain` is known, `n` equal to the count the check asks for in every domain.
    """
    checkpoint, why = _gen_side(got.get("checkpoint"))
    if checkpoint is None:
        return None, why
    original, why = _gen_side(got.get("original"))
    if original is None:
        return None, why
    if set(checkpoint) != set(original):
        return None, INCOMPLETE
    for d, mine in checkpoint.items():
        if mine["n"] != original[d]["n"]:
            return None, f"{d}: n {mine['n']:g} vs {original[d]['n']:g} (checkpoint vs original)"
        if per_domain is not None and mine["n"] != per_domain:
            return None, f"{d}: n {mine['n']:g}, the check asks {per_domain} prompts"
    return {"checkpoint": checkpoint, "original": original}, ""
