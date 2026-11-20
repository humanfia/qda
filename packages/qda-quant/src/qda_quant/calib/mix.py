"""Deterministic weighted mixing of calibration sources, with a pluggable contamination guard.

    python -m qda_quant.calib.mix --source a.jsonl:3 --source b.jsonl:1 --samples 128 --seed 0
        [--heldout ref-prompts.jsonl] --out mix.jsonl

``samples`` rows are split over the sources by weight, with largest remainders and ties broken by source
order. Each source's rows are shuffled with ``random.Random(f"{seed}:{i}")``. Rows the guard rejects are skipped,
and the first rows that pass are taken. The picked rows are then interleaved with ``random.Random(seed)``. The same
sources, weights, ``samples``, ``seed`` and guard always give the same file.

A guard is any callable ``row -> bool`` (keep). :class:`NGramGuard` drops a row that shares a word n-gram with a
held-out file, for example the KL reference prompts: calibration must not contain the evaluation prompts. Token-id
sources (``{"input_ids"}`` rows) must name their tokenizer (:func:`read_source`: a header row, a ``.tokenizer`` or
a ``.meta.json`` sidecar), all the same one. The result's ``.tokenizer`` sidecar carries it on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from ..step import Refused
from . import cache_root, meta_path, read_rows, row_text, sha256_file, write_jsonl


class Guard(Protocol):
    """A contamination guard: whether to keep a calibration row."""

    def __call__(self, row: dict) -> bool:
        """True to keep ``row``."""
        ...


def ngrams(text: str, n: int) -> set[int]:
    """Hashes of the lower-cased word ``n``-grams of ``text`` (stable across processes)."""
    w = re.findall(r"\w+", text.lower())
    return {
        int.from_bytes(hashlib.blake2b(" ".join(w[i : i + n]).encode(), digest_size=8).digest(), "little")
        for i in range(len(w) - n + 1)
    }


class NGramGuard:
    """Drop a row sharing any word ``n``-gram with a held-out JSONL's texts (``text``, ``prompt`` or ``messages``)."""

    def __init__(self, heldout: str | Path, n: int = 13):
        """Index ``heldout``'s word n-grams (n = 13 by default); a text shorter than n words is indexed whole.

        Raises:
            Refused: the held-out file cannot be read (missing, a directory, or not permitted).
        """
        self.path, self.n = Path(heldout), n
        try:
            self.sha256 = sha256_file(self.path)
            rows = read_rows(self.path)
        except OSError as e:
            raise Refused(f"calib mix heldout {self.path}: cannot read: {e}") from e
        self.grams: dict[int, set[int]] = {}  # gram width -> hashes (n, plus the widths of short texts)
        for r in rows:
            t = row_text(r)
            k = min(n, len(re.findall(r"\w+", t)))
            if k:
                self.grams.setdefault(k, set()).update(ngrams(t, k))
        self.dropped = 0

    def __call__(self, row: dict) -> bool:
        """True when ``row`` shares no n-gram (or whole short held-out text) with the held-out file."""
        t = row_text(row)
        if not t and "input_ids" in row:
            raise Refused("NGramGuard cannot check token-id rows (no text); mix them without this guard")
        hit = any(not ngrams(t, k).isdisjoint(g) for k, g in self.grams.items())
        self.dropped += hit
        return not hit

    def describe(self) -> dict:
        """What the sidecar records of this guard."""
        return {
            "kind": "ngram",
            "heldout": str(self.path),
            "heldout_sha256": self.sha256,
            "n": self.n,
            "dropped": self.dropped,
        }


def allocate(weights: list[float | Fraction], samples: int) -> list[int]:
    """``samples`` split in proportion to ``weights``: largest remainder, ties to the earlier source.

    Exact rational arithmetic (``Fraction``), so equal remainders really are equal and the tie rule decides.
    """
    if not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise Refused(
            f"weights {weights}: need at least one positive, none negative; give a part a positive weight"
        )
    ws = [Fraction(w) for w in weights]
    exact = [samples * w / sum(ws) for w in ws]
    counts = [int(x) for x in exact]
    order = sorted(range(len(ws)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: samples - sum(counts)]:
        counts[i] += 1
    return counts


HEX64 = re.compile(r"[0-9a-f]{64}")


def read_source(path: Path) -> tuple[list[dict], str | None]:
    """A source's rows and its tokenizer fingerprint.

    The fingerprint comes from a leading ``{"tokenizer": "<64 hex>"}`` header row (stripped from the rows), the
    engine's ``<path>.tokenizer`` sidecar, or this package's ``<path>.meta.json``; all that are present must agree.

    Raises:
        Refused: the file cannot be read (missing, a directory, or not permitted); the header, ``.tokenizer``
            and ``.meta.json`` name different tokenizers; or a header is malformed.
    """
    try:
        rows = read_rows(path)
    except OSError as e:
        raise Refused(f"calib mix part {path}: cannot read: {e}") from e
    found: dict[str, str] = {}
    if rows and set(rows[0]) == {"tokenizer"}:
        tag = str(rows.pop(0)["tokenizer"])
        if not HEX64.fullmatch(tag):
            raise Refused(f"{path}: header row tokenizer {tag!r} is not a sha256 hex digest")
        found["header row"] = tag
    side = path.with_name(path.name + ".tokenizer")
    if side.is_file():
        found[side.name] = side.read_text().strip()
    m = meta_path(path)
    if m.is_file() and json.loads(m.read_text()).get("tokenizer"):
        found[m.name] = json.loads(m.read_text())["tokenizer"]
    if len(set(found.values())) > 1:
        raise Refused(f"{path}: tokenizer fingerprints disagree: {found}")
    return rows, next(iter(found.values()), None)


def mix(
    sources: Sequence[tuple[str | Path, str | float | Fraction]],
    samples: int,
    seed: int = 0,
    *,
    guard: Guard | None = None,
    out: str | Path | None = None,
    inspect: Callable[[Path, list[dict]], None] | None = None,
) -> Path:
    """Mix ``sources`` ((path, weight) pairs) into ``samples`` rows; returns the written JSONL.

    Args:
        sources: calibration JSONLs and their weights.
        samples: rows in the result.
        seed: the mixing seed.
        guard: keep-predicate applied to every candidate row (none by default).
        out: where to write; default ``$QDA_CACHE/calib/mix/<key>.jsonl``.
        inspect: called with each source's path and every row it read, before any is picked (the data lint).

    Raises:
        Refused: a source has fewer guard-passing rows than its share, or token-id sources without, or with
            different, tokenizer fingerprints.
    """
    paths = [Path(p) for p, _ in sources]
    counts = allocate([Fraction(w) if isinstance(w, str) else w for _, w in sources], samples)
    tokenizers = set()
    picked: list[dict] = []
    per_source = []
    for i, (p, k) in enumerate(zip(paths, counts, strict=True)):
        rows, tok = read_source(p)
        if inspect is not None:
            inspect(p, rows)
        if any("input_ids" in r for r in rows):
            if not tok:
                raise Refused(f"{p}: token-id rows without a tokenizer fingerprint (header row or sidecar)")
            tokenizers.add(tok)
        elif tok:
            tokenizers.add(tok)
        random.Random(f"{seed}:{i}").shuffle(rows)
        take, skipped = [], 0
        for r in rows:
            if len(take) == k:
                break
            if guard is None or guard(r):
                take.append(r)
            else:
                skipped += 1
        if len(take) < k:
            raise Refused(
                f"{p}: {len(take)} usable rows ({skipped} rejected by the guard), its share is {k}; "
                "add rows, lower its weight, or relax the guard"
            )
        picked += take
        per_source.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "weight": str(sources[i][1]),
                "rows": k,
                "guard_rejected": skipped,
            }
        )
    if len(tokenizers) > 1:
        raise Refused(f"token-id sources made by different tokenizers: {sorted(tokenizers)}")
    random.Random(seed).shuffle(picked)
    describe = getattr(guard, "describe", None)
    meta = {
        "kind": "mix",
        "samples": samples,
        "seed": seed,
        "sources": per_source,
        "guard": describe() if describe else (None if guard is None else type(guard).__name__),
        "tokenizer": next(iter(tokenizers), None),
    }
    if out is None:
        key = hashlib.sha256(
            json.dumps({k: meta[k] for k in ("samples", "seed", "sources", "guard")}, sort_keys=True).encode()
        ).hexdigest()[:24]
        out = cache_root() / "mix" / f"{key}.jsonl"
    return write_jsonl(Path(out), picked, meta)


def main(argv: list[str] | None = None) -> int:
    """``python -m qda_quant.calib.mix``: prints the mixed file's path."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--source", action="append", required=True, metavar="PATH:WEIGHT")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", type=Path, help="drop rows sharing a word n-gram with this JSONL")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    sources = []
    for s in a.source:
        p, _, w = s.rpartition(":")
        sources.append((p, Fraction(w)) if p else (s, Fraction(1)))
    guard = NGramGuard(a.heldout, a.ngram) if a.heldout else None
    print(mix(sources, a.samples, a.seed, guard=guard, out=a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
