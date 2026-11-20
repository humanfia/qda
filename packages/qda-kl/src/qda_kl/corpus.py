"""``qda-kl corpus``: build the KL v2 corpus, the prompt file the KL suite's references and scores read.

    qda-kl corpus --out klv2-corpus.jsonl [--guard DIR]

Every repository and revision is pinned in :data:`DATASETS`; a file is downloaded into ``HF_HOME`` when absent.
Rows are ``{"id", "group", "domain", "split", "prompt"}`` (``text`` instead of ``prompt`` for web text), the
shape ``qda.corpora.file`` reads. Each domain fills a fixed quota, and one row in six of that quota is
``holdout``, chosen by ``sha256(id)`` and recorded in the id, so the split is fixed by the data and not by the
order a run took the rows in. Beside the output go its two splits, ``<stem>.eval.jsonl`` (the prompts a
``qda-kl ref`` is built on) and ``<stem>.holdout.jsonl`` (a search's ``judge.kl_holdout_ref``), and the manifest,
which records the counts, the pinned revisions, the filter rules and each file's sha256.

The 20-word guard is an optional step of the build. With ``--guard`` pointing at per-benchmark guard files (one
window sha256 per line, as the ``ngram`` contamination check reads them), a candidate whose decoded text or JSON
line holds a window of a benchmark test item is dropped and recorded (id, source, benchmark) in the manifest; the
finished file is then scanned whole, raw and decoded, and moved into place only if no window remains, so the
build never writes a contaminated corpus. Without ``--guard`` no row is checked, and the manifest says
``"guard": "none"``: such a corpus scores and compares checkpoints, but is no evidence that a score is clean.

Only training splits are read. The selection mirrors the paper's: olympiad proofs, rated Codeforces
problems, ARC-Challenge science, persona and RLVR instruction following, 16k+ token documents, and tool calls.
IFBench's training set has no first-party release, and its public dataset is the test set, so it is not a
source here; Tulu-3 and RLVR-IFeval carry the verifiable-constraint slice.

Requires ``huggingface_hub``, ``pyarrow`` and ``transformers`` (the serving environment has all three).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from qda_quant.calib.contamination import WINDOW, windows
from qda_quant.exit import refuse
from qda_quant.step import Refused

#: The tokenizer that puts a long row at 16k+ tokens; the corpus is built for the Qwen3.8-27B family.
TOKENIZER = "Qwen/Qwen3.8-27B"

#: The tokenizer revision the long slice's token window is counted in (the model's ``main`` at build time).
TOKENIZER_REVISION = "8c0b3d30f3ae96f5c7f774de704bf0ada83e422a"

#: The long slice's token window: 16k+ context, with room for an answer inside a 32k-context engine.
LONG_TOKENS = (16384, 20480)

#: Pinned dataset revisions (resolved from the HF API for this corpus; never a moving branch).
DATASETS = {
    "numina": ("project-numina/NuminaMath-CoT", "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"),
    "code": ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
    "tulu": ("allenai/tulu-3-sft-personas-instruction-following", "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e"),
    "rlvr": ("allenai/RLVR-IFeval", "47c03c73621c4aab2b824b7818681117d662770e"),
    "long": ("zai-org/LongAlign-10k", "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc"),
    "tools": ("NousResearch/hermes-function-calling-v1", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
    "chat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"),
    "web": ("allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
    "short": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"),
    "multi": ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"),
}

#: The files read from each dataset, at the revision above.
FILES = {
    "numina": ["data/train-00000-of-00005.parquet"],
    "code": [
        "data/train-00000-of-00039-e991a271dbfa9925.parquet",
        "data/train-00001-of-00039-e092fe56fda18715.parquet",
        "data/train-00002-of-00039-9cea23812e920e41.parquet",
    ],
    "arc": ["ARC-Challenge/train-00000-of-00001.parquet"],
    "tulu": ["data/train-00000-of-00001.parquet"],
    "rlvr": ["data/train-00000-of-00001.parquet"],
    "long": ["long.jsonl"],
    "tools": ["func-calling-singleturn.json"],
    "chat": ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet"],
    "web": ["en/c4-train.00000-of-01024.json.gz"],
    "short": ["databricks-dolly-15k.jsonl"],
    "multi": ["data/train-00000-of-00001.parquet"],
}

#: (domain, group, tag, rows, holdout rows) in build order: the hard groups, then the easy guard.
DOMAINS = [
    ("math", "hard", "math", 240, 40),
    ("code", "hard", "code", 220, 37),
    ("science", "hard", "science", 160, 27),
    ("if", "hard", "tulu", 200, 33),
    ("if", "hard", "rlvr", 120, 20),
    ("long", "hard", "long", 40, 7),
    ("tools", "hard", "tools", 100, 17),
    ("chat", "easy", "chat", 32, 5),
    ("web", "easy", "web", 30, 5),
    ("short", "easy", "short", 30, 5),
    ("multilingual", "easy", "multi", 28, 5),
]

#: Languages of the multilingual slice, four rows each.
LANGS = ("por", "vie", "jpn", "zho", "arb", "tur", "hin")

MATH_SUFFIX = (
    "\n\nGive a complete derivation, justify each step, then verify the result independently. "
    "Discuss the assumptions and any boundary cases."
)
CODE_SUFFIX = "\n\nExplain the approach, then give a complete, efficient Python solution."
SCIENCE_SUFFIX = (
    "\n\nExplain the relevant science from first principles, analyze every option, and justify your answer. "
    "Give two related examples and discuss a common misconception."
)

#: Markup that points at an image: the prompt is unanswerable from its text alone, so it is dropped.
IMAGE = re.compile(r"!\[[^\]]*\]\(|<image>|\[asy\]|\\begin\{asy\}|\\includegraphics", re.IGNORECASE)

#: One guard line: the sha256 hex digest of a window (`windows`).
HEX64 = re.compile(r"[0-9a-f]{64}")

#: Candidates dropped for a filter other than the guard, by reason (recorded in the manifest).
FILTERED: Counter[str] = Counter()


def path(dataset: str, name: str) -> str:
    """The local file of a pinned dataset (downloading it into ``HF_HOME`` when absent)."""
    from huggingface_hub import hf_hub_download

    repo, revision = DATASETS[dataset]
    return hf_hub_download(repo, name, repo_type="dataset", revision=revision)


def math() -> Iterator[str]:
    """Olympiad proof problems from NuminaMath-CoT with no AIME or HMMT text and no image in the prompt."""
    import pyarrow.parquet as pq

    t = pq.read_table(path("numina", FILES["numina"][0]), columns=["source", "problem", "solution"])
    for source, problem, solution in zip(*(t.column(c).to_pylist() for c in t.schema.names), strict=True):
        if source not in ("olympiads", "aops_forum") or not problem or len(problem) < 200:
            continue
        low = f"{problem} {solution or ''}".lower()
        if ("prove" not in problem.lower() and "show that" not in problem.lower()) or "aime" in low:
            continue
        if IMAGE.search(problem):
            FILTERED["math: image"] += 1
            continue
        if "hmmt" not in low:
            yield problem + MATH_SUFFIX


def code() -> Iterator[str]:
    """CodeContests descriptions rated 1800 or more."""
    import pyarrow.parquet as pq

    for name in FILES["code"]:
        t = pq.read_table(path("code", name), columns=["description", "cf_rating"])
        for desc, rating in zip(*(t.column(c).to_pylist() for c in t.schema.names), strict=True):
            if rating and rating >= 1800 and desc and 200 <= len(desc) <= 8000:
                yield desc + CODE_SUFFIX


def science() -> Iterator[str]:
    """ARC-Challenge training questions with their choices."""
    import pyarrow.parquet as pq

    t = pq.read_table(path("arc", FILES["arc"][0]), columns=["question", "choices"])
    for question, choices in zip(*(t.column(c).to_pylist() for c in t.schema.names), strict=True):
        options = "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"], strict=True)
        )
        yield f"{question}\n\n{options}{SCIENCE_SUFFIX}"


def if_tulu() -> Iterator[str]:
    """Tulu-3 persona instruction-following prompts (their constraints are in the prompt)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path("tulu", FILES["tulu"][0]), columns=["prompt"])
    yield from (p for p in t.column("prompt").to_pylist() if p and 50 <= len(p) <= 3000)


def if_rlvr() -> Iterator[str]:
    """RLVR-IFeval single-turn instructions, one per constraint type first."""
    import pyarrow.parquet as pq

    t = pq.read_table(path("rlvr", FILES["rlvr"][0]), columns=["messages", "constraint_type"])
    by_type: dict[str, list[str]] = {}
    for messages, ctype in zip(*(t.column(c).to_pylist() for c in t.schema.names), strict=True):
        user = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user and 50 <= len(user) <= 3000:
            by_type.setdefault(ctype or "", []).append(user)
    for rows in by_type.values():
        yield from rows[:1]
    for rows in by_type.values():
        yield from rows[1:]


def long(tokenizer: Any) -> Iterator[str]:
    """LongAlign-10k user messages of :data:`LONG_TOKENS` Qwen tokens (characters prefilter, tokens decide)."""
    lo, hi = LONG_TOKENS
    with Path(path("long", FILES["long"][0])).open(encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            user = next((m["content"] for m in messages if m.get("role") == "user"), "")
            if 60000 <= len(user) <= 130000 and lo <= len(tokenizer(user)["input_ids"]) <= hi:
                yield user


def tools() -> Iterator[str]:
    """Hermes single-turn tool requests, with the tool list carried in the system prompt."""
    for row in json.loads(Path(path("tools", FILES["tools"][0])).read_text(encoding="utf-8")):
        system = next((c["value"] for c in row["conversations"] if c["from"] == "system"), "")
        query = next((c["value"] for c in row["conversations"] if c["from"] == "human"), "")
        if "<tools>" in system and 10 <= len(query) <= 1000:
            yield f"{system}\n\nUser request: {query}"


def chat() -> Iterator[str]:
    """UltraChat first user turns (short chat prompts)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path("chat", FILES["chat"][0]), columns=["prompt"])
    for user in t.column("prompt").to_pylist():
        if user and 40 <= len(user) <= 1500:
            yield user


def web() -> Iterator[tuple[str, str]]:
    """C4 documents as ``text`` rows (teacher-forced: no answer is generated)."""
    with gzip.open(path("web", FILES["web"][0]), "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = json.loads(line)["text"].strip()
            if 2500 <= len(text) <= 6000:
                yield "text", text


def short() -> Iterator[str]:
    """Dolly short open, classification and extraction instructions."""
    with Path(path("short", FILES["short"][0])).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            context = row.get("context") or ""
            prompt = f"{row['instruction']}\n\nContext: {context}" if context else row["instruction"]
            if (
                row["category"] in ("open_qa", "classification", "information_extraction")
                and len(prompt) <= 400
            ):
                yield prompt


def multilingual() -> Iterator[str]:
    """Aya training instructions, one per language per round, so the quota cannot take one language first.

    The generator yields ``LANGS`` rows per round; the quota takes 28 rows, about four per language, and the
    manifest records the mix the file actually has.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(path("multi", FILES["multi"][0]), columns=["inputs", "language_code"])
    rows = list(zip(*(t.column(c).to_pylist() for c in t.schema.names), strict=True))
    per_lang: dict[str, list[str]] = {lang: [] for lang in LANGS}
    for text, code in rows:
        if code in per_lang and text and 100 <= len(text) <= 600 and len(per_lang[code]) < 12:
            per_lang[code].append(text)
    for i in range(12):
        for lang in LANGS:
            if i < len(per_lang[lang]):
                yield per_lang[lang][i]


SOURCES = {
    "math": math,
    "code": code,
    "science": science,
    "tulu": if_tulu,
    "rlvr": if_rlvr,
    "long": long,
    "tools": tools,
    "chat": chat,
    "web": web,
    "short": short,
    "multi": multilingual,
}


def holdout(ident: str) -> bool:
    """Whether the row with `ident` is on the fixed held-out side: one id in six."""
    return int.from_bytes(hashlib.sha256(ident.encode()).digest()[:8], "big") < (1 << 64) // 6


class Guard:
    """The shared 20-word guard over the benchmark test items, and the rows it dropped.

    The per-benchmark files are the shared guard directory; the union decides a drop and the per-benchmark
    sets name the benchmark. A row is checked both decoded (its text) and raw (the JSON line as it would be
    written), because escapes can hide a window from one of the two views. An empty or unreadable file is
    refused: a guard that holds nothing would let a contaminated corpus through without a word.
    """

    def __init__(self, per_bench: Path) -> None:
        files = sorted(Path(per_bench).glob("*.txt"))
        if not files:
            raise Refused(
                f"no guard window files (*.txt) in {per_bench}",
                "point --guard at the directory of per-benchmark window hashes, or drop it",
            )
        self.per_bench: dict[str, set[str]] = {}
        self.sha256: dict[str, str] = {}
        for f in files:
            try:
                raw = f.read_bytes()
                words = {line.strip() for line in raw.decode().splitlines() if line.strip()}
            except (OSError, UnicodeDecodeError) as why:
                raise Refused(f"cannot read the guard file {f}: {type(why).__name__}: {why}") from None
            if not words:
                raise Refused(f"the guard file {f} holds no windows", "give the file its window hashes")
            if bad := next((w for w in sorted(words) if not HEX64.fullmatch(w)), None):
                raise Refused(
                    f"the guard file {f}: {bad[:40]!r} is no window sha256",
                    "one 64-digit sha256 of a 20-word window per line",
                )
            self.per_bench[f.stem] = words
            self.sha256[f.stem] = hashlib.sha256(raw).hexdigest()
        self.union = set().union(*self.per_bench.values())
        self.dropped: list[dict[str, str]] = []
        self.per_source: Counter[str] = Counter()

    def where(self, hits: set[str]) -> list[str]:
        """The benchmarks whose guard holds one of `hits`."""
        return sorted(name for name, grams in self.per_bench.items() if grams & hits)

    def drop(self, row: dict[str, Any], text: str, tag: str) -> bool:
        """Whether `row` must be dropped; if so, record its id, source, view and the benchmark hit."""
        hits = windows(text) & self.union
        view = "decoded"
        if not hits:
            hits = windows(json.dumps(row, ensure_ascii=False)) & self.union
            view = "raw"
        if not hits:
            return False
        self.dropped.append(
            {
                "id": row["id"],
                "source": tag,
                "view": view,
                "benchmark": ", ".join(self.where(hits)) or "shared guard",
            }
        )
        self.per_source[tag] += 1
        return True


def scan(text: str, guard: Guard) -> dict[str, int]:
    """Windows of a finished file (or of its decoded strings) that any benchmark guard holds."""
    hits = windows(text) & guard.union
    return {name: len(hits & grams) for name, grams in guard.per_bench.items() if hits & grams}


def decoded(path: Path) -> str:
    """Every string value of a JSONL file, one document after another (the shared scan's decoded view)."""

    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return []

    chunks: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                chunks.extend(strings(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return "\n".join(chunks)


def take(
    domain: str,
    group: str,
    tag: str,
    rows: int,
    held: int,
    source: Iterator[Any],
    used: set[str],
    guard: Guard | None,
) -> list[dict[str, Any]]:
    """Fill the domain's quota, dropping duplicates and any prompt the guard (when given) hits."""
    out: list[dict[str, Any]] = []
    ev = ho = 0
    for serial, item in enumerate(source):
        key, text = item if isinstance(item, tuple) else ("prompt", item)
        norm = " ".join(text.split()).lower()
        if norm in used or len(norm) < 20:
            continue
        used.add(norm)
        ident = f"klv2-{domain}-{tag}-{serial:05d}"
        is_hold = holdout(ident)
        if is_hold and ho >= held:
            continue
        if not is_hold and ev >= rows - held:
            continue  # this side of the quota is full; the next row may fill the other
        row = {
            "id": ident,
            "group": group,
            "domain": domain,
            "split": "holdout" if is_hold else "eval",
            key: text,
        }
        if guard is not None and guard.drop(row, text, tag):
            continue
        out.append(row)
        if is_hold:
            ho += 1
        else:
            ev += 1
        if len(out) == rows:
            break
    hit = guard.per_source[tag] if guard is not None else 0
    if len(out) < rows:
        raise Refused(
            f"{domain}/{tag}: only {len(out)} of {rows} rows matched the pinned files "
            f"({hit} dropped by the guard)"
        )
    if hit >= max(10, rows // 4):
        raise Refused(f"{domain}/{tag}: {hit} candidates hit the guard; drop this source")
    return out


#: The search set: this many eval prompts a search reads every candidate on (`search96`).
SEARCH = 96
SEARCH_NAME = f"search{SEARCH}"


def search_rows(built: list[dict[str, Any]], n: int = SEARCH) -> list[dict[str, Any]]:
    """The fixed search set: `n` eval rows, no `long` row, split over the domains in proportion to their eval
    rows (largest remainder), each domain's taken in `sha256(id)` order; returned in corpus order.

    A search reads every candidate on these prompts, so their answers must stay short enough for a reading of
    minutes on any engine; the full eval split is the reference for a final evaluation.
    """
    from qda_quant.calib.mix import allocate

    pool = [r for r in built if r["split"] == "eval" and r["domain"] != "long"]
    domains = sorted({r["domain"] for r in pool})
    sizes: list[float | Fraction] = [float(sum(r["domain"] == d for r in pool)) for d in domains]
    counts = allocate(sizes, min(n, len(pool))) if pool else []
    picked: set[str] = set()
    for domain, k in zip(domains, counts, strict=True):
        ids = sorted(
            (r["id"] for r in pool if r["domain"] == domain),
            key=lambda i: hashlib.sha256(i.encode()).digest(),
        )
        picked.update(ids[:k])
    return [r for r in pool if r["id"] in picked]


def finish(
    out: Path,
    built: list[dict[str, Any]],
    guard: Guard | None,
    *,
    tokenizer: str,
    tokenizer_revision: str | None,
) -> dict[str, Any]:
    """Write the corpus, its eval, holdout and `search96` sets and its manifest; return the summary's fields.

    `search96` is written twice: as prompts (`<stem>.search96.jsonl`, the rows a search reference is built on)
    and as ids (`<stem>.search96.ids`, one per line, the reference subset `qda-kl ref --subset` records).

    With a guard the finished file is scanned whole first and refused if a window remains; without one the
    manifest records ``"guard": "none"`` and the summary claims nothing about contamination.
    """
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in built), encoding="utf-8")
    whole: dict[str, dict[str, int]] = {}
    if guard is not None:
        whole = {"raw": scan(tmp.read_text(errors="ignore"), guard), "decoded": scan(decoded(tmp), guard)}
        if any(whole.values()):
            tmp.unlink()
            raise Refused(
                f"the finished file still holds guard windows: {whole}; nothing was written",
                "drop this source, or extend the guard directory",
            )
    tmp.replace(out)
    splits: dict[str, dict[str, Any]] = {}
    for split, rows in (
        ("eval", [r for r in built if r["split"] == "eval"]),
        ("holdout", [r for r in built if r["split"] == "holdout"]),
        (SEARCH_NAME, search_rows(built)),
    ):
        path = out.with_name(f"{out.stem}.{split}.jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        splits[split] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ids = out.with_name(f"{out.stem}.{SEARCH_NAME}.ids")
    ids.write_text("".join(f"{r['id']}\n" for r in search_rows(built)), encoding="utf-8")
    splits[SEARCH_NAME]["ids"] = str(ids)
    splits[SEARCH_NAME]["ids_sha256"] = hashlib.sha256(ids.read_bytes()).hexdigest()
    splits[SEARCH_NAME]["rule"] = (
        "eval rows, no long, per domain by eval share (largest remainder), sha256(id) order"
    )

    counts: dict[str, int] = {}
    for r in built:
        key = f"{r['group']}/{r['domain']}/{r['split']}"
        counts[key] = counts.get(key, 0) + 1
    mix = Counter(
        LANGS[int(r["id"].rsplit("-", 1)[1]) % len(LANGS)] for r in built if r["domain"] == "multilingual"
    )
    manifest = {
        "path": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": len(built),
        "splits": splits,
        "holdout": "sha256(id) below 1/6",
        "tokenizer": {"repo": tokenizer, "revision": tokenizer_revision},
        "counts": dict(sorted(counts.items())),
        "multilingual_mix": dict(sorted(mix.items())),
        "datasets": {k: {"repo": r, "revision": v, "files": FILES[k]} for k, (r, v) in DATASETS.items()},
        "filters": {
            "math": (
                "olympiads|aops_forum, >=200 chars, asks for a proof, no AIME/HMMT anywhere in the row, "
                "no image markup in the prompt"
            ),
            "code": "cf_rating >= 1800, description 200-8000 chars",
            "science": "ARC-Challenge train",
            "if": "Tulu-3 persona IF prompts; RLVR-IFeval, one prompt per constraint type first",
            "long": f"LongAlign-10k user content, {LONG_TOKENS[0]}-{LONG_TOKENS[1]} tokens of {tokenizer}@{tokenizer_revision}",
            "tools": "Hermes single-turn, system prompt holds <tools>, query 10-1000 chars",
            "chat": "UltraChat train_sft first user turn, 40-1500 chars",
            "web": "C4 en train text 2500-6000 chars",
            "short": "Dolly open_qa|classification|information_extraction, prompt <=400 chars",
            "multilingual": f"Aya train, one row per language per round, {', '.join(LANGS)}",
        },
        "guard": "none"
        if guard is None
        else {
            "window": WINDOW,
            "per_bench_sha256": guard.sha256,
            "rows_dropped": len(guard.dropped),
            "dropped": guard.dropped,
            "whole_file": whole,
        },
        "other_filters": dict(sorted(FILTERED.items())),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {k: manifest[k] for k in ("path", "sha256", "rows", "counts", "multilingual_mix")}
    summary["splits"] = {k: v["path"] for k, v in splits.items()}
    if guard is None:
        return summary | {"guard": "none"}
    return summary | {"guard_dropped": len(guard.dropped), "guard_clean": True}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``qda-kl corpus``."""
    parser.add_argument("--out", required=True, type=Path, help="the JSONL file to write")
    parser.add_argument(
        "--tokenizer", default=TOKENIZER, help="the tokenizer the long slice's window is counted in"
    )
    parser.add_argument(
        "--tokenizer-revision", default=TOKENIZER_REVISION, help="the tokenizer revision (a commit hash)"
    )
    parser.add_argument(
        "--guard",
        type=Path,
        help="a directory of per-benchmark 20-word window hashes (*.txt); without it no row is checked",
    )


def cache_key(tokenizer: str, tokenizer_revision: str | None) -> dict[str, Any]:
    """What an unguarded corpus's rows depend on: this module's code (every pin and filter is in it), the
    tokenizer that counts the long slice, and the libraries that read the files and count the tokens."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str | None:
        """The library's version, or None where it is not installed (the build that reads it refuses then)."""
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "versions": {name: installed(name) for name in ("transformers", "tokenizers", "pyarrow")},
    }


def cache_dir(key: dict[str, Any]) -> Path:
    """``$QDA_CACHE/kl-corpus/<key digest>``: the rows of one unguarded build and its filter counts."""
    from qda_quant.io import qda_cache

    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return qda_cache("kl-corpus", digest[:24])


#: The splits a corpus build writes beside its JSONL.
SPLITS = ("eval", "holdout", "search96")


def cached_splits(
    tokenizer: str = TOKENIZER, tokenizer_revision: str | None = TOKENIZER_REVISION
) -> dict[str, Path]:
    """The KL v2 corpus's ``eval``, ``holdout`` and ``search96`` splits, built once into the corpus cache.

    The one owner of where a cached corpus's splits live, so a caller that only needs one of them (``qda-kl
    kl``'s default prompts are the holdout split) does not invent a layout of its own.

    Raises:
        Refused: the corpus build failed or wrote no split it promised.
    """
    root = cache_dir(cache_key(tokenizer, tokenizer_revision))
    splits = root.with_name(f"{root.name}.splits")
    if all((splits / f"corpus.{name}.jsonl").is_file() for name in SPLITS):
        return {name: splits / f"corpus.{name}.jsonl" for name in SPLITS}
    tmp = splits.with_name(f".{splits.name}.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    args = argparse.Namespace(
        out=tmp / "corpus.jsonl",
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        guard=None,
    )
    print(f"qda-kl corpus: building the cached corpus's splits once ({splits})", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):  # the corpus summary is not the caller's output
        rc = run(args)
    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise Refused(f"the KL corpus build failed (exit {rc})")
    if missing := [name for name in SPLITS if not (tmp / f"corpus.{name}.jsonl").is_file()]:
        shutil.rmtree(tmp, ignore_errors=True)
        raise Refused(f"the KL corpus build wrote no {missing[0]} split")
    try:
        tmp.rename(splits)  # one rename: a reader sees every split or none
    except OSError:  # another process put the same splits in place first
        shutil.rmtree(tmp, ignore_errors=True)
    return {name: splits / f"corpus.{name}.jsonl" for name in SPLITS}


def _select(args: argparse.Namespace, guard: Guard | None) -> list[dict[str, Any]]:
    """Read the pinned files and take each domain's quota (the slow part: tens of seconds)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    used: set[str] = set()
    built: list[dict[str, Any]] = []
    for domain, group, tag, rows, held in DOMAINS:
        source = long(tokenizer) if tag == "long" else SOURCES[tag]()
        built += take(domain, group, tag, rows, held, source, used, guard)
    return built


def _cached_select(args: argparse.Namespace) -> list[dict[str, Any]]:
    """An unguarded selection, read from the corpus cache, or made once and kept there.

    The rows are identical for every user and model (pinned files, fixed rules), so the corpus `finish` writes
    from cached rows is byte-identical to a fresh build's; the filter counts the manifest records are kept too.
    """
    key = cache_key(args.tokenizer, args.tokenizer_revision)
    root = cache_dir(key)
    rows, counts = root / "rows.jsonl", root / "filtered.json"
    if rows.is_file() and counts.is_file():
        print(f"qda-kl corpus: reusing the cached selection {root}", file=sys.stderr)
        FILTERED.update(json.loads(counts.read_text()))
        return [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
    built = _select(args, None)
    tmp = root.with_name(f".{root.name}.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in built), encoding="utf-8"
    )
    (tmp / "filtered.json").write_text(json.dumps(dict(FILTERED)))
    (tmp / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    try:
        tmp.rename(root)  # one rename: a reader sees the whole entry or none
    except OSError:  # another build put the same entry in place first
        shutil.rmtree(tmp, ignore_errors=True)
    return built


def run(args: argparse.Namespace) -> int:
    """Build the JSONL corpus, its splits and its manifest (an unguarded selection comes from the cache)."""
    try:
        guard = Guard(args.guard) if args.guard is not None else None
        built = _select(args, guard) if guard is not None else _cached_select(args)
        summary = finish(
            Path(args.out),
            built,
            guard,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
        )
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``tools/build_kl_corpus.py`` shim: the same options as ``qda-kl corpus``."""
    parser = argparse.ArgumentParser(description="Build the KL v2 corpus and its eval and holdout splits.")
    add_arguments(parser)
    return run(parser.parse_args(argv))
