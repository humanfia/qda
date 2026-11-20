"""``qda probe``: build one format scheme by round to nearest, read its KL on a small fixed set, one line out.

The scheme is a build config's ``format`` and ``formats:``. The probe replaces the config's steps by one RTN step
(``rtn: {}``), builds it, prices it with the judge's budget counting (`judge.checks.budget.bpw` against `--anchor`),
and scores it with ``qda-kl score`` against one of two references:

- ``--data FILE``: the probe's own reference, built once per (model, data, size) under `--work` with
  ``qda-kl ref``: `--sequences` rows of FILE, chosen by a seeded hash of their ids, each answered once by
  `--anchor` (a row's ``prompt``, or its first user message) or teacher-forced (a row's ``text``). This is the
  form for a search: it reads only data the caller already has.
- ``--ref DIR``: `--sequences` sequences of an existing qda-kl reference, chosen the same way and written once
  under `--work` as links (no extra bytes).

"Before" is the same config with ``formats:`` dropped (every unit in the body format), probed once and cached under
`--work` by its digest. Prints::

    probe <what changed> | bpw <b> | KL <before> -> <after> | <s> s (build <s>, score <s>)

and appends the full record to ``WORK/probes.jsonl``. The checkpoint is removed after it is scored unless
`--keep`. Each step runs as a child of this interpreter (``python -m qda_quant.cli build``,
``python -m qda_kl.cli ref|score``), so it sees the same PYTHONPATH and one card.

A probe measures a round-to-nearest build. It ranks schemes; it does not predict what GPTQ or AWQ makes of them.
An MoE checkpoint's readings differ run to run (the same bytes of a 30B-A3B read 0.044437 to 0.044610 KL in three
processes): take `--repeat` and compare a change against that spread.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION

from .isolation import reaper

#: The judge's reading settings for one card (`judge.kl_cmd`'s scorer), so a probe reads as the judge reads.
SERVE = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "enable_prefix_caching": False,
    "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
    "max_num_seqs": 128,
}
SEED = "qda-probe"
#: Seconds the reaper gives a probe's leftover processes between SIGTERM and SIGKILL.
PROBE_GRACE_S = 5.0


def _digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _reference_identity(ref: Path) -> str:
    """The content identity of a qda-kl reference: its manifest's content hash (the manifest's bytes if unset)."""
    from qda_kl.store import Reference

    sha = Reference(ref).manifest.content_sha256
    return sha or hashlib.sha256((ref / "manifest.json").read_bytes()).hexdigest()


def _anchor_identity(anchor: Path) -> str:
    """The original's identity for pricing: its path and the hash of its config.json (its shapes)."""
    return f"{anchor.resolve()}:{hashlib.sha256((anchor / 'config.json').read_bytes()).hexdigest()}"


def _generation_identity(anchor: Path) -> str | None:
    """The hash of the original's ``generation_config.json``, or None when it has none.

    Without ``--sampling`` qda-kl takes the sampling and the stop tokens from this file, so it is part of what
    an own reference was made from: changing or removing its sampling must not reuse the old reference.
    """
    gen = anchor / "generation_config.json"
    return hashlib.sha256(gen.read_bytes()).hexdigest() if gen.is_file() else None


def _reuse(out: Path, inputs: dict) -> bool:
    """Whether `out` was made from exactly `inputs` (``inputs.json`` beside it); refuse a different maker."""
    if not (out / "manifest.json").is_file():
        return False
    made = out.parent / f"{out.name}.inputs.json"
    if not made.is_file() or json.loads(made.read_text()) != inputs:
        raise ValueError(f"{out} exists but was made from other inputs ({made}); use another --work")
    return True


def reduced_reference(ref: Path, n: int, work: Path) -> Path:
    """`n` sequences of `ref` (seeded hash order of their ids) as a linked reference under `work`, made once.

    The directory is named by and checked against the source's content identity, `n` and the seed, so another
    reference or size in the same `work` never reuses it.
    """
    from qda_kl.store import Reference, write_reduced

    inputs = {"source": _reference_identity(ref), "n": n, "seed": SEED}
    out = work / f"ref-{n}-{_digest(inputs)}"
    if _reuse(out, inputs):
        return out
    r = Reference(ref)
    if n > len(r.manifest.sequences):
        raise ValueError(f"--sequences {n}: {ref} holds {len(r.manifest.sequences)} sequences")
    order = sorted(r.manifest.sequences, key=lambda s: hashlib.sha256(f"{SEED}:{s.id}".encode()).hexdigest())
    keep = {s.id for s in order[:n]}
    got = write_reduced(r, [s for s in r.manifest.sequences if s.id in keep], out)
    (work / f"{out.name}.inputs.json").write_text(json.dumps(inputs, sort_keys=True) + "\n")
    return got


def _rows(data: Path) -> list[dict]:
    """`data`'s rows as qda-kl prompt rows: ``{id, domain, prompt}`` or ``{id, domain, text}``."""
    rows = []
    for i, line in enumerate(data.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        rid, domain = str(r.get("id", f"row-{i}")), str(r.get("domain", "own"))
        if isinstance(r.get("text"), str):
            rows.append({"id": rid, "domain": domain, "text": r["text"]})
            continue
        prompt = r.get("prompt")
        if not isinstance(prompt, str):
            users = [m.get("content") for m in r.get("messages") or [] if m.get("role") == "user"]
            prompt = users[0] if users and isinstance(users[0], str) else None
        if prompt is None:
            raise ValueError(f"{data} row {i}: no text, prompt or user message")
        rows.append({"id": rid, "domain": domain, "prompt": prompt})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{data}: row ids repeat")
    return rows


def own_reference(
    anchor: Path, data: Path, n: int, work: Path, max_tokens: int, sampling: dict | None = None
) -> Path:
    """A reference of `anchor` on `n` rows of `data` (seeded hash order of their ids), built once under `work`.

    Named by and checked against the original, the data's content, `n`, the cap, the sampling and the original's
    ``generation_config.json`` (where qda-kl reads the sampling and stop tokens when neither `sampling` nor the
    model's released card names one). With none of the three qda-kl refuses (never a default).
    """
    from qda_kl.reference import reference_sampling

    effective = reference_sampling(anchor, sampling)  # refuses before any cache hit, as qda-kl does
    inputs = {
        "anchor": _anchor_identity(anchor),
        "generation_config": _generation_identity(anchor),
        "data": hashlib.sha256(data.read_bytes()).hexdigest(),
        "n": n,
        "max_tokens": max_tokens,
        "sampling": sampling,
        "effective_sampling": effective,
        "seed": SEED,
    }
    out = work / f"ref-own-{_digest(inputs)}"
    if _reuse(out, inputs):
        return out
    rows = _rows(data)
    if n > len(rows):
        raise ValueError(f"--sequences {n}: {data} holds {len(rows)} rows")
    keep = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())[:n]
    work.mkdir(parents=True, exist_ok=True)
    prompts = work / f"{out.name}.prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in keep))
    argv = [sys.executable, "-m", "qda_kl.cli", "ref", "--model", str(anchor), "--prompts", str(prompts)]
    argv += ["--out", str(out), "--answers", "1", "--max-tokens", str(max_tokens)]
    if sampling is not None:
        argv += ["--sampling", json.dumps(sampling)]
    _run(argv, work / f"{out.name}.log")
    (work / f"{out.name}.inputs.json").write_text(json.dumps(inputs, sort_keys=True) + "\n")
    return out


def _run(argv: list[str], log: Path) -> float:
    t = time.monotonic()
    with log.open("w") as f:
        # Under a reaper: an interrupted probe leaves no engine or build process behind (`isolation.reaper`).
        rc = reaper.run(argv, out=f, grace_s=PROBE_GRACE_S)
    if rc:
        lines = log.read_text().strip().splitlines() or [""]
        why = next(
            (ln for ln in reversed(lines) if ln.startswith("error: ") or "refused" in ln or "Error" in ln),
            lines[-1],
        )
        raise RuntimeError(f"{argv[2]} {argv[3]} exited {rc}: {why} (log {log})")
    return time.monotonic() - t


def probe_one(
    cfg: dict, anchor: Path, ref: Path, work: Path, keep: bool = False, repeat: int = 1
) -> dict[str, Any]:
    """Build `cfg` by RTN, price and score it `repeat` times on one engine; the record (cached by digest)."""
    scheme = {k: cfg.get(k) for k in ("model", "format", "formats", "ignore", "target")}
    drop = ("steps", "out", "format") if cfg.get("formats") else ("steps", "out")  # formats: names the body
    build = {**{k: v for k, v in cfg.items() if k not in drop}, "steps": [{"rtn": {}}]}
    # the record answers exactly these inputs: the build, the original that prices it, the reference's content
    # (a reduced one names its source and size) and the scorer's settings
    identity = {
        "build": build,
        "anchor": _anchor_identity(anchor),
        "reference": _reference_identity(ref),
        "serve": SERVE,
        "repeat": repeat,
    }
    key = _digest(identity)
    cache = work / f"probe-{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    work.mkdir(parents=True, exist_ok=True)
    with (work / f"probe-{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # probes in parallel in one --work share the "before": one builds it
        if cache.is_file():
            return json.loads(cache.read_text())
        return _probe_unlocked(anchor, ref, work, keep, repeat, scheme, build, identity, key)


def _probe_unlocked(
    anchor: Path,
    ref: Path,
    work: Path,
    keep: bool,
    repeat: int,
    scheme: dict,
    build: dict,
    identity: dict,
    key: str,
) -> dict[str, Any]:
    from .judge.checks.budget import bpw

    cache = work / f"probe-{key}.json"
    run = work / key
    if (
        run.exists()
    ):  # an earlier attempt that never wrote its record: its own partial files, held by the lock
        shutil.rmtree(run)
    run.mkdir(parents=True)
    build = {**build, "out": str(run / "ckpt")}
    (run / "build.yaml").write_text(yaml.safe_dump(build, sort_keys=False))
    build_s = _run(
        [sys.executable, "-m", "qda_quant.cli", "build", str(run / "build.yaml")], run / "build.log"
    )
    b = bpw(run / "ckpt", anchor)
    score = {
        "checkpoint": str(run / "ckpt"),
        "reference": str(ref),
        "out": str(run / "score"),
        "scorer": {
            "topk": {
                "tp": 1,
                "serve_args": SERVE,
                **({"repeat": repeat, "allow_nondeterministic": True} if repeat > 1 else {}),
            }
        },
    }
    (run / "score.yaml").write_text(yaml.safe_dump(score, sort_keys=False))
    score_s = _run([sys.executable, "-m", "qda_kl.cli", "score", str(run / "score.yaml")], run / "score.log")
    got = json.loads((run / "score" / "score.json").read_text())
    if not keep:
        shutil.rmtree(run / "ckpt")
    rec = {"scheme": scheme, "identity": identity, "digest": key, "bpw": b, "kl": got["kl"], "tokens": got["tokens"],
           "kernels": got.get("kernels"), "build_s": round(build_s, 1), "score_s": round(score_s, 1),
           "repeats": got.get("repeats"), "deterministic": got.get("deterministic"), "spread": got.get("spread")}  # fmt: skip
    cache.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def probe(
    cfg: dict,
    anchor: Path,
    ref: Path | None,
    n: int,
    work: Path,
    keep: bool = False,
    data: Path | None = None,
    max_tokens: int = 2048,
    repeat: int = 1,
    sampling: dict | None = None,
) -> tuple[str, dict]:
    """Probe `cfg` and its uniform body against `ref`'s subset or an own reference of `data`; line and record."""
    t = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    if (ref is None) == (data is None):
        raise ValueError("give exactly one of --ref and --data")
    if ref is not None:
        sub = reduced_reference(ref, n, work)
    else:
        assert data is not None  # exactly one of the two, checked above
        sub = own_reference(anchor, data, n, work, max_tokens, sampling)
    uniform = {k: v for k, v in cfg.items() if k != "formats"}
    uniform.setdefault("format", (cfg.get("formats") or {}).get("default"))
    base = probe_one(uniform, anchor, sub, work, keep, repeat)
    after = probe_one(cfg, anchor, sub, work, keep, repeat) if cfg.get("formats") else base
    overrides = {k: v for k, v in (cfg.get("formats") or {}).items() if k != "default"}
    changed = ", ".join(f"{k}: {v}" for k, v in overrides.items()) or f"none (uniform {uniform['format']})"
    if len(changed) > 160:
        changed = f"{len(overrides)} modules -> {sorted(set(overrides.values()))} (in probes.jsonl)"
    total = time.monotonic() - t
    spread = ""
    if repeat > 1:
        s0, s1 = ((r.get("spread") or {}).get("kl") for r in (base, after))
        spread = f" (x{repeat}, relative spread {s0!r} / {s1!r})"
    line = (
        f"probe {changed} | bpw {after['bpw']!r} | KL {base['kl']:.6f} -> {after['kl']:.6f}{spread} | "
        f"{total:.0f} s (build {after['build_s']:.0f}, score {after['score_s']:.0f})"
    )
    rec = {
        "line": line,
        "before": base,
        "after": after,
        "sequences": n,
        "reference": str(sub),
        "source": str(ref if ref is not None else data),
        "elapsed_s": total,
    }
    with (work / "probes.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return line, rec


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add ``probe`` to the ``qda`` command set (entry point ``qda.commands``)."""
    p = sub.add_parser("probe", help="RTN-build one format scheme and read its KL on a small fixed subset")
    _arguments(p)
    p.set_defaults(run=run)
    return p


def _arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", type=Path, help="a build config: its model, format and formats: are the scheme")
    p.add_argument("--anchor", required=True, type=Path, help="the bf16 original (the budget's denominator)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="JSONL of your own prompts or text: the probe's own reference")
    src.add_argument("--ref", type=Path, help="an existing qda-kl reference of the original")
    p.add_argument(
        "--sequences", type=int, default=32, help="rows of --data or sequences of --ref (default 32)"
    )
    p.add_argument(
        "--max-tokens", type=int, default=2048, help="cap of one answer in the own reference (default 2048)"
    )
    p.add_argument(
        "--work", required=True, type=Path, help="probe directory: builds, reduced reference, cache"
    )
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config key")
    p.add_argument("--keep", action="store_true", help="keep the built checkpoint")
    p.add_argument(
        "--sampling",
        type=json.loads,
        default=None,
        metavar="JSON",
        help='sampling of the own reference\'s answers, e.g. \'{"temperature": 1.0, "top_p": 0.95}\'; without it the '
        "original's generation_config.json must name one (refused otherwise, never a default)",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="readings per checkpoint on one engine, mean and spread kept (an MoE's readings differ run to run)",
    )


def run(a: argparse.Namespace) -> int:
    """Run a parsed ``probe`` command."""
    from qda_quant.exit import refuse
    from qda_quant.recipe import apply_overrides
    from qda_quant.step import Refused

    try:
        cfg = apply_overrides(yaml.safe_load(a.config.read_text()), a.set)
        line, _ = probe(
            cfg, a.anchor, a.ref, a.sequences, a.work, a.keep, a.data, a.max_tokens, a.repeat, a.sampling
        )
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    except (OSError, ValueError, RuntimeError) as e:
        return refuse(str(e))
    print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m qda_agent.probe``: the same command without the ``qda`` entry point."""
    p = argparse.ArgumentParser(prog="qda probe", description=(__doc__ or "").split("\n")[0])
    _arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
