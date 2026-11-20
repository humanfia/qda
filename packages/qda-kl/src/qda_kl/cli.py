"""``qda-kl``: build a reference once per original model, then score any checkpoint against it.

qda-kl ref --model M --prompts P.jsonl --out R [--answers 2] [--max-tokens 8192] [--sampling JSON]
    [--serve-env JSON] [--serve-args JSON] [--gpu-memory-utilization 0.85] [--cache]
    [--group '{"prompts": P2.jsonl, "out": R2, "subsets": {"NAME": IDS}}']...   # one engine load
qda-kl ref merge OUT SHARD_DIR... [--corpus FILE]         # one verified reference from shards
qda-kl score CONFIG.yaml [--set scorer.topk.repeat=2]       # one config file per run
qda-kl gencheck CONFIG.yaml [--set max_tokens=32768]       # compare long generations with the original
qda-kl corpus --out CORPUS.jsonl [--guard DIR]             # the KL v2 prompt corpus, its eval and holdout splits
qda-kl kl CHECKPOINT [--original M]                         # one checkpoint's KL from its original (cached reference)

Commands exit 0 on success, 2 for a refusal of the input, 3 for a refusal of this machine (a card, the disk,
a package, a sign-in) and 1 for an unexpected error, as every ``qda`` command does (`qda_quant.exit`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from qda_quant.exit import ENVIRONMENT, ERROR, crash, refuse
from qda_quant.io import line_buffered_stdout, show_only
from qda_quant.step import Refused

from qda_kl.engine_params import DEFAULT_TP, engine_parameters
from qda_kl.engine_params import add_arguments as add_engine_arguments
from qda_kl.lifetime import command

if TYPE_CHECKING:
    from pydantic import BaseModel

    from qda_kl.reference import Group
    from qda_kl.scorers import KLScorer


#: The commands `qda-kl --help` lists and 0.1.x keeps stable: every one the user docs tell you to run.
#: The rest still run when named.
PUBLIC = ("ref", "score", "kl", "gencheck")


@command("qda-kl")  # the engine a command opens ends with it, on a stop signal too
def main(argv: list[str] | None = None) -> int:
    """Parse ``qda-kl`` arguments and dispatch reference, scoring or generation-check commands."""
    line_buffered_stdout()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:2] == ["ref", "merge"]:
        from qda_kl.reference_merge import merge

        parser = argparse.ArgumentParser(prog="qda-kl ref merge")
        parser.add_argument("out", type=Path)
        parser.add_argument("shards", nargs="+", type=Path)
        parser.add_argument("--corpus", type=Path)
        parser.add_argument("--subset", action="append", default=[], metavar="NAME=FILE")
        args = parser.parse_args(argv[2:])
        try:
            subsets = read_subsets(args.subset)
            manifest = merge(args.out, args.shards, corpus=args.corpus, subsets=subsets or None)
        except (OSError, ValueError) as why:
            return refuse(str(why))
        summary: dict = {"out": str(args.out), "content_sha256": manifest.content_sha256}
        summary.update(sequences=len(manifest.sequences), positions=manifest.positions)
        print(json.dumps(summary))
        return 0
    assert __doc__ is not None  # this module always has a docstring
    ap = argparse.ArgumentParser(
        prog="qda-kl",
        description="Build a reference once per original model, then score any checkpoint against it.",
    )
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("ref", help="build a reference for an original model")
    r.add_argument(
        "--model", required=True, help="the original model: a local directory or a Hub id (REPO[@REV])"
    )
    r.add_argument(
        "--prompts", required=True, type=Path, help="JSONL: {id, domain, prompt} or {id, domain, text}"
    )
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--answers", type=int, default=2, help="answers per prompt (default 2)")
    r.add_argument("--seed", type=int, default=17, help="seed of draw 0; draw d uses seed + d (default 17)")
    r.add_argument("--max-tokens", type=int, default=8192, help="cap of one answer (default 8192)")
    r.add_argument("--top-k", type=int, metavar="K", default=128, help="top-k kept per position (default 128)")
    r.add_argument(
        "--chat-template-kwargs", type=json.loads, default=None, help='JSON, e.g. {"enable_thinking": false}'
    )
    r.add_argument(
        "--sampling",
        type=json.loads,
        default=None,
        help='JSON, e.g. {"temperature": 1.0, "top_p": 0.95}; each field wins over the model\'s released '
        "cards' row, which wins over its generation_config.json",
    )
    r.add_argument("--tp", type=int, default=DEFAULT_TP)
    r.add_argument(
        "--subset",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="record a named subset of prompt ids (one per line) in the manifest, e.g. search96=ids.txt",
    )
    add_engine_arguments(r)
    r.add_argument(
        "--cache",
        action="store_true",
        help="build into the reference cache ($QDA_CACHE/kl-ref) once and make --out a link to it; a later call "
        "on the same model, prompts and settings reuses it",
    )
    r.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="JSON",
        help="another prompt file and reference directory built in the same engine load: "
        '\'{"prompts": FILE, "out": DIR, "subsets": {"NAME": IDS_FILE}}\' (subsets optional; with --cache, '
        'a group without "out" only warms the cache)',
    )
    s = sub.add_parser("score", help="score a checkpoint against a reference")
    s.add_argument("config", type=Path, help="a config YAML (CONFIG.yaml)")
    s.add_argument("--set", action="append", default=[], metavar="K=V", help="config override")
    g = sub.add_parser("gencheck", help="check long generations against the original model")
    g.add_argument("config", type=Path, help="generation-check YAML config")
    g.add_argument("--set", action="append", default=[], metavar="K=V", help="config override")
    from qda_kl import corpus

    corpus.add_arguments(sub.add_parser("corpus", help="build the KL corpus and its eval/holdout splits"))
    from qda_kl import kl

    kl.add_arguments(sub.add_parser("kl", help="one checkpoint's KL from its own original, in one command"))
    show_only(sub, PUBLIC)
    a = ap.parse_args(argv)
    if a.verb == "kl":
        return kl.run(a)
    if a.verb == "corpus":
        return corpus.run(a)
    if a.verb == "gencheck":
        return _gencheck_config(a.config, a.set)
    if a.verb == "ref":
        from qda_kl.reference import Group, build, build_many, over_context_line, sampling_note

        try:
            subsets = read_subsets(a.subset)
            groups = [Group(a.prompts, a.out, subsets or None), *read_groups(a.group, cache=a.cache)]
        except (OSError, ValueError) as why:
            return refuse(str(why))
        try:
            settings = {
                "answers": a.answers,
                "seed": a.seed,
                "max_tokens": a.max_tokens,
                "k": a.top_k,
                "tp": a.tp,
                "chat_template_kwargs": a.chat_template_kwargs,
                "sampling": a.sampling,
                **engine_parameters(a),
            }
            if a.cache:  # one cache entry per group, the missing ones built in one engine load
                from qda_kl import refcache
                from qda_kl.store import Manifest

                targets, _ = refcache.cached_many(
                    a.model, groups, {**settings, "subsets": subsets or None}, log=_stderr
                )
                manifests = [Manifest.model_validate_json((t / "manifest.json").read_text()) for t in targets]
            elif len(groups) == 1:
                targets = [a.out]
                manifests = [build(a.model, a.prompts, a.out, **settings, subsets=subsets or None)]
            else:
                targets = [g.out for g in groups]
                manifests = build_many(a.model, groups, **settings)
            if (note := sampling_note(a.model, a.sampling)) is not None:  # the first line: where it came from
                print(note)
        except Refused as why:
            return refuse(str(why), why.fix or None, code=why.code)
        except ValidationError as why:
            return refuse(str(why))
        except Exception as why:  # noqa: BLE001 (a crash is an error, not a verdict on the model)
            return _crashed(why)
        refs = [
            {
                "out": str(g.out if g.out is not None else t),
                "sequences": len(m.sequences),
                "positions": m.positions,
                "tail_mass": m.tail_mass,
                "vllm": m.model.get("engine"),
                "serve_env": m.build.get("serve_env"),
                "seconds": m.build.get("seconds"),
            }
            for g, m, t in zip(groups, manifests, targets, strict=True)
        ]
        for m in manifests:  # rows past the model's context are dropped, never silently: say how many
            if line := over_context_line(m.build or {}):
                print(f"qda-kl: {line}", file=sys.stderr)
        # one group: the summary `ref` always printed; several: one entry per reference
        print(json.dumps(refs[0] if len(refs) == 1 else {"refs": refs}))
        return 0
    return _score(a)


def _stderr(line: str) -> None:
    print(f"qda-kl: {line}", file=sys.stderr, flush=True)


def read_subsets(entries: list[str]) -> dict[str, list[str]]:
    """``--subset NAME=FILE`` entries as ``{name: ids}``, one id per line of FILE.

    Raises:
        ValueError: An entry is not NAME=FILE, or names a subset twice.
        OSError: A file cannot be read.
    """
    subsets: dict[str, list[str]] = {}
    for entry in entries:
        name, separator, file = entry.partition("=")
        if not name or not separator or name in subsets:
            raise ValueError("--subset requires unique NAME=FILE entries")
        subsets[name] = read_ids(file)
    return subsets


def read_ids(file: str | Path) -> list[str]:
    """The prompt ids of one subset file, one per line, in file order."""
    return [line.strip() for line in Path(file).read_text().splitlines() if line.strip()]


def read_groups(entries: list[str], *, cache: bool = False) -> list[Group]:
    """``--group JSON`` entries as :class:`qda_kl.reference.Group` values, their subset files read.

    Each entry is ``{"prompts": FILE, "out": DIR}`` and may carry ``"subsets": {NAME: IDS_FILE}``. With
    ``cache``, ``out`` may be omitted: the group then only warms the reference cache (``qda-agent prepare``
    warms the holdout reference this way; `evaluate` links it into the run when it is needed).

    Raises:
        ValueError: An entry is not a JSON object with a string ``prompts``, its ``out`` is not a string (or
            missing without ``cache``), names an unknown key, or names a subset by something but a name and an
            ids FILE.
        OSError: A subset file cannot be read.
    """
    from qda_kl.reference import Group

    groups = []
    for entry in entries:
        try:
            got = json.loads(entry)
        except json.JSONDecodeError as why:
            raise ValueError(f"--group {entry!r}: not JSON ({why})") from None
        if not isinstance(got, dict):
            raise ValueError(f"--group {entry!r}: expected a JSON object")
        if unknown := sorted(set(got) - {"prompts", "out", "subsets"}):
            raise ValueError(f"--group: unknown key {unknown[0]!r}; it takes prompts, out, subsets")
        if not isinstance(got.get("prompts"), str):
            raise ValueError(f"--group {entry!r}: give `prompts` as a string")
        if not isinstance(got.get("out"), str) and not (cache and "out" not in got):
            raise ValueError(
                f"--group {entry!r}: give `out` as a string"
                + (", or omit it with --cache to only warm the reference cache" if cache else "")
            )
        named = got.get("subsets") or {}
        if not isinstance(named, dict) or not all(
            isinstance(name, str) and isinstance(file, str) for name, file in named.items()
        ):
            raise ValueError("--group: `subsets` maps a subset name to its ids FILE")
        groups.append(
            Group(
                got["prompts"],
                got.get("out") if isinstance(got.get("out"), str) else None,  # pyright: ignore[reportArgumentType]  # None only under --cache, which writes no directory
                {name: read_ids(file) for name, file in named.items()} or None,
            )
        )
    return groups


def _crashed(why: Exception) -> int:
    """The line for a failure that is no refusal: a missing engine, or an engine start whose failure has a
    known fix, is the machine; anything else a bug.

    vLLM is the ``vllm`` extra, and the commands that open an engine import it lazily, so its absence
    surfaces here as a ``ModuleNotFoundError`` rather than at a place that could name it (`qda_quant.exit`).
    """
    if isinstance(why, ModuleNotFoundError) and (why.name or "").startswith("vllm"):
        return refuse(
            f"this command needs vLLM ({why})",
            'uv pip install -c constraints.txt "./packages/qda-kl[vllm]"',
            code=ENVIRONMENT,
        )
    problem, fix = crash(why)  # an engine start's known failure carries its upstream line and fix
    return refuse(problem, fix, code=ENVIRONMENT if fix else ERROR)


def _gencheck_config(path: Path, overrides: list[str]) -> int:
    """Validate and run a generation check, returning 2 on refusal or 1 on a runtime error."""
    from qda_kl.gencheck import RESULT, GencheckConfig, run_gencheck

    try:
        text = path.read_text()
    except OSError as error:  # the config file itself: the path the user gave
        return refuse(f"{path}: {error}")
    try:
        out_overrides = [item for item in overrides if item.startswith("out=")]
        out_only = GencheckConfig.raw(text, out_overrides).get("out")
        if isinstance(out_only, str):
            for name in (RESULT, "resolved.yaml"):
                (Path(out_only) / name).unlink(missing_ok=True)
        config = GencheckConfig.from_yaml(text, overrides)
        report = run_gencheck(config)
    except Refused as error:
        return refuse(f"{path}: {error}", error.fix or None, code=error.code)
    except ValidationError as error:
        return refuse(f"{path}: {error}")
    except Exception as error:  # noqa: BLE001 (a crashed generation is an error, not a verdict)
        return _crashed(error)
    print(json.dumps(report))
    return 0


def _score(a: argparse.Namespace) -> int:
    """Dispatch scoring from a YAML config."""
    path = a.config
    if path.suffix in (".yaml", ".yml") and path.is_file():
        return _score_config(path, a.set)
    if a.set:
        return refuse("--set needs a config file", "qda-kl score CONFIG.yaml --set k=v")
    return refuse("score takes a config file", "qda-kl score CONFIG.yaml")


def _score_config(path: Path, overrides: list[str]) -> int:
    """Score from YAML, removing stale results before validation and publishing only on success."""
    import shutil
    import time

    from qda_quant.output.record import RESOLVED, plugins, provenance, write_resolved

    from qda_kl import gguf_checkpoint
    from qda_kl.config import RESULT, ScoreConfig, source_identity
    from qda_kl.datasets import DATASETS, parse_dataset
    from qda_kl.reference import model_identity
    from qda_kl.scorers import SCORERS, TopKScorer

    try:
        text = path.read_text()
    except OSError as error:  # the config file itself: the path the user gave
        return refuse(f"{path}: {error}")
    try:
        # Resolve only `out` first so a refused rerun cannot leave a stale result.
        out_only = ScoreConfig.raw(text, [o for o in overrides if o.startswith("out=")]).get("out")
        if isinstance(out_only, str):
            for old in (Path(out_only) / RESULT, Path(out_only) / RESOLVED):
                old.unlink(missing_ok=True)
        cfg = ScoreConfig.from_raw(ScoreConfig.raw(text, overrides))
        reference = cfg.reference_path()
    except Refused as why:
        return refuse(f"{path}: {why}", why.fix or None, code=why.code)
    except ValidationError as why:
        return refuse(f"{path}: {why}")
    out = Path(cfg.out)
    used: list[tuple[Any, str, Any]] = [(SCORERS, cfg.scorer.name, type(cfg.scorer))]
    if isinstance(cfg.reference, dict):
        named = parse_dataset(cfg.reference, "reference")
        used.append((DATASETS, named.name, type(named)))
    try:
        gguf = gguf_checkpoint.gguf_file(cfg.checkpoint)
        snapshot = gguf_checkpoint.snapshot_of(cfg.snapshot, reference) if gguf else None
    except Refused as why:
        return refuse(f"{path}: {why}", why.fix or None, code=why.code)
    manifest: dict[str, Any] = {}

    def done(result: BaseModel) -> None:
        out.mkdir(parents=True, exist_ok=True)
        # both models this run read, by content signature: the number rests on their bytes, not their paths
        sources = {
            "checkpoint": source_identity(cfg.checkpoint),
            "reference": source_identity(reference),
        }
        if gguf is not None:  # the reading is the file's: name it, not the decode that is removed next
            sources["checkpoint"] = {
                "snapshot": str(snapshot),
                "revision": None,
                "signature": manifest["gguf"]["sha256"],
            }
            identity = {**model_identity(str(gguf)), "sha256": manifest["gguf"]["sha256"]}
            result = result.model_copy(update={"checkpoint": identity})
            # the whole reading, decode and engine load included: `wall_s` times the scoring alone
            manifest["reading_s"] = round(time.monotonic() - started, 3)
            (out / gguf_checkpoint.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        write_resolved(out, cfg.to_dict(), provenance(plugins(used), sources=sources))
        publish(out / RESULT, result.model_dump_json(indent=1))

    if gguf is None:
        return _run(cfg.scorer, cfg.checkpoint, reference, done)
    assert snapshot is not None
    started = time.monotonic()
    print(f"qda-kl score: decoding {gguf} against {snapshot}", file=sys.stderr, flush=True)
    try:
        decoded, got = gguf_checkpoint.decode(gguf, snapshot, out)
    except Refused as why:
        return refuse(str(why), why.fix or None, code=why.code)
    manifest.update(got)
    scorer = cfg.scorer
    if gguf_checkpoint.language_only(manifest) and isinstance(scorer, TopKScorer):
        scorer = scorer.model_copy(update={"serve_args": {"language_model_only": True, **scorer.serve_args}})
    try:
        return _run(scorer, str(decoded), reference, done)
    finally:
        shutil.rmtree(decoded, ignore_errors=True)


def _run(scorer: KLScorer, checkpoint: str, reference: Path, done: Callable[[BaseModel], None]) -> int:
    """Print scorer readings and call ``done`` only when the result is accepted."""
    try:
        got = scorer.run(checkpoint, reference)
    except Refused as why:
        return refuse(str(why), why.fix or None, code=why.code)
    except ValidationError as why:
        return refuse(str(why))
    except Exception as why:  # noqa: BLE001 (a crashed engine is an error, not a verdict on the checkpoint)
        return _crashed(why)
    for r in got.readings:
        print(json.dumps(scorer.summary(r)))
    if len(got.readings) > 1:
        res = got.result
        keys = ("kl", "kl_text", "domains", "deterministic", "repeats", "spread")
        print(json.dumps({"aggregate": True, **{k: getattr(res, k, None) for k in keys}}))
    if got.refused:
        return refuse(str(got.refused))
    done(got.result)
    return 0


def publish(out: Path, text: str) -> None:
    """Atomically replace ``out`` with ``text``, removing the temporary file on failure."""
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        Path(tmp).replace(out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
