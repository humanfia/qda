"""Run an eval: every (checkpoint, part of a suite) job whose result file is missing or stale.

A job's result is ``out/results/<checkpoint key>/<suite key>.json``, or ``<suite key>.<label>.json`` for a part
of a suite (:meth:`qda_kl.eval.suites.Suite.parts`). It is up to date when its ``config_digest`` matches the
digest of the part's resolved settings and input files, the reference, the checkpoint and, for a local
checkpoint, its generation config, tokenizer config and chat template, so a rerun after a crash does only the
jobs that never finished, and a changed setting or metadata file reruns only the parts it changes. A refused job
writes a failure record: a result carrying the exact error and no metrics. A failure is never cached, so a later
run retries it, and a successful rerun replaces the record.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from qda_quant.calib import sha256_file
from qda_quant.exit import USAGE
from qda_quant.intake.corpus import CORPORA, Corpus
from qda_quant.output.record import plugins, provenance, write_resolved
from qda_quant.output.vllm_env import CACHE_ENV
from qda_quant.registry import Registry
from qda_quant.step import Refused

from qda_kl.cli import publish
from qda_kl.config import source_identity
from qda_kl.datasets import DATASETS
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.suites import SUITES, Layout, Suite, SuiteResult
from qda_kl.serve import ServeSettings, engine_environment


class _Servable(Protocol):
    """A suite that can generate on a server: it declares ``server``, and ``--server`` fills it."""

    server: str | None


@dataclass(frozen=True)
class Job:
    """One (checkpoint, part of a suite) job of a run.

    Attributes:
        key: the checkpoint's config key.
        checkpoint: the checkpoint as the config gives it (local directory or HF id).
        suite: the suite as configured, which is what ``prepare`` runs on.
        part: the part of the suite this job runs.
        label: the part's label, or None when the suite is one job.
        digest: the digest a result must carry to count as done.
        path: the result file, ``<suite>.json`` or ``<suite>.<label>.json``.
        cached: the result file exists, carries ``digest`` and is a measurement (not a failure).
    """

    key: str
    checkpoint: str
    suite: Suite
    part: Suite
    label: str | None
    digest: str
    path: Path
    cached: bool


def running_vllm() -> str | None:
    """The vLLM version installed in this process, or None when vLLM is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def checkpoint_digests(root: Path) -> dict[str, str | None]:
    """SHA-256 of the checkpoint files whose content changes a measurement (None when a file is absent).

    They are ``generation_config.json`` (sampling), ``tokenizer_config.json`` and the chat template (its own
    ``chat_template.jinja``/``chat_template.json``, else the one embedded in ``tokenizer_config.json``).
    """
    paths = {
        "generation_config": root / "generation_config.json",
        "tokenizer_config": root / "tokenizer_config.json",
    }
    out: dict[str, str | None] = {
        name: sha256_file(path) if path.is_file() else None for name, path in paths.items()
    }
    out["chat_template"] = _template_digest(root)
    return out


def _template_digest(root: Path) -> str | None:
    """The chat template's SHA-256: its own file, else the template embedded in ``tokenizer_config.json``."""
    for name in ("chat_template.jinja", "chat_template.json"):
        path = root / name
        if path.is_file():
            return sha256_file(path)
    config = root / "tokenizer_config.json"
    if config.is_file():
        try:
            template = json.loads(config.read_text()).get("chat_template")
        except (OSError, ValueError):
            return None
        if template is not None:
            return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()
    return None


def checkpoint_files(checkpoint: str) -> dict[str, str]:
    """The present metadata-file digests of a checkpoint that is already a local directory, else nothing.

    A hub id is resolved only when its job runs, so it is named by the configured id (and its pinned revision)
    in the digest; the cached snapshot itself is immutable.
    """
    if not Path(checkpoint).is_dir():
        return {}
    return {name: digest for name, digest in checkpoint_digests(Path(checkpoint)).items() if digest}


def source_record(checkpoint: str, directory: str | None) -> dict[str, Any]:
    """What a result records of the checkpoint it measured: the running engine, commit and metadata digests.

    ``directory`` is the resolved local directory, or None when resolving it failed. The commit is the one a
    hub snapshot path names (``.../snapshots/<commit>``); a plain local directory names none.
    """
    root = Path(directory or checkpoint)
    path = Path(directory) if directory else None
    return {
        "vllm": running_vllm(),
        "checkpoint_source": {
            "commit": path.name if path is not None and path.parent.name == "snapshots" else None,
            **checkpoint_digests(root),
        },
    }


def _digested(value: Any) -> Any:
    """``value`` as the digest reads it: host-local ``serve_env`` keys dropped at any depth.

    A cache location (:data:`~qda_quant.output.vllm_env.CACHE_ENV`: the engines' compile caches, node-local
    temporary directory unless the user names one, or vLLM's, Inductor's and Triton's caches, which
    default under ``$HOME``) changes only where cache files land, not the measurement: two people (or one
    person after moving HOME or to another node) measuring the same checkpoint get the same digest.
    """
    if isinstance(value, dict):
        return {k: _digested(v) for k, v in value.items() if k not in CACHE_ENV}
    if isinstance(value, (list, tuple)):
        return [_digested(v) for v in value]
    return value


def resolve_layouts(cfg: EvalConfig) -> None:
    """Fill every :class:`Layout` suite's ``serve_env`` with the engine's resolved settings, in place.

    The runner and ``--dry-run`` both call :func:`jobs`, which resolves the layouts first: the resolved
    ``serve_env`` is part of each job's digest, so a reader that skipped this step (``--dry-run`` used to)
    computed a different digest and reported every finished job as not cached.
    """
    for suite in cfg.suites:
        if isinstance(suite, Layout):
            suite.serve_env = engine_environment(
                ServeSettings(env=suite.serve_env, nccl_nvls=suite.nccl_nvls)
            )


def digest(
    suite: Suite,
    reference: str,
    checkpoint: str,
    files: dict[str, str] | None = None,
    inputs: dict[str, str | None] | None = None,
) -> str:
    """SHA-256 of a suite's or part's resolved settings and inputs, reference, checkpoint and present files.

    The checkpoint is named as configured; ``files`` are its present metadata digests (none for a hub id, see
    :func:`checkpoint_files`); ``inputs`` are ``suite.inputs()``, when the caller already has them. A suite
    with a ``server`` is a different measurement from the in-process one (the engine batches differently), so
    the engine path joins the key; the server's URL does not, as it is transport and changes every launch.
    A ``serve_env`` key naming a cache location (:data:`~qda_quant.output.vllm_env.CACHE_ENV`) is left out: it
    names where a cache is, not what is measured.
    """
    key = {
        "suite": _digested(suite.to_dict()),
        "inputs": suite.inputs() if inputs is None else inputs,
        "reference": reference,
        "checkpoint": checkpoint,
    }
    if getattr(suite, "server", None):
        key["engine"] = "server"
    if files:
        key["checkpoint_files"] = files
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


#: A part label names ``<suite>.<label>.json``: like a checkpoint key, but without the ``.`` that separates the
#: suite from its label, and so one name segment that cannot escape the checkpoint's result directory.
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def parts(suite: Suite) -> list[tuple[str, Suite, str | None]]:
    """A suite's parts as ``(name, part, label)``, where ``name`` is the result file's stem.

    ``name`` is the suite's name when the part has no label (:meth:`qda_kl.eval.suites.Suite.part_label`), else
    ``<suite>.<label>``.

    Raises:
        Refused: a label is not one name segment (:data:`LABEL`), or two parts share a result file.
    """
    out: list[tuple[str, Suite, str | None]] = []
    for part in suite.parts():
        label = part.part_label()
        if label is not None and not LABEL.fullmatch(label):
            raise Refused(f"suite {suite.name}: label {label!r} is not one name segment ({LABEL.pattern})")
        out.append((suite.name if label is None else f"{suite.name}.{label}", part, label))
    names = [name for name, _, _ in out]
    dup = sorted({name for name in names if names.count(name) > 1})
    if dup:
        raise Refused(f"suite {suite.name}: parts {dup} share a result file; give each part its own label")
    return out


def jobs(cfg: EvalConfig, only: list[str] | tuple[str, ...] = ()) -> list[Job]:
    """The run's jobs in config order, one per (checkpoint, part), restricted to ``only`` when given.

    An ``only`` entry is ``KEY``, ``KEY/SUITE`` (every part of that suite) or ``KEY/SUITE.LABEL`` (one part).

    Raises:
        Refused: an ``only`` entry names no checkpoint key, suite or part of the config, or a suite's parts
            cannot name their result files (see :func:`parts`).
    """
    resolve_layouts(cfg)
    split = [(suite, parts(suite)) for suite in cfg.suites]
    names = {suite.name for suite, _ in split} | {name for _, ps in split for name, _, _ in ps}
    for o in only:
        key, _, name = o.partition("/")
        if key not in cfg.checkpoints or (name and name not in names):
            raise Refused(f"--only {o}: checkpoints {sorted(cfg.checkpoints)}, suites {sorted(names)}")
    # Each part's inputs and each checkpoint's files are hashed once, not once per (checkpoint, part): they
    # read corpora, dataset trees and checkpoint metadata, often on shared storage.
    out, inputs, files = [], {}, {}
    for key, ckpt in cfg.checkpoints.items():
        for suite, ps in split:
            for name, part, label in ps:
                if (
                    only
                    and key not in only
                    and f"{key}/{name}" not in only
                    and f"{key}/{suite.name}" not in only
                ):
                    continue
                if id(part) not in inputs:
                    inputs[id(part)] = part.inputs()
                if ckpt not in files:
                    files[ckpt] = checkpoint_files(ckpt)
                d = digest(part, cfg.reference, ckpt, files[ckpt], inputs[id(part)])
                path = Path(cfg.out) / "results" / key / f"{name}.json"
                out.append(Job(key, ckpt, suite, part, label, d, path, _digest_of(path) == d))
    return out


def _sweep(cfg: EvalConfig) -> None:
    """Delete the result files of parts the config no longer has.

    For every checkpoint and suite, a ``<suite>.json`` or ``<suite>.<label>.json`` under
    ``out/results/<checkpoint>/`` that no current part writes is unlinked (a suite split into parts replaces its
    one-job file, and a removed part leaves nothing behind). The current parts come from the whole config, not
    from ``--only``, so parallel ``--only`` jobs sharing one ``out`` agree on what to delete.
    """
    root = Path(cfg.out) / "results"
    current = {f"{name}.json" for suite in cfg.suites for name, _, _ in parts(suite)}
    names = [suite.name for suite in cfg.suites]
    for key in cfg.checkpoints:
        directory = root / key
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            ours = any(path.stem == name or path.stem.startswith(f"{name}.") for name in names)
            if ours and path.name not in current:
                path.unlink(missing_ok=True)


def _digest_of(path: Path) -> str | None:
    """The ``config_digest`` of a measured result file, or None when it is missing, unreadable or a failure."""
    try:
        result = SuiteResult.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return None
    return None if result.error is not None else result.config_digest


def local(model: str) -> str:
    """A local directory holding ``model``: the path itself, or its HF hub snapshot (``org/name@rev``).

    Every file of the repo is fetched: the suite serves the checkpoint, it does not plan a build of it.

    Raises:
        ValueError: not a local directory, and the hub (or the offline cache) has no such repo or revision.
    """
    from qda_quant.engine.planning import local_source

    return local_source(model, weights=None)


def _plugins(cfg: EvalConfig) -> dict[str, dict[str, Any]]:
    """The plugins a run uses: its suites and any corpus or datasets a suite reads."""
    used: list[tuple[Registry, str, Any]] = [(SUITES, s.name, type(s)) for s in cfg.suites]
    for s in cfg.suites:
        corpus = getattr(s, "corpus", None)
        if isinstance(corpus, Corpus):
            used.append((CORPORA, corpus.name, type(corpus)))
        used += [(DATASETS, d.name, type(d)) for d in getattr(s, "datasets", ())]
    return plugins(used)


def _sources(cfg: EvalConfig) -> dict[str, Any]:
    """The pin for every model this run reads, keyed as the config names them.

    A model named twice (a reference that is also a checkpoint) is signed once.
    """
    seen: dict[str, dict[str, str | None]] = {}

    def pin(model: str) -> dict[str, str | None]:
        key = str(model)
        if key not in seen:
            seen[key] = source_identity(model)
        return dict(seen[key])

    return {
        "reference": pin(cfg.reference),
        "checkpoints": {key: pin(model) for key, model in cfg.checkpoints.items()},
    }


def run(
    cfg: EvalConfig,
    only: list[str] | tuple[str, ...] = (),
    *,
    server: str | None = None,
    log=print,
) -> int:
    """Run the pending jobs; ``log`` gets one JSON line per job.

    Returns:
        0 when every pending job was measured, else the status of the refusals (`qda_quant.exit`): 2 when an
        input was refused, 3 when a job needed something this machine does not have. A run with both is 3: the
        machine has to change before the rest of the run can finish.

    Before the first job, every result file of a part the config no longer has is deleted (:func:`_sweep`). Each
    suite with pending jobs is prepared once in this process, before its first job. A refusal (``Refused`` or a
    plugin's ``ValueError``/``RuntimeError`` from ``prepare``, ``run`` or resolving a model) refuses that job, or
    every job of the suite when ``prepare`` refused, and the run goes on. Any other exception stops the run.

    Raises:
        Refused: an ``only`` entry names nothing in the config.
    """
    if server is not None:
        # A suite opts in by declaring a ``server`` field; the URL is transport, and the suite decides what
        # it sends on it. A suite without the field generates in this process only.
        named = [cast(_Servable, s) for s in cfg.suites if "server" in type(s).model_fields]
        if not named:
            raise Refused("--server: the config has no suite that can generate on a server")
        for suite in named:
            suite.server = server
    todo = jobs(cfg, only)  # resolves the layouts first: the same step `--dry-run` takes

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved(out, cfg.to_dict(), provenance(_plugins(cfg), sources=_sources(cfg)))
    _sweep(cfg)
    code, models, prepared = 0, {}, {}

    def resolve(m: str) -> str:
        if m not in models:
            models[m] = local(m)
        return models[m]

    for job in todo:
        line: dict[str, Any] = {
            "checkpoint": job.key,
            "suite": job.suite.name,
            "part": job.label,
            "cards": job.part.cards(),
        }
        if job.cached:
            log(json.dumps({**line, "status": "cached"}))
            continue
        job.path.unlink(missing_ok=True)
        directory: str | None = None
        try:
            ref = resolve(cfg.reference)
            directory = resolve(job.checkpoint)
            if job.suite.name not in prepared:
                try:
                    job.suite.prepare(ref, out)
                    prepared[job.suite.name] = None
                except Refused as why:  # the suite's refusal: its own status, for every job of the suite
                    prepared[job.suite.name] = Refused(f"prepare: {why}", why.fix, code=why.code)
                except (ValueError, RuntimeError) as why:
                    prepared[job.suite.name] = Refused(f"prepare: {why}")
            if (unprepared := prepared[job.suite.name]) is not None:
                raise Refused(str(unprepared), unprepared.fix, code=unprepared.code)
            t0 = time.time()
            job.part.result_path = job.path
            got = job.part.run(directory, ref, out)
            record = source_record(job.checkpoint, directory)
            if got.detail.get("engine") == "server" and got.detail.get("vllm"):
                # The served suite recorded the server's own version; this client process need not have vLLM
                # installed, so its version (None) must not overwrite what the server reported.
                record["vllm"] = got.detail["vllm"]
            got = got.model_copy(
                update={
                    "checkpoint": job.key,
                    "wall_s": round(time.time() - t0, 3),
                    "config_digest": job.digest,
                    "detail": {**got.detail, **record},
                }
            )
        except (ValueError, RuntimeError) as why:
            code = max(code, why.code if isinstance(why, Refused) else USAGE)
            failed = SuiteResult(
                suite=job.suite.name,
                checkpoint=job.key,
                metrics={},
                higher_is_better={},
                detail=source_record(job.checkpoint, directory),
                config_digest=job.digest,
                error=str(why),
            )
            job.path.parent.mkdir(parents=True, exist_ok=True)
            publish(job.path, failed.model_dump_json(indent=1))
            log(json.dumps({**line, "status": "refused", "why": str(why)}))
            continue
        job.path.parent.mkdir(parents=True, exist_ok=True)
        publish(job.path, got.model_dump_json(indent=1))
        log(json.dumps({**line, "status": "done", "metrics": got.metrics, "wall_s": got.wall_s}))
    return code
