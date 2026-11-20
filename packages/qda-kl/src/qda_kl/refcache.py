"""The reference cache: one reference per (original model, prompts, reading settings, engine), built once per machine.

A reference is the expensive half of every KL reading (a 27B one is minutes of engine start, generation and
teacher forcing), and it is a pure function of its inputs. So it is built once into

    $QDA_CACHE/kl-ref/<model fingerprint[:16]>/<key>/      (a reference directory, as `qda-kl ref` writes it)
    $QDA_CACHE/kl-ref/<model fingerprint[:16]>/<key>.json  (what the key covers, for a reader)

and every later caller -- `qda-kl ref --cache`, `qda-agent prepare`, `qda-agent evaluate`, `qda-kl kl` -- gets
a link to that directory instead of another build. This module is the one place that decides whether a cached
directory is the reference of these inputs; a caller's own checks only decide whether its step runs at all.

The key covers everything the reference's bytes depend on:

- the original's content fingerprint (`qda_quant.intake.fingerprint.model_fingerprint`: every file's full
  bytes, memoized per file by path, size, mtime and inode, so a model is read whole once per machine and a
  changed file is read again);
- the prompt file's sha256;
- the reading settings (answers, seed, max_tokens, max_text_tokens, k, tp, sampling, chat-template arguments,
  serve env and args, NVLS, the gpu memory share, subsets), as the build resolves them: a setting a caller
  leaves out and the value the build derives for it are one setting (`_effective_settings`);
- the engine's versions (vLLM, torch, transformers, qda-kl) and the reference format.

Nothing is read less: a cached reference is the same directory a fresh build of the same inputs wrote.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qda_quant.intake.fingerprint import model_fingerprint
from qda_quant.io import qda_cache, sha256_file
from qda_quant.step import Refused

__all__ = ["cache_root", "cached", "cached_many", "entry", "key_payload", "link", "local_model", "lookup"]

#: Where a cache entry keeps its own copy of the prompt file, recorded relative to the entry: a cached
#: reference outlives the run whose data directory built it, so it may not point into that run.
PROMPTS_COPY = "inputs/prompts.jsonl"


def owns_prompts(target: Path, prompts: str | Path, log: Any = print) -> None:
    """Give the cache entry at ``target`` its own copy of ``prompts`` and record it relative to the entry.

    The reference's manifest records the prompt file as ``PROMPTS_COPY`` beside it, so a later run that links
    the entry reads the cache's copy, not the data directory of the run that built it (which users delete once
    it is done). The copy is byte-identical to the file the key hashed, and the manifest's ``prompts_sha256``
    is checked against it before the record is replaced. `Manifest.content_sha256` is recomputed: it covers the
    data and not the build record, so a data hash is unchanged, and a reference that still records the older
    hash (which covered the record) gets the data hash. A hit whose entry was built before this rule (an absolute
    record) is migrated here too, when the caller still holds the prompt file.

    Raises:
        Refused: the entry's recorded digest is not the copied file's, or the manifest cannot be read.
    """
    from qda_kl.store import Manifest, _content_sha256

    target, prompts = Path(target), Path(prompts)
    manifest_path = target / "manifest.json"
    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{target}: cannot read its manifest to own its prompts ({e})") from None
    if manifest.build.get("prompts") == PROMPTS_COPY and (target / PROMPTS_COPY).is_file():
        return  # already an entry that owns its inputs
    dest = target / PROMPTS_COPY
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    tmp.write_bytes(prompts.read_bytes())
    digest = sha256_file(tmp)
    expected = manifest.build.get("prompts_sha256")
    if expected is not None and digest != expected:
        tmp.unlink(missing_ok=True)
        raise Refused(
            f"{target}: {prompts} is not the prompt file this reference was built on "
            f"(sha256 {digest}, recorded {expected})"
        )
    os.replace(tmp, dest)
    manifest.build["prompts"] = PROMPTS_COPY
    manifest.build.setdefault("prompts_sha256", digest)
    manifest.content_sha256 = _content_sha256(manifest)
    out = manifest_path.with_name(f".{manifest_path.name}.tmp{os.getpid()}")
    out.write_text(manifest.model_dump_json(indent=1))
    os.replace(out, manifest_path)
    log(f"reference cache: {target} now keeps its prompt file ({prompts} -> {PROMPTS_COPY})")


def cache_root() -> Path:
    """``$QDA_CACHE/kl-ref``; without QDA_CACHE ``~/.cache/qda/kl-ref`` (the calibration cache's convention)."""
    return qda_cache("kl-ref")


def local_model(model: str | Path) -> Path:
    """The original's local directory: ``model`` itself when it is one, else the Hub snapshot of ``REPO[@REV]``.

    A Hub id is downloaded into the Hugging Face cache (``HF_HOME``) at the revision it names, or at ``main``
    resolved to a commit now; the snapshot directory is what the fingerprint reads and the engine loads.

    Raises:
        Refused: offline, the cache holds no snapshot for that revision (`offline_refusal`).
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    from qda_quant.intake.model_ref import offline_refusal, parse

    if (missing := offline_refusal(model)) is not None:
        raise missing
    ref = parse(model)
    return Path(snapshot_download(ref.name, **ref.hub()))


def _effective_settings(model: str | Path, settings: dict[str, Any]) -> dict[str, Any]:
    """The settings a build actually uses, from the flags a caller gives.

    A caller may leave a setting out and let the build derive it (``sampling`` from the model's released cards
    or its ``generation_config.json``, ``serve_args`` from the deterministic-reduction pin) or spell out the
    very value the build derives to; both are the same reference, so the key holds the effective value only.
    The build's own signature is the one owner of the defaults, and
    `engine_settings`/`reference_sampling` are the one owners of the derived values.

    Raises:
        Refused: the explicit ``sampling``, the model's released cards and its ``generation_config.json``
            name none.
    """
    from qda_kl.reference import build, engine_settings, reference_sampling

    defaults = {
        name: param.default
        for name, param in inspect.signature(build).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    got = {name: settings.get(name, default) for name, default in defaults.items()}
    engine = engine_settings(got["serve_env"], got["serve_args"], bool(got["nccl_nvls"]))
    got["sampling"] = reference_sampling(model, got["sampling"])
    got["serve_env"] = dict(engine.env)
    got["serve_args"] = dict(engine.args)  # the pin the build adds, not the caller's spelling of it
    got["nccl_nvls"] = engine.nccl_nvls
    got["chat_template_kwargs"] = dict(got["chat_template_kwargs"] or {})
    got["subsets"] = settings.get("subsets") or None
    return got


def key_payload(model: str | Path, prompts: str | Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Everything a reference's bytes depend on, as one JSON-able mapping.

    ``settings`` are the flags a caller gives; the key holds their effective values (`_effective_settings`), so
    an omitted setting and its spelled-out default never make two entries for one reference.

    Raises:
        Refused: ``model_fingerprint`` could not read a file of the model's, or neither the settings, the
            model's released cards nor its ``generation_config.json`` name a sampling setting.
    """
    from qda_kl.reference import engine_settings, versions
    from qda_kl.store import FORMAT

    effective = _effective_settings(model, settings)
    engine = engine_settings(effective["serve_env"], effective["serve_args"], effective["nccl_nvls"])
    return {
        "format": FORMAT,
        "model": model_fingerprint(model),
        "prompts_sha256": sha256_file(Path(prompts)),
        "settings": {k: effective[k] for k in sorted(effective)},
        # what the engine is started with, derived in code from the flags: a changed default is a new key
        "engine": engine.model_dump(mode="json", include={"env", "args", "nccl_nvls"}),
        "versions": versions(),
    }


def entry(payload: dict[str, Any]) -> Path:
    """The cache directory of the reference ``payload`` describes."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_root() / payload["model"][:16] / digest[:24]


def lookup(payload: dict[str, Any]) -> Path | None:
    """The cached reference of ``payload``, or None. A reference directory is complete once it has a manifest:
    `qda_kl.store.ReferenceWriter` moves the finished directory into place in one rename."""
    found = entry(payload)
    if (found / "manifest.json").is_file() and found.with_suffix(".json").is_file():
        return found
    return None


def link(out: Path, target: Path) -> None:
    """Make ``out`` a link to ``target``: replaces a link (a stale or dangling one included), never a directory.

    Raises:
        Refused: ``out`` exists and is not a link (a reference built in place, or anything else).
    """
    out = Path(out)
    if out.is_symlink():
        if Path(os.readlink(out)) == target:
            return
        out.unlink()
    elif out.exists():
        raise Refused(f"{out} exists and is not a link to the reference cache; remove it to use the cache")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.link")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target, target_is_directory=True)
    tmp.replace(out)


#: How often a call that waits for another process's build of the same reference says who it waits for.
PROGRESS_S = 300.0


@contextmanager
def _waiting_lock(path: Path, log: Any) -> Iterator[str | None]:
    """Hold an exclusive lock on ``path``, waiting for another build of the same reference to finish.

    The lock is `fcntl.flock`, which the kernel releases when its holder exits, so a holder that died is seen
    at once and the wait has no timeout: a healthy build of a large reference runs for an hour or more. Every
    `PROGRESS_S` the wait logs the holder's line (pid, host, start) and the time waited, so the user sees whom
    the run waits for.

    Yields the other holder's line when this call waited for it, else None.
    """
    from qda_kl.reference import _holder

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        other = None
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            other = _holder(handle)
            log(f"another process is building this reference; waiting for it ({path}: {other})")
            started = said = time.monotonic()
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - said >= PROGRESS_S:
                        said = time.monotonic()
                        log(
                            f"still waiting for {other} to build this reference ({said - started:.0f} s so far)"
                        )
                    time.sleep(0.2)
            log(f"waited {time.monotonic() - started:.0f} s for the other build")
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid {os.getpid()} on {os.uname().nodename} since {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        )
        handle.flush()
        yield other


def cached(
    model: str | Path,
    prompts: str | Path,
    out: str | Path | None,
    settings: dict[str, Any],
    log: Any = print,
) -> tuple[Path, bool]:
    """The cached reference of these inputs, built into the cache first when there is none; ``out`` links to it.

    ``settings`` are `qda_kl.reference.build`'s keyword arguments. Two processes asking for the same reference
    build it once: the second waits for the first and reuses its directory, and refuses when the first
    released the lock without building it.

    Raises:
        Refused: the run this call waited for released the lock without building the reference.

    Returns:
        The cache directory and whether it was built by this call.
    """
    model = local_model(model)
    payload = key_payload(model, prompts, settings)
    target = entry(payload)
    hit = lookup(payload)
    built = False
    if hit is None:
        with _waiting_lock(target.with_name(f".{target.name}.cache-lock"), log) as other:
            if lookup(payload) is None and other is not None:  # the run this one waited for built nothing
                raise Refused(
                    f"reference cache: the other run building {target} ({other}) released it without building it",
                    fix="check that run's log, then run this again (it builds the reference when no run holds it)",
                )
            if lookup(payload) is None:
                from qda_kl.reference import build

                log(f"reference cache: building {target} (later calls on this model and prompts reuse it)")
                # the key first: the reference lands in one rename, so a manifest means a finished entry
                target.with_suffix(".json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
                build(str(model), prompts, target, **settings)
                built = True
    else:
        log(f"reference cache: reusing {target}")
    owns_prompts(target, prompts, log)  # the entry keeps what it needs, not the caller's path
    if out is not None:
        link(Path(out), target)
    return target, built


def cached_many(
    model: str | Path,
    groups: Sequence[Any],
    settings: dict[str, Any],
    log: Any = print,
) -> tuple[list[Path], list[bool]]:
    """The cached reference of each group, built once per machine; each group's ``out`` links to its entry.

    Every group is a :class:`qda_kl.reference.Group`: its prompts, its optional output link and its named
    subsets. The groups missing from the cache are built in **one engine load**
    (:func:`qda_kl.reference.build_many`); a group already cached costs no engine. A group whose ``out`` is
    None only warms the cache (``qda-agent prepare`` warms the holdout reference this way, and `evaluate` links
    it into the run when it is needed). Two processes asking for the same references build each once: the
    second waits on the entry's lock.

    Returns:
        The cache directory and whether this call built it, per group, in the order given.
    """
    from contextlib import ExitStack

    from qda_kl.reference import Group, build_many

    model = local_model(model)
    common = {k: v for k, v in settings.items() if k != "subsets"}
    plans = []
    for g in groups:
        payload = key_payload(model, g.prompts, {**common, "subsets": g.subsets or None})
        plans.append((g, payload, entry(payload)))
    built: dict[Path, bool] = {target: False for _, _, target in plans}
    missing = [plan for plan in plans if lookup(plan[1]) is None]
    if missing:
        with ExitStack() as stack:
            for _, _, target in sorted(missing, key=lambda plan: str(plan[2])):
                stack.enter_context(_waiting_lock(target.with_name(f".{target.name}.cache-lock"), log))
            still = [
                plan for plan in missing if lookup(plan[1]) is None
            ]  # another process may have built one
            if still:
                log(
                    f"reference cache: building {len(still)} reference(s) in one engine load "
                    "(later calls on the same model, prompts and settings reuse them)"
                )
                for _, payload, target in still:
                    # the key first: the reference lands in one rename, so a manifest means a finished entry
                    target.with_suffix(".json").write_text(
                        json.dumps(payload, indent=1, sort_keys=True) + "\n"
                    )
                build_many(
                    str(model),
                    [Group(g.prompts, target, g.subsets) for g, _, target in still],
                    **common,
                )
                for _, _, target in still:
                    built[target] = True
    for g, payload, target in plans:
        if not built[target] and lookup(payload) is not None:  # a hit, or another process's build
            log(f"reference cache: reusing {target}")
        owns_prompts(target, g.prompts, log)  # the entry keeps what it needs, not the caller's path
        if g.out is not None:
            link(Path(g.out), target)
    return [target for _, _, target in plans], [built[target] for _, _, target in plans]
