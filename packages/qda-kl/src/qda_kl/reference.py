"""Build a reference for an original model: its own answers to held-out prompts and its top-k at every token.

The prompt file is JSONL, one object per line::

    {"id": "aime-3", "domain": "math", "prompt": "..."}     # the model answers it (``answers`` times)
    {"id": "wiki-17", "domain": "web", "text": "..."}       # human-written text, only teacher-forced

A row may also carry ``group`` (``hard``/``easy``) and ``split`` (``eval``/``holdout``), which the eval table
reports (see :mod:`qda_kl.eval.report`).

Each prompt is rendered with the model's chat template and answered ``answers`` times, draw ``d`` at seed
``seed + d``. Sampling is a config setting: the explicit ``sampling`` mapping wins, field by field, over the row
the model's released cards record (the ``full`` preset's one table:
:func:`qda_kl.eval.full_preset.card_sampling`), and a model the table does not name is sampled as its
``generation_config.json`` says; with none of the three, the build is refused (never a silent vLLM default).
:func:`reference_sampling` is that rule, and every reference build goes through
it. Each answer is kept as token ids (never re-tokenized). Then every sequence is teacher-forced through the
same engine and the hook keeps the model's top-k log-probabilities and remaining mass at each scored position
(:mod:`qda_kl.vllm_hooks`), written with :class:`qda_kl.store.ReferenceWriter`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from qda_quant.calib.selfgen import merge_sampling, resolve_sampling, sampling_params
from qda_quant.intake.model_ref import offline_refusal
from qda_quant.step import Refused

from qda_kl.engine import Item, open_engine, teacher_force
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.serve import (
    ServeSettings,
    batch_args,
    check_serve_env,
    deterministic_args,
    engine_environment,
    no_jit,
    pinned,
)
from qda_kl.store import TOKENIZER_DIR, Manifest, ReferenceWriter
from qda_kl.tokenizer_check import engine_tokenizer, preload_refusal, tokenizer_refusal
from qda_kl.topk import TOPK, TopK

#: Answers shorter than this many tokens are dropped (too few positions to say anything).
MIN_SCORED = 16

#: Above this share of a prompt file's rows, rows past the model's context refuse the build: a build that
#: skipped more than one row in twenty is a corpus built for another model, not a holdout with a long tail.
OVER_CONTEXT_SHARE = 0.05

#: The sampling a reference is built with when the model's ``generation_config.json`` names none.
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.95}


def default_sampling(model: str | Path) -> dict | None:
    """The ``sampling`` a reference of ``model`` is given: None when the build resolves it by itself.

    The one rule every caller that builds a reference for a user follows (`qda-agent prepare`, `qda-kl kl`), so
    their references of one model are the same reference and the cache serves both. None when the model's
    released cards name a row or its ``generation_config.json`` names a setting: the build takes those
    (:func:`reference_sampling`). :data:`DEFAULT_SAMPLING` answers for a model neither names -- a reference is
    always built here, so this path never refuses. It delegates to the model reader the build itself uses, so a
    corrupt file fails here as it fails there instead of being ignored.
    """
    from qda_quant.calib.selfgen import sampling_of

    from qda_kl.eval.full_preset import card_sampling

    if card_sampling(str(model)) is not None or sampling_of(model):
        return None  # the build reads the cards' row, or the file, itself
    return dict(DEFAULT_SAMPLING)


def reference_sampling(model: str | Path, sampling: dict | None = None) -> dict:
    """The sampling a reference of ``model`` is built with: the explicit setting over the released cards' row
    over the model's own file.

    One rule for every build -- ``qda-kl ref`` with or without ``--cache``, the cache's key, and the flag
    ``qda-agent prepare`` gives its ref step -- so one model has one reference and the cache serves all callers.
    The cards' row is the ``full`` preset's table, so a reference and a ``full`` eval of one model are sampled
    alike; it is read as a whole row, so a field the row omits is omitted here too and the model's file never
    tops it up. A model the table does not name takes its own ``generation_config.json``; a field the explicit
    setting names wins over either.

    Raises:
        Refused: none of the three names a sampling setting; the message names ``--sampling`` and gives an
            example to copy.
    """
    from qda_kl.eval.full_preset import card_sampling

    card = card_sampling(str(model))
    if card is None:
        return resolve_sampling(model, sampling, "--sampling")
    # the row wins over the file, and the file is not read at all: a field the row omits stays omitted
    return merge_sampling(card[0], sampling, "--sampling")


def sampling_note(model: str | Path, sampling: dict | None = None) -> str | None:
    """The line ``qda-kl ref`` prints when the released cards supplied (some of) the reference's sampling.

    None when the table names no row for the model, or the explicit setting already names every field the row
    has (the row then supplied nothing).
    """
    from qda_kl.eval.full_preset import card_sampling

    card = card_sampling(str(model))
    if card is None or not set(card[0]) - set(sampling or {}):
        return None
    return f"ref: sampling of {card[1]}'s released cards"


class Prompt(BaseModel):
    """One line of a prompt file: ``prompt`` (answered) or ``text`` (teacher-forced as written).

    ``group`` and ``split`` are optional labels the report groups sequences by (``hard``/``easy``, and
    ``eval``/``holdout``); a prompt without them is reported under no group and no split.
    """

    id: str
    domain: str = "default"
    group: str = ""
    split: str = ""
    prompt: str | None = None
    text: str | None = None


def read_prompts(path: str | Path) -> list[Prompt]:
    """Read and validate JSONL prompts in file order.

    Raises:
        Refused: the file is missing, or a line gives both or neither of ``prompt`` and ``text``.
    """
    if not Path(path).is_file():
        raise Refused(f"{path}: no such prompt file")
    out = [Prompt.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for p in out:
        if (p.prompt is None) == (p.text is None):
            raise Refused(f"{p.id}: give exactly one of prompt / text")
    return out


def dropped_prompts(build: dict[str, Any]) -> set[str]:
    """The prompts whose every answer draw was dropped as too short: the build covered them, it never scored them.

    :func:`build` records each dropped draw as ``<prompt>#<draw>`` in ``build["dropped_short"]``. A prompt counts
    only when all of its ``answers`` draws are there: one surviving draw has a sequence and is scored. A corpus
    that holds such a prompt is still covered by the reference, by a whole build or by a merge of its shards.
    """
    dropped = set(build.get("dropped_short") or ())
    ids = {draw.rsplit("#", 1)[0] for draw in dropped}
    return {p for p in ids if all(f"{p}#{d}" in dropped for d in range(int(build.get("answers", 1))))}


def versions() -> dict[str, str | None]:
    """Return versions of the libraries used to build and score references."""
    got = {}
    for name in ("vllm", "torch", "transformers", "qda-kl"):
        try:
            got[name] = version(name)
        except PackageNotFoundError:
            got[name] = None
    return got


def engine_settings(
    serve_env: dict[str, str] | None, serve_args: dict | None, nccl_nvls: bool = False
) -> ServeSettings:
    """The engine settings a reference is built with, from its flags.

    The one place that turns the flags into what the engine gets (`deterministic_args`, then `no_jit`, the
    settings every scoring and gencheck engine starts with): `build` starts
    its engine from it, and the reference cache (`qda_kl.refcache`) keys on it, so a change to these defaults
    in code is a new cache entry, never a stale reference reused under the old key.
    """
    settings = ServeSettings(
        env=serve_env or {}, args=deterministic_args(serve_args or {}), nccl_nvls=nccl_nvls
    )
    return no_jit(settings)  # as every scoring and gencheck engine: no FlashInfer kernel is JIT-built


def model_identity(model: str | Path, revision: str | None = None, *, hub: bool = False) -> dict:
    """Record the model's name, path and config.json SHA-256 hash, and the Hub revision it was loaded at.

    With `hub`, a Hub id's hash is its cached snapshot's config.json at `revision` (``main`` when unpinned),
    and ``commit`` names that snapshot.
    """
    p = Path(model)
    cfg = p / "config.json"
    cached = cached_config(str(model), revision) if hub and not p.exists() else None
    cfg = cached or cfg
    return {
        "name": p.name,
        "path": str(p),
        "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.is_file() else None,
        **({"revision": revision} if revision else {}),
        **({"commit": cached.parent.name} if cached else {}),
    }


def cached_config(model: str, revision: str | None = None) -> Path | None:
    """The config.json of Hub id `model` (``org/name[@rev]``) in the local Hugging Face cache at `revision`, or None."""
    from qda_quant.intake.model_ref import parse

    ref = parse(model)
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(ref.name, "config.json", revision=revision or ref.revision, local_files_only=True)
        )
    except Exception:  # noqa: BLE001 - not cached, not a Hub id, or no huggingface_hub
        return None


def tokenizer_identity(model: str | Path, into: Path, *, revision: str | None = None) -> dict:
    """Write the anchor's tokenizer into `into` and return what the manifest records about it.

    The reference carries the tokenizer a score needs (small files, never weights), so a score reads no model
    directory; the record still names the resolved directory it was copied from, so a reader can see which
    tokenizer the reference was built with and compare it with the original while the original exists. The
    copy is the corrected form ``qda build``'s rule produces, and the record is that rule's own
    (:func:`qda_kl.tokenizer_check.save_tokenizer`).
    """
    from qda_kl.tokenizer_check import save_tokenizer

    source, record = save_tokenizer(model, into, revision=revision)
    got: dict = {"dir": TOKENIZER_DIR if record else None, "path": str(source.resolve()) if source else None}
    return {**got, **({"revision": revision} if revision else {}), **({"record": record} if record else {})}


@contextmanager
def staged_tokenizer(out: str | Path, name: str = "tokenizer") -> Iterator[Path]:
    """A path beside ``out`` for a tokenizer the build writes (``name``), removed afterwards.

    It sits beside ``out`` (as the lock does) because ``out`` itself is created only by the atomic close; the
    build holds the lock, so no other build stages into the same path. Only the path is yielded: it is created
    when a correction applies (:func:`qda_kl.tokenizer_check.engine_tokenizer`), or to hold the anchor
    tokenizer copied into the reference (:func:`tokenizer_identity`).
    """
    path = Path(out).with_name(f".{Path(out).name}.{name}")
    shutil.rmtree(path, ignore_errors=True)  # a crashed build's leftover
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def output_lock(out: str | Path) -> Iterator[None]:
    """Hold an exclusive lock for the reference directory ``out`` for the length of a build.

    The lock file sits beside ``out``: ``out`` itself is never created before the atomic close, and a file
    inside it would break the never-overwrite guard. ``flock`` releases the lock when the process exits, a
    crash included, so a lock file left by a dead build never blocks the next one; removing the file instead
    would let two later builds lock two different inodes.

    Raises:
        Refused: another process holds the lock; the message names its pid, host and start time.
    """
    path = Path(out).with_name(f".{Path(out).name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused(f"{out}: another build holds the lock ({_holder(handle)})") from None
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid {os.getpid()} on {socket.gethostname()} since "
            f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield


def _holder(handle) -> str:
    """Read the holder line; a process that just took the lock may not have written it yet."""
    for _ in range(5):
        handle.seek(0)
        line = handle.read().strip()
        if line:
            return line
        time.sleep(0.01)  # the holder writes immediately after locking; do not race it to an empty read
    return "holder not yet recorded"


@dataclass(frozen=True)
class Group:
    """One prompt file and the reference directory its build writes.

    ``subsets`` are the named prompt-id subsets this group's manifest records (``search96``), so a scorer
    reads one by name (:func:`build_many`). Two groups never write the same directory.
    """

    prompts: str | Path
    out: str | Path
    subsets: dict[str, list[str]] | None = None


def build(
    model: str,
    prompts: str | Path,
    out: str | Path,
    *,
    answers: int = 2,
    seed: int = 17,
    max_tokens: int = 8192,
    max_text_tokens: int = 4096,
    k: int = TOPK,
    tp: int = DEFAULT_TP,
    serve_env: dict[str, str] | None = None,
    serve_args: dict | None = None,
    nccl_nvls: bool = False,
    chat_template_kwargs: dict | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    sampling: dict | None = None,
    subsets: dict[str, list[str]] | None = None,
) -> Manifest:
    """Build a reference from ``model`` and ``prompts`` in ``out``, and return its manifest.

    This is :func:`build_many` with one :class:`Group`; see it for the settings and what the record holds.

    Raises:
        Refused: another build holds the lock on ``out``, none of ``sampling``, the model's released cards
            and its ``generation_config.json`` names a sampling setting, ``tp`` is invalid, a
            prompt leaves no room for an answer in the model's window, or the engine's tokenizer does not split
            text like the checkpoint's ``tokenizer.json`` (checked before the engine loads, and again on the
            engine's own tokenizer), or ``out`` is already there (a reference is never overwritten).
    """
    return build_many(
        model,
        [Group(prompts, out, subsets)],
        answers=answers,
        seed=seed,
        max_tokens=max_tokens,
        max_text_tokens=max_text_tokens,
        k=k,
        tp=tp,
        serve_env=serve_env,
        serve_args=serve_args,
        nccl_nvls=nccl_nvls,
        chat_template_kwargs=chat_template_kwargs,
        gpu_memory_utilization=gpu_memory_utilization,
        sampling=sampling,
    )[0]


def build_many(
    model: str,
    groups: Sequence[Group],
    *,
    answers: int = 2,
    seed: int = 17,
    max_tokens: int = 8192,
    max_text_tokens: int = 4096,
    k: int = TOPK,
    tp: int = DEFAULT_TP,
    serve_env: dict[str, str] | None = None,
    serve_args: dict | None = None,
    nccl_nvls: bool = False,
    chat_template_kwargs: dict | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    sampling: dict | None = None,
) -> list[Manifest]:
    """Build one reference per prompt group with one engine load, and return their manifests in order.

    Every group reads the same model with the same settings, so the engine costs one load instead of one per
    group: its window holds the longest prompt of any group, its batch the most answers any group generates at
    once (:func:`qda_kl.serve.batch_args`). Answers are generated and teacher-forced group by group, and each
    group writes its own directory and manifest (:class:`qda_kl.store.ReferenceWriter`), so a group's reference
    is what a build of that group alone would write, apart from the shared window and ``max_num_seqs`` the
    engine records.

    Args:
        model: the original model's directory, or its hub id (``org/name[@rev]``).
        groups: the prompt files and the reference directories to write (see :class:`Group`).
        answers: answers per prompt.
        seed: the seed of draw 0 (draw ``d`` uses ``seed + d``).
        max_tokens: the requested cap of one answer; an answer is shorter when the window leaves less room, and the
            manifest records each such answer's cap in ``build["capped_answers"]``.
        max_text_tokens: human texts are truncated to this many tokens.
        k: top-k kept per position.
        tp: tensor-parallel size.
        serve_env: engine environment variables, as ``scorer.topk.serve_env``; they override the defaults
            (``NCCL_NVLS_ENABLE`` from ``nccl_nvls``), and may not name another value for it.
        serve_args: extra ``vllm.LLM`` arguments, such as expert and pipeline parallelism. ``max_num_seqs``
            defaults to the answers generated at once (:func:`qda_kl.serve.batch_args`); the manifest
            records the effective value as ``build["max_num_seqs"]``.
        nccl_nvls: let the engine's ranks join NCCL's NVLS collectives (:attr:`qda_kl.scorers.TopKScorer.nccl_nvls`).
        chat_template_kwargs: extra chat-template arguments (e.g. ``{"enable_thinking": False}``).
        gpu_memory_utilization: vLLM's.
        sampling: the sampling settings; its fields win over the settings the model's released cards record,
            which win over its ``generation_config.json`` (:func:`reference_sampling`).

    Raises:
        Refused: another build holds the lock on an output, two groups name one output, none of ``sampling``,
            the model's released cards and its ``generation_config.json`` names a sampling setting, ``tp`` is
            invalid, a prompt leaves no room for an answer in the model's window, or the engine's tokenizer does
            not split text like the checkpoint's ``tokenizer.json`` (checked before the engine loads, and again
            on the engine's own tokenizer), or an output directory is already there (a reference is never
            overwritten; checked before the engine loads).

    When the model declares a tokenizer class that rebuilds its ByteLevel ``tokenizer.json``, the engine is
    given the tokenizer ``qda build`` would write instead (``tokenizer_class`` corrected, every other file the
    model's own) unless ``serve_args`` names a ``tokenizer``; each manifest records the build's record under
    ``build["tokenizer_config"]``.

    That same tokenizer is copied into the reference (``tokenizer/``), so scoring against it needs no model
    directory; ``model["tokenizer"]`` records where it came from (:func:`tokenizer_identity`).
    """
    groups = [Group(Path(g.prompts), Path(g.out), g.subsets) for g in groups]
    if not groups:
        raise ValueError("build_many needs at least one prompt group")
    outs = [Path(g.out) for g in groups]
    if len(set(outs)) != len(outs):
        raise Refused(f"two groups write the same reference directory ({outs[0]})")
    with ExitStack() as stack:
        for out in sorted(outs, key=str):  # one order for two concurrent builds of the same outputs
            stack.enter_context(output_lock(out))
        stage = stack.enter_context(staged_tokenizer(groups[0].out))
        # the anchor tokenizer copied into every group's reference (ReferenceWriter copies it; one staging serves all)
        anchor = stack.enter_context(staged_tokenizer(groups[0].out, "anchor"))
        items = [read_prompts(g.prompts) for g in groups]
        for g, group_items in zip(groups, items, strict=True):
            known = {p.id for p in group_items}
            for name, ids in (g.subsets or {}).items():
                if missing := [i for i in ids if i not in known]:
                    raise Refused(
                        f"subset {name}: {len(missing)} ids are not prompts of {g.prompts} "
                        f"(first {missing[0]})"
                    )
        for out in outs:  # before any GPU work, and before this call writes another group
            if (out / "manifest.json").is_file():
                raise Refused(
                    f"{out} already holds a reference; qda-kl ref never overwrites one",
                    f"score against it with `qda-kl score` (`reference: {out}`), or pass another --out",
                )
            if os.path.lexists(out):  # a dangling link too: the rename onto it would fail after the build
                raise Refused(
                    f"{out} exists and holds no reference; qda-kl ref writes only a new directory",
                    "pass another --out",
                )
        if (missing := offline_refusal(model)) is not None:  # offline, a revision the cache cannot name
            raise missing
        settings = reference_sampling(model, sampling)  # before any GPU work
        serve = engine_settings(serve_env, serve_args, nccl_nvls)
        # ``org/name@rev``: every read below, the engine and the manifest take the revision from serve.args
        model, args = pinned(model, serve.args)
        serve = serve.model_copy(update={"args": args})
        check_serve_env(serve.env, nccl_nvls)  # before any GPU work
        engine_args, corrected = engine_tokenizer(model, serve.args, stage)  # the build's class correction
        source = engine_args.get("tokenizer") or model
        revision = serve.args.get("tokenizer_revision") or serve.args.get("revision")
        firsts = tuple(next((p.prompt or p.text for p in group_items), None) for group_items in items)
        options = {"trust_remote_code": True} if serve.args.get("trust_remote_code") else {}
        if refusal := preload_refusal(source, revision=revision, probes=firsts, **options):
            raise Refused(refusal)  # before the weights load
        rows = [len(group_items) for group_items in items]
        counts = prompt_token_counts(  # every group, one tokenizer load
            source,
            [p for group_items in items for p in group_items],
            revision=revision,
            max_text_tokens=max_text_tokens,
            chat_template_kwargs=chat_template_kwargs,
            **options,
        )
        per_group, start = [], 0
        for n in rows:
            per_group.append(counts[start : start + n])
            start += n
        counts = per_group
        # One row the model's window cannot hold used to refuse the whole holdout (#...): it is dropped and
        # reported instead, by the one rule `longest_prompt` counts by. A file with more than
        # OVER_CONTEXT_SHARE such rows is still refused, so a corpus built for another model is never silently
        # read on the handful of rows that happen to fit.
        context = context_length(model)
        kept: list[list[Prompt]] = []
        over: list[list[str]] = []
        longest = 0
        first_over = None
        for group_items, group_counts in zip(items, counts, strict=True):
            keep, lost = [], []
            for p, (_, n) in zip(group_items, group_counts, strict=True):
                if n + 2 > context:  # one answer token, plus the one teacher-forcing adds
                    lost.append(p.id)
                    first_over = first_over or p.id
                else:
                    keep.append(p)
                    longest = max(longest, n)
            kept.append(keep)
            over.append(lost)
        dropped_over = sum(len(lost) for lost in over)
        total = sum(rows)
        if dropped_over and dropped_over / total > OVER_CONTEXT_SHARE:
            raise Refused(
                f"{first_over}: {dropped_over} of {total} prompts ({dropped_over / total:.1%}) leave no answer "
                f"room in the model's {context}-token context; shorten them or use a model with a longer context"
            )
        items = kept  # the engine, the answers and the manifest see only the rows that fit
        from vllm import TokensPrompt

        # the answers generated at once, not vLLM's default: a hybrid model holds one Mamba block each;
        # the largest group's count, since the groups generate one after another
        answers_at_once = max(answers * sum(p.text is None for p in group_items) for group_items in items)
        engine_args = batch_args(engine_args, answers_at_once)
        t0 = time.time()
        llm = open_engine(
            model,
            tp=tp,
            max_model_len=max_model_len(model, max_tokens, longest),
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            settings=serve.model_copy(update={"args": engine_args}),
        )
        engine_s = time.time() - t0
        tok = llm.get_tokenizer()
        refusal = tokenizer_refusal(tok, source, revision=revision)
        for probe in firsts:
            if refusal or not probe:
                continue
            refusal = tokenizer_refusal(tok, source, revision=revision, probe=probe)
        if refusal:
            raise Refused(refusal)
        window = int(llm.llm_engine.model_config.max_model_len)
        vocab = int(llm.llm_engine.model_config.get_vocab_size())
        dtype = str(llm.llm_engine.model_config.dtype).removeprefix("torch.")
        tokenizer = tokenizer_identity(model, anchor, revision=revision)
        manifests = []
        started = t0  # the first group's total includes the engine load; each later one its own work
        engine_start_s = engine_s  # the one load, attributed to the group that paid it
        for i, (g, group_items) in enumerate(zip(groups, items, strict=True)):
            jobs, todo = [], []  # (prompt, draw, prompt ids), sequences to teacher-force
            for p in group_items:
                if p.text is not None:
                    ids = tok(p.text, add_special_tokens=True)["input_ids"][:max_text_tokens]
                    todo.append((p, None, Item(p.id, list(ids), 1), None))
                    continue
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": p.prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    **(chat_template_kwargs or {}),
                )
                ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
                jobs += [(p, d, ids) for d in range(answers)]
            caps = []
            for p, _, ids in jobs:
                cap = (
                    window - 1 - len(ids)
                )  # teacher-forcing submits the answer as a prompt and adds one token
                if cap < 1:
                    raise ValueError(
                        f"{p.id}: {len(ids)} prompt tokens leave no answer room in the {window}-token window"
                    )
                caps.append(min(max_tokens, cap))
            params = [
                sampling_params(settings, cap, seed + d) for (_, d, _), cap in zip(jobs, caps, strict=True)
            ]
            capped = {
                f"{p.id}#{d}": cap for (p, d, _), cap in zip(jobs, caps, strict=True) if cap < max_tokens
            }
            t_gen = time.time()
            said = llm.generate(
                [TokensPrompt(prompt_token_ids=ids) for _, _, ids in jobs], params, use_tqdm=False
            )
            gen_s = time.time() - t_gen
            dropped = []
            for (p, d, ids), o in zip(jobs, said, strict=True):
                c = o.outputs[0]
                if len(c.token_ids) < MIN_SCORED:
                    dropped.append(f"{p.id}#{d}")
                    continue
                todo.append((p, d, Item(f"{p.id}#{d}", ids + list(c.token_ids), len(ids)), c.finish_reason))
            todo.sort(
                key=lambda x: (
                    [i for i, p in enumerate(group_items) if p.id == x[0].id][0] * 1000 + (x[1] or 0)
                )
            )
            t_tf = time.time()
            got = teacher_force(llm, "ref", [x[2] for x in todo], k=k)
            tf_s = time.time() - t_tf
            recorded_sampling = {**settings, "max_tokens": max_tokens}
            w = ReferenceWriter(
                Path(g.out),
                k=k,
                vocab=vocab,
                subsets=dict(g.subsets or {}),
                tokenizer=anchor if tokenizer["dir"] else None,
                model={
                    **model_identity(model, serve.args.get("revision"), hub=True),
                    "dtype": dtype,
                    "engine": f"vllm {versions()['vllm']}",
                    "tokenizer": tokenizer,
                },
                build={
                    "prompts": str(g.prompts),
                    "prompts_sha256": hashlib.sha256(Path(g.prompts).read_bytes()).hexdigest(),
                    "answers": answers,
                    "seed": seed,
                    "sampling": recorded_sampling,
                    "chat_template_kwargs": chat_template_kwargs or {},
                    "max_text_tokens": max_text_tokens,
                    "dropped_short": dropped,
                    # rows past the model's context, with the file's row count: reported, not fatal (see above)
                    **(
                        {"dropped_over_context": over[i], "prompt_rows": rows[i], "context": context}
                        if over[i]
                        else {}
                    ),
                    # only when it binds: an uncapped build keeps the manifest it had, identity included
                    **({"capped_answers": capped} if capped else {}),
                    # only when the engine got the corrected tokenizer: qda build's record, as its build.json holds it
                    **({"tokenizer_config": corrected} if corrected else {}),
                    "tp": tp,
                    "serve_env": engine_environment(serve),
                    "serve_args": dict(serve.args),
                    # the effective batch, from this shard's own answer count unless serve_args names it
                    "max_num_seqs": engine_args["max_num_seqs"],
                    "nccl_nvls": nccl_nvls,
                    "versions": versions(),
                    "seconds": {
                        "engine_start": round(engine_start_s, 1),
                        "generate": round(gen_s, 1),
                        "teacher_force": round(tf_s, 1),
                        "total": round(time.time() - started, 1),
                    },
                },
            )
            for p, _, it, finish in todo:
                r = got[it.key]
                w.add(
                    it.key,
                    p.domain,
                    it.tokens,
                    it.start,
                    TopK(*(_from_numpy(r[a]) for a in ("topv", "topi", "log_rest"))),
                    group=p.group,
                    split=p.split,
                    kind="text" if p.text is not None else "answer",
                    finish=finish,
                )
            manifests.append(w.close())
            started = time.time()
            engine_start_s = 0.0
        return manifests


def _from_numpy(array):
    """Wrap a NumPy array as a tensor without copying its data."""
    import torch

    return torch.from_numpy(array)


def context_length(model: str) -> int:
    """The model's context length (``max_position_embeddings``, or its ``text_config``'s); unbounded unknown."""
    cfg = Path(model) / "config.json"
    raw = json.loads(cfg.read_text()) if cfg.is_file() else {}
    return int(
        raw.get("max_position_embeddings")
        or (raw.get("text_config") or {}).get("max_position_embeddings")
        or 1 << 30
    )


def window(model: str, tokens: int) -> int:
    """An engine window for sequences of up to ``tokens`` tokens, capped by the model's own context.

    The one rule every engine qda-kl opens on a model follows (the reference's and the scorer's): a window
    above the model's context is refused by vLLM at engine start, after the reference was already built.
    """
    return min(context_length(model), tokens)


def max_model_len(model: str, max_tokens: int, prompt_tokens: int = 0) -> int:
    """Room for the longest prompt (at least 8192 tokens) plus ``max_tokens``, capped by the model's context.

    ``prompt_tokens`` is the longest prompt the engine will see; the window holds it, the answer and the one
    token teacher-forcing adds, so a corpus of 16k-token documents is not refused after the model has loaded.
    """
    return window(model, max_tokens + max(prompt_tokens + 1, 8192))


def prompt_token_counts(
    model: str,
    items: list[Prompt],
    *,
    revision: str | None = None,
    max_text_tokens: int,
    chat_template_kwargs: dict | None = None,
    **options: Any,
) -> list[tuple[str, int]]:
    """Each prompt's token count and id, tokenized as :func:`build` does, before an engine loads.

    The one rule the build reads a prompt's length by: `longest_prompt` takes its maximum, and the rows past
    the model's context (:data:`OVER_CONTEXT_SHARE`) are the ones this count says so.

    `options` are ``from_pretrained`` keywords (``trust_remote_code``), the ones the run's ``serve_args`` name
    and :func:`preload_refusal` already got: a tokenizer that needs them would refuse the build here otherwise.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, **({"revision": revision} if revision else {}), **options)
    counts = []
    for p in items:
        if p.text is not None:
            n = min(len(tok(p.text, add_special_tokens=True)["input_ids"]), max_text_tokens)
        else:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p.prompt}],
                add_generation_prompt=True,
                tokenize=True,
                **(chat_template_kwargs or {}),
            )
            n = len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        counts.append((p.id, n))
    return counts


def longest_prompt(
    model: str,
    items: list[Prompt],
    *,
    revision: str | None = None,
    max_text_tokens: int,
    chat_template_kwargs: dict | None = None,
    **options: Any,
) -> tuple[int, str | None]:
    """The token count and id of the longest prompt, from :func:`prompt_token_counts`."""
    longest, which = 0, None
    for pid, n in prompt_token_counts(
        model,
        items,
        revision=revision,
        max_text_tokens=max_text_tokens,
        chat_template_kwargs=chat_template_kwargs,
        **options,
    ):
        if n > longest:
            longest, which = n, pid
    return longest, which


def over_context_line(build: dict[str, Any]) -> str | None:
    """One line naming the rows a build skipped as past the model's context, or None when it skipped none."""
    lost = build.get("dropped_over_context") or []
    if not lost:
        return None
    rows = int(build.get("prompt_rows") or 0)
    name = Path(str(build.get("prompts") or "prompts")).name
    return (
        f"{name}: {rows - len(lost)} of {rows} rows kept, {len(lost)} over the model's "
        f"{build.get('context')}-token context"
    )
