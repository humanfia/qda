"""Self-generated calibration: the original model's own answers to a prompt file, cached per model.

    python -m qda_quant.calib.selfgen --model M --prompts prompts.jsonl [--n 1] [--max-tokens 2048] [--seed 17]

The prompt file is JSONL with one ``{"id": ..., "prompt": ...}`` per line. vLLM answers each prompt ``n`` times,
seeding draw ``d`` with ``seed + d``. Sampling is a config setting: the explicit ``sampling`` mapping wins, field
by field, over the model's ``generation_config.json`` or, when that file names no sampling at all, over the
caller's ``fallback`` (the settings the released cards record, when a reference build passes them); with none of
the three, generation is refused (never a silent vLLM default). Supported fields are temperature, top_p, top_k,
min_p, presence_penalty, repetition_penalty and the stopping ids eos_token_id / bos_token_id / pad_token_id.
``do_sample=false`` selects greedy decoding (temperature 0); the
file's other sampling settings still apply. Other ``generation_config`` fields are ignored and excluded from the
cache key.

vLLM runs only in a child process (``python -m qda_quant.calib.selfgen``): it changes process-wide torch state, and
a build in the same process afterwards would write other weights (:func:`generate_isolated`).

The default ``render="text"`` writes ``{"text": <rendered prompt> + <the generation, verbatim>}``.
``render="messages"`` writes ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``, for loaders
that apply the chat template themselves, as the engine's calibration loader does.

Cache. ``$QDA_CACHE/calib/<model fingerprint>/<key>.jsonl``. The model fingerprint covers the full weight bytes,
config and tokenizer files (:func:`qda_quant.intake.fingerprint.model_fingerprint`, memoized). The key hashes the
settings:
- the prompt file's bytes
- the sampling (``generation_config.json``, its fields overridden by the explicit setting) and the stopping
  settings of ``generation_config.json`` (eos/bos/pad token ids)
- ``n``, ``max_tokens`` and ``seed``
- the chat-template arguments and the rendering
- ``gpu_memory_utilization`` (vLLM's share of the card; default
  `qda_quant.output.vllm_env.DEFAULT_GPU_MEMORY_UTILIZATION`, 0.85)
- on a host where FlashInfer cannot JIT-build, the engine's no-JIT settings (:func:`engine_settings`)

All of it is read from files, so the key is known before any engine starts. A hit is served only if its sidecar
``<key>.jsonl.meta.json`` records the same fingerprint and the same settings; otherwise the file is generated
again. A hit never loads vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from ..output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION, NO_JIT_ENV, flashinfer_cannot_build, no_jit_args
from ..step import Refused
from . import cache_root, meta_path, model_fingerprint, read_rows, sha256_file, write_jsonl

#: generation_config.json fields that are sampling settings (vLLM SamplingParams names).
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")

#: A generator: (model, prompts, n, max_tokens, seed, sampling, chat_template_kwargs, gpu_memory_utilization).
Generator = Callable[[str, list[str], int, int, int, dict, dict, float], list[list[str]]]

#: The engine's environment and arguments, besides the seed and the card share, on a host where FlashInfer cannot
#: JIT-build: qda-kl's no-JIT settings.
NO_JIT_ENGINE = {"env": NO_JIT_ENV, "args": no_jit_args({})}


def engine_settings() -> dict[str, Any]:
    """`NO_JIT_ENGINE` where FlashInfer cannot JIT-build (`qda_quant.output.vllm_env.flashinfer_cannot_build`),
    else {}: vLLM's defaults, with the cache key unchanged.

    Where they apply, the no-JIT settings are in the cache key: they choose other kernels (on H100 the Triton GDN
    prefill replaces FlashInfer's for a Qwen hybrid model), so a file the other engine wrote is not served. For
    Qwen3-0.6B on an RTX PRO 6000 (8 prompts x 2, seed 17) both engines wrote the same 16 answers.
    """
    return NO_JIT_ENGINE if flashinfer_cannot_build() else {}


def sampling_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """The sampling settings a ``generation_config.json`` mapping names (empty when it names none).

    ``do_sample: false`` selects greedy decoding: the temperature becomes 0.0, and the file's other settings
    (``repetition_penalty``, ...) still apply.
    """
    got = {k: cfg[k] for k in SAMPLING_FIELDS if cfg.get(k) is not None}
    if cfg.get("do_sample") is False:
        got["temperature"] = 0.0
    return got


def sampling_of(model: str | Path) -> dict[str, Any]:
    """The sampling settings the model's ``generation_config.json`` names (empty when it has none).

    A local directory is read from disk. A hub id (``org/name[@rev]``) is read from the hub, because a user who
    names one has no local file to inspect, or from the local Hugging Face cache when the hub cannot be reached
    (``HF_HUB_OFFLINE=1``). Both paths read the file's raw fields: parsing it into a ``GenerationConfig`` would
    add a ``None`` for every field the file does not name.

    Raises:
        Refused: the hub cannot be reached and the cache neither holds the file nor records that the repo ships
            none (``code=ENVIRONMENT``): an empty answer there would read as "the model names no sampling".
            When the cache holds no snapshot for the revision at all, the refusal says so
            (`qda_quant.intake.model_ref.offline_refusal`).
    """
    if Path(model).is_dir():
        f = Path(model) / "generation_config.json"
        return sampling_fields(json.loads(f.read_text()) if f.is_file() else {})
    from huggingface_hub import hf_hub_download, try_to_load_from_cache  # lazy: only a hub id needs it
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    from ..exit import ENVIRONMENT
    from ..intake.model_ref import offline_refusal, parse

    ref = parse(model)  # ``org/name@rev`` reads the file at that revision
    name = "generation_config.json"
    try:
        f = hf_hub_download(ref.name, name, **ref.hub())
    except LocalEntryNotFoundError:  # offline or unreachable, and not cached (a subclass of the next one)
        if (missing := offline_refusal(model)) is not None:  # the revision itself is not cached
            raise missing from None
        cached = try_to_load_from_cache(ref.name, name, **ref.hub())
        if isinstance(cached, str):
            return sampling_fields(json.loads(Path(cached).read_text()))
        if cached is not None:  # the cache records that the repo ships none
            return {}
        raise Refused(
            f"{model}: {name} is not in the local Hugging Face cache and the hub cannot be reached "
            "(HF_HUB_OFFLINE is set or there is no network), so its sampling settings are unknown",
            f"download it once with network access (hf download {ref.name} {name}"
            f"{f' --revision {ref.revision}' if ref.revision else ''}), or give the sampling explicitly",
            code=ENVIRONMENT,
        ) from None
    except EntryNotFoundError:  # the repo ships no generation_config.json
        return {}
    return sampling_fields(json.loads(Path(f).read_text()))


def _merged(
    model: str | Path, sampling: dict[str, Any] | None, fallback: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, str]]:
    """The merged sampling settings and, per field, the layer they came from.

    The model's own file is read as one mapping, not field by field: a file that names any sampling setting is
    the model's own word and the fallback stays out of it entirely; the fallback answers only for a model whose
    file names none. The explicit setting wins field by field over whichever of the two is in play.
    """
    own = sampling_of(model)
    base, layer = (own, "generation_config") if own else (fallback or {}, "fallback")
    got = dict(base)
    layers = {field: layer for field in base}
    for field, value in (sampling or {}).items():
        got[field] = value
        layers[field] = "config"
    return got, layers


def sampling_layers(
    model: str | Path, sampling: dict[str, Any] | None, fallback: dict[str, Any] | None = None
) -> dict[str, str]:
    """Which layer each field of :func:`resolve_sampling`'s result came from.

    ``fallback`` is the caller's own fallback (the released cards' settings, for a reference build);
    ``generation_config`` the model's file; ``config`` the explicit setting. A caller that must say where the
    sampling came from reads it here instead of merging the layers again.
    """
    return _merged(model, sampling, fallback)[1]


def resolve_sampling(
    model: str | Path,
    sampling: dict[str, Any] | None,
    key: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The sampling settings to generate with: the explicit setting wins, field by field.

    The precedence is ``sampling`` over the model's ``generation_config.json`` over ``fallback``. The file is
    read as one mapping: when it names any sampling setting it is the model's own and the fallback is not
    consulted; the fallback answers for a model whose file names none (a reference build passes the settings
    the model's released cards record). It is not written down here, because which settings those are is not
    qda-quant's to know.

    Args:
        model: the model's directory or hub id.
        sampling: the explicit setting; its fields win over the model's ``generation_config.json``.
        key: the config key to name in the refusal.
        fallback: the settings to use when ``sampling`` and the model's file together name none.

    Raises:
        Refused: a setting is not one of :data:`SAMPLING_FIELDS` or has no value, or the setting, the model and
            the fallback together name none; the message names ``key`` and an example of what to pass.
    """
    got, _ = _merged(model, sampling, fallback)
    return _validated(
        got, key, "the model's generation_config.json names none and the config sets none"
    )


def merge_sampling(
    base: dict[str, Any] | None, sampling: dict[str, Any] | None, key: str
) -> dict[str, Any]:
    """``sampling`` over ``base``, field by field, validated as :func:`resolve_sampling` validates its result.

    For a caller whose settings come from a record of its own and not from the model's file (a reference build
    of a model whose released cards name it): the model's ``generation_config.json`` is not read at all, so the
    settings cost no download and a field ``base`` omits is omitted, never topped up from the file.

    Args:
        base: the settings to start from (the released cards' row).
        sampling: the explicit setting; its fields win field by field.
        key: the config key to name in the refusal.

    Raises:
        Refused: a setting is not one of :data:`SAMPLING_FIELDS` or has no value, or neither `base` nor
            `sampling` names one.
    """
    return _validated({**(base or {}), **(sampling or {})}, key, "neither the settings nor the base name one")


def _validated(got: dict[str, Any], key: str, empty_message: str) -> dict[str, Any]:
    """`got` as a sampling mapping: known fields, every one with a value, not empty."""
    unknown = sorted(set(got) - set(SAMPLING_FIELDS))
    if unknown:
        raise Refused(f"{key}: unknown setting {unknown[0]!r}; known: {', '.join(SAMPLING_FIELDS)}")
    blank = sorted(k for k, v in got.items() if v is None)
    if blank:
        raise Refused(f"{key}: {blank[0]!r} has no value; remove it or give a number")
    if not got:
        example = '{"temperature": 1.0, "top_p": 0.95}'
        fix = f"pass {key} '{example}'" if key.startswith("--") else f"set {key}, e.g. '{example}'"
        raise Refused(f"{key}: no sampling settings: {empty_message}\nfix: {fix}")
    return got


def read_prompts(path: str | Path) -> list[str]:
    """The ``prompt`` of every line of a prompt file, in order.

    Raises:
        ValueError: a line has no ``prompt``.
    """
    out = []
    for i, r in enumerate(read_rows(path)):
        if "prompt" not in r:
            raise ValueError(f"{path}:{i + 1}: no 'prompt'")
        out.append(str(r["prompt"]))
    return out


#: Cache schema version for rendered rows; change when the same answers would produce different rows.
RENDER_VERSION = 2

#: generation_config.json fields that decide where an answer stops.
STOPPING_FIELDS = ("eos_token_id", "bos_token_id", "pad_token_id")


def stopping_of(model: str | Path) -> dict[str, Any]:
    """The stopping settings the model's ``generation_config.json`` names (empty when it has none)."""
    f = Path(model) / "generation_config.json"
    cfg = json.loads(f.read_text()) if f.is_file() else {}
    return {k: cfg[k] for k in STOPPING_FIELDS if k in cfg}


def settings_of(
    model: str | Path,
    prompts: str | Path,
    n: int,
    max_tokens: int,
    seed: int,
    chat_template_kwargs: dict,
    render: str,
    sampling: dict[str, Any] | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    """Everything besides the model fingerprint that decides a self-generation's rows.

    Raises:
        Refused: the prompt file cannot be read (missing, a directory, or not permitted).
        ValueError: neither ``sampling`` nor the model's ``generation_config.json`` names a setting.
    """
    try:
        prompts_sha = sha256_file(prompts)
    except OSError as e:
        raise Refused(f"calib selfgen: cannot read the prompt file {prompts}: {e}") from e
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("calib.selfgen.gpu_memory_utilization: must be greater than 0 and at most 1")
    return {
        "prompts_sha256": prompts_sha,
        "sampling": resolve_sampling(model, sampling, "calib.selfgen.sampling"),
        "stopping": stopping_of(model),
        "n": n,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": chat_template_kwargs,
        "render": render,
        "render_version": RENDER_VERSION,
        "gpu_memory_utilization": gpu_memory_utilization,
        **({"engine": engine} if (engine := engine_settings()) else {}),
    }


def cache_key(settings: dict[str, Any]) -> str:
    """The cache file name of one self-generation (the model fingerprint names its directory)."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]


def _hit(out: Path, fp: str, settings: dict) -> bool:
    """A cached file is served only when its sidecar records exactly this model and these settings."""
    if not (out.is_file() and meta_path(out).is_file()):
        return False
    try:
        meta = json.loads(meta_path(out).read_text())
    except ValueError:
        return False
    return meta.get("model", {}).get("fingerprint") == fp and meta.get("settings") == settings


def sampling_params(
    sampling: dict[str, Any], max_tokens: int, seed: int, *, skip_special_tokens: bool = True
) -> Any:
    """The vLLM ``SamplingParams`` of one resolved sampling mapping: the one both generation paths use.

    ``skip_special_tokens`` keeps vLLM's default (True) for every existing caller. An eval suite whose
    scoring cuts at a configured closing-thinking tag passes False: that tag is a special token, vLLM
    strips it at decode time, and a stripped boundary cannot be recovered from the stored text.
    """
    from vllm import SamplingParams  # lazy: qda_quant has no import-time vLLM dependency

    return SamplingParams(
        **sampling, max_tokens=max_tokens, seed=seed, skip_special_tokens=skip_special_tokens
    )


def vllm_generate(
    model: str,
    prompts: list[str],
    n: int,
    max_tokens: int,
    seed: int,
    sampling: dict,
    chat_template_kwargs: dict,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> list[list[str]]:
    """The default generator: vLLM, one engine, every prompt ``n`` times (draw ``d`` at ``seed + d``).

    Runs only in the child process of :func:`generate_isolated` (this module's ``main``). Never call it in a
    process that later builds: vLLM changes process-wide torch state (see :func:`generate_isolated`).
    """
    from vllm import LLM  # lazy: qda_quant has no import-time vLLM dependency

    from ..output.vllm_env import engine_start

    # the venv's ninja on PATH, the caches on a node-local disk, and a known failure's fix noted on its error
    engine = engine_settings()
    args = {**engine.get("args", {}), "seed": seed, "gpu_memory_utilization": gpu_memory_utilization}
    llm, _ = engine_start(engine.get("env", {}), lambda: LLM(model=model, **args))
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    jobs = [(i, d) for i in range(len(prompts)) for d in range(n)]
    params = [sampling_params(sampling, max_tokens, seed + d) for _, d in jobs]
    said = llm.chat(
        [msgs[i] for i, _ in jobs], params, use_tqdm=False, chat_template_kwargs=chat_template_kwargs or None
    )
    out: list[list[str]] = [[] for _ in prompts]
    for (i, _), o in zip(jobs, said, strict=True):
        out[i].append(o.outputs[0].text)
    return out


def cache_file(
    model: str | Path,
    prompts: str | Path,
    n: int = 1,
    max_tokens: int = 2048,
    seed: int = 17,
    *,
    chat_template_kwargs: dict | None = None,
    render: Literal["messages", "text"] = "text",
    sampling: dict[str, Any] | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> tuple[Path, bool]:
    """Where :func:`selfgen` caches this generation, and whether a valid file is there. Never generates.

    The key is the model fingerprint plus the generation settings (see the module docstring); everything is
    read from files, so no engine starts.

    Raises:
        Refused: the prompt file cannot be read.
        ValueError: neither ``sampling`` nor the model's ``generation_config.json`` names a setting.
    """
    model = str(model)
    fp = model_fingerprint(model)
    settings = settings_of(
        model,
        prompts,
        n,
        max_tokens,
        seed,
        dict(chat_template_kwargs or {}),
        render,
        sampling,
        gpu_memory_utilization,
    )
    out = cache_root() / fp / f"{cache_key(settings)}.jsonl"
    return out, _hit(out, fp, settings)


def generate_isolated(
    model: str | Path,
    prompts: str | Path,
    n: int = 1,
    max_tokens: int = 2048,
    seed: int = 17,
    *,
    chat_template_kwargs: dict | None = None,
    render: Literal["messages", "text"] = "text",
    sampling: dict[str, Any] | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> Path:
    """:func:`selfgen` in a child process (``python -m qda_quant.calib.selfgen``); returns the cache file.

    Both the build's generate stage and :func:`selfgen` use this isolation because ``LLM()`` disables cuDNN's
    attention backend, which can change subsequent Hugging Face forwards and build output. The child shares
    the cache (``QDA_CACHE``) while keeping its torch state separate.

    Raises:
        RuntimeError: the child failed (its stderr went to this process's stderr).
    """
    import subprocess
    import sys

    from ..output.vllm_env import child_env

    argv = [sys.executable, "-m", "qda_quant.calib.selfgen", "--model", str(model), "--prompts", str(prompts)]
    argv += ["--n", str(n), "--max-tokens", str(max_tokens), "--seed", str(seed), "--render", render]
    argv += ["--chat-template-kwargs", json.dumps(chat_template_kwargs or {})]
    argv += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if sampling is not None:
        argv += ["--sampling", json.dumps(sampling)]
    done = subprocess.run(argv, stdout=subprocess.PIPE, text=True, check=False, env=child_env())
    if done.returncode != 0:
        raise RuntimeError(f"self-generation failed (exit {done.returncode}): {' '.join(argv)}")
    return Path(done.stdout.strip().splitlines()[-1])


def selfgen(
    model: str | Path,
    prompts: str | Path,
    n: int = 1,
    max_tokens: int = 2048,
    seed: int = 17,
    *,
    chat_template_kwargs: dict | None = None,
    render: Literal["messages", "text"] = "text",
    generate: Generator | None = None,
    tokenizer: Any = None,
    sampling: dict[str, Any] | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> Path:
    """The model's own answers to ``prompts`` as a calibration JSONL (cached); returns its path.

    Without ``generate``, a cache miss runs vLLM in a child process (:func:`generate_isolated`), so this process
    can build afterwards. A ``generate`` callable runs in this process.

    Args:
        model: the original model's directory.
        prompts: JSONL of ``{"id", "prompt"}`` lines.
        n: answers per prompt.
        max_tokens: the cap of one answer.
        seed: the seed of draw 0 (draw ``d`` uses ``seed + d``).
        chat_template_kwargs: extra chat-template arguments (e.g. ``{"enable_thinking": False}``).
        render: ``text`` rows with the chat template applied (default), or ``messages`` rows.
        generate: a generator to run in this process; None runs :func:`vllm_generate` in a child process.
        tokenizer: the model's tokenizer for ``render="text"`` (loaded from ``model`` when None).
        sampling: the sampling settings; its fields win over the model's ``generation_config.json``.
        gpu_memory_utilization: vLLM's share of the card (default 0.85); use 0.2 on a shared card.

    Raises:
        ValueError: neither ``sampling`` nor the model's ``generation_config.json`` names a setting, or the
            generator returned a different number of answers than asked.
        RuntimeError: the child process failed.
    """
    model, ctk = str(model), dict(chat_template_kwargs or {})
    out, hit = cache_file(
        model,
        prompts,
        n,
        max_tokens,
        seed,
        chat_template_kwargs=ctk,
        render=render,
        sampling=sampling,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if hit:
        return out
    if generate is None:
        return generate_isolated(
            model,
            prompts,
            n,
            max_tokens,
            seed,
            chat_template_kwargs=ctk,
            render=render,
            sampling=sampling,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return _write(
        model,
        prompts,
        n,
        max_tokens,
        seed,
        ctk,
        render,
        generate,
        tokenizer,
        sampling,
        gpu_memory_utilization,
    )


def _write(
    model: str,
    prompts: str | Path,
    n: int,
    max_tokens: int,
    seed: int,
    ctk: dict,
    render: Literal["messages", "text"],
    generate: Generator,
    tokenizer: Any = None,
    sampling: dict[str, Any] | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> Path:
    """Generate with ``generate`` in this process and write the cache file (:func:`selfgen` after a miss)."""
    out, _ = cache_file(
        model,
        prompts,
        n,
        max_tokens,
        seed,
        chat_template_kwargs=ctk,
        render=render,
        sampling=sampling,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    fp = model_fingerprint(model)
    settings = settings_of(model, prompts, n, max_tokens, seed, ctk, render, sampling, gpu_memory_utilization)
    texts = read_prompts(prompts)
    answers = generate(model, texts, n, max_tokens, seed, settings["sampling"], ctk, gpu_memory_utilization)
    if len(answers) != len(texts) or any(len(a) != n for a in answers):
        raise ValueError(
            f"generator returned {[len(a) for a in answers]} answers, expected {n} for each of "
            f"{len(texts)} prompts"
        )
    rows = [
        {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}]}
        for p, per in zip(texts, answers, strict=True)
        for a in per
    ]
    if render == "text":
        rows = _as_text(model, rows, ctk, tokenizer)
    meta = {
        "kind": "selfgen",
        "model": {"path": model, "fingerprint": fp},
        "prompts": str(prompts),
        "settings": settings,
        "rows": len(rows),
        "tokenizer": sha256_file(Path(model) / "tokenizer.json")
        if (Path(model) / "tokenizer.json").is_file()
        else None,
    }
    return write_jsonl(out, rows, meta)


def _as_text(model: str, rows: list[dict], ctk: dict, tok: Any = None) -> list[dict]:
    """Each row as the text the model actually saw and wrote: the rendered prompt + the generation, verbatim.

    Not the chat template over both turns: templates rewrite assistant turns (Qwen3 adds an empty
    ``<think></think>`` block to an answer without ``</think>``, e.g. one cut at ``max_tokens``), which would put
    tokens in the calibration the model never produced.
    """
    if tok is None:
        from ..intake.tokenizer import load_tokenizer

        tok = load_tokenizer(model)
    out = []
    for r in rows:
        user, answer = r["messages"]
        prompt = tok.apply_chat_template([user], tokenize=False, add_generation_prompt=True, **ctk)
        out.append({"text": prompt + answer["content"]})
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m qda_quant.calib.selfgen``: prints the calibration file's path. The one caller of vLLM."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        metavar="FRACTION",
        help=f"vLLM's share of the card (default {DEFAULT_GPU_MEMORY_UTILIZATION}); use 0.2 when sharing",
    )
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    ap.add_argument("--render", choices=("text", "messages"), default="text")
    ap.add_argument("--sampling", type=json.loads, default=None, help='JSON, e.g. {"temperature": 1.0}')
    a = ap.parse_args(argv)
    model, ctk = str(a.model), dict(a.chat_template_kwargs or {})
    out, hit = cache_file(
        model,
        a.prompts,
        a.n,
        a.max_tokens,
        a.seed,
        chat_template_kwargs=ctk,
        render=a.render,
        sampling=a.sampling,
        gpu_memory_utilization=a.gpu_memory_utilization,
    )
    if not hit:
        out = _write(
            model,
            a.prompts,
            a.n,
            a.max_tokens,
            a.seed,
            ctk,
            a.render,
            vllm_generate,
            sampling=a.sampling,
            gpu_memory_utilization=a.gpu_memory_utilization,
        )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
