"""The ``vllm serve`` command a checkpoint's serve section asks for, as the line a user is told to run.

``build.json["serve"]`` is the one place a build records which kernel serves its checkpoint (the plan comes
from ``qda_quant.target``). This module turns that section into the engine's flags and process environment, so
every place that tells a user how to serve -- ``qda build``, ``qda-agent prepare``, the search's end report and
``qda-agent evaluate`` -- prints one command, and that command binds the kernel the checkpoint's KL was measured
on rather than vLLM's own pick.

qda-kl applies the same section to its own server: it reads the section through :func:`serve_section` and the
environment through :func:`command_environment`, plus its deterministic-reduction pin
(:func:`qda_kl.serve.deterministic_args`).
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..step import Refused
from ..target import ban_kernels, stored_marlin_pin, unpinned
from .serve_policy import check as check_serve

#: The model name the scoring engine's own server answers to (``qda-kl`` starts it, it is not a user's line).
SERVED_NAME = "qda"

#: ``vllm.LLM`` keyword -> the ``vllm serve`` flag that takes the same setting. A key outside this mapping is
#: refused: a serve command silently missing one of the checkpoint's settings would measure another engine.
SERVE_FLAGS = {
    "compilation_config": "--compilation-config",
    "dtype": "--dtype",
    "enable_expert_parallel": "--enable-expert-parallel",
    "enable_prefix_caching": "--enable-prefix-caching",
    "enforce_eager": "--enforce-eager",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "kv_cache_dtype": "--kv-cache-dtype",
    "linear_backend": "--linear-backend",
    "max_model_len": "--max-model-len",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_num_seqs": "--max-num-seqs",
    "moe_backend": "--moe-backend",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "quantization": "--quantization",
    "speculative_config": "--speculative-config",
    "tensor_parallel_size": "--tensor-parallel-size",
    "tokenizer": "--tokenizer",
    "tokenizer_mode": "--tokenizer-mode",
    "trust_remote_code": "--trust-remote-code",
}
#: Flags whose ``False`` is the flag's ``--no-`` form rather than an argument.
BOOLEAN_FLAGS = {
    "--enable-expert-parallel",
    "--enable-prefix-caching",
    "--enforce-eager",
    "--trust-remote-code",
}


def _render_flag(value: Any) -> str:
    """One flag value: a mapping or list as JSON (vLLM's own parsers accept it), anything else as text."""
    if isinstance(value, dict | list):
        return json.dumps(value)
    if isinstance(value, bool):
        raise Refused(f"boolean setting {value!r} reached a value flag; extend BOOLEAN_FLAGS")
    return str(value)


def _record(checkpoint: str | Path) -> dict[str, Any]:
    """The checkpoint's ``build.json`` as written, or ``{}`` when it has none."""
    path = Path(checkpoint) / "build.json"
    return json.loads(path.read_text()) if path.is_file() else {}


#: A path inside the HF cache: ``.../models--<org>--<name>/snapshots/<commit>`` names a hub repo.
_HUB_CACHE = re.compile(r"models--([^/]+?)--([^/]+?)/snapshots/")


def _hub_name(named: str) -> str | None:
    """The hub repo name `named` denotes, or None when it names no repo (a plain local path, a short name).

    A build records a local model as an absolute directory; a hub id stays ``org/name``, with or without an
    ``@revision``, and the local directory a hub id resolved to is the cache's snapshot of that repo.
    """
    text = str(named).split("@", 1)[0].rstrip("/")
    if m := _HUB_CACHE.search(text):
        return m.group(2)
    parts = text.split("/")
    if len(parts) == 2 and all(parts) and not text.startswith(("/", ".", "~")):
        return parts[1]
    return None


def _source_name(source: str) -> str:
    """The name of the original model `source` names: its directory's, or the hub repo's of its snapshot.

    A build records the local directory it read, which for a hub id is the HF cache's snapshot,
    ``.../models--<org>--<name>/snapshots/<commit>``; the repo name is the model's, the commit is not.
    """
    path = Path(source)
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--"):
        return repo.removeprefix("models--").rpartition("--")[2] or repo
    return path.name


def _config_name(checkpoint: Path) -> str | None:
    """The hub repo the checkpoint's own ``config.json`` names as its original, or None."""
    path = checkpoint / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for key in ("base_model", "_name_or_path", "name_or_path"):
        named = config.get(key) if isinstance(config, dict) else None
        if isinstance(named, str) and (repo := _hub_name(named)):
            return repo
    return None


def _format_name(recorded: str) -> str:
    """The short spelling of the format a build recorded (``w4a16-asym-g128`` -> ``w4a16``) in upper case.

    ``Format.spelling`` drops a default symmetry and group size, which is what a name wants; a format of its
    own (a codebook) spells itself differently, so it is used only where it parses back to the same format.
    """
    from ..format import Format

    try:
        fmt = Format.parse(recorded)
    except Refused:  # a format this library does not know keeps its recorded spelling
        return recorded.upper()
    short = fmt.spelling
    return (short if Format.parse(short) == fmt else fmt.name).upper()


def served_name(checkpoint: str | Path) -> str:
    """The model name a request to the checkpoint's server must carry: the original's name and its format.

    ``Qwen/Qwen3-0.6B`` built as ``w4a16`` is served as ``Qwen3-0.6B-W4A16``. Both parts come from the
    checkpoint's own record, so ``qda build``, a search's final report and ``qda inspect --serve`` name one
    checkpoint alike. The original's name is the hub repo its record carries (``source_repo``, else the repo
    ``source`` names), else the repo its ``config.json`` names as the original, else the source directory's
    name: a build reads a local model directory, and a prepared run downloads the model as ``.../model``,
    whose name alone says nothing.
    """
    record = _record(checkpoint)
    source = record.get("source")
    name = None
    for named in (record.get("source_repo"), source):
        if isinstance(named, str) and named and (repo := _hub_name(named)):
            name = repo
            break
    if name is None:
        name = _config_name(Path(checkpoint))
    if name is None:
        name = _source_name(str(source)) if source else Path(checkpoint).name
    fmt = record.get("format")
    return f"{name}-{_format_name(str(fmt))}" if fmt else name


def serve_argv(
    checkpoint: str | Path,
    args: Mapping[str, Any],
    *,
    name: str | None = None,
    host: str | None = None,
    port: int | None = None,
    seed: int | None = None,
    disable_log_requests: bool = False,
) -> list[str]:
    """The ``vllm serve`` argv of a serve section's ``args``; a setting without a flag is refused.

    Args:
        checkpoint: the local checkpoint directory the server loads.
        args: vLLM engine arguments (``build.json["serve"]["args"]``, plus a caller's own).
        name: the model name requests must carry; None: :func:`served_name` of the checkpoint.
        host: interface to bind; None leaves vLLM's own.
        port: port to bind; None leaves vLLM's own.
        seed: the engine's global seed (the per-request seed comes from each request); None leaves vLLM's.
        disable_log_requests: start with vLLM's per-request logging off (a long measurement, not a user's line).

    Raises:
        Refused: a setting has no ``vllm serve`` flag, so the command would start another engine than the
            settings describe.
    """
    argv = [
        "vllm",
        "serve",
        str(checkpoint),
        "--served-model-name",
        served_name(checkpoint) if name is None else name,
    ]
    # only what the checkpoint's settings need: a user's line leaves host, port and seed to vLLM's defaults
    for flag, value in (("--host", host), ("--port", port), ("--seed", seed)):
        if value is not None:
            argv += [flag, str(value)]
    if disable_log_requests:
        argv.append("--disable-log-requests")
    for key, value in args.items():
        flag = SERVE_FLAGS.get(key)
        if flag is None:
            raise Refused(
                f"serve_args names {key!r}, which has no vllm serve flag here; extend SERVE_FLAGS "
                "(a setting the command drops would measure another engine)"
            )
        if flag in BOOLEAN_FLAGS:
            argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            argv += [flag, _render_flag(value)]
    return argv


def command_environment(
    env: Mapping[str, str],
    plugins: Iterable[str] = (),
    nccl_nvls: bool | None = None,
) -> dict[str, str]:
    """The environment a serve command shows: the checkpoint's pins, its plugins, and ``NCCL_NVLS_ENABLE``.

    The checkpoint's ``serve.env`` carries what argv cannot, such as ``VLLM_DISABLED_KERNELS`` on a vLLM
    without ``--linear-backend``; a server started without it binds another kernel than the score measured.
    Every card line and every engine qda-kl starts gets the banned kernels off (`qda_quant.target.ban_kernels`).
    """
    nvls = {} if nccl_nvls is None else {"NCCL_NVLS_ENABLE": "1" if nccl_nvls else "0"}
    named = ",".join(plugins)
    return {**nvls, **ban_kernels(env), **({"VLLM_PLUGINS": named} if named else {})}


#: The shell text that sets ``LD_LIBRARY_PATH`` for one command: the venv's ``nvidia/*/lib`` directories first, then
#: the user's own value. The directories are found when the line runs, by the ``python`` next to the ``vllm`` the
#: line starts (the venv's, or the conda env's, whichever ``vllm`` is first on ``PATH``), so the line holds no path
#: of the machine that printed it and no directory of the printing shell's own ``LD_LIBRARY_PATH``. With no
#: ``vllm`` on ``PATH`` or no ``python`` beside it, the shell prints one line and the server is not started.
#: Bourne syntax, so bash and zsh run it, also under ``set -u``. Standard library only, so a serving venv without qda runs it; it finds
#: what `vllm_env.nvidia_lib_path` finds (a test runs both).
NVIDIA_LIBS = (
    'lib="$(v="$(command -v vllm)" && [ -x "${v%/*}/python" ] || '
    '{ echo "no python next to vllm: activate the venv that has vLLM" >&2; exit 1; }; '
    '"${v%/*}/python" -c \'import pathlib,sysconfig;p=sysconfig.get_paths();'
    'print(*sorted({str(d) for k in("purelib","platlib") for d in pathlib.Path(p[k],"nvidia").glob("*/lib")}),sep=":")\')" '
    '&& LD_LIBRARY_PATH="$lib${lib:+${LD_LIBRARY_PATH:+:}}${LD_LIBRARY_PATH-}"'
)


def binds_nvrtc_kernel(serve: Mapping[str, Any]) -> bool:
    """Whether the serve section plans a Humming kernel, or plans none (the engine picks, and may pick Humming).

    Humming compiles through NVRTC when the server starts. NVRTC opens ``libnvrtc-builtins`` by name in a child
    process, and the ``nvidia-*`` wheels keep it in a directory the loader does not search, so that server needs
    the venv's NVIDIA libraries on ``LD_LIBRARY_PATH`` (`NVIDIA_LIBS`); Marlin and the other kernels do not. An
    AWQ checkpoint records no kernels, so its line keeps the prefix.
    """
    kernels = serve.get("kernels") or {}
    return not kernels or any(name.startswith("Humming") for names in kernels.values() for name in names)


def command_line(argv: Iterable[str], env: Mapping[str, str], *, loader: str | None = None) -> str:
    """One pasteable line: the environment prefix, then the argv, each shell-quoted.

    The environment goes in front of the command rather than in a separate export: a user who copies one line
    gets the kernel pin with it, and a server started without it binds another kernel than the score measured.
    ``loader`` is shell text, put first and not quoted (`NVIDIA_LIBS`).
    """
    prefix = [loader] if loader else []
    prefix += [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    return " ".join([*prefix, *(shlex.quote(str(arg)) for arg in argv)])


def serve_section(checkpoint: str | Path) -> dict[str, Any] | None:
    """The checkpoint's checked ``build.json["serve"]`` section, or None when it has none.

    The one reader of the section: a checkpoint may come from anyone, so what it holds is checked against what
    ``qda build`` writes before any caller renders or applies it.
    """
    record = Path(checkpoint) / "build.json"
    if not record.is_file():
        return None
    doc = json.loads(record.read_text())
    serve = doc.get("serve")
    if serve is None:
        return None
    check_serve(serve)  # an unplanned section is neither rendered nor applied
    # a published sm90 record pins the Marlin default it was built on; it serves as an sm90 build does now
    return unpinned(serve) if stored_marlin_pin(doc) else serve


def serve_command_line(checkpoint: str | Path) -> str | None:
    """How to serve a checkpoint qda built, as one line, or None when qda planned no vLLM server for it.

    The line is rendered from ``build.json`` only -- ``serve`` is the section qda-kl scores the checkpoint with,
    the same section the caller's own scoring applies; ``source`` and ``format`` name the model
    (:func:`served_name`). A caller's own scoring overrides (a run's ``serve_args``) are not a property of the
    checkpoint and are not shown, and neither are the flags a user's line does not need (``--host``,
    ``--port``, ``--seed``, ``--disable-log-requests``): a checkpoint has no interface, port or seed. A
    checkpoint with no ``serve`` section is served by another runtime (a GGUF names llama.cpp), so there is no
    vLLM command to print.

    Raises:
        ServeNotAllowed: the section holds a setting ``qda build`` never writes.
        Refused: a setting has no ``vllm serve`` flag (:func:`serve_argv`).
    """
    serve = serve_section(checkpoint)
    if serve is None:
        return None
    argv = serve_argv(checkpoint, serve.get("args") or {})
    env = command_environment(serve.get("env") or {}, serve.get("plugins") or [])
    # a published sm90 record kept its Marlin kernels, which its pin no longer asks for: vLLM picks (`serve_section`)
    picked = binds_nvrtc_kernel(serve) or stored_marlin_pin(_record(checkpoint))
    return command_line(argv, env, loader=NVIDIA_LIBS if picked else None)
