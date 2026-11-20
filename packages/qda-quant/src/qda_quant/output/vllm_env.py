"""The environment a vLLM engine starts in: the venv's tools on ``PATH``, a fix noted on a failure QDA knows, and the
memory share and tensor parallelism it starts with when nothing names them.

Its compiled kernels and graphs go in one node-local directory per user (`cache_root`) unless the user names
the directories; `qda doctor` reports whether that one is this user's private directory and can be locked.

Both qda-kl's scorer and qda-quant's self-generated calibration start vLLM in this process, so both use this.
vLLM itself stays an optional dependency: nothing here imports it.
"""

from __future__ import annotations

import contextlib
import fcntl
import getpass
import importlib.metadata as md
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from ..exit import ENVIRONMENT, FIX_LINE
from ..step import Refused
from ..target import ban_kernels

T = TypeVar("T")

#: vLLM's share of each card when no flag or setting names one, a fraction of the card's *total* memory, and the
#: tensor-parallel size likewise: defined once here, beside the engine's start environment, and read by every
#: command and function of qda, qda-kl and qda-agent that starts vLLM.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_TP = 1

#: Where an engine keeps its on-disk caches (compiled graphs, Triton kernels, the FlashInfer workspace), and the
#: directory each gets under `cache_root` when the user names none. Each names a host-local location, never what
#: is measured; unset, vLLM, Triton and FlashInfer would cache under ``$HOME``, which is often a network
#: filesystem where two engines' first compiles cannot both take the JIT lock.
CACHE_DIRS = {
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "VLLM_CACHE_ROOT": "vllm",
}
CACHE_ENV = frozenset(CACHE_DIRS)

#: Environment the scoring, gencheck and ``qda inspect --serve`` engines start with, so they compile no FlashInfer
#: kernel at run time and need no nvcc. Each judge gencheck of 2026-11-13 died in nvcc while FlashInfer JIT-built
#: ``sampling`` (vLLM's top-k/top-p sampler) and, at tp > 1, ``trtllm_mnnvl_comm`` (its all-reduce). vLLM's own
#: sampler and NCCL's all-reduce need no build.
NO_JIT_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ALLREDUCE_USE_FLASHINFER": "0"}

#: vLLM's Gated DeltaNet prefill kernel that needs no build. On H100 vLLM picks FlashInfer's GDN prefill, which
#: it JIT-builds at the first prefill, for every Qwen hybrid model (linear-attention layers); other models ignore it.
NO_JIT_GDN_PREFILL = "triton"


def no_jit_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """``vllm.LLM`` keywords ``args`` with vLLM's all-reduce + RMSNorm fusion off and the Triton GDN prefill.

    The fusion pass (``compilation_config.pass_config.fuse_allreduce_rms``, on by default at tp > 1 on H100)
    builds the same ``trtllm_mnnvl_comm`` module whatever ``VLLM_ALLREDUCE_USE_FLASHINFER`` says. The GDN
    prefill backend (``gdn_prefill_backend``, also read from ``additional_config``) is a third path to a
    FlashInfer build. An ``args`` that sets the fusion or the GDN prefill backend itself keeps its value;
    compilation level 0 compiles nothing, so it has no pass to turn off. Started with `NO_JIT_ENV`.
    """
    args = dict(args)
    extra = args.get("additional_config")
    if "gdn_prefill_backend" not in args and not (isinstance(extra, dict) and "gdn_prefill_backend" in extra):
        args["gdn_prefill_backend"] = NO_JIT_GDN_PREFILL
    raw = args.get("compilation_config")
    if isinstance(raw, dict):
        passes = {"fuse_allreduce_rms": False, **raw.get("pass_config", {})}
        args["compilation_config"] = {**raw, "pass_config": passes}
    elif raw is None:
        args["compilation_config"] = {"pass_config": {"fuse_allreduce_rms": False}}
    return args


@contextlib.contextmanager
def environ(env: Mapping[str, str]) -> Iterator[None]:
    """``env`` set in this process's environment for the block, the old values restored after.

    An engine's processes are spawned while it starts and keep this environment.
    """
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def nvidia_lib_path() -> str:
    """The venv's bundled NVIDIA libraries for the dynamic loader: ``site-packages/nvidia/*/lib``.

    The ``nvidia-*`` wheels ship their shared objects (``libnvrtc.so``, ``libnvrtc-builtins.so``,
    ``libcudart.so``...) under ``nvidia/<package>/lib`` rather than in a directory the loader searches. Humming
    compiles its kernels through NVRTC in a child process, which dies with
    ``failed to open libnvrtc-builtins.so.<major>`` (``NVRTC_ERROR_BUILTIN_OPERATION_FAILURE``) when that
    directory is missing, so a W3A16 engine cannot start. Empty when this interpreter has no nvidia
    package (the wheels are not installed, or CUDA came from a system toolkit).
    """
    roots = {sysconfig.get_paths()[key] for key in ("purelib", "platlib")}
    dirs = sorted(
        str(path)
        for root in map(Path, roots)
        if (root / "nvidia").is_dir()
        for path in (root / "nvidia").glob("*/lib")
        if path.is_dir()
    )
    return os.pathsep.join(dict.fromkeys(dirs))


def loader_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``env`` with `nvidia_lib_path`'s directories first on ``LD_LIBRARY_PATH``, each path once.

    Paths already on the variable (in ``env`` or, when it does not set one, in this process's environment) are
    kept after the venv's directories, so an explicit search path is never dropped.

    Returns:
        A copy of ``env`` with ``LD_LIBRARY_PATH`` set, or unchanged when there is no nvidia directory to add.
    """
    merged = dict(env or {})
    dirs = [p for p in nvidia_lib_path().split(os.pathsep) if p]
    inherited = merged.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    current = [p for p in inherited.split(os.pathsep) if p]
    wanted = [*dirs, *[p for p in current if p not in set(dirs)]]
    if wanted:
        merged["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(wanted))
    return merged


@contextlib.contextmanager
def vllm_environ(env: Mapping[str, str] | None = None) -> Iterator[None]:
    """The environment to start vLLM in.

    First the engine start's own guards (`qda_quant.checks.refuse_serve`: a driver too old for torch's CUDA,
    the list `qda doctor` walks too). :func:`tool_path` goes on ``PATH``, then ``env`` with the banned kernels
    off (`qda_quant.target.ban_kernels`); the venv's NVIDIA library
    directories go on ``LD_LIBRARY_PATH`` (:func:`loader_env`, so Humming's NVRTC child process finds its
    builtins); the host compiler nvcc will use is chosen (:func:`host_compiler_env`). In that environment Triton's
    CUDA driver must be in its cache or compile (`qda_quant.checks.refuse_engine_compiler`: a C compiler and the
    Python headers). Whether FlashInfer can compile is left to the engine: :func:`engine_start` keeps its error
    and adds a ``fix:`` line when it knows one (:func:`add_fix`). The cache directories are made here
    (:func:`make_caches`), so no engine start fails on a directory nobody has created yet.

    Raises:
        Refused: a serve guard refused (`qda_quant.checks.refuse_serve`), :func:`host_compiler_env` found no
            host compiler nvcc can preprocess with, or Triton's driver does not compile.
    """
    from ..checks import refuse_engine_compiler, refuse_serve  # checks imports this module

    refuse_serve()
    full = {"PATH": tool_path(), **cache_env(disabled_kernels(env or {}))}
    make_caches(full)
    full.update(host_compiler_env({**os.environ, **full}))
    full = loader_env(full)
    with environ(full):
        refuse_engine_compiler()  # with the engine's PATH and CC, before it loads a weight
        yield


def cache_root() -> Path:
    """The compile caches of every vLLM engine qda starts, when the user names no directory of their own.

    One directory per user on the node's own disk: the caches survive a run (deleting a run directory does not
    clear them) and never land on the network filesystem a ``$HOME`` or a run directory may be on.
    """
    try:
        user = getpass.getuser()
    except (ImportError, KeyError, OSError):  # no password entry for this uid
        user = str(os.getuid())
    return Path("/tmp") / f"qda_{user}"  # noqa: S108 (this user's directory, checked by `cache_root_problem`)


def ensure_cache_root() -> Path:
    """Make the default root (`cache_root`) this user's 0700 directory, with a directory per cache, and check it.

    The root is made with `os.mkdir` and mode 0700 rather than `mkdir(parents=True)`, which would make it the
    parent of a cache directory and leave it 0755 under the usual umask: every other user on the node could then
    read the compiled kernels and graphs it holds. A root that exists already is not replaced; its type, owner
    and mode are read back after the attempt (`cache_root_problem`), so the creation and the check are one call
    and a root another user made first is found before anything is written under it.

    Returns:
        The root.

    Raises:
        Refused: the root cannot be made, or is not this user's private directory.
    """
    root = cache_root()
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        pass
    except OSError as error:
        raise Refused(f"{root}: {error}", CACHE_FIX, code=ENVIRONMENT) from error
    if (problem := cache_root_problem(root)) is not None:
        raise Refused(problem, CACHE_FIX, code=ENVIRONMENT)
    for leaf in CACHE_DIRS.values():
        with contextlib.suppress(OSError):
            (root / leaf).mkdir(mode=0o700, exist_ok=True)
    return root


def cache_root_problem(base: Path | None = None) -> str | None:
    """Why ``base`` (default `cache_root`) is not a private directory of this user, or None.

    ``/tmp`` is shared with every other user of the node, and an engine loads and runs the build output in its
    cache. Another user can therefore make ``/tmp/qda_<user>`` first and leave Triton, FlashInfer or Inductor
    output there. A root that exists must be a directory (not a symlink) owned by this user and writable by
    neither group nor others. A root that does not exist yet is not a problem: :func:`ensure_cache_root` makes
    it as this user.
    """
    base = cache_root() if base is None else base
    try:
        info = base.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        return f"{base}: {error}"
    if stat.S_ISLNK(info.st_mode):
        return f"{base}: a symbolic link, not a directory of this user's"
    if not stat.S_ISDIR(info.st_mode):
        return f"{base}: not a directory"
    if info.st_uid != os.getuid():
        return f"{base}: belongs to another user (uid {info.st_uid}), not this user (uid {os.getuid()})"
    if info.st_mode & 0o022:
        return f"{base}: writable by group or others (mode 0o{stat.S_IMODE(info.st_mode):03o})"
    return None


def cache_dir(name: str, env: Mapping[str, str] | None = None) -> Path:
    """Where the cache variable ``name`` (a `CACHE_DIRS` key) points: ``env``'s value, else this process's, else
    its directory under `cache_root`. No side effects: `cache_env` is the one that makes or refuses the root.
    """
    return Path((env or {}).get(name) or os.environ.get(name) or cache_root() / CACHE_DIRS[name])


def disabled_kernels(env: Mapping[str, str]) -> dict[str, str]:
    """`env` with ``VLLM_DISABLED_KERNELS`` = this process's exported list, then `env`'s, then the banned kernels.

    A user who exported ``VLLM_DISABLED_KERNELS`` (e.g. ``HummingLinearKernel``) keeps it: `env` naming the
    variable would otherwise replace it. The banned kernels are `qda_quant.target.ban_kernels`'s.
    """
    lists = (os.environ.get("VLLM_DISABLED_KERNELS", ""), env.get("VLLM_DISABLED_KERNELS", ""))
    return ban_kernels({**env, "VLLM_DISABLED_KERNELS": ",".join(x for x in lists if x)})


def cache_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``env`` with every cache variable (`CACHE_DIRS`) resolved: the one ``env`` or this process names, else
    a directory of `cache_root`.

    Every vLLM engine qda starts goes through this: :func:`vllm_environ` in process, and the agent's jobs
    through ``qda_agent.components.Serve.env``. A variable the user set is kept as it is. Any variable the user
    left unset falls back to the default root, and :func:`ensure_cache_root` makes that root this user's private
    directory and refuses it before the engine starts when it cannot: its build output is code the engine would
    run.

    Raises:
        Refused: the default root is used (some variable is unset) and is not this user's private directory.
    """
    resolved = dict(env or {})
    defaulted = [name for name in CACHE_DIRS if not (resolved.get(name) or os.environ.get(name))]
    if defaulted:
        ensure_cache_root()
    for name in CACHE_DIRS:
        resolved[name] = resolved.get(name) or str(cache_dir(name))
    return resolved


def make_caches(env: Mapping[str, str]) -> None:
    """Make the cache directories ``env`` names, private to this user, and leave a failure to the engine.

    The default root, when ``env`` falls back to it for any cache, is made and checked by
    :func:`ensure_cache_root` first: the root another user could have made first is found in the same call that
    creates it, with no window between the check `cache_env` does and the directories made here.

    Each library creates its own cache on first use, but not all of them create a missing parent; a directory
    that cannot be made at all is reported by `qda doctor`'s `engine cache` line and refused by the engine
    itself, with its own words.
    """
    root = cache_root()
    if any(env.get(name) == str(root / leaf) for name, leaf in CACHE_DIRS.items()):
        ensure_cache_root()
    for name in CACHE_DIRS:
        if (named := env.get(name)) is not None:
            with contextlib.suppress(OSError):
                private_mkdir(Path(named))


def private_mkdir(path: Path) -> None:
    """Make ``path`` and each missing parent mode 0700, whatever the umask.

    ``mkdir(mode=0o700, parents=True)`` gives the mode to the last directory only; the parents it makes take the
    umask's mode, so under the common 0002 ``/tmp/qda_<user>`` came out 0775 and `cache_root_problem` then
    refused the root qda had just made. Directories that exist are left as they are.
    """
    for directory in [*reversed(path.parents), path]:
        if not directory.exists():
            with contextlib.suppress(FileExistsError):  # another process made it first
                directory.mkdir(mode=0o700)


#: What to do about a `cache_problem`: the four variables, named, because the user moves them together.
CACHE_FIX = (
    "point the engine caches at a writable local directory: export FLASHINFER_WORKSPACE_BASE, "
    "TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR and VLLM_CACHE_ROOT"
)


def cache_problem(base: Path | None = None) -> str | None:
    """Why ``base`` (default `cache_root`) cannot hold an engine's caches, or None.

    A cache is usable when this user can make it a private directory and take a file lock in it: FlashInfer,
    Triton and vLLM all guard their first compile with one, and a filesystem that refuses the lock (a network
    mount without lockd) fails the engine start instead. `qda doctor` reports this line; nothing refuses a run
    on it, so a user whose caches work anyway is not stopped.
    """
    base = cache_root() if base is None else base
    try:
        private_mkdir(base)
        with tempfile.NamedTemporaryFile(dir=base) as probe:
            _flock(probe.fileno())
    except OSError as error:
        return f"{base}: {error}"
    return None


#: `mountinfo` escapes a space, a tab, a newline and a backslash as three octal digits.
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


def fstype(base: Path) -> str | None:
    """The filesystem type of ``base``, from `/proc/self/mountinfo`; None where that cannot be read.

    `qda doctor --report` names the filesystem each cache and temporary location is on, so a bug report says
    where a failing lock or a slow first compile lives without printing the machine's paths.
    """
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    except OSError:  # not Linux
        return None
    return fstype_from_mountinfo(base.expanduser().resolve(), mountinfo)


def fstype_from_mountinfo(path: str | Path, mountinfo: str) -> str | None:
    """The type of the filesystem mounted at the longest mount point ``path`` is under, or None."""
    base, best, found = Path(path), "", None
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        fields, rest = left.split(), right.split()
        if not separator or len(fields) < 5 or not rest:
            continue
        mount = Path(_MOUNTINFO_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), fields[4]))
        if (base == mount or base.is_relative_to(mount)) and len(str(mount)) > len(best):
            best, found = str(mount), rest[0]
    return found


@contextlib.contextmanager
def engine_output() -> Iterator[Callable[[], str]]:
    """Capture everything written to stderr, this process's and its children's, while it still reaches the screen.

    Every byte is passed through to the original stderr as it arrives, so an engine's progress is unchanged. The
    yielded function returns what was written while the block is open; it is the only record of a child's
    traceback, which vLLM replaces with a bare ``RuntimeError`` in the parent (see
    the parent). A running engine's children keep the pipe, so after the block the pump goes on
    forwarding their output but no longer records it.
    """
    try:
        saved = os.dup(2)
    except OSError:  # fd 2 is closed: there is nothing to capture, and nothing to forward to
        yield lambda: ""
        return
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    chunks: list[bytes] = []
    recording = [True]
    forward = os.dup(saved)  # the pump owns this until every child's pipe writer closes

    def pump() -> None:
        try:
            while chunk := os.read(read_fd, 65536):
                if recording[0]:
                    chunks.append(chunk)
                with contextlib.suppress(OSError):
                    os.write(forward, chunk)
        finally:
            os.close(read_fd)
            os.close(forward)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    # Python-level writes may go to a wrapper of the caller's own (a test runner's capture object), so
    # ``sys.stderr`` is pointed at the same pipe: everything written while an engine starts passes through one place.
    original = sys.stderr
    sys.stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", errors="backslashreplace", buffering=1)

    def text() -> str:
        sys.stderr.flush()
        thread.join(timeout=0.5)  # a child's last write is already in the pipe when its parent sees it exit
        return b"".join(chunks).decode("utf-8", "replace")

    try:
        yield text
    finally:
        recording[0] = False  # a live engine's later output is forwarded, not kept in memory
        restore, sys.stderr = sys.stderr, original
        with contextlib.suppress(OSError, ValueError):
            restore.close()
        # stop capturing: fd 2 is the terminal again, and the pipe's last local write end closes
        os.dup2(saved, 2)
        thread.join(timeout=5.0)
        os.close(saved)


def engine_start(
    env: Mapping[str, str],
    start: Callable[[], T],
    refusal: Callable[[str], str | Refused | None] | None = None,
) -> tuple[T, dict[str, str]]:
    """Start an engine in ``env`` with the caches resolved (:func:`cache_env`) and the banned kernels off.

    The one start rule, without a ``ServeSettings``: any command that starts a vLLM engine calls it with its own
    environment and a callable that builds the engine. ``refusal`` reads a failed start's error and the engine's
    own output (a child's traceback is only there) and may name the failure in one line, or return a
    :class:`~qda_quant.step.Refused` that carries its fix; it is then raised instead.

    Returns:
        the value ``start`` returned and the environment it started in (a caller that records its serve settings
        should record this one).
    """
    resolved = cache_env(disabled_kernels(env))
    with engine_output() as output:
        try:
            with vllm_environ(resolved):
                return start(), resolved
        except Exception as error:
            text = output()
            _refuse(refusal, error, text)
            add_fix(error, text)
            raise


def _download_kernels() -> str:
    """This venv's ``vllm download-kernels``, by path: it installs FlashInfer's prebuilt kernels."""
    vllm = Path(sys.executable).with_name("vllm")
    if vllm.exists():
        return f"{vllm} download-kernels"
    return f"{sys.executable} -m vllm.entrypoints.cli.main download-kernels"


#: The command that installs FlashInfer's prebuilt kernels (``flashinfer_jit_cache``) into the venv qda runs from.
DOWNLOAD_KERNELS = _download_kernels()


#: Upstream errors an engine start dies with whose fix QDA knows: what the error says, its fix, and what else the
#: output must say for that fix to apply (None: nothing).
ENGINE_FIXES: tuple[tuple[re.Pattern[str], str, re.Pattern[str] | None], ...] = (
    (
        # FlashInfer's MoE autotune runs vLLM's batch (16384 tokens) after the KV cache is allocated
        re.compile(r"OutOfMemoryError: CUDA out of memory"),
        "halve vLLM's batch: qda-kl kl: --serve-args '{\"max_num_batched_tokens\": 8192}'; qda-kl score: "
        "scorer.topk.serve_args.max_num_batched_tokens: 8192; a search: "
        "judge.kl_layout.serve_args.max_num_batched_tokens: 8192",
        re.compile(r"_run_flashinfer_autotune_dummy_runs"),
    ),
    (
        re.compile(r"Could not find nvcc|nvcc: not found|No such file or directory: '[^']*nvcc'"),
        "install the CUDA toolkit or put its nvcc on PATH (or set CUDA_HOME to it); or run "
        f"`{DOWNLOAD_KERNELS}` once for prebuilt kernels",
        None,
    ),
    (re.compile(r"No such file or directory: 'ninja'"), "pip install ninja into this venv", None),
    (re.compile(r"Failed to find C compiler"), "install gcc, or set CC to a C compiler", None),
    (
        re.compile(r"\bc\+\+: not found|No such file or directory: 'c\+\+'"),
        "install g++, or set CXX to a C++ compiler",
        None,
    ),
)

#: vLLM's warning on every engine start where the kernels are prebuilt (`vllm download-kernels`) and nvcc is
#: absent: not a failure, so it is removed before the patterns are read.
NVCC_QUERY = re.compile(
    r"Could not query nvcc for [^:]*: Could not find nvcc and default cuda_home='[^']*' doesn't exist\.?"
)


def engine_fix(text: str) -> tuple[str, str] | None:
    """The upstream line in ``text`` that :data:`ENGINE_FIXES` recognizes, and its fix; None when none does."""
    text = NVCC_QUERY.sub("", text)
    for pattern, fix, within in ENGINE_FIXES:
        if within is not None and not within.search(text):
            continue
        for line in text.splitlines():
            if pattern.search(line):
                return line.strip(), fix
    return None


def add_fix(error: BaseException, output: str = "") -> None:
    """Note a known fix on a failed engine start: the upstream line that names the failure, then ``fix:``.

    The error is kept as raised. vLLM reports a child's failure as a bare ``RuntimeError`` and leaves the cause in
    the engine's output, so that line is noted first when the error's own message lacks it.
    """
    found = engine_fix(f"{error}\n{output}")
    if found is None:
        return
    line, fix = found
    if line not in str(error):
        error.add_note(line)
    error.add_note(f"{FIX_LINE}{fix}")


def _refuse(refusal: Callable[[str], str | Refused | None] | None, error: Exception, output: str) -> None:
    """Raise ``refusal``'s refusal for a failed start, when it names one."""
    if refusal is not None and (why := refusal(f"{error}\n{output}")):
        raise (why if isinstance(why, Refused) else Refused(why)) from error


def _flock(fd: int) -> None:
    """An acquired or busy lock proves flock support; all other errors propagate."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """This process's environment for a child that imports what this process imports, then ``env``.

    ``PYTHONPATH`` is this process's, then its ``sys.path``: a source tree put there, or a package found only
    there. vLLM's engine belongs in a child process, which starts with this environment.
    """
    path = [p for p in (os.environ.get("PYTHONPATH", ""), *sys.path) if p]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(path), **(env or {})}


def tool_path(path: str | None = None) -> str:
    """``path`` (default this process's ``PATH``) with this interpreter's ``bin`` first, so venv tools are found.

    vLLM's sampler compiles FlashInfer kernels with ``ninja`` from ``PATH``. The ``ninja`` wheel puts it in the
    venv's ``bin``, which is on ``PATH`` only in an activated venv or under ``uv run``.
    """
    path = os.environ.get("PATH", "") if path is None else path
    venv_bin = str(Path(sys.executable).parent)
    return path if venv_bin in path.split(os.pathsep) else os.pathsep.join(p for p in (venv_bin, path) if p)


def _nvcc(env: Mapping[str, str]) -> str | None:
    """The ``nvcc`` FlashInfer's JIT would run under ``env`` (CUDA_HOME, CUDA_PATH, PATH, /usr/local/cuda), or None."""
    home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not home:
        found = shutil.which("nvcc", path=env.get("PATH", ""))
        home = str(Path(found).parent.parent) if found else "/usr/local/cuda"
    return shutil.which(str(Path(home) / "bin" / "nvcc"))


def flashinfer_kernels_problem() -> str | None:
    """Why FlashInfer has no prebuilt kernels it will use; None when it has them or FlashInfer is not installed.

    FlashInfer uses ``flashinfer_jit_cache`` only at its own version, a CUDA local suffix allowed
    (``0.7.2.post1+cu129``) or with ``FLASHINFER_DISABLE_VERSION_CHECK`` set, and JIT-compiles every kernel
    otherwise (``flashinfer.jit.env``).
    """
    if importlib.util.find_spec("flashinfer") is None:
        return None
    if importlib.util.find_spec("flashinfer_jit_cache") is None:
        return "no prebuilt FlashInfer kernels"
    if os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"):
        return None
    try:
        cache, flashinfer = md.version("flashinfer-jit-cache"), md.version("flashinfer-python")
    except md.PackageNotFoundError:
        return None
    if cache == flashinfer or cache.startswith(f"{flashinfer}+"):
        return None
    return f"prebuilt FlashInfer kernels {cache} are not for flashinfer {flashinfer}"


def flashinfer_cannot_build() -> bool:
    """Whether an engine started here would JIT-build a FlashInfer kernel and cannot.

    True when FlashInfer has no prebuilt kernels it will use (:func:`flashinfer_kernels_problem`) and its JIT finds no nvcc (:func:`_nvcc`) or no C++ compiler (``CXX``, else
    ``c++``) on the engine's ``PATH`` (:func:`tool_path`).
    """
    if flashinfer_kernels_problem() is None:
        return False
    env = {**os.environ, "PATH": tool_path()}
    return _nvcc(env) is None or shutil.which(env.get("CXX") or "c++", path=env["PATH"]) is None


#: Host C++ compilers tried, in order, when the one nvcc picks by default cannot find the system headers.
HOST_COMPILERS = ("/usr/bin/g++", "/usr/bin/c++")


@contextlib.contextmanager
def _probe_directory() -> Iterator[Path]:
    """A temporary directory for the nvcc probe, independent of the engine's (possibly replaced) tempdir.

    The FlashInfer fallback tests point ``tempfile`` at a directory that does not exist yet, and the probe is
    not part of that path; ``$TMPDIR`` and ``/tmp`` are tried when the process tempdir cannot hold it.
    """
    for candidate in (tempfile.gettempdir(), os.environ.get("TMPDIR"), "/tmp"):
        if not candidate or not Path(candidate).is_dir():
            continue
        try:
            folder = Path(tempfile.mkdtemp(prefix="qda-nvcc-", dir=candidate))
        except OSError:
            continue
        try:
            yield folder
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return
    raise OSError("no writable temporary directory for nvcc's probe")


def _host_compiler_problem(
    nvcc: str, compiler: str | None, env: Mapping[str, str], timeout: float = 120
) -> str | None:
    """What nvcc reports preprocessing a trivial file with ``compiler`` (``None``: nvcc's default), or None."""
    try:
        with _probe_directory() as folder:
            source = Path(folder, "probe.cu")
            source.write_text("int qda_probe;\n")
            command = [nvcc, "-E", *(["-ccbin", compiler] if compiler else []), str(source)]
            done = subprocess.run(command, env=dict(env), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as why:
        return str(why)
    if done.returncode == 0:
        return None
    lines = [line.strip() for line in (done.stderr or done.stdout).splitlines() if line.strip()]
    if not lines:
        return f"nvcc exited {done.returncode}"
    return next((line for line in lines if "error" in line.lower()), lines[-1])


def _host_compiler_candidates(env: Mapping[str, str]) -> Iterator[str]:
    """``CXX``, then :data:`HOST_COMPILERS`, then the g++ nvcc's default would find on ``PATH``."""
    seen: set[str] = set()
    found = (env.get("CXX"), *HOST_COMPILERS, shutil.which("g++", path=env.get("PATH", "")))
    for compiler in found:
        if compiler and compiler not in seen:
            seen.add(compiler)
            yield compiler


#: The variables nvcc reads extra command-line flags from, around the ones it is given (CUDA 11.2+).
NVCC_FLAG_VARS = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")


def _explicit_host_compiler(env: Mapping[str, str]) -> str | None:
    """The host compiler ``env`` already names for nvcc: ``NVCC_CCBIN``, or ``-ccbin`` in its flags, else None."""
    if compiler := env.get("NVCC_CCBIN"):
        return compiler
    for name in NVCC_FLAG_VARS:
        words = (env.get(name) or "").split()
        for index, word in enumerate(words):
            if word in ("-ccbin", "--compiler-bindir") and index + 1 < len(words):
                return words[index + 1]
            if word.startswith(("-ccbin=", "--compiler-bindir=")):
                return word.split("=", 1)[1]
    return None


@dataclass(frozen=True)
class HostCompilerSelection:
    """What nvcc's host compiler will be, or why the engine has none.

    Attributes:
        compiler: the ``NVCC_CCBIN`` to set; None when nvcc's own choice (or the environment's) preprocesses.
        problem: nvcc's own message when the engine would refuse to start.
        explicit: the compiler the environment already gives nvcc, when ``problem`` is about that one.
        unfinished: a probe hit the caller's deadline, so neither a compiler nor a refusal was established.
    """

    compiler: str | None = None
    problem: str | None = None
    explicit: str | None = None
    unfinished: bool = False


def select_host_compiler(
    env: Mapping[str, str], *, probe_timeout: float = 120, deadline: float | None = None
) -> HostCompilerSelection:
    """Select the host compiler nvcc runs, within an optional time budget (the engine's and the doctor's one rule).

    DeepGEMM's bundled JIT (vLLM's block-scaled FP8 experts) invokes ``nvcc`` without ``-ccbin``, and nvcc
    then runs the ``g++`` on ``PATH``. A module-loaded gcc found there can have an include search without
    ``/usr/include/<triplet>``, so the JIT dies on ``bits/wordsize.h`` before the engine starts.

    A compiler ``env`` already names for nvcc (``NVCC_CCBIN`` or a ``-ccbin`` in ``NVCC_PREPEND_FLAGS`` /
    ``NVCC_APPEND_FLAGS``) is the caller's choice: it is probed exactly as the engine will run it, and never
    replaced. Otherwise nvcc's default is probed first; when it fails, the first of ``CXX`` and
    :data:`HOST_COMPILERS` that preprocesses is selected. Nothing is probed when nvcc is absent.

    Args:
        env: the environment the engine will start with.
        probe_timeout: the most one ``nvcc -E`` may take.
        deadline: a ``time.monotonic()`` value after which no probe is started, or None for no budget.

    Returns:
        The selection; ``unfinished`` when the deadline is spent before a probe decides, so a caller with a
        budget can report that it could not establish either outcome.
    """
    nvcc = _nvcc(env)
    if nvcc is None:
        return HostCompilerSelection()
    timeout = _probe_timeout(probe_timeout, deadline)
    if timeout is None:
        return HostCompilerSelection(unfinished=True)
    problem = _host_compiler_problem(nvcc, None, env, timeout)
    if problem is None:
        return HostCompilerSelection()
    if explicit := _explicit_host_compiler(env):
        return HostCompilerSelection(problem=problem, explicit=explicit)
    for compiler in _host_compiler_candidates(env):
        timeout = _probe_timeout(probe_timeout, deadline)
        if timeout is None:
            return HostCompilerSelection(unfinished=True)
        if _host_compiler_problem(nvcc, compiler, env, timeout) is None:
            return HostCompilerSelection(compiler=compiler)
    return HostCompilerSelection(problem=problem)


def _probe_timeout(limit: float, deadline: float | None) -> float | None:
    """One probe's timeout: ``limit`` at most and no more than the deadline leaves; None when it is spent."""
    if deadline is None:
        return limit
    remaining = min(limit, deadline - time.monotonic())
    return remaining if remaining > 0 else None


def host_compiler_env(env: Mapping[str, str]) -> dict[str, str]:
    """``NVCC_CCBIN`` when nvcc's default host compiler cannot preprocess a translation unit.

    The selection is :func:`select_host_compiler` with no time budget, so a working compiler is found or the
    engine is refused before it starts.

    Args:
        env: the environment the engine will start with (this process's, merged with the caller's).

    Returns:
        ``{"NVCC_CCBIN": compiler}`` when nvcc's default host compiler cannot preprocess, else ``{}``.

    Raises:
        Refused: the explicit compiler, or the default and every candidate, failed to preprocess.
    """
    selection = select_host_compiler(env)
    if selection.unfinished:  # no deadline is given here; refuse rather than start an engine that may fail
        raise Refused(
            "vLLM's CUDA JIT cannot use a host compiler: nvcc's host compiler was not probed in time",
            code=ENVIRONMENT,
        )
    if selection.compiler is not None:
        return {"NVCC_CCBIN": selection.compiler}
    if selection.problem is None:
        return {}
    if selection.explicit:
        raise Refused(
            f"vLLM's CUDA JIT cannot use the host compiler nvcc was given ({selection.explicit}): "
            f"{selection.problem}. Remove -ccbin/NVCC_CCBIN or point it at a compiler that finds the system "
            "headers"
        )
    raise Refused(
        f"vLLM's CUDA JIT cannot use a host compiler: {selection.problem}. Set NVCC_CCBIN to a C++ compiler "
        "that finds the system headers (for example /usr/bin/g++)",
        code=ENVIRONMENT,
    )
