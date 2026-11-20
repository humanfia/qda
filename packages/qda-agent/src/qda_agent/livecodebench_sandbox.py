"""The bubblewrap sandbox that runs one LiveCodeBench program, and the trusted grader inside it.

Every program runs in its own bubblewrap mount namespace built from an empty root: no network, only the
interpreter's trees mounted read-only, one scratch directory writable, and a limit of ``timeout_s`` seconds
per case: wall clock under the official runner, CPU time under the paper's. The sandbox carries no GPU.
Resource caps inside it limit one program to 4 GiB, 64 processes and 64 MiB per file, and the tree is an
allowlist, so anything not named is absent rather than hidden. The sandbox's /proc is fresh by default
(``proc: private``); a node whose container refuses a fresh proc mount needs the explicit
``proc: host-readonly`` setting, which binds the host's /proc read-only.

The suite's ``sandbox: none`` (``--lcb-sandbox none``) runs the same grader with no sandbox at all
(:func:`direct_argv`): the generated program then sees this machine, and only the per-case limits and the
memory and file-size caps still apply.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from qda_agent.isolation.bwrap import (
    BIND_FD,
    BwrapTooOld,
    Pinned,
    bwrap_allowlist_argv,
    interpreter_roots,
    pin_sources,
    private_args,
)

#: The vendored official grader, the paper's runner and the program run inside the sandbox.
VENDOR = Path(__file__).parent / "third_party" / "livecodebench"
GRADER = VENDOR / "grade_one.py"
PAPER_RUNNER = VENDOR / "paper.py"
#: The paper runner's wall-clock limit per case, as a multiple of its CPU limit: a program blocked off-CPU
#: (asleep, waiting on a lock) still ends.
WALL_BACKSTOP = 3
RESULT_MARK = "@@qda-lcb-result@@"
#: The grader's own ``NO_PROCESS_CAP``: it runs as a script, so both ends name the flag as they do the mark.
NO_PROCESS_CAP = "--no-process-cap"

#: The read-only host trees the sandboxed interpreter loads from; besides them only the interpreter's own
#: installation (:func:`interpreter_trees`) is mounted from the host.
SYSTEM_TREES = ("/usr", "/lib", "/lib64")

#: The grader's way out of a bubblewrap too old to bind by descriptor: the eval's own knob, not the search's
#: `scheduler.job_sandbox` (:data:`qda_agent.isolation.bwrap.BWRAP_FIX`), which ``qda eval`` does not read.
BWRAP_FIX = (
    f"install bubblewrap >= {BIND_FD}, or accept running generated code unsandboxed on this machine with "
    "--lcb-sandbox none (sandbox: none on the suite)"
)

#: The sandbox's own mount points: the vendored grader, the interpreter's NumPy tree, and the case scratch.
GRADER_DIR = "/grader"
DEPS_DIR = "/deps"
SCRATCH_DIR = "/scratch"

#: The sandbox's tmpfs at /dev/shm: a program's working files, bounded.
SHM_CAP = 256 * 1024**2

#: The scratch directory each case's stdout and stderr are written to, and the bytes kept for reading them.
STDOUT_FILE = ".stdout"
STDERR_FILE = ".stderr"
READ_TAIL = 1 << 20


def dependency_mounts() -> list[tuple[Path, str]]:
    """The NumPy tree the vendored runner imports, as ``(host path, name under /deps)``.

    Raises:
        RuntimeError: NumPy is not importable, so the sandbox could not grade any case.
    """
    spec = importlib.util.find_spec("numpy")
    if spec is None or spec.origin is None:
        raise RuntimeError("livecodebench: cannot find numpy, which the vendored grader imports")
    package = Path(spec.origin).parent
    out = [(package, package.name)]
    libs = package.parent / f"{package.name}.libs"
    if libs.is_dir():
        out.append((libs, libs.name))
    return out


def interpreter() -> Path:
    """The base interpreter (not the venv's shim): its prefix and stdlib are the only trees mounted."""
    return Path(getattr(sys, "_base_executable", None) or sys.executable)


def interpreter_trees() -> list[Path]:
    """The base interpreter's own installation, where it lies outside :data:`SYSTEM_TREES`.

    A system Python lives under ``/usr`` and needs nothing more. A Python that uv (``--managed-python``),
    pyenv or conda installed lives elsewhere, often under the user's home, and the sandbox cannot exec it
    with only the system trees mounted. These are that installation's directories as
    :func:`qda_agent.isolation.bwrap.interpreter_roots` names them (as spelled and under their real paths),
    mounted read-only; the venv, the rest of the home directory and the host root stay absent.

    Raises:
        RuntimeError: the installation is the home directory or contains it (a Python in ``~/bin``), which
            would mount the whole home directory for the generated code.
    """
    system = [Path(tree) for tree in SYSTEM_TREES]
    home = Path.home()
    homes = {home, Path(os.path.realpath(home))}
    trees: list[Path] = []
    for root in interpreter_roots(interpreter(), base_prefix=sys.base_prefix):
        if root == Path("/") or any(root.is_relative_to(tree) for tree in system):
            continue
        if any(h.is_relative_to(root) for h in homes):
            raise RuntimeError(
                f"livecodebench: refused: the interpreter's installation {root} is the home directory or "
                "contains it, so the sandbox would mount the whole home directory for generated code; "
                "install Python elsewhere or use a system Python"
            )
        if root not in trees:
            trees.append(root)
    return trees


#: The case sandbox's whole environment: no host variable reaches it.
SANDBOX_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/nonexistent",
    "TMPDIR": SCRATCH_DIR,
    "PYTHONPATH": f"{GRADER_DIR}:{DEPS_DIR}",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "LC_CTYPE": "C.UTF-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def sandbox_argv(
    where: Path,
    status_fd: int | None,
    *,
    ready_fd: int | None = None,
    run_dir: Path | None = None,
    host_written: Iterable[Path] = (),
    proc: str = "private",
) -> tuple[list[str], dict[Path, Pinned]]:
    """The bubblewrap command that runs the grader in a sandbox on one scratch directory, and its sources.

    The sandbox is an allowlist (:func:`qda_agent.sched.sandbox.bwrap_allowlist_argv`): a fresh root with only
    the system trees and the interpreter's own installation, the NumPy tree the grader imports, the vendored
    grader and ``where``; ``where`` is the only writable mount, and /dev/shm is a sized tmpfs. Bubblewrap's JSON status
    reports ``child-pid`` before setup; only the trusted grader's readiness pipe proves it can run a case.

    Every source is opened and checked now by :func:`qda_agent.sched.sandbox.pin_sources` and bound by its
    descriptor. ``where`` is created by the caller inside ``host_written``: a case directory that resolves into
    ``run_dir`` anywhere else (a judge directory, say) is refused. The caller passes the descriptors with
    ``pass_fds`` and closes them once the child has started.

    Args:
        where: The case directory, created by the caller inside its scratch.
        status_fd: The descriptor bwrap writes its JSON status to; None disables status reporting.
        ready_fd: A pipe the trusted grader writes and closes before executing generated code.
        run_dir: The directory no source may come from (the eval's out directory); None: no check.
        host_written: Areas inside ``run_dir`` the caller writes, which a mount may come from (the scratch).
        proc: The /proc view: ``private`` (a fresh procfs, only this sandbox's processes) or ``host-readonly``
            (the host's /proc bound read-only, for a node whose container refuses a fresh mount).

    Returns:
        The bwrap command, and the opened sources, which the caller must close.

    Raises:
        ValueError: A source resolves into ``run_dir`` outside ``host_written``, or cannot be opened.
    """
    readable = {Path(tree): Path(tree) for tree in SYSTEM_TREES if Path(tree).exists()}
    readable.update({tree: tree for tree in interpreter_trees()})
    deps = {Path(DEPS_DIR) / name: path for path, name in dependency_mounts()}
    mounts = {**readable, **deps, Path(GRADER_DIR): VENDOR, Path(SCRATCH_DIR): where}
    pinned = pin_sources([], mounts, run_dir=run_dir, host_written=host_written)
    try:
        argv = bwrap_allowlist_argv(
            [
                str(interpreter()),
                "-s",
                "-S",
                f"{GRADER_DIR}/{GRADER.name}",
                *([str(ready_fd)] if ready_fd is not None else []),
            ],
            SCRATCH_DIR,
            mounts=mounts,
            writable=[Path(SCRATCH_DIR)],
            pinned=pinned,
            env=SANDBOX_ENV,
            status_fd=status_fd,
            shm_bytes=SHM_CAP,
            proc=proc,
        )
    except BaseException as e:
        for pin in pinned.values():
            os.close(pin.fd)
        if isinstance(e, BwrapTooOld):
            raise BwrapTooOld(e.found, sandbox="the LiveCodeBench grading sandbox", fix=BWRAP_FIX) from None
        raise
    return argv, pinned


def direct_argv(where: Path, ready_fd: int) -> tuple[list[str], dict[str, str]]:
    """The command and environment that run the grader on this machine directly, outside any sandbox.

    Only ``sandbox: none`` asks for this: the generated program sees the host's files and network. The
    interpreter runs with ``-s -S``, so the grader's own trees are named on ``PYTHONPATH``; the case
    directory is the working directory, the temporary directory and the home directory, which is as much as
    an unsandboxed run can keep apart. The grader leaves the process cap alone here, where it counts the
    whole login's processes.
    """
    deps = sorted({str(path.parent) for path, _ in dependency_mounts()})
    env = {
        **SANDBOX_ENV,
        "PATH": os.environ.get("PATH", SANDBOX_ENV["PATH"]),
        "PYTHONPATH": os.pathsep.join([str(VENDOR), *deps]),
        "TMPDIR": str(where),
        "HOME": str(where),
    }
    return [str(interpreter()), "-s", "-S", str(GRADER), str(ready_fd), NO_PROCESS_CAP], env


def _tail_fd(fd: int, limit: int) -> str:
    """The last ``limit`` bytes of an open regular file as text; anything else, or an error, reads as empty.

    The sandbox's program can rename or replace anything under its scratch, so the host reads only what it
    opened before the sandbox started, by that descriptor, and only when it is still a regular file.
    """
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return ""
        raw = os.pread(fd, limit, max(0, info.st_size - limit))
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")


def _remove_tree(root: Path) -> None:
    """Remove a case directory after giving the owner its rights back: the program may chmod a directory to 0.

    Only real directories are chmodded (``os.walk`` lists no symlink as a directory); nothing is followed.
    """
    with contextlib.suppress(OSError):
        root.chmod(0o700)
    for dirpath, dirnames, _ in os.walk(root):  # top-down: a directory is fixed before it is entered
        for name in dirnames:
            with contextlib.suppress(OSError):
                Path(dirpath, name).chmod(0o700)
    shutil.rmtree(root, ignore_errors=True)


def _run_test(
    code: str,
    fn_name: str | None,
    cases: Sequence[dict],
    scratch: Path,
    timeout_s: float,
    *,
    run_dir: Path | None = None,
    host_written: Iterable[Path] = (),
    proc: str = "private",
    protocol: str = "official",
    sandbox: str = "bwrap",
) -> tuple[bool, int, int, int]:
    """Run all cases in one fresh sandbox; return passed, cases run, timeouts and errors.

    The sandbox is an allowlist (:func:`sandbox_argv`): an empty root, no network, the host tree absent
    except the interpreter's own, and one fresh scratch directory writable. Output goes to files in that
    directory, so a program that floods stdout fills a bounded file rather than the parent's memory.

    Args:
        code: The program to run.
        fn_name: The function it defines, or None for a stdin problem.
        cases: The cases' inputs and expected outputs, in official order.
        scratch: The caller's scratch directory, which the case directory is made under.
        timeout_s: Per-case limit. The outer watchdog allows startup and compilation separately, and the
            paper runner's wall-clock backstop per case.
        run_dir: The directory no source may come from; see :func:`sandbox_argv`.
        host_written: Areas inside ``run_dir`` the caller writes; see :func:`sandbox_argv`.
        proc: The /proc view; see :func:`sandbox_argv`.
        protocol: The case runner: ``official`` or ``paper``; see :func:`grade_code`.
        sandbox: ``bwrap`` (the sandbox) or ``none`` (directly on this machine); see :func:`grade_code`.

    Raises:
        RuntimeError: the sandbox or trusted grader failed to start; that is never a verdict.
    """
    where = Path(tempfile.mkdtemp(prefix="case-", dir=scratch))
    nonce = secrets.token_hex(16)
    payload = {
        "code": code,
        "cases": cases,
        "fn_name": fn_name,
        "timeout": timeout_s,
        "protocol": protocol,
        "wall_backstop": WALL_BACKSTOP,
        "nonce": nonce,
    }
    per_case = timeout_s * (WALL_BACKSTOP if protocol == "paper" else 1)
    ready_read, ready_write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    out_path, err_path = where / STDOUT_FILE, where / STDERR_FILE
    pinned: dict[Path, Pinned] = {}
    out_fd = err_fd = -1
    try:
        env: dict[str, str] | None = None
        if sandbox == "none":
            argv, env = direct_argv(where, ready_write)
        else:
            argv, pinned = sandbox_argv(
                where, None, ready_fd=ready_write, run_dir=run_dir, host_written=host_written, proc=proc
            )
        out_fd = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        err_fd = os.open(err_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            with private_args(argv) as (command, fds):
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=out_fd,
                    stderr=err_fd,
                    cwd=where,
                    env=env,
                    pass_fds=(ready_write, *(pin.fd for pin in pinned.values()), *fds),
                    start_new_session=True,
                )
        except OSError as why:
            raise RuntimeError(
                f"livecodebench: refused: cannot launch the grader: {why}; "
                + (
                    "repair this machine's Python interpreter"
                    if sandbox == "none"
                    else "install bubblewrap (bwrap) and ensure it is executable"
                )
            ) from why
        os.close(ready_write)
        ready_write = -1
        timed_out = False
        try:
            child.communicate(json.dumps(payload).encode(), timeout=per_case * len(cases) + timeout_s * 2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)  # bwrap's group; --die-with-parent reaps the sandbox
            child.communicate()
            timed_out = True
        ready = False
        with contextlib.suppress(BlockingIOError):
            ready = os.read(ready_read, 1) == b"1"
        if not ready:
            cause = _tail_fd(err_fd, 400).strip() or (
                "startup timed out"
                if timed_out
                else f"launcher exited {child.returncode} before grader readiness"
            )
            if sandbox == "none":
                raise RuntimeError(
                    f"livecodebench: refused: the grader did not start: {cause}; ensure this machine's "
                    "Python interpreter and NumPy are usable"
                )
            hint = (
                "; if this node's container forbids a fresh /proc, set proc: host-readonly on the "
                "livecodebench suite, or ask the administrator to permit a private /proc"
                if proc == "private"
                else ""
            )
            raise RuntimeError(
                f"livecodebench: refused: the sandbox did not start: {cause}; use bubblewrap with "
                "descriptor-bind support and ensure the sandbox's Python interpreter and NumPy are usable"
                f"{hint}"
            )
        if timed_out:
            return False, 1, 1, 0
        out = _tail_fd(out_fd, READ_TAIL)
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
        for fd in (out_fd, err_fd):
            if fd >= 0:
                os.close(fd)
        os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)
        _remove_tree(where)
    marker = f"\n{RESULT_MARK}{nonce}"
    at = out.rfind(marker)
    if at < 0:
        return False, 1, 0, 1
    try:
        got = json.loads(out[at + len(marker) :].split("\n", 1)[0])
    except json.JSONDecodeError:
        return False, 1, 0, 1
    return bool(got.get("passed")), len(got.get("results", [])), int(got.get("timed_out", False)), 0


def grade_code(
    code: str,
    fn_name: str | None,
    cases: Sequence[dict],
    scratch: Path,
    timeout_s: float,
    *,
    run_dir: Path | None = None,
    host_written: Iterable[Path] = (),
    proc: str = "private",
    protocol: str = "official",
    sandbox: str = "bwrap",
) -> tuple[bool, int, int, int]:
    """Run one program against every case; it passes only if every case does.

    Args:
        code: The program to run.
        fn_name: The function it defines, or None for a stdin problem.
        cases: The problem's public and private cases.
        scratch: The caller's scratch directory.
        timeout_s: Limit of one case: wall clock under ``official``, CPU time under ``paper``.
        run_dir: The directory no source may come from; see :func:`sandbox_argv`.
        host_written: Areas inside ``run_dir`` the caller writes; see :func:`sandbox_argv`.
        proc: The /proc view; see :func:`sandbox_argv`.
        protocol: ``official`` runs LiveCodeBench's ``testing_util.run_test``; ``paper`` runs the paper's
            runner (``third_party/livecodebench/paper.py``): each stdin case in a child interpreter, every limit on
            CPU time with a wall-clock backstop.
        sandbox: ``bwrap`` runs the program in the sandbox; ``none`` runs it on this machine
            (:func:`direct_argv`), which the suite's ``sandbox: none`` asks for.

    Returns:
        ``(passed, cases_run, timeouts, errors)``; both runners stop at the first failure.

    Raises:
        RuntimeError: a sandbox failed to start; no score is returned.
    """
    return _run_test(
        code,
        fn_name,
        cases,
        scratch,
        timeout_s,
        run_dir=run_dir,
        host_written=host_written,
        proc=proc,
        protocol=protocol,
        sandbox=sandbox,
    )
