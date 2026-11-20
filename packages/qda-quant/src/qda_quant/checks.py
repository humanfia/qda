"""The built-in ``qda doctor`` checks: hardware, build and serve prerequisites.

A line is a MISS only where a command in doctor's scope refuses on the same condition: the GPU listing, a
faulty card or a driver too old for torch's CUDA (`BUILD_GUARDS`, which ``qda build`` walks; an engine start
walks the driver's too), the C compiler Triton builds with, a cache directory no engine can lock. What depends
on a model or format doctor is not given (a GDN model's kernels, a W3A16 checkpoint's Humming, a GGUF build's
converter, the kernels an engine compiles) is left to the step that decides it, which fails with its own error;
what only informs (glibc, the tools found, the SGLang cache's path form) is ``--report``'s. The checks read
facts only: an installed distribution's version, ``nvidia-smi``'s listing, a tool on ``PATH``, a path's real
form, one small C compile, the driver's CUDA version and ``cuInit`` in a child process and, when nvcc is
installed, one CUDA preprocess with the host compiler the engine gives it. No GPU job runs.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .doctor import Check, Context, Info, guard_lines
from .guards import BUILD, SERVE, Guard, Problem, all_problems, first_problem
from .output.vllm_env import (
    CACHE_ENV,
    CACHE_FIX,
    DOWNLOAD_KERNELS,
    _nvcc,
    cache_env,
    cache_problem,
    cache_root,
    cache_root_problem,
    flashinfer_kernels_problem,
    fstype,
    select_host_compiler,
    tool_path,
)
from .step import Refused
from .target import Target

if TYPE_CHECKING:
    from .arch import ModelGraph

#: The variable that gives a process a subset of the machine's cards, in the order it names them.
VISIBLE_ENV = "CUDA_VISIBLE_DEVICES"


def _visible(cards: list[tuple[str, str, str, str]], value: str) -> list[int]:
    """The positions in `cards` of the cards `value` names, in its order.

    An entry is a card's index or a GPU UUID (a prefix of one), as CUDA reads it. A card can be named twice; a
    process then gets it twice.

    Raises:
        ValueError: an entry names no card on this machine, or the value names none at all.
    """
    by_index = {card[0]: position for position, card in enumerate(cards)}
    out = []
    for entry in (part.strip() for part in value.split(",")):
        if not entry:
            continue
        if entry.removeprefix("-").isdigit():
            if entry not in by_index:
                known = ", ".join(card[0] for card in cards)
                raise ValueError(f"no card has index {entry} (this machine lists {known})")
            out.append(by_index[entry])
            continue
        found = [position for position, card in enumerate(cards) if card[1].startswith(entry)]
        if len(found) != 1:
            raise ValueError(f"{entry!r} matches no single card UUID")
        out.append(found[0])
    if not out:
        raise ValueError("it is empty, so CUDA sees no card")
    return out


#: The ``nvidia-smi --query-gpu`` fields `card_fault` reads, in its order: a card with an uncorrectable ECC error
#: (since the driver loaded) or memory pages waiting to retire fails every CUDA call that touches it.
HEALTH_FIELDS = ("ecc.errors.uncorrected.volatile.total", "retired_pages.pending")

#: What ``nvidia-smi`` prints for a field it cannot read: absent (no ECC, row remapping instead of retirement) or
#: permission-limited (a shared node's counters are root's). Either says nothing about the card's health.
_UNKNOWN = ("[N/A]", "[Not Supported]", "N/A", "[Insufficient Permissions]")


def card_fault(values: list[str]) -> str | None:
    """Why a card is unfit for a job, from its `HEALTH_FIELDS` values in order; None for a healthy card.

    A bracketed value other than an unknown one (``[GPU requires reset]``, ``[Unknown Error]``) is a fault too.
    """
    errors, pending = (v.strip() for v in values)
    if errors.isdigit() and int(errors) > 0:
        return f"{errors} uncorrectable ECC error(s) since the driver loaded"
    if pending == "Yes":
        return "retired memory pages are pending (the card needs a reset)"
    for field, value in zip(HEALTH_FIELDS, (errors, pending), strict=True):
        if value.startswith("[") and value not in _UNKNOWN:
            return f"nvidia-smi reads {field} as {value}"
    return None


#: The oldest Linux driver branch of each CUDA major version (NVIDIA's CUDA toolkit release notes). A driver
#: runs every minor version of its own CUDA major and none of a later one, so only the major is compared.
DRIVER_FOR_CUDA = {12: "525", 13: "580"}

#: Run in a child, so this process never initializes CUDA: the version and ``cuInit`` result of the
#: ``libcuda.so.1`` the loader finds (``LD_LIBRARY_PATH`` first, so forward-compat libraries count), as torch's
#: own first CUDA call would see them.
_DRIVER_PROBE = """
import ctypes, sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print("noload", str(e).replace("\\n", " "))
    sys.exit()
version = ctypes.c_int(0)
lib.cuDriverGetVersion(ctypes.byref(version))
rc = lib.cuInit(0)
name = ctypes.c_char_p()
if rc:
    lib.cuGetErrorName(rc, ctypes.byref(name))
print(version.value, (name.value or str(rc).encode()).decode() if rc else "")
"""


def torch_cuda() -> tuple[str, str] | None:
    """The installed torch's version and the CUDA version it was built for, from ``torch/version.py``.

    Read as text, so the check imports no torch. None when torch is not installed or is a CPU build.
    """
    spec = importlib.util.find_spec("torch")
    # a spec that names no package directory (a stand-in some tests install) has no version file to read
    for root in getattr(spec, "submodule_search_locations", None) or []:
        try:
            text = (Path(root) / "version.py").read_text()
        except OSError:
            continue
        cuda = re.search(r"^cuda\b[^=\n]*=\s*['\"]([\d.]+)['\"]", text, re.M)
        version = re.search(r"^__version__\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        return (version.group(1) if version else "?", cuda.group(1)) if cuda else None
    return None


def cuda_driver() -> tuple[int | None, str]:
    """``(cuDriverGetVersion, cuInit's error name)`` of the driver a CUDA process here loads, from a child.

    The version is ``1000 * major + 10 * minor`` (12080 is CUDA 12.8); the error is "" when ``cuInit``
    succeeds. ``(None, why)`` when ``libcuda.so.1`` does not load, and ``(None, "not checked: ...")`` when the
    child does not answer: no verdict on the driver, which the GPU line says rather than passing.
    """
    argv = [sys.executable, "-I", "-c", _DRIVER_PROBE]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"not checked: the CUDA driver query did not finish: {type(e).__name__}"
    word, _, rest = p.stdout.strip().partition(" ")
    if word == "noload":
        return None, f"libcuda.so.1 does not load: {rest}"
    if p.returncode != 0 or not word.isdigit():
        why = (p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}").splitlines()[-1]
        return None, f"not checked: the CUDA driver query failed: {why}"
    return int(word), rest.strip()


def cuda_driver_problem(*, verdict_only: bool = False) -> tuple[str, str] | None:
    """Why the installed torch cannot start CUDA on this driver, and the fix; None when it can.

    ``nvidia-smi`` reads the kernel driver and says nothing about the CUDA version torch was built for: torch
    2.14 for CUDA 13 on a 570 driver (CUDA 12.8) lists every card and then fails its first CUDA call with
    "The NVIDIA driver on your system is too old". This compares the version the loaded ``libcuda.so.1``
    reports (forward-compat libraries on ``LD_LIBRARY_PATH`` raise it) with torch's, and runs ``cuInit``. Call it
    only where cards are expected (``nvidia-smi`` listed them, or a build was given CUDA devices): with no card,
    ``cuInit`` fails by design. None for a CPU-only torch. A query that gave no answer is a problem whose detail
    starts ``not checked:`` (the doctor's rule: a line is the real check's verdict or says what was not
    checked). With ``verdict_only`` (the build), only a verdict on the driver is a problem: no ``libcuda.so.1``
    at all, or no answer, is left to torch's first call, as before.
    """
    built = torch_cuda()
    if built is None:
        return None
    torch_version, need = built
    need_major = int(need.split(".")[0])
    floor = DRIVER_FOR_CUDA.get(need_major)
    upgrade = f"install an NVIDIA driver {floor} or newer" if floor else f"install a driver for CUDA {need}"
    compat = f"cuda-compat-{need.replace('.', '-')}"
    version, error = cuda_driver()
    if version is None:
        if verdict_only:
            return None
        if error.startswith("not checked:"):
            return error, "rerun; if it fails again, check that nvidia-smi works and libcuda.so.1 loads"
        return error, "install the NVIDIA driver"
    if version // 1000 < need_major:
        found = f"{version // 1000}.{version % 1000 // 10}"
        return (
            f"the driver supports CUDA {found}, but torch {torch_version} is built for CUDA {need}",
            f"{upgrade}; or, on a data-center card, put NVIDIA's {compat} libraries first on LD_LIBRARY_PATH "
            "(install docs: An older driver)",
        )
    if not error:
        return None
    if error == "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE":
        fix = f"this card does not take forward-compat libraries: take them off LD_LIBRARY_PATH and {upgrade}"
    elif error == "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH":
        fix = (
            "the libcuda on LD_LIBRARY_PATH does not match the kernel driver: take it off LD_LIBRARY_PATH, "
            f"or use a {compat} package this driver supports"
        )
    else:
        fix = f"check the cards with nvidia-smi; if CUDA still does not start, {upgrade}"
    return f"CUDA does not start: cuInit returned {error}", fix


def _smi_cards() -> tuple[list[tuple[str, str, str, str]], dict[str, str], list[str], str | None]:
    """``nvidia-smi``'s cards (index, uuid, name, driver), their faults by index, the rows it could not read, and
    why it gave no listing (None when it gave one)."""
    fields = ",".join(("index", "uuid", "name", "driver_version", *HEALTH_FIELDS))
    argv = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [], {}, [], type(e).__name__
    cards, faults, unread = [], {}, []
    for line in p.stdout.splitlines():
        row = [field.strip() for field in line.rsplit(",", 3 + len(HEALTH_FIELDS))]
        if len(row) == 4 + len(HEALTH_FIELDS):
            cards.append((row[0], row[1], row[2], row[3]))
            if fault := card_fault(row[4:]):
                faults[row[0]] = fault
        elif line.strip():  # a card nvidia-smi cannot read gets a message instead of its row
            unread.append(line.strip())
    if p.returncode != 0 or not cards:
        return cards, faults, unread, (p.stderr.strip() or p.stdout.strip() or "no cards").splitlines()[-1]
    return cards, faults, unread, None


def check_gpu() -> Check:
    """The cards a build gets and their driver, from ``nvidia-smi``, narrowed by ``CUDA_VISIBLE_DEVICES``.

    ``nvidia-smi`` lists every card on the machine, but CUDA gives a process only the cards its
    ``CUDA_VISIBLE_DEVICES`` names, and a build follows CUDA. The line names the variable when it narrowed the
    list, so a user on a shared node can see which card a build will use. A card's health is the ``card health``
    guard's (`card_health_problem`), which the build refuses on; this line does not judge it.
    """
    cards, _faults, unread, error = _smi_cards()
    if error is not None:
        return Check("GPU", False, f"nvidia-smi: {error}", "install the NVIDIA driver")
    visible = os.environ.get(VISIBLE_ENV)
    chosen = list(range(len(cards)))
    if visible is not None:
        try:
            chosen = _visible(cards, visible)
        except ValueError as why:
            return Check(
                "GPU",
                False,
                f"{VISIBLE_ENV}={visible!r}: {why}",
                f"set {VISIBLE_ENV} to the cards this machine has, or unset it for all of them",
            )
    first = cards[chosen[0]]
    detail = f"{len(chosen)} × {first[2]}, driver {first[3]}"
    detail += f" ({VISIBLE_ENV}={visible})" if visible is not None else ""
    return Check("GPU", True, detail + "".join(f"; nvidia-smi: {line}" for line in unread))


#: What a build tells the user to do about a faulty card it was given.
CARD_HEALTH_FIX = f"have the card reset or drained, or set {VISIBLE_ENV} to the healthy cards"


def card_health_problem(ordinals: Sequence[int] | None = None) -> Problem | None:
    """Why a build cannot use its cards: one nvidia-smi reports faulty (`card_fault`), or cannot read.

    Args:
        ordinals: The CUDA ordinals the build uses (``cuda:0`` is the first card ``CUDA_VISIBLE_DEVICES``
            names); None: every card this process can see.

    Returns:
        The refusal (exit 3), or None: no fault, or no listing to judge (the ``GPU`` line says why).
    """
    cards, faults, unread, chosen = _health_cards(ordinals)
    bad = [f"card {cards[i][0]}: {faults[cards[i][0]]}" for i in chosen if cards[i][0] in faults]
    bad += [f"nvidia-smi: {line}" for line in unread]
    return Problem("; ".join(bad), CARD_HEALTH_FIX) if bad else None


def _health_cards(
    ordinals: Sequence[int] | None,
) -> tuple[list[tuple[str, str, str, str]], dict[str, str], list[str], list[int]]:
    """nvidia-smi's cards, faults and unread rows, and the indices into its listing the build uses."""
    cards, faults, unread, error = _smi_cards()
    if error is not None:
        return [], {}, [], []
    try:
        chosen = (
            _visible(cards, os.environ[VISIBLE_ENV]) if VISIBLE_ENV in os.environ else list(range(len(cards)))
        )
    except ValueError:
        return [], {}, [], []  # the GPU line names the variable
    if ordinals is not None:
        chosen = [chosen[i] for i in ordinals if 0 <= i < len(chosen)]
    return cards, faults, unread, list(dict.fromkeys(chosen))


def _no_cards_to_judge(ctx: BuildCards) -> str | None:
    """Why there is no card to judge (the GPU line says why there is no listing), or None."""
    _cards, _faults, unread, chosen = _health_cards(ctx.ordinals)
    return None if chosen or unread else "no card to judge (the GPU line says why)"


def _healthy(ctx: BuildCards) -> str:
    cards, _faults, _unread, chosen = _health_cards(ctx.ordinals)
    return f"no fault on cards {[int(cards[i][0]) for i in chosen]}"


@dataclass(frozen=True)
class BuildCards:
    """What the build's guards read: the CUDA ordinals of its cards (None: every card it can see)."""

    ordinals: tuple[int, ...] | None = None


def _card_health(ctx: BuildCards) -> Problem | None:
    return card_health_problem(ctx.ordinals)


def _cuda_driver(ctx: BuildCards) -> Problem | None:  # noqa: ARG001  (the driver is the machine's, not a card's)
    """`cuda_driver_problem`'s verdict on this driver, as the refusal: the build's and an engine start's."""
    found = cuda_driver_problem(verdict_only=True)
    return Problem(found[0], found[1]) if found is not None else None


def _driver_unjudged(ctx: BuildCards) -> str | None:
    """Why there is no verdict on the driver here, or None.

    A CPU torch starts no CUDA. Doctor and an engine start (no ordinals) judge only where ``nvidia-smi`` lists a
    card, since ``cuInit`` fails by design without one; a build given CUDA ordinals is judged as it was before
    the guard (`cuda_driver_problem`'s rule). A ``libcuda.so.1`` that does not load, or a query that does not
    answer, is no verdict: the command leaves it to torch's first CUDA call, so it is a skip that says why.
    """
    if torch_cuda() is None:
        return "torch is a CPU build"
    if ctx.ordinals is None and (why := _no_cards_to_judge(ctx)) is not None:
        return why
    version, error = cuda_driver()
    return error if version is None else None


def _driver_holds(ctx: BuildCards) -> str:  # noqa: ARG001
    built = torch_cuda()
    return f"torch {built[0]} (CUDA {built[1]}) starts CUDA on this driver" if built else ""


_BUILD = frozenset({BUILD})
_SERVE = frozenset({SERVE})

#: What ``qda build`` refuses before it loads a weight, and what it and an engine start can only decide when
#: they run. ``engine.build.build`` walks the advance ones (`refuse_build`); ``qda doctor`` lists them all.
BUILD_GUARDS: tuple[Guard[BuildCards], ...] = (
    Guard(
        "card health",
        _BUILD,
        _card_health,
        "qda build, before it loads a weight (checks.refuse_build)",
        applies=_no_cards_to_judge,
        held=_healthy,
    ),
    Guard(
        "CUDA driver",
        _BUILD | _SERVE,
        _cuda_driver,
        "qda build and every vLLM engine start, before either loads a weight (checks.refuse_build, "
        "output.vllm_env.vllm_environ -> checks.refuse_serve)",
        applies=_driver_unjudged,
        held=_driver_holds,
    ),
    Guard(
        "Hub access",
        _BUILD,
        None,
        "the build's model download and calibration step (an offline node needs both in the Hub cache)",
        advance=False,
    ),
    Guard(
        "kernel JIT",
        _BUILD | _SERVE,
        None,
        "the first build or engine start, which compiles its kernels",
        advance=False,
    ),
    Guard(
        "free GPU memory",
        _BUILD | _SERVE,
        None,
        "each build or engine start, on its own cards",
        advance=False,
    ),
)


#: What an engine start (``qda-kl``, ``qda-agent``'s judge and agent jobs, a bare ``vllm serve``) refuses before it
#: starts, and what it decides only as it starts.
SERVE_GUARDS: tuple[Guard[None], ...] = (
    Guard("ports", _SERVE, None, "the engine start, which binds its port", advance=False),
)


def refuse_build(ordinals: Sequence[int]) -> None:
    """``qda build``'s own refusal on `BUILD_GUARDS`, before it loads a weight: `Refused` with the guard's fix.

    Args:
        ordinals: The CUDA ordinals of the build's cards; an empty list (a CPU or meta build) checks nothing.
    """
    if not ordinals:
        return
    if (problem := first_problem(BUILD_GUARDS, BUILD, BuildCards(tuple(ordinals)))) is not None:
        raise Refused(problem.text, problem.fix or "", code=problem.code)


def refuse_serve() -> None:
    """An engine start's own refusal on the serve guards of `BUILD_GUARDS` (the CUDA driver), on the cards this
    process can see: `Refused` with the guard's fix. `output.vllm_env.vllm_environ` calls it first."""
    if (problem := first_problem(BUILD_GUARDS, SERVE, BuildCards())) is not None:
        raise Refused(problem.text, problem.fix or "", code=problem.code)


GLIBC_FLOOR = (2, 31)


def glibc_info() -> Info:
    """The host's glibc for ``--report``, with the vLLM extra's floor when it is below it.

    vLLM 0.33.0 requires llguidance 1.7.x, and the wheel a CPython 3.12 install takes (the ``cp39-abi3`` x86_64
    wheel of 1.7.6) is ``manylinux_2_31``; on an older glibc pip finds no matching wheel and builds the sdist,
    which needs Rust, and a wheel installed elsewhere fails at import with ``GLIBC_2.30 not found``. No qda
    command refuses on it, so it is a fact for the report, not a check.
    """
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")  # "glibc 2.28"
    except (ValueError, OSError):
        value = None
    if value is None:
        return Info("glibc", "no GNU libc version (not a glibc host)")
    try:
        found = tuple(int(part) for part in value.split()[-1].split(".")[:2])
    except (IndexError, ValueError):
        return Info("glibc", value)
    if found >= GLIBC_FLOOR:
        return Info("glibc", value)
    return Info(
        "glibc",
        f"{value}, below the vLLM extra's 2.31 (llguidance 1.7.6's cp39-abi3 x86_64 wheel is manylinux_2_31; "
        "below it pip builds the sdist, which needs Rust)",
    )


def check_vllm() -> Check:
    """The installed vLLM, which the built-in target's kernels are checked on in version `Target.vllm`."""
    wanted = Target().vllm
    try:
        found = md.version("vllm")
    except md.PackageNotFoundError:
        why = 'not installed, so serving is out of scope here (uv pip install -c constraints.txt "./packages/qda-kl[vllm]")'
        return Check("vLLM", True, why, skipped=True)
    if found.split(".")[:2] != wanted.split(".")[:2]:  # no command refuses another version: said, not judged
        return Check(
            "vLLM", True, f'{found} (the kernels are checked on {wanted}: pip install "vllm=={wanted}")'
        )
    return Check("vLLM", True, found)


def tools_info() -> Info:
    """Which build tools an engine finds, for ``--report``: on ``PATH`` with this interpreter's ``bin`` first
    (`qda_quant.output.vllm_env.tool_path`), named but not located."""
    path = tool_path()
    found = [f"{name} {'found' if shutil.which(name, path=path) else 'not found'}" for name in _TOOLS]
    return Info("tools", ", ".join(found))


#: The tools `tools_info` names: the CUDA compiler, the JIT build driver, and the host compilers.
_TOOLS = ("nvcc", "ninja", "cc", "gcc", "c++", "g++")


def check_engine_cache() -> Check:
    """The directory every vLLM engine qda starts compiles into can be made and locked.

    The caches are node-local by default (`qda_quant.output.vllm_env.cache_root`); a user who moved them with
    the environment variables is checked where they moved them. A first compile takes a file lock there, which
    a filesystem without working locks refuses and the engine start then fails on. A default root that cannot
    be made, that another user made first, or that is writable by group or others, is a miss too: the engine
    start refuses it (`qda_quant.output.vllm_env.ensure_cache_root`), and this line reports the same reason and
    fix.
    """
    if any(not os.environ.get(name) for name in CACHE_ENV):
        if (problem := cache_root_problem(cache_root())) is not None:
            return Check("engine cache", False, problem, CACHE_FIX)
    try:
        used = sorted(set(cache_env({}).values()))  # the directories the engines will really compile into
    except Refused as refused:  # the root could not be made, or is another user's
        return Check("engine cache", False, str(refused), CACHE_FIX)
    for path in used:
        if (problem := cache_problem(Path(path))) is not None:
            return Check("engine cache", False, problem, CACHE_FIX)
    named = sorted(name for name in CACHE_ENV if os.environ.get(name))
    where = ", ".join(used) + (f" (set: {', '.join(named)})" if named else "")
    return Check("engine cache", True, f"{where}: lockable")


#: The check takes under five seconds, however slow the compiler it finds is.
_BUDGET_S = 5.0
#: One attempt gets at most half of it, so a compiler that hangs still leaves time to name one that works.
_ATTEMPT_S = 2.5
#: CUDA library discovery gets at most this share, so a hung ``ldconfig`` still leaves the compiler its attempt.
_DISCOVERY_S = 1.0


class _DiscoveryTimeout(Exception):
    """CUDA library discovery outlasted its share of the compiler check's budget."""


class _TimedOut(str):
    """A `_build` failure that is its time limit, not the compiler's error."""


def check_triton_build(
    deadline: float | None = None, include: str | None = None, budget: float | None = None
) -> Check:
    """Triton's first use builds its CUDA driver with a C compiler; this is that compile, in a temp dir.

    The compiler, the flags and the directories are Triton's own (``triton.runtime.build``: ``CC`` when the
    variable is set, even empty, else ``gcc``, else ``clang``; the running Python's headers; the CUDA headers
    and libraries Triton ships). A compiler that cannot find the system C headers — a module or spack build on
    some clusters — fails there deep inside a GPU build; here it fails in half a second, naming itself, the
    first error and the fix. A Python installed without its development headers fails the same compile at
    ``Python.h: No such file or directory``; the fix is then the interpreter's ``-dev``/``-devel`` package,
    not another compiler. A card build runs this before it starts (`qda_quant.engine.build`).

    This is the compile, not the build: a cached native helper may still load without a fresh compile;
    this check verifies a fresh compile. Other uncached launchers can still need headers/compiler.

    A miss that hit a time limit has ``timed_out`` set: it says the machine is slow, not that the compile fails.

    Args:
        deadline: ``time.monotonic()`` by which the attempt ends; default ``budget`` from now.
        include: the Python include dir the compiler gets; default this interpreter's (`_python_include`),
            as Triton passes it. A test passes a directory without the headers.
        budget: the seconds the check takes at most, default `_BUDGET_S`; one compile gets half, CUDA library
            discovery a fifth.
    """
    budget = _BUDGET_S if budget is None else budget
    deadline = time.monotonic() + budget if deadline is None else deadline
    scale = budget / _BUDGET_S
    cc = _cc()
    if cc is None:
        missing = "not found: Triton's first use builds its CUDA driver with it"
        return Check("C compiler", False, missing, "install gcc")
    try:
        driver = _triton_driver(deadline, include, _DISCOVERY_S * scale)
    except _DiscoveryTimeout as why:
        return Check(
            "C compiler",
            False,
            f"{cc}: {why}; the compiler was not tried",
            "make /sbin/ldconfig -p finish (it lists libcuda.so.1), or set LD_LIBRARY_PATH to the CUDA "
            "library directory",
            timed_out=True,
        )
    if driver is None:
        return Check("C compiler", True, f"{cc}: no Triton CUDA driver to build here", skipped=True)
    used = cc or "CC=''"
    with tempfile.TemporaryDirectory() as tmp:
        failure = _build(cc, driver, Path(tmp) / "driver.so", deadline, _ATTEMPT_S * scale)
        if failure is None:
            return Check("C compiler", True, cc)
        working = _working_compiler(driver, cc, Path(tmp), deadline, _ATTEMPT_S * scale)
        if working:
            fix = f"CC={working}"
        elif _python_headers_missing(failure):
            fix = _python_headers_fix()
        else:
            fix = "install a gcc that can build Triton's driver"
        detail = f"{used} cannot build Triton's CUDA driver: {failure}"
        return Check("C compiler", False, detail, fix, timed_out=isinstance(failure, _TimedOut))


#: A driver check that timed out runs once more with this budget: a compiler that fails slowly is still refused
#: before the model load, and one slowed by a loaded node or a network filesystem passes.
_RETRY_BUDGET_S = 30.0


def triton_build_verdict(log: Callable[[str], None]) -> Check:
    """`check_triton_build`, run once more with `_RETRY_BUDGET_S` when it timed out.

    A second timeout is no verdict: one line to ``log``, and a skip. The engine then compiles the driver with
    no time limit.
    """
    check = check_triton_build()
    if check.timed_out:
        check = check_triton_build(budget=_RETRY_BUDGET_S)
    if check.timed_out:
        log(
            f"preflight: Triton's CUDA driver check did not finish in {_RETRY_BUDGET_S:g} s ({check.detail}); "
            "starting anyway"
        )
        return Check(check.name, True, check.detail, skipped=True, timed_out=True)
    return check


def triton_build_refusal(check: Check, note: str = "") -> Refused | None:
    """`Refused` for a `check_triton_build` miss, with ``note`` after its detail and its fix as `Refused.fix`;
    None for a pass or a skip."""
    if check.ok or check.skipped:
        return None
    return Refused(f"{check.name}: {check.detail}{note}", fix=check.fix, code=check.code)


def refuse_engine_compiler(log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> None:
    """An engine start's refusal when Triton's CUDA driver is not cached and does not compile here.

    Triton compiles its driver (``cuda_utils``) on first use, into ``TRITON_CACHE_DIR``; that compile is the only
    one Triton needs a C compiler and the Python headers for. Without them vLLM stops only after it loads the
    weights (112 s for Qwen3-0.6B on a GH200, minutes for a large model). This runs the build's compile
    (`check_triton_build`, about a second) first, in the engine's environment
    (`output.vllm_env.vllm_environ`), where the engine has a card and the driver is not in its cache: a cached
    driver loads with neither, so it is not checked. A check that times out twice is no verdict
    (`triton_build_verdict`): the engine compiles without a time limit.

    Raises:
        Refused: the compiler is missing or cannot build the driver; its ``fix`` names the change.
    """
    if torch_cuda() is None or _no_cards_to_judge(BuildCards()) is not None or _triton_driver_cached():
        return
    check = triton_build_verdict(log)
    if (refused := triton_build_refusal(check)) is not None:
        raise refused


def _triton_driver_cached() -> bool:
    """Whether Triton's driver is already built in ``TRITON_CACHE_DIR``, by Triton's own cache key.

    The key is Triton's private (``triton.runtime.build._get_cache_manager`` over the driver's source and build
    config, as ``triton.backends.nvidia.driver.CudaUtils`` passes them); a Triton that has changed it reads as not
    cached, so the compile is checked.
    """
    try:
        from triton.backends.nvidia import driver
        from triton.runtime import build

        source = Path(driver.dirname, "driver.c").read_bytes()
        config = {
            "language": ["c"],
            "library_dirs": driver.library_dirs(),
            "include_dirs": driver.include_dirs,
            "libraries": driver.libraries,
            "ccflags": None,
        }
        name = "cuda_utils" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
        return build._get_cache_manager(source, config).get_file(name) is not None
    except Exception:  # noqa: BLE001 (any Triton without this layout: check the compile)
        return False


def check_compiler() -> Check:
    """Triton's driver build (:func:`check_triton_build`), then nvcc's host compiler when nvcc is installed.

    The same broken compiler that fails Triton's driver is nvcc's default host compiler, which DeepGEMM's
    bundled JIT (vLLM's block-scaled FP8 experts) runs without ``-ccbin``; when ``nvcc`` is installed, the
    engine's own selection is run within this check's budget (:func:`_host_compiler_verdict`), so a pass means
    the engine would start. The whole check stays under five seconds (`_BUDGET_S`): a compiler that outlasts
    its attempt is reported as timed out, with how long it had, CUDA library discovery that outlasts its
    smaller share (`_DISCOVERY_S`) is named as the timeout it is, not as a compiler that was never tried, and
    an nvcc probe the budget ends before is a skip, never a pass.
    """
    deadline = time.monotonic() + _BUDGET_S
    verdict = check_triton_build(deadline)
    if not verdict.ok:
        return verdict
    if (cxx := check_flashinfer_cxx()) is not None:
        return cxx
    if _nvcc(os.environ) is None:
        return verdict
    cc = _cc() or ""
    host = _host_compiler_verdict(cc, deadline)
    return host if host is not None else Check("C compiler", True, cc)


def check_flashinfer_cxx() -> Check | None:
    """MISS when FlashInfer must JIT-compile and its C++ compiler is not on ``PATH``; None otherwise.

    FlashInfer compiles its kernels on an engine's first start with ``CXX``, else ``c++``
    (``flashinfer.jit.cpp_ext``), unless its prebuilt kernels are installed (`flashinfer_kernels_problem`). A machine with only ``g++-13`` has no ``c++``: ``vllm serve`` and every
    reading of a search then stop there, after the Triton build above passed.
    """
    if flashinfer_kernels_problem() is None:
        return None
    cxx = os.environ.get("CXX") or "c++"
    if shutil.which(cxx):
        return None
    found = shutil.which("g++") or _versioned_gxx()
    fix = f"CXX={found}" if found else "install g++, or set CXX to a C++ compiler"
    return Check(
        "C compiler",
        False,
        f"{cxx} not found: FlashInfer compiles its kernels with it when an engine starts, so vllm serve and a "
        "search's readings stop there",
        f"{fix}; or run `{DOWNLOAD_KERNELS}` once for prebuilt kernels",
    )


def check_flashinfer_nvcc() -> Check | None:
    """MISS when FlashInfer must JIT-compile and finds no ``nvcc``; None otherwise.

    ``vllm serve`` uses FlashInfer's top-k/top-p sampler, which FlashInfer builds with nvcc on the engine's first
    start unless its prebuilt kernels are installed (`flashinfer_kernels_problem`). qda's own engines start without
    it (`NO_JIT_ENV`), so a search and ``qda-kl`` run; the ``vllm serve`` line ``qda build`` prints stops with
    ``Could not find nvcc``, unless this shell sets ``VLLM_USE_FLASHINFER_SAMPLER=0`` too.
    """
    if (missing := flashinfer_kernels_problem()) is None:
        return None
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0":
        return None
    if _nvcc(os.environ) is not None:
        return None
    return Check(
        "FlashInfer kernels",
        False,
        f"no nvcc and {missing}: vllm serve stops at start building FlashInfer's sampler "
        "(qda's own engines do not need it)",
        f"run `{DOWNLOAD_KERNELS}` once (9.7 GB; 1-6 min), or put a CUDA toolkit's nvcc on PATH",
    )


def _versioned_gxx() -> str | None:
    """The newest ``g++-<n>`` on ``PATH`` (a distribution may install only the versioned name)."""
    found: dict[int, str] = {}
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for f in Path(d).glob("g++-[0-9]*") if d and Path(d).is_dir() else ():
            if f.name[4:].isdigit() and os.access(f, os.X_OK):
                found.setdefault(int(f.name[4:]), str(f))
    return found[max(found)] if found else None


def compiler_env() -> dict[str, str]:
    """`CC` and `CXX` as this shell's `C compiler` line found them, for a job that must build the same things.

    A job of a search gets a `PATH` of the run's own (`judge.startup.qda_resolution`), not this shell's: a
    compiler doctor found only because this shell's `PATH` names its directory is not on the job's, and every
    build of the search then refuses on Triton's compiler check -- after doctor said the machine was fine. The
    absolute programs are passed instead, so what doctor probed is what the job builds with. A variable the
    caller already set is kept.
    """
    found: dict[str, str] = {}
    if (cc := _cc()) is not None and cc:
        found["CC"] = cc
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if cxx:
        found["CXX"] = cxx
    return found


def _cc() -> str | None:
    """Triton's C compiler: ``CC`` when set (even empty), else ``gcc``, else ``clang``, as Triton reads it."""
    return os.environ["CC"] if "CC" in os.environ else shutil.which("gcc") or shutil.which("clang")


def _python_headers_missing(failure: str) -> bool:
    """Whether a driver build failed on a Python header, rather than on the compiler or the C headers."""
    return "Python.h" in failure or "pyconfig.h" in failure


def _python_headers_fix() -> str:
    """The package that ships this interpreter's headers, per platform.

    Debian and Ubuntu ship them as ``python3.12-dev`` (the interpreter's own version), the RHEL family as
    ``python3.12-devel``, and a conda environment keeps them in its ``python`` package. Without root, an
    interpreter uv downloads (``--managed-python``) ships its headers, so a new venv from it needs no package.
    """
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"install Python {version}'s development headers: sudo apt install python{version}-dev "
        f"(Debian/Ubuntu), sudo dnf install python{version}-devel (RHEL/Rocky/Alma), or conda install "
        f"python={version} (a conda Python includes them); without root, make the venv from a uv-managed "
        f"Python, which includes them: uv venv -p {version} --managed-python"
    )


def _triton_driver(
    deadline: float, include: str | None = None, discovery_s: float | None = None
) -> tuple[Path, list[str]] | None:
    """Triton's CUDA driver source and the flags Triton's compile adds, or None where it has neither.

    Triton builds ``backends/nvidia/driver.c`` against its own CUDA headers, the Python headers and the
    directories ``ldconfig`` lists ``libcuda.so.1`` in (``triton.runtime.build._build`` and the CUDA
    backend's ``driver.py``). The same flags go here, so a compile that fails here would fail there.
    """
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.submodule_search_locations:
        return None
    backend = Path(next(iter(spec.submodule_search_locations)), "backends", "nvidia")
    src = backend / "driver.c"
    if not src.is_file():
        return None
    python_include = _python_include() if include is None else include
    flags = [f"-I{backend / 'include'}", f"-I{backend}", f"-I{python_include}"]
    flags += [f"-L{d}" for d in [backend / "lib", *_libcuda_dirs(deadline, discovery_s)]]
    return src, [*flags, "-l:libcuda.so.1"]


def _python_include() -> str:
    """The include dir Triton passes for Python headers; Debian's ``posix_local`` scheme reads as ``posix_prefix``."""
    scheme = sysconfig.get_default_scheme()
    return sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]


def _libcuda_dirs(deadline: float, discovery_s: float | None = None) -> list[str]:
    """The directories that hold ``libcuda.so.1``: ``TRITON_LIBCUDA_PATH`` alone when set (as Triton reads it),
    else ``LD_LIBRARY_PATH``'s first, then ``ldconfig``'s.

    A ``LD_LIBRARY_PATH`` directory that holds the library already answers the question, so ``ldconfig``
    is not run: a hung one cannot block a directory the caller named.

    Raises:
        _DiscoveryTimeout: ``ldconfig`` did not finish within ``discovery_s`` (default `_DISCOVERY_S`), or within
            the budget left.
    """
    if triton := os.environ.get("TRITON_LIBCUDA_PATH"):
        return [triton]
    named = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    if any((Path(d) / "libcuda.so.1").exists() for d in named):
        return named
    remaining = min(_DISCOVERY_S if discovery_s is None else discovery_s, deadline - time.monotonic())
    if remaining <= 0:
        raise _DiscoveryTimeout("the check's budget was spent before CUDA library discovery")
    try:
        listing = subprocess.run(
            ["/sbin/ldconfig", "-p"], capture_output=True, text=True, timeout=remaining
        ).stdout
    except subprocess.TimeoutExpired:
        raise _DiscoveryTimeout(
            f"CUDA library discovery (/sbin/ldconfig -p) timed out after {remaining:.1f} s"
        ) from None
    except OSError:
        listing = ""
    found = [str(Path(line.split()[-1]).parent) for line in listing.splitlines() if "libcuda.so.1" in line]
    return found + named


def _build(
    cc: str, driver: tuple[Path, list[str]], out: Path, deadline: float, attempt_s: float | None = None
) -> str | None:
    """Compile ``driver`` with ``cc`` into ``out``: None when it builds, else the first line naming the error.

    The attempt gets the smaller of ``attempt_s`` (default `_ATTEMPT_S`) and the check's remaining budget; a compiler that does not
    finish in that time fails with the time it had (a `_TimedOut`), and the rest of the budget can name one that
    works.
    """
    src, flags = driver
    timeout = min(_ATTEMPT_S if attempt_s is None else attempt_s, deadline - time.monotonic())
    if timeout <= 0:
        return _TimedOut("timed out: the check's budget is spent")
    argv = [cc, str(src), "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", str(out), *flags]
    started = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _TimedOut(f"timed out after {time.monotonic() - started:.1f} s")
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    if p.returncode == 0:
        return None
    lines = [line.strip() for line in p.stderr.splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower() and "collect2" not in line]
    return (errors or lines or [f"exit status {p.returncode}"])[0]


def _working_compiler(
    driver: tuple[Path, list[str]], failed: str, tmp: Path, deadline: float, attempt_s: float | None = None
) -> str | None:
    """The first other ``gcc``, ``clang`` or ``cc`` on ``PATH`` that builds the driver, or None.

    A handful of retries: each one is a full compile, and a PATH has a few compilers at most. All of them
    share the check's remaining budget (``_build`` caps each attempt).
    """
    broken = Path(failed).resolve() if failed else None
    others = [c for c in _on_path() if broken is None or Path(c).resolve() != broken][:4]
    for i, cc in enumerate(others):
        if _build(cc, driver, tmp / f"driver-{i}.so", deadline, attempt_s) is None:
            return cc
    return None


def _on_path() -> list[str]:
    """Every ``gcc``, ``clang`` and ``cc`` on ``PATH`` once each by real path, in Triton's preference order."""
    found, seen = [], set()
    for name in ("gcc", "clang", "cc"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            tool = Path(d, name)
            if tool.is_file() and os.access(tool, os.X_OK) and tool.resolve() not in seen:
                seen.add(tool.resolve())
                found.append(str(tool))
    return found


def _host_compiler_verdict(cc: str, deadline: float) -> Check | None:
    """The nvcc verdict on the host compiler the engine would give it, or None when there is nothing to report.

    The selection is the engine's own (`qda_quant.output.vllm_env.select_host_compiler`), given this check's
    budget, so a pass means the engine would start and a miss means it would refuse. A probe that outlasts the
    budget is a skip, never a pass: nothing was established about the engine.
    """
    selection = select_host_compiler(os.environ, probe_timeout=_ATTEMPT_S, deadline=deadline)
    if selection.unfinished:
        return Check(
            "C compiler",
            True,
            f"{cc}: nvcc's host compiler was not probed (the check's {_BUDGET_S:g} s budget was spent)",
            skipped=True,
        )
    if selection.problem is None:
        return None
    if selection.explicit:
        return Check(
            "C compiler",
            False,
            f"the host compiler nvcc is given ({selection.explicit}) cannot preprocess a CUDA file: "
            f"{selection.problem}",
            "remove -ccbin/NVCC_CCBIN or point it at a compiler that finds the system headers",
        )
    return Check(
        "C compiler",
        False,
        f"nvcc cannot preprocess a CUDA file with any host compiler the engine would try: {selection.problem}",
        "install a g++ that finds the system headers, or set NVCC_CCBIN to one",
    )


def sglang_cache_info() -> Info | None:
    """Whether the SGLang JIT cache root is a real path, for ``--report``; None when sglang is not installed.

    SGLang records each kernel's source from the build's resolved path and looks it up under the root as
    given; where the home or cache directory is a symlink, no leaf ever hits and every load recompiles. That is
    slow, not refused, so it is a fact for the report. The cache-root mechanism is described in
    ``docs/verify.md`` (Kernel compilation). `SGLANG_JIT_CACHE_DIR` is the only setting.
    """
    if importlib.util.find_spec("sglang") is None:
        return None
    given = os.environ.get("SGLANG_JIT_CACHE_DIR") or "~/.cache/sglang/jit"
    root = Path(given).expanduser().absolute()
    if os.path.realpath(root) == str(root):
        return Info("SGLang JIT cache", "a real path")
    return Info(
        "SGLang JIT cache",
        "a symlinked path: every SGLang load recompiles its kernels (set SGLANG_JIT_CACHE_DIR to its real path)",
    )


@dataclass(frozen=True)
class GdnFastPath:
    """One kernel GDN modeling binds when its package is importable, and the path it falls back to.

    The names mirror the four `use_kernel_func_from_hub_with_fallback` calls in the pinned transformers'
    ``models/qwen3_5/modeling_qwen3_5.py`` (also ``qwen3_5_moe`` and ``qwen3_next``, which spell them the
    same way): the hub function name and the package the decorator imports, with transformers' torch
    reference when that import fails. ``fla`` also ships a Triton causal conv
    (``fla.modules.causal_conv1d``), which `qda_quant.arch.derive_trace.route_gdn` binds under the two conv
    names when no top-level ``causal_conv1d`` package exists; that is the fast conv path here.
    """

    function: str
    fast: str
    package: str
    distribution: str
    pin: str | None


#: The four GDN kernel bindings of transformers' Qwen3.5/3.6/3.8 modeling, in call order.
GDN_FAST_PATHS = (
    GdnFastPath(
        "causal_conv1d_fn", "causal_conv1d_fn from causal_conv1d", "causal_conv1d", "causal-conv1d", None
    ),
    GdnFastPath(
        "causal_conv1d_update",
        "causal_conv1d_update from causal_conv1d",
        "causal_conv1d",
        "causal-conv1d",
        None,
    ),
    GdnFastPath(
        "chunk_gated_delta_rule", "fla chunk_gated_delta_rule", "fla", "flash-linear-attention", "0.6.0"
    ),
    GdnFastPath(
        "fused_recurrent_gated_delta_rule",
        "fla fused_recurrent_gated_delta_rule",
        "fla",
        "flash-linear-attention",
        "0.6.0",
    ),
)


def has_gdn(graph: ModelGraph) -> bool:
    """Whether the graph has GDN / linear-attention groups (the `_slow_gdn` test)."""
    return any(g.kind.startswith("linear_attn") for layer in graph.layers for g in layer.groups)


def has_gdn_config(config: object) -> bool:
    """Whether a model config declares linear-attention layers (Qwen3.5/3.6/3.8, Qwen3-Next).

    Read from the config alone so a launch preflight needs no weights: these families name every layer's
    kind in ``layer_types`` (``linear_attention`` or ``full_attention``), and transformers fills the list
    in when a checkpoint leaves it out.
    """
    text = getattr(config, "text_config", None) or config
    return any(str(t) == "linear_attention" for t in (getattr(text, "layer_types", None) or ()))


#: Steps whose work a card runs; on CPU one such step turns a minutes-long build into an hours-long one.
GPU_STEPS = frozenset(
    {"gptq", "gptaq", "awq", "autoround", "gsq", "scale_kd", "kd", "guided", "fisher_trace"}
)

#: The variable a GPU lease sets for the cards a job was given (a job may hold cards this process cannot see).
GPU_LEASE = "GPU_RUN_LEASE"


def cpu_step_names(devices: object, steps: object) -> list[str]:
    """The card steps a config puts on CPU, sorted; empty when none runs on a card or no device is only CPU.

    `devices` is a config's raw value (as YAML reads it) or already-resolved `torch.device`s; each entry is
    resolved with the build's own `resolve_devices`, so the alternate CPU spellings (`CPU`, `cpu:0`,
    `torch.device("cpu")`) are the same CPU device. `devices` unset, null or empty means the job's own cards,
    so nothing is put on CPU; a list that resolves to a card or a mix is card work too; a list this process
    cannot resolve (a card it cannot see) is left to the build's own `resolve_devices` to report; and a step
    list with no `GPU_STEPS` entry (RTN alone) is CPU-only work that a CPU device serves as it is.
    """
    if devices is None:
        return []
    if isinstance(devices, str):  # a scalar spelling the list form allows
        devices = [devices]
    if not isinstance(devices, (list, tuple)) or not devices:
        return []
    from .engine.planning import resolve_devices

    try:
        # `torch.device` spells only lowercase `cpu`, while a config may write `CPU`; the same device after
        # normalization. Resolution is the build's own, so `cpu:0` is the same CPU device as `cpu`.
        resolved = resolve_devices([d.strip().lower() if isinstance(d, str) else d for d in devices])
    except (Refused, RuntimeError, ValueError):  # a card this process cannot see, or a bad device spelling
        return []
    if not all(d.type == "cpu" for d in resolved):
        return []
    names: list[str] = []
    for step in steps if isinstance(steps, (list, tuple)) else ():
        if isinstance(step, str):
            names.append(step)
        elif isinstance(step, dict):
            names.extend(str(k) for k in step)
        else:
            name = getattr(step, "name", None)  # a resolved `Recipe` step
            if isinstance(name, str):
                names.append(name)
    return sorted({name for name in names if name.lower() in GPU_STEPS})


def _devices_where(path: Path, text: str | None = None) -> str:
    """`path` with the line number of its `devices:` key, or the path alone when it has none."""
    if text is None:
        try:
            text = path.read_text()
        except OSError:
            return str(path)
    line = next(
        (n for n, text_line in enumerate(text.splitlines(), 1) if re.match(r"\s*devices\s*:", text_line)),
        None,
    )
    return f"{path}:{line}" if line is not None else str(path)


def cpu_step_config(path: Path) -> str | None:
    """Why a staged build config puts card steps on CPU, naming its devices line; None when it does not.

    A file that is not a readable build config is skipped, not refused: a directory an operator stages can
    hold anything. PyYAML's composer is recursive, so a pathologically nested document raises
    `RecursionError` rather than `yaml.YAMLError`; it is skipped like any other unreadable file.
    """
    import yaml

    try:
        text = path.read_text()
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    if not (bad := cpu_step_names(data.get("devices"), data.get("steps"))):
        return None
    return f"{_devices_where(path, text)} devices: cpu puts {', '.join(bad)} on CPU"


def job_has_cards() -> bool:
    """Whether the job this process runs in holds cards: a GPU lease, or CUDA's own visible count."""
    if os.environ.get(GPU_LEASE):
        return True
    import torch

    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def cpu_step_refusal(devices: object, steps: object, config: str | Path | None = None) -> str | None:
    """Why a build must refuse a config that puts card steps on CPU while its job holds cards; None to build.

    The message is the launch check's (`cpu_step_config`) plus what to do. It is read before any weight is
    loaded: a refused build writes no `build.json` and mints no build receipt, as the GDN fast-kernel refusal
    does (`engine.build._refuse_fast_gdn`), and the job log's `error:` line is what the ledger records.
    A job without cards keeps today's CPU builds, and so does a CPU-only step list (RTN alone).
    """
    if not (bad := cpu_step_names(devices, steps)):
        return None
    if not job_has_cards():
        return None
    where = _devices_where(Path(config)) if config is not None else ""
    return (
        f"{where + ' ' if where else ''}devices: cpu puts {', '.join(bad)} on CPU "
        "(the run has cards: remove `devices` or name a card)"
    )


def _fla_available() -> bool:
    from transformers.utils.import_utils import is_flash_linear_attention_available

    return is_flash_linear_attention_available()


def _causal_conv1d_available() -> bool:
    from transformers.utils.import_utils import is_causal_conv1d_available

    return is_causal_conv1d_available()


def _fla_conv_available() -> bool:
    """Whether fla's own Triton causal conv can be bound (the adapter `route_gdn` installs)."""
    from .arch.derive_trace import fla_causal_conv

    return fla_causal_conv() is not None


#: The pinned transformers modules whose GDN modeling binds the four kernels.
GDN_MODELING_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)

#: A `GdnFastPath.function` and the attribute the modeling module binds it to.
BOUND_NAMES = {
    "causal_conv1d_fn": "causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d_update",
    "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
}


def _modeling_modules() -> list[object]:
    """The pinned transformers modules that bind the GDN kernels, imported on demand."""
    import importlib

    modules = []
    for name in GDN_MODELING_MODULES:
        try:
            modules.append(importlib.import_module(name))
        except Exception:  # a transformers without this family, or an import that cannot run here
            continue
    return modules


def _bound_flag(fn: object) -> bool | None:
    """What transformers' fallback decorator bound: True for the package implementation, False for its torch
    reference; None when `fn` is not that wrapper."""
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or not closure:
        return None
    try:
        cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure), strict=True))
    except ValueError:  # an empty cell
        return None
    flag = cells.get("is_new_implementation")
    return None if flag is None else bool(flag)


def _bound_fast(fn: object) -> bool:
    """Whether this modeling-module name takes a fast kernel on a CUDA input."""
    if getattr(fn, "_qda_fast_on_cuda", False):  # the fla conv adapter `route_gdn` installs
        return True
    flag = _bound_flag(fn)
    if flag is not None:  # transformers' wrapper: its package implementation, or its torch reference
        return flag
    fast = getattr(fn, "__wrapped__", None)  # `route_gdn`'s device router: its CUDA side
    return bool(fast is not None and _bound_flag(fast))


def gdn_kernel_paths() -> list[tuple[GdnFastPath, str, bool]]:
    """Every GDN kernel binding, the path it will take, and whether that path is a fast kernel.

    The answer comes from the callables the pinned transformers modeling modules are actually bound to after
    `route_gdn` (the fallback decorator records in its closure whether it bound the package or its torch
    reference, and `route_gdn`'s adapter and device router are recognised from their own shape), not from
    package metadata: a discoverable package whose import fails leaves the torch reference bound. When no
    pinned module is available at all (an older transformers), the availability flags decide.
    """
    from .arch.derive_trace import route_gdn_modules

    modules = _modeling_modules()
    if not modules:  # nothing to inspect: the availability flags are the only evidence
        fla = _fla_available()
        top_conv = _causal_conv1d_available()
        fla_conv = fla and _fla_conv_available()
        out = []
        for path in GDN_FAST_PATHS:
            if path.package == "causal_conv1d":
                if top_conv:
                    out.append((path, path.fast, True))
                elif fla_conv:
                    out.append((path, f"{path.function} from fla.modules.causal_conv1d", True))
                else:
                    out.append((path, "torch reference", False))
            else:
                out.append((path, path.fast, True) if fla else (path, "torch reference", False))
        return out
    route_gdn_modules(modules)  # the same routing a build installs, then read what each name is bound to
    out = []
    for path in GDN_FAST_PATHS:
        bound = [getattr(module, BOUND_NAMES[path.function], None) for module in modules]
        bound = [fn for fn in bound if fn is not None]
        fast = bool(bound) and all(_bound_fast(fn) for fn in bound)
        kernel = path.fast if fast else "torch reference"
        if fast and path.package == "causal_conv1d" and getattr(bound[0], "_qda_fast_on_cuda", False):
            kernel = f"{path.function} from fla.modules.causal_conv1d"
        out.append((path, kernel, fast))
    return out


def gdn_kernel_problem(gdn: bool) -> str | None:
    """The message for a GDN model whose fast kernels would fall back; None when every one binds.

    A GDN model on a CUDA build refuses this message (`engine.build._refuse_fast_gdn`); a non-GDN model
    never gets it, and CPU / meta builds keep the reference path.
    """
    if not gdn:
        return None
    paths = gdn_kernel_paths()
    slow = [(path, kernel) for path, kernel, fast in paths if not fast]
    if not slow:
        return None
    fla_path = next(path for path in GDN_FAST_PATHS if path.package == "fla")
    conv_path = next(path for path in GDN_FAST_PATHS if path.package == "causal_conv1d")
    reasons = {
        "causal_conv1d": f"no {conv_path.distribution} package and fla's Triton causal conv is not bound",
        "fla": f"{fla_path.distribution} {fla_path.pin} is not installed",
    }
    fallbacks = ", ".join(f"{path.function} ({reasons[path.package]})" for path, _kernel in slow)
    fla_slow = any(path.package == "fla" for path, _kernel in slow)
    hint = (
        f"{fla_path.distribution}=={fla_path.pin} (it also binds fla's Triton causal conv)"
        if fla_slow
        else ""
    )
    if any(path.package == "causal_conv1d" for path, _kernel in slow):
        hint = f"{hint} or {conv_path.distribution}" if hint else conv_path.distribution
    report = "; ".join(f"{path.function} -> {kernel}" for path, kernel, _fast in paths)
    return (
        f"GDN layers would fall back to transformers' torch reference kernels: {fallbacks}. "
        f"Install {hint}. Kernel paths: {report}."
    )


def gdn_fallback(graph: ModelGraph) -> str | None:
    """`gdn_kernel_problem` for a derived model graph (the build's and the plan's test)."""
    return gdn_kernel_problem(has_gdn(graph))


def _serving() -> bool:
    """Whether this install can serve (vLLM is installed), which puts the engine's guards in doctor's scope."""
    return importlib.util.find_spec("vllm") is not None


def environment(ctx: Context) -> list[Check] | list[Info]:
    """The machine's build and serve prerequisites, in reading order; with ``--report``, its environment.

    The scope is what this install can run: ``qda build`` always, an engine start when vLLM is installed. The
    build's own guards (`BUILD_GUARDS`) and the engine's (`SERVE_GUARDS`) are walked as the commands walk them;
    with ``--search`` the run's cards are the search's own ``card health`` line, so the build's is left out, and
    the ``FlashInfer kernels`` line too: a search's engines start without FlashInfer's JIT (`NO_JIT_ENV`).
    """
    if ctx.report:
        return environment_report()
    commands = {BUILD, SERVE} if _serving() else {BUILD}
    build = [g for g in BUILD_GUARDS if not (ctx.search is not None and g.name == "card health")]
    later = all_problems(build, commands, BuildCards())
    lines = [check_gpu(), *guard_lines(v for v in later if v.advance)]
    lines += [
        check_vllm(),
        check_engine_cache(),
        check_compiler(),
    ]
    if _serving() and ctx.search is None and (kernels := check_flashinfer_nvcc()) is not None:
        lines.append(kernels)
    serve = all_problems(SERVE_GUARDS, commands, None)
    lines += guard_lines(v for v in serve if v.advance)
    lines += guard_lines(v for v in later if not v.advance)
    lines += guard_lines(v for v in serve if not v.advance)
    return lines


def _version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def _torch_cuda() -> str:
    """The CUDA version the installed torch was built for, read from ``torch/version.py`` (torch not imported)."""
    spec = importlib.util.find_spec("torch")
    for root in (spec.submodule_search_locations or []) if spec else []:
        try:
            text = (Path(root) / "version.py").read_text()
        except OSError:
            continue
        found = re.search(r"^cuda\b[^=\n]*=\s*['\"]([\d.]+)['\"]", text, re.M)
        return f"built for CUDA {found.group(1)}" if found else "a CPU build"
    return "not installed"


#: The packages ``--report`` lists, by distribution name.
REPORT_PACKAGES = (
    "qda-quant",
    "qda-kl",
    "qda-agent",
    "torch",
    "vllm",
    "flashinfer-python",
    "triton",
    "transformers",
    "flash-linear-attention",
    "sglang",
    "gguf",
)


def _filesystems() -> Info:
    """The filesystem type each cache and temporary location is on, named by its variable, never by its path."""
    import tempfile as _tempfile

    places = {
        "HF_HOME": os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"),
        "TMPDIR": _tempfile.gettempdir(),
        "engine caches": str(cache_root()),
    }
    found = []
    for name, path in places.items():
        try:
            kind = fstype(Path(path)) or "unknown"
        except OSError:
            kind = "unreadable"
        unset = "" if name in os.environ else " (unset, default)"
        found.append(f"{name}{unset} on {kind}")
    return Info("filesystems", ", ".join(found))


def environment_report() -> list[Info]:
    """``qda doctor --report``: the environment a bug report needs, with no verdict and no path of this machine."""
    import platform

    cards, _faults, unread, error = _smi_cards()
    if error is not None:
        gpu = Info("NVIDIA driver", f"no listing (nvidia-smi: {error})")
    else:
        names = sorted({c[2] for c in cards})
        gpu = Info("NVIDIA driver", f"{cards[0][3]}; {len(cards)} card(s): {', '.join(names)}")
    lines = [
        Info("packages", ", ".join(f"{n} {_version(n)}" for n in REPORT_PACKAGES)),
        Info("python", f"{platform.python_version()} ({platform.python_implementation()})"),
        glibc_info(),
        Info("system", f"{platform.system()} {platform.release()} {platform.machine()}"),
        gpu,
        Info("torch CUDA", _torch_cuda()),
        Info(VISIBLE_ENV, os.environ.get(VISIBLE_ENV, "unset")),
        tools_info(),
        _filesystems(),
    ]
    lines += [Info("nvidia-smi", line) for line in unread]
    if (sglang := sglang_cache_info()) is not None:
        lines.append(sglang)
    return lines
