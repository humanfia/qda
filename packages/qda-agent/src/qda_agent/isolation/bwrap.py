"""The bwrap argv builders, hidden paths, pid/proc forms and device nodes one sandbox is made from.

This is the standard library only, and it stays that way: the late binder runs this very file under
`python -I` (`private_bwrap_argv`), and the hidden-path guard runs its source under `python -I -S -c`.

Visible read-only: the system (`/usr`, `/bin`, `/lib*`, `/etc`, `/sys`), the Python environment, the
job's directory (its script snapshot) and the configured paths (the original model, ...). Writable:
the work directory only. Not visible: the run directory's `judge/` (frozen copies, evaluator, ledger),
the reference data, `$HOME`, every other path. No network (`--unshare-net`), its own pid namespace,
killed with its parent. GPU device nodes: the control nodes and the `nvidiaN` of the job's cards.

Card numbers (`cards`, a lease command's `CUDA_VISIBLE_DEVICES`) are PCI-bus-order ordinals
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`); `resolve_cards` maps each to its `/dev/nvidia<minor>`
and UUID through the driver's `/proc` tables, since minors need not follow bus order. In a job confined to
part of the node (Slurm's device cgroup, a container) the same function reads nvidia-smi's job-local numbering
instead, so `cards: [0]` names the GPU that job was given. Inside the sandbox `CUDA_VISIBLE_DEVICES` holds the
UUIDs: with only some nodes bound, CUDA renumbers the cards from 0.

The private-argument wrapper runs this stdlib-only file under `python -I` with --bwrap-file.

`bwrap_allowlist_argv` is the other shape: a fresh tmpfs at `/` and only the mounts it names, for untrusted
code that must not see the host at all (the LiveCodeBench grader).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

_log = logging.getLogger(__name__)

#: System paths every job sees read-only.
SYSTEM = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/sys")
#: The NVIDIA control nodes a GPU job needs besides its cards' own `nvidiaN`.
NV_CONTROL = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")
#: The driver's table of GPUs (bus id, device minor, UUID of each).
NV_PROC = Path("/proc/driver/nvidia/gpus")
#: A job's `PATH` after the interpreter's `bin/`.
PATH = "/usr/local/bin:/usr/bin:/bin"
#: The first bubblewrap with `--bind-fd`/`--ro-bind-fd`, which every QDA sandbox mounts by: 0.10.0, the
#: CVE-2024-42472 fix that binds the source the descriptor was checked at. `--size` needs 0.7.0 and
#: `--json-status-fd`/`--args` are older still, so this is the one floor a job sandbox has.
BIND_FD_VERSION = (0, 10, 0)
#: That floor as it is written: both the diagnosis and the fix are rendered from it, never retyped.
BIND_FD = ".".join(map(str, BIND_FD_VERSION))
#: The one fix a missing or too old bubblewrap gets: install a current one, or make the unisolated choice.
BWRAP_FIX = f"install bubblewrap >= {BIND_FD}, or set isolation.level: none (jobs run unisolated)"


class BwrapTooOld(ValueError):
    """The bwrap a sandbox would run cannot bind a source by descriptor (`--bind-fd`/`--ro-bind-fd`).

    `why` is the diagnosis and `fix` the action that leaves the sandbox working; `str()` is both on one line,
    so the doctor's sandbox and gpu sandbox lines and a run's own refusal say the same sentence. A sandbox
    other than a search's job sandbox (the LiveCodeBench grader's) names itself and its own way out, because
    `scheduler.job_sandbox` is a search key that command does not read.
    """

    def __init__(self, found: str, *, sandbox: str = "the job sandbox", fix: str = BWRAP_FIX) -> None:
        self.found = found
        self.why = (
            f"bubblewrap {found} is too old for {sandbox}: it needs >= {BIND_FD} for --bind-fd/--ro-bind-fd"
        )
        self.fix = fix
        super().__init__(f"{self.why}; {self.fix}")


#: What `bwrap --version` and `bwrap --help` answered, by executable: the binary does not change under a run,
#: and every job asks. A distro may backport the option (Ubuntu's 0.9.0-1ubuntu0.1 has it), so support is
#: asked of the binary's own help, never inferred from the version number.
_versions: dict[str, tuple[int, ...] | None] = {}
_supports: dict[str, bool | None] = {}


def bwrap_version(bwrap: str = "bwrap") -> tuple[int, ...] | None:
    """`(0, 10, 0)` for `bubblewrap 0.10.0`; None when `bwrap --version` gives no version to read."""
    if bwrap not in _versions:
        try:
            got = subprocess.run([bwrap, "--version"], capture_output=True, text=True, timeout=10)
            first = next((w for w in got.stdout.split() if w[:1].isdigit()), "")
            _versions[bwrap] = tuple(int(n) for n in first.split(".") if n.isdigit()) or None
        except (OSError, subprocess.SubprocessError, ValueError):
            _versions[bwrap] = None
    return _versions[bwrap]


def supports_bind_fd(bwrap: str = "bwrap") -> bool | None:
    """Whether `bwrap --help` lists `--bind-fd` and `--ro-bind-fd`; None when its help cannot be read."""
    if bwrap not in _supports:
        try:
            got = subprocess.run([bwrap, "--help"], capture_output=True, text=True, timeout=10)
            text = got.stdout + got.stderr
            _supports[bwrap] = "--bind-fd" in text and "--ro-bind-fd" in text
        except (OSError, subprocess.SubprocessError, ValueError):
            _supports[bwrap] = None
    return _supports[bwrap]


def require_bind_fd(bwrap: str = "bwrap") -> None:
    """Refuse a job sandbox whose bwrap cannot bind a source by descriptor.

    Asked of the binary (`supports_bind_fd`), because a distribution may backport the option into an older
    bubblewrap; the version is read only to name what the user has.

    Raises:
        BwrapTooOld: `bwrap --help` has no `--bind-fd`/`--ro-bind-fd`.
    """
    if supports_bind_fd(bwrap) is False:
        version = bwrap_version(bwrap)
        raise BwrapTooOld(".".join(map(str, version)) if version else "(version unknown)")


def python() -> str:
    """The running interpreter as an absolute path, spelled under its prefix's real path.

    The last name is kept as it is, not resolved: a venv's `python` is a symlink to the base interpreter,
    and following it would drop the venv's packages. Everything above it is resolved, because
    `python_roots()` mounts the prefix under its real path, and a venv reached through a link (`venvs/dev ->
    venvs/gate-a351929c2bc9`) would otherwise exec a path the sandbox does not hold ("No such file").
    """
    path = Path(sys.executable).absolute()
    return str(Path(os.path.realpath(path.parent)) / path.name)


@dataclass(frozen=True)
class Interpreter:
    """One interpreter's prefix (its environment) and the base installation it was built on.

    Attributes:
        prefix: The environment the interpreter runs in (a venv's directory), as spelled by the caller.
        roots: Every directory the interpreter needs, each as spelled and under its real path: the prefix,
            the base prefix its `pyvenv.cfg` names, and the running interpreter's `sys.base_prefix`.
        bases: The entries of `roots` that are the base installation, not the environment.
    """

    prefix: Path
    roots: tuple[Path, ...]
    bases: frozenset[Path]


def _pyvenv_cfg(cfg: Path) -> dict[str, str]:
    """A venv's `pyvenv.cfg` as a lower-case key/value map (empty: no file, or unreadable)."""
    out: dict[str, str] = {}
    with contextlib.suppress(OSError, UnicodeDecodeError):
        for line in cfg.read_text(errors="ignore").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                out[key.strip().lower()] = value.strip()
    return out


def interpreter(
    python: str | Path, *, prefix: str | Path | None = None, base_prefix: str | Path | None = None
) -> Interpreter:
    """Where one Python interpreter lives: its environment and the base installation it was built on.

    The environment is the prefix above the interpreter's `bin/` (`sys.prefix` when it is the running
    interpreter, passed as `prefix`). Its base installation is what its `pyvenv.cfg` names: the parent of
    its `home` line (and of virtualenv's `base-prefix`), which is where uv, pyenv and conda keep the Python
    their venvs run on -- often under the user's home. The running interpreter's `sys.base_prefix` is
    passed in as well, because another interpreter's is not available without running it; it need not be
    the same spelling as `home`, so both are kept.

    Every directory is kept as spelled and under its real path: `home` may name a link (a home directory
    that is a link to another filesystem), and the venv's interpreter reads its standard library under the
    path `pyvenv.cfg` spells.
    """
    path = Path(python).absolute()
    spelled = Path(os.path.normpath(Path(prefix).absolute() if prefix is not None else path.parent.parent))
    roots: list[Path] = []
    bases: set[Path] = set()

    def add(directory: Path, *, base: bool = False) -> None:
        for p in (directory, Path(os.path.realpath(directory))):
            if p not in roots:
                roots.append(p)
            if base:
                bases.add(p)

    add(spelled)
    add(Path(os.path.realpath(path.parent)).parent)  # the prefix under its real path
    cfg = _pyvenv_cfg(spelled / "pyvenv.cfg") or _pyvenv_cfg(Path(os.path.realpath(spelled)) / "pyvenv.cfg")
    named: list[Path] = []
    if (home := cfg.get("home")) and Path(home).is_absolute():
        named.append(Path(home).parent)
    named += [
        Path(cfg[key]) for key in ("base-prefix", "base-exec-prefix") if cfg.get(key, "").startswith("/")
    ]
    if base_prefix is not None:
        named.append(Path(base_prefix).absolute())
    for directory in named:
        add(directory, base=True)
    return Interpreter(spelled, tuple(roots), frozenset(bases))


def interpreter_roots(
    python: str | Path, *, prefix: str | Path | None = None, base_prefix: str | Path | None = None
) -> list[Path]:
    """The directories the interpreter at `python` needs to run: its environment and its base installation.

    See `interpreter`: each is returned as spelled and under its real path. `python_roots` (what a sandbox
    mounts) and the judge's `resolve_qda` (what a `qda` command's installation needs) both start here.
    """
    return list(interpreter(python, prefix=prefix, base_prefix=base_prefix).roots)


def python_roots() -> list[Path]:
    """What the running interpreter needs to import its packages inside the sandbox.

    Its environment and base installation (`interpreter_roots`), plus what the environment reaches outside
    itself: the targets of symlinks in its `site-packages` and the directories its `.pth` files add
    (editable installs).

    A venv reached through a link (`venvs/dev -> venvs/gate-a351929c2bc9`) is mounted at its real path and
    also at the spelling the caller used: a job command that names the interpreter that way is executed as
    written, and bwrap mounts by path, so the spelling must be there too. A `.pth` directory gets the same
    rule: the interpreter in the sandbox adds the line as written (an editable checkout under a home
    directory that is a link to another filesystem), so that spelling is mounted beside the real path.
    """
    roots = set(interpreter_roots(sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix))
    prefix = Path(sys.prefix).absolute()
    for site in (
        Path(p) for p in sys.path if Path(p).name == "site-packages" and Path(p).is_relative_to(prefix)
    ):
        for entry in site.iterdir() if site.is_dir() else ():
            if entry.is_symlink() and entry.exists():
                roots.add(entry.resolve())
            elif entry.suffix == ".pth" and entry.is_file():
                for line in entry.read_text(errors="ignore").splitlines():
                    d = Path(line.strip())
                    if d.is_absolute() and d.is_dir():
                        roots.add(d.resolve())
                        if d != d.resolve():
                            roots.add(d)
    return sorted(roots)


class CardError(ValueError):
    """A leased card cannot be named inside the sandbox; the job must not start."""


@dataclass(frozen=True)
class Card:
    """One GPU as the driver lists it: `ordinal` is its position in the node's PCI bus order.

    A run's `cards` number is this ordinal on a whole node, or the job-local index (nvidia-smi's, from 0) in a
    job that sees only part of the node; `resolve_cards` is the one place that decides which.
    """

    ordinal: int
    bus: str
    minor: int
    uuid: str

    @property
    def node(self) -> Path:
        """Its device node."""
        return Path("/dev") / f"nvidia{self.minor}"


def cards_by_bus(proc: Path | None = None) -> list[Card]:
    """Every GPU of the driver's `/proc` tables, in ascending PCI bus order (ordinal = position).

    This is the numbering of `CUDA_DEVICE_ORDER=PCI_BUS_ID` (what a lease command sets); a card's device minor
    (`/dev/nvidia<minor>`) need not equal it, and other tools may number cards otherwise (doctor checks
    nvidia-smi's index against it).

    Raises:
        CardError: An entry lacks its bus id, minor or UUID.
    """
    proc = NV_PROC if proc is None else proc
    if not proc.is_dir():
        return []
    rows = []
    for info in proc.glob("*/information"):
        f = dict(
            (k.strip(), v.strip()) for k, _, v in (x.partition(":") for x in info.read_text().splitlines())
        )
        bus, minor, uuid = f.get("Bus Location", "").lower(), f.get("Device Minor", ""), f.get("GPU UUID", "")
        if not (bus and minor.isdigit() and uuid.startswith("GPU-")):
            raise CardError(f"{info}: no bus id, device minor or GPU UUID")
        rows.append((bus, int(minor), uuid))
    return [Card(i, bus, minor, uuid) for i, (bus, minor, uuid) in enumerate(sorted(rows))]


#: The query naming each GPU this process's nvidia-smi sees, in its own index order.
SMI_INDEX = ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"]


def smi_bus(bus: str) -> str:
    """nvidia-smi's PCI bus id in the driver table's form: its 8-digit domain as 4 digits, lower case."""
    dom, _, rest = bus.strip().lower().partition(":")
    return f"{int(dom, 16):04x}:{rest}"


def smi_cards(proc: Path | None = None) -> list[Card] | None:
    """The GPUs this process's `nvidia-smi` lists, in its index order, matched to the driver's table by PCI bus.

    None when nvidia-smi cannot be run or read, or lists no card the driver's table has. Inside a Slurm or
    container allocation the device cgroup leaves nvidia-smi with only the job's cards while the driver's
    `/proc` table still lists every GPU of the node; that difference is the job boundary `job_visible` reads.
    """
    proc = NV_PROC if proc is None else proc
    try:
        run = subprocess.run(SMI_INDEX, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if run.returncode != 0:
        return None
    by_bus = {c.bus: c for c in cards_by_bus(proc)}
    rows: list[tuple[int, Card]] = []
    for line in run.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or ":" not in parts[1]:
            continue
        if (card := by_bus.get(smi_bus(parts[1]))) is not None:
            rows.append((int(parts[0]), card))
    return [card for _, card in sorted(rows)] or None


#: The query naming each GPU this process's nvidia-smi sees, with its compute capability.
SMI_CAPABILITIES = ["nvidia-smi", "--query-gpu=index,pci.bus_id,compute_cap", "--format=csv,noheader"]


def smi_capabilities(proc: Path | None = None) -> list[tuple[Card, tuple[int, int]]] | None:
    """Each GPU this process's `nvidia-smi` lists, in its index order, with its compute capability.

    The capability is the driver's own (`compute_cap`), so nothing opens a CUDA context: the judge prices a
    build in a CPU process that must not take a context on every card of the machine. None when nvidia-smi
    cannot be run or read, or lists no card the driver's table has (as `smi_cards`).
    """
    proc = NV_PROC if proc is None else proc
    try:
        run = subprocess.run(SMI_CAPABILITIES, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if run.returncode != 0:
        return None
    by_bus = {c.bus: c for c in cards_by_bus(proc)}
    rows: list[tuple[int, Card, tuple[int, int]]] = []
    for line in run.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit() or ":" not in parts[1]:
            continue
        major, _, minor = parts[2].partition(".")
        if not (major.isdigit() and minor.isdigit()):
            continue
        if (card := by_bus.get(smi_bus(parts[1]))) is not None:
            rows.append((int(parts[0]), card, (int(major), int(minor))))
    return [(card, capability) for _, card, capability in sorted(rows, key=lambda row: row[0])] or None


#: The query naming each GPU this process's nvidia-smi sees, with its memory in MiB.
SMI_MEMORY = ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.total", "--format=csv,noheader,nounits"]


def smi_memory(proc: Path | None = None) -> list[tuple[Card, int]] | None:
    """Each GPU this process's `nvidia-smi` lists, in its index order, with its total memory in bytes.

    Read from the driver (`memory.total`), so nothing opens a CUDA context and the numbers belong to the named
    cards, not to whichever devices torch would number first. None when nvidia-smi cannot be run or read, or
    lists no card the driver's table has (as `smi_cards`).
    """
    proc = NV_PROC if proc is None else proc
    try:
        run = subprocess.run(SMI_MEMORY, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if run.returncode != 0:
        return None
    by_bus = {c.bus: c for c in cards_by_bus(proc)}
    rows: list[tuple[int, Card, int]] = []
    for line in run.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit() or ":" not in parts[1] or not parts[2].isdigit():
            continue
        if (card := by_bus.get(smi_bus(parts[1]))) is not None:
            rows.append((int(parts[0]), card, int(parts[2]) * 2**20))
    return [(card, total) for _, card, total in sorted(rows, key=lambda row: row[0])] or None


def job_visible(proc: Path | None = None) -> list[Card] | None:
    """The job's visible cards, when this process sees fewer GPUs than the node's driver table; else None.

    A job boundary (Slurm's device cgroup, a container's device list) leaves `nvidia-smi` with only the job's
    cards, renumbered from 0, while `/proc/driver/nvidia/gpus` still lists every GPU of the node. Only a
    launch that names cards (`CUDA_VISIBLE_DEVICES` set and non-empty) is read this way; without it, `cards`
    keep their PCI-bus-order meaning (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), which `cards` checks rely on.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value or not value.strip():
        return None
    proc = NV_PROC if proc is None else proc
    if not proc.is_dir():
        return None
    table = cards_by_bus(proc)
    seen = smi_cards(proc)
    return seen if seen is not None and 0 < len(seen) < len(table) else None


def resolve_cards(
    cards: Iterable[int], proc: Path | None = None, visible: Iterable[Card] | None = None
) -> list[Card]:
    """The GPUs that card numbers name, within the job's visible set when it is confined.

    Two numberings, chosen by the job boundary: outside one, `cards` are PCI-bus-order ordinals (card `i` is
    the `i`-th GPU of the driver's table sorted by bus id, as `CUDA_DEVICE_ORDER=PCI_BUS_ID` numbers them).
    Inside one (`job_visible`), `cards` are job-local indices, the numbers `nvidia-smi` shows in this job;
    each is mapped through its bus id to the driver's table, so the run gets the GPU the job was given.

    Inside the sandbox only the cards' `/dev/nvidia<minor>` exist, so CUDA renumbers them from 0; the sandbox
    names them by UUID (`CUDA_VISIBLE_DEVICES=<uuid>,...`), which means the same card on both sides.

    Args:
        cards: PCI-bus-order ordinals, or job-local indices in a confined job.
        proc: The driver's `gpus/` directory (default `/proc/driver/nvidia/gpus`).
        visible: The job's visible cards, or None to read them (`job_visible`); a caller that already has the
            job's `nvidia-smi` rows passes them so the answer is one read.

    Returns:
        One `Card` per number, in order; empty when the machine has no NVIDIA driver (no GPU to show).

    Raises:
        CardError: The driver is present but a number names no GPU available to this job.
    """
    proc = NV_PROC if proc is None else proc
    if not proc.is_dir():
        return []
    table = cards_by_bus(proc)
    cards = list(cards)
    hidden = list(visible) if visible is not None else job_visible(proc)
    if hidden is not None and 0 < len(hidden) < len(table):
        missing = [c for c in cards if not 0 <= c < len(hidden)]
        if missing:
            raise CardError(
                f"card(s) {missing}: this job sees {len(hidden)} of the node's {len(table)} GPU(s), numbered "
                f"0..{len(hidden) - 1} by nvidia-smi; name a card this job can see (the job-local index)"
            )
        return [hidden[c] for c in cards]
    missing = [c for c in cards if not 0 <= c < len(table)]
    if missing:
        raise CardError(
            f"card(s) {missing}: the driver lists {len(table)} GPU(s) (numbered in PCI bus order)"
        )
    return [table[c] for c in cards]


def card_uuids(cards: Iterable[int], proc: Path | None = None) -> list[str]:
    """The GPU UUIDs of card numbers (PCI-bus-order ordinals); see `resolve_cards`."""
    return [c.uuid for c in resolve_cards(cards, proc)]


def device_nodes(cards: Iterable[int], proc: Path | None = None, dev: Path = Path("/dev")) -> list[Path]:
    """The NVIDIA nodes a job on card numbers `cards` needs: the control nodes and `nvidia<minor>` of each."""
    nodes = [Path(p) for p in NV_CONTROL] + [dev / f"nvidia{c.minor}" for c in resolve_cards(cards, proc)]
    return [n for n in nodes if n.exists()]


def in_run(path: Path, run_dir: Path, *, allow: Iterable[Path] = ()) -> bool:
    """Whether `path` resolves into `run_dir` outside `allow` (areas the host writes there).

    Follows links: a path outside the run dir that links into it is inside.
    """
    real = Path(os.path.realpath(path))
    root = Path(os.path.realpath(run_dir))
    return real.is_relative_to(root) and not any(
        real.is_relative_to(Path(os.path.realpath(a))) for a in allow
    )


def holds_run(path: Path, run_dir: Path) -> bool:
    """Whether `path` is `run_dir` or holds it: binding it would show the judge's files under it."""
    return Path(os.path.realpath(run_dir)).is_relative_to(Path(os.path.realpath(path)))


@dataclass(frozen=True)
class Pinned:
    """A read-only source opened and checked at the moment it is bound.

    Attributes:
        real: What the descriptor points at (the path that was checked).
        fd: An `O_PATH` descriptor; bwrap binds it by number (`--ro-bind-fd`), so a later swap of the path
            cannot change what the job sees.
    """

    real: Path
    fd: int


class Expanded(list[Path]):
    """`expand_links`' sources, with the (st_dev, st_ino) each checked final file had at check time.

    `pin_sources` reads the identities off the list it is given, so a source replaced between the plan
    check and the bind (a store file swapped for a link to held-out data) is refused instead of bound,
    and a caller has no second argument to forget. A list subclass compares equal to the plain list.
    """

    def __init__(self, paths: Iterable[Path], identities: Mapping[str, tuple[int, int]]) -> None:
        super().__init__(paths)
        self.identities = dict(identities)


def expand_links(readable: Iterable[Path]) -> list[Path]:
    """`readable`, plus every file a Hugging Face snapshot's links name, through all their hops.

    A hub snapshot (`.../models--X/snapshots/<rev>/config.json`) is a directory of relative links into its
    repository's sibling `blobs/` (`../../blobs/<hash>`); binding only the snapshot leaves every target
    absent inside the job. In this cache those blob entries are themselves links into the cache's shared
    content store one level above the repository (`hub/blobs/<2 hex>/<sha256>`), so the job also needs that
    second hop. For a readable path under a repository's `snapshots/`, each link's target is added at the
    spelling the kernel resolves the link to there (`link.parent / readlink`), and a target that is one of
    the repository's own `blobs/` entries and links into the shared store adds the store file at its own
    path too. Only the files this snapshot names are added: never the rest of the repository or of the
    store, so no other model is exposed. A link whose target leaves the `models--X` repository is refused,
    and so is one whose real path leaves it except through the repository's own `blobs/` entry into the
    shared store; that exception needs the store itself to be a real directory beside the repository (not
    a link to somewhere else) and the file it names to be a regular file there, so a planted `hub/blobs`
    or a link inside the store cannot widen the job's reads to data the run holds out. Broken links and
    targets already under a readable path are left out, and a readable path that is not an HF snapshot is
    not walked at all.

    The result is an `Expanded` list: each checked final file's (st_dev, st_ino), as `lstat` saw it here,
    is recorded under both its link's target spelling and its own, so `pin_sources` can refuse a source
    swapped after this check (the check-to-bind window) instead of binding held-out data.

    Raises:
        ValueError: A link under a readable snapshot resolves outside that snapshot's repository, or into
            the shared store other than through that repository's own blob entry.
    """
    roots = [Path(p) for p in readable]
    out = list(dict.fromkeys(roots))
    seen = set(out)
    identities: dict[str, tuple[int, int]] = {}
    absolute = [Path(os.path.abspath(r)) for r in roots]  # noqa: PTH100 (spellings kept for the binds)
    for root in roots:
        repo = _hub_repo(root)
        if repo is None or not root.is_dir():
            continue
        real_repo = Path(os.path.realpath(repo))
        store = repo.parent / "blobs"  # the shared content store
        real_store = Path(os.path.realpath(store))
        for here, dirs, files in os.walk(root, followlinks=False):
            for name in (*dirs, *files):
                link = Path(here) / name
                if not link.is_symlink():
                    continue
                target = Path(os.path.normpath(link.parent / link.readlink()))  # noqa: PTH100
                real = Path(os.path.realpath(target))
                if not target.is_relative_to(repo):
                    raise ValueError(
                        f"{link}: its link target {target} is outside the model repository {repo}; "
                        "refusing to bind it"
                    )
                try:
                    st = os.lstat(real)  # the last hop itself; realpath never leaves a link at its end
                except OSError:
                    st = None
                store_file = (
                    target.parent == repo / "blobs"
                    and not store.is_symlink()  # a planted `hub/blobs -> held-out` names no real store
                    and store.is_dir()
                    and st is not None
                    and stat.S_ISREG(st.st_mode)  # a regular file, not a link to one
                    and real.is_relative_to(real_store)
                )
                if not real.is_relative_to(real_repo) and not store_file:
                    raise ValueError(
                        f"{link}: its link target {target} resolves to {real}, outside the model "
                        f"repository {repo}; refusing to bind it"
                    )
                if st is not None:  # the identity the bind must still find (the swap window)
                    ident = (st.st_dev, st.st_ino)
                    identities[str(target)] = ident
                    identities[str(real)] = ident
                if target not in seen and target.exists():
                    spelled = Path(os.path.abspath(target))  # noqa: PTH100
                    if not any(spelled.is_relative_to(a) for a in absolute):
                        seen.add(target)
                        out.append(target)
                if store_file and real not in seen:  # the store file itself: the job's second hop
                    seen.add(real)
                    out.append(real)
    return Expanded(out, identities)


def _hub_repo(path: Path) -> Path | None:
    """The `.../models--X` repository holding `path` when it lies under that repository's `snapshots/`."""
    parts = path.parts
    for i in range(1, len(parts)):
        if parts[i] == "snapshots" and parts[i - 1].startswith("models--"):
            return Path(*parts[:i])
    return None


def pin_sources(
    readable: Iterable[Path],
    binds: Mapping[Path, Path],
    *,
    expected: Mapping[str, tuple[int, int]] | None = None,
    run_dir: Path | None = None,
    host_written: Iterable[Path] = (),
) -> dict[Path, Pinned]:
    """Open every read-only source of a sandbox now, refusing one that is the run dir's or holds it.

    The config check ran when the run loaded; this runs at the last moment before bwrap starts, and binds
    the opened descriptor rather than looking the path up again. The descriptor is opened first and its own
    target checked (`/proc/self/fd`), so a link swapped before, during or after the check cannot redirect
    the bind: what was checked is exactly what the job reads. A source whose (st_dev, st_ino) was recorded
    by `expand_links` (carried on the `Expanded` list) or in `expected` is compared against the descriptor:
    a store file swapped for held-out data between the plan check and this opening is refused. Sources
    under the run's host-written areas (the overlay, the scheduler's job directories) are allowed; a run
    dir of None (settings with no run behind them) skips only the run-dir check, never the opening.

    Args:
        readable: Paths bound read-only at themselves.
        binds: Read-only binds at another path, `{path inside: host source}`.
        expected: `(st_dev, st_ino)` by path spelling from the plan check; `readable`'s own identities
            (an `expand_links` result) apply as well.
        run_dir: The run's directory, resolved; None: no run to protect.
        host_written: Areas inside `run_dir` that may be a bind source (the overlay).

    Returns:
        A `Pinned` at each bind destination (for `binds`) and each path of `readable`.

    Raises:
        ValueError: A source cannot be opened, is not the file that was checked, resolves into the run
            dir (or a link into it) outside `host_written`, or holds it.
    """
    allowed = [Path(p) for p in host_written]
    known = dict(getattr(readable, "identities", {}) or {})
    known.update(expected or {})
    pinned: dict[Path, Pinned] = {}
    try:
        for path in readable:
            pinned.setdefault(Path(path), _pin(Path(path), run_dir, allowed, known))
        for dst, src in binds.items():
            old = pinned.pop(Path(dst), None)
            if old is not None:  # a readable path that is also a bind destination: one descriptor is enough
                os.close(old.fd)
            pinned[Path(dst)] = _pin(Path(src), run_dir, allowed, known)
    except BaseException:
        for p in pinned.values():
            os.close(p.fd)
        raise
    return pinned


def _pin(
    path: Path, run_dir: Path | None, allowed: list[Path], expected: Mapping[str, tuple[int, int]]
) -> Pinned:
    """Open `path` and check the descriptor's own target (see `pin_sources`)."""
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError as e:
        raise ValueError(f"{path}: cannot open it to bind ({e.strerror}); refusing to bind it") from None
    try:
        want = expected.get(str(path))
        if want is not None:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != tuple(want):
                raise ValueError(
                    f"{path}: it is not the file that was checked (its device or inode changed); "
                    "refusing to bind it"
                )
        real = Path(f"/proc/self/fd/{fd}").readlink()
        if run_dir is not None:
            if in_run(real, run_dir, allow=allowed):
                raise ValueError(
                    f"{path}: resolves into run_dir {run_dir} (or a link into it), which the agent's jobs "
                    "write, after the config was checked; refusing to bind it"
                )
            if holds_run(real, run_dir):
                raise ValueError(
                    f"{path}: holds run_dir {run_dir} after the config was checked; refusing to bind it"
                )
    except BaseException:
        os.close(fd)
        raise
    return Pinned(real, fd)


#: The one hidden-path scan: a mask is mounted after every other mount, but a later mount can still put the
#: host's object back; this stats each hidden path inside a sandbox's own mounts. Both callers build on it:
#: a job's guard (`_HIDDEN_GUARD`) refuses before the job's work starts, and the agent wall's self-check
#: (`_HIDDEN_CHECK`) reports what it found. The scan is shared because the policy must not fork: the wall
#: (`isolation.wall`) imports this module, and a job runs it standalone (`python -I -S -c`), so it is
#: stdlib-only and cannot import the wall.
_HIDDEN_SCAN = """
import json, os, sys
bad = []
for p, dev, ino in json.loads(sys.argv[1]):
    try:
        s = os.stat(p)
    except OSError:
        continue  # absent: hidden
    if (s.st_dev, s.st_ino) == (dev, ino):
        bad.append(p)  # the host's own object, shown by a bind
"""

#: Prefixed to a job's command: refuse before its work starts when a hidden path still shows the host's object.
_HIDDEN_GUARD = (
    _HIDDEN_SCAN
    + """
if bad:
    print(f"sandbox: {bad[0]} is still visible to the job", file=sys.stderr)
    sys.exit(70)
os.execvp(sys.argv[2], sys.argv[2:])
"""
)

#: The wall's self-check: report every hidden destination that still shows the host's object.
_HIDDEN_CHECK = (
    _HIDDEN_SCAN
    + """
print(json.dumps(bad))
"""
)


def hidden_paths(hide: Iterable[Path]) -> list[Path]:
    """Every spelling of each hidden path a job may reach it by.

    The config's spelling, the home's real path, and each fully resolved: a job's mounts may use any of them.
    """
    home = Path.home()
    root = Path(os.path.realpath(home))
    out: list[Path] = []
    for h in hide:
        q = Path(h)
        spelled = root / q.relative_to(home) if q.is_relative_to(home) else q
        for p in (q, spelled, Path(os.path.realpath(q)), Path(os.path.realpath(spelled))):
            if p not in out:
                out.append(p)
    return out


def hidden_specs(masks: Iterable[tuple[Path, Path]]) -> list[tuple[str, int, int]]:
    """Each `(mask destination, host object)` with the host's `(device, inode)`.

    What the guard must not find at the destination inside the job.
    """
    out, seen = [], set()
    for view, host in masks:
        if str(view) in seen:
            continue
        seen.add(str(view))
        try:
            s = Path(host).stat()
        except OSError:
            continue
        out.append((str(view), s.st_dev, s.st_ino))
    return out


def hidden_covers(hide: Iterable[Path], needed: Iterable[Path | str]) -> tuple[Path, Path] | None:
    """The first `(hide, path)` where a hidden path is, or contains, a path the job needs; None otherwise."""
    for h in hide:
        hp = Path(os.path.realpath(h))
        for n in needed:
            np = Path(os.path.realpath(n))
            if np == hp or np.is_relative_to(hp):
                return Path(h), np
    return None


def mask_after(
    mounts: Iterable[tuple[str, str, str | None]], hide: Iterable[Path]
) -> list[tuple[Path, Path]]:
    """`(mask destination, host object)` for each way the job's mounts still show a hidden object.

    A path is shown when the last mount covering it is a bind whose source holds it; an empty tmpfs or no
    mount at all leaves it absent, and a bind of another tree that does not hold it shows nothing. Each
    spelling is checked by itself, and a bind that holds a hidden tree shows it at the bind's own
    destination too -- a prefix mounted by both its real path and a symlinked spelling (a venv, as
    `python_roots` mounts it) needs both masked. `mounts` is `(kind, destination, source)` in mount order,
    as `bwrap_argv` makes them; each returned mask is mounted after every mount, at that destination.
    """

    mounts = list(mounts)

    def covering(p: Path) -> tuple[int, str, Path, str | None] | None:
        last: tuple[int, str, Path, str | None] | None = None
        for i, (kind, dst, src) in enumerate(mounts):
            d = Path(dst)
            if p == d or p.is_relative_to(d):
                last = (i, kind, d, src)
        return last

    def shown(p: Path) -> Path | None:
        last = covering(p)
        if last is None:
            return None
        _, kind, d, src = last
        if kind == "tmpfs" or src is None:
            return None
        inside = Path(src) if p == d else Path(src) / p.relative_to(d)
        return p if os.path.lexists(inside) else None

    entries = [
        (p, Path(os.path.realpath(p)))
        for p in map(Path, dict.fromkeys(os.path.normpath(str(h)) for h in hide))
    ]
    out: dict[Path, Path] = {}
    for p, real in entries:
        # a spelling through a link is masked at its target, which the link reaches too
        hit = shown(p) or (shown(real) if real != p else None)
        if hit is not None:
            out.setdefault(hit, real)
    for i, (kind, dst, src) in enumerate(mounts):
        if kind not in {"bind", "--bind", "--ro-bind"} or src is None:
            continue
        source = Path(os.path.realpath(src))
        for _, real in entries:
            # An absent hide needs no mask, and a mask point bwrap must make under a read-only bind fails.
            if not os.path.lexists(real) or not real.is_relative_to(source):
                continue
            view = Path(dst) / real.relative_to(source)
            last = covering(view)
            # A later mount over this destination decides what shows there; its own pass masks what it shows.
            if last is not None and last[0] == i:
                out.setdefault(view, real)
    return list(out.items())


def preflight(
    readable: Iterable[Path],
    binds: Mapping[Path, Path] | None = None,
    *,
    writable: Iterable[Path] = (),
    run_dir: Path | None = None,
    host_written: Iterable[Path] = (),
    hide: Iterable[Path] = (),
) -> Path | None:
    """Open one weight file through the binds a job gets, refusing when the sandbox cannot read it.

    The plan check (`expand_links`) can pass while the job's own mounts still do not resolve a file: a
    snapshot's link resolves inside the sandbox only when every hop's target is bound. This starts bwrap
    with exactly the read-only sources a job gets (the expanded `readable` and `binds`) and reads the first
    bytes of the largest file the jobs read -- a model snapshot's weights -- and of one file of every
    configured bind, at the path a job sees it (`baselines/<N>/...`). A failure names the file and bwrap's
    error, so the run is refused before the agent spends tools on jobs that cannot start.

    Args:
        readable: The job's read-only sources, as `bwrap_argv` gets them.
        binds: The job's read-only binds, `{path inside: host source}`.
        writable: The job's writable root (its workdir), so a bind under it is mounted as it is in a job.
        run_dir: The run's directory, resolved; None: no run to protect.
        host_written: Areas inside `run_dir` that may be a bind source.
        hide: The job's `sandbox.hide`; a hide covering a readable source or a bind refuses the run.

    Returns:
        The file checked in `readable`, or None when no readable source holds one.

    Raises:
        ValueError: `expand_links` or `pin_sources` refuses a source, or the sandbox cannot open a file.
    """
    require_bind_fd()  # the probe binds every source by its descriptor, as a job does
    roots = [Path(p) for p in readable]
    expanded = expand_links(roots)
    file = _largest_file(roots, expanded)
    binds = dict(binds or {})
    bound = _bound_files(binds)
    if file is None and not bound:
        return None
    pinned = pin_sources(expanded, binds, run_dir=run_dir, host_written=host_written)
    paths = ([file] if file is not None else []) + bound
    try:
        head = shutil.which("head") or "/usr/bin/head"
        argv = bwrap_argv(
            [head, "-c", "8", *map(str, paths)],
            str(paths[0].parent),
            writable=list(writable),
            readable=expanded,
            pinned=pinned,
            binds=binds,
            env=sandbox_env(Path("/tmp")),
            pid="private",  # no cards: nothing of the driver's to give a /proc
            hide=hide,
        )
        try:
            got = subprocess.run(  # bytes: a safetensors header is binary, not text
                argv,
                capture_output=True,
                timeout=60,
                pass_fds=tuple(pin.fd for pin in pinned.values()),  # bwrap binds them by number
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise ValueError(
                f"{', '.join(map(str, paths))}: the job sandbox cannot start ({e}); refusing the run"
            ) from None
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
    if got.returncode != 0 or not got.stdout:
        why = got.stderr.decode(errors="replace").strip() or "no output"
        raise ValueError(
            f"{', '.join(map(str, paths))}: the job sandbox cannot read it (bwrap exit {got.returncode}: "
            f"{why}); refusing the run: the model's links and every bind they need must resolve inside a job"
        )
    return file


def _largest_file(roots: Iterable[Path], expanded: Iterable[Path]) -> Path | None:
    """The largest regular file the jobs read: a snapshot's own spelling, else a readable file or target."""
    best, size = None, -1

    def consider(path: Path) -> None:
        nonlocal best, size
        if not path.is_file():
            return
        try:
            found = path.stat().st_size
        except OSError:
            return
        if found > size:  # a tie keeps the earlier spelling: a snapshot's own path first
            best, size = path, found

    for root in roots:
        if root.is_dir() and _hub_repo(root) is not None:
            for here, _, files in os.walk(root):
                for name in files:
                    consider(Path(here) / name)
        elif root.is_dir():
            for entry in root.iterdir():  # one level: a checkpoint copy, never a whole venv
                consider(entry)
        else:
            consider(root)
    for path in expanded:
        consider(path)
    return best


def _bound_files(binds: Mapping[Path, Path]) -> list[Path]:
    """One file of each configured bind, at the path a job reads it (the bind destination)."""
    out = []
    for dst, src in binds.items():
        src = Path(src)
        if src.is_file():
            out.append(Path(dst))
            continue
        found = _largest_file([src], ())  # the source's own files, as its destination shows them
        if found is not None and found.is_relative_to(src):
            out.append(Path(dst) / found.relative_to(src))
    return out


def _mount(
    dest: Path, pin: Pinned | None, src: Path, *, writable: bool = False, required: bool = False
) -> list[str]:
    """The bwrap arguments that mount `src` at `dest`, by its descriptor when `pin_sources` opened it.

    Args:
        dest: The mount point inside the sandbox.
        pin: What `pin_sources` opened for `dest`, bound by number so a later swap of the path cannot change it.
        src: The host path to bind when it was not pinned.
        writable: Bind read-write rather than read-only.
        required: Refuse an unpinned mount instead of binding it by path.

    Raises:
        ValueError: `required` and `dest` was not pinned: an allowlist sandbox binds by its descriptor only.
    """
    if pin is not None:
        return ["--bind-fd" if writable else "--ro-bind-fd", str(pin.fd), str(dest)]
    if required:
        raise ValueError(
            f"{src}: not opened by pin_sources; an allowlist sandbox binds only by its descriptor"
        )
    return ["--bind" if writable else "--ro-bind", str(src), str(dest)]


#: What this host can give a sandbox: its `/proc` form and its pid namespace (`_probe_ns`), decided once.
_ns: tuple[str, str] | None = None


def _probe_ns() -> tuple[str, str]:
    """Which `/proc` and pid namespace bwrap can give a sandbox here, best first.

    A fresh procfs in a pid namespace of the sandbox's own is the aim: only the sandbox's processes are in it.
    An outer systemd-nspawn sandbox on the DGX nodes refuses that mount ("Can't mount proc on /newroot/proc:
    Operation not permitted") while allowing the same fresh procfs once the sandbox shares the host's pid
    namespace -- and a fresh procfs is what keeps `/proc/self/task/<tid>/comm` writable, which the CUDA driver
    needs (`cuInit` fails with "Error 304" where it is not). That form shows every host pid, so it
    unshares the user namespace and the kernel then refuses another process's `environ`, `maps`, `mem`, `fd`
    and `root`, as the host `/proc` fallback does. Only where neither mount starts does the host's `/proc` get
    bound read-only; that bind then also needs the host's pid namespace, or the sandbox's own processes would
    be named by numbers the bound `/proc` does not hold.

    A failure for any other reason names no proc mount and keeps the best form: the real argv then reports the
    real error instead of hiding it behind a fallback that would not start either.
    """
    bwrap, true = shutil.which("bwrap"), shutil.which("true")
    if bwrap is None or true is None:
        return "fresh", "private"

    def starts(*flags: str) -> bool | None:
        """Whether the sandbox starts; None when the failure does not name the proc mount."""
        argv = [bwrap, "--die-with-parent", *flags, "--ro-bind", "/", "/", "--dev", "/dev", "--", true]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return not (p.returncode != 0 and "Can't mount proc" in p.stderr)

    for proc, pid, flags in (
        ("fresh", "private", ["--proc", "/proc", "--unshare-pid"]),
        ("fresh", "shared", ["--unshare-user", "--proc", "/proc"]),
    ):
        if starts(*flags) is not False:
            return proc, pid
    return "host", "shared"


def _forms() -> tuple[str, str]:
    """The probe's answer, logging it once so every run records which `/proc` and pid namespace it used."""
    global _ns
    if _ns is None:
        _ns = _probe_ns()
        proc, pid = _ns
        what = "a fresh procfs" if proc == "fresh" else "the host's /proc, read-only"
        _log.info("sandbox: bwrap gives each sandbox %s (pid: %s)", what, pid)
    return _ns


def proc_form() -> str:
    """Whether sandboxes here get a fresh procfs (`"fresh"`) or the host's read-only (`"host"`)."""
    return _forms()[0]


PID_FORMS = ("auto", "private", "shared")


def pid_form(requested: str = "auto") -> str:
    """The pid namespace a sandbox gets: `"private"` (`--unshare-pid`) or the host's (`"shared"`).

    `"auto"` takes what the probe found bwrap can start with (`_probe_ns`): a private namespace wherever a
    fresh procfs can be mounted in one. Where it cannot, the namespace is shared -- with a fresh procfs, or
    with the host's `/proc` bound read-only (`_pair`), which a private namespace would leave naming the
    sandbox's processes by numbers it does not hold (a tool that follows `/proc/self`, as Humanize's cred does
    for the tracee, then fails every call). A shared namespace lets a process name -- and signal -- every
    same-user process of the node, so a run that asks for `"auto"`/`"shared"` may only start on a node it has
    to itself: `isolation.wall.gpu_wall` refuses otherwise (`isolation.allocation.why_not_dedicated`).
    """
    if requested == "auto":
        return _forms()[1]
    if requested not in PID_FORMS:
        raise ValueError(f"pid: {requested!r} is not one of {', '.join(PID_FORMS)}")
    return requested


def _pair(pid: str = "auto") -> tuple[list[str], list[str]]:
    """The `/proc` and pid arguments for a sandbox that gets `pid` (see `pid_form`).

    A private pid namespace can hold a fresh procfs only where this host mounts one in it; elsewhere the host's
    `/proc` is bound read-only instead, with a user namespace of its own (`--unshare-user`). A shared one gets a
    fresh procfs wherever the host mounts one at all (`_probe_ns`) and always the user namespace: with the
    host's processes now reachable by pid, that is what leaves their `environ`, `maps`, `mem`, `fd` and `root`
    unreadable to the sandbox (a signal to one, and its `cmdline` and `status`, stay possible).
    """
    want = pid_form(pid)
    if want == "private":
        if _forms() == ("fresh", "private"):
            return ["--proc", "/proc"], ["--unshare-pid"]
        return ["--unshare-user", "--ro-bind", "/proc", "/proc"], ["--unshare-pid"]
    if proc_form() == "fresh":
        return ["--unshare-user", "--proc", "/proc"], []
    return ["--unshare-user", "--ro-bind", "/proc", "/proc"], []


def proc_args(form: str | None = None, pid: str = "auto") -> list[str]:
    """The bwrap arguments that give a sandbox `/proc`: one mechanism for the wall, the jobs and the grader.

    `--proc /proc` mounts a fresh procfs. Where the host refuses one in a pid namespace of the sandbox's own
    (an outer systemd-nspawn sandbox on the DGX nodes does: "Can't mount proc on /newroot/proc: Operation not
    permitted"), a sandbox that shares the host's pid namespace still gets a fresh procfs, with a user
    namespace of its own (`_pair`); one that keeps a private namespace binds the host's `/proc`
    read-only instead.

    Args:
        form: Which `/proc` to give. None (the wall's and the jobs' choice) follows `_pair(pid)`. `"private"`
            always asks for a fresh procfs (bwrap fails to start where it cannot mount one). `"host-readonly"`
            always binds the host's, exactly as the LiveCodeBench grader's fallback has shipped: no user
            namespace, because the grader starts from an empty root and pins every mount, and generated code
            that must run on these nodes has to see its own `/proc/self`.
        pid: The pid namespace the sandbox gets (`pid_form`), which decides what `/proc` can be.

    Raises:
        ValueError: `form` is not one of the three modes.
    """
    if form is None:
        return _pair(pid)[0]
    if form == "private":
        return ["--proc", "/proc"]
    if form == "host-readonly":
        return ["--ro-bind", "/proc", "/proc"]
    raise ValueError(f"proc: {form!r} is not 'private' or 'host-readonly'")


def pid_args(requested: str = "auto") -> list[str]:
    """The bwrap arguments that give a sandbox its pid namespace (see `pid_form`)."""
    return _pair(requested)[1]


#: Where the judge's receipted builds are published, `(run_dir, workdir) -> [(destination, custody tree)]`.
#: `judge.build_receipt` registers its reader at import: the sandbox holds no judge layout of its own.
PUBLISHED: Callable[[Path | None, Path | None], list[tuple[Path, Path]]] | None = None


def register_published(reader: Callable[[Path | None, Path | None], list[tuple[Path, Path]]]) -> None:
    """Let the judge answer which receipted builds are published for a workdir."""
    global PUBLISHED  # noqa: PLW0603 (one hook for the one judge)
    PUBLISHED = reader


def published_entries(run_dir: Path | None, workdir: Path | None) -> list[tuple[Path, Path]]:
    """`(destination, custody tree)` for every receipted build published for `workdir`.

    The judge owns the receipts and their layout; it registers the reader at import (`register_published`).
    A run whose judge is not imported yet asks for it here, since the agent wall binds published builds
    before the host exists. Nothing is published without the judge: the answer is the empty list.
    """
    global PUBLISHED  # noqa: PLW0603 (the one reader, set on first need)
    if PUBLISHED is None:
        try:
            from qda_agent.judge import build_receipt  # noqa: F401 -- it registers the reader
        except ImportError:
            return []
    return [] if PUBLISHED is None else PUBLISHED(run_dir, workdir)


def bwrap_argv(
    argv: list[str],
    cwd: str,
    *,
    writable: Iterable[Path],
    readable: Iterable[Path] = (),
    pinned: Mapping[Path, Pinned] | None = None,
    devices: Iterable[Path] = (),
    env: dict[str, str] | None = None,
    bwrap: str = "bwrap",
    binds: Mapping[Path, Path] | None = None,
    pid: str = "auto",
    hide: Iterable[Path] = (),
    readonly: Iterable[tuple[Path, Path]] = (),
) -> list[str]:
    """`argv` wrapped in `bwrap`.

    Args:
        argv: The job's command.
        cwd: Its working directory (must be visible).
        writable: Paths bound read-write (the work directory).
        readable: Paths bound read-only besides the system and Python (job dir, model, ...).
        pinned: Sources opened and checked by `pin_sources`, by the path they are bound at (a readable path
            or a bind destination); each is bound by its descriptor, a path not named binds itself.
        devices: Device nodes bound in (`device_nodes`).
        env: The job's whole environment, set inside the sandbox only (`--clearenv` + `--setenv`); None
            keeps bwrap's own (the caller's).
        bwrap: The bwrap executable.
        binds: Read-only binds at another path, `{path inside: host source}`, made after `writable` so they stay
            read-only there; `mount_points` makes their mount points safe first.
        pid: The pid namespace (`pid_form`). `"auto"` is the probe's best -- the only one that gives a GPU job
            a `/proc` the CUDA driver can write (`/proc/self/task/<tid>/comm`). A job with no card needs
            none of that and asks for `"private"`.
        hide: Paths the job must not reach (`sandbox.hide`). Each is masked after every other mount (an empty
            tmpfs for a directory, `/dev/null` for a file), and a job refuses to start if a mask did not hide
            it. A hide that covers a path the job needs is refused, naming both.
        readonly: `(destination, host tree)` mounts made read-only after the writable paths (a receipted
            build output at its published path, by its own and its resolved spelling): the job may read and
            copy it but cannot change it in place.

    Returns:
        The bwrap command line, ending in `-- argv`.

    Raises:
        ValueError: `hide` covers the workdir, a Python root, a readable source or a bind.
    """
    proc, pidns = _pair(pid)  # the pid namespace comes with the /proc: the two are chosen together
    writable = [Path(p) for p in writable]
    binds = dict(binds or {})
    readable = [Path(p) for p in readable]
    hide = hidden_paths(hide)
    roots = python_roots()
    needed = [Path(cwd), *writable, *roots, *readable, *binds, *(Path(s) for s in binds.values())]
    if (hit := hidden_covers(hide, needed)) is not None:
        raise ValueError(
            f"sandbox.hide {hit[0]} covers {hit[1]}, which the job needs (its command directory, workdir, "
            "the Python roots, a readable source or a bind); hide a narrower path"
        )
    # the job's root is the host's: a path no later mount covers is still there (`bwrap_argv` names no
    # `--tmpfs /`), so a hide outside every mount must be masked too
    mounts: list[tuple[str, str, str | None]] = [("bind", "/", "/")]  # (kind, destination, source)
    out = [bwrap, "--die-with-parent", "--unshare-net", *pidns, "--unshare-ipc", "--new-session"]
    for p in SYSTEM:
        if Path(p).exists():
            if Path(p).is_symlink():
                out += ["--symlink", str(Path(p).readlink()), p]
                mounts.append(("symlink", p, None))
            else:
                out += ["--ro-bind", p, p]
                mounts.append(("bind", p, p))
    out += [*proc, "--dev", "/dev", "--tmpfs", "/tmp"]
    mounts += [("tmpfs", "/proc", None), ("tmpfs", "/dev", None), ("tmpfs", "/tmp", None)]
    pins = pinned or {}
    if pins:  # every pinned source is bound by its descriptor: this bwrap must have --bind-fd
        require_bind_fd(bwrap)
    # a Python root can also be a readable source (prepare's config names the venv): bind it once. Since
    # bubblewrap 0.10 a --ro-bind-fd op closes its descriptor and fstats it, so a second op on the same fd
    # fails with EBADF ("Can't stat fd N") -- 0.9.0, which CI installs, has no such check.
    for p in dict.fromkeys([*roots, *readable]):
        out += _mount(p, pins.get(p), p)
        mounts.append(("bind", str(p), str(p)))
    for p in writable:
        out += ["--bind", str(p), str(p)]
        mounts.append(("bind", str(p), str(p)))
    mounted = mount_points(binds, writable, pinned=pins)  # after the workdir: read-only
    for dst, src in mounted.items():
        out += _mount(dst, pins.get(dst), src)
        mounts.append(("bind", str(dst), str(src)))
    views: dict[Path, Path] = {}
    for dst, src in readonly:
        for view in (Path(dst), Path(os.path.realpath(dst))):
            views.setdefault(view, src)
    for dst, src in mount_points(views, writable, deep=views).items():
        out += _mount(dst, None, src)
        mounts.append(("bind", str(dst), str(src)))
    for d in devices:
        out += ["--dev-bind", str(d), str(d)]
        mounts.append(("bind", str(d), str(d)))
    masks = mask_after(mounts, hide)
    for view, host in sorted(masks, key=lambda mask: len(mask[0].parts)):  # parents first
        out += ["--ro-bind", "/dev/null", str(view)] if host.is_file() else ["--tmpfs", str(view)]
    # every spelling is checked too, not only the views a mask was thought needed at: the guard must catch a
    # mask that was not mounted at all
    checks = [(p, Path(os.path.realpath(p))) for p in hide]
    specs = hidden_specs([*checks, *masks])
    if specs:  # checked inside the job's own mounts, before its work starts
        argv = [python(), "-I", "-S", "-c", _HIDDEN_GUARD, json.dumps(specs), *argv]
    if env is not None:
        out.append("--clearenv")
        for k, v in sorted(env.items()):
            out += ["--setenv", k, v]
    return [*out, "--chdir", cwd, "--", *argv]


def pinned_bwrap_argv(
    argv: list[str],
    cwd: str,
    *,
    readable: Iterable[Path],
    binds: Mapping[Path, Path],
    run_dir: Path | None,
    host_written: Iterable[Path],
    writable: Iterable[Path],
    devices: Iterable[Path] = (),
    env: dict[str, str] | None = None,
    pid: str = "auto",
    hide: Iterable[Path] = (),
    readonly: Iterable[tuple[Path, Path]] = (),
) -> tuple[list[str], list[int]]:
    """Prepare the queue's sandbox, returning argv and source descriptors to hold until launch.

    Expand every input link and pin its checked target before making mounts. The caller closes the
    descriptors after launching bwrap (or after a synchronous diagnostic probe finishes).
    """
    readable = expand_links(readable)
    pinned = pin_sources(readable, binds, run_dir=run_dir, host_written=host_written)
    try:
        command = bwrap_argv(
            argv,
            cwd,
            writable=writable,
            readable=readable,
            pinned=pinned,
            devices=devices,
            env=env,
            binds=binds,
            pid=pid,
            hide=hide,
            readonly=readonly,
        )
    except BaseException:
        for pin in pinned.values():
            os.close(pin.fd)
        raise
    return command, [pin.fd for pin in pinned.values()]


def bwrap_allowlist_argv(
    argv: list[str],
    cwd: str,
    *,
    mounts: Mapping[Path, Path],
    writable: Iterable[Path],
    pinned: Mapping[Path, Pinned],
    env: dict[str, str],
    status_fd: int | None = None,
    shm_bytes: int | None = None,
    proc: str = "private",
    bwrap: str = "bwrap",
) -> list[str]:
    """`argv` in a bwrap that starts from an empty root and mounts only what `mounts` names.

    For code that must not see the host (the LiveCodeBench grader): a fresh tmpfs at `/`, then every mount,
    then `/dev` remounted read-only (it is a tmpfs with no size of its own), a sized `/dev/shm` when asked,
    `/proc`, and `/` remounted read-only last, so the writable mounts stay writable and nothing else does.
    `/tmp`, the host's system trees and the Python roots are not mounted, the environment is cleared, and the
    network, pid, ipc and UTS namespaces are unshared.

    `proc` picks the /proc the sandbox sees. `private` mounts a fresh procfs, which shows only the sandbox's
    own processes; a container whose outer /proc has locked covering mounts refuses that mount
    (`mount_too_revealing`), and then `host-readonly` binds the host's /proc read-only instead. The host
    bind still honors the pid namespace's pid view and the host's access checks, but it is not a private
    table: a program can list the host's processes and read what the kernel lets its uid read.

    Every mount must come from `pin_sources` and is bound by its descriptor (`--ro-bind-fd`, `--bind-fd`): the
    source opened at the last moment is exactly what the sandbox sees.

    Args:
        argv: The program to run.
        cwd: Its working directory, inside the sandbox.
        mounts: `{path inside: host source}`.
        writable: The destinations of `mounts` bound read-write; every other mount is read-only.
        pinned: The sources `pin_sources` opened, by the path each is mounted at.
        env: The sandbox's whole environment; the host's is cleared.
        status_fd: bwrap's `--json-status-fd`; its child pid is reported before sandbox setup completes.
        shm_bytes: Size of the private tmpfs at `/dev/shm`; None leaves bwrap's own.
        proc: `private` for a fresh procfs, `host-readonly` to bind the host's `/proc` read-only.
        bwrap: The bwrap executable.

    Returns:
        The bwrap command line, ending in `-- argv`.

    Raises:
        ValueError: A mount's source was not pinned, `writable` names a destination that is not mounted, or
            `proc` is not one of the two modes.
    """
    if proc not in ("private", "host-readonly"):
        raise ValueError(f"proc: {proc!r} is not 'private' or 'host-readonly'")
    require_bind_fd(bwrap)  # every mount below is bound by its descriptor
    writable = {Path(p) for p in writable}
    unmounted = writable - {Path(dest) for dest in mounts}
    if unmounted:
        raise ValueError(f"{', '.join(sorted(map(str, unmounted)))}: writable but not in mounts")
    out = [
        bwrap,
        "--die-with-parent",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--tmpfs",
        "/",
    ]
    for dest, src in mounts.items():
        dest = Path(dest)
        out += _mount(dest, pinned.get(dest), src, writable=dest in writable, required=True)
    out += ["--dev", "/dev", "--remount-ro", "/dev"]
    if shm_bytes is not None:
        out += ["--size", str(shm_bytes), "--tmpfs", "/dev/shm"]
    out += [*proc_args(proc), "--remount-ro", "/", "--clearenv"]
    for k, v in sorted(env.items()):
        out += ["--setenv", k, v]
    if status_fd is not None:
        out += ["--json-status-fd", str(status_fd)]
    return [*out, "--chdir", cwd, "--", *argv]


def mount_points(
    binds: Mapping[Path, Path],
    writable: Iterable[Path],
    *,
    pinned: Mapping[Path, Pinned] | None = None,
    deep: Iterable[Path] = (),
) -> dict[Path, Path]:
    """The read-only binds whose mount points are safe, after making them so; the others are dropped.

    A bind's source must lie outside every writable path: the agent writes there, and could make a source a link
    to anything on the host (refused with ValueError). A mount point directly in a writable path (`work/.claude`)
    is made a real directory, or file, like its source; a link (or an entry of the other kind) put there is
    refused, so a planted link never redirects a bind. Preparation is entirely descriptor-relative
    (`_ensure_mount_point`): a swap between a check and a change cannot make the host follow it, and nothing
    already present is deleted. A deeper mount point must lie under an earlier one (its parents are then the
    host's) or be named in `deep`. Every bound mount point is busy in each sandbox that binds it, so the agent
    cannot rename or replace it.

    Args:
        binds: Read-only binds, `{path inside: host source}`.
        writable: Paths bound read-write.
        pinned: Opened sources by bind destination (`pin_sources`); a pinned bind is judged by what its
            descriptor points at, and its mount point's kind by the descriptor's, not by looking the path up.
        deep: Destinations that may be nested under a writable root (a receipted build at `out/model`):
            their missing parents are made there; a link, another kind or a change under the walk drops the
            bind rather than being followed.

    Raises:
        ValueError: A source inside a writable path.
    """
    roots = [Path(os.path.abspath(w)) for w in writable]  # noqa: PTH100 (links kept: compared as written)
    pins = pinned or {}
    nested = {Path(os.path.abspath(d)) for d in deep}  # noqa: PTH100
    out: dict[Path, Path] = {}
    for dst, src in binds.items():
        d = Path(os.path.abspath(dst))  # noqa: PTH100
        pin = pins.get(Path(dst), pins.get(d))
        s = pin.real if pin is not None else Path(os.path.realpath(src))
        if any(s.is_relative_to(r) or s.is_relative_to(os.path.realpath(r)) for r in roots):
            raise ValueError(f"a read-only bind's source {src} lies in a writable path")
        if any(d.is_relative_to(o) for o in out):
            out[d] = s  # under a bound entry: its parents come from that bind's source
            continue
        root = next((r for r in roots if d.is_relative_to(r) and d != r), None)
        if d.parent in roots:
            root = d.parent
        elif root is None or d not in nested:
            _skipped(d, "not under a writable root or an earlier bind")
            continue
        is_dir = stat.S_ISDIR(os.fstat(pin.fd).st_mode) if pin is not None else s.is_dir()
        why = _ensure_mount_point(root, d, is_dir)
        if why is not None:
            _skipped(d, why)
            continue
        out[d] = s
    return out


#: Called as `on_skip(path, why)` for every bind `mount_points` skips (the flow logs it to `flow.jsonl` while a
#: search runs); a bind skipped in the late binder's own process (a lease command) is in the job's log only.
on_skip: Callable[[str, str], None] | None = None


def _skipped(d: Path, why: str) -> None:
    """Report a bind `mount_points` dropped: on stderr (the job's log) and to `on_skip`."""
    print(f"sandbox: skipped the bind {d}: {why}", file=sys.stderr)
    if on_skip is not None:
        on_skip(str(d), why)


def _ensure_mount_point(root: Path, d: Path, is_dir: bool) -> str | None:
    """Make `d` a mount point under `root` with directory descriptors, or say why it cannot be one.

    Every step is relative to an open directory (`openat` with `O_DIRECTORY | O_NOFOLLOW`, `mkdirat`,
    `fstatat`): after the root is opened once, no pathname is resolved again, so a concurrent agent cannot
    swap an ancestor for a link between a check and a change and have the host follow it. Nothing already
    present is unlinked or replaced -- only missing directories are made -- so the host never deletes or
    overwrites an entry it did not just create. An existing entry is used only when it is of the right kind
    (a directory for a directory bind, a regular file for a file bind); a link, the other kind, or an entry
    that changes under the walk is refused (the caller drops the bind). The path is re-read once at the end
    and compared with the walked descriptors, because a rename outside the walk is invisible to them; a
    difference of any kind refuses the bind. Returns None when `d` is ready.
    """
    try:
        parts = d.relative_to(root).parts
    except ValueError:
        return f"{d} is not under the writable root {root}"
    if not parts:
        return f"{d} is the writable root itself"
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    seen = [os.fstat(fd)]  # the root, then every ancestor the walk opened
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, dir_fd=fd)
            except FileExistsError:
                pass  # it may be a directory; the O_NOFOLLOW open below proves what it is
            except OSError as e:
                return f"{type(e).__name__}: {e}"
            try:
                next_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd
                )
            except OSError:
                return f"{part} in the way of the mount point is not a real directory"
            os.close(fd)
            fd = next_fd
            seen.append(os.fstat(fd))
        name = parts[-1]
        final: os.stat_result | None = None
        for _ in range(3):  # a racing writer may create the name first: look again, never remove it
            try:
                final = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    if is_dir:
                        os.mkdir(name, dir_fd=fd)
                    else:
                        os.close(
                            os.open(
                                name,
                                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                0o644,
                                dir_fd=fd,
                            )
                        )
                    final = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileExistsError:
                    continue
                except OSError as e:
                    return f"{type(e).__name__}: {e}"
                break
            if not ((is_dir and stat.S_ISDIR(final.st_mode)) or (not is_dir and stat.S_ISREG(final.st_mode))):
                return f"{name} is a link or the wrong kind for its mount point"
            break
        if final is None:
            return f"{name} kept changing while its mount point was prepared"
        return _changed_under_the_walk(root, parts, seen, final)
    finally:
        os.close(fd)


def _changed_under_the_walk(
    root: Path, parts: tuple[str, ...], seen: list[os.stat_result], final: os.stat_result
) -> str | None:
    """Refuse when the path the walk was given no longer names the entries its descriptors saw.

    Descriptors prove what the host made, but they cannot see an ancestor renamed away during the walk: the
    pathname bwrap resolves later would then name something else (a link the agent planted). This pass only
    reads -- it never creates, unlinks or replaces -- and a difference of any kind drops the bind.
    """
    path = root
    for part, st in zip(parts[:-1], seen[1:], strict=True):
        path = path / part
        try:
            now = os.lstat(path)
        except OSError as e:
            return f"{path} vanished under the mount point walk: {type(e).__name__}"
        if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
            return f"{path} changed under the mount point walk"
    last = path / parts[-1]
    try:
        now = os.lstat(last)
    except OSError as e:
        return f"{last} vanished under the mount point walk: {type(e).__name__}"
    if (now.st_dev, now.st_ino) != (final.st_dev, final.st_ino):
        return f"{last} changed under the mount point walk"
    return None


def sandbox_env(home: Path, path: str = PATH) -> dict[str, str]:
    """A clean environment for a sandboxed job (no host variables leak in).

    `TZ` is UTC, the clock the deadline and `qsched` speak; a run that configures one (the toolset's
    `env`, or the job's own) overrides it where the caller merges its additions after this.
    """
    return {
        "PATH": f"{Path(os.path.normpath(Path(sys.prefix).absolute())) / 'bin'}:{path}",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "TZ": "UTC",
    }


def _settings_file(argv: list[str]) -> tuple[IO[bytes], int]:
    """The bwrap options of `argv` in an anonymous private file, and its `--` boundary."""
    if any("\0" in arg for arg in argv):
        raise ValueError("embedded null byte")
    boundary = argv.index("--")
    settings = tempfile.TemporaryFile()
    settings.write(b"\0".join(a.encode() for a in argv[1:boundary]) + b"\0")
    settings.seek(0)
    return settings, boundary


@contextlib.contextmanager
def private_args(argv: list[str]) -> Iterator[tuple[list[str], tuple[int, ...]]]:
    """Carry sandbox options in an anonymous private file, away from readable process command lines.

    Yield the launch argv and descriptors the caller must pass to its child. Commands inside bwrap stay
    visible, while environment values and mount metadata use bwrap's native ``--args`` descriptor.
    """
    if argv and Path(argv[0]).name == "bwrap":
        settings, boundary = _settings_file(argv)
        with settings:
            yield [argv[0], "--args", str(settings.fileno()), *argv[boundary:]], (settings.fileno(),)
    else:
        yield argv, ()


def private_args_kept(argv: list[str]) -> tuple[list[str], tuple[int, ...]]:
    """`private_args` without the context: the same argv, and a descriptor that outlives this call.

    A level builds a job's launch before the scheduler starts it, so the settings file must stay open until
    then. The caller passes the descriptors to the child and closes them once it has started.
    """
    if not (argv and Path(argv[0]).name == "bwrap"):
        return argv, ()
    settings, boundary = _settings_file(argv)
    fd = os.dup(settings.fileno())
    settings.close()  # the duplicate keeps the unlinked file's content open
    return [argv[0], "--args", str(fd), *argv[boundary:]], (fd,)


def private_bwrap_argv(argv: list[str], directory: Path) -> list[str]:
    """Carry bwrap options through a 0600 file when the launcher cannot pass descriptors.

    The host-side wrapper reads and removes the file, then uses bwrap's private args descriptor. Only the
    command inside the sandbox stays on argv. Unstarted wrappers leave a private file in the run directory.
    """
    if any("\0" in arg for arg in argv):
        raise ValueError("embedded null byte")
    boundary = argv.index("--")
    fd, path = tempfile.mkstemp(prefix="settings-", suffix=".json", dir=directory)
    with os.fdopen(fd, "w") as settings:
        json.dump(argv[:boundary], settings)
    return [python(), "-I", str(Path(__file__).resolve()), "--bwrap-file", path, *argv[boundary:]]


def visible_cards(value: str | None) -> list[int]:
    """The integer card ids of a `CUDA_VISIBLE_DEVICES` value.

    Raises:
        ValueError: It is unset, empty, or anything but a comma-separated list of card numbers
            (UUIDs and MIG names are refused, never read as "all cards").
    """
    if not value or not value.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES is unset or empty")
    parts = [x.strip() for x in value.split(",")]
    if not all(x.isdigit() for x in parts):
        raise ValueError(f"CUDA_VISIBLE_DEVICES={value!r}: only card numbers are supported")
    return [int(x) for x in parts]


def invisible_cards(cards: Iterable[int], value: str | None = None, proc: Path | None = None) -> str | None:
    """Why `cards` cannot be the run's cards under this shell's `CUDA_VISIBLE_DEVICES`, or None.

    `cards` are physical card numbers (PCI bus order, as `nvidia-smi` lists them), never indices into
    `CUDA_VISIBLE_DEVICES`: with `CUDA_VISIBLE_DEVICES=5` and `cards: [0]` a run would put its judge on card 0,
    which this shell was not given. In a job confined to part of the node they are nvidia-smi's job-local
    indices, and `resolve_cards` maps them to the visible card's UUID before this comparison. So when the
    variable names cards, every card must be in it (a number, or the card's UUID). Unset, or empty (a CPU
    session hiding the cards), names no card to mismatch: nothing is checked.

    Args:
        cards: The run's cards.
        value: The variable's value (default: this process's).
        proc: The driver's `gpus/` table, to read UUIDs.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if value is None:
        return None
    cards = list(cards)
    given = [x.strip() for x in value.split(",") if x.strip()]
    if not given:  # `CUDA_VISIBLE_DEVICES=""`: no card is visible by name, so none can be someone else's
        return None
    numeric = {int(x) for x in given if x.isdigit()}
    outside = [c for c in cards if c not in numeric]
    if outside and given:  # a mix of numbers and UUIDs is legal: the rest may still be named by UUID
        try:
            uuids = card_uuids(cards, proc)
        except CardError:
            return None  # the cards line names the card the driver does not list
        outside = [c for c, uuid in zip(cards, uuids, strict=False) if c not in numeric and uuid not in given]
    if not outside:
        return None
    return (
        f"cards {outside} are not in CUDA_VISIBLE_DEVICES={value!r}: `cards` are physical card numbers (PCI bus "
        "order, as the driver lists them), not indices into CUDA_VISIBLE_DEVICES; name the cards you hold, or "
        "unset CUDA_VISIBLE_DEVICES"
    )


def main(args: list[str]) -> int:
    """Read private bwrap settings, then exec the job."""
    if args[0] != "--bwrap-file":
        raise ValueError("expected --bwrap-file")
    with Path(args[1]).open() as settings:
        spec = json.load(settings)
    Path(args[1]).unlink()
    argv = args[args.index("--") + 1 :]
    with private_args([*spec, "--", *argv]) as (command, fds):
        for fd in fds:
            os.set_inheritable(fd, True)
        os.execvp(command[0], command)
    return 127  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
