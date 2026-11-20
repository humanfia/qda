"""Free room on the disk under a path: bytes and inodes, read with one `os.statvfs` (microseconds, no walk).

The one place QDA measures free space. `qda doctor`, `qda-agent prepare` and the judge's disk cap all read it
here, so a check that passes and a run that then fills the disk cannot disagree about what was free.

`cache_room` judges the caches and the temporary directory a start writes into (`places`): one line per
filesystem, naming the variable that points there. `qda doctor` prints those lines and `qda-agent prepare`
refuses on their MISS. The run directory's own room is `judge.custody.space_problem`'s, not checked here.
"""

from __future__ import annotations

import errno
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from qda_quant.doctor import Check, Info


@dataclass(frozen=True)
class Room:
    """What the filesystem under `path` has free for this user (the unprivileged figures)."""

    path: Path  #: the nearest existing directory the figures were read at
    free_bytes: int
    free_inodes: int
    total_bytes: int
    total_inodes: int

    @property
    def inode_limited(self) -> bool:
        """Whether this filesystem reports inodes at all (some network filesystems report 0)."""
        return self.total_inodes > 0


def existing(path: Path) -> Path:
    """`path`, or its nearest existing ancestor: a directory a run will create is measured where it will land."""
    p = Path(os.path.abspath(path))  # noqa: PTH100 -- normalize without following links
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def room(path: Path) -> Room:
    """Free bytes and inodes on the filesystem that holds (or will hold) `path`; one `statvfs`, no walk."""
    where = existing(path)
    info = os.statvfs(where)
    return Room(
        path=where,
        free_bytes=info.f_bavail * info.f_frsize,
        free_inodes=info.f_favail,
        total_bytes=info.f_blocks * info.f_frsize,
        total_inodes=info.f_files,
    )


#: Free inodes below which a cache or temp directory is a MISS, and the fraction of the filesystem's inodes
#: below which it is too. A build writes thousands of shards, logs and temporary files and an HF cache one file
#: per download, so a filesystem can hold the bytes and still refuse the next `create`. 10,000 files and 5% are
#: the lower of the usual warning thresholds: an ordinary shared disk passes and a nearly exhausted one does not.
INODE_FLOOR = 10_000
INODE_FRACTION = 0.05

#: Filesystem types served from another machine (`_quota_note`). A FUSE type is listed by name: most FUSE
#: filesystems (overlays, squashfs images) are local.
NETWORK_FILESYSTEMS = (
    "nfs",
    "lustre",
    "gpfs",
    "beegfs",
    "cifs",
    "smb3",
    "ceph",
    "glusterfs",
    "afs",
    "9p",
    "fuse.sshfs",
    "fuse.s3fs",
    "fuse.gcsfuse",
)

#: What a cache or temp directory needs free: a lower bound, not a prediction. Nothing measures a cache before it
#: is written (a vLLM kernel cache, a Hub download), so the floor is below the smallest output `qda-agent prepare`
#: does estimate, its search reference's upper bound (96 prompts x 2 answers x 8192 tokens x top-128, about
#: 1.6 GB): a disk without room for even one such output is refused rather than a model's size guessed.
FLOOR_BYTES = 1 << 30


@dataclass(frozen=True)
class Place:
    """One cache or temp directory a start writes into, and the variable that points there.

    Attributes:
        name: the environment variable that names the directory (`HF_HOME`), as the line names it: which of
            them points at a short disk is the whole answer.
        path: the directory (or where it will be made) to measure.
        purpose: what is written there, for the miss line.
        defaulted: the variable is unset, so the directory is where its library's default puts it.
    """

    name: str
    path: Path
    purpose: str
    defaulted: bool = False

    def fix(self, what: str = "a larger disk") -> str:
        """The one change that moves this place to `what`: the export that sets its variable."""
        return f"export {self.name}=<{what}>"


def _hf_home() -> Path | None:
    """`HF_HOME` as `huggingface_hub` resolves it; None when the library is not installed (nothing writes there)."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    return Path(constants.HF_HOME)


def places() -> list[Place]:
    """The cache and temporary directories a start writes into, each read from the code that defines it.

    QDA_CACHE is `qda_quant.io.qda_cache`, HF_HOME `huggingface_hub.constants`, the engines' caches
    `qda_quant.output.vllm_env.cache_dir` (the rule `cache_env` applies), and TMPDIR `tempfile.gettempdir`. A cache whose library is not
    installed writes nothing and is left out.
    """
    from qda_quant.io import qda_cache
    from qda_quant.output.vllm_env import cache_dir

    found = [Place("QDA_CACHE", qda_cache(), "the calibration and reference caches")]
    if (hf := _hf_home()) is not None:
        found.append(Place("HF_HOME", hf, "the Hub downloads"))
    found.append(Place("VLLM_CACHE_ROOT", cache_dir("VLLM_CACHE_ROOT"), "the engines' compiled kernels"))
    found.append(Place("TMPDIR", Path(tempfile.gettempdir()), "the temporary files"))
    return [replace(p, defaulted=not os.environ.get(p.name)) for p in found]


def _unescape(field: str) -> str:
    """A mountinfo field with its octal escapes decoded (``\\040`` is a space)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _mountinfo_rows(text: str) -> list[tuple[Path, str]]:
    """(mount point, fstype) per `/proc/self/mountinfo` line: field 5, and the field after the ``-``."""
    found: list[tuple[Path, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if "-" in fields and len(fields) > max(4, fields.index("-") + 1):
            found.append((Path(_unescape(fields[4])), fields[fields.index("-") + 1]))
    return found


def _mount_table() -> list[tuple[Path, str]]:
    """Every mount's (mount point, fstype); empty where `/proc/self/mountinfo` cannot be read."""
    try:
        return _mountinfo_rows(Path("/proc/self/mountinfo").read_text())
    except OSError:
        return []


def _mount(path: Path, table: list[tuple[Path, str]]) -> tuple[Path, str] | None:
    """The mount `path` lands on (the longest mount point over its real path), or None without a table."""
    real = Path(os.path.realpath(path))
    found = [row for row in table if real.is_relative_to(row[0])]
    return max(found, key=lambda row: len(row[0].parts)) if found else None


def _quota_note(group: list[tuple[Place, Room]], fstype: str | None) -> Info | None:
    """A note for the caches that land in `$HOME` by their library default, on a network filesystem.

    There `statvfs` reports the whole server's free bytes and inodes, and a per-user quota is enforced by the
    server where this check cannot see it: a home directory reads roomy while the user's own quota is nearly
    spent. These figures cannot prove the quota short, so it is a note (``qda doctor --report``, and a warning
    from prepare), never a MISS; it names the variables to move.
    """
    if fstype is None or not fstype.lower().startswith(NETWORK_FILESYSTEMS):
        return None
    home = Path(os.path.realpath(Path.home()))
    at_default = [p for p, _ in group if p.defaulted and Path(os.path.realpath(p.path)).is_relative_to(home)]
    if not at_default:
        return None
    from qda_agent.prompt import _and

    names = [p.name for p in at_default]
    return Info(
        "disk room",
        f"{_and(names)} default{'s' if len(names) == 1 else ''} into $HOME on {fstype}: per-user quotas there "
        "are invisible to this check; fix: "
        + "; ".join(p.fix("a local disk") for p in at_default)
        + "; or check your quota with your site's quota tool",
    )


def _line(group: list[tuple[Place, Room]], mount: Path) -> Check:
    """One filesystem's line: its free figures against `FLOOR_BYTES` for each place on it."""
    from qda_agent.judge.prune import _gb
    from qda_agent.prompt import _and

    names = [p.name for p, _ in group]
    r = min((r for _, r in group), key=lambda r: r.free_bytes)
    need = FLOOR_BYTES * len(group)
    short_bytes = r.free_bytes < need
    short_inodes = r.inode_limited and (
        r.free_inodes < INODE_FLOOR or r.free_inodes < r.total_inodes * INODE_FRACTION
    )
    if not short_bytes and not short_inodes:
        return Check(
            "disk room", True, f"{_and(names)} on {mount}: {_gb(r.free_bytes)} free, {r.free_inodes:,} inodes"
        )
    verb = "is" if len(names) == 1 else "are"
    why = [f"{_and(names)} {verb} on {mount} with {_gb(r.free_bytes)} / {r.free_inodes:,} inodes free"]
    if short_bytes:
        purposes = [p.purpose for p, _ in group]
        why.append(f"{_and(purposes)} need{'s' if len(purposes) == 1 else ''} at least {_gb(need)}")
    if short_inodes:
        below = (
            f"the {INODE_FLOOR:,}-inode floor"
            if r.free_inodes < INODE_FLOOR
            else f"{INODE_FRACTION:.0%} of the filesystem's total"
        )
        why.append(f"{r.free_inodes:,} free inodes are below {below}")
    return Check("disk room", False, "; ".join(why), "; ".join(p.fix() for p, _ in group))


def _groups() -> list[tuple[list[tuple[Place, Room]], tuple[Path, str] | None]]:
    """The places by filesystem (`st_dev`), each with its mount point and type when the mount table names it."""
    groups: dict[object, list[tuple[Place, Room]]] = {}
    for place in places():
        r = room(place.path)
        try:
            key: object = os.stat(r.path).st_dev
        except OSError:  # gone between statvfs and stat: a group of its own, not a traceback
            key = r.path
        groups.setdefault(key, []).append((place, r))
    table = _mount_table()
    return [(group, _mount(group[0][1].path, table)) for group in groups.values()]


def cache_room() -> list[Check]:
    """The cache and temp disks' lines: the one check `qda doctor` prints and `qda-agent prepare` refuses on.

    One line per filesystem (`st_dev`) among `places`: places on one disk write into the same free space, so
    their floors add up there and one line names every variable that points at it. A line is a MISS when the
    free bytes are below that sum, or the free inodes below `INODE_FLOOR` or `INODE_FRACTION` of the total.
    Each place costs one `statvfs` and one `stat`, plus one read of `/proc/self/mountinfo` in all.
    """
    return [_line(group, group[0][1].path if mount is None else mount[0]) for group, mount in _groups()]


def room_notes() -> list[Info]:
    """What the free figures cannot prove: a cache that defaults into a network `$HOME` (`_quota_note`)."""
    return [
        note
        for group, mount in _groups()
        if (note := _quota_note(group, None if mount is None else mount[1]))
    ]


def room_report() -> list[Info]:
    """``qda doctor --report``'s disks: each filesystem by the variables on it, its type and free figures (no path)."""
    from qda_agent.judge.prune import _gb
    from qda_agent.prompt import _and

    found = []
    for group, mount in _groups():
        r = min((r for _, r in group), key=lambda r: r.free_bytes)
        fstype = mount[1] if mount is not None else "unknown"
        names = _and([p.name for p, _ in group])
        found.append(
            Info("disk room", f"{names} on {fstype}: {_gb(r.free_bytes)} free, {r.free_inodes:,} inodes")
        )
        if note := _quota_note(group, None if mount is None else mount[1]):
            found.append(note)
    return found


#: The size of the file `full_place` writes: enough that a quota with a few KB left still refuses it.
PROBE_BYTES = 1 << 20


def _write_probe(directory: Path) -> None:
    """Write and fsync `PROBE_BYTES` into a temp file under `directory`, then delete it (the one write a probe makes)."""
    import tempfile as _tempfile

    with _tempfile.NamedTemporaryFile(dir=directory, prefix=".qda-probe-") as f:
        f.write(b"\0" * PROBE_BYTES)
        f.flush()
        os.fsync(f.fileno())


def is_full(place: Place) -> bool:
    """Whether a 1 MiB write under the place fails for want of room (ENOSPC or EDQUOT).

    Unlike `statvfs` this sees a per-user quota, since it asks the filesystem to accept bytes. Any other
    failure (a directory it cannot write, one that is gone) is not "full" and is left to the step to report.
    """
    try:
        _write_probe(existing(place.path))
    except OSError as e:
        return e.errno in (errno.ENOSPC, errno.EDQUOT)
    return False


def full_line(extra: list[Place] | None = None) -> str | None:
    """The one refusal line for the first full place among `places()` and `extra`, or None when none is full.

    `error: <VAR> (<path>) is full; fix: export <VAR>=<a directory on a disk with more room>`; a place named
    ``--out`` (the run directory) says ``fix: pass --out <a directory on a larger disk>``. The `error: ` prefix
    is `refuse`'s, so this is the text after it.
    """
    for place in [*places(), *(extra or [])]:
        if is_full(place):
            fix = (
                "pass --out <a directory on a larger disk>"
                if place.name == "--out"
                else f"export {place.name}=<a directory on a disk with more room>"
            )
            return f"{place.name} ({place.path}) is full; fix: {fix}"
    return None
