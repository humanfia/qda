"""Prune build custody when the disk approaches its free-space floor.

A trusted build keeps its output in host-only custody (`<run_dir>/judge/builds/<private>/published`), and a
search builds more candidates than the disk holds at once. A prune pass never frees a build a search still
needs: one a queued or running job reads, a hand-in (every one it did not refuse, whole: `_handed`), the
best-scored build by KL (`_best_kl`), the current final pick, candidate 0 and the baselines, and the frozen
copies a cutoff final rule reads after the run (`_frozen_keep`). It removes the weight files of the other
scored builds only when one more build's
output would not fit above the free-space floor (`judge.min_free_bytes`, `_short_of_room`), the worst KL
first, and stops as soon as it fits; each prune appends one record to the judge ledger. A build is
identified by its receipt's whole-tree digest, so two names for the same build count once. A frozen copy is
identified by its own digest: `kl`, `handin` and `gencheck` of the same files share one directory
(`intake.frozen_for`), and near the floor the pass deletes a copy every entry of which has a verdict
(`reclaim`) unless it is a hand-in, one the cutoff rule reads, the final pick or held by a queued or running
job (so a hand-in copy is either kept whole or removed whole, never left as metadata); the ledger line
and the reading's result file stay. A pruned publication is renamed `pruned/`, so no sandbox mounts it; its
receipt, `build.json`, score and logs stay, and a hand-in of those files later re-freezes them from the
agent's own copy when its digest matches the receipt (`intake.take`). Since a hand-in's frozen copy hardlinks
the custody tree, the stale frozen copies of a pruned build lose their weight files too (unless a job is
still reading one or it is the final pick), and the ledger records only bytes whose last link was removed. A
build whose reading passed or was refused is ranked (a refused reading is no reason to keep its weights), and
so is one whose kl or hand-in request was refused at intake: that refusal is a verdict on the build itself
(an off-menu format, an unreadable export), not a copy the judge is unsure of. One still being read or
checked, one not read yet, and the custody tree a hand-in is freezing right now are never pruned: the judge
is not sure of them. The same rules hold without build receipts (`judge.require_build_receipt: false`): there
is no custody tree then, a frozen copy's own digest is its build's identity, and a superseded frozen copy
(scored, not in use) loses its weight files near the floor.

Except at the disk cap. Once the run holds more than `judge.max_run_bytes` on disk (an install can land above
its estimate), every reading, hand-in and check is refused, so an unscored build would never be scored and
never become prunable: the run could not get back under the cap by itself. Then a pass also frees builds that
were never scored (no finished full reading), whatever else is not protected: never candidate 0 or a baseline,
the final pick, a build a queued or running job reads, one with a reading or check in flight, one ever handed
in, or the newest unscored build (the one the agent is most likely trying to get read). A build never read
goes before one with only a screening reading, the oldest first, and the pass stops as soon as the run is back
under the cap with room for what is already admitted (`check_room`'s sum: on disk plus reserved). It frees
nothing when even all of them would not get it there; the refusal (`over_cap`) then names the builds that
hold the bytes.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import stat
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..room import room
from ..sched import Job, Refused, State
from ..sched.snapshot import remove_tree
from .checks.budget import stored_bits
from .custody import FLOOR, floor_problem, free_floor
from .freeze import tree_size, walk
from .intake import read_json, same_dir
from .ledger import utc

if TYPE_CHECKING:
    from .host import Entry, JudgeHost

__all__ = [
    "Build",
    "SpaceRefused",
    "builds",
    "check_room",
    "disk_message",
    "expected_bytes",
    "floor_paths",
    "note_space_refusal",
    "drop_frozen",
    "over_cap",
    "prune",
    "reclaim",
    "pruned_at",
    "pruned_build",
    "reap",
    "reserved_bytes",
    "run_bytes",
    "is_whole",
    "kept_handins",
    "watch_floor",
    "weight_files",
]


class SpaceRefused(Refused):
    """A copy or job refused for disk space, not for what the checkpoint is.

    The distinction matters for a hand-in: a build refused for space can still be the best checkpoint of the
    run, so `final_rules` may submit its scored reading instead; a refusal for what the files are cannot.
    """


_log = logging.getLogger(__name__)

#: File suffixes that hold weights; a pruned tree keeps every other file.
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
#: Where a pruned publication moves: not `published/`, so no sandbox mounts it any more.
PRUNED = "pruned"
#: The prune record beside `receipt.json` (what, when, bytes freed); the ledger gets the same record.
RECORD = "pruned.json"
#: The largest snapshotted script or config a pending job's text scan reads.
MAX_SNAPSHOT = 1 << 20
#: The `reason` of a prune record for a build freed at the disk cap before any reading scored it.
AT_CAP = "over the disk cap before it was scored"
#: The most holding builds a refusal at the cap names.
NAMED = 8
#: What a `.gguf` holds beyond the anchor's metadata files and its tensors: the tensor table and the alignment
#: padding (a few bytes a tensor), with room to spare. Reserved on top of `judge.budget_bytes` when the budget
#: is the payload alone (`judge.budget_basis: payload`).
GGUF_HEADER_MARGIN = 64 << 20
#: Per host, what the last "nothing can be freed" warning was about: it is logged once per change, not per tick.
_STUCK: weakref.WeakKeyDictionary[JudgeHost, tuple] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class Build:
    """One build still published in judge custody.

    Attributes:
        private: The build's private custody directory (`<run_dir>/judge/builds/<name>`).
        tree: Its published tree (`private/published`).
        destination: The workdir-relative path the tree is mounted at.
        identity: The receipt's whole-tree digest (two names for one build share it).
        job: The trusted job id from the receipt.
    """

    private: Path
    tree: Path
    destination: str
    identity: str
    job: str


def builds(host: JudgeHost) -> list[Build]:
    """Every build still published for the run's workdir, one per custody directory.

    A custody directory whose receipt is missing or does not publish inside this run's workdir is skipped,
    the rule `sandbox.published_entries` applies to the mount list.
    """
    root = host.jdir / "builds"
    if not root.is_dir():
        return []
    out: list[Build] = []
    for private in sorted(root.iterdir()):
        tree = private / "published"
        if private.is_symlink() or not private.is_dir() or tree.is_symlink() or not tree.is_dir():
            continue
        receipt = read_json(private / "receipt.json") or {}
        destination, workdir = receipt.get("destination"), receipt.get("workdir")
        if not isinstance(destination, str) or not destination:
            continue
        if isinstance(workdir, str) and os.path.realpath(workdir) != os.path.realpath(host.workdir):
            continue
        rel = Path(destination)
        if rel.is_absolute() or ".." in rel.parts or rel == Path():
            continue
        job = receipt.get("job_id")
        identity = _identity(receipt)
        out.append(
            Build(
                private,
                tree,
                rel.as_posix(),
                identity or f"unreceipted:{private.name}",
                job if isinstance(job, str) else private.name,
            )
        )
    return out


def pruned_build(host: JudgeHost, rel: str) -> tuple[dict, dict] | None:
    """The newest pruned publication at `rel`: its prune record and its build receipt, or None.

    None while a publication is live at `rel` (a prune record of an older build there does not speak for
    the live one) and when no prune record exists.
    """
    if any(b.destination == rel for b in builds(host)):
        return None
    root = host.jdir / "builds"
    if not root.is_dir():
        return None
    newest: tuple[float, dict, dict] | None = None
    for private in root.iterdir():
        if private.is_symlink() or not private.is_dir():
            continue
        receipt = read_json(private / "receipt.json") or {}
        destination = receipt.get("destination")
        if not isinstance(destination, str) or Path(destination).as_posix() != rel:
            continue
        record = read_json(private / RECORD)
        if record is None:
            continue
        stamp = record.get("at")
        when = float(stamp) if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) else 0.0
        if newest is None or when > newest[0]:
            newest = (when, record, receipt)
    return None if newest is None else (newest[1], newest[2])


def pruned_at(host: JudgeHost, rel: str) -> dict | None:
    """The newest prune record of the build last published at `rel`, or None when a publication is live."""
    found = pruned_build(host, rel)
    return None if found is None else found[0]


def run_bytes(host: JudgeHost, *, scratch: bool = False) -> int:
    """Bytes the run holds on disk, each inode once.

    Everything under the run directory counts: published custody, the frozen copies, the judge's own files
    and the agent's work/ (its local scheduler builds included). The workdir counts even when configured
    outside the run directory (`work_dir`), except in the scratch count: it is not on the run dir's storage. A hardlink (a hand-in's copy of a custody tree) is one physical copy and
    counts once. The configured custody root is walked explicitly: judge/builds and judge/frozen are
    directory links in local mode and walk(run_dir) deliberately does not follow them. Other symlinks
    are never followed, so an agent link cannot charge unrelated host data. The engines' compile caches are
    node-local (`qda_quant.output.vllm_env.cache_root`), outside the run directory, so nothing here counts them.

    Args:
        host: The run whose owned paths are counted.
        scratch: Count the run dir's storage only: run_dir, without the explicit local custody root or a
            workdir outside run_dir.
    """
    tops = [host.run_dir]
    if not host.workdir.is_relative_to(host.run_dir) and not scratch:
        tops.append(host.workdir)
    if host.cfg.custody_dir is not None and not scratch:
        tops.append(host.cfg.custody_dir.resolve())
    inodes: dict[tuple[int, int], int] = {}
    for top in tops:
        if top.is_symlink() or not top.is_dir():
            continue
        for path in walk(top):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                info = path.lstat()
            except OSError:
                continue  # a job writing or already gone: the next count sees it
            inodes[(info.st_dev, info.st_ino)] = info.st_size
    return sum(inodes.values())


def _gb(n: int) -> str:
    """Decimal gigabytes, the unit of `judge.max_run_bytes` (1e9, not 2**30)."""
    return f"{n / 1e9:.1f} GB"


def disk_message(
    held: int, reserved: int, expected: int, cap: int, *, stuck: Sequence[tuple[str, int]] = ()
) -> str:
    """The agent-facing refusal at the run cap: each part in decimal GB, the total and the cap, and what to do.

    `stuck` names the published builds the judge's own pass never frees, with their weight bytes, so a jam
    is visible in the refusal itself.
    """
    total = held + reserved + expected
    parts = [f"{_gb(held)} on disk"]
    if reserved:
        parts.append(f"{_gb(reserved)} reserved for queued/running jobs")
    if expected:
        parts.append(f"{_gb(expected)} for this job")
    note = "" if held > cap else " (on-disk use alone is under the cap)"
    stranded = ""
    if stuck:
        named = ", ".join(f"{where} ({_gb(size)})" for where, size in stuck[:NAMED])
        more = f", {len(stuck) - NAMED} more" if len(stuck) > NAMED else ""
        stranded = f"; never scored, so no prune frees them: {named}{more}"
    return (
        f"disk budget: {' + '.join(parts)} = {_gb(total)} of {_gb(cap)}{note}{stranded}; "
        "prune finished builds, hand in, or wait for queued jobs to end"
    )


def _unprunable(host: JudgeHost) -> list[tuple[str, int]]:
    """Builds no ordinary prune pass frees, largest first, with the bytes their weights hold.

    A build with no finished reading or check and no intake refusal is not scored, so the floor's prune
    pass never reaches it. One with a reading in flight, or that a queued or running job reads, may still
    be scored later: it is not listed.
    """
    found = builds(host)
    ranked = _done(host) | _intake(host) | _pending(host)
    try:
        ranked |= _in_use(host, found)
    except Exception:  # noqa: BLE001 -- the refusal still states the bytes; the names are a courtesy
        _log.exception("could not tell which builds are in use while naming the never-scored ones")
    out: list[tuple[str, int]] = []
    for b in found:
        if b.identity in ranked:
            continue
        size = 0
        for path in weight_files(b.tree):
            try:
                size += path.lstat().st_size
            except OSError:
                continue  # gone already: nothing to name
        if size:
            out.append((b.destination, size))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def over_cap(host: JudgeHost) -> str | None:
    """The refusal message when the run already holds more than `judge.max_run_bytes`, else None.

    Over the cap, it first runs a prune pass (under the host lock, as the tick does), which frees the
    unscored builds the cap allows, and admits when that got the run back under. Otherwise the message names
    the builds that hold the bytes, each with why it is kept, and the bytes outside them, so the operator knows
    what to raise or drop.
    """
    if _over(host) is None:
        return None
    with host.lock:
        prune(host)
        return _over(host)


def _over(host: JudgeHost) -> str | None:
    """`over_cap`'s message from the bytes on disk now, without freeing anything."""
    cap = host.cfg.max_run_bytes
    if cap is not None and (used := run_bytes(host)) > cap:
        return disk_message(used, 0, 0, cap) + _holders(host, used)
    cap = host.cfg.max_scratch_bytes
    if cap is not None and (used := run_bytes(host, scratch=True)) > cap:
        # with local custody the builds are not on scratch: naming them would not explain the bytes
        holders = _holders(host, used) if host.cfg.custody_dir is None else ""
        return "scratch " + disk_message(used, 0, 0, cap) + holders
    return None


def _excess(host: JudgeHost) -> int:
    """How far the run is past its caps with what is admitted already (zero or less: room under them).

    The sum is `check_room`'s, on disk plus reserved, so freeing stops where the next admission fits. The
    scratch cap counts only when the builds are on scratch: with `judge.custody_dir` freeing them does not
    lower scratch use.
    """
    over = 0
    if (cap := host.cfg.max_run_bytes) is not None:
        over = run_bytes(host) + reserved_bytes(host) - cap
    if (cap := host.cfg.max_scratch_bytes) is not None and host.cfg.custody_dir is None:
        over = max(over, run_bytes(host, scratch=True) + reserved_bytes(host, scratch=True) - cap)
    return over


def _on_disk_over(host: JudgeHost) -> bool:
    """Whether the run holds more on disk than a cap allows: the state that refuses every reading."""
    if (cap := host.cfg.max_run_bytes) is not None and run_bytes(host) > cap:
        return True
    cap = host.cfg.max_scratch_bytes
    return cap is not None and host.cfg.custody_dir is None and run_bytes(host, scratch=True) > cap


def _inodes(tree: Path) -> dict[tuple[int, int], int]:
    """The size of each file inode under `tree` (symlinks are never followed)."""
    out: dict[tuple[int, int], int] = {}
    for path in walk(tree):
        if path.is_file() and not path.is_symlink():
            try:
                info = path.lstat()
            except OSError:
                continue
            out[(info.st_dev, info.st_ino)] = info.st_size
    return out


def _holders(host: JudgeHost, used: int) -> str:
    """Which builds hold the run's bytes at the cap, the largest first, and why the judge keeps each."""
    found = builds(host)
    try:
        why = _why_kept(host, found)
    except Exception:  # noqa: BLE001 -- the refusal still states the bytes; the reasons are a courtesy
        _log.exception("could not tell why each build is kept")
        why = {}
    sizes, every = [], {}
    for b in found:
        inodes = _inodes(b.tree)
        every.update(inodes)
        sizes.append((sum(inodes.values()), b))
    sizes.sort(key=lambda x: (-x[0], x[1].destination))
    named = [f"{b.destination} {_gb(n)} ({why.get(b.identity, 'kept')})" for n, b in sizes[:NAMED]]
    if len(sizes) > NAMED:
        named.append(f"{len(sizes) - NAMED} more")
    held = "; held by builds: " + ", ".join(named) if named else "; no build is published"
    return (
        f"{held}; {_gb(max(0, used - sum(every.values())))} outside the builds; the judge frees an unscored "
        "build here only if that gets the run back under the cap: the operator can raise judge.max_run_bytes"
    )


def _why_kept(host: JudgeHost, found: list[Build]) -> dict[str, str]:
    """Why each published build is kept, by identity (the first reason that applies)."""
    out: dict[str, str] = {}

    def mark(ids: set[str], reason: str) -> None:
        for i in ids:
            out.setdefault(i, reason)

    mark(_baselines(host, found), "candidate 0 or a baseline")
    mark(_frozen_keep(host), "a cutoff rule reads it after the run")
    mark(_best_kl(host), "the best scored build")
    mark(_pending(host), "a reading in flight")
    mark(_handed(host), "handed in")
    mark(_in_use(host, found), "in use by a job or the final pick")
    mark(_intake(host), "refused at intake")
    mark(_done(host), "scored")
    mark({b.identity for b in found}, "unscored: no prune pass frees it")
    return out


def _floor_holders(host: JudgeHost) -> str:
    """Who holds the run's bytes below the free-space floor: this run's own builds and frozen copies.

    The floor refusal is what the agent reads when its build or reading is refused, so it names only files
    this run holds (published builds and the judge's frozen copies) and what keeps each. Anything else on
    the disk is neither the run's to name nor its to delete; the closing sentence says so, because a
    refused agent has looked for other people's bytes to free.
    """
    found = builds(host)
    try:
        why = _why_kept(host, found)
    except Exception:  # noqa: BLE001 -- the refusal still states the free space; the names are a courtesy
        _log.exception("could not tell why each build is kept")
        why = {}
    counted: set[tuple[int, int]] = set()  # one inode once: a frozen copy hardlinks its build's weights
    named: list[tuple[int, str]] = []
    for b in found:
        size = 0
        for path in weight_files(b.tree):
            try:
                info = path.lstat()
            except OSError:
                continue  # gone already: nothing to name
            counted.add((info.st_dev, info.st_ino))
            size += info.st_size
        if size:
            named.append((size, f"build {b.destination} {_gb(size)} ({why.get(b.identity, 'kept')})"))
    named.extend(_frozen_floor_holds(host, counted))
    named.sort(key=lambda x: -x[0])
    held = (
        "held by this run's own builds and frozen copies: " + ", ".join(name for _n, name in named[:NAMED])
        if named
        else "no build or frozen copy of this run holds weights"
    )
    if len(named) > NAMED:
        held += f", {len(named) - NAMED} more"
    return (
        f"{held}; only the judge's own commands free this run's builds: free space only under your work "
        "directory, never delete a file outside it"
    )


def _frozen_floor_holds(host: JudgeHost, counted: set[tuple[int, int]]) -> list[tuple[int, str]]:
    """This run's frozen copies that hold weight bytes of their own, with what needs each.

    A copy hardlinks the custody tree of its build, so a build's line already counts those bytes; only
    weight files whose inode `counted` does not hold are named (and then added, so two copies of one tree
    still count once). Every copy is under `judge/frozen`, inside the run: no other path is read or named.
    """
    root = _norm(host.jdir / "frozen")
    if not root.is_dir():
        return []
    final = None
    try:
        final = _final_dir(host)
    except Exception:  # noqa: BLE001 -- an unknown pick costs a reason, not the refusal
        _log.exception("could not tell the final pick while naming frozen copies")
    rank = {"read": 0, "handed in": 1, "a reading in flight": 2, "the final pick": 3}
    why: dict[Path, str] = {}
    for e in host.entries:
        tree = _norm(e.dir)
        if tree.parent != root:
            continue
        if final is not None and same_dir(tree, final):
            reason = "the final pick"
        elif e.kind == "handin":
            reason = "handed in"
        elif e.valid is None:
            reason = "a reading in flight"
        else:
            reason = "read"
        # one copy can back several entries; the reason that needs it most speaks for it
        if rank[reason] > rank.get(why.get(tree, "read"), 0):
            why[tree] = reason
    out: list[tuple[int, str]] = []
    for tree, reason in why.items():
        size = 0
        for path in weight_files(tree):
            try:
                info = path.lstat()
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in counted:
                continue
            counted.add(key)
            size += info.st_size
        if size:
            out.append((size, f"frozen copy {tree.name} {_gb(size)} ({reason})"))
    return out


def check_room(host: JudgeHost, expected: int = 0, *, scratch_expected: int | None = None) -> None:
    """Refuse the next job or build when the run's use plus `expected` would pass `judge.max_run_bytes`.

    The use is the real byte count of the run directory plus the reservations of the trusted builds and the
    agent jobs still queued or running (their estimates are not on disk yet). The free-space floor
    (`custody.floor_problem`, `judge.min_free_bytes`) is checked first, cap or not: it reads the disk as it is
    now, so `expected`, an estimate, plays no part in it; without a cap it is the only refusal. No running job
    is stopped here (`watch_floor` stops them below the floor).

    Args:
        host: The run whose caps apply.
        expected: New output bytes reserved against the total cap.
        scratch_expected: New scratch bytes; None uses expected. Local trusted builds reserve their
            configured job_reserve_bytes for durable records/logs instead of their local weight size.

    Raises:
        Refused: The run's disk is below the floor, the run is at or over the cap, or `expected` more bytes
            would pass it.
    """
    if (problem := floor_problem(floor_paths(host), host.cfg)) is not None:
        # the agent gets the floor's own sentence plus what this run holds and the one rule it can follow:
        # the operator's fix (move the run, raise the disk, lower the floor) is for the launch refusal only
        raise SpaceRefused(f"{problem.agent or problem.detail}; {_floor_holders(host)}")
    cap = host.cfg.max_run_bytes
    if cap is not None:
        held, reserved = run_bytes(host), reserved_bytes(host)
        if held + reserved + expected > cap:
            raise SpaceRefused(disk_message(held, reserved, expected, cap, stuck=_unprunable(host)))
    cap = host.cfg.max_scratch_bytes
    if cap is not None:
        held, reserved = run_bytes(host, scratch=True), reserved_bytes(host, scratch=True)
        needed = expected if scratch_expected is None else scratch_expected
        if held + reserved + needed > cap:
            # with local custody the builds are on scratch: naming them explains part of the bytes
            stuck = _unprunable(host) if host.cfg.custody_dir is None else ()
            raise SpaceRefused("scratch " + disk_message(held, reserved, needed, cap, stuck=stuck))


def floor_paths(host: JudgeHost) -> list[Path]:
    """Where the run's builds land, whose free space the floor watches: the custody root (else `run_dir`) and
    the agent's workdir (its local builds), which may be on another disk."""
    return [host.cfg.custody_dir or host.run_dir, host.workdir]


def _short_of_room(host: JudgeHost) -> bool:
    """Whether one more build's output would not fit above the free-space floor.

    The next build lands in the custody root (else `run_dir`) or, without custody, in the agent's workdir:
    both are `floor_paths`, and either disk being short is a reason to make room. With the floor off
    (`judge.min_free_bytes: 0`) nothing is pruned here: only the disk cap's rule (`_at_cap`) frees anything.
    """
    estimate = _next_build_bytes(host)
    if estimate <= 0:
        return False
    for path in floor_paths(host):
        free = room(path)
        floor = free_floor(host.cfg)
        if floor and free.free_bytes < floor + estimate:
            return True
    return False


def _next_build_bytes(host: JudgeHost) -> int:
    """The bytes one more build's output is expected to take; 0 when the run gives no estimate.

    The anchor's estimate (`expected_bytes`) is the run's own basis, and a llama.cpp search's byte budget
    bounds it. Without either, the largest tree this run holds (a published build, else a frozen copy) is
    what builds of this run have cost; none means no estimate, and the floor then only refuses new work
    (`check_room`).
    """
    if host.cfg.anchor is not None or (host.cfg.engine == "llama.cpp" and host.cfg.budget_bytes is not None):
        return expected_bytes(host)
    sizes = [tree_size(b.tree) for b in builds(host)]
    if not sizes:
        sizes = [tree_size(e.dir) for e in host.entries if e.dir.is_dir() and not e.dir.is_symlink()]
    return max(sizes, default=0)


def watch_floor(host: JudgeHost) -> list[str]:
    """Each scheduler pass: below the free-space floor, stop the agent's running jobs; the jobs it stopped.

    The floor is read live (`custody.floor_problem`), so a build that crosses it is stopped by the next pass,
    not refused by an estimate made before it started. A pass that stops a job records it as a space refusal
    (`note_space_refusal`), so the `handin` rule submits the best scored eligible entry instead of waiting on
    a build that will not land; the first pass below the floor says so in two lines, and a later pass is
    silent until the disk has been above it again. Called under the host lock (`jobs.fold`).
    """
    problem = floor_problem(floor_paths(host), host.cfg)
    if problem is None:
        host.below_floor = False
        host.below_floor_since = None
        return []
    stopped = [
        host.sched.cancel(job.name).name
        for job in host.sched.list(origin="agent")
        if job.state is State.running
    ]
    if stopped:
        note_space_refusal(host, ", ".join(stopped), SpaceRefused(str(problem)), what="build")
    elif not host.below_floor:
        host.announce(f"{problem.detail}: no agent job is running to stop; {problem.fix}")
        host.announce(
            f"{FLOOR}: no new build, job or hand-in copy is admitted until the disk is above it again"
        )
    if not host.below_floor:  # the first pass below this stretch: the flow finalizes after `FINALIZE_AFTER_S`
        host.below_floor_since = host.clock.now()
    host.below_floor = True
    return stopped


#: The second of the two lines a stop or refusal at the free-space floor says (`note_space_refusal`).
FLOOR_SAID = (
    f"{FLOOR}: falling back to the best existing entry: the run submits the best scored eligible build "
    "(or candidate 0 as-is) instead"
)


@contextmanager
def disk_lock(host: JudgeHost) -> Iterator[None]:
    """Serialize admission through its recorded job or completed copy, across threads and processes.

    Hold this lock from `check_room` until the new reservation is recorded (or the copy is on disk).
    The empty lock file is host-owned and never removed: replacing it would split the lock's inode.
    """
    if host.cfg.max_run_bytes is None and host.cfg.max_scratch_bytes is None:
        yield
        return
    with host.lock, (host.jdir / "disk.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def expected_bytes(host: JudgeHost) -> int:
    """The judge's estimate of one build's output bytes.

    A build writes its checkpoint from the anchor's tensors: its decoder Linears at `judge.budget_bpw` bits
    a weight (their count is the budget's fixed denominator), every other tensor as the anchor stores it (a
    leased head stores no more than that). Subtract the original Linears' actual stored bytes, including FP8
    sources. Without a budget the anchor's own size bounds the output; without an anchor, the largest
    checkpoint the judge admits (`max_copy_bytes`) does.

    A llama.cpp search states its budget in bytes (`judge.budget_bytes`; it refuses `budget_bpw`), and a
    candidate over it is refused, so that is the output's bound: the whole file on the `file` basis, and on the
    `payload` basis the tensors plus a GGUF header (`_gguf_header_bytes`); never above the largest checkpoint
    the judge admits (`max_copy_bytes`).
    """
    if host.cfg.engine == "llama.cpp" and (budget_bytes := host.cfg.budget_bytes) is not None:
        bound = budget_bytes + (_gguf_header_bytes(host) if host.cfg.budget_basis == "payload" else 0)
        return min(bound, host.max_bytes)
    try:
        anchor = host.cfg.anchor_bytes()
    except ValueError:
        anchor = None
    if anchor is None:
        return host.max_bytes
    weights, budget = host.bpw_denominator, host.cfg.budget_bpw
    if weights and budget and host.cfg.anchor is not None:
        return max(0, anchor - stored_bits(host.cfg.anchor) // 8) + int(weights * budget / 8)
    return anchor


def _gguf_header_bytes(host: JudgeHost) -> int:
    """A bound on a `.gguf`'s header: the anchor's metadata files (the tokenizer and configs it copies) + a margin.

    The anchor's weight files are not metadata; an anchor that is not a directory adds only the margin.
    """
    anchor = host.cfg.anchor
    meta = 0
    if anchor is not None and Path(anchor).is_dir():
        for path in Path(anchor).iterdir():
            if path.is_file() and not path.is_symlink() and path.suffix not in WEIGHTS:
                meta += path.stat().st_size
    return meta + GGUF_HEADER_MARGIN


def reserved_bytes(host: JudgeHost, *, scratch: bool = False) -> int:
    """The expected output bytes of every trusted build and agent job queued or running.

    A trusted build records its full plan estimate in its `request.json` when it is queued. An agent job
    reserves what it declares: a full checkpoint when it writes one (`qda build`'s plan estimate, or `qsched
    add --checkpoint`'s estimate of the original at `judge.budget_bpw`), otherwise `judge.job_reserve_bytes`
    (default 2 GB), so inspect and eval jobs no longer each hold a checkpoint's worth. The reservation lasts
    while the job is pending and is replaced by the real bytes `run_bytes` counts when it ends; a
    checkpoint written by an undeclared job can therefore overshoot by at most the concurrent jobs' writes
    before the next admission is refused. A trusted build with no recorded estimate (an older custody
    directory) is reserved at `expected_bytes`. A request between its admission and its job
    (`host.build_reserved`, recorded under the lock in `setup`) counts until the job exists.
    """
    fallback = expected_bytes(host)
    total = 0
    known: set[str] = set()
    jobs = {job.id: job for job in host.sched.jobs.values()}
    # An admission in another process sees the scheduler's atomically published reservations too.
    for path in host.sched.dir.glob("*/job.json"):
        spec = read_json(path)
        if spec is not None and spec.get("id") not in jobs:
            job = Job.model_validate(spec)
            jobs[job.id] = job
    for job in jobs.values():
        if job.state.terminal:
            continue
        if job.build_private is None:
            # An agent job reserves what it declared at admission (a checkpoint, or the small default); a
            # judge job reserves its own recorded estimate (its results are small).
            short = host.cfg.job_reserve_bytes if job.origin == "agent" else 0
            if scratch and job.origin == "agent" and not host.workdir.is_relative_to(host.run_dir):
                total += short  # its output is in the workdir, off the run dir's storage; its log is not
            else:
                total += job.expected_bytes or short
            continue
        known.add(job.build_private)
        if scratch and host.cfg.custody_dir is not None:
            total += host.cfg.job_reserve_bytes
            continue
        spec = read_json(Path(job.build_private) / "request.json") or {}
        value = spec.get("expected_bytes")
        total += value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback
    for path, value in host.build_reserved.items():
        if path not in known and Path(path).is_dir():
            total += host.cfg.job_reserve_bytes if scratch and host.cfg.custody_dir is not None else value
    return total


def reap(host: JudgeHost) -> None:
    """Remove the custody directory of a trusted build that ended without publishing.

    A finished build's `published/` tree is the real build and counts in `run_bytes`; a failed or cancelled
    build's partial output would otherwise occupy the disk after its reservation was released.
    """
    root = (host.jdir / "builds").resolve()
    for job in host.sched.jobs.values():
        if job.build_private is None or not job.state.terminal or job.state is State.done:
            continue  # only a failed or cancelled build left nothing to keep
        try:
            private = Path(job.build_private).resolve()
        except OSError:
            continue
        if private.parent != root or not private.is_dir():
            continue
        if (private / "receipt.json").is_file() or (private / "published").is_dir():
            continue  # a completed build (its receipt, record and logs stay)
        if (private / "frozen").is_dir():
            continue  # a capture is running: the build's own output is still being sealed
        _mirror(host, private)
        remove_tree(private)


def prune(host: JudgeHost) -> list[dict]:
    """Remove the weight files of scored builds when the next build would not fit above the floor.

    Nothing is freed while one more build's output fits above the free-space floor (`_short_of_room`).
    Below that room the pass first deletes the frozen copies nothing reads (`reclaim`), then removes whole
    the weights of the scored builds outside the keep set, the worst KL first, until the next build fits
    again or no such build is left. The disk cap's rule (`_at_cap`, see the module) is separate.

    Returns:
        The prune records appended to the ledger (empty when nothing was pruned).
    """
    found = builds(host)
    try:
        keep = _in_use(host, found) | _pending(host) | _frozen_keep(host) | _best_kl(host)
        stop = _final_dir(host)
    except Exception:  # noqa: BLE001 -- an unsure judge keeps every build (the pass is logged by the tick)
        _log.exception("build pruning could not decide what to keep; nothing was pruned")
        return []
    # a build refused at intake has a verdict, so it ranks like a refused reading; one with no finished
    # reading or check is not scored yet: keep it (but `_at_cap`)
    done = _done(host) | _intake(host)
    freed: dict[tuple[int, int], int] = {}  # one entry per inode whose last link this pass removes
    records = []
    if _short_of_room(host):
        # a build pruned in an earlier pass whose frozen copies were then still in use: finish that prune
        left = {i for i in done if i not in keep and all(b.identity != i for b in found)}
        for identity in sorted(left):
            before = sum(freed.values())
            frozen = _prune_frozen(host, identity, freed, stop)
            if frozen:
                record = {
                    "kind": "prune",
                    "destination": "",
                    "job": "",
                    "at": host.clock.now(),
                    "utc": utc(host.clock.now()),
                    "bytes": sum(freed.values()) - before,
                    "files": [],
                    "frozen": frozen,
                }
                _append(host, record)
                records.append(record)
        reclaim(host, stop)  # the copies nothing reads go before any build's weights
        kl = _kl_scores(host)
        worst = sorted(
            (b for b in found if b.identity in done and b.identity not in keep),
            key=lambda b: (kl.get(b.identity, math.inf), b.destination),
            reverse=True,
        )
        for b in worst:
            if not _short_of_room(host):
                break
            before = sum(freed.values())
            try:
                record = _prune_one(host, b, freed, stop, before)
            except OSError:  # noqa: BLE001 -- a half-pruned build is kept as it is; the next pass retries
                _log.exception("build %s could not be pruned", b.destination)
                continue
            if record is not None:
                records.append(record)
    return records + _at_cap(host, keep, done, stop, freed)


def _at_cap(
    host: JudgeHost, keep: set[str], done: set[str], stop: Path | None, freed: dict[tuple[int, int], int]
) -> list[dict]:
    """Free unscored builds while the run is over its disk cap, until it is back under (the module's rule).

    `keep` holds every build in use or with a reading in flight; a build ever handed in and the newest
    unscored build are never freed either. Nothing is freed when all of the others together would not get
    the run under the cap.
    """
    if not _on_disk_over(host):
        _STUCK.pop(host, None)
        return []
    handed, screened = _handed(host), _screened(host)
    unscored = sorted(
        (b for b in builds(host) if b.identity not in done),  # this pass may have pruned some already
        key=lambda b: (_published_at(b), b.destination),
    )
    held = keep | handed | ({unscored[-1].identity} if unscored else set())
    spare = [b for b in unscored if b.identity not in held]
    weights: dict[tuple[int, int], int] = {}  # what freeing them all could give back, each inode once
    for b in spare:
        for path in weight_files(b.tree):
            try:
                info = path.lstat()
            except OSError:
                continue  # gone already: nothing to give back
            weights[(info.st_dev, info.st_ino)] = info.st_size
    if sum(weights.values()) < _excess(host):
        stuck = (
            host.cfg.max_run_bytes,
            host.cfg.max_scratch_bytes,
            tuple(sorted(b.destination for b in spare)),
        )
        if (
            _STUCK.get(host) != stuck
        ):  # every tick prunes: say it once until the cap or the spare builds change
            _STUCK[host] = stuck
            _log.warning("over the disk cap, and freeing every unscored build would not get the run under it")
        return []
    _STUCK.pop(host, None)
    spare.sort(key=lambda b: b.identity in screened)  # stable: oldest first within each group
    records = []
    for b in spare:
        if _excess(host) <= 0:
            break
        try:
            record = _prune_one(host, b, freed, stop, sum(freed.values()), reason=AT_CAP)
        except OSError:  # noqa: BLE001 -- a half-pruned build is kept as it is; the next pass retries
            _log.exception("build %s could not be freed at the disk cap", b.destination)
            continue
        if record is not None:
            _log.warning("over the disk cap: freed unscored build %s", b.destination)
            records.append(record)
    return records


def _intake(host: JudgeHost) -> set[str]:
    """Identities of builds whose candidate copy was refused at intake: a verdict, so prunable as refused."""
    return set(host.intake_refused)


def _handed(host: JudgeHost) -> set[str]:
    """Identities of every build ever handed in (checked, refused or still open)."""
    return {i for e in host.entries if e.kind == "handin" and (i := _entry_identity(e)) is not None}


def _screened(host: JudgeHost) -> set[str]:
    """Identities with a finished screening reading: ranked for a full reading, so freed last at the cap."""
    return {
        i
        for e in host.entries
        if e.kind != "gencheck" and e.stage == "screen" and e.valid is not None and (i := _entry_identity(e))
    }


def _published_at(b: Build) -> float:
    """When a build was published (its receipt's write time); 0 when unreadable, so it goes first."""
    try:
        return (b.private / "receipt.json").stat().st_mtime
    except OSError:
        return 0.0


def _frozen_keep(host: JudgeHost) -> set[str]:
    """The best-KL build identities a cutoff final rule reads from their frozen copies after the run.

    A final rule that decides at cutoff from fresh generation checks reads its candidates then;
    `frozen_keep` is how many of the best it needs. Such a rule ranks by KL whatever `judge.metric` names,
    so the lowest-KL builds are kept (their copies whole, `reclaim`) whichever build a prune pass frees
    first.
    """
    need = max((rule.frozen_keep() for rule in host.cfg.final_rules), default=0)
    if not need:
        return set()
    kl = _kl_scores(host)
    return set(sorted(kl, key=lambda i: kl[i])[:need])


def _counts(e: Entry) -> bool:
    """Whether a reading is a score: valid, not ineligible (judge (b) adds `eligible`), and a full one.

    A screening reading (in a ledger an older, screening search wrote) only ranked a candidate for its full
    reading; it decides no build's custody, so the keep set is ranked by full readings only.
    """
    return e.valid is True and getattr(e, "eligible", True) is not False and e.stage != "screen"


def _pending(host: JudgeHost) -> set[str]:
    """Identities with a reading or check still in flight: a build the judge is not sure of yet."""
    out: set[str] = set()
    for e in host.entries:
        if e.kind != "gencheck" and e.valid is None and (identity := _entry_identity(e)) is not None:
            out.add(identity)
    return out


def _done(host: JudgeHost) -> set[str]:
    """Identities with a finished full reading or check (a valid or a refused one): scored, so rankable.

    A build whose only finished reading is a screening reading is not rankable yet: its full reading may be
    queued, and its frozen copy must survive until it is.
    """
    out: set[str] = set()
    for e in host.entries:
        if (
            e.kind != "gencheck"
            and e.valid is not None
            and e.stage != "screen"
            and (identity := _entry_identity(e)) is not None
        ):
            out.add(identity)
    return out


def _kl_scores(host: JudgeHost) -> dict[str, float]:
    """The best valid KL reading per build identity (the frozen-copy ranking of a cutoff rule)."""
    out: dict[str, float] = {}
    for e in host.entries:
        if e.kind == "gencheck" or not _counts(e):
            continue
        value = e.metrics.get("kl")
        identity = _entry_identity(e)
        if identity is None or not isinstance(value, float) or not math.isfinite(value):
            continue
        if identity not in out or value < out[identity]:
            out[identity] = value
    return out


def kept_handins(host: JudgeHost) -> set[str]:
    """The ids of the hand-ins whose copies stay whole: every one not refused, for the rest of the run.

    A hand-in is the agent's submission; deleting it would delete a result the final rule may still pick,
    and a re-hand-in of it must not depend on the agent's copy still being unmodified. Its copy is
    therefore neither reclaimed nor stripped (`reclaim`, `_prune_frozen`), even when it is not the current
    pick. A refused hand-in is no result: its copy may go like any other. The current final pick is kept
    separately (`_in_use`, `reclaim`'s `stop`).
    """
    return {
        e.id for e in host.entries if e.kind == "handin" and e.valid is not False and _entry_identity(e) is not None
    }


def _in_use(host: JudgeHost, found: list[Build]) -> set[str]:
    """Identities still needed: pending jobs, open and kept hand-ins, the current final pick, candidate 0."""
    at: dict[str, set[str]] = {}
    for b in found:
        at.setdefault(b.destination, set()).add(b.identity)
    out: set[str] = set()
    # `reserved`: a hand-in is freezing that publication right now (its entry is not recorded yet)
    for destination in _pending_destinations(host) | set(host.reserved):
        out |= at.get(destination, set())
    out |= _baselines(host, found)  # candidate 0 and the other baselines as the agent's jobs read them
    kept = kept_handins(host)  # every hand-in that is not refused: its publication and copy stay
    for e in host.entries:
        if e.kind != "handin" or (identity := _entry_identity(e)) is None:
            continue
        if e.valid is None or e.id in kept:
            out.add(identity)
    final = None
    if _candidate0_read(host):
        try:
            final = host.final(generation_checks=False)
        except Exception:  # noqa: BLE001 -- no final pick is known; the other keep rules still protect
            _log.exception("the final pick could not be read while pruning; the other keep rules still apply")
    if final is not None and (choice := final.choice) is not None:
        for e in host.entries:
            if e.dir == choice.dir and (identity := _entry_identity(e)) is not None:
                out.add(identity)
    return out


def _baselines(host: JudgeHost, found: list[Build]) -> set[str]:
    """Identities of the builds published at candidate 0's or another baseline's path."""
    bases = {Path(os.path.normpath(b)) for b in host.cfg.baselines}
    return {b.identity for b in found if Path(os.path.normpath(host.workdir / b.destination)) in bases}


def _candidate0_read(host: JudgeHost) -> bool:
    """Whether candidate 0's KL is known, or never waited for: False only before the judge's own reading of it.

    With `judge.select_candidate0` or `judge.starts` the judge reads its baselines at start and records
    candidate 0's KL only when that reading is folded (`startup.select_baselines`). Until then no pick exists
    to keep, and a prune pass names none. The test is the value itself, not a clock or a phase; a run that configures `candidate0_kl`
    (or does not select candidate 0) has it from the start and is never skipped.
    """
    selecting = host.cfg.select_candidate0 or bool(host.cfg.starts)
    return not selecting or host.cfg.candidate0_kl is not None


def _pending_destinations(host: JudgeHost) -> set[str]:
    """Destinations a queued or running job reads, named in its argv, its cwd, or a snapshotted file."""
    jobs = host.sched.jobs
    refs: list[Path] = []
    texts: list[str] = []
    out: set[str] = set()
    for job in jobs.values():
        if job.state.terminal:
            continue
        for used in _job_tree(job, jobs):
            out |= _own_destination(used)
            refs.append(Path(used.cwd))
            refs += [p for a in used.argv if (p := _token_path(used.cwd, a)) is not None]
            texts += _job_texts(host, used)
    workdir = Path(os.path.normpath(host.workdir))
    for b in builds(host):
        if _read_by(b.destination, refs, workdir) or _mentioned(b.destination, texts):
            out.add(b.destination)
    return out


def _job_tree(job, jobs) -> list:
    """`job` and every dependency of it, each once."""
    seen, order = set(), []
    stack = [job]
    while stack:
        j = stack.pop()
        if j.name in seen:
            continue
        seen.add(j.name)
        order.append(j)
        stack += [jobs[n] for n in j.after if n in jobs]
    return order


def _own_destination(job) -> set[str]:
    """The destination a trusted build job writes; the previous publication there is still needed."""
    if job.build_private is None:
        return set()
    spec = read_json(Path(job.build_private) / "request.json") or {}
    destination = spec.get("destination")
    return {Path(destination).as_posix()} if isinstance(destination, str) and destination else set()


def _job_texts(host: JudgeHost, job) -> list[str]:
    """The snapshotted script or config a job runs: a build's config, or a queued job's script files."""
    out: list[str] = []
    if job.build_private is not None:
        spec = read_json(Path(job.build_private) / "request.json") or {}
        config = spec.get("config")
        if isinstance(config, str):
            out += _text(Path(config))
        overrides = spec.get("overrides")
        if isinstance(overrides, list):
            out.append(" ".join(str(x) for x in overrides))
    files = host.sched.dir / job.id / "files"
    if files.is_dir():
        for path in walk(files):
            if path.is_file() and not path.is_symlink():
                out += _text(path)
    return out


def _text(path: Path) -> list[str]:
    """The text of one snapshotted file, or nothing when it is missing or too large to scan."""
    try:
        if path.stat().st_size > MAX_SNAPSHOT:
            return []
        return [path.read_text(errors="replace")]
    except OSError:
        return []


def _mentioned(destination: str, texts: list[str]) -> bool:
    """Whether a snapshotted script or config names the destination, by path or by directory name."""
    name = Path(destination).name
    if not name:
        return False
    pattern = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])")
    return any(destination in text or pattern.search(text) for text in texts)


def _token_path(cwd: str, token: str) -> Path | None:
    """The path an argv token names, relative to the job's cwd; None for a flag carrying no value."""
    name, sep, value = token.partition("=")
    text = value if sep and name.startswith("-") else token
    if not text or text.startswith("-"):
        return None
    path = Path(text)
    if not path.is_absolute():
        path = Path(cwd) / path
    return Path(os.path.normpath(path))


def _read_by(destination: str, refs: list[Path], workdir: Path) -> bool:
    """Whether a job path names the destination, sits inside it, or is a parent directory of it.

    A reference to a parent counts: a job that reads `out/` reads every build published under it. The
    workdir itself does not count, or every job's cwd would protect every build.
    """
    dest = Path(os.path.normpath(workdir / destination))
    for ref in refs:
        if ref == dest or ref.is_relative_to(dest):
            return True
        if dest.is_relative_to(ref) and ref != workdir:
            return True
    return False


def _identity(receipt: dict) -> str | None:
    """A build's identity: its receipt's whole-tree digest, or the job id when the receipt predates it."""
    tree = receipt.get("tree_sha256")
    if isinstance(tree, str) and tree:
        return tree
    job = receipt.get("job_id")
    return f"job:{job}" if isinstance(job, str) and job else None


def _entry_identity(e: Entry) -> str | None:
    """The identity of the build an entry was taken from: its receipt's, else its frozen copy's own digest.

    The two agree for a receipted build (its frozen copy holds the receipted tree), so an entry without a
    receipt (`judge.require_build_receipt: false`, or no build matched) is ranked and kept by the same rules.
    """
    return _identity(e.build_receipt) if e.build_receipt else (e.digest or None)


def _prune_one(
    host: JudgeHost,
    b: Build,
    freed: dict[tuple[int, int], int],
    stop: Path | None,
    before: int,
    *,
    reason: str | None = None,
) -> dict | None:
    """Rename one publication out of the mount list, remove its weight links, and record what went.

    The record is written after the build's frozen copies lose their weight links too, so `bytes` is what
    this build's removals freed (`before` is the pass total before it): only a link whose removal left the
    inode with none counts, once. `reason` is recorded when the build was freed at the disk cap (`AT_CAP`).
    """
    root = (host.jdir / "builds").resolve()
    private, tree = b.private.resolve(), b.tree.resolve()
    if private.parent != root or tree.parent != private or not tree.is_dir():
        return None  # never touch anything outside the run's custody
    files = weight_files(tree)
    if not files:
        return None
    target = private / PRUNED
    if target.exists() or target.is_symlink():
        return None  # never overwrite an earlier pruned tree
    tree.rename(target)  # unmounted at once: no sandbox can read a half-pruned tree
    gone: list[str] = []
    for path in files:
        moved = target / path.relative_to(tree)
        try:
            _unlink(moved, freed)
        except OSError:  # a file that cannot be removed stays, and the record lists what went
            _log.exception("build %s: could not remove %s", b.destination, moved)
            continue
        gone.append(moved.relative_to(target).as_posix())
    frozen = _prune_frozen(host, b.identity, freed, stop)
    record = {
        "kind": "prune",
        "destination": b.destination,
        "job": b.job,
        "at": host.clock.now(),
        "utc": utc(host.clock.now()),
        "bytes": sum(freed.values()) - before,
        "files": gone,
        "frozen": frozen,
    }
    if reason is not None:
        record["reason"] = reason
    (private / RECORD).write_text(json.dumps(record) + "\n")
    _mirror(host, private)
    _append(host, record)
    return record


def _prune_frozen(
    host: JudgeHost, identity: str, freed: dict[tuple[int, int], int], stop: Path | None
) -> list[str]:
    """Remove one pruned build's weight links from its frozen copies no job needs and the final pick is not.

    A frozen copy of a hand-in or a reading hardlinks the custody tree. Removing only the published names
    frees no disk while those links remain, so a terminal entry of a pruned build loses its weight files
    too. A pending check or reading keeps them, and so does the final pick.
    """
    touched: list[str] = []
    frozen = (host.jdir / "frozen").resolve()
    for e in host.entries:
        if e.valid is None or e.kind == "gencheck":
            continue
        if _entry_identity(e) != identity or (stop is not None and e.dir == stop):
            continue
        tree = e.dir.resolve()
        if tree.parent != frozen or not tree.is_dir():
            continue
        files = weight_files(tree)
        if not files:
            continue
        for path in files:
            try:
                _unlink(path, freed)
            except OSError:
                _log.exception("frozen copy %s: could not remove %s", e.id, path)
        touched.append(e.id)
    return touched


def drop_frozen(host: JudgeHost, entry: Entry) -> bool:
    """Remove `entry`'s frozen copy when no other entry or in-flight copy still holds it; True if removed.

    Call with `entry` already out of `host.entries` (`jobs._discard`, `commands.gencheck`) or with it there
    (a reclaim after a verdict): the copy is content-addressed, so another entry of the same checkpoint may
    still point at it, and a `take` may be freezing it. Nothing that reads it later (a queued or running
    check, the final pick) is consulted here: only `reclaim` decides on a copy an entry still points at.
    """
    tree = _norm(entry.dir)
    held = any(same_dir(e.dir, tree) for e in host.entries if e is not entry)
    if held or tree in {_norm(p) for p in host.freezing}:
        return False
    remove_tree(tree)
    return True


def _norm(path: Path) -> Path:
    """`path` normalized without following links, as every frozen-copy comparison here is made."""
    return Path(os.path.normpath(str(path)))


def reclaim(host: JudgeHost, stop: Path | None = None) -> list[dict]:
    """Delete every frozen copy whose readings have landed and that nothing still needs; one record each.

    `prune` calls this when the next build would not fit above the free-space floor, before it frees any
    build's weights. A frozen copy is a copy, not a link: a search that was read, handed in and checked
    holds a copy per call unless `intake.take` reused the one of its digest. The score is kept elsewhere
    (the ledger entry and the reading's own result file), so the bytes are needed only while something reads
    them. A copy is kept when:

    - one of its entries has no verdict yet (a reading or check queued or running, `_pending`) or is a
      screening reading whose full reading may be queued next;
    - one of its entries is a hand-in (`kept_handins`: every handed-in build's copy stays whole, never
      left without its weights);
    - one of its entries is one of the best-KL builds a cutoff final rule reads (`_frozen_keep`);
    - one of its entries is the best-scored build by KL (`_best_kl`): the disk-space fallback and the
      final rules submit its reading when the agent's hand-in cannot count;
    - it is the final pick (`stop`);
    - a `take` is freezing it right now (`host.freezing`).

    The copies nothing needs go worst KL first, and only until the next build fits above the floor again
    (`_short_of_room`), so the best remaining copies are the last to go. The entry lines stay: a reclaimed
    entry's `dir` names a copy that is gone, and its score, reason and record are what the agent reads
    (`kl list`, `handin status`). A later `take` of the same files freezes again (`intake.frozen_for` reuses
    only a directory that still exists).

    Returns:
        The `reclaim` records appended to the ledger (empty when nothing was reclaimed).
    """
    root = _norm(host.jdir / "frozen")
    if not root.is_dir():
        return []
    keep = _frozen_keep(host)
    kept = kept_handins(host)
    best = _best_kl(host)
    needed = {_norm(p) for p in host.freezing}
    if stop is None:
        stop = _final_dir(host)
    if stop is not None:
        needed.add(_norm(stop))
    for e in host.entries:
        tree = _norm(e.dir)
        if tree.parent != root:
            continue
        identity = _entry_identity(e)
        if (
            e.valid is None  # a reading or check queued or running
            or e.stage == "screen"  # its full reading may be queued next
            or e.id in kept  # a hand-in the rule can pick
            or (identity is not None and (identity in keep or identity in best))
        ):
            needed.add(tree)
    spare: list[tuple[float, Path, list[str]]] = []
    for tree in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        if _norm(tree) in needed:
            continue
        # worst KL first: a copy whose entries have no KL is the worst (nothing can fall back to it)
        readings = [
            e.metrics["kl"]
            for e in host.entries
            if same_dir(e.dir, tree) and isinstance(e.metrics.get("kl"), float)
        ]
        spare.append((-min(readings, default=math.inf), tree, [e.id for e in host.entries if same_dir(e.dir, tree)]))
    spare.sort(key=lambda item: (item[0], item[1].name))
    records = []
    for _kl, tree, entries in spare:
        if not _short_of_room(host):
            break
        size = tree_size(tree)
        try:
            remove_tree(tree)
        except OSError:
            _log.exception("frozen copy %s could not be removed", tree.name)
            continue
        record = {
            "kind": "reclaim",
            "frozen": tree.name,
            "entries": entries,
            "bytes": size,
            "at": host.clock.now(),
            "utc": utc(host.clock.now()),
        }
        _append(host, record)
        records.append(record)
    return records


def _best_kl(host: JudgeHost) -> set[str]:
    """The identities of the best-scored builds by KL (the lowest reading; ties all kept)."""
    kl = _kl_scores(host)
    if not kl:
        return set()
    best = min(kl.values())
    return {identity for identity, value in kl.items() if value == best}


def _final_dir(host: JudgeHost) -> Path | None:
    """The directory of the current final pick, when a rule decided already."""
    try:
        final = host.final(generation_checks=False)
    except Exception:  # noqa: BLE001 -- an unknown pick prunes no frozen copy
        return None
    return None if final.choice is None else final.choice.dir


def is_whole(tree: Path) -> bool:
    """Whether frozen copy `tree` still loads as far as a prune pass is concerned: present, with weight files.

    A pass removes a copy's weight files and nothing else (`weight_files`), so a copy without them is one it
    stripped; the `handin` rule never picks one (`final_rules.Standing.whole`).
    """
    return tree.is_dir() and not tree.is_symlink() and bool(weight_files(tree))


def weight_files(tree: Path) -> list[Path]:
    """The weight files under `tree` (symlinks are listed, never followed): what a pass removes from a copy.

    A pruned copy differs from a whole one only by these, so `intake.frozen_for` reads them to tell the two
    apart.
    """
    return [p for p in walk(tree) if p.is_file() and not p.is_symlink() and p.suffix in WEIGHTS]


def _unlink(path: Path, freed: dict[tuple[int, int], int]) -> None:
    """Unlink one read-only weight file, recording its size only when this removes the inode's last link.

    The inode is read before the unlink: after it, the file is gone. A hardlinked frozen copy shares the
    inode with the custody tree, so a size recorded while another link remains would count bytes that are
    still on disk, and the ledger must sum to what was actually freed.
    """
    info = path.lstat()  # before the unlink: its last link removes the path
    parent = path.parent
    mode = stat.S_IMODE(parent.stat().st_mode)
    parent.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
    try:
        path.unlink()
    finally:
        parent.chmod(mode)
    if info.st_nlink == 1:
        freed[(info.st_dev, info.st_ino)] = info.st_blocks * 512  # the disk blocks the last link held


def note_space_refusal(
    host: JudgeHost, build: str, error: Refused, *, what: str = "hand-in"
) -> dict[str, object]:
    """Record a hand-in refused (or, `what="build"`, a running job stopped) for disk space, and say it.

    Disk space is not a verdict on the checkpoint: the refusal is appended to `judge/ledger.jsonl` (a
    `space_refusal` record, beside the prune records) and announced, so the final rules can submit the best
    scored build instead of falling back to candidate 0 (`final_rules.Handin`). One at the free-space floor
    (`custody.FLOOR`) is said in two lines: what the disk holds and what was stopped or refused, then the
    fallback.
    """
    floor = str(error).startswith(FLOOR)
    scratch = str(error).startswith("scratch ")
    held = run_bytes(host, scratch=scratch)
    cap = host.cfg.max_scratch_bytes if scratch else host.cfg.max_run_bytes
    now = host.clock.now()
    record: dict[str, object] = {
        "kind": "space_refusal",
        "build": build,
        "held": held,
        "cap": None if floor else cap,
        "at": now,
        "utc": utc(now),
        "reason": str(error),
    }
    if what != "hand-in":
        record["what"] = what
    with host.lock:
        host.space_refusals.append(record)
    _append(host, record)
    if floor:
        done = f"stopped {build}" if what == "build" else f"hand-in of {build} refused"
        host.announce(f"{FLOOR}: {done}: {str(error).removeprefix(FLOOR + ': ')}")
        host.announce(FLOOR_SAID)
    else:
        host.announce(
            f"hand-in of {build} refused for disk space: {_gb(held)} used of the {_gb(cap or 0)} cap; "
            "the run will submit the best scored eligible build instead"
        )
    return record


def _append(host: JudgeHost, record: dict) -> None:
    """Append one record (a prune or a space refusal) to the judge's ledger."""
    with (host.jdir / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _mirror(host: JudgeHost, private: Path) -> None:
    """Keep the durable build records before pruning or reaping local weights."""
    if host.cfg.custody_dir is not None:
        from .custody import mirror

        spec = read_json(private / "request.json") or {}
        config = spec.get("config")
        mirror(private, host.jdir / "records" / "builds" / private.name, Path(config) if config else None)
