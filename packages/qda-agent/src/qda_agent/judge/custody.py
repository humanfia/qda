"""Host-owned checkpoint storage outside the durable run directory."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from ..room import room
from .config import MIN_FREE_BYTES
from .freeze import tree_size

if TYPE_CHECKING:
    from .config import JudgeSettings

#: How many copies of the original model's bf16 weights a disk keeps free when `judge.min_free_bytes` keeps
#: its default: one build being written and one being read, whatever the disk's size.
MODEL_COPIES = 2


def validate(
    path: Path, forbidden: Iterable[Path], *, setting: str = "judge.custody_dir", fresh: bool = True
) -> Path:
    """Require a fresh directory below trusted existing ancestors, apart from all run inputs.

    Root-owned sticky temporary directories are safe parents: another user cannot replace the host's
    entry there. A writable non-sticky ancestor or a foreign-owned ancestor is not safe. `setting` names the
    setting in a refusal (the agent's `work_dir` is placed by the same rules). With `fresh` false an existing
    directory passes (its launch checked it was new); a link never does.
    """
    path = path.absolute()
    if path.is_symlink() or (path.exists() and (fresh or not path.is_dir())):
        raise ValueError(f"{setting} ({path}): already exists; give a fresh per-run directory")
    for parent in path.parents:
        try:
            info = parent.lstat()
        except OSError as error:
            raise ValueError(f"{setting} ({path}): unreadable ancestor {parent}: {error}") from error
        mode = info.st_mode
        if not stat.S_ISDIR(mode) or info.st_uid not in (0, os.getuid()):
            raise ValueError(f"{setting} ({path}): untrusted ancestor {parent}")
        if mode & 0o022 and not (info.st_uid == 0 and mode & stat.S_ISVTX):
            raise ValueError(f"{setting} ({path}): writable ancestor {parent}")
    path = path.resolve()
    for entry in forbidden:
        other = entry.resolve()
        if path.is_relative_to(other) or other.is_relative_to(path):
            raise ValueError(f"{setting} ({path}): overlaps run input or bind {other}")
    return path


def place(path: Path, jdir: Path, forbidden: Iterable[Path]) -> None:
    """Create custody at 0700 and host-owned links used by the existing receipt and snapshot paths."""
    path = validate(path, forbidden)
    path.mkdir(mode=0o700)
    jdir.mkdir(parents=True, exist_ok=True)
    for name in ("builds", "frozen"):
        (path / name).mkdir(mode=0o700)
        (jdir / name).symlink_to(path / name, target_is_directory=True)


def required_bytes(build_bytes: int, cards: int, handin_slots: int = 1) -> int:
    """The bytes a search must be able to hold at once: what `judge.max_run_bytes` has to cover.

    Each term is one retention rule of `judge.prune._handed`/`_in_use` or one area `judge.prune.run_bytes`
    counts:

    - `handin_slots`: the frozen copies held at once, at least one: an unscored hand-in is a full copy
      (`prune._in_use` keeps every hand-in whose check has not finished, and no frozen copy of its bytes
      exists to hardlink yet), plus the best builds a cutoff final rule reads after the run
      (`FinalRule.frozen_keep`, `prune._frozen_keep`, added to the keep set, so summed);
    - `cards`: one build being written per scheduler card, in the agent's `work/` (which `run_bytes` counts);
    - one more: a copy in flight, being written or sealed before it joins one of the trees above.

    A handed-in build's published tree stays whole for the rest of the run (`prune._handed`), so a run that
    hands in many builds holds them all; this bound does not promise a number of them. A hand-in list of
    several directories freezes one copy per directory while it is checked; the judge does not bound that
    list, so this counts the first (the common list) and the per-copy `check_room` refuses the rest rather
    than promising them. This is an upper bound: a hardlinked hand-in holds no extra bytes.
    """
    return build_bytes * (handin_slots + cards + 1)


def build_bytes(judge: JudgeSettings) -> int | None:
    """Bytes one build of this run's model holds, as the config states it; None: the config names no size.

    A llama.cpp search refuses a build over `judge.budget_bytes`, so that is one build. Otherwise candidate 0
    is the run's measured build (`judge.format_reference`, candidate 0's directory); without one, no build size
    can be read from the config alone.
    """
    if judge.engine == "llama.cpp" and judge.budget_bytes is not None:
        return judge.budget_bytes
    if judge.format_reference is not None and judge.format_reference.is_dir():
        return tree_size(judge.format_reference)
    return None


def handin_slots(judge: JudgeSettings) -> int:
    """The frozen copies the run holds at once (`required_bytes`'s `handin_slots`).

    One is the hand-in's own copy: an unscored hand-in is fully frozen (`prune._in_use` keeps it while its
    check runs), and it has no earlier frozen copy to hardlink. The cutoff rules' best builds add their own
    (`FinalRule.frozen_keep`): `prune._frozen_keep` keeps those copies too, so the counts sum rather than
    one covering the other.
    """
    return 1 + max((rule.frozen_keep() for rule in judge.final_rules), default=0)


@dataclass(frozen=True)
class CapProblem:
    """A disk refusal: `detail` says what does not fit, `fix` is the one change that makes it fit.

    `agent` is the same refusal as a refused agent command shows it: `detail` without the operator's fix
    (that fix names disks, `--out` and `judge.min_free_bytes`, which the agent can neither move nor set).
    """

    detail: str
    fix: str
    agent: str | None = None

    def __str__(self) -> str:
        return f"{self.detail}; {self.fix}"


#: What `--out` on a larger disk moves, as every floor refusal's fix names it.
MOVE = "move run_dir, judge.custody_dir or the agent's work_dir"

#: How every free-space floor refusal and stop begins: `prune.note_space_refusal` tells it from a cap's.
FLOOR = "DISK FLOOR"


@lru_cache(maxsize=8)
def weight_bytes(model: Path) -> int | None:
    """Bytes of a model directory's `*.safetensors` (links followed, as a Hub snapshot keeps them); None when
    there are none. Read once per directory: the floor is checked on every scheduler pass."""
    try:
        return sum(p.stat().st_size for p in model.glob("*.safetensors")) or None
    except OSError:
        return None


def _weights(judge: JudgeSettings, weights: int | None) -> int | None:
    if weights is None and judge.anchor is not None:
        return weight_bytes(Path(judge.anchor).absolute())
    return weights


def free_floor(judge: JudgeSettings, weights: int | None = None) -> int:
    """The free-space floor in bytes; 0: no floor.

    One owner of the rule: `floor_problem` (hence `check_space`, `qda doctor --search`, `qda-agent prepare`,
    every admission and every scheduler pass) reads `judge.min_free_bytes` only here. A value set by hand is
    used as is and 0 turns the floor off; the default is `MODEL_COPIES` x the original model's weights
    (`weights`, else those of `judge.anchor`), and `MIN_FREE_BYTES` when neither names a model.
    """
    if judge.min_free_bytes <= 0:
        return 0
    if "min_free_bytes" in judge.model_fields_set:
        return judge.min_free_bytes
    weights = _weights(judge, weights)
    return MODEL_COPIES * weights if weights else MIN_FREE_BYTES


def floor_problem(
    paths: Iterable[Path], judge: JudgeSettings, weights: int | None = None
) -> CapProblem | None:
    """The run's free-space floor: None while every disk under `paths` has its floor free, else the first not.

    The one check of `judge.min_free_bytes` (`free_floor`, the same function `qda doctor --search` reads). Free
    space is measured live with `room()` (one `statvfs`, no walk), never estimated: the judge's start and `qda
    doctor --search` (`space_problem`, `check_space`), every admission (`prune.check_room`) and every scheduler
    pass (`prune.watch_floor`) call this function. `weights` sizes the default floor before the original
    model is on disk (`qda-agent prepare`).
    """
    floor = free_floor(judge, weights)
    for path in paths:
        free = room(path)
        if not floor or free.free_bytes >= floor:
            continue
        have = free.free_bytes / 1e9
        if "min_free_bytes" in judge.model_fields_set:
            detail = (
                f"{FLOOR}: {have:.1f} GB free on {free.path}, below the judge.min_free_bytes floor of "
                f"{floor / 1e9:.1f} GB"
            )
            fix = f"free space on that disk or use --out on a larger disk ({MOVE}), or lower "
            fix += "judge.min_free_bytes"
        else:
            why = (
                f"{MODEL_COPIES} copies of the bf16 model"
                if _weights(judge, weights)
                else "the default floor"
            )
            detail = f"{FLOOR}: needs at least {floor / 1e9:.1f} GB free ({why}), {free.path} has only {have:.1f} GB"
            fix = f"use --out on a larger disk ({MOVE}), or set judge.min_free_bytes by hand"
        return CapProblem(
            detail=detail,
            fix=fix,
            agent=f"{FLOOR}: {have:.1f} GB free on the run's own disk, below its free-space floor",
        )
    return None


def space_problem(
    path: Path,
    *,
    cap: int | None,
    build: int | None,
    cards: int,
    handin_slots: int = 1,
    max_files: int | None = None,
    judge: JudgeSettings | None = None,
    weights: int | None = None,
) -> CapProblem | None:
    """The run's own disk check: None when it fits, else what is wrong and the one fix.

    This is the one check of `judge.max_run_bytes`, read once with `room()` (no walk). `check_space` is it for
    a run's settings, which `JudgeHost` refuses to start on and `qda doctor --search` reports as a MISS;
    `qda-agent prepare` calls it directly with a default `JudgeSettings` and no cap (it has no run settings yet).

    - free bytes below the floor (`floor_problem`): the run would stop its first build (the only check when no
      cap is set);
    - a cap above the free bytes fills the disk before it stops a build; it is reported first, because
      a larger disk, not a larger cap, is its fix;
    - a cap below `required_bytes` refuses the builds and hand-ins the run needs: the hand-in's own and
      cutoff rules' frozen copies, one working copy per card and one in flight;
    - with `max_files`, `max_files * (handin_slots + 2 * cards)` inodes must be free: the frozen copies and
      two trees per scheduler card being built or read.

    Args:
        path: Where the run's builds land (the custody root, else `run_dir`).
        cap: `judge.max_run_bytes`; None: no cap, so only the floor and the inode peak are checked.
        build: One build's bytes (`build_bytes`); None: the config names no size, so the cap is not checked
            against the run.
        cards: The scheduler's cards, one working copy each.
        handin_slots: `required_bytes`'s frozen copies beyond the hand-in's own (`handin_slots(judge)`).
        max_files: `judge.max_files`; None: skip the inode check.
        judge: The settings whose `judge.min_free_bytes` is the floor (set by hand, else twice the original
            model's weights, `free_floor`); None: no floor.
        weights: The original model's weight bytes when `judge.anchor` is not on disk yet (`free_floor`).
    """
    if judge is not None and (problem := floor_problem([path], judge, weights)) is not None:
        return problem
    free = room(path)
    # The disk first: when it cannot hold the cap, moving the run to a larger disk is the fix, and raising
    # the cap (the fix below) would only make it worse.
    if cap is not None and cap > free.free_bytes:
        return CapProblem(
            detail=(
                f"judge.max_run_bytes is {cap / 1e9:.1f} GB, above the {free.free_bytes / 1e9:.1f} GB free on "
                f"{free.path}: the disk fills before the cap stops a build"
            ),
            fix=(
                "use --out on a larger disk (move run_dir or judge.custody_dir), or lower judge.max_run_bytes"
            ),
        )
    if cap is not None and build is not None:
        need = required_bytes(build, cards, handin_slots)
        if cap < need:
            return CapProblem(
                detail=(
                    f"judge.max_run_bytes is {cap / 1e9:.1f} GB, below the {need / 1e9:.1f} GB the run must "
                    f"hold ({handin_slots} frozen copies (the hand-in's own and cutoff rules') + {cards} card "
                    f"working copies + 1 in flight, one {build / 1e9:.2f} GB build)"
                ),
                fix=f"--set judge.max_run_bytes={need}",
            )
    inodes = None if max_files is None else max_files * (handin_slots + 2 * cards)
    if inodes is not None and free.free_inodes < inodes:
        return CapProblem(
            detail=(
                f"local storage has {free.free_inodes} free inodes on {free.path}, below the {inodes} the run "
                f"needs (judge.max_files {max_files} * ({handin_slots} frozen copies + 2 * scheduler cards "
                f"{cards}))"
            ),
            fix="free local storage or reduce judge.max_files and the run bounds",
        )
    return None


class DiskRefused(ValueError):
    """`check_space`'s refusal: `str()` is the whole line, `problem` its detail and fix separately."""

    def __init__(self, label: str, path: Path, problem: CapProblem) -> None:
        super().__init__(f"{label} ({path}): {problem}")
        self.problem = problem


def check_space(path: Path, judge: JudgeSettings, cards: int) -> None:
    """The run's own disk check, for every run: `space_problem` as an exception.

    `JudgeHost` runs it at start, on the custody root when the run keeps local custody and on `run_dir`
    otherwise (a run without custody fills the same kinds of trees, `prune.run_bytes` counts them either way);
    `qda doctor --search` calls this same function, so a MISS and a refusal cannot disagree. The free-space
    floor (`judge.min_free_bytes`; when it keeps its default, twice the original model's bf16 weights,
    `free_floor`) is checked for every run; the cap and the inode peak only when `judge.max_run_bytes` is set,
    as before: without a cap, the floor is the only disk guard.

    Raises:
        DiskRefused: A `ValueError` whose text names the setting and path, then the check's detail and fix.
    """
    label = "judge.custody_dir" if judge.custody_dir is not None else "run_dir"
    if (
        problem := space_problem(
            path,
            cap=judge.max_run_bytes,
            build=build_bytes(judge),
            cards=cards,
            handin_slots=handin_slots(judge),
            max_files=judge.max_files if judge.max_run_bytes is not None else None,
            judge=judge,
        )
    ) is not None:
        raise DiskRefused(label, path, problem)


def mirror(private: Path, records: Path | None, config: Path | None = None) -> None:
    """Persist small custody records at each transition, including builds never submitted for scoring."""
    if records is None:
        return
    records.mkdir(parents=True, exist_ok=True)
    files = [private / n for n in ("request.json", "launch.json", "receipt.json", "pruned.json")]
    if config is not None:
        files.append(config)
    for tree in (
        private,
        *(private / n for n in ("scratch/output", "output", "frozen", "published", "pruned")),
    ):
        files.extend(tree / n for n in ("build.json", "resolved.yaml"))
    for source in files:
        if not source.is_file() or source.is_symlink():
            continue
        if source.stat().st_size > 1 << 20:
            raise ValueError(f"custody record {source.name} exceeds 1 MiB")
        target = records / source.name
        stage = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, stage)
        with stage.open("rb") as stream:
            os.fsync(stream.fileno())
        stage.replace(target)
