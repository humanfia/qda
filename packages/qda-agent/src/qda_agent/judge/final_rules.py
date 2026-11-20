"""Final rules (`qda.final_rules`): which checkpoint a search ends on.

`judge.final_rules` names the rules to try, in order; the first that decides makes FINAL. A run that names
none uses `handin`, the documented contract: the result is the agent's own hand-in. Each `handin` call is one
ordered list (best first); the first directory of the latest list that passed its check counts, and a list
none of whose directories passed falls back to the latest earlier list that has one. The judge adds no check
and does not rank the scored builds; the agent decides, and `gencheck` is its tool, never forced. Only when
no valid hand-in exists does the fallback choose candidate 0 as-is, as in the paper. FINAL.json's `chosen_by`
records which path decided (`handin` or `fallback_candidate0`).

Disk space never picks the checkpoint: a hand-in refused for space (not for what it is), or a build stopped at
the free-space floor (`prune.watch_floor`), is recorded (`prune.note_space_refusal`), and `Handin` then submits
the best reading of the hand-in list, the scored, eligible builds and candidate 0, saying so in its reason.

The other built-ins:

- `last_valid`: the last hand-in that passed its checks;
- `best_metric`: of every valid checkpoint with a reading of its metric (hand-ins and `kl`-scored ones), the
  best, in the direction the readings state;
- `baseline`: the first baseline, as-is.

A rule decides on a `Standing`. It holds only valid checkpoints, so a refused one or a check
still pending at the end cannot count. Generation rules may ask its evaluator for fresh judge-only readings.
What came too late never reaches the ledger
(`commands._one_dir` refuses it, and what was asked in time drains within `judge_grace_s`).
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from pydantic import Field
from qda_quant.registry import Registry

from qda_agent.host import Final, HandIn, best_words, label
from qda_agent.settings import Plugin, validate_plugin

from .freeze import digest, file_sha256, link_in, seal, walk

FINAL_RULES = Registry(
    "final_rules",
    {
        "baseline": "qda_agent.judge.final_rules:Baseline",
        "best_metric": "qda_agent.judge.final_rules:BestMetric",
        "handin": "qda_agent.judge.final_rules:Handin",
        "last_valid": "qda_agent.judge.final_rules:LastValid",
    },
    group="qda.final_rules",
    on_add=partial(validate_plugin, "qda.final_rules"),
)

if TYPE_CHECKING:
    from .host import JudgeHost

__all__ = [
    "Baseline",
    "BestMetric",
    "FinalRule",
    "Handin",
    "LastValid",
    "Pick",
    "Standing",
    "apply",
    "default_rules",
    "final_handin_statement",
    "handin_warning",
    "statement",
]


@dataclass(frozen=True)
class Standing:
    """What a rule decides on.

    Attributes:
        valid: The records of the valid, eligible submitted and scored checkpoints, oldest first.
        baselines: The baseline checkpoints, in fallback order.
        metric: The metric the search ranks readings by (`judge.metric`).
        candidate0_kl: Candidate 0's KL (`judge.candidate0_kl`), read by the same scorer and reference as the
            readings; None when the search did not read it.
        refused: The directories of hand-ins that failed their checks, so a reading of the same directory
            cannot count.
        disk_refusals: The records of hand-ins refused for disk space (not for what they are), oldest first.
            Disk space never decides which checkpoint the run submits: with one, the `handin` rule picks the
            best reading of the hand-in list, the scored, eligible builds and candidate 0 (`candidate0_kl`).
        whole: Whether a checkpoint's frozen copy is whole (present, with its weight files): the `handin` rule
            never picks one that is not. The judge passes `prune.is_whole`; a rule read on a ledger
            alone (a replay, a test) takes every copy as whole.
    """

    valid: list[HandIn]
    baselines: list[Path]
    metric: str
    candidate0_kl: float | None = None
    refused: frozenset[Path] = frozenset()
    disk_refusals: tuple[Mapping[str, Any], ...] = ()
    whole: Callable[[Path], bool] = lambda _tree: True


class Pick(NamedTuple):
    """A rule's decision.

    Attributes:
        reason: One line on why this rule fired, for FINAL.json and the end report.
        choice: The checkpoint it picked from `Standing.valid`, if any.
        baseline: The baseline it picked, if any. Neither: it decided that nothing can be picked.
        generation_check: The generation rule's evidence, when it ran.
        chosen_by: The `handin` contract's path that decided, when this rule is that contract:
            `handin` or `fallback_candidate0`.
        skipped: Valid entries the rule passed over because their frozen copies no longer hold weights
           ; `apply` lists each in `Final.why_not`.
    """

    reason: str
    choice: HandIn | None = None
    baseline: Path | None = None
    generation_check: dict[str, Any] | None = None
    chosen_by: Literal["handin", "fallback_candidate0"] | None = None
    skipped: tuple[HandIn, ...] = ()


class FinalRule(Plugin):
    """A rule for the checkpoint a search ends on; its settings are pydantic fields."""

    accepts_candidate0: ClassVar[bool] = False
    #: Whether the rule runs a generation suite at cutoff: its checkpoint is answerable by the suite's engine
    #: only, so a search whose engine has no suite cannot carry it (refused at config load).
    runs_suite: ClassVar[bool] = False

    def decide(self, standing: Standing) -> Pick | None:
        """The rule's decision, or None to leave it to the next rule."""
        raise NotImplementedError

    def prepare(self, work: Path) -> None:
        """Validate and freeze private inputs at host startup, before any GPU job."""

    def failsafe(self, reason: str) -> Pick | None:
        """Recover only an already checked model after a finalization failure, without new work."""
        del reason
        return None

    def reserve_s(self) -> float:
        """Additional judge admission time for this rule's final checks (default none)."""
        return 0

    def frozen_keep(self) -> int:
        """How many of the best builds this rule reads from their frozen copies after the run (default 0).

        A rule that decides at cutoff from fresh generation checks reads candidates then, so the judge keeps
        their frozen copies (`prune._frozen_keep`), and the disk budget reserves them
        (`custody.required_bytes`).
        """
        return 0

    def statement(self, metric: str) -> str:
        """What this rule makes count, as the task statement tells the agent: `the last submission ... counts`.

        Args:
            metric: The metric the search ranks readings by (`judge.metric`).
        """
        raise NotImplementedError


class LastValid(FinalRule):
    """The last submission that passed its checks."""

    name: Literal["last_valid"] = "last_valid"

    def decide(self, standing: Standing) -> Pick | None:
        """The last valid hand-in, if there is one."""
        handed = [e for e in standing.valid if e.kind == "handin"]
        return Pick("the last valid submission", choice=handed[-1]) if handed else None

    def statement(self, metric: str) -> str:
        """The sentence for TASK.md."""
        del metric
        return "the last submission that passed counts"


class BestMetric(FinalRule):
    """The valid checkpoint a metric ranks best, in the direction its readings state.

    Attributes:
        metric: A key of the readings' `metrics`; None: `judge.metric`.
    """

    name: Literal["best_metric"] = "best_metric"
    metric: str | None = Field(default=None, min_length=1)

    def decide(self, standing: Standing) -> Pick | None:
        """The best reading of the metric, if any valid checkpoint has one."""
        metric = self.metric or standing.metric
        if (picked := _best_reading(standing.valid, metric)) is None:
            return None
        best, higher = picked
        lead = "" if any(e.kind == "handin" for e in standing.valid) else "no valid submission; "
        return Pick(
            f"{lead}the valid scored checkpoint with the {'highest' if higher else 'lowest'} "
            f"{label(metric)} ({best.metrics[metric]:.4g})",
            choice=best,
        )

    def statement(self, metric: str) -> str:
        """The sentence for TASK.md."""
        return (
            f"the valid checkpoint you scored with `kl` that has {best_words(self.metric or metric)} counts"
        )


def _kl(e: HandIn) -> float | None:
    """The record's KL reading, or None when it has none (or it is not finite)."""
    value = e.metrics.get("kl")
    return value if value is not None and math.isfinite(value) else None


def _best_reading(
    valid: Sequence[HandIn], metric: str, refused: frozenset[Path] = frozenset()
) -> tuple[HandIn, bool] | None:
    """The valid checkpoint with the best reading of `metric`, and whether higher is better.

    Only a reading the entry itself states counts (one without it cannot be ranked), and a directory refused
    at hand-in never wins on a reading of the same files. None when no entry has a reading.
    """
    read = [
        e
        for e in valid
        if math.isfinite(e.metrics.get(metric, math.nan))
        and metric in e.higher_is_better
        and e.dir not in refused
    ]
    if not read:
        return None
    higher = read[0].higher_is_better[metric]  # one score command: one direction
    return (max if higher else min)(read, key=lambda e: e.metrics[metric]), higher


def pickable_handins(
    ledger: Sequence[HandIn], whole: Callable[[Path], bool] = lambda _tree: True
) -> list[HandIn]:
    """The hand-ins `Handin` can still pick, best first: the valid, eligible entries of the latest list that has
    one, whose frozen copies are `whole`.

    The rule takes the first (the lowest slot), and the next when the first stops counting; a later list
    supersedes an earlier one once it has a valid entry (an open or refused entry supersedes none). A record from
    before hand-in lists is a list of one: the ledger's last. `prune.kept_handins` keeps every valid
    hand-in's copy whole, so a rule never names a copy a prune pass took away; a copy an operator
    removed is passed over with the reason the rule states.
    """
    handed = [e for e in ledger if e.kind == "handin" and e.valid and e.eligible]
    lists = [e.handin_list for e in handed if e.handin_list is not None]
    if lists:
        latest = max(lists)
        handed = [e for e in handed if e.handin_list == latest]
    else:
        handed = handed[-1:]
    return sorted((e for e in handed if whole(e.dir)), key=lambda e: e.slot)


#: Why a valid entry was passed over: a prune pass removed its copy's weights, or reclaimed the copy.
STRIPPED = "its frozen copy no longer holds its weights"


def _skipped(entries: Sequence[HandIn], what: str = "hand-in") -> str:
    """The reason's clause for valid entries passed over because their copies are not whole."""
    names = ", ".join(e.dir.name for e in entries)
    if len(entries) == 1:
        return f"valid {what} {names} skipped: its frozen copy no longer holds weights"
    return f"valid {what}s {names} skipped: their frozen copies no longer hold weights"


def _gone_ahead(standing: Standing, loadable: Sequence[HandIn], pick: HandIn | None) -> tuple[HandIn, ...]:
    """The read entries better than `pick` (all of them with none) whose copies are not whole, best first."""
    gone = [e for e in standing.valid if e not in loadable]
    if (ranked := _best_reading(gone, standing.metric, standing.refused)) is None:
        return ()
    higher = ranked[1]
    metric = standing.metric
    read = sorted(
        (e for e in gone if math.isfinite(e.metrics.get(metric, math.nan)) and e.dir not in standing.refused),
        key=lambda e: e.metrics[metric],
        reverse=higher,
    )
    if pick is None:
        return tuple(read)
    bar = pick.metrics[metric]
    return tuple(e for e in read if (e.metrics[metric] > bar if higher else e.metrics[metric] < bar))


class Handin(FinalRule):
    """The first valid checkpoint of the agent's latest hand-in list; else the configured fallback.

    A `handin` call is one ordered list, best first. The first directory of the latest list that passed its
    check counts; a list none of whose directories passed falls back to the latest earlier list that has one.
    With no valid hand-in at all the fallback decides: candidate 0 as-is, as in the paper. The agent decides,
    and `gencheck` is its own tool: the judge adds no check and does not rank the scored builds.
    `FINAL.json`'s `chosen_by` says which path decided.

    One exception, so that disk space never decides which checkpoint the run submits: when a hand-in was
    refused for disk space (not for what the checkpoint is), or a build was stopped at the free-space floor,
    the pick is the best reading of the hand-in list and the scored, eligible builds; candidate 0's reading
    competes too (it is not in the ledger), and when none beats it the result is candidate 0 as-is. The reason
    says which hand-in was refused (or build stopped) and what decided instead. A reading is needed to rank, so a run with none keeps the rule unchanged.
    """

    name: Literal["handin"] = "handin"
    accepts_candidate0: ClassVar[bool] = True

    def decide(self, standing: Standing) -> Pick:
        """The first valid entry of the latest hand-in list; with none, candidate 0 as-is.

        A hand-in refused for disk space does not decide the result: the best reading of the hand-in list,
        the scored, eligible builds and candidate 0 counts instead (`Standing.disk_refusals`).
        """
        candidate0 = standing.baselines[0] if standing.baselines else None
        handed = [e for e in standing.valid if e.kind == "handin"]
        loadable = [e for e in standing.valid if standing.whole(e.dir)]  # never a copy without its weights
        # the read entries the disk path would have ranked ahead of its pick, but whose copies are gone
        ahead: tuple[HandIn, ...] = ()
        if standing.disk_refusals:
            kept = _best_reading(loadable, standing.metric, standing.refused)
            ahead = _gone_ahead(standing, loadable, None if kept is None else kept[0])
        if (
            standing.disk_refusals
            and (picked := _best_reading(loadable, standing.metric, standing.refused)) is not None
        ):
            best, higher = picked
            gone = f"; {_skipped(ahead, 'entry')}" if ahead else ""
            first = standing.disk_refusals[0]
            build = first.get("build") or "a hand-in"
            # a job stopped at the free-space floor (`prune.watch_floor`) is recorded with what="build"
            refused = (
                f"build {build} stopped" if first.get("what") == "build" else f"hand-in of {build} refused"
            )
            floor = standing.candidate0_kl if standing.metric == "kl" else None
            if (
                candidate0 is not None
                and floor is not None
                and math.isfinite(floor)
                and (best.metrics["kl"] <= floor if higher else best.metrics["kl"] >= floor)
            ):
                return Pick(
                    f"{refused} for disk space; no build beat candidate 0 ({label(standing.metric)} {floor:.4g})"
                    f"{gone}; candidate 0 as-is",
                    baseline=candidate0,
                    chosen_by="fallback_candidate0",
                    skipped=ahead,
                )
            return Pick(
                f"{refused} for disk space; submitted the best scored eligible "
                f"build {best.id} ({label(standing.metric)} {best.metrics[standing.metric]:.4g}) instead{gone}",
                choice=best,
                chosen_by="handin",
                skipped=ahead,
            )
        # The latest list is the call that came last, not the copy that finished freezing last: a call's
        # list id is taken when the command is accepted, before any copy starts.
        pickable = pickable_handins(handed, standing.whole)
        first = pickable[0] if pickable else None
        # valid entries of that list the rule passed over for the pick: a copy without its weights;
        # one after the pick was not passed over, so it is not named
        skipped = ahead + tuple(
            e
            for e in pickable_handins(handed)
            if not standing.whole(e.dir) and (first is None or e.slot < first.slot) and e not in ahead
        )
        if first is not None:
            gone = f"; {_skipped(skipped)}" if skipped else ""
            return Pick(
                f"the first valid entry of the latest hand-in list ({first.dir.name}, entry {first.slot + 1}){gone}",
                choice=first,
                chosen_by="handin",
                skipped=skipped,
            )
        why = _skipped(skipped) if skipped else "no valid hand-in"
        return Pick(
            f"{why}; candidate 0 as-is"
            if candidate0 is not None
            else f"{why}, and no baseline is configured",
            baseline=candidate0,
            chosen_by="fallback_candidate0",
            skipped=skipped,
        )

    def statement(self, metric: str) -> str:
        """The sentence for TASK.md: the paper's selection rule."""
        del metric
        return (
            "the first directory of your latest hand-in list that has passed counts; a check not finished by "
            "then does not; with none, the first that passed of the latest earlier list that has one; with "
            "none, candidate 0 as-is"
        )


class Baseline(FinalRule):
    """The first baseline as-is. It always decides, so a rule after it never runs."""

    name: Literal["baseline"] = "baseline"

    def decide(self, standing: Standing) -> Pick:
        """The first baseline; with none configured, the decision that nothing can be picked."""
        why = (
            "no earlier rule picked a checkpoint"
            if standing.valid
            else "nothing valid was submitted or scored"
        )
        if not standing.baselines:
            return Pick(f"{why}, and no baseline is configured")
        return Pick(f"{why}; the first baseline as-is", baseline=standing.baselines[0])

    def statement(self, metric: str) -> str:
        """The sentence for TASK.md."""
        del metric
        return "the first baseline as-is"


def default_rules() -> list[FinalRule]:
    """The rule a new run that names none tries: the agent's hand-in list, candidate 0 as the fallback."""
    return [Handin()]


def _why_not(e: HandIn) -> str:
    """Why a submitted checkpoint did not count: refused, ineligible, or still being checked."""
    if e.valid is False:
        return e.reason or "refused"
    if e.valid and not e.eligible:
        return e.reason or "not eligible"
    return "check not finished"


def apply(
    rules: Sequence[FinalRule],
    ledger: list[HandIn],
    baselines: list[Path],
    metric: str,
    candidate0_kl: float | None = None,
    disk_refusals: Sequence[Mapping[str, Any]] = (),
    whole: Callable[[Path], bool] | None = None,
) -> Final:
    """The decision of the first rule that makes one.

    Args:
        rules: The rules to try, in order (`judge.final_rules`).
        ledger: Every submitted and scored checkpoint, oldest first.
        baselines: The baseline checkpoints, in fallback order.
        metric: The metric the search ranks readings by (`judge.metric`).
        candidate0_kl: Candidate 0's KL (`judge.candidate0_kl`), for the `handin` rule.
        disk_refusals: The records of hand-ins refused for disk space (not for what they are).
        whole: Whether a frozen copy is whole (`Standing.whole`); None takes every copy as whole.

    Returns:
        The record of the decision; its `rule` is None when no rule made one.
    """
    hand = [e for e in ledger if e.kind == "handin"]
    last = max((i for i, e in enumerate(hand) if e.valid and e.eligible), default=-1)
    why_not = [f"{e.dir.name}: {_why_not(e)}" for e in hand[last + 1 :]]
    refused = frozenset(e.dir for e in hand if e.valid is False)
    standing = Standing(
        [e for e in ledger if e.valid and e.eligible],
        baselines,
        metric,
        candidate0_kl,
        refused,
        tuple(disk_refusals),
        **({} if whole is None else {"whole": whole}),
    )
    for rule in rules:
        if (got := rule.decide(standing)) is not None:
            return Final(
                choice=got.choice,
                rule=rule.name,
                baseline=got.baseline,
                reason=got.reason,
                why_not=[*why_not, *(f"{e.dir.name}: {STRIPPED}" for e in got.skipped)],
                ledger=ledger,
                generation_check=got.generation_check or {},
                chosen_by=got.chosen_by,
                warnings=unscored(got.choice, ledger, metric),
            )
    return Final(rule=None, reason="no rule of judge.final_rules decided", why_not=why_not, ledger=ledger)


def unscored(choice: HandIn | None, ledger: Sequence[HandIn], metric: str) -> list[str]:
    """A warning when ``choice`` has no ``metric`` reading and every reading the run asked for failed.

    The rule itself is unchanged: the agent's hand-in still stands, but the result is not a scored search's.
    """
    if choice is None or metric in choice.metrics:
        return []
    readings = [e for e in ledger if e.kind == "scored"]
    if not readings or any(metric in e.metrics for e in readings):
        return []
    return [f"no reading of this run succeeded ({len(readings)} attempted; the ledger has each reason)"]


def handin_warning(
    choice: HandIn | None,
    ledger: Sequence[HandIn],
    metric: str,
    candidate0_kl: float | None,
    use_candidate0: Callable[[], str | None] | None,
    score_with: str | None = None,
) -> str | None:
    """One line when the delivered build has no reading, or every reading of the run is worse than candidate 0.

    The pick is unchanged; the line gives the numbers (the pick's reading or "no reading", candidate 0's, the
    best scored one) and ``use_candidate0``, the command or path that delivers candidate 0 instead.

    Args:
        choice: The delivered build, with the reading it inherited from a scored entry of the same files; None
            when a baseline was delivered (no warning).
        ledger: Every submitted and scored checkpoint, oldest first.
        metric: The metric the search ranks readings by (`judge.metric`).
        candidate0_kl: Candidate 0's KL (`judge.candidate0_kl`), None when the run did not read it.
        use_candidate0: Called only when a warning is emitted: returns the serve line of candidate 0, or its
            path when qda planned none; None when the run has no candidate 0.
        score_with: The command that reads the pick (`qda eval RUN_DIR`), named when it has no reading.
    """
    if choice is None:
        return None
    name, word = choice.dir.name, label(metric)
    scored = [e.metrics[metric] for e in ledger if e.valid is not False and metric in e.metrics]
    if metric in choice.metrics:
        scored.append(choice.metrics[metric])
    higher = any(e.higher_is_better.get(metric) for e in ledger)
    best = (max if higher else min)(scored, default=None)
    # KL only: a lower reading is better, and candidate 0's reading is a KL
    worse = (
        metric == "kl"
        and candidate0_kl is not None
        and best is not None
        and math.isfinite(candidate0_kl)
        and best > candidate0_kl
    )
    zero = "no reading" if candidate0_kl is None else f"{candidate0_kl:.4f}"
    top = "no scored build" if best is None else f"{best:.4f}"
    if metric not in choice.metrics:
        said = f"the delivered build {name} has no {word} reading (candidate 0: {zero}, best scored build: {top})"
        said += "; every scored build is worse than candidate 0" if worse else ""
        said += "" if score_with is None else f"; `{score_with}` scores it on the holdout prompts"
    elif worse:
        said = (
            f"every build with a {word} reading is worse than candidate 0 ({name}: {choice.metrics[metric]:.4f}, "
            f"best scored build: {top}, candidate 0: {zero})"
        )
    else:
        return None
    use = None if use_candidate0 is None else use_candidate0()
    return said if use is None else f"{said}; to deliver candidate 0 instead: {use}"


def statement(rules: Sequence[FinalRule], metric: str) -> str:
    """What counts at the end, in the rules' order, as the task statement tells the agent."""
    first, *rest = [rule.statement(metric) for rule in rules]
    return f"Then {first}." + (f" With none, {'; with none, '.join(rest)}." if rest else "")


def final_handin_statement(rules: Sequence[FinalRule]) -> str:
    """The old separate Submission sentence for a `handin`-first rule: now empty.

    The hand-in rule states itself in TASK.md's deadline sentence (`Handin.statement`), so a second
    Submission sentence said the same thing twice. Kept for callers whose template has the slot.
    """
    del rules
    return ""


def export(host: JudgeHost, final: Final) -> None:
    """Copy one verified winner to staging, fsync it, and rename it atomically; keep custody on failure.

    The run's disk caps do not apply: they limit the search, and refusing the export would leave the
    winner only in custody.
    """
    source = final.choice.dir if final.choice else final.baseline
    if source is None:
        return
    target = host.run_dir / "winner"
    stage = None
    try:
        if target.exists() or target.is_symlink():
            raise ValueError("winner path already exists; refusing to replace it")
        if final.choice is not None:
            entry = next((e for e in host.entries if e.dir == source and e.valid and e.eligible), None)
            if entry is None or link_in(source) is not None or digest(source) != entry.digest:
                raise ValueError("selected checkpoint differs from its sealed digest")
            expected = entry.digest
        else:
            if source not in host.baseline_export_digests:
                raise ValueError("selected baseline is not a trusted source")
            expected = host.baseline_export_digests[source]
        stage = Path(tempfile.mkdtemp(prefix=".winner-", dir=host.run_dir))
        shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=False)
        actual = seal(stage)
        if actual != expected:
            raise ValueError("exported checkpoint differs from its sealed digest")
        for path in [*walk(stage), stage]:
            with sync_fd(path) as fd:
                os.fsync(fd)
        stage.rename(target)
        with sync_fd(host.run_dir) as fd:
            os.fsync(fd)
        final.export = {
            "winner": str(target),
            # The checkpoint this copy was made from. The pick is renamed to the copy below, and a baseline
            # keeps its reading under the name it had then, so this is what says whose reading the pick has.
            "source": str(source),
            "sha256": actual,
        }
        if final.choice is not None:
            final.choice = final.choice.model_copy(update={"dir": target})
        else:
            final.baseline = target
    except (OSError, ValueError) as error:
        final.export = {"winner": None, "error": str(error), "custody": str(host.cfg.custody_dir)}
        final.reason += f"; winner export failed: {error}; local custody retained"
    finally:
        if stage is not None and stage.exists():
            from ..sched.snapshot import remove_tree

            remove_tree(stage)


@contextmanager
def sync_fd(path: Path) -> Iterator[int]:
    """Open a file or directory for fsync and close its descriptor after use."""
    fd = os.open(path, os.O_RDONLY)
    try:
        yield fd
    finally:
        os.close(fd)


def baseline_digest(tree: Path) -> str:
    """Hash baseline files including metadata, allowing only its own trusted cache links."""
    from qda_quant.intake.fingerprint import link_inside

    result = hashlib.sha256()
    for path in walk(tree):
        if path.is_symlink() and (path.is_dir() or not link_inside(tree, path)):
            raise ValueError(f"baseline export refuses foreign link {path}")
        if path.is_file():
            result.update(path.relative_to(tree).as_posix().encode() + b"\0")
            result.update(file_sha256(path).encode())
    return result.hexdigest()
