"""Install judge inputs, admit baselines and check the build and reading paths before the agent starts."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

from qda_kl.eval.suites import SuiteResult
from qda_quant.checks import cpu_step_config
from qda_quant.exit import USAGE
from qda_quant.guards import SEARCH_LAUNCH, SEARCH_START, Guard, Problem, first_problem

from ..isolation.paths import interpreter, job_env
from ..run_sources import check_private_reads, private_paths
from ..sched import Kind, Refused, SchedConfig, State
from ..sched.snapshot import remove_tree
from . import build_receipt, custody
from .checks import CheckpointError, anchor_problem, format_problem, map_head_format, stored_formats
from .checks.budget import bpw_problem
from .checks.gguf import gguf_problem
from .contamination import ContaminationCheck
from .freeze import freeze, link_in, tree_size
from .intake import has_routed_experts, read_json, serve_problem
from .jobs import TOOLS, failed_reason, fill, job_python, queue, reading_result
from .ledger import Entry

if TYPE_CHECKING:
    from qda_quant.calib.contamination import PromptIndex

    from .config import JudgeConfig
    from .host import JudgeHost

#: A reference's own record of what it holds: where a holdout declaration is cross-checked.
_REFERENCE_RECORDS = ("selection.json", "ref-record.json")

#: Why a baseline's reading is never eligible: the final rules weigh candidate 0 by its KL, not as an entry.
BASELINE_REFERENCE = "not eligible: a baseline is the reference a candidate is compared with"


@dataclass(frozen=True)
class StartChecks:
    """What `static_checks` resolved before the host created anything.

    Both sides of a search read these: the host writes the index, copies the two trees and binds `qda`'s
    installations, and `qda doctor --search` reports what the same call refused.

    Attributes:
        sched: The scheduler's settings with the judge's own hides (`judge.custody_dir`, the receipts' builds).
        index: The builds' calibration filter (`PromptIndex`): the union of the reference's eval set and its
            holdout set. The host saves it under `run_dir`; nothing else writes one.
        evaluator: The evaluator tree the run copies into `judge/eval`.
        prompts: The long-generation prompt file the run copies; None: the evaluator's built-in set.
        forbidden: The paths custody must stay apart from (the run, the workdir, every readable path and bind
            source): `custody.validate`'s argument, which the host's `custody.place` passes on unchanged.
    """

    sched: SchedConfig
    index: PromptIndex
    evaluator: Path
    prompts: Path | None
    forbidden: list[Path]


def static_checks(judge: JudgeConfig, sched: SchedConfig, run_dir: Path, *, workdir: Path) -> StartChecks:
    """The start's own checks: what a search refuses before it creates anything (`qda doctor --search` too).

    A search reads its conditions here, once, instead of around each side effect it takes next: the custody
    directory's placement, the run's disk space, the calibration index's derivation, the evaluator tree, the
    prompt file it is about to freeze, and the `qda` program every job runs. Nothing is created, copied or
    written: the caller writes the index, copies the trees and binds `qda`'s installations where the run
    installs its inputs.

    Args:
        judge: The `judge` section.
        sched: The scheduler's settings (`RunConfig.sched_config`).
        run_dir: The run's directory; it need not exist.
        workdir: The agent's work directory, which the host resolved (`JudgeHost.__init__`) and which every
            `qda` job's `PATH` and custody's forbidden list are built from.

    Returns:
        The resolved start (`StartChecks`).

    Raises:
        ValueError: The first condition this search cannot start under, in the run's own words.
    """
    start = Start.of(judge, sched, run_dir, workdir)
    if (problem := first_problem(STATIC_GUARDS, SEARCH_START, start)) is not None:
        raise problem.error if problem.error is not None else ValueError(problem.text)
    assert start.index is not None and start.evaluator is not None  # set by the guards that just held
    return StartChecks(start.sched, start.index, start.evaluator, start.prompts, start.forbidden)


@dataclass
class Start:
    """What `static_checks`' guards read and resolve, in their order: each guard that holds fills its part.

    Attributes:
        judge: The `judge` section.
        sched: The scheduler's settings; the custody and receipt guards add the judge's own hides to it.
        run_dir: The run's directory, resolved.
        work: The agent's work directory, resolved.
        forbidden: The paths custody must stay apart from (the run, the workdir, every readable path and bind
            source): the judge's own trees stay out of the agent's reach, custody outside the run and the
            receipts' builds (grown, then pruned, during the search) inside it.
        index: The builds' calibration index, once its guard held.
        evaluator: The evaluator tree, once its guard held.
        prompts: The long-generation prompt file, once its guard held; None: the evaluator's built-in set.
    """

    judge: JudgeConfig
    sched: SchedConfig
    run_dir: Path
    work: Path
    forbidden: list[Path]
    index: PromptIndex | None = None
    evaluator: Path | None = None
    prompts: Path | None = None

    @property
    def jdir(self) -> Path:
        """The judge's directory in the run."""
        return self.run_dir / "judge"

    @classmethod
    def of(cls, judge: JudgeConfig, sched: SchedConfig, run_dir: Path, workdir: Path) -> Start:
        """The context `static_checks` walks its guards with (`qda doctor --search` builds the same one)."""
        run_dir, work = Path(run_dir).resolve(), Path(workdir).resolve()
        return cls(judge, sched, run_dir, work, [run_dir, work, *sched.readable, *sched.binds.values()])


#: What every start refusal but the disk's and the reference's tells the user to do.
_SETTING_FIX = "fix the setting it names"
#: The way out of `ReferenceUnreadable`: a cached reference whose prompt file went with its builder's run.
REFERENCE_FIX = (
    "run the `qda-agent prepare` that made it again, which rebuilds the reference (a cached entry now keeps its "
    "own copy of its prompts), or name a reference whose prompt file exists"
)


def _refused(step: Callable[[Start], None], fix: str = _SETTING_FIX) -> Callable[[Start], Problem | None]:
    """A guard's check from a step that raises the start's `ValueError`: its sentence on one line, its fix.

    The status is 2, `qda-agent run`'s for a start the host refuses (`flow.StartRefused`), and the exception is
    kept so `static_checks` raises the type it always raised (`custody.DiskRefused`, `ReferenceUnreadable`).
    """

    def check(start: Start) -> Problem | None:
        try:
            step(start)
        except ValueError as e:
            if isinstance(e, custody.DiskRefused):
                why = e.problem.fix
            else:
                why = REFERENCE_FIX if isinstance(e, ReferenceUnreadable) else fix
            return Problem(" ".join(str(e).split()), why, USAGE, error=e)
        return None

    return check


def _custody(start: Start) -> None:
    custody_dir = start.judge.custody_dir
    assert custody_dir is not None  # `applies`
    if start.sched.job_sandbox != "bwrap":
        raise ValueError(
            "judge.custody_dir needs the jobs in bwrap, and this run's jobs run on the host; "
            "fix: set isolation.level: sandbox or strict"
        )
    custody.validate(custody_dir, start.forbidden)
    start.sched = start.sched.model_copy(update={"hide": [*start.sched.hide, custody_dir.resolve()]})


def _receipts(start: Start) -> None:
    if start.sched.job_sandbox != "bwrap":
        raise ValueError(
            "require_build_receipt needs the jobs in bwrap, and this run's jobs run on the host; "
            "fix: set isolation.level: sandbox or strict"
        )
    start.sched = start.sched.model_copy(update={"hide": [*start.sched.hide, start.jdir / "builds"]})


def _index(start: Start) -> None:
    start.index = _calibration_index(start.judge, start.sched, start.run_dir)


def _evaluator(start: Start) -> None:
    # A link is copied as a link but left out of the tree digest: scoring could read bytes the record never
    # hashed. `freeze` checks the copy again; this refuses it before the run creates anything.
    evaluator = start.judge.evaluator or installed("qda_kl")
    if link := link_in(evaluator):
        raise ValueError(
            f"judge.evaluator {evaluator}: {link} is a symbolic link; a frozen copy must hold the bytes it "
            "hashes, so the evaluator tree must hold regular files only"
        )
    start.evaluator = evaluator


def _prompts(start: Start) -> None:
    # The prompt file the judge copies: a link or anything but a regular file is refused, here and at the copy.
    dataset = start.judge.prompt_set()
    assert dataset is not None  # `applies`
    prompts = dataset.local()
    _read_plain(prompts, "judge.gencheck_prompts").close()  # opened, not read: the copy reads it again
    start.prompts = prompts


def _qda(start: Start) -> None:
    # The program every job runs (`qda_resolution`): the host binds what this resolved (`resolve_qda`).
    qda_resolution(start.judge, start.sched, run_dir=start.run_dir, jdir=start.jdir, workdir=start.work)


def _space(start: Start) -> None:
    # without a cap it is the free-space floor alone (no estimate to check)
    custody.check_space(start.judge.custody_dir or start.run_dir, start.judge, len(start.sched.cards))


def _no_custody(start: Start) -> str | None:
    return (
        None if start.judge.custody_dir is not None else "judge.custody_dir unset: the run keeps no custody"
    )


def _no_receipts(start: Start) -> str | None:
    return None if start.judge.require_build_receipt else "judge.require_build_receipt is off"


def _no_prompts(start: Start) -> str | None:
    if start.judge.prompt_set() is None:
        return "no long-generation prompt file: the evaluator's built-in set"
    return None if "gencheck" in start.judge.commands else "no gencheck in judge.commands"


_START = frozenset({SEARCH_START})
_HOST = "qda-agent run: the judge host, before it creates anything (judge.startup.static_checks)"

#: The judge's start checks, in `static_checks`' order: custody's placement, the receipts' sandbox, the
#: calibration index, the evaluator tree, the prompt file it is about to freeze, the `qda` program every job
#: runs, and the run's disk space - last, so a full disk does not mask a refusal named before it. Nothing is
#: created, copied or written. `qda doctor --search` walks the same tuple (`qda_quant.guards`).
STATIC_GUARDS: tuple[Guard[Start], ...] = (
    Guard("custody", _START, _refused(_custody), _HOST, applies=_no_custody),
    Guard("build receipts", _START, _refused(_receipts), _HOST, applies=_no_receipts),
    Guard("calibration index", _START, _refused(_index), _HOST),
    Guard("evaluator tree", _START, _refused(_evaluator), _HOST),
    Guard("prompt file", _START, _refused(_prompts), _HOST, applies=_no_prompts),
    Guard("judge qda", _START, _refused(_qda, "fix judge.qda_cmd / judge.qda_env"), _HOST),
    Guard("disk space", _START, _refused(_space), _HOST),
)

#: What only the run's launch can decide: it needs the run's cards and its first minutes (`launch_smoke`).
#: `qda doctor --search` lists them with where they are decided, and never as passed.
LAUNCH_GUARDS: tuple[Guard[object], ...] = (
    Guard(
        "launch smoke",
        frozenset({SEARCH_LAUNCH}),
        None,
        "the run's launch: a trusted build and candidate 0's reading (judge.startup.launch_smoke)",
        advance=False,
    ),
)


class TrialRunFailed(ValueError):
    """The launch's trial run could not build, serve or read candidate 0 on this machine.

    The run ends with what it has instead of refusing silently: the flow finalizes it (`flow.finalize`) and
    `qda-agent run` exits 3, since the machine has to change before the same command can work.
    """


class ReferenceUnreadable(ValueError):
    """The start's refusal when a reference's prompts cannot be read as an index (`reference_index`).

    A `ValueError` like every start refusal; its own class so the start can name the way out for this cause
    (a cached reference whose builder's run, and the prompt file it recorded, is gone): `qda-agent run`
    prints `fix` after the refusal.
    """

    fix = REFERENCE_FIX


def reference_index(judge: JudgeConfig) -> PromptIndex:
    """The UNION of `judge.kl_ref`'s eval set and `judge.kl_holdout_ref`, the index every build filters against.

    A search names both prompt sets explicitly. The same reference directory in both fields is the old shape,
    whose rows carry ``split: holdout``; the index then partitions that one file rather than counting it
    twice. `_calibration_index` (hence `static_checks`, the start and `qda doctor --search`'s `judge start`
    line) derives the index here, once.

    Raises:
        ValueError: a search names no reference or no holdout set, or the reference record declares a
            different holdout set.
        ReferenceUnreadable: the reference's or the holdout's prompts cannot be read as an index (a recorded
            prompt file that is gone, a digest that does not match, a parent corpus that does not resolve).
    """
    reference = judge.kl_reference()
    holdout = judge.kl_holdout_reference()
    if reference is None:
        raise ValueError(
            "contamination-index: this search host derives the builds' calibration index from "
            "judge.kl_ref and judge.kl_holdout_ref, and neither is set; a prebuilt index is not "
            "provenance - its sources cannot be verified without the reference files it claims. "
            "Name both references (infrastructure error, not the candidate's)"
        )
    if holdout is None:
        raise ValueError(
            "contamination-index: this search filters every build's calibration against the "
            "reference's eval set AND its holdout set, but judge.kl_holdout_ref is unset; name "
            "the holdout set explicitly (the same reference directory is accepted when its rows "
            "carry split=holdout). A search must not run against the eval set alone"
        )
    from qda_quant.step import Refused

    try:
        _check_declared_holdout(reference, holdout)
        return _reference_union(reference, holdout)
    except Refused as e:
        raise ReferenceUnreadable(
            f"contamination-index: the builds' calibration index ({_definition()}): {e}"
        ) from None


def _definition() -> str:
    """The index's definition, as the start's refusals name it."""
    from qda_quant.calib.contamination import CHARS, KIND, WINDOW

    return f"{KIND}: {WINDOW}-word and {CHARS}-char windows"


def _calibration_index(judge: JudgeConfig, sched: SchedConfig, run_dir: Path) -> PromptIndex:
    """The calibration-filter index of every build job: `reference_index`, checked against what the run holds.

    The index holds sha256 digests only, so it can be bound into the agent's sandbox: a build filters
    its calibration rows against the reference prompts without ever reading one. A search whose index
    cannot be built does not start - an infrastructure error, not the agent's - so no build of a search
    runs unfiltered by omission.

    Raises:
        ValueError: `reference_index` refuses, `judge.contamination_index` is not the canonical union, or a
            selected prompt file is readable by the agent's jobs (`_refuse_readable_prompts`).
    """
    index = reference_index(judge)
    if judge.contamination_index is not None:
        from qda_quant.calib.contamination import PromptIndex
        from qda_quant.step import Refused

        try:
            actual = PromptIndex.load(judge.contamination_index)
        except Refused as e:
            raise ValueError(
                f"contamination-index: the builds' calibration index ({_definition()}): {e}"
            ) from None
        if not actual.same_content(index):
            raise ValueError(
                "contamination-index: judge.contamination_index is not the canonical union of "
                f"{judge.kl_reference()} and {judge.kl_holdout_reference()}: its roles, row counts, source "
                "digests or prompt hashes differ; rewrite it from those references or unset it (the index "
                f"must be {_definition()})"
            )
    _refuse_readable_prompts(index, sched)
    return index


def _prompt_file(path: Path) -> Path:
    """The prompt JSONL a reference path names: the file itself, or a reference directory's recorded one.

    A relative record resolves against the reference directory, then its parent when the manifest's
    ``prompts_sha256`` matches there (the shared-corpus layout): see `reference_prompt_file`.
    """
    if not path.is_dir():
        return path
    from qda_quant.calib.contamination import reference_prompt_file

    return reference_prompt_file(path)[0]


def _same_prompt_file(a: Path, b: Path) -> bool:
    """Whether two reference paths resolve to the same prompt file (the old one-file holdout shape)."""
    try:
        return _prompt_file(a).resolve() == _prompt_file(b).resolve()
    except (OSError, ValueError, KeyError, TypeError):
        return Path(a).resolve() == Path(b).resolve()


def _reference_union(reference: Path, holdout: Path):
    """The union index of a reference's eval set and its explicitly named holdout set.

    A reference that holds only a subset of the corpus it was scored on is indexed from that corpus's full
    split, resolved from its selection record (`qda_quant.calib.contamination.parent_corpus`); a record whose
    corpus cannot be resolved refuses the start rather than filtering against the subset.

    Raises:
        ValueError: both fields name the same prompt file (the old shape) but none of its rows is labeled
            ``split: holdout``, so the search would filter against the eval set alone.
    """
    from qda_quant.calib.contamination import PromptIndex

    if _same_prompt_file(reference, holdout):
        index = PromptIndex.read_reference(reference)
        if not any(s.role == "holdout" for s in index.sources):
            raise ValueError(
                f"contamination-index: judge.kl_ref and judge.kl_holdout_ref name the same prompt file "
                f"({reference}), but none of its rows carries split=holdout, so the holdout set is empty; "
                "name the separate holdout set instead"
            )
        return index
    return PromptIndex.union([PromptIndex.read_reference(reference), PromptIndex.read_holdout(holdout)])


def _refuse_readable_prompts(index: PromptIndex, sched: SchedConfig) -> None:
    """Refuse a start whose selected prompt files the agent's jobs can read and the wall cannot hide.

    A reference may record its prompt path relatively from a shared corpus beside it, so the files an index
    was built from need not sit inside the configured reference directories. Check the selected files
    themselves against every readable path the jobs get (readable roots and bind sources) and against the
    hidden paths: a file there and not hidden is refused, naming it.

    Raises:
        ValueError: a selected eval or holdout prompt file is readable by the agent's jobs and not hidden.
    """
    roots = [Path(os.path.realpath(p)) for p in (*sched.readable, *sched.binds.values())]
    hidden = [Path(os.path.realpath(p)) for p in sched.hide]
    for source in index.sources:
        prompt = Path(os.path.realpath(source.path))
        if any(prompt.is_relative_to(h) for h in hidden):
            continue
        box = next((root for root in roots if prompt.is_relative_to(root)), None)
        if box is None:
            continue
        where = "readable" if prompt == box else f"inside {box}, which is readable"
        raise ValueError(
            f"contamination-index: the {source.role} prompt file {prompt} is {where} by the agent's jobs "
            "and is not hidden; the selected reference prompts must stay apart from what the agent can read"
        )


def _check_declared_holdout(reference: Path, holdout: Path) -> None:
    """Refuse when the reference's own record declares a holdout set other than `holdout`.

    The record (``selection.json`` or ``ref-record.json`` beside the reference) may name a holdout directory
    or prompt file, usually with its digest, and may list the holdout rows' ids. A declared holdout that
    `judge.kl_holdout_ref` does not name, a digest it does not match, or an id the named set does not hold is
    a configuration error: the builds would be filtered against the wrong held-out prompts.

    Raises:
        ValueError: a declared holdout is elsewhere, its digest or ids do not match the named set, or the ids
            cannot be checked against the named set.
    """
    paths, ids = _declared_holdout(reference)
    if not paths and not ids:
        return
    if paths:
        matched = [(p, sha) for p, sha in paths if _same_prompt_file(p, holdout)]
        if not matched:
            raise ValueError(
                f"contamination-index: judge.kl_ref's reference record declares the holdout set "
                f"{paths[0][0]}, but judge.kl_holdout_ref names {holdout}; name the declared holdout set "
                "(or the same reference directory when its rows carry split=holdout)"
            )
        from qda_quant.calib import sha256_file

        try:
            digest = sha256_file(_prompt_file(holdout))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"contamination-index: judge.kl_holdout_ref {holdout}: cannot read ({e})"
            ) from None
        for _p, sha in matched:
            if sha is not None and sha != digest:
                raise ValueError(
                    f"contamination-index: judge.kl_holdout_ref {holdout} does not match the digest the "
                    f"reference record declares for its holdout set ({sha} != {digest})"
                )
    if ids:
        held = _holdout_ids(reference, holdout)
        if ids != held:
            missing, extra = sorted(ids - held), sorted(held - ids)
            raise ValueError(
                f"contamination-index: judge.kl_ref's reference record declares {len(ids)} holdout ids, "
                f"but judge.kl_holdout_ref {holdout} holds a different set ({len(held)} ids): "
                f"{len(missing)} declared ids are absent and {len(extra)} held ids are not declared "
                f"(first absent: {missing[0] if missing else None!r}, first undeclared: "
                f"{extra[0] if extra else None!r}); name the declared holdout set"
            )


def _holdout_ids(reference: Path, holdout: Path) -> set[str]:
    """The prompt ids the named holdout set holds (the split=holdout rows when both fields name one file)."""
    from qda_quant.calib.contamination import reference_rows

    try:
        rows, _sha = reference_rows(_prompt_file(holdout), "holdout reference")
    except Refused as e:
        raise ValueError(f"contamination-index: judge.kl_holdout_ref {holdout}: {e}") from None
    if _same_prompt_file(reference, holdout):
        rows = [r for r in rows if isinstance(r, dict) and r.get("split") == "holdout"]
    ids = {str(r["id"]) for r in rows if isinstance(r, dict) and "id" in r}
    if not ids:
        raise ValueError(
            f"contamination-index: judge.kl_holdout_ref {holdout} carries no prompt ids, so the holdout ids "
            "the reference record declares cannot be checked against it"
        )
    return ids


def _declared_holdout(reference: Path) -> tuple[list[tuple[Path, str | None]], set[str]]:
    """The holdout paths (with recorded digests) and ids a reference's own record declares."""
    base = reference if reference.is_dir() else reference.parent
    paths: list[tuple[Path, str | None]] = []
    ids: set[str] = set()
    for name in _REFERENCE_RECORDS:
        record = base / name
        if not record.is_file():
            continue
        try:
            data = json.loads(record.read_text())
        except (OSError, ValueError):
            continue
        _walk_holdout(data, base, paths, ids)
    return paths, ids


def _walk_holdout(node: object, base: Path, paths: list[tuple[Path, str | None]], ids: set[str]) -> None:
    """Collect ``holdout`` entries: a path, an object naming one, or a list of holdout ids."""
    if isinstance(node, dict):
        for key, value in node.items():
            low = key.lower()
            if "holdout" in low:
                if "sha" in low and isinstance(value, str):
                    continue  # a digest beside an entry is read with that entry
                _holdout_entry(value, base, paths, ids)
            else:
                _walk_holdout(value, base, paths, ids)
    elif isinstance(node, list):
        for value in node:
            _walk_holdout(value, base, paths, ids)


def _holdout_entry(value: object, base: Path, paths: list[tuple[Path, str | None]], ids: set[str]) -> None:
    """One declared holdout: a path, an object naming one (digest/ids optional), or a list of ids.

    A bare string is a path only when it looks like one (``a/b``, ``x.jsonl``, absolute): a record also
    labels a split with the bare word ``holdout``, and that label is not a file. A mapping of counts
    (``holdout_source``) names neither, so it declares nothing.
    """
    if isinstance(value, str) and value:
        if not _looks_like_path(value):
            return
        at = Path(value) if Path(value).is_absolute() else base / value
        paths.append((at, None))
        return
    if isinstance(value, list):
        ids.update(v for v in value if isinstance(v, str) and v)
        return
    if not isinstance(value, dict):
        return
    path = value.get("path") or value.get("dir") or value.get("prompts")
    sha = value.get("sha256") or value.get("file_sha256") or value.get("prompts_sha256")
    if isinstance(path, str) and path:
        at = Path(path) if Path(path).is_absolute() else base / path
        paths.append((at, sha if isinstance(sha, str) else None))
    for key in ("ids", "id", "prompt_ids", "row_ids"):
        listed = value.get(key)
        if isinstance(listed, list):
            ids.update(v for v in listed if isinstance(v, str) and v)
        elif isinstance(listed, str) and listed:
            ids.add(listed)


def _looks_like_path(value: str) -> bool:
    """Whether a bare record string names a file (not the split label ``holdout`` or a description)."""
    return Path(value).is_absolute() or "/" in value or value.endswith((".json", ".jsonl"))


def _read_plain(path: Path, setting: str) -> IO[bytes]:
    """Open `path` read-only as a regular file, never following a link.

    Raises:
        ValueError: The file is missing, a link, or not a regular file.
    """
    try:  # not following a link, and checking the file it opened (not a path looked up again)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        raise ValueError(f"{setting} {path} is not a regular file ({e.strerror})") from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"{setting} {path} is not a regular file")
    return os.fdopen(fd, "rb")


def install_evaluator(host: JudgeHost) -> Path:
    """Copy the evaluator package and the judge's scripts read-only under `judge/eval/`.

    `judge/judge.json` records both trees' sha256. Judge jobs get `judge/eval` first on `PYTHONPATH`,
    ahead of any copy installed in the serving venv. The source and its link check are `static_checks`'.

    Raises:
        ValueError: The copied tree holds a symbolic link that appeared since `static_checks` walked it.
    """
    if host.eval_dir is not None:
        return host.eval_dir
    src = host.start_checks.evaluator
    root = host.jdir / "eval"
    root.mkdir()
    dst, tools = root / src.name, root / "tools"
    rec = {
        "evaluator": str(src),
        "copy": str(dst),
        "sha256": freeze(src, dst),
        "tools": str(tools),
        "tools_sha256": freeze(TOOLS, tools),
        "utc": datetime.now(UTC).isoformat(),
    }
    root.chmod(0o555)
    (host.jdir / "judge.json").write_text(json.dumps(rec, indent=1) + "\n")
    host.eval_dir = dst
    return dst


def install_prompts(host: JudgeHost) -> Path | None:
    """Copy the file of `gencheck_prompts` read-only into the judge's dir (None: the built-in set, nothing to copy).

    The source and its plain-file check are `static_checks`'; it is opened again here so the bytes copied are
    the bytes of the file this call opened.

    Raises:
        ValueError: The file is missing, a link or not a regular file.
    """
    src = host.start_checks.prompts
    if src is None or host.prompts is not None:
        return host.prompts
    dst = host.jdir / "gencheck-prompts.jsonl"
    with _read_plain(src, "judge.gencheck_prompts") as f:
        dst.write_bytes(f.read())
    dst.chmod(0o444)
    host.prompts = dst
    return dst


@dataclass(frozen=True)
class Qda:
    """What `judge.qda_cmd` needs at start, resolved from settings rather than from a running host.

    Attributes:
        argv: The command made absolute (`qda_cmd[0]` rewritten to its path on `PATH`).
        path: The `PATH` the jobs run `qda` with: the program's `bin/` first.
        binds: The installations to add to the scheduler's `readable` (read-only), beyond those it holds.
    """

    argv: list[str]
    path: str
    binds: list[Path]


def qda_guarded(
    judge: JudgeConfig, *, run_dir: Path, jdir: Path, eval_dir: Path | None = None
) -> dict[str, Path]:
    """The paths a `qda` installation bind must not expose, by label (`qda_bind`)."""
    guarded = {"$HOME": Path.home().resolve(), "run_dir": run_dir, "the judge's dir": jdir}
    if (reference := judge.kl_reference()) is not None:
        guarded["kl_ref"] = reference.resolve()
    if eval_dir is not None:
        guarded["the evaluator copy"] = eval_dir.parent
    guard = cast(ContaminationCheck, judge.contamination).guard  # it holds (hashes of) the test items
    if guard is not None:
        guarded["contamination.guard"] = Path(guard).resolve()
    if judge.evaluator is not None:  # the evaluator's live source
        guarded["judge.evaluator"] = Path(judge.evaluator).resolve()
    guarded.update(
        {f"{label}[{i}]": path for i, (label, path) in enumerate(private_paths(judge, judge.contamination))}
    )
    return guarded


def qda_bind(root: Path, prog: str, guarded: Mapping[str, Path], *, base: bool = False) -> Path | None:
    """The directory to bind for one installation of the `qda` program (None: already visible).

    Args:
        root: One directory the program's interpreter needs (`interpreter_roots`).
        prog: The program, for the error messages.
        guarded: The paths the bind must not expose, by label (`qda_guarded`).
        base: `root` is the base installation a venv's `pyvenv.cfg` names. uv, pyenv and conda keep theirs
            outside the venv, often under the user's home; the venv's interpreter cannot run without it, so
            it is bound read-only too (the guarded paths below are still refused).

    Raises:
        ValueError: `root` is not a virtualenv, a base installation that exists, or a system prefix; or
            binding it would expose a guarded path ($HOME itself, the run, the judge's dir, the KL
            reference, the evaluator, the contamination guard).
    """
    r = Path(os.path.normpath(root))
    if r == Path("/usr") or r.is_relative_to("/usr"):
        return None  # the sandbox binds /usr already
    if base:
        if not r.is_dir():
            raise ValueError(
                f"toolset.qda / toolset.python: {prog}'s base interpreter is at {r}, which does not exist; "
                "fix the venv's pyvenv.cfg (its `home` line), or rebuild the venv"
            )
    elif not (r.parent == Path("/opt") or r.is_relative_to("/opt") or (r / "pyvenv.cfg").is_file()):
        raise ValueError(
            f"toolset.qda / toolset.python: {prog} is installed in {r}, which is neither a virtualenv "
            "(no pyvenv.cfg) nor a system prefix (/usr, /opt/...); install qda-quant into a virtualenv "
            "(a uv, pyenv or conda base interpreter is bound with the venv whose pyvenv.cfg names it)"
        )
    rr = r.resolve()
    for what, g in guarded.items():
        below = rr.is_relative_to(g) and rr != g
        if what == "$HOME" and below:
            continue  # a virtualenv inside $HOME is its own directory, not the home
        if rr == g or g.is_relative_to(rr) or below:
            raise ValueError(
                f"toolset.qda / toolset.python: binding {r} for {prog} would expose {what} ({g}) to the agent's jobs"
            )
    return r


def qda_resolution(
    judge: JudgeConfig,
    sched: SchedConfig,
    *,
    run_dir: Path,
    jdir: Path,
    workdir: Path,
    eval_dir: Path | None = None,
) -> Qda:
    """`judge.qda_cmd` resolved without a host: what to run, its job `PATH`, and the installations to bind.

    The installations are what `interpreter_roots` names for the program and for a script's interpreter:
    each venv's prefix and, for a uv, pyenv or conda venv, the base installation its `pyvenv.cfg` names
    (often under the user's home). They must expose none of `qda_guarded`'s paths. The host writes the
    result onto itself (`resolve_qda`); doctor checks it here directly, so neither re-derives the other's
    shape.

    A bind is what a job mounts read-only, which only a sandboxed job does: with `scheduler.job_sandbox:
    none` the jobs run as plain host processes and read the host's own paths, so nothing is bound -- and no
    installation is refused for a bind no job would make.

    Raises:
        ValueError: The program is not found, or its installation cannot be bound safely (a sandboxed run).
    """
    prog = shutil.which(judge.qda_cmd[0])
    if prog is None:
        raise ValueError(
            f"toolset.qda / toolset.python: {judge.qda_cmd[0]!r} is not found on PATH (install qda-quant, "
            "or set toolset.qda to an absolute path)"
        )
    prog = os.path.normpath(Path(prog).absolute())
    bindir = Path(prog).parent
    interpreters = [prog]
    with contextlib.suppress(OSError, UnicodeDecodeError):  # a script: its interpreter's installation too
        head = Path(prog).read_bytes()[:256].split(b"\n", 1)[0].decode()
        if head.startswith("#!") and (interp := head[2:].split()[0]).startswith("/"):
            interpreters.append(interp)
    binds: list[Path] = []
    if sched.job_sandbox == "bwrap":  # else no job mounts anything: nothing to bind
        guarded = qda_guarded(judge, run_dir=run_dir, jdir=jdir, eval_dir=eval_dir)
        wanted: dict[Path, bool] = {}  # root -> a base installation (not the venv or the program's own dir)
        for one in interpreters:
            found = interpreter(one)
            for root in found.roots:
                wanted[root] = wanted.get(root, False) or root in found.bases
        binds = [b for r, base in wanted.items() if (b := qda_bind(r, prog, guarded, base=base)) is not None]
    extra = [b for b in dict.fromkeys(binds) if b not in sched.readable]
    check_private_reads(judge, judge.contamination, [*sched.readable, *extra, *sched.binds.values()])
    # The program's bin/ first. `sched.job_sandbox` is the run's one resolution (`RunConfig.job_sandbox`). A
    # judge job that runs on the host (`none`: also an unset knob at `isolation.level: none`) keeps this
    # shell's PATH after it, so the compiler a build needs still resolves; a job sandbox keeps only the
    # directories it mounts (`sandbox_env`).
    tail = os.environ.get("PATH", "") if sched.job_sandbox != "bwrap" else ""
    path = f"{bindir}:{tail or job_env(workdir)['PATH']}"
    if judge.qda_env.get("PATH"):
        path = f"{judge.qda_env['PATH']}:{path}"
    return Qda([prog, *judge.qda_cmd[1:]], path, extra)


def resolve_qda(host: JudgeHost) -> list[str]:
    """`judge.qda_cmd` made absolute; its `bin/` first on the job PATH, its installations bound read-only.

    Resolves what the host's settings name (`qda_resolution`), then writes it back: `host.qda_path` and a
    copy of the scheduler's `readable` with the installations added; returns the argv `host.start` stores
    as `host.qda_argv`.

    Raises:
        ValueError: The program is not found, or its installation cannot be bound safely.
    """
    found = qda_resolution(
        host.cfg,
        host.sched.cfg,
        run_dir=host.run_dir,
        jdir=host.jdir,
        workdir=host.workdir,
        eval_dir=host.eval_dir,
    )
    cfg = host.sched.cfg  # a copy: the caller's SchedConfig is not changed
    host.sched.cfg = cfg.model_copy(update={"readable": [*cfg.readable, *found.binds]})
    host.qda_path = found.path
    host.qda_roots = list(found.binds)
    return found.argv


def installed(package: str) -> Path:
    """The directory of an installed package."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f"judge.evaluator is unset and {package} is not installed")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def read_baseline(
    host: JudgeHost, entry: Entry, cards: int, moe: bool, expected: float, *, record: bool = True
) -> tuple[SuiteResult | None, str]:
    """Queue one KL reading of `entry`, wait for its job, and return it with why it is not valid (`""`).

    The judge's own rules decide (`reading_result` -> `_incomplete`/`_reference_ids`): the whole reference
    scored, a positive token count, a complete reading per sequence, a finite metric and lower is better.
    A job that ends without a result is replaced by its failure line. `record` off keeps the entry out of the
    ledger (the launch smoke reads candidate 0, which is no candidate of the run).

    The wait is `reading_watchdog_s(expected)`, not `expected` itself: the estimate counts the load and the
    scoring, and a reading also starts an engine. A reading the watchdog still cancels is retried once
    (the job is named apart) with twice the wait: the estimate was too low once (`judge.kl_score_s` is one
    scoring time for every model, and a large model's reading outgrows it; `judge.reading_s`, which `prepare`
    measures, replaces it when the search is a llama.cpp one), so the same wait again would only cancel it
    again. If the retry is cancelled too, the reason names the watchdog, its waits and the estimate's own
    setting, and says the checkpoint was not read -- never the cancelled job's own line, which reads as a
    verdict on the checkpoint.

    Raises:
        Refused: The reading cannot be queued (`jobs.queue`).
    """
    first = host.cfg.reading_watchdog_s(expected)
    waits: list[float] = []  # the waits the watchdog cancelled at, one per cancelled attempt
    for attempt in (0, 1):
        wait = first * 2**attempt
        queue(
            host,
            entry,
            host.cfg.kl_cmd,
            Kind.score,
            cards,
            expected,
            moe=moe,
            record=record,
            attempt=attempt,
        )
        job = host.sched.get(entry.job or "")
        end = host.clock.now() + wait
        fired = False
        while not job.state.terminal:
            host.tick()
            if host.clock.now() >= end and not job.state.terminal:
                host.sched.cancel(job.name)
                fired = True
            if not job.state.terminal:
                host.clock.sleep(host.sched.cfg.tick_s)
        if not fired:
            waits = []
            break
        waits.append(wait)
    got = read_json(entry.out) if job.state is State.done else None
    if got is not None:
        reading, why = reading_result(host, entry, got, "kl")
    elif waits:
        after = ", then ".join(f"{w:.0f} s" for w in waits)
        # name the setting the estimate came from: a measured reading is not raised through the vLLM constant
        knob = (
            f"judge.reading_s ({host.cfg.reading_s:g} s, prepare's measured reading)"
            if host.cfg.reading_s is not None
            else f"judge.kl_score_s ({host.cfg.kl_score_s:g} s)"
        )
        reading, why = (
            None,
            (
                f"the judge's reading watchdog cancelled the job after {after} "
                f"(estimate {expected:.2f} s, plus an engine-start allowance) on "
                f"{len(waits)} attempt(s): the checkpoint was not read; a reading of this model takes "
                f"longer than {knob} allows: raise it"
            ),
        )
    else:
        reading, why = None, failed_reason(host, entry, job)
    if reading is not None and not why and reading.higher_is_better["kl"]:
        why = "the KL reading must state lower is better"
    return reading, why


def select_baselines(host: JudgeHost) -> None:
    """Refuse any failed baseline, then order admitted complete readings by KL (ties keep input order).

    A run that ranks pre-built starts (`judge.starts`) reads every start here like any baseline and records
    its label and origin; the lowest valid KL is candidate 0, the bar a candidate is compared with, and it is
    `baselines[0]`, so the hand-in fallback takes the bar as-is. A run that selects candidate 0 without
    labelling starts (`judge.select_candidate0`) reads the same way, with no labels.
    """
    cfg = host.cfg
    starts = {os.path.normpath(start.path): start for start in cfg.starts}
    for i, path in enumerate(cfg.baselines):
        try:
            if not path.is_dir():
                why = "not a checkpoint directory"
            elif cfg.engine == "llama.cpp":
                # a .gguf baseline gets the gate a hand-in gets: the file rules and the anchor's shape; the
                # lowest-KL one becomes candidate 0, so it also keeps the row's MTP choice (the menu and the
                # byte budget do not apply: a reference may be full precision and over a candidate's budget)
                why = gguf_problem(
                    path, cfg.anchor, mtp=cfg.llamacpp.mtp if cfg.llamacpp is not None else "any"
                )
            else:
                why = anchor_problem(path, cfg.anchor) if cfg.anchor is not None else None
                # a baseline is a reference, not a build to judge: the no-budget quantization floor is not its rule
                why = why or format_problem(
                    path,
                    cfg.format_menu,
                    anchor=cfg.anchor,
                    format_map=cfg.format_map,
                    lease=cfg.lease_formats(),
                    floor=False,
                )
            why = why or serve_problem(path)
            if not why and host.bpw_denominator is not None and cfg.budget_bpw is not None:
                why = bpw_problem(path, host.bpw_denominator, cfg.budget_bpw, lease=host.head_lease)
            if not why and host.contamination([path]):
                why = "contamination check: baseline holds test items"
        except CheckpointError as error:
            why = f"not a readable checkpoint ({error})"
        if why:
            raise Refused(f"baselines[{i}] {path}: {why}")
    for i, path in enumerate(cfg.baselines):
        start = starts.get(os.path.normpath(path))
        entry = Entry(
            id=f"b{i}",
            kind="scored",
            src=f"baselines/{i}",
            dir=path,
            digest=host.baseline_digests[path],
            submitted=host.clock.now(),
            eligible=False,
            reason=BASELINE_REFERENCE,
            name=start.name if start is not None else "",
            origin=start.origin if start is not None else None,
        )
        host.baseline_readings.append(entry)
        cards, moe = cfg.reading_cards(), has_routed_experts(path)
        expected = cfg.reading_expected_s(tree_size(path), cards, moe)
        try:
            reading, why = read_baseline(host, entry, cards, moe, expected)
        except Refused as error:
            raise Refused(f"baselines[{i}] {path}: {error}") from None
        host.update(
            entry,
            valid=not why,
            reason=why or entry.reason,  # a valid reading keeps why it is not eligible
            metrics=reading.metrics if reading is not None else {},
            higher_is_better=reading.higher_is_better if reading is not None else {},
        )
        if why:
            raise Refused(f"baselines[{i}] {path}: {why}")
    host.baseline_readings.sort(key=lambda entry: entry.metrics["kl"])
    if cfg.starts:  # the ledger names the bar: the start with the lowest valid KL, its label and its KL
        host.update(host.baseline_readings[0], bar=True)
    ordered = [entry.dir for entry in host.baseline_readings]
    for dest, source in host.sched.cfg.binds.items():
        if source in cfg.baselines:
            host.sched.cfg.binds[dest] = ordered[cfg.baselines.index(source)]
    cfg.baselines[:] = ordered
    cfg.candidate0_kl = host.baseline_readings[0].metrics["kl"]


TIMEOUT_S = 600.0


CANDIDATE0_TIMEOUT_S = 3600.0


def candidate0_expected_s(cfg: JudgeConfig) -> float:
    """What the trial build is queued as: twice candidate 0's measured build, else the timeout.

    The trial is a one-sample rtn of the model `prepare` built candidate 0 from, so `judge.candidate0_build_s`
    (that build's time on this host) bounds it. Queued at the timeout, it could not start under a budget whose
    drain window is shorter than an hour. The timeout itself stays `CANDIDATE0_TIMEOUT_S`.
    """
    measured = cfg.candidate0_build_s
    return CANDIDATE0_TIMEOUT_S if not measured else min(CANDIDATE0_TIMEOUT_S, max(60.0, 2 * measured))


CANDIDATE0 = ".qda-candidate0"
#: The trial build's calibration: one row of `CANDIDATE0_SEQLEN` tokens. An rtn that calibrates activation scales
#: (W4A4) reads it, and the build refuses a row shorter than that, so the row holds several words per token.
CANDIDATE0_SEQLEN = 16
CANDIDATE0_TEXT = " ".join(["The launch check builds the run's own model once."] * 8)


def tiny_model(directory: Path) -> Path:
    """Save a random tiny Qwen3 with a 128-word tokenizer into `directory` and return it.

    Hidden size 128 is one quantization group, so every uniform format admits it; one layer and 16
    calibration tokens keep the build at a few CPU seconds.
    """
    import torch
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
    )
    transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).save_pretrained(directory)
    words = models.WordLevel({f"w{i}": i for i in range(127)} | {"[UNK]": 127}, unk_token="[UNK]")
    tokenizer = Tokenizer(words)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}w1 {{ m['content'] }} w2 {% endfor %}"
    fast.save_pretrained(directory)
    return directory


def format_for(host: JudgeHost) -> str | None:
    """The run's format for the smoke: its menu's last non-bf16 entry, or None when it has none.

    A llama.cpp search names `.gguf` types, which no uniform `rtn` recipe takes; those searches skip it.
    """
    if host.cfg.engine == "llama.cpp":
        return None
    return next((name for name in reversed(host.cfg.format_menu) if name != "bf16"), None)


#: Directories a shipped-config scan never descends into (installs, VCS and caches hold no start configs).
_SKIP_DIRS = frozenset({"site-packages", "node_modules", "__pycache__", "bin", "lib", "lib64"})


def _config_files(path: Path) -> list[Path]:
    """The build configs a staged file or directory tree holds (`.yaml`/`.yml`; install trees pruned)."""
    if path.is_file():
        return [path] if path.suffix.lower() in (".yaml", ".yml") else []
    if not path.is_dir():
        return []
    found = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        found += [Path(root) / name for name in files if Path(name).suffix.lower() in (".yaml", ".yml")]
    return sorted(found)


def staged_config_problems(readable: Sequence[Path | str], binds: Sequence[Path | str] = ()) -> list[str]:
    """The CPU-step refusal of every shipped build config under the operator's staged inputs.

    `readable` and `binds` are the paths an operator stages (a directory tree or one file each): the starts
    and known-goods the agent copies. A pathologically nested YAML is skipped by `cpu_step_config`, so a
    directory that holds no build config cannot stop a launch.
    """
    problems = []
    for path in [*readable, *binds]:
        for config in _config_files(Path(path)):
            if why := cpu_step_config(config):
                problems.append(why)
    return problems


def _start_configs(host: JudgeHost) -> None:
    """Refuse a GPU launch whose staged start / candidate-0 / known-good configs put card steps on CPU.

    The configs an operator ships into the agent's read-only inputs (`scheduler.readable` and the paths
    `scheduler.binds` exposes) are the starts and known-goods the agent copies. A `devices` naming only CPU
    on a step that runs on a card (gptq, scale_kd, guided, ...) turns a minutes-long build into hours;
    `devices` unset, null or naming a card passes, as does a config whose steps are all CPU (e.g. RTN alone).
    The rule is `qda_quant.checks.cpu_step_config`, the same one `qda build` enforces before a weight is read.

    The interpreter installations `resolve_qda` adds to `readable` are bound so a job can import stdlib, not
    staged by the operator, and are not scanned: a conda prefix holds thousands of unrelated YAML test
    fixtures. A directory with no build config, or a file the parser cannot read, is skipped.

    Raises:
        ValueError: One or more staged configs do; the message names each file, its devices line and the step.
    """
    cfg = host.sched.cfg
    if not cfg.gpus or cfg.fake_gpus or not cfg.cards:
        return
    installs = [Path(os.path.abspath(root)) for root in getattr(host, "qda_roots", ())]

    def staged(path: Path | str) -> bool:
        """Whether `path` is an operator's input, not the interpreter installation bound for imports."""
        return not any(Path(os.path.abspath(path)).is_relative_to(root) for root in installs)

    readable = [p for p in cfg.readable if staged(p)]
    problems = staged_config_problems(readable, [v for v in cfg.binds.values() if staged(v)])
    if problems:
        raise ValueError(
            "start config: " + "; ".join(problems) + " (the run has cards: remove `devices` or name a card)"
        )


def launch_smoke(host: JudgeHost) -> None:
    """The run's trial run, minutes before the agent's first turn: candidate 0 really built, then served.

    What a search needs of a machine is not predicted here, it is done: `_start_configs` refuses a shipped
    start / candidate-0 config whose `devices` put a card step on CPU, then `_candidate0` builds the run's
    own model through the scheduler, as one of the run's build jobs, and `_serve_trial` starts an engine on
    what it built exactly as a hand-in is checked. A node whose compiler, kernels, caches or driver a build
    or an engine cannot use fails here, with that failure's own words, rather than after the agent has spent
    the run on builds and readings that could never finish. A CPU run, a test's fake GPUs and a run naming
    no model or format start no engine (`_on_cards`); a run that mints build receipts still builds and admits
    candidate 0 there, as it always has. A llama.cpp search serves no vLLM engine.

    A run that mints build receipts adds its two halves (`_receipt_smokes`): the tiny trusted build and
    candidate 0's KL reading, each off with its own setting.

    Raises:
        TrialRunFailed: A staged config puts a card step on CPU, candidate 0 cannot be built, admitted or
            served, the trusted build verb mints no receipt, or candidate 0's reading does not fold valid.
            The message carries the failing step's own error.
    """
    try:
        _start_configs(host)
        if (
            host.cfg.candidate0_smoke
            and host.cfg.engine != "llama.cpp"
            and (_on_cards(host) or host.cfg.require_build_receipt)
        ):
            _candidate0(
                host
            )  # a build job through the scheduler, then (on cards) the engine on what it built
        _receipt_smokes(host)
    except TrialRunFailed:
        raise
    except ValueError as why:
        raise TrialRunFailed(str(why)) from why


def _on_cards(host: JudgeHost) -> bool:
    """Whether this launch runs its trial run on real cards: a CPU run, a test's fake GPUs and a run that names
    no model or no format (nothing to build) do not."""
    sched = host.sched.cfg
    return bool(sched.gpus and not sched.fake_gpus and sched.cards and host.cfg.model and host.cfg.format)


def _receipt_smokes(host: JudgeHost) -> None:
    """The halves a run that mints build receipts adds: the tiny trusted build and candidate 0's reading."""
    if not host.cfg.require_build_receipt:
        return
    if host.cfg.build_receipt_smoke and (fmt := format_for(host)) is not None:
        interpreter, paths = build_receipt.program(host)
        scratch = Path(tempfile.mkdtemp(prefix="qda-build-smoke-"))
        try:
            _smoke_build(host, scratch, fmt, interpreter, paths)
        finally:
            remove_tree(scratch)
    if host.cfg.reading_smoke:
        _reading(host)


def _candidate0(host: JudgeHost) -> None:
    """Build the run's own model, uniform `rtn` in the run's own format, and admit it exactly as a hand-in is admitted.

    The tiny smoke proves the build verb works; it cannot show a defect of the run's own architecture (an
    export the judge refuses for this model, as NemotronH's remote code was). This builds the run's model
    (`judge.model`, as an agent's build names it) through the agent's own trusted path (`build_receipt.setup`, a queued build job on one card) and runs
    `candidate.admission_problem` on the published tree. Any refusal refuses the start with the judge's
    reason. Nothing stays: the request, the build's custody and the job's output are removed, and nothing
    reaches the ledger. The format is the run's (`judge.format`), the row itself rather than a menu entry; under
    a fixed map (`judge.format_map`) whose lm_head is quantized, a `head` step stores the map's head too.
    The bpw budget is not checked: it judges an agent's allocation, and a uniform build may pass it (a
    NemotronH W4 row: uniform rtn w4a16-asym-g64 is 4.489 bpw against a 4.4 budget) without any defect.

    Raises:
        ValueError: No model or format is configured, the build cannot be queued, fails or times out, or the judge
            does not admit what it built.
    """
    import yaml

    from ..sched import Kind
    from ..sched.job import State
    from .build_receipt import setup
    from .intake import admission_problem

    cfg = host.cfg
    if not cfg.model or not cfg.format:
        raise ValueError(
            "judge.candidate0_smoke: the launch check builds and admits the run's own model (judge.model) in "
            "its format (judge.format), and one is not configured: name both, or set "
            "judge.candidate0_smoke: false explicitly"
        )
    fmt = cfg.format
    steps: list[dict[str, dict[str, str]]] = [{"rtn": {}}]
    if cfg.format_map is not None and (head := map_head_format(stored_formats(cfg.format_map))) is not None:
        steps.append({"head": {"format": head}})  # the fixed map's quantized head is part of the allocation
    shown = fmt if len(steps) == 1 else f"{fmt}, head {steps[1]['head']['format']}"
    what = f"candidate 0 (uniform rtn {shown} of {cfg.model})"
    # The request and its calibration live in the judge's own directory, never in the agent's workdir: the
    # workdir is the agent's, and anything the run gives it must come from the workspace manifest.
    scratch = host.jdir / "candidate0"
    if scratch.exists():
        remove_tree(scratch)
    scratch.mkdir(mode=0o700)
    request, calib = scratch / f"{CANDIDATE0}.yaml", scratch / f"{CANDIDATE0}.jsonl"
    calib.write_text(json.dumps({"text": CANDIDATE0_TEXT}) + "\n")
    # The request keeps the source model's meaning: an absolute path or a Hub id (`org/name[@rev]`) is passed
    # through unchanged (the build's own resolver handles it); only a relative local path, which the request's
    # new base would otherwise re-anchor under the judge's dir, is made absolute against the workdir.
    model = cfg.model
    if not os.path.isabs(model) and (host.workdir / model).exists():
        model = str((host.workdir / model).resolve())
    request.write_text(
        yaml.safe_dump(
            {
                "model": model,
                "format": fmt,
                "steps": steps,
                "calib": {"file": {"path": calib.name, "samples": 1, "seqlen": CANDIDATE0_SEQLEN}},
                "out": CANDIDATE0,
            }
        )
    )
    private = None
    try:
        try:
            private, spec = setup(host, host.workdir, [str(request)], here=scratch)
            # The build job reads its calibration from its own custody directory (the one directory beside
            # the workdir it can see), so the judge's request never has to sit in the agent's workspace.
            kept = private / calib.name
            kept.write_bytes(calib.read_bytes())
            config = Path(spec["config"])
            config.chmod(0o644)
            raw = yaml.safe_load(config.read_text())
            raw["calib"]["file"]["path"] = str(kept)
            config.write_text(yaml.safe_dump(raw))
            config.chmod(0o444)
            (private / "request.json").write_text(json.dumps(spec))
            job = host.sched.submit(
                [*host.qda_program(), "build", spec["config"]],
                str(host.workdir),
                freeze=None,
                expected_s=candidate0_expected_s(cfg),
                ncards=len(host.sched.cfg.cards) if cfg.full_card_moe else 1,
                name="candidate0-smoke",
                kind=Kind.build,
                origin="judge",
                root=host.workdir,
                env={**cfg.qda_env, **({"PATH": host.qda_path} if host.qda_path else {})} or None,
                build_private=private,
            )
        except (ValueError, OSError, Refused) as e:
            raise ValueError(f"judge.candidate0_smoke: {what} cannot be queued: {e}") from None
        finally:
            with host.lock:
                host.build_reserved.pop(str(private), None)
        end, late = host.clock.now() + CANDIDATE0_TIMEOUT_S, False
        while (
            not job.state.terminal
        ):  # a cancelled build is waited for: its tree is removed only once it ended
            host.tick()
            if host.clock.now() >= end and not late and not job.state.terminal:
                host.sched.cancel(job.name)
                late = True
            if not job.state.terminal:
                host.clock.sleep(host.sched.cfg.tick_s)
        if late:
            raise ValueError(f"judge.candidate0_smoke: {what} did not build in {CANDIDATE0_TIMEOUT_S:.0f} s")
        published = Path(str(job.build_private)) / "published"
        if job.state is not State.done or not published.is_dir():
            raise ValueError(
                f"judge.candidate0_smoke: {what} failed to build ({job.state.value}): "
                f"{_cause(host.sched.log(job.name, lines=200, origin='judge'))}; every build of this model "
                "would fail the same way"
            )
        if why := admission_problem(host, published, budget=False):
            raise ValueError(
                f"judge.candidate0_smoke: the judge refuses {what}: {why}; every candidate of this model "
                "would be refused the same way (structural checks only: the bpw budget is deliberately not "
                "checked, since candidate 0 is a probe of the export, not a submission)"
            )
        # the build is sound: now prove an engine starts on it, as the hand-in check will (`_serve_trial`)
        if _on_cards(host):  # a test's fake cards build through a fake runner and start no engine
            _serve_trial(host, published)
    finally:
        if scratch.exists():
            remove_tree(scratch)
        for path in (private, host.workdir / CANDIDATE0):
            if path is not None and (path.is_dir() or path.is_symlink()):
                remove_tree(path)


def _serve_trial(host: JudgeHost, published: Path) -> None:
    """Start an engine on candidate 0, the way a submitted checkpoint is checked (`judge.check_cmd`).

    The hand-in check (`judge/tools/serve_check.py`) loads a checkpoint in stock vLLM and generates a little;
    the trial run runs that command once, on the tree candidate 0's build just published, as a job of the run.
    The engine starts exactly as it does for a hand-in, so a node where no submission could be served refuses
    the launch instead of every hand-in.
    A run that configures no check command (`judge.check_cmd: null`) serves nothing here either.

    Raises:
        ValueError: The job cannot be queued, does not end within `judge.check_expected_s` twice over, or
            reports the checkpoint unservable; the message carries the engine's own error.
    """
    template = host.cfg.check_cmd
    if not template:
        return
    out = host.jdir / "candidate0" / "serve_check.json"
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    slots = {
        "{cards}": str(host.cfg.check_cards),
        "{python}": str(host.cfg.serve_python or job_python()),
        "{ckpt}": str(published),
        "{out}": str(out),
        "{tools}": str(host.eval_dir.parent / "tools") if host.eval_dir is not None else str(TOOLS),
    }
    argv = [fill(arg, slots) for arg in template]
    tmp = host.temps.make("candidate0-serve", "judge-candidate0-serve")
    env = host.cfg.job_environ(host.eval_dir.parent if host.eval_dir is not None else None, tmp_dir=tmp)
    timeout_s = 2 * host.cfg.check_expected_s
    try:
        job = host.sched.submit(
            argv,
            str(host.jdir),
            expected_s=host.cfg.check_expected_s,
            ncards=host.cfg.check_cards,
            name="judge-candidate0-serve",
            kind=Kind.check,
            origin="judge",
            env=env or None,
        )
    except (ValueError, OSError, Refused) as e:
        host.temps.release("judge-candidate0-serve")
        raise ValueError(f"the trial run cannot queue candidate 0's serving check: {e}") from None
    end, late = host.clock.now() + timeout_s, False
    try:
        while not job.state.terminal:
            host.tick()
            if host.clock.now() >= end and not late:
                host.sched.cancel(job.name)
                late = True
            if not job.state.terminal:
                host.clock.sleep(host.sched.cfg.tick_s)
    finally:
        host.temps.release(job.name)
    log = _cause(host.sched.log(job.name, lines=200, origin="judge"))
    if late:
        raise ValueError(f"the trial run's engine did not start candidate 0 in {timeout_s:.0f} s: {log}")
    if job.state is not State.done:
        raise ValueError(
            f"the trial run could not serve candidate 0 ({job.state.value}): {log}; every hand-in of this "
            "run would be checked the same way"
        )
    try:
        said = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"the trial run's serving check wrote no result ({e}): {log}") from None
    if not said.get("ok"):
        raise ValueError(
            f"the trial run served candidate 0 and the engine refused it: {said.get('reason') or log}"
        )


def _reading(host: JudgeHost) -> None:
    """Read candidate 0 once through the judge's own path; refuse the start unless it folds valid.

    Candidate 0 (`baselines[0]`) is the bar every candidate is compared with: a search needs this
    reading anyway, from the same scorer, layout and reference as every candidate's. A scorer that measures
    only a subset of the reference (a wrapper, or a suite the run does not run), writes no per-sequence
    readings, or cannot start refuses the start here with the judge's own reason, instead of refusing every
    candidate's reading after the agent had spent the run. The reading is no candidate: it is not written to
    the ledger and its job's config, result and temp directory are removed; nothing the check did not create
    is ever deleted, so the baseline survives any refusal. A run that selects candidate 0 at start
    (`judge.select_candidate0`) already reads its baselines through the same fold and refuses an invalid
    one, so it skips this check; a run that configures no `baselines[0]` is refused, naming the fix, since
    the reading this check exists to prove could not be made at all.

    Raises:
        ValueError: The reading cannot be queued, or it does not fold valid; the message carries the
            judge's reason (e.g. `the reading leaves N of M sequences unscored`).
    """
    from .freeze import tree_size
    from .intake import has_routed_experts
    from .ledger import Entry

    cfg = host.cfg
    if cfg.select_candidate0 or cfg.starts:
        # `baselines.select` reads every baseline at start through this same fold and refuses a bad one
        return
    if not cfg.baselines:
        raise ValueError(
            "judge.reading_smoke: the launch check reads candidate 0 (baselines[0]) and no baseline is "
            "configured, so no reading could be proven: add a baseline, or set judge.reading_smoke: false "
            "explicitly"
        )
    path = cfg.baselines[0]
    entry = Entry(
        id="smoke",
        kind="scored",
        src="baselines/0",
        dir=path,
        digest=host.baseline_digests[path],
        submitted=host.clock.now(),
    )
    cards, moe = cfg.reading_cards(), has_routed_experts(path)
    expected = cfg.reading_expected_s(tree_size(path), cards, moe)
    try:
        _, why = read_baseline(host, entry, cards, moe, expected, record=False)
    except Refused as err:
        raise ValueError(
            f"judge.reading_smoke: candidate 0 ({path}) cannot be read: {err}; every reading this run "
            "makes would be refused"
        ) from None
    finally:
        _forget(host, entry)
    if why:
        raise ValueError(
            f"judge.reading_smoke: the KL reading of candidate 0 ({path}) is not valid: {why}; every "
            "reading this run makes would be refused"
        )


def _forget(host: JudgeHost, entry: Entry) -> None:
    """Remove the smoke reading's judge-side files: its job config, its result and its temp directory."""
    results = host.jdir / "results"
    for path in (results / f"{entry.id}.yaml", results / f"{entry.id}.json"):
        path.unlink(missing_ok=True)
    for path in (results / entry.id, host.jdir / "tmp" / entry.id):
        if path.is_dir():
            remove_tree(path)


def _smoke_build(host: JudgeHost, scratch: Path, fmt: str, interpreter: str, paths: list[str]) -> None:
    """`run`'s body: one tiny build in `scratch` through `build_receipt`, then the checks on what it left."""
    import yaml

    root = scratch / "root"
    root.mkdir()
    model = tiny_model(scratch / "model")
    calib = scratch / "calib.jsonl"
    calib.write_text(json.dumps({"text": " ".join(f"w{i}" for i in range(16))}) + "\n")
    config = scratch / "build.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": str(model),
                "format": fmt,
                "devices": ["cpu"],
                "steps": [{"rtn": {}}],
                "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 16}},
            }
        )
    )
    private = scratch / "private"
    private.mkdir()
    spec = {
        "private": str(private),
        "root": str(root),
        "destination": "smoke",
        "cwd": str(root),
        "base": str(scratch),
        "config": str(config),
        "source": str(model),
        "forced": [],
        "overrides": [],
        "interpreter": interpreter,
        "paths": paths,
    }
    argv = build_receipt.launch(
        host.sched,
        private,
        spec,
        job_id="build-smoke",
        home=root,
        env={**host.cfg.qda_env, **({"PATH": host.qda_path} if host.qda_path else {})},
        cards=[],
        gpus=False,
        readable=[model, *host.sched.cfg.readable],
    )
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    except FileNotFoundError as e:  # bwrap, or a program the verb names, is not on the host
        raise ValueError(
            f"judge.require_build_receipt: the trusted build of a tiny {fmt} model cannot start: {e}"
        ) from None
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"judge.require_build_receipt: the trusted build of a tiny {fmt} model under {interpreter} "
            f"did not finish in {TIMEOUT_S:.0f} s"
        ) from None
    receipt, record = private / "receipt.json", private / "published" / "build.json"
    if result.returncode != 0:
        stage = next(
            (
                line
                for line in map(str.strip, reversed(result.stdout.splitlines()))
                if line.startswith(("stage ", "plan:"))
            ),
            "",
        )
        raise ValueError(
            f"judge.require_build_receipt: the trusted build of a tiny {fmt} model failed "
            f"(rc {result.returncode}) under {interpreter}"
            + (f" at {stage}" if stage else "")
            + f": {_cause(result.stderr)}"
        )
    missing = [p.name for p in (receipt, record) if not p.is_file()]
    if missing:
        raise ValueError(
            f"judge.require_build_receipt: the trusted build of a tiny {fmt} model exited 0 under "
            f"{interpreter} but came back without {', '.join(missing)}; it cannot mint a receipt"
        )


def _cause(stderr: str) -> str:
    """The line that names why the build failed, not the runner's `CalledProcessError` wrapper around it."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith(("subprocess.CalledProcessError", "CalledProcessError")) and re.search(
            r"(Error|refused):", line
        ):
            return line
    return lines[-1] if lines else "no output"
