"""The host side of a search: the queue, the judge and the RPC server behind the agent's commands.

`JudgeHost` implements `qda_agent.host.Host`; the agent reaches it only through `rpc.sock`. The work is
split by job: `ledger` (the entries and what the agent reads of them), `intake` (taking a checkpoint in),
`jobs` (judge jobs and their results), and `commands` (the built-in commands). `startup` installs inputs,
admits baselines and checks the build and reading paths before the agent starts.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..host import Final, HandIn, serve_line, with_serve
from ..rpc import SUN_PATH_MAX, Reply, Request, Server, bind_path_len
from ..sched import Clock, Refused, SchedConfig, Scheduler, State, SystemClock, make_backend
from ..sched.backend import Backend
from ..settings import load_object
from . import build_receipt, gguf_template, startup
from .checks import (
    CheckpointError,
    anchor_problem,
    format_problem,
    map_menu_problem,
    stored_formats,
    tensor_fingerprint,
    trusted_files_problem,
)
from .checks.budget import bpw_problem, denominator, head_weights
from .checks.gguf import gguf_fingerprint, gguf_problem
from .cli import Usage
from .commands import BUILTIN, qda_args, rest
from .config import JudgeConfig
from .contamination import ContaminationCheck
from .final_rules import apply, handin_warning
from .intake import has_routed_experts, read_json, serve_problem
from .jobs import fold
from .jobtmp import JobTemps

if TYPE_CHECKING:
    from qda_quant.calib.contamination import PromptIndex

    from ..config import RunConfig
    from ..isolation.api import Isolation
    from .gguf_template import Template
from .ledger import Entry, neutral, space_refusals, utc

_log = logging.getLogger(__name__)

__all__ = ["Entry", "JudgeHost", "has_routed_experts", "make_host", "neutral", "serve_problem"]


def _candidate0_line(candidate0: Path) -> str:
    """How to deliver candidate 0: its serve line, else its path. Never raises: a bad build.json keeps the pick."""
    try:
        line = serve_line(candidate0)
    except Exception:
        line = None
    return line or f"use the checkpoint at {candidate0}"


class JudgeHost:
    """Queue + judge + RPC server of one run.

    Args:
        run_dir: The run's host-side directory (`rpc.sock`, `jobs/`, `judge/`).
        sched: The scheduler's settings (`RunConfig.sched_config`).
        judge: The `judge` section.
        clock: Time source (default the system clock).
        backend: Launch backend (default the one `sched` describes).
        autotick: Run the scheduler pass in a background thread after `start()`; tests call `tick()`.
        isolation: The run's one isolation (from `qda_agent.isolation.make`); None: the queue's own default.
        run_config: The run config, for the level's preflight; None: a host built on its own settings (a
            test, the judge CLI), which has no level to hold.
    """

    def __init__(
        self,
        run_dir: Path,
        sched: SchedConfig,
        judge: JudgeConfig,
        clock: Clock | None = None,
        backend: Backend | None = None,
        autotick: bool = True,
        isolation: Isolation | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_config = run_config
        self.cfg, self.clock = judge, clock or SystemClock()
        self.workdir = (judge.workdir or self.run_dir / "work").resolve()
        self.jdir = self.run_dir / "judge"
        if (self.jdir / "ledger.jsonl").exists() or (self.jdir / "eval").exists():
            raise ValueError(f"{self.run_dir} was used by an earlier host; give a new run directory")
        custody = judge.custody_dir
        # Everything this search refuses, before this host creates anything: `qda doctor --search` runs the
        # same function, so a condition the run dies on is never one doctor reported as fine. The custody
        # settings and the run's disk check (`check_space`) are `static_checks`' now; the
        # directories they need are still made here.
        self.start_checks = startup.static_checks(judge, sched, self.run_dir, workdir=self.workdir)
        sched = self.start_checks.sched
        if custody is not None:
            from .custody import place

            place(custody, self.jdir, self.start_checks.forbidden)
        else:
            (self.jdir / "frozen").mkdir(parents=True, exist_ok=True)
        (self.jdir / "results").mkdir(exist_ok=True)
        if backend is None:
            backend = make_backend(sched, self.clock, isolation)
        self.sched = Scheduler(self.run_dir, sched, backend, self.clock, isolation=isolation)
        if judge.require_build_receipt and judge.custody_dir is None:
            # the hide is `static_checks`'; the directory the receipts' builds land in is this host's
            (self.jdir / "builds").mkdir(mode=0o700)
        self.index: Path | None = None  # the builds' calibration filter, from `judge.kl_ref` (hashes only)
        self._install_index(self.start_checks.index)
        self.entries: list[Entry] = []
        self.baseline_readings: list[Entry] = []
        #: Sealed tree digests whose KL reading was refused (`judge.max_kl`): no entry of that file set may win.
        self.kl_refused: set[str] = set()
        #: Receipt digests of builds whose candidate copy was refused at intake (a verdict on the build):
        #: no reading of them can ever land, so pruning ranks them as refused instead of "not read yet".
        self.intake_refused: set[str] = set()
        #: Content fingerprints of the configured build sources, hashed on first use, never per submission.
        self.source_signatures: dict[str, str] = {}
        self.eval_dir: Path | None = None
        self.qda_argv: list[str] | None = None  # `start_checks`' answer, taken up at start (`qda_program`)
        self.qda_path = ""
        #: The interpreter installations `resolve_qda` binds for imports; they are readable by jobs but are
        #: not operator-staged inputs (the shipped-config scan skips them, `startup._start_configs`).
        self.qda_roots: list[Path] = []
        self.socket = judge.socket or self.run_dir / "rpc.sock"
        self.issued = {"handin": 0, "scored": 0, "gencheck": 0}
        self.handin_lists = 0  # one hand-in call is one ordered list (`handin DIR [DIR...]`, best first)
        self.full_card_build: str | None = None
        # Destinations whose custody tree a `take` is freezing right now, with how many takes hold each:
        # prune keeps them (the entry that will protect them is recorded only after the copy ends).
        self.reserved: Counter[str] = Counter()
        #: Frozen copies a `take` holds while it checks one (the entry that will point at them is recorded
        #: only after the copy ends): `prune.reclaim` never removes one of these.
        self.freezing: set[Path] = set()
        # Requests admitted for the run cap and not yet queued as jobs: private dir -> expected output bytes.
        self.build_reserved: dict[str, int] = {}
        #: Hand-ins refused for disk space (not for what they are): recorded in the ledger, read by the final
        #: rules so that disk space never decides which checkpoint the run submits.
        self.space_refusals: list[dict[str, Any]] = []
        #: Whether the last scheduler pass found the disk below `judge.min_free_bytes` (`prune.watch_floor`).
        self.below_floor = False
        #: When the current stretch below that floor began (`prune.watch_floor`); the flow finalizes a run
        #: whose disk has stayed there for ten minutes with nothing to free.
        self.below_floor_since: float | None = None
        # The run's one bf16 GGUF template, made or taken at start (`gguf_template.prepare`); None: none.
        self.gguf_template: Template | None = None
        # The run's one importance matrix, taken at start (`gguf_template.prepare`); None: a build may
        # supply its own, or fit the tensors taken from the template unweighted.
        self.imatrix: Path | None = None
        self.original_cache = self.jdir / "gencheck-original"  # the original model's answers, once per run
        self.temps = JobTemps(self.jdir)  # each judge job's temp dir, by a short absolute name
        self.prompts: Path | None = None  # the judge's copy of `gencheck_prompts` (`install_prompts`)
        # The judge's own inputs may be a Hugging Face cache snapshot (served files as links into its
        # repository's blobs/): a link that leaves that repository is refused here, before anything reads it.
        for name, p in (
            ("judge.anchor", judge.anchor),
            ("judge.format_reference", judge.format_reference),
            *((f"baselines[{i}]", b) for i, b in enumerate(judge.baselines)),
        ):
            if p is not None and p.is_dir() and (why := trusted_files_problem(p)):
                raise ValueError(f"{name} {p}: {why}")
        # A baseline as-is, by its loaded tensors (not its file names; a .gguf's header, on llama.cpp); an
        # unreadable baseline is the judge's own failure, raised here.
        self.baseline_digests = (
            {b: fp for b in judge.baselines if b.is_dir() and (fp := gguf_fingerprint(b)) is not None}
            if judge.engine == "llama.cpp"
            else {b: tensor_fingerprint(b, judge.unscored) for b in judge.baselines if b.is_dir()}
        )
        self.baseline_weights = set(self.baseline_digests.values())
        self.baseline_export_digests: dict[Path, str] = {}
        if judge.custody_dir is not None:
            from .final_rules import baseline_digest

            self.baseline_export_digests = {b: baseline_digest(b) for b in self.baseline_digests}
        # A checkpoint larger than twice the bf16 anchor (+ 1 GiB) is not one of the task's; the cap is that
        # unless `judge.max_copy_bytes` sets it.
        self.max_bytes = judge.max_checkpoint_bytes()
        # The bpw budget's fixed denominator (the original's decoder Linears) and, with `judge.head_lease`, the
        # original's head weights the lease charges, read once so every candidate is measured against the
        # same counts; an unreadable anchor is the judge's own failure, raised here.
        try:
            self.bpw_denominator = None if judge.budget_bpw is None else denominator(cast(Path, judge.anchor))
            self.head_lease = head_weights(cast(Path, judge.anchor)) if judge.head_lease else None
        except CheckpointError as e:
            raise ValueError(f"judge.anchor: {e}") from None
        # built once: an unknown check or a missing guard fails here, not at the first hand-in
        self.contamination = cast(ContaminationCheck, judge.contamination).scanner()
        self.lock = threading.RLock()
        if judge.max_scratch_bytes is not None:
            from .prune import check_room

            self.sched.check_start = functools.partial(check_room, self)
        self.autotick = autotick
        self.server: Server | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._final_lock = threading.RLock()
        self._final_result: Final | None = None

    def _install_index(self, index: PromptIndex) -> None:
        """Write the builds' calibration index `static_checks` derived, and bind its directory to the jobs.

        The index holds sha256 digests only, so it can be bound into the agent's sandbox: a build filters its
        calibration rows against the reference prompts without ever reading one. A search whose index cannot
        be built does not start (`static_checks`), so no build of a search runs unfiltered by omission.

        Args:
            index: The canonical union of the reference's eval set and its holdout set.
        """
        path = self.run_dir / "calibration-index" / "contamination-index.json"
        index.save(path)  # canonical, hashes only: the bind is this file, never a file as given
        self.index = path
        self.cfg = self.cfg.model_copy(update={"contamination_index": path})
        cfg = self.sched.cfg
        self.sched.cfg = cfg.model_copy(
            update={
                "readable": [*cfg.readable, path.parent],
                "host_written": [*cfg.host_written, path.parent],
            }
        )

    # ------------------------------------------------------------------ Host protocol
    def start(self) -> None:
        """Bind `rpc.sock` and (with `autotick`) start the scheduler pass.

        Raises:
            ValueError: A format map (`judge.format_map`) that is not a readable checkpoint, whose formats
                are not exactly the menu's, or that its own map refuses; a format reference that is not a
                readable checkpoint in the task's formats
                (`format_problem`); a local first baseline (candidate 0) that is not the anchor's
                architecture config and files, or not in the task's formats: candidate 0 is admitted as
                the agent would submit it; a trusted build verb that cannot mint a receipt for a tiny
                model (`judge.build_receipt_smoke`); a build of the run's own model that the judge does
                not admit (`judge.candidate0_smoke`); a scorer whose reading of candidate 0 is not a
                complete reading of the reference (`judge.reading_smoke`).
        """
        # A host brought up over an existing ledger (a resumed run) keeps the refusals the ledger records.
        self.restore_space_refusals()
        gguf = self.cfg.engine == "llama.cpp"
        fmap = self.cfg.format_map
        if fmap is not None and not gguf:  # the fixed allocation: readable, on the menu, and itself admitted
            try:
                why = map_menu_problem(stored_formats(fmap), self.cfg.format_menu) or format_problem(
                    fmap, self.cfg.format_menu, anchor=self.cfg.anchor, format_map=fmap, head=self.cfg.head
                )
            except (CheckpointError, ValueError) as e:
                why = f"not a readable checkpoint ({e})"
            if why:
                raise ValueError(f"judge.format_map {fmap}: {why}")
        ref = self.cfg.format_reference
        if ref is not None and not gguf:
            try:
                why = format_problem(
                    ref,
                    self.cfg.format_menu,
                    anchor=self.cfg.anchor,
                    format_map=self.cfg.format_map,
                    head=self.cfg.head,
                )
            except CheckpointError as e:
                why = f"not a readable checkpoint ({e})"
            if why:
                raise ValueError(f"judge.format_reference fails the task's format: {why}")
        # candidate 0 gets the gate a candidate gets at hand-in: the anchor's model files and config, and
        # the task's formats. Its own directory's other files (build logs) stay its own.
        first = self.cfg.baselines[0] if self.cfg.baselines else None
        if not (self.cfg.select_candidate0 or self.cfg.starts) and first is not None and first.is_dir():
            try:
                if gguf:
                    why = gguf_problem(
                        first,
                        self.cfg.anchor,
                        menu=self.cfg.format_menu,
                        budget_bytes=self.cfg.budget_bytes,
                        budget_basis=self.cfg.budget_basis,
                        mtp=self.cfg.llamacpp.mtp if self.cfg.llamacpp is not None else "any",
                        any_format=self.cfg.allow_any_format,
                        reference=first,
                        lock=self.cfg.llamacpp.lock if self.cfg.llamacpp is not None else (),
                        head=self.cfg.head,
                    )
                else:
                    why = anchor_problem(first, self.cfg.anchor) if self.cfg.anchor is not None else None
                    why = why or format_problem(
                        first,
                        self.cfg.format_menu,
                        anchor=self.cfg.anchor,
                        format_map=self.cfg.format_map,
                        head=self.cfg.head,
                        lease=self.cfg.lease_formats(),
                        floor=False,  # candidate 0 is the row's baseline, not a build the floor judges
                    )
            except CheckpointError as e:
                why = f"not a readable checkpoint ({e})"
            if why:
                raise ValueError(f"baselines[0] {first} (candidate 0): {why}")
        # candidate 0 (the first baseline) is the result a search that beats nothing ends on as-is, so it
        # must fit the search's own budget; a task whose fallback does not is misconfigured.
        budget, c0 = self.cfg.budget_bpw, next(iter(self.cfg.baselines), None)
        if (
            self.bpw_denominator is not None
            and not (self.cfg.select_candidate0 or self.cfg.starts)
            and budget is not None
            and c0 is not None
            and c0.is_dir()
            and (why := bpw_problem(c0, self.bpw_denominator, budget, lease=self.head_lease))
        ):
            raise ValueError(f"baselines[0]: {why}")
        if (need := bind_path_len(self.socket)) > SUN_PATH_MAX:
            raise ValueError(
                f"the RPC socket {self.socket} is too long for a unix socket ({need} > {SUN_PATH_MAX} bytes "
                "while binding): use a shorter run_dir"
            )
        self.workdir.mkdir(parents=True, exist_ok=True)
        installed = [self.install_evaluator()]
        if (prompts := self.install_prompts()) is not None:
            installed.append(prompts)
        for rule in self.cfg.final_rules:
            rule.prepare(self.jdir)
        # `static_checks` asked the same `qda_resolution` before this host created anything, so the program
        # is known to run; its installations join the readable paths here, where the rest of the start's
        # paths are installed and the queue is about to run its first job.
        self.qda_argv = startup.resolve_qda(self)
        # The judge's files are installed now: the level takes and verifies its trusted paths here, before
        # the agent's first turn. A trusted path missing when the digests were taken is adopted on its first
        # sight; were that sight the first job, an edit in the agent's first turn would become the baseline.
        self.sched.iso.verify_trusted(installed)
        # A llama.cpp search converts its source once, here, for every GGUF build: refused before
        # anything is written when the template does not fit the disk cap.
        gguf_template.prepare(self)
        # the trial run: a shipped start config that puts a card step on CPU (`startup._start_configs`),
        # candidate 0 built through the scheduler and served as a hand-in is (`startup._candidate0`), and --
        # for a run that mints receipts -- a trusted build verb that mints no receipt
        # (`judge.build_receipt_smoke`) and a scorer whose reading of candidate 0 is not complete
        # (`judge.reading_smoke`) end the run here, before the agent spends a job on builds or readings
        # that could never be eligible

        starter, self.sched.starter = self.sched.starter, threading.current_thread()
        try:
            startup.launch_smoke(self)
            if self.run_config is not None:
                # the level's own preflight, at the moment the queue's preflight ran before the levels
                # existed: a machine that cannot hold the configured level refuses the run before the agent
                # spends a job
                self.sched.iso.preflight(self.run_config)
            if self.cfg.select_candidate0 or self.cfg.starts:
                startup.select_baselines(self)
        finally:
            self.sched.starter = starter
        # the build planner's first import, before the server takes a request: two builds importing it at
        # once raced transformers' import; a failure here is only logged (`build_receipt.warm_planner`)
        build_receipt.warm_planner()
        self.server = Server(self.socket, self.handle, self.workdir)
        self.server.start()
        if self.autotick:
            self._thread = threading.Thread(target=self._loop, name="qsched", daemon=True)
            self.sched.starter = (
                self._thread
            )  # lives as long as the host: the jobs it starts outlive no parent
            self._thread.start()

    def install_evaluator(self) -> Path:
        """Copy the evaluator and the judge's scripts read-only under `judge/eval/`."""
        return startup.install_evaluator(self)

    def install_prompts(self) -> Path | None:
        """Copy `gencheck_prompts` read-only into the judge's dir."""
        return startup.install_prompts(self)

    def qda_program(self) -> list[str]:
        """The `qda` program as resolved at start (`judge.qda_cmd` as given, before it)."""
        return self.qda_argv if self.qda_argv is not None else list(self.cfg.qda_cmd)

    def tick(self) -> None:
        """One scheduler pass, then fold finished judge jobs into the ledger."""
        with self.lock:
            self.sched.tick()
            fold(self)

    def cutoff(self) -> None:
        """Judge-only queue; the agent's builds are cancelled, its checks already queued are kept.

        TASK.md tells the agent that checks already queued still finish, so the agent's long-generation
        checks (and its submission checks) run on through the drain within the drain window; the queue's
        admission fails a judge job that cannot start before the grace deadline instead of hiding it.
        """
        with self.lock:
            self.sched.cutoff()

    def drain(self, timeout_s: float) -> bool:
        """Tick until no judge job is pending or `timeout_s` passes; a cut entry is no verdict."""
        end = self.clock.now() + timeout_s
        last = ""
        while True:
            last = self._guarded_tick(last)  # a failing pass waits out the timeout; it does not raise
            if not self.sched.pending("judge"):
                return True
            if self.clock.now() >= end:
                self._cut_unfinished(timeout_s)
                return False
            self.clock.sleep(min(self.sched.cfg.tick_s, max(0.0, end - self.clock.now())))

    def _cut_unfinished(self, timeout_s: float) -> None:
        """The drain window ended: end every unfinished judge job and record its entry as cut, not failed."""
        with self.lock:
            ended = (
                self.sched.cancel(job.name)
                for job in self.sched.list(origin="judge")
                if not job.state.terminal
            )
            cut = {job.name for job in ended if job.state is State.cancelled}
            fold(self)  # the cancels land on their entries (refused) before being marked cut
            for e in self.entries:
                if e.job in cut:
                    self.update(
                        e,
                        valid=False,
                        cut=True,
                        reason=(
                            f"cut by drain window: the {timeout_s:g} s drain window ended before its job "
                            "finished; no verdict on the checkpoint"
                        ),
                    )

    def handins(self) -> list[HandIn]:
        """Every submission and reading in the ledger, oldest first."""
        with self.lock:
            return [e.record() for e in self.entries if e.kind != "gencheck"]

    def final(self, *, generation_checks: bool = True) -> Final:
        """Decide after cutoff; status previews never run or reveal private generation checks."""
        if self._final_result is not None and self._final_result.export:
            return self._with_gencheck(self._final_result).with_delivery(self.cfg.metric)
        result = self._final(generation_checks=generation_checks)
        if self.baseline_readings:
            result.baseline_readings = [entry.record() for entry in self.baseline_readings]
        return self._with_gencheck(self._choice_reading(result)).with_delivery(self.cfg.metric)

    def _choice_reading(self, final: Final) -> Final:
        """`final` with the pick's reading taken from a scored entry of the same frozen files, and its warning.

        A hand-in is checked, not read; when the agent hands in files the judge already read, that reading is
        the pick's. The pick stays what the rule chose; `handin_warning` says in FINAL.json when it has no
        reading, or when every reading of the run is worse than candidate 0, and how to deliver candidate 0.
        """
        choice, metric = final.choice, self.cfg.metric
        if choice is not None and metric not in choice.metrics:
            with self.lock:
                picked = next((e for e in self.entries if e.kind == "handin" and e.dir == choice.dir), None)
                digest = picked.digest if picked is not None else None
                same = [e for e in self.entries if e.kind == "scored" and e.valid and e.digest == digest]
            if read := next((e for e in reversed(same) if metric in e.metrics), None):  # the newest reading
                reading = {"metrics": read.metrics, "higher_is_better": read.higher_is_better}
                choice = choice.model_copy(update=reading)
                final = final.model_copy(update={"choice": choice, "metrics_from": read.id})
        candidate0 = self.cfg.baselines[0] if self.cfg.baselines else None
        warning = handin_warning(
            choice,
            final.ledger,
            metric,
            self.cfg.candidate0_kl,
            None if candidate0 is None else lambda: _candidate0_line(candidate0),
            f"qda eval {self.run_dir}",
        )
        if warning is None:
            return final
        return final.model_copy(update={"warnings": [warning, *final.warnings]})

    def _with_gencheck(self, final: Final) -> Final:
        """Attach the pick's own long-generation check when no final rule ran one, or record that it has none."""
        if not final.generation_check and final.choice is not None:
            final.generation_check = self.gencheck_evidence(final.choice) or {
                "outcome": "none: the agent ran no valid long-generation check of this pick"
            }
        return final

    def gencheck_evidence(self, choice: HandIn) -> dict[str, Any]:
        """The newest valid long-generation check of the picked checkpoint, with the evaluator's digest.

        A generation final rule records its own run; a rule that runs no generation leaves the agent's `gencheck` entries
        in the ledger only, so the winner's own counts are read back from there for FINAL.json.
        """
        with self.lock:
            picked = next((e for e in self.entries if e.kind != "gencheck" and e.dir == choice.dir), None)
            if picked is None:
                return {}
            checks = [
                e for e in self.entries if e.kind == "gencheck" and e.valid and e.digest == picked.digest
            ]
        if not checks:
            return {}
        newest = max(checks, key=lambda e: e.ended if e.ended is not None else e.submitted)
        evaluator = (read_json(self.jdir / "judge.json") or {}).get("sha256")
        return {
            "id": newest.id,
            "valid": True,
            "max_tokens": newest.max_tokens,
            "per_domain": newest.per_domain,
            "evaluator": evaluator if isinstance(evaluator, str) else "",
            "checkpoint": newest.gen.get("checkpoint", {}),
            "original": newest.gen.get("original", {}),
        }

    def _final(self, *, generation_checks: bool) -> Final:
        """Apply or reuse the final decision; startup readings remain separate from the candidate ledger."""
        if not generation_checks or not self.sched.judge_only:
            if any(rule.reserve_s() for rule in self.cfg.final_rules):
                return Final(
                    rule=None,
                    baseline=self.cfg.baselines[0] if self.cfg.baselines else None,
                    reason="final generation checks wait until cutoff",
                    ledger=self.handins(),
                    engine_floor=self.cfg.engine_floor,
                    candidate0_bar=self._candidate0_bar(),
                )
            result = apply(
                self.cfg.final_rules,
                self.handins(),
                list(self.cfg.baselines),
                self.cfg.metric,
                self.cfg.candidate0_kl,
                disk_refusals=self.space_refusals,
                whole=_whole,
            )
            result.engine_floor = self.cfg.engine_floor
            result.candidate0_bar = self._candidate0_bar()
            return result
        with self._final_lock:
            if self._final_result is not None:
                return self._final_result
            result = apply(
                self.cfg.final_rules,
                self.handins(),
                list(self.cfg.baselines),
                self.cfg.metric,
                self.cfg.candidate0_kl,
                disk_refusals=self.space_refusals,
                whole=_whole,
            )
            result.engine_floor = self.cfg.engine_floor
            result.candidate0_bar = self._candidate0_bar()
            if self.sched.judge_only:
                self._final_result = result
            return result

    def _candidate0_bar(self) -> Path | None:
        """The bar a candidate is compared with when the run ranks labelled starts (`judge.starts`).

        `select_baselines` orders the baselines by KL, so `baselines[0]` is the start with the lowest valid
        KL. A run without starts has no labelled bar; `FINAL.json` says nothing about one.
        """
        return self.cfg.baselines[0] if self.cfg.starts and self.cfg.baselines else None

    def write_final(self, path: Path, final: Final | None = None) -> None:
        """Write the selected final record, or `final()`, to `path` atomically."""
        final = with_serve(final if final is not None else self.final())  # one renderer, one pick rule
        if self.cfg.custody_dir is not None and not final.export:
            from .final_rules import export

            self.stop()
            export(self, final)
            self._final_result = final
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(final.model_dump_json(indent=1) + "\n")
        tmp.replace(path)

    def stop(self) -> None:
        """Stop the pass, the server and every job."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.server is not None:
            self.server.stop()
            self.server = None
        with self.lock:
            self.sched.stop_all()
            self.temps.close()

    def _loop(self) -> None:
        """Tick until `stop`, logging a failed pass (a full disk, a bug) and going on.

        This is the one thread that starts jobs (`Scheduler.starter`), and the jobs it started die with it.
        """
        last = ""
        while not self._stop.wait(self.sched.cfg.tick_s):
            last = self._guarded_tick(last)

    def _guarded_tick(self, last: str) -> str:
        """One `tick`; a failure is logged (once per run of the same error) instead of raised.

        Returns:
            The failure's text, or "" when the pass went through: pass it back as `last`.
        """
        try:
            self.tick()
            return ""
        except Exception as e:  # noqa: BLE001 -- one failed pass is logged; the tick thread goes on
            said = f"{type(e).__name__}: {e}"
            if said != last:
                # the log line itself must not raise: the text may hold anything (a raw surrogate from a path)
                with contextlib.suppress(Exception):
                    _log.exception("scheduler pass failed; the next one follows")
                with (
                    contextlib.suppress(Exception),
                    (self.jdir / "host-errors.log").open("a", errors="backslashreplace") as f,
                ):
                    f.write(f"{utc(self.clock.now())} tick: {said}\n")
            return said

    # ------------------------------------------------------------------ requests
    def handle(self, req: Request) -> Reply:
        """Answer one command (the RPC handler); `req.cwd` is already confined to the workdir."""
        if req.verb not in self.cfg.commands:
            return Reply.refuse(req.verb, "this search has no such command")
        verbs = self.cfg.subcommands.get(req.verb)
        if verbs is not None:
            args = qda_args(self, req.argv)[1] if req.verb == "qda" else rest(req.argv)
            sub = args[0] if args else None
            if sub is not None and sub not in ("-h", "--help") and sub not in verbs:
                return Reply.refuse(
                    req.verb, f"this search has no `{req.verb} {sub}`; it has: {', '.join(verbs)}"
                )
        if req.verb in BUILTIN:
            fn = functools.partial(BUILTIN[req.verb], self)
        elif req.verb in self.cfg.handlers:  # a component's command (`Component.handlers`)
            fn = functools.partial(load_object(self.cfg.handlers[req.verb]), self)
        else:
            return Reply.refuse(req.verb, "no component of this search answers it")
        try:
            return fn(req.argv, Path(req.cwd))
        except Usage as u:
            return Reply(u.status, u.text if u.status == 0 else "", "" if u.status == 0 else u.text)
        except Refused as e:
            return Reply.refuse(req.verb, str(e))

    # ------------------------------------------------------------------ the ledger
    def restore_space_refusals(self) -> None:
        """Rebuild the space refusals from the ledger, so a host resumed over this run keeps them.

        Each refusal is a record in `judge/ledger.jsonl`, beside the entries and prune records; a host that
        starts over an existing ledger reads them back instead of starting with none, or the final rules
        would fall back to candidate 0 after a restart.
        """
        with self.lock:
            self.space_refusals[:] = space_refusals(self.jdir / "ledger.jsonl")

    def announce(self, line: str) -> None:
        """Say one line on the operator's terminal now, where the flow prints its progress (`announce_disk_cap`).

        The judge runs in the flow's process, so stderr is the operator's terminal. A hand-in refused for
        disk space is one the operator must see while it happens, not only in the ledger afterwards.
        """
        print(f"judge: {line}", file=sys.stderr, flush=True)

    def update(self, e: Entry, **what: object) -> None:
        """Set fields of `e`, stamp `ended` once it has a verdict, and append it to the ledger."""
        for k, v in what.items():
            setattr(e, k, v)
        if e.valid is not None and e.ended is None:
            e.ended = self.clock.now()
        self.log(e)

    def log(self, e: Entry) -> None:
        """Append `e` as it is now to `judge/ledger.jsonl`."""
        if self.cfg.custody_dir is not None and e.dir.resolve().parent == (self.jdir / "frozen").resolve():
            from .custody import mirror

            mirror(e.dir, self.jdir / "records" / "frozen" / e.id)
        with (self.jdir / "ledger.jsonl").open("a") as f:
            f.write(e.model_dump_json() + "\n")

    def entry(self, eid: str, kind: str) -> Entry:
        """A copy of entry `eid` of `kind`.

        Raises:
            Refused: There is no such entry.
        """
        with self.lock:
            e = next((e.model_copy() for e in self.entries if e.id == eid and e.kind == kind), None)
        if e is None:
            raise Refused(f"no {eid!r}")
        return e

    def rows(self, kind: str) -> list[Entry]:
        """Copies of the entries of `kind`, oldest first."""
        with self.lock:
            return [e.model_copy() for e in self.entries if e.kind == kind]


def _whole(tree: Path) -> bool:
    """`prune.is_whole`: the final rule never names a frozen copy a prune pass stripped."""
    from .prune import is_whole

    return is_whole(tree)


def make_host(
    run_dir: Path,
    sched: dict,
    judge: dict,
    clock: Clock | None = None,
    *,
    backend: Backend | None = None,
    autotick: bool = True,
    isolation: Isolation | None = None,
    run_config: RunConfig | None = None,
) -> JudgeHost:
    """Build a `JudgeHost` from the dicts of `RunConfig.sched_config` and `RunConfig.judge_config`."""
    return JudgeHost(
        run_dir,
        SchedConfig.model_validate(sched),
        JudgeConfig.model_validate(judge),
        clock,
        backend=backend,
        autotick=autotick,
        isolation=isolation,
        run_config=run_config,
    )
