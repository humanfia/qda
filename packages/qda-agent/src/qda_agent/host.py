"""Define the host interface used by the search flow.

The flow drives the agent; the host runs the GPU scheduler, the judge's readings and submission checks through
`qda_agent.sched` and `qda_agent.judge`.

Lifecycle, in the order the flow calls it::

    host.start()                  # socket up, queue running, baselines read
    ... agent works until the GPU deadline (taken from the run config, not from the host) ...
    host.cutoff()                 # queue -> judge-only; the agent's builds are cancelled, its checks kept
    host.drain(timeout_s)         # every check already queued finishes (or the timeout hits)
    final = host.final()          # the first of `judge.final_rules` that decides (by default: the agent's
                                  # latest valid hand-in list, and only with none candidate 0 as-is)
    host.write_final(path)        # FINAL.json
    host.stop()
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

#: How the agent and the reports name the `kl` suite's metrics; any other metric goes by its own name.
LABELS = {"kl": "KL", "kl_text": "human text"}
#: The statement's words for a metric whose direction is known before any reading states it.
BEST_WORDS = {"kl": "the lowest KL"}

#: Set once a search closes (the flow's close has begun, or a stop signal came): a stop signal then no longer
#: stops anything except a third Ctrl-C (`cli._StopSignals`). Here, not in `flow`: Humanize
#: loads the flow's file as a module of its own, and both must see one flag.
CLOSING = threading.Event()
#: Set by the first stop signal (Ctrl-C, SIGTERM, SIGHUP): the flow sends no new agent turn, ends the one in
#: flight and closes the run (`flow._search`), whether or not the signal's exception reaches the flow's task.
STOPPING = threading.Event()


def serve_line(checkpoint: Path | str | None) -> str | None:
    """One line that serves a qda-built checkpoint, or None when qda planned no vLLM server for it.

    ``build.json["serve"]`` is rendered by
    :func:`qda_quant.output.serve_command.serve_command_line`, the same function `qda build` and `qda-agent
    evaluate` print, so every place that tells a user how to serve names the kernel the search scored on.
    """
    if checkpoint is None:
        return None
    from qda_quant.output.serve_command import serve_command_line
    from qda_quant.step import Refused

    try:
        return serve_command_line(checkpoint)
    except Refused as why:  # an unplanned or unrenderable section: no command rather than a wrong one
        print(f"qda-agent: warning: no serve command: {why}", file=sys.stderr)
        return None


def with_serve(final: Final) -> Final:
    """`final` carrying the line that serves its pick (a line it already has is kept)."""
    if final.serve is not None:
        return final
    pick = final.choice.dir if final.choice else final.baseline
    return final.model_copy(update={"serve": serve_line(pick)})


def label(metric: str) -> str:
    """`metric` as the agent and the reports read it."""
    return LABELS.get(metric, metric)


def best_words(metric: str) -> str:
    """The best reading of `metric`, as the task statement names it."""
    return BEST_WORDS.get(metric, f"the best {label(metric)}")


def delivery(
    choice: HandIn | None,
    baseline: Path | None,
    generation_check: Mapping[str, Any] | None,
    metric: str,
) -> tuple[bool | None, bool | None]:
    """Whether the delivered checkpoint has a reading of `metric` and a completed `gencheck`.

    The facts are the final record's, which the judge fills from its ledger, never the agent's report:
    `choice.metrics` is the pick's own reading or the reading of a scored entry with the same frozen files
    (`Final.metrics_from`), and `generation_check` is the judge's record of the pick's completed
    long-generation check (it writes `outcome: none: ...` when the ledger has none).

    Args:
        choice: The delivered agent checkpoint, or None when the result is a baseline.
        baseline: The delivered baseline, or None when nothing was delivered (a trust refusal).
        generation_check: The pick's long-generation check, as the judge recorded it.
        metric: The metric the search ranks readings by (`judge.metric`).

    Returns:
        `reading` and `gencheck`, each True/False, or (None, None) when nothing was delivered.
    """
    if choice is None and baseline is None:  # nothing was picked: there is no deliverable to measure
        return None, None
    # a baseline as-is carries candidate 0's KL by construction, so only an agent pick can lack a reading
    reading = choice is None or metric in choice.metrics
    check = generation_check or {}
    checked = bool(check) and not str(check.get("outcome") or "").startswith("none")
    return reading, checked


def unmeasured(reading: bool | None, gencheck: bool | None, metric: str) -> str:
    """The one line END-REPORT and `qda-agent evaluate` print for the facts `delivery` decided.

    Args:
        reading: `Final.reading`: False when the delivered checkpoint has no reading of `metric`.
        gencheck: `Final.gencheck`: False when no completed long-generation check of it is recorded.
        metric: The metric the search ranks readings by (`judge.metric`).

    Returns:
        One sentence naming each missing measurement, or "" when both are there or nothing was delivered.
    """
    if reading is None:  # nothing was delivered (a trust refusal)
        return ""
    missing = []
    if not reading:
        missing.append(f"no {label(metric)} reading")
    if not gencheck:
        missing.append("no long-generation check (`gencheck`)")
    return f"The delivered checkpoint has {' and '.join(missing)}." if missing else ""


def rotation_hint(checkpoint: str | Path | None) -> str:
    """One line for a delivered checkpoint whose own record says it runs in a rotated residual basis.

    A speculative draft reads the target's hidden states, so a DFlash or DSpark draft trained against the
    unrotated model reads the wrong basis there and has its tokens rejected. The fact is the checkpoint's
    record (`resolved.yaml`, the one owner `qda rotate-draft` reads too), not a new field.

    Args:
        checkpoint: The delivered checkpoint's directory (an exported copy keeps the run record), or None.

    Returns:
        The line to print, or "" when the record is absent or names no residual rotation.
    """
    if checkpoint is None:
        return ""
    from qda_quant.output.rotate_draft import residual_seeds

    try:
        seeds = residual_seeds(checkpoint)
    except (
        OSError,
        ValueError,
        yaml.YAMLError,
    ):  # an unreadable record says nothing; it never fails a report
        return ""
    if not seeds:
        return ""
    return (
        "this checkpoint runs in a rotated residual basis; a speculative draft (DFlash / DSpark) must be "
        f"rotated with it before serving: `qda rotate-draft --checkpoint {checkpoint} --draft <drafter> "
        "--out <dir>`"
    )


class HandIn(BaseModel):
    """Record a submitted or scored checkpoint and the judge's verdict.

    Attributes:
        id: The ledger entry's id (`h5`, `k12`); empty for a record not taken from one.
        dir: The checkpoint directory as frozen by the judge.
        submitted_at: When `handin` (or `kl`) recorded it.
        kind: `handin` for a submitted checkpoint, `scored` for one only scored with `kl`. Both are
            candidates for a rule that ranks readings; only submissions qualify for `last_valid`.
        valid: The checks' verdict; `None` while a check is still pending.
        eligible: Whether a final rule may pick it. A build-receipt search marks a scored checkpoint that
            matches no build's recorded outputs ineligible; it still counts as scored and reported.
        metrics: The reading's metrics by name, if the judge has read it (`judge.metric` ranks them).
        higher_is_better: Each metric's direction, as the reading states it.
        reason: Why it was refused or marked ineligible (empty when valid, eligible or pending).
        stage: `screen` for a reading of a screening subset only, `full` once it reads all of the reference;
            `None` (every current search) for one reading of the full reference. Only a ledger written by a
            search that screened its readings, under a setting since removed, holds the other two.
        sizes: A llama.cpp candidate's measured `.gguf` bytes, labelled `file` and `payload` (empty for
            vLLM).
    """

    id: str = Field(default="", exclude_if=lambda v: not v)
    dir: Path
    submitted_at: datetime
    kind: Literal["handin", "scored"] = "handin"
    valid: bool | None = None
    eligible: bool = True
    metrics: dict[str, float] = Field(default_factory=dict)
    higher_is_better: dict[str, bool] = Field(default_factory=dict)
    reason: str = ""
    build_receipt: dict[str, Any] = Field(default_factory=dict)
    #: `screen` / `full` in a ledger of a search that screened its readings (a setting since removed); left out
    #: of the JSON otherwise.
    stage: Literal["screen", "full"] | None = Field(default=None, exclude_if=lambda v: v is None)
    #: A llama.cpp candidate's measured `.gguf` bytes, labelled: `file` (the frozen file's `os.stat` size)
    #: and `payload` (its stored tensors, without the header). Both recorded; `judge.budget_basis` picks the
    #: one admission screened. Left out of the JSON when empty so a vLLM search's record is unchanged.
    sizes: dict[str, int] = Field(default_factory=dict, exclude_if=lambda v: not v)
    #: The `handin` call (its list) this submission belongs to, and its rank in that list (0 = best first).
    #: Left out when a record predates hand-in lists, so an old record reads as a list of one.
    handin_list: int | None = Field(default=None, exclude_if=lambda v: v is None)
    slot: int = Field(default=0, exclude_if=lambda v: v == 0)
    #: For a baseline reading the run ranks as a pre-built start (`judge.starts`): its method label and
    #: origin (`own` / `gsq`). Left out of the JSON for every other record.
    name: str = Field(default="", exclude_if=lambda v: not v)
    origin: Literal["own", "gsq"] | None = Field(default=None, exclude_if=lambda v: v is None)
    #: True on the start a candidate is compared with when the run ranks several (`judge.starts`): the one
    #: with the lowest valid KL. Its name and KL are in this record.
    bar: bool = Field(default=False, exclude_if=lambda v: not v)


class Final(BaseModel):
    """Record the final checkpoint and the rule that selected it.

    Attributes:
        choice: The selected checkpoint, or `None` when a baseline or nothing was selected.
        rule: The `qda.final_rules` key of the rule that decided (`judge.final_rules`); `None` when none did.
        baseline: The selected baseline's path, if the rule selected one.
        reason: One line on why this rule fired.
        why_not: Why each submission after the last valid, eligible one (all of them with none) did not count.
        ledger: Every submitted and scored checkpoint, oldest first, as the judge saw it at the end.
        generation_check: The final rule's generation counts, rates and refusal reasons, when checked.
        chosen_by: Which path of the `handin` contract chose the result: `handin` (the agent's own valid
            hand-in list) or `fallback_candidate0` (no valid hand-in: candidate 0 as-is). Absent for a rule
            that is not that contract.
        engine_floor: `judge.engine_floor`, the answer KL a bf16 GGUF reads against the sealed vLLM bf16
            reference; None for a vLLM search. Stated from the config, never measured here.
        baseline_readings: Baselines read at host start, in KL order, separately from agent submissions; a
            run that ranks pre-built starts (`judge.starts`) labels each with its name and origin.
        candidate0_bar: With `judge.starts`, the start a candidate is compared with: the one with the lowest
            valid KL. Its reading is the first of `baseline_readings`. Absent when the run configures none.
        agent_ran: False when the agent took no turn (no turn returned or ran without failing, no tool call
            was logged, no job of the agent's is in the scheduler and it wrote no output token): the pick is
            the fallback's, and `reason` says why the agent never ran. Absent when it ran.
        warnings: What a reader of the result must not miss, such as a hand-in picked with no reading while
            every reading of the run failed. Absent when there is none.
        metrics_from: The scored entry whose reading `choice.metrics` holds when the picked hand-in was never
            read itself but froze the same files (the same tree digest). Absent otherwise.
        serve: How to serve the picked checkpoint, as the one line `qda build` and `qda-agent evaluate` print;
            absent when qda planned no vLLM server for it (a llama.cpp search).
        reading: Whether the delivered checkpoint has a reading of `judge.metric` in the ledger,
            decided by `delivery`; a baseline as-is has candidate 0's. Absent when nothing was delivered.
        gencheck: Whether the ledger holds a completed long-generation check of the delivered checkpoint,
            decided by `delivery`. Absent when nothing was delivered.
    """

    choice: HandIn | None = None
    rule: str | None
    baseline: Path | None = None
    reason: str = ""
    why_not: list[str] = Field(default_factory=list)
    ledger: list[HandIn] = Field(default_factory=list)
    generation_check: dict[str, Any] = Field(default_factory=dict)
    chosen_by: Literal["handin", "fallback_candidate0"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    engine_floor: float | None = Field(default=None, exclude_if=lambda v: v is None)
    candidate0_bar: Path | None = Field(default=None, exclude_if=lambda v: v is None)
    baseline_readings: list[HandIn] = Field(default_factory=list)
    export: dict[str, Any] = Field(default_factory=dict)
    #: How to serve the picked checkpoint, one `qda_quant.output.serve_command` line; None when qda planned no
    #: vLLM server for it (a llama.cpp search). Left out of the JSON then.
    serve: str | None = Field(default=None, exclude_if=lambda v: v is None)
    sign_in_restarts: list[dict[str, str]] = Field(default_factory=list, exclude_if=lambda v: not v)
    agent_ran: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    warnings: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)
    metrics_from: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reading: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    gencheck: bool | None = Field(default=None, exclude_if=lambda v: v is None)

    def with_delivery(self, metric: str) -> Final:
        """This record with `reading` and `gencheck` decided from it (`delivery`).

        Args:
            metric: The metric the search ranks readings by (`judge.metric`).
        """
        # set on the record itself, as `JudgeHost._with_gencheck` does: `final()` keeps returning one record
        self.reading, self.gencheck = delivery(self.choice, self.baseline, self.generation_check, metric)
        return self

    def is_pick(self, row: Any) -> bool:
        """Whether ``row`` is the ledger entry the result picked: the same id, never the same record.

        The pick can be a copy of its ledger entry carrying another scored entry's reading
        (:attr:`metrics_from`), so the id is its identity. Comparing whole records said "no submission" for a
        submission whose files had also been read.
        """
        return self.choice is not None and bool(self.choice.id) and row.id == self.choice.id

    @model_serializer(mode="wrap")
    def _record(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep the existing final record unchanged when startup baseline readings are disabled."""
        result = handler(self)
        if not self.baseline_readings:
            result.pop("baseline_readings", None)
        if not self.export:
            result.pop("export", None)
        return result


@runtime_checkable
class Host(Protocol):
    """Provide the flow's host operations through blocking, idempotent calls."""

    def start(self) -> None:
        """Bring up the RPC socket, the scheduler and the judge."""

    def cutoff(self) -> None:
        """Stop accepting agent work: queue to judge-only, cancel the agent's jobs."""

    def drain(self, timeout_s: float) -> bool:
        """Wait for pending checks and readings.

        Args:
            timeout_s: Longest wait, in seconds.

        Returns:
            True when nothing is pending any more, False on timeout.
        """
        ...

    def handins(self) -> list[HandIn]:
        """Return submitted and scored checkpoints in the ledger, oldest first."""
        ...

    def final(self) -> Final:
        """Apply the final rules to the current ledger."""
        ...

    def write_final(self, path: Path, final: Final | None = None) -> None:
        """Write the selected final record, or `final()`, as FINAL.json at `path`."""

    def stop(self) -> None:
        """Shut the scheduler and the socket down."""
