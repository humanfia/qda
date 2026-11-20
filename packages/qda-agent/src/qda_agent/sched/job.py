"""Represent queued commands, their resource requirements and execution state."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Kind(StrEnum):
    """Classify a job to set its scheduling priority.

    `score` (a KL reading) and `check` (a submission check) are the judge's and go ahead of every queued
    `build` or `other` job; a running job is never stopped for them.
    """

    score = "score"
    check = "check"
    build = "build"
    other = "other"

    @property
    def priority(self) -> int:
        """Higher runs first."""
        return 1 if self in (Kind.score, Kind.check) else 0


class State(StrEnum):
    """Track a job from queuing through completion or cancellation."""

    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        """Whether the job has ended."""
        return self in (State.done, State.failed, State.cancelled)


class Job(BaseModel):
    """Store a queued command, its resource requirements and execution history.

    Attributes:
        id: `<seq>-<name>`, the job's directory under `<run_dir>/jobs/`.
        seq: Submission order (ties within a priority go first-in first-out).
        name: Unique within the run; dependencies name it.
        kind: See `Kind`.
        origin: `agent` or `judge`; `cutoff` cancels the agent's jobs and keeps the judge's.
        argv: The command as run, its script rewritten to the snapshot.
        cwd: Directory the command runs in.
        ncards: Number of cards it needs.
        expected_s: Expected duration; admission checks `start + expected_s <= deadline`.
        after: Names of jobs that must end `done` first; one that fails or is cancelled fails this one.
        state: See `State`.
        cards: The cards it runs (ran) on.
        submitted, started, ended: Epoch seconds.
        rc: Exit code once ended (None when it never ran).
        reason: Why it waits, or why it ended as it did.
        root: The agent's work directory (writable in the sandbox); None for the judge's jobs.
        env: Extra environment of the job (the judge's: its evaluator on PYTHONPATH).
        build_private: Host-owned custody directory for a trusted build; never set by agent job requests.
        expected_bytes: The scheduler's reservation for this agent job's output against the run's disk cap:
            a full checkpoint's estimate when it declares one, else `judge.job_reserve_bytes` (0: nothing
            reserved; a trusted build's estimate lives in its `request.json`).
    """

    id: str
    seq: int
    name: str
    kind: Kind = Kind.build
    origin: str = "agent"
    argv: list[str]
    cwd: str
    ncards: int = Field(default=1, ge=1)
    expected_s: float = Field(gt=0)
    after: list[str] = []
    state: State = State.queued
    cards: list[int] = []
    submitted: float
    started: float | None = None
    ended: float | None = None
    rc: int | None = None
    reason: str = ""
    root: str | None = None
    env: dict[str, str] = {}
    build_private: str | None = None
    expected_bytes: int = 0

    @property
    def priority(self) -> int:
        """Higher runs first."""
        return self.kind.priority
