"""Configure the run's GPU queue and job launch settings (`qda_agent.schedulers`).

The scheduler starts jobs locally on the run's allocated cards.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator
from qda_quant.registry import Registry

from qda_agent.settings import Plugin, validate_plugin

if TYPE_CHECKING:
    from .backend import Backend
    from .clock import Clock

__all__ = ["Qsched", "SchedConfig", "Scheduler"]


class Scheduler(Plugin):
    """Define settings for a GPU scheduler plugin."""


class Qsched(Scheduler):
    """Configure the priority GPU queue and its launch backend.

    Attributes:
        tick_s: Seconds between two scheduler passes of the host.
        kill_grace_s: Seconds between SIGTERM and SIGKILL when a job is stopped.
        job_sandbox: `bwrap` runs every agent job in the sandbox of `qda_agent.sched.sandbox`; `none` runs it
            as a plain host process with the shell's `PATH` (it can read the judge's files). A run resolves
            it with its `isolation.level` (`RunConfig.job_sandbox`): unset is `none` at level `none` and
            `bwrap` at `sandbox` and `strict`; level `none` always runs plain jobs.
        readable: Extra read-only paths of the job sandbox (the original model, the calibration data, ...).
        gpus: False: the cards are queue slots only and no job sees a GPU (`CUDA_VISIBLE_DEVICES` empty, no
            device node), so the run needs no GPU and any card numbers do; tests set it.
        fake_gpus: Start nothing: every job "runs" for a fixed time on the scheduler's clock (tests only).
        judge_first: The judge's jobs (KL readings, submission checks) go ahead of every queued agent job. False:
            one first-in first-out line for every job.
    """

    name: Literal["qsched"] = "qsched"
    tick_s: float = Field(default=2.0, gt=0)
    kill_grace_s: float = Field(default=10.0, ge=0)
    job_sandbox: Literal["bwrap", "none"] = "bwrap"
    readable: list[Path] = Field(default_factory=list)
    gpus: bool = True
    fake_gpus: bool = False
    judge_first: bool = True

    def backend(self, clock: Clock) -> Backend:
        """Select the fake or local process-group backend from these settings."""
        from .backend import FakeBackend, LocalBackend

        if self.fake_gpus:
            return FakeBackend(clock)
        return LocalBackend(self.kill_grace_s)


SCHEDULERS = Registry(
    "schedulers",
    {"qsched": Qsched},
    group="qda_agent.schedulers",
    on_add=partial(validate_plugin, "qda_agent.schedulers"),
)


class SchedConfig(Qsched):
    """Add the run's cards, deadline, drain window and job environment to the queue settings.

    Attributes:
        name: The scheduler plugin these settings came from (`qsched`, ...).
        cards: The run's pool, as PCI-bus-order ordinals: card `i` is the `i`-th GPU of the driver sorted by
            PCI bus id (`CUDA_DEVICE_ORDER=PCI_BUS_ID`); not the `/dev/nvidia<minor>` number, which may
            differ. Each is resolved once at start to its GPU UUID; jobs see their cards by UUID. No two
            running jobs ever share one.
        deadline: The GPU end. No job starts if its expected end passes it.
        budget_s: Used when `deadline` is unset: the deadline is the host's start plus this.
        judge_grace_s: How long after the GPU deadline the judge's jobs (KL readings, submission checks) are still
            admitted and started, so that a check asked for just before the deadline runs while the flow
            drains; set it to the flow's drain timeout.
        agent_job_env: Environment of every agent job (from the toolset: `PATH`, the serve env, ...): over the
            sandbox's clean default where there is one, over the host environment at `isolation.level: none`.
            A job that runs on the host (`job_sandbox(...) == "none"`) keeps the launching `PATH` after the
            toolset's directories (`RunConfig.agent_job_env`).
        binds: Read-only binds of every agent job at another path, `{path inside: host path}` (from the
            toolset: the baselines at `baselines/<N>`).
        run_dir: The run's directory, resolved (from `RunConfig.sched_config`); None: no run behind these
            settings, so the bind sources are not re-checked when a job starts.
        host_written: Areas inside `run_dir` the host writes and the jobs may bind read-only (the toolset's
            overlay). The scheduler's own `jobs/` is always allowed.
        hide: Paths no job may reach (`sandbox.hide`), masked after every job mount at each spelling; a hide
            covering a path the job needs is refused at start.
    """

    name: str = "qsched"  # type: ignore[assignment]
    judge_first: bool = True
    cards: list[int] = Field(min_length=1)
    deadline: datetime | None = None
    budget_s: float | None = Field(default=None, gt=0)
    judge_grace_s: float = Field(default=900.0, ge=0)
    agent_job_env: dict[str, str] = Field(default_factory=dict)
    binds: dict[Path, Path] = Field(default_factory=dict)
    run_dir: Path | None = None
    host_written: list[Path] = Field(default_factory=list)
    hide: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> SchedConfig:
        if len(set(self.cards)) != len(self.cards):
            raise ValueError(f"sched.cards has duplicates: {self.cards}")
        return self

    def resolve_deadline(self, start: float) -> float:
        """Return the deadline as epoch seconds, falling back to `start + budget_s` or infinity."""
        if self.deadline is not None:
            return self.deadline.timestamp()
        return start + self.budget_s if self.budget_s is not None else float("inf")
