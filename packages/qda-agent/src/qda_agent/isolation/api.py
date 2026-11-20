"""The one isolation interface the flow, the CLI, the scheduler and the judge use.

A run picks a level in its config (`isolation.level`):

* `none` -- plain subprocesses. No Slurm, bwrap, network namespace, host proxy or "node to yourself" check. This is the
  default for `qda-agent run`: agent search must work on a workstation, a cloud VM or a container.
* `sandbox` -- jobs and agent turns run in bwrap with read-only workspace mounts, no network and their own pid namespace
  where the host allows one.
* `strict` -- the experiment wall: everything `sandbox` does, plus the agent's network namespace with the model API reached
  through the host proxy, the dedicated-allocation check and no reach into other runs.

Three things hold at every level, because they are about correctness, not containment:

* every descendant of a job or an agent turn ends when it ends (`KillDomain`);
* the agent cannot change the judge's state, the reference or the scorer -- prevented by the sandbox at `sandbox` and
  `strict`, detected by digest at `none` (`Isolation.verify_trusted`), and a changed digest refuses the reading;
* the run records what it gave the agent and how it was isolated (`Isolation.provenance`).

Nothing outside this package imports `isolation.bwrap`, `isolation.wall`, `isolation.allocation`, `isolation.hostnet` or
`sched.sandbox` internals; they are the implementations behind this interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from hmz.flows import Permission

    from ..config import RunConfig

__all__ = [
    "LEVELS",
    "Handle",
    "Isolation",
    "JobSpec",
    "KillDomain",
    "Launch",
    "Level",
    "job_sandbox",
    "make",
    "make_for_build",
]

Level = Literal["none", "sandbox", "strict"]
LEVELS: tuple[Level, ...] = ("none", "sandbox", "strict")


@dataclass(frozen=True)
class JobSpec:
    """One job as the scheduler or the judge wants it run, before any isolation is applied.

    Args:
        argv: The command.
        cwd: Its working directory.
        root: The directory the job may write (its own output root).
        cards: GPU indices leased to it (empty for a CPU job).
        env: Its environment, complete: nothing is inherited that is not named here.
        readable: Host paths the job may read, beyond the system and Python roots.
        gpu: Whether it needs the GPU driver (device nodes, `/proc/driver/nvidia`).
        binds: Read-only binds at another path, `{path inside: host source}`.
        expected: The `(st_dev, st_ino)` each source had when the plan checked it: a source swapped after
            that check is refused instead of bound.
        host_written: Areas inside `run_dir` that the host writes and the job may bind (the overlay).
        run_dir: The run's directory, resolved; None: no run behind the spec, so bind sources are not
            re-checked against it.
        hide: Paths the job must not reach, on top of the level's own (`sandbox.hide`): a trusted build's
            settings name the receipts and other jobs' output it must stay out of.
        extra_writable: Further directories the job may write, besides `root` (a build's scratch).
        empty_root: Mount an empty tmpfs at `/` before the binds: the job sees only what it is given. A
            trusted build takes this, so nothing of the host's root is reachable beyond its inputs.
    """

    argv: Sequence[str]
    cwd: Path
    root: Path
    cards: Sequence[int] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    readable: Sequence[Path] = ()
    gpu: bool = False
    binds: Mapping[Path, Path] = field(default_factory=dict)
    expected: Mapping[Path, tuple[int, int]] = field(default_factory=dict)
    host_written: Sequence[Path] = ()
    run_dir: Path | None = None
    hide: Sequence[Path] = ()
    extra_writable: Sequence[Path] = ()
    empty_root: bool = False


@dataclass(frozen=True)
class Launch:
    """What `Isolation.job` turned a `JobSpec` into: the argv to exec and the file descriptors it needs.

    Attributes:
        env: The job's environment. `{}` is an empty environment; None inherits the caller's.
    """

    argv: Sequence[str]
    env: Mapping[str, str] | None
    pass_fds: Sequence[int] = ()


class Handle(Protocol):
    """A started job or turn, as its `KillDomain` tracks it."""

    pid: int


class KillDomain(Protocol):
    """Starts a process so that stopping it ends every descendant, at every level.

    The implementation (`isolation.reaper`) runs a reaper as the outermost process: a child subreaper in its own session
    that, on stop, sends TERM to everything below it, waits for the grace, then KILLs what is left and sweeps `/proc` for
    anything that re-parented to it. Where a private pid namespace or a delegated cgroup is available it adds that too.
    """

    def spawn(self, launch: Launch, *, cwd: Path, log: Path) -> Handle: ...

    def stop(self, handle: Handle, grace_s: float) -> None: ...

    def poll(self, handle: Handle) -> int | None: ...


class Isolation(Protocol):
    """A run's isolation, chosen once by `make(cfg)`."""

    level: Level

    @property
    def kill(self) -> KillDomain:
        """Start, poll and stop a job's process tree (`KillDomain`).

        Read-only: a level may build its reaper on first use (`isolation.reaper` is imported lazily), and a
        level that keeps one as an attribute satisfies this too.
        """

        ...

    def preflight(self, cfg: RunConfig) -> None:
        """Refuse the run before its clock starts if this level cannot be held here (raises with the reason)."""

    def session(self, cfg: RunConfig) -> AbstractContextManager[None]:
        """Hold the level for the whole run: at `strict` the agent wall, the host proxy and the sign-in mounts."""

        ...

    def agent_permission(self) -> Permission:
        """What an agent turn may touch, in the agent backend's terms."""

        ...

    def job(self, spec: JobSpec) -> Launch:
        """Turn a job into the argv that runs it at this level."""

        ...

    def size_out(self, root: Path, rel: str) -> int:
        """Bytes the tree `root/rel` holds, sized the way this level copies it.

        The host allocates this much before `copy_out` runs, so it must not open an agent-controlled path
        itself: `sandbox`/`strict` size it in the copy's confinement, `none` walks it (`lstat`).
        """

        ...

    def copy_out(self, root: Path, rel: str, into: Path) -> Path:
        """Copy `root/rel` out of an agent-writable tree into `into`, never following a link the agent planted."""

        ...

    def verify_trusted(self, paths: Sequence[Path]) -> None:
        """Refuse if a trusted path (judge state, reference, scorer) changed since the run started.

        A no-op where the sandbox already prevents the change; a digest comparison at `none`.
        """

    def provenance(self) -> dict[str, object]:
        """The level, the pid and /proc forms, and what was given to the agent: recorded in `resolved.yaml`."""

        ...


def job_sandbox(level: Level, knob: str = "bwrap") -> str:
    """`"bwrap"` or `"none"`: how a run at `level` runs its jobs.

    `none` runs plain subprocesses whatever the legacy knob says; the sandboxed levels keep
    `scheduler.job_sandbox`, whose `none` makes jobs plain while the agent stays walled. One rule, for the
    level objects and for the doctor's sandbox check (`doctor.check_bwrap`).
    """
    return "none" if level == "none" or knob == "none" else "bwrap"


def make(cfg: RunConfig, *, on_skip: Callable[[str, str], None] | None = None) -> Isolation:
    """The isolation for a run's configured level.

    Args:
        cfg: The run config; `cfg.isolation.level` selects the level.
        on_skip: Called as `(bind, why)` for every read-only bind a job's sandbox refuses to make, so the
            flow can record it. None: nothing is recorded (a caller with no run log).
    """
    level = cfg.isolation.level
    if level == "none":
        from .levels.none import NoneIsolation

        return NoneIsolation(cfg)  # nothing is skipped at `none`, so the flow's log callback has no work
    if level == "strict":
        from .levels.strict import StrictIsolation

        return StrictIsolation.for_config(cfg, on_skip=on_skip)
    if level in LEVELS:
        from .levels.legacy import LegacyIsolation

        # `sandbox` holds today's behaviour through the legacy paths until its own module lands.
        return LegacyIsolation.for_config(cfg, on_skip=on_skip)
    raise ValueError(f"isolation.level {level!r} is not one of {', '.join(LEVELS)}")


def make_for_build(level: Level) -> Isolation:
    """The isolation of a trusted build's own process, from the level its launcher wrote into its settings.

    A build runs as its own process, away from the run: its readable sources, binds, hidden trees and
    environment are host-authored into its settings file, so the level is all it needs. `none` runs the
    build as a plain process (the acceptance machine has no bwrap); the sandboxed levels build it in bwrap
    with an empty root. The build's argv is then `Isolation.job`'s, like every job's.
    """
    if level not in LEVELS:
        raise ValueError(f"isolation.level {level!r} is not one of {', '.join(LEVELS)}")
    from .levels.legacy import LegacyIsolation

    return LegacyIsolation(level=level)
