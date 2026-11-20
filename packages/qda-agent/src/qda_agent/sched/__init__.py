"""Expose the run's priority GPU queue, clocks and job launch backends."""

from .backend import Backend, FakeBackend, FakeRun, LocalBackend, kill_every_group
from .clock import Clock, FakeClock, SystemClock
from .config import SchedConfig
from .job import Job, Kind, State
from .queue import Refused, Scheduler


def make_backend(cfg: SchedConfig, clock: Clock, isolation: object | None = None) -> Backend:
    """Create the launch backend selected by `cfg`, using `clock` for simulated jobs."""
    backend = cfg.backend(clock)
    if hasattr(backend, "iso"):
        backend.iso = isolation  # type: ignore[attr-defined]
    return backend


__all__ = [
    "Backend",
    "Clock",
    "FakeBackend",
    "FakeClock",
    "FakeRun",
    "Job",
    "Kind",
    "LocalBackend",
    "kill_every_group",
    "Refused",
    "SchedConfig",
    "Scheduler",
    "State",
    "SystemClock",
    "make_backend",
]
