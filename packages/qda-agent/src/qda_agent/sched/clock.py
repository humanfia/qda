"""Provide real and controllable clocks for the scheduler and judge."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Expose epoch time and waiting so callers can use a real or fake clock."""

    def now(self) -> float:
        """Return the current time in seconds since the epoch."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait for `seconds`, or advance simulated time when using a fake clock."""
        ...


class SystemClock:
    """Read system time and wait using the operating system."""

    def now(self) -> float:
        """Return `time.time()`."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Sleep for `seconds`, treating negative durations as zero."""
        time.sleep(max(0.0, seconds))


class FakeClock:
    """Advance simulated time explicitly, without waiting.

    Args:
        t0: The starting time in seconds.
    """

    def __init__(self, t0: float = 1_000_000.0) -> None:
        self.t = t0

    def now(self) -> float:
        """Return the fake time."""
        return self.t

    def sleep(self, seconds: float) -> None:
        """Advance the fake time by `seconds` without waiting."""
        self.t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        """Advance the fake time by `seconds`."""
        self.sleep(seconds)
