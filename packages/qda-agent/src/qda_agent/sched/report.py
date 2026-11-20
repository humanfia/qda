"""Report card activity, job completion estimates and GPU usage without changing the queue.

Times are epoch seconds from the scheduler's clock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .job import Job, State
from .queue import Scheduler

#: A progress line: ``12/28``, ``layer 12/28``, a tqdm bar's ``12/28 [``.
FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
#: A per-layer line (``layer 12 done``), read with a known layer count.
LAYER = re.compile(r"\blayer\s+(\d+)\b", re.IGNORECASE)


def progress(log: str, layers: int | None = None) -> tuple[float, str] | None:
    """Read the completion fraction and its evidence from a job's last progress line.

    Args:
        log: The job's output (its tail is enough).
        layers: The model's layer count; lets ``layer N`` lines (0-based) count as progress.

    Returns:
        (fraction in (0, 1], what it was read from), or None when no line shows progress.
    """
    for line in reversed(log.splitlines()):
        for a, b in reversed(FRACTION.findall(line)):
            done, total = int(a), int(b)
            if 0 < done <= total:
                return done / total, f"{done}/{total}"
        if layers:
            m = LAYER.findall(line)
            if m and 0 <= int(m[-1]) < layers:
                n = int(m[-1]) + 1
                return n / layers, f"layer {n}/{layers}"
    return None


@dataclass(frozen=True)
class Eta:
    """Record a job's predicted or actual completion time and its source.

    Attributes:
        state: The job's state.
        end: Predicted (or actual) end; None while it is queued.
        basis: How `end` was found: ``progress 12/28``, ``expected duration``, ``ended``, or why it waits.
    """

    state: State
    end: float | None
    basis: str


def eta(sched: Scheduler, job: Job, layers: int | None = None) -> Eta:
    """Estimate completion from progress when available, otherwise from the expected duration."""
    if job.state.terminal:
        return Eta(job.state, job.ended, "ended")
    if job.state is State.queued or job.started is None:
        return Eta(job.state, None, job.reason or "queued")
    now = sched.clock.now()
    got = progress(sched.log(job.name, 200), layers)
    if got is not None and now > job.started:
        frac, what = got
        return Eta(job.state, job.started + (now - job.started) / frac, f"progress {what}")
    return Eta(job.state, job.started + job.expected_s, "expected duration")


@dataclass(frozen=True)
class Card:
    """Describe a card's current job and most recent completed job.

    Attributes:
        card: The card number.
        job: The job running on it (None: idle).
        since: When that job started, or when the card became idle.
        last: The last job that ended on it, with its state (None: none yet).
    """

    card: int
    job: Job | None
    since: float
    last: tuple[str, str] | None


def cards(sched: Scheduler) -> list[Card]:
    """Every card of the run, in the pool's order."""
    jobs = sched.list()
    out = []
    for c in sched.cfg.cards:
        running = next((j for j in jobs if j.state is State.running and c in j.cards), None)
        ended = [j for j in jobs if j.state.terminal and j.started is not None and c in j.cards]
        last = max(ended, key=lambda j: j.ended or 0.0, default=None)
        if running is not None:
            since = running.started or sched.t0
        else:
            since = max((last.ended or sched.t0) if last else sched.t0, sched.t0)
        out.append(Card(c, running, since, (last.name, last.state.value) if last else None))
    return out


def runnable(sched: Scheduler, job: Job) -> bool:
    """Whether a queued job waits for nothing but cards (every `--after` job is done)."""
    deps = {j.name: j for j in sched.list()}
    return job.state is State.queued and all(deps[a].state is State.done for a in job.after if a in deps)


def usage(sched: Scheduler, since: float | None = None, until: float | None = None) -> dict[str, float]:
    """Card-hours of the run in a window, each card-second counted once.

    Keys: ``allocated`` (every card of the pool), leased per job origin and kind (``agent build``, ``agent
    other``, ``judge score``, ``judge check``), and ``idle`` (allocated minus leased).

    Args:
        sched: The queue.
        since: Window start (default and earliest: when the queue started).
        until: Window end (default: now).
    """
    lo = max(sched.t0, since if since is not None else sched.t0)
    hi = sched.clock.now() if until is None else min(until, sched.clock.now())
    span = max(0.0, hi - lo)
    out: dict[str, float] = {"allocated": span * len(sched.cfg.cards) / 3600}
    leased = 0.0
    for j in sched.list():
        if j.started is None:
            continue
        end = j.ended if j.ended is not None else hi
        sec = max(0.0, min(end, hi) - max(j.started, lo)) * len(j.cards)
        if sec:
            key = f"{j.origin} {j.kind.value}"
            out[key] = out.get(key, 0.0) + sec / 3600
            leased += sec
    out["idle"] = max(0.0, out["allocated"] - leased / 3600)
    return out
