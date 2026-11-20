"""A live run's progress, read from the files the run already writes.

`watch` streams one stderr line per judge transition while `qda-agent run` works; `status` is the same view on
demand, for a run whose terminal the user has left. Neither records anything: `flow.jsonl` says where the search's
clock and the agent phase are, and `judge/ledger.jsonl` says what the judge did, each line as it was written.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from qda_agent.judge.ledger import ENTRY_KINDS, Entry

__all__ = ["QUIET_S", "TICK_S", "Progress", "status", "watch"]

#: How often the stream looks at the run's files.
TICK_S = 15.0
#: A judge that has written nothing for this long is reported: about three times the usual gap between lines.
QUIET_S = 10 * 60.0


def _minutes(seconds: float) -> str:
    """`seconds` as whole minutes, never negative."""
    return f"{max(0, round(seconds / 60))} min"


def _hhmm(when: float) -> str:
    """`when` (epoch seconds) as `HH:MM` UTC."""
    return datetime.fromtimestamp(when, UTC).strftime("%H:%M")


def _epoch(text: object) -> float | None:
    """The ISO datetime `text` as epoch seconds, or None if it is not one."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _record(line: str) -> dict[str, Any] | None:
    """The object a JSONL line holds, or None if it is not one."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _entry(line: str) -> Entry | None:
    """The ledger line as an `Entry`, or None for a line that is not one (a prune record)."""
    record = _record(line)
    if record is None or record.get("kind") not in ENTRY_KINDS:
        return None
    return Entry.model_validate(record)


class _Tail:
    """The complete lines a JSONL file has gained since the last read."""

    def __init__(self, path: Path) -> None:
        self.path, self.pos = path, 0

    def read(self) -> list[str]:
        try:
            with self.path.open("rb") as f:
                f.seek(self.pos)
                chunk = f.read()
        except OSError:  # the file is not written yet
            return []
        *lines, tail = chunk.split(b"\n")
        self.pos += len(chunk) - len(tail)  # a half-written line waits for its newline
        return [line.decode("utf-8", "replace") for line in lines if line]


class Progress:
    """A run's progress as its own files state it, read as they grow."""

    def __init__(self, run_dir: Path, *, now: float, quiet_s: float = QUIET_S) -> None:
        self.opened, self.quiet_s = now, quiet_s
        self.flow = _Tail(run_dir / "flow.jsonl")
        self.ledger = _Tail(run_dir / "judge" / "ledger.jsonl")
        self.first: float | None = None  # the first flow event: the search's clock starts around it
        self.deadline: float | None = None
        self.entries: dict[str, Entry] = {}  # the newest entry per id
        self.active: float | None = None  # the newest time the judge wrote a transition
        self.last = ""  # the newest transition's line
        self._drawn: dict[str, str] = {}  # the line already streamed per id
        self._quiet_at: float | None = None

    def read(self) -> tuple[list[Entry], bool]:
        """The entries whose line changed since the last read, and whether the agent phase began now."""
        began = False
        for line in self.flow.read():
            event = _record(line)
            if event is None:
                continue
            when = event.get("t")
            if not isinstance(when, (int, float)) or isinstance(when, bool):
                continue
            if self.first is None:
                self.first = float(when)
            if event.get("ev") == "start":
                self.deadline = _epoch(event.get("deadline"))
                began = True
        changed: list[Entry] = []
        for line in self.ledger.read():
            entry = _entry(line)
            if entry is None:
                continue
            self.entries[entry.id] = entry
            when = max(entry.submitted, entry.ended or 0.0)
            self.active = when if self.active is None else max(self.active, when)
            text = entry.line().strip()
            if self._drawn.get(entry.id) != text:
                self._drawn[entry.id] = text
                self.last = text
                changed.append(entry)
        return changed, began

    def clock(self, now: float) -> str:
        """`34 min elapsed, ~26 min remaining`; the start phase until the deadline is known."""
        elapsed = now - (self.first if self.first is not None else self.opened)
        if self.deadline is None:
            return f"start phase: {_minutes(elapsed)} so far"
        left = self.deadline - now
        if left < 0:
            return f"{_minutes(elapsed)} elapsed; the deadline passed {_minutes(-left)} ago"
        return f"{_minutes(elapsed)} elapsed, ~{_minutes(left)} remaining"

    def entry_lines(self) -> list[str]:
        """Each entry's newest line, oldest first."""
        return [entry.line().strip() for entry in self.entries.values()]

    def quiet(self, now: float) -> str | None:
        """The notice due now when the judge has been silent for `quiet_s`, and at most once per `quiet_s`."""
        waited = now - self._since()
        if waited < self.quiet_s or (self._quiet_at is not None and now - self._quiet_at < self.quiet_s):
            return None
        self._quiet_at = now
        tail = f" (last: {self.last})" if self.last else " (nothing submitted yet)"
        return f"{self.clock(now)}: no judge activity for {_minutes(waited)}{tail}"

    def _since(self) -> float:
        """When the judge last wrote, else when the run did, else when this view opened."""
        for when in (self.active, self.first):
            if when is not None:
                return when
        return self.opened

    def stream(self, now: float) -> list[str]:
        """The lines to print now: the agent phase's start, each transition, and a quiet notice when due."""
        changed, began = self.read()
        out = []
        if began:
            deadline = f"; the deadline is {_hhmm(self.deadline)} UTC" if self.deadline is not None else ""
            out.append(f"{self.clock(now)}: the agent phase begins{deadline}")
        out += [f"{self.clock(now)}: {entry.line().strip()}" for entry in changed]
        if (quiet := self.quiet(now)) is not None:
            out.append(quiet)
        return out

    def report(self, now: float) -> list[str]:
        """The current view: the clock with the deadline, each entry's newest line, and a quiet notice."""
        self.read()
        clock = self.clock(now)
        if self.deadline is not None:
            clock += f" (the deadline is {_hhmm(self.deadline)} UTC)"
        out = [clock, *self.entry_lines()]
        if (quiet := self.quiet(now)) is not None:
            out.append(quiet)
        return out


def _ended(run_dir: Path) -> str | None:
    """How a finished run ended, from its `result.json`; None while it runs or before that file exists."""
    try:
        result = json.loads((run_dir / "result.json").read_text())
    except (OSError, ValueError):
        return None
    picked = result.get("choice") or result.get("baseline") or "no checkpoint"
    return (
        f"the run finished: {result.get('ended', 'unknown')}; "
        f"the rule {result.get('rule', 'unknown')} picked {picked}"
    )


def status(run_dir: Path, *, clock: Callable[[], float] = time.time) -> list[str] | None:
    """A run's progress now, one line per fact; None when `run_dir` is not a run's directory."""
    if not run_dir.is_dir() or not any(
        (run_dir / name).is_file() for name in ("flow.jsonl", "resolved.yaml", "result.json")
    ):
        return None
    now = clock()
    view = Progress(run_dir, now=now)
    if (ended := _ended(run_dir)) is not None:
        view.read()
        return [ended, *view.entry_lines()]
    return view.report(now)


@contextmanager
def watch(
    run_dir: Path,
    *,
    stream: IO[str] | None = None,
    clock: Callable[[], float] = time.time,
    tick_s: float = TICK_S,
    quiet_s: float = QUIET_S,
) -> Iterator[None]:
    """Stream a run's progress to stderr while the block runs, one line per judge transition."""
    out = sys.stderr if stream is None else stream
    view, stop = Progress(run_dir, now=clock(), quiet_s=quiet_s), threading.Event()

    def emit() -> None:
        for line in view.stream(clock()):
            try:
                print(f"qda-agent: {line}", file=out, flush=True)
            except (OSError, ValueError):  # stderr has gone away (a closed terminal): the run goes on
                stop.set()
                return

    def loop() -> None:
        while not stop.wait(tick_s):
            emit()

    with contextlib.suppress(OSError, ValueError):  # a closed stderr must not stop a run
        print(
            f"qda-agent: progress on stderr; qda-agent status {run_dir} prints the same view on demand",
            file=out,
            flush=True,
        )
    thread = threading.Thread(target=loop, name="qda-agent-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5.0)
        emit()  # the transitions the run wrote since the last tick: none is dropped at the end
