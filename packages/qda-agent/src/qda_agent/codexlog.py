"""Recover Codex tool calls from session logs after a run for `tools.jsonl`.

Codex tool calls bypass the flow's `pre_tool_use` hook. `session_calls` converts rollout records
(`sessions/**/rollout-*.jsonl` in the CLI's live directory) to the hook's schema with
`source: "codex-session-log"`. These entries record calls after the fact.

The agent writes the live directory, so the host reads only a copy made by
`sched.snapshot.snapshot_tree` inside a restricted bwrap process. The reader follows no links, bounds
file and line sizes, and parses records as plain JSON.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from qda_agent.sched.snapshot import SnapshotError, snapshot_tree

__all__ = ["SOURCE", "epic_calls", "session_calls"]

#: The `source` of every entry made here (the hook's own entries have none).
SOURCE = "codex-session-log"
#: The rollout payload types that are tool calls, and the field holding each one's input.
CALLS = {"custom_tool_call": "input", "function_call": "arguments", "local_shell_call": "action"}
MAX_FILE = 64 << 20  # one rollout file
MAX_TOTAL = 256 << 20  # every rollout file of the run
MAX_LINE = 4 << 20  # one JSON record
MAX_INPUT = 64 << 10  # the input kept per entry (as JSON text)
MAX_NAME = 200
MAX_ENTRIES = 200_000  # entries kept from one run (memory and tools.jsonl stay bounded)
COPY_TIMEOUT_S = 300.0  # the copy runs during the run's close; a stuck one must not hold it


def session_calls(live: Path, scratch: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a snapshot of Codex rollout files and return `tools.jsonl` entries, oldest first.

    Args:
        live: The CLI's live dir (`<run_dir>/live/codex`), written by the agent.
        scratch: A directory of the host's that the agent cannot see; the copy is made and removed there.

    Returns:
        `(entries, refusals)`: the entries, and one line per file or part that was not read (and why).
    """
    st = os.lstat(live) if os.path.lexists(live) else None
    if st is None or not stat.S_ISDIR(st.st_mode):
        return [], [f"{live.name}: no live dir (or not a directory)"]
    copy_parent = Path(tempfile.mkdtemp(prefix="codex-log-", dir=scratch))
    try:
        try:
            copy = snapshot_tree(live, "sessions", copy_parent / "sessions", timeout_s=COPY_TIMEOUT_S)
        except SnapshotError as e:
            return [], [f"sessions: {e}"]
        return _read(copy)
    finally:
        _remove(copy_parent)


def epic_calls(sessions: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read Codex rollout files in place at `isolation.level: none` and return `tools.jsonl` entries.

    With no wall there is no live dir and nothing a copy would protect: Humanize keeps the run's Codex
    sessions in the run's epic (`<epic>/sessions/codex/sessions`). The reader is `session_calls`'s: it
    follows no links and bounds file, line and entry sizes.

    Returns:
        `(entries, refusals)`, as `session_calls`.
    """
    st = os.lstat(sessions) if os.path.lexists(sessions) else None
    if st is None or not stat.S_ISDIR(st.st_mode):
        return [], [f"{sessions}: no session log (or not a directory)"]
    return _read(sessions)


def _read(top: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The tool-call entries of every rollout file under `top`, oldest first, and what was not read."""
    entries: list[dict[str, Any]] = []
    refused: list[str] = []
    total = 0
    for rel, data, why in _rollouts(top):
        if why:
            refused.append(f"{rel}: {why}")
            continue
        total += len(data)
        if total > MAX_TOTAL:
            refused.append(f"{rel}: over {MAX_TOTAL} bytes of rollouts in all; the rest is not read")
            break
        got, bad = _calls(data)
        room = MAX_ENTRIES - len(entries)
        entries += got[:room]
        if bad:
            refused.append(f"{rel}: {bad} record(s) not read (not JSON, too long, or not a record)")
        if len(got) > room:
            refused.append(f"{rel}: over {MAX_ENTRIES} tool calls in all; the rest is not logged")
            break
    entries.sort(key=lambda e: (e["t"] is None, e["t"] or 0.0))
    return entries, refused


def _remove(top: Path) -> None:
    """Remove the snapshot without following links, making real directories writable first.

    Snapshot copies preserve read-only modes. Traverse with `follow_symlinks=False` so cleanup never
    stats a link's target; `shutil.rmtree` removes the links themselves.
    """
    stack = [top]
    while stack:
        d = stack.pop()
        d.chmod(stat.S_IRWXU)
        with os.scandir(d) as it:
            stack += [Path(e.path) for e in it if e.is_dir(follow_symlinks=False)]
    shutil.rmtree(top)


def _entries(copy: Path) -> Iterator[tuple[Path, str, bool]]:
    """Yield `(path, relative path, is_link)` for non-directory entries in depth-first name order.

    Inspect entries with `follow_symlinks=False` so links are listed without inspecting their targets.
    """
    stack = [copy]
    while stack:
        with os.scandir(stack.pop()) as it:
            found = sorted(it, key=lambda e: e.name)
        for e in reversed(found):  # reversed onto the stack, so names come out in order
            if not e.is_symlink() and e.is_dir(follow_symlinks=False):
                stack.append(Path(e.path))
        for e in found:
            path = Path(e.path)
            if e.is_symlink() or not e.is_dir(follow_symlinks=False):
                yield path, path.relative_to(copy).as_posix(), e.is_symlink()


def _rollouts(copy: Path) -> Iterator[tuple[str, bytes, str]]:
    """Yield `(relative_path, content, refusal)` for rollout files and refused links in the snapshot."""
    for path, rel, link in _entries(copy):
        if link:
            yield rel, b"", "a link; not followed"
            continue
        if not (path.name.startswith("rollout-") and path.name.endswith(".jsonl")):
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            yield rel, b"", "not a regular file (a link or a special file)"
            continue
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                yield rel, b"", "not a regular file"
            elif st.st_size > MAX_FILE:
                yield rel, b"", f"larger than {MAX_FILE} bytes"
            else:
                with os.fdopen(os.dup(fd), "rb") as f:
                    yield rel, f.read(MAX_FILE + 1)[:MAX_FILE], ""
        finally:
            os.close(fd)


def _calls(data: bytes) -> tuple[list[dict[str, Any]], int]:
    """Parse tool-call entries from one rollout and count unreadable records."""
    out: list[dict[str, Any]] = []
    bad = 0
    for raw in data.splitlines():
        if not raw.strip():
            continue
        if len(raw) > MAX_LINE:
            bad += 1
            continue
        try:
            rec = json.loads(raw, parse_constant=_no_constant)
        except (ValueError, RecursionError):
            bad += 1
            continue
        if not isinstance(rec, dict):
            bad += 1
            continue
        payload = rec.get("payload")
        if rec.get("type") != "response_item" or not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if not isinstance(kind, str):  # a list or object here would not even hash
            bad += 1
            continue
        field = CALLS.get(kind)
        if field is None:
            continue
        out.append(
            {
                "t": _epoch(rec.get("timestamp")),
                "ev": "tool",
                "tool": _name(payload),
                "input": _input(payload.get(field), field),
                "source": SOURCE,
            }
        )
    return out, bad


def _no_constant(name: str) -> None:
    """Reject non-JSON numeric constants so they cannot be written into the tool log."""
    raise ValueError(name)


def _epoch(ts: object) -> float | None:
    """Convert an ISO rollout timestamp to epoch seconds, returning None when invalid."""
    if not isinstance(ts, str) or len(ts) > 64:
        return None
    try:
        return round(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp(), 3)
    except ValueError:
        return None


def _name(payload: dict[str, Any]) -> str:
    """Return a bounded tool name, qualified by namespace, with the record type as fallback."""
    name, ns = payload.get("name"), payload.get("namespace")
    name = name if isinstance(name, str) and name else payload.get("type", "?")
    full = f"{ns}.{name}" if isinstance(ns, str) and ns else str(name)
    return full[:MAX_NAME]


def _input(value: object, field: str) -> dict[str, Any]:
    """Normalize call input as a bounded object, decoding JSON argument objects when available."""
    if field == "arguments" and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        inp = decoded if isinstance(decoded, dict) else {"arguments": value}
    else:
        inp = {field: value}
    try:
        text = json.dumps(inp)
    except (TypeError, ValueError, RecursionError):
        return {"unreadable": True}
    return inp if len(text) <= MAX_INPUT else {"truncated": text[:MAX_INPUT]}
