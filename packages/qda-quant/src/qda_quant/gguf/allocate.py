"""Allocate a ggml type per tensor class from measured class reads, inside a byte budget.

`sensitivity` says what a class weighs at each type; this module says which types to pick once the KL cost of
each cell is known. The choice is a multiple-choice knapsack: one type per class, cost is the **whole `.gguf`
file size** (header, fixed buckets and each tensor's aligned bytes, the same model the exporter sizes its file
with), value is the class's KL increment over bf16, and the answer is the allocation under the file-size budget
with the least total increment. The whole (file bytes, KL) frontier is emitted, so a caller can name another
budget without re-reading.

An allocation carries both numbers with explicit names: `file_bytes` is the `.gguf` file and is what a budget is
compared against; `payload_bytes` is the tensor payload, reported for internal screening only and always labelled
payload. Costs are discretized to `quantum` bytes, rounding a class's cost up, so a returned allocation always
fits the file-size budget it was produced for. A cell below the floor is counted as zero: a reading inside the baseline's spread
must not buy an allocation, though the raw value stays in the row that produced it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import classes as sensitivity

#: No class's choice is ever worse than its starting point.
BF16 = "bf16"


@dataclass(frozen=True)
class Cell:
    """One measured read: a class at a type, and its KL increment over bf16."""

    klass: str
    kind: str
    d_kl: float
    source: str | None = None


@dataclass(frozen=True)
class Allocation:
    """One type per class: its `.gguf` file bytes, its tensor payload, and its summed increment."""

    picks: dict[str, str]
    file_bytes: int
    payload_bytes: int
    d_kl: float

    def as_dict(self) -> dict[str, Any]:
        """The allocation as JSON-ready data: picks sorted by class, file and payload bytes named apart."""
        return {
            "picks": dict(sorted(self.picks.items())),
            "file_bytes": self.file_bytes,
            "payload_bytes": self.payload_bytes,
            "d_kl": self.d_kl,
        }


def _choices(plan: sensitivity.Plan, by_class: Mapping[str, Mapping[str, Cell]], klass: str, floor: float):
    """``(kind, file bytes, counted dKL)`` for every type this class may take."""
    c = plan.classes[klass]
    start = c.file_bytes.get(BF16)
    if start is None:  # bf16 is always legal; a plan built without a template still sizes it
        start = c.bf16_bytes
    out = [(BF16, start, 0.0)]
    for kind, cell in by_class.get(klass, {}).items():
        got = c.file_bytes.get(kind, c.bytes.get(kind))
        if got is None:
            raise ValueError(f"gguf: {klass}: {kind} is not legal for row widths {c.widths}")
        out.append((kind, got, max(cell.d_kl, 0.0) if cell.d_kl >= floor else 0.0))
    return out


def _solve(plan, by_class, floor, quantum):
    """Every reachable state, keyed by grid units, with its least increment and pick path.

    A class's cost is its file bytes minus its own cheapest choice, ceil-ed to `quantum`: the cheapest
    allocation therefore costs 0 units and is never erased by a coarse grid. Without that anchor a grid
    larger than a class (the 1 MiB default against a sub-MiB fixture) folds every choice into one unit and
    keeps only the least increment, hiding the smaller, feasible file.
    """
    if quantum <= 0:
        raise ValueError("gguf: quantum must be positive")
    states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
    for klass in plan.classes:
        choices = _choices(plan, by_class, klass, floor)
        lowest = min(nbytes for _, nbytes, _ in choices)
        nxt: dict[int, tuple[float, dict[str, str]]] = {}
        for units, (value, picks) in states.items():
            for kind, nbytes, d_kl in choices:
                cost = units + -(-(nbytes - lowest) // quantum)
                got = value + d_kl
                old = nxt.get(cost)
                if old is None or got < old[0]:
                    nxt[cost] = (got, {**picks, klass: kind})
        states = nxt
    return states


def _bytes(plan: sensitivity.Plan, picks: Mapping[str, str]) -> int:
    """The exact `.gguf` file size of `picks` (header, fixed buckets and aligned class bytes)."""
    return plan.file_total(picks)


def _index(plan: sensitivity.Plan, cells: Iterable[Cell]) -> dict[str, dict[str, Cell]]:
    """``{class: {kind: cell}}``, refusing a class the plan lacks and a non-finite increment."""
    by_class: dict[str, dict[str, Cell]] = {}
    for cell in cells:
        if cell.klass not in plan.classes:
            raise ValueError(f"gguf: {cell.klass}: not an allocatable class")
        if not math.isfinite(cell.d_kl):
            raise ValueError(f"gguf: {cell.klass} {cell.kind}: the increment is not finite ({cell.d_kl})")
        by_class.setdefault(cell.klass, {})[cell.kind] = cell
    return by_class


def frontier(
    plan: sensitivity.Plan,
    cells: Iterable[Cell],
    *,
    floor: float = 0.0,
    quantum: int = 1 << 20,
) -> list[Allocation]:
    """The nondominated allocations, ascending in bytes and descending in increment."""
    if not math.isfinite(floor):
        raise ValueError(f"gguf: the floor is not finite ({floor})")
    by_class = _index(plan, cells)
    states = _solve(plan, by_class, floor, quantum)
    out: list[Allocation] = []
    best = float("inf")
    for nbytes, value, picks in sorted(
        ((_bytes(plan, picks), value, picks) for value, picks in states.values()),
        key=lambda row: (row[0], row[1]),
    ):
        if value < best:
            best = value
            out.append(Allocation(picks, nbytes, plan.payload_total(picks), value))
    return out


def allocate(
    plan: sensitivity.Plan,
    cells: Iterable[Cell],
    budget: int,
    *,
    floor: float = 0.0,
    quantum: int = 1 << 20,
) -> Allocation:
    """The least-increment allocation whose `.gguf` file fits `budget`; refuses a budget below every choice."""
    got = frontier(plan, cells, floor=floor, quantum=quantum)
    least = min((a.file_bytes for a in got), default=0)
    if not got or budget < least:
        raise ValueError(f"gguf: {budget} B is below the smallest allocation's {least} B")
    # the frontier is ascending in bytes, so the last entry that fits has the least increment
    return min((a for a in got if a.file_bytes <= budget), key=lambda a: (a.d_kl, a.file_bytes))


def one_allocation(data: Any) -> dict[str, Any]:
    """The one allocation in a `qda sensitivity table` result: a dict, or a list holding exactly one.

    `table --budget-bytes --json` prints a one-entry list while `table --json` without a budget prints the whole
    frontier, which names no single allocation.

    Raises:
        ValueError: `data` holds no allocation, or more than one.
    """
    if isinstance(data, list):
        if len(data) > 1:
            raise ValueError(
                f"sensitivity: this result holds {len(data)} allocations (a frontier, table without "
                "--budget-bytes); name --budget-bytes to pick one"
            )
        data = data[0] if data else None
    if not isinstance(data, dict) or not isinstance(data.get("picks"), dict):
        raise ValueError('sensitivity: expected an allocation, {"picks": {"class": "type"}}')
    return data


def read_kl(path: str | Path) -> float:
    """The ``kl`` of one accepted `qda-kl score` result (``OUT/result.json``); a non-finite value is refused."""
    p = Path(path)
    if p.is_dir():
        p = p / "result.json"
    got = json.loads(p.read_text())
    if "kl" not in got:
        raise ValueError(f"gguf: {p}: no kl in this result")
    kl = float(got["kl"])
    if not math.isfinite(kl):
        raise ValueError(f"gguf: {p}: the kl is not finite ({got['kl']})")
    return kl


def rows_from_manifest(path: str | Path, *, baseline: float | None = None) -> list[Cell]:
    """Read a manifest of reads: ``{"class", "type", "out" | "d_kl"}`` per row.

    ``out`` names a run directory (or its ``result.json``); its ``kl`` minus `baseline` is the increment. A row
    that carries ``d_kl`` is used as is, which is how a table is rebuilt without the run directories.
    """
    if baseline is not None and not math.isfinite(baseline):
        raise ValueError(f"gguf: the baseline is not finite ({baseline})")
    cells = []
    for row in json.loads(Path(path).read_text()):
        klass, kind = row.get("class"), row.get("type")
        if not klass or not kind:
            raise ValueError(f"gguf: a manifest row needs class and type: {row}")
        if kind == BF16:
            raise ValueError("gguf: bf16 is the baseline, not a cell; give it as the manifest's baseline")
        if "d_kl" in row:
            d_kl = float(row["d_kl"])
            if not math.isfinite(d_kl):
                raise ValueError(f"gguf: {klass} {kind}: the increment is not finite ({row['d_kl']})")
        elif baseline is not None and row.get("out"):
            d_kl = read_kl(row["out"]) - baseline
        else:
            raise ValueError(f"gguf: {klass} {kind}: needs d_kl, or out and a baseline")
        cells.append(Cell(klass, kind, d_kl, row.get("out")))
    return cells


@dataclass(frozen=True)
class Additivity:
    """A measured mixed read against the sum of its classes' reads."""

    measured: float
    predicted: float

    @property
    def relative_error(self) -> float:
        """``|measured - predicted|`` over the larger of the two; 0 when both are 0."""
        scale = max(abs(self.measured), abs(self.predicted))
        return abs(self.measured - self.predicted) / scale if scale else 0.0

    def within(self, tolerance: float = 0.15) -> bool:
        """Whether the gap is at most `tolerance` of the larger reading."""
        return self.relative_error <= tolerance


def summed(cells: Iterable[Cell], picks: Mapping[str, str]) -> float:
    """The increment a `picks` allocation predicts by summing its classes' cells (zero where none was read)."""
    by_class: dict[str, dict[str, float]] = {}
    for cell in cells:
        by_class.setdefault(cell.klass, {})[cell.kind] = cell.d_kl
    return sum(max(by_class.get(k, {}).get(kind, 0.0), 0.0) for k, kind in picks.items() if kind != BF16)


def additivity(measured: float, predicted: float) -> Additivity:
    """Compare a measured mixed read with the sum of its classes' reads."""
    return Additivity(measured, predicted)
