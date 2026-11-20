"""Whether this node is the run's alone (`pid: shared` needs it).

`pid: shared` (see `bwrap.pid_form`) puts the agent's process in the host's pid namespace, so a
same-user signal or a ptrace reach every other process of this user on the node. That is acceptable only when
there is nothing else of ours to reach: no other running job of this user on the node, and no other process of
this user outside the run's own tree (this process, its ancestors and its descendants). Another step of the
same Slurm job is still another run's work, and the agent could signal it. The allocation's own batch-step
keepers are the exception: every Slurm allocation has the job's batch script and its idle children, and they
are not another run's work. Slurm itself names them as the pids of the job's batch step (`scontrol listpids
<job>.batch`), so they are recognized by step and never by process name. `why_not_dedicated` answers None in
that case and otherwise names what it found.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

from .turn_guard import descendants, rows

__all__ = ["why_not_dedicated"]


class _Stranger(NamedTuple):
    pid: int
    name: str


def why_not_dedicated(proc: Path = Path("/proc")) -> str | None:
    """None if the run has the node to itself; else a sentence naming what else is on it.

    Args:
        proc: The `/proc` to read (the host's, on the node the run starts on).
    """
    jobs, strangers = _other_jobs(), _other_processes(proc)
    if not jobs and not strangers:
        return None
    found = []
    if jobs:
        found.append(f"other running job(s) of this user on this node: {', '.join(jobs)}")
    if strangers:
        shown = ", ".join(f"{pid} ({name})" for pid, name in strangers[:5])
        more = f" and {len(strangers) - 5} more" if len(strangers) > 5 else ""
        found.append(f"other process(es) of this user on this node: {shown}{more}")
    return "; ".join(found)


def _other_jobs() -> list[str]:
    """Running Slurm jobs of this user that name this node besides the one this process belongs to."""
    job = os.environ.get("SLURM_JOB_ID", "").split(".")[0]
    node = os.environ.get("SLURMD_NODENAME")
    user = os.environ.get("USER")
    if not job or not node or not user:
        return []  # not under Slurm: the process check below is the whole answer
    out = _slurm(["squeue", "-h", "-w", node, "-o", "%i %u"])
    if out is None:
        return []
    jobs = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == user and fields[0].split(".")[0] != job:
            jobs.append(fields[0])
    return jobs


def _batch_step_pids() -> set[int]:
    """The pids Slurm lists for this allocation's batch step: the batch script and its keepers, not by name.

    Slurm tracks the job's batch step and its processes (`scontrol listpids <job>.batch`), which is the one
    place that says these processes belong to the allocation itself rather than to another run. Without
    Slurm, or when the step cannot be queried, nothing is excused and the check stays strict.
    """
    job = os.environ.get("SLURM_JOB_ID", "").split(".")[0]
    if not job:
        return set()
    out = _slurm(["scontrol", "listpids", f"{job}.batch"])
    if out is None:
        return set()
    pids = set()
    for line in out.splitlines():
        first = line.split()[:1]
        if first and first[0].isdigit():
            pids.add(int(first[0]))
    return pids


def _slurm(argv: list[str]) -> str | None:
    """A Slurm command's stdout, or None when Slurm is absent or the command fails."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _other_processes(proc: Path) -> list[_Stranger]:
    """Processes of this user on the node outside this run's tree (`_tree`) and the allocation's keepers."""
    pid = os.getpid()
    ours = _tree(pid, _parents(proc)) | _batch_step_pids()
    uid = os.getuid()
    others = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        other = int(entry.name)
        if other in ours:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            name = (entry / "comm").read_text().strip()
        except OSError:
            continue
        others.append(_Stranger(other, name))
    return sorted(others)


def _parents(proc: Path) -> dict[int, int]:
    """`{pid: ppid}` for every process `proc` lists (`stat`, or `status` where `stat` is unreadable)."""
    return {pid: ppid for pid, (_state, ppid, _session) in rows(proc).items()}


def _tree(pid: int, parents: dict[int, int]) -> set[int]:
    """`pid`, its ancestors and its own descendants (not their siblings: another run's tree is not ours)."""
    tree = {pid}
    walk = pid
    while parents.get(walk) and parents[walk] not in tree:
        walk = parents[walk]
        tree.add(walk)
    return tree | descendants(pid, parents)
