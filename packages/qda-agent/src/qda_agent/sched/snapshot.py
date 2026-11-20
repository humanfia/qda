"""Copy what the agent names out of its workdir, inside a bwrap of its own.

The implementation is `qda_agent.isolation.snapshot` (an isolation level holds the copy, through
`Isolation.copy_out`); this module keeps the public names its callers import: the error, the size limits, the
removal helper and the three copy entry points, which delegate.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

#: Largest file `snapshot_file` accepts, and its longest copy.
SNAPSHOT_MAX = 16 << 20
SNAPSHOT_TIMEOUT_S = 30.0


class SnapshotError(ValueError):
    """A file the agent named could not be snapshotted; the message says why (paths relative to the workdir)."""


def snapshot_file(
    root: Path, rel: str, into: Path, max_bytes: int = SNAPSHOT_MAX, timeout_s: float = SNAPSHOT_TIMEOUT_S
) -> Path:
    """Copy `root/rel` to `into/<name>` without the host ever following a path the agent controls."""
    from qda_agent.isolation.snapshot import snapshot_file as implementation

    return implementation(root, rel, into, max_bytes=max_bytes, timeout_s=timeout_s)


def snapshot_tree(root: Path, rel: str, dst: Path, timeout_s: float = 3600.0) -> Path:
    """Copy the directory `root/rel` to `dst` without the host following a path the agent controls."""
    from qda_agent.isolation.snapshot import snapshot_tree as implementation

    return implementation(root, rel, dst, timeout_s=timeout_s)


def snapshot_tree_bytes(root: Path, rel: str, into: Path) -> int:
    """Size a tree before allocating its snapshot, in the same confinement as the copy."""
    from qda_agent.isolation.snapshot import snapshot_tree_bytes as implementation

    return implementation(root, rel, into)


def remove_tree(p: Path) -> None:
    """Remove a file, link or read-only tree (a snapshot, a sealed frozen copy): directories get write back first."""
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        for d in [p, *(x for x in p.rglob("*") if x.is_dir() and not x.is_symlink())]:
            d.chmod(stat.S_IRWXU)
        shutil.rmtree(p)
