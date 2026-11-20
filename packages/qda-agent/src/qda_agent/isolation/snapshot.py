"""Copy what the agent names out of its workdir, inside a bwrap of its own.

The host never follows a path the agent controls: `snapshot_file` and `snapshot_tree` run `cp` in a bwrap that
sees only the workdir (read-only), the destination and the system, then open the copy without following links.
`remove_tree` removes a (read-only) snapshot the job or the host no longer needs.

This is `Isolation.copy_out`'s implementation for today's levels; `sched/snapshot.py` re-exports it so its
existing callers keep working.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

from qda_agent.sched.snapshot import SNAPSHOT_MAX, SNAPSHOT_TIMEOUT_S, SnapshotError, remove_tree

from .bwrap import pid_args, proc_args


def system_binds() -> list[str]:
    """Bwrap arguments that make `/usr` and the `/bin`, `/lib*`, `/sbin` entries visible read-only."""
    out = ["--ro-bind", "/usr", "/usr"]
    for p in ("/bin", "/sbin", "/lib", "/lib32", "/lib64"):
        q = Path(p)
        if q.is_symlink():
            out += ["--symlink", str(q.readlink()), p]
        elif q.is_dir():
            out += ["--ro-bind", p, p]
    return out


def snapshot_file(
    root: Path, rel: str, into: Path, max_bytes: int = SNAPSHOT_MAX, timeout_s: float = SNAPSHOT_TIMEOUT_S
) -> Path:
    """Copy `root/rel` to `into/<name>` without the host ever following a path the agent controls.

    `cp` runs in a bwrap that sees only `root` (read-only, at `/w`), `into` (at `/snap`) and the system:
    whatever `rel`, its parents or the file itself are swapped for while the copy runs, a symlink can only
    lead to those. The host then opens the copy without following links and accepts only a regular file of
    at most `max_bytes`.

    Args:
        root: The agent's work directory.
        rel: The file, relative to `root`; no `..` components.
        into: An existing directory the agent cannot write (the job's `files/`).
        max_bytes: Largest snapshot accepted.
        timeout_s: Longest copy (a FIFO the agent put there would block `cp`).

    Returns:
        The snapshot (read-only).

    Raises:
        SnapshotError: `rel` is absolute or has `..`, the copy failed, or it is not a regular file.
    """
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
    dst = into / parts[-1]
    _confined_cp(root, into, ["--no-dereference", "--preserve=mode"], "/".join(parts), dst, rel, timeout_s)
    try:
        fd = os.open(dst, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        _unlink(dst)
        raise SnapshotError(f"{rel}: not a file (a link or a special file)") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SnapshotError(f"{rel}: not a regular file")
        if st.st_size > max_bytes:
            raise SnapshotError(f"{rel}: larger than {max_bytes >> 20} MiB")
        os.fchmod(fd, 0o444)
    except SnapshotError:
        _unlink(dst)
        raise
    finally:
        os.close(fd)
    return dst


def _unlink(p: Path) -> None:
    """Remove `p` if it is there."""
    with contextlib.suppress(FileNotFoundError):
        p.unlink()


def _confined_cp(
    root: Path, into: Path, flags: list[str], rel: str, dst: Path, shown: str, timeout_s: float
) -> None:
    """`cp FLAGS /w/REL /snap/NAME` in a bwrap that sees only `root` (read-only), `into` and `/usr`."""
    argv = _confined_argv(root, into) + [
        "/usr/bin/cp",
        *flags,
        "--",
        "./" + rel,
        "/snap/" + dst.name,
    ]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        remove_tree(dst)
        raise SnapshotError(f"{shown}: could not be copied ({type(e).__name__})") from None
    if p.returncode != 0:
        remove_tree(dst)
        error = next(
            (line.strip() for line in p.stderr.splitlines() if line.strip()),
            f"bwrap exited {p.returncode}",
        )
        raise SnapshotError(
            f"{shown}: not readable in the work directory; snapshot copy failed: {error}; "
            "run 'qda doctor --search CONFIG.yaml' to check sandbox support"
        )


def _confined_argv(root: Path, into: Path) -> list[str]:
    """The shared confinement for sizing and copying an agent-controlled tree."""
    return [
        shutil.which("bwrap") or "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-user",  # what `--unshare-all` gave this copy, kept where the pid form does not add it
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        # A private pid namespace, with the /proc `sandbox.proc_args` gives one (a fresh procfs where this host
        # mounts one in it, else the host's read-only): neither `--unshare-all` nor the sandbox's own choice
        # may put a fresh procfs it cannot mount inside a private namespace (see `sandbox._pair`).
        *pid_args("private"),
        "--clearenv",
        *system_binds(),
        "--dev",
        "/dev",
        *proc_args(pid="private"),
        "--ro-bind",
        str(root),
        "/w",
        "--bind",
        str(into),
        "/snap",
        "--chdir",
        "/w",
        "--",
    ]


def snapshot_tree_bytes(root: Path, rel: str, into: Path) -> int:
    """Size a tree before allocating its snapshot, in the same confinement as the copy.

    Regular files count once per inode (`cp -a` preserves hardlinks); symlinks are never followed.
    Only numeric metadata leaves the sandbox, so the host never opens an agent-controlled path.
    """
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
    argv = _confined_argv(root, into) + [
        "/usr/bin/find",
        "-P",
        "./" + rel,
        "-type",
        "f",
        "-printf",
        "%D %i %s\n",
    ]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SNAPSHOT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SnapshotError(f"{rel}: could not be sized ({type(error).__name__})") from None
    if result.returncode:
        raise SnapshotError(f"{rel}: could not be sized in the work directory: {result.stderr.strip()}")
    files = {}
    for line in result.stdout.splitlines():
        dev, inode, size = map(int, line.split())
        files[dev, inode] = size
    return sum(files.values())


def snapshot_tree(root: Path, rel: str, dst: Path, timeout_s: float = 3600.0) -> Path:
    """Copy the directory `root/rel` to `dst` without the host following a path the agent controls.

    `cp -a` runs in the same bwrap as `snapshot_file`: symlinks are copied as links (the caller refuses
    them on the copy), and whatever the agent swaps in while it runs can only lead to `root` or `/usr`.

    It asks for a reflink clone (`--reflink=auto`): where the filesystem shares blocks copy-on-write, the copy
    costs neither time nor disk, which matters for a checkpoint of hundreds of GiB. A clone is still a copy,
    since the first write to either side splits the shared blocks. Where the filesystem cannot clone, `cp`
    copies the bytes in the same pass.

    Args:
        root: The agent's work directory.
        rel: The directory, relative to `root`; no `..` components ("." is `root` itself).
        dst: Where the copy goes; must not exist; its parent must exist and be the host's.
        timeout_s: Longest copy.

    Returns:
        `dst`, a real directory (not a link).

    Raises:
        SnapshotError: `rel` is absolute or has `..`, the copy failed, or it is not a directory.
    """
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts:
        raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
    _confined_cp(
        root,
        dst.parent,
        ["-a", "--no-dereference", "--reflink=auto"],
        "/".join(parts) or ".",
        dst,
        rel,
        timeout_s,
    )
    st = os.lstat(dst)
    if not stat.S_ISDIR(st.st_mode):
        remove_tree(dst)
        raise SnapshotError(f"{rel}: not a directory")
    return dst


def copy_out(root: Path, rel: str, into: Path) -> Path:
    """Copy `root/rel` out of the agent's tree, never following a link the agent planted.

    A directory is copied with `snapshot_tree`, to `into` as its own new path (which must not exist); a file
    is copied with `snapshot_file`, into the directory `into`, under its own name. The type is read with
    `lstat`, so a link the agent planted is a file to the copy (and is refused), never a directory to walk.

    Args:
        root: The agent-writable tree.
        rel: The entry, relative to `root`; no `..` components.
        into: The destination (see above).

    Returns:
        The copy's path.
    """
    try:
        mode = os.lstat(Path(root) / rel).st_mode
        # A link is copied as a link, never walked: the tree copy does that (and refuses a non-directory
        # copy), so a link the agent planted is refused with the same message it always was.
        tree = stat.S_ISDIR(mode) or stat.S_ISLNK(mode)
    except OSError:
        tree = False  # missing (or dangling): the file copy reports it
    return snapshot_tree(root, rel, into) if tree else snapshot_file(root, rel, into)
