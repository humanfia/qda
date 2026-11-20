"""`isolation.level: none`: plain subprocesses, for a workstation, a cloud VM or a container.

No bubblewrap, network namespace, host proxy, Slurm or "node to yourself" check: a search at this level
runs wherever the user can run a process. Two things still hold, because they are about correctness:

* every job and agent turn runs under the run's `KillDomain` (the reaper): stopping it ends every
  descendant, including a `setsid` grandchild that would otherwise re-parent away;
* the judge's state, the reference and the scorer are digested before every job and at the end
  (`isolation.trusted`); a changed digest refuses the job and is recorded.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ...sched.snapshot import SnapshotError
from ..api import Launch
from ..snapshot import SNAPSHOT_MAX
from ..trusted import RECORD, TrustChanged, Trusted, trusted_paths

if TYPE_CHECKING:
    from hmz.flows import Permission

    from ...config import RunConfig
    from ..api import JobSpec, KillDomain, Level

__all__ = ["NoneIsolation"]


class NoneIsolation:
    """The product level: plain subprocesses under the reaper, and a digest check of the judge's inputs.

    Args:
        cfg: The run config.
        kill: The kill domain to start jobs through (the reaper); None imports `isolation.reaper.Reaper`,
            so a test can pass a stand-in.
        on_skip: Unused here: `none` runs plain processes by design, so no sandbox is skipped. Accepted so
            `make(cfg, on_skip=...)` can hand every level the same arguments.
    """

    level: Level = "none"

    def __init__(
        self,
        cfg: RunConfig,
        *,
        kill: KillDomain | None = None,
        on_skip: Callable[[str], None] | None = None,
    ) -> None:
        del on_skip  # nothing is skipped at this level
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.tamper_check = getattr(cfg.isolation, "tamper_check", "digest")
        self.trusted = (
            Trusted(trusted_paths(cfg), record=self.run_dir / RECORD, level=self.level)
            if self.tamper_check == "digest"
            else None
        )
        self._kill = kill

    @property
    def kill(self) -> KillDomain:
        """The run's kill domain: the reaper, the outermost process of every job and turn at every level."""
        if self._kill is None:
            from ..reaper import Reaper

            self._kill = Reaper()
        return self._kill

    def preflight(self, cfg: RunConfig) -> None:
        """Nothing to hold here: the level needs no bwrap, Slurm or dedicated node.

        What a job may read is still checked by the flow (`isolation.check`: the prompt set the agent must not
        read), and a build receipt or judge custody still needs bwrap (refused by the run config).
        """

    @contextmanager
    def session(self, cfg: RunConfig) -> Iterator[None]:
        """Take the trusted digests for the whole run; record (do not raise) a change found at its end."""
        if self.trusted is None:  # tamper_check: none
            yield
            return
        self.trusted.take()
        try:
            yield
        finally:
            with contextlib.suppress(TrustChanged):
                self.trusted.verify("run end")

    def agent_permission(self) -> Permission:
        """What an agent turn may touch, exactly as the run's sandbox setting said before the level existed.

        `none` walls nothing, but the turn's permission is still the sandbox setting's (`isolation.permission`):
        a run that keeps the product default (`sandbox: bwrap`) hands the same permission here as it did on the
        flow's own path, so choosing `none` changes only the wall, never the turn.
        """
        from hmz.flows import Permission

        from qda_agent.isolation import permission

        return Permission() if self.cfg is None else permission(self.cfg.sandbox)

    def job(self, spec: JobSpec) -> Launch:
        """Run the job as given, plain, after the trusted paths still match what the run started with.

        Raises:
            TrustChanged: A trusted path changed; the scheduler fails the job with this reason.
        """
        if self.trusted is not None:
            self.trusted.verify("job")
        # The environment composes exactly as today's plain job did (isolation.levels.legacy): the caller's
        # environment with the job's own on top, and None (inherit) for a spec that names none.
        env = {**os.environ, **dict(spec.env)} if spec.env else None
        return Launch(argv=tuple(spec.argv), env=env, pass_fds=())

    def size_out(self, root: Path, rel: str) -> int:
        """Bytes the tree `root/rel` holds, walked with `lstat`: `none` has no sandbox to size it in.

        Regular files count once per inode (a hardlink adds no bytes), as the confined find does at the
        sandboxed levels; a link is never followed. `rel` is checked as `copy_out` checks it.

        Raises:
            SnapshotError: `rel` is absolute or has `..`.
        """
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
        total, seen = 0, set()
        for here, _dirs, files in os.walk(Path(root) / rel, followlinks=False):
            for name in files:
                try:
                    st = (Path(here) / name).lstat()
                except OSError:
                    continue
                inode = (st.st_dev, st.st_ino)
                if stat.S_ISREG(st.st_mode) and inode not in seen:
                    seen.add(inode)
                    total += st.st_size
        return total

    def copy_out(self, root: Path, rel: str, into: Path) -> Path:
        """Copy `root/rel` out of the agent's tree, never following a link it planted.

        The contract is `isolation.snapshot.copy_out`'s: a directory is copied to `into` as its own new path
        (which must not exist); a file is copied into the directory `into` under its own name. Links inside a
        copied tree travel as links, as they do at the other levels: the caller refuses them before reading.

        Raises:
            SnapshotError: `rel` is absolute or has `..`, is missing, is a link, or is not a directory or
                regular file there.
        """
        parts = Path(rel).parts
        if Path(rel).is_absolute() or ".." in parts:
            raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
        src, into = Path(root) / rel, Path(into)
        try:
            mode = os.lstat(src).st_mode
        except OSError as e:
            raise SnapshotError(f"{rel}: not readable in the work directory ({e.strerror})") from None
        if stat.S_ISLNK(mode):
            raise SnapshotError(f"{rel}: not a file (a link or a special file)")
        if stat.S_ISDIR(mode):
            if into.exists() or into.is_symlink():
                raise SnapshotError(f"{into}: already exists; copy into a new path")
            _copy_tree(src, into)
            return into
        if not stat.S_ISREG(mode):
            raise SnapshotError(f"{rel}: neither a directory nor a regular file")
        dst = into / src.name
        if dst.exists() or dst.is_symlink():
            raise SnapshotError(f"{dst}: already exists")
        _copy_file(src, dst)
        return dst

    def copy_file(self, root: Path, rel: str, into: Path, max_bytes: int = SNAPSHOT_MAX) -> Path:
        """Copy the regular file `root/rel` to `into/<name>`, never following a link.

        The file-only contract of `isolation.snapshot.snapshot_file` without a bwrap to copy in: `none` has
        no sandbox, so the host opens the source itself (`O_NOFOLLOW`: a link is refused, not followed) and
        refuses what that function refuses, with its wording -- a path outside the tree, a link ("not a
        file"), a directory or special file ("not a regular file"), and a file over `max_bytes`.

        Raises:
            SnapshotError: `rel` is absolute or has `..`, the source cannot be opened, or it is not one
                regular file within the limit.
        """
        parts = Path(rel).parts
        if not parts or Path(rel).is_absolute() or ".." in parts:
            raise SnapshotError(f"{rel}: give a path inside the work directory, without '..'")
        src, into = Path(root) / rel, Path(into)
        try:
            # O_NONBLOCK as `snapshot_file` opens its copy: a FIFO the agent put there never blocks the host.
            fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise SnapshotError(f"{rel}: not a file (a link or a special file)") from None
            raise SnapshotError(f"{rel}: not readable in the work directory ({e.strerror})") from None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise SnapshotError(f"{rel}: not a regular file")
            if st.st_size > max_bytes:
                raise SnapshotError(f"{rel}: larger than {max_bytes >> 20} MiB")
            dst = into / src.name
            if dst.exists() or dst.is_symlink():
                raise SnapshotError(f"{dst}: already exists")
            try:
                with os.fdopen(fd, "rb") as f:
                    fd = -1  # the file object owns the descriptor now
                    with dst.open("wb") as out:
                        shutil.copyfileobj(f, out, 1 << 20)  # 1 MiB at a time, not the whole file
            except BaseException:
                dst.unlink(missing_ok=True)
                raise
        finally:
            if fd >= 0:
                os.close(fd)
        os.chmod(dst, stat.S_IMODE(st.st_mode))  # after the bytes, as `_copy_file` does
        return dst

    def verify_trusted(self, paths: Sequence[Path] = ()) -> None:
        """Refuse when a trusted path changed since the run started.

        Args:
            paths: A caller's own paths. One that holds or lies inside a path of the run's set (the judge's
                state, the reference, the scorer) is already watched and adds nothing; any other is trusted
                from now on. The run's own set is always compared.

        Raises:
            TrustChanged: A trusted path changed; the event is recorded in `run_dir/trust.jsonl`.
        """
        if self.trusted is None:  # tamper_check: none
            return
        for path in paths:
            if not self.trusted.covered(path):
                self.trusted.add(path)
        self.trusted.verify("reading")

    def provenance(self) -> dict[str, object]:
        """The level, the pid form (None: nothing is walled) and the digests taken."""
        return {
            "level": self.level,
            "pid_form": None,
            "tamper_check": self.tamper_check,
            "trusted": {} if self.trusted is None else self.trusted.digests,
        }


def _copy_file(src: Path, dst: Path) -> None:
    """Copy one regular file, streaming it, and keep its mode bits as `cp -a` does.

    The source is opened without following a link (a link swapped in since the caller's `lstat` is refused
    here), and its bytes are streamed in chunks: a checkpoint shard is never read whole into memory. The
    copy ends with the source's permission bits, an executable bit included, so a script the agent wrote
    stays runnable in the judge's copy.

    Raises:
        SnapshotError: The source cannot be opened (a link, a permission error, an I/O error).
    """
    try:
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as e:
        raise SnapshotError(f"{src.name}: not readable ({e.strerror})") from None
    with os.fdopen(fd, "rb") as f:
        mode = stat.S_IMODE(os.fstat(f.fileno()).st_mode)
        with dst.open("wb") as out:
            shutil.copyfileobj(f, out, 1 << 20)  # 1 MiB at a time, not the whole file
    os.chmod(dst, mode)  # after the bytes: a failure here is not hidden behind a masked chmod


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree; a link travels as a link (never followed), and a caller refuses it."""
    dst.mkdir()
    for name in sorted(os.listdir(src)):
        child, target = src / name, dst / name
        mode = os.lstat(child).st_mode
        if stat.S_ISDIR(mode):
            _copy_tree(child, target)
        elif stat.S_ISLNK(mode):
            os.symlink(os.readlink(child), target)
        elif stat.S_ISREG(mode):
            _copy_file(child, target)
        else:
            raise SnapshotError(f"{child}: neither a directory, a regular file nor a link")
