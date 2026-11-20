"""A judge job's temp directory, and the short absolute name the job is given for it.

A job's scratch lives in the judge's dir, `<run>/judge/tmp/<key>` (mode 0700), never in the host's `$TMPDIR`,
which the agent's fence may grant. Sockets are bound in it -- torch's symmetric memory at
`$TMPDIR/symm_mem-<id>` (a multi-card job), vLLM's IPC under `VLLM_RPC_BASE_PATH` -- and the kernel caps a
socket path at 107 bytes, which a deep run dir outgrows (a 113-byte temp dir refused a 2-card gencheck).

So the job names its temp dir by a short absolute link to it, made when the job is queued in the host's
private directory for these links (`mkdtemp`, mode 0700, under `rpc.private_base()`, as the host's own socket
directory): the same name whatever directory a child of the job runs in (ninja runs nvcc in a build dir), and
short whatever the run dir's depth. A link is removed once its job has ended, and the directory with the host.
Judge jobs run on the host, outside the agent's sandbox (`Scheduler._command`), so every process of the job
sees the same name.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from ..rpc import private_base

if TYPE_CHECKING:
    from ..sched import Scheduler

#: The prefix of the host's private link directory.
PREFIX = "qda-jt-"


def example() -> Path:
    """A long temp-dir name a judge job is given (its 9999th): what `doctor` checks against the socket limit."""
    return Path(os.path.realpath(private_base())) / (PREFIX + "X" * 8) / "9999"


class JobTemps:
    """The temp dirs of a host's judge jobs, and the short links that name them.

    Safe to call from several threads: the host's tick folds ended jobs (`release_ended`) while the final
    rule's check makes and releases its own link outside the host's lock.

    Args:
        jdir: The judge's dir; the jobs' temp dirs are made in its `tmp/`.
    """

    def __init__(self, jdir: Path) -> None:
        self.jdir = Path(os.path.abspath(jdir))
        self.base: Path | None = None  # made at the first link, removed by `close`
        self._remove: weakref.finalize | None = (
            None  # removes `base` at `close`, else when collected or at exit
        )
        self.links: dict[str, Path] = {}  # job name -> its link
        self.count = 0
        # Guards `base`, `links` and `count`. Taken last: nothing else is locked while it is held.
        self._lock = threading.Lock()

    def make(self, name: str, job: str) -> Path:
        """Make the temp dir `tmp/<name>` (mode 0700) for the job named `job`; return the name it is given.

        That is a short absolute link to the directory, fixed now: it does not depend on the cwd of any
        process of the job.

        Raises:
            FileExistsError: The temp dir exists already (each job has its own).
        """
        real = self.jdir / "tmp" / name
        real.mkdir(parents=True, mode=0o700)
        with self._lock:
            if self.base is None:
                # resolved: a link in the base's own path would make every name longer than counted
                self.base = Path(os.path.realpath(tempfile.mkdtemp(prefix=PREFIX, dir=private_base())))
                self._remove = weakref.finalize(self, shutil.rmtree, self.base, True)
            self._release(job)  # a retry under the same job name never keeps the old link
            self.count += 1
            short = self.base / str(self.count)
            short.symlink_to(real.resolve(), target_is_directory=True)
            self.links[job] = short
            return short

    def release(self, job: str) -> None:
        """Remove the link of the job named `job`, if it has one (its temp dir stays in the judge's dir)."""
        with self._lock:
            self._release(job)

    def _release(self, job: str) -> None:
        if (short := self.links.pop(job, None)) is not None:
            short.unlink(missing_ok=True)

    def release_ended(self, sched: Scheduler) -> None:
        """Remove the link of every job that has ended.

        A link whose job is not queued yet stays: it is made just before the job is submitted, and a
        submission that fails removes it itself.
        """
        with self._lock:
            if not self.links:
                return
        ended = {
            j.name for j in sched.list(origin="judge") if j.state.terminal
        }  # the scheduler's lock, not ours
        with self._lock:
            for job in [job for job in self.links if job in ended]:
                self._release(job)

    def close(self) -> None:
        """Remove every link and the private directory holding them (the host has stopped its jobs)."""
        with self._lock:
            self.links.clear()
            if self._remove is not None:
                self._remove()  # the links in it go with it
            self.base = self._remove = None
