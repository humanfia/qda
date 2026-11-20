"""qsched: one priority queue over the run's cards.

Rules (paper §5.3 "Scheduling"):

- A job states its cards, its expected duration and the jobs it runs after.
- Its script is snapshotted at submission: a later edit of the file changes nothing.
- A running job holds its cards alone; no two running jobs share a card.
- With `judge_first`, the judge's jobs (`score`, `check`) go ahead of queued builds without preempting
  running jobs. Otherwise, jobs run in submission order. The first job that does not fit reserves the
  free cards it can use, preventing smaller jobs behind it from starving it.
- No job starts if `now + expected_s` passes its deadline; such a job fails at once, as it never could.
  An agent job's deadline is the GPU deadline; a judge job's is the GPU deadline plus `judge_grace_s`,
  so checks and readings asked for just before the GPU deadline still run while the flow drains.

The queue is a plain object driven by `tick()`; the host calls it every `tick_s` and tests call it on a
fake clock. Every state change is written to `<run_dir>/jobs/<id>/job.json`; output goes to `log`.
"""

from __future__ import annotations

import os
import re
import signal
import tempfile
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qda_agent.isolation import devices
from qda_agent.isolation.api import Isolation, JobSpec, Launch

from .backend import Backend
from .clock import Clock
from .config import SchedConfig
from .job import Job, Kind, State
from .snapshot import SnapshotError, remove_tree

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
FROZEN = "{frozen}"


class Refused(ValueError):
    """A request the queue will not take; the message says why."""


#: How a receipted build's argv is built: `(scheduler, job, cards) -> argv`. The judge registers its own
#: (`judge.build_receipt.command`) at import, so the scheduler holds no judge code of its own.
BUILD_COMMAND: Callable[[Scheduler, Job, list[int]], list[str]] | None = None


def register_build_command(command: Callable[[Scheduler, Job, list[int]], list[str]]) -> None:
    """Let the judge build a receipted build's argv (called by `qda_agent.judge.build_receipt`)."""
    global BUILD_COMMAND  # noqa: PLW0603 (one hook for the one judge)
    BUILD_COMMAND = command


def snapshot(
    path: str, cwd: str, root: Path, into: Path, copy_out: Callable[[Path, str, Path], Path]
) -> Path:
    """Snapshot the file `path` (relative to `cwd`, inside `root`) into `into/files/` (`Isolation.copy_out`).

    Raises:
        Refused: `path` is absolute, leaves `root`, or is not a regular file there.
    """
    if Path(path).is_absolute():
        raise Refused(f"{path}: give the path relative to the work directory")
    rel = os.path.normpath(Path(os.path.relpath(cwd, root), path))
    if rel == ".." or rel.startswith("../"):
        raise Refused(f"{path} is outside the work directory")
    files = into / "files"
    files.mkdir(parents=True, exist_ok=True)
    try:
        staged = copy_out(root, rel, files)
    except SnapshotError as e:
        raise Refused(str(e)) from None
    if staged.is_dir():
        remove_tree(staged)  # a directory is not a script: the copy is not kept
        raise Refused(f"{path}: not a file")
    return staged


def exit_reason(rc: int) -> str:
    """Why a job failed, in words the agent can act on (`qsched show`, `log`, `wait`).

    A negative code is a signal that ended the job's own process (`subprocess`'s code); 137 is a shell or
    bwrap reporting that the process it ran was SIGKILLed (`128 + 9`), as a kill inside the sandbox or the
    kernel's out-of-memory killer shows up.
    """
    if rc == 128 + signal.SIGKILL:
        return f"exit {rc}: {_KILLED}"
    if rc >= 0:
        return f"exit {rc}"
    if -rc == signal.SIGKILL:
        return _KILLED
    try:
        return f"killed by signal {-rc} ({signal.Signals(-rc).name})"
    except ValueError:
        return f"killed by signal {-rc}"


_KILLED = "killed by signal 9 (SIGKILL), not by the job itself; often out of memory or an operator kill"


class Scheduler:
    """The run's queue and its cards.

    Args:
        run_dir: The run's host-side directory; jobs live under `run_dir/jobs/`.
        cfg: The scheduler's settings (`RunConfig.sched_config`).
        backend: Launches the jobs.
        clock: Time source (a `FakeClock` in tests).
        proc: The driver's GPU table (`/proc/driver/nvidia/gpus`; a fake one in tests).

    Raises:
        CardError: A pool card names no GPU of the driver, or the machine has no NVIDIA driver while
            `cfg.gpus` is true (exact-card backends).
    """

    def __init__(
        self,
        run_dir: Path,
        cfg: SchedConfig,
        backend: Backend,
        clock: Clock,
        proc: Path | None = None,
        isolation: Isolation | None = None,
    ) -> None:
        self.dir = Path(run_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.backend, self.clock = cfg, backend, clock
        self.t0 = clock.now()  # when the queue started (`report.usage` counts card time from here)
        self.deadline = cfg.resolve_deadline(self.t0)
        self.proc = proc
        if isolation is None:  # a queue built on its own settings (tests, the judge CLI): today's behaviour
            from qda_agent.isolation.levels.legacy import LegacyIsolation

            isolation = LegacyIsolation.for_sched(cfg)
        self.iso: Isolation = isolation
        self.uuids: dict[int, str] = {}
        if backend.exact_cards and cfg.gpus:  # resolved once: a card number is a PCI-bus-order ordinal
            uuids = devices.card_uuids(cfg.cards, proc)
            if cfg.cards and not uuids:
                raise devices.CardError(
                    f"cards {cfg.cards}: this machine has no NVIDIA driver (no {proc or devices.NV_PROC}); "
                    "set scheduler.gpus: false to run the queue without GPUs"
                )
            self.uuids = dict(zip(cfg.cards, uuids, strict=True))
        self.jobs: dict[str, Job] = {}  # name -> job, in submission order
        self.handles: dict[str, object] = {}
        self.judge_only = False
        self.closed = False  # `stop_all` ran: nothing more is admitted or started
        self.check_start: Callable[[], None] | None = None
        #: The one thread that starts jobs (`tick` from any other only reaps). A sandboxed job runs under
        #: `bwrap --die-with-parent`, whose parent death signal fires when the *thread* that forked it ends: a
        #: job started from a request's thread (`qsched wait` ticks the queue) was SIGKILLed when the request
        #: returned. The host's tick thread takes this over (`JudgeHost.start`); else the main thread.
        self.starter = threading.main_thread()
        self.lock = threading.RLock()

    def submit(
        self,
        argv: list[str],
        cwd: str,
        *,
        expected_s: float,
        ncards: int = 1,
        name: str | None = None,
        after: Iterable[str] = (),
        kind: Kind = Kind.build,
        origin: str = "agent",
        root: Path | None = None,
        env: dict[str, str] | None = None,
        freeze: str | None = None,
        build_private: Path | None = None,
        expected_bytes: int = 0,
    ) -> Job:
        """Queue a command.

        With `freeze` (the agent's jobs), that file is snapshotted now -- it must resolve inside `root`
        (default `cwd`) -- and every `{frozen}` in `argv` becomes the snapshot; an empty `argv` runs
        `bash <snapshot>`. A later edit of the file changes nothing.

        Agent jobs (`origin != "judge"`) run in the sandbox when `cfg.job_sandbox` is `bwrap`, with `root`
        (default `cwd`) writable; `env` is added to the job's environment.

        Raises:
            Refused: Bad name, unknown dependency, more cards than the pool, an expected end past the
                deadline, or the queue taking judge jobs only.
        """
        after = list(after)
        if not argv and freeze is None:
            raise Refused("no command")
        with self.lock:
            self._admit(name, after, ncards, expected_s, origin)
        # the copy runs outside the lock (a FIFO or a slow disk must not stall the queue), into a staging
        # dir of the host's; the job dir is named, and the job recorded, under the lock afterwards
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.dir))
        try:
            run_argv = list(argv)
            if freeze is not None:
                staged = snapshot(
                    freeze, cwd, Path(root if root is not None else cwd), stage, self.iso.copy_out
                )
                run_argv = [a.replace(FROZEN, "{snap}") for a in argv] if argv else ["bash", "{snap}"]
            with self.lock:
                seq = len(self.jobs) + 1
                if name is None:  # auto-named under the lock; never one the agent took (job2 named by hand)
                    name = next(
                        f"job{n}" for n in range(seq, seq + len(self.jobs) + 2) if f"job{n}" not in self.jobs
                    )
                self._admit(
                    name, after, ncards, expected_s, origin
                )  # again, with the name: the queue moved on
                jid = f"{seq:04d}-{name}"
                jdir = self.dir / jid
                stage.rename(jdir)
                stage = None
                if freeze is not None:
                    snap = str(jdir / staged.relative_to(staged.parents[1]))
                    run_argv = [a.replace("{snap}", snap) for a in run_argv]
                job = Job(
                    id=jid,
                    seq=seq,
                    name=name,
                    kind=kind,
                    origin=origin,
                    argv=run_argv,
                    cwd=cwd,
                    ncards=ncards,
                    expected_s=expected_s,
                    after=after,
                    submitted=self.clock.now(),
                    root=str(root if root is not None else cwd) if origin != "judge" else None,
                    env=env or {},
                    build_private=str(build_private) if build_private is not None else None,
                    expected_bytes=expected_bytes,
                )
                self.jobs[name] = job
                self._save(job)
                return job
        except OSError as e:  # nothing of a failed submission is left behind (the name stays free)
            raise Refused(f"{freeze}: could not be copied ({type(e).__name__})") from None
        finally:
            if stage is not None:
                remove_tree(stage)

    def _admit(self, name: str | None, after: list[str], ncards: int, expected_s: float, origin: str) -> None:
        """Refuse a submission the queue cannot take now (called under the lock)."""
        if name is not None and not NAME.fullmatch(name):
            raise Refused(f"bad name {name!r} (letters, digits, . _ -; at most 64)")
        if name is not None and name in self.jobs:
            raise Refused(f"a job named {name!r} exists")
        if missing := [a for a in after if a not in self.jobs]:
            raise Refused(f"--after names unknown jobs: {', '.join(missing)}")
        if ncards < 1:
            raise Refused("a job needs at least one card")
        if ncards > len(self.cfg.cards):
            raise Refused(f"{ncards} cards asked, the run has {len(self.cfg.cards)}")
        if expected_s <= 0:
            raise Refused("the expected duration must be > 0")
        if self.closed:
            raise Refused("the run is closing; the queue takes no more jobs")
        if self.judge_only and origin != "judge":
            raise Refused("the GPU deadline has passed; the queue takes the judge's jobs only")
        if self.clock.now() + expected_s > self.deadline_of(origin):
            raise Refused(self._late(expected_s, origin))

    def cancel(self, name: str, origin: str | None = None) -> Job:
        """Cancel a queued job or stop a running one.

        Args:
            name: The job.
            origin: When set, only a job of this origin may be cancelled (the agent's own).

        Raises:
            Refused: Unknown job, or one of another origin.
        """
        with self.lock:
            job = self.get(name, origin)
            if job.state is State.running:
                self.backend.stop(self.handles.pop(name))
                self._end(job, State.cancelled, None, "cancelled while running")
            elif job.state is State.queued:
                self._end(job, State.cancelled, None, "cancelled before it started")
            return job

    def get(self, name: str, origin: str | None = None) -> Job:
        """The job named `name` (of `origin`, when set)."""
        with self.lock:
            job = self.jobs.get(name)
        if job is None or (origin is not None and job.origin != origin):
            raise Refused(f"no job {name!r}")
        return job

    def list(self, origin: str | None = None) -> list[Job]:
        """Every job (of `origin`, when set) in submission order."""
        with self.lock:
            return [j.model_copy() for j in self.jobs.values() if origin is None or j.origin == origin]

    def log(self, name: str, lines: int = 50, origin: str | None = None) -> str:
        """The last `lines` lines of a job's output."""
        p = self.dir / self.get(name, origin).id / "log"
        text = p.read_text(errors="replace") if p.exists() else ""
        return "\n".join(text.splitlines()[-lines:])

    def cutoff(self) -> list[Job]:
        """At the GPU deadline: take judge jobs only and cancel every other job, queued or running."""
        with self.lock:
            self.judge_only = True
            return [
                self.cancel(j.name)
                for j in list(self.jobs.values())
                if j.origin != "judge" and not j.state.terminal
            ]

    def pending(self, origin: str | None = None) -> bool:
        """Whether a job (of `origin`) is queued or running."""
        with self.lock:
            return any(
                not j.state.terminal for j in self.jobs.values() if origin is None or j.origin == origin
            )

    def stop_all(self) -> None:
        """Cancel everything and close the queue (the host shuts down).

        Running jobs are stopped together, so the close takes one `kill_grace_s`, not one per job. A request
        still in flight (`qsched wait` ticks the queue) can start nothing afterwards.
        """
        with self.lock:
            self.closed = True
            running = [j for j in self.jobs.values() if j.state is State.running]
            with ThreadPoolExecutor(max_workers=max(1, len(running))) as pool:
                list(pool.map(self.backend.stop, [self.handles.pop(j.name) for j in running]))
            for j in running:
                self._end(j, State.cancelled, None, "cancelled while running")
            for j in list(self.jobs.values()):
                if not j.state.terminal:
                    self.cancel(j.name)

    def tick(self) -> None:
        """Reap ended jobs, then start what can start (in the `starter` thread only)."""
        with self.lock:
            self._reap()
            self._schedule()

    def _reap(self) -> None:
        for name, h in list(self.handles.items()):
            rc = self.backend.poll(h)
            if rc is not None:
                del self.handles[name]
                job = self.jobs[name]
                self._end(
                    job, State.done if rc == 0 else State.failed, rc, "" if rc == 0 else exit_reason(rc)
                )

    def _schedule(self) -> None:
        if self.closed or threading.current_thread() is not self.starter:
            return
        now = self.clock.now()
        busy = {c for j in self.jobs.values() if j.state is State.running for c in j.cards}
        free = [c for c in self.cfg.cards if c not in busy]
        held: set[int] = set()
        held_by: str | None = None
        lanes = self.cfg.judge_first  # else one first-in first-out line
        queued = sorted(
            (j for j in self.jobs.values() if j.state is State.queued),
            key=lambda j: (-j.priority if lanes else 0, j.seq),
        )
        for job in queued:
            if self.judge_only and job.origin != "judge":
                continue
            if now + job.expected_s > self.deadline_of(job.origin):
                self._end(job, State.failed, None, self._late(job.expected_s, job.origin))
                continue
            wait = self._deps(job)
            if wait is not None:
                self._wait(job, wait)
                continue
            avail = [c for c in free if c not in held]
            if len(avail) < job.ncards:
                if held_by is None:
                    held_by, held = job.name, set(free[: job.ncards])
                    self._wait(job, "waiting for cards (free cards held for it)")
                else:
                    self._wait(job, f"waiting for cards (behind {held_by})")
                continue
            cards = avail[: job.ncards]
            free = [c for c in free if c not in cards]
            self._launch(job, cards)

    def _deps(self, job: Job) -> str | None:
        """Return the dependency wait reason, failing the job if a dependency failed or was cancelled."""
        deps = [self.jobs[d] for d in job.after]
        for d in deps:  # a failed dependency fails the job at once, whatever else it still waits for
            if d.state.terminal and d.state is not State.done:
                self._end(job, State.failed, None, f"after {d.name}: {d.state.value}")
                return "failed"
        d = next((d for d in deps if d.state is not State.done), None)
        return None if d is None else f"waiting: after {d.name} ({d.state.value})"

    def _command(self, job: Job, cards: list[int]) -> Launch:
        """The launch a job is started with: its isolation level builds the sandbox for an agent job.

        On an exact-card backend every job, the judge's on the host included, sees its cards by UUID
        (`CUDA_VISIBLE_DEVICES=<uuid,...>`, from the start-time card table); under a lease command the cards
        are known only once it leased them, so the late binder opens and checks the sources and names the
        cards then, and the lease tool itself names them for host jobs. A judge job runs on the host; an
        agent job goes through `Isolation.job`, whose descriptors stay open until the backend has started it.
        """
        if job.build_private is not None:
            if BUILD_COMMAND is None:  # a receipted build is the judge's: it registers its command at import
                raise Refused(f"{job.id}: a receipted build needs the judge's build command")
            return Launch(BUILD_COMMAND(self, job, cards), None, ())
        # Every agent job starts with the toolset's environment (from the toolset: PATH, the serve env, ...),
        # sandboxed or plain; its own `env` wins over it. The judge's own jobs keep their own (trusted) env.
        extra = {**job.env, **self._cuda(cards)}
        if job.origin != "judge":
            extra = {**self.cfg.agent_job_env, **extra}
        if job.origin == "judge":  # the judge's own jobs run on the host, never in the agent's sandbox
            return Launch(list(job.argv), ({**os.environ, **extra} if extra else None), ())
        return self.iso.job(
            JobSpec(
                argv=list(job.argv),
                cwd=Path(job.cwd),
                root=Path(job.root or job.cwd),
                cards=cards,
                env=extra,
                readable=[self.dir / job.id, *self.cfg.readable],
                gpu=self.cfg.gpus,
                binds=dict(self.cfg.binds),
                host_written=[*self.cfg.host_written, self.dir],
                run_dir=self.cfg.run_dir,
            )
        )

    def _cuda(self, cards: list[int]) -> dict[str, str]:
        """The CUDA environment naming `cards` (PCI-bus-order ordinals) by UUID; no GPU without `cfg.gpus`."""
        if not self.cfg.gpus:
            return {"CUDA_VISIBLE_DEVICES": ""}
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(self.uuids[c] for c in cards),
        }

    def _launch(self, job: Job, cards: list[int]) -> None:
        jdir = self.dir / job.id
        # Any failure to start fails this job only (a NUL in an agent's argv is a ValueError): the starter
        # thread must never die of one, or no job would start again.
        launch: Launch | None = None
        try:
            if self.check_start is not None:
                self.check_start()
            launch = self._command(job, cards)
            env = None if launch.env is None else dict(launch.env)
            self.handles[job.name] = self.backend.launch(
                list(launch.argv), job.cwd, cards, jdir / "log", job.id, env, pass_fds=launch.pass_fds
            )
        except Exception as e:  # noqa: BLE001 -- any start failure fails this job only
            self._end(job, State.failed, None, f"could not start: {type(e).__name__}: {e}")
            return
        finally:
            for fd in launch.pass_fds if launch is not None else ():
                os.close(fd)  # the backend has passed them on (Popen) or not (a fake)
        job.state, job.cards, job.started, job.reason = State.running, cards, self.clock.now(), ""
        self._save(job)

    def deadline_of(self, origin: str) -> float:
        """The latest end of a job of `origin`: the GPU deadline, plus `judge_grace_s` for the judge's."""
        return self.deadline + (self.cfg.judge_grace_s if origin == "judge" else 0.0)

    def _late(self, expected_s: float, origin: str = "agent") -> str:
        left = self.deadline_of(origin) - self.clock.now()
        what = "the judge's drain window" if origin == "judge" else "the GPU deadline"
        return f"expected {expected_s / 60:.1f} min would pass {what} ({max(left, 0) / 60:.1f} min left)"

    def _wait(self, job: Job, why: str) -> None:
        if job.state is State.queued and job.reason != why:
            job.reason = why
            self._save(job)

    def _end(self, job: Job, state: State, rc: int | None, why: str) -> None:
        job.state, job.rc, job.ended, job.reason = state, rc, self.clock.now(), why
        self._save(job)

    def _save(self, job: Job) -> None:
        p = self.dir / job.id / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(job.model_dump_json(indent=1))
        tmp.replace(p)
