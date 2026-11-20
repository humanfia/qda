"""Today's isolation, behind the one interface: a run keeps the behaviour its config had before the levels.

`make(cfg)` returns this for every level until that level's own module lands: a config with
`sandbox.name: bwrap` and `scheduler.job_sandbox: bwrap` still gets the agent wall and bwrap jobs, `none` on
both sides still gets plain processes, and one bwrap side still gets exactly what it did. The level modules
replace one branch at a time.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qda_agent.isolation import bwrap

from .. import devices, paths
from ..api import Handle, JobSpec, KillDomain, Launch, Level, job_sandbox

if TYPE_CHECKING:
    from collections.abc import Callable

    from hmz.flows import Permission

    from ...config import RunConfig
    from ...sched.config import SchedConfig


@dataclass(eq=False)
class _Proc:
    """A job this domain started, as the legacy process-group stop needs it."""

    popen: subprocess.Popen[bytes]
    grace_s: float
    pid: int  # the started process's pid (the `Handle` the interface promises)


class _GroupKill(KillDomain):
    """Today's kill domain: one process group per job, TERM then KILL (the reaper replaces it in B2)."""

    def spawn(self, launch: Launch, *, cwd: Path, log: Path) -> _Proc:
        command = list(launch.argv)
        env = None if launch.env is None else dict(launch.env)
        with Path(log).open("ab") as out:
            popen = subprocess.Popen(  # noqa: S603 (the level built argv)
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=tuple(launch.pass_fds),
            )
        return _Proc(popen, 0.0, popen.pid)

    def poll(self, handle: Handle) -> int | None:
        assert isinstance(handle, _Proc)
        return handle.popen.poll()

    def stop(self, handle: Handle, grace_s: float) -> None:
        assert isinstance(handle, _Proc)
        popen = handle.popen
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(popen.pid, signal.SIGTERM)
        end = time.monotonic() + grace_s
        while time.monotonic() < end:
            popen.poll()  # reap the leader, which would count as alive while a zombie
            try:
                os.killpg(popen.pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:  # a process of the group runs as another user: still there
                pass
            time.sleep(0.02)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(popen.pid, signal.SIGKILL)
        popen.wait()


class LegacyIsolation:
    """The run's isolation as the config described it before the levels existed (see this module)."""

    def __init__(
        self,
        *,
        level: str,
        cfg: RunConfig | None = None,
        sched: SchedConfig | None = None,
        on_skip: Callable[[str, str], None] | None = None,
    ) -> None:
        self.level: Level = level  # type: ignore[assignment]  # checked by `api.make`
        self.cfg = cfg
        self.sched = sched
        self.kill: KillDomain = _GroupKill()
        # The old module global: the flow logs skipped binds while its run is under way (it goes in B9).
        bwrap.on_skip = on_skip

    @classmethod
    def for_config(
        cls, cfg: RunConfig, *, on_skip: Callable[[str, str], None] | None = None
    ) -> LegacyIsolation:
        """Today's behaviour for a run config."""
        return cls(level=cfg.isolation.level, cfg=cfg, on_skip=on_skip)

    @classmethod
    def for_sched(cls, cfg: SchedConfig) -> LegacyIsolation:
        """Today's behaviour for a scheduler built on its own settings (a test, the judge CLI)."""
        return cls(level="strict", sched=cfg)

    def _settings(self) -> tuple[str, dict[str, str], tuple[Path, ...], bool]:
        """`(job_sandbox, agent_job_env, hide, gpus)`, read live: a caller may still adjust its config."""
        if self.cfg is not None:
            sched = self.cfg.scheduler
            return (
                self.cfg.job_sandbox(),
                self.cfg.agent_job_env(),
                tuple(getattr(self.cfg.sandbox, "hide", ())),
                getattr(sched, "gpus", True),
            )
        assert self.sched is not None
        return (
            job_sandbox(self.level, self.sched.job_sandbox),
            self.sched.agent_job_env,
            tuple(self.sched.hide),
            self.sched.gpus,
        )

    def preflight(self, cfg: RunConfig) -> None:
        """Refuse a run whose sandbox cannot hold, before its clock starts."""
        from qda_agent import isolation as levels

        levels.check(cfg)
        scheduler = cfg.scheduler
        if cfg.job_sandbox() != "bwrap" or getattr(scheduler, "fake_gpus", False):
            return
        readable, binds = cfg.read_paths()
        overlay = cfg.run.overlay
        bwrap.preflight(
            readable,
            binds,
            writable=[cfg.workdir],
            run_dir=Path(os.path.realpath(cfg.run_dir)),
            host_written=[
                *([] if overlay is None else [Path(os.path.realpath(overlay))]),
                cfg.run_dir / "jobs",
            ],
            hide=getattr(cfg.sandbox, "hide", ()),
        )

    def session(self, cfg: RunConfig) -> Any:
        """Hold the agent's wall for the run (the private CLI config when no wall is up)."""
        from qda_agent.isolation import gpu_wall

        return gpu_wall(cfg)

    def agent_permission(self) -> Permission:
        """What an agent turn may touch, as today's sandbox setting says."""
        from qda_agent.isolation import permission

        assert self.cfg is not None  # only a run's flow asks for a turn's permission
        return permission(self.cfg.sandbox)

    def job(self, spec: JobSpec) -> Launch:
        """The argv, environment and descriptors a job is started with at today's level."""
        build = (
            self.cfg is None and self.sched is None
        )  # a trusted build: the level alone, spec carries the rest
        if build:
            jobs, agent_job_env, hide = job_sandbox(self.level), {}, ()
        else:
            jobs, agent_job_env, hide, _gpus = self._settings()
        if jobs != "bwrap":
            env = {**os.environ, **dict(spec.env)} if spec.env else None
            return Launch(list(spec.argv), env, ())
        root = Path(spec.root)
        inner = (
            dict(spec.env)  # a build's settings carry its whole environment already
            if build
            else {**paths.job_env(root), **agent_job_env, **dict(spec.env)}
        )
        nodes = devices.device_nodes(spec.cards) if spec.gpu else []
        # The receipted builds live under the run dir. A spec with no run dir (a scheduler built on its own
        # settings) names its jobs directory in `host_written`; the run dir is that directory's parent.
        published_from = spec.run_dir
        if published_from is None and spec.host_written:
            published_from = Path(spec.host_written[-1]).parent
        argv, fds = bwrap.pinned_bwrap_argv(
            list(spec.argv),
            str(spec.cwd),
            writable=[root, *spec.extra_writable],
            readable=list(spec.readable),
            run_dir=spec.run_dir,
            host_written=list(spec.host_written),
            devices=nodes,
            env=inner,
            binds=dict(spec.binds),
            pid="auto" if spec.gpu else "private",  # no GPU: nothing of the driver's to give a /proc
            hide=[*hide, *spec.hide],
            # A trusted build sees nothing of the host's root beyond its inputs; a job sees the host root.
            readonly=() if build else paths.published_entries(published_from, root),
        )
        if spec.empty_root:
            argv[1:1] = ["--tmpfs", "/"]
        try:
            argv, private = bwrap.private_args_kept(argv)
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise
        return Launch(argv, None, (*fds, *private))

    def size_out(self, root: Path, rel: str) -> int:
        """Bytes the tree `root/rel` holds, sized in the same confinement as the copy (`snapshot_tree_bytes`)."""
        from ..snapshot import snapshot_tree_bytes

        with tempfile.TemporaryDirectory(prefix="qda-size-") as scratch:
            return snapshot_tree_bytes(root, rel, Path(scratch))

    def copy_out(self, root: Path, rel: str, into: Path) -> Path:
        """Copy `root/rel` out of the agent's tree, inside bwrap (see `isolation.snapshot.copy_out`)."""
        from ..snapshot import copy_out

        return copy_out(root, rel, into)

    def verify_trusted(self, paths: Any) -> None:
        """Today's levels prevent the write; a digest check is `none`'s to add."""
        del paths

    def provenance(self) -> dict[str, object]:
        """The level, the pid form and the cards' UUIDs, for `resolved.yaml` and `qda doctor`.

        The pid form is what a bwrap run's sandboxes got (None elsewhere): the probe that decides it is
        this level's record to make, not the config's. A run at `none`, or with a sandbox that is not
        bwrap, records None without probing - as the config's own record did.
        """
        sandbox = getattr(self.cfg, "sandbox", None) if self.cfg is not None else None
        walled = self.level != "none" and getattr(sandbox, "name", None) == "bwrap"
        form = bwrap.pid_form(getattr(sandbox, "pid", "auto")) if walled else None  # not bwrap: no probe
        out: dict[str, object] = {"level": self.level, "pid_form": form, "uuids": {}}
        cards = getattr(self.cfg, "cards", None)
        if cards:
            try:
                out["uuids"] = dict(zip(cards, devices.card_uuids(cards), strict=True))
            except (ValueError, OSError):
                pass  # no driver (or a card it does not list): the record says no UUID is known
        return out
