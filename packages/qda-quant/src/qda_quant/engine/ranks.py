"""Start build worker processes and deliver planned jobs."""

from __future__ import annotations

import contextlib
import errno
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..exit import ENVIRONMENT
from ..step import Refused
from .residency import use_expandable_segments

if TYPE_CHECKING:
    from .execute import _Job
    from .planning import Plan

#: The open-file limit the refusal asks for. The job reaches each rank as shared-memory file descriptors, about one a
#: calibration sample per card, held open in this process until the rank has read it: Qwen3-8B W4A16 GPTQ on 4 cards
#: with 512 samples peaked at 2048-2546 open files over two runs (and hung at 1023 under a 1024 limit before the send
#: was checked); 128 samples on 4 cards stayed under 1024. 65536 covers 8 cards at 8192 samples.
MIN_OPEN_FILES = 65536
#: Seconds a rank whose job could not be sent gets to exit on its own (its pipe broke because it died) before
#: the build stops the ranks and refuses.
SEND_GRACE = 2.0


def _ulimit_fix(n: int) -> str:
    return f"run `ulimit -n {n}` in the shell that starts the build (above the hard limit only root can)"


def raise_open_files() -> None:
    """Raise this process's soft open-file limit to its hard limit, before the card processes are spawned.

    The card processes inherit the limit. A build that needs more than the hard limit is not refused here, since its
    need depends on the samples and cards: its send runs out and `_Ranks.run` refuses in two lines.
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        with contextlib.suppress(ValueError, OSError):  # a platform that caps the soft limit below the hard
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class _Ranks:
    """The card processes of a multi-card build, started before the plan exists; each waits for its job.

    `run` hands every rank the job and waits for them (a rank's exception is raised here, as with
    ``torch.multiprocessing.spawn``). `close` tells ranks still waiting that there is no job, so they leave the
    process group and exit; ranks that do not exit in time are terminated. Safe to call more than once.

    ``nccl_nvls`` reaches every rank, which sets ``NCCL_NVLS_ENABLE`` from it before ``init_process_group``.
    The ranks rendezvous through a file in a private temporary directory, not through a TCP port: they are all
    on this node, and a socket probe has a window between choosing a port and the group taking it (and probes an
    address the group does not listen on, if it probes `127.0.0.1`). The directory goes when `close` returns.
    """

    def __init__(
        self,
        devices: Sequence[torch.device],
        fn: Callable[..., None] | None = None,
        *,
        nccl_nvls: bool = False,
    ) -> None:
        import torch.multiprocessing as mp
        from torch.multiprocessing.spawn import start_processes

        raise_open_files()
        ctx = mp.get_context("spawn")
        self.queues = [ctx.SimpleQueue() for _ in devices]
        self.rendezvous = tempfile.TemporaryDirectory(prefix="qda-rdzv-")
        procs = start_processes(
            fn or _rank_main,
            args=(list(devices), self.queues, str(Path(self.rendezvous.name) / "store"), nccl_nvls),
            nprocs=len(devices),
            join=False,
            start_method="spawn",
        )
        assert procs is not None  # join=False always returns the process context
        self.procs = procs
        self.sent = False

    def run(self, job: _Job) -> None:
        """Send `job` to every rank and wait until all have finished it.

        The job is sent from a daemon thread: a rank that died before reading its queue never drains the pipe,
        and a job larger than the pipe buffer would block a send here before the join that raises its error.
        A send that fails to a rank that is still alive (out of file descriptors, most often) would leave that
        rank waiting for ever: the ranks are stopped and the failure is raised here.
        """
        self.sent = True
        failed: list[tuple[int, BaseException]] = []

        def send() -> None:
            for i, q in enumerate(self.queues):
                try:
                    q.put(job)
                except Exception as e:  # noqa: BLE001  (a dead rank's pipe; its error is raised by the join)
                    failed.append((i, e))

        threading.Thread(target=send, name="qda-send-job", daemon=True).start()
        since: float | None = None
        while not self.procs.join(timeout=1.0):
            if not failed:
                continue
            since = since or time.monotonic()
            alive = [(i, e) for i, e in failed if (p := self.procs.processes[i]) is not None and p.is_alive()]
            if alive and time.monotonic() - since >= SEND_GRACE:
                self._stop()
                _drop_unsent_descriptors()
                raise _send_error(*alive[0])

    def _stop(self) -> None:
        for p in self.procs.processes:
            if p is not None and p.is_alive():
                p.terminate()
        for p in self.procs.processes:
            if p is not None:
                p.join()

    def close(self, timeout: float = 120.0) -> None:
        """Stop ranks that never got a job (or are still running after an error); no-op once they exited."""
        if not self.sent:
            self.sent = True
            for q in self.queues:
                with contextlib.suppress(Exception):
                    q.put(None)
        deadline = time.monotonic() + timeout
        with contextlib.suppress(Exception):
            while not self.procs.join(timeout=max(0.0, deadline - time.monotonic())):
                if time.monotonic() >= deadline:
                    break
        self._stop()
        self.rendezvous.cleanup()


def _drop_unsent_descriptors() -> None:
    """Close the descriptors the job's sends left for the stopped ranks to fetch.

    Each shared tensor is passed as a duplicate descriptor that `multiprocessing.resource_sharer` keeps open until
    the receiving process asks for it; the ranks are stopped, so nothing will. Left open, they keep this process at
    its limit, and the build's own cleanup (removing its temporary directories) fails on them. Its `stop` would
    open a connection first, which is what has run out, so the cache is closed directly.
    """
    from multiprocessing import resource_sharer

    sharer: Any = getattr(resource_sharer, "_resource_sharer")  # noqa: B009  (private: no stub names it)
    with sharer._lock:  # noqa: SLF001
        for _, close in sharer._cache.values():  # noqa: SLF001
            with contextlib.suppress(OSError):
                close()
        sharer._cache.clear()  # noqa: SLF001


def _out_of_files(e: BaseException) -> bool:
    """Whether a send failed for want of file descriptors: an EMFILE, or torch's message for one."""
    if isinstance(e, OSError) and e.errno == errno.EMFILE:
        return True
    return isinstance(e, RuntimeError) and "Too many open files" in str(e)


def _send_error(rank: int, e: BaseException) -> Exception:
    """What a failed send to a live rank raises: a refusal when the process ran out of file descriptors."""
    if _out_of_files(e):
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        need = max(MIN_OPEN_FILES, 4 * soft) if soft != resource.RLIM_INFINITY else MIN_OPEN_FILES
        err = Refused(
            f"sending the job to card process {rank} ran out of file descriptors (open-file limit {soft})",
            _ulimit_fix(need),
            code=ENVIRONMENT,
        )
    else:
        err = RuntimeError(f"sending the job to card process {rank} failed: {type(e).__name__}: {e}")
    err.__cause__ = e
    return err


def _run_ranks(
    job: _Job,
    devs: list[torch.device],
    plan: Plan,
    ranks: _Ranks | None,
    *,
    batch_size: int,
    nccl_nvls: bool,
    log: Callable[[str], None],
) -> None:
    """Run the layer work on `len(devs)` card processes, in a temporary directory they share."""
    import tempfile

    from . import calib as _calib
    from .execute import _model_steps, default_attention

    needs = any(step.needs for _, steps in plan.segments for step in steps)
    if needs:
        assert plan.calib_ids is not None  # the plan loads ids when any step needs statistics
        n_batches = len(_calib.batches(plan.calib_ids, batch_size))
    else:
        n_batches = len(devs)
    if needs and n_batches < len(devs):
        raise Refused(
            f"{n_batches} calibration batches for {len(devs)} cards: every card needs one; "
            "lower batch_size, add calibration samples, or use fewer cards"
        )
    with tempfile.TemporaryDirectory(
        prefix="qda-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None
    ) as tmp:
        if any(k == "model" for k, _ in plan.segments):
            # model steps run once, here, before the ranks get the job; ranks load the result,
            # re-apply nothing
            with default_attention():
                job.weights, job.qparams = _model_steps(job, devs[0], Path(tmp), log)
        if ranks is None:
            ranks = _Ranks(devs, nccl_nvls=nccl_nvls)
        ranks.run(job)


def _rank_main(
    rank: int, devices: list[torch.device], queues: list[Any], init_file: str, nccl_nvls: bool
) -> None:
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText  # noqa: F401

    from ..cli import line_buffered_stdout
    from .dist import Comm
    from .execute import _run, default_attention

    line_buffered_stdout()  # a card process's lines reach the job's log as they are printed, as the parent's do
    # NCCL reads this at group init: NVLS binds shared memory that some nodes cannot map, and qda's collectives
    # are point-to-point in rank order, so the multicast path buys nothing
    os.environ["NCCL_NVLS_ENABLE"] = "1" if nccl_nvls else "0"
    dev = devices[rank]
    cuda = dev.type == "cuda"
    if cuda:
        use_expandable_segments()  # this card's process allocates as the parent's does
        torch.cuda.set_device(dev)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(devices),
        device_id=dev if cuda else None,
    )
    job = queues[rank].get()
    if job is None:  # the build stopped before the layer work (a refusal, or another path)
        dist.destroy_process_group()
        return
    # the exporter's process-wide setup (the GGUF kernel, for one) ran in the parent before this rank was
    # spawned; this rank quantizes in its own process, so it prepares here before any weight is loaded
    job.prepare(job.recipe)
    with default_attention():
        _run(job, dev, Comm(rank, len(job.devices), dev), print if rank == 0 else (lambda _: None))
    # only on success: after an exception a peer may be blocked in a collective, and a collective teardown here
    # would hang too; letting the exception propagate exits this process, and spawn's join stops the others
    dist.destroy_process_group()
