"""Time and execute model and layer work in each build process."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..format import Format
from ..recipe import Recipe
from ..solve.quant import ACT_CUTS
from ..step import LayerContext, Refused, Step, keeps_float_stream
from ..target import KernelPlan, Target
from . import calib as _calib
from .layer_cache import LayerCache, LayerCacheSetting, digest, source_digest, tensors_digest
from .planning import Plan
from .stats import DEFAULT_HESSIAN, HessianSetting
from .token_weights import BatchWeights


class Timers:
    """Accumulates seconds per phase; synchronizes the given CUDA devices at phase edges."""

    def __init__(self, devices: Sequence[torch.device] = ()) -> None:
        self.devices = [d for d in devices if d.type == "cuda"]
        self.seconds: dict[str, float] = {}
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()

    def _sync(self) -> None:
        for d in self.devices:
            torch.cuda.synchronize(d)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the block under `name` (added to earlier blocks of the same name)."""
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t

    def add(self, name: str, seconds: float) -> None:
        """Add `seconds` to `name` (thread-safe; per-card busy time from the worker threads)."""
        with self._lock:
            self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def report(self) -> dict:
        """{"total": s, "phases": {...}, "peak_mem_gib": {device: GiB}}."""
        return {
            "total": round(time.perf_counter() - self.t0, 3),
            "phases": {k: round(v, 3) for k, v in self.seconds.items()},
            "peak_mem_gib": {
                str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in self.devices
            },
        }


@dataclass
class BuildRun:
    """The build context handed to model steps (``apply_model(model, graph, run)``)."""

    model: torch.nn.Module
    graph: Any
    recipe: Recipe
    fmt: Format
    target: Target
    plan: Plan
    devices: list[torch.device]
    calib_ids: list[torch.Tensor]
    timers: Timers
    batch_size: int = 1
    token_weights: list[BatchWeights] | None = None
    log: Callable[[str], None] = print
    info: dict = field(default_factory=dict)
    comm: Any = None  # engine.dist.Comm on multi-card builds
    local_batches: list[torch.Tensor] | None = None  # this rank's shard
    hessian: HessianSetting = DEFAULT_HESSIAN
    total_rows: int | None = None
    #: The build's quantization parameters, the one place steps put them and the Writer reads them:
    #: module name -> {"codes", "scale", "zero_point"?, "global_scale"?, "format"}. Layer steps see it as
    #: `LayerContext.qparams`.
    qparams: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The model's tensors a step changed outside the graph (a rotated vision merger), by model name; the
    #: Writer writes them where the source checkpoint is copied verbatim.
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def batches(self) -> list[torch.Tensor]:
        """This rank's calibration batches (all of them on one card), in a fixed order."""
        if self.plan.calib_inputs is not None:
            raise Refused(
                "this step runs calibration forwards on token ids alone; image calibration is not supported; "
                "use a text calibration source for this step"
            )
        if self.local_batches is not None:
            return self.local_batches
        return _calib.batches(self.calib_ids, self.batch_size)


def capture(*args: Any, **kwargs: Any) -> Any:
    """Import calibration execution only when a build captures layer inputs."""
    from .runner import capture as capture_inputs

    return capture_inputs(*args, **kwargs)


def bank_cut(plan: KernelPlan | None) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """The round trip a bank's inputs are cut with for `plan`, or None when its format never read a cut bank.

    Only NVFP4 measures a bank on the kernel's cut: its input scales are one per 16 values, so the statistics
    must see the distribution the kernel will. Every other act dtype's MoE recipes were solved on uncut expert
    inputs (`fp8_e4m3` and `int8` included), so their banks stay exactly as they were before NVFP4.
    """
    if plan is None or plan.act_dtype != "nvfp4":
        return None
    cut = ACT_CUTS["nvfp4"]
    assert cut is not None  # the ACT_CUTS table names one for every act dtype but bf16
    return cut.qdq


@contextlib.contextmanager
def act_inputs(
    cuts: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    layer: torch.nn.Module,
    lg: Any,
    bank_qdq: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[None]:
    """Make each dense linear read the round trip of its input under its unit's own cut, and NVFP4's routed
    experts too.

    `cuts` maps a unit key (a group's first module) to its activation quant-dequant (`solve.quant.ACT_CUTS`):
    a mixed NVFP4/FP8 layer cuts the modules stored in each format the way that format's kernel does, and a
    unit absent from the map (bf16 activations) reads its input as it is.

    With `bank_qdq` (NVFP4's, from `bank_cut`) an unfused bank's per-expert linears are cut at both of a bank's
    inputs, and a fused bank's gate/up input is cut here; the fused down input is formed inside the bank's
    forward, so only `engine.stats.observe_experts` cuts it (for the statistics, not the stream). Without it
    no expert bank is hooked, as on every pre-NVFP4 format: their released weights were solved on the plain
    expert inputs, and cutting them here would move those rows.
    """
    from ..arch import submodule

    def hook(cut: Callable[[torch.Tensor], torch.Tensor]):  # noqa: ANN202
        def cut_input(_m, args, kwargs):  # noqa: ANN001, ANN202
            if args:
                return (cut(args[0]), *args[1:]), kwargs
            key = next(iter(kwargs))
            return args, {**kwargs, key: cut(kwargs[key])}

        return cut_input

    mods = [(m, cuts[g.modules[0]]) for g in lg.groups if g.modules[0] in cuts for m in g.modules]
    if bank_qdq is not None:
        for b in lg.banks:
            if isinstance(b.experts, str):
                mods.append((b.experts, bank_qdq))  # one fused bank module: the call's first argument
            else:
                mods += [(m, bank_qdq) for ins, out in b.experts for m in (*ins, out)]
    hs = [
        submodule(layer, lg, m).register_forward_pre_hook(hook(cut), with_kwargs=True)
        for m, cut in dict.fromkeys(mods)  # no module cut twice with the same round trip (normalised banks)
    ]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def _qparam_names(names: str | tuple) -> list[str]:
    """The qparams keys of a unit: its parameter names, or an unfused bank's per-expert module names."""
    out: list[str] = []
    for n in (names,) if isinstance(names, str) else names:
        out.extend([n] if isinstance(n, str) else [k for k in n if isinstance(k, str)])
    return out


def input_scales(ctx: LayerContext) -> None:
    """Give every NVFP4 entry of the layer its ``input_global_scale``, from the largest input its unit saw.

    The modules of a dense group read one input, so they store one scale (vLLM takes one for the modules it
    fuses). Every expert of a bank reads one input per projection, so all the entries of one bank projection
    store one scale too: the largest over the experts (`input_global_scale` reduces the per-expert ranges).
    vLLM reduces the per-expert scales it reads the same way, to the one it serves.
    """
    from ..solve.nvfp4 import input_global_scale
    from ..solve.units import units_of

    for u in units_of(ctx.layer, ctx.graph):
        entries = [
            ctx.qparams[n]
            for n in _qparam_names(u.names)
            if n in ctx.qparams and Format.parse(ctx.qparams[n]["format"]).act == "nvfp4_block"
        ]
        if not entries:
            continue
        seen = ctx.stats[("act_absmax", u.key)]
        scale = input_global_scale(seen if u.part is None else seen[u.part])
        for qp in entries:
            qp["input_global_scale"] = scale


ATTENTION_BACKENDS = ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "OVERRIDEABLE")


DETERMINISTIC_BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "OVERRIDEABLE")


def attention_backends(plan: Plan) -> tuple[str, ...]:
    """The SDPA backends `plan`'s calibration forwards run under: the deterministic ones for image samples."""
    return DETERMINISTIC_BACKENDS if plan.calib_inputs is not None else ATTENTION_BACKENDS


@contextlib.contextmanager
def default_attention() -> Iterator[None]:
    """Enable exactly torch's default attention backends for a build; restore the caller's after."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([getattr(SDPBackend, b) for b in ATTENTION_BACKENDS]):
        yield


@contextlib.contextmanager
def calibration_attention(plan: Plan) -> Iterator[None]:
    """Pin the attention backends `plan`'s calibration forwards may use (see `attention_backends`)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([getattr(SDPBackend, b) for b in attention_backends(plan)]):
        yield


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {"torch": str(torch.__version__)}
    for p in ("transformers", "compressed-tensors", "qda-quant"):
        with contextlib.suppress(md.PackageNotFoundError):
            out[p] = md.version(p)
    return out


@dataclass
class _Job:
    model: str
    recipe: Recipe
    fmt: Format
    target: Target
    plan: Plan
    devices: list[torch.device]
    out_dir: str
    batch_size: int
    dtype: torch.dtype
    writer: Callable[..., Any]  #: the exporter's writer factory (`output.exporters.Exporter.writer`)
    #: the exporter's process-wide setup (`output.exporters.Exporter.prepare`); every rank calls it, because
    #: each quantizes in its own process and the parent's call does not reach it
    prepare: Callable[[Recipe], str | None]
    hessian: HessianSetting = DEFAULT_HESSIAN
    weights: str | None = None  # multi-card: the model after the model steps (safetensors), else None
    qparams: str | None = None  # multi-card: the model steps' `BuildRun.qparams` (torch.save), else None
    #: multi-card: seconds of each model step, by segment index (they run before the ranks get the job)
    model_seconds: dict[int, float] = field(default_factory=dict)
    exporter: str = ""  #: the exporter's config name, for messages
    #: the exporter's settings that can change a solve (`Exporter.solve_settings`), for the layer cache's key
    solve_settings: dict[str, Any] = field(default_factory=dict)
    layer_cache: LayerCacheSetting | None = None  #: reuse and keep solved layers (one card only)
    layer_context: Callable[..., contextlib.AbstractContextManager] | None = None


def _build_run(
    job: _Job,
    m: torch.nn.Module,
    graph: Any,
    ids: list[torch.Tensor],
    timers: Timers,
    device: torch.device,
    log: Callable[[str], None],
    **fields: Any,
) -> BuildRun:
    return BuildRun(
        m,
        graph,
        job.recipe,
        job.fmt,
        job.target,
        job.plan,
        [device],
        ids,
        timers,
        job.batch_size,
        log=log,
        **fields,
    )


def _needs_verified_graph(plan: Plan) -> bool:
    return any(s.needs_verified_graph for _, steps in plan.segments for s in steps)


def _warn_abandoned_folds(graph: Any, log: Callable[[str], None]) -> None:
    dropped = graph.extra.get("dropped") or ()
    if not dropped:
        return
    for relation in dropped:
        log(
            f"qda: warning: norm fold abandoned: {relation} could not be absorbed into the following "
            "linear group; the checkpoint is numerically unaffected (the trial fold was undone), but this "
            "optimisation opportunity was skipped"
        )
    log(
        "qda: warning: to get the fold, register an architecture adapter through the 'qda.arch_adapters' "
        "entry point (no fork is needed), or open an issue asking us to add it"
    )


def _model_steps(job: _Job, device: torch.device, tmp: Path, log: Callable[[str], None]) -> tuple[str, str]:
    import gc

    from safetensors.torch import save_model

    from ..arch import describe

    m = job.plan.store.load(job, device)
    graph = describe(m, verify=_needs_verified_graph(job.plan), keep=job.plan.keep)
    assert job.plan.calib_ids is not None  # loaded whenever any step needs statistics
    run = _build_run(job, m, graph, job.plan.calib_ids, Timers([device]), device, log)
    for si, (kind, steps) in enumerate(job.plan.segments):
        if kind == "model":
            t = time.perf_counter()
            steps[0].apply_model(m, graph, run)
            job.model_seconds[si] = _since(t, device)
    path, qpath = tmp / "model.safetensors", tmp / "qparams.pt"
    save_model(m, str(path))
    #: read back on the host (map_location="cpu") by rank 0
    torch.save({"qparams": run.qparams, "tensors": run.tensors, "info": run.info}, qpath)
    del m, run
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return str(path), str(qpath)


def _run(job: _Job, device: torch.device, comm: Any, log: Callable[[str], None]) -> None:
    t0 = time.perf_counter()
    plan, recipe, root = job.plan, job.recipe, comm is None or comm.root
    if plan.calib_inputs is not None and job.batch_size != 1:
        raise Refused(
            f"image calibration runs one sample per forward, not batch_size={job.batch_size}; set batch_size: 1"
        )
    timers = Timers([device])
    with timers.phase("load"):
        m = job.plan.store.load(job, device)
    from ..arch import describe

    # not overlapped with capture: describe traces the model's forward, capture swaps its layers. On several
    # cards rank 0 verifies (fp64, CPU) and the others take its graph: same weights, so the same graph
    with timers.phase("describe"):
        verified = _needs_verified_graph(plan)
        verify = lambda: describe(m, verify=verified, keep=plan.keep)  # noqa: E731
        verified_graph = verify() if comm is None else comm.from_root(verify)
    if verified:  # only the fp64 check can abandon a fold; a weightless trace cannot see one
        _warn_abandoned_folds(verified_graph, log)
    ids = plan.calib_ids or []
    # the format's own statistics count too: NVFP4 reads its input ranges with no step asking for them
    needs = frozenset().union(*(s.needs for _, steps in plan.segments for s in steps)) | plan.act_needs()
    all_batches = _calib.batches(ids, job.batch_size) if needs else []
    if comm is not None:
        from .dist import shard

        mine = [all_batches[j] for j in shard(len(all_batches), comm.rank, comm.world)]
    else:
        mine = all_batches
    run = _build_run(
        job,
        m,
        plan.graph,
        ids,
        timers,
        device,
        log,
        local_batches=mine,
        total_rows=sum(b.numel() for b in all_batches),
        comm=comm,
        hessian=job.hessian,
    )
    if job.qparams is not None and root:  # the model steps' output (a head format, a rotated merger)
        saved = torch.load(job.qparams, map_location="cpu", weights_only=True)
        run.qparams, run.tensors = saved["qparams"], saved["tensors"]
        run.info.update(saved["info"])  # a model step's record (a rotated source block)

    writer = job.writer
    last = max(i for i, (k, _) in enumerate(plan.segments) if k == "layers")
    w = None
    stages = []
    cache_enabled = job.layer_cache is not None
    for si, (kind, steps) in enumerate(plan.segments):
        t = time.perf_counter()
        if kind == "model" and comm is not None:  # applied once before the ranks started (`_model_steps`)
            stages.append(_stage(kind, steps, job.model_seconds[si]))
            continue
        if kind == "model":
            run.graph = verified_graph
            with timers.phase(f"model_step:{steps[0].name}"):
                steps[0].apply_model(m, run.graph, run)
        else:
            if si == last and root:
                w = writer(
                    job.out_dir,
                    job.model,
                    verified_graph,
                    job.fmt,
                    job.target,
                    dtype=job.dtype,
                    # only an exporter `check_exporter` admits for overrides takes `formats`
                    **({"formats": job.plan.formats} if job.plan.formats else {}),
                )
            cache = _cache(job, run, si, device) if cache_enabled else None
            cache_enabled = cache is not None
            _layer_pass(
                run,
                steps,
                w if si == last else None,
                verified_graph,
                calibration=bool(needs),
                layer_context=job.layer_context,
                cache=cache,
            )
            if cache is not None:
                run.info.setdefault("layer_cache", []).append({"segment": si, **cache.info()})
        stages.append(_stage(kind, steps, _since(t, device)))
    report = timers.report()
    ranks = comm.gather_objects(report) if comm is not None else [report]
    # every rank's solves, for the offsets record; the collective runs on every rank
    from ..solve import solve as _solve

    solved = _solve.solve_offsets()
    solved = comm.gather_objects(solved) if comm is not None else [solved]
    if not root:
        return
    with timers.phase("finish"):
        from ..config import target_name

        info = {
            "source": job.model,
            **_source_repo(job),
            "format": job.fmt.name,
            "target": target_name(job.target),
            "card_count": len(job.devices),
            "devices": [str(d) for d in job.devices],
            "batch_size": job.batch_size,
            "load_dtype": _load_dtype(job),
            "hessian": {"dtype": job.hessian.dtype, "buffer_rows": job.hessian.buffer_rows},
            "attention_backends": list(attention_backends(plan)),
            **({"format_defaults": plan.format_defaults} if plan.format_defaults else {}),
            **({"steps_skipped": plan.steps_skipped} if plan.steps_skipped else {}),
            # a grid-named clip run over a mixed-format recipe gives each stored format its own clip
            **({"clip_by_format": clips} if (clips := plan.clips_by_format()) else {}),
            "residency": plan.residency.as_dict(),
            "output_bytes": plan.output_bytes,
            "solves": _solve_record([s for part in solved if part for s in part]),
            "stages": stages,
            "versions": _versions(),
            "calibration": plan.calibration or {"filter": "not run (no reference index)"},
            "calib_template": _template_record(plan),
            "data_lint": _lint_record(plan),
            "calib_sha256": _calib.calib_record(plan.calib_ids, plan.calib_inputs),
            **run.info,
        }
        info["timings"] = timers.report()
        if comm is not None:
            info["rank_timings"] = ranks
        info["load_to_saved_s"] = round(time.perf_counter() - t0, 3)
        assert w is not None  # the last layer segment creates the writer
        if hasattr(w, "add_model"):  # embeddings, final norm, head as the model steps left them
            w.add_model(m, run.qparams, changed=any(k == "model" for k, _ in plan.segments))
        if run.tensors:  # modules outside the graph a step changed: the writer has to take them
            if not hasattr(w, "add_source_tensors"):
                names = ", ".join(sorted(run.tensors))
                raise Refused(
                    f"exporter {job.exporter!r}: the build changed tensors outside the graph ({names}) and "
                    "its writer has no add_source_tensors, so the build would write them unchanged; add the "
                    "hook (qda_quant.output.exporters.Exporter.writer)"
                )
            w.add_source_tensors(run.tensors)
        w.finish(m.config, recipe, info)


def _solve_record(solves: list[dict[str, Any]]) -> dict[str, Any]:
    from ..solve.solve import INT32_ELEMENTS

    def size(s: dict[str, Any]) -> int:
        return max(s["max_offset"], s["u_max_offset"])

    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for s in solves:
        key = (s["label"], s["impl"], s["experts"], s["rows"], s["cols"], *s["strides"], *s["u_strides"])
        entries.setdefault(key, s)
    ordered = sorted(entries.values(), key=size, reverse=True)
    return {
        "limit": INT32_ELEMENTS,
        "max_offset": max((s["max_offset"] for s in ordered), default=0),
        "max_factor_offset": max((s["u_max_offset"] for s in ordered), default=0),
        "entries": ordered,
    }


def _load_dtype(job: _Job) -> str:
    return str(job.dtype).removeprefix("torch.")


def _info(job: _Job, stages: list[dict]) -> dict:
    from ..config import target_name

    plan = job.plan
    # a stage that solves nothing (an export alone) records the empty summary
    return {
        "solves": _solve_record([]),
        "source": job.model,
        **_source_repo(job),
        "format": job.fmt.name,
        "target": target_name(job.target),
        "card_count": len(job.devices),
        "devices": [str(d) for d in job.devices],
        "batch_size": job.batch_size,
        "load_dtype": _load_dtype(job),
        "hessian": {"dtype": job.hessian.dtype, "buffer_rows": job.hessian.buffer_rows},
        "attention_backends": list(attention_backends(plan)),
        "residency": plan.residency.as_dict(),
        "output_bytes": plan.output_bytes,
        "stages": stages,
        "versions": _versions(),
        "calibration": plan.calibration or {"filter": "not run (no reference index)"},
        "calib_template": _template_record(plan),
        "data_lint": _lint_record(plan),
        "calib_sha256": _calib.calib_record(plan.calib_ids, plan.calib_inputs),
    }


def _source_repo(job: _Job) -> dict[str, str]:
    """build.json's ``source_repo``: the original's hub repo, from the recipe or the model directory.

    A build reads a local directory, and a prepared run downloads the model as ``.../model``, so the path
    names no model. The recipe's ``meta.source_repo`` is an explicit override; else the directory's own
    record (`qda_quant.config.SOURCE_RECORD`, written by the download) names the repo, so every recipe built
    from that directory gets it, including an agent's own. The serve name reads it
    (`qda_quant.output.serve_command.served_name`).
    """
    from ..config import SOURCE_RECORD

    repo = job.recipe.meta.get("source_repo")
    if not (isinstance(repo, str) and repo):
        try:
            record = json.loads((Path(job.model) / SOURCE_RECORD).read_text())
        except (OSError, ValueError):
            record = None
        repo = record.get("repo") if isinstance(record, dict) else None
    return {"source_repo": repo} if isinstance(repo, str) and repo else {}


def _template_record(plan: Plan) -> dict[str, Any]:
    """build.json's ``calib_template``: the chat-template check's row counts, or why it did not run."""
    return plan.calib_template or {"check": "not run (no text calibration rows read)"}


def _lint_record(plan: Plan) -> dict[str, Any]:
    """build.json's ``data_lint``: per data file read, its row classes and defect counts, or that none was read."""
    from ..calib.template import lint_receipt

    return lint_receipt(plan.data_lint)


def _export_alone(job: _Job, log: Callable[[str], None]) -> None:
    log("export: the exporter writes this recipe from the source; no weights loaded, no layer pass")
    t0 = time.perf_counter()
    timers = Timers()
    with timers.phase("finish"):
        w = job.writer(job.out_dir, job.model, job.plan.graph, job.fmt, job.target, dtype=job.dtype)
        steps = [s for _, part in job.plan.segments for s in part]
        info = _info(job, [_stage("export", steps, 0.0)])
        info["load_to_saved_s"] = 0.0  # `build` stamps it (`_restamp`)
        w.finish(None, job.recipe, info)
    path = Path(job.out_dir) / "build.json"  # the writer wrote it before these two could be known
    info = json.loads(path.read_text())
    info["stages"][0]["seconds"] = round(time.perf_counter() - t0, 3)
    info["timings"] = timers.report()
    path.write_text(json.dumps(info, indent=1))


def _since(t: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return round(time.perf_counter() - t, 3)


def _stage(kind: str, steps: list[Step], seconds: float) -> dict:
    return {"kind": kind, "steps": [s.name for s in steps], "seconds": seconds}


def _cache(job: _Job, run: BuildRun, segment: int, device: torch.device) -> LayerCache | None:
    """The layer cache of layer pass `segment`, keyed by everything a layer solve reads besides the formats.

    The source's full content, the calibration rows themselves, the recipe's steps, the body format and its
    kernels, the build's arithmetic (Hessian, batch size, expert chunks), this package's sources, the torch and
    CUDA builds and the card, and the exporter's solve settings. Per-unit formats key each layer (`LayerCache`).
    Inputs not covered by the key disable caching for the build and leave the normal solve intact.
    """
    from ..arch.adapters import ADAPTERS
    from ..intake.fingerprint import model_fingerprint
    from ..solve.grid import GRIDS
    from ..step import QuantStep

    assert job.layer_cache is not None
    plan = job.plan
    if run.token_weights is not None:
        run.log("layer_cache: disabled because realized token weights are not covered by the cache key")
        return None
    plugins: list[Any] = list(job.recipe.steps)
    if run.graph.adapter:
        plugins.append(ADAPTERS[run.graph.adapter])
    formats = {job.fmt.name: job.fmt, **{f.name: f for f in plan.formats.values()}}
    kernels = {(job.fmt.name, "dense"): plan.plan_dense, **plan.plans}
    if plan.plan_moe is not None:
        kernels[(job.fmt.name, "moe")] = plan.plan_moe
    for step in job.recipe.steps:
        if isinstance(step, QuantStep):
            if (clip := step.clip_search()) is not None:
                plugins.append(clip)
            plugins.extend(
                GRIDS[step.grid_for(formats[name], kernel)] for (name, _), kernel in kernels.items()
            )
    for plugin in plugins:
        cls = type(plugin)
        if cls.__module__.split(".")[0] != "qda_quant":
            run.log(f"layer_cache: disabled because {cls.__module__}.{cls.__qualname__} is outside qda_quant")
            return None
    recipe = {k: v for k, v in job.recipe.to_dict(full=True).items() if k not in ("calib", "meta")}
    key = digest(
        {
            "source": model_fingerprint(job.model),
            "calibration": tensors_digest(run.calib_ids),
            "inputs": None
            if plan.calib_inputs is None
            else [tensors_digest(list(x.values())) for x in plan.calib_inputs],
            "recipe": recipe,
            "segment": segment,
            "format": job.fmt.name,
            # an override format's kernels follow from it and the target; the layer keys carry the formats
            "target": repr(job.target),
            "kernels": [repr(plan.plan_dense), repr(plan.plan_moe)],
            "keep": sorted(plan.keep),
            "hessian": [job.hessian.dtype, job.hessian.buffer_rows],
            "batch_size": job.batch_size,
            "expert_chunks": plan.residency.expert_chunks,
            "dtype": str(job.dtype),
            "code": source_digest(),
            "versions": [torch.__version__, torch.version.cuda, str(device), _card(device)],
            "exporter": job.solve_settings,
        }
    )
    return LayerCache(job.layer_cache, key, {k: f.name for k, f in plan.formats.items()})


def _card(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def _layer_pass(
    run: BuildRun,
    steps: list[Step],
    writer: Any,
    graph: Any,
    *,
    calibration: bool = True,
    layer_context: Callable[..., contextlib.AbstractContextManager] | None = None,
    cache: LayerCache | None = None,
) -> None:
    from ..solve.units import FMT, down_key
    from .runner import Pass

    m, plan, timers, dev = run.model, run.plan, run.timers, run.devices[0]

    routers = {bank.router for lg in graph.layers for bank in lg.banks}

    def make_ctx(layer: torch.nn.Module, lg: Any, stats: dict) -> LayerContext:
        keys = {grp.modules[0] for grp in lg.groups} | {bank.router for bank in lg.banks}
        for key in plan.formats.keys() & keys:
            f = plan.formats[key]
            kind = "moe" if key in routers else "dense"
            stats[(FMT, key)] = (f, plan.plans[(f.name, kind)])
        for bank in lg.banks:  # a bank's down projections a GGUF stores apart from its gate/up
            if (f := plan.formats.get(down_key(bank) or "")) is not None:
                stats[(FMT, down_key(bank))] = (f, plan.plans[(f.name, "moe")])
        return LayerContext(
            layer,
            lg,
            graph,
            run.fmt,
            plan.plan_dense,
            plan.plan_moe if lg.banks else None,
            stats,
            run.qparams,
            device=dev,
        )

    # both forwards carry the calibration samples (images included), so they run under the plan's backends
    with calibration_attention(plan):
        slots = _capture(run, steps) if calibration else []
        run.graph = graph
        # A unit takes the cut of the format it is stored in: in a mixed build the FP8 modules see the
        # per-token cut and the NVFP4 modules the block cut, so each one calibrates against what its own
        # kernel computes. An override with no dense plan of its own (one that names only banks) keeps the
        # body's plan rather than falling out of the calibration: a unit is uncut only when its own plan
        # says bf16 activations.
        dense = {}
        for lg in graph.layers:
            for grp in lg.groups:
                key = grp.modules[0]
                f = plan.formats.get(key)
                dense[key] = (
                    plan.plan_dense if f is None else plan.plans.get((f.name, "dense"), plan.plan_dense)
                )
        cuts = {k: c.qdq for k, p in dense.items() if (c := ACT_CUTS[p.act_dtype]) is not None}
        # A bank is cut by the plan of the format it is stored in: with an NVFP4 body beside an FP8 attention
        # override that is the body's block cut. `resolve_formats` holds every bank to one format, so one
        # plan names them all.
        bank_f = next((plan.formats[r] for r in sorted(routers) if r in plan.formats), None)
        bank_qdq = bank_cut(
            plan.plan_moe if bank_f is None else plan.plans.get((bank_f.name, "moe"), plan.plan_moe)
        )
        ranged = bool(plan.act_needs())
        Pass(
            m,
            graph,
            steps,
            dev,
            make_ctx,
            timers,
            on_layer=None if writer is None else writer.add_layer,
            layer_context=layer_context,
            store=plan.store,
            act_cuts=cuts,
            bank_cut=bank_qdq,
            extra_needs=plan.act_needs(),
            after_steps=input_scales if ranged else None,
            comm=run.comm,
            total_rows=run.total_rows,
            token_weights=run.token_weights,
            hessian=run.hessian,
            expert_chunks=plan.residency.expert_chunks,
            cache=cache,
            log=run.log,
        ).run(slots)


def _capture(run: BuildRun, steps: list[Step]) -> list:
    m, graph, plan, timers, dev = run.model, run.graph, run.plan, run.timers, run.devices[0]
    with timers.phase("capture"), plan.store.outer(m, graph, dev):
        keep_float = any(keeps_float_stream(step.needs) for step in steps)
        inputs = plan.calib_inputs
        return capture(
            m,
            graph,
            run.batches() if inputs is None else _calib.batches(run.calib_ids),
            dev,
            host_acts=plan.residency.activations == "host",
            keep_float=keep_float,
            inputs=inputs,
        )
