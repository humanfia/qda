"""Build and atomically publish a planned checkpoint."""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ..format import Format
from ..intake.fingerprint import release_digest_buffer
from ..output.exporters import Exporter, check_exporter, parse_exporter, record_exporter
from ..output.record import BUILD_RECORDS
from ..recipe import Recipe
from ..step import Refused
from ..target import Target
from .execute import BuildRun, _export_alone, _Job, _run, default_attention
from .layer_cache import LayerCacheSetting
from .planning import (
    Plan,
    _refuse_early,
    check_output,
    compile_plan,
    local_source,
    one_card_devices,
    resolve_devices,
    shardable_needs,
)
from .ranks import _Ranks, _run_ranks
from .residency import use_expandable_segments
from .stats import HessianSetting, hessian_setting

_STAGING_NAME_BYTES = 100


@contextlib.contextmanager
def _staging(out: Path, log: Callable[[str], None]) -> Iterator[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        out.lstat()  # a name the filesystem cannot hold fails here, not at the rename after all the work
    name = os.fsencode(out.name)[:_STAGING_NAME_BYTES].decode(errors="ignore")  # whole characters only
    staging = out.parent / f".{name}.partial-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    try:
        yield staging
    except BaseException:
        log(f"build stopped: its partial output stays in {staging}")
        raise
    try:
        staging.rename(out)
    except OSError as e:
        raise Refused(f"cannot publish to {out} ({e}); the finished checkpoint stays in {staging}") from e


def _checkpoint_bytes(out: Path) -> int:
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and p.name not in BUILD_RECORDS)


def _restamp(path: Path, seconds: float, stages: list[dict] | None = None) -> None:
    if path.is_file():
        d = json.loads(path.read_text())
        d["load_to_saved_s"] = seconds
        d["output_bytes"] = _checkpoint_bytes(path.parent)
        if stages:
            d["stages"] = [*stages, *d.get("stages", [])]
        path.write_text(json.dumps(d, indent=1))


__all__ = ["BuildRun", "build", "default_attention"]


@dataclass(frozen=True)
class _Options:
    target: Target | str | None
    devices: Sequence[int | str | torch.device] | None
    keep_intermediate: bool
    batch_size: int
    residency: str | None
    nccl_nvls: bool
    dtype: torch.dtype
    hessian: HessianSetting
    keep: frozenset[str] | None
    exporter: Exporter
    format_overrides: tuple[tuple[str, str], ...] = ()
    layer_cache: LayerCacheSetting | None = None

    def build_kw(self) -> dict[str, Any]:
        """The options of the float stage's build (`engine.packed.staged`), re-sent from here."""
        return {
            "devices": self.devices,
            "keep_intermediate": self.keep_intermediate,
            "batch_size": self.batch_size,
            "residency": self.residency,
            "nccl_nvls": self.nccl_nvls,
            "dtype": self.dtype,
            "hessian": self.hessian,
            "ignore": None if self.keep is None else sorted(self.keep),
            "exporter": self.exporter,
            "format_overrides": self.format_overrides,
        }


def _capabilities(devs: Sequence[torch.device]) -> list[tuple[str, tuple[int, int]]]:
    if not torch.cuda.is_available():
        return []
    return [(str(d), torch.cuda.get_device_capability(d)) for d in devs if d.type == "cuda"]


def _target_for(exporter: Exporter, target: Target | str | None, devs: Sequence[torch.device]) -> Target:
    own = getattr(exporter, "target_name", None)
    if own is None:
        return Target.on_cards(target, _capabilities(devs))
    if target is not None and Target.parse(target).arch != own:
        raise Refused(
            f"target {target!r}: the {exporter.name!r} output is served by its own runtime, which decides the "
            f"target ({own!r}); remove `target` from the config"
        )
    return Target.parse(own)


def build(
    model: str | Path,
    recipe: Recipe | str,
    *,
    format: Format | str | None = None,  # noqa: A002
    target: Target | str | None = None,
    out_dir: str | Path,
    devices: Sequence[int | str | torch.device] | None = None,
    keep_intermediate: bool = False,
    batch_size: int = 1,
    residency: str | None = None,
    nccl_nvls: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    hessian: HessianSetting | dict | str | None = None,
    ignore: Sequence[str] | None = None,
    format_overrides: Sequence[tuple[str, str]] = (),
    exporter: Exporter | str | dict | None = None,
    layer_cache: LayerCacheSetting | dict | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build a quantized checkpoint; see `qda_quant.build`."""
    t0 = time.perf_counter()
    use_expandable_segments()  # before the first CUDA allocation: the card peak `qda plan` prints assumes it
    out = check_output(out_dir)
    hessian = hessian_setting(hessian)
    if isinstance(recipe, str):
        path = Path(recipe)
        recipe = (
            Recipe.from_yaml(path.read_text(), base=path.resolve().parent)
            if path.is_file()
            else Recipe.from_yaml(recipe)
        )
    # before the model is read (a repo id may download) or any weight is loaded: a config that puts card
    # steps on CPU would turn a minutes-long build into an hours-long one, and this job holds cards
    _refuse_cpu_steps(devices, recipe)
    # the target, exporter and cards are settled before the model is read (a repo id may download), so a
    # missing build prerequisite refuses before that work rather than after it
    exp = parse_exporter(exporter)
    devs = resolve_devices(devices)
    # the build's own guards (a faulty card, a driver too old for torch's CUDA), the list `qda doctor` walks too
    from ..checks import refuse_build

    refuse_build([d.index or 0 for d in devs if d.type == "cuda"])
    _refuse_unbuildable_triton(devs, log)
    tgt = _target_for(exp, target, devs)
    if note := tgt.on_other_cards(_capabilities(devs)):
        log(f"plan: {note}")
    if isinstance(model, Path):
        model = model.resolve()  # keep a Path's local meaning when the run record converts it to a string
    given, model = str(model), local_source(model)  # a repo id: its snapshot, before reading its config
    # a step that runs on one card only takes the first card when `devices` is unset, and refuses several named
    if len(devs) > 1:
        devs = one_card_devices(
            devices, devs, model, recipe, None if format is None else Format.parse(format), tgt, log
        )
    # no forward pass, no sharding: layer steps that ask for no statistics run in this process on one card even
    # when more are given (`shardable_needs`). Every rank of a multi-card run reads every layer from the source
    # and receives it again by broadcast, while the layer work itself stays on rank 0, so a second card only
    # adds time. The plan below is then made for that one card, which is also the card its residency sizes.
    if len(devs) > 1:
        fmts = [None if format is None else Format.parse(format)]
        fmts += [Format.parse(f) for _, f in format_overrides]
        needs = shardable_needs(model, recipe, fmts[0], tgt)
        if needs is not None and not needs:
            # a format's own plan can still read the calibration set (NVFP4's input ranges): the one-card
            # decision stays, because a second card only shares what the layer steps' statistics need
            reads = any(f is not None and f.act == "nvfp4_block" for f in fmts)
            log(
                f"plan: no step needs statistics{' (the format reads the calibration set)' if reads else ''}; "
                f"one process on {devs[0]} ({len(devs)} cards given)"
            )
            devs = devs[:1]
    options = _Options(
        target,
        devices,
        keep_intermediate,
        batch_size,
        residency,
        nccl_nvls,
        dtype,
        hessian,
        None if ignore is None else frozenset(ignore),
        exp,
        tuple(format_overrides),
        None if layer_cache is None else LayerCacheSetting.model_validate(layer_cache),
    )
    # several cards: the card processes start now, so their interpreter start, imports and NCCL setup overlap the
    # plan below; they wait for the job and do no other GPU work until it comes (or exit if the build stops first)
    ranks = (
        _Ranks(devs, nccl_nvls=options.nccl_nvls)
        if len(devs) > 1 and all(d.type == "cuda" for d in devs)
        else None
    )
    try:
        return _build(given, model, recipe, format, tgt, devs, ranks, out_dir, out, options, log, t0)
    finally:
        if ranks is not None:
            ranks.close()
        release_digest_buffer()  # the pinned copy buffer is the size of the largest weight; the build is over


def _refuse_unbuildable_triton(devs: Sequence[torch.device], log: Callable[[str], None]) -> None:
    if not any(d.type == "cuda" for d in devs):
        return
    from ..checks import triton_build_refusal, triton_build_verdict

    check = triton_build_verdict(log)
    # this checks a fresh compile: a cached native helper may still load without one, while other
    # uncached launchers can still need the headers and compiler
    cached = (
        "; a cached native helper may still load without a fresh compile, "
        "and uncached launchers can still need the headers and compiler"
    )
    if (refused := triton_build_refusal(check, cached)) is not None:
        raise refused
    if check.ok and not check.skipped:
        log(f"preflight: Triton's CUDA driver builds with {check.detail}")


def _refuse_fast_gdn(devs: Sequence[torch.device], graph: Any) -> None:
    """A CUDA build refuses a GDN model whose fast kernels would fall back to transformers' torch path.

    The reference conv and delta-rule kernels are correct but much slower than fla's Triton kernels over a
    whole calibration; a card must take the fast path, and the message names what is missing. CPU and meta
    builds keep the reference path.
    """
    if not any(d.type == "cuda" for d in devs):
        return
    from ..checks import gdn_fallback

    if why := gdn_fallback(graph):
        raise Refused(why)


def _refuse_cpu_steps(devices: Sequence[int | str | torch.device] | None, recipe: Recipe | str) -> None:
    """A build on a job that holds cards refuses a config that puts its card steps on CPU.

    The rule and its message are `qda_quant.checks`' (`cpu_step_refusal`), shared with the agent's launch
    check of shipped start configs and with `BuildConfig.build` (which names the config file). It reads the
    resolved `devices` and `steps` only: no model config and no weight is opened. A job without cards,
    `devices` unset (the job's own cards), a card or mixed `devices`, and a CPU-only step list (RTN alone)
    all build as before. A refused build writes no `build.json` and mints no build receipt, as the
    fast-kernel refusal below does.
    """
    from ..checks import cpu_step_refusal

    steps = recipe.steps if isinstance(recipe, Recipe) else ()
    if why := cpu_step_refusal(devices, steps):
        raise Refused(why)


def _plan(
    model: str,
    recipe: Recipe,
    fmt: Format | None,
    tgt: Target,
    devs: list[torch.device],
    options: _Options,
    log: Callable[[str], None] | None = None,
) -> Plan:
    return compile_plan(
        model,
        recipe,
        fmt,
        tgt,
        devs,
        options.dtype,
        options.residency,
        options.keep,
        options.batch_size,
        overrides=options.format_overrides,
        log=log,
        exporter=options.exporter,  # the output's own stored types bind their units before the solve
    )


def _build(
    given: str,
    model: str,
    recipe: Recipe,
    format: Format | str | None,  # noqa: A002
    tgt: Target,
    devs: list[torch.device],
    ranks: _Ranks | None,
    out_dir: str | Path,
    out: Path,
    options: _Options,
    log: Callable[[str], None],
    t0: float,
) -> Path:
    plan = _plan(model, recipe, None if format is None else Format.parse(format), tgt, devs, options, log)
    # before any weight is read or a layer forward runs: a GDN model on a card must have its fast kernels
    _refuse_fast_gdn(devs, plan.graph)
    options = replace(options, keep=plan.keep)
    # `dtype` is a config key (bf16, fp32); a Python-API dtype no config names cannot be recorded, so the
    # provenance names it as not recorded (`qda_quant.config.dtype_name`)
    from ..config import dtype_name

    not_recorded = [] if dtype_name(options.dtype) else ["dtype"]
    _refuse_early(plan, recipe, devs, options.batch_size)
    if options.layer_cache is not None and (len(devs) > 1 or plan.staged or plan.generate):
        raise Refused(
            "layer_cache: a cached build runs on one card, with no checkpoint steps or generate stage; "
            f"this one has {len(devs)} cards{', checkpoint steps' if plan.staged else ''}"
            f"{', a generate stage' if plan.generate else ''}"
        )
    check_exporter(options.exporter, plan, recipe)
    record_exporter(options.exporter, plan, recipe)
    note = options.exporter.prepare(recipe)
    for peak in plan.stage_peaks:  # each checkpoint stage's floor, before stage 0 runs
        log(f"stage {peak.step}: {peak}")
    if note:
        log(note)
    # the generate stage runs first, after every plan check; then the WHOLE recipe is planned and checked again on
    # the generated rows (checkpoint steps included), before any layer work or other stage
    generated: list[dict] = []
    alone = not plan.staged and options.exporter.writes_alone(plan, recipe)
    if ranks is not None and (plan.generate or plan.staged or alone):
        ranks.close()
        ranks = None
    with _staging(out, log) as staging:
        if plan.generate:
            generated, plan = _generate(plan, recipe, model, tgt, devs, options, log)
        if plan.staged:  # checkpoint steps, or a packed source: every stage checked above, now run in stages
            from .packed import staged

            staged(
                given,
                plan,
                recipe,
                target=options.target,
                out_dir=out_dir,
                staging=staging,
                log=log,
                not_recorded=not_recorded,
                generated=generated,
                **options.build_kw(),
            )
            seconds = round(time.perf_counter() - t0, 3)
        else:
            fmt = plan.fmt
            log(f"plan: {len(plan.graph.layers)} layers, {[k for k, _ in plan.segments]}, {plan.residency}")
            stages = generated
            job = _Job(
                model,
                recipe,
                fmt,
                tgt,
                plan,
                devs,
                str(staging),
                options.batch_size,
                options.dtype,
                options.exporter.writer,
                options.exporter.prepare,
                options.hessian,
                exporter=options.exporter.name,
                layer_context=options.exporter.layer_context,
                solve_settings={} if options.layer_cache is None else options.exporter.solve_settings(),
                layer_cache=options.layer_cache,
            )
            if alone:
                _export_alone(job, log)
            elif len(devs) == 1:
                with default_attention():
                    _run(job, devs[0], None, log)
            else:
                _run_ranks(
                    job,
                    devs,
                    plan,
                    ranks,
                    batch_size=options.batch_size,
                    nccl_nvls=options.nccl_nvls,
                    log=log,
                )
            from ..arch.adapters import used
            from ..config import BuildConfig

            BuildConfig.of_build(
                given,
                recipe,
                fmt=fmt,
                target=tgt,
                devices=options.devices,
                hessian=options.hessian,
                batch_size=options.batch_size,
                residency=options.residency,
                nccl_nvls=options.nccl_nvls,
                out=out_dir,
                dtype=options.dtype,
                ignore=sorted(plan.keep),
                exporter=options.exporter,
                format_overrides=options.format_overrides,
                format_defaults=plan.format_defaults,
                layer_cache=options.layer_cache,
            ).write_resolved(
                staging,
                source_snapshot=None if model == given else model,
                adapters=used(plan.graph),
                store=plan.residency.weights,
                cards=len(devs),
                not_recorded=not_recorded,
                format_defaults=plan.format_defaults,
            )
            seconds = round(time.perf_counter() - t0, 3)
            _restamp(staging / "build.json", seconds, stages)
    log(f"built {out} in {seconds} s")
    return out


def _generate(
    plan: Plan,
    recipe: Recipe,
    model: str,
    tgt: Target,
    devs: list[torch.device],
    options: _Options,
    log: Callable[[str], None],
) -> tuple[list[dict], Plan]:
    log(f"generate: {plan.generate}")
    t = time.perf_counter()
    cache = recipe.calib.generate(model)
    generated = [
        {
            "kind": "generate",
            "steps": [recipe.calib.name],
            "seconds": round(time.perf_counter() - t, 3),
            **cache,
        }
    ]
    plan = _plan(model, recipe, plan.fmt, tgt, devs, options, log)
    log(f"plan after generation: {plan.residency}")
    return generated, plan
