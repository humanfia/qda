"""Load source metadata and validate model, recipe and calibration compatibility."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import gcd, isqrt
from typing import TYPE_CHECKING, Any, cast

import torch

from ..arch import DEFAULT_KEEP, MIN_QUANT_ROWS, ModelGraph
from ..format import Format
from ..recipe import Recipe
from ..solve.grid import DEFAULT_GRID, GRIDS, Grid
from ..step import QuantStep, Refused, Step, probe_refusal, statistics_of
from ..target import KernelPlan, Target, shape_limit
from . import calib as _calib
from .checkpoint_size import _OwnModelFormat
from .format_bindings import _check_moe_bindings, _default_record, expert_defaults, resolve_formats


def _check_solve_offsets(meta: torch.nn.Module, graph: ModelGraph, steps: Sequence[Step]) -> None:
    from ..solve.solve import INT32_ELEMENTS
    from ..solve.units import units_of

    quant = [s for s in steps if isinstance(s, QuantStep)]
    if not quant:
        return
    for lg in graph.layers:
        for u in units_of(meta.get_submodule(lg.module), lg):
            rows, cols = u.shape()
            if not max((s.solver_workspace(rows, cols) for s in quant), default=0):
                continue  # no step solves this unit
            if rows * cols > INT32_ELEMENTS:
                what = "a dense group" if u.part is None else "an expert's matrix"
                raise Refused(
                    f"gptq {u.key}: {what} is {rows} x {cols} = {rows * cols} elements, past the "
                    f"{INT32_ELEMENTS} one solve stays inside; a dense group is never split, and an "
                    "expert's matrix is the smallest chunk"
                )
            if cols * cols > INT32_ELEMENTS:
                raise Refused(
                    f"gptq {u.key}: the factor's rows are {cols} x {cols} = {cols * cols} elements, past "
                    f"the {INT32_ELEMENTS} one solve stays inside"
                )


def _check_head_steps(graph: ModelGraph, steps: Sequence[Step], keep: frozenset[str]) -> None:
    from ..arch.derive import ignore_hits

    head = graph.lm_head
    if not head:
        return
    hit = next((p for p in sorted(keep) if ignore_hits((p,), head)), None)
    if hit is None:
        return
    for s in steps:
        if isinstance(s, QuantStep) and isinstance(s, _OwnModelFormat) and s.kind == "model":
            raise Refused(
                f"{s.name}: stores {head} in {s.format}, but `ignore` keeps it bf16 ({hit!r}); the head is "
                "stored one way: drop the head step or the bf16 entry"
            )


def _check_checkpoint_stage(
    graph: ModelGraph,
    meta: torch.nn.Module,
    fmt: Format,
    formats: dict[str, Format],
    steps: Sequence[Step],
    later: Sequence[Step],
) -> None:
    """Refuse a checkpoint step whose stage would store FP8 *experts*, which it cannot read back.

    A checkpoint step reads the stage through `engine.packed.Packed`: dense FP8 W8A8 modules read like the
    NVFP4 and integer ones (an FP8 ``lm_head`` too). An FP8 expert bank does not -- no serving row plans FP8
    experts on any card -- so the stage could be neither served nor checked. The recipe names that group
    before a weight loads (an FP8 body or ``format_overrides`` rule), so a plan or a build refuses here,
    before any layer is quantized, naming the step and what it would not read.
    """
    from ..solve.units import stored_format, units_of
    from .packed import _targets_label

    if fmt.weight_type != "fp8" and not any(f.weight_type == "fp8" for f in formats.values()):
        return  # no FP8 group reaches the stage
    groups: dict[str, list[str]] = {}
    for lg in graph.layers:
        for u in units_of(meta.get_submodule(lg.module), lg):
            if u.part is None:
                continue  # a dense module: an FP8 one is read back
            f = stored_format(formats, u, fmt)
            if f.weight_type != "fp8":
                continue
            names = (u.names,) if isinstance(u.names, str) else tuple(n for ns in u.names for n in ns)
            groups.setdefault(f.name, []).extend(names)
    if not groups:
        return
    what = [
        f"{len(names)} experts in {name} ({_targets_label(list(dict.fromkeys(names)))})"
        for name, names in sorted(groups.items())
    ]
    raise Refused(
        f"{later[0].name}: the stage it reads stores {' and '.join(what)}; a checkpoint step reads FP8 W8A8 "
        "dense modules, and no card serves FP8 experts to read them back from"
    )


def check_order(steps: Sequence[Step]) -> None:
    """Refuse a model step after a layer step unless it keeps every weight the layer step quantized."""
    first = next((s for s in steps if s.kind == "layer"), None)
    if first is None:
        return
    for s in steps[steps.index(first) + 1 :]:
        if s.kind == "model" and not s.preserves_quantized:
            raise Refused(
                f"{s.name} changes weights {first.name} already quantized; put it before the layer steps"
            )


_RERUN_STATS = frozenset({"layer_io", "layer_io_float", "outputs"})


def check_rerun_passes(steps: Sequence[Step]) -> None:
    """Refuse a step that reads the layer's calibration output in a pass after a model step split the passes."""
    model: Step | None = None  # the model step that would separate an earlier pass from this one
    passed = False  # a layer pass has begun
    for s in steps:
        if s.kind == "model":
            model = s
        elif s.kind == "layer":
            if passed and model is not None and set(s.needs) & _RERUN_STATS:
                raise Refused(
                    f"{s.name} re-runs layers after {model.name} split them into passes; its target would be "
                    f"the already-quantized layer's output, not the float layer's: "
                    f"put {model.name} after the layer steps"
                )
            passed = True


def check_token_weight_readers(steps: Sequence[Step], graph: ModelGraph) -> None:
    """Refuse a step that sets token weights when no later step reads a statistic they weight."""
    from .token_weights import WEIGHTED_STATISTICS

    weighted = WEIGHTED_STATISTICS - (set() if any(lg.banks for lg in graph.layers) else {"expert_hessian"})
    for i, s in enumerate(steps):
        if getattr(s, "weights_tokens", False) and not any(
            set(later.needs) & weighted
            for later in steps[i + 1 :]
            if later.kind == "layer" and getattr(later, "reads_token_weights", True)
        ):
            raise Refused(
                f"{s.name}: no later step reads a token-weighted statistic "
                f"({', '.join(sorted(WEIGHTED_STATISTICS))}), so its weights would "
                "change nothing; put a Hessian step (e.g. gptq) after it, or remove it"
            )


@dataclass(frozen=True)
class QuantWidth:
    """One module a build quantizes, as a plan-time group-size check sees it.

    `expert` modules have their shape checked for divisibility only: a fused bank's served shape and layout
    are the build's to read, and its own binding check stays authoritative. `bind` is False for a module whose
    kernel the build does not bind through `Target.binds` (the head and embedding steps; an expert's `in`/`down`).
    """

    name: str
    rows: int
    cols: int
    expert: bool = False
    bind: bool = True


@dataclass(frozen=True)
class WidthProblem:
    """The first module `fmt` cannot store, and why (a short reason for the plan's one line, a full sentence)."""

    name: str
    cols: int
    reason: str
    detail: str


def _indivisible(
    fmt: Format, widths: Iterable[tuple[str, int]], grids: Sequence[Grid]
) -> Iterator[WidthProblem]:
    """Each (name, width) that is not a whole number of `fmt`'s groups on every one of `grids` (`Grid.divides`).

    The one divisibility rule and refusal sentence: the build refuses with it, and `check_widths` judges by it.
    """
    g = fmt.group_size
    for name, cols in widths:
        if not all(grid.divides(fmt, cols) for grid in grids):
            yield WidthProblem(
                name,
                cols,
                f"is not a multiple of {g}",
                f"{name}: {cols} input columns at group size {g} ({fmt.name}) is not a whole number of groups",
            )


def check_widths(fmt: Format, modules: Sequence[QuantWidth], target: Target) -> WidthProblem | None:
    """Why `fmt` cannot store `modules` on `target`, or None: one check, for the plan and the build.

    The reasons are the build's: group size that divides no width (`check_group_sizes`), and a shape no kernel
    binds under the checkpoint's serve settings (`format_bindings`). Reading a model's widths is the caller's
    job; this function touches no weights and no GPU.
    """
    from ..output.export import merge_serve

    g = fmt.group_size
    if g > 0:
        widths = ((m.name, m.cols) for m in modules)
        if (problem := next(_indivisible(fmt, widths, [GRIDS[DEFAULT_GRID]]), None)) is not None:
            return problem
    moe = any(m.expert for m in modules)
    kinds = ("dense", "moe") if moe else ("dense",)
    plans = {}
    for kind in kinds:
        try:
            plans[kind] = target.plan(fmt, kind)
        except Refused as e:
            first = modules[0] if modules else QuantWidth(fmt.name, 0, g)
            return WidthProblem(first.name, first.cols, f"no {kind} kernel plan ({e})", f"{fmt.name}: {e}")
    bound = [m for m in modules if m.bind and not m.expert]
    if bound:
        serve = merge_serve({f"{fmt.name}:dense": plans["dense"]}, target.vllm)
        for m in bound:
            try:
                target.binds(fmt, (m.rows, m.cols), serve)
            except Refused as e:
                return WidthProblem(
                    m.name,
                    m.cols,
                    f"has no kernel at group size {g}",
                    f"{m.name}: {m.cols} input columns at group size {g} ({fmt.name}) is served by no "
                    f"kernel: {e}",
                )
    return None


def working_group_sizes(fmt: Format, modules: Sequence[QuantWidth], target: Target) -> list[int]:
    """The group sizes `fmt`'s family can store `modules` at on `target`, ascending (`-1`: per channel).

    The candidates are the divisors of the widths' gcd plus per-channel that `check_widths` accepts, so this
    also answers whether a group binds a kernel on the target (`Target.binds`), not just whether it divides.
    """
    widths = [m.cols for m in modules]
    if not widths:
        return []
    span = gcd(*widths)
    candidates = {-1, span} | {d for i in range(2, isqrt(span) + 1) if span % i == 0 for d in (i, span // i)}
    return [
        g for g in sorted(candidates) if check_widths(replace(fmt, group_size=g), modules, target) is None
    ]


def check_group_sizes(
    model: torch.nn.Module,
    graph: ModelGraph,
    fmt: Format,
    target: Target,
    steps: Sequence[Step],
    formats: dict[str, Format] | None = None,
) -> None:
    """Refuse a format whose group size does not divide a width the recipe will quantize."""
    from ..solve.units import stored_format, units_of

    # (format, grids, named widths, the same modules for `working_group_sizes`, target) per group of weights
    checks: list[tuple[Format, list, list[tuple[str, int]], list[QuantWidth], Target]] = []
    grids = [GRIDS[s.grid or DEFAULT_GRID] for s in steps if s.kind == "layer" and isinstance(s, QuantStep)]
    if grids:
        by_format: dict[tuple[str, str], tuple[list[tuple[str, int]], list[QuantWidth]]] = {}
        for lg in graph.layers:
            layer = model.get_submodule(lg.module)
            for u in units_of(layer, lg):
                f = stored_format(formats, u, fmt)
                named, units = by_format.setdefault((f.name, u.kind), ([], []))
                if u.part is None:  # a dense group: every module keeps its own served shape
                    for name, p in zip(u.names, u.params, strict=True):
                        named.append((name, int(p.shape[1])))
                        units.append(QuantWidth(name, int(p.shape[0]), int(p.shape[1])))
                    continue
                names = (u.names,) if isinstance(u.names, str) else tuple(n for pe in u.names for n in pe)
                named += [(name, int(u.shape()[1])) for name in names]
                # divisibility covers the bank; its fused layout and kernel binding stay the build's to check
                units.append(
                    QuantWidth(names[0], int(u.shape()[0]), int(u.shape()[1]), expert=True, bind=False)
                )
        checks += [
            (Format.parse(name), grids, named, units, target)
            for (name, kind), (named, units) in sorted(by_format.items())
        ]
    for s in steps:
        if not (isinstance(s, QuantStep) and isinstance(s, _OwnModelFormat) and s.kind == "model"):
            continue
        modules = [graph.lm_head] + ([graph.embed] if s.embed else [])
        named, units = [], []
        for name in modules:
            w = cast(torch.nn.Linear | torch.nn.Embedding, model.get_submodule(name)).weight
            named.append((name, int(w.shape[-1])))
            units.append(QuantWidth(name, int(w.shape[0]), int(w.shape[-1]), bind=False))
        checks.append((Format.parse(s.format), [GRIDS[s.grid or DEFAULT_GRID]], named, units, target))
    for f, f_grids, named, units, f_target in checks:
        bad = list(_indivisible(f, named, f_grids))
        if not bad:
            continue
        sizes = working_group_sizes(f, units, f_target)
        worked = ", ".join("-1 (per channel)" if g == -1 else str(g) for g in sizes) or "none on this target"
        raise Refused(
            f"{bad[0].detail}; {len(bad) - 1} other module(s) have the same problem; group sizes that work: {worked}"
        )


def _runtime_units(graph: ModelGraph, modules: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """`modules` split into the fused linears vLLM serves, with every undeclared module kept together.

    A `LinearGroup` is every linear sharing one input, which is not always one runtime linear: an adapter
    declares the engine's fusions in ``graph.extra["runtime_names"]``, one `FusedRuntime` per runtime
    linear -- several shards for a fused module, or one shard for a projection the engine serves unfused
    (MLA's ``q_proj`` and ``kv_a_proj_with_mqa`` when ``q_lora_rank`` is unset). Modules no declaration
    names stay together, and a group with no declaration is one unit, so q/k/v and gate/up are still
    checked whole, as the single linears vLLM builds them. A unit must be exactly one runtime linear:
    splitting a fused module would refuse a valid group whose parts each miss the divisor while their sum
    meets it.
    """
    from ..arch.adapters import FusedRuntime
    from ..arch.derive_groups import ignore_hits

    declared = graph.extra.get("runtime_names", {}).get("vllm", {})
    units: list[tuple[str, ...]] = []
    claimed: set[str] = set()
    for value in declared.values():
        if not isinstance(value, FusedRuntime):
            continue
        shards = tuple(name for name in modules if ignore_hits(value.shards, name))
        if not shards:
            continue  # this runtime linear is built from no module of this group
        units.append(shards)
        claimed.update(shards)
    rest = tuple(name for name in modules if name not in claimed)
    if rest:
        units.append(rest)
    return tuple(units)


#: The largest share of the parameters a build quantizes that it may leave bf16 because no kernel of the
#: row's format serves their shape. A handful of tiny projections is a rounding error; anything material is
#: a format the model should not use.
UNSERVABLE_SHARE = 0.01

#: The sentence TASK.md states beside the engine facts: the plan keeps a shape no kernel serves at 16 bits.
UNSERVABLE_KEPT_RULE = (
    "A module whose shape the row's format has no kernel for stays bf16; `qda plan` names it and its share "
    "of parameters."
)


@dataclass(frozen=True)
class Unservable:
    """Modules a format cannot serve at their shape, their parameter share, and the plan's one line."""

    modules: tuple[str, ...]
    share: float
    detail: str


def shape_unservable(fmt: Format, target: Target, rows: int, cols: int) -> str | None:
    """Why a served linear of `rows` x `cols` has no kernel in `fmt` on `target`; None when it has one.

    The rule is the plan's shape check's (`shape_limit` on the kernel the target plans for the format), so a
    module the plan would refuse is exactly a module this function reports.
    """
    return shape_limit(target.plan(fmt, "dense").kernel, fmt, (rows, cols))


def _unservable_detail(fmt: Format, names: Sequence[str], share: float) -> str:
    shown = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
    return (
        f"kept bf16 ({fmt.name} has no kernel for {len(names)} module shape(s), {100 * share:.2f}% of the "
        f"model's quantized parameters): {shown}"
    )


def _unservable(names: list[str], kept: int, total: int, fmt: Format) -> Unservable | None:
    if not names:
        return None
    share = kept / total if total else 0.0
    return Unservable(tuple(names), share, _unservable_detail(fmt, names, share))


def unservable_widths(widths: Sequence[QuantWidth], fmt: Format, target: Target) -> Unservable | None:
    """The modules a build would quantize whose shape `fmt` has no kernel for, or None.

    `prepare` asks this of a model's widths before any step runs, so a tiny projection that no kernel serves
    does not refuse the whole plan; the plan asks `unservable_units` and the two agree through
    `shape_unservable`. A width the group size does not divide is not this rule's (lowering the group is the
    existing fix), and a module whose `bind` is False is not checked by a kernel.
    """
    names: list[str] = []
    kept = total = 0
    for m in widths:
        total += m.rows * m.cols
        if not m.bind or m.expert or m.rows < MIN_QUANT_ROWS:
            continue  # `derive_groups` never quantizes a Linear this narrow; it keeps it bf16
        if fmt.group_size > 0 and m.cols % fmt.group_size:
            continue  # the group-size rule's problem, not a missing kernel
        if shape_unservable(fmt, target, m.rows, m.cols) is not None:
            names.append(m.name)
            kept += m.rows * m.cols
    return _unservable(names, kept, total, fmt)


def unservable_units(
    meta: torch.nn.Module, graph: ModelGraph, fmt: Format, target: Target
) -> Unservable | None:
    """The same for a planned model: each served (fused) linear of every group, as `check_dense_shapes` sees it.

    The shape is the runtime linear's (q/k/v fused, `_runtime_units`), so a module that is fine fused is not
    kept; a module that is not is collected with its whole unit, and the share counts its parameters once.
    """
    names: list[str] = []
    kept = total = 0
    for lg in graph.layers:
        for grp in lg.groups:
            weights = {name: cast(torch.nn.Linear, meta.get_submodule(name)).weight for name in grp.modules}
            total += sum(int(w.numel()) for w in weights.values())
            for unit in _runtime_units(graph, grp.modules):
                rows = sum(int(weights[name].shape[0]) for name in unit)
                if rows < MIN_QUANT_ROWS:
                    continue
                cols = int(weights[unit[0]].shape[1])
                if fmt.group_size > 0 and cols % fmt.group_size:
                    continue
                if shape_unservable(fmt, target, rows, cols) is not None:
                    names.extend(name for name in unit if name not in names)
                    kept += sum(int(weights[name].numel()) for name in unit)
    return _unservable(names, kept, total, fmt)


def kept_unservable(kept: Unservable | None, fmt: Format) -> tuple[str, ...]:
    """The modules a plan keeps bf16 for; an empty tuple when their share is material.

    One owner for the threshold: `prepare` and the plan both call this, so the format choice and the build
    agree about when a model is not quantizable in the row's format at all. Above `UNSERVABLE_SHARE` the
    modules stay in the plan, whose own shape check refuses the format with its established line.
    """
    if kept is None or kept.share > UNSERVABLE_SHARE:
        return ()
    return kept.modules


def check_dense_shapes(
    meta: torch.nn.Module,
    graph: ModelGraph,
    fmt: Format,
    steps: Sequence[Step],
    plan_dense: KernelPlan,
    plans: dict[tuple[str, str], KernelPlan],
    formats: dict[str, Format] | None = None,
    packed: Packed | None = None,
) -> None:
    """Refuse a dense unit whose served fused linear the kernel planned for it cannot take.

    vLLM serves a `LinearGroup` as fused linears: q/k/v become one QKV projection and gate/up one gate/up
    projection, so `CutlassW4A8LinearKernel.can_implement` sees the fused rows, and the checkpoint's
    individual projections are not the shape to check -- a 64-row `k_proj` is fine beside a 128-row
    `q_proj` when the fused 256-row projection is. A group can also be wider than one runtime linear
    (MLA's `q_proj` and `kv_a_proj_with_mqa` are separate linears when `q_lora_rank` is unset), so
    `_runtime_units` splits it by the adapter's declared vLLM runtime linears and checks each; a group
    with nothing declared, like a single-module group (DeepSeek's `kv_a_proj_with_mqa`), is checked whole.
    `check_group_sizes` covers the input columns a grid stores; a kernel can also limit the output rows,
    and the plan must refuse that before a weight loads. The rules are vLLM 0.33's own (`shape_limit`); a
    format whose dense kernel has no rule there passes. `formats` gives each unit its own format
    (`format_overrides`); the rest use `fmt`. A packed source is checked on the groups it stored whole,
    with the target its record names.
    """
    if packed is None and not any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps):
        return  # no layer step stores a dense unit in a format
    for lg in graph.layers:
        for grp in lg.groups:
            if packed is not None:
                if any(name not in packed.stems for name in grp.modules):
                    continue  # not stored whole in a format the kernel serves
                f = packed.format_of(grp.modules[0])
                plan = packed.target.plan(f, "dense")
            else:
                f = (formats or {}).get(grp.modules[0], fmt)
                plan = plans.get((f.name, "dense"), plan_dense)
            weights = {name: cast(torch.nn.Linear, meta.get_submodule(name)).weight for name in grp.modules}
            for unit in _runtime_units(graph, grp.modules):
                rows = sum(int(weights[name].shape[0]) for name in unit)
                cols = int(weights[unit[0]].shape[1])
                if why := shape_limit(plan.kernel, f, (rows, cols)):
                    label = unit[0] if len(unit) == 1 else f"the fused linear of {', '.join(unit)}"
                    raise Refused(
                        f"{label}: {plan.kernel} cannot serve {f.name}: {why}; "
                        "use a format whose dense kernel takes these dimensions (e.g. w4a16)"
                    )


#: Model types that are a pure state-space stack, with no attention layer. vLLM 0.33 builds each mixer's
#: ``in_proj`` as a ``MergedColumnParallelLinear``; its quantized weight loader receives that module where it
#: expects a parameter and raises ``AttributeError`` (``no attribute 'data'``), so a quantized checkpoint of
#: one of these models cannot be served. The target table only knows kernels and calls its linears servable,
#: so the plan must refuse the combination, beside its shape check.
PURE_MAMBA = frozenset({"mamba", "mamba2"})


def check_mamba_served(graph: ModelGraph, quantized: bool, target: Target) -> None:
    """Refuse a quantized pure Mamba body, which vLLM 0.33 cannot load.

    Args:
        graph: The model's graph; its `model_type` says whether the stack is pure state-space.
        quantized: Whether the checkpoint stores any layer in a quantized format (a layer step, or a packed
            source). A bf16 build of the same model is servable and passes.
        target: The serving target. `llama_cpp` names no vLLM model class, and its own exporter refuses a
            pure Mamba source by name before this.
    """
    if graph.model_type not in PURE_MAMBA or not quantized or target.arch == "llama_cpp":
        return
    raise Refused(
        f"{graph.model_type}: vLLM 0.33 cannot load a quantized pure Mamba checkpoint — the mixer's in_proj "
        "is a MergedColumnParallelLinear, whose weight loader has no quantized path (AttributeError: "
        "'MergedColumnParallelLinear' object has no attribute 'data') — so serve the bf16 original instead, "
        "or pick a model with attention layers"
    )


def check_calib_vocab(ids: list[torch.Tensor], label: str, vocab_size: int) -> None:
    """Refuse calibration ids the model's embedding does not have (a second guard after the fingerprint)."""
    top = max(int(r.max()) for r in ids if r.numel())
    low = min(int(r.min()) for r in ids if r.numel())
    if top >= vocab_size or low < 0:
        raise Refused(
            f"calibration {label}: token ids span [{low}, {top}], the model's vocab is [0, {vocab_size})"
        )


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .packed import Packed


def _skeleton(source: str, dtype: torch.dtype, target: Target, *, images: bool = False) -> PreTrainedModel:
    from ..intake.loading import load_model

    try:
        with torch.device("meta"):
            return cast(
                "PreTrainedModel",
                load_model(source, weights=False, dtype=dtype, target=target, images=images),
            )
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001 - the config's class cannot build a weightless model; refuse it
        raise probe_refusal(source, "constructing its weightless model", e) from e


def _source_format(
    source: str, fmt: Format | None, target: Target
) -> tuple[Packed | None, Format, KernelPlan]:
    from .packed import Packed, is_packed

    if not is_packed(source):
        if fmt is None:
            raise Refused("a build from a float model needs a format")
        return None, fmt, target.plan(fmt, "dense")
    packed = Packed(source)
    if fmt is not None and fmt != packed.fmt:
        raise Refused(f"format {fmt.name}: the checkpoint {source} is {packed.fmt.name}")
    return packed, packed.fmt, target.plan(packed.fmt, "dense")


def _steps(
    source: str, recipe: Recipe, fmt: Format, plan_dense: KernelPlan, packed: Packed | None
) -> tuple[list[Step], list[Step], Any]:
    from ..intake.loading import load_config
    from .packed import check_stage_order

    # the source's own config.json, not the meta skeleton's: for a multimodal checkpoint (Qwen3.5 / 3.8) the
    # skeleton is the text model, whose config has no `architectures`
    config = load_config(source)
    recipe.steps = [s.resolve(fmt, plan_dense, config, source) for s in recipe.steps]
    # every step names its grid now; a top-level one lets `--set grid` on the record still reach them
    recipe.grid = recipe.grid or DEFAULT_GRID
    check_stage_order(recipe.steps, packed is not None)
    later = [s for s in recipe.steps if s.kind == "checkpoint"]
    steps = [s for s in recipe.steps if s.kind != "checkpoint"]
    return steps, later, config


def _graph(
    source: str,
    dtype: torch.dtype,
    target: Target,
    keep: frozenset[str] | None,
    packed: Packed | None,
    fmt: Format,
    plan_dense: KernelPlan,
    overrides: Sequence[tuple[str, str]],
    recipe: Recipe,
    steps: Sequence[Step],
    needs: frozenset[str],
    devices: list[torch.device],
    exporter: Any = None,
) -> tuple[
    PreTrainedModel,
    ModelGraph,
    KernelPlan | None,
    dict[str, Format],
    dict[tuple[str, str], KernelPlan],
    dict[str, str],
    frozenset[str],
    frozenset[str],
    dict[str, str],
    dict[str, list[str]],
]:
    from ..arch import describe
    from ..arch.adapters import ADAPTERS, check_served
    from ..intake.source_names import names_under, unloaded

    #: the full image-text class is needed only when an image sample's pixels reach a calibration forward; a
    #: data-free build with an image source stays on the causal-LM class, as its class did before
    images = bool(needs) and getattr(recipe.calib, "images_source", False)
    if images:
        # known from the source and the steps, so refused before a pending generation spends GPU work
        if len(devices) > 1:
            raise Refused(f"calibration {recipe.calib.label}: image calibration builds on 1 card only")
        own = [s.name for s in steps if getattr(s, "requires_single_card_calibration", False)]
        if own:
            raise Refused(
                f"calibration {recipe.calib.label}: {own} run their own calibration forwards on token ids, "
                "without the images; use a text calibration source with them"
            )
    meta = _skeleton(source, dtype, target, images=images)
    if keep is None:
        names = tuple(
            name
            for _, adapter in ADAPTERS.items()
            if adapter.matches(meta) and (defaults := getattr(adapter, "default_ignore", None)) is not None
            for name in defaults(meta)
        )
        linears = {name for name, module in meta.named_modules() if isinstance(module, torch.nn.Linear)}
        if any(name not in linears for name in names):
            raise Refused(f"adapter default_ignore must name exact linear modules: {names}")
        keep = DEFAULT_KEEP | frozenset(names)
    graph = describe(meta, verify=False, keep=keep)
    if packed is None and any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps):
        # A shape no kernel of the row's format serves stays bf16, as the released checkpoints do;
        # the same rule refuses a material share. The graph is described again with those modules kept.
        found = unservable_units(meta, graph, fmt, target)
        servable = kept_unservable(found, fmt)
        if servable:
            keep = keep | frozenset(servable)
            graph = describe(meta, verify=False, keep=keep)
            graph.extra["kept_unservable"] = servable
            graph.extra["kept_unservable_line"] = found.detail if found is not None else ""
    #: source weights the class drops (an MTP head, a text build's vision tower); the writer copies them
    #: verbatim, so a step that changes the residual basis must see them before planning
    graph.extra["unloaded"] = () if packed is not None else unloaded(source, meta)
    stream = graph.extra.get("stream") or {}
    roots: set[str] = {*(stream.get("writers") or {}), *(stream.get("blocks") or {})}
    # Native parameters and codec parts may lack .weight; root presence must account for all source names.
    try:
        graph.extra["source_roots"] = tuple(
            root for root in sorted(roots) if packed is None and names_under(source, root)
        )
    except FileNotFoundError:
        graph.extra["source_roots"] = ()  # Config-only checkpoints supply no native tensor names.
    check_served(graph, fmt)
    plan_moe = (
        target.plan(fmt, "moe", decomposed=any(b.custom_router for lg in graph.layers for b in lg.banks))
        if any(lg.banks for lg in graph.layers)
        else None
    )
    if overrides and packed is not None:
        raise Refused(
            "format_overrides cannot change a packed source's stored formats; "
            "apply overrides when quantizing the float source"
        )
    # the output's own stored types may write solved codes in one format only: bind those units before the
    # solve, so the codes exist in the format the file writes (a pattern the recipe states keeps its value)
    stored_rules, unmet = (
        exporter.stored_format_rules(graph) if exporter is not None and packed is None else ([], {})
    )
    stated = {pattern for pattern, _fmt in overrides}
    stored_formats = {r["pattern"]: r["format"] for r in stored_rules if r["pattern"] not in stated}
    if stored_formats:
        overrides = (*overrides, *stored_formats.items())
    defaults = expert_defaults(graph, fmt)
    formats, alt_plans, applied = (
        resolve_formats(meta, graph, overrides, fmt, target, plan_dense, plan_moe, defaults)
        if overrides or defaults
        else ({}, {}, {})
    )
    return (
        meta,
        graph,
        plan_moe,
        formats,
        alt_plans,
        _default_record(graph, applied),
        frozenset(applied),
        keep,
        stored_formats,
        _derived_units(graph, formats, stored_formats),
        dict(unmet),
    )


def _derived_units(
    graph: ModelGraph, formats: Mapping[str, Format], stored_formats: Mapping[str, str]
) -> frozenset[str]:
    """The units bound only by a format the output's stored types required, not by a recipe rule.

    A plain round-to-nearest build whose *only* bindings are these still writes the same bytes without a
    layer pass (`Exporter.writes_alone`): the mix's fit of the template is that same solve.
    """
    from ..arch.derive import ignore_hits

    hit = set()
    for lg in graph.layers:
        for grp in lg.groups:
            if any(ignore_hits((pattern,), module) for pattern in stored_formats for module in grp.modules):
                hit.add(grp.modules[0])
    return frozenset(hit & formats.keys())


def single_card_steps(steps: Sequence[Step]) -> list[tuple[str, str]]:
    """The layer and model steps a build runs on one card only, as (name, why), in recipe order.

    `planning.one_card_devices` reads this before the cards are chosen; `_check_layer_steps` and
    `_check_multicard` refuse the same steps on several cards for a caller that names its cards itself.
    """
    out, layer_before = [], False
    for s in steps:
        if "cross_hessian" in s.needs:
            out.append((s.name, "pairs float-input calibration"))
        elif getattr(s, "requires_single_card_calibration", False):
            out.append((s.name, "computes calibration-row weights"))
        elif s.kind == "model" and layer_before:
            out.append((s.name, "is a model step after a layer step, which runs between layer passes"))
        elif _reruns_layers(s):
            out.append((s.name, "re-runs layers on the calibration set (each rank would see only its shard)"))
        layer_before = layer_before or s.kind == "layer"
    return out


def _reruns_layers(s: Step) -> bool:
    """The step reads outputs or layer_io it does not gather across cards, or a single-card statistic."""
    runs_layers = "outputs" in s.needs or ("layer_io" in s.needs and not s.multi_card_layer_io)
    return runs_layers or any(stat.single_card for stat in statistics_of(s.needs).values())


def one_card_fix(devices: list[torch.device]) -> str:
    """The `fix:` of a refused multi-card build of a one-card step: keep the first of the named cards."""
    return f"set devices: [{card_name(devices[0])}]"


def card_name(d: torch.device) -> str:
    """How `devices` names this card: its ``CUDA_VISIBLE_DEVICES`` entry when one is set, else its index."""
    if d.type != "cuda":
        return str(d)
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    i = d.index or 0
    return vis[i] if i < len(vis) else str(i)


def _check_layer_steps(
    steps: Sequence[Step],
    meta: PreTrainedModel,
    graph: ModelGraph,
    fmt: Format,
    target: Target,
    plan_dense: KernelPlan,
    plan_moe: KernelPlan | None,
    devices: list[torch.device],
    formats: dict[str, Format],
    alt_plans: dict[tuple[str, str], KernelPlan],
    packed: Packed | None,
) -> None:
    from ..steps._model import check_float_banks, check_stream_steps

    check_order(steps)
    check_float_banks(graph, steps)
    check_rerun_passes(steps)
    check_token_weight_readers(steps, graph)
    if len(devices) > 1:
        # a row-sharing step prices each rank's own calibration batches, so every rank must fold the same
        # weights it prices: nothing may have changed them with rank 0's statistics alone
        seq = [s for s in steps if s.kind == "layer"]
        first = next((n for n, s in enumerate(seq) if getattr(s, "shares_rows", False)), None)
        if first not in (None, 0) and not getattr(seq[first], "starts_from_root", False):
            raise Refused(
                f"step {seq[first].name!r} shares its calibration rows across cards, so it must be the first "
                f"layer step; {seq[0].name!r} runs before it and changes weights from rank 0's statistics, so "
                "the ranks would price different weights (put it first, start the fit from rank 0's parameters, "
                "or build on one card)"
            )
    for s in steps:
        s.check(graph, fmt, plan_dense)
        if isinstance(s, QuantStep):
            s.check_grid(fmt, plan_dense)
            if plan_moe is not None and s.kind == "layer":
                s.check_grid(fmt, plan_moe)
        if len(devices) > 1 and "cross_hessian" in s.needs:
            raise Refused(
                f"step {s.name!r} paired float-input calibration currently requires one card",
                one_card_fix(devices),
            )
        if len(devices) > 1 and getattr(s, "requires_single_card_calibration", False):
            raise Refused(
                f"step {s.name!r} computes calibration-row weights and currently requires one card",
                one_card_fix(devices),
            )
        if plan_moe is not None and s.kind == "layer":
            s.check(graph, fmt, plan_moe)
    for name, plan in alt_plans.items():
        f = Format.parse(name[0])
        for s in steps:
            if s.kind != "layer":
                continue
            # an override unit runs the step with the clip its format gets (`QuantStep.for_format`); the
            # body format's check above kept the named clip, so a body it cannot serve is still refused
            t = s.for_format(f, plan) if isinstance(s, QuantStep) and f.name != fmt.name else s
            t.check(graph, f, plan)
            if isinstance(t, QuantStep):
                t.check_grid(f, plan)
    check_stream_steps(graph, steps)
    check_group_sizes(meta, graph, fmt, target, steps, formats)
    check_dense_shapes(meta, graph, fmt, steps, plan_dense, alt_plans, formats, packed)
    check_mamba_served(
        graph,
        packed is not None or any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps),
        target,
    )
    if plan_moe is not None and (
        packed is not None or any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps)
    ):
        _check_moe_bindings(meta, graph, fmt, target, plan_dense, plan_moe, formats, alt_plans, packed)


def _check_multicard(
    recipe: Recipe, steps: Sequence[Step], devices: list[torch.device], batch_size: int = 1
) -> None:
    if len(devices) <= 1:
        return
    import pickle

    try:
        pickle.dumps(recipe)  # every step, checkpoint ones too: the float stage's ranks get the sub-recipe
    except Exception as e:  # noqa: BLE001 -- a plugin's __reduce__ may raise any type
        raise Refused(
            f"multi-card builds send the recipe to each card process; it cannot be pickled: {e}"
        ) from e
    kinds = [s.kind for s in steps]
    if "layer" in kinds and "model" in kinds[kinds.index("layer") :]:
        raise Refused(
            "a model step after a layer step runs between layer passes; v0 builds that on 1 card only "
            "(multi-card builds run model steps once, before the ranks start)",
            one_card_fix(devices),
        )
    for s in steps:
        if _reruns_layers(s):
            raise Refused(
                f"step {s.name!r} re-runs layers on the calibration set (needs outputs or layer_io it "
                f"does not gather across cards); v0 builds it on 1 card only, not {len(devices)}: each "
                f"rank would see only its shard",
                one_card_fix(devices),
            )
    # a row-sharing step combines its per-rank sums through the ranks' collective, so every rank must run it:
    # a card with no calibration batch would skip the step and the others would wait on it forever
    sharing = [s.name for s in steps if getattr(s, "shares_rows", False)]
    samples = getattr(recipe.calib, "samples", None)
    if sharing and samples is not None:
        n_batches = -(-samples // batch_size)
        if n_batches < len(devices):
            raise Refused(
                f"calibration: {samples} samples in batches of {batch_size} give {n_batches} batches for "
                f"{len(devices)} cards, and a row-sharing step ({', '.join(sharing)}) prices its own batches "
                "on every card: lower batch_size, add calibration samples, or use fewer cards"
            )


def _calibration(
    source: str,
    recipe: Recipe,
    meta: PreTrainedModel,
    needs: frozenset[str],
    log: Callable[[str], None] | None = None,
) -> tuple[list[torch.Tensor] | None, list[dict[str, torch.Tensor]] | None, str | None]:
    text_config = cast(Any, meta.config).get_text_config()
    recipe.calib.pin()
    recipe.calib.index()  # a search host's required index: refuse here, before any step or weight loads
    # the float stage's calibration rows (checkpoint steps check their own data); a generate stage comes first
    pending, ids = (recipe.calib.pending(source), None) if needs else (None, [])
    inputs = None
    if needs and pending is None and getattr(recipe.calib, "images_source", False):
        ids, inputs = _calib.load_with_inputs(recipe.calib, source, log=log)
        check_calib_vocab(ids, recipe.calib.label, text_config.vocab_size)
    elif needs and pending is None:
        ids = _calib.load_ids(recipe.calib, source, log=log)
        check_calib_vocab(ids, recipe.calib.label, text_config.vocab_size)
    elif log is not None:
        if pending is not None:
            log(f"calibration: rows are generated first ({pending}); the contamination filter runs after")
        elif recipe.calib.index() is None:
            log("contamination filter: not run (no reference index)")
        else:
            log("contamination filter: no calibration rows read (no step or format needs statistics)")
    return ids, inputs, pending
