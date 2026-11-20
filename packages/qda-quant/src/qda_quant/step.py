"""A `Step` is one quantization method. It holds settings and math, never a calibration loop.

Layer steps declare required statistics in `needs` and change weights or quantization parameters through a
`LayerContext`. The engine computes each statistic once per input group per layer. Model steps see the whole
float model before or between layer passes. Checkpoint steps read a packed checkpoint and write a new one
in the same format.

Recipes may mix all three kinds. `build` runs them in stages, feeding the previous stage's packed output to
each checkpoint step. A supported packed checkpoint can also be the build's source.

The engine owns devices, residency, calibration batching, statistics and multi-GPU coordination.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from .exit import USAGE
from .registry import Registry
from .solve.grid import DEFAULT_GRID, GRIDS, known_grid
from .steps import BUILTINS

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from .arch import ExpertBank, LayerGraph, LinearGroup, ModelGraph
    from .clips import Clip
    from .format import Format
    from .solve.quant import Quantizer
    from .solve.units import BoundUnit
    from .target import KernelPlan

#: Statistics a layer step may ask for. The engine computes them; steps never hook modules.
#:  "hessian"        per LinearGroup: H = sum_t w_t x_t x_t^T  (fp32 accumulate), [in, in]
#:  "cross_hessian"  per LinearGroup: C = sum_t w_t (xf_t - x_t) x_t^T, [in, in], where xf is the float model's
#:                   input to the group and x the quantized model's (GPTAQ). Asking for it makes the engine keep
#:                   a float copy of the layer inputs; no extra cost when no step asks.
#:  "expert_hessian" per ExpertBank: stacked [E, in, in], tokens routed to each expert only
#:                   (router-weighted when the step sets `router_weighted`)
#:  "act_absmax"     per LinearGroup: per-input-channel max |x|, [in]
#:  "inputs"         per LinearGroup: a bounded sample of the group's input rows (for AWQ / clip search)
#:  "outputs"        per layer: the float layer output on the calibration batch (for loss-based search)
#:  "n_tokens"       per LinearGroup: int tokens seen; per ExpertBank: [E] tokens routed (0 = never routed)
#:  "layer_io"       per layer: an engine object yielding (hidden_states, args, kwargs, float_output) per
#:                   calibration batch, for steps that must re-run the layer (AutoRound, GSQ, loss-based AWQ).
#:                   The hidden states are the quantized stream; float_output is the float layer on them. The
#:                   engine owns the batches and their devices; the step only iterates.
#: Statistics may also come from plugins (`STATISTICS`).
StatName = Literal[
    "hessian", "cross_hessian", "expert_hessian", "act_absmax", "inputs", "outputs", "n_tokens", "layer_io"
]


@dataclass
class LayerContext:
    """Everything a layer step may read or change for one decoder layer."""

    layer: torch.nn.Module
    graph: LayerGraph
    model_graph: ModelGraph
    fmt: Format
    plan_dense: KernelPlan
    plan_moe: KernelPlan | None
    stats: dict[tuple[str, str], Any]  # (stat name, group or bank key) -> tensor
    # module -> its stored tensors and its "format" name (and any global scales); see `solve.units`
    qparams: dict[str, dict[str, Any]] = field(default_factory=dict)
    device: torch.device | None = None
    #: the ranks' collective (`engine.dist.Comm`) on a multi-card build whose step shares its rows
    #: (`Step.shares_rows`): each rank holds its own calibration batches and the step combines the per-rank
    #: sums itself. None on one card, and on a multi-card layer the engine gathers instead.
    comm: Any = None

    def stat(self, name: str, of: LinearGroup | ExpertBank | LayerGraph) -> Any:
        """The statistic `name` of a group, bank or (for per-layer stats) the layer itself."""
        return self.stats[(name, _key(of))]


def _key(of: Any) -> str:
    """Stats key: first module of a group, router of a bank, module name of a layer."""
    if hasattr(of, "modules"):
        return of.modules[0]
    if hasattr(of, "router"):
        return of.router
    return of.module


class Step(BaseModel):
    """Base class of every method. Subclasses are pydantic models: their fields are the settings.

    Class attributes:
        kind: "layer", "model" or "checkpoint".
        needs: statistics a layer step asks the engine for.
        name: registry name used in YAML recipes (defaults to the class name, lower-cased).
        needs_verified_graph: True: this step reads a derived relation that only
            `arch.describe(verify=True)` checks (``LinearGroup.norm``, ``.prev``,
            ``ModelGraph.extra["norm_offset"]``, or group.norm keys in token weights). False lets the
            engine skip that numeric check.
        multi_card_layer_io: True: a multi-card build gathers every rank's ``layer_io`` batches for this step,
            in rank order, so it re-runs layers over the rows a one-card build does; False keeps such a step
            on one card.
        shares_rows: True: a multi-card build keeps every rank's own share of the calibration rows (``inputs``
            and ``layer_io`` are not gathered) and runs the step on every rank, which combines its per-rank
            sums itself through ``LayerContext.comm`` (``engine.dist.Comm.all_rows``). It must be the first
            step of the layer pass, so every rank folds the same weights it would alone. A layer with routed
            expert banks is not shared: its expert statistics are gathered as for the other steps.
        preserves_quantized: True: this model step changes no weight that a layer step has already quantized
            (e.g. lm_head or embedding format). Only such model steps may follow a layer step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ClassVar[Literal["layer", "model", "checkpoint"]] = "layer"
    needs: ClassVar[frozenset[str]] = frozenset()
    multi_card_layer_io: ClassVar[bool] = False
    shares_rows: ClassVar[bool] = False
    name: ClassVar[str] = ""
    needs_verified_graph: ClassVar[bool] = False
    preserves_quantized: ClassVar[bool] = False
    #: Experimental: kept for the search space, not yet a recommended method (the docs say so).
    experimental: ClassVar[bool] = False
    #: Most cards a checkpoint step can spread its stage over (None: as many as the build has). A build on more
    #: cards than this runs the step on its own subset instead of refusing: `compile_plan` picks the fewest cards,
    #: largest free first, whose priced peak (`checkpoint_peak`) fits, so no config has to name devices for it.
    max_devices: ClassVar[int | None] = None

    #: File settings resolved beside a config file by `recipe.resolve_paths`; output paths are excluded.
    paths: ClassVar[tuple[str, ...]] = ()
    #: Model settings: local directories resolved beside the file, otherwise Hub ids.
    models: ClassVar[tuple[str, ...]] = ()

    def apply_layer(self, ctx: LayerContext) -> None:
        """Quantize / transform one layer in place (layer steps)."""
        raise NotImplementedError

    def apply_model(self, model: torch.nn.Module, graph: ModelGraph, run: Any) -> None:
        """Transform the whole model (model steps). `run` is the engine's build context."""
        raise NotImplementedError

    def apply_checkpoint(self, src: Path, out: Path, run: Any) -> None:
        """Read the packed checkpoint at `src` and write the result into `out` (checkpoint steps).

        `out` exists and is empty when the step is called, and the step writes its files into it. It must not
        create, replace or delete `out` itself. One exception to empty: on the last stage of a build with
        intermediates, `out` already holds the engine's ``.stages`` directory, which the step must leave alone.
        """
        raise NotImplementedError

    def resolve(self, fmt: Format, plan: KernelPlan, config: Any, source: str) -> Step:  # noqa: ARG002
        """This step with every setting whose default depends on the build resolved; the base returns `self`.

        `compile_plan` calls this once per step, before any check, with the build's format, the dense kernel plan,
        the source's ``config.json`` (a transformers config) and the source's local directory (a step that reads
        another model, such as a teacher the source records, finds it from there). The resolved steps are the
        recipe's, so ``resolved.yaml`` records the values used. Refusals stay in `check` / `check_checkpoint`.
        """
        return self

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse (raise `Refused`) before any GPU work if this step cannot run here."""
        return

    def check_checkpoint(self, ctx: Any) -> None:  # noqa: ARG002
        """Checkpoint steps: refuse before any stage of the build runs, from metadata only (no weights).

        `ctx` is the plan's `qda_quant.engine.packed.CheckpointPlan`: the source's config and meta skeleton graph,
        the format, the target's plans, the devices, the default teacher. `compile_plan` calls this for every
        checkpoint step, together with every other step's `check`, before anything runs.
        """
        return

    def model_step_bytes(self, meta: torch.nn.Module, graph: ModelGraph) -> int:  # noqa: ARG002
        """Model steps: bytes `apply_model` holds on the card beyond the weights, when the weights are on the card.

        A resident build runs its model steps with the whole model on the card, so the planner prices that stage
        as the weights plus this (`engine.residency.plan`); a streamed build runs them on the host. 0 by default.
        """
        return 0

    def checkpoint_peak(self, ctx: Any) -> StagePeak | tuple[StagePeak, ...] | None:  # noqa: ARG002
        """Checkpoint steps: this step's footprint on the card it runs on, from metadata only.

        A checkpoint step holds weights the float build's residency never sees (a teacher, a packed student, an
        optimizer, training activations), so its stage has to be sized on its own. `compile_plan` calls this after
        `check_checkpoint`, with the same `CheckpointPlan`, and refuses the plan when the returned floor is above
        the card's free memory, before any stage runs. None (the base) means the step does not model its memory;
        the plan then lists the stage without a number.

        A step that spreads a stage over several cards returns one `StagePeak` per card it runs on, each against
        that card's free memory.
        """
        return None


class Refused(ValueError):
    """A recipe, step, model or target combination that the library will not build. Says why.

    Attributes:
        fix: The one change or command that unblocks it, when the library knows one; a command prints it as the
            `fix:` line of its refusal (`qda_quant.exit.refusal`), and leaves the line out when there is none.
        code: The status a command exits with (`qda_quant.exit`): `USAGE` for the input, which is the default,
            or `ENVIRONMENT` for the machine -- a card, its free memory, the disk, a package, a sign-in. A
            refusal of the machine is raised with ``code=ENVIRONMENT``: the same command works once it changes.
    """

    def __init__(self, why: str, fix: str = "", *, code: int = USAGE) -> None:
        super().__init__(why)
        self.fix = fix
        self.code = code


#: Re-raise a probe failure instead of refusing it, so a bug report can carry the traceback.
PROBE_TRACEBACK_ENV = "QDA_TRACEBACK"
#: The next step for a model whose structure QDA cannot read.
ADAPTER_NEXT_STEP = (
    "if this layout is new, write a small adapter "
    "(https://github.com/humanfia/qda/blob/HEAD/docs/add-architecture.md)"
)


def _hub_id(name: str) -> str:
    """`org/name` for a path inside the Hugging Face cache (`.../models--org--name/snapshots/<rev>`), else `name`."""
    import re

    hit = re.search(r"models--([^/]+?)--([^/]+)/snapshots/[0-9a-f]+", name)
    return f"{hit.group(1)}/{hit.group(2)}" if hit else name


def _probe_name(model: Any) -> str:
    """The model's checkpoint id or directory, or its class name (`model` is an id, a path or a module)."""
    if isinstance(model, str) or hasattr(model, "__fspath__"):
        return _hub_id(str(model))
    named_source = getattr(model, "_qda_source", None)  # the id or directory the user named
    if named_source:
        return _hub_id(str(named_source))
    config = getattr(model, "config", None)
    for owner in (config, model):
        for attr in ("_name_or_path", "name_or_path"):
            named = getattr(owner, attr, None)
            if named:
                return _hub_id(str(named))
    return type(model).__name__


def _failing_module(model: Any, exc: BaseException) -> str | None:
    """The deepest module of `model` whose frame raised `exc`, or None when no frame's `self` is one."""
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return None
    names = {id(module): name for name, module in named() if name}
    found = None
    traceback = exc.__traceback__
    while traceback is not None:
        owner = traceback.tb_frame.f_locals.get("self")
        found = names.get(id(owner), found) if owner is not None else found
        traceback = traceback.tb_next
    return found


def probe_refusal(
    model: Any, stage: str, exc: BaseException, *, module: str | None = None, next_step: str | None = None
) -> Refused:
    """The `Refused` for an unexpected failure while QDA probes `model`; re-raises when QDA_TRACEBACK is set.

    One boundary for every probe (config, tokenizer, weightless construction, forward trace, graph
    derivation): the message names the model, where it failed and the error's first line, and the next step
    (an adapter for a new layout, or a bug report). The CLI's `except Refused` prints it and exits 2;
    `QDA_TRACEBACK=1` re-raises the original instead (exit 1), so a bug report can carry the traceback.

    Args:
        model: the checkpoint id or directory, or the model (or tokenizer) object being probed.
        stage: what QDA was doing, e.g. ``"probing its forward"``.
        exc: the exception to convert.
        module: the module where it failed; found from the exception's frames when omitted.
        next_step: what the user should do instead; an architecture adapter when omitted.
    """
    if os.environ.get(PROBE_TRACEBACK_ENV) not in (None, "", "0"):
        raise exc
    if (
        isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"
    ):  # torch.cuda's, a RuntimeError
        raise exc  # out of memory is not a layout QDA cannot read: an adapter would not help
    where = module or _failing_module(model, exc)
    lines = str(exc).strip().splitlines()
    why = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    at = f" in {where}" if where else ""
    return Refused(
        f"model {_probe_name(model)}: {stage} failed{at} ({why}); QDA cannot go on with this model\n"
        f"qda: next: {next_step or ADAPTER_NEXT_STEP}; "
        f"otherwise rerun with {PROBE_TRACEBACK_ENV}=1 to report a QDA bug"
    )


def remote_code_refusal(model: Any, what: str, exc: BaseException) -> Refused | None:
    """The `Refused` for a model whose ``what`` is code from its own repository (``trust_remote_code``), else None.

    QDA never runs a model repository's code: transformers raises a ValueError naming ``trust_remote_code`` when
    it is asked for one with ``trust_remote_code=False``, and this turns that into a refusal that says so.
    """
    if not isinstance(exc, ValueError) or "trust_remote_code=True" not in str(exc):
        return None
    return Refused(
        f"model {_probe_name(model)}: its {what} is code from the model repository (trust_remote_code); "
        "QDA never runs it",
        "use a model whose config, tokenizer and processor transformers implements itself",
    )


@dataclass(frozen=True)
class StagePeak:
    """A checkpoint step's estimated card footprint, from metadata only (`Step.checkpoint_peak`).

    The estimate is a floor: allocator caching and library temporaries are not modelled. `compile_plan` refuses a
    stage whose floor is above the card's free memory, before any stage runs; `qda plan` and the build report it.

    Attributes:
        step: the step's registry name.
        device: the card the step will run on.
        free: that card's free memory when the plan was made (unbounded on the CPU).
        terms: (term, bytes) pairs, largest first.
    """

    step: str
    device: str
    free: int
    terms: tuple[tuple[str, int], ...]

    @property
    def bytes(self) -> int:
        """The estimated peak, in bytes."""
        return sum(b for _, b in self.terms)

    @property
    def over_free(self) -> bool:
        """True if the estimate is above the card's free memory."""
        return self.bytes > self.free

    @property
    def largest(self) -> tuple[str, int]:
        """The term that dominates the estimate, as (name, bytes); ("", 0) when there are none."""
        return max(self.terms, key=lambda t: t[1]) if self.terms else ("", 0)

    def __str__(self) -> str:
        """One readable line: the estimate on its card and its terms (a stage may run on two cards)."""
        top = [f"{name} {b / 2**30:.1f}" for name, b in self.terms[:6] if b]
        rest = sum(b for _, b in self.terms[6:]) / 2**30
        if rest >= 0.05:
            top.append(f"other {rest:.2f}")
        return f"estimated at least {self.bytes / 2**30:.1f} GiB on {self.device}: {' + '.join(top)}"


@dataclass(frozen=True)
class Statistic:
    """What a statistic named in a step's `needs` requires of the engine. Register it in ``qda.stats``.

    Attributes:
        float_stream: the engine keeps and advances the float stream (one more forward per batch).
        single_card: `qda plan` refuses it on more than one card.
        provide: builds the object steps read from ``ctx.stats``, from this rank's calibration slots, the layer
            index and the card. None when the engine computes the statistic itself.
    """

    float_stream: bool = False
    single_card: bool = False
    provide: Callable[[list, int, torch.device], Any] | None = None


#: Statistics by the name steps put in `needs`: built-ins and package registrations.
STATISTICS: Registry[Statistic] = Registry(
    "stats",
    {
        "cross_hessian": "qda_quant.engine.stat_plugins:CROSS_HESSIAN",
        "layer_io_float": "qda_quant.engine.stat_plugins:LAYER_IO_FLOAT",
    },
)


def statistics_of(needs: Iterable[str]) -> dict[str, Statistic]:
    """The registered statistics among `needs`, by name, in sorted order."""
    return {name: STATISTICS[name] for name in sorted(needs) if name in STATISTICS}


def keeps_float_stream(needs: Iterable[str]) -> bool:
    """Whether any statistic among `needs` makes the engine keep the float stream."""
    return any(stat.float_stream for stat in statistics_of(needs).values())


class QuantStep(Step):
    """A step that picks codes or prices candidates on a quantization grid (`qda_quant.solve.grid`).

    Attributes:
        grid: name of the grid; None takes the recipe's (`Recipe` fills it in).
        default_grid: the grid this step takes when neither it nor the recipe names one, on formats that grid
            serves (`resolve`); None takes the recipe's default.
        quantizes: True when the step stores codes the export writes; False for one that only searches
            settings for a later step (AWQ's scales), which leaves every weight in the build dtype.
    """

    grid: str | None = None
    default_grid: ClassVar[str | None] = None
    quantizes: ClassVar[bool] = True

    def solver_workspace(self, rows: int, cols: int) -> int:  # noqa: ARG002
        """Bytes one expert of an ``[E, rows, cols]`` unit keeps on the card while this step solves it.

        `engine.planning.memory_terms` reserves it; 0 for a step whose solve needs no workspace of its own.
        """
        return 0

    def dense_workspace(self, rows: int, cols: int) -> int:
        """Bytes a dense ``[rows, cols]`` unit (no experts) keeps on the card while this step solves it.

        The planner prices a dense unit's solve with this; by default it is the per-expert `solver_workspace`.
        """
        return self.solver_workspace(rows, cols)

    def resolve(self, fmt: Format, plan: KernelPlan, config: Any, source: str) -> QuantStep:  # noqa: ARG002
        """This step with its grid named: its own, else `default_grid` if it serves `fmt` on `plan`, else the default.

        `compile_plan` calls this (`Step.resolve`), so the grid used is the one ``resolved.yaml`` records.
        """
        if self.grid is not None:
            return self
        name = self.default_grid
        if name is None or GRIDS[name].refusal(fmt, plan):
            if name is not None:
                logging.getLogger(__name__).info(
                    "%s: grid %r does not serve %s, using %r", self.name, name, fmt.name, DEFAULT_GRID
                )
            name = DEFAULT_GRID
        return self.model_copy(update={"grid": name})

    @field_validator("grid")
    @classmethod
    def _known_grid(cls, v: str | None) -> str | None:
        return known_grid(v)

    def grid_for(self, fmt: Format, plan: KernelPlan) -> str:
        """This step's grid, or the one that serves `fmt` when its own does not.

        A step names one grid but a build may store units in more than one format (``format_overrides``), and a
        file format's grid writes one layout and nothing else: `q4k` writes Q4_K records. Rather than refuse
        the pair, a step quantizes such a unit on the grid that declares it stores that format (`Grid.stores`),
        so one recipe can store some tensors in one layout and some in another.

        The step's own grid is always preferred where it serves the format, so a build with no overrides is
        unchanged. When no grid stores the format, the step keeps its own and `check_grid` refuses the build:
        a unit is never quantized on a grid whose records cannot hold it.
        """
        name = self.grid or DEFAULT_GRID
        if GRIDS[name].refusal(fmt, plan) is None:
            return name
        return next((g for g in sorted(GRIDS) if GRIDS[g].stores == fmt.name), name)

    def quantizer(self, fmt: Format, plan: KernelPlan) -> Quantizer:
        """The quantizer of the grid that serves `fmt` on `plan` (`grid_for`)."""
        from .solve.quant import Quantizer

        return Quantizer.of(fmt, plan, self.grid_for(fmt, plan))

    def units(self, ctx: LayerContext) -> Iterator[BoundUnit]:
        """The layer's weights this step quantizes on this card, each with its quantizer (`qda_quant.solve.units`)."""
        from .solve.units import bound

        return bound(self, ctx)

    def picks_ranges(self) -> bool:
        """Whether this step chooses group scales from ranges (a clip search, a learned range).

        A grid that fits its scales on the values (`Grid.range_rule` False) has no range to choose, so
        `check_grid` refuses the pair.
        """
        return False

    def clip_search(self) -> Clip | None:
        """The clip search this step asks for its scales, if it has the setting."""
        return None

    @property
    def default_clip(self) -> str:
        """The clip this step runs where its named one works on another grid: its ``clip`` setting's default."""
        return getattr(type(self).model_fields.get("clip"), "default", "minmax")

    def clip_for(self, grid: str) -> Clip | None:
        """The clip this step runs on a unit solved on `grid`: its own where that works there, the default where not.

        A clip that names the grids it works on (`Clip.only_grid`) applies to a recipe whose modules are
        stored in more than one format (``format_overrides``) by unit: it runs on the units solved on a grid
        it names, and a unit on another grid takes the clip the step would take with ``clip`` unset. A clip
        that names no grid applies to every unit, as before.
        """
        from .clips import CLIPS

        clip = self.clip_search()
        if clip is None or clip.only_grid is None:
            return clip
        allowed = (clip.only_grid,) if isinstance(clip.only_grid, str) else clip.only_grid
        if grid in allowed:
            return clip
        return CLIPS[self.default_clip]

    def for_format(self, fmt: Format, plan: KernelPlan) -> QuantStep:
        """This step as it runs on a unit stored as `fmt` on `plan`: with the clip that unit gets (`clip_for`)."""
        clip, named = self.clip_for(self.grid_for(fmt, plan)), self.clip_search()
        if clip is None or named is None or clip.name == named.name:
            return self
        return self.model_copy(update={"clip": clip.name})

    def check_grid(self, fmt: Format, plan: KernelPlan) -> None:
        """Refuse a grid whose checkpoint would not be what `plan`'s kernel computes (`compile_plan` calls this).

        `grid_for` lets a unit in another format take the grid that serves it, so this refuses only a format
        that no grid serves at all.
        """
        name = self.grid_for(fmt, plan)
        if why := GRIDS[name].refusal(fmt, plan):
            raise Refused(f"{self.name}: grid {name!r} on {fmt.name} ({plan.kernel}): {why}")
        if self.picks_ranges() and not GRIDS[name].range_rule:
            raise Refused(
                f"{self.name}: grid {name!r} fits its scales on the values; {self.name} picks ranges"
            )
        clip = self.clip_search()
        if clip is not None and clip.only_grid is not None:
            allowed = (clip.only_grid,) if isinstance(clip.only_grid, str) else clip.only_grid
            if name not in allowed:
                only = allowed[0] if len(allowed) == 1 else f"one of {sorted(allowed)}"
                raise Refused(f"{self.name}: clip {clip.name!r} works on grid {only!r} only, not {name!r}")


def _adopt(key: str, cls: type[Step]) -> None:
    """Check a step plugin from either source and give it its recipe name."""
    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise TypeError(f"step plugin {key!r} is {cls!r}, not a Step subclass")
    cls.name = key


#: Steps by recipe name: built-ins and package registrations (see `qda_quant.registry`).
REGISTRY: Registry[type[Step]] = Registry("steps", BUILTINS, on_add=_adopt)
