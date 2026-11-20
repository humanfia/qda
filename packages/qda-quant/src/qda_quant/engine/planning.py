"""Compile and validate build plans before loading weights."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..arch import DEFAULT_KEEP, ModelGraph
from ..exit import ENVIRONMENT
from ..format import Format
from ..intake.model_ref import parse
from ..recipe import Recipe
from ..step import QuantStep, Refused, StagePeak, Step
from ..target import KernelPlan, Target
from . import residency as _res
from .build_validation import (
    UNSERVABLE_KEPT_RULE,
    UNSERVABLE_SHARE,
    QuantWidth,
    Unservable,
    WidthProblem,
    _calibration,
    _check_checkpoint_stage,
    _check_head_steps,
    _check_layer_steps,
    _check_multicard,
    _check_solve_offsets,
    _graph,
    _source_format,
    _steps,
    card_name,
    check_calib_vocab,
    check_group_sizes,
    check_order,
    check_rerun_passes,
    check_token_weight_readers,
    check_widths,
    kept_unservable,
    shape_unservable,
    single_card_steps,
    unservable_units,
    unservable_widths,
    working_group_sizes,
)
from .checkpoint_size import _output_bytes, packed_weights
from .format_bindings import expert_defaults, resolve_formats
from .memory import (
    CALL_ARGS_BYTES,
    GDN_CHUNK,
    _residency,
    _stage_devices,
    _store_peak,
    linear_attention_bytes,
    memory_terms,
    per_layer_inputs_bytes,
)
from .stores import WEIGHT_STORES, WeightStore

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .packed import Packed

__all__ = [
    "CALL_ARGS_BYTES",
    "GDN_CHUNK",
    "METADATA_FILES",
    "Plan",
    "QuantWidth",
    "UNSERVABLE_KEPT_RULE",
    "UNSERVABLE_SHARE",
    "Unservable",
    "SOURCE_FILES",
    "Segment",
    "WidthProblem",
    "check_calib_vocab",
    "check_group_sizes",
    "check_order",
    "check_output",
    "check_rerun_passes",
    "check_token_weight_readers",
    "check_widths",
    "compile_plan",
    "expert_defaults",
    "linear_attention_bytes",
    "local_source",
    "memory_terms",
    "per_layer_inputs_bytes",
    "resolve_devices",
    "resolve_formats",
    "segments",
    "shardable_needs",
    "kept_unservable",
    "shape_unservable",
    "unservable_units",
    "unservable_widths",
    "working_group_sizes",
]

Segment = tuple[str, list[Step]]  # ("model", [step]) or ("layers", [steps...])


@dataclass
class Plan:
    """Everything decided before a weight is loaded."""

    source: str
    graph: Any  # ModelGraph of the meta skeleton
    fmt: Format
    target: Target
    plan_dense: KernelPlan
    plan_moe: KernelPlan | None
    segments: list[Segment]
    residency: _res.Residency
    devices: list[torch.device]
    calib_ids: list[torch.Tensor] | None  # tokenized, cut and ordered at plan time; None until generated
    #: Calibration rows the build must generate first (its ``generate`` stage), or None. Their size and vocabulary
    #: are checked after generation, before any layer work.
    generate: str | None = None
    #: steps that run after the float build, each on the previous stage's packed output (`engine.packed.staged`)
    checkpoint_steps: list[Step] = field(default_factory=list)
    packed_source: bool = False  # the source is a packed checkpoint: no float build, checkpoint steps only
    #: per calibration sample, its model inputs besides the ids (an image's ``pixel_values``); None for text
    calib_inputs: list[dict[str, torch.Tensor]] | None = None
    #: the config's ``ignore`` entries (roles, names, ``re:`` patterns); the build describes the model with them
    keep: frozenset[str] = DEFAULT_KEEP
    #: Parameter bytes already included in each loaded layer that its adapter keeps on the host.
    host_parameter_bytes: dict[str, int] = field(default_factory=dict)
    #: bytes of the checkpoint a float build writes (`engine.checkpoint_size.checkpoint_bytes`); 0 when the source is
    #: already packed, whose stages rewrite the checkpoint they read. An exporter that writes another artefact
    #: and can size it exactly replaces this (`output.exporters.record_exporter`).
    output_bytes: int = 0
    #: True when `output_bytes` is the exporter's own exact size rather than the engine's estimate
    output_bytes_exact: bool = False
    #: bytes of the weights alone inside that output, without the container's header and metadata; 0 when the
    #: exporter does not know. A budget quoted against another toolchain's file compares this, not `output_bytes`
    weight_bytes: int = 0
    #: inputs the exporter would build itself instead of reading a path the config names (`Exporter.builds`)
    exporter_builds: tuple[str, ...] = ()
    #: unit key -> the format a config's ``format_overrides`` give it (the body `fmt` for every other unit)
    formats: dict[str, Format] = field(default_factory=dict)
    #: (format name, layer kind) -> kernel plan, for the formats in `formats`
    plans: dict[tuple[str, str], KernelPlan] = field(default_factory=dict)
    #: unit class -> the format the planner, not a config, gave it: ``{"routed_experts": "w4a16-sym-g128"}``
    #: when the body format names no symmetry and the experts default to symmetric (`expert_defaults`)
    format_defaults: dict[str, str] = field(default_factory=dict)
    #: the units `format_defaults` reached; they are in `formats`, but no ``format_overrides`` rule names them
    defaulted: frozenset[str] = frozenset()
    #: pattern -> format the exporter's own stored types required, which no config rule stated (`GGUF`: the
    #: mix's Q6_K/Q5_K tensors, so their codes are solved where the file writes them)
    stored_formats: dict[str, str] = field(default_factory=dict)
    #: unit id -> the stored types its tensors disagree on: a unit one format cannot cover, whose solved codes
    #: the export still drops (`Exporter.stored_format_rules`)
    format_unmet: dict[str, list[str]] = field(default_factory=dict)
    #: the units in `formats` bound only by `stored_formats`, not by a recipe rule: a plain rounding build
    #: still needs no layer pass for them (`Exporter.writes_alone`)
    derived_units: frozenset[str] = frozenset()
    #: each checkpoint step's estimated card footprint (`Step.checkpoint_peak`), in execution order; a step that
    #: does not model its memory has no entry. `residency` above sizes the float build only.
    stage_peaks: list[StagePeak] = field(default_factory=list)
    #: index in `checkpoint_steps` -> the cards that step runs on. A build's cards are `devices`; a step capped
    #: by `Step.max_devices` runs on the fewest of them that hold its priced stage (see
    #: `engine.memory._stage_devices`). Keyed by index: repeated steps of one name can need different cards.
    stage_devices: dict[int, list[torch.device]] = field(default_factory=dict)
    #: the calibration contamination filter's record (index, rows, dropped count and ids); written to build.json
    calibration: dict = field(default_factory=dict)
    #: the chat-template check of the calibration text rows (own / plain / foreign counts); written to build.json
    calib_template: dict | None = None
    #: the data lint of every data file the plan read (calibration, a step's own data): one record per file
    data_lint: list[dict] = field(default_factory=list)
    #: (rows, cols) and format of every tensor a layer step packs, one per stored tensor (`packed_weights`):
    #: the decoder Linears a bits-per-weight budget prices. Empty when no layer step quantizes.
    linears: tuple[tuple[tuple[int, int], Format], ...] = ()
    #: ``when: accepted`` steps the plan left out, each ``{"step", "reason"}`` (`_skip_refused`); in build.json
    steps_skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def store(self) -> WeightStore:
        """The weight store the residency names."""
        return WEIGHT_STORES[self.residency.weights]

    @property
    def staged(self) -> bool:
        """True if the build runs in stages (checkpoint steps, or a packed source)."""
        return bool(self.checkpoint_steps) or self.packed_source

    def act_needs(self) -> frozenset[str]:
        """Statistics the build's kernel plans read off the calibration set (NVFP4's input ranges)."""
        return _act_needs(self.plan_dense, self.plan_moe, *self.plans.values())

    def clips_by_format(self) -> dict[str, dict[str, str]]:
        """Each layer step's clip per stored format, where a grid-named clip falls back on one of them.

        The body format and every ``format_overrides`` / ``format_defaults`` format name the units a step
        solves. A clip that names its grids (`Clip.only_grid`) runs on the units whose grid it names; a unit
        on another grid takes the step's default clip (`QuantStep.clip_for`), and this records the pair. Empty
        when no step does that, so a single-format build records nothing new.
        """
        formats: dict[str, tuple[Format, KernelPlan]] = {self.fmt.name: (self.fmt, self.plan_dense)}
        for f in self.formats.values():
            plan = self.plans.get((f.name, "dense")) or self.plans.get((f.name, "moe")) or self.plan_dense
            formats.setdefault(f.name, (f, plan))
        out: dict[str, dict[str, str]] = {}
        for kind, steps in self.segments:
            if kind != "layers":
                continue
            for s in steps:
                if not isinstance(s, QuantStep) or (named := s.clip_search()) is None:
                    continue
                if named.only_grid is None:
                    continue
                got: dict[str, str] = {}
                for name, (f, plan) in sorted(formats.items()):
                    clip = s.clip_for(s.grid_for(f, plan))
                    got[name] = named.name if clip is None else clip.name
                if len(set(got.values())) > 1:
                    out[s.name] = got
        return out


def resolve_devices(devices: Sequence[int | str | torch.device] | None) -> list[torch.device]:
    """Card list of a build, any number of cards; default every visible card, or the CPU without one.

    Ints are physical indices when ``CUDA_VISIBLE_DEVICES`` lists them (as a GPU scheduler sets it), else logical.
    One request may use only one of the two readings, and each card may appear once: with a lease of ``2,3,4,5``,
    ``[0, 1, 2, 3]`` would otherwise turn 0 and 2 into the same card, and two ranks would share it silently. The
    visible count is ``CUDA_VISIBLE_DEVICES`` when it is set, else `torch.cuda.device_count`. The CPU is exempt
    from the repeat check: no two ranks share it as a card, and fixtures pass it twice to stand for several cards.

    Raises:
        Refused: a named CUDA card is not one this process can see; the request mixes the two readings of an int;
            or two of its entries resolve to the same card.
    """
    if devices is None or len(devices) == 0:
        n = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(n)] or [torch.device("cpu")]
    lease = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    vis = [v.strip() for v in lease.split(",") if v.strip()]
    out, physical, logical = [], [], []
    for d in devices:
        if isinstance(d, torch.device):
            out.append(d)
        elif isinstance(d, str) and not d.isdigit():
            out.append(torch.device(d))
        elif str(d) in vis:
            physical.append(d)
            out.append(torch.device(f"cuda:{vis.index(str(d))}"))
        else:
            logical.append(d)
            out.append(torch.device(f"cuda:{int(d)}"))
    visible = len(vis) if vis else torch.cuda.device_count()
    for d in out:
        if d.type == "cuda" and d.index is not None and d.index >= visible:
            raise Refused(
                f"{d} is not visible: the process sees {visible} card(s)",
                "name a card this process can see (--set devices=[0]), or unset CUDA_VISIBLE_DEVICES",
                code=ENVIRONMENT,
            )
    if physical and logical:
        raise Refused(
            f"devices {list(devices)} mix two readings: with CUDA_VISIBLE_DEVICES={lease}, {physical} name"
            f" physical cards and {logical} name logical ones. Pass only physical ids ({', '.join(vis)}) or only"
            f" logical ones (0..{visible - 1})"
        )
    # the CPU is exempt: no two ranks share it as a card, and fixtures pass it twice to stand for several cards
    seen: dict[torch.device, object] = {}
    for src, d in zip(devices, out, strict=True):
        if d.type != "cuda":
            continue
        if d in seen:
            if vis:
                pick = f"physical ids ({', '.join(vis)}) or logical ones (0..{visible - 1})"
            elif visible:
                pick = f"logical ids 0..{visible - 1}"
            else:
                pick = "one entry per card"
            raise Refused(
                f"devices {list(devices)} resolve to {', '.join(str(d) for d in out)}: {seen[d]!r} and {src!r}"
                f" both mean {d}. Pass each card once: {pick}"
            )
        seen[d] = src
    return out


def segments(steps: Sequence[Step]) -> list[Segment]:
    """Split recipe steps into passes: runs of layer steps, and single model steps.

    Checkpoint steps never reach a pass: `engine.packed.staged` splits them off first.
    """
    out: list[Segment] = []
    for s in steps:
        if s.kind == "checkpoint":
            raise Refused(f"step {s.name!r} works on a packed checkpoint; build() runs it as its own stage")
        if s.kind == "model":
            out.append(("model", [s]))
        elif out and out[-1][0] == "layers":
            out[-1][1].append(s)
        else:
            out.append(("layers", [s]))
    if not out or out[-1][0] != "layers":
        out.append(("layers", []))  # the export runs in a layer pass
    return out


#: files of a hub repo that a build reads: configs, safetensors weights, tokenizer, chat template
METADATA_FILES = ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")


SOURCE_FILES = (*METADATA_FILES, "*.safetensors")


def local_source(model: str | Path, *, weights: bool | None = True) -> str:
    """A local directory holding `model`: the path itself, or the HF hub snapshot of a repo id.

    The writer copies and packs the source safetensors shards directly, so a repo id is resolved to its files
    once, through the HF cache (``HF_HUB_OFFLINE`` and ``HF_HOME`` apply as usual). ``org/name@rev`` pins a
    branch, tag or commit; a bare ``org/name`` gets the hub's current main, so a later rebuild may read
    other weights. Local paths are absolute, explicitly relative (``./`` or ``../``), or Path objects.
    ``weights=False`` fetches configs and tokenizer files only: planning reads nothing else, and
    a model can be terabytes. ``weights=None`` fetches every file of the repo, for a consumer that serves the
    checkpoint rather than planning a build of it.

    Raises:
        Refused: `model` is not a directory, and the hub (or the offline cache) has no such repo or revision.
    """
    local = isinstance(model, Path) or Path(model).is_absolute() or str(model).startswith(".")
    if local and Path(model).is_dir():
        if weights and not any(Path(model).glob("*.safetensors")):  # a build packs these shards
            held = " (it holds pytorch_model*.bin)" if any(Path(model).glob("pytorch_model*.bin")) else ""
            raise Refused(
                f"model {str(model)!r}: no *.safetensors weights{held}",
                "QDA reads safetensors shards: put the model's *.safetensors in the directory "
                "(`hf download <repo> --local-dir <dir>`), or name the hub id",
            )
        return str(model)
    if local:  # a path is never a repo id: say what is wrong with the path instead of asking the hub
        if Path(model).exists():
            raise Refused(
                f"model {str(model)!r}: a file, not a model directory",
                "name the directory that holds config.json and the *.safetensors shards",
            )
        raise Refused(
            f"model {str(model)!r}: no such directory",
            "a path (/, ./ or ../) names a local model directory; if it is an earlier recipe's `out`, "
            "build that recipe first",
        )
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    repo, revision = parse(model)
    files: list[str] | None = None if weights is None else list(SOURCE_FILES if weights else METADATA_FILES)
    try:
        return snapshot_download(repo, revision=revision, allow_patterns=files)
    except GatedRepoError as e:
        # a licence-gated repo answers 401 without a token: the fix is an accepted licence and a token, not a path
        first = str(e).strip().splitlines()[0]
        raise Refused(
            f"model {str(model)!r}: the hub repo is gated: accept its licence at "
            f"https://huggingface.co/{repo}, then set HF_TOKEN or run `hf auth login`, and retry: {first}"
        ) from None
    except (HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as e:
        first = str(e).strip().splitlines()[0]
        raise Refused(f"model {str(model)!r}: not a local directory, and not a hub repo: {first}") from None


def check_output(out_dir: str | Path) -> Path:
    """The absolute path of a build's output; refuses a path the build could not publish to.

    A build publishes by renaming a directory beside the output onto it (`engine.build`), so the output must
    not exist, or be an empty directory on its parent's filesystem (the rename replaces that empty directory).
    Whatever else is at the path stays: this refuses it and removes nothing.

    Raises:
        Refused: the path is outside ``QDA_OUTPUT_ROOTS`` (`qda_quant.output.roots`); holds something; is a
            mount point (empty, but a rename cannot replace it); or is the current directory (empty, but the
            rename would leave this process's shell in a deleted one).
    """
    from ..output.roots import output_path

    out = output_path(out_dir)  # under QDA_OUTPUT_ROOTS when set: a relative path goes in the first root
    if os.path.lexists(out) and not (out.is_dir() and not any(out.iterdir())):
        raise Refused(
            f"output {out} exists and is not an empty directory; "
            "a build never replaces what is there, name a new path"
        )
    if out.is_dir() and out.samefile("."):
        raise Refused(
            f"output {out} is the current directory; publishing would replace it under your shell, "
            "so name a directory inside it or run from elsewhere"
        )
    if os.path.ismount(out):
        raise Refused(
            f"output {out} is a mount point; a build publishes by rename, so name a directory inside it"
        )
    return out


def shardable_needs(source: str, recipe: Recipe, fmt: Format | None, target: Target) -> frozenset[str] | None:
    """The statistics of a recipe of layer steps alone, or None when it has model or checkpoint steps.

    `build` asks this before it starts card processes. A recipe whose layer steps ask for no statistics has no
    forward pass to shard: the layer work is one process', and a second card only duplicates every layer's
    source read and the per-layer broadcast, so such a build runs in one process on one card whatever `devices`
    says. A recipe with a model step (whose weights fan out to the ranks) or a checkpoint step stays on the
    requested cards. Resolving the steps here is `compile_plan`'s own first step (config and metadata only,
    `Step.resolve` is idempotent for a resolved step) and `compile_plan` resolves the same steps again.
    """
    packed, fmt, plan_dense = _source_format(source, fmt, target)
    steps, later, _ = _steps(source, recipe, fmt, plan_dense, packed)
    if later or any(s.kind != "layer" for s in steps):
        return None
    return frozenset().union(*(s.needs for s in steps))


def one_card_devices(
    given: Sequence[int | str | torch.device] | None,
    devices: list[torch.device],
    source: str,
    recipe: Recipe,
    fmt: Format | None,
    target: Target,
    log: Callable[[str], None],
) -> list[torch.device]:
    """The cards a build of `recipe` runs on: `devices` (the resolution of `given`), or its first card alone.

    `build` and `BuildConfig.plan` both ask this, so a plan reports the cards its build uses. A recipe with a
    step that runs on one card only (`single_card_steps`) takes the first card when `given` is unset (None or
    empty: every visible card), and says so on `log`; a `given` that names several cards is refused with the
    one-card setting as its fix. Without such a step, or on one card, `devices` is returned unchanged.
    """
    if len(devices) <= 1:
        return devices
    packed, fmt, plan_dense = _source_format(source, fmt, target)
    steps, _, _ = _steps(source, recipe, fmt, plan_dense, packed)
    if not (one := single_card_steps(steps)):
        return devices
    name, why = one[0]
    first = card_name(devices[0])
    if not given:
        log(f"step {name!r} runs on one card: devices = [{first}]")
        return devices[:1]
    if isinstance(given[0], int | str):
        first = str(given[0])
    raise Refused(
        f"step {name!r} {why}; v0 builds it on 1 card only, not {len(devices)}",
        f"set devices: [{first}]",
    )


def _act_needs(*plans: KernelPlan | None) -> frozenset[str]:
    """Union of the plans' own statistics needs, skipping the absent plans (no MoE row, no overrides)."""
    return frozenset().union(*(p.act_needs for p in plans if p is not None))


def compile_plan(
    source: str,
    recipe: Recipe,
    fmt: Format | None,
    target: Target,
    devices: list[torch.device],
    dtype: torch.dtype = torch.bfloat16,
    residency: str | None = None,
    keep: frozenset[str] | None = None,
    batch_size: int = 1,
    overrides: Sequence[tuple[str, str]] = (),
    log: Callable[[str], None] | None = None,
    exporter: Any = None,
) -> Plan:
    """Plan a build without loading weights or touching a card's memory; raises `Refused` with the reason.

    Every stage is planned and every step checked here, before anything runs: layer and model steps with
    `Step.check`, checkpoint steps with `Step.check_checkpoint` on a `engine.packed.CheckpointPlan` (metadata
    only). `fmt` may be None only for a packed source, which has its own.

    `exporter` is the output the build will write, when the caller has one: an output that stores tensors in
    another type may write solved codes only in that type, so the units it names are bound before the solve
    (`Exporter.stored_format_rules`). A pattern `overrides` already states is left as the recipe stated it.
    """
    packed, fmt, plan_dense = _source_format(source, fmt, target)
    steps, later, config = _steps(source, recipe, fmt, plan_dense, packed)
    # a kernel plan may read statistics of its own (NVFP4's input ranges), whether or not a step does: a
    # data-free build (`rtn`) still needs its calibration rows cut through the kernel's activation round trip
    step_needs = frozenset().union(*(s.needs for s in steps))
    needs = step_needs | plan_dense.act_needs
    (
        meta,
        graph,
        plan_moe,
        formats,
        alt_plans,
        format_defaults,
        defaulted,
        keep,
        stored_formats,
        derived,
        unmet,
    ) = _graph(
        source,
        dtype,
        target,
        keep,
        packed,
        fmt,
        plan_dense,
        overrides,
        recipe,
        steps,
        needs,
        devices,
        exporter,
    )
    steps, skipped = _skip_refused(steps, graph, fmt, plan_dense, log)
    # a config's `format_overrides` may give the experts (`plan_moe`) or a unit (`plans`) a kernel plan of its
    # own, whose statistics no step names either
    step_needs = frozenset().union(*(s.needs for s in steps))
    needs = step_needs | plan_dense.act_needs | _act_needs(plan_moe, *alt_plans.values())
    _check_head_steps(graph, steps, keep)
    _check_layer_steps(
        steps, meta, graph, fmt, target, plan_dense, plan_moe, devices, formats, alt_plans, packed
    )
    _check_solve_offsets(meta, graph, steps)
    output_bytes = _output_bytes(source, packed, meta, graph, fmt, steps, dtype, formats)
    linears = packed_weights(meta, graph, fmt, steps, formats)
    checkpoint_ctx, stage_peaks, stage_devices = _check_checkpoint_steps(
        source,
        recipe,
        later,
        config,
        meta,
        graph,
        fmt,
        target,
        plan_moe,
        devices,
        packed,
        output_bytes,
        steps,
        formats,
    )
    if log is not None:
        for n, s in enumerate(later):
            run_on = stage_devices.get(n, devices)
            if len(run_on) != len(devices):
                log(
                    f"stage {s.name}: {', '.join(str(d) for d in run_on)} of this build's "
                    f"{len(devices)} card(s), the fewest that hold it (the step takes at most {s.max_devices})"
                )
    if over := next((p for p in stage_peaks if p.over_free), None):
        name, b = over.largest
        raise Refused(
            f"stage {over.step}: {over}, above the {over.free / 2**30:.1f} GiB free on {over.device}; "
            f"largest term {name} {b / 2**30:.1f} GiB",
            code=ENVIRONMENT,
        )
    _check_multicard(recipe, steps, devices, batch_size)
    ids, inputs, pending = _calibration(source, recipe, meta, needs, log)
    # no step keeps a whole-model state: let the planner take the store that holds the least (`_residency`); a
    # format's own statistics stream the calibration activations but keep no step's state
    least_memory = not step_needs
    res, layer_bytes, loaded, host_parameter_bytes, fit = _residency(
        recipe, meta, graph, steps, needs, fmt, devices, dtype, residency, batch_size, formats, least_memory
    )
    plan = Plan(
        source,
        graph,
        fmt,
        target,
        plan_dense,
        plan_moe,
        segments(steps) if packed is None else [],
        res,
        devices,
        ids,
        pending,
        later,
        packed is not None,
        inputs,
        frozenset(keep),
        host_parameter_bytes,
        output_bytes=output_bytes,
        formats=formats,
        plans=alt_plans,
        format_defaults=format_defaults,
        defaulted=defaulted,
        stored_formats=stored_formats,
        format_unmet=unmet,
        derived_units=derived,
        stage_peaks=stage_peaks,
        stage_devices=stage_devices,
        linears=linears,
        steps_skipped=skipped,
    )
    if (record := recipe.calib.contamination_record()) is not None:
        plan.calibration = record
    plan.calib_template = recipe.calib.template_record()
    plan.data_lint = recipe.calib.lint_records()
    if log is not None and plan.data_lint:
        from ..calib.template import lint_block

        log(lint_block(plan.data_lint))
    return _store_peak(plan, graph, loaded, layer_bytes, fit, residency, least_memory)


def _skip_refused(
    steps: list[Step],
    graph: Any,
    fmt: Format,
    plan_dense: KernelPlan,
    log: Callable[[str], None] | None,
) -> tuple[list[Step], list[dict[str, str]]]:
    """The steps left after every ``when: accepted`` step whose `Step.check` refuses, and one record per skip.

    Decided before anything else is planned, so the plan, its peak and every later step are what they would be
    without the skipped step. Only a `Refused` from that step's own `check` is a skip.
    """
    kept, skipped = [], []
    for s in steps:
        if getattr(s, "when", "always") == "accepted":
            try:
                s.check(graph, fmt, plan_dense)
            except Refused as e:
                skipped.append({"step": s.name, "reason": str(e)})
                if log is not None:
                    log(f"step {s.name}: skipped (when: accepted): {e}")
                continue
        kept.append(s)
    return kept, skipped


def _refuse_early(plan: Plan, recipe: Recipe, devs: list[torch.device], batch_size: int) -> None:
    """Refuse what the plan can see before the generate stage runs: too few batches."""
    if not (plan.generate and len(devs) > 1 and _equal_lengths(recipe.calib)):
        return
    n_batches = -(-recipe.calib.samples // batch_size)  # samples of one length: exactly this many batches
    if n_batches < len(devs):
        raise Refused(
            f"calibration: {recipe.calib.samples} samples in batches of {batch_size} give "
            f"{n_batches} batches for {len(devs)} cards: every card needs one; lower batch_size, add "
            "calibration samples, or use fewer cards"
        )


def _equal_lengths(calib: Any) -> bool:
    """True if the generated samples are known to share one length before they exist, so their batch count is.

    ``selfgen`` writes text or messages rows, and the default cut gives every such sample exactly ``seqlen``
    tokens. With ``chunks`` the last piece of a row is shorter, and a source's rows are otherwise unknown: then
    the count is checked after generation, from the rows.
    """
    from ..calib.sources import SelfgenSource

    return isinstance(calib, SelfgenSource) and not calib.chunks


def _check_checkpoint_steps(
    source: str,
    recipe: Recipe,
    later: Sequence[Step],
    config: Any,
    meta: PreTrainedModel,
    graph: ModelGraph,
    fmt: Format,
    target: Target,
    plan_moe: KernelPlan | None,
    devices: list[torch.device],
    packed: Packed | None,
    output_bytes: int,
    steps: Sequence[Step],
    formats: dict[str, Format],
) -> tuple[Any, list[StagePeak], dict[int, list[torch.device]]]:
    from dataclasses import replace

    from .packed import CheckpointPlan

    if not later:
        return None, [], {}
    _check_checkpoint_stage(graph, meta, fmt, formats, steps, later)
    teacher = packed.teacher() if packed is not None else source
    ctx = CheckpointPlan(
        source, packed, teacher, config, graph, meta, fmt, target, plan_moe, devices, recipe, output_bytes
    )
    peaks, by_step = [], {}
    for n, s in enumerate(later):
        # `_stage_devices` prices the stage once and that pricing is the plan's: reading free memory twice
        # (once to choose the cards, once to check them) let a card's reading move between the two reads, and a
        # plan chose on the first reading then refused on the second
        devs, priced = _stage_devices(s, ctx, devices)
        by_step[n] = devs
        s.check_checkpoint(replace(ctx, devices=devs) if devs is not devices else ctx)
        peaks.extend(priced)
    return ctx, peaks, by_step
