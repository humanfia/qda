"""Exporters: the output format a build writes, named by ``exporter: name`` or ``exporter: {name: {settings}}``.

Built-ins: ``compressed_tensors`` (default, vLLM), ``gguf`` (llama.cpp) and ``awq_bf16`` (SGLang MoE).
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict

from ..registry import Registry
from .export import Writer, check_unchanged  # noqa: F401  (check_unchanged stays importable from here)

if TYPE_CHECKING:
    from ..arch import LayerGraph, ModelGraph
    from ..engine.planning import Plan
    from ..format import Format
    from ..recipe import Recipe
    from ..target import Target

#: The exporter of a config that names none.
DEFAULT_EXPORTER = "compressed_tensors"


class Exporter(BaseModel):
    """One output format. Subclass it, add settings as fields, and register the class in ``qda.exporters``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ClassVar[str] = ""
    #: True when the output is a packed checkpoint later stages can read (a staged build hands its
    #: intermediates on in that layout). Inherited by subclasses.
    feeds_stages: ClassVar[bool] = False
    #: True when the output can declare that ``format_overrides`` store some modules in another format.
    supports_formats: ClassVar[bool] = False
    #: True when the output stores a routed bank's projections as separate tensors, so a bank's down
    #: projections may hold a format of their own (`solve.units.down_key`). vLLM and SGLang load a bank as one
    #: fused module with one scheme, so their outputs cannot.
    splits_banks: ClassVar[bool] = False
    #: The `qda_quant.target` this output is served on, when its runtime decides rather than the config: a
    #: GGUF is read by llama.cpp whatever card it runs on. None takes the config's ``target``.
    target_name: ClassVar[str | None] = None

    def refusal(self, plan: Plan, recipe: Recipe) -> str | None:  # noqa: ARG002
        """Why this exporter cannot write the planned build, or None."""
        return None

    def stored_format_rules(self, graph: ModelGraph) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
        """``format_overrides`` this output's own stored types require, and the units they cannot cover.

        Asked before units are bound, because codes solved in one type cannot be stored in another.

        Returns:
            ``([{"pattern", "format"}], {unit id: what it disagrees on})``; both empty by default.
        """
        return [], {}

    def output_bytes(self, plan: Plan, recipe: Recipe) -> int | None:  # noqa: ARG002
        """The exact size of the artefact this exporter writes, or None when it cannot be known yet.

        The engine estimates a compressed-tensors checkpoint (`engine.checkpoint_size.checkpoint_bytes`), which is the
        wrong artefact for an exporter that writes something else. An exporter whose layout follows from inputs
        it can read while planning returns the real size here; a caller sizing a disk budget should prefer it,
        and `Plan.output_bytes_exact` says which of the two it got.

        This is the artefact's own bytes. The build's records next to it (``build.json``, ``resolved.yaml``,
        ``recipe.yaml``) are kilobytes and are not counted, so a disk reservation wants a little headroom; a
        weight-byte budget wants exactly this number.
        """
        return None

    def weight_bytes(self, plan: Plan, recipe: Recipe) -> int | None:  # noqa: ARG002
        """The exact bytes of the weights alone inside the output, or None when it cannot be known yet.

        `output_bytes` is what the output costs on disk. This is what it costs as a model: a budget quoted
        against another toolchain's file is usually quoted on weights, since headers and vocabularies differ
        between writers. The file is always the larger of the two, so checking a weight budget against it is
        stricter than the budget and refuses outputs that fit.

        `Plan.weight_bytes` is 0 when no exporter answered. That means unknown, never that a build fits.
        """
        return None

    def builds(self, plan: Plan, recipe: Recipe) -> tuple[str, ...]:  # noqa: ARG002
        """Inputs this exporter would build itself rather than read from a path the config names.

        A search that reuses one prepared input across many candidates needs to know when a candidate would
        rebuild it instead: those rebuilds are per-candidate cost that the recipe does not otherwise show.
        """
        return ()

    def writes_alone(self, plan: Plan, recipe: Recipe) -> bool:  # noqa: ARG002
        """True when the output needs no weights loaded and no layer pass; the engine then calls only ``finish``."""
        return False

    def prepare(self, recipe: Recipe | None = None) -> str | None:  # noqa: ARG002
        """Set up process-wide state before the first layer, in every rank; return a one-line notice or None.

        `recipe` is None for a call with no recipe.
        """
        return None

    def writer(
        self,
        out_dir: str | Path,
        source: str | Path,
        graph: ModelGraph,
        fmt: Format,
        target: Target,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Any:
        """The engine's writer, writing float copies in `dtype`.

        It has ``add_layer(ctx)`` and ``finish(config, recipe, build_info) -> Path``, optionally
        ``add_model(model, qparams, changed=...)`` and ``add_source_tensors(tensors)``; without the latter, a step
        that changed tensors outside the graph is refused rather than silently overwritten from the source.
        """
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        """Facts the settings do not pin, recorded with this plugin in ``resolved.yaml`` (e.g. a tool's commit)."""
        return {}

    def layer_context(self, layer: torch.nn.Module, graph: LayerGraph) -> AbstractContextManager:  # noqa: ARG002
        """Context held from placing a layer to propagating its outputs, to keep an exporter's basis consistent."""
        return nullcontext()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """``{name: settings}``, the form a config holds (every default filled in)."""
        return {self.name: self.model_dump()}

    def solve_settings(self) -> dict[str, dict[str, Any]]:
        """Settings that change a layer solve, keying the layer cache; storage-only settings stay out."""
        return self.to_dict()


class CompressedTensors(Exporter):
    """compressed-tensors ``pack-quantized``, the checkpoint stock vLLM loads."""

    feeds_stages = True  # a checkpoint step reads this layout
    supports_formats = True  # one config group per format_overrides format

    def refusal(self, plan: Plan, recipe: Recipe) -> str | None:
        """Refuse INT8 activations, grids without integer zero points, and builds where no step stores codes.

        The last would publish a bf16 checkpoint under a quantized ``build.json``.
        """
        from ..format import FORMATS
        from ..solve.grid import DEFAULT_GRID, GRIDS
        from ..step import QuantStep

        # `ct_config` cannot write INT8 activations yet; refused here, at plan, not after the whole calibration run
        if bad := [f.name for f in (plan.fmt, *plan.formats.values()) if f.act == "int8_token"]:
            can = ", ".join(n for n, f in FORMATS.items() if f.act != "int8_token")
            return f"{bad[0]}: INT8 activations (int-quantized) cannot be written yet; this exporter writes {can}"
        for s in recipe.steps:
            if isinstance(s, QuantStep) and not GRIDS[s.grid or DEFAULT_GRID].range_rule:
                return f"{s.name}: grid {s.grid!r} has no integer zero points for compressed-tensors to store"
        if not plan.packed_source and not any(isinstance(s, QuantStep) and s.quantizes for s in recipe.steps):
            return (
                f"no step stores codes, so the checkpoint would hold no {plan.fmt.name} weight: a float "
                "build writes bf16 for every weight no step quantized; a step that only searches scales for "
                "a later one (awq) needs a quantizing step after it (e.g. rtn or gptq)"
            )
        return None

    def writer(  # noqa: ANN001, ANN201
        self, out_dir, source, graph, fmt, target, *, dtype=torch.bfloat16, formats=None
    ):
        """`qda_quant.output.export.Writer`."""
        return Writer(out_dir, source, graph, fmt, target, dtype=dtype, formats=formats)


def _adopt(key: str, cls: type[Exporter]) -> None:
    """Check an exporter plugin from either source and give it its config name."""
    if not (isinstance(cls, type) and issubclass(cls, Exporter)):
        raise TypeError(f"exporter plugin {key!r} is {cls!r}, not an Exporter subclass")
    cls.name = key


#: Exporters by config name: built-ins and package registrations (see `qda_quant.registry`).
EXPORTERS: Registry[type[Exporter]] = Registry(
    "exporters",
    {
        "awq_bf16": "qda_quant.output.awq:AWQBF16",
        "compressed_tensors": "qda_quant.output.exporters:CompressedTensors",
        "gguf": "qda_quant.gguf.export:GGUF",
    },
    on_add=_adopt,
)


def parse_exporter(d: Exporter | str | dict[str, Any] | None) -> Exporter:
    """The exporter a config names: None (the default), a name, or ``{name: settings}``.

    Raises:
        ValueError: not one name, or an unknown one (the message lists the known ones), or bad settings.
    """
    if isinstance(d, Exporter):
        return d
    if d is None:
        d = DEFAULT_EXPORTER
    if isinstance(d, str):
        d = {d: {}}
    if not isinstance(d, dict) or len(d) != 1:
        raise ValueError(
            f"exporter: expected a name or {{name: settings}}, known: {sorted(EXPORTERS)}; got {d!r}"
        )
    ((key, cfg),) = d.items()
    if key not in EXPORTERS:
        raise ValueError(f"exporter {key!r}: unknown; known: {sorted(EXPORTERS)}")
    return EXPORTERS[key](**(cfg or {}))


def check_exporter(exporter: Exporter, plan: Plan, recipe: Recipe) -> None:
    """Refuse a planned build the exporter cannot write (`qda plan` and the build both ask).

    Raises:
        Refused: `Exporter.refusal`, or the output lacks `Exporter.splits_banks`, `Exporter.supports_formats` or
            `Exporter.feeds_stages` where the plan needs it, or a checkpoint-only build names a non-default exporter.
    """
    from ..solve.units import down_key
    from ..step import Refused

    if why := exporter.refusal(plan, recipe):
        raise Refused(f"exporter {exporter.name!r}: {why}")
    banks = {k for lg in plan.graph.layers for b in lg.banks if (k := down_key(b)) is not None}
    split = sorted(banks & plan.formats.keys())
    if split and not exporter.splits_banks:
        raise Refused(
            f"exporter {exporter.name!r}: format_overrides store {split[0]} in {plan.formats[split[0]].name}, "
            f"apart from the rest of its routed bank ({len(split)} banks split); vLLM and SGLang load a bank as "
            "one fused module with one scheme. Name the whole bank, or write a GGUF, which stores the bank's "
            "projections as separate tensors"
        )
    if plan.formats and not exporter.supports_formats:
        raise Refused(
            f"exporter {exporter.name!r}: some modules are stored in another format (a format_overrides rule or "
            "a resolved expert default), and this exporter's output has no way to declare which; use the "
            "compressed_tensors exporter"
        )
    if plan.staged and not exporter.feeds_stages:
        raise Refused(
            f"exporter {exporter.name!r}: a staged build hands its intermediates on as compressed-tensors "
            "checkpoints; this exporter does not write one"
        )
    # A build with no float stage (a packed source) runs checkpoint steps only: no stage calls a writer, so only
    # the default compressed-tensors output, which those steps write, describes this build truthfully.
    checkpoint_only = plan.staged and not any(s.kind != "checkpoint" for s in recipe.steps)
    if checkpoint_only and type(exporter) is not CompressedTensors:
        raise Refused(
            f"exporter {exporter.name!r}: no float stage writes with an exporter, so its settings would be "
            "recorded and never applied; the checkpoint steps write the default compressed-tensors checkpoint"
        )


def record_exporter(exporter: Exporter, plan: Plan, recipe: Recipe) -> None:
    """Put what the exporter knows about its own output into the plan.

    The engine plans the build without knowing which exporter will write it, so its `Plan.output_bytes` sizes a
    compressed-tensors checkpoint. An exporter that writes a different artefact and can size it exactly replaces
    that number here, and says so in `Plan.output_bytes_exact`; `Plan.weight_bytes` is the same output's weights
    without its container, which is what a budget quoted against another toolchain compares; `Plan.exporter_builds`
    names the inputs it would build for itself. Callers admitting a build against a budget read these, not guesses.
    """
    if (exact := exporter.output_bytes(plan, recipe)) is not None:
        plan.output_bytes, plan.output_bytes_exact = exact, True
    if (weights := exporter.weight_bytes(plan, recipe)) is not None:
        plan.weight_bytes = weights
    plan.exporter_builds = exporter.builds(plan, recipe)
