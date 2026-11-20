"""Architecture adapters: per-layout fix-ups of the graph `derive` reads from a model's forward.

At most one adapter fixes a graph and one declares a multimodal stream (`qda_quant.arch.adapters.stream`);
a package adds one with ``ADAPTERS.register(name, obj)`` from its ``qda.plugins`` hook.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ...format import Format
from ...registry import Registry
from ...target import KernelPlan, Target
from .. import ExportLayout, ModelGraph
from .stream import SourceBlock as SourceBlock  # re-exported for declarations
from .stream import StreamDeclaration as StreamDeclaration  # re-exported for declarations


class _GraphAdapter(Protocol):
    """The required part of an architecture plugin's contract."""

    def matches(self, model: nn.Module) -> bool:
        """True if this adapter applies to `model` (a real or meta-device model)."""
        ...

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """The graph corrected for this layout: a new frozen graph (``dataclasses.replace``), not a mutation."""
        ...


@dataclass(frozen=True)
class FusedRuntime:
    """A serving engine's fused-module name and the shard names it probes for the quantization scheme.

    A ``re:`` pattern never matches the engine's literal fused name, so its ignore entry must carry the shards.

    Attributes:
        checkpoint: the checkpoint name or ``re:`` pattern the fused module stands for (its float parts).
        shards: the runtime names or ``re:`` patterns the engine probes for those parts.
    """

    checkpoint: str
    shards: tuple[str, ...]


class ArchAdapter(_GraphAdapter, Protocol):
    """A graph fix-up plus optional hooks; only ``matches`` and ``adjust`` are required, callers skip the rest."""

    def register_models(self) -> None:
        """Optional: register local HF classes explicitly before config/model loading."""
        ...

    def configure_model(self, model: nn.Module, target: Target) -> None:
        """Optional: bind target numerics before tracing or calibration; refuse unsupported targets."""
        ...

    def plugins(self, target: Target) -> list[tuple[Registry, str, object]]:
        """Optional: dependencies selected by the target, recorded in resolved provenance."""
        ...

    def layer_carry(self) -> tuple[str, ...]:
        """Optional: tuple outputs after hidden states passed as kwargs to the next layer."""
        ...

    def source_names(self, model: nn.Module) -> list:
        """Optional: transformers WeightRenaming/WeightConverter rules for the native checkpoint layout.

        They replace transformers' default mapping, so loading and export share one set of names.
        """
        ...

    def serve_plan(self, plan: KernelPlan) -> KernelPlan:
        """Optional: add architecture-specific serving metadata to the selected kernel plan."""
        ...

    def draft_ignore(self, model: nn.Module) -> tuple[str, ...]:
        """Optional: runtime module names a draft builds the model class's dropped layers under."""
        ...

    def pack_like(self, model: nn.Module) -> Mapping[str, str]:
        """Optional: checkpoint Linears the model class does not build -> the built module whose format they take.

        An engine may fuse such a weight with quantized siblings and refuses mixed schemes in one fused module.
        """
        ...

    def runtime_names(self, model: nn.Module) -> Mapping[str, Mapping[str, str | FusedRuntime]]:
        """Optional: per serving engine, runtime name -> checkpoint name, for modules kept float.

        An engine must find a float module in ``quantization_config.ignore`` under its own name, or it quantizes
        it and fails to load; declare a fused ``re:`` pattern as a `FusedRuntime`.
        """
        ...

    def awq_skip_aliases(self, names: set[str]) -> set[str]:
        """Optional: AWQ exporter's literal aliases for float modules; a fused alias needs every shard float."""
        ...

    def serve_names(self, model: nn.Module) -> Mapping[str, str]:
        """Optional: stored dotted path -> the path the serving engines build that module tree under.

        Paths match at dotted boundaries, longest first: one segment renames it everywhere (Mixtral ``mlp`` ->
        ``block_sparse_moe``), a longer path moves that subtree once; a name collision is refused.
        """
        ...

    def config_names(self, model: nn.Module) -> Mapping[str, Mapping[str, str]]:
        """Optional: per serving engine, ``serve_names``-form renames applied after it when reading the config.

        Needed where an engine loads weights under one name but looks up the scheme under another (MiniMax-M2).
        """
        ...

    def export_config(self, config: dict) -> dict:
        """Optional: serving config derived from source settings, preserving unrelated fields."""
        ...

    def check_served(self, graph: ModelGraph, fmt: Format) -> None:
        """Optional: raise ``Refused`` when the stock loader cannot read `fmt` as the exporter stores it."""
        ...

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """Optional: how the serving checkpoint stores quantized entry `name`, or None for its module name."""
        ...

    def default_ignore(self, model: nn.Module) -> tuple[str, ...]:
        """Optional: exact linear names kept float when the user omits ``ignore``; explicit lists override it."""
        ...


def _check(key: str, obj: object) -> None:
    if not all(callable(getattr(obj, name, None)) for name in ("matches", "adjust")):
        raise TypeError(
            f"arch adapter plugin {key!r} is {obj!r}, without matches(model) and adjust(model, graph)"
        )


#: Architecture adapters by name: built-ins and package registrations.
ADAPTERS: Registry[_GraphAdapter] = Registry(
    "arch_adapters",
    {
        "deepseek_v2": "qda_quant.arch.adapters.deepseek_v2:DEEPSEEK_V2",
        "deepseek_v4": "qda_quant.arch.adapters.deepseek_v4:DEEPSEEK_V4",
        "deepseek_v41": "qda_quant.arch.adapters.deepseek_v41:DEEPSEEK_V41",
        "fused_experts": "qda_quant.arch.adapters.fused_experts:FUSED_EXPERTS",
        "gemma3_vl": "qda_quant.arch.adapters.gemma3_vl:GEMMA3_VL",
        "gemma4_experts": "qda_quant.arch.adapters.gemma4:GEMMA4_EXPERTS",
        "gemma4_vl": "qda_quant.arch.adapters.gemma4_vl:GEMMA4_VL",
        "glm5_next": "qda_quant.arch.adapters.glm5_next:GLM5_NEXT",
        "gpt_oss": "qda_quant.arch.adapters.gpt_oss:GPT_OSS",
        "mistral3": "qda_quant.arch.adapters.mistral3:MISTRAL3",
        "minimax_m2": "qda_quant.arch.adapters.minimax_m2:MINIMAX_M2",
        "mixtral": "qda_quant.arch.adapters.mixtral:MIXTRAL",
        "qwen2_5_vl": "qda_quant.arch.adapters.qwen2_5_vl:QWEN2_5_VL",
        "qwen2_moe": "qda_quant.arch.adapters.qwen2_moe:QWEN2_MOE",
        "qwen3_5": "qda_quant.arch.adapters.qwen3_5:QWEN3_5",
        "qwen3_next": "qda_quant.arch.adapters.qwen3_next:QWEN3_NEXT",
        "qwen3_vl": "qda_quant.arch.adapters.qwen3_vl:QWEN3_VL",
    },
    on_add=_check,
)


def check_served(graph: ModelGraph, fmt: Format) -> None:
    """Raise `Refused` when the graph's adapter says its model cannot serve `fmt` as the exporter stores it."""
    adapter = ADAPTERS[graph.adapter] if graph.adapter else None
    check = getattr(adapter, "check_served", None)
    if check is not None:
        check(graph, fmt)


def export_layout(graph: ModelGraph, name: str, shape: torch.Size) -> ExportLayout | None:
    """How the graph's adapter stores quantized entry `name` (codes shape `shape`), or None for its module name."""
    adapter = ADAPTERS[graph.adapter] if graph.adapter else None
    layout = getattr(adapter, "export_layout", None)
    return layout(name, shape) if layout is not None else None


def register_models() -> None:
    """Let architecture plugins register local HF classes before a config is loaded."""
    for _, adapter in ADAPTERS.items():
        register = getattr(adapter, "register_models", None)
        if register is not None:
            register()


def configure_model(model: nn.Module, target: Target) -> None:
    """Let the matching architecture bind numerics from the explicit build target."""
    for _, adapter in ADAPTERS.items():
        configure = getattr(adapter, "configure_model", None)
        if configure is not None and adapter.matches(model):
            configure(model, target)


def used(graph: ModelGraph) -> tuple[str, ...]:
    """Names of the adapters `graph` came from: the one that fixed it and the one that declared its stream."""
    names = (graph.adapter, graph.extra.get("stream_adapter"))
    return tuple(dict.fromkeys(n for n in names if n))


def layer_carry(graph: ModelGraph) -> tuple[str, ...]:
    """Tuple-output names carried to the next layer as kwargs; ordinary layers carry nothing."""
    adapter = ADAPTERS[graph.adapter] if graph.adapter else None
    names = getattr(adapter, "layer_carry", None)
    return names() if names is not None else ()


def __getattr__(name: str):
    for key, obj in ADAPTERS._entries.items():
        if (obj.split(":")[1] if isinstance(obj, str) else key.upper()) == name:
            return ADAPTERS[key]
    raise AttributeError(name)
