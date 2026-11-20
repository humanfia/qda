"""Gemma 3 multimodal: the SigLIP tower stored as both engines build it, and SGLang's names for its float Linears.

transformers 5 holds the tower without its ``vision_model`` level and saves it that way (``vision_tower.encoder``);
the vendor checkpoint, vLLM 0.33 and SGLang 0.5.23 all name it ``vision_tower.vision_model.encoder``. Exported under
the transformers 5 names, neither engine finds the float tower in ``ignore`` (it is built quantized and refused) nor
loads its weights, so the adapter declares the move; a source already in the vendor layout keeps its names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .. import ModelGraph

#: the transformers 5 tower's top-level parts -> where the vendor checkpoint and both engines keep them
SERVING_NAMES = {
    f"vision_tower.{part}": f"vision_tower.vision_model.{part}"
    for part in ("embeddings", "encoder", "post_layernorm", "head")
}


class Gemma3VL:
    """The ``gemma3`` adapter: the generic graph, plus SGLang's runtime names for the float vision tower."""

    def matches(self, model: nn.Module) -> bool:
        """True for a Gemma 3 multimodal model, not its text-only class (``gemma3_text``)."""
        return getattr(getattr(model, "config", None), "model_type", None) == "gemma3"

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:  # noqa: ARG002
        """The generic graph already describes the language model."""
        return graph

    def serve_names(self, model: nn.Module) -> Mapping[str, str]:  # noqa: ARG002
        """The vendor layout of the tower; a vendor-layout source does not match and keeps its names."""
        return SERVING_NAMES

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """SGLang 0.5.23's names for the float SigLIP layers' Linears.

        SGLang builds each layer's attention as `VisionAttention`: a fused ``qkv_proj`` and an output ``proj``.
        The checkpoint's ``out_proj`` never matches ``proj``, so that Linear is built quantized and the start is
        refused. vLLM matches the stored (vendor-layout) names, which stay in the export.
        """
        from . import FusedRuntime  # the package defines it after importing its adapters

        vision: Any = getattr(getattr(model, "config", None), "vision_config", None)
        declared: dict[str, Any] = {}
        for i in range(getattr(vision, "num_hidden_layers", 0) if vision is not None else 0):
            layer = f"vision_tower.vision_model.encoder.layers.{i}"
            path = rf"re:^(?:model\.)?vision_tower\.(?:vision_model\.)?encoder\.layers\.{i}"
            declared[f"{layer}.self_attn.qkv_proj"] = FusedRuntime(
                checkpoint=rf"{path}\.self_attn\.(q|k|v)_proj$",
                shards=tuple(f"{layer}.self_attn.{p}_proj" for p in "qkv"),
            )
            declared[f"{layer}.self_attn.proj"] = rf"{path}\.self_attn\.out_proj$"
        return {"sglang": declared}


GEMMA3_VL = Gemma3VL()
