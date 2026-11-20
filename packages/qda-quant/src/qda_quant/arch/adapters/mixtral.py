"""Mixtral: stock engines build the sparse block as ``block_sparse_moe`` with experts ``w1``/``w2``/``w3``.

transformers 5 names the sparse block ``mlp`` and its experts ``gate_proj``/``up_proj``/``down_proj`` (fused
into ``gate_up_proj``/``down_proj`` in the model); its conversion rules rename the old checkpoint's
``block_sparse_moe`` on the way in. Stock vLLM 0.33 and SGLang 0.5.23 keep the old names, so a checkpoint
written from a transformers-5 source under the source's own names is refused by vLLM before any expert is
loaded and silently skipped by SGLang. The adapter declares the rename; the exporter writes the served names.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .. import ExportLayout, ModelGraph
from ..derive import text_config
from .fused_experts import FUSED_EXPERTS

#: source name segment -> the segment stock vLLM and SGLang build that module tree under
SERVING_NAMES = {
    "mlp": "block_sparse_moe",
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


class Mixtral:
    """The ``mixtral`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True for the Mixtral text model."""
        return text_config(model).model_type == "mixtral"

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Mixtral's sparse block is a fused expert bank, as for every fused layout."""
        return FUSED_EXPERTS.adjust(model, graph)

    def serve_names(self, model: nn.Module) -> Mapping[str, str]:  # noqa: ARG002
        """The sparse block's serving names; a source already named that way keeps them (the map does not match)."""
        return SERVING_NAMES

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """The fused bank's serving layout: this adapter holds the slot FUSED_EXPERTS would fix up.

        Delegates to `FUSED_EXPERTS` when it declares a layout, so a fused source becomes the per-expert
        tensors the stock engines probe. Until it does, the bank keeps its fused names.
        """
        layout = getattr(FUSED_EXPERTS, "export_layout", None)
        return None if layout is None else layout(name, shape)


#: The adapter object the registry holds.
MIXTRAL = Mixtral()
