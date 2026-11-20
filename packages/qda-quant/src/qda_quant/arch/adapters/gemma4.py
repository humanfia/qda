"""Gemma 4 experts have a nested router and a separate normalized input."""

from dataclasses import replace

import torch
from torch import nn

from .. import ExpertBank, ExportLayout, ModelGraph
from ..derive import routes_itself, text_config
from .fused_experts import per_expert_layout


class Gemma4Experts:
    """Describe Gemma 4's routed branch independently of its dense MLP."""

    def matches(self, model: nn.Module) -> bool:
        """Match text models with Gemma 4's optional expert branch enabled."""
        config = text_config(model)
        return config.model_type == "gemma4_text" and config.enable_moe_block

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Add banks and their actual input norm without folding the router into it."""
        readers = dict(graph.extra["readers"])
        layers = []
        for layer in graph.layers:
            prefix = layer.module
            router = model.get_submodule(f"{prefix}.router")
            bank = ExpertBank(
                f"{prefix}.experts",
                f"{prefix}.router",
                text_config(model).top_k_experts,
                fused=True,
                custom_router=routes_itself(router),
            )
            norm = f"{prefix}.pre_feedforward_layernorm_2"
            readers[norm] = (*readers.get(norm, ()), bank.experts)
            layers.append(replace(layer, banks=(*layer.banks, bank)))
        return replace(graph, layers=tuple(layers), extra={**graph.extra, "readers": readers})

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """Declare the shared per-expert serving names for Gemma 4's routed banks.

        Gemma 4 stores the canonic fused ``gate_up_proj`` halves and ``down_proj``, so the split itself
        is model-independent; only this adapter's registration is Gemma-specific (the generic fused-expert
        fallback does not claim a model another adapter matches).
        """
        return per_expert_layout(name, shape)


GEMMA4_EXPERTS = Gemma4Experts()
