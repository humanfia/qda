"""Mistral 3: HF config and tokenizer loading in vLLM, and SGLang's names for the bf16 Pixtral vision tower."""

from dataclasses import replace
from typing import Any, cast

import torch
from torch import nn

from ...target import KernelPlan
from .. import ModelGraph


class Mistral3:
    """Keep the derived graph, declare the serving modes, and declare the projector as a stream writer."""

    def matches(self, model: nn.Module) -> bool:
        """Match the composite transformers config, rather than its Mistral text config."""
        return getattr(getattr(model, "config", None), "model_type", None) == "mistral3"

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:  # noqa: ARG002
        """The generic graph already describes the language model."""
        return graph

    def stream(self, model: nn.Module) -> dict[str, Any]:  # noqa: ARG002
        """The projector's second Linear writes the residual stream; the tower only feeds the merger."""
        return {
            "writers": {
                "model.multi_modal_projector": ("model.multi_modal_projector.linear_2",),
                "model.vision_tower": (),
            }
        }

    def probe(self, model: nn.Module, root: str) -> dict[str, Any] | None:
        """Tower features of one image of 2 x 2 merged patches, through the projector."""
        if root != "model.multi_modal_projector":
            return None
        cfg: Any = cast(Any, model).config
        side = 2 * cfg.spatial_merge_size  # patches per image side
        width = side * cfg.vision_config.patch_size
        features = torch.randn(
            side * side,
            cfg.vision_config.hidden_size,
            generator=torch.Generator().manual_seed(0),
            dtype=torch.float64,
        )
        return {"image_features": features, "image_sizes": torch.tensor([[width, width]])}

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """SGLang 0.5.23's names for the float vision tower's Linears, which its LLaVA class matches literally.

        SGLang builds a Pixtral layer's attention as `VisionAttention` (one ``qkv_proj``, output ``proj``) and its MLP
        with one ``gate_up_proj``, and declares no fused-module mapping, so ``ignore`` must name those modules as
        built; under the checkpoint's q/k/v/o and gate/up names they are built quantized and refused. vLLM
        reads the checkpoint names, which stay in the export.
        """
        from . import FusedRuntime  # the package defines it after importing its adapters

        vision: Any = getattr(getattr(model, "config", None), "vision_config", None)
        declared: dict[str, Any] = {}
        for i in range(getattr(vision, "num_hidden_layers", 0) if vision is not None else 0):
            layer = f"vision_tower.transformer.layers.{i}"
            path = rf"re:^(?:model\.)?vision_tower\.transformer\.layers\.{i}"
            declared[f"{layer}.attention.qkv_proj"] = FusedRuntime(
                checkpoint=rf"{path}\.attention\.(q|k|v)_proj$",
                shards=tuple(f"{layer}.attention.{p}_proj" for p in "qkv"),
            )
            declared[f"{layer}.attention.proj"] = rf"{path}\.attention\.o_proj$"
            declared[f"{layer}.feed_forward.gate_up_proj"] = FusedRuntime(
                checkpoint=rf"{path}\.feed_forward\.(gate|up)_proj$",
                shards=(f"{layer}.feed_forward.gate_proj", f"{layer}.feed_forward.up_proj"),
            )
        return {"sglang": declared}

    def serve_plan(self, plan: KernelPlan) -> KernelPlan:
        """Preserve compressed-tensors config and use the checkpoint's converted HF tokenizer."""
        return replace(plan, serve_args={**plan.serve_args, "tokenizer_mode": "hf", "config_format": "hf"})


MISTRAL3 = Mistral3()
