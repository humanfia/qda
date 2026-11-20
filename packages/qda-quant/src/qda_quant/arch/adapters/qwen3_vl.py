"""Qwen3-VL: the vision merger's last Linear writes the residual stream."""

from __future__ import annotations

from typing import Any

from torch import nn

from .stream import StreamDeclaration, qwen_vision_probe


class Qwen3VL(StreamDeclaration):
    """The ``qwen3_vl`` declaration: ``model.visual.merger.linear_fc2`` ends the vision path."""

    def matches(self, model: nn.Module) -> bool:
        """True for a Qwen3-VL multimodal model, not its text-only class (``qwen3_vl_text``)."""
        return getattr(model.config, "model_type", "") == "qwen3_vl"

    def stream(self, model: nn.Module) -> dict[str, Any]:
        """The merger's output Linear and every DeepStack merger's, through which the tower reaches the stream.

        DeepStack adds one merger's output to an early decoder layer's hidden states (`deepstack_visual_indexes`
        in the config), so those mergers write the residual stream too.
        """
        deep = [
            n for n, _ in model.named_modules() if n.endswith(".linear_fc2") and "deepstack_merger_list" in n
        ]
        return {"writers": {"model.visual": ("model.visual.merger.linear_fc2", *deep)}}

    def probe(self, model: nn.Module, root: str) -> dict[str, Any] | None:
        """One small image through the vision tower."""
        return qwen_vision_probe(model) if root == "model.visual" else None


QWEN3_VL = Qwen3VL()
