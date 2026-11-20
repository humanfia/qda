"""Qwen2.5-VL: the merger's third Sequential entry writes the residual stream.

The merger is ``mlp = Sequential(Linear, GELU, Linear)``, so its two Linears both have the residual
width in a 7B model and the width alone does not say which one writes the stream; the last one does.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .stream import StreamDeclaration, qwen_vision_probe


class Qwen2_5VL(StreamDeclaration):
    """The ``qwen2_5_vl`` declaration: ``model.visual.merger.mlp.2`` ends the vision path."""

    def matches(self, model: nn.Module) -> bool:
        """True for a Qwen2.5-VL multimodal model, not its text-only class (``qwen2_5_vl_text``)."""
        return getattr(model.config, "model_type", "") == "qwen2_5_vl"

    def stream(self, model: nn.Module) -> dict[str, Any]:  # noqa: ARG002
        """The merger's last Linear, after the activation."""
        return {"writers": {"model.visual": ("model.visual.merger.mlp.2",)}}

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, str]]:
        """Keep float text MLP shards ignored after SGLang's checkpoint-name mapper.

        SGLang rewrites plain ``model.`` ignores to ``language_model.model.`` but probes text schemes
        under ``model.``. Dotted suffixes survive that mapper and name each fused MLP shard it probes.
        The exporter emits an alias only for its float source projection, for either HF source prefix.
        """
        config: Any = getattr(model, "config", None)
        return {
            "sglang": {
                f"layers.{layer}.mlp.{projection}": (
                    rf"re:^model(?:\.language_model)?\.layers\.{layer}\.mlp\.{projection}$"
                )
                for layer in range(config.get_text_config().num_hidden_layers)
                for projection in ("gate_proj", "up_proj")
            }
        }

    def probe(self, model: nn.Module, root: str) -> dict[str, Any] | None:
        """One small image through the vision tower."""
        return qwen_vision_probe(model) if root == "model.visual" else None


QWEN2_5_VL = Qwen2_5VL()
