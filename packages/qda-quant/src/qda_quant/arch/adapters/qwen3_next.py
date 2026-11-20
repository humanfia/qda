"""Qwen3-Next: stock SGLang builds its hybrid schedule from ``full_attention_interval``.

transformers 5.21's ``Qwen3NextConfig`` saves the explicit ``layer_types`` list and no
``full_attention_interval`` (it derives one only while constructing a config). Stock SGLang 0.5.23's config
class ignores ``layer_types`` and computes ``layers_block_type`` from ``full_attention_interval`` alone, so a
checkpoint exported from a 5.21 source aborts with ``AttributeError: 'Qwen3NextConfig' object has no
attribute 'full_attention_interval'``. This adapter adds the interval stock SGLang reads, derived from the
source's own schedule, and refuses a schedule one interval cannot express.
"""

from __future__ import annotations

import torch
from torch import nn

from ...step import Refused
from .. import ExportLayout, ModelGraph
from ..derive import text_config
from .fused_experts import FUSED_EXPERTS

#: a layer that runs full attention; the other kind is linear attention
FULL_ATTENTION = "full_attention"


class Qwen3Next:
    """The ``qwen3_next`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True for the Qwen3-Next text model."""
        return text_config(model).model_type == "qwen3_next"

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Refuse an inexpressible schedule at plan time, then add the fused expert banks."""
        self.export_config(text_config(model).to_dict())  # raises for a schedule stock SGLang cannot build
        return FUSED_EXPERTS.adjust(model, graph)

    def export_config(self, config: dict) -> dict:
        """Add the full-attention interval stock SGLang reads, derived from the source's ``layer_types``.

        Raises:
            Refused: the schedule has no full-attention layer, its positions are not one regular interval, or
                the source already declares an interval the schedule contradicts.
        """
        schedule = config.get("layer_types")
        if schedule is None:  # an older source declares full_attention_interval itself
            return config
        interval = _interval(list(schedule))
        declared = config.get("full_attention_interval")
        if declared is not None and declared != interval:
            raise Refused(
                f"Qwen3-Next config declares full_attention_interval={declared} but layer_types={schedule!r} "
                f"means {interval}"
            )
        return {**config, "full_attention_interval": interval}

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """The fused bank's serving layout: this adapter holds the slot FUSED_EXPERTS would fix up.

        A source that already stores per-expert weights needs no layout; a fused source becomes the
        per-expert tensors both stock engines load.
        """
        layout = getattr(FUSED_EXPERTS, "export_layout", None)
        return None if layout is None else layout(name, shape)


def _interval(schedule: list[str]) -> int:
    """The full-attention interval `schedule` names, or a refusal when one interval cannot express it."""
    positions = [i for i, kind in enumerate(schedule) if kind == FULL_ATTENTION]
    if not positions:
        raise Refused(f"Qwen3-Next stock serving requires a full-attention layer; layer_types={schedule!r}")
    interval = positions[0] + 1
    if positions != list(range(interval - 1, len(schedule), interval)):
        raise Refused(
            f"Qwen3-Next stock serving cannot express layer_types={schedule!r} as one full-attention interval"
        )
    return interval


#: The adapter object the registry holds.
QWEN3_NEXT = Qwen3Next()
