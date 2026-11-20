"""Qwen2-MoE: SGLang builds its shared expert's gate/up as one fused, quantized module.

The checkpoint keeps the shared expert float by default (the ``shared_experts`` role), under the two stored
shards ``shared_expert.gate_proj`` / ``shared_expert.up_proj``. SGLang's ``Qwen2MoeMLP`` does not declare
gate/up fusion for this architecture, so it probes the literal ``shared_expert.gate_up_proj`` when choosing a
quantization scheme: without an ``ignore`` entry for that name it builds the merged projection quantized, drops
both float shards and never writes the packed parameters. The alias below names that runtime module for every
shared expert whose shards stay float, exactly as DeepSeek-V2's adapter does for its ``shared_experts``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from torch import nn

from ..derive import text_config
from .fused_experts import FusedExperts


class Qwen2Moe(FusedExperts):
    """Qwen2-MoE's routed banks plus the shared expert SGLang builds as one fused module."""

    def matches(self, model: nn.Module) -> bool:
        """True for the Qwen2-MoE source model."""
        return text_config(model).model_type == "qwen2_moe"

    def runtime_names(self, model: nn.Module) -> Mapping[str, Mapping[str, str]]:
        """SGLang's literal shared gate/up module, conditional on both stored shards staying float.

        The exporter omits the entry when either shard is quantized, so an explicit ``ignore`` that packs the
        shared expert never leaves the loader looking for float weights the checkpoint does not store.
        """
        return {
            "sglang": {
                f"{name}.gate_up_proj": f"re:^{re.escape(name)}\\.(gate|up)_proj$"
                for name, module in model.named_modules()
                if name.endswith(".mlp.shared_expert")
                and isinstance(getattr(module, "gate_proj", None), nn.Linear)
            }
        }


QWEN2_MOE = Qwen2Moe()
