"""DeepSeek-V2/V3: per-expert routed weights and the serving names of the shared MLP and MLA inputs.

SGLang does not declare gate/up fusion in this architecture's packed-module mapping. It therefore probes
``shared_experts.gate_up_proj`` literally when choosing a quantization scheme, even when both checkpoint
shards are kept float. Declaring that runtime name keeps the shared MLP float without hiding packed shards.

With ``q_lora_rank`` unset the checkpoint stores ``q_proj`` instead of ``q_a_proj``/``q_b_proj``, and vLLM
builds ``q_proj`` and ``kv_a_proj_with_mqa`` as two linears although they read one input; declaring each on
its own shard keeps a kernel's width rule on the linear the engine serves, not on their sum.

SGLang's MLA postprocessing reads ``kv_b_proj.weight`` directly and cannot unpack compressed-tensors INT4;
keep that projection float by default, with the exact names recorded in the resolved config.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from torch import nn

from ..derive import text_config
from .fused_experts import FusedExperts


def unfused_mla(model: nn.Module) -> dict[str, Any]:
    """The names vLLM builds MLA's input projections under when ``q_lora_rank`` is unset.

    The checkpoint then stores ``q_proj`` where the compressed path stores ``q_a_proj``/``q_b_proj``, and
    vLLM builds ``q_proj`` and ``kv_a_proj_with_mqa`` as two linears although QDA's input-sharing group holds
    both. One `FusedRuntime` shard names each runtime linear, so a kernel's width rule applies to the linear
    the engine serves instead of to their sum.
    """
    from . import FusedRuntime

    modules = dict(model.named_modules())
    out: dict[str, Any] = {}
    for name, module in modules.items():
        root, _, leaf = name.rpartition(".")
        if leaf != "q_proj" or not isinstance(module, nn.Linear):
            continue
        if root.rpartition(".")[2] != "self_attn":
            continue
        for part in ("q_proj", "kv_a_proj_with_mqa"):
            shard = f"{root}.{part}"
            if isinstance(modules.get(shard), nn.Linear):
                out[shard] = FusedRuntime(shard, (shard,))
    return out


class DeepseekV2(FusedExperts):
    """The DeepSeek-V2/V3 layout; routed banks use the standard per-expert export."""

    def matches(self, model: nn.Module) -> bool:
        """True for stock HF DeepSeek-V2 and DeepSeek-V3 causal models."""
        return text_config(model).model_type in ("deepseek_v2", "deepseek_v3")

    def runtime_names(self, model: nn.Module) -> Mapping[str, Mapping[str, Any]]:
        """SGLang's literal shared gate/up module, and vLLM's names for the MLA projections it does not fuse.

        ``q_proj`` exists only when ``q_lora_rank`` is unset; with it set vLLM fuses ``q_a_proj`` and
        ``kv_a_proj_with_mqa`` into ``fused_qkv_a_proj`` and neither is declared on its own. The shared gate/up
        entry is conditional on both stored shards staying float.
        """
        out: dict[str, dict[str, Any]] = {
            "sglang": {
                f"{name}.gate_up_proj": f"re:^{re.escape(name)}\\.(gate|up)_proj$"
                for name, module in model.named_modules()
                if name.endswith(".shared_experts")
                and isinstance(getattr(module, "gate_proj", None), nn.Linear)
            }
        }
        unfused = unfused_mla(model)
        if unfused:
            out["vllm"] = unfused
        return out

    def default_ignore(self, model: nn.Module) -> tuple[str, ...]:
        """MLA KV expansion weights SGLang reads as float; an explicit ``ignore`` list can quantize them."""
        return tuple(
            name
            for name, module in model.named_modules()
            if name.endswith(".self_attn.kv_b_proj") and isinstance(module, nn.Linear)
        )


DEEPSEEK_V2 = DeepseekV2()
