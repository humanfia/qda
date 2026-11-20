"""Qwen3.5: the vision merger writes the residual stream; the MTP head is a declared source block."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from torch import nn

from .stream import SourceBlock, StreamDeclaration, qwen_vision_probe

#: The MTP head. Stock classes load none of it, so its tensors are rewritten in the source checkpoint.
_MTP = SourceBlock(
    fc="mtp.fc",
    inputs=("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"),
    folds=(
        (
            "mtp.layers.0.input_layernorm",
            (
                "mtp.layers.0.self_attn.q_proj",
                "mtp.layers.0.self_attn.k_proj",
                "mtp.layers.0.self_attn.v_proj",
            ),
        ),
        (
            "mtp.layers.0.post_attention_layernorm",
            ("mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj"),
        ),
    ),
    writers=("mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.down_proj"),
    norm="mtp.norm",
    kept=("mtp.layers.0.self_attn.q_norm", "mtp.layers.0.self_attn.k_norm"),
    offset=1.0,
)

_MTP_MOE = replace(
    _MTP,
    folds=(
        _MTP.folds[0],
        (
            "mtp.layers.0.post_attention_layernorm",
            (
                "mtp.layers.0.mlp.gate",
                "mtp.layers.0.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.shared_expert.gate_proj",
                "mtp.layers.0.mlp.shared_expert.up_proj",
                "mtp.layers.0.mlp.shared_expert_gate",
            ),
        ),
    ),
    writers=(
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.shared_expert.down_proj",
    ),
    fused=("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"),
    approximation=(
        "MoE MTP is exact only up to its last norm: it feeds the target's shared head, which the rotation "
        "stores as (head diag(g_f)) Q, and a Hadamard turns every per-channel gain c into the scalar "
        "mean(c), so the draft's gain is forced to be proportional to g_f while the source's is g_m; the "
        "two are equal only if g_m/g_f is constant. The rewrite stores the best constant fit (residue in "
        "build.json). The served logits are unchanged -- the draft only proposes tokens"
    ),
)


class Qwen3_5(StreamDeclaration):
    """The ``qwen3_5`` declaration: the merger's Linear, and the MTP head the class drops."""

    def matches(self, model: nn.Module) -> bool:
        """True for a Qwen3.5/3.6/3.8 model, dense or MoE, conditional or text-only."""
        return getattr(model.config, "model_type", "") in (
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        )

    def stream(self, model: nn.Module) -> dict[str, Any]:
        """The merger's output Linear, and the ``mtp`` block the checkpoint may hold."""
        cfg = getattr(model.config, "text_config", None) or model.config
        return {
            "writers": {"model.visual": ("model.visual.merger.linear_fc2",)},
            "blocks": {"mtp": _MTP_MOE if getattr(cfg, "num_experts", 0) else _MTP},
        }

    def probe(self, model: nn.Module, root: str) -> dict[str, Any] | None:
        """One small image through the vision tower."""
        return qwen_vision_probe(model) if root == "model.visual" else None

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """Fused attention inputs need literal override targets before SGLang's Linear fallback.

        Both engines fuse the GDN qkv/z inputs, the GDN b/a gates, full-attention q/k/v and every dense MLP's
        gate/up (the shared expert's too). Declare each actual module separately so the exporter can refuse
        a fused input whose shards take different schemes, and names the fused module in an override group
        (an override on gate/up that omits ``gate_up_proj`` loads as the default group in SGLang).
        """
        from . import FusedRuntime

        out: dict[str, dict[str, Any]] = {"sglang": {}, "vllm": {}}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            root, _, leaf = name.rpartition(".")
            if leaf == "in_proj_qkv":
                fused, parts = "in_proj_qkvz", ("in_proj_qkv", "in_proj_z")
            elif leaf == "q_proj":
                fused, parts = "qkv_proj", ("q_proj", "k_proj", "v_proj")
            elif leaf == "in_proj_b":
                fused, parts = "in_proj_ba", ("in_proj_b", "in_proj_a")
            elif leaf == "gate_proj" and ".experts." not in name:
                fused, parts = "gate_up_proj", ("gate_proj", "up_proj")
            else:
                continue
            # Text-only HF classes unwrap the public checkpoint's language_model root; vLLM
            # moves it to language_model.model. All forms describe the same actual layer.
            text = root.removeprefix("model.language_model.").removeprefix("model.")
            roots = (root, f"model.{text}", f"model.language_model.{text}")
            checkpoint = rf"re:^(?:{'|'.join(re.escape(r) for r in roots)})\.(?:{'|'.join(parts)})$"
            for engine, runtimes in (("sglang", roots), ("vllm", (*roots, f"language_model.model.{text}"))):
                for runtime in runtimes:
                    out[engine][f"{runtime}.{fused}"] = FusedRuntime(
                        checkpoint, tuple(f"{runtime}.{part}" for part in parts)
                    )
        return out

    @staticmethod
    def awq_skip_aliases(names: set[str]) -> set[str]:
        """AWQ's literal skip names for Qwen's fused projections and wrapped engine prefixes.

        Stock SGLang builds QKV outside self_attn and drops the conditional model's language_model
        prefix. CT regex ignores cannot be copied into its substring-based AWQ matcher.
        Only groups whose every native part is float get a fused alias.
        """
        from .fused_experts import FusedExperts

        aliases = FusedExperts.awq_skip_aliases(names)
        for name in names:
            if name.endswith(".in_proj_a"):
                base = name.removesuffix(".in_proj_a")
                if base + ".in_proj_b" in names:
                    aliases.add(base + ".in_proj_ba")
            if name.endswith(".in_proj_qkv"):
                base = name.removesuffix(".in_proj_qkv")
                if base + ".in_proj_z" in names:
                    aliases.add(base + ".in_proj_qkvz")
        aliases.update(n.replace(".self_attn.qkv_proj", ".qkv_proj") for n in list(aliases))
        for name in names | aliases:
            aliases.add(name.replace("model.language_model.", "model."))
            aliases.add(name.replace("model.language_model.", "language_model.model."))
        return aliases


QWEN3_5 = Qwen3_5()
