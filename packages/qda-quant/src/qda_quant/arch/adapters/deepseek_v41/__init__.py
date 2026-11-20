"""DeepSeek-V4.1 architecture plugin and explicit HF model registration."""

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn

from ...derive import text_config
from .. import ExportLayout
from ..deepseek_v4 import COMPRESSOR, GROUPED_OUTPUT, HYPER_CONNECTIONS, INDEXER, _groups
from ..fused_experts import FUSED_EXPERTS, per_expert_layout

#: stored name (transformers-5 shadow root) -> the name stock vLLM 0.33 and SGLang 0.5.23 build that module
#: under. Both load native DeepSeek-V4.1 checkpoints: SGLang remaps ``layers.``/``head.``/``embed.`` and
#: ``w1``/``w2``/``w3`` itself, vLLM's mappers translate the same names to its runtime modules.
SERVING_NAMES = {
    "model.embed_tokens": "embed",
    "lm_head": "head",
    "model.layers": "layers",
    "model.norm": "norm",
    "engram.embed_tokens.weight_scale": "engram.embed.scale",
    "engram.embed_tokens": "engram.embed",
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def _plain(rope: dict) -> bool:
    """True if `rope` selects the source's unscaled RoPE."""
    return rope.get("rope_type", rope.get("type")) in (None, "default")


def _serving_rope(rope: dict, max_position: int) -> dict:
    """`rope` with the length field SGLang's compressed-KV layers read.

    SGLang 0.5.23 builds its compressed layers as ``deepseek_yarn`` and reads
    ``rope_parameters["original_max_position_embeddings"]`` as the length the correction formula takes a
    logarithm of, so 0 is a math domain error. A plain source gets its own ``max_position_embeddings``,
    which with the factor 1.0 default reduces YaRN to the unscaled frequencies; vLLM 0.33 picks the same
    value on its own unscaled path. A scaled source keeps its value.
    """
    if "rope_type" in rope or "type" in rope:
        if _plain(rope) and "original_max_position_embeddings" not in rope:
            return {**rope, "original_max_position_embeddings": max_position}
        return rope
    return {
        key: _serving_rope(value, max_position) if isinstance(value, dict) else value
        for key, value in rope.items()
    }


def _float_aliases(name: str) -> tuple[str, ...]:
    """Every serving engine's runtime name for the float module stored at `name`.

    A shared expert's ``w1``/``w3`` pair is one stacked ``gate_up_proj`` and its ``w2`` is
    ``down_proj``. vLLM 0.33 builds the text model under ``language_model.``; SGLang 0.5.23 remaps
    ``.attn.``/``.ffn.`` to ``.self_attn.``/``.mlp.`` and matches an ``ignore`` entry literally, so
    each engine needs its own name for a module it must keep float.
    """
    stacked = (
        name.replace(".shared_experts.w1", ".shared_experts.gate_up_proj")
        .replace(".shared_experts.w3", ".shared_experts.gate_up_proj")
        .replace(".shared_experts.w2", ".shared_experts.down_proj")
    )
    sglang = stacked.replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
    return tuple(dict.fromkeys((stacked, f"language_model.{stacked}", sglang)))


class DeepseekV41:
    """CSA2/Engram adapter with fused routed experts and explicit layer carry."""

    def register_models(self):
        """Register config/model classes when QDA loads architecture plugins."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping

        from .config import DeepseekV41Config, DeepseekV41TextConfig
        from .model import DeepseekV41ForCausalLM

        AutoConfig.register(DeepseekV41Config.model_type, DeepseekV41Config, exist_ok=True)
        AutoConfig.register(DeepseekV41TextConfig.model_type, DeepseekV41TextConfig, exist_ok=True)
        AutoModelForCausalLM.register(DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True)
        register_checkpoint_conversion_mapping(
            DeepseekV41ForCausalLM.__name__, self.source_names(None), overwrite=True
        )

    def source_names(self, model) -> list:  # noqa: ARG002
        """Native text/Engram names, shared by HF loading, disk loading and export."""
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
            WeightRenaming,
        )

        renamings = [
            WeightRenaming(source, target)
            for source, target in (
                (r"^embed\.", "model.embed_tokens."),
                (r"^head\.", "lm_head."),
                (r"^layers\.", "model.layers."),
                (r"^norm\.", "model.norm."),
                (r"\.engram\.embed\.", ".engram.embed_tokens."),
                (r"\.engram\.embed_tokens\.scale", ".engram.embed_tokens.weight_scale"),
            )
        ]
        return [
            *renamings,
            WeightConverter(
                [r"\.experts.*.w1.weight", r"\.experts.*.w3.weight"],
                r"\.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                r"\.experts.*.w2.weight",
                r"\.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ]

    def matches(self, model):
        """True for the V4.1 text backbone."""
        return text_config(model).model_type == "deepseek_v41_text"

    def layer_carry(self):
        """Name tuple outputs after hidden states, and their next-layer keyword arguments."""
        return ("carry",)

    def serve_names(self, model) -> Mapping[str, str]:  # noqa: ARG002
        """The native names both stock engines load; a native-layout source already has them."""
        return SERVING_NAMES

    def runtime_names(self, model) -> Mapping[str, Mapping[str, str]]:
        """Both engines' fused name for the checkpoint's float ``wq_a`` / ``wkv`` pair.

        SGLang 0.5.23 fuses the pair by default (``SGLANG_OPT_FUSE_WQA_WKV``) into ``self_attn.wqkv_a`` and probes
        only ``.weight``, ``.weight_scale_inv``, ``.qweight`` and ``.qweight_type`` for its inputs, so the adapter keeps
        the pair float. vLLM 0.33 fuses it into ``attn.fused_wqa_wkv``; its multimodal V4.1 class (the one a native
        checkpoint loads) declares no fused-module mapping and no config name mapper, so it looks the fused module up
        literally under ``language_model.model.``, and the text class under ``model.``. Each needs that literal name.
        """
        pair = r"re:(?:model\.)?layers\.\d+\.attn\.(?:wq_a|wkv)$"
        layers = range(text_config(model).num_hidden_layers)
        return {
            "sglang": {f"model.layers.{i}.self_attn.wqkv_a": pair for i in layers},
            "vllm": {
                f"{root}layers.{i}.attn.fused_wqa_wkv": pair
                for root in ("language_model.model.", "model.")
                for i in layers
            },
        }

    def configure_model(self, model, target):
        """Select prefill numerics from the build target before any trace or calibration."""
        from .prefill import prefill_for

        name, prefill = prefill_for(target)
        model.prefill_numerics = name
        for layer in model.model.layers:
            layer.attn.prefill = prefill
            if layer.attn.indexer is not None:
                layer.attn.indexer.prefill = prefill

    def plugins(self, target):
        """Report the target-derived numerical plugin for resolved.yaml provenance."""
        from ..prefill import PREFILL_NUMERICS
        from .prefill import prefill_for

        name, obj = prefill_for(target)
        return [(PREFILL_NUMERICS, name, obj)]

    def draft_ignore(self, model):
        """Runtime names of float DSpark projections stored under the checkpoint's ``mtp.*`` names."""
        cfg = text_config(model)
        count = getattr(cfg, "num_nextn_predict_layers", 0)
        if not count:
            return ()
        projections = (
            "attn.fused_wqa_wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
            "ffn.shared_experts.gate_up_proj",
            "ffn.shared_experts.down_proj",
        )
        return (
            "model.main_proj",
            *(
                name
                for i in range(cfg.num_hidden_layers, cfg.num_hidden_layers + count)
                for name in (
                    *(f"model.layers.{i}.{suffix}" for suffix in projections),
                    rf"re:model\.layers\.{i}\.ffn\.experts\.\d+\.w[123]",
                )
            ),
        )

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """A directly stored fused routed bank becomes the per-expert ``w1``/``w3``/``w2`` both engines probe."""
        return per_expert_layout(name, shape)

    def export_config(self, config: dict) -> dict:
        """Give SGLang's compressed layers the RoPE length field it reads; other source settings stand."""
        text = config.get("text_config")
        rope = text.get("rope_parameters") if isinstance(text, dict) else None
        max_position = text.get("max_position_embeddings") if isinstance(text, dict) else None
        if not isinstance(text, dict) or not isinstance(rope, dict) or not isinstance(max_position, int):
            return config
        return {
            **config,
            "text_config": {**text, "rope_parameters": _serving_rope(rope, max_position)},
        }

    def adjust(self, model, graph):
        """Keep compressors, indexers, grouped output, Engram projections and the attention inputs float.

         The attention's ``wq_a`` / ``wkv`` stay float because stock SGLang 0.5.23 fuses them into ``wqkv_a`` by
         default and its loader skips packed inputs there without an error, leaving the fused weight unwritten
        ; `runtime_names` and the serve aliases name them for both engines.
        """
        from .attention import GroupedLinear
        from .engram import EngramEmbedding

        graph = FUSED_EXPERTS.adjust(model, graph)
        kept = {}
        for name, module in model.named_modules():
            role = None
            if isinstance(module, nn.Linear):
                if ".compressor." in name:
                    role = COMPRESSOR
                elif ".indexer." in name:
                    role = INDEXER
                elif ".engram." in name:
                    role = "engram"
                elif isinstance(module, GroupedLinear):
                    role = GROUPED_OUTPUT
                elif name.endswith((".attn.wq_a", ".attn.wkv")):
                    role = "attention_input"
            if role:
                kept.setdefault(role, []).append(name)
        for name, _ in model.named_parameters():
            if name.rsplit(".", 1)[-1].startswith("hc_"):
                kept.setdefault(HYPER_CONNECTIONS, []).append(name)
        kept = {k: tuple(v) for k, v in kept.items()}
        readers = dict(graph.extra["readers"])
        layers = []
        for lg in graph.layers:
            attention = model.get_submodule(lg.module).attn
            inputs = [f"{lg.module}.attn.wq_a", f"{lg.module}.attn.wkv"]
            if attention.compressor is not None:
                inputs.append(f"{lg.module}.attn.compressor.wkv")
                if attention.compressor.ratio > 1:
                    inputs.append(f"{lg.module}.attn.compressor.wgate")
            if attention.indexer is not None:
                inputs.append(f"{lg.module}.attn.indexer.weights_proj")
            readers[f"{lg.module}.attn.wq_a"] = tuple(inputs)
            readers[f"{lg.module}.attn_norm"] = tuple(inputs)
            shared = tuple(
                n
                for n, m in model.get_submodule(lg.module).named_modules(prefix=lg.module)
                if ".ffn.shared_experts." in n and isinstance(m, nn.Linear)
            )
            layers.append(
                replace(
                    lg,
                    groups=_groups(lg, kept, readers),
                    banks=tuple(replace(b, shared=shared) for b in lg.banks),
                )
            )
        preserved = tuple(
            f"{name}.{leaf}"
            for name, module in model.named_modules()
            if isinstance(module, EngramEmbedding)
            for leaf in ("weight", "weight_scale")
        )
        float_modules = (
            *(name for role, names in kept.items() if role != HYPER_CONNECTIONS for name in names),
            *(name for layer in layers for bank in layer.banks for name in bank.shared),
        )
        serve_ignore = {name: _float_aliases(name) for name in float_modules}
        return replace(
            graph,
            layers=tuple(layers),
            extra={
                **graph.extra,
                "readers": readers,
                "kept_bf16": kept,
                "preserved_tensors": preserved,
                "host_tensors": preserved,
                "draft_ignore": self.draft_ignore(model),
                "serve_ignore": serve_ignore,
            },
        )


DEEPSEEK_V41 = DeepseekV41()
