"""DeepSeek-V4: fused experts with a shared MLP, plus the weights a quantizer must leave in bf16.

Generic derivation sees the routed experts of ``DeepseekV4Experts`` as a stack it cannot open (the
``fused_experts`` bank), and it reads the decoder block's `HyperConnection` projections as no linear at all.
It does put the four linear kinds that must stay float into the groups beside the attention projections they
share an input with, so this adapter takes them out and lists them in ``kept_bf16``:

* the grouped output projection ``o_a_proj`` (one block-diagonal ``DeepseekV4GroupedLinear``, not a plain
  matrix: its rows are per group, so one GPTQ Hessian does not describe it);
* the CSA / HCA compressors' ``kv_proj`` / ``gate_proj`` and their ``position_bias``;
* the lightning indexer (its own compressor, query projection and scorer) inside the CSA compressor;
* the hyper-connection ``fn`` / ``base`` / ``scale`` projections of each block and of ``hc_head``;
* the attention's primary inputs ``wq_a`` / ``wkv`` (``q_a_proj`` / ``kv_proj`` in memory): stock SGLang 0.5.23
  fuses them into ``wqkv_a`` by default and its loader accepts no compressed-tensors packed field there, so
  packed inputs are skipped without an error and the fused weight stays unwritten. Kept float, both
  engines load them under the fused names `runtime_names` declares (vLLM ``fused_wqa_wkv``, SGLang ``wqkv_a``).

The routed experts carry a shared expert (a dense MLP on every token); `derive` finds it in the traced forward
like any other bank's shared expert, so this adapter only handles the fused bank and the float-only weights.

The model class drops the ``num_nextn_predict_layers`` MTP layers, but a vLLM spec-decode draft reloads them
from the same checkpoint. That draft builds them under its own runtime names (``model.layers.{n+i}``, one
block per MTP layer), so the compressed-tensors ``ignore`` must name those; `draft_ignore` declares them.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import cast

from torch import Size, nn

from ...format import Format
from ...step import Refused
from .. import ExportLayout, LayerGraph, LinearGroup, ModelGraph
from ..derive import served_name, text_config
from .fused_experts import FUSED_EXPERTS, per_expert_layout

#: Roles of weights that always stay bf16. They are not `qda_quant.arch.KEEP_ROLES`: no build may quantize
#: them, so the config's ``ignore`` neither lists nor changes them.
COMPRESSOR = "compressor"
INDEXER = "indexer"
HYPER_CONNECTIONS = "hyper_connections"
GROUPED_OUTPUT = "grouped_output"
ATTENTION_INPUT = "attention_input"

_NORM_ALIAS = (r"\.attn\.norm\.", ".self_attn.kv_norm.")
_EXPERT_PROJECTIONS = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))
_ATTENTION_INPUTS = (("wq_a", "q_a_proj"), ("wkv", "kv_proj"))


class DeepseekV4:
    """The ``deepseek_v4`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True for a DeepSeek-V4 causal LM."""
        return text_config(model).model_type == "deepseek_v4"

    def source_names(self, model: nn.Module) -> list:  # noqa: ARG002
        """Loader transforms for DeepSeek's native checkpoint names, shared by load and export.

        Native experts are per-expert w1/w3/w2; transformers holds fused gate/up and down parameters.
        The generic source map inverts these same transforms when exporting the solved bank.
        """
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
            WeightRenaming,
        )

        pairs = [
            (r"^embed\.", "model.embed_tokens."),
            (r"^head\.", "lm_head."),
            (r"^layers\.", "model.layers."),
            (r"^norm\.", "model.norm."),
            (r"^mtp\.", "model.mtp."),
            # transformers saves a converted attention norm back as ``attn.norm``; the checkpoint says kv_norm
            _NORM_ALIAS,
            (r"\.attn\.", ".self_attn."),
            (r"\.ffn\.", ".mlp."),
            (r"\.indexer\.compressor\.", ".compressor.indexer."),
            (r"\.indexer\.weights_proj\.", ".compressor.indexer.scorer.weights_proj."),
            (r"\.indexer\.wq_b\.", ".compressor.indexer.q_b_proj."),
            (r"\.attn_norm\.", ".input_layernorm."),
            (r"\.ffn_norm\.", ".post_attention_layernorm."),
            (r"\.q_norm\.", ".q_a_norm."),
            (r"\.compressor\.norm\.", ".compressor.kv_norm."),
            (r"\.indexer\.norm\.", ".indexer.kv_norm."),
            (r"\.ape$", ".position_bias"),
            (r"\.attn_sink$", ".sinks"),
            (r"\.gate\.bias$", ".gate.e_score_correction_bias"),
        ]
        pairs.extend(
            (rf"\.hc_{block}_{leaf}$", f".{block}_hc.{leaf}")
            for block in ("attn", "ffn")
            for leaf in ("fn", "base", "scale")
        )
        pairs.extend((rf"^hc_head_{leaf}$", f"model.hc_head.hc_{leaf}") for leaf in ("fn", "base", "scale"))
        pairs.extend(
            (rf"\.{native}\.", f".{memory}.")
            for native, memory in (
                *_ATTENTION_INPUTS,
                ("wq_b", "q_b_proj"),
                ("wgate", "gate_proj"),
                ("wo_a", "o_a_proj"),
                ("wo_b", "o_b_proj"),
            )
        )
        pairs.extend(
            (rf"\.shared_experts\.{native}\.", f".shared_experts.{memory}.")
            for native, memory in _EXPERT_PROJECTIONS
        )
        return [
            *(WeightRenaming(source, target) for source, target in pairs),
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

    def serve_names(self, model: nn.Module) -> dict[str, str]:
        """HF module trees as native DeepSeek checkpoint names, shared by both stock loaders.

        Specific inverse rules precede generic projections: the indexer's query lives beside its
        compressor in the native layout. Already-native source names match none of these full paths.
        """
        from transformers.core_model_loading import WeightRenaming, apply_renamings

        rules = [
            cast(WeightRenaming, rule.reverse_transform())
            for rule in self.source_names(model)
            if isinstance(rule, WeightRenaming) and rule.source_patterns != [_NORM_ALIAS[0]]
        ]
        rules.sort(key=lambda rule: len(rule.source_patterns[0]), reverse=True)

        def native(name: str) -> str:
            return apply_renamings(name, rules).removesuffix("$")

        names = {memory: stored for stored, memory in _EXPERT_PROJECTIONS}
        names["mlp"] = "ffn"
        for rule in self.source_names(model):
            if isinstance(rule, WeightRenaming):
                source = rule.source_patterns[0]
                if source.startswith("^") and source.endswith(r"\."):
                    names[rule.target_patterns[0].rstrip(".")] = source[1:].replace("\\", "").rstrip(".")
        for name in model.state_dict():
            stored = native(name).removesuffix(".weight")
            name = name.removesuffix(".weight")
            if name.startswith("model.layers."):
                # One relative subtree declaration serves every layer and expert, rather than
                # making the generic name scan grow with a model's number of experts.
                memory, checkpoint = name.split(".", 3)[3], stored.split(".", 2)[2]
                names[memory] = checkpoint
                if memory == _NORM_ALIAS[1].strip("."):
                    names[_NORM_ALIAS[0].replace("\\", "").strip(".")] = checkpoint
            else:
                names[name] = stored
                if name.startswith("model."):
                    names[name.removeprefix("model.")] = stored
                    names["model." + stored] = stored
        return names

    def export_layout(self, name: str, shape: Size) -> ExportLayout | None:
        """Split directly stored routed banks using the shared per-expert declaration."""
        return per_expert_layout(name, shape)

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, str]]:
        """Serving aliases for fused primary inputs, grouped output, compressors and shared experts.

        The engines fuse compressor and shared gate/up inputs; literal fused ignore names reach
        their quantization checks. Primary Q/KV aliases also support explicit float ignore entries;
        the exporter omits their ignores when packed. Patterns cover native and HF source layouts.
        """
        names = self.serve_names(model)
        modules = dict(model.named_modules())
        kept = {name for values in _float_weights(model).values() for name in values}
        kept.update(name for name in modules if ".shared_experts." in name)
        kept.update(
            name
            for name in modules
            if name.endswith(tuple(".self_attn." + memory for _, memory in _ATTENTION_INPUTS))
        )
        kept.add("lm_head")
        out: dict[str, dict[str, str]] = {"vllm": {}, "sglang": {}}
        for name in sorted(kept):
            if not isinstance(modules.get(name), nn.Linear):
                continue
            stored = served_name(name, names)
            checkpoint = (
                "re:(?:"
                + "|".join(re.escape(key) for key in sorted({name, name.removeprefix("model."), stored}))
                + ")$"
            )
            vllm = "lm_head" if name == "lm_head" else "model." + stored
            sglang = vllm.replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
            if stored.endswith(tuple(".attn." + native for native, _ in _ATTENTION_INPUTS)):
                vllm = vllm.rsplit(".", 1)[0] + ".fused_wqa_wkv"
                sglang = sglang.rsplit(".", 1)[0] + ".wqkv_a"
            elif ".compressor." in stored and stored.endswith((".wkv", ".wgate")):
                vllm = vllm.rsplit(".", 1)[0] + ".fused_wkv_wgate"
                sglang = sglang.rsplit(".", 1)[0] + ".wkv_gate"
            elif ".shared_experts." in stored:
                if stored.endswith((".w1", ".w3")):
                    vllm = vllm.rsplit(".", 1)[0] + ".gate_up_proj"
                    sglang = sglang.rsplit(".", 1)[0] + ".gate_up_proj"
                else:
                    vllm = vllm.rsplit(".", 1)[0] + ".down_proj"
                    sglang = sglang.rsplit(".", 1)[0] + ".down_proj"
            for engine, runtime in (("vllm", vllm), ("sglang", sglang)):
                previous = out[engine].get(runtime)
                out[engine][runtime] = checkpoint if previous is None else previous + "|" + checkpoint[3:]
        return out

    def export_config(self, config: dict) -> dict:
        """Translate explicit HF attention, routing and RoPE settings into stock serving aliases."""
        from transformers import DeepseekV4Config

        # Loading uses this same HF config normalization, including legacy native fields.
        cfg = {**config, **DeepseekV4Config(**config).to_dict()}
        # HF's non-hash router always has a correction bias; this legacy flag makes both
        # engines allocate it and still use V4's ungrouped scoring function.
        cfg["topk_method"] = "noaux_tc"
        if "layer_types" in cfg:
            cfg["compress_ratios"] = [
                0 if kind == "sliding_attention" else cfg["compress_rates"][kind]
                for kind in cfg["layer_types"]
            ]
        if "mlp_layer_types" in cfg:
            kinds = cfg["mlp_layer_types"]
            count = kinds.count("hash_moe")
            if kinds != ["hash_moe"] * count + ["moe"] * (len(kinds) - count):
                raise Refused("DeepSeek-V4 stock serving requires hash MoE layers to form a prefix")
            cfg["num_hash_layers"] = count
        rope = cfg.get("rope_parameters") or {}
        if "main" in rope:
            main, compress = rope["main"], rope["compress"]
            if main["rope_type"] != "default":
                raise Refused("DeepSeek-V4 stock serving requires unscaled main RoPE")
            cfg["rope_theta"] = main["rope_theta"]
            cfg["compress_rope_theta"] = compress["rope_theta"]
            cfg["qk_rope_head_dim"] = int(cfg["head_dim"] * main["partial_rotary_factor"])
            if compress["partial_rotary_factor"] != main["partial_rotary_factor"]:
                raise Refused("DeepSeek-V4 stock serving requires equal main/compressed rotary widths")
            flat = {**compress, "rope_theta": cfg["rope_theta"]}
            if compress["rope_type"] == "default":
                flat.update(
                    rope_type="yarn",
                    factor=1.0,
                    attention_factor=1.0,
                    original_max_position_embeddings=cfg["max_position_embeddings"],
                )
            elif compress["rope_type"] != "yarn":
                raise Refused("DeepSeek-V4 stock serving supports default or YaRN compressed RoPE")
            cfg["rope_parameters"] = flat
        return cfg

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Add the fused expert banks and take the always-float weights out of the quantized groups."""
        refusal = None
        try:
            self.export_config(text_config(model).to_dict())
        except Refused as error:
            refusal = str(error)
        graph = FUSED_EXPERTS.adjust(model, graph)
        kept = _float_weights(model)
        readers = dict(graph.extra["readers"])
        layers = tuple(replace(layer, groups=_groups(layer, kept, readers)) for layer in graph.layers)
        return replace(
            graph,
            layers=layers,
            extra={
                **graph.extra,
                "readers": readers,
                "kept_bf16": {**graph.extra.get("kept_bf16", {}), **kept},
                "draft_ignore": self.draft_ignore(model),
                "serving_refusal": refusal,
            },
        )

    def check_served(self, graph: ModelGraph, fmt: Format) -> None:  # noqa: ARG002
        """Refuse unrepresentable serving settings at plan time, while allowing graph inspection."""
        if graph.extra.get("serving_refusal"):
            raise Refused(graph.extra["serving_refusal"])

    def draft_ignore(self, model: nn.Module) -> tuple[str, ...]:
        """Runtime module names a vLLM MTP draft builds the checkpoint's ``mtp.{i}`` layers under.

        vLLM's ``DeepSeekV4MTP`` builds MTP layer ``i`` at ``model.layers.{num_hidden_layers + i}`` and
        constructs its linears there (``attn.fused_wqa_wkv`` merges the checkpoint's ``wq_a`` and ``wkv``),
        so a compressed-tensors ``ignore`` written in checkpoint names never matches the draft. The weights
        live in the checkpoint, not in the model this build quantizes, so they are listed under both names.
        The routed experts are matched by vLLM against their unfused checkpoint names
        (``...experts.{e}.w1``), one regex per MTP layer covers every expert.
        """
        cfg = text_config(model)
        first = cfg.num_hidden_layers
        suffixes = (
            "e_proj",
            "h_proj",
            "attn.fused_wqa_wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
            "ffn.shared_experts.gate_up_proj",
            "ffn.shared_experts.down_proj",
        )
        return tuple(
            name
            for i in range(getattr(cfg, "num_nextn_predict_layers", 0))
            for name in (
                *(f"model.layers.{first + i}.{suffix}" for suffix in suffixes),
                rf"re:model\.layers\.{first + i}\.ffn\.experts\.\d+\.w[123]",
            )
        )


def _float_weights(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Always-float weights by role: the linears' module names and the hyper-connections' parameter names."""
    kept: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if ".compressor.indexer." in name:
            kept.setdefault(INDEXER, []).append(name)
        elif ".compressor." in name:
            kept.setdefault(COMPRESSOR, []).append(name)
        elif getattr(module, "n_groups", None):
            kept.setdefault(GROUPED_OUTPUT, []).append(name)
        elif name.endswith(tuple(".self_attn." + memory for _, memory in _ATTENTION_INPUTS)):
            kept.setdefault(ATTENTION_INPUT, []).append(name)
    for name, _ in model.named_parameters():
        if name.endswith(".position_bias"):
            role = INDEXER if ".compressor.indexer." in name else COMPRESSOR
            kept.setdefault(role, []).append(name)
        elif {"attn_hc", "ffn_hc", "hc_head"} & set(name.split(".")[:-1]):
            kept.setdefault(HYPER_CONNECTIONS, []).append(name)
    return {role: tuple(names) for role, names in kept.items()}


def _groups(layer: LayerGraph, kept: dict[str, tuple[str, ...]], readers: dict) -> tuple[LinearGroup, ...]:
    """`layer`'s groups without the always-float linears; `readers` keeps every consumer of a kept group."""
    drop = {name for role, names in kept.items() if role != HYPER_CONNECTIONS for name in names}
    out = []
    for grp in layer.groups:
        mods = tuple(m for m in grp.modules if m not in drop)
        if len(mods) == len(grp.modules):
            out.append(grp)
        elif not mods:
            readers.pop(grp.modules[0], None)
        else:
            if mods[0] != grp.modules[0]:
                readers[mods[0]] = readers.pop(grp.modules[0], mods)
            out.append(replace(grp, modules=mods))
    return tuple(out)


#: The adapter object the registry and the ``qda.arch_adapters`` entry point hold.
DEEPSEEK_V4 = DeepseekV4()
