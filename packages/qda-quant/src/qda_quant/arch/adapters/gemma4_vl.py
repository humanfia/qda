"""Gemma 4 multimodal: the vision embedder's projection writes the stream; SGLang renames both towers."""

from __future__ import annotations

import re
from typing import Any

from torch import nn

from .stream import StreamDeclaration

_VISION = "vision_tower.encoder.layers"
_VISION_RE = r"model\.vision_tower\.encoder\.layers"
_AUDIO = "audio_tower.layers."
_AUDIO_RE = r"model\.audio_tower\.layers"

#: checkpoint path inside an audio layer -> the one SGLang 0.5.23 builds
#: (``Gemma4ForConditionalGeneration._remap_audio_tower_name``), the more specific first
_SGLANG_AUDIO = (
    ("self_attn.relative_k_proj", "attention.attn.relative_position_embedding.pos_proj"),
    ("self_attn.post.", "attention.post."),
    ("self_attn.", "attention.attn."),
    ("feed_forward1.", "ffw_layer_start."),
    ("feed_forward2.", "ffw_layer_end."),
)


def _sglang_audio(name: str) -> str:
    """The name SGLang builds an audio-layer linear under: ``layers.{i}`` becomes ``conformer.{i}``."""
    head, _, rest = name.partition(_AUDIO)
    index, _, path = rest.partition(".")
    for old, new in _SGLANG_AUDIO:
        if path.startswith(old):
            path = new + path[len(old) :]
            break
    return f"{head}audio_tower.conformer.{index}.{path}"


class Gemma4VL(StreamDeclaration):
    """The ``gemma4`` declaration: ``model.embed_vision`` projects the tower's features into the stream."""

    def matches(self, model: nn.Module) -> bool:
        """True for a Gemma 4 multimodal model, not its text-only class (``gemma4_text``)."""
        return getattr(model.config, "model_type", "") == "gemma4"

    def stream(self, model: nn.Module) -> dict[str, Any]:  # noqa: ARG002
        """The projection after the tower; the tower itself never reaches the stream."""
        return {
            "writers": {
                "model.embed_vision": ("model.embed_vision.embedding_projection",),
                "model.vision_tower": (),
            }
        }

    def pack_like(self, model: nn.Module) -> dict[str, str]:
        """A KV-shared layer's checkpoint k/v projections take its q_proj's format.

        transformers and vLLM 0.33 build only ``q_proj`` in a layer that reads an earlier layer's K/V (vLLM
        drops the checkpoint's ``k_proj``/``v_proj`` there), but SGLang 0.5.23 builds a fused ``qkv_proj`` and
        refuses one with a packed q and float k/v. The weights are never read, so their codes cannot matter.
        """
        return {
            f"{name}.{proj}": f"{name}.q_proj"
            for name, module in model.named_modules()
            if getattr(module, "is_kv_shared_layer", False)
            for proj in ("k_proj", "v_proj")
        }

    def runtime_names(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """SGLang's names for the towers' float linears: fused q/k/v and gate/up, and the audio tower's tree.

        SGLang drops the checkpoint's ``model.`` prefix and builds fused q/k/v and gate/up linears.
        Declare both their quantization prefixes and the shards it probes; the unfused linears also need
        names without ``model.``. SGLang also rebuilds the audio tower's layers under its own tree
        (``conformer.{i}``, ``ffw_layer_start``, ``attention.attn``, a fused ``qkv_proj``): an audio linear
        missing there is built quantized, and its GLU wrapper fails before any weight loads. Checkpoint
        names remain in the export for vLLM.
        """
        from . import FusedRuntime  # the package defines it after importing its adapters

        declared: dict[str, Any] = {
            name.removeprefix("model."): name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and name.startswith(
                ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")
            )
        }
        declared = {(_sglang_audio(k) if k.startswith(_AUDIO) else k): v for k, v in declared.items()}
        for name, module in model.named_modules():
            if getattr(module, "is_kv_shared_layer", False):
                for runtime in (name, name.removeprefix("model.")):
                    declared[f"{runtime}.qkv_proj"] = FusedRuntime(
                        checkpoint=rf"re:^{re.escape(name)}\.(q|k|v)_proj$",
                        shards=tuple(f"{runtime}.{part}_proj" for part in "qkv"),
                    )
        config: Any = getattr(model, "config", None)
        vision: Any = getattr(config, "vision_config", None)
        layers: int = getattr(vision, "num_hidden_layers", 0) if vision is not None else 0
        for i in range(layers):
            layer = f"{_VISION}.{i}"
            path = rf"re:{_VISION_RE}\.{i}"
            declared[f"{layer}.self_attn.qkv_proj"] = FusedRuntime(
                checkpoint=rf"{path}\.self_attn\.(q|k|v)_proj\.linear$",
                shards=tuple(f"{layer}.self_attn.{name}_proj" for name in "qkv"),
            )
            declared[f"{layer}.mlp.gate_up_proj"] = FusedRuntime(
                checkpoint=rf"{path}\.mlp\.(gate|up)_proj\.linear$",
                shards=(f"{layer}.mlp.gate_proj", f"{layer}.mlp.up_proj"),
            )
        audio: Any = getattr(config, "audio_config", None)
        for i in range(getattr(audio, "num_hidden_layers", 0) if audio is not None else 0):
            layer = f"audio_tower.conformer.{i}.attention.attn"
            declared[f"{layer}.qkv_proj"] = FusedRuntime(
                checkpoint=rf"re:{_AUDIO_RE}\.{i}\.self_attn\.(q|k|v)_proj\.linear$",
                shards=tuple(f"{layer}.{name}_proj" for name in "qkv"),
            )
        return {"sglang": declared}


GEMMA4_VL = Gemma4VL()
