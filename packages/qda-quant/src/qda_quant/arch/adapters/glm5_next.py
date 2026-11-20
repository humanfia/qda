"""GLM-5.3-Flash (``glm5_next``): fused experts with a clamped gate, and attention that stays bf16.

Three things generic derivation cannot see:

* The routed experts are a fused bank (`fused_experts`). They have no ``act_fn``: the model clamps gate and up to
  ``swiglu_limit`` before the gate (``_apply_gate``), so the bank's down input says so (`_down_input`).
* A decoder layer returns ``(hidden_states, topk_indices)`` and the next one reads the indices as
  ``prev_topk_indices`` (`layer_carry`).
* Stock vLLM 0.33 builds every attention projection with no quantization config, both the MLA projections and
  the linear attention's, so a packed attention weight does not load. They and the hyper-connections are always
  bf16, as in the vendor's own FP8 release of the linear-attention layers.
"""

from __future__ import annotations

from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING

from torch import nn

from .. import ModelGraph
from ..derive import text_config
from .deepseek_v4 import HYPER_CONNECTIONS, _groups
from .fused_experts import FUSED_EXPERTS, is_bank

if TYPE_CHECKING:
    import torch

    from ...target import Target

#: Role of the attention projections, which always stay bf16 (not a `qda_quant.arch.KEEP_ROLES`).
ATTENTION = "attention"


def _down_input(self: nn.Module, gate_up: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:  # noqa: ARG001
    """The bank's own gate: the router weights apply after the down projection, so they play no part here."""
    return self._apply_gate(gate_up)  # type: ignore[operator]


class Glm5Next:
    """The ``glm5_next`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True for a GLM-5.3-Flash text backbone."""
        return text_config(model).model_type == "glm5_next_text"

    def layer_carry(self) -> tuple[str, ...]:
        """The indexer's top-k indices go from each layer to the next."""
        return ("prev_topk_indices",)

    def linear_attention_bytes(self, block: nn.Module, tokens: int, esz: int, chunk: int) -> int:
        """The peak of one forward through the delta rule, reached as it builds its first intra-chunk matrix.

        `block` is the forget gate. Its ``g`` is per channel, so the pairwise decay is
        ``[batch, head, chunks, chunk, chunk, head_dim]``, and three sets of that size live at once: the mask
        and the two products of ``k_beta * key * decay_mask``. Beside them, in fp32, q, k, v, g, k_beta, v_beta
        and the caller's g, and in the model dtype the convolution's output. Each is one value per token and
        channel, and the rule pads the sequence to whole chunks.

        Reference rule, 12.532 GiB measured against 12.531 here at 64 heads of 128 and 2,048 tokens. `fla`'s
        kernel allocates its own and has not been measured.
        """
        pad = -(-tokens // chunk) * chunk
        return pad * block.qkv_dim * (3 * 4 * chunk + 7 * 4 + 3 * esz)  # type: ignore[operator]

    def configure_model(self, model: nn.Module, target: Target) -> None:  # noqa: ARG002
        """Give every expert bank the model's own down input."""
        for m in model.modules():
            if is_bank(m):
                object.__setattr__(m, "down_input", MethodType(_down_input, m))

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Add the fused banks and take the always-float weights out of the quantized groups."""
        graph = FUSED_EXPERTS.adjust(model, graph)
        kept = _float_weights(model, graph)
        readers = dict(graph.extra["readers"])
        layers = tuple(replace(layer, groups=_groups(layer, kept, readers)) for layer in graph.layers)
        return replace(
            graph,
            layers=layers,
            extra={
                **graph.extra,
                "readers": readers,
                "kept_bf16": {**graph.extra.get("kept_bf16", {}), **kept},
            },
        )


def _float_weights(model: nn.Module, graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """Always-float weights by role: the layers' attention linears and the hyper-connections' parameters."""
    attention = tuple(
        name
        for g in graph.layers
        for name, m in model.get_submodule(g.module).named_modules(prefix=g.module)
        if isinstance(m, nn.Linear) and ".self_attn." in name
    )
    hyper = tuple(
        name
        for name, _ in model.named_parameters()
        if {"attn_hc", "ffn_hc", "hc_head"} & set(name.split("."))
    )
    return {ATTENTION: attention, HYPER_CONNECTIONS: hyper}


#: The adapter object the registry and the ``qda.arch_adapters`` entry point hold.
GLM5_NEXT = Glm5Next()
