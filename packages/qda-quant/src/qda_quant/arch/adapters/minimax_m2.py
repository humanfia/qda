"""MiniMax-M2: Mixtral's serving names, and vLLM's own name for the sparse block's quantization scheme.

transformers 5 names the sparse block ``mlp`` with stacked ``gate_up_proj``/``down_proj``; the vendor checkpoint
stores ``block_sparse_moe.experts.{e}.w1``/``w3``/``w2``, and both stock engines load only those names (vLLM 0.33
refuses ``layers.N.mlp`` outright). So the export stores the sparse block as Mixtral's adapter does.

Stock vLLM 0.33 then builds the block's MoE layer with ``prefix=f"{prefix}.mlp"`` (``minimax_m2.py:288``) and
asks ``quantization_config`` for ``model.layers.N.mlp.experts.0.w1``, while SGLang 0.5.23 asks for
``block_sparse_moe``; `config_names` declares vLLM's name, so an expert override reaches both.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from ..derive import text_config
from .mixtral import Mixtral


class MiniMaxM2(Mixtral):
    """The ``minimax_m2`` architecture adapter: Mixtral's layout, plus vLLM's config name for the block."""

    def matches(self, model: nn.Module) -> bool:
        """True for the MiniMax-M2 text model."""
        return text_config(model).model_type == "minimax_m2"

    def config_names(self, model: nn.Module) -> Mapping[str, Mapping[str, str]]:  # noqa: ARG002
        """Stock vLLM matches the stored ``block_sparse_moe`` tree against ``quantization_config`` as ``mlp``."""
        return {"vllm": {"block_sparse_moe": "mlp"}}


#: The adapter object the registry holds.
MINIMAX_M2 = MiniMaxM2()
