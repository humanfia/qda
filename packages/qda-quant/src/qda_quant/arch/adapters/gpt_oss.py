"""gpt-oss: fused, interleaved expert banks, stored the way stock vLLM reads INT4 experts.

transformers stores interleaved gate/up weights as ``[E, D, 2I]`` and down weights as ``[E, I, D]``, with biases.
Stock vLLM 0.33 reads INT4 experts through ``experts.<e>.w13_weight_packed`` and ``..._weight_scale`` (``w2_``
for the down projection). Fused ``gate_up_proj_packed`` names are silently skipped, leaving experts uninitialized,
so the exporter writes one tensor per expert.

Its loader has no path for routed-expert zero points (the name falls through to the dense w13 branch and raises
``IndexError``), and its INT4 expert path is ``CompressedTensorsWNA16MoEMethod`` only: a W4A8 checkpoint selects
``CompressedTensorsW4A8Int8MoEMethod``, which ``GptOssRoutedExperts.weight_loader`` sends to its unquantized
branch. `check_served` refuses both at plan time.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ...format import Format
from ...step import Refused
from .. import ExportLayout, ModelGraph
from ..derive import text_config
from .fused_experts import DOWN, FUSED_EXPERTS, IN

#: fused parameter -> its projection name in vLLM's expert suffix map
PROJECTIONS = {IN: "w13", DOWN: "w2"}


class GptOss:
    """The ``gpt_oss`` architecture adapter (see `qda_quant.arch.adapters.ArchAdapter`)."""

    def matches(self, model: nn.Module) -> bool:
        """True for the gpt-oss text model (the only gpt-oss architecture)."""
        return text_config(model).model_type == "gpt_oss"

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """The same banks as any fused layout: gpt-oss stores the two 3-D parameters and their biases."""
        graph = FUSED_EXPERTS.adjust(model, graph)
        return replace(
            graph,
            extra={
                **graph.extra,
                "expert_biases": tuple(
                    name
                    for name, _ in model.named_parameters()
                    if ".experts." in name and name.endswith("_bias")
                ),
            },
        )

    def check_served(self, graph: ModelGraph, fmt: Format) -> None:
        """Refuse an int body whose stock vLLM gpt-oss loader cannot read its routed experts.

        Raises:
            Refused: the routed experts are asymmetric (no path for expert zero points) or W4A8 (no int8
                expert path). A bare spelling defaults the experts symmetric (`Format.expert_format`), so
                only an explicit ``-asym`` reaches the first refusal.
        """
        if not any(layer.banks for layer in graph.layers):
            return
        if not fmt.expert_format.symmetric:
            raise Refused(
                "gpt-oss: stock vLLM has no load path for routed-expert zero points (its gpt-oss loader raises "
                "IndexError on the w13/w2 zero-point name), so a checkpoint with routed experts must be "
                "symmetric, e.g. w4a16-sym-g64 (a spelling without -asym defaults them symmetric)"
            )
        if fmt.act != "bf16":
            raise Refused(
                "gpt-oss: stock vLLM loads INT4 routed experts only through CompressedTensorsWNA16MoEMethod "
                "(integer weights, bf16 activations); an FP8-activation body selects a method its gpt-oss loader "
                "reads as unquantized weights"
            )

    def export_layout(self, name: str, shape: torch.Size) -> ExportLayout | None:
        """The per-expert ``w13_*`` / ``w2_*`` tensors of a fused bank, and the fused source it replaces.

        No ``*_shape`` tensor: no gpt-oss load path accepts one, and the WNA16 method never reads one. The
        bank's biases keep their fused names (vLLM maps ``experts.gate_up_proj_bias`` to ``w13_bias``).
        """
        for param, projection in PROJECTIONS.items():
            suffix = f".{param}"
            if not name.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            pieces = tuple(
                (
                    f"{base}.{expert}.{projection}_weight",
                    (expert,),
                    {
                        "packed": f"{base}.{expert}.{projection}_weight_packed",
                        "scale": f"{base}.{expert}.{projection}_weight_scale",
                    },
                )
                for expert in range(shape[0])
            )
            return ExportLayout(pieces, (name, f"{name}_blocks", f"{name}_scales"))
        return None


#: The adapter object the registry holds.
GPT_OSS = GptOss()
