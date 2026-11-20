"""Head format: round ``lm_head`` (optionally the embedding) in its own format, e.g. an int4 head on an int3 body.

The head is stored compressed-tensors under its own config group (``re:.*lm_head$``, which vLLM matches
before the ``Linear`` class target): ``pack-quantized`` for an integer format, ``float-quantized`` for
``w8a8-fp8``, whose E4M3 codes and fp32 per-output-channel scales are the same tensors the FP8 body groups
store. An FP8 head's config group also declares the body groups' dynamic per-token FP8 input activations
(`output.ct_config._ct_acts`), so the engine quantizes the last hidden state at the head's input too; the
scheme is dynamic, so it needs no calibration and stores no input scale. The embedding, which vLLM builds
without a quant config, is stored dequantized in bf16 on the head's grid (code x scale in fp32, cast once),
as a quantized lookup would compute each row. Rounding is to nearest (the head has no Hessian in a layer
pass); the codes go to the build's ``run.qparams`` under the head's module name, where `export.Writer` reads
them.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from ..arch import ModelGraph
from ..format import Format
from ..step import QuantStep, Refused
from ..target import KernelPlan


class Head(QuantStep):
    """Store ``lm_head`` (and optionally the embedding) in its own format.

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Attributes:
        format: head format, e.g. ``w4a16-sym-g64``, ``w4a16`` or ``w8a8-fp8``.
        embed: also put the embedding table on that grid (stored dequantized, bf16).
    """

    name = "head"
    kind = "model"
    preserves_quantized = True  # only lm_head and the embedding, which no layer step quantizes
    format: str = "w4a16-sym-g64"
    embed: bool = False

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse a tied head (it would quantize the embedding lookup too) and formats the release does not offer.

        The head's input is bf16 or, in ``w8a8-fp8``, quantized to FP8 per token by the same dynamic scheme
        the FP8 body groups declare. An integer head stays weight-only, and an NVFP4 head is refused by name:
        it is not offered in this release.
        """
        if graph.extra.get("tied"):
            raise Refused("head: tied input/output embeddings; untie them first")
        f = Format.parse(self.format)
        if f.weight_type == "e2m1":
            raise Refused(f"head: {f.name}: an NVFP4 head is not offered in this release (bf16 or w8a8-fp8)")
        if f.act != "bf16" and f.weight_type != "fp8":
            raise Refused(f"head: {f.name}: the head runs weight-only (no FP8 activations)")
        plan.target.plan(f, "dense")  # refuses formats no kernel serves on the build's target

    def check_grid(self, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """The grid must be servable in the head's own format, not the body's."""
        f = Format.parse(self.format)
        super().check_grid(f, plan.target.plan(f, "dense"))

    @torch.no_grad()
    def apply_model(self, model: nn.Module, graph: ModelGraph, run: Any) -> None:
        """RTN the head (and embedding) on the served grid, in place; the head's codes go to ``run.qparams``."""
        f = Format.parse(self.format)
        q = self.quantizer(f, run.target.plan(f, "dense"))
        head = cast(nn.Linear, model.get_submodule(graph.lm_head))
        w = head.weight.float()
        s, z = q.params(w)
        codes = q.quantize(w, s, z)
        head.weight.data.copy_(q.dequant(codes, s, z).to(head.weight.dtype))
        qp = {"format": f.name, "codes": q.stored_codes(codes), "scale": q.stored_scale(s)}
        if z is not None:
            qp["zero_point"] = z.to(torch.int8)
        run.qparams[graph.lm_head] = qp
        if self.embed:
            emb = cast(nn.Embedding, model.get_submodule(graph.embed)).weight
            e = emb.float()
            se, ze = q.params(e)
            emb.data.copy_(q.fake(e, se, ze).to(emb.dtype))
