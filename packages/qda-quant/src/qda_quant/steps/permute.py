"""Channel permutation: reorder channels so each quantization group spans a narrow magnitude band.

A group-quantized weight spends one scale per block of consecutive input channels, set by its largest
weight; which channels share a block depends on the checkpoint's order. This model step
sorts channels by decreasing max ``|W|`` over every module that reads
them and moves every tensor that touches them, so the float model computes the same function:

* the residual stream (one permutation ``P_h``): every norm-reader's input columns (attention, GDN
  inputs, MLP, router, experts, ``lm_head``), every writer's output rows and biases, the embedding (once:
  a table tied to the head is the head's weight, already permuted as its input), and
  every residual norm's gain (a permutation commutes with an elementwise gain: nothing is folded);
* each layer's MLP intermediate (``P_i``): the rows of gate and up (both halves of fused experts, one
  permutation shared by the bank) and down's input columns.

Nothing is stored: the checkpoint is a permuted copy with the same config.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from ..arch import ModelGraph
from ..arch.adapters import fused_experts
from ..arch.adapters.fused_experts import in_param
from ..format import Format
from ..step import Refused, Step
from ..target import KernelPlan
from ._model import (
    OUTPUT_KINDS,
    Once,
    ignored_in,
    layer_without_classified_writer,
    layer_without_fold_anchor,
    unique,
)


class Permute(Step):
    """Residual-stream and MLP-intermediate channel permutation (order: max |W| of the readers).

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Attributes:
        residual: permute the residual stream.
        intermediate: permute each MLP's intermediate channels.
    """

    name = "permute"
    experimental = True
    kind = "model"
    needs_verified_graph = True
    residual: bool = True
    intermediate: bool = True

    @property
    def changes_stream(self) -> bool:
        """Whether this step's configured transform changes the residual basis."""
        return self.residual

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse a fused bank stored ``[E, in, out]``, or a permutation that cannot keep the float function.

        Both passes move a fused bank's hidden or intermediate channels on the last axis of
        ``gate_up_proj`` / ``down_proj``, i.e. they assume ``[E, out, in]``. For a bank stored
        ``[E, in, out]`` that last axis is the output, so the channels would be reordered on the wrong
        axis (or by the wrong length, when in != out). A residual permutation also needs a norm at layer
        scope to permute with its readers, and no permutation may skip a linear that ``ignore`` keeps
        float while the quantized ones around it move.
        """
        if not self.residual and not self.intermediate:
            raise Refused(
                "permute: residual and intermediate are both false, so the step changes nothing; remove it"
            )
        bad = next((b for g in graph.layers for b in g.banks if b.transposed), None)
        if bad is not None:
            raise Refused(
                f"permute: routed experts {bad.experts} are stored [E, in, out]; channel permutation "
                "assumes [E, out, in] — quantize without permute"
            )
        if module := layer_without_classified_writer(graph):
            raise Refused(
                f"permute: {module} writes the residual stream through linears that derive as an "
                "unclassified group kind (the architecture names its blocks something other than "
                "attention/mlp), so the permutation would move the readers and leave those writers in the "
                "old basis, changing the float function; quantize without permute (e.g. gptq alone) until "
                "those blocks are classified"
            )
        if self.residual:
            ignored_in(graph, "permute", OUTPUT_KINDS, "the linears that write the residual stream")
        if self.intermediate:
            ignored_in(graph, "permute", ("mlp_in",), "the linears that write an MLP's intermediate channels")
        if self.residual and (module := layer_without_fold_anchor(graph)):
            raise Refused(
                f"permute: {module} has no norm at layer scope (its input is normalized inside a "
                "submodule), so the residual stream cannot be permuted consistently; quantize without "
                "permute (e.g. gptq alone)"
            )

    @torch.no_grad()
    def apply_model(self, model: nn.Module, graph: ModelGraph, run: Any) -> None:  # noqa: ARG002
        """Permute in place (same float function)."""
        if self.intermediate:
            for g in graph.layers:
                self._intermediate(model, g)
        if self.residual:
            self._residual(model, graph)

    def _residual(self, model: nn.Module, graph: ModelGraph) -> None:
        get = model.get_submodule
        readers = [r for g in graph.layers for k, rs in graph.extra["readers"].items()
                   if k.rpartition(".")[0] == g.module for r in rs]  # fmt: skip
        readers.append(graph.lm_head)
        reader_params = unique([getattr(get(r), in_param(get(r))) for r in readers])
        cols = unique(
            [*reader_params, cast(torch.Tensor, get(graph.embed).weight)]
        )  # the embedding moves too, unless it is the head
        score = None
        for p in reader_params:  # order: max |W| over the readers
            m = p.detach().abs().reshape(-1, p.shape[-1]).amax(0).float()
            score = m if score is None else torch.maximum(score, m.to(score.device))
        assert score is not None
        perm = torch.argsort(score, descending=True)
        # each storage once: a tied embedding is the head's weight, permuted once on its hidden (last) dim
        for p in cols:
            p.data.copy_(p.data[..., perm.to(p.device)])
        once = Once()
        for g in graph.layers:
            for k in graph.extra["readers"]:
                if k.rpartition(".")[0] == g.module:  # decoder-layer norms
                    once(cast(torch.Tensor, get(k).weight), lambda t: _rows(t, perm))
            for grp in g.groups:
                if grp.kind in OUTPUT_KINDS:
                    for n in grp.modules:
                        once(cast(torch.Tensor, get(n).weight), lambda t: _rows(t, perm, -2))
                        bias = cast(torch.Tensor | None, get(n).bias)
                        if bias is not None:
                            once(bias, lambda t: _rows(t, perm))
            for bank in g.banks:
                if isinstance(bank.experts, str):
                    once(getattr(get(bank.experts), fused_experts.DOWN), lambda t: _rows(t, perm, -2))
                else:
                    for _ins, down in bank.experts:
                        once(cast(torch.Tensor, get(down).weight), lambda t: _rows(t, perm, -2))
        once(cast(torch.Tensor, get(graph.final_norm).weight), lambda t: _rows(t, perm))

    def _intermediate(self, model: nn.Module, g: Any) -> None:  # noqa: ANN401
        get = model.get_submodule
        for grp in g.groups:  # dense MLP: down reads act(gate) * up
            if grp.kind != "mlp_out" or grp.prev is None:
                continue
            down_weight = cast(torch.Tensor, get(grp.modules[0]).weight)
            perm = torch.argsort(down_weight.abs().amax(0).float(), descending=True)
            ins = next((x for x in g.groups if grp.prev in x.modules), None)
            for n in ins.modules if ins else (grp.prev,):
                _rows(cast(torch.Tensor, get(n).weight), perm, -2)
                bias = cast(torch.Tensor | None, get(n).bias)
                if bias is not None:
                    _rows(bias, perm)
            down_weight.data.copy_(down_weight.data[:, perm.to(down_weight.device)])
        for bank in g.banks:  # one permutation for the whole bank
            if not isinstance(bank.experts, str):
                continue
            m = get(bank.experts)
            gu, dn = getattr(m, fused_experts.in_param(m)), getattr(m, fused_experts.DOWN)
            gate, _ = fused_experts.split(m)
            perm = torch.argsort(dn.detach().abs().amax(dim=(0, 1)).float(), descending=True).to(gu.device)
            if gate:  # gated: permute the gate and up halves alike
                gu.data.copy_(torch.cat([gu.data[:, :gate][:, perm], gu.data[:, gate:][:, perm]], 1))
            else:  # non-gated: the one up projection
                gu.data.copy_(gu.data[:, perm])
            dn.data.copy_(dn.data[..., perm])


def _rows(p: torch.Tensor, perm: torch.Tensor, dim: int = 0) -> None:
    p.data.copy_(p.data.index_select(dim % p.dim(), perm.to(p.device)))
