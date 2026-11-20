"""Model steps that prepare per-token Hessian weights."""

from __future__ import annotations

from typing import ClassVar

import torch
from pydantic import Field

from ..engine import token_weights
from ..step import Refused, Step
from ._attention import attention_kernel


class _TokenWeights(Step):
    kind = "model"
    needs: ClassVar[frozenset[str]] = frozenset({"inputs"})
    needs_verified_graph: ClassVar[bool] = True
    requires_single_card_calibration: ClassVar[bool] = True
    #: sets the calibration rows' token weights (`planning.check_token_weight_readers`)
    weights_tokens: ClassVar[bool] = True

    def check(self, graph, fmt, plan) -> None:  # noqa: ANN001, ARG002
        """Require expert banks that expose row-aligned routed inputs."""
        if any(not isinstance(bank.experts, str) for layer in graph.layers for bank in layer.banks):
            raise Refused(f"{self.name}: token weights currently require fused expert banks")


class Guided(_TokenWeights):
    """Weight tokens by the squared output gradients of the anchor loss.

    Reproducibility: noise band by default (docs/reproduce.md): quote a 3-seed mean and band. The cuDNN attention
    backward varies in its last bits, so the weights and every Hessian built on them differ. ``reproducible``
    removes that source.
    """

    name = "guided"
    clip: float = Field(default=50.0, gt=0, description="Maximum normalized per-group token weight.")
    power: float = Field(default=1.0, gt=0, description="Exponent applied to squared gradient norms.")
    reproducible: bool = Field(
        default=False,
        description="Run the anchor pass on torch's math attention, so a repeat build writes the same bytes. "
        "The measured step took 2.3x as long, and the build 49.1 GiB of card memory instead of 29.1 "
        "(docs/reproduce.md).",
    )

    def apply_model(self, model: torch.nn.Module, graph, run) -> None:  # noqa: ANN001, ARG002
        """Attach output-gradient weights from the engine's calibration batches."""
        kernel = attention_kernel(self.reproducible)
        with kernel, token_weights.anchor_residency(model, graph, run.devices[0], store=run.plan.store):
            run.token_weights = token_weights.guided(
                model, run.batches(), run.devices[0], graph=graph, clip=self.clip, power=self.power
            )


class FisherTrace(_TokenWeights):
    """Weight tokens by the trace of the anchor next-token Fisher matrix.

    Reproducibility: bit-identical (forward passes only; same GPU type, software stack and card count;
    docs/reproduce.md).
    """

    name = "fisher_trace"
    window: int = Field(default=16, ge=1, description="Future positions included in each token's score.")
    floor: float = Field(default=0.25, ge=0, le=1, description="Minimum fraction of the mean token weight.")

    def apply_model(self, model: torch.nn.Module, graph, run) -> None:  # noqa: ANN001, ARG002
        """Attach Fisher weights from the engine's calibration batches."""
        with token_weights.anchor_residency(model, graph, run.devices[0], store=run.plan.store):
            run.token_weights = token_weights.fisher_trace(
                model, run.batches(), run.devices[0], window=self.window, floor=self.floor
            )
