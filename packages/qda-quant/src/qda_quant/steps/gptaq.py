"""GPTAQ: asymmetric calibration refit followed by the regular GPTQ solve."""

from __future__ import annotations

import logging
from typing import ClassVar

import torch
from pydantic import Field

from ..arch import ModelGraph
from ..format import Format
from ..solve.units import BoundUnit
from ..step import Refused
from ..target import KernelPlan
from .gptq import GPTQ


def refit(
    weight: torch.Tensor, hessian: torch.Tensor, cross: torch.Tensor, alpha: float, damp: float
) -> torch.Tensor:
    """Apply ``W + alpha (W C)(H + lambda I)^-1``, leaving weights alone if the refit is invalid."""
    hessian = hessian.float().clone()
    cross = cross.float()
    if not bool(torch.isfinite(hessian).all() & torch.isfinite(cross).all()):
        raise RuntimeError(
            "gptaq: non-finite Hessian or cross-Hessian statistics (the old qda stops here too)"
        )
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    hessian = (
        hessian
        + torch.eye(hessian.shape[-1], device=hessian.device)
        * diagonal.mean(-1, keepdim=True).unsqueeze(-1)
        * damp
    )
    try:
        factor = torch.linalg.cholesky(hessian)
        correction = torch.cholesky_solve(
            torch.matmul(weight.float(), cross).transpose(-1, -2), factor
        ).transpose(-1, -2)
    except torch.linalg.LinAlgError:  # pyright: ignore[reportPrivateImportUsage]  # exported at runtime
        logging.getLogger(__name__).warning("gptaq refit skipped: damped Hessian is not positive definite")
        return weight.float()
    return weight.float() + alpha * correction


class GPTAQ(GPTQ):
    """GPTQ with a cross-Hessian refit for quantized-prefix error compensation.

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    The refit uses the gap between the float model's and the quantized model's layer inputs (the quantized prefix);
    it is meant for asymmetric calibration, where activation quantization widens that gap, and is not expected to
    help weight-only W4A16. Nonzero alpha needs one card and refuses routed-expert banks.
    """

    name = "gptaq"
    experimental = True
    splits_experts: ClassVar[bool] = False  # the refit reads whole banks; not split across cards
    alpha: float = Field(default=0.30, description="Fraction of the float-input least-squares refit.")

    @property
    def needs(self) -> frozenset[str]:
        """Alpha zero uses ordinary GPTQ statistics and allocates no float stream."""
        return GPTQ.needs if self.alpha == 0 else frozenset({"hessian", "cross_hessian"})

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:
        """Require the cross statistic and a valid GPTQ act-order mode."""
        super().check(graph, fmt, plan)
        if not 0.0 <= self.alpha <= 1.0:
            raise Refused(f"gptaq alpha={self.alpha}: expected a value in [0, 1]")
        if self.alpha != 0 and any(layer.banks for layer in graph.layers):
            raise Refused("gptaq: cross-Hessian calibration currently supports dense groups only")

    def _target(
        self, u: BoundUnit, w: torch.Tensor, h: torch.Tensor, n: torch.Tensor | int | None
    ) -> torch.Tensor:
        """Refit the weights toward the float inputs, stored at the layer's dtype."""
        if self.alpha == 0.0:
            return w
        cross = u.cross_hessian().to(w.device).float()
        if n is not None:
            cross = cross / torch.as_tensor(n, device=w.device).clamp_min(1).reshape(-1, 1, 1)
        return refit(w, h, cross, self.alpha, self.damp).to(u.params[0].dtype).float()
