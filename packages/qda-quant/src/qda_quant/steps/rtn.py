"""RTN: round to nearest on the served grid. The floor every calibrated method must beat."""

from __future__ import annotations

import torch

from ..arch import ModelGraph
from ..clips import CLIPS, Clip, ClipName
from ..format import Format
from ..step import LayerContext, QuantStep, Refused
from ..target import KernelPlan


class RTN(QuantStep):
    """Round-to-nearest with a chosen scale search.

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Attributes:
        clip: group-scale search (`qda_quant.clips`): ``minmax`` (default), ``mse``, ``imse`` or ``hmse``,
            ``fp4mse`` or ``fp4hmse`` on the ``nvfp4`` grid, and ``imatrix`` on the ``q4k`` grid; ``grid3``
            needs GPTQ. At ``w4a4``, unset or a range search runs ``fp4hmse`` (`qda_quant.format.Format.default_clip`).
    """

    name = "rtn"
    clip: ClipName = "minmax"

    @property
    def needs(self) -> frozenset[str]:  # type: ignore[override]
        """Hessians only for the searches that weight errors by the Hessian diagonal."""
        return frozenset({"hessian", "expert_hessian"} if CLIPS[self.clip].needs_hessian else ())

    def picks_ranges(self) -> bool:
        """Whether the chosen search shrinks group ranges."""
        return CLIPS[self.clip].picks_ranges

    def clip_search(self) -> Clip:
        """The search `clip` names."""
        return CLIPS[self.clip]

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse a search whose scales only start a second-order solve, and one that rejects the format."""
        clip = CLIPS[self.clip]
        if clip.needs_solve:
            raise Refused(f"rtn clip={self.clip!r} requires GPTQ's Hessian-weighted solve")
        if reason := clip.refusal(fmt, plan):
            raise Refused(f"rtn clip={self.clip!r}: {reason}")

    @torch.no_grad()
    def apply_layer(self, ctx: LayerContext) -> None:
        """Quantize every unit of the layer in place."""
        for u in self.units(ctx):
            q = u.quantizer
            # a format the clip does not serve takes the step's default clip
            clip = self.clip_for(q.grid.name) or CLIPS[self.clip]
            w = u.weight().to(ctx.device or u.params[0].device).float()
            hd = None
            if clip.needs_hessian:
                h, _ = u.hessian()
                hd = torch.diagonal(h.reshape(-1, *h.shape[-2:]), dim1=-2, dim2=-1)
            made = q.fit_codes(w, clip.name)
            if made is None:
                s, z = clip.choose(q, w, hd)
                codes = q.quantize(w, s, z)
            else:
                s, z, codes = made
            u.write(codes, s, z)
