"""GPTQ (arXiv:2210.17323): second-order rounding with error feedback, solved as stacks.

Rows that share an input share one Hessian and one Cholesky (q/k/v; gate/up; every routed expert of a
layer at once). Defaults are llm-compressor 0.15's: ``dampening_frac`` 0.01, ``actorder`` static, block
128, min/max group scales fixed from the original weights before the solve.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
from pydantic import Field

from ..arch import ModelGraph
from ..clips import CLIPS, Clip, ClipName
from ..format import Format
from ..solve import solve
from ..solve.units import BoundUnit
from ..step import LayerContext, QuantStep, Refused
from ..target import KernelPlan


class GPTQ(QuantStep):
    """GPTQ over every linear and expert bank of each layer.

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Attributes:
        damp: diagonal dampening as a fraction of the mean Hessian diagonal.
        actorder: ``static`` (columns by decreasing Hessian diagonal, nothing stored) or ``none``.
        block: columns per lazy-update block.
        clip: group-scale search before the solve (`qda_quant.clips`: ``minmax``, ``mse``, ``imse``, ``hmse``,
            ``grid3``, ``fp4mse`` or ``fp4hmse`` on the ``nvfp4`` grid, and ``imatrix`` on the ``q4k`` grid). At
            ``w4a4``, unset or a range search runs ``fp4hmse`` (`qda_quant.format.Format.default_clip`).
        grid_k: grid3 search reach on either side of the starting scale; defaults to 1, as in the old
            CLI. The paper's ``recipes-1023.json`` q35 record says k=2. That conflicts with the CLI
            default; neither value establishes which setting the original run used.
        impl: ``auto`` (Triton on CUDA), ``torch`` (reference) or ``triton``.
        fused_grid: use a grid's declared fused CUDA block arithmetic (Q4_K/Q6_K). False preserves
            the previous torch grid solve under ``impl: auto``; ``impl: torch`` always uses the reference.
        router_weighted: routed-expert Hessians weight each token by its router probability (the engine
            reads this flag when it accumulates ``expert_hessian``).
        all_token_blend: share ``a`` of each expert's Hessian from every calibration token, the rest from
            the tokens routed to it: ``(1 - a) H_routed / n_routed + a H_all / n_all``; an expert no token
            was routed to takes the all-token Hessian. 0 keeps routed-token Hessians only.
    """

    name = "gptq"
    needs = frozenset({"hessian", "expert_hessian"})
    damp: float = 0.01
    actorder: Literal["none", "static"] = "static"
    block: int = 128
    clip: ClipName = "minmax"
    grid_k: int = Field(default=1, ge=0)
    impl: Literal["auto", "torch", "triton"] = "auto"
    fused_grid: bool = True
    router_weighted: bool = False
    all_token_blend: float = Field(default=0.0, ge=0.0, le=1.0)
    #: the engine may split each fused expert bank's solve across cards (`units.ExpertShare`)
    splits_experts: ClassVar[bool] = True

    def solver_workspace(self, rows: int, cols: int) -> int:
        """The Cholesky and the column loop, as measured on one expert of a real GLM layer.

        `solve.expert_workspace` documents the tensors behind the bound.
        """
        return solve.expert_workspace(rows, cols)

    def dense_workspace(self, rows: int, cols: int) -> int:
        """The most bytes solving one dense ``[rows, cols]`` unit costs, in fp32.

        Measured on Llama-3.1-8B's ``mlp.down_proj`` (``[4096, 14336]``, act-order static): at the solve's peak the
        card held six Hessian-sized ``[cols, cols]`` tensors (the averaged copy, its act-order permutation, and the
        factorization's Cholesky factor, inverse, upper factor and workspace) and three weight-sized ones (two fp32
        copies of the weight and the written-back codes and scales). `solve.expert_workspace` prices four
        Hessian-sized tensors, which a bank's weight-sized terms cover but a wide dense unit's do not (1.6 GiB short
        there).
        """
        return 4 * max(6 * cols * cols + 3 * rows * cols, 2 * cols * cols + 8 * rows * cols)

    @property
    def all_token_hessian(self) -> bool:
        """Whether the engine gathers each bank's all-token Hessians for this step."""
        return self.all_token_blend > 0

    def picks_ranges(self) -> bool:
        """Whether the chosen search shrinks group ranges."""
        return CLIPS[self.clip].picks_ranges

    def clip_search(self) -> Clip:
        """The search `clip` names."""
        return CLIPS[self.clip]

    def check_grid(self, fmt: Format, plan: KernelPlan) -> None:
        """Refuse an explicitly requested fused implementation without enabled grid arithmetic."""
        super().check_grid(fmt, plan)
        grid = self.quantizer(fmt, plan).grid
        if self.impl == "triton" and not grid.range_rule and not (self.fused_grid and grid.gptq_fused):
            raise Refused(f"gptq impl='triton': no enabled fused formula for grid {grid.name!r}")

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:
        """Refuse settings the serving stack does not support before calibration starts."""
        if self.actorder not in plan.act_order:
            raise Refused(f"gptq actorder={self.actorder!r}: {plan.kernel} loads {plan.act_order} only")
        if reason := CLIPS[self.clip].refusal(fmt, plan):
            raise Refused(f"gptq clip={self.clip!r}: {reason}")
        if self.grid_k != 1 and self.clip != "grid3":
            raise Refused(f"{self.name} grid_k={self.grid_k}: only clip='grid3' reads it, not {self.clip!r}")
        banks = [bank for layer in graph.layers for bank in layer.banks]
        fused = bool(banks) and all(isinstance(b.experts, str) for b in banks)
        for setting, on in (
            ("router_weighted=True", self.router_weighted),
            (f"all_token_blend={self.all_token_blend}", self.all_token_hessian),
        ):
            if on and not fused:
                raise Refused(
                    f"{self.name} {setting} requires fused expert banks; "
                    + ("the model has no routed experts" if not banks else "this model's are not fused")
                )

    @torch.no_grad()
    def apply_layer(self, ctx: LayerContext) -> None:
        """Solve every unit of the layer and write the served weights back."""
        for u in self.units(ctx):
            dev = ctx.device or u.params[0].device
            lo, hi = u.experts.indices(u.n_experts())[:2]
            if solve.chunk_experts(hi - lo, *u.shape(), solve.free_bytes(dev)) < hi - lo:
                self._solve_bank(u, dev, lo, hi)  # the whole bank does not fit the card
                continue
            u.write(*self._solve(u, dev))

    def _solve(
        self, u: BoundUnit, dev: torch.device, experts: slice | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """GPTQ this card's experts of a unit, or `experts`: its codes and the scales it solved against."""
        q = u.quantizer
        w = u.weight(experts).to(dev).float()
        h_raw, n = u.hessian(experts)
        h = solve.prepare_hessian(h_raw.to(dev), n)
        if self.all_token_hessian and u.part is not None:
            h = self._blend(u, h, n, experts, dev)
        w = self._target(u, w, h, n)
        hd = torch.diagonal(h, dim1=-2, dim2=-1).clone()
        # a format the clip does not serve takes the step's default clip
        clip = self.clip_for(q.grid.name) or CLIPS[self.clip]
        s, z = clip.choose(q, w, hd, grid_k=self.grid_k)
        # a bank, whole or in chunks, factors in the whole bank's batch however few experts it solves
        min_batch = solve.MIN_FACTOR_BATCH if u.part is not None else None
        codes = solve.gptq(
            w,
            h,
            q,
            s,
            z,
            damp=self.damp,
            actorder=self.actorder,
            block=self.block,
            impl=self.impl,
            fused_grid=self.fused_grid,
            min_batch=min_batch,
            label=u.key if u.part is None else f"{u.key}:{u.part}",  # `build.json`'s solve record
        )
        return codes, s, z

    def _blend(
        self,
        u: BoundUnit,
        h: torch.Tensor,
        n: torch.Tensor | int | None,
        experts: slice | None,
        dev: torch.device,
    ) -> torch.Tensor:
        """`h` (per token) blended with the all-token Hessian; an expert with no routed token takes the latter."""
        h_all, n_all = u.all_token_hessian(experts)
        h_all = solve.prepare_hessian(h_all.to(dev), n_all)
        routed = (
            torch.ones(h.shape[0], device=dev) if n is None else torch.as_tensor(n, device=dev).reshape(-1)
        )
        share = (1 - self.all_token_blend) * (routed > 0).float().reshape(-1, 1, 1)
        return share * h + (1 - share) * h_all

    def _target(
        self, u: BoundUnit, w: torch.Tensor, h: torch.Tensor, n: torch.Tensor | int | None
    ) -> torch.Tensor:
        """The fp32 weights the scale search and the solve round: the layer's own (GPTAQ refits them)."""
        del u, h, n
        return w

    def _solve_bank(self, u: BoundUnit, dev: torch.device, lo: int, hi: int) -> None:
        """Solve experts ``lo:hi`` of a bank in chunks that fit the card, writing each chunk as it lands.

        Only a chunk's workspace is on the card besides the bank's codes and scales, which are the output
        and are kept whole.
        """
        q, n, (rows, cols) = u.quantizer, hi - lo, u.shape()
        codes_all = torch.empty((n, rows, cols), dtype=torch.int8, device=dev)
        scale_all = torch.empty((n, rows, cols // q.group(cols)), dtype=torch.float32, device=dev)
        zero_all = None
        chunk = solve.chunk_experts(n, rows, cols, solve.free_bytes(dev))
        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)
            codes, s, z = self._solve(u, dev, slice(a, b))
            i = a - lo
            codes_all[i : i + b - a] = codes.to(torch.int8)
            scale_all[i : i + b - a] = s
            if z is not None:
                if zero_all is None:
                    zero_all = torch.empty_like(scale_all, dtype=z.dtype)
                zero_all[i : i + b - a] = z
            u.put_weights(q.dequant(codes, s, z), slice(a, b))
        u.record(codes_all, scale_all, zero_all)
