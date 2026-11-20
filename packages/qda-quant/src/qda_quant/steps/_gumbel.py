"""GSQ's Gumbel rounding and Lion updates, ported from IST-DASLab/GSQ.

The backward pass replays its Gumbel sample to avoid retaining five probabilities per weight.
Group-scale gradients use a fixed-order fp32 sum.

Source: IST-DASLab/GSQ, commit ``03fc16484c369e3127225615d5e03e8d3a6043e3``, by Alireza Dadgarnia,
Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtić, Max Kleinegger, and Dan Alistarh, under
Apache-2.0; see "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax
Sampling" (arXiv:2604.18556). The Lion update follows ``lion_pytorch`` 0.2.5, licensed under MIT.
See THIRD_PARTY_NOTICES.md for source links and the Lion license notice.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _sample(
    logits: torch.Tensor,
    shifts: torch.Tensor,
    valid: torch.Tensor,
    dtype: torch.dtype,
    temperature: float,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~valid, -1e9)
    uniform = torch.rand_like(masked)
    noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
    probabilities = ((masked * logit_scale + noise) / temperature).softmax(0).to(dtype)
    return probabilities, (probabilities * shifts[:, None, None]).sum(0)


class _SoftWeight(torch.autograd.Function):
    """Replay Gumbel samples in backward while preserving explicit mixed-precision arithmetic."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        scales: torch.Tensor,
        shifts: torch.Tensor,
        valid: torch.Tensor,
        initial: torch.Tensor,
        group: int,
        temperature: float,
        logit_scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(logits, scales)
        ctx.shifts, ctx.valid, ctx.initial = shifts, valid, initial
        ctx.group, ctx.temperature, ctx.logit_scale = group, temperature, logit_scale
        ctx.device = logits.device
        ctx.rng = torch.cuda.get_rng_state(logits.device) if logits.is_cuda else torch.get_rng_state()
        _, output = _sample(logits, shifts, valid, initial.dtype, temperature, logit_scale)
        return (initial + output) * scales.repeat_interleave(group, -1).to(output.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        logits, scales = ctx.saved_tensors
        devices = [ctx.device] if logits.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if logits.is_cuda:
                torch.cuda.set_rng_state(ctx.rng, ctx.device)
            else:
                torch.set_rng_state(ctx.rng)
            probabilities, output = _sample(
                logits, ctx.shifts, ctx.valid, ctx.initial.dtype, ctx.temperature, ctx.logit_scale
            )
        scale_columns = scales.repeat_interleave(ctx.group, -1).to(output.dtype)
        scale_gradient = (grad_output * (ctx.initial + output)).float()
        scale_gradient = scale_gradient.reshape(*scales.shape, ctx.group).sum(-1)
        frozen = (ctx.valid.sum(0) == 1).reshape(*scales.shape, ctx.group).all(-1)
        scale_gradient.masked_fill_(frozen, 0)
        code_gradient = (grad_output * scale_columns)[None] * ctx.shifts[:, None, None]
        dot = (code_gradient * probabilities).sum(0, keepdim=True)
        logit_gradient = probabilities * (code_gradient - dot) * ctx.logit_scale / ctx.temperature
        logit_gradient.masked_fill_(~ctx.valid, 0)
        return logit_gradient.float(), scale_gradient, None, None, None, None, None, None


class GumbelRounding(nn.Module):
    """Learn logits over five shifts of existing integer codes and fp32 group scales."""

    # torch types dynamic module attributes as `Tensor | Module`; name the tensors this module owns.
    logits: nn.Parameter
    scales: nn.Parameter
    initial: torch.Tensor
    shifts: torch.Tensor
    valid: torch.Tensor
    zero: torch.Tensor
    frozen: torch.Tensor

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        zero: torch.Tensor | None,
        group: int,
        qmin: int,
        qmax: int,
        std: float = 0.01,
        strength: float = 6.0,
        dtype: torch.dtype = torch.bfloat16,
        logits_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.shape, self.group = codes.shape, group
        codes = codes.reshape(-1, codes.shape[-1]).float()
        scales = scales.reshape(codes.shape[0], -1)
        offset = torch.zeros_like(scales) if zero is None else zero.reshape_as(scales).float()
        columns = offset.repeat_interleave(group, -1)
        initial = (codes - columns).to(dtype)
        shifts = torch.arange(-2, 3, device=codes.device, dtype=dtype)
        # Every five-shift array below is float32 over every weight of the unit: at a 32-expert bank unit's
        # shapes one is 2.97 GiB, so each temporary that outlives its one read is a first-order term of the
        # build. `candidates` was read once, by `valid`, and then held for the rest of the constructor beside
        # the logits chain (xy-2140 measured the constructor's peak at five of them alive at once, 17.2 GiB
        # for the unit); the chained forms are rewritten in place, with the same arithmetic: negation,
        # squaring, the strength rescale and the noise draw in the same order, so the built codes are bit
        # for bit what they were.
        candidates = initial[None] + shifts[:, None, None] + columns
        valid = (candidates >= qmin) & (candidates <= qmax)
        del candidates
        frozen = (scales == 0).repeat_interleave(group, -1)
        valid &= ~frozen[None] | (shifts == 0)[:, None, None]
        logits = (-shifts.to(torch.float32))[:, None, None].expand(5, *codes.shape).clone()
        logits.square_().mul_(-0.5)
        logits -= (logits * valid).sum(0, keepdim=True) / valid.sum(0, keepdim=True)
        logits.mul_(strength).add_(torch.randn_like(logits)).mul_(std)
        self.logits = nn.Parameter(logits.to(logits_dtype))
        self.scales = nn.Parameter(scales.float().detach().clone())
        self.register_buffer("initial", initial, persistent=False)
        self.register_buffer("shifts", shifts, persistent=False)
        self.register_buffer("valid", valid, persistent=False)
        self.register_buffer("zero", offset, persistent=False)
        self.register_buffer("frozen", scales == 0, persistent=False)

    def forward(self, temperature: float, logit_scale: float) -> torch.Tensor:
        """Sample differentiable weights on the annealed Gumbel distribution."""
        return _SoftWeight.apply(
            self.logits,
            self.scales,
            self.shifts,
            self.valid,
            self.initial,
            self.group,
            temperature,
            logit_scale,
        ).reshape(self.shape)

    @torch.no_grad()
    def hard_codes(self) -> torch.Tensor:
        """Return valid stored integer codes, with frozen zero points restored."""
        indices = self.logits.masked_fill(~self.valid, -1e9).argmax(0)
        return (self.initial + self.shifts[indices] + self.zero.repeat_interleave(self.group, -1)).reshape(
            self.shape
        )

    def reference(self, temperature: float, logit_scale: float) -> torch.Tensor:
        """Direct-autograd reference for the replay path, retaining the sampled probabilities.

        Compare gradients in fp32: the replay path uses explicit
        mixed-precision backward, including its bf16 rounding of intermediate products.
        """
        _, output = _sample(
            self.logits, self.shifts, self.valid, self.initial.dtype, temperature, logit_scale
        )
        scales = self.scales.masked_fill(self.frozen, 0)
        return ((self.initial + output) * scales.repeat_interleave(self.group, -1)).reshape(self.shape)


class Lion(torch.optim.Optimizer):
    """Lion updates used by GSQ, with independent parameter-group learning rates."""

    def __init__(self, params: list[dict], betas: tuple[float, float] = (0.9, 0.95)) -> None:
        super().__init__(params, {"betas": betas, "weight_decay": 0.0})

    @torch.no_grad()
    def step(self) -> None:
        """Apply decoupled weight decay, signed momentum updates, and momentum decay."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                update = momentum.clone().mul_(beta1).add_(parameter.grad, alpha=1 - beta1).sign_()
                parameter.add_(update, alpha=-group["lr"])
                momentum.mul_(beta2).add_(parameter.grad, alpha=1 - beta2)


def cosine_lr(base: float, step: int, total: int, warmup: int, minimum: float) -> float:
    """Return the GSQ cosine learning rate, including warmup and its last-step endpoint."""
    if step < warmup:
        return base * (minimum + (1 - minimum) * step / warmup)
    decay_steps = total - 1 - warmup
    progress = 1 if decay_steps == 0 else (step - warmup) / decay_steps
    return base * (minimum + 0.5 * (1 - minimum) * (1 + math.cos(math.pi * progress)))
