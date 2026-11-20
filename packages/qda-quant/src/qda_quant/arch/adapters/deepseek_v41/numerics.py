"""Torch prefill numerics cross-checked against DeepSeek-V4.1 and vLLM 0.33.

The cache round trips are part of the serving attention, including the RoPE tail.
They are deliberately separate from the checkpoint weight format.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Normalize each vector, accumulating in fp32 (fp64 during graph verification)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """Return a normalized vector in the input dtype."""
        h = x.double() if x.dtype == torch.float64 else x.float()
        return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x, positions, cfg, ratio, inverse=False):
    """Rotate adjacent pairs in the RoPE tail; compressed layers use YaRN."""
    rd = cfg.qk_rope_head_dim
    base = cfg.compress_rope_theta if ratio else cfg.rope_theta
    params = getattr(cfg, "rope_parameters", None) or {}
    params = params.get("compress" if ratio else "main", params)
    freq = base ** (-torch.arange(0, rd, 2, device=x.device, dtype=torch.float32) / rd)
    original = params.get("original_max_position_embeddings", 0) if ratio else 0
    if original:

        def corrected(rotations):
            return rd * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(params.get("beta_fast", 32))), 0)
        high = min(math.ceil(corrected(params.get("beta_slow", 1))), rd - 1)
        ramp = ((torch.arange(rd // 2, device=x.device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        freq = freq * (1 - ramp) + freq / params.get("factor", 1) * ramp
    angles = positions[..., None].float() * freq
    if x.ndim == 4:
        angles = angles.unsqueeze(-2)
    if inverse:
        angles = -angles
    tail = x[..., -rd:].float().unflatten(-1, (-1, 2))
    a, b = tail.unbind(-1)
    rotated = torch.stack((a * angles.cos() - b * angles.sin(), a * angles.sin() + b * angles.cos()), -1)
    return torch.cat((x[..., :-rd], rotated.flatten(-2).to(x.dtype)), -1)


def cache_roundtrip(x, bits, block, e4m3_scale=False):
    """Reference MXFP8 / MXFP4 / NVFP4 quantize-dequantize with nearest-even codes."""
    if x.dtype == torch.float64:  # fold verification measures algebra, not cache rounding
        return x
    width = x.shape[-1]
    groups = F.pad(x.float(), (0, -width % block)).unflatten(-1, (-1, block))
    max_code = 448 if bits == 8 else 6
    scale = (
        groups.abs().amax(-1, keepdim=True).clamp_min(max_code * 2 ** (-9 if e4m3_scale else -126)) / max_code
    )
    scale = scale.to(torch.float8_e4m3fn).float() if e4m3_scale else torch.exp2(torch.ceil(torch.log2(scale)))
    normalized = (groups / scale).clamp(-max_code, max_code)
    if bits == 8:
        codes = normalized.to(torch.float8_e4m3fn).float()
    else:
        # Midpoint ties choose an even e2m1 code, as CUDA's FP4 conversion does.
        levels = normalized.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        bounds = (levels[:-1] + levels[1:]) / 2
        code = torch.bucketize(normalized.abs().contiguous(), bounds)
        ties = (normalized.abs()[..., None] == bounds).any(-1) & (code % 2 == 1)
        code = code + ties
        codes = levels[code] * normalized.sign()
    return (codes * scale).flatten(-2)[..., :width].to(x.dtype)


def hc_mixes(x, fn, scale, base, cfg):
    """Produce delayed pre/post mixes and the doubly stochastic residual map."""
    h = x.double() if x.dtype == torch.float64 else x.float()
    flat = h.flatten(-2)
    mixed = F.linear(flat, fn.to(h.dtype)) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps
    )
    mult = cfg.hc_mult
    pre, post, comb = mixed.split([mult, mult, mult * mult], -1)
    pre = (pre * scale[0] + base[:mult]).sigmoid() + cfg.hc_eps
    post = 2 * (post * scale[1] + base[mult : 2 * mult]).sigmoid()
    comb = (comb * scale[2] + base[2 * mult :]).unflatten(-1, (mult, mult)).softmax(-1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    return pre, post, comb


def hc_pre(x, pre):
    """Collapse the stream with the previous sublayer's pre-mix."""
    h = x.double() if x.dtype == torch.float64 else x.float()
    return (h * pre.unsqueeze(-1)).sum(-2).to(x.dtype)


def hc_post(x, residual, post, comb):
    """Expand the sublayer output and mix input streams (source rows, destination columns)."""
    return (post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)).to(
        x.dtype
    )
