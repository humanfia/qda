"""CSA2 prefill: local KV plus source-owned compressed KV and reused sparse indices."""

import torch
from torch import nn
from torch.nn import functional as F

from .numerics import RMSNorm, rope
from .prefill import PrefillNumerics


class GroupedLinear(nn.Linear):
    """Block-diagonal output projection over head groups, with checkpoint-compatible rows."""

    def __init__(self, cfg):
        self.n_groups = cfg.o_groups
        super().__init__(
            cfg.num_attention_heads * cfg.head_dim // cfg.o_groups, cfg.o_groups * cfg.o_lora_rank, bias=False
        )

    def forward(self, x):
        """Project each head group with its own rows."""
        weight = self.weight.unflatten(0, (self.n_groups, -1))
        return torch.einsum("bsgd,grd->bsgr", x, weight).flatten(-2)


class Compressor(nn.Module):
    """Source-layer KV projection, optionally pooled over pairs in fp32."""

    def __init__(self, cfg, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)
        self.norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x):
        """Produce normalized pre-RoPE latents for complete compression groups."""
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        cutoff = x.shape[1] - x.shape[1] % self.ratio
        h = x[:, :cutoff].to(torch.float64 if x.dtype == torch.float64 else torch.float32)
        kv = F.linear(h, self.wkv.weight.to(h.dtype)).unflatten(1, (-1, self.ratio))
        score = F.linear(h, self.wgate.weight.to(h.dtype)).unflatten(1, (-1, self.ratio))
        return self.norm((kv * score.softmax(2)).sum(2).to(x.dtype))


class Indexer(nn.Module):
    """Rank compressed positions; decoder-side sources can reuse an encoder's index keys."""

    def __init__(self, cfg, owns_k):
        super().__init__()
        self.config, self.owns_k = cfg, owns_k
        self.prefill: PrefillNumerics | None = None
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.hidden_size, cfg.index_n_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False)
            self.k_norm = RMSNorm(cfg.index_head_dim, cfg.rms_norm_eps)

    def forward(self, x, qr, latent, positions, ratio, carry, layer_id):
        """Publish fresh index keys/top-k indices and, on the source, candidate blocks."""
        if self.prefill is None:
            raise ValueError("DeepSeek-V4.1 indexer needs target prefill numerics")
        cfg = self.config
        if self.owns_k:
            key = rope(
                self.k_norm(self.wk(latent)), positions[:, : latent.shape[1] * ratio : ratio], cfg, ratio
            )
            carry["index_k"] = self.prefill.index(key)
        q = rope(self.wq_b(qr).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim)), positions, cfg, ratio)
        q = self.prefill.index(q)
        weights = self.weights_proj(x).float() * (cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5)
        scores = (
            torch.einsum("bshd,btd->bsht", q.float(), carry["index_k"].float()).relu()
            * weights.float().unsqueeze(-1)
        ).sum(2)
        width = scores.shape[-1]
        lens = (torch.arange(1, x.shape[1] + 1, device=x.device) // ratio)[None, :, None]
        visible = torch.arange(width, device=x.device) < lens
        scores = scores.masked_fill(~visible, -torch.inf)
        if layer_id == cfg.candidate_source_layer_id and cfg.candidate_topk_blocks:
            block = cfg.candidate_block_size
            blocks = F.pad(scores, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(-1)
            last = (lens - 1) // block
            blocks = blocks.masked_fill(torch.arange(blocks.shape[-1], device=x.device) == last, torch.inf)
            top = blocks.topk(min(cfg.candidate_topk_blocks, blocks.shape[-1]), -1)
            mask = torch.zeros_like(blocks, dtype=torch.bool).scatter(
                -1, top.indices, top.values > -torch.inf
            )
            carry["candidates"] = mask.repeat_interleave(block, -1)[..., :width]
        elif 0 <= cfg.candidate_source_layer_id < layer_id and cfg.candidate_topk_blocks:
            scores = scores.masked_fill(~carry["candidates"], -torch.inf)
        indices = scores.topk(min(cfg.index_topk, width), -1, sorted=False).indices.sort(-1).values
        carry["indices"] = torch.where(visible.expand_as(scores).gather(-1, indices), indices, -1)


class Attention(nn.Module):
    """CSA2 Full, Reindex and Reuse modes, derived entirely from config source-layer IDs."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        self.config, self.layer_id = cfg, layer_id
        self.prefill: PrefillNumerics | None = None
        self.ratio = cfg.compress_ratios[layer_id]
        if self.ratio not in (0, 1, 2):
            raise ValueError(f"CSA2 layer {layer_id}: unsupported compression ratio {self.ratio}")
        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.wo_a = GroupedLinear(cfg)
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(cfg.num_attention_heads, dtype=torch.float32))
        owns_k = layer_id in cfg.kv_source_layer_ids
        owns_index = layer_id in cfg.index_source_layer_ids
        if self.ratio and (
            not any(i <= layer_id for i in cfg.kv_source_layer_ids)
            or not any(i <= layer_id for i in cfg.index_source_layer_ids)
        ):
            raise ValueError(f"CSA2 layer {layer_id}: compressed attention has no preceding source")
        if owns_k and not owns_index:
            raise ValueError(f"CSA2 layer {layer_id}: KV source must also produce index keys")
        self.compressor = Compressor(cfg, self.ratio) if owns_k else None
        self.indexer = Indexer(cfg, owns_k) if owns_index else None

    def forward(self, x, positions, carry):
        """Attend causally over local and selected compressed positions; update source state."""
        if self.prefill is None:
            raise ValueError("DeepSeek-V4.1 attention needs target prefill numerics")
        cfg, ratio = self.config, self.ratio
        qr = self.q_norm(self.wq_a(x))
        q = rope(self.wq_b(qr).unflatten(-1, (cfg.num_attention_heads, cfg.head_dim)), positions, cfg, ratio)
        kv = self.prefill.kv(rope(self.kv_norm(self.wkv(x)), positions, cfg, ratio), cfg)
        n = x.shape[1]
        query, key = torch.arange(n, device=x.device)[:, None], torch.arange(n, device=x.device)[None]
        mask = ((key <= query) & (key > query - cfg.sliding_window))[None].expand(x.shape[0], -1, -1)
        if ratio:
            latent = self.compressor(x) if self.compressor is not None else None
            if self.indexer is not None:
                self.indexer(x, qr, latent, positions, ratio, carry, self.layer_id)
            if latent is not None:
                compressed = rope(latent, positions[:, : latent.shape[1] * ratio : ratio], cfg, ratio)
                carry["compressed_kv"] = self.prefill.compressed(compressed, cfg)
            compressed = carry["compressed_kv"]
            indices = carry["indices"]
            extra = torch.zeros((*mask.shape[:2], compressed.shape[1] + 1), device=x.device, dtype=torch.bool)
            extra = extra.scatter(-1, indices + 1, indices >= 0)[..., 1:]
            mask = torch.cat((mask, extra), -1)
            kv = torch.cat((kv, compressed), 1)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        scores = torch.einsum("bshd,btd->bsht", q.to(dtype), kv.to(dtype)) * cfg.head_dim**-0.5
        scores = scores.masked_fill(~mask.unsqueeze(2), -torch.inf)
        sink = self.attn_sink[:, None].expand(*scores.shape[:-1], 1)
        probs = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        out = torch.einsum("bsht,btd->bshd", probs, kv.to(dtype)).to(x.dtype)
        out = (
            rope(out, positions, cfg, ratio, inverse=True)
            .unflatten(-2, (cfg.o_groups, cfg.num_attention_heads // cfg.o_groups))
            .flatten(-2)
        )
        out = self.prefill.output(out)
        return self.wo_b(self.wo_a(out))
