"""Token n-gram hashes and row-wise FP8 Engram lookup; tables never expand to bf16."""

import math

import numpy as np
import torch
from torch import nn


def compressed_token_map(tokenizer):
    """Reproduce training's normalization and compressed token numbering."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    keys, lookup = {}, []
    backend = tokenizer.backend_tokenizer
    for i in range(len(tokenizer)):
        text = backend.decode([i], skip_special_tokens=False)
        key = backend.id_to_token(i) if "\ufffd" in text else normalizer.normalize_str(text) or text
        lookup.append(keys.setdefault(key, len(keys)))
    return lookup, len(keys)


def bucket_layout(cfg):
    """Return disjoint prime-sized buckets for every Engram layer, n-gram and head."""
    seen, layers = set(), []
    for _ in cfg.engram_layer_ids:
        sizes = []
        for _ in range(cfg.engram_max_ngram_size - 1):
            current = cfg.engram_vocab_size - 1
            for _ in range(cfg.engram_n_heads):
                while True:
                    current += 1
                    if (
                        current not in seen
                        and current >= 2
                        and all(current % d for d in range(2, math.isqrt(current) + 1))
                    ):
                        break
                seen.add(current)
                sizes.append(current)
        layers.append(sizes)
    return layers


class EngramEmbedding(nn.Module):
    """Raw FP8 rows with one native E8M0 scale per 32 columns."""

    def __init__(self, rows, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.register_buffer(
            "weight_scale",
            torch.full((rows, (dim + 31) // 32), 1, dtype=torch.float8_e8m0fnu, device=self.weight.device),
        )

    def _apply(self, fn, recurse=True):  # noqa: ARG002
        # dtype conversions must leave the encoded table intact, including NaN scale bytes.
        device = fn(torch.empty(0, device=self.weight.device)).device
        if device.type == "cuda":
            device = torch.device("cpu")
        if self.weight.is_meta and device.type != "meta":
            weight = torch.empty(self.weight.shape, dtype=self.weight.dtype, device=device)
            scales = torch.empty(self.weight_scale.shape, dtype=self.weight_scale.dtype, device=device)
        else:
            weight, scales = self.weight.to(device), self.weight_scale.to(device)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = scales
        return self

    def forward(self, ids):
        """Read only selected rows, then decode their per-block exponents."""
        rows = ids.to(self.weight.device)
        values = self.weight[rows].to(device=ids.device, dtype=torch.float32)
        scales = self.weight_scale.view(torch.uint8)[rows].to(device=ids.device, dtype=torch.int32)
        scales = torch.exp2((scales - 127).float()).masked_fill(scales == 255, torch.nan)
        scales = scales.repeat_interleave(32, -1)[..., : values.shape[-1]]
        return (values * scales).to(torch.bfloat16)


class Engram(nn.Module):
    """Inject an n-gram memory value with the training model's signed-sqrt gate."""

    def __init__(self, cfg, layer_id, sizes):
        super().__init__()
        self.config = cfg
        self.layer_id, self.sizes = layer_id, tuple(sizes)
        index = cfg.engram_layer_ids.index(layer_id)
        self.embed_tokens = EngramEmbedding(cfg.engram_num_embeddings[index], cfg.engram_head_dim)
        self.wkv = nn.Linear(
            len(sizes) * cfg.engram_head_dim, cfg.hidden_size * (cfg.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.hidden_size))
        bound = max(1, (np.iinfo(np.int64).max // cfg.engram_compressed_vocab_size) // 2)
        gen = np.random.default_rng(10007 * layer_id)
        multipliers = gen.integers(0, bound, size=cfg.engram_max_ngram_size, dtype=np.int64) * 2 + 1
        self.hash_multipliers = tuple(int(v) for v in multipliers)
        self.offset_values = tuple(int(v) for v in np.cumsum([0, *sizes[:-1]]))
        if sum(sizes) > cfg.engram_num_embeddings[index]:
            raise ValueError("Engram table is smaller than its hash bucket layout")

    def hash_ids(self, ids, token_map):
        """Hash 2..N-grams, filling missing history with the compressed pad token."""
        cfg = self.config
        multipliers = torch.tensor(self.hash_multipliers, device=ids.device)
        primes = torch.tensor(self.sizes, device=ids.device)
        offsets = torch.tensor(self.offset_values, device=ids.device)
        tokens = token_map[ids]
        products = []
        for shift in range(cfg.engram_max_ngram_size):
            lookback = torch.nn.functional.pad(tokens, (shift, 0), value=0)[..., : ids.shape[-1]]
            missing = torch.arange(ids.shape[-1], device=ids.device) < shift
            lookback = torch.where(missing, token_map[cfg.engram_pad_token_id], lookback)
            products.append(lookback * multipliers[shift])
        rolling, hashes = products[0], []
        for i in range(1, len(products)):
            rolling = rolling ^ products[i]
            bucket_primes = primes[(i - 1) * cfg.engram_n_heads : i * cfg.engram_n_heads]
            hashes.append(rolling[..., None] % bucket_primes)
        return torch.cat(hashes, -1) + offsets

    def forward(self, x, input_ids, token_map):
        """Look up n-grams and gate each residual stream independently."""
        cfg = self.config
        rows = self.embed_tokens(self.hash_ids(input_ids, token_map))
        kv = self.wkv(rows.flatten(-2).to(x.dtype))
        key, value = kv.split([cfg.hc_mult * cfg.hidden_size, cfg.hidden_size], -1)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        h, key = x.to(dtype), key.unflatten(-1, (cfg.hc_mult, cfg.hidden_size)).to(dtype)
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.rms_norm_eps) * torch.rsqrt(
            key.square().mean(-1) + cfg.rms_norm_eps
        )
        dot = (h * self.q_weight * self.k_weight * key).sum(-1) * rstd * cfg.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.to(dtype).unsqueeze(-2)).to(x.dtype)
