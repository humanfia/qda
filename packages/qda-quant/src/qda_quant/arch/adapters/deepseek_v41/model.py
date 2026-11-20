"""A torch-only DeepSeek-V4.1 text model for layer-wise calibration.

Module names follow the native checkpoint, apart from the standard HF root embedding and head.
No KV cache, MTP, image processing or inference kernels are required.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .attention import Attention
from .config import DeepseekV41Config
from .engram import Engram, EngramEmbedding, bucket_layout, compressed_token_map
from .numerics import RMSNorm, hc_mixes, hc_post, hc_pre


class Gate(nn.Module):
    """Selection bias affects expert IDs; normalized routing weights use unbiased scores."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        self.bias = nn.Parameter(torch.zeros(cfg.n_routed_experts, dtype=torch.float32))

    def forward(self, x):
        """Return routing weights and expert IDs."""
        cfg = self.config
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        scores = F.linear(x.to(dtype), self.weight.to(dtype))
        if cfg.scoring_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif cfg.scoring_func == "softmax":
            scores = scores.softmax(-1)
        elif cfg.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise ValueError(f"unsupported DeepSeek routing score {cfg.scoring_func!r}")
        indices = (scores + self.bias).topk(cfg.num_experts_per_tok, -1).indices
        weights = scores.gather(-1, indices)
        if cfg.norm_topk_prob and cfg.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * cfg.routed_scaling_factor, indices


class Expert(nn.Module):
    """SwiGLU expert with the native clamp and routing weight applied before the down projection."""

    def __init__(self, cfg, intermediate=None):
        super().__init__()
        self.limit = cfg.swiglu_limit
        intermediate = intermediate or cfg.moe_intermediate_size
        self.w1 = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, cfg.hidden_size, bias=False)

    def forward(self, x, weights=None):
        """Apply gate/up, optional token weights, and down projection."""
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        gate, up = self.w1(x).to(dtype), self.w3(x).to(dtype)
        if self.limit > 0:
            gate, up = gate.clamp_max(self.limit), up.clamp(-self.limit, self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights
        return self.w2(hidden.to(x.dtype))


class RoutedExperts(nn.Module):
    """Fused storage with the native per-expert arithmetic and accumulation order."""

    def __init__(self, cfg):
        super().__init__()
        self.limit = cfg.swiglu_limit
        self.num_experts = cfg.n_routed_experts
        self.intermediate = cfg.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate, cfg.hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, cfg.hidden_size, self.intermediate))
        self.act_fn = F.silu

    def down_input(self, gate_up: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        """Native down input for both forward and statistics replay, including router weights."""
        dtype = torch.float64 if gate_up.dtype == torch.float64 else torch.float32
        gate, up = gate_up.to(dtype).chunk(2, dim=-1)
        if self.limit > 0:
            gate, up = gate.clamp_max(self.limit), up.clamp(-self.limit, self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights.unsqueeze(-1)
        return hidden.to(gate_up.dtype)

    def forward(self, x, top_k_index, top_k_weights):
        """Apply routing weights before down projection, keeping gate/up GEMMs separate."""
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        out = torch.zeros_like(x, dtype=dtype)
        for i in range(self.num_experts):
            rows, picks = torch.where(top_k_index == i)
            gate_weight, up_weight = self.gate_up_proj[i].split(self.intermediate)
            gate = F.linear(x[rows], gate_weight)
            up = F.linear(x[rows], up_weight)
            hidden = self.down_input(torch.cat((gate, up), -1), top_k_weights[rows, picks])
            value = F.linear(hidden, self.down_proj[i])
            out.index_add_(0, rows, value.to(dtype))
        return out


class MoE(nn.Module):
    """Native routing with one fused bank and a separate shared expert."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.gate = Gate(cfg)
        self.experts = RoutedExperts(cfg)
        self.shared_experts = Expert(cfg, cfg.moe_intermediate_size * cfg.n_shared_experts)

    def forward(self, x):
        """Sum routed and shared expert outputs."""
        shape = x.shape
        h = x.flatten(0, -2)
        weights, indices = self.gate(h)
        if h.is_meta:
            return self.shared_experts(x) + torch.zeros_like(x)
        out = self.experts(h, indices, weights)
        return (out + self.shared_experts(h)).to(x.dtype).view(shape)


class DecoderLayer(nn.Module):
    """A hyper-connected block with explicit, immutable per-batch attention carry."""

    def __init__(self, cfg, layer_id, sizes=None):
        super().__init__()
        self.config = cfg
        self.layer_id = layer_id
        self.attn = Attention(cfg, layer_id)
        self.ffn = MoE(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.engram = Engram(cfg, layer_id, sizes) if sizes is not None else None
        width = (2 + cfg.hc_mult) * cfg.hc_mult
        for kind in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{kind}_fn",
                nn.Parameter(torch.zeros(width, cfg.hc_mult * cfg.hidden_size, dtype=torch.float32)),
            )
            self.register_parameter(f"hc_{kind}_base", nn.Parameter(torch.zeros(width, dtype=torch.float32)))
            self.register_parameter(f"hc_{kind}_scale", nn.Parameter(torch.ones(3, dtype=torch.float32)))

    def forward(self, hidden_states, *, positions, input_ids, token_map, carry):
        """Return the full HC stream and the state needed by the next layer."""
        cfg = self.config
        state = dict(carry)
        x = hidden_states
        if self.engram is not None:
            x = self.engram(x, input_ids, token_map)
        pre, post, comb = hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, cfg)
        attn = self.attn(self.attn_norm(hc_pre(x, state["pre_mix"])), positions, state)
        x = hc_post(attn, x, post, comb)
        ffn_pre, post, comb = hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, cfg)
        ffn = self.ffn(self.ffn_norm(hc_pre(x, pre)))
        x = hc_post(ffn, x, post, comb)
        state["pre_mix"] = ffn_pre
        return x, state


class TextModel(nn.Module):
    """Standard HF-shaped embedding, layer list and final norm."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        sizes = dict(zip(cfg.engram_layer_ids, bucket_layout(cfg), strict=True))
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i, sizes.get(i)) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.register_buffer("token_map", torch.arange(cfg.vocab_size), persistent=False)


class DeepseekV41ForCausalLM(PreTrainedModel):
    """Text-only prefill model, registered by the DeepSeek-V4.1 architecture plugin."""

    config_class = DeepseekV41Config
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer"]
    _keep_in_fp32_modules_strict = [
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "attn_sink",
        "bias",
    ]

    def __init__(self, config):
        super().__init__(config)
        cfg = config.text_config
        self.model = TextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._token_map_ready = not cfg.engram_layer_ids or cfg.engram_compressed_vocab_size == cfg.vocab_size
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.text_config.initializer_range)
        elif isinstance(module, EngramEmbedding):
            if not module.weight.is_meta:
                module.weight.data.copy_(
                    torch.randn(module.weight.shape, device=module.weight.device).to(module.weight.dtype)
                )
                module.weight_scale.fill_(1)
        elif isinstance(module, Gate):
            module.weight.data.normal_(std=self.config.text_config.initializer_range)
        elif isinstance(module, RoutedExperts):
            module.gate_up_proj.data.normal_(std=self.config.text_config.initializer_range)
            module.down_proj.data.normal_(std=self.config.text_config.initializer_range)

    def get_input_embeddings(self):
        """Return the ordinary token embedding, separate from Engram tables."""
        return self.model.embed_tokens

    def get_output_embeddings(self):
        """Return the language-model head."""
        return self.lm_head

    def set_tokenizer(self, tokenizer):
        """Derive and validate the Engram map from this checkpoint's tokenizer."""
        lookup, size = compressed_token_map(tokenizer)
        cfg = self.config.text_config
        if size != cfg.engram_compressed_vocab_size or len(lookup) != cfg.vocab_size:
            raise ValueError(
                f"Engram tokenizer vocabulary {size}/{len(lookup)} differs from config {cfg.engram_compressed_vocab_size}/{cfg.vocab_size}"
            )
        self.model.token_map = torch.tensor(lookup, device=self.model.embed_tokens.weight.device)
        self._token_map_ready = True

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        """Compute all prefill logits; padding, cache and vision inputs are refused."""
        if use_cache:
            raise ValueError("DeepSeek-V4.1 calibration supports prefill without a KV cache")
        if not hasattr(self, "prefill_numerics"):
            raise ValueError(
                "DeepSeek-V4.1 prefill numerics need an explicit build target; configure the architecture adapter before forward"
            )
        if kwargs:
            raise TypeError(f"unsupported DeepSeek-V4.1 prefill arguments: {sorted(kwargs)}")
        if (
            attention_mask is not None
            and not input_ids.is_meta
            and (attention_mask.ndim != 2 or not bool(attention_mask.all()))
        ):
            raise ValueError("DeepSeek-V4.1 prefill calibration requires unpadded token rows")
        if not self._token_map_ready and not input_ids.is_meta:
            from ....intake.tokenizer import load_tokenizer

            self.set_tokenizer(load_tokenizer(self.config._name_or_path))
        cfg = self.config.text_config
        if cfg.engram_compressed_vocab_size == cfg.vocab_size:
            self.model.token_map = torch.arange(cfg.vocab_size, device=input_ids.device)
        h = self.model.embed_tokens(input_ids).unsqueeze(-2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
        pre = h.new_zeros((*input_ids.shape, cfg.hc_mult), dtype=torch.float32)
        pre[..., 0] = 1
        carry = {"pre_mix": pre}
        positions = torch.arange(input_ids.shape[-1], device=input_ids.device)[None].expand_as(input_ids)
        for layer in self.model.layers:
            h, carry = layer(
                h, positions=positions, input_ids=input_ids, token_map=self.model.token_map, carry=carry
            )
        h = self.model.norm(hc_pre(h, carry["pre_mix"]))
        return CausalLMOutput(logits=self.lm_head(h).float())
