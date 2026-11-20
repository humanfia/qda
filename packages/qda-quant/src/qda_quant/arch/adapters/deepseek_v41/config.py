"""DeepSeek-V4.1 text configuration, preserving the checkpoint's nested fields."""

from typing import Any

from transformers import PreTrainedConfig
from transformers.configuration_utils import SubConfigSpec


class DeepseekV41TextConfig(PreTrainedConfig):
    """Text-backbone shapes and numerics; defaults are the released Flash backbone."""

    model_type = "deepseek_v41_text"

    @property
    def num_experts(self):
        """Expert count for generic graph discovery, derived from the native field."""
        return self.n_routed_experts

    def __init__(self, **kwargs):
        defaults: dict[str, Any] = dict(
            vocab_size=129280,
            hidden_size=5120,
            moe_intermediate_size=2304,
            num_hidden_layers=40,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=1280,
            o_lora_rank=1024,
            o_groups=8,
            rms_norm_eps=1e-20,
            initializer_range=0.02,
            n_routed_experts=384,
            n_shared_experts=1,
            num_experts_per_tok=6,
            scoring_func="sqrtsoftplus",
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            hidden_act="silu",
            topk_method="noaux_tc",
            sliding_window=128,
            max_position_embeddings=1048576,
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
            candidate_source_layer_id=20,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[384006168, 384016682],
            engram_max_ngram_size=4,
            engram_vocab_size=16000000,
            engram_n_heads=8,
            engram_head_dim=256,
            engram_pad_token_id=2,
            engram_compressed_vocab_size=99092,
            tie_word_embeddings=False,
        )
        defaults.update(kwargs)
        rope = defaults.pop("rope_scaling", defaults.pop("rope_parameters", None))
        super().__init__(**defaults)
        if rope is not None:
            self.rope_parameters = rope


class DeepseekV41Config(PreTrainedConfig):
    """The original multimodal config; this model executes its text backbone only."""

    model_type = "deepseek_v41"
    sub_configs_defaults = {"text_config": SubConfigSpec(DeepseekV41TextConfig)}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        self.text_config = (
            text_config
            if isinstance(text_config, DeepseekV41TextConfig)
            else DeepseekV41TextConfig(**(text_config or {}))
        )
        self.vision_config = vision_config
        super().__init__(**kwargs)
