"""Tiny random models from real transformers configs (CPU, seconds)."""

import torch
import transformers

SMALL = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=128,
    max_position_embeddings=64,
)


def config(kind: str):
    if kind == "qwen3":
        return transformers.Qwen3Config(**SMALL)
    if kind == "qwen3-mha":
        return transformers.Qwen3Config(**{**SMALL, "num_key_value_heads": 4})
    if kind == "llama":
        return transformers.LlamaConfig(**SMALL)
    # 32 layers, R1-Distill-Llama-8B's depth, so a recipe's layer-indexed format overrides all match
    if kind == "llama32":
        return transformers.LlamaConfig(**{**SMALL, "num_hidden_layers": 32})
    if kind == "opt":  # LayerNorms with a bias: the fold must scale the bias as well as the weight
        return transformers.OPTConfig(
            hidden_size=128,
            ffn_dim=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=128,
            max_position_embeddings=64,
            do_layer_norm_before=True,
        )
    if kind == "qwen2":  # q/k/v biases
        return transformers.Qwen2Config(**SMALL)
    if kind == "llama-bias":  # biases on every linear
        return transformers.LlamaConfig(**SMALL, attention_bias=True, mlp_bias=True)
    if kind == "qwen3_moe":
        return transformers.Qwen3MoeConfig(
            **SMALL,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    if kind == "qwen3_5":  # Gated DeltaNet hybrid, zero-centred (1 + w) norms
        return transformers.Qwen3_5TextConfig(
            **{**SMALL, "num_hidden_layers": 4},
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
        )
    if (
        kind == "qwen3_5_36"
    ):  # Qwen3.8-27B's depth and layer types (full attention every 4th), so its layer-indexed format
        # overrides all match: layers 3, 7, ..., 35 are full attention, the rest Gated DeltaNet
        return transformers.Qwen3_5TextConfig(
            **{**SMALL, "num_hidden_layers": 36},
            layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(36)],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
        )
    if (
        kind == "qwen3_5-heads32"
    ):  # Gated DeltaNet with as many value heads as Qwen3.5-9B: in_proj_a/b are 32 wide
        return transformers.Qwen3_5TextConfig(
            **{**SMALL, "num_hidden_layers": 2},
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=4,
            linear_num_value_heads=32,
            linear_key_head_dim=32,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
        )
    if kind == "qwen2_moe":  # routed experts plus a shared expert and its scalar gate
        return transformers.Qwen2MoeConfig(
            **SMALL,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    if kind == "qwen3_5_moe":  # Gated DeltaNet hybrid with fused routed experts and a shared expert
        return transformers.Qwen3_5MoeTextConfig(
            **{**SMALL, "num_hidden_layers": 2},
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,  # as a real checkpoint's config carries; transformers 5 derives it
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
        )
    if kind == "deepseek_v2":  # MLA attention, one dense layer then fused routed + shared experts
        # MLA has one latent KV head: num_key_value_heads must equal num_attention_heads.
        # kv_lora_rank and v_head_dim keep every quantized input width a multiple of 64 (the recipe's group).
        return transformers.DeepseekV2Config(
            **{**SMALL, "num_key_value_heads": SMALL["num_attention_heads"]},
            q_lora_rank=None,
            kv_lora_rank=64,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            moe_intermediate_size=64,
            n_routed_experts=8,
            n_shared_experts=2,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
        )
    if kind == "gemma3_vlm":  # multimodal: SigLIP vision tower + Gemma 3 text model, checkpoint names differ
        text = {**SMALL, "vocab_size": 300}
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                      image_size=28, patch_size=14)  # fmt: skip
        return transformers.Gemma3Config(
            text_config=text,
            vision_config=vision,
            mm_tokens_per_image=4,
            boi_token_index=297,
            eoi_token_index=298,
            image_token_index=299,
        )
    # sliding + full attention; MoE adds an expert bank to the MLP; ``gemma4_keqv``'s full layer shares K/V
    if kind in ("gemma4", "gemma4_moe", "gemma4_keqv"):
        return transformers.Gemma4TextConfig(
            **SMALL,
            global_head_dim=32,
            num_global_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"],
            enable_moe_block=kind == "gemma4_moe",
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
            hidden_size_per_layer_input=0,
            # the full-attention layer shares K and V, as Gemma 4 26B's five full layers do: no ``v_proj``
            attention_k_eq_v=kind == "gemma4_keqv",
        )
    if (
        kind == "gemma4_moe_keqv30"
    ):  # Gemma-4-26B-A4B: the multimodal wrapper, whose text tower is ``model.language_model`` as the
        # checkpoint names it, at the model's depth: 30 layers, full attention every 6th (layers 5, 11, 17, 23,
        # 29) with K/V shared, so its layer-indexed format overrides (q/k/v/o and the dense MLP by layer) all
        # match and those five layers need no v_proj
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      head_dim=16, patch_size=16, image_size=32, pooling_kernel_size=2,
                      position_embedding_size=16)  # fmt: skip
        return transformers.Gemma4Config(
            text_config=dict(
                **{**SMALL, "num_hidden_layers": 30},
                global_head_dim=32,
                num_global_key_value_heads=2,
                layer_types=[
                    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(30)
                ],
                enable_moe_block=True,
                num_experts=4,
                top_k_experts=2,
                moe_intermediate_size=64,
                hidden_size_per_layer_input=0,
                attention_k_eq_v=True,
            ),
            vision_config=vision,
            boi_token_id=120,
            eoi_token_id=121,
            image_token_id=122,
        )
    if kind == "olmo2":  # post-norm blocks: the layer norms scale the attention and MLP outputs
        return transformers.Olmo2Config(**SMALL)
    # Mamba2/attention hybrid; routed experts are a fused non-gated up/down bank. "-nano": Nemotron-3-Nano-30B-A3B's
    # 52-layer pattern, so a recipe's layer-indexed format overrides match.
    if kind in ("nemotron_h", "nemotron_h-nano"):
        pattern = "MEM*" if kind == "nemotron_h" else "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        return transformers.NemotronHConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=len(pattern),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
            moe_intermediate_size=128,
            moe_shared_expert_intermediate_size=256,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
            mamba_num_heads=8,
            mamba_head_dim=32,
            ssm_state_size=16,
            n_groups=2,
            conv_kernel=4,
            expand=2,
            chunk_size=8,
            hybrid_override_pattern=pattern,
            tie_word_embeddings=False,
        )
    if kind == "gpt_oss":  # fused, transposed expert banks with biases; interleaved gate/up rows
        return transformers.GptOssConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=256,
        )
    if kind == "deepseek_v4":  # hash-routed MoE, hyper-connections, one layer of each attention kind
        return transformers.DeepseekV4Config(
            vocab_size=128,
            hidden_size=128,  # every quantized linear is a multiple of the w4a16 group (128)
            moe_intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=64,
            q_lora_rank=128,
            num_experts_per_tok=2,
            n_routed_experts=8,
            o_groups=2,
            o_lora_rank=64,
            index_n_heads=2,
            index_head_dim=64,
            index_topk=2,
            hc_mult=2,
            sliding_window=8,
            max_position_embeddings=64,
            layer_types=[
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
                "compressed_sparse_attention",
            ],
            mlp_layer_types=["hash_moe", "moe", "hash_moe", "moe"],
        )
    raise ValueError(kind)


def model(kind: str, dtype=torch.float32, meta: bool = False):
    cfg = config(kind)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"  # grouped_mm needs bf16
    torch.manual_seed(0)
    if meta:
        with torch.device("meta"):
            return transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    with torch.no_grad():  # random norms and biases so a wrong fold or a dropped bias cannot pass by accident
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
            elif n.endswith(".bias"):
                p.normal_(0, 1)
        if kind == "gpt_oss":  # from_config leaves the fused expert banks and the router at their empty init
            for layer in m.model.layers:
                ex = layer.mlp.experts
                ex.gate_up_proj.normal_(0, 0.1)
                ex.down_proj.normal_(0, 0.1)
                layer.mlp.router.weight.normal_(0, 0.5)
    return m
