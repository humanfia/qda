"""The Q4_K_M mixing rule against types and order ``llama-quantize`` (llama.cpp 9e2a04a) wrote for a 32-layer Llama.

Golden values come from ``llama-quantize model-bf16.gguf out.gguf Q4_K_M`` on tiny 32-layer Llamas (hidden 256,
FFN 512, tied and untied); no llama.cpp is needed here. They pin the ``use_more_bits`` pattern, the counter order
(numeric layer order, ``blk.10`` after ``blk.9``) and the tied-embedding case.
"""

import pytest
from qda_quant.gguf.rule import (
    BF16,
    F32,
    Q4_K,
    Q5_0,
    Q6_K,
    Q8_0,
    expert_count,
    loader_order,
    q4_k_m_types,
)
from qda_quant.step import Refused

N = 32
#: layers whose attn_v and ffn_down llama-quantize stored as Q6_K
GOLDEN_Q6K_LAYERS = {0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31}


def _tensors(tied: bool, width: int = 256) -> list[tuple[str, tuple[int, ...], int]]:
    """Convert's tensor list (its file order: output first, then per layer) for a tiny Llama."""
    t = [] if tied else [("output.weight", (width, 1000), BF16)]
    t += [("token_embd.weight", (width, 1000), BF16), ("output_norm.weight", (width,), F32)]
    for i in range(N):
        t += [
            (f"blk.{i}.attn_norm.weight", (width,), F32),
            (f"blk.{i}.ffn_down.weight", (512, width), BF16),
            (f"blk.{i}.ffn_gate.weight", (width, 512), BF16),
            (f"blk.{i}.ffn_up.weight", (width, 512), BF16),
            (f"blk.{i}.ffn_norm.weight", (width,), F32),
            (f"blk.{i}.attn_k.weight", (width, 128), BF16),
            (f"blk.{i}.attn_output.weight", (width, width), BF16),
            (f"blk.{i}.attn_q.weight", (width, width), BF16),
            (f"blk.{i}.attn_v.weight", (width, 128), BF16),
        ]
    return t


@pytest.mark.parametrize("tied", [False, True])
def test_q4_k_m_types_match_llama_quantize(tied):
    types = q4_k_m_types("llama", N, _tensors(tied))
    q6k = {n for n, t in types.items() if t == Q6_K}
    want = {f"blk.{i}.{k}.weight" for i in GOLDEN_Q6K_LAYERS for k in ("attn_v", "ffn_down")}
    want.add("token_embd.weight" if tied else "output.weight")
    assert q6k == want
    assert {n for n, t in types.items() if t == F32} == {n for n, _, t in _tensors(tied) if t == F32}
    assert all(t == Q4_K for n, t in types.items() if n not in q6k and not n.endswith("norm.weight"))


def test_loader_order_is_numeric_by_layer():
    names = loader_order([n for n, _, _ in _tensors(False)])
    assert names[:3] == ["output.weight", "output_norm.weight", "token_embd.weight"]
    assert names.index("blk.10.attn_k.weight") > names.index("blk.9.ffn_up.weight")
    assert names[3:6] == ["blk.0.attn_k.weight", "blk.0.attn_norm.weight", "blk.0.attn_output.weight"]


def test_rows_of_partial_super_blocks_fall_back_like_llama_quantize():
    """Golden: llama-quantize on an 8-layer Llama with hidden 320 (FFN 768, so ffn_down rows stay 768 wide)."""
    types = q4_k_m_types(
        "llama",
        8,
        [
            (n, ne, t)
            for n, ne, t in _tensors(False, width=320)
            if not n.startswith(tuple(f"blk.{i}." for i in range(8, N)))
        ],
    )
    q6k_layers = {0, 3, 6, 7}  # use_more_bits(i, 8)
    assert types["output.weight"] == Q8_0 and types["token_embd.weight"] == Q5_0
    for i in range(8):
        assert types[f"blk.{i}.attn_v.weight"] == (Q8_0 if i in q6k_layers else Q5_0)
        assert types[f"blk.{i}.ffn_down.weight"] == (
            Q6_K if i in q6k_layers else Q4_K
        )  # 512 wide in this list
        assert {
            types[f"blk.{i}.{k}.weight"] for k in ("attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up")
        } == {Q5_0}


def test_refuses_rows_llama_cpp_stores_in_f16():
    with pytest.raises(Refused, match="F16, not ported"):
        q4_k_m_types("llama", N, _tensors(False, width=300))


QWEN35_N = 8
#: ``qwen35`` puts a full-attention block every ``full_attention_interval``; the rest are gated delta net.
QWEN35_FULL = {3, 7}


def _qwen35_tensors(width: int = 256) -> list[tuple[str, tuple[int, ...], int]]:
    """A tiny ``qwen35``: GDN blocks carry a fused ``attn_qkv`` plus ``ssm_*``, full-attention blocks q/k/v/o."""
    t = [
        ("output.weight", (width, 1000), BF16),
        ("token_embd.weight", (width, 1000), BF16),
        ("output_norm.weight", (width,), F32),
    ]
    for i in range(QWEN35_N):
        t += [
            (f"blk.{i}.attn_norm.weight", (width,), F32),
            (f"blk.{i}.ffn_down.weight", (512, width), BF16),
            (f"blk.{i}.ffn_gate.weight", (width, 512), BF16),
            (f"blk.{i}.ffn_up.weight", (width, 512), BF16),
        ]
        if i in QWEN35_FULL:
            t += [
                (f"blk.{i}.attn_k.weight", (width, 128), BF16),
                (f"blk.{i}.attn_output.weight", (width, width), BF16),
                (f"blk.{i}.attn_q.weight", (width, width), BF16),
                (f"blk.{i}.attn_v.weight", (width, 128), BF16),
            ]
        else:
            t += [
                (f"blk.{i}.attn_gate.weight", (width, 256), BF16),
                (f"blk.{i}.attn_qkv.weight", (width, 320), BF16),
                (f"blk.{i}.ssm_a", (48,), F32),
                (f"blk.{i}.ssm_alpha.weight", (width, 48), BF16),
                (f"blk.{i}.ssm_beta.weight", (width, 48), BF16),
                (f"blk.{i}.ssm_conv1d.weight", (4, 576), BF16),
                (f"blk.{i}.ssm_dt.bias", (48,), F32),
                (f"blk.{i}.ssm_norm.weight", (128,), F32),
                (f"blk.{i}.ssm_out.weight", (512, width), BF16),
            ]
    return t


def test_qwen35_fused_qkv_shares_the_attn_v_counter():
    """The fused ``attn_qkv`` is an attn_v for `use_more_bits`, so both kinds advance one counter.

    Eight attn_v-like tensors in loader order: GDN ``attn_qkv`` at layers 0,1,2,4,5,6 and ``attn_v`` at 3 and 7.
    ``use_more_bits(i, 8)`` promotes i in {0, 3, 6, 7}, which lands on two of each kind. Giving ``attn_qkv`` its
    own counter, or leaving it out of the total, moves the promotions onto different layers.

    Golden: on a tiny qwen35, ``llama-quantize ... Q4_K_M`` stores attn_qkv as Q6_K twice and Q4_K four times,
    and both attn_v as Q6_K, which is what this rule returns.
    """
    types = q4_k_m_types("qwen35", QWEN35_N, _qwen35_tensors())
    assert types["blk.0.attn_qkv.weight"] == Q6_K
    assert types["blk.6.attn_qkv.weight"] == Q6_K
    assert types["blk.3.attn_v.weight"] == Q6_K
    assert types["blk.7.attn_v.weight"] == Q6_K
    assert {types[f"blk.{i}.attn_qkv.weight"] for i in (1, 2, 4, 5)} == {Q4_K}


def test_qwen35_gdn_state_space_tensors():
    """``ssm_conv1d`` keeps the template's type; the 2-D ``ssm_*`` weights are stored Q4_K.

    Golden: ``llama-quantize tiny-bf16.gguf out.gguf Q4_K_M`` (llama.cpp 9e2a04a) on a tiny qwen35 agrees with
    this rule on all 109 tensors. ``ssm_conv1d``'s rows are the conv kernel, so it is the one that must stay.
    """
    src = {n: t for n, _, t in _qwen35_tensors()}
    types = q4_k_m_types("qwen35", QWEN35_N, _qwen35_tensors())
    for i in sorted(set(range(QWEN35_N)) - QWEN35_FULL):
        for k in ("ssm_conv1d.weight", "ssm_norm.weight", "ssm_a"):
            assert types[f"blk.{i}.{k}"] == src[f"blk.{i}.{k}"]
        assert types[f"blk.{i}.ssm_dt.bias"] == F32
        for k in ("attn_gate", "ssm_out", "ssm_alpha", "ssm_beta"):
            assert types[f"blk.{i}.{k}.weight"] == Q4_K


def test_qwen35_still_refuses_narrow_rows_and_mla_kv_b():
    """Lifting the fused-qkv refusal keeps the guards behind it: a 4-wide quantizable row, and MLA's kv_b."""
    narrow = [(n, (4, 576) if n.endswith("attn_qkv.weight") else ne, t) for n, ne, t in _qwen35_tensors()]
    with pytest.raises(Refused, match="F16, not ported"):
        q4_k_m_types("qwen35", QWEN35_N, narrow)
    with_kv_b = [*_qwen35_tensors(), ("blk.0.attn_kv_b.weight", (256, 512), BF16)]
    with pytest.raises(Refused, match="kv_b"):
        q4_k_m_types("qwen35", QWEN35_N, with_kv_b)
    with pytest.raises(Refused, match="ported for"):
        q4_k_m_types("qwen3next", QWEN35_N, _qwen35_tensors())


MOE_N, MOE_EXPERTS = 8, 4


def _qwen35moe_tensors(experts: int = MOE_EXPERTS) -> list[tuple[str, tuple[int, ...], int]]:
    """A tiny ``qwen35moe``: qwen35's layers with an expert bank, a shared expert and a router per layer."""
    t = [(n, ne, ty) for n, ne, ty in _qwen35_tensors() if ".ffn_" not in n]
    for i in range(MOE_N):
        t += [
            (f"blk.{i}.ffn_gate_inp.weight", (256, experts), F32),
            (f"blk.{i}.ffn_gate_inp_shexp.weight", (256, 1), F32),
            (f"blk.{i}.ffn_down_exps.weight", (512, 256, experts), BF16),
            (f"blk.{i}.ffn_gate_exps.weight", (256, 512, experts), BF16),
            (f"blk.{i}.ffn_up_exps.weight", (256, 512, experts), BF16),
            (f"blk.{i}.ffn_down_shexp.weight", (512, 256), BF16),
            (f"blk.{i}.ffn_gate_shexp.weight", (256, 512), BF16),
            (f"blk.{i}.ffn_up_shexp.weight", (256, 512), BF16),
        ]
    return t


def test_moe_ffn_down_takes_its_layer_from_the_name():
    """``layer_info`` reads ``blk.N.`` when there are experts; a running counter would promote other layers.

    Each layer has two ``ffn_down`` tensors (bank and shared expert), so the counter reaches 15 over 8 layers
    and ``use_more_bits`` would fire on a different, larger set. Reading the layer from the name keeps the
    promoted layers at ``use_more_bits(i, 8)`` = {0, 3, 6, 7}, and both tensors of a layer agree.
    """
    types = q4_k_m_types("qwen35moe", MOE_N, _qwen35moe_tensors())
    promoted = {0, 3, 6, 7}
    for i in range(MOE_N):
        want = Q6_K if i in promoted else Q4_K
        assert types[f"blk.{i}.ffn_down_exps.weight"] == want, i
        assert types[f"blk.{i}.ffn_down_shexp.weight"] == want, i
    assert {types[f"blk.{i}.{k}.weight"] for i in range(MOE_N) for k in ("ffn_gate_exps", "ffn_up_exps")} == {
        Q4_K
    }


def test_moe_router_is_never_quantized_and_experts_are_counted_from_the_banks():
    """The shared-expert router is a one-row matrix: ``ggml_n_dims`` sees one dimension, so it stays.

    The bf16 template stores it ``[n_embd, 1]`` and a Q4_K_M file stores it ``[n_embd]``, so reading types back
    from a quantized file cannot catch a rule that quantizes it; this fixture keeps the template's shape.
    """
    src = {n: t for n, _, t in _qwen35moe_tensors()}
    types = q4_k_m_types("qwen35moe", MOE_N, _qwen35moe_tensors())
    for i in range(MOE_N):
        for k in ("ffn_gate_inp", "ffn_gate_inp_shexp"):
            assert types[f"blk.{i}.{k}.weight"] == src[f"blk.{i}.{k}.weight"]
    by_name = {n: (ne, t) for n, ne, t in _qwen35moe_tensors()}
    assert expert_count(by_name) == MOE_EXPERTS
    assert expert_count({n: (ne, t) for n, ne, t in _qwen35_tensors()}) == 1


def test_moe_refuses_eight_experts_and_disagreeing_banks():
    with pytest.raises(Refused, match="8-expert"):
        q4_k_m_types("qwen35moe", MOE_N, _qwen35moe_tensors(experts=8))
    mixed = [
        (n, (ne[0], ne[1], 6) if n.endswith("ffn_up_exps.weight") else ne, t)
        for n, ne, t in _qwen35moe_tensors()
    ]
    with pytest.raises(Refused, match="disagree on the expert count"):
        q4_k_m_types("qwen35moe", MOE_N, mixed)
