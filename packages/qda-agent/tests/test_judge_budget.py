"""The search's budget in bits per quantized weight (judge.budget_bpw): `qda_agent.judge.checks.budget`."""

import json
from pathlib import Path

import pytest
from qda_agent.judge.checks.budget import (
    HEAD_LEASE,
    bpw,
    bpw_problem,
    denominator,
    head_bits,
    lease_formats,
    leased_bpw,
    stored_bits,
)
from qda_agent.judge.checks.formats import format_problem
from qda_agent.judge.config import JudgeSettings
from safetensors_kit import write_safetensors
from test_judge import Rig

CONFIG = {"model_type": "qwen3", "hidden_size": 128, "num_hidden_layers": 1}
QCONFIG = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {"weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "int"}}
    },
}
Q = "model.layers.0.self_attn.q_proj"
K = "model.layers.0.self_attn.k_proj"


def model_dir(d: Path, tensors: dict, quantized: bool = False) -> Path:
    """A tiny checkpoint: the same config (with a quantization config when quantized) and tokenizer."""
    d.mkdir(parents=True)
    cfg = {**CONFIG, "quantization_config": QCONFIG} if quantized else dict(CONFIG)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return d


def anchor(d: Path) -> Path:
    """Two 128x128 decoder Linears: 32,768 weights, the fixed denominator."""
    return model_dir(
        d, {f"{Q}.weight": ("BF16", (128, 128), None), f"{K}.weight": ("BF16", (128, 128), None)}
    )


def quantized(module: str, bits: int, zero_points: bool = False) -> dict:
    """A module's stored parts: int`bits` group-128 codes, their bf16 scales, and zero points if asymmetric."""
    rows = cols = 128
    t = {
        f"{module}.weight_packed": ("I32", (rows, cols * bits // 32), None),
        f"{module}.weight_scale": ("BF16", (rows, 1), None),
        f"{module}.weight_shape": ("I64", (2,), None),
    }
    if zero_points:
        t[f"{module}.weight_zero_point"] = ("I32", (rows * bits // 32, 1), None)
    return t


def nvfp4(module: str, globals: bool = True) -> dict:
    """An NVFP4 module's stored parts: E2M1 codes two a U8 byte, an E4M3 scale per 16, two fp32 globals."""
    t = {
        f"{module}.weight_packed": ("U8", (128, 64), None),
        f"{module}.weight_scale": ("F8_E4M3", (128, 8), None),
    }
    if globals:
        t[f"{module}.weight_global_scale"] = ("F32", (1,), None)
        t[f"{module}.input_global_scale"] = ("F32", (1,), None)
    return t


def test_an_nvfp4_modules_global_scales_count_toward_its_bits(tmp_path):
    """The two fp32 globals are the module's stored parts: without them the module is not priced at all."""
    a = anchor(tmp_path / "anchor")
    full = model_dir(tmp_path / "full", {**nvfp4(Q), **nvfp4(K)})
    bare = model_dir(tmp_path / "bare", {**nvfp4(Q, globals=False), **nvfp4(K, globals=False)})
    assert bpw(full, a) == pytest.approx(4 + 8 / 16 + 2 * 32 / (128 * 128))
    assert stored_bits(full) - stored_bits(bare) == 2 * 64  # eight fp32 bytes a module


def test_a_uniform_w4a16_g128_checkpoint_gives_the_textbook_number(tmp_path):
    a = anchor(tmp_path / "anchor")
    # every decoder Linear quantized: 4 bits a code + a bf16 scale per 128-weight group (+ 4-bit zero
    # points when asymmetric)
    both = {**quantized(Q, 4), **quantized(K, 4)}
    symmetric = model_dir(tmp_path / "sym", both, quantized=True)
    assert bpw(symmetric, a) == pytest.approx(4 + 16 / 128)
    both = {**quantized(Q, 4, zero_points=True), **quantized(K, 4, zero_points=True)}
    asymmetric = model_dir(tmp_path / "asym", both, quantized=True)
    assert bpw(asymmetric, a) == pytest.approx(4 + 16 / 128 + 4 / 128)


def test_the_denominator_is_the_originals_decoder_linears(tmp_path):
    a = anchor(tmp_path / "anchor")
    # lm_head and the router are not decoder Linears llm-compressor's default quantizes; the embedding
    # is not a Linear at all.
    extra = model_dir(
        tmp_path / "orig",
        {
            f"{Q}.weight": ("BF16", (128, 128), None),
            f"{K}.weight": ("BF16", (128, 128), None),
            "lm_head.weight": ("BF16", (256, 128), None),
            "model.embed_tokens.weight": ("BF16", (256, 128), None),
            "model.layers.0.mlp.gate.weight": ("F32", (8, 128), None),
            "backbone.layers.0.mixer.gate.weight": ("F32", (8, 128), None),  # NemotronH names its router here
        },
    )
    assert denominator(extra) == 2 * 128 * 128
    # A candidate that quantizes only half of those weights is measured against the whole count.
    half = model_dir(tmp_path / "half", {**quantized(Q, 4), f"{K}.weight": ("BF16", (128, 128), None)}, True)
    assert bpw(half, a) == pytest.approx((4 + 16 / 128 + 16) / 2)


HEAD = "lm_head"


def test_the_head_is_outside_the_body_and_its_lease_lends_what_it_saves(tmp_path):
    """The card's bpw is the body's; with the lease the head pays 16 bits a weight up front, and a narrower
    head's savings are lent to the body: (body + head - 16 x head weights) / body weights."""
    body = {**quantized(Q, 4), **quantized(K, 4)}  # 4.125 bpw
    bf16 = model_dir(tmp_path / "bf16", {**body, f"{HEAD}.weight": ("BF16", (256, 128), None)}, True)
    head4 = {
        f"{HEAD}.weight_packed": ("I32", (256, 128 * 4 // 32), None),
        f"{HEAD}.weight_scale": ("BF16", (256, 1), None),
    }
    small = model_dir(tmp_path / "small", {**body, **head4}, True)
    weights, lease = 2 * 128 * 128, 256 * 128
    assert stored_bits(bf16) == stored_bits(small) == 4.125 * weights  # the head is in neither body
    assert head_bits(bf16) == 16 * lease
    assert leased_bpw(bf16, weights, lease) == pytest.approx(4.125)  # a 16-bit head lends nothing
    saved = 16 * lease - (4 * lease + 16 * 256)
    assert leased_bpw(small, weights, lease) == pytest.approx(4.125 - saved / weights)
    # over a 4.0 budget alone, inside it with the lease; the refusal names both numbers
    assert "4.125 bits per quantized weight" in str(bpw_problem(small, weights, 4.0))
    assert bpw_problem(small, weights, 4.0, lease=lease) is None
    why = bpw_problem(bf16, weights, 4.0, lease=lease)
    assert why is not None and "after the head's lease (the body alone 4.125)" in why


def test_the_lease_offers_each_row_its_own_head_formats():
    assert lease_formats("w4a16") == lease_formats("w3a16-g64") == ("bf16", "w8a16", "w4a16")
    assert lease_formats("w4a8") == ("bf16", "w8a16")  # no 4-bit head beside FP8 activations yet
    assert HEAD_LEASE["w4a4"] == ("bf16", "fp8")


def test_a_leased_head_is_admitted_only_in_a_format_its_row_offers(tmp_path):
    """The format gate: without the lease the head stays 16-bit; with it, a w4a16 head passes on a W4A16
    row and is refused on a W4A8 one, whose lease stops at 8 bits."""
    groups = {
        "body": {"targets": ["Linear"], "weights": QCONFIG["config_groups"]["group_0"]["weights"]},
        "head": {
            "targets": ["re:.*lm_head$"],
            "weights": {
                "num_bits": 4,
                "group_size": 128,
                "strategy": "group",
                "type": "int",
                "symmetric": True,
            },
        },
    }
    d = model_dir(
        tmp_path / "c",
        {
            **quantized(Q, 4),
            **quantized(K, 4),
            f"{HEAD}.weight_packed": ("I32", (256, 16), None),
            f"{HEAD}.weight_scale": ("BF16", (256, 1), None),
            "model.embed_tokens.weight": ("BF16", (256, 128), None),
        },
    )
    cfg = {**CONFIG, "quantization_config": {"quant_method": "compressed-tensors", "config_groups": groups}}
    (d / "config.json").write_text(json.dumps(cfg))
    menu = ["bf16", "w8a16", "w4a16"]
    assert "the task keeps it 16-bit" in str(format_problem(d, menu, floor=False))
    assert format_problem(d, menu, lease=lease_formats("w4a16"), floor=False) is None
    why = format_problem(d, menu, lease=lease_formats("w4a8"), floor=False)
    assert why is not None and "the head lease offers bf16, w8a16" in why


def test_a_handin_inside_the_budget_only_by_its_lease_is_admitted(tmp_path):
    a = model_dir(
        tmp_path / "anchor",
        {
            f"{Q}.weight": ("BF16", (128, 128), None),
            f"{K}.weight": ("BF16", (128, 128), None),
            f"{HEAD}.weight": ("BF16", (256, 128), None),
        },
    )
    head4 = {
        f"{HEAD}.weight_packed": ("I32", (256, 16), None),
        f"{HEAD}.weight_scale": ("BF16", (256, 1), None),
    }
    for lease in (False, True):
        r = Rig(
            tmp_path / str(lease),
            anchor=a,
            budget_bpw=4.0,
            head_lease=lease,
            format="w4a16",
            allow_any_format=True,
        )
        model_dir(r.work / "small", {**quantized(Q, 4), **quantized(K, 4), **head4}, quantized=True)
        rep = r.call("handin", "small")
        assert (rep.exit == 0) is lease, rep.stderr


def test_the_head_lease_needs_a_vllm_budget():
    with pytest.raises(ValueError, match="judge.head_lease lends"):
        JudgeSettings(head_lease=True)


def test_every_embedding_name_is_outside_both_sums(tmp_path):
    """A model may store more than one embedding, under names other than `embed_tokens` (Gemma-4's per-layer
    one, `embed_tokens_per_layer`, is the reported case): an nn.Embedding is never a decoder Linear."""
    for i, name in enumerate(
        ("embed_tokens_per_layer", "wte", "embed_positions", "word_embeddings", "shared", "embeddings")
    ):
        d = model_dir(
            tmp_path / f"orig{i}",
            {
                f"{Q}.weight": ("BF16", (128, 128), None),
                f"{K}.weight": ("BF16", (128, 128), None),
                f"model.{name}.weight": ("BF16", (256, 128), None),
            },
        )
        assert denominator(d) == 2 * 128 * 128, name


def test_a_mixed_w8a16_w4a16_candidate_under_budget_is_admitted(tmp_path):
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True)
    model_dir(r.work / "mixed", {**quantized(Q, 8), **quantized(K, 4)}, quantized=True)
    rep = r.call("handin", "mixed")
    assert rep.exit == 0, rep.stderr
    assert r.host.handins()[0].valid is None  # admitted: its serving check is queued


def test_a_candidate_over_budget_is_refused_with_its_bpw(tmp_path):
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True)
    # The second Linear is kept bf16 (16 bits a weight) instead of quantized: 12.0625 bpw.
    model_dir(r.work / "heavy", {**quantized(Q, 8), f"{K}.weight": ("BF16", (128, 128), None)}, True)
    rep = r.call("handin", "heavy")
    assert rep.exit == 2
    assert "12.062 bits per quantized weight" in rep.stderr and "6.5 bpw" in rep.stderr
    assert r.host.handins() == [] and not any(r.be.launched)


def test_candidate_0_over_budget_refuses_the_host(tmp_path):
    a = anchor(tmp_path / "anchor")
    c0 = anchor(tmp_path / "candidate0")
    r = Rig(tmp_path, anchor=a, budget_bpw=4.0, baselines=[c0], allow_any_format=True)
    with pytest.raises(ValueError, match=r"baselines\[0\]: 16\.000 bits per quantized weight"):
        r.host.start()
