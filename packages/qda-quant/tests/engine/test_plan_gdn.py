"""A gated-delta layer's calibration forward is a plan term, and a solve chunks from memory it can place.

Qwen3.6-35B-A3B GPTQ at the default residency planned 76.3 GiB of the 78.7 GiB its card had, ran its first
calibration forward to 72.2 GiB allocated, and then died in the layer's solve at 75.1 GiB of live tensors while
the allocator could not place the chunk's 0.9 GiB factor. Two things were missing: the forward's transients
through the delta rule (`linear_attention_bytes`), and a solve chunk sized from the memory the driver can hand
rather than the blocks the allocator happens to hold (`solve.free_bytes`).
"""

from __future__ import annotations

import types

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.arch import describe
from qda_quant.engine.planning import GDN_CHUNK, linear_attention_bytes, memory_terms
from qda_quant.engine.residency import HEADROOM_SHARE, headroom, plan_headroom
from qda_quant.format import Format

from ..tiny import model

GIB = 2**30


def _layers(kind: str):
    meta = model(kind, meta=True)
    graph = describe(meta, verify=False)
    return meta, graph.layers


def _layout(meta) -> tuple[int, int, int, int]:
    """The tiny GDN layer's (value heads, key width, value width, conv width), read off its module."""
    gdn = next(
        m for m in meta.modules() if {"A_log", "dt_bias"} <= set(dict(m.named_parameters(recurse=False)))
    )
    return gdn.num_v_heads, gdn.num_v_heads * gdn.head_k_dim, gdn.value_dim, gdn.conv_dim


def test_a_gdn_layers_forward_transients_are_sized():
    """16 tokens through the tiny 2-key/4-value-head layer: conv input and output plus `z`, the rule's fp32
    copies, the chunk states, the intra-chunk matrix and the gates.

    99072 = 16 x 2 x (2 x 256 + 128) [model dtype] + 16 x 4 x (2 x 128 + 5 x 128) [fp32] + 16 x 4 x 4 x 32 x 32 // 64
    [chunk states] + 16 x 4 x 4 x 64 [intra-chunk] + 16 x 4 x 3 x 4 [gates].  The convolution is 256 wide
    (2 x 2 key heads x 32 + 4 value heads x 32), q and k are repeated to the 4 value heads (128), and the value
    width is 4 x 32 = 128.
    """
    meta, layers = _layers("qwen3_5")
    heads, key, value, conv = _layout(meta)
    assert (heads, key, value, conv) == (4, 128, 128, 256)
    sized = [linear_attention_bytes(meta, lg, 16, 2) for lg in layers]
    # three linear-attention layers of four; the full-attention layer has no delta rule and adds nothing
    assert sized == [99072, 99072, 99072, 0]
    # the state's chunk is fla's 64: a shorter forward still allocates one chunk's states
    assert GDN_CHUNK == 64
    state = 16 * 4 * heads * 32 * 32 // GDN_CHUNK
    assert state == 4096
    # linear in the rows the forward runs
    assert linear_attention_bytes(meta, layers[0], 32, 2) == 2 * 99072
    # a bf16 forward and an fp32 one differ only in the model-dtype part (the rule is fp32 either way)
    assert linear_attention_bytes(meta, layers[0], 16, 4) == 99072 + 16 * 2 * (2 * conv + value)


def test_a_plan_without_gated_delta_layers_counts_nothing():
    """Qwen3 has no delta rule: every layer's term is 0, and `memory_terms` drops it."""
    meta, layers = _layers("qwen3")
    assert [linear_attention_bytes(meta, lg, 16, 2) for lg in layers] == [0, 0]
    terms, _ = memory_terms(
        meta,
        describe(meta, verify=False),
        [REGISTRY["gptq"]()],
        samples=8,
        seqlen=16,
        batch_size=1,
        world=1,
        fmt=Format.parse("w4a16-sym-pc"),
    )
    assert "linear-attention transients" not in {t.name for t in terms}


def test_the_term_reaches_the_plan_of_a_gated_delta_model():
    """It is one of the largest-term layer's terms, and the batch multiplies the rows one forward runs."""
    meta = model("qwen3_5", meta=True)
    graph = describe(meta, verify=False)
    fmt = Format.parse("w4a16-sym-pc")
    terms, _ = memory_terms(
        meta,
        graph,
        [REGISTRY["gptq"]()],
        samples=8,
        seqlen=16,
        batch_size=1,
        world=1,
        fmt=fmt,
    )
    assert {t.name: t.bytes for t in terms}["linear-attention transients"] == 99072
    terms, _ = memory_terms(
        meta,
        graph,
        [REGISTRY["gptq"]()],
        samples=8,
        seqlen=16,
        batch_size=2,
        world=1,
        fmt=fmt,
    )
    assert {t.name: t.bytes for t in terms}["linear-attention transients"] == 2 * 99072


def test_the_plan_headroom_is_read_off_the_card():
    """`plan_headroom` is `headroom` on the card's total: the same 3 %-or-2 GiB rule a plan applies."""
    total = 80 * GIB
    device = torch.device("cuda:0")
    props = types.SimpleNamespace(total_memory=total)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_properties", lambda dev: props)
        assert plan_headroom(device) == headroom(device, total) == int(total * HEADROOM_SHARE)
        assert plan_headroom(torch.device("cpu")) == 0


def test_the_solve_sizes_its_chunk_from_memory_the_driver_can_hand():
    """The driver's free memory less the plan's headroom; not the blocks the allocator already holds."""

    def refuse(name):
        raise AssertionError(f"free_bytes counted the allocator's {name}")

    from qda_quant.solve import solve

    device = torch.device("cuda:0")
    total, free = 80 * GIB, 10 * GIB
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "mem_get_info", lambda dev: (free, total))
        mp.setattr(torch.cuda, "get_device_properties", lambda dev: types.SimpleNamespace(total_memory=total))
        mp.setattr(torch.cuda, "memory_reserved", lambda dev: refuse("reserved blocks"))
        mp.setattr(torch.cuda, "memory_allocated", lambda dev: refuse("allocated bytes"))
        assert solve.free_bytes(device) == free - headroom(device, total)
    assert solve.free_bytes(torch.device("cpu")) == 1 << 50
