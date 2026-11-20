"""Determinism of the MoE statistics path on the real Qwen3-30B-A3B layer 0 (GPU).

Two calibration passes over the same 4 samples must give bit-identical Hessians, expert Hessians and
token counts, and GPTQ on two copies of the layer must give bit-identical codes and scales. Needs
``--model-dir qwen3-30b-a3b=PATH`` and ``--calib`` (see conftest.py); skips without them.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
def test_moe_layer0_stats_and_codes_bit_identical(real_model, calib_file):
    from qda_quant import REGISTRY, Format, Target
    from qda_quant.arch import describe
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.calib import load_ids
    from qda_quant.engine.runner import LayerStats, call, capture
    from qda_quant.step import LayerContext
    from transformers import AutoConfig, AutoModelForCausalLM

    q30, data = real_model("qwen3-30b-a3b"), calib_file
    dev = torch.device("cuda:0")
    cfg = AutoConfig.from_pretrained(q30)
    cfg.num_hidden_layers = 1
    m = AutoModelForCausalLM.from_pretrained(
        q30, config=cfg, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    ).eval()
    graph = describe(m, verify=False)
    lg = graph.layers[0]
    ids = load_ids(FileSource(path=str(data), samples=4, seqlen=2048), q30)
    slots = capture(m, graph, [r[None] for r in ids], dev)
    gptq = REGISTRY["gptq"]()
    fmt = Format.parse("w4a16-sym-g128")
    tgt = Target()

    def run():
        layer = copy.deepcopy(m.model.layers[0])
        st = LayerStats(layer, lg, frozenset(gptq.needs), dev)
        with torch.no_grad(), st:
            for s in slots:
                call(layer, s.inp(), *s.calls[0])
        stats = st.result()
        snap = {k: (copy.deepcopy(v)) for k, v in stats.items()}
        ctx = LayerContext(
            layer, lg, graph, fmt, tgt.plan(fmt, "dense"), tgt.plan(fmt, "moe"), stats, device=dev
        )
        gptq.apply_layer(ctx)
        return snap, ctx.qparams

    (s1, q1), (s2, q2) = run(), run()
    assert s1.keys() == s2.keys()
    for k in s1:
        a, b = s1[k], s2[k]
        for x, y in zip(*(list(v.values()) if isinstance(v, dict) else [v] for v in (a, b)), strict=True):
            assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, k
    assert q1.keys() == q2.keys() and q1
    for k in q1:
        assert q1[k].keys() == q2[k].keys(), k
        for f, a in q1[k].items():
            b = q2[k][f]
            if f == "weight_check":  # (identity, checksum): the identity is per load, the checksum must agree
                assert a[1] == b[1], (k, f)
            else:
                assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, (k, f)
    print(f"30B layer 0: {len(s1)} stats and {len(q1)} quantized params bit-identical across two runs")
