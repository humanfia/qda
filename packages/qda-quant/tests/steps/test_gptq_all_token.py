"""GPTQ's all-token blend: the bank statistics every token gives each expert, and the Hessian the solve reads."""

import pytest
import torch
from qda_quant import Format, LayerContext, Target
from qda_quant.arch import describe
from qda_quant.arch.adapters import fused_experts
from qda_quant.arch.adapters.fused_experts import down_input
from qda_quant.engine.stats import LayerStats
from qda_quant.solve import solve
from qda_quant.steps.gptq import GPTQ

from ..tiny import model


def _context(rows=33):
    source = model("qwen3_moe", dtype=torch.float32)
    graph = describe(source, verify=False)
    lg = graph.layers[0]
    layer = source.get_submodule(lg.module)
    x = torch.randn(1, rows, 128, generator=torch.Generator().manual_seed(7))
    inputs = layer.post_attention_layernorm(x)
    with (
        torch.no_grad(),
        LayerStats(
            layer, lg, frozenset({"hessian", "expert_hessian"}), torch.device("cpu"), all_token_hessian=True
        ) as collector,
    ):
        layer.mlp(inputs)
    fmt = Format.parse("w4a16")
    target = Target()
    ctx = LayerContext(
        layer, lg, graph, fmt, target.plan(fmt, "dense"), target.plan(fmt, "moe"), collector.result()
    )
    return ctx, layer.mlp.experts, inputs.reshape(-1, 128)


def test_all_token_statistics_equal_a_per_expert_loop():
    """Gate/up read every token's input; each expert's down input is its own projection of every token."""
    ctx, experts, x = _context()
    router = ctx.graph.banks[0].router
    w = fused_experts.in_weight(experts)
    down = torch.stack([down_input(experts, (x @ w[e].T)[None], None)[0] for e in range(w.shape[0])])
    assert ctx.stats[("all_tokens_seen", router)] == x.shape[0]
    torch.testing.assert_close(ctx.stats[("all_in", router)], x.T @ x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        ctx.stats[("all_down", router)], down.transpose(1, 2) @ down, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("blend", [0.25, 1.0])
def test_the_solve_reads_the_blend_and_an_unrouted_expert_reads_every_token(blend):
    ctx, _, _ = _context()
    router = ctx.graph.banks[0].router
    ctx.stats[("n_tokens", router)][0] = 0  # expert 0 saw no routed token
    for part in ("in", "down"):
        ctx.stats[("expert_hessian", router)][part][0].zero_()
    step = GPTQ(all_token_blend=blend)
    for u in step.units(ctx):
        if u.part is None:
            continue
        h_raw, n = u.hessian()
        h_all, n_all = u.all_token_hessian()
        routed, every = solve.prepare_hessian(h_raw, n), solve.prepare_hessian(h_all, n_all)
        got = step._blend(u, routed, n, None, torch.device("cpu"))
        torch.testing.assert_close(got[1:], (1 - blend) * routed[1:] + blend * every.expand_as(routed)[1:])
        torch.testing.assert_close(got[0], every[0])
