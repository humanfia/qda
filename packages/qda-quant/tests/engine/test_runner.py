"""Layer runner and statistics on tiny random Qwen3 / Qwen3-MoE models (CPU, seconds)."""

from __future__ import annotations

import pytest
import torch
from qda_quant.engine.execute import Timers
from qda_quant.engine.runner import LayerStats, Pass, capture, rel
from qda_quant.engine.stats import observe_experts, reference_expert_stats
from qda_quant.step import LayerContext, Step

from ..engine_util import graph_of, rtn_, tiny_gpt_oss, tiny_qwen3, tiny_qwen3_moe

CPU = torch.device("cpu")


class Probe(Step):
    """Records the statistics it gets; optionally RTN-quantizes the layer's linears in place."""

    needs = frozenset({"hessian", "act_absmax", "inputs", "expert_hessian", "outputs", "layer_io"})
    quantize: bool = False

    def apply_layer(self, ctx: LayerContext) -> None:
        SEEN.append(ctx)
        IO.append([o is not None for *_, o in ctx.stat("layer_io", ctx.graph)])
        if self.quantize:
            for g in ctx.graph.groups:
                for m in g.modules:
                    rtn_(ctx.layer.get_submodule(rel(m, ctx.graph.module)).weight.data)
            for b in ctx.graph.banks:
                ex = ctx.layer.get_submodule(rel(b.experts, ctx.graph.module))
                rtn_(ex.gate_up_proj.data)
                rtn_(ex.down_proj.data)


SEEN: list[LayerContext] = []
IO: list[list[bool]] = []


def run(model, steps, batches, keep_float=False):
    graph = graph_of(model)
    SEEN.clear()
    IO.clear()
    slots = capture(model, graph, batches, CPU, keep_float=keep_float)
    p = Pass(
        model,
        graph,
        steps,
        CPU,
        make_ctx=lambda layer, lg, st: LayerContext(layer, lg, graph, None, None, None, st, device=CPU),
        timers=Timers(),
    )
    p.run(slots)
    return graph, slots


def layer_inputs(model, ids):
    """Reference: each decoder layer's input and output under the model's own forward."""
    ins, outs = {}, {}
    hs = []
    for i, layer in enumerate(model.model.layers):
        hs.append(
            layer.register_forward_pre_hook(
                lambda m, a, kw, i=i: ins.__setitem__(i, a[0] if a else kw["hidden_states"]), with_kwargs=True
            )
        )
        hs.append(
            layer.register_forward_hook(
                lambda m, a, o, i=i: outs.__setitem__(i, o[0] if isinstance(o, tuple) else o)
            )
        )
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    return ins, outs


def batches(n=4, T=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]


@pytest.mark.parametrize("make", [tiny_qwen3, tiny_qwen3_moe])
def test_identity_pass_reproduces_forward(make):
    model = make()
    bs = batches()
    _, slots = run(model, [Probe()], bs)
    for b, s in zip(bs, slots, strict=True):
        _, outs = layer_inputs(model, b)
        torch.testing.assert_close(s.h, outs[len(model.model.layers) - 1], rtol=1e-5, atol=1e-5)


def test_dense_stats_match_reference():
    model = tiny_qwen3()
    bs = batches()
    graph, _ = run(model, [Probe()], bs)
    for i, ctx in enumerate(SEEN):
        layer = model.model.layers[i]
        xs: list[torch.Tensor] = []
        h = layer.input_layernorm.register_forward_hook(
            lambda m, a, o, xs=xs: xs.append(o.reshape(-1, o.shape[-1]))
        )
        for b in bs:
            with torch.no_grad():
                model(input_ids=b, use_cache=False)
        h.remove()
        X = torch.cat(xs).double()
        q = graph.layers[i].groups[0]
        torch.testing.assert_close(ctx.stat("hessian", q).double(), X.T @ X, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ctx.stat("act_absmax", q).double(), X.abs().amax(0), rtol=1e-5, atol=1e-6)
        assert ctx.stat("n_tokens", q) == X.shape[0]
        assert ctx.stat("inputs", q).shape == X.shape
        assert len(ctx.stat("outputs", graph.layers[i])) == len(bs)
        assert IO[i] == [True] * len(bs)


@pytest.mark.parametrize("make", [tiny_qwen3, tiny_qwen3_moe])
def test_quantized_propagation_matches_quantized_model(make):
    """RTN weights do not depend on the statistics, so the propagated stream must equal the forward of
    the fully RTN-quantized model."""
    model = make()
    ref = make()
    bs = batches()
    _, slots = run(model, [Probe(quantize=True)], bs)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if any(k in name for k in ("proj", "experts")) and p.dim() >= 2:
                rtn_(p.data)
    for b, s in zip(bs, slots, strict=True):
        _, outs = layer_inputs(ref, b)
        torch.testing.assert_close(s.h, outs[len(ref.model.layers) - 1], rtol=1e-4, atol=1e-4)


def test_float_stream_target_follows_the_float_model():
    """Target: the untouched model's layer output. Inputs and statistics: as if no float stream existed."""
    bs = batches()
    seen = []

    class Fit(Probe):
        needs = Probe.needs | {"layer_io_float"}

        def apply_layer(self, ctx: LayerContext) -> None:
            seen.append((list(ctx.stat("layer_io_float", ctx.graph)), list(ctx.stat("layer_io", ctx.graph))))
            super().apply_layer(ctx)

    run(tiny_qwen3(), [Probe(quantize=True)], bs)
    plain = list(SEEN)
    run(tiny_qwen3(), [Fit(quantize=True)], bs, keep_float=True)
    ref = tiny_qwen3()
    for k, b in enumerate(bs):
        _, float_outs = layer_inputs(ref, b)
        for i, (float_io, quant_io) in enumerate(seen):
            torch.testing.assert_close(float_io[k][3], float_outs[i], rtol=1e-5, atol=1e-5)
            assert torch.equal(float_io[k][0], quant_io[k][0])  # inputs: the quantized stream
            # the streams, so the two targets, are the same only until the first layer is quantized
            assert torch.allclose(float_io[k][3], quant_io[k][3], rtol=1e-4, atol=1e-4) is (i == 0)
    for fit, probe in zip(SEEN, plain, strict=True):  # the float forward must not feed the statistics
        q = fit.graph.groups[0]
        assert torch.equal(fit.stat("hessian", q), probe.stat("hessian", q))


@pytest.mark.parametrize("router_weighted", [False, True])
def test_fused_expert_stats_match_reference(router_weighted):
    model = tiny_qwen3_moe(layers=1)
    moe = model.model.layers[0].mlp
    x = torch.randn(40, model.config.hidden_size)
    _, w, idx = moe.gate(x)
    graph = graph_of(model)
    lg = graph.layers[0]
    ls = LayerStats(
        model.model.layers[0],
        lg,
        frozenset({"expert_hessian", "act_absmax"}),
        CPU,
        router_weighted=router_weighted,
    )
    st = ls.banks[lg.banks[0].router]
    observe_experts(moe.experts, x, idx, w, st)
    ref = reference_expert_stats(moe.experts, x, idx, w, router_weighted)
    r = st.result()
    torch.testing.assert_close(r["hessian"]["in"], ref["in"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(r["hessian"]["down"], ref["down"], rtol=1e-4, atol=1e-5)
    assert torch.equal(r["n_tokens"], ref["n_tokens"])


def test_transposed_expert_stats_match_reference():
    """A gpt-oss bank stores [E, in, out] with interleaved gate/up and biases: read as the module declares."""
    from qda_quant.arch import describe

    model = tiny_gpt_oss(layers=1)
    layer = model.model.layers[0]
    ex = layer.mlp.experts
    assert ex.is_transposed and ex.has_bias and not ex.is_concatenated
    x = torch.randn(40, model.config.hidden_size)
    _, w, idx = layer.mlp.router(x)
    lg = describe(model).layers[0]
    st = LayerStats(layer, lg, frozenset({"expert_hessian"}), CPU).banks[lg.banks[0].router]
    observe_experts(ex, x, idx, w, st)
    ref = reference_expert_stats(ex, x, idx, w)
    r = st.result()
    torch.testing.assert_close(r["hessian"]["in"], ref["in"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(r["hessian"]["down"], ref["down"], rtol=1e-4, atol=1e-5)
    assert torch.equal(r["n_tokens"], ref["n_tokens"])
    # the reference is the module's arithmetic: interleaved halves, clamped SwiGLU with (up + 1), biases
    alpha, limit = model.config.swiglu_alpha, model.config.swiglu_limit
    for e in range(model.config.num_local_experts):
        t = (idx == e).any(-1).nonzero().flatten()
        o = x[t] @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
        g, u = o[..., ::2].clamp(max=limit), o[..., 1::2].clamp(-limit, limit)
        h = (u + 1) * g * torch.sigmoid(g * alpha)
        torch.testing.assert_close(ref["down"][e], h.T @ h)


def test_moe_pass_exposes_bank_stats():
    model = tiny_qwen3_moe()
    graph, _ = run(model, [Probe()], batches())
    b = graph.layers[0].banks[0]
    H = SEEN[0].stat("expert_hessian", b)
    width = model.config.hidden_size
    assert H["in"].shape == (8, width, width) and H["down"].shape == (8, 32, 32)
    n = SEEN[0].stat("n_tokens", b)
    assert int(n.sum()) == 4 * 24 * 2


@pytest.mark.parametrize("make", [tiny_qwen3, tiny_qwen3_moe])
def test_runner_on_derived_graph(make):
    """The runner works on `arch.describe`'s graph as on the hand-built one."""
    from qda_quant.arch import describe

    model = make()
    graph = describe(model, verify=False)
    assert [len(lg.groups) for lg in graph.layers]
    slots = capture(model, graph, batches(2), CPU)
    got = []
    p = Pass(
        model,
        graph,
        [Probe()],
        CPU,
        make_ctx=lambda layer, lg, st: (
            got.append(sorted({k[0] for k in st}))
            or LayerContext(layer, lg, graph, None, None, None, st, device=CPU)
        ),
        timers=Timers(),
    )
    p.run(slots)
    assert "hessian" in got[0]
    if graph.layers[0].banks:
        assert "expert_hessian" in got[0]


def test_scale_input_stats_matches_reforward():
    """After an AWQ-style fold (norm weight / s, q/k/v columns * s) the analytic update equals the
    statistics of a real re-forward."""
    from qda_quant.engine.stats import scale_input_stats

    model = tiny_qwen3(layers=1)
    graph = graph_of(model)
    lg = graph.layers[0]
    q = lg.groups[0]
    bs = batches()
    want = frozenset({"hessian", "act_absmax", "inputs"})

    def stats():
        slots = capture(model, graph, bs, CPU)
        layer = model.model.layers[0]
        with torch.no_grad(), LayerStats(layer, lg, want, CPU) as st:
            for s in slots:
                layer(s.h, *s.calls[0][0], **s.calls[0][1])
        return st.result()

    before = stats()
    s = torch.rand(64) + 0.5
    ctx = LayerContext(model.model.layers[0], lg, graph, None, None, None, dict(before))
    scale_input_stats(ctx, q, s)
    with torch.no_grad():
        model.model.layers[0].input_layernorm.weight.div_(s)
        for m in ("q_proj", "k_proj", "v_proj"):
            getattr(model.model.layers[0].self_attn, m).weight.mul_(s)
    after = stats()
    k = q.modules[0]
    torch.testing.assert_close(ctx.stats[("hessian", k)], after[("hessian", k)], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ctx.stats[("act_absmax", k)], after[("act_absmax", k)], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ctx.stats[("inputs", k)], after[("inputs", k)], rtol=1e-5, atol=1e-6)


def test_expert_stats_buffered_flushes_match_one_shot():
    model = tiny_qwen3_moe(layers=1)
    moe = model.model.layers[0].mlp
    x = torch.randn(60, model.config.hidden_size)
    _, w, idx = moe.gate(x)
    lg = graph_of(model).layers[0]
    ls = LayerStats(model.model.layers[0], lg, frozenset({"expert_hessian"}), CPU, router_weighted=True)
    st = ls.banks[lg.banks[0].router]
    st.flush_rows = 25
    for sl in (slice(0, 20), slice(20, 45), slice(45, 60)):
        observe_experts(moe.experts, x[sl], idx[sl], w[sl], st)
    ref = reference_expert_stats(moe.experts, x, idx, w, True)
    r = st.result()
    torch.testing.assert_close(r["hessian"]["in"], ref["in"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(r["hessian"]["down"], ref["down"], rtol=1e-4, atol=1e-5)
    assert torch.equal(r["n_tokens"], ref["n_tokens"])


class AllInputs(Probe):
    input_rows: int | None = None


def test_input_rows_none_keeps_every_token_as_batches():
    """Every token is kept, as the batches it arrived in: one concatenated tensor doubles the peak."""
    model = tiny_qwen3(layers=1)
    bs = batches()
    run(model, [AllInputs()], bs)
    q = graph_of(model).layers[0].groups[0]
    chunks = SEEN[0].stat("inputs", q)
    assert isinstance(chunks, list)
    assert [c.shape[0] for c in chunks] == [b.shape[1] for b in bs]
    assert sum(c.shape[0] for c in chunks) == len(bs) * bs[0].shape[1]


def test_input_rows_of():
    from qda_quant.engine.stats import DEFAULT_INPUT_ROWS, input_rows_of

    assert input_rows_of([Probe()]) == DEFAULT_INPUT_ROWS
    assert input_rows_of([Probe(), AllInputs()]) is None
    assert input_rows_of([]) == DEFAULT_INPUT_ROWS


class _ClampedBank(torch.nn.Module):
    """A fused bank that forms its down input as DeepSeek-V4 does: fp32 SwiGLU with a clamp, router weight
    applied before the bf16 cast. Its forward feeds `down_proj` per expert and records what it fed."""

    def __init__(self, E=4, D=16, I=8, limit=0.5):  # noqa: E741
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * I, D, generator=g).bfloat16())
        self.down_proj = torch.nn.Parameter(torch.randn(E, D, I, generator=g).bfloat16())
        self.act_fn = torch.nn.SiLU()
        self.limit = limit
        self.fed: dict[int, list[torch.Tensor]] = {}

    def down_input(self, gate_up, weights):
        gate, up = gate_up.float().chunk(2, dim=-1)
        h = self.act_fn(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)
        return (h if weights is None else h * weights.float().unsqueeze(-1)).to(gate_up.dtype)

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        for e in range(self.gate_up_proj.shape[0]):
            t, s = torch.where(idx == e)
            h = self.down_input(torch.nn.functional.linear(x[t], self.gate_up_proj[e]), w[t, s])
            self.fed.setdefault(e, []).append(h)
            out.index_add_(0, t, torch.nn.functional.linear(h, self.down_proj[e]))
        return out


def test_a_bank_that_forms_its_own_down_input_gets_the_hessian_of_what_its_forward_feeds_down():
    from qda_quant.engine.stats import ExpertStats, GroupStats

    bank = _ClampedBank()
    E, D, I = 4, 16, 8  # noqa: E741
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, D, generator=g).bfloat16()
    idx = torch.stack([torch.randperm(E, generator=g)[:2] for _ in range(50)])
    w = torch.rand(50, 2, generator=g).bfloat16()
    st = ExpertStats(
        GroupStats(D, CPU, frozenset({"hessian"}), E=E), GroupStats(I, CPU, frozenset({"hessian"}), E=E)
    )
    observe_experts(bank, x, idx, w, st)
    bank(x, idx, w)
    real = torch.stack([torch.cat(bank.fed[e]).float().T @ torch.cat(bank.fed[e]).float() for e in range(E)])
    down = st.result()["hessian"]["down"]
    torch.testing.assert_close(down, real, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(reference_expert_stats(bank, x, idx, w)["down"], real, rtol=1e-4, atol=1e-5)
    plain = torch.nn.Module()  # the same weights without `down_input`: the default act(gate) * up
    plain.gate_up_proj, plain.act_fn = bank.gate_up_proj, bank.act_fn
    qwen = ExpertStats(
        GroupStats(D, CPU, frozenset({"hessian"}), E=E), GroupStats(I, CPU, frozenset({"hessian"}), E=E)
    )
    observe_experts(plain, x, idx, w, qwen)
    assert not torch.allclose(qwen.result()["hessian"]["down"], real, rtol=1e-2)


def test_a_joined_bank_records_the_weight_digest():
    """One card solving a bank in chunks joins the entries; its guard must be the exporter's digest."""
    from qda_quant.engine.runner import _joined
    from qda_quant.intake.fingerprint import tensor_digest

    p = torch.nn.Parameter(torch.zeros(4, 8))
    chunks = [
        {"codes": torch.zeros(2, 2, 8, dtype=torch.int8), "scale": torch.ones(2, 2, 1),
         "format": "w4a16-asym-g32", "experts": (0, 2)},
        {"codes": torch.ones(2, 2, 8, dtype=torch.int8), "scale": torch.ones(2, 2, 1),
         "format": "w4a16-asym-g32", "experts": (2, 4)},
    ]  # fmt: skip
    got = _joined(chunks, p)
    assert got["codes"].shape == (4, 2, 8)  # the chunks in expert order
    assert got["weight_check"] == (id(p), tensor_digest(p))
