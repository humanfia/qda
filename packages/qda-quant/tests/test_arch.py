import pytest
from qda_quant.arch import ExpertBank, LayerGraph, ModelGraph, describe
from qda_quant.arch.derive import _check_ignored
from qda_quant.step import Refused
from qda_quant.steps._model import OUTPUT_KINDS, check_float_banks, ignored_in

from .tiny import model


def _rel(g, n):
    return n[len(g.module) + 1 :] if n else n


def test_qwen3_dense():
    graph = describe(model("qwen3"))
    assert graph.verified and graph.extra["dropped"] == []
    g = graph.layers[1]
    got = [(tuple(_rel(g, m) for m in x.modules), x.kind, _rel(g, x.norm), _rel(g, x.prev)) for x in g.groups]
    assert got == [
        (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "attn_in", "input_layernorm", None),
        (("self_attn.o_proj",), "attn_out", None, None),  # GQA: v -> o is not a fold
        (("mlp.gate_proj", "mlp.up_proj"), "mlp_in", "post_attention_layernorm", None),
        (("mlp.down_proj",), "mlp_out", None, "mlp.up_proj"),
    ]
    assert (graph.embed, graph.lm_head, graph.final_norm) == ("model.embed_tokens", "lm_head", "model.norm")


def test_mha_v_to_o_is_a_fold():
    g = describe(model("qwen3-mha")).layers[0]
    assert g.groups[1].prev == "model.layers.0.self_attn.v_proj"


def test_llama():
    graph = describe(model("llama"))
    assert graph.verified and graph.extra["dropped"] == [] and len(graph.layers[0].groups) == 4


def test_biased_layernorm_folds():
    """The fold scales the norm's bias too: OPT's LayerNorms carry one, and the relation must survive."""
    m = model("opt")
    norm = m.model.decoder.layers[0].self_attn_layer_norm
    assert norm.bias is not None
    graph = describe(m)
    assert graph.verified and graph.extra["dropped"] == []
    assert graph.layers[0].groups[0].norm == "model.decoder.layers.0.self_attn_layer_norm"


def test_qwen3_moe_fused_bank():
    graph = describe(model("qwen3_moe"))
    g = graph.layers[0]
    assert graph.adapter == "fused_experts" and graph.extra["dropped"] == []
    (bank,) = g.banks
    assert (
        bank.fused
        and bank.experts == "model.layers.0.mlp.experts"
        and bank.router == "model.layers.0.mlp.gate"
    )
    assert bank.top_k == 2 and bank.shared == ()
    assert [x.kind for x in g.groups] == ["attn_in", "attn_out"]
    # the adapter puts the bank right after its router among the readers of the norm both read
    assert graph.extra["readers"]["model.layers.0.post_attention_layernorm"] == (bank.router, bank.experts)


@pytest.mark.parametrize(
    ("kind", "adapter"),
    [
        ("qwen3", None),
        ("llama", None),
        ("qwen3_5", None),
        ("qwen3_moe", "fused_experts"),
        ("qwen2_moe", "qwen2_moe"),
        ("gpt_oss", "gpt_oss"),
    ],
)
def test_which_adapter_runs(kind, adapter):
    assert describe(model(kind, meta=True), verify=False).adapter == adapter


def test_gpt_oss_banks_are_the_fused_layout():
    """gpt-oss keeps the fused-expert graph (transposed, interleaved, with biases); only its export differs."""
    graph = describe(model("gpt_oss", meta=True), verify=False)
    (bank,) = graph.layers[0].banks
    assert bank.fused and bank.transposed and bank.experts == "model.layers.0.mlp.experts"
    assert bank.router == "model.layers.0.mlp.router"


@pytest.mark.parametrize("kind", ["qwen3", "llama", "qwen3_moe"])
def test_meta_model_matches_real(kind):
    assert describe(model(kind, meta=True), verify=False).layers == describe(model(kind), verify=False).layers


def test_qwen3_5_hybrid_and_one_plus_w_norms():
    graph = describe(model("qwen3_5"))
    assert graph.verified and graph.extra["dropped"] == []
    kinds = {tuple(x.kind for x in g.groups) for g in graph.layers}
    assert ("linear_attn_in", "linear_attn_out", "mlp_in", "mlp_out") in kinds
    assert ("attn_in", "attn_out", "mlp_in", "mlp_out") in kinds
    # the decoder norms are x * (1 + w): the fold check only passes with offset 1
    assert graph.extra["norm_offset"] == {"input_layernorm": 1.0, "post_attention_layernorm": 1.0}


def test_verify_rejects_a_wrong_norm_relation():
    import dataclasses

    from qda_quant.arch.derive import derive
    from qda_quant.arch.verify import verify

    m = model("qwen3")
    graph = derive(m)
    g = graph.layers[0]
    attn, o, mlp, down = g.groups
    bad = dataclasses.replace(attn, norm=mlp.norm)  # q/k/v do not read post_attention_layernorm
    layers = (dataclasses.replace(g, groups=(bad, o, mlp, down)), *graph.layers[1:])
    out = verify(m, dataclasses.replace(graph, layers=layers))
    assert out.layers[0].groups[0].norm is None
    assert out.extra["dropped"] == [f"{mlp.norm} -> {attn.modules[0]}"]


@pytest.mark.parametrize("kind", ["qwen3", "llama", "qwen3_moe", "qwen3_5", "olmo2"])
def test_verify_capture_stops_after_representative_inputs(kind, monkeypatch):
    import torch
    from qda_quant.arch.derive import decoder_layers, derive
    from qda_quant.arch.verify import _capture, _signature, verify

    instance = model(kind)
    graph = derive(instance)
    _, layers = decoder_layers(instance)
    representatives = {}
    for layer_graph in graph.layers:
        representatives.setdefault(_signature(layer_graph), layer_graph.index)
    wanted = set(representatives.values())
    completed = []
    handles = [
        layer.register_forward_hook(lambda _module, _args, _output, index=index: completed.append(index))
        for index, layer in enumerate(layers)
    ]
    hooks_before = [dict(layer._forward_pre_hooks) for layer in layers]
    try:
        fast = _capture(instance, layers, wanted)
        assert completed == list(range(max(wanted)))
        completed.clear()
        reference = _capture(instance, layers, wanted, stop_early=False)
        assert completed == list(range(len(layers)))
        torch.testing.assert_close(fast, reference, rtol=0, atol=0)
        assert [dict(layer._forward_pre_hooks) for layer in layers] == hooks_before
        fast_graph = verify(instance, graph)
        monkeypatch.setattr(
            "qda_quant.arch.verify._capture",
            lambda model, layers, want: _capture(model, layers, want, stop_early=False),
        )
        assert verify(instance, graph) == fast_graph
    finally:
        for handle in handles:
            handle.remove()


def test_verify_capture_removes_hooks_on_model_error():
    from qda_quant.arch.derive import decoder_layers
    from qda_quant.arch.verify import _capture

    instance = model("qwen3")
    _, layers = decoder_layers(instance)

    def fail(_module, _args):
        raise RuntimeError("failed forward")

    handle = layers[0].register_forward_pre_hook(fail)
    hooks_before = [dict(layer._forward_pre_hooks) for layer in layers]
    try:
        with pytest.raises(RuntimeError, match="failed forward"):
            _capture(instance, layers, {0})
        assert [dict(layer._forward_pre_hooks) for layer in layers] == hooks_before
    finally:
        handle.remove()


def test_verify_rejects_a_wrong_prev_relation():
    import dataclasses

    from qda_quant.arch.derive import derive
    from qda_quant.arch.verify import verify

    m = model("qwen3")
    graph = derive(m)
    g = graph.layers[0]
    attn, o, mlp, down = g.groups
    gate = mlp.modules[0]  # act(gate) * up: scaling gate's rows is not a fold
    layers = (
        dataclasses.replace(g, groups=(attn, o, mlp, dataclasses.replace(down, prev=gate))),
        *graph.layers[1:],
    )
    out = verify(m, dataclasses.replace(graph, layers=layers))
    assert out.layers[0].groups[3].prev is None
    # and the true relation survives verification
    assert verify(m, graph).layers[0].groups[3].prev == mlp.modules[1]


# --------------------------------------------------------------------------- adapters as plugins


@pytest.fixture
def fresh_adapters(monkeypatch):
    """`ADAPTERS` with its built-ins but entry points not yet read; restored after the test."""
    from qda_quant.arch.adapters import ADAPTERS

    monkeypatch.setattr(ADAPTERS, "_entries", dict(ADAPTERS._entries))
    return ADAPTERS


class NoMlpNorm:
    """A third-party adapter: on Qwen3, the MLP input group of layer 0 gets no norm anchor."""

    def matches(self, model):
        return model.config.model_type == "qwen3"

    def adjust(self, model, graph):
        import dataclasses

        g = graph.layers[0]
        groups = tuple(dataclasses.replace(x, norm=None) if x.kind == "mlp_in" else x for x in g.groups)
        return dataclasses.replace(graph, layers=(dataclasses.replace(g, groups=groups), *graph.layers[1:]))


def test_third_party_adapter_from_an_entry_point(monkeypatch, fresh_adapters):
    from .test_plugins import fake_plugins

    fake_plugins(monkeypatch, lambda: fresh_adapters.register("no_mlp_norm", NoMlpNorm()))
    graph = describe(model("qwen3"))
    assert graph.adapter == "no_mlp_norm" and graph.verified and graph.extra["dropped"] == []
    assert [x.norm for x in graph.layers[0].groups if x.kind == "mlp_in"] == [None]
    assert graph.layers[1].groups[2].norm == "model.layers.1.post_attention_layernorm"  # untouched
    assert describe(model("qwen3_moe", meta=True), verify=False).adapter == "fused_experts"


def test_two_matching_adapters_are_an_error(monkeypatch, fresh_adapters):
    from .test_plugins import fake_plugins

    def register():
        fresh_adapters.register("a_first", NoMlpNorm())
        fresh_adapters.register("b_too", NoMlpNorm())

    fake_plugins(monkeypatch, register)
    with pytest.raises(Refused, match=r"more than one architecture adapter: \['a_first', 'b_too'\]"):
        describe(model("qwen3", meta=True), verify=False)


def test_adapter_registry_refuses_a_non_adapter(monkeypatch, fresh_adapters):
    from .test_plugins import fake_plugins

    fake_plugins(monkeypatch, lambda: fresh_adapters.register("nope", object()))
    with pytest.raises(TypeError, match=r"arch adapter plugin 'nope' .* without matches\(model\)"):
        fresh_adapters["fused_experts"]


def test_stacked_experts_without_an_adapter_are_refused():
    import torch

    class Stack(torch.nn.Module):  # experts as two [E, ., .] params, in a layout no adapter knows
        def __init__(self):
            super().__init__()
            self.w_in, self.w_out = (torch.nn.Parameter(torch.empty(4, 8, 8)) for _ in range(2))

    m = model("qwen3", meta=True)
    m.model.layers[0].mlp.stack = Stack()
    with pytest.raises(Refused, match=r"\['model.layers.0.mlp.stack'\].*'qda.arch_adapters'"):
        describe(m, verify=False)


def test_fused_bank_without_its_router_is_refused():
    import torch

    class Wrapped(torch.nn.Module):  # the router one level down: no longer the bank's sibling
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *a, **k):
            return self.inner(*a, **k)

    m = model("qwen3_moe", meta=True)
    mlp = m.model.layers[0].mlp
    mlp.gate = Wrapped(mlp.gate)
    with pytest.raises(Refused, match=r"fused expert bank model.layers.0.mlp.experts: no router"):
        describe(m, verify=False)


def test_gated_deltanet_head_gates_stay_bf16():
    """Regression against the old coverage (and the paper's checkpoints): Gated DeltaNet's per-head projections
    ``in_proj_a`` / ``in_proj_b`` (one value per value head) are not quantized, however many heads there are."""
    g = describe(model("qwen3_5-heads32"), verify=False)
    quantized = {m for lg in g.layers for grp in lg.groups for m in grp.modules}
    gdn = "model.layers.0.linear_attn."
    assert {gdn + "in_proj_qkv", gdn + "in_proj_z", gdn + "out_proj"} <= quantized
    assert not {gdn + "in_proj_a", gdn + "in_proj_b"} & quantized
    assert gdn + "in_proj_a" in g.extra["readers"][gdn + "in_proj_qkv"]  # a norm fold still scales them


def test_shared_expert_router_and_gate_stay_bf16():
    """The paper's coverage (appendix, "Quantized modules"): on a MoE with a shared expert, the router, the shared
    expert and its gate stay bf16. The router and the scalar gate are never quantized; the shared expert is kept
    by the default ``shared_experts`` role. The bank still lists the shared expert."""
    g = describe(model("qwen2_moe"), verify=False)
    quantized = {m for lg in g.layers for grp in lg.groups for m in grp.modules}
    (bank,) = g.layers[0].banks
    assert bank.shared and not set(bank.shared) & quantized
    assert bank.router == "model.layers.0.mlp.gate" and bank.router not in quantized
    assert "model.layers.0.mlp.shared_expert_gate" not in quantized
    assert {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"} <= quantized


@pytest.mark.parametrize(
    ("kind", "shared"), [("deepseek_v2", "shared_experts"), ("qwen3_5_moe", "shared_expert")]
)
def test_shared_experts_are_found_from_the_traced_forward(kind, shared):
    """The shared expert is the dense MLP beside the bank that reads the router's input, whatever its name:
    Qwen spells it ``shared_expert`` and DeepSeek ``shared_experts``. The default ``shared_experts`` keep role
    keeps it bf16; with the role dropped it is quantized with the rest."""
    graph = describe(model(kind), verify=False)
    (bank,) = next(lg.banks for lg in graph.layers if lg.banks)
    expected = tuple(
        f"{bank.router.rpartition('.')[0]}.{shared}.{p}" for p in ("gate_proj", "up_proj", "down_proj")
    )
    assert bank.shared == expected
    assert not set(expected) & {m for lg in graph.layers for grp in lg.groups for m in grp.modules}
    dropped = describe(model(kind), verify=False, keep=frozenset())
    (bank,) = next(lg.banks for lg in dropped.layers if lg.banks)
    assert bank.shared == expected
    assert set(expected) <= {m for lg in dropped.layers for grp in lg.groups for m in grp.modules}


def test_keep_nothing_quantizes_the_kept_roles():
    g = describe(model("qwen3_5-heads32"), verify=False, keep=frozenset())
    assert "model.layers.0.linear_attn.in_proj_a" in {m for grp in g.layers[0].groups for m in grp.modules}
    assert g.extra["kept_bf16"] == {}
    g = describe(model("qwen2_moe"), verify=False, keep=frozenset())
    (bank,) = g.layers[0].banks
    assert set(bank.shared) <= {m for grp in g.layers[0].groups for m in grp.modules}
    with pytest.raises(Refused, match="'routers' matches no linear"):
        describe(model("qwen3"), verify=False, keep=frozenset({"routers"}))


@pytest.mark.parametrize("kind", ["qwen2_moe", "qwen3_moe"])  # both tiny MoEs store a fused 3-D bank
def test_ignore_refuses_part_of_a_routed_bank(kind):
    """A bank is served with one scheme: an ``ignore`` pattern cannot keep some of its experts float."""
    with pytest.raises(Refused, match="routed expert"):
        describe(
            model(kind), verify=False, keep=frozenset({r"re:^model\.layers\.0\.mlp\.experts\.down_proj$"})
        )


@pytest.mark.parametrize("kind", ["qwen2_moe", "qwen3_moe"])
@pytest.mark.parametrize("pattern", [r"re:^model\.layers\.0\.mlp\.experts\.", "model.layers.0.mlp.experts"])
def test_ignore_keeps_a_whole_routed_bank_float(kind, pattern):
    """A pattern naming the bank module, or every linear of it, keeps the whole bank bf16.

    It leaves the graph (no unit, no statistics, no plan), its root is recorded for the export, and its linears
    are in ``ignored`` under the kinds rotate and permute refuse to move.
    """
    g = describe(model(kind), verify=False, keep=frozenset({pattern}))
    assert g.layers[0].banks == () and g.layers[1].banks  # only the named layer's bank
    assert g.extra["float_banks"] == ("model.layers.0.mlp.experts",)
    kinds = {n: k for n, (_p, k) in g.extra["ignored"].items()}
    assert kinds["model.layers.0.mlp.experts.gate_up_proj"] == "mlp_in"
    assert kinds["model.layers.0.mlp.experts.down_proj"] == "mlp_out"
    # a step that moves weights across layers cannot move a bank it does not see
    with pytest.raises(Refused, match="rotate: ignore"):
        ignored_in(g, "rotate", OUTPUT_KINDS, "the linears that write the residual stream")
    # a second pattern naming part of a bank that another pattern keeps whole is still a half-float bank
    with pytest.raises(Refused, match="part of the routed bank"):
        describe(model(kind), verify=False, keep=frozenset({pattern, "model.layers.0.mlp.experts.down_proj"}))


def test_two_patterns_may_each_cover_the_whole_bank():
    """Several patterns may each name the whole bank; a cover is not part of it."""
    keep = frozenset({r"re:^model\.layers\.0\.mlp\.experts\.", "model.layers.0.mlp.experts"})
    g = describe(model("qwen3_moe"), verify=False, keep=keep)
    assert g.extra["float_banks"] == ("model.layers.0.mlp.experts",)


def test_awq_is_refused_beside_a_float_kept_bank_but_gptq_is_not():
    """AWQ scales the bank's readers in the model; the export copies the bank from disk, so the two disagree."""
    from qda_quant import REGISTRY

    g = describe(model("qwen3_moe"), verify=False, keep=frozenset({r"re:^model\.layers\.0\.mlp\.experts\."}))
    with pytest.raises(Refused, match="kept float by `ignore`"):
        check_float_banks(g, [REGISTRY["awq"]()])
    check_float_banks(g, [REGISTRY["rtn"]()])  # neither folds activation scales into a norm's readers
    check_float_banks(describe(model("qwen3_moe"), verify=False), [REGISTRY["awq"]()])  # no float bank


def _per_expert_graph(n_banks: int = 1) -> ModelGraph:
    """A graph whose bank stores its experts as per-expert linears (`_banks`, i.e. a ModuleList layout)."""
    p = "model.layers.0.mlp.experts"
    experts = tuple(((f"{p}.{e}.gate_proj", f"{p}.{e}.up_proj"), f"{p}.{e}.down_proj") for e in range(2))
    bank = ExpertBank(experts, "model.layers.0.mlp.gate", 1)
    return ModelGraph(
        model_type="test",
        layers=(LayerGraph(0, "model.layers.0", (), (bank,) * n_banks),),
        embed="model.embed_tokens",
        lm_head="lm_head",
        final_norm="model.norm",
    )


def test_ignore_keeps_a_whole_per_expert_bank_float():
    """Same rule for a bank of per-expert linears: its module, or all of its linears, or none of it."""
    ignored, g = _check_ignored(_per_expert_graph(), frozenset({"model.layers.0.mlp.experts"}), {})
    assert g.layers[0].banks == () and g.extra["float_banks"] == ("model.layers.0.mlp.experts",)
    assert ignored["model.layers.0.mlp.experts.1.down_proj"] == ("model.layers.0.mlp.experts", "mlp_out")
    assert ignored["model.layers.0.mlp.experts.1.up_proj"] == ("model.layers.0.mlp.experts", "mlp_in")
    with pytest.raises(Refused, match="routed expert"):
        _check_ignored(_per_expert_graph(), frozenset({r"re:^model\.layers\.0\.mlp\.experts\.0\."}), {})


@pytest.mark.parametrize("kind", ["qwen3", "llama", "qwen2", "qwen3_moe", "olmo2", "gemma3_vlm", "qwen3_5"])
def test_keep_changes_nothing_on_families_without_those_roles(kind):
    """The default keeps only GDN gates and shared experts: a family without them has the same groups either way."""
    m = model(kind)
    a, b = describe(m, verify=False), describe(m, verify=False, keep=frozenset())
    assert a.layers == b.layers and a.extra["kept_bf16"] == {}


def test_a_pure_mamba_traces_without_an_attention_mask():
    """Mamba's forward reads attention_mask as a 2-D padding mask; the trace passes none (both devices)."""
    import torch
    import transformers

    config = transformers.MambaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        state_size=16,
        conv_kernel=4,
        expand=2,
    )
    cpu = transformers.AutoModelForCausalLM.from_config(config)
    assert len(describe(cpu, verify=False).layers) == 2
    with torch.device("meta"):
        meta = transformers.AutoModelForCausalLM.from_config(config)
    graph = describe(meta, verify=False)  # the plan's weightless path
    assert len(graph.layers) == 2
    assert graph.layers[0].groups  # the traced forward revealed its linears
