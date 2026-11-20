import pytest
import torch
from qda_quant import REGISTRY

from ..ctx import contexts
from ..tiny import model


@pytest.mark.parametrize("kind", ["qwen3", "qwen3-mha", "llama", "opt"])  # opt: LayerNorms with a bias
def test_awq_fold_keeps_the_float_model(kind):
    m = model(kind, dtype=torch.float64)
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    ctxs = contexts(m, "w4a16")
    for ctx in ctxs:
        REGISTRY["awq"]().apply_layer(ctx)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()  # eager attention softmaxes in fp32
    # the folds changed something, and the rescaled Hessians are what a re-forward would measure
    fresh = contexts(m, "w4a16")
    for a, b in zip(ctxs, fresh, strict=True):
        for grp in a.graph.groups:
            ha, hb = a.stats[("hessian", grp.modules[0])], b.stats[("hessian", grp.modules[0])]
            assert (ha - hb).abs().max() < 1e-5 * hb.abs().max()  # fp32 Hessians


def test_awq_then_rtn_beats_rtn_on_skewed_inputs():
    errs = {}
    for recipe in (["rtn"], ["awq", "rtn"]):
        m = model("qwen3")
        with torch.no_grad():  # outlier channels, the case AWQ exists for
            for n, p in m.named_parameters():
                if n.endswith("layernorm.weight"):
                    p.mul_(torch.where(torch.arange(p.numel()) % 16 == 0, 20.0, 1.0))
        ref = model("qwen3")
        ref.load_state_dict(m.state_dict())
        ids = torch.randint(0, 128, (1, 128), generator=torch.Generator().manual_seed(0))
        for ctx in contexts(m, "w4a16", tokens=256):
            for step in recipe:
                REGISTRY[step]().apply_layer(ctx)
        with torch.no_grad():
            errs[tuple(recipe)] = (m(input_ids=ids).logits - ref(input_ids=ids).logits).pow(2).mean().item()
    assert errs[("awq", "rtn")] < errs[("rtn",)]


@pytest.mark.parametrize("duo", [True, False])
def test_default_grid_is_llm_compressors(duo):
    awq_mod = pytest.importorskip("llmcompressor.modifiers.transform.awq.base")
    ref = awq_mod.AWQModifier(duo_scaling=duo, n_grid=20)._get_grid_search_params()
    assert REGISTRY["awq"](duo_scaling=duo).candidates() == ref


def test_linear_out_keeps_its_measured_grid():
    g = REGISTRY["awq"](objective="linear_out").candidates()
    assert g == [(i / 20, True) for i in range(20)]


@pytest.mark.gpu
def test_compiled_pricing_matches_the_eager_reference():
    """On a card the search prices candidates through torch.compile (`awq._fast`); the eager functions are the
    reference. Each compiled function must compute what its eager one does, within bf16 rounding: the attention
    block (norms, rotary, sdpa, o_proj), the MLP, a bare linear, and the float32 squared error.

    Fusion reorders floating-point arithmetic, so a near-tie between two candidates can resolve differently and
    the build is not byte-identical; that is judged by KL on a real model, not here.
    """
    from qda_quant.steps import awq

    from ..engine_util import tiny_qwen3

    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    assert awq._fast(awq._se, cpu) is awq._se
    layer_model = tiny_qwen3(layers=1, hidden=128, inter=256).to(cuda, torch.bfloat16)
    layer = layer_model.model.layers[0]
    g = torch.Generator(device=cuda).manual_seed(0)
    x = torch.randn(2, 64, 128, device=cuda, dtype=torch.bfloat16, generator=g)
    pos = torch.arange(64, device=cuda)[None].expand(2, -1)
    kw = {"position_embeddings": layer_model.model.rotary_emb(x, pos), "attention_mask": None}
    with torch.no_grad():
        for fn, args in (
            (awq._attn_out, (layer.self_attn, x, kw)),
            (awq._run, (layer.mlp, [], x)),
            (awq._run, (None, [layer.self_attn.o_proj], x)),
        ):
            fast = awq._fast(fn, cuda)
            assert fast is not fn
            torch.testing.assert_close(fast(*args).float(), fn(*args).float(), rtol=2e-2, atol=2e-2)
        out, ref = layer.mlp(x), layer.mlp(x * 1.01)
        torch.testing.assert_close(awq._fast(awq._se, cuda)(out, ref), awq._se(out, ref), rtol=1e-5, atol=0)
