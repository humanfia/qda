import pytest
import torch
from qda_quant import REGISTRY, Format
from qda_quant.solve.quant import Quantizer

from .ctx import contexts
from .tiny import model


def _check(ctx, name, w_now):
    qp = ctx.qparams[name]
    plan = ctx.plan_moe if "experts" in name else ctx.plan_dense
    q = Quantizer.of(ctx.fmt, plan, "absmax")
    served = q.dequant(qp["codes"].float(), qp["scale"].float(), qp.get("zero_point", None))
    assert torch.equal(served.to(w_now.dtype), w_now)
    assert qp["codes"].dtype == torch.int8 and qp["scale"].dtype == torch.bfloat16
    assert ("zero_point" in qp) == (not ctx.fmt.symmetric)


@pytest.mark.parametrize("step", ["rtn", "gptq"])
@pytest.mark.parametrize("fmt", ["w4a16", "w4a8", "w3a16"])
def test_dense_layer(step, fmt):
    m = model("qwen3")
    for ctx in contexts(m, fmt):
        REGISTRY[step]().apply_layer(ctx)
        for grp in ctx.graph.groups:
            for n in grp.modules:
                _check(ctx, n, m.get_submodule(n).weight)


def test_gptq_moe_bank():
    m = model("qwen3_moe")
    ctx = contexts(m, "w4a16")[0]
    REGISTRY["gptq"](impl="torch").apply_layer(ctx)
    bank = ctx.graph.banks[0]
    for p in ("gate_up_proj", "down_proj"):
        name = f"{bank.experts}.{p}"
        _check(ctx, name, getattr(m.get_submodule(bank.experts), p))
        assert ctx.qparams[name]["codes"].dim() == 3


def test_gptq_grid3_w4a8_tiny_model():
    m = model("qwen3")
    step = REGISTRY["gptq"](clip="grid3", impl="torch")
    contexts_for_model = contexts(m, "w4a8")
    step.check(contexts_for_model[0].model_graph, Format.parse("w4a8"), contexts_for_model[0].plan_dense)
    for ctx in contexts_for_model:
        step.apply_layer(ctx)
        for grp in ctx.graph.groups:
            for name in grp.modules:
                _check(ctx, name, m.get_submodule(name).weight)


@pytest.mark.parametrize("magnitude", [0.0, 1e-20, 1e-10])
@pytest.mark.parametrize("grid_k", [1, 2])
def test_gptq_grid3_zero_and_tiny_rows_stay_finite(magnitude, grid_k):
    tiny_model = model("qwen3")
    ctx = contexts(tiny_model, "w4a8")[0]
    name = ctx.graph.groups[0].modules[0]
    weight = tiny_model.get_submodule(name).weight
    with torch.no_grad():
        weight[0].fill_(magnitude)
    REGISTRY["gptq"](clip="grid3", grid_k=grid_k, impl="torch").apply_layer(ctx)
    for group in ctx.graph.groups:
        for module_name in group.modules:
            params = ctx.qparams[module_name]
            quantizer = Quantizer.of(ctx.fmt, ctx.plan_dense, "absmax")
            assert torch.isfinite(params["scale"]).all()
            assert (quantizer.served_scale(params["scale"]) > 0).all()
            assert (params["codes"] >= ctx.fmt.qmin).all()
            assert (params["codes"] <= ctx.fmt.qmax).all()
            assert torch.isfinite(tiny_model.get_submodule(module_name).weight).all()
            _check(ctx, module_name, tiny_model.get_submodule(module_name).weight)
    assert torch.count_nonzero(weight[0]) == 0


def test_grid3_reach_is_a_validated_recipe_setting():
    from pydantic import ValidationError
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe

    assert REGISTRY["gptq"]().grid_k == 1
    recipe = Recipe(steps=[REGISTRY["gptq"](clip="grid3", grid_k=2)], calib=FileSource(path="unused"))
    assert Recipe.from_yaml(recipe.to_yaml()).steps[0].grid_k == 2
    with pytest.raises(ValidationError):
        REGISTRY["gptq"](grid_k=-1)


def test_gptq_lowers_the_hessian_weighted_error_vs_rtn():
    loss = {}
    for step in ("rtn", "gptq"):
        m = model("qwen3")
        w0 = {n: p.detach().clone().float() for n, p in m.named_parameters()}
        total = 0.0
        for ctx in contexts(m, "w4a16-sym-g128", tokens=256):
            REGISTRY[step]().apply_layer(ctx)
            for grp in ctx.graph.groups:
                h = ctx.stats[("hessian", grp.modules[0])]
                for n in grp.modules:
                    d = m.get_submodule(n).weight.float() - w0[n + ".weight"]
                    total += ((d @ h) * d).sum().item()
        loss[step] = total
    assert loss["gptq"] < 0.9 * loss["rtn"]


@pytest.mark.parametrize("name", ["gptaq", "permute", "gsq"])
def test_the_experimental_methods_are_marked(name):
    """docs/recipe.md calls these three experimental, and the flag is what a caller reads."""
    assert REGISTRY[name]().experimental


@pytest.mark.parametrize("name", ["gptq", "rtn", "awq"])
def test_the_recommended_methods_are_not_marked_experimental(name):
    assert not REGISTRY[name]().experimental


def test_needs_follow_settings():
    assert REGISTRY["rtn"]().needs == frozenset()
    assert "hessian" in REGISTRY["rtn"](clip="imse").needs
    assert REGISTRY["gptq"]().needs == {"hessian", "expert_hessian"}


def test_unknown_clip_is_refused_with_the_registered_names():
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match=r"unknown clip 'nope'; known: \['fp4hmse', 'fp4mse', 'grid3', 'hmse', 'imatrix', 'imse', 'minmax', 'mse'\]",
    ):
        REGISTRY["rtn"](clip="nope")


@pytest.mark.parametrize("fmt", ["w4a8", "w4a16"])
def test_hessian_weighted_clip_beats_minmax(fmt):
    import torch
    from qda_quant import Format, Target
    from qda_quant.clips import CLIPS

    f = Format.parse(fmt)
    q = Quantizer.of(f, Target().plan(f, "dense"), "absmax")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(1, 64, 512, generator=g) * torch.linspace(0.5, 2, 512)
    hd = torch.rand(1, 512, generator=g) ** 4
    loss = {}
    for clip in ("minmax", "hmse"):
        s, z = CLIPS[clip].choose(q, w, hd)
        loss[clip] = ((q.fake(w, s, z) - w).pow(2) * hd[:, None]).sum().item()
    assert loss["hmse"] < loss["minmax"]
