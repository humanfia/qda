"""NVFP4 expert banks: one input scale per bank projection, cut calibration inputs, per-expert export."""

import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.quantization import preset_name_to_scheme
from qda_quant import Recipe
from qda_quant.arch import describe
from qda_quant.arch.adapters import fused_experts
from qda_quant.engine import build
from qda_quant.engine.execute import _qparam_names, act_inputs, bank_cut, input_scales
from qda_quant.engine.stats import LayerStats
from qda_quant.format import Format
from qda_quant.solve import nvfp4
from qda_quant.solve.nvfp4 import block_cut, input_global_scale
from qda_quant.solve.quant import ACT_CUTS
from qda_quant.target import KernelPlan, Target
from safetensors.torch import load_file

from .engine_util import save_tiny_tokenizer, write_ids
from .tiny import model

_SCHEME = preset_name_to_scheme("NVFP4", ["Linear"])
_BANK = "model.layers.0.mlp.experts"


def _setup(tmp_path, kind="qwen3_moe"):
    m = model(kind)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (64,), generator=g).tolist() for _ in range(8)], sha
    )
    return m


def _checkpoint(out):
    return {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}


def _nvfp4_recipe(tmp_path):
    return Recipe.from_dict(
        {
            "steps": [{"rtn": {}}],
            "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}},
        }
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One tiny qwen3_moe built at w4a4-nvfp4/sm100 for the layout tests (the build is the slow part)."""
    tmp = tmp_path_factory.mktemp("nvfp4-moe")
    m = _setup(tmp)
    out = build(
        str(tmp / "m"), _nvfp4_recipe(tmp), format="w4a4-nvfp4", target="sm100",
        out_dir=tmp / "out", devices=["cpu"], dtype=torch.float32, log=lambda s: None,
    )  # fmt: skip
    return m, out, _checkpoint(out)


def test_qparam_names_are_every_name_of_each_name_shape():
    """Dense group, fused bank and per-expert Linear bank, flattened without dropping gate/up."""
    assert _qparam_names(("a", "b")) == ["a", "b"]
    assert _qparam_names(_BANK + ".gate_up_proj") == [_BANK + ".gate_up_proj"]
    assert _qparam_names((("a.gate_proj", "a.up_proj"), ("b.gate_proj", "b.up_proj"))) == [
        "a.gate_proj",
        "a.up_proj",
        "b.gate_proj",
        "b.up_proj",
    ]
    assert _qparam_names((("a.down_proj",), ("b.down_proj",))) == ["a.down_proj", "b.down_proj"]


@pytest.mark.parametrize("part", ["in", "down"])
def test_bank_input_scale_is_the_largest_over_experts(part):
    """`input_scales` sets one scalar on every entry of a bank projection, from the experts' largest range."""
    m = model("qwen3_moe")
    graph = describe(m)
    lg = graph.layers[0]
    (bank,) = lg.banks
    attr = fused_experts.in_param(m.model.layers[0].mlp.experts) if part == "in" else fused_experts.DOWN
    name = f"{bank.experts}.{attr}"
    qparams = {name: {"format": "w4a4-nvfp4"}}
    # the experts saw different ranges; the first is the widest
    seen = torch.full((8, 32), 0.5)
    seen[0, 3] = 91.0
    ctx = SimpleNamespace(
        layer=m.model.layers[0],
        graph=lg,
        qparams=qparams,
        stats={("act_absmax", bank.router): {"in": seen.clone(), "down": seen.clone()}},
    )
    input_scales(ctx)
    scale = qparams[name]["input_global_scale"]
    assert scale.shape == () and scale.dtype == torch.float32
    assert torch.equal(scale, input_global_scale(seen))  # a power of two over the largest expert range
    assert torch.equal(scale, torch.tensor(16.0))  # 2688 / 91., rounded down to a power of two


@pytest.mark.parametrize("part", ["in", "down"])
def test_calibration_cuts_both_bank_inputs(part):
    """The bank's statistics read the kernel's cut input: the routed tokens for gate/up, act(gate)*up for down."""
    m = model("qwen3_moe")
    lg = describe(m).layers[0]
    layer = m.model.layers[0]
    calls = []

    def record(module, args):
        calls.append(args)

    h = layer.mlp.experts.register_forward_pre_hook(record)
    try:
        with (
            torch.no_grad(),
            # as the engine installs an NVFP4 plan, before the statistics hooks
            act_inputs({g.modules[0]: block_cut for g in lg.groups}, layer, lg, bank_qdq=block_cut),
            LayerStats(layer, lg, frozenset({"act_absmax"}), torch.device("cpu"), cut=block_cut) as st,
        ):
            m(torch.randint(0, 128, (2, 8), generator=torch.Generator().manual_seed(0)))
    finally:
        h.remove()
    ((x, idx, w),) = calls
    got = st.result()[("act_absmax", lg.banks[0].router)][part]
    assert torch.equal(got, _reference_absmax(layer.mlp.experts, x, idx, w, part))
    # the cut is what makes it differ from the raw input
    assert not torch.equal(got, _reference_absmax(layer.mlp.experts, x, idx, w, part, cut=None))


def test_only_the_nvfp4_plan_cuts_a_bank():
    """`bank_cut` is the NVFP4 round trip on NVFP4's row and None on every other act dtype (the released rows)."""
    assert bank_cut(None) is None
    for act_dtype in ("bf16", "fp8_e4m3", "int8"):
        assert bank_cut(KernelPlan("SomeExperts", act_dtype=act_dtype)) is None
    assert bank_cut(KernelPlan("TrtLlmNvFp4ExpertsMonolithic", act_dtype="nvfp4")) is block_cut
    # the tables the released recipes use: W4A8 on sm90 has no bank cut, W4A4 on sm100 does
    assert bank_cut(Target(arch="sm90").plan(Format.parse("w4a8"), "moe")) is None
    assert bank_cut(Target(arch="sm100").plan(Format.parse("w4a4-nvfp4"), "moe")) is block_cut
    # a self-routing bank binds the modular kernel: same grid, same cut
    modular = Target(arch="sm100").plan(Format.parse("w4a4-nvfp4"), "moe", decomposed=True)
    assert modular.kernel == "TrtLlmNvFp4ExpertsModular" and bank_cut(modular) is block_cut


def test_the_bank_records_whether_its_router_routes_itself():
    """Gemma 4's router applies a per-expert scale (custom routing); Qwen3 MoE's stock softmax top-k does not."""
    gemma = describe(model("gemma4_moe"))
    assert gemma.layers[0].banks[0].custom_router
    qwen = describe(model("qwen3_moe"))
    assert not qwen.layers[0].banks[0].custom_router


@pytest.mark.parametrize("act_dtype", ["fp8_e4m3", "int8"])
def test_other_act_dtypes_hook_no_bank_and_cut_no_down_input(act_dtype):
    """Only NVFP4 reads a cut bank: the W4A8/W8A8 MoE recipes were solved on the plain expert inputs."""
    m = model("qwen3_moe").eval()
    lg = describe(m).layers[0]
    layer = m.model.layers[0]
    cut = ACT_CUTS[act_dtype]
    assert cut is not None
    x = torch.randint(0, 128, (2, 8), generator=torch.Generator().manual_seed(0))

    def run(bank_qdq):
        """Run under the dense cut; return the bank's inputs and its down statistics as the engine measures them.

        The recorder is registered inside `act_inputs`, so it sees what the bank's forward is called with (a
        bank cut, when one is installed, runs first).
        """
        calls = []
        with (
            torch.no_grad(),
            act_inputs({g.modules[0]: cut.qdq for g in lg.groups}, layer, lg, bank_qdq=bank_qdq),
            LayerStats(layer, lg, frozenset({"act_absmax"}), torch.device("cpu"), cut=bank_qdq) as st,
        ):
            h = layer.mlp.experts.register_forward_pre_hook(lambda _m, args: calls.append(args))
            try:
                m(x)
            finally:
                h.remove()
        ((x_bank, idx, w),) = calls
        return x_bank, idx, w, st.result()[("act_absmax", lg.banks[0].router)]["down"]

    x_plain, idx, w, down_plain = run(None)  # what `Pass` installs for a non-NVFP4 plan
    x_cut, _, _, _ = run(cut.qdq)  # the same run with an NVFP4-style bank cut
    assert not torch.equal(x_plain, x_cut)
    assert torch.equal(x_cut, cut.qdq(x_plain))  # bank_qdq is what cuts; without it the bank sees `x_plain`
    # the down input is measured as the module forms it, with no cut (the reference loops per expert where the
    # statistics use one bmm, so fp32 rounding can differ in the last bits)
    torch.testing.assert_close(
        down_plain,
        _reference_absmax(layer.mlp.experts, x_plain, idx, w, "down", cut=None),
        rtol=1e-4,
        atol=1e-8,
    )
    assert not torch.equal(
        down_plain, _reference_absmax(layer.mlp.experts, x_plain, idx, w, "down", cut=cut.qdq)
    )


def _reference_absmax(experts, x, idx, w, part, cut=block_cut):
    """Per-expert loop (fp32, like `engine.stats.reference_expert_stats`): the largest |input| each expert saw.

    The bank reads the cut routed tokens; the down input is formed from them and cut again.
    """
    Wgu = fused_experts.in_weight(experts)
    bgu = fused_experts.bias(experts, fused_experts.in_param(experts))
    xr = cut(x) if cut is not None else x
    out = torch.zeros(Wgu.shape[0], x.shape[-1] if part == "in" else Wgu.shape[1] // 2)
    for e in range(Wgu.shape[0]):
        t, s = torch.where(idx == e)
        if t.numel() == 0:
            continue
        if part == "in":
            rows = xr[t]
        else:
            projected = torch.nn.functional.linear(xr[t], Wgu[e])
            if bgu is not None:
                projected = projected + bgu[e]
            rows = fused_experts.down_input(experts, projected, w[t, s])
            if cut is not None:
                rows = cut(rows)
        out[e] = rows.abs().amax(0)
    return out


def _served(w, scale, gscale):
    """The weight an NVFP4 kernel applies: `w` rounded on the stored step `scale / gscale`."""
    s, g = scale.float(), gscale.float()
    return nvfp4.dequant(nvfp4.quantize(w.float(), s, g), s, g)


def test_nvfp4_expert_bank_is_stored_one_linear_per_expert(built):
    """The sm100 MoE row serves per-expert packed Linears, with one input scale per bank projection and one
    weight global scale per expert for gate+up (vLLM reads the gate's)."""
    m, out, t = built
    info = json.loads((out / "build.json").read_text())
    assert info["serve"]["kernels"]["moe"] == ["TrtLlmNvFp4ExpertsMonolithic"]
    experts = m.model.layers[0].mlp.experts
    Wgu, Wd = fused_experts.in_weight(experts), fused_experts.down_weight(experts)
    n, rows, _ = Wgu.shape
    scales = {"in": set(), "down": set()}
    for e in range(n):
        unit_g = {
            "in": nvfp4.lattice_global(
                nvfp4.lattice(Wgu[e].float().unflatten(-1, (-1, 16)).abs().amax(-1) / 6)
            ),
            "down": nvfp4.lattice_global(
                nvfp4.lattice(Wd[e].float().unflatten(-1, (-1, 16)).abs().amax(-1) / 6)
            ),
        }
        for proj, part, unit in (
            ("gate_proj", Wgu[e, : rows // 2], "in"),
            ("up_proj", Wgu[e, rows // 2 :], "in"),
            ("down_proj", Wd[e], "down"),
        ):
            stem = f"{_BANK}.{e}.{proj}"
            assert {k for k in t if k.startswith(stem + ".")} == {
                f"{stem}.{field}"
                for field in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
            }
            g = t[f"{stem}.weight_global_scale"]
            assert g.shape == (1,) and torch.equal(g, unit_g[unit].reshape(1))  # gate and up share one
            back = NVFP4PackedCompressor.decompress(
                {
                    "weight_packed": t[f"{stem}.weight_packed"],
                    "weight_scale": t[f"{stem}.weight_scale"],
                    "weight_global_scale": g,
                },
                _SCHEME,
            )["weight"]
            assert back.shape == part.shape
            # the stored codes are `part` on its unit's stored scale, and decode to the served weight
            scale = t[f"{stem}.weight_scale"]
            assert torch.equal(back.float(), _served(part, scale, g).to(torch.bfloat16).float())
            assert t[f"{stem}.weight_packed"].dtype == torch.uint8
            assert scale.dtype == torch.float8_e4m3fn
            (scales["down"] if proj == "down_proj" else scales["in"]).add(
                t[f"{stem}.input_global_scale"].item()
            )
        gate = t[f"{_BANK}.{e}.gate_proj.weight_global_scale"]
        assert torch.equal(gate, t[f"{_BANK}.{e}.up_proj.weight_global_scale"])  # vLLM reads the gate's
    assert len(scales["in"]) == 1 and len(scales["down"]) == 1  # one scale per bank projection
    # the build ran its calibration forward and read the bank's ranges (an unread stat would leave the
    # default 1.0, which is what a data-free build used to write)
    assert next(iter(scales["in"])) > 1 and next(iter(scales["down"])) > 1


def test_nvfp4_expert_bank_layout_is_golden(built):
    """The names, dtypes and shapes vLLM 0.33 loads for an NVFP4 expert bank, spelled out (RedHat's layout)."""
    m, _, t = built
    hidden, inter = m.config.hidden_size, m.config.moe_intermediate_size
    golden = {}
    for proj, (rows, cols) in {
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }.items():
        stem = f"{_BANK}.0.{proj}."
        golden[f"{stem}weight_packed"] = (torch.uint8, (rows, cols // 2))  # E2M1 pairs
        golden[f"{stem}weight_scale"] = (torch.float8_e4m3fn, (rows, cols // 16))  # one per 16 inputs
        golden[f"{stem}weight_global_scale"] = (torch.float32, (1,))
        golden[f"{stem}input_global_scale"] = (torch.float32, (1,))
    assert {k: (v.dtype, tuple(v.shape)) for k, v in t.items() if k.startswith(f"{_BANK}.0.")} == golden
    # one set per expert, no stacked gate_up tensor, gate and up sharing one weight global scale per expert
    for e in range(m.config.num_experts):
        assert {k.replace(f"{_BANK}.0.", f"{_BANK}.{e}.") for k in golden} <= set(t)
        assert torch.equal(
            t[f"{_BANK}.{e}.gate_proj.weight_global_scale"],
            t[f"{_BANK}.{e}.up_proj.weight_global_scale"],
        )
    assert not any("gate_up_proj" in k for k in t)
    # one input scale per bank projection, shared by every expert (vLLM reduces the per-expert scales)
    in_scales = {t[f"{_BANK}.{e}.gate_proj.input_global_scale"].item() for e in range(m.config.num_experts)}
    down_scales = {t[f"{_BANK}.{e}.down_proj.input_global_scale"].item() for e in range(m.config.num_experts)}
    assert len(in_scales) == 1 and len(down_scales) == 1
    assert in_scales == {t[f"{_BANK}.0.up_proj.input_global_scale"].item()}


@pytest.fixture(scope="module")
def built_gemma(tmp_path_factory):
    """One tiny Gemma 4 MoE built at w4a4-nvfp4/sm100; its router scales each expert, so it routes itself."""
    tmp = tmp_path_factory.mktemp("nvfp4-gemma")
    _setup(tmp, kind="gemma4_moe")
    return build(
        str(tmp / "m"), _nvfp4_recipe(tmp), format="w4a4-nvfp4", target="sm100",
        out_dir=tmp / "out", devices=["cpu"], dtype=torch.float32, log=lambda s: None,
    )  # fmt: skip


def test_a_self_routing_bank_records_the_modular_expert_kernel(built_gemma):
    """Gemma 4's router applies a per-expert scale, so vLLM binds the modular experts kernel: the build's
    record names it (the monolithic kernel takes only stock routing methods)."""
    info = json.loads((built_gemma / "build.json").read_text())
    assert info["serve"]["kernels"]["moe"] == ["TrtLlmNvFp4ExpertsModular"]
    assert info["serve"]["kernels"]["dense"] == ["FlashInferCuteDslNvFp4LinearKernel"]
