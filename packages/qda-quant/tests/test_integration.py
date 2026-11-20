"""engine.build + the real steps + the real Writer, on tiny saved models (CPU)."""

import json
from typing import ClassVar

import pytest
import torch
import transformers
from qda_quant import REGISTRY, Format, Recipe, Step
from qda_quant.engine import build
from qda_quant.intake.fisher import load_bf16
from safetensors.torch import load_file

from .engine_util import save_tiny_tokenizer, write_ids
from .tiny import model


def _setup(tmp_path, kind):
    m = model(kind)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")  # token-id calibration must carry the tokenizer's fingerprint
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (64,), generator=g).tolist() for _ in range(8)], sha
    )
    return m


@pytest.mark.parametrize(
    ("kind", "steps", "fmt"),
    [
        ("qwen3", ["gptq"], "w4a16-sym-g128"),
        ("qwen3", ["awq", "rtn"], "w4a16"),
        ("qwen3", ["gptq"], "w4a8"),
        ("qwen3", ["autoround"], "w4a16"),
        # asym fused experts: transformers raises (it cannot collect the zero points)
        ("qwen3_moe", ["gptq"], "w4a16-sym-g128"),
    ],
)
def test_build_loads_and_stays_close(tmp_path, kind, steps, fmt):
    ref = _setup(tmp_path, kind)
    recipe = Recipe.from_dict(
        {
            "steps": [{s: {}} for s in steps],
            "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}},
        }
    )
    out = build(str(tmp_path / "m"), recipe, format=fmt, out_dir=tmp_path / "out", devices=["cpu"],
                dtype=torch.float32, log=lambda s: None)  # fmt: skip
    info = json.loads((out / "build.json").read_text())
    assert info["serve"]["kernels"]["dense"] and ("moe" in info["serve"]["kernels"]) == (kind == "qwen3_moe")
    kl = _kl(ref, out)
    assert 0 < kl < 0.5, kl  # smoke bound only: loads and is not garbage
    # The real bound: a calibrated method must beat round-to-nearest on the same build. AWQ is exempt here:
    # random tiny weights have no outlier channels, so it has nothing to gain; its bound is
    # output/test_awq.py::test_awq_then_rtn_beats_rtn_on_skewed_inputs.
    if steps != ["rtn"] and "awq" not in steps:
        rtn = Recipe.from_dict({"steps": [{"rtn": {}}], "calib": recipe.calib.to_dict()})
        out_rtn = build(str(tmp_path / "m"), rtn, format=fmt, out_dir=tmp_path / "rtn", devices=["cpu"],
                        dtype=torch.float32, log=lambda s: None)  # fmt: skip
        assert kl < _kl(ref, out_rtn), (kl, steps)


def test_asymmetric_moe_checkpoint_loads_with_zero_points(tmp_path):
    """An RTN w4a16-asym-g128 fused-expert checkpoint is asymmetric; `qda fisher map` loads it with its zero points.

    transformers 5.21 cannot collect ``weight_zero_point`` for fused experts and raises. A load that dropped
    them instead would be wrong by far more than quantization here: KL 0.045 against 0.010, expert weight max
    difference 0.15 against 0.027.
    """
    ref = _setup(tmp_path, "qwen3_moe")
    out = build(str(tmp_path / "m"), _recipe(tmp_path, ["rtn"]), format="w4a16-asym-g128",
                out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.float32, log=lambda s: None)  # fmt: skip
    tensors = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}
    zps = [v for k, v in tensors.items() if k.endswith("experts.0.gate_proj.weight_zero_point")]
    assert zps and all(int(z.abs().sum()) > 0 for z in zps)  # the checkpoint is really asymmetric

    q = load_bf16(out, "cpu")
    ids = torch.randint(0, 128, (4, 64), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        p, lq = ref(input_ids=ids).logits.log_softmax(-1), q(ids).logits.log_softmax(-1)
    assert 0 < (p.exp() * (p - lq)).sum(-1).mean().item() < 0.02
    a = q.model.layers[0].mlp.experts.gate_up_proj.float()
    b = ref.model.layers[0].mlp.experts.gate_up_proj.float()
    assert (a - b).abs().max().item() < 0.05  # quantization only; dropping the zero points is about 0.15


def test_head_step_in_either_order_writes_the_same_checkpoint(tmp_path):
    """[head, rtn] and [rtn, head] give byte-equal tensors: the body's codes from the first layer pass stay in the
    build's qparams dict until the last pass writes them, and the head's codes come from the same dict."""
    from safetensors.torch import load_file

    _setup(tmp_path, "qwen3")
    outs = []
    for i, steps in enumerate((["head", "rtn"], ["rtn", "head"])):
        r = Recipe.from_dict({"steps": [{s: {}} for s in steps],
                              "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}}})  # fmt: skip
        out = build(str(tmp_path / "m"), r, format="w4a16", out_dir=tmp_path / f"o{i}", devices=["cpu"],
                    dtype=torch.float32, log=lambda s: None)  # fmt: skip
        outs.append({k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()})
    a, b = outs
    assert a.keys() == b.keys() and "lm_head.weight_packed" in a
    assert any(k.endswith("q_proj.weight_packed") for k in a)  # the body is stored quantized in both
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_head_is_planned_and_quantized_on_the_builds_target(tmp_path, monkeypatch):
    """The head's own format is served by the build's target, not a default one: this target refuses an int8
    head and stores W4 group scales as FP8, where the default target accepts int8 and rounds them to bf16."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant.solve.quant import Quantizer
    from qda_quant.step import Refused
    from qda_quant.target import BINDINGS, TARGETS, KernelPlan, Target
    from safetensors import safe_open

    def table(target, fmt, kind):  # noqa: ANN001, ANN202
        if fmt.weight_bits == 8:
            raise Refused("test target serves no int8")
        return KernelPlan("TestKernel", scale_dtype="fp8_e4m3")

    def binds(target, fmt, shape, args, env):  # noqa: ANN001, ANN202
        return "TestKernel"

    monkeypatch.setitem(TARGETS._entries, "testhead", table)
    monkeypatch.setitem(BINDINGS._entries, "testhead", binds)
    m = _setup(tmp_path, "qwen3")

    def build_head(head, out):
        r = Recipe.from_dict({"steps": [{"rtn": {}}, {"head": {"format": head}}],
                              "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}}})  # fmt: skip
        return build(str(tmp_path / "m"), r, format="w4a16", target="testhead", out_dir=tmp_path / out,
                     devices=["cpu"], dtype=torch.float32, log=lambda s: None)  # fmt: skip

    with pytest.raises(Refused, match="serves no int8"):
        build_head("w8a16", "refused")
    out = build_head("w4a16-sym-g64", "out")
    f = Format.parse("w4a16-sym-g64")
    w = m.lm_head.weight.float()
    served, default = (Quantizer.of(f, p, "absmax") for p in (Target("testhead").plan(f, "dense"),
                                                               Target().plan(f, "dense")))  # fmt: skip
    s, _ = served.params(w)
    sd, _ = default.params(w)
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(out / idx["lm_head.weight_packed"], "pt") as h:
        shape = torch.Size(h.get_tensor("lm_head.weight_shape").tolist())
        codes = unpack_from_int32(h.get_tensor("lm_head.weight_packed"), f.weight_bits, shape).float()
    assert torch.equal(codes, served.quantize(w, s, None))  # the target's FP8-scale codes
    assert not torch.equal(codes, default.quantize(w, sd, None))  # not the default target's bf16-scale ones


def _recipe(tmp_path, steps):
    return Recipe.from_dict({"steps": [{s: {}} for s in steps],
                             "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}}})  # fmt: skip


@pytest.mark.parametrize("steps", [["rotate", "gptq"], ["rtn", "head"]])
def test_model_step_before_layers_or_preserving_builds(tmp_path, steps):
    """A body-changing model step before the layer steps, and a preserving one (head) after them, build.
    ([gptq, rotate] is refused at plan time: test_engine_build.py.)"""
    _setup(tmp_path, "qwen3")
    out = build(str(tmp_path / "m"), _recipe(tmp_path, steps), format="w4a16", out_dir=tmp_path / "out",
                devices=["cpu"], dtype=torch.float32, log=lambda s: None)  # fmt: skip
    assert (out / "model.safetensors.index.json").is_file()


class LyingHead(Step):
    """Claims to keep quantized weights but scales layer 0's q_proj (a buggy step). Module level: pickled."""

    kind: ClassVar[str] = "model"
    name: ClassVar[str] = "lying_head_test"
    preserves_quantized: ClassVar[bool] = True

    def apply_model(self, model, graph, run):
        model.model.layers[0].self_attn.q_proj.weight.data.mul_(2)


def test_export_refuses_codes_of_a_changed_weight(tmp_path):
    """Defense in depth behind the plan rule: the Writer checks each quantized weight's identity and checksum
    against the ones recorded with its codes."""
    _setup(tmp_path, "qwen3")
    r = Recipe(steps=[REGISTRY["rtn"](), LyingHead()], calib=_recipe(tmp_path, []).calib)
    with pytest.raises(RuntimeError, match="q_proj: the weight changed after its codes were recorded"):
        build(str(tmp_path / "m"), r, format="w4a16", out_dir=tmp_path / "out", devices=["cpu"],
              dtype=torch.float32, log=lambda s: None)  # fmt: skip


def _kl(ref, out) -> float:
    """Token-mean KL(ref || quantized) on 4 x 64 held-out random tokens."""
    q = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = torch.randint(0, 128, (4, 64), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        p, lq = ref(input_ids=ids).logits.log_softmax(-1), q(input_ids=ids).logits.log_softmax(-1)
    return (p.exp() * (p - lq)).sum(-1).mean().item()


def test_autoround_beats_rtn(tmp_path):
    ref = _setup(tmp_path, "qwen3")
    kls = {}
    for name, cfg in (("rtn", {}), ("autoround", {"iters": 60})):
        recipe = Recipe.from_dict(
            {
                "steps": [{name: cfg}],
                "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}},
            }
        )
        out = build(str(tmp_path / "m"), recipe, format="w4a16-sym-g128", out_dir=tmp_path / name,
                    devices=["cpu"], dtype=torch.float32, log=lambda s: None)  # fmt: skip
        q = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
        ids = torch.randint(0, 128, (4, 64), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            p, lq = ref(input_ids=ids).logits.log_softmax(-1), q(input_ids=ids).logits.log_softmax(-1)
        kls[name] = (p.exp() * (p - lq)).sum(-1).mean().item()
    assert kls["autoround"] < kls["rtn"], kls


@pytest.mark.parametrize("objective", ["parent", "linear_out"])
def test_awq_objectives(tmp_path, monkeypatch, objective):
    from qda_quant.steps import awq

    calls = []
    real = awq._attn_out
    monkeypatch.setattr(awq, "_attn_out", lambda *a: calls.append(1) or real(*a))
    _setup(tmp_path, "qwen3")
    recipe = Recipe.from_dict(
        {"steps": [{"awq": {"objective": objective}}, {"rtn": {}}],
         "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 8, "seqlen": 64}}}
    )  # fmt: skip
    build(str(tmp_path / "m"), recipe, format="w4a16", out_dir=tmp_path / "out", devices=["cpu"],
          dtype=torch.float32, log=lambda s: None)  # fmt: skip
    # parent: per layer, (1 reference + 20 candidates) x 8 batches of attention re-runs
    assert len(calls) == (2 * 21 * 8 if objective == "parent" else 0)
