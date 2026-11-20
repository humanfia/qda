"""AWQ packing must preserve CT dequant bits, and float fusions must never be built quantized."""

import json
from dataclasses import replace

import pytest
import torch
import yaml
from qda_quant import REGISTRY, Target
from qda_quant.arch import describe
from qda_quant.config import BuildConfig
from qda_quant.output.awq import AWQWriter, pack_awq
from qda_quant.output.export import Writer
from qda_quant.output.exporters import parse_exporter
from qda_quant.solve.packing import unpack
from qda_quant.step import Refused
from safetensors.torch import load_file

from ..ctx import contexts
from ..engine_util import save_tiny_tokenizer
from ..tiny import model


def _tensors(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    return {key: value for shard in set(index.values()) for key, value in load_file(path / shard).items()}


def _awq_codes(words):
    """Slow reference: AWQ words hold logical columns in 0,2,4,6,1,3,5,7 order."""
    lanes = [(words.to(torch.int64) >> (4 * bit)) & 15 for bit in range(8)]
    return torch.stack([lanes[bit] for bit in (0, 4, 1, 5, 2, 6, 3, 7)], dim=-1).flatten(-2)


@pytest.mark.parametrize("kind,fmt", [("qwen3_moe", "w4a16-asym-g128"), ("qwen3_5_moe", "w4a16-asym-g64")])
def test_tiny_moe_awq_and_ct_have_identical_dequantized_bits(tmp_path, kind, fmt):
    """All expert/dense matrices and untouched floats agree; BF16 subnormals survive packing."""
    m = model(kind, dtype=torch.bfloat16)
    source = tmp_path / "src"
    m.save_pretrained(source)
    graph = describe(m, verify=False)
    ct = Writer(tmp_path / "ct", source, graph, fmt, Target())
    awq = AWQWriter(tmp_path / "awq", source, graph, fmt, Target())
    for ctx in contexts(m, fmt, tokens=4):
        REGISTRY["rtn"]().apply_layer(ctx)
        # A scale that FP16 rounds to zero must remain present in both layouts.
        next(iter(ctx.qparams.values()))["scale"].flatten()[0] = torch.tensor(2.0**-130).bfloat16()
        ct.add_layer(ctx)
        awq.add_layer(ctx)
    ct.finish(m.config, None, {})
    awq.finish(m.config, None, {})
    left, right = _tensors(ct.out), _tensors(awq.out)
    packed = {name.removesuffix(".weight_packed") for name in left if name.endswith(".weight_packed")}
    assert packed and any(".experts." in name for name in packed)
    expected = set()
    for name in packed:
        shape = tuple(left[name + ".weight_shape"].tolist())
        scale = left[name + ".weight_scale"]
        codes = unpack(left[name + ".weight_packed"], 4, shape).to(torch.int16) + 8
        zero = unpack(left[name + ".weight_zero_point"], 4, scale.shape, packed_dim=0).to(torch.int16) + 8
        restored_codes = _awq_codes(right[name + ".qweight"]).T
        restored_zero = _awq_codes(right[name + ".qzeros"]).T
        stored_scale = right[name + ".scales"].T.contiguous()
        assert stored_scale.dtype == scale.dtype == torch.bfloat16
        assert torch.equal(scale.view(torch.int16), stored_scale.view(torch.int16))
        assert torch.equal(codes, restored_codes) and torch.equal(zero, restored_zero)
        group = shape[1] // scale.shape[1]
        original = (
            codes.float() - zero.float().repeat_interleave(group, dim=1)
        ) * scale.float().repeat_interleave(group, dim=1)
        restored = (
            restored_codes.float() - restored_zero.float().repeat_interleave(group, dim=1)
        ) * stored_scale.float().repeat_interleave(group, dim=1)
        assert torch.equal(original.view(torch.int32), restored.view(torch.int32)), name
        expected.update(name + suffix for suffix in (".qweight", ".qzeros", ".scales"))
    parts = (".weight_packed", ".weight_shape", ".weight_scale", ".weight_zero_point")
    floats = {name for name in left if not name.endswith(parts)}
    assert set(right) == expected | floats
    for name in floats:
        assert left[name].dtype == right[name].dtype
        assert torch.equal(
            left[name].contiguous().view(torch.uint8), right[name].contiguous().view(torch.uint8)
        )
    cfg = json.loads((awq.out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "awq" and cfg["torch_dtype"] == "bfloat16"
    assert parse_exporter(None).name == "compressed_tensors"
    assert parse_exporter({"awq_bf16": {}}).to_dict() == {"awq_bf16": {}}


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("zero_point", -1, "zero points"),
        ("zero_point", 16, "zero points"),
        ("zero_point", 0.5, "zero points"),
        ("zero_point", float("nan"), "zero points"),
        ("scale", float("nan"), "finite"),
        ("scale", float("inf"), "finite"),
        ("codes", 16, "codes"),
    ],
)
def test_invalid_awq_values_refuse_before_integer_packing(field, value, match):
    """Packing may otherwise wrap bad zero points into valid-looking nibbles."""
    qp = {
        "format": "w4a16-asym-g128",
        "codes": torch.zeros(8, 128),
        "zero_point": torch.zeros(8, 1),
        "scale": torch.ones(8, 1, dtype=torch.bfloat16),
    }
    qp[field].flatten()[0] = value
    with pytest.raises(Refused, match=rf"decoder.moe.expert7:.*{match}"):
        pack_awq("decoder.moe.expert7", qp)


def test_awq_refuses_fp16_scale_conversion():
    qp = {
        "format": "w4a16-asym-g128",
        "codes": torch.zeros(8, 128, dtype=torch.int8),
        "zero_point": torch.zeros(8, 1, dtype=torch.int8),
        "scale": torch.ones(8, 1, dtype=torch.float16),
    }
    with pytest.raises(Refused, match="expert7:.*already be BF16"):
        pack_awq("expert7", qp)


def test_awq_build_replays_from_resolved_with_identical_weight_bytes(tmp_path):
    """The selected exporter survives YAML resolution and a real engine build/replay."""
    source = tmp_path / "src"
    model("qwen3_moe", dtype=torch.bfloat16).save_pretrained(source)
    save_tiny_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4"}) + "\n")
    first = BuildConfig(
        model=str(source),
        format="w4a16-asym-g128",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 4}},
        exporter="awq_bf16",
        out=str(tmp_path / "first"),
    ).build()
    resolved = yaml.safe_load((first / "resolved.yaml").read_text())
    assert resolved["exporter"] == {"awq_bf16": {}}
    second = BuildConfig.from_file(first / "resolved.yaml", [f"out={tmp_path / 'second'}"]).build()
    for file in first.glob("*.safetensors"):
        assert file.read_bytes() == (second / file.name).read_bytes()
    assert (first / "config.json").read_bytes() == (second / "config.json").read_bytes()
    serve = json.loads((first / "build.json").read_text())["serve"]
    assert serve["args"] == {"dtype": "bfloat16"} and serve["kernels"] == {}


def test_float_qwen_fusions_and_layer_zero_bank_get_literal_awq_skips(tmp_path):
    """The matched topology's float bank/QKV must reach stock AWQ's substring matcher."""
    m = model("qwen3_5_moe", dtype=torch.bfloat16)
    source = tmp_path / "src"
    m.save_pretrained(source)
    graph = describe(m, verify=False)
    graph = replace(graph, extra={**graph.extra, "float_banks": {"model.layers.0.mlp.experts"}})
    w = AWQWriter(tmp_path / "awq", source, graph, "w4a16-asym-g64", Target())
    for ctx in contexts(m, "w4a16-asym-g64", tokens=4):
        REGISTRY["rtn"]().apply_layer(ctx)
        for name in tuple(ctx.qparams):
            if name.startswith("model.layers.0.mlp.experts.") or ".self_attn." in name:
                del ctx.qparams[name]
        w.add_layer(ctx)
    w.finish(m.config, None, {})
    skips = set(
        json.loads((w.out / "config.json").read_text())["quantization_config"]["modules_to_not_convert"]
    )
    assert "model.layers.0.mlp.experts" in skips
    assert "model.layers.1.qkv_proj" in skips
    assert "model.layers.1.self_attn.qkv_proj" in skips
    assert "model.layers.0.linear_attn.in_proj_ba" in skips
    assert not any(name.startswith("re:") for name in skips)
    tensors = _tensors(w.out)
    for name in (k.removesuffix(".qweight") for k in tensors if k.endswith(".qweight")):
        assert not any(skip in name for skip in skips), name


def test_awq_plan_refuses_bare_moe_format_with_symmetric_experts(tmp_path):
    """The new bare-MoE symmetry default cannot be represented by model-wide asymmetric AWQ."""
    source = tmp_path / "src"
    model("qwen3_moe", dtype=torch.bfloat16).save_pretrained(source)
    save_tiny_tokenizer(source)
    (tmp_path / "calib.jsonl").write_text(json.dumps({"text": "w1 w2 w3 w4"}) + "\n")
    cfg = BuildConfig(
        model=str(source),
        format="w4a16",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 1, "seqlen": 4}},
        exporter="awq_bf16",
    )
    with pytest.raises(Refused, match="explicitly asymmetric"):
        cfg.plan()
