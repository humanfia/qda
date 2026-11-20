"""GPTQ planning, calibration and packed exports on tiny models from supported families."""

import json

import pytest
import torch
import transformers
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from qda_quant import BuildConfig
from qda_quant.output.export import Writer
from safetensors.torch import load_file

from .engine_util import save_tiny_tokenizer, write_ids, writes

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")]


def _config(family):
    settings = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if family in {"llama", "mistral", "gemma2", "glm", "gemma3_text", "qwen3_5_text", "qwen3_5_moe_text"}:
        settings["head_dim"] = 32
    if family in {"mistral", "gemma2", "gemma3_text"}:
        settings["sliding_window"] = 64
    if family == "phi3":
        settings["original_max_position_embeddings"] = 128
    if family in {"qwen3_5_text", "qwen3_5_moe_text"}:
        settings.update(
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
        )
    if family == "qwen3_5_moe_text":
        settings.pop("intermediate_size")
        settings.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    config = transformers.AutoConfig.for_model(family, **settings)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


@pytest.mark.parametrize(
    "family",
    [
        "llama",
        "qwen2",
        "mistral",
        "gemma2",
        "phi3",
        "olmo2",
        "granite",
        "smollm3",
        "glm",
        "gemma3_text",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    ],
)
def test_family_gptq_build_and_pack_roundtrip(family, tmp_path):
    torch.manual_seed(0)
    source = tmp_path / "source"
    model = transformers.AutoModelForCausalLM.from_config(_config(family), dtype=torch.bfloat16).eval()
    model.save_pretrained(source)
    fingerprint = save_tiny_tokenizer(source)
    calibration = tmp_path / "calibration.jsonl"
    generator = torch.Generator().manual_seed(1)
    write_ids(
        calibration,
        [torch.randint(3, 128, (32,), generator=generator).tolist() for _ in range(4)],
        fingerprint,
    )
    config = BuildConfig.from_dict(
        {
            "model": str(source),
            "format": "w4a16-sym-g128",
            "devices": [0],
            "steps": [{"gptq": {}}],
            "calib": {"file": {"path": str(calibration), "samples": 4, "seqlen": 32}},
            "out": str(tmp_path / "checkpoint"),
        }
    )
    plan = config.plan()
    assert len(plan.graph.layers) == 2
    assert plan.graph.model_type == family
    assert plan.plan_dense.kernel == ("HummingLinearKernel" if plan.target.arch == "sm90" else "MarlinLinearKernel")
    propagated = {}

    class CaptureWriter(Writer):
        def add_layer(self, context):
            prefix = context.graph.module + "."
            for name in context.qparams:
                if name.startswith(prefix):
                    relative = name[len(prefix) :]
                    try:
                        weight = context.layer.get_parameter(relative)
                    except AttributeError:
                        weight = context.layer.get_submodule(relative).weight
                    weight = weight.detach().cpu().clone()
                    if weight.ndim == 2:
                        propagated[name] = weight
                    else:
                        root, _, parameter = name.rpartition(".")
                        for expert, values in enumerate(weight):
                            if parameter == "down_proj":
                                propagated[f"{root}.{expert}.down_proj"] = values
                            else:
                                gate, up = values.chunk(2, dim=0)
                                propagated[f"{root}.{expert}.gate_proj"] = gate
                                propagated[f"{root}.{expert}.up_proj"] = up
            super().add_layer(context)

    checkpoint = config.build(exporter=writes(CaptureWriter))
    info = json.loads((checkpoint / "build.json").read_text())
    assert info["card_count"] == 1
    assert info["timings"]["peak_mem_gib"]["cuda:0"] > 0
    assert (checkpoint / "resolved.yaml").is_file()
    tensors = {
        name: tensor
        for shard in checkpoint.glob("*.safetensors")
        for name, tensor in load_file(shard).items()
    }
    stored = {name.removesuffix(".weight_packed") for name in tensors if name.endswith(".weight_packed")}
    assert stored == propagated.keys()
    assert stored
    assert {name for layer in plan.graph.layers for group in layer.groups for name in group.modules} <= stored
    for name, expected in propagated.items():
        shape = torch.Size(tensors[name + ".weight_shape"].tolist())
        codes = unpack_from_int32(tensors[name + ".weight_packed"], 4, shape)
        scale = tensors[name + ".weight_scale"]
        assert scale.dtype == torch.bfloat16
        assert name + ".weight_zero_point" not in tensors
        restored = (codes.float() * scale.float().repeat_interleave(128, -1)).to(torch.bfloat16)
        assert torch.equal(restored, expected), name
        assert torch.isfinite(restored).all(), name
