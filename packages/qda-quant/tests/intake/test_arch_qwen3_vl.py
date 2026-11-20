"""Qwen3-VL-MoE: fused expert banks leave the exporter as per-expert serving tensors.

The architecture's checkpoints store one fused bank per layer and transformers converts it with a
``Transpose(check_dims=True)``. That is a physical layout hint, not an expert numbering, so the adapter's
serving layout must still name the pieces: stock vLLM 0.33 and SGLang 0.5.23 both load per-expert
compressed-tensors names, while a canonical packed bank under the fused name fails their weight copy.
"""

import json

import pytest
import torch
from qda_quant import FileSource, Recipe, build
from qda_quant.intake.source_names import SourceNames, model_rules
from qda_quant.steps.rtn import RTN
from safetensors import safe_open

from ..engine_util import save_tiny_tokenizer, write_ids

QDA_TRANSFORMERS = pytest.importorskip("transformers")

BANK = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DOWN = "model.language_model.layers.0.mlp.experts.down_proj"


def _tiny_source(tmp_path):
    """A one-layer Qwen3-VL-MoE checkpoint, g128-sized and with distinct expert values."""
    text = QDA_TRANSFORMERS.Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    vision = QDA_TRANSFORMERS.Qwen3VLMoeVisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=64,
        out_hidden_size=128,
        num_heads=2,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    cfg = QDA_TRANSFORMERS.Qwen3VLMoeConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = QDA_TRANSFORMERS.Qwen3VLMoeForConditionalGeneration(cfg).eval()
    with torch.no_grad():  # per (expert, row) values, so a reordered or transposed bank cannot match
        for name, p in model.named_parameters():
            if p.is_floating_point() and "experts" in name:
                p.copy_(torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape) * 1e-4)
            elif p.is_floating_point():
                p.normal_(0, 0.02)
    src = tmp_path / "src"
    model.to(torch.bfloat16).save_pretrained(src)
    save_tiny_tokenizer(src)
    return src, model


def _build(src, tmp_path):
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], save_tiny_tokenizer(src))
    return build(
        str(src),
        Recipe(steps=[RTN()], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-asym-g128",  # this test unpacks zero points: name the symmetry explicitly
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        log=lambda s: None,
    )


def test_fused_bank_is_written_as_per_expert_serving_tensors(tmp_path):
    """Every expert tensor is one the stock loaders map, with the bank's own halves and values."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    src, model = _tiny_source(tmp_path)
    out = _build(src, tmp_path)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert BANK + "_packed" not in index and DOWN + "_packed" not in index
    expected = {
        "gate_up_proj": model.model.language_model.layers[0].mlp.experts.gate_up_proj.detach(),
        "down_proj": model.model.language_model.layers[0].mlp.experts.down_proj.detach(),
    }
    for bank, projections in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
        tensor = expected[bank]
        rows = tensor.shape[1] // len(projections)
        for expert in range(tensor.shape[0]):
            for position, leaf in enumerate(projections):
                prefix = f"model.language_model.layers.0.mlp.experts.{expert}.{leaf}.weight_"
                assert prefix + "packed" in index, prefix
                with safe_open(out / index[prefix + "packed"], framework="pt") as shard:
                    shape = torch.Size(shard.get_tensor(prefix + "shape").tolist())
                    assert shape == (rows, tensor.shape[2])
                    codes = unpack_from_int32(shard.get_tensor(prefix + "packed"), 4, shape).float()
                    scale = shard.get_tensor(prefix + "scale").float()
                    zero = unpack_from_int32(
                        shard.get_tensor(prefix + "zero_point"),
                        4,
                        torch.Size((rows, scale.shape[-1])),
                        packed_dim=0,
                    ).float()
                group = shape[-1] // scale.shape[-1]
                dequantized = (codes - zero.repeat_interleave(group, dim=-1)) * scale.repeat_interleave(
                    group, dim=-1
                )
                # 4-bit rounding, so compare against the source bank at the scale of one code: a swapped
                # expert, projection or row order is off by many codes, not by one.
                wanted = tensor[expert, position * rows : (position + 1) * rows].float()
                error = (dequantized - wanted).abs().max()
                assert error <= scale.float().max(), (bank, expert, leaf, float(error))


def test_loading_keeps_the_architectures_transpose_converter(tmp_path):
    """The transpose hint stays in the load rules: only the export layout changes."""
    src, model = _tiny_source(tmp_path)
    converter = next(
        rule
        for rule in model_rules(model)
        if any(p.endswith("mlp.experts.gate_up_proj") for p in getattr(rule, "source_patterns", ()))
    )
    assert [type(op).__name__ for op in converter.operations] == ["Transpose"]
    names = SourceNames(src)
    assert not names.converter_owns(BANK) and not names.converter_owns(DOWN)
    assert names.stored(BANK, torch.Size((4, 256, 128))) == [(BANK, ())]
