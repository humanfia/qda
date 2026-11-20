"""An expert format override on a wrapped-name MoE checkpoint: every engine must select the override.

Qwen3.5/3.6-MoE checkpoints store the text model under ``model.language_model.`` and the routed experts as
fused 3-D banks. QDA holds the image-text class for an image-calibrated build, writes the experts per expert
under the checkpoint's wrapped names, and must claim them in a config group that also covers the names the
engines rename the root to (vLLM 0.33: ``language_model.model.``; SGLang 0.5.23 and transformers' text class:
``model.``). A group written from one name form alone let the experts fall back to the body's scheme.
"""

import json

import pytest
import torch
import transformers
from qda_quant import Recipe, build
from qda_quant.calib.sources import Rows, VLMSource
from qda_quant.output.export import _ignore_matches
from qda_quant.steps.gptq import GPTQ

from ..engine_util import save_tiny_tokenizer

BODY, OVERRIDE = "w4a16-sym-g32", "w8a16-sym-g32"
#: matches the bank under the graph's name, wrapped or not (the pattern a user writes for either model)
SELECTOR = r"re:^model\.(?:language_model\.)?layers\.\d+\.mlp\.experts$"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class _Processor:
    """One 2x2 patch group of a 4x4 image, with the multimodal positions transformers requires."""

    def apply_chat_template(self, messages, tokenize=False):
        return "image"

    def __call__(self, text, images, return_tensors):
        ids = torch.tensor([[3, 124, 125, 127, 4, 5]])
        return {
            "input_ids": ids,
            "pixel_values": torch.arange(384).reshape(4, 96).float() / 384,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": (ids == 125).long(),
        }


def _randomize(model):
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02, generator=gen)


@pytest.fixture
def wrapped(tmp_path, monkeypatch, cpu_build):  # cpu_build: it builds on the CPU while a GPU run holds cards
    """A tiny wrapped Qwen3.5-MoE checkpoint (fused expert banks, as the release checkpoints store them).

    Image calibration keeps the image-text class, so the graph names the bank under the checkpoint's wrapped
    prefix and the export writes per-expert tensors under it.
    """
    from PIL import Image
    from safetensors.torch import save_file

    cfg = transformers.Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 128, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "max_position_embeddings": 64, "layer_types": ["linear_attention", "full_attention"],
            "full_attention_interval": 2, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 128,
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 128,
            "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 16,
        },
        image_token_id=125, video_token_id=126, vision_start_token_id=124, vision_end_token_id=127,
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    model = transformers.Qwen3_5MoeForConditionalGeneration(cfg)
    _randomize(model)
    source = tmp_path / "source"
    model.eval().save_pretrained(source)
    # save_pretrained expands the fused banks; write the checkpoint layout the release models carry.
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(source / "model.safetensors"),
        metadata={"format": "pt"},
    )
    save_tiny_tokenizer(source)
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    monkeypatch.setattr(
        VLMSource,
        "read",
        lambda *_a: Rows([{"messages": [], "images": [Image.new("RGB", (4, 4))]}] * 2, None),
    )
    return build(
        str(source),
        Recipe(steps=[GPTQ(impl="torch")], calib=VLMSource(samples=2, seqlen=8)),
        format=BODY,
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        format_overrides=[(SELECTOR, OVERRIDE)],
        log=lambda _: None,
    )


def _targets(out):
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    return [t for group in groups.values() for t in group["targets"] if t != "Linear"]


def _stored_experts(out):
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    return {k.removesuffix(".weight_packed") for k in index if k.endswith(".weight_packed")}


def test_wrapped_expert_override_claims_the_bank_under_every_engine_name(wrapped):
    """The group covers the checkpoint's wrapped name, the text class's root and vLLM's renamed root."""
    targets = _targets(wrapped)
    stored = _stored_experts(wrapped)
    for layer in (0, 1):
        bank = f"model.language_model.layers.{layer}.mlp.experts"
        assert f"{bank}.0.down_proj" in stored  # written under the checkpoint's own name
        for served in (
            bank,  # the stored checkpoint
            f"model.layers.{layer}.mlp.experts",  # transformers' text class, SGLang 0.5.23
            f"language_model.model.layers.{layer}.mlp.experts",  # vLLM 0.33
        ):
            for proj in PROJECTIONS:
                assert any(_ignore_matches(f"{served}.0.{proj}", t) for t in targets), (served, proj, targets)
    assert not any(
        _ignore_matches("language_model.model.layers.2.mlp.experts.0.down_proj", t) for t in targets
    )


def test_wrapped_expert_override_leaves_the_body_and_other_linears_alone(wrapped):
    """Only the routed experts get the override format; the body keeps its own scheme."""
    groups = json.loads((wrapped / "config.json").read_text())["quantization_config"]["config_groups"]
    body = [g for g in groups.values() if g["targets"] == ["Linear"]]
    assert [g["weights"]["num_bits"] for g in body] == [4]
    overrides = [g for g in groups.values() if g["targets"] != ["Linear"]]
    assert [g["weights"]["num_bits"] for g in overrides] == [8]


@pytest.mark.gpu
def test_wrapped_expert_override_loads_in_stock_vllm(wrapped):
    """Stock vLLM loads the checkpoint's 8-bit experts with the override scheme, not the body's 4-bit.

    On main the group missed vLLM's ``language_model.model.`` root, so the routed-expert loader read the 8-bit
    weights through the body's 4-bit packing and the engine refused to start.
    """
    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration as VllmModel

    class RoutedExperts(torch.nn.Module):
        pass

    qc = json.loads((wrapped / "config.json").read_text())["quantization_config"]
    config = CompressedTensorsConfig.from_config(qc)
    config.apply_vllm_mapper(VllmModel.hf_to_vllm_mapper)
    config._add_fused_moe_to_target_scheme_map()
    for layer in (0, 1):
        bank = f"language_model.model.layers.{layer}.mlp.experts"
        bits = [
            config.get_scheme_dict(RoutedExperts(), f"{bank}.0.{p}")["weights"].num_bits for p in PROJECTIONS
        ]
        assert bits == [8, 8, 8], (bank, bits)

    llm = LLM(
        model=str(wrapped),
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=64,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    out = llm.generate(
        [{"prompt_token_ids": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}],
        SamplingParams(temperature=0, max_tokens=8),
    )
    assert len(out[0].outputs[0].token_ids) == 8
