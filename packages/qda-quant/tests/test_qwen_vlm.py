"""Qwen's composite config: image embeddings, text parity and untouched vision checkpoint tensors."""

import json

import pytest
import torch
import transformers
from qda_quant import FileSource, Recipe, build
from qda_quant.calib import selfgen
from qda_quant.calib.sources import MixSource, Rows, VLMSource
from qda_quant.engine import calib
from qda_quant.engine.build_validation import _skeleton
from qda_quant.engine.runner import capture
from qda_quant.intake.loading import load_model
from qda_quant.intake.source_names import SourceNames
from qda_quant.steps.gptq import GPTQ
from qda_quant.target import Target
from safetensors.torch import load_file

from .engine_util import save_tiny_tokenizer, write_ids


@pytest.fixture
def source(tmp_path, monkeypatch):
    cfg = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 3, 3],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 128,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=124,
        vision_end_token_id=127,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    path = tmp_path / "source"
    load_model(config=cfg, images=True, dtype=torch.bfloat16).save_pretrained(path)
    save_tiny_tokenizer(path)
    # The tiny tokenizer is WordLevel; Qwen's real tokenizer class expects its BPE vocabulary.
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", transformers.PreTrainedTokenizerFast.from_pretrained
    )
    return path


def _tensors(path):
    return {k: v for f in sorted(path.glob("*.safetensors")) for k, v in load_file(f).items()}


class _Processor:
    """One image patch group, with the multimodal positions required by transformers."""

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


@pytest.mark.parametrize("residency", ["resident", "stream", "disk"])
def test_selfgen_image_mix_build_keeps_pixels_and_vision(source, tmp_path, monkeypatch, residency):
    """The real hybrid/vision forward feeds GPTQ; a pending selfgen part must survive replanning."""
    from PIL import Image

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    monkeypatch.setattr(
        VLMSource,
        "read",
        lambda *_a: Rows([{"messages": [], "images": [Image.new("RGB", (8, 8))]}] * 2, None),
    )
    generated = []

    def generate(*args, **kwargs):
        generated.append(True)
        return selfgen.selfgen(*args, **kwargs, generate=lambda *_a: [["w3 w4 w5 w6 w7 w8 w9 w10"]] * 2)

    monkeypatch.setattr(selfgen, "generate_isolated", generate)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": str(i), "prompt": "w3"}) for i in range(2)))
    mix = MixSource(
        samples=4,
        parts=[
            {
                "selfgen": {
                    "prompts": str(prompts),
                    "seqlen": 6,
                    "sampling": {"temperature": 0},
                    "max_tokens": 16,
                },
                "weight": 1,
            },
            {"vlm": {"seqlen": 8}, "weight": 1},
        ],
    )
    recipe = Recipe(steps=[GPTQ(impl="torch")], calib=mix)
    out = build(
        source,
        recipe,
        format="w4a16-sym-g128",
        devices=["cpu"],
        out_dir=tmp_path / "out",
        residency=residency,
        log=lambda _: None,
    )
    assert generated == [True]
    record = json.loads((out / "build.json").read_text())
    assert [s["kind"] for s in record["stages"]] == ["generate", "layers"]
    assert mix.pending(source) is None
    ids, inputs = calib.load_with_inputs(mix, source)
    assert sum(bool(p) for p in inputs) == 2
    model = load_model(source, images=True, dtype=torch.bfloat16).eval()
    from qda_quant.arch import describe

    graph = describe(model, verify=False)
    slots = capture(model, graph, calib.batches(ids), torch.device("cpu"), inputs=inputs)
    # Changing only the pixels changes the hidden states consumed by the language layers.
    changed = [{**p, "pixel_values": -p["pixel_values"]} if p else {} for p in inputs]
    other = capture(model, graph, calib.batches(ids), torch.device("cpu"), inputs=changed)
    assert all(torch.equal(a.h, b.h) for a, b, p in zip(slots, other, inputs, strict=True) if not p)
    assert all(not torch.equal(a.h, b.h) for a, b, p in zip(slots, other, inputs, strict=True) if p)
    original, written = _tensors(source), _tensors(out)
    vision = {k for k in original if k.startswith("model.visual.")}
    assert vision
    assert all(written[k].dtype == torch.bfloat16 and torch.equal(original[k], written[k]) for k in vision)
    assert not any(k.startswith("model.visual.") and "weight_packed" in k for k in written)
    assert "model.language_model.layers.1.self_attn.q_proj.weight_packed" in written
    assert json.loads((out / "config.json").read_text())["architectures"] == [
        "Qwen3_5ForConditionalGeneration"
    ]


def test_text_build_has_identical_tensors_with_either_model_class(source, tmp_path, monkeypatch):
    """Both in-memory language prefixes map to the same source; full-model text math must agree too."""
    from qda_quant.arch import describe
    from qda_quant.intake import loading

    tensors = _tensors(source)
    names = SourceNames(source)
    text_graph = describe(_skeleton(str(source), torch.bfloat16, Target()), verify=False)
    image_graph = describe(_skeleton(str(source), torch.bfloat16, Target(), images=True), verify=False)
    text_modules = [names.module(m) for lg in text_graph.layers for g in lg.groups for m in g.modules]
    image_modules = [names.module(m) for lg in image_graph.layers for g in lg.groups for m in g.modules]
    assert text_modules == image_modules and all(text_modules)
    data = tmp_path / "text.jsonl"
    write_ids(data, [[3, 4, 5, 6] * 2] * 2, calib.fingerprint(source))
    recipe = Recipe(steps=[GPTQ(impl="torch")], calib=FileSource(path=str(data), samples=2, seqlen=8))
    text = build(
        source,
        recipe,
        format="w4a16-sym-g128",
        devices=["cpu"],
        out_dir=tmp_path / "text",
        log=lambda _: None,
    )
    original_loader = loading.load_model
    monkeypatch.setattr(loading, "load_model", lambda *a, **k: original_loader(*a, **{**k, "images": True}))
    full = build(
        source,
        recipe,
        format="w4a16-sym-g128",
        devices=["cpu"],
        out_dir=tmp_path / "full",
        log=lambda _: None,
    )
    a, b = _tensors(text), _tensors(full)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(torch.equal(a[k], v) for k, v in tensors.items() if k.startswith("model.visual."))
