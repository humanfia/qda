"""Qwen2.5-VL float MLP overrides retain the engine's scheme names."""

import json
from dataclasses import replace

import pytest
import torch
from qda_quant.arch import describe
from qda_quant.output.export import Writer
from qda_quant.step import REGISTRY
from qda_quant.target import Target
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .ctx import contexts
from .steps.test_rotate_stream import _tiny_qwen2_5_vl


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("float_layers", [(), (0,), (0, 1)])
def test_float_mlp_aliases_preserve_layer_selection_and_checkpoint_values(tmp_path, native, float_layers):
    """Each ignored shard gets a mapper-safe suffix; packed layers and source names stay unchanged."""
    model = _tiny_qwen2_5_vl().to(torch.bfloat16)
    src = tmp_path / "src"
    model.save_pretrained(src, save_original_format=False)
    if native:
        values = load_file(src / "model.safetensors")
        save_file(
            {
                name.replace("model.language_model.", "model.").replace("model.visual.", "visual."): value
                for name, value in values.items()
            },
            src / "model.safetensors",
            metadata={"format": "pt"},
        )
    writer = Writer(tmp_path / "out", src, describe(model, verify=False), "w4a16-sym-g128", Target())
    for layer, ctx in enumerate(contexts(model, "w4a16-sym-g128", tokens=8)):
        if layer in float_layers:
            ctx.graph = replace(
                ctx.graph,
                groups=tuple(
                    group for group in ctx.graph.groups if not group.modules[0].endswith("mlp.gate_proj")
                ),
            )
        REGISTRY["rtn"]().apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(model.config, None, {})
    config = json.loads((out / "config.json").read_text())
    aliases = {entry for entry in config["quantization_config"]["ignore"] if entry.startswith("layers.")}
    assert aliases == {
        f"layers.{layer}.mlp.{projection}"
        for layer in float_layers
        for projection in ("gate_proj", "up_proj")
    }
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model" if native else "model.language_model"
    with safe_open(src / "model.safetensors", "pt") as source:
        for layer in range(2):
            for projection in ("gate_proj", "up_proj"):
                stem = f"{prefix}.layers.{layer}.mlp.{projection}"
                if layer in float_layers:
                    key = f"{stem}.weight"
                    with safe_open(out / index[key], "pt") as packed:
                        assert torch.equal(packed.get_tensor(key), source.get_tensor(key))
                    assert f"{stem}.weight_packed" not in index
                else:
                    assert f"{stem}.weight_packed" in index
                    assert f"{stem}.weight" not in index
