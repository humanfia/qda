"""intake.test_serve_humming.expected_linears: quantized Linear modules vLLM serves, from a checkpoint's packed weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


def _module():
    spec = importlib.util.spec_from_file_location(
        "serve_t", Path(__file__).parent / "intake" / "test_serve_humming.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fused_projections_count_once_and_experts_are_excluded(tmp_path):
    tensors = {}
    for layer in range(2):
        p = f"model.layers.{layer}"
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            tensors[f"{p}.{proj}.weight_packed"] = torch.zeros(1, dtype=torch.int32)
            tensors[f"{p}.{proj}.weight_scale"] = torch.zeros(1, dtype=torch.int32)
        tensors[f"{p}.mlp.experts.0.gate_proj.weight_packed"] = torch.zeros(1, dtype=torch.int32)
        tensors[f"{p}.mlp.experts.0.down_proj.weight_packed"] = torch.zeros(1, dtype=torch.int32)
    tensors["model.layers.0.mlp.gate_proj.weight_packed"] = torch.zeros(1, dtype=torch.int32)
    tensors["model.layers.0.mlp.up_proj.weight_packed"] = torch.zeros(1, dtype=torch.int32)
    tensors["model.layers.0.mlp.down_proj.weight_packed"] = torch.zeros(1, dtype=torch.int32)
    tensors["lm_head.weight"] = torch.zeros(1, dtype=torch.int32)
    save_file(dict(list(tensors.items())[:9]), str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(dict(list(tensors.items())[9:]), str(tmp_path / "model-00002-of-00002.safetensors"))
    # per layer: qkv_proj + o_proj; layer 0 also gate_up_proj + down_proj; experts and lm_head not counted
    assert _module().expected_linears(str(tmp_path)) == 2 * 2 + 2
