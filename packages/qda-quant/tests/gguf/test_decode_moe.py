"""Fused expert-bank decode keeps the source order and refuses incomplete or mismatched layouts."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from qda_quant.gguf.decode import decode
from safetensors.torch import load_file, save_file


def test_fused_experts_round_trip(tmp_path):
    gguf = pytest.importorskip("gguf")
    source, path, snap = _tiny_moe(tmp_path, gguf)
    out = tmp_path / "out"
    manifest = decode(path, out, snap)
    weights = load_file(str(out / "model-00001-of-00001.safetensors"))
    assert set(weights) == set(source)
    for name, want in source.items():
        assert weights[name].shape == want.shape, name
        assert torch.equal(weights[name].view(torch.int16), want.to(torch.bfloat16).view(torch.int16)), name
    assert manifest["gguf"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["mtp"] == []


@pytest.mark.parametrize("fault", ["missing", "shape"])
def test_source_layout_mismatch_refuses_before_writing(tmp_path, fault):
    gguf = pytest.importorskip("gguf")
    source, path, snap = _tiny_moe(tmp_path, gguf)
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    if fault == "missing":
        del source[name]
    else:
        source[name] = source[name][:, :-1].contiguous()
    save_file(source, str(snap / "model.safetensors"))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="snapshot|names differ"):
        decode(path, out, snap)
    assert not out.exists()


def _tiny_moe(tmp_path: Path, gguf) -> tuple[dict[str, torch.Tensor], Path, Path]:  # noqa: ANN001
    """A ``qwen35moe`` GGUF (one block, two experts) and the snapshot whose index names what it decodes to.

    Gate and up GGUF banks become the source fused bank, in gate/up order for each expert.
    """
    hidden, inter, experts = 16, 8, 2

    def vals(n: int) -> torch.Tensor:
        return (torch.arange(n) % 16 - 8 + 0.25).float()  # quarters of small integers: exact in bf16

    def bank(size: tuple[int, int]) -> torch.Tensor:
        return torch.stack([vals(size[0] * size[1]).reshape(size) + i for i in range(experts)])

    prefix = "model.language_model.layers.0."
    gate, up, down = bank((inter, hidden)), bank((inter, hidden)) + 8, bank((hidden, inter))
    hf = {
        "model.language_model.embed_tokens.weight": vals(32 * hidden).reshape(32, hidden),
        "model.language_model.norm.weight": vals(hidden),
        "lm_head.weight": vals(32 * hidden).reshape(32, hidden),
        prefix + "input_layernorm.weight": vals(hidden),
        prefix + "mlp.gate.weight": vals(experts * hidden).reshape(experts, hidden),
        prefix + "mlp.shared_expert_gate.weight": vals(hidden).reshape(1, hidden),
        prefix + "mlp.shared_expert.gate_proj.weight": vals(inter * hidden).reshape(inter, hidden),
        prefix + "mlp.shared_expert.up_proj.weight": vals(inter * hidden).reshape(inter, hidden),
        prefix + "mlp.shared_expert.down_proj.weight": vals(hidden * inter).reshape(hidden, inter),
    }
    hf[prefix + "mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
    hf[prefix + "mlp.experts.down_proj"] = down
    tensors = {
        "token_embd.weight": hf["model.language_model.embed_tokens.weight"],
        "output_norm.weight": hf["model.language_model.norm.weight"] + 1,  # RMS gains are stored +1
        "output.weight": hf["lm_head.weight"],
        "blk.0.attn_norm.weight": hf[prefix + "input_layernorm.weight"] + 1,
        "blk.0.ffn_gate_inp.weight": hf[prefix + "mlp.gate.weight"],
        # the shared-expert gate is one row; ggml drops that axis, so the file itself stores it 1-D
        "blk.0.ffn_gate_inp_shexp.weight": hf[prefix + "mlp.shared_expert_gate.weight"].reshape(-1),
        "blk.0.ffn_gate_shexp.weight": hf[prefix + "mlp.shared_expert.gate_proj.weight"],
        "blk.0.ffn_up_shexp.weight": hf[prefix + "mlp.shared_expert.up_proj.weight"],
        "blk.0.ffn_down_shexp.weight": hf[prefix + "mlp.shared_expert.down_proj.weight"],
        "blk.0.ffn_up_exps.weight": up,
        "blk.0.ffn_gate_exps.weight": gate,
        "blk.0.ffn_down_exps.weight": down,
    }
    w = gguf.GGUFWriter(str(tmp_path / "moe.gguf"), "qwen35moe")
    w.add_block_count(1)
    w.add_expert_count(experts)
    w.add_uint32("qwen35moe.ssm.group_count", 1)
    w.add_uint32("qwen35moe.ssm.time_step_rank", 1)
    w.add_uint32("qwen35moe.ssm.inner_size", 4)
    w.add_uint32("qwen35moe.ssm.state_size", 4)
    for name, t in tensors.items():
        w.add_tensor(
            name, t.numpy().astype("<f4"), raw_shape=tuple(t.shape), raw_dtype=gguf.GGMLQuantizationType.F32
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    snap = tmp_path / "moe-snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "text_config": {}}) + "\n")
    (snap / "tokenizer.json").write_text('{"model": {}}\n')
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(hf)}}) + "\n"
    )
    save_file(hf, str(snap / "model.safetensors"))
    return hf, tmp_path / "moe.gguf", snap
