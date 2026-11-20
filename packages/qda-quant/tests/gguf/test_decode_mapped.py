"""Decoding the gguf-py-mapped architectures (llama, qwen3moe, gemma3): each tiny file is written the way the pinned
converter writes it (``LlamaModel.permute`` on Q/K, stacked expert banks, ``+1`` gemma norms), and the decode must give back the snapshot's tensors bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qda_quant.gguf.decode import _unpermute, decode

gguf = pytest.importorskip("gguf")
from safetensors.torch import load_file, save_file  # noqa: E402

HIDDEN = 64


def _vals(*shape: int, seed: int = 0) -> torch.Tensor:
    """Quarters of small integers, exact in bf16, different per `seed` so a swapped tensor is caught."""
    n = int(np.prod(shape))
    return ((torch.arange(n) * (seed + 3) + seed) % 16 - 8 + 0.25).float().reshape(shape)


def _permute(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """The pinned converter's ``LlamaModel.permute``, copied as the forward the decoder must undo."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def _write(path: Path, arch: str, tensors: dict[str, torch.Tensor], kv: dict[str, int]) -> Path:
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    for key, value in kv.items():
        w.add_uint32(f"{arch}.{key}", value)
    for name, t in tensors.items():
        w.add_tensor(name, t.numpy().astype("<f4"), raw_shape=tuple(t.shape))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _snapshot(tmp_path: Path, config: dict, hf: dict[str, torch.Tensor]) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(config) + "\n")
    (snap / "tokenizer.json").write_text('{"model": {}}\n')
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()}, str(snap / "model.safetensors"))
    return snap


def _decoded(tmp_path: Path, gguf_path: Path, snap: Path) -> tuple[dict[str, torch.Tensor], dict]:
    manifest = decode(gguf_path, tmp_path / "out", snap)
    index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    for part in set(index.values()):
        weights.update(load_file(str(tmp_path / "out" / part)))
    return weights, manifest


def test_the_unpermute_inverts_the_converters_permute() -> None:
    for heads in (1, 2, 4):
        w = _vals(4 * 8, 3)
        assert torch.equal(_unpermute(_permute(w, heads), heads), w)
        assert heads == 1 or not torch.equal(_permute(w, heads), w)


def test_a_llama_file_undoes_the_qk_permute(tmp_path: Path) -> None:
    """Q is permuted with the head count, K with the KV head count; everything else is renamed only."""
    n_head, n_kv, head_dim = 4, 2, 16
    hf = {
        "model.embed_tokens.weight": _vals(32, HIDDEN, seed=1),
        "model.norm.weight": _vals(HIDDEN, seed=2),
        "lm_head.weight": _vals(32, HIDDEN, seed=3),
        "model.layers.0.input_layernorm.weight": _vals(HIDDEN, seed=4),
        "model.layers.0.self_attn.q_proj.weight": _vals(n_head * head_dim, HIDDEN, seed=5),
        "model.layers.0.self_attn.k_proj.weight": _vals(n_kv * head_dim, HIDDEN, seed=6),
        "model.layers.0.self_attn.v_proj.weight": _vals(n_kv * head_dim, HIDDEN, seed=7),
    }
    tensors = {
        "token_embd.weight": hf["model.embed_tokens.weight"],
        "output_norm.weight": hf["model.norm.weight"],
        "output.weight": hf["lm_head.weight"],
        "blk.0.attn_norm.weight": hf["model.layers.0.input_layernorm.weight"],
        "blk.0.attn_q.weight": _permute(hf["model.layers.0.self_attn.q_proj.weight"], n_head),
        "blk.0.attn_k.weight": _permute(hf["model.layers.0.self_attn.k_proj.weight"], n_kv),
        "blk.0.attn_v.weight": hf["model.layers.0.self_attn.v_proj.weight"],
        "rope_freqs.weight": _vals(head_dim // 2, seed=8),  # llama3 rope scaling, computed from config.json
    }
    path = _write(
        tmp_path / "llama.gguf",
        "llama",
        tensors,
        {"attention.head_count": n_head, "attention.head_count_kv": n_kv},
    )
    snap = _snapshot(tmp_path, {"model_type": "llama"}, hf)
    weights, _ = _decoded(tmp_path, path, snap)
    assert set(weights) == set(hf)
    for name, want in hf.items():
        assert torch.equal(weights[name], want.to(torch.bfloat16)), name


def test_a_qwen3moe_file_splits_each_bank_into_the_snapshots_experts(tmp_path: Path) -> None:
    """Banks are ``torch.stack`` of the experts on dim 0; each snapshot expert is read back from its row."""
    experts, inter = 4, 32
    hf = {
        "model.embed_tokens.weight": _vals(32, HIDDEN, seed=1),
        "model.norm.weight": _vals(HIDDEN, seed=2),
        "lm_head.weight": _vals(32, HIDDEN, seed=3),
        "model.layers.0.mlp.gate.weight": _vals(experts, HIDDEN, seed=4),
    }
    banks: dict[str, list[torch.Tensor]] = {"gate": [], "up": [], "down": []}
    for e in range(experts):
        for i, w in enumerate(("gate", "up", "down")):
            t = _vals(*((inter, HIDDEN) if w != "down" else (HIDDEN, inter)), seed=10 + 3 * e + i)
            hf[f"model.layers.0.mlp.experts.{e}.{w}_proj.weight"] = t
            banks[w].append(t)
    tensors = {
        "token_embd.weight": hf["model.embed_tokens.weight"],
        "output_norm.weight": hf["model.norm.weight"],
        "output.weight": hf["lm_head.weight"],
        "blk.0.ffn_gate_inp.weight": hf["model.layers.0.mlp.gate.weight"],
        **{f"blk.0.ffn_{w}_exps.weight": torch.stack(ts, dim=0) for w, ts in banks.items()},
    }
    path = _write(tmp_path / "moe.gguf", "qwen3moe", tensors, {"expert_count": experts})
    snap = _snapshot(tmp_path, {"model_type": "qwen3_moe"}, hf)
    weights, _ = _decoded(tmp_path, path, snap)
    assert set(weights) == set(hf)
    for name, want in hf.items():
        assert torch.equal(weights[name], want.to(torch.bfloat16)), name


def _gemma3(tmp_path: Path, cut: bool = False) -> tuple[dict[str, torch.Tensor], Path, Path]:
    """A multimodal gemma3 snapshot (``language_model.`` names, no lm_head, a visual tower) and its file; `cut`
    writes the embedding cut to fewer rows, as the converter does when the tokenizer's vocab is smaller."""
    lm = "language_model.model."
    hf = {
        f"{lm}embed_tokens.weight": _vals(40, HIDDEN, seed=1),
        f"{lm}norm.weight": _vals(HIDDEN, seed=2),
        f"{lm}layers.0.input_layernorm.weight": _vals(HIDDEN, seed=3),
        f"{lm}layers.0.post_feedforward_layernorm.weight": _vals(HIDDEN, seed=4),
        f"{lm}layers.0.self_attn.q_norm.weight": _vals(16, seed=5),
        f"{lm}layers.0.self_attn.q_proj.weight": _vals(64, HIDDEN, seed=6),
    }
    tower = {"vision_tower.vision_model.post_layernorm.weight": _vals(8, seed=7)}
    tensors = {
        "token_embd.weight": hf[f"{lm}embed_tokens.weight"][: 32 if cut else 40],
        "output_norm.weight": hf[f"{lm}norm.weight"] + 1,
        "blk.0.attn_norm.weight": hf[f"{lm}layers.0.input_layernorm.weight"] + 1,
        "blk.0.post_ffw_norm.weight": hf[f"{lm}layers.0.post_feedforward_layernorm.weight"] + 1,
        "blk.0.attn_q_norm.weight": hf[f"{lm}layers.0.self_attn.q_norm.weight"] + 1,
        "blk.0.attn_q.weight": hf[f"{lm}layers.0.self_attn.q_proj.weight"],  # gemma3 is not permuted
    }
    path = _write(tmp_path / "gemma3.gguf", "gemma3", tensors, {"attention.head_count": 4})
    config = {"model_type": "gemma3", "text_config": {}, "vision_config": {}}
    return hf, path, _snapshot(tmp_path, config, {**hf, **tower})


def test_a_gemma3_file_drops_the_plus_one_and_skips_the_tower(tmp_path: Path) -> None:
    hf, path, snap = _gemma3(tmp_path)
    weights, manifest = _decoded(tmp_path, path, snap)
    assert set(weights) == set(hf)
    for name, want in hf.items():
        assert torch.equal(weights[name], want.to(torch.bfloat16)), name
    assert "language_model_only" in manifest["config"]["note"] and "MTP" not in manifest["config"]["note"]


def test_a_snapshot_lm_head_without_an_output_tensor_reads_the_embedding(tmp_path: Path) -> None:
    """llama.cpp serves a file with no ``output.weight`` from ``token_embd``; the decode writes the same head."""
    hf = {
        "model.embed_tokens.weight": _vals(32, HIDDEN, seed=1),
        "model.norm.weight": _vals(HIDDEN, seed=2),
    }
    path = _write(
        tmp_path / "tied.gguf",
        "qwen3",
        {"token_embd.weight": hf["model.embed_tokens.weight"], "output_norm.weight": hf["model.norm.weight"]},
        {},
    )
    snap = _snapshot(
        tmp_path, {"model_type": "qwen3"}, {**hf, "lm_head.weight": hf["model.embed_tokens.weight"]}
    )
    weights, _ = _decoded(tmp_path, path, snap)
    assert torch.equal(weights["lm_head.weight"], weights["model.embed_tokens.weight"])


def test_a_snapshot_language_tensor_the_file_lacks_is_refused(tmp_path: Path) -> None:
    """A language tensor with no GGUF counterpart is refused, not left out, even in a multimodal snapshot."""
    lm = "language_model.model."
    hf = {f"{lm}norm.weight": _vals(HIDDEN), f"{lm}layers.0.input_layernorm.weight": _vals(HIDDEN)}
    path = _write(tmp_path / "g.gguf", "gemma3", {"output_norm.weight": hf[f"{lm}norm.weight"] + 1}, {})
    snap = _snapshot(tmp_path, {"model_type": "gemma3", "vision_config": {}}, hf)
    with pytest.raises(ValueError, match=r"input_layernorm\.weight: no tensor blk\.0\.attn_norm\.weight"):
        decode(path, tmp_path / "out", snap)


def test_a_gemma3_embedding_cut_to_the_tokenizer_is_refused(tmp_path: Path) -> None:
    """Zero rows put back would be extra output classes llama.cpp does not have, so the decode refuses."""
    _, path, snap = _gemma3(tmp_path, cut=True)
    with pytest.raises(ValueError, match=r"token_embd\.weight: shape \(32, 64\) differs from snapshot"):
        decode(path, tmp_path / "out", snap)
