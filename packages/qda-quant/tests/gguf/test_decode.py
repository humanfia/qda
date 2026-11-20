"""Decoding a GGUF back to the HF checkpoint: the pinned dequantizer, the layout inverse, and the output files.

The values a GGUF holds are fixed by llama.cpp, not by us: `gguf.quants.dequantize` is the pinned reader's own code,
so the digests below are what every type must keep returning. The name and layout inverse is then checked against
tiny files written as the converter writes them, including the value-head reorder and the MTP rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from qda_quant.gguf import file, kernel
from qda_quant.gguf.decode import _BLOCK, Gdn, _hf_name, _invert, _perm, decode

#: sha256 of (three blocks of seeded bytes, their fp32 values) from the pinned ``gguf/quants.py``; a change in
#: either is a change in what a decoded row means, so it must be redone with the pin, not waved through
_PIN_DIGESTS = {
    "F32": (
        "92afd7ea64b85970f545347027fa521598b8cc5ca586e3bea74548cea4a66d87",
        "92afd7ea64b85970f545347027fa521598b8cc5ca586e3bea74548cea4a66d87",
    ),
    "F16": (
        "e4264914588af0fa6209da9f820d027bee5afa2b90d6f489a67b2a8e91e22ba9",
        "b88776a0acce261e5ddb61149eb42da194015ac24c9f779a7fadf74fd584509b",
    ),
    "BF16": (
        "a56016c05ccabe1ff7be8ee15eaf6157d547963e7c254a34d917fd01d4e8d82e",
        "91618b14373aa5f0e6691f72439ee0234e7363ccd34b11ab4bd2872a42f46393",
    ),
    "Q4_0": (
        "c931b2f1cbbe36824224f8773f70f2cf5cc1ca19068fe50f351f1acb20ec86c9",
        "15575d7dfb56cbe18936c5a3b7986cddbc68bbe5f602f26785d0a0755607e9a7",
    ),
    "Q5_0": (
        "43376e567baed8faf5dbb0be878fe13c8020b351bf00edd0a0facb3d7e1e8ebd",
        "008238c47ae656f9c8b712ec9eb85c353a0b4079a4e2c4dd5843f84028e480a0",
    ),
    "Q8_0": (
        "6fea7c8f03d7217e2b4a84b748efd45c130f2bfe9844be3b892af8da491fa8f2",
        "7c4fb41c06b8be34ee66b3216806b4a5174568126c661a20f432ad5725d5f024",
    ),
    "Q3_K": (
        "65e1a418cf388edf25cd33bfc9c8eca28a8e0180aa64e9d56dc508710c6f39d5",
        "0751896dbf8a3d928c9a3d0619be1cf3c2e691cdc94a9f491ad51b0e5a106a9c",
    ),
    "Q4_K": (
        "c37aad9a2755671e037f2261faebe20cfc3d9eb27b2ee5f1052c10ef9d6b6a52",
        "feb61f6e323b3beca340267a124bfe181043bda912fb8fed62d087109f4bde12",
    ),
    "Q5_K": (
        "70aba77466162a86195945c87209b33a0771201048697cc4fec84888631e895e",
        "0046557e05c0282fc3313a4d09c912f362a573b3818e239bbeb10c49f5ef9d9f",
    ),
    "Q6_K": (
        "b00f7352dd15a464f7704afec0b26ebcea6f6e79924588fbf5a1066160c168a1",
        "664505c9b6e4f9534da532aabd6be7940d7762e9373a9c6f34db804ad746dd74",
    ),
    "IQ4_NL": (
        "bf18280603e17bdfad8635c0aa3fc3c0816d1eeaeaee5dc64797e2a9548dc713",
        "0c115c1a3c929d767f9c4f01013e17f48a9d0a2b47ab106a87f3823a643c6a23",
    ),
    "IQ4_XS": (
        "3dfc71edd79462bcc8a27a2133a981ab080432ec4fbd4463f7f953257232b3ba",
        "49b278969c9e52d73af448073caf00b7981a96242bf84a9b62c7dcc952f491ba",
    ),
    "IQ3_S": (
        "edda1365f7d50298cd4d58de9d3d20d0ccba2596bd6d84d192d12101cef6bc91",
        "bfed85f14b92860e0f1f2c8de5844daaa021b5cd8f221c4de07381a80c2f4c1f",
    ),
}


@pytest.mark.parametrize("name", list(_PIN_DIGESTS))
def test_the_pinned_dequantizer_returns_what_the_pin_returned(name: str) -> None:
    """Three blocks of bytes decode to the pinned llama.cpp's values, byte for byte."""
    gguf = pytest.importorskip("gguf")
    t = gguf.GGMLQuantizationType[name]
    _, size = gguf.GGML_QUANT_SIZES[t]
    raw = np.random.default_rng(20261107 + list(_PIN_DIGESTS).index(name)).integers(
        0, 256, size=3 * size, dtype=np.uint8
    )
    assert hashlib.sha256(raw.tobytes()).hexdigest() == _PIN_DIGESTS[name][0]
    vals = gguf.quants.dequantize(raw, t).astype("<f4")
    assert hashlib.sha256(vals.tobytes()).hexdigest() == _PIN_DIGESTS[name][1]


@pytest.mark.parametrize("name", ["Q4_K", "Q6_K"])
def test_our_codes_dequantize_exactly_as_the_pinned_reader_does(name: str) -> None:
    """What a build encodes reads back as the pinned ``dequantize`` says: our fit and ggml's agree on values."""
    gguf = pytest.importorskip("gguf")
    import qda_quant.gguf.kernel as k

    rng = np.random.default_rng(4)
    w = torch.from_numpy((rng.standard_normal((2, 512)) * 0.05).astype(np.float32))
    if name == "Q4_K":
        raw = k.quantize_q4k(w, workers=1)
        vals = _q4k_values(raw, 2, 512)
    else:
        raw = k.quantize_q6k(w, workers=1)
        vals = _q6k_values(raw, 2, 512)
    decoded = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType[name])
    assert np.array_equal(vals.reshape(-1), decoded)


def _q4k_values(raw: bytes, rows: int, width: int) -> np.ndarray:
    """``dequantize_row_q4_K``'s arithmetic, read through `q4k_fields`."""
    d, dmin, sc, m, codes = kernel.q4k_fields(raw)
    f16 = lambda bits: bits.view("<f2").astype("<f4")[:, None]  # noqa: E731
    return (
        f16(d) * np.repeat(sc.astype("<f4"), 32, -1) * codes - f16(dmin) * np.repeat(m.astype("<f4"), 32, -1)
    ).reshape(rows, width)


def _q6k_values(raw: bytes, rows: int, width: int) -> np.ndarray:
    """``dequantize_row_q6_K``'s arithmetic, read through `q6k_fields`."""
    d, scales, codes = kernel.q6k_fields(raw)
    per_group = codes.reshape(-1, 16, 16).astype("<f4") - 32
    return (
        d.view("<f2").astype("<f4")[:, None, None] * scales.astype("<f4")[:, :, None] * per_group
    ).reshape(rows, width)


def test_type_block_reads_the_writers_table_then_upstreams() -> None:
    """A type this writer encodes uses its own block table; another is read from upstream's; neither is guessed."""
    gguf = pytest.importorskip("gguf")
    assert file.type_block(file.rule.Q4_K) == (144, 256)
    assert file.type_block(int(gguf.GGMLQuantizationType.Q2_K)) == (84, 256)  # upstream's, not the writer's
    assert file.row_bytes(file.rule.Q6_K, 5120) == 5120 // 256 * 210
    with pytest.raises(ValueError, match="need a multiple of 256"):
        file.row_bytes(file.rule.Q4_K, 1000)
    with pytest.raises(ValueError, match="not one this build reads"):
        file.type_block(999)


def test_every_tensor_name_maps_to_its_hf_module() -> None:
    """Language tensors, the MTP block's four renames, and nothing else."""
    from qda_quant.gguf.decode import _GLOBAL

    for suffix, hf in _BLOCK.items():
        assert _hf_name(f"blk.3.{suffix}", 64, _BLOCK) == (f"model.language_model.layers.3.{hf}", suffix)
    for gguf_name, hf in _GLOBAL.items():
        assert _hf_name(gguf_name, 64, _BLOCK) == (hf, gguf_name)
    assert _hf_name("blk.64.attn_norm.weight", 64, _BLOCK) == (
        "mtp.layers.0.input_layernorm.weight",
        "attn_norm.weight",
    )
    for suffix, hf in {
        "nextn.eh_proj.weight": "mtp.fc.weight",
        "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "nextn.shared_head_norm.weight": "mtp.norm.weight",
    }.items():
        assert _hf_name(f"blk.64.{suffix}", 64, _BLOCK)[0] == hf
    for bad in ("visual.blocks.0.weight", "blk.0.mystery.weight", "blk.x.attn_norm.weight", "output.bias"):
        with pytest.raises(ValueError, match="not a language-model tensor of this architecture"):
            _hf_name(bad, 64, _BLOCK)


def test_the_value_head_reorder_is_its_own_inverse() -> None:
    """The grouped-to-tiled permutation undoes itself, so no layer order can survive the round trip by accident."""
    for nk, per, dim in ((16, 3, 128), (2, 2, 4), (1, 1, 1)):
        forward, inverse = _perm(nk, per, dim, False), _perm(nk, per, dim, True)
        assert torch.equal(forward[inverse], torch.arange(nk * per * dim))
        assert torch.equal(inverse[forward], torch.arange(nk * per * dim))


def test_the_layout_transforms_invert_the_converters_writes() -> None:
    """Each stored layout decodes to the HF tensor the converter started from, shapes included."""
    g = cast(Gdn, SimpleNamespace(nk=2, per=3, head_v=4, head_k=5))
    qk, v, nv = 2 * g.head_k * g.nk, g.nk * g.per * g.head_v, g.nk * g.per  # 20 Q+K rows, 24 V, 6 heads
    fwd, fwd1 = _perm(g.nk, g.per, g.head_v, False), _perm(g.nk, g.per, 1, False)
    x = torch.arange((qk + v) * 8, dtype=torch.float32).reshape(qk + v, 8) / 4  # 44 rows: 20 Q+K, 24 V
    heads = torch.arange(nv * 8, dtype=torch.float32).reshape(nv, 8) / 4  # in_proj_a/b: one row per head
    conv = torch.arange((qk + v) * 4, dtype=torch.float32).reshape(qk + v, 1, 4) / 4
    out = torch.arange(8 * v, dtype=torch.float32).reshape(8, v) / 4  # out_proj: one column per V channel
    a_log = torch.linspace(-2, 2, nv)

    # the file's layout (the converter's writes) and the HF tensor each must decode to
    stored = {
        "attn_qkv.weight": torch.cat([x[:qk], x[qk:][fwd]]),
        "attn_gate.weight": x[:v][fwd],
        "ssm_alpha.weight": heads[fwd1],
        "ssm_beta.weight": heads[fwd1],
        "ssm_dt.bias": heads[:, 0][fwd1],
        "ssm_a": -torch.exp(a_log)[fwd1],
        "ssm_out.weight": out[:, fwd],
        "ssm_conv1d.weight": torch.cat([conv[:qk, 0], conv[qk:, 0][fwd]]),
    }
    want = {
        "attn_qkv.weight": x,
        "attn_gate.weight": x[:v],
        "ssm_alpha.weight": heads,
        "ssm_beta.weight": heads,
        "ssm_dt.bias": heads[:, 0],
        "ssm_a": torch.log(torch.exp(a_log)),  # log(-x) in the decoder: the same two operations
        "ssm_out.weight": out,
        "ssm_conv1d.weight": conv,
    }
    for suffix, made in stored.items():
        got = _invert(suffix, made, g)
        assert got.shape == want[suffix].shape, suffix
        assert torch.equal(got, want[suffix]), suffix


def test_norms_are_stored_plus_one_except_the_gdn_norm() -> None:
    """Every RMS gain but ``linear_attn.norm`` was written +1; subtracting it is exact, and the GDN norm is left."""
    gain = torch.tensor([0.5, -0.25, 1.0, 2.0, -3.0])
    assert torch.equal(_invert("attn_norm.weight", gain + 1, cast(Gdn, None)), gain)
    assert torch.equal(_invert("ssm_norm.weight", gain, cast(Gdn, None)), gain)


def test_shards_keep_order_and_stay_within_the_size(monkeypatch) -> None:  # noqa: ANN001
    """A shard boundary falls before a tensor that would overflow it, and the order is the file's."""
    import qda_quant.gguf.decode as m

    monkeypatch.setattr(m, "SHARD", 6)
    sizes = {"a": 3, "b": 3, "c": 1, "d": 5}
    assert m._shards(list(sizes), sizes) == [["a", "b"], ["c", "d"]]
    monkeypatch.setattr(m, "SHARD", 4)
    assert m._shards(["a"], {"a": 10**9}) == [["a"]]  # one tensor larger than a shard is still written


def test_a_decoded_row_is_a_servable_checkpoint(tmp_path: Path) -> None:
    """A whole tiny row: names, the MTP split, the index, the config copy, and the Q4_K values.

    The GGUF is written the way the converter writes one (``ne[0]`` the row width, ``+1`` norms, the tiled value
    heads), so the test covers the reader, every inverse transform, the shard/index layout and the manifest at once.
    """
    gguf = pytest.importorskip("gguf")
    snap = _snapshot(tmp_path)
    source, gguf_path = _tiny_row(tmp_path, gguf)
    out = tmp_path / "out"
    manifest = decode(gguf_path, out, snap)

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        k: "model-00001-of-00001.safetensors" for k in source if not k.startswith("mtp.")
    }
    from safetensors.torch import load_file

    weights = load_file(str(out / "model-00001-of-00001.safetensors"))
    for name, want in source.items():
        if name.startswith("mtp."):
            continue
        assert name in weights, name
        assert weights[name].shape == want.shape, name
        # exact in bf16 (or the Q4_K tensor's dequantized values, also compared after the decoder's cast)
        assert torch.equal(weights[name], want.to(torch.bfloat16)), name
    mtp = load_file(str(out / "model-mtp.safetensors"))
    assert set(mtp) == {k for k in source if k.startswith("mtp.")}
    assert (out / "config.json").read_bytes() == (snap / "config.json").read_bytes()
    assert (out / "tokenizer.json").read_bytes() == (snap / "tokenizer.json").read_bytes()
    assert manifest["gguf"]["sha256"] == hashlib.sha256(gguf_path.read_bytes()).hexdigest()
    assert manifest["types"] == {"F32": len(source) - 1, "Q4_K": 1}
    assert manifest["dtype_inexact"]["model.language_model.layers.0.mlp.gate_proj.weight"] > 0
    for name, f in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == f["sha256"]
    assert decode.__module__ == "qda_quant.gguf.decode"


def test_a_file_without_the_mtp_count_decodes(tmp_path: Path) -> None:
    """A `--no-mtp` file omits the optional count: it reads as zero, so every tensor is language-model."""
    gguf = pytest.importorskip("gguf")
    snap = _snapshot(tmp_path)
    source, gguf_path = _tiny_row(tmp_path, gguf, mtp=False)
    out = tmp_path / "out"
    manifest = decode(gguf_path, out, snap)

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {k: "model-00001-of-00001.safetensors" for k in source}
    assert manifest["mtp"] == []
    assert not (out / "model-mtp.safetensors").exists()
    assert manifest["tensors"] == len(source)
    assert all(not k.startswith("mtp.") for k in index["weight_map"])
    assert manifest["types"] == {"F32": len(source) - 1, "Q4_K": 1}


def test_scoring_reads_the_same_checkpoint_with_or_without_the_mtp_head(tmp_path: Path) -> None:
    """The head is a separate, unindexed file: a KL reading of the two rows is the same score.

    vLLM loads the indexed shards only, so keeping or dropping the head must leave every indexed byte -- and
    the index itself -- as it was. This is what lets a re-export that adds the head keep its KL.
    """
    gguf = pytest.importorskip("gguf")
    with_head, without = tmp_path / "with", tmp_path / "without"
    with_head.mkdir()
    without.mkdir()
    _tiny_row(with_head, gguf, mtp=True)
    _tiny_row(without, gguf, mtp=False)
    snap = _snapshot(tmp_path)

    kept = decode(with_head / "tiny.gguf", with_head / "out", snap)
    plain = decode(without / "tiny.gguf", without / "out", snap)

    index = "model.safetensors.index.json"
    assert (with_head / "out" / index).read_bytes() == (without / "out" / index).read_bytes()
    shards = [name for name in kept["files"] if name.startswith("model-") and name != "model-mtp.safetensors"]
    assert shards == [
        name for name in plain["files"] if name.startswith("model-") and name != "model-mtp.safetensors"
    ]
    for name in shards:
        assert (with_head / "out" / name).read_bytes() == (without / "out" / name).read_bytes(), name
    assert kept["mtp"] and not plain["mtp"]
    assert not (without / "out" / "model-mtp.safetensors").exists()


def test_an_absent_mtp_count_reads_as_zero_and_is_still_required_explicitly(tmp_path: Path) -> None:
    """The NextN count is optional in the pinned loader: absent means zero, and only a caller asking strictly fails.

    A 40-layer file with no NextN key stands in for the multi-layer no-MTP rows (`--no-mtp`), header only.
    """
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(tmp_path / "no-mtp.gguf"), "qwen35")
    w.add_block_count(40)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    t = file.Template(tmp_path / "no-mtp.gguf")
    assert t.number("qwen35.nextn_predict_layers", 0) == 0
    assert t.number("qwen35.block_count") == 40
    with pytest.raises(KeyError):
        t.number("qwen35.nextn_predict_layers")


def _snapshot(tmp_path: Path) -> Path:
    """The two files the decoder takes from the snapshot: a Qwen3.5 config and a tokenizer."""
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {}}
        )
        + "\n"
    )
    (snap / "tokenizer.json").write_text('{"model": {}}\n')
    return snap


def _tiny_row(tmp_path: Path, gguf, mtp: bool = True) -> tuple[dict[str, torch.Tensor], Path]:  # noqa: ANN001
    """A qwen35 GGUF (one GDN layer, and one MTP block unless `mtp` is off) and the HF tensors it came from.

    Values are quarters of small integers, which bf16 holds exactly, so a bit comparison is meaningful. With
    `mtp` off the file carries neither the NextN count nor the MTP block, as one converted with `--no-mtp` does.
    """
    nk, nv, per, head_v, head_k, hidden = 2, 4, 2, 4, 4, 256
    qk, v = 2 * head_k * nk, nv * head_v
    fwd, fwd1 = _perm(nk, per, head_v, False), _perm(nk, per, 1, False)
    w = gguf.GGUFWriter(str(tmp_path / "tiny.gguf"), "qwen35")
    w.add_block_count(2 if mtp else 1)
    if mtp:
        w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_uint32("qwen35.ssm.group_count", nk)
    w.add_uint32("qwen35.ssm.time_step_rank", nv)
    w.add_uint32("qwen35.ssm.inner_size", nv * head_v)
    w.add_uint32("qwen35.ssm.state_size", head_k)

    def vals(n: int) -> torch.Tensor:
        return (torch.arange(n) % 16 - 8 + 0.25).float()  # quarters of small integers: exact in bf16

    tensors = {
        "token_embd.weight": vals(32 * hidden).reshape(32, hidden),
        "output_norm.weight": vals(hidden) + 1,
        "output.weight": vals(32 * hidden).reshape(32, hidden),
        "blk.0.attn_norm.weight": vals(hidden) + 1,
        "blk.0.post_attention_norm.weight": vals(hidden) + 1,
        "blk.0.ffn_up.weight": vals(hidden * hidden).reshape(hidden, hidden),
        "blk.0.ffn_down.weight": vals(hidden * hidden).reshape(hidden, hidden),
        "blk.0.attn_qkv.weight": torch.cat(
            [vals(qk * hidden).reshape(qk, hidden), vals(v * hidden).reshape(v, hidden)[fwd]]
        ),
        "blk.0.attn_gate.weight": vals(v * hidden).reshape(v, hidden)[fwd],
        "blk.0.ssm_alpha.weight": vals(nv * hidden).reshape(nv, hidden)[fwd1],
        "blk.0.ssm_beta.weight": vals(nv * hidden).reshape(nv, hidden)[fwd1],
        "blk.0.ssm_out.weight": vals(hidden * v).reshape(hidden, v)[:, fwd],
        "blk.0.ssm_a": -torch.exp(vals(nv))[fwd1],
        "blk.0.ssm_dt.bias": vals(nv)[fwd1],
        "blk.0.ssm_conv1d.weight": torch.cat([vals(qk * 4).reshape(qk, 4), vals(v * 4).reshape(v, 4)[fwd]]),
        "blk.0.ssm_norm.weight": vals(head_v),
        "blk.1.nextn.eh_proj.weight": torch.arange(hidden * 2 * hidden).reshape(2 * hidden, hidden),
        "blk.1.nextn.enorm.weight": vals(hidden) + 1,
        "blk.1.nextn.hnorm.weight": vals(hidden) + 1,
        "blk.1.nextn.shared_head_norm.weight": vals(hidden) + 1,
    }
    if not mtp:
        tensors = {n: t for n, t in tensors.items() if not n.startswith("blk.1.nextn.")}
    for (
        name,
        t,
    ) in tensors.items():  # numpy order (rows, cols); the writer stores ne reversed, as GGUF defines
        w.add_tensor(
            name, t.numpy().astype("<f4"), raw_shape=tuple(t.shape), raw_dtype=gguf.GGMLQuantizationType.F32
        )
    hf = {
        "model.language_model.embed_tokens.weight": tensors["token_embd.weight"],
        "model.language_model.norm.weight": tensors["output_norm.weight"] - 1,
        "lm_head.weight": tensors["output.weight"],
        **{
            f"model.language_model.layers.0.{n}": t
            for n, t in {
                "input_layernorm.weight": tensors["blk.0.attn_norm.weight"] - 1,
                "post_attention_layernorm.weight": tensors["blk.0.post_attention_norm.weight"] - 1,
                "mlp.up_proj.weight": tensors["blk.0.ffn_up.weight"],
                "mlp.down_proj.weight": tensors["blk.0.ffn_down.weight"],
                "mlp.gate_proj.weight": tensors["blk.0.ffn_up.weight"].clone(),  # filled from Q4_K below
                "linear_attn.in_proj_qkv.weight": torch.cat(
                    [
                        tensors["blk.0.attn_qkv.weight"][:qk],
                        tensors["blk.0.attn_qkv.weight"][qk:][_perm(nk, per, head_v, True)],
                    ]
                ),
                "linear_attn.in_proj_z.weight": tensors["blk.0.attn_gate.weight"][
                    _perm(nk, per, head_v, True)
                ],
                "linear_attn.in_proj_a.weight": tensors["blk.0.ssm_alpha.weight"][_perm(nk, per, 1, True)],
                "linear_attn.in_proj_b.weight": tensors["blk.0.ssm_beta.weight"][_perm(nk, per, 1, True)],
                "linear_attn.out_proj.weight": tensors["blk.0.ssm_out.weight"][
                    :, _perm(nk, per, head_v, True)
                ],
                "linear_attn.A_log": torch.log(-tensors["blk.0.ssm_a"][_perm(nk, per, 1, True)]),
                "linear_attn.dt_bias": tensors["blk.0.ssm_dt.bias"][_perm(nk, per, 1, True)],
                "linear_attn.conv1d.weight": torch.cat(
                    [
                        tensors["blk.0.ssm_conv1d.weight"][:qk],
                        tensors["blk.0.ssm_conv1d.weight"][qk:][_perm(nk, per, head_v, True)],
                    ]
                ).reshape(qk + v, 1, 4),
                "linear_attn.norm.weight": tensors["blk.0.ssm_norm.weight"],
            }.items()
        },
        **(
            {
                "mtp.fc.weight": tensors["blk.1.nextn.eh_proj.weight"],
                "mtp.pre_fc_norm_embedding.weight": tensors["blk.1.nextn.enorm.weight"] - 1,
                "mtp.pre_fc_norm_hidden.weight": tensors["blk.1.nextn.hnorm.weight"] - 1,
                "mtp.norm.weight": tensors["blk.1.nextn.shared_head_norm.weight"] - 1,
            }
            if mtp
            else {}
        ),
    }
    # the quantized tensor: encoded into the file, so its decoded values are the pinned dequantizer's, rounded
    gate = vals(hidden * hidden).reshape(hidden, hidden)
    raw = kernel.quantize_q4k(gate, workers=1)
    w.add_tensor(  # a uint8 payload: the writer reads raw_shape as byte shape, 144 bytes per 256 values
        "blk.0.ffn_gate.weight",
        np.frombuffer(raw, dtype=np.uint8),
        raw_shape=(hidden, 144),
        raw_dtype=gguf.GGMLQuantizationType.Q4_K,
    )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    dequant = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType.Q4_K)
    hf["model.language_model.layers.0.mlp.gate_proj.weight"] = torch.from_numpy(dequant).reshape(
        hidden, hidden
    )
    return hf, tmp_path / "tiny.gguf"


def test_decode_gguf_refuses_another_architecture_in_two_lines(tmp_path, capsys):
    """`qda decode-gguf` on a GGUF it cannot read prints a refusal listing the architectures it reads, not a traceback."""
    gguf = pytest.importorskip("gguf")
    from qda_quant.gguf.decode import OTHER_GGUF, run

    path = tmp_path / "phi3.gguf"
    w = gguf.GGUFWriter(str(path), "phi3")
    w.add_block_count(1)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    snap = tmp_path / "snap"
    snap.mkdir()
    a = argparse.Namespace(gguf=str(path), out=str(tmp_path / "out"), snapshot=str(snap), dtype="bf16")
    code = run(a)
    err = capsys.readouterr().err
    assert code != 0
    assert "architecture 'phi3'" in err and "Traceback" not in err
    assert "fix:" in err and OTHER_GGUF in err
    assert "qwen3moe" in OTHER_GGUF and "llamacpp" not in OTHER_GGUF  # the llama.cpp KL scorer is removed


def _qwen3_row(tmp_path: Path, gguf, extra: str | None = None) -> tuple[dict[str, torch.Tensor], Path, Path]:  # noqa: ANN001
    """A one-layer qwen3 GGUF (tied embedding, one Q4_K projection), the HF tensors, and their snapshot.

    The converter renames qwen3 tensors only, so the snapshot holds the same values under HF names. `extra` adds a
    GGUF tensor no snapshot tensor names.
    """
    from safetensors.torch import save_file

    hidden = 256
    vals = lambda n: (torch.arange(n) % 16 - 8 + 0.25).float()  # noqa: E731  quarters: exact in bf16
    hf = {
        "model.embed_tokens.weight": vals(32 * hidden).reshape(32, hidden),
        "model.norm.weight": vals(hidden),
        "model.layers.0.input_layernorm.weight": vals(hidden),
        "model.layers.0.self_attn.k_norm.weight": vals(128),
        "model.layers.0.self_attn.o_proj.weight": vals(hidden * hidden).reshape(hidden, hidden),
    }
    gguf_names = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
        "model.layers.0.self_attn.k_norm.weight": "blk.0.attn_k_norm.weight",
    }
    w = gguf.GGUFWriter(str(tmp_path / "qwen3.gguf"), "qwen3")
    w.add_block_count(1)
    for name, t in hf.items():
        if name in gguf_names:
            w.add_tensor(gguf_names[name], t.numpy().astype("<f4"), raw_shape=tuple(t.shape))
    raw = kernel.quantize_q4k(hf["model.layers.0.self_attn.o_proj.weight"], workers=1)
    w.add_tensor(
        "blk.0.attn_output.weight",
        np.frombuffer(raw, dtype=np.uint8),
        raw_shape=(hidden, 144),
        raw_dtype=gguf.GGMLQuantizationType.Q4_K,
    )
    if extra:
        w.add_tensor(extra, np.zeros(hidden, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3", "tie_word_embeddings": True}) + "\n")
    (snap / "tokenizer.json").write_text('{"model": {}}\n')
    save_file({k: v.to(torch.bfloat16) for k, v in hf.items()}, str(snap / "model.safetensors"))
    dequant = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType.Q4_K)
    hf["model.layers.0.self_attn.o_proj.weight"] = torch.from_numpy(dequant).reshape(hidden, hidden)
    return hf, tmp_path / "qwen3.gguf", snap


def test_a_qwen3_file_decodes_under_the_snapshots_names(tmp_path: Path) -> None:
    """qwen3 is renamed only: every snapshot tensor comes back under its own name with the file's values."""
    gguf = pytest.importorskip("gguf")
    from safetensors.torch import load_file

    source, gguf_path, snap = _qwen3_row(tmp_path, gguf)
    manifest = decode(gguf_path, tmp_path / "out", snap)

    weights = load_file(str(tmp_path / "out" / "model-00001-of-00001.safetensors"))
    assert set(weights) == set(source)
    for name, want in source.items():
        assert torch.equal(weights[name], want.to(torch.bfloat16)), name
    assert manifest["types"] == {"F32": 4, "Q4_K": 1}
    assert manifest["mtp"] == [] and "note" not in manifest["config"]
    assert (tmp_path / "out" / "config.json").read_bytes() == (snap / "config.json").read_bytes()


def test_a_qwen3_file_with_a_tensor_the_snapshot_lacks_is_refused(tmp_path: Path) -> None:
    """A GGUF tensor no snapshot tensor names would be dropped silently; the decoder refuses instead."""
    gguf = pytest.importorskip("gguf")
    _, gguf_path, snap = _qwen3_row(tmp_path, gguf, extra="blk.0.ffn_norm.weight")
    with pytest.raises(ValueError, match=r"blk\.0\.ffn_norm\.weight"):
        decode(gguf_path, tmp_path / "out", snap)


def test_a_snapshot_of_another_model_type_is_refused_before_writing(tmp_path: Path) -> None:
    """The snapshot's config is copied as the row's, so a Qwen3.5 snapshot under a qwen3 file is refused."""
    gguf = pytest.importorskip("gguf")
    _, gguf_path, snap = _qwen3_row(tmp_path, gguf)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {}}))
    with pytest.raises(ValueError, match="model_type 'qwen3_5'; a qwen3 GGUF needs 'qwen3'"):
        decode(gguf_path, tmp_path / "out", snap)
    assert not (tmp_path / "out").exists()


def test_a_fused_qkv_qwen3_file_is_refused_by_name(tmp_path: Path) -> None:
    """`convert_hf_to_gguf.py --fuse-qkv` writes one attn_qkv; the decoder says so rather than a missing attn_q."""
    gguf = pytest.importorskip("gguf")
    _, gguf_path, snap = _qwen3_row(tmp_path, gguf, extra="blk.0.attn_qkv.weight")
    with pytest.raises(ValueError, match=r"blk\.0\.attn_qkv\.weight: a fused QKV file \(--fuse-qkv\)"):
        decode(gguf_path, tmp_path / "out", snap)
