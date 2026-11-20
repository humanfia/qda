"""Source codecs: the fast decode against an explicit per-block reference, and per-weight selection."""

from __future__ import annotations

import torch
from qda_quant.intake.source import codec_for

#: e2m1 values by code, written out here so the reference does not share the codec's table.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def ref_blocks(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    """fp32 `weight * scale`, one scale per `block`, by explicit slicing; the last block may be short."""
    rows, cols = weight.shape
    out = torch.zeros(rows, cols, dtype=torch.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            r = slice(i * block[0], min((i + 1) * block[0], rows))
            c = slice(j * block[1], min((j + 1) * block[1], cols))
            out[r, c] = weight[r, c].float() * scale[i, j].item()
    return out


def ref_fp4(packed: torch.Tensor, scale: torch.Tensor, block_rows: int) -> torch.Tensor:
    """The e2m1 codes of `packed` (low nibble first), one scale per 32 codes, by scalar indexing."""
    rows, packed_cols = packed.shape
    out = torch.zeros(rows, 2 * packed_cols, dtype=torch.float32)
    for r in range(rows):
        for b in range(scale.shape[1]):
            raw = int(scale[r // block_rows, b])
            s = 2.0 ** (raw - 127) if scale.dtype == torch.uint8 else scale[r // block_rows, b].item()
            for c in range(b * 32, min((b + 1) * 32, out.shape[1])):
                byte = int(packed[r, c // 2])
                code = (byte >> 4) if c % 2 else byte & 0xF
                out[r, c] = E2M1[code] * s
    return out


def e8m0(exponents: list[list[int]]) -> torch.Tensor:
    """An E8M0 scale tensor (values ``2 ** (exponent - 127)``) from biased-exponent bytes."""
    return torch.tensor(exponents, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Two e2m1 codes per byte, the even one in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def test_fp8_block_f32_scale_with_short_last_block():
    """GLM's style: F8 weights, an F32 ``weight_scale_inv`` per 128x128, and a grid that does not divide."""
    codes = torch.randn(130, 129).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2, dtype=torch.float32) + 0.5
    headers = {"m.weight": ("F8_E4M3", [130, 129]), "m.weight_scale_inv": ("F32", [2, 2])}
    config = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}
    codec = codec_for(config, headers, "m.weight")
    assert codec.name == "fp8_block"
    decode = codec.decode({"m.weight": codes, "m.weight_scale_inv": scale}, torch.bfloat16)
    assert torch.equal(decode, ref_blocks(codes, scale, (128, 128)).to(torch.bfloat16))


def test_fp8_block_e8m0_scale():
    """DeepSeek-V4-Flash's dense style: the same grid with an E8M0 ``.scale``."""
    codes = torch.randn(256, 384).to(torch.float8_e4m3fn)
    scale = e8m0([[123, 125, 120], [127, 121, 126]])
    headers = {"m.weight": ("F8_E4M3", [256, 384]), "m.scale": ("F8_E8M0", [2, 3])}
    codec = codec_for({"quantization_config": {"quant_method": "fp8"}}, headers, "m.weight")
    assert codec.name == "fp8_block"
    decode = codec.decode({"m.weight": codes, "m.scale": scale}, torch.bfloat16)
    assert torch.equal(decode, ref_blocks(codes, scale, (128, 128)).to(torch.bfloat16))


def test_fp4_block_experts():
    """DeepSeek-V4-Flash's experts: e2m1 codes in I8, one E8M0 scale per row per 32 inputs."""
    codes = torch.randint(0, 16, (5, 64), dtype=torch.int64)
    packed = pack_fp4(codes)
    scale = e8m0([[124, 123], [126, 125], [122, 127], [127, 121], [121, 126]])
    headers = {"e.weight": ("I8", [5, 32]), "e.scale": ("F8_E8M0", [5, 2])}
    codec = codec_for({"expert_dtype": "fp4"}, headers, "e.weight")
    assert codec.name == "fp4_block"
    decode = codec.decode({"e.weight": packed, "e.scale": scale}, torch.bfloat16)
    assert torch.equal(decode, ref_fp4(packed, scale, block_rows=1).to(torch.bfloat16))


def test_fp4_block_dense_32x32():
    """V4.1-Flash's dense style: F8 weights, one E8M0 scale per 32x32 block (short blocks on 40 rows)."""
    codes = torch.randn(40, 40).to(torch.float8_e4m3fn)
    scale = e8m0([[125, 126], [123, 127]])
    headers = {"d.weight": ("F8_E4M3", [40, 40]), "d.scale": ("F8_E8M0", [2, 2])}
    codec = codec_for({}, headers, "d.weight")
    assert codec.name == "fp4_block"
    decode = codec.decode({"d.weight": codes, "d.scale": scale}, torch.bfloat16)
    assert torch.equal(decode, ref_blocks(codes, scale, (32, 32)).to(torch.bfloat16))


def test_mxfp4_packed():
    """Kimi K3's style: MXFP4 in ``weight_packed`` with a uint8 E8M0 exponent in ``weight_scale``."""
    codes = torch.randint(0, 16, (4, 64), dtype=torch.int64)
    packed = pack_fp4(codes)
    exponents = torch.tensor([[124, 123], [126, 125], [122, 127], [127, 121]], dtype=torch.uint8)
    headers = {
        "m.weight_packed": ("U8", [4, 32]),
        "m.weight_scale": ("U8", [4, 2]),
    }
    codec = codec_for({"quantization_config": {"quant_method": "compressed-tensors"}}, headers, "m.weight")
    assert codec.name == "mxfp4"
    assert codec.parts("m.weight") == ("m.weight_packed", "m.weight_scale")
    decode = codec.decode({"m.weight_packed": packed, "m.weight_scale": exponents}, torch.bfloat16)
    assert torch.equal(decode, ref_fp4(packed, exponents, block_rows=1).to(torch.bfloat16))


def test_selection_on_a_mixed_source():
    """V4-Flash mixes encodings: the codec follows the stored dtype and scale grid of each weight."""
    headers = {
        "d.weight": ("F8_E4M3", [256, 384]),
        "d.scale": ("F8_E8M0", [2, 3]),
        "e.weight": ("I8", [5, 32]),
        "e.scale": ("F8_E8M0", [5, 2]),
        "norm.weight": ("BF16", [8]),
    }
    config = {
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        "expert_dtype": "fp4",
    }
    assert codec_for(config, headers, "d.weight").name == "fp8_block"
    assert codec_for(config, headers, "e.weight").name == "fp4_block"
    assert codec_for(config, headers, "norm.weight").name == "float"
