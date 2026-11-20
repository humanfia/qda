"""The ``iq4nl``, ``iq4xs`` and ``q3k`` grids against llama.cpp's own decoders.

A record's meaning is llama.cpp's, not ours: ``gguf.quants.dequantize`` is the reference dequantizer, and a
tensor written at one of these types has to decode through it to what the grid says its codes and scales were.
Each test writes records and reads them back, so a field that lands in the wrong place -- the bug class this
format work is about -- cannot pass as plausible numbers. One test flips a code bit and a scale bit and requires
the comparison to fail, so the readback is shown able to fail and is not a label.

Every test except ``test_gptq_codes_reach_a_file_at_that_type`` runs without extra flags, and CI runs those. A
build fits its template's other tensors through native ggml, so that test needs ``--ggml-lib PATH`` (a built
``libggml-base``; see ``tests/conftest.py``); CI does not pass the flag, so it is skipped there. Whoever changes
``qda_quant/gguf`` runs it with the flag before pushing, and the reviewer re-runs it on the branch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from qda_quant.config import BuildConfig
from qda_quant.format import IQ4NL_FORMAT, IQ4XS_FORMAT, Q3K_FORMAT, Format
from qda_quant.gguf.iq4 import (
    iq4nl_fields,
    iq4nl_records,
    iq4xs_fields,
    iq4xs_records,
    quantize_iq4nl,
    quantize_iq4xs,
)
from qda_quant.gguf.q3k import q3k_fields, q3k_records, quantize_q3k
from qda_quant.solve.grid import GRIDS
from qda_quant.step import Refused

#: ``kvalues_iq4nl``, the sixteen values an IQ4 code means.
KVALUES = np.array(
    (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113), dtype=np.float32
)

#: name, format, round-to-nearest quantizer, llama.cpp type, values per stored block, values per code group
_CASES = [
    ("iq4nl", IQ4NL_FORMAT, quantize_iq4nl, 20, 32, 32),
    ("iq4xs", IQ4XS_FORMAT, quantize_iq4xs, 23, 256, 32),
    ("q3k", Q3K_FORMAT, quantize_q3k, 11, 256, 16),
]


def _evidence(gguf, kind: int, raw: bytes) -> np.ndarray:
    """llama.cpp's dequantization of `raw` as fp32."""
    return gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType(kind)).astype(
        np.float32
    )


def _fp16(value: float) -> int:
    """The fp16 bit pattern of `value`, which must be exact."""
    return int(np.array([value], dtype=np.float16).view("<u2")[0])


@pytest.mark.parametrize(("name", "fmt", "quantize", "kind", "block", "group"), _CASES)
def test_a_written_tensor_decodes_to_the_grids_own_values(name, fmt, quantize, kind, block, group) -> None:
    """A tensor quantized at one of these types decodes through llama.cpp to the grid's values, bit for bit.

    This is the readback the format exists for: what the fit and the codes said is what a llama.cpp reader
    gets back, not a plausible approximation of it. The records also re-encode from their fields, so the
    reader is the writer's inverse on the bytes a build produces.
    """
    gguf = pytest.importorskip("gguf")
    torch.manual_seed(0)
    w = torch.randn(3, 512) * 0.05
    w[1, :64] = 0.0  # a zero group keeps a zero scale
    raw = quantize(w)
    grid, parsed = GRIDS[name], Format.parse(fmt)
    scale, _ = grid.fit(parsed, w)
    wide = scale.reshape(3, -1).repeat_interleave(group, -1)
    mine = grid.value(grid.codes(w, wide, None, parsed.qmin, parsed.qmax), wide, None)
    decoded = _evidence(gguf, kind, raw).reshape(w.shape)
    assert np.array_equal(decoded, mine.numpy()), (
        f"{name}: llama.cpp disagrees at {np.argwhere(decoded != mine.numpy())[:4]}"
    )
    fields = {"iq4nl": iq4nl_fields, "iq4xs": iq4xs_fields, "q3k": q3k_fields}[name](raw)
    packed = {"iq4nl": iq4nl_records, "iq4xs": iq4xs_records, "q3k": q3k_records}[name](*fields)
    assert packed == raw, f"{name}: the fields do not re-encode to the bytes they came from"


def _both(values: np.ndarray, width: int, what: str) -> None:
    """Assert each of the low `width` bits of a packed field takes both values."""
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    for k in range(width):
        col = (flat >> k) & 1
        assert col.min() == 0 and col.max() == 1, f"{what}: bit {k} never takes both values"


@pytest.mark.parametrize(
    ("name", "fmt", "kind", "block", "group"), [(n, f, k, b, g) for n, f, _, k, b, g in _CASES]
)
def test_every_code_and_scale_bit_position_decodes_as_llama_cpp_reads_it(
    name, fmt, kind, block, group
) -> None:
    """Codes and scales set independently of any fit exercise every stored bit, and llama.cpp agrees.

    A fit round-trips whatever its blocks happened to use; a fixed pattern is what makes a positional mix-up
    visible, including the sign of a 6-bit scale and the inverted high bit of a three-bit code. The coverage is
    asserted, not claimed: every code value is present with both values of every code bit, all 64 stored 6-bit
    scale values are present with both values of every scale bit, and IQ4_NL's fp16 ``d`` appears with both
    values of each of its 16 bits.
    """
    gguf = pytest.importorskip("gguf")
    d = _fp16(0.5)  # every product is exact in fp32
    base = np.arange(block, dtype=np.int64) * 7 + 3
    if name == "q3k":
        codes = np.broadcast_to((base % 8 - 4)[None], (4, block)).copy()
        scales = np.arange(-32, 32, dtype=np.int16).reshape(4, 16)  # all 64 stored values, both signs
        raw = q3k_records(np.full(4, d, dtype="<u2"), scales, codes)
        _, stored, got = q3k_fields(raw)
        expect = 0.5 * np.repeat(scales.astype(np.float32), 16, axis=1) * codes
        assert set(got.reshape(-1).tolist()) == set(range(-4, 4))
        _both(got + 4, 3, f"{name} code")
        assert set(stored.reshape(-1).tolist()) == set(range(-32, 32))
        _both(stored + 32, 6, f"{name} scale")
    elif name == "iq4xs":
        codes = np.broadcast_to((base % 16)[None], (8, block)).copy()
        scales = np.arange(-32, 32, dtype=np.int16).reshape(8, 8)  # all 64 stored values, both signs
        raw = iq4xs_records(np.full(8, d, dtype="<u2"), scales, codes)
        _, stored, got = iq4xs_fields(raw)
        expect = 0.5 * np.repeat(scales.astype(np.float32), 32, axis=1) * KVALUES[codes]
        assert set(got.reshape(-1).tolist()) == set(range(16))
        _both(got, 4, f"{name} code")
        assert set(stored.reshape(-1).tolist()) == set(range(-32, 32))
        _both(stored + 32, 6, f"{name} scale")
    else:
        bits = np.array([d ^ (1 << b) for b in range(16)], dtype="<u2")  # every bit of d, both values
        codes = np.broadcast_to((base % 16)[None], (16, block)).copy()
        raw = iq4nl_records(bits, codes)
        _, got = iq4nl_fields(raw)
        expect = bits.view(np.float16).astype(np.float32)[:, None] * KVALUES[codes]
        assert set(got.reshape(-1).tolist()) == set(range(16))
        _both(got, 4, f"{name} code")
        _both(bits, 16, f"{name} d")
    assert np.array_equal(_evidence(gguf, kind, raw), expect.astype(np.float32).reshape(-1)), (
        f"{name}: llama.cpp disagrees with the fields the record holds"
    )


@pytest.mark.parametrize(
    ("name", "fmt", "quantize", "kind", "group", "code_byte", "scale_byte"),
    [
        ("iq4nl", IQ4NL_FORMAT, quantize_iq4nl, 20, 32, 2, 0),
        ("iq4xs", IQ4XS_FORMAT, quantize_iq4xs, 23, 32, 8, 4),
        ("q3k", Q3K_FORMAT, quantize_q3k, 11, 16, 32, 96),
    ],
)
def test_the_readback_comparison_catches_a_flipped_code_bit_and_a_flipped_scale_bit(
    name, fmt, quantize, kind, group, code_byte, scale_byte
) -> None:
    """The readback is a control, not a label: flipping one code bit and one scale bit makes it fail.

    Each flip is a single bit of the written bytes: one inside a packed code and one inside a packed scale (for
    IQ4_NL, inside the fp16 ``d`` that is its scale). Either must stop llama.cpp's decode matching the grid's
    own values. The clean record is compared first, so a comparison that fails on everything cannot pass this.
    """
    gguf = pytest.importorskip("gguf")
    torch.manual_seed(3)
    w = torch.randn(2, 512) * 0.05
    raw = quantize(w)
    grid, parsed = GRIDS[name], Format.parse(fmt)
    scale, _ = grid.fit(parsed, w)
    wide = scale.reshape(2, -1).repeat_interleave(group, -1)
    mine = grid.value(grid.codes(w, wide, None, parsed.qmin, parsed.qmax), wide, None).numpy()
    assert np.array_equal(_evidence(gguf, kind, raw).reshape(w.shape), mine), f"{name}: the clean record"

    for what, at in (("code", code_byte), ("scale", scale_byte)):
        corrupt = bytearray(raw)
        corrupt[at] ^= 0x01
        assert not np.array_equal(_evidence(gguf, kind, bytes(corrupt)).reshape(w.shape), mine), (
            f"{name}: flipping the {what} bit at byte {at} left the decode equal; the comparison proves nothing"
        )


def test_the_grids_take_only_their_own_format_and_whole_blocks() -> None:
    """A format another grid stores, and a row that is not whole blocks, are refused rather than rounded."""
    for name, fmt, other in (
        ("iq4nl", IQ4NL_FORMAT, Q3K_FORMAT),
        ("iq4xs", IQ4XS_FORMAT, Q3K_FORMAT),
        ("q3k", Q3K_FORMAT, IQ4NL_FORMAT),
    ):
        grid = GRIDS[name]
        assert grid.refusal(Format.parse(fmt), None) is None
        assert grid.refusal(Format.parse(other), None), f"{name} accepts {other}"
    with pytest.raises(Refused):
        GRIDS["q3k"].fit(Format.parse(Q3K_FORMAT), torch.zeros(1, 24))
    with pytest.raises(Refused):
        GRIDS["iq4nl"].fit(Format.parse(IQ4NL_FORMAT), torch.zeros(1, 24))


@pytest.mark.parametrize(("name", "fmt"), [("iq4nl", IQ4NL_FORMAT), ("iq4xs", IQ4XS_FORMAT)])
def test_the_fit_puts_a_groups_extreme_on_a_table_entry(name, fmt) -> None:
    """The group's largest magnitude codes to the table's extreme entry for its side, before the scale rounds.

    IQ4_XS then rounds its 6-bit factor, which can leave a group's reconstruction short of that magnitude (see
    ``iq4.py``); the code the extreme gets is still the extreme entry.
    """
    torch.manual_seed(1)
    w = torch.randn(2, 256) * 0.05
    grid, parsed = GRIDS[name], Format.parse(fmt)
    scale, _ = grid.fit(parsed, w)
    wide = scale.reshape(2, -1).repeat_interleave(32, -1)
    codes = grid.codes(w, wide, None, parsed.qmin, parsed.qmax)
    groups = w.reshape(2, -1, 32)
    at = groups.abs().argmax(-1)[..., None]
    at_code = codes.reshape(2, -1, 32).gather(-1, at).squeeze(-1)
    signed = groups.gather(-1, at).squeeze(-1)
    assert torch.equal(
        at_code, torch.where(signed >= 0, torch.full_like(at_code, 15.0), torch.zeros_like(at_code))
    )


def test_the_fit_never_needs_more_than_the_six_bit_scale_range() -> None:
    """Every fitted product is an exact fp16 ``d`` times a 6-bit integer, which is what the writer recovers."""
    from qda_quant.gguf.scales import SCALE_MAX, SCALE_MIN, split_6bit

    torch.manual_seed(2)
    for name, fmt in (("iq4xs", IQ4XS_FORMAT), ("q3k", Q3K_FORMAT)):
        w = torch.randn(4, 512) * 0.05
        scale, _ = GRIDS[name].fit(Format.parse(fmt), w)
        products = scale.reshape(-1, 256 // (32 if name == "iq4xs" else 16)).numpy()
        d, s = split_6bit(products, name)
        restored = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)[:, None] * s
        assert np.array_equal(restored, products), (
            f"{name}: the recovered pair does not reproduce the products"
        )
        assert s.min() >= SCALE_MIN and s.max() <= SCALE_MAX


@pytest.mark.parametrize(
    ("name", "fmt", "type_name", "code_byte", "scale_byte"),
    [
        ("iq4nl", IQ4NL_FORMAT, "IQ4_NL", 2, 0),
        ("iq4xs", IQ4XS_FORMAT, "IQ4_XS", 8, 4),
        ("q3k", Q3K_FORMAT, "Q3_K", 32, 96),
    ],
)
def test_gptq_codes_reach_a_file_at_that_type(
    tmp_path, ggml_lib, name, fmt, type_name, code_byte, scale_byte
) -> None:
    """A GPTQ build writes the tensor at that GGUF type, and llama.cpp's decoder reads the stored bytes.

    This is the file-level half of the readback: the layer pass's solved codes go through the writer's encoder
    and the file's layout, and the bytes that land in the file are the ones a llama.cpp reader dequantizes.
    The step takes the default unweighted clip; weighted scale fits are checked separately.
    """
    gguf = pytest.importorskip("gguf")
    from qda_quant.gguf import file

    from .test_export import WIDE, _source

    src, calib, template, _, _ = _source(tmp_path, WIDE)
    tensor, module = "blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj"
    cfg = BuildConfig(
        model=str(src),
        formats={"default": "w4a16-asym-g32", module: fmt},
        grid="q4k",
        devices=["cpu"],
        steps=[{"gptq": {"impl": "torch"}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "libggml": str(ggml_lib),
                "tensor_types": [{"pattern": tensor, "type": type_name}],
            }
        },
        out=str(tmp_path / "out"),
    )
    out = cfg.build(log=lambda _: None)
    written = out / "model-Q4_K_M.gguf"
    got = file.Template(written)
    kind = file.type_id(type_name)
    assert got.tensors[tensor].tensor_type == kind
    raw = got.raw(tensor)
    assert len(raw) == got.rows(tensor) * file.row_bytes(kind, got.tensors[tensor].shape[0])
    fields = {"iq4nl": iq4nl_fields, "iq4xs": iq4xs_fields, "q3k": q3k_fields}[name](raw)
    assert {"iq4nl": iq4nl_records, "iq4xs": iq4xs_records, "q3k": q3k_records}[name](*fields) == raw
    values = _evidence(gguf, kind, raw)
    assert np.isfinite(values).all() and values.size == int(np.prod(got.tensors[tensor].shape))
    # the readback is a control here too: one flipped code bit and one flipped scale bit each change the decode
    for what, at in (("code", code_byte), ("scale", scale_byte)):
        corrupt = bytearray(raw)
        corrupt[at] ^= 0x01
        assert not np.array_equal(_evidence(gguf, kind, bytes(corrupt)), values), (
            f"{name}: flipping the {what} bit at byte {at} of the stored tensor left the decode equal"
        )
    # the mix never chose this type: it is the build's own override, so the template's rounding did not stand
    assert tensor not in json.loads((out / "build.json").read_text())["gguf"]["rounded_from_template"]
