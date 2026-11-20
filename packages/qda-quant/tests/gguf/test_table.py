"""The header-only tensor table: the names, shapes, ggml types and offsets a GGUF states about itself."""

from __future__ import annotations

import struct

import pytest
from qda_quant.gguf.file import StoredTensor, read_table, tensor_bytes
from qda_quant.gguf.rule import F32, Q4_K, Q6_K

_U32 = 4  #: ``GGUF_TYPE_UINT32``, the type of ``general.alignment``


def _text(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _write(path, tensors: list[tuple[str, tuple[int, ...], int, bytes]], alignment: int = 32):
    """A little-endian GGUF: one ``general.alignment`` KV pair, then every tensor at that alignment."""
    kv = _text("general.alignment") + struct.pack("<II", _U32, alignment)
    infos, bodies, offset = bytearray(), [], 0
    for name, shape, kind, data in tensors:
        infos += _text(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        infos += struct.pack("<IQ", kind, offset)
        bodies.append(data + b"\0" * (-len(data) % alignment))
        offset += len(bodies[-1])
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv + bytes(infos)
    path.write_bytes(head + b"\0" * (-len(head) % alignment) + b"".join(bodies))
    return path


def test_a_table_is_the_headers_names_shapes_types_and_offsets(tmp_path):
    """Every tensor's stated layout, with offsets relative to the alignment-padded end of the header."""
    p = _write(
        tmp_path / "m.gguf",
        [("a.weight", (256, 2), Q4_K, b"\xa5" * 288), ("b.weight", (4,), F32, b"\x5a" * 16)],
        alignment=64,
    )
    table = read_table(p)
    assert table.tensors == (
        StoredTensor("a.weight", (256, 2), Q4_K, 0),
        StoredTensor("b.weight", (4,), F32, 320),  # 288 bytes of Q4_K rows, padded to 64
    )
    assert table.data_start % 64 == 0
    assert p.read_bytes()[table.data_start : table.data_start + 288] == b"\xa5" * 288


def test_a_stored_tensors_bytes_follow_its_type_and_shape(tmp_path):
    """A tensor sizes from its row width and row count; a type or shape this module cannot lay out is refused."""
    p = _write(tmp_path / "m.gguf", [("q", (256, 2), Q4_K, bytes(288)), ("k", (256,), Q6_K, bytes(210))])
    q, k = read_table(p).tensors
    assert (tensor_bytes(q), tensor_bytes(k)) == (288, 210)
    with pytest.raises(ValueError, match="ggml type 10 with shape .* is not one this module reads"):
        tensor_bytes(StoredTensor("x", (256,), 10, 0))  # Q2_K: stored, but not a layout this module sizes
    with pytest.raises(ValueError, match="is not one this module reads"):
        tensor_bytes(StoredTensor("x", (), Q4_K, 0))


def test_a_header_that_cannot_be_walked_is_refused(tmp_path):
    """A bad magic, a header that ends early, an impossible name length and a bad alignment all refuse."""
    good = _write(tmp_path / "m.gguf", [("x", (256,), Q4_K, bytes(144))])
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF\x03")
    with pytest.raises(ValueError, match="not a little-endian GGUF"):
        read_table(bad)
    inside = tmp_path / "inside.gguf"
    inside.write_bytes(good.read_bytes()[:28])  # inside the KV pair
    with pytest.raises(ValueError, match="the header ends before its tensor table"):
        read_table(inside)
    past = tmp_path / "past.gguf"
    past.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + struct.pack("<Q", 1 << 40))
    with pytest.raises(ValueError, match="the header ends before its tensor table"):
        read_table(past)
    with pytest.raises(ValueError, match="general.alignment 48 is not a power of two"):
        read_table(_write(tmp_path / "align.gguf", [("x", (256,), Q4_K, bytes(144))], alignment=48))
