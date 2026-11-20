"""Read a bf16 GGUF template and write a quantized GGUF laid out byte for byte as ``llama-quantize`` writes it.

The template is ``convert_hf_to_gguf.py --outtype bf16`` of the source checkpoint: it fixes every KV pair, tensor
name, shape and the permutes llama.cpp applies. The output keeps the template's KV pairs in order with
``general.file_type`` set, and writes the tensors in llama.cpp's loader order (`rule.loader_order`), each padded to
``general.alignment``, as ``llama_model_quantize_impl`` does.
"""

from __future__ import annotations

import copy
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import rule

#: why a GGUF build cannot run without the optional ``gguf`` package, and the install line that fixes it
NEEDS_GGUF = (
    "it needs the optional 'gguf' package, which this environment cannot import "
    '(install the extra: uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]")'
)

#: bytes per block and values per block of each ggml type this module reads or writes
_LAYOUT = {
    rule.F32: (4, 1),
    rule.F16: (2, 1),
    rule.BF16: (2, 1),
    rule.Q4_K: (144, 256),
    rule.Q5_K: (176, 256),
    rule.Q6_K: (210, 256),
    rule.Q5_0: (22, 32),
    rule.Q8_0: (34, 32),
    rule.IQ4_XS: (136, 256),
    rule.Q3_K: (110, 256),
    rule.IQ4_NL: (18, 32),
    rule.IQ3_S: (110, 256),
}
_U32 = 4  #: ``GGUF_TYPE_UINT32``
_STRING, _ARRAY = 8, 9  #: ``GGUF_TYPE_STRING``, ``GGUF_TYPE_ARRAY``
#: bytes of each fixed-size ``gguf_type``, and those that are signed
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SIGNED = {1, 3, 5, 11}
#: ``GGML_QNT_VERSION``
QNT_VERSION = 2
_SET = ("general.quantization_version", "general.file_type")
_BLOCK = re.compile(r"blk\.(\d+)\.")


def type_block(ggml_type: int) -> tuple[int, int]:
    """``(bytes, values)`` of one block of a ggml type: the writer's own table, else upstream gguf's.

    ``gguf``'s table covers every quantized type llama.cpp can write, which is what a decoder reads; the writer
    keeps `_LAYOUT`, whose types it can encode and whose rows it can lay out.
    """
    if ggml_type in _LAYOUT:
        return _LAYOUT[ggml_type]
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    for t in GGMLQuantizationType:
        if int(t) == ggml_type and t in GGML_QUANT_SIZES:
            block, size = GGML_QUANT_SIZES[t]
            return size, block
    raise ValueError(f"ggml type {ggml_type} is not one this build reads")


def row_bytes(ggml_type: int, width: int) -> int:
    """Bytes of one row of `width` values."""
    size, block = type_block(ggml_type)
    if width % block:
        raise ValueError(f"rows of {width} values need a multiple of {block}")
    return width // block * size


def type_id(name: str) -> int:
    """A supported ggml tensor type by its upstream name; refuse types this writer cannot encode.

    A config's ``tensor_types`` rules call this while it is read, so a missing ``gguf`` is a refusal too.
    """
    try:
        from gguf import GGMLQuantizationType
    except ImportError:
        raise ValueError(f"type {name!r}: {NEEDS_GGUF}") from None

    supported = {t.name: int(t) for t in GGMLQuantizationType if int(t) in _LAYOUT}
    if name not in supported:
        raise ValueError(f"unknown or unsupported ggml type {name!r}; choose from {sorted(supported)}")
    return supported[name]


def type_name(ggml_type: int) -> str:
    """The upstream name of a ggml type, for a message or a record; the number itself if it has no name."""
    from gguf import GGMLQuantizationType

    return next((t.name for t in GGMLQuantizationType if int(t) == ggml_type), str(ggml_type))


def _str_at(data: np.ndarray, at: int) -> tuple[str, int]:
    """The GGUF string at `at`, and where it ends.

    Raises:
        IndexError: `at` is past the file: a header that claims more than the file holds.
    """
    if at < 0 or at + 8 > data.nbytes:
        raise IndexError(f"a string at {at} starts past the file's {data.nbytes} bytes")
    n = int.from_bytes(data[at : at + 8], "little")
    return bytes(data[at + 8 : at + 8 + n]).decode(), at + 8 + n


def _skip_at(data: np.ndarray, kind: int, at: int) -> int:
    """Where the value of GGUF type `kind` at `at` ends."""
    if kind == _STRING:
        return _str_at(data, at)[1]
    if kind != _ARRAY:
        return at + _SCALAR[kind]
    item, n = struct.unpack_from("<IQ", data, at)
    at += 12
    if item in _SCALAR:
        return at + n * _SCALAR[item]
    for _ in range(n):  # an array of strings (or of arrays): the only walk here that is per element
        at = _skip_at(data, item, at)
    return at


@dataclass(frozen=True)
class StoredTensor:
    """One tensor a GGUF's header states: its name, shape, ggml type and data offset."""

    name: str
    shape: tuple[int, ...]
    tensor_type: int
    offset: int


@dataclass(frozen=True)
class Table:
    """A GGUF's tensor table, read from its header alone.

    Attributes:
        tensors: every tensor, in file order.
        data_start: where the first tensor's data begins (after the header and its alignment padding).
    """

    tensors: tuple[StoredTensor, ...]
    data_start: int


def _walk_header(
    data: np.ndarray, n_tensors: int, n_kv: int, table_path: str | Path | None
) -> tuple[dict[str, tuple[int, int, int, int]], list[StoredTensor], int]:
    """Read KV spans and tensor declarations; a table path keeps admission's alignment validation."""
    at = struct.calcsize("<4sIQQ")
    kv = {}
    for _ in range(n_kv):
        start = at
        key, at = _str_at(data, at)
        kind = int.from_bytes(data[at : at + 4], "little")
        at += 4
        end = _skip_at(data, kind, at)
        if table_path is not None and key == "general.alignment":
            alignment = int.from_bytes(data[at:end], "little")
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(f"{table_path}: general.alignment {alignment} is not a power of two")
        kv[key] = (start, at, kind, end)
        at = end
    tensors = []
    for _ in range(n_tensors):
        name, at = _str_at(data, at)
        n_dims = int.from_bytes(data[at : at + 4], "little")
        ne = struct.unpack_from(f"<{n_dims}Q", data, at + 4)
        at += 4 + 8 * n_dims
        kind, offset = struct.unpack_from("<IQ", data, at)
        at += 12
        tensors.append(StoredTensor(name, ne, kind, offset))
    return kv, tensors, at


def read_table(path: str | Path) -> Table:
    """Every tensor's name, shape, ggml type and offset from a GGUF's header alone.

    Unlike `Template`, this reads any ggml type, and expects the file's own layout: admission refuses a
    stored type outside the task's menu by name, and sizes only the types this module knows
    (`tensor_bytes`). The data is not read.

    Raises:
        ValueError: not a little-endian GGUF of version 2 or later.
    """
    data = np.memmap(Path(path), mode="r", dtype=np.uint8)
    try:
        magic, version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", data, 0)
    except struct.error:
        raise ValueError(f"{path}: not a little-endian GGUF of version 2 or later") from None
    if magic != b"GGUF" or not 2 <= version < 1 << 16:
        raise ValueError(f"{path}: not a little-endian GGUF of version 2 or later")
    try:
        kv, tensors, at = _walk_header(data, n_tensors, n_kv, path)
        alignment = 32
        if "general.alignment" in kv:
            _, begin, _, end = kv["general.alignment"]
            alignment = int.from_bytes(data[begin:end], "little")
    except (struct.error, IndexError, KeyError) as e:  # a truncated header, or a scalar type it does not know
        raise ValueError(f"{path}: the header ends before its tensor table ({e})") from None
    return Table(tuple(tensors), at + _pad(at, alignment))


def tensor_bytes(t: StoredTensor) -> int:
    """The stored bytes of one tensor.

    Raises:
        ValueError: a shape this module cannot lay out, or a ggml type it does not know.
    """
    if not t.shape or t.tensor_type not in _LAYOUT:
        raise ValueError(
            f"{t.name}: ggml type {t.tensor_type} with shape {t.shape} is not one this module reads"
        )
    return row_bytes(t.tensor_type, t.shape[0]) * int(np.prod(t.shape[1:], dtype=np.int64))


@dataclass(frozen=True)
class TensorInfo:
    """One tensor of a template.

    Attributes:
        shape: ``ne``; ``ne[0]`` is the row width.
        tensor_type: the ggml type.
        data: its stored bytes, mapped from the file.
    """

    shape: tuple[int, ...]
    tensor_type: int
    data: np.ndarray


class Template:
    """A GGUF file: its KV pairs as raw bytes and its tensors.

    The writer reads a bf16 template; the decoder (`decode`) reads any type `type_block` knows.
    It walks the file's layout itself rather than through ``gguf.GGUFReader``, which makes an object per vocabulary
    entry and merge rule (6 s on Qwen3-8B, a fifth of what ``llama-quantize`` takes for the whole file), none of which
    the exporter reads: it copies the KV pairs as bytes.

    Args:
        path: the ``.gguf`` file.

    Raises:
        ValueError: not a little-endian GGUF of version 2 or later, or a tensor of a type this module does not read.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint8)
        magic, self.version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", self.data, 0)
        if magic != b"GGUF" or not 2 <= self.version < 1 << 16:
            raise ValueError(f"{self.path}: not a little-endian GGUF of version 2 or later")
        #: key -> (start of the pair, start of its value, value's type, end of the pair)
        self.kv, infos, at = _walk_header(self.data, n_tensors, n_kv, None)
        self.arch = self.string("general.architecture")
        self.n_layer = self.number(f"{self.arch}.block_count")
        self.alignment = self.number("general.alignment") if "general.alignment" in self.kv else 32
        #: KV pairs a view rewrites (`without_mtp`) instead of copying, and the ones it leaves out
        self.kv_override: dict[str, int] = {}
        self.kv_dropped: tuple[str, ...] = ()
        base = at + _pad(at, self.alignment)
        self.tensors: dict[str, TensorInfo] = {}
        for info in infos:
            name, ne, kind, offset = info.name, info.shape, info.tensor_type, info.offset
            try:
                size = row_bytes(kind, ne[0]) * int(np.prod(ne[1:], dtype=np.int64))
            except ValueError as e:
                raise ValueError(f"{self.path}: {name}: {e}") from e
            self.tensors[name] = TensorInfo(ne, kind, self.data[base + offset : base + offset + size])

    def without_mtp(self) -> Template:
        """This file without its MTP/NextN blocks: what a conversion with ``--no-mtp`` writes.

        The converter stores a head's blocks after the decoder's and says how many there are
        (``<arch>.nextn_predict_layers``); the view keeps every tensor below that block, lowers
        ``block_count`` by the head's count and leaves the count itself out, so the rest of the file reads
        exactly as it does with the head. A file without a head is returned as is.
        """
        n = self.number(f"{self.arch}.nextn_predict_layers", 0)
        if n <= 0:
            return self
        first = self.n_layer - n
        view = copy.copy(self)
        view.tensors = {
            name: t for name, t in self.tensors.items() if not (m := _BLOCK.match(name)) or int(m[1]) < first
        }
        view.n_layer = first
        view.kv_override = {f"{self.arch}.block_count": first}
        view.kv_dropped = (f"{self.arch}.nextn_predict_layers",)
        return view

    def number(self, key: str, default: int | None = None) -> int:
        """An integer KV value, or `default` when the key is absent and a default is given.

        Raises:
            KeyError: the key is absent and no default was passed (the writer's own reading is strict).
        """
        if key not in self.kv:
            if default is None:
                raise KeyError(key)
            return default
        _, at, kind, end = self.kv[key]
        return int.from_bytes(self.data[at:end], "little", signed=kind in _SIGNED)

    def string(self, key: str) -> str:
        """A string KV value."""
        return _str_at(self.data, self.kv[key][1])[0]

    def shapes(self) -> list[tuple[str, tuple[int, ...], int]]:
        """``(name, ne, ggml type)`` of every tensor, in file order; ``ne[0]`` is the row width."""
        return [(name, t.shape, t.tensor_type) for name, t in self.tensors.items()]

    def raw(self, name: str) -> bytes:
        """A tensor's stored bytes."""
        return self.tensors[name].data.tobytes()

    def rows(self, name: str) -> int:
        """How many rows of ``ne[0]`` values a tensor has."""
        return int(np.prod(self.tensors[name].shape[1:], dtype=np.int64))

    def values(self, name: str, start: int = 0, stop: int | None = None) -> torch.Tensor:
        """Rows ``start:stop`` of a bf16, f16 or f32 tensor as fp32 ``[rows, ne[0]]``.

        Safe on many threads: numpy releases the GIL here, where torch's cast would start an OpenMP team per caller.
        """
        t = self.tensors[name]
        width, kind = t.shape[0], t.tensor_type
        if kind == rule.BF16:  # bf16 is the top half of fp32, whatever the value
            bits = t.data.view("<u2").reshape(-1, width)[start:stop]
            return torch.from_numpy(np.left_shift(bits, 16, dtype="<u4").view("<f4"))
        if kind in (rule.F16, rule.F32):
            a = t.data.view("<f2" if kind == rule.F16 else "<f4").reshape(-1, width)[start:stop]
            return torch.from_numpy(a.astype(np.float32))
        raise ValueError(f"{name}: the template stores ggml type {kind}; convert with --outtype bf16")

    def kv_bytes(self, file_type: int) -> tuple[int, bytes]:
        """``(count, bytes)`` of the output's KV pairs, as ``llama_model_quantize_impl`` leaves them.

        It copies the template's pairs, drops the split keys and a view's own (`without_mtp`), then sets
        ``general.quantization_version`` and ``general.file_type``; ``gguf_set_val_u32`` removes a key and
        appends it, so those two come last. A pair a view rewrites keeps its place with the new value.
        """
        out = []
        for key, (start, _, _, end) in self.kv.items():
            if key in _SET or key.startswith("split.") or key in self.kv_dropped:
                continue
            out.append(
                _kv_u32(key, self.kv_override[key]) if key in self.kv_override else bytes(self.data[start:end])
            )
        out += [_kv_u32("general.quantization_version", QNT_VERSION), _kv_u32("general.file_type", file_type)]
        return len(out), b"".join(out)


def _kv_u32(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<II", _U32, value)


def _pad(n: int, align: int) -> int:
    return -n % align


class Layout:
    """Where every tensor of the quantized file goes, and how large the file is.

    The header and every tensor's offset follow from the template's shapes and `types` alone, so this is fixed
    before the first tensor exists -- which is why a plan can state the output's exact size without writing
    anything (`Exporter.output_bytes`). Offsets are relative to the end of the header, as GGUF defines them.

    Args:
        template: the bf16 template.
        types: ggml type per tensor (`rule.q4_k_m_types`).
    """

    def __init__(self, template: Template, types: dict[str, int]):
        self.names = rule.loader_order(list(types))
        self.written: set[str] = set()
        self.offset: dict[str, int] = {}
        self.size: dict[str, int] = {}
        self.row: dict[str, int] = {}
        infos, offset = [], 0
        for name in self.names:
            ne: tuple[int, ...] = template.tensors[name].shape
            # llama.cpp writes ggml_n_dims, which omits trailing singleton axes (e.g. a one-row gate).
            n_dims = len(ne)
            while n_dims > 1 and ne[n_dims - 1] == 1:
                n_dims -= 1
            ne = ne[:n_dims]
            self.row[name] = row_bytes(types[name], ne[0])
            size = self.row[name] * int(np.prod(ne[1:], dtype=np.int64))
            nb = name.encode()
            infos.append(
                struct.pack("<Q", len(nb))
                + nb
                + struct.pack(f"<I{len(ne)}QIQ", len(ne), *ne, types[name], offset)
            )
            self.offset[name] = offset
            self.size[name] = size
            offset += size + _pad(size, template.alignment)
        n_kv, kv = template.kv_bytes(rule.FTYPE_Q4_K_M)
        head = b"GGUF" + struct.pack("<IQQ", template.version, len(self.names), n_kv) + kv + b"".join(infos)
        self.head = head + bytes(_pad(len(head), template.alignment))
        self.total = len(self.head) + offset
        #: stored bytes of the tensors themselves, without the header or the padding between them: what a
        #: budget on a model's weights compares, where `total` is what the file costs on disk
        self.payload = sum(self.size.values())


class Sink(Layout):
    """The quantized file: each tensor goes straight to the place `Layout` gives it.

    A tensor is written the moment it is encoded instead of being held until the end.

    Args:
        path: output file (created when the sink is).
        template: the bf16 template.
        types: ggml type per tensor (`rule.q4_k_m_types`).
    """

    def __init__(self, path: str | Path, template: Template, types: dict[str, int]):
        super().__init__(template, types)
        self.path = Path(path)
        self.f = self.path.open("w+b")

    def put(self, name: str, data: bytes) -> None:
        """Write one tensor's stored bytes into its place.

        Raises:
            ValueError: the tensor is not in the layout, or `data` is the wrong size.
        """
        if name not in self.size:
            raise ValueError(f"{name}: not a tensor of this file")
        if len(data) != self.size[name]:
            raise ValueError(f"{name}: {len(data)} bytes, expected {self.size[name]}")
        self.f.seek(len(self.head) + self.offset[name])
        self.f.write(data)
        self.written.add(name)

    def put_rows(self, name: str, start: int, data: bytes) -> None:
        """Write the stored bytes of a tensor's rows from `start` on; safe on many threads (a positioned write).

        The caller marks the tensor `written` when all its rows are in.
        """
        at = start * self.row[name]
        if at + len(data) > self.size[name]:
            raise ValueError(f"{name}: {len(data)} bytes at {at}, past its {self.size[name]}")
        os.pwrite(self.f.fileno(), data, len(self.head) + self.offset[name] + at)

    def close(self) -> Path:
        """Write the header, pad the last tensor and close the file."""
        self.f.seek(0)
        self.f.write(self.head)
        self.f.truncate(self.total)
        self.f.close()
        return self.path
