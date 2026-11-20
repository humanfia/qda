"""Source codecs: decode a native-quantized checkpoint's stored weights into the build dtype.

A source codec is a plugin in ``qda.source_codecs``. It answers three questions: `matches` (are these stored
tensors one weight of my format?), `parts` (which stored tensors make up a logical weight?) and `decode` (the
weight in the requested dtype). The registry holds the built-ins and any entry point of the group, so a new
encoding arrives without an ``if`` in the loader.

Selection is per weight, not per source: a checkpoint may mix encodings. DeepSeek-V4-Flash stores its dense
weights as 128x128 FP8 blocks and its routed experts as 1x32 FP4, and Kimi K3 stores MXFP4 experts beside bf16
attention. `codec_for` picks the codec from the stored dtypes and the scale grid.

Built-ins:

* ``float``: no encoding (bf16/f16/f32). A config that merely claims ``fp8`` while the weights are bf16, as
  DeepSeek-V3's does, lands here because the stored dtypes decide.
* ``fp8_block``: ``F8_E4M3`` weights, one scale per 128x128 block. GLM's F32 ``.weight_scale_inv`` and the
  E8M0 ``.scale`` of DeepSeek-V4-Flash are both multipliers, and the grid may cover a short last block (GLM's
  576-row projections have 5 scale rows).
* ``fp4_block``: e2m1 codes, two per ``I8`` byte (low nibble first), with one E8M0 scale per 32 inputs
  (DeepSeek-V4-Flash's experts). ``F8_E4M3`` weights with one E8M0 scale per 32 inputs use the same codec
  (DeepSeek-V4.1-Flash's dense 32x32 blocks): only the value decoding differs.
* ``mxfp4``: compressed-tensors MXFP4, ``.weight_packed`` U8 with two e2m1 codes per byte and a ``.weight_scale``
  U8 exponent per 32 inputs (Kimi K3).
* ``nvfp4``: compressed-tensors NVFP4, ``.weight_packed`` U8 with two e2m1 codes per byte, an E4M3
  ``.weight_scale`` per 16 inputs and a per-module fp32 ``.weight_global_scale``.

All arithmetic is fp32 and rounded once to the requested dtype. Scales are expanded by block size and then cut
to the weight, so a short last block works. Decoding runs wherever the stored tensors are, normally the card.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import torch

from ..registry import Registry
from ..step import Refused

__all__ = ["SOURCE_CODECS", "SourceCodec", "codec_for"]

#: Safetensors header entry of one stored tensor: its dtype name and its shape.
Header = tuple[str, list[int]]
#: Stored tensor name -> header.
Headers = dict[str, Header]

#: e2m1 values by code: the sign bit then a 3-bit exponent/mantissa field (0, 0.5, 1, 1.5, 2, 3, 4, 6).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})


@runtime_checkable
class SourceCodec(Protocol):
    """One stored encoding of weights, decoded to a build dtype."""

    name: ClassVar[str]
    """Registry name, the value of ``source_codec`` in a config."""

    def matches(self, config: dict, headers: Headers) -> bool:
        """Whether `headers` (the stored parts of one weight) are a complete weight of this format."""
        ...

    def parts(self, tensor: str) -> tuple[str, ...]:
        """Stored tensor names that make up logical weight `tensor`; the first is the weight itself."""
        ...

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        """The weight `parts` store, in `dtype`."""
        ...


def _check(key: str, obj: object) -> None:
    if not isinstance(obj, SourceCodec):
        raise TypeError(
            f"source codec plugin {key!r} is {obj!r}, without matches(config, headers), parts(tensor) "
            "and decode(parts, dtype)"
        )


#: Source codecs by name: built-ins and the ``qda.source_codecs`` entry points.
SOURCE_CODECS: Registry[SourceCodec] = Registry(
    "source_codecs",
    {
        "float": "qda_quant.intake.source:FLOAT",
        "fp8_block": "qda_quant.intake.source:FP8_BLOCK",
        "fp4_block": "qda_quant.intake.source:FP4_BLOCK",
        "mxfp4": "qda_quant.intake.source:MXFP4",
        "nvfp4": "qda_quant.intake.source:NVFP4",
    },
    on_add=_check,
)


def _stem(tensor: str) -> str:
    """The module name of a logical weight name, without ``.weight``.

    Raises:
        Refused: `tensor` does not name a logical weight (it does not end in ``.weight``), so no codec looks
            for its parts.
    """
    if not tensor.endswith(".weight"):
        raise Refused(
            f"{tensor!r}: not a logical weight name (it must end in '.weight'); no source codec looks for it"
        )
    return tensor[: -len(".weight")]


def _dtype(header: Header) -> str:
    return header[0].upper()


def _weights(headers: Headers, suffix: str = ".weight") -> list[tuple[str, Header]]:
    return [(n, h) for n, h in headers.items() if n.endswith(suffix)]


def _scale(headers: Headers, weight: str) -> tuple[str, Header] | None:
    """The scale stored beside `weight`: ``.weight_scale_inv`` (GLM) first, then ``.scale`` (DeepSeek)."""
    stem = _stem(weight)
    for suffix in (".weight_scale_inv", ".scale"):
        if stem + suffix in headers:
            return stem + suffix, headers[stem + suffix]
    return None


def _ceil_grid(shape: list[int], block: tuple[int, int]) -> tuple[int, int]:
    rows, cols = shape[-2:]
    return -(-rows // block[0]), -(-cols // block[1])


def _scale_f32(scale: torch.Tensor) -> torch.Tensor:
    """A stored scale as fp32. A uint8 scale is an E8M0 exponent with bias 127; E8M0 and F32 cast directly."""
    if scale.dtype == torch.uint8:
        return (scale.to(torch.int32) - 127).to(torch.float32).exp2()
    return scale.to(torch.float32)


def _e2m1(codes: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    return lut[codes.long()]


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Two e2m1 codes per byte, low nibble first, to fp32 ``[rows, 2 * columns]``."""
    u8 = packed.view(torch.uint8)
    low = u8 & 0x0F
    high = (u8 >> 4) & 0x0F
    return _e2m1(torch.stack((low, high), dim=-1).flatten(-2))


def _apply_blocks(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    """fp32 ``weight * scale`` with each scale covering one ``block``; a short last block reuses its scale."""
    rows, cols = weight.shape
    scale_rows, scale_cols = scale.shape
    if scale_rows * block[0] == rows and scale_cols * block[1] == cols:
        shaped = weight.reshape(scale_rows, block[0], scale_cols, block[1])
        return (shaped * scale[:, None, :, None]).reshape(rows, cols)
    full = scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    if full.shape[0] < rows or full.shape[1] < cols:
        raise ValueError(f"scale grid {tuple(scale.shape)} does not cover weight {tuple(weight.shape)}")
    return weight * full[:rows, :cols]


class Float(SourceCodec):
    """Unencoded weights: bf16, f16 or f32."""

    name: ClassVar[str] = "float"

    def matches(self, config: dict, headers: Headers) -> bool:  # noqa: ARG002
        weights = _weights(headers)
        return (
            bool(weights)
            and len(weights) == len(headers)
            and all(_dtype(h) in _FLOAT_DTYPES for _, h in weights)
        )

    def parts(self, tensor: str) -> tuple[str, ...]:
        return (tensor,)

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        (weight,) = parts.values()
        return weight.to(dtype)


class Fp8Block(SourceCodec):
    """``F8_E4M3`` weights with one multiplicative scale per 128x128 block, stored F32 or E8M0."""

    name: ClassVar[str] = "fp8_block"
    block: ClassVar[tuple[int, int]] = (128, 128)

    def matches(self, config: dict, headers: Headers) -> bool:
        block = config.get("quantization_config", {}).get("weight_block_size") or list(self.block)
        if tuple(block) != self.block:
            return False
        weights = _weights(headers)
        if not weights:
            return False
        for name, header in weights:
            if _dtype(header) != "F8_E4M3":
                return False
            scale = _scale(headers, name)
            if scale is None or _dtype(scale[1]) not in {"F32", "F8_E8M0"}:
                return False
            if _ceil_grid(header[1], self.block) != tuple(scale[1][1][-2:]):
                return False
        return True

    def parts(self, tensor: str) -> tuple[str, ...]:
        stem = _stem(tensor)
        return (tensor, stem + ".weight_scale_inv", stem + ".scale")

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        weight_name = next(n for n in parts if n.endswith(".weight"))
        scale_name = next(n for n in parts if n != weight_name)
        weight = parts[weight_name].to(torch.float32)
        scale = _scale_f32(parts[scale_name])
        return _apply_blocks(weight, scale, self.block).to(dtype)


class Fp4Block(SourceCodec):
    """e2m1 codes (two per I8 byte) or F8 weights, with one E8M0 scale per 32 inputs."""

    name: ClassVar[str] = "fp4_block"
    group: ClassVar[int] = 32

    def _layout(self, header: Header, scale: Header) -> tuple[int, int] | None:
        """(logical columns, block rows) if `header` and `scale` are this codec's, else None."""
        rows, stored_cols = header[1][-2:]
        if _dtype(header) == "I8":
            cols, block_rows = 2 * stored_cols, 1
        elif _dtype(header) in {"F8_E4M3", "F8_E5M2"}:
            cols, block_rows = stored_cols, self.group
        else:
            return None
        if _dtype(scale) != "F8_E8M0" or len(scale[1]) < 2:
            return None
        scale_rows, scale_cols = scale[1][-2:]
        if scale_cols != -(-cols // self.group):
            return None
        if scale_rows not in (rows, -(-rows // block_rows)):
            return None
        return cols, (block_rows if scale_rows != rows else 1)

    def matches(self, config: dict, headers: Headers) -> bool:  # noqa: ARG002
        weights = _weights(headers)
        if not weights:
            return False
        for name, header in weights:
            scale = _scale(headers, name)
            if scale is None or self._layout(header, scale[1]) is None:
                return False
        return True

    def parts(self, tensor: str) -> tuple[str, ...]:
        stem = _stem(tensor)
        return (tensor, stem + ".weight_scale_inv", stem + ".scale")

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        weight_name = next(n for n in parts if n.endswith(".weight"))
        weight = parts[weight_name]
        scale = parts[next(n for n in parts if n != weight_name)]
        rows = weight.shape[0]
        if weight.dtype in (torch.int8, torch.uint8):
            weight, block_rows = _unpack_fp4(weight), 1
        else:
            weight, block_rows = weight.to(torch.float32), self.group
        block_rows = 1 if scale.shape[0] == rows else block_rows
        return _apply_blocks(weight, _scale_f32(scale), (block_rows, self.group)).to(dtype)


class Mxfp4(SourceCodec):
    """compressed-tensors MXFP4: ``.weight_packed`` U8 codes and a ``.weight_scale`` U8 exponent per 32."""

    name: ClassVar[str] = "mxfp4"
    group: ClassVar[int] = 32

    def matches(self, config: dict, headers: Headers) -> bool:  # noqa: ARG002
        weights = _weights(headers, ".weight_packed")
        if not weights:
            return False
        for name, header in weights:
            stem = name[: -len(".weight_packed")]
            scale = headers.get(stem + ".weight_scale")
            if _dtype(header) != "U8" or scale is None or _dtype(scale) != "U8":
                return False
            rows, packed_cols = header[1][-2:]
            scale_rows, scale_cols = scale[1][-2:]
            if scale_cols != -(-2 * packed_cols // self.group):
                return False
            if scale_rows not in (rows, -(-rows // self.group)):
                return False
        return True

    def parts(self, tensor: str) -> tuple[str, ...]:
        stem = _stem(tensor)
        return (stem + ".weight_packed", stem + ".weight_scale")

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        packed_name = next(n for n in parts if n.endswith(".weight_packed"))
        packed = parts[packed_name]
        scale = parts[packed_name.removesuffix(".weight_packed") + ".weight_scale"]
        rows = packed.shape[0]
        block_rows = 1 if scale.shape[0] == rows else self.group
        return _apply_blocks(_unpack_fp4(packed), _scale_f32(scale), (block_rows, self.group)).to(dtype)


class Nvfp4(SourceCodec):
    """compressed-tensors NVFP4: ``.weight_packed`` U8 E2M1 codes, an E4M3 scale per 16 and a global scale.

    The checkpoint stores ``e2m1(code) * weight_scale / weight_global_scale``; the scale is one E4M3 value per
    16 input columns and ``weight_global_scale`` is one fp32 per stored module (per expert of a bank).
    """

    name: ClassVar[str] = "nvfp4"
    group: ClassVar[int] = 16

    def matches(self, config: dict, headers: Headers) -> bool:  # noqa: ARG002
        weights = _weights(headers, ".weight_packed")
        if not weights:
            return False
        for name, header in weights:
            stem = name[: -len(".weight_packed")]
            scale = headers.get(stem + ".weight_scale")
            gscale = headers.get(stem + ".weight_global_scale")
            if _dtype(header) != "U8" or scale is None or _dtype(scale) != "F8_E4M3":
                return False
            if gscale is None or _dtype(gscale) != "F32":
                return False
            rows, packed_cols = header[1][-2:]
            if scale[1][-2:] != [rows, -(-2 * packed_cols // self.group)]:
                return False
        return True

    def parts(self, tensor: str) -> tuple[str, ...]:
        stem = _stem(tensor)
        return (stem + ".weight_packed", stem + ".weight_scale", stem + ".weight_global_scale")

    def decode(self, parts: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        packed_name = next(n for n in parts if n.endswith(".weight_packed"))
        stem = packed_name.removesuffix(".weight_packed")
        gscale = parts[stem + ".weight_global_scale"].to(torch.float32).reshape(-1)
        if gscale.numel() != 1 or not bool((gscale > 0).all()):
            raise Refused(f"{stem}.weight_global_scale: expected one positive value, got {gscale.numel()}")
        scale = _scale_f32(parts[stem + ".weight_scale"]) / gscale
        return _apply_blocks(_unpack_fp4(parts[packed_name]), scale, (1, self.group)).to(dtype)


FLOAT = Float()
FP8_BLOCK = Fp8Block()
FP4_BLOCK = Fp4Block()
MXFP4 = Mxfp4()
NVFP4 = Nvfp4()


def codec_for(config: dict, headers: Headers, tensor: str) -> SourceCodec:
    """The codec that decodes logical weight `tensor` of the source described by `config` and `headers`.

    Args:
        config: the source's ``config.json``.
        headers: every stored tensor of the source: name -> (safetensors dtype name, shape).
        tensor: the logical weight name, e.g. ``model.layers.0.self_attn.q_proj.weight``.

    Returns:
        The one codec whose `matches` holds for the stored parts of `tensor`.

    Raises:
        Refused: no codec decodes this weight, or more than one does.
    """
    found = []
    for name in SOURCE_CODECS:
        codec = SOURCE_CODECS[name]
        parts = tuple(p for p in codec.parts(tensor) if p in headers)
        if parts and codec.matches(config, {p: headers[p] for p in parts}):
            found.append(codec)
    if len(found) == 1:
        return found[0]
    quant = config.get("quantization_config") or {}
    where = f"quant_method {quant.get('quant_method')!r}" if quant else "no quantization_config"
    if not found:
        stored = ", ".join(f"{p}: {headers[p][0]}" for p in _stored_parts(headers, tensor))
        raise Refused(f"{tensor}: no source codec decodes {stored or 'nothing stored'} ({where})")
    raise Refused(f"{tensor}: more than one source codec decodes it: {[c.name for c in found]} ({where})")


def _stored_parts(headers: Headers, tensor: str) -> list[str]:
    """Every stored name any codec would look for beside logical weight `tensor`, that exists."""
    stem = _stem(tensor)
    candidates = (tensor, stem + ".weight_scale_inv", stem + ".scale", stem + ".weight_packed")
    return [p for p in candidates if p in headers]
