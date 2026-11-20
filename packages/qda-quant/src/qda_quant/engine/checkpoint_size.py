"""Estimate the tensor and file sizes of a planned checkpoint."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from ..arch import ModelGraph
from ..format import Format
from ..intake.source import codec_for
from ..intake.source_names import SourceNames
from ..output.export import _DT, SMALL_FILES
from ..step import QuantStep, Step

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .packed import Packed

#: Bytes the safetensors header and the index spend on one tensor besides its data: its name, dtype, shape and
#: offsets, written once in a shard's header and once in ``model.safetensors.index.json``.
_TENSOR_OVERHEAD = 190


def checkpoint_bytes(
    source: str,
    meta: PreTrainedModel,
    graph: Any,
    fmt: Format,
    steps: Any,
    dtype: torch.dtype,
    head_fmt: Format | None = None,
    formats: dict[str, Format] | None = None,
) -> int:
    """Bytes the build's checkpoint takes, from the source's headers and the plan.

    Args:
        source: the float checkpoint the build reads.
        meta: the plan's meta skeleton (its parameter shapes and the source names it loads from).
        graph: the model graph, whose layers give the units a layer step quantizes.
        fmt: the body format.
        steps: the build's non-checkpoint steps.
        dtype: the build dtype, which a decoded source weight is written in.
        head_fmt: the format a model step stores the head in (`steps.head`), if one runs.
        formats: unit key -> the format a ``format_overrides`` rule gives it.
    """
    from ..arch.adapters import export_layout

    esz = torch.finfo(dtype).bits // 8
    quant = _quantized(meta, graph, fmt, steps, head_fmt, formats)
    names = _local_names(source, meta)
    if names is None:
        params = list(meta.parameters())
        return (
            sum(p.numel() for p in params) * esz
            - sum(math.prod(shape) * esz for _, shape, _ in quant)
            + _packed_bytes(quant)
            + _TENSOR_OVERHEAD * (len(params) + sum(_tensors_per_weight(f) for _, _, f in quant))
            + _small(source)
        )
    written = _packed_bytes(quant)
    preserved = {k for k in (names.tensor(n) for n in graph.extra.get("preserved_tensors", ())) if k}
    quantized: set[str] = set()  # source weights the build packs
    replaced: set[str] = set()  # source tensors superseded by a packed or decoded weight
    tensors = 0
    for name, shape, f in quant:
        source_name = names.tensor(name) if len(shape) == 3 else names.module(name)
        layout = (
            export_layout(graph, source_name or name, torch.Size(shape))
            if name not in names.converted
            else None
        )
        if layout is not None:
            replaced.update(layout.parts)
            tensors += sum(
                len(fields) if fields is not None else _tensors_per_weight(f)
                for _, _, fields in layout.pieces
            )
            continue
        keys = _source_keys(names, name, shape)
        tensors += _tensors_per_weight(f) * max(1, len(keys))
        for key in keys:
            quantized.add(key)
            replaced |= _stored_parts(names, key, preserved)
    for mod, m in _covered(meta, graph):
        for leaf, p in m.named_parameters():
            key = names.tensor(f"{mod}.{leaf}")
            if key is None or key in quantized or key in replaced:
                continue
            parts = _stored_parts(names, key, preserved)
            # an encoded weight the loader decoded: the writer stores it in the build dtype
            if len(parts) > 1:
                replaced |= parts
                written += p.numel() * esz
                tensors += 1
    copied: list[str] = []
    for key in names.meta:
        if key in replaced:
            continue
        parts = _stored_parts(names, key, preserved)
        if len(parts) > 1:  # a weight no step touched that the source encodes: the writer decodes it too
            replaced |= parts
            written += math.prod(names.meta[key][1]) * esz
            tensors += 1
        else:
            copied.append(key)
    return (
        sum(_tensor_bytes(*names.meta[k]) for k in copied)
        + written
        + _TENSOR_OVERHEAD * (len(copied) + tensors)
        + _small(source)
    )


def _quantized(
    meta: nn.Module,
    graph: Any,
    fmt: Format,
    steps: Any,
    head_fmt: Format | None,
    formats: dict[str, Format] | None = None,
) -> list[tuple[str, tuple[int, ...], Format]]:
    """(weight name, its shape, the format it is packed in) for every weight the build quantizes."""
    from ..solve.units import stored_format, units_of

    out = []
    if any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps):
        for lg in graph.layers:
            for u in units_of(meta.get_submodule(lg.module), lg):
                if isinstance(u.names, str):
                    names = [u.names]
                elif u.part is None:
                    names = list(u.names)
                else:
                    names = [n for per_expert in u.names for n in per_expert]
                f = stored_format(formats, u, fmt)
                out += [(n, _shape(meta, n), f) for n in names]
    if head_fmt is not None and graph.lm_head:
        out.append((graph.lm_head, _shape(meta, graph.lm_head), head_fmt))
    return out


def packed_weights(
    meta: nn.Module,
    graph: Any,
    fmt: Format,
    steps: Any,
    formats: dict[str, Format] | None = None,
) -> tuple[tuple[tuple[int, int], Format], ...]:
    """(rows, cols) and format of every tensor a layer step packs, one entry per tensor stored.

    A dense group stores one tensor per module. A routed bank stores one per expert and projection, as
    `per_expert_layout` splits a fused one: gate/up are two tensors, a non-gated up or a down projection one.
    The shapes are canonical (``[out, in]``), so a transposed fused parameter is priced as the writer stores
    it. The head is no decoder Linear and has no entry; neither has a Linear a step leaves float.
    """
    from ..solve.units import units_of

    out: list[tuple[tuple[int, int], Format]] = []
    if not any(isinstance(s, QuantStep) and s.kind == "layer" for s in steps):
        return ()
    for lg in graph.layers:
        for u in units_of(meta.get_submodule(lg.module), lg):
            f = (formats or {}).get(u.key, fmt)
            if u.part is None:  # a dense group: each module is its own stored tensor
                out += [((p.shape[0], p.shape[1]), f) for p in u.params]
                continue
            _, rows, cols = u.weight().shape  # canonical [experts, out, in], whatever the parameter stores
            projections = 2 if isinstance(u.names, str) and u.names.endswith(".gate_up_proj") else 1
            if not isinstance(u.names, str):  # per-expert modules: gate/up are two tensors, down one
                projections = len(u.rows)
            out += [((rows // projections, cols), f)] * (u.n_experts() * projections)
    return tuple(out)


def _covered(meta: nn.Module, graph: Any) -> list[tuple[str, nn.Module]]:
    """(name prefix, module) of every module whose tensors the writer takes off the model, not from the source."""
    mods = [lg.module for lg in graph.layers]
    mods += [m for m in (graph.embed, graph.final_norm, graph.lm_head) if m]
    return [(m, meta.get_submodule(m)) for m in dict.fromkeys(mods)]


def _shape(meta: nn.Module, name: str) -> tuple[int, ...]:
    """The stored shape of a unit's weight: a fused bank's 3-D parameter, else the module's 2-D weight."""
    try:
        return tuple(meta.get_parameter(name).shape)
    except AttributeError:
        w = cast("torch.Tensor", meta.get_submodule(name).weight)
        return tuple(w.shape)


def _source_keys(names: SourceNames, name: str, shape: tuple[int, ...]) -> list[str]:
    """The source tensors the writer reads a quantized weight from."""
    if len(shape) == 2:
        key = names.tensor(name + ".weight")
        return [key] if key else []
    return [key for key, _ in names.stored(name, torch.Size(shape))]


def _stored_parts(names: SourceNames, key: str, preserved: set[str]) -> set[str]:
    """The stored parts of one logical weight: its codec's pieces when the source encodes it.

    A tensor the graph preserves (`graph.extra["preserved_tensors"]`, DeepSeek-V4.1's Engram lookup tables) is
    kept as stored whatever its dtype, as ``export.Writer`` does.
    """
    if key in preserved or not key.endswith(".weight"):
        return {key}
    return {p for p in codec_for(names.config, names.meta, key).parts(key) if p in names.meta}


def _packed_bytes(quant: list[tuple[str, tuple[int, ...], Format]]) -> int:
    """Bytes of every packed weight."""
    return sum(_packed(shape, f) for _, shape, f in quant)


def _tensors_per_weight(f: Format) -> int:
    """Stored tensors `export.pack_module` writes per module: codes, scale, and shape, zero points or globals.

    An FP8 weight is stored as itself plus its per-channel scale, with no shape metadata.
    """
    if f.weight_type == "e2m1":
        return 4 if f.act == "nvfp4_block" else 3  # packed, scale, weight_global_scale [, input_global_scale]
    if f.weight_type == "fp8":
        return 2
    return 3 if f.symmetric else 4


def _packed(shape: tuple[int, ...], f: Format) -> int:
    """Bytes `export.pack_module` writes for one weight of `shape`: codes, scales, zero points, shape metadata."""
    if f.weight_type == "e2m1":
        return (
            packed_bits(shape, f) // 8
        )  # NVFP4 stores no weight_shape: its global scales are in packed_bits
    if f.weight_type == "fp8":
        return packed_bits(shape, f) // 8  # the E4M3 weight and its fp32 per-channel scale
    return packed_bits(shape, f) // 8 + 16 * math.prod(
        shape[:-2]
    )  # weight_shape: int64 [2] per stored module


def packed_bits(shape: tuple[int, ...], f: Format) -> int:
    """Bits `export.pack_module` stores for one weight of `shape`: its codes, scales and zero points.

    Codes and zero points are packed along one axis in 32-bit words, one word per 32 codes or part of one, and
    a fused bank packs each expert's slice on its own. Scales are stored in 16 bits at the build's grid; NVFP4
    stores one 8-bit E4M3 scale per 16 weights and two fp32 global scales a module. FP8 weights are one byte a
    weight under an fp32 scale per output channel, and no shape metadata is stored. The shape metadata is not
    counted: a checkpoint's budget counts the stored parts of its weights.
    """
    *lead, rows, cols = shape
    if f.weight_type == "fp8":
        return math.prod(lead) * rows * cols * 8 + math.prod(lead) * rows * 32
    n = math.prod(lead)
    groups = 1 if f.group_size == -1 else -(-cols // f.group_size)
    if f.weight_type == "e2m1":
        return n * (rows * cols * 4 + rows * groups * 8 + 64)
    words = -(-cols * f.weight_bits // 32)
    codes = n * rows * words * 32
    scales = n * rows * groups * 16
    zeros = 0 if f.symmetric else n * (-(-rows * f.weight_bits // 32)) * groups * 32
    return codes + scales + zeros


def _tensor_bytes(dtype: str, shape: list[int]) -> int:
    """Bytes of one stored source tensor."""
    return _DT[dtype].itemsize * math.prod(shape)


def _local_names(source: str, meta: PreTrainedModel) -> SourceNames | None:
    """The source's stored tensors and name map; None when its shards are not all local.

    A partly downloaded snapshot (an interrupted or concurrent download) holds some shards of its index: its
    size is then estimated like a remote source's instead of reading headers of shards that are not there.
    """
    src = Path(source)
    index = src / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {f.name for f in src.glob("*.safetensors")}
    if not shards or not all((src / f).is_file() for f in shards):
        return None
    return SourceNames(source, model=meta)


def _small(source: str) -> int:
    """The files the writer copies or rewrites beside the shards: tokenizer, config, index."""
    src = Path(source)
    files = [src / f for f in (*SMALL_FILES, "config.json", "model.safetensors.index.json")]
    return sum(f.stat().st_size for f in files if f.is_file())


@runtime_checkable
class _OwnModelFormat(Protocol):
    format: str
    embed: bool


def _output_bytes(
    source: str,
    packed: Packed | None,
    meta: PreTrainedModel,
    graph: ModelGraph,
    fmt: Format,
    steps: Sequence[Step],
    dtype: torch.dtype,
    formats: dict[str, Format],
) -> int:
    if packed is not None:
        return 0
    head_fmt = next(
        (
            Format.parse(s.format)
            for s in steps
            if isinstance(s, QuantStep) and isinstance(s, _OwnModelFormat) and s.kind == "model"
        ),
        None,
    )
    return checkpoint_bytes(source, meta, graph, fmt, steps, dtype, head_fmt, formats=formats)
