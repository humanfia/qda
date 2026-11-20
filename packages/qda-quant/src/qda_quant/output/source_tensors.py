"""Source tensors as the checkpoint stores them: the Writer's methods that read, decode and repack them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ..arch import ModelGraph
from ..format import Format
from ..intake.source import codec_for
from ..intake.source_names import SourceNames
from ..target import Target


class _SourceTensors:
    """The Writer's source-tensor methods (mixed into `qda_quant.output.export.Writer`)."""

    src: Path
    graph: ModelGraph
    fmt: Format
    target: Target
    formats: dict[str, Format]
    dtype: torch.dtype
    names: SourceNames
    where: dict[str, str]
    meta: dict[str, Any]
    config: Any
    preserved: set[str | None]
    quantized: set[str]
    decoded: set[str]
    kept_decoded: set[str]
    replaced: set[str]
    covered: set[str]
    pack_like: dict[str, str]

    def _unit_format(self, module: str) -> Format | None:
        """The ``format_overrides`` format of the dense unit holding `module`, or None for the body's."""
        for lg in self.graph.layers:
            for grp in lg.groups:
                if module in grp.modules:
                    return self.formats.get(grp.modules[0])
        return None

    def _served(self, name: str) -> str: ...  # Writer's

    def _push(self, tensors: dict[str, torch.Tensor]) -> None: ...  # Writer's

    def _pack_module(self, mod, qp, sl=(), names=None, card=None) -> dict[str, torch.Tensor]: ...  # Writer's

    def _stored(self, key: str, v: torch.Tensor) -> torch.Tensor:
        """In-memory tensor `v` as source tensor `key` is stored: its dtype, and its bytes if no step changed it.

        The model may hold a tensor in a narrower dtype than the source (Qwen3.5's float32 gated norms run in
        bf16). Casting that copy back would round the source; when `v` is the source in `v`'s dtype, the
        source's own tensor is written.

        Encoded weights are already decoded by the loader. Write them in the build dtype and retire all
        their stored parts, so a source scale never accompanies the new float weight.
        """
        parts = self._parts(key)
        if parts != (key,):
            self.replaced.update(parts)
            self.decoded.add(key.removesuffix(".weight"))
            return v.detach()
        dt = _DT[self.meta[key][0]]
        if v.dtype == dt:
            return v.detach()
        with safe_open(self.src / self.where[key], "pt") as h:
            orig = h.get_tensor(key)
        return orig if torch.equal(orig.to(v.device, v.dtype), v) else v.detach().to(dt)

    def _parts(self, key: str) -> tuple[str, ...]:
        """Existing source parts of a logical weight, or the tensor itself for norms and fused parameters."""
        if key in self.preserved or not key.endswith(".weight"):
            return (key,)
        codec = codec_for(self.config, self.meta, key)
        return tuple(part for part in codec.parts(key) if part in self.meta)

    def _source_weight(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        """Source weight `key`: its own tensor when the checkpoint stores it whole, else its codec's decode.

        An encoded weight (a native FP8 or FP4 checkpoint) is stored as several parts and decodes into `dtype`.
        """
        parts = {}
        for part in self._parts(key):
            with safe_open(self.src / self.where[part], "pt") as h:
                parts[part] = h.get_tensor(part)
        if tuple(parts) == (key,):
            return parts[key]
        return codec_for(self.config, self.meta, key).decode(parts, dtype)

    def _layout(
        self, name: str, shape: torch.Size
    ) -> tuple[list[tuple[str, tuple, dict[str, str] | None]], set[str]]:
        """Stored pieces of the qparams entry `name`, and the source tensors they supersede.

        One piece is (module, entry slice, stored key per field or None).

        The graph's adapter gives the layout when the serving checkpoint does not name the entry after its
        module; else a converter that restructures per-expert tensors does, through its inverse; otherwise
        the name map does, splitting a fused source bank into the slices it stores.
        """
        from ..arch.adapters import export_layout

        source = self.names.tensor(name) if len(shape) == 3 else self.names.module(name)
        layout = None if self.names.converter_owns(name) else export_layout(self.graph, source or name, shape)
        if layout is not None:
            return list(layout.pieces), set(layout.parts)
        weight = name + ".weight" if len(shape) == 2 else name
        pieces, parts = [], set()
        for key, sl in self.names.stored(weight, shape):
            mod = key.removesuffix(".weight")
            pieces.append((mod, sl, None))
            parts.update(self._parts(mod if mod in self.meta else mod + ".weight"))
        return pieces, parts

    @torch.no_grad()
    def _pack_source_only(self) -> None:
        """Pack each `pack_like` weight round-to-nearest (min/max ranges) in its reference module's format.

        These weights have no module in the model class, so no step solved them and no forward reads them; only
        their stored scheme matters, so it matches the fused sibling an engine builds them with. A weight whose
        reference stayed float stays float.
        """
        from ..clips import CLIPS
        from ..solve.grid import DEFAULT_GRID
        from ..solve.quant import Quantizer

        for mod, ref in sorted(self.pack_like.items()):
            key, ref_src = self.names.tensor(mod + ".weight"), self.names.module(ref)
            if key is None or key in self.covered or key in self.replaced or ref_src not in self.quantized:
                continue
            fmt = self._unit_format(ref) or self.fmt
            q = Quantizer.of(fmt, self.target.plan(fmt, "dense"), DEFAULT_GRID)
            w = self._source_weight(key, self.dtype)
            scale, zero = CLIPS["minmax"].choose(q, w.float())
            qp = {
                "codes": q.stored_codes(q.quantize(w.float(), scale, zero)),
                "scale": q.stored_scale(scale),
                "format": fmt.name,
            }
            if zero is not None:
                qp["zero_point"] = q.stored_zero(zero)
            src = key.removesuffix(".weight")
            self._push({self._served(k): v for k, v in self._pack_module(src, qp).items()})
            self.quantized.add(src)
            self.replaced.update(self._parts(key))

    def _decode_kept(self) -> list[str]:
        """Encoded source weights no step touched, written decoded.

        A build quantizes the modules of the model; a weight the model class does not hold (an MTP layer a
        draft model reloads from the same checkpoint) stays untouched. Copying it as stored would leave its
        native FP8/FP4 bytes and scales beside this checkpoint's compressed-tensors config, which reads one
        encoding per Linear, so an encoded one decodes through its codec into the build dtype. Nothing here
        knows the module's name: `_parts` asks the source codecs which weights are encoded.
        """
        out: list[str] = []
        for k in self.where:
            if k in self.covered or k in self.replaced or not k.endswith(".weight") or k not in self.meta:
                continue
            if self._parts(k) != (k,):
                out.append(k)
        for k in out:
            self.replaced.update(self._parts(k))  # its scales, which a decoded weight does not use
            self.decoded.add(k.removesuffix(".weight"))
            self.kept_decoded.add(k.removesuffix(".weight"))
        return out


_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
       "U64": torch.uint64, "U32": torch.uint32, "U16": torch.uint16, "U8": torch.uint8, "BOOL": torch.bool,
       "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu}  # fmt: skip
