"""Pack a qparams entry into the compressed-tensors tensors a stored module is written as.

`pack_module` is the one packer of the exporter (`qda_quant.output.export.Writer`).
"""

from __future__ import annotations

from typing import Any

import torch

from ..format import Format
from ..solve.packing import pack

_FUSED = (".experts.gate_up_proj", ".experts.down_proj")


def _pack_on(
    device: torch.device, mod: str, qp: dict[str, Any], sl: tuple, names: dict[str, str] | None = None
) -> dict[str, torch.Tensor]:
    """`pack_module` of a host entry on `device`, an expert bank one slice of about 1 GiB of codes at a time.

    Codes, scales and zero points pack per expert, so the slices' tensors concatenate to the whole entry's.
    """
    from ..solve.units import QPARAM_TENSORS  # here, not at the top: units imports the adapters

    fields = {f: qp[f][sl[:1]] if f == "global_scale" else qp[f][sl] for f in QPARAM_TENSORS if f in qp}
    codes = fields["codes"]
    step = max(1, (1 << 30) // codes[0].numel()) if mod.endswith(_FUSED) else codes.shape[0]
    parts = [
        pack_module(mod, {**qp, **{f: t[a : a + step].to(device) for f, t in fields.items()}}, names=names)
        for a in range(0, codes.shape[0], step)
    ]
    return {k: torch.cat([p[k] for p in parts]) if len(parts) > 1 else parts[0][k] for k in parts[0]}


def pack_module(
    mod: str, qp: dict[str, Any], sl: tuple = (), names: dict[str, str] | None = None
) -> dict[str, torch.Tensor]:
    """Stored tensors of one qparams entry (or its slice `sl`) under source module name `mod`.

    Codes are unsigned for asymmetric formats, signed for symmetric ones.

    Fused shape metadata keeps expert and projection axes so vLLM's fused loader distributes each
    two-value matrix shape to the corresponding expert, splitting gate/up as it splits the codes.

    `names` gives the stored key per field (``packed``, ``scale``, ``shape``, ``zero_point``) when the serving
    checkpoint does not name the tensors after `mod`. A field left out is not stored; an entry with zero points
    must name them, so no checkpoint is written with values its loader cannot read.
    """
    f = Format.parse(qp["format"])
    if f.weight_type == "e2m1":
        return _pack_nvfp4(f, mod, qp, sl, names)
    if f.weight_type == "fp8":
        return _pack_fp8(mod, qp, sl, names)
    shift = 0 if f.symmetric else -(1 << (f.weight_bits - 1))  # unsigned -> CT's signed codes
    codes = qp["codes"][sl] if sl else qp["codes"]
    b = f.weight_bits
    fused = mod.endswith(_FUSED)
    if names is None:
        if fused:
            names = {field: f"{mod}_{field}" for field in ("packed", "scale", "shape", "zero_point")}
        else:
            names = {field: f"{mod}.weight_{field}" for field in ("packed", "scale", "shape", "zero_point")}
    t = {
        names["packed"]: pack((codes.to(torch.int16) + shift).to(torch.int8), b, packed_dim=1),
        names["scale"]: qp["scale"][sl] if sl else qp["scale"],
    }
    if "shape" in names:
        shape = torch.tensor(list(codes.shape), dtype=torch.int64)
        if fused:
            projections = 2 if mod.endswith(".gate_up_proj") else 1
            shape = torch.tensor([codes.shape[1] // projections, codes.shape[2]], dtype=torch.int64)
            shape = shape.repeat(codes.shape[0], projections, 1)
        t[names["shape"]] = shape
    if "zero_point" in qp:
        if "zero_point" not in names:
            raise ValueError(f"{mod}: the entry stores zero points, and the layout has no place for them")
        z = qp["zero_point"][sl] if sl else qp["zero_point"]
        t[names["zero_point"]] = pack((z.to(torch.int16) + shift).to(torch.int8), b, packed_dim=0)
    return {k: v.contiguous() for k, v in t.items()}


def _pack_fp8(
    mod: str, qp: dict[str, Any], sl: tuple, names: dict[str, str] | None
) -> dict[str, torch.Tensor]:
    """`pack_module` of an FP8 W8A8 entry: compressed-tensors' ``float-quantized`` tensors of one Linear.

    ``weight`` holds the E4M3 values themselves and ``weight_scale`` the fp32 scale of each output channel
    (shape ``[out, 1]``); the kernel multiplies the two. The names follow the layout's ``weight`` and ``scale``
    names, which must end in ``.weight`` and ``.weight_scale`` so vLLM loads them into the parameters its FP8
    W8A8 scheme creates.

    Raises:
        ValueError: a stacked bank (no engine loads FP8 experts under one name), or a layout that names the
            weight or the scale something else.
    """
    if mod.endswith(_FUSED):
        raise ValueError(f"{mod}: FP8 experts are stored one Linear per expert, not as a stacked bank")
    weight_name = (names or {}).get("weight", f"{mod}.weight")
    scale_name = (names or {}).get("scale", f"{mod}.weight_scale")
    if not weight_name.endswith(".weight") or not scale_name.endswith(".weight_scale"):
        raise ValueError(
            f"{mod}: an FP8 layout names its tensors weight and weight_scale, not {weight_name!r} and "
            f"{scale_name!r}"
        )
    codes = qp["codes"][sl] if sl else qp["codes"]
    scale = qp["scale"][sl] if sl else qp["scale"]
    if scale.reshape(-1, 1).shape[0] != codes.reshape(-1, codes.shape[-1]).shape[0]:
        raise ValueError(f"{mod}: {scale.shape} scales for {codes.shape} weights")
    return {
        weight_name: codes.to(torch.float8_e4m3fn).contiguous(),
        scale_name: scale.reshape(-1, 1).to(torch.float32).contiguous(),
    }


def _pack_nvfp4(
    f: Format, mod: str, qp: dict[str, Any], sl: tuple, names: dict[str, str] | None
) -> dict[str, torch.Tensor]:
    """`pack_module` of an NVFP4 entry: compressed-tensors' ``nvfp4-pack-quantized`` tensors of one Linear.

    ``weight_packed`` holds two E2M1 codes a byte, ``weight_scale`` the E4M3 group scales and
    ``weight_global_scale`` the fp32 scale they are divided by, and ``input_global_scale`` the static scale the
    kernel cuts the module's input under. The names follow the layout's ``scale`` name, which must be
    ``<module>.weight_scale``.

    Raises:
        ValueError: a stacked bank (no engine loads NVFP4 experts under one name), a layout whose scale is not
            ``weight_scale``, or an entry without the global scale its grid records or the input scale the
            build records.
    """
    from ..solve.nvfp4 import codes_of_positions, pack_nibbles

    if mod.endswith(_FUSED):
        raise ValueError(f"{mod}: NVFP4 experts are stored one Linear per expert, not as a stacked bank")
    scale_name = (names or {}).get("scale", f"{mod}.weight_scale")
    if not scale_name.endswith("weight_scale"):
        raise ValueError(f"{mod}: an NVFP4 layout names its scales weight_scale, not {scale_name!r}")
    stem = scale_name[: -len("weight_scale")]
    if "global_scale" not in qp:
        raise ValueError(f"{mod}: the entry has no global scale; NVFP4 codes are recorded on the nvfp4 grid")
    codes = qp["codes"][sl] if sl else qp["codes"]
    scale = qp["scale"][sl] if sl else qp["scale"]
    # the global scale is one tensor for the modules of a fused group: each stored copy is its own, or
    # `safetensors` refuses the checkpoint for sharing memory between two names
    g = (qp["global_scale"][sl[:1]] if sl else qp["global_scale"]).reshape(1).to(torch.float32)
    t = {
        f"{stem}weight_packed": pack_nibbles(codes_of_positions(codes)),
        scale_name: (scale.to(torch.float32) * g).to(torch.float8_e4m3fn),
        f"{stem}weight_global_scale": g.clone(),
    }
    if f.act == "nvfp4_block":
        if "input_global_scale" not in qp:
            raise ValueError(f"{mod}: the entry has no input scale; the build records one per NVFP4 module")
        t[f"{stem}input_global_scale"] = qp["input_global_scale"].reshape(1).to(torch.float32).clone()
    return {k: v.contiguous() for k, v in t.items()}
