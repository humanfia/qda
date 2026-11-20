"""Opt-in AWQ storage with BF16 scales for asymmetric MoE in stock SGLang.

This changes packing, not the quantizer: unsigned codes and integral zero points are
transposed into AWQ's group-major layout. BF16 scales keep every bit. Plain AWQ's
FP16-only kernels are unsuitable; serving needs a BF16 zero-point kernel.
"""

from __future__ import annotations

from typing import Any

import torch

from ..format import Format
from ..step import Refused
from .export import Writer
from .exporters import CompressedTensors
from .pack import _FUSED


def pack_awq(mod: str, qp: dict[str, Any], sl: tuple = ()) -> dict[str, torch.Tensor]:
    """Pack one unsigned INT4 matrix into AWQ fields without rounding any scale.

    Raises:
        Refused: unsupported shape/format, non-BF16 or nonfinite scale, or codes/zero points
            that are not integers in [0, 15]. Every refusal names the module.
    """
    fmt = Format.parse(qp["format"])
    if fmt.weight_bits != 4 or fmt.symmetric or fmt.act != "bf16":
        raise Refused(f"{mod}: AWQ BF16 requires asymmetric W4A16")
    codes = qp["codes"][sl] if sl else qp["codes"]
    scale = qp["scale"][sl] if sl else qp["scale"]
    if "zero_point" not in qp:
        raise Refused(f"{mod}: AWQ BF16 requires integral zero points in [0, 15]")
    zero = qp["zero_point"][sl] if sl else qp["zero_point"]
    for field, value in (("codes", codes), ("zero points", zero)):
        if not bool(torch.isfinite(value).all()) or not bool(
            ((value >= 0) & (value <= 15) & (value == value.round())).all()
        ):
            raise Refused(f"{mod}: AWQ {field} must be integers in [0, 15]")
    if scale.dtype != torch.bfloat16:
        raise Refused(f"{mod}: AWQ scales must already be BF16; scale down-casts are refused")
    if not bool(torch.isfinite(scale).all()):
        raise Refused(f"{mod}: AWQ scales must be finite")
    if (
        codes.ndim != 2
        or codes.shape[0] % 8
        or fmt.group_size <= 0
        or codes.shape[1] % fmt.group_size
        or scale.shape != (codes.shape[0], codes.shape[1] // fmt.group_size)
        or zero.shape != scale.shape
    ):
        raise Refused(f"{mod}: AWQ needs an [out, in] matrix, out divisible by 8 and whole scale groups")

    def pack(value: torch.Tensor) -> torch.Tensor:
        lanes = value.T.contiguous().to(torch.int64).reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        shifts = 4 * torch.arange(8, device=value.device)
        return (lanes << shifts).sum(-1).reshape(value.shape[1], -1).to(torch.int32)

    return {f"{mod}.qweight": pack(codes), f"{mod}.qzeros": pack(zero), f"{mod}.scales": scale.T.contiguous()}


class AWQBF16(CompressedTensors):
    """AWQ INT4/BF16 layout for SGLang asymmetric MoE; selected explicitly as exporter: awq_bf16."""

    feeds_stages = False
    supports_formats = False

    def refusal(self, plan, recipe):  # noqa: ANN001, ANN201
        """Refuse combinations whose one model-wide AWQ scheme cannot represent the plan."""
        if why := super().refusal(plan, recipe):
            return why
        fmt = plan.fmt
        if (fmt.weight_bits, fmt.symmetric, fmt.act) != (4, False, "bf16") or fmt.expert_format.symmetric:
            return "requires explicitly asymmetric W4A16 with BF16 scales"
        if fmt.group_size not in (64, 128):
            return "requires group size 64 or 128"
        if plan.formats:
            return "a model-wide AWQ scheme cannot represent format_overrides"
        if plan.graph.extra.get("expert_biases"):
            return "biased expert banks are unsupported by stock SGLang AWQ"
        return None

    def writer(self, out_dir, source, graph, fmt, target, *, dtype=torch.bfloat16):  # noqa: ANN001, ANN201
        """The normal streaming writer with AWQ packing/config; no FP16 scale conversion."""
        if dtype != torch.bfloat16:
            raise Refused("awq_bf16: build dtype must be BF16; FP16 scale down-casts are refused")
        return AWQWriter(out_dir, source, graph, fmt, target, dtype=dtype)


class AWQWriter(Writer):
    """Reuse source accounting, architecture layouts, float copies and atomic build publication."""

    def _pack_module(self, mod, qp, sl=(), names=None, card=None):  # noqa: ANN001, ANN201, ARG002
        """Pack the adapter's per-matrix piece directly, before CT packing could truncate invalid values."""
        if mod.endswith(_FUSED):
            raise Refused(f"{mod}: AWQ needs an adapter's per-expert matrix export layout")
        if names is not None and names != {
            field: f"{mod}.weight_{field}" for field in ("packed", "scale", "shape", "zero_point")
        }:
            raise Refused(f"{mod}: AWQ does not support this adapter's custom stored field names")
        return pack_awq(mod, qp, sl)

    def _output_config(self, recipe: Any) -> dict:
        """Write literal float skip names and adapter-declared fused aliases, not CT regex patterns."""
        from ..arch.adapters import ADAPTERS

        cfg = super()._output_config(recipe)
        ignore = {n for n in cfg["quantization_config"]["ignore"] if not n.startswith("re:")}
        # A float fused bank's CT ignore is a per-expert regex. AWQ probes the bank module itself.
        ignore.update(self._served(k.rpartition(".")[0]) for k in self.index if k.endswith(_FUSED))
        # A per-expert float bank must likewise be ignored as a whole by the AWQ MoE method.
        ignore.update(self._served(n) for n in self._float_banks_per_expert())
        for key in {self.graph.adapter, self.graph.extra.get("stream_adapter")} - {None}:
            aliases = getattr(ADAPTERS[key], "awq_skip_aliases", None)
            if aliases is not None:
                ignore.update(aliases(ignore))
        for name in self.quantized:
            if any(skip in self._served(name) for skip in ignore):
                raise Refused(f"{name}: AWQ float skip alias matches a quantized module")
        cfg["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": self.fmt.group_size,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": sorted(ignore),
        }
        for node in (cfg, cfg.get("text_config")):
            if node is not None:
                node["dtype"] = node["torch_dtype"] = "bfloat16"
        return cfg

    def serve(self) -> dict:
        """BF16 activations, leaving backend selection to the engine for this different storage layout.

        CT kernel expectations cannot be reused: stock vLLM may choose Humming for AWQ experts.
        Selected kernels must be recorded from the engine by each scoring/serving run.
        """
        return {
            "vllm": self.target.vllm,
            "env": {},
            "args": {"dtype": "bfloat16"},
            "plugins": [],
            "kernels": {},
        }
