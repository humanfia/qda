"""Checkpoint formats: `qda inspect` asks each registered one whether it recognises a checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from ..format import NVFP4_FORMAT, W8A8_FP8_FORMAT, Format
from ..registry import Registry
from .packed import unpacked
from .probe import IntakeRefused, Source

if TYPE_CHECKING:
    from ..target import LayerKind, Target


def _ct_group(g: dict) -> dict:
    w, a = g.get("weights") or {}, g.get("input_activations") or {}
    bits, wtype = w.get("num_bits"), w.get("type", "int")
    if a:
        act = f"a{a.get('num_bits')}-{a.get('type')}"
        scheme = f"w{bits}a8" if (wtype, a.get("type"), a.get("num_bits")) == ("int", "float", 8) else None
    else:
        act, scheme = "a16", f"w{bits}a16"
    if wtype != "int":
        scheme = f"{wtype}{bits}" + ("-block" if w.get("strategy") == "block" else "")
    return {"targets": g.get("targets") or [], "bits": bits, "type": wtype, "group_size": w.get("group_size"),
            "symmetric": w.get("symmetric"), "weight_strategy": w.get("strategy"), "act": act,
            "act_strategy": a.get("strategy"), "act_dynamic": bool(a.get("dynamic")),
            "actorder": w.get("actorder"), "scheme": scheme, "format": g.get("format")}  # fmt: skip


def int_format(g: dict) -> Format | None:
    """The format this group describes, when it is one this library implements."""
    if g["type"] != "int" or g["scheme"] is None or g["format"] not in (None, "pack-quantized"):
        return None
    act = "fp8_token" if g["act"] == "a8-float" else "bf16"
    return Format(g["bits"], g["group_size"] or -1, bool(g["symmetric"]), "int", act)


def group_format(g: dict) -> Format | None:
    """The format this config group describes, when it is one this library builds.

    ``pack-quantized`` integer groups go through `int_format`. ``float-quantized`` groups are vLLM's FP8
    W8A8 when the weights are 8-bit floats symmetric per channel and the activations dynamic 8-bit floats per
    token (``w8a8-fp8``); ``nvfp4-pack-quantized`` groups are NVFP4 when the weights are 4-bit floats in
    groups of 16 (``w4a4-nvfp4``). Any other scheme is a layout this library does not build, and `serve`
    reports it unstudied rather than guessing a kernel.
    """
    f = int_format(g)
    if f is not None:
        return f
    if g["format"] == "float-quantized":
        fp8_weights = (
            g["type"] == "float"
            and g["bits"] == 8
            and g["symmetric"]
            and g["weight_strategy"] == "channel"
            and g["group_size"] is None
        )
        fp8_acts = g["act"] == "a8-float" and g["act_strategy"] == "token" and g["act_dynamic"]
        return Format.parse(W8A8_FP8_FORMAT) if fp8_weights and fp8_acts else None
    if g["format"] == "nvfp4-pack-quantized":
        nvfp4 = g["type"] == "float" and g["bits"] == 4 and g["group_size"] == 16
        return Format.parse(NVFP4_FORMAT) if nvfp4 else None
    return None


def _claims_experts(targets: Iterable[str]) -> bool:
    """Whether a config group's targets can claim a routed expert.

    The class target ``Linear`` claims the per-expert Linears an engine builds (and the NVFP4 body group is
    written that way); a ``re:`` or exact pattern that names experts claims them directly. A shared expert
    is a dense Linear in the engines, so a name that names only a shared expert (Nemotron-H's
    ``mixer.shared_experts.up_proj``) claims none.
    """
    return any(t == "Linear" or "expert" in t.lower().replace("shared_expert", "") for t in targets)


def _dense_dtype(src: Source) -> str:
    count: dict[str, int] = {}
    for k, (dt, shape) in src.tensors.items():
        if k.endswith(".weight") and len(shape) == 2:
            count[dt] = count.get(dt, 0) + 1
    return max(count, key=lambda d: count[d]) if count else "none"


def _gguf(src: Source) -> bool:
    """A single ``.gguf`` file: it carries its own config, tokenizer and weights."""
    return not src.tensors and any(f.endswith(".gguf") for f in src.files)


def _qc(src: Source) -> dict[str, Any]:
    """The source's quantization config: ``quantization_config``, or the legacy ``compression_config``."""
    cfg = src.jsons.get("config.json") or {}
    return cfg.get("quantization_config") or cfg.get("compression_config") or {}


class CheckpointFormat:
    """One kind of checkpoint that `qda inspect` recognises: its description and what a target serves of it.

    A plugin of ``qda.checkpoint_formats``. `matches` says whether a checkpoint is this format, `describe`
    fills ``format``, ``layout``, ``groups`` and ``kept_bf16`` in the report, and `serve` appends the kernels
    the target binds, or why it cannot serve the checkpoint.

    The registered formats are asked in decreasing `priority`, ties by registry name. A format that also
    looks like another declares the higher priority: ``gguf`` outranks ``unquantized``, and ``unknown``
    matches anything, so it is asked last.

    Attributes:
        name: Registry name, the key in ``qda.checkpoint_formats``.
        priority: Higher is asked first; the default is 0 and ``unknown`` is -100.
    """

    name: ClassVar[str]
    priority: ClassVar[int] = 0

    def matches(self, src: Source) -> bool:
        """Whether `src` is a checkpoint of this format."""
        raise NotImplementedError

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """Fill the report's ``format``, ``layout``, ``groups`` and ``kept_bf16`` for `src`."""
        raise NotImplementedError

    def serve(self, src: Source, rep: dict[str, Any], target: Target) -> None:
        """Append to `rep`'s ``kernels``, ``notes`` and ``reasons`` what `target` serves of this checkpoint."""
        del src
        rep["notes"].append(
            f"{rep['format']}: not in the {target.arch} target table; servability not checked"
        )


class Gguf(CheckpointFormat):
    """A single ``.gguf`` file: it carries its own config, tokenizer and weights, and llama.cpp serves it."""

    name: ClassVar[str] = "gguf"
    priority: ClassVar[int] = 100

    def matches(self, src: Source) -> bool:
        """A file list of ``.gguf`` files and no safetensors."""
        return _gguf(src)

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """The format is ``gguf``; a GGUF file carries its own groups and layout."""
        del src
        out["format"] = "gguf"

    def serve(self, src: Source, rep: dict[str, Any], target: Target) -> None:
        """llama.cpp serves a GGUF file, not vLLM from a checkpoint directory."""
        del src, target
        rep["reasons"].append("a GGUF file is served by llama.cpp, not by vLLM from a checkpoint directory")


class Unquantized(CheckpointFormat):
    """No quantization config: the stored dtypes decide the format.

    A config that merely claims ``fp8`` while the weights are bf16 lands here because the stored dtypes decide.
    """

    name: ClassVar[str] = "unquantized"

    def matches(self, src: Source) -> bool:
        """A config with no quantization config at all."""
        return not _qc(src)

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """The format of the dtype most 2D weights are stored in; FP8 also needs a ``weight_scale_inv``."""
        dt = _dense_dtype(src)
        fp8 = dt == "F8_E4M3" and any(k.endswith("weight_scale_inv") for k in src.tensors)
        out["format"] = (
            "fp8-block" if fp8 else {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}.get(dt, dt.lower())
        )

    def serve(self, src: Source, rep: dict[str, Any], target: Target) -> None:
        """Bf16/fp16/fp32 kernels serve it as it is; any other stored dtype is outside the target table."""
        if rep["format"] not in ("bf16", "fp16", "fp32"):
            super().serve(src, rep, target)


class CompressedTensors(CheckpointFormat):
    """compressed-tensors: one config group per compressed set of tensors, and this library's own layout."""

    name: ClassVar[str] = "compressed_tensors"

    def matches(self, src: Source) -> bool:
        """``quant_method: compressed-tensors``, integer or float."""
        return _qc(src).get("quant_method") == "compressed-tensors"

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """Summarize each config group and name the layout: ``qda`` first, then legacy, llm-compressor, hf."""
        qc = _qc(src)
        ctf: str | None = qc.get("format")
        out["groups"] = {n: _ct_group(g) for n, g in (qc.get("config_groups") or {}).items()}
        for g in out["groups"].values():
            g["format"] = g["format"] or ctf
        kinds = {"nvfp4-pack-quantized": "nvfp4", "mxfp4-pack-quantized": "mxfp4"}
        schemes = sorted({str(g["scheme"]) for g in out["groups"].values()})
        out["format"] = kinds.get(ctf or "") or f"compressed-tensors {ctf} {'+'.join(schemes)}"
        out["kept_bf16"] = list(qc.get("ignore") or [])
        if isinstance((src.jsons.get("build.json") or {}).get("serve"), dict):
            out["layout"] = "qda"  # a converted row keeps its old record, so this is asked first
        elif "quant-record.json" in src.files:
            out["layout"] = "legacy"
        elif "recipe.yaml" in src.files:
            out["layout"] = "llm-compressor"

    def serve(self, src: Source, rep: dict[str, Any], target: Target) -> None:
        """Plan every group whose layout this library builds, and list modules whose stored tensors are missing.

        A group's layout is `group_format`: integer ``pack-quantized`` (any WNA16/W4A8 spelling), FP8 W8A8
        (``float-quantized``) and NVFP4 (``nvfp4-pack-quantized``) are planned; a mixed checkpoint plans each
        group on its own, which is how vLLM serves it (one quant method per module). A group of another
        scheme is reported unstudied. When no group's layout is known, the whole checkpoint is.
        """
        from ..step import Refused

        kinds: list[LayerKind] = ["dense", "moe"] if rep["experts"] else ["dense"]
        # the router's own per-expert scale makes vLLM route outside the fused kernel (as in the build)
        decomposed = bool(rep.get("custom_router"))
        planned = False
        for name, g in rep["groups"].items():
            f = group_format(g)
            if f is None:
                rep["reasons"].append(f"{name}: {g['scheme']} is not a format of this library")
                continue
            planned = True
            # a mixed checkpoint's group that names its modules stores dense units only (an attention
            # override), so its format is not asked to serve a bank it can never claim
            group_kinds: list[LayerKind] = kinds if _claims_experts(g.get("targets") or ()) else ["dense"]
            for k in group_kinds:
                try:
                    plan = target.plan(f, k, decomposed=decomposed and k == "moe")
                except Refused as e:
                    rep["reasons"].append(f"{name} ({f.name}, {k}): {e}")
                    continue
                rep["kernels"][f"{name}/{k}"] = plan.kernel
                if rep["g_idx"] and "group" not in plan.act_order:
                    rep["reasons"].append(
                        f"{name} ({k}): weight_g_idx (act order 'group') does not load on {target.vllm}"
                    )
        if not planned:
            super().serve(src, rep, target)
            return
        rep["unpacked"] = unpacked(src, rep)
        rep["reasons"] += [
            f"{m}: the config quantizes it, but its stored tensors are missing" for m in rep["unpacked"]
        ]


class GptqAwq(CheckpointFormat):
    """GPTQ or AWQ: one format string naming its bits, group size and symmetry."""

    name: ClassVar[str] = "gptq_awq"

    def matches(self, src: Source) -> bool:
        """``quant_method`` is ``gptq`` or ``awq``."""
        return _qc(src).get("quant_method") in ("gptq", "awq")

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """The method with its bits, group size and symmetry; the modules it keeps out of ``modules_to_not_convert``."""
        qc = _qc(src)
        qm = qc.get("quant_method")
        sym = qc.get("sym", not qc.get("zero_point", False))
        out["format"] = f"{qm} w{qc.get('bits')} g{qc.get('group_size')} {'sym' if sym else 'asym'}"
        out["kept_bf16"] = list(qc.get("modules_to_not_convert") or [])


class Fp8(CheckpointFormat):
    """FP8, with one scale per tensor or per 128x128 block."""

    name: ClassVar[str] = "fp8"

    def matches(self, src: Source) -> bool:
        """``quant_method: fp8``."""
        return _qc(src).get("quant_method") == "fp8"

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """``fp8-block`` when the scale is per 128x128 block, else ``fp8``."""
        qc = _qc(src)
        out["format"] = "fp8-block" if qc.get("weight_block_size") else "fp8"
        out["kept_bf16"] = list(qc.get("ignore") or qc.get("modules_to_not_convert") or [])


class Modelopt(CheckpointFormat):
    """NVIDIA ModelOpt: the format string names its ``quant_algo``."""

    name: ClassVar[str] = "modelopt"

    def matches(self, src: Source) -> bool:
        """``quant_method: modelopt``."""
        return _qc(src).get("quant_method") == "modelopt"

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """``modelopt`` with the ``quant_algo`` it names, at the top level or under ``quantization``."""
        qc = _qc(src)
        out["format"] = f"modelopt {qc.get('quant_algo') or (qc.get('quantization') or {}).get('quant_algo')}"
        out["kept_bf16"] = list(qc.get("ignore") or qc.get("modules_to_not_convert") or [])


class Mxfp4(CheckpointFormat):
    """MXFP4 at the checkpoint level (``quant_method: mxfp4``)."""

    name: ClassVar[str] = "mxfp4"

    def matches(self, src: Source) -> bool:
        """``quant_method: mxfp4``."""
        return _qc(src).get("quant_method") == "mxfp4"

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """The format is ``mxfp4``."""
        qc = _qc(src)
        out["format"] = "mxfp4"
        out["kept_bf16"] = list(qc.get("ignore") or qc.get("modules_to_not_convert") or [])


class Unknown(CheckpointFormat):
    """A quantization config whose ``quant_method`` no registered format recognises; asked last."""

    name: ClassVar[str] = "unknown"
    priority: ClassVar[int] = -100

    def matches(self, src: Source) -> bool:
        """Everything: this class is the last one asked."""
        del src
        return True

    def describe(self, src: Source, out: dict[str, Any]) -> None:
        """The ``quant_method`` this library does not know, and the modules kept out of the quantizer."""
        qc = _qc(src)
        out["format"] = f"{qc.get('quant_method')} (unknown to qda)"
        out["kept_bf16"] = list(qc.get("ignore") or qc.get("modules_to_not_convert") or [])


def _check(key: str, fmt: object) -> None:
    """Refuse a checkpoint format plugin of the wrong type, from either source."""
    if not isinstance(fmt, CheckpointFormat):
        raise TypeError(f"checkpoint format plugin {key!r} is {fmt!r}, not a CheckpointFormat instance")


GGUF = Gguf()


UNQUANTIZED = Unquantized()


COMPRESSED_TENSORS = CompressedTensors()


GPTQ_AWQ = GptqAwq()


FP8 = Fp8()


MODELOPT = Modelopt()


MXFP4 = Mxfp4()


UNKNOWN = Unknown()


#: Checkpoint formats by name: built-ins and the ``qda.checkpoint_formats`` entry points.
CHECKPOINT_FORMATS: Registry[CheckpointFormat] = Registry(
    "checkpoint_formats",
    {
        "compressed_tensors": "qda_quant.intake.formats:COMPRESSED_TENSORS",
        "fp8": "qda_quant.intake.formats:FP8",
        "gguf": "qda_quant.intake.formats:GGUF",
        "gptq_awq": "qda_quant.intake.formats:GPTQ_AWQ",
        "modelopt": "qda_quant.intake.formats:MODELOPT",
        "mxfp4": "qda_quant.intake.formats:MXFP4",
        "unknown": "qda_quant.intake.formats:UNKNOWN",
        "unquantized": "qda_quant.intake.formats:UNQUANTIZED",
    },
    on_add=_check,
)


def checkpoint_format(src: Source) -> CheckpointFormat:
    """The registered format that recognises `src`, asked in decreasing `priority`, ties by registry name.

    Raises:
        IntakeRefused: no registered format matches (never with the built-in ``unknown``).
    """
    formats = sorted(CHECKPOINT_FORMATS.items(), key=lambda kv: (-kv[1].priority, kv[0]))
    for _, fmt in formats:
        if fmt.matches(src):
            return fmt
    raise IntakeRefused(f"{src.where}: no registered checkpoint format recognises it")


def detect(src: Source) -> dict[str, Any]:
    """Format, scheme and layout of `src`.

    ``{"format", "layout", "groups", "kept_bf16", "g_idx", "experts", "custom_router"}``; ``custom_router``
    says the checkpoint stores a router that scales each selected expert's weight (``per_expert_scale``), so
    vLLM selects the experts outside its fused kernel and binds the target's decomposed expert kernel.
    """
    out: dict[str, Any] = {"groups": {}, "kept_bf16": [], "layout": "hf"}
    out["g_idx"] = any(k.endswith("weight_g_idx") for k in src.tensors)
    out["experts"] = any(".experts." in k for k in src.tensors)
    out["custom_router"] = any(k.endswith(".per_expert_scale") for k in src.tensors)
    checkpoint_format(src).describe(src, out)
    return out
