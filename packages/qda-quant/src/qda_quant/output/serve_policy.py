"""Which `build.json["serve"]` sections a scorer may apply to a checkpoint it did not build.

`serve` sets the serving engine's process environment (`env`) and arguments (`args`). A scorer applies it on its
own host, so a section written by someone else must hold nothing but what `qda build` itself writes: the
environment and engine arguments of the kernel plans (`qda_quant.target`), and a plain activation dtype. The
allowed keys and values are enumerated from the plan tables of every registered target, every registered format,
both layer kinds and the vLLM versions the tables know; a section `qda build` wrote for any of them passes,
and anything else (another variable such as `PYTHONPATH` or `LD_PRELOAD`, another engine argument, a value
the plans never write) is refused.

    check(build_json["serve"])      # raises ServeNotAllowed
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ..step import Refused

if TYPE_CHECKING:
    from ..arch import ModelGraph
    from ..format import Format
    from ..target import LayerKind

__all__ = ["ServeNotAllowed", "allowed", "check"]

#: Activation dtypes a section may name (`args.dtype`).
DTYPES = ("bfloat16", "float16")
#: The keys a section may have.
KEYS = {"vllm", "env", "args", "plugins", "kernels"}
VERSION = re.compile(r"\d+\.\d+\.\d+")


class ServeNotAllowed(Refused):
    """The serve section holds a setting `qda build` never writes."""


@functools.cache
def allowed() -> dict[str, Any]:
    """The values `qda build` can write, enumerated from the plan tables.

    `env` (variable -> values; `VLLM_DISABLED_KERNELS` -> kernel names), `args` (argument -> values), `plugins`,
    `kernels`, which also holds `BANNED_KERNELS`: no plan names them, but a published checkpoint's record may
    (the sm90 W3A16 int4 head names Machete); `qda-kl` expects this card's kernel in its place
    (`qda_kl.serve.card_kernels`).
    """
    from ..format import FORMATS
    from ..target import BANNED_KERNELS, TARGETS, Target

    env: dict[str, set[str]] = {}
    args: dict[str, set[Any]] = {"dtype": set(DTYPES)}
    plugins: set[str] = set()
    kernels: set[str] = set(BANNED_KERNELS)
    for arch, table in TARGETS.items():
        for ver in {*getattr(table, "versions", ()), Target.vllm}:
            for backend in getattr(table, "backends", (None,)):
                target = Target(arch, ver, backend)
                for _, fmt in FORMATS.items():
                    # the layer kinds a plan may serve: dense, and a bank with its router inside or outside it
                    for kind, decomposed in cast(
                        "tuple[tuple[LayerKind, bool], ...]",
                        (
                            ("dense", False),
                            ("moe", False),
                            ("moe", True),  # a bank whose router selects its experts outside the kernel
                        ),
                    ):
                        try:
                            plan = target.plan(fmt, kind, decomposed=decomposed)
                        except Refused:  # a combination no table serves
                            continue
                        kernels.add(plan.kernel)
                        plugins.update(plan.serve_plugins)
                        for k, v in plan.serve_env.items():
                            vals = v.split(",") if k == "VLLM_DISABLED_KERNELS" else [v]
                            env.setdefault(k, set()).update(x for x in vals if x)
                        for k, v in plan.serve_args.items():
                            args.setdefault(k, set()).add(v)
    return {"env": env, "args": args, "plugins": plugins, "kernels": kernels}


def sglang_limits(
    fmt: Format,
    kind: LayerKind,
    *,
    biased_experts: bool = False,
    graph: ModelGraph | None = None,
    text_config: Any = None,
    source_config: Any = None,
    formats: Mapping[str, Format] | None = None,
) -> tuple[str, ...]:
    """Known compressed-tensors limits in stock SGLang 0.5.23, with evidence and serving alternatives.

    The zero-point limit judges each routed bank's resolved format (`formats`, keyed by its router).
    The resolved map includes expert defaults; a missing bank keeps the body format. Without a map,
    the body's expert default applies. Dense formats do not select this limit.

    These warnings do not change the build or claim support for another engine version.
    """
    if fmt.weight_type != "int":
        return ()
    limits = []
    if fmt.act == "bf16" and (
        getattr(source_config, "model_type", None) == "kimi_k3"
        or getattr(text_config, "model_type", None) == "kimi_k3_text"
    ):
        limits.append(
            "SGLang 0.5.23 cannot load Kimi-K3 WNA16 exports: its MXFP4 loader rewrites "
            "weight_packed to weight for WNA16 too (dense fused-MLP lookup fails); "
            "the bf16 base loads. Use stock vLLM 0.33.0"
        )
    if "Qwen4ExpForCausalLM" in (getattr(source_config, "architectures", None) or ()):
        limits.append(
            "SGLang 0.5.23 cannot serve Qwen4ExpForCausalLM (has no SGlang implementation); "
            "use stock vLLM 0.33.0"
        )
    if fmt.weight_bits == 3:
        limits.append(
            "SGLang 0.5.23 cannot serve INT3 (unsupported CompressedTensorsW4A16Sparse24); "
            "use vLLM 0.33.0 or symmetric W4A16"
        )
    if fmt.weight_bits == 4 and fmt.act == "fp8_token":
        limits.append(
            "SGLang 0.5.23 cannot serve W4A8 dense projections (no compatible compressed-tensors scheme); "
            "use vLLM 0.33.0 or W4A16"
        )
    if graph is not None and fmt.weight_bits == 4 and fmt.act == "bf16":
        from ..arch.adapters import FusedRuntime
        from ..arch.derive import ignore_hits

        aliases = {
            runtime: value.checkpoint if isinstance(value, FusedRuntime) else value
            for runtime, value in graph.extra.get("runtime_names", {}).get("sglang", {}).items()
        }
        if any(
            len(
                {name for group in layer.groups for name in group.modules if ignore_hits((checkpoint,), name)}
            )
            >= 2
            for runtime, checkpoint in aliases.items()
            if not runtime.startswith("re:")
            for layer in graph.layers
        ):
            limits.append(
                "SGLang 0.5.23 cannot serve separately packed inputs of fused dense modules in INT4/BF16 "
                "(skipped payloads or dense weight lookup failure); use vLLM 0.33.0"
            )
    if kind == "moe" and fmt.weight_bits == 4 and fmt.act == "bf16":
        if (
            graph is not None
            and getattr(text_config, "model_type", None) in ("qwen3_5_moe", "qwen3_5_moe_text")
            and any(
                name.endswith((".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"))
                for layer in graph.layers
                for group in layer.groups
                for name in group.modules
            )
        ):
            limits.append(
                "SGLang 0.5.23 leaves separately quantized Qwen3.5-MoE shared-expert gate/up weights "
                "unwritten in its fused gate_up_proj; keep shared_experts float (the default), or use "
                "stock vLLM 0.33.0"
            )
        if graph is not None and getattr(text_config, "model_type", None) == "qwen3_next":
            names = set(graph.extra.get("ignored") or {})
            for role_names in (graph.extra.get("kept_bf16") or {}).values():
                names.update(role_names)
            packed_qkvz = any(
                name.endswith(".linear_attn.in_proj_qkvz")
                for layer in graph.layers
                for group in layer.groups
                for name in group.modules
            )
            qkvz_float = any(n.endswith(".linear_attn.in_proj_qkvz") for n in names) and not packed_qkvz
            shared_float = any(".mlp.shared_expert." in n for n in names)
            if not (qkvz_float and not shared_float):
                limits.append(
                    "SGLang 0.5.23 cannot load Qwen3-Next INT4 with this recipe (its GDN constructor reads "
                    "`.weight` on a packed in_proj_qkvz, and its shared-expert fusion needs the shared expert "
                    "in the packed expert slot); keep in_proj_qkvz float and quantize the shared experts with "
                    'ignore: [gdn_gates, "re:^model\\.layers\\.\\d+\\.linear_attn\\.in_proj_qkvz$"], or use '
                    "stock vLLM 0.33.0"
                )
        activation = getattr(text_config, "hidden_act", None) or getattr(text_config, "hidden_activation", "")
        if isinstance(activation, str) and activation.startswith("gelu"):
            limits.append(
                "SGLang 0.5.23 cannot serve GELU INT4 experts (W4A16 MoE supports only SiLU); use vLLM 0.33.0"
            )
        if biased_experts:
            limits.append(
                "SGLang 0.5.23 cannot serve the biased INT4 expert layout (missing expert bias buffers "
                "and skipped payloads); use vLLM 0.33.0"
            )
    if kind == "moe":
        from ..solve.units import bank_module

        fallback = fmt if formats is not None else fmt.expert_format
        banks = (
            [
                (bank_module(bank), (formats or {}).get(bank.router, fallback))
                for layer in graph.layers
                for bank in layer.banks
            ]
            if graph is not None
            else [(None, fmt.expert_format)]
        )
        affected: dict[int, list[str]] = {}
        for name, experts in banks:
            if (
                experts.weight_type == "int"
                and experts.weight_bits == 4
                and experts.act == "bf16"
                and not experts.symmetric
                and experts.group_size in (64, 128)
            ):
                names = affected.setdefault(experts.group_size, [])
                if name is not None:
                    names.append(name)
        for group_size, names in sorted(affected.items()):
            scope = f" in banks {', '.join(sorted(names))}" if names else ""
            limits.append(
                f"SGLang 0.5.23 cannot load compressed-tensors asymmetric per-expert INT4 g{group_size} zero points"
                f"{scope} (loader orientation mismatch); for SGLang opt in to exporter: awq_bf16 "
                f"(BF16 scales, different kernels), or use vLLM 0.33.0 / w4a16-sym-g{group_size}"
            )
    return tuple(limits)


def check(serve: Any) -> None:
    """Refuse a serve section with anything `qda build` does not write.

    Raises:
        ServeNotAllowed: with a neutral reason that names the offending key and says what to do.
    """
    if serve is None:
        return
    if not isinstance(serve, dict):
        raise ServeNotAllowed(
            "serve settings not allowed: not a mapping; a serve section holds what `qda build` writes, "
            "so rebuild the checkpoint with `qda build`"
        )
    a = allowed()
    if extra := set(serve) - KEYS:
        raise ServeNotAllowed(
            f"serve settings not allowed: {sorted(extra)}; a serve section holds only {sorted(KEYS)}, "
            "so remove it or rebuild the checkpoint with `qda build`"
        )
    v = serve.get("vllm")
    if v is not None and not (isinstance(v, str) and VERSION.fullmatch(v)):
        raise ServeNotAllowed(
            "serve settings not allowed: vllm; give a version like `0.33.0`, the form `qda build` writes"
        )
    env = serve.get("env") or {}
    if not isinstance(env, dict):
        raise ServeNotAllowed(
            "serve settings not allowed: env; `env` is a mapping of the variables `qda build` writes"
        )
    for k, val in env.items():
        ok = k in a["env"] and isinstance(val, str)
        if ok and k == "VLLM_DISABLED_KERNELS":
            ok = all(x in a["env"][k] for x in val.split(",") if x)
        elif ok:
            ok = val in a["env"][k]
        if not ok:
            raise ServeNotAllowed(
                f"serve settings not allowed: env {k}; `qda build` writes only {sorted(a['env'])}, "
                "so remove it or rebuild the checkpoint with `qda build`"
            )
    args = serve.get("args") or {}
    if not isinstance(args, dict):
        raise ServeNotAllowed(
            "serve settings not allowed: args; `args` is a mapping of the engine arguments `qda build` writes"
        )
    for k, val in args.items():
        if k not in a["args"] or not isinstance(val, str | int | float | bool) or val not in a["args"][k]:
            raise ServeNotAllowed(
                f"serve settings not allowed: args {k}; `qda build` writes only {sorted(a['args'])}, "
                "so remove it or rebuild the checkpoint with `qda build`"
            )
    plugins = serve.get("plugins") or []
    if not isinstance(plugins, list) or not all(isinstance(p, str) and p in a["plugins"] for p in plugins):
        raise ServeNotAllowed(
            "serve settings not allowed: plugins; `qda build` writes only the plugins its kernel plans name, "
            "so remove it or rebuild the checkpoint with `qda build`"
        )
    kern = serve.get("kernels") or {}
    if not isinstance(kern, dict) or not all(
        k in ("dense", "moe")
        and isinstance(x, str | list)
        and all(isinstance(c, str) and c in a["kernels"] for c in ([x] if isinstance(x, str) else x))
        for k, x in kern.items()
    ):
        raise ServeNotAllowed(
            "serve settings not allowed: kernels; `qda build` writes only the kernels its plans bind, "
            "so remove it or rebuild the checkpoint with `qda build`"
        )
