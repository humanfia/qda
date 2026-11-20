"""Resolve checkpoint serving settings and validate the loaded engine against them.

A checkpoint qda-quant writes carries ``build.json["serve"]``::

    {"vllm": "0.33.0",                                  # the vLLM it was built for
     "env": {"VLLM_DISABLED_KERNELS": "..."},           # the engine process's environment
     "args": {"moe_backend": "marlin"},                  # vllm.LLM keyword arguments
     "plugins": ["..."],                                 # vLLM general plugins that must be present
     "kernels": {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]}}   # what vLLM must bind

The KL is read on the kernels the plan names, so after load the kernels vLLM really bound are compared with
``kernels`` and a mismatch is refused. On a card whose target table is not the build's (an sm90 checkpoint on an
RTX PRO 6000), or when ``kernels`` names a kernel this card's table no longer binds, the kernels compared are this
card's plan for the same groups (:func:`card_kernels`), so the KL is the KL of serving on this card. Apart from the
engine's deterministic-reduction pin (:func:`deterministic_args`), nothing is defaulted here: a checkpoint without
``build.json`` is served with no settings unless the caller gives them (the scorer's ``serve_env`` /
``serve_args``).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from importlib.metadata import entry_points, version
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, Field, field_validator
from qda_quant.format import Format
from qda_quant.intake.formats import _claims_experts, _ct_group, group_format
from qda_quant.output.serve_command import command_environment, serve_section
from qda_quant.output.vllm_env import (
    DEFAULT_TP,
    NO_JIT_ENV,
    cache_env,
    disabled_kernels,
    engine_start,
    no_jit_args,
    vllm_environ,
)
from qda_quant.step import Refused
from qda_quant.target import Target, serving_refusal, stored_marlin_pin, unpinned

T = TypeVar("T")


class ServeSettings(BaseModel):
    """The serve settings of one checkpoint (``build.json["serve"]``, plus any explicit override)."""

    vllm: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)
    plugins: list[str] = Field(default_factory=list)
    kernels: dict[str, set[str]] = Field(default_factory=dict)
    source: str = "none"
    nccl_nvls: bool | None = None  # NCCL_NVLS_ENABLE for the engine's ranks (engine_env); None: left as it is

    @field_validator("kernels", mode="before")
    @classmethod
    def _sets(cls, v: dict) -> dict:
        return {k: kernel_set(x) for k, x in (v or {}).items()}


def checkpoint_format(checkpoint: str | Path) -> Format | None:
    """The quantization format a checkpoint's ``build.json`` names, or None when it names none.

    A released checkpoint and every ``qda build`` write ``build.json``; a source model (an HF directory, the
    original a reference is built from) has none, and a format this build does not know parses to None rather
    than to a refusal: the engine's own load check is the authority for those.
    """
    try:
        data = json.loads((Path(checkpoint) / "build.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("format") if isinstance(data, dict) else None
    if not isinstance(name, str):
        return None
    try:
        return Format.parse(name)
    except Refused:
        return None


def check_servable(checkpoint: str | Path, tp: int = DEFAULT_TP) -> None:
    """Refuse a checkpoint whose format needs a newer card than this machine has, before the engine starts.

    The fact is the target table's (`qda_quant.target.serving_refusal`): W4A8 is planned only for compute
    capability 9.0, so on an L40S (8.9) or an RTX PRO 6000 (12.0) the engine would die inside vLLM's scheme
    search with "No compressed-tensors compatible scheme was found". The card's capability comes from
    the devices the engine will use (``tp`` of them). A checkpoint without ``build.json``, a machine where torch
    sees no card, and every weight-only format keep the engine's own refusal.

    Raises:
        Refused: the format needs a capability this card does not have; the message names the format, the
            capability it needs, what the card is, and the W4A16 alternative.
    """
    fmt = checkpoint_format(checkpoint)
    if fmt is None:
        return
    if why := serving_refusal(fmt, cards(tp)):
        raise Refused(why)


def cards(tp: int = DEFAULT_TP) -> list[tuple[str, tuple[int, int]]]:
    """(device label, compute capability) of the ``tp`` cards an engine uses; empty where torch sees no card."""
    try:
        import torch

        return [(f"cuda:{i}", torch.cuda.get_device_capability(i)) for i in range(max(tp, 1))]
    except Exception:  # noqa: BLE001  (no usable driver: the engine's own words are clearer than ours here)
        return []


def card_kernels(
    checkpoint: str | Path, recorded: Mapping[str, set[str]], here: Sequence[tuple[str, tuple[int, int]]]
) -> tuple[dict[str, set[str]], str | None]:
    """The kernels vLLM must bind for ``checkpoint`` on the cards ``here``, and the table they come from.

    ``recorded`` is ``build.json["serve"]["kernels"]``, the build target's plan. vLLM binds the plan of the card it
    runs on, so when that card's target table (`qda_quant.target.Target.on_cards`) is another than the build's,
    or ``recorded`` names a kernel this card's table no longer binds, each config group of ``config.json`` is
    planned on both tables: a recorded kernel becomes this card's kernel for the groups it served, and a head in
    its own format (``re:.*lm_head$``) gets this card's `Target.binds`. An sm90 W3A16 checkpoint records dense
    ``[HummingLinearKernel, MacheteLinearKernel]`` (its int4 head binds Machete on an H100); on an RTX PRO 6000
    the same groups are ``[HummingLinearKernel, MarlinLinearKernel]``.

    Returns:
        The kernels by kind and this card's table name; ``recorded`` and None when ``build.json`` names no
        target, no one table covers the cards, the cards are the build's and bind every recorded kernel, or this
        card's table refuses a format the checkpoint stores (the engine's binding is then refused as before).
    """
    record = _json(Path(checkpoint) / "build.json")
    if not recorded or "target" not in record or not here:
        return dict(recorded), None
    try:
        built = Target.stored(record["target"])
        arch = Target.on_cards(None, here).arch
        serve = record.get("serve") or {}
        if stored_marlin_pin(record):  # built on the old Marlin default, read without the pin (`serve_section`)
            built, serve = replace(built, backend="marlin"), unpinned(serve)
            card = Target(arch, built.vllm)
        else:
            card = built if arch == built.arch else Target(arch, built.vllm)  # a backend pin is the build card's
        planned = _replan(Path(checkpoint), recorded, built, card, serve)
    except Refused:
        return dict(recorded), None
    same = card == built and all(recorded[k] <= planned[k] for k in recorded)
    return (dict(recorded), None) if same else (planned, card.arch)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _replan(
    checkpoint: Path, recorded: Mapping[str, set[str]], built: Target, card: Target, serve: dict
) -> dict[str, set[str]]:
    """`card_kernels`' translation of ``recorded`` from the ``built`` table to the ``card`` table.

    The build's side is its plan, as ``build.json`` records it. This card's side is what its binding rules say
    vLLM binds under the build's ``serve`` arguments, since the engine starts with them: `Target.binds` for a
    head, `Target.binds_moe` for experts (a ``moe_backend: humming`` build binds Humming experts on any card),
    and the dense row for the body, whose rule on every card but 9.0 does not read the arguments. A group's
    format plans a dense kernel only when the checkpoint stores a dense module in it (`_dense_formats`): the
    published Nemotron-3-Nano W3A16 keeps its 3-bit body group for the experts alone, every dense module being
    in an 8-bit or 4-bit group, and vLLM binds no Humming dense kernel on an RTX PRO 6000.

    Raises:
        Refused: a table refuses a format the checkpoint stores.
    """
    config = _json(checkpoint / "config.json")
    qc = config.get("quantization_config") or {}
    text = config.get("text_config") or config
    hidden = int(text.get("hidden_size", 0))
    head_shape = (int(text.get("vocab_size", 0)), hidden)
    bank_shape = (hidden, int(text.get("moe_intermediate_size", 0)))
    pairs: dict[str, list[tuple[str, str]]] = {"dense": [], "moe": []}
    head: set[str] = set()
    dense_formats = _dense_formats(checkpoint)
    for raw in (qc.get("config_groups") or {}).values():
        g = _ct_group(raw)
        g["format"] = g["format"] or qc.get("format")
        fmt = group_format(g)
        if fmt is None:
            continue
        if any("lm_head" in t for t in g["targets"]):
            head.add(card.binds(fmt, head_shape, serve))
            continue
        # only the kinds the record names: a dense model's ``Linear`` group has no expert row to plan (FP8 W8A8)
        if "dense" in recorded and (dense_formats is None or fmt.name in dense_formats):
            pairs["dense"].append((built.plan(fmt, "dense").kernel, card.plan(fmt, "dense").kernel))
        if "moe" in recorded and _claims_experts(g["targets"]):
            for decomposed in (False, True):
                b = built.plan(fmt, "moe", decomposed=decomposed).kernel
                pairs["moe"].append((b, card.binds_moe(fmt, bank_shape, serve, decomposed=decomposed)))
    out: dict[str, set[str]] = {}
    for kind, names in recorded.items():
        known = {b for b, _ in pairs[kind]}
        out[kind] = {c for b, c in pairs[kind] if b in names}
        # a kernel no group explains was the head's on the build's table; without a head group it stays
        out[kind] |= head if kind == "dense" and head else names - known
    return out


#: a routed expert's stored stem (``...experts.<e>.<proj>``); a shared expert's is a dense module
_ROUTED = re.compile(r"(?<!shared_)experts\.\d+\.")


def _dense_formats(checkpoint: Path) -> set[str] | None:
    """The formats the checkpoint stores its dense modules in (not routed experts, not the head), by name.

    None when the checkpoint has no ``model.safetensors.index.json`` to read the stored modules from, or is
    not a layout `Packed` reads (an FP8 W8A8 ``float-quantized`` build).
    """
    from qda_quant.engine.packed import Packed, is_packed

    if not (checkpoint / "model.safetensors.index.json").is_file() or not is_packed(checkpoint):
        return None
    packed = Packed(checkpoint)
    return {
        packed.format_of(s).name for s in packed.stems if not _ROUTED.search(s) and not s.endswith("lm_head")
    }


def kernel_set(x: str | Iterable[str]) -> set[str]:
    """The kernel classes a serve section names for one layer kind.

    A list, as builds write it; a string is a one-element set (``build.json`` files written before lists, among
    them published checkpoints).
    """
    return {x} if isinstance(x, str) else set(x)


def settings_for(
    checkpoint: str | Path,
    env: dict[str, str] | None = None,
    args: dict[str, Any] | None = None,
    nccl_nvls: bool | None = None,
) -> ServeSettings:
    """Read ``build.json["serve"]`` when present and apply explicit overrides.

    ``build.json``'s section is applied on this host, so it must hold only what ``qda build`` writes
    (``qda_quant.output.serve_policy``); the explicit overrides are the caller's own and are not checked.
    ``nccl_nvls`` is the caller's config key (:attr:`ServeSettings.nccl_nvls`); a checkpoint never names it.

    Raises:
        Refused: an override names another ``dtype`` than ``build.json`` does (the checkpoint's dtype wins,
            and a conflicting request is refused rather than dropped); ``ServeNotAllowed`` when ``build.json``'s
            serve section holds anything else.
    """
    serve = serve_section(checkpoint)
    got = ServeSettings()
    if serve is not None:
        got = ServeSettings.model_validate({**serve, "source": "build.json"})
    args = dict(args or {})
    got = got.model_copy(update={"nccl_nvls": nccl_nvls})
    if "dtype" in got.args and "dtype" in args and args["dtype"] != got.args["dtype"]:
        raise Refused(
            f"scorer.topk.serve_args dtype={args['dtype']} conflicts with build.json's dtype {got.args['dtype']}"
        )
    if env or args:
        source = f"{got.source}+override" if got.source != "none" else "override"
        got = got.model_copy(
            update={"env": {**got.env, **(env or {})}, "args": {**got.args, **args}, "source": source}
        )
    return got.model_copy(update={"args": deterministic_args(got.args)})


def deterministic_args(args: dict[str, Any]) -> dict[str, Any]:
    """Add the engine's deterministic-reduction pin under the caller's own compilation settings.

    A cold engine otherwise autotunes the fused-RMSNorm reduction's ``R0_BLOCK`` by timing; two candidates tie
    within a millisecond and the winner changes bf16 rounding, so the same weights self-score two different KLs.
    ``compilation_config.inductor_compile_config.deterministic`` reaches the compile process (an env
    variable does not) and picks one config without timing. ``serve_args`` may set another value for the key.

    vLLM also takes an integer compilation level: ``0`` is no compilation at all, so it gets no pin, and any
    other level becomes vLLM's own dict form, ``{"mode": level}``, with the pin under it. A caller-built
    ``CompilationConfig`` object is returned unchanged, as the caller's settings are the explicit ones.
    """
    raw = args.get("compilation_config")
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw == 0:  # CompilationMode.NONE: nothing is compiled, so there is no config to choose
            return dict(args)
        compilation: dict[str, Any] = {"mode": raw}
    elif raw is None:
        compilation = {}
    elif isinstance(raw, dict):
        compilation = dict(raw)
    else:
        return dict(args)
    inductor = {"deterministic": True, **compilation.get("inductor_compile_config", {})}
    return {**args, "compilation_config": {**compilation, "inductor_compile_config": inductor}}


#: ``vllm.LLM`` keywords every reference, scoring and gencheck engine starts with: FlashInfer autotune off (its
#: cost and effect on a reading: docs/verify.md). The printed serve line keeps vLLM's default.
NO_AUTOTUNE_ARGS = {"enable_flashinfer_autotune": False}


def no_jit(settings: ServeSettings) -> ServeSettings:
    """`settings` with `NO_JIT_ENV`, `no_jit_args` and `NO_AUTOTUNE_ARGS`: the engine compiles no FlashInfer kernel
    at run time and times no FlashInfer tactic.

    A gencheck gives both models the same settings, and its original's cache key includes them; a scoring engine
    starts with them too. A ``serve_args`` that sets ``enable_flashinfer_autotune`` itself (also read from
    ``kernel_config``; vLLM refuses both spellings at once) keeps its value.
    """
    args = no_jit_args(settings.args)
    kernel = args.get("kernel_config")
    if "enable_flashinfer_autotune" not in args and not (
        isinstance(kernel, dict) and "enable_flashinfer_autotune" in kernel
    ):
        args = {**NO_AUTOTUNE_ARGS, **args}
    return settings.model_copy(update={"env": {**settings.env, **NO_JIT_ENV}, "args": args})


def check_dtype(dtype: str, anchor_dtype: str | None, allow_mismatch: bool = False) -> None:
    """Refuse a different activation dtype unless the scorer explicitly allows it.

    Raises:
        Refused: the dtype differs from the anchor's and ``allow_mismatch`` is off.
    """
    aliases = {
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "float": "float32",
        "fp32": "float32",
    }

    def normalized(value: str) -> str:
        name = str(value).lower().removeprefix("torch.")
        return aliases.get(name, name)

    if anchor_dtype and normalized(dtype) != normalized(anchor_dtype) and not allow_mismatch:
        raise Refused(
            f"serve dtype {dtype} differs from reference anchor dtype {anchor_dtype}; "
            "set scorer.topk.allow_dtype_mismatch: true to score anyway"
        )


def resolve_dtype(
    s: ServeSettings, anchor_dtype: str | None, *, allow_mismatch: bool = False
) -> tuple[ServeSettings, str]:
    """Resolve the activation dtype and return the settings with its source.

    Precedence: ``build.json`` serve.args.dtype, then ``scorer.topk.serve_args.dtype``, then the reference's
    anchor dtype; never vLLM's ``auto`` (a float16 checkpoint read in float16 against a bfloat16 anchor is
    another number).

    Raises:
        Refused: ``dtype`` is ``auto``, differs from the anchor without permission, or is unspecified everywhere.
    """
    if str(s.args.get("dtype", "")).lower() == "auto":
        raise Refused("dtype 'auto' is not accepted: name the dtype (the reference anchor's is the default)")
    if "dtype" in s.args:
        check_dtype(s.args["dtype"], anchor_dtype, allow_mismatch)
        return s, "serve-settings" if s.source == "build.json" else s.source
    if not anchor_dtype:
        raise Refused(
            "no dtype: the reference does not record its anchor's; set scorer.topk.serve_args.dtype=..."
        )
    return s.model_copy(update={"args": {**s.args, "dtype": anchor_dtype}}), "reference anchor"


def check_vllm(s: ServeSettings, running: str, allow_mismatch: bool) -> str | None:
    """Return any vLLM version mismatch, raising unless the caller allows it."""
    if s.vllm is None or s.vllm == running:
        return None
    why = f"the checkpoint names vLLM {s.vllm}, this is vLLM {running}"
    if not allow_mismatch:
        raise Refused(f"{why} (set scorer.topk.allow_vllm_mismatch: true to score anyway)")
    return why


def check_plugins(s: ServeSettings) -> None:
    """Require every named plugin to be installed as a vLLM general plugin."""
    have = {e.name for e in entry_points(group="vllm.general_plugins")}
    missing = [p for p in s.plugins if p not in have]
    if missing:
        raise RuntimeError(f"vLLM general plugins {missing} are not installed (installed: {sorted(have)})")


def backend_source(s: ServeSettings, name: str) -> str:
    """Where a backend override came from: ``build.json``, ``serve_args``, or the engine's ``default``.

    ``name`` is a vLLM backend argument (``linear_backend``, ``moe_backend``). The kernel the engine binds
    depends on it, so a recorded result says who forced it.
    """
    if name not in s.args:
        return "default"
    return "build.json" if s.source == "build.json" else "serve_args"


def check_kernels(expected: Mapping[str, str | Iterable[str]], bound: dict[str, list[str]]) -> None:
    """Refuse when, for a kind (``dense`` / ``moe``) named in ``expected``, vLLM bound another set of kernels.

    Args:
        expected: kind -> the kernel classes the checkpoint names (`kernel_set`); a body and a head stored in
            different formats are two dense kernels.
        bound: kind -> the kernel classes vLLM bound, over every quantized layer of that kind.
    """
    bad = [
        f"{kind}: expected {sorted(kernel_set(want))}, bound {bound.get(kind) or 'none'}"
        for kind, want in expected.items()
        if set(bound.get(kind) or ()) != kernel_set(want)
    ]
    if bad:
        raise Refused("the engine did not bind the checkpoint's kernels: " + "; ".join(bad))


@contextlib.contextmanager
def engine_env(s: ServeSettings) -> Iterator[None]:
    """``s.env`` (and ``VLLM_PLUGINS`` = ``s.plugins``) set while the engine starts; old values restored after.

    This is :func:`qda_quant.output.vllm_env.vllm_environ`: the venv's ``bin`` goes on ``PATH`` unless ``s.env`` sets
    its own, and FlashInfer's toolchain is checked on the merged environment before the engine starts.

    ``s.nccl_nvls`` sets ``NCCL_NVLS_ENABLE``: an engine on more than one card otherwise fails at NCCL group init
    on nodes that cannot bind NVLS's multicast memory (a build's ``nccl_nvls`` is the same setting).

    Raises:
        RuntimeError: the toolchain check refused the merged environment.
    """
    s.env = engine_environment(s)
    with vllm_environ(s.env):
        yield


#: The smaller of vLLM's defaults for ``vllm.LLM``'s ``max_num_seqs`` (1024 on an 80 GB H100, 256 on smaller
#: cards): a prompt-set engine never asks for more sequences than vLLM would on any card.
VLLM_MAX_NUM_SEQS = 256


def batch_args(args: Mapping[str, Any], prompts: int) -> dict[str, Any]:
    """``args`` with the ``max_num_seqs`` of an engine that serves ``prompts`` sequences, unless ``args`` names it.

    The prompt count, at least 1 and at most :data:`VLLM_MAX_NUM_SEQS`. vLLM sizes its memory for ``max_num_seqs``
    sequences, and a hybrid model's cache holds one Mamba block per sequence: at vLLM's default an engine that
    serves a few hundred sequences can refuse to start on one card.
    """
    if "max_num_seqs" in args:
        return dict(args)
    return {**args, "max_num_seqs": max(1, min(prompts, VLLM_MAX_NUM_SEQS))}


def pinned(model: str, args: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``model`` as the engine takes it, and ``args`` with the ``revision`` its ``org/name@rev`` names.

    vLLM reads ``@rev`` as part of the repo id; it takes the revision as ``revision=`` (and its tokenizer follows
    it), so ``org/name@rev`` is ``org/name`` with ``revision=rev``. A local path, or an id without ``@``, is
    returned as given.

    Raises:
        Refused: ``args`` names another ``revision`` than the one after ``@``.
    """
    from qda_quant.intake.model_ref import parse

    ref = parse(model)
    if ref.revision is None:
        return model, dict(args)
    if args.get("revision") not in (None, ref.revision):
        raise Refused(f"{model}: serve_args names revision {args['revision']}, the model id {ref.revision}")
    return ref.name, {**args, "revision": ref.revision}


#: What vLLM says when its cache holds fewer per-sequence blocks than ``max_num_seqs`` (a hybrid model's Mamba
#: cache), in the engine core's traceback.
_CACHE_BLOCKS = re.compile(r"max_num_seqs \((\d+)\) exceeds available (\w+) cache blocks \((\d+)\)")


def cache_refusal(text: str) -> Refused | None:
    """The refusal of an engine start that ``text`` says has too few cache blocks, or None.

    Its fix names a ``max_num_seqs`` for each place one is set: the largest power of two within the blocks.
    """
    m = _CACHE_BLOCKS.search(text)
    if m is None:
        return None
    asked, kind, blocks = m.groups()
    n = 1 << (int(blocks).bit_length() - 1) if int(blocks) > 0 else 1
    return Refused(
        f"max_num_seqs={asked} exceeds the engine's {blocks} {kind} cache blocks (one per sequence)",
        f"qda-kl kl: add --serve-args '{{\"max_num_seqs\": {n}}}'; qda-kl score: set "
        f"scorer.topk.serve_args.max_num_seqs: {n}; a judge run: set judge.kl_layout.serve_args.max_num_seqs: {n} "
        f"(any value up to {blocks} starts), or raise --gpu-memory-utilization",
    )


def start_engine(s: ServeSettings, start: Callable[[], T]) -> T:
    """Run ``start`` (which builds an engine) in ``s``'s environment (:func:`engine_environment`).

    ``s.env`` is set to what the engine started with, the resolved cache directories included, so a caller that
    records its serve settings records them.

    Raises:
        Refused: the engine's cache holds fewer blocks than ``max_num_seqs`` (:func:`cache_refusal`, with its fix).
        Exception: what ``start`` raised, unchanged.
    """
    value, s.env = engine_start(engine_environment(s), start, cache_refusal)
    return value


def llm(
    s: ServeSettings,
    kwargs: dict[str, Any],
    *,
    constructor: Callable[..., Any] | None = None,
    refusals: bool = True,
    **explicit: Any,
) -> Any:
    """Start vLLM or its harness constructor with unchanged arguments.

    ``refusals``: read the engine's own output and raise :func:`cache_refusal`'s refusal instead of vLLM's
    traceback, and record the environment it started in on ``s``.
    """
    if constructor is None:
        from vllm import LLM

        constructor = cast(Callable[..., Any], LLM)
    if refusals:
        return start_engine(s, lambda: constructor(**explicit, **kwargs))
    with engine_env(s):
        return constructor(**explicit, **kwargs)


def check_tp(kwargs: dict, tp: int, err: type[Exception] = Refused) -> None:
    """Refuse an engine argument that overrides the caller's card count."""
    if int(kwargs["tensor_parallel_size"]) != tp:
        raise err(
            f"serve_args names tensor_parallel_size={kwargs['tensor_parallel_size']}; set tp={tp} instead"
        )


def check_serve_env(env: Mapping[str, str], nccl_nvls: bool) -> None:
    """Refuse an ``env`` that names ``NCCL_NVLS_ENABLE`` with another value than ``nccl_nvls``.

    ``NCCL_NVLS_ENABLE`` has one home, ``nccl_nvls``; a ``serve_env`` naming it is refused rather than silently
    overriding it.

    Raises:
        Refused: ``env`` names another value than ``nccl_nvls`` sets.
    """
    named = env.get("NCCL_NVLS_ENABLE")
    if named not in (None, "1" if nccl_nvls else "0"):
        raise Refused(f"serve_env names NCCL_NVLS_ENABLE={named}; set nccl_nvls instead")


def engine_environment(s: ServeSettings) -> dict[str, str]:
    """Resolve the engine's NVLS, plugin and cache directory settings for launch and records.

    ``VLLM_DISABLED_KERNELS`` keeps the list the user exported (`qda_quant.output.vllm_env.disabled_kernels`).
    """
    return cache_env(disabled_kernels(command_environment(s.env, s.plugins, s.nccl_nvls)))


def vllm_version() -> str:
    """Return the installed vLLM version."""
    return version("vllm")
