"""Candidate 0's group size: read the model's widths, then ask `qda-quant` which group sizes store them.

The build's group-size check (`qda_quant.engine.planning.working_group_sizes`) needs only a model's linear
widths, so `prepare` can ask it before any step runs: the widths come from `config.json` and the safetensors
headers (a few KB per shard, no weights, no GPU). One fact, one owner: this module picks the group and the
same `Format` is written to candidate 0's recipe and to `search.yaml`.

A model whose default group size (128) divides every width keeps it. Otherwise prepare takes the largest of
128 > 64 > 32 that works. It never takes per-channel (-1) on its own: that is a different precision preset,
so the refusal names it (and the other formats that work) and asks for `--format`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qda_quant.arch import MIN_QUANT_ROWS
from qda_quant.engine.planning import (
    QuantWidth,
    WidthProblem,
    check_widths,
    kept_unservable,
    unservable_widths,
    working_group_sizes,
)
from qda_quant.format import Format
from qda_quant.intake.probe import Source
from qda_quant.step import Refused
from qda_quant.target import Target

__all__ = [
    "AUTO_GROUPS",
    "Decision",
    "auto_format",
    "hub_widths",
    "local_widths",
    "spelling",
    "run_target",
    "target_for",
    "with_target",
]

#: The group sizes prepare may lower a default format to on its own, largest first (per-channel is not one).
AUTO_GROUPS = (128, 64, 32)


@dataclass(frozen=True)
class Decision:
    """What prepare stores: the format, what ruled the default out, and every group size that works."""

    fmt: Format
    trigger: WidthProblem | None
    feasible: tuple[int, ...]


def _quant_modules(tensors: Any) -> tuple[QuantWidth, ...]:
    """The 2-D text weights of `tensors` (name -> shape), as the build's width check sees them.

    Token embeddings and the head are left out (a body build does not store them by `format`), and so are
    tensor names of a vision, audio or projector tower: they are not in the text graph the build quantizes.
    Routed-expert tensors keep their divisibility check but not a dense kernel binding (their served shape and
    layout are the build's to read). A fused expert parameter is 3-D and does not reach this function.
    """
    skip = ("vision", "visual", "audio", "multi_modal", "mm_projector", "patch_embed", "image", "video")
    out = []
    for name, shape in sorted(tensors.items()):  # one order, so the recorded trigger is stable
        if len(shape) != 2:
            continue
        base = name.lower()
        if base.endswith(
            (
                "embed_tokens.weight",
                "embeddings.weight",
                "wte.weight",
                "word_embeddings.weight",
                "lm_head.weight",
            )
        ):
            continue
        if any(word in base for word in skip):
            continue
        rows, cols = int(shape[0]), int(shape[1])
        if rows < MIN_QUANT_ROWS:  # the plan keeps a Linear this narrow bf16 (derive_groups)
            continue
        expert = "experts." in name or ".experts" in name
        module = name.removesuffix(".weight")  # the build names the module, not its parameter
        out.append(QuantWidth(module, rows, cols, expert=expert, bind=not expert))
    return tuple(out)


def _widths(src: Source) -> tuple[QuantWidth, ...]:
    """The modules a build of `src` quantizes, from its `config.json` and safetensors headers (no weights)."""
    if "config.json" not in src.jsons:
        raise FileNotFoundError(f"{src.where}: no config.json")
    if not src.tensors:
        raise FileNotFoundError(f"{src.where}: no *.safetensors to read widths from")
    return _quant_modules({name: shape for name, (_, shape) in src.tensors.items()})


def local_widths(model_dir: Path) -> tuple[QuantWidth, ...]:
    """The modules a build on the local model directory quantizes."""
    return _widths(Source.open(str(model_dir)))


def hub_widths(repo: str, revision: str | None = None) -> tuple[QuantWidth, ...]:
    """The same, before the download: the Hub's `config.json` and safetensors metadata (range reads, no weights)."""
    return _widths(Source.open(f"{repo}@{revision}" if revision else repo))


def target_for(cards: Sequence[int]) -> Target:
    """The serving target for `cards` (physical, PCI-bus numbers), asked of the driver without loading a model.

    The capabilities are nvidia-smi's own (`smi_capabilities`), so no CUDA context is opened: prepare runs
    beside the cards, and the judge prices the run's builds in a CPU process that must not take a context on
    every card of the machine. A process that cannot see the cards (a login node) falls back to torch, then
    to `Target()`, the table a machine with no card plans.
    """
    caps = _smi_caps(cards)
    if caps:
        return Target.on_cards(None, caps)
    return _torch_target(cards)


def cuda_architectures(cards: Sequence[int] | None) -> str:
    """CMake's `CMAKE_CUDA_ARCHITECTURES` for `cards` (physical numbers): their compute capabilities, as
    nvidia-smi reports them (sm120 -> `120`, sm103 -> `103`, sm90 -> `90`), `;`-joined when the cards differ.

    `native` where the cards are not named or the driver does not list them: CMake then builds for the GPUs of
    the machine it runs on, which is the card a build there is for.
    """
    caps = _smi_caps(cards) if cards else []
    archs = sorted({major * 10 + minor for _, (major, minor) in caps})
    return ";".join(str(a) for a in archs) or "native"


def run_target(cards: Sequence[int]) -> Target | None:
    """`target_for` as the judge asks it: nvidia-smi only, never a CUDA context; None when that cannot tell.

    None (cards nvidia-smi cannot read, or cards no one table covers, such as a mix of sm90 and sm100) leaves a
    recipe that names no target on `Target()`, as before: the judge then admits the build on its fallback
    estimate, and `qda build` on the cards plans it for them.
    """
    caps = _smi_caps(cards)
    if not caps:
        return None
    try:
        return Target.on_cards(None, caps)
    except Refused:
        return None


def _smi_caps(cards: Sequence[int]) -> list[tuple[str, tuple[int, int]]]:
    """Each of `cards` with the compute capability nvidia-smi reports; [] when it reports none of them."""
    from .isolation.devices import resolve_cards, smi_capabilities

    try:
        capability = {card.bus: cap for card, cap in smi_capabilities() or ()}
        resolved = resolve_cards(cards)
        return [
            (str(n), capability[c.bus]) for n, c in zip(cards, resolved, strict=False) if c.bus in capability
        ]
    except (OSError, ValueError):  # no driver table, or a card this job cannot see
        return []


def _torch_target(cards: Sequence[int]) -> Target:
    """`target_for` for a machine whose driver nvidia-smi cannot read; `Target()` where torch sees no card."""
    try:
        import torch
    except ImportError:
        return Target()

    if not torch.cuda.is_available():
        return Target()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # prepare's cards are physical numbers
    try:
        caps = [(f"cuda:{c}", tuple(torch.cuda.get_device_capability(c))) for c in cards]
    except (RuntimeError, AssertionError):
        return Target()
    return Target.on_cards(None, caps)


def with_target(raw: dict[str, Any], target: Target | None) -> dict[str, Any]:
    """`raw` with `target` added when the recipe names none and its output has no runtime of its own.

    A recipe's own `target:` wins, and a GGUF exporter names its own (`llama_cpp`), which a build beside
    another target is refused for: neither is overridden. `target` None leaves `raw` as it is.
    """
    from qda_quant.config import target_name
    from qda_quant.output.exporters import parse_exporter

    if target is None or "target" in raw:
        return raw
    if getattr(parse_exporter(raw.get("exporter")), "target_name", None) is not None:
        return raw
    return {**raw, "target": target_name(target)}


def spelling(fmt: Format) -> str:
    """`fmt` as a string that parses back to itself (the short spelling drops an explicit symmetry)."""
    short = fmt.spelling
    return short if Format.parse(short) == fmt else fmt.name


def _spellings(fmt: Format, groups: Sequence[int]) -> list[str]:
    return [spelling(replace(fmt, group_size=g)) for g in groups]


def auto_format(
    fmt: Format, modules: Sequence[QuantWidth], target: Target, *, explicit: bool = False
) -> Decision:
    """The format to store: `fmt` when it works, else the largest auto group, else `Refused`.

    `explicit` means `--format` itself named a group size: it is then taken as given, and refused when it does
    not work instead of being changed.
    """
    # A shape no kernel of `fmt` serves stays bf16 instead of refusing the whole plan; the plan
    # reads the same rule (`kept_unservable`), so candidate 0 and every build agree.
    kept = kept_unservable(unservable_widths(modules, fmt, target), fmt)
    modules = tuple(m for m in modules if m.name not in kept)
    feasible = tuple(working_group_sizes(fmt, modules, target))
    if explicit:
        if fmt.group_size in feasible:
            return Decision(fmt, None, feasible)
        raise Refused(_refusal(fmt, modules, target, feasible))
    chosen = next((g for g in AUTO_GROUPS if g in feasible), None)
    if chosen is None:
        raise Refused(_refusal(fmt, modules, target, feasible))
    if chosen == fmt.group_size:
        return Decision(fmt, None, feasible)
    return Decision(replace(fmt, group_size=chosen), check_widths(fmt, modules, target), feasible)


def _refusal(fmt: Format, modules: Sequence[QuantWidth], target: Target, feasible: Sequence[int]) -> str:
    """Why no format prepare may choose on its own, and the `--format` that would store the model."""
    problem = check_widths(fmt, modules, target)
    why = problem.detail if problem is not None else f"{fmt.name} stores no module of this model"
    if not feasible:
        return f"{why}; no group size works on {target.arch}"
    names = _spellings(fmt, feasible)
    listed = ", ".join(f"--format {name}" for name in names[:4])
    across = " > ".join(map(str, AUTO_GROUPS))
    return (
        f"{why}; prepare lowers a default format only across {across}, not to another preset: "
        f"pass {listed} to store the model"
    )
