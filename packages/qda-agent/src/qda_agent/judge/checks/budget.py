"""The search's budget in bits per quantized weight (bpw): what a candidate's decoder Linears cost.

The budget is one number per task, `judge.budget_bpw`: the bits the checkpoint stores per weight of its
decoder Linears (the body), counting packed codes, scales and zero points, and counting a Linear the
candidate keeps in bf16 at its stored width (16 bits). A candidate over the budget is refused at
admission, and so is candidate 0 (`baselines[0]`, the first baseline) when the host starts: the
fallback must be inside the budget the search states.

The denominator is fixed per model and comes from the original (`judge.anchor`), never from the
candidate: it is the number of weights of the decoder Linears llm-compressor's uniform default
quantizes. So two candidates are comparable when one quantizes fewer modules or keeps more of them bf16:
the weights it leaves out still count at 16 bits.

The output head is outside that count. With `judge.head_lease` off it stays 16-bit (the format gate's
rule) and costs nothing. With the lease on, the head is charged 16 bits a weight up front and may be
stored in a narrower format (`HEAD_LEASE`); the bits it saves are lent to the body:
``(body bits + head bits - 16 x head weights) / body weights`` must be inside the budget (`leased_bpw`).
A tied head stores no tensor of its own and has nothing to lease. The bpw a card or `bpw DIR` reports is
the body's own number.

llm-compressor's uniform default is the recipe of its README (0.15.0's METADATA, the Qwen3-30B-A3B
example): ``QuantizationModifier(targets="Linear", scheme=..., ignore=["lm_head", "re:.*mlp.gate$"])``.
Its targets are the model's `nn.Linear` modules, so:

- `lm_head` is a Linear, kept by name (the lease above prices it on its own);
- a MoE router (`mlp.gate`, or `mixer.gate` in NemotronH) is kept by name;
- a MoE shared expert's small gate (`mlp.shared_expert_gate`) is *not* in the ignore set: llm-compressor
  quantizes it like any Linear, so it is in the denominator (qda keeps that narrow gate bf16, which
  costs it 16 bits a weight there);
- an embedding is an `nn.Embedding`, not a Linear: in neither sum. A model may store more than one (Gemma-4's
  per-layer embedding, `embed_tokens_per_layer`), and position embeddings are matrices shaped like a Linear,
  so the exclusion covers the architectures' embedding names, not just `embed_tokens`;
- the MTP draft head and the vision towers are not decoder Linears: left out (`UNSCORED`).

A model may store its experts as one fused 3-D bank without a `.weight` suffix (`...experts.gate_up_proj`,
`...experts.down_proj`, Qwen's MoE checkpoints): that bank is the experts' Linears and counts as they
would. Both sides are summed per stored module, not matched name by name, so a candidate that stores
per-expert tensors where the original fuses them (or the other way round) still compares.

Compressed-tensors' stored parts are `<module>.weight_packed`, `.weight_scale` and `.weight_zero_point`,
plus an NVFP4 module's two fp32 globals `.weight_global_scale` and `.input_global_scale`; autoawq's GPTQ
layout spells the three common ones `.qweight`, `.scales` and `.qzeros`. Only headers are read, never tensor
data: a hand-in costs seconds whatever the model's size.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from qda_quant.head_lease import HEAD_LEASE, LEASE_BITS
from qda_quant.head_lease import lease_formats as _row_lease
from qda_quant.head_lease import leased_bpw as _leased

from . import UNSCORED, CheckpointError, loaded

if TYPE_CHECKING:
    from qda_quant.format import Format

__all__ = [
    "bpw",
    "bpw_problem",
    "decoder_linears",
    "denominator",
    "head_bits",
    "head_shape",
    "head_weights",
    "kept_width",
    "lease_formats",
    "leased_bpw",
    "linear_names",
    "HEAD_LEASE",
    "LEASE_BITS",
    "planned_bpw",
    "stored_bits",
]

#: A stored module's suffixes, longest first: the shape metadata and the three stored parts of a
#: quantized weight, autoawq's GTPQ spelling, then a Linear kept as it is.
SUFFIXES = (
    ".weight_shape",
    ".weight_packed",
    ".weight_scale",
    ".weight_zero_point",
    ".weight_global_scale",
    ".input_global_scale",
    ".qweight",
    ".qzeros",
    ".scales",
    ".g_idx",
    ".weight",
)
#: The stored parts of a quantized Linear that count toward the budget.
PARTS = (
    ".weight_packed",
    ".weight_scale",
    ".weight_zero_point",
    ".weight_global_scale",
    ".input_global_scale",
    ".qweight",
    ".qzeros",
    ".scales",
)
#: The stored form of a quantized Linear's codes (its presence makes any module a quantized Linear).
CODES = (".weight_packed", ".qweight")
#: The output head: llm-compressor's default keeps it out of its targets (its `ignore` list), and so does the
#: body's count; the head lease prices it on its own.
LM_HEAD = re.compile(r"(^|\.)lm_head$")
ROUTER = re.compile(r"(^|\.)(mlp|mixer)\.gate$")
#: An `nn.Embedding` under any of the architectures' stored names: a model may hold more than one embedding
#: (Gemma-4's per-layer embedding, `embed_tokens_per_layer`), and position embeddings are matrices of the same
#: shape as a Linear. llm-compressor quantizes `nn.Linear` modules, so none of these is in either sum.
EMBEDDING = re.compile(
    r"(^|\.)(embed_tokens|embed_tokens_per_layer|token_embedding|input_embeddings|word_embeddings|"
    r"position_embeddings|embed_positions|embeddings|embedding|wte|wpe|shared)$"
)
#: A fused expert bank: the experts' projections as one stacked tensor, with no `.weight` suffix.
FUSED_BANK = re.compile(r"(^|\.)experts\.(gate_up_proj|down_proj|gate_proj|up_proj)$")


def module(name: str) -> str:
    """The module a stored tensor belongs to: its name without its stored-part suffix."""
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _modules(d: Path | Mapping[str, dict]) -> dict[str, dict[str, dict]]:
    """Stored tensors by module: module -> stored name -> its safetensors header entry.

    `d` is a checkpoint directory, or its headers already read (stored name -> header entry, as a Hub
    repository's range reads give them).
    """
    headers = {name: h for name, (_p, h, _start) in loaded(d).items()} if isinstance(d, Path) else d
    by: dict[str, dict[str, dict]] = {}
    for name, header in headers.items():
        by.setdefault(module(name), {})[name] = header
    return by


def _linear(base: str, stored: dict[str, dict]) -> bool:
    """Whether a stored module is a decoder Linear (or a bank of them) that llm-compressor's default quantizes."""
    if LM_HEAD.search(base) or ROUTER.search(base) or EMBEDDING.search(base):
        return False
    if any(fnmatch.fnmatchcase(base, glob) for glob in UNSCORED):
        return False
    weight = stored.get(base + ".weight")
    if weight is not None:
        return len(weight["shape"]) == 2
    if any(base + codes in stored for codes in CODES):
        return True  # both storage conventions quantize only Linears
    return bool(FUSED_BANK.search(base)) and len(stored[base]["shape"]) >= 2


def _counted(name: str, base: str) -> bool:
    """Whether a stored tensor of a priced module counts: its weight (bf16 or a fused bank) or a stored part."""
    return name == base or name == base + ".weight" or name[len(base) :] in PARTS


def _head(base: str, stored: dict[str, dict]) -> bool:
    """Whether a stored module is the output head: its weight (bf16 or FP8), or the codes of a quantized one."""
    if not LM_HEAD.search(base) or any(fnmatch.fnmatchcase(base, glob) for glob in UNSCORED):
        return False
    weight = stored.get(base + ".weight")
    if weight is not None:
        return len(weight["shape"]) == 2
    return any(base + codes in stored for codes in CODES)


def _bits(d: Path | Mapping[str, dict], which: Callable[[str, dict[str, dict]], bool]) -> int:
    """The stored bits of the modules `which` picks: their codes, scales, zero points and float weights."""
    total = 0
    for base, stored in _modules(d).items():
        if not which(base, stored):
            continue
        for name, header in stored.items():
            if _counted(name, base):
                a = header["data_offsets"]
                total += (a[1] - a[0]) * 8
    return total


def stored_bits(d: Path | Mapping[str, dict]) -> int:
    """The stored bits of the checkpoint's decoder Linears: codes, scales, zero points, bf16-kept weights.

    Raises:
        CheckpointError: `loaded`'s (an unreadable shard, header or index).
    """
    return _bits(d, _linear)


def head_bits(d: Path | Mapping[str, dict]) -> int:
    """The stored bits of the checkpoint's output head (0 when it stores none: a tied head).

    Raises:
        CheckpointError: `loaded`'s (an unreadable shard, header or index).
    """
    return _bits(d, _head)


def linear_names(d: Path | Mapping[str, dict]) -> set[str]:
    """The decoder Linears a checkpoint stores, by module name (`_linear`'s rule, headers only).

    Raises:
        CheckpointError: `loaded`'s (an unreadable shard, header or index).
    """
    return {base for base, stored in _modules(d).items() if _linear(base, stored)}


def decoder_linears(d: Path) -> dict[str, tuple[int, int]]:
    """Each decoder Linear of `d`: its weight count and its output width (`_linear`'s rule, headers only).

    The output width is what tells a scalar gate (a MoE's shared-expert gate, `out_features < 32`, which a
    build keeps bf16) apart from a decoder projection.

    Raises:
        CheckpointError: `loaded`'s (an unreadable shard, header or index).
    """
    out: dict[str, tuple[int, int]] = {}
    for base, stored in _modules(d).items():
        if not _linear(base, stored):
            continue
        shape = (stored.get(base + ".weight") or stored.get(base) or {}).get("shape")
        if not shape or len(shape) < 2:
            continue
        out[base] = (math.prod(shape), shape[-2])
    return out


def denominator(anchor: Path) -> int:
    """The weights of the original's decoder Linears (its `*.weight` / fused banks), the budget's divisor.

    Raises:
        CheckpointError: `loaded`'s, or an original that stores a decoder Linear only quantized (its
            weight shapes are not in its headers; the budget is read against the bf16 original).
    """
    total = 0
    for base, stored in _modules(anchor).items():
        if not _linear(base, stored):
            continue
        shape = (stored.get(base + ".weight") or stored.get(base) or {}).get("shape")
        if shape is None:
            raise CheckpointError(f"{base}: the original stores no weight tensor")
        total += math.prod(shape)
    return total


def head_shape(anchor: Path) -> tuple[int, int] | None:
    """The original's stored output head as ``(rows, cols)``, or None when it stores no head tensor.

    Raises:
        CheckpointError: `loaded`'s, or a head stored without a 2-D weight (a quantized original).
    """
    for base, stored in _modules(anchor).items():
        if not _head(base, stored):
            continue
        shape = (stored.get(base + ".weight") or stored.get(base) or {}).get("shape")
        if shape is None or len(shape) != 2:
            raise CheckpointError(f"{base}: the original stores no 2-D head weight")
        return (shape[0], shape[1])
    return None


def head_weights(anchor: Path) -> int:
    """The original's head weights the lease charges (`head_shape`'s product; 0 when the head is tied).

    Raises:
        CheckpointError: `head_shape`'s.
    """
    shape = head_shape(anchor)
    return shape[0] * shape[1] if shape else 0


def lease_formats(row: str | None) -> tuple[str, ...]:
    """The formats the output head may take under the lease on a row of format `row` (`HEAD_LEASE`).

    Raises:
        ValueError: `row` is None, no format, or a family with no lease.
    """
    if row is None:
        raise ValueError("the head lease needs the run's format")
    return _row_lease(row)


def leased_bpw(candidate: Path | Mapping[str, dict], weights: int, head: int) -> float:
    """The body's bpw after the head's lease: the head pays `LEASE_BITS` a weight, its savings go to the body.

    Args:
        candidate: The checkpoint, or its headers.
        weights: The original's decoder-Linear weight count (`denominator`).
        head: The original's head weight count (`head_shape`'s product; 0 for a tied head).
    """
    return _leased(stored_bits(candidate), head_bits(candidate), weights, head)


def bpw(candidate: Path, anchor: Path) -> float:
    """The candidate's bits per quantized weight of its body, against the original's fixed denominator.

    Raises:
        CheckpointError: `stored_bits`'s or `denominator`'s, or an original with no decoder Linear.
    """
    weights = denominator(anchor)
    if weights == 0:
        raise CheckpointError("the original has no decoder Linear weight")
    return stored_bits(candidate) / weights


def planned_bpw(
    linears: Sequence[tuple[tuple[int, int], Format]],
    weights: int,
    *,
    kept_bits: int = 16,
    kept_name: str = "bf16",
) -> tuple[list[tuple[str, int, int]], float]:
    """Per format class (name, its weights, its stored bits) of a plan, and the plan's total bpw.

    `linears` is `qda_quant`'s `Plan.linears`: the canonical (rows, cols) and format of every tensor the recipe
    packs, one per stored tensor. A packed tensor costs its format's packed bits (`qda_quant.engine.checkpoint_size.
    packed_bits`: codes, scales and zero points, the parts `stored_bits` counts). Every decoder Linear the plan
    leaves unpacked keeps the original's stored width -- `kept_bits` and the dtype's name, which
    `kept_width` reads from the original's headers, so a kept weight costs what `stored_bits` would count in
    the built checkpoint (16 bits for a bf16 source, 32 for an fp32 one); the classes' weights sum to
    `weights`.

    Raises:
        ValueError: the plan packs more weights than the original's decoder Linears have (a recipe whose
            model is not the original).
    """
    from qda_quant.engine.checkpoint_size import packed_bits

    by: dict[str, list[int]] = {}
    packed = 0
    for (rows, cols), f in linears:
        name = f.name
        entry = by.setdefault(name, [0, 0])
        entry[0] += rows * cols
        entry[1] += packed_bits((rows, cols), f)
        packed += rows * cols
    if packed > weights:
        raise ValueError(
            f"the plan packs {packed} weights, more than the original's {weights} decoder-Linear weights"
        )
    if left := weights - packed:
        by[kept_name] = [left, kept_bits * left]
    classes = [(name, n, bits) for name, (n, bits) in by.items()]
    return classes, sum(bits for _, _, bits in classes) / weights


def kept_width(anchor: Path) -> tuple[int, str]:
    """The bits a weight and the dtype name the original stores its decoder Linears at, when they all agree.

    A plan's unpacked weights keep the original's stored tensors, so their width is the original's, not the
    16 bits of a bf16 assumption: an fp32 original keeps fp32.

    Raises:
        CheckpointError: `loaded`'s, an original with no decoder Linear weight, or an original whose decoder
            Linears are stored at more than one width (the plan cannot price the kept part then).
    """
    widths: dict[tuple[int, str], set[str]] = {}
    for base, stored in _modules(anchor).items():
        if not _linear(base, stored):
            continue
        header = stored.get(base + ".weight") or stored.get(base)
        if header is None or (shape := header.get("shape")) is None:
            raise CheckpointError(f"{base}: the original stores no weight tensor")
        a = header["data_offsets"]
        n = math.prod(shape)
        bits, rest = divmod((a[1] - a[0]) * 8, n)
        if rest:
            raise CheckpointError(f"{base}: {a[1] - a[0]} stored bytes is not whole bits a weight")
        widths.setdefault((bits, str(header.get("dtype", "")).lower()), set()).add(base)
    if not widths:
        raise CheckpointError("the original has no decoder Linear weight")
    if len(widths) > 1:
        seen = ", ".join(f"{dtype} ({bits} bits)" for bits, dtype in sorted(widths))
        raise CheckpointError(
            f"the original stores its decoder Linears at more than one width ({seen}); the plan cannot "
            "price the weights a recipe leaves unpacked"
        )
    (bits, dtype), _ = next(iter(widths.items()))
    return bits, dtype


def bpw_problem(candidate: Path, weights: int, budget: float, *, lease: int | None = None) -> str | None:
    """Why the candidate is over the budget in bits per quantized weight, or None.

    Args:
        candidate: The checkpoint.
        weights: The original's decoder-Linear weight count (`denominator`).
        budget: The largest bpw the task admits (`judge.budget_bpw`).
        lease: The original's head weight count when `judge.head_lease` is on (`leased_bpw`); None: the
            body alone is priced.
    """
    try:
        body = stored_bits(candidate) / weights
        value = body if lease is None else leased_bpw(candidate, weights, lease)
    except CheckpointError as e:
        return f"not a readable checkpoint ({e})"
    if value <= budget:
        return None
    if lease is None:
        return f"{value:.3f} bits per quantized weight, over the judge's budget of {budget:g} bpw"
    return (
        f"{value:.3f} bits per quantized weight after the head's lease (the body alone {body:.3f}), over the "
        f"judge's budget of {budget:g} bpw"
    )
