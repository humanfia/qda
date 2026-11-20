"""Choose weight and activation placement within each rank’s memory budget."""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ..exit import ENVIRONMENT
from ..step import Refused
from .stores import WEIGHT_STORES, MemoryNeed, WeightStore

GiB = 2**30
#: free memory of a device with no ceiling: a CPU build, or a host without ``/proc/meminfo``
_UNBOUNDED = 1 << 50
#: headroom a plan keeps on the least free card: `HEADROOM_SHARE` of its free memory, at least `HEADROOM_MIN`.
#: The peak counts allocated tensors only, so the allocator's reserve, the CUDA context and fragmentation come
#: on top (`docs/recipe.md`); a plan that would use this memory picks the next weight store instead.
HEADROOM_SHARE = 0.03
HEADROOM_MIN = 2 * GiB
#: what `nvidia-smi` shows on the card beyond the allocator's reserve: the CUDA context and the kernels and library
#: workspaces a build loads. Measured 11-14 as card used minus torch's reserved memory on six one-card builds
#: (Qwen3-0.6B, Qwen3-8B, Llama-3.1-8B; resident and streamed): 0.78-0.79 GiB on every one.
CUDA_CONTEXT = int(0.8 * GiB)
#: the caching allocator's segments beyond the allocated peak, as a share of it. With expandable segments
#: (`ALLOC_CONF`, a build's default) the six builds reserved at most 7.2 % more than they allocated (Qwen3-8B
#: streamed: 7.11 allocated, 7.62 reserved); with the fixed-size segments torch uses otherwise, up to 36 % more
#: (Llama-3.1-8B streamed: 8.66 allocated, 11.75 reserved).
RESERVE_SHARE = {True: 0.08, False: 0.40}
#: the allocator setting a build runs with unless the environment names one (`use_expandable_segments`): the
#: same tensors cost 2.6 GiB less on the card on a streamed Llama-3.1-8B build (12.5 -> 9.9 GiB)
ALLOC_CONF = "expandable_segments:True"
#: torch reads the CUDA-specific name first (``c10/cuda/CUDAAllocatorConfig.h``); so does `expandable_segments`
_ALLOC_ENV = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
#: whether `use_expandable_segments` failed to set the allocator in this process (it then warns)
_SET_FAILED: list[bool] = []


def _alloc_env() -> str | None:
    # torch goes by presence: an empty CUDA-specific variable still overrides the generic one
    return next((os.environ[k] for k in _ALLOC_ENV if k in os.environ), None)


def expandable_segments() -> bool:
    """Whether a build in this environment runs with expandable segments: qda's default, or the environment's."""
    given = _alloc_env()
    if given is not None:
        return "expandable_segments:true" in given.replace(" ", "").lower()
    return not _SET_FAILED and _allocator_setter() is not None


def _allocator_setter():  # noqa: ANN202
    return getattr(torch._C, "_accelerator_setAllocatorSettings", None) or getattr(
        torch.cuda.memory, "_set_allocator_settings", None
    )


def use_expandable_segments() -> None:
    """Run this process's CUDA allocator with `ALLOC_CONF`, unless the environment names a setting of its own.

    Called before a build's first CUDA allocation (the build and each card process); the environment is left as
    it is, so a process the build starts inherits nothing from it.
    """
    if _alloc_env() is not None:
        return
    setting = _allocator_setter()
    try:
        if setting is None:  # a torch without either hook keeps its own default
            raise RuntimeError("this torch has no allocator-settings hook")
        setting(ALLOC_CONF)
    except RuntimeError as e:
        _SET_FAILED.append(True)  # `expandable_segments` now prices the fixed-segment reserve
        warnings.warn(
            f"could not set the CUDA allocator to {ALLOC_CONF} ({e}); the card peak may be higher "
            "than `qda plan` printed: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            stacklevel=2,
        )


@dataclass(frozen=True)
class Term:
    """One part of a rank's peak, in bytes."""

    name: str
    bytes: int


@dataclass(frozen=True)
class Solve:
    """One unit's solve: the experts it holds, one expert's workspace, and the unit's write-back buffers.

    The step solves the unit whole when `experts x workspace` fits the card; otherwise it allocates `buffers`
    first and solves as many experts at a time as still fit (`qda_quant.solve.solve.chunk_experts`). The two terms
    never hold at once: the whole solve's bound includes the fp32 served copy it writes back. An expert bank's
    short chunk is factored at `factor_batch` members (`qda_quant.solve.solve.MIN_FACTOR_BATCH`); a dense group is 1.
    """

    name: str
    experts: int
    workspace: int
    factor_batch: int
    buffers: int = 0
    joint: bool = False  # trained beside every other joint unit, whole: see `_peak`


@dataclass(frozen=True)
class Residency:
    """The plan, also recorded in ``build.json``."""

    weights: str  # a `engine.stores.WEIGHT_STORES` name
    activations: Literal["device", "host"]
    device: str  # the least free card, the one the plan was made against
    model_gib: float
    act_gib: float
    free_gib: float
    peak_gib: float
    peak_terms: tuple[tuple[str, float], ...]  # the peak's terms in GiB, largest first
    expert_chunks: int = 1  # chunks of experts a layer's expert statistics are gathered and solved in
    #: what `nvidia-smi` shows at the build's peak: the allocated peak, the CUDA context and the allocator's
    #: reserve (`card_peak`), rounded up to 0.1 GiB; the number `qda plan` prints and ``build.json`` records
    card_gib: float = 0.0
    card_terms: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict:
        """Plain data for ``build.json``."""
        return {
            "weights": self.weights,
            "activations": self.activations,
            "device": self.device,
            "model_gib": self.model_gib,
            "act_gib": self.act_gib,
            "free_gib": self.free_gib,
            "peak_gib": self.peak_gib,
            "peak_terms": dict(self.peak_terms),
            "card_peak_gib": self.card_gib,
            "card_terms": dict(self.card_terms),
            "expert_chunks": self.expert_chunks,
        }

    def __str__(self) -> str:
        """One readable line: where weights and activations live, the card's peak, the tensors' peak and its terms."""
        free = "cpu" if self.free_gib * GiB >= _UNBOUNDED else f"{self.free_gib:.1f} GiB on {self.device}"
        terms = [t for t in self.peak_terms if t[1] >= 0.05] or self.peak_terms[:1]
        top = [f"{name} {gib:.1f}" for name, gib in terms[:4]]
        rest = sum(gib for _, gib in terms[4:])
        if rest >= 0.05:
            top.append(f"other {rest:.2f}")
        card = " + ".join(f"{name} {gib:.1f}" for name, gib in self.card_terms)
        return (
            f"weights {self.weights}, activations {self.activations}, model {self.model_gib} GiB, "
            f"free {free}, card peak (nvidia-smi) {self.card_gib:.1f} GiB: {card}; "
            f"tensor peak {self.peak_gib:.1f} GiB: {' + '.join(top)}"
        )


def card_peak(tensors: int, expandable: bool | None = None) -> tuple[float, tuple[tuple[str, float], ...]]:
    """What `nvidia-smi` shows at a build's peak of `tensors` allocated bytes, in GiB rounded up to 0.1, and its terms.

    The allocator holds segments beyond the allocated peak (`RESERVE_SHARE`, by whether the build runs with
    expandable segments: `expandable_segments`), and the card holds the CUDA context besides (`CUDA_CONTEXT`).
    """
    share = RESERVE_SHARE[expandable_segments() if expandable is None else expandable]
    terms = (
        ("tensors", tensors / GiB),
        ("CUDA context", CUDA_CONTEXT / GiB),
        ("allocator reserve", share * tensors / GiB),
    )
    total = math.ceil(round(sum(gib for _, gib in terms), 6) * 10) / 10
    return total, tuple((name, round(gib, 2)) for name, gib in terms)


def free_bytes(device: torch.device) -> int:
    """Free memory of `device`; the CPU reports no ceiling (a CPU build always goes resident)."""
    return _UNBOUNDED if device.type != "cuda" else int(torch.cuda.mem_get_info(device)[0])


def _readable_free(device: torch.device) -> int:
    """`free_bytes` of a card the plan fits against; a card too full to open a CUDA context on is refused."""
    try:
        return free_bytes(device)
    except RuntimeError as e:  # torch's AcceleratorError and OutOfMemoryError are RuntimeErrors
        raise Refused(
            f"{device} cannot report its free memory: {(str(e) or type(e).__name__).splitlines()[0]}",
            f'free {device} (another process holds it), or plan on the CPU: CUDA_VISIBLE_DEVICES="" with no card'
            " in devices",
            code=ENVIRONMENT,
        ) from e


def headroom(device: torch.device, free: int) -> int:
    """Free memory a plan keeps on `device`: 3 % of `free`, at least 2 GiB; none on a CPU device."""
    return 0 if device.type != "cuda" else max(HEADROOM_MIN, int(free * HEADROOM_SHARE))


def plan_headroom(device: torch.device) -> int:
    """The headroom a plan keeps on `device`, read off the card's total memory.

    `headroom` is computed from the memory free when the plan is made, which a step that sizes its own
    allocation later does not see; the same rule on the card's total is the same reserve, since a plan is made
    on a card no other build of this process is using.
    """
    if device.type != "cuda":
        return 0
    return headroom(device, int(torch.cuda.get_device_properties(device).total_memory))


def _room(free: int, reserve: int) -> str:
    """How much of `free` a plan may use, for a refusal message."""
    if not reserve:
        return f"above its {free / GiB:.1f} GiB free"
    return (
        f"above the {(free - reserve) / GiB:.1f} GiB it may use "
        f"({free / GiB:.1f} GiB free, {reserve / GiB:.1f} GiB headroom)"
    )


def plan(
    model_bytes: int,
    layer_bytes: int,
    act_bytes: int,
    devices: Sequence[torch.device],
    terms: Sequence[Term] = (),
    solves: Sequence[Solve] = (),
    force: str | None = None,
    retained_bytes: int = 0,
    host_free: int | None = None,
    least_memory: bool = False,
    model_step_bytes: int = 0,
) -> Residency:
    """Choose where weights and activations live, and refuse a peak the least free card cannot hold.

    `act_bytes` is the activation memory that follows the hidden states: the states of the calibration samples
    and, when a step needs it, the float stream. `retained_bytes` is activation memory that stays on the card
    either way: the layer outputs a step keeps for every batch. It is reserved separately, because offloading
    the hidden states does not free it, so it still competes with the weights.

    The chosen peak leaves `headroom(device, free)` unused: a store that would use it does not fit the budget,
    so the next store is taken; when none does, the one that asks least of the card chunks its solves inside the
    room the headroom leaves. An explicit `force` runs at the full room and warns when it uses the headroom.

    Args:
        model_bytes: total parameter bytes as loaded (bf16).
        layer_bytes: bytes of the largest decoder layer.
        act_bytes: bytes of the activation streams that move with the hidden states.
        devices: the build's cards; every rank carries the same weights and activations, so the planner fits
            against the card with the least free memory.
        terms: the rest of the rank's peak (`engine.planning.memory_terms`): statistics, calibration
            transients and the fold's receive buffer.
        solves: the quantizing steps' solves; each one's workspace fills what the card leaves, up to its
            whole size, so a bank larger than the card is solved in chunks rather than refused.
        force: a weight store's name to override the choice; it may use the headroom, and warns when it does.
        retained_bytes: bytes of activation memory the build always keeps on the card.
        host_free: available host memory (default: `host_available`).
        least_memory: take the last store that fits in the planner's order, which keeps the least in memory,
            instead of the first. The engine asks for it when the steps keep no whole-model state (`needs`
            empty), where no store's extra state buys anything; a store that refuses the plan later hands
            over to the default choice (`engine.planning._store_peak`).
        model_step_bytes: what the model steps hold beyond the weights (`Step.model_step_bytes`). They run with
            the whole model on the card only when it is resident; that stage's peak is the weights plus this, and
            the build's peak is the larger of it and the layer stage's.

    Raises:
        Refused: the store name is unknown (`USAGE`: the message lists the known stores), or the least free
            card cannot hold the predicted peak (`ENVIRONMENT`: the message names the largest term, and the
            same build runs once the card is free), or a card is too full to report its free memory
            (`ENVIRONMENT`: the message quotes CUDA's first line).
    """
    device, free = min(((d, _readable_free(d)) for d in devices), key=lambda pair: pair[1])
    keep = headroom(device, free)  # an explicit store may use it, and warns
    host = host_available() if host_free is None else host_free

    def at(reserve: int) -> Residency:
        return _plan_at(
            model_bytes,
            layer_bytes,
            act_bytes,
            devices,
            terms,
            solves,
            force,
            retained_bytes,
            host,
            least_memory,
            model_step_bytes,
            device,
            free,
            keep,
            reserve,
        )

    res = at(0 if force is not None else keep)
    share = RESERVE_SHARE[expandable_segments()]
    over_card = device.type == "cuda" and res.peak_gib * GiB * (1 + share) + CUDA_CONTEXT > free
    if over_card and force is None:
        # the headroom did not cover this plan's CUDA context and allocator reserve: plan again keeping free what
        # the card peak needs beside the tensors at the most they may then reach, `(free - context) / (1 + share)`,
        # so any placement within it fits the card. `free` already excludes the context this process holds, so the
        # second plan errs high by up to that context (~0.5 GiB)
        res = at(max(keep, free - int((free - CUDA_CONTEXT) / (1 + share))))
    elif over_card:
        warnings.warn(
            f"residency {res.weights}: the card peak (nvidia-smi) is {res.card_gib:.1f} GiB, above the "
            f"{free / GiB:.1f} GiB free on {device}; the explicit residency is honoured",
            stacklevel=2,
        )
    return res


def _plan_at(
    model_bytes: int,
    layer_bytes: int,
    act_bytes: int,
    devices: Sequence[torch.device],
    terms: Sequence[Term],
    solves: Sequence[Solve],
    force: str | None,
    retained_bytes: int,
    host: int,
    least_memory: bool,
    model_step_bytes: int,
    device: torch.device,
    free: int,
    keep: int,
    reserve: int,
) -> Residency:
    """`plan` keeping `reserve` of the card's `free` bytes unused."""
    other = sum(t.bytes for t in terms)
    # the padded factorization of a short chunk, plus its buffers, always has to fit
    least = max((s.factor_batch * s.workspace + s.buffers for s in solves), default=0)
    acts_on_device = act_bytes + retained_bytes + other + least + layer_bytes <= free - reserve
    budget = free - retained_bytes - other - least - reserve - (act_bytes if acts_on_device else 0)
    need = MemoryNeed(model_bytes, layer_bytes, budget, host, len(devices))

    def model_stage(store: WeightStore) -> int:
        """A resident build's model steps run with the whole model on the card: weights plus their transients."""
        return (
            store.card_bytes(need) + model_step_bytes if model_step_bytes and store.name == "resident" else 0
        )

    stores = [s for _, s in WEIGHT_STORES.items()]
    fitting = [s for s in stores if s.fits(need) and (not model_stage(s) or model_stage(s) <= free - reserve)]
    if force is not None:
        store = WEIGHT_STORES[force]  # an unknown name fails here, listing the known stores
    elif fitting:
        # priority runs from the store that holds the most in memory to the one that holds the least
        store = (max if least_memory else min)(fitting, key=lambda s: s.priority)
    else:  # nothing fits: refuse with the peak of the store that asks least of the card
        store = min(stores, key=lambda s: (s.card_bytes(need), s.priority))
    if model_stage(store) and model_stage(store) > free - reserve:
        raise Refused(
            f"{device}: the build needs {model_stage(store) / GiB:.1f} GiB while its model steps run on the "
            f"resident weights, {_room(free, reserve)}; its model steps hold {model_step_bytes / GiB:.1f} GiB "
            "beside the weights (residency: stream runs them on the host)",
            code=ENVIRONMENT,
        )
    at_peak = [Term("weights", store.card_bytes(need)), *terms]
    if retained_bytes:
        at_peak.append(Term("retained outputs", retained_bytes))
    if acts_on_device and act_bytes:
        at_peak.append(Term("activations", act_bytes))
    fixed = sum(t.bytes for t in at_peak)
    if fixed > free - reserve:
        largest = max(at_peak, key=lambda t: t.bytes)
        raise Refused(
            f"{device}: the build needs {fixed / GiB:.1f} GiB at its peak, {_room(free, reserve)}; "
            f"its largest term is the {largest.name} ({largest.bytes / GiB:.1f} GiB)",
            code=ENVIRONMENT,
        )
    total, solver, buffers = _peak(fixed, free, solves, device, reserve)
    if solver is not None:
        at_peak.extend((solver, Term("write-back buffers", buffers)))
    top = sorted((t for t in at_peak if t.bytes), key=lambda t: -t.bytes)
    if model_stage(store) > total:
        total, top = (
            model_stage(store),
            [Term("weights", store.card_bytes(need)), Term("model steps", model_step_bytes)],
        )
    if force is not None and total > free - keep:
        warnings.warn(
            f"residency {store.name}: the plan peaks at {total / GiB:.1f} GiB, within the {keep / GiB:.1f} GiB "
            f"headroom on the {free / GiB:.1f} GiB free card; the explicit residency is honoured",
            stacklevel=3,
        )
    card, card_terms = card_peak(total)
    return Residency(
        weights=store.name,
        activations="device" if acts_on_device else "host",
        device=str(device),
        model_gib=round(model_bytes / GiB, 2),
        act_gib=round((act_bytes + retained_bytes) / GiB, 2),
        free_gib=round(free / GiB, 2),
        peak_gib=round(total / GiB, 2),
        peak_terms=tuple((t.name, round(t.bytes / GiB, 2)) for t in top),
        card_gib=card,
        card_terms=card_terms,
    )


def _peak(
    fixed: int, free: int, solves: Sequence[Solve], device: torch.device, reserve: int = 0
) -> tuple[int, Term | None, int]:
    """The peak, its solver term and the other units' write-back buffers that term sits beside.

    One unit is solved at a time, so the other units' write-back buffers stay on the card beside it, while
    their workspaces do not. A chunk is sized by memory alone, but its factorization pads it to `factor_batch`
    members, so the padded bound is what has to fit. A `joint` solve is the exception on both counts: a step
    that trains every unit of the layer in one optimizer (`engine.memory._solves`) builds them all before it
    trains any and never cuts one itself, so the joint units' whole workspaces are held together and none of
    them is chunked. `reserve` is left unused on top of the peak.
    """
    joint = sum(o.experts * o.workspace for o in solves if o.joint)
    peak, term, held_others = fixed, None, 0
    for s in solves:
        if not s.workspace:
            continue
        others = sum(o.buffers for o in solves if o is not s)
        if s.joint:
            count, held = s.experts, joint
        else:
            room = free - reserve - fixed - others - joint
            fit = max(0, room - s.buffers) // s.workspace
            count = min(s.experts, max(1, fit))
            pad = max(count, s.factor_batch) * s.workspace
            held = pad if count == s.experts else s.buffers + pad
        if held > free - reserve - fixed - others:
            need = fixed + others + held
            span = "the whole unit" if count == s.experts else f"{count} of {s.experts} experts at a time"
            raise Refused(
                f"{device}: the build needs {need / GiB:.1f} GiB at its peak, {_room(free, reserve)}; "
                f"its {s.name} needs {held / GiB:.1f} GiB for {span}",
                code=ENVIRONMENT,
            )
        if fixed + others + held > peak:
            name = s.name if count == s.experts else f"{s.name}, {count} of {s.experts} at a time"
            peak, term, held_others = fixed + others + held, Term(name, held), others
    return peak, term, held_others


def host_available() -> int:
    """Host memory available to a new allocation without swapping (Linux ``MemAvailable``), in bytes.

    Without ``/proc/meminfo`` the host counts as unbounded, like a CPU device's memory.
    """
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return _UNBOUNDED
