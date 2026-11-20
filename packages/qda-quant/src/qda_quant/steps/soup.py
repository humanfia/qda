"""Scale soup (``soup``): average several packed checkpoints that share their codes.

Every input must have the same format, the same quantized modules, and byte-identical codes and zero points.
Per group the scale is the (weighted) mean in log space, ``exp(sum_i w_i log s_i)``, computed from each
input's exact training state (``qda_state``: ``s_i = s0_i * exp(delta_i)``, fp32). Scales may be negative
(``signed8``): the mean is of magnitudes, and inputs whose signs differ for a group are refused. Float tensors
that an input's state lists as trained (norm gains, Gated DeltaNet parameters) are averaged arithmetically from
their fp32 masters. Any other tensor that differs between inputs is refused, by name.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from pydantic import ConfigDict, Field

from ..engine import packed as P
from ..step import Refused, Step


class Soup(Step):
    """Log-mean of scales over the build's input and the checkpoints in `with_`.

    Reproducibility: bit-identical (deterministic CPU math given its input checkpoints; no GPU, no gradients;
    docs/reproduce.md).

    Attributes:
        with_: the other checkpoints (YAML key ``with``).
        weights: one weight per input (the build's input first); default equal weights. They are normalized.
    """

    name = "soup"
    kind = "checkpoint"

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    with_: list[str] = Field(default_factory=list, alias="with")
    weights: list[float] | None = None

    def check_checkpoint(self, ctx: P.CheckpointPlan) -> None:
        """Refuse before any stage runs: `with_` must name packed checkpoints of the build's format; weights valid."""
        if not self.with_:
            raise Refused("soup: needs at least one other checkpoint in `with`")
        n = len(self.with_) + 1
        w = self.weights or [1.0] * n
        if len(w) != n or min(w) < 0 or sum(w) <= 0:
            raise Refused(f"soup: {len(w)} weights for {n} checkpoints (non-negative, not all zero)")
        for p in self.with_:
            if not P.is_packed(p):
                raise Refused(f"soup: {p} is not a packed checkpoint")
            if P.Packed(p).fmt != ctx.fmt:
                raise Refused(f"soup: {p} is {P.Packed(p).fmt.name}, the build {ctx.fmt.name}")

    def apply_checkpoint(self, src: Path, out: Path, run: P.CheckpointRun) -> None:
        """Average `src` and `with_` into `out` (`check_checkpoint` has passed)."""
        t0 = time.perf_counter()
        cks = [P.Packed(src), *(P.Packed(p) for p in self.with_)]
        w = self.weights or [1.0] * len(cks)
        w = [x / sum(w) for x in w]
        base = cks[0]
        for c in cks[1:]:
            _same_codes(base, c)
        states = [c.state(run.log) for c in cks]
        trained = set().union(*(f for _, f in states))
        for c in cks[1:]:
            _same_floats(base, c, trained)
        scales = {}
        for s in base.stems:
            s0 = states[0][0][s][0]
            if any(((st[0][s][0] < 0) != (s0 < 0)).any() for st in states[1:]):
                raise Refused(f"soup: the scales of {s} differ in sign between inputs")
            logs = sum(
                wi * (st[0][s][0].float().abs().log() + st[0][s][1].float())
                for wi, st in zip(w, states, strict=True)
            )
            d = logs - s0.float().abs().log()
            # A zero starting scale stays zero instead of subtracting two infinities.
            scales[s] = (s0, torch.where(s0 == 0, torch.zeros_like(d), d))
        floats = {}
        for k in sorted(trained):
            parts = [
                st[1][k] if k in st[1] else c.tensor(k).float() for c, st in zip(cks, states, strict=True)
            ]
            floats[k] = sum(wi * t.float() for wi, t in zip(w, parts, strict=True))
        stage = {"step": self.name, "inputs": [str(c.path) for c in cks], "weights": w,
                 "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        P.write(base, out, scales, floats, stage, base.teacher() or run.teacher, run.devices)


def _same_codes(a: P.Packed, b: P.Packed) -> None:
    """Refuse unless `b` has `a`'s format, modules and byte-identical codes / zero points / shapes."""
    if a.fmt != b.fmt or a.stems != b.stems:
        raise Refused(f"soup: {b.path} is not {a.path}'s format and modules")
    for s in a.stems:
        # an integer or NVFP4 module's codes are its `weight_packed`; an FP8 module's are its `.weight`
        code = (".weight",) if a.format_of(s).weight_type == "fp8" else (".weight_packed",)
        for suf in (*code, ".weight_zero_point", ".weight_shape"):
            k = s + suf
            if (k in a.where) != (k in b.where) or (
                k in a.where and not torch.equal(a.tensor(k), b.tensor(k))
            ):
                raise Refused(f"soup: {k} differs between {a.path} and {b.path}; a soup needs shared codes")


def _same_floats(a: P.Packed, b: P.Packed, trained: set[str]) -> None:
    """Refuse if a float tensor that no input lists as trained differs between `a` and `b`."""
    quant = (".weight_packed", ".weight_zero_point", ".weight_shape", ".weight_scale")
    if set(a.where) != set(b.where):
        raise Refused(f"soup: {b.path} has other tensors than {a.path}")
    for k in a.where:
        if k in trained or k.endswith(quant) or _is_codes(a, k):
            continue
        if not torch.equal(a.tensor(k), b.tensor(k)):
            raise Refused(
                f"soup: {k} differs between {a.path} and {b.path} and no input records it as trained"
            )


def _is_codes(a: P.Packed, key: str) -> bool:
    """True for an FP8 module's ``weight``: its code tensor, compared with the other inputs by `_same_codes`."""
    stem = a.stem_of(key)
    return stem is not None and a.format_of(stem).weight_type == "fp8"
