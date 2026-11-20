"""The head lease: the one rule by which a narrower output head lends its savings to the body's bpw budget.

A budget in bits per quantized weight counts the decoder Linears only (the body). Under the lease the output
head is charged ``LEASE_BITS`` a weight up front and may be stored in a format its row offers (`HEAD_LEASE`);
what a narrower head saves is lent to the body, so the total stays the same:
``body bits + head bits <= budget * body weights + LEASE_BITS * head weights``.

The judge (`qda_agent.judge.checks.budget`) and `qda fisher allocate --head-lease` both read this module.
"""

from __future__ import annotations

__all__ = ["HEAD_LEASE", "LEASE_BITS", "family", "lease_formats", "leased_bpw", "limit_bits"]

#: The bits a head weight is charged up front under the lease, whatever the original stores.
LEASE_BITS = 16
#: The formats the output head may be stored in under the lease, by the row's family: the weight-only integer
#: heads a `head` step builds beside that body. W4A8 offers no 4-bit head until one serves beside its FP8
#: activations; a W4A4 row offers an FP8 head.
HEAD_LEASE = {
    "w4a16": ("bf16", "w8a16", "w4a16"),
    "w3a16": ("bf16", "w8a16", "w4a16"),
    "w4a8": ("bf16", "w8a16"),
    "w4a4": ("bf16", "fp8"),
}


def family(fmt: str) -> str:
    """The `HEAD_LEASE` key of a format spelling: ``w4a16-asym-g128`` -> ``w4a16``, ``w4a8`` -> ``w4a8``,
    ``w4a4-nvfp4`` -> ``w4a4``, ``w8a8-fp8`` -> ``fp8``, ``bf16``."""
    from .format import Format

    if fmt.strip().lower() == "bf16":
        return "bf16"
    f = Format.parse(fmt)
    if f.weight_type == "fp8":  # FP8 W8A8: the W4A4 row's FP8 head
        return "fp8"
    if f.act == "nvfp4_block":
        return "w4a4"
    return f"w{f.weight_bits}a{16 if f.act == 'bf16' else 8 if f.act == 'fp8_token' else f.act}"


def lease_formats(row: str) -> tuple[str, ...]:
    """The format families the head may take under the lease on a row of format `row` (`HEAD_LEASE`).

    Raises:
        ValueError: `row` is no format, or a family with no lease.
    """
    key = family(row)
    if key not in HEAD_LEASE:
        raise ValueError(f"no head lease for a {key} row (one of {', '.join(HEAD_LEASE)})")
    return HEAD_LEASE[key]


def limit_bits(budget: float, body_weights: int, head_weights: int) -> float:
    """The bits the body and the head may store together under the lease."""
    return budget * body_weights + LEASE_BITS * head_weights


def leased_bpw(body_bits: float, head_bits: float, body_weights: int, head_weights: int) -> float:
    """The body's bpw after the lease: the head pays `LEASE_BITS` a weight and its savings go to the body."""
    return (body_bits + head_bits - LEASE_BITS * head_weights) / body_weights
