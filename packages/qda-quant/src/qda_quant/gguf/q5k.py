"""Q5_K: Q4_K's asymmetric group products with five-bit codes and a native ggml fit."""

from __future__ import annotations

from ..target import Q5K as Q5K_FORMAT
from .kernel import q5k_fields, quantize_q5k
from .q4k import Q4K


class Q5K(Q4K):
    """Eight 32-value groups per block, unsigned 0..31 codes, and Q4_K's fp16 × six-bit products."""

    name = "q5k"
    gptq_fused = False
    stores = Q5K_FORMAT
    fields = staticmethod(q5k_fields)
    records = staticmethod(quantize_q5k)
    weighted_records = staticmethod(quantize_q5k)

    def refusal(self, fmt, plan) -> str | None:  # noqa: ARG002
        """Only Q5_K's five-bit asymmetric format."""
        if fmt.name != self.stores:
            return f"Q5_K stores 5-bit asymmetric codes in groups of 32: format {self.stores}, not {fmt.name}"
        return None


Q5K_GRID = Q5K()
