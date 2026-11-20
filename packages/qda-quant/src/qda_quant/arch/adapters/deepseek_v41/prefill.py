"""Target-derived prefill numerics: vLLM's sm90 path and DeepSeek's NVFP4 reference path.

vLLM 0.33 uses different native attention-cache layouts on sm90 and sm100.
The build target selects these numerics; the model has no implicit serving target.
"""

from typing import Any, Protocol, cast

import torch

from ....step import Refused
from ....target import Target
from ..prefill import prefill_for as _select
from .numerics import cache_roundtrip


class PrefillNumerics(Protocol):
    """Target support and native activation round trips supplied by a prefill plugin."""

    arch: str

    def supports(self, target: Target) -> bool:
        """Whether this implementation serves the configured target."""
        ...

    def kv(self, x: torch.Tensor, cfg: Any) -> torch.Tensor:
        """Round sliding-window KV as the target cache stores it."""
        ...

    def compressed(self, x: torch.Tensor, cfg: Any) -> torch.Tensor:
        """Round compressed KV as the target cache stores it."""
        ...

    def index(self, x: torch.Tensor) -> torch.Tensor:
        """Round sparse-indexer keys or queries."""
        ...

    def output(self, x: torch.Tensor) -> torch.Tensor:
        """Round activations consumed by the grouped output projection."""
        ...


class Sm90Prefill:
    """vLLM FlashMLA on sm90: FP8 NoPE in groups of 64, bf16 RoPE, FP8 index heads."""

    arch = "deepseek_v41"

    def supports(self, target: Target) -> bool:
        """Serve the pinned vLLM sm90 implementation, independently of the weight-only backend."""
        return (target.arch, target.vllm) == ("sm90", "0.33.0")

    def kv(self, x, cfg):
        """Round the NoPE region; the cache preserves its RoPE tail in bf16."""
        rd = cfg.qk_rope_head_dim
        return torch.cat((cache_roundtrip(x[..., :-rd], 8, 64), x[..., -rd:]), -1)

    def index(self, x):
        """Round each index head using one power-of-two FP8 scale."""
        return cache_roundtrip(x, 8, x.shape[-1])

    def compressed(self, x, cfg):
        """Compressed and sliding-window KV share the same sm90 record."""
        return self.kv(x, cfg)

    def output(self, x):
        """The kept bf16 grouped output projection reads bf16 activations in vLLM."""
        return x


class ReferencePrefill:
    """DeepSeek's reference: MXFP8 window, NVFP4 compressed KV and MXFP4 index heads."""

    arch = "deepseek_v41"

    def supports(self, target: Target) -> bool:
        """Use reference numerics only for an explicitly requested reference target."""
        return (target.arch, target.vllm) == ("reference", "0")

    def kv(self, x, cfg):  # noqa: ARG002
        """Round all dimensions, including the RoPE tail, in MXFP8 groups of 32."""
        return cache_roundtrip(x, 8, 32)

    def index(self, x):
        """Round index heads in MXFP4 groups of 32."""
        return cache_roundtrip(x, 4, 32)

    def compressed(self, x, cfg):  # noqa: ARG002
        """Round compressed latents in NVFP4 groups of 16 with E4M3 scales."""
        return cache_roundtrip(x, 4, 16, e4m3_scale=True)

    def output(self, x):
        """The reference rounds the grouped output's input in MXFP8 groups of 32."""
        return cache_roundtrip(x, 8, 32)


SM90_PREFILL = Sm90Prefill()
REFERENCE_PREFILL = ReferencePrefill()

#: Why a V4.1 build for sm120 is refused: a build and `qda plan` print it, the support matrix and the gpu tests
#: quote it.
SM120_REFUSAL = (
    "DeepSeek V4.1 does not run on an RTX PRO 6000 (sm120): vLLM 0.33's V4.1 indexer fails there in DeepGEMM's "
    "get_paged_mqa_logits_metadata (block_kv must be 32 or 64), and QDA has V4.1's prefill numerics for Hopper "
    "only; serve it on an H100 or H200 (build with --set target=h100)"
)


def prefill_for(target: Target) -> tuple[str, PrefillNumerics]:
    """Select exactly one V4.1 plugin declaring support for the YAML serving target.

    Raises:
        Refused: no plugin, or more than one, supports the target; for sm120 the reason is `SM120_REFUSAL`.
    """
    if target.arch == "sm120":
        raise Refused(SM120_REFUSAL)
    name, plugin = _select("deepseek_v41", target)
    return name, cast(PrefillNumerics, plugin)
