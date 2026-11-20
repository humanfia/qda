"""Target-selected prefill numerics: one registry for every architecture's plugins.

A plugin names the architecture adapter it serves (``arch``) and declares the build targets it reproduces
(``supports``). The adapter binds the single plugin that matches; none or several is a plan-time refusal.
"""

from typing import Protocol

from ...registry import Registry
from ...step import Refused
from ...target import Target


class PrefillPlugin(Protocol):
    """What every prefill plugin declares; each architecture adds the numerics its model calls."""

    arch: str

    def supports(self, target: Target) -> bool:
        """Whether this implementation reproduces the serving numerics of `target`."""
        ...


def _check(key: str, obj: object) -> None:
    if not (isinstance(getattr(obj, "arch", None), str) and callable(getattr(obj, "supports", None))):
        raise TypeError(
            f"prefill numerics plugin {key!r} is {obj!r}, without an arch name and supports(target)"
        )


PREFILL_NUMERICS: Registry[PrefillPlugin] = Registry(
    "prefill_numerics",
    {
        "deepseek_v41_sm90": "qda_quant.arch.adapters.deepseek_v41.prefill:SM90_PREFILL",
        "deepseek_v41_reference": "qda_quant.arch.adapters.deepseek_v41.prefill:REFERENCE_PREFILL",
    },
    on_add=_check,
)


def prefill_for(arch: str, target: Target) -> tuple[str, PrefillPlugin]:
    """The one plugin of architecture `arch` that supports `target`, with its registry name.

    Raises:
        Refused: no plugin, or more than one, supports the target.
    """
    matches = [(n, p) for n, p in PREFILL_NUMERICS.items() if p.arch == arch and p.supports(target)]
    if len(matches) != 1:
        known = [n for n, p in PREFILL_NUMERICS.items() if p.arch == arch]
        reason = (
            f"no matching plugin; these numerics are read per target, and {known} are for others"
            if not matches
            else f"multiple matching plugins: {[n for n, _ in matches]}"
        )
        raise Refused(f"{arch} prefill on {target.arch}@{target.vllm}: {reason}")
    return matches[0]
