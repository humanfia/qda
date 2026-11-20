"""The judge: frozen copies, KL readings, hand-in checks, the contamination scan and the final rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import JudgeConfig

if TYPE_CHECKING:
    from .host import JudgeHost, make_host

__all__ = ["JudgeConfig", "JudgeHost", "make_host"]


def __getattr__(name: str) -> Any:
    """Import the host side on first use: a run config needs only the judge's settings."""
    if name in ("JudgeHost", "make_host"):
        from . import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
