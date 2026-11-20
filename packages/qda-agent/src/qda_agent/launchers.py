"""Launch a search in the current process."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

from qda_quant.registry import Registry

from qda_agent.settings import Plugin, validate_plugin

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

__all__ = ["Launcher", "Local"]


class Launcher(Plugin):
    """Configure how a search starts and reports its exit status."""

    def launch(self, cfg: RunConfig, run: Callable[[], int]) -> int:
        """Start the configured search using `run` locally, and return its exit status."""
        raise NotImplementedError


class Local(Launcher):
    """Run a search in the current process."""

    name: Literal["local"] = "local"

    def launch(self, cfg: RunConfig, run: Callable[[], int]) -> int:
        """Run the search in the current process and return its exit status."""
        del cfg
        return run()


LAUNCHERS = Registry(
    "launchers",
    {"local": Local},
    group="qda_agent.launchers",
    on_add=partial(validate_plugin, "qda_agent.launchers"),
)
