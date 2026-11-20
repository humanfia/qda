"""Configure one search from YAML, with named plugins for each replaceable part.

`RunConfig.load` applies dotted overrides and resolves configured paths: relative to the YAML file when written
there, relative to the current directory when given with `--set`.
`resolved` adds commit and plugin provenance for `OUT/resolved.yaml`, which can be loaded again.

The models live in `qda_agent.run_config` and are imported on first use here: importing this module pulls
neither the judge nor the scheduler, which a config's fields and plugin names may otherwise need at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qda_agent.run_config import IsolationSettings, RunConfig

__all__ = ["IsolationSettings", "RunConfig"]


def __getattr__(name: str) -> Any:
    """Import the run config models on first use (PEP 562)."""
    if name in __all__:
        from qda_agent import run_config

        value = getattr(run_config, name)
        globals()[name] = value  # the second read is a plain one, and `dir()` lists it
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
