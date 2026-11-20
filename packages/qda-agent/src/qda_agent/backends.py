"""Create Humanize runs with the configured agent, retry policy and fallback chain.

Retry and fallback settings apply to the current run only, through an in-memory settings overlay.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hmz.sdk import Hmz, Refused

from qda_agent import flow as flow_module
from qda_agent.config import RunConfig
from qda_agent.isolation import env_spec

__all__ = ["make_run", "run_fallbacks"]

FLOW_FILE = Path(flow_module.__file__)
# Humanize imports a flow named by a *file* under the file's stem, which would import this module a second
# time and leave `flow.StartRefused` two different classes (what the CLI catches would not be what the flow
# raises). Registering the module we already imported under that name is the "already imported" case hmz keeps.
if sys.modules.setdefault(FLOW_FILE.stem, flow_module) is not flow_module:
    raise RuntimeError(
        f"{FLOW_FILE.stem!r} is imported from another file; Humanize would import this flow a second time "
        "and the run's refusals would not be the ones the CLI catches"
    )


@contextmanager
def run_fallbacks(cfg: RunConfig) -> Iterator[None]:
    """Apply retry and fallback settings for one run without changing the user's settings file.

    Args:
        cfg: The run config (`agent.retries`, `retry_policy`, `retry_timeout_s`, `fallbacks`).

    Raises:
        RuntimeError: This Humanize stores its fallbacks some other way (the hooks used here changed).
    """
    from hmz.coganchor import fallbacks, settings

    held = getattr(fallbacks, "_HELD", None)
    if not (
        isinstance(held, str)
        and callable(getattr(settings, "read", None))
        and callable(getattr(settings, "changes", None))
    ):
        raise RuntimeError(
            "this Humanize keeps fallbacks some other way; qda-agent cannot scope them to a run"
        )
    real_read, real_changes = settings.read, settings.changes
    mine: dict[str, Any] = {held: copy.deepcopy(real_read().get(held, []))}

    def changes_here(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        view = {**real_read(), held: copy.deepcopy(mine[held])}
        change(view)
        mine[held] = view.get(held, [])
        return view

    def read_overlaid() -> dict[str, Any]:
        return {**real_read(), held: copy.deepcopy(mine[held])}

    a = cfg.agent
    settings.read, settings.changes = read_overlaid, changes_here
    try:
        fallbacks.retrying(a.place, a.retries, a.retry_policy, a.retry_timeout_s)
        fallbacks.points(a.place, a.fallback_places)
        settings.changes = real_changes
        yield
    finally:
        settings.read, settings.changes = real_read, real_changes


def make_run(cfg: RunConfig, config_path: Path, *, worker: Any = None) -> Any:
    """Create a Humanize search run without starting it.

    Args:
        cfg: The run config (its workdir must exist).
        config_path: The run YAML, handed to the flow.
        worker: An in-process driver that fills the agent role for this run.

    Returns:
        The `hmz.sdk.Run`.

    Raises:
        hmz.sdk.Refused: Humanize refused the run (no such CLI, no goal feature when `agent.goal` is
            True, a fence this machine cannot hold, ...).
    """
    hmz = Hmz(workspace=cfg.workdir)
    budget: dict[str, Any] = {"duration": cfg.wall_s + cfg.drain_window_s() + 600}
    if cfg.agent.cost_usd is not None:
        budget["cost"] = cfg.agent.cost_usd

    def run(name: str) -> Any:
        return hmz.run(
            f"{FLOW_FILE}:{name}",
            "",
            agents={"worker": worker if worker is not None else cfg.agent.spec},
            envs={"workspace": env_spec(cfg.sandbox, cfg.workdir)},
            params={"config": str(config_path.resolve())},
            budget=budget,
        )

    if cfg.agent.goal is False:
        return run("qda_search_plain")
    try:
        return run("qda_search")
    except Refused as why:
        if cfg.agent.goal or "GoalCommandAgentMixin" not in str(why):
            raise
        return run("qda_search_plain")
