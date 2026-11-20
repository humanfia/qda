"""What the flow hands an agent turn, at every level and every sandbox setting.

The flow asks the run's isolation for the turn's permission (`Isolation.agent_permission`). This pins that
answer to the one the sandbox setting gives (`isolation.permission`), for all three levels and both sandbox
settings: choosing a level changes the wall, never the turn. The search is stopped at `worker.derive`, before
any session is spawned, so nothing runs.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from qda_agent import flow, isolation
from qda_agent.config import RunConfig
from qda_agent.isolation.api import make

#: The three CLIs a run can drive, each with the spec the run would pass to Humanize.
BACKENDS = {
    "claude": {"name": "claude", "model": "claude-opus-5-5", "effort": "high"},
    "codex": {"name": "codex", "model": "gpt-5.6-sol", "effort": "high"},
    "dsh": {"name": "dsh", "model": "deepseek-v4.1-flash-1017", "account": "infini"},
}


class _Captured(BaseException):
    """Raised by the stand-in worker to stop the search at `derive`."""


class _Worker:
    """A worker whose first act (deriving the turn's permission) is the last thing the search does."""

    permission: object

    def derive(self, **kwargs: object) -> None:
        self.permission = kwargs["permission"]
        raise _Captured


@pytest.mark.parametrize("level", ["none", "sandbox", "strict"])
@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
@pytest.mark.parametrize("cli", sorted(BACKENDS))
def test_the_turn_permission_is_the_sandbox_setting_at_every_level(
    tmp_path, monkeypatch, level, sandbox, cli
) -> None:
    """For every level x backend x sandbox setting, the turn gets `isolation.permission(cfg.sandbox)`."""
    fields = {
        "contamination": "none",
        "model": "example/model",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": tmp_path / f"{level}-{sandbox}-{cli}",
        "agent": {"backend": BACKENDS[cli]},
        "isolation": {"level": level},
        "sandbox": sandbox,
    }
    cfg = RunConfig(**cast(Any, fields))  # the dict spellings a YAML config would use
    iso = make(cfg)

    async def fake_open(*args: object, **kwargs: object) -> tuple[object, object, float, float]:
        return None, iso, 0.0, 3600.0

    monkeypatch.setattr(flow, "_open", fake_open)
    worker = _Worker()
    with pytest.raises(_Captured):
        asyncio.run(
            flow._search(cast(Any, worker), cast(Any, None), cfg, goal=False, ctx=None, clock=lambda: 0.0)
        )
    assert worker.permission == isolation.permission(cfg.sandbox)
