"""The strict level: today's experiment wall, held for the run.

`strict` is the level our recorded runs were made under, and it is today's behaviour: every agent turn runs
inside `gpu_wall` (a mount namespace, their own network namespace with the model's API through the host
proxy, the sign-in files mounted one at a time), the run refuses what today's run refuses
(`isolation.check`, the job sandbox's mount readability), and its jobs keep today's job sandbox.

All of that is `LegacyIsolation`, which holds the code until B9 moves it here; this level inherits it, so
strict cannot drift from today's decisions. It adds one thing: the wall's dedicated-node rule for a
`pid: shared` turn, asked before the run's clock starts rather than when the wall is put up, with the wall's
own sentence (`isolation.wall.shared_pid_refusal`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import pid_refusal
from ..api import Level
from ..wall import WallRefused
from .legacy import LegacyIsolation

if TYPE_CHECKING:
    from ...config import RunConfig

__all__ = ["StrictIsolation"]


class StrictIsolation(LegacyIsolation):
    """Today's wall as a level of its own (`isolation.api`): strict holds what the config always held."""

    level: Level = "strict"

    def preflight(self, cfg: RunConfig) -> None:
        """Refuse the run before its clock starts wherever today's run refuses.

        That is everything `LegacyIsolation.preflight` refuses, then the agent wall's rule for a
        `pid: shared` turn: the host's pid namespace is acceptable only on a node the run has to itself. The
        wall asks the same question when it is put up (`gpu_wall`), with the same sentence; asking it here
        ends the run before it writes its clock, and the wall still checks what is left.

        Raises:
            WallRefused: This node cannot hold a `pid: shared` turn.
        """
        super().preflight(cfg)
        # the wall's own rule and the one place that decides whether it applies (`isolation.pid_refusal`)
        if (problem := pid_refusal(cfg)) is not None:
            raise WallRefused(problem)
