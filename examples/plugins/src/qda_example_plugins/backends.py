"""A scripted backend that submits the lowest-KL checkpoint already in the ledger."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from qda_agent.agents import Backend
from qda_agent.commands import request

LINE = re.compile(r"^\S+ (?P<path>\S+): (?!refused)\w+: KL (?P<kl>[-+0-9.eE]+)")


class GreedyTurn:
    """Submit the checkpoint with the lowest KL in the current reading list."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.submitted: str | None = None

    async def __call__(self, prompt: str, **_: object) -> str:
        """Read the ledger, then hand in a new best checkpoint if one is available."""
        del prompt
        socket = str(
            self.cfg.host_socket
        )  # read each turn: the flow settles the socket after this object is made
        response = request(socket, "kl", ["list"], ".")
        if response.get("exit", 1):
            return response.get("stderr", "Could not read KL results.")
        scores = [
            (float(match["kl"]), match["path"])
            for line in response.get("stdout", "").splitlines()
            if (match := LINE.match(line))
        ]
        if not scores:
            return "No scored checkpoints are available yet."
        score, path = min(scores)
        if path != self.submitted:
            submitted = request(socket, "handin", [path], ".")
            if submitted.get("exit", 1):
                return submitted.get("stderr", f"Could not submit {path}.")
            self.submitted = path
        return f"Best reading: {path} at KL {score:g}; submission is up to date."


class GreedyBackend(Backend):
    """Demonstrate a scripted greedy backend using QDA's KL and submission commands."""

    name: Literal["greedy_demo"] = "greedy_demo"
    cli: ClassVar[str | None] = None

    @property
    def spec(self) -> str:
        """Name the backend in run records."""
        return f"script/{self.name}"

    @property
    def place(self) -> str:
        """Return the script's place, which has no external CLI."""
        return self.spec

    def turn(self, cfg: Any) -> GreedyTurn:
        """Create the scripted turn for a run's command socket."""
        return GreedyTurn(cfg)

    def driver(self, cfg: Any) -> Any:
        """Return a Humanize fake driver that runs one greedy ledger pass per turn."""
        from hmz.sdk import fakes

        return fakes.FakeAgentDriver(reply=self.turn(cfg), model=self.spec)
