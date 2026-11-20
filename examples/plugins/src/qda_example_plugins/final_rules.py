"""A final rule showing the ``qda.final_rules`` interface: the first valid submission, not the last."""

from typing import Literal

from qda_agent.judge.final_rules import FinalRule, Pick, Standing


class FirstValid(FinalRule):
    """The earliest submission that passed its checks."""

    name: Literal["first_valid"] = "first_valid"

    def decide(self, standing: Standing) -> Pick | None:
        """The earliest valid hand-in, or None to leave the decision to the next rule."""
        handed = [e for e in standing.valid if e.kind == "handin"]
        return Pick("the first valid submission", choice=handed[0]) if handed else None

    def statement(self, metric: str) -> str:
        """The sentence for the agent's task statement."""
        del metric
        return "the first submission that passed counts"
