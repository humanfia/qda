"""A component that installs one read-only note in an agent workdir."""

from typing import Literal

from qda_agent.components import Component, Run


class ReadOnlyNote(Component):
    """Install a short note in the overlay and bind it read-only over the agent's workdir."""

    name: Literal["read_only_note"] = "read_only_note"
    text: str = "This file is supplied by an external QDA component."

    def section(self, run: Run) -> str:
        """Tell the agent where the note is installed."""
        del run
        return "- Read `PLUGIN_NOTE.md`, a read-only note from an external component."

    def install(self, run: Run) -> None:
        """Write the note into the host-owned overlay."""
        path = run.ro("PLUGIN_NOTE.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text + "\n")

    def protected(self, run: Run) -> list[str]:
        """Bind the note read-only over its workdir path."""
        del run
        return ["PLUGIN_NOTE.md"]
