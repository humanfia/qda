"""A dataset showing the ``qda.datasets`` interface: a prompt set selected by name and version."""

from pathlib import Path
from typing import ClassVar

from qda_kl.datasets import PROMPTS, Dataset


class VersionedPrompts(Dataset):
    """``<root>/<version>.jsonl``: a team's long-generation prompts, kept apart from what the agent can read.

    Attributes:
        root: the directory holding one JSONL of ``{id, domain, prompt}`` rows per version.
        version: the version to read.
    """

    name: ClassVar[str] = "versioned_prompts"
    uses: ClassVar[frozenset[str]] = frozenset({PROMPTS})

    root: str
    version: str

    def local(self) -> Path:
        """The version's file."""
        return Path(self.root) / f"{self.version}.jsonl"
