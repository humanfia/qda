"""Agent sandboxes (`qda_agent.sandbox_settings`): where the agent's sessions run (see `qda_agent.isolation`).

These are the settings only; `qda_agent.isolation` holds the wall itself and reads `name` and the settings.

- `bwrap`: Humanize's fence inside a bubblewrap namespace with no GPU device nodes, the home directory and the
  run dir hidden.
- `none`: no sandbox (development only).
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field
from qda_quant.registry import Registry

from qda_agent.settings import Plugin, validate_plugin

SANDBOXES = Registry(
    "sandboxes",
    {
        "bwrap": "qda_agent.sandbox_settings:Bwrap",
        "none": "qda_agent.sandbox_settings:NoSandbox",
    },
    group="qda_agent.sandboxes",
    on_add=partial(validate_plugin, "qda_agent.sandboxes"),
)

__all__ = ["Bwrap", "NoSandbox", "Sandbox"]


class Sandbox(Plugin):
    """Settings of an agent sandbox.

    Class attributes:
        private_socket: Whether the host listens in a short private directory when the run's socket path does
            not fit. The built-in sandboxes allow it.
        host_paths: Whether the agent sees the host's paths. Else a mount namespace or a container maps the
            socket over its parent.
        walled: Whether the agent's shell is in the wall, which binds the overlay read-only over the workdir.
    """

    private_socket: ClassVar[bool] = True
    host_paths: ClassVar[bool] = True
    walled: ClassVar[bool] = False

    def writable(self) -> list[Path] | None:
        """The host directories outside the run dir that the agent (or its jobs) can write.

        None: the mode leaves the filesystem open, so no directory can be named; a check that must not trust
        them cannot enumerate them. The run dir is checked apart, and `scheduler.job_sandbox: none` makes the
        jobs unconfined too (`qda_agent.run_sources`).
        """
        return None


class Bwrap(Sandbox):
    """Bubblewrap around the Humanize fence.

    Attributes:
        hide: More directories covered with an empty tmpfs (test sets, reference traces, ... that do not live
            under the home directory or the run dir).
        pid: The turn's pid namespace. `"auto"` keeps a private one (`--unshare-pid`) wherever bwrap can mount
            a fresh procfs; where the host refuses that, it becomes `"shared"` -- which the setting alone does
            not record, so the effective form is written to `resolved.yaml`'s provenance and shown by
            `qda doctor`. `"shared"` needs a node the run has to itself and is refused otherwise; `"private"`
            on such a host leaves Humanize's cred unable to resolve the tracee, so the run fails visibly.
    """

    name: Literal["bwrap"] = "bwrap"
    hide: list[Path] = Field(default_factory=list)
    pid: Literal["auto", "private", "shared"] = "auto"
    host_paths: ClassVar[bool] = False
    walled: ClassVar[bool] = True

    def writable(self) -> list[Path] | None:
        """`$TMPDIR`, when the wall shows the host's directory there.

        The wall binds the fence's temporary paths writable. A `$TMPDIR` of the home (hidden) or exactly
        `/tmp` or `/var/tmp` (replaced by a private tmpfs) is not the host's, and grants nothing.
        """
        named = Path(os.environ.get("TMPDIR", ""))
        if not named.is_absolute():
            return []
        real = Path(os.path.realpath(named))
        private = {Path(os.path.realpath(t)) for t in ("/tmp", "/var/tmp")}
        if real in private or real.is_relative_to(Path(os.path.realpath(Path.home()))):
            return []
        return [real]


class NoSandbox(Sandbox):
    """No sandbox (development only)."""

    name: Literal["none"] = "none"

    def writable(self) -> list[Path] | None:
        """None: without a sandbox the agent can write whatever the user can."""
        return None
