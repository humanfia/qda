"""Strict settings and validation for named agent plugins."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from qda_quant.registry import Registry

__all__ = [
    "Plugin",
    "git_commit",
    "load_object",
    "pick",
    "provenance",
]


class Plugin(BaseModel):
    """Strict settings of a named plugin, with ownership of any extra path fields."""

    model_config = ConfigDict(extra="forbid")

    #: Extra path fields: agent-writable, judge-read (existing outside the run), or host-run (outside the run).
    source_kinds: ClassVar[Mapping[str, str]] = {}

    name: str


class Strict(BaseModel):
    """Settings that refuse unknown keys."""

    model_config = ConfigDict(extra="forbid")  # a misspelled or unknown key is refused, never ignored


def validate_plugin(group: str, key: str, cls: type[Plugin]) -> None:
    """Refuse entries that cannot validate agent plugin settings."""
    if not (isinstance(cls, type) and issubclass(cls, Plugin)):
        raise TypeError(f"{group} plugin {key!r} is {cls!r}, not a qda_agent.plugins.Plugin class")


def pick(registry: Registry[type[Plugin]], value: Any, default: str | None = None) -> Any:
    """Validate a named plugin's config value, using the section's default when no name is set."""
    if isinstance(value, Plugin):
        return value
    if isinstance(value, str):
        value = {"name": value}
    if isinstance(value, dict) and "name" not in value and default is not None:
        value = {"name": default, **value}
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ValueError(f"expected a {registry.group} key or a mapping with `name: KEY`, got {value!r}")
    if value["name"] not in registry:
        raise ValueError(
            f"no {registry.group} plugin {value['name']!r} is installed; installed: {', '.join(registry)}"
        )
    return registry[value["name"]].model_validate(value)


def provenance(registry: Registry[type[Plugin]], key: str) -> dict[str, str | None]:
    """`{"package": ..., "version": ...}` of a plugin, for the run's `resolved.yaml`."""
    package, version = registry.provenance(key)
    return {"package": package, "version": version}


def git_commit(where: Path) -> str | None:
    """Read the checkout commit at `where`, marking tracked changes with `+dirty`; return None on failure."""
    try:
        head = subprocess.run(
            ["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(where), "status", "--porcelain", "--untracked-files=no", "."],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}+dirty" if dirty else head


def load_object(ref: str) -> Any:
    """Import an attribute named by `module:attr`, such as a host factory or command installer."""
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"{ref!r} is not module:attr")
    return getattr(importlib.import_module(module), attr)
