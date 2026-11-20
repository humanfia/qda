"""Dotted module-name matching shared by the output paths that select, rename and configure modules."""

from __future__ import annotations


def covers(prefix: str, name: str) -> bool:
    """Whether `name` is the dotted module `prefix` or lies under it (``layers.3`` covers ``layers.3.mlp``)."""
    return name == prefix or name.startswith(prefix + ".")
