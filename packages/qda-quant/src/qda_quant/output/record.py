"""Run records: the ``resolved.yaml`` every run writes into its output directory.

``resolved.yaml`` is the run's full config, every default filled in, plus ``provenance``::

    provenance:
      qda_commit: <git sha of the qda checkout, "-dirty" when it has local changes; "unknown" outside one>
      source_snapshot: <the local directory a hub model id resolved to, or null>
      source_revision: <the hub commit of that snapshot, or null>
      source_signature: <the signature of a local model directory, or null>
      sources: {<key>: {snapshot: ..., revision: ..., signature: ...}}   # a run that reads several models
      cards: <the card count the run used>
      not_recorded: [<options that differ from their defaults and are not config keys>]
      plugins: {<name>: {group: <entry-point group>, package: <distribution>, version: <version>}}

The three ``source_*`` keys name the one model a build reads. A run that reads several models (an eval reads a
reference and every checkpoint) records one entry per model under ``sources``, keyed the way its config names
them, with the same three facts; ``sources`` is absent from a run that reads one model.

Running from the file repeats the run: a config reader drops ``provenance`` (:func:`without_provenance`).
``qda build`` and ``qda-kl score`` write the same shape.
"""

from __future__ import annotations

import functools
import importlib.metadata as md
import os
import secrets
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from ..registry import Registry

RESOLVED = "resolved.yaml"
BUILD_RECORDS = ("build.json", "recipe.yaml", RESOLVED)


def provenance_of(obj: Any) -> tuple[str, str | None]:
    """Return the distribution owning an object's top-level module."""
    if isinstance(obj, str):
        module = obj.split(":")[0]
    elif isinstance(obj, ModuleType):
        module = obj.__name__
    else:
        module = getattr(obj, "__module__", type(obj).__module__)
    return _dist_of_module(module.split(".")[0])


@functools.cache
def _dist_of_module(top: str) -> tuple[str, str | None]:
    for package in sorted(md.packages_distributions().get(top, [])):
        try:
            return package, md.version(package)
        except md.PackageNotFoundError:
            continue
    try:
        package = top.replace("_", "-")
        return package, md.version(package)
    except md.PackageNotFoundError:
        return top, None


@functools.cache
def qda_commit() -> str:
    """The git commit of the qda checkout this package runs from, ``-dirty`` when tracked files changed.

    ``unknown`` outside a checkout (an installed wheel, even one inside another git repository) or without git.
    """
    here = Path(__file__).resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(here.parent), *args], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()

    try:
        git("ls-files", "--error-unmatch", here.name)  # this file is tracked: the checkout is qda's
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def plugin(reg: Registry, key: str, obj: Any = None) -> dict[str, Any]:
    """``{group, package, version}`` of the plugin ``key`` of ``reg``.

    ``obj`` is the object the run used (default: the registered one). An object the registry does not hold under
    ``key`` (a step class defined in a test or a script) is recorded with its module's package and group null.
    """
    if obj is None or (key in reg and reg[key] is obj):
        (package, version), group = reg.provenance(key), reg.group
    else:
        (package, version), group = provenance_of(obj), None
    return {"group": group, "package": package, "version": version}


def plugins(used: Iterable[tuple[Registry, str, Any]]) -> dict[str, dict[str, Any]]:
    """``{name: plugin(...)}`` of every ``(registry, key, object)`` a run used, sorted by name.

    Registries keyed by architecture share names (``sm90`` is a target table and a binding rule): the first use
    keeps the bare name, a later one from another group is recorded as ``<group>:<name>``.

    Raises:
        ValueError: one name is used for two different plugins of one group.
    """
    out: dict[str, dict[str, Any]] = {}
    for reg, key, obj in used:
        entry = plugin(reg, key, obj)
        name = key if out.get(key, entry)["group"] == entry["group"] else f"{entry['group']}:{key}"
        if out.get(name, entry) != entry:
            raise ValueError(f"plugin name {key!r} is used for two plugins: {out[name]} and {entry}")
        out[name] = entry
    return dict(sorted(out.items()))


def provenance(
    used: dict[str, dict[str, Any]],
    source_snapshot: str | None = None,
    *,
    source_revision: str | None = None,
    source_signature: str | None = None,
    cards: int | None = None,
    not_recorded: Sequence[str] = (),
    sources: Mapping[str, Mapping[str, str | None]] | None = None,
) -> dict[str, Any]:
    """The ``provenance`` block of a run: what its config does not pin.

    Args:
        used: the plugins the run used (:func:`plugins`).
        source_snapshot: the local directory a hub repo id resolved to (None for a local model).
        source_revision: the hub commit of that snapshot (None for a local model).
        source_signature: the signature of a local model directory
            (`qda_quant.intake.fingerprint.model_signature`); None for a hub id, which its revision pins.
        cards: the card count the run used.
        not_recorded: options of the run that differ from their defaults and are not config keys.
        sources: one ``{snapshot, revision, signature}`` per model, for a run that reads several; None for a
            run that reads one, which the three ``source_*`` keys name. Recorded only when given, so the shape
            of a single-source run is unchanged.
    """
    d = {
        "qda_commit": qda_commit(),
        "source_snapshot": source_snapshot,
        "source_revision": source_revision,
        "source_signature": source_signature,
        "cards": cards,
        "not_recorded": list(not_recorded),
        "plugins": used,
    }
    if sources is not None:
        # every model of a multi-model run, each with the three facts a build records for its one model
        d["sources"] = {key: dict(facts) for key, facts in sources.items()}
    return d


def without_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` without its ``provenance`` block: what a run reads back from ``resolved.yaml``."""
    return {k: v for k, v in config.items() if k != "provenance"}


def write_resolved(
    out_dir: str | Path,
    config: dict[str, Any],
    prov: dict[str, Any],
    *,
    name: str = RESOLVED,
    mode: int | None = None,
) -> Path:
    """Write ``config`` plus ``provenance`` atomically, optionally naming a private launcher config."""
    out = Path(out_dir) / name
    tmp = out.with_name(f".{out.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        if mode is not None:
            tmp.touch(mode=mode)
            tmp.chmod(mode)
        tmp.write_text(yaml.safe_dump({**config, "provenance": prov}, sort_keys=False))
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out
