"""The derived-group cache: a model's solve units and the GGUF tensors each one stores, written once.

A GGUF candidate names a *unit* -- the linears a build solves with one scheme, such as ``ffn_gate|ffn_up`` --
and the build needs two tables from it: ``tensor_types`` in GGUF tensor names, and ``format_overrides`` in HF
module names. Spanning those two name spaces needs the model's graph (:func:`qda_quant.gguf.candidates.units`)
and the template's headers, and the model graph is expensive to derive. This module derives both **once** per
(model, template) and writes them beside the build config as JSON; a config then expands its ``type_map`` from
that cache with no model and no template in reach, which is what lets a search host hand a build config to a
different machine without the live module.

The cache is not a substitute for the plan: a config that carries it still plans and prices exactly as it would
without one, because the expansion returns the same ``tensor_types``/``format_overrides`` a hand-written recipe
would have carried (:func:`expand`).

:func:`expand` is also the one place the cached tensor headers are checked against a type: it hands the
cache's units to :func:`qda_quant.gguf.candidates.expand_units` over a stand-in template built from the cache,
so a type that cannot hold a tensor's rows is refused at config time with the same message a build would give.
:func:`expand_units` is the one expansion of a unit map, so a live graph and this cache accept and refuse the
same units in the same words.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..io import sha256_file as _sha256
from ..step import Refused
from . import file
from .candidates import expand_units
from .candidates import units as model_units

if TYPE_CHECKING:
    from ..arch import ModelGraph

#: The cache file's name beside a build config, when the caller does not name its own.
NAME = "derived-groups.json"
#: The JSON shape this module writes and reads; a cache of another version is refused, not reinterpreted.
VERSION = 1


@dataclass(frozen=True)
class _Tensor:
    """One template tensor as the cache stores it: its ggml shape and stored type."""

    shape: tuple[int, ...]
    tensor_type: int


class CachedTemplate:
    """The template headers the type checks read, rebuilt from a cache -- not the template file.

    It answers exactly what `qda_quant.gguf.file.Template` answers to `resolve_types` and `candidates.expand`
    (``arch``, ``n_layer``, ``tensors[*].shape``, ``shapes()``), so a cache and the file it was derived from
    reach the same decisions. It does not hold bytes: nothing here quantizes.
    """

    def __init__(self, data: Mapping[str, Any]):
        self.arch = str(data["arch"])
        self.n_layer = int(data["n_layer"])
        self.tensors = {
            str(name): _Tensor(tuple(int(n) for n in ne), int(kind)) for name, ne, kind in data["tensors"]
        }

    def shapes(self) -> list[tuple[str, tuple[int, ...], int]]:
        """``(name, ne, ggml type)`` of every tensor, in the cache's (file) order."""
        return [(name, t.shape, t.tensor_type) for name, t in self.tensors.items()]


def derive(graph: ModelGraph, template: file.Template, *, model: str | None = None) -> dict[str, Any]:
    """The cache for `graph` and `template`: every solve unit, its modules, and the tensors the template stores.

    A unit is kept only when the template stores at least one tensor of it: a unit the template does not hold
    (an adapter module, a tied head) names nothing a GGUF can store, so it is not a candidate axis. The tensor
    list is the template's own ``shapes()``, because that is what the type checks read.

    Args:
        graph: the model's graph (`qda_quant.arch.describe`), or ``plan.graph`` from a planned build.
        template: the bf16 template the builds will quantize; its headers are read, no weights.
        model: the model path or name, recorded for provenance only.

    Returns:
        A JSON-ready mapping, the shape :func:`load` validates.
    """
    import gguf  # the PyPI package, as the writer imports it

    arch = next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == template.arch)  # pyright: ignore[reportAttributeAccessIssue]
    names = gguf.get_tensor_name_map(arch, template.n_layer)  # pyright: ignore[reportAttributeAccessIssue]
    # the tensors the default mix stores: a module the template keeps float is not a stored type to move
    from .tensor_types import resolve_types

    stored = set(resolve_types(template, []))
    out: dict[str, dict[str, Any]] = {}
    for uid, modules in model_units(graph).items():
        held: list[dict[str, str]] = []
        for mod in modules:
            name = names.get_name(mod + ".weight", try_suffixes=(".weight",))
            if name is not None and name in stored:
                held.append({"module": mod, "tensor": name})
        if held:
            out[uid] = {"modules": list(modules), "tensors": held}
    return {
        "version": VERSION,
        "arch": template.arch,
        "n_layer": template.n_layer,
        "model_type": graph.model_type,
        "model": model,
        "template": {
            "bytes": template.path.stat().st_size,
            "sha256": _sha256(template.path),
        },
        "tensors": [list(s) for s in template.shapes()],
        "units": out,
    }


def load(path: str | Path) -> dict[str, Any]:
    """A written cache, validated: the version, the unit table, and the template headers it must carry.

    Raises:
        Refused: the file is missing or unreadable, is not this version, or does not hold a unit table, the
            template headers or the shapes those headers need.
    """
    where = Path(path)
    try:
        data = json.loads(where.read_text())
    except OSError as e:
        raise Refused(f"gguf groups: {where}: {e}; derive it with `qda sensitivity groups`") from None
    except json.JSONDecodeError as e:
        raise Refused(f"gguf groups: {where}: not JSON ({e})") from None
    if not isinstance(data, dict):
        raise Refused(f"gguf groups: {where}: expected an object")
    if data.get("version") != VERSION:
        raise Refused(
            f"gguf groups: {where}: version {data.get('version')!r}, this build reads {VERSION}; derive it again"
        )
    for key in ("arch", "n_layer", "tensors", "units"):
        if key not in data:
            raise Refused(f"gguf groups: {where}: missing {key!r}; derive it again")
    if not isinstance(data["units"], dict) or not isinstance(data["tensors"], list):
        raise Refused(f"gguf groups: {where}: units must be an object and tensors a list")
    try:
        CachedTemplate(data)
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"gguf groups: {where}: malformed template headers ({e}); derive it again") from None
    return data


def write(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a cache atomically (``<path>.new`` then rename), so a half-written file is never read."""
    where = Path(path)
    tmp = where.with_suffix(where.suffix + ".new")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    tmp.replace(where)
    return where


def expand(data: Mapping[str, Any], spec: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    """``{unit id: ggml type}`` -> the two config tables, with no model and no template file in reach.

    Args:
        data: a cache (:func:`load`).
        spec: unit id -> stored ggml type. A unit's type is written for every tensor the cache holds of its
            modules; when the type stores solved codes (Q6_K, Q5_K) one ``format_overrides`` entry names the
            unit's modules with the format those codes must be solved in.

    Returns:
        ``{"tensor_types": [...], "format_overrides": [...]}``, plain dicts in a stable order -- exactly what
        `qda_quant.gguf.candidates.expand_recipe` returns from a live graph.

    Raises:
        Refused: an id names no unit of this cache, a unit's tensor is named by two units with different types,
            or an expanded rule names a type its tensor cannot hold.
    """
    template = CachedTemplate(data)
    held = {
        uid: [(entry["module"], entry["tensor"]) for entry in unit["tensors"]]
        for uid, unit in data["units"].items()
    }
    # `CachedTemplate` answers exactly what `expand_units` reads of a `Template` (arch, n_layer, tensor shapes,
    # shapes()); the cast is the type system's, not a second implementation.
    return expand_units(cast("file.Template", template), held, spec)
