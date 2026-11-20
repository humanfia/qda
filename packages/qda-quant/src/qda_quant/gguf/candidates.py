"""Expand a GGUF candidate's per-class stored-type map to exact-name rules, refusing bad ones at config time.

A mistake the build finds costs a load and a solve, and after scoring it looks like a bad candidate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..arch import ModelGraph
from ..step import Refused
from . import file
from .tensor_types import TensorTypeRule, resolve_types

#: The format a solved unit's codes must be written with, by the type they are stored as. A type absent here
#: needs no override: Q8_0 stores the template's values, and Q4_K codes may be written into any stored type.
SOLVED_FORMATS: dict[str, str] = {"Q6_K": "w6a16-sym-g16", "Q5_K": "w5a16-asym-g32"}


def class_key(name: str) -> str:
    """The layer-free class of a GGUF tensor name: ``blk.12.ffn_down.weight`` -> ``blk.*.ffn_down.weight``."""
    layer, dot, rest = name.partition(".")
    if layer == "blk" and dot and rest.split(".", 1)[0].isdigit():
        return "blk.*." + rest.split(".", 1)[1]
    return name


def classes(template: file.Template) -> dict[str, tuple[str, ...]]:
    """Every two-dimensional tensor of the template, grouped by :func:`class_key`.

    Norms stay F32 and expert banks follow their solve unit, so neither is a per-tensor axis.
    """
    grouped: dict[str, list[str]] = {}
    for name, info in template.tensors.items():
        if len(info.shape) == 2:
            grouped.setdefault(class_key(name), []).append(name)
    return {key: tuple(sorted(names)) for key, names in grouped.items()}


def _members(grouped: Mapping[str, tuple[str, ...]], template: file.Template, key: str) -> tuple[str, ...]:
    """The tensors a map key names: a class, or one exact two-dimensional tensor of the template."""
    if key in grouped:
        return grouped[key]
    info = template.tensors.get(key)
    if info is not None and len(info.shape) == 2:
        return (key,)
    raise Refused(
        f"gguf candidate: {key!r} names no two-dimensional tensor of the template and no class of one "
        f"(classes: {', '.join(sorted(grouped))})"
    )


def _select(
    template: file.Template, types: Mapping[str, str], overrides: Mapping[str, str] | None
) -> dict[str, str]:
    """The tensor a map names -> its type, before any fused-unit or width check."""
    grouped = classes(template)
    selected: dict[str, str] = {}
    for key, kind in types.items():
        for name in _members(grouped, template, key):
            selected[name] = kind
    for name, kind in (overrides or {}).items():
        if name not in selected:
            raise Refused(f"gguf candidate: override {name!r} names a tensor the class map did not select")
        selected[name] = kind
    return selected


def expand(
    template: file.Template,
    types: Mapping[str, str],
    *,
    fused: Sequence[Sequence[str]] = (),
    overrides: Mapping[str, str] | None = None,
) -> list[TensorTypeRule]:
    """Expand ``{class or name: ggml type}`` to one exact-name rule per tensor, validated before a build.

    Args:
        types: Class (``blk.*.ffn_down.weight``) or exact tensor name to stored type; a tensor left unnamed
            keeps the default Q4_K_M mix.
        fused: Units solved and stored with one scheme, each a sequence of class or tensor keys; every member
            must be named in `types` with the same type.
        overrides: Per-tensor types applied after the class map.

    Returns:
        The exact-name rules, in sorted tensor order (``tensor_types``).

    Raises:
        Refused: a key names nothing, a fused unit disagrees or is partly unnamed, or a selected type cannot
            hold a tensor's rows (through :func:`resolve_types`).
    """
    grouped = classes(template)
    selected = _select(template, types, overrides)
    for unit in fused:
        names = tuple(name for key in unit for name in _members(grouped, template, key))
        named = [name for name in names if name in selected]
        missing = [name for name in names if name not in selected]
        if missing:
            raise Refused(
                f"gguf candidate: fused unit {list(unit)} is solved with one scheme, but "
                f"{', '.join(missing)} has no type in the map; name every member of the unit"
            )
        kinds = {selected[name] for name in named}
        if len(kinds) != 1:
            raise Refused(
                f"gguf candidate: fused unit {list(unit)} is solved and stored with one type, but its "
                f"members name {len(kinds)}: "
                + ", ".join(f"{name} {selected[name]}" for name in named)
                + "; one type per unit, or the writer refuses the unit after the solve"
            )
    rules = [TensorTypeRule(pattern=name, type=selected[name]) for name in sorted(selected)]
    resolve_types(template, rules)
    return rules


def solved_formats(rules: Sequence[TensorTypeRule]) -> dict[str, str]:
    """The format each rule's solved codes need, by tensor name; an absent tensor needs no override."""
    return {rule.pattern: SOLVED_FORMATS[rule.type] for rule in rules if rule.type in SOLVED_FORMATS}


def unit_key(modules: Sequence[str]) -> str:
    """A unit's candidate id: its fused module names joined by ``|``, in graph order."""
    return "|".join(modules)


def unit_pattern(modules: Sequence[str]) -> str:
    """The ``format_overrides`` pattern naming exactly `modules`: the one name, or one ``re:`` group."""
    return modules[0] if len(modules) == 1 else "re:^(?:" + "|".join(re.escape(m) for m in modules) + ")$"


def units(graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """The model's solve units, ``id -> its HF module names``: one per `LinearGroup` of every layer.

    Embeddings and the head are not units: ``format_overrides`` cannot name them, so :func:`expand` does.
    """
    out: dict[str, tuple[str, ...]] = {}
    for layer in graph.layers:
        for group in layer.groups:
            out[unit_key(group.modules)] = tuple(group.modules)
    return out


def unit_tensors(template: file.Template, graph: ModelGraph) -> dict[str, list[tuple[str, str]]]:
    """Unit id -> ``(module, template tensor)`` for every solve unit the template stores.

    :func:`expand_recipe` and :func:`mix_formats` share this map, so they agree on which tensor a unit owns.
    """
    import gguf  # the PyPI package, as the writer imports it

    arch = next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == template.arch)  # pyright: ignore[reportAttributeAccessIssue]
    names = gguf.get_tensor_name_map(arch, template.n_layer)  # pyright: ignore[reportAttributeAccessIssue]
    stored = set(resolve_types(template, []))
    held: dict[str, list[tuple[str, str]]] = {}
    for uid, modules in units(graph).items():
        pairs = [
            (module, name)
            for module in modules
            if (name := names.get_name(module + ".weight", try_suffixes=(".weight",))) is not None
            and name in stored
        ]
        if pairs:
            held[uid] = pairs
    return held


def mix_formats(
    template: file.Template,
    graph: ModelGraph,
    types: Sequence[TensorTypeRule] = (),
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    """The ``format_overrides`` the stored types themselves require, and the units they cannot cover.

    The Q4_K_M mix stores some tensors as Q6_K or Q5_K, whose codes are dropped unless solved so.

    Args:
        types: The GGUF-side stored types a recipe names (``exporter.gguf.tensor_types``).

    Returns:
        ``(rules, unresolved)``: ``[{"pattern", "format"}]`` per unit whose stored type is in
        :data:`SOLVED_FORMATS`, and unit id -> the types a fused unit's tensors disagree on (no single
        override covers such a unit).
    """
    stored = resolve_types(template, list(types))
    rules: list[dict[str, str]] = []
    unresolved: dict[str, list[str]] = {}
    for uid, pairs in unit_tensors(template, graph).items():
        kinds = {file.type_name(stored[name]) for _module, name in pairs}
        if len(kinds) != 1:
            unresolved[uid] = sorted(kinds)
            continue
        if (fmt := SOLVED_FORMATS.get(kinds.pop())) is not None:
            rules.append({"pattern": unit_pattern([module for module, _name in pairs]), "format": fmt})
    return rules, unresolved


def expand_units(
    template: file.Template,
    by_id: Mapping[str, Sequence[tuple[str, str]]],
    spec: Mapping[str, str],
    *,
    selected: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """``{unit id: ggml type}`` -> both config tables, from each unit's stored ``(module, tensor)`` pairs.

    A live graph and the derived-group cache share this expansion, so both accept and refuse the same units.

    Args:
        by_id: Unit id -> ``(module, tensor)`` per tensor the template stores of that unit.
        selected: Types already chosen for exact tensor names, which a unit must agree with.

    Returns:
        ``{"tensor_types": [...], "format_overrides": [...]}``, plain dicts in a stable order.

    Raises:
        Refused: an id names no unit, a unit holds no tensor, a tensor is named with two types, or a rule's
            type cannot hold its tensor's rows.
    """
    chosen = dict(selected or {})
    formats: list[dict[str, str]] = []
    for uid, kind in spec.items():
        held = by_id.get(uid)
        if held is None:
            sample = next(iter(sorted(by_id)), None)
            raise Refused(
                f"gguf candidate: {uid!r} names no solve unit; units are the fused linear groups, "
                f"e.g. {sample!r}"
            )
        if not held:
            raise Refused(f"gguf candidate: unit {uid!r} holds no tensor the template stores")
        for _module, name in held:
            if name in chosen and chosen[name] != kind:
                raise Refused(
                    f"gguf candidate: {name} is {chosen[name]} in one rule and {kind} in unit {uid!r}; "
                    "one type per tensor"
                )
            chosen[name] = kind
        if kind in SOLVED_FORMATS:
            modules = [module for module, _name in held]
            formats.append({"pattern": unit_pattern(modules), "format": SOLVED_FORMATS[kind]})
    try:
        rules = [TensorTypeRule(pattern=name, type=chosen[name]) for name in sorted(chosen)]
    except ValueError as e:  # a bad type name, refused by the rule model's own validator
        raise Refused(f"gguf candidate: {e}") from None
    resolve_types(template, rules)
    return {
        "tensor_types": [rule.model_dump() for rule in rules],
        "format_overrides": formats,
    }


def expand_recipe(
    template: file.Template,
    graph: ModelGraph,
    spec: Mapping[str, str],
    *,
    types: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """The exporter-side candidate: unit types plus optional GGUF-side types -> both tables, as plain dicts.

    Args:
        spec: ``{unit id: ggml type}`` over :func:`units`; a type that solves codes also gets one
            ``format_overrides`` entry from :data:`SOLVED_FORMATS`.
        types: Optional GGUF-side ``{class or exact name: type}``, expanded by :func:`expand`; a tensor a unit
            also names must agree.
        overrides: Per-tensor GGUF-name types (one layer's exception), applied after `types`.

    Returns:
        ``{"tensor_types": [...], "format_overrides": [...]}`` in a stable order, so a caller can hash it.

    Raises:
        Refused: an id names no unit the template stores, a unit and the GGUF-side map disagree on a tensor's
            type, or an expanded rule fails :func:`resolve_types`.
    """
    return expand_units(
        template, unit_tensors(template, graph), spec, selected=_select(template, types or {}, overrides)
    )
