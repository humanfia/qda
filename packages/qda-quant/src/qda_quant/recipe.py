"""Recipes: ordered ``{name: settings}`` steps, a ``calib`` source and a default ``grid``, as Python or YAML.

Example: ``{grid: absmax, steps: [{rotate: {}}, {gptq: {damp: 0.01}}], calib: {file: {path: calib.jsonl}}}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .calib.sources import SOURCES, CalibSource, parse_source
from .intake.corpus import CORPORA
from .registry import Registry
from .solve.grid import known_grid
from .step import REGISTRY, QuantStep, Refused, Step

_NESTED: dict[str, Registry[Any]] = {"source": SOURCES, "corpus": CORPORA, "step": REGISTRY}


class Recipe(BaseModel):
    """An ordered list of steps and the calibration data they share."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    steps: list[Step]
    calib: CalibSource
    meta: dict[str, Any] = Field(default_factory=dict)
    grid: str | None = None

    @field_validator("grid")
    @classmethod
    def _known_grid(cls, v: str | None) -> str | None:
        return known_grid(v)

    @model_validator(mode="after")
    def _steps_take_the_recipe_grid(self) -> Recipe:
        """Give each grid step that names no grid the recipe's; with none, `compile_plan` resolves it later."""
        if self.grid is not None:
            self.steps = [
                s.model_copy(update={"grid": self.grid}) if isinstance(s, QuantStep) and s.grid is None else s
                for s in self.steps
            ]
        return self

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """Plain data: each step's non-default settings, or every setting when `full` (``resolved.yaml``).

        A step's ``grid`` appears only when it differs from the recipe's, so ``--set grid=...`` still reaches it.
        """
        dump: dict[str, Any] = {"mode": "json"} if full else {"exclude_defaults": True}

        def settings(s: Step) -> dict[str, Any]:
            d = s.model_dump(**dump)
            if isinstance(s, QuantStep) and s.grid == self.grid:
                d.pop("grid", None)
            return d

        return {
            "grid": self.grid,
            "steps": [{s.name: settings(s)} for s in self.steps],
            "calib": self.calib.to_dict(),
            **({"meta": self.meta} if self.meta else {}),
        }

    def to_yaml(self) -> str:
        """YAML text of `to_dict()`."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any], overrides: list[str] | tuple[str, ...] = (),
                  *, base: str | Path | None = None) -> Recipe:  # fmt: skip
        """Build from plain data after `apply_overrides`; declared relative paths resolve under `base`.

        Raises:
            Refused: a bad override, step or calib value; the message names the dotted key.
        """
        d = apply_overrides(d, overrides)
        if base is not None:
            d = resolve_paths(d, base)
        steps = []
        for i, item in enumerate(d.get("steps", [])):
            if not (isinstance(item, dict) and len(item) == 1):
                raise Refused(f"steps.{i}: expected one {{name: settings}} entry, got {item!r}")
            ((key, cfg),) = item.items()
            if key not in REGISTRY:
                raise Refused(f"unknown step {key!r}; known: {sorted(REGISTRY)}")
            try:
                steps.append(REGISTRY[key](**(cfg or {})))
            except ValidationError as e:
                raise Refused("; ".join(error_lines(e, f"steps.{i}.{key}"))) from None
        try:
            calib = parse_source(d["calib"])
        except ValidationError as e:  # the source's settings: name them as config keys
            raise Refused("; ".join(error_lines(e, f"calib.{next(iter(d['calib']))}"))) from None
        try:
            return cls(
                steps=steps,
                calib=calib,
                meta=d.get("meta", {}),
                grid=d.get("grid"),
            )
        except ValidationError as e:
            raise Refused("; ".join(error_lines(e))) from None

    @classmethod
    def from_yaml(
        cls, text: str, overrides: list[str] | tuple[str, ...] = (), *, base: str | Path | None = None
    ) -> Recipe:
        """Parse YAML text; as `from_dict`."""
        return cls.from_dict(yaml.safe_load(text), overrides, base=base)


def resolve_paths(d: dict[str, Any], base: str | Path) -> dict[str, Any]:
    """Recipe-shaped data `d` with each plugin-declared (`paths`, `models`, `nested`) relative path under `base`."""
    root = Path(base)
    out = dict(d)
    if isinstance(out.get("calib"), dict):
        out["calib"] = _plugin(out["calib"], root, "source")
    steps = out.get("steps")
    if isinstance(steps, list):
        out["steps"] = [_plugin(e, root, "step") if isinstance(e, dict) else e for e in steps]
    return out


def expand_path(value: str) -> Path:
    """Expand a home path, refusing an unknown user with a usable correction."""
    try:
        return Path(value).expanduser()
    except RuntimeError:
        raise Refused(f"{value!r}: cannot resolve home directory; use an absolute path") from None


def _plugin(d: dict[str, Any], base: Path, kind: str) -> dict[str, Any]:
    """One ``{name: settings}`` plugin with its declared paths resolved."""
    if len(d) != 1:
        return d
    ((name, settings),) = d.items()
    return {name: _settings(settings, base, _NESTED[kind].get(name))}


def _settings(settings: Any, base: Path, cls: Any) -> Any:
    """One plugin's settings: `paths` absolute, `models` local directories absolute, `nested` plugins resolved."""
    if cls is None or (settings is not None and not isinstance(settings, dict)):
        return settings
    out = dict(settings or {})
    for name in {*getattr(cls, "paths", ()), *getattr(cls, "models", ()), *getattr(cls, "nested", {})}:
        field = cls.model_fields.get(name)
        if name not in out and field is not None and not field.is_required():
            out[name] = field.get_default(call_default_factory=True)
    for name in getattr(cls, "paths", ()):
        value = out.get(name)
        if isinstance(value, str) and not Path(value).is_absolute():
            out[name] = str((base / expand_path(value)).resolve())
    for name in getattr(cls, "models", ()):
        value = out.get(name)
        if isinstance(value, str):
            path = expand_path(value)
            if value.startswith((".", "~")) or (base / path).is_dir():
                out[name] = str((base / path).resolve())
    for field, (kind, part) in getattr(cls, "nested", {}).items():
        if field == "*":  # the settings mapping itself holds the nested plugin (a corpus source's keys)
            for name, value in list(out.items()):
                if isinstance(value, dict) and name in _NESTED[kind]:
                    out[name] = _plugin({name: value}, base, kind)[name]
        elif isinstance(out.get(field), list):
            out[field] = [_part(p, base, kind, part) for p in out[field]]
    return out


def _part(part: Any, base: Path, kind: str, cls: Any) -> Any:
    """One element of a nested list: `cls`'s settings, or the settings of the plugin it names."""
    if not isinstance(part, dict):
        return part
    named = [name for name in part if name in _NESTED[kind]]
    if len(named) == 1 and isinstance(part[named[0]], dict):
        name = named[0]
        return {**part, name: _plugin({name: part[name]}, base, kind)[name]}
    return _settings(part, base, cls)


def error_lines(e: ValidationError, prefix: str = "") -> list[str]:
    """One ``dotted.key: what is wrong`` line per error of `e`; `prefix` is the key of the validated value."""
    lines = []
    for err in e.errors():
        key = ".".join([prefix, *map(str, err["loc"])] if prefix else map(str, err["loc"]))
        if err["type"] == "missing":
            msg = "missing required key"
        elif err["type"] == "extra_forbidden":
            msg = "unknown key"
        elif "error" in (ctx := err.get("ctx", {})):  # validator exception: message, no pydantic prefix
            msg = str(ctx["error"])
        else:
            msg = err["msg"]
        lines.append(f"{key}: {msg}" if key else msg)
    return lines


def error_fix(e: ValidationError) -> str:
    """The fixes the validators' own refusals name (`Refused.fix`), joined: what `error_lines` alone drops."""
    from .step import Refused

    fixes = [
        f for err in e.errors() if isinstance(r := err.get("ctx", {}).get("error"), Refused) and (f := r.fix)
    ]
    return "; ".join(dict.fromkeys(fixes))


def apply_overrides(d: dict[str, Any], overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """A copy of `d` with ``dotted.path=value`` overrides applied; values are parsed as YAML.

    A step's name is optional in the path (``steps.1.damp`` == ``steps.1.gptq.damp``), so an error's key can be
    passed back as an override; missing mapping keys are created, list indices must exist.

    Raises:
        Refused: an override that does not apply; the message starts with ``--set OVERRIDE:``.
    """
    import copy

    d = copy.deepcopy(d)
    for ov in overrides:
        try:
            _apply_override(d, ov)
        except (ValueError, yaml.YAMLError) as e:
            raise Refused(f"--set {ov}: {e}") from None
    return d


def _apply_override(d: dict[str, Any], ov: str) -> None:
    """Apply one ``dotted.path=value`` to `d` in place; raises Refused saying which part of the path fails."""
    path, eq, raw = ov.partition("=")
    if not eq or not path:
        raise Refused("expected KEY=VALUE")
    keys = path.split(".")
    cur: Any = d
    i = 0
    while i < len(keys) - 1:
        cur, from_list = _child(cur, keys[:i], keys[i]), isinstance(cur, list)
        i += 1
        if from_list and isinstance(cur, dict) and len(cur) == 1:  # a step entry {name: settings}
            ((name, inner),) = cur.items()
            if inner is None:
                inner = cur[name] = {}
            if isinstance(inner, dict):
                if keys[i] == name:  # the named form: let the loop step into the settings
                    continue
                cur = inner  # the short form: skip the name
    container, last = cur, keys[-1]
    if isinstance(container, list):
        container[_index(container, keys[:-1], last)] = yaml.safe_load(raw)
    elif isinstance(container, dict):
        container[last] = yaml.safe_load(raw)
    else:
        raise Refused(_not_a_container(keys[:-1], container))


def _child(cur: Any, where: list[str], key: str) -> Any:
    """`cur[key]` along an override path; a missing mapping key becomes ``{}``."""
    if isinstance(cur, list):
        return cur[_index(cur, where, key)]
    if isinstance(cur, dict):
        return cur.setdefault(key, {})
    raise Refused(_not_a_container(where, cur))


def _index(seq: list[Any], where: list[str], key: str) -> int:
    name = ".".join(where)
    if not key.isdigit():
        raise Refused(f"{name} is a list; {key!r} is not an index")
    if int(key) >= len(seq):
        entries = "1 entry" if len(seq) == 1 else f"{len(seq)} entries"
        raise Refused(f"{name} has {entries}; index {key} does not exist")
    return int(key)


def _not_a_container(where: list[str], value: Any) -> str:
    return f"{'.'.join(where)} is {type(value).__name__} {value!r}, not a mapping or a list"
