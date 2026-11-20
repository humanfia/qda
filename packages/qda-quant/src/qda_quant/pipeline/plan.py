"""Load pipeline specs and resolve their stages and dependencies."""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..config import BuildConfig
from ..output import roots as outputs
from ..step import Refused

_BUILD = ("name", "steps", "model", "out", "set")
_COMMAND = ("name", "model", "out", "command")
_TOP = ("name", "recipe", "out", "data", "data_dir", "stages")
FILL = re.compile(r"\{(data|dir|stem|root):([^}]*)\}")
_ANY = re.compile(r"\{(?:data|dir|stem|root|model|out|python)(?::[^}]*)?\}")
_WRITER_ENDINGS = ("-traces", "-answers")


def strings(value: Any, seen: list[str]) -> list[str]:
    """Every string in plain data, in order."""
    if isinstance(value, dict):
        for v in value.values():
            strings(v, seen)
    elif isinstance(value, list):
        for v in value:
            strings(v, seen)
    elif isinstance(value, str):
        seen.append(value)
    return seen


def paths(value: Any, seen: list[Path]) -> list[Path]:
    """Every existing path a config's plain data names, in order, without repeats."""
    if isinstance(value, dict):
        for v in value.values():
            paths(v, seen)
    elif isinstance(value, list):
        for v in value:
            paths(v, seen)
    elif isinstance(value, str) and value:
        try:
            p = Path(value).expanduser()
            exists = p.is_file() or p.is_dir()
        except (OSError, ValueError, RuntimeError):
            return seen
        if exists and p.resolve() not in seen:
            seen.append(p.resolve())
    return seen


@dataclass(frozen=True)
class Data:
    """A named input, optionally pinned or written by a command stage."""

    name: str
    path: Path
    sha256: str | None = None
    produced_by: str | None = None
    raw: str = ""  # the spec's own path, kept so ``--data-dir`` can move a relative one


@dataclass(frozen=True)
class Stage:
    """One recipe slice or command in the chain."""

    name: str
    kind: str  # "build" or "command"
    model: str | None  # "previous", "recipe", "data:NAME", or a path / hub id
    out: Path | None  # None: the data entries ``produced_by`` this stage are its outputs
    steps: tuple[int, int] | None = None  # build stages
    overrides: dict[str, Any] = field(default_factory=dict)  # build stages
    command: list[str] = field(default_factory=list)  # command stages
    out_default: bool = False  # True: ``out`` follows the pipeline's root, so ``--out`` moves it


@dataclass(frozen=True)
class Pipeline:
    """A checked spec with resolved data and stage paths."""

    name: str
    spec: Path
    root: Path
    recipe: Path
    data: dict[str, Data]
    stages: list[Stage]
    recipe_yaml: dict[str, Any]
    data_dir: Path | None = None  # a relative ``data`` path's directory; ``--data-dir`` names another

    def check(self) -> None:
        """Refuse invalid stage ranges, names, outputs and data dependencies."""
        self._check_producers()
        names, outs = set(), set()
        for s in self.stages:
            if s.name in names:
                raise Refused(f"stage {s.name!r}: named twice")
            names.add(s.name)
            written = [str(p) for p in stage_paths(self, s)]
            if not written:
                raise Refused(
                    f"stage {s.name!r}: writes nothing; a command stage writes its ``out`` or the data "
                    "entries whose ``produced_by`` names it"
                )
            for path in written:
                if path in outs:
                    raise Refused(f"stage {s.name!r}: writes {path}, which another stage writes too")
                outs.add(path)
        builds = [s for s in self.stages if s.kind == "build"]
        if not builds:
            raise Refused("pipeline: no build stage; a chain with only commands builds nothing")
        n = len(self.recipe_yaml.get("steps") or ())
        for s in builds:
            assert s.steps is not None  # load() gives every build stage one
            start, stop = s.steps
            if not 0 <= start < stop <= n:
                raise Refused(f"stage {s.name!r}: steps {start}:{stop} is outside the recipe's {n} step(s)")
            if not self.recipe_yaml.get("steps"):
                raise Refused(f"{self.recipe}: no steps; a build stage slices a recipe's steps")
        covered = [i for s in builds for i in range(s.steps[0], s.steps[1])]  # type: ignore[index]
        if covered != list(range(n)):
            gaps = sorted(set(range(n)) - set(covered))
            twice = sorted({i for i in covered if covered.count(i) > 1})
            raise Refused(
                f"pipeline: the build stages must run every recipe step once, in order "
                f"(recipe has {n}); missing {gaps or 'none'}, repeated {twice or 'none'}"
            )
        for i, s in enumerate(self.stages):
            before = self.stages[:i]
            if s.model == "previous" and not before:
                raise Refused(
                    f"stage {s.name!r}: model 'previous' names the stage before it, and it is first"
                )
            if s.model and s.model.startswith("data:") and s.model[5:] not in self.data:
                raise Refused(
                    f"stage {s.name!r}: model {s.model!r} is not a data entry; known: {sorted(self.data)}"
                )

    def check_out(self) -> None:
        """Refuse outputs outside the configured output roots."""
        paths = [(self.root, "the pipeline's out")] + [
            (s.out, f"stage {s.name!r}") for s in self.stages if s.out is not None
        ]
        for path, what in paths:
            try:
                outputs.output_path(path)
            except Refused as e:
                raise Refused(f"{what}: {e}") from None

    def check_data(self) -> None:
        """Refuse missing inputs and mismatched pins; defer inputs a stage writes."""
        from .record import path_sha

        missing, wrong = [], []
        for d in self.data.values():
            if not d.path.exists():
                if not d.produced_by:
                    missing.append(f"{d.name}: {d.path}")
                continue
            if d.sha256:
                got = path_sha(d.path)["sha256"]
                if got != d.sha256:
                    wrong.append(f"{d.name}: {d.path}: {got[:12]}, pinned {d.sha256[:12]}")
        if missing:
            raise Refused("pipeline data, missing input(s): " + "; ".join(missing))
        if wrong:
            raise Refused("pipeline data, the pin does not match: " + "; ".join(wrong))

    def _check_producers(self) -> None:
        dumps = {s.name: cfg_dump(config(self, s)) for s in self.stages if s.kind == "build"}
        produced = {d.name: d for d in self.data.values() if d.produced_by}
        if not produced:
            return
        at = {s.name: i for i, s in enumerate(self.stages)}
        for d in produced.values():
            assert d.produced_by is not None
            if d.produced_by not in at:
                raise Refused(
                    f"data.{d.name}: produced_by {d.produced_by!r} is not a stage; known: {sorted(at)}"
                )
            if self.stages[at[d.produced_by]].kind != "command":
                raise Refused(
                    f"data.{d.name}: produced_by {d.produced_by!r} is a build stage; only a command "
                    "writes one file of the chain"
                )
        for i, s in enumerate(self.stages):
            if s.kind == "build":
                reads = set(strings(dumps[s.name], []))
            else:  # a command names its inputs by entry, in a {data:}, {dir:} or {stem:} placeholder
                reads = {
                    str(self.data[m.group(2)].path)
                    for a in s.command
                    for m in FILL.finditer(a)
                    if m.group(2) in self.data and self.data[m.group(2)].produced_by != s.name
                }
            for d in produced.values():
                if str(d.path) not in reads:
                    continue
                assert d.produced_by is not None
                if at[d.produced_by] >= i:
                    raise Refused(
                        f"stage {s.name!r} reads data.{d.name} ({d.path}), which stage "
                        f"{d.produced_by!r} writes after it; put the writer before the reader"
                    )


def stage_paths(p: Pipeline, s: Stage) -> list[Path]:
    """Return each path a stage writes, in declaration order without duplicates."""
    paths = [s.out] if s.out is not None else []
    paths += [d.path for d in p.data.values() if d.produced_by == s.name]
    out: list[Path] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


def _stage_entry(entry: Any, where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise Refused(f"{where}: expected a mapping, got {type(entry).__name__}")
    kind = "command" if "command" in entry else "build"
    allowed = _COMMAND if kind == "command" else _BUILD
    unknown = sorted(set(entry) - set(allowed))
    if unknown:
        raise Refused(f"{where}: unknown key(s) {unknown}; a {kind} stage takes {list(allowed)}")
    return {**entry, "kind": kind}


def _slice(stub: Any, where: str, n: int) -> tuple[int, int]:
    if stub == "all" or stub is None:
        return 0, n
    if isinstance(stub, str):
        try:
            start, stop = (int(x) for x in stub.split(":"))
        except ValueError:
            raise Refused(f"{where}: steps {stub!r} is not 'START:STOP' or 'all'") from None
        return start, stop
    if isinstance(stub, list) and len(stub) == 2 and all(isinstance(x, int) for x in stub):
        return stub[0], stub[1]
    raise Refused(f"{where}: steps {stub!r} is not 'START:STOP' or 'all'")


def _out_path(base: Path, raw: str, where: str) -> Path:
    allowed = outputs.roots()
    if allowed and not Path(raw).expanduser().is_absolute():
        return outputs.output_path(allowed[0] / Path(raw).expanduser())
    try:
        return outputs.output_path((base / Path(raw).expanduser()).resolve())
    except Refused as e:
        raise Refused(f"{where}: {e}") from None


def load(path: str | Path) -> Pipeline:
    """Read and validate a pipeline spec and resolve its paths."""
    spec = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(spec.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise Refused(f"{spec}: {e}; check the path and that the file is YAML") from None
    if not isinstance(raw, dict):
        raise Refused(f"{spec}: expected a mapping of keys, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_TOP))
    if unknown:
        raise Refused(f"{spec}: unknown key(s) {unknown}; a pipeline spec takes {list(_TOP)}")
    base = spec.parent
    if "out" not in raw:
        raise Refused(f"{spec}: missing required key 'out' (the pipeline's directory)")
    root = _out_path(base, str(raw["out"]), "out")
    entries = raw.get("stages")
    if not isinstance(entries, list) or not entries:
        if isinstance(entries, dict):
            raise Refused(f"{spec}: 'stages' must be a list; each entry is one stage")
        raise Refused(f"{spec}: 'stages' must be a non-empty list")
    if "recipe" not in raw:
        raise Refused(f"{spec}: missing required key 'recipe' (a build stage slices its steps)")
    recipe = (base / str(raw["recipe"])).resolve()
    try:
        recipe_yaml = yaml.safe_load(recipe.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise Refused(f"{spec}: recipe {recipe}: {e}") from None
    if not isinstance(recipe_yaml, dict):
        raise Refused(f"{recipe}: expected a mapping of keys, got {type(recipe_yaml).__name__}")
    data_dir = None
    if "data_dir" in raw:
        if not isinstance(raw["data_dir"], str) or not raw["data_dir"]:
            raise Refused(f"{spec}: 'data_dir' must be a directory path")
        data_dir = Path(raw["data_dir"]).expanduser()
        data_dir = data_dir if data_dir.is_absolute() else (base / data_dir).resolve()
    data: dict[str, Data] = {}
    entries_data = raw.get("data") or {}
    if not isinstance(entries_data, dict):
        raise Refused(f"{spec}: 'data' must be a mapping of NAME to a path or to {{path, sha256}}")
    for key, value in entries_data.items():
        if isinstance(value, dict):
            extra = sorted(set(value) - {"path", "sha256", "produced_by"})
            if extra:
                raise Refused(
                    f"{spec}: data.{key}: unknown key(s) {extra}; takes ['path', 'sha256', 'produced_by']"
                )
            p, pin, producer = value.get("path"), value.get("sha256"), value.get("produced_by")
        else:
            p, pin, producer = value, None, None
        if not isinstance(p, str) or not p:
            raise Refused(f"{spec}: data.{key}: needs a path")
        if producer is not None and not isinstance(producer, str):
            raise Refused(f"{spec}: data.{key}: produced_by must name a stage")
        under = data_dir if data_dir is not None else recipe.parent
        data[str(key)] = Data(
            str(key), (under / Path(p).expanduser()).resolve(), str(pin) if pin else None, producer, str(p)
        )
    n = len(recipe_yaml.get("steps") or ())
    stages = []
    for i, entry in enumerate(entries):
        where = f"{spec}: stages[{i}]"
        k = _stage_entry(entry, where)
        if not isinstance(k.get("name"), str) or not k["name"]:
            raise Refused(f"{where}: needs a name")
        produces = any(d.produced_by == k["name"] for d in data.values())
        if "out" in k:
            out = _out_path(base, str(k["out"]), f"{where}.out")
        elif produces:
            out = None  # the data entries it writes are its outputs
        else:
            out = root / k["name"]
        if k["kind"] == "command":
            cmd = k.get("command")
            argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd) if isinstance(cmd, list) else None
            if not argv or not all(isinstance(a, str) for a in argv):
                raise Refused(f"{where}: command must be a string or a list of strings")
            stages.append(
                Stage(
                    k["name"],
                    "command",
                    k.get("model"),
                    out,
                    command=argv,
                    out_default="out" not in k and not produces,
                )
            )
        else:
            start, stop = _slice(k.get("steps"), where, n)
            stages.append(
                Stage(
                    k["name"],
                    "build",
                    k.get("model"),
                    out,
                    (start, stop),
                    dict(k.get("set") or {}),
                    out_default="out" not in k,
                )
            )
    p = Pipeline(str(raw.get("name") or spec.stem), spec, root, recipe, data, stages, recipe_yaml, data_dir)
    p.check()
    return p


def with_root(p: Pipeline, root: Path) -> Pipeline:
    """The same spec writing its stage outputs under another root (``--out``); an explicit stage ``out`` stays."""
    root = _out_path(Path.cwd(), str(root), "--out")
    stages = [replace(s, out=root / s.name) if s.out_default else s for s in p.stages]
    moved = replace(p, root=root, stages=stages)
    moved.check()
    return moved


def with_data_dir(p: Pipeline, data_dir: Path) -> Pipeline:
    """The same spec reading and writing its ``data`` entries under another directory (``--data-dir``)."""
    base = Path(data_dir).expanduser()
    base = base if base.is_absolute() else (Path.cwd() / base).resolve()

    def moved(d: Data) -> Data:
        raw = Path(d.raw).expanduser()
        return replace(d, path=raw if raw.is_absolute() else (base / raw).resolve())

    moved_p = replace(p, data_dir=base, data={name: moved(d) for name, d in p.data.items()})
    moved_p.check()
    return moved_p


def _stem(path: Path) -> Path:
    bare = path.with_suffix("")
    for end in _WRITER_ENDINGS:
        if bare.name.endswith(end) and len(bare.name) > len(end):
            return bare.with_name(bare.name[: -len(end)])
    return bare


def fill(p: Pipeline, text: str) -> str:
    """A path with its ``{data:NAME}``, ``{dir:NAME}``, ``{stem:NAME}`` and ``{root}`` placeholders filled."""

    def one(m: re.Match[str]) -> str:
        kind, name = m.group(1), m.group(2)
        d = p.data.get(name)
        if d is None:
            raise Refused(f"{{{kind}:{name}}}: not a data entry; known: {sorted(p.data)}")
        return str({"data": d.path, "dir": d.path.parent, "stem": _stem(d.path)}[kind])

    return FILL.sub(one, text.replace("{root}", str(p.root)))


def _fill_value(p: Pipeline, value: Any) -> Any:
    if isinstance(value, str):
        text = fill(p, value)
        if left := _ANY.search(text):
            raise Refused(
                f"set: {value!r}: {left.group(0)} is not filled here; a set value takes {{data:NAME}}, "
                "{dir:NAME}, {stem:NAME} and {root}"
            )
        return text
    if isinstance(value, list):
        return [_fill_value(p, v) for v in value]
    if isinstance(value, dict):
        return {k: _fill_value(p, v) for k, v in value.items()}
    return value


def model_of(p: Pipeline, s: Stage) -> str:
    """The source a stage reads: the recipe's own, the stage before, a data entry, or a path."""
    i = next((n for n, x in enumerate(p.stages) if x is s), -1)

    def chained() -> str:
        if s.kind == "command":
            prior = [x for x in p.stages[:i] if x.out is not None]
            return str(prior[-1].out) if prior else str(p.recipe_yaml.get("model", ""))
        builds = [x for x in p.stages[:i] if x.kind == "build"]
        return str(builds[-1].out) if builds else str(p.recipe_yaml.get("model", ""))

    if s.model is None or s.model == "previous":
        return chained()
    if s.model == "recipe":
        return str(p.recipe_yaml.get("model", ""))
    if s.model.startswith("data:"):
        return str(p.data[s.model[5:]].path)
    local = p.spec.parent / s.model
    return str(local.resolve()) if local.is_dir() else s.model


def config(p: Pipeline, s: Stage) -> BuildConfig:
    """The `BuildConfig` a build stage runs: its slice of the recipe, its source and its own output."""
    assert s.steps is not None  # load() gives every build stage one
    raw = dict(p.recipe_yaml)
    steps = raw.get("steps") or []
    raw["steps"] = steps[s.steps[0] : s.steps[1]]
    raw["model"] = model_of(p, s)
    raw["out"] = str(s.out)
    overrides = [f"{k}={json.dumps(_fill_value(p, v))}" for k, v in s.overrides.items()]
    return BuildConfig.from_dict(raw, overrides, name=str(p.recipe), base=p.recipe.parent)


def cfg_dump(cfg: BuildConfig) -> dict[str, Any]:
    """The config as plain data, without its own output path and without the run's provenance."""
    return cfg.model_dump(mode="json", exclude={"out", "provenance"})


def chain_outs(p: Pipeline) -> set[str]:
    """The checkpoints this chain builds, which a later stage chains on rather than reads as an input."""
    return {str(s.out) for s in p.stages if s.kind == "build"}


def _served(p: Pipeline, s: Stage) -> str:
    model = model_of(p, s)
    if s.model == "recipe" and "@" in model:
        from ..engine.planning import local_source

        return local_source(model)
    return model


def _out_of(p: Pipeline, s: Stage) -> Path:
    if s.out is not None:
        return s.out
    written = stage_paths(p, s)
    if written:
        return written[0]
    raise Refused(
        f"stage {s.name!r}: {{out}} names nothing; the stage sets no ``out`` and writes no data entry"
    )


def expand(p: Pipeline, s: Stage) -> list[str]:
    """A command stage's argv with its placeholders filled."""
    out = []
    for a in s.command:
        a = fill(p, a)
        if "{model}" in a:  # resolved only for a command that serves the model: a hub pin costs a lookup
            a = a.replace("{model}", _served(p, s))
        if "{out}" in a:
            a = a.replace("{out}", str(_out_of(p, s)))
        a = a.replace("{python}", sys.executable)
        if left := _ANY.search(a):
            raise Refused(
                f"stage {s.name!r}: command argument {a!r} still holds {left.group(0)}; a command takes "
                "{model}, {out}, {root}, {python}, {data:NAME}, {dir:NAME} and {stem:NAME}"
            )
        out.append(a)
    return out
