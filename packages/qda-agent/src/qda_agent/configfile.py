"""Load and edit a YAML configuration file: dotted overrides, placeholders and base-relative paths.

`RunConfig` (`qda_agent.config`) loads its YAML through these; a second config class can reuse them as is.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

__all__ = [
    "ConfigError",
    "PATHS",
    "PLACEHOLDER",
    "load",
    "placeholders",
    "read",
    "set_key",
]

#: Path settings made absolute (written in the file: against its directory; given with `--set`: against the
#: current directory), as key paths (`*` = every list item or value).
PATHS: tuple[str, ...] = (
    "run_dir",
    "work_dir",
    "task",
    "baselines.*",
    "agent.backend.sign_in_dir",
    "agent.fallbacks.*.sign_in_dir",
    "judge.evaluator",
    "judge.custody_dir",
    "judge.starts.*.path",
    "judge.format_reference",
    "judge.format_map",
    "judge.kl_ref",
    "judge.kl_ref.file.path",
    "judge.kl_holdout_ref",
    "judge.anchor",
    "judge.llamacpp.bridge",
    "judge.llamacpp.llama_cpp",
    "judge.llamacpp.server",
    "judge.llamacpp.template",
    "judge.llamacpp.imatrix",
    "judge.gencheck_prompts",
    "judge.gencheck_prompts.file.path",
    "judge.final_rules.*.suite.sanity.prompts",
    "judge.suite.sanity.prompts",
    "contamination.guard",
    "scheduler.readable.*",
    "sandbox.hide.*",
    "toolset.python",
    "toolset.components.*.python",  # serve
    "toolset.components.*.sets.*",  # data
    "toolset.components.*.package",  # source
    "judge.serve_python",
)


def load(config: Any, path: str | Path, overrides: list[str] | dict[str, Any] | None = None) -> Any:
    """Load a YAML config: its paths resolved, overrides applied (`read`), then validated.

    Args:
        config: The pydantic model class to validate.
        path: The YAML file.
        overrides: `KEY.PATH=VALUE` strings (`--set`; the value is YAML), or a mapping of key paths to values,
            applied before validation; they may only replace or add keys.

    Returns:
        The validated config.

    Raises:
        ConfigError: The file cannot be read, a value is still a `<...>` placeholder (every such key is named,
            before anything is made), or the config is invalid.
    """
    raw = read(path, overrides)
    try:
        return config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}: " + "; ".join(_reason(err) for err in e.errors())) from e


def read(
    path: str | Path,
    overrides: list[str] | dict[str, Any] | None = None,
    *,
    need: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Read a config file unvalidated: resolve its paths, apply overrides, refuse placeholders.

    A relative path written in the file resolves against the file's directory; a relative path given in
    `overrides` (`--set run_dir=runs/a`) resolves against the current directory.

    Args:
        path: The YAML file.
        overrides: As `load`'s.
        need: Refuse placeholders only in these key paths and below them (None: in every key). A command that
            reads a few keys (`qda-agent cand0` builds from `model`, `format`, `calib` and the `qda` command)
            runs before the config's other inputs exist, so it must not be refused for them.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a run config is a mapping")
    raw.pop("provenance", None)  # a resolved.yaml's record
    # A relative path written in the file is relative to the file's directory; one given with `--set`, to the
    # current directory. The file's paths are made absolute first, so the second pass changes only the set ones.
    _resolve_paths(raw, path.resolve().parent)
    items = list(overrides.items() if isinstance(overrides, dict) else map(_split, overrides or []))
    for key, value in items:
        set_key(raw, key, value)
    _resolve_paths(raw, Path.cwd(), local={key for key, _ in items})
    todo = placeholders(raw)
    if need is not None:
        todo = [k for k in todo if any(k == n or k.startswith((f"{n}.", f"{n}[")) for n in need)]
    if todo:
        raise ConfigError(
            f"{path}: fill in {', '.join(todo)} (still a <...> placeholder; edit the file or "
            "--set KEY.PATH=VALUE)"
        )
    return raw


def _resolve_paths(raw: dict[str, Any], base: Path, local: set[str] | None = None) -> None:
    """Make the `PATHS` settings, and a local `model` or `calib`, absolute against `base`, in place.

    An absolute path is left as it is, so a second pass changes only what is still relative. `local`: the keys
    whose `model` / `calib` may be read as a local source here (None: both); a hub or dataset name set in the
    file must not turn into a path because the current directory happens to hold one by that name.
    """
    for key in PATHS:
        resolve = _python if key.endswith("python") else _spelling if key == "sandbox.hide.*" else _absolute
        _resolve(raw, key.split("."), base, resolve)
    for key in ("model", "calib"):
        if local is not None and key not in local:
            continue
        if isinstance(raw.get(key), (str, Path)) and (base / Path(raw[key]).expanduser()).exists():
            raw[key] = _absolute(base, raw[key])  # a local source; leave hub/dataset names alone
    if isinstance(raw.get("sandbox"), dict) and (hide := raw["sandbox"].get("hide")):
        raw["sandbox"]["hide"] = list(dict.fromkeys(p for h in hide for p in (h, _absolute(base, h))))


class ConfigError(ValueError):
    """Report a run configuration error with its file or affected key paths; `fix`, when known, makes it load."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


#: Placeholder syntax used by the example search and ablation configs: `<path/to/what>`.
PLACEHOLDER = re.compile(r"<[^<>\s]+>")


def placeholders(node: Any, key: str = "") -> list[str]:
    """List key paths of strings containing placeholders, such as `baselines[0]` or `judge.kl_ref`."""
    if isinstance(node, dict):
        return [k for name, v in node.items() for k in placeholders(v, f"{key}.{name}" if key else str(name))]
    if isinstance(node, list):
        return [k for i, v in enumerate(node) for k in placeholders(v, f"{key}[{i}]")]
    return [key] if isinstance(node, str) and PLACEHOLDER.search(node) else []


def _reason(err: Any) -> str:
    """Format a pydantic error as `key.path: message`, omitting an empty path."""
    where = ".".join(str(part) for part in err["loc"])
    msg = err["msg"].removeprefix("Value error, ")
    return f"{where}: {msg}" if where else msg


def _split(item: str) -> tuple[str, Any]:
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise ConfigError(f"--set {item!r}: expected KEY.PATH=VALUE")
    try:
        return key, yaml.safe_load(value)
    except yaml.YAMLError as e:
        raise ConfigError(f"--set {item!r}: the value is not YAML ({' '.join(str(e).split())})") from None


def set_key(raw: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted configuration key, creating missing mappings along its path.

    A plugin given by its bare key (`sandbox: bwrap`) becomes `{name: bwrap}` first, so its settings can be set.

    Raises:
        ConfigError: A part of the path is not a mapping (a list or a scalar).
    """
    *parents, last = key.split(".")
    node = raw
    for i, part in enumerate(parents):
        child: Any = node.setdefault(part, {})
        if isinstance(child, str):
            child = node[part] = {"name": child}
        if not isinstance(child, dict):
            raise ConfigError(f"--set {key}: {'.'.join(parents[: i + 1])} is not a mapping")
        node = child
    node[last] = value


def _resolve(node: Any, parts: list[str], base: Path, resolve: Callable[[Path, str], str]) -> None:
    """Resolve configured paths against `base` in place, expanding `*` across lists and mappings."""
    head, rest = parts[0], parts[1:]
    if head == "*":
        keys = range(len(node)) if isinstance(node, list) else list(node) if isinstance(node, dict) else []
        for i in keys:
            if rest:
                _resolve(node[i], rest, base, resolve)
            elif isinstance(node[i], (str, Path)):
                node[i] = resolve(base, str(node[i]))
        return
    if not isinstance(node, dict) or node.get(head) is None:
        return
    if rest:
        _resolve(node[head], rest, base, resolve)
    elif isinstance(node[head], (str, Path)):
        node[head] = resolve(base, str(node[head]))


def _absolute(base: Path, value: str) -> str:
    """Canonicalize a configured path, including existing parents of a future output directory."""
    return str((base / Path(value).expanduser()).resolve())


def _spelling(base: Path, value: str) -> str:
    """Keep the named spelling of a hide path as well as its canonical target."""
    return os.path.abspath(base / Path(value).expanduser())  # noqa: PTH100 (resolve() follows the link)


def _python(base: Path, value: str) -> str:
    """Canonicalize the interpreter's parents while keeping its venv-selecting executable entry.

    Resolving `bin/python` itself selects the base interpreter and loses the virtualenv's packages.
    """
    path = base / Path(value).expanduser()
    return str(path.parent.resolve() / path.name)
