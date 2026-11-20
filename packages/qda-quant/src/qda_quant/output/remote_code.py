"""Carry the source's auto-map modules and local relative imports without executing them."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from ..step import Refused


def copy_remote_code(source: Path, out: Path, configs: list[dict]) -> None:
    """Copy referenced Python files byte for byte, refusing an incomplete source dependency tree.

    Args:
        source: local source checkpoint directory.
        out: exported checkpoint directory.
        configs: exported model and tokenizer/processor configuration dictionaries.
    """
    pending: list[Path] = []
    seen: set[Path] = set()

    def module(parts: tuple[str, ...], *, required: bool = True) -> None:
        if not parts or not all(p.isidentifier() for p in parts):
            raise Refused(f"remote-code export: invalid module {'.'.join(parts)!r}")
        path = Path(*parts)
        choices = [path.with_suffix(".py"), path / "__init__.py"]
        found = next((p for p in choices if (source / p).is_file()), None)
        if found is None:
            if required:
                raise Refused(f"remote-code export: required source module {path}.py is missing in {source}")
            return
        pending.append(found)
        for parent in found.parents:
            init = parent / "__init__.py"
            if parent != Path() and (source / init).is_file():
                pending.append(init)

    def references(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "auto_map" and isinstance(child, dict):
                    for ref in child.values():
                        for entry in ref if isinstance(ref, (list, tuple)) else [ref]:
                            if entry is not None:
                                name = entry.split("--")[-1].rsplit(".", 1)[0]
                                module(tuple(name.split(".")))
                else:
                    references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    for config in configs:
        references(config)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse((source / path).read_bytes(), filename=str(path))
        package = path.parent.parts
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.level > len(package) + 1:
                raise Refused(f"remote-code export: relative import escapes source in {path}")
            base = package[: len(package) - node.level + 1]
            if node.module:
                parts = (*base, *node.module.split("."))
                module(parts)
                # An imported name can be a submodule or an attribute of the module.
                for alias in node.names:
                    if alias.name != "*":
                        module((*parts, alias.name), required=False)
            else:
                for alias in node.names:
                    module((*base, alias.name))
    for path in sorted(seen):
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / path, out / path)


def asset_configs(source: Path, files: tuple[str, ...]) -> list[dict]:
    """Read auto-map-bearing JSON assets the exporter carries beside the model config."""
    return [
        json.loads((source / name).read_text())
        for name in files
        if name.endswith("_config.json") and (source / name).is_file()
    ]
