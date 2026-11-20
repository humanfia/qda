"""Hash pipeline inputs and outputs, write records and check resumability."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..calib import sha256_file
from ..config import BuildConfig
from ..step import Refused
from .plan import Pipeline, Stage, cfg_dump, paths, stage_paths

RECORD = "pipeline.json"
FULL_HASH_LIMIT = 64 << 20


def now() -> str:
    """The time, as a record quotes it."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> dict[str, Any]:
    """A file's sha256 and size."""
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size, "kind": "file"}


def _dir_sha(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    total = files = 0
    for p in sorted(path.iterdir(), key=lambda q: q.name):
        if p.is_dir():
            h.update(f"d {p.name}\n".encode())
            continue
        size = p.stat().st_size
        h.update(f"f {p.name} {size}\n".encode())
        total += size
        files += 1
        if size <= FULL_HASH_LIMIT:
            h.update(file_sha(p)["sha256"].encode())
    return {"sha256": h.hexdigest(), "bytes": total, "files": files, "kind": "dir"}


def path_sha(path: Path) -> dict[str, Any]:
    """`_file_sha` for a file, `_dir_sha` for a directory; refuses anything else."""
    if path.is_dir():
        return _dir_sha(path)
    if path.is_file():
        return file_sha(path)
    raise Refused(f"{path}: no such file or directory")


def _inputs(paths: list[Path], outs: set[str]) -> dict[str, dict[str, Any]]:
    return {str(p): path_sha(p) for p in paths if str(p) not in outs}


def build_key(cfg: BuildConfig, s: Stage, outs: set[str]) -> dict[str, Any]:
    """What makes this stage this stage: its config (paths resolved, output excluded) and its inputs."""
    return {
        "steps": list(s.steps or ()),
        "config": cfg_dump(cfg),
        "inputs": {k: v["sha256"] for k, v in _inputs(paths(cfg_dump(cfg), []), outs).items()},
    }


def _holders(written: list[Path]) -> set[str]:
    return {str(parent) for path in written for parent in path.parents}


def command_key(p: Pipeline, s: Stage, argv: list[str], outs: set[str]) -> dict[str, Any]:
    """A command stage's key: its argv and the hash of every path in it that exists and the stage did not write."""
    written = stage_paths(p, s)
    skip = outs | {str(path) for path in written} | _holders(written)
    return {
        "command": argv,
        "model": s.model,
        "out": str(s.out),
        "inputs": {k: v["sha256"] for k, v in _inputs(list(paths(argv, [])), skip).items()},
    }


def recorded_data(p: Pipeline) -> dict[str, Any]:
    """Record each declared input by its hash or by the stage that will write it."""
    out: dict[str, Any] = {}
    for d in p.data.values():
        entry: dict[str, Any] = {"path": str(d.path)}
        if d.path.exists():
            entry.update(path_sha(d.path))
            if d.sha256 and entry["sha256"] != d.sha256:
                raise Refused(f"data.{d.name}: {d.path}: {entry['sha256'][:12]}, pinned {d.sha256[:12]}")
        else:
            entry["produced_by"] = d.produced_by
        out[d.name] = entry
    return out


def read_record(root: Path) -> dict[str, Any]:
    """The pipeline record, or an empty one before the first stage has finished."""
    path = root / RECORD
    if not path.exists():
        return {"stages": {}, "history": []}
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise Refused(
            f"{path}: {e}; it is this pipeline's record, so it is not rebuilt from scratch"
        ) from None
    if not isinstance(record, dict):
        raise Refused(f"{path}: expected a mapping of keys, got {type(record).__name__}")
    record.setdefault("stages", {})
    record.setdefault("history", [])
    return record


def write_record(p: Pipeline, record: dict[str, Any]) -> Path:
    """Rewrite the record atomically: a crash mid-write leaves the previous one, never half a file."""
    p.root.mkdir(parents=True, exist_ok=True)
    path = p.root / RECORD
    tmp = path.with_name(f".{RECORD}.{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.replace(path)
    return path


def _changed(mine: Any, theirs: Any) -> list[str]:
    if not isinstance(mine, dict) or not isinstance(theirs, dict):
        return [f"recorded {str(theirs)[:12]}, now {str(mine)[:12]}"]
    return [
        f"{k}: recorded {str(theirs.get(k))[:12]}, now {str(mine.get(k))[:12]}"
        for k in sorted(set(mine) | set(theirs))
        if mine.get(k) != theirs.get(k)
    ][:5]


def resume(s: Stage, record: dict[str, Any], key: dict[str, Any], paths: list[Path]) -> tuple[bool, str]:
    """Whether the recorded stage is this stage, and can be skipped; ``(True, "")`` means skip."""
    was = record["stages"].get(s.name)
    if was is None:
        return False, ""
    if s.kind == "command" and isinstance(was.get("key", {}).get("inputs"), dict):
        # a record written before `_holders` hashed the directory the stage writes into; compare without it
        held = _holders(paths)
        was = {
            **was,
            "key": {**was["key"], "inputs": {k: v for k, v in was["key"]["inputs"].items() if k not in held}},
        }
    if was.get("key") != key:
        for name, mine, theirs in (
            ("config", key.get("config", {}), was.get("key", {}).get("config", {})),
            ("inputs", key["inputs"], was.get("key", {}).get("inputs", {})),
        ):
            if mine != theirs:
                return False, (
                    f"stage {s.name!r}: its {name} changed since the run that wrote {paths[0]}: "
                    + "; ".join(_changed(mine, theirs))
                )
        return False, f"stage {s.name!r}: its steps or command changed since the run that wrote {paths[0]}"
    written = was.get("outputs") or {}
    if not written and was.get("output"):  # a record from before a stage could write several files
        written = {was.get("out"): was.get("output")}
    for path in paths:
        if not path.exists():
            return False, ""  # recorded and unchanged, but an output is gone: rebuild rather than skip
        was_sha = (written.get(str(path)) or {}).get("sha256")
        if path_sha(path)["sha256"] != was_sha:
            return False, (
                f"stage {s.name!r}: {path} changed since the pipeline wrote it (recorded "
                f"{str(was_sha)[:12]}, now {path_sha(path)['sha256'][:12]}); "
                "a stage never overwrites an output it did not write"
            )
    return True, ""
