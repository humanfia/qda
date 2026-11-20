"""The workspace manifest: one record of everything a run gives the agent (commands, mounts, files, texts).

`prepare` writes it to `run_dir/manifest.json`; the contamination check, wall and scheduler read it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RunConfig
    from .toolsets import Toolset

__all__ = [
    "CONTENT_MAX",
    "MANIFEST",
    "Manifest",
    "Mount",
    "build_manifest",
    "content_digest",
    "file_sha256",
    "model_name",
    "model_source",
    "read_only",
    "tree_sha256",
]

#: The manifest's file name in the run dir (outside the agent's workdir).
MANIFEST = "manifest.json"

#: Bumped when the record's shape changes; a reader refuses an unknown one. 2 added each mount's `digest`
#: (how its `sha256` was made) and the agent's private socket bind.
VERSION = 2


def file_sha256(path: Path) -> str:
    """The SHA-256 of `path`'s bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Largest file a bind's digest reads for its bytes. Above it the entry records its size and modification
#: time instead: a baseline checkpoint is tens of gigabytes, and reading it twice per run would cost more than
#: the record is worth. A file at or below it is identified by its content, so editing it changes the digest.
CONTENT_MAX = 32 << 20


def content_digest(path: Path) -> tuple[str, str]:
    """One file's digest and the method that made it: its bytes, or size+mtime above `CONTENT_MAX`."""
    stat = path.stat()
    if stat.st_size <= CONTENT_MAX:
        return file_sha256(path), "sha256"
    return f"{stat.st_size}:{stat.st_mtime_ns}", "size+mtime"


def read_only(path: Path) -> None:
    """Make a staged tree read-only (files 0444, directories 0555) without following symlinks.

    Directories too, because removing a file needs write access to its directory, not to the file.
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            p = Path(root, name)
            if not p.is_symlink():
                p.chmod(0o444)
        for name in dirs:
            p = Path(root, name)
            if not p.is_symlink():
                p.chmod(0o555)
    path.chmod(0o555)


def _walk(
    root: Path,
    current: Path,
    rows: list[tuple[str, str]],
    methods: set[str],
    seen: set[Path],
) -> int:
    """Walk `current` under `root`, recording `(relpath, entry-digest)` for every file and link; return bytes.

    A link records its target (and a file target's digest); directory links are followed only inside `root`.
    """
    total = 0
    for entry in sorted(current.iterdir()):
        rel = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            target = os.readlink(entry)
            if entry.is_file():
                digest, method = content_digest(entry.resolve())
                rows.append((rel, f"link:{target}\0{digest}"))
                methods.add(method)
                total += entry.resolve().stat().st_size
            else:
                rows.append((rel, f"link:{target}"))
                if entry.is_dir():
                    resolved = entry.resolve()
                    if resolved.is_relative_to(root.resolve()) and resolved not in seen:
                        seen.add(resolved)
                        total += _walk(root, entry, rows, methods, seen)
        elif entry.is_dir():
            resolved = entry.resolve()
            if resolved not in seen:
                seen.add(resolved)
                total += _walk(root, entry, rows, methods, seen)
        elif entry.is_file():
            digest, method = content_digest(entry)
            rows.append((rel, digest))
            methods.add(method)
            total += entry.stat().st_size
    return total


def tree_sha256(path: Path) -> tuple[str, int, int, str]:
    """A directory's digest, file count, byte count and the method its entries used.

    Files above `CONTENT_MAX` contribute size and mtime instead of content; the method is then `size+mtime`.
    """
    rows: list[tuple[str, str]] = []
    methods: set[str] = set()
    root = path
    seen = {root.resolve()}
    total = _walk(root, root, rows, methods, seen)
    h = hashlib.sha256()
    for rel, digest in sorted(rows):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest(), len(rows), total, "size+mtime" if "size+mtime" in methods else "sha256"


def model_name(model: str | Path) -> str:
    """How a record names a model: its resolved local directory, or a Hub id as given.

    `prepare`, `cand0` and the judge all name models with it, so their records compare as strings.
    """
    path = Path(model).expanduser()
    return str(path.resolve()) if path.is_dir() else str(model)


def model_source(model: str | Path) -> dict[str, str | None]:
    """A record's ``source`` (`model_name`) and ``source_sha256`` (`tree_sha256`; None for a Hub id)."""
    name = model_name(model)
    path = Path(name)
    return {"source": name, "source_sha256": tree_sha256(path)[0] if path.is_dir() else None}


@dataclass(frozen=True)
class Mount:
    """One read-only bind: where the agent sees it, what it exposes, and that source's digest.

    Attributes:
        sha256: A file's content digest or a directory's `tree_sha256`; empty in a plan (a manifest built
            without touching the file system).
        digest: How `sha256` was made: `sha256`, or `size+mtime` for a file above `CONTENT_MAX`.
    """

    dest: str
    source: str
    kind: str
    sha256: str = ""
    digest: str = ""
    files: int = 0
    bytes: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "dest": self.dest,
            "source": self.source,
            "kind": self.kind,
            "sha256": self.sha256,
            "digest": self.digest,
            "files": self.files,
            "bytes": self.bytes,
        }


@dataclass
class Manifest:
    """Everything the run gives the agent, as `build_manifest` collects it.

    Attributes:
        subcommands: For each command, the verbs this toolset adds.
        starts: The row's pre-built starts as `{name, path, origin}`; TASK.md shows the names only.
        mounts: `{"agent": {dest: Mount}, "jobs": {dest: Mount}}`: the shell's binds and the jobs' binds.
        files: Every file in the workdir and the overlay, absolute path -> SHA-256.
        texts: The generated texts the run wrote, path relative to the run dir -> SHA-256.
    """

    workdir: str
    run_dir: str
    commands: list[str] = field(default_factory=list)
    subcommands: dict[str, list[str]] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    inputs: dict[str, list[str]] = field(default_factory=dict)
    starts: list[dict[str, str]] = field(default_factory=list)
    mounts: dict[str, dict[str, Mount]] = field(default_factory=lambda: {"agent": {}, "jobs": {}})
    readable: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)

    def has(self, command: str) -> bool:
        """Whether the run gives the agent `command`."""
        return command in self.commands

    def verb(self, command: str, verb: str) -> bool:
        """Whether the run gives the agent `command verb`."""
        return verb in self.subcommands.get(command, ())

    def paths(self) -> list[Path]:
        """Every file the manifest records, as paths (what the contamination scan reads)."""
        return [Path(p) for p in self.files]

    def agent_binds(self) -> dict[Path, Path]:
        """The shell's read-only binds: workdir destination -> host source."""
        return {Path(m.dest): Path(m.source) for m in self.mounts.get("agent", {}).values()}

    def job_binds(self) -> dict[Path, Path]:
        """The jobs' read-only binds: workdir destination -> host source."""
        return {Path(m.dest): Path(m.source) for m in self.mounts.get("jobs", {}).values()}

    def as_json(self) -> dict[str, Any]:
        """The record as one JSON-serializable object (sorted, so two runs compare byte for byte)."""
        return {
            "version": VERSION,
            "workdir": self.workdir,
            "run_dir": self.run_dir,
            "commands": self.commands,
            "subcommands": {k: self.subcommands[k] for k in sorted(self.subcommands)},
            "tools": self.tools,
            "inputs": {k: self.inputs[k] for k in sorted(self.inputs)},
            "starts": list(self.starts),
            "mounts": {
                which: {
                    dest: self.mounts[which][dest].as_json() for dest in sorted(self.mounts.get(which, {}))
                }
                for which in ("agent", "jobs")
            },
            "readable": self.readable,
            "env": {k: self.env[k] for k in sorted(self.env)},
            "files": {k: self.files[k] for k in sorted(self.files)},
            "texts": {k: self.texts[k] for k in sorted(self.texts)},
        }

    def write(self, path: Path) -> str:
        """Write the record to `path` and return its SHA-256."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_json(), indent=2) + "\n")
        return file_sha256(path)


def _mount(dest: Path, source: Path, *, snapshot: bool) -> Mount:
    """Record one bind; its digest is read only when `snapshot` is set."""
    kind = "dir" if source.is_dir() else "file"
    if not snapshot:
        return Mount(dest=str(dest), source=str(source), kind=kind)
    if kind == "file":
        digest, method = content_digest(source)
        return Mount(
            dest=str(dest),
            source=str(source),
            kind=kind,
            sha256=digest,
            digest=method,
            files=1,
            bytes=source.stat().st_size,
        )
    digest, files, size, method = tree_sha256(source)
    return Mount(
        dest=str(dest),
        source=str(source),
        kind=kind,
        sha256=digest,
        digest=method,
        files=files,
        bytes=size,
    )


def build_manifest(
    cfg: RunConfig,
    toolset: Toolset,
    *,
    texts: dict[str, str] | None = None,
    snapshot: bool = False,
) -> Manifest:
    """Collect what `cfg` gives the agent through `toolset`.

    Args:
        texts: The generated texts, path relative to the run dir -> path to hash.
        snapshot: Hash every file and bind source; False builds the plan alone, leaving `files` and each
            bind's `sha256` empty, without touching large trees.
    """
    run = cfg.run
    workdir, run_dir = cfg.workdir, cfg.run_dir
    # The shell's binds are the overlay parents; the jobs' binds add each component's own (a baseline, say).
    agent = toolset.mounts(run, jobs=False)
    jobs = toolset.mounts(run, jobs=True)
    # What the jobs read: the scheduler's settings, the private socket directory of an unwalled run, the
    # baselines not already bound under the workdir, the components' own inputs, and the run's calibration file.
    bound = {v for v in jobs.values() if v.exists()}
    private = cfg.socket_binds()
    readable = [
        *cfg.scheduler.model_dump().get("readable", []),
        *([] if cfg.sandbox.walled else private.values()),
        *(b for b in cfg.baselines if b.exists() and b not in bound),
        *toolset.readable(),
        # the calibration file the run names (prepare's agent-calib.jsonl): the agent's builds read it
        *(
            [Path(cfg.calib)]
            if cfg.calib and Path(cfg.calib).exists() and Path(cfg.calib) not in bound
            else []
        ),
    ]
    if cfg.sandbox.walled:
        # The wall mounts the private socket directory over `work/.qda` for the shell too
        # (`isolation/wall.py`), not only for the jobs: the manifest lists every bind of both.
        agent = {**agent, **private}
        jobs = {**jobs, **private}
    tools = toolset.sections(run)
    names = toolset.baseline_names(run) or [str(b) for b in cfg.baselines]
    manifest = Manifest(
        workdir=str(workdir),
        run_dir=str(run_dir),
        commands=list(toolset.commands(run)),
        subcommands={k: list(v) for k, v in toolset.subcommands(run).items()},
        tools=list(tools),
        inputs={
            "model": [cfg.model] if cfg.model else [],
            "calib": [cfg.calib] if cfg.calib else [],
            "baselines": list(names),
        },
        starts=[
            {"name": start.name, "path": str(start.path), "origin": start.origin}
            for start in cfg.judge.starts
        ],
        mounts={
            "agent": {str(d): _mount(d, s, snapshot=snapshot) for d, s in agent.items() if s.exists()},
            "jobs": {str(d): _mount(d, s, snapshot=snapshot) for d, s in jobs.items() if s.exists()},
        },
        readable=[str(p) for p in readable],
        env=dict(toolset.job_env(run)),
    )
    if snapshot:
        for root in (workdir, cfg.run.overlay):
            if root is None or not Path(root).exists():
                continue
            for p in sorted(Path(root).rglob("*")):
                if p.is_file() and not p.is_symlink():
                    manifest.files[str(p)] = file_sha256(p)
        for rel, path in (texts or {}).items():
            manifest.texts[str(Path(rel))] = file_sha256(Path(path))
            manifest.files[str(Path(path))] = manifest.texts[str(Path(rel))]
    return manifest
