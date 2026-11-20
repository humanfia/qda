"""Tamper detection for the paths a search's numbers depend on (P2 at `isolation.level: none`).

At `sandbox` and `strict` the wall keeps the agent out of the judge's directory, its reference and the
scorer. `none` runs plain subprocesses, so the agent can write whatever the user can: P2 is then held by
detection. A run takes a digest of the trusted paths when it starts and compares it before every job (so
before every reading) and at the end:

* the judge's state: `judge/judge.json` (the evaluator and tool digests the judge recorded), the installed
  evaluator tree `judge/eval` (the scripts every reading runs) and the judge's copy of the prompt set;
* the reference: `judge.kl_ref`, its holdout prompt set, the long-generation prompt set, the builds'
  contamination index, the anchor, the format reference and the baselines;
* the scorer: `judge/eval`, the frozen copy of the evaluator and of the judge's tools, or the llama.cpp
  bridge and server binaries a `.gguf` search scores with.

A directory is digested entry by entry, and a link inside it by its target string. A link to a file is
digested by the file behind it (a hub snapshot is all such links); a link to a directory that stays inside
the tree is walked, so content it reaches counts as the tree's own; a link to a directory outside the tree
is taken by its target string alone, never walked.

A mismatch refuses the reading and is recorded in `run_dir/trust.jsonl`.

Every comparison re-reads the paths: no digest, size or stat tuple is cached between readings. A regular
file of `CONTENT_MAX_BYTES` or under is digested by its content on every reading; a larger one (a model's
weights, an array too big to re-read in a reading's shadow) by its size and modification time, because
hashing tens of gigabytes before every reading would cost as much as the reading.

This is DETECTION ONLY, not prevention, and the detection of a large file is a hint: a writable mapping can
change a file's bytes without moving its size or its modification time, and the digest never writes anything
back. What it buys is that a changed judge, reference or scorer refuses the number it would have produced;
what prevents the change is the wall at `sandbox` and `strict`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RunConfig

__all__ = ["CONTENT_MAX_BYTES", "TrustChanged", "Trusted", "path_digest", "trusted_paths"]

#: Files up to this size are digested by content; larger ones by size and mtime (see the module docstring).
CONTENT_MAX_BYTES = 32 << 20
#: What a digest says of a path that does not exist (yet).
MISSING = "missing"
#: The record file in the run directory.
RECORD = "trust.jsonl"


class TrustChanged(RuntimeError):
    """A trusted path changed while the agent could write it: the reading is refused."""

    def __init__(self, what: str, changed: Mapping[str, tuple[str, str]], paths: Mapping[str, Path]):
        self.what = what
        self.changed = dict(changed)
        self.paths = dict(paths)
        listed = "; ".join(f"{name} ({self.paths[name]})" for name in changed)
        super().__init__(
            f"the {what} is refused: a trusted file changed while the agent could write it: {listed}"
        )


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_digest(p: Path) -> str:
    """One file's identity: its content when small, its size and mtime when large; stat follows a link."""
    try:
        st = p.stat()
    except FileNotFoundError:
        return MISSING
    except OSError as e:  # a permission or I/O error: not the file that could be read before
        return f"unreadable:{e.strerror}"
    if not stat.S_ISREG(st.st_mode):
        return f"not-a-file:{stat.S_IFMT(st.st_mode):o}"
    if st.st_size > CONTENT_MAX_BYTES:
        return f"large:{st.st_size}:{st.st_mtime_ns}"
    try:
        return "sha256:" + _sha256(p)
    except OSError as e:
        return f"unreadable:{e.strerror}"


def _tree_digest(d: Path) -> str:
    """A directory's identity: every entry below it by relative path, its links by target and content.

    A link is part of the digest by its target string (`readlink`), so retargeting it or swapping it for a
    file changes the digest. A link to a file is digested by the file behind it, inside or out: a hub
    snapshot reads that way (`snapshots/<rev>/config.json -> ../../blobs/<hash>`, every small file of the
    model). A link to a directory that stays inside the trusted tree (its resolved target lies under this
    directory's resolved path) is walked, and the files behind it are digested: content the tree reaches
    that way is content a reading depends on. A link to a directory outside the tree is hashed by its
    target string alone (`link-out`): walking it would digest a tree the run does not own, and a link to
    `/` would cost a walk of the whole filesystem on every reading. A link back into a directory already
    walked is recorded and not walked again (a cycle would never end).
    """
    h = hashlib.sha256()
    root = Path(os.path.realpath(d))
    walked: set[str] = set()

    def walk(directory: Path, label: str) -> None:
        real = os.path.realpath(directory)
        if real in walked:  # a cycle, or a second link to a directory already digested
            h.update(label.encode() + b"\0cycle:" + real.encode() + b"\0")
            return
        walked.add(real)
        try:
            names = sorted(os.listdir(directory))
        except OSError as e:
            h.update(label.encode() + b"\0unreadable:" + str(e.strerror).encode() + b"\0")
            return
        for name in names:
            p = directory / name
            rel = f"{label}/{name}" if label else name
            try:
                mode = p.lstat().st_mode
            except OSError as e:
                h.update(rel.encode() + b"\0unstatable:" + str(e.strerror).encode() + b"\0")
                continue
            if stat.S_ISLNK(mode):
                try:
                    target = os.readlink(p)
                except OSError as e:
                    target = f"unreadable:{e.strerror}"
                h.update(rel.encode() + b"\0link:" + target.encode() + b"\0")
                try:
                    behind = p.resolve(strict=True)
                except OSError:
                    continue  # a dangling link: its target string is all of it
                if behind.is_dir():
                    if behind.is_relative_to(root):
                        walk(behind, rel)
                    else:
                        # a directory out of the tree: its target string, not a walk of a tree the run does
                        # not own (a link to `/` would cost the whole filesystem on every reading)
                        h.update(rel.encode() + b"\0link-out:" + target.encode() + b"\0")
                elif behind.is_file():
                    # a file out of the tree is digested like one inside it: a hub snapshot is built of
                    # exactly such links (snapshots/<rev>/config.json -> ../../blobs/<hash>), and the
                    # digest rule is the content of every trusted input, its links included
                    h.update(_file_digest(behind).encode() + b"\0")
            elif stat.S_ISDIR(mode):
                walk(p, rel)
            else:
                h.update(rel.encode() + b"\0" + _file_digest(p).encode() + b"\0")

    walk(Path(d), "")
    return "tree:" + h.hexdigest()


def path_digest(path: Path) -> str:
    """A path's digest: `MISSING`, a file's, or a directory tree's."""
    try:
        st = Path(path).stat()  # the configured path itself may be a link (a hub snapshot)
    except FileNotFoundError:
        return MISSING
    except OSError as e:
        return f"unreadable:{e.strerror}"
    if stat.S_ISDIR(st.st_mode):
        return _tree_digest(Path(path))
    return _file_digest(Path(path))


def trusted_paths(cfg: RunConfig) -> dict[str, Path]:
    """The named paths a `none` run must see unchanged: the judge's state, the reference and the scorer.

    A path that does not exist yet (the judge installs its evaluator during start) is recorded as missing
    and adopted the first time it is seen; see `Trusted.verify`.
    """
    judge = cfg.judge
    run_dir = Path(cfg.run_dir)
    out: dict[str, Path] = {
        "judge-config": run_dir / "judge" / "judge.json",
        "judge-eval": run_dir / "judge" / "eval",
        "judge-prompts": run_dir / "judge" / "gencheck-prompts.jsonl",
        "contamination-index": run_dir / "calibration-index" / "contamination-index.json",
    }
    for name, path in (
        ("reference", _local(judge, "kl_reference")),
        ("holdout", judge.kl_holdout_ref),
        ("gencheck-prompts", _local(judge, "prompts_path")),
        ("anchor", judge.anchor),
        ("format-reference", judge.format_reference),
        ("format-map", judge.format_map),
    ):
        if path is not None:
            out[name] = Path(path)
    for i, baseline in enumerate(cfg.baselines):
        out[f"baselines[{i}]"] = Path(baseline)
    llamacpp = judge.llamacpp
    if llamacpp is not None:
        out["llama-bridge"] = Path(llamacpp.bridge)
        out["llama-server"] = Path(llamacpp.server)
    return out


def _local(judge: object, method: str) -> Path | None:
    """The local path a judge dataset method names, or None (unset, or a dataset that cannot be made local)."""
    getter = getattr(judge, method, None)
    if getter is None:
        return None
    try:
        return getter()
    except ValueError:
        return None


class Trusted:
    """A run's trusted paths, the digests taken at its start, and the record of a comparison.

    Args:
        paths: name -> path.
        record: where a refusal or a late adoption is appended (`run_dir/trust.jsonl`); None records nothing.
        level: the isolation level, for the record.
    """

    def __init__(
        self,
        paths: Mapping[str, Path],
        *,
        record: Path | None = None,
        level: str = "none",
    ) -> None:
        self.paths = dict(paths)
        self.record = record
        self.level = level
        self._taken: dict[str, str] = {}
        self._lock = threading.RLock()

    def take(self, what: str = "run start") -> dict[str, str]:
        """Digest every path now and record the digests; a missing path is adopted on its first sight."""
        with self._lock:
            self._taken = {name: path_digest(path) for name, path in self.paths.items()}
            self._event("taken", what, digests=self._taken)
            return dict(self._taken)

    @property
    def digests(self) -> dict[str, str]:
        """The digests taken at the start (empty before `take`)."""
        with self._lock:
            return dict(self._taken)

    def covered(self, path: Path) -> bool:
        """Whether `path` is already watched: itself, inside a trusted path, or holding one.

        A caller that names a directory holding the judge's installed evaluator needs no entry of its own:
        the run already digests the judge's files, and digesting the whole directory would also digest the
        ledger and the results the host writes as it goes.
        """
        p = Path(path)
        for q in self.paths.values():
            if p == q or q in p.parents or p in q.parents:
                return True
        return False

    def add(self, path: Path) -> str:
        """Trust `path` from now on (a caller's own path); returns the name it is recorded under."""
        with self._lock:
            for name, existing in self.paths.items():
                if existing == Path(path):
                    return name
            name = str(path)
            self.paths[name] = Path(path)
            self._taken[name] = path_digest(Path(path))
            return name

    def verify(self, what: str) -> None:
        """Compare every trusted path with its digest; raise `TrustChanged` when one changed.

        A path that was missing when the digest was taken and exists now is adopted (the judge installs its
        evaluator after the run's session starts) and the adoption is recorded.

        Raises:
            TrustChanged: A trusted path changed or disappeared; the event is written to `record` first.
        """
        with self._lock:
            if not self._taken:
                self.take()
            changed: dict[str, tuple[str, str]] = {}
            for name, path in self.paths.items():
                now = path_digest(path)
                if name not in self._taken:
                    self._taken[name] = now
                    self._event("adopted", what, name=str(path))
                elif self._taken[name] == MISSING and now != MISSING:
                    # installed after the digest was taken (the judge's evaluator during start)
                    self._taken[name] = now
                    self._event("adopted", what, name=str(path))
                elif now != self._taken[name]:
                    changed[name] = (self._taken[name], now)
            if changed:
                self._event(
                    "refused",
                    what,
                    changed={n: {"before": b, "after": a} for n, (b, a) in changed.items()},
                )
                raise TrustChanged(what, changed, {n: self.paths[n] for n in changed})

    def _event(self, event: str, what: str, **detail: object) -> None:
        if self.record is None:
            return
        line: dict[str, object] = {
            "utc": datetime.now(UTC).isoformat(),
            "event": event,
            "level": self.level,
            "what": what,
        }
        line.update(detail)
        try:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            with self.record.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass  # the record is evidence; a failed write must not mask the refusal itself
