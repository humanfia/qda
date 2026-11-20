"""``qda export``: a finished checkpoint in release shape, copied, hashed and read back.

A build's output directory holds the checkpoint and the run's records. The records stay with the build; the
release is the rest of the top level:

- left out: the run records (``build.json``, ``resolved.yaml``, ``recipe.yaml``, ``rotate-draft.json`` and ``splice.json``
  from ``qda rotate-draft``) and every subdirectory (``qda_state/`` holds resume state);
- copied: every other top-level file, byte for byte (weights, index, config, tokenizer, chat template).

The records carry the build's local paths and timings, and the recipe that made the checkpoint is published
on its own, so a release that leaves them out loses nothing a user needs to load or rebuild it. Every copied
file is hashed at the source, copied, then read back and hashed again; any difference refuses. The sums are
printed in ``sha256sum`` format, so a pipeline's log keeps them, and no manifest is written into the release.
A text file that names the source directory's absolute path is refused: a release must not name a local path.

    qda export builds/qwen3.5-9b-w4a16/kd-r3 release/qwen3.5-9b-w4a16
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from qda_quant.exit import refuse
from qda_quant.output import rotate_draft
from qda_quant.output.record import BUILD_RECORDS
from qda_quant.output.roots import output_path
from qda_quant.step import Refused

from ..io import sha256_file as _sha

#: run records the tools write beside a checkpoint; they describe a build, not the model
RECORDS = frozenset({*BUILD_RECORDS, rotate_draft.RECORD, "splice.json"})  # splice.json: older builds' base/donor paths
#: files whose bytes are not text, so they are not searched for local paths
_BINARY = (".safetensors", ".bin", ".model", ".gguf")


def release_files(src: Path) -> list[Path]:
    """The top-level files of ``src`` that a release ships, sorted by name."""
    return sorted(p for p in src.iterdir() if p.is_file() and p.name not in RECORDS)


def export(src: Path, out: Path, link: bool = False) -> dict[str, str]:
    """Copy ``src``'s release files to a new ``out``; returns ``{name: sha256}`` after the read-back.

    ``link`` hard-links instead of copying (same filesystem, no second copy of the weights); the read-back then
    checks the link, which is the same inode. Refuses an existing ``out``, a source without ``config.json``,
    a text file naming ``src``'s absolute path, and any byte difference between source and copy.
    """
    if out.exists():
        raise Refused(f"{out} already exists; this command never writes into an existing directory")
    files = release_files(src)
    if not any(p.name == "config.json" for p in files):
        raise Refused(f"{src} has no config.json; not a checkpoint")
    local = str(src.resolve())
    if leaks := [
        p.name for p in files if not p.name.endswith(_BINARY) and local in p.read_text(errors="replace")
    ]:
        raise Refused(f"{', '.join(leaks)} contain the local path {local}")
    sums = {p.name: _sha(p) for p in files}
    out.mkdir(parents=True)
    for p in files:
        (os.link if link else shutil.copyfile)(p, out / p.name)
    bad = [name for name, h in sums.items() if _sha(out / name) != h]
    if bad:
        raise Refused(f"read-back differs from the source: {', '.join(bad)} (left in {out})")
    return sums


def _run(a: argparse.Namespace) -> int:
    src = Path(a.checkpoint)
    try:
        out = output_path(a.out)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    left = sorted(
        p.name + ("/" if p.is_dir() else "") for p in src.iterdir() if p.is_dir() or p.name in RECORDS
    )
    try:
        sums = export(src, out, a.link)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    for name, h in sums.items():
        print(f"{h}  {name}")
    print(
        f"qda: {len(sums)} files to {out}, read back OK; left out: {', '.join(left) or 'nothing'}",
        file=sys.stderr,
    )
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``qda export``."""
    p = sub.add_parser("export", help="copy a checkpoint without its run records, hashed and read back")
    p.add_argument("checkpoint", help="a build's output directory")
    p.add_argument("out", help="the release directory to create (must not exist)")
    p.add_argument("--link", action="store_true", help="hard-link the files instead of copying them")
    p.set_defaults(run=_run)
