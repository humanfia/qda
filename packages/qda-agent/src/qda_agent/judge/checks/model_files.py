"""The files a checkpoint may hold, the anchor's own files, and the trusted directories the judge reads.

Moved here from `checks.py` (pure move): `files_problem` and its file policy (`CODE`, `INERT`), the anchor
comparison (`anchor_problem`) and `trusted_files_problem`. `checks` re-exports the public names.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qda_kl.tokenizer_check import tokenizer_config_difference

from ..freeze import INDEX, SAME_AS_ANCHOR, SAME_AS_ANCHOR_DIRS, trusted_model_files, walk
from ..freeze import file_sha256 as _sha
from .formats import DEFAULT_MENU, HEAD_16BIT, _quantization_config_problem, format_problem
from .tensors import CheckpointError, _json, loaded

CODE = re.compile(r"\.(py|pyc|so|sh|pkl|pickle|pt|pth|bin|ckpt|npy|npz)$|(^|/)__pycache__(/|$)")
WEIGHTS = re.compile(r"[^/]+\.safetensors")
#: Files a checkpoint may carry that nothing loads (globs).
INERT = (  # Inert files allowed by the checkpoint policy, plus what qda's own builds write
    "README.md",
    "LICENSE*",
    "recipe.yaml",
    "resolved.yaml",
    "build.json",
    # scale_kd's trained deltas: vLLM loads only top-level safetensors. This exact path, nothing else below it.
    "qda_state/scales.safetensors",
    "quant-metrics.json",
    "quant-record.json",
    "splice-head.json",
    "scaletune.json",
    "moefast.json",
    "gsq.json",
    "sym4-record.json",
    "splice-assign.json",
    "conversion.json",
    "assets-manifest.json",
)
#: The one `SAME_AS_ANCHOR` file compared as content: `qda build` corrects its tokenizer class (and records it).
TOKENIZER_CONFIG = "tokenizer_config.json"
#: `config.json` keys that may differ from the anchor's.
#: `config.json` keys free at any depth: the quantization, the save's dtype and bookkeeping. The
#: architecture itself (dims, layers, vocab, rope, `architectures`, `model_type`) must be the anchor's.
CONFIG_FREE = (
    "quantization_config",
    "transformers_version",
    "_name_or_path",
    "name_or_path",
    "dtype",
    "torch_dtype",
)


def trusted_files_problem(d: Path) -> str | None:
    """Why a trusted model directory cannot be read through its links, or None.

    The judge's own inputs (`judge.anchor`, `baselines`, `judge.format_reference`) may be kept as a
    Hugging Face cache snapshot: a directory whose files are links into its repository's `blobs/`, which
    are in turn links into the cache's shared store. A served file's link is admitted when it resolves to a
    regular file inside that same repository, through the repository's own `blobs/` entry into the store --
    never straight into the store or into another repository -- and a directory that is not a snapshot may
    hold only links that stay inside it. What the judge then reads and hashes (`_sha`,
    `tensor_fingerprint`, `loaded`) is a file the directory's owner put there, not one a planted link chose.
    Only the files the judge reads are checked (`trusted_model_files`): a release's other files, hidden
    ones included, are ignored. Agent hand-ins are unaffected: `files_problem` keeps them regular files
    only, hidden files refused.

    Args:
        d: The model directory.

    Returns:
        The first offending link's reason, naming the file inside `d` and where it leads, or None.
    """
    from qda_quant.intake.fingerprint import link_inside  # the intake's one reading of a hub snapshot's link

    for p in trusted_model_files(d):
        if p.is_symlink() and not link_inside(d, p):
            return (
                f"{p.relative_to(d).as_posix()} is a link to {os.path.realpath(p)}, not a regular file "
                "inside the model directory (a hub snapshot's own blobs/ store is the one exception)"
            )
    return None


def anchor_problem(d: Path, anchor: Path) -> str | None:
    """Why `d` is not the anchor's architecture config, tokenizer, templates and auxiliary files, or None.

    The architecture config may differ only in `CONFIG_FREE` (the quantization config and the save's
    dtype and bookkeeping, at any depth); the first other field that differs is named. Every
    `SAME_AS_ANCHOR` file (and `SAME_AS_ANCHOR_DIRS` directory) must be the anchor's, byte for byte, except
    `tokenizer_config.json`: it may differ by exactly the tokenizer-class correction `qda build` makes and
    records (`qda_kl.tokenizer_check.tokenizer_config_difference`; `tokenizer.json` itself stays byte for
    byte), so a build of the anchor, or the anchor's original beside a corrected anchor, passes. This is the
    gate candidate 0 and every other baseline get at the host's start, as a candidate gets it at hand-in.
    """
    if why := _anchor_files_problem(d, anchor):
        return why
    try:
        mine, theirs = _json(d / "config.json"), _json(anchor / "config.json")
    except (OSError, ValueError):
        return "no readable config.json"
    if not isinstance(mine, dict) or not isinstance(theirs, dict):
        return "config.json is not an object"
    if "auto_map" in mine:
        return "config.json has auto_map (remote code)"
    # an anchor may be a release that ships remote code for a type transformers and vLLM implement natively
    # (NemotronH); a candidate never may (above), so the anchor's pointer is not part of the architecture
    theirs = {k: v for k, v in theirs.items() if k != "auto_map"}
    if (why := _config_difference(mine, theirs)) is not None:
        return f"config.json differs from the anchor's: {why}"
    return None


def _config_difference(mine: Any, theirs: Any, path: str = "") -> str | None:
    """The first field two architecture configs differ in, free keys dropped at any depth, or None."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for k in sorted(set(mine) | set(theirs)):
            if k in CONFIG_FREE:
                continue
            if k not in mine or k not in theirs:
                return f"{path}{k} is only on one side"
            if (why := _config_difference(mine[k], theirs[k], f"{path}{k}.")) is not None:
                return why
        return None
    if isinstance(mine, list) and isinstance(theirs, list):
        if len(mine) != len(theirs):
            return path.rstrip(".")
        for i, (a, b) in enumerate(zip(mine, theirs, strict=True)):
            if (why := _config_difference(a, b, f"{path}{i}.")) is not None:
                return why
        return None
    return None if mine == theirs else path.rstrip(".")


def _anchor_files_problem(d: Path, anchor: Path) -> str | None:
    for n in SAME_AS_ANCHOR:  # tokenizer.json comes first: the config's correction is read against it
        a, b = anchor / n, d / n
        if a.exists() != b.exists() or (a.exists() and _sha(a) != _sha(b)):
            if n != TOKENIZER_CONFIG:
                return f"{n} differs from the anchor's"
            if (why := tokenizer_config_difference(d, anchor)) is not None:
                return f"{n} differs from the anchor's: {why}"
    for n in SAME_AS_ANCHOR_DIRS:

        def files(root: Path, n: str = n) -> set[str]:
            return {p.relative_to(root).as_posix() for p in (root / n).rglob("*") if not p.is_dir()}

        mine, theirs = files(d), files(anchor)
        if (d / n).exists() != (anchor / n).exists() or mine != theirs:
            miss, extra = sorted(theirs - mine), sorted(mine - theirs)
            return f"{n}/ differs from the anchor's (missing {miss[:2]}, extra {extra[:2]})"
        for rel in sorted(mine):
            if _sha(anchor / rel) != _sha(d / rel):
                return f"{rel} differs from the anchor's"
    return None


def inert_match(rel: str, glob: str) -> bool:
    """Whether the checkpoint path `rel` matches an `INERT` glob at the same depth.

    `fnmatch`'s `*` also matches `/`, so `LICENSE*` alone would accept `LICENSE/model.safetensors`; a glob
    matches only paths with as many `/` as it has (top-level globs, top-level files).
    """
    return rel.count("/") == glob.count("/") and fnmatch.fnmatchcase(rel, glob)


def files_problem(
    d: Path,
    anchor: Path | None = None,
    format_ref: Path | None = None,  # noqa: ARG001 -- kept for callers of the earlier API; not read
    inert: tuple[str, ...] | list[str] = INERT,
    *,
    format_menu: Sequence[str] = DEFAULT_MENU,
    any_format: bool = False,
    head_format: str | None = None,
    format_map: Path | None = None,
    head: str = HEAD_16BIT,
    lease: Sequence[str] = (),
    floor: bool = True,
) -> str | None:
    """Why `d` is not a servable checkpoint of the task's format, or None.

    Args:
        d: The checkpoint directory.
        anchor: The original model (tokenizer files, templates, architecture config and head dtype to match).
        format_ref: Kept for callers of the earlier API; the format gate is the menu's and the engine's,
            not a reference's (`format_problem` runs whether or not a reference is given).
        inert: Globs of files nothing loads, accepted as they are.
        format_menu: The task's formats (`judge.format_menu`).
        any_format: Skip the format gate (`judge.allow_any_format`, development only).
        head_format: `format_problem`'s: for publishing an already-built paper checkpoint with a quantized
            lm_head; searches leave it None (lm_head unquantized unless `format_map` stores it quantized).
        format_map: `format_problem`'s: the checkpoint whose per-module formats are the task's fixed
            allocation (`judge.format_map`), or None.
        head: `format_problem`'s: `judge.head`, the output head's rule (`16-bit` by default).
        lease: `format_problem`'s: the head formats `judge.head_lease` offers; empty when it is off.

    Returns:
        The reason, in words naming paths relative to `d` only, or None.
    """
    if not d.is_dir():
        return "not a directory"
    for p in walk(d):
        rel = p.relative_to(d).as_posix()
        if any(part.startswith(".") for part in p.relative_to(d).parts):
            return f"{rel}: hidden files and directories are not served"
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            return f"{rel} is not a regular file"
        if CODE.search(rel):
            return f"{rel}: code or pickle files are not served"
        if p.is_dir() or WEIGHTS.fullmatch(rel) or rel in ("config.json", INDEX, *SAME_AS_ANCHOR):
            continue
        if any(inert_match(rel, g) for g in inert):
            continue
        a = anchor / rel if anchor is not None else None
        if not (a is not None and a.is_file() and not a.is_symlink() and _sha(a) == _sha(p)):
            return f"{rel}: not served (weights are safetensors; any other file must be the anchor's own, byte for byte)"
    if not any(d.glob("*.safetensors")):
        return "no *.safetensors weights"
    try:
        cfg = _json(d / "config.json")
    except (OSError, ValueError):
        return "no readable config.json"
    if not isinstance(cfg, dict):
        return "config.json is not an object"
    if why := _quantization_config_problem(cfg):
        return f"config.json: {why}"
    try:
        loaded(d)
    except CheckpointError as e:
        return f"not a readable checkpoint ({e})"
    if "auto_map" in cfg:
        return "config.json has auto_map (remote code)"
    if anchor is not None and (why := anchor_problem(d, anchor)):
        return why
    if any_format:  # development stand-ins: any servable checkpoint passes
        return None
    try:  # the menu gate is the task's and the engine's: a format reference is not required for it
        return format_problem(
            d,
            format_menu,
            cfg,
            anchor=anchor,
            head_format=head_format,
            format_map=format_map,
            head=head,
            lease=lease,
            floor=floor,
        )
    except CheckpointError as e:
        return f"not a readable checkpoint ({e})"
