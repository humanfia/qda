"""Frozen copies: what the judge scores and checks is a read-only copy taken at request time."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path

#: The shard index and the files the engine takes from the anchor (tokenizer, templates, processor,
#: generation config), with the template directory that must hold exactly the anchor's files.
INDEX = "model.safetensors.index.json"
SAME_AS_ANCHOR = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "chat_template.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
SAME_AS_ANCHOR_DIRS = ("additional_chat_templates",)


def walk(d: Path) -> list[Path]:
    """Every path under `d` (dirs, files, links), sorted; symlinks are listed, never followed."""
    out = []
    for root, dirs, files in os.walk(d, followlinks=False):
        out.extend(Path(root, x) for x in (*dirs, *files))
    return sorted(out)


def file_sha256(p: Path) -> str:
    """sha256 of one file's bytes."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_size(src: Path) -> int:
    """Total bytes of the regular files under `src` (symlinks are not followed)."""
    return sum(p.lstat().st_size for p in walk(src) if p.is_file() and not p.is_symlink())


def trusted_model_files(d: Path) -> Iterator[Path]:
    """The files the judge reads of a model directory, as paths under it.

    Only the engine's: the top-level weights and shard index, `config.json`, the files the anchor rule names
    (`SAME_AS_ANCHOR`, `SAME_AS_ANCHOR_DIRS`). Anything else a public release carries (a hidden file such as
    `.gitattributes`, `README.md`, `LICENSE`, an upload's metadata) is never loaded, so it is neither read
    nor checked: a snapshot holding them is admitted as it is.
    """
    yield from d.glob("*.safetensors")
    for n in (INDEX, "config.json", *SAME_AS_ANCHOR, *SAME_AS_ANCHOR_DIRS):
        p = d / n
        if p.is_symlink() or p.exists():
            yield p
    for n in SAME_AS_ANCHOR_DIRS:
        root = d / n
        if root.is_dir() and not root.is_symlink():
            yield from walk(root)


def model_size(model: Path) -> int:
    """Total bytes of the files the judge reads of a model directory, its measure of `judge.anchor`.

    Unlike `tree_size` (for frozen copies, where a link is never followed), a link counts as its target's
    size: a Hugging Face cache snapshot keeps its served files as links into the repository's `blobs/`
    (`snapshots/<rev>/x -> ../../blobs/<hash>`). What is not read is not counted (`trusted_model_files`),
    so a release's other files, hidden ones included, neither count nor can refuse the anchor.

    Raises:
        ValueError: `model` is not a directory, has no served file, or one of them cannot be sized (a broken
            link).
    """
    if not model.is_dir():
        raise ValueError(f"{model} is not a directory")
    try:
        size = sum(p.stat().st_size for p in trusted_model_files(model))
    except OSError as e:
        raise ValueError(f"{model}: {e}") from None
    if size == 0:
        raise ValueError(f"{model} has no files")
    return size


def digest(d: Path, pattern: str = "*") -> str:
    """sha256 over (relative path, file sha256) of the files under `d` matching `pattern`, sorted."""
    h = hashlib.sha256()
    for p in (x for x in walk(d) if x.match(pattern) and x.is_file() and not x.is_symlink()):
        h.update(p.relative_to(d).as_posix().encode() + b"\0")
        h.update(file_sha256(p).encode())
    return h.hexdigest()


def link_in(tree: Path) -> Path | None:
    """The first symbolic link under `tree`, in walk order (None: no link)."""
    return next((p for p in walk(tree) if p.is_symlink()), None)


def hardlink_in(tree: Path) -> Path | None:
    """The first regular file at or under `tree` with another hard link (`st_nlink != 1`), or None.

    A second link is a second name for the same bytes: whoever can write through that name can change them.
    A symbolic link is never followed (one at `tree` itself is not walked), and a path that cannot be read is
    skipped: the copy that follows reports it.
    """
    for p in [tree, *([] if tree.is_symlink() else walk(tree))]:
        try:
            info = p.lstat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            return p
    return None


def freeze(src: Path, dst: Path) -> str:
    """Copy `src` to `dst` (symlinks copied as links, never followed), make it read-only, return its digest.

    Later edits of `src` change nothing in `dst`: the files are copies, not links. A link that reaches the
    copy is refused before it is sealed: the digest skips links, so keeping one would leave bytes scoring
    can read that the record never hashed.

    Raises:
        ValueError: The copied tree holds a symbolic link.
    """
    shutil.copytree(src, dst, symlinks=True)
    if link := link_in(dst):  # the copy, not the caller's view of the source: a link added since is refused
        raise ValueError(
            f"{link} is a symbolic link in the frozen copy; a frozen copy must hold the bytes it hashes"
        )
    return seal(dst)


def seal(dst: Path) -> str:
    """Make the tree at `dst` read-only and return its digest."""
    for root, dirs, files in os.walk(dst):
        for f in files:
            p = Path(root, f)
            if not p.is_symlink():
                p.chmod(0o444)
        for d in dirs:
            Path(root, d).chmod(0o555)
    dst.chmod(0o555)
    return digest(dst)


def deletable(top: Path) -> None:
    """Make every directory under `top` owner-writable again, so that `rm -rf top` works.

    Removing a file needs write access to its directory, not to the file: sealed files keep their read-only
    mode, so a frozen copy still cannot be edited by accident. Symlinks are never followed.
    """
    for root, dirs, _ in os.walk(top):
        for d in dirs:
            p = Path(root, d)
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | 0o700)
    if top.is_dir() and not top.is_symlink():
        top.chmod(top.stat().st_mode | 0o700)
