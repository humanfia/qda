"""A model's content fingerprint and cheap signature: what its files are, independent of where it lives.

Weight-file digests are memoized under ``$QDA_CACHE/fingerprint/``, so a 16 GB model is read once.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
import threading
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ..io import qda_cache
from ..io import sha256_file as _sha256
from ..step import Refused

if TYPE_CHECKING:
    import torch  # imported where it is used: calibration sources import this module, and they stay torch-free

#: Weight files, hashed whole.
WEIGHT_GLOBS = ("*.safetensors", "*.bin")


def memo_root() -> Path:
    """``$QDA_CACHE/fingerprint``; without QDA_CACHE ``~/.cache/qda/fingerprint``."""
    return qda_cache("fingerprint")


def file_digest(path: str | Path) -> str:
    """sha256 of a file's full bytes, memoized by (realpath, size, mtime_ns, inode)."""
    real = Path(os.path.realpath(path))
    st = real.stat()
    key = {"realpath": str(real), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
    memo = memo_root() / f"{hashlib.sha256(str(real).encode()).hexdigest()}.json"
    if memo.is_file():
        try:
            got = json.loads(memo.read_text())
            if got.get("key") == key:
                return got["sha256"]
        except (OSError, ValueError):
            pass  # an unreadable memo is recomputed
    digest = _sha256(real)
    try:
        memo.parent.mkdir(parents=True, exist_ok=True)
        tmp = memo.with_name(f".{memo.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        tmp.write_text(json.dumps({"key": key, "sha256": digest}))
        tmp.replace(memo)
    except OSError:
        pass  # no writable cache: correct, only slower next time
    return digest


def _excluded(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts) or (
        rel.name.startswith("README") and rel.suffix.lower() == ".md"
    )


def _hub_blobs(root: Path) -> Path | None:
    """The ``blobs`` directory of the hub cache repo when ``root`` is one of its ``snapshots/<rev>`` (else None)."""
    blobs = root.parent.parent / "blobs"
    return blobs if root.parent.name == "snapshots" and blobs.is_dir() else None


def link_inside(model: str | Path, p: Path) -> bool:
    """Whether link ``p`` in directory ``model`` leads to a file inside it, or to a hub snapshot's blob.

    The link must pass through this repo's own ``blobs``: a direct store link could reach another repo.
    """
    root = Path(os.path.realpath(model))
    blobs = _hub_blobs(root)
    target = Path(os.path.realpath(p))
    inside = target.is_relative_to(root)
    if blobs is not None and not inside:
        first = Path(os.path.normpath(Path(os.path.realpath(p.parent)) / p.readlink()))
        store = blobs.parent.parent / "blobs"
        inside = first.parent == blobs and (target.parent == blobs or target.parent.parent == store)
    return target.is_file() and inside


def model_files(model: str | Path) -> list[Path]:
    """The files the fingerprint covers, as paths relative to ``model``, sorted.

    Raises:
        Refused: a symlink that is broken or escapes the directory (other than a hub snapshot's blob link).
    """
    d = Path(model)
    out = []
    for p in d.rglob("*"):
        rel = p.relative_to(d)
        if _excluded(rel) or p.is_dir():
            continue
        if p.is_symlink() and not link_inside(d, p):
            raise Refused(
                f"{p}: symlink to {os.path.realpath(p)} is broken or outside {d}; "
                "remove it or point it inside the directory"
            )
        if p.is_file():
            out.append(rel)
    return sorted(out)


def model_fingerprint(model: str | Path) -> str:
    """sha256 over ``(relative path, content sha256)`` of every file in a local model directory.

    Dotfiles, dot-directories and ``README*.md`` are skipped; a loader may read any other file.

    Raises:
        Refused: ``model`` is not a local directory (a hub id), the directory has no config.json, or a symlink
            is broken or escapes the directory (other than a hub snapshot's blob link).
    """
    d = Path(model)
    if not d.is_dir():
        raise Refused(f"{model}: not a local directory; pass the model's local snapshot directory")
    if not (d / "config.json").is_file():
        raise Refused(f"{d}: not a model directory (no config.json); pass the directory that holds the model")
    return _digest(d, model_files(d), _full)


def model_signature(model: str | Path) -> str:
    """Like `model_fingerprint`, but a weight file counts by its size and safetensors header, not its bytes.

    It misses weight values edited in place; symlinks are followed wherever they lead, as a loader would.

    Raises:
        Refused: ``model`` is not a local directory (a hub id is pinned by its revision).
    """
    d = Path(model)
    if not d.is_dir():
        raise Refused(f"{model}: not a local directory; pass the model's local snapshot directory")
    return _digest(d, _signature_files(d), _header)


def _digest(d: Path, files: Iterable[Path], weight: Callable[[Path, Path], str]) -> str:
    """sha256 over ``(relative path, digest)`` of every `files` entry of `d`; ``weight(path, real)`` on a weight."""
    h = hashlib.sha256()
    for rel in files:
        p, real = d / rel, Path(os.path.realpath(d / rel))
        digest = weight(p, real) if any(p.match(g) for g in WEIGHT_GLOBS) else _sha256(real)
        h.update(rel.as_posix().encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def _signature_files(d: Path) -> list[Path]:
    """Every file under `d`, exclusions as `model_files`, sorted; links are followed and a broken one skipped."""
    rels = [p.relative_to(d) for p in d.rglob("*") if p.is_file()]
    return sorted(rel for rel in rels if not _excluded(rel))


def _full(p: Path, real: Path) -> str:  # noqa: ARG001  (the signature of a weight digest; see `_header`)
    """A weight file's whole content."""
    return file_digest(real)


def _header(p: Path, real: Path) -> str:
    """A weight file's header where the format has one; a file of another format is read whole."""
    return _header_digest(real) if p.suffix == ".safetensors" else file_digest(real)


#: safetensors' own limit on a header: a corrupt length field cannot make the signature read a whole shard.
_MAX_HEADER = 100 << 20


def _header_digest(real: Path) -> str:
    """sha256 over a safetensors file's size and header: every tensor's name, dtype, shape and byte offsets."""
    h = hashlib.sha256()
    with real.open("rb") as f:
        h.update(struct.pack("<Q", real.stat().st_size))
        head = f.read(8)
        if len(head) < 8:  # not a safetensors file at all (a few stray bytes): they are all there is
            h.update(head)
            return h.hexdigest()
        (n,) = struct.unpack("<Q", head)
        h.update(f.read(min(n, _MAX_HEADER)))
    return h.hexdigest()


#: Bytes per digest chunk, and the most threads one digest uses. Fixed chunking keeps a tensor's digest the same
#: whatever the thread count or machine; crc32 stops getting faster past a handful of threads.
_DIGEST_CHUNK = 8 << 20
_DIGEST_WORKERS = 8
#: The one reusable pinned host buffer a card's bytes are copied into (grown to the largest tensor), and the lock
#: that keeps a second digest from overwriting it mid-hash.
_PINNED: list = [None]
_PINNED_LOCK = threading.Lock()


def tensor_digest(t: torch.Tensor) -> int:
    """A fast digest of a tensor's raw bytes: crc32 of 8 MB chunks, combined in order.

    A change detector, not a unique fingerprint; a card's tensor is hashed from one bulk pinned-host copy.
    """
    import torch

    x = t.detach().reshape(-1)
    b = (x if x.is_contiguous() else x.contiguous()).view(torch.uint8)
    if b.device.type == "cpu":
        return _digest_host(b)
    with _PINNED_LOCK:  # the buffer is process-wide: no second digest until this one has read it
        return _digest_host(_host_bytes(b))


def release_digest_buffer() -> None:
    """Drop the pinned host buffer `tensor_digest` reuses; a build's `finally` calls this.

    PyTorch's pinned-allocator cache may keep the freed block, so the process's pinned memory need not fall.
    """
    with _PINNED_LOCK:
        _PINNED[0] = None


def _host_bytes(b: torch.Tensor) -> torch.Tensor:
    """`b`'s bytes in the process's reused pinned host buffer; the caller holds `_PINNED_LOCK`.

    The buffer grows to the model's largest single weight and stays until `release_digest_buffer`.
    """
    import torch

    if b.device.type != "cuda":  # another accelerator: no pinned path, a plain copy
        return b.cpu()
    n = b.numel()
    buf = _PINNED[0]
    if buf is None or buf.numel() < n:
        # drop the old one first: its block may stay in the allocator cache while the larger one is allocated
        _PINNED[0] = buf = None
        buf = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        _PINNED[0] = buf
    view = buf[:n]
    view.copy_(b, non_blocking=True)
    torch.cuda.synchronize(b.device)
    return view


def _digest_host(b: torch.Tensor) -> int:
    """crc32 of `b`'s 8 MB chunks (a host-tensor uint8 view), combined in order."""
    starts = range(0, b.numel(), _DIGEST_CHUNK)
    if len(starts) > 1:
        with ThreadPoolExecutor(min(_DIGEST_WORKERS, len(starts))) as pool:
            parts = list(pool.map(lambda s: _digest_part(b, s), starts))
    else:
        parts = [_digest_part(b, s) for s in starts]
    h = zlib.crc32(b.numel().to_bytes(8, "little"))
    for d in parts:
        h = zlib.crc32(d.to_bytes(4, "little"), h)
    return h


def _digest_part(b: torch.Tensor, start: int) -> int:
    """crc32 of the ``_DIGEST_CHUNK``-byte chunk at `start` of the uint8 view `b`."""
    c = b[start : start + _DIGEST_CHUNK]
    return zlib.crc32(memoryview(c.numpy() if c.device.type == "cpu" else c.cpu().numpy()))
