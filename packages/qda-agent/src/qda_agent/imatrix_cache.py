"""llama.cpp's importance matrix, measured once per machine per model, calibration file and llama.cpp commit.

Callers get a hard link, never a symlink, because a build's sandbox binds only the path it is given.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qda_quant.intake.fingerprint import model_fingerprint
from qda_quant.io import qda_cache, sha256_file
from qda_quant.step import Refused

__all__ = [
    "FILE",
    "cache_root",
    "cached",
    "command",
    "entry",
    "key_payload",
    "lookup",
    "place",
    "record_path",
    "text_file",
]

#: The file's name in an entry directory.
FILE = "imatrix.gguf"
#: The measurement text's name beside the calibration file a prepare renders.
TEXT = "calib.imatrix.txt"


def cache_root() -> Path:
    """``$QDA_CACHE/imatrix``; without QDA_CACHE ``~/.cache/qda/imatrix`` (the calibration cache's convention)."""
    return qda_cache("imatrix")


def key_payload(
    model: str | Path,
    calib: str | Path,
    commit: str,
    *,
    context: int,
    chunks: int,
    process_output: bool,
) -> dict[str, Any]:
    """Everything a measurement's bytes depend on: the model, calibration file, llama.cpp and settings.

    Raises:
        Refused: ``model`` is not a local model directory with a ``config.json`` (download a Hub id first).
    """
    return {
        "model": model_fingerprint(model),
        "calib_sha256": sha256_file(Path(calib)),
        "llama_cpp": commit,
        "settings": {"context": int(context), "chunks": int(chunks), "process_output": bool(process_output)},
    }


def entry(payload: dict[str, Any]) -> Path:
    """The cache directory of the measurement ``payload`` describes."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_root() / payload["model"][:16] / digest[:24]


def record_path(where: Path) -> Path:
    """The record beside an entry directory: ``<key>.json``."""
    return where.with_name(where.name + ".json")


def lookup(payload: dict[str, Any]) -> Path | None:
    """The cached file of ``payload``, or None until its record is written and matches the file's sha256."""
    where = entry(payload)
    found, record = where / FILE, record_path(where)
    if not found.is_file() or not record.is_file():
        return None
    try:
        got = json.loads(record.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or got.get("payload") != payload or got.get("sha256") != sha256_file(found):
        return None
    return found


def text_file(calib: str | Path, out: str | Path) -> Path:
    """Write `llama-imatrix`'s plain-text input: the rendered calibration rows joined by blank lines.

    Raises:
        Refused: a row carries no ``text``, or the file holds none.
    """
    rows: list[str] = []
    with Path(calib).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise Refused(f"{calib}:{number}: not JSON ({e})") from None
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise Refused(
                    f"{calib}:{number}: expected a rendered calibration row (a 'text' string); run the calib "
                    "step first"
                )
            rows.append(text)
    if not rows:
        raise Refused(f"{calib}: no calibration rows; run the calib step first")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    tmp.write_text("\n\n".join(rows) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def command(
    program: str | Path,
    template: str | Path,
    text: str | Path,
    out: str | Path,
    *,
    context: int,
    chunks: int,
    process_output: bool = True,
) -> list[str]:
    """The `llama-imatrix` command that measures `template` over `text` into `out`.

    ``--process-output`` also measures the output head, which a GGUF build would otherwise fit unweighted.
    """
    cmd = [
        str(program),
        "-m",
        str(template),
        "-f",
        str(text),
        "-o",
        str(out),
        "-c",
        str(int(context)),
        "--chunks",
        str(int(chunks)),
    ]
    if process_output:
        cmd.append("--process-output")
    return cmd


@contextmanager
def _waiting_lock(path: Path, log: Any) -> Iterator[None]:
    """Hold an exclusive lock on ``path``, waiting for another measurement of the same key to finish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"importance matrix: another process is measuring this key; waiting for it ({path})")
            started = time.monotonic()
            fcntl.flock(handle, fcntl.LOCK_EX)
            log(f"importance matrix: waited {time.monotonic() - started:.0f} s for the other measurement")
        yield


def cached(
    *,
    program: str | Path,
    template: str | Path,
    text: str | Path,
    model: str | Path,
    calib: str | Path,
    commit: str,
    context: int,
    chunks: int,
    process_output: bool = True,
    env: dict[str, str] | None = None,
    log: Any = print,
) -> tuple[Path, bool]:
    """The cached measurement of these inputs, measured first if missing (one measurement per key).

    Returns:
        The cache's file and whether this call measured it.

    Raises:
        Refused: the record cannot be written, or `llama-imatrix` wrote no file.
        subprocess.CalledProcessError: `llama-imatrix` failed (its output is on the log).
    """
    payload = key_payload(model, calib, commit, context=context, chunks=chunks, process_output=process_output)
    target = entry(payload)
    hit = lookup(payload)
    if hit is not None:
        log(f"importance matrix: reusing {hit}")
        return hit, False
    with _waiting_lock(target.with_name(f".{target.name}.cache-lock"), log):
        hit = lookup(payload)
        if hit is not None:  # another process may have measured it while this one waited
            log(f"importance matrix: reusing {hit}")
            return hit, False
        target.mkdir(parents=True, exist_ok=True)
        partial = target / f".{FILE}.partial"
        partial.unlink(missing_ok=True)
        log(
            f"importance matrix: measuring {target} "
            f"(context {context}, {chunks} chunks; later calls on this model and calibration reuse it)"
        )
        subprocess.run(
            command(
                program,
                template,
                text,
                partial,
                context=context,
                chunks=chunks,
                process_output=process_output,
            ),
            check=True,
            env=env,
        )
        if not partial.is_file():
            partial.unlink(missing_ok=True)
            raise Refused(f"llama-imatrix exited 0 and wrote no {partial}")
        partial.replace(target / FILE)
        record = {
            "payload": payload,
            "file": FILE,
            "sha256": sha256_file(target / FILE),
            "bytes": (target / FILE).stat().st_size,
            "template": str(template),
            "text": str(text),
            "command": command(
                program,
                template,
                text,
                target / FILE,
                context=context,
                chunks=chunks,
                process_output=process_output,
            ),
        }
        tmp = target.with_name(f".{target.name}.json.partial")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        tmp.replace(record_path(target))
        return target / FILE, True


def place(cached_file: str | Path, out: str | Path) -> Path:
    """Hard-link the cached file at ``out`` (a copy across file systems), replacing ``out``."""
    from qda_agent.cand0 import _place  # the bf16 template's placement, the one rule for a cached file

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _place(Path(cached_file), out)
    return out
