"""Read a `.gguf` with vLLM: decode it to bf16 (`qda decode-gguf`), read the decode, then remove it.

``qda-kl score`` and ``qda-kl gencheck`` take a `.gguf` file (or a directory holding one `.gguf` and no
`config.json`) as `checkpoint`. The decoder needs the HF snapshot the file was converted from: score's
`snapshot`, else the model the reference was built from; gencheck's `original`. The decode goes to
``<out>/decoded`` and is removed when the reading or the checkpoint's generations end.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from qda_quant.step import Refused

#: the decode's directory under the run's `out`, removed when the reading ends
DECODED = "decoded"
#: the decoder's manifest, kept beside ``score.json``: which file was read, against which snapshot
MANIFEST = "gguf-decode.json"


def gguf_file(checkpoint: str) -> Path | None:
    """The `.gguf` `checkpoint` names: the file itself, or the only one in a directory with no `config.json`.

    Raises:
        Refused: a directory with no `config.json` holds several `.gguf` files.
    """
    path = Path(checkpoint).expanduser()
    if path.suffix == ".gguf" and path.is_file():
        return path
    if not path.is_dir() or (path / "config.json").is_file():
        return None
    files = sorted(path.glob("*.gguf"))
    if len(files) > 1:
        raise Refused(f"{path}: {len(files)} .gguf files; name the one to score as `checkpoint`")
    return files[0] if files else None


def snapshot_of(given: str | None, reference: Path | None = None, key: str = "snapshot") -> Path:
    """The HF snapshot to decode against: `given` (the config's `key`), else the model the reference records (a
    Hub id: its cached snapshot).

    Raises:
        Refused: neither names a local model directory; the Hub id is not cached; or the cached snapshot's
            config.json is not the one the reference was built from.
    """
    if given or reference is None:
        path = Path(given or "").expanduser()
    else:
        try:
            model = json.loads((reference / "manifest.json").read_text()).get("model") or {}
        except (OSError, ValueError):
            model = {}
        name = str(model.get("path") or "")
        path = Path(name).expanduser()
        if name and not path.exists() and "/" in name and not name.startswith(("/", ".", "~")):
            return _cached(
                name, model.get("revision") or model.get("commit"), model.get("config_sha256"), reference
            )
    if not str(path) or not (path / "config.json").is_file():
        raise Refused(
            f"a .gguf checkpoint is decoded against the HF snapshot it was converted from; {path or 'none'} "
            "is not one",
            fix=f"set `{key}:` to the local HF snapshot the .gguf was converted from (a qda-agent search: set "
            "`judge.anchor`, which it passes as `{key}`)",
        )
    return path


def _cached(model: str, revision: str | None, sha: str | None, reference: Path) -> Path:
    """The cached snapshot of Hub id `model` (``qda-kl ref --model ORG/NAME``) whose config.json the reference hashed."""
    from qda_quant.exit import ENVIRONMENT
    from qda_quant.intake.model_ref import offline_refusal, parse

    from qda_kl.reference import cached_config

    ref = parse(model)
    revision = revision or ref.revision
    pin = f" --revision {revision}" if revision else ""
    config = cached_config(ref.name, revision)
    if config is None:
        raise offline_refusal(f"{ref.name}@{revision}" if revision else ref.name) or Refused(
            f"{ref.name}{'@' + revision if revision else ''}: the reference's model is not in the local "
            "Hugging Face cache",
            fix=f"hf download {ref.name}{pin}",
            code=ENVIRONMENT,
        )
    if sha is None and revision is None:
        raise Refused(
            f"{reference} records neither a revision nor a config.json hash for {ref.name}",
            fix="set `snapshot: path/to/original-snapshot` in the score config",
        )
    if sha is not None and hashlib.sha256(config.read_bytes()).hexdigest() != sha:
        raise Refused(
            f"{config}: not the config.json {reference} was built from (sha256 {sha[:12]})",
            fix="set `snapshot:` to the snapshot the reference was built from in the score config",
        )
    return config.parent


def decode(gguf: Path, snapshot: Path, out: Path) -> tuple[Path, dict]:
    """Decode `gguf` into ``<out>/decoded`` (replacing an earlier decode); return it and the manifest.

    Raises:
        Refused: the decoder cannot read the file (`qda_quant.gguf.decode`).
    """
    from qda_quant.gguf.decode import OTHER_GGUF
    from qda_quant.gguf.decode import decode as run_decode

    target = out / DECODED
    shutil.rmtree(target, ignore_errors=True)
    try:
        manifest = run_decode(gguf, target, snapshot)
    except ValueError as why:
        shutil.rmtree(target, ignore_errors=True)
        raise Refused(f"{gguf}: {why}", fix=OTHER_GGUF if "architecture" in str(why) else "") from None
    return target, manifest


def language_only(manifest: dict) -> bool:
    """Whether the decode holds a multimodal model's language weights only (Qwen3.5: no visual tower)."""
    return "note" in (manifest.get("config") or {})
