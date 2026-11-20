"""The one reading of a model reference: a local path, or a Hub id with an optional ``@revision``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ..step import Refused

#: A full commit hash: the name of a snapshot directory in the Hugging Face cache.
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ModelRef(NamedTuple):
    """`name` is what transformers and the hub take (a path, or the repo id); `revision` follows ``@``."""

    name: str
    revision: str | None

    def hub(self) -> dict[str, Any]:
        """The keyword that pins the revision for `from_pretrained` and the hub: ``{"revision": rev}`` or none."""
        return {} if self.revision is None else {"revision": self.revision}


def parse(model: str | Path) -> ModelRef:
    """Split ``org/name@rev`` at its first ``@``; a `Path`, or a path that exists, is never split.

    Transformers does not read ``@rev``: handed the raw string, it looks for a repo named ``name@rev``.
    """
    text = str(model)
    if isinstance(model, Path) or Path(text).expanduser().exists():
        return ModelRef(text, None)
    name, _, revision = text.partition("@")
    return ModelRef(name, revision or None)


def offline_refusal(model: str | Path) -> Refused | None:
    """Offline, the refusal for a Hub id whose revision the local Hugging Face cache cannot map to a snapshot.

    A download pinned to a commit (``hf download ID --revision SHA``) writes that snapshot and no ``refs/main``,
    so offline the bare id names nothing even though the files are there. None online, for a local path, and
    when the cache holds the revision (a commit's snapshot directory, or a branch or tag under ``refs/``); a name
    that cannot be a Hub id (no ``/``, or a leading ``/``, ``.`` or ``~``) is left to the caller's own error.
    """
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    from ..exit import ENVIRONMENT
    from ..step import Refused

    ref = parse(model)
    local = ref.name.startswith(("/", ".", "~")) or "/" not in ref.name  # cannot be ``org/name``
    if not constants.HF_HUB_OFFLINE or local or Path(ref.name).expanduser().exists():
        return None
    repo = Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=ref.name, repo_type="model")
    snapshots = repo / "snapshots"
    held = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime) if snapshots.is_dir() else []
    revision = ref.revision or "main"
    named = repo / "refs" / revision
    commit = (
        revision if _COMMIT.fullmatch(revision) else (named.read_text().strip() if named.is_file() else "")
    )
    if commit and (snapshots / commit).is_dir():
        return None
    offline = "and the hub cannot be reached (HF_HUB_OFFLINE is set)"
    pin = f" --revision {ref.revision}" if ref.revision else ""
    if not held:
        return Refused(
            f"{ref.name}: the local Hugging Face cache holds no snapshot of it {offline}",
            f"download it once with network access: hf download {ref.name}{pin}",
            code=ENVIRONMENT,
        )
    what = f"no snapshot at revision {revision}" if _COMMIT.fullmatch(revision) else f"no `{revision}` ref"
    return Refused(
        f"{ref.name}: the local Hugging Face cache has {what} for it (it holds "
        f"{', '.join(p.name for p in held)}) {offline}",
        f"name the cached revision, {ref.name}@{held[-1].name}, or download it once with network access: "
        f"hf download {ref.name}{pin}",
        code=ENVIRONMENT,
    )
