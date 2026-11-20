"""Open a checkpoint without loading its weights: file names, tensor headers and JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .model_ref import parse

INDEX = "model.safetensors.index.json"


#: any one of these sets is a tokenizer vLLM can load
TOKENIZER_SETS = (("tokenizer.json",), ("tokenizer.model",), ("vocab.json", "merges.txt"))


PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")


class IntakeRefused(ValueError):
    """The checkpoint cannot be read or converted; the message says why."""


@dataclass
class Source:
    """What `inspect` reads of a checkpoint: its file names, its JSON files and its tensor headers."""

    where: str
    files: set[str]
    tensors: dict[str, tuple[str, list[int]]]
    jsons: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str) -> Source:
        """A local directory, or an HF repo id (``org/name[@rev]``) read through the hub API without weights."""
        p = Path(path)
        if p.is_dir():
            files = {f.name for f in p.iterdir() if f.is_file()}
            tensors = _headers(p, {f for f in files if f.endswith(".safetensors")})
            jsons = {n: json.loads((p / n).read_text()) for n in _JSONS if n in files}
            return cls(str(p), files, tensors, jsons)
        if p.suffix == ".gguf" and p.is_file():
            return cls(str(p), {p.name}, {})
        if p.exists() or path.count("/") != 1:
            raise IntakeRefused(f"{path}: not a checkpoint directory, a .gguf file or an HF repo id")
        return cls._hub(path)

    @classmethod
    def _hub(cls, repo_rev: str) -> Source:
        from huggingface_hub import HfApi, hf_hub_download

        repo, rev = parse(repo_rev)
        api = HfApi()
        try:
            files = set(api.list_repo_files(repo, revision=rev))
        except Exception as e:  # noqa: BLE001  (any hub error: say it, refuse)
            raise IntakeRefused(f"{repo_rev}: not a local path, and the hub says: {e}") from None
        tensors: dict[str, tuple[str, list[int]]] = {}
        if any(f.endswith(".safetensors") for f in files):
            meta = api.get_safetensors_metadata(repo, revision=rev)
            for fm in meta.files_metadata.values():
                tensors.update({k: (t.dtype, list(t.shape)) for k, t in fm.tensors.items()})
        jsons = {}
        for n in _JSONS:
            if n in files:
                jsons[n] = json.loads(Path(hf_hub_download(repo, n, revision=rev)).read_text())
        return cls(repo_rev, {f for f in files if "/" not in f}, tensors, jsons)


_JSONS = ("config.json", "build.json", "tokenizer_config.json")


def _headers(src: Path, files: set[str]) -> dict[str, tuple[str, list[int]]]:
    """(dtype, shape) of every tensor, from the safetensors headers (no tensor is read)."""
    out = {}
    for f in files:
        with (src / f).open("rb") as h:
            n = int.from_bytes(h.read(8), "little")
            if n > (src / f).stat().st_size - 8:  # not safetensors: refuse, never allocate `n` bytes
                raise IntakeRefused(f"{src / f}: not a safetensors file (its header is {n} bytes)")
            head = json.loads(h.read(n))
        out.update({k: (v["dtype"], v["shape"]) for k, v in head.items() if k != "__metadata__"})
    return out


def _weight_map(src: Path) -> dict[str, str]:
    """Source tensor name -> safetensors shard file, from the index or from the shards' headers."""
    idx = src / INDEX
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    out = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out.update(dict.fromkeys(h.keys(), f.name))
    if not out:
        raise FileNotFoundError(
            f"{src}: no *.safetensors weights; the writer copies and packs the source shards"
        )
    return out


def _tied(cfg: dict[str, Any]) -> bool:
    """Whether the config ties the head to the embedding: ``text_config``'s flag, else the top-level one.

    A vision model nests its text model's config under ``text_config``; a text model keeps the flag at the top.
    """
    text = cfg.get("text_config") or {}
    return bool(text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
