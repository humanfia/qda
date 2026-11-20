"""On-disk reference: the anchor's token sequences and its top-k log-probabilities at every scored position.

A reference directory holds four arrays and a manifest::

    tokens.npy    int32 [N_tok]      every sequence's token ids (prompt + scored continuation), concatenated
    topv.npy      fp32  [N_pos, k]   the anchor's top-k log-probabilities at each scored position, descending
    topi.npy      int32 [N_pos, k]   their token ids
    log_rest.npy  fp32  [N_pos]      log of the anchor's mass outside its top-k (fp32: it is the bucket's weight)
    manifest.json                    `Manifest`: how it was built, one `Sequence` entry per sequence, file hashes
    tokenizer/                       the anchor's own tokenizer files, so a score needs no model directory

Sequence ``s`` owns ``tokens[s.tok_off : s.tok_off + s.n_tokens]``; its scored tokens are those from ``s.start`` on,
and its rows ``[s.row_off, s.row_off + s.n_tokens - s.start)`` are the anchor's distributions at positions
``s.start - 1 .. s.n_tokens - 2`` (the positions that predict them). A directory is written whole into a sibling
temporary directory and renamed into place, so a reader never sees half of one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from qda_quant.step import Refused

from qda_kl.topk import TopK

FORMAT = "qda-kl-reference/1"
ARRAYS = ("tokens", "topv", "topi", "log_rest")
#: The anchor's tokenizer inside a reference: what a score re-reads, so the model directory may be gone.
TOKENIZER_DIR = "tokenizer"
DTYPES = {"tokens": np.int32, "topv": np.float32, "topi": np.int32, "log_rest": np.float32}


class Sequence(BaseModel):
    """One scored sequence of a reference.

    Attributes:
        id: unique id, ``<prompt id>#<draw>`` for an answer, the text's id for a human text.
        domain: the prompt's domain (per-domain KL groups by it).
        group: the prompt's report group, such as ``hard`` or ``easy`` ("" when the corpus gives none).
        split: the prompt's split, such as ``eval`` or ``holdout`` ("" when the corpus gives none).
        kind: ``answer`` (the anchor's own generation) or ``text`` (human-written text, teacher-forced only).
        tok_off: offset of its tokens in ``tokens.npy``.
        n_tokens: prompt + continuation length.
        start: index of the first scored token.
        row_off: offset of its rows in ``topv.npy`` / ``topi.npy`` / ``log_rest.npy``.
        finish: the generation's finish reason (answers).
    """

    id: str
    domain: str
    group: str = ""
    split: str = ""
    kind: Literal["answer", "text"] = "answer"
    tok_off: int
    n_tokens: int
    start: int
    row_off: int
    finish: str | None = None

    @property
    def n_scored(self) -> int:
        """Scored positions (the continuation's length)."""
        return self.n_tokens - self.start

    @property
    def prompt_id(self) -> str:
        """The source prompt id, shared by all of its answer draws."""
        return self.id.rsplit("#", 1)[0] if self.kind == "answer" else self.id


class Manifest(BaseModel):
    """Describe a reference's model, build settings, sequences and array hashes."""

    format: str = FORMAT
    content_sha256: str = Field(
        default="",
        description="sha256 of the data this reference holds: the array hashes, sequences and the metadata a "
        "score reads, excluding the build record (paths, versions, the clock) and this field. A reference built "
        "before the hash covered only the data records the old manifest hash instead; the reader accepts both. "
        "Empty in a reference built before the field existed; the reader then computes the data hash.",
    )
    k: int
    vocab: int
    model: dict = Field(
        default_factory=dict,
        description="the anchor: name, config hash, dtype, and `tokenizer` (where its own copy came from)",
    )
    build: dict = Field(
        default_factory=dict, description="sampling, seeds, answers per prompt, versions, timing"
    )
    sequences: list[Sequence] = Field(default_factory=list)
    subsets: dict[str, list[str]] = Field(default_factory=dict, description="named subsets of prompt ids")
    positions: int = 0
    tail_mass: dict = Field(
        default_factory=dict, description="the anchor's mass outside its top-k: max, p99, mean"
    )
    files: dict[str, str] = Field(
        default_factory=dict,
        description="sha256 of each file the reference holds: the arrays, and the anchor's tokenizer copy "
        "under `tokenizer/`",
    )


def _content_sha256(manifest: Manifest) -> str:
    """Hash what the reference holds, never how it was built.

    The build record names the prompt file's path, the library versions and the clock (``written_utc`` and
    ``seconds``), so two builds of byte-identical arrays would get different identities under it; a data identity
    covers the array hashes, sequences and the metadata a score reads. Newly added defaults cannot change a
    frozen identity because default values are excluded.
    """
    data = {**manifest.model_dump(exclude_defaults=True, exclude={"build"}), "content_sha256": ""}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _legacy_content_sha256(manifest: Manifest) -> str:
    """The identity a reference recorded before this one covered only the data; it hashed the build record too."""
    blank = {**manifest.model_dump(exclude_defaults=True), "content_sha256": ""}
    return hashlib.sha256(json.dumps(blank, sort_keys=True).encode()).hexdigest()


@dataclass
class ReferenceWriter:
    """Collect sequences in memory and atomically write their reference directory on :meth:`close`."""

    out: Path
    k: int
    vocab: int
    model: dict = field(default_factory=dict)
    build: dict = field(default_factory=dict)
    subsets: dict[str, list[str]] = field(default_factory=dict)
    #: Directory of the anchor's tokenizer files, copied into the reference so a score needs no model directory.
    tokenizer: Path | None = None
    _seqs: list[Sequence] = field(default_factory=list)
    _parts: dict[str, list[np.ndarray]] = field(default_factory=lambda: {a: [] for a in ARRAYS})
    _tok: int = 0
    _row: int = 0

    def add(
        self,
        id: str,
        domain: str,
        tokens: list[int],
        start: int,
        anchor: TopK,
        *,
        group: str = "",
        split: str = "",
        kind: Literal["answer", "text"] = "answer",
        finish: str | None = None,
    ) -> Sequence:
        """Add one sequence: its tokens, the first scored index and the anchor's top-k at its scored positions."""
        n_rows = len(tokens) - start
        if not 0 < start < len(tokens):
            raise ValueError(f"{id}: start {start} outside (0, {len(tokens)})")
        if anchor.topv.shape != (n_rows, self.k):
            raise ValueError(f"{id}: top-k {tuple(anchor.topv.shape)}, expected {(n_rows, self.k)}")
        seq = Sequence(
            id=id,
            domain=domain,
            group=group,
            split=split,
            kind=kind,
            tok_off=self._tok,
            n_tokens=len(tokens),
            start=start,
            row_off=self._row,
            finish=finish,
        )
        got = {
            "tokens": np.asarray(tokens),
            "topv": anchor.topv.cpu().numpy(),
            "topi": anchor.topi.cpu().numpy(),
            "log_rest": anchor.log_rest.cpu().numpy(),
        }
        for name, arr in got.items():
            self._parts[name].append(arr.astype(DTYPES[name]))
        self._seqs.append(seq)
        self._tok += len(tokens)
        self._row += n_rows
        return seq

    def close(self) -> Manifest:
        """Write the arrays and manifest into ``out`` (a temporary sibling renamed into place).

        Raises:
            Refused: ``out`` exists, a dangling link included (an existing reference is never replaced or
                deleted).
        """
        if len({s.id for s in self._seqs}) != len(self._seqs):
            raise ValueError("sequence ids are not unique")
        out = Path(self.out)
        if os.path.lexists(out):
            raise _taken(out)
        tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            manifest = self._write(tmp)
            if os.path.lexists(out):  # written while this one was
                raise _taken(out)
            tmp.rename(out)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return manifest

    def _write(self, tmp: Path) -> Manifest:
        """Write the arrays and the manifest into the directory ``tmp``."""
        files = {}
        for name in ARRAYS:
            parts = self._parts[name]
            arr = (
                np.concatenate(parts)
                if parts
                else np.zeros((0, self.k) if name in ("topv", "topi") else 0, DTYPES[name])
            )
            np.save(tmp / f"{name}.npy", arr)
            files[f"{name}.npy"] = _sha256(tmp / f"{name}.npy")
        files.update(copy_tokenizer(self.tokenizer, tmp) if self.tokenizer is not None else {})
        rest = (
            np.exp(np.concatenate(self._parts["log_rest"]).astype(np.float64)) if self._row else np.zeros(1)
        )
        manifest = Manifest(
            k=self.k,
            vocab=self.vocab,
            model=self.model,
            build={**self.build, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            sequences=self._seqs,
            subsets=self.subsets,
            positions=self._row,
            files=files,
            tail_mass={
                "max": float(rest.max()),
                "p99": float(np.quantile(rest, 0.99)),
                "mean": float(rest.mean()),
            },
        )
        manifest.content_sha256 = _content_sha256(manifest)
        (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=1))
        return manifest


def _taken(out: Path) -> Refused:
    """The refusal for an output path that is already there."""
    return Refused(f"{out} exists; qda-kl ref writes only a new directory", "pass another --out")


def copy_tokenizer(src: Path, out: Path) -> dict[str, str]:
    """Copy the tokenizer files of `src` into ``out/tokenizer``; return their sha256 by path in the reference.

    `src` holds the tokenizer files themselves and the directory of named chat templates a checkpoint may
    carry; everything in it is copied. The keys are the paths the manifest records, so a reader verifies the
    copy as it verifies the arrays.
    """
    dst = out / TOKENIZER_DIR
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return {
        f"{TOKENIZER_DIR}/{f.relative_to(dst).as_posix()}": _sha256(f)
        for f in sorted(dst.rglob("*"))
        if f.is_file()
    }


def write_reduced(ref: Reference, sequences: list, out: Path) -> Path:
    """Write `out` as a reference holding only `sequences` of `ref`, at no extra bytes.

    The arrays and the tokenizer are symlinks to `ref`'s and every sequence keeps its absolute offsets, so a
    scorer reads `out` as an ordinary reference. The manifest keeps no subsets and records its own content sha256.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name in (*(f"{a}.npy" for a in ARRAYS), TOKENIZER_DIR):
        link, source = out / name, ref.path / name
        if not link.exists() and source.exists():
            link.symlink_to(source.resolve())
    manifest = ref.manifest.model_copy(
        update={
            "sequences": sequences,
            "subsets": {},
            "positions": sum(s.n_scored for s in sequences),
            "content_sha256": "",
        }
    )
    manifest = manifest.model_copy(update={"content_sha256": _content_sha256(manifest)})
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=1))
    return out


class Reference:
    """Read a reference directory using memory-mapped arrays."""

    def __init__(self, path: str | Path, verify: bool = False, *, subset: str | None = None):
        """Open ``path``; ``subset`` selects its prompts; ``verify`` re-hashes arrays against the manifest.

        Raises:
            Refused: ``path`` holds no reference, another format, a manifest whose content no longer matches
                the sha256 it records, or (``verify``) a missing or changed array.
        """
        self.path = Path(path)
        if not (self.path / "manifest.json").is_file():
            what = "no such directory" if not self.path.is_dir() else "no manifest.json in it"
            raise Refused(f"{self.path}: not a qda-kl reference ({what}); build one with `qda-kl ref`")
        self.manifest = Manifest.model_validate_json((self.path / "manifest.json").read_text())
        self.subset, self._sequences = subset, self.manifest.sequences
        if subset is not None:
            if subset not in self.manifest.subsets:
                raise ValueError(f"{self.path}: unknown reference subset {subset!r}")
            ids = set(self.manifest.subsets[subset])
            self._sequences = [s for s in self._sequences if s.prompt_id in ids]
        if self.manifest.format != FORMAT:
            raise Refused(f"{self.path}: format {self.manifest.format!r}, this reader reads {FORMAT!r}")
        recorded = self.manifest.content_sha256
        current = _content_sha256(self.manifest)
        # A reference built before the hash covered only the data records the old manifest hash: still checked.
        if recorded and recorded != current and recorded != _legacy_content_sha256(self.manifest):
            raise Refused(
                f"{self.path / 'manifest.json'}: recorded content sha256 {recorded} differs from current "
                f"{current}; explicitly re-hash or rebuild the reference"
            )
        if verify:
            for name, sha in self.manifest.files.items():
                if not (self.path / name).is_file():
                    raise Refused(f"{self.path / name}: missing; the reference is incomplete")
                if _sha256(self.path / name) != sha:
                    raise Refused(f"{self.path / name}: sha256 differs from the manifest")
        self._arr = {a: np.load(self.path / f"{a}.npy", mmap_mode="r") for a in ARRAYS}

    @property
    def sequences(self) -> list[Sequence]:
        """Return the selected subset's sequences, or all sequences, in build order."""
        return self._sequences

    @property
    def content_sha256(self) -> str:
        """The reference's identity, as recorded when it was built.

        A reference built before the manifest recorded it gets the hash computed from the manifest read now;
        nothing is written into a frozen reference.
        """
        return self.manifest.content_sha256 or _content_sha256(self.manifest)

    @property
    def tokenizer(self) -> Path | None:
        """The anchor's own tokenizer inside the reference, or None in one built before references held it."""
        path = self.path / TOKENIZER_DIR
        return path if (path / "tokenizer_config.json").is_file() else None

    def tokens(self, s: Sequence) -> list[int]:
        """Return the prompt and continuation token ids of sequence ``s``."""
        return self._arr["tokens"][s.tok_off : s.tok_off + s.n_tokens].tolist()

    def topk(self, s: Sequence, device: str | torch.device = "cpu") -> TopK:
        """Load the anchor's stored top-k at the scored positions of ``s`` onto ``device``."""
        r = slice(s.row_off, s.row_off + s.n_scored)
        return TopK(
            *(torch.from_numpy(np.array(self._arr[a][r])).to(device) for a in ("topv", "topi", "log_rest"))
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()
