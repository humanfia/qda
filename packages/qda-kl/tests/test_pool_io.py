from __future__ import annotations

import hashlib
import json
import mmap
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pytest
import qda_kl.eval.run as runner
import qda_kl.store as store
import torch
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.suites import Suite
from qda_kl.topk import anchor_topk
from qda_quant.step import Refused


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "ref"
    writer = store.ReferenceWriter(root, k=2, vocab=4)
    writer.add("one", "math", [1, 2, 3], 1, anchor_topk(torch.log_softmax(torch.zeros(2, 4), -1), 2))
    writer.close()
    return root


@pytest.mark.parametrize("replacement", [False, True])
def test_modified_array_is_refused_even_with_preserved_mtime(reference, replacement):
    store.Reference(reference, verify=True)
    path = reference / "tokens.npy"
    before = path.stat()
    target = path.with_suffix(".replacement") if replacement else path
    if replacement:
        shutil.copy2(path, target)
    with target.open("r+b") as array:
        array.seek(-4, os.SEEK_END)
        array.write((0).to_bytes(4, "little"))
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    if replacement:
        target.replace(path)
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(Refused, match="sha256 differs"):
        store.Reference(reference, verify=True)


def _sees_unflushed_mmap_writes(directory: Path) -> bool:
    """Whether a `read()` on this filesystem sees a write through a mapping that is not yet flushed (not on GPFS)."""
    probe = directory / "mmap-probe"
    probe.write_bytes(b"\0" * mmap.PAGESIZE)
    with probe.open("r+b") as f, mmap.mmap(f.fileno(), 0) as mapped:
        mapped[0] = 1
        return probe.read_bytes()[0] == 1


def test_repeated_mmap_write_is_refused(reference):
    if not _sees_unflushed_mmap_writes(reference.parent):
        pytest.skip(f"{reference.parent}: read() here does not see an unflushed mmap write (GPFS)")
    path = reference / "tokens.npy"
    with path.open("r+b") as array, mmap.mmap(array.fileno(), 0) as mapped:
        offset = len(mapped) - 4
        mapped[offset] = mapped[offset]
        time.sleep(1.1)
        store.Reference(reference, verify=True)
        mapped[offset] = 0
        expected = store.Reference(reference).manifest.files[path.name]
        assert store._sha256(path) != expected
        with pytest.raises(Refused, match="sha256 differs"):
            store.Reference(reference, verify=True)


class _Inputs(Suite):
    name: ClassVar[str] = "pool_io_probe"
    path: str
    label: str = ""

    def parts(self):
        return [self.model_copy(update={"label": label}) for label in ("a", "b")]

    def part_label(self):
        return self.label or None

    def inputs(self):
        return {"data": runner.sha256_file(Path(self.path))}


def _old_digest(part, reference, checkpoint, files):
    key = {
        "suite": part.to_dict(),
        "inputs": part.inputs(),
        "reference": reference,
        "checkpoint": checkpoint,
    }
    if files:
        key["checkpoint_files"] = files
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def test_eval_hashes_once_and_preserves_job_digests(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"one"}\n')
    checkpoints = {}
    for key in ("one", "two"):
        root = tmp_path / key
        root.mkdir()
        (root / "generation_config.json").write_text('{"temperature":0.5}')
        checkpoints[key] = str(root)
    cfg = EvalConfig.model_construct(
        reference="reference",
        checkpoints=checkpoints,
        out=str(tmp_path / "out"),
        suites=[_Inputs(path=str(data))],
    )
    expected = [
        _old_digest(part, cfg.reference, checkpoint, runner.checkpoint_files(checkpoint))
        for checkpoint in checkpoints.values()
        for part in cfg.suites[0].parts()
    ]
    calls = Counter()
    original = runner.sha256_file

    def counted(path):
        calls[str(path)] += 1
        return original(path)

    monkeypatch.setattr(runner, "sha256_file", counted)
    planned = runner.jobs(cfg)
    assert [job.digest for job in planned] == expected
    assert calls[str(data)] == 2
    assert all(
        calls[str(Path(checkpoint) / "generation_config.json")] == 1 for checkpoint in checkpoints.values()
    )
    data.write_text('{"text":"two"}\n')
    assert all(before.digest != after.digest for before, after in zip(planned, runner.jobs(cfg), strict=True))
