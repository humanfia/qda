"""The launch check's KL half: candidate 0's reading must fold valid before the host opens (CPU)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from qda_agent.judge import jobs
from qda_agent.sched import Refused
from test_build_smoke import needs_bwrap, sources
from test_judge import KL_RESULT, PY, Rig, ckpt, evaluator


def reference(path: Path, ids: list[str]) -> Path:
    """A reference the judge's own reader opens, its prompt file recorded beside it (no arrays to check)."""
    import torch
    from qda_kl.store import ReferenceWriter
    from qda_kl.topk import anchor_topk

    prompts = path.parent / f"{path.name}-prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps({"id": i, "prompt": "a reference prompt used by the judge tests only"}) + "\n"
            for i in ids
        )
    )
    anchor = anchor_topk(torch.tensor([[1.0, 2.0, 3.0, 4.0]]).log_softmax(-1), 2)
    writer = ReferenceWriter(path, k=2, vocab=4, model={"name": "tiny"}, build={"prompts": str(prompts)})
    for i in ids:
        writer.add(i, "math", [0, 1], 1, anchor)
    writer.close()
    return path


def rig(tmp_path: Path, ref: Path, **judge) -> Rig:
    """A receipt-requiring host whose only launch check under test is the candidate-0 reading."""
    r = Rig(
        tmp_path,
        baselines=[ckpt(tmp_path / "base")],
        qda_cmd=[PY],
        evaluator=evaluator(tmp_path),
        require_build_receipt=True,
        build_receipt_smoke=False,
        candidate0_smoke=False,
        kl_ref=str(ref),
        **judge,
    )
    r.host.cfg.kl_cmd = ["score", "{config}", "{out}"]  # the judge writes the config the scorer reads
    return r


def scored(ids: list[str]) -> dict:
    """A complete reading of `ids`: every one of them scored with a finite KL and a token count."""
    value = KL_RESULT["per_seq"]["p1#0"]
    return {**KL_RESULT, "per_seq": {i: value for i in ids}}


def writes(reading: dict):
    """An effect that writes `reading` where the scorer's `{out}` names it, making its directory."""

    def effect(argv):
        out = Path(argv[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reading))

    return effect


def test_a_complete_reading_of_candidate_0_opens_the_run_and_leaves_no_trace(tmp_path):
    ids = ["s0", "s1", "s2"]
    r = rig(tmp_path, reference(tmp_path / "ref", ids))
    r.be.default.effect = writes(scored(ids))
    r.host.start()
    try:
        assert r.host.server is not None
        assert r.host.entries == [], "candidate 0's launch reading is no candidate"
        results = r.host.jdir / "results"
        assert not (results / "smoke.yaml").exists() and not (results / "smoke").exists()
        assert not (r.host.jdir / "tmp" / "smoke").exists()
        ledger = r.host.jdir / "ledger.jsonl"
        assert not ledger.exists() or "smoke" not in ledger.read_text()
    finally:
        r.host.stop()


def test_a_reading_of_only_a_subset_of_the_reference_is_refused_naming_both_counts(tmp_path):
    ids = ["s0", "s1", "s2"]
    r = rig(tmp_path, reference(tmp_path / "ref", ids))
    r.be.default.effect = writes(scored(ids[:2]))
    with pytest.raises(ValueError, match=r"candidate 0.*leaves 1 of the reference's 3 sequences unscored"):
        r.host.start()
    assert r.host.server is None
    assert r.host.entries == []


def test_a_configured_reading_refusal_leaves_the_baseline_byte_identical(tmp_path, monkeypatch):
    def refuse(*args):  # the judge cannot assemble the reading's config
        raise Refused("the configured reading cannot be assembled")

    monkeypatch.setattr(jobs, "job_config", refuse)
    r = rig(tmp_path, reference(tmp_path / "ref", ["s0", "s1", "s2"]))
    baseline = r.host.cfg.baselines[0]
    before = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="cannot be read"):
        r.host.start()
    assert r.host.server is None
    after = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    assert after == before, "a launch refusal deleted or changed the owner's baseline"


@needs_bwrap
def test_no_baseline_refuses_instead_of_skipping_the_reading(tmp_path):
    r = Rig(
        tmp_path,
        readable=sources(),
        require_build_receipt=True,
        candidate0_smoke=False,  # this test is about the reading half; the run names no model to build
        qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
        qda_env={"PYTHONPATH": os.pathsep.join(map(str, sources()))},
    )
    assert r.host.cfg.reading_smoke and r.host.cfg.build_receipt_smoke, "both halves are on by default"
    with pytest.raises(
        ValueError, match=r"judge\.reading_smoke.*add a baseline, or set judge\.reading_smoke: false"
    ):
        r.host.start()
    assert r.host.server is None
