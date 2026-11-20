import contextlib
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from qda_agent import rpc
from qda_agent.commands import install_bin
from qda_agent.host import HandIn, delivery, unmeasured
from qda_agent.isolation.bwrap import interpreter
from qda_agent.judge import JudgeConfig, JudgeHost, contamination, jobs, make_host, startup
from qda_agent.judge.checks import files_problem
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.final_rules import apply, final_handin_statement, statement
from qda_agent.judge.jobs import failing_line
from qda_agent.rpc import Request
from qda_agent.sched import FakeBackend, FakeClock, FakeRun, SchedConfig, State
from safetensors_kit import packed_tensors, quant_dir, write_safetensors

_make_host = make_host


def make_host(run, sched, judge, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
    """`make_host` with a test index: these rigs test the judge, not the search's contamination setup."""
    judge = {"contamination": "none", **judge}  # a JudgeConfig has no default check: the loader fills it
    if not {"kl_ref", "kl_holdout_ref", "contamination_index"} & set(judge):
        ref = str(_ref_dir(Path(run).parent))
        judge |= {"kl_ref": ref, "kl_holdout_ref": ref}  # these rigs test the judge, not the search
    return _make_host(run, sched, judge, *args, **kwargs)


def _ref_dir(root, name="judge-ref"):
    """An eval+holdout reference: the judge host derives its calibration index from configured references."""
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    prompts = d / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "r1", "prompt": "a reference prompt used by the judge tests only"})
        + "\n"
        + json.dumps(
            {"id": "h1", "prompt": "a holdout prompt used by the judge tests only", "split": "holdout"}
        )
        + "\n"
    )
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return d


def _index_file(root):
    """A prebuilt index equal to the canonical union of `_ref_dir`'s eval and holdout sets."""
    from qda_quant.calib.contamination import PromptIndex

    return PromptIndex.read_reference(_ref_dir(root)).save(Path(root) / "contamination-index.json")


PY = os.path.normpath(Path(sys.executable).absolute())
KL_RESULT = {  # a qda_kl Reading (qda_kl/tally.py)
    "kl": 0.0712,
    "tokens": 123456,
    "domains": {"code": 0.08, "math": 0.065},
    "domain_tokens": {"code": 1000, "math": 2000},
    "kl_text": 0.069,
    "text_tokens": 777,
    "domains_text": {"chat": 0.0691},
    "domain_text_tokens": {"chat": 777},
    "flip_rate": 0.0123,
    "per_seq": {
        "p1#0": {"kl": 0.31415, "tokens": 40, "domain": "code", "kind": "answer"},
        "p2#0": {"kl": 0.27182, "tokens": 30, "domain": "math", "kind": "answer"},
    },
    "trace": "/data/secret/traces.npz",
    "kernels": {"dense": "MarlinLinearKernel"},
    "vllm": "0.33.0",
    "vllm_mismatch": False,
    "dtype": "bfloat16",
    "dtype_from": "config.json",
    "serve": {"env": {"VLLM_SECRET_FLAG": "1"}, "args": {"moe_backend": "marlin"}},
    "checkpoint": "/data/run/judge/frozen/k1",
    "reference": "/data/refs/q8b-kl3/secret-ref",
    "seconds": {"load": 61.5, "score": 14.2},
    "wall_s": 88.8,
}


def evaluator(tmp_path: Path) -> Path:
    ev = tmp_path / "src" / "fake_eval"
    ev.mkdir(parents=True)
    (ev / "__init__.py").write_text("VERSION = 'v1'\n")
    return ev


def ckpt(d: Path, weights: bytes = b"w1") -> Path:
    return quant_dir(d, weights)  # a real (tiny) safetensors checkpoint


def shard_bytes(tmp_path: Path, weights: bytes = b"w1") -> bytes:
    p = tmp_path / f"expected-{weights.hex()}.safetensors"
    write_safetensors(p, packed_tensors(weights))
    return p.read_bytes()


def write_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> Path:
    """A minimal little-endian GGUF: no KV pairs, every tensor's data aligned to 32 bytes."""

    def text(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    infos, bodies, at = bytearray(), [], 0
    for name, shape, kind, data in tensors:
        infos += text(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        infos += struct.pack("<IQ", kind, at)
        bodies.append(data + b"\0" * (-len(data) % 32))
        at += len(bodies[-1])
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0) + bytes(infos)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + b"".join(bodies))
    return path


class Rig:
    def __init__(
        self,
        tmp_path,
        check=True,
        budget_s=3600.0,
        baselines=(),
        grace_s=900.0,
        hide=(),
        readable=(),
        **judge,
    ):
        judge.setdefault("kl_ref", str(_ref_dir(tmp_path)))
        judge.setdefault("kl_holdout_ref", str(_ref_dir(tmp_path)))
        judge.setdefault("kl_cmd", ["score", "{ckpt}", "{out}"])  # a test names its own command
        judge.setdefault("kl_score_s", 60)  # a test may name its own estimate
        judge.setdefault("kl_load_s_per_gib", 0)
        judge.setdefault("contamination", "none")  # a JudgeConfig has no default check: the loader fills it
        self.clock = FakeClock()
        self.run = tmp_path / "run"
        self.work = self.run / "work"
        self.work.mkdir(parents=True)
        self.seen: dict[str, bytes] = {}
        self.be = FakeBackend(self.clock, default=FakeRun(30, effect=self.effect))
        self.host = JudgeHost(
            self.run,
            SchedConfig(
                cards=[0, 1],
                budget_s=budget_s,
                fake_gpus=True,
                judge_grace_s=grace_s,
                hide=list(hide),
                readable=list(readable),
            ),
            JudgeConfig(
                check_cmd=["check", "{ckpt}", "{out}"] if check else None,
                check_expected_s=60,
                baselines=list(baselines),
                **judge,
            ),
            clock=self.clock,
            backend=self.be,
            autotick=False,
        )
        self.check_result = {"ok": True}

    def effect(self, argv):
        if argv[0] not in ("score", "check"):
            return
        what, ck, out = argv
        self.seen[ck] = (Path(ck) / "model.safetensors").read_bytes()
        Path(out).write_text(json.dumps(KL_RESULT if what == "score" else self.check_result))

    def call(self, verb, *argv, cwd="."):
        return self.host.handle(Request(verb, list(argv), str((self.work / cwd).resolve())))

    def settle(self):
        for _ in range(20):
            self.host.tick()
            self.clock.advance(30)
        self.host.tick()


def test_kl_scores_a_frozen_copy_outside_the_workdir(tmp_path):
    r = Rig(tmp_path)
    src = ckpt(r.work / "out" / "a")
    rep = r.call("kl", "out/a")
    assert rep.exit == 0 and rep.stdout.startswith("k1 out/a: reading")
    (src / "model.safetensors").write_bytes(b"edited")  # after the request: changes nothing
    r.settle()
    (frozen,) = r.seen
    assert r.seen[frozen] == shard_bytes(tmp_path)
    assert Path(frozen).is_relative_to(r.run / "judge") and not Path(frozen).is_relative_to(r.work)
    assert not os.access(Path(frozen) / "model.safetensors", os.W_OK) or os.geteuid() == 0
    assert stat.S_IMODE(Path(frozen).stat().st_mode) & 0o222 == 0


def kl_effect(r, value):
    """Score every reading with `value` (the rest of KL_RESULT unchanged)."""
    old = r.effect

    def effect(argv):
        old(argv)
        if argv[0] == "score":
            Path(argv[-1]).write_text(json.dumps(dict(KL_RESULT, kl=value)))

    r.be.default.effect = effect


@pytest.mark.parametrize("value", [math.nextafter(1.0, 0.0), 1.0])
def test_a_kl_up_to_the_limit_is_a_reading(tmp_path, value):
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    kl_effect(r, value)
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is True and e.metrics["kl"] == value


def test_a_kl_over_the_limit_is_refused_with_the_rotation_hint(tmp_path):
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    kl_effect(r, 1.2)
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is False and "KL > 1" in e.reason, e.reason
    assert "rotation" in e.reason and "basis" in e.reason, e.reason
    # a refused reading is spent, but its copy stays until a pass near the floor reclaims it
    assert e.dir.exists()
    assert r.host.final(generation_checks=False).choice is None


def test_the_kl_limit_is_a_configured_setting(tmp_path):
    r = Rig(tmp_path, max_kl=2.0)
    ckpt(r.work / "a")
    kl_effect(r, 1.2)
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is True and e.metrics["kl"] == 1.2


def result_effect(r, result):
    def effect(argv):
        if argv[0] == "score":
            Path(argv[-1]).write_text(json.dumps(result))

    r.be.default.effect = effect


@pytest.mark.parametrize(
    ("result", "why"),
    [
        ({"kl": 0.01}, "no scored tokens"),
        ({"kl": 0.01, "tokens": 0, "per_seq": {}}, "no scored tokens"),
        ({"kl": 0.01, "tokens": 10, "per_seq": {}}, "no per-sequence readings"),
        ({"kl": 0.01, "tokens": 10, "per_seq": {"p1#0": None}}, "per-sequence readings are incomplete"),
        ({"kl": 0.01, "tokens": 10, "per_seq": {"p1#0": {}}}, "per-sequence readings are incomplete"),
        (
            # a raw KL reading: an otherwise complete tally whose command reports a failure
            {
                "kl": 0.01,
                "tokens": 10,
                "per_seq": {"p1#0": {"kl": 0.01, "tokens": 1}},
                "error": "scoring failed",
            },
            "failed: scoring failed",
        ),
        (
            {"suite": "kl", "metrics": {"kl": 0.01}, "higher_is_better": {"kl": False}, "error": "no ref"},
            "failed: no ref",
        ),
    ],
)
def test_an_incomplete_reading_is_refused(tmp_path, result, why):
    """A bare stat line, an empty tally or a suite result with an error may never become valid or pick."""
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    result_effect(r, result)
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is False and why in e.reason, e.reason
    assert not e.metrics and not e.higher_is_better
    assert r.host.final(generation_checks=False).choice is None


def test_a_reading_must_cover_the_references_sequences(tmp_path, monkeypatch):
    """A tally of part of the reference cannot count: every sequence the reference holds must be read."""
    monkeypatch.setattr(jobs, "_reference_ids", lambda *args: {"a#0", "b#0"})
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    result_effect(r, dict(KL_RESULT, per_seq={"a#0": {}}))
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is False and "1 of the reference's 2" in e.reason, e.reason
    assert r.host.final(generation_checks=False).choice is None


def test_the_expected_cohort_comes_from_the_jobs_config(tmp_path, monkeypatch):
    """A result's own subset claim changes nothing: the config names the reference a reading must cover."""
    import qda_kl.store as store

    seen = []

    class Fake:
        def __init__(self, path, subset=None):
            seen.append((str(path), subset))
            self.sequences = [SimpleNamespace(id=i) for i in ("a#0", "b#0")]

    monkeypatch.setattr(store, "Reference", Fake)
    r = Rig(tmp_path)
    config = r.host.jdir / "results" / "x.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"reference": "/refs/original"}))
    assert jobs._reference_ids(r.host, SimpleNamespace(id="x")) == {"a#0", "b#0"}
    assert seen == [("/refs/original", None)]


@pytest.mark.parametrize(
    ("claimed_subset", "per_seq"),
    [
        ("half", {"a#0": {"kl": 0.01, "tokens": 1, "domain": "math"}}),
        ("unknown", {"a#0": {"kl": 0.01, "tokens": 1, "domain": "math"}}),
        (None, {"a#0": None, "b#0": None}),
        (None, {"a#0": {}, "b#0": {}}),
    ],
    ids=["smaller-subset", "unknown-subset", "null-readings", "empty-readings"],
)
def test_a_result_cannot_replace_missing_reference_readings(tmp_path, claimed_subset, per_seq):
    """A result's subset or empty per-sequence values cannot replace readings of the configured reference."""
    import numpy as np
    from qda_kl.store import Manifest, Sequence

    reference = tmp_path / "reference"
    reference.mkdir()
    sequences = [
        Sequence(id=f"{name}#0", domain="math", tok_off=2 * i, n_tokens=2, start=1, row_off=i)
        for i, name in enumerate(("a", "b"))
    ]
    prompts = reference / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "r1", "prompt": "a reference prompt used by the judge tests only"})
        + "\n"
        + json.dumps(
            {"id": "h1", "prompt": "a holdout prompt used by the judge tests only", "split": "holdout"}
        )
        + "\n"
    )
    manifest = Manifest(
        k=1,
        vocab=3,
        positions=2,
        sequences=sequences,
        subsets={"half": ["a"]},
        build={"prompts": str(prompts)},  # the judge derives its contamination index from this record
    )
    (reference / "manifest.json").write_text(manifest.model_dump_json())
    arrays = {
        "tokens": np.array([1, 2, 1, 2], dtype=np.int32),
        "topv": np.zeros((2, 1), dtype=np.float32),
        "topi": np.ones((2, 1), dtype=np.int32),
        "log_rest": np.zeros(2, dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(reference / f"{name}.npy", array)

    r = Rig(tmp_path, kl_ref=str(reference))
    r.host.cfg.kl_cmd = ["score", "{ref}", "{out}"]
    ckpt(r.work / "a")
    result_effect(
        r,
        {
            "kl": 0.01,
            "tokens": len(per_seq),
            "domains": {"math": 0.01},
            "domain_tokens": {"math": len(per_seq)},
            "per_seq": per_seq,
            "subset": claimed_subset,
        },
    )
    assert r.call("kl", "a").exit == 0
    r.settle()
    (entry,) = r.host.entries
    assert entry.valid is False, "the configured reference has no complete per-sequence readings"
    assert r.host.final(generation_checks=False).choice is None


@pytest.mark.parametrize("handin_first", [False, True])
def test_a_kl_refused_build_cannot_win_by_handin(tmp_path, handin_first):
    """The identical file set handed in later, or already checked earlier, cannot carry a refused build
    to the final rules: every entry of that build's tree is ineligible."""
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    if handin_first:
        assert r.call("handin", "a").exit == 0
        r.settle()
        assert r.host.entries[-1].valid is True and r.host.entries[-1].eligible is True
    kl_effect(r, 1.2)
    assert r.call("kl", "a").exit == 0
    r.settle()
    assert r.host.entries[-1].valid is False
    if not handin_first:
        assert r.call("handin", "a").exit == 0
        r.settle()
    handed = [e for e in r.host.entries if e.kind == "handin"]
    assert handed and all(e.valid is True and e.eligible is False for e in handed), handed
    assert all("KL" in e.reason and "eligible" in e.reason for e in handed), handed
    final = r.host.final(generation_checks=False)
    assert final.choice is None
    assert final.why_not and all("eligible" in why for why in final.why_not)


@pytest.mark.parametrize("verb", ["kl", "handin"])
@pytest.mark.parametrize("value", [None, 0.0, 0.11649179775139838])
def test_baseline_bind_refusal_gives_the_bar_before_any_snapshot(tmp_path, monkeypatch, verb, value):
    from qda_agent import prompt

    base = ckpt(tmp_path / "baseline")
    rules = {"final_rules": ["handin"], "candidate0_build_s": 1} if value is not None else {}
    r = Rig(tmp_path, baselines=[base], candidate0_kl=value, **rules)
    bind = r.work / "baselines" / "0"
    r.host.sched.cfg.binds[bind] = base

    def no_snapshot(*args, **kwargs):
        raise AssertionError("a configured baseline path must refuse before copying")

    monkeypatch.setattr(r.host.sched.iso, "copy_out", no_snapshot)  # the level does the copy-out
    for path, cwd in [
        ("baselines/0", "."),
        ("./baselines/0/", "."),
        ("../baselines/0", "scripts"),
        (str(base), "."),
    ]:
        (r.work / cwd).mkdir(exist_ok=True)
        reply = r.call(verb, path, cwd=cwd)
        assert reply.exit == 2 and "a read-only baseline path, not your result" in reply.stderr
        if value is None:
            assert "Candidate 0's KL is" not in reply.stderr
        else:
            assert prompt.candidate0_kl_statement(value) in reply.stderr
    assert not r.host.entries and not r.host.sched.jobs


def test_baseline_path_names_do_not_classify_an_ordinary_candidate(tmp_path):
    base = ckpt(tmp_path / "baseline")
    r = Rig(tmp_path, baselines=[base])
    r.host.sched.cfg.binds[r.work / "baselines" / "0"] = base
    ckpt(r.work / "baselines" / "0-new", b"other")
    assert r.call("kl", "baselines/0-new").exit == 0


def test_kl_shows_only_overall_domain_and_human_kl(tmp_path):
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    r.call("kl", "a")
    r.settle()
    out = r.call("kl", "show", "k1").stdout + r.call("kl", "list").stdout
    assert "0.0712" in out and "code 0.08" in out and "math 0.065" in out and "human text 0.069" in out
    assert "chat 0.0691" in out
    for leak in (
        "/",
        "per_seq",
        "reference",
        "secret-ref",
        "checkpoint",
        "frozen",
        "serve",
        "VLLM_SECRET_FLAG",
        "marlin",
        "Marlin",
        "kernels",
        "0.33",
        "bfloat16",
        "61.5",
        "14.2",
        "88.8",
        "0.314",
        "0.271",
        "flip",
        "0.0123",
        "123456",
        "777",
        "trace",
        "npz",
        str(r.run),
    ):
        assert leak not in out
    (e,) = r.host.handins()
    assert e.kind == "scored" and e.valid and e.metrics["kl"] == pytest.approx(0.0712)


def test_handin_checks_a_frozen_copy_then_records(tmp_path):
    r = Rig(tmp_path)
    src = ckpt(r.work / "a")
    rep = r.call("handin", "a")
    assert rep.exit == 0 and "h1 a: checking" in rep.stdout
    (src / "model.safetensors").write_bytes(b"edited")
    assert r.host.handins()[0].valid is None
    r.settle()
    assert list(r.seen.values()) == [shard_bytes(tmp_path)]
    (h,) = r.host.handins()
    assert h.kind == "handin" and h.valid is True
    st = r.call("handin", "status").stdout
    assert "h1 a: valid" in st and "counts now: h1" in st
    ledger = (r.run / "judge" / "ledger.jsonl").read_text().splitlines()
    assert json.loads(ledger[-1])["valid"] is True


def test_a_handin_hard_linked_to_a_frozen_copy_is_refused_at_level_none(tmp_path, monkeypatch):
    from qda_agent.sched.snapshot import remove_tree

    r = Rig(tmp_path)
    monkeypatch.setattr(r.host.sched.iso, "level", "none")  # the agent runs as the judge's user
    ckpt(r.work / "a")
    assert r.call("kl", "a").exit == 0
    frozen = r.host.entries[0].dir
    linked = r.work / "linked"
    shutil.copytree(frozen, linked, copy_function=os.link)  # the agent's `cp -al judge/frozen/k1 linked`
    reply = r.call("handin", "linked")
    assert reply.exit == 2 and "has another hard link" in reply.stderr and "cp -al" in reply.stderr
    assert [e.id for e in r.host.entries] == ["k1"]  # nothing frozen, nothing recorded
    remove_tree(linked)
    shutil.copytree(frozen, linked)  # a copy of the bytes is its own
    assert r.call("handin", "linked").exit == 0


def test_handin_status_names_the_pick_whose_reading_came_from_a_scored_copy(tmp_path):
    """A hand-in of files the judge already scored is the pick, and its reading is that copy's.

    The status line compared whole records, so the copied reading made the pick differ from its own ledger
    entry and the command said "no submission" while FINAL.json submitted it.
    """
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    assert r.call("kl", "a").exit == 0
    assert r.call("handin", "a").exit == 0
    r.settle()
    rows = r.host.handins()
    handin = next(e for e in rows if e.kind == "handin")
    final = r.host.final(generation_checks=False)
    assert (
        final.choice.id == handin.id and final.metrics_from == "k1"
    )  # the reading came from the scored copy
    st = r.call("handin", "status").stdout
    assert f"counts now: {handin.id}" in st and "no submission" not in st


def test_handin_status_names_a_pick_that_is_not_a_handin(tmp_path):
    """The final rule can submit a scored reading (a hand-in refused for space); "no submission" would lie."""
    from qda_agent.host import Final

    r = Rig(tmp_path)
    choice = HandIn(id="k8", dir=tmp_path / "frozen" / "k8", submitted_at=datetime.now(UTC), kind="scored")
    r.host.final = lambda **kw: Final(rule="handin", choice=choice, reason="best scored build")  # noqa: ARG005
    st = r.call("handin", "status").stdout
    assert "counts now: k8 (scored entry: the final rule submitted it)" in st and "no submission" not in st


def test_handin_check_failure_reason_is_neutral(tmp_path):
    r = Rig(tmp_path)
    r.check_result = {
        "ok": False,
        "reason": "vllm could not load /data/run/judge/frozen/h1/model.safetensors",
    }
    ckpt(r.work / "a")
    r.call("handin", "a")
    r.settle()
    st = r.call("handin", "status").stdout
    assert "refused" in st and "<path>" in st and "/data" not in st


@pytest.mark.parametrize("prefix", ["", "(EngineCore_DP0 pid=4211) "])
def test_a_handin_refused_by_the_engine_names_the_loaders_own_error(tmp_path, monkeypatch, prefix):
    """vLLM's wrapper says only "See root cause above"; the cause is in the job log, and the
    hand-in's recorded reason carries it, so the agent can tell a judge-side missing file from a broken build."""
    r = Rig(tmp_path)
    r.check_result = {
        "ok": False,
        "reason": "vLLM could not serve it: RuntimeError: Engine core initialization failed. See root cause "
        "above. Failed core proc(s): {}",
    }
    log = (
        f"{prefix}Traceback (most recent call last):\n"
        f'{prefix}  File "default_loader.py", line 210, in _prepare_weights\n'
        f"{prefix}RuntimeError: Cannot find any model weights with `/data/run/judge/frozen/h1`\n"
        "(MainProcess pid=4200) INFO engine shut down\n"
    )
    monkeypatch.setattr(r.host.sched, "log", lambda name, lines=50, origin=None: log)
    ckpt(r.work / "a")
    r.call("handin", "a")
    r.settle()
    (entry,) = [e for e in r.host.entries if e.kind == "handin"]
    assert entry.valid is False
    assert entry.reason.startswith("vLLM could not serve it: RuntimeError: Engine core initialization failed")
    assert entry.reason.endswith(
        "; the job log ends: RuntimeError: Cannot find any model weights with `<path>`"
    ), entry.reason
    assert "/data" not in entry.reason and len(entry.reason) <= 300


def test_host_checks_refuse_at_once(tmp_path):
    r = Rig(tmp_path)
    bad = ckpt(r.work / "bad")
    (bad / "hack.py").write_text("import os")
    rep = r.call("handin", "bad")
    assert rep.exit == 2 and "code or pickle" in rep.stderr
    cfg = ckpt(r.work / "remote")
    (cfg / "config.json").write_text('{"auto_map": {}}')
    assert "auto_map" in r.call("kl", "remote").stderr
    assert r.call("kl", "../..").exit == 2 and "relative" in r.call("kl", "/etc").stderr
    assert r.host.handins() == [] and not any(r.be.launched)


def test_files_problem_against_anchor(tmp_path):
    anchor = ckpt(tmp_path / "anchor")
    d = ckpt(tmp_path / "d")
    assert files_problem(d, anchor) is None
    (d / "tokenizer.json").write_text('{"x": 1}')
    assert "tokenizer.json differs" in files_problem(d, anchor)
    e = ckpt(tmp_path / "e")
    (e / "model.safetensors.index.json").write_text('{"weight_map": {"a": "../x.safetensors"}}')
    assert "names" in files_problem(e)
    f = tmp_path / "f"
    f.mkdir()
    assert "safetensors" in files_problem(f)


def test_an_anchor_that_ships_remote_code_admits_a_candidate_without_it(tmp_path):
    """A release may carry `auto_map` for a type engines serve natively (NemotronH); a candidate never may."""
    anchor = ckpt(tmp_path / "anchor")
    config = json.loads((anchor / "config.json").read_text())
    (anchor / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "modeling_x.X"}}))
    assert files_problem(ckpt(tmp_path / "d"), anchor) is None
    remote = ckpt(tmp_path / "remote")
    (remote / "config.json").write_text((anchor / "config.json").read_text())
    assert "auto_map" in files_problem(remote, anchor)


def test_baseline_as_is_is_not_a_result(tmp_path):
    base = ckpt(tmp_path / "base", weights=b"base")
    r = Rig(tmp_path, baselines=[base])
    ckpt(r.work / "copy", weights=b"base")
    assert "baseline" in r.call("handin", "copy").stdout
    assert r.host.handins()[0].valid is False and not r.be.launched


#: Enough tensors that one moved element is far below the weight-space tolerance (1e-3), and few enough
#: that a test writes them at once.
NEAR_N = 200_000


def near_tensors(codes: bytes | None = None, scale: bytes | None = None) -> dict:
    """A checkpoint of `NEAR_N` one-byte codes (3) and bf16 scales (1.0), for the near-copy tests."""
    return {
        "model.embed_tokens.weight": ("BF16", (16, 8), b"\x01\x02"),
        "model.layers.0.self_attn.q_proj.weight_packed": ("U8", (NEAR_N,), codes or b"\x03" * NEAR_N),
        "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (NEAR_N,), scale or b"\x80\x3f" * NEAR_N),
    }


def test_a_one_ulp_edit_of_a_baseline_is_not_a_result(tmp_path):
    """The real run's trick: a baseline's packed codes with one scale value one bf16 ulp up."""
    r = Rig(tmp_path, baselines=[quant_dir(tmp_path / "base", tensors=near_tensors())])
    one_up = b"\x81\x3f" + b"\x80\x3f" * (NEAR_N - 1)  # 1.0 + 1 ulp, then 1.0
    quant_dir(r.work / "ulp", tensors=near_tensors(scale=one_up))
    rep = r.call("handin", "ulp")
    assert rep.exit == 0 and "packed codes" in rep.stdout
    assert r.host.handins()[0].valid is False and not r.be.launched


def test_a_baseline_with_perturbed_scales_is_not_a_result(tmp_path):
    """The same codes twice as large: the weights are re-scaled, which no rebuild does."""
    r = Rig(tmp_path, baselines=[quant_dir(tmp_path / "base", tensors=near_tensors())])
    quant_dir(r.work / "rescaled", tensors=near_tensors(scale=b"\x00\x40" * NEAR_N))  # 2.0
    rep = r.call("handin", "rescaled")
    assert rep.exit == 0 and "packed codes" in rep.stdout
    assert r.host.handins()[0].valid is False


def test_a_barely_changed_baseline_is_not_a_result(tmp_path):
    """One code of 200,000 moved one step: not identical, and still within the weight-space tolerance."""
    r = Rig(tmp_path, baselines=[quant_dir(tmp_path / "base", tensors=near_tensors())])
    quant_dir(r.work / "nudged", tensors=near_tensors(codes=b"\x04" + b"\x03" * (NEAR_N - 1)))
    rep = r.call("handin", "nudged")
    assert rep.exit == 0 and "within" in rep.stdout
    assert r.host.handins()[0].valid is False


def test_a_rebuilt_candidate_is_not_near_a_baseline(tmp_path):
    r = Rig(tmp_path, baselines=[quant_dir(tmp_path / "base", tensors=near_tensors())])
    quant_dir(r.work / "mine", tensors=near_tensors(codes=b"\x07" * NEAR_N))  # every code another
    assert "mine: checking" in r.call("handin", "mine").stdout
    r.settle()
    assert r.host.handins()[0].valid is True


def test_contamination_is_pluggable(tmp_path):
    from typing import Literal

    from qda_agent.judge.contamination import CHECKS

    class FlagAll(contamination.ContaminationCheck):  # a check from another package, by its key
        name: Literal["flag-all"] = "flag-all"

        def scanner(self):
            return lambda paths: [f"{p}: x" for p in paths]

    CHECKS.register("flag-all", FlagAll)
    r = Rig(tmp_path, contamination="flag-all")
    ckpt(r.work / "a")
    assert "contamination" in r.call("handin", "a").stdout
    assert r.host.handins()[0].valid is False
    assert contamination.scan([tmp_path], contamination.NoCheck()) == []
    with pytest.raises(
        ValueError, match="no qda_agent.contamination_checks plugin 'nope' is installed; installed: .*ngram"
    ):
        Rig(tmp_path / "b", contamination="nope")


def test_ngram_guard_must_hold_usable_hashes(tmp_path):
    """An empty, blank, placeholder or malformed guard refuses instead of reporting every file clean."""
    item = " ".join(f"word{i}" for i in range(20))
    guard = tmp_path / "guard.txt"
    for value in ("", "\n \n", "TODO\n", "g" * 64 + "\n", "a" * 63 + "\n"):
        guard.write_text(value)
        with pytest.raises(ValueError, match="contamination.guard"):
            contamination.Ngram(guard=guard).scanner()
    guard.write_text("\n".join(sorted(contamination.windows(item))) + "\n")
    check = contamination.Ngram(guard=guard).scanner()
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "copy.md").write_text(item)
    (tmp_path / "d" / "clean.md").write_text("a short unrelated text")
    assert [Path(x.split(":")[0]).name for x in check([tmp_path / "d"])] == ["copy.md"]


def test_ngram_check(tmp_path):
    item = " ".join(f"w{i}" for i in range(30))
    guard = tmp_path / "guard"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    check = contamination.Ngram(guard=guard).scanner()
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "notes.md").write_text("intro " + item.upper() + " outro")
    (tmp_path / "d" / "clean.md").write_text("w1 w2 w3")
    assert [Path(x.split(":")[0]).name for x in check([tmp_path / "d"])] == ["notes.md"]


#: A test item in a script written without spaces (Thai here; any such script exercises the character windows).
THAI_ITEM = "ร้านหนึ่งขายเค้กชิ้นละสามร้อยบาทซื้อเค้กสี่ชิ้นและขนมปังชิ้นละสองร้อยบาทสองชิ้นต้องจ่ายเงินทั้งหมดเท่าไร"


def test_scripts_without_spaces_are_covered_by_character_windows(tmp_path):
    """A Thai test item has no 20-word window, but its character windows still find a copy of it."""
    item = THAI_ITEM
    guard = tmp_path / "guard"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    check = contamination.Ngram(guard=guard).scanner()
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "copy.md").write_text("คำนำ " + item + " คำท้าย")
    (tmp_path / "d" / "clean.md").write_text("วันนี้อากาศดีมากไปเดินเล่นที่สวนกันเถอะ")
    assert [Path(x.split(":")[0]).name for x in check([tmp_path / "d"])] == ["copy.md"]


def test_a_spaced_row_does_not_hide_a_thai_row_beside_it(tmp_path):
    """The no-space share is decided per JSONL row: an English row in the same file cannot dilute it away."""
    item = THAI_ITEM
    guard = tmp_path / "guard"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    check = contamination.Ngram(guard=guard).scanner()
    (tmp_path / "d").mkdir()
    english = json.dumps({"text": " ".join(f"uniqueMarker{i}" for i in range(1000))})
    (tmp_path / "d" / "mixed.jsonl").write_text(english + "\n" + json.dumps({"text": item}) + "\n")
    assert [Path(x.split(":")[0]).name for x in check([tmp_path / "d"])] == ["mixed.jsonl"]


def test_english_padding_in_the_same_string_does_not_hide_a_no_space_item(tmp_path):
    """A no-space run inside a spaced string gets its own character windows."""
    item = THAI_ITEM
    guard = tmp_path / "guard"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    check = contamination.Ngram(guard=guard).scanner()
    english = " ".join(f"uniqueMarker{i}" for i in range(1000))
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "padded.jsonl").write_text(
        json.dumps({"text": english + " " + item}, ensure_ascii=False) + "\n"
    )
    (tmp_path / "d" / "one-line.md").write_text(english + " " + item)
    found = [Path(x.split(":")[0]).name for x in check([tmp_path / "d"])]
    assert sorted(found) == ["one-line.md", "padded.jsonl"]


@pytest.mark.parametrize("tail", ["", " ไทย"])
def test_spaced_text_keeps_exactly_its_word_windows(tail):
    """A spaced text whose no-space runs are shorter than CHARS keeps exactly its word windows."""
    text = " ".join(f"word{i}" for i in range(41)) + tail
    words = text.split()
    expected = {
        hashlib.sha256(" ".join(words[i : i + 20]).encode()).hexdigest() for i in range(len(words) - 19)
    }
    assert contamination.windows(text) == expected


def test_deadline_refuses_and_cleans_up(tmp_path):
    r = Rig(tmp_path, budget_s=30, grace_s=20)  # a reading takes 60 s: past the deadline + drain window
    ckpt(r.work / "a")
    rep = r.call("kl", "a")
    assert rep.exit == 2 and "drain window" in rep.stderr
    assert r.host.handins() == [] and list((r.run / "judge" / "frozen").iterdir()) == []


def test_cutoff_drain_final(tmp_path):
    r = Rig(tmp_path)
    ckpt(r.work / "a")
    ckpt(r.work / "b", weights=b"w2")
    (r.work / "s.sh").write_text("sleep 1\n")
    assert r.call("qsched", "add", "--expected", "30m", "--name", "long", "s.sh").exit == 0
    r.call("handin", "a")
    r.settle()
    r.call("handin", "b")  # its check is still pending at the cutoff
    r.call("qsched", "add", "--expected", "5", "--name", "mine", "s.sh")
    r.host.cutoff()
    assert r.host.sched.get("mine").state is State.cancelled
    assert r.call("handin", "a").exit == 2
    assert r.host.drain(600)
    final = r.host.final()
    assert final.rule == "handin" and final.choice.dir.name == "h2"
    r.host.write_final(r.run / "FINAL.json")
    assert json.loads((r.run / "FINAL.json").read_text())["rule"] == "handin"


def test_qda_build_is_queued_other_subcommands_run(tmp_path):
    r = Rig(tmp_path, qda_cmd=[PY, "-c", "import sys; print(sys.argv[1:])"])
    (r.work / "recipe.yaml").write_text("steps: v1\n")
    rep = r.call("qda", "--cards", "2", "--expected", "20m", "build", "recipe.yaml", "--out", "o")
    assert rep.exit == 0 and "queued build" in rep.stdout
    (r.work / "recipe.yaml").write_text("steps: v2\n")
    (job,) = r.host.sched.list(origin="agent")
    assert job.ncards == 2 and job.kind.value == "build" and job.expected_s == 1200
    assert job.argv[-4] == "build" and job.argv[-2:] == ["--out", "o"]
    assert Path(job.argv[-3]).read_text() == "steps: v1\n"
    assert r.call("qda", "build", "../x.yaml").exit == 2 and r.call("qda", "build").exit == 2
    rep = r.call("qda", "plan", "x")
    assert rep.exit == 0 and rep.stdout.strip() == "['plan', 'x']"


def test_qsched_verbs(tmp_path):
    r = Rig(tmp_path)
    (r.work / "s.sh").write_text("echo 1\n")
    assert r.call("qsched", "add", "--expected", "90s", "--after", "", "s.sh").exit == 0
    assert "required" in r.call("qsched", "add", "s.sh").stderr  # --expected is required
    assert r.call("qsched", "add", "--expected", "1m", "s.sh", "extra").exit == 2  # one script, no argv
    assert r.call("qsched", "add", "--expected", "1m", "--", "bash", "-c", "x").exit == 2
    assert "outside" in r.call("qsched", "add", "--expected", "1m", "../s.sh").stderr
    assert r.call("qsched", "--help").exit == 0
    r.call("kl", "nope")
    r.host.tick()
    tab = r.call("qsched", "list").stdout
    assert "job1" in tab and "running" in tab and "judge jobs pending" in tab
    assert r.call("qsched", "cancel", "job1").stdout == "job1: cancelled\n"
    assert r.call("qsched", "log", "judge-k1").exit == 2


# ---------------------------------------------------------------------- final rules
def pick(
    ledger,
    baselines,
    metric="kl",
    rules=("last_valid", "best_metric", "baseline"),
    candidate0_kl=None,
    disk_refusals=(),
):
    settings = JudgeSettings(final_rules=list(rules), candidate0_kl=candidate0_kl)
    return apply(settings.final_rules, ledger, baselines, metric, candidate0_kl, disk_refusals=disk_refusals)


def h(name, kind="handin", valid=True, kl=None, t=0, lst=None, slot=0):
    read = {} if kl is None else {"kl": kl}
    return HandIn(
        dir=Path(name),
        submitted_at=datetime.fromtimestamp(t, UTC),
        kind=kind,
        valid=valid,
        metrics=read,
        higher_is_better=dict.fromkeys(read, False),
        handin_list=lst,
        slot=slot,
    )


def test_fallback_rule():
    base = [Path("b1"), Path("b2")]
    led = [h("h1"), h("k1", "scored", kl=0.05), h("h2", valid=False), h("h3", valid=None)]
    f = pick(led, base)
    assert f.rule == "last_valid" and f.choice.dir.name == "h1" and len(f.why_not) == 2
    assert f.reason == "the last valid submission"
    led = [
        h("h1", valid=False),
        h("k1", "scored", kl=0.07),
        h("k2", "scored", kl=0.05),
        h("k3", "scored", False),
    ]
    f = pick(led, base)
    assert (
        f.rule == "best_metric" and f.choice.dir.name == "k2" and f.reason.startswith("no valid submission;")
    )
    f = pick([h("h1", valid=False), h("k1", "scored", valid=False, kl=0.01)], base)
    assert f.rule == "baseline" and f.baseline == Path("b1") and f.choice is None
    assert f.reason == "nothing valid was submitted or scored; the first baseline as-is"
    assert pick([], []).baseline is None
    assert pick([], []).reason == "nothing valid was submitted or scored, and no baseline is configured"


def test_the_rule_ranks_by_the_named_metric_in_the_readings_direction():
    def read(name, size, higher):
        return HandIn(
            dir=Path(name),
            submitted_at=datetime.fromtimestamp(0, UTC),
            kind="scored",
            valid=True,
            metrics={"kl": 1 / size, "size": size},
            higher_is_better={"kl": False, "size": higher},
        )

    def best(metric, higher):
        return pick([read("small", 1.0, higher), read("big", 2.0, higher)], [], metric=metric)

    assert best("kl", True).choice.dir.name == "big"  # each metric keeps its own direction
    assert best("size", True).choice.dir.name == "big" and "highest size (2)" in best("size", True).reason
    assert best("size", False).choice.dir.name == "small" and "lowest size (1)" in best("size", False).reason
    assert best("absent", True).rule == "baseline"


def test_the_handin_rule_takes_the_latest_valid_handin():
    """The agent chooses: the latest list's first valid entry wins, whatever the judge scored elsewhere."""
    base = [Path("b1"), Path("b2")]
    # the latest hand-in wins even with a higher KL than an earlier one
    f = pick([h("h1", kl=0.01), h("h2", kl=0.9)], base, rules=["handin"])
    assert f.rule == "handin" and f.choice.dir.name == "h2" and f.baseline is None
    assert f.chosen_by == "handin"
    # a scored checkpoint with a lower KL that was never handed in does not win
    f = pick([h("h1", kl=0.5), h("k1", "scored", kl=0.01)], base, rules=["handin"])
    assert f.rule == "handin" and f.choice.dir.name == "h1" and f.chosen_by == "handin"
    # an invalid last hand-in does not count: the previous valid one does
    f = pick([h("h1", kl=0.5), h("h2", valid=False)], base, rules=["handin"])
    assert f.choice.dir.name == "h1"
    # nor does an ineligible one
    ineligible = h("h2", kl=0.9)
    ineligible.eligible = False
    f = pick([h("h1", kl=0.5), ineligible], base, rules=["handin"])
    assert f.choice.dir.name == "h1"
    # a rule that is not the hand-in contract states no path
    assert pick([h("h1")], base, rules=["last_valid"]).chosen_by is None


def test_a_handin_picked_after_every_reading_failed_is_said_in_final_json():
    """Every KL reading failed and FINAL.json picked an unscored hand-in, reading as a normal result."""
    failed = [h("k1", "scored", valid=False), h("k2", "scored", valid=False)]
    f = pick([*failed, h("h1")], [Path("b1")], rules=["handin"])
    assert f.choice.dir.name == "h1" and f.chosen_by == "handin"  # the rule is unchanged
    assert f.warnings == ["no reading of this run succeeded (2 attempted; the ledger has each reason)"]
    assert json.loads(f.model_dump_json())["warnings"] == f.warnings
    # one reading that read a KL, a scored pick, or no reading asked for: no warning, and no key in FINAL.json
    for ledger in ([*failed, h("k3", "scored", kl=0.1), h("h1")], [*failed, h("h1", kl=0.1)], [h("h1")]):
        f = pick(ledger, [Path("b1")], rules=["handin"])
        assert f.warnings == [] and "warnings" not in json.loads(f.model_dump_json())


def test_a_picked_handin_takes_the_reading_of_the_same_files_or_says_it_has_none(tmp_path):
    """The agent handed in k1's files again as h2; FINAL.json showed h2 with no KL, though k1 was read."""
    from qda_agent.judge.ledger import Entry

    def entry(name, kind, digest, kl=None, valid=True):
        read = {} if kl is None else {"kl": kl}
        return Entry(
            id=name,
            kind=kind,
            src=name,
            dir=Path(name),
            digest=digest,
            submitted=0,
            valid=valid,
            metrics=read,
            higher_is_better=dict.fromkeys(read, False),
        )

    host = SimpleNamespace(
        cfg=SimpleNamespace(metric="kl", baselines=[], candidate0_kl=None),
        lock=threading.Lock(),
        run_dir=tmp_path,
    )
    host.entries = [
        entry("k0", "scored", "w0", 0.09),
        entry("k1", "scored", "w1", 0.0401),
        entry("h2", "handin", "w1"),
    ]
    final = apply(
        JudgeSettings(final_rules=["handin"]).final_rules, [e.record() for e in host.entries], [], "kl"
    )
    got = JudgeHost._choice_reading(host, final)
    assert got.choice.metrics == {"kl": 0.0401} and got.metrics_from == "k1" and got.warnings == []
    assert final.choice.metrics == {}  # the rule's own record is not changed
    by_id = {e.id: e for e in got.ledger}
    # the pick is its ledger entry by id, though its reading is the scored copy's (#): the one rule the
    # status line and END-REPORT read
    assert got.is_pick(by_id["h2"]) and not got.is_pick(by_id["k1"])
    host.entries[1].digest = "other"  # no reading of these files
    got = JudgeHost._choice_reading(host, final)
    assert got.choice.metrics == {} and got.metrics_from is None
    assert got.warnings == [
        "the delivered build h2 has no KL reading (candidate 0: no reading, best scored build: 0.0401); "
        f"`qda eval {tmp_path}` scores it on the holdout prompts"
    ]


def hand_in_host(tmp_path, scored, c0=0.1929):
    """A judge stand-in with candidate 0 (reading `c0`) and `scored` as its ledger; no build.json: a path."""
    from qda_agent.judge.ledger import Entry

    entries = [
        Entry(
            id=name,
            kind=kind,
            src=name,
            dir=Path(name),
            digest=name,
            submitted=0,
            valid=True,
            metrics={} if kl is None else {"kl": kl},
            higher_is_better={} if kl is None else {"kl": False},
        )
        for name, kind, kl in scored
    ]
    cand0 = tmp_path / "cand0"
    cand0.mkdir(exist_ok=True)
    cfg = SimpleNamespace(metric="kl", baselines=[cand0], candidate0_kl=c0)
    return SimpleNamespace(cfg=cfg, lock=threading.Lock(), run_dir=tmp_path, entries=entries), cand0


def warned(host):
    ledger = [e.record() for e in host.entries]
    final = apply(JudgeSettings(final_rules=["handin"]).final_rules, ledger, list(host.cfg.baselines), "kl")
    return JudgeHost._choice_reading(host, final)


def test_the_hand_in_warns_when_it_has_no_reading_and_says_how_to_deliver_candidate_0(tmp_path):
    """The real run: the pick had no KL reading, both scored builds (0.2000, 0.2775) lost to candidate 0."""
    host, cand0 = hand_in_host(
        tmp_path, [("k1", "scored", 0.2), ("k2", "scored", 0.2775), ("asymawq", "handin", None)]
    )
    got = warned(host)
    assert got.choice.dir.name == "asymawq" and got.chosen_by == "handin"  # the rule's pick is unchanged
    assert got.warnings == [
        "the delivered build asymawq has no KL reading (candidate 0: 0.1929, best scored build: 0.2000); "
        "every scored build is worse than candidate 0; "
        f"`qda eval {tmp_path}` scores it on the holdout prompts; to deliver candidate 0 instead: use the checkpoint at {cand0}"
    ]
    assert json.loads(got.model_dump_json())["warnings"] == got.warnings
    # a scored build that beat candidate 0 leaves only the missing reading to say
    host, _ = hand_in_host(tmp_path, [("k1", "scored", 0.15), ("h1", "handin", None)])
    only = warned(host).warnings[0]
    assert "has no KL reading (candidate 0: 0.1929, best scored build: 0.1500); `qda eval" in only
    assert "worse than candidate 0" not in only


def test_the_hand_in_warns_when_every_reading_is_worse_than_candidate_0(tmp_path):
    host, cand0 = hand_in_host(
        tmp_path, [("k1", "scored", 0.2775), ("h1", "handin", 0.2), ("k2", "scored", 0.31)]
    )
    got = warned(host)
    assert got.choice.dir.name == "h1"
    assert got.warnings == [
        "every build with a KL reading is worse than candidate 0 (h1: 0.2000, best scored build: 0.2000, "
        f"candidate 0: 0.1929); to deliver candidate 0 instead: use the checkpoint at {cand0}"
    ]


def test_a_malformed_candidate_0_build_json_changes_neither_the_pick_nor_stderr(tmp_path, capsys):
    """`final()` runs on status and prune passes too: the serve line is only built when a warning is emitted."""
    host, cand0 = hand_in_host(tmp_path, [("k1", "scored", 0.2775), ("h1", "handin", 0.15)])
    (cand0 / "build.json").write_text("{not json")
    got = warned(host)  # no warning applies: candidate 0's build.json is never read
    assert got.choice.dir.name == "h1" and got.warnings == []
    assert capsys.readouterr().err == ""
    host, cand0 = hand_in_host(tmp_path, [("k1", "scored", 0.2), ("h1", "handin", None)])
    (cand0 / "build.json").write_text("{not json")
    got = warned(host)  # a warning applies: the JSON error falls back to the path, the pick is unchanged
    assert got.choice.dir.name == "h1" and got.chosen_by == "handin"
    assert got.warnings[0].endswith(f"to deliver candidate 0 instead: use the checkpoint at {cand0}")


def test_the_hand_in_does_not_warn_when_candidate_0_was_delivered(tmp_path):
    """No hand-in (`choice=None`): the baseline is delivered, so there is nothing to warn about."""
    host, _ = hand_in_host(tmp_path, [("k1", "scored", 0.2775)])
    got = warned(host)
    assert got.choice is None and got.warnings == []


def test_the_hand_in_does_not_warn_when_it_is_read_and_beats_candidate_0(tmp_path):
    host, _ = hand_in_host(tmp_path, [("k1", "scored", 0.2775), ("h1", "handin", 0.15)])
    got = warned(host)
    assert got.choice.dir.name == "h1" and got.warnings == []
    assert "warnings" not in json.loads(got.model_dump_json())
    # the reading of the same files counts as the pick's own: no warning for a hand-in of a better build
    host, _ = hand_in_host(tmp_path, [("k1", "scored", 0.15), ("h1", "handin", None)])
    host.entries[0].digest = host.entries[1].digest = "same"
    assert warned(host).warnings == []


def test_delivery_decides_the_two_facts_and_unmeasured_names_the_missing_one():
    """One decision from the final record, recorded in FINAL.json and printed by both readers."""

    def check(*, choice=Path("ckpt"), baseline=None, kl=None, generation_check=None):
        return delivery(None if choice is None else h("ckpt", kl=kl), baseline, generation_check, "kl")

    # the search measured neither of the delivered hand-in: both facts are false
    assert check(generation_check={}) == (False, False)
    # one fact is there, the other is not
    assert check(kl=0.01, generation_check={}) == (True, False)
    assert check(generation_check={"id": "g1", "valid": True}) == (False, True)
    # both measured: the agent's `gencheck` entry or a generation rule's own evidence
    assert check(kl=0.01, generation_check={"id": "g1", "valid": True}) == (True, True)
    assert check(kl=0.01, generation_check={"outcome": "passed"}) == (True, True)
    # the judge's own note that the pick has no valid check
    assert check(kl=0.01, generation_check={"outcome": "none: none"}) == (True, False)
    # a baseline kept as-is carries candidate 0's KL; only the check can be missing
    assert check(choice=None, baseline=Path("b1"), generation_check={}) == (True, False)
    # nothing delivered (a trust refusal): nothing decided or named
    assert check(choice=None, baseline=None) == (None, None)
    assert unmeasured(None, None, "kl") == ""

    # the one sentence both readers use: only the missing facts are named
    assert unmeasured(False, False, "kl") == (
        "The delivered checkpoint has no KL reading and no long-generation check (`gencheck`)."
    )
    assert unmeasured(True, False, "kl") == (
        "The delivered checkpoint has no long-generation check (`gencheck`)."
    )
    assert unmeasured(False, True, "kl") == "The delivered checkpoint has no KL reading."
    assert unmeasured(True, True, "kl") == ""


def test_a_rotated_checkpoint_names_rotate_draft_in_the_hint(tmp_path):
    """the hint is read from the checkpoint's own resolved.yaml, the field `qda rotate-draft` reads too."""
    from qda_agent.host import rotation_hint

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    assert rotation_hint(ckpt) == ""  # no record: nothing claimed
    record = {"steps": [{"rotate": {"residual": True, "seed": 3}}]}
    (ckpt / "resolved.yaml").write_text(json.dumps(record))
    line = rotation_hint(ckpt)
    assert "DFlash / DSpark" in line and f"qda rotate-draft --checkpoint {ckpt}" in line
    (ckpt / "resolved.yaml").write_text(json.dumps({"steps": [{"rotate": {"residual": False}}]}))
    assert rotation_hint(ckpt) == ""  # value heads only: a draft runs as it is
    assert rotation_hint(None) == ""


def test_a_handin_list_counts_its_first_valid_entry_and_falls_back_to_an_earlier_list():
    """One call is one ordered list: the first entry that passed, else the latest earlier list that has one."""
    base = [Path("b1")]
    # list 1: a, b; list 2: c (refused), d (valid) -> d
    led = [
        h("a", lst=1, slot=0),
        h("b", lst=1, slot=1),
        h("d", lst=2, slot=1),
        h("c", lst=2, slot=0, valid=False),
    ]
    f = pick(led, base, rules=["handin"])
    assert f.choice is not None and f.choice.dir.name == "d"
    assert f.chosen_by == "handin" and "entry 2" in f.reason
    # list 2 has no valid entry at all: list 1's first valid entry counts
    led = [h("a", lst=1, slot=0), h("b", lst=1, slot=1), h("c", lst=2, slot=0, valid=False)]
    f = pick(led, base, rules=["handin"])
    assert f.choice is not None and f.choice.dir.name == "a"
    # a single-directory call is a list of one and replaces the earlier list
    led = [h("a", lst=1, slot=0), h("b", lst=2, slot=0)]
    f = pick(led, base, rules=["handin"])
    assert f.choice is not None and f.choice.dir.name == "b"


def test_the_fallback_is_candidate_0_as_is():
    """No valid hand-in: candidate 0 as-is, as the paper and main's rule do; FINAL says which path decided."""
    base = [Path("b1"), Path("b2")]
    led = [h("k1", "scored", kl=0.05), h("k2", "scored", kl=0.01), h("h2", valid=False)]
    f = pick(led, base, rules=["handin"])
    assert f.choice is None and f.baseline == Path("b1") and f.rule == "handin"
    assert f.chosen_by == "fallback_candidate0"
    # nothing at all: candidate 0 as-is (or the decision that none is configured)
    f = pick([], [], rules=["handin"])
    assert f.baseline is None and f.chosen_by == "fallback_candidate0"


def test_a_handin_refused_for_disk_space_submits_the_best_scored_build(tmp_path, capsys):
    """A full disk is not a verdict on the checkpoint, so it never decides the result.

    k8 is the best scored build and the agent handed it in; the run is over its disk cap, so the hand-in is
    refused for space while k1 (the worse build, handed in as h1) passed. The result is k8, END-REPORT's
    reason (FINAL.json's `reason`, written verbatim) says why, and the ledger records the refusal.
    """
    from qda_agent.judge.ledger import Entry

    r = Rig(tmp_path)
    ckpt(r.work / "out" / "k1", b"w1")
    ckpt(r.work / "out" / "k8", b"w8")

    def entry(name, kind, digest, kl=None):
        read = {} if kl is None else {"kl": kl}
        frozen = ckpt(r.host.jdir / "frozen" / name, digest.encode())  # a whole copy, as a run holds
        return Entry(
            id=name,
            kind=kind,
            src=f"out/{name}",
            dir=frozen,
            digest=digest,
            submitted=r.clock.now(),
            ended=r.clock.now(),
            valid=True,
            metrics=read,
            higher_is_better=dict.fromkeys(read, False),
        )

    # k8 read best; h1 handed in k1's files, which read far worse
    r.host.entries += [
        entry("k1", "scored", "d1", 0.9),
        entry("k8", "scored", "d8", 0.05),
        entry("h1", "handin", "d1"),
    ]
    r.host.cfg.max_run_bytes = 1  # over the cap: the hand-in of k8 cannot be copied into the run
    reply = r.call("handin", "out/k8")
    assert reply.exit == 2 and "disk budget" in reply.stderr
    (record,) = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
    assert record["kind"] == "space_refusal" and record["build"] == "out/k8" and record["cap"] == 1
    assert "judge: hand-in of out/k8 refused for disk space" in capsys.readouterr().err

    final = r.host.final()
    assert final.rule == "handin" and final.choice is not None and final.choice.id == "k8"
    assert final.chosen_by == "handin"
    assert final.reason == (
        "hand-in of out/k8 refused for disk space; submitted the best scored eligible build k8 (KL 0.05) instead"
    )

    # a host resumed over this run rebuilds the refusal from the ledger: the rule still submits k8
    r.host.space_refusals.clear()
    r.host.restore_space_refusals()
    resumed = r.host.final()
    assert resumed.choice is not None and resumed.choice.id == "k8"

    # without the space refusal the same ledger keeps the agent's contract: h1 counts, though k8 read better
    r.host.space_refusals.clear()  # the check refuses nothing now; the ledger's record stays in the file
    kept = r.host.final()
    assert kept.choice is not None and kept.choice.id == "h1"


def test_a_ledger_reads_its_entries_and_its_records_past_a_torn_line(tmp_path):
    """A resume reads `ledger.jsonl`: the entries (newest snapshot each) and the records beside them.

    A record line states a `kind` no entry has, so it is skipped rather than parsed as an `Entry`; a torn
    line (a crashed append) is skipped too, and a resumed host still gets its space refusals.
    """
    from qda_agent.judge import ledger as records
    from qda_agent.judge.ledger import Entry

    def entry(name, kl=None):
        read = {} if kl is None else {"kl": kl}
        return Entry(
            id=name,
            kind="scored",
            src=name,
            dir=Path(name),
            digest="d",
            submitted=0,
            valid=True,
            metrics=read,
            higher_is_better=dict.fromkeys(read, False),
        )

    refusal = {"kind": "space_refusal", "build": "out/k8", "held": 10, "cap": 5, "at": 0}
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(doc)
            for doc in (
                entry("k1", 0.9).model_dump(mode="json"),
                entry("k1", 0.8).model_dump(mode="json"),  # a newer snapshot of the same entry
                refusal,
                {"kind": "prune", "destination": "b1", "bytes": 3},  # another record, never an Entry
            )
        )
        + '\n{"kind": "scored", "id": "tr'  # a torn append
    )
    got = records.entries(path)
    assert [(e.id, e.metrics) for e in got] == [("k1", {"kl": 0.8})]  # the newest, once
    assert records.space_refusals(path) == [refusal]
    assert records.space_refusals(tmp_path / "missing.jsonl") == []  # a fresh run has no ledger


def test_a_space_refusal_hands_back_to_candidate_0_when_its_reading_is_best():
    """Candidate 0 competes in the space refusal's pick (`Standing.valid` never holds the baselines)."""
    led = [h("k8", "scored", kl=0.05), h("h1")]
    refusal = ({"kind": "space_refusal", "build": "out/k8"},)
    base = [Path("b1")]
    # candidate 0 reads best: it is the result, as-is, and FINAL says why
    f = pick(led, base, rules=["handin"], candidate0_kl=0.02, disk_refusals=refusal)
    assert f.choice is None and f.baseline == Path("b1") and f.rule == "handin"
    assert f.chosen_by == "fallback_candidate0"
    assert f.reason == (
        "hand-in of out/k8 refused for disk space; no build beat candidate 0 (KL 0.02); candidate 0 as-is"
    )
    # a tie does not beat candidate 0 either
    f = pick(led, base, rules=["handin"], candidate0_kl=0.05, disk_refusals=refusal)
    assert f.choice is None and f.baseline == Path("b1")
    # a build that beats it is submitted
    f = pick(led, base, rules=["handin"], candidate0_kl=0.09, disk_refusals=refusal)
    assert f.choice is not None and f.choice.dir.name == "k8"
    # no candidate 0 reading: the best build counts
    f = pick(led, base, rules=["handin"], disk_refusals=refusal)
    assert f.choice is not None and f.choice.dir.name == "k8"


def test_a_new_run_defaults_to_the_handin_rule():
    assert [rule.name for rule in JudgeSettings().final_rules] == ["handin"]
    text = statement(JudgeSettings().final_rules, "kl")
    assert "the first directory of your latest hand-in list that has passed counts" in text
    assert "candidate 0 as-is" in text
    assert final_handin_statement(JudgeSettings().final_rules) == ""
    assert final_handin_statement(JudgeSettings(final_rules=["best_metric"]).final_rules) == ""


def test_the_config_orders_the_rules_and_each_reason_stays_true():
    led = [h("h1"), h("k1", "scored", kl=0.07), h("k2", "scored", kl=0.05)]
    f = pick(led, [Path("b1")], "kl", ["best_metric", "last_valid"])
    assert f.rule == "best_metric" and f.choice.dir.name == "k2"
    assert f.reason.startswith("the valid scored")  # a valid submission exists: not "no valid submission"
    f = pick(led, [Path("b1")], "kl", ["baseline", "last_valid"])
    assert (
        f.rule == "baseline" and f.reason == "no earlier rule picked a checkpoint; the first baseline as-is"
    )
    f = pick([h("k1", "scored", kl=0.07)], [Path("b1")], "kl", ["last_valid"])
    assert f.rule is None and f.choice is None and f.baseline is None  # no rule decided: nothing added


def test_the_statement_follows_the_rules():
    rules = JudgeSettings().final_rules
    assert statement(rules, "kl") == (
        "Then the first directory of your latest hand-in list that has passed counts; a check not finished "
        "by then does not; with none, the first that passed of the latest earlier list that has one; with "
        "none, candidate 0 as-is."
    )
    assert statement(JudgeSettings(final_rules=["best_metric"]).final_rules, "size") == (
        "Then the valid checkpoint you scored with `kl` that has the best size counts."
    )


# ---------------------------------------------------------------------- end to end through the shim
def test_commands_through_the_socket(tmp_path):
    run = tmp_path / "run"
    work = run / "work"
    ckpt(work / "ck")
    host = make_host(
        run,
        {"cards": [0, 1], "budget_s": 3600, "tick_s": 0.05},
        {
            "allow_any_format": True,
            "kl_cmd": [
                sys.executable,
                "-c",
                f"import json,sys; open(sys.argv[2],'w').write({json.dumps(KL_RESULT)!r})",
                "{ckpt}",
                "{out}",
            ],
            "kl_score_s": 60,
            "evaluator": str(evaluator(tmp_path)),
            "qda_cmd": [PY],
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
        },
    )
    bindir = work / "bin"
    install_bin(bindir, run / "rpc.sock")
    host.start()
    try:

        def cmd(name, *args, cwd=work):
            p = subprocess.run(
                [sys.executable, str(bindir / name), *args], cwd=cwd, capture_output=True, text=True
            )
            return p.returncode, p.stdout, p.stderr

        assert cmd("kl", "ck")[0] == 0
        assert host.drain(10)
        rc, out, _ = cmd("kl", "show", "k1", cwd=work / "ck")
        assert rc == 0 and "KL 0.0712" in out and "/" not in out
        (work / "hi.sh").write_text("echo hi\n")
        rc, out, _ = cmd("qsched", "add", "--expected", "1m", "hi.sh")
        assert rc == 0
        assert cmd("qsched", "list")[0] == 0
        rc, _, err = cmd("kl", "ck", cwd=tmp_path)
        assert rc == 2 and "inside the work directory" in err
    finally:
        host.stop()
    assert not (run / "rpc.sock").exists()


def test_symlinks_are_refused_not_followed(tmp_path):
    r = Rig(tmp_path)
    d = ckpt(r.work / "a")
    (d / "loop").symlink_to(tmp_path)  # a link out of the workdir (and a loop back into it)
    rep = r.call("handin", "a")
    assert rep.exit == 2 and "loop is not a regular file" in rep.stderr
    (d / "loop").unlink()
    (d / "model.safetensors").unlink()
    (d / "model.safetensors").symlink_to(tmp_path / "elsewhere.safetensors")
    assert r.call("kl", "a").exit == 2


def test_handin_without_serving_check(tmp_path):
    r = Rig(tmp_path, check=False)
    ckpt(r.work / "a")
    assert "h1 a: valid" in r.call("handin", "a").stdout and not r.be.launched
    r.host.cutoff()
    assert "deadline" in r.call("handin", "a").stderr
    assert r.host.final().rule == "handin"


def test_evaluator_links_are_refused(tmp_path):
    """A link is copied as a link but left out of the tree digest: scoring could read bytes the record never
    hashed. Regular files still freeze; a tree with a link refuses before any copy is made."""
    ev = evaluator(tmp_path)
    cfg = {"evaluator": str(ev), "kl_cmd": ["score", "{out}"], "qda_cmd": [PY]}
    ok = make_host(tmp_path / "ok", {"cards": [0], "fake_gpus": True}, cfg, autotick=False)
    assert (startup.install_evaluator(ok) / "__init__.py").read_text() == "VERSION = 'v1'\n"
    live = tmp_path / "live.py"
    live.write_text("VALUE = 1\n")
    (ev / "__init__.py").unlink()
    (ev / "__init__.py").symlink_to(live)
    # the source and its link check are `static_checks`', so the start refuses before it creates anything
    with pytest.raises(ValueError, match="__init__.py is a symbolic link"):
        make_host(tmp_path / "bad", {"cards": [0], "fake_gpus": True}, cfg, autotick=False)
    assert not (tmp_path / "bad" / "judge" / "eval").exists()


def test_a_link_that_appears_during_the_copy_is_refused(tmp_path, monkeypatch):
    """Checking only the source before copying leaves a window: `freeze` checks the copy it seals."""
    ev = evaluator(tmp_path)
    live = tmp_path / "live.py"
    live.write_text("VALUE = 1\n")
    real = startup.freeze

    def racing(src, dst):
        (Path(src) / "late.py").symlink_to(live)  # after the source check, before the copy
        return real(src, dst)

    monkeypatch.setattr(startup, "freeze", racing)
    host = make_host(
        tmp_path / "run",
        {"cards": [0], "fake_gpus": True},
        {"evaluator": str(ev), "kl_cmd": ["score", "{out}"], "qda_cmd": [PY]},
        autotick=False,
    )
    with pytest.raises(ValueError, match="late.py is a symbolic link in the frozen copy"):
        host.start()


def test_evaluator_runs_from_a_read_only_copy(tmp_path):
    ev = evaluator(tmp_path)
    run = tmp_path / "run"
    probe = (
        "import json, sys, fake_eval; "
        "open(sys.argv[2], 'w').write(json.dumps({'overall': 0.5, 'where': fake_eval.__file__, 'v': fake_eval.VERSION}))"
    )
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "tick_s": 0.05},
        {
            "allow_any_format": True,
            "kl_cmd": [sys.executable, "-c", probe, "{ckpt}", "{out}"],
            "evaluator": str(ev),
            "socket": str(run / "work" / ".qda" / "rpc.sock"),
            "qda_cmd": [PY],
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
        },
    )
    (run / "work" / ".qda").mkdir(parents=True)
    host.start()
    try:
        assert (run / "work" / ".qda" / "rpc.sock").exists() and not (run / "rpc.sock").exists()
        (ev / "__init__.py").write_text("VERSION = 'edited'\n")  # the live tree changes after start
        ckpt(run / "work" / "a")
        assert host.handle(Request("kl", ["a"], str(run / "work"))).exit == 0
        assert host.drain(10)
        got = json.loads((run / "judge" / "results" / "k1.json").read_text())
        assert got["v"] == "v1" and Path(got["where"]).is_relative_to(run / "judge" / "eval")
        rec = json.loads((run / "judge" / "judge.json").read_text())
        copy = Path(rec["copy"])
        assert len(rec["sha256"]) == 64 and copy == run / "judge" / "eval" / "fake_eval"
        assert stat.S_IMODE((copy / "__init__.py").stat().st_mode) & 0o222 == 0
    finally:
        host.stop()
    with pytest.raises(ValueError, match="used by an earlier host"):
        make_host(run, {"cards": [0]}, {"evaluator": str(ev), "kl_ref": "ref"})


def test_evaluator_default_needs_qda_kl(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="qda_kl is not installed"):
        make_host(
            tmp_path / "run",
            {"cards": [0], "fake_gpus": True},
            {
                "kl_ref": str(_ref_dir(tmp_path)),
                "kl_holdout_ref": str(_ref_dir(tmp_path)),
                "allow_any_format": True,
            },
            autotick=False,
        )


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_subcommands_run_sandboxed(tmp_path):
    r = Rig(tmp_path, qda_cmd=[PY, "-c", "import sys; print(open(sys.argv[1]).read())"])
    (r.run / "judge" / "secret.txt").write_text("judge-secret")
    (r.work / "mine.txt").write_text("mine")
    assert r.call("qda", "mine.txt").stdout.strip() == "mine"
    rep = r.call("qda", str(r.run / "judge" / "secret.txt"))
    assert rep.exit != 0 and "judge-secret" not in rep.stdout


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_subcommands_read_a_hugging_face_snapshot(tmp_path):
    """A hub snapshot's files are links into its blobs and the cache's shared store: the verb reads them
    through both hops, as a build job does, and a link out of the repository is refused."""
    hub = tmp_path / "hub"
    repo = hub / "models--X"
    (repo / "snapshots" / "rev").mkdir(parents=True)
    (repo / "blobs").mkdir()
    (hub / "blobs" / "12").mkdir(parents=True)
    (hub / "blobs" / "12" / "abc").write_text("model-config")
    (repo / "blobs" / "abc").symlink_to("../../blobs/12/abc")  # the repository's entry into the store
    snap = repo / "snapshots" / "rev"
    (snap / "config.json").symlink_to("../../blobs/abc")
    r = Rig(tmp_path, readable=[snap], qda_cmd=[PY, "-c", "import sys; print(open(sys.argv[1]).read())"])
    rep = r.call("qda", str(snap / "config.json"))
    assert rep.exit == 0 and rep.stdout.strip() == "model-config", rep
    (snap / "escape.json").symlink_to("../../../../escape.txt")  # outside the model repository
    rep = r.call("qda", str(snap / "escape.json"))
    assert rep.exit == 1 and "refusing to bind it" in rep.stderr, rep


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_subcommands_honour_a_hidden_path(tmp_path):
    pool = tmp_path / "pool"  # a readable source, as the calibration pool or a package tree is
    (pool / "data").mkdir(parents=True)
    (pool / "data" / "gencheck-prompts.jsonl").write_text("private")
    (pool / "public.txt").write_text("public")
    r = Rig(
        tmp_path,
        readable=[pool],
        hide=[pool / "data"],
        qda_cmd=[PY, "-c", "import sys; print(open(sys.argv[1]).read())"],
    )
    rep = r.call("qda", str(pool / "data" / "gencheck-prompts.jsonl"))
    assert rep.exit != 0 and "private" not in rep.stdout
    assert r.call("qda", str(pool / "public.txt")).stdout.strip() == "public"
    assert (pool / "data" / "gencheck-prompts.jsonl").read_text() == "private"  # the host still sees it


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_subcommands_refuse_a_hide_covering_what_they_need(tmp_path):
    r = Rig(tmp_path, qda_cmd=[PY, "-c", "print('ran')"])
    r.host.sched.cfg.hide.append(r.work)
    rep = r.call("qda", "x")
    assert rep.exit == 1 and "ran" not in rep.stdout
    assert "sandbox.hide" in rep.stderr and str(r.work) in rep.stderr


def test_refused_reading_with_a_stale_file_is_no_score(tmp_path):
    clock = FakeClock()

    def effect(argv):  # writes a result file, then exits 2 (a refusal): the file must not count
        out = Path(json.loads(Path(argv[argv.index("score") + 1]).read_text())["out"])
        out.mkdir(parents=True)
        (out / "score.json").write_text(json.dumps(KL_RESULT))

    be = FakeBackend(clock, default=FakeRun(10, rc=2, effect=effect))
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {"kl_ref": str(_ref_dir(tmp_path)), "kl_holdout_ref": str(_ref_dir(tmp_path))},
        clock,
        backend=be,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    ckpt(run / "work" / "b", weights=b"w2")
    host.handle(Request("kl", ["a"], str(run / "work")))
    host.handle(Request("kl", ["b"], str(run / "work")))
    for _ in range(3):
        host.tick()
        clock.advance(10)
    outs = [json.loads(Path(a[a.index("score") + 1]).read_text())["out"] for a in be.argvs.values()]
    assert len(set(outs)) == 2  # every reading has its own config and out dir
    assert all(e.valid is False and not e.metrics for e in host.handins())
    assert host.sched.get("judge-k1").ncards == 1


def failed_reading(tmp_path, output, rc=1):
    """A host whose one reading exits `rc` after writing `output` to its log and no result."""
    clock = FakeClock()
    be = FakeBackend(clock, default=FakeRun(10, rc=rc, output=output))
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {"kl_ref": str(_ref_dir(tmp_path)), "kl_holdout_ref": str(_ref_dir(tmp_path))},
        clock,
        backend=be,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    host.handle(Request("kl", ["a"], str(run / "work")))
    for _ in range(3):
        host.tick()
        clock.advance(10)
    return host, run


KERNEL_REFUSAL = (
    "Traceback (most recent call last):\n"
    '  File "/host/eval/qda_kl/serve.py", line 235, in check_kernels\n'
    '    raise Refused("the engine did not bind the checkpoint\'s kernels: " + ";".join(bad))\n'
    "qda_quant.step.Refused: the engine did not bind the checkpoint's kernels: dense: expected "
    "['HummingLinearKernel', 'MacheteLinearKernel'], bound ['HummingLinearKernel']\n"
) + "".join(
    f"(MainProcess pid={3867544 + i}) INFO [core_client.py:{i}] [shutdown] MPClient: step {i}\n"
    for i in range(11)
)


def test_a_failed_reading_quotes_its_own_refusal(tmp_path):
    host, run = failed_reading(tmp_path, KERNEL_REFUSAL)
    (e,) = host.handins()
    assert e.valid is False
    assert e.reason.startswith("the reading job failed (rc 1): ")
    assert "did not bind the checkpoint's kernels" in e.reason
    assert "bound ['HummingLinearKernel']" in e.reason
    assert "shutdown" not in e.reason and "MPClient" not in e.reason  # not the cleanup's last line
    assert str(run) not in e.reason and "/" not in e.reason  # a host path would be cut
    assert len(e.reason) <= 400
    shown = host.handle(Request("kl", ["show", "k1"], str(run / "work"))).stdout
    assert "did not bind the checkpoint's kernels" in shown and "rc 1" in shown


def test_a_failed_reading_with_an_empty_log_keeps_only_state_and_rc(tmp_path):
    host, _ = failed_reading(tmp_path, "", rc=2)
    (e,) = host.handins()
    assert e.reason == "the reading job failed (rc 2)"


#: The lines the shipped commands print on a refusal or an error, with no traceback: the contract's own
#: `error:` line (`qda-kl score`), and the `<command>: ...` line a tool inside the run writes.
SHIPPED_FAILURES = (
    "error: checkpoint cannot use the requested kernels",
    "error: RuntimeError: failed to bind the engine",
    "suite_job: refused: evaluator config splits into parts",
)


def test_the_contracts_own_line_is_a_diagnostic_the_ledger_quotes():
    """The line `qda`, `qda-kl` and `qda-agent` print on a refusal is the one the ledger folds and quotes."""
    from qda_quant.exit import refusal

    line = refusal(
        "checkpoint cannot use the requested kernels", "qda-kl ref --model M --out R"
    ).splitlines()[0]
    assert failing_line(line) == line


@pytest.mark.parametrize("line", SHIPPED_FAILURES)
def test_a_real_failed_childs_diagnostic_reaches_kl_show(tmp_path, line):
    run = tmp_path / "run"
    probe = f"import sys; print({line!r}, file=sys.stderr); sys.exit(2)"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "tick_s": 0.05},
        {
            "allow_any_format": True,
            "kl_cmd": [PY, "-c", probe, "{ckpt}", "{out}"],
            "socket": str(run / "work" / ".qda" / "rpc.sock"),
            "qda_cmd": [PY],
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
        },
    )
    (run / "work" / ".qda").mkdir(parents=True)
    host.start()
    try:
        ckpt(run / "work" / "a")
        assert host.handle(Request("kl", ["a"], str(run / "work"))).exit == 0
        assert host.drain(10)
        (e,) = host.handins()
        assert e.reason == f"the reading job failed (rc 2): {line}"
        shown = host.handle(Request("kl", ["show", "k1"], str(run / "work"))).stdout
        assert line in shown and "rc 2" in shown
    finally:
        host.stop()


def test_the_quoted_failure_line_has_host_paths_cut(tmp_path):
    missing = tmp_path / "run" / "judge" / "frozen" / "k1" / "model.safetensors"
    crash = (
        "Traceback (most recent call last):\n"
        '  File "/host/eval/build_kit/load.py", line 12, in load\n'
        f"    open({missing})\n"
        f"build_kit.KernelMismatch: cannot read {missing}\n"  # a name `EXCEPTION` does not know
    )
    host, _ = failed_reading(tmp_path, crash)
    (e,) = host.handins()
    assert "build_kit.KernelMismatch: cannot read" in e.reason  # the traceback's own last line
    assert "<path>" in e.reason
    assert str(missing) not in e.reason and str(tmp_path) not in e.reason


def test_the_layout_sets_a_readings_cards(tmp_path):
    clock = FakeClock()
    be = FakeBackend(clock)
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": list(range(8)), "budget_s": 3600, "fake_gpus": True},
        {
            "allow_any_format": True,
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
            "kl_cmd": ["score", "{config}"],
            "kl_layout": {
                "tp": 4,
                "serve_args": {"pipeline_parallel_size": 2, "enable_expert_parallel": True},
            },
            "check_cmd": None,
        },
        clock,
        backend=be,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    assert host.handle(Request("kl", ["a"], str(run / "work"))).exit == 0
    assert host.sched.get("judge-k1").ncards == 8  # tp x pipeline_parallel_size
    host.tick()
    argv = next(iter(be.argvs.values()))
    config = json.loads(Path(argv[argv.index("score") + 1]).read_text())
    assert config["scorer"]["topk"] == {
        "tp": 4,
        "serve_args": {"pipeline_parallel_size": 2, "enable_expert_parallel": True},
    }


def test_a_layout_cannot_name_its_own_tensor_parallel_size():
    with pytest.raises(ValueError, match="set kl_layout.tp=2"):
        JudgeConfig(
            contamination="none", kl_ref="r", kl_layout={"tp": 2, "serve_args": {"tensor_parallel_size": 2}}
        )


def test_a_readings_estimate_is_its_load_over_cards_plus_its_scoring():
    cfg = JudgeConfig(contamination="none", kl_ref="r", kl_load_s_per_gib=10, kl_score_s=100, kl_moe_factor=3)
    assert cfg.reading_expected_s(2 << 30, cards=1) == 120
    assert cfg.reading_expected_s(2 << 30, cards=2) == 110
    assert cfg.reading_expected_s(2 << 30, cards=1, moe=True) == 320  # the score is read three times


def test_a_measured_reading_replaces_the_estimate_and_moves_only_its_load_with_the_cards():
    """`prepare` ran this backend's reading for the engine floor: its seconds win over the vLLM constants.

    V's 8B GGUF search: prepare measured 548 s on 4 cards, the judge read on 1 card in 634 s, and the formula
    said 95 s, so the first watchdog cancelled it. At the measured card count the number is used as it stands.
    """
    measured = JudgeConfig(
        contamination="none", kl_ref="r", kl_load_s_per_gib=10, kl_score_s=60, reading_s=548, measured_cards=4
    )
    assert measured.reading_expected_s(2 << 30, cards=4) == 548
    # one card: the scoring stays, the load grows at the formula's per-gib rate (10 x 2 GiB over 1 vs 4 cards)
    assert measured.reading_expected_s(2 << 30, cards=1) == pytest.approx(548 + 20 * 0.75)
    assert measured.reading_expected_s(2 << 30, cards=1, moe=True) == pytest.approx(548 + 20 * 0.75)
    # no measured cards: the reading is taken as it stands, whatever the reading's cards
    assert (
        JudgeConfig(contamination="none", kl_ref="r", reading_s=548).reading_expected_s(2 << 30, cards=3)
        == 548
    )


def test_the_old_kl_keys_are_refused_naming_the_new_ones():
    with pytest.raises(ValueError, match="kl_cards was replaced by kl_layout"):
        JudgeConfig(contamination="none", kl_ref="r", kl_cards=2)
    with pytest.raises(ValueError, match="kl_expected_s was replaced by kl_load_s_per_gib and kl_score_s"):
        JudgeConfig(contamination="none", kl_ref="r", kl_expected_s=180)


def llama_paths(tmp_path, monkeypatch, *, name="llama.cpp", commit=None, gguf_py=True):
    """A fake pinned checkout and probe library; `commit_of` reads `commit` (default: the pin)."""
    from qda_quant.gguf.llama_cpp import PINNED

    root = tmp_path / name
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("")
    if gguf_py:
        (root / "gguf-py").mkdir()
    bridge = tmp_path / f"{name}-probe.so"
    bridge.write_bytes(b"")
    server = tmp_path / f"{name}-llama-server"
    server.write_bytes(b"")
    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda _root: commit or PINNED)
    return {"bridge": str(bridge), "llama_cpp": str(root), "server": str(server)}


def llama(**kw):
    """A llama.cpp JudgeSettings with the fields the cases below share; `kw` overrides them."""
    fields = {"engine": "llama.cpp", "budget_bytes": 1, "engine_floor": 0}
    return JudgeSettings(**{**fields, **kw})


def test_a_llama_cpp_search_states_its_bytes_floor_and_the_gguf_menu(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    cfg = llama(budget_bytes=17_548_181_504, engine_floor=0.000273, llamacpp=pair)
    assert (cfg.engine, cfg.budget_bytes, cfg.engine_floor) == ("llama.cpp", 17_548_181_504, 0.000273)
    assert cfg.format_menu == ["Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "Q3_K", "Q8_0"]
    spelled = llama(llamacpp=pair, format_menu=["q6_k", "Q4_K", "q5_k", "iq4_xs", "q8_0", "q6_k"])
    assert spelled.format_menu == ["Q6_K", "Q4_K", "Q5_K", "IQ4_XS", "Q8_0"]
    assert llama(llamacpp=pair, format_menu=[]).format_menu == []  # explicit empty: the task's statement
    assert llama(llamacpp=pair, format_menu=None).format_menu == cfg.format_menu


def test_a_llama_cpp_menu_may_name_bf16_and_f16_but_no_f32_or_fp8(tmp_path, monkeypatch):
    """The decided GGUF menu (six solved types, Q8_0, BF16) loads; no default lists a float; FP8 never loads."""
    pair = llama_paths(tmp_path, monkeypatch)
    decided = ["Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "Q3_K", "Q8_0", "BF16"]
    assert llama(llamacpp=pair, format_menu=decided).format_menu == decided
    assert llama(llamacpp=pair, format_menu=["q4_k", "bf16", "f16"]).format_menu == ["Q4_K", "BF16", "F16"]
    assert not {"BF16", "F16", "F32"} & set(llama(llamacpp=pair).format_menu)  # the default is unchanged
    for name in ("F32", "FP8", "F8_E4M3", "F8_E5M2"):
        with pytest.raises(ValueError, match=f"judge.format_menu: '{name}' is no GGUF type"):
            llama(llamacpp=pair, format_menu=["Q4_K", name])


def test_the_engine_floor_is_a_finite_nonnegative_kl(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    assert llama(llamacpp=pair).engine_floor == 0
    with pytest.raises(ValueError, match="is not a finite KL"):
        llama(engine_floor=float("inf"), llamacpp=pair)
    with pytest.raises(ValueError, match="greater than or equal to 0"):  # NaN: `ge=0` refuses it too
        llama(engine_floor=float("nan"), llamacpp=pair)


def test_llamacpp_names_the_bridge_and_the_pinned_checkout(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="needs judge.llamacpp.bridge"):
        llama()
    with pytest.raises(ValueError, match="bridge: .* is not a file"):
        llama(llamacpp={**pair, "bridge": str(tmp_path / "missing.so")})
    with pytest.raises(ValueError, match="llama_cpp: .* is not a directory"):
        llama(llamacpp={**pair, "llama_cpp": str(tmp_path / "missing")})
    with pytest.raises(ValueError, match="has no gguf-py"):
        llama(llamacpp=llama_paths(tmp_path, monkeypatch, name="bare", gguf_py=False))
    with pytest.raises(ValueError, match="not the pinned llama.cpp"):
        llama(llamacpp=llama_paths(tmp_path, monkeypatch, name="old", commit="0" * 40))
    with pytest.raises(ValueError, match="judge.llamacpp needs judge.engine: llama.cpp"):
        JudgeSettings(llamacpp=pair)


def test_a_vllm_search_is_unchanged_by_the_engine_fields():
    cfg = JudgeSettings()
    assert cfg.engine == "vllm"
    assert cfg.budget_bytes is None and cfg.engine_floor is None
    assert cfg.format_menu == ["bf16", "w8a16", "w4a16"]


def test_one_budget_per_engine_and_the_fields_that_need_llama_cpp():
    with pytest.raises(ValueError, match="no GGUF type the llama.cpp target declares"):
        llama(format_menu=["Q2_K"])
    with pytest.raises(ValueError, match="drop judge.budget_bpw"):
        llama(budget_bpw=4)
    with pytest.raises(ValueError, match="needs judge.budget_bytes"):
        JudgeSettings(engine="llama.cpp", engine_floor=0)
    with pytest.raises(ValueError, match="needs judge.engine_floor"):
        JudgeSettings(engine="llama.cpp", budget_bytes=1)
    with pytest.raises(ValueError, match="needs judge.engine: llama.cpp"):
        JudgeSettings(budget_bytes=1)
    with pytest.raises(ValueError, match="has no engine floor"):
        JudgeSettings(engine_floor=0.1)


def _reading_with_an_override(tmp_path, monkeypatch, cmd=(), moe=()):
    """Queue a MoE reading with `cmd` added to its command and `moe` as its MoE args.

    Returns the host, the run dir and the reply.
    """
    clock = FakeClock()
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {
            "allow_any_format": True,
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
            "check_cmd": None,
            "kl_layout": {"tp": 1, "serve_args": {"pipeline_parallel_size": 1}},
            "kl_cmd": ["{python}", "-m", "qda_kl.cli", "score", "{config}", "{moe}", *cmd],
            "kl_moe_args": [*moe],
        },
        clock,
        backend=FakeBackend(clock),
        autotick=False,
    )
    ckpt(run / "work" / "a")
    monkeypatch.setattr("qda_agent.judge.commands.has_routed_experts", lambda _: True)
    return host, run, host.handle(Request("kl", ["a"], str(run / "work")))


def _assert_nothing_was_queued(host, run, reply):
    assert reply.exit == 2, reply
    assert "a reading's layout comes only from judge.kl_layout" in reply.stderr
    assert host.sched.jobs == {}  # no reading was scheduled
    assert host.entries == []  # the entry and its frozen copy are gone
    assert list((run / "judge" / "frozen").iterdir()) == []


@pytest.mark.parametrize("where", ["cmd", "moe"])
@pytest.mark.parametrize(
    "override",
    ["{topk: {tp: 2}}", "{topk: {tp: 1, serve_args: {pipeline_parallel_size: 2}}}"],
    ids=["tp", "pipeline_split"],
)
def test_a_whole_scorer_override_in_a_reading_cannot_change_its_layout(
    tmp_path, monkeypatch, where, override
):
    """`kl_cmd` and `kl_moe_args` are two ways into the same command; neither may replace the layout."""
    added = ("--set", f"scorer={override}")
    if where == "cmd":
        host, run, reply = _reading_with_an_override(tmp_path, monkeypatch, cmd=added)
    else:
        host, run, reply = _reading_with_an_override(tmp_path, monkeypatch, moe=added)
    _assert_nothing_was_queued(host, run, reply)


def test_a_fused_abbreviated_set_option_is_read_as_an_override(tmp_path, monkeypatch):
    # `scorer={...}` replaces the ancestor a key-name check cannot see, and `--se=` is argparse's `--set=`.
    host, run, reply = _reading_with_an_override(tmp_path, monkeypatch, cmd=("--se=scorer={topk: {tp: 2}}",))
    _assert_nothing_was_queued(host, run, reply)


def test_a_scorer_setting_that_is_not_the_layout_is_accepted(tmp_path, monkeypatch):
    # The check is the parsed split, not the word `scorer`: MoE repeat settings still reach the command.
    host, _, reply = _reading_with_an_override(tmp_path, monkeypatch, moe=("--set", "scorer.topk.repeat=3"))
    assert reply.exit == 0, reply
    assert host.sched.get("judge-k1").argv[-2:] == ["--set", "scorer.topk.repeat=3"]


def test_a_replayed_record_still_refuses_an_empty_anchor(tmp_path):
    from qda_agent.config import RunConfig

    anchor = tmp_path / "empty"
    anchor.mkdir()
    (tmp_path / "ref").mkdir()  # a configured path must exist when the run config is validated
    cfg = RunConfig.model_validate(
        {
            "model": "tiny/model",
            "format": "w4a16",
            "budget_gpu_hours": 1,
            "contamination": "none",
            "cards": [0],
            "run_dir": str(tmp_path / "run"),
            "judge": {
                "anchor": str(anchor),
                "kl_ref": str(_ref_dir(tmp_path, "ref")),
                "kl_holdout_ref": str(_ref_dir(tmp_path, "ref")),
            },
        }
    )
    resolved = cfg.resolved()
    assert resolved["judge"]["max_copy_bytes"] is None  # no fallback cap the replayed host would trust
    path = tmp_path / "resolved.yaml"
    path.write_text(json.dumps(resolved))  # JSON is YAML
    again = RunConfig.load(path)
    clock = FakeClock()
    with pytest.raises(ValueError, match="has no files"):
        make_host(
            tmp_path / "run2",
            again.sched_config(datetime.now(UTC)),
            again.judge_config(),
            clock,
            backend=FakeBackend(clock),
            autotick=False,
        )


def test_the_drain_window_is_one_reading_of_the_anchor_and_resolved_records_it(tmp_path):
    from qda_agent.config import RunConfig

    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "config.json").write_text("{}")
    size = (anchor / "config.json").stat().st_size
    cfg = RunConfig.model_validate(
        {
            "model": "tiny/model",
            "format": "w4a16",
            "budget_gpu_hours": 1,
            "contamination": "none",
            "cards": [0, 1],
            "run_dir": str(tmp_path / "run"),
            "judge": {"anchor": str(anchor), "kl_score_s": 1200, "kl_load_s_per_gib": 0},
        }
    )
    assert cfg.drain_window_s() == 1200  # one reading of the anchor
    resolved = cfg.resolved()
    assert resolved["drain_timeout_s"] == cfg.drain_window_s()
    assert resolved["judge"]["max_copy_bytes"] == 2 * size + (1 << 30)
    plain = RunConfig.model_validate(
        {
            "model": "tiny/model",
            "format": "w4a16",
            "budget_gpu_hours": 1,
            "contamination": "none",
            "cards": [0],
            "run_dir": str(tmp_path / "r2"),
        }
    )
    assert plain.drain_window_s() == 900 and plain.resolved()["judge"]["max_copy_bytes"] == 200 << 30


def test_judge_jobs_run_in_the_drain_window(tmp_path):
    r = Rig(tmp_path, budget_s=600, grace_s=120)  # readings and checks take 60 s
    ckpt(r.work / "a")
    ckpt(r.work / "b", weights=b"w2")
    r.clock.advance(599)
    assert r.call("handin", "a").exit == 0  # 1 s before the GPU deadline: its check ends after it
    assert r.call("kl", "b").exit == 0  # a reading asked for 1 s before the deadline too
    r.clock.advance(1)  # exactly at the deadline
    r.host.cutoff()
    assert "deadline" in r.call("kl", "b").stderr  # the agent's new readings stop with its hand-ins
    assert "deadline" in r.call("handin", "b").stderr
    assert r.host.drain(300)  # ... but what it asked for in time runs in the drain window
    (h, k) = r.host.handins()
    assert h.valid is True and k.valid is True and "kl" in k.metrics
    assert r.host.final().rule == "handin"


def test_kl_after_the_deadline_cannot_become_final(tmp_path):
    """With no valid hand-in, a scored build asked for after the deadline cannot become final."""
    r = Rig(tmp_path, budget_s=600)
    ckpt(r.work / "late")
    r.clock.advance(600)  # the deadline, no cutoff called yet
    rep = r.call("kl", "late")
    assert rep.exit == 2 and "deadline" in rep.stderr
    assert r.host.handins() == [] and list((r.run / "judge" / "frozen").iterdir()) == []
    assert r.host.final().rule == "handin"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_freeze_survives_dir_swaps(tmp_path):
    """B3: swap the checkpoint dir for a link to an outside dir while kl/handin copy it."""
    r = Rig(tmp_path, check=False)
    outside = ckpt(tmp_path / "outside", weights=b"OUTSIDE-SECRET" * 100)
    good = ckpt(tmp_path / "good-src")
    stop = threading.Event()

    def swapper():
        d, tmp = r.work / "ck", r.work / "ck.tmp"
        while not stop.is_set():
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
                tmp.symlink_to(outside)
                if d.is_dir() and not d.is_symlink():
                    shutil.rmtree(d)
                tmp.replace(d)  # now a link out of the workdir
            with contextlib.suppress(OSError):
                d.unlink()
                shutil.copytree(good, d)

    shutil.copytree(good, r.work / "ck")
    t = threading.Thread(target=swapper, daemon=True)
    t.start()
    try:
        for i in range(40):
            rep = r.call("kl" if i % 2 else "handin", "ck")
            assert rep.exit in (0, 2), rep
    finally:
        stop.set()
        t.join()
    frozen = r.run / "judge" / "frozen"
    for d in frozen.iterdir():
        assert not d.is_symlink()
        assert b"OUTSIDE-SECRET" not in (d / "model.safetensors").read_bytes()
    assert all(e.dir.is_dir() and not e.dir.is_symlink() for e in r.host.handins())


def test_ngram_resolves_without_a_host(tmp_path):
    """The flow builds the scanner before any host exists, with the judge's policy and guard."""
    import importlib
    import subprocess

    item = " ".join(f"q{i}" for i in range(25))
    guard = tmp_path / "guard.txt"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "notes.md").write_text(item)
    code = (
        "import sys; from pathlib import Path; from qda_agent.judge.contamination import Ngram, scan; "
        "from qda_agent.settings import pick; from qda_agent.judge.contamination import CHECKS; "
        "print(list(CHECKS)); print(scan([Path(sys.argv[1])], pick(CHECKS, {'name': 'ngram', 'guard': sys.argv[2]})))"
    )
    p = subprocess.run(
        [PY, "-c", code, str(tmp_path / "w"), str(guard)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path(contamination.__file__).parents[2]), *sys.path]),
        },
    )
    assert p.returncode == 0, p.stderr
    known, found = p.stdout.splitlines()
    assert "'ngram', 'none'" in known and "notes.md: contains a test item" in found
    assert (
        importlib.import_module("qda_agent.judge.contamination").scan(
            [tmp_path / "w"], contamination.NoCheck()
        )
        == []
    )
    with pytest.raises(ValueError, match="guard"):
        contamination.Ngram().scanner()
    with pytest.raises(ValueError, match="guard"):
        contamination.Ngram(guard=tmp_path / "missing").scanner()


def test_host_uses_the_configured_ngram_policy(tmp_path):
    item = " ".join(f"t{i}" for i in range(25))
    guard = tmp_path / "guard.txt"
    guard.write_text("\n".join(sorted(contamination.windows(item))))
    r = Rig(tmp_path, check=False, contamination={"name": "ngram", "guard": guard})
    d = ckpt(r.work / "a")
    (d / "README.md").write_text("model card\n" + item)
    assert "contamination" in r.call("handin", "a").stdout and r.host.handins()[0].valid is False
    with pytest.raises(ValueError, match="guard"):
        Rig(tmp_path / "b", contamination="ngram")


def test_a_run_naming_no_contamination_check_gets_ngram_and_needs_its_guard(tmp_path):
    """The default check is ngram: a run that names none is refused AT LOAD, naming contamination.guard (never
    silently `none`, never a failure at the host's start); `none` written out still runs without a check."""
    from qda_agent.config import RunConfig
    from qda_agent.configfile import ConfigError

    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
    }
    path = tmp_path / "run.yaml"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match=r"contamination\.guard is not set: .*ngram, the default"):
        RunConfig.load(path)
    with pytest.raises(ValueError, match=r"contamination\.guard"):
        RunConfig.model_validate({**raw, "contamination": "ngram"})  # named without a guard: the same
    assert RunConfig.model_validate({**raw, "contamination": {"name": "none"}}).contamination.name == "none"
    guard = tmp_path / "guard.txt"
    guard.write_text("0" * 64 + "\n")
    cfg = RunConfig.model_validate({**raw, "contamination": {"guard": str(guard)}})  # no name: the default
    assert cfg.contamination.name == "ngram" and cfg.contamination.guard == guard
    assert cfg.resolved()["contamination"] == {"name": "ngram", "guard": str(guard)}


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_subcommand_env_is_set_inside_bwrap(tmp_path, monkeypatch):
    """Like agent jobs, bwrap keeps the host env and the agent's env is set inside it only."""
    from qda_agent.judge import commands as commands_mod

    seen = {}
    real = commands_mod.subprocess.run

    def spy(argv, **kw):
        seen.update(argv=argv, env=kw.get("env"))
        if argv[:2] == ["bwrap", "--args"]:  # the options file is open only while the spy runs
            fd = int(argv[2])
            seen["options"] = os.pread(fd, os.fstat(fd).st_size, 0).decode().split("\0")
        return real(argv, **kw)

    monkeypatch.setattr(commands_mod.subprocess, "run", spy)
    monkeypatch.setenv("QDA_TEST_SECRET", "leak")
    r = Rig(
        tmp_path,
        qda_cmd=[PY, "-c", "import os; print(os.environ.get('HOME'), os.environ.get('QDA_TEST_SECRET'))"],
    )
    out = r.call("qda", "plan").stdout.split()
    assert out == [str(r.work), "None"]
    # since 41c19a82 the bwrap options travel through `--args`, not the process command line.
    assert seen["argv"][:2] == ["bwrap", "--args"], seen["argv"]
    options = seen["options"]
    inner = {options[i + 1]: options[i + 2] for i, x in enumerate(options) if x == "--setenv"}
    assert seen["env"] is None and "--clearenv" in options and inner["HOME"] == str(r.work)
    assert inner["CUDA_VISIBLE_DEVICES"] == ""


def test_qda_binds_its_sources_through_the_same_last_moment_check(tmp_path):
    """A readable source swapped after validation must not carry the judge's dir into a `qda` verb."""
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    (run / "judge").mkdir()
    (run / "judge" / "planted.txt").write_text("judge secret\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "data"
    link.symlink_to(outside)  # the source as the config check saw it
    host = make_host(
        run,
        {
            "cards": [0],
            "budget_s": 3600,
            "job_sandbox": "bwrap",
            "readable": [str(link)],
            "run_dir": str(run),
        },
        {
            "kl_cmd": ["score", "{ckpt}", "{out}"],
            "check_cmd": None,
            "qda_cmd": [PY],
            "allow_any_format": True,
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
        },
        autotick=False,
    )
    shutil.rmtree(outside)
    link.unlink()
    link.symlink_to(run / "judge")  # the agent's swap, after the config was checked
    rep = host.handle(Request("qda", ["plan"], str(run / "work")))
    assert rep.exit == 1 and "refusing to bind it" in rep.stderr, rep


def stub_qda(tmp_path):
    """A stand-in `qda` in a venv-like tree outside the workdir and the system dirs."""
    bindir = tmp_path / "qvenv" / "bin"
    bindir.mkdir(parents=True)
    (bindir.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
    q = bindir / "qda"
    q.write_text(f"#!{PY}\nimport sys\nprint('stub-qda', *sys.argv[1:])\n")
    q.chmod(0o755)
    return q


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_qda_build_job_finds_qda_in_the_sandbox(tmp_path, monkeypatch):
    """Seen in a smoke run: `bwrap: execvp qda: No such file or directory` -- the host's qda must run in the job."""
    q = stub_qda(tmp_path)
    monkeypatch.setenv("PATH", f"{q.parent}:{os.environ['PATH']}")
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "job_sandbox": "bwrap", "tick_s": 0.05},
        {
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
            "check_cmd": None,
            "evaluator": str(evaluator(tmp_path)),
            "qda_cmd": ["qda"],
            "allow_any_format": True,
        },
        autotick=False,
    )
    host.start()
    try:
        (run / "work" / "recipe.yaml").write_text("steps: []\n")
        rep = host.handle(
            Request("qda", ["--expected", "1m", "build", "recipe.yaml", "--out", "o"], str(run / "work"))
        )
        assert rep.exit == 0, rep
        t = time.time() + 20
        while host.sched.pending():
            assert time.time() < t
            host.tick()
            time.sleep(0.05)
        (job,) = host.sched.list(origin="agent")
        log = host.sched.log(job.name)
        assert job.state is State.done, log
        assert log.startswith("stub-qda build ") and log.endswith("--out o")
        assert host.handle(Request("qda", ["plan"], str(run / "work"))).stdout.strip() == "stub-qda plan"
    finally:
        host.stop()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
@pytest.mark.parametrize("autotick", [False, True])
def test_a_job_outlives_the_request_that_waited_for_it(tmp_path, autotick):
    """Seen in a smoke search: a job started by `qsched wait`'s tick, in the request's thread, was SIGKILLed
    (rc -9) when the wait timed out and its thread ended: `bwrap --die-with-parent` takes the forking thread
    as its parent. Jobs are now started by the host's tick thread only (the main thread without autotick)."""
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "job_sandbox": "bwrap", "tick_s": 0.05, "kill_grace_s": 1},
        {
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
            "check_cmd": None,
            "evaluator": str(evaluator(tmp_path)),
            "qda_cmd": [shutil.which("true")],
            "allow_any_format": True,
        },
        autotick=autotick,
    )
    host.start()
    try:
        (run / "work" / "s.sh").write_text("sleep 120\n")
        assert rpc.call(host.socket, "qsched", ["add", "--expected", "10m", "--name", "c", "s.sh"]).exit == 0
        wait = rpc.call(host.socket, "qsched", ["wait", "--any", "--timeout", "2s"])  # its thread then ends
        assert wait.exit == 124, wait
        if not autotick:
            assert host.sched.get("c").state is State.queued  # the request's thread started nothing
            host.tick()
        end = time.time() + 3
        while time.time() < end:
            if not autotick:
                host.tick()
            time.sleep(0.1)
        job = host.sched.get("c")
        assert job.state is State.running and job.rc is None, (job.state, job.rc, job.reason)
    finally:
        host.stop()


def _local_host(tmp_path: Path):
    run = tmp_path / "run"
    host = make_host(
        run,
        {
            "cards": [0],
            "budget_s": 3600,
            "job_sandbox": "none",
            "gpus": False,
            "tick_s": 0.05,
            "kill_grace_s": 1,
        },
        {
            "kl_ref": str(_ref_dir(tmp_path)),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
            "check_cmd": None,
            "evaluator": str(evaluator(tmp_path)),
            "qda_cmd": [shutil.which("true")],
            "allow_any_format": True,
        },
    )
    host.start()
    (run / "work" / "ok.sh").write_text("true\n")
    return host, run


def _settled(host, name: str, timeout_s: float = 10.0) -> State:
    end = time.time() + timeout_s
    while not host.sched.get(name).state.terminal:
        assert time.time() < end, host.sched.get(name)
        time.sleep(0.05)
    return host.sched.get(name).state


def test_the_tick_thread_survives_a_failed_pass(tmp_path, monkeypatch):
    """One exception in a pass (here a full disk while a job is saved) is logged and the next pass follows: the
    tick thread is the only one that starts jobs, so a later job must still start."""
    host, run = _local_host(tmp_path)
    try:
        real, failed = host.sched._save, []

        def full_once(job):
            if job.state is State.running and not failed:
                failed.append(job.name)
                raise OSError(28, "No space left on device")
            real(job)

        monkeypatch.setattr(host.sched, "_save", full_once)
        assert rpc.call(host.socket, "qsched", ["add", "--expected", "1m", "--name", "a", "ok.sh"]).exit == 0
        end = time.time() + 5
        while not failed:
            assert time.time() < end
            time.sleep(0.05)
        assert rpc.call(host.socket, "qsched", ["add", "--expected", "1m", "--name", "b", "ok.sh"]).exit == 0
        assert _settled(host, "b") is State.done and host._thread.is_alive()
        assert (
            "tick: OSError: [Errno 28] No space left on device"
            in (run / "judge" / "host-errors.log").read_text()
        )
    finally:
        host.stop()


def test_a_failing_pass_with_any_text_neither_kills_the_thread_nor_raises_from_drain(tmp_path, monkeypatch):
    """A pass that fails every time, with a raw surrogate in its text: the log line is written, the tick thread
    lives, and `drain` waits out its timeout instead of raising."""
    host, run = _local_host(tmp_path)
    try:
        assert rpc.call(host.socket, "qsched", ["add", "--expected", "1m", "--name", "a", "ok.sh"]).exit == 0
        _settled(host, "a")
        host.sched.submit(["sleep", "30"], str(run / "work"), name="j", expected_s=60, origin="judge")

        name = b"x\xff".decode(errors="surrogateescape")  # as a path read from disk may hold

        def broken():
            raise OSError(f"bad name {name}")

        monkeypatch.setattr(host.sched, "tick", broken)
        time.sleep(0.3)
        assert host._thread.is_alive()
        assert host.drain(0.3) is False
        assert "tick: OSError: bad name x\\udcff" in (run / "judge" / "host-errors.log").read_text()
    finally:
        monkeypatch.undo()
        host.stop()


def test_a_job_that_cannot_start_fails_alone(tmp_path):
    """An agent's argv with a NUL (a ValueError from the launch) fails that job; the next one runs."""
    host, run = _local_host(tmp_path)
    try:
        (run / "work" / "r.yaml").write_text("steps: []\n")
        bad = rpc.call(host.socket, "qda", ["--expected", "1m", "build", "r.yaml", "a\0b"])
        assert bad.exit == 0, bad
        (a,) = [j.name for j in host.sched.list(origin="agent")]
        assert _settled(host, a) is State.failed
        assert host.sched.get(a).reason.startswith("could not start: ValueError")
        assert rpc.call(host.socket, "qsched", ["add", "--expected", "1m", "--name", "b", "ok.sh"]).exit == 0
        assert _settled(host, "b") is State.done and host._thread.is_alive()
    finally:
        host.stop()


def test_host_start_refuses_missing_qda_and_long_socket(tmp_path):
    run = tmp_path / "run"
    base = {
        "kl_ref": str(_ref_dir(tmp_path)),
        "kl_holdout_ref": str(_ref_dir(tmp_path)),
        "check_cmd": None,
        "evaluator": str(evaluator(tmp_path)),
        "allow_any_format": True,
    }
    with pytest.raises(ValueError, match="no-such-qda-xyz.*not found"):
        make_host(
            run, {"cards": [0], "fake_gpus": True}, {**base, "qda_cmd": ["no-such-qda-xyz"]}, autotick=False
        )
    long_run = tmp_path / ("r" * 120)
    host = make_host(long_run, {"cards": [0], "fake_gpus": True}, {**base, "qda_cmd": [PY]}, autotick=False)
    with pytest.raises(ValueError, match="too long for a unix socket"):
        host.start()
    assert not (long_run / "rpc.sock").exists()


def _qda_host(tmp_path, run, q, *, sched=None, **judge):
    base = {
        "kl_ref": str(_ref_dir(tmp_path)),
        "kl_holdout_ref": str(_ref_dir(tmp_path)),
        "check_cmd": None,
        "evaluator": str(evaluator(tmp_path)),
        "allow_any_format": True,
    }
    return make_host(
        run,
        sched or {"cards": [0], "fake_gpus": True},
        {**base, "qda_cmd": [str(q)], **judge},
        autotick=False,
    )


def _venv_qda(root):
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    q = root / "bin" / "qda"
    q.write_text("#!/usr/bin/python3\nprint('q')\n")
    q.chmod(0o755)
    return q


def test_qda_binds_exactly_its_venvs(tmp_path):
    """Only a virtualenv (or a system prefix) is bound for qda; the exact list."""
    q = stub_qda(tmp_path)  # #!<quant venv python>: its venv is bound too; /usr already is
    host = _qda_host(tmp_path, tmp_path / "run", q)
    before = list(host.sched.cfg.readable)
    host.start()
    try:
        py_venv = Path(os.path.normpath(Path(PY).absolute())).parent.parent
        added = host.sched.cfg.readable[len(before) :]
        assert added[:2] == [q.parent.parent, py_venv]
        # the rest is what the venvs' interpreters need beyond /usr: a uv, pyenv or conda Python's base installation
        rest = [
            r
            for r in interpreter(PY).roots
            if r not in (q.parent.parent, py_venv) and not r.is_relative_to("/usr")
        ]
        assert added[2:] == [r for r in dict.fromkeys(rest) if r not in before]
    finally:
        host.stop()


def _uv_qda(root, home, *, base_exists=True):
    """A qda in a venv whose `pyvenv.cfg` names a uv-style base under `home` (not /usr, not /opt)."""
    base = home / ".local" / "share" / "uv" / "python" / "cpython-3.12.3-linux-x86_64-gnu"
    if base_exists:
        base.mkdir(parents=True)
    venv = root / "venvs" / "quant"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base}/bin\n")
    q = venv / "bin" / "qda"
    q.write_text(f"#!{PY}\nprint('q')\n")
    q.chmod(0o755)
    return q, venv, base


def test_qda_binds_the_uv_base_interpreter_under_home(tmp_path, monkeypatch):
    """A venv's base interpreter under $HOME (uv, pyenv or conda) is bound read-only with its venv."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    q, venv, base = _uv_qda(tmp_path, home)
    host = _qda_host(tmp_path, tmp_path / "run", q)
    host.start()
    try:
        assert venv in host.sched.cfg.readable and base in host.sched.cfg.readable
        assert base not in host.sched.cfg.binds.values()  # read-only, like every qda installation
    finally:
        host.stop()


def _user_qda(root):
    """A `qda` installed outside any venv (a `pip install --user` style directory), which no bind accepts."""
    bindir = Path(root) / "home" / ".local" / "bin"
    bindir.mkdir(parents=True)
    q = bindir / "qda"
    q.write_text("#!/usr/bin/python3\nprint('q')\n")
    q.chmod(0o755)
    return q


def test_an_unsandboxed_run_binds_nothing_for_qda(tmp_path):
    """`scheduler.job_sandbox: none` runs the jobs on the host: no bind, so no installation is refused for one."""
    q = _user_qda(tmp_path)
    host = _qda_host(
        tmp_path, tmp_path / "run", q, sched={"cards": [0], "fake_gpus": True, "job_sandbox": "none"}
    )
    host.start()
    try:
        assert q.parent.parent not in host.sched.cfg.readable  # a bind no job would mount
        assert not host.sched.cfg.binds
    finally:
        host.stop()


def _run_jobs(tmp_path, level, knob):
    """How a run at `level` with `scheduler.job_sandbox` `knob` (None: unset) runs its jobs (`sched_config`'s)."""
    from qda_agent.config import RunConfig

    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 0.01,
        "cards": [0],
        "run_dir": str(tmp_path / "cfg-run"),
        "contamination": "none",
        "isolation": {"level": level},
        "scheduler": {"name": "qsched"} | ({} if knob is None else {"job_sandbox": knob}),
    }
    return RunConfig.model_validate(raw).job_sandbox()


@pytest.mark.parametrize(
    ("level", "knob"),
    [("none", None), ("none", "none"), ("sandbox", "bwrap"), ("sandbox", None)],
    ids=["unset", "none", "bwrap", "unset-at-sandbox"],
)
def test_a_build_jobs_path_keeps_the_shell_path_unless_jobs_are_sandboxed(tmp_path, monkeypatch, level, knob):
    """A judge job on the host keeps the shell's PATH after qda's bin/, so a compiler only the shell has still
    resolves; a bwrap job keeps the directories it mounts. Unset at the default level is the host."""
    from qda_agent.isolation.paths import job_env

    shell_only = tmp_path / "shell-only" / "bin"  # a compiler directory only this shell has
    monkeypatch.setenv("PATH", f"{shell_only}:{os.environ['PATH']}")
    shell = os.environ["PATH"]
    q = _venv_qda(tmp_path / "qvenv")
    jobs = _run_jobs(tmp_path, level, knob)
    host = _qda_host(
        tmp_path, tmp_path / "run", q, sched={"cards": [0], "fake_gpus": True, "job_sandbox": jobs}
    )
    host.start()
    try:
        if knob == "bwrap" or level != "none":
            assert host.qda_path == f"{q.parent}:{job_env(host.workdir)['PATH']}"
            assert str(shell_only) not in host.qda_path
        else:
            assert host.qda_path == f"{q.parent}:{shell}"
    finally:
        host.stop()


def test_custody_with_jobs_on_the_host_is_refused_with_its_fix(tmp_path):
    """Unset at `isolation.level: none` runs the jobs on the host, where custody cannot hold: refused at start."""
    q = _venv_qda(tmp_path / "qvenv")
    jobs = _run_jobs(tmp_path, "none", None)
    with pytest.raises(
        ValueError, match=r"custody_dir needs the jobs in bwrap.*fix: set isolation.level: sandbox"
    ):
        _qda_host(
            tmp_path,
            tmp_path / "run",
            q,
            sched={"cards": [0], "fake_gpus": True, "job_sandbox": jobs},
            custody_dir=str(tmp_path / "custody"),
        ).start()


def test_the_same_install_is_refused_when_the_jobs_are_sandboxed(tmp_path):
    """The skip is the job sandbox's: a sandboxed job cannot be given that installation, and is told so.

    `static_checks` resolves `qda` before the host creates anything, so it refuses at construction.
    """
    q = _user_qda(tmp_path)
    with pytest.raises(ValueError, match="neither a virtualenv"):
        _qda_host(tmp_path, tmp_path / "run", q)


def test_a_qda_venv_whose_base_interpreter_is_missing_is_refused_naming_it(tmp_path):
    """The error says where the base interpreter should be, not only that the directory is not a venv."""
    q, _venv, base = _uv_qda(tmp_path, tmp_path / "home", base_exists=False)
    # `static_checks` resolves `qda` before the host creates anything, so it refuses at construction
    with pytest.raises(ValueError, match=f"base interpreter is at {base}.*does not exist"):
        _qda_host(tmp_path, tmp_path / "run", q)


@pytest.mark.parametrize(
    "case",
    [
        "home-bin",
        "home-itself",
        "dotlocal",
        "run-bin",
        "venv-holds-run",
        "kl-ref-in-venv",
        "guard-in-venv",
        "evaluator-in-venv",
    ],
)
def test_qda_installation_that_would_expose_too_much_is_refused(tmp_path, monkeypatch, case):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    run = tmp_path / "run"
    extra = {}
    if case == "home-bin":  # ~/bin/qda, not a venv
        (home / "bin").mkdir()
        q = home / "bin" / "qda"
        q.write_text("#!/usr/bin/python3\n")
        q.chmod(0o755)
        match = "neither a virtualenv"
    elif case == "home-itself":  # a "venv" that is $HOME
        q, match = _venv_qda(home), r"expose \$HOME"
    elif case == "dotlocal":  # ~/.local/bin/qda
        q = home / ".local" / "bin" / "qda"
        q.parent.mkdir(parents=True)
        q.write_text("#!/usr/bin/python3\n")
        q.chmod(0o755)
        match = "neither a virtualenv"
    elif case == "run-bin":  # <run_dir>/bin/qda in a venv that is the run dir
        q, match = _venv_qda(run), "expose run_dir"
    elif case == "venv-holds-run":
        q = _venv_qda(tmp_path / "v")
        run = tmp_path / "v" / "runs" / "r1"
        match = "expose run_dir"
    elif case == "kl-ref-in-venv":
        q = _venv_qda(tmp_path / "v")
        extra = {
            "kl_ref": str(_ref_dir(tmp_path / "v", "ref")),
            "kl_holdout_ref": str(_ref_dir(tmp_path)),
        }
        match = "expose kl_ref"
    elif case == "guard-in-venv":  # the guard holds test items
        q = _venv_qda(tmp_path / "v")
        (tmp_path / "v" / "guard.txt").write_text("0" * 64 + "\n")
        extra = {"contamination": {"name": "ngram", "guard": str(tmp_path / "v" / "guard.txt")}}
        match = "expose contamination.guard"
    else:  # the evaluator's live source tree inside qda's venv
        q = _venv_qda(tmp_path / "v")
        ev = tmp_path / "v" / "lib" / "fake_eval"
        ev.mkdir(parents=True)
        (ev / "__init__.py").write_text("")
        extra = {"evaluator": str(ev)}
        match = "expose judge.evaluator"
    with pytest.raises(ValueError, match=match):
        _qda_host(tmp_path, run, q, **extra)


@pytest.mark.parametrize(
    "serve",
    [
        {"env": {"PYTHONPATH": "/tmp/agent"}},
        {"env": {"LD_PRELOAD": "/tmp/x.so"}},
        {"args": {"trust_remote_code": True}},
        {"args": {"moe_backend": "evil"}},
    ],
)
def test_a_candidate_with_its_own_serve_settings_is_refused_before_anything_runs(tmp_path, serve):
    """The KL reading and the serving check apply build.json's serve on the host: only qda's own settings pass."""
    r = Rig(tmp_path)
    for name, verb in (("a", "kl"), ("b", "handin")):
        src = ckpt(r.work / name)
        (src / "build.json").write_text(json.dumps({"serve": {"vllm": "0.33.0", **serve}}))
        rep = r.call(verb, name)
        assert rep.exit != 0 and "serve settings not allowed" in rep.stderr, rep
    r.settle()
    assert r.seen == {} and r.host.handins() == []  # nothing was scored or checked


def test_a_candidate_with_qdas_serve_settings_is_scored(tmp_path):
    r = Rig(tmp_path)
    src = ckpt(r.work / "a")
    serve = {
        "vllm": "0.33.0",
        "env": {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"},
        "args": {"moe_backend": "marlin"},
    }
    (src / "build.json").write_text(json.dumps({"serve": serve, "timings": {}}))
    assert r.call("kl", "a").exit == 0
    r.settle()
    (e,) = r.host.handins()
    assert e.valid and e.metrics["kl"] == pytest.approx(0.0712)


def test_a_receipted_layer_cache_build_cannot_be_a_finalist_or_checked(tmp_path):
    """A layer-cache build says so in the `build.json` its receipt hashes: screening only, however it scores."""
    from qda_agent.commands import NAMES
    from qda_agent.judge.build_receipt import SCREENING_BUILD
    from qda_agent.judge.freeze import file_sha256, seal
    from qda_agent.sched import Job, State

    r = Rig(
        tmp_path,
        require_build_receipt=True,
        commands=[*NAMES, "gencheck"],
        gencheck_cmd=["check", "{config}"],
        model="org/original-model",
    )
    private = r.host.jdir / "builds" / "cached"
    tree = ckpt(private / "published", b"a")  # screens best of the candidates
    (tree / "build.json").write_text(
        json.dumps({"serve": {}, "layer_cache": [{"segment": 0, "screening_only": True}]})
    )
    receipt = {
        "job_id": "cached",
        "destination": "cached",
        "workdir": str(r.work),
        "source": "org/original-model",
        "source_sha256": "",
        "tree_sha256": seal(tree),
        "build_sha256": file_sha256(tree / "build.json"),
    }
    (private / "receipt.json").write_text(json.dumps(receipt))
    r.host.sched.jobs["cached"] = Job(
        id="cached",
        name="cached",
        seq=0,
        argv=["true"],
        cwd=str(r.work),
        expected_s=1,
        submitted=r.clock.now(),
        state=State.done,
        rc=0,
        build_private=str(private),
    )
    assert r.call("kl", "cached").exit == 0
    rep = r.call("gencheck", "cached")
    assert rep.exit == 2 and "screening-only" in (rep.stdout + rep.stderr)
    r.settle()
    (e,) = r.host.entries
    assert e.valid is True and e.build_receipt["tree_sha256"] == e.digest
    assert e.eligible is False and e.reason == SCREENING_BUILD
    final = r.host.final(generation_checks=False)
    assert final.choice is None and "screening-only" in final.model_dump_json()
    assert "screening-only" in r.host.entry(e.id, "scored").record().reason


def test_start_verifies_the_trusted_files_the_judge_just_installed(tmp_path):
    """The level's `verify_trusted` runs after the judge installs its files, before any turn.

    A trusted path that was missing when the digests were taken is adopted at the first verify, so verifying
    at the first job would let an edit in the agent's first turn become the baseline. The host passes the
    files it installed (the evaluator, the prompts), not the whole judge directory: that also holds the
    ledger and the results the host writes as it goes.
    """
    run = tmp_path / "run"
    seen: list[tuple] = []

    class Recording:
        level = "none"
        kill = None

        def preflight(self, cfg):  # noqa: ANN001, ANN202
            seen.append(("preflight",))

        def verify_trusted(self, paths):  # noqa: ANN001, ANN202
            eval_dir = run / "judge" / "eval"
            seen.append(("verify", [Path(p) for p in paths], eval_dir.is_dir() and any(eval_dir.iterdir())))

        def session(self, cfg):  # noqa: ANN001, ANN202
            import contextlib

            return contextlib.nullcontext()

        def agent_permission(self):  # noqa: ANN201
            return None

        def job(self, spec):  # noqa: ANN001, ANN202
            raise AssertionError("no job is started")

        def copy_out(self, root, rel, into):  # noqa: ANN001, ANN202
            raise AssertionError("nothing is copied")

        def provenance(self):  # noqa: ANN201
            return {"level": "none"}

    host = make_host(
        run,
        {"cards": [0], "fake_gpus": True},
        {"allow_any_format": True},
        isolation=Recording(),
    )
    host.start()
    try:
        assert seen, "the level's verify never ran"
        kind, paths, installed = seen[0]
        assert kind == "verify", "the files must be verified before the launch checks and any turn"
        assert installed, "the evaluator was not installed before the check"
        assert any(Path(p).is_relative_to(run / "judge" / "eval") for p in paths)  # the scorer's files
    finally:
        host.stop()
