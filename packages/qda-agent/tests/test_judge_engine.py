"""A llama.cpp search's reading: the scorer config, its digests, and the engine floor the search states."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from qda_agent.judge import JudgeConfig, JudgeHost
from qda_agent.judge.config import LlamaCpp
from qda_agent.judge.freeze import seal
from qda_agent.judge.ledger import Entry
from qda_agent.rpc import Request
from qda_agent.sched import FakeBackend, FakeClock, FakeRun, Refused, SchedConfig
from qda_kl.eval.suites import SUITES, Layout
from qda_quant.gguf.llama_cpp import PINNED
from test_judge import KL_RESULT, ckpt

FLOOR = 0.000273
BUDGET = 17_548_181_504
#: The refusal text for a `llama.cpp` search's suite job, byte for byte.
OLD_REFUSAL = (
    "a llama.cpp search runs no suite job: a suite reads a checkpoint vLLM loads; "
    "`qda-kl score` (the default kl_cmd) and the long-generation check take a .gguf"
)


@pytest.mark.parametrize("n_files", [0, 1, 2])
def test_the_reading_config_selects_exactly_one_frozen_gguf(monkeypatch, tmp_path, n_files):
    from qda_agent.judge.jobs import job_config

    paths = _paths(tmp_path)
    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    monkeypatch.setattr("qda_quant.gguf.llama_cpp.commit_of", lambda root: PINNED)
    host = SimpleNamespace(cfg=_config(tmp_path, paths=paths), jdir=tmp_path / "judge")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "config.json").write_text("{}")
    for i in range(n_files):
        (frozen / f"model-{i}.gguf").write_bytes(b"GGUF fixture")
    entry = Entry(id="k1", kind="scored", src="candidate", dir=frozen, digest="fixture", submitted=0)
    if n_files != 1:
        with pytest.raises(Refused, match="(?i)gguf"):
            job_config(host, entry, tmp_path / "out")
    else:
        config = job_config(host, entry, tmp_path / "out")
        assert Path(json.loads(config.read_text())["checkpoint"]) == frozen / "model-0.gguf"


def _checkout(root: Path) -> Path:
    """A stand-in pinned llama.cpp checkout: the pin's marker and the `gguf-py` reader."""
    root.mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# pin marker\n")
    (root / "gguf-py").mkdir()
    return root


def _paths(tmp_path: Path) -> LlamaCpp:
    bridge = tmp_path / "probe.so"
    bridge.write_bytes(b"bridge")
    server = tmp_path / "llama-server"
    server.write_bytes(b"binary")
    return LlamaCpp(bridge=bridge, llama_cpp=_checkout(tmp_path / "llama.cpp"), server=server)


def _ref(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    """A reference directory the host's calibration index reads its prompt rows from."""
    root = tmp_path / name
    root.mkdir(parents=True)
    prompts = root / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return root


def _config(tmp_path: Path, paths: LlamaCpp | None = None, **fields) -> JudgeConfig:
    return JudgeConfig(
        contamination="none",
        engine="llama.cpp",
        budget_bytes=BUDGET,
        engine_floor=FLOOR,
        allow_any_format=True,
        llamacpp=paths or _paths(tmp_path),
        kl_ref=str(_ref(tmp_path, "eval", [{"id": "r1", "prompt": "An eval prompt about Paris"}])),
        kl_holdout_ref=str(_ref(tmp_path, "holdout", [{"id": "h1", "prompt": "A held-out prompt"}])),
        **fields,
    )


def _settle(host, clock) -> None:
    for _ in range(20):
        host.tick()
        clock.advance(30)
    host.tick()


def _take(host, kind, arg, cwd):
    """Enter a taken entry without admission: a llama.cpp candidate is refused until its GGUF check lands."""
    dst = host.jdir / "frozen" / "k1"
    shutil.copytree(Path(cwd) / arg, dst)
    e = Entry(id="k1", kind=kind, src=arg, dir=dst, digest=seal(dst), submitted=host.clock.now())
    host.entries.append(e)
    return e


def _reading(
    monkeypatch,
    tmp_path: Path,
    score: dict | None = None,
    paths: LlamaCpp | None = None,
    settle: bool = True,
    files: dict[str, bytes] | None = None,
    replies: list | None = None,
    cmd: list[str] | None = None,
    suite: dict | None = None,
):
    """A host whose fake `qda-kl score CONFIG.yaml` writes `score` as its `score.json`."""
    paths = paths or _paths(tmp_path)
    score = (
        score
        if score is not None
        else {
            **KL_RESULT,
            "engine": "llama.cpp",
            "llama_cpp": str(paths.llama_cpp),
            "model_sha256": hashlib.sha256(b"gguf").hexdigest(),
            "bridge_sha256": hashlib.sha256(paths.bridge.read_bytes()).hexdigest(),
        }
    )
    # the pin reader at config load, and the one jobs imports when it queues and folds
    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    monkeypatch.setattr("qda_quant.gguf.llama_cpp.commit_of", lambda root: PINNED)
    monkeypatch.setattr("qda_agent.judge.commands.take", _take)
    clock = FakeClock()
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)

    def effect(argv):
        if Path(argv[-1]).suffix != ".yaml":
            return  # a `{ckpt}`-style job: what the judge handed it is the assertion
        config = Path(argv[-1])
        out = Path(json.loads(config.read_text())["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.json").write_text(json.dumps(score))

    host = JudgeHost(
        run,
        SchedConfig(cards=[0], budget_s=3600, fake_gpus=True),
        _config(tmp_path, paths=paths, kl_cmd=cmd or ["fake-qda-kl", "{config}"], suite=suite),
        clock,
        FakeBackend(clock, default=FakeRun(30, effect=effect)),
        autotick=False,
    )
    ckpt(work / "a")
    for name, data in ({"model.gguf": b"gguf"} if files is None else files).items():
        (work / "a" / name).write_bytes(data)
    reply = host.handle(Request("kl", ["a"], str(work)))
    if replies is not None:
        replies.append(reply)
    if settle:
        _settle(host, clock)
    return host, clock


def test_a_llama_cpp_job_gets_the_single_gguf_not_the_entry_directory(monkeypatch, tmp_path):
    """The `{ckpt}` slot of a llama.cpp job names the frozen `.gguf`, never the directory vLLM would misread."""
    host, _ = _reading(monkeypatch, tmp_path, settle=False, cmd=["fake-ckpt-reader", "{ckpt}", "{out}"])
    (job,) = host.sched.list()
    assert job.argv[1].endswith("/frozen/k1/model.gguf"), "the job must be handed one file, not a directory"


class _Suite(Layout):
    """A suite with no settings, so a `{suite}` command template can be configured for the refusal test."""

    name: ClassVar[str] = "engine_suite"


def test_a_llama_cpp_search_loads_with_the_default_final_rules(monkeypatch, tmp_path):
    """No default rule runs a suite: a llama.cpp search loads, and only suite-based rules are refused."""
    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    assert [rule.name for rule in _config(tmp_path).final_rules] == ["handin"]


def test_a_suite_job_is_refused_for_a_llama_cpp_search_and_runs_no_vllm(monkeypatch, tmp_path):
    """A suite whose scorer is not llama.cpp's would answer the `.gguf` with vLLM, so the judge refuses it."""
    if _Suite.name not in SUITES:
        SUITES.register(_Suite.name, _Suite)
    replies: list = []
    host, _ = _reading(
        monkeypatch,
        tmp_path,
        replies=replies,
        suite={_Suite.name: {}},
        cmd=["fake-suite", "{suite}", "{ckpt}", "{out}"],
    )
    assert host.entries == [] and host.sched.list() == []
    assert "no suite job" in replies[0].stderr and "vLLM loads" in replies[0].stderr


def test_a_non_kl_suite_is_refused_with_the_existing_diagnostic(monkeypatch, tmp_path):
    """A suite job for a `llama.cpp` search keeps its established refusal text, byte for byte."""
    if _Suite.name not in SUITES:
        SUITES.register(_Suite.name, _Suite)
    replies: list = []
    host, _ = _reading(
        monkeypatch,
        tmp_path,
        replies=replies,
        suite={_Suite.name: {}},
        cmd=["fake-suite", "{suite}", "{ckpt}", "{out}"],
    )
    assert host.entries == [] and host.sched.list() == []
    assert replies[0].stderr == "kl: refused: a: " + OLD_REFUSAL + "\n"


def test_a_llama_cpp_reading_scores_the_single_frozen_gguf_file(monkeypatch, tmp_path):
    """`qda-kl score` decodes one `.gguf` and reads it with vLLM: the config names the file and the run's layout."""
    host, _ = _reading(monkeypatch, tmp_path, score=KL_RESULT)
    config = json.loads((host.jdir / "results" / "k1.yaml").read_text())
    assert config["checkpoint"].endswith("/k1/model.gguf")
    assert config["scorer"] == {"topk": {"tp": 1, "serve_args": {}}}
    assert "snapshot" not in config  # no anchor: `qda-kl score` decodes against the reference's model
    (entry,) = host.entries
    assert entry.valid is True


def test_a_llama_cpp_reading_decodes_against_the_anchor(monkeypatch, tmp_path):
    """With an anchor the judge names it as the snapshot the `.gguf` was converted from."""
    from qda_agent.judge.jobs import job_config

    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    cfg = _config(tmp_path).model_copy(update={"anchor": anchor})
    host = SimpleNamespace(cfg=cfg, jdir=tmp_path / "judge")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "model.gguf").write_bytes(b"GGUF fixture")
    entry = Entry(id="k1", kind="scored", src="candidate", dir=frozen, digest="fixture", submitted=0)
    config = json.loads(job_config(host, entry, tmp_path / "out").read_text())
    assert config["snapshot"] == str(anchor) and config["checkpoint"] == str(frozen / "model.gguf")


def test_a_llama_cpp_gencheck_hands_vllm_the_gguf_and_the_original(monkeypatch, tmp_path):
    """A `.gguf` long-generation check is `qda-kl gencheck` on the file: no llama-server in its config."""
    from qda_agent.judge.jobs import job_config

    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    cfg = _config(tmp_path).model_copy(update={"anchor": anchor})
    host = SimpleNamespace(
        cfg=cfg,
        jdir=tmp_path / "judge",
        original_cache=tmp_path / "judge" / "gencheck-original",
        install_prompts=lambda: None,
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "model.gguf").write_bytes(b"GGUF fixture")
    entry = Entry(id="g1", kind="gencheck", src="candidate", dir=frozen, digest="fixture", submitted=0)
    config = json.loads(job_config(host, entry, tmp_path / "out").read_text())
    assert config["checkpoint"] == str(frozen / "model.gguf") and config["original"] == str(anchor)
    assert "llamacpp" not in config and entry.model_dump().get("provenance") is None


def test_a_llama_cpp_gencheck_without_a_local_original_is_refused_at_load(monkeypatch, tmp_path):
    """Each .gguf is decoded against the original: a Hub id with no `judge.anchor` is refused at config load,
    naming `judge.anchor`, the setting the user owns (the job's `original:` is the judge's)."""
    monkeypatch.setattr("qda_agent.judge.config.commit_of", lambda root: PINNED)
    commands = ["gencheck"]
    for d in ("a", "b", "c"):
        (tmp_path / d).mkdir()
    with pytest.raises(ValueError, match="set judge.anchor to the local HF snapshot"):
        _config(tmp_path / "a", commands=commands, model="Qwen/Qwen3-0.6B")
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "config.json").write_text("{}")
    hub = "Qwen/Qwen3-0.6B"
    assert _config(tmp_path / "b", commands=commands, model=hub, anchor=anchor).anchor == anchor
    assert _config(tmp_path / "c", model=hub).anchor is None  # no gencheck command: nothing to decode


@pytest.mark.parametrize("files,count", [({}, 0), ({"a.gguf": b"1", "b.gguf": b"2"}, 2)])
def test_a_llama_cpp_reading_needs_exactly_one_frozen_gguf(monkeypatch, tmp_path, files, count):
    replies: list = []
    host, _ = _reading(monkeypatch, tmp_path, files=files, replies=replies)
    assert host.entries == []
    assert f"holds {count} .gguf files" in replies[0].stderr
    assert host.final(generation_checks=False).choice is None


@pytest.mark.parametrize(
    "failure,reason",
    [
        ({"per_seq": {}}, "no per-sequence readings"),
        ({"error": "scoring failed"}, "failed: scoring failed"),
    ],
)
def test_an_incomplete_gguf_reading_is_not_admitted(monkeypatch, tmp_path, failure, reason):
    host, _ = _reading(monkeypatch, tmp_path, score={**KL_RESULT, **failure})
    (entry,) = host.entries
    assert entry.valid is False and reason in entry.reason
    assert entry.metrics == {} and host.final(generation_checks=False).choice is None


def test_a_vllm_reading_keeps_no_engine_floor(tmp_path):
    clock = FakeClock()
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)

    def effect(argv):
        out = Path(argv[-1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.json").write_text(json.dumps(KL_RESULT))

    host = JudgeHost(
        run,
        SchedConfig(cards=[0], budget_s=3600, fake_gpus=True),
        JudgeConfig(
            contamination="none",
            kl_ref=str(_ref(tmp_path, "eval", [{"id": "r1", "prompt": "An eval prompt about Paris"}])),
            kl_holdout_ref=str(_ref(tmp_path, "holdout", [{"id": "h1", "prompt": "A held-out prompt"}])),
            kl_cmd=["fake-qda-kl", "{out}"],
            allow_any_format=True,
        ),
        clock,
        FakeBackend(clock, default=FakeRun(30, effect=effect)),
        autotick=False,
    )
    ckpt(work / "a")
    host.handle(Request("kl", ["a"], str(work)))
    _settle(host, clock)
    assert "engine_floor" not in host.final().model_dump()


def test_the_final_record_states_the_configured_engine_floor(monkeypatch, tmp_path):
    host, _ = _reading(monkeypatch, tmp_path)
    final = host.final()
    assert final.engine_floor == FLOOR and json.loads(final.model_dump_json())["engine_floor"] == FLOOR


def test_task_md_states_the_engine_floor_and_its_meaning():
    from datetime import UTC, datetime

    from qda_agent.prompt import task_statement

    args = dict(
        model="org/a",
        fmt="gguf_q4_k",
        n_cards=1,
        start=datetime(2026, 11, 8, tzinfo=UTC),
        deadline=datetime(2026, 11, 8, 3, tzinfo=UTC),
        baselines=[],
        metric="kl",
        final_rules="",
        budget_bpw=None,
        format_menu=["Q4_K"],
        require_build_receipt=False,
        max_kl=0.5,
        gen_max_tokens=8192,
        eval_max_tokens=32768,
    )
    text = task_statement(engine="llama.cpp", budget_bytes=BUDGET, engine_floor=FLOOR, **args)
    assert f"Engine floor: the BF16 GGUF, decoded and scored like every candidate, has KL {FLOOR!r}" in text
    assert "Candidate KL includes this floor" in text
    assert "Engine floor" not in task_statement(**{**args, "fmt": "w4a16-sym-g128", "format_menu": ["w4a16"]})
