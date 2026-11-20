"""Startup baseline selection uses the judge's admission and scoring rules before exposing the agent RPC."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qda_agent import progress
from qda_agent.config import RunConfig
from qda_agent.judge.config import JudgeConfig, JudgeSettings, Start
from qda_agent.judge.startup import BASELINE_REFERENCE
from qda_agent.sched import FakeRun, Refused, State
from test_judge import KL_RESULT, PY, Rig, ckpt, evaluator


def rig(tmp_path, **settings):
    baselines = [ckpt(tmp_path / "higher", b"high"), ckpt(tmp_path / "lower", b"low")]
    return Rig(
        tmp_path,
        baselines=baselines,
        qda_cmd=[PY],
        evaluator=evaluator(tmp_path),
        select_candidate0=True,
        **settings,
    )


def test_start_orders_baselines_and_records_the_judges_own_readings(tmp_path):
    r = rig(tmp_path, final_rules=["handin"], candidate0_build_s=10)
    higher, lower = r.host.cfg.baselines
    r.host.sched.cfg.binds = {r.work / "baselines" / "0": higher, r.work / "baselines" / "1": lower}
    r.host.cfg.kl_cmd = ["score", "{ckpt}", "{out}", "{config}"]
    configs = []

    def score(argv):
        _, path, out, config = argv
        value = 0.04 if path == str(higher) else 0.02
        configs.append(json.loads(Path(config).read_text()))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({**KL_RESULT, "kl": value}))

    r.be.default.effect = score
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert (r.host.cfg.baselines, r.host.cfg.candidate0_kl) == ([lower, higher], 0.02)
        assert [cfg["reference"] for cfg in configs] == [str(r.host.cfg.kl_reference())] * 2
        assert r.host.sched.cfg.binds[r.work / "baselines" / "0"] == lower
        assert r.host.handins() == []
        r.host.write_final(r.run / "FINAL.json")
        result = json.loads((r.run / "FINAL.json").read_text())
        assert result["baseline"] == str(lower)
        assert [row["metrics"]["kl"] for row in result["baseline_readings"]] == [0.02, 0.04]
        records = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
        assert {row["dir"] for row in records if row["valid"]} == {str(higher), str(lower)}
    finally:
        r.host.stop()


def test_candidate0_is_ineligible_with_its_reason_in_the_ledger_and_the_progress_line(tmp_path):
    """A baseline is never an eligible entry; its valid reading still says why, wherever it is shown."""
    r = rig(tmp_path, final_rules=["handin"], candidate0_build_s=10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        records = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
        final = {row["id"]: row for row in records}
        assert {(row["valid"], row["eligible"]) for row in final.values()} == {(True, False)}
        assert all(row["reason"] == BASELINE_REFERENCE for row in final.values())
        view = progress.Progress(r.run, now=r.clock.now())
        changed, _ = view.read()  # the progress line is each ledger entry's newest line
        assert {e.id for e in changed} == {"b0", "b1"}
        for line in view.entry_lines():
            assert ": ineligible: KL 0.0712" in line and line.endswith(f"; {BASELINE_REFERENCE}")
    finally:
        r.host.stop()


def test_a_refused_second_baseline_blocks_start_before_any_reading(tmp_path):
    r = rig(tmp_path)
    bad = r.host.cfg.baselines[1]
    config = json.loads((bad / "config.json").read_text())
    config["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = 2
    (bad / "config.json").write_text(json.dumps(config))
    with pytest.raises(Refused, match=r"baselines\[1\].*lower"):
        r.host.start()
    assert r.seen == {} and r.host.server is None


def test_second_baseline_must_keep_the_source_head_dtype_before_any_reading(tmp_path):
    from qda_agent.judge.checks import format_problem
    from safetensors_kit import anchor_dir, packed_tensors, write_safetensors

    anchor = anchor_dir(tmp_path / "source")
    config = json.loads((anchor / "config.json").read_text())
    config["tie_word_embeddings"] = False
    (anchor / "config.json").write_text(json.dumps(config))
    write_safetensors(anchor / "model.safetensors", {"lm_head.weight": ("F32", (16, 8), None)})
    r = rig(tmp_path, anchor=anchor)
    for i, baseline in enumerate(r.host.cfg.baselines):
        config = json.loads((baseline / "config.json").read_text())
        config["tie_word_embeddings"] = False
        config["quantization_config"]["ignore"] = ["lm_head"]
        (baseline / "config.json").write_text(json.dumps(config))
        tensors = packed_tensors(b"high" if i == 0 else b"low")
        tensors["lm_head.weight"] = ("F32" if i == 0 else "BF16", (16, 8), None)
        write_safetensors(baseline / "model.safetensors", tensors)
    assert format_problem(r.host.cfg.baselines[0], anchor=anchor) is None
    with pytest.raises(Refused, match=r"baselines\[1\].*lower.*lm_head"):
        r.host.start()
    assert not r.seen and r.host.server is None and r.host.cfg.candidate0_kl is None


@pytest.mark.parametrize(
    "reading",
    [
        {},
        {"kl": float("nan")},
        {"kl": 2.0},
        {"suite": "kl", "metrics": {"kl": 0.01}},
        {"kl": 0.001},
        {**KL_RESULT, "tokens": 0},
        {**KL_RESULT, "tokens": 0.5},
        {**KL_RESULT, "error": "scoring failed"},
        {"suite": "kl", "metrics": {"kl": 0.001}, "higher_is_better": {"kl": False}, "error": "failed"},
        {**KL_RESULT, "per_seq": {"p1#0": None}},
        {**KL_RESULT, "per_seq": {"p1#0": {**KL_RESULT["per_seq"]["p1#0"], "tokens": 0}}},
    ],
)
def test_an_incomplete_or_refused_reading_blocks_start_and_preserves_inputs(tmp_path, reading):
    r = rig(tmp_path)
    r.be.default.effect = lambda argv: Path(argv[2]).write_text(json.dumps(reading))
    with pytest.raises(Refused, match=r"baselines\[0\].*higher"):
        r.host.start()
    assert r.host.server is None
    assert all((path / "model.safetensors").is_file() for path in r.host.cfg.baselines)


def test_selection_off_keeps_the_existing_final_bytes_and_does_not_score(tmp_path):
    r = Rig(
        tmp_path,
        baselines=[ckpt(tmp_path / "base")],
        qda_cmd=[PY],
        evaluator=evaluator(tmp_path),
        final_rules=["handin"],
        candidate0_kl=0.03,
        candidate0_build_s=10,
    )
    before = r.host.final().model_dump_json(indent=1, exclude={"baseline_readings"})
    r.host.start()
    try:
        r.host.write_final(r.run / "FINAL.json")
        assert (r.run / "FINAL.json").read_text() == before + "\n"
        assert r.seen == {} and r.host.cfg.candidate0_kl == 0.03
    finally:
        r.host.stop()


def test_automatic_and_manual_candidate0_readings_cannot_be_combined():
    with pytest.raises(ValueError, match=r"select_candidate0.*candidate0_kl"):
        JudgeSettings(select_candidate0=True, candidate0_kl=0.02)


def test_a_reading_that_finishes_after_the_estimate_is_not_cancelled(tmp_path):
    """The defect card's numbers: estimate 89.56 s, the reading needs 94 s.

    `read()` used to cancel the job at the bare estimate, while the reading also starts an engine
    (process start, weight load, torch.compile, CUDA graphs, warm-up) that the estimate does not count.
    The wait is now `max(2 * estimate, estimate + 300 s)`, which is 389.56 s here, so a reading that ends at
    94 s is kept and folded.
    """
    r = rig(tmp_path, final_rules=["handin"], candidate0_build_s=10, kl_score_s=89.56)
    r.be.default = FakeRun(94, effect=r.effect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert r.host.cfg.candidate0_kl == KL_RESULT["kl"]
        assert [j.state for j in r.host.sched.list("judge")] == [State.done, State.done]
    finally:
        r.host.stop()


def test_a_reading_the_first_wait_cancels_is_read_on_the_longer_retry(tmp_path):
    """A large model's reading outgrows `kl_score_s`: 27B W4A16's baseline took ~500 s of job time against a
    497 s wait and was cancelled 3 s before it ended, and a retry with the same wait could only be cancelled
    again. The retry waits twice as long (779 s here), so a 400 s reading is read on it."""
    r = rig(tmp_path, final_rules=["handin"], candidate0_build_s=10, kl_score_s=89.56)
    r.be.default = FakeRun(400, effect=r.effect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert r.host.cfg.candidate0_kl == KL_RESULT["kl"]
        jobs = r.host.sched.list("judge")
        assert [j.name for j in jobs][:2] == ["judge-b0", "judge-b0-r1"]
        assert [j.state for j in jobs][:2] == [State.cancelled, State.done]
    finally:
        r.host.stop()


def test_a_reading_the_watchdog_cancels_is_retried_once_and_named(tmp_path):
    """A reading the watchdog still cancels says so, and is tried again before the run is refused.

    Both attempts outlive their waits here (389.56 s, then twice that). The refusal must name the judge's
    own watchdog, its waits and the setting to raise, and say the checkpoint was not read -- never the
    cancelled job's line, which reads as a verdict on the checkpoint.
    """
    r = rig(tmp_path, final_rules=["handin"], candidate0_build_s=10, kl_score_s=89.56)
    r.be.default = FakeRun(800, effect=r.effect)
    with pytest.raises(Refused) as error, ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    text = str(error.value)
    assert "reading watchdog cancelled the job after 390 s, then 779 s" in text
    assert "on 2 attempt(s): the checkpoint was not read" in text
    assert "judge.kl_score_s (89.56 s) allows: raise it" in text
    assert "cancelled (rc None)" not in text
    assert [j.name for j in r.host.sched.list("judge")] == ["judge-b0", "judge-b0-r1"]


def _two_start_rig(tmp_path, **settings):
    """A rig whose two baselines are labelled starts: `higher` (0.04) and `lower` (0.02)."""
    higher = ckpt(tmp_path / "higher", b"high")
    lower = ckpt(tmp_path / "lower", b"low")
    starts = [
        {"name": "awq", "path": higher, "origin": "own"},
        {"name": "gptq", "path": lower, "origin": "own"},
    ]
    r = Rig(
        tmp_path,
        baselines=[higher, lower],
        qda_cmd=[PY],
        evaluator=evaluator(tmp_path),
        starts=starts,
        **settings,
    )

    def score(argv):
        _, path, out = argv
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({**KL_RESULT, "kl": 0.04 if path == str(higher) else 0.02}))

    r.be.default.effect = score
    return r, higher, lower


def test_every_start_is_a_candidate_0_and_the_lowest_kl_is_the_bar(tmp_path):
    """`judge.starts` labels the pre-built starts; the bar is the lowest valid KL, and FINAL names each."""
    r, _, lower = _two_start_rig(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert (r.host.cfg.baselines[0], r.host.cfg.candidate0_kl) == (lower, 0.02)
        r.host.write_final(r.run / "FINAL.json")
        result = json.loads((r.run / "FINAL.json").read_text())
        assert result["candidate0_bar"] == str(lower)
        rows = result["baseline_readings"]
        assert [(row["name"], row["origin"], row["metrics"]["kl"]) for row in rows] == [
            ("gptq", "own", 0.02),
            ("awq", "own", 0.04),
        ]
        assert rows[0]["bar"] is True and all("bar" not in row for row in rows[1:])  # the bar is named
        records = [json.loads(line) for line in (r.host.jdir / "ledger.jsonl").read_text().splitlines()]
        bars = [row for row in records if row.get("bar")]
        assert [(row["name"], row["metrics"]["kl"]) for row in bars] == [("gptq", 0.02)]
    finally:
        r.host.stop()


def test_a_candidate_between_the_starts_kls_does_not_beat_candidate_0(tmp_path):
    """The bar is the BEST start: a candidate below the worse start but above the better one is no result."""
    from test_judge import kl_effect

    r, _, lower = _two_start_rig(tmp_path, final_rules=["handin"], candidate0_build_s=10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert r.host.cfg.candidate0_kl == 0.02  # the bar, not the first start (0.04)
        ckpt(r.work / "a")
        kl_effect(r, 0.03)
        assert r.call("kl", "a").exit == 0
        r.settle()
        r.host.space_refusals.append({"kind": "space_refusal", "build": "out/a"})  # the path that reads the bar
        final = r.host.final(generation_checks=False)
        assert final.choice is None and final.baseline == lower
        assert final.chosen_by == "fallback_candidate0"
        assert "no build beat candidate 0 (KL 0.02)" in final.reason
    finally:
        r.host.stop()


def test_a_single_start_keeps_todays_behaviour(tmp_path):
    """One labelled start is today's single candidate 0: it is read, it is the fallback, and it is named."""
    base = ckpt(tmp_path / "base", b"base")
    r = Rig(
        tmp_path,
        baselines=[base],
        qda_cmd=[PY],
        evaluator=evaluator(tmp_path),
        starts=[Start(name="gptq", path=base, origin="own")],
    )

    def score(argv):
        Path(argv[2]).write_text(json.dumps({**KL_RESULT, "kl": 0.03}))

    r.be.default = FakeRun(30, effect=score)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(r.host.start).result()
    try:
        assert (r.host.cfg.baselines, r.host.cfg.candidate0_kl) == ([base], 0.03)
        final = r.host.final(generation_checks=False)
        assert final.rule == "handin" and final.choice is None and final.baseline == base
        assert final.chosen_by == "fallback_candidate0" and final.candidate0_bar == base
        (reading,) = final.baseline_readings
        assert (reading.name, reading.origin, reading.metrics["kl"]) == ("gptq", "own", 0.03)
    finally:
        r.host.stop()


def _run_config(tmp_path, *, fmt="w4a16", starts=(), baselines=(), run_dir=None):
    return RunConfig(
        contamination="none",  # type: ignore[arg-type]
        model="org/m",
        format=fmt,
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run_dir or tmp_path / "run",
        baselines=list(baselines),
        judge={"starts": list(starts)},  # type: ignore[arg-type]
    )


def test_a_start_origin_is_own_or_gsq(tmp_path):
    path = tmp_path / "run" / "starts" / "gptq"
    with pytest.raises(ValueError, match="'own' or 'gsq'"):
        JudgeConfig(
            contamination="none",
            baselines=[path],
            starts=[Start.model_validate({"name": "gptq", "path": path, "origin": "vendor"})],
        )


def test_a_gsq_start_is_w3a16_only(tmp_path):
    path = tmp_path / "starts" / "gsq"
    path.mkdir(parents=True)
    start = Start(name="gsq", path=path, origin="gsq")
    with pytest.raises(ValueError, match="gsq start is only allowed on a W3A16 row"):
        _run_config(tmp_path, starts=[start], baselines=[path])
    ok = _run_config(tmp_path, fmt="w3a16", starts=[start], baselines=[path])
    assert [(s.name, s.origin) for s in ok.judge.starts] == [("gsq", "gsq")]


def test_a_start_inside_the_run_dir_is_refused_like_any_baseline(tmp_path):
    """The agent's jobs write the run dir, so a baseline (and so a start) may not lie there."""
    inside = tmp_path / "run" / "starts" / "gptq"
    start = Start(name="gptq", path=inside, origin="own")
    with pytest.raises(ValueError, match="inside run_dir"):
        _run_config(tmp_path, starts=[start], baselines=[inside])


def test_starts_label_every_baseline_once(tmp_path):
    a, b = tmp_path / "starts" / "a", tmp_path / "starts" / "b"
    with pytest.raises(ValueError, match="label every baselines path once"):
        JudgeConfig(contamination="none", baselines=[a, b], starts=[Start(name="gptq", path=a, origin="own")])
    twice = [
        Start(name="gptq", path=a, origin="own"),
        Start(name="gptq", path=b, origin="own"),
    ]
    with pytest.raises(ValueError, match="names one label twice"):
        JudgeConfig(contamination="none", baselines=[a, b], starts=twice)


def test_starts_read_candidate_0_so_candidate0_kl_cannot_also_be_set(tmp_path):
    a = tmp_path / "starts" / "gptq"
    with pytest.raises(ValueError, match="drop judge.candidate0_kl"):
        JudgeSettings(starts=[Start(name="gptq", path=a, origin="own")], candidate0_kl=0.02)


def test_a_start_name_is_a_build_method_qda_quant_knows(tmp_path):
    """A release, a publisher or any other word is refused: the name is a method qda_quant builds with."""
    a = tmp_path / "starts" / "gptq"
    with pytest.raises(ValueError, match="build method qda_quant knows"):
        Start(name="some-release-v1", path=a, origin="own")
