"""The `gencheck` command: a long-generation check of a frozen copy, run by the judge, counts only.

The judge side is tested against the contract with a stand-in command (`fixtures/standin_gencheck.py`), which
writes a contract-shaped `gencheck.json` plus texts, a prompt and a host path the agent must never see.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from qda_agent.commands import NAMES, install_bin
from qda_agent.judge import JudgeConfig
from qda_agent.judge.ledger import INCOMPLETE, gen_report
from qda_agent.sched import State
from test_judge import PY, Rig, ckpt, evaluator, make_host

FAKE = Path(__file__).parent / "fixtures" / "standin_gencheck.py"
SECRETS = ("PROMPT-SECRET", "ANSWER-SECRET", "passwd", "texts", "/")


def _sources() -> list[Path]:
    """The three packages' source roots, as a run's `qda_env.PYTHONPATH` names them."""
    from qda_agent import __file__ as agent_file
    from qda_kl import __file__ as kl_file
    from qda_quant import __file__ as quant_file

    return [Path(p).parents[1] for p in (quant_file, kl_file, agent_file)]


class GenRig(Rig):
    """The judge rig with `gencheck` on and the stand-in as its command."""

    def __init__(self, tmp_path, **judge):
        judge = {
            "commands": [*NAMES, "gencheck"],
            "gencheck_cmd": [PY, str(FAKE), "{config}"],
            "gencheck_expected_s": 600,
            "model": "org/original-model",
            **judge,
        }
        super().__init__(tmp_path, **judge)

    def effect(self, argv):
        if argv[1:2] == [str(FAKE)]:
            subprocess.run(argv, check=False)
        else:
            super().effect(argv)

    def config(self, eid):
        return json.loads((self.run / "judge" / "results" / f"{eid}.yaml").read_text())


def test_gencheck_runs_on_a_frozen_copy_and_shows_counts_only(tmp_path):
    r = GenRig(tmp_path, gencheck_max_tokens=4096, gencheck_per_domain=8, gencheck_cards=2)
    ckpt(r.work / "out" / "a")
    rep = r.call("gencheck", "out/a")
    assert rep.exit == 0 and rep.stdout == "g1 out/a: generating, asked 13:46:40 UTC; `gencheck show g1`\n"
    cfg = r.config("g1")
    jdir = r.run / "judge"
    assert cfg["checkpoint"] == str(jdir / "frozen" / "g1") and cfg["original"] == "org/original-model"
    assert (cfg["max_tokens"], cfg["per_domain"], cfg["tp"], cfg["prompts"]) == (4096, 8, 2, "builtin")
    assert cfg["original_cache"] == str(jdir / "gencheck-original") and cfg["out"] == str(
        jdir / "results" / "g1"
    )
    (job,) = r.host.sched.list(origin="judge")
    assert job.ncards == 2 and job.expected_s == 600
    r.settle()
    shown = r.call("gencheck", "show", "g1").stdout
    assert shown == (
        "g1 out/a: done (asked 13:46:40 UTC, finished 13:47:10 UTC): checkpoint vs original, max 4096 tokens; "
        "code (8): cut 0 vs 0, loops 0 vs 0, tokens 812 vs 790; "
        "math (8): cut 3 vs 1, loops 1 vs 0, tokens 5123 vs 4870\n"
    )
    assert r.call("gencheck", "list").stdout == shown
    for s in [*SECRETS[:-1], str(tmp_path)]:
        assert s not in shown
    # a copy nothing reads stays until a pass near the free-space floor reclaims it
    assert (jdir / "frozen" / "g1").exists()
    assert "PROMPT-SECRET" in (jdir / "results" / "g1" / "texts" / "checkpoint.jsonl").read_text()


def test_the_original_is_generated_once_per_run(tmp_path):
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    ckpt(r.work / "b", weights=b"w2")
    r.call("gencheck", "a")
    r.settle()
    r.call("gencheck", "b")
    r.settle()
    assert r.config("g1")["original_cache"] == r.config("g2")["original_cache"]
    assert "done" in r.call("gencheck", "show", "g2").stdout


def test_the_original_is_the_anchor_when_set(tmp_path):
    anchor = ckpt(tmp_path / "anchor")
    r = GenRig(tmp_path, anchor=anchor)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    assert r.config("g1")["original"] == str(anchor)


def test_a_failed_check_says_so_and_a_bad_id_is_refused(tmp_path):
    r = GenRig(tmp_path, gencheck_cmd=["false", "{config}"])
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert r.call("gencheck", "show", "g1").stdout.startswith("g1 a: failed (asked ")
    assert (
        "finished" in r.call("gencheck", "show", "g1").stdout
        and "the long-generation check job" in r.call("gencheck", "show", "g1").stdout
    )
    assert r.call("gencheck", "show", "k1").exit == 2
    assert "usage: gencheck" in r.call("gencheck").stderr


def test_gencheck_is_not_a_hand_in_and_never_counts_at_the_end(tmp_path):
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert r.host.handins() == [] and r.host.final().rule != "last_valid"
    assert r.call("handin", "status").stdout.startswith("nothing submitted yet")
    assert r.call("kl", "list").stdout == "no readings yet\n"


def test_a_gencheck_queued_before_the_cutoff_finishes_in_the_drain(tmp_path):
    """TASK.md promises that checks already queued still finish: the cutoff keeps them, the drain ends them."""
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    r.call("gencheck", "a")  # queued before the cutoff
    r.host.tick()
    r.host.cutoff()
    (job,) = r.host.sched.list(origin="judge")
    assert job.state is State.running  # not cancelled at the cutoff
    assert r.host.drain(900)  # the drain waits for it, within the drain window
    assert r.call("gencheck", "show", "g1").stdout.startswith("g1 a: done")
    assert "deadline" in r.call("gencheck", "a").stderr  # a new check is still refused after the cutoff


def test_a_gencheck_that_cannot_finish_in_the_drain_window_is_cut_not_failed(tmp_path):
    """A check the window cannot hold is recorded CUT ('cut by drain window'): no verdict on the checkpoint."""
    r = GenRig(tmp_path, budget_s=6000, gencheck_expected_s=5000, grace_s=900)
    ckpt(r.work / "a")
    r.call("gencheck", "a")  # admitted before the deadline
    r.clock.advance(6000)  # the GPU deadline passes with the check still queued
    r.host.cutoff()
    r.host.tick()
    (job,) = r.host.sched.list(origin="judge")
    assert job.state is State.failed and "drain window" in job.reason
    shown = r.call("gencheck", "show", "g1").stdout
    assert shown.startswith("g1 a: cut") and "cut by drain window" in shown
    assert "failed" not in shown
    assert r.host.drain(1)


def test_a_gencheck_still_running_when_the_window_ends_is_cut(tmp_path):
    """The drain is bounded: a check it cannot finish is recorded cut, not failed and not left generating."""
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    r.be.default.seconds = 5000  # still running when the window ends
    r.call("gencheck", "a")
    r.host.tick()
    r.host.cutoff()
    assert not r.host.drain(1)  # one second of drain: the check cannot finish
    shown = r.call("gencheck", "show", "g1").stdout
    assert shown.startswith("g1 a: cut") and "cut by drain window" in shown
    assert "failed" not in shown


def test_a_baseline_as_is_may_be_checked(tmp_path):
    base = ckpt(tmp_path / "base")
    r = GenRig(tmp_path, baselines=[base])
    ckpt(r.work / "a")  # the same weights as the baseline
    assert r.call("gencheck", "a").stdout.startswith("g1 a: generating")


def test_a_prompt_set_is_copied_into_the_judges_dir(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id": "1", "domain": "math", "prompt": "x"}\n')
    r = GenRig(tmp_path, gencheck_prompts=prompts)
    r.host.install_prompts()
    copy = r.run / "judge" / "gencheck-prompts.jsonl"
    prompts.write_text("changed\n")
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    assert r.config("g1")["prompts"] == str(copy) and copy.read_text().startswith('{"id"')
    assert not copy.stat().st_mode & 0o222


def test_a_linked_prompt_set_is_refused(tmp_path):
    # the prompt file is `static_checks`': the host refuses it before it copies anything
    (tmp_path / "real.jsonl").write_text("{}\n")
    linked = tmp_path / "prompts.jsonl"
    linked.symlink_to(tmp_path / "real.jsonl")
    a_dir = tmp_path / "a-dir"
    a_dir.mkdir()
    for name, prompts in (("linked", linked), ("dir", a_dir)):
        with pytest.raises(ValueError, match="not a regular file"):
            GenRig(tmp_path / name, gencheck_prompts=prompts)


def test_a_prompt_set_is_installed_before_the_first_check(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{}\n")
    r = GenRig(tmp_path, gencheck_prompts=prompts)  # no start(): the first check installs it
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    assert r.config("g1")["prompts"] == str(r.run / "judge" / "gencheck-prompts.jsonl")


def test_judge_jobs_find_the_serving_venvs_programs_first(tmp_path):
    venv = tmp_path / "serve"
    (venv / "bin").mkdir(parents=True)
    r = GenRig(tmp_path, serve_python=venv / "bin" / "python", job_env={"PATH": "/opt/cuda/bin:/usr/bin"})
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.call("kl", "a")
    jobs = {j.name: j for j in r.host.sched.list(origin="judge")}
    for job in ("judge-g1", "judge-k1"):  # the long-generation check gets what a KL reading gets
        assert jobs[job].env["PATH"] == f"{venv / 'bin'}:/opt/cuda/bin:/usr/bin"


def test_gencheck_needs_the_original_model():
    with pytest.raises(ValueError, match="needs the original model"):
        JudgeConfig(contamination="none", commands=[*NAMES, "gencheck"], kl_cmd=["score", "{ckpt}", "{out}"])
    JudgeConfig(
        contamination="none", kl_cmd=["score", "{ckpt}", "{out}"]
    )  # without the command, nothing is needed


def test_gencheck_through_the_socket(tmp_path):
    run, work = tmp_path / "run", tmp_path / "run" / "work"
    model = ckpt(work / "ck")
    host = make_host(
        run,
        {"cards": [0, 1], "budget_s": 3600, "tick_s": 0.05},
        {
            "allow_any_format": True,
            "kl_cmd": ["true", "{ckpt}", "{out}"],
            "gencheck_cmd": [PY, str(FAKE), "{config}"],
            "commands": [*NAMES, "gencheck"],
            "model": str(model),  # the launch check reads its config; the fake command ignores it
            "evaluator": str(evaluator(tmp_path)),
            "qda_cmd": [PY],
            "qda_env": {"PYTHONPATH": os.pathsep.join(map(str, _sources()))},
        },
    )
    bindir = work / "bin"
    install_bin(bindir, run / "rpc.sock", names=(*NAMES, "gencheck"))
    host.start()
    try:

        def cmd(name, *args):
            p = subprocess.run([PY, str(bindir / name), *args], cwd=work, capture_output=True, text=True)
            return p.returncode, p.stdout, p.stderr

        rc, out, _ = cmd("gencheck", "ck")
        assert (
            rc == 0 and out.startswith("g1 ck: generating, asked ") and out.endswith("; `gencheck show g1`\n")
        )
        assert host.drain(30)
        rc, out, _ = cmd("gencheck", "show", "g1")
        assert rc == 0 and "math (4): cut 3 vs 1" in out and "SECRET" not in out
        rc, _, err = cmd("qda-rpc", "list")
        assert rc == 2 and "command's name" in err
        (bindir / "probe").symlink_to("qda-rpc")  # a command this search does not have
        rc, _, err = cmd("probe", "x")
        assert rc == 2 and "probe: refused: this search has no such command" in err
    finally:
        host.stop()


def test_the_judges_config_is_a_qda_kl_gencheck_config(tmp_path):
    from qda_kl.gencheck import GencheckConfig

    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    cfg = GencheckConfig.model_validate(r.config("g1"))  # extra keys are refused there
    assert (cfg.per_domain, cfg.max_tokens, cfg.prompts) == (4, 8192, "builtin")


def test_judge_jobs_have_their_own_temp_dir_in_the_judges_dir(tmp_path):
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.call("kl", "a")
    for job in r.host.sched.list(origin="judge"):
        name = job.env["TMPDIR"]  # absolute and short (sockets are bound in it), fixed at queueing
        tmp = Path(os.path.realpath(name))
        assert Path(name).is_absolute() and tmp.is_relative_to((r.host.jdir / "tmp").resolve())
        assert job.env["TMP"] == job.env["TEMP"] == name and tmp.is_dir()
        assert not tmp.stat().st_mode & 0o077


def test_checks_can_be_followed_over_time(tmp_path):
    """Several checks through a search: each line says when it was asked and, once ended, when it finished."""
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    ckpt(r.work / "b", weights=b"w2")
    r.call("gencheck", "a")
    r.settle()
    r.clock.advance(600)
    r.call("gencheck", "b")
    first, second = r.call("gencheck", "list").stdout.splitlines()
    assert first.startswith("g1 a: done (asked 13:46:40 UTC, finished 13:47:10 UTC): ")
    assert second == "g2 b: generating (asked 14:06:40 UTC)"


ROW = {"n": 4, "cut": 1, "loops": 0, "mean_tokens": 900.5}


def _writes(doc):
    """A fake gencheck job that writes `doc` as its `gencheck.json`."""

    def effect(argv):
        out = Path(json.loads(Path(argv[-1]).read_text())["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "gencheck.json").write_text(json.dumps(doc))

    return effect


def _both(**kw):
    """One `math` domain on both sides (`ROW` unless overridden on the checkpoint's side)."""
    return {"checkpoint": {"math": {**ROW, **kw}}, "original": {"math": ROW}}


@pytest.mark.parametrize(
    "doc, why",
    [
        ({"checkpoint": {"math": {"n": 4}}}, INCOMPLETE),  # checkpoint only, and a partial row
        ({"checkpoint": {"math": ROW}}, INCOMPLETE),  # no original
        ({"checkpoint": {"math": ROW}, "original": {"code": ROW}}, INCOMPLETE),  # mismatched domains
        ({"checkpoint": {"math": ROW, "code": ROW}, "original": {"math": ROW}}, INCOMPLETE),
        (_both(loops=None), INCOMPLETE),
        (
            {"checkpoint": {"math": ROW}, "original": {"math": {k: v for k, v in ROW.items() if k != "cut"}}},
            INCOMPLETE,
        ),
        (_both(mean_tokens=float("nan")), INCOMPLETE),
        (_both(cut=5), "math: cut 5 is not between 0 and n 4"),
        (_both(n=0, cut=0), "math: n 0 is not a positive count"),
        (_both(loops=-1), "math: loops -1 is not between 0 and n 4"),
        (_both(n=True), INCOMPLETE),
        ({"checkpoint": {}, "original": {}}, INCOMPLETE),
        ({"checkpoint": {"../x": ROW}, "original": {"../x": ROW}}, INCOMPLETE),  # nothing showable left
    ],
)
def test_an_incomplete_report_is_a_failed_check(tmp_path, doc, why):
    r = GenRig(tmp_path)
    r.be.default.effect = _writes(doc)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert r.call("gencheck", "show", "g1").stdout.endswith(
        f": failed (asked 13:46:40 UTC, finished 13:47:10 UTC): {why}\n"
    )


def test_unequal_and_fractional_counts_are_refused(tmp_path):
    """A paired report needs the same integer count in each domain: one answer per prompt the check asks for."""
    small = {"n": 1, "cut": 0, "loops": 0, "mean_tokens": 1}
    bad = {"checkpoint": {"math": small}, "original": {"math": {**small, "n": 32}}}
    fractional = {"checkpoint": {"math": {**small, "n": 1.5, "loops": 0.5}}, "original": {"math": small}}
    assert gen_report(bad) == (None, "math: n 1 vs 32 (checkpoint vs original)")
    assert gen_report(fractional) == (None, "math: n 1.5 is not an integer")
    assert gen_report({"checkpoint": {}, "original": {}})[0] is None
    assert gen_report({"checkpoint": {"math": ROW}, "original": {"math": ROW}}, ROW["n"])[0] is not None
    assert gen_report({"checkpoint": {"math": ROW}, "original": {"math": ROW}}, 8)[1] == (
        "math: n 4, the check asks 8 prompts"
    )
    r = GenRig(tmp_path)
    r.be.default.effect = _writes(bad)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert r.call("gencheck", "show", "g1").stdout.endswith(
        ": failed (asked 13:46:40 UTC, finished 13:47:10 UTC): math: n 1 vs 32 (checkpoint vs original)\n"
    )


def test_a_complete_report_is_shown(tmp_path):
    doc = {"max_tokens": 8192, "checkpoint": {"math": ROW}, "original": {"math": {**ROW, "cut": 0}}}
    r = GenRig(tmp_path)
    r.be.default.effect = _writes(doc)
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    r.settle()
    assert r.call("gencheck", "show", "g1").stdout.endswith(
        "done (asked 13:46:40 UTC, finished 13:47:10 UTC): checkpoint vs original, max 8192 tokens; "
        "math (4): cut 1 vs 0, loops 0 vs 0, tokens 900 vs 900\n"
    )


def test_the_final_record_carries_the_winners_own_gencheck(tmp_path):
    """A KL rule decides no generation suite; the pick's own valid `gencheck` still lands in FINAL.json."""
    r = GenRig(tmp_path, evaluator=str(evaluator(tmp_path)), gencheck_max_tokens=32768)
    r.host.install_evaluator()
    ckpt(r.work / "ckpt" / "f20")
    effect = r.be.default.effect

    def failed(argv):
        return 1 if argv[1:2] == [str(FAKE)] else effect(argv)

    assert r.call("gencheck", "ckpt/f20").exit == 0
    r.settle()
    assert r.call("handin", "ckpt/f20").exit == 0
    r.settle()
    final = r.host.final(generation_checks=False)
    assert final.rule == "handin" and final.choice is not None
    assert final.generation_check["id"] == "g1"
    # The two facts the end report and `evaluate` read, decided once from the record
    assert final.reading is False and final.gencheck is True

    r.be.default.effect = failed  # a failed check is no evidence of the winner
    assert r.call("gencheck", "ckpt/f20").exit == 0
    r.settle()
    assert r.host.final(generation_checks=False).generation_check["id"] == "g1"

    r.be.default.effect = effect  # a later valid check of the same checkpoint is the newest evidence
    assert r.call("gencheck", "ckpt/f20").exit == 0
    r.settle()
    evidence = r.host.final(generation_checks=False).generation_check
    assert evidence["id"] == "g3" and evidence["valid"] is True and evidence["max_tokens"] == 32768
    assert evidence["per_domain"] == 4  # the check ran the config's defaults
    assert evidence["checkpoint"]["math"] == {"n": 4, "cut": 3, "loops": 1, "mean_tokens": 5123.4}
    assert evidence["checkpoint"]["code"]["mean_tokens"] == 812.0
    assert evidence["original"]["math"]["cut"] == 1
    assert evidence["evaluator"] == json.loads((r.run / "judge" / "judge.json").read_text())["sha256"]
    r.host.write_final(r.run / "FINAL.json")
    got = json.loads((r.run / "FINAL.json").read_text())
    assert got["generation_check"] == evidence
    # The audit survives the run's weights: FINAL.json states both facts itself
    assert got["reading"] is False and got["gencheck"] is True


def test_the_handin_rule_records_the_chosen_id_and_its_own_gencheck(tmp_path):
    """FINAL.json names the rule and the hand-in id; the agent's own valid check is copied in by item 1."""
    r = GenRig(tmp_path, final_rules=["handin"])
    ckpt(r.work / "ckpt" / "a")
    assert r.call("gencheck", "ckpt/a").exit == 0
    r.settle()
    assert r.call("handin", "ckpt/a").exit == 0
    r.settle()
    final = r.host.final(generation_checks=False)
    assert final.rule == "handin" and final.choice is not None and final.choice.id == "h1"
    r.host.write_final(r.run / "FINAL.json")
    got = json.loads((r.run / "FINAL.json").read_text())
    assert got["rule"] == "handin" and got["choice"]["id"] == "h1"
    assert got["generation_check"]["id"] == "g1"


def test_a_handin_pick_without_a_check_says_none(tmp_path):
    r = GenRig(tmp_path, final_rules=["handin"])
    ckpt(r.work / "ckpt" / "a")
    assert r.call("handin", "ckpt/a").exit == 0
    r.settle()
    r.host.write_final(r.run / "FINAL.json")
    got = json.loads((r.run / "FINAL.json").read_text())
    assert got["choice"]["id"] == "h1"
    assert got["generation_check"]["outcome"].startswith("none: the agent ran no valid long-generation check")


def test_a_check_asks_for_its_own_length(tmp_path):
    """`--max-tokens` and `--per-domain` reach the job's config, the ledger and the queue's estimate."""
    r = GenRig(tmp_path, eval_max_tokens=32768)
    ckpt(r.work / "a")
    rep = r.call("gencheck", "a", "--max-tokens", "32768", "--per-domain", "2")
    assert rep.exit == 0 and "generating" in rep.stdout
    cfg = r.config("g1")
    assert (cfg["max_tokens"], cfg["per_domain"]) == (32768, 2)
    (job,) = r.host.sched.list(origin="judge")
    # `gencheck_expected_s` is for 8192 answers at 4 prompts per domain; 32768 at 2 is twice that product.
    assert job.expected_s == 1200
    (entry,) = r.host.entries
    assert (entry.max_tokens, entry.per_domain) == (32768, 2)


def test_max_tokens_over_the_rows_cap_is_refused_in_two_lines(tmp_path):
    r = GenRig(tmp_path, eval_max_tokens=16384)
    ckpt(r.work / "a")
    rep = r.call("gencheck", "a", "--max-tokens", "32768")
    assert rep.exit == 2
    assert rep.stderr.splitlines() == [
        "gencheck: refused: --max-tokens 32768 is over this row's cap of 16384 tokens",
        "use --max-tokens N with N <= 16384",
    ]
    frozen = r.run / "judge" / "frozen"
    assert not frozen.exists() or not any(frozen.iterdir())  # nothing was taken or queued


def test_max_tokens_is_capped_at_32768_when_the_row_allows_more(tmp_path):
    r = GenRig(tmp_path, eval_max_tokens=65536)
    ckpt(r.work / "a")
    rep = r.call("gencheck", "a", "--max-tokens", "32769")
    assert rep.exit == 2 and "over this row's cap of 32768 tokens" in rep.stderr


def test_per_domain_over_the_prompt_set_is_refused(tmp_path):
    r = GenRig(tmp_path)
    ckpt(r.work / "a")
    rep = r.call("gencheck", "a", "--per-domain", "25")
    assert rep.exit == 2
    assert rep.stderr.splitlines() == [
        "gencheck: refused: --per-domain 25 is outside the prompt set: it holds 24 prompts in every domain",
        "use --per-domain K with 1 <= K <= 24",
    ]
