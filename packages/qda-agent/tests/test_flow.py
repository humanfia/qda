import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import testkit
import tests_support  # noqa: F401  (registers the tests' contamination checks)
import yaml
from hmz.flows import HarnessRefused, HarnessThrottled, HarnessUnrecoverable
from hmz.sdk import fakes
from qda_agent import isolation
from qda_agent.agents import endpoint_vars
from qda_agent.flow import ContaminationFound, qda_search, qda_search_plain

pytestmark = pytest.mark.asyncio


def write_config(tmp: Path, *, seconds: float = 0.5, **extra) -> Path:
    (tmp / "base" / "gptq").mkdir(parents=True, exist_ok=True)  # the baseline the config names
    cfg = {
        "model": "tiny/model",
        "format": "w4a16",
        "budget_gpu_hours": seconds / 3600,
        "contamination": "none",
        "cards": [0],
        "run_dir": "run",
        "sandbox": "none",
        "host": "testkit:fake_host",
        "baselines": ["base/gptq"],
        "drain_timeout_s": 5,
        "commands": None,
        "agent": {"resend_wait_s": 0.02},
        "judge": {"eval_max_tokens": 32768},
        **extra,
    }
    if isinstance(cfg.get("judge"), dict):
        cfg["judge"].setdefault("eval_max_tokens", 32768)
    path = tmp / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def host_of(run: Path) -> testkit.FakeHost:
    return testkit.HOSTS[run]


def guard_of(tmp: Path, letter: str = "w", n: int = 25) -> tuple[Path, str]:
    """A guard file holding the n-gram windows of a fresh n-word item, and that item."""
    from qda_agent.judge.contamination import windows

    item = " ".join(f"{letter}{i}" for i in range(n))
    path = tmp / "guard.txt"
    path.write_text("\n".join(sorted(windows(item))) + "\n")
    return path, item


def fake_venv(at: Path) -> tuple[Path, Path]:
    """A real virtualenv whose `python` answers sysconfig, and its site-packages as that interpreter names it."""
    (at / "bin").mkdir(parents=True)
    (at / "bin" / "python").symlink_to(sys.executable)
    (at / "pyvenv.cfg").write_text(f"home = {sys.base_prefix}\n")
    site = subprocess.run(
        [str(at / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return at, Path(site)


def site_dirs(venv: Path) -> list[str]:
    """`venv`'s purelib and platlib as real paths: one directory on Debian, `lib` and `lib64` on RHEL."""
    out = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import sysconfig; p = sysconfig.get_paths(); print(p['purelib']); print(p['platlib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted({str(Path(d).resolve()) for d in out})


async def test_goal_resent_on_idle_until_deadline(tmp_path):
    config = write_config(tmp_path, seconds=0.4)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    run = tmp_path / "run"
    assert all(p.startswith("/goal Read TASK.md in full and follow it") for p in worker.prompts)
    assert len(worker.prompts) >= 3  # idle -> sent again, until the deadline
    assert len(worker.sessions) == 1  # one conversation throughout
    assert "Candidate 0's KL is" not in (run / "work" / "TASK.md").read_text()

    assert out["ended"] == "deadline" and out["rule"] == "handin"
    final = json.loads((run / "FINAL.json").read_text())
    assert final["rule"] == "handin" and final["baseline"].endswith("base/gptq")
    assert final["chosen_by"] == "fallback_candidate0"  # nothing was handed in or scored
    assert (run / "work" / "TASK.md").read_text().startswith("# Goal: the best w4a16 tier")
    assert "Submission" in (run / "END-REPORT.md").read_text()


@pytest.mark.parametrize(("level", "jobs"), [("none", "none"), ("sandbox", "bwrap")])
async def test_end_report_states_the_isolation_its_jobs_ran_at(tmp_path, monkeypatch, level, jobs):
    """END-REPORT names `isolation.level` and the job sandbox it resolves to, not the legacy `sandbox.name`."""
    # a sandboxed level refuses a run dir under /tmp: this pins the report, not that check
    monkeypatch.setattr(isolation, "check", lambda cfg: None)
    config = write_config(tmp_path, seconds=0.3, isolation={"level": level})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    report = (tmp_path / "run" / "END-REPORT.md").read_text()
    assert f"isolation `{level}`, jobs `{jobs}`" in report and "sandbox `none`" not in report


@pytest.mark.parametrize(
    "fmt,bpw,menu",
    [
        ("w4a16", 4.650715900527506, ["bf16", "w8a16", "w4a16"]),
        ("w3a16", 3.4, ["bf16", "w8a16", "w4a16", "w3a16"]),
    ],
)
async def test_prepare_puts_the_configured_judge_contract_in_task(tmp_path, fmt, bpw, menu):
    from qda_agent.config import RunConfig
    from qda_agent.flow import prepare

    (tmp_path / "original").mkdir()
    config = write_config(
        tmp_path,
        format=fmt,
        sandbox="bwrap",
        judge={
            "anchor": "original",
            "budget_bpw": bpw,
            "format_menu": menu,
            "require_build_receipt": True,
            "max_kl": 0.75,
        },
    )
    cfg = RunConfig.load(config)
    prepare(cfg, datetime.now(UTC), datetime.now(UTC), installed=True)
    text = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert f"Budget: at most {bpw} bits per quantized weight" in text
    assert "Format menu: " + ", ".join(f"`{value}`" for value in cfg.judge.format_menu) + "." in text
    assert "keeps every build you handed in whole" in text and "KL > 0.75" in text


async def test_prepare_uses_the_gguf_engine_contract_from_the_run_config(tmp_path, monkeypatch):
    from qda_agent.config import RunConfig
    from qda_agent.flow import prepare
    from test_judge import llama_paths

    (tmp_path / "llama-server").write_text("#!/bin/sh\n")
    config = write_config(
        tmp_path,
        sandbox="bwrap",
        judge={
            "engine": "llama.cpp",
            "budget_bytes": 17_548_181_504,
            "format_menu": ["Q4_K", "Q6_K"],
            "engine_floor": 0.000273,
            "llamacpp": llama_paths(tmp_path, monkeypatch),
            "require_build_receipt": True,
            "max_run_bytes": 100_000_000_000,
        },
    )
    prepare(RunConfig.load(config), datetime.now(UTC), datetime.now(UTC), installed=True)
    text = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert "Budget: at most 17548181504 bytes for the whole .gguf file." in text
    assert "Format menu: `Q4_K`, `Q6_K`." in text
    assert "KL 0.000273 against the sealed BF16 reference" in text
    assert "short inference in stock llama.cpp" in text
    assert "Disk budget: the run may hold 100.0 GB in all" in text
    assert "keeps every build you handed in whole" in text
    assert "bpw" not in text and "bits-per-weight" not in text


@pytest.mark.parametrize("engine", ["vllm", "llama.cpp"])
async def test_prepare_advertises_the_bpw_tool_only_on_a_vllm_run(tmp_path, monkeypatch, engine):
    """TASK.md offers the judge's bits-per-weight tool on vLLM and withholds it on llama.cpp."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import prepare
    from test_judge import llama_paths

    (tmp_path / "original").mkdir()
    (tmp_path / "llama-server").write_text("#!/bin/sh\n")
    judge = {"anchor": "original", "budget_bpw": 4.5}
    if engine == "llama.cpp":
        judge = {
            "engine": "llama.cpp",
            "budget_bytes": 17_548_181_504,
            "format_menu": ["Q4_K", "Q6_K"],
            "engine_floor": 0.000273,
            "llamacpp": llama_paths(tmp_path, monkeypatch),
        }
    config = write_config(tmp_path, sandbox="bwrap", judge=judge)
    prepare(RunConfig.load(config), datetime.now(UTC), datetime.now(UTC), installed=True)
    text = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert ("`bpw DIR`" in text) is (engine == "vllm")
    assert ("bits per quantized weight" in text) is (engine == "vllm")
    assert ("Six commands in `bin/`" in text) is (engine == "vllm")
    assert ("Five commands in `bin/`" in text) is (engine == "llama.cpp")


def test_task_md_names_each_pre_built_start_with_its_dir_and_kl(tmp_path):
    """A row with `judge.starts` lists each start: label, the dir the agent sees, its measured KL, no host path."""
    from qda_agent import flow
    from qda_agent.config import RunConfig
    from qda_agent.host import HandIn as Reading

    config = write_config(
        tmp_path,
        seconds=0.2,
        baselines=[str(tmp_path / "base" / "gptq")],
        judge={
            "eval_max_tokens": 32768,
            "starts": [{"name": "gptq", "path": str(tmp_path / "base" / "gptq"), "origin": "own"}],
        },
    )
    t = datetime(2026, 1, 1, tzinfo=UTC)
    cfg = RunConfig.load(config)
    read = Reading(
        dir=tmp_path / "base" / "gptq",
        submitted_at=t,
        kind="scored",
        valid=True,
        eligible=False,
        metrics={"kl": 0.03},
        higher_is_better={"kl": False},
        name="gptq",
        origin="own",
        bar=True,
    )
    flow.prepare(cfg, t, t, installed=True, baseline_readings=[read])
    raw = (cfg.workdir / "TASK.md").read_text()
    text = " ".join(raw.split())
    assert "Your pre-built starts, best KL first" in text
    assert "- gptq: `baselines/0`, KL 0.03" in text
    assert (
        str(tmp_path / "base" / "gptq") not in raw
    )  # the agent sees the run-relative dir, never a host path


async def test_the_launch_rerenders_task_md_with_each_starts_kl(tmp_path, monkeypatch):
    """After the host reads the starts, the flow rewrites TASK.md with each measured KL."""
    import testkit
    from qda_agent.host import HandIn as Reading

    config = write_config(
        tmp_path,
        seconds=0.2,
        baselines=[str(tmp_path / "base" / "gptq")],
        judge={
            "eval_max_tokens": 32768,
            "starts": [{"name": "gptq", "path": str(tmp_path / "base" / "gptq"), "origin": "own"}],
        },
    )
    original = testkit.FakeHost.final

    def final(self):  # the in-memory host reports what the judge read at start
        out = original(self)
        out.baseline_readings = [
            Reading(
                dir=tmp_path / "base" / "gptq",
                submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
                kind="scored",
                valid=True,
                eligible=False,
                metrics={"kl": 0.03},
                higher_is_better={"kl": False},
                name="gptq",
                origin="own",
                bar=True,
            )
        ]
        return out

    monkeypatch.setattr(testkit.FakeHost, "final", final)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    task = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert "- gptq: `baselines/0`, KL 0.03" in task


async def test_task_states_the_runs_generation_cap(tmp_path):
    """TASK.md gives the row's final-evaluation cap (`judge.eval_max_tokens`), not gencheck's length."""
    config = write_config(tmp_path, seconds=0.2, judge={"eval_max_tokens": 81920})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    task = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert "The evaluation generates up to 81920 tokens" in task


async def test_a_run_that_states_the_eval_cap_is_refused_without_it(tmp_path):
    """TASK.md states the row's final-evaluation cap: without it the run is refused before the clock."""
    config = write_config(tmp_path, seconds=0.2)
    shipped = yaml.safe_load(config.read_text())
    del shipped["judge"]["eval_max_tokens"]
    config.write_text(yaml.safe_dump(shipped))
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    with pytest.raises(ValueError, match="judge.eval_max_tokens"):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []  # refused before the agent's first turn
    config = write_config(tmp_path, seconds=0.2, judge={"eval_max_tokens": 81920})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts  # with the cap set, the same run starts


async def test_deadline_cuts_a_busy_turn(tmp_path):
    config = write_config(tmp_path, seconds=0.3)
    run = tmp_path / "run"

    async def stuck(prompt, **_):
        await asyncio.sleep(30)  # a turn that would run far past the deadline

    import time

    t = time.monotonic()
    worker = fakes.FakeAgentDriver(reply=stuck)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert time.monotonic() - t < 5
    assert out["ended"] == "deadline" and len(worker.prompts) == 1
    events = [json.loads(line)["ev"] for line in (run / "flow.jsonl").read_text().splitlines()]
    assert events[-3:] == ["cutoff", "drained", "final"]


async def test_last_valid_handin_wins(tmp_path):
    config = write_config(tmp_path, seconds=0.5)
    run = tmp_path / "run"
    work = run / "work"

    def handin(name, **kw):
        return lambda: host_of(run).submit(testkit.build_ckpt(work, name), **kw)

    def score(name, kl):
        return lambda: host_of(run).score(testkit.build_ckpt(work, name), kl)

    script = testkit.ScriptedWorker(
        [score("a", 0.09), handin("b", kl=0.08), score("c", 0.05), handin("d", valid=False)]
    )
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["rule"] == "handin" and out["choice"].endswith("out/b")
    final = json.loads((run / "FINAL.json").read_text())
    assert final["chosen_by"] == "handin"  # the agent's own hand-in decides, not the lower KL of out/c
    assert [h["kind"] for h in final["ledger"]] == ["scored", "handin", "scored", "handin"]


async def test_the_fallback_is_candidate_0_without_a_valid_handin(tmp_path):
    """No valid hand-in, default fallback: candidate 0 as-is, and FINAL.json says the fallback chose it."""
    config = write_config(tmp_path, seconds=0.4)
    run = tmp_path / "run"
    work = run / "work"
    steps = [
        lambda: host_of(run).score(testkit.build_ckpt(work, "a"), 0.09),
        lambda: host_of(run).score(testkit.build_ckpt(work, "c"), 0.05),
        lambda: host_of(run).submit(testkit.build_ckpt(work, "d"), valid=False, reason="does not load"),
    ]
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(steps))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["rule"] == "handin" and out["choice"] is None
    final = json.loads((run / "FINAL.json").read_text())
    assert final["chosen_by"] == "fallback_candidate0" and final["baseline"].endswith("base/gptq")


async def test_a_handin_list_counts_its_first_entry_that_passed(tmp_path):
    """One `handin` call is one best-first list: its first valid entry counts, an earlier list if none do."""
    config = write_config(tmp_path, seconds=0.6)
    run = tmp_path / "run"
    work = run / "work"

    def first_list() -> str:
        a, b = testkit.build_ckpt(work, "a"), testkit.build_ckpt(work, "b")
        host_of(run).submit_list([(a, False, 0.02, "does not load"), (b, True, 0.09, "")])
        return "handed in a, b"

    def second_list() -> str:
        c, d = testkit.build_ckpt(work, "c"), testkit.build_ckpt(work, "d")
        host_of(run).submit_list([(c, False, 0.01, "does not load"), (d, True, 0.10, "")])
        return "handed in c, d"

    def latest_list_all_refused() -> str:
        e = testkit.build_ckpt(work, "e")
        host_of(run).submit_list([(e, False, 0.01, "does not load")])
        return "handed in e"

    worker = fakes.FakeAgentDriver(
        reply=testkit.ScriptedWorker([first_list, second_list, latest_list_all_refused])
    )
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["rule"] == "handin" and out["choice"].endswith("out/d")
    final = json.loads((run / "FINAL.json").read_text())
    assert final["chosen_by"] == "handin"
    assert [h.get("handin_list") for h in final["ledger"] if h["kind"] == "handin"] == [1, 1, 2, 2, 3]
    assert final["choice"]["dir"].endswith("out/d")  # list 2's first valid entry, not list 3's refused one


async def test_the_agents_last_handin_wins_over_an_unhanded_lower_kl_build(tmp_path):
    """The agent's own last valid hand-in is the result: an unhanded build with a lower KL does not win."""
    config = write_config(tmp_path, seconds=0.5)
    run = tmp_path / "run"
    work = run / "work"

    def handin(name, kl):
        def go():
            d = testkit.build_ckpt(work, name)
            host_of(run).submit(d, kl=kl)
            return f"handed in {name}"

        return go

    def score(name, kl):
        def go():
            host_of(run).score(testkit.build_ckpt(work, name), kl)
            return f"scored {name}"

        return go

    script = testkit.ScriptedWorker([handin("h1", 0.08), score("u", 0.01), handin("h2", 0.09)])
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["rule"] == "handin" and out["choice"].endswith("out/h2")
    final = json.loads((run / "FINAL.json").read_text())
    assert final["chosen_by"] == "handin"
    assert final["choice"]["dir"].endswith("out/h2")  # not out/u, the unhanded lowest KL


async def test_a_check_queued_before_the_cutoff_finishes_in_the_drain(tmp_path):
    """TASK.md promises that checks already queued still finish: the drain runs before the pick."""
    config = write_config(tmp_path, seconds=2.0, agent={"resend_wait_s": 0.05})
    run = tmp_path / "run"
    work = run / "work"
    seen: list[testkit.FakeHost] = []

    async def late_handin() -> str:
        await asyncio.sleep(1.0)  # the check is queued about 1 s before the 2 s deadline
        d = testkit.build_ckpt(work, "late")
        host = host_of(run)
        seen.append(host)
        return host.submit(d, valid=None, kl=0.02)

    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([late_handin]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    (host,) = seen
    calls = host.calls
    assert calls.index("cutoff") < next(i for i, c in enumerate(calls) if c.startswith("drain("))
    final = json.loads((run / "FINAL.json").read_text())
    assert out["rule"] == "handin" and final["choice"]["dir"].endswith("out/late")
    assert final["choice"]["valid"] is True  # the check finished during the drain, before the pick
    assert final["chosen_by"] == "handin"


async def test_the_task_says_what_the_default_calibration_lacks(tmp_path):
    """A thinking model's default calib says it holds no thinking text and names `qda traces`."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import prepare

    calib = tmp_path / "calib.jsonl"
    calib.write_text("")
    thinking = {"policy": "keep", "template": {"think_markers": True, "reasoning": True}, "chats": 3}
    (tmp_path / "calib.jsonl.meta.json").write_text(json.dumps({"thinking": thinking}))
    config = write_config(tmp_path, calib=str(calib))
    prepare(RunConfig.load(config), datetime.now(UTC), datetime.now(UTC), installed=True)
    text = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert f"the default calibration file is `{calib}`." in text
    assert "The default calibration holds no thinking text" in text
    assert "qda traces <model> <prompts.jsonl> <prefix> --max-tokens <tokens> --seed <seed>" in text
    assert "scale_kd" in text


async def test_the_task_uses_the_paper_objective_and_states_the_rule_once(tmp_path):
    """TASK.md's objective text: the paper's wording, `kl` as one tool, the hand-in rule once."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import prepare

    config = write_config(tmp_path)  # no judge.final_rules: the documented default
    prepare(RunConfig.load(config), datetime.now(UTC), datetime.now(UTC), installed=True)
    text = " ".join((tmp_path / "run" / "work" / "TASK.md").read_text().split())
    assert "aim for the lowest KL" not in text.lower()
    assert "the valid checkpoint with the lowest KL" not in text
    assert "The four count equally" in text  # the paper's objective: the main-table evaluation
    assert "The judge's `kl` is one tool you can use; your measurements are your responsibility." in text
    assert "It is the main measurement you have here" in text
    assert "are a tie" not in text  # judge.kl_noise unset: no tie sentence
    assert "The evaluation generates up to 32768 tokens" in text
    assert "a check with a smaller N is faster and misses loops that appear only in long answers" in text
    assert "Every calibration and training row must be in the model's own chat template" in text
    assert "A start handed in as-is is not your result." in text
    assert (
        "`handin DIR [DIR...]` hands in checkpoint directories you made in this search, best first; a later "
        "call replaces the list." in text
    )
    # The judge neither scores nor genchecks a new hand-in; measuring before handing in is the agent's
    assert (
        "Handing in does not measure: the judge takes no KL reading of a newly handed-in checkpoint and runs "
        "no `gencheck` of it. Whether you score a candidate and whether you run a long-generation check of it "
        "before handing it in are your calls and your risk" in text
    )
    assert text.count("the first directory of your latest hand-in list that has passed counts") == 1


async def test_api_errors_resend_and_fatal_stops(tmp_path):
    config = write_config(tmp_path, seconds=2.0, agent={"resend_wait_s": 0.1})
    script = testkit.ScriptedWorker(
        [HarnessThrottled("429"), HarnessThrottled("429"), lambda: "ok", HarnessRefused("bad key")]
    )
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["failed"] == 2 and out["ended"] == "fatal: HarnessRefused"
    assert out["fatal"] == "HarnessRefused"  # the class only: the CLI's words stay in flow.jsonl
    assert (
        len(worker.prompts) == 4
    )  # no wait of the flow's own between them: Humanize owns the retry schedule


async def test_api_failures_for_the_whole_window_finalize_the_run(tmp_path, monkeypatch):
    """A failed turn is a failed round; ten minutes of them with none returning end the run with its pick."""
    from qda_agent import flow

    monkeypatch.setattr(flow, "FINALIZE_AFTER_S", 0.2)
    monkeypatch.setattr(flow, "FLOOR_POLL_S", 0.02)
    config = write_config(tmp_path, seconds=5.0, agent={"resend_wait_s": 0.05})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([HarnessThrottled("429")] * 20))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    run = tmp_path / "run"
    assert out["end_kind"] == "finalize" and out["ended"].startswith("finalize: the agent API failed")
    final = json.loads((run / "FINAL.json").read_text())
    assert final["baseline"].endswith("base/gptq") and final["chosen_by"] == "fallback_candidate0"
    assert final["warnings"] == [f"the search was finalized early: {out['finalize_reason']}"]
    assert json.loads((run / "result.json").read_text())["finalize_reason"] == out["finalize_reason"]


async def test_a_turn_that_returns_resets_the_api_failure_window(tmp_path, monkeypatch):
    """Only consecutive failures count: a turn that returns starts the ten minutes again (fail, ok, fail…)."""
    from qda_agent import flow

    monkeypatch.setattr(flow, "FINALIZE_AFTER_S", 0.2)
    monkeypatch.setattr(flow, "FLOOR_POLL_S", 0.02)
    config = write_config(tmp_path, seconds=1.0, agent={"resend_wait_s": 0.05})
    script = testkit.ScriptedWorker([HarnessThrottled("429"), lambda: "ok"] * 40, idle_s=0.02)
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and out["failed"] >= 2
    events = [json.loads(line)["ev"] for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert "finalize_trigger" not in events  # no ten-minute stretch of failures, so no early close


async def test_a_disk_below_its_floor_for_the_whole_window_finalizes_the_run(tmp_path, monkeypatch):
    """The disk has stayed below the run's floor with nothing to free: the turn is cut and the run closes."""
    from qda_agent import flow

    monkeypatch.setattr(flow, "FINALIZE_AFTER_S", 0.2)
    monkeypatch.setattr(flow, "FLOOR_POLL_S", 0.02)

    async def long_turn() -> str:
        testkit.HOSTS[tmp_path / "run"].below_floor_since = time.time()
        await asyncio.sleep(5.0)  # a turn only the finalize watch can end
        return "done"

    config = write_config(tmp_path, seconds=5.0)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([long_turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    run = tmp_path / "run"
    assert out["end_kind"] == "finalize"
    assert out["ended"].startswith("finalize: the disk stayed below the run's free-space floor")
    assert out["choice"] is None and out["baseline"].endswith("base/gptq")
    assert len(worker.prompts) == 1  # the turn in flight was cut; no new one was sent
    final = json.loads((run / "FINAL.json").read_text())
    assert final["warnings"] == [f"the search was finalized early: {out['finalize_reason']}"]


@pytest.mark.parametrize("reachable", [False, True])
async def test_an_api_out_of_reach_mid_turn_finalizes_the_run(tmp_path, monkeypatch, reachable):
    """A turn that neither fails nor returns while the API's route fails every probe (Codex reconnecting for
    ever) is cut once the route has failed for the whole window; a route that answers leaves the turn alone."""
    from qda_agent import api_route, flow

    monkeypatch.setattr(flow, "FINALIZE_AFTER_S", 0.2)
    monkeypatch.setattr(flow, "API_POLL_S", 0.02)
    monkeypatch.setattr(api_route, "api_endpoint", lambda cfg: ("api.example", 443))
    probes = []

    def probe(host, port, *, direct):
        probes.append((host, port, direct))
        return None if reachable else "the HTTPS_PROXY at proxy:3128 cannot be reached: Connection refused"

    monkeypatch.setattr(api_route, "api_route_problem", probe)

    async def hung_turn() -> str:
        await asyncio.sleep(1.5)  # a turn only the finalize watch can end
        return "done"

    config = write_config(tmp_path, seconds=1.0)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([hung_turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert probes and probes[0] == ("api.example", 443, False)
    if reachable:
        assert out["ended"] == "deadline"
        return
    assert out["end_kind"] == "finalize"
    assert out["ended"].startswith(
        "finalize: the agent API could not be reached for 0 minutes (the HTTPS_PROXY"
    )
    assert out["choice"] is None and out["baseline"].endswith("base/gptq")
    assert len(worker.prompts) == 1  # the turn in flight was cut; no new one was sent


def watchdog_stop(silent_s: int = 360) -> BaseException:
    """The error a turn Humanize's watchdog stopped raises, made the way Humanize makes it."""
    from hmz.coganchor.agents import Failed
    from hmz.runtime.flowing.harnessing import harness_error

    verdict = Failed(1, ["dsh"], "", f"the watchdog stopped this turn: dsh has said nothing for {silent_s}s")
    leaf = harness_error(verdict, "dsh")
    assert isinstance(leaf, HarnessUnrecoverable)  # what made the flow end the whole search
    leaf.__cause__ = verdict
    return leaf


async def test_a_turn_the_watchdog_stopped_is_resent_and_the_search_goes_on(tmp_path):
    """One tool call silent past the watchdog's window fails that turn only: the goal is sent again and the
    search runs to its deadline."""
    config = write_config(tmp_path, seconds=1.0, agent={"resend_wait_s": 0.05})
    script = testkit.ScriptedWorker([watchdog_stop(), lambda: "ok"], idle_s=0.05)
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and out["failed"] == 1 and len(worker.prompts) >= 3
    failed = [
        json.loads(line)
        for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()
        if json.loads(line)["ev"] == "failed"
    ]
    assert "the watchdog stopped the turn (1 of 3 in a row)" in failed[0]["why"]


async def test_the_watchdog_stopping_every_turn_ends_the_search(tmp_path, monkeypatch):
    from hmz.coganchor.agents.watchdog import WATCHDOG

    monkeypatch.delenv(WATCHDOG, raising=False)
    config = write_config(tmp_path, seconds=2.0, agent={"resend_wait_s": 0.05})
    script = testkit.ScriptedWorker([watchdog_stop(), watchdog_stop(), watchdog_stop(), lambda: "ok"])
    worker = fakes.FakeAgentDriver(reply=script)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "fatal: HarnessUnrecoverable" and out["failed"] == 2
    assert len(worker.prompts) == 3  # the third stop in a row is the last
    assert WATCHDOG not in os.environ  # restored on the fatal path too


async def test_a_turn_that_returns_resets_the_watchdog_count(tmp_path):
    config = write_config(tmp_path, seconds=1.5, agent={"resend_wait_s": 0.05})
    stops = [watchdog_stop(), watchdog_stop(), lambda: "ok"] * 3
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(stops, idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and out["failed"] == 6


async def test_an_unrecoverable_failure_not_the_watchdogs_still_ends_the_search(tmp_path):
    from hmz.coganchor.agents import Failed

    config = write_config(tmp_path, seconds=2.0, agent={"resend_wait_s": 0.05})
    own = HarnessUnrecoverable("dsh: cannot be configured at that model")
    own.__cause__ = Failed(1, ["dsh"], "", "the model is not supported")  # a CLI's own words
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([own, lambda: "ok"]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "fatal: HarnessUnrecoverable" and out["failed"] == 0 and len(worker.prompts) == 1


async def test_a_tool_call_longer_than_dshs_own_window_is_not_silence(tmp_path, monkeypatch):
    """The case that stopped a search: one dsh bash call `sleep 420; kl list` declared 480 s, past dsh's own
    360 s window. While a search runs every CLI's window is at least `WATCHDOG_FLOOR_S`, so that call is
    not silence; the launcher's environment is as it was once the search ends."""
    from hmz.coganchor.agents.watchdog import WATCHDOG, silence
    from qda_agent.flow import WATCHDOG_FLOOR_S

    monkeypatch.delenv(WATCHDOG, raising=False)
    assert silence("dsh") < 480 < WATCHDOG_FLOOR_S  # what the watchdog gave the call before
    windows: dict[str, float] = {}

    def turn():
        windows.update({cli: silence(cli) for cli in ("dsh", "claude", "codex")})
        return "ok"

    config = write_config(tmp_path, seconds=0.5)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline"
    assert windows["dsh"] > 480_000 / 1000  # the call's timeoutMs: not silence
    assert windows["claude"] >= 900 and windows["codex"] >= 900  # no CLI's window got shorter
    assert WATCHDOG not in os.environ and silence("dsh") == 360


@pytest.mark.parametrize("given", ["1800", "0"])
async def test_a_launchers_watchdog_window_stands(tmp_path, monkeypatch, given):
    from hmz.coganchor.agents.watchdog import WATCHDOG

    monkeypatch.setenv(WATCHDOG, given)  # a row's own window, or the watchdog turned off
    seen: list[str | None] = []
    config = write_config(tmp_path, seconds=0.5)
    script = testkit.ScriptedWorker([lambda: seen.append(os.environ.get(WATCHDOG)) or "ok"])
    worker = fakes.FakeAgentDriver(reply=script)
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert seen == [given] and os.environ[WATCHDOG] == given


RELAY = "https://relay.example/v1"  # the launching shell's endpoint: the run configures none


async def _endpoint_seen(
    monkeypatch, tmp_path, agent: dict
) -> tuple[dict, dict[str, str | None], list[dict]]:
    seen: dict[str, str | None] = {}

    def turn():
        seen.update({v: os.environ.get(v) for v in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")})
        return "ok"

    config = write_config(tmp_path, seconds=0.5, agent={"resend_wait_s": 0.02, **agent})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    return out, seen, events


@pytest.mark.parametrize("account", [None, "work"])
async def test_a_cli_signed_in_by_file_or_account_never_follows_an_inherited_endpoint(
    tmp_path, monkeypatch, account
):
    """The 401 that ended a search at its first turn: the CLI signed in by its file (or an account), and the
    launching shell's `ANTHROPIC_BASE_URL` sent that sign-in to a relay. The variable is gone while the search
    runs, its name (never its value) is in flow.jsonl, and the shell's environment is as it was afterwards."""
    for t in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(t, raising=False)
    if account:
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "relay-token")  # an account signs in on its own
    monkeypatch.setenv("ANTHROPIC_BASE_URL", RELAY)
    monkeypatch.setenv("OPENAI_BASE_URL", RELAY)  # not a CLI of this run: left alone
    backend = {"backend": {"name": "claude", "account": account}} if account else {}
    out, seen, events = await _endpoint_seen(monkeypatch, tmp_path, backend)
    assert out["ended"] == "deadline"
    assert seen == {"ANTHROPIC_BASE_URL": None, "OPENAI_BASE_URL": RELAY}
    (dropped,) = [e for e in events if e["ev"] == "env_dropped"]
    assert dropped["names"] == ["ANTHROPIC_BASE_URL"] and RELAY not in json.dumps(events)
    assert os.environ["ANTHROPIC_BASE_URL"] == RELAY


async def test_a_cli_signed_in_by_a_token_variable_keeps_its_endpoint(tmp_path, monkeypatch):
    """A token and the endpoint it was issued for are one sign-in: neither is dropped."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "relay-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", RELAY)
    out, seen, events = await _endpoint_seen(monkeypatch, tmp_path, {})
    assert out["ended"] == "deadline" and seen["ANTHROPIC_BASE_URL"] == RELAY
    assert not [e for e in events if e["ev"] == "env_dropped"]


MIXED = {"backend": {"name": "claude", "account": "work"}, "fallbacks": [{"name": "claude"}]}


async def test_one_cli_signed_in_by_an_account_and_by_a_token_is_refused_at_the_start(tmp_path, monkeypatch):
    """An account-signed claude and a token-signed claude fallback share one environment: with an endpoint
    variable set, no choice is right for both, so the run is refused before a card is used."""
    from qda_agent.isolation import SignInRefused

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "relay-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", RELAY)
    monkeypatch.setenv("OPENAI_BASE_URL", RELAY)  # an earlier CLI's endpoint: still there after the refusal
    codex = {"name": "codex", "model": "gpt-5"}
    agent = {**MIXED, "backend": codex, "fallbacks": [MIXED["backend"], *MIXED["fallbacks"]]}
    config = write_config(tmp_path, seconds=0.5, agent={"resend_wait_s": 0.02, **agent})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([lambda: "ok"]))
    with pytest.raises(SignInRefused) as refused:
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    why = str(refused.value)
    assert (
        "claude@work/claude-opus-5-5:high signs in by an account and claude/claude-opus-5-5:high by a" in why
    )
    assert "ANTHROPIC_BASE_URL is set" in why and "Sign both in the same way" in why
    assert "unset ANTHROPIC_BASE_URL where the run is launched" in why
    assert worker.prompts == []
    assert os.environ["ANTHROPIC_BASE_URL"] == RELAY and os.environ["OPENAI_BASE_URL"] == RELAY


async def test_the_same_mix_with_no_endpoint_variable_starts(tmp_path, monkeypatch):
    """The account and a token for the provider's own endpoint (no base URL anywhere): nothing to drop or keep,
    so the run starts as it did before."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    for v in endpoint_vars("claude"):
        monkeypatch.delenv(v, raising=False)
    out, _, events = await _endpoint_seen(monkeypatch, tmp_path, MIXED)
    assert out["ended"] == "deadline" and not [e for e in events if e["ev"] == "env_dropped"]


async def test_the_endpoint_variables_come_from_the_clis_humanize_profile():
    """Humanize's profile names where a turn goes (`endpoint`) and the vendor's other base URLs (`ambient`):
    codex's own `CODEX_PROVIDER_URL` and claude's cloud base URLs are covered, not only the two named by hand."""
    from hmz.coganchor.backends import named

    for cli in ("claude", "codex"):
        profile = named(cli)
        assert profile is not None and profile.endpoint in endpoint_vars(cli)
        assert {v for v in profile.ambient if v.endswith("_URL")} <= set(endpoint_vars(cli))
    assert "CODEX_PROVIDER_URL" in endpoint_vars("codex") and "OPENAI_BASE_URL" in endpoint_vars("codex")
    assert "ANTHROPIC_BEDROCK_BASE_URL" in endpoint_vars("claude")  # named by hand: not in the profile
    assert endpoint_vars("no-such-cli") == ()


async def test_an_alias_spelling_of_a_cli_is_matched_by_its_canonical_name():
    """A spec spelled with a CLI alias (`claude-code`) is the CLI `clis()` names (`claude`): the refusal message
    finds it instead of failing on an empty match."""
    from qda_agent.agents import canonical_cli as _canonical

    assert _canonical("claude-code@work/claude-opus-5-5:high") == "claude"
    assert _canonical("claude/claude-opus-5-5") == "claude"
    assert _canonical("codex@acct/gpt-5") == "codex"
    assert _canonical("no-such-cli/m") == "no-such-cli"


async def test_a_file_signed_codex_drops_its_provider_url(tmp_path, monkeypatch):
    for t in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        monkeypatch.delenv(t, raising=False)
    monkeypatch.setenv("CODEX_PROVIDER_URL", RELAY)
    seen: dict[str, str | None] = {}

    def turn():
        seen["CODEX_PROVIDER_URL"] = os.environ.get("CODEX_PROVIDER_URL")
        return "ok"

    agent = {"resend_wait_s": 0.02, "backend": {"name": "codex", "model": "gpt-5"}}
    config = write_config(tmp_path, seconds=0.5, agent=agent)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and seen == {"CODEX_PROVIDER_URL": None}
    assert os.environ["CODEX_PROVIDER_URL"] == RELAY


async def test_a_cli_whose_sign_in_qda_does_not_know_keeps_its_endpoint(tmp_path, monkeypatch):
    """Only a CLI `isolation.CLI_CONFIG` knows loses an inherited endpoint. A plugin backend's CLI may take its
    endpoint from its account, so its environment is left alone."""
    from qda_agent.agents import BACKENDS

    monkeypatch.setitem(BACKENDS._entries, "qwen", tests_support.NoConfigCli)  # gone when the test ends
    monkeypatch.setenv("QWEN_BASE_URL", RELAY)
    assert "QWEN_BASE_URL" in endpoint_vars("qwen")  # the profile lists it, yet it is kept
    seen: dict[str, str | None] = {}

    def turn():
        seen["QWEN_BASE_URL"] = os.environ.get("QWEN_BASE_URL")
        return "ok"

    agent = {"resend_wait_s": 0.02, "backend": {"name": "qwen", "model": "m"}}
    config = write_config(tmp_path, seconds=0.5, agent=agent)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([turn]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and seen == {"QWEN_BASE_URL": RELAY}
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert not [e for e in events if e["ev"] == "env_dropped"]


def _witness(monkeypatch, fn):
    """Patch the sign-in witness the flow takes and the one `sign_in_refreshed` compares against."""
    monkeypatch.setattr(isolation, "sign_in_witness", fn)
    monkeypatch.setattr(isolation.cli_signin, "sign_in_witness", fn)


async def test_a_refresh_between_successful_goals_does_not_stop_the_search(tmp_path, monkeypatch):
    """The persistent-process case: goal 1 succeeds and its process keeps the old bind; the host refreshes in
    the resend wait; goal 2 refuses the old token. The goal-boundary witness alone sees the new file, so a
    flow that compared turn-start to turn-end would go FATAL on new/new. It must recycle and send again."""
    config = write_config(tmp_path, seconds=1.0)
    state = {"turns": 0, "now": 0}

    def witness(cfg):
        return {"claude": (state["now"], 0)}

    async def reply(prompt, **_):
        state["turns"] += 1
        if state["turns"] == 1:
            asyncio.get_running_loop().call_later(0.005, lambda: state.update(now=1))
            return "ok"  # the turn succeeds; the warm process keeps the old file
        if state["turns"] == 2:
            raise HarnessRefused("old mounted token revoked")
        return "ok"

    _witness(monkeypatch, witness)
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline"  # the refresh between successful turns did not end the search
    assert out["failed"] == 1 and len(worker.prompts) >= 3
    assert len(worker.sessions) == 2  # the warm process was let go before goal 2


async def test_a_refresh_between_goals_recycles_the_warm_process(tmp_path, monkeypatch):
    """A `claude`/`codex` stream process is reused across goals and keeps the mounts it started with: a
    refresh since the wall was built must recycle the process before the next goal, or the old bind outlives
    the refresh and the next goal fails on a file the host had already replaced."""
    config = write_config(tmp_path, seconds=0.6)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    _witness(monkeypatch, lambda cfg: {"claude": (2, 0)})  # the host refreshed the file
    # the process the flow spawns first was built with the old file; a new wall records the new one
    monkeypatch.setattr(
        isolation,
        "sign_in_mounted",
        lambda cfg: {"claude": (1, 0)} if len(worker.sessions) < 2 else {"claude": (2, 0)},
    )
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline"
    assert len(worker.sessions) == 2  # the warm process was let go once, before the goal
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert any(e["ev"] == "restart" and "sign-in refreshed" in e["why"] for e in events)


async def test_a_mid_turn_sign_in_refresh_resends_instead_of_stopping(tmp_path, monkeypatch):
    """A turn the provider refuses after the host refreshed the sign-in is sent again on the new file: the
    search lives on (a refresh during a long turn can revoke the token the turn was bound to)."""
    config = write_config(tmp_path, seconds=0.6)
    state = {"sign_in": 0, "turns": 0}

    async def reply(prompt, **_):
        state["turns"] += 1
        if state["turns"] == 1:
            state["sign_in"] = 1  # the host replaced the file while this turn ran
            raise HarnessRefused("expired")
        return "ok"

    _witness(monkeypatch, lambda cfg: {"claude": (state["sign_in"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline" and out["failed"] == 1
    assert len(worker.prompts) >= 2  # the refused turn was sent again
    failed = [
        json.loads(line)
        for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()
        if json.loads(line)["ev"] == "failed"
    ]
    assert failed and "sign-in refreshed" in failed[0]["why"]


@pytest.mark.parametrize("changes,failed,prompts", [(0, 0, 1), (1, 1, 2)])
async def test_a_refusal_is_fatal_and_one_retry_per_sign_in_change(
    tmp_path, monkeypatch, changes, failed, prompts
):
    """With no refresh the refusal is fatal at once; the file changed once, so the retry is sent and the
    second refusal (nothing new) is fatal: one retry per change, never a loop."""
    config = write_config(tmp_path, seconds=2.0)
    state = {"sign_in": 0, "turns": 0}

    async def reply(prompt, **_):
        state["turns"] += 1
        if state["turns"] == 1:
            state["sign_in"] = changes
        raise HarnessRefused("bad key" if not changes else "expired")

    _witness(monkeypatch, lambda cfg: {"claude": (state["sign_in"], 0)})
    worker = fakes.FakeAgentDriver(reply=reply)
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "fatal: HarnessRefused" and out["failed"] == failed
    assert len(worker.prompts) == prompts


async def test_contamination_aborts_before_the_agent(tmp_path):
    config = write_config(tmp_path, contamination="dirty")
    worker = fakes.FakeAgentDriver()
    with pytest.raises(ContaminationFound):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == [] and tmp_path / "run" not in testkit.HOSTS


async def test_tool_calls_are_logged(tmp_path):
    config = write_config(tmp_path, seconds=0.3)

    async def uses_tools(prompt, *, session, **_):
        await session.tool("Bash", {"command": "kl out/a"})
        return "ok"

    worker = fakes.FakeAgentDriver(reply=uses_tools)
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    lines = [json.loads(line) for line in (tmp_path / "run" / "tools.jsonl").read_text().splitlines()]
    assert lines and lines[0]["tool"] == "Bash" and lines[0]["input"] == {"command": "kl out/a"}


async def test_plain_flow_for_cli_without_goal(tmp_path):
    config = write_config(tmp_path, seconds=0.2)
    worker = fakes.FakeAgentDriver("pi", reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search_plain, agents={"worker": worker}, params={"config": config})
    assert worker.prompts and worker.prompts[0].startswith("Read TASK.md")


class BrokenHost(testkit.FakeHost):
    def cutoff(self):
        super().cutoff()
        raise RuntimeError("queue gone")

    def final(self):
        raise RuntimeError("ledger unreadable")


def broken_host(run_dir, sched, judge, **kwargs):
    return BrokenHost(run_dir, sched, judge, **kwargs)


async def test_host_failures_still_drain_stop_and_write_final(tmp_path, monkeypatch):
    monkeypatch.setattr(testkit, "broken_host", broken_host, raising=False)
    config = write_config(tmp_path, seconds=0.2, host="testkit:broken_host")
    run = tmp_path / "run"
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    final = json.loads((run / "FINAL.json").read_text())
    assert final["rule"] == "baseline" and final["baseline"].endswith("base/gptq")
    assert out["host_errors"] == 2 and run not in testkit.HOSTS  # drain and stop still ran
    errors = [json.loads(line) for line in (run / "flow.jsonl").read_text().splitlines()]
    assert [e["message"] for e in errors if e["ev"] == "host_error"] == ["queue gone", "ledger unreadable"]


async def test_production_scan_setting_starts(tmp_path):
    """configs/search.yaml's contamination setting (ngram + its guard) starts a run: no host is needed for it."""
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" / "search.yaml").read_text())
    guard, item = guard_of(tmp_path)
    config = write_config(
        tmp_path,
        seconds=0.2,
        contamination={**raw["contamination"], "guard": str(guard)},
    )
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline"
    host_judge = json.loads((tmp_path / "run" / "flow.jsonl").read_text().splitlines()[0])
    assert (
        host_judge["ev"] == "contamination"
        and host_judge["check"] == "ngram"
        and host_judge["findings"] == []
    )
    # the same policy reaches the judge, and a test item in the workdir stops the run before the agent
    (tmp_path / "item.md").write_text(item)
    config = write_config(
        tmp_path,
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        task=str(tmp_path / "item.md"),
    )
    from qda_agent.config import RunConfig

    assert RunConfig.load(config).judge_config()["contamination"].name == "ngram"
    worker = fakes.FakeAgentDriver()
    with pytest.raises(ContaminationFound):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []


async def test_write_final_failure_keeps_the_judges_pick(tmp_path, monkeypatch):
    class NoWrite(testkit.FakeHost):
        def write_final(self, path):
            raise OSError("disk full")

    monkeypatch.setattr(testkit, "nowrite_host", lambda r, s, j, **kw: NoWrite(r, s, j, **kw), raising=False)
    config = write_config(tmp_path, seconds=0.3, host="testkit:nowrite_host")
    run = tmp_path / "run"
    step = lambda: testkit.HOSTS[run].submit(testkit.build_ckpt(run / "work", "a"))  # noqa: E731
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([step]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    final = json.loads((run / "FINAL.json").read_text())
    assert out["rule"] == final["rule"] == "handin" and final["choice"]["dir"].endswith("out/a")
    assert final["chosen_by"] == "handin"
    assert run not in testkit.HOSTS  # stop ran


async def test_goal_sends_are_at_least_one_tick_apart_and_none_after_the_deadline(tmp_path):
    config = write_config(tmp_path, seconds=1.0, agent={"resend_wait_s": 0.3})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.0))  # returns at once, every time
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    ev = [json.loads(x) for x in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    t = [e["t"] for e in ev if e["ev"] == "goal"]
    assert 2 <= len(t) <= 4  # 1 s / 0.3 s tick, not hundreds
    assert all(b - a >= 0.29 for a, b in zip(t, t[1:], strict=False))
    cut = next(e["t"] for e in ev if e["ev"] == "cutoff")
    # Each goal's own time against the cut-off and the deadline, not its `left_s`: that is rounded to 0.1 s
    # for the log, so a goal sent inside the last 50 ms before the deadline reads 0.0.
    deadline = datetime.fromisoformat(next(e for e in ev if e["ev"] == "start")["deadline"]).timestamp()
    assert all(x < cut for x in t) and all(x < deadline for x in t)


async def test_scan_covers_the_calib_file_and_the_read_only_inputs(tmp_path):
    guard, item = guard_of(tmp_path, "q")
    inputs = tmp_path / "inputs"
    (inputs / "model").mkdir(parents=True)
    (inputs / "model" / "README.md").write_text("clean")
    (inputs / "model" / "model.safetensors").write_bytes(item.encode())  # weights are not text: skipped
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "clean"}\n')
    common = dict(
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        scheduler={"readable": [str(inputs)]},
        calib=str(calib),
    )
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    await fakes.run_fake(
        qda_search, agents={"worker": worker}, params={"config": write_config(tmp_path, **common)}
    )
    first = json.loads((tmp_path / "run" / "flow.jsonl").read_text().splitlines()[0])
    assert first["ev"] == "contamination" and first["findings"] == [] and first["files"] >= 4
    for dirty in (calib, inputs / "model" / "README.md"):
        dirty.write_text(item)
        worker = fakes.FakeAgentDriver()
        with pytest.raises(ContaminationFound, match=dirty.name):
            await fakes.run_fake(
                qda_search, agents={"worker": worker}, params={"config": write_config(tmp_path, **common)}
            )
        assert worker.prompts == []
        dirty.write_text("clean")


async def test_the_scan_skips_installed_packages_and_vendored_checkers_by_path(tmp_path, monkeypatch):
    """A run's four hits do not stop startup: a `site-packages` tree (a `.so.13` and a `.hpp` header in it, so
    a rule about binaries would not do) and qda_kl's vendored IFBench checkers are left out by exact path, and
    `resolved.yaml`'s provenance and the flow's log name those paths. A readable file outside them still stops."""
    from qda_agent import run_config as cfgmod
    from qda_agent.config import RunConfig

    guard, item = guard_of(tmp_path, "x")
    venv, site = fake_venv(tmp_path / "inputs" / "serve-venv")  # sysconfig names its own site-packages
    (site / "nvidia" / "cu13" / "lib").mkdir(parents=True)
    (site / "nvidia" / "cu13" / "lib" / "libcublas.so.13").write_text(item)
    hpp = site / "tilelang" / "3rdparty" / "composable_kernel" / "block_masking.hpp"
    hpp.parent.mkdir(parents=True)
    hpp.write_text(item)
    vend = tmp_path / "inputs" / "qda-kl" / "src" / "qda_kl" / "eval" / "vendor" / "ifbench"
    vend.mkdir(parents=True)
    (vend / "instructions.py").write_text(item)
    (vend / "instructions_test.py").write_text(item)
    monkeypatch.setattr(cfgmod, "_vendored_ifbench", lambda: vend.resolve())  # the package importlib finds
    (tmp_path / "inputs" / "prompts.jsonl").write_text("{}\n")
    calib = tmp_path / "calib.jsonl"
    calib.write_text("{}\n")
    common = dict(
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        scheduler={"readable": [str(tmp_path / "inputs")]},
        calib=str(calib),
        judge={"serve_python": str(venv / "bin" / "python")},
    )
    config = write_config(tmp_path, **common)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["ended"] == "deadline", out
    first = json.loads((tmp_path / "run" / "flow.jsonl").read_text().splitlines()[0])
    assert first["ev"] == "contamination" and first["findings"] == [], first
    skipped = sorted([*site_dirs(venv), str(vend.resolve())])
    assert first["skipped"] == skipped
    assert RunConfig.load(config).resolved()["provenance"]["contamination_skipped"] == skipped
    # the vendored checkers' text, copied anywhere else the agent reads, still refuses
    (tmp_path / "inputs" / "prompts.jsonl").write_text(item)
    with pytest.raises(ContaminationFound, match="prompts.jsonl"):
        await fakes.run_fake(
            qda_search,
            agents={"worker": fakes.FakeAgentDriver()},
            params={"config": write_config(tmp_path, **common)},
        )


@pytest.mark.parametrize(
    "where", ["calibration-in-site-packages", "copied-checker-tree", "prompt-in-site-packages"]
)
async def test_a_path_shaped_copy_is_not_an_install(tmp_path, where):
    """A directory merely named site-packages, or a copy of the checker tree, is not where a package is
    installed: the configured calibration file and the prompt there, and the copied checkers, all refuse."""
    guard, item = guard_of(tmp_path, "z")
    read = tmp_path / "inputs"
    read.mkdir()
    calib = tmp_path / "calib.jsonl"
    calib.write_text("{}\n")
    if where == "calibration-in-site-packages":
        target = read / "venv" / "lib" / "python3.12" / "site-packages" / "calibration.jsonl"
        calib = target
    elif where == "copied-checker-tree":
        target = read / "copied" / "qda_kl" / "eval" / "vendor" / "ifbench" / "instructions.py"
    else:
        target = read / "copied" / "site-packages" / "prompts.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(item)
    config = write_config(
        tmp_path,
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        scheduler={"readable": [str(read)]},
        calib=str(calib),
    )
    worker = fakes.FakeAgentDriver()
    with pytest.raises(ContaminationFound, match=target.name):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []


async def test_configured_data_under_an_installed_tree_is_still_scanned(tmp_path):
    """The calibration file is scanned even inside the serving venv's real site-packages: an installed tree is
    skipped for library files, not for the data the run configured."""
    guard, item = guard_of(tmp_path, "k")
    venv, site = fake_venv(tmp_path / "serve-venv")
    (site / "nvidia").mkdir(parents=True)
    (site / "nvidia" / "libcublas.so.13").write_text("clean\n")
    calib = site / "calibration.jsonl"
    calib.write_text(item)
    config = write_config(
        tmp_path,
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        scheduler={"readable": [str(site)]},
        calib=str(calib),
        judge={"serve_python": str(venv / "bin" / "python")},
    )
    worker = fakes.FakeAgentDriver()
    with pytest.raises(ContaminationFound, match="calibration.jsonl"):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []


async def test_the_input_scan_does_not_list_an_installed_tree(tmp_path, monkeypatch):
    """A readable venv's site-packages is not walked (minutes on a network file system for files that are
    dropped anyway); a configured file inside it is still found, and so is the rest of the venv."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import readable_inputs

    venv, site = fake_venv(tmp_path / "serve-venv")
    (site / "big" / "deep").mkdir(parents=True)
    (site / "big" / "deep" / "module.py").write_text("x = 1\n")
    (site / "data").mkdir()
    calib = site / "data" / "calibration.jsonl"
    calib.write_text("{}\n")
    (venv / "bin" / "activate").write_text("# a script\n")
    cfg = RunConfig.load(
        write_config(
            tmp_path,
            scheduler={"readable": [str(venv)]},
            calib=str(calib),
            judge={"serve_python": str(venv / "bin" / "python")},
        )
    )
    listed: list[Path] = []
    real_scandir = os.scandir

    def scandir(path="."):
        listed.append(Path(os.fsdecode(path)))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    files = {p.resolve() for p in readable_inputs(cfg)}
    assert calib.resolve() in files and (venv / "bin" / "activate").resolve() in files
    assert (site / "big" / "deep" / "module.py").resolve() not in files
    assert not any(d.is_relative_to(site.resolve() / "big") for d in listed), listed


@pytest.mark.parametrize("where", ["workspace", "calib", "prompt"])
async def test_a_test_item_still_refuses_outside_the_skipped_trees(tmp_path, where):
    """A real test item in the workspace, a calibration file or a prompt still refuses: the scope leaves out
    installed trees, not the data the agent can read."""
    guard, item = guard_of(tmp_path, "y")
    read = tmp_path / "read"
    read.mkdir()
    prompt = read / "prompts.jsonl"
    prompt.write_text("{}\n")
    calib = tmp_path / "calib.jsonl"
    calib.write_text("{}\n")
    if where == "workspace":
        target = tmp_path / "run" / "work" / "notes" / "leak.md"  # what the agent wrote
        target.parent.mkdir(parents=True)
        target.write_text(item)
    else:
        target = {"calib": calib, "prompt": prompt}[where]
        target.write_text(item)
    config = write_config(
        tmp_path,
        seconds=0.2,
        contamination={"name": "ngram", "guard": str(guard)},
        scheduler={"readable": [str(read)]},
        calib=str(calib),
    )
    worker = fakes.FakeAgentDriver()
    with pytest.raises(ContaminationFound, match=target.name):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []


async def test_symlinked_readable_entry_is_scanned_as_its_target(tmp_path):
    from qda_agent.config import RunConfig
    from qda_agent.flow import readable_inputs

    real = tmp_path / "elsewhere" / "notes.txt"
    real.parent.mkdir()
    real.write_text("text a job can read")
    (tmp_path / "link.txt").symlink_to(real)
    (tmp_path / "linkdir").symlink_to(real.parent)
    cfg = RunConfig.load(write_config(tmp_path, scheduler={"readable": ["link.txt", "linkdir"]}))
    files = readable_inputs(cfg)
    assert real.resolve() in files  # mounted through a link: still scanned


async def test_readable_is_relative_to_the_yaml(tmp_path, monkeypatch):
    from datetime import datetime

    from qda_agent.config import RunConfig
    from qda_agent.flow import readable_inputs

    (tmp_path / "ro").mkdir()
    (tmp_path / "ro" / "readme.txt").write_text("hello")
    config = write_config(tmp_path, scheduler={"readable": ["ro"]})
    other = tmp_path / "elsewhere-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    cfg = RunConfig.load(config)
    assert cfg.scheduler.readable == [(tmp_path / "ro").resolve()]
    assert (tmp_path / "ro" / "readme.txt").resolve() in readable_inputs(cfg)
    assert str((tmp_path / "ro").resolve()) in cfg.sched_config(datetime.now(UTC))["readable"]


async def test_the_config_reloads_after_the_run_dir_is_made(tmp_path):
    """`execute` makes `work/` before the flow starts, and the flow reloads the config there (`qda_search`)."""
    from qda_agent.config import RunConfig

    path = write_config(tmp_path)
    cfg = RunConfig.load(path)
    cfg.workdir.mkdir(parents=True)
    resolved = cfg.run_dir / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg.resolved()))
    again = RunConfig.load(resolved)
    assert again.run_dir == cfg.run_dir


async def test_the_input_scan_does_not_spend_the_gpu_budget(tmp_path):
    """The read-only inputs (a venv: ~25k files, over a minute) are scanned before the deadline is set."""
    from datetime import datetime

    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / "readme.txt").write_text("an input the agent's jobs can read")
    tests_support.SCANS.clear()
    config = write_config(tmp_path, seconds=0.5, contamination="slow", scheduler={"readable": [str(ro)]})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    (inputs_end, n_inputs), (workdir_end, _) = tests_support.SCANS
    assert n_inputs == 1
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    start = next(e for e in events if e["ev"] == "start")
    clock_start = datetime.fromisoformat(start["deadline"]).timestamp() - 0.5
    assert inputs_end <= clock_start < workdir_end


async def test_gate_runs_the_search_start_phase_alone_and_records_its_w(tmp_path):
    """`qda-agent gate` measures W on the search's own start code and starts no agent."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import gate

    cfg = RunConfig.load(write_config(tmp_path, seconds=60))
    out = await gate(cfg)
    run = tmp_path / "run"
    assert out["w_s"] >= 0 and json.loads((run / "gate.json").read_text())["w_s"] == out["w_s"]
    assert (run / "work" / "TASK.md").is_file()  # the workdir the search writes inside its clock
    events = [json.loads(line)["ev"] for line in (run / "flow.jsonl").read_text().splitlines()]
    assert "contamination" in events and events[-1] == "gate" and "start" not in events
    assert cfg.run_dir not in testkit.HOSTS  # the fake host forgets itself on stop()


@pytest.mark.parametrize(("level", "jobs"), [("none", "none"), ("sandbox", "bwrap")])
async def test_gate_records_the_isolation_its_jobs_ran_at(tmp_path, monkeypatch, level, jobs):
    """`gate.json` names the configured `isolation.level` and the job sandbox the run resolved from it."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import gate

    # a sandboxed level refuses a run dir under /tmp: this pins the record, not that check
    monkeypatch.setattr(isolation, "check", lambda cfg: None)
    cfg = RunConfig.load(write_config(tmp_path, seconds=60, isolation={"level": level}))
    await gate(cfg)
    got = json.loads((tmp_path / "run" / "gate.json").read_text())
    assert (got["isolation_level"], got["job_sandbox"]) == (level, jobs) == (level, cfg.job_sandbox())


async def test_result_json_carries_the_picks_generation_check(tmp_path, monkeypatch):
    """`result.json` repeats the winner's own long-generation check that FINAL.json records."""
    evidence = {
        "id": "g7",
        "valid": True,
        "max_tokens": 32768,
        "evaluator": "f83186f7",
        "checkpoint": {"math": {"n": 16, "cut": 0, "loops": 0, "mean_tokens": 1290.0}},
        "original": {"math": {"n": 16, "cut": 0, "loops": 0, "mean_tokens": 1261.0}},
    }

    class Checked(testkit.FakeHost):
        def final(self):
            got = super().final()
            got.generation_check = evidence
            return got

    monkeypatch.setattr(testkit, "checked_host", lambda r, s, j, **kw: Checked(r, s, j, **kw), raising=False)
    config = write_config(tmp_path, seconds=0.3, host="testkit:checked_host")
    run = tmp_path / "run"
    step = lambda: testkit.HOSTS[run].submit(testkit.build_ckpt(run / "work", "a"))  # noqa: E731
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([step]))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert out["generation_check"] == evidence
    assert json.loads((run / "result.json").read_text())["generation_check"] == evidence
    assert json.loads((run / "FINAL.json").read_text())["generation_check"] == evidence


def _r1_model(path: Path) -> Path:
    """A local model directory holding only a tiny tokenizer whose chat template is DeepSeek-R1's."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.WordLevel({"w0": 0, "[UNK]": 1}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
        "{% else %}<｜Assistant｜>{{ m['content'] }}<｜end▁of▁sentence｜>{% endif %}{% endfor %}"
    )
    fast.save_pretrained(path)
    return path


@pytest.mark.parametrize("rows", ["r1", "chatml"])
async def test_a_default_calib_file_in_another_models_template_refuses_the_launch(tmp_path, rows):
    """TASK.md's calibration file in ChatML on a model whose template is R1's: the search does not start."""
    from qda_agent.flow import CalibTemplateRefused, check_calib_template

    text = {
        "r1": "<｜User｜>question<｜Assistant｜>answer<｜end▁of▁sentence｜>",
        "chatml": "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>",
    }[rows]
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "plain words"}) + "\n" + json.dumps({"text": text}) + "\n")
    config = write_config(tmp_path, seconds=0.2, model=str(_r1_model(tmp_path / "model")), calib=str(calib))
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    if rows == "r1":
        out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
        assert out["ended"] == "deadline", out
        return
    with pytest.raises(
        CalibTemplateRefused, match=r"calib\.jsonl: 1 of 2 text rows .* row 1, .*'<\|im_start\|>'"
    ):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []
    from qda_agent.config import RunConfig

    with pytest.raises(CalibTemplateRefused, match="as messages rows"):
        check_calib_template(RunConfig.load(config))


async def test_a_clean_default_calib_file_needs_no_tokenizer(tmp_path):
    """Plain and messages rows pass without loading the model's tokenizer (the config's model is not local)."""
    from qda_agent.config import RunConfig
    from qda_agent.flow import check_calib_template

    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "plain words"}\n{"messages": [{"role": "user", "content": "hi"}]}\n{}\n')
    check_calib_template(RunConfig.load(write_config(tmp_path, calib=str(calib))))


async def test_a_gencheck_prompt_set_missing_a_domain_refuses_the_launch(tmp_path):
    """TASK.md evaluates four domains; a set that leaves one out refuses before the clock, no turn sent."""
    from qda_agent.flow import GencheckDomainsRefused

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps({"id": f"p{i}", "domain": domain, "prompt": f"question {i}"}) + "\n"
            for i, domain in enumerate(["math", "code", "science"])
        )
    )
    config = write_config(tmp_path, seconds=0.2, judge={"gencheck_prompts": str(prompts)})
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    with pytest.raises(GencheckDomainsRefused, match="instruction following"):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": config})
    assert worker.prompts == []


@pytest.mark.parametrize("rows", ["r1", "chatml"])
async def test_a_pinned_hub_model_is_resolved_with_its_revision(tmp_path, monkeypatch, rows):
    """`org/name@rev` reaches the hub as the repo and its revision (metadata files only), offline here."""
    import huggingface_hub
    from qda_agent.config import RunConfig
    from qda_agent.flow import CalibTemplateRefused, check_calib_template

    model, rev = _r1_model(tmp_path / "snapshot"), "0123456789abcdef0123456789abcdef01234567"
    calls = []

    def snapshot(repo, revision=None, allow_patterns=None):
        calls.append((repo, revision, allow_patterns))
        return str(model)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    text = {
        "r1": "<｜User｜>question<｜Assistant｜>answer",
        "chatml": "<|im_start|>user\nquestion<|im_end|>",
    }[rows]
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": text}) + "\n")
    cfg = RunConfig.load(write_config(tmp_path, model=f"example/model@{rev}", calib=str(calib)))
    if rows == "r1":
        check_calib_template(cfg)
    else:
        with pytest.raises(CalibTemplateRefused, match="of chatml"):
            check_calib_template(cfg)
    ((repo, revision, files),) = calls
    assert (repo, revision) == ("example/model", rev)
    assert files is not None and "*.json" in files and not any("safetensors" in f for f in files)


def test_a_refresh_replaces_the_baseline_views_after_the_kl_reorder(tmp_path):
    """The reviewer's probe: after the reorder `baselines/0` holds only the new start's view, none of the old's."""
    from qda_agent import flow
    from qda_agent.config import RunConfig

    a, b = tmp_path / "base" / "a", tmp_path / "base" / "b"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    (a / "tokenizer.json").write_text("{}")  # only A holds a tokenizer view
    cfg = RunConfig.load(write_config(tmp_path, seconds=0.2, baselines=[str(a), str(b)]))
    flow.install_toolset(cfg)
    assert (cfg.workdir / "baselines" / "0" / "tokenizer.json").is_file()
    cfg.baselines[:] = [b, a]  # the host's KL reorder: B is now candidate 0
    flow.install_toolset(cfg)
    assert (cfg.workdir / "baselines" / "0" / "config.json").is_file()
    assert not (cfg.workdir / "baselines" / "0" / "tokenizer.json").exists()  # A's view does not survive
    assert (cfg.workdir / "baselines" / "1" / "tokenizer.json").is_file()


def test_the_staged_views_are_read_only_and_a_refresh_still_replaces_them(tmp_path):
    """The input views are host-owned, not disk the agent may reclaim. An unwalled run gets real copies
    (`work/model`, `work/baselines/<N>`) and the host-owned overlay holds one of each: files 0444 and
    directories 0555, so an ordinary `rm -rf` fails. A refresh returns the rights (`remove_tree`) and
    replaces a sealed tree."""
    from qda_agent import flow
    from qda_agent.config import RunConfig

    model, a, b = tmp_path / "base" / "orig", tmp_path / "base" / "a", tmp_path / "base" / "b"
    for d in (model, a, b):
        d.mkdir(parents=True)
        (d / "config.json").write_text(f'{{"tag": "{d.name}"}}')
    (model / "tokenizer.json").write_text("{}")
    (a / "tokenizer.json").write_text("{}")  # only A holds a tokenizer view
    cfg = RunConfig.load(write_config(tmp_path, seconds=0.2, model=str(model), baselines=[str(a), str(b)]))
    flow.install_toolset(cfg)
    work, over = cfg.workdir, cfg.run_dir / "overlay"
    for root in (work / "model", over / "model", work / "baselines" / "0", over / "baselines" / "0"):
        assert (root / "config.json").stat().st_mode & 0o777 == 0o444, root
        assert root.stat().st_mode & 0o777 == 0o555, root
    if os.geteuid() != 0:  # root ignores the mode bits; the agent runs unprivileged as its own owner
        refusal = subprocess.run(["rm", "-rf", str(work / "model")], capture_output=True, text=True)
        assert refusal.returncode != 0 and (work / "model" / "config.json").is_file()
        assert (work / "model" / "tokenizer.json").stat().st_mode & 0o777 == 0o444
    (b / "tokenizer.json").write_text("{}")  # the refresh: B is now candidate 0 and has a tokenizer view
    cfg.baselines[:] = [b, a]
    flow.install_toolset(cfg)
    assert (work / "baselines" / "0" / "config.json").read_text() == '{"tag": "b"}'
    assert (work / "baselines" / "0" / "tokenizer.json").stat().st_mode & 0o777 == 0o444
    assert (work / "baselines" / "1" / "config.json").read_text() == '{"tag": "a"}'
    assert (over / "baselines" / "0").stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    ("said", "goal", "cause"),
    [
        (
            "function_call_output requires call_id or previous_response_id; ensure store=true",
            True,
            "goal_flow",
        ),
        ("function_call_output requires previous_response_id", True, "goal_flow"),
        ("function_call_output requires call_id or previous_response_id; ensure store=true", False, None),
        ("400: Invalid 'input[0].call_id': string too long", True, None),
    ],
)
def test_only_the_stateless_endpoints_refusal_names_the_goal_flow(said, goal, cause):
    """The goal-flow fix is for the endpoint's refusal under the goal flow, not any error that names `call_id`."""
    from hmz.flows import HarnessUnrecoverable
    from qda_agent.flow import _fatal_cause

    assert _fatal_cause(HarnessUnrecoverable(said), goal=goal) == cause
