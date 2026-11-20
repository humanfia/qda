"""The contrast test: every guard a command refuses on, refused by the real command and by `qda doctor`.

For each guard that can be checked in advance there is one failing case. The real command runs as a user runs
it (`qda-agent run CONFIG`, `qda build`'s own entry), and so does doctor (`qda doctor --search CONFIG`,
`qda doctor`): the command must refuse with the guard's sentence, and doctor must print that sentence as the
guard's MISS and count it. The two sides are separate code paths over one guard list, so a change to one side
only - a command that stops walking its guards, a doctor that drops or rewords a line - fails here.

Where a condition cannot be made on a CPU test machine (a faulty card, a Python without `os.pidfd_open`, a
machine that cannot hold the fence, a node shared with other processes), the case replaces the guard's leaf
fact (the function both sides call) and nothing else: what is compared is the walk, not the fact.

`test_every_advance_guard_has_a_case` keeps the list complete: a new guard without a case fails it.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from qda_agent.cli import START_GUARDS
from qda_agent.judge.startup import STATIC_GUARDS
from qda_quant.checks import BUILD_GUARDS
from qda_quant.guards import BUILD, SEARCH_START

#: The tests' fixtures. `_search_config` is a copy of test_cli's, not an import: `test_cli` is also a module of
#: qda-kl's tests, and whichever is imported first is the one every later `from test_cli import` gets.
FIXTURES = Path(__file__).parent / "fixtures"


def _search_config(tmp_path: Path, **judge) -> Path:
    """The scripted search config with its paths made absolute, and `judge` overridden."""
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "run")
    raw["baselines"] = [str((FIXTURES / b).resolve()) for b in raw["baselines"]]
    raw["judge"]["evaluator"] = str((FIXTURES / raw["judge"]["evaluator"]).resolve())
    for key in ("kl_ref", "kl_holdout_ref"):
        if raw["judge"].get(key):
            raw["judge"][key] = str((FIXTURES / raw["judge"][key]).resolve())
    raw["judge"].update(judge)
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


Setup = Callable[[Path, pytest.MonkeyPatch], tuple[Path, list[str]]]


def _config(tmp_path: Path, *overrides: str, **judge) -> tuple[Path, list[str]]:
    return _search_config(tmp_path, **judge), list(overrides)


def _run_dir_not_empty(tmp_path, monkeypatch):
    path, over = _config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "FINAL.json").write_text("{}\n")
    return path, over


def _run_dir_in_tmp(tmp_path, monkeypatch):
    # the rule's leaf: where the run dir is (a test's tmp_path may or may not be under a temporary directory)
    why = f"run_dir {tmp_path / 'run'} is under /var/tmp, which the agent can read and write at isolation.level sandbox"
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: why)
    return _config(tmp_path, "isolation.level=sandbox")


def _cards_not_visible(tmp_path, monkeypatch):
    why = "cards [0, 1] are not among the cards CUDA_VISIBLE_DEVICES=5 gives this shell"
    monkeypatch.setattr("qda_agent.isolation.devices.invisible_cards", lambda cards, *a, **k: why)
    return _config(tmp_path, "scheduler.gpus=true")


def _python_without_pidfd(tmp_path, monkeypatch):
    why = "this Python has no os.pidfd_open, and Humanize 0.1's fence crashes without it"
    monkeypatch.setattr("qda_agent.cli.pidfd_refusal", lambda cfg: why)
    return _config(tmp_path)


def _sandbox_cannot_hold(tmp_path, monkeypatch):
    why = "this host cannot hold the agent's CLI to its fence (no Landlock ABI 4)"
    monkeypatch.setattr("qda_agent.isolation.hold_problem", lambda cfg: why)
    return _config(tmp_path)


def _pid_rule(tmp_path, monkeypatch):
    why = "pid: shared gives the agent's process the host's pid namespace, and this node is shared"
    monkeypatch.setattr("qda_agent.isolation.pid_refusal", lambda cfg: why)
    return _config(tmp_path)


def _work_dir_exists(tmp_path, monkeypatch):
    work = tmp_path / "work-elsewhere"
    work.mkdir()
    return _config(tmp_path, f"work_dir={work}")


def _faulty_card(tmp_path, monkeypatch):
    monkeypatch.setattr("qda_agent.isolation.devices.invisible_cards", lambda cards, *a, **k: None)
    fault = "card 0: 2 uncorrectable ECC error(s) since the driver loaded"
    monkeypatch.setattr("qda_agent.doctor.card_health", lambda cards, proc=None: (fault, None))
    return _config(tmp_path, "scheduler.gpus=true")


def _no_holdout(tmp_path, monkeypatch):
    return _config(tmp_path, kl_holdout_ref=None)


def _evaluator_with_a_link(tmp_path, monkeypatch):
    import shutil

    tree = tmp_path / "evaluator"
    shutil.copytree(FIXTURES / "../../examples/standin_kl", tree)
    (tree / "planted.py").symlink_to(tree / "kl.py")
    return _config(tmp_path, evaluator=str(tree))


def _qda_not_found(tmp_path, monkeypatch):
    return _config(tmp_path, "toolset.qda=[no-such-qda-xyz]")


def _disk_below_floor(tmp_path, monkeypatch):
    # one fixed reading for both sides: the live free space moves between two calls on a busy disk
    from qda_agent.room import Room

    short = Room(
        tmp_path, free_bytes=5 * 10**9, free_inodes=10**6, total_bytes=100 * 10**9, total_inodes=10**7
    )
    monkeypatch.setattr("qda_agent.judge.custody.room", lambda path: short)
    return _config(tmp_path)


#: One failing case per search-start guard that can be checked in advance, by guard name.
SEARCH_CASES: dict[str, Setup] = {
    "run dir": _run_dir_not_empty,
    "run dir place": _run_dir_in_tmp,
    "cards visible": _cards_not_visible,
    "python fence": _python_without_pidfd,
    "sandbox hold": _sandbox_cannot_hold,
    "pid rule": _pid_rule,
    "work dir": _work_dir_exists,
    "card health": _faulty_card,
    "calibration index": _no_holdout,
    "evaluator tree": _evaluator_with_a_link,
    "judge qda": _qda_not_found,
    "disk space": _disk_below_floor,
}


def _linked_prompts(tmp_path: Path) -> str:
    real = tmp_path / "prompts.jsonl"
    domains = ("math", "code", "science", "instruction")  # the set the run's input check asks for
    real.write_text(
        "".join(f'{{"id": "{d}", "domain": "{d}", "prompt": "a {d} question"}}\n' for d in domains)
    )
    link = tmp_path / "linked-prompts.jsonl"
    link.symlink_to(real)
    return str(link)


#: Guards a config file cannot reach, as (judge settings, scheduler settings, the config file's own override):
#: the loader refuses custody and receipts without bwrap first (`RunConfig`), and it resolves a linked prompt
#: file to its target (`configfile`), so the link refusal is the host API's. `test_a_guard_only_the_host_api_reaches`
#: compares the host's own refusal (`make_host`) with doctor's walk of the guard, and what the loader does with
#: the same setting on both sides.
HOST_CASES: dict[str, Callable[[Path], tuple[dict, dict, dict]]] = {
    "custody": lambda t: (
        {"custody_dir": str(t / "custody")},
        {"job_sandbox": "none"},
        {"custody_dir": "custody"},
    ),
    "build receipts": lambda t: (
        {"require_build_receipt": True},
        {"job_sandbox": "none"},
        {"require_build_receipt": True},
    ),
    "prompt file": lambda t: ({"gencheck_prompts": _linked_prompts(t)}, {}, {}),
}

#: ``nvidia-smi``'s listing of two H100s (index, uuid, name, driver, ECC errors, retired pages pending).
HEALTHY = "0, GPU-0000, NVIDIA H100 80GB HBM3, 550.54.15, 0, [N/A]\n1, GPU-0001, NVIDIA H100 80GB HBM3, 550.54.15, 0, [N/A]\n"


def _listing(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """``nvidia-smi`` (the leaf both sides read, through `qda_quant.checks`) lists `text`."""
    from qda_quant import checks

    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, text, ""))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _card_with_ecc_errors(monkeypatch):
    _listing(monkeypatch, HEALTHY.replace("550.54.15, 0,", "550.54.15, 2,", 1))


def _driver_too_old_for_torch(monkeypatch):
    """Torch for CUDA 13 on a 570 driver (CUDA 12.8); the driver's answer is the leaf both sides call."""
    from qda_quant import checks

    _listing(monkeypatch, HEALTHY)
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (12080, ""))


#: The build's guards that can be checked in advance, with their cases (`test_the_build_and_doctor_refuse_alike`).
BUILD_CASES: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "card health": _card_with_ecc_errors,
    "CUDA driver": _driver_too_old_for_torch,
}


def test_every_advance_guard_has_a_case():
    """A guard added to a command's list without a contrast case fails here, so no guard goes unchecked."""
    search = [g.name for g in (*START_GUARDS, *STATIC_GUARDS) if g.advance and SEARCH_START in g.commands]
    assert sorted(search) == sorted([*SEARCH_CASES, *HOST_CASES]), (
        "every search-start guard has exactly one case"
    )
    build = [g.name for g in BUILD_GUARDS if g.advance and BUILD in g.commands]
    assert sorted(build) == sorted(BUILD_CASES)


def _error_line(err: str) -> str:
    """The `error:` line of a refusal on stderr (`qda_quant.exit.refusal`)."""
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0].removeprefix("error: ")


@pytest.mark.parametrize("name", sorted(SEARCH_CASES))
def test_the_search_refuses_on_every_guard(name, tmp_path, monkeypatch, capsys):
    """`qda-agent run` refuses with the guard's own sentence, in one line, and creates nothing.

    Doctor no longer repeats these: the search's own conditions are refused by the search when it starts
    (`test_doctor_search.py` has what `qda doctor --search` still decides)."""
    from qda_agent import cli as agent

    path, overrides = SEARCH_CASES[name](tmp_path, monkeypatch)
    sets = [a for o in overrides for a in ("--set", o)]

    code = agent.main(["run", str(path), *sets])
    err = capsys.readouterr().err
    assert code in (2, 3) and "Traceback" not in err, err
    _error_line(err)  # one `error:` line, the guard's sentence
    if name in {
        g.name for g in START_GUARDS
    }:  # `execute`'s pre-mkdir block; the host's run after resolved.yaml
        assert not (tmp_path / "run" / "resolved.yaml").exists(), "a refused start creates nothing"


@pytest.mark.parametrize("name", sorted(HOST_CASES))
def test_a_guard_only_the_host_api_reaches(name, tmp_path, capsys):
    """The host refuses with the guard's sentence (`make_host`), and doctor's walk of the same guard prints it.

    From a config file the loader decides first: it refuses custody and receipts without bwrap, and then
    `qda doctor --search` and `qda-agent run` print the same sentence; a linked prompt file is resolved to its
    target, which neither side refuses.
    """
    from datetime import UTC, datetime

    from qda_agent import cli as agent
    from qda_agent.config import RunConfig
    from qda_agent.judge import make_host
    from qda_agent.judge.config import JudgeConfig
    from qda_agent.judge.startup import Start
    from qda_agent.sched import SchedConfig
    from qda_quant import cli as qda
    from qda_quant.doctor import guard_lines
    from qda_quant.guards import all_problems

    judge_set, sched_set, loader_set = HOST_CASES[name](tmp_path)
    cfg = RunConfig.load(_search_config(tmp_path))
    sched = {**cfg.sched_config(datetime.now(UTC)), **sched_set}
    judge = {**cfg.judge_config(), **judge_set}
    with pytest.raises(ValueError) as refused:
        make_host(cfg.run_dir, sched, judge, autotick=False)
    start = Start.of(
        JudgeConfig.model_validate(judge), SchedConfig.model_validate(sched), cfg.run_dir, cfg.workdir
    )
    lines = {c.name: c for c in guard_lines(all_problems(STATIC_GUARDS, {SEARCH_START}, start))}
    assert lines[name].missed and lines[name].detail == " ".join(str(refused.value).split())

    if not loader_set:
        return
    path = _search_config(tmp_path, **loader_set)
    sets = [a for k, v in sched_set.items() for a in ("--set", f"scheduler.{k}={v}")]
    assert qda.main(["doctor", "--search", str(path), *sets]) == 2, "the loader's refusal is the input"
    out = capsys.readouterr().out
    assert agent.main(["run", str(path), *sets]) == 2
    loaded = _error_line(capsys.readouterr().err)
    assert any(x.startswith(f"MISS run config: {loaded}") for x in out.splitlines()), out


@pytest.mark.parametrize("name", sorted(BUILD_CASES))
def test_the_build_and_doctor_refuse_alike(name, tmp_path, monkeypatch, capsys):
    """`qda build`'s entry refuses on what a guard rejects before it reads the model, and `qda doctor` prints
    the same sentence as that guard's MISS."""
    import torch
    from qda_quant import cli as qda

    engine = importlib.import_module("qda_quant.engine.build")  # the module (the package exports `build`)
    from qda_quant.recipe import Recipe
    from qda_quant.step import Refused

    BUILD_CASES[name](monkeypatch)
    monkeypatch.setattr(engine, "resolve_devices", lambda devices: [torch.device("cuda", 0)])

    def never(*a, **k):
        raise AssertionError("the build went past its guards")

    monkeypatch.setattr(engine, "_refuse_unbuildable_triton", never)
    with pytest.raises(Refused) as refused:
        recipe = Recipe.from_yaml("steps: []\ncalib: {file: {path: rows.jsonl}}\n")
        engine.build("never/read", recipe, out_dir=tmp_path / "out", devices=["cuda:0"])
    assert refused.value.code == 3

    qda.main(["doctor"])
    out = capsys.readouterr().out
    assert f"MISS {name}: {refused.value} (fix: {refused.value.fix})" in out.splitlines(), out


#: The engine start's advance guards, with their cases (`test_an_engine_start_and_doctor_refuse_alike`): its own
#: list (`SERVE_GUARDS`) and the build's guards an engine start refuses on too (`checks.refuse_serve`).
SERVE_CASES: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "CUDA driver": _driver_too_old_for_torch,
}


def test_every_serve_guard_has_a_case():
    from qda_quant.checks import SERVE_GUARDS
    from qda_quant.guards import SERVE

    serve = [g.name for g in (*SERVE_GUARDS, *BUILD_GUARDS) if g.advance and SERVE in g.commands]
    assert sorted(serve) == sorted(SERVE_CASES)


@pytest.mark.parametrize("name", sorted(SERVE_CASES))
def test_an_engine_start_and_doctor_refuse_alike(name, tmp_path, monkeypatch, capsys):
    """The engine start (`vllm_environ`, which every qda engine enters) refuses with a guard's sentence, and
    `qda doctor` prints it as that guard's MISS."""
    from qda_quant import cli as qda
    from qda_quant.output import vllm_env
    from qda_quant.step import Refused

    SERVE_CASES[name](monkeypatch)
    with pytest.raises(Refused) as refused, vllm_env.vllm_environ({}):  # what every engine start enters
        pass
    assert refused.value.code == 3
    qda.main(["doctor"])
    out = capsys.readouterr().out
    assert f"MISS {name}: {refused.value} (fix: {refused.value.fix})" in out.splitlines(), out


def test_the_agent_api_guard_has_a_case():
    from qda_agent.api_route import AGENT_GUARDS

    assert [g.name for g in AGENT_GUARDS if g.advance] == ["agent API reachable"]


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_a_node_without_a_route_to_the_api_is_refused_alike(
    route, tmp_path, monkeypatch, capsys, real_api_route
):
    """A node with no route to the agent's API ran its search to the deadline with 0 tool calls and exit 0,
    and nothing refused. `qda-agent run` (driving a real CLI) now walks the `agent API reachable` guard: one
    TCP connect, or one CONNECT to HTTPS_PROXY, by the route the turns take."""
    import socket as socket_module

    from qda_agent import cli as agent

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    for name in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    asked: list[tuple[str, int]] = []

    def refused(address, timeout=None):
        asked.append(address)
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(socket_module, "create_connection", refused)
    if route == "proxy":
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    path = _search_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["agent"] = {"backend": "claude"}
    path.write_text(yaml.safe_dump(raw))
    code = agent.main(["run", str(path)])
    err = capsys.readouterr().err
    assert code == 3 and "Traceback" not in err, err
    refusal = _error_line(err)
    want = ("api.anthropic.com", 443) if route == "direct" else ("proxy.invalid", 3128)
    assert asked and asked[0] == want, "the route the turn takes: direct, or the CLI's own HTTPS_PROXY"
    assert not (tmp_path / "run" / "resolved.yaml").exists(), "refused before the run creates anything"
    assert "api" in refusal or "proxy" in refusal, refusal  # the route it could not take


#: The settings `testkit.scripted` gives a scripted search on CPU stand-ins, as `qda-agent search` overrides.
_CPU_RUN = ["sandbox=none", "scheduler.gpus=false", "budget_gpu_hours=0.0008"]


def _search_runs(monkeypatch) -> list[object]:
    """`qda-agent run`'s `execute` is the real one, with the scripted agent driver a test run uses (as
    `testkit.scripted` passes it); returns the configs it was called with, so a test sees that the run started."""
    from hmz.sdk import fakes
    from qda_agent import cli as agent
    from testkit import _flow_worker, shell_worker

    started: list[object] = []
    real = agent.execute

    def execute(cfg, **kwargs):
        started.append(cfg)
        worker = shell_worker(cfg) if cfg.commands else _flow_worker(cfg)  # as `testkit.scripted` picks it
        driver = fakes.FakeAgentDriver(reply=worker, model="script/test_scripted")
        return real(cfg, worker=driver)

    monkeypatch.setattr(agent, "execute", execute)
    return started


def test_a_search_runs_past_a_doctor_miss_the_run_does_not_refuse(tmp_path, monkeypatch, capsys):
    """`qda-agent search` prints doctor's report and no longer stops on its status. A search on CPU
    stand-ins with no card visible: doctor's `GPU` line is a MISS (exit 3), and the run, which needs no card,
    starts and finishes."""
    from qda_agent import cli as agent

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CUDA sees no card: doctor's GPU line misses
    started = _search_runs(monkeypatch)
    path = _search_config(tmp_path)
    sets = [a for o in _CPU_RUN for a in ("--set", o)]

    code = agent.main(["search", str(path), *sets])
    captured = capsys.readouterr()
    assert any(x.startswith("MISS GPU: ") for x in captured.out.splitlines()), captured.out
    assert len(started) == 1, "the run started after doctor's MISS"
    assert "qda doctor reported a miss above" in captured.err
    assert code == 0, captured.err
    assert (tmp_path / "run" / "FINAL.json").is_file()


def test_a_search_that_the_run_refuses_exits_with_the_runs_refusal(tmp_path, monkeypatch, capsys):
    """Where the run refuses (a run dir that is not empty), the search's status is the run's own
    refusal (exit 2), in one line, and the run creates nothing."""
    from qda_agent import cli as agent

    started = _search_runs(monkeypatch)
    path = _search_config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "FINAL.json").write_text("{}\n")
    sets = [a for o in _CPU_RUN for a in ("--set", o)]

    code = agent.main(["search", str(path), *sets])
    captured = capsys.readouterr()
    _error_line(captured.err)  # the run's own refusal, in one line
    assert len(started) == 1, "the run itself decided"
    assert code == 2 and "Traceback" not in captured.err, captured.err
    assert not (run / "resolved.yaml").exists(), "a refused start creates nothing"
