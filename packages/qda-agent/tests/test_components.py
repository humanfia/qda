"""Toolset components: presets, the config diff between toolsets, TASK.md, the verbs the host allows, the wiki."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import Field, ValidationError
from qda_agent import isolation
from qda_agent.commands import request
from qda_agent.components import COMPONENTS, Component, Fisher, Gencheck, Run
from qda_agent.config import RunConfig
from qda_agent.run_sources import check
from qda_agent.sched import LocalBackend, SchedConfig, Scheduler
from qda_agent.sched.clock import FakeClock
from qda_agent.sched.sandbox import bwrap_argv
from qda_agent.toolsets import Qda, Toolset
from testkit import ScriptedWorker, scripted, shell_worker
from tests_support import agent, filled

ROOT = Path(__file__).parents[1]
CONFIGS, EXAMPLES = ROOT / "configs", ROOT / "examples"
FIXTURES = Path(__file__).parent / "fixtures"
V01 = ["qda", "kl", "qsched", "handin"]


def keys(cfg: RunConfig) -> list[str]:
    return [c.name for c in cfg.toolset.components]


def _base(run_dir: Path | str, **over) -> dict:
    """A minimal valid config for the refusal cases below; `over` replaces or adds fields."""
    return {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": str(run_dir),
        "judge": {"eval_max_tokens": 32768, **over.pop("judge", {})},
        **over,
    }


def _abs_refs(raw: dict) -> None:
    """The fixture's reference paths made absolute: its config is copied outside the fixtures directory."""
    raw["judge"]["kl_ref"] = str(FIXTURES / "ref")
    raw["judge"]["kl_holdout_ref"] = str(FIXTURES / "holdout")


def test_search_yaml_is_the_qda_preset(tmp_path):
    cfg = RunConfig.load(filled(CONFIGS / "search.yaml", tmp_path))
    assert cfg.toolset.name == "qda" and keys(cfg) == list(Qda.preset)
    assert (
        RunConfig(
            contamination="none", model="m", format="w4a16", budget_gpu_hours=1, cards=[0], run_dir=tmp_path
        ).toolset.name
        == "qda"
    )


def test_the_v01_toolset_is_a_component_list_diff(tmp_path):
    """v0.1's four commands = the `qda` preset without gencheck, wiki, fisher, probe and the inputs."""
    path = filled(CONFIGS / "search.yaml", tmp_path)
    full = RunConfig.load(path)
    four = RunConfig.load(path, [f"toolset.components=[{', '.join(V01)}]"])
    extra = {
        "bpw",
        "gencheck",
        "wiki",
        "fisher",
        "probe",
        "data",
        "serve",
        "baselines",
        "views",
        "source",
    }
    assert keys(four) == V01 and set(keys(full)) - set(V01) == extra
    a, b = full.model_dump(mode="json"), four.model_dump(mode="json")
    a["toolset"].pop("components"), b["toolset"].pop("components")
    assert a == b
    assert four.toolset.commands(four.run) == tuple(V01)
    assert full.toolset.commands(full.run) == (
        "qda",
        "bpw",
        "kl",
        "gencheck",
        "qsched",
        "handin",
    )  # bpw and gencheck more
    assert four.toolset.subcommands(four.run) == {"qda": ["build", "plan"]}
    assert full.toolset.subcommands(full.run) == {"qda": ["build", "plan", "fisher"]}


def test_bpw_is_offered_on_vllm_only(tmp_path):
    """The judge's bits-per-weight rule is the vLLM budget's; a llama.cpp search states .gguf weight bytes."""
    toolset = Qda()
    vllm, gguf = Run(workdir=tmp_path, engine="vllm"), Run(workdir=tmp_path, engine="llama.cpp")
    assert "bpw" in toolset.commands(vllm) and "bpw" not in toolset.commands(gguf)
    assert toolset.commands(gguf) == tuple(c for c in toolset.commands(vllm) if c != "bpw")
    assert any("`bpw DIR`" in s for s in toolset.sections(vllm))
    assert not any("`bpw" in s for s in toolset.sections(gguf))
    assert not any("bits per quantized weight" in s for s in toolset.sections(gguf))


def test_fisher_states_its_limits_on_vllm_only(tmp_path):
    """`qda fisher` prices the vLLM budget's bits; its TASK.md item says what it gives and what it cannot see."""
    toolset = Qda()
    vllm, gguf = Run(workdir=tmp_path, engine="vllm"), Run(workdir=tmp_path, engine="llama.cpp")
    assert "fisher" in toolset.subcommands(vllm)["qda"] and "fisher" not in toolset.subcommands(gguf)["qda"]
    (item,) = [s for s in toolset.sections(vllm) if "`qda fisher map" in s]
    assert "ranks and does not calibrate" in item
    assert "routing flips are discrete and second order cannot see them" in " ".join(item.split())
    # No measured result reaches the agent (P3): the item states what the map is, not what it found.
    assert not any(measured in item for measured in ("48 s", "7 min", "Spearman", "97 %", "9.9 s", "~8x"))


def test_public_is_refused_and_bad_components_are_named(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(ValueError, match="public.*not available"):
        RunConfig.model_validate({**base, "toolset": "public"})
    with pytest.raises(ValueError, match="twice"):
        RunConfig.model_validate({**base, "toolset": {"name": "qda", "components": ["kl", "kl"]}})
    with pytest.raises(ValueError, match="qda_agent.components plugin 'no-such-part'"):
        RunConfig.model_validate({**base, "toolset": {"name": "qda", "components": ["no-such-part"]}})


def test_a_resolved_yaml_loads_back_with_its_components(tmp_path):
    cfg = RunConfig.load(filled(CONFIGS / "search.yaml", tmp_path), ["toolset.components=[kl, handin, wiki]"])
    out = tmp_path / "resolved.yaml"
    out.write_text(yaml.safe_dump(cfg.resolved()))
    assert keys(RunConfig.load(out)) == ["kl", "handin", "wiki"]


def test_every_builtin_component_is_registered():
    assert sorted(COMPONENTS) == sorted(Qda.preset)


def test_the_fisher_line_reads_the_tool_card_and_cache_rules():
    """One owner: the tool's help, its epilog and TASK.md state the card and cache rules from one place."""
    from qda_quant.intake.fisher import CACHE_RULE, CARD_RULE

    text = " ".join(Fisher.section_text.split())
    assert " ".join(CARD_RULE.split()) in text and " ".join(CACHE_RULE.split()) in text


def _scripted(tmp_path: Path, components: list[str], argv: list[list[str]]) -> tuple[Path, list[dict]]:
    """A scripted worker sends each `argv` (command first) to the host."""
    run, asked = tmp_path / "run", []

    def script(cfg: RunConfig) -> ScriptedWorker:
        worker = shell_worker(cfg)
        for a in reversed(argv):
            worker.steps.insert(
                0, lambda a=a: asked.append(request(str(cfg.host_socket), a[0], a[1:], ".")) or "asked"
            )
        return worker

    over = [f"toolset.components=[{', '.join(components)}]"]
    assert scripted(FIXTURES / "scripted-search.yaml", run, 8, script=script, overrides=over) == 0
    return run, asked


def test_the_qda_preset_in_a_scripted_flow(tmp_path):
    """The preset's TASK.md names every part; its verbs reach the host; the wiki is in the workdir."""
    run, asked = _scripted(tmp_path, list(Qda.preset), [["qda", "plan", "x.yaml"], ["qda", "probe"]])
    work = run / "work"
    task = (work / "TASK.md").read_text()
    for line in (
        "`qda build CONFIG.yaml",
        "`bpw DIR`",
        "`kl DIR`",
        "`gencheck DIR [--max-tokens N] [--per-domain K]`",
        "`qsched`",
        "`handin DIR [DIR...]`",
        "quant-methods",
    ):
        assert line in task, line
    assert (
        "Six commands in `bin/`" in task and task.count("lists them).\n") == 0
    )  # the last item ends the list
    assert "`qda mix" not in task and "`qda capture" not in task
    assert asked[0]["exit"] == 0 and asked[0]["stdout"].strip() == "stand-in qda: plan x.yaml"
    assert asked[1]["exit"] != 0 and "no `qda probe`" in asked[1]["stderr"]
    assert "Your workspace also has:\n- `.claude/skills/quant-methods/`" in task
    for d in (".claude/skills", ".agents/skills"):
        for root in (run / "overlay", work):  # host copy and agent copy in this CPU test
            skill = root / d / "quant-methods"
            assert (skill / "SKILL.md").is_file() and (skill / "references" / "capability.md").is_file()


def test_a_qda_verb_no_component_adds_is_refused_and_task_md_is_silent(tmp_path):
    """A verb the toolset does not give (there is no `qda mix` component) is refused, whatever the option form."""
    asks = [
        ["qda", "mix", "--list-groups"],
        ["qda", "--cards", "1", "plan", "x.yaml"],
        ["qda", "--", "mix", "--list-groups"],
        ["qda", "--cards", "1", "--", "mix"],
        ["qda", "--car", "1", "mix"],
        ["qda", "--bogus", "mix"],
    ]
    run, asked = _scripted(tmp_path, V01, asks)
    task = (run / "work" / "TASK.md").read_text()
    assert "qda mix" not in task and "quant-methods" not in task and "qda capture" not in task
    assert task.rstrip().count("(`handin status` lists them).") == 1  # v0.1's last item, ending the sentence
    assert asked[0]["exit"] != 0 and "no `qda mix`; it has: build, plan" in asked[0]["stderr"]
    assert asked[1]["exit"] == 0  # the verb after the options is found
    for a in asked[
        2:
    ]:  # `--` ends the options, an abbreviated option takes its value, an unknown one is the verb
        assert a["exit"] != 0 and "this search has no `qda " in a["stderr"], a
    assert not (run / "work" / ".claude").exists()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
def test_a_bind_from_a_host_copy_is_read_only_inside_a_job(tmp_path):
    """The job sandbox binds a host-owned copy over its mount point in the writable workdir, read-only."""
    work, host = tmp_path / "work", tmp_path / "overlay"
    work.mkdir()
    (host / "wiki").mkdir(parents=True)
    (host / "wiki" / "page.md").write_text("card\n")
    script = (
        "cat wiki/page.md; (echo x >> wiki/page.md) 2>/dev/null || echo append-refused; "
        "touch wiki/new 2>/dev/null || echo create-refused; mv wiki w2 2>/dev/null || echo rename-refused; "
        "touch mine && echo own-ok"
    )
    argv = bwrap_argv(
        ["/bin/sh", "-c", script], str(work), writable=[work], binds={work / "wiki": host / "wiki"}
    )
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert r.stdout.split() == ["card", "append-refused", "create-refused", "rename-refused", "own-ok"], r
    assert (work / "wiki").is_dir() and not (work / "wiki" / "page.md").exists()  # only an empty mount point
    with pytest.raises(ValueError, match="lies in a writable path"):
        bwrap_argv(["true"], str(work), writable=[work], binds={work / "x": work / "mine"})


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
def test_a_planted_link_never_redirects_a_job_bind(tmp_path):
    """Job A tries to swap the skill's parents for links out of the workdir; the host plants such links too.

    The host now refuses a mount point that is a link or the wrong kind (it deletes nothing), so job B
    gets no redirect: the planted entries stay as the agent left them, the host's copies are not mounted
    there, and the run dir (its secret) is not readable."""
    cfg = _inputs(tmp_path)
    cfg = cfg.model_copy(update={"toolset": Qda(components=[*cfg.toolset.components, "wiki"])})
    _prepare(cfg)
    secret = cfg.run_dir / "judge" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("TOPSECRET\n")
    work = cfg.workdir
    attack = (
        "(mv .claude .old) 2>/dev/null || echo parent-busy; "
        "(mv .claude/skills .claude/old) 2>/dev/null || echo skills-ro; "
        "(rm -rf .agents/skills/quant-methods) 2>/dev/null; [ -f .agents/skills/quant-methods/SKILL.md ] && echo kept; "
        "(rm -rf baselines) 2>/dev/null; [ -d baselines/0 ] && echo baselines-kept; "
        "(ln -s ../.. .agents/skills/x) 2>/dev/null || echo link-ro"
    )
    assert _job(cfg, attack, "a").split() == ["parent-busy", "skills-ro", "kept", "baselines-kept", "link-ro"]
    for name, target in ((".claude", ".."), ("model", "../judge"), ("DATA.md", "../judge/secret.txt")):
        p = work / name
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
        p.symlink_to(target)  # as if the agent had managed it: the host refuses each such mount point
    (work / ".agents").rmdir() if (work / ".agents").is_dir() else None
    (work / ".agents").write_text("a file where a directory goes")
    out = _job(
        cfg,
        "head -c 22 .claude/skills/quant-methods/SKILL.md; echo; head -c 22 .agents/skills/quant-methods/SKILL.md; "
        "echo; cat model/config.json; echo; head -1 DATA.md; grep -rl TOPSECRET . 2>/dev/null; "
        "cat ../judge/secret.txt 2>/dev/null || echo run-dir-hidden",
        "b",
    )
    assert "TOPSECRET" not in out, out
    lines = out.splitlines()
    assert not any("name: quant-method" in line for line in lines), lines  # no host copy at a link
    assert '{"tag": "orig"}' not in out and "# Data" not in out, lines
    assert lines[-1] == "run-dir-hidden", lines
    # the host neither removed nor replaced what the agent planted: the binds were dropped instead
    for name in (".claude", "model", "DATA.md"):
        assert (work / name).is_symlink(), name
    assert (work / ".agents").is_file() and not (work / ".agents").is_symlink()


#: A stand-in Claude Code for the wall test: in its turn, it reads and tries to change the wiki and calls the
#: commands, then records what happened.
FAKE_CLAUDE = r"""#!/usr/bin/python3
import errno, json, os, subprocess, sys
if "--version" in sys.argv:
    print("2.1.317 (Claude Code)")
    sys.exit(0)
out = {}
def t(name, fn):
    try:
        out[name] = fn()
    except OSError as e:
        out[name] = errno.errorcode.get(e.errno, str(e.errno))
skill = ".claude/skills/quant-methods"
t("read", lambda: open(f"{skill}/SKILL.md").read()[:30])
t("read codex", lambda: open(".agents/skills/quant-methods/references/capability.md").read()[:12])
t("append", lambda: open(f"{skill}/SKILL.md", "a").write("x"))
t("create", lambda: open(f"{skill}/references/new.md", "w").write("x"))
t("chmod", lambda: os.chmod(f"{skill}/SKILL.md", 0o644))
t("remove", lambda: os.remove(f"{skill}/SKILL.md"))
t("rename", lambda: os.rename(skill, "stolen"))
t("rename parent", lambda: os.rename(".claude/skills", ".claude/old"))
t("rename top", lambda: os.rename(".claude", "old"))
t("own file", lambda: open("notes/mine.md", "w").write("ok"))
t("baseline config", lambda: open("baselines/0/config.json").read())
t("baseline weights", lambda: open("baselines/0/model.safetensors", "rb").read())
for cmd in (["kl", "list"], ["handin", "status"], ["qda", "plan", "x.yaml"], ["qda", "probe"]):
    p = subprocess.run(["bin/" + cmd[0], *cmd[1:]], capture_output=True, text=True)
    out[" ".join(cmd)] = [p.returncode, (p.stdout + p.stderr).strip()[:120]]
if len(os.path.abspath(".qda/rpc.sock")) > 107:  # the host's private socket directory, bound read-only
    t("sock list", lambda: os.listdir(".qda"))
    t("sock remove", lambda: os.remove(".qda/rpc.sock"))
    t("sock replace", lambda: os.rename("notes/mine.md", ".qda/rpc.sock"))
    t("sock rename dir", lambda: os.rename(".qda", "old-qda"))
open("PROBELOG", "a").write(json.dumps(out) + "\n")
sys.exit(3)
"""


@pytest.mark.skipif(not isolation.shutil.which("bwrap"), reason="needs bwrap")
@pytest.mark.parametrize("name", ["cw", "c" * 120])
def test_the_wiki_and_the_commands_inside_the_real_wall(tmp_path, monkeypatch, name):
    """A real Humanize turn of a stand-in claude inside the bwrap wall: it reads the wiki, cannot change it,
    and reaches the host through the commands (the fixture's stand-ins behind them). It reads a baseline's
    config but not its weights (those are for its jobs only).

    With a run dir too long for a unix socket, the host listens in a private directory under
    `$XDG_RUNTIME_DIR`, bound read-only at `work/.qda`: the commands still reach it, the agent cannot remove,
    replace or move the socket, and the directory is gone after the run.
    """
    fence = pytest.importorskip("hmz.coganchor.fence")
    if not fence.enforceable(net=False):  # the wall holds the filesystem; the netns holds the network
        pytest.skip("this machine cannot hold the wall's filesystem fence")
    from qda_agent import backends

    run = ROOT / "runs" / name  # not under /tmp (hidden in the wall)
    if any(
        run.resolve().is_relative_to(Path(d).resolve()) for d in ("/tmp", "/var/tmp", tempfile.gettempdir())
    ):
        pytest.skip("the wall refuses a run dir under a temporary directory, and this checkout is one")
    runtime = Path(tempfile.mkdtemp(prefix="qda-test-", dir="/tmp"))  # short, whatever the checkout's path
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    _rm(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "claude").write_text(FAKE_CLAUDE)
    (bindir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    base = _ckpt(tmp_path / "b-by-name", "b0")  # the judge scans a baseline: a real (empty) safetensors file
    header = b'{"__metadata__": {"tag": "b0"}}'.ljust(32)
    (base / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw.update(
        run_dir=str(run),
        budget_gpu_hours=40 / 3600 * 2,
        agent=agent("claude/claude-haiku-5-5:low", resend_wait_s=1),
        sandbox="bwrap",
        isolation={"level": "strict"},  # the fixture names `none`; this test runs the real wall
        drain_timeout_s=5,
        baselines=[str(base)],
    )
    raw["toolset"]["qda"] = [sys.executable, "-c", "import sys; print('stand-in qda:', *sys.argv[1:])"]
    raw["judge"]["evaluator"] = str(EXAMPLES / "standin_kl")
    _abs_refs(raw)
    config = run.parent / "cw.yaml"
    config.write_text(yaml.safe_dump(raw))
    try:
        cfg = RunConfig.load(config)
        cfg.workdir.mkdir(parents=True)
        (cfg.workdir / ".claude").symlink_to(
            "../.."
        )  # planted: the host refuses the mount point (it removes nothing), so the turn starts without one
        (cfg.workdir / ".agents").write_text("a file where a directory goes")
        r = backends.make_run(cfg, config)
        with isolation.gpu_wall(cfg):
            r.run()
        probes = [json.loads(x) for x in (run / "work" / "PROBELOG").read_text().splitlines()]
        print(json.dumps(probes[0], indent=1))
        p = probes[0]
        assert p["read"] == "ENOENT" and p["read codex"] == "ENOTDIR", p  # no bind at a link or a file
        assert p["append"] == p["create"] == p["chmod"] == p["remove"] == "ENOENT", p
        assert p["rename"] == p["rename parent"] == "ENOENT", p
        assert p["rename top"] is None, p  # not a mount point: the agent's own link is its to move
        assert (cfg.workdir / "old").is_symlink()  # the host removed nothing; the turn moved its own link
        assert (cfg.workdir / ".agents").is_file() and not (cfg.workdir / ".agents").is_symlink()
        assert p["own file"] == 2, p  # the rest of the workdir stays the agent's
        assert p["kl list"][0] == 0 and "no readings yet" in p["kl list"][1], p
        assert p["handin status"][0] == 0, p
        assert p["qda plan x.yaml"] == [0, "stand-in qda: plan x.yaml"], p
        assert p["baseline config"] == '{"tag": "b0"}', p
        assert p["baseline weights"] in ("ENOENT", "EACCES"), p  # the shell sees the views copy only
        assert p["qda probe"][0] != 0 and "no `qda probe`" in p["qda probe"][1], p
        long = len(str(cfg.socket)) > 110
        # `cw` fits from a short checkout only; the long name never fits
        assert (long or name == "cw") and ("sock list" in p) == long, p
        if long:
            assert p["sock list"] == ["rpc.sock"], p
            assert p["sock remove"] == "EROFS" and p["sock replace"] in ("EROFS", "EXDEV"), p
            assert p["sock rename dir"] == "EBUSY", p
        assert list(runtime.iterdir()) == []  # the private directory is removed with the run
    finally:
        _rm(run)
        config.unlink(missing_ok=True)
        shutil.rmtree(runtime, ignore_errors=True)


@pytest.mark.skipif(not isolation.shutil.which("bwrap"), reason="needs bwrap")
def test_a_turn_the_wall_refuses_fails_the_run(tmp_path, monkeypatch):
    """A turn the isolation cannot allow (here: the private socket directory is writable inside) fails the run:
    Humanize retries the turn quietly, but the run ends at the first such turn, names it in END-REPORT and exits
    non-zero, with the submission still picked. Before, the turn was re-sent until the deadline and the run
    exited 0. Where the wall can check the socket dir it is a refusal; where the sandbox cannot be set up at all
    (a fresh /proc cannot be mounted on the host, say) Humanize fails the turn before the check, which is a
    fatal end — both are a run that did not complete.
    """
    fence = pytest.importorskip("hmz.coganchor.fence")
    if not fence.enforceable(net=False):  # the wall holds the filesystem; the netns holds the network
        pytest.skip("this machine cannot hold the wall's filesystem fence")
    from qda_agent.cli import execute
    from qda_agent.isolation import wall

    run = ROOT / "runs" / ("r" * 120)  # too long for a socket: the host uses a private directory
    runtime = Path(tempfile.mkdtemp(prefix="qda-test-", dir="/tmp"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(wall, "exposed", lambda argv, d: "/planted")  # as if a writable bind reached it
    _rm(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "claude").write_text(FAKE_CLAUDE)
    (bindir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    base = _ckpt(tmp_path / "b", "b0")
    header = b'{"__metadata__": {"tag": "b0"}}'.ljust(32)
    (base / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw.update(
        run_dir=str(run),
        budget_gpu_hours=60 / 3600 * 2,
        agent=agent("claude/claude-haiku-5-5:low", resend_wait_s=1),
        sandbox="bwrap",
        drain_timeout_s=5,
        baselines=[str(base)],
        isolation={"level": "strict"},  # the fixture names `none`; this test runs the real wall
    )
    raw["judge"]["evaluator"] = str(EXAMPLES / "standin_kl")
    _abs_refs(raw)
    config = run.parent / "refused.yaml"
    config.write_text(yaml.safe_dump(raw))
    try:
        assert execute(RunConfig.load(config)) == 2
        result = json.loads((run / "result.json").read_text())
        report = (run / "END-REPORT.md").read_text()
        if result["ended"].startswith("refused: the private socket directory "):
            assert "(through /planted)" in result["ended"], result
            # the first turn was refused, so no turn ran: the report's head says so and names the refusal
            assert (
                "**The agent never ran.** the agent never ran (WallRefused: the private socket directory"
                in report
            )
        else:
            assert result["ended"] == "fatal: HarnessSandboxed", result
            assert "**The agent never ran.** the agent never ran (" in report
        assert result["end_kind"] in ("refused", "fatal") and result["goals"] == 1, result
        assert result["rule"] == "handin" and (run / "FINAL.json").is_file()
        events = [json.loads(x)["ev"] for x in (run / "flow.jsonl").read_text().splitlines()]
        assert "fatal" in events and "failed" not in events
    finally:
        _rm(run)
        config.unlink(missing_ok=True)
        shutil.rmtree(runtime, ignore_errors=True)


def test_a_turn_that_ends_fatal_fails_the_run(tmp_path, capsys):
    """A failure no retry would change (here: the CLI cannot set up its sandbox) ends the agent phase at once
    and picks a checkpoint; the machine is what must change, so the status is 3 with a fix line."""
    from hmz.flows import HarnessSandboxed

    run = tmp_path / "run"
    script = lambda _: ScriptedWorker([HarnessSandboxed("no user namespaces here")])  # noqa: E731
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 30, script=script) == 3
    result = json.loads((run / "result.json").read_text())
    assert result["ended"] == "fatal: HarnessSandboxed" and result["end_kind"] == "fatal"
    assert result["rule"] == "handin" and (run / "FINAL.json").is_file()
    err = capsys.readouterr().err
    assert "this machine cannot hold the agent's sandbox" in err
    level = RunConfig.load(FIXTURES / "scripted-flow.yaml").isolation.level
    assert (
        f'fix:   rerun with --set run_dir=NEW_DIR --set "sandbox={{name: none}}" --set isolation.level={level} '
        in err
    )


def test_a_refused_sign_in_shows_no_word_of_the_agent_clis_stderr(tmp_path, capsys):
    """The agent CLI's refusal can echo the key it was given: stderr, result.json, FINAL.json and END-REPORT.md
    carry the error class and a fixed phrase, the run log keeps the CLI's words, and the status is 3 (no sign-in)."""
    from hmz.flows import HarnessRefused

    run = tmp_path / "run"
    said = "401 Unauthorized: invalid x-api-key sk-ant-TESTKEY-0000"
    script = lambda _: ScriptedWorker([HarnessRefused(said)])  # noqa: E731
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 30, script=script) == 3
    out, err = capsys.readouterr()
    shown = {name: (run / name).read_text() for name in ("result.json", "FINAL.json", "END-REPORT.md")}
    assert "TESTKEY" not in err + out and not [name for name, text in shown.items() if "TESTKEY" in text]
    assert json.loads(shown["result.json"])["fatal"] == "HarnessRefused"
    said = (
        "HarnessRefused: the agent CLI refused the run (sign-in or permission); "
        f"details in {run.resolve() / 'flow.jsonl'}"
    )
    assert (
        f"error: the run failed: {said}\n"
        "fix:   sign in to the agent CLI again, or check the agent CLI's permissions, then rerun with "
        "--set run_dir=NEW_DIR\n"
    ) in err
    assert said in json.loads(shown["FINAL.json"])["reason"] and said in shown["END-REPORT.md"]
    assert "TESTKEY" in (run / "flow.jsonl").read_text()


def test_an_api_that_keeps_failing_shows_no_word_of_the_agent_clis_stderr(tmp_path, capsys, monkeypatch):
    """An API that fails until the run finalizes early can echo the key in the CLI's error: stdout, stderr,
    result.json, FINAL.json and END-REPORT.md carry the error class and a fixed phrase naming flow.jsonl, and
    the run log keeps the CLI's words."""
    from hmz.flows import HarnessThrottled
    from qda_agent import flow

    monkeypatch.setattr(flow, "FINALIZE_AFTER_S", 0.2)
    run = tmp_path / "run"
    said = "429 Too Many Requests: rate limit for key sk-ant-TESTKEY-0002"
    script = lambda _: ScriptedWorker([HarnessThrottled(said)] * 200)  # noqa: E731
    code = scripted(
        FIXTURES / "scripted-flow.yaml", run, 30, script=script, overrides=["agent.resend_wait_s=0.05"]
    )
    out, err = capsys.readouterr()
    assert code == 3, err
    assert (
        "fix:   check the agent's sign-in and endpoint (`qda doctor --search CONFIG.yaml`), then rerun with "
        "--set run_dir=NEW_DIR\n"
    ) in err
    shown = {name: (run / name).read_text() for name in ("result.json", "FINAL.json", "END-REPORT.md")}
    assert "TESTKEY" not in err + out and not [name for name, text in shown.items() if "TESTKEY" in text]
    result = json.loads(shown["result.json"])
    assert result["end_kind"] == "finalize", result
    phrase = f"(last: HarnessThrottled; details in {run.resolve() / 'flow.jsonl'})"
    assert result["finalize_reason"].endswith(phrase)
    assert [w for w in json.loads(shown["FINAL.json"])["warnings"] if phrase in w]
    assert "TESTKEY" in (run / "flow.jsonl").read_text()


def test_a_refused_run_with_its_own_work_dir_asks_for_a_new_one_too(tmp_path, capsys):
    """A config that sets `work_dir` leaves it behind, and a rerun refuses an existing one: the fix names both."""
    from hmz.flows import HarnessRefused

    script = lambda _: ScriptedWorker([HarnessRefused("expired")])  # noqa: E731
    work = [f"work_dir={tmp_path / 'work'}"]
    assert scripted(FIXTURES / "scripted-flow.yaml", tmp_path / "run", 30, script=script, overrides=work) == 3
    assert "rerun with --set run_dir=NEW_DIR --set work_dir=NEW_WORK_DIR\n" in capsys.readouterr().err


@pytest.mark.parametrize("goal", [True, False])
def test_an_endpoint_that_refuses_the_goal_flow_says_to_turn_it_off(tmp_path, capsys, goal):
    """A Responses endpoint that keeps no state refuses the goal flow's follow-up, naming `call_id`: the run ends
    fatal, exits 3 and says to set `agent.goal: false`, in fixed words (the CLI's words and the key it echoed
    stay in flow.jsonl). The same words under the plain flow get the class's own fix."""
    from hmz.flows import HarnessUnrecoverable

    run = tmp_path / "run"
    said = (
        "400 invalid_request_error: function_call_output requires call_id or previous_response_id; "
        "ensure store=true (key sk-ant-TESTKEY-0001)"
    )
    script = lambda _: ScriptedWorker([HarnessUnrecoverable(said)])  # noqa: E731
    sets = [f"agent.goal={str(goal).lower()}"]
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 30, script=script, overrides=sets) == 3
    out, err = capsys.readouterr()
    shown = {name: (run / name).read_text() for name in ("result.json", "FINAL.json", "END-REPORT.md")}
    assert "TESTKEY" not in err + out and not [name for name, text in shown.items() if "TESTKEY" in text]
    assert "call_id" not in err and "TESTKEY" in (run / "flow.jsonl").read_text()
    if goal:
        assert json.loads(shown["result.json"])["fatal_cause"] == "goal_flow"
        assert (
            "HarnessUnrecoverable: the endpoint does not take codex's goal flow; details in"
            in shown["END-REPORT.md"]
        )
        assert (
            "fix:   the endpoint does not take codex's goal flow: rerun with --set run_dir=NEW_DIR "
            "--set agent.goal=false\n"
        ) in err
    else:
        assert "fatal_cause" not in json.loads(shown["result.json"])
        assert (
            "error: the run failed: HarnessUnrecoverable: the agent CLI failed in a way no resend changes"
            in err
        )
        assert "fix:   fix the cause the last `fatal` line of flow.jsonl names, then rerun with" in err


def test_a_spent_budget_ends_the_run_as_the_deadline_does(tmp_path):
    """The only budget a run has is the time to its GPU deadline, so a spent budget is a completed run: exit 0."""
    from hmz.flows import BudgetExceeded

    run = tmp_path / "run"
    script = lambda _: ScriptedWorker([BudgetExceeded("spent")])  # noqa: E731
    assert scripted(FIXTURES / "scripted-flow.yaml", run, 30, script=script) == 0
    result = json.loads((run / "result.json").read_text())
    assert result["ended"] == "deadline" and result["end_kind"] == "deadline"


def _rm(run: Path) -> None:
    """Remove a run dir, the judge's read-only evaluator copy included."""
    if run.exists():
        for p in [run, *run.rglob("*")]:
            if p.is_dir() and not p.is_symlink():
                p.chmod(0o755)
        shutil.rmtree(run)


# ------------------------------------------------------------------------------------------------- the inputs


def _ckpt(d: Path, tag: str) -> Path:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"tag": tag}))
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"weights-" + tag.encode())
    return d


def _inputs(tmp_path: Path, *, inputs: Path | None = None, **kw) -> RunConfig:
    """A run whose toolset has every input component, its model and one baseline real directories (`kw`: more
    settings; `inputs`: where those live, when the sandbox refuses the temporary dirs)."""
    base = tmp_path if inputs is None else inputs
    base.mkdir(parents=True, exist_ok=True)
    corpus = base / "corpora" / "c.jsonl"
    corpus.parent.mkdir()
    corpus.write_text('{"text": "hello"}\n')
    venv = Path(os.path.normpath(Path(sys.prefix).absolute()))  # stands for a vLLM venv
    comps = [
        *V01,
        {"name": "data", "sets": {"corpus": str(corpus)}},
        {"name": "serve", "python": str(venv / "bin" / "python")},
    ]
    kw.setdefault("run_dir", tmp_path / "run")
    kw.setdefault("judge", {"eval_max_tokens": 32768})
    return RunConfig(
        contamination="none",
        model=str(_ckpt(base / "orig", "orig")),
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        baselines=[_ckpt(base / "gptq-by-name", "b0")],
        toolset={"name": "qda", "components": [*comps, "baselines", "views", "source"]},
        **kw,
    )


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
@pytest.mark.parametrize("sandbox", ["bwrap", "none"])
def test_a_job_reaches_the_private_socket_of_a_long_run_dir(tmp_path, monkeypatch, sandbox):
    """With a run dir too long for a unix socket, an agent's job still reaches the host through `bin/`, in every
    sandbox mode that allows one (`fence` refuses it): at `work/.qda` (bwrap, bound read-only), or at the private
    directory's own path (none)."""
    from qda_agent import rpc

    runtime = Path(tempfile.mkdtemp(prefix="qda-test-", dir="/tmp"))  # short, whatever pytest's tmp path is
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    cfg = _inputs(tmp_path, run_dir=tmp_path / ("r" * 110) / "run", sandbox=sandbox)
    assert cfg.needs_private_socket
    try:
        with rpc.private_dir(cfg.run_dir) as d:
            _prepare(cfg)
            assert (cfg.workdir / "bin" / "rpc.path").read_text().split("\n")[0] == str(
                cfg.socket if sandbox == "bwrap" else d / "rpc.sock"
            )
            srv = rpc.Server(
                cfg.host_socket, lambda r: rpc.Reply(0, f"host: {r.verb} {r.argv}\n"), cfg.workdir
            )
            srv.start()
            try:
                out = _job(
                    cfg, "bin/kl list; rm -f .qda/rpc.sock " + str(d / "rpc.sock") + " 2>/dev/null; echo done"
                )
                assert (d / "rpc.sock").exists()  # the job could not remove it
            finally:
                srv.stop()
            assert out.split("\n")[:2] == ["host: kl ['list']", "done"], out
        assert not d.exists()
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def _job(cfg: RunConfig, script: str, name: str = "p") -> str:
    """Run `script` as an agent job of `cfg` in the real job sandbox; its output.

    The sandbox is asked for: `cfg` is at the default level, whose jobs run on the host (`RunConfig.job_sandbox`).
    """
    settings = {**cfg.sched_config(deadline=None), "budget_s": 600, "gpus": False, "job_sandbox": "bwrap"}
    sched = SchedConfig.model_validate(settings)
    s = Scheduler(cfg.run_dir, sched, LocalBackend(1), FakeClock())
    s.submit(
        ["/bin/sh", "-c", script],
        str(cfg.workdir),
        name=name,
        expected_s=60,
        root=cfg.workdir,
        origin="agent",
    )
    import time

    t = time.time() + 30
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    return s.log(name, 200)


def _prepare(cfg: RunConfig) -> str:
    from datetime import UTC, datetime

    from qda_agent.flow import prepare

    now = datetime.now(UTC)
    prepare(cfg, now, now)
    return (cfg.workdir / "TASK.md").read_text()


def test_input_components_write_their_files_and_task_lines(tmp_path):
    cfg = _inputs(tmp_path)
    task = _prepare(cfg)
    work = cfg.workdir
    over = (
        cfg.run_dir / "overlay"
    )  # the host's read-only copies (bound over the workdir in the wall and jobs)
    data = (over / "DATA.md").read_text()
    assert "| corpus | `" in data and "c.jsonl` | file |" in data
    assert "`baselines/0`" in task and "gptq-by-name" not in task  # the neutral name only
    assert "read-only inputs, not disk you can reclaim" in task  # the views are not the agent's disk
    assert "`vllm` is on their `PATH`" in task and "`DATA.md` lists" in task and "src/qda_quant" in task
    assert sorted(p.name for p in (over / "model").iterdir()) == [
        "config.json",
        "tokenizer.json",
    ]  # no weights
    assert sorted(p.name for p in (over / "baselines" / "0").iterdir()) == ["config.json", "tokenizer.json"]
    assert (work / "src" / "qda_quant" / "step.py").is_file() and os.access(
        work / "src" / "qda_quant" / "step.py", os.W_OK
    )
    sched, judge = cfg.sched_config(deadline=None), cfg.judge_config()
    assert str(tmp_path / "gptq-by-name") not in sched["readable"]  # bound at baselines/0 instead
    assert sched["binds"] == {
        **{str(work / n): str(over / n) for n in ("DATA.md", "baselines", "model")},
        str(work / "baselines" / "0"): str(tmp_path / "gptq-by-name"),  # after its parent's bind
    }
    assert sched["agent_job_env"]["PYTHONPATH"] == str(work / "src") == judge["agent_qda_env"]["PYTHONPATH"]
    assert "PYTHONPATH" not in judge["qda_env"]  # the judge's own qda (probes, smokes) never imports the copy
    assert "PYTHONPATH" not in judge.get("job_env", {})  # the judge's own jobs never import the agent's copy
    assert f"{Path(sys.prefix).absolute()}/bin" in sched["agent_job_env"]["PATH"].split(":")


def test_a_job_path_puts_the_launching_path_after_the_toolsets_dirs():
    """A build needs the user's compiler; a sandbox keeps only the system directories it mounts."""
    from qda_agent.toolsets import SYSTEM_PATH, Qda

    toolset = Qda(python=Path(sys.executable))
    inherited = toolset.job_env(base_path="/opt/site/gcc/bin:/usr/bin:/bin")["PATH"]
    assert inherited.startswith(f"{Path(sys.executable).parent}:")
    assert inherited.endswith(":/opt/site/gcc/bin:/usr/bin:/bin")
    assert toolset.job_env()["PATH"].endswith(f":{SYSTEM_PATH}")
    assert "/opt/site/gcc/bin" not in toolset.job_env()["PATH"]


def test_the_judge_never_sees_the_workdir(tmp_path):
    """The judge's jobs run in the judge's dir, with no path into the workdir in their environment.

    If this fails, a KL reading or a hand-in check could import the agent's `src/` copy (or any file it wrote).
    """
    cfg = _inputs(tmp_path)
    _prepare(cfg)
    work = str(cfg.workdir)
    from qda_agent.judge.config import JudgeConfig

    judge = JudgeConfig.model_validate({**cfg.judge_config(), "kl_ref": tmp_path / "ref"})  # as the host does
    for parent in (None, cfg.run_dir / "judge" / "evaluator"):
        env = judge.job_environ(parent)
        assert not [k for k, v in env.items() if work in v], env
    assert "PYTHONPATH" not in judge.job_env and not str(cfg.run_dir / "judge").startswith(work)


def test_views_copy_a_hub_snapshot_and_never_a_link_out_of_the_checkpoint(tmp_path, caplog):
    """A hub snapshot's links into `blobs/` are copied as files; a link out of the checkpoint (or to weights)
    is skipped with a warning, so the host leaks nothing the agent planted."""
    blobs = tmp_path / "hub" / "blobs"
    snap = tmp_path / "hub" / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (tmp_path / "outside.txt").write_text("SECRET")
    for name, text in (("config.json", '{"tag": "hub"}'), ("model.safetensors", "weights")):
        (blobs / name).write_text(text)
        (snap / name).symlink_to(f"../../blobs/{name}")
    (snap / "vocab.json").write_text("{}")
    (snap / "merges.txt").symlink_to("vocab.json")  # inside, not weights: copied
    (snap / "tokenizer.json").symlink_to("model.safetensors")  # inside, but weights: not
    (snap / "generation_config.json").symlink_to(tmp_path / "outside.txt")  # out of it: not
    cfg = _inputs(tmp_path).model_copy(update={"model": str(snap)})
    with caplog.at_level("WARNING"):
        _prepare(cfg)
    view = cfg.run_dir / "overlay" / "model"
    assert sorted(p.name for p in view.iterdir()) == ["config.json", "merges.txt", "vocab.json"]
    assert not (view / "config.json").is_symlink() and (view / "config.json").read_text() == '{"tag": "hub"}'
    assert "tokenizer.json" in caplog.text and "generation_config.json" in caplog.text


def test_source_install_refuses_a_link_in_its_place(tmp_path):
    """`Source.install` looks at the path itself (lstat): a link there is refused, never followed."""
    from qda_agent.components import Source

    cfg = _inputs(tmp_path)
    _prepare(cfg)
    dst = cfg.workdir / "src" / "qda_quant"
    (dst / "step.py").write_text("# patched\n")
    run = type("R", (), {"workdir": cfg.workdir})()
    Source().install(run)  # a real copy is kept as it is
    assert (dst / "step.py").read_text() == "# patched\n"
    shutil.rmtree(dst)
    (tmp_path / "elsewhere").mkdir()
    dst.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="is a link or a file"):
        Source().install(run)
    assert not list((tmp_path / "elsewhere").iterdir())


def test_no_input_components_no_words(tmp_path):
    cfg = _inputs(tmp_path).model_copy(update={"toolset": Qda(components=V01)})
    task = _prepare(cfg)
    for word in ("DATA.md", "vllm", "baselines/0", "src/qda_quant", "`model/`"):
        assert word not in task, word
    assert str(tmp_path / "gptq-by-name") in task  # named by its path, as in v0.1
    assert not (cfg.workdir / "src").exists() and not (cfg.workdir / "model").exists()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
def test_input_components_inside_a_job(tmp_path):
    """In a real job sandbox: the dataset, `baselines/0` (whole, read-only), the views and the source copy."""
    cfg = _inputs(tmp_path)
    _prepare(cfg)
    script = (
        f"cat {tmp_path}/corpora/c.jsonl; cat baselines/0/model.safetensors; echo; "
        "(echo x > baselines/0/new) 2>/dev/null || echo bl-refused; "
        "(echo x >> model/config.json) 2>/dev/null || echo view-refused; "
        "(echo x >> DATA.md) 2>/dev/null || echo data-refused; "
        "echo '# patched' >> src/qda_quant/step.py && echo src-ok; "
        f"ls {tmp_path}/gptq-by-name 2>/dev/null || echo host-path-hidden; "
        'echo "PP=$PYTHONPATH"'
    )
    out = _job(cfg, script).split()
    assert out[:2] == ['{"text":', '"hello"}'] and out[2] == "weights-b0", out
    assert out[3:8] == ["bl-refused", "view-refused", "data-refused", "src-ok", "host-path-hidden"], out
    assert out[8] == f"PP={cfg.workdir / 'src'}", out


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")
def test_a_sandboxed_build_job_reads_the_runs_calibration_file(tmp_path):
    """At isolation.level sandbox the agent's builds run in bwrap: the calibration file the run names (prepare's
    `data/agent-calib.jsonl`, beside no other bound path) is readable there, read-only, with no
    `scheduler.readable` entry for it."""
    calib = tmp_path / "prep" / "data" / "agent-calib.jsonl"
    calib.parent.mkdir(parents=True)
    calib.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    cfg = _inputs(tmp_path, calib=str(calib), isolation={"level": "sandbox"})
    assert cfg.job_sandbox() == "bwrap" and str(calib) not in [str(p) for p in cfg.scheduler.readable]
    assert str(calib) in [str(p) for p in cfg.read_paths()[0]]
    _prepare(cfg)
    out = _job(cfg, f"head -c 13 {calib}; echo; (echo x >> {calib}) 2>/dev/null || echo calib-ro").split()
    assert out == ['{"messages":', "calib-ro"], out


def test_bound_baselines_are_still_scanned(tmp_path):
    """The pre-start contamination scan reads a baseline bound at `baselines/<N>` too."""
    from qda_agent.flow import readable_inputs

    cfg = _inputs(tmp_path)
    files = readable_inputs(cfg)
    assert tmp_path / "gptq-by-name" / "config.json" in files


@pytest.mark.parametrize(
    "where",
    ["baselines", "model", "data", "serve", "readable", "link"],
)
def test_a_read_only_source_inside_the_run_dir_is_refused(tmp_path, where):
    """Every read-only input of the agent's jobs must be the host's: none may resolve into run_dir."""
    run = tmp_path / "run"
    inside = _ckpt(run / "judge" / "planted", "x")
    base = _base(run)
    if where == "link":  # a path outside the run dir that is a link into it
        (tmp_path / "out-link").symlink_to(inside)
        inside = tmp_path / "out-link"
    if where == "serve":  # the interpreter is there, so its venv is bound and must be checked
        (inside / "bin").mkdir()
        (inside / "bin" / "python").write_text("")
    raw = {
        "baselines": {**base, "baselines": [str(inside)]},
        "link": {**base, "baselines": [str(inside)]},
        "model": {**base, "model": str(inside)},
        "data": {
            **base,
            "toolset": {"name": "qda", "components": [{"name": "data", "sets": {"c": str(inside)}}]},
        },
        "serve": {
            **base,
            "toolset": {
                "name": "qda",
                "components": [{"name": "serve", "python": str(inside / "bin" / "python")}],
            },
        },
        "readable": {**base, "scheduler": {"name": "qsched", "readable": [str(inside)]}},
    }[where]
    with pytest.raises(ValueError, match="inside run_dir"):
        RunConfig.model_validate(raw)
    ok = {**base, "baselines": [str(_ckpt(tmp_path / "outside", "o"))]}
    RunConfig.model_validate(ok)


class _Binds(Component):
    """A component whose `binds()` names several host paths (built in the test, not registered)."""

    name: Literal["binds"] = "binds"
    paths: list[Path] = Field(default_factory=list)

    def protected(self, run: Run) -> list[str]:  # noqa: ARG002
        return ["bound"]

    def binds(self, run: Run) -> dict[Path, Path]:
        return {run.workdir / "bound" / f"bind-{i}": p for i, p in enumerate(self.paths)}


@pytest.mark.parametrize("order", list(permutations(["inside", "a", "b", "c"])))
def test_every_dataset_of_a_component_is_checked(tmp_path, order):
    """Four datasets of one `data` component: the one inside the run dir is refused in every position."""
    run = tmp_path / "run"
    paths = {
        "inside": _ckpt(run / "work" / "planted", "x"),
        "a": _ckpt(tmp_path / "outside" / "a", "a"),
        "b": _ckpt(tmp_path / "outside" / "b", "b"),
        "c": _ckpt(tmp_path / "outside" / "c", "c"),
    }
    raw = _base(
        run,
        toolset={"name": "qda", "components": [{"name": "data", "sets": {k: str(paths[k]) for k in order}}]},
    )
    with pytest.raises(ValueError, match="inside run_dir") as e:
        RunConfig.model_validate(raw)
    assert "toolset (data)" in str(e.value) and str(paths["inside"]) in str(e.value)


@pytest.mark.parametrize("order", list(permutations(["inside", "a", "b"])))
def test_every_bind_of_a_component_is_checked(tmp_path, order):
    """A component's several binds: the one inside the run dir is refused in every position."""
    run = tmp_path / "run"
    paths = {
        "inside": _ckpt(run / "judge" / "planted", "x"),
        "a": _ckpt(tmp_path / "outside" / "a", "a"),
        "b": _ckpt(tmp_path / "outside" / "b", "b"),
    }
    cfg = RunConfig.model_validate(_base(run, toolset={"name": "qda", "components": ["qda"]}))
    cfg.toolset.components.append(_Binds(paths=[paths[k] for k in order]))
    with pytest.raises(ValueError, match="inside run_dir") as e:
        check(cfg)
    assert "toolset (binds)" in str(e.value) and str(paths["inside"]) in str(e.value)


@pytest.mark.parametrize("order", list(permutations(["holder", "a", "b"])))
def test_a_holder_in_a_multi_path_component_is_refused(tmp_path, order):
    """A source that holds the run dir is refused wherever it sits in a component's several paths."""
    run = tmp_path / "run"
    paths = {
        "holder": tmp_path,
        "a": _ckpt(tmp_path / "outside" / "a", "a"),
        "b": _ckpt(tmp_path / "outside" / "b", "b"),
    }
    raw = _base(
        run,
        toolset={"name": "qda", "components": [{"name": "data", "sets": {k: str(paths[k]) for k in order}}]},
    )
    with pytest.raises(ValueError, match="holds run_dir") as e:
        RunConfig.model_validate(raw)
    assert "toolset (data)" in str(e.value) and str(paths["holder"]) in str(e.value)


class _ReadsMore(Toolset):
    """A toolset whose `readable()` adds a path no component declares (built in the test, not registered)."""

    name: Literal["reads-more"] = "reads-more"  # type: ignore[assignment]
    extra: Path | None = None

    def readable(self) -> list[Path]:
        return [*super().readable(), *([self.extra] if self.extra is not None else [])]


def test_a_toolset_that_adds_a_readable_path_is_checked(tmp_path):
    """The check validates what `read_paths` binds, not only what the components declare."""
    run = tmp_path / "run"
    inside = _ckpt(run / "judge" / "planted", "x")
    cfg = RunConfig.model_validate(_base(run, toolset={"name": "qda", "components": ["qda"]}))
    cfg.toolset = _ReadsMore(extra=inside)
    with pytest.raises(ValueError, match="inside run_dir"):
        check(cfg)


@pytest.mark.parametrize("where", ["toolset", "judge"])
def test_a_set_but_absent_serving_interpreter_is_refused(tmp_path, where):
    """An interpreter that is set and not there is refused: its venv dir would hand the jobs what holds it."""
    root = tmp_path / "proj"
    root.mkdir()
    python = str(root / "bin" / "python")
    raw = _base(
        tmp_path / "run",
        toolset={"name": "qda", "components": [{"name": "serve", "python": python}]}
        if where == "toolset"
        else {"name": "qda", "components": ["serve"]},
        judge={"serve_python": python} if where == "judge" else {},
    )
    label = r"toolset \(serve\)" if where == "toolset" else "judge.serve_python"
    with pytest.raises(ValueError, match=rf"{label} .*does not exist"):
        RunConfig.model_validate(raw)


def test_only_a_placeholder_hides_a_path_from_the_run_dir_check(tmp_path):
    """A path with a stray `<` is checked; only a full `<...>` placeholder waits for the loader's error."""
    run = tmp_path / "run"

    def cfg(path: Path | str) -> dict:
        return _base(run, toolset={"name": "qda", "components": [{"name": "data", "sets": {"c": str(path)}}]})

    with pytest.raises(ValueError, match="inside run_dir"):
        RunConfig.model_validate(cfg(_ckpt(run / "judge" / "a<b", "x")))
    RunConfig.model_validate(cfg(_ckpt(tmp_path / "out<side", "o")))
    RunConfig.model_validate(cfg("<path/to/data>"))


def test_a_relative_model_path_inside_the_run_dir_is_refused(tmp_path, monkeypatch):
    """A relative model path that does not exist yet is checked; only `./` used to mark it as a path."""
    monkeypatch.chdir(tmp_path)
    raw = _base(tmp_path / "run", model="run/work/planted")
    with pytest.raises(ValueError, match="model .*inside run_dir"):
        RunConfig.model_validate(raw)
    RunConfig.model_validate({**raw, "model": "elsewhere/model"})  # relative, outside: a hub id or a path
    RunConfig.model_validate({**raw, "model": "<model>"})  # a placeholder waits for the loader


@pytest.mark.parametrize(
    "where", ["kl_ref", "anchor", "evaluator", "format_reference", "gencheck_prompts", "guard"]
)
def test_a_judge_input_inside_the_run_dir_is_refused(tmp_path, where):
    """The judge runs or reads these; a job could write them inside the run dir."""
    run = tmp_path / "run"
    inside = run / "judge" / "trusted"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}")
    raw = _base(run)
    if where == "guard":
        raw["contamination"] = {"name": "ngram", "guard": str(inside)}
    else:
        raw["judge"] = {where: str(inside)}
    with pytest.raises(ValueError, match="inside run_dir") as e:
        RunConfig.model_validate(raw)
    assert (where if where != "guard" else "contamination.guard") in str(e.value)


def test_a_python_root_that_holds_the_run_dir_is_refused(tmp_path, monkeypatch):
    """Every job binds the host's Python roots; one that holds the run dir would expose the judge's files."""
    from qda_agent import run_sources

    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(run_sources, "python_roots", lambda: [root])
    raw = _base(root / "runs" / "x")
    with pytest.raises(ValueError, match="holds run_dir") as e:
        RunConfig.model_validate(raw)
    assert "python root" in str(e.value)


def test_task_md_words_the_wiki_per_sandbox(tmp_path):
    from qda_agent.components import Run, Wiki

    walled = Wiki().section(Run(workdir=tmp_path, walled=True))
    copy = Wiki().section(Run(workdir=tmp_path, walled=False))
    assert "(read-only;" in walled and "(your own copy;" in copy and "read-only" not in copy


def test_a_toolset_component_setting_manual_is_refused():
    """The experience manual is gone; `wiki.manual` is refused with the reason, not silently ignored."""
    with pytest.raises(ValidationError, match="`manual` was removed"):
        Toolset.model_validate({"name": "qda", "components": [{"name": "wiki", "manual": "qwen3-dense"}]})


def test_a_skipped_bind_is_reported(tmp_path, monkeypatch):
    """A mount point the host cannot repair is skipped (the sandbox still starts) and reported to `on_skip`."""
    from qda_agent.sched import sandbox

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(sandbox, "on_skip", lambda p, why: seen.append((p, why)))
    work, src = tmp_path / "work", tmp_path / "ov" / "x"
    work.mkdir()
    src.mkdir(parents=True)
    out = sandbox.mount_points({work / "a" / "deep": src}, [work])
    assert out == {} and seen and seen[0][0] == str(work / "a" / "deep")


def test_the_component_is_in_the_default_preset_after_kl(tmp_path):
    assert COMPONENTS["gencheck"] is Gencheck
    names = [c.name for c in Qda().components]
    assert names.index("gencheck") == names.index("kl") + 1
    assert "gencheck" in Qda().commands(Run(workdir=tmp_path))
    line = Gencheck().section(Run(workdir=tmp_path))
    assert line.startswith("- `gencheck DIR [--max-tokens N] [--per-domain K]`") and line.endswith(";\n")


def test_serve_defaults_to_the_judges_vllm(tmp_path):
    """`serve` without `python` gets `judge.serve_python`, so the shipped config gives the agent vLLM."""
    from qda_agent.doctor import check_components

    path = filled(CONFIGS / "search.yaml", tmp_path)
    cfg = RunConfig.load(path)
    (serve,) = [c for c in cfg.toolset.components if c.name == "serve"]
    assert serve.python == cfg.judge.serve_python == tmp_path / "path/to/vllm-venv/bin/python"
    assert "serve" not in [c.name for c in check_components(list(cfg.toolset.components))]
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg.resolved()))
    assert [c.python for c in RunConfig.load(resolved).toolset.components if c.name == "serve"] == [
        serve.python
    ]
    own = tmp_path / "own-venv"
    (own / "bin").mkdir(parents=True)
    (own / "bin" / "python").touch()
    (own / "pyvenv.cfg").write_text("home = /usr/bin\n")
    setting = f"toolset.components=[{{name: serve, python: {own / 'bin' / 'python'}}}]"
    assert RunConfig.load(path, [setting]).toolset.components[0].python == own / "bin" / "python"
    none = RunConfig.load(path, ["toolset.components=[serve]", "judge.serve_python=null"])
    assert none.toolset.components[0].python is None


def test_probe_reads_the_agents_own_data_never_the_judges(tmp_path):
    """The probe's TASK.md item names the agent's own data and says it is not `kl`; vLLM rows only."""
    toolset = Qda()
    vllm, gguf = Run(workdir=tmp_path, engine="vllm"), Run(workdir=tmp_path, engine="llama.cpp")
    (item,) = [s for s in toolset.sections(vllm) if "qda_agent.probe" in s]
    assert "--data FILE.jsonl" in item and "--ref" not in item and "not `kl`" in item
    assert not any("qda_agent.probe" in s for s in toolset.sections(gguf))


def test_the_serve_item_names_no_host_path(tmp_path):
    """P3: the serving interpreter reaches the agent through its job environment, not through TASK.md."""
    interpreter = tmp_path / "vllm-venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    toolset = Qda.model_validate({"components": [{"name": "serve", "python": str(interpreter)}]})
    run = Run(workdir=tmp_path)
    (item,) = [s for s in toolset.sections(run) if "vllm" in s]
    assert str(tmp_path) not in " ".join(item.split()) and "$QDA_VLLM_PYTHON" in item
    assert toolset.job_env(run)["QDA_VLLM_PYTHON"] == str(interpreter)
