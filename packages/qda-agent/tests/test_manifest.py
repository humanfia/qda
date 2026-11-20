"""The workspace manifest: one record of what the run gives the agent, and TASK.md read from it."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from qda_agent import workspace
from qda_agent.components import COMPONENTS, Component
from qda_agent.config import RunConfig
from qda_agent.flow import prepare
from qda_agent.prompt import NUMBERS
from testkit import scripted

FIXTURES = Path(__file__).parent / "fixtures"
V01 = ["qda", "kl", "qsched", "handin"]
INPUTS_ONLY = ["wiki", "baselines", "views", "source"]


def _config(tmp_path: Path, components: list | None = None, **over) -> RunConfig:
    """A minimal run: no commands installed, no sandbox, one real baseline directory."""
    baseline = tmp_path / "base"
    baseline.mkdir(exist_ok=True)
    (baseline / "config.json").write_text("{}")
    raw = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "contamination": "none",
        "sandbox": "none",
        "commands": None,
        "baselines": [str(baseline)],
        "judge": {"eval_max_tokens": 32768},
        **over,
    }
    if components is not None:
        raw["toolset"] = {"name": "qda", "components": components}
    return RunConfig.model_validate(raw)


def _prepared(
    tmp_path: Path, components: list | None = None, **over
) -> tuple[RunConfig, dict, workspace.Manifest]:
    cfg = _config(tmp_path, components, **over)
    now = datetime.now(UTC)
    manifest = prepare(cfg, now, now)
    return cfg, json.loads((cfg.run_dir / workspace.MANIFEST).read_text()), manifest


def test_every_workspace_file_and_bind_is_in_the_manifest_with_its_sha(tmp_path):
    """The manifest names every file the agent can read and every bind, each with a digest."""
    cfg, record, manifest = _prepared(tmp_path)
    path = cfg.run_dir / workspace.MANIFEST
    assert path.is_file() and not path.is_relative_to(cfg.workdir)  # beside the workdir, never inside it
    assert record["version"] == workspace.VERSION

    shown = [cfg.workdir, *([cfg.run.overlay] if cfg.run.overlay is not None else [])]
    expected = {str(p) for root in shown for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()}
    assert set(record["files"]) == expected
    for name, sha in record["files"].items():
        assert sha == hashlib.sha256(Path(name).read_bytes()).hexdigest(), name
    assert cfg.manifest().files == manifest.files  # the wall and the scheduler read the same record

    for rel in ("work/TASK.md", "work/CLOCK.md"):
        assert record["texts"][rel] == record["files"][str(cfg.run_dir / rel)]

    mounts = cfg.toolset.mounts(cfg.run, jobs=False)
    assert set(record["mounts"]["agent"]) == {str(d) for d, s in mounts.items() if s.exists()}
    assert set(record["mounts"]["jobs"]) >= set(record["mounts"]["agent"])
    assert str(cfg.workdir / "baselines" / "0") in record["mounts"]["jobs"]
    for which in ("agent", "jobs"):
        for entry in record["mounts"][which].values():
            assert entry["sha256"] and entry["kind"] in ("file", "dir"), entry
            assert entry["digest"] in ("sha256", "size+mtime"), entry
            assert Path(entry["source"]).exists()


def test_the_tools_block_names_only_the_commands_the_manifest_has(tmp_path):
    """Every command the Tools block names is in the manifest, and every manifest command is named there.

    The statement's fixed sections (the approved wording of the KL, Long generations and Prior paragraphs)
    name `kl` and `handin` on purpose; the part rendered from the toolset is the Tools list, so a run
    without `gencheck` must not read about it, nor a run with no commands name any.
    """
    universe, verbs = _universe()
    cases = [
        (None, {}),
        (V01, {}),
        (["qda", "kl"], {}),
        (["gencheck"], {}),  # a tool whose item explains what it is not (`kl`): the clause must go with it
        (["handin"], {}),
        (["qda"], {}),
        (["fisher"], {}),  # a verb of `qda`, and its map runs as a queued job: neither may be named here
        (["probe"], {}),  # runs as `python -m`, but its item explains itself against `kl` and `qsched`
        (INPUTS_ONLY, {}),
    ]
    for n, (components, over) in enumerate(cases):
        case = tmp_path / str(n)
        case.mkdir()
        cfg, record, _ = _prepared(case, components, **over)
        full = (cfg.workdir / "TASK.md").read_text()
        task = _tools_block(full)
        named = _command_spans(task, universe, verbs)
        assert all(cmd in record["commands"] for _, cmd, _ in named), (components, over, named)
        for cmd, verb in ((c, v) for _, c, v in named if v is not None):
            if record["subcommands"].get(cmd):  # a command that declares its verbs (qda) accepts only these
                assert verb in record["subcommands"][cmd], (components, over, cmd, verb)
        for cmd in record["commands"]:
            assert re.search(rf"`{re.escape(cmd)}(?:[ `])", full), (components, cmd)
        assert _stated_count(task) == _expected_count(len(record["commands"]))


def test_the_command_count_is_a_word_for_every_preset_size(tmp_path):
    """Six commands read "Six", not "6": the statement's count agrees with the manifest at every size."""
    cfg, record, _ = _prepared(tmp_path)  # the `qda` preset: six commands
    assert len(record["commands"]) == 6
    task = _tools_block((cfg.workdir / "TASK.md").read_text())
    assert "Six commands in `bin/`" in task
    assert "6 commands in `bin/`" not in task


def test_candidate_zero_rewrites_task_md_and_the_new_manifest_is_scanned_again(tmp_path):
    """A second `prepare` after candidate 0's reading replaces the manifest and is logged as its own scan."""
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    raw = {
        "model": "tiny/model",
        "format": "w4a16",
        "budget_gpu_hours": 0.5 / 3600,
        "contamination": "none",
        "cards": [0],
        "run_dir": "run",
        "sandbox": "none",
        "host": "testkit:selecting_host",
        "baselines": ["base/gptq"],
        "judge": {"select_candidate0": True, "eval_max_tokens": 32768},
        "drain_timeout_s": 5,
        "commands": None,
        "agent": {"resend_wait_s": 0.02},
    }
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(raw))
    run = tmp_path / "run"
    assert scripted(config, run, 0.5) == 0

    events = [json.loads(line) for line in (run / "flow.jsonl").read_text().splitlines()]
    scans = [e for e in events if e["ev"] == "contamination"]
    assert [e.get("after") for e in scans] == [None, "candidate0"], scans
    first, second = scans
    assert first["manifest"] == second["manifest"] == str(run / workspace.MANIFEST)
    assert first["manifest_sha256"] != second["manifest_sha256"]  # TASK.md (and so the record) changed
    assert second["manifest_sha256"] == workspace.file_sha256(run / workspace.MANIFEST)


def _tools_block(task: str) -> str:
    """The part of TASK.md rendered from the toolset (between the Tools heading and the Prior heading)."""
    return task.split("## Tools", 1)[1].split("## Prior", 1)[0]


def _universe() -> tuple[set[str], dict[str, set[str]]]:
    """Every command and verb any component can add (so a hard-coded off-toolset name is detectable)."""
    commands: set[str] = set()
    verbs: dict[str, set[str]] = {}
    for _, plugin in COMPONENTS.items():
        component = cast(Component, plugin)
        commands |= set(component.commands)
        for command, its in component.subcommands.items():
            verbs.setdefault(command, set()).update(its)
    return commands, verbs


def _command_spans(
    text: str, commands: set[str], verbs: dict[str, set[str]]
) -> list[tuple[str, str, str | None]]:
    """The code spans of `text` that start with a known command: (span, command, verb or None)."""
    out = []
    for span in re.findall(r"`([^`\n]+)`", text):
        words = span.split()
        if not words or words[0] not in commands:
            continue
        verb = words[1] if len(words) > 1 and words[1] in verbs.get(words[0], set()) else None
        out.append((span, words[0], verb))
    return out


def _stated_count(text: str) -> int | str | None:
    """The number of commands TASK.md states (`Four commands in bin/`), as its word or None."""
    found = re.search(r"(\w+) commands? in `bin/`", text)
    if found is None:
        return None
    word = found.group(1)
    return word


def _expected_count(n: int) -> str | None:
    return None if n == 0 else NUMBERS.get(n, str(n))


@pytest.mark.parametrize("components", [None, V01, ["qda", "kl"], INPUTS_ONLY])
def test_the_manifest_shortens_to_the_toolset(tmp_path, components):
    """Each toolset's manifest describes exactly its own components, never the preset's."""
    cfg, record, _ = _prepared(tmp_path, components)
    keys = [c.name for c in cfg.toolset.components]
    expected = set()
    for key in keys:
        expected |= set(cast(Component, COMPONENTS[key]).commands)
    assert set(record["commands"]) <= expected
    assert record["workdir"] == str(cfg.workdir) and record["run_dir"] == str(cfg.run_dir)


def test_the_generated_data_md_is_recorded_as_a_text(tmp_path):
    """`DATA.md` is written by the data component: the manifest records it among the generated texts."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "x"}\n')
    cfg, record, _ = _prepared(tmp_path, ["wiki", {"name": "data", "sets": {"corpus": str(corpus)}}])
    overlay = cfg.run.overlay
    assert overlay is not None
    data = overlay / "DATA.md"
    assert data.is_file()
    assert record["texts"]["overlay/DATA.md"] == record["files"][str(data)]


def test_the_full_card_moe_paragraph_names_only_the_tools_the_run_gives():
    """The paragraph moved from the flow into the template: behind its flag, and filtered by the toolset."""
    from qda_agent.prompt import _full_card_moe_section

    assert _full_card_moe_section(False, {"qda", "kl"}, {"build"}) == ""
    text = _full_card_moe_section(True, {"qda", "kl"}, {"build"})
    assert "`qda build`" in text and "`kl`" in text
    assert "The host refuses overlapping candidates." in text
    text = _full_card_moe_section(True, {"qda"}, {"build"})
    assert "`kl`" not in text
    text = _full_card_moe_section(True, set(), set())
    assert "`qda" not in text and "`kl" not in text and "The host refuses" in text


def test_a_dir_binds_digest_tracks_content_not_only_shape(tmp_path):
    """Replacing a small bound file's bytes changes the bind's digest (a shape hash would miss it)."""
    cfg = _config(tmp_path)
    first = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    source = cfg.baselines[0] / "config.json"
    source.write_text("[]")
    second = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    dest = str(cfg.workdir / "baselines" / "0")
    assert first.mounts["jobs"][dest].sha256 != second.mounts["jobs"][dest].sha256
    assert first.mounts["jobs"][dest].digest == second.mounts["jobs"][dest].digest == "sha256"


def test_a_large_file_is_identified_by_size_and_mtime_not_read(tmp_path, monkeypatch):
    """Above the content limit a bind records size+mtime, and says so in the entry."""
    monkeypatch.setattr(workspace, "CONTENT_MAX", 4)
    big = tmp_path / "big.gguf"
    big.write_bytes(b"x" * 64)
    digest, method = workspace.content_digest(big)
    assert method == "size+mtime" and digest.startswith("64:")


def test_the_private_socket_bind_is_in_the_agent_mounts(tmp_path, monkeypatch):
    """The wall mounts the private socket directory over `work/.qda` for the shell too: the manifest lists it."""
    from qda_agent import rpc
    from qda_agent.sandbox_settings import Bwrap

    cfg = _config(tmp_path)
    cfg.sandbox = Bwrap()  # walled: the private directory is bound instead of read at its own path
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with rpc.private_dir(cfg.run_dir):
        now = datetime.now(UTC)
        prepare(cfg, now, now)
        expected = cfg.socket_binds()
        assert expected, "a held private directory is a bind"
        assert expected.items() <= cfg.manifest().agent_binds().items()
        assert expected.items() <= cfg.manifest().job_binds().items()


def test_the_candidate_zero_request_keeps_a_hub_model_id(tmp_path, monkeypatch):
    """A judge model named by a Hub id (`org/name[@rev]`) reaches the smoke request unchanged, not as a path."""
    import threading
    from types import SimpleNamespace

    from qda_agent.judge import build_receipt, startup

    work, judge = tmp_path / "work", tmp_path / "judge"
    work.mkdir()
    judge.mkdir()
    host = SimpleNamespace(
        workdir=work,
        jdir=judge,
        lock=threading.Lock(),
        build_reserved={},
        cfg=SimpleNamespace(model="org/model", format="w4a16", format_map=None),
    )
    seen = {}

    def capture(host, cwd, args, **kwargs):
        seen.update(yaml.safe_load((cwd / args[0]).read_text()))
        raise ValueError("captured")

    monkeypatch.setattr(build_receipt, "setup", capture)
    with pytest.raises(ValueError, match="captured"):
        startup._candidate0(cast(Any, host))
    assert seen["model"] == "org/model"


def test_a_symlinked_bound_file_records_its_target_and_the_bytes_behind_it(tmp_path):
    """A Hub-style snapshot link (`config.json -> ../blobs/<h>`) is digested by target and content."""
    cfg = _config(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "h"
    blob.write_text("{}")
    link = cfg.baselines[0] / "config.json"
    link.unlink()  # a regular file made by the fixture; the snapshot holds a link in its place
    link.symlink_to("../blobs/h")
    first = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    blob.write_text("[]")
    second = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    dest = str(cfg.workdir / "baselines" / "0")
    assert first.mounts["jobs"][dest].files == 1  # the link is a bound file, not an omission
    assert first.mounts["jobs"][dest].sha256 != second.mounts["jobs"][dest].sha256


def test_retargeting_an_in_tree_link_changes_the_digest(tmp_path):
    """The link's target string is part of the digest, so pointing it at another in-tree file changes it."""
    cfg = _config(tmp_path)
    baseline = cfg.baselines[0]
    (baseline / "first.json").write_text("{}")
    (baseline / "second.json").write_text("[]")
    link = baseline / "config.json"
    link.unlink()
    link.symlink_to("first.json")
    first = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    link.unlink()
    link.symlink_to("second.json")
    second = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    dest = str(cfg.workdir / "baselines" / "0")
    assert first.mounts["jobs"][dest].sha256 != second.mounts["jobs"][dest].sha256


def test_a_directory_link_is_walked_only_inside_the_bound_root(tmp_path):
    """A link to a directory in the bind is walked; one that leads outside records only its target."""
    cfg = _config(tmp_path)
    baseline = cfg.baselines[0]
    (baseline / "inside").mkdir()
    (baseline / "inside" / "f.json").write_text("{}")
    (baseline / "alias").symlink_to("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.json").write_text("{}")
    (baseline / "out").symlink_to("../outside")
    first = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    (outside / "f.json").write_text("[]")  # behind an outside link: not read, not recorded
    second = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    dest = str(cfg.workdir / "baselines" / "0")
    assert first.mounts["jobs"][dest].sha256 == second.mounts["jobs"][dest].sha256
    (baseline / "inside" / "f.json").write_text("[]")  # inside the bind: recorded
    third = workspace.build_manifest(cfg, cfg.toolset, snapshot=True)
    assert third.mounts["jobs"][dest].sha256 != second.mounts["jobs"][dest].sha256


def test_the_manifest_lists_every_pre_built_start_with_its_path_and_origin(tmp_path):
    """`judge.starts` reaches the manifest as {name, path, origin} in menu order; TASK.md shows no path."""
    a, b = tmp_path / "starts" / "a", tmp_path / "starts" / "b"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    cfg, record, _ = _prepared(
        tmp_path,
        baselines=[str(a), str(b)],
        judge={
            "eval_max_tokens": 32768,
            "starts": [
                {"name": "gptq", "path": str(a), "origin": "own"},
                {"name": "awq", "path": str(b), "origin": "own"},
            ],
        },
    )
    assert record["starts"] == [
        {"name": "gptq", "path": str(a), "origin": "own"},
        {"name": "awq", "path": str(b), "origin": "own"},
    ]
    text = (cfg.workdir / "TASK.md").read_text()
    assert "gptq" in text and "awq" in text
    assert str(a) not in text and str(b) not in text  # the paths stay in the manifest
