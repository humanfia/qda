"""A dataset named in the run config keeps the judge's hidden-data guarantees, wherever its plugin keeps the data.

`Stored`'s settings never name its path, so a check that read the settings instead of `local()` would pass it.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from qda_agent import isolation
from qda_agent.config import RunConfig
from qda_agent.judge import JudgeConfig
from qda_kl.datasets import DATASETS, Dataset
from test_gencheck import GenRig
from test_judge import ckpt

CMD = ["{python}", "{tools}/suite_job.py", "{suite}", "{ckpt}", "{ref}", "{out}"]


class Stored(Dataset):
    """A hidden dataset kept where the class says."""

    name: ClassVar[str] = "stored"
    where: ClassVar[Path]
    revision: str = "v3"

    def local(self) -> Path:
        return self.where


class Packaged(Stored):
    """The same, declared public, as data shipped in a package must be."""

    name: ClassVar[str] = "packaged"
    public: ClassVar[bool] = True


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """Register both; returns a place outside the run dir, with a prompt file, the plugins keep their data at."""
    list(DATASETS)
    monkeypatch.setitem(DATASETS._entries, "stored", Stored)
    monkeypatch.setitem(DATASETS._entries, "packaged", Packaged)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "p.jsonl").write_text('{"id": "1", "domain": "math", "prompt": "x"}\n')
    monkeypatch.setattr(Stored, "where", shared / "p.jsonl", raising=False)
    return shared


def _cfg(tmp_path: Path, **kw) -> RunConfig:
    base = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": tmp_path / "run",
    }
    return RunConfig.model_validate({**base, **kw})


GENCHECK = {"name": "qda", "components": ["kl", "gencheck"]}


@pytest.mark.parametrize("field", ["kl_ref", "gencheck_prompts"])
def test_a_named_dataset_under_a_path_the_jobs_read_is_refused(tmp_path, stored, field):
    with pytest.raises(ValueError, match=rf"judge\.{field}.*which the agent's jobs read"):
        _cfg(tmp_path, scheduler={"name": "qsched", "readable": [str(stored)]}, judge={field: {"stored": {}}})


@pytest.mark.parametrize("field", ["kl_ref", "gencheck_prompts"])
def test_a_named_dataset_in_the_run_dir_is_refused(tmp_path, stored, monkeypatch, field):
    planted = tmp_path / "run" / "planted"
    planted.parent.mkdir()
    planted.write_text("{}\n")
    monkeypatch.setattr(Stored, "where", planted, raising=False)
    with pytest.raises(ValueError, match=rf"judge\.{field}.*inside run_dir"):
        _cfg(tmp_path, judge={field: {"stored": {}}})


def test_a_suite_s_task_dataset_is_checked_where_its_data_is(tmp_path, stored):
    ref = tmp_path / "ref"
    ref.mkdir()
    judge = {"kl_cmd": CMD, "kl_ref": str(ref)}
    readable = {"name": "qsched", "readable": [str(stored)]}
    suite = {"lm_eval": {"tasks": ["t"], "datasets": [{"stored": {}}]}}
    with pytest.raises(ValueError, match=r"judge\.suite.*which the agent's jobs read"):
        _cfg(tmp_path, scheduler=readable, judge={**judge, "suite": suite})
    # The shipped tasks are public: they lie in the Python roots every job binds, and the suite still starts.
    _cfg(tmp_path, scheduler=readable, judge={**judge, "suite": {"lm_eval": {"tasks": ["t"]}}})


def test_a_public_task_dataset_may_be_read_but_never_written_by_the_agent(tmp_path, stored, monkeypatch):
    ref = tmp_path / "ref"
    ref.mkdir()
    judge = {
        "kl_cmd": CMD,
        "kl_ref": str(ref),
        "suite": {"lm_eval": {"tasks": ["t"], "datasets": [{"packaged": {}}]}},
    }
    monkeypatch.setattr(Stored, "where", stored, raising=False)
    _cfg(tmp_path, scheduler={"name": "qsched", "readable": [str(stored)]}, judge=judge)
    monkeypatch.setattr(Stored, "where", tmp_path / "run" / "work" / "tasks", raising=False)
    with pytest.raises(ValueError, match=r"judge\.suite.*inside run_dir"):
        _cfg(tmp_path, judge=judge)


@pytest.mark.parametrize("field", ["kl_ref", "gencheck_prompts"])
def test_a_named_dataset_is_recorded_with_its_defaults(tmp_path, stored, field):
    cfg = _cfg(tmp_path, judge={field: {"stored": {}}})
    recorded = cfg.model_dump(mode="json")["judge"][field]
    assert recorded == {"stored": {"revision": "v3"}}
    assert _cfg(tmp_path, judge={field: recorded}).judge == cfg.judge


def test_a_suite_s_dataset_is_recorded_with_its_defaults(tmp_path, stored):
    ref = tmp_path / "ref"
    ref.mkdir()
    suite = {"lm_eval": {"tasks": ["t"], "datasets": [{"stored": {}}]}}
    cfg = _cfg(tmp_path, judge={"kl_cmd": CMD, "kl_ref": str(ref), "suite": suite})
    recorded = cfg.model_dump(mode="json")["judge"]["suite"]
    assert recorded["lm_eval"]["datasets"] == [{"stored": {"revision": "v3"}}]
    assert recorded["lm_eval"]["repeats"] == 1  # the suite's own defaults too
    # what a job gets is the same fully filled form, and reading it back changes nothing
    assert JudgeConfig.model_validate(cfg.judge_config()).suite == recorded
    assert _cfg(tmp_path, judge={"kl_cmd": CMD, "kl_ref": str(ref), "suite": recorded}).judge == cfg.judge


def test_the_wall_and_the_jobs_guard_a_named_prompt_set(tmp_path, stored, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    named = {"gencheck_prompts": {"stored": {}}}
    with pytest.raises(ValueError, match="readable from the agent's shell"):
        isolation._prompts_hidden(_cfg(tmp_path, toolset=GENCHECK, judge=named))
    cfg = _cfg(tmp_path, toolset=GENCHECK, judge=named)
    cfg.scheduler.readable = [stored]
    with pytest.raises(ValueError, match="which the agent's jobs read"):
        isolation._prompts_unread(cfg)


def test_a_public_prompt_set_may_lie_where_the_agent_reads(tmp_path, stored, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    readable = {"name": "qsched", "readable": [str(stored)]}
    cfg = _cfg(tmp_path, toolset=GENCHECK, scheduler=readable, judge={"gencheck_prompts": {"packaged": {}}})
    isolation._prompts_unread(cfg)
    isolation._prompts_hidden(cfg)


def test_a_public_dataset_is_never_a_kl_reference(tmp_path, stored):
    with pytest.raises(ValueError, match="may never read a KL reference"):
        JudgeConfig(contamination="none", kl_ref={"packaged": {}})


def test_a_dataset_of_another_use_is_refused_when_the_config_loads():
    with pytest.raises(ValueError, match=r"judge\.kl_ref\.lm_eval_builtin: serves"):
        JudgeConfig(contamination="none", kl_ref={"lm_eval_builtin": {}})
    with pytest.raises(ValueError, match=r"judge\.gencheck_prompts\.lm_eval_builtin: serves"):
        JudgeConfig(contamination="none", kl_ref="r", gencheck_prompts={"lm_eval_builtin": {}})


def test_the_judge_copies_a_named_prompt_set_read_only_and_jobs_read_the_copy(tmp_path, stored):
    r = GenRig(tmp_path, gencheck_prompts={"stored": {}})
    copy = r.run / "judge" / "gencheck-prompts.jsonl"
    r.host.install_prompts()
    Stored.where.write_text("changed\n")
    ckpt(r.work / "a")
    r.call("gencheck", "a")
    assert r.config("g1")["prompts"] == str(copy) and copy.read_text().startswith('{"id"')
    assert not copy.stat().st_mode & 0o222


def test_the_judge_passes_a_named_reference_as_its_path(tmp_path, stored):
    r = GenRig(tmp_path, kl_ref={"stored": {}})
    r.host.cfg.kl_cmd = ["score-config", "{config}", "{ref}"]
    ckpt(r.work / "a")
    r.call("kl", "a")
    job = r.host.sched.get(r.host.entries[-1].job)
    # The job binds the judge's symlink (a real path stays out of argv), which resolves to the dataset's file.
    reference = Path(job.argv[2])
    assert reference == r.host.jdir / "reference" and reference.resolve() == (stored / "p.jsonl").resolve()
    assert yaml.safe_load(Path(job.argv[1]).read_text())["reference"] == str(stored / "p.jsonl")


def test_a_file_entry_resolves_against_the_config_and_its_record_repeats_the_run(tmp_path, stored):
    run_yaml = tmp_path / "cfg" / "run.yaml"
    run_yaml.parent.mkdir()
    (run_yaml.parent / "ref").mkdir()
    (run_yaml.parent / "p.jsonl").write_text("{}\n")
    run_yaml.write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "format": "w4a16",
                "budget_gpu_hours": 1,
                "contamination": "none",
                "cards": [0],
                "run_dir": str(tmp_path / "run"),
                "judge": {"kl_ref": {"file": {"path": "ref"}}, "gencheck_prompts": "p.jsonl"},
            }
        )
    )
    cfg = RunConfig.load(run_yaml)
    assert cfg.judge.kl_reference() == run_yaml.parent / "ref"
    assert cfg.judge.gencheck_prompts == run_yaml.parent / "p.jsonl"  # a plain path stays a plain path
    record = cfg.resolved()
    assert record["judge"]["kl_ref"] == {"file": {"path": str(run_yaml.parent / "ref")}}
    assert record["judge"]["gencheck_prompts"] == str(run_yaml.parent / "p.jsonl")
    assert record["provenance"]["plugins"]["judge.kl_ref"]["name"] == "file"
    assert "judge.gencheck_prompts" not in record["provenance"]["plugins"]
    again = tmp_path / "resolved.yaml"
    again.write_text(yaml.safe_dump(record, sort_keys=False))
    assert RunConfig.load(again).resolved() == record


def test_a_named_prompt_set_resolves_through_its_plugin_before_the_scan(tmp_path, stored, monkeypatch):
    """The scan and the record ask the dataset for its file: the settings never name it."""
    from qda_agent.flow import readable_inputs

    cfg = _cfg(tmp_path, judge={"gencheck_prompts": {"stored": {}}})
    assert cfg.judge.prompts_path() == stored / "p.jsonl"
    assert stored / "p.jsonl" in {p.resolve() for p in readable_inputs(cfg)}
    assert cfg.resolved()["judge"]["gencheck_prompts"] == {"stored": {"revision": "v3"}}
