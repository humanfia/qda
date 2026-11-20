"""`qda.datasets`: a third-party package adds a dataset with an entry point alone, and every use selects it by name."""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import BaseModel
from qda_kl import cli
from qda_kl.config import ScoreConfig
from qda_kl.datasets import DATASETS
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.lm_eval import LmEvalSuite
from qda_kl.eval.run import digest
from qda_kl.gencheck import GencheckConfig, read_prompts
from qda_kl.scorers import Outcome, TopKScorer

#: The whole third-party package: no line of it, and no line of qda, names the other.
PLUGIN = """
from pathlib import Path
from typing import ClassVar

from qda_kl.datasets import Dataset


class Shelf(Dataset):
    name: ClassVar[str] = "shelf"
    root: str
    item: str
    revision: str = "v3"

    def local(self):
        return Path(self.root) / self.item


def register():
    from qda_kl.datasets import DATASETS
    DATASETS.register("shelf", Shelf)
"""


class Reading(BaseModel):
    kl: float


@pytest.fixture
def shelf(tmp_path, monkeypatch):
    """A real installed distribution declaring ``shelf``; the registry reads it as it reads any entry point."""
    site = tmp_path / "site"
    dist = site / "qda_shelf-1.2.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: qda-shelf\nVersion: 1.2.0\n")
    (dist / "entry_points.txt").write_text("[qda.plugins]\nshelf = qda_shelf:register\n")
    (site / "qda_shelf.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(site))
    from qda_quant import registry

    monkeypatch.setattr(DATASETS, "_entries", dict(DATASETS._entries))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    root = tmp_path / "shelf"
    root.mkdir()
    return root


def test_a_third_party_prompt_set_names_its_own_domains(shelf, tmp_path):
    rows = [{"id": f"l{i}", "domain": "legal", "prompt": "p"} for i in range(2)]
    (shelf / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    named = {"shelf": {"root": str(shelf), "item": "p.jsonl"}}
    cfg = GencheckConfig(checkpoint="c", original="o", out="o", prompts=named, per_domain=2)
    assert cfg.domains == ["legal"]
    assert [p.id for p in read_prompts(cfg)] == ["l0", "l1"]
    assert cfg.prompts == {"shelf": {**named["shelf"], "revision": "v3"}}  # the default is on record
    assert GencheckConfig.model_validate(cfg.model_dump()).prompts == cfg.prompts


def test_a_third_party_reference_is_scored_and_recorded(shelf, tmp_path, monkeypatch):
    seen = []

    def run(self, checkpoint, reference):
        seen.append(reference)
        return Outcome([Reading(kl=0.25)], Reading(kl=0.25))

    monkeypatch.setattr(TopKScorer, "run", run)
    monkeypatch.setattr(TopKScorer, "summary", lambda self, reading: reading.model_dump())
    named = {"shelf": {"root": str(shelf), "item": "ref"}}
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"checkpoint": "c", "reference": named, "out": str(tmp_path / "run")}))
    assert cli.main(["score", str(config)]) == 0
    assert seen == [shelf / "ref"]
    resolved = yaml.safe_load((tmp_path / "run" / "resolved.yaml").read_text())
    assert resolved["reference"] == {
        "shelf": {**named["shelf"], "revision": "v3"}
    }  # the default is on record
    assert resolved["provenance"]["plugins"]["shelf"] == {
        "group": "qda.datasets",
        "package": "qda-shelf",
        "version": "1.2.0",
    }


def test_a_task_dataset_reruns_the_suite_when_edited_and_survives_its_record(shelf, tmp_path):
    tasks = shelf / "tasks"
    tasks.mkdir()
    (tasks / "t.yaml").write_text("task: t\n")
    named = {"lm_eval": {"tasks": ["t"], "datasets": [{"shelf": {"root": str(shelf), "item": "tasks"}}]}}
    cfg = EvalConfig(reference="r", checkpoints={"k": "c"}, out=str(tmp_path / "o"), suites=[named])
    again = EvalConfig.from_yaml(yaml.safe_dump(cfg.to_dict()))
    assert again.to_dict() == cfg.to_dict()
    before = digest(cfg.suites[0], "r", "c")
    assert digest(again.suites[0], "r", "c") == before
    (tasks / "__pycache__").mkdir()
    (tasks / "__pycache__" / "t.pyc").write_text("compiled by the harness's run")
    assert digest(cfg.suites[0], "r", "c") == before  # running a task does not make it stale
    (tasks / "t.yaml").write_text("task: t\nnum_fewshot: 1\n")
    assert digest(cfg.suites[0], "r", "c") != before


@pytest.mark.parametrize(
    ("build", "why"),
    [
        (lambda v: ScoreConfig(checkpoint="c", reference=v, out="o"), "serves"),
        (lambda v: LmEvalSuite(tasks=["t"], datasets=[v]), "serves"),
    ],
)
def test_a_dataset_of_another_use_is_refused_at_load(build, why):
    with pytest.raises(ValueError, match=why):
        build({"gencheck_builtin": {}})


def test_a_bad_selection_names_the_known_datasets_and_the_setting():
    with pytest.raises(ValueError, match=r"reference\.nope: unknown dataset; known: .*'file'"):
        ScoreConfig(checkpoint="c", reference={"nope": {}}, out="o")
    with pytest.raises(ValueError, match=r"reference\.file\.pth"):
        ScoreConfig(checkpoint="c", reference={"file": {"path": "p", "pth": "p"}}, out="o")
    with pytest.raises(ValueError, match="settings are a mapping"):
        ScoreConfig(checkpoint="c", reference={"file": "p"}, out="o")


def test_the_built_in_word_and_key_read_the_shipped_set():
    word = GencheckConfig(checkpoint="c", original="o", out="o")
    assert word.prompts == "builtin"  # a recorded config keeps its meaning
    key = word.model_copy(update={"prompts": {"gencheck_builtin": {}}})
    assert read_prompts(word) == read_prompts(key) and word.domains
