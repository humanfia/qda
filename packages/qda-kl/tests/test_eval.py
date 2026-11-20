"""``qda eval`` resumes: done jobs are skipped, a changed setting reruns only its suite, a refusal keeps its error."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.report import load, table_main, table_to_markdown
from qda_kl.eval.run import jobs, run
from qda_kl.eval.suites import SUITES, Suite, SuiteResult, parse_suite
from qda_quant.exit import ENVIRONMENT
from qda_quant.step import Refused

CALLS: list[tuple[str, ...]] = []


class _Stub(Suite):
    """Counts its runs; ``value`` is the metric, ``refuse`` names a checkpoint directory it refuses."""

    name: ClassVar[str] = "stub_a"
    value: float = 1.0
    refuse: str = ""

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        CALLS.append((self.name, Path(checkpoint).name))
        if self.refuse and Path(checkpoint).name.startswith(self.refuse):
            raise RuntimeError("refused on purpose")
        return SuiteResult(suite=self.name, metrics={"m": self.value}, higher_is_better={"m": True})


class _StubB(_Stub):
    name: ClassVar[str] = "stub_b"


class _Part(Suite):
    """One labelled part of :class:`_TwoParts`: the metric ``label`` at ``value``."""

    name: ClassVar[str] = "stub_parts"
    label: str
    value: float = 1.0

    def part_label(self) -> str | None:
        return self.label

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        CALLS.append((self.name, Path(checkpoint).name, self.label))
        return SuiteResult(
            suite=self.name, metrics={self.label: self.value}, higher_is_better={self.label: True}
        )


class _TwoParts(Suite):
    """A suite of two independent parts, each with its own setting; ``split=false`` makes it one job."""

    name: ClassVar[str] = "stub_parts"
    a: float = 1.0
    b: float = 1.0
    split: bool = True

    def parts(self) -> list[Suite]:
        if not self.split:
            return [self]
        return [_Part(label="a", value=self.a), _Part(label="b", value=self.b)]

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        CALLS.append((self.name, Path(checkpoint).name))
        return SuiteResult(
            suite=self.name, metrics={"a": self.a, "b": self.b}, higher_is_better={"a": True, "b": True}
        )


class _Unprepared(_Stub):
    """A suite whose preparation refuses: the FlashInfer toolchain, on a machine without it."""

    name: ClassVar[str] = "stub_unprepared"
    code: int = 2

    def prepare(self, reference: str, work: Path) -> None:
        raise Refused("card 0 has 3.1 GiB free", code=self.code)


class _BadParts(Suite):
    """A suite whose parts carry the given labels, to be refused when they cannot name result files."""

    name: ClassVar[str] = "stub_bad"
    labels: list[str] = []

    def parts(self) -> list[Suite]:
        return [_Part(label=label) for label in self.labels]


for _cls in (_Stub, _StubB, _TwoParts, _BadParts, _Unprepared):
    if _cls.name not in SUITES:
        SUITES.register(_cls.name, _cls)


def _go(tmp: Path, a: dict, b: dict) -> tuple[int, list[tuple[str, str]]]:
    for d in ("ref", "c1", "c2"):
        (tmp / d).mkdir(exist_ok=True)
    cfg = {
        "reference": str(tmp / "ref"),
        "checkpoints": {"one": str(tmp / "c1"), "two": str(tmp / "c2")},
        "out": str(tmp / "out"),
        "suites": [{"stub_a": a}, {"stub_b": b}],
    }
    CALLS.clear()
    code = run(EvalConfig.from_yaml(yaml.safe_dump(cfg)), log=lambda _: None)
    return code, sorted(CALLS)


@pytest.mark.parametrize(("code", "want"), [(2, 2), (ENVIRONMENT, 3)])
def test_a_refused_prepare_keeps_its_status(tmp_path: Path, code: int, want: int) -> None:
    """A suite that cannot be prepared refuses every job of it, and the run exits with why it could not.

    The FlashInfer toolchain is the case: preparing a suite starts an engine, which refuses on a machine
    without the kernels, `nvcc` or `ninja` — the machine's status, not the input's.
    """
    for d in ("ref", "c1"):
        (tmp_path / d).mkdir()
    cfg = {
        "reference": str(tmp_path / "ref"),
        "checkpoints": {"one": str(tmp_path / "c1")},
        "out": str(tmp_path / "out"),
        "suites": [{"stub_unprepared": {"code": code}}],
    }
    assert run(EvalConfig.from_yaml(yaml.safe_dump(cfg)), log=lambda _: None) == want
    failed = json.loads((tmp_path / "out" / "results" / "one" / "stub_unprepared.json").read_text())
    assert failed["error"] == "prepare: card 0 has 3.1 GiB free"


def test_resume_reruns_only_what_changed(tmp_path: Path) -> None:
    res = tmp_path / "out" / "results"
    all4 = [("stub_a", "c1"), ("stub_a", "c2"), ("stub_b", "c1"), ("stub_b", "c2")]
    assert _go(tmp_path, {}, {}) == (0, all4)
    assert json.loads((res / "one" / "stub_a.json").read_text())["checkpoint"] == "one"
    assert _go(tmp_path, {}, {}) == (0, [])  # nothing to do
    assert _go(tmp_path, {"value": 2.0}, {}) == (0, [("stub_a", "c1"), ("stub_a", "c2")])
    # A refused job keeps its exact error instead of a measurement; the other job of the suite still runs.
    assert _go(tmp_path, {"value": 2.0}, {"refuse": "c2"}) == (2, [("stub_b", "c1"), ("stub_b", "c2")])
    failed = json.loads((res / "two" / "stub_b.json").read_text())
    assert failed["error"] == "refused on purpose"
    assert failed["metrics"] == {}
    assert (res / "one" / "stub_b.json").exists()
    assert _go(tmp_path, {"value": 2.0}, {"refuse": "c2"}) == (2, [("stub_b", "c2")])  # retried, not cached
    # The resolved config is a config: it records every default.
    resolved = yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())
    assert resolved["suites"][1] == {"stub_b": {"value": 1.0, "refuse": "c2"}}
    assert (
        EvalConfig.from_yaml((tmp_path / "out" / "resolved.yaml").read_text()).to_dict()["suites"]
        == (resolved["suites"])
    )


def _config(tmp: Path, suites: list[dict]) -> EvalConfig:
    (tmp / "ref").mkdir(exist_ok=True)
    (tmp / "c1").mkdir(exist_ok=True)
    cfg = {
        "reference": str(tmp / "ref"),
        "checkpoints": {"one": str(tmp / "c1")},
        "out": str(tmp / "out"),
        "suites": suites,
    }
    return EvalConfig.from_yaml(yaml.safe_dump(cfg))


def test_resolved_pins_every_model_the_run_reads(tmp_path: Path) -> None:
    """The record names the bytes of the reference and of each checkpoint, not only their paths."""
    from qda_quant.intake.fingerprint import model_signature

    cfg = _config(tmp_path, [{"stub_a": {}}])
    for d in ("ref", "c1"):
        (tmp_path / d / "config.json").write_text('{"model_type": "llama"}')
    assert run(cfg, log=lambda _: None) == 0
    sources = yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())["provenance"]["sources"]
    assert sources["reference"] == {
        "snapshot": None,
        "revision": None,
        "signature": model_signature(tmp_path / "ref"),
    }
    assert sources["checkpoints"]["one"] == {
        "snapshot": None,
        "revision": None,
        "signature": model_signature(tmp_path / "c1"),
    }


def _go_parts(
    tmp: Path, a: float, b: float, only: tuple[str, ...] = (), split: bool = True
) -> list[tuple[str, ...]]:
    cfg = _config(tmp, [{"stub_parts": {"a": a, "b": b, "split": split}}])
    CALLS.clear()
    assert run(cfg, only, log=lambda _: None) == 0
    return sorted(CALLS)


def test_a_failed_row_prints_its_error_and_a_rerun_replaces_it(tmp_path: Path) -> None:
    """A refusal is a result with the exact error; the table shows it, and a success replaces it."""
    cfg = _config(tmp_path, [{"stub_a": {"refuse": "c1"}}])
    assert run(cfg, log=lambda _: None) == 2
    path = tmp_path / "out" / "results" / "one" / "stub_a.json"
    assert json.loads(path.read_text())["error"] == "refused on purpose"
    assert table_to_markdown(load(tmp_path / "out")) == "\n".join(
        ["| checkpoint | error |", "| --- | --- |", "| one | refused on purpose |"]
    )
    assert run(_config(tmp_path, [{"stub_a": {}}]), log=lambda _: None) == 0
    assert json.loads(path.read_text())["error"] is None
    assert "error" not in table_to_markdown(load(tmp_path / "out"))


def test_a_table_where_every_row_failed_still_prints(tmp_path: Path, capsys) -> None:
    """Every row failed: the table still prints each checkpoint with its error."""
    for name in ("ref", "c1", "c2"):
        (tmp_path / name).mkdir()
    cfg = EvalConfig.from_yaml(
        yaml.safe_dump(
            {
                "reference": str(tmp_path / "ref"),
                "checkpoints": {"one": str(tmp_path / "c1"), "two": str(tmp_path / "c2")},
                "out": str(tmp_path / "out"),
                "suites": [{"stub_a": {"refuse": "c"}}],
            }
        )
    )
    assert run(cfg, log=lambda _: None) == 2
    error = "refused on purpose"
    assert table_to_markdown(load(tmp_path / "out")).splitlines() == [
        "| checkpoint | error |",
        "| --- | --- |",
        f"| one | {error} |",
        f"| two | {error} |",
    ]
    assert table_main(tmp_path / "out") == 0
    assert error in capsys.readouterr().out


def test_editing_generation_config_makes_the_result_stale(tmp_path: Path) -> None:
    """The result records the metadata digests that decide it, so an edit at the same path reruns the job."""
    ckpt = tmp_path / "c1"
    ckpt.mkdir()
    generation = ckpt / "generation_config.json"
    generation.write_text('{"temperature": 0.6}')
    (ckpt / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))
    (ckpt / "chat_template.jinja").write_text("{{ x }}")
    cfg = _config(tmp_path, [{"stub_a": {}}])
    assert run(cfg, log=lambda _: None) == 0
    path = tmp_path / "out" / "results" / "one" / "stub_a.json"
    before = json.loads(path.read_text())
    source = before["detail"]["checkpoint_source"]
    assert source["generation_config"] == hashlib.sha256(generation.read_bytes()).hexdigest()
    assert source["tokenizer_config"] and source["chat_template"] and source["commit"] is None
    CALLS.clear()
    assert run(cfg, log=lambda _: None) == 0 and CALLS == []  # still cached
    generation.write_text('{"temperature": 0.7}')
    CALLS.clear()
    assert run(cfg, log=lambda _: None) == 0 and CALLS == [("stub_a", "c1")]
    after = json.loads(path.read_text())
    assert after["config_digest"] != before["config_digest"]
    assert after["detail"]["checkpoint_source"]["generation_config"] != source["generation_config"]


def test_a_suite_is_one_job_per_part(tmp_path: Path) -> None:
    res = tmp_path / "out" / "results" / "one"
    assert _go_parts(tmp_path, 1, 1) == [("stub_parts", "c1", "a"), ("stub_parts", "c1", "b")]
    assert json.loads((res / "stub_parts.a.json").read_text())["metrics"] == {"a": 1.0}
    assert json.loads((res / "stub_parts.b.json").read_text())["metrics"] == {"b": 1.0}
    assert _go_parts(tmp_path, 1, 1) == []  # both parts cached
    assert _go_parts(tmp_path, 2, 1) == [("stub_parts", "c1", "a")]  # only the changed part reruns
    assert _go_parts(tmp_path, 3, 2, ("one/stub_parts.b",)) == [("stub_parts", "c1", "b")]  # one part only
    assert _go_parts(tmp_path, 4, 3, ("one/stub_parts",)) == [
        ("stub_parts", "c1", "a"),
        ("stub_parts", "c1", "b"),
    ]
    cfg = EvalConfig.from_yaml((tmp_path / "out" / "resolved.yaml").read_text())
    assert [job.label for job in jobs(cfg, ["one/stub_parts.b"])] == ["b"]
    with pytest.raises(ValueError, match="stub_parts.c"):
        jobs(cfg, ["one/stub_parts.c"])


def test_a_split_suite_replaces_its_one_job_result(tmp_path: Path) -> None:
    """A result file of the parts a suite had before cannot override what the current parts measure."""
    res = tmp_path / "out" / "results" / "one"
    assert _go_parts(tmp_path, 0.1, 0.1, split=False) == [("stub_parts", "c1")]
    assert json.loads((res / "stub_parts.json").read_text())["metrics"] == {"a": 0.1, "b": 0.1}
    assert _go_parts(tmp_path, 0.9, 0.9) == [("stub_parts", "c1", "a"), ("stub_parts", "c1", "b")]
    assert not (res / "stub_parts.json").exists()
    assert table_to_markdown(load(tmp_path / "out")) == "\n".join(
        ["| checkpoint | a | b |", "| --- | --- | --- |", "| one | **90.0%** | **90.0%** |"]
    )


def test_two_parts_may_not_share_a_result_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stub_bad.a"):
        jobs(_config(tmp_path, [{"stub_bad": {"labels": ["a", "a"]}}]))


def test_a_label_is_one_name_segment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not one name segment"):
        jobs(_config(tmp_path, [{"stub_bad": {"labels": ["a.b"]}}]))


def test_a_suite_name_may_not_contain_a_dot() -> None:
    with pytest.raises(ValueError, match=r"suites\.0\.x\.y: a suite name cannot contain"):
        parse_suite({"x.y": {}}, "suites.0")


def test_resolved_records_the_kl_corpus_plugin(tmp_path: Path, monkeypatch) -> None:
    """The run records the corpus plugin the kl suite reads, not only the suite itself."""
    from qda_kl import holdout as holdout_defaults
    from qda_kl.eval.suites import KLSuite

    # the suite's default corpus is the holdout split; a file stands in for the cached one, so no corpus build
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text('{"id": "a", "prompt": "2+2?"}\n')
    monkeypatch.setattr(holdout_defaults, "default_prompts", lambda: holdout)
    monkeypatch.setattr(KLSuite, "prepare", lambda self, reference, work: None)
    monkeypatch.setattr(
        KLSuite,
        "run",
        lambda self, checkpoint, reference, work: SuiteResult(
            suite=self.name, metrics={"kl": 0.0}, higher_is_better={"kl": False}
        ),
    )
    for d in ("ref", "ck"):
        (tmp_path / d).mkdir()
    cfg = EvalConfig.from_yaml(
        yaml.safe_dump(
            {
                "reference": str(tmp_path / "ref"),
                "checkpoints": {"ck": str(tmp_path / "ck")},
                "out": str(tmp_path / "out"),
                "suites": [{"kl": {}}],
            }
        )
    )
    assert run(cfg, log=lambda _: None) == 0
    prov = yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())["provenance"]
    assert prov["plugins"]["kl"]["group"] == "qda.eval_suites"
    assert prov["plugins"]["kl_holdout"]["group"] == "qda.corpora"


def test_the_kl_suite_puts_its_layout_on_both_engines(tmp_path: Path) -> None:
    """One tp, serve_env and serve_args for the reference build and the score, and a reference per layout."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    corpus = {"file": {"path": str(rows)}}
    two = parse_suite(
        {
            "kl": {
                "corpus": corpus,
                "tp": 2,
                "serve_env": {"TOKENIZERS_PARALLELISM": "false"},
                "serve_args": {"enable_expert_parallel": True, "pipeline_parallel_size": 2},
                "scorer": {
                    "topk": {
                        "serve_env": {"TOKENIZERS_PARALLELISM": "true", "A": "1"},
                        "serve_args": {"dtype": "bfloat16"},
                    }
                },
            }
        },
        "suites.0",
    )
    assert two.cards() == 4
    at = two._scorer()
    assert at.tp == 2
    assert at.serve_env == {"TOKENIZERS_PARALLELISM": "false", "A": "1"}  # the suite's value wins
    assert at.serve_args == {
        "dtype": "bfloat16",
        "enable_expert_parallel": True,
        "pipeline_parallel_size": 2,
    }
    plain = parse_suite({"kl": {"corpus": corpus}}, "suites.0")
    assert two.reference_dir("m", tmp_path) != plain.reference_dir("m", tmp_path)
    env_only = parse_suite({"kl": {"corpus": corpus, "serve_env": {"A": "1"}}}, "suites.0")
    assert env_only.reference_dir("m", tmp_path) != plain.reference_dir("m", tmp_path)
    with pytest.raises(ValueError, match="serve_env names NCCL_NVLS_ENABLE=0"):
        parse_suite(
            {
                "kl": {
                    "corpus": corpus,
                    "nccl_nvls": True,
                    "serve_env": {"NCCL_NVLS_ENABLE": "0"},
                }
            },
            "suites.0",
        )
    with pytest.raises(ValueError, match="the scorer names tp=3"):
        parse_suite({"kl": {"corpus": corpus, "tp": 2, "scorer": {"topk": {"tp": 3}}}}, "suites.0")
    on = parse_suite({"kl": {"corpus": corpus, "nccl_nvls": True}}, "suites.0")
    assert on._scorer().nccl_nvls is True
    assert on.reference_dir("m", tmp_path) != plain.reference_dir("m", tmp_path)
    with pytest.raises(ValueError, match="the scorer names nccl_nvls=True"):
        parse_suite({"kl": {"corpus": corpus, "scorer": {"topk": {"nccl_nvls": True}}}}, "suites.0")
    with pytest.raises(
        ValueError, match="serve_env names NCCL_NVLS_ENABLE=0"
    ):  # the suite's value, not the default
        parse_suite(
            {
                "kl": {
                    "corpus": corpus,
                    "nccl_nvls": True,
                    "scorer": {"topk": {"serve_env": {"NCCL_NVLS_ENABLE": "0"}}},
                }
            },
            "suites.0",
        )


def test_the_kl_suite_builds_its_reference_on_the_suite_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``prepare`` hands the reference build the suite's tp, serve_env and serve_args, not its defaults."""
    import qda_kl.reference as reference

    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    suite = parse_suite(
        {
            "kl": {
                "corpus": {"file": {"path": str(rows)}},
                "tp": 2,
                "serve_env": {"A": "1"},
                "serve_args": {"pipeline_parallel_size": 2},
            }
        },
        "suites.0",
    )
    seen: dict = {}
    monkeypatch.setattr(reference, "build", lambda *a, **k: seen.update(k))
    suite.prepare(str(tmp_path / "m"), tmp_path / "out")
    assert (seen["tp"], seen["serve_env"], seen["serve_args"]) == (
        2,
        {"A": "1"},
        {"pipeline_parallel_size": 2},
    )


def test_the_paired_settings_reach_resolved_yaml_and_are_validated(tmp_path: Path) -> None:
    """``paired`` is recorded with the run's other defaults, so a rendering is reproducible from the run alone."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "c1").mkdir()
    cfg = {
        "reference": str(tmp_path / "ref"),
        "checkpoints": {"one": str(tmp_path / "c1")},
        "out": str(tmp_path / "out"),
        "suites": [{"stub_a": {}}],
    }
    assert EvalConfig.from_yaml(yaml.safe_dump(cfg)).to_dict()["paired"] == {"resamples": 1000, "seed": 17}
    assert run(EvalConfig.from_yaml(yaml.safe_dump(cfg)), (), log=lambda _: None) == 0
    resolved = yaml.safe_load((tmp_path / "out" / "resolved.yaml").read_text())
    assert resolved["paired"] == {"resamples": 1000, "seed": 17}

    cfg["paired"] = {"resamples": 23, "seed": 19}
    cfg["out"] = str(tmp_path / "custom")
    assert run(EvalConfig.from_yaml(yaml.safe_dump(cfg)), (), log=lambda _: None) == 0
    assert yaml.safe_load((tmp_path / "custom" / "resolved.yaml").read_text())["paired"] == {
        "resamples": 23,
        "seed": 19,
    }

    cfg["paired"] = {"resamples": 0}
    with pytest.raises(ValueError, match="resamples: Input should be greater than or equal to 1"):
        EvalConfig.from_yaml(yaml.safe_dump(cfg))


def test_a_resolved_kl_suite_reloads_as_it_stands(tmp_path: Path) -> None:
    """The config `qda eval` writes carries the suite's tp into its scorer, so the file parses again."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n')
    (tmp_path / "ref").mkdir()
    (tmp_path / "ckpt").mkdir()
    cfg = EvalConfig.from_yaml(
        yaml.safe_dump(
            {
                "reference": str(tmp_path / "ref"),
                "checkpoints": {"one": str(tmp_path / "ckpt")},
                "out": str(tmp_path / "out"),
                "suites": [
                    {
                        "kl": {
                            "corpus": {"file": {"path": str(rows)}},
                            "tp": 2,
                            "nccl_nvls": True,
                            "serve_args": {"pipeline_parallel_size": 2},
                            "scorer": {"topk": {"repeat": 2, "serve_args": {"dtype": "bfloat16"}}},
                        }
                    }
                ],
            }
        )
    )
    wrote = cfg.to_dict()
    again = EvalConfig.from_yaml(yaml.safe_dump(wrote))
    assert again.to_dict()["suites"] == wrote["suites"]
    assert again.suites[0]._scorer() == cfg.suites[0]._scorer()
    assert again.suites[0].cards() == 4


@pytest.mark.parametrize("key", ["tensor_parallel_size", "pipeline_parallel_size", "enable_expert_parallel"])
def test_the_scorer_may_not_hold_a_parallel_arg(tmp_path: Path, key: str) -> None:
    """A parallel argument on the scorer reaches only the checkpoint, so the suite owns it."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n')
    with pytest.raises(ValueError, match=f"the scorer names {key}; the suite owns the layout"):
        parse_suite(
            {
                "kl": {
                    "corpus": {"file": {"path": str(rows)}},
                    "tp": 2,
                    "scorer": {"topk": {"serve_args": {key: 2}}},
                }
            },
            "suites.0",
        )


def test_two_runs_share_frozen_refs_with_a_minimal_scorer(tmp_path: Path, monkeypatch) -> None:
    """Separate eval runs build one reference and record its identity even when a plugin returns only KL."""
    import qda_kl.reference as reference
    import qda_kl.scorers as scorers
    import torch
    from pydantic import BaseModel
    from qda_kl.store import Reference, ReferenceWriter
    from qda_kl.topk import anchor_topk
    from qda_quant.registry import Registry

    class Reading(BaseModel):
        kl: float = 0.5

    class MinimalScorer(scorers.KLScorer):
        name: ClassVar[str] = "minimal"

        def run(self, checkpoint: str, reference: Path) -> scorers.Outcome:
            assert reference == built[0]
            return scorers.Outcome([], Reading())

    refs, built, identities = tmp_path / "shared", [], []

    def build(model, prompts, out, **kwargs):  # noqa: ANN001, ARG001
        built.append(Path(out))
        writer = ReferenceWriter(Path(out), k=2, vocab=4)
        logp = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).log_softmax(-1)
        writer.add("a", "math", [0, 1], 1, anchor_topk(logp, 2))
        return writer.close()

    monkeypatch.setattr(reference, "build", build)
    monkeypatch.setattr(scorers, "SCORERS", Registry("scorers"))
    scorers.SCORERS.register(MinimalScorer.name, MinimalScorer)
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    settings = {"corpus": {"file": {"path": str(rows)}}, "refs": str(refs), "scorer": {"minimal": {}}}
    monkeypatch.setattr("qda_kl.eval.run.local", lambda model: model)
    for key in ("run1", "run2"):
        work = tmp_path / key
        work.mkdir()
        cfg = _config(work, [{"kl": settings}])
        cfg.reference = "model"
        assert run(cfg, log=lambda _: None) == 0
        got = load(cfg.out).results[0]
        identities.append(got.detail["reference_sha256"])
    assert len(built) == 1 and built[0].parent == refs
    assert identities == [Reference(built[0]).content_sha256] * 2


@pytest.mark.parametrize("where", ["reference", "checkpoint"])
def test_a_gguf_model_is_refused_with_the_decode_route(tmp_path: Path, where: str) -> None:
    """Every suite loads its model in vLLM, which has no GGUF loader: name the documented decode step."""
    ref, checkpoint = tmp_path / "ref", tmp_path / "c1"
    ref.mkdir()
    checkpoint.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    data = {
        "reference": str(gguf if where == "reference" else ref),
        "checkpoints": {"one": str(gguf if where == "checkpoint" else checkpoint)},
        "out": str(tmp_path / "out"),
        "suites": [{"kl": {}}],
    }
    with pytest.raises(Refused) as why:
        EvalConfig.from_yaml(yaml.safe_dump(data))
    assert "does not score a .gguf" in str(why.value)
    assert "qda decode-gguf" in why.value.fix


def test_the_cli_prints_the_gguf_refusal_as_two_lines(tmp_path: Path, capsys) -> None:
    """A refused config prints its own fix line: the reader copies the decode command from stderr."""
    from types import SimpleNamespace

    from qda_kl.eval.cli import main as eval_main

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reference": str(tmp_path / "ref"),
                "checkpoints": {"one": str(gguf)},
                "out": str(tmp_path / "out"),
                "suites": [{"sanity": {}}],
            }
        )
    )
    args = SimpleNamespace(
        config=config,
        out=None,
        format=None,
        baseline=None,
        resamples=None,
        seed=None,
        set=[],
        only=[],
        server=None,
        dry_run=True,
        suite=None,
        lcb_sandbox=None,
    )
    assert eval_main(args) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"error: {config}: checkpoints.one: qda eval does not score a .gguf model")
    assert lines[1] == (
        "fix:   decode the file to a bf16 checkpoint (`qda decode-gguf`), then evaluate that directory"
    )


def test_a_gguf_build_directory_is_refused_like_a_gguf_file(tmp_path: Path) -> None:
    """`qda build` writes a directory holding the .gguf and no config.json; that is not a vLLM checkpoint."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    data = {
        "reference": str(tmp_path / "ref"),
        "checkpoints": {"one": str(build)},
        "out": str(tmp_path / "out"),
        "suites": [{"kl": {}}],
    }
    with pytest.raises(Refused, match="does not score a .gguf"):
        EvalConfig.from_yaml(yaml.safe_dump(data))
