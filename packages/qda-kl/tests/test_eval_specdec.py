"""The specdec suite's acceptance reading from vLLM's counter snapshot, and the layout both engines run on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from qda_kl.eval.lm_eval import LmEvalSuite
from qda_kl.eval.specdec import ACCEPTED_PER_POS, DIGEST, REFERENCE, SpecDecSuite, read_spec_metrics
from qda_kl.eval.suites import LayoutSuite, parse_suite
from qda_kl.serve import ServeSettings


class _Counter:
    """A ``vllm.v1.metrics.reader.Counter``: its name and value."""

    def __init__(self, name: str, value: int) -> None:
        self.name, self.value = name, value


class _Vector:
    """A ``vllm.v1.metrics.reader.Vector``: the accepted tokens at each draft position."""

    def __init__(self, values: list[int]) -> None:
        self.name, self.values = ACCEPTED_PER_POS, values


def test_acceptance_sums_every_engine_and_position() -> None:
    """Two engines' counters add up; the rate is accepted over drafted and the length counts the target's token."""
    snapshot = [
        _Counter("vllm:spec_decode_num_drafts", 15),
        _Counter("vllm:spec_decode_num_draft_tokens", 30),
        _Counter("vllm:spec_decode_num_accepted_tokens", 11),
        _Vector([9, 2]),
        _Counter("vllm:num_requests_running", 0),
        _Counter("vllm:spec_decode_num_drafts", 15),
        _Counter("vllm:spec_decode_num_draft_tokens", 30),
        _Counter("vllm:spec_decode_num_accepted_tokens", 7),
        _Vector([7, 0]),
    ]
    got = read_spec_metrics(snapshot)
    assert (got["num_drafts"], got["num_draft_tokens"], got["num_accepted_tokens"]) == (30, 60, 18)
    assert got["acceptance_rate"] == pytest.approx(0.3)
    assert got["mean_acceptance_length"] == pytest.approx(1.6)
    assert got["per_position_acceptance"] == pytest.approx([16 / 30, 2 / 30])


def test_a_snapshot_without_the_counters_is_refused() -> None:
    with pytest.raises(RuntimeError, match="did not run with speculative decoding"):
        read_spec_metrics([_Counter("vllm:num_requests_running", 0)])


def _suite(tmp_path: Path, **cfg: Any) -> SpecDecSuite:
    """A specdec suite on two prompt rows (one on each side of the split), with ``cfg``'s settings."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    return parse_suite({"specdec": {"corpus": {"file": {"path": str(rows)}}, **cfg}}, "suites.0")


def test_the_layout_reaches_both_engines(tmp_path: Path) -> None:
    """``tp`` and ``serve_args`` reach the reference's reading and the checkpoint, and ``cards`` is their size."""
    suite = _suite(tmp_path, tp=2, serve_args={"enable_expert_parallel": True, "pipeline_parallel_size": 2})
    assert suite.cards() == 4
    got = suite.engine_kwargs("m", ServeSettings(args={"dtype": "bfloat16", **suite.serve_args}))
    assert got["tensor_parallel_size"] == 2
    assert got["pipeline_parallel_size"] == 2
    assert got["enable_expert_parallel"] is True
    assert got["disable_log_stats"] is False
    assert got["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 1}


def test_the_original_is_read_on_the_suite_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``prepare`` measures the original with the suite's layout, not empty settings."""
    suite = _suite(
        tmp_path,
        tp=2,
        serve_env={"A": "1"},
        serve_args={"pipeline_parallel_size": 2},
        nccl_nvls=True,
        sampling={"temperature": 0.0},
    )
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "generation_config.json").write_text("{}")
    seen: list[tuple[dict[str, Any], dict[str, Any], bool]] = []

    def measure(self, model, settings, rows, sampling, **kw):  # noqa: ANN001, ARG001
        seen.append((settings.env, settings.args, settings.nccl_nvls, kw["checkpoint"]))
        return {
            "acceptance_rate": 1.0,
            "mean_acceptance_length": 2.0,
            "num_drafts": 1,
            "num_draft_tokens": 1,
            "num_accepted_tokens": 1,
            "per_position_acceptance": [1.0],
            "seconds": {"load": 0.0, "generate": 0.0},
        }

    monkeypatch.setattr(SpecDecSuite, "_measure", measure)
    suite.prepare(str(ref), tmp_path / "out")
    assert seen == [({"A": "1"}, {"pipeline_parallel_size": 2}, True, False)]  # the original's own draft


def test_nccl_nvls_reaches_the_checkpoint_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` starts the checkpoint's engine with the suite's ``nccl_nvls``."""
    suite = _suite(tmp_path, nccl_nvls=True)
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "generation_config.json").write_text("{}")
    work = tmp_path / "out"
    reading = {
        "acceptance_rate": 1.0,
        "mean_acceptance_length": 2.0,
        "num_drafts": 1,
        "num_draft_tokens": 1,
        "num_accepted_tokens": 1,
        "per_position_acceptance": [1.0],
    }
    stored = suite.reference_dir(str(ref), work) / REFERENCE
    stored.parent.mkdir(parents=True)
    stored.write_text(
        json.dumps({**reading, "sampling": {"temperature": 0.0}, "seconds": {"load": 0.0, "generate": 0.0}})
    )
    (stored.parent / DIGEST).write_text(hashlib.sha256(stored.read_bytes()).hexdigest() + "\n")
    seen: list[ServeSettings] = []

    def measure(self, model, settings, rows, sampling, **kw):  # noqa: ANN001, ARG001
        seen.append((settings, kw["checkpoint"]))
        return {
            **reading,
            "seconds": {"load": 0.0, "generate": 0.0},
            "serve": {},
            "vllm": "0",
            "versions": {},
        }

    monkeypatch.setattr(SpecDecSuite, "_measure", measure)
    suite.run("checkpoint", str(ref), work)
    assert seen[0][0].nccl_nvls is True and seen[0][1] is True  # the checkpoint's own draft


def test_a_layout_may_not_name_nvls_another_way(tmp_path: Path) -> None:
    """``nccl_nvls`` has one home: a ``serve_env`` naming another value is refused on every layout suite."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n')
    with pytest.raises(ValueError, match="serve_env names NCCL_NVLS_ENABLE=1; set nccl_nvls instead"):
        parse_suite(
            {
                "specdec": {
                    "corpus": {"file": {"path": str(rows)}},
                    "serve_env": {"NCCL_NVLS_ENABLE": "1"},
                }
            },
            "suites.0",
        )
    with pytest.raises(ValueError, match="set nccl_nvls instead"):
        LmEvalSuite(tasks=["t1"], serve_env={"NCCL_NVLS_ENABLE": "1"})


def test_a_reference_is_measured_per_layout(tmp_path: Path) -> None:
    """A card count or another ``serve_args`` is another reading, so the cached reference is not reused."""
    plain = _suite(tmp_path)
    assert _suite(tmp_path, tp=2).reference_dir("m", tmp_path) != plain.reference_dir("m", tmp_path)
    layered = _suite(tmp_path, serve_args={"pipeline_parallel_size": 2})
    assert layered.reference_dir("m", tmp_path) != plain.reference_dir("m", tmp_path)


@pytest.mark.parametrize("key", ["kl", "specdec"])
def test_both_suites_read_the_layout_the_same(tmp_path: Path, key: str) -> None:
    """The layout is the shared base's: one ``cards`` and one card-count refusal for every suite on it."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    corpus = {"file": {"path": str(rows)}}
    suite = parse_suite(
        {key: {"corpus": corpus, "tp": 2, "serve_args": {"pipeline_parallel_size": 2}}}, "suites.0"
    )
    assert isinstance(suite, LayoutSuite)
    assert suite.cards() == 4
    with pytest.raises(ValueError, match="set tp=2 instead"):
        parse_suite({key: {"corpus": corpus, "tp": 2, "serve_args": {"tensor_parallel_size": 3}}}, "suites.0")


@pytest.mark.parametrize("key", ["kl", "specdec"])
def test_a_written_suite_keeps_the_corpus_fields_before_the_layout(tmp_path: Path, key: str) -> None:
    """A resolved config writes corpus, refs, then the layout: the order it had before the corpus class merged."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n{"id": "b", "prompt": "3+3?"}\n')
    suite = parse_suite(
        {key: {"corpus": {"file": {"path": str(rows)}}, "refs": "refs", "tp": 2, "serve_args": {}}},
        "suites.0",
    )
    assert list(suite.to_dict()[key])[:5] == ["corpus", "refs", "tp", "serve_env", "serve_args"]


@pytest.mark.parametrize("stages", [0, -1, 2.5, "2"])
def test_a_pipeline_split_is_a_positive_integer(tmp_path: Path, stages: Any) -> None:
    """The shared layout refuses any other pipeline split, for a corpus suite and for ``lm_eval`` alike."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "prompt": "2+2?"}\n')
    corpus = {"file": {"path": str(rows)}}
    with pytest.raises(ValueError, match=r"serve_args\.pipeline_parallel_size"):
        parse_suite({"kl": {"corpus": corpus, "serve_args": {"pipeline_parallel_size": stages}}}, "suites.0")
    with pytest.raises(ValueError, match=r"serve_args\.pipeline_parallel_size"):
        LmEvalSuite(tasks=["t1"], serve_args={"pipeline_parallel_size": stages})


def test_a_checkpoint_may_not_bring_its_own_card_count(tmp_path: Path) -> None:
    """An engine's arguments (a ``build.json`` under ``serve_args``) may not change the suite's layout."""
    with pytest.raises(ValueError, match="set tp=2 instead"):
        _suite(tmp_path, tp=2).engine_kwargs("m", ServeSettings(args={"tensor_parallel_size": 3}))


def test_serve_args_may_not_set_the_draft(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="drop speculative_config"):
        _suite(tmp_path, serve_args={"speculative_config": {"method": "mtp"}})


def test_one_draft_serves_both_sides_when_no_other_is_named(tmp_path: Path) -> None:
    """Backward compatibility: one draft for the original and the checkpoint, whichever form it takes."""
    mtp = _suite(tmp_path)
    assert mtp.draft(checkpoint=False) == mtp.draft(checkpoint=True) == "mtp"
    assert mtp.speculative_config() == mtp.speculative_config(checkpoint=True)
    external = _suite(tmp_path, method="z-lab/Qwen3.5-9B-DFlash")
    assert external.draft(checkpoint=True) == "z-lab/Qwen3.5-9B-DFlash"
    assert external.speculative_config() == {"model": "z-lab/Qwen3.5-9B-DFlash", "num_speculative_tokens": 1}


def test_the_checkpoint_may_be_measured_with_its_rotated_draft(tmp_path: Path) -> None:
    """A rotated build's draft is ``qda rotate-draft``'s output: the original keeps the unrotated one."""
    suite = _suite(
        tmp_path,
        method="z-lab/Qwen3.5-9B-DFlash",
        checkpoint_method="work/qwen3.5-9b-dflash-rot",
        num_speculative_tokens=2,
    )
    assert suite.draft(checkpoint=False) == "z-lab/Qwen3.5-9B-DFlash"
    assert suite.draft(checkpoint=True) == "work/qwen3.5-9b-dflash-rot"
    assert suite.engine_kwargs("model", ServeSettings())["speculative_config"] == {
        "model": "z-lab/Qwen3.5-9B-DFlash",
        "num_speculative_tokens": 2,
    }
    assert suite.engine_kwargs("checkpoint", ServeSettings(), checkpoint=True)["speculative_config"] == {
        "model": "work/qwen3.5-9b-dflash-rot",
        "num_speculative_tokens": 2,
    }


def test_the_rotated_draft_does_not_move_the_original_readings_identity(tmp_path: Path) -> None:
    """The reference is the original model's reading: another checkpoint draft is another job, not another entry."""
    plain = _suite(tmp_path, method="z-lab/Qwen3.5-9B-DFlash")
    rotated = _suite(tmp_path, method="z-lab/Qwen3.5-9B-DFlash", checkpoint_method="work/rot")
    assert plain.reference_dir("m", tmp_path) == rotated.reference_dir("m", tmp_path)


def test_the_detail_names_the_draft_each_side_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _suite(tmp_path, method="z-lab/DFlash", checkpoint_method="work/rot")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "generation_config.json").write_text("{}")
    work = tmp_path / "out"
    reading = {
        "acceptance_rate": 1.0,
        "mean_acceptance_length": 2.0,
        "num_drafts": 1,
        "num_draft_tokens": 1,
        "num_accepted_tokens": 1,
        "per_position_acceptance": [1.0],
    }
    stored = suite.reference_dir(str(ref), work) / REFERENCE
    stored.parent.mkdir(parents=True)
    stored.write_text(json.dumps({**reading, "sampling": {"temperature": 0.0}, "seconds": {}}))
    (stored.parent / DIGEST).write_text(hashlib.sha256(stored.read_bytes()).hexdigest() + "\n")
    monkeypatch.setattr(
        SpecDecSuite,
        "_measure",
        lambda self, model, settings, rows, sampling, **kw: {
            **reading,
            "seconds": {},
            "serve": {},
            "vllm": "0",
            "versions": {},
        },
    )
    got = suite.run("checkpoint", str(ref), work)
    assert got.detail["method"] == "z-lab/DFlash"
    assert got.detail["checkpoint_method"] == "work/rot"
