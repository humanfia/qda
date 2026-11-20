"""The paired eval comparison: the bootstrap interval, the flips and the refusals."""

import json

import pytest
import yaml
from qda_kl.eval.report import _seeds, load, paired_main, paired_to_markdown
from qda_kl.eval.suites import SuiteResult

#: 8 questions x 2 seeds per question; the per-question differences are -1, 0.5, 0, 0.5, -0.5, -0.5, 0, -1.
BASE = {
    "q1": [1, 1],
    "q2": [1, 0],
    "q3": [1, 1],
    "q4": [0, 0],
    "q5": [1, 0],
    "q6": [0, 1],
    "q7": [0, 0],
    "q8": [1, 1],
}
ROW = {
    "q1": [0, 0],
    "q2": [1, 1],
    "q3": [1, 1],
    "q4": [1, 0],
    "q5": [0, 0],
    "q6": [0, 0],
    "q7": [0, 0],
    "q8": [0, 0],
}


def run(tmp_path, base, row, seed=17, row_seed=17, omit_seed=None, paired=None):
    """A run with bf16 (the baseline), ours, and a legacy result that has no per_question.

    Each result records the seed of its first sample and the task's repeats (2), as the lm_eval suite does;
    ``row_seed`` changes the seed ours records without touching the scores, ``omit_seed`` drops one result's
    seed record, and ``paired`` writes the run's paired-settings block.
    """
    out = tmp_path / "run"
    (out / "resolved.yaml").parent.mkdir(parents=True, exist_ok=True)
    config = {
        "reference": "R",
        "checkpoints": {"bf16": "/c/bf16", "ours": "/c/ours", "legacy": "/c/legacy"},
        "out": str(out),
        "suites": [{"lm_eval": {"tasks": ["gpqa_diamond"]}}],
    }
    if paired is not None:
        config["paired"] = paired
    (out / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    task = {"bf16": base, "ours": row, "legacy": {"metric": "acc,none"}}
    score = {"bf16": 0.5625, "ours": 0.3125, "legacy": 0.5}
    for checkpoint, detail in task.items():
        hashes = {question: "a" * 64 for question in detail.get("per_question", {})}
        recorded = {"seed": row_seed if checkpoint == "ours" else seed} if checkpoint != omit_seed else {}
        (out / "results" / checkpoint).mkdir(parents=True)
        (out / "results" / checkpoint / "lm_eval.gpqa_diamond.json").write_text(
            json.dumps(
                {
                    "suite": "lm_eval",
                    "checkpoint": checkpoint,
                    "metrics": {"gpqa_diamond": score[checkpoint]},
                    "higher_is_better": {"gpqa_diamond": True},
                    "detail": {
                        **recorded,
                        "gpqa_diamond": {
                            "cap_hits": [1, 0],
                            "repeats": 2,
                            "prompt_hashes": hashes,
                            **detail,
                        },
                    },
                    "wall_s": 1.0,
                    "config_digest": "0" * 64,
                }
            )
        )
    return load(out)


def test_paired_means_interval_flips_and_missing_per_question(tmp_path):
    """Mean, paired CI, both flip directions and cap hits; a result without per_question says why.

    8 questions x 2 seeds: the means are 9/16 and 5/16, the paired mean is -0.25, and the flips are 6
    right-to-wrong against 2 wrong-to-right of 16 pairs. The interval is the 2.5th and 97.5th percentile of
    1000 question resamples from seed 17, computed independently of this implementation.
    """
    expected = "\n".join(
        [
            "bootstrap 95% CI: 1000 resamples of the questions, RNG seed 17.",
            "",
            "### gpqa_diamond (8 questions x 2 seeds)",
            "",
            "| checkpoint | mean | delta vs bf16 [95% CI] | right->wrong | wrong->right | cap hits |",
            "| --- | --- | --- | --- | --- | --- |",
            "| bf16 | 56.2% | - | - | - | 1 |",
            "| ours | 31.2% | -25.0pp [-62.5, +12.5] | 37.5% (6/16) | 12.5% (2/16) | 1 |",
            "| legacy | 50.0% | - | - | - | 1 |",
            "",
            "note: legacy: no per_question in the result; paired columns are '-'.",
        ]
    )
    assert (
        paired_to_markdown(run(tmp_path, {"per_question": BASE}, {"per_question": ROW}), "bf16") == expected
    )


def test_paired_refuses_rows_that_do_not_share_questions_or_seeds(tmp_path):
    """Two rows are never intersected: a different question set or seed count is refused."""
    shorter = dict(list(ROW.items())[:-1])
    with pytest.raises(ValueError, match="do not share the same questions"):
        paired_to_markdown(run(tmp_path / "q", {"per_question": BASE}, {"per_question": shorter}), "bf16")

    three = {question: [*values, 0] for question, values in ROW.items()}
    with pytest.raises(ValueError, match="3 seeds, bf16 has 2"):
        paired_to_markdown(run(tmp_path / "s", {"per_question": BASE}, {"per_question": three}), "bf16")


def test_paired_refuses_different_seed_ids_and_non_finite_scores(tmp_path):
    """Recorded seed ids are compared, not just their number; a non-finite score is refused, never printed."""
    with pytest.raises(ValueError, match=r"has seeds \[18, 19\], bf16 has seeds \[17, 18\]"):
        paired_to_markdown(
            run(tmp_path / "ids", {"per_question": BASE}, {"per_question": ROW}, row_seed=18), "bf16"
        )

    nan = {question: list(values) for question, values in ROW.items()}
    nan["q2"][1] = float("nan")
    with pytest.raises(ValueError, match=r"per_question\['q2'\] holds a non-finite score"):
        paired_to_markdown(run(tmp_path / "nan", {"per_question": BASE}, {"per_question": nan}), "bf16")


def test_paired_cli_refuses_changed_prompt_before_printing_numbers(tmp_path, capsys):
    """Identical question and seed IDs do not establish pairing when a rendered prompt changed."""
    hashes = {question: "a" * 64 for question in ROW}
    hashes["q2"] = "b" * 64
    run(tmp_path, {"per_question": BASE}, {"per_question": ROW, "prompt_hashes": hashes})
    assert paired_main([str(tmp_path / "run"), "--baseline", "bf16"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err.startswith("error: gpqa_diamond: ours and bf16 have different prompt_hashes")
        and "question 'q2'" in captured.err
    )
    assert "rerun both with the same rendered prompts" in captured.err


@pytest.mark.parametrize("checkpoint", ["bf16", "ours"])
@pytest.mark.parametrize("hashes", [None, {}, {"q1": "a" * 64}])
def test_paired_missing_or_incomplete_hashes_leave_columns_empty_with_note(tmp_path, checkpoint, hashes):
    """Neither row can certify paired results without prompt provenance for every scored question."""
    base, row = {"per_question": BASE}, {"per_question": ROW}
    (base if checkpoint == "bf16" else row)["prompt_hashes"] = hashes
    text = paired_to_markdown(run(tmp_path, base, row), "bf16")
    assert "| ours | 31.2% | - | - | - | 1 |" in text
    assert f"note: {checkpoint}: no complete prompt_hashes in the result; paired columns are '-'." in text


@pytest.mark.parametrize("checkpoint", ["bf16", "ours"])
def test_paired_missing_seed_records_leave_columns_empty_with_note(tmp_path, checkpoint):
    """A recorded score count is not seed provenance: without seed ids no paired column is certified."""
    text = paired_to_markdown(
        run(tmp_path, {"per_question": BASE}, {"per_question": ROW}, omit_seed=checkpoint), "bf16"
    )
    assert "| ours | 31.2% | - | - | - | 1 |" in text
    assert f"note: {checkpoint}: does not record its seeds; paired columns are '-'." in text


def test_seed_ids_come_from_both_recorded_layouts_and_never_from_a_count():
    """The harness records the seed at the top level and repeats per task; LiveCodeBench repeats at the top.

    A missing or malformed record yields no seed ids, and repeats that disagree with the scores are refused.
    """
    scores = {"q1": [0.0, 1.0]}

    def result(detail):
        return SuiteResult(
            suite="livecodebench_v6",
            metrics={"livecodebench_v6": 0.5},
            higher_is_better={"livecodebench_v6": True},
            detail=detail,
        )

    harness = result({"seed": 17, "livecodebench_v6": {"per_question": scores, "repeats": 2}})
    livecodebench = result({"seed": 17, "repeats": 2, "livecodebench_v6": {"per_question": scores}})
    assert _seeds(harness, "livecodebench_v6", scores, "harness") == [17, 18]
    assert _seeds(livecodebench, "livecodebench_v6", scores, "livecodebench") == [17, 18]

    assert _seeds(result({"livecodebench_v6": {"repeats": 2}}), "livecodebench_v6", scores, "x") is None
    assert _seeds(result({"seed": 17}), "livecodebench_v6", scores, "x") is None
    assert (
        _seeds(result({"seed": True, "repeats": 2}), "livecodebench_v6", scores, "x") is None
    )  # bool is not a seed id
    assert _seeds(result({"seed": 17, "repeats": "2"}), "livecodebench_v6", scores, "x") is None
    with pytest.raises(ValueError, match="repeats=3 but per_question has 2 scores per question"):
        _seeds(result({"seed": 17, "repeats": 3}), "livecodebench_v6", scores, "x")


def test_paired_uses_the_recorded_bootstrap_settings_unless_flags_override_them(tmp_path, capsys):
    """The run's ``paired`` block reproduces a rendering; ``--resamples``/``--seed`` still win."""
    run(tmp_path, {"per_question": BASE}, {"per_question": ROW}, paired={"resamples": 23, "seed": 19})
    text = paired_to_markdown(load(tmp_path / "run"), "bf16")
    assert text.splitlines()[0] == "bootstrap 95% CI: 23 resamples of the questions, RNG seed 19."

    assert paired_main([str(tmp_path / "run"), "--baseline", "bf16"]) == 0
    assert "23 resamples of the questions, RNG seed 19." in capsys.readouterr().out

    assert paired_main([str(tmp_path / "run"), "--baseline", "bf16", "--resamples", "5", "--seed", "6"]) == 0
    assert "5 resamples of the questions, RNG seed 6." in capsys.readouterr().out


def test_paired_refuses_a_malformed_recorded_settings_block(tmp_path):
    """A resolved run that records nonsense is refused, naming the block and the value."""
    with pytest.raises(ValueError, match=r"resolved.yaml paired: .*greater than or equal to 1"):
        run(tmp_path, {"per_question": BASE}, {"per_question": ROW}, paired={"resamples": 0, "seed": 19})
