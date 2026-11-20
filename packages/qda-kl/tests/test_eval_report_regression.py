"""The easy-group guard includes labelled teacher-forced web text."""

import pytest
import yaml
from qda_kl.eval.report import load, table_to_markdown
from qda_kl.eval.suites import SuiteResult


def test_easy_text_contributes_to_weighted_total_and_guard(tmp_path):
    config = {
        "checkpoints": {"ours": "ours", "base": "base"},
        "suites": [{"kl": {}}],
        "report": {"weights": {"hard": 0.2, "easy": 0.8}, "guard_threshold": 0.3, "resamples": 5},
    }
    (tmp_path / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for checkpoint, easy_kl in (("ours", 0.8), ("base", 0.4)):
        result = SuiteResult(
            suite="kl",
            checkpoint=checkpoint,
            metrics={"kl": 0.1, "kl_text": easy_kl},
            higher_is_better={"kl": False, "kl_text": False},
            percent={"kl": False, "kl_text": False},
            detail={
                "per_seq": {
                    "answer": {
                        "kl": 0.1,
                        "tokens": 1,
                        "domain": "math",
                        "group": "hard",
                        "split": "eval",
                        "kind": "answer",
                    },
                    "web": {
                        "kl": easy_kl,
                        "tokens": 9,
                        "domain": "web",
                        "group": "easy",
                        "split": "holdout",
                        "kind": "text",
                    },
                }
            },
        )
        directory = tmp_path / "results" / checkpoint
        directory.mkdir(parents=True)
        (directory / "kl.json").write_text(result.model_dump_json())

    table = load(tmp_path)
    ours = table.results[0]
    assert ours.metrics["kl"] == pytest.approx(0.1)
    assert ours.metrics["kl/total"] == pytest.approx(0.66)
    assert ours.metrics["kl/easy"] == pytest.approx(0.8)
    assert ours.metrics["kl/holdout"] == pytest.approx(0.8)
    assert table_to_markdown(table, "base").splitlines()[-2].endswith("| FAIL |")
