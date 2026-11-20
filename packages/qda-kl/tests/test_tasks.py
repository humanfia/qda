"""The shipped lm-eval task files: their pins."""

from __future__ import annotations

from pathlib import Path

import yaml

TASKS = Path(__file__).parents[1] / "src/qda_kl/eval/tasks"


def _yaml(name: str) -> dict:
    """Load one shipped task file, leaving ``!function`` references as their dotted names."""

    class Loader(yaml.SafeLoader):
        pass

    yaml.add_constructor("!function", lambda loader, node: loader.construct_scalar(node), Loader=Loader)
    return yaml.load((TASKS / name).read_text(encoding="utf-8"), Loader=Loader)


MMLU_PRO_SUBJECTS = [
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def test_mmlu_pro_matches_the_harness_subject_prompts_with_the_dataset_pinned() -> None:
    """The harness's standard group: a subject-named instruction and that subject's five validation examples,
    with the dataset revision pinned (it replaced one flat prompt that had neither)."""
    group = _yaml("mmlu_pro.yaml")

    assert group["group"] == "mmlu_pro"
    assert group["task"] == [f"mmlu_pro_{subject}" for subject in MMLU_PRO_SUBJECTS]
    assert group["aggregate_metric_list"] == [
        {
            "aggregation": "mean",
            "metric": "exact_match",
            "weight_by_size": True,
            "filter_list": "custom-extract",
        }
    ]
    template = _yaml("_mmlu_pro_template.yaml")
    assert template["dataset_kwargs"]["revision"] == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert template["num_fewshot"] == 5
    assert template["fewshot_config"]["sampler"] == "first_n"
    for subject in MMLU_PRO_SUBJECTS:
        cfg = _yaml(f"mmlu_pro_{subject}.yaml")
        assert cfg["include"] == "_mmlu_pro_template.yaml"
        assert cfg["task"] == f"mmlu_pro_{subject}"
        assert "about " in cfg["description"], subject
        assert cfg["process_docs"] == f"lm_eval.tasks.mmlu_pro.utils.process_{subject}"
        assert "process_docs" not in template  # the filter is per subject, never shared
