"""The eval result table: config order, missing cells, bolding and recovery."""

import json
import types
from pathlib import Path

import pytest
import yaml
from qda_kl.eval import cli, report
from qda_kl.eval.report import EvalTable, load, table_to_html, table_to_markdown, values_of
from qda_kl.eval.suites import SuiteResult
from qda_kl.tally import SeqKL


def result(checkpoint, suite, metrics, higher_is_better, detail=None, percent=None, error=None):
    return {
        "suite": suite,
        "checkpoint": checkpoint,
        "metrics": metrics,
        "higher_is_better": higher_is_better,
        "percent": percent or {},
        "detail": detail or {},
        "wall_s": 1.0,
        "config_digest": "0" * 64,
        "error": error,
    }


def write_run(out, checkpoints, suites, *, reference="R", report=None, dirs=None):
    """Leave the run directory as eval does: each checkpoint's results dir and the resolved config."""
    out = Path(out)
    for checkpoint in checkpoints if dirs is None else dirs:
        (out / "results" / checkpoint).mkdir(parents=True)
    config = {
        "reference": reference,
        "checkpoints": {name: f"/checkpoints/{name}" for name in checkpoints},
        "out": str(out),
        "suites": suites,
    }
    if report is not None:
        config["report"] = report
    (out / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def test_three_results_build_the_table(tmp_path):
    out = write_run(
        tmp_path / "run",
        ("ours", "redhat", "unfinished"),
        [
            {"kl": {"corpus": {"reasoning": {}}, "scorer": {"topk": {}}}},
            {"lm_eval": {"tasks": ["gpqa_diamond"]}},
        ],
        reference="Qwen/Qwen3.5-9B",
        dirs=("ours", "redhat"),
    )
    (out / "results" / "ours" / "kl.json").write_text(
        json.dumps(
            result("ours", "kl", {"kl": 0.009823}, {"kl": False}, {"stderr": {"kl": 0.000124}}, {"kl": False})
        )
    )
    (out / "results" / "ours" / "lm_eval.json").write_text(
        json.dumps(
            result(
                "ours",
                "lm_eval",
                {"gpqa_diamond": 0.612},
                {"gpqa_diamond": True},
                {"stderr": {"gpqa_diamond": 0.011}},
            )
        )
    )
    (out / "results" / "redhat" / "lm_eval.json").write_text(
        json.dumps(result("redhat", "lm_eval", {"gpqa_diamond": 0.655}, {"gpqa_diamond": True}))
    )

    expected = "\n".join(
        [
            "| checkpoint | kl | gpqa_diamond | recovery % |",
            "| --- | --- | --- | --- |",
            "| ours | **0.009823 ±0.000124** | 61.2% ±1.1% | 93.4% |",
            "| redhat | - | **65.5%** | **100.0%** |",
            "| unfinished | - | - | - |",
        ]
    )
    assert table_to_markdown(load(out), "redhat") == expected

    page = table_to_html(load(out), "redhat")
    assert '<table class="qda-eval">' in page
    assert "<strong>65.5%</strong>" in page
    assert page.count("<td>-</td>") == 4

    kl_only = EvalTable(
        ["ours"],
        [
            SuiteResult(
                suite="kl",
                checkpoint="ours",
                metrics={"kl": 0.009823},
                higher_is_better={"kl": False},
                percent={"kl": False},
            )
        ],
    )
    assert table_to_markdown(kl_only, "ours") == "| checkpoint | kl |\n| --- | --- |\n| ours | **0.009823** |"


def test_a_cached_result_keeps_its_own_unit_in_any_row_order():
    """An old cached KL result (no ``percent``) and a fresh one render each with its own unit, whichever is first."""
    old = SuiteResult(suite="kl", checkpoint="old", metrics={"kl": 0.01}, higher_is_better={"kl": False})
    new = SuiteResult(
        suite="kl",
        checkpoint="new",
        metrics={"kl": 0.008},
        higher_is_better={"kl": False},
        percent={"kl": False},
    )
    rows = ["| new | **0.008** |", "| old | 1.0% |"]
    for checkpoints, results in ((["old", "new"], [old, new]), (["new", "old"], [new, old])):
        text = table_to_markdown(EvalTable(checkpoints, results), None)
        assert text.splitlines()[2:] == (rows if checkpoints[0] == "new" else rows[::-1])


def test_a_suites_parts_share_its_columns(tmp_path):
    """A suite's parts are merged into its columns; a part a checkpoint has not run shows ``-``."""
    out = write_run(tmp_path / "run", ("ours", "redhat"), [{"bench": {"tasks": ["alpha", "beta"]}}])
    (out / "results" / "ours" / "bench.alpha.json").write_text(
        json.dumps(result("ours", "bench", {"alpha": 0.5}, {"alpha": True}, {"stderr": {"alpha": 0.01}}))
    )
    (out / "results" / "ours" / "bench.beta.json").write_text(
        json.dumps(result("ours", "bench", {"beta": 0.7}, {"beta": True}, {"stderr": {"beta": 0.02}}))
    )
    (out / "results" / "redhat" / "bench.alpha.json").write_text(
        json.dumps(result("redhat", "bench", {"alpha": 0.6}, {"alpha": True}))
    )

    expected = "\n".join(
        [
            "| checkpoint | alpha | beta |",
            "| --- | --- | --- |",
            "| ours | 50.0% ±1.0% | **70.0% ±2.0%** |",
            "| redhat | **60.0%** | - |",
        ]
    )
    assert table_to_markdown(load(out)) == expected


@pytest.mark.parametrize(
    "detail", [{"reference_sha256": value} for value in ("b" * 64, "", None, 123)] + [{}]
)
@pytest.mark.parametrize("parts", [False, True])
def test_results_from_different_references_are_refused(tmp_path, detail, parts):
    """One suite's checkpoints measured against different references are not comparable, so the table refuses."""
    out = write_run(tmp_path / "run", ("ours", "redhat"), [{"kl": {}}])
    (out / "results" / "ours" / "kl.json").write_text(
        json.dumps(
            result("ours", "kl", {"kl": 0.01}, {"kl": False}, {"reference_sha256": "a" * 64}, {"kl": False})
        )
    )
    second = out / "results" / ("ours" if parts else "redhat") / ("kl.text.json" if parts else "kl.json")
    second.write_text(
        json.dumps(
            result("ours" if parts else "redhat", "kl", {"kl": 0.02}, {"kl": False}, detail, {"kl": False})
        )
    )
    with pytest.raises(ValueError, match="shared `refs`") as refused:
        load(out)
    assert "ours" in str(refused.value)
    if not parts:
        assert "redhat" in str(refused.value)
    # One reference for both is a table.
    second.write_text(
        json.dumps(
            result("redhat", "kl", {"kl": 0.02}, {"kl": False}, {"reference_sha256": "a" * 64}, {"kl": False})
        )
    )
    assert "redhat" in table_to_markdown(load(out))


def test_a_failed_row_does_not_refuse_the_table(tmp_path):
    """A failed row measured nothing and has no reference, so it shows its error instead of failing the table."""
    out = write_run(tmp_path / "run", ("bad", "good"), [{"kl": {}}])
    (out / "results" / "good" / "kl.json").write_text(
        json.dumps(
            result("good", "kl", {"kl": 0.01}, {"kl": False}, {"reference_sha256": "a" * 64}, {"kl": False})
        )
    )
    (out / "results" / "bad" / "kl.json").write_text(
        json.dumps(result("bad", "kl", {}, {}, error="cannot load vendor checkpoint"))
    )
    markdown = table_to_markdown(load(out))
    assert "| good | - | **0.01** |" in markdown
    assert "| bad | cannot load vendor checkpoint | - |" in markdown


def test_a_partly_measured_row_is_checked_against_the_others_reference(tmp_path):
    """A merge keeps a failed part's error beside the good parts' metrics, so the row still records its
    reference and must be checked; a fully failed row measured nothing and is still skipped."""
    out = write_run(tmp_path / "run", ("partial", "failed", "other"), [{"kl": {}}])
    (out / "results" / "partial" / "kl.reasoning.json").write_text(
        json.dumps(
            result(
                "partial", "kl", {"kl": 0.01}, {"kl": False}, {"reference_sha256": "a" * 64}, {"kl": False}
            )
        )
    )
    (out / "results" / "partial" / "kl.chat.json").write_text(
        json.dumps(result("partial", "kl", {}, {}, error="cannot load calibration data"))
    )
    (out / "results" / "failed" / "kl.json").write_text(
        json.dumps(result("failed", "kl", {}, {}, error="cannot load vendor checkpoint"))
    )
    other = out / "results" / "other" / "kl.json"
    other.write_text(
        json.dumps(
            result("other", "kl", {"kl": 0.02}, {"kl": False}, {"reference_sha256": "b" * 64}, {"kl": False})
        )
    )

    with pytest.raises(ValueError, match="shared `refs`") as refused:
        load(out)
    assert "partial" in str(refused.value) and "other" in str(refused.value)
    assert "failed" not in str(refused.value)

    # One reference across the measured parts is a table; both failed rows keep their exact errors.
    other.write_text(
        json.dumps(
            result("other", "kl", {"kl": 0.02}, {"kl": False}, {"reference_sha256": "a" * 64}, {"kl": False})
        )
    )
    markdown = table_to_markdown(load(out))
    assert "| partial | cannot load calibration data | **0.01** |" in markdown
    assert "| failed | cannot load vendor checkpoint | - |" in markdown


def seq(kl, tokens, group, domain, split, kind="answer"):
    return {"kl": kl, "tokens": tokens, "group": group, "domain": domain, "split": split, "kind": kind}


def test_a_label_named_total_is_refused():
    """A domain called ``total`` would overwrite the weighted total column; refuse it instead."""
    seqs = [SeqKL(kl=0.1, tokens=10, group="hard", domain="total", split="eval")]
    with pytest.raises(ValueError, match="is named 'total', which the total column uses"):
        values_of(seqs, {"hard": 1.0})


def test_a_domain_and_split_with_one_name_are_refused():
    """A domain named like a split would lose its column to the split; refuse the ambiguity."""
    seqs = [SeqKL(kl=0.1, tokens=10, group="hard", domain="eval", split="eval")]
    with pytest.raises(ValueError, match=r"domain and split share the name\(s\) \['eval'\]"):
        values_of(seqs, {"hard": 1.0})


def test_sequence_records_give_group_split_total_cis_and_a_guard(tmp_path):
    """The kl suite's records become group/domain/split columns, a weighted total and a guard.

    The labelled web-text record joins the easy group and the web/eval columns, while the plain ``kl`` stays the
    suite's answer-only 0.011. ``ours`` ties the hard group (0.015) and loses the easy one (0.0063 vs 0.005),
    which the guard flags.
    """
    out = write_run(
        tmp_path / "run",
        ("ours", "base"),
        [{"kl": {"corpus": {"reasoning": {}}, "scorer": {"topk": {}}}}],
        report={"weights": {"hard": 0.9, "easy": 0.1}, "resamples": 100, "seed": 3},
    )
    ours = {
        "h1#0": seq(0.010, 100, "hard", "math", "eval"),
        "h2#0": seq(0.020, 100, "hard", "code", "eval"),
        "e1#0": seq(0.006, 100, "easy", "chat", "eval"),
        "e2#0": seq(0.008, 100, "easy", "chat", "holdout"),
        "t1": seq(0.005, 100, "easy", "web", "eval", kind="text"),
    }
    base = {k: v for k, v in ours.items() if k != "t1"}
    base["e1#0"] = seq(0.004, 100, "easy", "chat", "eval")
    base["e2#0"] = seq(0.006, 100, "easy", "chat", "holdout")
    for checkpoint, per in (("ours", ours), ("base", base)):
        (out / "results" / checkpoint / "kl.json").write_text(
            json.dumps(
                result(checkpoint, "kl", {"kl": 0.011}, {"kl": False}, {"per_seq": per}, {"kl": False})
            )
        )

    run = load(out)
    got = {r.checkpoint: r.metrics for r in run.results}
    assert got["ours"]["kl"] == 0.011  # the suite's answer-only metric is unchanged
    assert got["ours"]["kl/hard"] == pytest.approx(0.015)
    assert got["ours"]["kl/easy"] == pytest.approx(0.0063333333)  # includes the labelled text record
    assert got["ours"]["kl/total"] == pytest.approx(0.0141333333)
    assert got["ours"]["kl/eval"] == pytest.approx(0.01025)
    assert got["ours"]["kl/holdout"] == pytest.approx(0.008)
    assert {k for k in got["ours"] if k.startswith("kl/")} >= {"kl/hard", "kl/easy", "kl/total", "kl/chat"}

    ci = run.results[0].detail["ci"]
    assert ci["kl/total"][0] <= got["ours"]["kl/total"] <= ci["kl/total"][1]
    assert ci["kl/holdout"] == pytest.approx([0.008, 0.008])
    assert load(out).results[0].detail["ci"] == ci  # the seed makes the intervals repeat

    page = table_to_markdown(run, "base")
    assert (
        "| checkpoint | kl | kl/hard | kl/easy | kl/total | kl/chat | kl/code | kl/math | kl/web | kl/eval |"
        " kl/holdout | guard |" in page
    )
    assert "| ours |" in page and "| base |" in page
    assert page.splitlines()[-2].endswith("| FAIL |") and page.splitlines()[-1].endswith("| ok |")


def test_a_text_only_easy_group_reaches_the_total_and_the_guard(tmp_path):
    """An easy group of web-text records only still weighs into the total and still guards.

    The old answer-only filter dropped it, reporting the hard group's 0.1 as the total with no guard column.
    With the labelled text record the total is 0.2*0.1 + 0.8*0.8 = 0.66, and 0.8 exceeds the baseline's 0.4 by
    more than the 0.3 threshold.
    """
    out = write_run(
        tmp_path / "run",
        ("ours", "base"),
        [{"kl": {"corpus": {"reasoning": {}}, "scorer": {"topk": {}}}}],
        report={
            "weights": {"hard": 0.2, "easy": 0.8},
            "resamples": 20,
            "seed": 3,
            "guard_threshold": 0.3,
        },
    )
    ours = {"h": seq(0.1, 100, "hard", "math", "eval"), "w": seq(0.8, 100, "easy", "web", "eval", "text")}
    base = {"h": seq(0.4, 100, "hard", "math", "eval"), "w": seq(0.4, 100, "easy", "web", "eval", "text")}
    for checkpoint, per in (("ours", ours), ("base", base)):
        (out / "results" / checkpoint / "kl.json").write_text(
            json.dumps(
                result(
                    checkpoint, "kl", {"kl": per["h"]["kl"]}, {"kl": False}, {"per_seq": per}, {"kl": False}
                )
            )
        )

    got = {r.checkpoint: r.metrics for r in load(out).results}
    assert got["ours"]["kl"] == 0.1  # the suite's answer-only metric is unchanged
    assert got["ours"]["kl/easy"] == pytest.approx(0.8)
    assert got["ours"]["kl/total"] == pytest.approx(0.66)
    page = table_to_markdown(load(out), "base")
    assert next(line for line in page.splitlines() if line.startswith("| ours |")).endswith("| FAIL |")


def test_eval_reaches_the_table_and_still_takes_a_config(monkeypatch):
    """`qda eval table OUT` forwards to the table; `qda eval CONFIG.yaml` keeps its parser."""
    import argparse

    seen = []
    monkeypatch.setattr(
        report, "table_main", lambda out, *, fmt, baseline: seen.append((out, fmt, baseline)) or 0
    )
    parser = argparse.ArgumentParser(prog="qda")
    cli.add_parser(parser.add_subparsers(dest="cmd", required=True))

    assert parser.parse_args(["eval", "config.yaml"]).config == Path("config.yaml")
    assert (
        cli.main(parser.parse_args(["eval", "table", "run", "--format", "html", "--baseline", "ours"])) == 0
    )
    assert seen == [(Path("run"), "html", "ours")]


def test_server_belongs_to_the_config_and_reaches_the_run(monkeypatch, tmp_path):
    """`qda eval CONFIG --server URL` is the served path; it must not be refused as a table argument."""
    import argparse

    from qda_kl.eval import config as config_module
    from qda_kl.eval import run as run_module

    calls = []
    cfg_stub = types.SimpleNamespace(full_note="")
    monkeypatch.setattr(config_module.EvalConfig, "from_yaml", classmethod(lambda cls, text, sets: cfg_stub))
    monkeypatch.setattr(run_module, "jobs", lambda cfg, only: [])
    monkeypatch.setattr(
        run_module, "run", lambda cfg, only, *, server=None, log=print: calls.append(server) or 0
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text("checkpoints: {}\n")
    parser = argparse.ArgumentParser(prog="qda")
    cli.add_parser(parser.add_subparsers(dest="cmd", required=True))
    url = "http://127.0.0.1:8123/v1/completions"
    assert cli.main(parser.parse_args(["eval", str(cfg), "--server", url])) == 0
    assert calls == [url]


def test_table_refuses_server(monkeypatch):
    """A server URL on `qda eval table` is refused, not silently ignored."""
    import argparse

    parser = argparse.ArgumentParser(prog="qda")
    cli.add_parser(parser.add_subparsers(dest="cmd", required=True))
    assert cli.main(parser.parse_args(["eval", "table", "run", "--server", "http://x/v1/completions"])) == 2
