"""The sanity gate's review regressions: custom constraints, an interrupted reference record, a dropped MTP head."""

import json

import pytest
from qda_kl.eval import sanity
from qda_kl.eval.sanity import SanitySuite
from qda_kl.gencheck import Generation


def measure_answer(model, settings, prompts, sampling):
    """Return a compliant custom answer without loading an engine."""
    rows = [
        Generation(
            id=prompt.id,
            domain=prompt.domain,
            text="OK",
            token_ids=[1],
            token_count=1,
            finish_reason="stop",
        )
        for prompt in prompts
    ]
    specdec = {"acceptance_rate": 0.5} if "mtp" in model else None
    return rows, {
        "chat_template": True,
        "generation_config": True,
        "load_s": 0.0,
        "kernels": None,
        "specdec": specdec,
    }


def test_custom_instruction_is_scored_as_requested_or_refused_before_load(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "ok", "domain": "instruction", "prompt": "Reply with exactly OK."}) + "\n"
    )
    measured = []

    def measure(self, model, settings, prompts, sampling):
        measured.append(model)
        return measure_answer(model, settings, prompts, sampling)

    monkeypatch.setattr(SanitySuite, "_measure", measure)
    monkeypatch.setattr(SanitySuite, "_sampling", lambda self, model: {"temperature": 0.0})
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    monkeypatch.setattr(sanity, "versions", lambda: {})
    monkeypatch.setattr(sanity, "vllm_version", lambda: "test")
    try:
        suite = SanitySuite(prompts=str(prompts), per_domain=1, sampling={"temperature": 0.0})
        suite.prepare("original", tmp_path)
        result = suite.run("candidate", "original", tmp_path)
    except (ValueError, RuntimeError):
        assert measured == [], "unsupported custom constraints must be refused before loading"
    else:
        assert result.metrics["constraint_pass_rate"] == pytest.approx(1.0)


def test_a_kill_between_the_reference_writes_is_repaired(tmp_path, monkeypatch):
    """A record with answers but no metadata is rebuilt under the lock instead of skipped."""
    calls = []

    def measure(self, model, settings, prompts, sampling):
        calls.append(model)
        return measure_answer(model, settings, prompts, sampling)

    monkeypatch.setattr(SanitySuite, "_measure", measure)
    monkeypatch.setattr(SanitySuite, "_sampling", lambda self, model: {"temperature": 0.0})
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: 0)
    monkeypatch.setattr(sanity, "versions", lambda: {})
    monkeypatch.setattr(sanity, "vllm_version", lambda: "test")
    suite = SanitySuite(domains=["math"], per_domain=1)
    real_publish = sanity.publish

    def killed(path, text):
        if path.name == sanity.META:
            raise RuntimeError("killed before the metadata write")
        real_publish(path, text)

    monkeypatch.setattr(sanity, "publish", killed)
    with pytest.raises(RuntimeError, match="killed"):
        suite.prepare("original", tmp_path)
    out = suite.reference_dir("original", tmp_path)
    assert (out / sanity.ANSWERS).is_file() and not (out / sanity.META).is_file()

    monkeypatch.setattr(sanity, "publish", real_publish)
    suite.prepare("original", tmp_path)
    assert sanity._complete(out)
    assert calls == ["original", "original"], "the incomplete record must be rebuilt, not reused"
    result = suite.run("candidate", "original", tmp_path)
    assert result.metrics["chat_template"] == 1.0


def test_candidate_cannot_silently_drop_original_mtp_head(tmp_path, monkeypatch):
    suite = SanitySuite(domains=["math"], per_domain=1)
    monkeypatch.setattr(SanitySuite, "_measure", lambda self, *args: measure_answer(*args))
    monkeypatch.setattr(SanitySuite, "_sampling", lambda self, model: {"temperature": 0.0})
    monkeypatch.setattr(sanity, "_mtp_layers", lambda model: int("mtp" in model))
    monkeypatch.setattr(sanity, "versions", lambda: {})
    monkeypatch.setattr(sanity, "vllm_version", lambda: "test")
    suite.prepare("original-mtp", tmp_path)
    result = suite.run("candidate", "original-mtp", tmp_path)
    assert result.metrics["mtp_acceptance_rate"] == 0.0
    assert result.metrics["mtp_acceptance_rate_ratio"] == 0.0
    assert result.detail["specdec"]["checkpoint"] is None
    assert result.detail["specdec"]["missing"].startswith("the original has an MTP head")
