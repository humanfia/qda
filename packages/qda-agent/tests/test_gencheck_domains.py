"""`flow.check_gencheck_domains`: the long-generation set must cover the four evaluated domains."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qda_agent.config import RunConfig
from qda_agent.flow import GencheckDomainsRefused, check_gencheck_domains


def _cfg(tmp_path: Path, prompts: Path | None, *, gencheck_tool: bool = True) -> RunConfig:
    """A run whose long-generation set is `prompts` (None: the one qda-kl ships)."""
    judge: dict[str, object] = {} if prompts is None else {"gencheck_prompts": str(prompts)}
    components = ["qda", "bpw", "kl", "qsched", "handin"] if not gencheck_tool else None
    toolset = {} if components is None else {"name": "qda", "components": components}
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        judge=judge,  # type: ignore[arg-type]
        toolset=toolset,  # type: ignore[arg-type]
    )


def _prompts(tmp_path: Path, domains: list[str]) -> Path:
    """A prompt set whose rows name `domains`, in order."""
    path = tmp_path / "prompts.jsonl"
    rows = [{"id": f"p{i}", "domain": domain, "prompt": f"question {i}"} for i, domain in enumerate(domains)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_the_shipped_set_covers_all_four(tmp_path):
    """No `gencheck_prompts`: qda-kl's shipped set has maths, code, science and instruction following."""
    check_gencheck_domains(_cfg(tmp_path, None))


def test_a_set_with_all_four_passes(tmp_path):
    prompts = _prompts(tmp_path, ["math", "code", "science", "instruction"])
    check_gencheck_domains(_cfg(tmp_path, prompts))


def test_a_set_missing_one_domain_is_refused_by_name(tmp_path):
    prompts = _prompts(tmp_path, ["math", "code", "science"])
    with pytest.raises(GencheckDomainsRefused, match="has no prompt for instruction following") as error:
        check_gencheck_domains(_cfg(tmp_path, prompts))
    assert "has no prompt for maths" not in str(error.value)  # only what is missing is named
    assert str(prompts) in str(error.value)  # the message names the set


def test_the_r1_if_only_shape_is_refused(tmp_path):
    """The 1113 R1 row's set held instruction-following prompts only: the other three were never read."""
    prompts = _prompts(tmp_path, ["instruction"] * 4)
    with pytest.raises(GencheckDomainsRefused) as error:
        check_gencheck_domains(_cfg(tmp_path, prompts))
    message = str(error.value)
    assert "maths, code, science" in message


def test_a_run_without_the_gencheck_tool_is_not_refused(tmp_path):
    """No long-generation command in the toolset means the set is never read, so an IF-only set passes."""
    prompts = _prompts(tmp_path, ["instruction"] * 4)
    check_gencheck_domains(_cfg(tmp_path, prompts, gencheck_tool=False))


def test_a_set_that_cannot_be_made_available_is_refused(tmp_path, monkeypatch):
    """A dataset that refuses to hand over its file (a bad revision, no cache) refuses the run, not a traceback."""
    from qda_kl.datasets import GencheckBuiltin
    from qda_quant.step import Refused

    def refuse(self):
        raise Refused("no cached snapshot")

    monkeypatch.setattr(GencheckBuiltin, "local", refuse)
    with pytest.raises(GencheckDomainsRefused, match="cannot read the prompt set: no cached snapshot"):
        check_gencheck_domains(_cfg(tmp_path, None))


def test_a_set_that_cannot_be_read_is_refused(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{not json\n")
    with pytest.raises(GencheckDomainsRefused, match="cannot read"):
        check_gencheck_domains(_cfg(tmp_path, prompts))
