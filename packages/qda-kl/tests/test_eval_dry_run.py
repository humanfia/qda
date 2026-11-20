"""``qda eval --dry-run`` reports a finished job as cached, and a host-local workspace does not move a digest.

A ``--dry-run`` is a second process: it loads the config afresh, so it must resolve the same layout settings
(``serve_env``) the runner wrote into the result's digest. And the FlashInfer workspace defaults to ``$HOME``,
so a digest that kept it would call the same checkpoint uncached after a HOME change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.run import jobs, run
from qda_kl.eval.suites import SUITES, Layout, SuiteResult


class _LayoutStub(Layout):
    """A Layout suite with a local run, so the runner writes a real result file."""

    name: ClassVar[str] = "stub_layout"
    value: float = 1.0

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        return SuiteResult(suite=self.name, metrics={"m": self.value}, higher_is_better={"m": True})


if _LayoutStub.name not in SUITES:
    SUITES.register(_LayoutStub.name, _LayoutStub)


def _config_text(tmp: Path, **settings: Any) -> str:
    for name in ("ref", "c1"):
        (tmp / name).mkdir(exist_ok=True)
    return yaml.safe_dump(
        {
            "reference": str(tmp / "ref"),
            "checkpoints": {"one": str(tmp / "c1")},
            "out": str(tmp / "out"),
            "suites": [{"stub_layout": settings}],
        }
    )


def test_dry_run_reports_a_finished_job_cached(tmp_path: Path) -> None:
    """A run leaves a result whose digest includes the resolved serve_env; a fresh --dry-run must match it."""
    text = _config_text(tmp_path)
    assert run(EvalConfig.from_yaml(text), log=lambda _: None) == 0
    fresh = EvalConfig.from_yaml(text)  # what a separate `qda eval --dry-run` process loads
    assert [job.cached for job in jobs(fresh)] == [True]


def test_digest_does_not_follow_the_flashinfer_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workspace is where vLLM caches kernels, not what a job measures: HOME must not move the digest."""

    def fake_cache_env(env: dict[str, str] | None = None) -> dict[str, str]:
        return {**(env or {}), "FLASHINFER_WORKSPACE_BASE": os.environ["HOME"]}

    monkeypatch.setattr("qda_kl.serve.cache_env", fake_cache_env)
    text = _config_text(tmp_path)

    monkeypatch.setenv("HOME", str(tmp_path / "first"))
    first = EvalConfig.from_yaml(text)
    first_digests = [job.digest for job in jobs(first)]
    assert first.suites[0].serve_env["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "first")

    monkeypatch.setenv("HOME", str(tmp_path / "second"))
    second = EvalConfig.from_yaml(text)
    second_digests = [job.digest for job in jobs(second)]
    assert second.suites[0].serve_env["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "second")
    assert first_digests == second_digests

    # Neither does a serve_env that names another cache location (`CACHE_ENV`): it is where files land.
    moved = EvalConfig.from_yaml(
        _config_text(tmp_path, serve_env={"VLLM_CACHE_ROOT": "/a", "TRITON_CACHE_DIR": "/b"})
    )
    assert [job.digest for job in jobs(moved)] == second_digests

    # A setting that changes the measurement still moves the digest.
    assert [job.digest for job in jobs(EvalConfig.from_yaml(_config_text(tmp_path, value=2.0)))] != (
        first_digests
    )
