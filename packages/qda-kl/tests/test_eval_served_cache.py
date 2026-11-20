"""A served job and an in-process job of the same cell are different measurements, not a cache hit.

``server`` is excluded from the suite's settings dump (the URL is transport and changes every launch), so
without a marker the two jobs share a digest and a result path: ``qda eval CONFIG --server URL`` on a cell
already measured in-process reports it ``cached`` and generates nothing, and an in-process run after a served
one reuses the served numbers. The digest names the engine path instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.lm_eval import LmEvalSuite
from qda_kl.eval.run import jobs


def _cfg(tmp: Path) -> EvalConfig:
    for d in ("ref", "c1"):
        (tmp / d).mkdir(exist_ok=True)
    return EvalConfig.from_yaml(
        yaml.safe_dump(
            {
                "reference": str(tmp / "ref"),
                "checkpoints": {"one": str(tmp / "c1")},
                "out": str(tmp / "out"),
                "suites": [{"lm_eval": {"tasks": ["t1"], "sampling": {"temperature": 0.0}}}],
            }
        )
    )


def test_a_served_job_is_not_the_in_process_job(tmp_path: Path) -> None:
    """Either the digest or the result path tells the two engines apart; the digest does."""
    in_process = jobs(_cfg(tmp_path), ())
    cfg = _cfg(tmp_path)
    for suite in cfg.suites:
        if isinstance(suite, LmEvalSuite):
            suite.server = "http://127.0.0.1:8123/v1/completions"
    served = jobs(cfg, ())
    assert len(in_process) == len(served) == 1
    a, b = in_process[0], served[0]
    assert (a.digest, a.path) != (b.digest, b.path), (
        f"served and in-process jobs share digest {a.digest[:12]} and path {a.path}: a served run would report "
        "the in-process result as cached"
    )
    # The in-process digest is unchanged by the fix: results already on disk keep counting as done.
    assert jobs(_cfg(tmp_path), ())[0].digest == a.digest
