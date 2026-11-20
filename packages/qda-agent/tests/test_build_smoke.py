"""The host-start build smoke: a run opens only when its trusted build verb can mint a receipt (CPU)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from qda_agent.judge import build_receipt, startup


def _bwrap_starts() -> bool:
    """Whether bubblewrap starts a sandbox here: installed, and allowed a user namespace (Ubuntu 24.04's AppArmor
    default refuses one: `setting up uid map: Permission denied`). These tests run the trusted build in it."""
    if (bwrap := shutil.which("bwrap")) is None:
        return False
    argv = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--", "true"]
    return subprocess.run(argv, capture_output=True, timeout=10, check=False).returncode == 0


needs_bwrap = pytest.mark.skipif(not _bwrap_starts(), reason="bubblewrap cannot start a sandbox here")


def sources() -> list[Path]:
    """The three installed packages' source roots, as a real run's `qda_env.PYTHONPATH` names them."""
    from qda_agent import __file__ as agent_file
    from qda_kl import __file__ as kl_file
    from qda_quant import __file__ as quant_file

    return [Path(p).parents[1] for p in (quant_file, kl_file, agent_file)]


def rig(tmp_path, qda_cmd, readable):
    """A judge host that requires build receipts and names `qda_cmd` as its build verb.

    These tests cover the build half only: the host has no candidate 0 and no model, so the reading half and
    the candidate-0 build are turned off explicitly (a run that leaves it on and configures no baseline is refused; see `test_reading_smoke`).
    """
    from test_judge import Rig

    return Rig(
        tmp_path,
        readable=readable,
        require_build_receipt=True,
        build_receipt_smoke=True,
        candidate0_smoke=False,
        reading_smoke=False,
        qda_cmd=qda_cmd,
        qda_env={"PYTHONPATH": os.pathsep.join(map(str, readable))},
    )


@needs_bwrap
def test_a_healthy_build_verb_opens_the_run_and_leaves_no_trace(tmp_path, monkeypatch):
    """`start` runs one real tiny build; it leaves nothing in the run's custody, ledger or jobs."""
    made: list[Path] = []
    mkdtemp = tempfile.mkdtemp

    def recording(*args, **kwargs):
        path = mkdtemp(*args, **kwargs)
        made.append(Path(path))
        return path

    monkeypatch.setattr(startup.tempfile, "mkdtemp", recording)
    launch = build_receipt.launch
    seen: dict = {}

    def capture(sched, private, spec, **kwargs):
        seen.update(spec)
        return launch(sched, private, spec, **kwargs)

    monkeypatch.setattr(build_receipt, "launch", capture)
    paths = sources()
    r = rig(tmp_path, [sys.executable, "-m", "qda_quant.cli"], paths)
    r.host.start()
    try:
        assert made and not any(p.exists() for p in made), "the smoke directory is removed"
        assert list((r.host.jdir / "builds").iterdir()) == [], "no private build dir is left behind"
        assert r.host.sched.jobs == {} and r.host.entries == [], "the smoke is no job and no ledger entry"
        assert not list(r.run.rglob("receipt.json")), "no receipt lands in the run"
        # exactly the configured build python and PYTHONPATH, as the job's verb would get them
        assert seen["interpreter"] == str(Path(sys.executable).parent.resolve() / Path(sys.executable).name)
        assert seen["paths"] == [str(p.resolve()) for p in paths]
    finally:
        r.host.stop()


@needs_bwrap
def test_a_build_python_missing_a_module_is_refused_and_the_module_is_named(tmp_path):
    """A build interpreter that cannot import the build path's modules refuses the start, naming it."""
    r = rig(tmp_path, ["/usr/bin/python3", "-m", "qda_quant.cli"], sources())
    with pytest.raises(ValueError) as refused:
        r.host.start()
    message = str(refused.value)
    assert re.search(r"No module named '[^']+'", message), message
    assert "/usr/bin/python3" in message and "w4a16" in message
    assert r.host.server is None
    assert list((r.host.jdir / "builds").iterdir()) == []


@needs_bwrap
def test_a_build_verb_that_returns_no_receipt_is_refused(tmp_path, monkeypatch):
    """An rc-0 verb that writes no receipted build record does not open the run."""
    monkeypatch.setattr(build_receipt, "launch", lambda *args, **kwargs: [sys.executable, "-c", "pass"])
    r = rig(tmp_path, [sys.executable, "-m", "qda_quant.cli"], sources())
    with pytest.raises(ValueError, match="came back without receipt.json, build.json"):
        r.host.start()
    assert r.host.server is None


@needs_bwrap
@pytest.mark.parametrize("remote, full_card", [(True, False), (False, False), (False, True)])
def test_the_runs_own_model_is_built_and_admitted_before_the_start(tmp_path, remote, full_card):
    """Candidate 0 is the run's model, built as a queued trusted build and admitted by the hand-in gate.

    An export the judge refuses (a `config.json` with `auto_map`, as NemotronH's was) refuses the start with the
    judge's reason; the same build without it opens the run. Either way nothing it made stays.
    """
    import json

    from qda_agent.judge import intake as candidate
    from test_judge import Rig, ckpt

    original = ckpt(tmp_path / "original")
    r = Rig(
        tmp_path,
        model=str(original),
        format="w4a16-asym-g128",
        require_build_receipt=True,
        build_receipt_smoke=False,
        reading_smoke=False,
        qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
    )
    r.host.qda_argv = [sys.executable, "-m", "qda_quant.cli"]
    r.host.cfg.full_card_moe = full_card
    if full_card:
        r.host.sched.cfg.cards = list(range(8))
    admitted: list = []
    gate = candidate.admission_problem

    def spy(host, d, **kwargs):
        admitted.append((d, kwargs))
        return gate(host, d, **kwargs)

    def build(argv):  # the trusted runner's outcome: the build's tree, receipted and published
        job = r.host.sched.get("candidate0-smoke", "judge")
        assert job.ncards == (8 if full_card else 1)
        private = Path(job.build_private or "")
        if (private / "published").exists():
            return
        output = ckpt(private / "output")
        if remote:
            config = json.loads((output / "config.json").read_text())
            (output / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "m.M"}}))
        (output / "build.json").write_text("{}")
        build_receipt.capture(
            private, json.loads((private / "request.json").read_text()) | {"job_id": job.id}
        )
        build_receipt.publish(private)

    r.be.default.effect = build
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(candidate, "admission_problem", spy)
        try:
            if remote:
                with pytest.raises(ValueError, match=r"the judge refuses candidate 0 .*auto_map"):
                    r.host.start()
            else:
                r.host.start()
        finally:
            r.host.stop()
    # the hand-in's own gate on the published tree, without the budget (it judges an allocation)
    assert [(d.name, kwargs) for d, kwargs in admitted] == [("published", {"budget": False})], admitted
    assert not list(r.work.glob(".qda-candidate0*")), "the request and output leave the workdir"
    assert list((r.host.jdir / "builds").iterdir()) == [], "no private build dir is left behind"
    assert r.host.entries == [], "candidate 0's build is no ledger entry"
