"""The job sandbox's bubblewrap floor: doctor and a run's own preflight say the same one line.

The stand-in bwrap is as old as the RHEL8 one: it prints a version, a `--help` without the descriptor
options, and "Unknown option" when one is passed. Support is asked of the binary and not of the version
number, because a distribution may backport it (Ubuntu's 0.9.0-1ubuntu0.1 has `--bind-fd`).
"""

import os

import pytest
from qda_agent import doctor as d
from qda_agent.isolation import bwrap
from qda_agent.livecodebench import LiveCodeBenchSuite
from qda_agent.sched import SchedConfig


@pytest.fixture
def old_bwrap(tmp_path, monkeypatch):
    """PATH's bwrap: bubblewrap 0.6.2, no `--bind-fd`/`--ro-bind-fd`."""
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "bwrap"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  --version) echo 'bubblewrap 0.6.2'; exit 0;;\n"
        "  --help) echo 'usage: bwrap [OPTION...]'; echo '    --ro-bind SRC DEST'; exit 0;;\n"
        "esac\n"
        'case " $* " in\n'
        '  *" --ro-bind-fd "*|*" --bind-fd "*) echo "bwrap: Unknown option --ro-bind-fd" >&2; exit 1;;\n'
        "esac\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    # the module caches what one executable answered, by name: give this test its own caches
    monkeypatch.setattr(bwrap, "_supports", {})
    monkeypatch.setattr(bwrap, "_versions", {})
    return script


def test_support_is_asked_of_the_binary_not_of_the_version(old_bwrap):
    """A 0.6.2 without the option is refused, and the one line names the version and the way out."""
    assert bwrap.supports_bind_fd() is False
    with pytest.raises(bwrap.BwrapTooOld) as got:
        bwrap.require_bind_fd()
    line = str(got.value)
    assert "bubblewrap 0.6.2 is too old for the job sandbox" in line
    assert "isolation.level: none" in line and "gpus: false" not in line


def test_a_backported_option_is_not_refused():
    """Ubuntu's 0.9.0-1ubuntu0.1 lacks the version but has the option: it stays accepted."""
    bwrap._supports.clear()  # the real bwrap, not a stand-in's cached answer
    if not bwrap.supports_bind_fd():
        pytest.skip("this machine's bwrap has no --bind-fd")
    bwrap.require_bind_fd()


def test_a_runs_own_preflight_refuses_an_old_bwrap(old_bwrap, tmp_path):
    """`LegacyIsolation.preflight` calls this: the run refuses before its clock starts, in one sentence."""
    weights = tmp_path / "snapshot"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w" * 8)
    with pytest.raises(bwrap.BwrapTooOld, match="too old for the job sandbox"):
        bwrap.preflight([weights], writable=[tmp_path])


def test_the_doctor_reports_the_one_line(old_bwrap):
    """The sandbox line carries the diagnosis and the fix, never `scheduler.gpus: false`."""
    c = d.check_bwrap(SchedConfig(cards=[0]), level="sandbox")
    assert not c.ok and not c.skipped
    assert "bubblewrap 0.6.2 is too old for the job sandbox" in c.detail
    assert "isolation.level: none" in c.fix and "gpus: false" not in c.fix


def test_a_none_run_checks_no_job_sandbox(old_bwrap, tmp_path):
    """The run is `isolation.level: none`: its jobs are plain, so doctor must not demand a bwrap it never uses."""
    cfg = SchedConfig(cards=[0], fake_gpus=True, run_dir=tmp_path / "run")
    none = d.check_bwrap(cfg, level="none")
    assert none.skipped and "isolation.level: none" in none.detail
    sandbox = d.check_bwrap(cfg, level="sandbox")
    assert not sandbox.ok and "too old" in sandbox.detail


def test_each_sandbox_names_its_own_way_out(old_bwrap, tmp_path):
    """A search's job sandbox names `scheduler.job_sandbox`; `qda eval`'s LiveCodeBench grader names its own knob.

    The grader's refusal reaches the user through the suite's preflight, before any download, and must not
    send them to a search key that `qda eval` never reads.
    """
    with pytest.raises(bwrap.BwrapTooOld) as job:
        bwrap.require_bind_fd()
    assert job.value.fix == bwrap.BWRAP_FIX
    assert "too old for the job sandbox" in str(job.value)
    assert "isolation.level: none" in str(job.value) and "--lcb-sandbox" not in str(job.value)

    with pytest.raises(bwrap.BwrapTooOld) as grader:
        LiveCodeBenchSuite(sandbox="auto").prepare("unused", tmp_path)
    line = str(grader.value)
    assert "bubblewrap 0.6.2 is too old for the LiveCodeBench grading sandbox" in line
    assert "--lcb-sandbox none (sandbox: none on the suite)" in line
    assert "scheduler.job_sandbox" not in line
