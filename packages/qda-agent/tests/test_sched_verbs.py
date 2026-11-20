"""`qsched show / eta / status / usage / wait` on a fake clock and fake cards."""

from __future__ import annotations

import pytest
from qda_agent.sched import FakeRun
from qda_agent.sched.report import progress
from test_judge import Rig


class _ByName(dict):
    """Fake runs keyed by job name (the backend looks them up by job id, `<seq>-<name>`)."""

    def get(self, key, default=None):
        return super().get(key.split("-", 1)[1], default)


def _rig(tmp_path, runs=None):
    r = Rig(tmp_path)
    r.be.runs = _ByName(runs or {})
    (r.work / "s.sh").write_text("echo 1\n")
    return r


def _add(r, name, *opts):
    rep = r.call("qsched", "add", "--expected", "10m", "--name", name, *opts, "s.sh")
    assert rep.exit == 0, rep.stderr


@pytest.mark.parametrize(
    ("log", "layers", "want"),
    [
        ("start\nlayer 3/28 done\n", None, (3 / 28, "3/28")),
        ("x 1/4\n 50%|#####     | 2/4 [00:01<00:01]\n", None, (0.5, "2/4")),
        ("layer 6 done\nlayer 7 done\n", 32, (8 / 32, "layer 8/32")),
        ("layer 7 done\n", None, None),
        ("0/10\n", None, None),
    ],
)
def test_progress_reads_the_last_progress_line(log, layers, want):
    assert progress(log, layers) == want


def test_show_and_eta_from_progress_then_expected(tmp_path):
    r = _rig(tmp_path, {"a": FakeRun(600, output="layer 1/4 done\n"), "b": FakeRun(600)})
    _add(r, "a")
    _add(r, "b", "--after", "a")
    r.host.tick()
    r.clock.advance(120)
    show = r.call("qsched", "show", "a").stdout
    assert "state      running" in show and "ran        2.0 min" in show and "after      -" in show
    eta = r.call("qsched", "eta", "a").stdout
    assert "a running: ends" in eta and "(in 6.0 min)" in eta and "progress 1/4" in eta
    assert "waiting: after a" in r.call("qsched", "eta", "b").stdout
    assert "after      a" in r.call("qsched", "show", "b").stdout
    r.clock.advance(600)
    r.host.tick()
    assert "(exit 0)" in r.call("qsched", "show", "a").stdout
    assert "a done: ended" in r.call("qsched", "eta", "a").stdout
    r.host.tick()
    r.clock.advance(60)
    assert "expected duration" in r.call("qsched", "eta", "b").stdout  # no progress line: --expected
    assert r.call("qsched", "show", "nope").exit == 2
    r.call("kl", "nope")
    assert r.call("qsched", "show", "judge-k1").exit == 2  # the judge's jobs are not the agent's


def test_status_cards_queue_and_idle_flag(tmp_path):
    r = _rig(tmp_path, {"big": FakeRun(600)})
    _add(r, "big", "--cards", "2")
    _add(r, "next", "--after", "big")
    r.host.tick()
    r.clock.advance(90)
    st = r.call("qsched", "status").stdout
    assert "card 0: big, running 1.5 min" in st and "card 1: big, running 1.5 min" in st
    assert "0 ready for cards, 1 waiting on --after" in st and "GPU deadline in" in st
    r.call("qsched", "cancel", "big")
    r.host.tick()  # `next` fails (its --after was cancelled); the cards are idle
    st = r.call("qsched", "status").stdout
    assert "card 0: idle 0.0 min (last: big cancelled)" in st and "IDLE!" not in st
    _add(r, "c")  # queued, not yet ticked: a free card and a job that could use it
    assert "IDLE! a queued job could use it" in r.call("qsched", "status").stdout


def test_usage_counts_each_card_second_once(tmp_path):
    r = _rig(tmp_path, {"a": FakeRun(1800), "b": FakeRun(900)})
    _add(r, "a", "--expected", "40m")
    _add(r, "b")
    for _ in range(4):  # a job's end is seen at the next pass: one every 15 min here
        r.host.tick()
        r.clock.advance(900)
    out = r.call("qsched", "usage").stdout
    vals = {line[:14].strip(): float(line[14:].split()[0]) for line in out.splitlines()}
    assert vals["allocated"] == pytest.approx(2.0)  # 2 cards x 1 h
    assert vals["agent other"] == pytest.approx(0.75)  # 0.5 h + 0.25 h
    assert vals["idle"] == pytest.approx(1.25)


def test_wait_for_jobs_any_and_timeout(tmp_path):
    r = _rig(tmp_path, {"a": FakeRun(60), "bad": FakeRun(30, rc=3), "long": FakeRun(3000)})
    _add(r, "a")
    _add(r, "bad")
    rep = r.call("qsched", "wait", "--job", "a")
    assert rep.exit == 0 and rep.stdout.startswith("MET a done at ")
    rep = r.call("qsched", "wait", "--job", "a", "--job", "bad")
    assert rep.exit == 1 and rep.stdout.startswith("FAILED bad failed")
    _add(r, "long", "--expected", "55m")
    t0 = r.clock.now()
    rep = r.call("qsched", "wait", "--job", "long", "--timeout", "5m")
    assert rep.exit == 124 and rep.stdout.startswith("TIMEOUT after 300 s")
    assert r.clock.now() - t0 == pytest.approx(300)
    rep = r.call("qsched", "wait", "--job", "long", "--timeout", "1h")  # capped at 590 s
    assert rep.exit == 124 and "after 590 s" in rep.stdout
    _add(r, "quick")
    rep = r.call("qsched", "wait", "--any")
    assert rep.exit == 0 and rep.stdout.startswith("MET quick done")
    r.call("qsched", "cancel", "long")
    rep = r.call("qsched", "wait", "--any")
    assert rep.exit == 1 and rep.stdout.startswith("FAILED nothing queued or running")
    assert r.call("qsched", "wait").exit == 2
    assert r.call("qsched", "wait", "--any", "--job", "a").exit == 2
    assert r.call("qsched", "wait", "--job", "nope").exit == 2


def test_wait_takes_no_file_condition(tmp_path):
    r = _rig(tmp_path)
    (r.work / "link").symlink_to(tmp_path)  # the host must never read what an agent path points to
    for argv in (["--file", "link"], ["--file", "link", "--grep", "x"], ["--any", "--fail-grep", "x"]):
        rep = r.call("qsched", "wait", *argv)
        assert rep.exit == 2 and "unrecognized arguments" in rep.stderr
