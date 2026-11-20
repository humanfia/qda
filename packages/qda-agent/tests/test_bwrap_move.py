"""The sandbox's bwrap code moved to `isolation.bwrap`; every old import path still resolves to it.

The move itself is proved by identity: `qda_agent.sched.sandbox` is a deprecation shim that forwards every
name (and every module-global write) to `qda_agent.isolation.bwrap`, so no caller has to change yet and no
copy of the code can drift.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from qda_agent.isolation import allocation, bwrap, turn_guard


def test_every_name_of_the_new_module_resolves_through_the_old_path():
    """Each module-level name of `isolation.bwrap` is the very object the old path hands out."""
    from qda_agent.sched import sandbox

    names = [name for name in vars(bwrap) if not name.startswith("__")]
    assert names  # the module is not empty
    for name in names:
        assert getattr(sandbox, name) is getattr(bwrap, name), name


def test_importing_the_old_path_warns_but_works():
    """The old module imports, with a DeprecationWarning, and still answers the names callers import."""
    from qda_agent.sched import sandbox

    with pytest.warns(DeprecationWarning, match="isolation.bwrap"):
        importlib.reload(sandbox)
    for name in ("bwrap_argv", "Pinned", "pin_sources", "python_roots", "pid_form", "sandbox_env"):
        assert getattr(sandbox, name) is getattr(bwrap, name)


def test_the_old_path_keeps_its_module_globals_one(tmp_path, monkeypatch):
    """A write on the old module changes the new module's global, so a probe/on_skip override still applies."""
    from qda_agent.sched import sandbox

    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))
    assert bwrap._ns == ("fresh", "private")
    monkeypatch.setattr(sandbox, "on_skip", lambda path, why: None)
    assert bwrap.on_skip is not None


def test_the_late_binder_runs_the_moved_file(tmp_path):
    """`private_bwrap_argv` still names the stdlib-only file it runs under `python -I`, now in isolation."""
    argv = bwrap.private_bwrap_argv(["bwrap", "--clearenv", "--", "true"], tmp_path)
    assert Path(argv[2]).name == "bwrap.py"
    assert Path(argv[2]).parent.name == "isolation"
    assert Path(argv[2]).is_file()


def test_the_two_hidden_checkers_share_one_scan():
    """The job's guard and the wall's self-check are the same scan with two endings, not two policies."""
    assert bwrap._HIDDEN_SCAN in bwrap._HIDDEN_GUARD
    assert bwrap._HIDDEN_SCAN in bwrap._HIDDEN_CHECK
    assert bwrap._HIDDEN_GUARD != bwrap._HIDDEN_CHECK


def test_the_two_proc_walkers_are_one_implementation():
    """The allocation check imports the turn guard's `/proc` scan and descendant walk; it has no copy."""
    assert allocation.rows is turn_guard.rows
    assert allocation.descendants is turn_guard.descendants


def test_the_old_path_reads_back_a_probe_cache_update(monkeypatch):
    """A global the implementation advances itself (the pid-form probe cache) is current on the old path."""
    from qda_agent.sched import sandbox

    monkeypatch.setattr(sandbox, "_ns", None)
    monkeypatch.setattr(sandbox, "_probe_ns", lambda: ("fresh", "private"))
    assert sandbox.proc_form() == "fresh"
    assert sandbox._ns == ("fresh", "private")


def test_the_shared_walker_keeps_status_only_parent_links(tmp_path):
    """A process whose `status` is readable and `stat` is not keeps its parent link (the union rule)."""
    entry = tmp_path / "123"
    entry.mkdir()
    (entry / "status").write_text("Name:\tworker\nPPid:\t42\n")
    assert allocation._parents(tmp_path) == {123: 42}


def test_a_new_name_added_through_the_old_path_leaves_no_trace():
    """A monkeypatch add on the old path is removed from this module's dict and from the implementation.

    The module copies each name into its own `__dict__` so `pytest.monkeypatch` can restore what it found;
    deleting a name it added must remove both copies, or a later patch's undo would resurrect the old one.
    """
    from qda_agent.sched import sandbox

    mp = pytest.MonkeyPatch()
    mp.setattr(sandbox, "ZZ_PROBE", "first", raising=False)
    assert sandbox.ZZ_PROBE == "first" and bwrap.ZZ_PROBE == "first"
    mp.undo()
    assert "ZZ_PROBE" not in vars(sandbox)
    assert not hasattr(bwrap, "ZZ_PROBE")
    mp.setattr(sandbox, "ZZ_PROBE", "second", raising=False)  # a second patch of the name works as the first
    assert sandbox.ZZ_PROBE == "second"
    mp.undo()
    assert "ZZ_PROBE" not in vars(sandbox)
    assert not hasattr(bwrap, "ZZ_PROBE")
