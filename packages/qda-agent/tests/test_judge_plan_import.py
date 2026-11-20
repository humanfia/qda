"""The judge's build planner: imported once, planned concurrently, and its errors kept in the judge's log."""

from __future__ import annotations

import builtins
import logging
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
from qda_agent.judge import build_receipt
from qda_agent.sched import Refused
from test_judge import Rig, llama_paths

PLAN = SimpleNamespace(output_bytes=1, output_bytes_exact=True, weight_bytes=1)


class FakeConfig:
    """`BuildConfig` stand-in: `plan()` blocks while `raw["block"]` (an Event) is unset."""

    out = None

    def __init__(self, raw: dict) -> None:
        self.raw = raw

    @classmethod
    def from_dict(cls, raw: dict, base: str) -> FakeConfig:  # noqa: ARG003 -- `BuildConfig.from_dict`'s signature
        return cls(raw)

    def plan(self) -> SimpleNamespace:
        if (gate := self.raw.get("block")) is not None:
            assert gate.wait(10)
        return PLAN


@pytest.fixture
def racy_import(monkeypatch) -> dict:
    """A first import of `qda_quant.config` that a second, concurrent import of it fails, as transformers' did."""
    state = {"in_flight": 0, "imports": 0}
    lock = threading.Lock()
    real = builtins.__import__
    module = ModuleType("qda_quant.config")
    module.BuildConfig = FakeConfig  # type: ignore[attr-defined]

    def importer(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001, ANN202
        if name != "qda_quant.config" or "BuildConfig" not in (fromlist or ()):
            return real(name, globals, locals, fromlist, level)
        with lock:
            if state["in_flight"]:
                raise ImportError("cannot import name 'AutoConfig' from 'transformers'")
            state["in_flight"] += 1
            state["imports"] += 1
        try:
            time.sleep(0.2)  # the slow first import of transformers
            return module
        finally:
            with lock:
                state["in_flight"] -= 1

    monkeypatch.setattr(build_receipt, "_BUILD_CONFIG", None)
    monkeypatch.setattr(build_receipt, "_WITH_TARGET", None)
    monkeypatch.setattr(builtins, "__import__", importer)
    return state


def test_two_first_plans_at_once_both_get_a_plan(racy_import):
    """Two build requests racing the planner's first import both plan; the import runs once."""
    plans: list[object] = []
    threads = [
        threading.Thread(target=lambda: plans.append(build_receipt._build_plan({}, "."))) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert plans == [PLAN, PLAN]
    assert racy_import["imports"] == 1


def test_the_import_lock_does_not_cover_plan(racy_import):
    """A plan that runs long does not hold up another request's plan."""
    gate = threading.Event()
    slow = threading.Thread(target=build_receipt._build_plan, args=({"block": gate}, "."))
    slow.start()
    try:
        deadline = time.monotonic() + 10
        while build_receipt._BUILD_CONFIG is None and time.monotonic() < deadline:
            time.sleep(0.01)  # until the slow plan has imported and is blocked in plan()
        done: list[object] = []
        fast = threading.Thread(target=lambda: done.append(build_receipt._build_plan({}, ".")))
        fast.start()
        fast.join(5)
        assert done == [PLAN]
        assert slow.is_alive()  # still in its plan
    finally:
        gate.set()
        slow.join(10)


def test_an_unreadable_plan_keeps_its_error_in_the_judges_log(tmp_path, monkeypatch, caplog):
    """The agent's refusal names no judge-side error text; the judge's log keeps it."""
    from qda_quant.config import BuildConfig

    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=100,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        format_menu=["Q4_K"],
    )

    def unreadable(raw, base):  # noqa: ANN001, ANN202, ARG001 -- `BuildConfig.from_dict`'s own signature
        raise ImportError("cannot import name 'AutoConfig' from /judge/secret/path/transformers")

    monkeypatch.setattr(BuildConfig, "from_dict", staticmethod(unreadable))
    with caplog.at_level(logging.WARNING, logger=build_receipt.__name__):
        with pytest.raises(Refused, match="could not be read") as refused:
            build_receipt.budget_check(r.host, {}, str(tmp_path))
    assert "/judge/secret/path" not in str(refused.value)
    assert "ImportError" not in str(refused.value)
    assert "ImportError: cannot import name 'AutoConfig' from /judge/secret/path" in caplog.text


def test_a_start_that_cannot_import_the_planner_only_logs(monkeypatch, caplog):
    """`warm_planner` at the judge's start never raises."""

    def broken():  # noqa: ANN202
        raise ImportError("no planner")

    monkeypatch.setattr(build_receipt, "_planner", broken)
    with caplog.at_level(logging.WARNING, logger=build_receipt.__name__):
        build_receipt.warm_planner()
    assert "ImportError: no planner" in caplog.text
