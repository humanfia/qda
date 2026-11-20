"""Lazy built-ins and package extensions preserve recipe validation and provenance."""

from __future__ import annotations

import importlib.metadata as md
import sys
from types import SimpleNamespace

import pytest
from qda_quant import registry
from qda_quant.registry import Registry
from qda_quant.step import Refused


def test_lazy_table_and_once_only_extension(monkeypatch):
    reg = Registry("things", {"json": "json:dumps"})
    calls = []

    def extend():
        calls.append("extend")
        reg.register("extra", "json:loads")

    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(
        md, "entry_points", lambda **kw: [SimpleNamespace(name="extension", load=lambda: extend)]
    )
    assert calls == []
    assert list(reg) == ["extra", "json"]
    assert all(isinstance(obj, str) for obj in reg._entries.values())
    assert reg["json"] is sys.modules["json"].dumps
    assert reg["extra"] is sys.modules["json"].loads
    reg.register("extra", "json:loads")
    assert reg["extra"] is sys.modules["json"].loads
    assert calls == ["extend"]
    assert reg.provenance("extra") == ("json", None)
    with pytest.raises(Refused, match=r"unknown things plugin 'bad'; known: \['extra', 'json'\]"):
        reg["bad"]


def test_extension_step_validates_in_recipe(monkeypatch):
    from qda_quant import Recipe, Step
    from qda_quant.step import REGISTRY

    class Extra(Step):
        kind = "model"
        amount: int = 1

    monkeypatch.setattr(REGISTRY, "_entries", dict(REGISTRY._entries))
    REGISTRY.register("extra", Extra)
    recipe = Recipe.from_dict({"steps": [{"extra": {"amount": 3}}], "calib": {"file": {"path": "c.jsonl"}}})
    assert isinstance(recipe.steps[0], Extra)
    assert recipe.to_dict()["steps"] == [{"extra": {"amount": 3}}]
    with pytest.raises(ValueError, match="provided twice"):
        REGISTRY.register("extra", type("Other", (Step,), {"__module__": "other.module"}))
    with pytest.raises(TypeError, match="not a Step subclass"):
        REGISTRY.register("bad", int)
