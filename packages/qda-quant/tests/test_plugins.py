"""Package extensions participate in recipe validation and built-in lookup without heavy imports."""

import importlib.metadata as md
from types import SimpleNamespace

import pytest
from qda_quant import registry
from qda_quant.step import Refused


def fake_plugins(monkeypatch, register):
    """Discover one package registration hook, without installing a distribution."""
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(registry, "_plugins_loading", False)
    monkeypatch.setattr(registry, "_plugin_failure", None)
    monkeypatch.setattr(
        md,
        "entry_points",
        lambda **kw: (
            [SimpleNamespace(name="extension", load=lambda: register)]
            if kw.get("group") == "qda.plugins"
            else []
        ),
    )


@pytest.mark.parametrize("fail_on_import", [False, True])
def test_failed_hook_is_named_and_never_retried(monkeypatch, fail_on_import):
    """A failed provider cannot hide or rerun earlier hooks that already changed registries."""
    plugins = registry.Registry("example", {"builtin": object()})
    calls = []
    cause = ValueError("broken provider")

    def first():
        calls.append("first")
        plugins.register("added", object())
        assert "added" in plugins  # hooks may read a registry while registering

    def fail():
        calls.append("failed")
        raise cause

    fake_plugins(monkeypatch, first)
    monkeypatch.setattr(
        md,
        "entry_points",
        lambda **kw: [
            SimpleNamespace(name="a-first", load=lambda: first),
            SimpleNamespace(name="b-broken", load=fail if fail_on_import else lambda: fail),
            SimpleNamespace(name="c-last", load=lambda: lambda: calls.append("last")),
        ],
    )
    with pytest.raises(ValueError, match="broken provider") as error:
        plugins["builtin"]
    assert error.value is cause
    assert str(error.value) == "broken provider"
    assert error.value.__notes__ == ["qda.plugins hook 'b-broken' failed"]
    for read in (lambda: plugins["builtin"], lambda: list(plugins), lambda: "added" in plugins):
        with pytest.raises(ValueError, match="broken provider") as repeated:
            read()
        assert repeated.value is error.value
    assert calls == ["first", "failed"]


def test_successful_hook_runs_once_with_reentrant_reads(monkeypatch):
    """A hook can resolve its registrations without triggering another discovery pass."""
    plugins = registry.Registry("example")
    calls = []
    value = object()

    def hook():
        calls.append("hook")
        plugins.register("added", value)
        assert plugins["added"] is value
        assert not registry._plugins_loaded

    fake_plugins(monkeypatch, hook)
    assert plugins["added"] is value
    assert list(plugins) == ["added"]
    assert registry._plugins_loaded and calls == ["hook"]


@pytest.fixture
def fresh_steps(monkeypatch):
    from qda_quant.step import REGISTRY

    list(REGISTRY)
    monkeypatch.setattr(REGISTRY, "_entries", dict(REGISTRY._entries))
    return REGISTRY


@pytest.fixture
def fresh_clips(monkeypatch):
    from qda_quant.clips import CLIPS

    list(CLIPS)
    monkeypatch.setattr(CLIPS, "_entries", dict(CLIPS._entries))
    return CLIPS


def test_third_party_clip_by_name_in_a_step(monkeypatch, fresh_clips):
    """A package's clip hook changes the scales a named step chooses."""
    import torch
    from qda_quant import Format, Target
    from qda_quant.clips import Clip
    from qda_quant.solve.quant import Quantizer
    from qda_quant.step import REGISTRY

    class Halve(Clip):
        name = "halve"

        def _choose(self, q, w, hdiag, grid_k):
            scale, zero = q.params(w)
            return scale * 0.5, zero

    fake_plugins(monkeypatch, lambda: fresh_clips.register("halve", Halve()))
    step = REGISTRY["rtn"](clip="halve")
    assert step.clip == "halve" and step.picks_ranges() and not step.needs
    fmt = Format.parse("w4a16-sym-g128")
    quantizer = Quantizer.of(fmt, Target().plan(fmt, "dense"), "absmax")
    weights = torch.randn(1, 2, 128, generator=torch.Generator().manual_seed(0))
    scale, zero = fresh_clips["halve"].choose(quantizer, weights)
    torch.testing.assert_close(scale, quantizer.params(weights)[0] * 0.5, rtol=0, atol=0)
    assert zero is None
    assert "halve" in sorted(fresh_clips) and "mse" in fresh_clips


def test_rtn_honours_a_clip_refusal(monkeypatch, fresh_clips):
    """A clip's refusal reaches planning before its range search runs."""
    from qda_quant import Format, Target
    from qda_quant.clips import Clip
    from qda_quant.step import REGISTRY, Refused

    class NoW4A16(Clip):
        name = "no-w4a16"

        def _choose(self, q, w, hdiag, grid_k):
            raise AssertionError("a refused clip must not be chosen")

        def refusal(self, fmt, plan):
            return "this search does not serve W4A16" if fmt.act == "bf16" else None

    fake_plugins(monkeypatch, lambda: fresh_clips.register("no-w4a16", NoW4A16()))
    step = REGISTRY["rtn"](clip="no-w4a16")
    fmt = Format.parse("w4a16-sym-g128")
    with pytest.raises(Refused, match="rtn clip='no-w4a16': this search does not serve W4A16"):
        step.check(None, fmt, Target().plan(fmt, "dense"))


def test_unknown_step_in_a_recipe(monkeypatch, fresh_steps):
    from qda_quant import Recipe

    fake_plugins(monkeypatch, lambda: None)
    with pytest.raises(Refused, match=r"unknown step 'gtpq'; known: \['autoround', 'awq', .*'rtn', .*\]"):
        Recipe.from_dict({"steps": [{"gtpq": {}}], "calib": {"file": {"path": "c.jsonl"}}})


def test_target_tables_by_name():
    from qda_quant import Target

    assert Target.parse("h200:marlin") == Target("sm90", "0.33.0", "marlin")
    with pytest.raises(
        Refused, match="unknown architecture.*llama_cpp, sm100, sm103, sm120, sm80, sm89, sm90"
    ):
        Target.parse("blackwell")


def test_package_and_registry_import_without_torch():
    """Importing the shared registry keeps the build API usable without torch."""
    import subprocess
    import sys

    code = (
        "import sys, qda_quant, qda_quant.registry, qda_quant.step, qda_quant.recipe, qda_quant.target\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))[:5]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_step_registry_finds_every_builtin_without_entry_points():
    import subprocess
    import sys

    code = (
        "import importlib.metadata as md\nmd.entry_points = lambda **kw: []\n"
        "from qda_quant.step import REGISTRY\n"
        "assert {'autoround', 'awq', 'fisher_trace', 'gptq', 'gsq', 'guided', 'head', 'permute', 'rotate', 'rtn'} <= set(REGISTRY), sorted(REGISTRY)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_registered_step_names_its_reproducibility_class():
    from qda_quant.step import REGISTRY

    builtin = [(k, cls) for k, cls in REGISTRY.items() if cls.__module__.startswith("qda_quant.steps")]
    assert len(builtin) >= 11
    missing = [k for k, cls in builtin if "Reproducibility:" not in (cls.__doc__ or "")]
    assert not missing, missing
