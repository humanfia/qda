"""The fla import guard: keep its added models, put back the ones it replaced."""

from __future__ import annotations

import sys
import types

from qda_quant.arch import derive_trace


class _Mapping:
    def __init__(self, extra):
        self._extra_content = extra


def test_the_guard_keeps_added_keys_and_reverts_replaced_ones() -> None:
    native = object()
    replaced = object()
    added = object()
    mapping = _Mapping({"native": replaced, "added": added})
    # fla's import: overwrite "native", leave "added" as it was, add "new"
    mapping._extra_content = {"native": object(), "added": added, "new": object()}
    before = [(mapping, {"native": native}, {"native"})]
    derive_trace._restore_native_models(before)
    assert mapping._extra_content["native"] is native
    assert mapping._extra_content["added"] is added
    assert "new" in mapping._extra_content


def test_import_fla_removes_a_shadowing_entry_when_fla_came_first(monkeypatch) -> None:
    fla = types.ModuleType("fla")
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setattr(derive_trace, "_FLA_GUARDED", False)
    monkeypatch.setattr(derive_trace, "_has_fla", lambda: True)

    fla_config = type("FlaMambaConfig", (), {"model_type": "mamba"})
    # a QDA adapter's own registration (``register_models``): not a native type, so the guard keeps it
    qda_config = type("DeepseekV41Config", (), {"model_type": "deepseek_v41"})
    from transformers import AutoModel, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "deepseek_v41" not in CONFIG_MAPPING._mapping
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "mamba", fla_config)
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "deepseek_v41", qda_config)
    for auto in (AutoModel, AutoModelForCausalLM):
        monkeypatch.setitem(auto._model_mapping._extra_content, fla_config, object())
    monkeypatch.setitem(AutoModelForCausalLM._model_mapping._extra_content, qda_config, object())

    assert derive_trace._import_fla() is fla
    assert "mamba" not in CONFIG_MAPPING._extra_content
    assert CONFIG_MAPPING._extra_content["deepseek_v41"] is qda_config
    assert qda_config in AutoModelForCausalLM._model_mapping._extra_content
    for auto in (AutoModel, AutoModelForCausalLM):
        assert fla_config not in auto._model_mapping._extra_content
