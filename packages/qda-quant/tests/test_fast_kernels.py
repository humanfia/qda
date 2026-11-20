"""A CUDA build refuses a GDN model whose fast kernels would fall back to transformers' torch path.

The four bindings are the `use_kernel_func_from_hub_with_fallback` calls of transformers 5.20/5.21's
Qwen3.5/3.6/3.8 modeling: `causal_conv1d_fn` / `causal_conv1d_update` (top-level `causal_conv1d`, or the
adapter `route_gdn` binds from fla's own Triton conv) and `chunk_gated_delta_rule` /
`fused_recurrent_gated_delta_rule` (fla). CPU / meta builds keep the reference path; these tests pin the
availability functions, so no card and no download is needed.
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from qda_quant.arch import ModelGraph
from qda_quant.checks import gdn_fallback, gdn_kernel_paths, has_gdn_config
from qda_quant.step import Refused

FUNCTIONS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)


def graph(gdn: bool = True) -> ModelGraph:
    kind = "linear_attn_in" if gdn else "attn_in"
    fake = SimpleNamespace(layers=[SimpleNamespace(groups=[SimpleNamespace(kind=kind)])])
    return cast(ModelGraph, fake)


def _bound(flag: bool):
    """A stand-in for transformers' fallback wrapper: its closure records what it bound."""
    is_new_implementation = flag

    def wrapper(*_args, **_kwargs):
        return is_new_implementation

    return wrapper


def _adapter():
    """A stand-in for the fla conv adapter `route_gdn` installs."""

    def wrapper(*_args, **_kwargs):
        return None

    wrapper._qda_fast_on_cuda = True
    return wrapper


def _modeling(monkeypatch, *, conv: str, fla: str) -> None:
    """Pin the modeled bindings: `conv`/`fla` are "fast", "adapter" or "torch"."""

    def fn(kind):
        return _adapter() if kind == "adapter" else _bound(kind == "fast")

    module = SimpleNamespace(
        causal_conv1d_fn=fn(conv),
        causal_conv1d_update=fn(conv),
        torch_chunk_gated_delta_rule=fn(fla),
        torch_recurrent_gated_delta_rule=fn(fla),
    )
    import qda_quant.checks as checks

    monkeypatch.setattr(checks, "_modeling_modules", lambda: [module])


def unavailable(monkeypatch) -> None:
    """No pinned modeling module to inspect, and every availability flag false (the fallback path)."""
    import qda_quant.checks as checks
    import transformers.utils.import_utils as iu

    monkeypatch.setattr(checks, "_modeling_modules", lambda: [])
    monkeypatch.setattr(iu, "is_flash_linear_attention_available", lambda: False)
    monkeypatch.setattr(iu, "is_causal_conv1d_available", lambda: False)
    monkeypatch.setattr("qda_quant.arch.derive_trace.fla_causal_conv", lambda: None)


def test_every_fallback_is_named_with_the_missing_packages(monkeypatch):
    unavailable(monkeypatch)
    message = gdn_fallback(graph())
    assert message is not None
    for name in FUNCTIONS:
        assert name in message, name
    assert "causal-conv1d" in message
    assert "flash-linear-attention==0.6.0" in message
    assert "Kernel paths:" in message


def test_a_non_gdn_model_is_never_refused(monkeypatch):
    unavailable(monkeypatch)
    assert gdn_fallback(graph(gdn=False)) is None


def test_all_four_kernels_are_quiet(monkeypatch):
    _modeling(monkeypatch, conv="fast", fla="fast")
    assert gdn_fallback(graph()) is None


def test_the_bound_callable_decides_not_package_metadata(monkeypatch):
    """A discoverable-but-broken conv package leaves the torch reference bound: that must refuse."""
    import transformers.utils.import_utils as iu

    _modeling(monkeypatch, conv="torch", fla="fast")
    monkeypatch.setattr(iu, "is_causal_conv1d_available", lambda: True)  # metadata says installed
    monkeypatch.setattr(iu, "is_flash_linear_attention_available", lambda: True)
    # no fla conv to install either: the binding stays transformers' torch reference
    monkeypatch.setattr("qda_quant.arch.derive_trace.fla_causal_conv", lambda: None)
    message = gdn_fallback(graph())
    assert message is not None
    assert "causal_conv1d_fn" in message and "causal_conv1d_update" in message
    paths = {path.function: (kernel, fast) for path, kernel, fast in gdn_kernel_paths()}
    assert paths["causal_conv1d_fn"] == ("torch reference", False)
    assert paths["chunk_gated_delta_rule"][1] is True


def test_the_conv_is_fast_through_flas_triton_conv(monkeypatch):
    """The conv names are the adapter route_gdn installs: both are fast and say where they run."""
    _modeling(monkeypatch, conv="adapter", fla="fast")
    assert gdn_fallback(graph()) is None
    paths = {path.function: kernel for path, kernel, _fast in gdn_kernel_paths()}
    assert paths["causal_conv1d_fn"] == "causal_conv1d_fn from fla.modules.causal_conv1d"
    assert paths["causal_conv1d_update"] == "causal_conv1d_update from fla.modules.causal_conv1d"
    assert paths["chunk_gated_delta_rule"] == "fla chunk_gated_delta_rule"
    assert paths["fused_recurrent_gated_delta_rule"] == "fla fused_recurrent_gated_delta_rule"


def test_a_cuda_build_refuses_and_a_cpu_build_keeps_the_reference(monkeypatch):
    from qda_quant.engine.build import _refuse_fast_gdn

    unavailable(monkeypatch)
    with pytest.raises(Refused, match="flash-linear-attention"):
        _refuse_fast_gdn([torch.device("cuda")], graph())
    _refuse_fast_gdn([torch.device("cpu")], graph())
    _refuse_fast_gdn([torch.device("cuda")], graph(gdn=False))


def test_the_config_declares_the_gdn_layers():
    assert has_gdn_config(SimpleNamespace(layer_types=["linear_attention", "full_attention"]))
    assert not has_gdn_config(SimpleNamespace(layer_types=["full_attention"]))
    assert has_gdn_config(SimpleNamespace(text_config=SimpleNamespace(layer_types=["linear_attention"])))


def test_a_real_gdn_config_is_recognised():
    from .tiny import config

    assert has_gdn_config(config("qwen3_5"))
    assert not has_gdn_config(config("qwen3"))
