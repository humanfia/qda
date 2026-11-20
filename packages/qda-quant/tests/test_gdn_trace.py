"""GDN forwards away from a card must take transformers' torch reference, not fla's Triton kernels.

With `fla` importable, transformers binds `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`
in the Qwen3.5/3.6/3.8 model modules to a wrapper around its reference implementation that runs fla's Triton
kernels; those cannot take meta or CPU tensors. `derive.route_gdn` installs a router that sends a CUDA input
to the bound wrapper and every other device to the reference, so the trace, the fp64 checks and ordinary CPU
forwards all work. The stand-in below raises exactly where fla would crash.
"""

import functools
import sys

import pytest
from qda_quant.arch import describe
from qda_quant.step import Refused

from .tiny import model

KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


def _kernel_module(kind: str):
    """The module a model of this kind reads its GDN kernels from (the one holding its layer classes)."""
    return sys.modules[type(model(kind, meta=True).model.layers[0]).__module__]


def _bind_standin(monkeypatch, kind: str, ref=None):
    """Bind both kernel names to a fla-like wrapper over `ref`; return the wrappers and the call log."""
    module, wrapped, calls = _kernel_module(kind), {}, []
    for name in KERNELS:
        reference = getattr(module, name) if ref is None else ref

        @functools.wraps(reference)
        def fla(*args, _name=name, _reference=reference, **kwargs):
            x = args[0]
            if x.is_meta or x.device.type != "cuda":
                raise RuntimeError(f"fla {_name} on {x.device}")
            calls.append(_name)
            return _reference(*args, **kwargs)

        monkeypatch.setattr(module, name, fla)
        wrapped[name] = fla
    return module, wrapped, calls


@pytest.mark.parametrize("meta", [False, True], ids=["cpu", "meta"])
def test_describe_does_not_run_fla(monkeypatch, meta):
    _module, _wrapped, calls = _bind_standin(monkeypatch, "qwen3_5")
    graph = describe(model("qwen3_5", meta=meta), verify=False)
    assert calls == [], "the trace ran the fla kernel"
    assert "linear_attn_in" in {g.kind for layer in graph.layers for g in layer.groups}


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_verified_describe_does_not_run_fla(monkeypatch, kind):
    """The default `describe` checks every relation in fp64 on CPU after the trace; that forward is CPU too.

    Restoring the wrapper when the trace ends left these checks calling fla on a CPU tensor, which raises.
    """
    _module, _wrapped, calls = _bind_standin(monkeypatch, kind)
    graph = describe(model(kind))
    assert calls == [], "the fp64 verification ran the fla kernel"
    assert graph.verified


def test_the_trace_leaves_the_kernel_routed_by_device(monkeypatch):
    from qda_quant.arch.derive import _ROUTED

    module, wrapped, _calls = _bind_standin(monkeypatch, "qwen3_5")
    describe(model("qwen3_5"), verify=False)
    for name in KERNELS:
        routed = getattr(module, name)
        assert routed is not wrapped[name]
        assert id(routed) in _ROUTED, name


def test_a_failed_trace_keeps_the_kernel_routed(monkeypatch):
    def failing(*_args, **_kwargs):
        raise ZeroDivisionError("the reference fails mid-trace")

    module, _wrapped, _calls = _bind_standin(monkeypatch, "qwen3_5", ref=failing)
    with pytest.raises(Refused, match="the reference fails mid-trace"):  # one probe refusal
        describe(model("qwen3_5"), verify=False)
    from qda_quant.arch.derive import _ROUTED

    for name in KERNELS:
        assert id(getattr(module, name)) in _ROUTED, name


def test_a_loaded_model_forwards_on_cpu_with_fla_installed():
    """fla makes transformers bind its Triton kernels; an ordinary CPU forward must take the torch reference."""
    pytest.importorskip("fla")
    import torch
    from qda_quant.intake.loading import load_model

    from .tiny import config

    cfg = config("qwen3_5")
    cfg._attn_implementation = "eager"
    m = load_model(None, config=cfg, weights=False)
    ids = torch.arange(1, 9)[None]
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert tuple(out.logits.shape) == (1, 8, cfg.vocab_size)
