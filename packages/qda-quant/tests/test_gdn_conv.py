"""The GDN conv adapter: fla's own Triton causal conv under transformers' two conv names.

transformers' Qwen3.5/3.6/3.8 modeling calls a module-level ``causal_conv1d_fn`` (prefill, `[B, D, T]`) and
``causal_conv1d_update`` (decode, `[B, D, 1]` plus an in-place `[B, D, W]` state), and looks for a top-level
``causal_conv1d`` package that does not exist for this tree's torch. `route_gdn` binds fla's
``fla.modules.causal_conv1d`` under those names instead. The CPU tests pin the device choice to exercise the
argument mapping without a card; the marked test compares the Triton kernels with the torch path on one.
"""

import importlib.util
import inspect
import sys
import types

import pytest
import torch
from qda_quant.arch import derive_trace
from qda_quant.step import Refused
from transformers.models.qwen3_5 import modeling_qwen3_5 as _qwen

#: transformers' own torch conv functions, captured at import: `route_gdn` later binds fla under these names.
TORCH_CONV = inspect.unwrap(_qwen.causal_conv1d_fn)
TORCH_UPDATE = inspect.unwrap(_qwen.causal_conv1d_update)


class _Conv:
    """A fla-shaped causal conv standin: records the mapped call and returns the fast path's shapes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def causal_conv1d(self, **kw):
        self.calls.append(("prefill", kw))
        return torch.zeros_like(kw["x"]), None

    def causal_conv1d_update(self, x, cache, **kw):
        self.calls.append(("update", {"x": x, **kw}))
        return torch.zeros_like(x), cache


def test_prefill_maps_the_modeling_signature_to_fla(monkeypatch):
    conv, ref = _Conv(), lambda *_a, **_k: torch.ones(2, 8, 16)
    fn = derive_trace._conv_fn(conv, ref)
    monkeypatch.setattr(derive_trace, "_on_cuda", lambda _t: True)
    out = fn(torch.zeros(2, 8, 16), torch.zeros(8, 4), activation="silu")
    ((kind, kw),) = conv.calls
    assert kind == "prefill"
    assert tuple(kw["x"].shape) == (2, 16, 8), "x is transposed to [B, T, D] for fla"
    assert kw["activation"] == "silu"
    assert kw["initial_state"] is None
    assert tuple(out.shape) == (2, 8, 16), "the output comes back as [B, D, T]"


def test_cpu_takes_the_torch_reference(monkeypatch):
    conv, seen = _Conv(), []

    def ref(hidden_states, weight, bias=None, activation=None):
        seen.append(activation)
        return torch.full_like(hidden_states, 7)

    fn = derive_trace._conv_fn(conv, ref)
    out = fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), activation="silu")
    assert conv.calls == [] and seen == ["silu"], "a CPU forward never reaches fla"
    assert float(out[0, 0, 0]) == 7


def test_decode_update_maps_the_step_and_keeps_the_state_in_place(monkeypatch):
    class Conv(_Conv):
        def causal_conv1d_update(self, x, cache, **kw):
            self.calls.append(("update", {"x": x, **kw}))
            cache.add_(1)  # fla writes the caller's cache in place
            return torch.zeros_like(x), cache

    conv, ref = Conv(), lambda *_a, **_k: torch.zeros(2, 8, 1)
    fn = derive_trace._conv_update(conv, ref)
    monkeypatch.setattr(derive_trace, "_on_cuda", lambda _t: True)
    state = torch.zeros(2, 8, 4)
    out = fn(torch.zeros(2, 8, 1), state, torch.zeros(8, 4), activation="silu")
    ((kind, kw),) = conv.calls
    assert kind == "update" and tuple(kw["x"].shape) == (2, 1, 8)
    assert torch.equal(state, torch.ones(2, 8, 4)), "the caller's conv state is updated"
    assert tuple(out.shape) == (2, 8, 1)


def test_an_activation_fla_cannot_apply_is_refused(monkeypatch):
    fn = derive_trace._conv_fn(_Conv(), lambda *_a, **_k: None)
    monkeypatch.setattr(derive_trace, "_on_cuda", lambda _t: True)
    with pytest.raises(Refused, match="gelu"):
        fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), activation="gelu")


def test_an_argument_the_adapter_cannot_honour_is_refused(monkeypatch):
    fn = derive_trace._conv_fn(_Conv(), lambda *_a, **_k: None)
    monkeypatch.setattr(derive_trace, "_on_cuda", lambda _t: True)
    with pytest.raises(Refused, match="seq_idx"):
        fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), seq_idx=torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(Refused, match="output_final_state"):
        fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), output_final_state=True)


def test_cpu_refuses_a_state_or_packed_input_it_cannot_honour():
    """The torch reference has no initial state and mixes packed sequences: refuse, never drop."""
    fn = derive_trace._conv_fn(_Conv(), lambda *_a, **_k: None)
    with pytest.raises(Refused, match="initial state"):
        fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), initial_state=torch.ones(1, 4, 2))
    with pytest.raises(Refused, match="packed"):
        fn(torch.zeros(1, 4, 5), torch.zeros(4, 3), cu_seqlens=torch.tensor([0, 2, 4]))


def test_route_gdn_binds_flas_conv_under_the_modeling_names(monkeypatch):
    if importlib.util.find_spec("causal_conv1d") is not None:
        pytest.skip("a top-level causal_conv1d package makes transformers' own binding fast")
    module = types.ModuleType("fake_gdn_modeling_for_adapter")
    ref = lambda *_a, **_k: torch.ones(1, 4, 5)  # noqa: E731
    setattr(module, "causal_conv1d_fn", ref)  # noqa: B010
    setattr(module, "causal_conv1d_update", ref)  # noqa: B010
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(derive_trace, "fla_causal_conv", _Conv)

    class Fake(torch.nn.Module):
        __module__ = module.__name__

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    model = Fake()
    model.add_module("layer", Layer())
    derive_trace.route_gdn(model)
    for name in derive_trace._CONV_KERNELS:
        bound = getattr(module, name)
        assert bound is not ref and id(bound) in derive_trace._ROUTED, name
    prefill = module.causal_conv1d_fn
    assert torch.equal(prefill(torch.zeros(1, 4, 5), torch.zeros(4, 3)), torch.ones(1, 4, 5))


def _error(a: torch.Tensor, b: torch.Tensor, tol: float) -> tuple[float, float]:
    """(max|a-b| / max|ref|, fraction of elements whose |a-b| exceeds tol * max|ref|).

    A normalised max and a violating-element fraction, so one element near zero in bf16 cannot decide the
    comparison through its own relative error.
    """
    scale = b.float().abs().max().clamp_min(1e-12)
    diff = (a.float() - b.float()).abs()
    return (diff.max() / scale).item(), (diff > tol * scale).float().mean().item()


def _assert_conv_close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    norm_rel, frac = _error(a, b, tol)
    assert norm_rel <= tol and frac == 0, (
        f"{what}: norm_rel {norm_rel:.3g} (tol {tol}), {frac:.3g} of elements exceed tol * max|ref|"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float32, 1e-5)], ids=["bf16", "fp32"])
def test_the_conv_matches_the_torch_path_on_a_card(dtype, tol):
    """Prefill (with and without an initial state) and the decode update, on the model's conv shapes.

    The reference is transformers' own unwrapped torch conv (captured at import, before `route_gdn` binds
    fla under the modeling module's names). An initial state is compared by running the reference over the
    state's last W-1 tokens followed by the input and keeping the input's T outputs.
    """
    pytest.importorskip("fla")
    torch.manual_seed(0)
    b, d, t, w = 2, 256, 32, 4
    conv = derive_trace.fla_causal_conv()
    assert conv is not None, "fla's Triton causal conv must import on this machine"
    prefill, update = derive_trace._conv_fn(conv, TORCH_CONV), derive_trace._conv_update(conv, TORCH_UPDATE)
    x = torch.randn(b, d, t, device="cuda", dtype=dtype)
    weight = torch.randn(d, w, device="cuda", dtype=dtype)
    # a cache state holds the last W tokens; the oldest (index 0) is outside the next window
    state = torch.randn(b, d, w, device="cuda", dtype=dtype)

    _assert_conv_close(prefill(x, weight, activation="silu"), TORCH_CONV(x, weight, activation="silu"),
                       tol, "prefill")  # fmt: skip
    with_state = prefill(x, weight, activation="silu", initial_state=state)
    reference = TORCH_CONV(torch.cat([state[..., 1:], x], dim=-1), weight, activation="silu")[..., -t:]
    _assert_conv_close(with_state, reference, tol, "prefill with an initial state")

    x1 = torch.randn(b, d, 1, device="cuda", dtype=dtype)
    fast_state = torch.randn(b, d, w, device="cuda", dtype=dtype)
    slow_state = fast_state.clone()
    fast = update(x1, fast_state, weight, activation="silu")
    slow = TORCH_UPDATE(x1, slow_state, weight, activation="silu")
    _assert_conv_close(fast, slow, tol, "decode update output")
    _assert_conv_close(fast_state, slow_state, tol, "updated conv state")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float32, 1e-5)], ids=["bf16", "fp32"])
def test_packed_prefill_keeps_sequences_apart_on_a_card(dtype, tol):
    """One row of packed segments: each segment's outputs match the torch reference run on that segment."""
    pytest.importorskip("fla")
    torch.manual_seed(0)
    conv = derive_trace.fla_causal_conv()
    assert conv is not None, "fla's Triton causal conv must import on this machine"
    prefill = derive_trace._conv_fn(conv, TORCH_CONV)
    b, d, w = 1, 256, 4
    bounds = [0, 100, 260, 300]  # three segments in one row
    x = torch.randn(b, d, bounds[-1], device="cuda", dtype=dtype)
    weight = torch.randn(d, w, device="cuda", dtype=dtype)
    cu = torch.tensor(bounds, dtype=torch.int32)
    packed = prefill(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu.to("cuda"),
        cu_seqlens_cpu=cu,
    )
    reference = torch.cat(
        [
            TORCH_CONV(x[..., start:end], weight, activation="silu")
            for start, end in zip(bounds[:-1], bounds[1:], strict=True)
        ],
        dim=-1,
    )
    _assert_conv_close(packed, reference, tol, "packed prefill")
