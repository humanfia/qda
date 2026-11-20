"""Trace one tiny forward: the module calls, the tensor identities they share, and the GDN kernels."""

from __future__ import annotations

import functools
import importlib.util
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from ..step import Refused, probe_refusal
from .derive_experts import _expert_stack

_TRACE_TOKENS = 8


def _root(t: torch.Tensor | None) -> torch.Tensor | None:
    return t._base if t is not None and t._base is not None else t


class _Trace:
    """Records, per module call: the id of its first tensor input and of its (first) tensor output."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []  # (module name, input id, output id)
        self.keep: list[torch.Tensor] = []  # keep tensors alive so ids are not reused

    def hook(self, name: str):  # noqa: ANN201
        def fn(_m: nn.Module, args: tuple, out: Any) -> None:
            x = next((a for a in args if isinstance(a, torch.Tensor)), None)
            y = (
                out
                if isinstance(out, torch.Tensor)
                else next(
                    (
                        o
                        for o in (out if isinstance(out, tuple | list) else ())
                        if isinstance(o, torch.Tensor)
                    ),
                    None,
                )
            )
            x, y = _root(x), _root(y)  # a view (e.g. MoE's hidden.view(-1, D)) is the same data
            self.keep += [t for t in (x, y) if t is not None]
            self.calls.append((name, id(x) if x is not None else -1, id(y) if y is not None else -2))

        return fn


@contextmanager
def _meta_nonzero() -> Iterator[None]:
    """Let ``torch.nonzero`` run on meta tensors, as if every element were non-zero.

    A forward that turns a mask into indices calls ``torch.nonzero`` (the QSA indexer of Qwen4Exp does), and
    torch leaves its meta kernel unimplemented because the shape is data-dependent. The trace reads module
    calls and tensor identity, never values, so the fake shape is enough here; a model that branches on that
    shape can still trace differently from its real forward. The flag is process-global on older torch and
    context-local on recent versions, so traces there must not overlap; `qda plan` traces one model at a time.

    The switch is private to torch, so a version without it is not an error: this is then a no-op and a forward
    that still calls ``nonzero`` on meta refuses below (see `trace`), as any other op without a meta kernel.
    """
    try:
        from torch.fx.experimental import _config as exp
    except ImportError:
        yield
        return

    missing = object()
    prev = getattr(exp, "meta_nonzero_assume_all_nonzero", missing)
    if prev is missing:
        yield
        return
    exp.meta_nonzero_assume_all_nonzero = True
    try:
        yield
    finally:
        exp.meta_nonzero_assume_all_nonzero = prev


@contextmanager
def _meta_scalar() -> Iterator[None]:
    """Let `.item()` on a meta tensor answer 0 while the trace runs.

    transformers' dynamic RoPE picks its short or long frequency set with ``if torch.max(position_ids) + 1 >
    original_max_position_embeddings``, an ``if`` on a meta scalar, and torch leaves the scalar read
    unimplemented (up to transformers 5.20; 5.21 spells it as `torch.where`). The trace reads module calls and
    tensor identity, never values, and both frequency sets have the same shape, so 0 is as good as the real
    length; a model that branches on a value can still trace differently from its real forward, as with
    `_meta_nonzero`. The override is a dispatcher entry for the Meta key alone and goes away with `lib`.
    """
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_local_scalar_dense", lambda _t: 0.0, "Meta")
    try:
        yield
    finally:
        lib._destroy()


_GDN_KERNELS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "chunk_kimi_delta_attention",  # GLM-5.3-Flash's linear attention
    "recurrent_kimi_delta_attention",
)

#: The conv names transformers' Qwen3.5/3.6/3.8 modeling binds from a top-level ``causal_conv1d`` package.
_CONV_KERNELS = ("causal_conv1d_fn", "causal_conv1d_update")

#: The activations fla's Triton causal conv applies; any other would silently change the model.
_FLA_CONV_ACTIVATIONS = (None, "silu", "swish")


#: The routers installed so far, by id. Keeping the function alive keeps its id unique, so a second model of a
#: family whose module already holds a router neither wraps it again nor mistakes a fresh reuse of an id for one.
_ROUTED: dict[int, Any] = {}


_FLA_GUARDED = False


def _fla_registries() -> list[tuple[Any, dict[Any, Any], set[str]]]:
    """The auto registries fla rewrites, with a copy of their extra content and the native model types."""
    from transformers import AutoModel, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    native = set(CONFIG_MAPPING._mapping)
    return [
        (mapping, dict(mapping._extra_content), native)
        for mapping in (CONFIG_MAPPING, AutoModel._model_mapping, AutoModelForCausalLM._model_mapping)
    ]


def _restore_native_models(
    before: list[tuple[Any, dict[Any, Any], set[str]]] | None = None,
) -> None:
    """Undo what fla's import registered over transformers: restore replaced values, drop shadowing adds."""
    if before is None:  # fla was imported before QDA: no snapshot, remove only the shadowing entries
        before = [(mapping, {}, native) for mapping, _, native in _fla_registries()]
    for mapping, previous, native in before:
        extra = mapping._extra_content
        for key in list(extra):
            if key in previous:
                if extra[key] is not previous[key]:
                    extra[key] = previous[key]
                continue
            model_type = key if isinstance(key, str) else getattr(key, "model_type", None)
            if model_type in native:
                del extra[key]


def _has_fla() -> bool:
    """Whether fla is installed (without importing it)."""
    return importlib.util.find_spec("fla") is not None


def _import_fla() -> Any | None:
    """Import fla once, after snapshotting transformers' auto registries, and undo its class takeover.

    ``fla/__init__`` imports ``fla.models``, whose per-family modules call ``AutoConfig.register(...,
    exist_ok=True)`` and ``AutoModel[ForCausalLM].register(...)``: a key already in a base mapping (or an
    extra-content entry) is replaced. Snapshot before the first import and put back every replaced value;
    keys fla only adds (GLA, HGRN, ...) stay. If fla was imported before QDA, no snapshot exists: remove
    only entries that shadow a native model type, which is the same replacement.
    """
    global _FLA_GUARDED
    if not _has_fla():
        return None
    if "fla" in sys.modules:
        _restore_native_models()
        _FLA_GUARDED = True
        return sys.modules["fla"]
    if _FLA_GUARDED:
        return sys.modules.get("fla")
    before = _fla_registries()
    try:
        import fla
    except Exception:  # fla present but its Triton/torch build does not import here
        _restore_native_models(before)  # a partial import may already have registered its classes
        return None
    _restore_native_models(before)
    _FLA_GUARDED = True
    return fla


def fla_causal_conv() -> Any | None:
    """fla's own Triton causal conv module (``fla.modules.causal_conv1d``), or None when it cannot be imported.

    transformers' GDN modeling asks for a *top-level* ``causal_conv1d`` package for its two conv kernels;
    fla ships the same Triton kernels under its own namespace. `route_gdn` binds this module under the
    modeling module's names when the bound conv is transformers' torch reference, and `qda_quant.checks`
    counts that binding as the conv fast path.
    """
    if _import_fla() is None:
        return None
    try:
        import fla.modules.causal_conv1d as conv
    except Exception:  # fla present but its Triton/torch build does not import here
        return None
    if not all(hasattr(conv, name) for name in ("causal_conv1d", "causal_conv1d_update")):
        return None
    return conv


def _closure_flag(fn: Any) -> bool | None:
    """What transformers' fallback decorator bound: True for the package implementation, False for its torch
    reference; None when `fn` is not that wrapper."""
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or not closure:
        return None
    try:
        cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure), strict=True))
    except ValueError:  # an empty cell
        return None
    flag = cells.get("is_new_implementation")
    return None if flag is None else bool(flag)


def fast_binding(fn: Any) -> bool:
    """Whether this modeling-module kernel name takes a fast kernel on a CUDA input.

    It reads what the name is actually bound to: the fla conv adapter or a device router this module
    installed, or the implementation transformers' decorator captured (its closure records whether the
    package imported or the torch reference did).
    """
    if getattr(fn, "_qda_fast_on_cuda", False):  # the fla conv adapter
        return True
    flag = _closure_flag(fn)
    if flag is not None:  # transformers' wrapper
        return flag
    fast = getattr(fn, "__wrapped__", None)  # a `_by_device` router: its CUDA side
    return bool(fast is not None and _closure_flag(fast))


def route_gdn(model: nn.Module) -> None:
    """Route this model's GDN kernels by the device of their input, for as long as the modules are loaded.

    Once `fla` is importable, transformers binds the model-module names to a wrapper around its reference
    implementation (`integrations/hub_kernels.py::use_kernel_func_from_hub_with_fallback`) that calls fla's
    Triton kernels. Those kernels are the fast path on a card, and impossible everywhere else: a host with no
    driver raises "0 active drivers", and Triton rejects a CPU pointer. The meta skeleton has null pointers, and
    the trace, the fp64 checks, the test suite and any CPU user run ordinary forwards on CPU — with fla
    installed they all reach the wrapper.

    The same wrapper holds the two conv kernels to transformers' torch reference when its package import
    failed (whether or not a module of that name is discoverable). `fla_causal_conv` then supplies an adapter
    with the modeling module's own signatures, CUDA input to fla's Triton conv and every other device to the
    wrapper's reference, so a card gets the fast conv without a separate package.

    The router installed here calls the bound function for a CUDA input and `inspect.unwrap` of it — the torch
    reference the wrapper itself falls back to for dynamo export — for anything else. It is installed once per
    module (`_ROUTED` remembers it), so a second model of the same family reuses it; no-op when fla is absent.
    """
    modules = {
        sys.modules[cls.__module__]
        for m in model.modules()
        for cls in type(m).__mro__
        if cls.__module__ in sys.modules
    }
    route_gdn_modules(modules)


def route_gdn_modules(modules: Any) -> None:
    """`route_gdn` without a model: route these modeling modules' names (idempotent, called once per module).

    A probe that has the modules but no model calls this so it inspects the same bindings a build installs.
    """
    conv: Any | None = None
    conv_ready = False
    for mod in modules:
        for name in _GDN_KERNELS:
            fn = getattr(mod, name, None)
            if fn is None or id(fn) in _ROUTED:
                continue
            ref = inspect.unwrap(fn)
            if ref is not fn:  # the wrapper: a CUDA forward gets fla, every other device the reference
                routed = _by_device(fn, ref)
                _ROUTED[id(routed)] = routed
                setattr(mod, name, routed)
        if not conv_ready and any(getattr(mod, name, None) is not None for name in _CONV_KERNELS):
            conv_ready = True
            conv = fla_causal_conv()
        if conv is not None:
            for name in _CONV_KERNELS:
                fn = getattr(mod, name, None)
                if fn is None or id(fn) in _ROUTED or fast_binding(fn):
                    continue  # absent, already ours, or already bound to a real fast package
                ref = inspect.unwrap(fn)  # transformers' torch reference, the wrapper's own fallback
                adapter = _conv_fn(conv, ref) if name == "causal_conv1d_fn" else _conv_update(conv, ref)
                _ROUTED[id(adapter)] = adapter
                setattr(mod, name, adapter)


def _by_device(fast: Any, ref: Any) -> Any:
    """`fast` for a CUDA input, `ref` otherwise (meta and CPU cannot run Triton)."""

    @functools.wraps(fast)
    def routed(*args: Any, **kwargs: Any) -> Any:
        dev = next((v.device for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
        return fast(*args, **kwargs) if dev is not None and dev.type == "cuda" else ref(*args, **kwargs)

    return routed


def _on_cuda(t: torch.Tensor) -> bool:
    """Whether this tensor's conv runs on a card (a test can pin it to exercise the adapter's mapping)."""
    return t.device.type == "cuda"


def _conv_refused(name: str, kwargs: dict[str, Any]) -> None:
    """Refuse a conv argument whose meaning the adapter's return shape cannot carry, rather than drop it.

    Other keyword arguments are the model's own flags (`use_cache` and the like) forwarded down the layer;
    the torch reference accepts and ignores them, and they are passed to fla the same way.
    """
    if "output_final_state" in kwargs:
        raise Refused(
            f"{name}: the adapter cannot return a final state (output_final_state); the modeling call site "
            "keeps its state in the cache instead"
        )


def _conv_activation(name: str, activation: str | None) -> None:
    if activation not in _FLA_CONV_ACTIVATIONS:
        raise Refused(
            f"{name}: fla's Triton causal conv applies only {'/'.join(map(str, _FLA_CONV_ACTIVATIONS))}, "
            f"not {activation!r}; refusing rather than run a different nonlinearity"
        )


def _conv_fn(fast: Any, ref: Any) -> Any:
    """transformers' ``causal_conv1d_fn`` over fla's Triton causal conv on CUDA, the reference elsewhere.

    transformers calls it with ``hidden_states`` ``[B, D, T]``; fla takes ``x`` ``[B, T, D]``. fla's
    ``initial_state`` is the same ``[N, D, W]`` window transformers keeps in a cache's conv state, and its
    left-padded kernel is the reference's ``F.conv1d(..., padding=W - 1)[..., :T]``.
    """

    def causal_conv1d_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        *,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _conv_refused("causal_conv1d_fn", kwargs)
        if seq_idx is not None:
            raise Refused(
                "causal_conv1d_fn: fla's Triton causal conv cannot keep packed sequences apart (seq_idx); "
                "refusing rather than drop it"
            )
        if cu_seq_lens_q is not None:
            if cu_seqlens is None:
                raise Refused(
                    "causal_conv1d_fn: the adapter cannot honour cu_seq_lens_q without cu_seqlens; "
                    "refusing rather than drop it"
                )
            if not torch.equal(torch.as_tensor(cu_seq_lens_q), torch.as_tensor(cu_seqlens)):
                raise Refused(
                    "causal_conv1d_fn: cu_seq_lens_q and cu_seqlens disagree; refusing rather than pick one"
                )
        if not _on_cuda(hidden_states):
            if initial_state is not None:
                raise Refused(
                    "causal_conv1d_fn: the torch reference takes no initial state; refusing rather than "
                    "drop it"
                )
            if cu_seqlens is not None or cu_seqlens_cpu is not None or chunk_indices is not None:
                raise Refused(
                    "causal_conv1d_fn: the torch reference cannot keep packed sequences apart "
                    "(cu_seqlens); refusing rather than drop it"
                )
            return ref(hidden_states, weight, bias, activation=activation, **kwargs)
        _conv_activation("causal_conv1d_fn", activation)
        out, _final = fast.causal_conv1d(
            x=hidden_states.transpose(1, 2),
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            **kwargs,
        )
        return out.transpose(1, 2)

    causal_conv1d_fn._qda_fast_on_cuda = True  # what `qda_quant.checks` reads: this name is fast on a card
    return causal_conv1d_fn


def _conv_update(fast: Any, ref: Any) -> Any:
    """transformers' ``causal_conv1d_update`` over fla's Triton per-step kernel on CUDA.

    transformers passes ``hidden_states`` ``[B, D, 1]`` and a ``conv_state`` ``[B, D, W]`` it expects updated
    in place; fla's update takes ``[B, 1, D]`` and the same ``[B, D, W]`` cache and writes it in place.
    """

    def causal_conv1d_update(
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _conv_refused("causal_conv1d_update", kwargs)
        if kwargs:  # its modeling call site passes none, and neither the reference nor fla takes any
            raise Refused(
                f"causal_conv1d_update: the adapter cannot pass {', '.join(sorted(kwargs))} to the per-step "
                "kernel; refusing rather than drop it"
            )
        if not _on_cuda(hidden_states):
            return ref(hidden_states, conv_state, weight, bias, activation)
        _conv_activation("causal_conv1d_update", activation)
        out, cache = fast.causal_conv1d_update(
            hidden_states.transpose(1, 2), conv_state, weight=weight, bias=bias, activation=activation
        )
        if cache is not conv_state:  # fla returns the cache it was given; keep the caller's view current
            conv_state.copy_(cache)
        return out.transpose(1, 2)

    causal_conv1d_update._qda_fast_on_cuda = True  # what `qda_quant.checks` reads: fast on a card
    return causal_conv1d_update


@torch.no_grad()
def trace(model: nn.Module) -> _Trace:
    """Run `_TRACE_TOKENS` tokens through `model` with a hook on every module.

    Raises:
        Refused: the forward uses an operation the trace's device does not implement, so the graph is unknown.
    """
    tr = _Trace()
    handles = [m.register_forward_hook(tr.hook(k)) for k, m in model.named_modules() if k]
    banks = [m for m in model.modules() if _expert_stack(m)]
    for m in banks:  # routing is data-dependent (no meta kernel); derivation only needs the bank's input
        m.forward = lambda h, *_a, **_k: torch.zeros_like(h)
    try:
        dev = next(model.parameters()).device
        ids = torch.arange(1, _TRACE_TOKENS + 1, device=dev)[None]
        try:
            with _meta_nonzero(), _meta_scalar():
                # No mask: forwards take a 2-D padding mask or a 4-D causal one (transformers has both
                # conventions), and this trace reads module calls and tensor identity, never values. Passing
                # no mask is the one input every forward accepts and it keeps the hidden states' identity.
                model(input_ids=ids, attention_mask=None, use_cache=False)
        except NotImplementedError as e:
            raise Refused(
                f"{type(model).__name__}: the forward uses an operation {dev.type} tensors do not implement, "
                f"so the graph cannot be derived from it: {e}"
            ) from e
        except Refused:
            raise
        except Exception as e:  # noqa: BLE001 - the model's own forward failed; refuse it, do not traceback
            raise probe_refusal(model, "probing its forward", e) from e
    finally:
        for h in handles:
            h.remove()
        for m in banks:
            del m.forward
    return tr
