"""Accumulate and collect dense and expert statistics for a layer."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from ..arch import ExpertBank, LayerGraph
from ..step import Refused
from .token_weights import BatchWeights

if TYPE_CHECKING:
    from ..solve.units import ExpertShare

DEFAULT_INPUT_ROWS = 8192


def gram(x: torch.Tensor) -> torch.Tensor:
    """``x^T x`` in fp32 for x [..., R, C]: bf16 tensor cores with fp32 output on CUDA, fp32 elsewhere."""
    xt = x.transpose(-1, -2)
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        return (torch.bmm if x.dim() == 3 else torch.mm)(xt, x, out_dtype=torch.float32)
    x = x.float()
    return xt.float() @ x


HessianDtype = Literal["bf16", "tf32", "fp32"]


@dataclass(frozen=True)
class HessianSetting:
    """How the engine accumulates Hessians (recorded in build.json as ``hessian``)."""

    dtype: Literal["bf16", "tf32", "fp32"] = "bf16"
    buffer_rows: int = 0


DEFAULT_HESSIAN = HessianSetting("bf16", 0)


FAST_HESSIAN = HessianSetting("bf16", 8192)


def hessian_setting(h: HessianSetting | dict | str | None) -> HessianSetting:
    """Validate a Hessian setting: a `HessianSetting`, a dict of its fields, ``"fast"``, or None (the default)."""
    if h is None or h == "default":
        return DEFAULT_HESSIAN
    if h == "fast":
        return FAST_HESSIAN
    if isinstance(h, dict):
        unknown = set(h) - {"dtype", "buffer_rows"}
        if unknown:
            raise Refused(f"hessian: unknown keys {sorted(unknown)}; expected dtype, buffer_rows")
        h = HessianSetting(**{**DEFAULT_HESSIAN.__dict__, **h})
    elif not isinstance(h, HessianSetting):
        raise Refused(f"hessian {h!r}: expected fast, default or {{dtype, buffer_rows}}")
    if h.dtype not in ("bf16", "tf32", "fp32"):
        raise Refused(f"hessian dtype {h.dtype!r}: expected bf16, tf32 or fp32")
    if not isinstance(h.buffer_rows, int) or isinstance(h.buffer_rows, bool) or h.buffer_rows < 0:
        raise Refused(f"hessian buffer_rows {h.buffer_rows!r}: expected an integer >= 0 (0 = per sample)")
    return h


@contextlib.contextmanager
def _fp32_matmul(precision: str) -> Iterator[None]:
    b = torch.backends.cuda.matmul
    old = b.fp32_precision
    b.fp32_precision = precision
    try:
        yield
    finally:
        b.fp32_precision = old


def accumulate_fp32(
    H: torch.Tensor | None, x: torch.Tensor, w: torch.Tensor | None, tf32: bool
) -> torch.Tensor:
    """``H += x^T diag(w) x`` with fp32 inputs (llm-compressor's arithmetic); TF32 tensor cores if `tf32`."""
    x = x.float()
    xw = x if w is None else x * w.float().unsqueeze(-1)
    if H is None:
        H = torch.zeros(*x.shape[:-2], x.shape[-1], x.shape[-1], dtype=torch.float32, device=x.device)
    with _fp32_matmul("tf32" if tf32 else "ieee"):
        return H.add_(x.transpose(-1, -2) @ xw)


def accumulate(H: torch.Tensor | None, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """``H += x^T diag(w) x`` (fp32 H; x [R, C] or [E, R, C]; optional row weights w [R] or [E, R])."""
    cuda = x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
    if w is None:
        if H is None:
            return gram(x)
        if cuda:
            f = torch.baddbmm if x.dim() == 3 else torch.addmm
            return f(H, x.transpose(-1, -2), x, out_dtype=torch.float32, out=H)
        return H.add_(gram(x))
    xw = x.float() * w.float().unsqueeze(-1)
    if H is None:
        H = torch.zeros(*x.shape[:-2], x.shape[-1], x.shape[-1], dtype=torch.float32, device=x.device)
    xt = x.transpose(-1, -2)
    if not cuda:
        return H.add_(xt.float() @ xw)
    hi = xw.to(x.dtype)
    lo = (xw - hi.float()).to(x.dtype)
    f = torch.baddbmm if x.dim() == 3 else torch.addmm
    f(H, xt, hi, out_dtype=torch.float32, out=H)
    return f(H, xt, lo, out_dtype=torch.float32, out=H)


def symmetrize_(H: torch.Tensor) -> torch.Tensor:
    """``H = (H + H^T) / 2`` in place (after one-sided weighted accumulation)."""
    return H.copy_((H + H.transpose(-1, -2)) * 0.5)


@dataclass
class GroupStats:
    """Accumulators for one input (a `LinearGroup`, or one input of an expert bank as a stack)."""

    C: int
    device: torch.device
    want: frozenset[str]
    E: int | None = None
    input_stride: int = 1
    input_rows: int | None = DEFAULT_INPUT_ROWS  # None: every row
    flush_rows: int = 0  # buffered input rows per GEMM; 0 = one GEMM per add() call (one sample)
    dtype: HessianDtype = "bf16"
    H: torch.Tensor | None = None
    cross: torch.Tensor | None = None
    absmax: torch.Tensor | None = None
    inputs: list[torch.Tensor] = field(default_factory=list)
    _pend: list[tuple[torch.Tensor, torch.Tensor | None]] = field(default_factory=list)
    _pend_rows: int = 0
    weighted: bool = False
    n: int | torch.Tensor = 0
    enabled: bool = True

    def _lead(self) -> tuple[int, ...]:
        return () if self.E is None else (self.E,)

    def add(self, x: torch.Tensor, w: torch.Tensor | None = None, xf: torch.Tensor | None = None) -> None:
        """Dense: x [..., C] rows of the group's input; `w` optional per-row weights (token weights)."""
        x = x.detach().reshape(-1, self.C)
        self.n += x.shape[0]
        if "act_absmax" in self.want:
            a = x.abs().amax(0).float()
            self.absmax = a if self.absmax is None else torch.maximum(self.absmax, a)
        if "inputs" in self.want and (
            self.input_rows is None or sum(t.shape[0] for t in self.inputs) < self.input_rows
        ):
            self.inputs.append(x[:: self.input_stride].clone())
        if "hessian" in self.want:
            self._pend.append((x, None if w is None else w.reshape(-1)))
            self._pend_rows += x.shape[0]
            if self._pend_rows >= max(self.flush_rows, 1):
                self._flush()
        if "cross_hessian" in self.want and xf is not None:
            xq = x.float()
            xf = xf.detach().reshape(-1, self.C).float()
            if xf.shape != xq.shape:
                raise RuntimeError("cross_hessian float and quantized inputs have different shapes")
            c = (xf - xq).transpose(-1, -2) @ (xq * (1.0 if w is None else w.reshape(-1, 1).float()))
            self.cross = c if self.cross is None else self.cross + c

    def _flush(self) -> None:
        if self._pend:
            xs, ws = zip(*self._pend, strict=True)
            x = xs[0] if len(xs) == 1 else torch.cat(xs)
            w = None if ws[0] is None else (ws[0] if len(ws) == 1 else torch.cat(ws))
            self.H = self._acc(self.H, x, w)
            self.weighted |= w is not None
            self._pend, self._pend_rows = [], 0

    def _acc(self, H: torch.Tensor | None, x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        if self.dtype == "bf16":
            return accumulate(H, x, w)
        return accumulate_fp32(H, x, w, tf32=self.dtype == "tf32")

    def add_stacked(self, xp: torch.Tensor, counts: torch.Tensor, w: torch.Tensor | None = None) -> None:
        """Bank: xp [E, Cmax, C] padded per-expert rows (zero padding), counts [E]; `w` [E, Cmax] row weights."""
        self.n = counts.clone() if isinstance(self.n, int) else self.n + counts
        if "act_absmax" in self.want:
            a = xp.abs().amax(1).float()
            self.absmax = a if self.absmax is None else torch.maximum(self.absmax, a)
        if "hessian" in self.want:
            self.H = self._acc(self.H, xp, w)
            self.weighted |= w is not None

    def result(self) -> dict[str, object]:
        """{stat name: value} with zeros where nothing was seen."""
        self._flush()
        if self.weighted and self.H is not None:
            symmetrize_(self.H)
            self.weighted = False
        out: dict[str, object] = {"n_tokens": self.n}
        if "hessian" in self.want:
            out["hessian"] = (
                self.H
                if self.H is not None
                else torch.zeros(*self._lead(), self.C, self.C, device=self.device, dtype=torch.float32)
            )
        if "cross_hessian" in self.want:
            if self.cross is None and self.n:
                raise RuntimeError(
                    "cross_hessian: rows reached this group but no float-model input did; the paired float "
                    "stream is missing (refusing to return C = 0, which would silently turn GPTAQ into GPTQ)"
                )
            out["cross_hessian"] = (
                self.cross
                if self.cross is not None
                else torch.zeros(self.C, self.C, device=self.device, dtype=torch.float32)
            )
        if "act_absmax" in self.want:
            out["act_absmax"] = (
                self.absmax
                if self.absmax is not None
                else torch.zeros(*self._lead(), self.C, device=self.device, dtype=torch.float32)
            )
        if "inputs" in self.want:
            # a bounded sample is one tensor; every row stays the batches it arrived in, because concatenating
            # a whole model's layer inputs beside its weights is what makes a parent-objective AWQ build run out
            # of memory (`steps.awq` searches chunk by chunk; `planning.memory_terms` counts the chunks)
            out["inputs"] = (
                None
                if not self.inputs
                else list(self.inputs)
                if self.input_rows is None
                else torch.cat(self.inputs)[: self.input_rows]
            )
        return out


def input_chunks(inputs: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """The retained input rows as the batches they arrived in (a bounded sample is one chunk)."""
    return inputs if isinstance(inputs, list) else [inputs]


def input_stride(total_rows: int, max_rows: int | None = DEFAULT_INPUT_ROWS) -> int:
    """Stride that keeps at most about `max_rows` of `total_rows` input rows, evenly spread (None: 1, all)."""
    return 1 if max_rows is None else max(1, math.ceil(total_rows / max_rows))


def input_rows_of(steps) -> int | None:  # noqa: ANN001
    """The input-sample bound of a pass."""
    rows = [getattr(s, "input_rows", DEFAULT_INPUT_ROWS) for s in steps if "inputs" in set(s.needs)]
    if not rows:
        return DEFAULT_INPUT_ROWS
    return None if any(r is None for r in rows) else max(rows)


#: the bank statistics of GPTQ's all-token blend, which every card of a split solve reads
ALL_TOKEN_HESSIAN = ("all_in", "all_down", "all_tokens_seen")


@dataclass
class ExpertStats:
    """Both inputs of one bank of fused experts: gate/up read the routed x, down reads act(gate) * up."""

    inp: GroupStats
    down: GroupStats
    first: int = 0
    router_weighted: bool = False
    enabled: bool = True
    flush_rows: int = 0  # routed (token, expert) rows per padded stack; 0 = one stack per call (one sample)
    all_tokens: bool = False
    all_down: GroupStats | None = None
    #: the gate/up input Hessian over every token (``all_token_hessian``); `all_down` then holds this span's experts
    all_in: GroupStats | None = None
    input_abs_sum: torch.Tensor | None = None
    down_abs_sum: torch.Tensor | None = None
    all_tokens_seen: int = 0
    inputs: list[torch.Tensor] = field(default_factory=list)
    _pend: list[tuple[torch.Tensor, ...]] = field(default_factory=list)
    _rows: int = 0
    _token_weighted: bool = False

    def add(
        self,
        e: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        token_weights: torch.Tensor | None = None,
        down_token_weights: torch.Tensor | None = None,
        *,
        counts: torch.Tensor,
    ) -> None:
        """Buffer routed rows with router weights and separate input/down token weights."""
        if down_token_weights is None:
            down_token_weights = token_weights
        if token_weights is not None or down_token_weights is not None:
            self._token_weighted = True
        g = g.float() if self.router_weighted else torch.ones_like(g, dtype=torch.float32)
        input_weights = g if token_weights is None else g * token_weights.float()
        down_weights = g if down_token_weights is None else g * down_token_weights.float()
        self._pend.append((e, x, h, input_weights, down_weights, counts))
        self._rows += int(counts.sum())
        if self._rows >= max(self.flush_rows, 1):
            self._flush()

    def _flush(self) -> None:
        if not self._pend:
            return
        # the whole bank's widest expert, whatever span this holds
        cmax = int(torch.stack([p[-1] for p in self._pend]).sum(0).max())
        e, x, h, input_weights, down_weights = (
            torch.cat(t) if len(self._pend) > 1 else t[0]
            for t in zip(*(p[:-1] for p in self._pend), strict=True)
        )
        self._pend, self._rows = [], 0
        assert self.inp.E is not None  # an expert bank always knows its expert count
        E = self.inp.E
        order, e, pos, counts, _ = route(e, E)
        x, h = x[order], h[order]
        xp = x.new_zeros(E, cmax, x.shape[-1])
        xp[e, pos] = x
        hp = h.new_zeros(E, cmax, h.shape[-1])
        hp[e, pos] = h
        if self.router_weighted or self._token_weighted:
            input_padded = input_weights.new_zeros(E, cmax)
            input_padded[e, pos] = input_weights[order]
            down_padded = down_weights.new_zeros(E, cmax)
            down_padded[e, pos] = down_weights[order]
            self.inp.add_stacked(xp, counts, input_padded)
            self.down.add_stacked(hp, counts, down_padded)
        else:
            self.inp.add_stacked(xp, counts)
            self.down.add_stacked(hp, counts)

    def result(self) -> dict[str, object]:
        """Routed Hessians and, when requested, all-token AWQ statistics."""
        self._flush()
        a, b = self.inp.result(), self.down.result()
        out: dict[str, object] = {k: {"in": a[k], "down": b[k]} for k in a if k != "n_tokens"}
        out["n_tokens"] = a["n_tokens"]
        if self.all_tokens:
            out["input_abs_sum"] = (
                self.input_abs_sum
                if self.input_abs_sum is not None
                else torch.zeros(self.inp.C, device=self.inp.device)
            )
            if self.down_abs_sum is not None:
                out["down_abs_sum"] = self.down_abs_sum
            else:
                assert self.down.E is not None  # an expert group always knows its expert count
                out["down_abs_sum"] = torch.zeros(self.down.E, self.down.C, device=self.down.device)
            out["all_tokens_seen"] = self.all_tokens_seen
            out["inputs"] = torch.cat(self.inputs) if self.inputs else None
            if self.all_down is not None:
                out["all_down"] = self.all_down.result()["hessian"]
        if self.all_in is not None:
            assert self.all_down is not None  # built together (`LayerStats._bank_stats`)
            out["all_in"] = self.all_in.result()["hessian"]
            out["all_tokens_seen"] = self.all_tokens_seen
            out.setdefault("all_down", self.all_down.result()["hessian"])
        return out


def route(
    top_k_index: torch.Tensor, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sort (token, slot) assignments by expert."""
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(flat.numel(), device=flat.device) - starts[e_sorted]
    return order, e_sorted, pos, counts, int(counts.max())


def observe_experts(
    experts: torch.nn.Module,
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    stats: ExpertStats,
    token_weights: torch.Tensor | None = None,
    down_token_weights: torch.Tensor | None = None,
    all_tokens: bool = False,
    cut: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Feed `stats` with one call's routed rows of a fused experts module, without computing its output.

    `x` is the bank's input as the kernel reads it: for a plan that reads a cut bank (NVFP4's, `cut` below)
    the hook (`engine.execute.act_inputs`) has already replaced it; for every other act dtype `cut` is None
    and neither input is cut. `cut` is the same round trip, applied to the down input this function forms —
    the kernel quantizes it too, but the module forms it inside its own forward, so the propagated stream
    keeps the uncut one (`engine.runner.LayerStats`).
    """
    from ..arch.adapters import fused_experts
    from ..arch.adapters.fused_experts import down_input

    Wgu = fused_experts.in_weight(experts)  # [E, R, D], whatever the module stores (R = 2I gated, I not)
    bgu = fused_experts.bias(experts, fused_experts.in_param(experts))
    E, k = Wgu.shape[0], top_k_index.shape[-1]
    if all_tokens or stats.all_in is not None:
        if stats.all_down is None:
            raise ValueError("all-token expert statistics need an all-down accumulator")
        token_rows = x.detach().reshape(-1, x.shape[-1])
        stats.all_tokens_seen += token_rows.shape[0]
        if stats.all_in is not None:
            stats.all_in.add(token_rows)
        if all_tokens:
            stats.inputs.append(token_rows.clone())
            abs_sum = token_rows.float().abs().sum(0)
            stats.input_abs_sum = abs_sum if stats.input_abs_sum is None else stats.input_abs_sum + abs_sum
        # AWQ's statistics cover the whole bank; GPTQ's all-token Hessians cover this span's experts
        lo_all, span_all = (0, E) if all_tokens else (stats.first, cast(int, stats.all_down.E))
        w_all = Wgu[lo_all : lo_all + span_all]
        b_all = None if bgu is None else bgu[lo_all : lo_all + span_all]
        down_sum = torch.zeros(
            span_all, fused_experts.down_weight(experts).shape[-1], dtype=torch.float32, device=x.device
        )
        chunk_rows = max(1, min(token_rows.shape[0], 128))
        for start in range(0, token_rows.shape[0], chunk_rows):
            rows = token_rows[start : start + chunk_rows]
            expanded = rows.unsqueeze(0).expand(span_all, -1, -1)
            projected = torch.bmm(expanded, w_all.transpose(1, 2))
            if b_all is not None:
                projected = projected + b_all.unsqueeze(1)
            h = down_input(experts, projected, None)
            if cut is not None:
                h = cut(h)
            down_sum += h.float().abs().sum(1)
            stats.all_down.add_stacked(
                h,
                torch.full((span_all,), rows.shape[0], dtype=torch.long, device=x.device),
            )
        if all_tokens:
            stats.down_abs_sum = down_sum if stats.down_abs_sum is None else stats.down_abs_sum + down_sum
    order, e_s, pos, counts, cmax = route(top_k_index, E)
    lo, span = stats.first, cast(int, stats.inp.E)
    if span != E:  # this span's rows only, at the whole bank's padded width
        mine = (e_s >= lo) & (e_s < lo + span)
        order, e_s, pos = order[mine], e_s[mine] - lo, pos[mine]
    tok = order // k
    xr = x[tok]
    xp = x.new_zeros(span, cmax, x.shape[-1])
    xp[e_s, pos] = xr
    g = top_k_weights.reshape(-1)[order]
    gp = g.new_zeros(span, cmax)
    gp[e_s, pos] = g
    projected = torch.bmm(xp, Wgu[lo : lo + span].transpose(1, 2))
    if bgu is not None:
        projected = projected + bgu[lo : lo + span].unsqueeze(1)
    h = down_input(experts, projected, gp)
    if cut is not None:
        h = cut(h)
    stats.add(
        e_s,
        xr,
        h[e_s, pos],
        g,
        None if token_weights is None else token_weights.reshape(-1)[tok],
        None if down_token_weights is None else down_token_weights.reshape(-1)[tok],
        counts=counts,
    )


def reference_expert_stats(
    experts: torch.nn.Module,
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    router_weighted: bool = False,
    cut: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slow reference: per-expert loop, fp32. Returns H_in [E, D, D], H_down [E, I, I], counts [E].

    `cut`: the kernel's input round trip (`observe_experts`'s), applied to the down input.
    """
    from ..arch.adapters import fused_experts
    from ..arch.adapters.fused_experts import down_input

    Wgu = fused_experts.in_weight(experts)
    bgu = fused_experts.bias(experts, fused_experts.in_param(experts))
    E, D, Ii = Wgu.shape[0], x.shape[-1], fused_experts.down_weight(experts).shape[-1]
    Hin = torch.zeros(E, D, D, dtype=torch.float32, device=x.device)
    Hd = torch.zeros(E, Ii, Ii, dtype=torch.float32, device=x.device)
    counts = torch.zeros(E, dtype=torch.long, device=x.device)
    for e in range(E):
        t, s = torch.where(top_k_index == e)
        if t.numel() == 0:
            continue
        xe = x[t]
        projected = torch.nn.functional.linear(xe, Wgu[e])
        if bgu is not None:
            projected = projected + bgu[e]
        he = down_input(experts, projected, top_k_weights[t, s])
        if cut is not None:
            he = cut(he)
        w = top_k_weights[t, s].float()[:, None] if router_weighted else 1.0
        Hin[e] = (xe.float() * w).T @ xe.float()
        Hd[e] = (he.float() * w).T @ he.float()
        counts[e] = t.numel()
    return {"in": Hin, "down": Hd, "n_tokens": counts}


def scale_input_stats(ctx, group, s: torch.Tensor) -> None:  # noqa: ANN001
    """Update `group`'s statistics in `ctx` after its input became ``x / s`` (an AWQ / smoothing fold)."""
    from ..step import _key

    key = _key(group)
    s = s.reshape(-1).to(torch.float32)
    for name in ("hessian", "cross_hessian"):
        if (name, key) in ctx.stats:
            statistic = ctx.stats[(name, key)]
            scales = s.to(statistic.device)
            statistic.div_(scales[:, None] * scales[None, :])
    if ("act_absmax", key) in ctx.stats:
        a = ctx.stats[("act_absmax", key)]
        a.div_(s.abs().to(a.device))
    if (X := ctx.stats.get(("inputs", key))) is not None:
        dev = X[0].device if isinstance(X, list) else X.device
        scale = s.to(dev)
        if isinstance(X, list):  # every retained batch (the parent objective keeps them apart)
            ctx.stats[("inputs", key)] = [(c.float() / scale).to(c.dtype) for c in X]
        else:
            ctx.stats[("inputs", key)] = (X.float() / scale).to(X.dtype)


def group_hook(
    st: GroupStats,
    weights: Callable[[], torch.Tensor | None] | None = None,
    float_inputs: Callable[[], torch.Tensor | None] | None = None,
):  # noqa: ANN201
    """A forward pre-hook feeding `st` with the module's first input."""

    def hook(_mod, args, kwargs):  # noqa: ANN001, ANN202
        if not st.enabled:
            return
        x = args[0] if args else next(iter(kwargs.values()))
        st.add(
            x,
            None if weights is None else weights(),
            None if float_inputs is None else float_inputs(),
        )

    return hook


DENSE_STATS = frozenset({"hessian", "cross_hessian", "act_absmax", "inputs"})


def rel(name: str, prefix: str) -> str:
    """`name` relative to the module `prefix` (``model.layers.3.self_attn.q_proj`` -> ``self_attn.q_proj``)."""
    return name[len(prefix) + 1 :] if name.startswith(prefix + ".") else name


class LayerStats:
    """Statistics hooks of one replica of one layer, and the fused expert forwards."""

    def __init__(
        self,
        layer: torch.nn.Module,
        lg: LayerGraph,
        needs: frozenset[str],
        device: torch.device,
        stride: int = 1,
        router_weighted: bool = False,
        hessian: HessianSetting = DEFAULT_HESSIAN,
        input_rows: int | None = DEFAULT_INPUT_ROWS,
        weights: Callable[[], BatchWeights | None] | None = None,
        all_token_experts: bool = False,
        share: ExpertShare | None = None,
        all_token_hessian: bool = False,
        cut: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.layer, self.lg = layer, lg
        self.device = device
        self.share = share
        #: the kernel's input round trip, for a fused bank's down input (the one its forward forms itself);
        #: None unless the plan reads a cut bank (NVFP4, `engine.execute.bank_cut`)
        self.cut = cut
        #: each fused bank call's inputs, kept for `replay` when the statistics cover one share of the experts
        self._calls: dict[str, list[tuple[torch.Tensor | None, ...]]] | None = None
        self.needs = needs
        self.groups: dict[str, GroupStats] = {}
        self.banks: dict[str, ExpertStats | list[tuple[GroupStats, GroupStats]]] = {}
        self._handles: list = []
        self._patched: list[torch.nn.Module] = []
        self._weights = weights
        self.weights: BatchWeights | None = None
        self.float_inputs: dict[str, torch.Tensor] = {}
        dense = needs & DENSE_STATS
        bank_mods = [b.experts for b in lg.banks if isinstance(b.experts, str)]
        for g in lg.groups:
            if any(m == bm or m.startswith(bm + ".") for m in g.modules for bm in bank_mods):
                continue
            if dense:
                mod = self._mod(g.modules[0])
                self.groups[g.modules[0]] = GroupStats(
                    _input_features(mod),
                    device,
                    dense,
                    input_stride=stride,
                    input_rows=input_rows,
                    dtype=hessian.dtype,
                    flush_rows=hessian.buffer_rows,
                )
        bwant = frozenset({"hessian"} if "expert_hessian" in needs else set()) | (needs & {"act_absmax"})
        if router_weighted and not all(self._fused(b) for b in lg.banks):
            raise NotImplementedError("router-weighted Hessians need a fused expert bank")
        if all_token_hessian and not all(self._fused(b) for b in lg.banks):
            raise NotImplementedError("all-token expert Hessians need a fused expert bank")
        self._bank_args = (bwant, router_weighted, hessian, all_token_experts, all_token_hessian)
        if share and all(self._fused(b) for b in lg.banks):
            self._calls = {}
        for b in lg.banks:
            if bwant:
                self.banks[b.router] = self._bank_stats(b, share)

    def _bank_stats(
        self, b: ExpertBank, share: ExpertShare | None
    ) -> ExpertStats | list[tuple[GroupStats, GroupStats]]:
        want, router_weighted, hessian, all_tokens, all_hessian = self._bank_args
        d, hd = self.device, hessian.dtype
        if not self._fused(b):
            pairs = b.experts if share is None else b.experts[share.experts(len(b.experts))]
            rows = hessian.buffer_rows
            return [
                (
                    GroupStats(_input_features(self._mod(ins[0])), d, want, dtype=hd, flush_rows=rows),
                    GroupStats(_input_features(self._mod(out)), d, want, dtype=hd, flush_rows=rows),
                )
                for ins, out in pairs
            ]
        from ..arch.adapters import fused_experts

        ex = self._mod(cast(str, b.experts))
        # a bank may store [E, in, out] (transposed): the canonical accessors give [E, out, in]
        E, _rows, D = fused_experts.in_weight(ex).shape
        Ii = fused_experts.down_weight(ex).shape[-1]
        span = share.experts(E) if share else slice(0, E)
        n = span.stop - span.start
        return ExpertStats(
            GroupStats(D, d, want, E=n, dtype=hd),
            GroupStats(Ii, d, want, E=n, dtype=hd),
            first=span.start,
            router_weighted=router_weighted,
            flush_rows=hessian.buffer_rows * b.top_k,
            all_tokens=all_tokens,
            all_down=(
                GroupStats(Ii, d, frozenset({"hessian"}), E=E if all_tokens else n, dtype=hd)
                if all_tokens or all_hessian
                else None
            ),
            all_in=GroupStats(D, d, frozenset({"hessian"}), dtype=hd) if all_hessian else None,
        )

    def replay(
        self, share: ExpertShare, forward: Callable[[LayerStats], dict[tuple[str, str], Any]]
    ) -> dict[tuple[str, str], Any]:
        """The banks' statistics for the experts of `share`."""
        self.groups, self.banks, self.needs = {}, {}, self.needs & {"expert_hessian"}
        self.share = share
        for b in self.lg.banks:
            self.banks[b.router] = self._bank_stats(b, share)
        if self._calls is None:
            return forward(self)
        for b in self.lg.banks:
            st, ex = cast(ExpertStats, self.banks[b.router]), self._mod(cast(str, b.experts))
            for call in self._calls[b.router]:
                x, idx, w, tw, dtw = (
                    None if t is None else t.to(self.device, non_blocking=True) for t in call
                )
                assert x is not None and idx is not None and w is not None  # recorded from a real call
                observe_experts(ex, x, idx, w, st, tw, dtw, cut=self.cut)
        return self.result()

    def _fused(self, b: ExpertBank) -> bool:
        from ..arch.adapters import fused_experts

        return isinstance(b.experts, str) and fused_experts.is_bank(self._mod(b.experts))

    def _mod(self, name: str) -> torch.nn.Module:
        return self.layer.get_submodule(rel(name, self.lg.module))

    def __enter__(self) -> LayerStats:
        for key, st in self.groups.items():
            self._handles.append(
                self._mod(key).register_forward_pre_hook(
                    group_hook(
                        st,
                        lambda key=key: self._group_weights(key),
                        lambda key=key: self.float_inputs.get(key),
                    ),
                    with_kwargs=True,
                )
            )
        for b in self.lg.banks:
            st = self.banks.get(b.router)
            if self._fused(b):
                assert isinstance(b.experts, str)
                if isinstance(st, ExpertStats):
                    self._observe(self._mod(b.experts), st, b.router, b.experts + ".down_proj")
            elif st:
                pairs = b.experts if self.share is None else b.experts[self.share.experts(len(b.experts))]
                for (ins, out), (si, so) in zip(pairs, st, strict=True):  # type: ignore[arg-type]
                    self._handles.append(
                        self._mod(ins[0]).register_forward_pre_hook(group_hook(si), with_kwargs=True)
                    )
                    self._handles.append(
                        self._mod(out).register_forward_pre_hook(group_hook(so), with_kwargs=True)
                    )
        return self

    def _group_weights(self, key: str) -> torch.Tensor | None:
        weights = self.weights if self._weights is None else self._weights()
        return weights[key] if isinstance(weights, dict) else weights

    def capture_float(self, forward: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Record paired inputs while statistics are disabled; always remove the temporary hooks."""
        accumulators: list[GroupStats | ExpertStats] = list(self.groups.values())
        for bank_stats in self.banks.values():
            if isinstance(bank_stats, ExpertStats):
                accumulators.append(bank_stats)
            else:
                accumulators.extend(stat for pair in bank_stats for stat in pair)
        flags = [(stat, stat.enabled) for stat in accumulators]
        handles = []
        self.float_inputs = {}

        def hook(key: str) -> Callable[..., None]:
            def record(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
                value = args[0] if args else next(iter(kwargs.values()))
                self.float_inputs[key] = value.detach()

            return record

        try:
            for stat, _ in flags:
                stat.enabled = False
            for key in self.groups:
                handles.append(self._mod(key).register_forward_pre_hook(hook(key), with_kwargs=True))
            return forward()
        finally:
            for handle in handles:
                handle.remove()
            for stat, flag in flags:
                stat.enabled = flag

    def _observe(self, ex: torch.nn.Module, st: ExpertStats, key: str, down_key: str) -> None:
        own = ex.forward

        def fwd(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            if st.enabled:
                tw, dtw = self._group_weights(key), self._group_weights(down_key)
                observe_experts(ex, x, idx, w, st, tw, dtw, st.all_tokens, self.cut)
                if self._calls is not None:
                    self._calls.setdefault(key, []).append(tuple(map(_pinned, (x, idx, w, tw, dtw))))
            return own(x, idx, w)

        ex.forward = fwd  # type: ignore[method-assign]
        self._patched.append(ex)

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        for ex in self._patched:
            del ex.forward
        self._handles, self._patched = [], []

    def result(self) -> dict[tuple[str, str], Any]:
        """The ``ctx.stats`` entries of this replica."""
        out: dict[tuple[str, str], Any] = {}
        for key, st in self.groups.items():
            for name, v in st.result().items():
                out[(name, key)] = v
        for router, st in self.banks.items():
            if isinstance(st, ExpertStats):
                r = st.result()
            else:
                ins, outs = [s.result() for s, _ in st], [s.result() for _, s in st]
                r: dict[str, object] = {
                    k: {
                        "in": torch.stack([cast(torch.Tensor, a[k]) for a in ins]),
                        "down": torch.stack([cast(torch.Tensor, a[k]) for a in outs]),
                    }
                    for k in ins[0]
                    if k != "n_tokens"
                }
                r["n_tokens"] = torch.tensor([a["n_tokens"] for a in ins], dtype=torch.long)
            for name, v in r.items():
                out[("expert_hessian" if name == "hessian" else name, router)] = v
        return out


def _pinned(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None or not t.is_cuda:
        return None if t is None else t.detach().clone()
    return torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t.detach(), non_blocking=True)


def _input_features(mod: torch.nn.Module) -> int:
    return int(getattr(mod, "in_features", 0) or cast(torch.Tensor, mod.weight).shape[-1])
