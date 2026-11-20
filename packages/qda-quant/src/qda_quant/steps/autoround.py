"""AutoRound (arXiv:2309.05516): sign-gradient descent on rounding offsets and range clips, per layer.

Every quantized weight of the layer gets a rounding offset ``V`` and per-group clip
factors ``(alpha, beta)`` in ``[min_clip, 1]`` on its min / max range. They are trained together with
signed-gradient steps (straight-through rounding) to make the quantized layer reproduce the float layer's
output on the calibration batches (`layer_io_float` by default, `layer_io` with ``target: quant_stream``). The best
iterate is kept. Training and export both use the shared quantizer's served grid.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.func import functional_call

from ..solve.units import Unit
from ..step import LayerContext
from ._attention import attention_kernel
from ._layer_fit import LayerFit


class AutoRound(LayerFit):
    """Sign-gradient rounding.

    Reproducibility: noise band by default (docs/reproduce.md): quote KL as a 3-seed mean and band. The
    attention backward varies in its last bits, and a sign step turns a rare flip near zero into a different
    trajectory. ``reproducible`` removes that source.

    Attributes:
        iters: optimizer steps per layer.
        lr: initial step size, linearly decayed to zero (default ``1 / iters``, as AutoRound).
        min_clip: lower bound of the per-group range factors.
        batches: calibration batches per step (cycled).
        target: the output to reproduce; inputs are the quantized stream either way. ``float_stream``
            (default): the float layer on the float stream, auto-round's target, so each layer also corrects
            the error it inherits; the engine keeps the float stream. ``quant_stream``: the float layer on the
            quantized stream, so each layer fits its local map only.
        reproducible: train with torch's math attention instead of whatever backends the build runs, so a
            repeat build writes the same bytes. Training takes about 1.7x as long and needs more card memory
            (docs/reproduce.md).
    """

    name = "autoround"
    default_grid = "signed8"  # auto-round's grid; where it cannot serve, the default (`QuantStep.resolve`)
    iters: int = 200
    lr: float | None = None
    min_clip: float = 0.0
    batches: int = 1
    target: Literal["quant_stream", "float_stream"] = "float_stream"
    reproducible: bool = False

    def apply_layer(self, ctx: LayerContext) -> None:
        """Train offsets and clips for every unit of the layer, then write codes and served weights."""
        io = self.layer_io(ctx)
        if not io:
            return
        units = list(self.units(ctx))
        layer = ctx.layer
        names = {id(p): n for n, p in layer.named_parameters()}
        lr = self.lr or 1.0 / self.iters
        state = []
        for u in units:
            q = u.quantizer
            w = u.weight().float().detach()
            g = q.group(w.shape[-1])
            wg = w.reshape(*w.shape[:-1], -1, g)
            lo, hi = wg.amin(-1).clamp(max=0), wg.amax(-1).clamp(min=0)
            v = torch.zeros_like(w, requires_grad=True)
            a = torch.ones_like(lo, requires_grad=True)
            b = torch.ones_like(hi, requires_grad=True)
            state.append((u, q, w, g, lo, hi, v, a, b))
        params = [t for st in state for t in st[6:]]
        best_loss, best = float("inf"), None
        with torch.enable_grad(), attention_kernel(self.reproducible):
            for it in range(self.iters):
                over = {}
                for u, q, w, g, lo, hi, v, a, b in state:
                    wq = _fake(q, w, g, lo * a, hi * b, v)
                    over.update(_split(u, wq, names))
                loss: torch.Tensor | None = None
                for k in range(self.batches):
                    h, args, kw, ref = io[(it * self.batches + k) % len(io)]
                    out = functional_call(layer, over, (h, *args), kw, strict=False)
                    out = out[0] if isinstance(out, tuple) else out
                    term = (out.float() - ref.float()).pow(2).mean()
                    loss = term if loss is None else loss + term
                assert loss is not None  # batches is at least one
                lv = float(loss)
                if lv < best_loss:
                    best_loss = lv
                    best = [
                        (v.detach().clone(), a.detach().clone(), b.detach().clone()) for *_, v, a, b in state
                    ]
                grads = torch.autograd.grad(loss * 1000, params)
                with torch.no_grad():
                    for p, gr in zip(params, grads, strict=True):
                        p.sub_(lr * (1.0 - it / self.iters) * gr.sign())
                    for *_, a, b in state:
                        a.clamp_(self.min_clip, 1.0)
                        b.clamp_(self.min_clip, 1.0)
        assert best is not None
        with torch.no_grad():
            for (u, q, w, g, lo, hi, *_), (v, a, b) in zip(state, best, strict=True):
                s, z, shifted = _params(q, w, g, lo * a, hi * b, v)
                codes = q.quantize(shifted, s, z)
                u.write(codes, s, z)


def _cols(t: torch.Tensor, g: int) -> torch.Tensor:
    return t.repeat_interleave(g, dim=-1)


def _params(quantizer, weight, group_size, lower, upper, offset):  # noqa: ANN001, ANN202
    """Shared range parameters and weights shifted by offsets in served-code units."""
    scale, zero = quantizer.range_params(lower, upper)
    shifted = weight + offset * _cols(quantizer.served_scale(scale), group_size)
    return scale, zero, shifted


def _fake(q, w, g, lo, hi, v):  # noqa: ANN001, ANN202
    """Shared fake-quant forward with straight-through gradients on bounded codes."""
    scale, zero, shifted = _params(q, w, g, lo, hi, v)
    served = q.fake(shifted, scale, zero)
    normalized = shifted / _cols(q.served_scale(scale), g)
    if zero is not None:
        normalized = normalized + _cols(zero, g)
    bounded = normalized.clamp(q.fmt.qmin, q.fmt.qmax)
    codes = q.quantize(shifted, scale, zero)
    surrogate_codes = (codes - bounded).detach() + bounded
    surrogate = q.dequant(surrogate_codes, scale, zero)
    return served.detach() + (surrogate - surrogate.detach())


def _split(u: Unit, wq: torch.Tensor, names: dict[int, str]) -> dict[str, torch.Tensor]:
    """Parameter name (relative to the layer) -> its quantized replacement, for `functional_call`."""
    out = {}
    if u.part is None:
        r0 = 0
        for p, r in zip(u.params, u.rows, strict=True):
            out[names[id(p)]] = wq[0, r0 : r0 + r].to(p.dtype)
            r0 += r
    elif isinstance(u.names, str):
        out[names[id(u.params[0])]] = u.stored(wq).to(u.params[0].dtype)
    else:
        for e, ps in enumerate(u.params):
            r0 = 0
            for p in ps:
                out[names[id(p)]] = wq[e, r0 : r0 + p.shape[0]].to(p.dtype)
                r0 += p.shape[0]
    return out
