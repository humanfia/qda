"""Hand-built LayerContexts for step tests (the engine builds them in real runs)."""

import torch
from qda_quant import Format, LayerContext, Target
from qda_quant.arch import describe
from qda_quant.arch.adapters import fused_experts


def contexts(model, fmt="w4a16", tokens=64, target=None, *, verify=False):
    """One LayerContext per layer, with raw Hessian sums of the real layer inputs (experts: random SPD).

    `target` defaults to `Target()` (sm90); a format a card serves only elsewhere passes its target, e.g.
    ``w4a4-nvfp4`` on ``Target.parse("sm100")``. `verify` runs the numeric checks (`arch.describe`'s default):
    a step that folds a zero-centred norm — Qwen3.5's ``x * (1 + w)`` — needs the offsets they record in
    ``graph.extra["norm_offset"]``, so it tests a graph of the kind a build hands it.
    """
    fmt = Format.parse(fmt)
    target = target or Target()
    graph = describe(model, verify=verify)
    first = {grp.modules[0]: grp for g in graph.layers for grp in g.groups}
    xs = {}

    def grab(n):
        def hook(_m, a, _o):
            xs[n] = a[0].reshape(-1, a[0].shape[-1])

        return hook

    hooks = [model.get_submodule(n).register_forward_hook(grab(n)) for n in first]
    with torch.no_grad():
        model(input_ids=torch.randint(0, 128, (1, tokens), generator=torch.Generator().manual_seed(0)))
    for h in hooks:
        h.remove()
    out = []
    g = torch.Generator().manual_seed(1)
    for lg in graph.layers:
        stats = {}
        for grp in lg.groups:
            x = xs[grp.modules[0]].float()
            stats[("hessian", grp.modules[0])] = x.T @ x
            stats[("n_tokens", grp.modules[0])] = x.shape[0]
            stats[("inputs", grp.modules[0])] = xs[grp.modules[0]].clone()
        for bank in lg.banks:
            exp = model.get_submodule(bank.experts)
            e, two_i, d = fused_experts.in_weight(exp).shape  # canonical [E, out, in], however it is stored

            def spd(c, e=e):
                a = torch.randn(e, 4 * c, c, generator=g)
                return a.transpose(1, 2) @ a

            stats[("expert_hessian", bank.router)] = {"in": spd(d), "down": spd(two_i // 2)}
            stats[("n_tokens", bank.router)] = torch.tensor([7] * (e - 1) + [0])
        out.append(LayerContext(model.get_submodule(lg.module), lg, graph, fmt, target.plan(fmt, "dense"),
                                target.plan(fmt, "moe"), stats))  # fmt: skip
    return out
