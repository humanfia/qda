"""Engine propagation == stock HF forward, on the real models (GPU).

Float weights, no steps, 4 calibration samples. The engine captures layer 0's input and propagates through
every decoder layer; each layer's output must equal the stock model's on the same samples. Also checks the
stock forward with an all-ones ``attention_mask`` (as llm-compressor calls it) against the mask-free one.
Prints the max relative difference per layer; the bar is bit-identical outputs. Needs ``--calib`` and
``--model-dir NAME=PATH`` for each name of `MODELS` (see conftest.py); a model not given is skipped.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

#: the ``--model-dir`` names this test runs on
MODELS = ("qwen3-8b", "r1-llama-8b", "qwen3-30b-a3b")


def _rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return float(d.max() / b.float().abs().max()), float(d.norm() / b.float().norm())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
@pytest.mark.parametrize("name", MODELS)
def test_propagation_equals_hf(name, real_model, calib_file):
    from qda_quant.arch import describe
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.calib import load_ids
    from qda_quant.engine.execute import Timers
    from qda_quant.engine.runner import Pass, capture
    from qda_quant.step import LayerContext
    from transformers import AutoModelForCausalLM

    path = real_model(name)
    dev = torch.device("cuda:0")
    m = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map={"": "cuda:0"}
    ).eval()
    ids = load_ids(FileSource(path=str(calib_file), samples=4, seqlen=2048), path)
    batches = [r[None] for r in ids]
    graph = describe(m, verify=False)
    layers = [m.get_submodule(lg.module) for lg in graph.layers]

    # stock forward: every layer's output via hooks, and hidden_states[-1] (after the final norm)
    ref: list[list[torch.Tensor]] = []
    last_ref, mask_dev = [], []
    for b in batches:
        outs: list[torch.Tensor] = []
        hs = [
            ly.register_forward_hook(
                lambda _m, _a, o, outs=outs: outs.append((o[0] if isinstance(o, tuple) else o).cpu())
            )
            for ly in layers
        ]
        with torch.no_grad():
            o = m(input_ids=b.to(dev), use_cache=False, output_hidden_states=True)
        for h in hs:
            h.remove()
        ref.append(outs)
        last_ref.append(o.hidden_states[-1].cpu())
        with torch.no_grad():  # llm-compressor passes an all-ones attention_mask
            om = m(input_ids=b.to(dev), attention_mask=torch.ones_like(b).to(dev), use_cache=False)
            on = m(input_ids=b.to(dev), use_cache=False)
        mask_dev.append(_rel(om.logits, on.logits)[0])
    assert all(len(r) == len(layers) for r in ref)

    # engine: capture + one pass with no steps; record each layer's propagated output
    got: list[list[torch.Tensor]] = [[] for _ in batches]
    slots = capture(m, graph, batches, dev)

    class Tap(Pass):
        def _forward(self, layer, lg, i, sl, st, keep=False, propagate=False):
            r = super()._forward(layer, lg, i, sl, st, keep, propagate)
            if propagate:
                for j, s in enumerate(sl):
                    got[j].append(s.h.cpu())
            return r

    Tap(
        m,
        graph,
        [],
        dev,
        lambda layer, lg, st: LayerContext(layer, lg, graph, None, None, None, st, device=dev),
        Timers([dev]),
    ).run(slots)

    worst = 0.0
    for i in range(len(layers)):
        mx = max(_rel(got[j][i], ref[j][i])[0] for j in range(len(batches)))
        fr = max(_rel(got[j][i], ref[j][i])[1] for j in range(len(batches)))
        print(f"{name} layer {i:2d}: max rel {mx:.3e}  frob rel {fr:.3e}")
        worst = max(worst, mx)
    with torch.no_grad():
        fin = [m.get_submodule(graph.final_norm)(s.h).cpu() for s in slots]
    fl = max(_rel(fin[j], last_ref[j])[0] for j in range(len(batches)))
    print(f"{name} final norm vs hidden_states[-1]: max rel {fl:.3e}")
    print(f"{name} stock logits with vs without all-ones attention_mask: max rel {max(mask_dev):.3e}")
    print(f"{name} WORST layer max rel {worst:.3e}")
    # the engine runs the same modules with the same per-sample arguments as the stock forward (the expert
    # statistics only observe around the experts' own forward), so the outputs are bit-identical
    assert worst == 0.0 and fl == 0.0
