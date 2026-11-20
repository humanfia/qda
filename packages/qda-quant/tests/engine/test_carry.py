"""Auxiliary state follows each batch and each stream through a layer pass."""

import copy
from types import SimpleNamespace
from typing import ClassVar

import torch
from qda_quant.arch import LayerGraph, ModelGraph
from qda_quant.arch.adapters import ADAPTERS
from qda_quant.engine.execute import Timers
from qda_quant.engine.runner import Pass, capture
from qda_quant.step import LayerContext, Step
from torch import nn


class CarryAdapter:
    def layer_carry(self):
        return ("memory",)


class CarryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x, *, memory):
        out = x @ self.weight + memory
        return out, out.sum(-1, keepdim=True) / 8


class CarryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=3)
        self.embed = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([CarryLayer() for _ in range(3)])

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        memory = torch.zeros_like(x[..., :1])
        for layer in self.layers:
            x, memory = layer(x, memory=memory)
        return x


class Shrink(Step):
    needs: ClassVar[frozenset[str]] = frozenset({"layer_io_float"})

    def apply_layer(self, ctx):
        ctx.layer.weight.data.mul_(0.5)


def test_carry_isolated_between_batches_and_float_stream(monkeypatch):
    monkeypatch.setitem(ADAPTERS._entries, "test_carry", CarryAdapter())
    torch.manual_seed(3)
    model = CarryModel().eval()
    original = copy.deepcopy(model)
    graph = ModelGraph(
        "test",
        tuple(LayerGraph(i, f"layers.{i}", (), ()) for i in range(3)),
        "embed",
        "",
        "",
        adapter="test_carry",
    )
    batches = [torch.arange(1, 9)[None], torch.arange(8, 0, -1)[None]]
    cpu = torch.device("cpu")
    slots = capture(model, graph, batches, cpu, keep_float=True)
    with torch.no_grad():
        Pass(
            model,
            graph,
            [Shrink()],
            cpu,
            make_ctx=lambda layer, lg, stats: LayerContext(
                layer, lg, graph, None, None, None, stats, device=cpu
            ),
            timers=Timers(),
        ).run(slots)
        for ids, slot in zip(batches, slots, strict=True):
            torch.testing.assert_close(slot.h, model(ids), rtol=0, atol=0)
            torch.testing.assert_close(slot.float_h, original(ids), rtol=0, atol=0)
