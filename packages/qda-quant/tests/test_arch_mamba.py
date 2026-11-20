"""Pure Mamba (``mamba``/``mamba2``): the plan refuses a quantized body vLLM 0.33 cannot load.

vLLM builds a Mamba mixer's ``in_proj`` as a ``MergedColumnParallelLinear``; its quantized weight loader
receives the module instead of a parameter and raises ``AttributeError``. The plan used to call the row
servable because the target table only knows kernels, so a build spent GPU hours before vLLM refused it.
The case hangs off the plan's shape rule (`engine.build_validation`), beside the fused-shape check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qda_quant.engine.build_validation import check_mamba_served
from qda_quant.step import Refused


@pytest.mark.parametrize("model_type", ["mamba", "mamba2"])
def test_a_pure_mamba_plan_refuses_a_quantized_body(model_type):
    graph = SimpleNamespace(model_type=model_type)
    with pytest.raises(Refused, match="cannot load a quantized pure Mamba checkpoint"):
        check_mamba_served(graph, True, SimpleNamespace(arch="sm90"))
    with pytest.raises(Refused, match="serve the bf16 original instead"):
        check_mamba_served(graph, True, SimpleNamespace(arch="sm100"))


def test_a_bf16_pure_mamba_body_and_the_llama_cpp_exporter_pass():
    """vLLM serves the bf16 original; the GGUF exporter names a pure Mamba source itself."""
    graph = SimpleNamespace(model_type="mamba")
    check_mamba_served(graph, False, SimpleNamespace(arch="sm90"))
    check_mamba_served(graph, True, SimpleNamespace(arch="llama_cpp"))


@pytest.mark.parametrize("model_type", ["nemotron_h", "qwen3_next", "qwen3"])
def test_an_attention_or_hybrid_graph_is_not_refused_here(model_type):
    """The hybrid and attention rows state their own serving facts through their adapters."""
    check_mamba_served(SimpleNamespace(model_type=model_type), True, SimpleNamespace(arch="sm90"))
