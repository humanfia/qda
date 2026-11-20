"""Router-weighted GPTQ settings and engine integration."""

from dataclasses import replace

import pytest
import torch
from qda_quant import Format, Recipe, Target
from qda_quant.engine.stats import reference_expert_stats
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ

from .engine.test_runner import SEEN, Probe, batches, run
from .engine_util import graph_of, tiny_qwen3_moe


class WeightedProbe(Probe):
    router_weighted: bool = True


def test_recipe_exposes_router_weighted():
    recipe = Recipe.from_dict(
        {"steps": [{"gptq": {"router_weighted": True}}], "calib": {"file": {"path": "calib.jsonl"}}}
    )
    assert recipe.steps[0].router_weighted
    assert Recipe.from_yaml(recipe.to_yaml()).steps[0].router_weighted


def test_router_weighted_pass_matches_reference():
    model = tiny_qwen3_moe(layers=1)
    moe = model.model.layers[0].mlp
    inputs = []
    handle = moe.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    calibration = batches(n=2)
    with torch.no_grad():
        for ids in calibration:
            model(input_ids=ids, use_cache=False)
    handle.remove()
    graph, _ = run(model, [WeightedProbe()], calibration)
    router = graph.layers[0].banks[0].router
    actual = SEEN[0].stats[("expert_hessian", router)]
    counts = SEEN[0].stats[("n_tokens", router)]
    with torch.no_grad():
        hidden = torch.cat(inputs).reshape(-1, model.config.hidden_size)
        _, weights, indices = moe.gate(hidden)
        expected = reference_expert_stats(moe.experts, hidden, indices, weights, True)
    for part in ("in", "down"):
        torch.testing.assert_close(actual[part], expected[part], rtol=1e-4, atol=1e-5)
    assert torch.equal(counts, expected["n_tokens"])


@pytest.mark.parametrize("setting", [{"router_weighted": True}, {"all_token_blend": 0.1}])
def test_unfused_router_weighting_refused_before_calibration(setting):
    graph = graph_of(tiny_qwen3_moe(layers=1))
    layer = graph.layers[0]
    bank = replace(layer.banks[0], experts=((("expert.up",), "expert.down"),))
    graph = replace(graph, layers=(replace(layer, banks=(bank,)),))
    format_spec = Format.parse("w4a16")
    with pytest.raises(Refused, match="requires fused expert banks"):
        GPTQ(**setting).check(graph, format_spec, Target().plan(format_spec, "moe"))
