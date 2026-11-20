"""A cached build writes the whole build's bytes for every layer it may reuse exactly, and says what it reused.

GPTQ solves each layer on the quantized stream of the layers before it, so a changed layer changes every
later layer's result: the tests change a format in the middle layer and check both sides of it.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from qda_quant import REGISTRY, build
from qda_quant.calib.sources import FileSource
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from safetensors.torch import load_file

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids

MID = ("re:^model\\.layers\\.1\\.mlp\\.", "w8a16-sym-g32")


@pytest.fixture
def tiny(tmp_path):
    tiny_qwen3(layers=3).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha
    )
    return tmp_path


def run(tmp, out, rules=(), match=None, steps=None, log=None):
    recipe = Recipe(
        steps=[REGISTRY["gptq"]()] if steps is None else steps,
        calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=16),
    )
    out = build(
        str(tmp / "m"), recipe, format="w4a16-sym-g32", out_dir=tmp / out, devices=["cpu"], dtype=torch.float32,
        format_overrides=list(rules), log=(lambda s: None) if log is None else log,
        layer_cache=None if match is None else {"dir": str(tmp / "cache"), "match": match},
    )  # fmt: skip
    weights = {}
    for f in sorted(out.glob("*.safetensors")):
        weights.update(load_file(f))
    record = json.loads((out / "build.json").read_text()).get("layer_cache")
    return weights, None if record is None else record[0]


def layer(weights, i):
    return {k: v for k, v in weights.items() if k.startswith(f"model.layers.{i}.")}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def reused(record):
    return [r["reused"] for r in record["layers"]]


def test_a_cached_build_reuses_exactly_up_to_the_first_changed_layer(tiny):
    """Layers before the change are reused byte for byte; the changed layer and every later one are solved."""
    whole, _ = run(tiny, "whole", [MID])
    first, record = run(tiny, "first", match="exact")  # fills the cache with the body-format build
    assert reused(record) == [None, None, None] and not record["screening_only"]
    again, record = run(tiny, "again", match="exact")
    assert reused(record) == ["exact"] * 3 and same(again, first)
    assert not record["screening_only"]
    changed, record = run(tiny, "changed", [MID], match="exact")
    assert reused(record) == ["exact", None, None]
    assert same(changed, whole)  # an exact cache hit is the whole build's bytes


def test_matching_formats_reuses_a_later_layer_solved_under_another_upstream(tiny):
    """Layer 2 keeps codes solved behind the other layer 1, so it differs from a whole build: screening only."""
    whole, _ = run(tiny, "whole", [MID])
    run(tiny, "first", match="formats")
    changed, record = run(tiny, "changed", [MID], match="formats")
    assert reused(record) == ["exact", None, "upstream differs"] and record["screening_only"]
    assert same(layer(changed, 1), layer(whole, 1))  # the changed layer is solved on this build's own stream
    assert not same(layer(changed, 2), layer(whole, 2))
    after, record = run(tiny, "after", [MID], match="formats")  # its layer 2 was kept under a stale chain
    assert reused(record)[2] == "upstream differs"


def test_a_key_part_that_changes_the_solve_misses_the_cache(tiny):
    """A different calibration set shares no layer, even with the same formats."""
    run(tiny, "first", match="exact")
    g = torch.Generator().manual_seed(1)
    sha = json.loads((tiny / "c.jsonl").read_text().splitlines()[0])["tokenizer"]
    write_ids(tiny / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha)
    _, record = run(tiny, "other", match="exact")
    assert reused(record) == [None, None, None]


def test_a_multi_card_build_refuses_the_cache(tiny):
    with pytest.raises(Refused, match="layer_cache: a cached build runs on one card"):
        build(
            str(tiny / "m"),
            Recipe(steps=[REGISTRY["gptq"]()], calib=FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16)),
            format="w4a16-sym-g32", out_dir=tiny / "x", devices=["cpu", "cpu"], log=lambda s: None,
            layer_cache={"dir": str(tiny / "cache")},
        )  # fmt: skip


def test_external_plugin_code_changes_with_the_same_config_do_not_reuse_layers(tiny, monkeypatch):
    """External code is outside the source digest; even an unchanged registered recipe must solve again."""
    import sys
    import types

    plugin = types.ModuleType("external_quant")
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    source = tiny / "external_quant.py"

    def install(factor):
        source.write_text(
            "import torch\nfrom qda_quant.steps.gptq import GPTQ\n"
            "class ExternalGPTQ(GPTQ):\n"
            "    def apply_layer(self, ctx):\n"
            "        with torch.no_grad():\n"
            f"            for p in ctx.layer.parameters(): p.mul_({factor})\n"
            "        super().apply_layer(ctx)\n"
        )
        exec(compile(source.read_text(), str(source), "exec"), plugin.__dict__)
        monkeypatch.setitem(REGISTRY._entries, "gptq", plugin.ExternalGPTQ)
        return plugin.ExternalGPTQ()

    first_step = install(1.0)
    logs = []
    first, record = run(tiny, "first", match="exact", steps=[first_step], log=logs.append)
    assert record is None and sum(s.startswith("layer_cache: disabled") for s in logs) == 1
    second_step = install(0.5)
    assert first_step.model_dump() == second_step.model_dump()
    whole, _ = run(tiny, "whole", steps=[second_step])
    logs.clear()
    changed, record = run(tiny, "changed", match="exact", steps=[second_step], log=logs.append)
    assert record is None and sum(s.startswith("layer_cache: disabled") for s in logs) == 1
    assert not same(first, whole) and same(changed, whole)
    assert not (tiny / "cache").exists()


def test_changed_upstream_token_weights_do_not_reuse_layers(tiny, monkeypatch):
    """Realized model-step weights can change with the same settings; cached output must equal a fresh solve."""
    guided = REGISTRY["guided"]
    reverse = False

    def weights(self, model, graph, run):
        run.token_weights = []
        for ids in run.batches():
            w = torch.arange(1, ids.shape[-1] + 1, dtype=torch.float32).expand_as(ids)
            run.token_weights.append(w.flip(-1) if reverse else w)

    monkeypatch.setattr(guided, "apply_model", weights)
    steps = [guided(), REGISTRY["gptq"]()]
    logs = []
    first, record = run(tiny, "first", match="exact", steps=steps, log=logs.append)
    assert record is None and sum(s.startswith("layer_cache: disabled") for s in logs) == 1
    reverse = True
    whole, _ = run(tiny, "whole", steps=steps)
    logs.clear()
    changed, record = run(tiny, "changed", match="exact", steps=steps, log=logs.append)
    assert record is None and sum(s.startswith("layer_cache: disabled") for s in logs) == 1
    assert not same(first, whole) and same(changed, whole)
    assert not (tiny / "cache").exists()


def test_a_reused_layer_advances_the_float_stream_before_restoring_its_weights(tmp_path, monkeypatch):
    """A later solve needs the original float targets and per-batch carry even when earlier layers are reused."""
    from qda_quant.arch import LayerGraph, ModelGraph
    from qda_quant.arch.adapters import ADAPTERS
    from qda_quant.engine.execute import Timers
    from qda_quant.engine.layer_cache import LayerCache, LayerCacheSetting
    from qda_quant.engine.runner import Pass, capture
    from qda_quant.step import LayerContext

    from .test_carry import CarryAdapter, CarryModel, Shrink

    monkeypatch.setitem(ADAPTERS._entries, "test_carry", CarryAdapter())
    torch.manual_seed(3)
    original = CarryModel().eval()
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
    setting = LayerCacheSetting(dir=str(tmp_path / "cache"))
    for formats in ({}, {"layers.1.weight": "changed"}):
        model = deepcopy(original)
        slots = capture(model, graph, batches, cpu, keep_float=True)
        cache = LayerCache(setting, "float-stream", formats)
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
                cache=cache,
            ).run(slots)
            for ids, slot in zip(batches, slots, strict=True):
                torch.testing.assert_close(slot.h, model(ids), rtol=0, atol=0)
                torch.testing.assert_close(slot.float_h, original(ids), rtol=0, atol=0)
        assert reused(cache.info()) == (["exact", None, None] if formats else [None, None, None])


def test_a_gguf_rebuilt_from_its_cache_is_the_same_file(tmp_path, monkeypatch):
    """A routed bank's codes and the exporter's column order come back exactly: the same GGUF, byte for byte.

    A candidate that differs only in a stored type shares every solved layer (the type is not in the key).
    """
    pytest.importorskip("gguf")
    from ..gguf.test_bank_down_format import DOWN, Q6K, TYPES, _config, _tiny

    src, calib, template = _tiny(tmp_path, monkeypatch)
    cache = {"dir": str(tmp_path / "cache"), "match": "exact"}

    def gguf(out, types=TYPES):
        cfg = _config(tmp_path, src, calib, template, out, overrides=[(DOWN, Q6K)], types=types)
        out = cfg.model_copy(update={"layer_cache": cache}).build(log=lambda _: None)
        return (out / "model-Q4_K_M.gguf").read_bytes(), json.loads((out / "build.json").read_text())[
            "layer_cache"
        ][0]

    whole = _config(tmp_path, src, calib, template, "whole", overrides=[(DOWN, Q6K)]).build(
        log=lambda _: None
    )
    first, record = gguf("first")
    assert reused(record) == [None] and first == (whole / "model-Q4_K_M.gguf").read_bytes()
    again, record = gguf("again")
    assert reused(record) == ["exact"] and again == first
    q8_attn = [{**t, "type": "Q8_0"} if "attn" in t["pattern"] else t for t in TYPES]
    _, record = gguf("q8-attn", q8_attn)
    assert reused(record) == ["exact"]
