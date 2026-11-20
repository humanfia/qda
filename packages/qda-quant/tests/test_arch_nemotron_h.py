"""Nemotron-3-Nano (nemotron_h): Mamba2/attention hybrid whose routed experts are a fused, non-gated bank.

The bank holds ``experts.up_proj`` and ``experts.down_proj`` (act_fn(up) feeds down), not a gated
``gate_up_proj``, so every step that assumed a gate has to follow the bank's shape.
"""

from __future__ import annotations

import json
import re

import pytest
import torch
import yaml
from qda_quant import BuildConfig
from qda_quant.arch import describe
from qda_quant.cli import main
from qda_quant.step import Refused
from safetensors import safe_open

from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids
from .tiny import model


def _saved(root):
    source = root / "m"
    model("nemotron_h").save_pretrained(source)
    sha = save_tiny_tokenizer(source)
    g = torch.Generator().manual_seed(0)
    write_ids(root / "c.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(6)], sha)
    return source


def _config(source):
    return {
        "model": str(source),
        "format": "w4a16-sym-g64",
        "devices": ["cpu"],
        "steps": [{"gptq": {}}],
        "calib": {"file": {"path": str(source.parent / "c.jsonl"), "samples": 6, "seqlen": 32}},
        "out": str(source.parent / "out"),
    }


def _keys(ckpt):
    keys = set()
    for f in sorted(ckpt.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            keys |= set(h.keys())
    return keys


def test_the_routed_experts_are_a_fused_non_gated_bank():
    graph = describe(model("nemotron_h"), verify=True)
    (bank,) = [bank for layer in graph.layers for bank in layer.banks]
    assert (bank.experts, bank.router, bank.top_k, bank.fused, bank.gated) == (
        "model.layers.1.mixer.experts",
        "model.layers.1.mixer.gate",
        2,
        True,
        False,
    )
    assert bank.shared == (
        "model.layers.1.mixer.shared_experts.up_proj",
        "model.layers.1.mixer.shared_experts.down_proj",
    )


def test_the_export_writes_a_per_expert_up_and_down_and_never_a_gate(tmp_path):
    source = _saved(tmp_path)
    cfg = BuildConfig.from_dict(_config(source))
    cfg.build()
    keys = _keys(tmp_path / "out")
    for expert in range(8):
        for proj in ("up_proj", "down_proj"):
            assert f"backbone.layers.1.mixer.experts.{expert}.{proj}.weight_packed" in keys
    assert not [k for k in keys if "gate_proj" in k]
    # the shared expert has no gate either: it stays one float up/down pair beside the quantized bank
    assert "backbone.layers.1.mixer.shared_experts.up_proj.weight" in keys


def test_awq_refuses_a_non_gated_bank(tmp_path):
    source = _saved(tmp_path)
    cfg = BuildConfig.from_dict({**_config(source), "steps": [{"awq": {}}, {"gptq": {}}]})
    with pytest.raises(Refused, match="non-gated bank"):
        cfg.build()


def test_rotate_classifies_the_mixer_writers_and_permute_refuses_them(tmp_path, capsys):
    """NemotronH names its blocks ``mixer``, so no group derives as a residual writer.

    rotate classifies the linears no layer norm feeds (``mixer.out_proj``, ``mixer.o_proj``) as the writers and
    plans and builds. permute still moves the readers only and would leave those writers in the old basis, so it
    must refuse at plan and at build, exit 2, no traceback.
    """
    source = _saved(tmp_path)
    path = tmp_path / "rotate.yaml"
    path.write_text(yaml.safe_dump({**_config(source), "steps": [{"rotate": {}}, {"rtn": {}}]}))
    for cmd in ("plan", "build"):
        assert main([cmd, str(path)]) == 0
        assert "Traceback" not in capsys.readouterr().err
    path = tmp_path / "permute.yaml"
    path.write_text(
        yaml.safe_dump(
            {**_config(source), "out": str(tmp_path / "p"), "steps": [{"permute": {}}, {"rtn": {}}]}
        )
    )
    for cmd in ("plan", "build"):
        assert main([cmd, str(path)]) == 2
        err = capsys.readouterr().err
        assert "permute: model.layers.0 writes the residual stream through linears" in err
        assert "unclassified group kind" in err
        assert "Traceback" not in err


def test_a_classified_layout_still_accepts_rotate_and_permute(tmp_path, capsys):
    tiny_qwen3(layers=2, hidden=64, inter=96).save_pretrained(tmp_path / "qwen")
    sha = save_tiny_tokenizer(tmp_path / "qwen")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "q.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(2)], sha
    )
    base = {
        "model": str(tmp_path / "qwen"),
        "format": "w4a16-sym-g32",
        "devices": ["cpu"],
        "calib": {"file": {"path": str(tmp_path / "q.jsonl"), "samples": 2, "seqlen": 32}},
        "out": str(tmp_path / "out"),
    }
    for steps in ([{"rotate": {}}, {"rtn": {}}], [{"permute": {}}, {"rtn": {}}]):
        path = tmp_path / "q.yaml"
        path.write_text(yaml.safe_dump({**base, "steps": steps}))
        assert main(["plan", str(path)]) == 0
        assert "Traceback" not in capsys.readouterr().err


def test_gsq_trains_a_fused_bank_in_the_engine_s_chunks(tmp_path, monkeypatch):
    """gsq over a fused bank survives the engine's expert chunks, and trains every expert (xy-1712).

    One card cannot hold gsq over a 128-expert bank: the bank is ONE unit of all experts, and `GumbelRounding`
    builds five code shifts plus a centring temporary over the whole of it (38.3 GB at 638M parameters, the
    Nemotron 2-layer smoke). The engine already solves a bank in chunks of experts for steps that declare
    `splits_experts`; gsq did not, and its `_split` handed `functional_call` a chunk-sized tensor for the whole
    parameter, so the flag alone was not enough. Before this test's change the build below raises on the second
    chunk; after it, every one of the 8 experts differs from the gptq-only build.
    """
    import dataclasses
    import importlib

    from qda_quant import REGISTRY
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.build import build
    from qda_quant.recipe import Recipe

    source = _saved(tmp_path)
    engine_build = importlib.import_module("qda_quant.engine.build")
    compile_plan = engine_build.compile_plan

    def files(chunks, steps):
        def chunked(*args, **kwargs):
            p = compile_plan(*args, **kwargs)
            return dataclasses.replace(p, residency=dataclasses.replace(p.residency, expert_chunks=chunks))

        monkeypatch.setattr(engine_build, "compile_plan", chunked)
        out = tmp_path / f"out-{chunks}-{len(steps)}"
        recipe = Recipe(
            steps=steps, calib=FileSource(path=str(source.parent / "c.jsonl"), samples=4, seqlen=32)
        )
        build(str(source), recipe, format="w4a16-sym-g32", out_dir=out, devices=["cpu"], log=lambda *_: None)
        banks = {}
        for f in sorted(out.glob("*.safetensors")):
            with safe_open(f, "pt") as h:
                keys = [k for k in set(h.keys()) if re.search(r"\.experts\.\d+\.", k)]
                banks |= {k: h.get_tensor(k) for k in keys if k.endswith("weight_packed")}
        return banks, json.loads((out / "build.json").read_text())

    gptq = [REGISTRY["gptq"](impl="torch")]
    gsq = [*gptq, REGISTRY["gsq"](epochs=1, batches=1, validation=1)]
    reference, _ = files(1, gptq)
    whole, whole_info = files(1, gsq)
    chunked, chunked_info = files(2, gsq)

    assert "replay" not in whole_info["timings"]["phases"]
    assert "replay" in chunked_info["timings"]["phases"]
    assert set(chunked) == set(whole) == set(reference)
    assert chunked  # the export splits the fused bank into one tensor per expert
    expert = re.compile(r"\.experts\.(\d+)\.")
    seen = {int(m.group(1)) for k in chunked if (m := expert.search(k))}
    assert seen == set(range(8)), f"experts exported {sorted(seen)}"
    for name, w in chunked.items():  # every expert was trained, on both paths
        assert w.shape == whole[name].shape == reference[name].shape
        e = int(expert.search(name).group(1))
        assert (w != reference[name]).any().item(), f"expert {e} untrained in {name}"
        assert (w != whole[name]).any().item(), f"chunked and whole agree on expert {e} of {name}"
