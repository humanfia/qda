"""The ``disk`` weight store: layers read from the checkpoint for their turn give the bytes of a resident build."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.target import Target

from ..engine_util import save_tiny_tokenizer, tiny_qwen3_moe, write_ids


@pytest.fixture
def moe(tmp_path):
    tiny_qwen3_moe(layers=3).to(torch.bfloat16).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(8)]
    write_ids(tmp_path / "c.jsonl", rows, sha)
    return tmp_path


def _recipe(tmp, *steps):
    return Recipe(steps=list(steps), calib=FileSource(path=str(tmp / "c.jsonl"), samples=8, seqlen=32))


def _files(out):
    return {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name not in ("build.json", "resolved.yaml")
    }


def test_a_layer_on_a_card_is_mapped_and_on_the_cpu_is_copied():
    """The byte copy a pread costs is not paid on a card; the CPU build keeps its own buffer (d789f3d1)."""
    from qda_quant.engine.disk import _backend

    shard = "model-00001-of-00005.safetensors"
    assert _backend(shard, set(), torch.device("cuda")) == "mmap"
    assert _backend(shard, set(), torch.device("cpu")) == "pread"
    # a host-side tensor is selected rows from, never written
    assert _backend(shard, {shard}, torch.device("cpu")) == "mmap"


def test_disk_build_matches_resident_bytes(moe):
    """Fused experts, renamed keys and the dtype plan come from transformers' loader: same tensors, same bytes."""
    rec = _recipe(moe, REGISTRY["gptq"]())
    got = {}
    for residency in ("resident", "disk"):
        out = moe / residency
        build(str(moe / "m"), rec, format="w4a16-sym-g32", out_dir=out, devices=["cpu"], residency=residency,
              log=lambda _: None)  # fmt: skip
        assert json.loads((out / "build.json").read_text())["residency"]["weights"] == residency
        got[residency] = _files(out)
    assert got["disk"] == got["resident"]


def test_planner_gives_a_calibration_free_build_the_store_that_holds_the_least(moe):
    """RTN keeps no whole-model state, so it runs one layer at a time without asking for the store by name."""
    rec, fmt, dev = _recipe(moe, REGISTRY["rtn"]()), Format.parse("w4a16-sym-g32"), [torch.device("cpu")]
    assert compile_plan(str(moe / "m"), rec, fmt, Target(), dev).residency.weights == "disk"
    got = {}
    for residency in (None, "resident"):  # the planner's choice against the slow path, as the reference
        out = moe / ("planner" if residency is None else residency)
        build(
            str(moe / "m"), rec, format=fmt, out_dir=out, devices=dev, residency=residency, log=lambda _: None
        )
        assert json.loads((out / "build.json").read_text())["residency"]["weights"] == (residency or "disk")
        got[residency] = _files(out)
    assert got[None] == got["resident"]


def test_a_source_without_shards_still_plans_the_disk_store(moe):
    """Planning a repo id reads its configs only: the codecs are checked at build time, when the shards are there,
    not refused at plan time for want of weights to check."""
    meta = moe / "meta"
    meta.mkdir()
    for p in (moe / "m").iterdir():
        if p.suffix in (".json", ".txt", ".model"):
            shutil.copy(p, meta / p.name)
    rec = _recipe(moe, REGISTRY["rtn"]())
    plan = compile_plan(str(meta), rec, Format.parse("w4a16-sym-g32"), Target(), [torch.device("cpu")])
    assert plan.store.name == "disk"


def test_a_calibrated_build_keeps_the_default_store(moe):
    """A step that keeps per-layer statistics uses the whole model: the planner does not take the disk store."""
    fmt, dev = Format.parse("w4a16-sym-g32"), [torch.device("cpu")]
    rec = _recipe(moe, REGISTRY["gptq"]())
    assert compile_plan(str(moe / "m"), rec, fmt, Target(), dev).residency.weights == "resident"


def test_a_store_that_refuses_the_plan_hands_over_to_the_default_one(moe):
    """A model step needs the whole model: the planner's least-memory store steps aside instead of refusing the
    build, and a store the config forces is kept (and its refusal reported)."""
    fmt, dev = Format.parse("w4a16-sym-g32"), [torch.device("cpu")]
    rec = _recipe(moe, REGISTRY["permute"](), REGISTRY["rtn"]())
    assert compile_plan(str(moe / "m"), rec, fmt, Target(), dev).residency.weights == "resident"
    with pytest.raises(Refused, match="model steps"):
        compile_plan(str(moe / "m"), rec, fmt, Target(), dev, residency="disk")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_disk_moves_constructor_buffers_with_the_layer(tmp_path, device):
    """GPT-Neo's causal mask is no checkpoint tensor: it must follow each layer to the card, and nothing may
    keep the build's skeleton alive after it."""
    import gc

    import transformers
    from qda_quant.engine.disk import DISK

    cfg = transformers.GPTNeoConfig(
        vocab_size=128, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128,
        attention_types=[[["global", "local"], 1]], window_size=8, max_position_embeddings=64,
    )  # fmt: skip
    torch.manual_seed(0)
    transformers.GPTNeoForCausalLM(cfg).to(torch.bfloat16).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(8)], sha
    )
    rec = _recipe(tmp_path, REGISTRY["gptq"]())
    got = {}
    for residency in ("resident", "disk"):
        out = tmp_path / residency
        build(str(tmp_path / "m"), rec, format="w4a16-sym-g32", out_dir=out, devices=[device],
              residency=residency, log=lambda _: None)  # fmt: skip
        got[residency] = _files(out)
    assert got["disk"] == got["resident"]
    gc.collect()
    assert len(DISK._readers) == 0


def test_disk_refuses_model_steps_at_plan_time(moe):
    """A model step needs every weight at once, and splits the layer steps into passes that would lose changes."""
    with pytest.raises(Refused, match="model steps"):
        compile_plan(str(moe / "m"), _recipe(moe, REGISTRY["rotate"](), REGISTRY["gptq"]()),
                     Format.parse("w4a16-sym-g32"), Target(), [torch.device("cpu")], residency="disk")  # fmt: skip


def _fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`w` as F8_E4M3 codes and one F32 scale per 128x128 block, and the bf16 those decode to."""
    r, c = w.shape
    grid = torch.zeros(-(-r // 128) * 128, -(-c // 128) * 128)
    grid[:r, :c] = w.float()
    s = grid.reshape(grid.shape[0] // 128, 128, -1, 128).abs().amax((1, 3)).clamp(min=1e-8) / 448
    full = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:r, :c]
    q = (w.float() / full).to(torch.float8_e4m3fn)
    return q, s, (q.float() * full).to(torch.bfloat16)


def test_disk_builds_a_quantized_source_like_its_decoded_twin(tmp_path):
    """Per-weight decoding, including kept weights, gives the float twin's layers and packed build bytes."""
    import json
    import shutil
    from types import SimpleNamespace

    from qda_quant.arch import describe
    from qda_quant.engine.build_validation import _skeleton
    from qda_quant.engine.disk import DISK
    from safetensors.torch import load_file, save_file

    tiny_qwen3_moe(layers=2).to(torch.bfloat16).save_pretrained(tmp_path / "bf16")
    fp8, twin = {}, {}
    for k, w in load_file(tmp_path / "bf16" / "model.safetensors").items():
        fp8[k] = twin[k] = w
        if w.dim() == 2 and "norm" not in k and ".mlp.gate." not in k and "embed_tokens" not in k:
            fp8[k], fp8[k.removesuffix(".weight") + ".weight_scale_inv"], twin[k] = _fp8_block(w)
    for name, tensors in (("fp8", fp8), ("twin", twin)):
        shutil.copytree(tmp_path / "bf16", tmp_path / name, ignore=shutil.ignore_patterns("*.safetensors"))
        save_file(tensors, str(tmp_path / name / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((tmp_path / "fp8" / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    (tmp_path / "fp8" / "config.json").write_text(json.dumps(cfg))
    got = {}
    for name in ("fp8", "twin"):
        src = str(tmp_path / name)
        graph = describe(_skeleton(src, torch.bfloat16, Target()), verify=False)
        m = DISK.load(
            SimpleNamespace(
                model=src,
                dtype=torch.bfloat16,
                target=Target(),
                plan=SimpleNamespace(graph=graph),
                recipe=_recipe(tmp_path),
            ),
            "cpu",
        )
        for lg in graph.layers:
            layer = m.get_submodule(lg.module)
            DISK.place(layer, lg, torch.device("cpu"))
            got.setdefault(name, {}).update(
                {f"{lg.module}.{k}": v.clone() for k, v in layer.state_dict().items()}
            )
            DISK.release(layer)
    assert got["fp8"].keys() == got["twin"].keys()
    assert all(torch.equal(got["fp8"][k], got["twin"][k]) for k in got["twin"])
    sha = save_tiny_tokenizer(tmp_path / "fp8")
    write_ids(tmp_path / "c.jsonl", [list(range(32))] * 8, sha)
    rec = _recipe(tmp_path, REGISTRY["rtn"]())
    outputs = {}
    for name in ("fp8", "twin"):
        save_tiny_tokenizer(tmp_path / name)
        src, out = str(tmp_path / name), tmp_path / (name + "-out")
        plan = compile_plan(src, rec, Format.parse("w4a16-sym-g32"), Target(), [torch.device("cpu")],
                            residency="disk")  # fmt: skip
        assert plan.store is DISK
        build(src, rec, format="w4a16-sym-g32", out_dir=out, devices=["cpu"], residency="disk",
              log=lambda _: None)  # fmt: skip
        outputs[name] = _files(out)
    assert outputs["fp8"] == outputs["twin"]
    # A native source can mix float and encoded weights; a missing scale must fail before weight loading.
    missing = next(k for k in fp8 if k.endswith(".weight_scale_inv"))
    del fp8[missing]
    save_file(fp8, str(tmp_path / "fp8" / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(Refused, match="no source codec decodes.*F8_E4M3"):
        compile_plan(str(tmp_path / "fp8"), rec, Format.parse("w4a16-sym-g32"), Target(),
                     [torch.device("cpu")], residency="disk")  # fmt: skip
