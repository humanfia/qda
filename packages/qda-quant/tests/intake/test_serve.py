"""``qda inspect --serve``: vLLM's own weight loading names the tensors it never read.

The check hands every checkpoint tensor to vLLM's loaders as a marked tensor; a tensor is read once an
operation reads its values, not when a loader asks its shape or rank. vLLM's MoE loader asks every expert
tensor's rank before dropping the ones that match no mapping, so that emulation runs on CPU under vLLM. The
GPU test exports a tiny Qwen3 W4A16 checkpoint: a tensor a loader ignores by name must be reported, and a
tensor the model class drops on purpose must be listed without failing the check.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from qda_quant import REGISTRY, Format, Target
from qda_quant.arch import describe
from qda_quant.intake.cli import _run_inspect
from qda_quant.intake.probe import _tied
from qda_quant.intake.report import inspect
from qda_quant.output.export import Writer
from qda_quant.output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION

from ..ctx import contexts
from ..engine_util import save_tiny_tokenizer
from ..tiny import config

SCALE = "model.layers.0.self_attn.q_proj.weight_scale"
RENAMED = SCALE + "x"
IGNORED = "model.layers.0.self_attn.q_norm.bias"
DROPPED = "model.layers.0.self_attn.rotary_emb.inv_freq"


def test_an_expert_tensor_vllm_drops_stays_unread():
    """vLLM's MoE loader asks .dim() on every expert tensor before it drops one that matches no mapping; a
    metadata query is not a read, or the loader's silent skip would pass the check."""
    routed = pytest.importorskip("vllm.model_executor.layers.fused_moe.routed_experts")
    from qda_quant.intake import serve

    Read = serve._reader()
    mapping = routed.RoutedExperts.build_expert_params_mapping(
        "gate_proj", "down_proj", "up_proj", num_experts=4, routed_experts_prefix="", include_fused=True
    )
    layer = SimpleNamespace(
        get_expert_mapping=lambda include_fused: mapping, layer_name="model.layers.0.mlp.experts"
    )
    for name in ("0.w1.weight_packed", "4.gate_proj.weight_packed"):  # unknown projection, expert id 4 of 4
        t = torch.ones(3, 4, dtype=torch.bfloat16).as_subclass(Read)
        t.read = False
        assert list(routed.RoutedExperts.load_weights(layer, iter([(name, t)]))) == []
        assert not t.read


def test_a_stored_tied_head_is_dropped_only_as_a_copy_of_the_embedding():
    """A tied config's stored lm_head is harmless only when it holds the embedding's own bytes; another head is
    the checkpoint its config describes wrongly (vLLM serves the embedding as the head), so it is reported."""
    from qda_quant.intake import serve

    emb = torch.zeros(4, 4, dtype=torch.bfloat16)
    tensors = {"model.embed_tokens.weight": emb, "lm_head.weight": emb.clone()}
    assert serve._classify({"lm_head.weight"}, True, tensors) == ([], ["lm_head.weight"])
    tensors["lm_head.weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
    assert serve._classify({"lm_head.weight"}, True, tensors) == (["lm_head.weight"], [])
    assert serve._classify({"lm_head.weight"}, False, tensors) == (["lm_head.weight"], [])


def test_the_tie_flag_comes_from_one_helper():
    """A vision model nests its text model's config: text_config's flag wins when it has one, else the top one's."""
    assert _tied({"tie_word_embeddings": True})
    assert _tied({"text_config": {"tie_word_embeddings": True}})
    assert _tied({"tie_word_embeddings": True, "text_config": {"hidden_size": 8}})
    assert not _tied({"tie_word_embeddings": True, "text_config": {"tie_word_embeddings": False}})
    assert not _tied({})


def test_a_serve_check_that_cannot_run_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """An interpreter without vLLM (the CPU one): the child dies, and the check says why in one line, exit 1."""
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("needs an interpreter without vLLM")
    from safetensors.torch import save_file

    ckpt = tmp_path / "bf16"
    ckpt.mkdir()
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)}, str(ckpt / "model.safetensors"))
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    (ckpt / "tokenizer.json").write_text("{}")
    code = _run_inspect(
        SimpleNamespace(
            path=str(ckpt),
            target=None,
            json=False,
            serve=True,
            gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        )
    )
    assert code == 1
    err = capsys.readouterr().err.strip()
    # one line: a crash has no fix to print, and the type names what died
    assert err.startswith("error: inspect --serve: the serving check failed") and "vllm" in err
    assert len(err.splitlines()) == 1


def test_the_serving_check_starts_vllm_with_qda_kls_no_jit_settings(tmp_path: Path, monkeypatch):
    """`qda inspect --serve` on a host without nvcc died in FlashInfer's JIT (`Could not find nvcc`) where
    `qda-kl kl` started: its engine started without qda-kl's no-JIT environment and arguments."""
    vllm = pytest.importorskip("vllm")
    from qda_kl.serve import NO_AUTOTUNE_ARGS, ServeSettings, no_jit
    from qda_quant.intake import serve
    from qda_quant.output import vllm_env
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    ckpt = tmp_path / "bf16"
    ckpt.mkdir()
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)}, str(ckpt / "model.safetensors"))
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    started: dict = {}
    monkeypatch.setattr(DefaultModelLoader, "get_all_weights", DefaultModelLoader.get_all_weights)  # restored
    monkeypatch.setattr(vllm_env, "engine_start", lambda env, start: (started.update(env=env), start()))
    monkeypatch.setattr(vllm, "LLM", lambda **kwargs: started.update(kwargs=kwargs))
    serve._serve_in_vllm(str(ckpt), 0.5)
    kl = no_jit(ServeSettings(args={"gpu_memory_utilization": 0.5, "enforce_eager": True}))
    assert started["env"] == {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", **kl.env}
    # FlashInfer autotune stays vLLM's default here: only reference, scoring and gencheck engines turn it off
    args = {k: v for k, v in kl.args.items() if k not in NO_AUTOTUNE_ARGS}
    assert started["kwargs"] == {"model": str(ckpt), **args}


def _export(out: Path, tmp: Path) -> Path:
    """A tiny Qwen3 W4A16 checkpoint with a 64-token tokenizer, written by our exporter."""
    cfg = config("qwen3")
    cfg.max_position_embeddings = 64
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    m.save_pretrained(tmp / "src")
    save_tiny_tokenizer(tmp / "src")
    m = transformers.AutoModelForCausalLM.from_pretrained(tmp / "src", dtype=torch.bfloat16).eval()
    graph = describe(m, verify=False)
    run = SimpleNamespace(qparams={})
    w = Writer(out, tmp / "src", graph, Format.parse("w4a16"), Target())
    for ctx in contexts(m, "w4a16", tokens=256):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        w.add_layer(ctx)
    w.add_model(m, run.qparams)
    w.finish(m.config, None, {})
    return out


def _edit(ckpt: Path, rename: str = "", add: str = "") -> None:
    """Rename one tensor and/or store a new zero tensor, in the shard and the index."""
    from safetensors.torch import load_file, save_file

    shard = ckpt / "model-00001.safetensors"
    t = load_file(str(shard))
    idx = ckpt / "model.safetensors.index.json"
    m = json.loads(idx.read_text())
    if rename:
        t[rename + "x"] = t.pop(rename)
        m["weight_map"][rename + "x"] = m["weight_map"].pop(rename)
    if add:
        t[add] = torch.zeros(128, dtype=torch.bfloat16)
        m["weight_map"][add] = "model-00001.safetensors"
    save_file(t, str(shard), metadata={"format": "pt"})
    idx.write_text(json.dumps(m))


@pytest.mark.gpu
def test_serve_names_the_tensor_vllm_never_reads(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A clean load that silently drops a tensor by name exits 2 and names it; a weight the model class drops
    on purpose is listed without failing; a renamed scale is named too (vLLM refuses it)."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("needs vLLM in the test interpreter (run it with the serving venv's python)")

    from qda_quant.intake import serve

    clean = _export(tmp_path / "clean", tmp_path)
    _edit(clean, add=DROPPED)  # a rotary buffer no loader sees: the model class drops it on purpose
    assert inspect(str(clean))["reasons"] == []  # the exit codes below are the serve check's
    res = serve.serve_check(str(clean))
    assert res["ok"] and res["unconsumed"] == [] and res["dropped"] == [DROPPED], res

    skipped = tmp_path / "skipped"  # vLLM loads this checkpoint and ignores the tensor by name
    shutil.copytree(clean, skipped)
    _edit(skipped, add=IGNORED)
    capsys.readouterr()  # drop the export's progress bars: the contract below is the inspect's stderr alone
    code = _run_inspect(
        SimpleNamespace(
            path=str(skipped),
            target=None,
            json=False,
            serve=True,
            gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        )
    )
    assert code == 2
    said = capsys.readouterr()
    out = said.out
    # the export's tqdm bars share stderr under capsys: the contract is one `error: inspect: ` line among them
    lines = [ln for seg in said.err.split("\n") for ln in seg.split("\r") if ln.startswith("error: inspect: ")]
    assert len(lines) == 1
    assert f"vLLM never read 1 tensor(s), e.g. {IGNORED}" in lines[0]
    assert f"    - {IGNORED}" in out
    assert "could not load" not in out  # vLLM loaded the checkpoint and ignored the tensor by name

    broken = tmp_path / "broken"  # vLLM refuses this one before it reads the renamed tensor
    shutil.copytree(clean, broken)
    _edit(broken, rename=SCALE)
    res = serve.serve_check(str(broken))
    assert not res["ok"] and res["unconsumed"] == [RENAMED]
