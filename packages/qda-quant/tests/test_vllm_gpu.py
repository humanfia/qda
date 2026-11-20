"""Stock vLLM 0.33.0 loads our exports and computes what the build propagated (GPU).

Tiny random models, exported with the real steps and Writer, are served by vLLM with their own
``build.json["serve"]`` settings; the kernels vLLM binds must be the planned ones and its prompt logprobs
must match the served (dequantized) model run in transformers. Covers asymmetric fused-expert MoE, the
default preset ``w4a16`` on MoE, which transformers cannot decompress. vLLM runs in its own venv, given by
``--vllm-python PATH`` (see conftest.py); the test skips without it.
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from qda_quant import REGISTRY, Target
from qda_quant.arch import describe
from qda_quant.output.export import Writer
from qda_quant.solve.quant import int8_token, int8_token_parts

from .ctx import contexts
from .tiny import config

LOAD = Path(__file__).parents[1] / "benchmarks" / "vllm_load.py"


def _card_target() -> Target:
    """The target of a build on this card (sm80 on an A100, sm90 on an H100): vLLM binds that table's kernels."""
    return Target.on_cards(None, [("cuda:0", torch.cuda.get_device_capability(0))])


def _export(kind, fmt, tmp, head=None):
    cfg = config(kind)
    cfg.max_position_embeddings = 512
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    m.save_pretrained(tmp / "src")
    m = transformers.AutoModelForCausalLM.from_pretrained(tmp / "src", dtype=torch.bfloat16).eval()
    m.config._experts_implementation = "eager"
    graph = describe(m, verify=False)
    target = _card_target()
    run = SimpleNamespace(qparams={}, target=target)  # the build's qparams dict and target (BuildRun)
    if head:
        REGISTRY["head"](format=head).apply_model(m, graph, run)
    w = Writer(tmp / "out", tmp / "src", graph, fmt, target)
    for ctx in contexts(m, fmt, tokens=256):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        w.add_layer(ctx)
    w.add_model(m, run.qparams)
    return m, w.finish(m.config, None, {})


def _reference(src: Path, ckpt: Path) -> torch.nn.Module:
    """The float source model with every quantized weight replaced by ``(q - z) * s`` (bf16) computed here
    from the checkpoint's stored tensors (`qda_quant.solve.packing`, which test_packing.py holds to compressed-tensors); per-expert
    tensors are stacked back into the fused in-memory parameters."""
    from qda_quant.solve.packing import unpack as unpack_from_int32
    from safetensors import safe_open

    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    m.config._experts_implementation = "eager"
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for k, f in idx.items():
        with safe_open(ckpt / f, "pt") as h:
            tensors[k] = h.get_tensor(k)
    groups = json.loads((ckpt / "config.json").read_text())["quantization_config"]["config_groups"]
    deq = {}
    for k in [k for k in tensors if k.endswith(".weight_packed")]:
        mod = k[: -len(".weight_packed")]
        grp = (
            groups["group_head"] if "group_head" in groups and mod.endswith("lm_head") else groups["group_0"]
        )
        bits = grp["weights"]["num_bits"]
        shape = torch.Size(tensors[mod + ".weight_shape"].tolist())
        q = unpack_from_int32(tensors[k], bits, shape).float()
        s = tensors[mod + ".weight_scale"].float()
        g = shape[1] // s.shape[1]
        if mod + ".weight_zero_point" in tensors:
            z = unpack_from_int32(
                tensors[mod + ".weight_zero_point"], bits, torch.Size((shape[0], s.shape[1])), packed_dim=0
            )
            q = q - z.float().repeat_interleave(g, -1)
        deq[mod] = (q * s.repeat_interleave(g, -1)).to(torch.bfloat16)
    sd = m.state_dict()
    with torch.no_grad():
        for name, p in sd.items():
            mod = name.rpartition(".")[0]
            if name.endswith("experts.gate_up_proj"):
                e = p.shape[0]
                p.copy_(
                    torch.stack(
                        [
                            torch.cat([deq[f"{mod}.{i}.gate_proj"], deq[f"{mod}.{i}.up_proj"]])
                            for i in range(e)
                        ]
                    )
                )
            elif name.endswith("experts.down_proj"):
                p.copy_(torch.stack([deq[f"{mod}.{i}.down_proj"] for i in range(p.shape[0])]))
            elif name.endswith(".weight") and mod in deq:
                p.copy_(deq[mod])
    return m


@pytest.mark.gpu
@pytest.mark.parametrize(("kind", "fmt", "head"), [("qwen3_moe", "w4a16", None), ("qwen3_moe", "w4a16-sym-g128", None),
                                                   ("qwen3", "w3a16", "w4a16-sym-g64")])  # fmt: skip
def test_vllm_serves_our_export(kind, fmt, head, tmp_path, vllm_python):
    served, out = _export(kind, fmt, tmp_path, head)
    ids = torch.randint(0, 128, (48,), generator=torch.Generator().manual_seed(2))
    env = {**os.environ, "PATH": f"{vllm_python.parent}:" + os.environ["PATH"]}
    if "CUDA_HOME" not in env and Path("/usr/local/cuda/bin").is_dir():  # else the host's nvcc on PATH
        env["PATH"] = f"{vllm_python.parent}:/usr/local/cuda/bin:" + os.environ["PATH"]
        env["CUDA_HOME"] = "/usr/local/cuda"
    r = subprocess.run([str(vllm_python), str(LOAD), str(out), "--ids", ",".join(map(str, ids.tolist())),
                        "--max-tokens", "1", "--prompt-logprobs"], env=env, capture_output=True, text=True)  # fmt: skip
    assert r.stdout.strip(), f"vllm_load.py printed no result (exit {r.returncode}): {r.stderr[-3000:]}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["ok"], f"{res}\n{r.stderr[-3000:]}"  # a string, so pytest does not truncate the log
    assert res["bound"] == res["expected"]
    ref_model = _reference(tmp_path / "src", out)  # independent of qda_quant: built from the stored tensors
    with torch.no_grad():
        logits = ref_model(input_ids=ids[None]).logits[0].float().log_softmax(-1)
    ref = logits[:-1].gather(-1, ids[1:, None])[:, 0]
    got = torch.tensor(res["result"]["prompt_logprobs"])
    err = (got - ref).abs()
    # bf16 serving vs fp32 reference: small; a wrong zero point / expert layout gives O(1) errors
    assert err.mean() < 0.02 and err.max() < 0.15, (err.mean().item(), err.max().item())


@pytest.mark.gpu
def test_int8_cut_is_vllm_scaled_int8_quant(tmp_path, vllm_python):
    """`int8_token_parts` on a card gives the bits of vLLM's compiled per-token op: codes, scales, dequantized."""
    g = torch.Generator().manual_seed(0)
    xs = []
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        xs += [torch.randn(s, generator=g).to(dt) for s in ((1, 8), (7, 64), (3, 5, 128), (4096, 4096))]
        outliers, zero_row = torch.randn(64, 512, generator=g), torch.randn(16, 512, generator=g)
        outliers[:, 7] *= 1000
        zero_row[3] = 0
        ties = torch.randint(-1000, 1000, (256, 4096), generator=g).float() / 8  # many exact x.5 codes
        xs += [t.to(dt) for t in (outliers, zero_row, ties)]
    pinned = torch.tensor(  # the rows of test_target_quant.py::test_int8_cut_rounds_as_the_kernel_does
        [
            [1.0, -0.9409448504447937, 0.0],
            [1.003999948501587, -0.9921416640281677, 0.0],
            [127.0, 0.5, 1.5],
            [0.0, 0.0, 0.0],
            [1e-40, -1e-40, 0.0],
        ]
    )
    tiny = torch.randn(16, 256, generator=g) * 1e-38  # subnormal scales; 127 / absmax overflows next to zeros
    tiny[:, 3:9] = 0
    xs += [pinned, tiny, tiny.to(torch.bfloat16)]
    torch.save(xs, tmp_path / "x.pt")
    run = (
        "import sys, torch; from vllm import _custom_ops as ops; xs = torch.load(sys.argv[1]);"
        "torch.save([ops.scaled_int8_quant(x.cuda())[:2] for x in xs], sys.argv[2])"
    )
    subprocess.run([str(vllm_python), "-c", run, tmp_path / "x.pt", tmp_path / "y.pt"], check=True)
    for x, (q_kernel, s_kernel) in zip(xs, torch.load(tmp_path / "y.pt"), strict=True):
        x, s_kernel = x.cuda(), s_kernel.reshape(*x.shape[:-1], 1)  # noqa: PLW2901
        q, s = int8_token_parts(x)
        assert torch.equal(q.to(torch.int8), q_kernel) and torch.equal(s, s_kernel)
        assert torch.equal(int8_token(x), (q_kernel.float() * s_kernel).to(x.dtype))
