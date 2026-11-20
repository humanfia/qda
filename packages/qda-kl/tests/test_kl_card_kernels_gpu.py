"""`qda-kl kl` binds the kernels the checkpoint's printed serve line binds (GPU, the vLLM venv, one card).

Each case builds a W4A16 checkpoint with RTN on the default target of the card, reads it with ``qda-kl kl``
(``score.json`` records the kernels its engine bound) and starts the line ``qda build`` printed for it as written,
plus ``--port`` and a probe on ``PYTHONPATH`` (``kernel_probe/sitecustomize.py``) that records the kernels the
server's worker bound. Dense: Qwen3-0.6B (from the Hub). MoE: a two-layer random Qwen3-MoE made here, with
symmetric experts (the preset) and asymmetric ones.

    CUDA_VISIBLE_DEVICES=0 python -m pytest -m gpu -s packages/qda-kl/tests/test_kl_card_kernels_gpu.py
"""

import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

# No `torch.cuda.is_available()` here: called at collection it makes every later fork in this process unable
# to start CUDA, and test_gencheck.py starts a vLLM engine whose core process is forked.
pytestmark = pytest.mark.gpu

PROBE = Path(__file__).parent / "kernel_probe"
BIN = Path(sys.executable).parent


def _tiny_moe(out: Path) -> str:
    import transformers

    sys.path.insert(0, str(Path(__file__).parents[2] / "qda-quant" / "tests"))
    from engine_util import save_tiny_tokenizer

    cfg = transformers.AutoConfig.for_model(
        "qwen3_moe", vocab_size=128, hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
        num_experts=8, num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )  # fmt: skip
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(out)
    save_tiny_tokenizer(out)
    return str(out)


def _build(tmp: Path, model: str, fmt: str) -> Path:
    rng = random.Random(0)
    rows = [{"text": " ".join(f"w{rng.randrange(3, 127)}" for _ in range(80))} for _ in range(16)]
    (tmp / "calib.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp / "ckpt"
    recipe = {
        "model": model, "format": fmt, "out": str(out), "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp / "calib.jsonl"), "samples": 8, "seqlen": 64}},
    }  # fmt: skip
    (tmp / "recipe.yaml").write_text(json.dumps(recipe))
    subprocess.run([BIN / "qda", "build", tmp / "recipe.yaml"], check=True)
    return out


def _kl_kernels(tmp: Path, ckpt: Path) -> dict:
    prompts = tmp / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"w{i + 3} w7 w9 w11 w13"}) + "\n" for i in range(4)))
    out = tmp / "kl"
    env = {**os.environ, "QDA_CACHE": str(tmp / "cache")}  # the reference is this test's own
    subprocess.run([BIN / "qda-kl", "kl", ckpt, "--prompts", prompts, "--out", out], check=True, env=env)
    return json.loads((out / "score.json").read_text())["kernels"]


def _served_kernels(tmp: Path, ckpt: Path) -> dict:
    from qda_kl.vllm_hooks import merge_kernels
    from qda_quant.output.serve_command import serve_command_line

    line = serve_command_line(ckpt)
    print(f"card line: {line}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    seen = tmp / "probe"
    seen.mkdir()
    env = {**os.environ, "QDA_KERNEL_PROBE": str(seen), "PATH": f"{BIN}:{os.environ['PATH']}",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(PROBE), os.environ.get("PYTHONPATH")]))}  # fmt: skip
    log = (tmp / "serve.log").open("w")
    server = subprocess.Popen(
        ["bash", "-c", f"{line} --port {port}"], env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 900
        while not any(seen.iterdir()) and server.poll() is None and time.monotonic() < deadline:
            time.sleep(2)
        time.sleep(2)  # every worker of the engine has written
        found = [json.loads(p.read_text()) for p in seen.iterdir()]
        assert found, f"the server bound no kernels (exit {server.poll()}): {shlex.quote(str(tmp / 'serve.log'))}"
        return merge_kernels(found)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:  # the API server can outlive its engine core
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(30)


#: (case, format): the preset stores routed experts symmetric (Humming); explicit asym experts bind Marlin
CASES = [("dense", "w4a16"), ("moe", "w4a16"), ("moe-asym", "w4a16-asym-g128")]


@pytest.mark.parametrize(("case", "fmt"), CASES, ids=[c for c, _ in CASES])
def test_kl_binds_the_card_lines_kernels(tmp_path, case, fmt):
    model = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca" if case == "dense" else _tiny_moe(tmp_path / "tiny")
    ckpt = _build(tmp_path, model, fmt)
    planned = json.loads((ckpt / "build.json").read_text())["serve"]["kernels"]
    kl, served = _kl_kernels(tmp_path, ckpt), _served_kernels(tmp_path, ckpt)
    print(f"{case}: planned {planned}; qda-kl kl bound {kl}; card line bound {served}")
    assert kl == served
    assert "MacheteLinearKernel" not in served.get("dense", [])
