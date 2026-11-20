"""Stock vLLM 0.33.0 serves a scale_kd checkpoint, sidecar included (GPU; needs ``--vllm-python``).

A tiny Qwen3 goes through one staged build, ``[gptq, scale_kd]`` (trained on the CPU). vLLM must bind the
kernels ``build.json["serve"]`` names; it must ignore ``qda_state/``; and for W4A16 its prompt logprobs must
match a reference built from the stored tensors alone, so the trained scales are what it serves.
"""

import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest
import torch
import transformers
from qda_quant import REGISTRY, Format
from qda_quant.calib.sources import FileSource
from qda_quant.engine import packed as P
from qda_quant.engine.build import build
from qda_quant.recipe import Recipe
from qda_quant.target import serving_refusal

from ..engine_util import save_tiny_tokenizer, write_ids
from ..test_vllm_gpu import LOAD, _card_target, _reference
from ..tiny import config


def _run_group(argv: list[str], env: dict, tmp_path: Path, timeout: float = 600) -> tuple[str, str]:
    """Run `argv` in its own process group and kill the whole group when it ends, fails, times out or the test is
    interrupted, so no vLLM engine process outlives the test."""
    proc = subprocess.Popen(argv, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)  # fmt: skip
    try:
        return proc.communicate(timeout=timeout)
    finally:
        assert proc.pid > 1
        assert proc.pid != os.getpgid(0)
        assert proc.pid != os.getpgid(os.getppid())
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == proc.pid and fields[0] != "Z":
                    assert (stat.parent / "cwd").readlink().is_relative_to(tmp_path), stat.parent
            except FileNotFoundError:
                continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["w4a16", "w4a8"])
@pytest.mark.usefixtures("cpu_build")  # trained on the CPU while the run holds cards
def test_vllm_serves_scale_kd(fmt, tmp_path, vllm_python):
    if why := serving_refusal(Format.parse(fmt), [("cuda:0", torch.cuda.get_device_capability(0))]):
        pytest.skip(why)
    cfg = config("qwen3")
    cfg.max_position_embeddings = 512
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    m.save_pretrained(tmp_path / "src")
    sha = save_tiny_tokenizer(tmp_path / "src")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (64,), generator=g).tolist() for _ in range(8)], sha
    )
    kd = REGISTRY["scale_kd"](steps=6, batch_tokens=128, lr=1e-2, warmup=1, attn="eager")
    r = Recipe(
        steps=[REGISTRY["gptq"](impl="torch"), kd],
        calib=FileSource(path=str(tmp_path / "c.jsonl"), samples=8, seqlen=64),
    )
    out = build(
        str(tmp_path / "src"), r, format=fmt, out_dir=tmp_path / "out", devices=["cpu"],
        target=_card_target(), log=lambda _: None,
    )  # fmt: skip
    assert (out / P.STATE).is_file()
    assert json.loads((out / "build.json").read_text())["history"][-1]["scales_changed"][0] > 0

    ids = torch.randint(0, 128, (48,), generator=torch.Generator().manual_seed(2))
    env = {**os.environ, "PATH": f"{vllm_python.parent}:" + os.environ["PATH"]}
    if "CUDA_HOME" not in env and Path("/usr/local/cuda/bin").is_dir():  # else the host's nvcc on PATH
        env["PATH"] = f"{vllm_python.parent}:/usr/local/cuda/bin:" + os.environ["PATH"]
        env["CUDA_HOME"] = "/usr/local/cuda"
    argv = [str(vllm_python), str(LOAD), str(out), "--ids", ",".join(map(str, ids.tolist())),
            "--max-tokens", "1", "--prompt-logprobs"]  # fmt: skip
    stdout, stderr = _run_group(argv, env, tmp_path)
    got = json.loads(stdout.strip().splitlines()[-1])
    assert got["ok"], f"{got}\n{stderr[-3000:]}"  # a string, so pytest does not truncate the log
    assert got["bound"] == got["expected"]
    if fmt != "w4a16":
        return  # the W4A8 product is FP8: the bf16 reference below does not apply
    ref_model = _reference(tmp_path / "src", out)
    with torch.no_grad():
        lp = ref_model(input_ids=ids[None]).logits[0].float().log_softmax(-1)
    err = (torch.tensor(got["result"]["prompt_logprobs"]) - lp[:-1].gather(-1, ids[1:, None])[:, 0]).abs()
    assert err.mean() < 0.02 and err.max() < 0.15, (err.mean().item(), err.max().item())
