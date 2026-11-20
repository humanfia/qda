"""The quickstart recipe at ``w4a4`` builds an NVFP4 checkpoint and stock vLLM 0.33 serves it (GPU).

Qwen3-0.6B from ``--model-dir qwen3-0.6b``, ``configs/quickstart.yaml`` with only ``format: w4a4`` changed (its
``hmse`` clip runs as ``fp4hmse``), built by ``qda build`` on the card that will serve it. The checkpoint's
``build.json`` names NVFP4 and the card's dense kernel; vLLM, started as ``benchmarks/vllm_load.py`` starts it,
binds that kernel and generates text. Skips off sm100 / sm103 / sm120 (the cards that serve W4A4,
``docs/formats.md``), and without ``--model-dir qwen3-0.6b`` or ``--vllm-python``.
"""

import gc
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml
from qda_quant.cli import main

LOAD = Path(__file__).parents[1] / "benchmarks" / "vllm_load.py"

#: The cards vLLM 0.33 serves NVFP4 on, and the dense kernel each row plans: FlashInfer's CuTe-DSL kernel on a
#: B200 / GB300, its CUTLASS one on an RTX PRO 6000 (10.x only for the CuTe-DSL kernel; `Target` rows).
DENSE_KERNEL = {
    (10, 0): "FlashInferCuteDslNvFp4LinearKernel",
    (10, 3): "FlashInferCuteDslNvFp4LinearKernel",
    (12, 0): "FlashInferCutlassNvFp4LinearKernel",
}


@pytest.fixture
def nvfp4_card() -> tuple[int, int]:
    """The card's compute capability, or a skip off sm100 / sm103 / sm120. Read in the test: asking for the
    capability at collection initializes CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("needs a Blackwell card (sm100, sm103 or sm120)")
    capability = torch.cuda.get_device_capability()
    if capability not in DENSE_KERNEL:
        pytest.skip(f"w4a4 is served on sm100/sm103/sm120, not sm{capability[0]}{capability[1]}")
    return capability


@pytest.mark.gpu
def test_w4a4_quickstart_builds_and_vllm_serves_it(nvfp4_card, real_model, vllm_python, tmp_path):
    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "quickstart.yaml").read_text()
    )
    assert "@" in shipped["model"]  # the recipe pins the weights' revision
    out = tmp_path / "checkpoint"
    config = tmp_path / "w4a4.yaml"
    config.write_text(
        yaml.safe_dump({**shipped, "format": "w4a4", "model": str(real_model("qwen3-0.6b")), "out": str(out)})
    )
    assert main(["build", str(config)]) == 0

    record = json.loads((out / "build.json").read_text())
    assert record["format"] == "w4a4-nvfp4"
    assert record["target"].split("@")[0] == f"sm{nvfp4_card[0]}{nvfp4_card[1]}"
    assert record["serve"]["kernels"]["dense"] == [DENSE_KERNEL[nvfp4_card]]

    # the build's blocks go back to the driver: the vLLM child sizes its engine from the free memory
    gc.collect()
    torch.cuda.empty_cache()
    env = {**os.environ, "PATH": f"{vllm_python.parent}:" + os.environ["PATH"]}
    if "CUDA_HOME" not in env and Path("/usr/local/cuda/bin").is_dir():  # else the host's nvcc on PATH
        env["PATH"] = f"{vllm_python.parent}:/usr/local/cuda/bin:" + os.environ["PATH"]
        env["CUDA_HOME"] = "/usr/local/cuda"
    r = subprocess.run(
        [str(vllm_python), str(LOAD), str(out), "--max-tokens", "8"],
        env=env,
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert r.stdout.strip(), f"vllm_load.py printed no result (exit {r.returncode}): {r.stderr[-3000:]}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["ok"], f"{res}\n{r.stderr[-3000:]}"  # a string, so pytest does not truncate the log
    assert res["bound"] == res["expected"]
    assert res["bound"]["dense"] == [DENSE_KERNEL[nvfp4_card]]
    assert res["result"]["text"].strip()
    print(f"bound {res['bound']['dense']}; text {res['result']['text']!r}")  # `-s` shows the measured line
