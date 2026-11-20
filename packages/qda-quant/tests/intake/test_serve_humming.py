"""W4A16 / W3A16 compressed-tensors checkpoints serve correctly on vLLM 0.33 with the kernels qda ships (GPU).

Run in the serving venv qda pins, on one card; the kernel-free reference runs in the quantization venv::

    CUDA_VISIBLE_DEVICES=0 python -m pytest -m gpu -s packages/qda-quant/tests/intake/test_serve_humming.py \\
        --serve-case w4a16=QUANT_DIR:BASE_DIR --serve-case w3a16=QUANT_DIR:BASE_DIR --ref-python QUANT_VENV_PYTHON

Serve settings (build.json ``serve``) and the kernel each must bind on sm90:

* W4A16 dense, default: no args -> ``HummingLinearKernel`` (Machete is disabled, ``serve_probe`` and every
  engine qda starts set ``VLLM_DISABLED_KERNELS=MacheteLinearKernel``; Humming is next on sm90);
* W4A16 dense, option ``:marlin``: ``{linear_backend: "marlin"}`` -> ``MarlinLinearKernel``;
* W3A16: no args -> ``HummingLinearKernel`` (the only int3 kernel).

Sequences: the bf16 base's prompts (6 short, 4 long up to ~7k tokens) plus its own 48-token greedy continuations,
scored teacher-forced (``benchmarks/serve_probe.py``, one vLLM process per model). Each setting must:

1. bind that kernel on every quantized Linear layer: the probe sums the kernel-class counts of every engine
   worker, so on a one-worker engine each count is the checkpoint's own and a pipeline's stages add up to it
   (a tensor-parallel engine holds every layer on each rank, so its counts would be the tensor-parallel size
   times the checkpoint's; these settings use no parallelism);
2. kernel correctness: agree with the SAME quantized weights decompressed to dense bf16 and run by HF transformers
   (``benchmarks/dequant_reference.py``) about as closely as vLLM bf16 agrees with HF bf16 on the base (the noise
   floor, measured in the same run): |NLL difference| < 0.01 nats / token, mean |logprob difference| < 6x the
   floor's, 99th percentile < 4x the floor's. Measured (H200, vLLM 0.33.0): floor mean 0.0028-0.0032, p99
   0.082-0.097; W4A16 Qwen3-14B 0.0046-0.0048 / 0.092-0.098 on all three kernels, NLL diff -0.001; W3A16 Qwen3-8B
   0.0103 / 0.224, -0.0004. (The bar factors were chosen after inspecting these first dequant measurements and
   fixed before the recorded run. The per-token max is not bounded: the HF dequant reference itself has rare ~9-nat
   outliers, so it is not a bit-exact oracle at every position.)
3. batched vs one-at-a-time scoring of the same tokens drifts about as little as the bf16 base's does (vLLM is not
   batch-invariant; a tile wrong only at some row counts shows up here);
4. smoke only, not a quality bar: the NLL increase over bf16 stays below gross-failure bounds (~3x / 2x the RTN
   values measured: W4A16 0.018 / 0.155 worst sequence, W3A16 0.158 / 1.135). Quality is the KL evaluator's job.

This end-to-end test is supplementary coverage, not a detector guaranteed to catch every BAD_TILES-class error: a
GEMM row error (8-59 % relative L2 in the old Humming bugs) has no guaranteed lower bound on target-token NLL, and a
rare error passes the mean / p99 / NLL checks (one 7.5-nat token among 11067 adds ~0.0007 to the mean; only a
drift-max check catches it). The direct detector is ``benchmarks/humming_tile_sweep.py``: every tile
config against an fp32 GEMM reference with a > 5 % per-row bar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"

#: (case id, checkpoint kind, LLM kwargs, kernel every quantized Linear must bind)
SETTINGS = [
    ("w4a16-marlin", "w4a16", {"linear_backend": "marlin"}, "MarlinLinearKernel"),
    ("w4a16-humming", "w4a16", {}, "HummingLinearKernel"),
    ("w3a16-humming", "w3a16", {}, "HummingLinearKernel"),
]
#: served vs dequantized reference: |NLL diff| bound, and multiples of the bf16 noise floor's mean / p99
MAX_REF_DNLL, REF_MEAN_FACTOR, REF_P99_FACTOR = 0.01, 6.0, 4.0
#: smoke bounds on the NLL increase over bf16 (token-weighted mean, worst sequence); see the module doc, item 4
SMOKE_DNLL = {"w4a16": (0.06, 0.35), "w3a16": (0.4, 2.3)}
#: batched-vs-alone drift allowed: this multiple of the bf16 base's own, or these floors (nats)
DRIFT_FACTOR = 4.0
MIN_MAX_DRIFT, MIN_MEAN_DRIFT = 0.1, 0.005


#: vLLM's fused modules: checkpoint projections served by one Linear
FUSED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}


def expected_linears(checkpoint: str) -> int:
    """Number of quantized Linear modules vLLM serves for a compressed-tensors checkpoint (MoE experts excluded).

    Counts the checkpoint's packed weights (``*.weight_packed``) outside ``.experts.``, grouping the projections
    vLLM fuses (`FUSED`), from the safetensors headers only. Assumes Qwen-style fusion (q/k/v -> qkv_proj, gate/up ->
    gate_up_proj), as vLLM 0.33 does for the Qwen3 checkpoints here. It is not a generic completeness check for other
    architectures (other fused or split projections would count differently). A quantized lm_head (a
    ``lm_head.weight_packed``) would be counted as one more module.
    """
    import struct

    names = []
    for f in sorted(Path(checkpoint).glob("*.safetensors")):
        with f.open("rb") as fh:
            names += json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    modules = set()
    for n in names:
        if n.endswith(".weight_packed") and ".experts." not in n:
            prefix, _, proj = n[: -len(".weight_packed")].rpartition(".")
            modules.add((prefix, FUSED.get(proj, proj)))
    return len(modules)


def _run(cmd: list[str], out: Path) -> dict:
    from qda_quant.output.vllm_env import loader_env

    if not out.exists():
        # a vLLM run (this venv's python) gets this venv's nvidia/*/lib on the loader path, as `qda build`'s serve
        # command sets it: Humming's NVRTC child process opens libnvrtc-builtins there. The --ref-python venv's
        # run keeps the inherited environment (its own libraries).
        env = loader_env(os.environ) if cmd[0] == sys.executable else None
        r = subprocess.run([*cmd, "--out", str(out)], capture_output=True, text=True, timeout=1500, env=env)
        assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.fixture(scope="session")
def opts(request) -> tuple[dict[str, tuple[str, str]], str | None]:
    """``--serve-case NAME=QUANT_DIR:BASE_DIR`` as {name: (quant, base)}, and ``--ref-python``."""
    specs = (s.split("=", 1) for s in request.config.getoption("--serve-case"))
    return {k: tuple(v.split(":", 1)) for k, v in specs}, request.config.getoption("--ref-python")


@pytest.fixture(scope="session")
def records(tmp_path_factory) -> Path:
    """One directory for the session's records: each base and reference is computed once for all settings."""
    return tmp_path_factory.mktemp("serve")


@pytest.mark.parametrize(("case", "kind", "llm_args", "kernel"), SETTINGS, ids=[s[0] for s in SETTINGS])
def test_checkpoint_serves(case, kind, llm_args, kernel, opts, records):
    cases, ref_python = opts
    if kind not in cases or not ref_python:
        pytest.skip(f"needs --serve-case {kind}=QUANT_DIR:BASE_DIR and --ref-python")
    sys.path.insert(0, str(BENCH))
    import serve_probe

    quant, base = cases[kind]
    probe = [sys.executable, str(BENCH / "serve_probe.py")]
    dequant = [ref_python, str(BENCH / "dequant_reference.py")]
    base_rec = records / f"{kind}-base.json"
    ref = _run([*probe, base], base_rec)
    hf_base = _run([*dequant, base, "--score", str(base_rec)], records / f"{kind}-hf-base.json")
    hf_quant = _run([*dequant, quant, "--score", str(base_rec)], records / f"{kind}-hf-dequant.json")
    args = [a for k, v in llm_args.items() for a in ("--arg", f"{k}={json.dumps(v)}")]
    got = _run([*probe, quant, "--score", str(base_rec), *args], records / f"{case}.json")

    floor = serve_probe.against(hf_base, ref)
    vs_ref = serve_probe.against(hf_quant, got)
    vs_bf16 = serve_probe.compare(ref, got)
    print(
        f"\n{case}: kernels {got['kernels']}\n  floor (vLLM bf16 vs HF bf16) {floor}"
        f"\n  served vs HF dequant {vs_ref}\n  vs bf16 {json.dumps(vs_bf16)}"
    )

    # summed over the engine's workers: a pipeline stage binds a disjoint part of the checkpoint
    bound = {k: n for k, n in got["kernels"].items() if not k.startswith("moe:")}
    want = {kernel: expected_linears(quant)}
    assert bound == want, (
        f"{case}: bound {got['kernels']}, want {want} (every quantized Linear of the checkpoint)"
    )
    assert abs(vs_ref["dnll"]) < MAX_REF_DNLL, (case, vs_ref)
    assert vs_ref["abs_mean"] < REF_MEAN_FACTOR * floor["abs_mean"], (case, vs_ref, floor)
    assert vs_ref["abs_p99"] < REF_P99_FACTOR * floor["abs_p99"], (case, vs_ref, floor)
    assert vs_bf16["drift_max"] < max(MIN_MAX_DRIFT, DRIFT_FACTOR * vs_bf16["base_drift_max"]), vs_bf16
    assert vs_bf16["drift_mean"] < max(MIN_MEAN_DRIFT, DRIFT_FACTOR * vs_bf16["base_drift_mean"]), vs_bf16
    smoke_mean, smoke_seq = SMOKE_DNLL[kind]
    assert vs_bf16["dnll_mean"] < smoke_mean and vs_bf16["dnll_max_seq"] < smoke_seq, vs_bf16
