"""Score token sequences with HF transformers on a checkpoint's weights dequantized to dense bf16 (no vLLM kernel).

The kernel-free reference for ``serve_probe.py``: the same quantized weights, dequantized by the library's reader
(zero points included) and run as plain bf16 matmuls. Served-vs-reference then isolates the serving kernel; run on
the bf16 base, the same comparison measures the vLLM-vs-HF noise floor. Runs in the quantization venv::

    python dequant_reference.py MODEL_DIR --score base.json --out ref.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    """Write {"model", "seqs", "batched": per-token logprobs of each sequence after its first token}."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--score", required=True, help="a serve_probe record whose sequences to score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from qda_quant.engine.packed import Packed, load_float
    from qda_quant.step import Refused
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig

    try:
        ck = Packed(args.model)
    except Refused:  # a float model, or a quantized layout the reader does not take
        ck = None
    if ck is not None:
        model = load_float(ck, torch.device("cuda"))
    else:
        config = json.loads((Path(args.model) / "config.json").read_text())
        kw = {}
        if config.get("quantization_config", {}).get("quant_method") == "compressed-tensors":
            kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda", **kw
        ).eval()
    seqs = json.loads(Path(args.score).read_text())["seqs"]
    out = []
    with torch.inference_mode():
        for ids in seqs:
            x = torch.tensor([ids], device="cuda")
            logits = model(x).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(1, x[0, 1:, None])[:, 0]
            out.append(lp.tolist())
            del logits
    Path(args.out).write_text(json.dumps({"model": args.model, "seqs": seqs, "batched": out}))
    print(f"scored {len(seqs)} sequences, {sum(map(len, out))} tokens -> {args.out}")


if __name__ == "__main__":
    main()
