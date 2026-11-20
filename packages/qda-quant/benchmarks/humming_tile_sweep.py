"""Sweep Humming's W4A16 / W3A16 GEMMs against an fp32 reference, per tile config its heuristic picks.

Humming (the kernel library vLLM >= 0.32 prefers on sm90 for W4A16 dense and every MoE layer) chooses a
tuning config per token count from tables built by ``humming.tune``. Some configs have computed wrong
outputs on sm90 (vLLM 0.27.1 / humming 0.1.10: block M 56, and TMA multicast of A). This script runs every
(shape, dtype, M) of a model family through the kernel exactly as vLLM calls it (dense: heuristic picked at
run time; MoE: indexed gemm with the default tuning table and ``moe_align_block_size``) and compares with
``dequantized W (fp32) @ x (fp32)``. One JSON line per case: the config picked, mean / max relative error
per output row, and the number of rows above ``--bad``.

Runs in a vLLM venv (any of 0.27.1 / 0.33); it does not import qda::

    python humming_tile_sweep.py --models q8b q30a3b q38 --out sweep.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

#: (name, N, K) of the dense projections Humming serves, and (name, E, top_k, N, K) of the MoE gemms
MODELS: dict[str, dict[str, list]] = {
    "q8b": {  # Qwen3-8B: hidden 4096, inter 12288, 32 q / 8 kv heads x 128
        "dense": [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 24576, 4096), ("down", 4096, 12288)],
        "moe": [],
    },
    "q30a3b": {  # Qwen3-30B-A3B: hidden 2048, 32 q / 4 kv heads x 128, 128 experts top 8, moe inter 768
        "dense": [("qkv", 5120, 2048), ("o", 2048, 4096)],
        "moe": [("w13", 128, 8, 1536, 2048), ("w2", 128, 8, 2048, 768)],
    },
    "q38": {  # Qwen3.8-27B: hidden 5120, inter 17408, 24 q (gated) / 4 kv x 256; DeltaNet 16 k / 48 v x 128
        "dense": [
            ("qkv", 14336, 5120),
            ("o", 5120, 6144),
            ("gate_up", 34816, 5120),
            ("down", 5120, 17408),
            ("in_proj_qkvz", 16384, 5120),
            ("out_proj", 5120, 6144),
        ],
        "moe": [],
    },
    "q36": {  # Qwen3.6-35B-A3B experts (the old E79 MoE case): hidden 2048, 256 experts top 8, moe inter 512
        "dense": [],
        "moe": [("w13", 256, 8, 1024, 2048), ("w2", 256, 8, 2048, 512)],
    },
}

#: token counts: the card's list (powers of 2 and non-multiples of 8), every M up to 128 (the block-M 56
#: range is 49-56), and a stride through 2048 (multicast was >= 536)
DEFAULT_M = sorted(
    {1, 3, 7, 9, 17, 33, 65, 129, 257, 513, 1025, 2047}
    | {2**i for i in range(12)}
    | set(range(1, 129))
    | set(range(129, 2049, 29))
    | {2048}
)
#: MoE token counts: valid_shape_m = tokens * top_k picks the tile; 929-1104 tokens was the bad range
DEFAULT_MOE_M = sorted(
    {1, 3, 7, 9, 17, 33, 65, 129, 257, 513, 1025, 2047}
    | {2**i for i in range(12)}
    | set(range(64, 2049, 24))
    | set(range(900, 1130, 7))
)


def humming_version() -> str:
    """The installed humming-kernels version."""
    import humming

    return getattr(humming, "__version__", "?")


def make_layer(n: int, k: int, bits: int, experts: int | None, pad_n: int, pad_k: int):
    """A HummingLayer of random weights quantized to uint``bits`` g128, transformed for forward.

    Returns the layer and its dequantized fp32 weight ([E,] N, K), from the loaded (pre-transform) params.
    """
    from humming.layer import HummingLayer
    from humming.utils.weight import dequantize_weight

    torch.manual_seed(n * 7 + k * 13 + bits + (experts or 0))
    shape = (experts, n, k) if experts else (n, k)
    w = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.02
    layer = HummingLayer(
        shape_n=n,
        shape_k=k,
        weight_config={"quant_method": "humming", "dtype": f"uint{bits}", "group_size": 128},
        num_experts=experts,
        pad_n_to_multiple=pad_n,
        pad_k_to_multiple=pad_k,
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_unquantized(w)
    params = {name: p.data for name, p in layer.named_parameters()}
    schema = layer.weight_schema
    kwargs = {
        "weight": params["weight"],
        "weight_scale": params.get("weight_scale"),
        "zero_point": params.get("zero_point"),
        "dtype": schema.b_dtype,
        "packed": True,
    }
    sig = inspect.signature(dequantize_weight).parameters
    for extra in ("global_scale", "weight_scale_2"):
        if extra in sig:
            kwargs[extra] = params.get(extra)
    if "group_size" in sig:
        kwargs["group_size"] = 128
    kwargs = {key: v for key, v in kwargs.items() if key in sig}
    w_ref = dequantize_weight(**kwargs).float().reshape(shape)
    del w, params
    layer.transform()
    return layer, w_ref


def layer_meta(layer):
    """The layer's Humming layer config (0.1.10: humming_metas[""]; 0.1.19: humming_config)."""
    metas = getattr(layer, "humming_metas", None) or {}
    return metas.get("") if metas else getattr(layer, "humming_config", None)


def picked_config(layer, gemm_type: str, shape_m: int) -> dict:
    """The tuning config Humming's heuristic picks for ``shape_m`` rows (the fields that shape the tile)."""
    from humming.tune import get_heuristics_config

    meta = layer_meta(layer)
    cfg = get_heuristics_config(meta, shape_m=shape_m, gemm_type=gemm_type)
    return {
        key: (list(v) if isinstance(v, tuple) else v)
        for key, v in dict(cfg).items()
        if key
        in (
            "block_shape",
            "warp_shape",
            "multi_cast_size_a",
            "multi_cast_size_b",
            "use_stream_k",
            "num_stages",
            "use_tma",
            "num_ctas_per_sm",
            "raster_group_m",
        )
    }


def row_errors(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Relative L2 error of each output row against the reference."""
    return (out.float() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)


def summarize(errs: torch.Tensor, bad: float) -> dict:
    """Worst over repeats of row errors [repeats, rows]: the worst run's mean, the worst row, bad rows / runs."""
    return {
        "mean": errs.mean(dim=1).max().item(),
        "max": errs.max().item(),
        "n_bad": int((errs.max(dim=0).values > bad).sum()),
        "bad_runs": int((errs > bad).any(dim=1).sum()),
        "repeats": errs.shape[0],
    }


def run_dense(layer, w_ref: torch.Tensor, m: int, repeats: int) -> torch.Tensor:
    """Row errors [repeats, m] of a dense gemm with ``m`` rows."""
    x = torch.randn(m, w_ref.shape[1], dtype=torch.bfloat16, device="cuda")
    ref = x.float() @ w_ref.T
    return torch.stack([row_errors(layer(x)[:, : w_ref.shape[0]], ref) for _ in range(repeats)])


def zipf_topk(tokens: int, experts: int, top_k: int) -> torch.Tensor:
    """Skewed routing: expert weights 1/rank in a random order, ``top_k`` distinct per token.

    A few experts then fill whole blocks of rows, as real routers do; uniform routing leaves most rows of
    every block empty (rows 40-55 of the 56-row tile were the wrong ones).
    """
    weights = 1.0 / torch.arange(1, experts + 1, device="cuda", dtype=torch.float32)
    weights = weights[torch.randperm(experts, device="cuda")]
    return torch.multinomial(weights.expand(tokens, experts), top_k, replacement=False).to(torch.int32)


def moe_table(layer):
    """The indexed-gemm tuning table vLLM builds for an experts layer ([lo, hi, config] per valid_shape_m)."""
    from humming.tune import get_heuristics_config

    return get_heuristics_config(layer_meta(layer), gemm_type="indexed")


def run_moe(layer, w_ref: torch.Tensor, tokens: int, top_k: int, table, block_table, repeats: int):
    """One indexed MoE gemm as vLLM's HummingExpertsBase calls it; returns (row errors, valid_shape_m, block M).

    Rows are aligned to the block M of ``block_table``: w13's for both gemms on vLLM 0.27.1, each gemm's own on
    vLLM 0.33 (``vllm_realigns_w2``).
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    e, n, k = w_ref.shape
    x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    topk_ids = zipf_topk(tokens, e, top_k)
    valid = tokens * top_k
    block_m = next((c["block_shape"][0] for lo, hi, c in block_table if lo < valid <= hi), 64)
    sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
        topk_ids=topk_ids, block_size=block_m, num_experts=e, expert_map=None, ignore_invalid_experts=True
    )
    flat = topk_ids.flatten().long()
    ref = torch.empty(valid, n, dtype=torch.float32, device="cuda")
    for ex in flat.unique().tolist():
        rows = (flat == ex).nonzero().squeeze(1)
        ref[rows] = x[rows // top_k].float() @ w_ref[ex].T
    out = torch.empty(valid, n, dtype=torch.bfloat16, device="cuda")
    errs = []
    for _ in range(repeats):
        out.zero_()
        layer(
            x,
            outputs=out,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid,
            compute_config=json.dumps(
                {"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "indexed"}
            ),
            tuning_config=json.dumps(table),
        )
        errs.append(row_errors(out, ref))
    return torch.stack(errs), valid, block_m


def vllm_realigns_w2() -> bool:
    """Whether the installed vLLM aligns w2's rows to w2's own block M (0.33) or reuses w13's (0.27.1)."""
    import inspect

    from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as fhm

    return "w2_block_size" in inspect.getsource(fhm)


def table_config(table, valid: int) -> dict:
    """The tile fields of the table entry serving ``valid`` rows."""
    cfg = next((c for lo, hi, c in table if lo < valid <= hi), None)
    if cfg is None:
        return {"block_shape": None}
    return {
        key: cfg.get(key)
        for key in ("block_shape", "warp_shape", "multi_cast_size_a", "use_stream_k")
        if key in cfg
    }


def main() -> None:
    """Run the sweep and write one JSON line per case."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3])
    ap.add_argument("--m", nargs="*", type=int, default=None, help="dense token counts (default: DEFAULT_M)")
    ap.add_argument(
        "--moe-m", nargs="*", type=int, default=None, help="MoE token counts (default: DEFAULT_MOE_M)"
    )
    ap.add_argument(
        "--repeats", type=int, default=3, help="runs per case; the worst is kept (intermittent bugs)"
    )
    ap.add_argument(
        "--repeats-large", type=int, default=10, help="runs per case at >= 512 rows (multicast range)"
    )
    ap.add_argument("--bad", type=float, default=0.05, help="a row is bad above this relative error")
    ap.add_argument("--only", nargs="*", default=None, help="only these layer names (qkv, o, w13, ...)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm

    torch.cuda.set_device(0)
    head = {
        "kind": "env",
        "vllm": vllm.__version__,
        "humming": humming_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "sm": "".join(map(str, torch.cuda.get_device_capability(0))),
    }
    print(json.dumps(head), flush=True)
    dense_m = args.m or DEFAULT_M
    moe_m = args.moe_m or DEFAULT_MOE_M
    t0 = time.time()
    own_w2 = vllm_realigns_w2()
    head["w2_own_alignment"] = own_w2

    def repeats(rows: int) -> int:
        return args.repeats_large if rows >= 512 else args.repeats

    with Path(args.out).open("w") as f:
        f.write(json.dumps(head) + "\n")

        def emit(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec["n_bad"]:
                print(json.dumps(rec), flush=True)

        for model in args.models:
            w13_table = None
            for bits in args.bits:
                for name, n, k in MODELS[model]["dense"]:
                    if args.only and name not in args.only:
                        continue
                    layer, w_ref = make_layer(n, k, bits, None, 256, 128)
                    for m in dense_m:
                        err = run_dense(layer, w_ref, m, repeats(m))
                        emit(
                            {
                                "kind": "dense",
                                "model": model,
                                "layer": name,
                                "bits": bits,
                                "n": n,
                                "k": k,
                                "m": m,
                                "config": picked_config(layer, "dense", m),
                                **summarize(err, args.bad),
                                "rows": m,
                            }
                        )
                    del layer, w_ref
                    torch.cuda.empty_cache()
                    print(f"[{time.time() - t0:7.1f}s] {model} w{bits} {name} done", flush=True)
                for name, e, top_k, n, k in MODELS[model]["moe"]:
                    if args.only and name not in args.only:
                        continue
                    layer, w_ref = make_layer(n, k, bits, e, 1, 1)
                    table = moe_table(layer)
                    if name == "w13":
                        w13_table = table
                    # w2 sees one row per (token, expert): its tokens are the routed rows, top_k 1
                    kk = 1 if name == "w2" else top_k
                    for tokens in moe_m:
                        t = tokens * top_k if name == "w2" else tokens
                        err, valid, block_m = run_moe(
                            layer,
                            w_ref,
                            t,
                            kk,
                            table,
                            table if own_w2 else w13_table,
                            repeats(tokens * top_k),
                        )
                        emit(
                            {
                                "kind": "moe",
                                "model": model,
                                "layer": name,
                                "bits": bits,
                                "n": n,
                                "k": k,
                                "experts": e,
                                "top_k": top_k,
                                "tokens": tokens,
                                "valid_shape_m": valid,
                                "block_m": block_m,
                                "config": table_config(table, valid),
                                **summarize(err, args.bad),
                                "rows": valid,
                            }
                        )
                    del layer, w_ref
                    torch.cuda.empty_cache()
                    print(f"[{time.time() - t0:7.1f}s] {model} w{bits} {name} done", flush=True)
    print(f"done in {time.time() - t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
