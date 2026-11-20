"""Serve one checkpoint in vLLM and record what it bound and how it scores; compare against a bf16 base's record.

Run in the serving venv, one model per process (env settings must precede the vLLM import)::

    python serve_probe.py BASE_DIR --out base.json
    python serve_probe.py QUANT_DIR --out q.json --score base.json --arg linear_backend='"marlin"'
    python serve_probe.py --compare base.json q.json

The record: the kernel classes bound (dense ``scheme.kernel`` and MoE experts, counted over every engine
worker and stored with the worker count), greedy continuations (one batch and one prompt at a time), and the
per-token logprobs of fixed token sequences scored as one batch and one at a time (prefill steps from tens to
thousands of rows). Without ``--score`` the sequences are this model's prompts plus its own greedy
continuations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PARAGRAPHS = [
    "The history of mathematics spans thousands of years, from counting marks on bones to modern abstract "
    "algebra. Early civilizations developed arithmetic for trade and astronomy, and the Greeks introduced proof.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly blue and red light, "
    "and the Calvin cycle fixes carbon dioxide into sugars that feed nearly every food chain on Earth.",
    "A compiler translates source code into machine instructions. It parses the program, builds an intermediate "
    "representation, optimizes it, allocates registers, and finally emits code for the target processor.",
    "The Silk Road was a network of trade routes linking China with the Mediterranean. Merchants carried silk, "
    "spices, and paper westward, while ideas, religions, and diseases travelled in both directions.",
    "In a market economy, prices coordinate the decisions of millions of buyers and sellers. When a good becomes "
    "scarce its price rises, which encourages producers to supply more and consumers to use less.",
    "Plate tectonics explains earthquakes, volcanoes, and mountain ranges. The lithosphere is broken into plates "
    "that move a few centimetres a year over the hotter, more fluid mantle beneath them.",
]


def prompts() -> list[str]:
    """Short prompts, and long ones (up to ~7k tokens) so prefill steps span small and large row counts."""
    short = [p.split(". ")[0] + "." for p in PARAGRAPHS]
    long = [
        " ".join(PARAGRAPHS[i:] + PARAGRAPHS[:i]) * reps for i, reps in [(0, 1), (2, 4), (4, 12), (1, 30)]
    ]
    return short + long


def census(model) -> dict[str, int]:
    """Count the kernel classes a vLLM model bound: dense ``scheme.kernel``, MoE ``quant_method`` experts."""
    names: dict[str, int] = {}
    for mod in model.modules():
        found = []
        kernel = getattr(getattr(mod, "scheme", None), "kernel", None)
        if kernel is not None:
            found.append(type(kernel).__name__)
        qm = getattr(mod, "quant_method", None)
        for attr in ("experts_cls", "moe_kernel", "fused_experts"):
            v = getattr(qm, attr, None)
            if v is not None:
                found.append("moe:" + (v if isinstance(v, type) else type(v)).__name__)
        for n in found:
            names[n] = names.get(n, 0) + 1
    return names


def merge_census(per_worker: list[dict[str, int]]) -> dict[str, int]:
    """Sum the kernel-class counts over the engine's workers.

    A pipeline stage holds a different set of layers, so the sum is one count per bound module of the
    checkpoint. Under tensor parallelism every worker holds every layer, so there each count is per rank,
    times the tensor-parallel size.
    """
    names: dict[str, int] = {}
    for worker in per_worker:
        for n, count in worker.items():
            names[n] = names.get(n, 0) + count
    return names


def probe(model: str, score: list[list[int]] | None, max_new: int, llm_args: dict) -> dict:
    """Load ``model`` in vLLM and record kernels, greedy continuations and logprobs (see module doc)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, **llm_args)
    per_worker = llm.apply_model(census)  # one dict per worker; a pipeline stage holds different layers
    out: dict = {
        "model": model,
        "args": llm_args,
        "workers": len(per_worker),
        "kernels": merge_census(per_worker),
    }
    ps = prompts()
    greedy = SamplingParams(temperature=0, max_tokens=max_new)
    results = llm.generate(ps, greedy)
    out["greedy"] = [list(o.outputs[0].token_ids) for o in results]
    out["greedy_alone"] = [list(llm.generate([p], greedy)[0].outputs[0].token_ids) for p in ps]
    out["seqs"] = score or [list(o.prompt_token_ids) + list(o.outputs[0].token_ids) for o in results]
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    seqs = [{"prompt_token_ids": ids} for ids in out["seqs"]]

    def logprobs(res) -> list[list[float]]:
        return [[next(iter(d.values())).logprob for d in r.prompt_logprobs[1:]] for r in res]

    out["batched"] = logprobs(llm.generate(seqs, params))
    out["alone"] = [logprobs(llm.generate([s], params))[0] for s in seqs]
    return out


def drift(rec: dict) -> tuple[float, float]:
    """Max and mean per-token |logprob| difference between batched and one-at-a-time scoring."""
    d = [
        abs(a - b)
        for s, t in zip(rec["batched"], rec["alone"], strict=True)
        for a, b in zip(s, t, strict=True)
    ]
    return max(d), sum(d) / len(d)


def nll(rec: dict) -> list[float]:
    """Mean negative logprob per sequence (batched scoring)."""
    return [-sum(s) / len(s) for s in rec["batched"]]


def compare(base: dict, rec: dict) -> dict:
    """Numbers of one record against its bf16 base's (same sequences)."""
    assert rec["seqs"] == base["seqs"], "records scored different sequences"
    d_nll = [a - b for a, b in zip(nll(rec), nll(base), strict=True)]
    tokens = [len(s) for s in rec["batched"]]
    lp = [
        abs(a - b)
        for s, t in zip(rec["batched"], base["batched"], strict=True)
        for a, b in zip(s, t, strict=True)
    ]
    agree = [a == b for a, b in zip(rec["greedy"], base["greedy"], strict=True)]
    return {
        "kernels": rec["kernels"],
        "args": rec.get("args"),
        "dnll_mean": sum(x * n for x, n in zip(d_nll, tokens, strict=True)) / sum(tokens),
        "dnll_max_seq": max(d_nll),
        "abs_dlogprob_vs_base_mean": sum(lp) / len(lp),
        "greedy_equal_base": f"{sum(agree)}/{len(agree)}",
        "drift_max": drift(rec)[0],
        "drift_mean": drift(rec)[1],
        "base_drift_max": drift(base)[0],
        "base_drift_mean": drift(base)[1],
    }


def against(ref: dict, rec: dict) -> dict:
    """Per-token logprob agreement of a served record with a reference record on the same sequences.

    Args:
        ref: e.g. ``dequant_reference.py``'s scoring of the same weights without the serving kernel.
        rec: a serve_probe record (its batched scoring is compared).

    Returns:
        dnll: token-weighted mean NLL difference (rec - ref); abs_mean / p99 / max: |logprob difference| per token.
    """
    assert rec["seqs"] == ref["seqs"], "records scored different sequences"
    d = [
        a - b for s, t in zip(rec["batched"], ref["batched"], strict=True) for a, b in zip(s, t, strict=True)
    ]
    ad = sorted(abs(x) for x in d)
    return {
        "dnll": -sum(d) / len(d),
        "abs_mean": sum(ad) / len(ad),
        "abs_p99": ad[int(0.99 * (len(ad) - 1))],
        "abs_max": ad[-1],
        "tokens": len(d),
    }


def main() -> None:
    """Probe a model, or compare records."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--score", help="a base record whose sequences to score")
    ap.add_argument("--env", action="append", default=[], help="K=V set before vLLM is imported")
    ap.add_argument("--arg", action="append", default=[], help="LLM kwarg as k=<json>")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--compare", nargs="+", help="BASE.json REC.json...: print the comparison")
    args = ap.parse_args()
    if args.compare:
        base = json.loads(Path(args.compare[0]).read_text())
        for path in args.compare[1:]:
            print(json.dumps({"record": path, **compare(base, json.loads(Path(path).read_text()))}))
        return
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    from qda_quant.output.vllm_env import disabled_kernels

    os.environ.update(disabled_kernels({}))  # Machete is off here as on every engine qda starts
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model pickles `census`
    llm_args = {k: json.loads(v) for k, v in (a.split("=", 1) for a in args.arg)}
    score = json.loads(Path(args.score).read_text())["seqs"] if args.score else None
    rec = probe(args.model, score, args.max_new, llm_args)
    rec["env"] = dict(kv.split("=", 1) for kv in args.env)
    Path(args.out).write_text(json.dumps(rec))
    print("KERNELS " + json.dumps(rec["kernels"]))


if __name__ == "__main__":
    main()
