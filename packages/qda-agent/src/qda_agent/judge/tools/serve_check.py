"""The serving check of a hand-in: stock vLLM loads it and completes a few short prompts.

Run by the judge with the serving interpreter (`judge.serve_python`), from the read-only evaluator copy::

    python serve_check.py CKPT OUT [--max-tokens 64] [--max-model-len 4096] [--tp 1]

It applies the checkpoint's `build.json["serve"]` the way `qda_kl` does (`env` before vLLM is imported,
`args` as `vllm.LLM` keyword arguments). The judge runs it only on a frozen, sealed copy whose serve section
passed `qda_quant.output.serve_policy` when it was taken in (`judge.candidate.take`); this file cannot import that
policy. It writes `{"ok": bool, "reason": str}` to OUT. Every prompt must produce non-empty text that ends on
a stop or the token limit. Stdlib, vLLM and qda-quant's engine start, which the serving venv needs: it does not
import qda_agent, which the serving venv may not have.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROMPTS = (
    "The capital of France is",
    "Write one sentence about the sea.",
    "def fibonacci(n):",
    "2 + 2 =",
)


def serve_settings(ckpt: Path) -> dict:
    """`build.json["serve"]` of the checkpoint, or {}."""
    try:
        serve = json.loads((ckpt / "build.json").read_text()).get("serve")
    except (OSError, ValueError, AttributeError):
        return {}
    return serve if isinstance(serve, dict) else {}


def check(ckpt: Path, max_tokens: int, max_model_len: int, tp: int = 1) -> tuple[bool, str]:
    """Load `ckpt` in vLLM and generate; (ok, reason).

    The engine starts as every qda engine does: through `qda_quant.output.vllm_env.engine_start` (the compiled
    kernels in a node-local cache), with qda-kl's no-JIT settings (`NO_JIT_ENV`, `no_jit_args`: no FlashInfer
    kernel is compiled, so no nvcc is needed). A setting the checkpoint's serve section names keeps its value:
    the check starts the engine the hand-in is served with.
    """
    serve = serve_settings(ckpt)
    env = {str(k): str(v) for k, v in (serve.get("env") or {}).items()}
    os.environ.update(env)
    import vllm  # after the env: vLLM reads it at import

    try:
        from qda_quant.output.vllm_env import NO_JIT_ENV, engine_start, no_jit_args
    except ImportError as e:
        return False, f"judge.serve_python needs qda-quant: {e}"
    args = no_jit_args({"max_model_len": max_model_len, "enforce_eager": True, **(serve.get("args") or {})})
    args["tensor_parallel_size"] = tp
    llm, _ = engine_start({**NO_JIT_ENV, **env}, lambda: vllm.LLM(model=str(ckpt), **args))
    outs = llm.generate(list(PROMPTS), vllm.SamplingParams(max_tokens=max_tokens, temperature=0.0))
    if len(outs) != len(PROMPTS):
        return False, f"{len(outs)} answers to {len(PROMPTS)} prompts"
    for prompt, out in zip(PROMPTS, outs, strict=True):
        c = out.outputs[0] if out.outputs else None
        if c is None or not c.text.strip():
            return False, f"no text for prompt {prompt!r}"
        if c.finish_reason not in ("stop", "length"):
            return False, f"generation ended with {c.finish_reason!r} for prompt {prompt!r}"
    return True, ""


def main(argv: list[str]) -> int:
    """Write the verdict; exit 0 whenever OUT was written."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    a = p.parse_args(argv)
    if a.tp < 1:
        p.error("--tp must be positive")
    try:
        ok, reason = check(a.ckpt, a.max_tokens, a.max_model_len, a.tp)
    except Exception as e:  # noqa: BLE001  (any load or generation failure is a refusal, with its type)
        notes = "".join(
            f"\n{note}" for note in getattr(e, "__notes__", ())
        )  # the cause and fix engine_start knows
        ok, reason = False, f"vLLM could not serve it: {type(e).__name__}: {str(e)[:200]}{notes}"
    a.out.write_text(json.dumps({"ok": ok, "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
