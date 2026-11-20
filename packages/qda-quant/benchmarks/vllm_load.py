"""Load a qda checkpoint in stock vLLM with its ``build.json["serve"]`` settings, generate, and check kernels.

    python vllm_load.py CKPT [--prompt TEXT | --ids 1,2,3] [--max-tokens 16]

Run it with the serving venv's python (vLLM 0.27.1 or 0.33.0). It applies ``serve.env`` before vLLM is
imported and ``serve.args`` as engine arguments, generates greedily, then reads the kernels vLLM bound from
the model itself (``qda_kl.vllm_hooks.bound_kernels`` through ``LLM.apply_model``, merged over the engine's
ranks) and prints one JSON line: ``{"ok": ..., "expected": {...}, "bound": {...}, "text": ...}``. Exit code 1
if a bound kernel differs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# The serving venv may hold vLLM alone; read this checkout's packages rather than an installed copy.
sys.path[:0] = [str(Path(__file__).parents[2] / p) for p in ("qda-quant/src", "qda-kl/src")]

from qda_kl.serve import check_kernels, kernel_set  # noqa: E402
from qda_kl.vllm_hooks import bound_kernels, merge_kernels  # noqa: E402
from qda_quant.output.vllm_env import disabled_kernels, loader_env  # noqa: E402


def _child(a: argparse.Namespace, serve: dict) -> None:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model pickles `bound_kernels`
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(serve.get("args", {}))
    llm = LLM(model=a.ckpt, enforce_eager=True, gpu_memory_utilization=0.5, max_model_len=a.max_model_len,
              skip_tokenizer_init=a.ids is not None, **kw)  # fmt: skip
    kernels = merge_kernels(llm.apply_model(bound_kernels))  # one result per worker; stages split the layers
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, detokenize=a.ids is None,
                        prompt_logprobs=0 if a.prompt_logprobs else None)  # fmt: skip
    ids = [int(x) for x in a.ids.split(",")] if a.ids else None
    prompt = TokensPrompt(prompt_token_ids=ids) if ids else a.prompt
    res = llm.generate([prompt], sp)[0]
    out = res.outputs[0]
    lp = None
    if a.prompt_logprobs:  # log p(token_t | tokens_<t) of each prompt token after the first
        lp = [d[t].logprob for d, t in zip(res.prompt_logprobs[1:], res.prompt_token_ids[1:], strict=True)]
    got = {"text": out.text, "ids": list(out.token_ids), "prompt_logprobs": lp, "kernels": kernels}
    print("QDA_RESULT " + json.dumps(got), flush=True)


def main() -> int:
    """Run the load in a child process with the serve env, then check the kernels it bound."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--ids", default=None)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--prompt-logprobs", action="store_true")
    p.add_argument("--child", action="store_true")
    a = p.parse_args()
    serve = json.loads((Path(a.ckpt) / "build.json").read_text())["serve"]
    if a.child:
        _child(a, serve)
        return 0
    # the venv's nvidia/*/lib on the loader path: Humming's NVRTC child process opens libnvrtc-builtins there
    env = loader_env({**os.environ, **disabled_kernels(serve.get("env", {}))})
    r = subprocess.run(
        [sys.executable, __file__, *sys.argv[1:], "--child"], env=env, capture_output=True, text=True
    )
    log = r.stdout + r.stderr
    res = next((json.loads(line[11:]) for line in log.splitlines() if line.startswith("QDA_RESULT ")), None)
    expected = {k: sorted(kernel_set(v)) for k, v in serve["kernels"].items()}
    bound = (res or {}).get("kernels") or {}
    why = ""
    try:
        check_kernels(expected, bound)
    except ValueError as e:  # `Refused`: the engine bound other kernels than the checkpoint names
        why = str(e)
    ok = r.returncode == 0 and res is not None and not why
    print(json.dumps({"ok": ok, "vllm": serve["vllm"], "env": serve.get("env"), "args": serve.get("args"),
                      "expected": expected, "bound": bound, "result": res}))  # fmt: skip
    if not ok:
        sys.stderr.write((why + "\n" if why else "") + log[-6000:])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
