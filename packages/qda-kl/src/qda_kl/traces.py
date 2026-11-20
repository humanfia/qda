"""``qda traces``: a checkpoint's own answers and ``scale_kd`` traces for a prompt file, in one vLLM pass.

The recipes calibrate on a model's own answers and distil on traces the original model, or an earlier round's
checkpoint, wrote (docs/verify.md, "Own answers and KD traces"). For every prompt this writes one row to
each of:

- ``PREFIX-traces.jsonl``: ``{"prompt_ids", "completion_ids"}``, the verbatim token ids ``scale_kd`` trains on;
- ``PREFIX-answers.jsonl``: ``{"text"}``, prompt and completion decoded, the calibration text; with ``--join``
  one row holding every answer, for recipes that cut samples with ``chunks: true``;
- ``PREFIX-generation.json``: the checkpoint, prompt file and its sha256, rows, seed, sampling and the serve
  settings, so a rerun with the same arguments asks for the same draws.

Prompts go through the checkpoint's chat template with its default thinking mode. A packed checkpoint is served
with its ``build.json`` kernel settings (`qda_kl.serve.settings_for`), as a later-round writer must. The command
refuses any output file that already exists: a pipeline hashes its inputs, and a rewritten one is a new build.

    qda traces Qwen/Qwen3.5-9B data/prompts.jsonl data/original --max-tokens 8192 --seed 17
    python -m qda_kl.traces builds/round1 data/prompts.jsonl data/round1 \\
        --max-tokens 8192 --seed 18 --rows 0:710
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qda_quant.output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP

from qda_kl.lifetime import command

#: the README writer's sampling: a public alternative to each paper run's own
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


def _rows(spec: str | None, n: int) -> slice:
    if not spec:
        return slice(0, n)
    start, _, stop = spec.partition(":")
    return slice(int(start or 0), int(stop) if stop else n)


def write(
    model: str,
    prompts: Path,
    prefix: str,
    max_tokens: int,
    seed: int,
    rows: str | None = None,
    join: bool = False,
    tp: int = DEFAULT_TP,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> int:
    """Generate one completion per prompt row and write the traces, answers and generation record."""
    from qda_quant.exit import ENVIRONMENT
    from qda_quant.output.roots import output_path
    from qda_quant.step import Refused

    try:
        base = output_path(prefix)  # under QDA_OUTPUT_ROOTS when set
    except Refused as e:
        raise Refused(str(e), e.fix, code=e.code) from None
    prefix = str(base)
    outs = {k: Path(f"{prefix}-{k}") for k in ("traces.jsonl", "answers.jsonl", "generation.json")}
    if there := [str(p) for p in outs.values() if p.exists()]:
        raise Refused(
            f"{', '.join(there)} already exists",
            f"remove it, or pick another prefix than {prefix}",
        )
    try:  # the prompt file itself: the path the user gave, not a crash
        data = prompts.read_bytes()
        table = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise Refused(f"{prompts}: {error}") from None
    try:
        cut = _rows(rows, len(table))
    except ValueError as error:  # the slice as given
        raise Refused(f"--rows {rows!r}: {error}", "--rows START:STOP, for example 0:10") from None
    table = table[cut]
    if not table:
        raise Refused(
            f"rows {rows!r} of {prompts} select no prompt",
            "drop --rows (every row), or give a slice the file has (e.g. --rows 0:10)",
        )

    try:
        from transformers import AutoTokenizer
        from vllm import SamplingParams
    except ImportError as why:  # the engine and the tokenizer: the ``vllm`` extra
        raise Refused(
            f"traces needs vLLM and transformers ({why})",
            'uv pip install -c constraints.txt "./packages/qda-kl[vllm]"',
            code=ENVIRONMENT,
        ) from why

    from qda_kl.serve import llm as start_llm
    from qda_kl.serve import settings_for

    tok = AutoTokenizer.from_pretrained(model)
    ids = [
        tok(
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )["input_ids"]
        for r in table
    ]
    settings = settings_for(
        model,
        args={
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max(map(len, ids)) + max_tokens,
            "tensor_parallel_size": tp,
        },
    )
    llm = start_llm(settings, settings.args, refusals=False, model=model, seed=seed)
    sampling = {**SAMPLING, "max_tokens": max_tokens, "seed": seed}
    gens = llm.generate([{"prompt_token_ids": p} for p in ids], SamplingParams(**sampling))
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    texts = []
    with outs["traces.jsonl"].open("w") as f:
        for prompt, gen in zip(ids, gens, strict=True):
            completion = list(gen.outputs[0].token_ids)
            f.write(json.dumps({"prompt_ids": prompt, "completion_ids": completion}) + "\n")
            texts.append(tok.decode(prompt + completion))
    with outs["answers.jsonl"].open("w") as f:
        for text in ["".join(texts)] if join else texts:
            f.write(json.dumps({"text": text}) + "\n")
    record = {
        "model": model,
        "prompts": str(prompts),
        "prompts_sha256": hashlib.sha256(data).hexdigest(),
        "rows": [cut.start, cut.start + len(table)],
        "joined": join,
        "seed": seed,
        "sampling": sampling,
        "serve": settings.model_dump(mode="json"),
    }
    outs["generation.json"].write_text(json.dumps(record, indent=2) + "\n")
    print(f"qda: wrote {len(gens)} trace and answer rows to {prefix}-*", file=sys.stderr)
    return 0


@command("qda")
def _run(a: argparse.Namespace) -> int:
    from qda_quant.exit import refuse
    from qda_quant.step import Refused

    try:
        return write(
            a.model,
            Path(a.prompts),
            a.prefix,
            a.max_tokens,
            a.seed,
            a.rows,
            a.join,
            a.tp,
            a.gpu_memory_utilization,
        )
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``qda traces``."""
    p = sub.add_parser("traces", help="a checkpoint's own answers and scale_kd traces for a prompt file")
    _arguments(p)


def _arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "model", help="checkpoint directory or Hub id (packed checkpoints keep their kernel settings)"
    )
    p.add_argument("prompts", help='JSONL of {"prompt": ...} rows')
    p.add_argument(
        "prefix", help="output prefix: PREFIX-traces.jsonl, PREFIX-answers.jsonl, PREFIX-generation.json"
    )
    p.add_argument("--max-tokens", type=int, required=True, help="completion tokens per prompt")
    p.add_argument("--seed", type=int, required=True, help="sampling seed; change it for fresh draws")
    p.add_argument("--rows", help="START:STOP slice of the prompt rows (default: all)")
    p.add_argument("--join", action="store_true", help="write every answer into one text row")
    p.add_argument("--tp", type=int, default=DEFAULT_TP, help=f"tensor-parallel cards (default {DEFAULT_TP})")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"vLLM memory budget per card (default {DEFAULT_GPU_MEMORY_UTILIZATION})",
    )
    p.set_defaults(run=_run)


def main(argv: list[str] | None = None) -> int:
    """``python -m qda_kl.traces``: the same command without the ``qda`` entry point."""
    p = argparse.ArgumentParser(
        prog="python -m qda_kl.traces",
        description="a checkpoint's own answers and scale_kd traces for a prompt file",
    )
    _arguments(p)
    a = p.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
