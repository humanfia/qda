# qda-kl

Scores a quantized checkpoint by its KL divergence from the original model, inside vLLM, and checks its long
generations.

```sh
qda-kl ref --model ORIGINAL --prompts prompts.jsonl --out REF    # once per original model
qda-kl score run.yaml                                            # checkpoint, reference, out
qda-kl gencheck gencheck.yaml                                    # cut-off and looping answers
```

Documentation: [scoring](../../docs/verify.md), [evaluation](../../docs/verify.md), and
[adding a scorer](../../docs/add-scorer.md).

## Generation check

```yaml
checkpoint: path/to/checkpoint
original: Qwen/Qwen3-0.6B
prompts: builtin             # math, code, science, instruction following (src/qda_kl/data/README.md)
per_domain: 4
max_tokens: 8192
original_cache: path/to/original-cache
out: path/to/check
```

The full check is `--set max_tokens=32768 --set per_domain=16`. Sampling uses seed 17 and temperature 0.6; other
sampling settings come from the original model's generation config, for both models. `chat_template_kwargs` goes
to both chat templates.

| `out/gencheck.json` field, per domain | meaning |
|---|---|
| `cut` | answers that reached `max_tokens` |
| `loops` | answers ending in a periodic tail (2 % mismatches allowed; answers under 256 tokens are not tested) |
| `mean_tokens`, `delta` | mean answer length, and the checkpoint's minus the original's |

An answer can count as both `cut` and `loops`. These are behavior checks, not correctness scores. Texts and token
ids are in `out/texts/`, the config in `out/resolved.yaml`. The original's answers are cached and reused when the
model, prompts, length, seed, sampling, chat options and engine versions match; an incomplete cache is regenerated.
