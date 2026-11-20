# Validate a checkpoint

Check a checkpoint before you ship it: its files, its KL, a running server, benchmarks. The examples use the
quickstart's `qwen3-0.6b-w4a16`.

## 1. Inspect the files

```bash
qda inspect qwen3-0.6b-w4a16
```

`qda inspect` checks the checkpoint files and vLLM kernels for the visible GPU, or the recorded target when no
GPU is visible. See [example output and options](checkpoint-tools.md#inspect).

## 2. Measure KL

KL is the mean per-token divergence of the checkpoint's next-token distribution from the original model's.
0 is identical; lower is closer. It reads the checkpoint through the vLLM kernel it will be served with.

For a QDA checkpoint, one command. It finds the original model in the checkpoint's record or model card:

```bash
qda-kl kl qwen3-0.6b-w4a16
```

To control the prompts and the reference, use two steps. Record the original's answers once, then score any
number of checkpoints against them:

```bash
qda-kl ref --model Qwen/Qwen3-0.6B --prompts packages/qda-kl/examples/prompts-small.jsonl --out qwen3-0.6b-ref
qda-kl score my-score.yaml
```

```yaml
checkpoint: qwen3-0.6b-w4a16
reference: qwen3-0.6b-ref
out: qwen3-0.6b-score
```

`score` prints one JSON line and writes it to `qwen3-0.6b-score/score.json`.

- `ref` samples with the row the model's released cards record, else its own `generation_config.json`. A model in
  neither is refused with `no sampling settings`; pass `--sampling '{"temperature": 1.0, "top_p": 0.95}'`. A field
  `--sampling` names wins over the row. Offline (`HF_HUB_OFFLINE=1`), a Hub id's `generation_config.json` is read
  from the local Hugging Face cache; when the cache does not hold it, `ref` stops and prints the `hf download`
  command that fetches it.
- `--model` (and `qda-kl kl --original`) takes `ID@REVISION`, as `qda build` does, e.g.
  `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. The revision reaches the sampling read, the
  tokenizer, the engine and the manifest. An offline cache filled by a pinned `hf download` holds that revision and
  no `main`: name it.
- `qda-kl kl` reads the original at the commit the checkpoint records: `resolved.yaml`'s
  `provenance.source_revision`, else the commit `build.json`'s `source_repo` pins for the same repo (a published
  checkpoint ships only `build.json`); pass `--original ID@REVISION` to read another.
- `qda-kl ref --cache` builds a reference once into `$QDA_CACHE/kl-ref/` and links `--out` to it.
- Compare two scores only when they read the same reference at the same `tp`.
- `qda-kl ref`, `score`, `kl` and `gencheck` start vLLM with FlashInfer autotune off
  (`enable_flashinfer_autotune: false`). On one RTX PRO 6000, `qda-kl score` of `qda-lab/QDA-Qwen3.5-9B-W4A4` on 10
  prompts started its engine in 77 s instead of 379 s (301 s of autotune, repeated in full on a second start with
  the same caches), at the same KL (0.024546 both ways). `serve_args: {enable_flashinfer_autotune: true}` turns it
  on. The serve line a checkpoint prints keeps vLLM's default, autotune on.
- The `topk` scorer's settings (`gpu_memory_utilization`, `tp`, `serve_args`, ...) go under `scorer: {topk: {...}}`;
  the `TopKScorer` docstring in `packages/qda-kl/src/qda_kl/scorers.py` lists them.
- A real reading needs the model's own prompts: `qda-agent prepare` builds 96 search prompts and a held-out set.

### A GGUF file

Give `qda-kl score` the `.gguf` as `checkpoint`. It decodes the file to BF16 (`qda decode-gguf`), scores the decode
with vLLM against the original's reference, and deletes the decode.

```yaml
checkpoint: model-Q6_K.gguf
reference: path/to/reference
out: q6-score
snapshot: path/to/original-snapshot    # default: the model the reference was built from (a Hub id: its cached snapshot)
```

- The decoder reads `qwen2`, `qwen3`, `qwen3moe`, `llama`, `gemma3`, `qwen35` and `qwen35moe` files.
- A Qwen3.5 or Qwen3.8 decode holds the language model only, and `qda-kl score` serves it with
  `language_model_only`. Qwen3.8-27B Q6_K, default held-out set: KL 0.00093381.
- To keep the decode, run `qda decode-gguf model-Q6_K.gguf --snapshot path/to/original-snapshot --out q6-decoded`
  and score that directory.

To check long generations, give `qda-kl gencheck` the `.gguf` and the original's local directory:

```yaml
checkpoint: model-Q4_K_M.gguf
original: path/to/original-snapshot
out: q4-gencheck
```

```bash
qda-kl gencheck gencheck.yaml
```

Both models answer 4 prompts per domain, up to 8,192 tokens, on vLLM. `q4-gencheck/gencheck.json` counts the
answers cut off at the limit (`cut`) and stuck in a loop (`loops`) for each. Qwen3-0.6B Q4_K_M on one GH200: 211 s;
cut 7/16 for both, loops 1 vs 0.

This measures quantization error on the BF16 decode under vLLM, not `llama-server`'s kernels, chat template or
sampler. To check those before shipping a GGUF, run `python -m qda_kl.gguf.run`
([GGUF for llama.cpp](formats.md#measure-a-gguf)).

## 3. Serve it

```bash
lib="$(...)" && LD_LIBRARY_PATH="$lib..." VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve qwen3-0.6b-w4a16 --served-model-name Qwen3-0.6B-W4A16   # the line `qda build` prints
```

Run the line `qda build` printed as [the quickstart](quickstart.md#3-serve) says. A bare `vllm serve` needs the
checkpoint's `build.json` `serve` settings by hand: each `serve.args` key is a flag with dashes (`linear_backend: marlin`
is `--linear-backend marlin`), each `serve.env` entry an environment variable. `qda-kl` applies them itself. A hybrid
model needs [one more step](published.md#serve-with-vllm), and so does a bare `vllm serve` of a checkpoint whose
printed line starts with `lib=`.

A checkpoint built with the residual `rotate` step needs a rotated speculative draft:
[rotation and speculative decoding](recipe.md#rotation-and-speculative-decoding).

## 4. Benchmarks

If you only ran `qda build`, give `qda eval` the checkpoint and a preset: `kl`, `sanity` (chat template, cut-offs and
loops), `full` or `specdec` (MTP draft acceptance). It
reads the original from the checkpoint's `build.json`, builds or reuses its reference, and writes the results to
`qwen3-0.6b-w4a16-eval/sanity` (`--set out=DIR` moves them):

```bash
qda eval ./qwen3-0.6b-w4a16 --suite sanity    # more checkpoints of the same original may follow
```

`kl` reads the KL v2 corpus's holdout split by default, the prompts `qda-kl kl` reads (built into the corpus
cache on first use); score another prompt file with `--set suites.0.kl.corpus.file.path=prompts.jsonl`.

To compare checkpoints from different sources, `qda eval` takes a config naming one reference and prints a table:

```yaml
reference: Qwen/Qwen3-0.6B
checkpoints:
  bf16: Qwen/Qwen3-0.6B
  w4a16: ./qwen3-0.6b-w4a16      # a local directory: start it with ./ or /
out: runs/eval-qwen3-0.6b
suites:
  - kl:
      corpus: {file: {path: packages/qda-kl/examples/prompts-small.jsonl}}
```

```bash
qda eval eval.yaml --dry-run      # the jobs, and which are cached
qda eval eval.yaml
qda eval table runs/eval-qwen3-0.6b --baseline bf16
```

`bf16` is the original scored against itself, so its row is 0. Brackets are each domain's confidence interval.
Finished jobs are cached. Other suites (`lm_eval`, `full`, ...): `qda eval --help`. `lm_eval` needs
`uv pip install -c constraints.txt "./packages/qda-kl[eval]"`.

After a search, `qda eval <run> --suite full` runs the original model's released card settings (five
`lm_eval` tasks and LiveCodeBench v6) on the winner and candidate 0. LiveCodeBench runs each generated program in a
bubblewrap sandbox: `--lcb-sandbox` defaults to `auto`, and only an explicit `--lcb-sandbox none` runs them on the
host.
