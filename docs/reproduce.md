# Reproduce

For researchers who want to rebuild a published checkpoint from its recipe, or to repeat the build-speed
comparison with llm-compressor.

Run these commands from the repository root after [installing QDA](install.md).

## Rebuild a published model

Each published checkpoint's recipe and inputs are in the
[`qda-recipe-inputs`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs) dataset, in a folder named after the
checkpoint. For `QDA-Qwen3.5-9B-W4A16`:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset \
  --include 'QDA-Qwen3.5-9B-W4A16/*' --local-dir recipe-inputs    # 606 MB
(
  cd recipe-inputs/QDA-Qwen3.5-9B-W4A16 &&
  sha256sum -c SHA256SUMS &&
  qda plan qwen3.5-9b-w4a16.yaml --set out=rebuild &&
  qda build qwen3.5-9b-w4a16.yaml --set out=rebuild
)
```

- Memory: the streamed plan (`residency: stream`) peaks at 29.4 GiB, 37.6 GiB in the three `scale_kd` stages; it
  fits one L40S. On a bigger card `qda plan` picks the resident plan: 47.0 GiB on an H200.
- The published checkpoint was built on one H100 80 GB. An interrupted build writes no checkpoint.
- More cards: `--set 'devices=[0,1]'`. Steps that run on one card (`guided`, `fisher_trace`, `gptaq`) use the first
  visible card when `devices` is unset.

Compare the rebuild with the published checkpoint by KL, not bytes ([validate](verify.md#2-measure-kl)): the published
checkpoint uses the `legacy` layout (`qda inspect <dir>` names it).

## What "the same" means

- **Bit-identical.** Steps without gradients (`gptq`, `rtn`, `awq`, `rotate`, ...) on the same GPU type, software
  stack and card count. `qda build OUT/resolved.yaml --set out=NEW` writes the same bytes. Another card count
  changes the Hessian sum order: bytes differ, KL moves by noise.
- **Within a noise band.** The gradient steps (`scale_kd`, `gsq`, `autoround`, `guided`) at their defaults, because
  cuDNN's attention backward is not deterministic, and KL scores of quantized MoE checkpoints. Quote their KL as a
  3-seed mean and band.

Each gradient step has a setting that makes it bit-identical. Its cost, measured on Qwen3-8B, seed 0, one H200,
128 × 2,048 calibration, whole builds:

| step and setting | KL per run | distinct checkpoints | build | peak card memory |
|---|---:|---:|---:|---:|
| `guided` + `gptq` | 0.073690, 0.073756, 0.073487 | 3 | 84 to 85 s | 29.13 GiB |
| `guided` `reproducible: true` + `gptq` | 0.073229 (4 runs) | 1 | 115 to 118 s | 49.09 GiB |
| `gptq` + `gsq` | 0.504405, 0.508952, 0.508952 | 2 | 367 to 369 s | 33.69 GiB |
| `gptq` + `gsq` `reproducible: true` | 0.513404 (4 runs) | 1 | 404 s | 33.69 GiB |
| `scale_kd`, 40 steps | 0.068805, 0.069039, 0.068771 | 3 | 1155 to 1159 s | 36.76 GiB |
| `scale_kd`, 40 steps, `attn: eager` | 0.069155 (3 runs) | 1 | 1430 to 1439 s | 49.09 GiB |

The cost is in the build and memory columns. The reproducible setting moves KL by at most 1.5%, not a quality difference.

## Build speed against llm-compressor

Setup: H100 80 GB; both libraries build from the same 128 × 2,048-token calibration file with GPTQ
`w4a16-sym-g128` (actorder static, damp 0.01) and AWQ + RTN `w4a16-asym-g128`. Time is model load to saved
checkpoint. Ratio is llm-compressor 0.15's time over QDA's. Peak is the highest per-card memory.

| model | method | cards | QDA | llm-compressor | ratio | QDA peak GiB | llm-compressor peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | GPTQ | 1 | 58.7 s | 156.8 s | 2.67x | 23.2 | 20.7 |
| Qwen3-8B | GPTQ | 2 | 53.4 s | 157.7 s | 2.95x | 25.0 | 14.3 |
| Qwen3-8B | GPTQ | 4 | 48.7 s | 158.2 s | 3.25x | 25.6 | 10.9 |
| Qwen3-8B | AWQ + RTN | 1 | 216.0 s¹ | 267.0 s | 1.24x | 44.5 | 31.6 |
| Qwen3-8B | AWQ + RTN | 2 | 133.0 s¹ | 265.8 s | 2.00x | 34.1 | 25.5 |
| Qwen3-8B | AWQ + RTN | 4 | 91.2 s¹ | 271.7 s | 2.98x | 27.9 | 21.9 |
| Qwen3.8-27B | GPTQ | 1 | 190.8 s | 501.3 s | 2.63x | 63.5 | 61.5 |
| Qwen3.8-27B | GPTQ | 2 | 155.4 s | 500.7 s | 3.22x | 66.9 | 39.4 |
| Qwen3.8-27B | AWQ + RTN | 1 | 653.6 s¹ (stream) | 903.3 s | 1.38x | 49.1 | 76.9 |
| Qwen3.8-27B | AWQ + RTN | 2 | 387.7 s¹ (stream) | 907.3 s | 2.34x | 31.4 | 51.8 |
| Qwen3.8-27B | AWQ + RTN | 4 | 241.1 s¹ | 915.9 s | 3.80x | 67.8 | 39.6 |
| Qwen3.6-35B-A3B | GPTQ | 1 | 181.2 s (stream) | OOM | — | 39.1 | — |
| Qwen3.6-35B-A3B | GPTQ | 2 | 156.0 s (stream) | 1,668.2 s | 10.69x | 35.3 | 78.6 |
| Qwen3.6-35B-A3B | AWQ + RTN | 1 | 456.9 s (stream) | OOM | — | 42.2 | — |
| Qwen3.6-35B-A3B | AWQ + RTN | 2 | 393.8 s (stream) | 10,889.4 s | 27.65x | 62.6 | 78.2 |
| Qwen3.6-35B-A3B | AWQ + RTN | 4 | 376.6 s (stream) | 10,870.8 s | 28.87x | 57.4 | 47.7 |

- OOM: out of memory on the cards it had.
- (stream): QDA streams the weights from host memory (`residency: stream`).
- Software: torch 2.14.1+cu130, transformers 5.21.0, compressed-tensors 0.20.1, flash-linear-attention 0.6.0.
- Raw rows: `docs/assets/speed_table.jsonl`.

¹ QDA on the `constraints.txt` stack: torch 2.14.0+cu130, transformers 5.20.2, compressed-tensors 0.17.0.

### Why QDA builds faster

The GPTQ algorithm is the same.

| | llm-compressor 0.15 | QDA |
|---|---|---|
| Hessian arithmetic | bf16 activations upcast, fp32 GEMM on CUDA cores | bf16 inputs on tensor cores, fp32 accumulation |
| shared inputs | one Hessian per linear | one per input: q/k/v share one, gate/up share one |
| activations and weights | moved through pageable CPU memory each layer | kept on the GPU when they fit |
| MoE experts | a Python loop over experts, a host sync each | one fused pass per expert bank |

The MoE row is why the gap is largest on Qwen3.6-35B-A3B.

### Repeat it

Build the GPTQ comparison checkpoint with both libraries, then score both against one reference. llm-compressor
needs its own environment:

```bash
qda build configs/compare-gptq.yaml
UV_PROJECT_ENVIRONMENT=.venv-llmc uv sync --no-default-groups --all-packages --extra llmc
UV_PROJECT_ENVIRONMENT=.venv-llmc uv run --no-sync python bench/llmc_quickstart.py Qwen/Qwen3-0.6B qwen3-0.6b-llmc
qda-kl ref --model Qwen/Qwen3-0.6B --prompts packages/qda-kl/examples/prompts-small.jsonl --out qwen3-0.6b-ref
qda-kl score packages/qda-kl/examples/quickstart-score.yaml --set checkpoint=qwen3-0.6b-gptq-c4 --set out=qwen3-0.6b-gptq-c4-score
qda-kl score packages/qda-kl/examples/quickstart-score.yaml --set checkpoint=qwen3-0.6b-llmc --set out=qwen3-0.6b-llmc-score
```

- `bench/llmc_quickstart.py` runs llm-compressor's `GPTQModifier` with its `W4A16` scheme (QDA's `w4a16-sym-g128`)
  on the token ids `configs/compare-gptq.yaml` names.
- One build is one sample: compare at least 3 builds per library. For each seed `N`, add `seed: N` under
  `calib.hf` in `configs/compare-gptq.yaml` (both libraries read it), then rerun the commands above except
  `qda-kl ref`, with new directories: `qda build ... --set out=qwen3-0.6b-gptq-c4-sN`,
  `llmc_quickstart.py ... qwen3-0.6b-llmc-sN`, and the matching `--set checkpoint=` / `--set out=` on each
  `qda-kl score`.

Time one cell of the tables (`--method gptq`, `awq-rtn` or `rtn`; `--cards N`); each run appends one line to
`cells.jsonl`:

```bash
qda calib render --model Qwen/Qwen3-8B --names chat=3,multilingual=2 --samples 128 --seqlen 2048 --out calib.jsonl
python bench/bench_cell.py --library qda --model Qwen/Qwen3-8B --method gptq --cards 1 --calib calib.jsonl --out qwen3-8b-qda
UV_PROJECT_ENVIRONMENT=.venv-llmc uv run --no-sync python bench/bench_cell.py --library llmc --model Qwen/Qwen3-8B --method gptq --cards 1 --calib calib.jsonl --out qwen3-8b-llmc
```
