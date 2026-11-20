# Recipe reference

When you write a recipe for `qda build`, look up its keys, steps, calibration sources and outputs here.

```yaml
model: Qwen/Qwen3-8B             # a Hub id (optionally @revision) or a local directory
format: w4a16-sym-g128           # what the checkpoint stores
out: qwen3-8b-w4a16              # the output directory: new, or empty
steps:                           # the methods, run in order
  - gptq: {clip: mse}
calib:                           # the calibration data
  file: {path: calib.jsonl, samples: 128, seqlen: 2048}
```

```bash
qda plan my.yaml                                   # resolve and check, no GPU work
qda build my.yaml --set out=other/ --set steps.0.gptq.clip=imse --set 'devices=[0,1]'
```

Write input paths relative to the recipe file; `--set` paths resolve against it too. `out` resolves against the
directory you run from. To repeat a build from anywhere, run
`qda build out/resolved.yaml --set out=again`: `resolved.yaml` records every path absolute.

## Keys

| key | default | meaning |
|---|---|---|
| `model` | required | Hub id or local path; `org/name@REVISION` pins a revision. A QDA checkpoint runs only checkpoint steps. |
| `format` | required, or `formats` | stored format: `w4a16`, `w4a8`, `w3a16`, `w4a4`, or spelled out, `w4a16-sym-g128` ([formats](formats.md)) |
| `target` | from the cards, else `sm90@0.33.0` | GPU and vLLM version that will serve it; `h100`, `l40s`, `h200` |
| `grid` | `absmax` | quantization grid of every step that names none ([below](#quantization-grids)) |
| `steps` | required | the methods, in order: `[{name: {settings}}, ...]` |
| `calib` | required | one calibration source ([below](#calibration-data)) |
| `out` | `./<recipe name>/` | output directory, new or empty |
| `devices` | all visible | card indices, or `["cpu"]` |
| `batch_size` | `1` | calibration rows per forward pass |
| `hessian` | `{dtype: bf16, buffer_rows: 0}` | how second-order statistics accumulate (`bf16`, `tf32`, `fp32`) |
| `residency` | chosen by the planner | `resident` keeps the model on the GPU; `stream` moves one layer at a time; `disk` reads each layer from disk |
| `dtype` | `bf16` | load dtype, `bf16` or `fp32` |
| `ignore` | `[gdn_gates, shared_experts]` | linears kept bf16: a role, an exact module name or a `re:` pattern |
| `formats` | none | in place of `format`, `format_overrides` and `ignore`: `{default: FORMAT, UNIT: FORMAT}` stores some units in another format ([mixed precision](quantize.md#mixed-precision-under-a-budget)) |
| `exporter` | compressed-tensors | `{gguf: {...}}` writes a llama.cpp file ([GGUF](formats.md#gguf-for-llamacpp)) |
| `drafts` | `[]` | speculative drafts to rotate into the build's basis ([below](#rotation-and-speculative-decoding)) |
| `keep_intermediate` | `false` | keep each stage's checkpoint of a multi-stage build in `out/.stages` |
| `nccl_nvls` | `false` | let card processes use NCCL's NVLS collectives |
| `meta` | `{}` | free-form notes, carried into `resolved.yaml` |

Use `devices: ["cpu"]` only on a machine with no visible card, or for a recipe whose steps need no card (`rtn`):
with a card visible, a recipe with `gptq`, `gptaq`, `awq`, `autoround`, `gsq`, `scale_kd`, `guided` or
`fisher_trace` on CPU is refused (`devices: cpu puts gptq on CPU`); remove `devices` or name a card.

## Methods

Write each method as a step: its name is the key under `steps`, its settings the keys under that. Steps run in
order.

| step | kind | what it does | main settings |
|---|---|---|---|
| `rotate` | model | QuaRot rotations: R1 on the residual stream, R2 on the value heads. | `residual`, `value_heads`, `seed`, `when` |
| `permute` | model | Reorders channels so that channels of similar size share a group. | `residual`, `intermediate` |
| `guided` | model | GuidedQuant token weights: tokens whose loss is most sensitive count more in the next step. | `clip`, `power`, `reproducible` |
| `fisher_trace` | model | Token weights from the trace of the original model's next-token Fisher matrix. | `window`, `floor` |
| `head` | model | Stores the output head (and optionally the embedding) in its own format. | `format`, `embed` |
| `gptq` | layer | Solves each layer's rounding against its input Hessian; q/k/v, gate/up and all experts share one pass. | `clip`, `actorder`, `damp`, `router_weighted` |
| `gptaq` | layer | GPTQ plus a correction for the error earlier quantized layers made. | as `gptq`, plus `alpha` |
| `awq` | layer | Searches per-channel scales from activation sizes. Only rescales: put `rtn` or `gptq` after it. | `n_grid`, `duo_scaling`, `objective` |
| `rtn` | layer | Round to nearest; needs no calibration. | `clip` |
| `autoround` | layer | Learns rounding offsets and clipping by sign-gradient descent. | `iters`, `lr`, `min_clip`, `target`, `reproducible` |
| `gsq` | layer | Learned rounding refined from an earlier step's codes. | `epochs`, `batches`, `target`, `reproducible` |
| `scale_kd` | checkpoint | Distils the scales (optionally norms and Gated DeltaNet parameters) from the original model. | `data`, `steps`, `lr`, `trainable`, `attn` |
| `soup` | checkpoint | Averages the scales of checkpoints that share their codes, in log space. | `with`, `weights` |

No published checkpoint uses `permute`, `gptaq` or `gsq`: measure KL against plain `gptq` before you ship one.
Every setting and its default is in the step's docstring (`qda_quant.steps`); `qda plan` refuses a setting the
model, format or kernel cannot take.

`clip` picks each group's scale for `gptq`, `gptaq` and `rtn`: `minmax` (default), `mse`, `imse`, `hmse` (weighted by
the Hessian) or `grid3`. At `w4a4`, no clip or a range search runs `fp4hmse` (`fp4mse` is its unweighted form;
[formats](formats.md)). On the `q4k` grid, `imatrix` is llama.cpp's importance-weighted fit.

`autoround`, `gsq` and `guided` take `reproducible: true`, and `scale_kd` takes `attn: eager`: each then writes the
same bytes on every rebuild, at a cost ([measured](reproduce.md#what-the-same-means)).

A W3A16 body with a 4-bit head, then scale distillation on traces of the original model:

```yaml
model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
format: w3a16-sym-g128
out: r1-llama-8b-w3a16
steps:
  - gptq: {clip: hmse}
  - head: {format: w4a16-sym-g64}
  - scale_kd: {data: data/kd-traces.jsonl, steps: 300, lr: 3.0e-4}
calib: {file: {path: data/calib.jsonl, samples: 442, seqlen: 2048}}
```

Trace rows are `{"prompt_ids": [...], "completion_ids": [...]}`. Without `data`, `scale_kd` trains on the calibration
samples. A checkpoint can be the `model` of a build, which then runs only checkpoint steps: `soup` averages several
`scale_kd` runs of one build (`soup: {with: [other-checkpoint]}`).

`scale_kd` on W4A8 multiplies on TF32 tensor cores in dense layers and with `experts: loop`; grouped experts stay
bf16. Its scales differ from the published W4A8 checkpoints' at KL
within 0.16%.

## Rotation and speculative decoding

After `rotate`, a speculative draft must be rotated into the same basis: it reads the model's hidden states.

| you want | do |
|---|---|
| the model's own MTP head | nothing: `rotate` rewrites a declared MTP head |
| an external draft, rotated build | `qda rotate-draft`, or list it under `drafts` |
| an external draft, unchanged | `rotate: {residual: false}` (R2 only) |

```bash
qda rotate-draft --checkpoint qwen3.5-9b-w4a16 --draft z-lab/Qwen3.5-9B-DFlash --out qwen3.5-9b-dflash-rot
```

```yaml
model: Qwen/Qwen3.5-9B
format: w4a16-sym-g128
out: qwen3.5-9b-w4a16
steps:
  - rotate: {residual: true}
  - gptq: {}
calib:
  file: {path: data/calib.jsonl, samples: 128, seqlen: 2048}
drafts: [z-lab/Qwen3.5-9B-DFlash]   # written to qwen3.5-9b-w4a16-Qwen3.5-9B-DFlash
```

Keep `dflash` or `dspark` in the draft's directory name: vLLM recognizes the draft type by its path. An unrotated
draft on a rotated build has every token rejected.

`rotate` refuses tied embeddings, post-norm blocks (OLMo-2, Gemma 2 and 3), and modules outside the text model that no
adapter declares ([add an architecture](add-architecture.md)). `rotate: {when: accepted}` skips the rotation on such a
model instead: `qda plan` prints one line.

## Quantization grids

| grid | rule at 4 bits | stored scales | served on |
|---|---|---|---|
| `absmax` | `absmax / 7.5`, or `(max - min) / 15` asymmetric (compressed-tensors) | bf16, positive | every format |
| `signed8` | `absmax / 8`, sign puts the larger side on code -8 (auto-round) | fp16, either sign | symmetric formats with bf16-scale kernels |

`autoround` defaults to `signed8` where it can serve, else `absmax`. GGUF grids (`q4k`, ...): [formats](formats.md#gguf-for-llamacpp).

## Calibration data

Every source cuts its rows into `samples` samples (default 128) of `seqlen` tokens (2048) with the model's tokenizer.
`seed` (0) shuffles the samples when non-zero. `chunks: true` cuts every row into consecutive pieces.
Give `{"input_ids"}` rows the fingerprint of the tokenizer that made them; QDA refuses a mismatch before GPU work.

| source | reads | main settings |
|---|---|---|
| `file` | JSONL of `{"text"}`, chat `{"messages"}` or `{"input_ids"}` rows | `path` |
| `hf` | a Hugging Face dataset at build time (`hf` extra) | `dataset`, `revision`, `subset`, `split`, `column`, `rows` |
| `mix` | a weighted mix of files or of sources | `parts`, `rows`, `heldout` |
| `slice` | named slices from a pool directory | `pool`, `names` |
| `selfgen` | the original model's own answers to a prompt file, generated with vLLM and cached | `prompts`, `n`, `max_tokens` |
| `vlm` | image + question + answer rows through the model's processor (`vlm` extra) | — |

Calibrate a base model on web text, and a chat or reasoning model on chat `messages` rows or its own answers
(`selfgen`).

```yaml
model: Qwen/Qwen3-8B
format: w4a16
out: qwen3-8b-w4a16-mix
steps: [{gptq: {}}]
calib:
  mix:
    parts:
      - {path: data/chat.jsonl, weight: 2}
      - {path: data/code.jsonl, weight: 1}
    rows: 4096
    heldout: data/kl-prompts.jsonl
    samples: 256
    seqlen: 4096
```

A pool is a directory of `<name>.jsonl` slices; `slice` reads them by name:

```yaml
model: Qwen/Qwen3.8-27B
format: w4a16
out: qwen3.8-27b-w4a16-mixed
steps: [{gptq: {}}]
calib:
  slice:
    pool: data/calib-pool
    names: {long-8k-16k-v2: 3, chat-clean: 1}
    samples: 128
    seqlen: 4096
```

`qda calib render` writes a calibration file in the model's own chat template, from the public pools
(`qda-lab/qda-calibration-pools`) or a pool directory:

```bash
qda calib render --model Qwen/Qwen3-8B --names chat=3,multilingual=2 --samples 128 --seqlen 2048 --out calib.jsonl
```

- `--samples` and `--seqlen` are what the build that reads the file asks for. `--rows N` keeps N rows instead.
- `--thinking keep` (default) writes reasoning in the model's thinking format; `off` renders thinking-off; `drop`
  removes it.

## What a build writes

`out/` appears only when the build completes. It holds a compressed-tensors checkpoint, the source's tokenizer,
chat template, processor configs and `generation_config.json` when the source has one (so it serves with the
source's sampling), and:

- `resolved.yaml`: the recipe with every default filled in.
- `build.json`: stages and seconds, peak memory, and a `serve` entry with the vLLM settings for the planned kernel.

A Hub original that ships no `generation_config.json` but whose released card sampling QDA lists (such as
`Qwen/Qwen3.5-4B`) gets one with that sampling; any other source without one serves with the engine's defaults.

Set `QDA_OUTPUT_ROOTS` (`:`-separated) to limit where outputs may be written; other locations:
[where things go](install.md#where-things-go). The Python API takes the same recipe:

```python
>>> from qda_quant.config import BuildConfig
>>> cfg = BuildConfig.from_dict(
...     {"model": "m", "format": "w4a16", "steps": [{"gptq": {}}], "calib": {"file": {"path": "c.jsonl"}}}
... )
>>> cfg.format, cfg.target
('w4a16-asym-g128', 'sm90@0.33.0')

```

`cfg.plan()` returns the plan; `cfg.build()` runs it.
