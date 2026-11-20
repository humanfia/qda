# Checkpoint tools

Inspect any checkpoint and export a checkpoint for release. They run on the CPU, except `qda inspect --serve`
(one card).

| command | what it does |
|---|---|
| `qda inspect PATH\|HF_ID` | format, groups, kept-bf16 modules, layout and missing files, from headers only |
| `qda export` | copy a finished checkpoint without the run's records, with sha256 sums |

## Inspect

Run it before downloading or serving a checkpoint: it says in seconds whether vLLM can serve the checkpoint on
this card, and with which kernel, without downloading weights. With no card visible, it checks the target the
checkpoint's `build.json` records; `--target` names another.

```sh
qda inspect qda-lab/QDA-Qwen3.5-9B-W4A4
```

```text
qda-lab/QDA-Qwen3.5-9B-W4A4: nvfp4 (layout qda)
  group group_0: float4 on Linear, actorder static
  kept bf16: 337 modules, e.g. language_model.model.layers.0.linear_attn.in_proj_a, ...
  experts: no; g_idx: no
  note: the checkpoint was built for sm100@0.33.0 (build.json); this card is sm103
  serve name: QDA-Qwen3.5-9B-W4A4
  servable as is on sm103@0.33.0: group_0/dense FlashInferCuteDslNvFp4LinearKernel
```

It reads the checkpoint's JSON files and safetensors headers, not its weights. It says whether vLLM's kernels can
serve the checkpoint, not whether its weights are accurate; `--serve` starts the engine once and names every tensor
vLLM never read.
Every option is in the [`qda inspect` reference](cli.md#qda-inspect) and every exit code in [exit
status](troubleshooting.md#exit-status).

## Export for release

```sh
qda export builds/qwen3.5-9b-w4a16 release/qwen3.5-9b-w4a16 > release.sha256
```

`qda export` drops `build.json`, `resolved.yaml`, `recipe.yaml`, `rotate-draft.json`, `splice.json` and subdirectories, reads every
copy back, and refuses a text file that names the checkpoint's local path. `--link` hard-links instead of copying.
