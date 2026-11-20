# Quickstart

For a first run after [the install](install.md): build a 4-bit Qwen3-0.6B on one GPU, measure it, serve it.
Run every command from the repository root.

## 1. Build

```bash
qda build configs/quickstart.yaml
```

It downloads 1.7 GB (the model and the calibration set), takes 46 s on an H100 80GB and prints the lines below. A
rerun needs a new output directory (`--set out=DIR`, or delete `qwen3-0.6b-w4a16/` first) and takes 30 s with the
download cached.

```text
preflight: ...
step rotate: skipped (when: accepted): rotate: tied input/output embeddings; untie them first
contamination filter: not run (no reference index)
data lint (defects counted on rows with chat markup; plain and messages rows are not):
  ...
plan: 28 layers, ['layers'], weights stream, activations device, model 1.11 GiB, free 78.7 GiB on cuda:0, ...
layer 0 done ... layer 27 done
built /path/to/qwen3-0.6b-w4a16 in ... s
serve it with:
  ...
```

`contamination filter: not run (no reference index)` is expected here.

- The recipe, `configs/quickstart.yaml`, quantizes Qwen3-0.6B to W4A16 (4-bit weights, groups of 128) with GPTQ,
  calibrated on 125 samples of 2,048 tokens from a public chat set.
- `qwen3-0.6b-w4a16/` holds the checkpoint, `resolved.yaml` (the recipe with every default filled in) and
  `build.json` (the build record).
- For another model, copy `configs/quickstart.yaml` and change `model:` ([quantize your own model](quantize.md)).

## 2. Measure

```bash
qda-kl kl qwen3-0.6b-w4a16
```

```text
KL 0.104905  /path/to/qwen3-0.6b-w4a16  (original Qwen/Qwen3-0.6B@..., reference built now)
  chat           0.137786
  ...
  read on vLLM 0.33.0, tp 1, kernels: ...; ... tokens, top-1 flip rate 0.1070
```

- It reads the original model and its revision from the build's `resolved.yaml`.
- The first call builds the prompt set and the original's reference: 7 min on an H100 80GB, 8.7 min on an H200,
  12.6 min on an RTX PRO 6000. Later calls reuse both: 95 s.
- `kl` read 0.105 on an H100 80GB. 0 is identical to the original.
- The flip rate is the share of positions where the top token changed.
- Offline, add `--prompts packages/qda-kl/examples/prompts-small.jsonl` ([install](install.md#network)).
- More options and checks: [validate a checkpoint](verify.md#2-measure-kl).

## 3. Serve

Run the line `qda build` printed after `serve it with:`. This prints it again:

```bash
python -c 'from qda_quant.output.serve_command import serve_command_line; print(serve_command_line("qwen3-0.6b-w4a16"))'
```

- Run it as printed, in a bash or zsh shell with the venv activated: its environment prefix and flags bind the
  kernel the KL above read. On another machine, copy the checkpoint there and change the path after `vllm serve`.
- A line whose kernel compiles through NVRTC at start (Humming) starts with `lib="$(...)" && LD_LIBRARY_PATH="$lib..."`.
  Your shell runs it to put the NVIDIA libraries of the `python` in the same directory as `vllm` on the loader path.
  With no such `python`, it prints `no python next to vllm: activate the venv that has vLLM` and starts nothing.
- Stock vLLM serves the checkpoint (compressed-tensors); no plugin.
- The served name follows `--served-model-name`: `Qwen3-0.6B-W4A16` here, `Qwen3-0.6B-W4A4-NVFP4` for `format: w4a4`.
  Put that name in the request below.
- vLLM listens on port 8000 and takes 92% of the card (73 GB on an H100 80GB). On a shared card, add
  `--gpu-memory-utilization 0.3` and `--port N` with a free port, and send the request to `N`.
- It accepts requests about 150 s after the first start (an A100 or H100).

From another terminal:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-0.6B-W4A16", "prompt": "The capital of France is", "max_tokens": 8}'
```

## Next

- [Use a published model](published.md): skip the build.
- [Quantize your own model](quantize.md): another model, format or calibration set.
- [Agent search](search.md): let an agent find a better recipe.
