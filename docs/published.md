# Use a published model

For anyone who wants a quantized model to serve now, with no build: pick a checkpoint, then run its card's serve line.

## Pick one

All are on [Hugging Face, qda-lab](https://huggingface.co/qda-lab). The repository name is
`qda-lab/QDA-<model>-<format>`, for example `qda-lab/QDA-Qwen3.5-9B-W4A16`.

| model | vLLM (compressed-tensors) | llama.cpp (GGUF) |
|---|---|---|
| Qwen3.8-27B | W4A16 · W4A8 · W3A16 · W4A4 | Q4 · Q6 |
| Qwen3.6-35B-A3B | W4A16 · W4A8 · W3A16 · W4A4 | Q4 |
| Qwen3.5-9B | W4A16 · W4A8 · W3A16 · W4A4 | |
| Qwen3.5-4B | W4A16 · W3A16 | |
| Qwen3-Coder-30B-A3B | W4A16 | |
| Gemma-4-31B | W4A16 | |
| Gemma-4-26B-A4B | W4A16 · W4A4 | |
| Nemotron-3-Nano-30B-A3B | W4A16 · W3A16 | |
| DeepSeek-R1-Distill-Llama-8B | W4A16 · W4A8 · W3A16 · W4A4 | |
| GLM-5.3-Flash | W4A4 | |

Two W4A16 variants have separate cards, [Qwen3.6-35B-A3B for SGLang](https://huggingface.co/qda-lab/QDA-Qwen3.6-35B-A3B-W4A16-SGLang)
and [Qwen3.8-27B searched with DeepSeek Harness](https://huggingface.co/qda-lab/QDA-Qwen3.8-27B-W4A16-dsh).

- Take W4A16 unless you need less memory: W3A16 (Qwen3.8-27B: 19.5 GB at W4A16, 14.5 GB at W3A16).
- W4A8 needs an H100 / H200; W4A4 needs a B200 / GB300 / RTX PRO 6000 ([which GPU serves which format](formats.md#does-my-gpu-run-it)).

## Serve with vLLM

Copy the serve line from the model card and run it unchanged. For `QDA-Qwen3.5-9B-W4A16`:

```bash
VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve qda-lab/QDA-Qwen3.5-9B-W4A16
```

The served name is the repository id. vLLM listens on port 8000; on a shared machine, add `--port N` with a free port
and send the request to `N`. From another terminal:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qda-lab/QDA-Qwen3.5-9B-W4A16", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 512}'
```

Add to the line when your case needs it:

- **A thinking model (Qwen3.5 and later).** Add `--reasoning-parser qwen3` to get the reasoning in a separate field
  instead of the reply text.
- **W3A16, and W4A16 on H100 / H200, without a CUDA toolkit.** Put the venv's NVIDIA libraries on the loader
  path first (the Humming kernel compiles through NVRTC at start):

  ```bash
  # Needs only vLLM in the venv, not QDA. Without it vLLM stops with `failed to open
  # libnvrtc-builtins.so.13.0`, also on a machine with only a CUDA 12 toolkit.
  lib="$(python -c 'import glob, sysconfig as s; p = s.get_paths(); print(":".join(sorted({d for k in ("purelib", "platlib") for d in glob.glob(p[k] + "/nvidia/*/lib")})))')" &&
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
  ```

- **A hybrid model (Qwen3.5 and later, Nemotron-H) that stops at start asking for a lower `--max-num-seqs`.** Pass
  the number in its message.
- **Nemotron-3-Nano.** Serve with `--tensor-parallel-size 1`: on two cards each expert splits to a width its
  64-wide groups do not divide.

## Serve with llama.cpp

```bash
llama-server -hf qda-lab/QDA-Qwen3.8-27B-Q4-GGUF
```

`-hf` downloads over HTTPS. A `llama-server` built from source without the OpenSSL headers cannot: it stops with
`HTTPS is not supported` and `failed to load model ''`. Install them (`libssl-dev`) or add
`-DLLAMA_BUILD_BORINGSSL=ON` to the `llama-server` `cmake` line ([pinned checkout](formats.md#gguf-for-llamacpp),
[build line](formats.md#measure-a-gguf)), and rebuild; an older llama.cpp that downloads through curl ignores the
flag. Downloads go to `LLAMA_CACHE` when set, else to the Hugging Face cache: `HF_HUB_CACHE`, else `$HF_HOME/hub`,
else `~/.cache/huggingface/hub`.

## Check it yourself

```bash
qda-kl kl qda-lab/QDA-DeepSeek-R1-Distill-Llama-8B-W4A16
```

- It reads the original model and its commit from the checkpoint's `build.json` and prints the checkpoint's KL.
- The first call downloads the original (its full size on disk) and builds the reference: 20 minutes for this
  8B model on an A100. Later calls reuse both.
- On a card another job uses, add `--gpu-memory-utilization 0.5`.
- On a Mamba hybrid (Nemotron, Qwen3.8), add `--serve-args '{"max_num_seqs": 128}'`.
- On a 96 GB card (RTX PRO 6000), add `--gpu-memory-utilization 0.95` for Gemma-4-31B and
  `--serve-args '{"max_num_batched_tokens": 8192}'` for Qwen3.6-35B-A3B W4A4; at the defaults vLLM runs out of memory at start.
- More options and checks: [validate a checkpoint](verify.md).
