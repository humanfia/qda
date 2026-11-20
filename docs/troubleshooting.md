# Troubleshooting

For anyone whose `qda`, `qda-kl` or `qda-agent` command stopped. Search this page for the start of the first
error line, then run the fix. A QDA refusal prints two lines, `error:` and `fix:`; run the `fix:` line first. Not
listed: run `qda doctor --report` and [open an issue](https://github.com/humanfia/qda/issues) with its output
and the full error.

## Exit status

| status | meaning |
|---|---|
| 0 | success |
| 1 | a bug in QDA, or a crash in vLLM, a compiler or another helper: report it |
| 2 | the input: a config, an argument, a path or a checkpoint: change the command |
| 3 | the machine: card, memory, disk, package or sign-in: fix the machine, then rerun unchanged |
| 130 | interrupted: Ctrl-C, or SIGTERM / SIGHUP in a search or a command that owns an engine |

A search that has started and stops on status 3 writes `FINAL.json` with its best checkpoint first.

## Install and `qda doctor`

| message (start) | fix |
|---|---|
| `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` | Driver below 580. Upgrade it, or load the [CUDA compat libraries](install.md#an-older-driver-cuda-12). Do not install another torch ([why](install.md#an-older-driver-cuda-12)). |
| `MISS CUDA driver: the driver supports CUDA 12.8, but torch ... is built for CUDA 13.0` | Same as the row above. |
| `MISS GPU: nvidia-smi: ...` | No NVIDIA driver here. Run on a GPU node; `qda plan` runs on a CPU. |
| `MISS GPU: CUDA_VISIBLE_DEVICES='': it is empty, so CUDA sees no card` | `unset CUDA_VISIBLE_DEVICES`, or set it to the cards you hold. |
| `MISS card health: card 4: 2 uncorrectable ECC error(s) since the driver loaded` | Set `CUDA_VISIBLE_DEVICES` (a search: `--set 'cards=[...]'`) to the other cards, and have the card reset. |
| `skip vLLM: not installed, so serving is out of scope here` | `uv pip install -c constraints.txt "./packages/qda-kl[vllm]"` |
| `vLLM: 0.31.0 (the kernels are checked on 0.33.0 ...)` | `uv pip install -c constraints.txt vllm` |
| `C compiler: not found: Triton's first use builds its CUDA driver with it` | Install `gcc`, or `export CC=<your C compiler>`. |
| `C compiler: gcc cannot build Triton's CUDA driver: ... Python.h: No such file or directory` | `uv venv -p 3.12 --managed-python`, then reinstall; or, as root, `sudo apt install python3.12-dev` (RHEL: `sudo dnf install python3.12-devel`). |
| `C compiler: c++ not found: FlashInfer compiles its kernels with it when an engine starts` | `export CXX=/usr/bin/g++-13` (the `g++-<n>` you have), or `vllm download-kernels` once. |
| `error: GDN layers would fall back to transformers' torch reference kernels ...` | `uv pip install -c constraints.txt flash-linear-attention` |
| `MISS sandbox: bwrap: not found` | `sudo apt install bubblewrap` (RHEL: `sudo dnf install bubblewrap`), or `--set isolation.level=none`. |
| `MISS engine cache: <path>: [Errno 5] Input/output error` | Move the four caches to a node-local disk: [where things go](install.md#where-things-go). |
| `MISS engine cache: /tmp/qda_<user>: belongs to another user ...` or `... writable by group or others ...` | Move the four caches to a directory of your own: [where things go](install.md#where-things-go). |
| `python3: No module named ensurepip`, or Python below 3.12 | `uv venv -p 3.12 --managed-python` |

## `qda plan` and `qda build`

| message (start) | fix |
|---|---|
| `steps.0.gptq.dampx: unknown key` | Remove or rename the key in the YAML. |
| `format: unknown format 'w4a8x'; expected e.g. 'w4a16-asym-g128'` | `--set format=w4a16-asym-g128` |
| `output <path> exists and is not an empty directory; a build never replaces what is there` | `--set out=NEW_DIR` |
| `output <path> is outside QDA_OUTPUT_ROOTS=...` | `--set out=$QDA_OUTPUT_ROOTS/<name>`, or `unset QDA_OUTPUT_ROOTS`. |
| `no supported target serves compute capability <N> yet` | `--set target=h100`, or [pick a card that serves your format](formats.md#does-my-gpu-run-it). |
| `FP8 activations need Hopper: vLLM's CUTLASS W4A8 kernel needs compute capability 90` | Build `w4a16`, or `--set target=h100` for a checkpoint served on H100 / H200 ([GPU table](formats.md#does-my-gpu-run-it)). |
| `stage <step>: <peak>, above the <X> GiB free on <device>; largest term` | `--set residency=stream`, or more cards: `--set 'devices=[0,1]'`. |
| `rotate: tied input/output embeddings; untie them first` | `- rotate: {when: accepted}`, or remove `rotate`. |
| `rotate: post-norm blocks (...)` | `- rotate: {when: accepted}`, or remove `rotate`. |
| `model '<id>': the hub repo is gated` | Accept the licence on the model page, then `export HF_TOKEN=<token>` (or `hf auth login`). |
| `model <id>: cannot read its transformers config` | [Add an architecture](add-architecture.md). |
| `model type '<type>' (<Class>): installed transformers has no ...` | [Add an architecture](add-architecture.md). |
| `qda: refused: model <id>: probing its forward failed in <module> (...)` | A new layout: [add an architecture](add-architecture.md). Otherwise rerun with `QDA_TRACEBACK=1` and [report](https://github.com/humanfia/qda/issues) the traceback. |
| `qda: refused: model <id>: reading its tokenizer failed (...)` | Check the id and the network; gated: `export HF_TOKEN=<token>`. |
| `qda: refused: model <id>: tokenizing its calibration rows failed (...)` | The tokenizer has no chat template: calibrate on `{"text": ...}` rows. |
| `calib hf <dataset>@<revision>: cannot read the dataset: ConnectionError: Couldn't reach ...` | No internet on this node: download first, then build offline ([network](install.md#network)). |
| `model '<id>': not a local directory, and not a hub repo: ...` | Name `org/name` or a local directory; the message quotes the Hub's reason. |
| `calibration <label>: no rows` | Write JSONL rows (`{"messages": ...}` or `{"text": ...}`) at the path `calib:` names. |
| `<d> is not visible: the process sees <N> card(s)` | `--set 'devices=[0,1]'` with cards from `nvidia-smi -L`. |

## KL and serving

| message (start) | fix |
|---|---|
| `ERROR ... Error parsing config for <out>.tokenizer: Unrecognized model` (twice, then `retrying`) | None: vLLM probes the tokenizer copy; the run continues. |
| `RuntimeError: Could not find nvcc ...` or `nvcc: not found` | `vllm download-kernels` once in this venv ([disk](install.md#requirements), needs internet), or `export CUDA_HOME=<toolkit>` and `uv pip install ninja`. |
| `torch.OutOfMemoryError: CUDA out of memory` while an engine starts, then `fix:   halve vLLM's batch ...` | `max_num_batched_tokens: 8192` where the `fix:` line says: FlashInfer's MoE autotune runs vLLM's 16384-token batch after the KV cache is allocated (W4A4 MoE on a 96 GB card). |
| `c++: not found` or `Failed to find C compiler` | `export CXX=/usr/bin/g++-13` / `export CC=/usr/bin/gcc` (the compilers you have), or install `g++`. |
| `nvrtc: error: failed to open libnvrtc-builtins.so.13.0` | The line `qda build` prints sets the path itself; before a model card's line or a bare `vllm serve`, run the export in [serve with vLLM](published.md#serve-with-vllm). |
| `qda-kl: refused: w4a8-sym-g128 needs vLLM's CUTLASS W4A8 kernel, which serves compute capability 9.0 only` | Score on H100 / H200, or build `w4a16`. |
| `<id>: <N> prompt tokens leave no answer room in the <ctx>-token window` | Shorten the prompt, or use a model with a longer context. |
| `the checkpoint names vLLM <v>, this is vLLM <v>` | `uv pip install -c constraints.txt vllm` |
| `serve dtype <dtype> differs from reference anchor dtype <dtype>` | `--set scorer.topk.allow_dtype_mismatch=true` |
| `the engine did not bind the checkpoint's kernels:` | Pass `build.json`'s `serve.args` and `serve.env` to `vllm serve`. |
| `tokenizer fingerprint mismatch in <part>` | Rebuild the reference from the checkpoint's own model revision. |
| `<checkpoint>: vocabulary <N> < the reference's <M>` | [Build a reference](verify.md#2-measure-kl) from this checkpoint's own original, then score against it. |
| `no checkpoint at <path> (did the build fail?...)` | Build it: `qda build CONFIG.yaml`, or fix the path. |
| `scorer.<key>: unknown scorer; known: [...]` | `scorer: {topk: {}}` |
| `<path>: no such prompt file` | `--prompts <existing .jsonl>` |
| `<id>: give exactly one of prompt / text` | Write `{"id": ..., "prompt": ...}` or `{"id": ..., "text": ...}` per row. |
| `<out> already holds a reference; qda-kl ref never overwrites one` | Score against it with `qda-kl score` (`reference: <out>`), or pass another `--out`. |
| `<path>: not a qda-kl reference (no manifest.json in it)` | Point `reference:` at a `qda-kl ref --out` directory. |
| `<out>: another build holds the lock (pid ... on ... since ...)` | Wait for that `qda-kl ref` to exit, or use another `--out`. |
| `qda inspect: <path>: not a checkpoint directory, a .gguf file or an HF repo id` | `qda inspect <the build's out directory>` or `qda inspect org/name`. |
| `OSError: [Errno 98] Address already in use` | `vllm serve ... --port <free port>` |

## Search (`qda-agent`)

Sign-in errors (`agent sign-in: ...`): see [sign-in](search-config.md#sign-in).

| message (start) | fix |
|---|---|
| `MISS agent CLI: claude is not on PATH` | Install Claude Code, or `qda-agent prepare ... --backend dsh`. |
| `qda-agent search: qda doctor reported a miss above` | Fix the `MISS` line above it. |
| `contamination-index: this search filters every build's calibration against the ...` | `--set judge.kl_holdout_ref=<holdout.jsonl>` |
| `contamination-index: judge.kl_ref and judge.kl_holdout_ref name the same prompt file` | `--set judge.kl_holdout_ref=<a separate holdout.jsonl>` |
| `sandbox bwrap needs bubblewrap (bwrap) on PATH; or use docker` | Install `bubblewrap`, or `--set isolation.level=none`. |
| `livecodebench: bubblewrap (bwrap) is not installed, so the generated code cannot be sandboxed` | Install `bubblewrap`, or run unsandboxed: `qda eval RUN_DIR --suite full --lcb-sandbox none`. |
| `baselines[0] <path>: <x> bits per quantized weight, over the judge's budget of <y> bpw` | Raise `--budget-bpw`, or `--budget-bpw candidate0`. |
| `cards [0, 1, 2, 3] are not in CUDA_VISIBLE_DEVICES=` | `--cards 0` (physical numbers you hold), or `unset CUDA_VISIBLE_DEVICES`. |
| `card(s) [2, 3]: the driver lists 2 GPU(s)` | `--set 'cards=[0,1]'` with the cards `nvidia-smi -L` lists. |
| `card order: cards [<n>]: nvidia-smi index != PCI-bus-order ordinal` | `--set 'cards=[<nvidia-smi indices in PCI-bus order>]'` |
| ``model: `<cmd>` exited 1 (its output is above)`` | Gated: `export HF_TOKEN=<token>`; wrong id: correct it; then rerun `qda-agent prepare`. |
| ``<step>: `<cmd>` exited <code> (its output is above)`` | Fix what the output names, rerun `qda-agent prepare`; it skips finished steps. |
| `<VAR> (<path>) is full; fix: export <VAR>=<a directory on a disk with more room>` | `export <VAR>=<dir on a disk with room>` (for `--out (<path>) is full`: `--out <dir on a larger disk>`), then rerun `qda-agent prepare`; a per-user quota counts as full. |
| `<cand0 path> is an unfinished or other build (...); remove it and its .recipe.yaml` | `rm -r <cand0 path> <cand0 path>.recipe.yaml`, then rerun `qda-agent prepare`. |
| `` no `hf` command beside this interpreter or on PATH `` | `uv pip install "huggingface_hub[cli]"` |
| `error: <run_dir>: no FINAL.json; evaluate a run that has finished` | Let `qda-agent search <dir>` finish, then `qda eval <dir>/run`. |
| `<dir>: a directory with neither FINAL.json (a search run) nor build.json (a checkpoint)` | `qda eval <prepare dir>/run` after the search finishes, or `qda eval <checkpoint dir>`. |
| `run_dir <run> is under <tmp>, which the agent can read and write at isolation.level <level>` | `--set run_dir=<dir outside /tmp and /var/tmp>`; keep builds local with `--set work_dir=/var/tmp/<fresh dir>`. |
| `<run_dir> is not empty; give a new run_dir` | `--set run_dir=NEW_DIR`; a relative `NEW_DIR` is relative to the current directory, not to the config file. |
| `work_dir <dir> exists; give a fresh one per run` | `--set work_dir=<new dir>`, or drop `work_dir` to use `run_dir/work`. |
| `no candidate 0 at <path>: run the same prepare again to build it` | Rerun the same `qda-agent prepare` command. |
| `<cand0 out> already exists: candidate 0 is built into a new directory` | `qda-agent cand0 <config> --out <new dir>`, or remove that directory. |
| `<calib path> was made for another model or mix; remove it and its .meta.json` | `rm <calib> <calib>.meta.json`, then rerun `qda-agent prepare`. |
| `--sign-in-dir <dir> is not a directory` | Name the CLI's config dir (`--sign-in-dir ~/.claude`), or drop the flag. |
| `--format '<name>': unknown format '<name>'; expected e.g. 'w4a16-asym-g128'` | `--format w4a16` (or `w3a16`, `w4a8`, `w4a4`, `gguf-q4km`). |
| `judge.anchor <dir>: ...` | `--set judge.anchor=<the model directory prepare downloaded>` |
| `MISS run config: <path>: ... (an earlier qda-agent prepare wrote it ...)` | Run the `qda-agent prepare` command its `fix:` prints. |
| `MISS run config: <path>: ...` | Edit the file it names, or `--set KEY.PATH=VALUE`. |
| `DISK FLOOR: needs at least <X> GB free (2 copies of the bf16 model), <dir> has only <Y> GB` | `qda-agent prepare ... --out <dir on a larger disk>` |
| `disk cap: none (judge.max_run_bytes unset)` | To cap the run: `--set judge.max_run_bytes=<bytes>`. |
| `MISS sandbox: <path>: the job sandbox cannot read it` | `--set 'scheduler.readable=[<dir>]'` |
