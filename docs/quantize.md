# Quantize your own model

For one checkpoint from one `qda build`, with no agent, start from the [quickstart](quickstart.md) recipe.

## 1. Point the recipe at your model

```bash
sed -e 's|^model: .*|model: Qwen/Qwen3-8B|' -e 's|^out: .*|out: qwen3-8b-w4a16|' configs/quickstart.yaml > my-model.yaml
qda plan my-model.yaml
qda build my-model.yaml
```

Set `model:` to a Hub id (`org/name@<commit>` pins a revision) or a local directory.

Run `qda plan` before you spend GPU time: it downloads no weights, needs no GPU, and prints the plan (layers,
kernels, peak card memory) or why the build would refuse.

For a gated model such as Llama or Gemma, accept its licence on the Hub and set `HF_TOKEN` or run `hf auth login`
before building.

A model published only as `pytorch_model.bin` is not supported. A model type `transformers` does not include
(such as `kimi_k2`, `internlm3` or `chatglm`) is not supported: `qda plan`, `qda build` and `qda fisher map`
exit 2 with `its config is code from the model repository (trust_remote_code); QDA never runs it`.

## 2. Pick the format

Pick from [formats](formats.md): `w4a16` when unsure, `w3a16` for less memory, `w4a8` to serve on H100 / H200, `w4a4` on B200 / GB300 / RTX PRO 6000. Set it with
`format:` in the recipe or `--set format=w3a16`. To build on one card for another, add `--set target=h100`
([the GPU table](formats.md#does-my-gpu-run-it)). For llama.cpp, write a GGUF recipe instead
([GGUF](formats.md#gguf-for-llamacpp)).

### Mixed precision under a budget

To store some units wider than the body, list them in `formats:` (it replaces `format:`):

```yaml
model: Qwen/Qwen3-8B
formats: {default: w4a16-g128, layers.0-3.mlp: w8a16-g128}
out: qwen3-8b-mixed
steps: [{gptq: {}}]
calib:
  file: {path: calib.jsonl, samples: 128, seqlen: 2048}
```

To choose formats from measured sensitivity, build a map from the local original checkpoint and your calibration
file, then allocate the budget:

```bash
model_dir=$(hf download Qwen/Qwen3-8B)
qda fisher map --model "$model_dir" --calib calib.jsonl --menu bf16,w8a16-sym-g128,w4a16-asym-g128 --out map.json
qda fisher allocate map.json --body w4a16-asym-g128 --budget-bpw 4.4    # JSON; "formats" is the mapping
```

Copy `formats` into the recipe and check it with `qda plan`. The budget counts the decoder Linears, and the head
with `--head-lease`, as a search's does ([bits per weight](search.md#bits-per-weight)).

`--head-lease` picks the format of an untied output head too and lends what it saves to the body
([head lease](formats.md#head-lease)); the output names it (`head`) and, when not bf16, the step to add to `steps:`
(`head_step`).

## 3. Use your own calibration data

For a domain model, calibrate on its own data: a JSONL file of `{"messages": [...]}` or `{"text": ...}` rows.

```yaml
model: Qwen/Qwen3-8B
format: w4a16
out: qwen3-8b-w4a16-mine
steps: [{gptq: {}}]
calib:
  file: {path: my-domain.jsonl, samples: 128, seqlen: 2048}
```

To render a file in the model's chat template, or for Hub datasets, weighted mixes and the model's own answers:
[recipe reference](recipe.md#calibration-data).

## 4. Use several GPUs

A build uses every visible card; name them with `--set 'devices=[0,1]'`. `guided`, `fisher_trace`, `gptaq` with a
nonzero `alpha`, and a model step after a layer step run on one card: with `devices` unset the build takes the first
visible card; with several named it is refused with `fix: set devices: [<first>]`. When a card cannot hold a stage,
`qda plan` refuses: add cards, or `--set residency=stream` to keep the weights in host memory. A multi-card build
opens about one file per calibration sample per card ([open-file limit](install.md#open-file-limit)).

## Model families

These families are built, served by stock vLLM 0.33 and checked; QDA reads a model's structure from one forward
pass, so most other decoder models need no model-specific code:

| family | examples |
|---|---|
| Llama | Llama-3.1-8B, Llama-3.2-3B, DeepSeek-R1-Distill-Llama-8B |
| Qwen 2.5 / 3 / 3.5 / 3.6 / 3.8 | dense, MoE, hybrid, and VL |
| Gemma 2 / 3 / 4 | Gemma-2-9B-it, Gemma-3-12B-it, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| GLM | GLM-4-32B-0414, GLM-4.7-Flash, GLM-4.5-Air |
| Mistral, Mixtral | Mistral-7B-Instruct-v0.3, Mixtral-8x7B, Mistral-Small-3.2-24B |
| DeepSeek V2 | DeepSeek-V2-Lite-Chat |
| Nemotron-H | Nemotron-3-Nano-30B-A3B |
| others | gpt-oss-20b, Phi-4-mini, OLMo-2-7B, Granite-3.3-8B, SmolLM3-3B |

## When `qda plan` refuses

The refusal names the reason and, when QDA knows it, the fix.

| the refusal says | do |
|---|---|
| `<N> input columns at group size 128 ... is not a whole number of groups; ... group sizes that work: ...` | use a listed group size: `--set format=w4a16-asym-g64` |
| `w4a8-sym-g128 dense on sm80 ...: FP8 activations need Hopper` | pick another format, or `--set target=h100` |
| `deriving its graph failed in <module>` | a new layout: write an adapter ([add an architecture](add-architecture.md)) |

A line `qda: warning: norm fold abandoned: <norm> -> <group>` is not fatal; the checkpoint does not change.

## Next

- [Validate the checkpoint](verify.md) before you ship it.
- [Agent search](search.md) for a lower KL at the same size.
- [Recipe reference](recipe.md): every key and method.
