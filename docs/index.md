# QDA

Paper: [QDA: Closed-Loop LLM Quantization Design with Agents](https://arxiv.org/abs/2611.23388). Blog: [mit-han-lab.github.io/qda-blog](https://mit-han-lab.github.io/qda-blog).

## Pick your path

- **One GPU.** [Install](install.md) (Linux, NVIDIA driver 580+, about 10 GB of disk). [Download a published
  model](published.md) of 10, or [build Qwen3-0.6B](quickstart.md) at W4A16: a 1.7 GB download, then 46 s on an H100. Then
  [quantize your own model](quantize.md).
- **Your model, 1 to N GPUs or Slurm.** [Agent search](search.md): 4 GPU-hours by default; `prepare` takes 458 s
  for Qwen3-8B on one H100. [Validate the result](verify.md).
- **Research.** [Reproduce a published model](reproduce.md).

```{toctree}
:hidden:
:caption: Get started

install
quickstart
published
```

```{toctree}
:hidden:
:caption: Guides

quantize
verify
search
reproduce
```

```{toctree}
:hidden:
:caption: Extend

extend
design
add-method
add-scorer
add-architecture
add-target
```

```{toctree}
:hidden:
:caption: Reference

cli
recipe
search-config
formats
checkpoint-tools
troubleshooting
```
