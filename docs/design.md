# Design

Read this before you change `qda-quant`, `qda-kl` or `qda-agent`. Each rule below holds for every change.

## Packages

| package | import | owns |
|---|---|---|
| `packages/qda-quant` | `qda_quant` | formats, targets, model graph, steps, recipes, the engine, export, CLI `qda` |
| `packages/qda-kl` | `qda_kl` | references of the original model, in-vLLM top-k KL scoring, eval suites, CLI `qda-kl` |
| `packages/qda-agent` | `qda_agent` | the recipe search: sandbox, GPU scheduler `qsched`, the agent's commands, submission checks, CLI `qda-agent` |

Imports go one way: `qda_agent` → `qda_kl` → `qda_quant`. Do not import torch at module level in `qda_quant`.

## Five concepts

| concept | file | what it is |
|---|---|---|
| `Format` | `qda_quant/format.py` | what is stored: weight bits, activation precision, symmetry, group size |
| `Target`, `KernelPlan` | `qda_quant/target.py` | where it is served, and the numerics the kernel really uses there; methods branch on the plan, never on the GPU name |
| `ModelGraph` | `qda_quant/arch/__init__.py` | derived from the model and checked numerically: groups of linears that share an input, expert banks with their router |
| `Step` | `qda_quant/step.py` | one method: a layer step declares the statistics it `needs`; model steps see the whole model; checkpoint steps read a checkpoint and write one |
| `Recipe` | `qda_quant/recipe.py` | steps plus calibration data; YAML and Python are the same object |

Changing one of them needs a design discussion in the pull request.

## Plan, then run

`build` first compiles model × recipe × format × target into a plan: the graph, the kernel plan per layer kind,
the statistics each layer needs, and every step's `check()`. Anything refused is refused here, with the reason,
before a GPU is touched. The build then runs its stages, each listed with its seconds in `build.json`:

1. `generate`, only when a `selfgen` source must make its rows (vLLM in a child process);
2. the float build: model steps one at a time, consecutive layer steps in one pass over the layers;
3. each checkpoint step, on the previous stage's packed output.

An exporter may write a build alone (`Exporter.writes_alone`) when it needs only the weights on disk; it keeps the
layer pass as its reference, with a test that both write the same bytes.

Every fast path keeps a slow reference path and an equivalence test. What the fast paths do:
[why QDA builds faster](reproduce.md#why-qda-builds-faster). Which builds repeat bit for bit:
[what "the same" means](reproduce.md#what-the-same-means). Calibration sources:
[calibration data](recipe.md#calibration-data).

## Plugins and configs

- Parts outside users extend (methods, formats, architectures, eval suites, datasets, agent backends) are chosen
  by name in the config; code never branches on "which variant is this run" ([extend QDA](extend.md)).
- The quantization grid (`qda_quant/solve/grid.py`) and the clip search (`qda_quant/clips.py`) are plugins too;
  `qda plan` refuses a grid its kernel cannot serve.
- One YAML file starts a run (`qda build`, `qda-kl score`, `qda-agent run`), validated strictly, and the run writes
  its fully resolved config; passing it back repeats the run.
- Ablations are config diffs: change a plugin with `--set agent.backend.name=codex`.
