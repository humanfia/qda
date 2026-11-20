# Extend QDA

For a researcher who adds a part from their own package, without changing QDA. Pick the page for your part:

| to add | read |
|---|---|
| a quantization method | [Add a method](add-method.md) |
| a stored format | [Format](#format) below |
| an architecture `qda plan` refuses | [Add an architecture](add-architecture.md) |
| a KL scorer | [Add a scorer](add-scorer.md) |
| a GPU or a kernel row | [Add a target](add-target.md) |
| a grid, clip search, exporter, eval suite, dataset or agent part | [the table](#what-you-can-register) below |

## The registration hook

A package declares one `qda.plugins` entry point: a function with no arguments that QDA calls once, before the
first registry read. It registers each object, or a lazy `"module:attribute"` reference, under the name a config
selects it by:

```toml
[project.entry-points."qda.plugins"]
my_pkg = "my_pkg:register"
```

```python
def register():
    from qda_quant.step import REGISTRY

    REGISTRY.register("shrink", "my_pkg.steps:Shrink")
```

One `register()` may register any number of objects, of any kinds. Then:

1. Install the package: `uv pip install -e .`. Run it again after you change the entry points; an editable install
   picks up code edits only.
2. Check the name: `python -c "from qda_quant.step import REGISTRY; print(sorted(REGISTRY))"` (any registry of
   [the table](#what-you-can-register) prints its names the same way).

| you see | fix |
|---|---|
| `unknown step 'shrink'; known: [...]` | the package is not installed in this environment, or its entry point changed: step 1 |
| `steps plugin 'shrink' is provided twice: by ... and by ...` | rename yours; built-in names are taken |
| a traceback ending `qda.plugins hook 'my_pkg' failed` | your `register()` or a module it names raised: fix the traceback above it |

### A complete package

[examples/plugins](https://github.com/humanfia/qda/tree/main/examples/plugins) is a whole plugin package. The
files a method needs, its other modules left out:

```text
examples/plugins/
  pyproject.toml
  src/qda_example_plugins/__init__.py   # register(), one line per object, e.g. REGISTRY.register("clip_rtn", ClipRTN)
  src/qda_example_plugins/steps.py      # ClipRTN, the step register() adds as clip_rtn
```

Its `pyproject.toml`, as it is:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "qda-example-plugins"
version = "0.1.0"
description = "Examples of external QDA plugins"
requires-python = ">=3.12"
license = "Apache-2.0"
dependencies = ["qda-agent", "qda-kl", "qda-quant"]

[tool.hatch.build.targets.wheel]
packages = ["src/qda_example_plugins"]

[project.entry-points."qda.plugins"]
example = "qda_example_plugins:register"
```

List in `dependencies` the packages whose registries `register()` imports. Once `uv pip install -e examples/plugins`
has installed it, `clip_rtn` is a recipe step name.

## What you can register

| kind | registry | config key | subclass or shape | example |
|---|---|---|---|---|
| quantization step | `qda_quant.step.REGISTRY` | a recipe step name | a `Step` class ([add a method](add-method.md)) | [steps.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/steps.py) |
| quantization grid | `qda_quant.solve.grid.GRIDS` | `grid:` | `Grid`: `scale`, `usable`, `stored` | [grids.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/grids.py) |
| format | `qda_quant.format.FORMATS` | `format:` | a `Format` instance ([below](#format)) | `qda_quant/format.py` |
| clip search | `qda_quant.clips.CLIPS` | a step's `clip:` | `Clip`: `_choose` returns `(scale, zero)` | `qda_quant/clips.py` |
| weight store | `qda_quant.engine.stores.WEIGHT_STORES` | `residency:` | `WeightStore`: `fits`, `load` | `qda_quant/engine/stores.py` |
| statistic | `qda_quant.step.STATISTICS` | a step's `needs` | `Statistic`: `provide` | `qda_quant/engine/stat_plugins.py` |
| exporter | `qda_quant.output.exporters.EXPORTERS` | `exporter:` | `Exporter`: `refusal`, `writer`, optional `prepare` | `qda_quant/output/exporters.py` |
| target | `qda_quant.target.TARGETS`, `BINDINGS` | `target:` | a `Table` and two binding rules ([add a target](add-target.md)) | `qda_quant/target.py` |
| checkpoint format | `qda_quant.intake.formats.CHECKPOINT_FORMATS` | read by `qda inspect` | `CheckpointFormat`: `matches`, `describe`, `serve` | `qda_quant/intake/formats.py` |
| architecture adapter | `qda_quant.arch.adapters.ADAPTERS` | matched on the model | an instance with `matches`, `adjust` ([add an architecture](add-architecture.md)) | [architecture.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/architecture.py) |
| doctor check | `qda_quant.doctor.CHECKS` | run by `qda doctor` | a function of `Context` returning `Check` lines | `qda_quant/checks.py` |
| KL scorer | `qda_kl.scorers.SCORERS` | `scorer:` | a `KLScorer` class ([add a scorer](add-scorer.md)) | [scorers.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/scorers.py) |
| eval suite | `qda_kl.eval.suites.SUITES` | `suites:` | `Suite`: `run` returns `metrics`, `higher_is_better` | [suites.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/suites.py) |
| dataset | `qda_kl.datasets.DATASETS` | a dataset setting | `Dataset`: `local() -> Path` | [datasets.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/datasets.py) |
| final rule | `qda_agent.judge.final_rules.FINAL_RULES` | `judge.final_rules` | `FinalRule`: `decide`, `statement` | [final_rules.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/final_rules.py) |
| agent backend | `qda_agent.agents.BACKENDS` | `agent.backend` | `Backend`: `spec`, `place` | [backends.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/backends.py) |
| tool component | `qda_agent.components.COMPONENTS` | `toolset.components` | `Component`: `install`, `protected` | [components.py](https://github.com/humanfia/qda/blob/main/examples/plugins/src/qda_example_plugins/components.py) |

Unlinked paths are under `packages/qda-quant/src/`.

The rules every plugin keeps:

- A setting is a pydantic field, recorded in `resolved.yaml`.
- A part that cannot serve a format or plan says so in `refusal` (grid, clip search, exporter) or `check`
  (weight store); `qda plan` refuses before any GPU work.
- A checkpoint holds exactly what the kernel computes.

## Format

A format is what a checkpoint stores. Subclass the frozen dataclass `Format`, give it a `name`, and override
`qmin`/`qmax` when its integer codes are not the full `weight_bits` range:

```python
>>> from dataclasses import dataclass
>>> from qda_quant.format import Format
>>> @dataclass(frozen=True)
... class Ternary(Format):
...     weight_bits: int = 2
...     group_size: int = 64
...
...     @property
...     def name(self) -> str:
...         return "w2a16-ternary-g64"
...
...     @property
...     def qmin(self) -> int:
...         return -1
...
...     @property
...     def qmax(self) -> int:
...         return 1
>>> TERNARY = Ternary()
>>> TERNARY.name, TERNARY.qmin, TERNARY.qmax, TERNARY.symmetric
('w2a16-ternary-g64', -1, 1, True)

```

Register the instance, not the class: `FORMATS.register("ternary", "my_pkg.formats:TERNARY")`. A config then names
it by its key or its `name` (`format: ternary`), and `resolved.yaml` records the `name`.

A format builds only on a target whose table has a row for it. Until one does, `qda plan` refuses, e.g.
`w2a16-ternary-g64 dense on sm90 vLLM 0.33.0: no WNA16 kernel serves int2`. Add the row as in
[add a target](add-target.md). Integer codes need nothing else: the quantizer clamps to `[qmin, qmax]` and the
compressed-tensors exporter writes `num_bits`, `symmetric` and `group_size` from the format.

## Clip search

Implement `_choose`. Set `needs_hessian` when errors are weighted by the Hessian diagonal, `picks_ranges` when the
range shrinks, `needs_solve` when it only gives a second-order solve its starting scales, and `only_grid` when it
works on one grid.

## Weight store and doctor check

- With `residency` null, the planner takes the store of lowest `priority` that fits; a recipe whose steps declare
  no `needs` takes the highest. A store whose `check` refuses the plan hands over to the default choice.
- A doctor check returns `Check` lines: a name, `ok`, a detail and, for a miss, the one command that fixes it
  (`fix`). `optional=True` prints `opt` and keeps exit status 0; a required miss prints `MISS` and exits 3
  ([exit status](troubleshooting.md#exit-status)).

## Exporter `prepare`

`prepare(recipe)` runs in every process of a build, after the plan's checks and before any weight loads: a spawned
card process does not inherit the parent's choices. Put a setting every process shares in the exporter's fields.

## Final rule and dataset

- A final rule's `decide(standing)` returns a `Pick`, or None to leave it to the next rule in `judge.final_rules`. A
  `Standing` holds only valid checkpoints, baselines and refusals, never the run's private data.
- A dataset's `local()` returns the same path every time and does no GPU work. Its `uses` (`KL_REFERENCE`,
  `PROMPTS`, `TASKS`) limit where a config may name it. A search refuses a dataset path the agent can read, unless
  the dataset declares `public = True`; a KL reference is never public.
