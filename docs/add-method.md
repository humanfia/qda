# Add a method

For a researcher with a new quantization algorithm: write it as a step, test it on a tiny model on a CPU, register
it, then build a real checkpoint with `qda build`.

A method is a `Step` subclass (`qda_quant/step.py`): settings and math. The engine owns the calibration loop, the
devices and the speed tricks ([design](design.md)).

## Pick the kind

| `kind` | override | sees | examples |
|---|---|---|---|
| `"layer"` (default) | `apply_layer(ctx)` | one decoder layer and the statistics it asked for | gptq, awq, rtn, gsq |
| `"model"` | `apply_model(model, graph, run)` | the whole float model, before or between layer passes | rotate, permute, head, guided |
| `"checkpoint"` | `apply_checkpoint(src, out, run)` | a packed checkpoint; writes a new one in the same format | scale_kd, soup |

Set `kind` as a class attribute for a model or checkpoint step (`kind = "model"`). A layer step never registers
hooks or loops over calibration data: it lists the statistics it needs in `needs` (`StatName` in `step.py`:
`hessian`, `cross_hessian`, `expert_hessian`, `act_absmax`, `inputs`, `outputs`, `n_tokens`, `layer_io`), and the
engine computes each once per layer and passes it in the `LayerContext`.

Set `needs_verified_graph = True` when the step reads `LinearGroup.norm` or `.prev`,
`ModelGraph.extra["norm_offset"]` or the norm keys of token weights: only then does the engine run the fp64 check
of those relations.

## Write the step

Settings are pydantic fields with defaults; unknown settings are refused. A step that picks codes subclasses
`QuantStep`, which adds a `grid` setting ([grids](recipe.md#quantization-grids)) and `self.units(ctx)`: one unit per
stack that shares a Hessian (q/k/v together; every expert of a bank at once). Each unit has

- `u.weight()`: the stacked weights `[E, R, C]` (`E = 1` for a dense group);
- `u.quantizer`: this unit's served grid, with `params(w, shrink)` (min/max group scales) and `quantize(w, s, z)`;
- `u.hessian()`, when `needs` lists `hessian` and `expert_hessian`;
- `u.write(codes, scale, zero)`: stores the codes and scales, and the weights the kernel will compute with;
- `u.fmt`: this unit's format. Read it from the unit, not `ctx.fmt` (the body format): `format_overrides` can give
  one unit another.

This step shrinks each group's range, then rounds to nearest:

```python
>>> import torch
>>> from qda_quant.step import LayerContext, QuantStep
>>> class Shrink(QuantStep):
...     """Round to nearest after shrinking each group's min/max range by `ratio`."""
...     name = "shrink"
...     ratio: float = 0.9
...
...     def picks_ranges(self) -> bool:
...         return True
...
...     @torch.no_grad()
...     def apply_layer(self, ctx: LayerContext) -> None:
...         for u in self.units(ctx):
...             w, q = u.weight().float(), u.quantizer
...             s, z = q.params(w, self.ratio)
...             u.write(q.quantize(w, s, z), s, z)
>>> Shrink(ratio=0.8)
Shrink(grid=None, ratio=0.8)
>>> Shrink(ration=0.8)
Traceback (most recent call last):
...
pydantic_core._pydantic_core.ValidationError: 1 validation error for Shrink
ration
  Extra inputs are not permitted ...

```

`steps/rtn.py` and `steps/gptq.py` use the same API; their clip searches are plugins in `qda_quant.clips`
([clip search](extend.md#clip-search)).

- To refuse what cannot run, raise `Refused` with the reason in `check(graph, fmt, plan)`. The engine calls it for
  every step before it loads a weight.
- Return True from `picks_ranges` when the step chooses group ranges, as `Shrink` does: `qda plan` then refuses a
  grid that fits its scales on the values (`q4k`).
- For a setting whose default depends on the build (format, kernel plan, model config), declare the field with
  default `None` and fill it in `resolve(fmt, plan, config, source)`, which returns a copy of the step. The planner
  calls it once per step before `check`; a value the user gave wins; `resolved.yaml` records the result
  (`QuantStep.resolve` does this for AutoRound's grid).
- Put a setting that names a file in the class's `paths`.

## Try it on a tiny model

Build a random model from a `transformers` config, give it a small tokenizer, and run the step through the Python
API. It runs on the CPU in seconds.

```python
>>> import json, tempfile, transformers
>>> from pathlib import Path
>>> from tokenizers import Tokenizer, models, pre_tokenizers
>>> from qda_quant import Recipe, build
>>> from qda_quant.calib.sources import FileSource
>>> tmp = Path(tempfile.mkdtemp())
>>> cfg = transformers.Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
...     num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=128)
>>> _ = torch.manual_seed(0)
>>> transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(tmp / "m")
>>> tok = Tokenizer(models.WordLevel({f"w{i}": i for i in range(128)}, unk_token="w0"))
>>> tok.pre_tokenizer = pre_tokenizers.Whitespace()
>>> _ = transformers.PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(tmp / "m")
>>> rows = [{"text": " ".join(f"w{(7 * i + k) % 128}" for k in range(64))} for i in range(4)]
>>> _ = (tmp / "calib.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
>>> calib = FileSource(path=str(tmp / "calib.jsonl"), samples=4, seqlen=32)
>>> recipe = Recipe(steps=[Shrink(ratio=0.9)], calib=calib)
>>> out = build(str(tmp / "m"), recipe, format="w4a16-sym-g128", out_dir=tmp / "out", devices=["cpu"],
...             log=lambda s: None)
>>> sorted(p.name for p in out.iterdir())  # doctest: +NORMALIZE_WHITESPACE
['build.json', 'config.json', 'generation_config.json', 'model-00001.safetensors',
 'model.safetensors.index.json', 'resolved.yaml', 'tokenizer.json', 'tokenizer_config.json']
>>> import yaml
>>> yaml.safe_load((out / "resolved.yaml").read_text())["steps"]
[{'shrink': {'ratio': 0.9}}]

```

## Register it and build

Put the class in your package (`my_pkg/steps.py`) and register it from the package's `qda.plugins` hook:
[the registration hook](extend.md#the-registration-hook) uses this very step, and
[a complete package](extend.md#a-complete-package) shows the file layout and `pyproject.toml`. The registered name
is the step's key in a config (`- shrink: {ratio: 0.9}`).

Then plan (no GPU) and build the quickstart model with your step in place of its recipe:

```bash
uv pip install -e .
qda plan configs/quickstart.yaml --set 'steps=[{shrink: {ratio: 0.9}}]' --set out=qwen3-shrink
qda build configs/quickstart.yaml --set 'steps=[{shrink: {ratio: 0.9}}]' --set out=qwen3-shrink
```

`qda build qwen3-shrink/resolved.yaml --set out=qwen3-shrink-2` repeats it. Measure the checkpoint as in
[validate a checkpoint](verify.md).

## Before it merges into QDA

For a method added to this repository rather than to your own package:

- Put the class in `packages/qda-quant/src/qda_quant/steps/<name>.py` and its lazy `"module:Class"` reference in
  `BUILTINS` in `steps/__init__.py`.
- Add CPU tests on a tiny random model in `packages/qda-quant/tests/` (`tests/tiny.py` builds them) that show how
  to use the step.
- State its repeat class ([what "the same" means](reproduce.md#what-the-same-means)) in its docstring.
- Keep the rules in [CONTRIBUTING](https://github.com/humanfia/qda/blob/main/CONTRIBUTING.md#rules): fast
  paths keep a reference path, claims name their log.
- Add a row to the [methods](recipe.md#methods) table, and a line in [troubleshooting](troubleshooting.md) for a
  real caveat.
