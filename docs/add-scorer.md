# Add a KL scorer

For a researcher who measures a checkpoint with another divergence or another engine: write a scorer, register it,
and `qda-kl score` runs it. A config names one by its key (`scorer: {topk: {...}}`); a config that names none gets
the built-in `topk` ([validate a checkpoint](verify.md#2-measure-kl)).

## Write the scorer

Subclass `qda_kl.scorers.KLScorer`. Its settings are pydantic fields, `name` is the key you register it under, and
`run` returns an `Outcome`: every reading, and the one result `qda-kl score` writes:

```python
>>> from pathlib import Path
>>> from pydantic import BaseModel
>>> from qda_kl.scorers import KLScorer, Outcome
>>> class Result(BaseModel):
...     kl: float
>>> class Constant(KLScorer):
...     name = "constant"
...     value: float = 0.5
...     def run(self, checkpoint, reference, *, subset=None):
...         r = Result(kl=self.value)
...         return Outcome(readings=[r], result=r)
>>> Constant(value=0.25).to_dict()
{'constant': {'value': 0.25}}
>>> Constant().run("ckpt", Path("ref")).result.kl
0.5

```

- `checkpoint` is the config's `checkpoint` string as given; `reference` is the reference directory, which
  `qda-kl score` checks exists before `run`.
- `subset` is the name a `qda eval` KL suite's `subset:` passes; score only those prompts.
- To refuse, raise `qda_quant.step.Refused("why", fix="what to run instead")`: `qda-kl score` prints `error: why` and
  `fix: what to run instead`, writes no result and exits 2, or 3 with `code=qda_quant.exit.ENVIRONMENT` when the machine, not the input,
  is the cause ([exit status](troubleshooting.md#exit-status)).

## Register and run it

Register the class from your package's [registration hook](extend.md#the-registration-hook):

```python
def register():
    from qda_kl.scorers import SCORERS

    SCORERS.register("constant", "my_scorers:Constant")
```

After `uv pip install -e .`, a `qda-kl score` config selects it:

```yaml
checkpoint: qwen3-0.6b-w4a16
reference: qwen3-0.6b-ref
out: qwen3-0.6b-constant
scorer: {constant: {value: 0.25}}
```

`checkpoint`, `reference` and `out` are required; any other key is refused. `qda-kl score score.yaml` prints each
reading as one JSON line (`{"kl": 0.25}`) and writes `result` to `qwen3-0.6b-constant/score.json`, beside the
`resolved.yaml` that records the scorer and its package.
