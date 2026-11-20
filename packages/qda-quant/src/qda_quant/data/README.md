# Bundled corpus data

## `reasoning-kl3.jsonl`

The paper's reference prompts: 48 prompts from AIME 2000--2020, SuperGPQA, OpenThoughts and
OpenHermes (paper, appendix: "The reference traces"), one JSON object per line with `id`, `domain`
and `prompt`. `qda_quant.intake.corpus.ReasoningCorpus` reads it and pins its sha256:

```
535f7848b9227880482104dba6c1f50d0dc371b0be0ea23755795b7b91359ea2
```

The file is not distributed with this package yet. It is the `prompts/prompts.jsonl` of the paper's
kl3 corpus, whose traces were built from the original model's own continuations at seeds 17 and 18.
The prompts come from the public datasets named above and keep those datasets' licenses; the
distribution terms are the source datasets' terms, not this package's Apache-2.0 license. The paper
marks the corpus "do not calibrate", which is why `reasoning` has `holdout = 1.0`.

Until the file is present, `ReasoningCorpus.rows()` refuses and names the `file` corpus to use
instead.
