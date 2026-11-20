# Generation-check prompts

96 public training prompts: 24 each for math, code, science, and instruction.
These check generation behavior; they are not a task-accuracy benchmark. No
answers or reference solutions are included.

| Domain | Dataset | Revision | Split / file | License |
|---|---|---|---|---|
| math | [project-numina/NuminaMath-CoT](https://huggingface.co/datasets/project-numina/NuminaMath-CoT/tree/9d8d210c9f6a36c8f3cd84045668c9b7800ef517) (formerly AI-MO/NuminaMath-CoT) | `9d8d210c9f6a36c8f3cd84045668c9b7800ef517` | train, `data/train-00000-of-00005.parquet` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| code | [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests/tree/802411c3010cb00d1b05bad57ca77365a3c699d6) | `802411c3010cb00d1b05bad57ca77365a3c699d6` | train, `data/train-00014-of-00039-dc3ebb07a3cba8e4.parquet` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| science | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453), ARC-Challenge | `210d026faf9955653af8916fad021475a3f00453` | train, `ARC-Challenge/train-00000-of-00001.parquet` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| instruction | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a) | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | train, `databricks-dolly-15k.jsonl` | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |

Attribution: Jia Li and the NuminaMath authors (NuminaMath), Yujia Li and the
DeepMind AlphaCode authors, with source problems from [Codeforces](https://codeforces.com/)
(CodeContests), Peter Clark and the AI2 ARC authors (ARC), and Databricks and
its employee contributors (Dolly). Source prompts retain their listed licenses;
the science and instruction adaptations are distributed under their respective
ShareAlike licenses. Dataset-card links above identify the original work and
its terms. The library's code license does not replace these data licenses.

## Selection and changes

Ids end in the zero-based row index in the pinned source file. Math and code ids
also include the shard index (`00000` for math, `00014` for code). File order is
math, code, science, instruction, with increasing source indices in each group.
Only training data is used, including the training split of ARC. No AIME, HMMT,
GPQA, LiveCodeBench, MATH-500, IFEval, validation, or test split is selected.
Rows whose text repeats a prompt already used by this project's evaluation or
calibration sets, or by an earlier version of this file, are skipped; each rule
below takes the first admissible rows that remain.

Every shipped row carries `source`: the NuminaMath source group, the
`code_contests` dataset name, the ARC-Challenge split name, or the Dolly
category.

- Math: the first 24 admissible training rows in shard 0 (source indices
  5181–6122) whose `source` is `olympiads` or `aops_forum`, whose `problem` has
  at least 200 characters and includes the word `prove` or phrase `show that`,
  and whose entire source row contains neither `AIME` nor `HMMT`
  (case-insensitive substring matching, including solutions and messages before
  those fields are discarded). The `amc_aime` source is excluded. Use only
  `problem` as the source text. Append: "Give a complete derivation, justify
  each step, then verify the result independently. Discuss the assumptions and
  any boundary cases."
- Code: the first 24 admissible training rows in shard 14 (source indices
  17–147) with `cf_rating >= 2200`. Use only the unmodified `description` field,
  including any public examples already present in that description. No
  solutions, separate test cases, ratings, or extra instructions are shipped in
  the prompt.
- Science: the first 24 admissible `question` fields of the ARC-Challenge
  training split (source indices 164–217), followed by each original choice as
  `label. text` in source order. Append: "Explain the relevant science from
  first principles, analyze every option, and justify your answer. Give two
  related examples and discuss a common misconception."
- Instruction: the first 24 admissible rows whose `category` is `brainstorming`
  or `creative_writing` (source indices 157–269). Use `instruction`, followed by
  `Context:` and `context` when nonempty. Append: "Write a detailed response in
  exactly five numbered sections. Give each section a descriptive heading,
  include a concrete example in each, and end the fifth section with a
  one-sentence summary."

Each appended instruction is separated from the source text by a blank line.
These additions request explanations, tests, and structured answers to exercise
longer generation. They do not enforce a minimum output length. The default run
takes the first 4 prompts per domain; a full check uses 16. `per_domain` may be
at most 24 with this set.
