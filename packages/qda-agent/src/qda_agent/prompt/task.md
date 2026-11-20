# Goal: the best ${format} tier checkpoint of ${model} you can make by ${deadline}

Make the best ${format} quantization of `${model}` you can in this search and hand in your result
(`handin`, see Submission). It is evaluated after the search, outside this workspace, on the main table
(math, code, science, instruction following). The four count equally: a gain in one does not make up for a
loss in another, and the model's other general abilities should not be lost either. Nothing you can measure
here is that evaluation. The final evaluation grades only the answer after the thinking, so a candidate that
loops or never finishes its reasoning loses the whole problem.

## Clock and resources
${n_cards} GPU(s) from ${start} until **${deadline}** (UTC); the clock is in `CLOCK.md`. At ${deadline}
your session is stopped and every job of yours is cancelled: keep `notes/` current as you go. Nobody will
message you during the search. Builds started in the last hour rarely finish, and cards then sit idle.
Candidates can be built and scored in parallel; consider running several at once.
The calibration and training data are yours to choose; ${calib_line}
${disk_line}
${floor_line}
${full_card_moe_section}
## KL
`kl DIR` reads the KL of a checkpoint against the original model on fixed hidden prompts, total and per domain.
It is the main measurement you have here, and large KL differences usually show up in the evaluation; differences
of a few percent do not decide it.${tie_line} `kl` reads text the original model wrote, so it cannot see behaviour
that only appears when the model writes on its own: answer format, length, when it stops, whether it does what the
prompt asks.

## Long generations
`gencheck DIR [--max-tokens N] [--per-domain K]` writes long answers with a checkpoint and the original model and
counts, per domain, the answers cut at N tokens, the answers ending in a repetition loop and the mean answer
length. The evaluation generates up to ${eval_max_tokens} tokens; a check with a smaller N is faster and misses
loops that appear only in long answers.

## Data
When you train the quantized model against the original, the training text matters: besides text the original
model wrote, you can train on answers the quantized model writes itself, scored token by token by the original,
with prompts from all four evaluated domains. Every calibration and training row must be in the model's own chat
template, with its thinking complete and an end-of-turn token. You can write your own calibration JSONL (or copy
the default file and add rows): `messages` rows, which `qda build` renders in the model's own template, or text
already in it. Point `calib.file.path` at that file and name any `calib.file.samples` its rows pack.

## Tools
Your sandbox has no GPU and no network but your model's API. ${commands_line}${tools}Keep each GPU step in a script under `scripts/`. Write your own code in `scripts/`; it counts as your work.

## Prior
Original model: `${model}`.${prior_line}
${candidate0_kl_line}
A start handed in as-is is not your result. The judge's `kl` is one tool you can use; your measurements are
your responsibility.
`kl` and `handin` take your candidate directories, not the read-only baseline paths.

## Budget and formats
${budget_line}
${budget_accounting}
Format menu: ${format_menu}. Bit width may differ per module within this menu and the budget.
${group_line}${module_formats_line}
${head_line}
${engine_facts} No extra operator at inference: anything a method adds
is folded into the weights offline.${engine_floor_line}${gguf_build_line}

## Submission
`handin DIR [DIR...]` hands in checkpoint directories you made in this search, best first; a later call
replaces the list.
Handing in does not measure: the judge takes no KL reading of a newly handed-in checkpoint and runs no
`gencheck` of it. Whether you score a candidate and whether you run a long-generation check of it before
handing it in are your calls and your risk; an unscored, unchecked hand-in can still be the final result,
and `END-REPORT.md` and `qda-agent evaluate` name whichever of the two the delivered checkpoint lacks.
Scoring requests, hand-ins and generation checks of the same build are separate ledger entries, not new builds.
Report request counts separately from distinct completed builds, identified by the recorded checkpoint digest.
The judge copies a submission at once and checks the copy against the same budget and format rules as the first baseline. Its tokenizer,
chat template and auxiliary files must match the original model byte for byte; its model's architecture
settings must match the original. It must load and run a short inference in ${serving_engine}.
KL > ${max_kl} is refused at once: check rotation / basis (a rotated checkpoint scored without its
rotation, or a wrong basis).
Build-receipt policy: ${receipt_policy}. When enabled, build outputs are hashed at build time and a submitted
file set must match one build's recorded outputs. Hand-edited files and builds from a non-original source
are scored but cannot win; a configured baseline is also an allowed source. With this policy enabled, the
judge keeps every build you handed in whole. When the disk approaches the run's free-space floor it may remove
the weights of other scored builds, the worst ${best_metric} first, and it may remove a build's frozen copies
once nothing reads them. A build whose weights were removed this way can be handed in again: if your own copy
still matches the build's recorded files, the judge freezes it again; otherwise build it again. Keep their
recipes and notes. New operations go through plugins so they are recorded
in the build.${receipt_facts}
At ${deadline} your jobs stop; checks already queued still finish. ${final_rules}
Later changes to a directory you submitted change nothing.

## Notes
Before you compare two numbers, check they were made the same way (reference data, format, size basis,
engine, prompt, output cap, sampling, draws), or state the difference beside them. A claim that one change
helps one measure and hurts another needs the same two checkpoints measured both ways; name both.
Write long tool output to a file and read it with `head`, `tail` or `grep`. Keep `notes/REPORT.md` (what
you built, what you measured and how, why the submission; every number names its file) and `notes/state.md`
(enough to resume) current.

## Scope
The evaluation's items are not in your workspace: do not look for them or rebuild them. Do not edit the
evaluation, grading, serving or search code you were given. The rest of the workspace is yours.
