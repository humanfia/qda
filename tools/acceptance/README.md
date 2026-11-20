# Track B acceptance

`accept.py` runs the five acceptance checks of the flow/search refactor and prints one PASS/FAIL line per
item, with the command it ran and the path of its evidence. It is for a machine that has a card, the serving
engine and a small model; it is not part of the package.

    python tools/acceptance/accept.py \
        --tree  /path/to/candidate \
        --main-tree /path/to/release-checkout \
        --rig   /path/to/search-rig \
        --model /path/to/small-model \
        --python /path/to/venv/bin/python \
        --serve-python /path/to/serving/venv/bin/python \
        --serve-pythonpath /path/to/staged/engine \
        --refuse-under /path/to/shared-fs

`--tree` defaults to the repository this file is in. Everything the run needs is an argument, so the same
command works on any machine; nothing about a cluster is written down here. Every path argument is
resolved, and one under any `--refuse-under DIR` is refused, symlinks included: name the shared filesystem
the run must not read (there is no default), and a pass cannot come from an install there.

## Items

| Item | What it proves | Needs |
| --- | --- | --- |
| A1 | a search at `isolation.level: none` reaches `FINAL.json`, which says which rule chose the result, with `bwrap` unreachable and no `SLURM_*` in the environment | `--rig` |
| A2 | the strict level's run-start decision is the release commit's, sentence for sentence | `--main-tree` |
| A3 | a tiny-model server job, stopped mid-run through the job backend, leaves no process of its tree | `--model`, a card |
| A4 | the repository's public-tree check passes | - |
| A5 | the CPU suite has no failure the release commit does not also have | `--main-tree` |

A1 takes the search rig unchanged: a directory holding `search.yaml` (with `isolation: {level: none}`) and the
`drive.py` that runs it with a scripted agent. A1 runs `drive.py` with `run_dir=` set to a fresh directory
under `--evidence`, so the rig's own run directory is never touched. Three logging shims (`bwrap`, `squeue`,
`scontrol`) are put first on `PATH`: each records its own execution and exits 127, so an execution is
evidence, not a guess. Behind them is a farm: a fresh directory of links to every program of the original
`PATH` except those three names. No directory is dropped (`bwrap` shares a directory with `git` and
`nvidia-smi`), and `which bwrap` must still fail. A rig whose `search.yaml` or `drive.py` names a path under
a `--refuse-under` directory is refused: it would pass for the wrong reason.

A2 runs `strict_probe.py` twice, once against each tree, with one fixed answer to "why is this node not the
run's" (the live answer changes from second to second, so two trees could not be compared). The candidate
probe sets `isolation: {level: strict}` explicitly, so it cannot fall back to the product default, and A2
fails unless the candidate really ran at the strict level. Both probes run on the machine the harness runs
on; repeat the whole command on another machine to compare there too.

A3 needs a card: run the harness under the card queue, or set `CUDA_VISIBLE_DEVICES` to a card this run
holds. The serving engine is imported from `--serve-pythonpath` when given, so it can be staged on the node
rather than installed. The job carries a tag (`QDA_ACCEPT_TAG`) in its environment: this node runs many
sessions of one account, so only a process that reparented to init and carries the tag counts as this job's
stray; other new orphans are reported, not counted. The checkpoint is the job's outer process and every
process below it before the stop; afterwards that tree and the tagged strays must be gone. Zombies are
reported, not counted as survivors. A staged engine may need a name-only `scipy` stand-in ahead of it when
it ships no scipy and the node's own is built for numpy 1.x: the import chain names
`linear_sum_assignment`, and inference never calls it.

A5 runs all three packages' CPU suites twice (release and candidate), with `TMPDIR` unset (a child with no
TMPDIR of its own is what the suites expect). Failures that only the environment causes are the same on both
trees and do not fail the item; a failure that appears only on the candidate does. It is the slowest item:
`--skip A5` leaves it to the merge gate.

## Interpreting the result

An item whose code is not in the candidate tree reports `FAIL  not landed: <file>`, without running anything:
the harness is written to be usable before all of the refactor is merged. Exit status is 0 when every item
that ran is PASS or SKIP, 1 when one is FAIL.
