"""``qda data public``: the recipes' public prompts and default calibration text, from pinned datasets.

Writes two files into ``--out``, the ones docs/verify.md ("Public prompts and calibration") describes:

- ``prompts.jsonl``: the first N problems of NuminaMath-CoT (Apache-2.0), ``{"id", "prompt"}`` per row, the
  prompts the own-answer and trace writers continue;
- ``default-calib.jsonl``: the first N rows of Dolly-15k (CC BY-SA 3.0) as one user/assistant chat each.

Both datasets are read at a fixed commit, streamed, so the same counts give the same bytes on any machine.
Those licenses apply to the files written. The command refuses an existing file rather than replace it: a
pipeline hashes its inputs, and a silently rewritten input is a different build.

    qda data public --prompts 640 --calib 256 --out data/qwen3.5-9b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from qda_quant.exit import refuse
from qda_quant.step import Refused

#: (dataset, commit) of each source; changing either changes every recipe's inputs
NUMINA = ("project-numina/NuminaMath-CoT", "9d8d210c9f6a36c8f3cd84045668c9b7800ef517")
DOLLY = ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")


def _rows(source: tuple[str, str], n: int):
    from datasets import load_dataset

    name, revision = source
    return load_dataset(name, split="train", streaming=True, revision=revision).take(n)


def prompt_rows(n: int) -> list[dict]:
    """The first ``n`` NuminaMath-CoT problems as ``{"id", "prompt"}`` rows."""
    return [{"id": f"numina-{i}", "prompt": row["problem"]} for i, row in enumerate(_rows(NUMINA, n))]


def calib_rows(n: int) -> list[dict]:
    """The first ``n`` Dolly-15k rows as one user/assistant chat each (context appended to the instruction)."""
    out = []
    for row in _rows(DOLLY, n):
        user = row["instruction"] + ("\n\n" + row["context"] if row["context"] else "")
        out.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
        )
    return out


def _write(path: Path, rows: list[dict], want: int) -> str:
    if len(rows) != want:
        raise Refused(f"{path.name}: the dataset gave {len(rows)} rows, asked for {want}")
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _run(a: argparse.Namespace) -> int:
    from qda_quant.output.roots import output_path

    try:
        out = output_path(a.out)
        jobs = [
            (out / "prompts.jsonl", prompt_rows, a.prompts),
            (out / "default-calib.jsonl", calib_rows, a.calib),
        ]
        jobs = [j for j in jobs if j[2]]
        if not jobs:
            raise Refused("nothing to write (--prompts and --calib are both 0)")
        if there := [str(p) for p, _, _ in jobs if p.exists()]:
            raise Refused(f"{', '.join(there)} already exists; remove it or pick another --out")
        out.mkdir(parents=True, exist_ok=True)
        for path, rows, n in jobs:
            print(f"{_write(path, rows(n), n)}  {path}  ({n} rows)")
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    # A stream left partway through a NuminaMath shard aborts the interpreter at exit (datasets 5.1,
    # huggingface_hub 2.4: "PyGILState_Release ... finalizing", status -6), every time when no Dolly
    # stream follows it, so `--prompts N --calib 0` failed after writing a good file. Every file is written
    # and closed by now: leave without the interpreter's teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``qda data public``."""
    p = sub.add_parser("data", help="write the recipes' public data files")
    q = p.add_subparsers(dest="what", required=True)
    one = q.add_parser("public", help="NuminaMath prompts and Dolly calibration text at pinned commits")
    one.add_argument("--prompts", type=int, default=0, help="rows of prompts.jsonl (0: skip the file)")
    one.add_argument("--calib", type=int, default=0, help="rows of default-calib.jsonl (0: skip the file)")
    one.add_argument("--out", required=True, help="directory to write into")
    one.set_defaults(run=_run)
