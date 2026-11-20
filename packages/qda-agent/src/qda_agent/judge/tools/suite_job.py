"""A reading by a named eval suite, for `judge.kl_cmd`: the suite's `SuiteResult` goes to OUT.

    python suite_job.py {suite} CKPT REF OUT

`{suite}` is `judge.suite` as JSON, one `{key: settings}` of a `qda eval` config. The suite runs through `qda eval`'s own runner with
its files in `suites/` beside OUT, so a reference it builds once serves every reading. A refusal, or a suite split
into parts (the judge reads one result), exits 2 and writes no file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qda_kl.cli import publish
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.run import run


def main(argv: list[str]) -> int:
    """Run the suite on the checkpoint; publish its result only when the run accepted it."""
    suite, checkpoint, reference, out = argv
    key, work = Path(out).stem, Path(out).parent / "suites"
    try:
        cfg = EvalConfig(
            reference=reference, checkpoints={key: checkpoint}, out=str(work), suites=[json.loads(suite)]
        )
    except ValueError as why:
        print(f"suite_job: refused: {why}", file=sys.stderr)
        return 2
    if rc := run(cfg):
        return rc
    result = work / "results" / key / f"{cfg.suites[0].name}.json"
    if not result.is_file():
        print(f"suite_job: refused: suite {cfg.suites[0].name} splits into parts", file=sys.stderr)
        return 2
    publish(Path(out), result.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
