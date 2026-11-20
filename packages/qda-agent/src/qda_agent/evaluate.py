"""`qda-agent evaluate RUN_DIR`: an alias of `qda eval RUN_DIR`.

The command is `qda_kl.eval.rundir.evaluate`, called in this process with the same arguments; this module
keeps the old spelling and adds the lines only the run's writer knows (`report`), which qda-kl prints after
its verdict through `qda_kl.eval.rundir.RUN_REPORTS` for both spellings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from qda_kl.eval.rundir import evaluate, winner

from qda_agent import summary
from qda_agent.host import rotation_hint, serve_line, unmeasured

__all__ = ["NOTE", "add_parser", "main", "report", "winner"]

#: The one stderr line the alias prints before it runs.
NOTE = "note: qda-agent evaluate is an alias of qda eval RUN_DIR"


def report(run_dir: Path, resolved: dict[str, Any], final: dict[str, Any], checked: Path, kl: float) -> str:
    """The lines after a run's verdict: the summary card, the serve line, what is unmeasured, the rotation.

    `checked` is the checkpoint the verdict is about (the winner, or candidate 0 when the run kept it) and
    `kl` its holdout KL.
    """
    out = []
    judge = resolved["judge"]
    anchor = judge.get("anchor")
    against = Path(anchor) if anchor and Path(anchor).is_dir() else None
    if text := summary.card(checked, original=against, kl=kl, head_lease=bool(judge.get("head_lease"))):
        out.append(text)
    # The card deliberately renders no `vllm serve` line (`summary`'s module docstring): that line belongs to
    # `serve_command_line`, reached through `host.serve_line`. Printed here, once, after the card.
    if line := serve_line(checked):
        out.append(f"\nserve it with:\n  {line}\n")
    # The same two facts FINAL.json records and END-REPORT.md prints; a run from before they were recorded
    # has neither (absent keys) and so has no line
    metric = str(resolved["judge"].get("metric") or "kl")
    if gaps := unmeasured(final.get("reading"), final.get("gencheck"), metric):
        out.append(f"WARNING: {gaps}\n")
    if rotation := rotation_hint(winner(run_dir)):  # a rotated checkpoint needs its draft rotated too
        out.append(f"rotation: {rotation}\n")
    return "".join(out)


def add_parser(sub: Any) -> Any:
    """Add the `evaluate` subcommand."""
    p = sub.add_parser(
        "evaluate",
        help="an alias of `qda eval RUN_DIR`: score a finished run's winner and candidate 0 on the holdout "
        "prompts",
    )
    p.add_argument("run_dir", type=Path, help="the run directory (holds FINAL.json and resolved.yaml)")
    p.add_argument(
        "--suite",
        metavar="PRESET",
        help="a benchmark preset to run after the KL reading: kl, sanity, full (the original model's released "
        "card settings: six lm_eval tasks including MMLU-Pro on the model's own sampling, plus "
        "LiveCodeBench v6) or specdec; one `{suite: settings}` entry in `qda eval`'s form also works, e.g. "
        "'{lm_eval: {tasks: [gpqa_diamond]}}' or '{full: {sampling: {temperature: 0.7}}}'. Without it this "
        "command reports KL only",
    )
    p.add_argument(
        "--lcb-sandbox",
        choices=("auto", "bwrap", "none"),
        default="auto",
        help="what runs LiveCodeBench's generated programs: auto (bubblewrap when it is installed, else "
        "stop), bwrap (require it) or none (run them unsandboxed on this machine)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the benchmark's jobs; score and run nothing")
    return p


def main(args: Any) -> int:
    """The `evaluate` subcommand: the note line, then `qda eval RUN_DIR` with the same arguments."""
    print(NOTE, file=sys.stderr, flush=True)
    return evaluate(args.run_dir, args.suite, dry_run=args.dry_run, lcb_sandbox=args.lcb_sandbox)
