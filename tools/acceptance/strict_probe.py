"""One tree's run-start decision: today's wall (`main`) or the strict level (`candidate`).

Run by `accept.py` for item A2, twice per node: once against a checkout of the release commit (`main`, which
enters today's `isolation.gpu_wall`) and once against the candidate tree (which runs `api.make(cfg)`'s strict
preflight). It prints one JSON object; `accept.py` compares the two: the pid form must be the same, and the
decision sentence must be identical byte for byte. The tree is given as an argument, so no tree path, node
name or cluster detail lives in this file.

Usage:
    strict_probe.py --tree DIR --mode main|candidate [--why TEXT] [--run-dir DIR]

`--why` replaces the live "who else is on this node" answer with one fixed sentence: the live answer differs
from second to second (other sessions start and end), so only a fixed answer lets two trees' sentences be
compared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: The answer to "why is this node not the run's" that both trees are given, so the sentences can be compared.
FIXED_WHY = "other process(es) of this user on this node: 4242 (bash)"


def result(step) -> str:
    """`PASS`, or `REFUSE [<what>] <sentence>` for whatever the step raised."""
    try:
        step()
    except BaseException as error:  # noqa: BLE001 -- the refusal is the answer, of any kind
        return f"REFUSE [{type(error).__name__}] {error}"
    return "PASS"


def hold(cfg) -> None:
    """The "this machine cannot hold the agent CLI" check of today's isolation layer."""
    from qda_agent import isolation

    problem = isolation.hold_problem(cfg)
    if problem is not None:
        raise RuntimeError(problem)


def strict_session(iso, cfg) -> None:
    """What the strict level holds for a whole run: the preflight, then the session."""
    iso.preflight(cfg)
    with iso.session(cfg):
        pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, help="the checkout to import qda-agent from")
    parser.add_argument("--mode", choices=("main", "candidate"), required=True)
    parser.add_argument("--why", default=FIXED_WHY, help="the fixed dedicated-node answer for this run")
    parser.add_argument(
        "--run-dir", default=None, help="a directory the run config may name (default: a temp one)"
    )
    args = parser.parse_args(argv)
    tree = Path(args.tree).resolve()
    for package in ("qda-agent", "qda-kl", "qda-quant"):
        sys.path.insert(0, str(tree / "packages" / package / "src"))

    import tempfile

    from qda_agent import isolation
    from qda_agent.config import RunConfig
    from qda_agent.extensions import register
    from qda_agent.isolation import allocation, wall

    register()  # what the installed `qda.plugins` entry point does; this tree is imported by path

    run_dir = Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix="strict-probe-"))
    fields = {"contamination": "none", "model": "m", "format": "w4a16", "budget_gpu_hours": 1}
    fields |= {"cards": [0], "run_dir": run_dir}
    if (
        args.mode == "candidate"
    ):  # named, not defaulted: the default level follows the product, not this check
        fields["isolation"] = {"level": "strict"}
    cfg = RunConfig(**fields)
    live = allocation.why_not_dedicated()
    out = {
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "mode": args.mode,
        "tree": str(tree),
        "level": getattr(getattr(cfg, "isolation", None), "level", ""),
        "sandbox_pid": cfg.sandbox.pid,
        "pid_form_auto": wall.pid_form("auto"),
        "live_why": live,
        "check": result(lambda: isolation.check(cfg)),
        "hold_problem": result(lambda: hold(cfg)),
    }
    wall.why_not_dedicated = lambda: args.why  # from here on: the fixed answer, for both trees
    out["why_not_dedicated"] = args.why
    if args.mode == "main":
        out["strict"] = result(lambda: _enter(isolation.gpu_wall(cfg)))
    else:
        from qda_agent.isolation import api

        iso = api.make(cfg)
        out["level"] = iso.level
        out["strict"] = result(lambda: strict_session(iso, cfg))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def _enter(manager) -> None:
    """Enter a context manager and leave it at once: what starting a run at that level does."""
    with manager:
        pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
