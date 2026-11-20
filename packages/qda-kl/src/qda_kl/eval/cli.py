"""``qda eval``: score checkpoints against a reference, or print one table of a run's results.

``qda eval CONFIG.yaml [--set k=v] [--only KEY[/SUITE[.LABEL]]] [--dry-run]`` scores the checkpoints.
``qda eval CKPT [CKPT...] --suite PRESET`` scores ``qda build`` outputs against the original their
``build.json`` names, with no config file.
``qda eval RUN_DIR [--suite PRESET]`` scores a search run's winner and candidate 0 (`qda_kl.eval.rundir`).
``qda eval table OUT [--format md|html] [--baseline KEY]`` prints the results table.
``qda eval paired OUT --baseline KEY [--resamples N] [--seed N]`` prints the paired per-question comparison.

Reaches the ``qda`` command line through qda-kl's ``qda.plugins`` hook. Exits 0 when every job is done, 2
when the config or a job was refused, 3 when the machine cannot run it, 1 on an unexpected error
(`qda_quant.exit`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from qda_quant.exit import ENVIRONMENT, refuse

from qda_kl.lifetime import command

#: How the first argument is read; `--help` prints it and a refusal names the rule it failed.
RULE = (
    "the first argument decides the form: a file is an eval config (YAML); a directory holding FINAL.json is "
    "a search run (RUN_DIR); a directory holding build.json is a checkpoint (CKPT, more may follow); "
    "`table` and `paired` print a finished eval's results"
)

FORMS = "qda eval CONFIG.yaml | qda eval CKPT [CKPT...] --suite PRESET | qda eval RUN_DIR [--suite PRESET]"


def form(path: Path) -> str:
    """What the first argument is: ``"config"``, ``"run"`` or ``"checkpoint"``.

    Raises:
        ValueError: it is none of the three, or a directory that reads as both a run and a checkpoint. The
            message is the problem; the caller prints `FORMS` under it.
    """
    if path.is_file():
        return "config"
    if not path.is_dir():
        raise ValueError(f"{path}: no such file or directory")
    run, built = (path / "FINAL.json").is_file(), (path / "build.json").is_file()
    if run and built:
        raise ValueError(f"{path}: holds both FINAL.json (a search run) and build.json (a checkpoint)")
    if run:
        return "run"
    if built:
        return "checkpoint"
    if (path / "resolved.yaml").is_file():  # a search still running (or one that failed) has no FINAL.json
        raise ValueError(f"{path}: no FINAL.json; evaluate a run that has finished")
    raise ValueError(
        f"{path}: a directory with neither FINAL.json (a search run) nor build.json (a checkpoint)"
    )


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add ``eval`` to a subparser set."""
    p = sub.add_parser(
        "eval",
        help="score checkpoints against one reference, one JSON per (checkpoint, suite part)",
        description=f"Score checkpoints against their original. {RULE[0].upper()}{RULE[1:]}.",
    )
    p.add_argument(
        "config",
        type=Path,
        metavar="CONFIG|CKPT|RUN_DIR",
        help="an eval config YAML (an eval's resolved.yaml works too); a checkpoint `qda build` wrote; a "
        "finished search run; or `table` / `paired`",
    )
    p.add_argument(
        "out",
        nargs="*",
        type=Path,
        metavar="CKPT|OUT",
        help="more checkpoints after a first CKPT; qda eval table/paired: the eval run directory",
    )
    p.add_argument(
        "--suite",
        metavar="PRESET",
        help="CKPT and RUN_DIR: the benchmark to run, one of kl, sanity, full (the original model's released "
        "card settings) or specdec; one `{suite: settings}` entry also works, e.g. "
        "'{lm_eval: {tasks: [gpqa_diamond]}}'. Required with CKPT; without it a RUN_DIR reports holdout KL "
        "only. Results go to CKPT-eval/PRESET (change it with --set out=DIR) or RUN_DIR/evaluate/PRESET",
    )
    p.add_argument(
        "--lcb-sandbox",
        choices=("auto", "bwrap", "none"),
        help="CKPT and RUN_DIR: what runs LiveCodeBench's generated programs: auto (default; bubblewrap when "
        "it is installed, else stop), bwrap (require it) or none (run them unsandboxed on this machine)",
    )
    p.add_argument(
        "--format", choices=("md", "html"), help="qda eval table/paired: output format (default md)"
    )
    p.add_argument(
        "--baseline", metavar="KEY", help="qda eval table/paired: checkpoint key to compare against"
    )
    p.add_argument(
        "--resamples",
        metavar="N",
        type=int,
        help="qda eval paired: question resamples per interval (default: the run's paired.resamples)",
    )
    p.add_argument(
        "--seed",
        metavar="N",
        type=int,
        help="qda eval paired: RNG seed of those resamples (default: the run's paired.seed)",
    )
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one key")
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY[/SUITE[.LABEL]]",
        help="run only these jobs: a checkpoint, every part of one suite, or one part (repeatable)",
    )
    p.add_argument(
        "--server",
        metavar="URL",
        help="generate on a running vLLM completions server (e.g. http://127.0.0.1:8000/v1/completions) "
        "instead of starting an engine per job; the transport is recorded, not digested",
    )
    p.add_argument("--dry-run", action="store_true", help="print the jobs and which are cached; run nothing")
    p.set_defaults(run=main)
    return p


@command("qda")
def main(a: argparse.Namespace) -> int:
    """Validate the config, then list (``--dry-run``) or run its jobs; or print a run's table."""
    if a.config == Path("table"):
        return _table(a)
    if a.config == Path("paired"):
        return _paired(a)
    try:
        kind = form(a.config)
    except ValueError as why:
        return refuse(str(why), FORMS)
    if kind == "run":
        return _run(a)
    if kind == "checkpoint":
        return _checkpoints(a)
    if a.suite is not None or a.lcb_sandbox is not None:
        return refuse(
            "--suite and --lcb-sandbox belong to a checkpoint or a run directory; a config names its suites",
            "qda eval CKPT --suite PRESET",
        )
    if (
        a.out
        or a.format is not None
        or a.baseline is not None
        or a.resamples is not None
        or a.seed is not None
    ):
        return refuse(
            "a second argument, --format, --baseline, --resamples and --seed belong to "
            "`qda eval table` or `qda eval paired`",
            "qda eval table OUT",
        )
    try:
        text = a.config.read_text()
    except OSError as why:
        return refuse(f"{a.config}: {why}")
    return _score(text, a, str(a.config))


def _score(text: str, a: argparse.Namespace, where: str) -> int:
    """Validate one config's text with ``--set``, then list (``--dry-run``) or run its jobs."""
    from qda_kl.eval.config import EvalConfig
    from qda_kl.eval.run import jobs, run

    try:
        cfg = EvalConfig.from_yaml(text, a.set)
        todo = jobs(cfg, a.only)
    except (OSError, ValueError) as why:
        # A refused config's own `fix:` line is what the reader needs next; keep it.
        return refuse(f"{where}: {why}", getattr(why, "fix", ""))
    if cfg.full_note:  # the `full` entry's expansion: name the card row applied, before any job
        print(cfg.full_note)
    if a.dry_run:
        for j in todo:
            print(
                json.dumps(
                    {
                        "checkpoint": j.key,
                        "suite": j.suite.name,
                        "part": j.label,
                        "cards": j.part.cards(),
                        "cached": j.cached,
                    }
                )
            )
        return 0
    return run(cfg, a.only, server=a.server)


def _run(a: argparse.Namespace) -> int:
    """``qda eval RUN_DIR``: the run's winner and candidate 0 (`qda_kl.eval.rundir.evaluate`)."""
    from qda_kl.eval.rundir import evaluate

    if a.out or a.set or a.only or a.server is not None or a.format is not None or a.baseline is not None:
        return refuse(
            "a run directory takes --suite, --lcb-sandbox and --dry-run only (it writes its own configs)",
            "qda eval RUN_DIR --suite PRESET",
        )
    return evaluate(a.config, a.suite, dry_run=a.dry_run, lcb_sandbox=a.lcb_sandbox or "auto")


def original(checkpoint: Path) -> str:
    """The original model a ``qda build`` output was built from: its ``build.json`` source.

    ``source_repo`` (the hub id) is preferred when the build recorded one, else ``source`` (what the build
    read: a hub id or a local directory). ``qda-kl kl`` reads it too when a checkpoint has no
    ``resolved.yaml`` (`qda_kl.kl._built_model`).

    Raises:
        ValueError: the build.json is unreadable or names no source.
    """
    try:
        built = json.loads((checkpoint / "build.json").read_text())
    except (OSError, ValueError) as why:
        raise ValueError(f"{checkpoint / 'build.json'}: {why}") from None
    source = built.get("source_repo") or built.get("source") if isinstance(built, dict) else None
    if not isinstance(source, str) or not source:
        raise ValueError(f"{checkpoint / 'build.json'}: names no source model")
    return source


def checkpoint_configs(
    checkpoints: list[Path], suite: str, lcb_sandbox: str = "auto"
) -> list[dict[str, Any]]:
    """The eval configs ``qda eval CKPT... --suite PRESET`` runs: one per preset group, no file written.

    The reference is the original every checkpoint's ``build.json`` names (they must agree); its reference
    answers are built or reused by the ``kl`` rules (the reference cache), as for a config naming it. Each
    checkpoint is keyed by its directory name; the output is ``<first CKPT>-eval/<preset>`` beside it.

    Raises:
        ValueError: a checkpoint has no source, two name different originals, or `suite` is unknown.
    """
    from qda_kl.eval.rundir import suite_groups, with_lcb_sandbox

    first = checkpoints[0].resolve()
    sources = {path: original(path) for path in checkpoints}
    if len(set(sources.values())) > 1:
        named = ", ".join(f"{path} -> {source}" for path, source in sources.items())
        raise ValueError(
            f"the checkpoints were built from different originals ({named}); evaluate them apart"
        )
    keys: dict[str, str] = {}
    for path in checkpoints:
        base = re.sub(r"[^A-Za-z0-9._-]", "-", path.resolve().name).lstrip("._-") or "ckpt"
        key, n = base, 2
        while key in keys:
            key, n = f"{base}-{n}", n + 1
        keys[key] = str(path.resolve())
    name, groups = suite_groups(suite)
    groups = with_lcb_sandbox(groups, lcb_sandbox)
    out = first.parent / f"{first.name}-eval" / name
    return [
        {
            "reference": sources[checkpoints[0]],
            "checkpoints": keys,
            "out": str(out if group.label == name else out / group.label),
            "suites": list(group.suites),
        }
        for group in groups
    ]


def _checkpoints(a: argparse.Namespace) -> int:
    """``qda eval CKPT [CKPT...] --suite PRESET``: score `qda build` outputs with no config file."""
    from qda_kl.eval.rundir import suite_ready

    paths = [a.config, *a.out]
    if a.suite is None:
        return refuse("a checkpoint needs --suite: kl, sanity, full or specdec", "qda eval CKPT --suite sanity")
    if a.format is not None or a.baseline is not None:
        return refuse("--format and --baseline belong to `qda eval table`", "qda eval table OUT")
    for path in paths[1:]:
        if path == Path("table") or path == Path("paired"):
            return refuse(f"{path}: not a checkpoint", FORMS)
        try:
            kind = form(path)
        except ValueError as why:
            return refuse(str(why), FORMS)
        if kind != "checkpoint":
            return refuse(f"{path}: not a checkpoint (no build.json); list checkpoints only", FORMS)
    try:
        configs = checkpoint_configs(paths, a.suite, a.lcb_sandbox or "auto")
    except ValueError as why:
        return refuse(str(why))
    names = {key for config in configs for entry in config["suites"] for key in entry}
    if missing := suite_ready(names):
        return refuse(*missing, code=ENVIRONMENT)
    for config in configs:
        if code := _score(yaml.safe_dump(config, sort_keys=False), a, "qda eval CKPT"):
            return code
    return 0


def _table(a: argparse.Namespace) -> int:
    """Dispatch ``qda eval table OUT`` to the table generator."""
    from qda_kl.eval.report import table_main

    if len(a.out) != 1:
        return refuse("table needs the eval run directory", "qda eval table OUT")
    if a.suite is not None or a.lcb_sandbox is not None:
        return refuse("--suite and --lcb-sandbox need a checkpoint or a run directory", FORMS)
    if a.set or a.only or a.dry_run:
        return refuse("--set, --only and --dry-run need a config, not a table", "qda eval CONFIG.yaml")
    if a.resamples is not None or a.seed is not None:
        return refuse(
            "--resamples and --seed belong to `qda eval paired`", "qda eval paired OUT --baseline KEY"
        )
    if a.server is not None:
        return refuse(
            "--server belongs to `qda eval CONFIG`, not to `qda eval table`",
            "qda eval CONFIG.yaml --server URL",
        )
    return table_main(a.out[0], fmt=a.format or "md", baseline=a.baseline)


def _paired(a: argparse.Namespace) -> int:
    """Dispatch ``qda eval paired OUT --baseline KEY`` to the paired comparison."""
    from qda_kl.eval.report import paired_main

    if len(a.out) != 1 or a.baseline is None:
        return refuse("paired needs the run directory and --baseline", "qda eval paired OUT --baseline KEY")
    if a.suite is not None or a.lcb_sandbox is not None:
        return refuse("--suite and --lcb-sandbox need a checkpoint or a run directory", FORMS)
    if a.set or a.only or a.dry_run:
        return refuse("--set, --only and --dry-run need a config, not a table", "qda eval CONFIG.yaml")
    if a.format is not None:
        return refuse("paired prints Markdown only", "qda eval table OUT --format html")
    if a.server is not None:
        return refuse("--server belongs to `qda eval CONFIG`", "qda eval CONFIG.yaml --server URL")
    argv = [str(a.out[0]), "--baseline", a.baseline]
    if a.resamples is not None:
        argv += ["--resamples", str(a.resamples)]
    if a.seed is not None:
        argv += ["--seed", str(a.seed)]
    return paired_main(argv)
