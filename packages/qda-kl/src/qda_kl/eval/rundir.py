"""`qda eval RUN_DIR`: a search run's winner and candidate 0, scored on the held-out prompts.

The search ranks its candidates on `judge.kl_ref`; the holdout prompts (`judge.kl_holdout_ref`) were read by no
build and no agent, so their KL is the comparison to report. The command reads the run's `FINAL.json` (the
winner) and `resolved.yaml` (candidate 0, the reference settings, the engine), links the holdout prompts'
reference into RUN_DIR/evaluate from the reference cache (built once per machine, with the run reference's
sampling and lengths; `qda-agent prepare` warms the same cache entry in the search reference's engine load),
scores both checkpoints with `qda-kl score`, and prints one line per checkpoint. Each step reuses an
output already there. `qda-agent evaluate RUN_DIR` is an alias that calls :func:`evaluate` here.

qda-kl never imports qda-agent: the run directory is read by its file format (the `FINAL.json` and
`resolved.yaml` keys qda-agent writes; a contract test in qda-agent keeps the two in step). What a run's
writer adds after the verdict (qda-agent's summary card, serve line and warnings) comes from `RUN_REPORTS`,
which qda-agent fills when it is installed.

TODO: a benchmark comparison paired by question belongs in qda-kl; this command reports KL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

import yaml
from qda_quant.exit import ENVIRONMENT, INTERRUPTED, refuse
from qda_quant.registry import Registry

from qda_kl.eval import INSTALL_EVAL

from .presets import PRESETS, Group


class RunReport(Protocol):
    """Lines printed after a run's verdict: what the run's writer knows about the checkpoint it delivered."""

    def __call__(
        self, run_dir: Path, resolved: dict[str, Any], final: dict[str, Any], checked: Path, kl: float
    ) -> str: ...


#: The writers of run directories add their report lines here (qda-agent: its summary card, serve line,
#: missing-measurement warning and rotation hint); each is printed after the verdict, in name order.
RUN_REPORTS: Registry[RunReport] = Registry("eval_run_reports")

__all__ = [
    "RUN_REPORTS",
    "benchmark_config",
    "evaluate",
    "group_out",
    "holdout_reference",
    "original",
    "reading_layout",
    "score_config",
    "suite_groups",
    "verdict",
    "winner",
    "with_lcb_sandbox",
]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def winner(run_dir: Path) -> Path:
    """The checkpoint the run picked: its exported copy, else the one FINAL.json names.

    Raises:
        ValueError: The run has no FINAL.json, or it names no checkpoint.
    """
    final = _json(run_dir / "FINAL.json")
    if not final:
        raise ValueError(f"{run_dir}: no FINAL.json; evaluate a run that has finished")
    exported = (final.get("export") or {}).get("winner")
    picked = exported or (final.get("choice") or {}).get("dir") or final.get("baseline")
    if not picked:
        raise ValueError(f"{run_dir / 'FINAL.json'}: no checkpoint was picked ({final.get('reason', '')})")
    return Path(picked)


def reading_layout(built: dict[str, Any], cards: int) -> dict[str, Any]:
    """The reading layout a holdout reference repeats: the run reference's own (`build` of its manifest).

    `tp` is the reference's recorded tensor parallelism, never the number of cards the run held: a model that
    splits another way (pipeline or expert parallelism beside TP) reads differently, and asking for every card
    can refuse a model those cards cannot hold. The serving arguments are the run reference's too.
    """
    sampling = dict(built.get("sampling") or {})
    return {
        "answers": int(built.get("answers", 2)),
        "seed": int(built.get("seed", 17)),
        "max_tokens": int(sampling.pop("max_tokens", 8192)),
        "sampling": sampling,
        "chat_template_kwargs": dict(built.get("chat_template_kwargs") or {}),
        "tp": int(built.get("tp") or cards),
        "serve_args": dict(built.get("serve_args") or {}),
    }


def reference_note(reference: Path) -> str | None:
    """The one line naming holdout rows the reference skipped as past the model's context, or None."""
    from qda_kl.reference import over_context_line

    return over_context_line(_json(Path(reference) / "manifest.json").get("build") or {})


def holdout_reference(
    resolved: dict[str, Any], python: str, cards: int, run_dir: Path
) -> tuple[Path, list[str] | None]:
    """The holdout reference, and the `qda-kl ref --cache` command that links it into the run.

    The reference cache (`qda_kl.refcache`) is the one owner of "which reference is this original's on these
    prompts with this layout": the command builds the reference once per machine and links
    `RUN_DIR/evaluate/holdout.ref` to it, or only links it when it is cached. Two fine-tunes that share a holdout
    file get a reference each (the key is the original's content), and a rerun pays no engine load.
    """
    judge = resolved["judge"]
    prompts = Path(judge["kl_holdout_ref"])
    if prompts.is_dir():  # already a reference
        return prompts, None
    model = str(judge.get("anchor") or resolved["model"])
    built = _json(Path(judge["kl_ref"]) / "manifest.json").get("build") or {}
    layout = reading_layout(built, cards)
    ref = run_dir / "evaluate" / "holdout.ref"
    command = [python, "-m", "qda_kl.cli", "ref", "--model", model, "--cache"]
    command += ["--prompts", str(prompts), "--out", str(ref), "--tp", str(layout["tp"])]
    command += ["--answers", str(layout["answers"]), "--seed", str(layout["seed"])]
    command += ["--max-tokens", str(layout["max_tokens"])]
    if layout["sampling"]:
        command += ["--sampling", json.dumps(layout["sampling"])]
    if layout["chat_template_kwargs"]:
        command += ["--chat-template-kwargs", json.dumps(layout["chat_template_kwargs"])]
    if layout["serve_args"]:
        command += ["--serve-args", json.dumps(layout["serve_args"])]
    return ref, command


def _scored(out: Path, config: dict[str, Any]) -> bool:
    """Whether `out/score.json` is this config's reading: this checkpoint, against this reference's content.

    A score is a reading of one checkpoint against one reference; `evaluate` run again after the holdout
    prompts (or the checkpoint) changed would otherwise print the old reading beside the new reference.
    """
    from qda_kl.reference import model_identity

    saved, scored = out.with_suffix(".yaml"), _json(out / "score.json")
    if not scored or not saved.is_file():
        return False
    try:
        if yaml.safe_load(saved.read_text()) != config:
            return False
    except yaml.YAMLError:
        return False
    content = _json(Path(config["reference"]) / "manifest.json").get("content_sha256")
    if not content or (scored.get("reference") or {}).get("content_sha256") != content:
        return False
    checkpoint = Path(str(config["checkpoint"]))
    expected = model_identity(checkpoint)
    if checkpoint.suffix == ".gguf" and checkpoint.is_file():  # a .gguf has no config.json: its bytes name it
        from qda_quant.io import sha256_file

        expected["sha256"] = sha256_file(checkpoint)
    return scored.get("checkpoint") == expected


def score_config(resolved: dict[str, Any], checkpoint: Path, reference: Path, out: Path) -> dict[str, Any]:
    """The `qda-kl score` config of one checkpoint: the run's engine and reading layout."""
    judge = resolved["judge"]
    layout = judge.get("kl_layout") or {}
    # the run's whole reading layout, its serve_args too (a `max_num_seqs` the run set)
    topk = {"tp": int(layout.get("tp", 1)), "serve_args": dict(layout.get("serve_args") or {})}
    config = {
        "checkpoint": str(checkpoint),
        "reference": str(reference),
        "out": str(out),
        "scorer": {"topk": topk},
    }
    if judge.get("engine") == "llama.cpp":  # one .gguf, decoded against the anchor and read with vLLM
        files = [checkpoint] if checkpoint.is_file() else sorted(checkpoint.glob("*.gguf"))
        config["checkpoint"] = str(files[0] if len(files) == 1 else checkpoint)
        config["snapshot"] = str(judge.get("anchor") or resolved.get("model") or "")
    return config


def verdict(rows: list[tuple[str, Path, float]], kept: str = "no hand-in beat it") -> str:
    """One line per checkpoint (KL, and its ratio to candidate 0's), then the verdict; `kept` is why a run
    that picked no agent checkpoint kept candidate 0 (FINAL.json's reason)."""
    base = rows[0][2]
    lines = [f"{'':12} {'holdout KL':>12} {'vs cand 0':>10}  checkpoint"]
    lines += [
        f"{name:12} {kl:12.6f} {kl / base if base else float('nan'):10.3f}  {path}" for name, path, kl in rows
    ]
    if len(rows) == 1:
        lines.append(f"verdict: the run kept candidate 0 ({kept}); only candidate 0 is scored")
    else:
        kl = rows[1][2]
        word = "better" if kl < base else "worse" if kl > base else "the same"
        lines.append(
            f"verdict: the winner is {word} than candidate 0 on the holdout prompts ({kl:.6f} vs {base:.6f})"
        )
    return "\n".join(lines) + "\n"


def suite_entry(text: str) -> tuple[str, dict[str, Any]]:
    """The one free `{suite: settings}` entry `--suite` names, as `qda eval` takes it, and its suite's name.

    Raises:
        ValueError: The text is not YAML, is not one `{suite: settings}` mapping or names an unknown suite.
    """
    from qda_kl.eval.suites import parse_suite

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"--suite: not YAML ({e})") from None
    if isinstance(raw, str) and raw:
        raise ValueError(
            f"--suite {raw}: no such preset; give one of {', '.join(PRESETS)} "
            "or one {suite: settings} entry"
        )
    suite = parse_suite(raw, "--suite")  # one key, a known suite, its settings valid
    return suite.name, suite.to_dict()


def suite_groups(text: str) -> tuple[str, tuple[Group, ...]]:
    """What `--suite` names: a preset's groups, or one free `{suite: settings}` entry as one group.

    Raises:
        ValueError: The text names no preset and is not one valid `{suite: settings}` entry.
    """
    if (preset := PRESETS.get(text)) is not None:
        return text, preset
    name, entry = suite_entry(text)
    return name, (Group(name, (entry,)),)


def with_lcb_sandbox(groups: tuple[Group, ...], choice: str) -> tuple[Group, ...]:
    """The groups with `--lcb-sandbox` written into every `livecodebench` entry.

    The suite reads one key (`sandbox`), and the config this command writes is the one place it is set, so
    the choice is in `resolved.yaml` with the rest of the measurement; a config written by hand sets the same
    key.
    """
    return tuple(
        Group(
            group.label,
            tuple(
                {**entry, "livecodebench": {**entry["livecodebench"], "sandbox": choice}}
                if "livecodebench" in entry
                else entry
                for entry in group.suites
            ),
        )
        for group in groups
    )


def suite_ready(names: set[str]) -> tuple[str, str] | None:
    """What a `--suite` run still needs installed and the command that installs it, or None.

    `lm_eval` is behind qda-kl's `eval` extra, so a machine without it is told what to install before any GPU
    work, not after KL has been scored. `full` expands to `lm_eval` and `livecodebench`; `livecodebench`
    refuses through its own suite when bwrap or its grader is missing.
    """
    if "lm_eval" in names or "full" in names:
        import importlib.util

        if importlib.util.find_spec("lm_eval") is None:
            return (
                "lm_eval needs the harness, which is not installed",
                INSTALL_EVAL,
            )
    return None


def original(resolved: dict[str, Any]) -> str:
    """The original model a run read: the judge's anchor when it is a string, else the run's model.

    A dataset plugin names the anchor as a mapping: then the run's model, not the mapping, is the original.
    """
    anchor = resolved["judge"].get("anchor")
    return anchor if isinstance(anchor, str) else str(resolved["model"])


def benchmark_config(
    run_dir: Path,
    resolved: dict[str, Any],
    checkpoints: dict[str, str],
    out: Path,
    suites: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """The `qda eval` config of one group: the run's original, its checkpoints and the group's suite entries.

    The reference is the original the run read (its anchor, else the run's model); the checkpoints are named
    by their role, so the table and the paired view line up with the KL verdict above them. The output sits
    under the run directory (`evaluate/<preset>[/<group>]`), where `qda eval` writes its own `resolved.yaml`.
    """
    return {
        "reference": original(resolved),
        "checkpoints": checkpoints,
        "out": str(out),
        "suites": list(suites),
    }


def group_out(run_dir: Path, preset: str, label: str) -> Path:
    """The output directory of one group: `evaluate/<preset>`, or `evaluate/<preset>/<label>` for a split one."""
    base = run_dir / "evaluate" / preset
    return base if label == preset else base / label


def _benchmark(
    run_dir: Path,
    resolved: dict[str, Any],
    preset: str,
    groups: tuple[Group, ...],
    checkpoints: dict[str, str],
    python: str,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    """Write one `qda eval` config per group, run it, and print its table against candidate 0."""
    # `qda eval` is a qda plugin: run it through the module that loads plugins (`python -m qda_quant.cli`
    # would execute as `__main__` and read a registry no plugin filled).
    qda = [python, "-c", "import sys; from qda_quant.cli import main; sys.exit(main(sys.argv[1:]))"]
    for group in groups:
        out = group_out(run_dir, preset, group.label)
        config = benchmark_config(run_dir, resolved, checkpoints, out, group.suites)
        path = out / "eval.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"evaluate: benchmark {preset if group.label == preset else group.label} -> {path}", flush=True)
        command = [*qda, "eval", str(path)]
        if dry_run:  # every group's jobs are listed; nothing is scored
            if code := subprocess.run([*command, "--dry-run"], env=env).returncode:
                return code
            continue
        if code := subprocess.run(command, env=env).returncode:
            return _refused(run_dir, resolved["cards"], Path(config["out"]), code)
        table = [*qda, "eval", "table", config["out"], "--baseline", "candidate0"]
        if code := subprocess.run(table, env=env).returncode:
            return code
    return 0


def _refused(run_dir: Path, cards: list[int], out: Path, code: int) -> int:
    """Refuse a benchmark whose `qda eval` refused jobs: the run's cards with each job's error, then how to pick
    other cards. The child's status passes through; a signal that ended it is an interruption.

    The child records each refused job as a result carrying its error (`qda_kl.eval.run`); it prints them on
    stdout only, so this is the refusal a reader sees on stderr.
    """
    why: list[str] = []
    for path in sorted((out / "results").glob("*/*.json")):
        error = " ".join(str(_json(path).get("error") or "").split())
        if error and error not in why:
            why.append(error)
    named = f"card{'s' if len(cards) > 1 else ''} {', '.join(map(str, cards))}"
    return refuse(
        f"{named} (the run's cards): the benchmark's jobs were refused: {'; '.join(why) or f'exited {code}'}",
        f"set `cards` in {run_dir / 'resolved.yaml'} to free cards (physical numbers, as nvidia-smi lists "
        "them), then run this qda eval again",
        code=code if code > 0 else INTERRUPTED,
    )


def run(run_dir: Path, suite: str | None = None, dry_run: bool = False, lcb_sandbox: str = "auto") -> int:
    """Score the winner and candidate 0 on the holdout reference and print the verdict.

    With `suite` (one of the `eval_presets.PRESETS` names, or one free `{suite: settings}` entry in `qda
    eval`'s form), also run that benchmark on both checkpoints and print its table; without it this command
    reports KL only, as before. `full` expands the original model's row of the released cards' table
    (`qda_kl.eval.full_preset`) before anything is scored, and prints the row's line first. `lcb_sandbox` is
    what runs LiveCodeBench's generated programs (`--lcb-sandbox`), written into the suite's `sandbox` key.
    """
    resolved = yaml.safe_load((run_dir / "resolved.yaml").read_text())
    python = str(resolved["judge"].get("serve_python") or sys.executable)
    cards = resolved["cards"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))}
    groups: tuple[Group, ...] | None = None
    suite_name = ""
    if suite is not None:
        try:
            suite_name, groups = suite_groups(suite)
        except ValueError as e:  # the suite named is the input: a different command fixes it
            return refuse(str(e))
        if missing := suite_ready({key for group in groups for entry in group.suites for key in entry}):
            return refuse(
                *missing, code=ENVIRONMENT
            )  # the machine: the same command works once it is installed
    try:
        picked = winner(run_dir)
    except ValueError as e:
        return refuse(str(e))
    first = Path(resolved["baselines"][0])
    final = _json(run_dir / "FINAL.json")
    # no agent checkpoint was picked: the result is candidate 0 (the only baseline, or the fallback to it)
    kept = final.get("chosen_by") == "fallback_candidate0" or (
        not final.get("choice") and len(resolved["baselines"]) == 1
    )
    checkpoints = {"candidate0": str(first)} | ({} if kept else {"winner": str(picked)})
    if suite_name == "full":
        # `full` is the released cards' table (qda_kl.eval.full_preset): the run names the original, so the
        # row is expanded here and its line printed before anything else. `qda eval` expands the same table.
        from qda_quant.step import Refused

        from qda_kl.eval.full_preset import applied

        overrides = groups[0].suites[0].get("full") or {}
        try:
            got = applied(original(resolved), list(checkpoints.values()), overrides)
        except Refused as e:  # the table's own verdict, with its status and fix line
            return refuse(str(e), e.fix or None, code=e.code)
        print(got.note, flush=True)
        groups = (Group("full", got.entries),)
    if groups is not None:
        groups = with_lcb_sandbox(groups, lcb_sandbox)
    if dry_run:  # nothing is scored: no holdout reference is built
        if groups is None:
            print(
                f"evaluate: no --suite: would score candidate 0"
                f"{'' if kept else ' and the winner'} on the holdout prompts"
            )
            return 0
        return _benchmark(run_dir, resolved, suite_name, groups, checkpoints, python, env, dry_run=True)
    # the reference cache is the one owner of the holdout reference's name and identity
    try:  # a reference this run cannot build is the input's refusal, not a traceback
        reference, build = holdout_reference(resolved, python, len(cards), run_dir)
    except ValueError as e:
        return refuse(str(e))
    if build is not None:
        print(f"evaluate: holdout reference -> {reference}")
        subprocess.run(build, check=True, env=env)
    else:  # `qda-agent prepare` built it in the search reference's engine load
        print(f"evaluate: holdout reference -> {reference} (reused)")
        if note := reference_note(reference):  # the reading covers fewer rows than the holdout file names
            print(f"evaluate: {note}")
    rows = []
    for name, checkpoint in [("candidate 0", first)] + ([] if kept else [("winner", picked)]):
        out = run_dir / "evaluate" / name.replace(" ", "")
        config = score_config(resolved, checkpoint, reference, out)
        if not _scored(out, config):
            out.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run(
                [python, "-m", "qda_kl.cli", "score", str(out.with_suffix(".yaml"))], check=True, env=env
            )
        rows.append((name, checkpoint, float(_json(out / "score.json")["kl"])))
    sys.stdout.write(verdict(rows, final.get("reason") or "no hand-in beat it"))
    # The checkpoint the verdict is about (the winner, or candidate 0 when the run kept it), with the
    # holdout KL read here: the row the table above ends on. The run's writer says the rest.
    _name, checked, kl = rows[-1]
    for _key, report in RUN_REPORTS.items():
        sys.stdout.write(report(run_dir, resolved, final, checked, kl))
    if groups is None:
        return 0
    return _benchmark(run_dir, resolved, suite_name, groups, checkpoints, python, env, dry_run=False)


def evaluate(
    run_dir: Path, suite: str | None = None, dry_run: bool = False, lcb_sandbox: str = "auto"
) -> int:
    """`qda eval RUN_DIR` (and its alias `qda-agent evaluate RUN_DIR`): check the directory, then :func:`run`."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "resolved.yaml").is_file():
        return refuse(f"{run_dir}: no resolved.yaml", "give a run directory: qda eval RUN_DIR")
    if not (run_dir / "FINAL.json").is_file():  # a `qda build` output holds a resolved.yaml too
        return refuse(
            f"{run_dir}: no FINAL.json: not a finished search run",
            "give a finished run directory: qda eval RUN_DIR; benchmark a checkpoint with qda eval CKPT --suite",
        )
    try:
        return run(run_dir, suite, dry_run=dry_run, lcb_sandbox=lcb_sandbox)
    except subprocess.CalledProcessError as e:
        # The child is a `qda` command: its status is the answer a caller acts on, so it passes through. A
        # signal the child did not catch (a negative status) ended it from outside: an interruption.
        return refuse(
            f"`{' '.join(map(str, e.cmd))}` exited {e.returncode}",
            code=e.returncode if e.returncode > 0 else INTERRUPTED,
        )
