"""Judge jobs: queueing a reading, a check or a long generation of an entry, and folding in its result.

A judge job runs a command template from the `judge` config (`kl_cmd`, `check_cmd`, `gencheck_cmd`) on the
frozen copy, with its own temp dir and a config file in the judge's dir. When it ends, `fold` reads only
the fields the agent may see from its result file.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..host import label
from ..sched import Job, Kind, Refused, State
from .intake import read_json
from .ledger import Entry, floats, gen_report, neutral
from .prune import drop_frozen, prune, reap, watch_floor

if TYPE_CHECKING:
    from qda_kl.eval.suites import SuiteResult

    from .config import JudgeConfig
    from .host import JudgeHost

#: How a failed job of each kind is named in its entry's reason.
WHAT = {"scored": "reading", "handin": "check", "gencheck": "long-generation check"}
#: What a reading over `judge.max_kl` tells the agent to check (`{max}`: the configured limit).
KL_HINT = (
    "KL > {max}: check rotation / basis (a rotated checkpoint scored without its rotation, or a wrong basis)"
)
#: Why every entry of a build is ineligible once one of its readings was over `judge.max_kl`.
KL_REFUSED = "not eligible: this build's KL reading was refused"
#: How many of a failed job's log lines `fold` reads to find the failure line it quotes.
LOG_TAIL = 200
#: An exception's summary line, unindented: `pkg.Refused: why`, `ValueError: why`.
EXCEPTION = re.compile(r"^(?:\w+\.)*\w*(?:Refused|Error|Exception|Interrupt)\b.*$")
#: A shipped command's own failure line: the contract's `error: why` (`qda_quant.exit.ERROR_LINE`, printed
#: by `qda`, `qda-kl` and `qda-agent`), or the `<command>: refused|error: why` a tool inside the run writes
#: to its own job log (`suite_job`, the judge's own command line). `test_judge` checks the first against
#: `refusal`.
DIAGNOSTIC = re.compile(r"^(?:[\w.-]+: )?(?:refused|error): \S.*$")
#: The prefix an engine's own processes put on each log line (`(EngineCore_DP0 pid=7) `).
ENGINE_PREFIX = re.compile(r"^\([\w.-]+ pid=\d+\) ", re.MULTILINE)
#: What starts a traceback (the interpreter and `traceback.print_exception` write it).
TRACEBACK = "Traceback (most recent call last):"


TOOLS = Path(__file__).parent / "tools"


def job_key(e: Entry) -> str:
    """The entry's job key: its id, or `<id>-full` for the full reading that follows a screening one.

    The jobs of one entry are named apart: a screening reading and its promoted full reading are two jobs,
    each with its own config, temp dir and result.
    """
    return f"{e.id}-full" if getattr(e, "stage", None) == "full" else e.id


def queue(
    host: JudgeHost,
    e: Entry,
    template: list[str],
    kind: Kind,
    ncards: int,
    expected_s: float,
    moe: bool = False,
    *,
    record: bool = True,
    attempt: int = 0,
) -> None:
    """Submit the judge job of `e`: `template` with its slots filled; the entry records the job's name.

    `record` off leaves the entry out of the ledger (the launch smoke's candidate-0 reading, which is no
    candidate of the run). `attempt` above 0 names a retry apart (`judge-<key>-r1`), so a job the judge's
    own watchdog cancelled can be run again without colliding with the first name.

    Raises:
        Refused: The scheduler refused the job; the entry and its frozen copy are removed.
    """
    key = job_key(e)
    suffix = f"-r{attempt}" if attempt else ""  # a retry gets its own job name and scratch dir
    e.out = host.jdir / "results" / f"{key}.json"
    config = ""
    if any("{config}" in a for a in template):  # `qda-kl score|gencheck CONFIG.yaml`, in the judge's dir
        out_dir = host.jdir / "results" / key
        e.out = out_dir / ("gencheck.json" if e.kind == "gencheck" else "score.json")
        try:
            config = str(job_config(host, e, out_dir))
        except Refused as err:
            # `_discard` removes the judge's own frozen copy only: a configured baseline's directory (a
            # baseline reading, or the launch smoke's candidate 0) is the owner's, never ours to delete.
            _discard(host, e)
            raise Refused(f"{e.src}: {err}") from None
    suites = json.dumps(host.cfg.suite)
    slots = {
        "{cards}": str(ncards),
        "{config}": config,
        "{python}": str(host.cfg.serve_python or job_python()),
        "{ckpt}": str(e.dir),
        "{out}": str(e.out),
        "{ref}": "",
        "{eval}": str(host.eval_dir or ""),
        "{tools}": str(host.eval_dir.parent / "tools") if host.eval_dir is not None else str(TOOLS),
        "{suite}": suites,  # last: a setting's text is no slot to fill
    }
    if host.cfg.engine == "llama.cpp":
        # A job gets the single frozen .gguf file.
        try:
            if any("{suite}" in arg for arg in template):
                raise Refused(
                    "a llama.cpp search runs no suite job: a suite reads a checkpoint vLLM loads; "
                    "`qda-kl score` (the default kl_cmd) and the long-generation check take a .gguf"
                )
            slots["{ckpt}"] = str(_checkpoint_file(e))
        except Refused as err:
            _discard(host, e)
            raise Refused(f"{e.src}: {err}") from None
    if any("{ref}" in arg for arg in template) and host.cfg.kl_ref is not None:
        # A named dataset resolves to its local path; the job binds the judge's symlink to it.
        local = host.cfg.kl_reference()
        reference = host.jdir / "reference"
        if local is not None:
            if not reference.is_symlink():
                reference.symlink_to(local.resolve(), target_is_directory=local.is_dir())
            slots["{ref}"] = str(reference)
    argv = []
    for a in template:  # `{moe}` alone is a list slot: the MoE scorer args, or nothing on a dense model
        argv += ([*host.cfg.kl_moe_args] if moe else []) if a == "{moe}" else [fill(a, slots)]
    if e.kind == "scored" and config:
        _check_layout(host, e, Path(config), argv)
    # Its own temp dir in the judge's: a job's scratch (a gencheck's prompts and answers) never lands in the
    # host's $TMPDIR, which the agent's fence may grant. Sockets are bound in it (vLLM's ZMQ IPC, torch's
    # symmetric memory) and the kernel caps a socket path at 107 bytes, which a deep run dir outgrows: the job
    # names it by an absolute path short enough for them, fixed now (`jobtmp`), so a child that changes
    # directory (ninja running nvcc) still finds it. A configured VLLM_RPC_BASE_PATH wins.
    name = f"judge-{key}{suffix}"
    tmp = host.temps.make(f"{key}{suffix}", name)
    env = host.cfg.job_environ(host.eval_dir.parent if host.eval_dir is not None else None, tmp_dir=tmp)
    try:
        job = host.sched.submit(
            argv,
            str(host.jdir),
            expected_s=expected_s,
            ncards=ncards,
            name=name,
            kind=kind,
            origin="judge",
            env=env or None,
        )
    except Refused as err:
        host.temps.release(name)
        _discard(host, e)
        raise Refused(f"{e.src}: {err}") from None
    if record:
        host.update(e, job=job.name)
    else:
        e.job = job.name


def _check_layout(host: JudgeHost, e: Entry, config: Path, argv: list[str]) -> None:
    """Refuse a reading whose command would split the model differently from `judge.kl_layout`.

    The reading was scheduled for `kl_layout`'s cards. Build the scorer the command will actually run --
    the config the judge wrote with every `--set` value of `argv` applied, exactly as `qda-kl score` applies
    them -- and compare its `tp` and `serve_args` with the layout's. An override that replaces the whole
    `scorer` ancestor changes both without naming either key, so only the parsed result can decide.

    Raises:
        Refused: The scorer's layout differs (the entry and its frozen copy are removed), or qda-kl would
            refuse the config or an override.
    """
    layout = host.cfg.kl_layout
    try:
        tp, serve_args = _scorer_layout(config, argv)
    except ValueError as why:
        _discard(host, e)
        raise Refused(f"{e.src}: qda-kl would refuse the reading's config: {neutral(str(why))}") from None
    if (tp, serve_args) != (layout.tp, dict(layout.serve_args)):
        _discard(host, e)
        raise Refused(
            f"{e.src}: its scorer would run with tp {tp} and serve_args {serve_args}, the reading was "
            f"scheduled for judge.kl_layout's tp {layout.tp} and serve_args {dict(layout.serve_args)}: "
            "a reading's layout comes only from judge.kl_layout"
        )


def _scorer_layout(config: Path, argv: list[str]) -> tuple[object, dict[str, object]]:
    """The `tp` and `serve_args` of the scorer `config` runs with once `argv`'s overrides are applied.

    Raises:
        ValueError: qda-kl would refuse the config or an override.
    """
    from qda_kl.config import ScoreConfig  # the evaluator's own parser: one reading of a config

    scorer = ScoreConfig.from_yaml(config.read_text(), _overrides(argv)).scorer
    d = scorer.model_dump()
    return d.get("tp"), dict(d.get("serve_args") or {})


def _overrides(argv: list[str]) -> list[str]:
    """The `K=V` values `argv` passes to `--set`, read the way argparse reads them.

    `--set K=V`, `--set=K=V` and an unambiguous abbreviation of the option (`--s`, `--se`) all count; `--`
    ends the options. Only values carrying an assignment count, since qda-kl refuses the others anyway.
    """
    values: list[str] = []
    for i, tok in enumerate(argv):
        if tok == "--":
            break
        name, fused, value = tok.partition("=")
        if not (name.startswith("--") and name[2:] and "set".startswith(name[2:])):
            continue
        if fused:
            values.append(value)
        elif i + 1 < len(argv):
            values.append(argv[i + 1])
    return [v for v in values if "=" in v]


def _checkpoint_file(e: Entry) -> Path:
    """The single ``.gguf`` file in a llama.cpp entry's frozen copy.

    The scorer's decoder and the generator take one file, not the entry's directory, so a copy holding none,
    or several, would stand for no model or for two.

    Raises:
        Refused: the frozen copy holds zero or more than one ``.gguf``; the message names the count.
    """
    from qda_quant.gguf.llama_cpp import model_file
    from qda_quant.step import Refused as EngineRefused

    try:
        return model_file(e.dir)
    except EngineRefused as why:
        raise Refused(str(why)) from None


def _count(value: object) -> int | None:
    """A count a check's report states: an int, never a bool (`None`: it stated none)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def gencheck_expected_s(cfg: JudgeConfig, max_tokens: int, per_domain: int) -> float:
    """The judge's time estimate for a check of this answer cap and width, from `judge.gencheck_expected_s`.

    The configured estimate is for `gencheck_max_tokens` answers at `gencheck_per_domain` prompts per domain;
    a check costs the answers it generates, so the estimate scales with that product. One place computes it,
    for the queue's admission and its stated end alike.
    """
    configured = cfg.gencheck_max_tokens * cfg.gencheck_per_domain
    return cfg.gencheck_expected_s * (max_tokens * per_domain) / configured


def job_config(host: JudgeHost, e: Entry, out_dir: Path) -> Path:
    """Write the `qda-kl` config of one job into the judge's own dir; return its path.

    A reading (`qda-kl score`): `checkpoint` the frozen copy, `reference` the run's KL reference, `out` a
    fresh dir the scorer fills with `score.json` on success only, and `scorer.topk` carrying the judge's card
    layout (`kl_layout.tp` and its `serve_args`). Under llama.cpp the checkpoint is the frozen `.gguf`, which
    `qda-kl score` decodes to bf16 against `snapshot` (the anchor) and reads with vLLM. A long-generation
    check (`qda-kl gencheck`): the frozen copy against the original model, with the judge's prompt set, length and
    `original_cache` (the original's answers, made once per run); a `.gguf` is decoded against the original and
    answered by vLLM. JSON is YAML, so no YAML writer is needed.
    """
    path = host.jdir / "results" / f"{job_key(e)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, object] = {"checkpoint": str(e.dir), "out": str(out_dir)}
    if host.cfg.engine == "llama.cpp":
        cfg["checkpoint"] = str(_checkpoint_file(e))  # the scorer's decoder and the generator take one file
    if e.kind == "gencheck":
        host.original_cache.mkdir(exist_ok=True)
        cfg |= {
            "original": str(host.cfg.anchor or host.cfg.model or ""),
            "prompts": str(host.install_prompts() or "builtin"),
            "per_domain": e.per_domain or host.cfg.gencheck_per_domain,
            "max_tokens": e.max_tokens or host.cfg.gencheck_max_tokens,
            "original_cache": str(host.original_cache),
            "tp": host.cfg.gencheck_cards,
        }
    else:
        reference = host.cfg.kl_reference()
        cfg["reference"] = str(reference or "")
        if host.cfg.engine == "llama.cpp" and (snapshot := host.cfg.anchor or host.cfg.model):
            cfg["snapshot"] = str(snapshot)  # what the .gguf was converted from; else the reference's model
        layout = host.cfg.kl_layout
        cfg["scorer"] = {"topk": {"tp": layout.tp, "serve_args": dict(layout.serve_args)}}
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def fold(host: JudgeHost) -> None:
    """Record the verdict or result of every entry whose judge job has ended (call with the host lock held).

    A job that did not end done, or left no result, fails its entry. A long-generation check's frozen copy
    is removed when the check ends: nothing else reads it. Every pass trims build custody (`prune`) and
    removes the private directory of a build that failed without publishing (`reap`): a finished check or a
    last consumer ends a protection without a new reading landing, and its build is prunable then. Below the
    free-space floor, it then stops the agent's running jobs (`watch_floor`).
    """
    for e in host.entries:
        if e.valid is not None or e.job is None:
            continue
        job = host.sched.get(e.job)
        if not job.state.terminal:
            continue
        got = read_json(e.out) if job.state is State.done else None
        if got is None:
            # A job the drain window cut before it could start gives no verdict on the checkpoint.
            cut = job.state is State.failed and "drain window" in (job.reason or "")
            host.update(
                e,
                valid=False,
                cut=cut,
                reason=(
                    f"cut by drain window: {neutral(job.reason)}" if cut else failed_reason(host, e, job)
                ),
            )
        elif e.kind == "gencheck":
            gen, why = gen_report(got, e.per_domain or host.cfg.gencheck_per_domain)
            tokens = got.get("max_tokens")
            per_domain = got.get("per_domain")
            host.update(
                e,
                valid=gen is not None,
                gen=gen or {},
                max_tokens=_count(tokens),
                per_domain=_count(per_domain) or e.per_domain,
                reason=why,
            )
        elif e.kind == "scored":
            _fold_reading(host, e, got)
        else:
            ok = got.get("ok") is True
            host.update(e, valid=ok, reason=e.reason if ok else check_reason(host, got, job))
    # Every pass, not only a landed reading: a check that ended (passed or failed) or the last job that read
    # a build ends a protection, and the build is prunable then even if no later reading lands.
    reap(host)
    prune(host)
    watch_floor(host)  # after the prune pass: what it freed counts before a running build is stopped
    host.temps.release_ended(host.sched)  # an ended job's short temp-dir name; the dir stays in the judge's


def failed_reason(host: JudgeHost, e: Entry, job: Job) -> str:
    """Why `job` failed `e`: its kind, state and rc, then the failure line the job itself wrote.

    The job's log is the only place a refusal or a crash explains itself, and the agent cannot read it: the
    line is quoted here, bounded and with host paths cut, so the agent reads the reason instead of `rc 1`
    alone. A job that wrote no failure line keeps the state and rc only.
    """
    why = f"the {WHAT[e.kind]} job {job.state.value} (rc {job.rc})"
    line = failing_line(host.sched.log(job.name, LOG_TAIL))
    return f"{why}: {neutral(line)}" if line else why


def check_reason(host: JudgeHost, got: dict, job: Job) -> str:
    """Why a hand-in's serving check refused it: the check's own reason, then the failure line its log ends on.

    The check writes the exception it caught, which for an engine that would not start is only "Engine core
    initialization failed. See root cause above"; the cause (a loader error such as "Cannot find any model
    weights") is in the job's log. The log's last failure line (`failing_line`) is added when it says
    something the reason does not, bounded and with host paths cut.
    """
    said = neutral(str(got.get("reason") or "check failed"))
    # an engine's own processes prefix their lines (`(EngineCore_DP0 pid=7) RuntimeError: ...`): read past it
    log = ENGINE_PREFIX.sub("", host.sched.log(job.name, LOG_TAIL))
    line = neutral(failing_line(log))
    if not line or line in said:
        return said
    tail = f"; the job log ends: {line}"
    return said[: max(0, 300 - len(tail))] + tail  # the cause is never the part `neutral`'s bound cuts


def failing_line(text: str) -> str:
    """The last failure line in a job's log `text`, or `""` when it wrote none.

    A refusal or an uncaught exception ends a traceback with an unindented `<module>.<Name>: <why>` line. A
    shipped command reports its own refusal as `error: <why>` (`qda`, `qda-kl`, `qda-agent`), and a tool
    inside the run as `<command>: refused|error: <why>` (`suite_job`), with no traceback; both count. The
    engine's shutdown lines that follow start with a process prefix (`(MainProcess pid=...) ...`), so neither
    rule picks up that shutdown noise. Any failure
    line counts wherever it is in the tail; after the last traceback the first unindented line is the
    traceback's summary even when `EXCEPTION` does not know its name. The last match in the tail wins, so a
    job that refused and then crashed reports the crash.
    """
    lines = text.splitlines()[-LOG_TAIL:]
    best = -1
    for i, line in enumerate(lines):
        if EXCEPTION.match(line) or DIAGNOSTIC.match(line):
            best = i
    marks = [i for i, line in enumerate(lines) if line.startswith(TRACEBACK)]
    if marks:
        for i in range(marks[-1] + 1, len(lines)):
            line = lines[i]
            if line and not line[0].isspace() and not line.startswith("("):
                best = max(best, i)
                break
    return lines[best].strip() if best >= 0 else ""


def _discard(host: JudgeHost, e: Entry) -> None:
    """Remove an agent entry and its frozen copy; trusted baseline inputs belong to their owner."""
    if e in host.entries:
        host.entries.remove(e)
        drop_frozen(host, e)  # its copy goes only if no other entry of the same files holds it


def reading_result(host: JudgeHost, e: Entry, got: dict, metric: str) -> tuple[SuiteResult | None, str]:
    """Check completeness, metric, direction and KL using the reading's trusted job configuration."""
    reading = _suite_result(got)
    metrics = floats(reading.metrics) if reading else {}
    if reading is not None:
        reading = reading.model_copy(update={"metrics": metrics})
    if why := _incomplete(host, e, got, reading, reading.detail if reading else {}):
        return None, why
    if (kl := metrics.get("kl")) is not None and kl > (max_kl := host.cfg.max_kl):
        return reading, f"{KL_HINT.format(max=f'{max_kl:g}')} (read {kl:.4g})"
    if metric not in metrics:
        return reading, f"no finite overall {label(metric)}"
    if reading is None or metric not in reading.higher_is_better:
        return reading, f"the reading does not state the direction of {label(metric)}"
    return reading, ""


def _fold_reading(host: JudgeHost, e: Entry, got: dict) -> None:
    """Record what the agent may read of a reading: its metrics and `domains`, `domains_text` of its detail.

    Only a complete result may count (see `_incomplete`). A reading whose `kl` exceeds `judge.max_kl` is
    refused at once with `KL_HINT`, and its frozen copy is removed: such a checkpoint cannot be the search's
    result, and keeping it would only spend disk.
    """
    reading, why = reading_result(host, e, got, host.cfg.metric)
    metrics = reading.metrics if reading else {}
    higher = reading.higher_is_better if reading else {}
    detail = reading.detail if reading else {}
    kl = metrics.get("kl")
    if kl is not None and kl > host.cfg.max_kl:
        host.update(e, valid=False, reason=why)
        _refused_build(host, e)  # its frozen copy goes in the same pass `reclaim` runs
        return
    # A screening reading's metrics also stay in `screen_metrics`: the rank reads them again after a full
    # reading has replaced `metrics`.
    ranked = {"screen_metrics": {} if why else dict(metrics)} if e.stage == "screen" else {}
    host.update(
        e,
        valid=not why,
        metrics={} if why else metrics,
        higher_is_better={} if why else higher,
        domains=floats(detail.get("domains")),
        domains_text=floats(detail.get("domains_text")),
        # a valid reading keeps the reason it already had (an ineligible build), a failed one gives its own
        reason=why or e.reason,
        **ranked,
    )


def _incomplete(host: JudgeHost, e: Entry, got: dict, reading: SuiteResult | None, detail: dict) -> str:
    """Why a score result is not a complete reading, or "" when it is: whole reference, tokens, per-seq.

    A raw KL reading's command reports a failure in the file's own `error`, which `kl_result` keeps in the
    detail rather than in the mapped result's `error`; both spellings are checked.
    """
    if reading is None:
        return ""  # a malformed file has no metric to rank on; the metric check names that
    failure = reading.error or got.get("error")
    if failure:
        return f"the reading failed: {neutral(' '.join(str(failure).split()))}"
    if "metrics" in got and reading.suite != "kl":
        return ""  # another suite's result is its own measurement
    toks = detail.get("tokens")
    if not isinstance(toks, int) or isinstance(toks, bool) or toks <= 0:
        return "the reading has no scored tokens"
    per = detail.get("per_seq")
    if not isinstance(per, dict) or not per:
        return "the reading has no per-sequence readings"
    if (ids := _reference_ids(host, e)) and (missing := ids - per.keys()):
        return f"the reading leaves {len(missing)} of the reference's {len(ids)} sequences unscored"
    bad = sum(not _seq_reading(value) for value in per.values())
    if bad:
        return f"{bad} per-sequence readings are incomplete"
    return ""


def _seq_reading(value: object) -> bool:
    """Whether a `per_seq` value is a complete per-sequence reading, in the scorer's own schema.

    A key with a null or empty value covers nothing: `SeqKL` requires the sequence's KL and its token count.
    """
    from qda_kl.tally import SeqKL  # the scorer's own schema

    if not isinstance(value, dict):
        return False
    try:
        seq = SeqKL.model_validate(value, strict=True)
    except ValueError:
        return False
    return math.isfinite(seq.kl) and seq.tokens > 0


def _reference_ids(host: JudgeHost, e: Entry) -> set[str]:
    """The sequence ids of the reference the job was given; empty when this host cannot read it.

    The judge's own config decides them: the reference it wrote into the job's config, and the subset
    `judge.suite` names. The result's claim about which subset it scored is never read: a partial reading
    must not name itself complete.
    """
    from qda_kl.store import Reference  # the evaluator's own reader

    path = host.jdir / "reference"
    named = (read_json(host.jdir / "results" / f"{job_key(e)}.yaml") or {}).get("reference")
    if isinstance(named, str) and named:
        path = Path(named)
    elif not (path / "manifest.json").is_file():
        return set()
    try:
        return {s.id for s in Reference(path, subset=_job_subset(host, e)).sequences}
    except (Refused, ValueError, OSError):
        return set()


def _job_subset(host: JudgeHost, e: Entry | None = None) -> str | None:
    """The subset this job's suite names, or None: a `qda-kl score` job reads the whole reference.

    The judge's config decides the subset, never the reading's claim.
    """
    del e
    suite = host.cfg.suite
    settings = next(iter(suite.values())) if isinstance(suite, dict) and suite else None
    subset = settings.get("subset") if isinstance(settings, dict) else None
    return subset if isinstance(subset, str) and subset else None


def _refused_build(host: JudgeHost, e: Entry) -> None:
    """No copy of a refused build's file set may win: every other entry of the same tree becomes ineligible.

    A hand-in of the identical files would otherwise carry the build to the final rules after its reading
    was refused. The sealed tree digest is the same for every entry taken from those files, whatever path
    the agent names; `take` also refuses a new entry of a digest recorded here.
    """
    host.kl_refused.add(e.digest)
    for other in host.entries:
        if (
            other is not e
            and other.kind != "gencheck"
            and other.digest == e.digest
            and other.reason != KL_REFUSED
        ):
            host.update(other, eligible=False, reason=KL_REFUSED)


def _suite_result(got: dict) -> SuiteResult | None:
    """A score job's result file as a `SuiteResult`; None when it holds a malformed one.

    A suite job (`judge/tools/suite_job.py`) writes one. `qda-kl score`, which the default `kl_cmd` runs, writes
    a reading with no `metrics`, which is mapped the way the `kl` suite maps it.
    """
    # like the layout check: the evaluator's own reading of its own file
    from qda_kl.eval.suites import SuiteResult, kl_result

    try:
        return SuiteResult.model_validate(got, strict=True) if "metrics" in got else kl_result(got)
    except ValueError:
        return None


def fill(arg: str, slots: dict[str, str]) -> str:
    """`arg` with each slot (`{ckpt}`, ...) replaced by its value."""
    for k, v in slots.items():
        arg = arg.replace(k, v)
    return arg


def job_python() -> str:
    """This interpreter's path (not resolved: a venv's `python` link stays the venv's)."""
    return os.path.normpath(Path(sys.executable).absolute())
