"""Keep trusted build outputs private until their receipts are sealed and published."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..isolation import reaper, register_published
from ..isolation.api import JobSpec, make_for_build
from ..isolation.devices import card_uuids, visible_cards
from ..isolation.paths import expand_links, python, sandbox_env
from ..sched import Refused
from ..sched.queue import register_build_command
from ..sched.snapshot import remove_tree
from .cli import Parser
from .freeze import file_sha256, hardlink_in, seal, walk

if TYPE_CHECKING:
    from qda_quant.target import Target

    from ..sched import Job, Scheduler
    from .host import JudgeHost

_log = logging.getLogger(__name__)

#: Seconds between SIGTERM and SIGKILL when a build is ended (a cancelled or interrupted host capture).
BUILD_GRACE_S = 5.0


def _copy_config(host: JudgeHost, root: Path, rel: str, into: Path) -> Path:
    """Copy a build's config file out of `root` into `into`, with the run's level holding the copy.

    The contract is `snapshot_file`'s: one regular file, `into/<name>`, never a tree, and a link the agent
    planted is refused rather than followed. `strict`/`sandbox` copy it exactly as before, in the level's
    bwrap snapshot (`snapshot_file`: 30 s, 16 MiB, not recursive). `none` has no bwrap, so the level's own
    single-file copy does it (`NoneIsolation.copy_file`), refusing what that snapshot refuses.

    Raises:
        SnapshotError: The path is outside `root`, unreadable, a link, or not one regular file.
    """
    iso = host.sched.iso
    if iso.level == "none":
        from ..isolation.levels.none import NoneIsolation

        return cast(NoneIsolation, iso).copy_file(root, rel, into)  # `make()` gives none this level
    from ..sched.snapshot import snapshot_file

    return snapshot_file(root, rel, into)


def import_paths(qda_env: dict[str, str], workdir: Path | None) -> list[str]:
    """`qda_env`'s PYTHONPATH, resolved, for the judge's own qda (trusted builds and the trial run's build).

    Raises:
        ValueError: A path reaches the agent's workdir (the judge would import code the agent wrote).
    """
    paths = [str(Path(p).resolve()) for p in qda_env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if workdir is not None and any(
        workdir.is_relative_to(Path(p)) or Path(p).is_relative_to(workdir) for p in paths
    ):
        raise ValueError("the judge's qda cannot import code from the agent workdir (toolset.env PYTHONPATH)")
    return paths


def program(host: JudgeHost) -> tuple[str, list[str]]:
    """The trusted build's interpreter and import paths: the toolset's qda Python and `toolset.env`'s PYTHONPATH.

    The executable's last symlink names a venv's Python and is kept; the prefix above it is resolved, because
    the empty-root sandbox mounts that prefix, not aliases through the caller's home.

    Raises:
        ValueError: the qda program (`toolset.qda` / `toolset.python`) is neither a `qda` script with one
            interpreter nor `python -m qda_quant.cli`, or PYTHONPATH reaches the agent's workdir (the build would import code the agent wrote).
    """
    argv = host.qda_program()
    interpreter = argv[0]
    if Path(interpreter).name == "qda":
        head = Path(interpreter).read_text().splitlines()[0]
        if not head.startswith("#!") or len(head[2:].split()) != 1:
            raise ValueError("trusted qda build needs a Python interpreter")
        interpreter = head[2:]
    elif argv[1:] != ["-m", "qda_quant.cli"]:
        raise ValueError("trusted qda build needs qda or Python -m qda_quant.cli")
    interpreter = str(Path(interpreter).parent.resolve() / Path(interpreter).name)
    try:
        return interpreter, import_paths(host.cfg.qda_env, host.workdir)
    except ValueError:
        raise ValueError("trusted build cannot import code from the agent workdir") from None


def setup(
    host: JudgeHost, cwd: Path, args: list[str], *, here: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Parse a snapshotted build request and confine its destination to the agent workdir.

    Args:
        host: The judge.
        cwd: The directory the request's relative paths resolve against (the job's working directory).
        args: The build's arguments (`config`, `--set`).
        here: The host-owned directory the request path is relative to; None: the agent's workdir. The judge
            writes its own requests (candidate 0's smoke) outside the workdir, where the agent never sees
            them, and snapshots them from there.
    """
    if host.cfg.require_build_receipt and host.sched.cfg.job_sandbox != "bwrap":
        raise ValueError(
            "require_build_receipt needs the jobs in bwrap, and this run's jobs run on the host; "
            "fix: set isolation.level: sandbox or strict"
        )
    parser = Parser(add_help=False)
    parser.add_argument("config")
    parser.add_argument("--set", action="append", default=[])
    request = parser.parse_args(args)
    import yaml
    from qda_quant.recipe import apply_overrides

    from .gguf_template import for_build
    from .prune import check_room, expected_bytes

    root = host.jdir / "builds"
    root.mkdir(mode=0o700, exist_ok=True)
    private = Path(tempfile.mkdtemp(dir=root))
    try:
        root = host.workdir if here is None else Path(here)
        rel = os.path.normpath(os.path.relpath(cwd / request.config, root))
        config = _copy_config(host, root, rel, private)
        base = (cwd / request.config).parent
        raw = apply_overrides(yaml.safe_load(config.read_text()), request.set)
        if not isinstance(raw, dict):
            raise ValueError("trusted build config must be a mapping")
        # The source is resolved here, once, and then handed to the build as an absolute override: a link
        # the agent swaps between this request and the build cannot change which directory the build reads.
        source, read = _source(raw.get("model"), base, host.sched.cfg.binds)
        if why := placeholder(host, read):
            raise ValueError(why)
        output = raw.get("out")
        if not isinstance(output, str) or not output:
            raise ValueError("trusted build needs a nonempty out setting")
        destination = Path(os.path.abspath(cwd / output))  # noqa: PTH100 -- never follow agent symlinks
        if destination == host.workdir or not destination.is_relative_to(host.workdir):
            raise ValueError("trusted build output must be inside the workdir")
        interpreter, paths = program(host)
        if raw.get("trust_remote_code"):
            raise ValueError("trusted build refuses remote model code")
        # The plan is the estimate that matters: a W8 build writes more than the budget's W4, and only the
        # plan knows which. Admission reserves it, under the lock, with every build already queued or
        # running; a plan the judge cannot read falls back to the judge's own estimate of the original. A
        # llama.cpp build is admitted by its plan's weight bytes against the byte budget, or refused.
        # A GGUF build reads the run's one bf16 template instead of converting the source again, set
        # before the plan is read so the plan prices the file; one that still converts its own reserves it.
        handed = for_build(host, raw, source, read, {**host.sched.cfg.agent_job_env, **host.cfg.qda_env})
        raw = handed.raw
        gguf = budget_check(host, raw, str(base))
        expected = (
            gguf
            if gguf is not None
            else _plan_bytes(raw, str(base), _run_target(host)) or expected_bytes(host)
        )
        expected += handed.extra
        with host.lock:
            check_room(
                host,
                expected,
                scratch_expected=host.cfg.job_reserve_bytes if host.cfg.custody_dir else expected,
            )
            host.build_reserved[str(private)] = expected
        spec = {
            "private": str(private),
            "root": str(host.workdir),
            "destination": destination.relative_to(host.workdir).as_posix(),
            "cwd": str(cwd),
            "base": str(base),
            "config": str(config),
            "source": source,
            # host-authored, applied after the agent's overrides: the build reads the resolved source (a
            # bind's mount path when the source is one; the receipt names the bind's backing directory)
            "forced": [*([f"model={json.dumps(read)}"] if read else []), *handed.forced],
            "overrides": request.set,
            "interpreter": interpreter,
            "paths": paths,
            "expected_bytes": expected,  # the cap's reservation until this build's real bytes land
        }
        if host.cfg.custody_dir is not None:
            spec["records"] = str(host.jdir / "records" / "builds" / private.name)
        if any(
            p.resolve().is_relative_to(private.parent) or private.parent.is_relative_to(p.resolve())
            for p in [*host.sched.cfg.readable, *host.sched.cfg.binds.values()]
        ):
            raise ValueError("trusted build custody overlaps an agent-readable bind")
        return private, spec
    except BaseException:
        with host.lock:
            host.build_reserved.pop(str(private), None)  # the request never became a job
        remove_tree(private)
        raise


#: Bytes a llama.cpp build's reservation adds to its plan's exact output size: the writer's own copies
#: (a staged checkpoint beside the .gguf) are not part of the file the plan sizes.
GGUF_HEADROOM = 64 << 20


def _run_target(host: JudgeHost) -> Target | None:
    """The serving target of the run's cards, for a recipe that names none; None when nvidia-smi cannot tell."""
    from ..group_size import run_target

    return run_target(host.sched.cfg.cards)


#: Guards the planner's first import only (`_planner`): the first import of `qda_quant.config` pulls in
#: transformers, and two build requests importing it at once failed one of them with
#: `cannot import name 'AutoConfig' from 'transformers'`. Plans themselves run outside it, concurrently.
_PLANNER_LOCK = threading.Lock()
_BUILD_CONFIG: Any = None
_WITH_TARGET: Any = None


def _planner() -> tuple[Any, Any]:
    """`BuildConfig` and `group_size.with_target`, imported once per process under `_PLANNER_LOCK`."""
    global _BUILD_CONFIG, _WITH_TARGET  # noqa: PLW0603 -- the once-only import's cache
    if _BUILD_CONFIG is None or _WITH_TARGET is None:
        with _PLANNER_LOCK:
            if _BUILD_CONFIG is None or _WITH_TARGET is None:
                from qda_quant.config import BuildConfig  # the build's own parser: it validates what will run

                from ..group_size import with_target

                _BUILD_CONFIG, _WITH_TARGET = BuildConfig, with_target
    return _BUILD_CONFIG, _WITH_TARGET


def warm_planner() -> None:
    """Import the planner at the judge's start, before any request can race it; a failure is only logged.

    `_build_plan` imports it again (once) on its first call, so a start that could not import it still plans.
    """
    try:
        _planner()
    except Exception as error:  # noqa: BLE001 -- never fatal: the first plan retries the import
        _log.warning("build planner could not be imported at start (%s: %s)", type(error).__name__, error)


def _build_plan(raw: dict, base: str, target: Target | None = None) -> Any | None:
    """The build's plan, or None when the judge cannot compute one.

    The build's own parser (`BuildConfig`, which validates what will run) reads the recipe; an unreadable
    model or a recipe whose stages the planner refuses is the build's own refusal when it runs. The plan is
    made on the CPU, as `qda bpw`'s is (`commands._pinned`): the judge holds no card, the build's cards are
    chosen when it is scheduled, and planning on cards reads their free memory, which opens a CUDA context
    in this process on every card it can see, other runs' cards included. A recipe that names no target is
    planned for the run's cards (`target`, `_run_target`), as `qda build` plans it there; a recipe's own
    target wins (`group_size.with_target`). The planner is imported once (`_planner`); plans run concurrently.
    Why a plan could not be read goes to the judge's log only.
    """
    try:
        build_config, with_target = _planner()
        config = build_config.from_dict({**with_target(raw, target), "devices": ["cpu"]}, base=base)
        config.out = None  # the estimate does not depend on where the build writes
        return config.plan()
    except Exception as error:  # noqa: BLE001 -- an unreadable plan falls back to the judge's estimate
        _log.warning(
            "build plan could not be read (%s: %s); reserving the fallback estimate",
            type(error).__name__,
            error,
        )
        return None


def _plan_bytes(raw: dict, base: str, target: Target | None = None) -> int | None:
    """The build plan's own estimate of the output checkpoint's bytes, or None when it cannot be read.

    `Plan.output_bytes` sizes what the recipe writes (a W8 candidate writes more than a W4 one), so the run
    cap reserves that rather than the judge's coarser estimate. A plan the judge cannot compute (an
    unreadable model, a recipe whose stages the planner refuses) reserves the fallback; the build itself
    reports the same refusal when it runs. `target` is `_build_plan`'s.
    """
    plan = _build_plan(raw, base, target)
    value = None if plan is None else plan.output_bytes
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def budget_check(host: JudgeHost, raw: dict, base: str) -> int | None:
    """Refuse a llama.cpp build its plan cannot be priced for, and return the bytes to reserve.

    A llama.cpp search's budget is the whole `.gguf` file the plan writes (`Plan.output_bytes`: the
    exporter's own exact figure, header included), unless the config opts into
    `judge.budget_basis: payload`, which screens `Plan.weight_bytes` (the stored tensors only) instead.
    Only the plan knows either figure before the build writes anything. It is read only when the plan's
    output size is the exporter's own exact figure (`Plan.output_bytes_exact`), so a reservation is never a
    guess; under the payload basis a plan with no weight figure (0) is UNKNOWN, never "fits". The
    reservation is the plan's exact file size plus `GGUF_HEADROOM`, whichever basis screens.

    Args:
        host: The judge (its `judge.budget_bytes`, `judge.budget_basis` and engine).
        raw: The build's config, already snapshotted and overridden.
        base: The directory the config's relative paths resolve against.

    Returns:
        The bytes to reserve, or None when this engine has no such gate (the caller reserves its own).

    Raises:
        Refused: the plan cannot be read, is not exact, does not price the bytes its basis screens, or is
            over budget.
    """
    if host.cfg.engine != "llama.cpp":
        return None
    plan = _build_plan(raw, base, _run_target(host))  # a GGUF names its own target; with_target keeps it
    if plan is None:
        raise Refused(
            "the build plan could not be read (a judge-side error, in the judge's log), so the .gguf's bytes "
            "cannot be checked against judge.budget_bytes"
        )
    if not plan.output_bytes_exact:
        raise Refused(
            "the plan's output size is an estimate, not the exporter's own exact figure "
            "(Plan.output_bytes_exact is false): the byte budget and the disk reservation cannot rest on it"
        )
    basis = host.cfg.budget_basis
    field = "weight_bytes" if basis == "payload" else "output_bytes"
    size, budget = getattr(plan, field), host.cfg.budget_bytes
    if not size:
        raise Refused(
            f"the plan does not price the .gguf's {basis} bytes (Plan.{field} is 0: UNKNOWN), so the "
            "byte budget cannot be checked"
        )
    if budget is not None and size > budget:
        raise Refused(f"{basis} {size} B > budget {budget} B")
    return plan.output_bytes + GGUF_HEADROOM


def estimate(host: JudgeHost, cwd: Path, config: str, overrides: list[str]) -> int:
    """The output bytes to reserve for an agent's local `qda build` (no receipt): `local_build`'s first half."""
    return local_build(host, cwd, config, overrides)[0]


def local_build(host: JudgeHost, cwd: Path, config: str, overrides: list[str]) -> tuple[int, list[str]]:
    """The output bytes to reserve for an agent's local `qda build` (no receipt), and the host's overrides.

    The config is copied out, file-only, as the queued job's own copy is (`_copy_config`): the host never
    follows a path the agent controls, `strict`/`sandbox` snapshot it in bwrap, and at `level: none` the
    level's own single-file copy runs instead (no bwrap). The plan's own estimate is used when it can be
    read, else the judge's estimate of the original's quantized size (`expected_bytes`), as a trusted
    build's admission does. A llama.cpp build is admitted by `budget_check` instead: its weight bytes are
    priced from the plan, or it is refused.

    A GGUF build is handed the run's bf16 template as a trusted one is (`gguf_template.for_build`): the
    returned overrides set it and pin the model to the source it was matched against, and the job applies them
    after the agent's own. A GGUF build that still converts its own template reserves it too.

    Raises:
        Refused: `budget_check`'s (the plan cannot be priced, is not exact, or is over budget), or a recipe
            that names its own GGUF template.
    """
    from .gguf_template import for_build, gguf_settings
    from .prune import expected_bytes

    rel = os.path.normpath(Path(os.path.relpath(cwd, host.workdir), config))
    if Path(config).is_absolute() or rel == ".." or rel.startswith("../"):
        return expected_bytes(host), []  # the job itself refuses a path outside the work directory
    stage = Path(tempfile.mkdtemp(prefix=".estimate-", dir=host.jdir))
    raw = None
    try:
        snap = _copy_config(host, host.workdir, rel, stage)
        import yaml
        from qda_quant.recipe import apply_overrides

        raw = apply_overrides(yaml.safe_load(snap.read_text()), overrides)
    except Exception as error:  # noqa: BLE001 -- an unreadable config is the build's own refusal later
        _log.warning("local build plan could not be read (%s); reserving the fallback estimate", error)
    finally:
        remove_tree(stage)
    if not isinstance(raw, dict):
        return expected_bytes(host), []
    base = (cwd / config).parent
    extra, forced = 0, []
    if gguf_settings(raw) is not None:  # only a GGUF build reads a template
        source, read = _source(raw.get("model"), base, host.sched.cfg.binds)
        env = {**host.sched.cfg.agent_job_env, **(host.cfg.agent_qda_env or host.cfg.qda_env)}
        handed = for_build(host, raw, source, read, env)
        raw, extra = handed.raw, handed.extra
        forced = [*handed.forced, *([f"model={json.dumps(read)}"] if handed.forced and read else [])]
    if (value := budget_check(host, raw, str(base))) is not None:
        return value + extra, forced
    if value := _plan_bytes(raw, str(base), _run_target(host)):
        return value + extra, forced
    return expected_bytes(host) + extra, forced


def _source(model: object, base: Path, binds: dict[Path, Path]) -> tuple[str, str]:
    """The host identity of the model a build reads, and the path it is handed to read it with.

    Returns ``(identity, read)``, both "" when the config names no model. `identity` is the receipt's
    `source`: an absolute resolved directory, a Hub id, or -- for a model the agent names at a judge bind's
    mount path -- the bind's backing directory (`source_sha256` fingerprints that directory, the bytes the
    read-only mount serves). `read` is the override the build gets: the mount path for a bind (the backing
    path need not be readable inside the build's sandbox), the resolved directory otherwise. Paths are
    normalized lexically, never through an agent-controlled link.

    Nested binds are applied parent first, then child (the stock run binds `baselines` to its overlay, then
    `baselines/0` to the baseline), so the *deepest* destination containing the model is the one whose bytes
    the build reads: a shallower match would fingerprint the overlay's empty mount-point directory.
    """
    if not isinstance(model, str) or not model:
        return "", ""
    from qda_quant.recipe import expand_path

    path = expand_path(model)
    named = Path(os.path.normpath(base / path))  # noqa: PTH100 -- never follow agent symlinks
    best: tuple[int, Path, Path] | None = None
    for destination, backing in binds.items():
        dest = Path(os.path.normpath(destination))
        if (named == dest or named.is_relative_to(dest)) and (best is None or len(dest.parts) >= best[0]):
            best = (len(dest.parts), dest, Path(backing))
    if best is not None:
        _, dest, backing = best
        return str((backing / named.relative_to(dest)).resolve()), str(named)
    if model.startswith((".", "~")) or (base / path).is_dir():
        resolved = str((base / path).resolve())
        return resolved, resolved
    return model, model


def placeholder(host: JudgeHost, source: str) -> str:
    """Why `source` is a published build's empty workdir destination, not a model directory.

    A trusted build's output lives in host-only custody (``builds/<name>/published``) and is mounted
    read-only at its workdir destination while judge jobs read it; the destination itself stays empty. A
    follow-on build naming that path would build from an empty directory and fail reading a transformers
    config, so the request is refused up front with the receipted tree. Returns "" for any other source.
    """
    if not source:
        return ""
    tree = Path(source)
    try:
        rel = tree.relative_to(host.workdir).as_posix()
    except ValueError:  # outside the workdir: the original, a baseline, a bind or a Hub id
        return ""
    try:
        if not tree.is_dir() or any(tree.iterdir()):
            return ""
    except OSError:
        return ""
    from .prune import builds, pruned_at

    live = next((b for b in builds(host) if b.destination == rel), None)
    if live is not None:
        return (
            f"{rel} is a custody placeholder, not a model directory: build {live.job}'s tree is host-only "
            f"at {live.tree}. Build from the original model or a configured baseline"
        )
    if pruned_at(host, rel) is not None:
        return (
            f"{rel} is a custody placeholder, not a model directory: its build was pruned from custody. "
            "Build from the original model or a configured baseline"
        )
    return ""


def signature(source: str) -> str:
    """Content fingerprint of a local model directory: every tensor's bytes, under its name, dtype and shape.

    `capture` takes it on the exact, resolved directory the build was given (the host's `forced` override),
    so it names the bytes that directory actually held. `checks.tensor_fingerprint` reads the loaded tensors
    whole, so a changed weight value changes it (a changed file header alone does not); no tensor is skipped.
    A Hub id is pinned by its revision and gets no signature; so does an unreadable directory, whose builds
    are then never eligible.
    """
    try:
        if not source or not Path(source).is_dir():
            return ""
        from .checks import CheckpointError, tensor_fingerprint

        return tensor_fingerprint(Path(source), [])
    except (CheckpointError, ImportError, OSError, ValueError):
        return ""


def command(sched: Scheduler, job: Job, cards: list[int]) -> list[str]:
    """Launch the trusted host runner, carrying only host-authored settings."""
    private = Path(job.build_private or "")
    spec = json.loads((private / "request.json").read_text())
    return launch(
        sched,
        private,
        spec,
        job_id=job.id,
        home=Path(job.root or job.cwd),
        env=job.env,
        cards=cards,
        gpus=sched.cfg.gpus,
        readable=[sched.dir / job.id, *sched.cfg.readable],
    )


def launch(
    sched: Scheduler,
    private: Path,
    spec: dict[str, Any],
    *,
    job_id: str,
    home: Path,
    env: dict[str, str],
    cards: list[int],
    gpus: bool,
    readable: list[Path],
) -> list[str]:
    """Write the trusted runner's launch file for `private` and return its argv.

    `readable` is what the build sandbox sees read-only; everything the runner needs beyond it is
    host-authored here (the binds, the hidden trees, the environment), so nothing the agent wrote reaches
    the build.
    """
    spec.update(
        isolation={"level": sched.iso.level},  # the run's level governs the build (see `main`)
        job_id=job_id,
        readable=list(map(str, expand_links(readable))),
        binds={str(k): str(v) for k, v in sched.cfg.binds.items()},
        hide=list(map(str, sched.cfg.hide)),
        env={**sandbox_env(home), **sched.cfg.agent_job_env, **env},
        run_dir=str(sched.dir.parent),
        host_written=list(map(str, [*sched.cfg.host_written, sched.dir, private])),
        cards=cards if sched.backend.exact_cards else None,
        gpus=gpus,
    )
    # The trusted sandbox mounts only its scratch, not the receipts or other jobs' output.
    roots = [private.parent]
    if private.parent.is_symlink():
        roots.append(private.resolve().parent.parent)
    spec["hide"] = [p for p in spec["hide"] if Path(p) not in roots]
    (private / "launch.json").write_text(json.dumps(spec))
    from .custody import mirror

    mirror(private, Path(spec["records"]) if "records" in spec else None, Path(spec["config"]))
    # -I removes the host's cwd/PYTHONPATH; only this installed package tree is added.
    source = str(Path(__file__).resolve().parents[2])
    bootstrap = "import sys; sys.path.insert(0, sys.argv[1]); from qda_agent.judge.build_receipt import main; main(sys.argv[2])"
    return [python(), "-I", "-c", bootstrap, source, str(private / "launch.json")]


def capture(private: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Move the finished output into the receipted tree, checking it first, and hash it there.

    The build process has ended and its output lives in the host-only private directory, so a rename keeps
    the same inodes: one physical copy of the weights, not two. The output is checked before anything moves
    (a link, special file or hardlink could have another writer).
    """
    source, frozen = private / "output", private / "frozen"
    if not source.is_dir() or source.is_symlink():
        raise ValueError("trusted build did not produce a directory")
    for path in walk(source):
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError("trusted build output contains a link or special file")
    if hardlink_in(source) is not None:
        raise ValueError("trusted build output contains a hardlink; its bytes may have another writer")
    if frozen.exists() or frozen.is_symlink():
        raise ValueError("trusted build custody already holds a frozen tree")
    source.rename(frozen)
    record = frozen / "build.json"
    if not record.is_file() or record.stat().st_size > 1 << 20:
        raise ValueError("trusted build output needs bounded build.json")
    # Admission still validates format, metadata policy and contamination on the submitted copy.
    receipt = {
        "job_id": spec["job_id"],
        "destination": spec["destination"],
        "workdir": spec["root"],
        "config_sha256": file_sha256(Path(spec["config"])),
        "runner_sha256": file_sha256(Path(__file__)),
        "overrides": spec["overrides"],
        "source": spec.get("source", ""),
        "source_sha256": signature(spec.get("source", "")),
        "build_sha256": file_sha256(record),
        "tree_sha256": seal(frozen),
    }
    (private / "receipt.json").write_text(json.dumps(receipt))
    from .custody import mirror

    mirror(private, Path(spec["records"]) if "records" in spec else None, Path(spec["config"]))
    return receipt


def publish(private: Path) -> None:
    """Move the receipted tree into the judge's host-only custody, never into the agent's workdir.

    The tree becomes `private/published` (a directory, the "published" marker) with directories at 0555 and
    files at 0444: the sandboxes mount it read-only at the build's destination, and nothing in the workdir is
    the receipted copy any more. Only the host writes here, and no sandbox binds this directory.
    """
    frozen, published = private / "frozen", private / "published"
    frozen.rename(published)
    for d, dirs, files in os.walk(published):
        for name in files:
            Path(d, name).chmod(0o444)
        for name in dirs:
            Path(d, name).chmod(0o555)
    published.chmod(0o555)


def main(settings: str) -> None:
    """Build in confinement, receipt a private copy, then publish; failures mint no usable receipt."""
    spec = json.loads(Path(settings).read_text())
    private = Path(spec["private"])
    scratch = private / "scratch"
    scratch.mkdir()
    # Only this job sees its private scratch, at an otherwise unbound host path.
    config = scratch / "config.yaml"
    shutil.copyfile(spec["config"], config)
    output = scratch / "output"
    paths = spec["paths"]
    overrides = [*spec["overrides"], *spec.get("forced", [])]
    code = (
        "import sys; sys.path[:0]=__import__('json').loads(sys.argv[1]); "
        "import yaml; from qda_quant.config import BuildConfig; "
        "cfg=BuildConfig.from_dict(yaml.safe_load(open(sys.argv[2])), "
        "__import__('json').loads(sys.argv[3]), base=sys.argv[4]); "
        "cfg.out=sys.argv[5]; cfg.build()"
    )
    build = [
        spec["interpreter"],
        "-I",
        "-c",
        code,
        json.dumps(paths),
        str(config),
        json.dumps(overrides),
        spec["base"],
        str(output),
    ]
    spec["env"].pop("PYTHONPATH", None)  # -I plus validated explicit source paths is authoritative
    cards = spec["cards"]
    if cards is None:
        cards = visible_cards(os.environ.get("CUDA_VISIBLE_DEVICES"))
    env = spec["env"] | {"CUDA_VISIBLE_DEVICES": ",".join(card_uuids(cards)) if spec["gpus"] else ""}
    # The build is the agent's own config built by the trusted runner: the run's isolation level governs it
    # like every job the run starts. At `none` it is a plain process (no bwrap needed); the sandboxed levels
    # give it an empty root and only its inputs, and the runner starts what the level built.
    launch = make_for_build(spec["isolation"]["level"]).job(
        JobSpec(
            argv=build,
            cwd=Path(spec["cwd"]),
            root=Path(spec["root"]),
            cards=cards,
            env=env,
            readable=[Path(p) for p in spec["readable"]],
            gpu=spec["gpus"],
            binds={Path(k): Path(v) for k, v in spec["binds"].items()},
            host_written=[Path(p) for p in spec["host_written"]],
            run_dir=Path(spec["run_dir"]) if spec["run_dir"] else None,
            hide=[Path(p) for p in spec["hide"]],
            extra_writable=[scratch],
            empty_root=True,
        )
    )
    try:
        # Under a reaper: a GPU sandbox may share the pid namespace and bwrap starts the build in a session
        # of its own, so only the reaper's sweep ends every build process. No leftover build child may keep
        # a writable descriptor while the host captures the receipt.
        rc = reaper.run(
            launch.argv,
            env=dict(launch.env) if launch.env is not None else None,
            pass_fds=launch.pass_fds,
            grace_s=BUILD_GRACE_S,
        )
        if rc:
            raise subprocess.CalledProcessError(rc, launch.argv)
    finally:
        for fd in launch.pass_fds:
            with contextlib.suppress(OSError):
                os.close(fd)
    output.rename(private / "output")
    capture(private, spec)
    publish(private)
    remove_tree(private / "scratch")
    remove_tree(private / "output")


#: Why a submitted file set cannot win: it is no one build's recorded output. It is still scored.
UNMATCHED = (
    "not eligible: no trusted build recorded exactly these files (hand-edited, or mixed from two builds)"
)
#: Why a build's output cannot win: its source was neither the original model nor a configured baseline.
NON_ORIGINAL = "not eligible: built from a non-original source"
#: Why a build's output cannot win: its trusted receipt marks it as a layer-cache screening build.
SCREENING_BUILD = "screening-only (layer cache) build: rebuild whole"


def screening_build(frozen: Path, receipt: dict[str, Any]) -> str:
    """Why `frozen` may not win ("" when it may): the trusted `build.json` marks a screening build.

    `qda build`'s layer cache writes `layer_cache[<segment>].screening_only = true` whenever a build reused
    any layer (an exact hit included); the mark is read only from the `build.json` the receipt hashes
    (`build_sha256`), never from a copy the agent can write: a receipt that is absent, or a `build.json` that
    no longer hashes to the receipt's, is no evidence either way.
    """
    if not receipt:
        return ""
    try:
        if receipt["build_sha256"] != file_sha256(frozen / "build.json"):
            return ""
        metadata = json.loads((frozen / "build.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    segments = metadata.get("layer_cache") if isinstance(metadata, dict) else None
    if isinstance(segments, dict):  # a segment-keyed mapping, should the writer name them so
        segments = list(segments.values())
    marked = isinstance(segments, list) and any(
        isinstance(segment, dict) and segment.get("screening_only") is True for segment in segments
    )
    return SCREENING_BUILD if marked else ""


def match(host: JudgeHost, frozen: Path, tree: str) -> tuple[dict[str, Any] | None, str]:
    """The successful trusted build whose recorded outputs are exactly `frozen`, and a reason when none is.

    The comparison is the sealed full-tree digest `tree` of the submitted copy against each build's recorded
    `tree_sha256`, so it takes seconds and never rebuilds. The destination plays no part: a file set equal to
    one build's outputs is that build's however it is named or moved. A tampered or mixed set matches no
    build; it is scored all the same, but `UNMATCHED` explains why no final rule may pick it.

    A build's source must be the original model or a configured baseline, by resolved path and by the
    content signature `capture` took of it; any other source (an agent-written copy, even byte-equal) makes
    its output `NON_ORIGINAL`: still scored and reported, never final. A publication a prune pass moved to
    `pruned/` (its weights cleaned at the floor or the cap) still matches: the digest says the submitted
    files are that recorded build, and the receipt stays beside the cleaned tree.
    """
    from .prune import PRUNED

    metadata = frozen / "build.json"
    if not metadata.is_file():
        return None, UNMATCHED
    configured = _signatures(host)
    other = None
    for job in host.sched.jobs.values():
        if job.build_private is None or job.rc != 0 or job.state.value != "done":
            continue
        private = Path(job.build_private)
        if not (private / "published").is_dir() and not (private / PRUNED).is_dir():
            continue
        try:
            receipt = json.loads((private / "receipt.json").read_text())
            matched = (
                receipt["job_id"] == job.id
                and receipt["tree_sha256"] == tree
                and receipt["build_sha256"] == file_sha256(metadata)
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # a missing or unreadable receipt is no build's output
        if not matched:
            continue
        if _original(configured, receipt):
            return receipt, ""
        other = other or receipt  # keep looking: another build may record the same tree from the original
    return (other, NON_ORIGINAL) if other is not None else (None, UNMATCHED)


def _original(configured: dict[str, str], receipt: dict[str, Any]) -> bool:
    """Whether a receipt's recorded source is the original or a baseline, by resolved path and fingerprint."""
    source, recorded = receipt.get("source"), receipt.get("source_sha256")
    if not isinstance(source, str) or not isinstance(recorded, str) or not source:
        return False
    return configured.get(source) == recorded  # a Hub id has no local fingerprint, and "" matches ""


def _signatures(host: JudgeHost) -> dict[str, str]:
    """Source identity of every model a build may read: resolved path (or Hub id) -> content fingerprint.

    The original model (`judge.model`), `judge.anchor` (the original under another name) and every configured
    baseline qualify. The sources are read from the settings at match time, so a baseline added after the host
    started counts, and each source is fingerprinted once (`host.source_signatures`): a submission compares
    against the cached value instead of re-reading the original's weights.
    """
    sources = [str(m) for m in (host.cfg.model, host.cfg.anchor) if m is not None and str(m)]
    sources += [str(b) for b in host.cfg.baselines]
    out: dict[str, str] = {}
    for model in sources:
        p = Path(model).expanduser()
        key = str(p.resolve()) if p.is_dir() else model
        if key not in host.source_signatures:
            host.source_signatures[key] = signature(key)
        out[key] = host.source_signatures[key]
    return out


#: The scheduler builds a receipted build's argv through this module, which owns the receipt and its runner.
register_build_command(command)


def published(run_dir: Path | None, workdir: Path | None) -> list[tuple[Path, Path]]:
    """`(destination, custody tree)` for every receipted build published for `workdir`.

    The trusted build keeps its receipted tree in the judge's host-only custody
    (`<run_dir>/judge/builds/<job>/published`), never in the agent's writable workdir; the same helper is
    used where the sandboxes are built and by the late binder, so it re-derives what to mount read-only
    from the receipts at start time (a job queued before a build finished still sees it). Nothing the agent
    can write is consulted: the custody is not bound into any sandbox, and a destination rebuilt more than
    once resolves to its newest publication. A missing or unreadable receipt is skipped.
    """
    if run_dir is None or workdir is None:
        return []
    root = Path(os.path.realpath(run_dir)) / "judge" / "builds"
    if not root.is_dir():
        return []
    newest: dict[Path, tuple[float, Path]] = {}
    for private in root.iterdir():
        published = private / "published"
        if private.is_symlink() or not private.is_dir() or published.is_symlink() or not published.is_dir():
            continue
        try:
            receipt = json.loads((private / "receipt.json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        destination = receipt.get("destination")
        recorded = receipt.get("workdir")
        if not isinstance(destination, str) or not destination:
            continue
        if isinstance(recorded, str) and os.path.realpath(recorded) != os.path.realpath(workdir):
            continue
        rel = Path(destination)
        if rel.is_absolute() or ".." in rel.parts or rel == Path():
            continue
        try:
            stamp = published.stat().st_mtime
        except OSError:
            continue
        path = Path(workdir) / rel
        if path not in newest or stamp > newest[path][0]:
            newest[path] = (stamp, published)
    return [(path, tree) for path, (_, tree) in sorted(newest.items())]


#: The sandbox asks the judge which receipted builds are published for a workdir (see `bwrap.mount_points`).
register_published(published)
