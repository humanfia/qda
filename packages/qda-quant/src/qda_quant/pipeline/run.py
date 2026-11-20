"""Plan and run pipeline stages sequentially or on reserved devices."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePath
from typing import Any, TypeVar

import yaml

from ..exit import USAGE, refuse
from ..output.record import qda_commit
from ..step import Refused
from .plan import (
    FILL,
    Pipeline,
    Stage,
    cfg_dump,
    chain_outs,
    config,
    expand,
    load,
    model_of,
    stage_paths,
    strings,
    with_data_dir,
    with_root,
)
from .record import (
    RECORD,
    build_key,
    command_key,
    file_sha,
    now,
    path_sha,
    read_record,
    recorded_data,
    resume,
    write_record,
)

SLOW_RESOLVE_S = 60.0
_T = TypeVar("_T")


def _ran_on(record: dict[str, Any], name: str) -> str:
    was = record["stages"].get(name, {}).get("commit", "not recorded")
    now = qda_commit()
    return "" if was == now else f" (it ran on commit {was}; this run is {now})"


def _run_command(p: Pipeline, s: Stage, argv: list[str]) -> dict[str, Any]:
    logs = p.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{s.name}.log"
    started = time.perf_counter()
    with log.open("wb") as f:
        f.write((f"# {now()} {shlex.join(argv)}\n").encode())
        f.flush()
        code = subprocess.run(argv, cwd=p.spec.parent, stdout=f, stderr=subprocess.STDOUT).returncode
    seconds = round(time.perf_counter() - started, 1)
    if code != 0:
        raise Refused(f"stage {s.name!r}: {shlex.join(argv)} exited {code}; its output is in {log}")
    missing = [str(path) for path in stage_paths(p, s) if not path.exists()]
    if missing:
        raise Refused(
            f"stage {s.name!r}: {shlex.join(argv)} exited 0 but wrote no {', '.join(missing)}; see {log}"
        )
    return {"seconds": seconds, "log": str(log)}


def deferred(p: Pipeline, s: Stage) -> str:
    """What this stage waits for, when it cannot be planned yet (else ``""``)."""
    if s.kind != "build":
        return ""
    src = str(Path(model_of(p, s)))
    if src in chain_outs(p) and not Path(src).exists():
        stage = next(x.name for x in p.stages if str(x.out) == src)
        return f"its source is stage {stage!r}'s checkpoint, not built yet"
    reads = set(strings(cfg_dump(config(p, s)), []))
    for d in p.data.values():
        if d.produced_by and not d.path.exists() and str(d.path) in reads:
            return f"its input data.{d.name} is stage {d.produced_by!r}'s output, not written yet"
    return ""


def _ready(p: Pipeline, s: Stage, record: dict[str, Any]) -> tuple[bool, str, Any]:
    outs = chain_outs(p)
    written = stage_paths(p, s)
    if s.kind == "command":
        argv = expand(p, s)
        if shutil.which(argv[0]) is None:
            return (
                False,
                f"stage {s.name!r}: {argv[0]!r} is not on PATH; it is the command this stage runs",
                argv,
            )
        skip, why = resume(s, record, command_key(p, s, argv, outs), written)
        if not skip and not why and (there := [str(path) for path in written if path.exists()]):
            return (
                False,
                (
                    f"stage {s.name!r}: {', '.join(there)} holds something and no record matches it; "
                    "a stage never replaces an output it did not write"
                ),
                argv,
            )
        return skip, why, argv
    cfg = config(p, s)
    skip, why = resume(s, record, build_key(cfg, s, outs), written)
    if skip or why:
        return skip, why, cfg
    # a build stage always writes its own checkpoint; `check` refuses one that writes nothing
    out = written[0]
    if out.exists() and any(out.iterdir()):
        return (
            False,
            (
                f"stage {s.name!r}: {out} holds something and no record matches it; "
                "a stage never replaces an output it did not write"
            ),
            cfg,
        )
    return False, "", cfg


def plan(p: Pipeline) -> int:
    """Print each stage's plan on stdout; a chain or a stage that does not plan is refused on stderr.

    Returns:
        0 when every stage plans, else the most serious refusal's status (`qda_quant.exit`): a stage that
        needs something this machine does not have (no such card, not enough free memory) is 3, and 2 is a
        chain or an input that has to change.
    """
    code = 0
    try:
        p.check_out()
        p.check_data()
    except Refused as e:
        return refuse(f"pipeline {p.name}: not planned: {e}", e.fix or None, code=e.code)
    record = read_record(p.root)
    print(f"pipeline {p.name}: {len(p.stages)} stage(s) -> {p.root}")
    print(f"recipe {p.recipe}")
    for d in p.data.values():
        if not d.path.exists():
            print(f"data {d.name}: {d.path} (stage {d.produced_by!r} writes it)")
            continue
        state = path_sha(d.path)
        print(f"data {d.name}: {d.path} sha256 {state['sha256'][:12]}" + (" pinned" if d.sha256 else ""))
    refused: list[tuple[str, str, str]] = []  # (stage, why, fix) of each stage that does not plan
    for s in p.stages:
        head = f"stage {s.name}: {s.kind}"
        try:
            skip, why, work = _ready(p, s, record)
            if why:
                print(f"{head}: REFUSED: {why}")
                refused.append((s.name, why, ""))
                code = max(code, USAGE)  # `_ready` answers in words: the chain or its data has to change
            elif skip:
                print(f"{head} -> {s.out}: recorded, would skip")
            elif s.kind == "command":
                wrote = ", ".join(str(path) for path in stage_paths(p, s))
                print(f"{head} -> {wrote}: WOULD RUN {shlex.join(work)}")
            else:
                assert s.steps is not None
                prev = deferred(p, s)
                if prev:
                    print(
                        f"{head} {s.steps[0]}:{s.steps[1]} -> {s.out}: DEFERRED, {prev}; its config validates"
                    )
                else:
                    pl = work.model_copy(update={"out": None}).plan()
                    print(
                        f"{head} {s.steps[0]}:{s.steps[1]} model {model_of(p, s)} -> {s.out}: PLANNED "
                        f"{len(pl.graph.layers)} layer(s), {pl.fmt}, {len(pl.devices)} card(s), "
                        f"{[k for k, _ in pl.segments]}, out ~{pl.output_bytes / 2**30:.1f} GiB"
                    )
        except Refused as e:
            print(f"{head}: REFUSED: {e}")
            refused.append((s.name, str(e), e.fix))
            code = max(code, e.code)
    if not refused:
        print("PLANNED: every stage plans")
        return code
    return refuse(
        f"pipeline {p.name}: {len(refused)} stage(s) do not plan: "
        + "; ".join(f"stage {name}: {why}" for name, why, _ in refused),
        (refused[0][2] or None) if len(refused) == 1 else None,  # one stage's fix, never another's
        code=code,
    )


def _reads(p: Pipeline, s: Stage) -> set[str]:
    if s.kind == "build":
        return set(strings(cfg_dump(config(p, s)), []))
    return {
        str(p.data[m.group(2)].path)
        for a in s.command
        for m in FILL.finditer(a)
        if m.group(2) in p.data and p.data[m.group(2)].produced_by != s.name
    }


def _deps(p: Pipeline, s: Stage) -> list[str]:
    where = {str(path): st.name for st in p.stages for path in stage_paths(p, st)}
    reads = _reads(p, s)
    out: list[str] = []

    def add(name: str | None) -> None:
        if name and name != s.name and name not in out:
            out.append(name)

    for d in p.data.values():
        if d.produced_by and str(d.path) in reads:
            add(d.produced_by)
    add(where.get(model_of(p, s)))
    return out


def _kd(p: Pipeline, s: Stage) -> bool:
    if s.kind != "build" or s.steps is None:
        return False
    steps = (p.recipe_yaml.get("steps") or [])[s.steps[0] : s.steps[1]]
    return any(isinstance(step, dict) and "scale_kd" in step for step in steps)


def _reservation() -> list[str]:
    return [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]


def cards_of(spec: str | None) -> list[str] | None:
    """Resolve a device subset within this process's CUDA reservation."""
    if not spec:
        return None
    if spec == "auto":
        have = _reservation()
        if not have:
            raise Refused(
                "--devices auto: CUDA_VISIBLE_DEVICES names no card. A pipeline allocates from its own "
                "reservation only; it never reads the node's card table. Run it under the reservation that "
                "holds its cards, or name them: --devices 4,5,6,7"
            )
        return have
    have = _reservation()
    named = [t.strip() for t in spec.split(",") if t.strip()]
    if not have:
        raise Refused(
            f"--devices {spec}: CUDA_VISIBLE_DEVICES names no card, so nothing is reserved; a pipeline "
            "allocates from its own reservation only"
        )
    bad = [t for t in named if t not in have]
    if bad:
        raise Refused(
            f"--devices {spec}: {', '.join(bad)} not in the reservation (CUDA_VISIBLE_DEVICES="
            f"{','.join(have)}); a pipeline never starts a card it was not given"
        )
    return named


def _launch(
    p: Pipeline, s: Stage, work: Any, cards: list[str], logs: Path
) -> tuple[subprocess.Popen[bytes], Any]:
    logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f"{s.name}.log").open("wb")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(cards)
    if pp := env.get("PYTHONPATH"):
        # the child runs from the spec's directory: a relative entry that named this checkout here would name
        # nothing there, and the child would import whatever qda_quant is installed instead of the one running
        env["PYTHONPATH"] = os.pathsep.join(
            x if not x or Path(x).is_absolute() else str((Path.cwd() / x).resolve())
            for x in pp.split(os.pathsep)
        )
    if s.kind == "build":
        cfg_dir = p.root / "stages"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / f"{s.name}.yaml"
        path.write_text(yaml.safe_dump(cfg_dump(work) | {"out": str(s.out)}, sort_keys=False))
        argv = [sys.executable, "-m", "qda_quant.cli", "build", str(path)]
    else:
        argv = list(work)
    log.write((f"# {now()} cards {','.join(cards)} {shlex.join(argv)}\n").encode())
    log.flush()
    proc = subprocess.Popen(argv, cwd=p.spec.parent, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def _resolving(thread: int) -> str:
    frame = sys._current_frames().get(thread)
    while frame is not None:
        if isinstance(here := frame.f_locals.get("self"), PurePath):
            return str(here)
        frame = frame.f_back
    return ""


def _watched(what: str, fn: Callable[[], _T]) -> _T:
    caller = threading.get_ident()
    t0 = time.perf_counter()
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(SLOW_RESOLVE_S):
            at = _resolving(caller)
            print(
                f"  {what}: resolving its paths has not returned after {time.perf_counter() - t0:.0f} s"
                f"{f' (at {at})' if at else ''}; no finished stage is recorded until it does",
                flush=True,
            )

    threading.Thread(target=watch, name="qda-pipeline-watch", daemon=True).start()
    try:
        return fn()
    finally:
        stop.set()


def _run_parallel(p: Pipeline, record: dict[str, Any], cards: list[str]) -> int:
    logs = p.root / "logs"
    done = set(record["stages"])
    pending = list(p.stages)
    deps = {s.name: _watched(f"stage {s.name!r}", partial(_deps, p, s)) for s in pending}
    # a stage's config or argv once `_ready` has passed it, while it waits for a card
    planned: dict[str, Any] = {}
    running: dict[str, tuple[subprocess.Popen[bytes], Any, list[str], Any, float]] = {}
    free = list(cards)
    started = len(done)
    try:
        while pending or running:
            for name in list(running):
                proc, log, got, work, t0 = running[name]
                if proc.poll() is None:
                    continue
                if proc.returncode != 0:
                    raise Refused(
                        f"stage {name!r}: exited {proc.returncode}; its output is in {logs / (name + '.log')}"
                    )
                s = next(x for x in p.stages if x.name == name)
                missing = [str(path) for path in stage_paths(p, s) if not path.exists()]
                if missing:
                    raise Refused(
                        f"stage {name!r}: exited 0 but wrote no {', '.join(missing)}; "
                        f"see {logs / (name + '.log')}"
                    )
                seconds = round(time.perf_counter() - t0, 1)
                log.close()
                if s.kind == "build":
                    assert s.out is not None
                    stage = {
                        "kind": "build",
                        "key": build_key(work, s, chain_outs(p)),
                        "out": str(s.out),
                        "output": path_sha(s.out),
                        "cards": got,
                        "seconds": seconds,
                        "log": str(logs / (name + ".log")),
                    }
                else:
                    stage = {
                        "kind": "command",
                        "key": command_key(p, s, work, chain_outs(p)),
                        "out": str(s.out) if s.out is not None else None,
                        "argv": work,
                        "cards": got,
                        "seconds": seconds,
                        "log": str(logs / (name + ".log")),
                    }
                stage["outputs"] = {str(path): path_sha(path) for path in stage_paths(p, s)}
                stage["finished"] = now()
                stage["commit"] = qda_commit()
                record["stages"][name] = stage
                record["history"].append(
                    {
                        "stage": name,
                        "finished": stage["finished"],
                        "seconds": stage["seconds"],
                        "commit": stage["commit"],
                    }
                )
                record["updated"] = stage["finished"]
                record["data"] = recorded_data(p)
                write_record(p, record)
                wrote = ", ".join(str(path) for path in stage_paths(p, s))
                print(f"  done in {stage['seconds']} s on card(s) {','.join(got)} -> {wrote}", flush=True)
                free.extend(got)
                done.add(name)
                del running[name]
            for s in list(pending):
                if any(dep not in done for dep in deps[s.name]):
                    continue
                if s.name not in planned:
                    skip, why, work = _watched(f"stage {s.name!r}", partial(_ready, p, s, record))
                    if why:
                        raise Refused(why)
                    if skip:
                        print(
                            f"  skipped {s.name}: the record and {s.out} match{_ran_on(record, s.name)}",
                            flush=True,
                        )
                        done.add(s.name)
                        pending.remove(s)
                        continue
                    planned[s.name] = work
                work = planned[s.name]
                want = 2 if _kd(p, s) else 1
                if not free:
                    continue
                got = free[:want]
                del free[: len(got)]
                started += 1
                print(
                    f"[{started}/{len(p.stages)}] {s.name}: {s.kind} on card(s) {','.join(got)}", flush=True
                )
                proc, log = _launch(p, s, work, got, logs)
                running[s.name] = (proc, log, got, work, time.perf_counter())
                pending.remove(s)
            if not running:
                if pending:
                    waits = {x.name: [d for d in deps[x.name] if d not in done] for x in pending}
                    stalling = next((x for x in pending if waits[x.name]), pending[0])
                    raise Refused(
                        f"stage {stalling.name!r}: nothing is running and it cannot start "
                        f"({'waits for ' + ', '.join(waits[stalling.name]) if waits[stalling.name] else 'no card is free'}"
                        f"); its reserved cards are {', '.join(cards)}"
                    )
                break
            if running and all(proc.poll() is None for proc, *_ in running.values()):
                time.sleep(1.0)
    finally:
        for proc, log, *_ in running.values():
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
    print(f"pipeline {p.name}: {len(p.stages)} stage(s) complete; record {p.root / RECORD}")
    return 0


def run(p: Pipeline, devices: str | None = None) -> int:
    """Plan and run the chain, resuming recorded stages; optionally schedule on reserved cards."""
    cards = cards_of(devices)
    p.check_out()
    p.check_data()
    record = read_record(p.root)
    record.update(
        {
            "name": p.name,
            "out": str(p.root),
            "updated": now(),
            "spec": {"path": str(p.spec), "sha256": file_sha(p.spec)["sha256"]},
            "recipe": {"path": str(p.recipe), "sha256": file_sha(p.recipe)["sha256"]},
            "data": recorded_data(p),
        }
    )
    # plan every stage before any work: a chain that cannot finish is not started
    for s in p.stages:
        skip, why, work = _ready(p, s, record)
        if why:
            raise Refused(why)
        if not skip and s.kind == "build" and not deferred(p, s):
            work.model_copy(update={"out": None}).plan()
    if cards is not None:
        return _run_parallel(p, record, cards)
    for i, s in enumerate(p.stages, 1):
        print(f"[{i}/{len(p.stages)}] {s.name}: {s.kind}", flush=True)
        skip, why, work = _ready(p, s, record)
        if why:
            raise Refused(why)
        if skip:
            print(f"  skipped: the record and {s.out} match{_ran_on(record, s.name)}", flush=True)
            continue
        if s.kind == "build":
            t0 = time.perf_counter()
            done = work.build(log=lambda message: print(f"  {message}", flush=True))
            stage = {
                "kind": "build",
                "key": build_key(work, s, chain_outs(p)),
                "out": str(done),
                "output": path_sha(Path(done)),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        else:
            stage = {
                "kind": "command",
                "key": command_key(p, s, work, chain_outs(p)),
                "out": str(s.out) if s.out is not None else None,
                "argv": work,
                **_run_command(p, s, work),
            }
        stage["outputs"] = {str(path): path_sha(path) for path in stage_paths(p, s)}
        stage["finished"] = now()
        stage["commit"] = qda_commit()
        record["stages"][s.name] = stage
        record["history"].append(
            {
                "stage": s.name,
                "finished": stage["finished"],
                "seconds": stage["seconds"],
                "commit": stage["commit"],
            }
        )
        record["updated"] = stage["finished"]
        record["data"] = recorded_data(
            p
        )  # a command stage may have written a data entry; a pin is checked there
        write_record(p, record)
        wrote = ", ".join(str(path) for path in stage_paths(p, s))
        print(f"  done in {stage['seconds']} s -> {wrote}", flush=True)
    print(f"pipeline {p.name}: {len(p.stages)} stage(s) complete; record {p.root / RECORD}")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the pipeline subcommand."""
    p = sub.add_parser("pipeline", help="run a staged spec to a checkpoint: plan every stage, hash, resume")
    q = p.add_subparsers(dest="action", required=True)
    one = q.add_parser("plan", help="plan every stage on CPU and print the chain; builds nothing")
    one.add_argument("spec", help="pipeline spec YAML")
    one.add_argument("--out", default=None, help="write the stages under this directory, not the spec's")
    one.add_argument("--data-dir", default=None, help="read and write relative data entries under this one")
    one.add_argument("--json", action="store_true", help="also print the chain as JSON")
    one.set_defaults(run=_run_plan)
    two = q.add_parser("build", help="plan every stage, then run the chain, skipping recorded stages")
    two.add_argument("spec", help="pipeline spec YAML")
    two.add_argument("--out", default=None, help="write the stages under this directory, not the spec's")
    two.add_argument("--data-dir", default=None, help="read and write relative data entries under this one")
    two.add_argument(
        "--devices",
        default=None,
        metavar="auto|CARD,CARD",
        help="run independent stages concurrently on the cards a reservation gave this process "
        "(CUDA_VISIBLE_DEVICES); 'auto' is every reserved card, and no other card is ever used",
    )
    two.set_defaults(run=_run_build)


def _loaded(a: argparse.Namespace) -> Pipeline:
    p = load(a.spec)
    if getattr(a, "data_dir", None):
        p = with_data_dir(p, Path(a.data_dir))
    return with_root(p, Path(a.out)) if a.out else p


def _run_plan(a: argparse.Namespace) -> int:
    try:
        p = _loaded(a)
        code = plan(p)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    if a.json:
        record = read_record(p.root)
        print(
            json.dumps(
                {
                    "stages": {
                        s.name: {
                            "kind": s.kind,
                            "out": str(s.out),
                            "steps": list(s.steps) if s.steps else None,
                        }
                        for s in p.stages
                    },
                    "recorded": sorted(record["stages"]),
                },
                indent=1,
                default=str,
            )
        )
    return code


def _run_build(a: argparse.Namespace) -> int:
    try:
        return run(_loaded(a), getattr(a, "devices", None))
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
