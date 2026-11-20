"""Track B acceptance: one run, one PASS/FAIL line per item.

    python tools/acceptance/accept.py --tree DIR --main-tree DIR --rig DIR --model DIR [options]

Each item prints the command it ran, the path of its evidence and PASS or FAIL with the reason. An item whose
code is not in the tree says so ("not landed") instead of crashing. The trees, the search rig and the model are
arguments: this file names no cluster host, machine path or session.

Items
  A1  a none-level search runs to FINAL.json with bwrap unreachable and no Slurm in the environment
  A2  the strict level's run-start decision equals the release commit's, sentence for sentence
  A3  a real tiny-model server job, stopped mid-run through the job backend, leaves no process behind
  A4  the repository's public-tree check passes on the tree
  A5  the CPU suite has no failure the release commit does not also have

Options
  --tree DIR            the candidate checkout to accept (default: the repository root)
  --main-tree DIR       a checkout of the release commit (A2, A5)
  --rig DIR             a search rig: a directory holding `search.yaml` and the `drive.py` that runs it (A1)
  --model DIR           a small model directory the serving engine can load (A3)
  --python PATH         the interpreter that runs the tree's code (default: this one)
  --serve-python PATH   the interpreter with the serving engine installed (A3; default: --python)
  --serve-pythonpath DIR  the PYTHONPATH the serving engine is imported from (A3; default: none)
  --refuse-under DIR    refuse any path argument or rig file under DIR (repeatable; no default)
  --probe-run-base DIR  where A2's probe run dirs go: not under a temporary directory, which a fenced run
                        refuses (default: ~/.qda-acceptance)
  --evidence DIR        where the logs go (default: a fresh directory under $TMPDIR)
  --only A1,A2          run only these items
  --skip A5             skip these items
  --port N              the port A3's server binds (default: one the kernel says is free)

Exit status: 0 when every item that ran is PASS or SKIP, 1 when one is FAIL.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

#: The variables a child of this harness never inherits: Slurm, session credentials, and TMPDIR (`test runs use
#: no TMPDIR of their own: a non-default one is treated as a directory the agent may write`).
DROPPED_PREFIXES = ("SLURM_", "CLAUDE", "ANTHROPIC_")
DROPPED_NAMES = frozenset({"OPENAI_API_KEY", "CODEX_API_KEY", "TMPDIR"})

#: The programs a none-level search must never execute: a logging shim stands in for each and exits 127.
FORBIDDEN_EXEC = ("bwrap", "squeue", "scontrol")

PACKAGES = ("qda-agent", "qda-kl", "qda-quant")
#: The suites A5 compares: every package's, so a change to one cannot pass on another package's results.
SUITES = tuple(f"packages/{name}/tests" for name in PACKAGES)

#: The variable that tags A3's job, so a process it leaves behind can be told from another session's.
ACCEPT_TAG = "QDA_ACCEPT_TAG"

ITEM_TITLES = {
    "A1": "none-level search: no bwrap, no Slurm, FINAL.json with its rule",
    "A2": "strict decisions equal the release commit's",
    "A3": "a stopped job leaves no process of its tree",
    "A4": "the repository's public-tree check",
    "A5": "the CPU suite: no failure the release commit does not have",
}


class Verdict(NamedTuple):
    """One item's answer: PASS or FAIL, why, where its evidence is, and the command that produced it."""

    verdict: str
    reason: str
    evidence: Path | None = None
    command: str = ""


class Ctx:
    """What every item needs: the trees, the rig, the model, the interpreters and the evidence directory."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.tree = Path(args.tree).resolve()
        self.main_tree = Path(args.main_tree).resolve() if args.main_tree else None
        self.rig = Path(args.rig).resolve() if args.rig else None
        self.model = Path(args.model).resolve() if args.model else None
        self.python = args.python
        self.serve_python = args.serve_python or args.python
        self.serve_pythonpath = args.serve_pythonpath or None
        self.refuse_under = refused_roots(args.refuse_under)
        self.probe_run_base = Path(args.probe_run_base).expanduser()
        self.evidence = Path(args.evidence).resolve()
        self.port = args.port
        self.started = time.monotonic()


def child_env(buf: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """The environment a child gets: this one, without Slurm or session credentials, plus `extra`."""
    env = {
        name: value
        for name, value in (buf if buf is not None else os.environ).items()
        if not name.startswith(DROPPED_PREFIXES) and name not in DROPPED_NAMES
    }
    env.update(extra)
    return env


def sources(tree: Path) -> str:
    """`PYTHONPATH` for `tree`: its three package source directories."""
    return os.pathsep.join(str(tree / "packages" / name / "src") for name in PACKAGES)


def refused_roots(values: list[str]) -> tuple[Path, ...]:
    """The `--refuse-under` directories, resolved once (a symlinked root is refused as itself)."""
    return tuple(Path(os.path.realpath(os.path.expanduser(value))) for value in values)


def refused_by(path: str, roots: tuple[Path, ...]) -> Path | None:
    """The root that holds `path` (resolved, so a symlink into it is refused too), if any."""
    if not path:
        return None
    real = os.path.realpath(os.path.expanduser(path))
    for root in roots:
        prefix = str(root) if str(root).endswith(os.sep) else str(root) + os.sep
        if real == str(root) or real.startswith(prefix):
            return root
    return None


def names_refused(text: str, roots: tuple[Path, ...]) -> Path | None:
    """The root that `text` mentions as a path, if any."""
    for root in roots:
        if re.search(re.escape(str(root)) + r"(?![A-Za-z0-9_.-])", text):
            return root
    return None


def call(command: list[str], *, log: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run `command`, its output to `log` (only the child's output); its exit status."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        return subprocess.run(  # noqa: S603 (the caller built argv)
            command,
            cwd=cwd,
            env=env or child_env(),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode


def shown(command: list[str]) -> str:
    """A command as the report prints it."""
    return " ".join(command)


def missing(tree: Path, *names: str) -> str:
    """The reason an item cannot run on `tree` ("" when its files are all there)."""
    absent = [
        name for name in names if not (tree / "packages" / "qda-agent" / "src" / "qda_agent" / name).is_file()
    ]
    return f"not landed: {', '.join(absent)} not in this tree" if absent else ""


def free_port() -> int:
    """A port the kernel says is free now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def alive(pid: int) -> bool:
    """Whether /proc still names `pid`."""
    return Path(f"/proc/{pid}").exists()


def zombie(pid: int) -> bool:
    """Whether `pid` is a zombie (it holds nothing; init reaps it)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


def children_of(root: int) -> dict[int, int]:
    """Every live process below `root`, by /proc parent links, as {pid: parent}."""
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        parents[int(entry.name)] = int(fields[1])
    found: dict[int, int] = {}
    stack = [pid for pid, parent in parents.items() if parent == root]
    while stack:
        pid = stack.pop()
        found[pid] = parents[pid]
        stack.extend(child for child, parent in parents.items() if parent == pid and child not in found)
    return found


def path_farm(into: Path) -> None:
    """A directory holding a link to every program of this PATH by name, except the forbidden three.

    A directory that holds `bwrap` holds `ls`, `git`, `nvidia-smi` and the rest too, so no directory is
    dropped: the three names alone are left out, and the shims take their place.
    """
    into.mkdir(parents=True)
    linked: set[str] = set()
    for directory in (Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry):
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in FORBIDDEN_EXEC or entry.name in linked:
                continue
            try:
                if entry.is_file() and os.access(entry, os.X_OK):
                    (into / entry.name).symlink_to(entry)
                    linked.add(entry.name)
            except OSError:
                continue


def carries(pid: int, tag: str) -> bool:
    """Whether `pid` was started with `ACCEPT_TAG` set to `tag` (its environment is read, not guessed)."""
    try:
        return f"{ACCEPT_TAG}={tag}".encode() in Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False


def our_orphans() -> set[int]:
    """The pids of this user whose parent is init: where a job's escaped processes reparent to."""
    uid, found = os.getuid(), set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == 1:
            found.add(int(entry.name))
    return found


# ----------------------------------------------------------------------------------------------- the items
def item_a1(ctx: Ctx) -> Verdict:
    """A search at the none level: bwrap unreachable, no Slurm, and FINAL.json names its rule."""
    if reason := missing(ctx.tree, "isolation/levels/none.py"):
        return Verdict("FAIL", reason)
    if ctx.rig is None:
        return Verdict("FAIL", "--rig is required (a directory with search.yaml and drive.py)")
    if not (ctx.rig / "drive.py").is_file():
        return Verdict("FAIL", f"{ctx.rig / 'drive.py'} is missing")
    for name in (
        "search.yaml",
        "drive.py",
    ):  # a rig that reads a refused root would pass for the wrong reason
        path = ctx.rig / name
        if path.is_file() and (root := names_refused(path.read_text(errors="replace"), ctx.refuse_under)):
            return Verdict(
                "FAIL", f"{path} names a path under --refuse-under {root}: the rig is not node-local"
            )
    shim, farm, exec_log = ctx.evidence / "a1-bin", ctx.evidence / "a1-path", ctx.evidence / "a1-exec.log"
    shim.mkdir(parents=True, exist_ok=True)
    for (
        name
    ) in FORBIDDEN_EXEC:  # a search that executes one records it; it then fails its job (127, not found)
        path = shim / name
        path.write_text(f'#!/bin/sh\nprintf "%s %s\\n" "{name}" "$*" >> "{exec_log}"\nexit 127\n')
        path.chmod(0o755)
    if farm.exists():
        shutil.rmtree(farm)
    path_farm(farm)
    if shutil.which("bwrap", path=str(farm)):
        return Verdict("FAIL", "a bwrap is still reachable on the farm")
    run_dir = ctx.evidence / "a1-run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    env = child_env(
        PATH=f"{shim}{os.pathsep}{farm}",
        PYTHONPATH=sources(ctx.tree),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    command = [ctx.python, str(ctx.rig / "drive.py"), f"run_dir={run_dir}"]
    log = ctx.evidence / "a1.log"
    rc = call(command, log=log, cwd=ctx.rig, env=env)
    if rc != 0:
        return Verdict("FAIL", f"the search exited {rc}", log, shown(command))
    final = run_dir / "FINAL.json"
    if not final.is_file():
        return Verdict("FAIL", f"the search wrote no {final}", log, shown(command))
    document = json.loads(final.read_text())
    if "chosen_by" not in document:
        return Verdict(
            "FAIL",
            "FINAL.json does not say which rule chose the result (not landed here)",
            final,
            shown(command),
        )
    executed = sorted(set(exec_log.read_text().split())) if exec_log.is_file() else []
    if executed:
        return Verdict("FAIL", f"the search executed {executed}", exec_log, shown(command))
    return Verdict(
        "PASS",
        f"FINAL.json by {document['chosen_by']!r}; no {', '.join(FORBIDDEN_EXEC)} exec",
        log,
        shown(command),
    )


def item_a2(ctx: Ctx) -> Verdict:
    """The strict level's run-start decision against the release commit's, on this node, sentence for sentence."""
    if ctx.main_tree is None:
        return Verdict("FAIL", "--main-tree is required (a checkout of the release commit)")
    if reason := missing(ctx.tree, "isolation/levels/strict.py"):
        return Verdict("FAIL", reason)
    probe = Path(__file__).with_name("strict_probe.py")
    ctx.probe_run_base.mkdir(parents=True, exist_ok=True)
    seen, commands = {}, []
    for mode, tree in (("main", ctx.main_tree), ("candidate", ctx.tree)):
        log = ctx.evidence / f"a2-{mode}.json"
        command = [
            ctx.python, str(probe), "--tree", str(tree), "--mode", mode,
            "--run-dir", str(ctx.probe_run_base / f"accept-{mode}"),
        ]  # fmt: skip
        commands.append(shown(command))
        rc = call(command, log=log, env=child_env(PYTHONPATH=sources(tree)))
        if rc != 0:
            return Verdict("FAIL", f"the {mode} probe exited {rc}", log, "\n".join(commands))
        seen[mode] = json.loads(log.read_text())
    main, candidate = seen["main"], seen["candidate"]
    if candidate.get("level") != "strict":  # else this compares today's wall with another level by accident
        return Verdict(
            "FAIL",
            f"the candidate probe ran level {candidate.get('level')!r}, not 'strict'",
            ctx.evidence / "a2-candidate.json",
            "\n".join(commands),
        )
    for key in ("pid_form_auto", "check", "hold_problem", "strict"):
        if main[key] != candidate[key]:
            return Verdict(
                "FAIL",
                f"{key} differs: release {main[key]!r} vs candidate {candidate[key]!r}",
                ctx.evidence / "a2-candidate.json",
                "\n".join(commands),
            )
    return Verdict(
        "PASS", f"strict, both: {main['strict']}", ctx.evidence / "a2-candidate.json", "\n".join(commands)
    )


def item_a3(ctx: Ctx) -> Verdict:
    """A tiny-model server through the job backend, stopped mid-run: no process of its tree survives."""
    if reason := missing(ctx.tree, "isolation/reaper.py"):
        return Verdict("FAIL", reason)
    if ctx.model is None:
        return Verdict("FAIL", "--model is required (a small model directory)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return Verdict("FAIL", "no card is visible: run this item under the card queue")
    sys.path.insert(0, str(ctx.tree / "packages" / "qda-agent" / "src"))
    from qda_agent.sched import LocalBackend  # noqa: PLC0415 (the tree under test, not an installed copy)

    backend = LocalBackend(kill_grace_s=10)
    port = ctx.port or free_port()
    command = [
        ctx.serve_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(ctx.model), "--served-model-name", "acceptance",
        "--host", "127.0.0.1", "--port", str(port),
        "--gpu-memory-utilization", "0.2", "--max-model-len", "1024", "--enforce-eager",
    ]  # fmt: skip
    shown_command = (f"PYTHONPATH={ctx.serve_pythonpath} " if ctx.serve_pythonpath else "") + shown(command)
    log = ctx.evidence / "a3-engine.log"
    tag = f"qda-accept-{os.getpid()}-{int(time.monotonic())}"  # this job's own mark, read back from /proc
    job_env = {ACCEPT_TAG: tag}
    if ctx.serve_pythonpath:  # the engine may be staged on this node instead of installed
        job_env["PYTHONPATH"] = ctx.serve_pythonpath
    before = our_orphans()
    handle = backend.launch(command, str(ctx.evidence), [], log, "a3", env=child_env(**job_env))
    try:
        ready, end = False, time.monotonic() + 420
        while time.monotonic() < end and backend.poll(handle) is None:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(0.5)
        if not ready:
            return Verdict("FAIL", "the server never answered", log, shown_command)
        # The backend's handle is the job's outer process (its reaper): count it and everything below it.
        tree = sorted({handle.popen.pid, *children_of(handle.popen.pid)})
        backend.stop(handle)
        survivors = [pid for pid in tree if alive(pid) and not zombie(pid)]
        # Other sessions of this account run on this node too: only a process that carries this job's tag is
        # this job's stray. An untagged new orphan is another session's and is reported, not counted.
        new = [pid for pid in our_orphans() - before if not zombie(pid)]
        strays = sorted(pid for pid in new if carries(pid, tag))
        untagged = sorted(pid for pid in new if pid not in strays)
        zombies = [pid for pid in tree if zombie(pid)]
        record = ctx.evidence / "a3-pids.json"
        record.write_text(
            json.dumps(
                {
                    "tree": tree,
                    "survivors": survivors,
                    "strays_with_this_jobs_tag": strays,
                    "new_orphans_other_sessions": untagged,
                    "zombies": zombies,
                },
                indent=1,
            )
        )
        if survivors or strays:
            return Verdict(
                "FAIL",
                f"{len(survivors)} of its tree and {len(strays)} tagged orphan(s) survived",
                record,
                shown_command,
            )
        return Verdict(
            "PASS",
            f"{len(tree)} processes stopped mid-run; {len(zombies)} zombie(s), reaped by init; "
            f"{len(untagged)} new orphan(s) of other sessions",
            record,
            shown_command,
        )
    finally:
        with contextlib.suppress(OSError, AssertionError):
            backend.stop(handle)


def item_a4(ctx: Ctx) -> Verdict:
    """The repository's own public-tree check on the tree under test."""
    check = ctx.tree / "tools" / "check_public_tree.py"
    if not check.is_file():
        return Verdict("FAIL", f"{check} is missing")
    log = ctx.evidence / "a4-public-tree.log"
    command = [sys.executable, str(check)]
    rc = call(command, log=log, cwd=ctx.tree)
    if rc != 0:
        return Verdict("FAIL", f"the public-tree check exited {rc}", log, shown(command))
    return Verdict("PASS", "no internal name, machine path or clock stamp found", log, shown(command))


def item_a5(ctx: Ctx) -> Verdict:
    """The CPU suite on both trees: every failure on the candidate must also fail on the release commit."""
    if ctx.main_tree is None:
        return Verdict("FAIL", "--main-tree is required (a checkout of the release commit)")
    failures, commands = {}, []
    for mode, tree in (("main", ctx.main_tree), ("candidate", ctx.tree)):
        base, log = ctx.evidence / f"a5-bt-{mode}", ctx.evidence / f"a5-{mode}.log"
        command = [
            ctx.python, "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider",
            f"--basetemp={base}", *SUITES,
        ]  # fmt: skip
        commands.append(shown(command))
        rc = call(
            command, log=log, cwd=tree, env=child_env(PYTHONPATH=sources(tree), CUDA_VISIBLE_DEVICES="")
        )
        with contextlib.suppress(OSError):
            for path in sorted(base.rglob("*"), reverse=True):
                path.chmod(0o700)
        shutil.rmtree(base, ignore_errors=True)
        lines = log.read_text().splitlines()
        failures[mode] = {line.split()[1] for line in lines if line.startswith(("FAILED ", "ERROR "))}
        if rc != 0 and not failures[mode]:
            return Verdict(
                "FAIL",
                f"the {mode} suite exited {rc} with no FAILED line (a collection error?)",
                log,
                "\n".join(commands),
            )
    new = sorted(failures["candidate"] - failures["main"])
    if new:
        return Verdict(
            "FAIL",
            f"{len(new)} failure(s) only on the candidate: {new[:5]}",
            ctx.evidence / "a5-candidate.log",
            "\n".join(commands),
        )
    return Verdict(
        "PASS",
        f"{len(failures['candidate'])} shared failure(s) over {', '.join(SUITES)}, none new",
        ctx.evidence / "a5-candidate.log",
        "\n".join(commands),
    )


ITEMS = {"A1": item_a1, "A2": item_a2, "A3": item_a3, "A4": item_a4, "A5": item_a5}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tree", default=str(Path(__file__).resolve().parents[2]), help="the candidate checkout"
    )
    parser.add_argument("--main-tree", default=None, help="a checkout of the release commit (A2, A5)")
    parser.add_argument("--rig", default=None, help="a directory holding search.yaml and drive.py (A1)")
    parser.add_argument("--model", default=None, help="a small model directory the engine can load (A3)")
    parser.add_argument("--python", default=sys.executable, help="the interpreter that runs the tree's code")
    parser.add_argument("--serve-python", default=None, help="the interpreter with the serving engine (A3)")
    parser.add_argument(
        "--serve-pythonpath", default=None, help="the PYTHONPATH the serving engine is imported from (A3)"
    )
    parser.add_argument(
        "--refuse-under",
        action="append",
        default=[],
        metavar="DIR",
        help="refuse any path argument or rig file under DIR (repeatable; no default)",
    )
    parser.add_argument(
        "--probe-run-base", default=str(Path.home() / ".qda-acceptance"), help="A2's run dirs"
    )
    parser.add_argument("--evidence", default=None, help="where the logs go")
    parser.add_argument("--only", default=None, help="run only these items, e.g. A1,A2")
    parser.add_argument("--skip", default="", help="skip these items, e.g. A5")
    parser.add_argument("--port", type=int, default=None, help="the port A3's server binds")
    args = parser.parse_args(argv)
    # A refused root (a shared filesystem, when one is named) is not to be read by this run: a symlink into
    # it is refused as itself, because the paths are resolved.
    looked_at = [
        ("--rig", args.rig),
        ("--model", args.model),
        ("--python", args.python),
        ("--serve-python", args.serve_python),
        *(("--serve-pythonpath", entry) for entry in (args.serve_pythonpath or "").split(os.pathsep)),
    ]
    roots = refused_roots(args.refuse_under)
    refused = [
        f"{label} {value} (under {root})"
        for label, value in looked_at
        if value and (root := refused_by(value, roots)) is not None
    ]
    if refused:
        print(f"REFUSED  {'; '.join(refused)}: paths under --refuse-under are not read")
        return 1
    if args.evidence is None:
        args.evidence = tempfile.mkdtemp(prefix="qda-acceptance-")
    Path(args.evidence).mkdir(parents=True, exist_ok=True)
    chosen = [name.strip().upper() for name in args.only.split(",")] if args.only else list(ITEMS)
    skipped = {name.strip().upper() for name in args.skip.split(",") if name.strip()}
    ctx = Ctx(args)
    print(f"tree:     {ctx.tree}")
    print(f"release:  {ctx.main_tree or '-'}")
    print(f"evidence: {ctx.evidence}\n")
    failed = 0
    for name in chosen:
        title, item = ITEM_TITLES.get(name), ITEMS.get(name)
        if item is None:
            print(f"{name:2}  SKIP  no such item\n")
            continue
        if name in skipped:
            print(f"{name:2}  SKIP  {title}\n")
            continue
        try:
            verdict = item(ctx)
        except Exception as error:  # noqa: BLE001 -- one broken item must not hide the others
            verdict = Verdict("FAIL", f"the item itself failed: {type(error).__name__}: {error}")
        failed += verdict.verdict == "FAIL"
        for line in verdict.command.splitlines():
            print(f"      $ {line}")
        print(f"{name:2}  {verdict.verdict}  {verdict.reason}")
        if verdict.evidence is not None:
            print(f"      evidence: {verdict.evidence}")
        print()
    print(
        f"{'FAIL' if failed else 'PASS'}  ({failed} item(s) failed, {time.monotonic() - ctx.started:.0f} s)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
