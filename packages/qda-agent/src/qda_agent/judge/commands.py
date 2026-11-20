"""The judge's commands: `qda`, `bpw`, `kl`, `gencheck` and `handin`, each `function(host, argv, cwd) -> Reply`.

`kl`, `gencheck` and `handin` take a checkpoint in (`intake.take`) and queue a judge job for it; `qda`
queues a build or runs any other subcommand in the agent's sandbox; `bpw` prices a checkpoint or a recipe on
the host. What each command does is in this module and `host.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qda_quant.checks import compiler_env

from ..isolation.api import JobSpec, Launch
from ..isolation.paths import job_env
from ..rpc import Reply
from ..sched import Job, Kind, Refused, State, report
from ..sched.queue import FROZEN
from ..sched.snapshot import SnapshotError, remove_tree
from .build_receipt import screening_build
from .checks import MAX_HEADER, CheckpointError, budget
from .cli import Parser, Usage, duration, names
from .freeze import INDEX, tree_size
from .intake import has_routed_experts, take
from .jobs import gencheck_expected_s, queue
from .ledger import neutral, utc

if TYPE_CHECKING:
    from qda_quant.format import Format

    from .host import JudgeHost

#: The longest a `qda` subcommand other than `build` may run on the host.
QDA_TIMEOUT_S = 120.0
QDA_BUILD_USAGE = (
    "usage: qda [--cards N] [--expected DUR] [--name X] [--after J] build CONFIG.yaml [--set k=v ...]\n"
)


def rest(argv: list[str]) -> list[str]:
    """`argv` without a leading `--` (the end of the options: the next argument is the subcommand)."""
    return argv[1:] if argv[:1] == ["--"] else argv


def qda_args(host: JudgeHost, argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """The agent's `qda` options (`--cards`, ...) and the rest, the subcommand first.

    One `--` that ends the options is dropped, so the rest is what the gate checks and what runs.
    """
    p = Parser(prog="qda", description="qda_quant; `build` is queued on the run's cards.", add_help=False)
    job_options(p, default_s=host.cfg.build_expected_s)
    if host.cfg.full_card_moe:
        p.set_defaults(cards=len(host.sched.cfg.cards))
    ns, more = p.parse_known_args(argv)
    return ns, rest(more)


def qda(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`qda build CONFIG` is queued on the run's cards; any other subcommand runs now, without a GPU."""
    ns, args = qda_args(host, argv)
    if args[:1] == ["build"]:
        from .prune import disk_lock

        with disk_lock(host):
            return _build(host, ns, args, cwd)
    cmd, env = [*host.qda_program(), *args], {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    launch = Launch(cmd, env, ())
    if host.sched.cfg.job_sandbox == "bwrap":  # the agent's argv: the same sandbox as its jobs, no GPU
        inner = {**job_env(host.workdir), "CUDA_VISIBLE_DEVICES": ""}  # HOME=workdir inside only
        inner.update(host.cfg.agent_qda_env or host.cfg.qda_env)
        inner.update(host.cfg.contamination_env())
        if host.qda_path:
            inner["PATH"] = host.qda_path
        # The same last-moment check and descriptor bind as every job (a link the agent swapped since the
        # config was checked must not carry the judge's files into the agent's qda verb), through the level.
        try:
            launch = host.sched.iso.job(
                JobSpec(
                    argv=cmd,
                    cwd=cwd,
                    root=host.workdir,
                    env=inner,
                    readable=list(host.sched.cfg.readable),
                    binds=dict(host.sched.cfg.binds),
                    host_written=[*host.sched.cfg.host_written, host.sched.dir],
                    run_dir=host.sched.cfg.run_dir,
                )
            )
        except ValueError as e:  # a hide covering what the verb needs (e.g. a readable source)
            return Reply(1, "", f"qda: {e}\n")
    try:
        r = subprocess.run(
            list(launch.argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=QDA_TIMEOUT_S,
            env=None if launch.env is None else dict(launch.env),
            pass_fds=tuple(launch.pass_fds),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Reply(1, "", f"qda: {type(e).__name__}\n")
    finally:
        for fd in launch.pass_fds:
            os.close(fd)
    return Reply(r.returncode, r.stdout, r.stderr)


def _build(host: JudgeHost, ns: argparse.Namespace, args: list[str], cwd: Path) -> Reply:
    """Admit and record a build while the whole-run disk lock is held."""
    if len(args) < 2 or args[1].startswith("-"):
        raise Usage(2, QDA_BUILD_USAGE)
    if host.cfg.full_card_moe and ns.cards != len(host.sched.cfg.cards):
        raise Refused(f"full_card_moe builds must use all {len(host.sched.cfg.cards)} cards")
    if host.cfg.full_card_moe:
        if host.full_card_build is not None:
            previous = host.sched.get(host.full_card_build, origin="agent")
            if previous.state is State.failed or previous.state is State.cancelled:
                host.full_card_build = None
            else:
                raise Refused(f"full_card_moe: score build {host.full_card_build} before another build")
        active = [
            job.name
            for job in host.sched.list(origin="agent")
            if not job.state.terminal and job.kind is Kind.build
        ]
        if active:
            raise Refused(f"full_card_moe is serialized: build {active[0]} is still active")
        scored = [
            job.name
            for job in host.sched.list(origin="judge")
            if not job.state.terminal and job.kind is Kind.score
        ]
        if scored:
            raise Refused(f"full_card_moe is serialized: score {scored[0]} is still active")
    private, expected, forced = None, 0, []
    if host.cfg.require_build_receipt:
        from .build_receipt import setup

        try:
            private, spec = setup(host, cwd, args[1:])
        except (ValueError, OSError) as error:
            raise Refused(str(error)) from None
        (private / "request.json").write_text(json.dumps(spec))
        from .custody import mirror

        mirror(private, Path(spec["records"]) if "records" in spec else None, Path(spec["config"]))
    else:  # a local build writes into work/: reserve its estimate against the run's disk cap
        from .build_receipt import local_build
        from .prune import check_room

        expected, forced = local_build(host, cwd, args[1], args[2:])
        check_room(host, expected)
    try:
        job = host.sched.submit(
            # a local GGUF build's host overrides (the run's template) come after the agent's own
            [
                *host.qda_program(),
                "build",
                spec["config"] if private else FROZEN,
                *args[2:],
                *[a for f in forced for a in ("--set", f)],
            ],
            str(cwd),
            freeze=None if private else args[1],
            expected_s=ns.expected,
            ncards=ns.cards,
            name=ns.name,
            after=ns.after,
            kind=Kind.build,
            origin="agent",
            root=host.workdir,
            env={
                # the compilers doctor's `C compiler` line probed, by absolute path: the job's PATH is the
                # run's, not this shell's, so a build must not look the compiler up again (`compiler_env`)
                **compiler_env(),
                # a receipt build is the judge's own (trusted); a plain one is the agent's, with its source copy
                **(host.cfg.qda_env if private is not None else host.cfg.agent_qda_env or host.cfg.qda_env),
                **host.cfg.contamination_env(),
                **({"PATH": host.qda_path} if host.qda_path else {}),
            }
            or None,
            build_private=private,
            expected_bytes=expected,
        )
    except BaseException:
        if private is not None:
            from ..sched.snapshot import remove_tree

            with host.lock:
                host.build_reserved.pop(str(private), None)
            remove_tree(private)
        raise
    if private is not None:  # the job's own request now holds the reservation for the cap
        with host.lock:
            host.build_reserved.pop(str(private), None)
    if host.cfg.full_card_moe:
        host.full_card_build = job.name
    return Reply(0, f"queued build {job.name} ({job.ncards} cards); `qsched log {job.name}`\n")


def kl(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`kl DIR`: a reading of a frozen copy (`judge.kl_cmd`, KL by default); `kl list`, `kl show ID`."""
    if argv[:1] == ["list"]:
        return Reply(0, "".join(e.line() for e in host.rows("scored")) or "no readings yet\n")
    if argv[:1] == ["show"] and len(argv) == 2:
        return Reply(0, host.entry(argv[1], "scored").line())
    if usage := _one_dir(host, argv, "usage: kl DIR | kl list | kl show ID\n"):
        return usage
    full_card_build = host.full_card_build
    if full_card_build is not None:
        job = host.sched.get(full_card_build, origin="agent")
        if job.state is not State.done:
            raise Refused(f"full_card_moe: wait for build {full_card_build} to finish before scoring")
        if job.build_private is None:
            raise Refused("full_card_moe requires a trusted build receipt")
        receipt = json.loads((Path(job.build_private) / "receipt.json").read_text())
        if _inside(host, cwd, argv[0]) != receipt["destination"]:
            raise Refused(f"full_card_moe: score {receipt['destination']} before another build")
    e = take(host, "scored", argv[0], cwd)
    with host.lock:
        if e.valid is None:
            cards = host.cfg.reading_cards()
            moe = has_routed_experts(e.dir)
            cfg = host.cfg
            expected = cfg.reading_expected_s(tree_size(e.dir), cards, moe)
            queue(host, e, cfg.kl_cmd, Kind.score, cards, expected, moe=moe)
        if full_card_build is not None and host.full_card_build == full_card_build:
            host.full_card_build = None
    return Reply(0, f"{e.said()}; `kl show`\n")


def gencheck(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`gencheck DIR [--max-tokens N] [--per-domain K]`: long generations of a frozen copy and the original."""
    if argv[:1] == ["list"]:
        return Reply(0, "".join(e.line() for e in host.rows("gencheck")) or "no checks yet\n")
    if argv[:1] == ["show"] and len(argv) == 2:
        return Reply(0, host.entry(argv[1], "gencheck").line())
    request = GENCHECK_PARSER.parse_args(argv)
    _open(host)
    max_tokens, per_domain = gencheck_length(host, request.max_tokens, request.per_domain)
    e = take(host, "gencheck", request.dir, cwd)
    with host.lock:
        # A trusted layer-cache receipt marks a build that cannot be a finalist or checked.
        if why := screening_build(e.dir, e.build_receipt):
            host.entries.remove(e)
            from .prune import drop_frozen

            drop_frozen(host, e)  # only if no other entry of these files still points at the copy
            raise Refused(f"{request.dir}: {why}")
        host.update(e, max_tokens=max_tokens, per_domain=per_domain)  # the ledger says what was asked for
        if e.valid is None:
            cfg = host.cfg
            queue(
                host,
                e,
                cfg.gencheck_cmd,
                Kind.check,
                cfg.gencheck_cards,
                gencheck_expected_s(cfg, max_tokens, per_domain),
            )
    return Reply(0, f"{e.said()}, asked {utc(e.submitted)}; `gencheck show {e.id}`\n")


#: The longest answer a check may ask for, in tokens; a row's `judge.eval_max_tokens` may set a lower cap.
GENCHECK_MAX_TOKENS = 32768

GENCHECK_PARSER = Parser(prog="gencheck", description="long generations of a frozen copy and the original")
GENCHECK_PARSER.add_argument("dir", metavar="DIR", help="a checkpoint directory in the work directory")
GENCHECK_PARSER.add_argument(
    "--max-tokens",
    type=int,
    dest="max_tokens",
    metavar="N",
    help="the longest answer, in tokens (default: judge.gencheck_max_tokens)",
)
GENCHECK_PARSER.add_argument(
    "--per-domain",
    type=int,
    dest="per_domain",
    metavar="K",
    help="prompts per domain (default: judge.gencheck_per_domain)",
)


def gencheck_length(host: JudgeHost, max_tokens: int | None, per_domain: int | None) -> tuple[int, int]:
    """A check's answer cap and prompts per domain: the config's defaults, or the agent's request, in range.

    A request is checked; the config's own defaults pass through unchanged, as they always did.

    Raises:
        Refused: an answer cap over the row's cap (`judge.eval_max_tokens`, and at most
            :data:`GENCHECK_MAX_TOKENS`), or more prompts per domain than the set holds. The message is one
            line of reason and one line of fix.
    """
    cfg = host.cfg
    tokens = cfg.gencheck_max_tokens if max_tokens is None else max_tokens
    domains = cfg.gencheck_per_domain if per_domain is None else per_domain
    if max_tokens is not None:  # a request is checked; the config's own defaults stand as they are
        cap = min(GENCHECK_MAX_TOKENS, cfg.eval_max_tokens or GENCHECK_MAX_TOKENS)
        if tokens < 1:
            raise Refused(
                f"--max-tokens {tokens} is below one token\nuse --max-tokens N with 1 <= N <= {cap}"
            )
        if tokens > cap:
            raise Refused(
                f"--max-tokens {tokens} is over this row's cap of {cap} tokens\n"
                f"use --max-tokens N with N <= {cap}"
            )
    if per_domain is not None:
        most = min(_prompts_per_domain(host).values(), default=0)
        if domains < 1 or domains > most:
            raise Refused(
                f"--per-domain {domains} is outside the prompt set: it holds {most} prompts in every domain\n"
                f"use --per-domain K with 1 <= K <= {most}"
            )
    return tokens, domains


def _prompts_per_domain(host: JudgeHost) -> dict[str, int]:
    """How many prompts the judge's long-generation set holds in each domain: the copy its jobs read."""
    from qda_kl.datasets import PROMPTS, parse_dataset

    try:
        path = host.install_prompts()  # the judge's own copy, once; None: the shipped set
    except ValueError as why:
        raise Refused(f"the long-generation prompt set cannot be read: {why}") from None
    if path is None:
        path = parse_dataset({"gencheck_builtin": {}}, "judge.gencheck_prompts", PROMPTS).local()
    counts: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            domain = json.loads(line).get("domain")
            if isinstance(domain, str):
                counts[domain] = counts.get(domain, 0) + 1
    return counts


def handin(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`handin DIR [DIR...]`: one hand-in list, best first, each a frozen copy checked when set; `handin status`."""
    from .prune import SpaceRefused, note_space_refusal

    if argv[:1] == ["status"]:
        rows, now = host.rows("handin"), host.final(generation_checks=False)
        counts = next((e.id for e in rows if now.is_pick(e)), None)
        if (
            counts is None and now.choice is not None
        ):  # the final rule submitted a scored reading, not a hand-in
            counts = f"{now.choice.id} ({now.choice.kind} entry: the final rule submitted it)"
        none = "no submission" if any(e.valid for e in rows) else "no valid submission"
        text = "".join(e.line() for e in rows) or "nothing submitted yet\n"
        return Reply(0, text + f"counts now: {counts or none}\n")
    if not argv or any(arg.startswith("-") for arg in argv):
        return Reply(
            0 if argv[:1] in (["-h"], ["--help"]) else 2, "", "usage: handin DIR [DIR...] | handin status\n"
        )
    if host.sched.judge_only or host.clock.now() >= host.sched.deadline:
        raise Refused("the GPU deadline has passed")
    check = host.cfg.llama_cmd if host.cfg.engine == "llama.cpp" else host.cfg.check_cmd
    with host.lock:
        host.handin_lists += 1
        list_id = host.handin_lists  # one call is one list; a later call replaces it
    said, refused = [], []
    for slot, arg in enumerate(argv):
        try:
            e = take(host, "handin", arg, cwd, list_id=list_id, slot=slot)
        except SpaceRefused as error:
            # A full disk is not a verdict on the checkpoint: record the refusal (the final rules may still
            # submit its scored reading, `final_rules.Handin`) and say it on the terminal.
            note_space_refusal(host, _inside(host, cwd, arg), error)
            refused.append(str(error))
            continue
        except Refused as error:  # this directory only: the rest of the list stays as submitted
            refused.append(str(error))
            continue
        with host.lock:
            if e.valid is None and check is None:
                late = host.sched.judge_only or host.clock.now() >= host.sched.deadline
                # a valid late-free submission keeps the reason it already had (`take` may mark it ineligible)
                host.update(
                    e, valid=not late, reason="submitted after the GPU deadline" if late else e.reason
                )
            elif e.valid is None and check is not None:
                queue(host, e, check, Kind.check, host.cfg.check_cards, host.cfg.check_expected_s)
        said.append(e.said())
    text = "".join(f"submitted {line}\n" for line in said)
    if refused:
        return Reply(2, text, "".join(f"handin: refused: {why}\n" for why in refused))
    return Reply(0, text)


def _open(host: JudgeHost) -> None:
    """Refuse after the GPU deadline.

    A late reading could become FINAL by a rule that ranks readings; what was asked in time drains
    (`judge_grace_s`). The rules have no clock: this gate is what keeps a late checkpoint from them.
    """
    if host.sched.judge_only or host.clock.now() >= host.sched.deadline:
        raise Refused("the GPU deadline has passed")


def _one_dir(host: JudgeHost, argv: list[str], usage: str) -> Reply | None:
    """The usage reply unless `argv` is one DIR; refused after the GPU deadline."""
    if len(argv) != 1 or argv[0].startswith("-"):
        return Reply(0 if argv[:1] in (["-h"], ["--help"]) else 2, "", usage)
    _open(host)
    return None


USAGE = "usage: bpw DIR | bpw --plan RECIPE.yaml\n"


#: The index bytes priced whatever the shards state. Beyond it an index may be as large as the bytes of the
#: shards' safetensors headers, which the host reads and parses anyway, so the index adds at most that much
#: again. A real index entry ("name": "shard") is shorter than the header entry naming the same tensor (its
#: dtype, shape and offsets), so a quantized checkpoint's index (57 MB over 96 shards whose headers state
#: 72 MB) prices; the floor covers the largest public BF16 indexes (Kimi-K2 12.5 MB, DeepSeek-V3 8.9 MB) and
#: the slack of an indented index over small tensors.
INDEX_FLOOR = 16 << 20
#: The index bytes never priced, whatever the shards state: about 4x the largest real index known (57 MB), so
#: shards claiming large headers cannot make the judge read and parse an index of any size.
INDEX_CAP = 256 << 20


def bpw(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`bpw DIR` prices a built checkpoint; `bpw --plan RECIPE` prices a recipe's formats, without building."""
    if argv[:1] in (["-h"], ["--help"]):
        return Reply(0, "", USAGE)
    anchor = host.cfg.anchor
    if anchor is None:
        raise Refused(
            "bpw needs judge.anchor: the budget's denominator is the original model's decoder-Linear "
            "weight count"
        )
    if argv[:1] == ["--plan"]:
        if len(argv) != 2 or argv[1].startswith("-"):
            return Reply(2, "", USAGE)
        return Reply(0, _plan(host, anchor, cwd, argv[1]))
    if len(argv) != 1 or argv[0].startswith("-"):
        return Reply(2, "", USAGE)
    return Reply(0, _checkpoint(host, anchor, cwd, argv[0]))


def _inside(host: JudgeHost, cwd: Path, name: str) -> str:
    """The workdir-relative path `name` names, read lexically: never through a link the agent may swap.

    Raises:
        Refused: `name` is absolute or climbs out of the work directory.
    """
    if Path(name).is_absolute():
        raise Refused(f"{name}: name a path inside the work directory")
    rel = os.path.normpath(Path(os.path.relpath(cwd, host.workdir), name))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        raise Refused(f"{name}: not inside the work directory")
    return rel


def _checkpoint(host: JudgeHost, anchor: Path, cwd: Path, name: str) -> str:
    """The checkpoint's bpw and its budget, in one line.

    Raises:
        Refused: not a directory of the work directory, a link (pricing the bytes a link names would price a
            host file), or a checkpoint the loader refuses.
    """
    rel = _inside(host, cwd, name)
    stage = Path(tempfile.mkdtemp(prefix=".bpw-", dir=host.jdir))
    try:
        try:
            _stage(host.workdir, rel, stage, name)
            value = budget.bpw(stage, anchor)
            lease = host.head_lease
            leased = None if lease is None else budget.leased_bpw(stage, budget.denominator(anchor), lease)
        except CheckpointError as e:
            raise Refused(f"{name}: {e}") from None
    finally:
        remove_tree(stage)
    if leased is None:
        return f"{name}: {value:.3f} bits per quantized weight{_budget(host, value)}\n"
    return (
        f"{name}: {value:.3f} bits per quantized weight, {leased:.3f} after the head's lease"
        f"{_budget(host, leased)}\n"
    )


def _stage(root: Path, rel: str, stage: Path, name: str) -> None:
    """Copy the checkpoint's tensor headers into `stage`, opening every entry through a pinned directory.

    No path the agent controls is followed by the host: the directory is reached from `root` one component at
    a time with `O_NOFOLLOW`, and each shard or index is opened the same way from that descriptor. Only a
    safetensors header (its length prefix and JSON) is copied, so pricing costs kilobytes per shard whatever
    the checkpoint's size; the copy is what `budget.bpw` reads, so a link (or a swap) cannot redirect it.

    The index is read after the shards, so its cap is `INDEX_FLOOR` plus the bytes of the shard headers that
    parse (a length a shard merely claims counts for nothing), and never more than `INDEX_CAP`; opening it
    first and reading it last keeps the cap independent of the directory's listing order. An index over its
    cap that names a shard file the directory lacks is refused for that (a build still writing), not its size.

    Raises:
        Refused: a component is a link or not a directory, a shard is not a regular file, or the index is too
            large to copy.
    """
    dfd = _dir_fd(root, rel, name)
    index = -1
    bound = INDEX_FLOOR
    shards: set[str] = set()
    try:
        for entry in os.listdir(dfd):  # noqa: PTH208 -- the descriptor pins the directory a path would not
            if entry != INDEX and not entry.endswith(".safetensors"):
                continue
            try:
                src = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            except OSError:
                raise Refused(f"{name}/{entry}: a link is not priced") from None
            if entry == INDEX:
                index = src
                continue
            with os.fdopen(src, "rb") as fh:
                info = os.fstat(fh.fileno())
                size = info.st_size
                if not stat.S_ISREG(info.st_mode):
                    raise Refused(f"{name}/{entry}: not a regular file")
                head = fh.read(8)
                n = int.from_bytes(head, "little") if len(head) == 8 else 0
                body = fh.read(n) if 0 < n <= MAX_HEADER else b""
                if bound < INDEX_CAP and n <= size - 8 and _is_header(body, n):
                    bound += n  # a header `budget` parses out of this shard, not a length it claims
                shards.add(entry)
                copy = stage / entry
                copy.write_bytes(head + body)
                os.truncate(copy, size)  # a sparse file: the offsets the header states stay in range
        if index >= 0:
            fd, index = index, -1
            with os.fdopen(fd, "rb") as fh:
                if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                    raise Refused(f"{name}/{INDEX}: not a regular file")
                bound = min(bound, INDEX_CAP)
                chunks, got = [], 0
                while got <= INDEX_CAP and (chunk := fh.read(min(1 << 20, INDEX_CAP + 1 - got))):
                    chunks.append(chunk)  # chunked to INDEX_CAP: past the bound, parsed for absent shards
                    got += len(chunk)
                body = b"".join(chunks)
                if got > bound:
                    if got <= INDEX_CAP and (absent := _absent_shard(body, shards)):
                        raise Refused(f"{name}: {INDEX} names {absent!r}, not a shard file of the checkpoint")
                    limit = (
                        f"the {INDEX_CAP >> 20} MiB cap on any index"
                        if bound == INDEX_CAP
                        else f"the {INDEX_FLOOR >> 20} MiB floor plus its shards' headers"
                    )
                    raise Refused(
                        f"{name}/{INDEX}: the shard index is too large to price: over {bound} B, {limit} "
                        f"(fix: keep only weight_map entries for tensors the shards hold, and no other metadata)"
                    )
                (stage / INDEX).write_bytes(body)
    finally:
        if index >= 0:
            os.close(index)
        os.close(dfd)


def _is_header(body: bytes, n: int) -> bool:
    """Whether `body` is all `n` bytes of a safetensors header: a JSON object."""
    if len(body) != n:
        return False
    try:
        return isinstance(json.loads(body), dict)
    except (ValueError, RecursionError):
        return False


def _absent_shard(body: bytes, shards: set[str]) -> str | None:
    """The first shard file the index's weight_map names that is not among `shards` (by `loaded`'s rule), or
    None (also when the index does not parse: `budget` reports that).
    """
    try:
        wm = json.loads(body).get("weight_map")
    except (ValueError, RecursionError, AttributeError):
        return None
    if not isinstance(wm, dict):
        return None
    named = {v for v in wm.values() if isinstance(v, str)}
    return min((v for v in named if "/" in v or v.startswith(".") or v not in shards), default=None)


def _dir_fd(root: Path, rel: str, name: str) -> int:
    """The checkpoint directory's descriptor, opened one no-follow component at a time from `root`.

    Raises:
        Refused: a component is a link, or not a directory.
    """
    parts = [p for p in Path(rel).parts if p != "."]
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError:
                raise Refused(
                    f"{name}: not a directory of the work directory (a link is not priced)"
                ) from None
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _budget(host: JudgeHost, value: float) -> str:
    """How a value stands against `judge.budget_bpw`, or that the search states none."""
    if (budget_bpw := host.cfg.budget_bpw) is None:
        return "; no budget set"
    return f"; budget {budget_bpw:g} bpw ({'inside' if value <= budget_bpw else 'over'})"


def _plan(host: JudgeHost, anchor: Path, cwd: Path, recipe: str) -> str:
    """The plan's per-class bpw and its total, from the library's own resolution of the recipe's formats.

    Raises:
        Refused: the recipe cannot be snapshotted or planned, or its model packs more decoder-Linear weights
            than the original has (it is not a recipe for this original).
    """
    rel = _inside(host, cwd, recipe)
    stage = Path(tempfile.mkdtemp(prefix=".bpw-", dir=host.jdir))
    try:
        try:
            snap = host.sched.iso.copy_out(host.workdir, rel, stage)
            raw = _pinned(snap, anchor, host)
            from qda_quant.config import BuildConfig

            config = BuildConfig.from_dict(raw, base=str((cwd / recipe).parent))
            plan = config.plan()
        except (SnapshotError, ValueError, OSError) as e:
            raise Refused(f"{recipe}: {e}") from None
        except Exception as e:  # noqa: BLE001 -- the planner's own refusal (a bad recipe) is reported as one
            raise Refused(f"{recipe}: {e}") from None
        try:
            weights = budget.denominator(anchor)
            kept_bits, kept_name = budget.kept_width(anchor)
            classes, total = budget.planned_bpw(
                plan.linears, weights, kept_bits=kept_bits, kept_name=kept_name
            )
            head = _planned_head(config, anchor) if host.head_lease is not None else None
        except (CheckpointError, ValueError) as e:
            raise Refused(f"{recipe}: {e}") from None
    finally:
        remove_tree(stage)
    lines = [f"{recipe}: {weights} decoder-Linear weights of the original\n"]
    lines += [f"  {name}: {n} weights, {bits / weights:.3f} bpw\n" for name, n, bits in classes]
    if host.head_lease is None:
        lines.append(f"total: {total:.3f} bits per quantized weight{_budget(host, total)}\n")
        return "".join(lines)
    from qda_quant.engine.checkpoint_size import packed_bits
    from qda_quant.head_lease import leased_bpw

    leased = total
    if head is not None:  # a head step packs the head; without one it stays 16-bit and lends nothing
        (rows, cols), f = head
        leased = leased_bpw(total * weights, packed_bits((rows, cols), f), weights, rows * cols)
        lent = total - leased
        lines.append(f"  head lease ({f.name}): {rows * cols} weights, lends {lent:.3f} bpw to the body\n")
    lines.append(
        f"total: {total:.3f} bits per quantized weight, {leased:.3f} after the head's lease"
        f"{_budget(host, leased)}\n"
    )
    return "".join(lines)


def _planned_head(config: Any, anchor: Path) -> tuple[tuple[int, int], Format] | None:
    """The head a recipe stores, with the original's head shape, to price its lease.

    `qda_quant`'s `Plan.linears` holds no entry for the head (`packed_weights` leaves it out), so the format
    its `head` step names and the original's `lm_head` shape are read here. None: no head step, so the head
    is written as the original stores it.
    """
    from qda_quant.format import Format

    step = next((s["head"] for s in config.steps if "head" in s), None)
    if step is None:
        return None
    shape = budget.head_shape(anchor)
    if shape is None:
        raise ValueError("the recipe stores a head step, and the original stores no lm_head weight")
    return shape, Format.parse(step["format"])


def _pinned(snap: Path, anchor: Path, host: JudgeHost) -> dict[str, Any]:
    """The recipe with its model pinned to the original, no output, and the host's CPU.

    The plan prices the original's shapes with the recipe's formats; the recipe's own model is not read and
    nothing is built, so its output path and cards play no part. A recipe that names no target is planned
    for the run's cards, as its build there would be; a recipe's own target wins.

    Raises:
        Refused: the snapshot is not a YAML mapping.
    """
    import yaml

    try:
        raw = yaml.safe_load(snap.read_text())
    except yaml.YAMLError as e:
        raise Refused(f"not a YAML recipe ({e})") from None
    if not isinstance(raw, dict):
        raise Refused("a recipe is a YAML mapping")
    # The host reads `judge.anchor` from its own working directory elsewhere (`anchor_bytes`); the pin does
    # the same, so a relative setting means the same directory in both places.
    from ..group_size import run_target, with_target

    raw = with_target(raw, run_target(host.sched.cfg.cards))
    return {**raw, "model": str(Path(anchor).resolve()), "out": None, "devices": ["cpu"]}


WAIT_MAX_S = 590.0


WAIT_POLL_S = 2.0


def job_options(p: Parser, default_s: float | None) -> None:
    """--cards, --expected, --name, --after (repeatable, or comma-separated)."""
    p.add_argument("--cards", type=int, default=1)
    if default_s is None:
        p.add_argument("--expected", "--expect", type=duration, required=True, help="e.g. 20m, 1.5h, 90s")
    else:
        p.add_argument("--expected", "--expect", type=duration, default=default_s, help="e.g. 20m, 1.5h, 90s")
    p.add_argument("--name")
    p.add_argument("--after", type=names, action="extend", default=[], help="job names (repeat or a,b)")


def _parser() -> Parser:
    p = Parser(prog="qsched", description="The run's GPU queue (your jobs).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="queue a script: it is copied now and run as `bash <copy>`")
    job_options(a, default_s=None)
    a.add_argument("script", help="a script in the work directory (interpreters and flags go inside it)")
    a.add_argument(
        "--checkpoint",
        metavar="DIR",
        help="a checkpoint directory this job writes: reserve a full checkpoint against the disk budget "
        "(without it a job reserves `judge.job_reserve_bytes`)",
    )
    sub.add_parser("list", help="your jobs")
    lg = sub.add_parser("log", help="a job's output")
    lg.add_argument("name")
    lg.add_argument("-n", type=int, default=50)
    c = sub.add_parser("cancel", help="cancel jobs")
    c.add_argument("names", nargs="+")
    sh = sub.add_parser("show", help="one job: state, cards, times, exit code, why it waits or ended")
    sh.add_argument("name")
    e = sub.add_parser("eta", help="when a job should end, from its progress lines (else its --expected)")
    e.add_argument("name")
    e.add_argument("--layers", type=int, help="the model's layer count: count `layer N` lines as progress")
    sub.add_parser("status", help="each card's job or idle time, the queue, the time left")
    sub.add_parser("usage", help="card-hours so far: leased by job kind, and idle")
    w = sub.add_parser(
        "wait",
        help="block until jobs end (at most 590 s); the last line starts MET, FAILED or TIMEOUT",
        description="Block until every --job has ended, or until any of your queued or running jobs ends "
        "(--any). Exit 0 met, 1 failed (a job did not end done), 124 timeout: call it again. To wait on a "
        "file, poll it in your own shell.",
    )
    w.add_argument("--job", action="append", default=[], metavar="NAME", help="wait for this job (repeat)")
    w.add_argument("--any", action="store_true", help="wait for any job queued or running now to end")
    w.add_argument("--timeout", type=duration, default=WAIT_MAX_S, help="at most 590s (the default)")
    return p


def qsched(host: JudgeHost, argv: list[str], cwd: Path) -> Reply:
    """`qsched add|list|log|show|eta|status|usage|wait|cancel` on the agent's own jobs."""
    ns = _parser().parse_args(argv)
    sched = host.sched
    if ns.cmd == "add":
        from .prune import check_room, disk_lock, expected_bytes

        with disk_lock(host):
            # A job that writes a checkpoint reserves a full one; another job runs in place and writes at
            # most what it creates, so it reserves the small default (`judge.job_reserve_bytes`).
            expected = expected_bytes(host) if ns.checkpoint else host.cfg.job_reserve_bytes
            check_room(host, expected)
            job = sched.submit(
                [],
                str(cwd),
                expected_s=ns.expected,
                ncards=ns.cards,
                name=ns.name,
                after=ns.after,
                kind=Kind.other,
                origin="agent",
                root=host.workdir,
                freeze=ns.script,
                expected_bytes=expected,
            )
        return Reply(0, f"queued {job.name} ({job.ncards} cards, expected {job.expected_s / 60:.1f} min)\n")
    if ns.cmd == "list":
        return Reply(0, _table(host))
    if ns.cmd == "log":
        return Reply(0, sched.log(ns.name, ns.n, origin="agent") + "\n")
    if ns.cmd == "show":
        return Reply(0, _show(host, sched.get(ns.name, origin="agent")))
    if ns.cmd == "eta":
        job = sched.get(ns.name, origin="agent")
        e = report.eta(sched, job, ns.layers)
        when = f"{utc(e.end)} (in {max(0.0, e.end - host.clock.now()) / 60:.1f} min)" if e.end else "unknown"
        verb = "ended" if e.state.terminal else "ends"
        return Reply(0, f"{job.name} {e.state.value}: {verb} {when}; {neutral(e.basis)}\n")
    if ns.cmd == "status":
        return Reply(0, _status(host))
    if ns.cmd == "usage":
        u = report.usage(sched)
        return Reply(0, "".join(f"{k:<14} {v:8.2f} card-h\n" for k, v in u.items()))
    if ns.cmd == "wait":
        return _wait(host, ns)
    done = [sched.cancel(n, origin="agent") for n in ns.names]
    return Reply(0, "".join(f"{j.name}: {j.state.value}\n" for j in done))


def _table(host: JudgeHost) -> str:
    """`qsched list`: the agent's jobs, the judge jobs pending and the time left."""
    rows = host.sched.list(origin="agent")
    judge = sum(not j.state.terminal for j in host.sched.list(origin="judge"))
    left = max(0.0, host.sched.deadline - host.clock.now()) / 60
    out = [f"{'name':<24} {'state':<9} {'cards':<8} {'expect':>7}  reason"]
    for j in rows:
        cards = ",".join(map(str, j.cards)) if j.cards else str(j.ncards)
        out.append(
            f"{j.name:<24} {j.state.value:<9} {cards:<8} {j.expected_s / 60:>6.1f}m  {neutral(j.reason)}"
        )
    out.append(f"judge jobs pending: {judge}; GPU deadline in {left:.1f} min")
    return "\n".join(out) + "\n"


def _show(host: JudgeHost, j: Job) -> str:
    """`qsched show`: one job's state, cards and times."""
    now = host.clock.now()
    ran = ((j.ended or now) - j.started) / 60 if j.started else None
    rows = [
        ("name", j.name),
        ("state", j.state.value + (f" (exit {j.rc})" if j.rc is not None else "")),
        ("cards", ",".join(map(str, j.cards)) or f"{j.ncards} (not started)"),
        ("expected", f"{j.expected_s / 60:.1f} min"),
        ("submitted", utc(j.submitted)),
        ("started", utc(j.started) if j.started else "-"),
        ("ended", utc(j.ended) if j.ended else "-"),
        ("ran", f"{ran:.1f} min" if ran is not None else "-"),
        ("after", ", ".join(j.after) or "-"),
        ("reason", neutral(j.reason) or "-"),
    ]
    return "".join(f"{k:<10} {v}\n" for k, v in rows)


def _status(host: JudgeHost) -> str:
    """`qsched status`: each card's job or idle time, the queue and the time left."""
    now = host.clock.now()
    mine = host.sched.list(origin="agent")
    ready = [j for j in mine if report.runnable(host.sched, j)]
    out = []
    for c in report.cards(host.sched):
        mins = (now - c.since) / 60
        if c.job is not None:
            who = c.job.name if c.job.origin == "agent" else "a judge job"
            out.append(f"card {c.card}: {who}, running {mins:.1f} min")
        else:
            idle = "  IDLE! a queued job could use it" if any(j.ncards <= 1 for j in ready) else ""
            last = (
                f" (last: {c.last[0] if _mine(host, c.last[0]) else 'a judge job'} {c.last[1]})"
                if c.last
                else ""
            )
            out.append(f"card {c.card}: idle {mins:.1f} min{last}{idle}")
    waiting = sum(j.state is State.queued and not report.runnable(host.sched, j) for j in mine)
    judge = sum(not j.state.terminal for j in host.sched.list(origin="judge"))
    left = max(0.0, host.sched.deadline - now) / 60
    out.append(
        f"your jobs: {len(ready)} ready for cards, {waiting} waiting on --after; judge jobs pending: {judge}"
    )
    out.append(f"GPU deadline in {left:.1f} min ({utc(host.sched.deadline)})")
    return "\n".join(out) + "\n"


def _mine(host: JudgeHost, name: str) -> bool:
    """Whether `name` is one of the agent's jobs."""
    with contextlib.suppress(Refused):
        return host.sched.get(name, origin="agent") is not None
    return False


def _wait(host: JudgeHost, ns: argparse.Namespace) -> Reply:
    """`qsched wait`: poll the jobs every `WAIT_POLL_S` until the condition holds or the timeout passes.

    It waits on jobs only: the host never reads the agent's files (a file condition would follow the agent's
    symlinks out of the work directory).
    """
    if bool(ns.job) == ns.any:
        raise Usage(2, "qsched wait: give either --job NAME... or --any\n")
    timeout = min(ns.timeout, WAIT_MAX_S)
    jobs = [host.sched.get(n, origin="agent").name for n in dict.fromkeys(ns.job)]
    if ns.any:
        jobs = [j.name for j in host.sched.list(origin="agent") if not j.state.terminal]
        if not jobs:
            return Reply(1, f"FAILED nothing queued or running to wait for at {utc(host.clock.now())}\n")
    end = host.clock.now() + timeout
    while True:
        host.tick()
        line = _met(host, jobs, ns.any)
        if line is not None:
            code, text = line
            return Reply(code, f"{text} at {utc(host.clock.now())}\n")
        if host.clock.now() >= end:
            return Reply(124, f"TIMEOUT after {timeout:.0f} s at {utc(host.clock.now())}; call it again\n")
        host.clock.sleep(min(WAIT_POLL_S, max(0.0, end - host.clock.now())))


def _met(host: JudgeHost, jobs: list[str], any_job: bool) -> tuple[int, str] | None:
    """(exit, line) once the jobs the wait names have ended as it asks; None while it still waits."""
    states = {n: host.sched.get(n).state for n in jobs}
    ended = [n for n, s in states.items() if s.terminal]
    if any_job and ended:
        n = ended[0]
        met = states[n] is State.done
        return (0 if met else 1), f"{'MET' if met else 'FAILED'} {n} {states[n].value}"
    if not any_job and len(ended) == len(jobs):
        bad = [n for n in jobs if states[n] is not State.done]
        if bad:
            return 1, "FAILED " + ", ".join(f"{n} {states[n].value}" for n in bad)
        return 0, "MET " + ", ".join(f"{n} done" for n in jobs)
    return None


BUILTIN = {"qsched": qsched, "qda": qda, "bpw": bpw, "kl": kl, "gencheck": gencheck, "handin": handin}
