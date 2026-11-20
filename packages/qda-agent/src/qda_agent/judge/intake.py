"""Taking a candidate in: a confined copy of the agent's checkpoint, checked and sealed before anything reads it.

The host never resolves or reads the agent's directory. `take` copies it inside bwrap (`snapshot_tree`) and
runs every check on the copy; a refused copy is removed at once.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ..isolation.paths import published_entries
from ..prompt import candidate0_kl_statement
from ..sched import Refused
from ..sched.snapshot import SnapshotError, remove_tree
from .checks import CheckpointError, files_problem, near_copy, tensor_fingerprint
from .checks.budget import bpw_problem
from .checks.gguf import gguf_fingerprint, gguf_near_copy, gguf_problem, gguf_sizes
from .freeze import hardlink_in, seal, tree_size
from .ledger import Entry, Kind

if TYPE_CHECKING:
    from .host import JudgeHost

#: The largest `build.json` the judge reads from a candidate.
BUILD_JSON_MAX = 1 << 20
#: The letter of each kind's entry ids (`h1`, `k1`, `g1`).
ID_PREFIX = {"handin": "h", "scored": "k", "gencheck": "g"}


def same_dir(a: Path, b: Path) -> bool:
    """Whether two judged paths name the same directory (normalized, never following links)."""
    return Path(os.path.normpath(str(a))) == Path(os.path.normpath(str(b)))


def frozen_for(host: JudgeHost, digest: str, *, besides: Path | None = None) -> Path | None:
    """Where `digest` is already frozen: the one directory every entry of those files points at, or None.

    One owner of the fact "this digest is frozen at": `take` asks this before it records a new copy, so a
    checkpoint read, handed in and generation-checked is frozen once. `besides` is the caller's own copy (not
    a candidate for reuse). A directory a reclaim removed is not reused, and neither is one a prune pass left
    without the weight files `besides` holds (`prune.weight_files`): the files are frozen again, and
    the new entry keeps its own whole copy.
    """
    from .prune import weight_files

    def names(tree: Path) -> set[Path]:
        return {p.relative_to(tree) for p in weight_files(tree)}

    wanted = names(besides) if besides is not None else set()
    for e in host.entries:
        if e.digest != digest or (besides is not None and same_dir(e.dir, besides)):
            continue
        if e.dir.is_dir() and not e.dir.is_symlink() and wanted <= names(e.dir):
            return e.dir
    return None


def take(
    host: JudgeHost, kind: Kind, arg: str, cwd: Path, *, list_id: int | None = None, slot: int = 0
) -> Entry:
    """Copy `arg` (a checkpoint dir in the workdir) in confinement, check the copy, seal it, record it.

    The copy runs outside the host lock (it may take minutes). `list_id`/`slot` record a hand-in's list and
    its rank in it (0 = best first).

    Raises:
        Refused: `arg` leaves the work directory, cannot be copied, or the copy fails a check.
    """
    if kind != "gencheck" and _baseline_path(host, arg, cwd):
        bar = candidate0_kl_statement(host.cfg.candidate0_kl)
        raise Refused(f"{arg}: a read-only baseline path, not your result. {bar}".rstrip())
    if Path(arg).is_absolute():
        raise Refused(f"{arg}: give the path relative to the work directory")
    rel = os.path.normpath(Path(os.path.relpath(cwd, host.workdir), arg))
    if rel == ".." or rel.startswith("../"):
        raise Refused(f"{arg} is outside the work directory")
    if kind == "handin" and (alias := _shared_inode(host, rel)) is not None:
        raise Refused(
            f"{rel}: {alias} has another hard link, so another tree (a judge frozen copy, say) shares its bytes; "
            "a hand-in must hold its own: copy it (cp -r, not cp -al or ln) and hand in the copy"
        )
    with host.lock:
        host.issued[kind] += 1
        eid = f"{ID_PREFIX[kind]}{host.issued[kind]}"
        dst = host.jdir / "frozen" / eid
    copied = dst
    cfg = host.cfg
    # Refuse an already excessive run before even sizing the candidate.
    from .prune import SpaceRefused, check_room, disk_lock, over_cap

    if why := over_cap(host):
        raise SpaceRefused(why)
    # A receipted build is taken from the judge's host-only custody, never from the agent's workdir: a
    # copy the agent made there (before or after publication) is not the build that was receipted.
    custody = None
    reserved = False
    build_identity: str | None = None
    if cfg.require_build_receipt:
        # The lookup and the reservation are one step under the lock: a prune pass never deletes a custody
        # tree this hand-in is freezing (the entry that will protect it is recorded only after the copy).
        with host.lock:
            custody = next(
                (
                    tree
                    for destination, tree in published_entries(host.run_dir, host.workdir)
                    if os.path.normpath(str(destination.relative_to(host.workdir))) == rel
                ),
                None,
            )
            if custody is not None:
                host.reserved[rel] += 1  # counted: two concurrent hand-ins of one build each hold a token
                reserved = True
                receipt = read_json(custody.parent / "receipt.json") or {}
                tree = receipt.get("tree_sha256")
                build_identity = tree if isinstance(tree, str) and tree else None
        if custody is None:
            from . import prune as pruning
            from .freeze import digest

            if (pruned := pruning.pruned_build(host, rel)) is not None:
                record, receipt = pruned
                wanted = receipt.get("tree_sha256")
                source = host.workdir / rel
                intact = (
                    isinstance(wanted, str)
                    and bool(wanted)
                    and source.is_dir()
                    and not source.is_symlink()
                )
                if intact:
                    try:  # the agent's own copy still hashes to the receipted build: freeze it again
                        intact = digest(source) == wanted
                    except OSError:
                        intact = False
                if not intact:
                    why = (
                        "the run passed its disk cap (judge.max_run_bytes) and this build was not scored yet"
                        if record.get("reason") == pruning.AT_CAP
                        else "the run's disk reached its free-space floor"
                    )
                    raise Refused(
                        f"{rel}: its weights were cleaned ({why}), so the judge's copy is gone; this path no "
                        "longer holds the build's recorded files either: build it again"
                    )
                build_identity = wanted  # the re-frozen copy is this recorded build; `match` finds it again
    with host.lock:
        host.freezing.add(copied)  # a prune pass never takes the copy this take holds
    try:
        with disk_lock(host):
            # the free-space floor before any copy, cap or not; the copy's size is read only against a cap
            capped = cfg.max_run_bytes is not None or cfg.max_scratch_bytes is not None
            expected = host.sched.iso.size_out(host.workdir, rel) if capped and custody is None else 0
            check_room(host, expected, scratch_expected=0 if cfg.custody_dir is not None else expected)
            if custody is not None:
                # Each hardlink adds no held bytes; a fallback copy must check its growth before writing.
                shutil.copytree(
                    custody,
                    dst,
                    copy_function=lambda src, target: _link_or_copy(src, target, host),
                    symlinks=True,
                )
            else:
                host.sched.iso.copy_out(host.workdir, rel, dst)
            try:
                check_room(host)
            except Refused:
                remove_tree(dst)
                raise

        def refuse_build(why: str) -> Refused:
            """A verdict on the build itself, not a missing copy or a full disk: it can never be read."""
            if build_identity is not None:
                host.intake_refused.add(build_identity)
            return Refused(why)

        if (n := sum(1 for _ in dst.rglob("*"))) > host.cfg.max_files:
            remove_tree(dst)
            raise refuse_build(f"{rel}: more than {host.cfg.max_files} files ({n})")
        if (size := tree_size(dst)) > host.max_bytes:
            remove_tree(dst)
            raise refuse_build(
                f"{rel}: {size / 2**30:.2f} GiB, more than the cap ({host.max_bytes / 2**30:.2f} GiB)"
            )
        if why := admission_problem(host, dst):
            remove_tree(dst)
            raise refuse_build(f"{rel}: {why}")
        sizes: dict[str, int] = {}
        if cfg.engine == "llama.cpp":
            # Both figures are recorded; `judge.budget_basis` picked the one admission screened.
            try:
                sizes = gguf_sizes(dst) or {}
            except CheckpointError as e:
                remove_tree(dst)
                raise refuse_build(f"{rel}: not a readable GGUF ({e})") from None
        frozen = seal(dst)
        with host.lock:
            # One digest is frozen once: when these bytes are already frozen (an earlier `kl`, `handin` or
            # `gencheck` of the same files), this entry points at that directory and the copy just made goes.
            # The digest is only known after sealing the copy (the source is never read directly), so the
            # duplicate exists for the length of its own check, never beyond the lock.
            if (shared := frozen_for(host, frozen, besides=dst)) is not None:
                remove_tree(dst)
                dst = shared
        e = Entry(
            id=eid,
            kind=kind,
            src=rel,
            dir=dst,
            digest=frozen,
            submitted=host.clock.now(),
            sizes=sizes,
            list_id=list_id,
            slot=slot,
        )
        if cfg.require_build_receipt:
            from .build_receipt import match, screening_build

            receipt, why = match(host, dst, e.digest)
            e.build_receipt = receipt or {}
            if why or (why := screening_build(dst, e.build_receipt)):  # the cache's mark, from the receipt
                e.eligible, e.reason = False, why
        found = host.contamination([dst])
        if found:
            e.valid, e.reason = False, f"contamination check: {len(found)} file(s) hold test items"
        elif kind != "gencheck" and (why := _baseline_problem(host, dst)):
            e.valid, e.reason = False, why
        with host.lock:
            from .jobs import KL_REFUSED

            if kind != "gencheck" and e.digest in host.kl_refused:
                e.eligible, e.reason = False, KL_REFUSED
            host.entries.append(e)  # from here on the entry itself keeps the custody tree
            host.log(e)
            if reserved:
                _release(host, rel)
            reserved = False
        return e
    except Refused:
        remove_tree(copied)  # never `dst`: after a reuse that path is another entry's frozen copy
        raise
    except (OSError, SnapshotError) as err:
        remove_tree(copied)
        raise Refused(str(err)) from None
    finally:
        with host.lock:
            host.freezing.discard(copied)
            # The copy failed or a check refused it: this tree is prunable again (if no one else holds it).
            if reserved:
                _release(host, rel)


def _release(host: JudgeHost, rel: str) -> None:
    """Drop one in-flight reservation; only the last one makes the custody tree prunable again."""
    host.reserved[rel] -= 1
    if host.reserved[rel] <= 0:
        del host.reserved[rel]


def _shared_inode(host: JudgeHost, rel: str) -> str | None:
    """The first file of the hand-in at `rel` with a second hard link (`freeze.hardlink_in`), or None.

    Only at `isolation.level: none`, where the agent sees judge custody and could link it (`cp -al`); the host
    walks the tree as `size_out` does there. A sandboxed level hides custody and never lets the host open it.
    """
    if getattr(host.sched.iso, "level", None) != "none":
        return None
    src = host.workdir / rel
    found = hardlink_in(src)
    return None if found is None else (found.relative_to(src).as_posix() if found != src else src.name)


def _link_or_copy(source: str, target: str, host: JudgeHost) -> None:
    """Hardlink `source` to `target`, or copy its bytes when the two trees are on different filesystems."""
    try:
        os.link(source, target)
    except OSError:  # EXDEV (another filesystem), or a filesystem without hardlinks: fall back to a copy
        from .prune import check_room

        check_room(host, Path(source).stat().st_size)
        shutil.copyfile(source, target)


def _baseline_path(host: JudgeHost, arg: str, cwd: Path) -> bool:
    """Match configured baseline paths and bind names without following agent-controlled paths."""
    baselines = {Path(os.path.abspath(p)) for p in host.cfg.baselines}  # noqa: PTH100
    baselines.update(p.resolve() for p in host.cfg.baselines)
    names = baselines | {
        Path(os.path.abspath(dst))  # noqa: PTH100
        for dst, src in host.sched.cfg.binds.items()
        if src.resolve() in baselines
    }
    return Path(os.path.abspath(cwd / arg)) in names  # noqa: PTH100 -- normalize, never follow agent links


def _baseline_problem(host: JudgeHost, frozen: Path) -> str | None:
    """Why `frozen` is a baseline as-is, exactly or within the judge's tolerance, or None."""
    cfg = host.cfg
    if cfg.engine == "llama.cpp":
        # a .gguf has no safetensors to load: its identity is the tensors its header stores
        try:
            fp = gguf_fingerprint(frozen)
        except CheckpointError as e:  # fail closed: a copy cannot be ruled out if the tensors cannot be read
            return f"its stored tensors cannot be read, so a baseline copy cannot be ruled out ({e})"
        if fp is not None and fp in host.baseline_weights:
            return "a baseline checkpoint as-is, not your result"
        refs = [(f"baselines[{i}]", b) for i, b in enumerate(cfg.baselines)]
        if cfg.format_reference is not None:
            refs.append(("judge.format_reference", cfg.format_reference))
        for name, ref in refs:
            if not ref.is_dir():  # a hub id: no tree on this host to compare with
                continue
            try:
                why = gguf_near_copy(frozen, ref)
            except CheckpointError as e:  # fail closed: never skip a baseline it could not compare against
                return f"{name} cannot be read, so a copy of it cannot be ruled out ({e})"
            if why:
                return why
        return None
    if _fingerprint(frozen, cfg.unscored) in host.baseline_weights:
        return "a baseline checkpoint as-is, not your result"
    # the original itself is a reference too: a hand-in of the anchor is a copy, not a quantization
    refs = [*cfg.baselines, *([cfg.format_reference] if cfg.format_reference is not None else [])]
    if cfg.anchor is not None:
        refs.append(cfg.anchor)
    for ref in refs:
        if not ref.is_dir():  # a hub id, or a baseline the host could not load: nothing to compare
            continue
        try:
            why = near_copy(frozen, ref)
        except CheckpointError:
            continue
        if why:
            return why
    return None


def _fingerprint(d: Path, unscored: list[str] | tuple[str, ...]) -> str | None:
    """`tensor_fingerprint` of `d`, or None when its tensors cannot be read."""
    try:
        return tensor_fingerprint(d, unscored)
    except CheckpointError:
        return None


def admission_problem(host: JudgeHost, d: Path, *, budget: bool = True) -> str | None:
    """Why the judge does not admit checkpoint `d` as a candidate, or None: its files, format, serving and budget.

    The one admission gate: a hand-in (`take`) and the launch check's candidate 0 (`smoke`) both go through it.
    `budget=False` leaves out the bpw budget, which judges an allocation, not whether the export is admissible.
    """
    cfg = host.cfg
    why = (
        gguf_problem(
            d,
            cfg.anchor,
            menu=cfg.format_menu,
            budget_bytes=cfg.budget_bytes,
            budget_basis=cfg.budget_basis,
            mtp=cfg.llamacpp.mtp if cfg.llamacpp is not None else "any",
            any_format=cfg.allow_any_format,
            reference=cfg.baselines[0] if cfg.baselines else None,
            lock=cfg.llamacpp.lock if cfg.llamacpp is not None else (),
            head=cfg.head,
        )
        if cfg.engine == "llama.cpp"
        else files_problem(
            d,
            cfg.anchor,
            cfg.format_reference,
            format_menu=cfg.format_menu,
            any_format=cfg.allow_any_format,
            format_map=cfg.format_map,
            head=cfg.head,
            lease=cfg.lease_formats(),
            # The floor holds only where no budget prices the allocation: with `judge.budget_bpw` the budget
            # is the rule (a partly bf16 body is legal if it fits it), and a candidate 0 (`budget=False`) is
            # the row's baseline, not a build to judge.
            floor=budget and cfg.budget_bpw is None,
        )
    )
    if why or (why := serve_problem(d)):
        return why
    if budget and cfg.engine == "vllm" and host.bpw_denominator is not None and cfg.budget_bpw is not None:
        return bpw_problem(d, host.bpw_denominator, cfg.budget_bpw, lease=host.head_lease)
    return None


def serve_problem(frozen: Path) -> str | None:
    """Why the frozen candidate's `build.json["serve"]` may not be applied, or None.

    The KL reading and the serving check apply `serve` (the engine's environment and arguments) on the host,
    outside any sandbox. So a candidate's section must hold only what `qda build` writes
    (`qda_quant.output.serve_policy`); the judge refuses the candidate otherwise, before anything serves it.
    """
    from qda_quant.output.serve_policy import ServeNotAllowed, check

    f = frozen / "build.json"
    if f.is_symlink():
        return "build.json is a link"
    if not f.is_file():
        return None
    if f.stat().st_size > BUILD_JSON_MAX:
        return f"build.json is larger than {BUILD_JSON_MAX} bytes"
    try:
        doc = json.loads(f.read_text())
    except (UnicodeDecodeError, ValueError):
        return "build.json is not valid JSON"
    if not isinstance(doc, dict):
        return "build.json is not a JSON object"
    try:
        check(doc.get("serve"))
    except ServeNotAllowed as e:
        return str(e)
    return None


def has_routed_experts(ckpt: Path) -> bool:
    """Whether a (frozen) checkpoint has routed experts.

    `build.json["serve"]["kernels"]` names a `moe` kernel, or else `config.json` (or its `text_config`)
    has `num_experts` / `num_local_experts` > 0. An unreadable file counts as no experts.
    """
    build = read_json(ckpt / "build.json") or {}
    raw_serve = build.get("serve")
    serve = raw_serve if isinstance(raw_serve, dict) else {}
    if isinstance(serve.get("kernels"), dict) and "moe" in serve["kernels"]:
        return True
    cfg = read_json(ckpt / "config.json") or {}
    raw_text = cfg.get("text_config")
    for c in (cfg, raw_text if isinstance(raw_text, dict) else {}):
        for k in ("num_experts", "num_local_experts"):
            v = c.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                return True
    return False


def read_json(p: Path | None) -> dict | None:
    """The JSON object in `p`, or None when it is missing, unreadable or not an object."""
    try:
        d = json.loads(p.read_text()) if p is not None else None
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None
