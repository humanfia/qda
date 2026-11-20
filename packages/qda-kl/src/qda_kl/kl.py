"""``qda-kl kl CHECKPOINT``: one checkpoint's KL from its own original, in one command.

    qda-kl kl <checkpoint>                         # a local directory or a Hub id (REPO[@REV])
    qda-kl kl <checkpoint> --original M            # when the original cannot be resolved, or to override it
    qda-kl kl <checkpoint> --prompts F --tp 2      # another prompt file; the reference's tensor parallelism
    qda-kl kl <checkpoint> --gpu-memory-utilization 0.3   # a card another job partly uses

The original is, in order: ``--original``; the ``model`` the checkpoint's own ``resolved.yaml`` records, at the
Hub commit its ``provenance.source_revision`` records (the commit the build read; a later stage of a
multi-stage recipe is followed back to the model the first stage read) or the commit its ``build.json``
``source_repo`` pins for the same repo; the original ``build.json`` records when there is no ``resolved.yaml``;
``base_model`` in its ``config.json`` or its model card's metadata. With none of them the command refuses in one
line naming ``--original``: it never guesses.

The prompts default to the holdout split of the KL v2 corpus (the prompts `qda-agent evaluate` reports on),
built once into the corpus cache. The reference is the reference cache's (`qda_kl.refcache`): the first call on
an original builds it and says so; every later call on any checkpoint of that original reuses it, and so does
`qda-agent evaluate` when tp and sampling agree (the cache keys the effective settings). The reading is
`qda-kl score`'s, written under ``--out``
(default ``$QDA_CACHE/kl-score/<checkpoint name>``), and the command prints its KL, the per-domain KL and the
engine it read on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from qda_quant.exit import ERROR, refuse
from qda_quant.step import Refused

from qda_kl.engine_params import DEFAULT_TP, engine_parameters, topk_settings
from qda_kl.engine_params import add_arguments as add_engine_arguments


def _front_matter(readme: Path) -> dict[str, Any]:
    if not readme.is_file():
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", readme.read_text(errors="ignore"), re.S)
    try:
        got = yaml.safe_load(m.group(1)) if m else {}
    except yaml.YAMLError:
        return {}
    return got if isinstance(got, dict) else {}


def _recorded_model(checkpoint: Path) -> tuple[str, str] | None:
    """The `model` a QDA checkpoint's own `resolved.yaml` records, and the file that names it.

    A Hub id is pinned to the commit the build read (``provenance.source_revision``) when the record has one:
    a bare ``org/name`` would otherwise read whatever the Hub's ``main`` holds now.
    """
    resolved = checkpoint / "resolved.yaml"
    if not resolved.is_file():
        return None
    try:
        record = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError:
        return None
    model = record.get("model") if isinstance(record, dict) else None
    if not isinstance(model, str) or not model:
        return None
    provenance = record.get("provenance")
    revision = provenance.get("source_revision") if isinstance(provenance, dict) else None
    if isinstance(revision, str) and revision and _local(model, checkpoint.parent) is None:
        return f"{model.partition('@')[0]}@{revision}", "resolved.yaml"
    return model, "resolved.yaml"


def _built_model(checkpoint: Path) -> tuple[str, str] | None:
    """The original a checkpoint's ``build.json`` records (``source_repo``, else ``source``), and the file.

    A published checkpoint ships ``build.json`` and no ``resolved.yaml``; the commit it built from is in
    ``source_repo``. `qda eval` reads that file with `qda_kl.eval.cli.original`, and so does this: one parser.
    """
    if not (checkpoint / "build.json").is_file():
        return None
    from qda_kl.eval.cli import original

    try:
        return original(checkpoint), "build.json"
    except ValueError:
        return None


def _local(model: str, beside: Path) -> Path | None:
    """The local directory a recorded `model` names, or None when it names a Hub id.

    A relative path is relative to where that build ran: beside the checkpoint first (the stages of one recipe
    write next to each other), then the working directory.
    """
    path = Path(model).expanduser()
    if not model.startswith(("/", "~", ".")) and not (beside / path).is_dir():
        return None  # `org/name[@rev]`: a Hub id
    for candidate in [path] if path.is_absolute() else [beside / path, path]:
        if candidate.is_dir():
            return candidate.resolve()
    return path


def _pinned_by_build(model: str, checkpoint: Path, origin: str | None) -> tuple[str, str | None]:
    """`model` at the commit `build.json` records for the same repo, when the resolved record names none."""
    if model.partition("@")[2]:
        return model, origin
    built = _built_model(checkpoint)
    repo, _, commit = built[0].partition("@") if built else ("", "", "")
    return (f"{repo}@{commit}", "build.json") if commit and repo == model else (model, origin)


def original_record(checkpoint: Path) -> tuple[str, str | None] | None:
    """The original model a checkpoint was made from and the file that records it; None when nothing does.

    A later stage of a multi-stage recipe records the earlier stage's local output as its `model`; that output
    is a QDA checkpoint too (it has a `resolved.yaml`), so the chain is followed back to the model the first
    stage read. Scoring against an intermediate would give a KL nothing else is comparable with.

    Raises:
        Refused: a stage in the chain names a local directory that is not on this machine, so the original
            cannot be told.
    """
    found = _recorded_model(checkpoint)
    seen = {checkpoint.resolve()}
    here = checkpoint
    while found is not None:
        model, origin = found
        local = _local(model, here.parent)
        if local is None:
            return _pinned_by_build(model, checkpoint, origin)  # a Hub id: the original
        if not local.is_dir():
            if here == checkpoint:
                return model, origin  # the build's own original, not here: `run` refuses it with its fix
            raise Refused(
                f"{checkpoint}: built from the QDA checkpoint {here}, whose own model {model} is not on this "
                "machine, so the original model is unknown",
                fix="pass --original ID@REVISION (the model the recipe's first stage read)",
            )
        earlier = _recorded_model(local)
        if earlier is None or local in seen:
            return str(local), origin  # a plain model directory (or a loop): the original
        seen.add(local)
        here, found = local, earlier
    built = _built_model(checkpoint)
    if built is not None and ((local := _local(built[0], checkpoint.parent)) is None or local.is_dir()):
        return built  # a Hub id, or a directory on this machine
    config = checkpoint / "config.json"
    if config.is_file():
        base = json.loads(config.read_text()).get("base_model")
        if isinstance(base, str) and base:
            return base, "config.json"
    base = _front_matter(checkpoint / "README.md").get("base_model")
    if isinstance(base, list) and len(base) == 1:
        base = base[0]
    if isinstance(base, str) and base:
        return base, "README.md"
    return built  # the build's local source, not here: `run` refuses it with its fix


def original_of(checkpoint: Path) -> str | None:
    """The original model a checkpoint was made from, from its own files; None when they do not say.

    See :func:`original_record` for the order: the ``resolved.yaml`` chain, then ``build.json``, then the
    ``config.json``/model card ``base_model``.
    """
    found = original_record(checkpoint)
    return found[0] if found else None


def revision_note(original: str, source: Path, given: str | None, origin: str | None = None) -> str:
    """Which revision of the original the reference reads, and the file that named it."""
    from huggingface_hub import constants

    if Path(original).expanduser().is_dir():
        return "a local directory"
    # snapshot_download resolves the cache dir's symlinks; HF_HUB_CACHE keeps the configured form
    in_cache = source.parent.name == "snapshots" and source.resolve().is_relative_to(
        Path(constants.HF_HUB_CACHE).resolve()
    )
    commit = source.name if in_cache else str(source)
    named = original.partition("@")[2]
    if given:
        return f"revision {commit}, from --original"
    if named == commit:
        return f"revision {commit}, from {origin or 'the checkpoint'}"
    if named:
        return f"revision {commit}: the checkpoint names {named}, resolved now"
    return f"revision {commit}: the checkpoint records no commit, so the Hub's main"


def default_prompts() -> Path:
    """The KL v2 corpus's holdout split (`qda_kl.holdout.default_prompts`, which the `kl` eval suite reads too)."""
    from qda_kl import holdout

    return holdout.default_prompts()


def _log(line: str) -> None:
    print(f"qda-kl kl: {line}", file=sys.stderr, flush=True)


def _rows_note(reference: Path) -> str | None:
    """The line naming rows the reference skipped as past the model's context (None: it skipped none)."""
    from qda_kl.reference import over_context_line

    try:
        build = json.loads((Path(reference) / "manifest.json").read_text()).get("build") or {}
    except (OSError, json.JSONDecodeError):
        return None
    return over_context_line(build)


def gguf_files(given: str) -> list[str]:
    """The `.gguf` files of `given` when it is a GGUF with nothing vLLM loads; empty otherwise.

    A `.gguf` file, a directory, or a Hub repo (`REPO[@REV]`, its file list only: nothing is downloaded)
    whose files include a `.gguf` and no `.safetensors`. A repo that cannot be listed (offline) is left to
    the normal path.
    """
    path = Path(given).expanduser()
    if path.suffix == ".gguf":
        return [path.name]
    if path.is_dir():
        names = [p.name for p in path.iterdir()]
    elif "/" in given and not given.startswith((".", "/", "~")):
        repo, _, revision = given.partition("@")
        try:
            from huggingface_hub import list_repo_files

            names = list_repo_files(repo, revision=revision or None)
        except Exception:  # noqa: BLE001 -- offline, private or missing: the normal path says which
            return []
    else:
        return []
    ggufs = sorted(Path(n).name for n in names if n.endswith(".gguf"))
    return ggufs if ggufs and not any(n.endswith(".safetensors") for n in names) else []


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``qda-kl kl``."""
    parser.add_argument("checkpoint", help="the checkpoint: a local directory or a Hub id (REPO[@REV])")
    parser.add_argument(
        "--original",
        help="the original model (a local directory or REPO[@REV]; default: the model and Hub revision the "
        "checkpoint's resolved.yaml or build.json records)",
    )
    parser.add_argument(
        "--prompts", type=Path, help="a prompt JSONL (default: the KL corpus's holdout split)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=DEFAULT_TP,
        help=f"cards the reference and the reading use (default {DEFAULT_TP})",
    )
    add_engine_arguments(parser)
    parser.add_argument("--out", type=Path, help="where score.json goes (default $QDA_CACHE/kl-score/<name>)")


def run(args: argparse.Namespace) -> int:
    """Resolve the original, get its cached reference, score the checkpoint and print the reading."""
    from qda_quant.io import qda_cache

    from qda_kl.refcache import cached, local_model
    from qda_kl.reference import default_sampling

    try:
        if ggufs := gguf_files(args.checkpoint):
            raise Refused(
                f"{args.checkpoint}: a GGUF ({', '.join(ggufs[:2])}); `qda-kl kl` reads a checkpoint vLLM loads",
                fix='score it with `qda-kl score`, which decodes the file (docs/verify.md, "A GGUF file")',
            )
        checkpoint = local_model(args.checkpoint)
        recorded = None if args.original else original_record(checkpoint)
        original = args.original or (recorded[0] if recorded else None)
        if not original:
            raise Refused(
                f"{args.checkpoint}: its files name no original model (no resolved.yaml `model`, no build.json "
                "source, no base_model); pass --original M"
            )
        if (
            not args.original
            and original.startswith(("/", "~", "."))
            and not Path(original).expanduser().is_dir()
        ):
            raise Refused(
                f"{args.checkpoint}: its original is the local directory {original}, which is not on this "
                "machine; pass --original M (the Hub id, or where the original is here)"
            )
        source = local_model(original)
        prompts = args.prompts or default_prompts()
        # the reference cache fills every setting left out from the build's signature; the engine's four
        # parameters are the ones only this command's flags know
        settings = {"tp": args.tp, "sampling": default_sampling(source), **engine_parameters(args)}
        print(
            f"qda-kl kl: {checkpoint} against its original {original} "
            f"({revision_note(original, source, args.original, recorded[1] if recorded else None)})",
            file=sys.stderr,
        )
        reference, built = cached(source, prompts, None, settings, log=_log)
        note = _rows_note(reference)
        if note:
            _log(note)
    except Refused as why:
        return refuse(str(why), why.fix or None, code=why.code)
    except Exception as why:  # noqa: BLE001 (a crash is an error, not a verdict on the model)
        return refuse(f"{type(why).__name__}: {why}", code=ERROR)
    out = args.out or qda_cache("kl-score", checkpoint.name)
    out.mkdir(parents=True, exist_ok=True)
    config = out.with_suffix(".yaml")
    body: dict[str, Any] = {"checkpoint": str(checkpoint), "reference": str(reference), "out": str(out)}
    # the reading's engine starts on the same card as the reference's: the same parameters reach both
    body["scorer"] = {"topk": topk_settings(args)}
    config.write_text(yaml.safe_dump(body, sort_keys=False))
    from qda_kl.cli import _score_config

    rc = _score_config(config, [])
    if rc != 0:
        return rc
    result = json.loads((out / "score.json").read_text())
    how = "built now" if built else "cached"
    lines = [f"KL {result['kl']:.6f}  {checkpoint}  (original {original}, reference {how})"]
    for domain, value in sorted((result.get("domains") or {}).items()):
        kl = value.get("kl") if isinstance(value, dict) else value
        lines.append(f"  {domain:14} {kl:.6f}" if isinstance(kl, (int, float)) else f"  {domain:14} {kl}")
    kernels = "; ".join(f"{kind} {', '.join(names)}" for kind, names in (result.get("kernels") or {}).items())
    if table := result.get("kernels_from"):  # checked against this card's plan
        built = json.loads((checkpoint / "build.json").read_text())["target"].partition("@")[0]
        why = "a recorded kernel is no longer planned on this card"
        why = why if built == table else f"the checkpoint was built for {built}"
        kernels += f" (this card's {table} plan; {why})"
    lines.append(
        f"  read on vLLM {result.get('vllm')}, tp {result.get('tp')}, kernels: {kernels or 'unrecorded'}; "
        f"{result.get('tokens')} tokens, top-1 flip rate {result.get('flip_rate', float('nan')):.4f}"
    )
    lines.append(f"  reading: {out / 'score.json'}")
    print("\n".join(lines))
    return 0
