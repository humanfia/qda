"""The report `qda inspect` prints: what a checkpoint is, what it lacks and why it does not serve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .formats import _gguf, checkpoint_format, detect, int_format
from .probe import INDEX, PROCESSOR_FILES, TOKENIZER_SETS, Source

if TYPE_CHECKING:
    from ..target import Target


def missing_files(src: Source) -> tuple[list[str], list[str]]:
    """What vLLM needs before it can serve `src`, and what it can do without."""
    if _gguf(src):
        return [], []
    fatal, soft = [], []
    if "config.json" not in src.files:
        fatal.append("config.json")
    if INDEX in src.files and src.where and Path(src.where).is_dir():
        shards = set(json.loads((Path(src.where) / INDEX).read_text())["weight_map"].values())
        fatal += sorted(s for s in shards if s not in src.files)
    if not fatal and not src.tensors:
        fatal.append("*.safetensors")
    if not any(all(f in src.files for f in s) for s in TOKENIZER_SETS):
        fatal.append("tokenizer (tokenizer.json, tokenizer.model or vocab.json + merges.txt)")
    if "tokenizer_config.json" not in src.files:
        soft.append("tokenizer_config.json")
    cfg = src.jsons.get("config.json") or {}
    if "vision_config" in cfg and not any(f in src.files for f in PROCESSOR_FILES):
        fatal.append("preprocessor_config.json (a vision model needs its processor)")
    if "generation_config.json" not in src.files:
        soft.append("generation_config.json (sampling falls back to the server's defaults)")
    template = "chat_template" in (src.jsons.get("tokenizer_config.json") or {})
    if not template and not any(f.startswith("chat_template") for f in src.files):
        soft.append("chat template (chat_template.jinja or tokenizer_config.json's chat_template)")
    return fatal, soft


def _card_capability() -> tuple[int, int] | None:
    """The compute capability of the first card this process sees, or None on a host with no card."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except (ImportError, RuntimeError, OSError):
        return None


def local_target() -> Target | None:
    """The target of the card this process sees, as `qda plan` takes it, or None with no card (or no table)."""
    from ..step import Refused
    from ..target import Target

    capability = _card_capability()
    if capability is None:
        return None
    try:
        return Target.on_cards(None, [("cuda:0", capability)])
    except Refused:  # a capability no table covers: the report has no card to be about
        return None


def _built_elsewhere_note(built: Target, here: Target) -> str:
    """That the checkpoint was built for another target than the card this report is about."""

    # the record's target in full; the card is named by its arch only, since the table version is the library's,
    # not the installed vLLM's
    built_for = f"{built.arch}@{built.vllm}" + (f":{built.backend}" if built.backend else "")
    return f"the checkpoint was built for {built_for} (build.json); this card is {here.arch}"


def inspect(path: str, target: str | None = None) -> dict[str, Any]:
    """What `path` is, what it lacks and whether `target` serves it as it is; ``reasons`` empty = servable.

    Without `target`, a card this process sees is the target, as `qda plan` takes it, and a ``build.json`` that
    records another one gets a note. On a host with no card the checkpoint is checked against the target its
    ``build.json`` records, the one it was built for; a checkpoint that records none or no usable target falls
    back to the default table, and the unusable one is named in a note.
    """
    from ..output.serve_command import served_name
    from ..step import Refused
    from ..target import Target

    src = Source.open(path)
    # the name a request sends, as `qda build`'s serve line prints it: only a checkpoint whose record plans
    # a vLLM server has one (another runtime, or another tool's checkpoint, is served under no name of ours)
    serve = (src.jsons.get("build.json") or {}).get("serve")
    named = {"serve_name": served_name(src.where)} if isinstance(serve, dict) else {}
    rep = {"path": src.where, **named, **detect(src)}
    fatal, soft = missing_files(src)
    rep["missing"], rep["missing_optional"] = fatal, soft
    here = local_target()
    recorded = (src.jsons.get("build.json") or {}).get("target")
    recorded = recorded if isinstance(recorded, str) else None  # a record with no target, or not a name
    notes: list[str] = []
    if target is not None:  # an explicit target is the one to judge, whatever card and record there are
        t = Target.parse(target)
    elif here is not None:  # the card here is the one the user would serve on: judge it, not the record
        t = here
        if recorded is not None:
            try:
                built = Target.stored(recorded)
            except Refused:  # another version's or tool's spelling: this card is what this report can plan
                pass
            else:
                if (built.arch, built.vllm, built.backend) != (t.arch, t.vllm, t.backend):
                    notes.append(_built_elsewhere_note(built, here))
    else:
        try:
            t = Target.stored(recorded)
        except Refused:  # another version's or tool's spelling: plan on the default and say so, not refuse
            t = Target()
            notes.append(
                f"build.json records target {recorded!r}, which this version cannot plan against; "
                f"checked against the default {t.arch}@{t.vllm}"
            )
        else:
            if recorded is not None:
                notes.append(f"target {t.arch}@{t.vllm}, as build.json records it")
    rep["target"] = f"{t.arch}@{t.vllm}"
    rep["kernels"], rep["notes"] = {}, notes
    if here is not None:
        rep["card"] = here.arch  # which card the serve check below loads on, when this process sees one
    rep["reasons"] = [f"missing {m}" for m in fatal]
    checkpoint_format(src).serve(src, rep, t)
    if rep["layout"] == "legacy":
        rep["notes"].append("legacy research layout: not supported")
    return rep


def _text(rep: dict[str, Any]) -> str:
    lines = [f"{rep['path']}: {rep['format']} (layout {rep['layout']})"]
    for name, g in rep["groups"].items():
        f = int_format(g)
        what = f.name if f else g["scheme"]
        tg = ", ".join(g["targets"][:3]) + (f" (+{len(g['targets']) - 3})" if len(g["targets"]) > 3 else "")
        lines.append(
            f"  group {name}: {what} on {tg}" + (f", actorder {g['actorder']}" if g["actorder"] else "")
        )
    kept = rep["kept_bf16"]
    if kept:
        lines.append(f"  kept bf16: {len(kept)} modules, e.g. {', '.join(kept[:3])}")
    lines.append(f"  experts: {'yes' if rep['experts'] else 'no'}; g_idx: {'yes' if rep['g_idx'] else 'no'}")
    for m in rep["missing"]:
        lines.append(f"  missing: {m}")
    for m in rep["missing_optional"]:
        lines.append(f"  missing (optional): {m}")
    for n in rep["notes"]:
        lines.append(f"  note: {n}")
    if "serve_name" in rep:  # what a request's `model` field sends, as `qda build`'s serve line names it
        lines.append(f"  serve name: {rep['serve_name']}")
    if rep["reasons"]:
        lines.append(f"  not servable as is on {rep['target']}:")
        lines += [f"    - {r}" for r in rep["reasons"]]
    else:
        ks = ", ".join(f"{k} {v}" for k, v in rep["kernels"].items())
        checked = ks or ("bf16 kernels" if rep["format"] in ("bf16", "fp16", "fp32") else "")
        verdict = (
            f"servable as is on {rep['target']}: {checked}"
            if checked
            else "files complete; kernels not checked"
        )
        lines.append(f"  {verdict}")
    if (serve := rep.get("serve")) is not None:
        # the verdict above is about the checkpoint's target; the engine loads here, on the card this process sees
        where = f" on this card ({rep['card']})" if rep.get("card") else ""
        if serve["error"]:
            lines.append(f"  serve check{where}: vLLM could not load it: {serve['error']}")
        if serve["unconsumed"]:
            lines.append(f"  serve check{where}: vLLM never read {len(serve['unconsumed'])} tensor(s):")
            lines += [f"    - {n}" for n in serve["unconsumed"]]
        if serve["dropped"]:
            lines.append(
                f"  serve check{where}: not used by vLLM's model class: {len(serve['dropped'])} tensor(s):"
            )
            lines += [f"    - {n}" for n in serve["dropped"]]
        if not serve["error"] and not serve["unconsumed"] and not serve["dropped"]:
            lines.append(f"  serve check{where}: vLLM read every tensor")
    return "\n".join(lines)
