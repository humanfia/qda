"""The end-of-run summary card: what a run's checkpoint is, in one block three commands print the same way.

`qda-agent prepare`, `qda-agent search` and `qda-agent evaluate` end with it. Each line is a fact that run
measured: the format(s) the checkpoint's own config stores (the judge's reader,
`qda_agent.judge.checks.stored_formats`), its size against the original's weights, its bits per quantized weight
(the judge's own rule, `qda_agent.judge.checks.budget`) and the KL the run read for it. A fact the run did not
measure is left out rather than filled in from the config or an earlier run: nothing here measures decode
throughput, so no card carries a decode line, and `prepare`, which builds candidate 0 and reads no reference,
prints no KL.

The block is the same on a terminal and in a redirected log: no colour, no carriage returns, one fact to a
line, ASCII only. It does not end with the `vllm serve` line the interface spec has: that line belongs to the
one function that knows which kernel serves a checkpoint
(`qda_quant.output.serve_command.serve_command_line`), which each caller reaches through
`qda_agent.host.serve_line` and prints once, after the card. A second renderer here would print a command
whose engine is not the one the KL above it was read on.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from qda_agent.judge.checks.formats import Stored

__all__ = ["card", "of_run"]

#: The block's shape: its indent, the width of the label column, and a rule as wide as a full line.
INDENT = "  "
LABEL = 14
RULE = "-" * 58


def card(
    checkpoint: Path,
    *,
    original: Path | None = None,
    kl: float | None = None,
    head_lease: bool = False,
) -> str:
    """The summary card of one checkpoint: its measured facts, one to a line; "" when it has none.

    Args:
        checkpoint: The checkpoint a run made or picked. A path that is not there (a stand-in build) has no
            fact to state.
        original: The unquantized model the size and the bpw are read against: the original's weights
            (`judge.anchor`), when it is a directory here. None leaves both lines out.
        kl: The KL this run read for `checkpoint`, or None when this run read none.
        head_lease: The run's `judge.head_lease`. On: the bpw is the judge's priced number, the output head
            charged 16 bits a weight and its savings lent to the body; off: the body's own.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return ""
    lines = []
    if name := _format(checkpoint):
        lines.append(_line("format", name))
    if original is not None and (bits := _bpw(checkpoint, original, head_lease)) is not None:
        lines.append(_line("bpw", f"{bits:.2f}"))
    if (size := _weights_bytes(checkpoint)) is not None:
        lines.append(_line("size", _size(size, _weights_bytes(original))))
    if kl is not None:
        lines.append(_line("KL", f"{kl:.4f}  vs the original"))
    if not lines:
        return ""
    return "\n".join([f"{INDENT}{RULE}", *lines, f"{INDENT}{RULE}"]) + "\n"


def of_run(run_dir: Path) -> str:
    """The card of a finished run: the checkpoint it picked, with the KL that run read for it.

    Nothing at all for a run that picked no checkpoint, and no KL line when no record of the run holds a
    reading of the pick itself: a submission the judge never read is scored on the holdout prompts by
    `qda-agent evaluate`, which prints its own card.
    """
    from qda_agent.evaluate import winner  # at call time: `evaluate` imports this module

    try:
        picked = winner(run_dir)
    except ValueError:  # no FINAL.json, or it names no checkpoint
        return ""
    judge = _doc(run_dir / "resolved.yaml").get("judge") or {}
    anchor = judge.get("anchor")
    return card(
        picked,
        original=Path(anchor) if anchor and Path(anchor).is_dir() else None,
        kl=_kl(_doc(run_dir / "FINAL.json"), picked),
        head_lease=bool(judge.get("head_lease")),
    )


def _line(label: str, value: str) -> str:
    """One fact, in the label column."""
    return f"{INDENT}{label:<{LABEL}}{value}"


def _format(checkpoint: Path) -> str | None:
    """What the checkpoint stores: one format name, or every name it mixes (`w4a16-sym-g32 + w8a16-sym-g32`).

    The names come from the checkpoint's own config, read through the judge's reader, so a build that overrode
    some modules (`format_overrides`) is not reported as its default alone. A `.gguf`, a stand-in, or a config
    that states no format the box can name falls back to the name its build record carries.
    """
    names = _stored_names(checkpoint)
    if names:
        return " + ".join(names)
    record = _doc(checkpoint / "build.json")
    if gguf := record.get("gguf"):
        mix = gguf.get("mix")
        return str(mix) if mix else None
    name = record.get("format")
    return str(name) if name else None


def _stored_names(checkpoint: Path) -> list[str]:
    """The quantized formats the checkpoint's config stores, named and sorted; [] when it states none.

    A module the config keeps bf16 is not a stored format: `qda inspect` reports those apart from the format.
    A format with no spelling of its own (`_format_name` returns None) is left out rather than misnamed.
    """
    from qda_agent.judge.checks import stored_formats

    try:
        stored = stored_formats(checkpoint)
    except (ValueError, OSError):  # not a checkpoint this reader takes, or one it refuses
        return []
    quantized = [fmt for fmt in stored.values() if fmt.weight_bits is not None]
    if not quantized or any(fmt.group_size is None or fmt.symmetric is None for fmt in quantized):
        return []
    return sorted({name for fmt in quantized if (name := _format_name(fmt)) is not None})


def _format_name(stored: Stored) -> str | None:
    """A stored format as the name a config spells it, one `Format.parse` takes back to the same format.

    Most families spell group and symmetry (`w4a16-asym-g128`). Three have a name of their own, and the
    integer spelling those would get names another format or none: FP8 W8A8 (`w8a8-fp8`), INT8 W8A8
    (`w8a8i-sym-pc`) and NVFP4 (`w4a4-nvfp4`). A float weight-only format (FP8 weights, MXFP4) has no
    spelling that parses back to it -- its integer spelling would name an integer format -- so it has none.
    """
    if stored.weight_type == "float" and stored.act is None:
        return None
    act = stored.act[0] if stored.act is not None else None
    fp8 = stored.weight_type == "float" and stored.act is not None and stored.act[:2] == (8, "float")
    if fp8 and stored.weight_bits == 8:
        return "w8a8-fp8"
    if act == 4:
        return "w4a4-nvfp4"
    group = "pc" if stored.group_size == -1 else f"g{stored.group_size}"
    symmetry = "sym" if stored.symmetric else "asym"
    if stored.weight_bits == 8 and act == 8:
        return f"w8a8i-{symmetry}-{group}"
    return f"w{stored.weight_bits}a{'8' if act == 8 else '16'}-{symmetry}-{group}"


def _weights_bytes(path: Path | None) -> int | None:
    """The weights a checkpoint or a model stores: its safetensors shards, a `.gguf`, or one file's bytes.

    A path that is not there, or a file this process cannot read, states no size: the card of a finished
    command is never the reason that command fails.
    """
    if path is None:
        return None
    if path.is_file():
        return _bytes(path) or None
    if not path.is_dir():
        return None
    for stored in ("*.safetensors", "*.gguf"):
        if total := sum(_bytes(f) for f in path.glob(stored)):
            return total
    return None


def _bytes(path: Path) -> int:
    """A file's bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _size(size: int, original: int | None) -> str:
    """`4.9 GB`, and what share of the original's weights that is when they are known."""
    gb = f"{size / 1e9:.1f} GB"
    if not original:
        return gb
    return f"{gb}{INDENT}({100 * size / original:.0f}% of the original's {original / 1e9:.1f} GB)"


def _bpw(checkpoint: Path, original: Path, head_lease: bool) -> float | None:
    """The checkpoint's bits per quantized weight by the judge's own rule, or None when it has none to price.

    The number is the body's, or with `judge.head_lease` on the judge's after the lease (`leased_bpw`): the
    output head is charged 16 bits a weight and its savings are lent to the body. Both ways read the body's
    own `bpw` first, so an original with no decoder Linear states no number either way rather than dividing
    by zero. A `.gguf` and a directory of anything but packed safetensors have no decoder Linears to price,
    and a checkpoint this rule cannot read states no number at all.
    """
    from qda_agent.judge.checks import CheckpointError  # at call time: the judge imports the scheduler
    from qda_agent.judge.checks.budget import bpw, denominator, head_weights, leased_bpw

    if not checkpoint.is_dir() or not any(checkpoint.glob("*.safetensors")):
        return None
    try:
        body = bpw(checkpoint, original)
        if not head_lease:
            return body
        return leased_bpw(checkpoint, denominator(original), head_weights(original))
    except (CheckpointError, OSError):
        return None


def _kl(final: dict[str, Any], picked: Path) -> float | None:
    """The `kl` the run's own records hold for `picked`: its submission, a baseline reading, or the ledger.

    An exported pick is found under the checkpoint the export copied as well (`export.source`): the copy is
    never read, and a kept baseline keeps its reading under the name it had before the export renamed the pick.
    """
    names = {str(picked)}
    export = final.get("export") or {}
    if export.get("winner") == str(picked) and export.get("source"):
        names.add(str(export["source"]))
    records = [final.get("choice"), *(final.get("baseline_readings") or []), *(final.get("ledger") or [])]
    for record in records:
        if record and str(record.get("dir")) in names:
            kl = (record.get("metrics") or {}).get("kl")
            if isinstance(kl, int | float):
                return float(kl)
    return None


def _doc(path: Path) -> dict[str, Any]:
    """A record a run wrote (JSON, which YAML reads), or `{}` when it is not there or is not an object."""
    try:
        got = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return got if isinstance(got, dict) else {}
