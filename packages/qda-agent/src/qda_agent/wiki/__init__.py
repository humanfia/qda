"""Install the `quant-methods` skill and generate pages from the available quantization steps.

`skill/quant-methods/` contains the index, calibration guide, mechanism cards and phenomena pages. They
describe mechanisms and starting points without model results, scores or benchmark names.

Two pages are generated from the installed `qda_quant` step registry:

- `references/capability.md`: steps implementing each card and steps with no card, using `cards.toml`;
- `references/stacking.md`: step kinds, required statistics and valid recipe ordering.
"""

from __future__ import annotations

import logging
import shutil
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["capability", "gaps", "install", "load_cards", "skill_source", "stacking", "steps"]

#: The skill's name (its directory and `SKILL.md`'s `name:`).
NAME = "quant-methods"


def skill_source() -> Path:
    """Locate the packaged skill tree containing the authored pages."""
    return Path(str(resources.files(__package__).joinpath("skill", NAME)))


def load_cards() -> dict[str, Any]:
    """Load `cards.toml` mappings for card implementations, notes and steps without cards."""
    return tomllib.loads(resources.files(__package__).joinpath("cards.toml").read_text())


def steps() -> dict[str, type]:
    """Return installed step classes keyed by recipe name."""
    from qda_quant.step import REGISTRY

    return {k: REGISTRY[k] for k in REGISTRY}


def _summary(cls: type) -> str:
    return (cls.__doc__ or "").strip().splitlines()[0].rstrip(".") if cls.__doc__ else ""


def gaps(registry: dict[str, type] | None = None) -> tuple[list[str], list[str]]:
    """Return unmapped registered steps and missing steps named in `cards.toml`, both sorted."""
    reg = steps() if registry is None else registry
    data = load_cards()
    named = {s for ids in data["cards"].values() for s in ids} | set(data.get("no_card", {}))
    return sorted(set(reg) - named), sorted(named - set(reg))


def capability(registry: dict[str, type] | None = None) -> str:
    """Render `references/capability.md` from card mappings and the installed step registry.

    List unmapped steps using their docstrings' first lines and omit mapped steps that are unavailable.
    Log a warning for either registry mismatch.
    """
    reg = steps() if registry is None else registry
    data = load_cards()
    cards, note, no_card = data["cards"], data.get("note", {}), dict(data.get("no_card", {}))
    unmapped, missing = gaps(reg)
    if unmapped:
        log.warning("wiki: steps with no row in cards.toml, listed as described by no card: %s", unmapped)
    if missing:
        log.warning("wiki: cards.toml names steps that are not installed, left out: %s", missing)
    no_card = {s: w for s, w in no_card.items() if s in reg} | {
        s: _summary(reg[s]) or "no description" for s in unmapped
    }
    lines = [
        "# capability",
        "",
        "Which cards this library implements, and by which `qda build` steps (generated from the installed",
        "library). The card describes the mechanism either way; a step's settings are in `qda plan` and its",
        "docstring. How steps stack: [stacking](stacking.md).",
        "",
        "| card | implemented in this library |",
        "|---|---|",
    ]
    for card, ids in sorted(cards.items()):
        have = [s for s in ids if s in reg]
        status = note.get(card) or (
            ", ".join(f"`{s}`" for s in have) if have else "no implementation in this library"
        )
        lines.append(f"| [{card}](methods/{card}.md) | {status} |")
    if no_card:
        lines += ["", "Steps no card describes:", ""]
        lines += [f"- `{s}`: {what}" for s, what in sorted(no_card.items())]
    return "\n".join(lines) + "\n"


#: The order a recipe runs step kinds in, and what each kind is.
KINDS = (
    ("model", "sees the whole float model before the layer pass (rotations, permutations, token weights)"),
    ("layer", "runs inside the engine's one pass over the layers, on the statistics it asks for"),
    (
        "checkpoint",
        "reads the packed checkpoint the earlier steps wrote and writes a new one of the same format",
    ),
)


def stacking(registry: dict[str, type] | None = None) -> str:
    """Render `references/stacking.md` from the step classes' `kind`, `needs` and `preserves_quantized`."""
    reg = steps() if registry is None else registry
    lines = [
        "# stacking",
        "",
        "A recipe is the list `steps:` of a `qda build` config, run in order. Generated from the installed library.",
        "",
        "## Kinds",
        "",
    ]
    for kind, what in KINDS:
        names = ", ".join(f"`{k}`" for k, c in sorted(reg.items()) if c.kind == kind) or "none"
        lines.append(f"- **{kind}** steps {what}: {names}.")
    late = [k for k, c in sorted(reg.items()) if c.kind == "model" and c.preserves_quantized]
    lines += [
        "",
        "## Order",
        "",
        "Model steps come first, then layer steps, then checkpoint steps. A model step may follow a layer step "
        "only when it changes no weight a layer step has quantized"
        + (f" ({', '.join(f'`{k}`' for k in late)})" if late else "")
        + ". Layer steps run in the order listed, inside one pass: each layer is finished by every layer step "
        "before the next layer starts. A checkpoint step takes the packed output of everything before it, so "
        "it can also refine a checkpoint built earlier (give that checkpoint as the build's `model`). The format "
        "is set once for the whole recipe.",
        "",
        "## Steps",
        "",
        "| step | kind | what it is | statistics it asks for |",
        "|---|---|---|---|",
    ]
    for k, c in sorted(reg.items()):
        needs = c.needs
        if not isinstance(needs, frozenset):
            try:  # a step whose needs depend on its settings: read the defaults
                needs = frozenset(c().needs)
            except Exception:  # noqa: BLE001  (a step that cannot be built from defaults shows '-')
                needs = frozenset()
        lines.append(f"| `{k}` | {c.kind} | {_summary(c)} | {', '.join(sorted(needs)) or '-'} |")
    return "\n".join(lines) + "\n"


def install(dst: Path, registry: dict[str, type] | None = None) -> Path:
    """Copy the skill to `dst`, generate its capability and stacking pages, and return `dst`."""
    shutil.copytree(skill_source(), dst)
    ref = dst / "references"
    (ref / "capability.md").write_text(capability(registry))
    (ref / "stacking.md").write_text(stacking(registry))
    return dst
