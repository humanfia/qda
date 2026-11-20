"""The ``formats:`` recipe: one mapping from a class pattern to the format a build stores it in.

A build config may spell its per-unit formats as one mapping::

    formats:
      default: w4a16-g128
      lm_head: bf16
      layers.0-3.mlp: w8a16-g32

It is sugar for the two keys the planner already takes: `format_overrides` (every quantized value) and
`ignore` (every ``bf16`` value), with ``default`` naming the body ``format``. The compilation is purely
syntactic -- it needs no model -- so a build, a recipe writer and the judge's bpw tool all read the same
key the same way.

Three key forms, no others:

- an exact module name as ``qda plan`` prints it (``model.layers.0.mlp.down_proj``);
- ``re:`` and a regex matched from the start of the name (the rule of `ignore`);
- ``layers.<a>-<b>[.<suffix>]``: any module under ``layers.<n>.`` for ``a <= n <= b``, optionally under
  ``.<suffix>`` (``layers.0-3.mlp`` matches ``model.layers.2.mlp.down_proj``, not ``model.layers.4.mlp``).

Values are a registered format name (`qda_quant.format.Format.parse`) or ``bf16``, which keeps the unit
as it is stored. ``default: bf16`` is refused: it would leave nothing to quantize. A ``bf16`` entry may name
the output head: no layer step quantizes it, so the entry states what the build already does, and a config
whose steps include a head step is refused at plan time (the step stores the head, the entry keeps it bf16).

A ``bf16`` entry wins over a format entry that also names the same unit: the compiled ``ignore`` removes the
unit before the compiled ``format_overrides`` resolve, the order the planner applies those keys in. That is
what lets ``layers.0-3.mlp: w8a16-g32`` beside ``layers.0-3.mlp.down_proj: bf16`` say "the mlp in W8 except
down_proj". The opposite request works when the format entry names only the unit it quantizes: a rule whose
every match is kept bf16 is refused (the planner names the ``ignore`` entry), and a rule that also reaches
quantized units applies to those alone. Two format entries that name one unit differently are refused, as
they are for an override list.
"""

from __future__ import annotations

import re

from .format import Format

DEFAULT = "default"
BF16 = "bf16"
#: ``layers.<a>-<b>`` with an optional ``.<suffix>`` after it; a key with any other prefix is an exact name.
_RANGE = re.compile(r"layers\.(\d+)-(\d+)(?:\.(.+))?\Z")


def _pattern(key: str) -> str:
    """The `format_overrides`/`ignore` pattern a recipe key compiles to."""
    m = _RANGE.match(key)
    if m is None:
        return key
    lo, hi, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    if lo > hi:
        raise ValueError(f"formats: {key!r}: the layer range runs {lo}-{hi}; write the lower index first")
    layers = "|".join(str(n) for n in range(lo, hi + 1))
    tail = rf"\.{re.escape(suffix)}(\.|$)" if suffix else r"\."
    return rf"re:(.*\.)?layers\.({layers}){tail}"


def compile_formats(recipe: object) -> tuple[str | None, list[dict[str, str]], list[str]]:
    """Compile a ``formats:`` mapping to ``(body format, format_overrides rules, ignore patterns)``.

    Raises:
        ValueError: the mapping is not a mapping of strings, a key is not one of the three forms, a value
            is neither a format nor ``bf16``, ``default`` is ``bf16``, or the mapping is empty.
    """
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError(
            "formats: expected a non-empty mapping of patterns to formats, e.g. "
            "{default: w4a16-g128, layers.0-3.mlp: w8a16-g32}"
        )
    body: str | None = None
    rules: list[dict[str, str]] = []
    ignores: list[str] = []
    for key, name in recipe.items():
        if not isinstance(key, str) or not isinstance(name, str):
            raise ValueError(f"formats: {key!r}: {name!r}: patterns and formats are strings")
        pattern = _pattern(key)
        if name.strip().lower() == BF16:
            if key == DEFAULT:
                raise ValueError(
                    "formats: default: bf16 leaves nothing to quantize; name the body format, e.g. "
                    "default: w4a16-g128, and bf16 only the units that stay as they are"
                )
            ignores.append(pattern)
            continue
        try:
            canonical = Format.parse(name).name
        except ValueError as e:
            raise ValueError(f"formats: {key!r}: {e}") from None
        if key == DEFAULT:
            body = canonical
        else:
            rules.append({"pattern": pattern, "format": canonical})
    return body, rules, sorted(set(ignores))
