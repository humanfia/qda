"""The serving engine's format facts of a checkpoint: its quantization config, fused units and activation formats.

Moved here from `checks.py` (pure move): a module's declared format (`Stored`), the menu gate (`format_key`,
`DEFAULT_MENU`) and the fused-unit and per-layer activation rules (`format_problem`). `checks` re-exports the
public names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qda_quant.format import W8A8_FP8_FORMAT, Format

from .tensors import (
    AWQ,
    FUSED,
    GATE_UP,
    QKV,
    STORED,
    TIED_EMBEDDING,
    _json,
    _module_tensors,
    _modules,
    inventory,
)

#: The formats a task's tier admits, by name: the W4 tier's menu. A W3 search adds "w3a16".
DEFAULT_MENU = ("bf16", "w8a16", "w4a16")


@dataclass(frozen=True)
class Stored:
    """The format one quantization config group declares: its weights and, when it has any, activations."""

    weight_bits: int | None
    group_size: int | None
    symmetric: bool | None
    weight_type: str | None
    strategy: str | None
    act: tuple[int | None, str | None, bool | None] | None

    @property
    def family(self) -> str:
        """The menu key of this format: `w<weight bits>a<activation bits>`, or `bf16` for a float module.

        Float 8-bit weights and activations are the FP8 W8A8 family (`w8a8-fp8`), whose own key keeps it apart
        from the INT8 W8A8 family (spelled `w8a8i`, keyed `w8a8`).
        """
        if self.weight_bits is None:
            return "bf16"
        if (
            self.weight_bits == 8
            and self.weight_type == "float"
            and self.act is not None
            and self.act[:2] == (8, "float")
        ):
            return W8A8_FP8_FORMAT
        bits = "?" if self.act is None or self.act[0] is None else str(self.act[0])
        return f"w{self.weight_bits}a{'16' if self.act is None else bits}"

    def __str__(self) -> str:
        """The family, then the group and symmetry when the format is quantized."""
        if self.weight_bits is None:
            return "bf16"
        group = "channel" if self.group_size == -1 else f"g{self.group_size}"
        return f"{self.family} ({group}, {'sym' if self.symmetric else 'asym'})"


#: A module no quantization group claims (or one its config's `ignore` list keeps float): stored in bf16.
BF16 = Stored(None, None, None, None, None, None)


def format_key(name: str) -> str:
    """The menu key a format spelling names: `bf16`, `w4a16`, `w8a16`, `w3a16`, `w4a8`, `w4a4`, `w8a8-fp8`.

    A float-8 weight family keeps its own name (`w8a8-fp8`), which is not the INT8 W8A8 family's key (`w8a8`,
    spelled `w8a8i`).

    Raises:
        ValueError: `name` spells no format (`qda_quant.format.Format.parse` refuses it).
    """
    if name.strip().lower() == "bf16":
        return "bf16"
    f = Format.parse(name)
    if f.weight_type == "fp8":
        return f.name
    return f"w{f.weight_bits}a{f.act_bits}"


def _quantization_config_problem(cfg: dict) -> str | None:
    """Why `cfg`'s `quantization_config` is not a set of format groups an engine can read, or None.

    A candidate's `config.json` is the agent's to write, so its quantization config is untrusted: admission
    names a malformed one instead of raising inside the first read. A config without `config_groups` (an
    engine's own, such as autoawq's) is not malformed: its global `bits` gives every module it packs a format
    (`_global_stored`), and every other module reads as bf16.
    """
    q = cfg.get("quantization_config")
    if q is None:
        return None
    if not isinstance(q, dict):
        return "quantization_config is not a layout (not an object)"
    groups = q.get("config_groups")
    if groups is not None and not isinstance(groups, dict):
        return "quantization_config is not a layout (config_groups is not a name -> group map)"
    for name, g in (groups or {}).items():
        where = f"config_groups[{name!r}]"
        if not isinstance(g, dict):
            return f"quantization_config is not a layout ({where} is not an object)"
        w = g.get("weights")
        if w is not None and not isinstance(w, dict):
            return f"quantization_config is not a layout ({where}.weights is not an object)"
        a = g.get("input_activations")
        if a is not None and not isinstance(a, dict):
            return f"quantization_config is not a layout ({where}.input_activations is not an object)"
        t = g.get("targets")
        if t is not None and (not isinstance(t, list) or not all(isinstance(x, str) for x in t)):
            return f"quantization_config is not a layout ({where}.targets is not a list of names)"
        for key in ("num_bits", "group_size"):
            v = (w or {}).get(key)
            if v is not None and type(v) is not int:
                return f"quantization_config is not a layout ({where}.weights.{key} is not an integer)"
    ignore = q.get("ignore")
    if ignore is not None and (not isinstance(ignore, list) or not all(isinstance(x, str) for x in ignore)):
        return "quantization_config is not a layout (ignore is not a list of names)"
    return None


def _stored(g: dict) -> Stored:
    """The format a config group declares; a field that is not of its type stays None, as the engine reads it."""
    w = g.get("weights")
    w = w if isinstance(w, dict) else {}
    a = g.get("input_activations")
    act = None
    if isinstance(a, dict):
        bits = a.get("num_bits")
        act = (
            bits if type(bits) is int else None,
            a.get("type") if isinstance(a.get("type"), str) else None,
            a.get("dynamic") if isinstance(a.get("dynamic"), bool) else None,
        )
    bits, group = w.get("num_bits"), w.get("group_size")
    return Stored(
        weight_bits=bits if type(bits) is int else None,
        group_size=-1 if group is None else group if type(group) is int else None,
        symmetric=w.get("symmetric") if isinstance(w.get("symmetric"), bool) else None,
        weight_type=w.get("type") if isinstance(w.get("type"), str) else None,
        strategy=w.get("strategy") if isinstance(w.get("strategy"), str) else None,
        act=act,
    )


def _declared(cfg: dict) -> list[tuple[list[str], Stored]]:
    """The config groups in order: the targets each claims and the format it declares.

    A group without `targets` is the fallback over every Linear, as compressed-tensors reads it. Order
    matters to `_format_of`: an explicit name or `re:` match wins over the "Linear" fallback even when the
    fallback's group comes first, as qda's writer orders them.
    """
    q = cfg.get("quantization_config")
    groups = q.get("config_groups") if isinstance(q, dict) else None
    if not isinstance(groups, dict):
        return []
    out = []
    for g in groups.values():
        if isinstance(g, dict):
            t = g.get("targets")
            targets = [x for x in t if isinstance(x, str)] if isinstance(t, list) else ["Linear"]
            out.append((targets, _stored(g)))
    return out


def _global_stored(cfg: dict) -> Stored | None:
    """The format an engine's own quantization config gives every module it packs (autoawq's), or None.

    autoawq writes no `config_groups`: its `bits`, `group_size` and `zero_point` describe every module it
    stores packed (`qweight`, `qzeros`, `scales`), and its zero points make the scheme asymmetric.
    """
    q = cfg.get("quantization_config")
    if not isinstance(q, dict) or isinstance(q.get("config_groups"), dict):
        return None
    bits = q.get("bits", q.get("w_bit"))
    if type(bits) is not int:
        return None
    group = q.get("group_size", q.get("q_group_size"))
    zero = q.get("zero_point")
    return Stored(
        weight_bits=bits,
        group_size=-1 if group is None else group if type(group) is int else None,
        symmetric=(not zero) if isinstance(zero, bool) else None,
        weight_type=None,
        strategy=None,
        act=None,
    )


def _claimed(target: str, module: str) -> bool:
    """Whether a config target's name or `re:` pattern claims `module`."""
    if not target.startswith("re:"):
        return target == module
    try:
        return re.match(target[3:], module) is not None
    except re.error:
        return False


def _ignored(ignore: Any, module: str) -> bool:
    """Whether a config's `ignore` list keeps `module` bf16, as the engines read it.

    compressed-tensors and vLLM match a plain entry exactly, SGLang also as a dotted suffix, and every
    engine matches `re:` from the start (`qda_quant.output.ct_config._ignore_matches`).
    """
    if not isinstance(ignore, list):
        return False
    for pattern in ignore:
        if not isinstance(pattern, str):
            continue
        if pattern.startswith("re:"):
            try:
                if re.match(pattern[3:], module):
                    return True
            except re.error:
                continue
        elif module == pattern or module.endswith("." + pattern):
            return True
    return False


def _probe_names(module: str) -> list[str]:
    """The names a config's targets are matched against for a module: itself, or a bank's first experts.

    A fused bank's stored tensor is written for its experts, so its config groups name the leaves they probe
    (`<bank>.experts.0.gate_proj`, `...up_proj`, `...down_proj`); the fused name itself is checked last.
    """
    for suffix, leaves in FUSED:
        if module.endswith(suffix):
            bank = module[: -len(suffix)]
            return [f"{bank}.experts.0.{leaf}" for leaf in leaves] + [module]
    return [module]


def _claimed_format(name: str, declared: list[tuple[list[str], Stored]]) -> Stored:
    """The format a module name resolves to: its first explicit target, else the "Linear" fallback; bf16 if none."""
    fallback = None
    for targets, fmt in declared:
        for target in targets:
            if target == "Linear":
                fallback = fmt if fallback is None else fallback
            elif _claimed(target, name):
                return fmt
    return BF16 if fallback is None else fallback


def _formats(
    modules: Iterable[str],
    stored: set[str],
    declared: list[tuple[list[str], Stored]],
    ignore: Any,
    default: Stored | None = None,
) -> tuple[dict[str, Stored], str | None]:
    """Each module's stored format from its quantization config, and why one has none, or None.

    A module a config's `ignore` list names (or one of a fused bank's probed leaves it names) is bf16. Any
    other module that stores packed codes must be declared by a config group or by the config's own global
    format (autoawq's `bits`): one that is neither would be read as bf16 and sized on no menu, so it is
    refused rather than admitted on a format no engine was told.
    """
    out: dict[str, Stored] = {}
    for module in modules:
        names = _probe_names(module)
        probed = names[:-1] if len(names) > 1 else names
        if any(_ignored(ignore, name) for name in probed):
            out[module] = BF16
            continue
        found = [_claimed_format(name, declared) for name in probed]
        if any(fmt != found[0] for fmt in found[1:]):
            return out, (
                f"{module} is one stored tensor, but its config gives {probed[0]} {found[0]} and "
                f"{probed[1]} {found[1]}; a fused projection is stored and served in one format"
            )
        if found[0].weight_bits is None and len(names) > 1:
            own = _claimed_format(names[-1], declared)  # a config that names the fused module itself
            found = [own] if own.weight_bits is not None else found
        if found[0].weight_bits is None and module in stored:
            if default is None:
                return out, f"{module} stores packed weights its quantization_config does not declare"
            found = [default]
        out[module] = found[0]
    return out, None


def _layer(module: str) -> str | None:
    """The layer a module belongs to: its path through the first integer component; None outside layers."""
    parts = module.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return None


def _expert_bank(module: str) -> str | None:
    """The routed expert bank a module belongs to: the path ending at `.experts`."""
    m = re.match(r"^(.+\.experts)\.\d+\.", module)
    if m is not None:
        return m.group(1)
    for suffix, _ in FUSED:
        if module.endswith(suffix):
            return module[: -len(suffix)] + ".experts"
    return None


def _units(modules: Iterable[str]) -> list[tuple[str, list[str]]]:
    """Each unit the serving engine fuses into one kernel: (what it is, its modules, sorted).

    q, k and v of one attention, gate and up of one MLP, and every expert projection of one routed expert
    bank are each fused, so the members must agree on one stored format.
    """
    found: dict[tuple[str, str, str], list[str]] = {}
    labels: dict[tuple[str, str, str], str] = {}
    for module in modules:
        layer = _layer(module)
        if layer is None:
            continue
        parent, _, leaf = module.rpartition(".")
        bank = _expert_bank(module)
        if bank is not None:
            key, label = (layer, "expert bank", bank), "routed expert bank"
        elif leaf in QKV:
            key, label = (layer, "q/k/v", parent), "fused q/k/v"
        elif leaf in GATE_UP:
            key, label = (layer, "gate/up", parent), "fused gate/up"
        else:
            continue
        found.setdefault(key, []).append(module)
        labels[key] = label
    return [(labels[key], sorted(m)) for key, m in found.items()]


def _menu_problem(formats: dict[str, Stored], menu: Sequence[str]) -> str | None:
    """Why a quantized module's format is outside the task's menu, or None.

    A menu entry admits its family, group and symmetry; an entry that spells no symmetry (``w4a16``) admits
    either. A config's menu names every symmetry (`RunConfig` binds bare entries to the row), so the gate
    checks symmetry as it checks the group.
    """
    try:
        allowed = set()
        for name in menu:
            if format_key(name) == "bf16":
                continue
            f = Format.parse(name)
            allowed.add((format_key(name), f.group_size, None if f.symmetric_defaulted else f.symmetric))
    except ValueError as e:
        return f"judge.format_menu: {e}"
    for module in sorted(formats):
        fmt = formats[module]
        if fmt.weight_bits is None:
            continue
        if not {(fmt.family, fmt.group_size, fmt.symmetric), (fmt.family, fmt.group_size, None)} & allowed:
            fact = fmt.family if fmt.family not in {family for family, _, _ in allowed} else str(fmt)
            return f"{module} is {fact}, outside judge.format_menu ({', '.join(menu)})"
    return None


# The dense MLP beside a routed bank, as Qwen (`shared_expert`) and DeepSeek (`shared_experts`) spell it.
_SHARED_EXPERT = re.compile(r"\.shared_experts?\.")

#: The default output-head rule: the head stays in a 16-bit float type, on the menu or not
#: (`judge.head`). An exception model, released with a quantized head, names its release's rule instead.
HEAD_16BIT = "16-bit"
#: The float types a 16-bit output head may use: BF16 or F16, the GGUF names and the safetensors dtypes alike.
HEAD_16BIT_TYPES = ("BF16", "F16")


def _row_problem(formats: dict[str, Stored], mixed: bool = False) -> str | None:
    """Why a checkpoint's quantized modules do not share one group and, per family, one symmetry; or None.

    Group and symmetry are row properties: the agent picks them once, and bit width alone varies by module.
    Experts keep their own symmetry beside the dense modules', since the planner stores routed expert banks,
    and the shared experts beside them when a config quantizes those, symmetric when the row's format spells
    no symmetry (`qda_quant.engine.format_bindings.expert_defaults`). The planner finds a shared expert by tracing the
    model; the judge sees only names, so it knows the two spellings that tracing finds (`_SHARED_EXPERT`), and
    a shared expert spelled otherwise counts as dense. lm_head is outside the rule: it has its own format rule.

    A `format: mixed-precision` checkpoint takes one group per family rather than one for the whole body: each
    group declares its own weights, and the families are stored at the groups their formats have (NVFP4 at
    16, FP8 W8A8 per channel).
    """
    from .budget import LM_HEAD

    quantized = {m: f for m, f in formats.items() if f.weight_bits is not None and not LM_HEAD.search(m)}
    groups = sorted({f.group_size for f in quantized.values()}, key=str)
    if len(groups) > 1 and mixed:
        for family in sorted({f.family for f in quantized.values()}):
            at = sorted({f.group_size for f in quantized.values() if f.family == family}, key=str)
            if len(at) > 1:
                first = {
                    g: next(
                        m for m, f in sorted(quantized.items()) if f.family == family and f.group_size == g
                    )
                    for g in at
                }
                shown = ", ".join(f"{m} at group {g}" for g, m in first.items())
                return f"{family} modules use {len(at)} groups ({shown}); one family takes one group"
    elif len(groups) > 1:
        first = {g: next(m for m, f in sorted(quantized.items()) if f.group_size == g) for g in groups}
        shown = ", ".join(f"{m} at group {g}" for g, m in first.items())
        return f"quantized modules use {len(groups)} groups ({shown}); one checkpoint takes one group"
    seen: dict[tuple[str, str], dict[bool | None, str]] = {}
    for module, f in sorted(quantized.items()):
        kind = "expert" if _expert_bank(module) is not None or _SHARED_EXPERT.search(module) else "dense"
        seen.setdefault((kind, f.family), {}).setdefault(f.symmetric, module)
    for (kind, family), syms in seen.items():
        if len(syms) > 1:
            shown = ", ".join(f"{m} {'sym' if s else 'asym'}" for s, m in syms.items())
            return f"{kind} {family} modules mix symmetries ({shown}); one checkpoint takes one symmetry"
    return None


def _engine_problem(formats: dict[str, Stored], mixed: bool = False) -> str | None:
    """Why a serving-engine fact about fused units or a layer's activations does not hold, or None.

    A layer has one activation format, so the engine runs one activation kernel per layer. A
    `format: mixed-precision` checkpoint is the exception: each config group declares its own activation
    scheme and the engine serves each module with its group's (NVFP4 MLP beside FP8 attention), so only
    fused units still have to agree.
    """
    for label, modules in _units(formats):
        if len(modules) < 2:
            continue
        first = modules[0]
        for module in modules[1:]:
            if formats[module] != formats[first]:
                return (
                    f"{module} is {formats[module]} but {first} is {formats[first]}: the engine fuses them "
                    f"into one {label}, which takes one format"
                )
    if mixed:
        return None
    acts: dict[str, dict[object, str]] = {}
    for module, fmt in formats.items():
        layer = _layer(module)
        if fmt.weight_bits is not None and layer is not None:
            acts.setdefault(layer, {}).setdefault(fmt.act, module)
    for layer, seen in acts.items():
        if len(seen) > 1:
            (a, one), (b, other) = list(seen.items())[:2]
            return f"{layer} mixes two activation formats: {one} activates in {a}, {other} in {b}"
    return None


def _lm_head_problem(
    formats: dict[str, Stored],
    stored: set[str],
    inv: dict[str, tuple[str, tuple[int, ...]]],
    declared: list[tuple[list[str], Stored]],
    cfg: dict,
    source: dict[str, tuple[str, tuple[int, ...]]],
) -> str | None:
    """Why the output head (or a tied model's shared embedding) is not 16-bit, or None.

    Stored tensors decide, even when an `ignore` entry claims a packed head float. The dtype must match
    the source when available; otherwise a 16-bit type (BF16, F16) and F32 are admitted. Quantization
    parameters are refused.
    """
    from .budget import LM_HEAD  # budget imports this module: read here, not when the module loads

    text = cfg.get("text_config")
    tied = (text if isinstance(text, dict) else {}).get(
        "tie_word_embeddings", cfg.get("tie_word_embeddings", False)
    )
    modules = {n.rpartition(".")[0] for n in inv}
    heads = {m for m in modules if LM_HEAD.search(m) or (tied and m.rpartition(".")[2] in TIED_EMBEDDING)}
    for module in sorted(heads):
        fmt = None
        if module in stored:  # packed codes: an ignore entry can hide the format its group gives it
            declared_fmt = formats[module]
            if declared_fmt.weight_bits is None:
                declared_fmt = _claimed_format(module, declared)
            fmt = str(declared_fmt) if declared_fmt.weight_bits is not None else "quantized"
        else:
            weight = inv.get(module + ".weight")
            original = source.get(module + ".weight")
            if original is None and tied:
                original = next(
                    (
                        t
                        for n, t in source.items()
                        if n.endswith(".weight")
                        and (
                            LM_HEAD.search(n.removesuffix(".weight"))
                            or n.removesuffix(".weight").rpartition(".")[2] in TIED_EMBEDDING
                        )
                    ),
                    None,
                )
            if weight is not None and (
                weight[0] not in (*HEAD_16BIT_TYPES, "F32")
                or (original is not None and weight[0] != original[0])
            ):
                fmt = weight[0]
            if fmt is None and any(module + suffix in inv for suffix, _ in (*STORED, *AWQ)):
                fmt = "quantized"
        if fmt is not None:
            return f"lm_head is stored {fmt}; the task keeps it 16-bit ({' or '.join(HEAD_16BIT_TYPES)})"
    return None


def _head_format_problem(
    head_format: str,
    formats: dict[str, Stored],
    inv: dict[str, tuple[str, tuple[int, ...]]],
    cfg: dict,
    source: dict[str, tuple[str, tuple[int, ...]]],
) -> str | None:
    """Why the output head is not stored in exactly `head_format`, or None.

    For publishing checkpoints that are already built in a paper format with a quantized head (the paper's
    W3A16 stores lm_head INT4 g64), and for a search whose fixed map (`judge.format_map`) stores a quantized
    head: there the map's head is the `head_format` (`format_problem`). Without a map a search keeps lm_head
    unquantized (`_lm_head_problem`); no search setting names a `head_format` directly. Only `lm_head` is exempt
    from the menu; a tied head is the input embedding, so a tied model is refused.
    The declaration is not trusted: the head's bits and group are read back from its stored tensors (packed
    codes ``[out, in * bits / 32]`` int32, scales ``[out, in / group]``), with ``in`` from the source's head
    or the config's hidden size. An FP8 head stores its values themselves (an F8_E4M3 weight under an fp32
    scale a row), so it is read back as that shape.
    """
    from .budget import LM_HEAD  # budget imports this module: read here, not when the module loads

    want = Format.parse(head_format)
    text = cfg.get("text_config")
    text = text if isinstance(text, dict) else cfg
    if text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)):
        return f"head_format {want.name} needs an untied lm_head; this model ties it to the embedding"
    heads = sorted({n.rpartition(".")[0] for n in inv if LM_HEAD.search(n.rpartition(".")[0])})
    if not heads:
        return f"no lm_head tensors; head_format stores lm_head in {want.name}"
    for head in heads:
        declared = formats.get(head, BF16)
        if (declared.weight_bits, declared.group_size, declared.symmetric, declared.family) != (
            want.weight_bits,
            want.group_size,
            want.symmetric,
            format_key(want.name),
        ):
            return f"lm_head is declared {declared}; head_format stores lm_head in {want.name}"
        if want.weight_type == "fp8":
            weight, scale = inv.get(f"{head}.weight"), inv.get(f"{head}.weight_scale")
            if (
                weight is None
                or weight[0] != "F8_E4M3"
                or scale is None
                or len(scale[1]) != 2
                or scale[1][1] != 1
            ):
                return (
                    f"lm_head is declared {declared}, but its stored tensors are not an E4M3 weight under an "
                    "fp32 scale per output row"
                )
            continue
        packed, scale = inv.get(f"{head}.weight_packed"), inv.get(f"{head}.weight_scale")
        src = source.get(f"{head}.weight")
        cols = src[1][1] if src is not None and len(src[1]) == 2 else text.get("hidden_size")
        if packed is None or scale is None or not isinstance(cols, int):
            return f"lm_head: no packed codes and scales to read its stored {want.name} back from"
        bits = packed[1][-1] * 32 // cols if packed[0] == "I32" else None
        group = cols // scale[1][-1] if scale[1][-1] else None
        if bits != want.weight_bits or group != want.group_size or cols % want.group_size:
            return (
                f"lm_head is stored {bits}-bit g{group} (read from its tensors), not the declared {want.name}"
            )
        if want.symmetric and f"{head}.weight_zero_point" in inv:
            return f"lm_head stores zero points; {want.name} is symmetric"
    return None


def _lease_head_problem(
    lease: Sequence[str],
    formats: dict[str, Stored],
    stored: set[str],
    inv: dict[str, tuple[str, tuple[int, ...]]],
    declared: list[tuple[list[str], Stored]],
    cfg: dict,
    source: dict[str, tuple[str, tuple[int, ...]]],
) -> str | None:
    """Why the output head is in none of the formats the head lease offers (`lease`), or None.

    A 16-bit head passes as without the lease. An FP8 head is read from its weight's dtype; an integer head
    must declare a lease family, and its bits and group are read back from its tensors (`_head_format_problem`).
    """
    from .budget import LM_HEAD

    if _lm_head_problem(formats, stored, inv, declared, cfg, source) is None:
        return None
    offered = ", ".join(lease)
    heads = sorted({n.rpartition(".")[0] for n in inv if LM_HEAD.search(n.rpartition(".")[0])})
    for head in heads:
        weight = inv.get(f"{head}.weight")
        if weight is not None and head not in stored:
            if not (weight[0].startswith("F8_") and "fp8" in lease):
                return f"lm_head is stored {weight[0]}; the head lease offers {offered}"
            continue
        fmt = formats.get(head, BF16)
        if fmt.weight_bits is None or fmt.family not in lease:
            return f"lm_head is stored {fmt}; the head lease offers {offered}"
        group = "pc" if fmt.group_size == -1 else f"g{fmt.group_size}"
        name = f"{fmt.family}-{'sym' if fmt.symmetric else 'asym'}-{group}"
        if why := _head_format_problem(name, formats, inv, cfg, source):
            return why
    return None


def _effective(formats: dict[str, Stored], stored: set[str]) -> tuple[dict[str, Stored], str | None]:
    """Each module's format as its tensors store it, and why one has none, or None.

    A module with no packed codes is float, whatever a config target claims for its name (an embedding under
    the "Linear" fallback is served bf16). A module with packed codes that its config keeps float (an `ignore`
    entry over packed tensors) has no format an engine serves, so it is refused.
    """
    out: dict[str, Stored] = {}
    for module, fmt in formats.items():
        if module not in stored:
            out[module] = BF16
        elif fmt.weight_bits is None:
            return out, f"{module} stores packed codes, but its quantization_config keeps it float"
        else:
            out[module] = fmt
    return out, None


def _same(a: Stored, b: Stored) -> bool:
    """Whether two stored formats are one: bits, group, symmetry, code type and activations."""
    if a.weight_bits is None or b.weight_bits is None:
        return a.weight_bits is None and b.weight_bits is None
    return (a.weight_bits, a.group_size, a.symmetric, a.weight_type or "int", a.act) == (
        b.weight_bits,
        b.group_size,
        b.symmetric,
        b.weight_type or "int",
        b.act,
    )


def _spelled(fmt: Stored) -> str:
    """The `Format` name of a weight-only stored format: `w4a16-sym-g64`."""
    group = "pc" if fmt.group_size == -1 else f"g{fmt.group_size}"
    return f"w{fmt.weight_bits}a16-{'sym' if fmt.symmetric else 'asym'}-{group}"


def stored_formats(d: Path) -> dict[str, Stored]:
    """The per-module format map checkpoint `d` stores: every module's effective format (`judge.format_map`).

    Read as a hand-in is read: the quantization config's groups, `ignore` list and global format resolved per
    stored module (`_formats`), then each module's tensors (`_effective`), so two configs that spell one
    allocation differently (an `ignore` list with alias spellings) give the same map.

    Raises:
        CheckpointError: A shard cannot be read.
        ValueError: No readable config.json, a malformed quantization config, or a module it gives no format.
    """
    try:
        cfg = _json(d / "config.json")
    except (OSError, ValueError):
        raise ValueError(f"{d}: no readable config.json") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{d}: config.json is not an object")
    if why := _quantization_config_problem(cfg):
        raise ValueError(f"{d}: {why}")
    q = cfg.get("quantization_config")
    inv = inventory(d)
    stored = set(_module_tensors(inv))
    formats, why = _formats(
        _modules(inv),
        stored,
        _declared(cfg),
        q.get("ignore") if isinstance(q, dict) else None,
        _global_stored(cfg),
    )
    out, why = (formats, why) if why else _effective(formats, stored)
    if why:
        raise ValueError(f"{d}: {why}")
    return out


def map_head_format(want: dict[str, Stored]) -> str | None:
    """The `Format` name of the quantized lm_head a format map stores (`w4a16-sym-g64`), or None for a float one."""
    from .budget import LM_HEAD

    return next(
        (_spelled(f) for m, f in sorted(want.items()) if LM_HEAD.search(m) and f.weight_bits is not None),
        None,
    )


def _map_problem(have: dict[str, Stored], want: dict[str, Stored]) -> str | None:
    """Why a checkpoint's per-module formats are not exactly the map's, naming the first module that differs."""
    for module in sorted(have.keys() | want.keys()):
        if module not in have:
            return f"{module} is not stored; judge.format_map stores it {want[module]}"
        if module not in want:
            return f"{module} is stored {have[module]}, and judge.format_map has no such module"
        if not _same(have[module], want[module]):
            return (
                f"{module} is stored {have[module]}; judge.format_map fixes it at {want[module]} "
                "(the per-module allocation is fixed: only the method may change)"
            )
    return None


def map_menu_problem(want: dict[str, Stored], menu: Sequence[str]) -> str | None:
    """Why `menu` is not the formats map `want` stores outside lm_head, or None.

    Every quantized body format of the map must be on the menu, and every quantized menu entry must be one the
    map stores: with a fixed map the menu holds only the map's formats. `bf16` may be listed or not (the menu
    gate never refuses a float module). The map's lm_head is outside the menu, as `head_format`'s is.
    """
    from .budget import LM_HEAD

    body = {m: f for m, f in want.items() if not LM_HEAD.search(m)}
    if why := _menu_problem(body, menu):
        return why
    used = {(f.family, f.group_size, f.symmetric) for f in body.values() if f.weight_bits is not None}
    for name in menu:
        if format_key(name) == "bf16":
            continue
        f = Format.parse(name)
        sym = None if f.symmetric_defaulted else f.symmetric
        if not any(
            family == format_key(name) and group == f.group_size and sym in (None, s)
            for family, group, s in used
        ):
            return (
                f"judge.format_menu offers {name}, which judge.format_map stores in no module outside lm_head: "
                "with a fixed map the menu holds only the map's formats"
            )
    return None


#: Decoder Linear module classes a build may keep bf16 by design (`qda_quant.arch.KEEP_ROLES` and its
#: `out_features < 32` scalar gates): a GDN per-head gate, and a MoE layer's shared experts.
GDN_GATE = re.compile(r"(^|\.)in_proj_(a|b|ba)$")
SHARED_EXPERT = re.compile(r"\.shared_experts?\.")
#: A gate narrower than this is a scalar (a MoE's shared-expert gate), kept bf16 by the planner.
SCALAR_GATE = 32


def _may_stay_float(module: str, out_features: int) -> bool:
    """Whether a decoder Linear may legitimately stay bf16 in a quantized row."""
    return (
        out_features < SCALAR_GATE
        or GDN_GATE.search(module) is not None
        or SHARED_EXPERT.search(module) is not None
    )


def floor_problem(formats: dict[str, Stored], menu: Sequence[str], anchor: Path | None) -> str | None:
    """Why too little of the checkpoint is stored quantized for this row, or None.

    The one owner of "is this build the requested quantization at all", independent of any budget: the share
    of the original's decoder-Linear weights stored quantized must be every weight the format implies. The
    modules a build may keep bf16 by design are outside the count (`_may_stay_float`: scalar gates, GDN
    per-head gates, shared experts), and lm_head, a MoE router and embeddings are never decoder Linears
    (`budget._linear`). A `bf16`-only row asks for no quantization and is not checked.

    Admission applies this only when `judge.budget_bpw` is unset: with a budget the budget is the rule, and a
    body that keeps some modules bf16 is legal if it fits. Without one this is the only check that tells an
    unquantized model from a quantized build.

    Args:
        formats: Each stored module's effective format (`format_problem`'s map).
        menu: The row's formats; a menu that names a quantized format makes the row a quantization row.
        anchor: The original model, whose decoder Linears are the denominator; None: not checked.

    Returns:
        The refusal in one sentence naming the share, the menu and the weights that would stay bf16, or None.
    """
    from . import CheckpointError
    from .budget import decoder_linears

    if anchor is None or not anchor.is_dir():
        return None
    try:
        if not any(format_key(name) != "bf16" for name in menu):
            return None
        linears = decoder_linears(anchor)
    except (CheckpointError, ValueError):
        return None  # the original itself is unreadable: the anchor check reports that
    required = {m: n for m, (n, out) in linears.items() if not _may_stay_float(m, out)}
    total = sum(required.values())
    if not total:
        return None
    quantized = sum(
        n for m, n in required.items() if (fmt := formats.get(m)) is not None and fmt.weight_bits is not None
    )
    if quantized >= total:
        return None
    return (
        f"{quantized / total:.1%} of the decoder-Linear weights are stored quantized: "
        f"{total - quantized} of {total} would stay bf16, and this row quantizes them all "
        f"(judge.format_menu: {', '.join(menu)})"
    )


def format_problem(
    d: Path,
    menu: Sequence[str] = DEFAULT_MENU,
    cfg: dict | None = None,
    *,
    anchor: Path | None = None,
    head_format: str | None = None,
    format_map: Path | None = None,
    head: str = HEAD_16BIT,
    lease: Sequence[str] = (),
    floor: bool = True,
) -> str | None:
    """Why `d` is not in one of the task's formats, or None.

    The rules are the serving engine's, not a reference checkpoint's inventory: every quantized module's
    format is in `menu` (`judge.format_menu`), and q/k/v, gate/up and each routed expert bank share one
    format, group size and symmetry included; a layer has one activation format. Each quantized unit
    uses a menu family at the row's fixed group. Symmetry may differ between whole units. The output head
    stays unquantized in the source dtype in every tier, and so does the input embedding of a model
    that ties it to the head.

    A checkpoint whose config declares `format: mixed-precision` is served group by group instead: it takes
    one group per family (`w4a4-nvfp4` at g16 beside `w8a8-fp8` per channel) and each module activates as its
    own group declares. Fused units still take one stored format.

    Args:
        d: The checkpoint.
        menu: The task's formats, as `format_key` reads a spelling (`DEFAULT_MENU` is the W4 tier's).
        cfg: Its `config.json` when the caller has read it; None reads `d`'s.
        anchor: The source checkpoint whose head dtype is preserved; None admits BF16 or F32.
        head_format: For publishing an already-built paper checkpoint whose lm_head is quantized (a search
            states none: without a map it keeps lm_head unquantized, with one the map's head applies): lm_head, and only lm_head, must be stored in exactly this
            format, read back from its tensors, and is left out of the menu check. None: lm_head unquantized.
        format_map: A checkpoint whose per-module stored formats are the task's fixed allocation
            (`judge.format_map`): every module of `d` must be stored exactly as there (`stored_formats`; the
            first module that differs is named), and lm_head is checked against the map's own head (a
            quantized one read back from its tensors, as `head_format`'s; a float one as the 16-bit rule's).
            Cannot be combined with `head_format`.
        head: The output head's rule (`judge.head`): `16-bit` (the default, `_lm_head_problem`) or, for an
            exception model released with a quantized head, the `Format` name its release stores
            (`_head_format_problem`). `head_format` and a `format_map` outrank it, since each already states
            the head it stores.
        lease: The formats `judge.head_lease` offers the head (`budget.lease_formats`), with the 16-bit
            rule; empty: the head lease is off.

    Raises:
        CheckpointError: A shard cannot be read.
    """
    if cfg is None:
        try:
            cfg = _json(d / "config.json")  # type: ignore[assignment]
        except (OSError, ValueError):
            return "format: no readable config.json"
        if not isinstance(cfg, dict):
            return "format: config.json is not an object"
    if why := _quantization_config_problem(cfg):
        return f"format: {why}"
    declared = _declared(cfg)
    q = cfg.get("quantization_config")
    ignore = q.get("ignore") if isinstance(q, dict) else None
    # compressed-tensors writes the top-level `format` for a checkpoint whose groups differ in format
    declared_mixed = isinstance(q, dict) and str(q.get("format") or "").strip().lower() == "mixed-precision"
    inv = inventory(d)
    stored = set(_module_tensors(inv))
    formats, why = _formats(_modules(inv), stored, declared, ignore, _global_stored(cfg))
    source = inventory(anchor) if anchor is not None else {}
    mapped = None
    if format_map is not None:
        from .budget import LM_HEAD

        if head_format is not None:
            raise ValueError("format_problem: head_format and format_map are two head rules; pass one")
        try:
            want = stored_formats(format_map)
        except ValueError as e:
            return f"format: judge.format_map: {e}"
        have, mapped = _effective(formats, stored)
        mapped = mapped or _map_problem(have, want)
        if (
            spelled := map_head_format(want)
        ) is not None:  # the map's quantized head, read back as head_format's
            head_why = _head_format_problem(spelled, formats, inv, cfg, source)
        else:
            head_why = _lm_head_problem(formats, stored, inv, declared, cfg, source)
        body = {m: f for m, f in formats.items() if not LM_HEAD.search(m)}
    elif head_format is None and head == HEAD_16BIT and lease:
        from .budget import LM_HEAD

        head_why = _lease_head_problem(lease, formats, stored, inv, declared, cfg, source)
        body = {m: f for m, f in formats.items() if not LM_HEAD.search(m)}  # the lease prices the head
    elif head_format is None and head == HEAD_16BIT:
        head_why, body = _lm_head_problem(formats, stored, inv, declared, cfg, source), formats
    else:
        from .budget import LM_HEAD

        head_why = _head_format_problem(head_format or head, formats, inv, cfg, source)
        body = {m: f for m, f in formats.items() if not LM_HEAD.search(m)}  # the exemption is lm_head's alone
    for problem in (
        why,
        mapped,
        head_why,
        _menu_problem(body, menu),
        floor_problem(formats, menu, anchor) if floor else None,
        _engine_problem(formats, mixed=declared_mixed),
        _row_problem(body, mixed=declared_mixed),
    ):
        if problem is not None:
            return f"format: {problem}"
    return None
