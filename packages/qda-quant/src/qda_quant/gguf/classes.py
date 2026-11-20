"""Which tensor classes a GGUF file holds, and what each would weigh at each ggml type.

A type allocation is a choice per tensor class (``attn_q``, ``ffn_down``, ``ssm_out``, ...), not per tensor:
llama.cpp stores a per-tensor type, but a class is what a mix can move as one unit and what a reader can measure
one at a time. This module answers the static half of that: the classes of one template, their parameters and
rows, the bytes every class takes at each type the sensitivity grid may assign, and the bytes the Q4_K_M rule
itself gives it. The dynamic half -- the KL increment of a fake-quantized class -- is measured separately.

The four grid types are ``Q8_0``, ``Q6_K``, ``Q5_K`` and ``Q4_K``; bf16 is the baseline, not a cell. A class
whose row width is not a whole block of a type has no entry for it (`None`), rather than a silently rounded
number. The rule's own mix is reported per class, so a plan can say what an allocation buys over the file it
replaces.

``blk.N.nextn.*`` names mark the MTP block; every tensor of that block is reported under ``mtp`` (``nextn.eh_proj``
under ``mtp_eh``) and none is a grid class: the block is not in a reference forward, so a read of it measures
nothing. Tensors the rule never quantizes (norms, routers, ``ssm_conv1d``, 1-D tensors) are reported as ``keeps``.
A quantized tensor that matches no class is refused: a new architecture must add its classes here instead of
being folded into a neighbouring one.

The module reads tensor headers, not tensor data, and importing it needs neither torch nor the ``gguf`` package.
The row-byte formula is the only thing it takes from `gguf.file` (which uses torch for tensor values), imported
on first use in both `plan_from_tensors` and `plan`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import rule

#: The grid's types, most bytes first. bf16 (the baseline) is not here: it is where a class starts.
CELL_TYPES: tuple[tuple[str, int], ...] = (
    ("Q8_0", rule.Q8_0),
    ("Q6_K", rule.Q6_K),
    ("Q5_K", rule.Q5_K),
    ("Q4_K", rule.Q4_K),
)
_TYPE_NAME = {kind: name for name, kind in CELL_TYPES} | {
    rule.BF16: "BF16",
    rule.F32: "F32",
    rule.F16: "F16",
    rule.Q5_0: "Q5_0",
}

#: The tensor classes a dense gated-delta-net(hybrid) file has, in reporting order. A model without one of
#: them simply has no entry; a class with tensors must be one of these.
CLASSES: tuple[str, ...] = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_qkv",
    "attn_gate",
    "ssm_out",
    "ssm_alpha",
    "ssm_beta",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "token_embd",
    "output",
)

#: tensor names the rule quantizes, and the class each one is
_SUFFIX: dict[str, str] = {
    **{f"attn_{p}.weight": f"attn_{p}" for p in ("q", "k", "v", "output", "qkv", "gate")},
    **{f"ssm_{p}.weight": f"ssm_{p}" for p in ("out", "alpha", "beta")},
    **{f"ffn_{p}.weight": f"ffn_{p}" for p in ("gate", "up", "down")},
}
_GLOBAL = {"token_embd.weight": "token_embd", "output.weight": "output"}
#: types the rule leaves a tensor in: a tensor in one of them is not quantized (a fallback type is)
_UNQUANTIZED = frozenset({rule.F32, rule.F16, rule.BF16})
_NEXTN = ".nextn."


@dataclass(frozen=True)
class ClassPlan:
    """One class: its tensors, parameters and weight bytes under each type.

    Attributes:
        name: the class.
        tensors: how many tensors of the template it covers.
        rows: row count per distinct row width (``ne[0]``).
        bf16_bytes: their weight bytes stored bf16.
        bytes: weight bytes at each of `CELL_TYPES`; `None` where a width is not a whole block of that type.
        rule_bytes: weight bytes under the Q4_K_M rule's own mix; `rule_mix` is that mix in bytes by type, not
            in tensor counts.
        file_bytes: the same cells as they cost inside a `.gguf`: every tensor rounded up to the file's
            alignment as the writer lays it out. ``bf16`` is the unquantized cost; `None` where a type is
            illegal. `rule_file_bytes` is the aligned cost of the rule's mix.
    """

    name: str
    tensors: int
    rows: dict[int, int]
    bf16_bytes: int
    bytes: dict[str, int | None]
    rule_bytes: int
    rule_mix: dict[str, int] = field(default_factory=dict)
    file_bytes: dict[str, int | None] = field(default_factory=dict)
    rule_file_bytes: int = 0

    @property
    def params(self) -> int:
        """Parameter count of the class's tensors."""
        return sum(w * n for w, n in self.rows.items())

    @property
    def widths(self) -> tuple[int, ...]:
        """The distinct row widths, ascending."""
        return tuple(sorted(self.rows))


@dataclass(frozen=True)
class Plan:
    """The classes of one template, and the buckets beside them.

    Attributes:
        arch: ``general.architecture``.
        n_layer: ``<arch>.block_count``.
        classes: class name -> `ClassPlan`, in `CLASSES` order.
        excluded: ``mtp``, ``mtp_eh`` and ``keeps`` -> `ClassPlan` (reported, never allocated).
        template: the file read, when the plan came from one.
        alignment: the file's tensor alignment; `header_bytes` its metadata and tensor-table size. Together
            with `file_total`, these turn an allocation into the exact size of the `.gguf` a writer emits.
    """

    arch: str
    n_layer: int
    classes: dict[str, ClassPlan]
    excluded: dict[str, ClassPlan]
    template: str | None = None
    alignment: int = 32
    header_bytes: int = 0

    def total(self, picks: Mapping[str, str] | None = None) -> int:
        """Weight bytes of the classes under `picks` (class -> type, ``bf16`` or ``rule``); default: the rule.

        Only allocatable classes are summed: `excluded` (``mtp``, ``mtp_eh``, ``keeps``) is fixed at the rule's
        mix, is never part of a `picks`, and must be added by a caller that wants the whole file's **weight
        payload**. The file itself adds its header and the alignment between tensors; `file_total` is the exact
        size a writer emits.
        """
        picks = picks or {}
        unknown = sorted(set(picks) - set(self.classes))
        if unknown:
            raise ValueError(f"gguf: {unknown[0]}: not an allocatable class; picks name classes only")
        return sum(_pick(c, picks.get(c.name, "rule")) for c in self.classes.values())

    def payload_total(self, picks: Mapping[str, str] | None = None) -> int:
        """The whole tensor payload of `picks`: the classes' weight bytes plus the fixed buckets'.

        `total` sums allocatable classes only; a writer's tensor payload also holds `excluded` (``mtp``,
        ``mtp_eh``, ``keeps``) at the rule's mix. This is the figure a reading labels **payload**; budgets
        compare `file_total`, which is larger by the header and the alignment between tensors.
        """
        return self.total(picks) + sum(c.rule_bytes for c in self.excluded.values())

    def file_total(self, picks: Mapping[str, str] | None = None) -> int:
        """The exact `.gguf` file size of `picks`: header, the fixed buckets, and every class's aligned bytes.

        This is what a budget on the file compares (the exporter's exact size); `total` is the same file's
        tensor payload, smaller by the header and by the alignment between tensors. Refuses picks naming no
        allocatable class, as `total` does.
        """
        picks = picks or {}
        unknown = sorted(set(picks) - set(self.classes))
        if unknown:
            raise ValueError(f"gguf: {unknown[0]}: not an allocatable class; picks name classes only")
        got = self.header_bytes + sum(c.rule_file_bytes for c in self.excluded.values())
        return got + sum(_file_pick(c, picks.get(c.name, "rule")) for c in self.classes.values())


def _pick(c: ClassPlan, kind: str) -> int:
    """Weight bytes of class `c` at `kind` (a grid type, ``bf16`` or ``rule``)."""
    if kind == "bf16":
        return c.bf16_bytes
    if kind == "rule":
        return c.rule_bytes
    got = c.bytes.get(kind)
    if got is None:
        raise ValueError(f"{c.name}: {kind} is not legal for row widths {c.widths}")
    return got


def _file_pick(c: ClassPlan, kind: str) -> int:
    """Aligned weight bytes of class `c` at `kind`, as the file lays them out."""
    if kind == "rule":
        return c.rule_file_bytes
    got = c.file_bytes.get(kind)
    if got is None:
        raise ValueError(f"{c.name}: {kind} is not legal for row widths {c.widths}")
    return got


def _mtp_block(by_name: Mapping[str, object]) -> int | None:
    """The block index whose ``nextn.*`` tensors mark it as the MTP block, if the file has one."""
    blocks = {int(m.group(1)) for n in by_name if _NEXTN in n and (m := rule._LAYER.match(n))}
    return min(blocks) if blocks else None


class _Acc:
    """Rows of one class, before they become a `ClassPlan`."""

    def __init__(self, alignment: int = 32) -> None:
        self.alignment = alignment
        self.tensors = 0
        self.rows: dict[int, int] = {}
        self.rule_bytes = 0
        self.rule_mix: dict[str, int] = {}
        self.rule_file_bytes = 0
        self.file_cells: dict[str, int | None] = {label: 0 for label, _ in CELL_TYPES}
        self.file_bf16 = 0

    def _aligned(self, nbytes: int) -> int:
        """The bytes one tensor occupies once the writer aligns it."""
        return nbytes + (-nbytes) % self.alignment

    def add(self, ne: tuple[int, ...], kind: int) -> None:
        width, n = int(ne[0]), 1
        for d in ne[1:]:
            n *= int(d)
        self.tensors += 1
        self.rows[width] = self.rows.get(width, 0) + n
        got = _row_bytes(kind, width) * n
        self.rule_bytes += got
        self.rule_file_bytes += self._aligned(got)
        self.file_bf16 += self._aligned(2 * width * n)
        name = _TYPE_NAME.get(kind, str(kind))
        self.rule_mix[name] = self.rule_mix.get(name, 0) + got
        for label, t in CELL_TYPES:
            got = self.file_cells[label]
            if got is None:
                continue
            try:
                self.file_cells[label] = got + self._aligned(_row_bytes(t, width) * n)
            except ValueError:
                self.file_cells[label] = None

    def plan(self, name: str) -> ClassPlan:
        rows = dict(sorted(self.rows.items()))
        cells: dict[str, int | None] = {}
        for label, kind in CELL_TYPES:
            try:
                cells[label] = sum(_row_bytes(kind, w) * n for w, n in rows.items())
            except ValueError:
                cells[label] = None
        return ClassPlan(
            name=name,
            tensors=self.tensors,
            rows=rows,
            bf16_bytes=size(self.rows) * 2,
            bytes=cells,
            rule_bytes=self.rule_bytes,
            rule_mix=dict(sorted(self.rule_mix.items())),
            file_bytes={"bf16": self.file_bf16, **self.file_cells},
            rule_file_bytes=self.rule_file_bytes,
        )


def _row_bytes(kind: int, width: int) -> int:
    """Weight bytes of one row at `kind`; the only dependency this module takes from `gguf.file`, imported here.

    `gguf.file` imports torch for tensor values, so importing this module must not pull it in.
    """
    from .file import row_bytes

    return row_bytes(kind, width)


def size(rows: Mapping[int, int]) -> int:
    """Parameter count of ``{width: row count}``."""
    return sum(w * n for w, n in rows.items())


def _class_of(name: str) -> str | None:
    """The class a tensor name belongs to, or `None` when it is not a grid class."""
    if name in _GLOBAL:
        return _GLOBAL[name]
    m = rule._LAYER.match(name)
    return _SUFFIX.get(name[m.end() :]) if m else None


def plan_from_tensors(
    arch: str,
    n_layer: int,
    tensors: Sequence[tuple[str, tuple[int, ...], int]],
    *,
    alignment: int = 32,
    header_bytes: int = 0,
) -> Plan:
    """Build a `Plan` from ``(name, ne, type)`` rows, without reading a file.

    ``tensors`` is what `rule.q4_k_m_types` takes; the type is used only to find which tensors the rule
    quantizes. Raises `rule.Refused` for an architecture the rule cannot type, and `ValueError` for a quantized
    tensor that matches no class.
    """
    rule_types = rule.q4_k_m_types(arch, n_layer, tensors)
    mtp = _mtp_block({n: None for n, _, _ in tensors})
    acc: dict[str, _Acc] = {}
    excluded: dict[str, _Acc] = {}
    for name, ne, _ in tensors:
        ne = tuple(int(d) for d in ne)
        kind = rule_types[name]
        m = rule._LAYER.match(name)
        if mtp is not None and m and int(m.group(1)) == mtp:
            bucket = "mtp_eh" if _NEXTN in name else "mtp"
            excluded.setdefault(bucket, _Acc(alignment)).add(ne, kind)
            continue
        if kind in _UNQUANTIZED:
            excluded.setdefault("keeps", _Acc(alignment)).add(ne, kind)
            continue
        cls = _class_of(name)
        if cls is None:
            raise ValueError(f"gguf: {name}: quantized but in no class; add its class to sensitivity.CLASSES")
        acc.setdefault(cls, _Acc(alignment)).add(ne, kind)
    order = {c: i for i, c in enumerate(CLASSES)}
    return Plan(
        arch=arch,
        n_layer=n_layer,
        classes={c: acc[c].plan(c) for c in sorted(acc, key=order.__getitem__)},
        excluded={b: a.plan(b) for b, a in excluded.items()},
        alignment=alignment,
        header_bytes=header_bytes,
    )


def plan(template: str | Path) -> Plan:
    """Build a `Plan` from a bf16 GGUF template, sized exactly as the writer will lay it out."""
    from .file import Layout, Template

    t = Template(template)
    rows = [(name, tuple(info.shape), info.tensor_type) for name, info in t.tensors.items()]
    layout = Layout(t, rule.q4_k_m_types(t.arch, t.n_layer, rows))
    got = plan_from_tensors(t.arch, t.n_layer, rows, alignment=t.alignment, header_bytes=len(layout.head))
    if got.file_total() != layout.total:
        raise ValueError(f"gguf: {template}: plan file size {got.file_total()} != layout {layout.total}")
    return replace(got, template=str(template))


def expand(template: str | Path, picks: Mapping[str, str]) -> list[dict[str, str]]:
    """``{class: type}`` -> ``tensor_types`` rules naming every stored tensor of a picked class.

    The allocation is per class; a `.gguf` stores a type per tensor. This is the one expansion between them
    for the class map: each tensor the Q4_K_M rule quantizes takes its class's type, exactly by name, and a
    tensor the rule keeps float, and the MTP block (which no reference forward reads), takes no rule. The
    types are the recipe's own names, ``bf16`` as ``BF16``.

    Raises:
        ValueError: a pick names a class this template has no tensor of.
    """
    from .file import Template

    t = Template(template)
    rows = [(name, tuple(info.shape), info.tensor_type) for name, info in t.tensors.items()]
    rule_types = rule.q4_k_m_types(t.arch, t.n_layer, rows)
    mtp = _mtp_block({n: None for n, _, _ in rows})
    unknown = sorted(set(picks) - set(_class_of(n) for n, _, _ in rows if _class_of(n) is not None))
    if unknown:
        raise ValueError(f"gguf: {unknown[0]}: the template has no tensor of this class")
    out: list[dict[str, str]] = []
    for name, _ne, _kind in rows:
        if rule_types[name] in _UNQUANTIZED:
            continue
        m = rule._LAYER.match(name)
        if mtp is not None and m and int(m.group(1)) == mtp:
            continue
        klass = _class_of(name)
        if klass is not None and klass in picks:
            kind = picks[klass]
            out.append({"pattern": name, "type": "BF16" if kind == "bf16" else kind})
    return out
