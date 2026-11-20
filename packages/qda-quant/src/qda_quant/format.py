"""What a checkpoint stores: the quantized format, independent of the GPU that serves it.

A `Format` says how weights (and, for W4A8, W8A8 and W4A4, activations) are represented. Which kernel serves it,
and therefore which numerics the quantizer has to match, is the job of `qda_quant.target`.

Formats are plugins: `FORMATS` holds the built-ins and package registrations, and
`Format.parse` is the one place a name becomes a `Format` (see `qda_quant.registry`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import ClassVar

from .registry import Registry

#: The built-in integer spelling: bits, then activations, optionally symmetry and group size.
_NAME = re.compile(r"w(\d)a(16|8)(?:-(sym|asym))?(?:-(g\d+|pc))?")

#: Bits an activation cut computes a value in, by `Format.act`.
ACT_BITS: dict[str, int] = {"bf16": 16, "fp8_token": 8, "int8_token": 8, "nvfp4_block": 4}


@dataclass(frozen=True)
class Format:
    """A stored quantization format.

    Subclass it and override `name`; override `qmin`/`qmax` or `from_name` too when the built-in integer
    codes or spelling do not describe your format. Register an instance in ``qda.formats``.

    Attributes:
        weight_bits: bits per weight code (3 or 4 in the built-in presets).
        group_size: weights sharing one scale along the input dimension; -1 = per channel.
        symmetric: symmetric codes (no zero point) if True, asymmetric (zero point) if False.
        weight_type: code type of the weights; ``"int"`` for the built-in integer codes, else a plugin's own.
        act: activation precision the kernel computes in; ``"fp8_token"`` is FP8 E4M3 with dynamic
            per-token scales (W4A8), ``"int8_token"`` INT8 with dynamic per-token symmetric scales (W8A8),
            ``"nvfp4_block"`` E2M1 with a dynamic E4M3 scale per 16 values under a static global scale (W4A4).
        symmetric_defaulted: True when the spelling named no symmetry and `DEFAULT_SYMMETRIC` supplied it. The
            planner may then store a unit class in another symmetry (routed expert banks default to symmetric);
            the value never changes the stored format, and two formats that differ only in it are equal.
    """

    weight_bits: int
    group_size: int = 128
    symmetric: bool = True
    weight_type: str = "int"
    act: str = "bf16"
    symmetric_defaulted: bool = field(default=False, compare=False)

    #: Each built-in family's default symmetry, keyed by (weight bits, activation). A spelling that names
    #: neither ``-sym`` nor ``-asym`` inherits it: W4A16 is asymmetric (the paper's format); W3A16 is symmetric
    #: (the paper's and the released checkpoints are INT3 g128 sym); W4A8 is symmetric (CUTLASS W4A8 takes
    #: symmetric weights only). A family with a spelling of its own declares its default in `from_name`. A
    #: spelling that inherits records that it did (`symmetric_defaulted`): the planner stores routed expert
    #: banks symmetric unless the config names a symmetry.
    DEFAULT_SYMMETRIC: ClassVar[dict[tuple[int, str], bool]] = {
        (4, "bf16"): False,
        (3, "bf16"): True,
        (4, "fp8_token"): True,
    }
    #: The clip a quant step runs here when its recipe names none or a range search (``hmse``, ``mse``,
    #: ``imse``): a grid that fits its scales on the values has no range to search. None keeps the step's own.
    default_clip: ClassVar[str | None] = None

    @property
    def act_bits(self) -> int:
        """Bits per activation value the kernel computes in: 16 for bf16, 8 for FP8/INT8 tokens, 4 for NVFP4."""
        return ACT_BITS[self.act]

    @property
    def name(self) -> str:
        """Canonical short name, e.g. ``w4a16-asym-g128``."""
        a = "a16" if self.act == "bf16" else "a8"
        s = "sym" if self.symmetric else "asym"
        g = "pc" if self.group_size == -1 else f"g{self.group_size}"
        return f"w{self.weight_bits}{a}-{s}-{g}"

    @property
    def spelling(self) -> str:
        """The bare spelling that inherits the symmetry: ``w4a16-g64`` where `name` is ``w4a16-asym-g64``."""
        a = "a16" if self.act == "bf16" else "a8"
        g = "" if self.group_size == 128 else ("-pc" if self.group_size == -1 else f"-g{self.group_size}")
        return f"w{self.weight_bits}{a}{g}"

    @property
    def expert_format(self) -> Format:
        """The format routed-expert banks resolve to: `self`, or its symmetric twin for a bare spelling."""
        return replace(self, symmetric=True) if self.symmetric_defaulted else self

    @property
    def qmin(self) -> int:
        """Smallest integer code."""
        return -(1 << (self.weight_bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        """Largest integer code."""
        return (1 << (self.weight_bits - 1)) - 1 if self.symmetric else (1 << self.weight_bits) - 1

    @staticmethod
    def from_name(name: str) -> Format | None:
        """The built-in integer format `name` spells, or None.

        The spelling is ``w4a16``, ``w4a16-asym-g128``, ``w3a16-sym-g64``, ``w4a8-sym-pc``: bits, then
        activations, then optionally symmetry and group size. A spelling that names neither ``-sym`` nor
        ``-asym`` inherits its family's default (`DEFAULT_SYMMETRIC`); a family outside that table defaults
        to symmetric. A family with a spelling of its own overrides it; `parse` asks every registered format
        class.
        """
        m = _NAME.fullmatch(name)
        if m is None:
            return None
        bits, a, sym, g = m.groups()
        group = -1 if g == "pc" else int(g[1:]) if g else 128
        act = "bf16" if a == "16" else "fp8_token"
        symmetric = Format.DEFAULT_SYMMETRIC.get((int(bits), act), True) if sym is None else sym == "sym"
        return Format(int(bits), group, symmetric, "int", act, sym is None)

    @staticmethod
    def parse(s: str | Format) -> Format:
        """The format a config names: a registered key or `name`, a preset's short name, else a spelling.

        Spellings are case-insensitive and surrounding whitespace is ignored. A spelling without ``-sym`` or
        ``-asym`` inherits its family's default: ``w4a16*`` asymmetric, ``w3a16*``, ``w4a8*`` and ``w8a8i*``
        symmetric. The result records that it inherited (`symmetric_defaulted`); a spelling that names a
        symmetry does not.

        Raises:
            Refused: `s` names no format; the message gives an example name.
        """
        if isinstance(s, Format):
            return s
        name = s.strip().lower()
        if (
            name == "w8a8"
        ):  # the earlier preset name: as a generic spelling it means int8 weights, FP8 activations
            from .step import Refused  # here, not at the top: keeps this module light

            raise Refused(
                f"format {s!r}: QDA 0.1.0 has no format of that name",
                "use w8a8-fp8 (FP8 W8A8) or w8a8i (INT8 W8A8)",
            )
        fmt = _resolve(name)
        if fmt is not None:
            return _defaulted(fmt, name)
        from .step import Refused  # here, not at the top: keeps this module light

        raise Refused(f"unknown format {s!r}; expected e.g. 'w4a16-asym-g128'")


#: The INT8 W8A8 spelling: the family name, then optionally symmetry and a group spelling.
_W8A8I = re.compile(r"w8a8i(?:-(sym|asym))?(?:-(g\d+|pc))?")


@dataclass(frozen=True)
class Int8W8A8(Format):
    """INT8 weights with dynamic per-token symmetric INT8 activations: vLLM's INT8 W8A8.

    The kernel, ``CompressedTensorsW8A8Int8``, takes symmetric weights per channel or per tensor only, so
    this family has one member: symmetric, per-output-channel, spelled ``w8a8i-sym-pc`` and defaulted by
    ``w8a8i``. `from_name` refuses an asymmetric or grouped spelling with the reason.
    """

    weight_bits: int = 8
    group_size: int = -1
    symmetric: bool = True
    act: str = "int8_token"

    @property
    def name(self) -> str:
        """Canonical short name: ``w8a8i-sym-pc``."""
        s = "sym" if self.symmetric else "asym"
        g = "pc" if self.group_size == -1 else f"g{self.group_size}"
        return f"w8a8i-{s}-{g}"

    @staticmethod
    def from_name(name: str) -> Format | None:
        """The W8A8 INT8 format `name` spells, or None.

        Raises:
            Refused: `name` spells asymmetric or grouped weights, which vLLM's INT8 W8A8 does not serve.
        """
        m = _W8A8I.fullmatch(name)
        if m is None:
            return None
        sym, g = m.groups()
        if sym == "asym" or g not in (None, "pc"):
            from .step import Refused  # here, not at the top: keeps this module light

            raise Refused(
                f"{name}: vLLM's INT8 W8A8 accepts only symmetric per-channel or per-tensor weights; "
                "w8a8i is symmetric per-output-channel ('w8a8i' or 'w8a8i-sym-pc')"
            )
        return Int8W8A8()


#: The FP8 W8A8 spelling: ``float-quantized`` E4M3 weights, per-output-channel scale, per-token activations.
W8A8_FP8_FORMAT = "w8a8-fp8"


@dataclass(frozen=True)
class W8A8Fp8(Format):
    """FP8 E4M3 weights with dynamic per-token FP8 E4M3 activations: vLLM's FP8 W8A8.

    The family has one member: vLLM's ``CompressedTensorsW8A8Fp8`` takes symmetric weights, per output channel
    (``channel``) or per tensor (``tensor``); QDA writes the per-channel scheme, spelled ``w8a8-fp8``. A code
    is an E4M3 value itself, so `qmin`/`qmax` are the format's largest magnitude, 448, not an integer range.
    """

    weight_bits: int = 8
    group_size: int = -1
    symmetric: bool = True
    weight_type: str = "fp8"
    act: str = "fp8_token"

    @property
    def name(self) -> str:
        """Canonical short name: ``w8a8-fp8``."""
        return W8A8_FP8_FORMAT

    @property
    def spelling(self) -> str:
        """The one spelling: ``w8a8-fp8``."""
        return W8A8_FP8_FORMAT

    @property
    def qmin(self) -> int:
        """Smallest code: the largest magnitude E4M3 holds, negated."""
        return -448

    @property
    def qmax(self) -> int:
        """Largest code: the largest magnitude E4M3 holds."""
        return 448

    @staticmethod
    def from_name(name: str) -> Format | None:
        """`name`, when it is this format's spelling."""
        return W8A8Fp8() if name == W8A8_FP8_FORMAT else None


#: The GGUF nonlinear formats: a code is an index into llama.cpp's 16-entry table, scaled per group.
IQ4XS_FORMAT = "w4a16-iq4xs"
IQ4NL_FORMAT = "w4a16-iq4nl"
Q3K_FORMAT = "w3a16-q3k"


@dataclass(frozen=True)
class _Codebook(Format):
    """A GGUF nonlinear format: codes index a fixed 16-entry table, scaled per group.

    The codes are not integers whose value is the number (`Format.weight_type` is the table's name), so the
    spelling is the family's own: ``w4a16-iq4xs``, ``w4a16-iq4nl``, ``w3a16-q3k``. The built-in integer
    spelling cannot express them, and a name that matched it would collide with an int format of the same
    bits and group size.
    """

    @property
    def qmin(self) -> int:
        """Smallest code: an index into the table, so 0."""
        return 0

    @property
    def qmax(self) -> int:
        """Largest code: the last table entry."""
        return 15


@dataclass(frozen=True)
class Iq4XS(_Codebook):
    """IQ4_XS: 4-bit table indices, one 6-bit signed scale per 32 values, one fp16 ``d`` per 256."""

    weight_bits: int = 4
    group_size: int = 32
    symmetric: bool = True
    weight_type: str = "iq4nl"

    @property
    def name(self) -> str:
        """Canonical short name: ``w4a16-iq4xs``."""
        return IQ4XS_FORMAT

    @staticmethod
    def from_name(name: str) -> Format | None:
        """`name`, when it is this format's spelling."""
        return Iq4XS() if name == IQ4XS_FORMAT else None


@dataclass(frozen=True)
class Iq4NL(_Codebook):
    """IQ4_NL: 4-bit table indices, one fp16 scale per 32 values."""

    weight_bits: int = 4
    group_size: int = 32
    symmetric: bool = True
    weight_type: str = "iq4nl"

    @property
    def name(self) -> str:
        """Canonical short name: ``w4a16-iq4nl``."""
        return IQ4NL_FORMAT

    @staticmethod
    def from_name(name: str) -> Format | None:
        """`name`, when it is this format's spelling."""
        return Iq4NL() if name == IQ4NL_FORMAT else None


@dataclass(frozen=True)
class Q3KFormat(Format):
    """Q3_K: signed 3-bit codes in ``-4 .. 3``, one 6-bit signed scale per 16 values, one fp16 ``d`` per 256."""

    weight_bits: int = 3
    group_size: int = 16
    symmetric: bool = True
    weight_type: str = "q3k"

    @property
    def name(self) -> str:
        """Canonical short name: ``w3a16-q3k``."""
        return Q3K_FORMAT

    @staticmethod
    def from_name(name: str) -> Format | None:
        """`name`, when it is this format's spelling."""
        return Q3KFormat() if name == Q3K_FORMAT else None


#: NVFP4 as compressed-tensors stores it and Blackwell's FP4 kernels serve it.
NVFP4_FORMAT = "w4a4-nvfp4"


@dataclass(frozen=True)
class Nvfp4(Format):
    """NVFP4: E2M1 weights with one FP8 E4M3 scale per 16 values, and activations cut the same way at run time.

    A code is a signed position on the E2M1 ladder (``0, 0.5, 1, 1.5, 2, 3, 4, 6``), so `qmin`/`qmax` are -7
    and 7 and code 0 is the value 0. The family has one member: the kernels take symmetric groups of 16 only.
    """

    weight_bits: int = 4
    group_size: int = 16
    symmetric: bool = True
    weight_type: str = "e2m1"
    act: str = "nvfp4_block"
    default_clip: ClassVar[str | None] = "fp4hmse"

    @property
    def name(self) -> str:
        """Canonical short name: ``w4a4-nvfp4``."""
        return NVFP4_FORMAT

    @property
    def spelling(self) -> str:
        """The one spelling: ``w4a4-nvfp4``."""
        return NVFP4_FORMAT

    @property
    def qmin(self) -> int:
        """Smallest code: -6, the ladder's last entry, negated."""
        return -7

    @property
    def qmax(self) -> int:
        """Largest code: 6, the ladder's last entry."""
        return 7

    @staticmethod
    def from_name(name: str) -> Format | None:
        """`name`, when it is this format's spelling."""
        return Nvfp4() if name == NVFP4_FORMAT else None


def _adopt(key: str, fmt: Format) -> None:
    """Refuse invalid types, canonical-name collisions and formats that cannot be recorded and replayed.

    Validate against already-added entries without triggering discovery while built-ins initialize.
    """
    if not isinstance(fmt, Format):
        raise TypeError(f"format plugin {key!r} is {fmt!r}, not a Format instance")
    from .step import Refused  # here, not at the top: keeps this module light

    canonical = fmt.name
    if not isinstance(canonical, str):
        raise Refused(f"format plugin {key!r} has a non-string canonical name {canonical!r}")
    for other_key, other in FORMATS._entries.items():
        if isinstance(other, str):
            continue
        if canonical in (other_key, other.name):
            raise Refused(
                f"format plugin {key!r}: canonical name {canonical!r} already belongs to {other_key!r}"
            )
    previous = FORMATS._entries.get(key)
    FORMATS._entries[key] = fmt
    try:
        for known in list(FORMATS._entries.values()):
            if isinstance(known, str):
                continue
            if Format.parse(known.name) != known:
                raise Refused(f"format plugin {key!r} makes canonical name {known.name!r} fail to round-trip")
    finally:
        if previous is None:
            del FORMATS._entries[key]
        else:
            FORMATS._entries[key] = previous


def _resolve(name: str) -> Format | None:
    """The format `name` spells, from the registry or a registered class.

    `name` arrives lower-cased, so the registry key and the canonical name are folded the same way: a plugin
    registered under a mixed-case key or spelling still round-trips through ``resolved.yaml``.

    Folding makes collisions: two registered formats of one name (``W4A16`` and ``w4a16``), or a registry
    entry that means another format than the built-in spelling of the same name, are ambiguity, refused so
    that installing a plugin cannot change what an existing config builds. The built-in presets own their
    short names (the registry refuses a second provider of one, and a plugin taking one is ambiguous here);
    a preset may spell a format the generic spelling does not describe, as ``w4a4`` is the NVFP4 format whose
    canonical name is ``w4a4-nvfp4``. FP8 W8A8 has no preset: it is the internal precision a W4A4 mixed build
    and an FP8 head use, named ``w8a8-fp8``.
    """
    from .step import Refused  # here, not at the top: keeps this module light

    matches = [(key, fmt) for key, fmt in FORMATS.items() if name in (key.lower(), fmt.name.lower())]
    if not matches:
        return PRESETS[name] if name in PRESETS else _spelled(name)
    key, fmt = matches[0]
    if any(other != fmt for _, other in matches[1:]):
        raise Refused(
            f"format name {name!r} is ambiguous: registered as "
            + ", ".join(f"{k!r} ({f.name!r})" for k, f in matches)
        )
    builtin = _spelled(name)
    if builtin is not None and builtin != fmt and PRESETS.get(key) != fmt:
        raise Refused(
            f"format name {name!r} is ambiguous: the built-in spelling gives {builtin.name!r} and "
            f"the format registered as {key!r} gives {fmt.name!r}"
        )
    return fmt


def _defaulted(fmt: Format, name: str) -> Format:
    """`fmt`, marked when `name` is a bare spelling that inherits its symmetry (`DEFAULT_SYMMETRIC`).

    A registry key and the built-in spelling of the same name resolve to the same format (`_resolve` folds
    them), so the mark cannot come from `from_name` alone: the preset instance is registered without it. A
    registered format that the bare spelling does not describe keeps its own symmetry and is not marked.
    """
    m = _NAME.fullmatch(name)
    if m is None or m.group(3) is not None:
        return fmt
    spelled = type(fmt).from_name(name)
    return replace(fmt, symmetric_defaulted=True) if spelled is not None and spelled == fmt else fmt


def _spelled(name: str) -> Format | None:
    """The format the registered classes' built-in spellings give for `name`, or None."""
    for fmt_class in _classes():
        fmt = fmt_class.from_name(name)
        if fmt is not None:
            return fmt
    return None


def _classes() -> list[type[Format]]:
    """The distinct classes among the registered formats, in registry order."""
    classes: list[type[Format]] = []
    for _, fmt in FORMATS.items():
        if type(fmt) not in classes:
            classes.append(type(fmt))
    return classes


#: Formats by config name: built-ins and package registrations (see `qda_quant.registry`).
W4A16 = Format(4, 128, symmetric=False)
W4A8 = Format(4, 128, symmetric=True, act="fp8_token")
W3A16 = Format(3, 128, symmetric=True)
W8A8I = Int8W8A8()
W8A8_FP8 = W8A8Fp8()
IQ4XS = Iq4XS()
IQ4NL = Iq4NL()
Q3K = Q3KFormat()
NVFP4 = Nvfp4()
FORMATS: Registry[Format] = Registry(
    "formats",
    {
        "w3a16": W3A16,
        "w4a16": W4A16,
        "w4a8": W4A8,
        "w8a8i": W8A8I,
        W8A8_FP8_FORMAT: W8A8_FP8,
        NVFP4_FORMAT: NVFP4,
        IQ4XS_FORMAT: IQ4XS,
        IQ4NL_FORMAT: IQ4NL,
        Q3K_FORMAT: Q3K,
    },
    on_add=_adopt,
)


#: Built-in formats under the short names: W4A16 asymmetric, W4A8 symmetric with FP8 activations,
#: W3A16 symmetric, W8A8 FP8 weights with FP8 activations, and NVFP4 W4A4.
PRESETS: dict[str, Format] = {"w4a16": W4A16, "w4a8": W4A8, "w3a16": W3A16, "w4a4": NVFP4}
