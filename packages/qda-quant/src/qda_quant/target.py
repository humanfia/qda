"""Where a checkpoint is served, and what that implies for the numbers the quantizer must pick.

A `Target` names the GPU architecture and the serving stack. `Target.plan(fmt, kind)` returns a
`KernelPlan`: which vLLM kernel serves that format on that layer kind, and how that kernel really
stores and uses scales (for example, the Hopper CUTLASS W4A8 kernel casts group scales to FP8).
Quantizers read the plan to match the serving kernel's numerics.

The built-in tables are Hopper (sm90), Ampere (sm80), Ada (sm89, L40S / L4 / RTX 4090), the datacenter
Blackwells (sm100, B200 / B100 / GB200; sm103, B300 / GB300) and the RTX PRO 6000 (sm120). Methods depend on
the plan; target plugins supply tables for additional architectures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .exit import ENVIRONMENT
from .format import IQ4NL_FORMAT, IQ4XS_FORMAT, NVFP4_FORMAT, Q3K_FORMAT, Format
from .registry import Registry

LayerKind = Literal["dense", "moe"]

#: Formats whose solved codes a GGUF holds (`qda_quant.gguf`).
Q4K, Q5K, Q6K = "w4a16-asym-g32", "w5a16-asym-g32", "w6a16-sym-g16"
#: Solved formats and their stored GGUF types.
GGUF_TYPES = {
    Q4K: "Q4_K",
    Q5K: "Q5_K",
    Q6K: "Q6_K",
    IQ4XS_FORMAT: "IQ4_XS",
    IQ4NL_FORMAT: "IQ4_NL",
    Q3K_FORMAT: "Q3_K",
}
#: Served search types, including template-fitted types without a solved grid.
GGUF_MENU = (*GGUF_TYPES.values(), "Q8_0")

#: Target tables by architecture name: built-ins and package registrations. A table is called as
#: ``table(target, fmt, kind)`` and returns the `KernelPlan`, or raises `Refused`.
TARGETS: Registry[Callable[[Target, Format, LayerKind], KernelPlan]] = Registry(
    "targets",
    {
        "llama_cpp": "qda_quant.target:llama_cpp",
        "sm80": "qda_quant.target:sm80",
        "sm89": "qda_quant.target:sm89",
        "sm90": "qda_quant.target:sm90",
        "sm100": "qda_quant.target:sm100",
        "sm103": "qda_quant.target:sm103",
        "sm120": "qda_quant.target:sm120",
    },
)

#: Binding rules by architecture name (the names of `TARGETS`): which dense kernel
#: vLLM binds for a module in format `fmt` of shape (out, in) under a serve section's args and env. Called as
#: ``rule(target, fmt, shape, args, env)``; returns the kernel class, or raises `Refused` when the rule cannot tell.
#: MoE rules use ``<architecture>_moe`` and shape ``(hidden, unpadded intermediate)``.
BINDINGS: Registry[Callable[[Target, Format, tuple[int, int], dict, dict], str]] = Registry(
    "bindings",
    {
        "llama_cpp": "qda_quant.target:llama_cpp_binds",
        "llama_cpp_moe": "qda_quant.target:llama_cpp_moe_binds",
        "sm80": "qda_quant.target:sm80_binds",
        "sm80_moe": "qda_quant.target:sm80_moe_binds",
        "sm89": "qda_quant.target:sm89_binds",
        "sm89_moe": "qda_quant.target:sm89_moe_binds",
        "sm90": "qda_quant.target:sm90_binds",
        "sm90_moe": "qda_quant.target:sm90_moe_binds",
        "sm100": "qda_quant.target:sm100_binds",
        "sm100_moe": "qda_quant.target:sm100_moe_binds",
        "sm103": "qda_quant.target:sm100_binds",
        "sm103_moe": "qda_quant.target:sm100_moe_binds",
        "sm120": "qda_quant.target:sm120_binds",
        "sm120_moe": "qda_quant.target:sm120_moe_binds",
    },
)

#: Why every card but Hopper refuses W4A8: one reason, so the support matrix gives it one footnote.
W4A8_NEEDS_HOPPER = (
    "FP8 activations need Hopper: vLLM's CUTLASS W4A8 kernel needs compute capability 90 (H100 / H200); "
    "use w4a16"
)
#: The note on a row no card of its architecture has run. `qda plan` prints it.
UNVERIFIED_ON_HARDWARE = "unverified on real hardware: no card has run this row, it comes from vLLM's source"

#: Names users give the cards that still have no table; a refusal for one of them says so and names
#: what can be built there. "blackwell" stays here: it names the datacenter and the consumer cards at once.
_PENDING = frozenset({"blackwell"})


def _capabilities_text(known: Iterable[tuple[int, int]]) -> str:
    """A table's declared capabilities, as ``9.0`` or ``8.0/8.6``."""
    return "/".join(f"{a}.{b}" for a, b in sorted(known))


def _build_elsewhere() -> str:
    """What a user on a card with no table can still do: build here for a card that has one."""
    tables = {arch: getattr(table, "capabilities", ()) for arch, table in TARGETS.items()}
    arches = sorted((a for a, known in tables.items() if known), key=lambda a: max(tables[a]), reverse=True)
    names = " or ".join(f"{a} ({_capabilities_text(tables[a])})" for a in arches)
    anywhere = sorted(a for a, known in tables.items() if not known)  # e.g. llama_cpp: a GGUF names no card
    gguf = f"; or build for {' or '.join(anywhere)}, whose table names no card" if anywhere else ""
    return (
        f"you can build a checkpoint here to serve on a supported card with target {names} (that checkpoint "
        f"is for that card, not this one){gguf}"
    )


@dataclass(frozen=True)
class KernelPlan:
    """How one (format, layer kind) is served on one target.

    Attributes:
        kernel: the vLLM kernel expected to serve it (checked by attestation, not trusted).
        scale_dtype: dtype the kernel stores group scales in ("bf16", "fp8_e4m3", "nvfp4", "fp32"; "nvfp4" is
            E4M3 under one fp32 global scale per module).
        act_dtype: dtype the kernel's matmul really consumes activations in ("bf16", or a cut of
            `solve.quant.ACT_CUTS`: per token "fp8_e4m3" and "int8", per 16 values "nvfp4").
        zero_point: whether the kernel supports asymmetric (zero-point) codes here.
        act_order: act-order modes the serving stack loads ("static" only on vLLM >= 0.30).
        notes: anything a quantizer must know that the fields above do not say.
        serve_env: environment the server must run with for `kernel` to bind (data written into the
            checkpoint's serve config, never read by qda itself), e.g. ``VLLM_DISABLED_KERNELS``.
        serve_args: vLLM engine arguments the server needs, e.g. ``{"moe_backend": "marlin"}``.
        serve_plugins: vLLM plugins the server must load (e.g. ``qda_humming_moe`` for int3 experts on 0.27.1).
        target: the target whose table produced this plan (`Target.plan` sets it). A step that stores a module
            in a format of its own asks it for that format's plan.
    """

    kernel: str
    scale_dtype: Literal["bf16", "fp8_e4m3", "nvfp4", "fp32"] = "bf16"
    act_dtype: Literal["bf16", "fp8_e4m3", "int8", "nvfp4"] = "bf16"
    zero_point: bool = True
    act_order: tuple[str, ...] = ("none", "static")
    notes: tuple[str, ...] = field(default=())
    serve_env: dict[str, str] = field(default_factory=dict)
    serve_args: dict[str, str] = field(default_factory=dict)
    serve_plugins: tuple[str, ...] = ()
    target: Target = field(default_factory=lambda: Target(), compare=False, repr=False)

    @property
    def act_needs(self) -> frozenset[str]:
        """Statistics the kernel's activation cut reads off the calibration set (NVFP4's input ranges)."""
        return frozenset({"act_absmax"}) if self.act_dtype == "nvfp4" else frozenset()


@dataclass(frozen=True)
class Target:
    """A serving target: GPU architecture plus the vLLM version (and optional backend pins)."""

    arch: str = "sm90"  #: a key of `TARGETS`
    vllm: str = "0.33.0"
    #: Dense WNA16 backend pin. "marlin" pins Marlin dense and experts, "humming" Humming dense and experts; the
    #: default pins nothing (`BANNED_KERNELS` stay off).
    backend: str | None = None

    @classmethod
    def parse(cls, s: str | Target | None) -> Target:
        """Parse a target name.

        A key of `TARGETS` names its table, and so does a GPU name in the table's ``aliases`` (``h200`` -> sm90);
        ``sm90@0.33.0`` pins a vLLM version; a ``:marlin`` or ``:humming`` suffix (``h200:marlin``,
        ``sm90@0.27.1:marlin``) pins the dense WNA16 kernel.

        Raises:
            Refused: `s` names no architecture (the message lists the known ones), two tables claim its name, its
                version is not numeric, or it names a banned kernel (``:machete``, `BANNED_KERNELS`).
        """
        if isinstance(s, Target):
            return s
        if s is None:
            return cls()
        from .step import Refused  # here, not at the top (as `plan` does): the target tables load early

        rest, _, backend = s.lower().partition(":")
        if backend == "machete":
            raise Refused(
                f"target {s!r}: drop :machete; use target {rest!r} (Machete is disabled on every vLLM engine qda "
                "starts, and vLLM binds Humming or Marlin instead)"
            )
        name, _, ver = rest.partition("@")
        if name not in TARGETS:
            aliases = {arch: getattr(table, "aliases", ()) for arch, table in TARGETS.items()}
            claimed = [arch for arch, names in aliases.items() if name in names]
            if len(claimed) > 1:
                raise Refused(f"target {s!r}: the tables {claimed} both claim the name {name!r}")
            if not claimed:
                known = sorted(a for names in aliases.values() for a in names)
                pending = f"; no target serves this GPU yet; {_build_elsewhere()}" if name in _PENDING else ""
                raise Refused(
                    f"target {s!r}: unknown architecture; pass --set target=NAME (or set target: in the config) "
                    f"with one of {', '.join(sorted(TARGETS))}; the GPU names are {', '.join(known)}{pending}"
                )
            name = claimed[0]
        if ver:
            try:
                _version(ver)
            except ValueError:
                raise Refused(
                    f"target {s!r}: vLLM version {ver!r} is not numeric; give a version like `0.33.0`"
                ) from None
        return cls(name, ver or cls.vllm, backend or None)

    @classmethod
    def stored(cls, s: str | None) -> Target:
        """The target a checkpoint's ``build.json`` names, read with a ``:machete`` suffix dropped.

        A checkpoint built for ``:machete`` serves on the same sm90 table with Machete off, so it is read as the
        plain target rather than refused: `parse` refuses the suffix only where a user types it.
        """
        if isinstance(s, str) and s.lower().endswith(":machete"):
            s = s[: -len(":machete")]
        return cls.parse(s)

    @classmethod
    def on_cards(cls, s: Target | str | None, cards: Sequence[tuple[str, tuple[int, int]]]) -> Target:
        """The target for a build on `cards`: (device label, compute capability) of every CUDA card it uses.

        A target names where the checkpoint is served, not where it is built, so an explicit one is always
        taken as given: building on one card for serving on another is legitimate (`on_other_cards` notes it).
        An unset target needs a default, and that default is the table whose `Table.capabilities` cover every
        card (Ampere rows on 8.0/8.6, Hopper's on 9.0), so a build on an Ampere card no longer plans Hopper
        rows. Cards no one table covers have no default: the build is refused and told which targets to pass
        for serving on a supported card, rather than being planned with an architecture nobody asked for.

        Raises:
            Refused: the target is unset and no one table covers `cards`, or more than one does.
        """
        from .step import Refused

        t = cls.parse(s)
        if s is not None or not cards:
            return t
        caps = {c for _, c in cards}
        tables = {arch: getattr(table, "capabilities", ()) for arch, table in TARGETS.items()}
        match = [arch for arch, known in tables.items() if caps <= set(known)]
        if len(match) == 1:
            return cls(match[0], t.vllm, t.backend)
        rows = " and ".join(f"{a}.{b}" for a, b in sorted(caps, reverse=True))
        # newest first, so the list reads the same wherever the tables were registered
        arches = sorted(
            (arch for arch, known in tables.items() if known), key=lambda a: max(tables[a]), reverse=True
        )
        if not match:
            support = " and ".join(f"{arch} ({_capabilities_text(tables[arch])})" for arch in arches)
            first, *rest = arches
            choice = f"--set target={first}" + (f" (or {' or '.join(rest)})" if rest else "")
            raise Refused(
                f"no supported target serves compute capability {rows} yet - QDA supports {support}.",
                f"build it here with {choice}; that checkpoint is for that card, not this one",
                code=ENVIRONMENT,
            )
        raise Refused(
            f"no target given, and more than one table is a default for compute capability {rows} "
            f"({', '.join(match)}); pass --set target=NAME"
        )

    def on_other_cards(self, cards: Sequence[tuple[str, tuple[int, int]]]) -> str | None:
        """A note when this target's rows plan for other cards than the build's, else None.

        An explicit target is the serving target (`on_cards` takes it as given), so this says what is the case
        and does not refuse: the build log names the target's capability and the cards it runs on.
        """
        known = set(getattr(TARGETS[self.arch], "capabilities", ()))
        caps = {c for _, c in cards}
        if not caps or not known or caps <= known:
            return None
        here = ", ".join(f"{name} is compute capability {a}.{b}" for name, (a, b) in cards)
        return (
            f"target {self.arch}: plans for compute capability {_capabilities_text(known)}, but {here}; "
            "building here for serving there"
        )

    def plan(self, fmt: Format, kind: LayerKind, *, decomposed: bool = False) -> KernelPlan:
        """The kernel plan for `fmt` on a `kind` layer, or `Refused` saying why nothing serves it.

        The plan names this target (`KernelPlan.target`), so a step storing a module in a format of its own
        asks the right target for that format's plan.

        Args:
            decomposed: the routed bank's router selects its experts outside the fused kernel (vLLM's
                ``RoutingMethodType.Custom``; `ExpertBank.custom_router`). A table with a decomposed row names
                the kernel such a bank binds; one without a decomposed row keeps its row.
        """
        table = TARGETS[self.arch]
        plan = (
            table(self, fmt, kind, decomposed=True)
            if decomposed and isinstance(table, Table)
            else table(self, fmt, kind)
        )
        return replace(plan, target=self)

    def binds(self, fmt: Format, shape: tuple[int, int], serve: dict) -> str:
        """The dense kernel vLLM binds for a module of `fmt` and shape (out, in) under `serve` (`merge_serve`'s).

        Raises:
            Refused: this architecture has no binding rule, or its rule cannot tell.
        """
        from .step import Refused

        if self.arch not in BINDINGS:
            raise Refused(
                f"target {self.arch}: no binding rule says which kernel serves {fmt.name} beside the body"
            )
        return BINDINGS[self.arch](self, fmt, shape, serve.get("args") or {}, serve.get("env") or {})

    def binds_moe(self, fmt: Format, shape: tuple[int, int], serve: dict, *, decomposed: bool = False) -> str:
        """The expert kernel for ``(hidden, unpadded intermediate)`` under the checkpoint's settings.

        A bank whose router selects its experts outside the fused kernel (`ExpertBank.custom_router`) binds a
        different expert kernel on a table that has a decomposed row: where the architecture's rule answers
        with the table's stock row, this returns the decomposed row instead (sm100's modular kernel). A rule
        whose answer the serve settings changed (an engine argument selects another kernel) is left as it is.

        Raises:
            Refused: the target has no MoE binding rule, or the selected kernel cannot serve this shape.
        """
        from .step import Refused

        key = f"{self.arch}_moe"
        if key not in BINDINGS:
            raise Refused(f"target {self.arch}: no MoE binding rule validates {fmt.name} expert shapes")
        bound = BINDINGS[key](self, fmt, shape, serve.get("args") or {}, serve.get("env") or {})
        if decomposed and bound == self.plan(fmt, "moe").kernel:
            bound = self.plan(fmt, "moe", decomposed=True).kernel
        return bound


def capabilities_serving(
    fmt: Format, kind: LayerKind = "dense", vllm: str = Target.vllm
) -> frozenset[tuple[int, int]]:
    """The compute capabilities whose table plans ``fmt`` on ``kind`` layers, read from `TARGETS`.

    A table plans the format when calling it does not raise `Refused`; the capabilities are the ones the table
    declares (`Table.capabilities`). The tables are the only owner of the fact: nothing else in QDA names which
    hardware a format needs.
    """
    from .step import Refused

    caps: set[tuple[int, int]] = set()
    for arch, table in TARGETS.items():
        try:
            table(Target(arch, vllm), fmt, kind)
        except Refused:
            continue
        caps |= set(getattr(table, "capabilities", ()))
    return frozenset(caps)


def serving_refusal(
    fmt: Format, cards: Sequence[tuple[str, tuple[int, int]]], vllm: str = Target.vllm
) -> str | None:
    """Why serving ``fmt`` on ``cards`` should be refused before an engine starts, or None.

    A format a table plans only for newer capabilities cannot be served on a card below the lowest of them;
    vLLM's CUTLASS W4A8 kernel is the case this exists for (compute capability 9.0 only), and its FP8
    W8A8 kernel starts at 8.9. The message names the format, the capability it needs, what the card is, and
    the W4A16 alternative.

    Only formats whose activations a kernel family fixes (``fp8_token``: W4A8 and FP8 W8A8) are refused here. A
    weight-only format a table does not plan yet is not a verdict: an sm90-built W4A16/W3A16 checkpoint still
    loads on a card whose table has not landed, so refusing it would reject working checkpoints.

    Returns:
        One sentence, or None when the format is served here or names no capability floor.
    """
    if fmt.act != "fp8_token":
        return None
    served = capabilities_serving(fmt, "dense", vllm) | capabilities_serving(fmt, "moe", vllm)
    here = {c for _, c in cards}
    if not served or not here or served & here:
        return None
    need = " or ".join(f"{a}.{b}" for a, b in sorted(served))
    where = ", ".join(f"{name} is compute capability {a}.{b}" for name, (a, b) in cards)
    what, example = ("FP8 W8A8", "") if fmt.weight_type == "fp8" else ("CUTLASS W4A8", " (H100 / H200)")
    return (
        f"{fmt.name} needs vLLM's {what} kernel, which serves compute capability {need} only{example}; "
        f"{where}. Use a w4a16 checkpoint, which runs here."
    )


@dataclass(frozen=True)
class Table:
    """A `TARGETS` entry that also says which GPU names mean it and what its rows rest on.

    Any callable is a table; `Target.parse` reads an ``aliases`` attribute if it has one.

    Attributes:
        rows: ``rows(target, fmt, kind)``: the `KernelPlan`, or `Refused`.
        decomposed: ``decomposed(target, fmt, kind)``: the row for a bank whose router selects its experts
            outside the fused kernel (vLLM's ``RoutingMethodType.Custom``). None: the rows serve both, which is
            what every table but sm100 says.
        aliases: GPU names that mean this architecture in a target name.
        observed: (vLLM, kernel, layer kind, symmetric) rows whose binding was observed on a card, any weight
            width; a row that adds the width as a fifth item (``(..., 4)``) counts for that width only, as a
            kernel that serves several (Marlin: int4 and int8) may have been observed at one.
        verified_vllm: vLLM versions all of whose rows count as observed.
        unverified: the note on any other row; ``{vllm}`` is the target's version.
        versions: vLLM versions its rows differ on, beside `Target.vllm`; `output.serve_policy` asks each.
        backends: backend pins it accepts; `output.serve_policy` asks each.
        capabilities: (major, minor) compute capabilities of the cards its rows plan for; a build that names no
            target takes the table covering its cards (`Target.on_cards`). Empty: the table makes no claim about
            cards, so a target naming it is never refused for the hardware it runs on.
    """

    rows: Callable[[Target, Format, LayerKind], KernelPlan]
    decomposed: Callable[[Target, Format, LayerKind], KernelPlan] | None = None
    aliases: tuple[str, ...] = ()
    observed: frozenset[tuple[str, str, str, bool] | tuple[str, str, str, bool, int]] = frozenset()
    verified_vllm: tuple[str, ...] = ()
    unverified: str = UNVERIFIED_ON_HARDWARE
    versions: tuple[str, ...] = ()
    backends: tuple[str | None, ...] = (None,)
    capabilities: tuple[tuple[int, int], ...] = ()

    def __call__(
        self, target: Target, fmt: Format, kind: LayerKind, *, decomposed: bool = False
    ) -> KernelPlan:
        """The row's plan, with the `unverified` note unless something observed it."""
        row = self.decomposed if decomposed and self.decomposed is not None else self.rows
        plan = row(target, fmt, kind)
        row_key = (target.vllm, plan.kernel, kind, fmt.symmetric)
        if (
            target.vllm in self.verified_vllm
            or row_key in self.observed
            or (*row_key, fmt.weight_bits) in self.observed
        ):
            return plan
        return replace(plan, notes=(*plan.notes, self.unverified.format(vllm=target.vllm)))


#: (vLLM, kernel, layer kind, symmetric) rows whose binding was observed on an H200 with this plan's serve
#: settings: Qwen3-14B int4 g128 sym (160/160 linears), Qwen3-30B-A3B-w4a16 (sym g128; dense 96 Marlin,
#: MoE 48 MarlinExperts with linear_backend + moe_backend = marlin; left alone: Machete + HummingIndexedExperts),
#: Qwen3-8B W3A16 RTN (144 HummingLinearKernel, no args), Qwen3-8B GPTQ sym and AWQ asym. Everything else on
#: vLLM != 0.27.1 is marked unverified. Every Marlin run recorded here (dense and experts) was int4, so the Marlin
#: rows are keyed at 4 bits: a W8A16 Marlin plan on 0.33.0 (``:marlin``) keeps the unverified note.
_SM90_OBSERVED = frozenset({
    ("0.33.0", "MarlinLinearKernel", "dense", True, 4),
    ("0.33.0", "MarlinLinearKernel", "dense", False, 4),
    ("0.33.0", "MarlinExperts", "moe", True, 4),
    ("0.33.0", "HummingLinearKernel", "dense", True),
    # Qwen3-8B GPTQ w4a8
    ("0.33.0", "CutlassW4A8LinearKernel", "dense", True),
    # tiny Qwen3-MoE w4a8 (hidden 256, expert intermediate 256): prompt logprobs == a torch reference with
    # per-token FP8 activations, 3.4 nats from the same reference with bf16 activations
    ("0.33.0", "CutlassExpertsW4A8Fp8", "moe", True),
    # tiny Qwen3-MoE asym export: MarlinExperts bound and prompt logprobs == an independent dequant reference
    # (tests/test_vllm_gpu.py, w4a16); Qwen3-30B-A3B GPTQ sym also binds MarlinExperts (vLLM 0.27.1)
    ("0.33.0", "MarlinExperts", "moe", False, 4),
    # w3a16 on tiny random models through tests/test_vllm_gpu.py, served by stock vLLM 0.33.0 on one H200
    # on 2026-11-09: each cell bound the kernel its build.json planned, confirmed by vLLM's own log lines,
    # and its prompt logprobs matched a dequantized reference
    # (mean/max error: sym MoE 0.00084/0.00485, asym dense 0.00142/0.00509, asym MoE 0.00153/0.01186;
    # the harness asserts bound == expected, mean < 0.02 and max < 0.15). 4 passed, 0 skipped.
    ("0.33.0", "HummingIndexedExperts", "moe", True),
    ("0.33.0", "HummingLinearKernel", "dense", False),
    ("0.33.0", "HummingIndexedExperts", "moe", False),
    # W4A16 (asym g128) with Machete off and nothing pinned, on one H100 (tests/test_kl_card_kernels_gpu.py):
    # Qwen3-0.6B binds HummingLinearKernel; a two-layer Qwen3-MoE binds HummingIndexedExperts with the preset's
    # symmetric experts and MarlinExperts with asymmetric ones, under `qda-kl kl` and the printed serve line alike
})  # fmt: skip


# ---------------------------------------------------------------- the sm90 table

_FP8_GRID = (
    "CUTLASS W4A8 casts the bf16 group scales to FP8: per row chan = amax(scale) / 448, "
    "f8 = e4m3(e4m3(scale / chan) / 8), served weight = e4m3(q * f8) * (8 * chan) "
    "(vLLM convert_bf16_scales_to_fp8)",
    "activations: dynamic per-token FP8 E4M3, scale = absmax / 448 floored at 1 / (448 * 512)",
)


#: Kernels no vLLM engine qda starts may bind. `ban_kernels` adds them to every engine's environment, every sm90
#: plan records them in ``serve.env``, and `sm90_binds` never predicts them.
BANNED_KERNELS = ("MacheteLinearKernel",)


def ban_kernels(env: Mapping[str, str]) -> dict[str, str]:
    """`env` with every `BANNED_KERNELS` class added to the end of its ``VLLM_DISABLED_KERNELS`` list.

    The one rule every vLLM launch applies: the checkpoint's plan, its card serve line, `qda-kl kl`, `qda eval`
    and the agent's and judge's engines, for qda's checkpoints and anyone else's.
    """
    named = [k for k in env.get("VLLM_DISABLED_KERNELS", "").split(",") if k]
    return {**env, "VLLM_DISABLED_KERNELS": ",".join(dict.fromkeys([*named, *BANNED_KERNELS]))}


#: serve arguments that pin Marlin; an sm90 record whose target names no backend but carries one was built on the
#: Marlin default that predates the Machete ban (the published sm90 W4A16 checkpoints)
_MARLIN_PINS = {"linear_backend": "marlin", "moe_backend": "marlin"}


def stored_marlin_pin(record: Mapping[str, Any]) -> bool:
    """Whether a ``build.json`` record is an sm90 build on the old Marlin default (`unpinned` serves it as built now).

    A ``:marlin`` build names the backend in its target and keeps its pin.
    """
    target, args = record.get("target"), (record.get("serve") or {}).get("args") or {}
    if not isinstance(target, str) or not isinstance(args, dict):
        return False
    t = Target.stored(target)
    return t.arch == "sm90" and t.backend is None and any(args.get(k) == v for k, v in _MARLIN_PINS.items())


def unpinned(serve: Mapping[str, Any]) -> dict[str, Any]:
    """``serve`` without its Marlin pins: vLLM picks the kernel, as for an sm90 build today (Machete stays off)."""
    args = {k: v for k, v in (serve.get("args") or {}).items() if _MARLIN_PINS.get(k) != v}
    return {**serve, "args": args}


#: dense WNA16 kernels an sm90 target may pin (None = the default: vLLM's own pick with `BANNED_KERNELS` off)
_DENSE_BACKENDS = (None, "marlin", "humming")


def _sm90_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """Hopper: the kernel plan for `fmt` on a `kind` layer, or `Refused` saying why nothing serves it.

    Rows come from research/gpu-format-matrix.md (vLLM source read at 0.27.1 and 0.33.0). Rows for
    vLLM other than 0.27.1 carry an ``unverified`` note until a serving check has run on a card
    (`_SM90_OBSERVED`). The kernel names are the exact class names vLLM logs; qda-kl refuses to score a
    checkpoint whose bound kernels differ from ``build.json["serve"]["kernels"]``.
    """
    from .step import Refused

    ver = _version(target.vllm)
    if fmt.act == "int8_token":
        why = _int8_w8a8_refusal(fmt, kind, target.vllm)
    elif fmt.weight_type == "fp8":
        why = _fp8_w8a8_refusal(fmt, kind, target)
    else:
        why = _servable(fmt, ver)
    if why:
        raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")
    if target.backend not in _DENSE_BACKENDS:
        raise Refused(f"backend {target.backend!r}: one of {sorted(b for b in _DENSE_BACKENDS if b)}")
    plan = _sm90(fmt, kind, ver, target.backend)
    return replace(plan, serve_env=ban_kernels(plan.serve_env))


sm90 = Table(
    _sm90_rows,
    aliases=("h100", "h200", "gh200", "hopper"),
    observed=_SM90_OBSERVED,
    verified_vllm=("0.27.1",),
    unverified="unverified on vLLM {vllm}: from source reading only",
    versions=("0.27.1", "0.30.0", "0.32.0", "0.33.0"),
    backends=_DENSE_BACKENDS,
    capabilities=((9, 0),),
)


def llama_cpp(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:  # noqa: ARG001
    """llama.cpp: the plan for a format the GGUF writer can store solved codes in, or `Refused`.

    A GGUF names a type per tensor and llama.cpp serves every type the Q4_K_M mix emits, but a target answers
    a narrower question: which formats may a build *solve units in*. The writer packs solved codes as Q4_K,
    Q5_K, Q6_K, IQ4_XS, IQ4_NL and Q3_K records, so those formats are what this declares. A tensor stored
    Q5_0, Q8_0, F32 or BF16 reaches the file from the template, never from a solved unit, and no grid writes it.

    There is no kernel table to version: llama.cpp reads the records in the file, and this build's parity is
    checked against its own quantizer byte for byte rather than against a kernel's behaviour.
    """
    from .step import Refused

    want = {
        # format: (stores a zero point, the GGUF type that holds its codes)
        Q4K: (True, GGUF_TYPES[Q4K]),
        Q5K: (True, GGUF_TYPES[Q5K]),
        Q6K: (False, GGUF_TYPES[Q6K]),
        IQ4XS_FORMAT: (False, GGUF_TYPES[IQ4XS_FORMAT]),
        IQ4NL_FORMAT: (False, GGUF_TYPES[IQ4NL_FORMAT]),
        Q3K_FORMAT: (False, GGUF_TYPES[Q3K_FORMAT]),
    }
    if fmt.name not in want:
        named = ", ".join(want)
        raise Refused(
            f"{fmt.name} {kind} on llama.cpp: a GGUF holds solved codes as {named}; other types are "
            "written from the template, so no unit is stored in them"
        )
    zero_point, record = want[fmt.name]
    return KernelPlan(
        kernel="llama.cpp",
        zero_point=zero_point,
        # ``static`` reorders columns during the solve and stores nothing, so the records are ordinary Q4_K
        # or Q6_K and llama.cpp reads them as it reads any others. A mode that stored a permutation would not
        # be loadable, and none is offered.
        act_order=("none", "static"),
        notes=(
            f"{record} records written by this exporter and checked byte for byte against llama.cpp's own",
        ),
    )


def _version(v: str) -> tuple[int, int]:
    major, minor, *_ = (int(x) for x in v.split(".")[:3] + ["0", "0"])
    return major, minor


def _int8_w8a8_refusal(fmt: Format, kind: LayerKind, vllm: str) -> str | None:
    """Why vLLM's INT8 W8A8 does not serve `fmt`, or None. Its dense kernel has no capability floor."""
    if vllm != "0.33.0":
        return "the INT8 W8A8 row is read from vLLM 0.33.0; which kernel binds it here is not known"
    if fmt.weight_type != "int" or (fmt.weight_bits, fmt.group_size, fmt.symmetric) != (8, -1, True):
        return (
            "vLLM's INT8 W8A8 takes only int8 weights symmetric per channel, not asymmetric or grouped ones"
        )
    if kind != "dense":
        return "no INT8 W8A8 expert row: vLLM serves those through Triton or Humming, which QDA has not read"
    return None


#: vLLM's ``CompressedTensorsW8A8Int8``, the first INT8 kernel it tries on any CUDA card.
_INT8_W8A8 = KernelPlan(
    "CutlassInt8ScaledMMLinearKernel",
    act_dtype="int8",
    zero_point=False,
    notes=("int8 weights times per-token int8 activations, then the fp32 weight_scale; no bf16 dequant",),
)


def _fp8_w8a8_refusal(fmt: Format, kind: LayerKind, target: Target) -> str | None:
    """Why vLLM's FP8 W8A8 does not serve `fmt` on `target`, or None.

    ``CompressedTensorsW8A8Fp8`` takes e4m3 weights, symmetric per output channel or per tensor, with
    dynamic per-token e4m3 activations, and admits compute capability 8.9 and newer
    (``compressed_tensors_w8a8_fp8.py:57-85``). Below that the same checkpoint plans through
    ``CompressedTensorsW8A16Fp8`` instead (``compressed_tensors.py:897-908``), which dequantizes the weights
    to bf16 activations: the card serves W8A16, not W8A8. At 8.9 and up, FlashInfer comes first in
    ``_POSSIBLE_FP8_KERNELS`` (``kernels/linear/__init__.py:458-466``) but needs per-tensor scales on both
    sides (``kernels/linear/scaled_mm/flashinfer.py:56-65``); ours are per channel and per token, so
    ``CutlassFP8ScaledMMLinearKernel`` (``scaled_mm/cutlass.py:157-175``) takes the module.
    """
    if target.vllm != "0.33.0":
        return "the FP8 W8A8 row is read from vLLM 0.33.0; which kernel binds it here is not known"
    if fmt.weight_type != "fp8" or (fmt.weight_bits, fmt.group_size, fmt.symmetric) != (8, -1, True):
        return "vLLM's FP8 W8A8 takes e4m3 weights symmetric per channel, with dynamic per-token e4m3 activations"
    caps = getattr(TARGETS[target.arch], "capabilities", ())
    if caps and all(c < (8, 9) for c in caps):
        return (
            "FP8 W8A8 needs compute capability 8.9 or newer: below it vLLM serves this checkpoint through "
            "Marlin as W8A16 (bf16 activations), not W8A8; use w4a16"
        )
    if kind != "dense":
        return (
            "no FP8 expert row: QDA writes no FP8 expert banks, and vLLM's compressed-tensors MoE FP8 "
            "methods (Triton, DeepGEMM, FlashInfer and Marlin) are not read; "
            "use w4a16"
        )
    return None


#: vLLM's Cutlass FP8 W8A8, the dense kernel it binds for channel weights with dynamic per-token activations.
_FP8_W8A8_GRID = (
    "weight = e4m3(weight / scale) * scale, scale = absmax / 448 per output channel, fp32; CUTLASS pads K "
    "and N to 16",
    "activations: dynamic per-token FP8 E4M3, scale = absmax / 448 floored at 1 / (448 * 512)",
)

_FP8_W8A8 = KernelPlan(
    "CutlassFP8ScaledMMLinearKernel",
    scale_dtype="fp32",
    act_dtype="fp8_e4m3",
    zero_point=False,
    notes=_FP8_W8A8_GRID,
)


def _servable(fmt: Format, ver: tuple[int, int]) -> str | None:
    """Why no sm90 kernel serves `fmt`, or None."""
    if fmt.weight_type != "int":
        return "only integer weight codes are served on sm90 in v0"
    if fmt.act == "fp8_token":
        if (fmt.weight_bits, fmt.group_size, fmt.symmetric) != (4, 128, True):
            return "CUTLASS W4A8 takes int4, group 128, symmetric weights only"
        return None
    if fmt.act != "bf16":
        return f"{fmt.act} activations: no sm90 kernel row serves them"
    if fmt.weight_bits not in (3, 4, 8):
        return f"no WNA16 kernel serves int{fmt.weight_bits}"
    if fmt.weight_bits == 3:
        if fmt.group_size not in (64, 128):
            return "Humming int3 is tuned for groups 64 and 128 only"
        if not fmt.symmetric and ver < (0, 33):
            return "int3 zero points load from vLLM 0.33 (0.27.1's zero-point types are uint4/uint8)"
        return None
    if fmt.group_size not in (-1, 32, 64, 128):
        return "Marlin groups are -1, 32, 64 or 128"
    return None


def _sm90(fmt: Format, kind: LayerKind, ver: tuple[int, int], backend: str | None) -> KernelPlan:
    from .step import Refused

    # vLLM 0.32 puts Humming ahead of Marlin on sm90: https://github.com/vllm-project/vllm/pull/56997
    new = ver >= (0, 32)
    acts = ("none", "static") if ver >= (0, 30) else ("none", "static", "group")
    zp = not fmt.symmetric
    if fmt.act == "int8_token":
        return _INT8_W8A8
    if fmt.weight_type == "fp8":
        return _FP8_W8A8
    if fmt.act == "fp8_token":
        if kind == "dense":
            return KernelPlan(
                "CutlassW4A8LinearKernel",
                "fp8_e4m3",
                "fp8_e4m3",
                False,
                acts,
                (*_FP8_GRID, "K and N must be multiples of 128"),
            )
        return KernelPlan(
            "CutlassExpertsW4A8Fp8",
            "fp8_e4m3",
            "fp8_e4m3",
            False,
            acts,
            (*_FP8_GRID, "hidden and intermediate-per-partition must be multiples of 256"),
        )
    if fmt.weight_bits == 3:
        if kind == "dense":
            return KernelPlan(
                "HummingLinearKernel",
                zero_point=zp,
                act_order=acts,
                notes=("Marlin has no int3; Humming serves W3 on every arch", _HUMMING_SCOPE),
            )
        if ver < (0, 29):
            return KernelPlan(
                "HummingExperts",
                zero_point=zp,
                act_order=acts,
                notes=("vLLM < 0.29 needs the qda_humming_moe plugin for int3 experts",),
                serve_plugins=("qda_humming_moe",),
            )
        return KernelPlan("HummingIndexedExperts", zero_point=zp, act_order=acts, notes=(_HUMMING_SCOPE,))
    if kind == "dense":
        return _dense_wna16(fmt, acts, ver, backend)
    if backend == "humming":
        if not new:
            raise Refused("target :humming for WNA16 MoE requires vLLM >= 0.32")
        return KernelPlan(
            "HummingIndexedExperts",
            zero_point=zp,
            act_order=acts,
            serve_args={"moe_backend": "humming"},
            notes=(_HUMMING_SCOPE,),
        )
    if new and backend == "marlin":
        return KernelPlan(
            "MarlinExperts",
            zero_point=zp,
            act_order=acts,
            serve_args={"moe_backend": "marlin"},
            notes=("vLLM >= 0.32 prefers Humming for MoE on sm90; moe_backend=marlin keeps Marlin",),
        )
    if new and humming_takes_experts(fmt):  # vLLM's own pick: Humming ahead of Marlin on sm90
        return KernelPlan("HummingIndexedExperts", zero_point=zp, act_order=acts, notes=(_HUMMING_SCOPE,))
    return KernelPlan(
        "MarlinExperts", zero_point=zp, act_order=acts, notes=("bf16 dequant (q - z) * s, fp32 accumulate",)
    )


#: vLLM versions whose ``choose_mp_linear_kernel`` `sm90_binds` was written from
SM90_BINDS_READ = ("0.33.0",)


def _marlin_fits(group: int, in_features: int) -> bool:
    """Marlin's shape rule: a group must not straddle a partition (padding fixes tile misalignment)."""
    return group == -1 or group >= in_features or in_features % group == 0


def sm90_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, env: dict) -> str:
    """The dense WNA16 kernel vLLM picks on Hopper (``choose_mp_linear_kernel``, source read at 0.33.0).

    Selection changes between vLLM versions (Humming moved ahead of Marlin in 0.32), so any version not in
    `SM90_BINDS_READ` is refused rather than predicted.

    For a weight-only format, vLLM's priority is Machete, Humming, Marlin, and Machete is never a candidate:
    every engine qda starts disables `BANNED_KERNELS` (`ban_kernels`). ``linear_backend`` keeps only its kernel
    (both when it names neither), ``VLLM_DISABLED_KERNELS`` drops kernels; the first whose ``can_implement``
    admits the module wins. Humming: any int weight-only module. Marlin: int4/int8, groups -1/32/64/128, in a
    multiple of the group.

    A format whose activations a kernel family fixes (W4A8's and FP8 W8A8's per-token FP8, int8) is not a
    weight-only head: the table's plan is the one owner of which kernel serves it (`_servable`, `_sm90`), and
    this rule checks the module's shape against the same `can_implement` facts every kernel states
    (`shape_limit`). That is the same split `sm90_moe_binds` makes, so the plan, the width check and the build
    cannot disagree about W4A8. Anything else, or no kernel left, is refused rather than guessed.
    """
    from .step import Refused

    out_f, in_f = shape
    if target.vllm not in SM90_BINDS_READ:
        raise Refused(
            f"vLLM {target.vllm}: the sm90 rule for a head in its own format is read from vLLM "
            f"{', '.join(SM90_BINDS_READ)}; which kernel binds it on {target.vllm} is not known"
        )
    if fmt.weight_type not in ("int", "fp8"):
        raise Refused(f"{fmt.name}: the sm90 binding rule covers integer and FP8 weight formats")
    if fmt.act != "bf16":
        kernel = target.plan(fmt, "dense").kernel
        if why := shape_limit(kernel, fmt, shape):
            raise Refused(f"{fmt.name} [{out_f}, {in_f}]: {kernel} cannot serve it: {why}")
        return kernel
    order = [b for b in ("humming", "marlin") if b == args.get("linear_backend")] or ["humming", "marlin"]
    off = set(ban_kernels(env)["VLLM_DISABLED_KERNELS"].split(","))
    b, g = fmt.weight_bits, fmt.group_size
    fits = {
        "humming": True,
        "marlin": b in (4, 8) and g in (-1, 32, 64, 128) and _marlin_fits(g, in_f),
    }
    for name in order:
        if _KERNEL[name] not in off and fits[name]:
            return _KERNEL[name]
    raise Refused(f"{fmt.name} [{out_f}, {in_f}]: none of {order} binds it on vLLM {target.vllm} here")


def shape_limit(kernel: str, fmt: Format, shape: tuple[int, int]) -> str | None:
    """The first width `kernel` cannot serve, or None if it serves them all.

    ``shape`` is a module's ``(out, in)``, or for a dense `LinearGroup` the ``(out, in)`` of the fused
    linear vLLM serves; for a routed bank it is ``(hidden, intermediate)``, the down projection's rows and
    columns. The rules are vLLM 0.33's own: MarlinExperts needs ``hidden % 128`` and
    the group size dividing ``intermediate``; CUTLASS W4A8 dense needs both
    ``in % 128`` and ``out % 128`` (``kernels/linear/mixed_precision/cutlass.py``, ``can_implement``), and
    its expert method both ``hidden % 256`` and ``intermediate % 256``. Marlin pads intermediate thread
    tiles, but never its hidden I/O extent. Humming pads its tiles.
    """
    out, inp = shape
    if kernel == "MarlinExperts":
        if out % 128:
            return f"hidden {out} % 128 = {out % 128}; Marlin requires hidden % 128 == 0"
        if fmt.group_size > 0 and inp % fmt.group_size:
            return f"intermediate {inp} is not divisible by group size {fmt.group_size}"
    if kernel == "CutlassW4A8LinearKernel" and (out % 128 or inp % 128):
        return f"{out} out by {inp} in (not multiples of 128)"
    if kernel == "CutlassExpertsW4A8Fp8" and (out % 256 or inp % 256):
        return f"hidden {out} and intermediate {inp} must both be multiples of 256"
    return None


def sm90_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, _env: dict) -> str:
    """Bind stock sm90 experts and reject their shape limits before a checkpoint is built."""
    from .step import Refused

    kernel = target.plan(fmt, "moe").kernel
    if fmt.act == "bf16":
        backend = args.get("moe_backend")
        if backend in (None, "auto"):
            if _version(target.vllm) >= (0, 32):
                kernel = "HummingIndexedExperts" if humming_takes_experts(fmt) else "MarlinExperts"
        elif backend == "humming":
            kernel = "HummingIndexedExperts"
        elif backend == "marlin":
            if fmt.weight_bits == 3:
                raise Refused("Marlin has no INT3 expert kernel; use target: sm90@0.33.0:humming")
            kernel = "MarlinExperts"
        else:
            raise Refused(f"MoE backend {backend!r}: no sm90 binding rule validates this backend")
    if why := shape_limit(kernel, fmt, shape):
        alternative = (
            f"; use target: {target.arch}@{target.vllm}:humming to select Humming experts"
            if kernel == "MarlinExperts" and _version(target.vllm) >= (0, 32)
            else "; use a format or target whose expert kernel supports these dimensions"
        )
        raise Refused(f"{kernel} cannot serve {fmt.name}: {why}{alternative}")
    return kernel


def llama_cpp_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, env: dict) -> str:  # noqa: ARG001
    """llama.cpp reads the records in the file, so the kernel it uses is not a property of the checkpoint.

    A binding rule exists to say which kernel a serving stack will pick for a shape, and to refuse a shape it
    would pick badly for. llama.cpp chooses per type at load time from the tensor's own header, so there is
    nothing to predict and nothing to refuse: a shape the writer could not store was refused by the mix long
    before this. The name is returned for the record, as the plan's kernel.
    """
    return "llama.cpp"


def llama_cpp_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, env: dict) -> str:
    """`llama_cpp_binds` for an expert bank, which llama.cpp also reads from the tensor's own header.

    A separate function rather than the same one under two names: the registry holds one name per object, and
    a dense rule and a MoE rule are different questions even where both have the same answer today.
    """
    return llama_cpp_binds(target, fmt, shape, args, env)


# ---------------------------------------------------------------- the sm80 table


def _sm80_refusal(target: Target, fmt: Format, kind: LayerKind) -> str | None:
    """Why no sm80 row serves `fmt`, or None."""
    if target.vllm != "0.33.0":
        return "the sm80 rows are read from vLLM 0.33.0; which kernel binds on another version is not known"
    if target.backend not in (None, "marlin"):
        return f"backend {target.backend!r}: Marlin is the only WNA16 kernel with an sm80 row"
    if fmt.weight_type == "fp8":
        return _fp8_w8a8_refusal(fmt, kind, target)
    if fmt.act == "fp8_token":
        return W4A8_NEEDS_HOPPER
    if fmt.act == "int8_token":
        return _int8_w8a8_refusal(fmt, kind, target.vllm)
    if fmt.weight_type != "int" or fmt.act != "bf16" or fmt.weight_bits not in (3, 4):
        return (
            "this format has no sm80 row (int4 and int3 weight-only are the WNA16 rows here); for this card "
            "build w4a16 or w3a16"
        )
    if fmt.weight_bits == 3:
        if fmt.group_size not in (64, 128):
            return "Humming int3 is tuned for groups 64 and 128 only"
        return None
    if fmt.group_size not in (-1, 32, 64, 128):
        return "Marlin groups are -1, 32, 64 or 128"
    return None


def _sm80_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """Ampere: INT8 W8A8 through CUTLASS, W4A16 through Marlin, W3A16 through Humming.

    The W4A16 (asymmetric group 128) dense and expert bindings and the W3A16 (symmetric group 128) dense and
    expert bindings are observed on an A100 (vLLM 0.33.0); every other row is read from vLLM 0.33.0's source.
    Machete and CUTLASS W4A8 need compute capability 90 and vLLM prefers Humming on 90 only, so Marlin is
    what vLLM picks for int4 here with no serve settings, dense and MoE. Marlin stores no 3-bit codes, so
    int3 takes Ada's Humming row (`_humming_int3`): Humming's ``get_min_capability`` is 75. FP8 W8A8 needs
    8.9, and below it vLLM serves the checkpoint as W8A16, so no row here serves it (`_fp8_w8a8_refusal`).
    """
    from .step import Refused

    if why := _sm80_refusal(target, fmt, kind):
        raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")
    if fmt.act == "int8_token":
        return _INT8_W8A8
    if fmt.weight_bits == 3:
        return _humming_int3(fmt, kind)
    notes = ("bf16 dequant (q - z) * s; vLLM says fp16 is faster than bf16 on Marlin before sm90",)
    kernel = "MarlinLinearKernel" if kind == "dense" else "MarlinExperts"
    return KernelPlan(kernel, zero_point=not fmt.symmetric, notes=notes)


#: Rows an Ampere card has run: W4A16 (asymmetric, the preset) dense and expert rows through Marlin and
#: symmetric W3A16 dense and expert rows through Humming, all with no serve settings and scored through `qda-kl`
#: against a reference on vLLM 0.33.0 (A100: Qwen3.5-4B KL 0.020576 W4A16, 0.103828 W3A16; Qwen3-30B-A3B
#: 0.155589 W3A16). The INT8 rows have not run here and keep the table's unverified note; Marlin ran at 4 bits only,
#: so its rows are keyed at 4.
_SM80_OBSERVED = frozenset({
    ("0.33.0", "MarlinLinearKernel", "dense", False, 4),
    ("0.33.0", "MarlinExperts", "moe", False, 4),
    ("0.33.0", "HummingLinearKernel", "dense", True),
    ("0.33.0", "HummingIndexedExperts", "moe", True),
})  # fmt: skip

sm80 = Table(
    _sm80_rows,
    aliases=("sm86", "ampere", "a100", "a30", "a10", "a40", "rtx3060", "rtx3070", "rtx3080", "rtx3090"),
    observed=_SM80_OBSERVED,
    backends=(None, "marlin"),
    capabilities=((8, 0), (8, 6)),
)


def sm80_binds(target: Target, fmt: Format, shape: tuple[int, int], _args: dict, _env: dict) -> str:
    """The dense kernel on Ampere: the row's, refused for a format with no row or a shape Marlin cannot take."""
    from .step import Refused

    kernel = target.plan(fmt, "dense").kernel
    if kernel == "MarlinLinearKernel" and not _marlin_fits(fmt.group_size, shape[1]):
        raise Refused(
            f"{fmt.name} {list(shape)}: Marlin cannot take these inputs in groups of {fmt.group_size}"
        )
    return kernel


def sm80_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], _args: dict, _env: dict) -> str:
    """The expert kernel on Ampere: the row's, refused for a shape it cannot take."""
    from .step import Refused

    kernel = target.plan(fmt, "moe").kernel
    if why := shape_limit(kernel, fmt, shape):
        raise Refused(f"{kernel} cannot serve {fmt.name}: {why}")
    return kernel


# ---------------------------------------------------------------- the ada and Blackwell-consumer tables

#: Rows a card of these architectures has run: W4A16 (asymmetric, the preset) through Marlin and symmetric
#: W3A16 through Humming, dense and expert, both on vLLM 0.33.0 (L40S: KL 0.025030 W4A16; RTX PRO 6000: KL
#: 0.025359 W4A16, 0.153409 W3A16). Dense symmetric W8A16 binds Marlin on an RTX PRO 6000 (no L40S has run
#: it): the published Qwen3.5-4B W4A16 (KL 0.010664), whose 8-bit attention groups sit beside its 4-bit
#: body, bound MarlinLinearKernel only. So Ada's Marlin rows, dense and expert, are observed at 4 bits (the width is
#: the fifth item of the key; no card has run 8-bit MarlinExperts) and `_SM120_OBSERVED` adds the dense 8-bit one;
#: an L40S's W8A16 keeps the unverified note.
#: Humming compiles through NVRTC, so its JIT needs the venv's ``nvidia/*/lib`` (libnvrtc-builtins) on the
#: loader path; the plan's notes say so at serve time.
_ADA_OBSERVED = frozenset({
    ("0.33.0", "MarlinLinearKernel", "dense", False, 4),
    ("0.33.0", "MarlinLinearKernel", "dense", True, 4),
    ("0.33.0", "HummingLinearKernel", "dense", True),
    ("0.33.0", "MarlinExperts", "moe", False, 4),
    ("0.33.0", "HummingIndexedExperts", "moe", True),
})  # fmt: skip


def _ada_refusal(target: Target, fmt: Format, kind: LayerKind) -> str | None:
    """Why no row serves `fmt` on Ada or on an RTX PRO 6000, or None."""
    if target.vllm != "0.33.0":
        return f"the {target.arch} rows are read from vLLM 0.33.0; which kernel binds on another version is not known"
    if target.backend not in (None, "marlin"):
        return f"backend {target.backend!r}: Marlin is the only WNA16 kernel with a {target.arch} row"
    if fmt.weight_type == "fp8":
        return _fp8_w8a8_refusal(fmt, kind, target)
    if fmt.act == "fp8_token":
        return W4A8_NEEDS_HOPPER
    if fmt.act == "int8_token":
        return _int8_w8a8_refusal(fmt, kind, target.vllm)
    if fmt.weight_type != "int" or fmt.act != "bf16":
        return f"the {target.arch} rows are int4 (w4a16) and int3 (w3a16) weight-only"
    if fmt.weight_bits == 3:
        if fmt.group_size not in (64, 128):
            return "Humming int3 is tuned for groups 64 and 128 only"
        return None
    if fmt.weight_bits == 8 and not (fmt.symmetric and kind == "dense"):
        return f"the {target.arch} w8a16 row is dense and symmetric (Marlin's uint8b128)"
    if fmt.weight_bits not in (4, 8):
        return f"the {target.arch} rows are w4a16, w3a16 and w8a16 only"
    if fmt.group_size not in (-1, 32, 64, 128):
        return "Marlin groups are -1, 32, 64 or 128"
    return None


def _humming_int3(fmt: Format, kind: LayerKind) -> KernelPlan:
    """The int3 weight-only row of every card below Hopper that serves it (Ampere, Ada, RTX PRO 6000, B200)."""
    return KernelPlan(
        "HummingLinearKernel" if kind == "dense" else "HummingIndexedExperts",  # sm90's 0.33 expert row
        zero_point=not fmt.symmetric,
        act_order=("none", "static"),
        notes=(
            "Marlin has no int3; Humming serves W3 here",
            _HUMMING_SCOPE,
            "Humming compiles through NVRTC: the venv's nvidia/*/lib (libnvrtc-builtins) must be on "
            "the loader path or the engine refuses with NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
        ),
    )


def _ada_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """Ada (L40S, L4, RTX 4090) and the RTX PRO 6000: W4A16 through Marlin, W3A16 through Humming.

    Both cards behave like Ampere for WNA16 (Marlin, no Machete) and like Hopper for int3 (Humming is the only
    kernel that stores 3-bit codes). CUTLASS W4A8 needs compute capability 90, so W4A8 is refused with the
    alternative named. Both reach the FP8 W8A8 floor (8.9), whose dense rows are the shared `_FP8_W8A8`.
    """
    from .step import Refused

    if why := _ada_refusal(target, fmt, kind):
        raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")
    if fmt.act == "int8_token":
        return _INT8_W8A8
    if fmt.weight_type == "fp8":
        return _FP8_W8A8
    if fmt.weight_bits == 3:
        return _humming_int3(fmt, kind)
    kernel = "MarlinLinearKernel" if kind == "dense" else "MarlinExperts"
    args = {"linear_backend": "marlin"} if kind == "dense" else {"moe_backend": "marlin"}
    return KernelPlan(
        kernel,
        zero_point=not fmt.symmetric,
        act_order=("none", "static"),
        serve_args=args,
        notes=(
            "bf16 dequant (q - z) * s; Marlin is pinned at serve time (vLLM 0.33 puts Humming first on this "
            "card for MoE; Marlin is the kernel this row was measured with)",
        ),
    )


sm89 = Table(
    _ada_rows,
    aliases=("l40s", "l4", "4090", "ada"),
    observed=_ADA_OBSERVED,
    capabilities=((8, 9),),
)

_NVFP4_GRID = (
    "weight = e2m1 code * E4M3 weight_scale / fp32 weight_global_scale, one scale per 16 inputs; vLLM takes the "
    "largest weight_global_scale of the modules it fuses (q/k/v, gate/up) and the gate's for an expert's "
    "gate/up, so a unit stores one",
    "activations: each 16 inputs cut to E2M1 under a dynamic scale e4m3(input_global_scale * absmax / 6), "
    "saturating above 448; a bank's experts share one input_global_scale per projection",
)


def _sm100_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """Blackwell datacenter (B200, B100, GB200, and B300 / GB300 through `sm103`): NVFP4 through FlashInfer,
    FP8 W8A8 through CUTLASS; every other format takes Ada's W4A16 / W3A16 rows.

    vLLM's dense order for W4A4 starts with FlashInfer's CuTe-DSL kernel, admitted on capability 10.x when
    FlashInfer is installed; its experts go to FlashInfer's TensorRT-LLM kernel first. FP8 W8A8 with
    per-channel weights skips FlashInfer (which takes per-tensor scales on both sides) and binds CUTLASS.

    Both dense weight-only rows were measured on a B200: W4A16 (asymmetric, the preset) built with
    ``linear_backend=marlin`` and bound MarlinLinearKernel (KL 0.025812); symmetric W3A16 bound
    HummingLinearKernel (KL 0.169431) with the NVRTC loader-path fix in place. That selection is Ada's, so it
    reuses `_ada_rows`; vLLM's CUTLASS W4A8 kernel needs compute capability 90, so W4A8 is refused here
    naming W4A16.
    """
    from .step import Refused

    if fmt.name == NVFP4_FORMAT or fmt.weight_type == "fp8":
        if target.vllm != "0.33.0":
            raise Refused(
                f"the {target.arch} rows are read from vLLM 0.33.0; which kernel binds on another version "
                "is not known"
            )
        if target.backend is not None:
            raise Refused(f"backend {target.backend!r}: the {target.arch} NVFP4 and FP8 rows pin no backend")
        if fmt.weight_type == "fp8":
            if why := _fp8_w8a8_refusal(fmt, kind, target):
                raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")
            return _FP8_W8A8
        kernel = "FlashInferCuteDslNvFp4LinearKernel" if kind == "dense" else "TrtLlmNvFp4ExpertsMonolithic"
        return KernelPlan(kernel, scale_dtype="nvfp4", act_dtype="nvfp4", zero_point=False, notes=_NVFP4_GRID)
    if fmt.act == "fp8_token":
        raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {W4A8_NEEDS_HOPPER}")
    return _ada_rows(target, fmt, kind)


#: A plan note for the decomposed row: why the same arithmetic binds another expert kernel.
_MODULAR_NOTE = (
    "vLLM routes this bank outside the fused kernel (the router applies a per-expert scale, "
    "RoutingMethodType.Custom), so the experts bind TrtLlmNvFp4ExpertsModular, which takes the selected "
    "experts; the monolithic kernel takes only stock routing methods"
)


def _sm100_decomposed(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """The sm100 row for a bank whose router selects its experts outside the fused kernel.

    vLLM 0.33.0's TRT-LLM NVFP4 experts come as two kernels: ``TrtLlmNvFp4ExpertsMonolithic`` runs the
    router inside the kernel, and its ``_supports_routing_method`` admits only stock routing methods
    (``trtllm_nvfp4_moe.py:493-508``); ``TrtLlmNvFp4ExpertsModular`` takes the routed experts as input
    (``:355``), and the base class admits any routing method. A bank whose router applies work of its own to
    the selected weights (Gemma 4's ``per_expert_scale``) is ``RoutingMethodType.Custom``, so the backend's
    selection loop (``oracle/nvfp4.py:303-317``) falls through to the modular kernel. The dense row is the
    same kernel.

    The two kernels read the same tensors and cut the same values: both quantize the routed tokens with the
    shared ``_quantize_input`` helper, and both take the gate's weight global scale and the bank's one input
    global scale per projection, so the calibrated statistics stay the ones the kernel applies.
    """
    plan = _sm100_rows(target, fmt, kind)
    if kind == "moe" and fmt.name == NVFP4_FORMAT:
        return replace(plan, kernel="TrtLlmNvFp4ExpertsModular", notes=(*plan.notes, _MODULAR_NOTE))
    return plan


#: Rows a B200 has run beyond Ada's: a Qwen3-0.6B ``w4a4`` (NVFP4) RTN checkpoint that QDA built and vLLM
#: 0.33.0 served in a container bound ``FlashInferCuteDslNvFp4LinearKernel`` for its dense layers, and
#: Qwen3.6-35B-A3B ``w4a4`` bound ``TrtLlmNvFp4ExpertsMonolithic`` for its experts.
_SM100_OBSERVED = _ADA_OBSERVED | {
    ("0.33.0", "FlashInferCuteDslNvFp4LinearKernel", "dense", True),
    ("0.33.0", "TrtLlmNvFp4ExpertsMonolithic", "moe", True),
}

sm100 = Table(
    _sm100_rows,
    decomposed=_sm100_decomposed,
    aliases=("b200", "b100", "gb200"),
    # the same rows (Marlin int4, Humming int3, dense and expert) have run on a B200 and an L40S, as has
    # NVFP4, dense and experts, on a B200
    observed=_SM100_OBSERVED,
    capabilities=((10, 0),),
)

#: Rows a GB300 (compute capability 10.3) has run on vLLM 0.33.0 (build, qda-kl score, vllm serve): Qwen3-8B W4A16
#: (MarlinLinearKernel, asymmetric preset KL 0.024862, and symmetric) and W3A16 (HummingLinearKernel), Qwen3-30B-A3B
#: W4A16 (MarlinExperts, KL 0.049679) and W3A16 (HummingIndexedExperts, KL 0.257768), both RTN; the mixed
#: FP8/NVFP4 build of R1-Llama-8B (FP8 attention, NVFP4 MLP) bound ``CutlassFP8ScaledMMLinearKernel`` for its
#: FP8 modules, and Gemma 4 MoE's bank, whose router applies a per-expert scale, bound
#: ``TrtLlmNvFp4ExpertsModular``. A Qwen3-0.6B ``w4a4`` build (``vllm serve`` and ``qda-kl score``, KL 0.1100) bound
#: ``FlashInferCuteDslNvFp4LinearKernel`` for its dense NVFP4 layers. GLM-5.3-Flash ``w4a4`` on four GB300s chose the
#: ``FLASHINFER_TRTLLM`` NVFP4 MoE backend with ``MoEPrepareAndFinalizeNoDPEPMonolithic``, which vLLM 0.33.0 binds to
#: ``TrtLlmNvFp4ExpertsMonolithic`` (``fused_moe/oracle/nvfp4.py``); the log names the backend, not the class.
#: Marlin ran at 4 bits only (the W4A16 builds above), dense and experts, so its rows are keyed at 4: a GB300's
#: W8A16 keeps the unverified note.
_GB300_OBSERVED = frozenset({
    ("0.33.0", "MarlinLinearKernel", "dense", False, 4),
    ("0.33.0", "MarlinLinearKernel", "dense", True, 4),
    ("0.33.0", "HummingLinearKernel", "dense", True),
    ("0.33.0", "CutlassFP8ScaledMMLinearKernel", "dense", True),
    ("0.33.0", "FlashInferCuteDslNvFp4LinearKernel", "dense", True),
    ("0.33.0", "MarlinExperts", "moe", False, 4),
    ("0.33.0", "HummingIndexedExperts", "moe", True),
    ("0.33.0", "TrtLlmNvFp4ExpertsModular", "moe", True),
    ("0.33.0", "TrtLlmNvFp4ExpertsMonolithic", "moe", True),
})  # fmt: skip

#: B300 and GB300: vLLM 0.33.0 selects the same kernels on 10.3 as on 10.0 (Machete and CUTLASS W4A8 need 9.0,
#: Humming outranks Marlin on 9.0 only), so the rows are sm100's; what has run is the GB300's own.
sm103 = Table(
    _sm100_rows,
    decomposed=_sm100_decomposed,
    aliases=("b300", "gb300"),
    observed=_GB300_OBSERVED,
    capabilities=((10, 3),),
)


def _sm120_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    """The RTX PRO 6000 (12.0): NVFP4 through FlashInfer's CUTLASS kernels; every other format takes Ada's rows.

    vLLM 0.33.0's dense NVFP4 order starts with FlashInfer's CuTe-DSL kernel, which admits capability 10.x only;
    the next, ``FlashInferCutlassNvFp4LinearKernel``, admits 10.0 and up, so 12.0 binds it. The expert order
    skips TRT-LLM and CuTe-DSL (10.x only) and binds FlashInfer's CUTLASS experts. Both cut activations and read
    scales as sm100's kernels do.
    """
    from .step import Refused

    if fmt.name != NVFP4_FORMAT:
        return _ada_rows(target, fmt, kind)
    if target.vllm != "0.33.0":
        raise Refused(
            f"the {target.arch} rows are read from vLLM 0.33.0; which kernel binds on another version is not known"
        )
    if target.backend is not None:
        raise Refused(f"backend {target.backend!r}: the {target.arch} NVFP4 rows pin no backend")
    kernel = "FlashInferCutlassNvFp4LinearKernel" if kind == "dense" else "FlashInferExperts"
    return KernelPlan(kernel, scale_dtype="nvfp4", act_dtype="nvfp4", zero_point=False, notes=_NVFP4_GRID)


#: Rows an RTX PRO 6000 has run beyond Ada's, on vLLM 0.33.0: a Qwen3-0.6B ``w4a4`` build (``vllm serve`` and
#: ``qda-kl score``, KL 0.1230) bound ``FlashInferCutlassNvFp4LinearKernel`` for its dense layers; the published
#: Qwen3.6-35B-A3B ``w4a4`` (``vllm serve`` and ``qda-kl score``, KL 0.011091) chose the ``FLASHINFER_CUTLASS`` NVFP4
#: MoE backend, whose one kernel is ``FlashInferExperts``, and bound ``CutlassFP8ScaledMMLinearKernel`` for its
#: FP8 modules.
_SM120_OBSERVED = _ADA_OBSERVED | {
    ("0.33.0", "MarlinLinearKernel", "dense", True, 8),  # Qwen3.5-4B W4A16's 8-bit groups
    ("0.33.0", "FlashInferCutlassNvFp4LinearKernel", "dense", True),
    ("0.33.0", "CutlassFP8ScaledMMLinearKernel", "dense", True),
    ("0.33.0", "FlashInferExperts", "moe", True),
}

sm120 = Table(
    _sm120_rows,
    aliases=("rtx6000", "rtx_pro_6000"),
    observed=_SM120_OBSERVED,
    capabilities=((12, 0),),
)


def sm89_binds(target: Target, fmt: Format, shape: tuple[int, int], _args: dict, _env: dict) -> str:
    """The dense kernel on Ada: the row's, refused for a shape Marlin cannot take."""
    return _wna16_binds(target, fmt, shape)


def sm89_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, _env: dict) -> str:
    """The expert kernel on Ada: the row's or ``moe_backend``'s, refused for a shape it cannot take."""
    return _moe_binds(target, fmt, shape, args)


def sm120_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, env: dict) -> str:
    """The dense kernel on an RTX PRO 6000: sm100's rule (NVFP4 inputs in whole groups of 16, else Marlin's)."""
    return sm100_binds(target, fmt, shape, args, env)


def sm120_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, env: dict) -> str:
    """The expert kernel on an RTX PRO 6000: sm100's rule, and for NVFP4 an intermediate in multiples of 64.

    sm100's rule names the row's kernel or ``moe_backend``'s. FlashInfer's CUTLASS experts pad the gate/up block
    scales to 128 rows and raise ``NotImplementedError`` rather than pad the gated weights to match
    (``prepare_nvfp4_moe_layer_for_fi_or_cutlass``), so ``2 * intermediate`` must be a multiple of 128.
    """
    from .step import Refused

    kernel = sm100_moe_binds(target, fmt, shape, args, env)
    if fmt.name == NVFP4_FORMAT and shape[1] % 64:
        raise Refused(
            f"{kernel} cannot serve {fmt.name}: intermediate {shape[1]} must be a multiple of 64 on an "
            "RTX PRO 6000; use w4a16"
        )
    return kernel


def sm100_binds(target: Target, fmt: Format, shape: tuple[int, int], _args: dict, _env: dict) -> str:
    """The dense kernel on a B200 or a GB300: the row's, refused for NVFP4 inputs that are not whole groups of 16."""
    from .step import Refused

    if fmt.name == NVFP4_FORMAT:
        if shape[1] % 16:
            raise Refused(
                f"{fmt.name} {list(shape)}: NVFP4 cuts inputs in groups of 16, and {shape[1]} is not whole"
            )
        return target.plan(fmt, "dense").kernel
    return _wna16_binds(target, fmt, shape)


def sm100_moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict, _env: dict) -> str:
    """The expert kernel on a B200 or a GB300: the row's, refused unless hidden and intermediate are whole groups of 16."""
    from .step import Refused

    if fmt.name == NVFP4_FORMAT and (shape[0] % 16 or shape[1] % 16):
        raise Refused(
            f"{target.plan(fmt, 'moe').kernel} cannot serve {fmt.name}: hidden {shape[0]} and "
            f"intermediate {shape[1]} must both be multiples of 16"
        )
    return _moe_binds(target, fmt, shape, args)


def _wna16_binds(target: Target, fmt: Format, shape: tuple[int, int]) -> str:
    """The new tables' dense rule: the row's kernel, refused for a shape Marlin cannot take."""
    from .step import Refused

    kernel = target.plan(fmt, "dense").kernel
    if kernel == "MarlinLinearKernel" and not _marlin_fits(fmt.group_size, shape[1]):
        raise Refused(
            f"{fmt.name} {list(shape)}: Marlin cannot take these inputs in groups of {fmt.group_size}"
        )
    return kernel


def humming_takes_experts(fmt: Format) -> bool:
    """Whether vLLM 0.33's Humming experts take a weight-only bank in `fmt` (else MarlinExperts serves it).

    ``CompressedTensorsWNA16MoEMethod`` keys an int4 or int8 bank's scales per token unless it is int4 g32, and
    Humming takes that key only for symmetric weights (``kInt4Static``, ``kInt8Static``). Observed on an H100:
    asymmetric g64 and g128 banks bind MarlinExperts, symmetric g128 HummingIndexedExperts.
    """
    return fmt.symmetric or fmt.weight_bits not in (4, 8) or (fmt.weight_bits == 4 and fmt.group_size == 32)


#: The expert kernel a ``moe_backend`` engine argument selects for integer weight-only experts (vLLM 0.33.0,
#: ``fused_moe/oracle/int_wna16.py``).
_MOE_BACKENDS = {"marlin": "MarlinExperts", "humming": "HummingIndexedExperts"}


def _moe_binds(target: Target, fmt: Format, shape: tuple[int, int], args: dict) -> str:
    """The new tables' expert rule: the row's kernel, refused for a shape it cannot take.

    For integer weight-only experts an explicit ``moe_backend`` (`_MOE_BACKENDS`) wins. Without one vLLM tries
    Marlin before Humming on every card but 9.0 (``humming/priority.py``), which is the row's kernel: Marlin for
    4 and 8 bits, Humming for 3.
    """
    from .step import Refused

    kernel = target.plan(fmt, "moe").kernel
    if fmt.weight_type == "int" and fmt.act == "bf16":
        kernel = _MOE_BACKENDS.get(args.get("moe_backend") or "", kernel)
    if why := shape_limit(kernel, fmt, shape):
        raise Refused(f"{kernel} cannot serve {fmt.name}: {why}")
    return kernel


_KERNEL = {"marlin": "MarlinLinearKernel", "humming": "HummingLinearKernel"}


def _dense_wna16(fmt: Format, acts: tuple[str, ...], ver: tuple[int, int], backend: str | None) -> KernelPlan:
    """Dense int4/int8 weight-only on sm90.

    vLLM's own order is Machete, (0.32+: Humming), Marlin, and Machete is off on every engine (`BANNED_KERNELS`,
    which `_sm90_rows` adds to every plan's ``serve.env``). The default pins nothing else, so vLLM binds Humming
    from 0.32 (its ``can_implement`` takes every bf16 weight-only module) and Marlin before. ``:marlin`` binds
    Marlin with ``linear_backend=marlin`` on 0.33 and by also disabling Humming on 0.32.
    """
    from .step import Refused

    zp = not fmt.symmetric
    if backend == "humming" and ver < (0, 32):
        raise Refused("backend 'humming' for dense int4 needs vLLM >= 0.32 (Marlin is ahead of it before)")
    notes = ["bf16 dequant (q - z) * s, fp32 accumulate"]
    if backend == "marlin" and ver >= (0, 33):  # observed: linear_backend=marlin binds Marlin on 0.33.0
        notes.append("Marlin chosen at serve time with linear_backend=marlin (vLLM's own pick is Humming)")
        return KernelPlan(_KERNEL[backend], zero_point=zp, act_order=acts, serve_args={"linear_backend": "marlin"},
                          notes=tuple(notes))  # fmt: skip
    env = {}
    if backend == "marlin" and ver >= (0, 32):
        env = {"VLLM_DISABLED_KERNELS": "HummingLinearKernel"}
        notes.append("Humming disabled at serve time (vLLM >= 0.32 puts it ahead of Marlin)")
    kernel = "humming" if backend == "humming" or (backend is None and ver >= (0, 32)) else "marlin"
    if kernel == "humming":
        notes.append(_HUMMING_SCOPE)
    return KernelPlan(_KERNEL[kernel], zero_point=zp, act_order=acts, serve_env=env, notes=tuple(notes))


#: scope of the Humming check
_HUMMING_SCOPE = (
    "Humming 'safe' means the tested sm90 tile tables and shapes with random weights and inputs, "
    "not a universal kernel certification"
)


def merge_serve(plans: dict[str, KernelPlan], vllm: str, extra: dict[str, str] | None = None) -> dict:
    """The checkpoint's serve config (``build.json["serve"]``) for the plans in use, keyed by layer kind.

    ``VLLM_DISABLED_KERNELS`` lists are merged; any other env or arg must agree between plans. ``kernels`` maps a
    kind to the sorted list of every kernel class vLLM binds for it: each plan's, plus `extra` (kind -> one more
    class), for a module stored in another format whose settings are not merged, as `Target.binds` resolves it.
    """
    env: dict[str, str] = {}
    args: dict[str, str] = {}
    disabled: list[str] = []
    plugins: list[str] = []
    for p in plans.values():
        for k, v in p.serve_env.items():
            if k == "VLLM_DISABLED_KERNELS":
                disabled += [x for x in v.split(",") if x and x not in disabled]
            elif env.setdefault(k, v) != v:
                raise ValueError(f"serve env {k}: plans disagree ({env[k]!r} vs {v!r})")
        for k, v in p.serve_args.items():
            if args.setdefault(k, v) != v:
                raise ValueError(f"serve arg {k}: plans disagree ({args[k]!r} vs {v!r})")
        plugins += [x for x in p.serve_plugins if x not in plugins]
    if disabled:
        env["VLLM_DISABLED_KERNELS"] = ",".join(disabled)
    kernels = {k: {p.kernel} for k, p in plans.items()}
    for k, cls in (extra or {}).items():
        kernels.setdefault(k, set()).add(cls)
    return {"vllm": vllm, "env": env, "args": args, "plugins": plugins,
            "kernels": {k: sorted(v) for k, v in kernels.items()}}  # fmt: skip
