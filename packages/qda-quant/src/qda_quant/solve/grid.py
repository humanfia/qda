"""Quantization grids: how a group's range becomes (scale, zero), and how that scale is stored and served.

A `Grid` is a stateless plugin chosen by name: ``grid:`` in a config is the recipe's default, and ``grid:`` in a
step's settings overrides it for that step. `Quantizer.of` is the one place a name becomes a grid.

The invariant a grid must keep is that a checkpoint holds exactly what the kernel computes. vLLM loads
``weight_scale`` into a parameter of the served dtype, so a kernel with ``plan.scale_dtype == "bf16"`` multiplies
with ``bf16(stored)`` whatever dtype the file holds (`Grid.served`). `Grid.refusal` says which plans a grid
cannot serve; `compile_plan` asks it before any weight is loaded.

Built-ins:

* ``absmax`` (default): compressed-tensors' min/max rule, ``absmax / 7.5`` at 4 bits, positive scales stored in
  bf16. Every format and kernel.
* ``signed8``: auto-round's rule, ``absmax / 8`` at 4 bits, stored in fp16. The sign of the scale puts the larger
  side of the range on code -8. Symmetric formats on bf16-scale kernels.
* ``q4k``: llama.cpp's Q4_K (`qda_quant.gguf.q4k`), fitted on the values; format ``w4a16-asym-g32``.
* ``q5k``: llama.cpp's native Q5_K (`qda_quant.gguf.q5k`); format ``w5a16-asym-g32``.
* ``q6k``: llama.cpp's Q6_K (`qda_quant.gguf.q6k`), likewise; format ``w6a16-sym-g16``.
* ``nvfp4``: NVFP4's E4M3 group scales under a global scale (`qda_quant.solve.nvfp4`); format ``w4a4-nvfp4``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ..registry import Registry

if TYPE_CHECKING:
    import torch

    from ..format import Format
    from ..target import KernelPlan

#: The grid of a recipe that names none: the validated path, and the only one `Packed` needs to read scales back.
DEFAULT_GRID = "absmax"


class Grid:
    """One rule for group scales. Subclass it and register an instance in ``qda.grids``.

    Attributes:
        name: registry name, the value of ``grid:`` in a config.
    """

    name: ClassVar[str] = ""
    #: True when scales come from group ranges (`scale`), codes are ``round(w / s + z)`` and values
    #: ``(q - z) * s``: the contract clip searches, learned ranges and the Triton GPTQ solve rely on. A grid that
    #: fits its scales on the values and owns its formulas (`fit`, `codes`, `value`) sets it False.
    range_rule: ClassVar[bool] = True
    #: The one format this grid writes, when it writes exactly one: a file format's grid stores a layout and
    #: nothing else (`q4k` writes Q4_K records), where a general grid serves whatever the kernel plan allows.
    #: A step quantizes a unit the build stores in another format on the grid that declares it (`grid_for`).
    stores: ClassVar[str | None] = None
    #: The fused GPTQ block implements round((w + z) / s), zero codes at s == 0, and fp32 q*s - z
    #: (z absent for symmetric grids). A grid declares this only when those are its exact codes/value formulas.
    gptq_fused: ClassVar[bool] = False

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Why this grid cannot be served as `fmt` on `plan`, or None."""
        if plan.scale_dtype != "bf16":
            return f"the kernel casts group scales to {plan.scale_dtype}, which this grid does not model"
        return None

    def divides(self, fmt: Format, cols: int) -> bool:
        """Whether `cols` input columns are a whole number of `fmt.group_size` groups on this grid.

        A grid whose build pads a partial group before quantizing (a fixed super-block, say) overrides this to
        return True; `qda plan` asks the step's grid before refusing a width.
        """
        return fmt.group_size == -1 or cols % fmt.group_size == 0

    def scale(self, fmt: Format, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """The raw fp32 scale of group ranges ``lo <= 0 <= hi``; it scales linearly with a clip ratio."""
        raise NotImplementedError

    def fit_weighted(
        self, w: torch.Tensor, importance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fit `w` weighted by one importance per column, as `fit` returns it; not every grid has one.

        A clip that weighs its fit names the grids that do (`Clip.only_grid`), and `QuantStep.check_grid`
        refuses the others at plan time, so this is reached only by a grid that overrides it.

        Raises:
            NotImplementedError: this grid fits on ranges or values alone.
        """
        raise NotImplementedError(f"grid {self.name!r} has no importance-weighted fit")

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """A raw scale as the grid keeps it (fp32): never zero, and rounded if the rule rounds before coding."""
        raise NotImplementedError

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """The scale in the dtype the checkpoint holds."""
        raise NotImplementedError

    def zero(self, fmt: Format, lo: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Zero points of an asymmetric format, as compressed-tensors computes them."""
        import torch

        return torch.round(torch.clamp(fmt.qmin - lo / s, fmt.qmin, fmt.qmax))

    def stored_zero(self, z: torch.Tensor) -> torch.Tensor:
        """Zero points as the checkpoint holds them: integers, in int8."""
        import torch

        return z.to(torch.int8)

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """The scale a bf16-scale kernel multiplies with (fp32): the stored value, rounded to bf16 at load."""
        import torch

        return self.stored(s).to(torch.bfloat16).float()

    def global_scale(self, s: torch.Tensor) -> torch.Tensor | None:  # noqa: ARG002
        """A second scale level the stored scales ``[..., R, G]`` of one unit share, ``[..., 1, 1]``; default none.

        A layout with one (NVFP4's global scale) derives it from the whole unit, before the unit's rows are
        recorded per module, so every module of a fused group stores the same value.
        """
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(scale, zero) ``[..., R, G]`` fitted on the weights ``w [..., R, C]``; only when `range_rule` is False."""
        raise NotImplementedError

    def fit_codes(
        self,
        fmt: Format,  # noqa: ARG002
        w: torch.Tensor,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None:
        """(scale, zero, codes) ``[..., R, C]`` when this grid's fit makes the codes anyway.

        None (the default) when coding the weights takes a second pass, as it does for every grid whose scales come
        from a range: the caller then uses `fit` and `codes`.
        """
        return None

    def codes(
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """Integer codes (fp32) of `w` against per-column served (s, z); only when `range_rule` is False."""
        raise NotImplementedError

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """The weight (fp32) codes `q` decode to under per-column served (s, z); only when `range_rule` is False."""
        raise NotImplementedError


class AbsMax(Grid):
    """Compressed-tensors' ``calculate_qparams``: positive scales, stored in bf16.

    The range math is done in fp32 and rounded once when stored. Known difference: compressed-tensors does it in the
    weight's dtype (``(max - min) / 15`` in bf16, two roundings), so asymmetric group scales can differ from
    llm-compressor's; symmetric scales use the same range calculation.
    """

    name = "absmax"

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Nothing for bf16 and FP8 scales (`Quantizer` models the FP8 scale kernels on this grid)."""
        if plan.scale_dtype == "nvfp4":
            return "the kernel reads E4M3 group scales under a global scale, which the nvfp4 grid models"
        return None

    def scale(self, fmt: Format, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """``absmax / ((qmax - qmin) / 2)`` when symmetric, else ``(hi - lo) / (qmax - qmin)``."""
        import torch

        if fmt.symmetric:
            return torch.maximum(-lo, hi) / ((fmt.qmax - fmt.qmin) / 2)
        return (hi - lo) / (fmt.qmax - fmt.qmin)

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """An all-zero group gets the smallest fp32 step instead of 0."""
        import torch

        return torch.where(s == 0, torch.finfo(torch.float32).eps, s)

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Bf16."""
        import torch

        return s.to(torch.bfloat16)


class Signed8(Grid):
    """Auto-round's ``quant_tensor_rtn_sym``: signed scales ``+-absmax / 2^(b-1)``, stored in fp16.

    The scale is negative when ``hi >= -lo``, so the larger side of the range reaches code ``qmin`` (-8 at 4 bits),
    the side with one more level. On vLLM the kernel still multiplies with ``bf16(fp16(s))``: codes are chosen for
    that value, not for the fp16 scale auto-round itself rounds against.
    """

    name = "signed8"
    #: auto-round's ``q_scale_thresh``: the smallest scale magnitude
    MIN_MAGNITUDE: ClassVar[float] = 1e-5

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:
        """Symmetric codes only, on a kernel that multiplies with bf16 scales."""
        if not fmt.symmetric:
            return "its signed range rule is defined for symmetric codes only; asymmetric codes have a zero point"
        return super().refusal(fmt, plan)

    def scale(self, fmt: Format, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """``sign * absmax / -qmin``; the sign is +1 when the negative side is strictly larger, else -1."""
        import torch

        return (2 * (hi < -lo).to(hi.dtype) - 1) * torch.maximum(-lo, hi) / -fmt.qmin

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """Rounded to fp16, then floored in magnitude; an all-zero group gets the positive floor."""
        import torch

        h = self.stored(s)
        lim = self.MIN_MAGNITUDE
        return torch.where(h < 0, h.clamp(max=-lim), h.clamp(min=lim)).float()

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Fp16.

        Raises:
            Refused: a scale fp16 cannot hold; it would become infinite and its weights NaN.
        """
        import torch

        from ..step import Refused

        h = s.to(torch.float16)
        if torch.isinf(h).any():
            raise Refused(f"grid {self.name!r}: a group scale of {s.abs().max().item():g} overflows fp16")
        return h


def _adopt(key: str, grid: Grid) -> None:
    """Refuse a grid plugin of the wrong type, from either source."""
    if not isinstance(grid, Grid):
        raise TypeError(f"grid plugin {key!r} is {grid!r}, not a Grid instance")


#: Grids by config name: built-ins and package registrations (see `qda_quant.registry`).
GRIDS: Registry[Grid] = Registry(
    "grids",
    {
        "absmax": "qda_quant.solve.grid:ABSMAX",
        "signed8": "qda_quant.solve.grid:SIGNED8",
        "q4k": "qda_quant.gguf.q4k:Q4K_GRID",
        "q5k": "qda_quant.gguf.q5k:Q5K_GRID",
        "q6k": "qda_quant.gguf.q6k:Q6K_GRID",
        "iq4nl": "qda_quant.gguf.iq4:IQ4NL_GRID",
        "iq4xs": "qda_quant.gguf.iq4:IQ4XS_GRID",
        "q3k": "qda_quant.gguf.q3k:Q3K_GRID",
        "nvfp4": "qda_quant.solve.nvfp4:NVFP4_GRID",
    },
    on_add=_adopt,
)
ABSMAX = AbsMax()
SIGNED8 = Signed8()


def known_grid(name: str | None) -> str | None:
    """`name` if it is None or a registered grid (a config validator's check).

    Raises:
        Refused: an unknown name; the message lists the known ones. (Pydantic wraps it in ValidationError
            when a config validator calls this.)
    """
    if name is not None and name not in GRIDS:
        from ..step import Refused  # here, not at the top: step.py imports this module

        raise Refused(f"unknown grid {name!r}; known: {sorted(GRIDS)}")
    return name
