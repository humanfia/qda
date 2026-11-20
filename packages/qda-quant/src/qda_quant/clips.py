"""Group-scale choices ("clip searches"): how a group's range becomes its (scale, zero).

A clip is a stateless plugin chosen by name: the value of ``clip:`` in a step's settings (``rtn``, ``gptq``,
``gptaq``) is the key of its entry in ``qda.clips``. The registry holds built-ins and package registrations,
so a new search arrives without an ``if`` in a step. `qda plan` asks a clip's `refusal` before any
weight is loaded, and a clip declares in `needs_hessian` whether the step must request the Hessian statistics.

Built-ins:

* ``minmax``: compressed-tensors' min/max qparams; no shrink, no weighting.
* ``mse``: llm-compressor's ``mse`` observer: per group the shrink ``p`` in ``1.00 .. 0.81`` of the range
  minimizing ``sum |w - fq_p(w)|^2.4``.
* ``imse``: llm-compressor's ``imatrix_mse``: shrink ``p`` in ``1.00 .. 0.05`` minimizing
  ``sum imp_j |w_j - fq_p(w_j)|^3`` with ``imp = diag(H) / mean(diag(H))``.
* ``hmse``: ratio ``a`` in ``0.70 .. 1.05`` (36 points, so ranges may also grow) minimizing
  ``sum_j H_jj (w_j - fq_a(w_j))^2``. With FP8 scales (W4A8) every candidate is priced on the kernel's FP8
  scale grid, which makes this the "FP8 scale grid" search of the paper. WNA16 uses
  fp32 candidate weights, before storing scales in bf16.
* ``grid3``: a W4A8 search that checks each group's starting e4m3 scale and ``grid_k`` neighbours
  on either side (default 1).

* ``fp4mse`` and ``fp4hmse``: NVFP4 searches over the E4M3 scales a checkpoint can hold: each group's fitted
  scale and its neighbours, 2 values down to 12 up, by squared error, plain or weighted by ``diag(H)``. A group's
  scale may grow: E2M1's values are sparse near its largest, so the fitted scale is often not the best one.

``imatrix`` (`qda_quant.gguf.q4k`): llama.cpp's importance-weighted fit, on every solved GGUF grid
(``ImatrixFit.only_grid``: ``q4k``, ``q5k``, ``q6k``, ``iq4xs``, ``iq4nl``, ``q3k``).

W4A8 errors are priced on the served grid (`Quantizer.fake`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import AfterValidator

from .registry import Registry

if TYPE_CHECKING:
    import torch

    from .format import Format
    from .solve.quant import Quantizer
    from .target import KernelPlan


class Clip:
    """One group-scale search. Subclass it and register an instance in ``qda.clips``.

    Attributes:
        name: registry name, the value of ``clip:`` in a step's settings.
        needs_hessian: True when `choose` weights errors by the Hessian diagonal; the step then requests the
            Hessian statistics.
        picks_ranges: False when `choose` returns the group's own min/max qparams, so there is no chosen range.
        needs_solve: True when `choose` only supplies the starting scales of a second-order solve; a step that
            then just rounds (`rtn`) must refuse it.
        only_grid: the grid, or grids, `choose` works on, or None for any; `QuantStep.check_grid` refuses
            the others. A clip that calls a grid method not every grid has names the ones that have it.
    """

    name: ClassVar[str] = ""
    needs_hessian: ClassVar[bool] = False
    picks_ranges: ClassVar[bool] = True
    needs_solve: ClassVar[bool] = False
    only_grid: ClassVar[str | tuple[str, ...] | None] = None

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Why this clip cannot be served as `fmt` on `plan`, or None (`qda plan` asks before calibration)."""
        return None

    def choose(
        self,
        q: Quantizer,
        w: torch.Tensor,
        hdiag: torch.Tensor | None = None,
        grid_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return group qparams; ``hdiag [E, C]`` is required by the Hessian-weighted searches."""
        import torch

        with torch.no_grad():
            return self._choose(q, w.float(), hdiag, grid_k)

    def _choose(
        self,
        q: Quantizer,
        w: torch.Tensor,
        hdiag: torch.Tensor | None,
        grid_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


class MinMax(Clip):
    """Compressed-tensors' min/max rule: the group's own range, never a shrink."""

    name = "minmax"
    picks_ranges = False

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        return q.params(w)


class MSEObserver(Clip):
    """llm-compressor's ``mse`` observer: shrink the range to minimize ``sum |w - fq_p(w)|^2.4``."""

    name = "mse"
    SHRINKS: ClassVar[list[float]] = [1 - i / 100 for i in range(20)]
    POWER: ClassVar[float] = 2.4

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        return _range_search(q, w, self.SHRINKS, self.POWER, None)


class IMatrixMSE(Clip):
    """llm-compressor's ``imatrix_mse``: ``sum imp_j |w_j - fq_p(w_j)|^3``, ``imp`` from the Hessian diagonal."""

    name = "imse"
    needs_hessian = True
    SHRINKS: ClassVar[list[float]] = [1 - i / 20 for i in range(20)]
    POWER: ClassVar[float] = 3.0

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        if hdiag is None:
            raise ValueError(f"clip {self.name!r} weights errors by the Hessian diagonal; none was given")
        return _range_search(q, w, self.SHRINKS, self.POWER, hdiag)


class HessianMSE(Clip):
    """Hessian-weighted squared error over a range ratio that may shrink or grow; FP8-grid aware."""

    name = "hmse"
    needs_hessian = True

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        return _hmse(q, w, hdiag)


class Grid3(Clip):
    """A W4A8 search over the served e4m3 group-scale grid; it starts GPTQ's solve with the best neighbour."""

    name = "grid3"
    needs_hessian = True
    needs_solve = True

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:
        """Explain when the neighbouring-scale search does not match the format's served scale grid."""
        if not (
            fmt.weight_bits == 4
            and fmt.group_size == 128
            and fmt.symmetric
            and fmt.act == "fp8_token"
            and plan.scale_dtype == "fp8_e4m3"
        ):
            return "grid3 requires symmetric W4A8 g128 and a target with FP8 group scales"
        return None

    def _choose(self, q, w, hdiag, grid_k):
        return _grid3(self, q, w, hdiag, grid_k)


class Fp4MSE(Clip):
    """NVFP4: per group, the E4M3 scale among the fitted one's neighbours with the lowest squared error."""

    name = "fp4mse"
    picks_ranges = False  # it steps along the stored scale values; it shrinks no range
    only_grid = "nvfp4"
    #: E4M3 values tried below and above the fitted scale; the fitted scale wins a tie, then the lower ones.
    DOWN: ClassVar[int] = 2
    UP: ClassVar[int] = 12

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:
        """Only where group scales are NVFP4's."""
        if plan.scale_dtype != "nvfp4":
            return f"{self.name} steps along NVFP4's E4M3 group scales; {fmt.name} here has {plan.scale_dtype} scales"
        return None

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        return _fp4_search(q, w, None, self.DOWN, self.UP)


class Fp4HessianMSE(Fp4MSE):
    """`Fp4MSE` with each column's error weighted by the Hessian diagonal."""

    name = "fp4hmse"
    needs_hessian = True

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        if hdiag is None:
            raise ValueError(f"clip {self.name!r} weights errors by the Hessian diagonal; none was given")
        return _fp4_search(q, w, hdiag, self.DOWN, self.UP)


def _adopt(key: str, clip: Clip) -> None:
    """Refuse a clip plugin of the wrong type, from either source."""
    if not isinstance(clip, Clip):
        raise TypeError(f"clip plugin {key!r} is {clip!r}, not a Clip instance")


#: Clip searches by config name: built-ins and package registrations (see `qda_quant.registry`).
CLIPS: Registry[Clip] = Registry(
    "clips",
    {
        "fp4hmse": "qda_quant.clips:FP4HMSE",
        "fp4mse": "qda_quant.clips:FP4MSE",
        "grid3": "qda_quant.clips:GRID3",
        "hmse": "qda_quant.clips:HMSE",
        "imatrix": "qda_quant.gguf.q4k:IMATRIX",
        "imse": "qda_quant.clips:IMSE",
        "minmax": "qda_quant.clips:MINMAX",
        "mse": "qda_quant.clips:MSE",
    },
    on_add=_adopt,
)
MINMAX = MinMax()
MSE = MSEObserver()
IMSE = IMatrixMSE()
HMSE = HessianMSE()
GRID3 = Grid3()
FP4MSE = Fp4MSE()
FP4HMSE = Fp4HessianMSE()


def known_clip(name: str) -> str:
    """`name` if it is a registered clip (a config validator's check).

    Raises:
        Refused: an unknown name; the message lists the known ones. (Pydantic wraps it in ValidationError
            when a config validator calls this.)
    """
    if name not in CLIPS:
        from .step import Refused  # here, not at the top: step.py imports this module's validator

        raise Refused(f"unknown clip {name!r}; known: {sorted(CLIPS)}")
    return name


#: A step's ``clip`` setting: a name in ``qda.clips``, checked when the config is read.
ClipName = Annotated[str, AfterValidator(known_clip)]


def _range_search(
    q: Quantizer,
    w: torch.Tensor,
    shrinks: list[float],
    power: float,
    hdiag: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pick, per group, the shrink whose weighted rounding error is lowest."""
    import torch

    from .solve.quant import groups

    g = q.group(w.shape[-1])
    imp = None
    if hdiag is not None:
        imp = hdiag.float().clamp(min=0) / (hdiag.float().mean(-1, keepdim=True) + 1e-30)
        imp = groups(imp, g)[:, None]  # [E, 1, G, g]
    best_s: torch.Tensor | None = None
    best_z: torch.Tensor | None = None
    best_e: torch.Tensor | None = None
    for p in shrinks:
        s, z = q.params(w, p)
        err = groups((q.fake(w, s, z) - w).abs().pow(power), g)
        e = (err * imp if imp is not None else err).sum(-1)
        if best_e is None:
            best_s, best_z, best_e = s, z, e
            continue
        better = e < best_e
        assert best_s is not None
        best_s = torch.where(better, s, best_s)
        if z is not None:
            assert best_z is not None
            best_z = torch.where(better, z, best_z)
        else:
            best_z = None
        best_e = torch.minimum(e, best_e)
    assert best_s is not None
    return best_s, best_z


def _fp4_search(
    q: Quantizer, w: torch.Tensor, hdiag: torch.Tensor | None, down: int, up: int
) -> tuple[torch.Tensor, None]:
    """Per group, the scale `down` .. `up` E4M3 values from the fitted one with the lowest (weighted) squared error.

    An E4M3 value is ``(8 + m) / 16 * 2^e`` with ``m`` in 0 .. 7, so its neighbours are one step of ``8 e + m``.
    The steps are walked on the scale itself, not on a stored code: the unit's global scale follows the result
    (`Quantizer.usable`), so the largest group may grow as any other.
    """
    import torch

    from .solve.quant import groups

    g = q.group(w.shape[-1])
    fitted, _ = q.params(w)
    mantissa, exponent = torch.frexp(fitted)
    index = exponent * 8 + (mantissa * 16 - 8).round().to(exponent.dtype)
    weight = None if hdiag is None else groups(hdiag.float().clamp(min=0), g)[:, None]
    best, best_error = fitted, None
    for offset in (0, *range(-down, 0), *range(1, up + 1)):
        moved = index + offset
        scale = torch.ldexp(
            (8 + torch.remainder(moved, 8)).float() / 16, torch.div(moved, 8, rounding_mode="floor")
        )
        error = groups((q.dequant(q.quantize(w, scale, None), scale, None) - w).square(), g)
        error = (error if weight is None else error * weight).sum(-1)
        if best_error is None:
            best_error = error
            continue
        better = error < best_error
        best, best_error = torch.where(better, scale, best), torch.minimum(error, best_error)
    return q.usable(best), None


def _grid3(
    clip: Grid3,
    q: Quantizer,
    weights: torch.Tensor,
    hdiag: torch.Tensor | None,
    grid_k: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Search neighbouring e4m3 scales, keeping zero and tiny groups valid on the served grid.

    The serving conversion floors its channel scale. Stored scales must survive that floor and
    the subsequent division by eight: the smallest positive e4m3 factor is 1/512, so the minimum
    normalized stored scale is 8/512 = 1/64.
    """
    import torch

    from .solve.quant import FP8_MAX, FP8_MIN_SCALE, groups

    refusal = clip.refusal(q.fmt, q.plan)
    if refusal:
        raise ValueError(refusal)
    if hdiag is None:
        raise ValueError("clip 'grid3' weights errors by the Hessian diagonal; none was given")
    if grid_k < 0:
        raise ValueError("grid_k must be nonnegative")

    group = q.group(weights.shape[-1])
    experts, _, columns = weights.shape
    groups_per_row = columns // group
    grouped = groups(weights, group)
    positive = grouped.clamp(min=0).amax(-1)
    negative = (-grouped).clamp(min=0).amax(-1)
    start = torch.maximum(positive / q.fmt.qmax, negative / -q.fmt.qmin).clamp(min=1e-10)
    channel = (start.amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    normalized = (start / channel).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
    spacing = torch.exp2(torch.floor(torch.log2(normalized.clamp(min=2**-6))) - 3)
    importance = hdiag.float().clamp(min=0).reshape(experts, 1, groups_per_row, group)
    best: torch.Tensor | None = None
    best_error: torch.Tensor | None = None
    at_start: torch.Tensor | None = None
    for offset in range(-grid_k, grid_k + 1):
        candidate = (normalized + offset * spacing).clamp(min=2**-9, max=FP8_MAX)
        scale = (candidate / 8).to(torch.float8_e4m3fn).float() * (channel * 8)
        codes = torch.round(grouped / scale[..., None]).clamp(q.fmt.qmin, q.fmt.qmax)
        error = ((grouped - codes * scale[..., None]).square() * importance).sum(-1)
        if offset == 0:
            at_start = scale
        if best_error is None:
            best, best_error = scale, error
        else:
            better = error < best_error
            assert best is not None
            best = torch.where(better, scale, best)
            best_error = torch.where(better, error, best_error)

    assert at_start is not None
    assert best is not None
    largest_group = start >= start.amax(-1, keepdim=True)
    best = torch.where(largest_group, at_start, best)
    channel_floor = (best.amax(-1, keepdim=True) / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    scale_floor = (channel_floor / 64).clamp(min=FP8_MIN_SCALE)
    return _fp8_scale_grid(best.clamp(min=scale_floor)).to(torch.bfloat16), None


def _fp8_scale_grid(scale: torch.Tensor) -> torch.Tensor:
    """Snap fp32 W4A8 scales to the CUTLASS e4m3 group grid before bf16 storage."""
    import torch

    from .solve.quant import FP8_MAX

    rows = scale.float().reshape(-1, scale.shape[-1])
    channel = (rows.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    group_scale = (rows / channel).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    group_scale = (group_scale / 8).to(torch.float8_e4m3fn).float()
    return (group_scale * (channel * 8)).reshape(scale.shape)


def _hmse(
    q: Quantizer, weights: torch.Tensor, hdiag: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Choose a Hessian-weighted scale ratio, evaluating W4A8 candidates on the served FP8 grid."""
    import torch

    from .solve.quant import groups

    if hdiag is None:
        raise ValueError("clip 'hmse' weights errors by the Hessian diagonal; none was given")
    group = q.group(weights.shape[-1])
    grouped = groups(weights, group)
    importance = groups(hdiag.float().clamp(min=0), group)[:, None]
    if q.fmt.symmetric:
        initial = q.grid.scale(q.fmt, grouped.amin(-1).clamp(max=0), grouped.amax(-1).clamp(min=1e-10))
    best_scale: torch.Tensor | None = None
    best_zero: torch.Tensor | None = None
    best_error: torch.Tensor | None = None
    for ratio in torch.linspace(0.70, 1.05, 36, device=weights.device).tolist():
        scale, zero = (q.usable(initial * ratio), None) if q.fmt.symmetric else q.params(weights, ratio)
        if q.fp8_scales:
            error = groups(q.fake(weights, scale, zero) - weights, group)
        else:
            scaled = grouped / scale[..., None]
            offset = 0 if zero is None else zero[..., None]
            codes = (scaled + offset).round().clamp(q.fmt.qmin, q.fmt.qmax)
            error = (codes - offset) * scale[..., None] - grouped
        loss = (error.square() * importance).sum(-1)
        if best_error is None:
            best_scale, best_zero, best_error = scale, zero, loss
            continue
        better = loss < best_error
        assert best_scale is not None
        best_scale = torch.where(better, scale, best_scale)
        if zero is not None:
            assert best_zero is not None
            best_zero = torch.where(better, zero, best_zero)
        else:
            best_zero = None
        best_error = torch.minimum(loss, best_error)
    assert best_scale is not None
    return best_scale, best_zero
