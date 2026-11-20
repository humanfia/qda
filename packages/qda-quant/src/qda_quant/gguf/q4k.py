"""The ``q4k`` grid: llama.cpp's Q4_K, so a build's codes are the codes ``llama-quantize`` writes.

Q4_K stores a row in super-blocks of 256 values, 8 groups of 32. A group's value is ``d * sc * q - dmin * m`` with
codes ``q`` in ``0 .. 15``, 6-bit ``sc`` and ``m``, and fp16 ``d`` and ``dmin`` per super-block. The scales are
fitted on the values (``make_qkx2_quants``), not derived from a range, so this grid has no range rule: clip searches
and learned ranges do not apply. GPTQ uses an exact fused CUDA block by default, with its torch reference
selected by ``fused_grid: false``. The ``imatrix`` clip (`ImatrixFit`) is the one
choice a step has: ggml's fit weighted by the Hessian diagonal, as ``llama-quantize --imatrix`` fits.

In the grid's terms the format is ``w4a16-asym-g32``, the scale of a group is ``fp16(d) * sc`` and its zero is the
offset ``fp16(dmin) * m`` (a float, not an integer zero point). Both are exact in fp32, which is how they are stored,
so the exporter recovers ``d``, ``sc``, ``dmin`` and ``m`` from them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..clips import Clip
from ..solve.grid import Grid
from ..step import Refused
from .kernel import QK_K, q4k_fields, quantize_q4k, quantize_q4k_weighted

if TYPE_CHECKING:
    from ..format import Format
    from ..target import KernelPlan

#: The format Q4_K's codes and groups are, in qda's terms.
Q4K_FORMAT = "w4a16-asym-g32"


def _f16(bits) -> torch.Tensor:  # noqa: ANN001
    """fp16 bits (numpy uint16) -> fp32 tensor."""
    return torch.from_numpy(bits.view("<f2").astype("<f4"))


class Q4K(Grid):
    """llama.cpp's Q4_K: ``quantize_row_q4_K_ref`` (`fit`), or ``quantize_q4_K`` with importances (`fit_weighted`)."""

    name = "q4k"
    gptq_fused = True
    range_rule = False
    stores = Q4K_FORMAT
    fields = staticmethod(q4k_fields)
    records = staticmethod(lambda x: quantize_q4k(x))
    weighted_records = staticmethod(lambda x, importance: quantize_q4k_weighted(x, importance))

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Only the format Q4_K is."""
        if fmt.name != Q4K_FORMAT:
            return f"Q4_K stores 4-bit asymmetric codes in groups of 32: format {Q4K_FORMAT}, not {fmt.name}"
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        """Ggml's fit of ``w [..., R, C]``: (``fp16(d) * sc``, ``fp16(dmin) * m``), both ``[..., R, C / 32]``.

        A row that is not a whole number of super-blocks is fitted zero-padded to one, keeping its own groups. Q4_K
        cannot store such a row (llama.cpp writes it as Q5_0 or Q8_0 from the original weights), so these values
        only stand in for it in the build's calibration stream; no padded block reaches a file.

        Raises:
            Refused: `C` is not a multiple of 32.
        """
        scale, zero, _ = self._fit(w)
        return scale, zero

    def fit_weighted(self, w: torch.Tensor, importance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ggml's fit of ``w [E, R, C]`` weighted by ``importance [E, C]``, in `fit`'s terms; padding weighs nothing.

        Raises:
            Refused: `C` is not a multiple of 32.
        """
        scale, zero, _ = self._fit(w, importance=importance)
        return scale, zero

    def fit_codes(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: ARG002
        """`fit` and the codes ggml made on the way: coding the weights again would be a second pass for nothing."""
        scale, zero, codes = self._fit(w, codes=True)
        assert codes is not None  # codes=True
        return scale, zero, codes

    def _fit(
        self, w: torch.Tensor, codes: bool = False, importance: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(scale, zero, codes) of ``w``; the kernel produces the codes whatever we keep."""
        c = w.shape[-1]
        if c % 32:
            raise Refused(f"grid {self.name!r}: a row of {c} values is not a whole number of 32-value groups")
        rows = w.reshape(-1, c).float()
        pad = -c % QK_K
        if pad:
            rows = torch.nn.functional.pad(rows, (0, pad))
        if importance is None:
            records = self.records(rows)
        else:
            imp = torch.nn.functional.pad(importance, (0, pad))
            n = rows.shape[0] // imp.shape[0]
            records = b"".join(
                self.weighted_records(rows[i * n : (i + 1) * n], imp[i]) for i in range(imp.shape[0])
            )
        d, dmin, sc, m, made = self.fields(records)
        g = rows.shape[-1] // 32
        scale = (_f16(d)[:, None] * torch.from_numpy(sc).float()).reshape(-1, g)[:, : c // 32]
        zero = (_f16(dmin)[:, None] * torch.from_numpy(m).float()).reshape(-1, g)[:, : c // 32]
        shape = (*w.shape[:-1], c // 32)
        scale, zero = scale.reshape(shape).to(w.device), zero.reshape(shape).to(w.device)
        if not codes:
            return scale, zero, None
        q = torch.from_numpy(made.reshape(rows.shape)[:, :c]).to(torch.int8).reshape(w.shape)
        return scale, zero, q.to(w.device)

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """As fitted."""
        return s

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Fp32: ``fp16 * 6-bit`` is exact there."""
        return s.float()

    def stored_zero(self, z: torch.Tensor) -> torch.Tensor:
        """Fp32: the min offset ``fp16(dmin) * m`` is not an integer."""
        return z.float()

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """llama.cpp multiplies with the fp32 product itself."""
        return s.float()

    def codes(
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """``nearest_int((w + m) / d)`` clamped; a group with ``d == 0`` gets code 0."""
        assert z is not None
        q = torch.clamp(torch.round((w + z) / s), qmin, qmax)
        return torch.where(s != 0, q, torch.zeros_like(q))

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """``d * q - m`` in fp32, as ``dequantize_row_q4_K``."""
        assert z is not None
        return s * q - z


class ImatrixFit(Clip):
    """Ggml's fit weighted by the Hessian diagonal: what ``llama-quantize --imatrix`` does with an importance matrix.

    ``llama-imatrix`` measures each column's mean squared input, which is the diagonal of the Hessian per token.
    Q4_K fits through ``quantize_q4_K``; Q5_K, Q6_K, IQ4 and Q3_K through ``ggml_quantize_chunk``.
    A build may weigh tensors in all six grids, including a custom mix's promoted tensors.
    """

    name = "imatrix"
    needs_hessian = True
    picks_ranges = False
    only_grid = ("q4k", "q5k", "q6k", "iq4xs", "iq4nl", "q3k")

    def _choose(self, q, w, hdiag, grid_k):  # noqa: ARG002
        if hdiag is None:
            raise ValueError(f"clip {self.name!r} weights its fit by the Hessian diagonal; none was given")
        return q.grid.fit_weighted(w, hdiag)


Q4K_GRID = Q4K()
IMATRIX = ImatrixFit()
