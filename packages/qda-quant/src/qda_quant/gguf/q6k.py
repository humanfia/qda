"""The ``q6k`` grid: llama.cpp's Q6_K, so a build's codes are the codes ``llama-quantize`` writes.

Q6_K stores a row in super-blocks of 256 values, 16 groups of 16. A group's value is ``d * s * q`` with
signed 6-bit codes ``q`` in ``-32 .. 31``, a signed 8-bit ``s`` per group and one fp16 ``d`` per super-block.
There is no minimum: unlike Q4_K this grid is symmetric.

The scales come from ggml's own fit (``make_qx_quants``), not from a range rule, so like `q4k` this grid
sets ``range_rule = False`` and a method solves the codes against scales it does not move. What a block can
express is narrow: the sixteen int8 factors are fixed by the *ratios* between its group scales, and ``d``
only sets their common size. A change that keeps those ratios -- rescaling a whole block -- stays
representable whenever the new ``d`` is still an exact fp16; a change that alters them does not. Measured
over 134,000 super-blocks of a real checkpoint: ggml's own scales factor exactly in every block, a uniform
rescale of a block by 2 or 3 in all of them, and a 0.1 % per-group perturbation in none. So a grid that
fitted these scales group by group would be writing something Q6_K cannot hold.

In the grid's terms the format is ``w6a16-sym-g16`` and the scale of a group is ``fp16(d) * s``, exact in fp32,
which is how it is stored; the exporter recovers ``d`` and ``s`` from the product.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..solve.grid import Grid
from ..step import Refused
from .file import row_bytes, type_id
from .kernel import QK_K, q6k_fields, quantize_q6k, quantize_weighted_chunk

if TYPE_CHECKING:
    from ..format import Format
    from ..target import KernelPlan

#: The format Q6_K's codes and groups are, in qda's terms.
Q6K_FORMAT = "w6a16-sym-g16"
#: Values in one Q6_K group; a super-block holds ``QK_K // GROUP`` of them.
GROUP = 16


def _TYPE() -> int:  # noqa: N802
    """``ggml_type`` of Q6_K, for the weighted fit's type-dispatched entry.

    Looked up on use, not at import: ``gguf`` is an optional package, and every build config imports this module
    through the exporter registry, so a constant here made a non-GGUF build need it.
    """
    return type_id("Q6_K")


def _ROW_BYTES(width: int) -> int:  # noqa: N802
    """Encoded bytes of one Q6_K row of `width` values."""
    return row_bytes(_TYPE(), width)


class Q6K(Grid):
    """llama.cpp's Q6_K: ``quantize_row_q6_K_ref``, read back through `q6k_fields`."""

    name = "q6k"
    gptq_fused = True
    range_rule = False
    stores = Q6K_FORMAT

    def refusal(self, fmt: Format, plan: KernelPlan) -> str | None:  # noqa: ARG002
        """Only the format Q6_K is."""
        if fmt.name != Q6K_FORMAT:
            return f"Q6_K stores 6-bit symmetric codes in groups of 16: format {Q6K_FORMAT}, not {fmt.name}"
        return None

    def fit(self, fmt: Format, w: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa: ARG002
        """Ggml's fit of ``w [..., R, C]``: ``fp16(d) * s`` as ``[..., R, C / 16]``, and no zero point.

        A row that is not a whole number of super-blocks is fitted zero-padded to one, keeping its own groups,
        as `q4k` does: llama.cpp stores such a row from the original weights in another type, so no padded
        block reaches a file and these values only stand in for it in the build's stream.

        Raises:
            Refused: `C` is not a multiple of `GROUP`.
        """
        scale, _ = self._fit(w)
        return scale, None

    def fit_weighted(self, w: torch.Tensor, importance: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Ggml's fit of ``w [E, R, C]`` weighted by ``importance [E, C]``, in `fit`'s terms.

        What ``llama-quantize --imatrix`` does for a Q6_K tensor: the group scales are searched against the
        importances rather than the plain range. The scales still come from ggml, so `range_rule` stays False
        and a method still solves the codes against scales it does not move.

        Raises:
            Refused: `C` is not a whole number of `GROUP`-value groups.
        """
        scale, _ = self._fit(w, importance=importance)
        return scale, None

    def fit_codes(
        self,
        fmt: Format,  # noqa: ARG002
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """`fit` and the codes ggml made on the way, signed as the grid uses them."""
        scale, codes = self._fit(w, codes=True)
        assert codes is not None  # codes=True
        return scale, None, codes

    def _fit(
        self, w: torch.Tensor, codes: bool = False, importance: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(scale, codes) of ``w``; the kernel produces the codes whatever we keep."""
        c = w.shape[-1]
        if c % GROUP:
            raise Refused(
                f"grid {self.name!r}: a row of {c} values is not a whole number of {GROUP}-value groups"
            )
        rows = w.reshape(-1, c).float()
        pad = -c % QK_K
        if pad:
            rows = torch.nn.functional.pad(rows, (0, pad))
        if importance is None:
            records = quantize_q6k(rows)
        else:
            imp = torch.nn.functional.pad(importance, (0, pad))
            n = rows.shape[0] // imp.shape[0]
            # one call per importance row, as the Q4_K fit does: ggml weighs a whole tensor by one vector
            records = b"".join(
                quantize_weighted_chunk(
                    _TYPE(), rows[i * n : (i + 1) * n], _ROW_BYTES(rows.shape[-1]), imp[i]
                )
                for i in range(imp.shape[0])
            )
        d, scales, stored = q6k_fields(records)
        fp16 = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)
        groups = rows.shape[-1] // GROUP
        scale = (torch.from_numpy(fp16)[:, None] * torch.from_numpy(scales.astype(np.float32))).reshape(
            -1, groups
        )[:, : c // GROUP]
        scale = scale.reshape(*w.shape[:-1], c // GROUP).to(w.device)
        if not codes:
            return scale, None
        # ggml stores a code as ``q + 32``; the grid works in the signed value.
        signed = torch.from_numpy(stored.astype(np.int16)) - 32
        q = signed.reshape(rows.shape)[:, :c].to(torch.int8).reshape(w.shape)
        return scale, q.to(w.device)

    def usable(self, s: torch.Tensor) -> torch.Tensor:
        """As fitted."""
        return s

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Fp32: ``fp16 * int8`` is exact there, and the exporter factors it back."""
        return s.float()

    def served(self, s: torch.Tensor) -> torch.Tensor:
        """llama.cpp multiplies with the fp32 product itself."""
        return s.float()

    def codes(
        self, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None, qmin: int, qmax: int
    ) -> torch.Tensor:
        """``nearest_int(w / d)`` clamped; a group with ``d == 0`` gets code 0, as ggml leaves it."""
        assert z is None, "Q6_K is symmetric"
        q = torch.clamp(torch.round(w / s), qmin, qmax)
        return torch.where(s != 0, q, torch.zeros_like(q))

    def value(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """``d * q`` in fp32, as ``dequantize_row_q6_K``."""
        assert z is None, "Q6_K is symmetric"
        return s * q


Q6K_GRID = Q6K()
