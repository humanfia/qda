"""Base of the steps that train quantized weights to reproduce a layer's output on the calibration batches."""

from __future__ import annotations

from typing import Literal

from ..step import LayerContext, QuantStep


class LayerFit(QuantStep):
    """Declares the training target once; the engine sees the choice only through `needs`.

    Attributes:
        target: ``quant_stream`` or ``float_stream``; each method's docstring says what they mean for it.
    """

    target: Literal["quant_stream", "float_stream"] = "quant_stream"

    def picks_ranges(self) -> bool:
        """It learns group ranges."""
        return True

    @property
    def _stat(self) -> str:
        return "layer_io_float" if self.target == "float_stream" else "layer_io"

    @property
    def needs(self) -> frozenset[str]:  # type: ignore[override]
        """The `layer_io` statistic that yields this step's target."""
        return frozenset({self._stat})

    def layer_io(self, ctx: LayerContext) -> list:
        """This layer's ``(hidden_states, args, kwargs, target)``, one per calibration batch."""
        return list(ctx.stats[(self._stat, ctx.graph.module)])
