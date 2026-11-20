"""Built-in statistics for the ``qda.stats`` table in ``qda_quant.step``."""

from __future__ import annotations

import torch

from ..step import Statistic
from .runner import LayerIO, Slot


def _float_target(slots: list[Slot], index: int, device: torch.device) -> LayerIO:
    return LayerIO(slots, index, device, float_stream=True)


#: As ``layer_io``, but the output is the float layer on the float stream: the float model's own output, so a step
#: fitting it also corrects the error the layer inherits. The inputs stay the quantized stream.
LAYER_IO_FLOAT = Statistic(float_stream=True, single_card=True, provide=_float_target)

#: ``LayerStats`` computes it from the paired float inputs, so the engine only has to keep their stream.
CROSS_HESSIAN = Statistic(float_stream=True)
