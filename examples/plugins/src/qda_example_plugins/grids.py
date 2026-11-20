"""A small grid showing how a quantization grid becomes an external plugin."""

import torch
from qda_quant.solve.grid import Signed8


class Signed8Bf16(Signed8):
    """The signed range rule, stored in the bf16 a bf16-scale kernel multiplies with, not in fp16."""

    name = "signed8_bf16"

    def stored(self, s: torch.Tensor) -> torch.Tensor:
        """Bf16."""
        return super().stored(s).to(torch.bfloat16)


SIGNED8_BF16 = Signed8Bf16()
