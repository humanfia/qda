"""A small RTN specialization showing how a recipe step becomes an external plugin."""

from typing import Literal

from qda_quant.steps.rtn import RTN


class ClipRTN(RTN):
    """Round to nearest with MSE clipping as the example's default setting."""

    name = "clip_rtn"
    clip: Literal["minmax", "mse"] = "mse"
