"""The attention kernel of the steps that offer a ``reproducible`` setting."""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

from torch.nn.attention import SDPBackend, sdpa_kernel


def attention_kernel(reproducible: bool) -> AbstractContextManager:
    """Torch's math attention if `reproducible`, else the default kernel.

    cuDNN's attention backward varies in its last bits from one call to the next, on the same graph. The math
    kernel's backward repeats, at a cost in time and card memory that each step's docstring states.
    """
    return sdpa_kernel(SDPBackend.MATH) if reproducible else contextlib.nullcontext()
