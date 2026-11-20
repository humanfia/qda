"""Solve GDN output projections in the pinned converter's value-head column order."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import torch
from torch import nn

from ..step import Refused

# A derived basis fact held only while the layer is calibrated, solved and written.
ORDER = "_qda_gguf_columns"


def value_head_columns(nk: int, nv: int, dim: int) -> torch.Tensor:
    """The converter's grouped-to-tiled value-head order, as a column index map.

    The same map the decoder inverts: ``arange(nv * dim)`` reshaped grouped (key head, value head within the
    group, dim) and read tiled (value head, key head, dim). Its ``argsort`` is the inverse.
    """
    return torch.arange(nv * dim).reshape(nk, nv // nk, dim).transpose(0, 1).reshape(-1)


def column_orders(layer: nn.Module) -> Iterator[tuple[nn.Linear, torch.Tensor]]:
    """The converter's grouped-to-tiled value-head indices for each GDN output projection."""
    for module in layer.modules():
        nk = getattr(module, "num_k_heads", None)
        nv = getattr(module, "num_v_heads", None)
        dim = getattr(module, "head_v_dim", None)
        projection = getattr(module, "out_proj", None)
        if nk is None or nv is None or dim is None or not isinstance(projection, nn.Linear) or nk == nv:
            continue
        if nk <= 0 or nv % nk or projection.in_features != nv * dim:
            raise Refused("gguf: GDN output projection has inconsistent key/value head dimensions")
        yield projection, value_head_columns(nk, nv, dim)


@contextmanager
def template_columns(layer: nn.Module) -> Iterator[None]:
    """Pair each output projection's column permutation with the same input permutation.

    HF's internal key/value pairing stays unchanged. The paired permutation preserves the forward,
    while calibration and GPTQ see the template's input order. On exit, served weights return to HF order.
    """
    with ExitStack() as stack:
        for projection, order in column_orders(layer):
            if hasattr(projection, ORDER):
                raise Refused("gguf: output projection is already in template column order")
            with torch.no_grad():
                projection.weight.copy_(projection.weight.index_select(1, order.to(projection.weight.device)))
            setattr(projection, ORDER, order)

            def restore(projection=projection, order=order):
                with torch.no_grad():
                    projection.weight.copy_(
                        projection.weight.index_select(1, order.argsort().to(projection.weight.device))
                    )
                delattr(projection, ORDER)

            stack.callback(restore)

            def permute(_module, args, order=order):
                return (args[0].index_select(-1, order.to(args[0].device)), *args[1:])

            hook = projection.register_forward_pre_hook(permute, prepend=True)
            stack.callback(hook.remove)
        yield
