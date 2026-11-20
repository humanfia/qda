"""Declaring what lies outside the text model, for a residual rotation.

A module outside the text model (a vision tower, a projector, an MTP head) is not reachable by derivation:
its forward never shows up in the decoder layers. A rotate with ``residual: true`` therefore refuses every
such module that no adapter declares. An adapter subclasses `StreamDeclaration` and returns, from
``stream(model)``::

    {
        "writers": {outside module: (names of its Linears that write the residual stream,)},
        "blocks": {source root: SourceBlock(...)},
    }

``writers`` maps every outermost module outside the text model to the Linears in it whose output is the
residual stream (a vision merger's last Linear); an empty tuple declares a module that never reaches the
stream (a vision tower). ``blocks`` names a head outside the text model that the model class does not load
(an MTP head): its tensors come from the source checkpoint, and ``rotate`` rewrites them into the rotated
basis (`qda_quant.steps._source_block`).

An entry whose module the model does not have is a source-only name: the text-only class of a multimodal
checkpoint loads neither the vision tower nor the MTP head, and a residual rotation still has to reach them
in the published checkpoint.

A declaration names Linears; it cannot say that nothing runs after them before the stream (a GELU after a
merger's last Linear). For a module the model loads, ``probe(model, root)`` gives a small input to run it on,
and `rotate` refuses unless rotating the declared writers rotates the module's output (by execution, fp64).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from .. import ModelGraph


@dataclass(frozen=True)
class SourceBlock:
    """A head outside the text model whose weights come from the source checkpoint.

    The residual rotation folds every norm into the linears that read its output and rotates those linears
    on their input, and rotates the writers on their output. The block's last norm feeds the shared output
    head, which the rotation gives the target's final-norm gain; no per-channel gain commutes with a
    Hadamard rotation, so that one norm is stored as the best constant fit (a `rotate` run records the
    constant and its residual in ``build.json``).

    Attributes:
        fc: the Linear that mixes both inputs into the block's residual, in ``inputs`` order.
        inputs: the two norms on the block's inputs (the embedding side, the hidden state side).
        folds: (norm, its readers) inside the block: the norm's gain moves into its readers.
        writers: the block's Linears that write its residual stream.
        norm: the block's final norm, whose output feeds the shared head.
        kept: modules of the block whose tensors the rewrite copies unchanged (head-dimension norms).
        offset: the block norms' storage convention: a gain is the stored weight plus ``offset``.
        fused: readers or writers stored as stacked expert parameters ``[experts, out, in]``,
            by exact source tensor name (without a ``.weight`` suffix).
        refusal: why this block cannot use the residual rewrite, checked from source names at plan time.
        approximation: why the residual rewrite is exact only up to one recorded constant (a block that feeds
            a shared head), checked from source names at plan time: the plan warns, the build records the
            error in ``build.json``.
    """

    fc: str
    inputs: tuple[str, str]
    folds: tuple[tuple[str, tuple[str, ...]], ...]
    writers: tuple[str, ...]
    norm: str
    kept: tuple[str, ...]
    offset: float
    fused: tuple[str, ...] = ()
    refusal: str | None = None
    approximation: str | None = None


class StreamDeclaration:
    """An adapter that only declares the residual stream of the modules outside the text model."""

    def stream(self, model: nn.Module) -> dict:
        """``{"writers": {module: (Linear, ...)}, "blocks": {source root: SourceBlock}}``."""
        raise NotImplementedError

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:  # noqa: ARG002
        """The graph unchanged: a declaration fixes no derivation."""
        return graph

    def probe(self, model: nn.Module, root: str) -> dict[str, Any] | None:  # noqa: ARG002
        """Keyword arguments that run the loaded module `root` on a small input; None if there are none.

        `rotate` refuses a loaded module with declared writers and no probe: its rotation is unverified.
        """
        return None


def qwen_vision_probe(model: nn.Module) -> dict[str, Any]:
    """A Qwen vision tower's input for one image of 2 x 2 merged patches (``hidden_states``, ``grid_thw``)."""
    v: Any = cast(Any, model).config.vision_config
    m = v.spatial_merge_size
    features = v.in_channels * v.temporal_patch_size * v.patch_size**2
    pixels = torch.randn(4 * m * m, features, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    return {"hidden_states": pixels, "grid_thw": torch.tensor([[1, 2 * m, 2 * m]])}
