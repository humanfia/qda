"""An adapter that adds a structurally identified projection to the model graph."""

from dataclasses import replace

from qda_quant.arch import LinearGroup, ModelGraph
from torch import nn


class TinyProjectionAdapter:
    """Add a decoder layer's linear projection when generic tracing leaves it out."""

    def matches(self, model: nn.Module) -> bool:
        """Match a model whose first decoder layer has a linear module named ``projection``."""
        stack = getattr(getattr(model, "model", model), "layers", ())
        return bool(stack and isinstance(getattr(stack[0], "projection", None), nn.Linear))

    def adjust(self, model: nn.Module, graph: ModelGraph) -> ModelGraph:
        """Append the projection group to the first decoder layer's graph."""
        del model
        if not graph.layers:
            return graph
        first, *rest = graph.layers
        group = LinearGroup((f"{first.module}.projection",), "other")
        layer = replace(first, groups=(*first.groups, group))
        return replace(graph, layers=(layer, *rest), adapter="tiny_projection")


TINY_PROJECTION = TinyProjectionAdapter()
