"""Select and implement model weight placement policies."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..registry import Registry

if TYPE_CHECKING:
    from ..arch import LayerGraph, ModelGraph
    from .planning import Plan


@dataclass(frozen=True)
class MemoryNeed:
    """What the planner knows when it picks a store.

    Attributes:
        model_bytes: parameter bytes of the whole model as loaded.
        layer_bytes: parameter bytes of the largest decoder layer.
        card_budget: bytes left for weights on the least free card, after the activations and the plan's terms.
        host_free: available host memory in bytes.
        cards: number of cards (one process each) of the build.
    """

    model_bytes: int
    layer_bytes: int
    card_budget: int
    host_free: int
    cards: int = 1


class WeightStore:
    """One way to hold a model's weights during a build; subclass it and register an instance in its registry.

    Third-party stores register through their package's ``qda.plugins`` hook.

    Attributes:
        name: registry name, the value of ``residency:`` in a config.
        priority: the planner takes the lowest-priority store that fits.
        on_card: the whole model stays on the card, so `place`, `release` and `outer` do nothing.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 0
    on_card: ClassVar[bool] = False

    def card_bytes(self, need: MemoryNeed) -> int:
        """Bytes of the model this store keeps on every card: one layer unless it keeps the whole model."""
        return need.layer_bytes

    def layer_bytes(self, plan: Plan, lg: LayerGraph, loaded: int) -> int:  # noqa: ARG002
        """Bytes of layer `lg` this store keeps on the card; `loaded` is its size as loaded.

        A store that keeps some tensors as stored (a compressed layer, a lookup table) reports their stored size.
        """
        return loaded

    def fits(self, need: MemoryNeed) -> bool:
        """Whether a build of `need` can use this store; the planner's peak check covers the rest."""
        return self.card_bytes(need) <= need.card_budget

    def check(self, plan: Plan) -> None:  # noqa: B027
        """Raise `Refused` if this store cannot run `plan` (called at plan time, before any weight loads)."""

    def load(self, job: Any, device: torch.device) -> torch.nn.Module:
        """The source model of `job`, configured for ``job.target``, its weights where this store keeps them."""
        raise NotImplementedError

    def place(self, layer: torch.nn.Module, lg: LayerGraph, device: torch.device) -> None:
        """Make decoder layer `layer` (graph `lg`) ready to run on `device`."""

    def release(self, layer: torch.nn.Module) -> None:
        """Undo `place` after the layer's turn."""

    @contextlib.contextmanager
    def outer(self, model: torch.nn.Module, graph: ModelGraph, device: torch.device) -> Iterator[None]:  # noqa: ARG002
        """Keep the non-layer modules (embedding, rotary, final norm, head) on `device` inside the block."""
        yield


def from_pretrained(job: Any) -> torch.nn.Module:
    """`job`'s source model in host memory (mmap), with the model steps' result when ``job.weights`` is set."""
    from ..intake.loading import load_model

    m = load_model(
        job.model,
        dtype=job.dtype,
        target=job.target,
        images=getattr(job.plan, "calib_inputs", None) is not None,
    ).eval()
    if job.weights is not None:
        from safetensors.torch import load_model

        load_model(m, job.weights, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Resident(WeightStore):
    """The whole model on the card: it fits with room for the activations and the workspace."""

    name = "resident"
    priority = 0
    on_card = True

    def card_bytes(self, need: MemoryNeed) -> int:  # noqa: D102
        return need.model_bytes

    def load(self, job: Any, device: torch.device) -> torch.nn.Module:  # noqa: D102
        # host (mmap) first, then one bulk move: a device_map load dispatches every tensor through the CPU,
        # which costs about 2 s more for an 8B model than reading the checkpoint lazily and moving it once
        m = from_pretrained(job)
        return m if device.type == "cpu" else m.to(device)


class Stream(WeightStore):
    """The whole model in host memory; each layer on the card for its turn only."""

    name = "stream"
    priority = 1

    def fits(self, need: MemoryNeed) -> bool:  # noqa: D102
        # every card's process holds a copy of the model
        return super().fits(need) and need.cards * need.model_bytes <= need.host_free

    def load(self, job: Any, device: torch.device) -> torch.nn.Module:  # noqa: D102, ARG002
        return from_pretrained(job)

    def place(self, layer: torch.nn.Module, lg: LayerGraph, device: torch.device) -> None:  # noqa: D102, ARG002
        layer.to(device)

    def release(self, layer: torch.nn.Module) -> None:  # noqa: D102
        layer.to("cpu")

    @contextlib.contextmanager
    def outer(self, model: torch.nn.Module, graph: ModelGraph, device: torch.device) -> Iterator[None]:  # noqa: D102
        from .runner import recorders

        with recorders(model, graph, {}):
            model.to(device)  # the non-layer modules only: the layers are recorders now
        try:
            yield
        finally:
            with recorders(model, graph, {}):
                model.to("cpu")


def _adopt(key: str, store: Any) -> None:
    if not isinstance(store, WeightStore):
        raise TypeError(f"weight store {key!r} must be a WeightStore instance, not {type(store).__name__}")


WEIGHT_STORES: Registry[WeightStore] = Registry(
    "weight_stores",
    {
        "resident": "qda_quant.engine.stores:RESIDENT",
        "stream": "qda_quant.engine.stores:STREAM",
        "disk": "qda_quant.engine.disk:DISK",
    },
    on_add=_adopt,
)
RESIDENT = Resident()
STREAM = Stream()
