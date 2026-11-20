"""Compute and transport calibration token weights for layer objectives."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, cast

import torch

from ..arch import ModelGraph, describe

if TYPE_CHECKING:
    from ..arch import LayerGraph
    from .stores import WeightStore

BatchWeights = torch.Tensor | dict[str, torch.Tensor]
#: the statistics the engine weights by a step's token weights (`stats.GroupStats`, `stats.ExpertStats`); a later
#: step must read one of them, or the weights change nothing (`planning.check_token_weight_readers`)
WEIGHTED_STATISTICS = frozenset({"hessian", "cross_hessian", "expert_hessian"})


@contextlib.contextmanager
def anchor_residency(
    model: torch.nn.Module, graph: ModelGraph, device: torch.device, *, store: WeightStore
) -> Iterator[None]:
    """Run an anchor pass with the weight store's layer placement (`engine.stores`).

    A store that keeps the model off the card places each layer for its forward and releases it immediately
    afterwards; autograd then saves its tensors on the host and restores them as backward needs them.
    """
    if store.on_card:
        yield
        return
    handles = []

    def place(lg: LayerGraph) -> Callable[..., None]:
        return lambda module, _args: store.place(module, lg, device)

    try:
        with store.outer(model, graph, device):
            for lg in graph.layers:
                layer = model.get_submodule(lg.module)
                handles.append(layer.register_forward_pre_hook(place(lg)))
                handles.append(layer.register_forward_hook(lambda module, _args, _out: store.release(module)))
            with torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda"):
                yield
    finally:
        for handle in handles:
            handle.remove()
        for lg in graph.layers:  # a forward that raised skipped its release hook
            store.release(model.get_submodule(lg.module))


def _loss(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Return the anchor next-token loss for one calibration batch."""
    logits = model(input_ids=ids, use_cache=False).logits
    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
        ids[..., 1:].reshape(-1).to(logits.device),
        reduction="sum",
    )


def _guided_groups(graph: ModelGraph) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Map shared-input groups to their output modules and statistic keys."""
    modules: dict[str, list[str]] = {}
    aliases = {}
    for layer in graph.layers:
        for group in layer.groups:
            key = group.norm or group.modules[0]
            modules.setdefault(key, []).extend(group.modules)
            aliases[group.modules[0]] = key
        for bank in layer.banks:
            if not isinstance(bank.experts, str):
                raise ValueError("guided requires fused expert banks with a full-token output")
            shared_norm = next(
                (
                    group.norm
                    for group in layer.groups
                    if group.norm
                    and any(
                        name == shared or name.startswith(shared + ".")
                        for name in group.modules
                        for shared in bank.shared
                    )
                ),
                None,
            )
            key = shared_norm or bank.router
            modules.setdefault(key, []).append(bank.experts)
            aliases[bank.router] = key
            down_key = bank.experts if shared_norm else key
            if shared_norm:
                modules[down_key] = [bank.experts]
            aliases[bank.experts + ".down_proj"] = down_key
    if graph.lm_head:
        modules[graph.lm_head] = [graph.lm_head]
        aliases[graph.lm_head] = graph.lm_head
    return {key: tuple(dict.fromkeys(names)) for key, names in modules.items()}, aliases


def guided(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    *,
    graph: ModelGraph | None = None,
    clip: float = 50.0,
    power: float = 1.0,
) -> list[dict[str, torch.Tensor]]:
    """Compute per-group output-gradient weights over the calibration stream.

    Modules reading a shared input contribute squared output-gradient norms to one group. Each
    group's weights are normalized over all batches, then clipped. The final token has no label.
    cuDNN's attention backward varies between runs; the `guided` step's ``reproducible`` setting avoids it.

    Args:
        model: Anchor model, already placed by the engine.
        batches: The engine's calibration batches, in their build order.
        device: Device receiving the token IDs.
        graph: Verified build graph, or a derived graph for standalone use.
        clip: Maximum normalized weight (default 50).
        power: Exponent applied to raw gradient norms before normalization (default 1).
    """
    groups, aliases = _guided_groups(graph or describe(model, verify=False))
    raw: dict[str, list[torch.Tensor]] = {key: [] for key in groups}
    current: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = {}
    shapes: list[torch.Size] = []
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    training_flags = [(module, module.training) for module in model.modules()]
    handles = []

    def capture(key: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _args: tuple, value: object) -> None:
            tensor = value[0] if isinstance(value, tuple) else value
            if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
                return
            calls[key] = calls.get(key, 0) + 1

            def gradient(grad: torch.Tensor) -> None:
                score = grad.float().square().reshape(-1, grad.shape[-1]).sum(-1)
                current[key] = current[key] + score if key in current else score

            tensor.register_hook(gradient)

        return hook

    try:
        model.eval()
        model.requires_grad_(False)
        for key, names in groups.items():
            handles.extend(model.get_submodule(name).register_forward_hook(capture(key)) for name in names)
        embeddings = cast(Callable[[], torch.nn.Module], model.get_input_embeddings)()
        handles.append(
            embeddings.register_forward_hook(lambda _module, _args, value: value.requires_grad_(True))
        )
        for ids_cpu in batches:
            ids = ids_cpu.to(device)
            current.clear()
            calls.clear()
            with torch.enable_grad():
                loss = _loss(model, ids)
                loss.backward()
            if not bool(torch.isfinite(loss)):
                raise ValueError("guided anchor loss is not finite")
            for key, names in groups.items():
                if calls.get(key, 0) != len(names) or key not in current:
                    raise ValueError(f"guided group {key!r} must contribute once per module to the loss")
                score = current[key]
                if score.numel() != ids.numel():
                    raise ValueError(
                        f"guided group {key!r} has {score.numel()} gradients for {ids.numel()} tokens"
                    )
                raw[key].append(score.detach())
            shapes.append(ids.shape)
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in parameter_flags:
            parameter.requires_grad_(flag)
        for module, flag in training_flags:
            module.training = flag
    if not shapes:
        raise ValueError("guided requires calibration tokens")
    weights: dict[str, tuple[torch.Tensor, ...]] = {}
    for key, parts in raw.items():
        score = torch.cat(parts).clamp_min(0)
        if power != 1.0:
            score = score.pow(power)
        mean = score.double().mean()
        if not bool(torch.isfinite(score).all()) or not bool(mean > 0):
            raise ValueError(f"guided group {key!r} requires finite gradient norms with a positive mean")
        weights[key] = (score / mean).float().clamp_max(clip).split([shape.numel() for shape in shapes])
    return [
        {alias: weights[key][index].reshape(shape) for alias, key in aliases.items()}
        for index, shape in enumerate(shapes)
    ]


@torch.no_grad()
def fisher_trace(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    *,
    window: int = 16,
    floor: float = 0.25,
) -> list[torch.Tensor]:
    """Compute Fisher-trace weights with a causal window and a stream-wide mean.

    Args:
        model: Anchor model, already placed by the engine.
        batches: The engine's calibration batches, in their build order.
        device: Device receiving the token IDs.
        window: Number of future positions priced by each hidden state (default 16).
        floor: Minimum fraction of the stream's mean weight (default 0.25).
    """
    model.eval()
    out: list[torch.Tensor] = []
    for ids_cpu in batches:
        ids = ids_cpu.to(device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        score = 1.0 - probabilities.square().sum(-1)
        cumsum = torch.cat([score.new_zeros((*score.shape[:-1], 1)).double(), score.double().cumsum(-1)], -1)
        positions = torch.arange(score.shape[-1], device=score.device)
        high = (positions + window).clamp(max=score.shape[-1])
        score = ((cumsum[..., high] - cumsum[..., positions]) / (high - positions).double()).float()
        out.append(score.detach())
    if not out:
        raise ValueError("fisher_trace requires calibration tokens")
    scores = torch.cat([score.reshape(-1) for score in out])
    mean = scores.mean()
    if not bool(torch.isfinite(scores).all()) or not bool(mean > 0):
        raise ValueError("fisher_trace requires finite scores with a positive mean")
    return [floor + (1.0 - floor) * score / mean for score in out]
