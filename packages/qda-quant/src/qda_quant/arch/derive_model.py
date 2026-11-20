"""Find a model's text model, its decoder layers, and the modules beside or outside it."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..step import Refused


def decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The decoder layer list: the ModuleList with ``config.num_hidden_layers`` entries."""
    cfg = text_config(model)
    n = getattr(cfg, "num_hidden_layers", None)
    lists = [(k, m) for k, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) == n]
    if not lists:
        raise Refused(
            f"no ModuleList of num_hidden_layers={n} decoder layers in {type(model).__name__}; "
            "add an architecture adapter for this model"
        )
    return min(lists, key=lambda km: km[0].count("."))


def text_config(model: nn.Module) -> Any:
    """The config of the language model (the text part of a multimodal config)."""
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def _num_experts(cfg: Any) -> int | None:
    return getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)


def _name_of(model: nn.Module, mod: nn.Module | None) -> str:
    if mod is None:
        return ""
    return next((k for k, m in model.named_modules() if m is mod), "")


def _outside(model: nn.Module, text_root: str, skip: set[str]) -> tuple[str, ...]:
    """The outermost modules of `model` that are neither the text model nor under or above it.

    A multimodal model puts a vision tower and its projector beside the text model (``model.visual`` beside
    ``model.language_model``); an MTP head sits beside it too. A residual rotation cannot rotate the stream
    they write without knowing which of them write it, so an architecture adapter declares each one (see
    `qda_quant.arch.adapters.stream`) and there is no declaration to infer. `skip` holds the text model's
    own embedding, head and final norm, which its path reaches outside the decoder layers.
    """
    roots: set[str] = set()
    for name in dict(model.named_modules()):
        if not name or name == text_root or name.startswith(text_root + "."):
            continue
        if text_root.startswith(name + "."):
            continue  # an ancestor of the text model
        if any(name == s or name.startswith(s + ".") for s in skip if s):
            continue
        parts = name.split(".")
        for k in range(1, len(parts) + 1):
            root = ".".join(parts[:k])
            if not (text_root == root or text_root.startswith(root + ".")):
                roots.add(root)
                break
    return tuple(sorted(roots))
