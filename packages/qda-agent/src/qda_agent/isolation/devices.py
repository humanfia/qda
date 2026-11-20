"""The host's GPU facts every isolation level needs: card numbering, UUIDs and device nodes.

A card number is a PCI-bus-order ordinal (what `CUDA_DEVICE_ORDER=PCI_BUS_ID` means) -- or, in a job confined
to part of the node, nvidia-smi's job-local index, which `resolve_cards` maps the same way. The driver's
`/proc` tables map it to its device minor and UUID. These are hardware facts, not sandbox mechanics: `none`
names its cards by UUID too, and `strict` binds their nodes. Any module may import this.
"""

from __future__ import annotations

from .bwrap import (
    NV_CONTROL,
    NV_PROC,
    Card,
    CardError,
    card_uuids,
    cards_by_bus,
    device_nodes,
    invisible_cards,
    job_visible,
    resolve_cards,
    smi_bus,
    smi_capabilities,
    smi_cards,
    smi_memory,
    visible_cards,
)

__all__ = [
    "NV_CONTROL",
    "NV_PROC",
    "Card",
    "CardError",
    "card_uuids",
    "cards_by_bus",
    "device_nodes",
    "invisible_cards",
    "job_visible",
    "resolve_cards",
    "smi_bus",
    "smi_capabilities",
    "smi_cards",
    "smi_memory",
    "visible_cards",
]
