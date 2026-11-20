"""Shared helpers for model steps: visit each parameter storage once, and refuse layouts they cannot transform.

Tied parameters (e.g. ``lm_head.weight`` is ``embed_tokens.weight`` when ``tie_word_embeddings``) share one
storage; a model step that visits both would apply its transform twice. `Once` remembers storages by
``data_ptr()`` and skips any it has already transformed in the same role.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..arch import ModelGraph
from ..step import Refused

#: Group kinds whose linears write the residual stream (a residual rotation or permutation moves their outputs).
OUTPUT_KINDS = ("attn_out", "mlp_out", "linear_attn_out")


def check_stream_steps(graph: ModelGraph, steps: Sequence[object]) -> None:
    """Refuse a step that changes the residual basis without carrying the declared outside modules.

    A declaration says how the modules outside the text model meet the residual stream (a vision merger
    writes it, an MTP head reads and writes it). A step that rotates the stream (``rotate``) or reorders
    its channels (``permute``) moves the basis; unless it rewrites those modules too (``carries_stream``),
    they would keep the old basis while serving uses them. A source module no adapter declares is refused
    the same way: nothing says which basis it reads. Two steps that carry declared modules are refused,
    because each writes them from the original checkpoint and the second would overwrite the first's; on a
    model with nothing declared, two such steps compose in the model as before.

    Raises:
        Refused: a step with ``changes_stream`` on has no ``carries_stream`` and the model has outside
            modules (declared or not), or a second step would carry the same source transforms.
    """
    stream = graph.extra.get("stream") or {}
    roots: list[str] = sorted({*(stream.get("writers") or {}), *(stream.get("blocks") or {})})
    outside = list(graph.extra.get("outside") or ())
    unloaded = list(graph.extra.get("unloaded") or ())
    loose = [root for root in outside if root not in roots]
    loose += [n for n in unloaded if not any(n == root or n.startswith(root + ".") for root in roots)]
    carrier: str | None = None
    for step in steps:
        if not getattr(step, "changes_stream", False):
            continue
        name = getattr(step, "name", type(step).__name__)
        if loose:
            raise Refused(
                f"{name}: it changes the residual basis, and the source has {', '.join(sorted(loose)[:3])} "
                "outside the text model that no architecture adapter declares, so nothing says which basis "
                "it reads; declare the module with stream(model) in an adapter, or quantize without "
                f"{name} (e.g. gptq alone); see docs/add-architecture.md"
            )
        if not getattr(step, "carries_stream", False):
            if roots:
                raise Refused(
                    f"{name}: it changes the residual basis, and {', '.join(roots)} outside the text model "
                    "also reads or writes that stream; the step does not carry them through, so they would "
                    f"keep the old basis; quantize without {name} (e.g. gptq alone)"
                )
            continue
        if carrier is not None and roots:
            raise Refused(
                f"{name}: it changes the residual basis a second time ({carrier} already did), and a step "
                "that carries a source module rewrites it from the original checkpoint, so this one would "
                f"overwrite the first's transform; use a single {carrier} (combine its settings) or "
                f"quantize without {name}"
            )
        carrier = name


def check_float_banks(graph: ModelGraph, steps: Sequence[object]) -> None:
    """Refuse a step that folds activation scales into the readers of a bank ``ignore`` keeps float.

    `ignore` may keep a whole routed bank float. The export copies such a bank as the source checkpoint
    stores it, so a step that scales the bank's linears in the model (by folding into their norm or their
    producer) would change weights the checkpoint never receives. AWQ is the only such step.

    Raises:
        Refused: `graph` keeps a routed bank float and `steps` holds a step with ``folds_readers``.
    """
    banks = graph.extra.get("float_banks") or ()
    for step in steps:
        if banks and getattr(step, "folds_readers", False):
            name = getattr(step, "name", type(step).__name__)
            raise Refused(
                f"{name}: it folds activation scales into a norm's readers, and {banks[0]} is kept float by "
                "`ignore`; the export copies that bank as the source stores it, so the folded weights would "
                f"never reach the checkpoint; quantize a float-kept routed bank with rtn or gptq"
            )


def layer_without_classified_writer(graph: ModelGraph) -> str | None:
    """First decoder layer whose residual-writing linears fall in no recognized group kind.

    A group's kind comes from the first block name after the layer prefix (``attn``, ``mlp``, ...). An
    architecture that names its blocks something else (NemotronH's ``mixer``) derives every group as
    ``other``: a residual rotation or permutation moves the readers (gathered regardless of kind) and
    leaves those writers in the old basis, which changes the float function.
    """
    for g in graph.layers:
        if g.groups and not any(grp.kind in OUTPUT_KINDS for grp in g.groups):
            return g.module
    return None


def layer_without_fold_anchor(graph: ModelGraph) -> str | None:
    """First decoder layer with groups but no norm at layer scope (its input comes from inside a submodule)."""
    readers = graph.extra.get("readers", {})
    for g in graph.layers:
        if g.groups and not any(
            k.startswith(g.module + ".") and k.rpartition(".")[0] == g.module for k in readers
        ):
            return g.module
    return None


def ignored_in(graph: ModelGraph, step: str, kinds: tuple[str, ...], what: str) -> None:
    """Refuse a model step when the config's ``ignore`` took a linear out of a group of one of `kinds`.

    The step moves weights across layers by group kind; a linear left out of the groups would be skipped while
    the modules around it move, which changes the float function.

    Args:
        graph: the model graph.
        step: the step's name, for the message.
        kinds: group kinds the step transforms by kind.
        what: what those groups are to the step, for the message.
    """
    for mod, (pattern, kind) in (graph.extra.get("ignored") or {}).items():
        if kind in kinds:
            raise Refused(
                f"{step}: ignore {pattern!r} keeps {mod} float, but {step} finds {what} among the quantized "
                f"groups, so it would skip {mod}; quantize it, or build without {step}"
            )


class Once:
    """``once(p, fn)`` applies ``fn(p)`` in place unless ``p``'s storage was already transformed."""

    def __init__(self) -> None:
        self.seen: set[int] = set()

    def __call__(self, p: torch.Tensor, fn: Callable[[torch.Tensor], None]) -> bool:
        """Apply ``fn`` to ``p`` if its storage is new; returns whether it ran."""
        key = p.data_ptr()
        if key in self.seen:
            return False
        self.seen.add(key)
        fn(p)
        return True


def unique(params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Parameters with duplicate storages removed (first occurrence kept)."""
    seen, out = set(), []
    for p in params:
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            out.append(p)
    return out
