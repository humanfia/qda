"""What the registered architecture adapters add to the graph, and the serving engines' weight names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from torch import nn

from ..step import Refused
from . import ModelGraph
from .derive_model import text_config


def _adapt(model: nn.Module, graph: ModelGraph) -> ModelGraph:
    """`graph` after the registered adapters that match `model`, and with the declared stream.

    At most one adapter fixes the graph (its name is ``graph.adapter``) and at most one adapter declares the
    residual stream of the modules outside the text model (``graph.extra["stream"]``, its name in
    ``graph.extra["stream_adapter"]``, which build provenance records beside ``graph.adapter``). At most one
    gives the serving engines' names of weights it keeps float (``graph.extra["runtime_names"]``).
    A declaration-only adapter subclasses `StreamDeclaration`, so a layout that also needs a fix-up (a fused
    expert bank) can declare its stream without competing with it.

    Raises:
        Refused: more than one adapter fixes the graph, declares the stream or names runtime modules; a
            declaration names a module the model does not have, or a stream writer that is not a Linear of
            the residual width.
    """
    from .adapters import ADAPTERS  # here, not at the top: the adapters import this module's helpers
    from .adapters.stream import StreamDeclaration

    hits = [(k, a) for k, a in ADAPTERS.items() if a.matches(model)]
    fixing = [(k, a) for k, a in hits if not isinstance(a, StreamDeclaration)]
    declaring = [(k, a) for k, a in hits if callable(getattr(a, "stream", None))]
    naming = [(k, a) for k, a in hits if callable(getattr(a, "runtime_names", None))]
    if len(naming) > 1:
        raise Refused(
            f"{type(model).__name__} matches more than one architecture adapter that names runtime "
            f"modules: {[k for k, _ in naming]}; remove or narrow one of them"
        )
    if len(fixing) > 1:
        raise Refused(
            f"{type(model).__name__} matches more than one architecture adapter: {[k for k, _ in fixing]}; "
            "remove or narrow one of them so only one matches"
        )
    if len(declaring) > 1:
        raise Refused(
            f"{type(model).__name__} matches more than one architecture adapter that declares its residual "
            f"stream: {[k for k, _ in declaring]}; remove or narrow one of them"
        )
    for key, adapter in fixing:
        graph = replace(adapter.adjust(model, graph), adapter=key)
        served = getattr(adapter, "serve_names", None)
        if served is not None:
            graph.extra["serve_names"] = _serving_names(key, served(model))
        configured = getattr(adapter, "config_names", None)
        if configured is not None:
            graph.extra["config_names"] = {
                engine: _serving_names(key, names) for engine, names in configured(model).items()
            }
    packing = [a for _, a in hits if callable(getattr(a, "pack_like", None))]
    if packing:  # the exporter packs these; no model names live there
        like = {k: v for a in packing for k, v in cast(Any, a).pack_like(model).items()}
        graph = replace(graph, extra={**graph.extra, "pack_like": like})
    if naming:  # the exporter applies these; no engine naming lives there
        key, adapter = naming[0]
        names = cast(Any, adapter).runtime_names(model)  # optional protocol member; the match checked it
        graph = replace(graph, extra={**graph.extra, "runtime_names": dict(names)})
    if declaring:
        key, adapter = declaring[0]
        declared = cast(StreamDeclaration, adapter).stream(model)
        stream = _checked_stream(model, key, declared, text_config(model))
        graph = replace(graph, extra={**graph.extra, "stream": stream, "stream_adapter": key})
    return graph


def _serving_names(adapter: str, declared: Mapping[str, str]) -> dict[str, str]:
    """An adapter's ``serve_names`` declaration, checked: dotted paths in, dotted paths out.

    Raises:
        Refused: a key or value is empty or has an empty segment (``a..b``, ``.a``).
    """
    names = dict(declared)
    bad = {k: v for k, v in names.items() if not _dotted(k) or not _dotted(v)}
    if bad:
        raise Refused(
            f"arch adapter {adapter!r}: serve_names renames {bad!r}; its keys and values must each be dotted "
            "paths of non-empty name segments"
        )
    return names


def _dotted(name: str) -> bool:
    """True when `name` is a dotted path with no empty segment."""
    return bool(name) and all(name.split("."))


def served_name(name: str, renames: Mapping[str, str]) -> str:
    """`name` as the serving engines build it: the longest declared path at each dotted boundary moves once.

    The value is never rescanned, unmatched segments keep their names, and rules compose inside one name,
    so one map carries both a one-segment rule at any depth (``mlp``) and a whole subtree move.
    """
    if not renames:
        return name
    rules = sorted(((k.split("."), v.split(".")) for k, v in renames.items()), key=lambda r: -len(r[0]))
    parts, out, i = name.split("."), [], 0
    while i < len(parts):
        for src, dst in rules:
            if parts[i : i + len(src)] == src:
                out += dst
                i += len(src)
                break
        else:
            out.append(parts[i])
            i += 1
    return ".".join(out)


def _checked_stream(model: nn.Module, key: str, stream: dict, cfg: Any) -> dict:
    """`stream` after checking its blocks and every writer the model loads.

    A declaration is code that names modules; when a layout changes under it the names stop existing, and
    that must refuse the plan rather than rotate the wrong tensors. A declared name that `model` has no
    module for is a source-only name (a text build's vision merger, an unloaded MTP head); `rotate` checks
    it against the source checkpoint, which is not known here.

    Raises:
        Refused: a declared writer the model loads is not a Linear, or its output is not the residual
            width; a block is not a `SourceBlock`.
    """
    from .adapters.stream import SourceBlock

    mods = dict(model.named_modules())
    writers = stream.get("writers") or {}
    blocks = stream.get("blocks") or {}
    wrong = [k for k, b in blocks.items() if not isinstance(b, SourceBlock)]
    if wrong:
        raise Refused(
            f"architecture adapter {key!r} declares block {wrong[0]} as a {type(blocks[wrong[0]]).__name__}; "
            "declare a qda_quant.arch.adapters.SourceBlock"
        )
    hidden = getattr(cfg, "hidden_size", None)
    for name in (n for ns in writers.values() for n in ns):
        m = mods.get(name)
        if m is None:
            continue
        if not isinstance(m, nn.Linear):
            raise Refused(
                f"architecture adapter {key!r} declares {name} as a stream writer, but it is a "
                f"{type(m).__name__}; declare the Linear whose output is the residual stream"
            )
        if hidden and m.out_features != hidden:
            raise Refused(
                f"architecture adapter {key!r} declares {name} as a stream writer, but it writes "
                f"{m.out_features} channels and the residual stream is {hidden}; declare the Linear whose "
                "output is the residual stream"
            )
    return stream
