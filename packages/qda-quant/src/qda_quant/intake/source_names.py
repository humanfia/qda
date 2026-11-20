"""One source-to-model name map shared by weight loading and checkpoint export.

Source codecs own physical weight/scale parts. This map owns logical stored names, parameter names and
expert slices. Transformers supplies ordinary conversion rules; an architecture adapter may supply its
own rules for native checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from .probe import _headers, _weight_map
from .source import SOURCE_CODECS

if TYPE_CHECKING:
    from transformers import PreTrainedModel

__all__ = ["SourceNames", "model_rules", "names_under", "under", "unloaded"]


def model_rules(model: PreTrainedModel) -> list:
    """Checkpoint conversion rules for `model`: a matching adapter's, else transformers' own.

    A loader can hand these to ``convert_and_load_state_dict_in_model`` instead of the default mapping, so a
    native checkpoint (DeepSeek's names for DeepSeek-V4) loads through the same rules the export writes by.
    """
    from transformers.conversion_mapping import get_model_conversion_mapping

    from ..arch.adapters import ADAPTERS

    for name in ADAPTERS:
        adapter = ADAPTERS[name]
        naming = getattr(adapter, "source_names", None)
        if naming is not None and adapter.matches(model):
            return naming(model)
    return get_model_conversion_mapping(model)


def unloaded(source: str | Path, model: PreTrainedModel) -> tuple[str, ...]:
    """Source weight names that `model`'s class does not load: an MTP head, a text build's vision tower.

    Names come from the shard index when the weights are not local (`qda plan` fetches metadata only), so a
    source with neither an index nor shards names nothing. A source codec's parts are not model weights and
    are left out.
    """
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    try:
        where = _weight_map(Path(source))
    except FileNotFoundError:
        return ()
    meta = model.state_dict()
    rules = model_rules(model)
    renamings = [r for r in rules if isinstance(r, WeightRenaming)]
    converters = [r for r in rules if isinstance(r, WeightConverter)]
    out = []
    for key in sorted(where):
        if not key.endswith(".weight"):
            continue
        target, _ = rename_source_key(key, renamings, converters, model.base_model_prefix, meta)
        if target not in meta:
            out.append(key)
    return tuple(out)


def names_under(source: str | Path, root: str) -> tuple[str, ...]:
    """Names of the source checkpoint's tensors under module `root`, from the index (no tensor is read)."""
    return tuple(k for k in _weight_map(Path(source)) if k.startswith(root + "."))


def under(source: str | Path, root: str) -> dict[str, torch.Tensor]:
    """Every tensor of the source checkpoint under module `root` (its own and its submodules').

    Reads the shards, unlike `unloaded`, which reads names only. A root the checkpoint has nothing under
    returns an empty dict: the source may hold no MTP head or vision tower at all.
    """
    src = Path(source)
    where = _weight_map(src)
    names = names_under(src, root)
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(where[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for f, keys in by_file.items():
        with safe_open(src / f, "pt") as h:
            out.update({k: h.get_tensor(k) for k in keys})
    return out


class SourceNames:
    """In-memory names -> source checkpoint names.

    When transformers renames the tensors of `source` as it loads them (a multimodal checkpoint's
    ``language_model.model.layers.3`` is ``model.language_model.layers.3`` in memory), those renamings are the
    whole map: a name they do not produce has no source tensor. A guess by suffix could pair a vision-tower
    module with a text layer's packed tensor (``layers.0.self_attn.q_proj``). Otherwise: exact, else the
    unique source name with the same suffix (a text model loaded from a multimodal checkpoint:
    ``model.layers.3`` vs ``model.language_model.layers.3``).
    """

    def __init__(self, source: str | Path, *, model: PreTrainedModel | None = None):
        self.src = Path(source)
        self.where = _weight_map(self.src)
        self.meta = _headers(self.src, set(self.where.values()))
        self.config = json.loads((self.src / "config.json").read_text())
        logical = dict(self.where)
        for key, file in self.where.items():
            weight = key.rpartition(".")[0] + ".weight"
            if weight not in logical and any(
                key in SOURCE_CODECS[name].parts(weight) for name in SOURCE_CODECS
            ):
                logical[weight] = file
        self.keys = set(logical)
        self.converted: dict[str, tuple[Any, dict[str, str], Any]] = {}
        self.renamed, self._memory = _loaded_names(self.src, logical, self.converted, model)
        for weight in logical:
            if not weight.endswith(".weight"):
                continue
            for name in SOURCE_CODECS:
                codec = SOURCE_CODECS[name]
                parts = {p: self.meta[p] for p in codec.parts(weight) if p in self.meta}
                if parts and codec.matches(self.config, parts):
                    for part in parts:
                        self._memory[part] = self._memory[weight]
        where = logical
        # roots of the decoder-layer lists, most layers first (``model.language_model.`` before ``mtp.``)
        roots: dict[str, int] = {}
        for k in where:
            if ".layers." in k:
                r = k[: k.index(".layers.") + 1]
                roots[r] = roots.get(r, 0) + 1
        self.roots = sorted(roots, key=lambda r: -roots[r])
        self.suffix: dict[str, str | None] = {}
        for k in where:
            parts = k.split(".")
            for i in range(1, len(parts)):
                s = ".".join(parts[i:])
                self.suffix[s] = None if s in self.suffix and self.suffix[s] != k else k

    def tensor(self, name: str) -> str | None:
        """Source tensor name of in-memory tensor `name`, or None."""
        if name in self.keys:
            return name
        if self.renamed is not None:
            return self.renamed.get(name)
        head, _, rest = name.partition(".")
        for r in self.roots:  # the in-memory text model's root ("model.") under the checkpoint's
            if head and (r + rest) in self.keys:
                return r + rest
        parts = name.split(".")
        for i in range(1, len(parts) - 1):
            k = self.suffix.get(".".join(parts[i:]))
            if k is not None:
                return k
        return None

    def module(self, name: str) -> str | None:
        """Source module name of in-memory module `name` (found through its ``.weight``), or None."""
        k = self.tensor(name + ".weight")
        return k[: -len(".weight")] if k else None

    def converter_owns(self, name: str) -> bool:
        """True when a source converter defines this fused entry's expert rows and order.

        A converter that only moves one stored tensor between physical layouts (``Transpose``, as
        Qwen3-VL-MoE's fused banks declare) keeps no expert structure of its own: an adapter's serving
        layout may name the pieces instead. A merge or concatenation of per-expert tensors does keep it,
        and its inverse layout is the one to write.
        """
        entry = self.converted.get(name)
        if entry is None:
            return False
        from transformers.core_model_loading import Transpose

        converter = entry[0]
        return len(converter.source_patterns) > 1 or not all(
            isinstance(op, Transpose) for op in converter.operations
        )

    def stored(self, name: str, shape: torch.Size) -> list[tuple[str, tuple]]:
        """Source weights and row slices of a fused parameter, using the loader's inverse conversion.

        Only row labels go through the inverse, so mapping a bank never copies its weight-sized tensors.
        This preserves the converter's projection order and the checkpoint's expert numbering.
        """
        direct = self.tensor(name)
        if direct is not None:
            return [(direct, ())]
        if name not in self.converted:
            raise KeyError(f"{name}: no fused or per-expert weights in the source checkpoint")
        converter, sources, config = self.converted[name]
        reverse = converter.reverse_transform()
        target, pattern = reverse.rename_source_key(name)
        experts, rows, _ = shape
        labels = torch.arange(experts * rows).reshape(experts, rows, 1)
        reverse.add_tensor(target, name, pattern, labels)
        pieces = reverse.convert(target, config=config)
        if set(pieces) != set(sources):
            raise KeyError(f"{name}: the source weights do not match its conversion: {sorted(sources)}")
        out = []
        for key, labels in pieces.items():
            first = int(labels.reshape(-1)[0])
            expert, row = divmod(first, rows)
            sl = (expert, slice(row, row + labels.numel()))
            out.append((sources[key], sl))
        return out

    def split(self, name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Stored tensors of in-memory `tensor`, which the loader merged from several, or None if it did not."""
        if name not in self.converted:
            return None
        converter, sources, config = self.converted[name]
        reverse = converter.reverse_transform()
        target, pattern = reverse.rename_source_key(name)
        reverse.add_tensor(target, name, pattern, tensor)
        return {sources[key]: piece for key, piece in reverse.convert(target, config=config).items()}

    def memory(self, stored: str) -> str | None:
        """Model name for a stored tensor or codec part; None if the source has no such name.

        Unused modules such as MTP still have names, although the build's skeleton may omit them.
        """
        return self._memory.get(stored)


def _loaded_names(
    source: str | Path,
    where: dict[str, str],
    converted: dict[str, tuple[Any, dict[str, str], Any]],
    model: PreTrainedModel | None = None,
) -> tuple[dict[str, str] | None, dict[str, str]]:
    """In-memory name -> source tensor name for every source tensor, as transformers names it loading `source`.

    Uses the renamings ``from_pretrained`` applies (``transformers.conversion_mapping``) to the model class
    the engine loads: `model`, a meta-device skeleton of the source's architecture, or one built here from
    ``config.json``. None without a config or a renaming.
    """
    if model is None and not (Path(source) / "config.json").is_file():
        return None, {}
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    if model is None:
        from transformers import AutoConfig

        from ..step import remote_code_refusal
        from .loading import load_model

        try:
            cfg = AutoConfig.from_pretrained(str(source), trust_remote_code=False)
        except ValueError as e:
            if remote := remote_code_refusal(source, "config", e):
                raise remote from e
            raise
        cfg.quantization_config = None
        delattr(cfg, "quantization_config")  # dynamic config attribute
        with torch.device("meta"):
            model = load_model(config=cfg)
    assert model is not None  # built above, or the caller's meta skeleton
    cfg = model.config
    rules = model_rules(model)
    has_rules = bool(rules)
    renamings = [r for r in rules if isinstance(r, WeightRenaming)]
    converters = [r for r in rules if isinstance(r, WeightConverter)]
    by_pattern = {p: c for c in converters for p in c.source_patterns}
    meta = model.state_dict()
    prefix = model.base_model_prefix
    renamed, memory = {}, {}
    for key in sorted(where):
        direct, _ = rename_source_key(key, renamings, [], prefix, meta)
        target, pattern = rename_source_key(key, renamings, converters, prefix, meta)
        if pattern is not None:
            unprefixed, _ = rename_source_key(direct, [], converters)
            if target == f"{prefix}.{unprefixed}":
                direct = f"{prefix}.{direct}"
            elif unprefixed == f"{prefix}.{target}":
                direct = direct.removeprefix(prefix + ".")
            entry = converted.setdefault(target, (by_pattern[pattern], {}, cfg))
            entry[1][direct] = key
        renamed[direct] = key
        memory[key] = target
    return (renamed if has_rules else None), memory
