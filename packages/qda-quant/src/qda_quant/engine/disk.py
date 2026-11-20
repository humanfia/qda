"""Load model layers from checkpoint shards for the disk weight store."""

from __future__ import annotations

import contextlib
import copy
import json
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from ..step import Refused
from .stores import Stream, WeightStore

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..arch import LayerGraph
    from .planning import Plan


class _Reader:
    """Reads one model's tensors from its checkpoint into the model, a set of modules at a time.

    Args:
        source: local checkpoint directory.
        model: the model built on the meta device.
        layers: module names of the decoder layers.
        dtype: dtype the model is loaded in.
        preserved: encoded lookup tensors the graph declares must stay in their stored dtype.
        host_tensors: lookup tensors kept on the host; their modules transfer only selected rows.
    """

    def __init__(
        self,
        source: str,
        model: PreTrainedModel,
        layers: list[str],
        dtype: torch.dtype,
        preserved: tuple[str, ...] = (),
        host_tensors: tuple[str, ...] = (),
    ) -> None:
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        from ..intake.source_names import SourceNames, model_rules

        self.src, self.dtype = Path(source), dtype
        self.host_tensors = host_tensors
        self._model = weakref.ref(model)
        self.buffers = {n: b for n, b in model.named_buffers(remove_duplicate=False) if not b.is_meta}
        names = SourceNames(self.src)
        self.where = names.where
        self.host_files = {self.where[k] for k in self.where if names.memory(k) in host_tensors}
        #: logical weight -> (codec, its stored parts) for every weight a source codec decodes; float weights
        #: are read as stored, through the loader's own dtype cast
        raw = frozenset(k for k in self.where if names.memory(k) in preserved)
        self.coded = _coded(self.src, names.meta, raw)
        parts = {p for _, ps in self.coded.values() for p in ps}
        logical = [k for k in self.where if k not in parts] + list(self.coded)
        #: the loader and exporter share the adapter's native naming rules
        self.mapping = cast(list[WeightConverter | WeightRenaming], model_rules(model))
        self.dtype_plan = model._get_dtype_plan(dtype)
        renamings = [m for m in self.mapping if isinstance(m, WeightRenaming)]
        converters = [m for m in self.mapping if isinstance(m, WeightConverter)]
        meta = model.state_dict()
        prefixes = sorted(((name + ".", name) for name in layers), reverse=True)  # longest first on ties
        #: checkpoint keys by the decoder layer they load into; "" for the non-layer modules. Keys that load into
        #: nothing (an MTP layer the model class drops) are left out, as `from_pretrained` leaves them.
        self.keys: dict[str, list[str]] = {}
        for key in logical:
            target, _ = rename_source_key(key, renamings, converters, model.base_model_prefix, meta)
            if target not in meta:
                continue
            owner = next((name for p, name in prefixes if target.startswith(p)), "")
            self.keys.setdefault(owner, []).append(key)

    @property
    def model(self) -> PreTrainedModel:
        """The model this reader fills."""
        model = self._model()
        if model is None:
            raise RuntimeError(f"{self.src}: the model this reader filled no longer exists")
        return model

    def restore(self, module: torch.nn.Module, prefix: str, device: torch.device) -> None:
        """Put the constructor's buffers on `device` where `module` (named `prefix` in the model) has none read."""
        for name, b in module.named_buffers(prefix=prefix, remove_duplicate=False):
            if b.is_meta and name in self.buffers:
                owner, _, leaf = name[len(prefix) + 1 if prefix else 0 :].rpartition(".")
                module.get_submodule(owner)._buffers[leaf] = self.buffers[name].to(device)

    def load(self, owner: str, device: torch.device) -> None:
        """Read the tensors of `owner` (a layer's module name, or "" for the non-layer modules) onto `device`."""
        from safetensors import safe_open
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig
        from transformers.utils import logging as hf_logging

        keys = self.keys.get(owner, [])
        stored = [p for k in keys for p in (self.coded[k][1] if k in self.coded else (k,))]
        files = sorted({self.where[k] for k in stored})
        quiet = not hf_logging.is_progress_bar_enabled()
        with contextlib.ExitStack() as stack:
            handles = {
                f: stack.enter_context(
                    safe_open(
                        self.src / f,
                        framework="pt",
                        device="cpu",
                        backend=_backend(f, self.host_files, device),
                    )
                )
                for f in files
            }
            state = {}
            for k in keys:
                if k in self.coded:  # decoded on `device`, in the build dtype
                    codec, ps = self.coded[k]
                    state[k] = codec.decode(
                        {p: handles[self.where[p]].get_tensor(p).to(device) for p in ps}, self.dtype
                    )
                else:
                    state[k] = handles[self.where[k]].get_slice(k)
            config = LoadStateDictConfig(
                device_map={"": device, **dict.fromkeys(self.host_tensors, "cpu")},
                dtype=self.dtype,
                dtype_plan=self.dtype_plan,
                weight_mapping=copy.deepcopy(self.mapping),
            )
            hf_logging.disable_progress_bar()
            try:
                info, _ = convert_and_load_state_dict_in_model(self.model, state, config)
            finally:
                if not quiet:
                    hf_logging.enable_progress_bar()
        errors = [
            *info.error_msgs,
            *map(str, info.mismatched_keys),
            *map(str, info.conversion_errors.items()),
        ]
        if errors:
            raise RuntimeError(f"{self.src}: loading {owner or 'the non-layer modules'} failed: {errors[:3]}")


def _backend(file: str, host_files: set[str], device: torch.device) -> str:
    """How a shard is read for a load onto `device`: mapped for a card, a private copy on the CPU.

    A layer placed on a card is read as views of its shard and copied to the card once; the views go with the
    layer, so mapping costs no host copy (16 GB of copies over an 8B model's build). On the CPU the layer's
    weights *are* those views, and a step that writes one dirties the shard's pages into anonymous memory, so
    there the read owns its buffer. A tensor the adapter declares host-side is mapped: its rows are selected
    from the mapping for every load, never written.
    """
    return "mmap" if file in host_files or device.type != "cpu" else "pread"


def _coded(
    src: Path,
    headers: dict[str, tuple[str, list[int]]],
    preserved: frozenset[str] = frozenset(),
) -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Logical weight -> (source codec, stored parts) of every weight of a quantized source that is not float.

    A source without ``quantization_config`` is float throughout: nothing to decode.

    Raises:
        Refused: a weight no codec decodes.
    """
    from ..intake.source import FLOAT, codec_for

    cfg = json.loads((src / "config.json").read_text())
    if "quantization_config" not in cfg:
        return {}
    out = {}
    for k in headers:
        stem, _, leaf = k.rpartition(".")
        if k in preserved or leaf not in ("weight", "weight_packed"):
            continue
        codec = codec_for(cfg, headers, stem + ".weight")
        if codec is not FLOAT:
            out[stem + ".weight"] = (codec, tuple(p for p in codec.parts(stem + ".weight") if p in headers))
    return out


class Disk(Stream):
    """The model on the meta device; each layer read from the checkpoint for its turn and dropped after it.

    Refuses what needs the whole model at once: model steps (and so a second layer pass), a verified graph and
    staged builds. Native encodings are checked per weight through the source codecs before loading.
    """

    name = "disk"
    priority = 2
    #: one layer on the card: `Stream.fits` would also ask the host for the whole model, which disk never holds
    fits = WeightStore.fits

    def __init__(self) -> None:
        self._readers: weakref.WeakKeyDictionary[torch.nn.Module, _Reader] = weakref.WeakKeyDictionary()

    def layer_bytes(self, plan: Plan, lg: LayerGraph, loaded: int) -> int:
        """Exclude lookup parameters held in host mmap; their selected rows alone reach the card."""
        return loaded - plan.host_parameter_bytes.get(lg.module, 0)

    def check(self, plan: Plan) -> None:  # noqa: D102
        why = "residency disk keeps one layer in memory at a time"
        kinds = [k for k, _ in plan.segments]
        if "model" in kinds:
            steps = [s.name for k, ss in plan.segments if k == "model" for s in ss]
            raise Refused(
                f"{why}; model steps {steps} need the whole model (and split the layer steps into passes)"
            )
        if any(s.needs_verified_graph for _, ss in plan.segments for s in ss):
            raise Refused(f"{why}; a step needs the verified graph, which runs the whole model")
        if plan.staged:
            raise Refused(f"{why}; staged builds (checkpoint steps or a packed source) are not supported")
        from ..intake.probe import _headers, _weight_map

        src = Path(plan.source)
        # planning a repo id reads its configs only: without the shards there is nothing to check before the build
        if any(src.glob("*.safetensors")):
            _coded(src, _headers(src, set(_weight_map(src).values())))

    def load(self, job: Any, device: torch.device) -> torch.nn.Module:  # noqa: D102, ARG002
        from accelerate import init_empty_weights

        from ..intake.loading import load_model

        with init_empty_weights(include_buffers=False):
            m = cast(
                "PreTrainedModel",
                load_model(
                    job.model,
                    weights=False,
                    dtype=job.dtype,
                    target=job.target,
                    images=getattr(job.plan, "calib_inputs", None) is not None,
                ),
            ).eval()
        layers = [lg.module for lg in job.plan.graph.layers]
        reader = _Reader(
            job.model,
            m,
            layers,
            job.dtype,
            job.plan.graph.extra.get("preserved_tensors", ()),
            job.plan.graph.extra.get("host_tensors", ()),
        )
        _drop(m)  # the model at rest, including buffers, is meta
        self._readers[m] = reader
        for name in layers:
            self._readers[m.get_submodule(name)] = reader
        return m

    def place(self, layer: torch.nn.Module, lg: LayerGraph, device: torch.device) -> None:  # noqa: D102
        reader = self._readers[layer]
        reader.load(lg.module, device)
        reader.restore(layer, lg.module, device)
        _loaded(layer, lg.module)

    def release(self, layer: torch.nn.Module) -> None:  # noqa: D102
        _drop(layer)

    @contextlib.contextmanager
    def outer(self, model: torch.nn.Module, graph: Any, device: torch.device) -> Iterator[None]:  # noqa: D102
        from .runner import recorders

        reader = self._readers[model]
        reader.load("", device)
        reader.model.tie_weights()
        with recorders(model, graph, {}):  # the layers are recorders: only the non-layer modules are seen
            reader.restore(model, "", device)
            _loaded(model, "the non-layer modules")
        try:
            yield
        finally:
            with recorders(model, graph, {}):
                _drop(model)


def _loaded(module: torch.nn.Module, what: str) -> None:
    """Check that every tensor of `module` was read, and freeze its parameters."""
    missing = [n for n, t in module.state_dict().items() if t.is_meta]
    if missing:
        raise RuntimeError(f"{what}: no tensors in the checkpoint for {missing[:3]}")
    for p in module.parameters():
        p.requires_grad_(False)


def _drop(module: torch.nn.Module) -> None:
    """Put every parameter and buffer of `module` on the meta device (`_Reader.restore` brings buffers back)."""
    for sub in module.modules():
        for name, p in sub._parameters.items():
            if p is not None and not p.is_meta:
                sub._parameters[name] = torch.nn.Parameter(
                    torch.empty_like(p, device="meta"), requires_grad=False
                )
        for name, b in sub._buffers.items():
            if b is not None and not b.is_meta:
                sub._buffers[name] = torch.empty_like(b, device="meta")


DISK = Disk()
