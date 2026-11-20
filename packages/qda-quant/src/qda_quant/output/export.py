"""Write a compressed-tensors ``pack-quantized`` checkpoint that stock vLLM loads, one layer at a time.

The engine calls `Writer.add_layer` as soon as a layer is solved (the layer can then leave the card),
and `Writer.finish` once. Tensor names follow the *source checkpoint*, not the in-memory module. The
fused-expert adapter splits canonical banks into per-expert serving tensors. A text model loaded from a
multimodal checkpoint keeps the checkpoint's prefixes. An architecture adapter may override that with a
`qda_quant.arch.ExportLayout` when the serving loader reads other names (gpt-oss: one tensor per expert), and
may rename a module tree to the engines' names (`ArchAdapter.serve_names`; Mixtral's ``mlp`` is
``block_sparse_moe`` there) for every tensor and source-derived ``ignore`` entry alike.

Compressed-tensors representation per quantized Linear ``[out, in]``:
``weight_packed`` int32 ``[out, in * b / 32]`` of signed codes, ``weight_scale`` bf16 ``[out, groups]``,
``weight_shape`` int64 ``[2]``, and for asymmetric formats ``weight_zero_point`` int32 packed along the
output axis ``[out * b / 32, groups]``. Value = ``(q - z) * s``.

The Writer reads quantization parameters only from the build's qparams dict (`BuildRun.qparams`, which
`LayerContext.qparams` is): module name -> ``{"codes", "scale", "zero_point"?, "format"}``. It never reads
module attributes.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import stat
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..arch import ModelGraph
from ..arch.derive import served_name
from ..format import Format
from ..intake.source_names import SourceNames
from ..intake.tokenizer import tokenizer_config_record
from ..step import LayerContext
from ..target import Target, merge_serve
from .ct_config import _CheckpointConfig, _experts_pattern, _ignore_matches, ct_quantization_config
from .generation import card_generation_config
from .pack import _FUSED, _pack_on, pack_module
from .remote_code import asset_configs, copy_remote_code
from .source_tensors import _DT, _SourceTensors

SHARD_BYTES = 4 << 30

#: ``build.json["serve_only"]`` of a checkpoint whose `ignore`-kept routed banks are stored one module per expert.
#: Stock transformers 5 opens such a checkpoint without an error and leaves every routed expert at its random
#: initialization (`docs/quantize.md`), so QDA marks it for the serving engines and `intake.load_model`
#: refuses it. Every float bank is stored that way: a source that stores it fused is split by
#: `Writer._spread_float_bank`.
_SERVE_ONLY = {
    "why": "a routed expert bank is kept float and stored one module per expert, which stock transformers loads "
    "as uninitialized weights without an error; serve this checkpoint with vLLM or SGLang",
}

SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "generation_config.json", "chat_template.jinja",
               "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
               "processor_config.json", "tekken.json")  # fmt: skip


def _without_native_model_code(cfg: dict) -> dict:
    """`cfg` without the remote config and model classes of a model type transformers implements itself.

    QDA loads such a model through transformers' own classes (`intake.loading.load_config` does not trust
    remote code), so the source's ``AutoConfig``/``AutoModel*`` entries name code this checkpoint was never
    built with, and engines load the type natively. Keeping them would also copy that code
    (`copy_remote_code`). Other entries (a processor's or tokenizer's own code) stay.
    """
    # transformers' own types only, not the ones `register_models` adds for QDA's adapters
    from transformers.models.auto.configuration_auto import (
        CONFIG_MAPPING_NAMES,  # pyright: ignore[reportPrivateImportUsage]
    )

    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict) or cfg.get("model_type") not in CONFIG_MAPPING_NAMES:
        return cfg
    kept = {k: v for k, v in auto_map.items() if k != "AutoConfig" and not k.startswith("AutoModel")}
    out = {k: v for k, v in cfg.items() if k != "auto_map"}
    return {**out, "auto_map": kept} if kept else out


def check_unchanged(layer: torch.nn.Module, rel: str, qp: dict[str, Any]) -> None:
    """Raise if the weight behind a qparams entry was replaced or changed in place after its codes were recorded.

    Layer steps' entries carry ``weight_check`` = (identity, `tensor_digest`) of the parameter at record time; a
    later step that writes the weight (steps write through ``.data``, which the autograd version counter does not
    see) would make the stored codes stale. Catches engine or step bugs only: the plan refuses recipes that would
    do this (`engine.planning.check_order`).
    """
    from ..intake.fingerprint import tensor_digest

    if "weight_check" not in qp:
        return
    try:
        p = layer.get_parameter(rel + ".weight")
    except AttributeError:  # a fused expert parameter: the entry names the parameter itself
        p = layer.get_parameter(rel)
    if (id(p), tensor_digest(p)) != qp["weight_check"]:
        raise RuntimeError(f"{rel}: the weight changed after its codes were recorded; the codes are stale")


class Writer(_SourceTensors, _CheckpointConfig):
    """Streams a quantized checkpoint to `out_dir`.

    Args:
        out_dir: output directory (created).
        source: local directory of the original checkpoint (``*.safetensors`` + ``config.json``).
        graph: the model graph.
        fmt: stored format of the body.
        target: serving target (recorded in ``build.json``).
        dtype: the build dtype; a source weight no step touched decodes into it.
        formats: unit key -> the format a ``format_overrides`` rule gives it; each such format gets its own
            ``config_group``, so the checkpoint serves those modules differently from the body.
    """

    def __init__(
        self,
        out_dir: str | Path,
        source: str | Path,
        graph: ModelGraph,
        fmt: Format,
        target: Target,
        dtype: torch.dtype = torch.bfloat16,
        formats: dict[str, Format] | None = None,
    ):
        self.out = Path(out_dir)
        self.src = Path(source)
        #: the tokenizer_config.json decision (checked before any tensor is written; refused if it cannot run)
        self.tokenizer_config = tokenizer_config_record(self.src)
        self.out.mkdir(parents=True, exist_ok=True)
        self.graph, self.fmt, self.target = graph, Format.parse(fmt), target
        self.formats = dict(formats or {})
        self._format_names = {self.fmt.name} | {f.name for f in self.formats.values()}
        self.dtype = dtype
        self.names = SourceNames(self.src)
        self.where, self.meta, self.config = self.names.where, self.names.meta, self.names.config
        self.preserved = {self.names.tensor(k) for k in graph.extra.get("preserved_tensors", ())}
        self.quantized: set[str] = set()  # source module names written quantized
        self.decoded: set[str] = set()  # source module names written as decoded float weights
        #: source module names decoded in `finish`: weights no step touched (a draft's layers)
        self.kept_decoded: set[str] = set()
        self.replaced: set[str] = set()  # source codec parts superseded by a packed or decoded weight
        self.covered: set[str] = set()  # tensor names written from the in-memory model
        #: stored dotted path -> the path the serving engines build that module tree under
        self.renames = dict(graph.extra.get("serve_names", {}))
        #: source-only Linear -> the module whose format it is stored in (`ArchAdapter.pack_like`)
        self.pack_like: dict[str, str] = dict(graph.extra.get("pack_like", {}))
        #: per engine, the further renames it applies to a served name when it reads quantization_config
        self.config_names: dict[str, dict[str, str]] = dict(graph.extra.get("config_names", {}))
        self.written: set[str] = set()  # stored tensor names copied or decoded in `finish`
        self.index: dict[str, str] = {}
        self.head_fmt: Format | None = None
        self.head_shape: tuple[int, int] = (0, 0)  # (out, in) of a head stored in its own format
        #: in-memory expert bank -> the served paths its packed experts are written under (``<root>.<e>.<proj>``)
        self.bank_roots: dict[str, set[str]] = {}
        self._buf: dict[str, torch.Tensor] = {}
        self._bytes = 0
        self._cuda = False
        self._n = 0
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._err: list[BaseException] = []
        self._mode = _file_mode(self.out)
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    @torch.no_grad()
    def add_layer(self, ctx: LayerContext) -> None:
        """Pack the layer's entries of ``ctx.qparams``, queue them and the layer's other tensors.

        An entry is packed on its own device, except one the solve left on the host beside a layer on a card (an
        expert bank solved in chunks): that one is packed on the card, which is many times faster.
        """
        from ..solve.units import bank_module

        card = ctx.device if ctx.device is not None and ctx.device.type == "cuda" else None
        banks = [bank_module(bank) for bank in ctx.graph.banks]
        out: dict[str, torch.Tensor] = {}
        done_params: set[str] = set()
        prefix = ctx.graph.module + "."
        for name, qp in ctx.qparams.items():
            if not name.startswith(prefix):
                continue
            if qp["format"] not in self._format_names:
                raise ValueError(
                    f"{name}: stored as {qp['format']}, neither the body ({self.fmt.name}) nor a "
                    f"format_overrides format ({sorted(self._format_names - {self.fmt.name})})"
                )
            check_unchanged(ctx.layer, name[len(prefix) :], qp)
            done_params.add(name)
            pieces, parts = self._layout(name, qp["codes"].shape)
            for mod, sl, names in pieces:
                packed = self._pack_module(mod, qp, sl, names, card)
                out.update({self._served(k): v for k, v in packed.items()})
                self.quantized.add(mod)
                bank = next((b for b in banks if name.startswith(b + ".")), None)
                if bank is not None:
                    # a stacked piece is `<root>.gate_up_proj`, an expert's `<root>.<e>.<proj>`
                    root = mod.rpartition(".")[0] if mod.endswith(_FUSED) else mod.rsplit(".", 2)[0]
                    self.bank_roots.setdefault(bank, set()).add(self._served(root))
            self.replaced.update(parts)
        float_banks = tuple(f"{root}." for root in self.graph.extra.get("float_banks") or ())
        for k, v in ctx.layer.state_dict().items():
            full = f"{ctx.graph.module}.{k}"
            mod, _, leaf = full.rpartition(".")
            if full in done_params or (mod in done_params and leaf != "bias"):
                continue  # packed above (with a packed source's scale offsets); only its bias is written as it is
            if full.startswith(float_banks):
                # an `ignore`-kept float bank the source stores fused is spread over per-expert names here; a
                # bank the source stores per expert is copied in `finish` under those names
                out.update(self._spread_float_bank(full, v))
                continue
            key = self.names.tensor(full)
            if key is None:
                if k.endswith(("inv_freq", "original_inv_freq")):
                    continue
                pieces = self.names.split(full, v)
                if pieces is None:
                    raise KeyError(f"{full}: no tensor of that name in the source checkpoint")
                for part, piece in pieces.items():
                    out[self._served(part)] = self._stored(part, piece)
                    self.covered.add(part)
                continue
            out[self._served(key)] = self._stored(key, v)
            self.covered.add(key)
        self._push(out)

    @torch.no_grad()
    def add_model(
        self, model: torch.nn.Module, qparams: dict[str, dict[str, Any]], *, changed: bool = True
    ) -> None:
        """Write the embedding, final norm and head from the in-memory model and the build's `qparams`.

        Model steps such as rotations change them, so `finish` must not copy the source's. Call once, before
        `finish`. A module with a `qparams` entry (the head of `steps.head`) is packed in the entry's format,
        under its own config group. With `changed` false no model step ran, so the modules are read from the
        source checkpoint instead of copied off the card.
        """
        out: dict[str, torch.Tensor] = {}
        for mod in (self.graph.embed, self.graph.final_norm, self.graph.lm_head):
            if not mod:
                continue
            qp = qparams.get(mod)
            if qp is not None:
                if mod != self.graph.lm_head:
                    raise NotImplementedError(f"{mod}: only the head is stored packed outside the layers")
                src = self.names.module(mod)
                if src is None:
                    raise KeyError(f"{mod}: no module of that name in the source checkpoint")
                out.update({self._served(k): v for k, v in self._pack_module(src, qp).items()})
                self.quantized.add(src)
                self.replaced.update(self._parts(src + ".weight"))
                self.head_fmt = Format.parse(qp["format"])
                self.head_shape = (int(qp["codes"].shape[0]), int(qp["codes"].shape[1]))
                continue
            for k, v in model.get_submodule(mod).state_dict().items():
                key = self.names.tensor(f"{mod}.{k}")
                if key is None or key in self.covered:
                    continue  # a tied head has no tensor of its own
                if not changed:  # no model step ran: read the source, not the card
                    v = self._source_weight(key, v.dtype)
                out[self._served(key)] = self._stored(key, v)
                self.covered.add(key)
        self._push(out)

    @torch.no_grad()
    def add_source_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Write source tensors a model step transformed outside the graph (a rotated vision merger).

        Keys are model tensor names; the source-name map resolves them, and `finish` then copies neither them
        nor their stored parts.
        """
        out: dict[str, torch.Tensor] = {}
        for name, v in tensors.items():
            key = self.names.tensor(name)
            if key is None:
                raise KeyError(f"{name}: no tensor of that name in the source checkpoint")
            out[self._served(key)] = self._stored(key, v)
            self.covered.add(key)
        self._push(out)

    def _served(self, name: str) -> str:
        """`name` under the graph adapter's declared ``serve_names`` (packed parts and scales alike)."""
        return served_name(name, self.renames)

    def _pack_module(self, mod, qp, sl=(), names=None, card=None):  # noqa: ANN001, ANN201
        """Pack one stored piece; exporters may use another layout for the same qparams."""
        if card is None or qp["codes"].is_cuda:
            return pack_module(mod, qp, sl, names)
        return _pack_on(card, mod, qp, sl, names)

    def _output_config(self, recipe: Any) -> dict:
        """Source model settings normalized by its adapter, with this exporter's quantization config."""
        from ..arch.adapters import ADAPTERS

        cfg = _without_native_model_code(dict(self.config))
        adapter = ADAPTERS[self.graph.adapter] if self.graph.adapter else None
        normalize = getattr(adapter, "export_config", None)
        if normalize is not None:
            cfg = normalize(cfg)
        cfg["quantization_config"] = self.quantization_config(recipe)
        return cfg

    def _spread_float_bank(self, name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """An `ignore`-kept float bank the checkpoint stores fused, under the per-expert names engines load.

        Stock SGLang 0.5.23's Qwen3.5 loader keeps one ``is_fused_expert`` flag for the whole checkpoint and
        switches to its fused-expert table for good at the first ``.experts.gate_up_proj``/``.experts.down_proj``
        name, so every per-expert tensor stored after that one is skipped with a warning and the model serves
        garbage (measured on a 35B row: 6,912 skipped tensors, layers 37-39). Splitting the stored tensor keeps
        its values, source dtype, expert order and projections, and leaves no name that flips the flag.

        Returns the stored tensors of a bank written here, or nothing for a bank the source stores per expert
        (no tensor of this fused name, copied in `finish` under its own names).
        """
        from ..arch.adapters.fused_experts import per_expert_layout

        key = self.names.tensor(name)
        # the source name carries the container the engines read (`model.language_model.` on the VL models)
        # and the source tensor carries the stored dtype: a source F32 bank under a bf16 build must stay F32
        if key is None:  # the source stores this bank per expert; `finish` copies it under its own names
            return {}
        layout = per_expert_layout(key, tensor.shape)
        if layout is None:
            return {}
        self.covered.add(key)
        stored = self._stored(key, tensor)
        return {self._served(f"{mod}.weight"): stored[sl].contiguous() for mod, sl, _fields in layout.pieces}

    def _float_banks_per_expert(self) -> list[str]:
        """The roots the `ignore`-kept routed banks (`graph.extra["float_banks"]`) are stored per expert under.

        A bank's experts are stored per expert when their names carry an index between the bank and the
        projection (``<bank>.0.gate_proj``): the source naming of Qwen3-MoE and Mixtral, and the one this writer
        gives a bank whose source is fused (`_spread_float_bank`), so every float bank a build stores is one.
        The roots recorded are the ones written, not the graph's: a bank split from a fused source is stored
        under the source's container (``model.language_model.`` on the VL models), which `_stored_roots`
        follows.
        """
        roots = {
            stored
            for root in self.graph.extra.get("float_banks") or ()
            for stored in self._stored_roots(root)
            if any(re.match(rf"{re.escape(stored)}\.\d+\.", k) for k in self.index)
        }
        return sorted(roots)

    def _stored_roots(self, root: str) -> set[str]:
        """The containers the experts of bank `root` may be stored under: runtime, served, and source."""
        roots = {root, self._served(root)}
        for leaf in ("gate_up_proj", "down_proj"):
            key = self.names.tensor(f"{root}.{leaf}")
            if key is not None:
                roots.add(self._served(key.removesuffix(f".{leaf}")))
        return roots

    def _push(self, tensors: dict[str, torch.Tensor]) -> None:
        if self._err:
            raise RuntimeError("checkpoint writer failed") from self._err[0]
        for k, v in tensors.items():
            if k in self._buf or k in self.index:  # a serving rename must not merge two stored tensors
                raise RuntimeError(f"{k}: two stored tensors are written under one name")
            self._cuda |= v.is_cuda
            self._buf[k] = v.to("cpu", non_blocking=True)
            self._bytes += v.numel() * v.element_size()
        if self._bytes >= SHARD_BYTES:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        if self._cuda:  # non-blocking copies from the card must land before the shard is written
            torch.cuda.synchronize()
            self._cuda = False
        self._n += 1
        fname = f"model-{self._n:05d}.safetensors"
        for k in self._buf:
            self.index[k] = fname
        self._q.put((fname, self._buf))
        self._buf, self._bytes = {}, 0

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                save_file(item[1], str(self.out / item[0]), metadata={"format": "pt"})
                (self.out / item[0]).chmod(self._mode)  # save_file creates 0600; match the other files
            except BaseException as e:  # noqa: BLE001  (re-raised in the caller's thread)
                self._err.append(e)

    def finish(self, config: Any, recipe: Any, build_info: dict) -> Path:  # noqa: ARG002
        """Copy every untouched source tensor, then write the index, config and build info.

        Every source tensor is replaced through its codec, written from the model, decoded or copied. A packed
        or decoded weight retires its stored parts, including scales. A quantized module's bias is copied when
        nothing wrote it; any unexplained tensor of that module is an error.

        The recipe is recorded by `build`, in ``resolved.yaml`` with the rest of the build's config.

        Args:
            config: unused when the source has a ``config.json`` (it is copied, with ``quantization_config``).
            recipe: the `Recipe` (its GPTQ act order goes into ``quantization_config``).
            build_info: timings, versions, card count (written as ``build.json``).

        Returns:
            The checkpoint directory.
        """
        self._pack_source_only()
        decoded = self._decode_kept()
        by_file: dict[str, list[str]] = {}
        for k, f in self.where.items():
            if k in self.covered or k in self.replaced:
                continue
            mod, _, leaf = k.rpartition(".")
            if mod in self.quantized and leaf != "bias":
                raise ValueError(f"{k}: the export cannot keep this tensor of the quantized module {mod}")
            by_file.setdefault(f, []).append(k)
        for f, keys in sorted(by_file.items()):
            with safe_open(self.src / f, "pt") as h:
                self._push({self._served(k): h.get_tensor(k) for k in keys})
            self.written.update(keys)
        for k in sorted(decoded):
            self._push({self._served(k): self._source_weight(k, self.dtype)})
            self.written.add(k)
        self._flush()
        self._q.put(None)
        self._t.join()
        if self._err:
            raise RuntimeError("checkpoint writer failed") from self._err[0]
        done = self.covered | self.written | self.replaced
        lost = [k for k in self.where if k not in done]
        if lost:
            raise RuntimeError(f"{len(lost)} source tensors were neither packed nor written: {lost[:5]}")
        total = sum((self.out / f).stat().st_size for f in set(self.index.values()))
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": dict(sorted(self.index.items()))}, indent=2
            )
        )
        cfg = self._output_config(recipe)
        (self.out / "config.json").write_text(json.dumps(cfg, indent=2))
        for f in SMALL_FILES:
            if (self.src / f).exists():
                shutil.copy2(self.src / f, self.out / f)
        card = (
            None if (self.src / "generation_config.json").exists() else card_generation_config(self.src, cfg)
        )
        if card:
            # the original ships no sampling defaults; its released card's row stands in, so engines serve alike
            (self.out / "generation_config.json").write_text(json.dumps(card[0], indent=2) + "\n")
            build_info = {**build_info, "generation_config": {"from_card": card[1]}}
        if self.tokenizer_config.get("written", "unchanged") != "unchanged":
            # the copied config declares a class that does not reproduce the source's own tokenizer.json
            path = self.out / "tokenizer_config.json"
            declared = json.loads(path.read_text())
            declared["tokenizer_class"] = self.tokenizer_config["written"]
            path.write_text(json.dumps(declared, indent=2) + "\n")
        if self.tokenizer_config:
            build_info = {**build_info, "tokenizer_config": self.tokenizer_config}
        copy_remote_code(self.src, self.out, [cfg, *asset_configs(self.src, SMALL_FILES)])
        if self.kept_decoded:  # what this build decoded instead of keeping the source's native precision
            build_info = {
                **build_info,
                "kept_decoded": {
                    "dtype": str(self.dtype).removeprefix("torch."),
                    "modules": sorted(self.kept_decoded),
                },
            }
        if per_expert := self._float_banks_per_expert():
            build_info = {**build_info, "serve_only": _SERVE_ONLY | {"float_banks": per_expert}}
        (self.out / "build.json").write_text(
            json.dumps({**build_info, "serve": self.serve()}, indent=2, default=str)
        )
        return self.out


def _file_mode(where: Path) -> int:
    """Mode a new regular file gets in `where` (what ``open(..., "w")`` would give under the process umask).

    Measured on a probe file rather than read with ``os.umask``, which would change the umask process-wide for a
    moment and race with other threads.
    """
    probe = where / f".mode-probe-{uuid.uuid4().hex}"  # unique even after a killed run left one behind
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        return stat.S_IMODE(os.fstat(fd).st_mode)
    finally:
        os.close(fd)
        probe.unlink()


# Re-exported for the callers that import them from here.
__all__ = [
    "SHARD_BYTES", "SMALL_FILES", "Writer", "ct_quantization_config", "merge_serve", "pack_module",
    "_DT", "_experts_pattern", "_file_mode", "_ignore_matches",
]  # fmt: skip
