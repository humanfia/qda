"""Stream the build's codes and changed values into a GGUF, the template's values for the rest, and the record.

The template (llama.cpp's own bf16 conversion of the source) fixes every name, row order and convention the
file must follow. A step that changes a tensor (AWQ folding scales into norms, an offline rotation) changes
it in the *source's* basis, so the writer derives, per tensor, how the converter maps one basis to the other
and applies it; a basis it cannot name is refused rather than guessed.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from ..arch.adapters import export_layout
from ..intake.fingerprint import file_digest
from ..intake.source_names import SourceNames
from ..output.export import check_unchanged
from ..step import LayerContext, Refused
from . import file, rule
from . import imatrix as imat
from . import kernel as kern
from .columns import ORDER
from .iq4 import IQ4NL_FORMAT, IQ4XS_FORMAT, iq4nl_bytes, iq4xs_bytes
from .kernel import quantize_q4k, quantize_q5k, quantize_q6k
from .legacy import quantize_q5_0, quantize_q8_0
from .llama_cpp import ENV, PINNED, checkout_command, commit_of, converter, locate
from .packing import q4k_bytes, q5k_bytes, q6k_bytes, row_order
from .q3k import Q3K_FORMAT, q3k_bytes
from .q4k import Q4K_FORMAT
from .q5k import Q5K_FORMAT
from .q6k import Q6K_FORMAT
from .tensor_types import TensorTypeRule, dropped_head_hint, resolve_types

OUT_NAME = "model-Q4_K_M.gguf"
#: The GGUF type that holds each solved format's codes, and the encoder of its records.
_SOLVED = {
    Q4K_FORMAT: (rule.Q4_K, q4k_bytes),
    Q5K_FORMAT: (rule.Q5_K, q5k_bytes),
    Q6K_FORMAT: (rule.Q6_K, q6k_bytes),
    IQ4NL_FORMAT: (rule.IQ4_NL, iq4nl_bytes),
    IQ4XS_FORMAT: (rule.IQ4_XS, iq4xs_bytes),
    Q3K_FORMAT: (rule.Q3_K, q3k_bytes),
}
_TEMPLATE = ".template-bf16.gguf"
_PART_VALUES = 1 << 22  #: values one thread takes at a time in `GGUFWriter._fill`: 16 MB of fp32
#: types the mix falls a tensor back to when its rows are not whole super-blocks (`rule`)
_FALLBACK_OF = frozenset(rule._FALLBACK.values())
#: stored types that are not quantized at all, so a tensor converted to one is not rounded from anything
_FLOAT = frozenset({rule.F32, rule.F16, rule.BF16})


#: llama.cpp's converter renames an MTP head's own tensors before its tensor map runs, and names the four
#: head outputs from the block itself (`_QwenMtpMixin.filter_tensors`); the installed `gguf` package may
#: predate those patterns, so `GGUFWriter._block_name` names them from the template.
_HEAD_LEAF = {
    "fc": "eh_proj",
    "pre_fc_norm_embedding": "enorm",
    "pre_fc_norm_hidden": "hnorm",
    "norm": "shared_head_norm",
}
_MTP_LAYER = re.compile(r"(?:model\.)?mtp\.layers\.(\d+)\.(.+)")
_MTP_HEAD = re.compile(r"(?:model\.)?mtp\.(fc|pre_fc_norm_embedding|pre_fc_norm_hidden|norm)\.(.+)")


def _round_to(kind: int, values: torch.Tensor) -> torch.Tensor:
    """`values` as a tensor of ggml type `kind` stores them: F32 keeps every fp32 value, the rest round.

    The converter computes a shifted norm in fp32 and casts it on its way into the template, so reproducing
    the template means casting the same way; rounding an F32 tensor to bf16 first would lose the low bits a
    Qwen norm's ``+1`` carries.
    """
    if kind == rule.F32:
        return values.float()
    if kind == rule.F16:
        return values.to(torch.float16).float()
    return values.to(torch.bfloat16).float()


def _weighted(kind: int, values: torch.Tensor, row: int, weights: np.ndarray | None) -> bytes:
    """Records of ``values`` as llama.cpp's ``ggml_quantize_chunk(kind, ..., weights)`` writes them on x86.

    llama.cpp's Q8_0 ignores the weights and its Q5_0 without weights is the plain fit; both cast inf to int8 in
    blocks of tiny weights, which aarch64's libggml writes as 0xff, so `legacy` encodes them here instead.
    """
    if kind == rule.Q8_0:
        return quantize_q8_0(values)
    if kind == rule.Q5_0 and weights is None:
        return quantize_q5_0(values)
    return kern.quantize_weighted_chunk(
        kind, values, row, None if weights is None else torch.from_numpy(weights)
    )


class _Replaced:
    """A tensor a model step changed, held as the build has it, plus the converter's basis into the template.

    Args:
        value: the build's tensor, in the source checkpoint's layout (``[rows, width]`` or one row).
        offset: the constant the converter adds to the source's values (Qwen's ``norm.weight + 1``), else 0.
        order: template row ``i`` is source row ``order[i]`` where the converter permutes rows, else None.
        kind: the template tensor's ggml type, whose storage rounds the written values.
    """

    def __init__(
        self,
        value: torch.Tensor,
        offset: float = 0.0,
        order: torch.Tensor | None = None,
        kind: int = rule.BF16,
    ):
        self.value, self.offset, self.order, self.kind = value, offset, order, kind
        self._selected: torch.Tensor | None = None

    def rows(self, start: int, stop: int) -> torch.Tensor:
        """Template rows ``start:stop`` as the template stores them: the build's values in its basis."""
        v = self.value
        if self.order is not None:
            if self._selected is None:
                self._selected = self.value.index_select(0, self.order)
            v = self._selected
        return _round_to(self.kind, v[start:stop].float().cpu() + self.offset)

    @property
    def basis(self) -> str:
        """What was done to the build's value, for ``build.json``."""
        if self.order is not None:
            return "the build's value in the converter's row order"
        if self.offset:
            return f"the build's value plus the converter's offset {self.offset:+g}"
        return "the build's value in the converter's basis"


def _overridden(types: dict[str, int], rules: list[TensorTypeRule] | None) -> set[str]:
    """Tensors whose type a `TensorTypeRule` chose, so a report can say so rather than blame the mix."""
    named: set[str] = set()
    for rule_ in rules or []:
        pattern = rule_.pattern
        if pattern.startswith("re:"):
            named |= {n for n in types if re.compile(pattern[3:]).match(n)}
        elif pattern in types:
            named.add(pattern)
    return named


def make_template(llama_cpp: str | Path, source: str | Path, out: str | Path) -> Path:
    """Run the checkout's ``convert_hf_to_gguf.py`` on `source`: a bf16 GGUF at `out`.

    Raises:
        Refused: `llama_cpp` is not a llama.cpp checkout.
    """
    script = converter(llama_cpp)
    cmd = [sys.executable, str(script), str(source), "--outtype", "bf16", "--outfile", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"convert_hf_to_gguf.py failed:\n{r.stderr[-4000:]}")
    return Path(out)


def _refuse_unless_bf16(tp: file.Template) -> None:
    """Refuse a template that stores f16: f16 rounds values below 2^-14, so the bytes would differ from bf16.

    Raises:
        Refused: a tensor is f16.
    """
    for name, t in tp.tensors.items():
        if t.tensor_type == rule.F16:
            raise Refused(f"gguf: template tensor {name} is f16; convert with --outtype bf16")


class GGUFWriter:
    """Streams a ``q4k`` build into a Q4_K_M GGUF; the engine's writer protocol.

    Args:
        out_dir: output directory (created); the file is ``OUT/model-Q4_K_M.gguf``.
        source: local directory of the original checkpoint.
        graph: the model graph.
        fmt: stored format (must be ``w4a16-asym-g32``).
        target: unused (a GGUF is served by llama.cpp).
        llama_cpp: a llama.cpp checkout; its converter makes the template unless `template` is given. None takes
            ``$QDA_LLAMA_CPP``, and neither is needed when `template` is given.
        workers: threads of llama.cpp's quantizers (`kernel`); None lets the kernel choose (`default_workers`). The
            bytes do not depend on it.
        template: a bf16 GGUF of `source` the user already has; the converter is not run and the file is not
            deleted.
        tensor_types: ordered type overrides after the default Q4_K_M rule; the last match wins.
        mtp: write the template's MTP/NextN head. False leaves the head's blocks out of the file, as a
            conversion with ``--no-mtp`` does; a build that changed one of their tensors is then refused.
    """

    def __init__(  # noqa: ANN001
        self,
        out_dir,
        source,
        graph,
        fmt,  # noqa: ARG002
        target,  # noqa: ARG002
        *,
        llama_cpp: str | Path | None,
        workers: int | None = None,
        template: str | Path | None = None,
        tensor_types: list[TensorTypeRule] | None = None,
        imatrix: str | Path | None = None,
        mtp: bool = True,
    ):
        import gguf  # the PyPI package (extra gguf); pyright without it resolves this directory instead

        self.workers = workers
        self.tensor_types = tensor_types or []
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.src = Path(source)
        root = locate(llama_cpp)
        self.own_template = template is None
        if template is None:
            if root is None:
                raise Refused(
                    "gguf: no llama.cpp checkout for convert_hf_to_gguf.py; set `llama_cpp` or "
                    f"${ENV}, or give a bf16 GGUF with `template` ({checkout_command()})"
                )
            self.commit = commit_of(root)
            self.template_path = make_template(root, self.src, self.out / _TEMPLATE)
        else:
            self.commit = commit_of(root) if root is not None else None
            self.template_path = Path(template)
        self.tp = file.Template(self.template_path)
        _refuse_unless_bf16(self.tp)
        if not mtp:
            self.tp = self.tp.without_mtp()
        try:
            self.types = resolve_types(self.tp, self.tensor_types)
        except Refused as e:
            if mtp:
                raise
            raise Refused(
                dropped_head_hint(str(e), self.tensor_types, file.Template(self.template_path), self.tp)
            ) from None
        arch = next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == self.tp.arch)  # pyright: ignore[reportAttributeAccessIssue]
        self.names = gguf.get_tensor_name_map(arch, self.tp.n_layer)  # pyright: ignore[reportAttributeAccessIssue]
        self.where = _weight_map(self.src)
        self.graph = graph
        self.source_names: SourceNames | None = None
        self.overridden = _overridden(self.types, self.tensor_types)
        self.imatrix_path = Path(imatrix) if imatrix else None
        self.unweighted: set[str] = set()  # in an imatrix build, the tensors it does not cover
        self.from_codes: set[str] = set()  # written from the build's own codes
        self.replaced: dict[
            str, _Replaced
        ] = {}  # changed by a step: written from the build, not the template
        self.skipped: list[str] = []  # source tensors a step changed that this file does not hold
        self.dropped: dict[str, str] = {}  # solved tensor -> module; stored as not Q4_K
        self.head = graph.lm_head
        self.sink = file.Sink(self.out / OUT_NAME, self.tp, self.types)
        self.imatrix = self._load_imatrix()

    def _rounded(self) -> dict[str, int]:
        """Tensors `_fill` encodes from the template: the ones an importance matrix can still weigh.

        Called before the layer pass as well, where `from_codes` is empty and every quantized tensor is still a
        candidate. That is the right set to validate a matrix against: a build must not get half way and then
        find the matrix cannot weigh a tensor it has already started writing.
        """
        return {
            n: t
            for n, t in self.types.items()
            if n not in self.from_codes and n not in self.replaced and t != self.tp.tensors[n].tensor_type
        }

    def _load_imatrix(self) -> dict[str, Any] | None:
        """The importance matrix this build fits with, checked against the template it will quantize.

        Raises:
            Refused: no llama.cpp library to fit with, a tensor the matrix cannot weigh, or an entry that
                does not belong to this template.
        """
        if self.imatrix_path is None:
            return None
        if not kern.has_weighted():
            raise Refused(
                f"gguf: an importance matrix needs llama.cpp's own fit; set `libggml` to a built "
                f"libggml-base (the Python port has no weighted fit). {self.imatrix_path} not used"
            )
        table = imat.load(self.imatrix_path)
        scope = sorted(self._rounded())
        if not any(n in table for n in scope):
            raise Refused(
                f"gguf: {self.imatrix_path} covers none of the {len(scope)} tensors this build would fit "
                "with it; it was measured on another model or template"
            )
        self.unweighted = {n for n in scope if n not in table}
        for name in scope:
            imat.column_weights(table, name, self.tp.tensors[name].shape)
        return table

    def gguf_name(self, module: str) -> str:
        """The template tensor of a source module's weight."""
        name = self.names.get_name(module + ".weight", try_suffixes=(".weight",))
        if name is None or name not in self.types:
            raise KeyError(f"{module}: no tensor in the GGUF template")
        return name

    def _source_of(self, hf: str) -> tuple[str, Path]:
        """The source checkpoint's tensor name and file for an in-memory tensor name.

        Raises:
            Refused: the checkpoint has no tensor of that name.
        """
        if self.source_names is None:
            self.source_names = SourceNames(self.src)
        key = self.source_names.tensor(hf)
        if key is None:
            raise Refused(f"gguf: {hf}: no tensor in the source checkpoint")
        return key, self.src / self.where[key]

    def _replace(
        self,
        name: str,
        hf: str,
        value: torch.Tensor,
        *,
        parts: bool = False,
        columns: torch.Tensor | None = None,
    ) -> None:
        """Write template tensor `name` from the build's `value` of `hf`, when a step changed it.

        The converter's basis (the same values, a constant offset, or a row permutation) is derived from the
        template against the *source* checkpoint, never from the build's value, so a step's change cannot
        disguise it. A tensor the build left as the source has it keeps the template's bytes. A changed tensor
        whose basis is none of the three is refused by name: a writer that cannot name the basis it needs does
        not guess one.

        Args:
            name: the template's tensor.
            hf: the in-memory name the value belongs to (resolved to the source's name).
            value: the build's tensor, in the source checkpoint's layout.
            parts: read in ``_PART_VALUES``-value parts (an embedding is too large to compare whole); only a
                constant offset is recognised that way, and a row permutation is refused rather than guessed.
            columns: the converter's column map when the layer pass has already permuted the module's weight
                (GDN's output projection), so the build's value is in the template's column order and the
                *source* is the side that must be permuted for the comparison.

        Raises:
            Refused: the value's size differs from the tensor's, or the converter's basis cannot be named.
        """
        rows, width = self.tp.rows(name), self.tp.tensors[name].shape[0]
        if value.numel() != rows * width:
            raise Refused(
                f"gguf: {name}: {hf} holds {value.numel()} values; the template tensor holds {rows * width}"
            )
        if columns is not None and columns.numel() != width:
            raise Refused(
                f"gguf: {name}: the converter's column order spans {columns.numel()} values; the template "
                f"tensor is {width} wide"
            )
        kind = self.tp.tensors[name].tensor_type
        build = value.detach().reshape(rows, width)
        key, path = self._source_of(hf)
        changed = False
        shift = 0.0
        whole: tuple[torch.Tensor, torch.Tensor] | None = None
        for start, stop, template, source in self._chunks(name, key, path, parts=parts, columns=columns):
            changed = changed or not torch.equal(build[start:stop].to(torch.bfloat16).float().cpu(), source)
            if start == 0:
                delta = (template - source).double()
                shift = float(delta.median()) if delta.numel() else 0.0
            whole = (template, source)
        if not changed:
            return
        for offset in (0.0, float(round(shift)), shift):
            if self._matches(name, key, path, offset, parts=parts, columns=columns):
                # the layer pass restores a permuted projection's own column order on its way out, and the
                # file is filled after that, so this value has to be held rather than viewed
                self.replaced[name] = _Replaced(
                    build.clone() if columns is not None else build, offset, kind=kind
                )
                return
        if parts:
            raise Refused(
                f"gguf: {name}: the converter's map from {key} is not a constant offset, and a tensor read in "
                "parts cannot be matched row by row; the exporter will not write the build's changed value in "
                "a basis it cannot name"
            )
        assert whole is not None
        held = build.clone() if columns is not None else build
        self.replaced[name] = _Replaced(held, 0.0, row_order(whole[0], whole[1], name), kind=kind)

    def _chunks(  # noqa: ANN202
        self, name: str, key: str, path: Path, *, parts: bool, columns: torch.Tensor | None = None
    ):
        """Yield ``(start, stop, template, source)`` over template tensor `name` and source tensor `key`.

        `parts` reads ``_PART_VALUES`` values at a time (an embedding is too large to hold whole); otherwise the
        tensor is one chunk. Source values are bf16-exact fp32 in the template's ``[rows, width]`` shape, in the
        template's column order when `columns` says the layer pass permuted the module's weight.
        """
        rows, width = self.tp.rows(name), self.tp.tensors[name].shape[0]
        step = max(1, _PART_VALUES // width) if parts else rows
        with safe_open(path, "pt") as h:
            stored = h.get_slice(key) if parts else None
            flat = None if parts else h.get_tensor(key).reshape(rows, width)
            scalar = stored is not None and len(stored.get_shape()) == 1  # a norm: its one axis is the width
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                if stored is not None:
                    source = stored[:] if scalar else stored[start:stop]
                else:
                    assert flat is not None
                    source = flat[start:stop]
                source = source.reshape(-1, width).to(torch.bfloat16).float()
                if columns is not None:
                    source = source.index_select(1, columns)
                yield start, stop, self.tp.values(name, start, stop), source

    def _matches(
        self,
        name: str,
        key: str,
        path: Path,
        offset: float,
        *,
        parts: bool,
        columns: torch.Tensor | None = None,
    ) -> bool:
        """True when the template is exactly the source plus `offset`, as the template's type stores it.

        The converter rounds the shifted value on its way into the template, so "constant offset" is checked by
        reproducing the template from the source, not by comparing per-element differences. The rounding is
        the template tensor's own: F32 keeps fp32, so a Qwen norm's `+1` matches exactly and a bf16-rounded
        sum does not.
        """
        kind = self.tp.tensors[name].tensor_type
        return all(
            torch.equal(template, _round_to(kind, source + offset))
            for _, _, template, source in self._chunks(name, key, path, parts=parts, columns=columns)
        )

    def _block_name(self, hf: str) -> str | None:
        """The template tensor of a declared source block's tensor (an MTP head the model class drops).

        The converter renames ``mtp.*`` to the model's own layer names and lets its tensor map name them;
        the installed `gguf` package may predate the four head patterns, so those are named from the
        template's own ``blk.<first>.nextn.*`` tensors, `first` being the block the file says its head starts
        at. A name neither route produces is None, and the caller refuses it by name.
        """
        name = self.names.get_name(hf, try_suffixes=(".weight", ".bias"))
        if name is not None and name in self.types:
            return name
        first = self._nextn_block()
        if first is None:
            return None
        if m := _MTP_LAYER.fullmatch(hf):
            # the file stores the head's layers after its decoder blocks, and names them through the converter's
            # map, whose keys are `model.layers.<index>` whatever root the checkpoint nests its text model under
            mapped = self.names.get_name(
                f"model.layers.{first + int(m[1])}.{m[2]}", try_suffixes=(".weight", ".bias")
            )
            return mapped if mapped in self.types else None
        if m := _MTP_HEAD.fullmatch(hf):
            leaf = _HEAD_LEAF[m[1]]
            named = f"blk.{first}.nextn.{leaf}.{m[2]}"
            return named if named in self.types else None
        return None

    def _nextn_block(self) -> int | None:
        """The template block the head's first layer sits in, from the file's own NextN count.

        A file that carries a head stores it after its decoder blocks and says how many blocks it has
        (`<arch>.nextn_predict_layers`, the key llama.cpp reads to find them). A file without the key has no
        head block -- llama.cpp reads its last blocks as decoder layers -- so there is nothing a head's tensor
        could be written to and the caller refuses it by name.
        """
        nextn = self.tp.number(f"{self.tp.arch}.nextn_predict_layers", 0)
        return self.tp.n_layer - nextn if nextn > 0 else None

    def _add(
        self, qparams: dict[str, dict[str, Any]], prefix: str, layer: torch.nn.Module | None = None
    ) -> None:
        if self.source_names is None:
            self.source_names = SourceNames(self.src)
        for mod, qp in qparams.items():
            if not mod.startswith(prefix):
                continue
            if qp["codes"].dim() == 3:
                self._add_bank(mod, qp, prefix, layer)
                continue
            if qp["codes"].dim() != 2:
                raise Refused(f"gguf: {mod}: expected a matrix or a fused expert bank")
            name = self.gguf_name(mod)
            if (encode := self._encoder(mod, name, qp["format"])) is None:
                if layer is not None:
                    # the template's rounding stands for a dropped tensor, so the codes reach nothing -- but a
                    # later step that changed the weight still changed it for nothing, and says so here
                    check_unchanged(layer, mod[len(prefix) :], qp)
                continue
            if layer is not None:  # the codes are written, so they must still describe this weight
                check_unchanged(layer, mod[len(prefix) :], qp)
            values = self.tp.values(name)
            source = self.source_names.tensor(mod + ".weight")
            if source is None:
                raise Refused(f"gguf: {mod}: no weight in the source checkpoint")
            with safe_open(self.src / self.where[source], "pt") as h:
                src = h.get_tensor(source).to(torch.bfloat16).float()
            if layer is not None:
                projection = layer.get_submodule(mod[len(prefix) :])
                if (order := getattr(projection, ORDER, None)) is not None:
                    src = src.index_select(1, order)
            order = row_order(values, src, name)
            self.sink.put(name, encode(qp, order, values, name, self.workers))
            self.from_codes.add(name)

    def _encoder(self, mod: str, name: str, fmt: str) -> Any:
        """The encoder of `mod`'s solved `fmt` codes as tensor `name`, or None when `finish` fits it from the template.

        Q4_K codes in a tensor the mix stores another way are dropped, and the original weight is quantized
        instead. Codes in another format were solved for one type only, so a tensor stored any other way
        is refused rather than written with the template's rounding.
        """
        kind, encode = _SOLVED[fmt]  # `GGUF._format_refusal` admits these formats only
        if self.types[name] == kind:
            return encode
        if fmt != Q4K_FORMAT:
            raise Refused(
                f"gguf: {mod} was solved as {fmt}, but the mix stores {name} as "
                f"{file.type_name(self.types[name])}; its codes have nowhere to go. Store it as "
                f"{file.type_name(kind)} in `tensor_types`, or drop the override"
            )
        self.dropped.setdefault(name, mod)
        return None

    def _add_bank(self, module: str, qp: dict[str, Any], prefix: str, layer: torch.nn.Module | None) -> None:
        """Write canonical expert slices at their GGUF expert offsets; never match rows across experts.

        The architecture's export layout declares projection rows (gate/up halves or interleaved rows).
        SourceNames reads the corresponding native fused tensor or per-expert weights. The template may
        permute rows within a projection, but expert numbering must remain fixed.
        """
        layout = export_layout(self.graph, module, qp["codes"].shape)
        if layout is None or self.source_names is None:
            raise Refused(f"gguf: {module}: no architecture export layout for the fused expert bank")
        if layer is not None:
            check_unchanged(layer, module[len(prefix) :], qp)
        sources = self.source_names.stored(module, qp["codes"].shape)
        direct = sources[0][0] if len(sources) == 1 and not sources[0][1] else None
        native = None
        if direct is not None:
            with safe_open(self.src / self.where[direct], "pt") as h:
                native = h.get_tensor(direct).to(torch.bfloat16).float()
            bank = module.rpartition(".")[0]
            transposed = next(b.transposed for lg in self.graph.layers for b in lg.banks if b.experts == bank)
            if transposed:
                native = native.transpose(1, 2)
        written: dict[str, set[int]] = defaultdict(set)
        for projection, sl, _ in layout.pieces:
            root, _, part = projection.rpartition(".")
            bank, _, index = root.rpartition(".")
            if not index.isdigit() or int(index) != sl[0]:
                raise Refused(f"gguf: {module}: export layout does not name numbered expert projections")
            name = self.gguf_name(bank + "." + part)
            if (encode := self._encoder(module, name, qp["format"])) is None:
                continue
            entry = {
                **qp,
                **{field: qp[field][sl] for field in ("codes", "scale", "zero_point") if field in qp},
            }
            rows, width = entry["codes"].shape
            ne = self.tp.tensors[name].shape
            if ne != (width, rows, qp["codes"].shape[0]):
                raise Refused(f"gguf: {name}: template shape {ne} does not match expert projection shape")
            if native is not None:
                source = native[sl]
            else:
                key = next((key for key, source_slice in sources if source_slice == sl), None)
                if key is None:
                    raise Refused(f"gguf: {projection}: no corresponding expert slice in the source")
                with safe_open(self.src / self.where[key], "pt") as h:
                    source = h.get_tensor(key).to(torch.bfloat16).float()
            start = int(index) * rows
            values = self.tp.values(name, start, start + rows)
            self.sink.put_rows(
                name, start, encode(entry, row_order(values, source, name), values, name, self.workers)
            )
            written[name].add(int(index))
        for name, experts in written.items():
            if experts != set(range(self.tp.tensors[name].shape[2])):
                raise Refused(f"gguf: {name}: export layout does not cover every expert")
        self.sink.written.update(written)
        self.from_codes.update(written)  # every expert of these banks came from the build's codes

    @torch.no_grad()
    def add_layer(self, ctx: LayerContext) -> None:
        """Encode the layer's qparams entries, then take every other tensor a step changed from the build.

        A Q4_K tensor's codes, recorded by the layer's quantize step, are written here, so a later step that changed
        that weight would store stale codes; `check_unchanged` refuses it. Every other tensor would come from the
        template, which holds the source's values, so a step that changed one (AWQ's norm scaling, a rotation) would
        be lost: each is compared with the source and, when it changed, written in the converter's basis
        (`_replace`).
        """
        prefix = ctx.graph.module + "."
        self._add(ctx.qparams, prefix, ctx.layer)
        for k, v in ctx.layer.state_dict().items():
            hf = prefix + k
            name = self.names.get_name(hf, try_suffixes=(".weight", ".bias"))
            if name is None or name not in self.types or name in self.from_codes:
                continue  # no tensor in the file, or its codes are already the build's solution
            if name in self.dropped:
                continue  # the build solved it and the mix stores another type: the template's rounding stands
            if hf in ctx.qparams or (k.endswith(".weight") and hf[: -len(".weight")] in ctx.qparams):
                continue  # `_add` wrote it (and checked it against the weight it was solved from)
            columns = None
            if k.endswith(".weight"):
                module = k[: -len(".weight")]
                columns = getattr(ctx.layer.get_submodule(module) if module else ctx.layer, ORDER, None)
            # the layer pass holds a GDN output projection in the converter's column order (`template_columns`),
            # so that weight is already permuted and its *source* is the side to permute for the comparison
            self._replace(name, hf, v, columns=columns)

    def add_model(
        self,
        model: torch.nn.Module,
        qparams: dict[str, dict[str, Any]],
        *,
        changed: bool = True,
    ) -> None:
        """Write a quantized head, and the embedding, final norm and head as the model's steps left them.

        Those three are the model-level tensors `add_layer` never sees, so a rotation's embedding and head and
        an AWQ-folded final norm would otherwise be the template's source values. `changed` says whether a
        model step ran at all; without one they cannot have changed and are left to the template.
        """
        if self.head in qparams:
            check_unchanged(model, self.head, qparams[self.head])
            self._add({self.head: qparams[self.head]}, self.head)
        if not changed:
            return
        for mod in {self.graph.embed, self.graph.final_norm, self.head}:
            if not mod:
                continue
            sub = model.get_submodule(mod)
            for leaf in ("weight", "bias"):
                p = getattr(sub, leaf, None)
                if p is None:
                    continue
                hf = f"{mod}.{leaf}"
                name = self.names.get_name(hf, try_suffixes=(".weight", ".bias"))
                if name is not None and name in self.types and name not in self.from_codes:
                    self._replace(name, hf, p, parts=True)

    def add_source_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Write the source-only tensors a model step rewrote (a rotated MTP head, a rotated vision merger).

        Keys are in-memory tensor names, as a model step hands them over. A tensor this file does not hold is
        recorded and skipped (the vision tower belongs to the mmproj, which this exporter does not write); one it
        does hold is written in the converter's basis, and one named like a declared source block that the file
        has no tensor for is refused rather than silently left as the source's.
        """
        for hf, value in sorted(tensors.items()):
            if (name := self._block_name(hf)) is not None and name in self.types:
                self._replace(name, hf, value)
            elif hf.startswith(("mtp.", "model.mtp.")):
                raise Refused(
                    f"gguf: {hf}: the template has no tensor for it, so the build's changed value would be lost"
                )
            else:
                self.skipped.append(hf)

    def _fill(self) -> None:
        """Store every tensor the layer pass did not, llama.cpp's way, from the template.

        Each thread takes a part of a tensor from the template to the file by itself, so no read, cast or write
        waits for the others. The parts are whole rows, and a block's record depends only on that block, so neither
        the part size nor the thread count changes a byte.
        """
        import gguf

        quant = {
            rule.Q4_K: lambda x: quantize_q4k(x, workers=1),
            rule.Q5_K: quantize_q5k,
            rule.Q6_K: lambda x: quantize_q6k(x, workers=1),
            rule.Q5_0: quantize_q5_0,
            rule.Q8_0: quantize_q8_0,
            **{
                t: lambda x, t=t: kern.quantize_weighted_chunk(t, x, file.row_bytes(t, x.shape[1]), None)
                for t in rule.NATIVE
            },
            **{
                int(t): lambda x, t=t: gguf.quants.quantize(x.numpy(), t).tobytes()
                for t in (
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.BF16,
                )
            },
        }

        def part(name: str, start: int, stop: int) -> None:
            rep = self.replaced.get(name)
            if rep is not None:  # a step changed this tensor: the template's values are the source's
                if self.imatrix is not None:
                    raise Refused(
                        f"gguf: {name}: the build changed this tensor, and an importance matrix measured on "
                        "the source's activations cannot weigh it"
                    )
                self.sink.put_rows(name, start, quant[self.types[name]](rep.rows(start, stop)))
                return
            values = self.tp.values(name, start, stop)
            if self.imatrix is None:
                self.sink.put_rows(name, start, quant[self.types[name]](values))
                return
            kind, ne = self.types[name], self.tp.tensors[name].shape
            weights = imat.column_weights(self.imatrix, name, ne)
            if weights is None or weights.shape[0] == 1:
                self.sink.put_rows(
                    name,
                    start,
                    _weighted(kind, values, self.sink.row[name], None if weights is None else weights[0]),
                )
                return
            # An expert bank: `rows` runs over every expert's rows end to end, and each expert has its own
            # measured importance, so a part is cut at the expert it belongs to rather than weighed by one
            # vector. Parts are whole rows, so a part never straddles two experts unless it is larger than
            # one, which `_fill` avoids by stepping in whole experts for a bank.
            per = int(np.prod(ne[1:-1], dtype=np.int64))
            for expert in range(start // per, (stop + per - 1) // per):
                lo, hi = max(start, expert * per), min(stop, (expert + 1) * per)
                if lo >= hi:
                    continue
                self.sink.put_rows(
                    name,
                    lo,
                    _weighted(kind, values[lo - start : hi - start], self.sink.row[name], weights[expert]),
                )

        parts = []
        for name, t in self.types.items():
            if name in self.sink.written:
                continue
            if name not in self.replaced and t == self.tp.tensors[name].tensor_type:
                self.sink.put(name, self.tp.raw(name))
                continue
            rows = self.tp.rows(name)
            step = max(1, _PART_VALUES // self.tp.tensors[name].shape[0])
            parts += [(name, i, min(i + step, rows)) for i in range(0, rows, step)]
        with ThreadPoolExecutor(max_workers=self.workers or kern.default_workers()) as pool:
            for done in [pool.submit(part, *p) for p in parts]:
                done.result()
        self.sink.written.update(self.types)

    def rounded_from_template(self) -> dict[str, str]:
        """Quantized tensors `finish` encoded from the template, and why, newly-rounded ones first.

        `_add` writes Q4_K only, so every other quantized tensor is llama.cpp's round-to-nearest of the original
        weight. When the build solved that weight, its codes are dropped; the file cannot show the difference,
        so the build record does.
        """
        why = {}
        for name, kind in self.types.items():
            if name in self.from_codes or name in self.replaced:
                continue  # the build's own codes or values
            if kind == self.tp.tensors[name].tensor_type or kind in _FLOAT:
                continue  # copied unchanged, or converted between float types
            solved = "solved by the build, codes dropped: " if name in self.dropped else ""
            if self.imatrix_path is not None and kind == rule.Q8_0:
                why[name] = f"{solved}Q8_0 fit ignores the importance matrix"
            elif name in self.unweighted:
                why[name] = f"{solved}{file.type_name(kind)}, fitted without the importance matrix"
            elif name in self.overridden:
                why[name] = f"{solved}tensor type override to {file.type_name(kind)}"
            elif kind in _FALLBACK_OF:
                why[name] = f"{solved}{file.type_name(kind)}: rows are not whole super-blocks"
            else:
                why[name] = f"{solved}the mix stores it as {file.type_name(kind)}"
        return dict(sorted(why.items(), key=lambda kv: (kv[0] not in self.dropped, kv[0])))

    def codes_dropped(self) -> dict[str, dict[str, str]]:
        """Each tensor the build solved but the file stores in another type: its type and the solved module.

        The file holds the template's tensor there (a fit of it, or its float bytes), so the build's codes for it
        reach nothing. Recorded, not refused: the file is still a valid file of the mix.
        """
        return {
            name: {"type": file.type_name(self.types[name]), "module": mod}
            for name, mod in sorted(self.dropped.items())
        }

    def finish(self, config: Any, recipe: Any, build_info: dict) -> Path:  # noqa: ARG002
        """Fill every tensor still unwritten, write the file and ``build.json``, drop the template.

        A tensor a step changed is encoded from the build's value in the converter's basis; every other comes
        from the template's bytes, or is llama.cpp's fit of them when the mix stores it another way.
        """
        self._fill()
        self.sink.close()
        # The record names the template this build fitted against, so hash it before a generated one is
        # dropped; `own_template` deletes it a few lines down.
        matrix = (
            None
            if self.imatrix_path is None
            else {
                "file": str(self.imatrix_path),
                "sha256": imat.sha256(self.imatrix_path),
                "template_sha256": file_digest(self.template_path),
            }
        )
        if self.own_template:
            self.template_path.unlink()
        rounded = self.rounded_from_template()
        # Every tensor whose solved codes the file does not hold, with the type it holds instead -- a float type
        # too, which `rounded_from_template` leaves out because a float copy is not a rounding.
        dropped = self.codes_dropped()
        if dropped:
            first = next(iter(dropped))
            logging.getLogger(__name__).warning(
                "gguf: the build solved %d tensors the mix stores in another type, so the file holds the "
                "template's tensor there, not the build's codes (e.g. %s, stored as %s); see build.json "
                "`gguf.codes_dropped`. To keep the codes, store them as Q4_K in `exporter.gguf.tensor_types`, "
                "or solve them in their stored type's format (`format_overrides`, Q5_K/Q6_K)",
                len(dropped),
                first,
                dropped[first]["type"],
            )
        info = {
            **build_info,
            "gguf": {
                "file": OUT_NAME,
                "mix": "custom" if self.tensor_types else "Q4_K_M",
                "llama_cpp": self.commit,
                "llama_cpp_pinned": PINNED,
                "kernel": kern.ggml_backend() or "numpy",
                "rounded_from_template": rounded,
                **({} if not dropped else {"codes_dropped": dropped}),
                # the tensors a step changed, which the file therefore holds the build's values of
                "from_build": {n: r.basis for n, r in sorted(self.replaced.items())},
                **({} if not self.skipped else {"changed_not_in_file": sorted(self.skipped)}),
                **({} if matrix is None else {"imatrix": {**matrix, "tensors": len(rounded)}}),
            },
        }
        (self.out / "build.json").write_text(json.dumps(info, indent=2, default=str))
        return self.out


def _weight_map(src: Path) -> dict[str, str]:
    """Source tensor name -> shard file."""
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    files = sorted(p.name for p in src.glob("*.safetensors"))
    out: dict[str, str] = {}
    for f in files:
        with safe_open(src / f, "pt") as h:
            out.update(dict.fromkeys(h.keys(), f))
    return out
