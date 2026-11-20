"""Decode a GGUF back to the HF checkpoint the pinned converter made it from (qwen2, qwen3, qwen3moe, llama, gemma3, qwen35, qwen35moe).

The inverse of the pinned converter (`llama_cpp.PINNED`): every name, shape and value transform it applies is
undone, and the values go through gguf-py's own ``dequantize`` (the pin's ``quants.py``, whose blocks are the only
arithmetic there), so every ggml type is exact by construction rather than re-derived here.

``qwen2``, ``qwen3``, ``qwen3moe``, ``llama`` and ``gemma3`` take names from gguf-py's own HF-to-GGUF map
(``get_tensor_name_map``, the table the converter renames with), run over the snapshot's tensor names, as
transformers' and the vLLM GGUF plugin's loaders do. The value transforms are the ones transformers undoes: the
llama Q/K rotary permute, the gemma ``+1`` norms, the MoE expert banks split per expert. A snapshot ``lm_head`` the
file does not carry is read from ``token_embd``, as llama.cpp does. ``qwen35``/``qwen35moe`` have their own table
below (value-head reorder, ``+1`` norms, MTP).

The output is a checkpoint stock vLLM serves as a language model: the snapshot's ``config.json`` and tokenizer
files byte for byte, the weights as bf16, and an index over them. A Qwen3.5 GGUF has no visual tower, so the row is
served with vLLM's ``language_model_only``; its 15 MTP tensors are written to ``model-mtp.safetensors``, which the
index does not name (vLLM loads indexed shards only, and without a speculative config never builds the MTP module).
``qda-decode-manifest.json`` records the row's sha256, the decoded files', and every tensor bf16 cannot hold
exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..exit import refuse
from ..io import sha256_file as _sha256
from . import file
from .llama_cpp import PINNED

#: output bytes of one shard of tensors (the snapshot's own shards are ~3 GB)
SHARD = 4 << 30
#: values dequantized at once by tensors that need no transform across rows
CHUNK = 32 << 20
#: GGUF tensor name -> HF module, under the block's prefix
_BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
_BLOCK_MOE = {
    **_BLOCK,
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_exps.weight": "mlp.experts.gate_up_proj",
    "ffn_up_exps.weight": "mlp.experts.gate_up_proj",
    "ffn_down_exps.weight": "mlp.experts.down_proj",
}
#: tensors outside a block
_GLOBAL = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "output.weight": "lm_head.weight",
}
#: ``blk.N.nextn.*`` -> MTP module
_NEXTN = {
    "eh_proj.weight": "fc.weight",
    "enorm.weight": "pre_fc_norm_embedding.weight",
    "hnorm.weight": "pre_fc_norm_hidden.weight",
    "shared_head_norm.weight": "norm.weight",
}
#: tensors the converter transforms across rows or columns: decoded whole
_WHOLE = (
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_a",
    "ssm_dt.bias",
    "ssm_conv1d.weight",
    "ssm_out.weight",
)
#: files a stock model directory carries beside the weights
_AUX = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
MANIFEST = "qda-decode-manifest.json"
#: GGUF architectures read through gguf-py's name map -> the snapshot ``model_type`` each comes from
_MAPPED = {
    "qwen2": ("qwen2",),
    "qwen3": ("qwen3",),
    "qwen3moe": ("qwen3_moe",),
    "llama": ("llama",),
    "gemma3": ("gemma3", "gemma3_text"),
}
#: GGUF architectures with their own inverse -> the snapshot's ``model_type``
_QWEN35 = {"qwen35": ("qwen3_5",), "qwen35moe": ("qwen3_5_moe",)}
ARCHITECTURES = (*_MAPPED, *_QWEN35)
#: GGUF tensors the converter computes from ``config.json`` rather than reads from the weights; the decode copies
#: that config, and vLLM computes the same values from it (llama3 rope scaling: ``LlamaModel.generate_extra_tensors``)
_FROM_CONFIG = ("rope_freqs.weight",)
#: one expert of a per-expert snapshot name, e.g. ``mlp.experts.7.up_proj``
_EXPERT = re.compile(r"\.experts\.(\d+)\.")


def _hf_name(name: str, first_mtp: int, block: dict[str, str]) -> tuple[str, str]:
    """``(HF name, GGUF suffix)`` of one tensor; the suffix picks its inverse transform.

    Raises:
        ValueError: the tensor is not one of the architecture's language-model tensors.
    """
    if name in _GLOBAL:
        return _GLOBAL[name], name
    index, _, suffix = name.removeprefix("blk.").partition(".")
    if not name.startswith("blk.") or not index.isdecimal():
        raise ValueError(f"{name}: not a language-model tensor of this architecture")
    if suffix.startswith("nextn."):
        return f"mtp.{_NEXTN[suffix.removeprefix('nextn.')]}", suffix
    if suffix not in block:
        raise ValueError(f"{name}: not a language-model tensor of this architecture")
    block_index = int(index)
    prefix = (
        f"mtp.layers.{block_index - first_mtp}."
        if block_index >= first_mtp
        else f"model.language_model.layers.{block_index}."
    )
    return prefix + block[suffix], suffix


class Gdn:
    """The GDN head counts and dims the value-head reorder needs, read from the file's own KV pairs."""

    def __init__(self, t: file.Template):
        a = t.arch
        self.nk = t.number(f"{a}.ssm.group_count")
        self.nv = t.number(f"{a}.ssm.time_step_rank")
        self.per = self.nv // self.nk
        self.head_v = t.number(f"{a}.ssm.inner_size") // self.nv
        self.head_k = t.number(f"{a}.ssm.state_size")
        if self.nk <= 0 or self.nk * self.per != self.nv or self.head_v <= 0:
            raise ValueError(f"ssm group_count {self.nk} does not divide time_step_rank {self.nv}")


def _perm(nk: int, per: int, dim: int, inverse: bool) -> torch.Tensor:
    """The converter's grouped-to-tiled value-head permutation, or its inverse."""
    p = torch.arange(nk * per * dim).reshape(nk, per, dim).transpose(0, 1).reshape(-1)
    return p.argsort() if inverse else p


def _invert(suffix: str, x: torch.Tensor, g: Gdn) -> torch.Tensor:
    """The HF tensor `x` is: undo the converter's transform for this tensor, in fp32."""
    if suffix == "ffn_gate_inp_shexp.weight":
        return x.reshape(1, -1)  # ggml may drop the shared-expert gate's size-1 axis
    if suffix.endswith("norm.weight") and not suffix.endswith("ssm_norm.weight"):
        return x - 1  # RMS gains are stored +1 (``linear_attn.norm`` excepted), exactly invertible in fp32
    if suffix == "attn_qkv.weight":
        qk = g.head_k * g.nk
        return torch.cat([x[: 2 * qk], x[2 * qk :][_perm(g.nk, g.per, g.head_v, True)]])
    if suffix == "attn_gate.weight":
        return x[_perm(g.nk, g.per, g.head_v, True)]
    if suffix in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias"):
        return x[_perm(g.nk, g.per, 1, True)]
    if suffix == "ssm_a":
        return torch.log(-x[_perm(g.nk, g.per, 1, True)])  # stored -exp(A_log)
    if suffix == "ssm_out.weight":
        return x[:, _perm(g.nk, g.per, g.head_v, True)]
    if suffix == "ssm_conv1d.weight":
        qk = 2 * g.head_k * g.nk
        v = x[qk:][_perm(g.nk, g.per, g.head_v, True)]
        return torch.cat([x[:qk], v]).reshape(x.shape[0], 1, x.shape[1])
    return x


def _values(info: file.TensorInfo, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Rows ``start:stop`` as fp32 ``[rows, ne[0]]``: gguf-py's dequantize of the stored blocks."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    width = info.shape[0]
    row = file.row_bytes(info.tensor_type, width)
    rows = int(np.prod(info.shape[1:], dtype=np.int64))
    stop = rows if stop is None else stop
    raw = np.asarray(info.data[start * row : stop * row])
    vals = dequantize(raw, GGMLQuantizationType(info.tensor_type))
    if not vals.flags.writeable:  # F32 hands back a view of the mapped file, which torch will not take
        vals = vals.copy()
    return vals.reshape(stop - start, width)


def _cast(x: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """`x` in `dtype`, and how many of its values `dtype` cannot hold exactly."""
    if dtype == torch.float32:
        return x.clone(), 0
    y = x.to(dtype)
    return y, int((y.to(torch.float32) != x).sum())


def _decode(info: file.TensorInfo, suffix: str, g: Gdn, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """One tensor in HF shape and `dtype`, and its count of values `dtype` rounds."""
    rows, width = int(np.prod(info.shape[1:], dtype=np.int64)), info.shape[0]
    if suffix in _WHOLE or rows * width <= CHUNK:
        x = torch.from_numpy(_values(info)).reshape(info.shape[::-1])
        return _cast(_invert(suffix, x, g), dtype)  # _values and _invert already give the HF shape
    return _chunked(info, dtype)


def _chunked(info: file.TensorInfo, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """A tensor that needs no transform, dequantized `CHUNK` values at a time into `dtype`."""
    rows, width = int(np.prod(info.shape[1:], dtype=np.int64)), info.shape[0]
    out = torch.empty(rows * width, dtype=dtype)
    step, bad = max(1, CHUNK // width), 0
    for start in range(0, rows, step):
        y, n = _cast(torch.from_numpy(_values(info, start, min(rows, start + step))).reshape(-1), dtype)
        out[start * width : (start + step) * width] = y
        bad += n
    return out.reshape(info.shape[::-1]), bad


def _unpermute(x: torch.Tensor, n_head: int) -> torch.Tensor:
    """Undo the converter's llama Q/K rotary permute (``LlamaModel.permute``; transformers'
    ``LlamaTensorProcessor._reverse_permute_weights``)."""
    dim = x.shape[0] // n_head // 2
    return x.reshape(n_head, dim, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)


class _Mapped:
    """Decodes the tensors `_mapped` paired, keeping the last expert bank so its experts are decoded once."""

    def __init__(self, t: file.Template, dtype: torch.dtype):
        self.dtype = dtype
        self.heads = {
            "q": t.number(f"{t.arch}.attention.head_count", 0),
            "k": t.number(f"{t.arch}.attention.head_count_kv", 0)
            or t.number(f"{t.arch}.attention.head_count", 0),
        }
        self.bank: tuple[str, torch.Tensor] | None = None

    def __call__(self, name: str, how: str, info: file.TensorInfo) -> tuple[torch.Tensor, int]:
        """The HF tensor for one ``(GGUF name, transform)`` pair that `_mapped` made."""
        kind, _, arg = how.partition(":")
        if kind == "expert":
            if self.bank is None or self.bank[0] != name:
                self.bank = (name, _chunked(info, torch.float32)[0])
            return _cast(self.bank[1][int(arg)], self.dtype)
        if not kind:
            return _chunked(info, self.dtype)
        x = torch.from_numpy(_values(info)).reshape(info.shape[::-1])
        if kind in self.heads:
            x = _unpermute(x, self.heads[kind])
        elif kind == "norm":
            x = x - 1  # stored +1 (``Gemma3Model.norm_shift``), exactly invertible in fp32
        return _cast(x, self.dtype)


def _shards(names: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """`names` split into shards of at most `SHARD` output bytes, keeping their order."""
    groups: list[list[str]] = []
    current: list[str] = []
    total = 0
    for name in names:
        if current and total + sizes[name] > SHARD:
            groups.append(current)
            current, total = [], 0
        current.append(name)
        total += sizes[name]
    if current:
        groups.append(current)
    return groups


def _snapshot_shapes(snap: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor names and shapes from safetensors headers without loading the source weights."""
    index = snap / "model.safetensors.index.json"
    parts = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.is_file()
        else {"model.safetensors"}
    )
    shapes = {}
    for part in sorted(parts):
        with safe_open(str(snap / part), framework="pt", device="cpu") as stream:
            shapes.update({name: tuple(stream.get_slice(name).get_shape()) for name in stream.keys()})  # noqa: SIM118
    return shapes


def _mapped(t: file.Template, snap: Path, config: dict) -> dict[str, list[tuple[str, str, file.TensorInfo]]]:
    """HF name -> ``[(GGUF name, transform, tensor)]`` for an architecture in `_MAPPED`, in file order.

    Each snapshot name goes through gguf-py's map the way the converter's own does: ``language_model.`` dropped
    (``ModelBase.filter_tensors``), one expert's ``.experts.N.`` read as the bank's ``.experts.``. A
    multimodal snapshot's visual tensors have no GGUF name and are left out.

    Raises:
        ValueError: a snapshot language tensor with no GGUF tensor of its shape, or a GGUF tensor no snapshot
            tensor names.
    """
    import gguf as gguf_py  # the PyPI package (extra gguf)

    arch = next(k for k, v in gguf_py.MODEL_ARCH_NAMES.items() if v == t.arch)
    names = gguf_py.get_tensor_name_map(arch, t.n_layer)
    fused = sorted(n for n in t.tensors if n.endswith(".attn_qkv.weight"))
    if fused:  # `qda build` never fuses; only `convert_hf_to_gguf.py --fuse-qkv` does
        raise ValueError(
            f"{fused[0]}: a fused QKV file (--fuse-qkv) is not decoded; convert without --fuse-qkv"
        )
    order = {name: i for i, name in enumerate(t.tensors)}
    by_hf: dict[str, tuple[tuple[int, int], str, str, file.TensorInfo]] = {}
    for hf, shape in _snapshot_shapes(snap).items():
        expert = _EXPERT.search(hf)
        key = _EXPERT.sub(".experts.", hf.replace("language_model.", ""))
        name = names.get_name(key, try_suffixes=(".weight", ".bias"))
        if name is None and "vision_config" in config and "language_model." not in hf:
            continue  # the visual tower: a GGUF holds the language model only
        how = ""
        if name == "output.weight" and name not in t.tensors:
            name = "token_embd.weight"  # tied: llama.cpp reads the output head from the embedding
        if name is None or name not in t.tensors:
            raise ValueError(f"{hf}: no tensor {name} in {t.path.name}")
        info = t.tensors[name]
        stored = tuple(info.shape[::-1])
        if expert:
            how, stored = f"expert:{expert[1]}", stored[1:]
            if int(expert[1]) >= info.shape[-1]:
                raise ValueError(f"{hf}: expert {expert[1]} beyond the {info.shape[-1]} of {name}")
        elif t.arch == "llama" and name.endswith(("attn_q.weight", "attn_q.bias")):
            how = "q"
        elif t.arch == "llama" and name.endswith(("attn_k.weight", "attn_k.bias")):
            how = "k"
        elif t.arch == "gemma3" and hf.endswith("norm.weight"):
            how = "norm"
        if stored != shape:
            # e.g. a gemma3 embedding cut to the tokenizer's vocab (``Gemma3Model.modify_tensors``): padding it back
            # would add output classes llama.cpp does not have
            raise ValueError(f"{name}: shape {stored} differs from snapshot {hf} {shape}")
        by_hf[hf] = ((order[name], int(expert[1]) if expert else 0), name, how, info)
    unread = set(t.tensors) - {name for _, name, _, _ in by_hf.values()} - set(_FROM_CONFIG)
    if unread:
        raise ValueError(
            f"{t.path.name}: {len(unread)} tensor(s) no snapshot tensor names: {sorted(unread)[:3]}"
        )
    # file order, then expert: one bank's experts are written one after another, so the bank is decoded once
    return {
        hf: [(name, how, info)] for hf, (_, name, how, info) in sorted(by_hf.items(), key=lambda e: e[1][0])
    }


def decode(gguf: str | Path, out: str | Path, snapshot: str | Path, dtype: str = "bf16") -> dict:
    """Decode one GGUF into a servable HF checkpoint at `out`; return the manifest also written there.

    Args:
        gguf: the ``.gguf`` file to decode.
        out: output directory, which must be new or empty.
        snapshot: the HF snapshot the template was converted from: its ``config.json`` and tokenizer files.
        dtype: ``bf16`` (the template's own) or ``fp32``; tensors bf16 cannot hold exactly are listed.

    Raises:
        ValueError: an architecture not in `ARCHITECTURES`, a snapshot of another model type, a tensor with no
            HF counterpart, or `out` is not empty.
    """
    t = file.Template(gguf)
    if t.arch not in ARCHITECTURES:
        raise ValueError(f"{t.path}: architecture {t.arch!r}; this decoder reads {', '.join(ARCHITECTURES)}")
    out, snap = Path(out), Path(snapshot)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: exists and is not empty")
    config = json.loads((snap / "config.json").read_text())
    model_types = {**_MAPPED, **_QWEN35}[t.arch]
    if config.get("model_type") not in model_types or (t.arch in _QWEN35 and "text_config" not in config):
        raise ValueError(
            f"{snap / 'config.json'}: model_type {config.get('model_type')!r}; a {t.arch} GGUF needs "
            f"{' or '.join(map(repr, model_types))}" + (" with a text_config" if t.arch in _QWEN35 else "")
        )
    started = time.monotonic()
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    width = 4 if dtype == "fp32" else 2
    if t.arch in _MAPPED:
        by_hf = _mapped(t, snap, config)
        decode_one: Callable[[str, str, file.TensorInfo], tuple[torch.Tensor, int]] = _Mapped(t, torch_dtype)
        known = _snapshot_shapes(snap)
        sizes = {hf: width * int(np.prod(known[hf])) for hf in by_hf}
    else:
        g, by_hf = Gdn(t), _qwen35(t, snap)

        def decode_one(_name: str, suffix: str, info: file.TensorInfo) -> tuple[torch.Tensor, int]:
            return _decode(info, suffix, g, torch_dtype)

        sizes = {
            hf: width * sum(int(np.prod(info.shape)) for _, _, info in entries)
            for hf, entries in by_hf.items()
        }
    multimodal = t.arch in _QWEN35 or "vision_config" in config
    return _write(t, by_hf, sizes, decode_one, out, snap, dtype, multimodal, started)


def _qwen35(t: file.Template, snap: Path) -> dict[str, list[tuple[str, str, file.TensorInfo]]]:
    """HF name -> its GGUF tensors (two for a fused expert bank), for ``qwen35``/``qwen35moe``.

    Raises:
        ValueError: a tensor with no HF counterpart, or a shape that disagrees with the file or the snapshot.
    """
    block = _BLOCK_MOE if t.arch == "qwen35moe" else _BLOCK
    # A file converted with --no-mtp omits the key: the pinned loader reads it with `required=false` and
    # defaults the NextN count to zero (llama.cpp `src/llama-model.cpp`, the `nextn_predict_layers` entry).
    n_layer = t.number(f"{t.arch}.block_count") - t.number(f"{t.arch}.nextn_predict_layers", 0)
    by_hf: dict[str, list[tuple[str, str, file.TensorInfo]]] = {}
    for name, info in t.tensors.items():
        hf, suffix = _hf_name(name, n_layer, block)
        by_hf.setdefault(hf, []).append((name, suffix, info))
    for hf, entries in by_hf.items():
        if hf.endswith(".experts.gate_up_proj"):
            entries.sort(key=lambda entry: entry[1])  # gate then up, regardless of file order
            if [suffix for _, suffix, _ in entries] != ["ffn_gate_exps.weight", "ffn_up_exps.weight"]:
                raise ValueError(f"{hf}: needs both gate and up expert banks")
            if entries[0][2].shape != entries[1][2].shape:
                raise ValueError(f"{hf}: gate and up expert bank shapes differ")
        elif len(entries) != 1:
            raise ValueError(f"{hf}: multiple GGUF tensors decode to this name")
        for name, suffix, info in entries:
            if suffix.endswith("_exps.weight") and (
                len(info.shape) != 3 or info.shape[2] != t.number(f"{t.arch}.expert_count")
            ):
                raise ValueError(f"{name}: expert bank shape {info.shape} disagrees with expert_count")
    if t.arch == "qwen35moe":
        known = _snapshot_shapes(snap)
        expected = {name for name in known if name.startswith(("model.language_model.", "lm_head."))}
        actual = {name for name in by_hf if not name.startswith("mtp.")}
        if actual != expected:
            raise ValueError(f"snapshot language tensor names differ: {sorted(actual ^ expected)[:3]}")
        for hf, entries in by_hf.items():
            shape = entries[0][2].shape[::-1]
            if len(entries) == 2:
                shape = (shape[0], shape[1] * 2, shape[2])
            elif entries[0][1] == "ffn_gate_inp_shexp.weight":
                shape = (1, int(np.prod(shape)))
            elif entries[0][1] == "ssm_conv1d.weight":
                shape = (shape[0], 1, shape[1])
            if known.get(hf) != shape:
                raise ValueError(f"{hf}: decoded shape {shape} differs from snapshot {known.get(hf)}")
    return by_hf


def _write(  # noqa: PLR0913
    t: file.Template,
    by_hf: dict[str, list[tuple[str, str, file.TensorInfo]]],
    sizes: dict[str, int],
    decode_one: Callable[[str, str, file.TensorInfo], tuple[torch.Tensor, int]],
    out: Path,
    snap: Path,
    dtype: str,
    multimodal: bool,
    started: float,
) -> dict:
    """Write the shards, the index, the snapshot's config and tokenizer files, and the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    # MTP stays out of the index: vLLM loads the indexed shards and, with no speculative config, never builds the
    # MTP module, so an indexed mtp.* weight would be an unexpected tensor. The 15 stay here, unconsumed.
    mtp = [hf for hf in by_hf if hf.startswith("mtp.")]
    shards = _shards([hf for hf in by_hf if hf not in mtp], sizes)
    print(f"{t.path.name}: {len(by_hf)} tensors ({len(mtp)} MTP), {len(shards)} shards", flush=True)

    files: dict[str, dict] = {}
    inexact: dict[str, int] = {}

    def write(names: list[str], part: str) -> None:
        tensors = {}
        for hf in names:
            decoded = [decode_one(name, suffix, info) for name, suffix, info in by_hf[hf]]
            tensors[hf] = torch.cat([x for x, _ in decoded], dim=1) if len(decoded) == 2 else decoded[0][0]
            bad = sum(n for _, n in decoded)
            if bad:
                inexact[hf] = bad
        save_file(tensors, str(out / part), metadata={"format": "pt"})
        print(
            f"  {part}: {len(names)} tensors, {sum(sizes[n] for n in names) / 2**30:.2f} GiB, "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )
        files[part] = {"bytes": (out / part).stat().st_size, "sha256": _sha256(out / part)}

    for i, names in enumerate(shards, 1):
        write(names, f"model-{i:05d}-of-{len(shards):05d}.safetensors")
    if mtp:
        write(mtp, "model-mtp.safetensors")

    weight_map = {
        hf: f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        for i, names in enumerate(shards, 1)
        for hf in names
    }
    index = out / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"metadata": {"total_size": sum(sizes[hf] for hf in weight_map)}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )
    files[index.name] = {"bytes": index.stat().st_size, "sha256": _sha256(index)}

    shutil.copy2(snap / "config.json", out / "config.json")
    files["config.json"] = {
        "bytes": (out / "config.json").stat().st_size,
        "sha256": _sha256(out / "config.json"),
    }
    for name in _AUX:
        if (snap / name).is_file():
            shutil.copy2(snap / name, out / name)
            files[name] = {"bytes": (out / name).stat().st_size, "sha256": _sha256(out / name)}

    types: dict[str, int] = {}
    for info in t.tensors.values():
        types[file.type_name(info.tensor_type)] = types.get(file.type_name(info.tensor_type), 0) + 1
    manifest = {
        "decoder": {"module": "qda_quant.gguf.decode", "llama_cpp": PINNED},
        "gguf": {"path": str(t.path.resolve()), "bytes": t.path.stat().st_size, "sha256": _sha256(t.path)},
        "snapshot": str(snap.resolve()),
        "dtype": dtype,
        "tensors": len(by_hf),
        "mtp": mtp,
        "types": types,
        "dtype_inexact": inexact,
        "config": {
            "snapshot_files": "config.json and tokenizer copied byte for byte",
            **(
                {
                    "note": "the GGUF has no visual tower: serve the language model (vLLM language_model_only)"
                    + (
                        ", and the MTP tensors are present in model-mtp.safetensors but not indexed"
                        if mtp
                        else ""
                    ),
                }
                if multimodal
                else {}
            ),
        },
        "files": files,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{t.path.name}: decoded to {out} in {time.monotonic() - started:.0f} s; "
        f"{len(inexact)} tensor(s) not {dtype}-exact",
        flush=True,
    )
    return manifest


def _arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The decode command's arguments."""
    p.add_argument("gguf", metavar="GGUF", help="the .gguf file")
    p.add_argument("--snapshot", required=True, metavar="DIR", help="the HF snapshot the template came from")
    p.add_argument("--out", required=True, metavar="DIR", help="output directory (new or empty)")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="weight dtype (default: bf16)")
    return p


#: The refusal's fix line for a GGUF this decoder cannot read (`docs/verify.md`, "A GGUF file").
OTHER_GGUF = (
    f"decode-gguf reads the GGUF architectures {', '.join(ARCHITECTURES)}; another architecture has "
    'no KL path (docs/verify.md, "A GGUF file")'
)


def run(a: argparse.Namespace) -> int:
    """Handler for ``qda decode-gguf``: a file it cannot decode is a two-line refusal, not a traceback."""
    try:
        decode(a.gguf, a.out, a.snapshot, a.dtype)
    except ValueError as e:
        return refuse(str(e), OTHER_GGUF if "architecture" in str(e) else None)
    return 0


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add ``decode-gguf`` to a subparser set."""
    p = sub.add_parser(
        "decode-gguf",
        help="decode a GGUF back to a servable HF checkpoint",
        description=(__doc__ or "").split("\n")[0],
    )
    p.set_defaults(run=run)
    return _arguments(p)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point, so the decoder runs from a checkout with no install."""
    ap = argparse.ArgumentParser(
        prog="python -m qda_quant.gguf.decode", description=(__doc__ or "").split("\n")[0]
    )
    _arguments(ap)
    a = ap.parse_args(argv)
    print(f"llama.cpp pin {PINNED}")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
