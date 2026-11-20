"""llama.cpp's Q4_K_M tensor mixing rule: which GGUF tensor ``llama-quantize ... Q4_K_M`` stores as which type.

A port of ``llama_tensor_get_type`` and ``tensor_allows_quantization`` (``src/llama-quant.cpp``) for the case this
exporter writes: ftype Q4_K_M, no importance matrix, no manual overrides, a model of an architecture in `ARCHS`.
Everything else llama.cpp special-cases there (Falcon, 8-expert MoE, 70B Llama, MLA's fused kv_b, rows that are
not a multiple of 32) is refused, not guessed. Rows that are not whole 256-value super-blocks fall back to Q5_0
(for Q4_K) or Q8_0 (for Q6_K and the output head), as ``tensor_type_fallback`` does.

The rule depends on the order tensors are visited: ``attn_v`` and ``ffn_down`` get Q6_K by a counter that advances
once per tensor, in the order of llama.cpp's model loader (`loader_order`).

``qwen35``'s gated-delta-net layers carry a fused ``attn_qkv`` and an ``attn_gate`` in place of separate q/k/v,
and ``ssm_*`` state-space tensors. The fused weight takes ``attn_v``'s schedule (`_is_attn_v`), the gate and
``ssm_out`` take the Q4_K default, and `_KEEP_GDN` names the tensors a Q4_K_M file leaves alone.

``qwen35moe`` adds expert banks (``ffn_*_exps``, one 3-D tensor per layer), a shared expert (``ffn_*_shexp``)
and a router (``ffn_gate_inp``, never quantized). A MoE file reads its ``ffn_down`` layer from the tensor name
rather than a running counter (`_ffn_layer`), so a layer's bank and shared expert get that layer's type.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..step import Refused

F32, F16, Q5_0, Q8_0, Q4_K, Q6_K, BF16 = 0, 1, 6, 8, 12, 14, 30  #: ``ggml_type`` values
Q5_K = 13  #: ``ggml_type`` for explicit Q5_K tensor overrides
IQ4_XS = 23  #: nonlinear 4-bit codes, template fitted by native ggml
#: three more template-fitted types a mix may store, each fitted by native ggml as IQ4_XS is:
#: 3-bit k-quant, 4-bit nonlinear codes in 32-value blocks, 3-bit grid codes with signs
Q3_K, IQ4_NL, IQ3_S = 11, 20, 21
#: types with no Python port: ``ggml_quantize_chunk`` fits them, weighted when an importance matrix is given
NATIVE = (IQ4_XS, IQ4_NL, Q3_K, IQ3_S)
#: ``llama_ftype`` of Q4_K_M, written as ``general.file_type``
FTYPE_Q4_K_M = 15
QK_K = 256
#: ``general.architecture`` values the ported branches cover
ARCHS = ("llama", "qwen2", "qwen3", "qwen35", "qwen35moe")

#: ``tensor_type_fallback``: the 32-value block type of a 256-value one, for rows that are not whole super-blocks
_FALLBACK = {Q4_K: Q5_0, Q6_K: Q8_0}
_LAYER = re.compile(r"blk\.(\d+)\.")

#: Gated-delta-net tensors a Q4_K_M file never stores quantized: ``tensor_allows_quantization`` excludes
#: ``ssm_conv1d`` by name, and its rows are the conv kernel (4 values), so quantizing it is not expressible.
#: ``ssm_alpha``/``ssm_beta`` are not here: ``llama-quantize`` stores them Q4_K like any other 2-D weight.
_KEEP_GDN = ("ssm_conv1d",)


def loader_order(names: Sequence[str]) -> list[str]:
    """`names` in llama.cpp's ``weight_name_comparer`` order: by ``blk.N.`` layer (others first), then by name."""

    def key(n: str) -> tuple[int, bytes]:
        m = _LAYER.match(n)
        return (int(m.group(1)) if m else -1, n.encode())

    return sorted(names, key=key)


def use_more_bits(i: int, n: int) -> bool:
    """llama.cpp's ``use_more_bits``: the first and last eighth of the layers, and every third one between."""
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def _ndims(ne: tuple[int, ...]) -> int:
    """``ggml_n_dims``: one past the last dimension larger than one, so trailing singletons do not count.

    A converted checkpoint can hold a one-row matrix: the shared-expert router is ``[n_embd, 1]``. llama.cpp
    sees one dimension there and leaves it alone, so counting the stored dimensions would quantize a tensor it
    keeps. The quantized file stores the same tensor with the singleton dropped, which is why reading types back
    from a Q4_K_M file cannot catch this; the rule must be checked against the source table.
    """
    n = 1
    for i, d in enumerate(ne):
        if d > 1:
            n = i + 1
    return n


def _quantizable(name: str, n_dims: int) -> bool:
    """``tensor_allows_quantization`` for the architectures in `ARCHS`."""
    return (
        n_dims >= 2
        and name.endswith("weight")
        and "_norm.weight" not in name
        and "ffn_gate_inp.weight" not in name
        and not any(k in name for k in _KEEP_GDN)
    )


def _is_attn_v(name: str) -> bool:
    """``category_is_attn_v``: the fused ``attn_qkv`` shares ``attn_v``'s type schedule and its counter.

    Gated-delta-net layers carry one fused ``attn_qkv`` instead of separate q/k/v. llama.cpp sorts it into the
    attn_v category (``category_is_attn_v``, ``llama-quant.cpp:170-173``), which is tested before the
    ``ATTENTION_QKV`` branch of the same chain, so that branch's Q5_K never runs for Q4_K_M.
    """
    return "attn_v.weight" in name or "attn_qkv.weight" in name


def expert_count(by_name: dict[str, tuple[tuple[int, ...], int]]) -> int:
    """Experts per bank, from the banks themselves: an ``_exps`` tensor is ``[cols, rows, experts]``.

    llama.cpp reads this from ``<arch>.expert_count``. Taking it from the shapes keeps the rule a function of
    the tensor table alone, and a file whose banks disagree is not one llama.cpp would have written.

    Raises:
        Refused: the expert banks do not agree on a count.
    """
    counts = {ne[2] for n, (ne, _) in by_name.items() if "_exps" in n and len(ne) == 3}
    if not counts:
        return 1
    if len(counts) != 1:
        raise Refused(f"gguf: expert banks disagree on the expert count {sorted(counts)}")
    return next(iter(counts))


def _ffn_layer(name: str, counter: int, n_expert: int) -> int:
    """``layer_info``: a MoE file takes the layer from the name, a dense one from the running counter.

    Mixtral's experts are not in layer order, so llama.cpp parses ``blk.N.`` instead of dividing the counter.
    Every ``ffn_down`` of one layer, bank and shared expert alike, then gets that layer's type.

    Raises:
        Refused: a MoE ``ffn_down`` that is not in a block.
    """
    if n_expert <= 1:
        return counter
    m = _LAYER.match(name)
    if m is None:
        raise Refused(f"gguf: {name}: a MoE ffn_down outside a block has no layer for its type; not ported")
    return int(m.group(1))


def q4_k_m_types(
    arch: str, n_layer: int, tensors: Sequence[tuple[str, tuple[int, ...], int]]
) -> dict[str, int]:
    """The ggml type of every tensor in a Q4_K_M file.

    Args:
        arch: ``general.architecture``.
        n_layer: ``<arch>.block_count``.
        tensors: ``(name, ne, type)`` of the source file's tensors; ``ne[0]`` is the row width.

    Returns:
        ``{name: ggml type}``; a tensor that is not quantized keeps its type.

    Raises:
        Refused: an architecture, tensor or width outside the ported rule.
    """
    if arch not in ARCHS:
        raise Refused(f"gguf: the Q4_K_M rule is ported for {list(ARCHS)}, not {arch!r}")
    if arch == "llama" and n_layer == 80:
        raise Refused(
            "gguf: llama.cpp treats an 80-layer llama as 70B and changes its attn_v types; not ported"
        )
    by_name = {n: (ne, t) for n, ne, t in tensors}
    tied = "output.weight" not in by_name
    n_wv = sum(_is_attn_v(n) for n in by_name)
    n_expert = expert_count(by_name)
    if n_expert == 8:
        raise Refused(
            "gguf: llama.cpp gives an 8-expert model its own attn_v, attn_k and attn_output types; not ported"
        )
    i_wv = i_down = 0
    out: dict[str, int] = {}
    for name in loader_order(list(by_name)):
        ne, t = by_name[name]
        if not _quantizable(name, _ndims(ne)):
            out[name] = t
            continue
        if "attn_kv_b" in name:
            raise Refused(f"gguf: {name}: MLA's fused kv_b takes another type in Q4_K_M; not ported")
        new = Q4_K
        if name == "output.weight" or (tied and name == "token_embd.weight"):
            new = Q6_K if ne[0] % QK_K == 0 else Q8_0
        elif _is_attn_v(name):
            if use_more_bits(i_wv, n_wv):
                new = Q6_K
            i_wv += 1
        elif "ffn_down" in name:
            if use_more_bits(_ffn_layer(name, i_down, n_expert), n_layer):
                new = Q6_K
            i_down += 1
        if ne[0] % QK_K and new in _FALLBACK:  # ``tensor_type_fallback``
            new = _FALLBACK[new]
        if ne[0] % 32:
            raise Refused(f"gguf: {name}: rows of {ne[0]} values; llama.cpp stores them in F16, not ported")
        out[name] = new
    return out
