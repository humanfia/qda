"""What the library knows about a model's structure, derived from the model itself.

`describe(model)` returns a `ModelGraph` of shared linear inputs, normalization anchors and routed expert banks
for each decoder layer. With `verify=True`, derived relations are checked numerically in fp64; adapters supply
only the structure that cannot be derived from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GroupKind = Literal["attn_in", "attn_out", "mlp_in", "mlp_out", "linear_attn_in", "linear_attn_out", "other"]

#: Roles of linears that derivation tags and keeps bf16 when the role is in ``describe``'s ``keep``. The default
#: (both) gives the coverage of the paper's checkpoints; neither is a kernel limit (vLLM 0.33's Marlin serves any
#: width). A new role is one more name here and one rule in `derive`.
GDN_GATES = (
    "gdn_gates"  #: Gated DeltaNet per-head projections (``in_proj_a`` / ``in_proj_b``, ``in_proj_ba``)
)
SHARED_EXPERTS = "shared_experts"  #: a MoE layer's shared expert (dense, every token)
KEEP_ROLES = (GDN_GATES, SHARED_EXPERTS)

#: The fewest output rows a Linear is quantized at: a smaller one is a gate or a tiny projection, and every
#: build keeps it bf16 (a kernel never binds that shape). `derive_groups` skips it, and `prepare`'s width
#: check reads the same number so it does not measure a module the plan never stores quantized.
MIN_QUANT_ROWS = 32
DEFAULT_KEEP = frozenset(KEEP_ROLES)


@dataclass(frozen=True)
class LinearGroup:
    """Linear modules of one layer that read exactly the same input tensor.

    Attributes:
        modules: fully qualified module names (e.g. ``model.layers.3.self_attn.q_proj``).
        kind: role of the group.
        norm: the normalization module whose output is this input, if any (AWQ / rotation anchor).
        prev: the module whose output this group's input is, if it is a single linear (e.g. v->o).
    """

    modules: tuple[str, ...]
    kind: GroupKind
    norm: str | None = None
    prev: str | None = None


@dataclass(frozen=True)
class ExpertBank:
    """Routed experts of one MoE layer, solved as a stack.

    Attributes:
        experts: per expert, its (in_modules, out_module) names; or one fused 3-D parameter name.
        router: the router (gate) module name; its top-k decides which tokens each expert sees.
        top_k: experts per token.
        fused: True if the in-memory module stores the experts as stacked 3-D parameters (transformers >= 5
            does this for Qwen3-MoE even when the checkpoint has per-expert tensors). It says nothing about
            checkpoint naming: export reads that from the source checkpoint's index.
        shared: names of shared-expert modules (dense, every token), if any.
        transposed: True if the fused bank stores ``[E, in, out]`` (``is_transposed`` on the module) rather
            than the canonical ``[E, out, in]`` the quantization units and statistics use.
        gated: True if a fused bank multiplies a gate into its up projection (``gate_up_proj``); False for a
            non-gated bank (``up_proj`` + ``down_proj``, the down input is ``act_fn(up)``).
        custom_router: True if the router applies work of its own to the selected weights (Gemma 4 multiplies
            each by a ``per_expert_scale``), so vLLM selects the experts outside the fused kernel
            (``RoutingMethodType.Custom``) and binds the table's decomposed expert kernel.
    """

    experts: tuple[tuple[tuple[str, ...], str], ...] | str
    router: str
    top_k: int
    fused: bool = False
    shared: tuple[str, ...] = ()
    transposed: bool = False
    gated: bool = True
    custom_router: bool = False


@dataclass(frozen=True)
class LayerGraph:
    """One decoder layer."""

    index: int
    module: str
    groups: tuple[LinearGroup, ...]
    banks: tuple[ExpertBank, ...] = ()


@dataclass(frozen=True)
class ModelGraph:
    """The whole model, as the engine and every method see it."""

    model_type: str
    layers: tuple[LayerGraph, ...]
    embed: str
    lm_head: str
    final_norm: str
    adapter: str | None = None  # name of the adapter that filled in what derivation could not
    verified: bool = False
    extra: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ExportLayout:
    """Where a quantized entry is stored when it is not written under its module's own name.

    Attributes:
        pieces: per stored tensor group, ``(module name, slice of the entry, stored key per field or None)``.
            The fields are ``packed``, ``scale``, ``shape`` and ``zero_point``; a field left out of the mapping is
            not stored, so a loader that cannot read it is never handed it.
        parts: source tensors the pieces supersede; the writer neither copies nor keeps them.
    """

    pieces: tuple[tuple[str, tuple, dict[str, str] | None], ...]
    parts: tuple[str, ...]


def describe(model, *, verify: bool = True, keep: frozenset[str] = DEFAULT_KEEP) -> ModelGraph:  # noqa: ANN001
    """Derive the `ModelGraph` of a Hugging Face causal LM.

    Args:
        model: the model; a meta-device skeleton works with ``verify=False`` (plan before loading).
        verify: check every derived relation in fp64 on a CPU copy of one layer per distinct layer kind;
            relations that fail are dropped and listed in ``graph.extra["dropped"]``.
        keep: the config's ``ignore`` entries, whose linears stay bf16: roles (`KEEP_ROLES`), exact module
            names and ``re:`` patterns (`derive.derive`). An empty set quantizes every role too.

    Returns:
        The graph (``verified`` says whether the numeric checks ran).
    """
    from . import verify as _verify_mod
    from .derive import derive

    # The trace and the fp64 checks both run forwards on CPUs (the checks also on a meta skeleton while
    # planning); `derive` routes the model's GDN kernels by device, so they take the reference while
    # calibration on a card keeps fla's fast path.
    graph = derive(model, keep)
    return _verify_mod.verify(model, graph) if verify else graph


def submodule(layer, graph: LayerGraph, name: str):  # noqa: ANN001, ANN201
    """The module `name` (fully qualified) inside decoder layer module `layer` described by `graph`."""
    return layer.get_submodule(name[len(graph.module) + 1 :])
