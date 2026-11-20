"""The compressed-tensors ``quantization_config`` and ``build.json["serve"]`` of a checkpoint.

`ct_quantization_config` builds the config; `_CheckpointConfig` holds the Writer's methods that decide its
groups and ``ignore`` entries under every serving engine's names, and the serve plan recorded beside it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..arch import ModelGraph
from ..arch.derive import served_name
from ..format import Format
from ..intake.source_names import SourceNames
from ..step import Refused
from ..target import KernelPlan, LayerKind, Target, merge_serve
from .names import covers
from .pack import _FUSED

if TYPE_CHECKING:  # adapters import transformers at module level; keep BuildConfig's import light
    from ..arch.adapters import FusedRuntime


class _CheckpointConfig:
    """The Writer's quantization-config and serve-plan methods (a base of `output.export.Writer`)."""

    graph: ModelGraph
    fmt: Format
    target: Target
    formats: dict[str, Format]
    names: SourceNames
    meta: dict[str, Any]
    quantized: set[str]
    decoded: set[str]
    covered: set[str]
    written: set[str]
    replaced: set[str]
    head_fmt: Format | None
    head_shape: tuple[int, int]
    bank_roots: dict[str, set[str]]
    config_names: dict[str, dict[str, str]]
    pack_like: dict[str, str]

    def _served(self, name: str) -> str: ...  # Writer's

    def _unit_format(self, module: str) -> Format | None: ...  # _SourceTensors'

    def serve(self) -> dict:
        """``build.json["serve"]``: vLLM version, env, engine args, plugins and the kernels expected to bind."""
        from ..arch.adapters import ADAPTERS

        kinds: list[LayerKind] = ["dense"]
        if any(g.banks for g in self.graph.layers):
            kinds.append("moe")
        adapter = ADAPTERS[self.graph.adapter] if self.graph.adapter else None
        adjust = getattr(adapter, "serve_plan", None)
        routers = {bank.router for g in self.graph.layers for bank in g.banks}
        custom = {bank.router: bank.custom_router for g in self.graph.layers for bank in g.banks}
        # A kind keeps the body format while some unit of that kind is not overridden. An expert rule that
        # covers every bank leaves no expert in the body's MoE format, so the body's MoE kernel would be a
        # kernel the engine never binds (and `check_kernels` refuses a recorded set that differs from bound).
        body = {
            "dense": any(
                self.formats.get(grp.modules[0]) is None for g in self.graph.layers for grp in g.groups
            ),
            "moe": any(self.formats.get(bank.router) is None for g in self.graph.layers for bank in g.banks),
        }
        # A bank whose router selects its experts outside the fused kernel binds another expert kernel
        # (`Target.plan(..., decomposed=True)`); the record must name the one the engine binds.
        custom_moe = any(custom.values())
        entries: list[tuple[str, KernelPlan]] = [
            (k, self.target.plan(self.fmt, k, decomposed=k == "moe" and custom_moe)) for k in kinds if body[k]
        ]
        # A separately stored head binds under the body's adjusted settings, so the body's dense plan still
        # contributes them to the merge when no dense unit keeps the body format; it just is not recorded.
        merged = list(entries)
        if self.head_fmt is not None and not body["dense"]:
            merged.append(("dense", self.target.plan(self.fmt, "dense")))
        for key, f in self.formats.items():
            kind = "moe" if key in routers else "dense"
            plan = self.target.plan(f, kind, decomposed=kind == "moe" and custom.get(key, False))
            entries.append((kind, plan))
            merged.append((kind, plan))
        if adjust is not None:
            entries = [(k, adjust(plan)) for k, plan in entries]
            merged = [(k, adjust(plan)) for k, plan in merged]
        plans = {f"{k}:{i}": plan for i, (k, plan) in enumerate(merged)}
        serve = merge_serve(plans, self.target.vllm)  # every kernel's serve env and args must agree
        kernels = {k: {plan.kernel for kk, plan in entries if kk == k} for k in kinds}
        if self.head_fmt is not None:  # under the body's adjusted settings
            kernels["dense"].add(self.target.binds(self.head_fmt, self.head_shape, serve))
        serve["kernels"] = {k: sorted(v) for k, v in kernels.items()}
        return serve

    def quantization_config(self, recipe: Any = None) -> dict:
        """compressed-tensors ``quantization_config``: one group over every Linear, the rest ignored.

        ``actorder`` records what the recipe did: ``static`` when a GPTQ step solved in Hessian order (groups
        unchanged, no ``g_idx``), as llm-compressor writes it; else null. Each format
        ``format_overrides`` stores modules in gets a group of its own, after the body's.
        """
        static = any(getattr(st, "actorder", None) == "static" for st in getattr(recipe, "steps", ()))
        # every source module stored as float, under the names the serving engines build it as (see `_served`)
        source = (
            {
                k.rpartition(".")[0]
                for k, (_, shape) in self.meta.items()
                if k.endswith(".weight") and len(shape) == 2
            }
            | self.decoded
        ) - self.quantized
        implied = self._implied_v_proj()
        ignore = sorted(
            {self._served(m) for m in source}
            # A tied head has no ``lm_head.weight`` in the source, so the scan above cannot name it; left
            # out of ``ignore``, the group over every Linear would claim the float head and break loading.
            | {
                self._served(m)
                for m in (self.graph.embed, self.graph.lm_head)
                if m and self.names.module(m) is None
            }
            # A module the graph's adapter gives a second runtime name (a spec-decode draft's layers)
            # is built and matched under that name, whatever the checkpoint calls it.
            | set(self.graph.extra.get("draft_ignore", ()))
            | {
                alias
                for name, aliases in self.graph.extra.get("serve_ignore", {}).items()
                if self.names.module(name) not in self.quantized
                for alias in aliases
            }
            | self._runtime_ignore()
            | self._float_banks()
            # a layer with no v projection of its own keeps one in the fused unit (see `_implied_v_proj`);
            # `source` holds the checkpoint's names, so the graph name is what maps into it
            | {
                f"{self._served(stored).rpartition('.')[0]}.v_proj"
                for module in implied
                if (stored := self.names.module(module) or module) in source
            }
        )
        head = None if self.head_fmt is None else (self.graph.lm_head.rpartition(".")[2], self.head_fmt)
        return ct_quantization_config(self.fmt, ignore, static, head, self._override_groups())

    def _override_groups(self) -> list[tuple[list[str], Format]]:
        """Each ``format_overrides`` format and the modules it claims.

        A dense unit by the exact names it is solved and served as. An expert bank by the names the loaders
        probe for its experts, under the root its packed experts were written to (the source-name map may
        move it: MiniMax-M2's per-expert ``block_sparse_moe``): transformers reads the group naming experts as
        every bank's scheme, and stock vLLM's MoE method asks each bank for its ``.0.gate_proj``,
        ``.0.up_proj`` and ``.0.down_proj``.
        The bank module itself is left out: a model that keeps the experts stacked has no per-expert modules
        to initialize, and transformers refuses to quantize that module type.
        """
        from ..arch.adapters import FusedRuntime
        from ..solve.units import bank_module

        groups: dict[str, tuple[Format, list[str]]] = {}
        for lg in self.graph.layers:
            for grp in lg.groups:
                f = self.formats.get(grp.modules[0])
                if f is not None:
                    groups.setdefault(f.name, (f, []))[1].extend(grp.modules)
            for bank in lg.banks:
                f = self.formats.get(bank.router)
                if f is not None:
                    roots = self.bank_roots.get(bank_module(bank), set())
                    groups.setdefault(f.name, (f, []))[1].extend(self._bank_entries(roots, anchored=False))
        for mod, ref in sorted(self.pack_like.items()):
            f = self._unit_format(ref)
            key = self.names.tensor(mod + ".weight")
            if f is not None and key is not None and key.removesuffix(".weight") in self.quantized:
                groups[f.name][1].append(mod)
        implied = self._implied_v_proj()
        result = []
        for f, modules in groups.values():
            targets = set(modules)
            for module in modules:
                stored = self._served(self.names.module(module) or module)
                targets.add(stored)
                targets.update(served_name(stored, names) for names in self.config_names.values())
                if module in implied:
                    # the fused unit's derived v: same scheme, under the stored root the engine maps
                    v = f"{stored.rpartition('.')[0]}.v_proj"
                    targets.add(v)
                    targets.update(served_name(v, names) for names in self.config_names.values())
            for names in self.graph.extra.get("runtime_names", {}).values():
                for runtime, declaration in names.items():
                    packed = {
                        name for name in self.quantized if _ignore_matches(name, _stands_for(declaration))
                    }
                    if isinstance(declaration, FusedRuntime) and packed & targets:
                        parts = {
                            key.removesuffix(".weight")
                            for key in self.names.keys
                            if key.endswith(".weight")
                            and _ignore_matches(key.removesuffix(".weight"), declaration.checkpoint)
                        }
                        if parts != packed or not packed <= targets:
                            raise Refused(
                                f"fused runtime module {runtime!r} has shards with different schemes; "
                                "format_overrides must give every shard the same format"
                            )
                    if packed and packed <= targets:
                        # SGLang matches the body's Linear class before expanding fused shards. Name the
                        # fused runtime module explicitly so its stored override scheme wins first.
                        targets.update(_ignored_names(runtime, declaration))
                        targets.update(
                            served_name(entry, names)
                            for entry in _ignored_names(runtime, declaration)
                            for names in self.config_names.values()
                        )
            result.append((sorted(targets), f))
        return result

    def _implied_v_proj(self) -> dict[str, str]:
        """Stored attention module -> the ``v_proj`` its fused unit serves in place of, for a layer with no v.

        Some architectures share K and V in one tensor: the layer stores ``q_proj`` and ``k_proj`` and no
        ``v_proj``, while the model's other layers of the same attention shape store all three beside each
        other. The graph records that from the model itself — an attention-input group holding q and k and
        no v, beside a group elsewhere holding q, k and v — so this reads the architecture, not a model
        name. An engine still builds one fused ``qkv_proj`` for such a layer and, when it resolves the
        fused layer's scheme, expands that name back to the three shards (vLLM 0.33's
        `compressed_tensors.utils._match_fused_layer` matches a fused layer only when every shard matches
        one group). A config that names only q and k leaves the fused unit unmatched, so the engine takes
        the body's class-name group instead of the scheme the two stored shards are in. Name the derived v
        in the same group, and in ``ignore`` when they are kept float.

        An ``ignore`` entry leaves its modules out of the groups (`graph.extra["ignored"]` keeps them with
        their group kind), so a layer whose q and k are kept float is read there too.
        """
        units: dict[str, set[str]] = {}
        family: dict[str, str] = {}

        def note(module: str, layer: str, kind: str) -> None:
            parent, _, leaf = module.rpartition(".")
            if kind == "attn_in" and leaf in ("q_proj", "k_proj", "v_proj"):
                units.setdefault(parent, set()).add(leaf)
                family[parent] = parent[len(layer) :]

        for layer in self.graph.layers:
            for group in layer.groups:
                for module in group.modules:
                    note(module, layer.module, group.kind)
        for module, (_pattern, kind) in self.graph.extra.get("ignored", {}).items():
            owner = next((lg.module for lg in self.graph.layers if covers(lg.module, module)), "")
            note(module, owner, kind)
        complete = {family[parent] for parent, leaves in units.items() if len(leaves) == 3}
        return {
            f"{parent}.{leaf}": f"{parent}.v_proj"
            for parent, leaves in units.items()
            if leaves == {"q_proj", "k_proj"} and family[parent] in complete
            for leaf in leaves
        }

    def _float_banks(self) -> set[str]:
        """``ignore`` entries for the stacked expert banks this checkpoint stores float (an MoE MTP layer's).

        The scan in `quantization_config` sees only 2-D weights, so a stacked bank is missed there. vLLM and
        SGLang ask whether a fused MoE module is ignored through its experts' names (``<bank>.0.gate_proj``,
        ...), not the bank's own name, so the entry is `_experts_pattern`, as `_override_groups` writes.
        """
        stored = (self.covered | self.written) - self.replaced
        banks = {k.rpartition(".")[0] for k in stored if k.endswith(_FUSED)}
        return set(self._bank_entries({self._served(bank) for bank in banks}))

    def _bank_entries(self, roots: set[str], *, anchored: bool = True) -> list[str]:
        """The config entries for the experts of banks served at `roots`, under every engine's config name.

        An engine that looks a bank's scheme up under another name (`ArchAdapter.config_names`) needs the
        entry under that name too; a name no engine renames is written once.
        """
        names = {served_name(r, m) for r in roots for m in self.config_names.values()} | set(roots)
        return [_experts_pattern(name, anchored=anchored) for name in sorted(names)]

    def _runtime_ignore(self) -> set[str]:
        """Every serving engine's names for the weights it must build float.

        An adapter declares, per engine, a runtime name -> the checkpoint name it stands for
        (`ArchAdapter.runtime_names`), or a `FusedRuntime` for a module the engine fuses and expands. The
        exporter only writes them. One ``ignore`` list serves every engine, so an entry is written only when
        no engine can read it as naming a packed weight: neither its checkpoint side nor any of its runtime
        names may match a quantized module, under any engine's names. A declaration that cannot meet both is
        refused before the checkpoint's config is written.

        A plain ``re:`` runtime pattern that stands for several checkpoint parts of one layer names a fused
        module, which the engines expand into shard names; a pattern cannot reach those shards, so the
        declaration is refused rather than written and silently missed. A `FusedRuntime` names the shards,
        and the exporter writes them beside the fused name.
        """
        from ..arch.adapters import FusedRuntime  # here, not at the top: see TYPE_CHECKING above

        declared = self.graph.extra.get("runtime_names", {})
        packed_names = {  # runtime names under which some engine builds a packed module
            entry
            for names in declared.values()
            for runtime, value in names.items()
            if any(_ignore_matches(name, _stands_for(value)) for name in self.quantized)
            for entry in _ignored_names(runtime, value)
        }
        float_modules = {
            key.rpartition(".")[0]: memory.rpartition(".")[0]
            for key, (_, shape) in self.meta.items()
            if (
                key.endswith(".weight")
                and len(shape) == 2
                and key.rpartition(".")[0] not in self.quantized
                and (memory := self.names.memory(key)) is not None
            )
        }
        layer_names = tuple(layer.module for layer in self.graph.layers)
        out: set[str] = set()
        for engine, names in declared.items():
            for runtime, value in names.items():
                checkpoint = _stands_for(value)
                if any(_ignore_matches(name, checkpoint) for name in self.quantized):
                    continue
                if isinstance(value, FusedRuntime) and not value.shards:
                    raise Refused(
                        f"runtime_names[{engine!r}][{runtime!r}] declares a fused runtime module without "
                        "the shard names the engine probes; name them in FusedRuntime.shards"
                    )
                fused = (
                    ()
                    if isinstance(value, FusedRuntime)
                    else _fused_parts(runtime, checkpoint, float_modules, layer_names)
                )
                if fused:
                    raise Refused(
                        f"runtime_names[{engine!r}][{runtime!r}] stands for the checkpoint modules "
                        f"{list(fused)} of one layer: that is a fused runtime module, whose ignore name "
                        "must be what the engine probes after expansion. A regex cannot reach those shard "
                        "names (SGLang expands a fused module unless its literal name is in ignore), so "
                        "declare one entry per shard name, or the literal fused names, or a FusedRuntime "
                        "with those shard names"
                    )
                for entry in _ignored_names(runtime, value):
                    clash = next(
                        (name for name in sorted(self.quantized) if _ignore_matches(name, entry)), None
                    )
                    if clash is None:
                        clash = next((name for name in sorted(packed_names) if _claims(name, entry)), None)
                    if clash is not None:
                        raise Refused(
                            f"runtime_names[{engine!r}][{runtime!r}] writes {entry!r}, which would ignore "
                            f"the quantized module {clash!r}; one ignore list serves every engine"
                        )
                    out.add(entry)
        return out


def ct_quantization_config(
    f: Format,
    ignore: list[str],
    static: bool = False,
    head: tuple[str, Format] | None = None,
    overrides: list[tuple[list[str], Format]] | None = None,
) -> dict:
    """compressed-tensors ``quantization_config`` of a body format: one group over every Linear, `ignore` kept bf16.

    Args:
        f: the body format.
        ignore: module names stored unquantized.
        static: GPTQ solved in Hessian order (``actorder: static``, no ``g_idx``).
        head: (the head's leaf module name, its format) when the head is stored in its own format (steps.head);
            it gets a config group targeting it by name, with that format's weight and activation scheme (an
            FP8 head declares the body FP8 groups' dynamic per-token FP8 input activations; a weight-only
            head declares none).
        overrides: (module names or ``re:`` patterns, a format), one group each for the units
            ``format_overrides`` store apart from the body; they follow ``group_0`` so a non-regex fallback
            still finds the body group first.
    """
    if any(x.act == "int8_token" for x in (f, *(of for _, of in overrides or ()))):
        # Written as pack-quantized with no activations, vLLM would bind another kernel than the plan's.
        raise Refused(f"{f.name}: this exporter cannot yet write INT8 activations (int-quantized)")
    groups = {"group_0": {"targets": ["Linear"], "weights": _ct_weights(f, static),
                          "input_activations": _ct_acts(f), "output_activations": None,
                          "format": _ct_format(f)}}  # fmt: skip
    if head is not None:
        groups["group_head"] = {"targets": [f"re:.*{head[0]}$"], "weights": _ct_weights(head[1], False),
                                "input_activations": _ct_acts(head[1]), "output_activations": None,
                                "format": _ct_format(head[1])}  # fmt: skip
    for i, (targets, of) in enumerate(overrides or (), 1):
        # a unit stored apart keeps the body's activation scheme unless its own format names one
        acts = _ct_acts(of) if "nvfp4_block" in (f.act, of.act) else _ct_acts(f)
        groups[f"group_{i}"] = {"targets": list(targets), "weights": _ct_weights(of, static),
                                "input_activations": acts, "output_activations": None,
                                "format": _ct_format(of)}  # fmt: skip
    formats = {g["format"] for g in groups.values()}
    return {
        "quant_method": "compressed-tensors",
        "format": formats.pop() if len(formats) == 1 else "mixed-precision",
        "quantization_status": "compressed",
        "global_compression_ratio": None, "kv_cache_scheme": None, "sparsity_config": {},
        "transform_config": {}, "version": "0.20.1", "ignore": ignore, "config_groups": groups,
    }  # fmt: skip


def _experts_pattern(bank: str, *, anchored: bool = True) -> str:
    """The ``re:`` config entry for every expert of `bank`, under each name an engine serves its root as.

    Engines rename a wrapped text model's root and leave ``re:`` entries as written: vLLM 0.33 builds
    ``model.language_model.layers.3.mlp.experts`` as ``language_model.model.layers.3.mlp.experts``, SGLang
    0.5.23 as ``model.layers.3.mlp.experts``. A rename may reorder or drop the root's components *or
    substitute them* (NemotronH's vLLM class maps ``backbone`` to ``model``). A group entry therefore asks
    for the bank's path from its layer onwards and accepts any root prefix (or none) (``anchored=False``);
    an ``ignore`` entry of a source-only bank (``mtp.layers.0``) keeps the checkpoint's own root by default,
    so it cannot claim a sibling root's layer of the same index.
    """
    parts = bank.split(".")
    index = next((i for i, p in enumerate(parts) if p.isdigit()), 1)  # the layer's index in its list
    root, rest = parts[: index - 1], parts[index - 1 :]
    if anchored:
        head = f"(?:(?:{'|'.join(re.escape(p) for p in dict.fromkeys(root))})\\.)+" if root else ""
    else:
        head = "(?:.*\\.)?"  # any engine's root, or none: the path below the root is what identifies the bank
    return f"re:^{head}{re.escape('.'.join(rest))}\\.\\d+\\..*$"


def _stands_for(declared: str | FusedRuntime) -> str:
    """The checkpoint name or pattern a ``runtime_names`` value stands for."""
    from ..arch.adapters import FusedRuntime  # here, not at the top: see TYPE_CHECKING above

    return declared.checkpoint if isinstance(declared, FusedRuntime) else declared


def _ignored_names(runtime: str, declared: str | FusedRuntime) -> tuple[str, ...]:
    """The ignore entries a ``runtime_names`` value writes: the runtime name, then a fused module's shards."""
    from ..arch.adapters import FusedRuntime  # here, not at the top: see TYPE_CHECKING above

    return (runtime, *declared.shards) if isinstance(declared, FusedRuntime) else (runtime,)


def _ignore_matches(module: str, pattern: str) -> bool:
    """Whether a compressed-tensors ``ignore`` entry `pattern` can claim module `module`.

    compressed-tensors and vLLM match a plain pattern exactly; SGLang also accepts a dotted suffix.
    All three match ``re:`` with ``re.match`` from the start. The exporter checks with the most permissive
    reading, so a name this misses is ignored by no engine.
    """
    if pattern.startswith("re:"):
        return re.match(pattern[3:], module) is not None
    return module == pattern or module.endswith("." + pattern)


def _fused_parts(
    runtime: str, checkpoint: str, modules: dict[str, str], layers: tuple[str, ...]
) -> tuple[str, ...]:
    """The checkpoint modules one ``re:`` runtime pattern stands for in one layer, if there are several.

    A runtime module that stands for several checkpoint modules is fused at serving time: the engines
    expand its name into the shard names they build, so an ignore entry for the fused name alone misses
    them. Literal runtime names are safe (an engine checks a literal fused name before expanding), so
    only patterns are reported. Only modules inside a graph layer count: a fused module combines siblings,
    while a declaration outside the layer graph (a draft's bulk alias) is not a fused module.
    ``modules`` maps stored module names to the model names used by the graph's layers.
    """
    if not runtime.startswith("re:"):
        return ()
    by_layer: dict[str, list[str]] = {}
    for name, memory in sorted(modules.items()):
        if not _ignore_matches(name, checkpoint):
            continue
        layer = next(
            (p for p in sorted(layers, key=len, reverse=True) if covers(p, memory)),
            None,
        )
        if layer is not None:
            by_layer.setdefault(layer, []).append(name)
    return next((tuple(names) for names in by_layer.values() if len(names) > 1), ())


def _claims(packed: str, entry: str) -> bool:
    """Whether ignore `entry` can claim the module a packed weight is built under.

    `packed` is a runtime name or pattern. A plain `entry` also matches dotted suffixes in SGLang, so
    against a `packed` pattern it can claim any name the pattern stands for; two patterns may overlap in
    ways Python cannot decide. Both count as claims: the declaration is refused rather than risk hiding a
    packed weight.
    """
    if entry.startswith("re:"):
        return True if packed.startswith("re:") else _ignore_matches(packed, entry)
    if packed.startswith("re:"):
        return True
    return packed == entry or packed.endswith("." + entry)


def _ct_format(f: Format) -> str:
    """compressed-tensors' name of the layout `f`'s tensors are packed in."""
    if f.weight_type == "e2m1":
        return "nvfp4-pack-quantized"
    if f.weight_type == "fp8":
        return "float-quantized"
    return "pack-quantized"


def _ct_acts(f: Format) -> dict | None:
    """compressed-tensors QuantizationArgs (as a dict) of `f`'s input activations, or None for bf16."""
    if f.act == "fp8_token":
        return {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "token", "dynamic": True,
                "group_size": None, "observer": None, "actorder": None, "block_structure": None,
                "observer_kwargs": {}}  # fmt: skip
    if f.act == "nvfp4_block":
        return {**_NVFP4_ARGS, "dynamic": "local", "observer": "static_minmax", "actorder": None}
    return None


#: What compressed-tensors' NVFP4 preset says of weights and input activations alike.
_NVFP4_ARGS = {
    "num_bits": 4, "type": "float", "symmetric": True, "strategy": "tensor_group", "group_size": 16,
    "block_structure": None, "scale_dtype": "torch.float8_e4m3fn", "zp_dtype": None, "observer_kwargs": {},
}  # fmt: skip


def _ct_weights(f: Format, static: bool) -> dict:
    """compressed-tensors QuantizationArgs (as a dict) of a weight format."""
    if f.weight_type == "e2m1":
        return {
            **_NVFP4_ARGS,
            "dynamic": False,
            "observer": "minmax",
            "actorder": "static" if static else None,
        }
    if f.weight_type == "fp8":
        return {
            "num_bits": 8, "type": "float", "symmetric": True,
            "strategy": "channel" if f.group_size == -1 else "group",
            "group_size": None if f.group_size == -1 else f.group_size,
            "dynamic": False, "actorder": "static" if static else None, "observer": "minmax",
            "block_structure": None, "observer_kwargs": {},
        }  # fmt: skip
    return {
        "num_bits": f.weight_bits, "type": "int", "symmetric": f.symmetric,
        "strategy": "channel" if f.group_size == -1 else "group",
        "group_size": None if f.group_size == -1 else f.group_size,
        "dynamic": False, "actorder": "static" if static else None, "observer": "minmax",
        "block_structure": None, "observer_kwargs": {},
    }  # fmt: skip
