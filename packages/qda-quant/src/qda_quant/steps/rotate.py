"""Hadamard rotations (QuaRot, arXiv:2404.00456): R1 on the residual stream, R2 on attention value heads.

R1: every decoder-layer norm (and the final norm) is folded into the modules that read it, which leaves
an unweighted RMSNorm that commutes with any orthogonal ``Q``. Then the embedding and every module that
writes the residual stream (``o_proj``, ``down_proj``, routed experts' down, ``out_proj``) are rotated on
the output side (``Q^T W``) and every reader (q/k/v, gate/up, router, expert gate/up, GDN inputs,
``lm_head``) on the input side (``W Q``). The float model computes the same function; outliers are spread
over channels, which helps low-bit weights and FP8 activations.

R2: each value head is rotated by a ``head_dim`` Hadamard (``v_proj`` rows) and undone in ``o_proj``'s
columns of every query head that reads it (GQA-safe).

``Q`` is a randomized Hadamard matrix ``diag(signs) H / sqrt(d)`` (Sylvester, or Paley-I x Sylvester such as
5120 = 20 x 256; `hadamard`) when ``d`` has one, else a seeded random orthogonal matrix. All products run in fp64.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, Literal, cast

import torch
from pydantic import Field
from torch import nn

from ..arch import LayerGraph, ModelGraph, submodule
from ..arch.adapters import fused_experts
from ..arch.adapters.fused_experts import in_param
from ..format import Format
from ..intake.source_names import names_under, under
from ..step import Refused, Step
from ..target import KernelPlan
from ._model import OUTPUT_KINDS, layer_without_fold_anchor
from ._source_block import rewrite as rewrite_block


def _sylvester(n: int) -> torch.Tensor:
    """The ``n x n`` Sylvester Hadamard matrix of +-1 (``n`` a power of two), fp64."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _paley(q: int) -> torch.Tensor:
    """The Paley-I Hadamard matrix of order ``q + 1`` (``q`` prime, ``q % 4 == 3``), +-1, fp64.

    ``[[1, 1^T], [-1, Q + I]]`` with the Jacobsthal matrix ``Q[i, j] = chi(j - i)``, ``chi`` the quadratic character
    mod ``q``: the convention of the published checkpoints' Paley factor.
    """
    res = {(x * x) % q for x in range(1, q)}
    jac = torch.tensor(
        [[0 if i == j else (1 if (j - i) % q in res else -1) for j in range(q)] for i in range(q)],
        dtype=torch.float64,
    )
    top = torch.ones(1, q + 1, dtype=torch.float64)
    rest = torch.cat([-torch.ones(q, 1, dtype=torch.float64), jac + torch.eye(q, dtype=torch.float64)], 1)
    return torch.cat([top, rest], 0)


#: Paley factors whose construction was read back from a published checkpoint (20: the 27B's 5120 = 20 x 256). Other
#: orders would be Hadamard too, but nothing shipped verifies their convention, so those widths keep the seeded random
#: orthogonal matrix they always had (Nemotron's 2688, say) rather than change under a rotation already measured.
PALEY_ORDERS = (20,)


def hadamard(d: int) -> torch.Tensor | None:
    """A ``d x d`` Hadamard matrix of +-1 by the published construction, or None when there is none here for ``d``.

    Sylvester for a power of two; ``Paley-I(q) (x) Sylvester(d / (q + 1))`` when ``d = (q + 1) 2^n`` for a
    ``q + 1`` in `PALEY_ORDERS` (5120 = 20 x 256, q = 19).
    """
    if d & (d - 1) == 0:
        return _sylvester(d)
    for k in PALEY_ORDERS:
        if d % k == 0 and (d // k) & (d // k - 1) == 0:
            return torch.kron(_paley(k - 1), _sylvester(d // k))
    return None


def orthogonal(d: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Randomized Hadamard ``diag(s) H / sqrt(d)`` (`hadamard`), else a seeded random orthogonal ``[d, d]``; fp64.

    ``s`` is ``seed``'s draw of ``d`` random signs (``torch.Generator().manual_seed(seed)``), one per ROW: the
    published 9B and 27B checkpoints carry exactly this matrix for seed 35. A rotation is exact for any orthogonal
    matrix, but GPTQ on the rotated weights is not invariant to which one, so a rebuild needs this one.
    """
    g = torch.Generator().manual_seed(seed)
    h = hadamard(d)
    if h is not None:
        signs = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
        return (signs[:, None] * h / d**0.5).to(device)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).to(device)


@torch.no_grad()
def _norm_offset(n: nn.Module) -> float:
    """0 for ``x_hat * w`` norms, 1 for zero-centred ``x_hat * (1 + w)`` ones, probed on the module itself."""
    w = cast(torch.Tensor, n.weight)  # nn.Module attribute access is dynamic
    d = w.shape[-1]
    x = torch.randn(3, d, generator=torch.Generator().manual_seed(0)).to(w.device, w.dtype)
    y = n(x).double()
    xh = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6)
    w = w.double()
    err = [((y - xh * (w + off)).abs().max() / (y.abs().max() + 1e-12)).item() for off in (0.0, 1.0)]
    if min(err) > 2e-2:
        raise Refused(f"rotate: {type(n).__name__} is neither x*w nor x*(1+w) (errors {err})")
    return float(err.index(min(err)))


def _right(p: torch.Tensor, q: torch.Tensor) -> None:
    """``p <- p @ q`` on the last (input) dim, fp64."""
    p.data.copy_((p.data.double() @ q.to(p.device)).to(p.dtype))


def _left(p: torch.Tensor, q: torch.Tensor) -> None:
    """``p <- q^T @ p`` on the output dim (rows of a weight, possibly stacked; a bias vector)."""
    p.data.copy_((q.to(p.device).T @ p.data.double()).to(p.dtype))


def _rotate_output(m: nn.Module, q: torch.Tensor) -> None:
    """Rotate a linear that writes the residual stream: its weight and its bias."""
    lin = cast(nn.Linear, m)
    _left(lin.weight, q)
    if lin.bias is not None:
        _left(lin.bias, q)


def _scale_inputs(p: torch.Tensor, w: torch.Tensor, transposed: bool) -> None:
    """``p <- p * w`` on its input axis, fp64: the middle one of ``[E, in, out]``, else the last."""
    shape = (1, w.numel(), 1) if transposed else (1,) * (p.dim() - 1) + (w.numel(),)
    p.data.copy_((p.data.double() * w.to(p.device).reshape(shape)).to(p.dtype))


def _rotate_reader(m: nn.Module, q: torch.Tensor, gain: torch.Tensor | None = None) -> None:
    """Fold a norm `gain` into a module that reads the residual stream, then rotate its input side.

    A fused bank declares its own layout: ``[E, out, in]`` is rotated like a weight, and ``[E, in, out]``
    (``is_transposed``) is the transpose of one, so the input dim is its middle axis.
    """
    p = getattr(m, in_param(m))
    transposed = fused_experts.transposed(m)
    if gain is not None:
        _scale_inputs(p, gain, transposed)
    if transposed:
        _left(p, q)
    else:
        _right(p, q)


def _rotate_bank_output(m: nn.Module, q: torch.Tensor) -> None:
    """Rotate a fused bank that writes the residual stream: its down projection and its bias."""
    p = getattr(m, fused_experts.DOWN)
    if fused_experts.transposed(m):
        _right(p, q)
    else:
        _left(p, q)
    b = fused_experts.bias(m, fused_experts.DOWN)
    if b is not None:
        _right(b, q)  # ``[E, out]``: every expert's residual row turns with the stream


def _source_output(root: str, name: str, source: str, q: torch.Tensor) -> dict[str, torch.Tensor]:
    """A declared writer the model does not load, rotated on its output side, by source name.

    A source with nothing under the writer's `root` ships no such module (a text-only checkpoint of a
    multimodal family): there is nothing to rotate and this returns no tensors.

    Raises:
        Refused: the source has tensors under `root` but none under `name` (the declaration no longer matches
            the checkpoint), or a tensor under `name` is not a weight or bias (the rotation would miss it).
    """
    tensors = under(source, name)
    if not tensors and not names_under(source, root):
        return {}
    if not tensors:
        raise Refused(
            f"rotate: the source checkpoint has no tensors under the declared writer {name}; update the "
            "declaration (docs/add-architecture.md)"
        )
    bad = [k for k in tensors if k.rpartition(".")[2] not in ("weight", "bias")]
    if bad:
        raise Refused(
            f"rotate: the source has {bad[0]} under the declared writer {name}, which is not a weight or "
            "bias; the rotation would miss it"
        )
    qt = q.to("cpu").T  # the source's tensors are host tensors
    return {k: qt @ t.double() for k, t in tensors.items()}


def _verify_writers(model: nn.Module, graph: ModelGraph, declared: dict, q: torch.Tensor) -> None:
    """Refuse unless rotating each loaded module's declared writers rotates that module's output.

    A declaration names the Linears that write the stream, not what runs after them: a GELU after a merger's
    last Linear passes every structural check. So each loaded module with writers runs, on a float64 copy
    on the host, on its declaration's probe input before and after its writers are rotated; every output of
    the residual width must come out rotated. Runs before any weight is written.

    Raises:
        Refused: the declaration gives no probe for a loaded module with writers, no output of the module
            has the residual width, or one does not rotate with its writers.
    """
    import copy

    from ..arch.adapters import ADAPTERS
    from ..arch.verify import _close

    mods = dict(model.named_modules())
    key = graph.extra.get("stream_adapter")
    probe = getattr(ADAPTERS[key], "probe", None) if key else None
    qc = q.to("cpu", torch.float64)
    for root, names in declared.items():
        if not names or root not in mods:
            continue  # a module with no writers, or a source-only one: there is nothing to run
        kwargs = probe(model, root) if probe is not None else None
        if kwargs is None:
            raise Refused(
                f"rotate: architecture adapter {key!r} declares stream writers in {root} but gives no probe "
                "input for it, so their rotation cannot be checked; add probe(model, root) to the adapter "
                "(docs/add-architecture.md) or quantize without rotate (e.g. gptq alone)"
            )
        m = copy.deepcopy(mods[root]).to("cpu", torch.float64)
        with torch.no_grad():
            ref = _stream_outputs(m(**kwargs), qc.shape[0])
            for name in names:
                _rotate_output(m.get_submodule(name[len(root) + 1 :]), qc)
            got = _stream_outputs(m(**kwargs), qc.shape[0])
        if (
            not ref
            or len(got) != len(ref)
            or not all(_close(g, r @ qc) for g, r in zip(got, ref, strict=True))
        ):
            raise Refused(
                f"rotate: rotating the declared writers {', '.join(names)} does not rotate the output of "
                f"{root}, so something after them does not commute with the residual rotation (or the "
                "declaration misses a writer); fix the declaration or quantize without rotate (e.g. gptq "
                "alone)"
            )


def _unnamed_writers(graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """Per decoder layer with groups of no residual-writing kind, the linears that write the residual stream.

    An architecture that names its blocks something other than attention/mlp (NemotronH's ``mixer``) derives
    every group as ``other``. In a pre-norm block the residual stream reaches a linear only through a
    layer-scope norm, so the groups no such norm feeds are its writers (Mamba2's ``out_proj`` after its gated
    norm, attention's ``o_proj``, a quantized shared expert's ``down_proj``). `_verify_unnamed` checks the
    claim numerically before any weight is written.
    """
    readers = graph.extra.get("readers", {})
    out = {}
    for g in graph.layers:
        if not g.groups or any(grp.kind in OUTPUT_KINDS for grp in g.groups):
            continue
        fed = {r for k, rs in readers.items() if k.rpartition(".")[0] == g.module for r in rs}
        out[g.module] = tuple(n for grp in g.groups if not fed & set(grp.modules) for n in grp.modules)
    return out


def _ignored_writers(graph: ModelGraph, layer: str) -> tuple[str, ...]:
    """The layer's linears that write the residual stream and `ignore` keeps float.

    `ignore` decides quantisation, not basis: a linear left unquantised is still served, so it must be rotated
    with the stream or the model mixes two bases. Such a module is out of every group, which is why the grouped
    writers above cannot find it. A routed bank's parts are parameters of one module, not modules, and are
    rotated by `_rotate_bank_output` in `_r1`.
    """
    banks = tuple(f"{root}." for root in graph.extra.get("float_banks") or ())
    return tuple(
        n
        for n, (_pattern, kind) in (graph.extra.get("ignored") or {}).items()
        if kind in OUTPUT_KINDS and n.startswith(layer + ".") and not n.startswith(banks)
    )


def _rotate_output_measuring(m: nn.Module, q: torch.Tensor) -> float:
    """`_rotate_output` on a module stored as it is, returning the relative error the store throws away."""
    lin = cast(nn.Linear, m)
    exact = q.to(lin.weight.device).T @ lin.weight.data.double()
    lin.weight.data.copy_(exact.to(lin.weight.dtype))
    if lin.bias is not None:
        line = q.to(lin.bias.device).T @ lin.bias.data.double()
        lin.bias.data.copy_(line.to(lin.bias.dtype))
    return (lin.weight.data.double() - exact).norm().item() / exact.norm().item()


def _rotate_bank_output_measuring(m: nn.Module, q: torch.Tensor) -> float:
    """`_rotate_bank_output` on a bank stored as it is, returning the relative error the store throws away."""
    p = cast(torch.Tensor, getattr(m, fused_experts.DOWN))
    exact = (
        (p.data.double() @ q.to(p.device))
        if fused_experts.transposed(m)
        else (q.to(p.device).T @ p.data.double())
    )
    p.data.copy_(exact.to(p.dtype))
    b = fused_experts.bias(m, fused_experts.DOWN)
    if b is not None:
        line = b.data.double() @ q.to(b.device)
        b.data.copy_(line.to(b.dtype))
    return (p.data.double() - exact).norm().item() / exact.norm().item()


def _rotate_writers(
    get: Callable[[str], nn.Module],
    g: LayerGraph,
    writers: tuple[str, ...],
    reading: set[str],
    q: torch.Tensor,
    ignored: tuple[str, ...] = (),
    record: dict[str, float] | None = None,
) -> None:
    """Rotate every module of layer `g` that writes the residual stream on its output side.

    Args:
        get: resolves a fully qualified module name (on the model, or on a copy of the layer).
        g: the layer's `LayerGraph`.
        writers: its grouped writers (groups of a residual-writing kind, or `_unnamed_writers`).
        reading: every module the residual stream reaches on its input.
        q: the residual rotation.
        ignored: its writers that `ignore` keeps float; they are rotated too, in their own dtype.
        record: when given, receives each ignored writer's relative rounding error (for ``build.json``).
    """
    for n in writers:
        _rotate_output(get(n), q)
    for n in ignored:
        if n in writers:
            continue
        err = _rotate_output_measuring(get(n), q)
        if record is not None:
            record[n] = round(err, 9)
    grouped = {n for grp in g.groups for n in grp.modules}
    for bank in g.banks:
        if isinstance(bank.experts, str):
            _rotate_bank_output(get(bank.experts), q)
        else:
            for _ins, down in bank.experts:
                _rotate_output(get(down), q)
        # a shared expert kept bf16 is out of the groups but still writes the residual stream; its
        # in-projections read it (and are rotated with the norm's other readers), its output linear not
        for n in bank.shared:
            if n not in grouped and n not in reading:
                _rotate_output(get(n), q)


def _verify_unnamed(
    model: nn.Module,
    graph: ModelGraph,
    unnamed: dict[str, tuple[str, ...]],
    reading: set[str],
    q: torch.Tensor,
) -> None:
    """Refuse unless rotating each unnamed layer's writers rotates exactly what the layer adds to the stream.

    Runs on a float64 host copy of one layer per layout: with the input unchanged, ``layer(x) - x`` must come
    out multiplied by ``q``. A missed writer, a reader taken for one, or an op after a writer that does not
    commute with ``q`` (a per-channel scale, a projection to another width) all fail it.

    Raises:
        Refused: a layer layout's output does not rotate with its writers.
    """
    from ..arch.derive import decoder_layers
    from ..arch.verify import _capture, _close, _f64, _f64_copy, _out

    reps: dict[tuple, LayerGraph] = {}
    for g in graph.layers:
        if g.module in unnamed:
            names = [n for grp in g.groups for n in grp.modules] + [str(b.experts) for b in g.banks]
            reps.setdefault(
                (type(model.get_submodule(g.module)).__name__, *(n[len(g.module) :] for n in names)), g
            )
    if not reps:
        return
    _, layers = decoder_layers(model)
    io = _capture(model, layers, {g.index for g in reps.values()})
    qc = q.to("cpu", torch.float64)
    for g in reps.values():
        layer = _f64_copy(layers[g.index])
        args, kwargs = _f64(io[g.index])
        x = args[0] if args else kwargs["hidden_states"]
        ref = _out(layer, args, kwargs) - x
        try:
            _rotate_writers(
                partial(submodule, layer, g),
                g,
                unnamed[g.module],
                reading,
                qc,
                _ignored_writers(graph, g.module),
            )
            ok = _close(_out(layer, args, kwargs) - x, ref @ qc)
        except RuntimeError:  # a writer or bank of another width than the stream
            ok = False
        if not ok:
            raise Refused(
                f"rotate: {g.module} names its blocks something other than attention/mlp, and rotating the "
                f"linears no layer norm feeds ({', '.join(unnamed[g.module])}) does not rotate what it adds to "
                "the residual stream, so those blocks stay unclassified; quantize without rotate (e.g. gptq "
                "alone)"
            )


def _stream_outputs(out: Any, d: int) -> list[torch.Tensor]:
    """Every tensor of width `d` in a module's output (a tensor, a tuple or a ModelOutput), in order."""
    if isinstance(out, torch.Tensor):
        return [out] if out.shape[-1] == d else []
    items = out.values() if isinstance(out, dict) else out if isinstance(out, (list, tuple)) else ()
    return [t for o in items for t in _stream_outputs(o, d)]


class Rotate(Step):
    """QuaRot R1 (residual) and R2 (value heads).

    Reproducibility: bit-identical (same GPU type, software stack and card count; docs/reproduce.md).

    Attributes:
        residual: apply R1.
        value_heads: apply R2 on full-attention layers, on each layer class R2 keeps exact; a class it does not
            (an elementwise output gate, Qwen3.5) keeps no R2, nor does a head width whose values a
            query-only layer reads from a layer R2 cannot rotate (Gemma 4 KV sharing over k_eq_v);
            ``build.json`` lists both under ``value_heads_skipped``.
        seed: seed of the random signs / orthogonal matrix.
        when: ``always`` (the default) refuses the build when the model's layout cannot take the rotation;
            ``accepted`` plans the build without this step instead, prints one line and records the refusal in
            ``build.json`` under ``steps_skipped``. Only a refusal of `check` is turned into a skip, never an error
            raised later in the build. Left out of the record at its default, so existing recipes are unchanged.
    """

    name = "rotate"
    kind = "model"
    residual: bool = True
    value_heads: bool = True
    seed: int = 0
    when: Literal["always", "accepted"] = Field("always", exclude_if=lambda v: v == "always")
    #: R1 rewrites the declared writers and blocks outside the text model along with the stream
    carries_stream: ClassVar[bool] = True

    @property
    def changes_stream(self) -> bool:
        """Whether this step's configured transform changes the residual basis."""
        return self.residual

    def check(self, graph: ModelGraph, fmt: Format, plan: KernelPlan) -> None:  # noqa: ARG002
        """Refuse layouts where R1 cannot keep the float function.

        - Tied embeddings: the head would need the norm the embedding must not get.
        - A routed bank `ignore` keeps float: the export may copy it from the source as stored, so the rotation
          cannot be carried to the checkpoint.
        - Untransformed source modules (an MTP head, a text build's vision tower): the writer copies them, and a
          residual rotation would leave them reading the old basis while serving still uses them.
        - A module outside the text model that no architecture adapter declares: nothing says whether it writes
          the stream, and the rotation cannot carry an unknown one.
        - A source that ships a declared module but not its declared writer: the declaration no longer matches
          the checkpoint (a text-only checkpoint ships neither, and plans).
        - Post-norm blocks (a decoder-layer norm whose output no linear reads, e.g. OLMo-2): the block writes
          ``w * norm(y)`` to the residual stream, and a per-channel ``w`` neither commutes with ``Q`` nor folds
          into a linear.
        - Blocks normalized inside a submodule (a hyper-connection mixes and normalizes several residual
          streams): there is no norm at layer scope to fold into the readers.
        - A layer of unclassified groups (blocks named other than attention/mlp) whose layer norms feed every
          linear: no linear is known to write the stream.
        """
        if not self.residual and not self.value_heads:
            raise Refused(
                "rotate: residual and value_heads are both false, so the step changes nothing; remove it"
            )
        if not self.residual:
            return
        for root in graph.extra.get("float_banks") or ():
            if any(n.startswith(root + ".") for n in graph.extra.get("unloaded") or ()):
                raise Refused(
                    f"rotate: `ignore` keeps the routed bank {root} float, and the source holds names under it "
                    "that the model class does not load, so the export copies those tensors as stored and the "
                    "rotation would not reach them; the model would mix two bases. Quantize the bank (it "
                    "rotates like any other) or build without rotate"
                )
        if graph.extra.get("tied"):
            raise Refused("rotate: tied input/output embeddings; untie them first")
        stream = graph.extra.get("stream") or {}
        declared = stream.get("writers") or {}
        blocks = stream.get("blocks") or {}
        known = {*declared, *blocks}
        unloaded = graph.extra.get("unloaded") or ()
        for root, block in blocks.items():
            if root not in graph.extra.get("source_roots", ()):
                continue
            if block.refusal:
                raise Refused(f"rotate: {root}: {block.refusal}")
            if block.approximation:
                logging.getLogger(__name__).warning(
                    "rotate: %s: %s. The build rewrites it with the best constant fit and records the "
                    "residual in build.json under source_blocks.%s; the served model is unchanged.",
                    root,
                    block.approximation,
                    root,
                )
        orphans = [name for name in unloaded if not any(name.startswith(root + ".") for root in known)]
        if orphans:
            roots = sorted({name.split(".")[0] for name in orphans})
            raise Refused(
                f"rotate: the source checkpoint holds {len(orphans)} weights under {', '.join(roots)} "
                "that its model class does not load, and no architecture adapter declares them; the writer "
                "copies them, but they read the residual stream in the old basis (an MTP head feeds "
                "speculative decoding, a vision tower feeds the LM), so the build would silently degrade "
                "those paths; declare the module with stream(model) in an adapter, or quantize without "
                "rotate (e.g. gptq alone)"
            )
        loaded = set(graph.extra.get("outside") or ())
        for root, names in declared.items():
            shipped = [n for n in unloaded if n.startswith(root + ".")]
            absent = [w for w in names if shipped and not any(n.startswith(w + ".") for n in shipped)]
            if root not in loaded and absent:
                raise Refused(
                    f"rotate: the source checkpoint holds weights under {root} but none under its declared "
                    f"stream writer {absent[0]}, so the rotation cannot reach that module's output; update the "
                    "declaration (docs/add-architecture.md) or quantize without rotate (e.g. gptq "
                    "alone)"
                )
        undeclared = [root for root in graph.extra.get("outside") or () if root not in known]
        if undeclared:
            root = undeclared[0]
            raise Refused(
                f"rotate: {root} is outside the text model and no architecture adapter declares whether it "
                "writes the residual stream; add stream(model) to an adapter, returning "
                f"{{'writers': {{{root!r}: (writer, ...)}}}} (an empty tuple if it never writes the stream); "
                "see docs/add-architecture.md"
            )
        if module := next((m for m, ws in _unnamed_writers(graph).items() if not ws), None):
            raise Refused(
                f"rotate: {module} writes the residual stream through linears that derive as an "
                "unclassified group kind (the architecture names its blocks something other than "
                "attention/mlp) and its layer norms feed every one of them, so no linear is known to write "
                "the stream; the rotation would move the readers and leave the writers in the old basis, "
                "changing the float function; quantize without rotate (e.g. gptq alone)"
            )
        readers = graph.extra.get("readers", {})
        post = sorted(k for k, rs in readers.items() if not rs)
        if post:
            raise Refused(
                f"rotate: post-norm blocks ({post[0]} scales the block output before the residual add), so "
                "a residual rotation cannot keep the float function; quantize without rotate (e.g. gptq alone)"
            )
        if module := layer_without_fold_anchor(graph):
            raise Refused(
                f"rotate: {module} has no norm at layer scope to fold into its readers (its input is "
                "normalized inside a submodule, e.g. by a hyper-connection), so a residual rotation "
                "cannot keep the float function; quantize without rotate (e.g. gptq alone)"
            )

    def model_step_bytes(self, meta: nn.Module, graph: ModelGraph) -> int:
        """The residual rotation of the embedding and the head: a float64 copy and its float64 product at once.

        Measured on a resident Qwen3.5-9B build (vocabulary 248k x 4096): torch's peak was 31.99 GiB, the bf16
        weights plus two float64 copies of the embedding (`_r1`, `_right`) and the float64 rotation itself; the
        layer phase planned 23.35 GiB.
        """
        if not self.residual:
            return 0
        weights = [
            cast(torch.Tensor, meta.get_submodule(name).weight) for name in (graph.embed, graph.lm_head)
        ]
        width = int(weights[0].shape[-1])
        return 16 * max(int(w.numel()) for w in weights) + 8 * width**2

    @torch.no_grad()
    def apply_model(self, model: nn.Module, graph: ModelGraph, run: Any) -> None:
        """Rotate the model in place (same float function, rotated weights)."""
        if self.residual:
            self._r1(model, graph, run)
        if self.value_heads and (skipped := self._r2(model, graph)):
            logging.getLogger(__name__).warning(
                "rotate value_heads: R2 is not exact on %s; those layers keep no R2", ", ".join(skipped)
            )
            if run is not None:
                run.info["value_heads_skipped"] = skipped

    def _r1(self, model: nn.Module, graph: ModelGraph, run: Any) -> None:
        readers = graph.extra["readers"]
        reading = {r for rs in readers.values() for r in rs}  # modules the residual reaches on their input
        emb = cast(torch.Tensor, model.get_submodule(graph.embed).weight)
        q = orthogonal(emb.shape[-1], self.seed, emb.device)
        stream = graph.extra.get("stream") or {}
        blocks = stream.get("blocks") or {}
        source = getattr(getattr(run, "plan", None), "source", None) if run is not None else None
        g_f = None
        unnamed = _unnamed_writers(graph)
        kept_float: dict[str, float] = {}  # ignored writers: what storing their rotated weight costs
        for root in graph.extra.get("float_banks") or ():
            # a bank the model holds one expert per module is copied by the export from the source as stored,
            # so a rotation of it would not reach the checkpoint (a fused bank is written from the model)
            if not fused_experts.is_bank(model.get_submodule(root)):
                raise Refused(
                    f"rotate: `ignore` keeps the routed bank {root} float and the model holds it one expert per "
                    "module; the export copies such a bank from the source as stored, so the rotation would not "
                    "reach the checkpoint and the model would mix two bases; quantize the bank (it rotates like "
                    "any other) or build without rotate"
                )
        _verify_writers(model, graph, stream.get("writers") or {}, q)  # before any weight is written
        _verify_unnamed(model, graph, unnamed, reading, q)
        if blocks:
            final_norm = model.get_submodule(graph.final_norm)
            g_f = cast(torch.Tensor, final_norm.weight).double() + _norm_offset(final_norm)

        def fold(norm_name: str, reader_names: tuple[str, ...]) -> None:
            n = model.get_submodule(norm_name)
            nw = cast(torch.Tensor, n.weight)  # nn.Module attribute access is dynamic
            off = _norm_offset(n)
            w = nw.data.double() + off
            for r in reader_names:
                m = model.get_submodule(r)
                _scale_inputs(getattr(m, in_param(m)), w, fused_experts.transposed(m))
            nw.data.fill_(1.0 - off)

        for g in graph.layers:
            for name, rs in readers.items():
                if name.startswith(g.module + ".") and name.rpartition(".")[0] == g.module:
                    fold(name, rs)
                    for r in rs:
                        _rotate_reader(model.get_submodule(r), q)
            writers = unnamed.get(g.module) or tuple(
                n for grp in g.groups if grp.kind in OUTPUT_KINDS for n in grp.modules
            )
            _rotate_writers(
                model.get_submodule, g, writers, reading, q, _ignored_writers(graph, g.module), kept_float
            )
            for root in graph.extra.get("float_banks") or ():
                # a bank `ignore` keeps float is out of `g.banks` and writes the stream too (its readers were
                # rotated with the norm above); a fused bank is written from the model by the export
                if root.startswith(g.module + "."):
                    kept_float[root] = round(_rotate_bank_output_measuring(model.get_submodule(root), q), 9)
        if kept_float and run is not None:
            errs = sorted(kept_float.values())
            run.info["ignored_writers"] = {
                "why": "`ignore` keeps these float; they are rotated with the stream anyway, and storing the "
                "rotated weight in its own dtype is the only loss (no quantisation follows it)",
                "count": len(errs),
                "roundtrip_relative": {"worst": max(errs), "mean": round(sum(errs) / len(errs), 9)},
                "modules": dict(sorted(kept_float.items())),
            }
        head = cast(torch.Tensor, model.get_submodule(graph.lm_head).weight)
        fold(graph.final_norm, (graph.lm_head,))
        _right(head, q)
        emb.data.copy_((emb.data.double() @ q).to(emb.dtype))
        # declared writers (a vision merger) and blocks (an MTP head) outside the text model: the writer
        # copies them from the source, so their rotated tensors travel to the export through the run
        declared = stream.get("writers") or {}
        tensors = getattr(run, "tensors", None) if run is not None else None
        for root, name in ((r, n) for r, ns in declared.items() for n in ns):
            try:
                mod = model.get_submodule(name)
            except AttributeError:
                mod = None
            if mod is not None:
                _rotate_output(mod, q)
                if tensors is not None:
                    for leaf, p in mod.state_dict().items():
                        tensors[f"{name}.{leaf}"] = p.detach()
            elif source is not None:
                # a text-only class loads neither its vision tower nor an MTP head; the source keeps them
                rotated = _source_output(root, name, source, q)
                if tensors is not None:
                    tensors.update(rotated)
        for root, block in blocks.items():
            if source is None:
                if any(k == root or k.startswith(root + ".") for k in dict(model.named_modules())):
                    raise Refused(
                        f"rotate: {root} is a declared source block, but the model loads it and this build "
                        "has no source checkpoint to rewrite it from; it would keep the old basis; quantize "
                        "without rotate (e.g. gptq alone)"
                    )
                continue
            src = under(source, root)
            if not src:
                continue  # the checkpoint ships no such block
            assert g_f is not None
            changed, record = rewrite_block(root, block, q, g_f, src)
            if tensors is not None:
                tensors.update(changed)
            if run is not None:
                run.info.setdefault("source_blocks", {})[root] = record

    def _r2(self, model: nn.Module, graph: ModelGraph) -> list[str]:
        """Apply R2 where it keeps the float function exact; return what it skipped.

        The head width can differ by layer (Gemma 4: 256 on sliding, 512 on full attention), so each width
        gets its own Hadamard and its own exactness check. A layer that projects only queries reads the keys
        and values of another layer (Gemma 4's KV-shared layers): its ``o_proj`` undoes the rotation of
        values it does not compute, which is exact only if every layer of its width that computes values
        gets R2; otherwise no layer of that width does.
        """
        heads = cast(Any, getattr(model.config, "text_config", None) or model.config).num_attention_heads
        pairs: dict[int, tuple[str, str | None, str, int]] = {}  # layer -> (module, v_proj, o_proj, width)
        borrowers, unrotated = [], set()  # query-only layers; widths with a value producer R2 cannot rotate
        # `ignore` keeps a linear float by taking it out of its group, but a kept v/o still carries the values
        kept = {n: kind for n, (_, kind) in (graph.extra.get("ignored") or {}).items()}

        def linears(g: Any, kind: str) -> tuple[str, ...]:
            grouped = next((x.modules for x in g.groups if x.kind == kind), ())
            return (*grouped, *(n for n, k in kept.items() if k == kind and n.startswith(g.module + ".")))

        for g in graph.layers:
            ins, outs = linears(g, "attn_in"), linears(g, "attn_out")
            if not outs:
                continue
            o = outs[0]
            hd = cast(nn.Linear, model.get_submodule(o)).in_features // heads
            v = next((n for n in ins if n.endswith("v_proj")), None)
            rel = (g.module, v[len(g.module) + 1 :] if v else None, o[len(g.module) + 1 :], hd)
            if v:
                pairs[g.index] = rel
            elif ins and all(n.endswith("q_proj") for n in ins):
                borrowers.append(rel)
            else:
                unrotated.add(hd)
        rs = {hd: orthogonal(hd, self.seed + 1) for *_, hd in pairs.values()}
        ok = _r2_valid(model, pairs, rs)
        kind = {i: (type(model.get_submodule(p[0])).__name__, p[3]) for i, p in pairs.items()}
        unrotated |= {kind[i][1] for i in pairs if not ok[kind[i]]}
        blocked = {hd for *_, hd in borrowers if hd in unrotated}
        for i, (mod, v, o, hd) in pairs.items():
            if ok[kind[i]] and hd not in blocked:
                _r2_apply(model.get_submodule(mod), v, o, rs[hd], hd)
        for mod, _, o, hd in borrowers:  # a width no layer rotates leaves its readers as they are
            if hd in rs and hd not in blocked:
                _r2_apply(model.get_submodule(mod), None, o, rs[hd], hd)
        skipped = {name for (name, _), exact in ok.items() if not exact}
        return sorted(skipped | {f"head_dim {hd} (query-only layers read its values)" for hd in blocked})


def _r2_apply(layer: nn.Module, v: str | None, o: str, r: torch.Tensor, hd: int) -> None:
    """Rotate each value head of ``v`` (none for a layer that reads another's values) and undo it in ``o``."""
    om = cast(nn.Linear, layer.get_submodule(o))
    rd = r.to(om.weight.device)
    if v is not None:
        vm = cast(nn.Linear, layer.get_submodule(v))
        wv = vm.weight.data.double().reshape(-1, hd, vm.weight.shape[-1])  # [kv heads, hd, D]
        vm.weight.data.copy_(
            torch.einsum("ij,kjd->kid", rd.T, wv).reshape(vm.weight.shape).to(vm.weight.dtype)
        )
        if vm.bias is not None:
            vm.bias.data.copy_((vm.bias.data.double().reshape(-1, hd) @ rd).reshape(-1).to(vm.bias.dtype))
    wo = om.weight.data.double().reshape(om.weight.shape[0], -1, hd)  # [D, heads, hd]
    om.weight.data.copy_((wo @ rd).reshape(om.weight.shape).to(om.weight.dtype))


def _r2_valid(model: nn.Module, pairs: dict, rs: dict[int, torch.Tensor]) -> dict[tuple[str, int], bool]:
    """Per decoder-layer class and head width, whether R2 keeps a fp64 copy of one such layer exact.

    An elementwise output gate on the heads (Qwen3.5 attention) breaks it; such layers keep no R2.
    """
    from ..arch.derive import decoder_layers
    from ..arch.verify import _capture, _close, _f64, _f64_copy, _out

    _, layers = decoder_layers(model)
    reps: dict[tuple[str, int], int] = {}
    for i, (mod, _v, _o, hd) in pairs.items():
        reps.setdefault((type(model.get_submodule(mod)).__name__, hd), i)
    io = _capture(model, layers, set(reps.values()))
    ok = {}
    for key, i in reps.items():
        layer = _f64_copy(layers[i])
        args, kwargs = _f64(io[i])
        base = _out(layer, args, kwargs)
        _r2_apply(layer, pairs[i][1], pairs[i][2], rs[key[1]], key[1])
        ok[key] = _close(_out(layer, args, kwargs), base)
    return ok
