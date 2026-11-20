"""Rewrite a source block (an MTP head the model class drops) into the rotated residual basis.

The block's tensors come from the source checkpoint, so the rotation cannot change them in the model; this
returns the rewritten tensors for the export's source-tensor channel. Each of the block's norms is folded
into the linears that read its output, which are then rotated on their input, and the block's own writers
are rotated on their output (see `qda_quant.arch.adapters.stream.SourceBlock`).

The last norm feeds the shared output head, which the rotation gives the target's final-norm gain
(``head <- (head diag(g_f)) Q``). The block needs ``unit(z) diag(g_m) @ head``, so its norm gain must be
``g_m / g_f``; a per-channel gain is diagonal and does not commute with a Hadamard ``Q``, so the rewrite
stores the best constant fit ``mean(g_m / g_f)`` and records it in ``build.json``. The second inexact step,
the per-token RMS factor ``rms(g_f)`` that replaces ``g_f`` when it is folded into ``fc``, is not recorded
separately. The block drafts tokens for speculative decoding; the served model's output is unchanged.
"""

from __future__ import annotations

import torch

from ..arch.adapters.stream import SourceBlock
from ..step import Refused


def rewrite(
    root: str, block: SourceBlock, q: torch.Tensor, g_f: torch.Tensor, tensors: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict]:
    """The block's tensors rewritten in the rotated basis, and its record for ``build.json``.

    Args:
        root: the source root the block sits at (its tensors are under it).
        block: the declaration.
        q: the residual rotation ``[d, d]``, fp64.
        g_f: the target's final-norm gain, fp64, read before the rotation folded it into the head.
        tensors: the source checkpoint's tensors under `root`, by source name.

    Raises:
        Refused: a tensor under `root` is not a declared weight, `root` holds a weight the declaration does not
            account for, or a declared module has no weight.
    """
    leaves = [n for n in tensors if not n.endswith(".weight") and n not in block.fused]
    if leaves:
        raise Refused(
            f"rotate: the source has {leaves[0]} under the declared block {root}; the rewrite accounts for "
            "weights only, so the build would copy it into the rotated basis unchanged"
        )

    def module(n: str) -> str:
        return n if n in block.fused else n.removesuffix(".weight")

    def tensor(m: str) -> str:
        return m if m in block.fused else f"{m}.weight"

    mods = {module(n) for n in tensors}
    declared = {
        block.fc,
        *block.inputs,
        *(n for n, _ in block.folds),
        *(r for _, rs in block.folds for r in rs),
        *block.writers,
        block.norm,
        *block.kept,
    }
    if missing := sorted(declared - mods):
        raise Refused(f"rotate: the source has no {tensor(missing[0])} for the declared block {root}")
    if extra := sorted(mods - declared):
        raise Refused(
            f"rotate: the source has {tensor(extra[0])} under the declared block {root}, which the "
            "declaration does not account for; the build would copy it into the rotated basis unchanged"
        )
    w = {module(n): t.double() for n, t in tensors.items()}
    # the source tensors are host tensors; a build on a card reads them on the host too
    q, g_f = q.to("cpu"), g_f.to("cpu")
    d = g_f.numel()
    _shaped(root, w[block.fc], (d, 2 * d), "fc")
    for m in (*block.inputs, *(n for n, _ in block.folds), block.norm):
        _shaped(root, w[m], (d,), "norm")
    for _, readers in block.folds:
        for r in readers:
            _shaped(root, w[r], (-1, -1, d) if r in block.fused else (-1, d), "reader")
    for r in block.writers:
        _shaped(root, w[r], (-1, d, -1) if r in block.fused else (d, -1), "writer")

    def gain(m: str) -> torch.Tensor:
        return w[m] + block.offset

    fc = w[block.fc]
    g_e, g_h = (gain(m) for m in block.inputs)
    g_m = gain(block.norm)
    s = g_f.pow(2).mean().sqrt()
    new = {
        block.fc: torch.cat([q.T @ (fc[:, :d] * g_e) @ q, q.T @ (fc[:, d:] * (g_h * g_f / s)) @ q], 1),
    }
    for m in block.inputs:
        new[m] = torch.full((d,), 1.0 - block.offset, dtype=torch.float64)
    for norm, readers in block.folds:
        new[norm] = torch.full((d,), 1.0 - block.offset, dtype=torch.float64)
        for r in readers:
            new[r] = (w[r] * gain(norm)) @ q
    for r in block.writers:
        new[r] = q.T @ w[r]
    ratio = g_m / g_f
    v = ratio.mean()
    new[block.norm] = torch.full((d,), float(v - block.offset), dtype=torch.float64)
    fit = ((ratio - v).pow(2).mean().sqrt() / ratio.pow(2).mean().sqrt()).item()
    # A Hadamard turns any per-channel gain c into the scalar mean(c) in the rotated basis, so this norm's
    # per-channel freedom is one scalar and the draft's gain is forced proportional to g_f, not to g_m
    # (docs/recipe.md#rotation-and-speculative-decoding).
    residue = ((g_m - v * g_f).norm() / g_m.norm()).item()
    worst = (ratio / v - 1).abs().max().item()
    record = {
        "tensors": len(new),
        "norm_fit": {
            "kind": "best constant",
            "constant": round(float(v), 6),
            "stored": round(float(v - block.offset), 6),
            "relative_residual": round(fit, 6),
            "gain_residue": round(residue, 6),
            "worst_channel": round(worst, 6),
            "exact_when": "g_m/g_f is constant: a Hadamard maps every per-channel gain to its mean",
        },
        "pre_fc_rms": {
            "constant": round(float(s), 6),
            "exact_when": "the folded constant equals the token's own 1/rms(g_f * unit(z))",
        },
        "draft": "approximate: the block's last norm is the best constant fit; served logits unchanged",
    }
    return {tensor(m): t for m, t in new.items()}, record


def _shaped(root: str, t: torch.Tensor, shape: tuple[int, ...], role: str) -> None:
    """Refuse a declared module whose source weight does not have the residual width."""
    if len(shape) != t.dim() or any(a != -1 and a != b for a, b in zip(shape, t.shape, strict=True)):
        raise Refused(
            f"rotate: the source weight of {role} under the declared block {root} has shape "
            f"{tuple(t.shape)}, not {shape}; update the declaration or quantize without rotate"
        )
