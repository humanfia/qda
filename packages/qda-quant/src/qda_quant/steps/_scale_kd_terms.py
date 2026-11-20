"""The memory terms ``scale_kd``'s ``checkpoint_peak`` adds up, from sizes only, and the shapes they need."""

from __future__ import annotations

from typing import cast

import torch

from ..engine import packed as P
from ..format import Format


def kd_terms(
    *,
    layers: int,
    tokens: int,
    hidden: int,
    inter: int,
    vocab: int,
    head_chunk: int,
    teacher: int,
    quantized: int,
    floats: int,
    groups: int,
    symmetric: bool,
    head: tuple[int, int, int, bool] | None = None,
    buffers: int = 0,
    tensors: int = 0,
    served: int = 0,
) -> tuple[tuple[str, int], ...]:
    """One `scale_kd` stage's footprint terms, in bytes, from sizes only (`ScaleKD.checkpoint_peak`).

    Args:
        layers: decoder blocks, each keeping its input for the backward pass.
        tokens: rows of one forward: the resolved `seqs_per_forward` sequences, padded to `seqlen`.
        hidden: hidden size.
        inter: the widest hidden-to-hidden product of a block.
        vocab: the head's outputs.
        head_chunk: positions the head scores in one piece.
        teacher: the teacher's bf16 weights, as `apply_checkpoint` loads them on the card.
        quantized: parameters of the student's quantized weights (a packed head too), which `load_student` holds
            as int8 codes.
        floats: the student's other parameters, which `load_student` holds in bf16.
        groups: groups of the student's quantized weights; each holds one trained scale and, unless the format is
            symmetric, one frozen int8 zero point.
        symmetric: the format stores no zero point.
        head: ``(rows, cols, group, symmetric)`` of an output head the checkpoint packs in a format of its own
            (a model step's, or the packed source's); its codes, scales and zero points join `student`, and its
            scales the optimizer's.
        buffers: the student's other buffers (rotary frequencies, masks), whose own dtypes `load_student` keeps.
        tensors: trained scale tensors; Adam keeps one fp32 step scalar for each.
        served: bytes of the weights the step decodes once (`ServedLinear.materialize`): bf16, or fp32 when the
            kernel cuts its input; a bank's stay bf16 (its grouped matmul's dtype).
    """
    bf16, fp32 = 2, 4
    zeros = 0 if symmetric else groups
    packed = 0
    if head is not None:
        hrows, hcols, hgroup, hsym = head
        hgroups = _groups(hrows, hcols, hgroup)
        packed = hrows * hcols  # its int8 codes; its scales join `groups` below, once
        groups += hgroups
        zeros += 0 if hsym else hgroups
    # the student as `load_student` holds it: its fp32 delta is part of the load, the served parameter itself
    student = quantized + packed + floats * bf16 + groups * (bf16 + fp32) + zeros + buffers
    return (
        ("teacher", teacher),
        ("student", student),
        ("served weights", served),
        ("layer inputs", layers * tokens * hidden * bf16),
        ("recompute", 2 * tokens * max(hidden, inter) * bf16),
        ("head log-probs", 4 * min(tokens, head_chunk) * vocab * 4),
        # one trained scale (zero points are frozen int8): its gradient and Adam's two moments; its fp32 delta
        # is the student's own parameter, sized above. Adam keeps one fp32 step scalar per trained tensor.
        ("optimizer", 12 * groups + 4 * tensors),
    )


def kd_teacher_terms(
    *, teacher: int, tokens: int, hidden: int, inter: int, vocab: int, head_chunk: int
) -> tuple[tuple[str, int], ...]:
    """The terms of the card the teacher runs on when a `scale_kd` build has two (`kd_terms` sizes the other).

    Args:
        teacher: the teacher's bf16 weights, as `apply_checkpoint` loads them on its card.
        tokens: rows of one forward (`kd_terms`).
        hidden: hidden size.
        inter: the widest hidden-to-hidden product of a block.
        vocab: the head's outputs.
        head_chunk: positions the head scores in one piece.

    The teacher runs under ``no_grad``, so nothing of its backward is kept: one forward's transients are bounded
    by gate and up, the activation and the down projection (an expert bank's top-k outputs sit inside it). Only
    the fp32 log-probs of a chunk cross to the student's card, and its ``exp`` is taken there.
    """
    bf16 = 2
    return (
        ("teacher", teacher),
        ("teacher activations", 3 * tokens * max(hidden, inter) * bf16),
        (
            "teacher log-probs",
            10 * min(tokens, head_chunk) * vocab,
        ),  # bf16 logits, their fp32 copy, the softmax
    )


def _groups(rows: int, cols: int, group: int) -> int:
    """Scale groups of a ``rows x cols`` weight: `group` columns each, the last one shorter (``-1``: one group)."""
    return rows * (-(-cols // (cols if group == -1 else group)))


def _largest_first(terms: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    """`StagePeak.terms`: the non-zero terms, largest first."""
    return tuple(sorted((t for t in terms if t[1]), key=lambda t: -t[1]))


def _packed_head(ctx: P.CheckpointPlan) -> tuple[int, int, int, bool] | None:
    """``(rows, cols, group, symmetric)`` of the output head when the checkpoint packs it, else None (bf16).

    `engine.packed.load_student` serves a ``Linear`` whose codes the checkpoint holds (``weight_packed``, or
    ``weight`` for FP8), and ``engine.checkpoint_size._quantized`` packs the head when a model step gives it a
    format of its own; a packed source says so from its own index. The head's format is its own (a model step's
    `format`, or the packed source's entry for that weight), not the body's: a symmetric g64 head on an
    asymmetric g128 body has different scales.
    """
    from ..engine.checkpoint_size import _OwnModelFormat
    from ..step import QuantStep

    name = ctx.graph.lm_head
    if not name:
        return None
    if ctx.packed is not None:
        from ..intake.source_names import SourceNames

        names = SourceNames(ctx.packed.path)
        key = names.tensor(name + ".weight_packed") or names.tensor(name + ".weight")
        stem = ctx.packed.stem_of(key or "")
        if stem is None:
            return None
        fmt = ctx.packed.quantizer(stem).fmt
    else:
        step = next(
            (
                s
                for s in ctx.recipe.steps
                if isinstance(s, QuantStep) and isinstance(s, _OwnModelFormat) and s.kind == "model"
            ),
            None,
        )
        if step is None:
            return None
        fmt = Format.parse(step.format)
    w = cast(torch.Tensor, ctx.meta.get_submodule(name).weight)
    return int(w.shape[0]), int(w.shape[1]), fmt.group_size, fmt.symmetric
