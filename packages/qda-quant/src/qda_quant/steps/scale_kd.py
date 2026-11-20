"""Scale distillation (``scale_kd``): train a packed checkpoint's group scales against the float model, codes fixed.

The student is the checkpoint run through its served numerics (`engine.served_modules.ServedLinear`: bf16 scales,
and for W4A8 the FP8 scale grid and FP8 per-token inputs, all straight-through). Each scale trains as
``s = s0 * exp(delta)``. The loss is KL(teacher || student) over the full vocabulary, summed over the scored
positions of a step's sequences and divided by their count. The teacher is the original float model.

At NVFP4 the block scale is an E4M3 code both in the checkpoint and in the served forward, and E4M3 codes are
12.5% apart: an update smaller than the half-step to the next code (``|delta|`` below about 0.06) rounds to the
same code, so it changes neither the served weight nor the stored checkpoint, and only the accumulated ``delta``
remembers it. The deltas are not lost: ``qda_state`` keeps them (``s0.*``, ``delta.*``, ``float.*``), so a later
stage or a resume continues from them; the exported checkpoint stores the scale at the code they round to.

Training uses shuffled passes with batches closed at ``batch_tokens``, Adam on the deltas (betas 0.9 / 0.99),
linear warmup then cosine decay to
``lr_min_frac``, a global gradient-norm clip, the head applied in chunks of ``head_chunk`` positions,
``seqs_per_forward`` sequences per forward (resolved at plan time), and activation checkpointing per decoder layer.

The weights a served module reads depend only on its `delta`, which changes when the optimizer steps: each step
decodes them once (`engine.served_modules.ServedLinear.materialize`) and every forward and backward of that step
reads the decode, rather than decoding per sequence. With ``scales`` untrained the weights never change, so a
``trainable: [norms]`` stage decodes once for the whole stage; that mode has no straight-through path at all.

``trainable`` may add float tensors, each with its own learning rate on an fp32 master (bf16 in the forward):

* ``norms``: every parameter of a normalization module (RMSNorm gains, q/k norms, gated norms, the final norm);
* ``gdn``: the float parameters of Gated DeltaNet blocks that are neither quantized nor norms (``in_proj_a/b``,
  ``conv1d``, ``dt_bias``, ``A_log``).

Data (``data``): a JSONL of traces ``{"prompt_ids": [...], "completion_ids": [...]}`` made with the teacher's
tokenizer; the scored positions are the completion's. Without ``data`` the recipe's calibration samples are
used, every position after the first scored.

Reproducibility: on the CPU a rebuild from the same recipe is bit-identical. On CUDA, with ``attn: sdpa``, cuDNN's
attention backward is not deterministic run to run, so a re-run reproduces the configuration and the KL band.
``attn: eager`` has no such node and writes the same bytes, in 1.24x the time and with more card memory
(docs/reproduce.md, section 'What "the same" means': 1,430 to 1,439 s against 1,155 to 1,159 s on Qwen3-8B
with 2,048-token sequences).

Scope in this version: dense, hybrid (Gated DeltaNet) and fused-expert MoE models (transformers 5 Qwen3-MoE layout,
per-expert stored modules) on one card, or on two with the teacher on the second (the student, its optimizer and the
loss stay on the first; only the teacher's log-probs cross the cards). The experts run grouped on a card or per
expert (`experts`, `engine.served_modules.ServedExperts`), and the teacher runs transformers' matching experts forward.
More than two cards is refused, and so are activation-ordered checkpoints that store ``weight_g_idx``.
"""


# This module is the step's public face: it re-exports the pieces that live in the sibling modules, so an
# import here is not an unused import.
# ruff: noqa: F401

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import Field, PrivateAttr

from ..engine import packed as P
from ..step import Refused, StagePeak, Step
from ._scale_kd_plan import (
    BATCHED_ROPE,
    _float_params,
    _is_linear_attention,
    _rope_types,
    _same_model,
    _token_ids,
    batched_rope,
)
from ._scale_kd_terms import (
    _groups,
    _largest_first,
    _packed_head,
    kd_teacher_terms,
    kd_terms,
)
from ._scale_kd_train import (
    Batcher,
    kl_sequences,
)

Trainable = Literal["scales", "norms", "gdn"]

#: Steps measured before the stage projects its own time (`ScaleKD.apply_checkpoint`'s log line). Three because
#: the first step also pays Triton's and cuDNN's one-off kernel work, and one more keeps the median off it.
_PROJECT_AFTER = 3


#: the largest ``seqs_per_forward`` measured (8 x 2048 tokens on Qwen3-30B-A3B: 100 GiB peak against 95 at 1), and
#: the default where it applies
MAX_SEQS_PER_FORWARD = 8


def _is_id(i: Any, vocab_size: int) -> bool:
    """Whether `i` is a token id of a vocabulary of `vocab_size` (an integer, not a bool, in range)."""
    return isinstance(i, int) and not isinstance(i, bool) and 0 <= i < vocab_size


class ScaleKD(Step):
    """Distil the scales (and optionally norms / Gated DeltaNet parameters) of a packed checkpoint.

    Reproducibility: noise band with ``attn: sdpa`` (quote a 3-seed mean and band; docs/reproduce.md); see the
    module docstring for ``attn: eager``.

    Attributes:
        trainable: what trains: ``scales``, ``norms``, ``gdn`` (any non-empty subset).
        teacher: the float model; default the build's source model, or the original model the packed
            input records (chained rounds keep the first round's teacher).
        data: trace JSONL (see module docstring); default the recipe's calibration samples.
        steps: optimizer steps.
        batch_tokens: tokens (prompt + completion) per step.
        max_len: sequences are cut to this many tokens.
        max_seqs: use only the first this many sequences of `data` (0 = all).
        lr: Adam learning rate on the scale log-offsets; a non-default value requires ``scales`` in trainable.
        norm_lr: learning rate of norm tensors; a non-default value requires ``norms`` in trainable.
        gdn_lr: learning rate of Gated DeltaNet tensors; a non-default value requires ``gdn`` in trainable.
        lr_min_frac: the cosine ends at ``lr * lr_min_frac``.
        warmup: steps of linear warmup from ``lr / warmup``.
        betas: Adam betas.
        eps: Adam denominator epsilon (default 1e-8).
        grad_clip: global gradient-norm clip (0 = off).
        seed: batch order seed.
        head_chunk: positions per head chunk (bounds the fp32 logits).
        seqs_per_forward: sequences of a step in one forward, longest first, padded on the right to the
            longest (`kl_sequences`); 1 is one sequence per forward, the reference. More is faster where every
            forward rebuilds all served weights (MoE: all experts), at the cost of padding. With position-fixed
            rotary embeddings (`BATCHED_ROPE`) the objective is the same, the GPU sums in another order; other
            rotary types are refused above 1, since they pick frequencies from the padded length. At most
            `MAX_SEQS_PER_FORWARD`, the largest run. The cap does not guarantee that a build fits: the plan does
            not estimate training memory, and a group costs more (on Qwen3-30B-A3B at 8 x 2048 tokens, 100 GiB
            peak against 95 one per forward). If a build runs out of memory, set ``seqs_per_forward: 1``. Default (None): resolved at plan time (`resolve`) to `MAX_SEQS_PER_FORWARD` where
            the rotary type allows it, else 1; ``resolved.yaml`` records the value.
        attn: attention implementation of student and teacher. None (default) lets transformers choose a
            supported implementation for each model component; an explicit value is passed unchanged.
            ``eager`` makes a repeat build write the same bytes.
        checkpointing: recompute each decoder layer in backward instead of storing its activations.
        experts: how routed experts run (`engine.served`): ``grouped`` (fast), ``loop`` (per expert, the
            reference), ``auto`` (grouped on a card, else loop). The teacher runs the same way.
        tf32: TF32 for fp32 matmuls (faster, different numerics).
    """

    name = "scale_kd"
    kind = "checkpoint"
    max_devices = 2  # the student and the optimizer, and the teacher on the second
    paths = ("data",)
    models = ("teacher",)

    trainable: list[Trainable] = Field(default_factory=lambda: ["scales"], min_length=1)
    teacher: str | None = None
    data: str | None = None
    #: the `data` rows the plan read, linted and checked; training uses them rather than reading the file again
    _read: list[dict] | None = PrivateAttr(default=None)
    steps: int = 200
    batch_tokens: int = 65536
    max_len: int = 16384
    max_seqs: int = 0
    lr: float = 3e-4
    norm_lr: float = 1e-4
    gdn_lr: float = 3e-5
    lr_min_frac: float = 0.1
    warmup: int = 10
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 0
    head_chunk: int = 1024
    seqs_per_forward: int | None = Field(None, ge=1)
    attn: str | None = None
    checkpointing: bool = True
    experts: Literal["auto", "grouped", "loop"] = "auto"
    tf32: bool = False

    def apply_checkpoint(self, src: Path, out: Path, run: P.CheckpointRun) -> None:
        """Train on the packed checkpoint `src`, write the result to `out`."""
        from ..engine.served_modules import ServedExperts
        from ..intake.loading import load_model

        ck = P.Packed(src)
        teacher_dir = self._teacher(ck.teacher() or run.teacher)  # checked at plan time (`check_checkpoint`)
        dev = run.devices[0]  # the student, the optimizer and the loss
        tdev = (
            run.devices[1] if len(run.devices) > 1 else dev
        )  # the teacher and its head (`check_checkpoint`)
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        rows = self._rows(run, teacher_dir)
        t0 = time.perf_counter()
        scales, floats = ck.state(run.log)
        grouped = self.experts == "grouped" or (self.experts == "auto" and dev.type == "cuda")
        student = P.load_student(ck, scales, floats, dev, self.attn, grouped)
        teacher = load_model(
            teacher_dir, dtype=torch.bfloat16, device_map={"": str(tdev)}, attn_implementation=self.attn
        ).eval()
        if any(
            isinstance(m, ServedExperts) for m in student.modules()
        ):  # the teacher runs as the student does
            teacher.set_experts_implementation("grouped_mm" if grouped else "eager")
        for p in teacher.parameters():
            p.requires_grad_(False)
        served = P.served_modules(student)
        groups, masters, float_names = self._params(student, served)
        frozen = "scales" not in self.trainable
        # NVFP4 and FP8 decode their served weights once per step (per stage when the scales are frozen); the
        # other formats decode per call, so the card holds no decoded copy of their weights
        once = [m for m in served.values() if m.decodes_once]
        for m in once:
            m.materialize()
        if self.checkpointing:
            cast(Any, student).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            student.train()  # HF checkpoints only in train mode; these models have no dropout at 0
        load_s = time.perf_counter() - t0
        run.log(
            f"scale_kd: {len(rows)} sequences, {sum(len(r[0]) for r in rows)} tokens; "
            + ", ".join(f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.2f}M" for g in groups)
            + (f"; served weights decoded once per {'stage' if frozen else 'step'}" if once else "")
            + f"; student on {dev}, teacher on {tdev}; loaded in {load_s:.0f} s"
        )
        opt = torch.optim.Adam(groups, lr=self.lr, betas=self.betas, eps=self.eps, fused=dev.type == "cuda")
        batcher = Batcher([len(r[0]) for r in rows], self.batch_tokens, self.seed)
        params = [p for g in groups for p in g["params"]]
        log: list[dict] = []
        t_train = time.perf_counter()
        for step in range(self.steps):
            ts = time.perf_counter()
            if step and not frozen:  # the optimizer moved the scales: decode the weights this step serves
                for m in once:
                    m.materialize()
            batch = batcher.next()
            n = sum(len(rows[i][0]) - rows[i][1] for i in batch)
            kl = 0.0
            order = [rows[i] for i in sorted(batch, key=lambda j: -len(rows[j][0]))]
            spf = self.seqs_per_forward
            assert spf is not None  # `compile_plan` resolved it (`resolve`)
            for a in range(0, len(order), spf):
                group = order[a : a + spf]
                kl += kl_sequences(student, teacher, group, self.head_chunk, n, dev, backward=True)
            for mp, p in masters:
                mp.grad = None if p.grad is None else p.grad.float()
                p.grad = None
            for p in params:  # parameters without gradients still take an optimizer step with zero gradients
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            grads = [p.grad for p in params]
            gnorm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))
            if not math.isfinite(kl) or not math.isfinite(gnorm):
                raise FloatingPointError(f"scale_kd: step {step}: loss {kl}, gradient norm {gnorm}")
            if self.grad_clip and gnorm > self.grad_clip:
                for g in grads:
                    g.mul_(self.grad_clip / gnorm)
            f = self._lr(step) / self.lr
            for g in groups:
                g["lr"] = g["base_lr"] * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for mp, p in masters:
                    p.copy_(mp.to(p.dtype))
            rec = {"step": step + 1, "kl": kl / n, "gnorm": gnorm, "lr": self._lr(step), "seqs": len(batch),
                   "tokens": sum(len(rows[i][0]) for i in batch), "step_s": round(time.perf_counter() - ts, 3)}  # fmt: skip
            log.append(rec)
            run.log("scale_kd " + json.dumps(rec))
            if len(log) == _PROJECT_AFTER and self.steps > len(log):
                median = statistics.median(r["step_s"] for r in log)
                per_step = sum(r["tokens"] for r in log) / len(log)
                run.log(
                    f"scale_kd: measured {median:.2f} s/step (median of {len(log)} at {per_step:.0f} "
                    f"tokens/step); projected {median * self.steps / 60:.1f} min of training for the "
                    f"configured {self.steps} steps ({median * (self.steps - len(log)) / 60:.1f} min left), "
                    "at this batch_tokens and trainable set"
                )
        train_s = time.perf_counter() - t_train
        new_scales = {k: v for m in served.values() for k, v in m.state().items()}
        new_floats = dict(floats)
        names = P.float_names(student, ck)
        for (mp, _), name in zip(masters, float_names, strict=True):
            new_floats[names[name]] = mp.detach().cpu()
        changed = sum(int((m.scale() != m.s0).sum()) for m in served.values())
        total = sum(m.s0.numel() for m in served.values())
        tokens = sum(r["tokens"] for r in log)
        stage = {
            "step": self.name,
            "settings": self.model_dump(),
            "teacher": teacher_dir,
            "load_s": round(load_s, 1),
            "train_s": round(train_s, 1),
            "tokens_per_s": round(tokens / max(train_s, 1e-9), 1),
            "scales_changed": [changed, total],
            "log": log,
        }
        run.log(f"scale_kd: {changed} of {total} scales changed; {stage['tokens_per_s']} tokens/s")
        moe = any(isinstance(m, ServedExperts) for m in student.modules())
        del student, teacher
        P.write(ck, out, new_scales, new_floats, stage, teacher_dir, run.devices, moe=moe)

    def resolve(self, fmt: Any, plan: Any, config: Any, source: str) -> ScaleKD:  # noqa: ARG002
        """This step with `seqs_per_forward` named: its own, else 8 if batching is exact for both models, else 1.

        Exact: every rotary type of the student and of the teacher is position-fixed (`batched_rope`). The teacher is found as `check_checkpoint` finds it (`teacher`, else the original model a packed source
        records, else the source). When its config cannot be read, the default is 1; `check_checkpoint` then
        refuses with the reason.
        """
        if self.seqs_per_forward is not None:
            return self
        from ..intake.loading import load_config

        default = P.Packed(source).teacher() if P.is_packed(source) else source
        try:
            tcfg = load_config(self._teacher(default, weights=False))
        except Refused:  # `check_checkpoint` refuses with the reason
            tcfg = None
        n = MAX_SEQS_PER_FORWARD if tcfg is not None and not batched_rope(config, tcfg) else 1
        return self.model_copy(update={"seqs_per_forward": n})

    def check_checkpoint(self, ctx: P.CheckpointPlan) -> None:
        """Refuse before any stage runs, from metadata only (configs, tokenizer files, the data file's first row).

        Checks: one or two cards (the teacher on the second); the teacher resolves and has a config; teacher and
        student have the same architecture,
        vocabulary size and tokenizer; routed experts are readable and have a kernel plan; ``trainable`` names
        tensors the model has; every row of ``data`` is a trace and the selection training uses (`select`) keeps
        one or more. Without ``data``, the calibration samples get the same selection at plan time, unless a
        generate stage still has to make them: then the selection runs after generation, in training.

        Raises:
            Refused: with the reason.
        """
        for field, kind in (("lr", "scales"), ("norm_lr", "norms"), ("gdn_lr", "gdn")):
            if kind not in self.trainable and getattr(self, field) != type(self).model_fields[field].default:
                raise Refused(
                    f"scale_kd.{field} has no effect without {kind!r} in trainable; "
                    f"add {kind!r} to scale_kd.trainable or leave {field} at its default"
                )
        if len(ctx.devices) > 2:
            raise Refused(
                f"scale_kd runs on one card, or on two with the teacher on the second; this build has "
                f"{len(ctx.devices)} (pass one or two devices; a multi-card quantization stage then runs on its "
                "own, with scale_kd on its output)"
            )
        from ..intake.loading import load_config

        teacher = self._teacher(ctx.teacher, weights=False)
        if not (Path(teacher) / "config.json").is_file():
            raise Refused(f"scale_kd: teacher {teacher} has no config.json")
        try:
            tcfg = load_config(teacher)
        except Refused as e:
            raise Refused(f"scale_kd: teacher {e}") from None
        same = _same_model(tcfg, ctx.config)
        if same is not None:
            raise Refused(f"scale_kd: teacher {teacher} is {same[0]}, the student {same[1]}")
        tv, sv = tcfg.get_text_config().vocab_size, ctx.config.get_text_config().vocab_size
        if tv != sv:
            raise Refused(f"scale_kd: teacher {teacher} has vocabulary {tv}, the student {sv}")
        if _token_ids(teacher) != _token_ids(ctx.source):
            raise Refused(f"scale_kd: teacher {teacher} maps tokens to ids otherwise than {ctx.source}")
        if self.seqs_per_forward is not None and self.seqs_per_forward > 1:
            if self.seqs_per_forward > MAX_SEQS_PER_FORWARD:
                raise Refused(
                    f"scale_kd.seqs_per_forward {self.seqs_per_forward}: at most {MAX_SEQS_PER_FORWARD} (the plan "
                    "does not estimate training memory, and larger groups have not been run)"
                )
            other = batched_rope(ctx.config, tcfg)
            if other:
                raise Refused(
                    f"scale_kd.seqs_per_forward {self.seqs_per_forward}: rotary type {other!r} (of the "
                    f"{'student' if other in _rope_types(ctx.config) else 'teacher'}) picks its frequencies from "
                    "the padded length, so a row would train otherwise than alone; use 1"
                )
        if any(lg.banks for lg in ctx.graph.layers):
            ctx.check_banks()
            if ctx.plan_moe is None:
                raise Refused(f"scale_kd: no kernel plan serves routed experts in {ctx.fmt.name}")
        for kind in set(self.trainable) - {"scales"}:
            if not _float_params(ctx.meta, {kind}):
                raise Refused(f"scale_kd: trainable {kind!r}: the model has no such tensors")
        if self.data is not None:
            rows = self._check_vocab(self._lint(ctx.recipe, teacher, self.trace_rows()), tv)
            self._read = rows  # what training uses: the file is read and linted once per build
            rows = self._screen(ctx.recipe, teacher, rows)  # the calibration intake's filter
            self.select(self._as_traces(rows))  # every row read and checked; the selection training uses
        elif ctx.recipe.calib.pending(teacher) is None:
            from ..engine.calib import load_samples

            self.select((x.ids.tolist(), x.prompt_len) for x in load_samples(ctx.recipe.calib, teacher))
        elif ctx.packed is not None:
            raise Refused(
                f"scale_kd: the calibration rows ({ctx.recipe.calib.label}) are not generated yet, and a build from"
                " a packed checkpoint has no generate stage; generate them with a float build, or set scale_kd.data"
            )
        # else: the float stage's generate stage makes the rows; training checks them (`_rows`) after it ran

    def checkpoint_peak(self, ctx: P.CheckpointPlan) -> StagePeak | tuple[StagePeak, ...]:
        """Estimate the training peak on the step's card from metadata (see `StagePeak`).

        Terms: the teacher in bf16 (`apply_checkpoint` loads it so); the student as `engine.packed.load_student`
        holds it on the card, not as the packed checkpoint is written (every quantized weight is unpacked to int8
        codes, so the card holds about twice the file's bytes, plus a bf16 scale, an fp32 delta and an int8 zero
        point per group, and the model's own buffers in their dtypes), including a packed head (a model step's
        format); at NVFP4 and FP8 the served weights it decodes once (bf16, or fp32 where the kernel cuts its
        input); the inputs of every checkpointed decoder block; one block rebuilt in backward; the head's fp32
        log-probs; and, per trained scale, its gradient and Adam's two moments (the zero points are frozen), plus
        Adam's fp32 step scalar per trained tensor. The forward pads `seqs_per_forward`
        sequences to the longest (`resolve`), so the activations follow that resolved group, not `batch_tokens`.
        A tied head is tied before materializing, so the student term also covers the peak of its load. Allocator
        caching, `engine.served_modules`'s dequant transients and the writer's buffers are not modelled, so the stage total
        is a floor, and it — not the student term alone — is what covers a real load.

        With two devices the teacher runs on the second (`apply_checkpoint`), so the two cards are sized apart and
        the step returns one `StagePeak` per card.
        """
        from ..engine.residency import free_bytes
        from ..solve.units import units_of

        dev = ctx.devices[0]
        text = ctx.config.get_text_config()
        hidden, vocab = text.hidden_size, text.vocab_size
        inter = getattr(text, "intermediate_size", 0) or getattr(text, "moe_intermediate_size", hidden)
        # Rows come from the calibration source (cut to `seqlen`) or from `data` (cut to `max_len`; its lengths
        # are not read at plan time, so the bound is one full `max_len` per sequence).
        seqlen = min(self.max_len, ctx.recipe.calib.seqlen if self.data is None else self.max_len)
        from ..engine.served_modules import decodes_once

        tokens = (self.seqs_per_forward or 1) * seqlen
        params = sum(p.numel() for p in ctx.meta.parameters())
        buffers = sum(b.numel() * b.element_size() for b in ctx.meta.buffers())
        group, groups, quantized, tensors, banked = ctx.fmt.group_size, 0, 0, 0, 0
        for lg in ctx.graph.layers:
            for u in units_of(ctx.meta.get_submodule(lg.module), lg):
                rows, cols = u.shape()
                experts = u.n_experts() if u.part is not None else 1
                quantized += rows * cols * experts
                groups += _groups(rows, cols, group) * experts
                # one served scale per stored Linear; a bank's experts share one per fused parameter
                tensors += 1 if u.part is not None else len(u.names)
                banked += rows * cols * experts if u.part is not None else 0
        head = _packed_head(ctx)  # `load_student` serves a packed head; its scale trains too
        head_params = head[0] * head[1] if head is not None else 0
        tensors += int(head is not None)
        terms = partial(
            kd_terms,
            layers=len(ctx.graph.layers),
            tokens=tokens,
            hidden=hidden,
            inter=inter,
            vocab=vocab,
            head_chunk=self.head_chunk,
            quantized=quantized,
            floats=params - quantized - head_params,
            groups=groups,
            symmetric=ctx.fmt.symmetric,
            head=head,
            buffers=buffers,
            tensors=tensors,
            # NVFP4 and FP8 decode each served weight once (`decodes_once`): bf16, or fp32 where the kernel cuts
            # its input, except a bank's, which stays bf16 (its grouped matmul's dtype); other formats hold none
            served=((quantized - banked) * (2 if ctx.fmt.act == "bf16" else 4) + banked * 2)
            if decodes_once(ctx.fmt)
            else 0,
        )
        student = StagePeak(self.name, str(dev), free_bytes(dev), _largest_first(terms(teacher=0)))
        if len(ctx.devices) == 1:
            return StagePeak(
                self.name,
                str(dev),
                free_bytes(dev),
                _largest_first(terms(teacher=params * 2)),
            )
        tdev = ctx.devices[1]
        teacher = kd_teacher_terms(
            teacher=params * 2,
            tokens=tokens,
            hidden=hidden,
            inter=inter,
            vocab=vocab,
            head_chunk=self.head_chunk,
        )
        return student, StagePeak(self.name, str(tdev), free_bytes(tdev), _largest_first(teacher))

    # ------------------------------------------------------------ pieces
    def _teacher(self, default: str | None, *, weights: bool = True) -> str:
        """The teacher's local directory: `teacher`, else `default` (the original model the input records).

        A hub repo id (``org/name`` or ``org/name@rev``) resolves to its snapshot as the build's source does
        (`engine.planning.local_source`; ``HF_HUB_OFFLINE`` uses the cache only). Refused if neither resolves.
        """
        from ..engine.planning import local_source

        t = self.teacher or default
        if t:
            try:
                return local_source(t, weights=weights)
            except Refused:
                pass
        raise Refused(f"scale_kd: teacher {t!r} not found; set scale_kd.teacher to the float model")

    def _lr(self, step: int) -> float:
        """Old scaletune's schedule: linear warmup, then cosine from `lr` to ``lr * lr_min_frac``."""
        if step < self.warmup:
            return self.lr * (step + 1) / self.warmup
        prog = (step - self.warmup) / max(1, self.steps - self.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return self.lr * (self.lr_min_frac + (1 - self.lr_min_frac) * cos)

    def _rows(self, run: P.CheckpointRun, teacher_dir: str) -> list[tuple[list[int], int]]:
        """(ids, P) per sequence; positions P-1 .. len-2 are scored (each predicts the next token)."""
        if self.data is not None:
            read = (
                self._read
                if self._read is not None
                else self._lint(run.recipe, teacher_dir, self.trace_rows())
            )
            rows = self._screen(run.recipe, teacher_dir, read, log=run.log, record=False)
            return self.select(self._as_traces(rows))
        from ..engine.calib import load_samples

        return self.select(
            (s.ids.tolist(), s.prompt_len) for s in load_samples(run.recipe.calib, teacher_dir)
        )

    def trace_rows(self) -> list[dict]:
        """Every row of `data` (a ``{prompt_ids, completion_ids}`` trace).

        Raises:
            Refused: `data` is missing, or a row is not a trace ``{prompt_ids, completion_ids}``.
        """
        if self.data is None:
            raise Refused("scale_kd.data: no trace file is set")
        path = Path(self.data)
        if not path.is_file():
            raise Refused(f"scale_kd.data {self.data}: no such file")
        out = []
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict) or not {"prompt_ids", "completion_ids"} <= set(r):
                raise Refused(
                    f"scale_kd.data {self.data}: line {n} is not a trace {{prompt_ids, completion_ids}}"
                )
            out.append(r)
        return out

    def _check_vocab(self, rows: list[dict], vocab_size: int) -> list[dict]:
        """`rows`, after every id of every trace was found an integer in the model's vocabulary ``[0, vocab_size)``.

        Raises:
            Refused: a trace holds an id that is not an integer, or one outside the vocabulary (its row and id named).
        """
        for n, r in enumerate(rows):
            for key in ("prompt_ids", "completion_ids"):
                ids = r[key]
                if not isinstance(ids, list):
                    raise Refused(
                        f"scale_kd.data {self.data}: row {n} (counted from 0): {key} is not a list of token ids"
                    )
                at = next((j for j, i in enumerate(ids) if not _is_id(i, vocab_size)), None)
                if at is not None:
                    raise Refused(
                        f"scale_kd.data {self.data}: row {n} (counted from 0) holds token ids outside the model's "
                        f"vocab [0, {vocab_size}): {key}[{at}] = {ids[at]!r}"
                    )
        return rows

    def _lint(self, recipe: Any, teacher: str, rows: list[dict]) -> list[dict]:
        """`rows`, after the data lint passed them (:func:`qda_quant.calib.template.lint_rows`, ids decoded).

        Its record joins the calibration's in build.json's ``data_lint`` and the build log's lint block.

        Raises:
            Refused: a trace is written in another model family's chat template.
        """
        from ..calib.template import lint_rows
        from ..intake.tokenizer import load_tokenizer

        label = f"scale_kd.data {self.data}"
        found = lint_rows(rows, load_tokenizer(teacher), label)
        if (why := found.refusal(label)) is not None:
            raise Refused(why)
        recipe.calib.note_lint(found.lint_record())
        return rows

    def _as_traces(self, rows: Iterable[dict]) -> list[tuple[list[int], int]]:
        """The rows as ``(prompt + completion, len(prompt))``."""
        return [(list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])) for r in rows]

    def traces(self) -> list[tuple[list[int], int]]:
        """Every row of `data` as ``(prompt + completion, len(prompt))``."""
        return self._as_traces(self.trace_rows())

    def _screen(
        self, recipe: Any, teacher: str, rows: list[dict], log: Any = None, record: bool = True
    ) -> list[dict]:
        """The KD rows the calibration contamination filter keeps (the same filter and log as calibration).

        The rows screened are exactly the rows this step reads (``data``): its prompt side (``prompt_ids``,
        decoded) is compared, never the completion. The source's own ``screen`` is not used: for a selfgen
        calibration source that would screen its unrelated prompt list instead of these traces. `record` adds the drops to the build record (plan time does; the training pass
        re-reads the file and only applies them).

        Raises:
            Refused: every row matched a reference prompt.
        """
        calib = recipe.calib
        if calib.index() is None:
            return rows
        from ..intake.tokenizer import load_tokenizer

        seen, seen_x = len(calib.dropped), len(calib.exempted)
        kept = calib.apply_index(rows, load_tokenizer(teacher), record=record)
        dropped = calib.dropped[seen:]
        exempted = calib.exempted[seen_x:]
        if log is not None and record:
            log(
                f"scale_kd.data {self.data}: contamination filter: {len(rows)} rows, {len(dropped)} dropped, "
                f"{len(exempted)} exempted; compared: prompts only, never answer bodies"
            )
        if not kept:
            ids = ", ".join(str(d["id"]) for d in dropped[:20])
            raise Refused(
                f"scale_kd.data {self.data}: every row matched the reference prompts ({len(dropped)} rows: "
                f"{ids}); use traces the evaluation does not use"
            )
        return kept

    def select(self, seqs: Iterable[tuple[list[int], int]]) -> list[tuple[list[int], int]]:
        """The sequences training uses, under the rules `check_checkpoint` applies at plan time too.

        Each sequence is cut to ``max_len`` and kept when its prompt has 1 or more ids and 2 or more scored tokens
        are left; the first ``max_seqs`` kept are used (0 = all).

        Raises:
            Refused: no sequence is left.
        """
        rows = []
        for ids, p in seqs:
            ids = ids[: self.max_len]
            if p >= 1 and len(ids) - p >= 2:
                rows.append((ids, p))
                if self.max_seqs and len(rows) >= self.max_seqs:
                    break
        if not rows:
            src = f"scale_kd.data {self.data}" if self.data is not None else "scale_kd (calibration rows)"
            raise Refused(f"{src}: no sequence with 2 or more scored tokens within max_len {self.max_len}")
        return rows

    def _params(self, student: torch.nn.Module, served: dict) -> tuple[list[dict], list, list[str]]:
        """Optimizer groups, (fp32 master, bf16 parameter) pairs of the trained float tensors, and their names."""
        groups: list[dict] = []
        if "scales" in self.trainable:
            for m in served.values():
                m.delta.requires_grad_(True)
            groups.append(
                {"name": "scales", "params": [m.delta for m in served.values()], "base_lr": self.lr}
            )
        picked = _float_params(student, set(self.trainable) - {"scales"})
        masters, names = [], []
        for kind, lr in (("norms", self.norm_lr), ("gdn", self.gdn_lr)):
            ps = [(n, p) for n, p, k in picked if k == kind]
            if not ps:
                if kind in self.trainable:
                    raise Refused(f"scale_kd: trainable {kind!r}: the model has no such tensors")
                continue
            ms = []
            for n, p in ps:
                p.requires_grad_(True)
                mp = p.detach().float().clone().requires_grad_(True)
                masters.append((mp, p))
                ms.append(mp)
                names.append(n)
            groups.append({"name": kind, "params": ms, "base_lr": lr})
        return groups, masters, names
