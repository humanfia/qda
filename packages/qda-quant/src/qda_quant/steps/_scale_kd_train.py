"""The training loop's pieces of ``scale_kd``: the shuffled batcher and the padded-group KL forward."""

from __future__ import annotations

import random
from typing import Any, cast

import torch


class Batcher:
    """Shuffled passes over the rows on one rank; a batch closes at ``>= batch_tokens``."""

    def __init__(self, lens: list[int], batch_tokens: int, seed: int):
        self.lens, self.bt, self.seed = lens, batch_tokens, seed
        self.epoch, self.pos = 0, 0
        self._perm()

    def _perm(self) -> None:
        rng = random.Random(self.seed * 1000003 + self.epoch)
        self.perm = list(range(len(self.lens)))
        rng.shuffle(self.perm)

    def next(self) -> list[int]:
        """Indices of the next batch."""
        batch, tok = [], 0
        while tok < self.bt:
            if self.pos >= len(self.perm):
                self.epoch, self.pos = self.epoch + 1, 0
                self._perm()
            batch.append(self.perm[self.pos])
            tok += self.lens[self.perm[self.pos]]
            self.pos += 1
        return batch


def kl_sequences(
    student: torch.nn.Module, teacher: torch.nn.Module, rows: list[tuple[list[int], int]], chunk: int, norm: int,
    device: torch.device, backward: bool,
) -> float:  # fmt: skip
    """KL(teacher || student) summed over positions ``p-1 .. len-2`` of each row ``(ids, p)``, in one forward.

    Rows are padded on the right to the longest. The models are causal, so no real position attends to a pad, and
    with position-fixed rotary embeddings (`BATCHED_ROPE`, checked at plan time) each scored position sees what it
    sees alone; pads are never scored. With `backward`, the gradient of that sum
    / `norm` is accumulated. The head runs in chunks on detached hidden states, then the trunk's backward runs once.

    `teacher` may sit on another card than `device`: its forward runs there and only each chunk's fp32 log-probs
    move to the student's card, an exact copy, so the sum is the same whichever card the teacher is on.
    """
    ids = torch.zeros((len(rows), max(len(r) for r, _ in rows)), dtype=torch.long)
    for b, (r, _) in enumerate(rows):
        ids[b, : len(r)] = torch.tensor(r)
    ids = ids.to(device)
    sls = [(b, slice(p - 1, len(r) - 1)) for b, (r, p) in enumerate(rows)]
    # transformers model attributes (base_model, ...) are dynamic on nn.Module.
    t, s = cast(Any, teacher), cast(Any, student)
    tdev = next(teacher.parameters()).device  # the teacher's own card, when the build has two
    with torch.no_grad():
        ht = t.base_model(input_ids=ids.to(tdev)).last_hidden_state
        ht = torch.cat([ht[b, sl] for b, sl in sls])
    t_head, s_head = t.get_output_embeddings(), s.get_output_embeddings()
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.set_grad_enabled(backward):
        h = s.base_model(input_ids=ids).last_hidden_state
        h = torch.cat([h[b, sl] for b, sl in sls])
        hd = h.detach().requires_grad_(backward)
        for a in range(0, hd.shape[0], chunk):
            with torch.no_grad():
                lpt = torch.log_softmax(t_head(ht[a : a + chunk]).float(), -1).to(device)
            lps = torch.log_softmax(s_head(hd[a : a + chunk]).float(), -1)
            kl = (lpt.exp() * (lpt - lps)).sum(-1)
            total += kl.detach().sum().double()
            if backward:
                (kl.sum() / norm).backward()
        if backward:
            h.backward(hd.grad)
    return float(total)
