"""Exchange routed tokens and gradients between expert ranks."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterator


class AllToAllTokens(torch.autograd.Function):
    """Send rows of `x` to `world` cards by split size, and send the gradient of what arrived back the same way.

    A verbatim port of upstream `src/moe/autograd_ops.AllToAllTokens` (Apache-2.0, IST-DASLab/GSQ `03fc1648`).
    """

    @staticmethod
    def forward(  # noqa: D102  # upstream's, verbatim
        ctx: Any, x: torch.Tensor, out_split: list[int], in_split: list[int], group: Any
    ) -> torch.Tensor:  # noqa: ANN401
        ctx.group, ctx.out_split, ctx.in_split = group, out_split, in_split
        y = x.new_empty((sum(out_split), x.shape[-1]))
        dist.all_to_all_single(y, x, output_split_sizes=out_split, input_split_sizes=in_split, group=group)
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple:  # noqa: ANN401, D102  # upstream's, verbatim
        grad_x = grad_y.new_empty((sum(ctx.in_split), grad_y.shape[-1]))
        dist.all_to_all_single(
            grad_x,
            grad_y.contiguous(),
            output_split_sizes=ctx.in_split,
            input_split_sizes=ctx.out_split,
            group=ctx.group,
        )
        return grad_x, None, None, None


def owners(experts: int, world: int, device: torch.device) -> torch.Tensor:
    """The owning card of every expert: `ExpertShare.experts`' blocks, not `e * world // experts`."""
    own = torch.empty(experts, dtype=torch.long)
    for r in range(world):
        own[experts * r // world : experts * (r + 1) // world] = r
    return own.to(device)


def forward(  # noqa: A001  # the module's own entry point, named as the module it replaces
    module: Any,  # noqa: ANN401  # a fused expert module: num_experts, up_proj/down_proj, act_fn, has_gate
    hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    first: int,
    stop: int,
    world: int,
    group: Any,  # noqa: ANN401
) -> torch.Tensor:
    """The routed experts' contribution for `hidden` — this card's experts only, every card's tokens.

    `hidden` is ``[T, H]`` for this card's tokens and `top_k_index`/`top_k_weights` are ``[T, k]``; the result is
    ``[T, H]``, weighted and summed over the pairs, as the fused expert module gives it. `module.up_proj` and
    `module.down_proj` hold every expert — the functional call hands them over whole — but only rows
    ``[first, stop)`` are read, so only their gradients are produced.
    """
    device = hidden.device
    rank = dist.get_rank(group)
    t, _ = hidden.shape
    k = top_k_index.shape[1]
    ids = top_k_index.reshape(-1).to(torch.long)
    weights = top_k_weights.reshape(-1).to(hidden.dtype)
    tokens = torch.arange(t, device=device).repeat_interleave(k)
    owner = owners(module.num_experts, world, device)[ids]
    # Sort by (expert, token): each card's experts are ONE contiguous block of ids (`owners`), so this is also
    # owner-major in rank order, which is what the all-to-all splits need; and per token it makes the add-back
    # order match the fused module's expert-ordered accumulation, so world=1 is bit-for-bit the fused forward.
    order = torch.argsort(ids * t + tokens, stable=True)
    ids, weights, tokens = ids[order], weights[order], tokens[order]
    send = torch.bincount(owner, minlength=world).tolist()
    sizes = torch.tensor(send, device=device, dtype=torch.long)
    got = [torch.empty_like(sizes) for _ in range(world)]
    dist.all_gather(got, sizes, group=group)
    recv = torch.stack(got)[:, rank].tolist()

    xin = AllToAllTokens.apply(hidden[tokens], recv, send, group)
    eid = (
        AllToAllTokens.apply(ids.to(torch.float32).unsqueeze(1), recv, send, group).squeeze(1).to(torch.long)
    )
    win = AllToAllTokens.apply(weights.unsqueeze(1), recv, send, group)
    if xin.shape[0]:
        local, inverse = torch.unique(eid, sorted=True, return_inverse=True)
        if int(local.min()) < first or int(local.max()) >= stop:
            raise RuntimeError(
                f"card {rank} received experts {local.min().item()}..{local.max().item()}, not {first}..{stop}"
            )
        by_expert = torch.argsort(inverse, stable=True)
        rows = torch.bincount(inverse, minlength=local.numel()).tolist()
        offset = 0
        parts = []
        for i, count in enumerate(rows):
            if count:
                part = xin.index_select(0, by_expert[offset : offset + count])
                e = int(local[i])  # the bank is whole on the card: index it by the global expert id
                parts.append(F.linear(module.act_fn(F.linear(part, module.up_proj[e])), module.down_proj[e]))
                offset += count
        out = (torch.cat(parts) if parts else xin.new_empty((0, xin.shape[-1]))).index_select(
            0, torch.argsort(by_expert, stable=True)
        )
    else:
        out = xin
    back = AllToAllTokens.apply(out * win, send, recv, group)
    return hidden.new_zeros((t, hidden.shape[-1])).index_add(0, tokens, back)


@contextlib.contextmanager
def expert_parallel(module: Any, first: int, stop: int, world: int, group: Any) -> Iterator[None]:  # noqa: ANN401
    """Run `module` as this card's experts only, every card's tokens reaching them, for the block's duration."""
    original = module.forward

    def patched(
        hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        return forward(module, hidden_states, top_k_index, top_k_weights, first, stop, world, group)

    module.forward = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = original  # type: ignore[method-assign]
