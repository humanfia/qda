"""Gather rank statistics and broadcast quantized layers."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist

from .runner import GatheredLayerIO, to_device, tree
from .stats import DEFAULT_INPUT_ROWS


def free_port() -> int:
    """A free TCP port for a rendezvous that listens on every local address (`TCPStore`, hence a process group).

    The probe binds the wildcard address the store binds, not `127.0.0.1`: a port another process holds on one
    local address only (say `127.0.1.1`) passes a `127.0.0.1` probe and then fails the store's listen with
    `EADDRINUSE`. The socket closes before the store binds, so this still only narrows the race, never closes it.
    """
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def shard(n: int, rank: int, world: int) -> list[int]:
    """Indices of the batches rank `rank` runs: contiguous, the same split as the in-process path."""
    return [i for i in range(n) if i * world // n == rank]


class Comm:
    """Rank `rank` of `world` processes in an initialized NCCL group, on `device`."""

    def __init__(self, rank: int, world: int, device: torch.device) -> None:
        self.rank, self.world, self.device = rank, world, device

    @property
    def root(self) -> bool:
        """True on rank 0 (solves, writes, logs)."""
        return self.rank == 0

    def _gather(self, t: torch.Tensor) -> list[torch.Tensor] | None:
        """Rank 0 gets every rank's `t` (same shape on all ranks), in rank order."""
        t = t.contiguous().to(self.device)
        out = [torch.empty_like(t) for _ in range(self.world)] if self.root else None
        dist.gather(t, out, dst=0)
        return out

    def _gather_rows(
        self, t: torch.Tensor | list[torch.Tensor] | None, like: tuple[int, torch.dtype]
    ) -> tuple[list[torch.Tensor] | None, list[int]]:
        """Gather row-stacked tensors whose row counts may differ between ranks, and each rank's row count.

        `t` is one tensor or a list of per-batch tensors; a list is copied into this rank's pad batch by batch,
        so a rank never stacks its own rows into a second copy first. Rank 0 gets one tensor per
        rank, `top` rows each: a shorter shard's tail is unused, and `g[:n]` is a view.
        """
        cols, dtype = like
        batches = [] if t is None else (list(t) if isinstance(t, (list, tuple)) else [t])
        rows = sum(int(b.shape[0]) for b in batches)
        n = torch.tensor([rows], device=self.device)
        ns = [torch.empty_like(n) for _ in range(self.world)]
        dist.all_gather(ns, n)
        counts = [int(v) for v in ns]
        pad = torch.zeros(max(counts), cols, dtype=dtype, device=self.device)
        off = 0
        for b in batches:
            x = b.reshape(-1, cols)
            pad[off : off + x.shape[0]] = x
            off += x.shape[0]
        got = self._gather(pad)
        return (None if got is None else [g[:k] for g, k in zip(got, counts, strict=True)]), counts

    def all_rows(self, t: torch.Tensor) -> torch.Tensor:
        """Every rank's rows of `t`, in rank order, on every rank.

        A row-sharing step (`Step.shares_rows`) gives each rank its own calibration batches; this returns the
        ranks' per-batch entries in the order a one-card build holds the batches (`engine.dist.shard` gives a
        rank a contiguous span), so summing or stacking them here is the one-card arithmetic, term for term —
        the reduction is not merely close to it. A rank with no rows of its own passes a 0-row tensor; the
        returned tensor has every rank's rows and the same shape on every rank.
        """
        t = t.contiguous().to(self.device)
        n = torch.tensor([t.shape[0]], device=self.device)
        ns = [torch.empty_like(n) for _ in range(self.world)]
        dist.all_gather(ns, n)
        counts = [int(v) for v in ns]
        pad = torch.zeros(max(counts), *t.shape[1:], dtype=t.dtype, device=self.device)
        pad[: t.shape[0]] = t
        got = [torch.empty_like(pad) for _ in range(self.world)]
        dist.all_gather(got, pad)
        return torch.cat([g[:k] for g, k in zip(got, counts, strict=True)])

    def _merge_inputs(
        self,
        t: torch.Tensor | list[torch.Tensor] | None,
        like: tuple[int, torch.dtype],
        per_rank: list[list[int]] | None,
        cap: int | None,
    ) -> Any:
        """The ranks' retained input rows in rank order, on rank 0 (None elsewhere).

        A step that re-runs the layer (`per_rank` names each rank's batch sizes) gets the batches back as views
        into the gathered shards: the gather itself is the sample's only copy, where the pairwise
        `merge`/`torch.cat` needed one per rank and held two again for the last one. Every other
        consumer gets one tensor with the shards' unused tails dropped.
        """
        got, counts = self._gather_rows(t, like)
        if got is None:
            return None
        total = sum(counts)
        if (
            per_rank is not None
            and all(c == sum(rs) for c, rs in zip(counts, per_rank, strict=True))
            and (cap is None or total <= cap)
        ):
            return [b for g, rows in zip(got, per_rank, strict=True) for b in torch.split(g, rows)]
        rows = torch.cat(got)
        return rows if cap is None else rows[:cap]

    def reduce_stats(
        self,
        stats: dict[tuple[str, str], Any],
        input_rows: int | None = DEFAULT_INPUT_ROWS,
        gather_layer_io: bool = False,
        keep_local: frozenset[str] = frozenset(),
    ) -> dict[tuple[str, str], Any] | None:
        """Combine every rank's statistics on rank 0 (None elsewhere); keys are the same on all ranks.

        With `gather_layer_io`, the ranks' `layer_io` batches are merged in rank order and the input samples
        come back split into those batches, as a one-card build hands them to a step that re-runs the layer.
        The input samples are gathered once, into the ranks' row ranges, and those batches are views into the
        gather: rank 0 never concatenates the sample again.

        Tensor statistics on rank 0 become the accumulators: this consumes their local values. One receive
        buffer is reused across ranks and released before the next leaf, with the same left-to-right
        sum/max as `runner.merge`. Other ranks keep their tensors for the subsequent expert broadcast.

        `keep_local` names statistics (`"inputs"`, `"layer_io"`) a row-sharing step combines itself: they are
        neither gathered nor freed, and rank 0's reduced dict carries this rank's own entry for them.
        """
        out: dict[tuple[str, str], Any] = {}
        io_key = next((k for k in sorted(stats) if k[0] == "layer_io"), None) if gather_layer_io else None
        io_merged: Any = None
        shard_rows: list[list[int]] | None = None
        # first: batches that cannot be row-stacked refuse here, not in the gather below
        if io_key is not None:
            io_merged, _, shard_rows = self.gather_layer_io(stats[io_key])
        # group inputs a one-card build keeps apart, one list entry a batch; a rank with no batch of its own
        # still knows its share from the ranks that have one (a bank's all-token inputs are tensors and stay so)
        batched = [k for k in sorted(stats) if k[0] == "inputs" and isinstance(stats[k], (list, tuple))]
        batch_keys = {k for ks in self.gather_objects(batched) or [] for k in ks}
        local = {k for k in stats if k[0] in keep_local}  # a row-sharing step keeps and combines its own
        merged_inputs: dict[tuple[str, str], Any] = {}
        for key in sorted(stats):
            if key in local or key[0] in ("layer_io", "outputs"):
                continue
            if key[0] == "inputs":
                # every rank's rows in rank order, gathered once: views into the shards for a step that
                # re-runs the layer, one tensor otherwise
                merged_inputs[key] = self._merge_inputs(
                    stats[key],
                    _row_like(stats[key], stats),
                    shard_rows if key in batch_keys else None,
                    input_rows,
                )
                stats[key] = None  # this rank's shard is no longer needed: free it before the next gather
                continue
            got = [(path, self._fold(key[0], value)) for path, value in _leaves(stats[key])]
            if self.root:
                out[key] = _unleaves(stats[key], got)
        if not self.root:
            return None
        out.update(merged_inputs)
        out.update({k: stats[k] for k in local})
        for k, v in stats.items():
            if k[0] in ("layer_io", "outputs") and k not in local:
                out[k] = v
        if io_key is not None:
            out[io_key] = io_merged
        return out

    def _fold(self, name: str, value: Any) -> Any:
        tensor = isinstance(value, torch.Tensor)
        t = value.to(self.device) if tensor else torch.tensor([int(value)], device=self.device)
        if not self.root:
            dist.send(t.contiguous(), dst=0)
            return None
        result: Any = t if tensor else int(value)
        if self.world > 1:
            received = torch.empty_like(t, memory_format=torch.contiguous_format)
            for rank in range(1, self.world):
                dist.recv(received, src=rank)
                if not tensor:
                    result += int(received)
                elif name == "act_absmax":
                    torch.maximum(result, received, out=result)
                else:
                    result.add_(received)
        return result

    def gather_layer_io(
        self, io: Any
    ) -> tuple[GatheredLayerIO | None, list[int] | None, list[list[int]] | None]:
        """Merge every rank's `layer_io` batches on rank 0, in rank order, and name each batch's rows.

        The ranks' hidden states move in batch order; a calibration whose batches differ in shape or call
        (a variable sequence length, image samples) refuses on every rank rather than pricing a mix. Rank 0
        also gets each rank's batch sizes (the order's split by rank), which the input gather uses to hand a
        re-run step the same batches without concatenating the ranks' rows.

        A batch's float target (the `layer_io` item's fourth element, what GSQ fits) travels with it when
        every batch has one: rank 0 then holds the one-card ``(hidden, args, kwargs, target)`` list, so a
        gathered step fits exactly the rows a one-card build fits. A step whose batches carry no target
        gathers the hidden states alone.
        """
        batches = list(io) if io is not None else []
        rows = [h.numel() // h.shape[-1] for h, _, _, _ in batches]
        shape = tuple(batches[0][0].shape) if batches else None
        sample = (_plain(batches[0][1]), _plain(batches[0][2])) if batches else None
        uniform = batches and all(
            tuple(h.shape) == shape and _same((args, kw), (batches[0][1], batches[0][2]))
            for h, args, kw, _ in batches
        )
        local = {
            "uniform": bool(uniform),
            "shape": shape,
            "dtype": batches[0][0].dtype if batches else None,
            "sample": sample,
            "rows": rows,
            "target": _target_meta(batches),
        }
        boxes = cast(list[dict[str, Any] | None], [None] * self.world) if self.root else None
        dist.gather_object(local, boxes, dst=0)  # the group's device: CPU (gloo), the card (NCCL)
        box: list[Any] = [None]
        if self.root and boxes is not None:
            box[0] = self._layer_io_verdict(boxes)
        dist.broadcast_object_list(box, src=0, device=self.device)
        ok, message, shape, dtype, target = box[0]
        target_shape: tuple[int, ...] | None = None if target is None else tuple(target[0])
        target_edge = 0 if target_shape is None else target_shape[-1]
        if not ok:
            raise RuntimeError(message)
        count = torch.tensor([sum(rows)], device=self.device)
        counts = [torch.empty_like(count) for _ in range(self.world)]
        dist.all_gather(counts, count)
        top = max(int(c) for c in counts)
        width = shape[-1]
        pad = torch.zeros(top, width, dtype=dtype, device=self.device)
        off = 0
        for h, _, _, _ in batches:
            flat = h.reshape(-1, width)
            pad[off : off + flat.shape[0]] = flat
            off += flat.shape[0]
        got = self._gather(pad)
        target_pad = None
        if target is not None:  # the float targets, packed like the hidden states, one gather
            _, tdtype = target
            target_pad = torch.zeros(top, target_edge, dtype=tdtype, device=self.device)
            off = 0
            for item in batches:
                flat = cast(torch.Tensor, item[3]).reshape(-1, target_edge)
                target_pad[off : off + flat.shape[0]] = flat
                off += flat.shape[0]
        got_targets = None if target_pad is None else self._gather(target_pad)
        if not self.root:
            return None, None, None
        assert got is not None and boxes is not None
        merged: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        order: list[int] = []
        by_rank: list[list[int]] = []
        for r, (g, n) in enumerate(zip(got, counts, strict=True)):
            flat = g[: int(n)]
            start = 0
            b = boxes[r]
            here = list(b["rows"]) if b is not None else []
            by_rank.append(here)
            for k in here:
                merged.append(flat[start : start + k].reshape(shape))
                if got_targets is not None and target_shape is not None:
                    targets.append(got_targets[r][start : start + k].reshape(target_shape))
                start += k
                order.append(k)
        ref = next((b for b in boxes if b and b["rows"]), None)
        if ref is None:  # the verdict refused an empty gather above
            raise RuntimeError("layer_io gather: no calibration batches on any card")
        args, kwargs = ref["sample"]
        return (
            GatheredLayerIO(
                merged,
                to_device(args, self.device),
                to_device(kwargs, self.device),
                self.device,
                targets if got_targets is not None else None,
            ),
            order,
            by_rank,
        )

    def _layer_io_verdict(self, boxes: list[dict[str, Any] | None]) -> tuple[bool, str, Any, Any, Any]:
        ref = next((b for b in boxes if b and b["rows"]), None)
        message = (
            "multi-card builds gather every card's calibration batches for steps that re-run a layer; the "
            "batches must all call the layer with the same shape and arguments (calibration batches that "
            "differ need one card)"
        )
        if ref is None:
            return False, "layer_io gather: no calibration batches on any card", None, None, None
        mixed = any(
            b
            and b["rows"]
            and (not b["uniform"] or b["shape"] != ref["shape"] or not _same(b["sample"], ref["sample"]))
            for b in boxes
        )
        if mixed:
            return False, message, None, None, None
        # a step that fits the float target needs every rank's targets, of the one shape: a card that kept
        # none would fit a different list on rank 0, so refuse the mix rather than the wrong list
        meta = [b["target"] if b is not None else None for b in boxes]
        have = [m for m, b in zip(meta, boxes, strict=True) if b is not None and b["rows"]]
        if any(m is not None for m in have):
            shapes = {m for m in have if m is not None}
            if len(shapes) != 1 or any(m is None for m in have):
                return (
                    False,
                    "layer_io gather: some cards kept the float target and some did not, or at two shapes; "
                    "a step that fits the target needs all of them at one shape on one card",
                    None,
                    None,
                    None,
                )
        return True, "", ref["shape"], ref["dtype"], have[0] if have else None

    def broadcast_stats(self, stats: dict[tuple[str, str], Any], keys: list[tuple[str, str]]) -> None:
        """Replace `keys` of `stats` on every rank with rank 0's values (its reduced statistics).

        Rank 0 passes its reduced statistics, the others their own; the shapes come from rank 0.
        """
        for key in sorted(keys):
            leaves = _leaves(stats[key])
            meta = (
                [(p, (tuple(v.shape), v.dtype) if isinstance(v, torch.Tensor) else None) for p, v in leaves]
                if self.root
                else None
            )
            box = [meta]
            dist.broadcast_object_list(box, src=0, device=self.device)
            assert box[0] is not None  # rank 0 fills the metadata for every rank
            got = []
            for (path, m), (_, v) in zip(box[0], leaves, strict=True):
                if m is None:  # python int token count
                    t = torch.tensor([int(v) if self.root else 0], device=self.device)
                    dist.broadcast(t, src=0)
                    got.append((path, int(t)))
                    continue
                t = (
                    v.contiguous().to(self.device)
                    if self.root
                    else torch.empty(m[0], dtype=m[1], device=self.device)
                )
                dist.broadcast(t, src=0)
                got.append((path, t))
            stats[key] = _unleaves(stats[key], got)

    def gather_experts(self, qparams: dict[str, dict[str, Any]], layer: torch.nn.Module, prefix: str) -> None:
        """Assemble every expert bank solved in shares (`units.ExpertShare`) on rank 0, in rank order.

        Each rank's qparams entry of a bank holds its experts' codes / scales (``"experts": (start, stop)``)
        and its parameter holds their served weights in those rows. Rank 0 ends with the whole bank in both,
        as a 1-card solve leaves them; the other ranks drop their entries (their layer is overwritten by
        `broadcast_module`).
        """
        from ..intake.fingerprint import tensor_digest
        from ..solve.units import QPARAM_TENSORS
        from .runner import rel

        for name in sorted(k for k, v in qparams.items() if k.startswith(prefix + ".") and "experts" in v):
            qp = qparams[name]
            p = layer.get_parameter(rel(name, prefix))
            n = p.shape[0]
            spans = [(n * r // self.world, n * (r + 1) // self.world) for r in range(self.world)]
            if tuple(qp["experts"]) != spans[self.rank]:
                raise RuntimeError(
                    f"{name}: rank {self.rank} solved experts {qp['experts']}, not {spans[self.rank]}"
                )
            lo, hi = spans[self.rank]
            top = max(b - a for a, b in spans)
            full = {}
            for field, t in [("weight", p.data[lo:hi])] + [(f, qp[f]) for f in QPARAM_TENSORS if f in qp]:
                pad = torch.zeros((top, *t.shape[1:]), dtype=t.dtype, device=self.device)
                pad[: t.shape[0]] = t
                got = self._gather(pad)
                if self.root:
                    assert got is not None  # _gather returns the gathered tensors on rank 0
                    full[field] = torch.cat([g[: b - a] for g, (a, b) in zip(got, spans, strict=True)])
            if not self.root:
                del qparams[name]
                continue
            p.data.copy_(full.pop("weight"))
            qparams[name] = {**full, "format": qp["format"], "weight_check": (id(p), tensor_digest(p))}

    def broadcast_module(self, module: torch.nn.Module) -> None:
        """Copy rank 0's parameters and buffers of `module` (a solved layer, on the card) to every rank."""
        for name, t in sorted(module.state_dict(keep_vars=True).items()):
            if t.device != self.device:
                raise RuntimeError(
                    f"cannot broadcast {name} on {t.device}: the layer must be on {self.device}"
                )
            dist.broadcast(t.data, src=0)

    def average_grads(self, optimizer: Any, only: set[int] | None = None) -> None:
        """Sum every parameter's gradient over the ranks and divide by the world.

        Upstream GSQ's `average_grads` (`src/trainer.py` at 03fc16484c): a data-parallel fit prices its own
        calibration batches on every rank, so the step it takes is the mean of the ranks' gradients. `only`
        names the `id`s of the parameters to reduce: an expert-parallel layer also trains a bank whose rows
        belong to the card that owns them (its loss sums its own tokens' contributions), so it averages its
        DENSE groups alone -- every rank must train the same set, or the collectives disagree and hang.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and (only is None or id(p) in only):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world)

    def min_across(self, value: int) -> int:
        """The smallest `value` over the ranks: a collective step must take the same count on every rank."""
        t = torch.tensor(value, dtype=torch.long, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())

    def broadcast_qparams(
        self, qparams: dict[str, dict[str, torch.Tensor]], skip_experts: bool = False
    ) -> None:
        """Copy rank 0's quantization parameters (codes, scales, zero points) to every rank.

        A row-sharing step that refines an earlier step's codes (`gsq` after `gptq`/`rtn`) starts from the
        fold rank 0 wrote: every card must begin the fit from the same codes and scales, or the gradients it
        averages price different weights. The entries are codes and scales, not weights, so they travel as
        pickled host tensors. `skip_experts` leaves the expert banks out: an expert-parallel solve keeps each
        card's own bank entries (its experts, and `gather_experts` checks the span), and takes only the
        layer's dense codes from rank 0.
        """
        payload: dict[str, dict[str, Any]] | None = None
        if self.root:
            # `weight_check` is a per-process parameter identity (id + digest): it cannot cross ranks, and a
            # card that did not run the writing step has no parameter it could name
            payload = {
                name: {
                    key: (v.cpu() if isinstance(v, torch.Tensor) else v)
                    for key, v in group.items()
                    if key != "weight_check"
                }
                for name, group in qparams.items()
                if not (skip_experts and "experts" in group)
            }
        box: list[Any] = [payload]
        dist.broadcast_object_list(box, src=0, device=self.device)
        got = box[0]
        if got is not None and not self.root:
            qparams.update(got)

    def from_root(self, fn: Callable[[], Any]) -> Any:
        """Run `fn` on rank 0 only and give its result to every rank.

        If `fn` raises on rank 0, every rank raises (no rank is left waiting in a collective).
        """
        box: list[Any] = [None]
        if self.root:
            try:
                box[0] = ("ok", fn())
            except Exception as e:  # noqa: BLE001  (re-raised on every rank below)
                box[0] = ("error", f"{type(e).__name__}: {e}")
        dist.broadcast_object_list(box, src=0, device=self.device)
        status, value = box[0]
        if status == "error":
            raise RuntimeError(f"rank 0 failed: {value}")
        return value

    def gather_objects(self, obj: Any) -> list[Any] | None:
        """Small Python objects (timings) from every rank to rank 0."""
        out = [None] * self.world if self.root else None
        dist.gather_object(obj, out, dst=0)
        return out


def _leaves(v: Any) -> list[tuple[str | None, Any]]:
    if isinstance(v, dict):
        return [(k, v[k]) for k in sorted(v)]
    return [(None, v)]


def _unleaves(like: Any, leaves: list[tuple[str | None, Any]]) -> Any:
    if isinstance(like, dict):
        got = dict(leaves)
        return {k: got[k] for k in like}  # keep the original key order
    return leaves[0][1]


def _row_like(v: Any, stats: dict) -> tuple[int, torch.dtype]:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, torch.Tensor):
        return v.shape[-1], v.dtype
    for (name, _), h in stats.items():  # no rows on this rank: take the width from a Hessian
        if name == "hessian" and isinstance(h, torch.Tensor):
            return h.shape[-1], torch.bfloat16
    raise RuntimeError("cannot infer the width of an empty input sample")


def _target_meta(batches: list[Any]) -> Any:
    """``(shape, dtype)`` of this rank's float targets when every batch has one at that shape, else None.

    None means "no target to gather": a step that fits the target (`layer_io`'s fourth element) refuses the
    mix in `_layer_io_verdict` when another card kept one and this card did not.
    """
    outs = [item[3] for item in batches]
    if not batches or any(o is None for o in outs):
        return None
    shape = tuple(outs[0].shape)
    if any(tuple(o.shape) != shape for o in outs):
        return None
    return (shape, outs[0].dtype)


def _plain(v: Any) -> Any:
    """A call argument on the host: `gather_object` moves it between ranks; tensors keep their values."""
    return tree(v, lambda t: t.detach().to("cpu", copy=False))


def _same(a: Any, b: Any) -> bool:
    """Structural equality of two call arguments (tensors by shape and value)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b
