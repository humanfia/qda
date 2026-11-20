"""Fisher map: what each unit's bit width costs in bits and is predicted to cost in KL, measured once per model.

``qda fisher map`` runs the calibration samples through the bf16 original and accumulates the diagonal Fisher
``F`` of every quantizable weight: the squared gradient of the next-token cross-entropy against labels drawn
from the model's own distribution, one draw per position (so ``F`` estimates the Fisher of the KL the judge
scores, not the empirical Fisher of the corpus text), averaged per token. Then, for every unit the build
solves and serves as one (a fused q/k/v, gate/up, a GDN input, each ``o``/``down``/``out_proj``, and each MoE
layer's expert bank, its input and down projections together) and every format of the menu, it writes:

- ``bits``: what the unit stores at that format (`engine.checkpoint_size.packed_bits`, the parts the judge counts);
- ``dkl``: ``0.5 * sum(F * delta**2)``, with ``delta`` the round-to-nearest error at that format.

The number prices ONE operation: storing that unit alone at that format by round to nearest, at the format's
own group, every other unit unchanged. It does not price a GPTQ or AWQ build, a group or symmetry change of
the menu's formats, or several units at once (it adds them as if independent).

What it is worth was measured, and every map carries it (`LIMITS`): it ranks and does not calibrate, and on the
4B it ranked the full-attention q/k/v fourth of ten where a build measured them first (1 = largest).

An untied output head is priced too, beside the units (``head``): it is outside the budget's denominator, and
``qda fisher allocate --head-lease`` gives it a format under the head lease (`qda_quant.head_lease`).

``qda fisher allocate MAP --body F --budget-bpw B`` picks one format per unit by greedy dKL saved per bit spent,
inside the budget, and prints a ``formats:`` block. Every routed-expert bank takes one format together,
because a build refuses anything else (`engine.format_bindings`); the map still prices each bank on its own.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from ..solve.units import Unit

#: What the numbers are worth, measured; carried in every map and printed with it.
LIMITS = (
    "a ranking signal, not a calibrated KL: on Qwen3.5-4B (10 structural groups, w4->w8, RTN probes on 32 "
    "search96 sequences) Spearman 0.709, the sum over-predicts the measured uniform-RTN KL about 4x, and "
    "full-attention q/k/v ranked 4th of 10 predicted but 1st measured (descending, 1 = largest); on Qwen3-Coder-30B-A3B (10 attention "
    "groups and no expert bank, the same probes) Spearman 0.939, which ranks a different kind of group and "
    "says nothing about the banks (97 % of its weights); on a tiny MoE it under-predicts small "
    "deltas about 8x (routing flips are discrete and second order cannot see them). Check a choice with a "
    "build and kl before trusting it."
)
OPERATION = (
    "one unit alone stored at the format by round to nearest (absmax grid, the format's own group), every "
    "other unit unchanged; dkl = 0.5 * sum(F * delta^2), F = per-token diagonal Fisher of the model's own "
    "next-token distribution on the calibration samples"
)
#: The map's layout: 2 = a body-only denominator and the untied head priced beside the units (`head`). A map of
#: another layout is another cache entry, never reused.
MAP_LAYOUT = 2
#: What a map needs of the cards: 3x the bf16 weights (the model, its fp32 Fisher -- about twice the weights,
#: every routed expert's bank included -- and the bf16 copy `_spread` keeps) plus the activation margin: the
#: eager-attention backward and fp32 logits of a sample. The margin is the measured total on an
#: eight-billion-parameter model at 2048 tokens (the peaks of the two- and three-card cells put it near
#: 60 GB). A smaller margin changes no decision on 80 GB cards, but would size a job on 48 GB cards for
#: an OOM: the map needed 50.7 GiB of each of its two 80 GB cards.
ACTIVATION_BYTES = 60_000_000_000
#: The share of a card's memory a map may use: 75 of an 80 GB card, the rest left to the allocator and the
#: CUDA context. The same share applies to whatever card the job has (`card_bytes`).
CARD_SHARE = 15 / 16
#: The card rule as one sentence: the tool's help, TASK.md (`components.Fisher`) and the refusal read this.
CARD_RULE = (
    "without --device the map takes the cards the model needs, from every GPU the job sees: three times the "
    "model's bf16 weight bytes plus the activation margin of a sample, in the share of the smallest visible "
    f"card a map may use ({CARD_SHARE:.2f} of it) -- about two cards for an eight-billion-parameter model on "
    "eighty-gigabyte cards, three for a twenty-seven-billion one; the three times is the fp32 Fisher "
    "accumulator beside the bf16 model (about three times the model's weights in all, every routed-expert "
    "bank included), and a job with fewer cards is refused instead of run to an OOM"
)
#: The cache rule, one sentence for the same readers.
CACHE_RULE = (
    "a map of the same model, calibration, menu, samples and seqlen is reused from $QDA_CACHE/fisher: a "
    "repeat run prints `from cache` and the path"
)


def _canon(u: Unit, tensors: tuple) -> torch.Tensor:
    """`tensors` (aligned with ``u.params``) in the unit's canonical ``[E, R, C]`` layout, as `Unit.weight`."""
    return replace(u, params=tensors).weight()


def _flat(params: tuple) -> list:
    """The parameters of a unit, flattened (an unfused bank nests them per expert)."""
    out = []
    for p in params:
        out.extend(_flat(p) if isinstance(p, tuple) else [p])
    return out


def _nest(params: tuple, f: dict) -> tuple:
    """`params` with every parameter replaced by its Fisher accumulator."""
    return tuple(_nest(p, f) if isinstance(p, tuple) else f[id(p)] for p in params)


def accumulate(
    model: torch.nn.Module, units: list[Unit], ids: list[torch.Tensor], seed: int = 0, log: Any = None
) -> dict:
    """The per-token diagonal Fisher of every unit parameter: {id(param): fp32 tensor of its shape}.

    One backward per sample. A parameter's squared gradient is folded into its accumulator as soon as it is
    complete (a post-accumulate hook), and the gradient is dropped, so no full gradient is kept.
    """
    import torch

    params = {id(p): p for u in units for p in _flat(u.params)}
    for p in model.parameters():
        p.requires_grad_(id(p) in params)
    f = {k: torch.zeros(p.shape, dtype=torch.float32, device=p.device) for k, p in params.items()}
    tokens = 0

    def fold(p: torch.Tensor) -> None:
        g = p.grad
        assert g is not None  # a post-accumulate hook runs after the grad is written
        f[id(p)].add_(g.float().square())
        p.grad = None

    hooks = [p.register_post_accumulate_grad_hook(fold) for p in params.values()]
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    try:
        for i, x in enumerate(ids, 1):
            x = x.to(dev).reshape(1, -1)
            logits = model(input_ids=x).logits[0, :-1].float()
            with torch.no_grad():
                y = torch.multinomial(logits.softmax(-1), 1, generator=g).squeeze(1)
            torch.nn.functional.cross_entropy(logits, y, reduction="sum").backward()
            tokens += logits.shape[0]
            if log is not None:
                log(i)
    finally:
        for h in hooks:
            h.remove()
    for t in f.values():
        t.div_(tokens)
    return f


def price(u: Unit, f: torch.Tensor, fmts: list[tuple[str, Any]], target: Any) -> dict[str, dict[str, float]]:
    """Per format of the menu: the bits `u` stores and its predicted dKL (bf16 costs 16 bits and 0 dKL)."""
    import torch

    from ..engine.checkpoint_size import packed_bits
    from ..solve.grid import DEFAULT_GRID
    from ..solve.quant import Quantizer

    w = u.weight()
    e, _, cols = w.shape
    n = w.numel()
    out: dict[str, dict[str, float]] = {}
    for key, fmt in fmts:
        if fmt is None:
            out[key] = {"bits": 16.0 * n, "dkl": 0.0}
            continue
        sf = fmt.expert_format if u.kind == "moe" else fmt
        q = Quantizer.of(sf, target.plan(sf, u.kind), DEFAULT_GRID)
        dkl = 0.0
        for i in range(e):  # one expert at a time: a bank's delta never exists whole
            wi = w[i].float()
            s, z = q.params(wi)
            dkl += 0.5 * float((f[i] * (q.fake(wi, s, z) - wi).square()).sum(dtype=torch.float64))
        bits = sum(packed_bits((e, r, cols) if u.part is not None else (r, cols), sf) for r in u.rows)
        out[key] = {"bits": float(bits), "dkl": dkl}
    return out


def decoder_weights(model: torch.nn.Module, graph: Any) -> int:
    """The weights of the decoder Linears the judge's denominator counts.

    Every 2-D Linear of a decoder layer except a MoE router (``mlp.gate``), plus each fused expert bank's
    stacked parameters; the output head is outside it (the judge prices it by its own lease). `qda_quant`
    may not import `qda_agent`, so this is a copy; `qda_agent/tests/test_budget_parity.py` pins it equal to
    the judge's `denominator`.
    """
    import torch

    from ..arch import submodule

    total = 0
    for lg in graph.layers:
        layer = model.get_submodule(lg.module)
        for name, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) and not name.endswith("mlp.gate"):
                total += m.weight.numel()
        for bank in lg.banks:
            if isinstance(bank.experts, str):
                total += sum(
                    p.numel() for p in submodule(layer, lg, bank.experts).parameters() if p.dim() == 3
                )
    return total


def fisher_map(
    model: torch.nn.Module,
    ids: list[torch.Tensor],
    menu: list[str],
    target: str | None,
    pricer: Any = None,
) -> dict:
    """The map of `model` over `menu` (format names; ``bf16`` allowed) on calibration `ids`.

    `pricer(unit, F, menu)` returns ``{menu key: {"bits", "dkl"}}`` for one unit; the default (`price`) is the
    compressed-tensors one. Another storage (GGUF types) passes its own, so the Fisher is computed once.
    """
    from ..arch import describe
    from ..format import Format
    from ..solve.units import units_of
    from ..target import Target

    t0 = time.monotonic()
    graph = describe(model, verify=False)
    tgt = Target.parse(target)
    fmts = [(m, None if m == "bf16" else Format.parse(m)) for m in menu] if pricer is None else None
    per_layer = [(lg, list(units_of(model.get_submodule(lg.module), lg))) for lg in graph.layers]
    units = [u for _, us in per_layer for u in us]
    head = _head_unit(model, graph)

    def log(done: int) -> None:
        if done % 16 == 0 or done == len(ids):
            print(f"fisher: {done}/{len(ids)} samples, {time.monotonic() - t0:.0f} s", file=sys.stderr)

    f = accumulate(model, units if head is None else [*units, head], ids, log=log)
    t1 = time.monotonic()
    rows: list[dict[str, Any]] = []
    for lg, us in per_layer:
        banks: dict[str, dict[str, Any]] = {}
        for u in us:
            fu = _canon(u, _nest(u.params, f))
            if pricer is None:
                assert fmts is not None  # parsed above whenever no pricer is given
                options = price(u, fu, fmts, tgt)
            else:
                options = pricer(u, fu, menu)
            names = list(u.names) if u.part is None else [u.names] if isinstance(u.names, str) else None
            if u.part is None:
                rows.append({"unit": u.key, "layer": lg.index, "kind": "dense", "modules": names,
                             "weights": u.weight().numel(), "options": options})  # fmt: skip
                continue
            b = banks.setdefault(u.key, {"unit": u.key, "layer": lg.index, "kind": "bank", "modules": [],
                                         "weights": 0, "options": {}})  # fmt: skip
            b["weights"] += u.weight().numel()
            if names:
                b["modules"] += names
            for k, v in options.items():
                o = b["options"].setdefault(k, {"bits": 0.0, "dkl": 0.0})
                o["bits"] += v["bits"]
                o["dkl"] += v["dkl"]
        rows.extend(banks.values())
    weights = decoder_weights(model, graph)
    head_row = None
    if head is not None:  # the lease prices it; it is in neither the denominator nor `kept_weights`
        fh = _canon(head, _nest(head.params, f))
        if pricer is None:
            assert fmts is not None  # parsed above whenever no pricer is given
            options = price(head, fh, fmts, tgt)
        else:
            options = pricer(head, fh, menu)
        head_row = {"module": head.key, "weights": head.weight().numel(), "options": options}
    return {
        "operation": OPERATION if pricer is None else getattr(pricer, "operation", "custom pricer"),
        "limits": LIMITS,
        "menu": menu,
        "target": str(tgt),
        "denominator": weights,
        "kept_weights": weights - sum(r["weights"] for r in rows),
        "fisher_s": round(t1 - t0, 1),
        "price_s": round(time.monotonic() - t1, 1),
        "tokens_per_sample": int(ids[0].numel()) if ids else 0,
        "samples": len(ids),
        "units": rows,
        "head": head_row,
    }


def _head_unit(model: torch.nn.Module, graph: Any) -> Unit | None:
    """The untied output head as a dense unit to price, or None (no head, or one tied to the embedding)."""
    import torch

    from ..solve.units import Unit

    if not graph.lm_head or graph.extra.get("tied"):
        return None
    w = model.get_submodule(graph.lm_head).weight
    if not isinstance(w, torch.Tensor):  # a Module attribute is typed Tensor | Module
        return None
    return Unit(graph.lm_head, None, (graph.lm_head,), (int(w.shape[0]),), "dense", (w,))


def unit_rule(modules: list[str]) -> str:
    """One ``formats:`` rule naming every module of a unit: a rule may not name part of a fused unit."""
    if len(modules) == 1:
        return modules[0]
    return "re:^(?:" + "|".join(re.escape(n) for n in modules) + ")$"


def _fits(fmt: str, body: str, bank: bool) -> str | None:
    """Why a checkpoint whose body is `body` cannot hold `fmt` on a unit (`bank`: an expert bank), or None.

    One checkpoint stores every quantized module at the body's group; the body's own family keeps the body's
    symmetry (a bank's as the planner stores it, `Format.expert_format`); every other family is stored symmetric.
    An NVFP4 body's checkpoint is mixed-precision: FP8 W8A8 dense modules (and the head) keep their own scales.
    """
    from ..format import Format

    if fmt == "bf16":
        return None
    f, b = Format.parse(fmt), Format.parse(body)
    if f.weight_type == "fp8" and b.act == "nvfp4_block":  # a mixed-precision checkpoint: FP8 keeps its own scale
        return "QDA writes no FP8 expert bank" if bank else None
    if f.group_size != b.group_size:
        return f"group {f.group_size}, not the body's {b.group_size}"
    sym = (f.expert_format if bank else f).symmetric
    if (f.weight_bits, f.act) == (b.weight_bits, b.act):
        want = (b.expert_format if bank else b).symmetric
        if sym != want:
            return f"{'sym' if sym else 'asym'}, not the body's {'sym' if want else 'asym'}"
    elif not sym:
        return "another family than the body's, stored asymmetric"
    return None


def allocate(m: dict, body: str, budget: float, *, head_lease: bool = False) -> dict[str, Any]:
    """One format per unit by greedy dKL saved per bit, inside `budget` bpw; every bank in one format.

    `head_lease`: the output head is one more unit, starting at the narrowest format the lease offers the
    body's row that the map priced, and the limit is the lease's (`head_lease.limit_bits`): what a narrower
    head saves goes to the body, the total stays the same. Off, the head is not allocated and stays as the
    original stores it.

    Starts every unit at `body` and keeps taking the move (to another menu format a checkpoint of `body` can
    hold on that class, `_fits`) that saves the most predicted dKL per added bit while the total stays inside the
    budget. Returns the choice, its bpw and its predicted dKL, the ``formats`` block (one rule per unit,
    `unit_rule`; banks by a pattern naming every bank), and the menu formats left out with the reason.

    Raises:
        ValueError: the body alone is over the budget, `body` is not in the map's menu, or `head_lease` on a
            map with no head entry (a tied head, or a map measured before heads were priced).
    """
    from ..head_lease import family, lease_formats, leased_bpw, limit_bits

    if body not in m["menu"]:
        raise ValueError(f"body {body!r} is not in the map's menu {m['menu']}")
    head = m.get("head") if head_lease else None
    if head_lease and head is None:
        raise ValueError(
            "--head-lease needs the head in the map: this model ties its head to the embedding (nothing to "
            "lease), or the map was measured before heads were priced (measure it again with --no-cache)"
        )
    head_menu: list[str] = []
    if head is not None:  # the formats the lease offers this row that a checkpoint of `body` can store
        offered = lease_formats(body)
        head_menu = sorted(
            (k for k in head["options"] if family(k) in offered and _fits(k, body, False) is None),
            key=lambda k: head["options"][k]["bits"],
        )
        if not head_menu:
            raise ValueError(
                f"--head-lease: the map's menu {m['menu']} has no head format a {body} checkpoint can store "
                f"(the lease offers {', '.join(offered)}); measure a map whose menu has one, or drop --head-lease"
            )
    dense = [r for r in m["units"] if r["kind"] == "dense"]
    banks = [r for r in m["units"] if r["kind"] == "bank"]
    classes: list[tuple[str, list[dict]]] = [(r["unit"], [r]) for r in dense]
    if banks:
        classes.append(("experts", banks))

    def cost(rs: list[dict], fmt: str) -> tuple[float, float]:
        return sum(r["options"][fmt]["bits"] for r in rs), sum(r["options"][fmt]["dkl"] for r in rs)

    kept = 16.0 * m["kept_weights"]
    choice = {k: body for k, _ in classes}
    bits = kept + sum(cost(rs, body)[0] for _, rs in classes)
    limit = budget * m["denominator"]
    if head is not None:  # the head is one more class: the lease charges it 16 bits a weight up front
        classes.append(("head", [head]))
        choice["head"] = head_menu[0]
        bits += head["options"][head_menu[0]]["bits"]
        limit = limit_bits(budget, m["denominator"], head["weights"])
    if bits > limit:  # with a head, at its narrowest: the body may use what that head saves
        body_bits = bits - (head["options"][head_menu[0]]["bits"] if head is not None else 0.0)
        lent = f" with the head at {head_menu[0]}" if head is not None else ""
        raise ValueError(f"the body alone is {body_bits / m['denominator']:.4f} bpw{lent}, over the budget {budget}")
    excluded = {
        f"{fmt} ({'banks' if bank else 'dense'})": why
        for bank in (False, True)
        for fmt in m["menu"]
        if (why := _fits(fmt, body, bank)) is not None
    }
    while True:
        best = None
        for k, rs in classes:
            b0, d0 = cost(rs, choice[k])
            for fmt in head_menu if k == "head" else m["menu"]:
                if k != "head" and _fits(fmt, body, k == "experts") is not None:
                    continue
                b1, d1 = cost(rs, fmt)
                if b1 <= b0 or d1 >= d0 or bits + b1 - b0 > limit:
                    continue
                gain = (d0 - d1) / (b1 - b0)
                if best is None or gain > best[0]:
                    best = (gain, k, fmt, b1 - b0)
        if best is None:
            break
        _, k, fmt, db = best
        choice[k], bits = fmt, bits + db
    formats: dict[str, str] = {"default": body}
    for k, rs in classes:
        if choice[k] == body or k == "head":
            continue
        if k == "experts":
            formats[r"re:.*\.mlp\.experts.*"] = choice[k]
        else:
            formats[unit_rule(rs[0]["modules"])] = choice[k]
    dkl = sum(cost(rs, choice[k])[1] for k, rs in classes)
    out = {"bpw": bits / m["denominator"], "dkl": dkl, "budget": budget, "body": body,
           "moved": {k: v for k, v in choice.items() if v != body and k != "head"}, "formats": formats,
           "excluded": excluded}  # fmt: skip
    if head is not None:
        head_bits = head["options"][choice["head"]]["bits"]
        out["bpw"] = (bits - head_bits) / m["denominator"]  # the body's own, as a card reports it
        out["leased_bpw"] = leased_bpw(bits - head_bits, head_bits, m["denominator"], head["weights"])
        out["head"] = choice["head"]
        if choice["head"] != "bf16":  # the recipe stores it with a `head` step, in the format priced above
            out["head_step"] = {"head": {"format": choice["head"]}}
    return out


def weight_bytes(model: str | Path) -> int:
    """The model's bf16 weight bytes: its safetensors (or `.bin`s), or its index's total size."""
    root = Path(model)
    for pattern in ("*.safetensors", "*.bin"):
        total = sum(p.stat().st_size for p in root.glob(pattern) if p.is_file())
        if total:
            return total
    index = root / "model.safetensors.index.json"
    return json.loads(index.read_text())["metadata"]["total_size"] if index.is_file() else 0


def cards_for(weights: int, card_bytes: int) -> int:
    """The cards a map of `weights` bytes of bf16 weights needs, each holding at most `card_bytes`."""
    return max(1, math.ceil((3 * weights + ACTIVATION_BYTES) / card_bytes))


def visible_cards() -> int:
    """The GPUs this process may use: CUDA_VISIBLE_DEVICES when set, else torch's device count."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip()])
    import torch

    return torch.cuda.device_count()


def card_bytes(count: int) -> int:
    """What a map may use of the smallest of the `count` GPUs this process sees (`CARD_SHARE` of its total).

    Raises:
        Refused: CUDA does not report the devices' memory; the caller names them with `--device` instead.
    """
    from ..step import Refused

    try:
        import torch

        total = min(torch.cuda.get_device_properties(i).total_memory for i in range(count))
    except Exception as e:  # noqa: BLE001 - no CUDA, or a driver that will not report: one line, no traceback
        raise Refused(f"cannot read the GPUs' memory ({e}); pass --device with the cards you have") from None
    return int(total * CARD_SHARE)


def auto_devices(model: str | Path) -> list[str]:
    """The devices a map gets without `--device`: cuda:0..n-1, on the cards the model needs.

    The card size is the smallest of the GPUs the job sees, so a small card (a 48 GB or 24 GB one) takes more
    of them or is refused.

    Raises:
        Refused: the job sees no GPU, or fewer than the model needs; the message states both numbers (it
            refuses before a load, so a job sized for builds cannot run the map into an OOM).
    """
    from ..step import Refused

    weights, have = weight_bytes(model), visible_cards()
    if not have:
        raise Refused("this job sees no GPU: give the job a card, or pass --device with the cards you have")
    per = card_bytes(have)
    need = cards_for(weights, per)
    if have < need:
        raise Refused(
            f"a map of {weights / 1e9:.1f} GB of bf16 weights needs {need} cards of "
            f"{per / CARD_SHARE / 2**30:.0f} GiB and this job sees {have}: give the job {need} cards, or pass "
            "--device with the cards you have"
        )
    return [f"cuda:{i}" for i in range(need)]


def cache_root() -> Path:
    """``$QDA_CACHE/fisher``; without QDA_CACHE ``~/.cache/qda/fisher`` (the reference cache's root rule)."""
    return Path(os.environ.get("QDA_CACHE") or Path.home() / ".cache" / "qda") / "fisher"


def cache_payload(
    model: str | Path,
    calib: str | Path,
    menu: list[str],
    *,
    samples: int,
    seqlen: int,
    target: str | None = None,
) -> dict[str, Any]:
    """Everything a map's bytes depend on: the model's content fingerprint, the calibration's sha256, the menu
    in the order given, and the two sampling knobs (a map of 128 x 2048 is not the map of 256 x 4096)."""
    from ..calib import sha256_file
    from ..target import Target
    from .fingerprint import model_fingerprint

    return {
        "layout": MAP_LAYOUT,
        "model": model_fingerprint(model),
        "calib_sha256": sha256_file(calib),
        "menu": list(menu),
        "samples": int(samples),
        "seqlen": int(seqlen),
        "target": None if target is None else str(Target.parse(target)),
    }


def cache_entry(payload: dict[str, Any]) -> Path:
    """The cache directory of the map `payload` describes: ``$QDA_CACHE/fisher/<model[:16]>/<digest[:24]>``."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_root() / payload["model"][:16] / digest[:24]


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `path`, waiting for another process that is building the same map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("fisher map: another process is building this map; waiting for it", file=sys.stderr)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def cached_map(
    payload: dict[str, Any], build: Callable[[Path], None], log: Callable[[str], None] = print
) -> tuple[Path, bool]:
    """The map of `payload`, built by `build(dest)` once into the cache; returns the map file and whether this
    call built it (`False`: it was there).

    A finished entry is a directory with `map.json` and its payload beside it in `<key>.json`, moved in with
    one rename under a file lock, so two processes asking for the same map build it once.
    """
    entry = cache_entry(payload)
    map_file, key_file = entry / "map.json", entry.with_suffix(".json")
    if map_file.is_file() and key_file.is_file():
        log(f"fisher map: from cache {map_file}")
        return map_file, False
    entry.parent.mkdir(parents=True, exist_ok=True)
    with _lock(entry.with_name(f".{entry.name}.cache-lock")):
        if map_file.is_file() and key_file.is_file():
            log(f"fisher map: from cache {map_file}")
            return map_file, False
        tmp = entry.with_name(f".{entry.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"fisher map: building {entry} (a later run of these inputs reuses it)")
        try:
            build(tmp / "map.json")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        key_file.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        tmp.replace(entry)  # one rename: a directory with map.json is a finished entry
    return map_file, True


def load_bf16(path: str | Path, device: str) -> torch.nn.Module:
    """The model at ``path`` in bf16 on ``device``, eager attention.

    A ``pack-quantized`` checkpoint is dequantized by `qda_quant.engine.packed.load_float`, which supplies
    its zero points and gives the model class's own module tree; another quantized format is decompressed by
    transformers.
    """
    import torch

    from ..engine.packed import Packed, load_float
    from ..step import Refused, remote_code_refusal

    try:
        ck = Packed(path)
    except Refused:  # not a pack-quantized checkpoint, or a layout this reader does not take
        ck = None
    if ck is not None:
        return load_float(ck, torch.device(device), attn="eager")

    from .loading import load_model as load

    cfg = None
    kw: dict[str, Any] = {}
    if "quantization_config" in json.loads((Path(path) / "config.json").read_text()):
        from transformers import AutoConfig, CompressedTensorsConfig

        from .source_names import SourceNames

        # compressed-tensors matches ignores against in-memory modules, before transformers renames
        # the source weights. Keep the checkpoint's own names on disk; translate only for this load.
        names = SourceNames(path)
        try:  # SourceNames already refuses a remote-code config today; kept should it stop reading one
            cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=False)
        except ValueError as e:
            if remote := remote_code_refusal(path, "config", e):
                raise remote from e
            raise
        quant = cfg.quantization_config
        ignores = quant.get("ignore", [])
        quant["ignore"] = [
            (names.memory(name + ".weight") or name + ".weight").removesuffix(".weight") for name in ignores
        ]
        kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    m = load(
        str(path),
        config=cfg,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
        **kw,
    )
    return m.eval()


def _spread(path: str, devices: list[str]) -> torch.nn.Module:
    """The bf16 model over `devices`, its weights split evenly so each card keeps room for twice its share (F)."""
    import torch

    from .loading import load_model as load

    index = Path(path) / "model.safetensors.index.json"
    size = json.loads(index.read_text())["metadata"]["total_size"] if index.is_file() else 0
    share = f"{int(size / len(devices) / (1 << 30) * 1.1) + 1}GiB"
    m = load(
        str(path),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={int(d.split(":")[1]): share for d in devices},
        attn_implementation="eager",
    )
    return m.eval()


EPILOG = f"""examples:
  qda fisher map --model Qwen3.5-4B --calib calib.jsonl --menu bf16,w8a16-sym-g128,w4a16-asym-g128 \\
      --samples 128 --seqlen 2048 --device cuda:0 --out map.json
  qda fisher map --model Qwen3-8B --calib calib.jsonl --menu bf16,w8a16 --out map.json  # its cards

cards: {CARD_RULE}
cache: {CACHE_RULE}

exit status: 0 ok, 2 refused (bad calibration file, a format the target cannot serve, a body over budget,
a job with fewer cards than the model needs)"""


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add ``fisher`` to a subparser set."""
    p = sub.add_parser(
        "fisher",
        help="per-unit bits and predicted dKL for each menu format (diagonal Fisher), and an allocation",
        description=(__doc__ or "").split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    q = p.add_subparsers(dest="what", required=True)
    mp = q.add_parser("map", help="measure the map of a bf16 model on calibration samples")
    mp.add_argument("--model", required=True, metavar="DIR", help="the bf16 original")
    mp.add_argument(
        "--calib", required=True, metavar="FILE", help="calibration JSONL (text, messages or ids)"
    )
    mp.add_argument("--menu", required=True, help="comma-separated formats, bf16 allowed")
    mp.add_argument("--samples", type=int, default=128, help="calibration samples (default 128)")
    mp.add_argument("--seqlen", type=int, default=2048, help="tokens per sample (default 2048)")
    mp.add_argument("--target", help="serving target (default: the library's)")
    mp.add_argument(
        "--device",
        help="cpu, one GPU (cuda:0), or several (cuda:0,cuda:1,...); " + CARD_RULE,
    )
    mp.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        default=True,
        help="rebuild the map even when one of these inputs is already cached",
    )
    mp.add_argument("--out", required=True, metavar="FILE", help="the map, JSON")
    al = q.add_parser("allocate", help="one format per unit inside a bpw budget, as a formats: block")
    al.add_argument("map", metavar="MAP")
    al.add_argument("--body", required=True, help="the row's format, every unit's start")
    al.add_argument(
        "--budget-bpw",
        dest="budget",
        required=True,
        type=float,
        help="bits per weight of the decoder Linears with --body (with --head-lease, the head too)",
    )
    al.add_argument(
        "--head-lease",
        action="store_true",
        help="also give the untied output head a format: it is charged 16 bits a weight and what a narrower "
        "head saves goes to the body (the search's judge.head_lease); default off, the head stays as stored",
    )
    p.set_defaults(run=run)
    return p


def _measure(a: argparse.Namespace, menu: list[str], devices: list[str], dest: Path) -> None:
    """Measure the map into `dest`: the calibration ids, the bf16 model on `devices`, the map and its record."""
    from ..calib import sha256_file
    from ..calib.sources import FileSource
    from ..engine.calib import load_ids

    ids = load_ids(FileSource(path=str(a.calib), samples=a.samples, seqlen=a.seqlen), a.model)
    model = load_bf16(a.model, devices[0]) if len(devices) == 1 else _spread(a.model, devices)
    m = fisher_map(model, ids, menu, a.target)
    m |= {"model": str(a.model), "calib": str(a.calib), "calib_sha256": sha256_file(a.calib)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(m, indent=1) + "\n")


def _copy(src: Path, out: Path) -> None:
    """Put the map at `out` in one rename (a reader never sees half of it)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_bytes(src.read_bytes())
    tmp.replace(out)


def run(a: argparse.Namespace) -> int:
    """Run a parsed ``fisher`` command."""
    from ..exit import refuse
    from ..step import Refused

    try:
        if a.what == "allocate":
            m = json.loads(Path(a.map).read_text())
            print(json.dumps(allocate(m, a.body, a.budget, head_lease=a.head_lease), indent=1))
            return 0
        out = Path(a.out)  # `map` is the subcommand that defines --out; `allocate` returns above
        menu = a.menu.split(",")

        def measure(dest: Path) -> None:
            # the cards are chosen only for a build: a cache hit needs none, so a small job still gets its map
            devices = a.device.split(",") if a.device else auto_devices(a.model)
            print(f"fisher map: {len(devices)} card(s): {','.join(devices)}", file=sys.stderr)
            _measure(a, menu, devices, dest)

        if a.cache:
            payload = cache_payload(
                a.model, a.calib, menu, samples=a.samples, seqlen=a.seqlen, target=a.target
            )
            map_file, built = cached_map(payload, measure, log=lambda line: print(line, file=sys.stderr))
        else:
            map_file, built = out, True
            measure(out)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    except ValueError as e:  # a menu entry or a calibration file the map cannot use
        return refuse(str(e))
    if map_file != out:
        _copy(map_file, out)
    if not built:
        print(f"fisher map: from cache {map_file} -> {out}")
        return 0
    m = json.loads(map_file.read_text())
    print(f"fisher map: {len(m['units'])} units, {m['samples']} x {m['tokens_per_sample']} tokens, "
          f"fisher {m['fisher_s']} s, price {m['price_s']} s -> {out}\nlimits: {LIMITS}")  # fmt: skip
    return 0
