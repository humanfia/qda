"""The allocation must beat brute force's answer under the same budget, and refuse what does not fit."""

from __future__ import annotations

import itertools
import json

import pytest
from qda_quant.gguf import allocate, rule
from qda_quant.gguf import classes as sensitivity

BF16 = allocate.BF16  #: the allocation's starting point, the string "bf16"
_TYPE = rule.BF16  #: the ggml type of every template row in this test
_H = 256
#: every class is 256 wide, so all four grid types are legal for all of them
ROWS = [
    ("token_embd.weight", (256, _H), _TYPE),
    ("output.weight", (256, _H), _TYPE),
    ("blk.0.attn_q.weight", (256, _H), _TYPE),
    ("blk.0.attn_k.weight", (256, _H), _TYPE),
    ("blk.0.attn_v.weight", (256, _H), _TYPE),
    ("blk.0.attn_output.weight", (_H, 256), _TYPE),
    ("blk.0.ffn_gate.weight", (256, _H), _TYPE),
    ("blk.0.ffn_up.weight", (256, _H), _TYPE),
    ("blk.0.ffn_down.weight", (_H, 256), _TYPE),
]
CELLS = [
    allocate.Cell("attn_q", "Q8_0", 0.0002),
    allocate.Cell("attn_q", "Q4_K", 0.0010),
    allocate.Cell("attn_k", "Q8_0", 0.0001),
    allocate.Cell("attn_k", "Q4_K", 0.0008),
    allocate.Cell("ffn_down", "Q8_0", 0.0003),
    allocate.Cell("ffn_down", "Q4_K", 0.0040),
]


def _plan():
    return sensitivity.plan_from_tensors("qwen35", 1, ROWS)


def _brute(plan, slots, cells, budget):
    """Minimum (dKL, bytes) over every assignment that fits `budget`; ties keep fewer bytes."""
    by = {(c.klass, c.kind): c.d_kl for c in cells}
    best = None
    for choice in itertools.product(*slots):
        nbytes = value = 0
        for klass, kind in zip(plan.classes, choice, strict=True):
            got = plan.classes[klass]
            nbytes += got.file_bytes["bf16"] if kind == BF16 else got.file_bytes[kind]
            value += 0.0 if kind == BF16 else max(by.get((klass, kind), 0.0), 0.0)
        if nbytes <= budget and (best is None or (value, nbytes) < best):
            best = (value, nbytes)
    return best


def test_the_allocation_matches_brute_force() -> None:
    plan = _plan()
    base = plan.file_total({c: BF16 for c in plan.classes})
    measured = {(c.klass, c.kind) for c in CELLS}
    slots = [[BF16, *sorted(k for k in plan.classes[c].bytes if (c, k) in measured)] for c in plan.classes]
    for budget in (base, base + 3_000, base + 12_000, base + 40_000):
        got = allocate.allocate(plan, CELLS, budget, quantum=1)
        assert (got.d_kl, got.file_bytes) == _brute(plan, slots, CELLS, budget)
        assert got.file_bytes <= budget


def test_the_frontier_is_monotone_and_nondominated() -> None:
    got = allocate.frontier(_plan(), CELLS, quantum=1)
    # ascending in bytes: the smallest file first, bf16 (zero increment, most bytes) last
    assert {k: v for k, v in got[0].picks.items() if v != BF16} == {
        "attn_q": "Q4_K",
        "attn_k": "Q4_K",
        "ffn_down": "Q4_K",
    }
    assert got[0].d_kl == pytest.approx(0.0010 + 0.0008 + 0.0040) and got[-1].d_kl == 0.0
    for a, b in zip(got, got[1:], strict=False):
        assert a.file_bytes < b.file_bytes and b.d_kl < a.d_kl


def _cheapest(plan, cells) -> int:
    """The smallest file the measured choices allow: every class at its cheapest legal choice."""
    allowed: dict[str, set[str]] = {c: {BF16} for c in plan.classes}
    for cell in cells:
        allowed[cell.klass].add(cell.kind)
    picks = {}
    for name, kinds in allowed.items():
        pairs = [
            (int(v), k) for k, v in plan.classes[name].file_bytes.items() if k in kinds and v is not None
        ]
        picks[name] = min(pairs)[1]
    return plan.file_total(picks)


def test_the_default_grid_keeps_the_true_minimum() -> None:
    # a grid wider than the whole fixture (the 1 MiB default against sub-MiB files) must not fold the
    # cheapest allocation into a more expensive one: the anchor is each class's own cheapest choice
    plan = _plan()
    least = _cheapest(plan, CELLS)
    assert least < 1 << 20
    got = allocate.frontier(plan, CELLS)  # the default quantum
    assert got[0].file_bytes == least
    assert allocate.allocate(plan, CELLS, least).file_bytes == least
    with pytest.raises(ValueError, match=f"below the smallest allocation's {least} B"):
        allocate.allocate(plan, CELLS, least - 1)


def test_a_non_finite_increment_or_floor_is_refused() -> None:
    plan = _plan()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            allocate.frontier(plan, [allocate.Cell("attn_q", "Q8_0", bad)])
        with pytest.raises(ValueError, match="not finite"):
            allocate.allocate(plan, CELLS, 1 << 30, floor=bad)


def test_as_dict_names_the_file_and_the_payload_apart() -> None:
    got = allocate.frontier(_plan(), CELLS, quantum=1)[0]
    row = got.as_dict()
    assert row["file_bytes"] == got.file_bytes and row["payload_bytes"] == got.payload_bytes
    assert row["payload_bytes"] <= row["file_bytes"] and row["d_kl"] == got.d_kl


def test_a_budget_below_every_allocation_is_refused_with_the_minimum() -> None:
    plan = _plan()
    least = allocate.frontier(plan, CELLS, quantum=1)[0].file_bytes
    with pytest.raises(ValueError, match=f"below the smallest allocation's {least} B"):
        allocate.allocate(plan, CELLS, least - 1, quantum=1)


def test_the_payload_holds_the_fixed_buckets_and_the_file_is_larger() -> None:
    # a norm is a fixed keep: it is no class, but its bytes are in the tensor payload a reading reports
    rows = [*ROWS, ("blk.0.attn_norm.weight", (256,), _TYPE)]
    plan = sensitivity.plan_from_tensors("qwen35", 1, rows, header_bytes=16)
    got = allocate.frontier(plan, CELLS, quantum=1)[0]
    keeps = sum(c.rule_bytes for c in plan.excluded.values())
    assert keeps > 0 and got.payload_bytes == plan.total(got.picks) + keeps
    assert got.file_bytes == plan.file_total(got.picks) > got.payload_bytes  # the header is in the file


def test_the_reader_refuses_a_non_finite_result_baseline_or_increment(tmp_path) -> None:
    import json

    (tmp_path / "nan.json").write_text(json.dumps({"kl": float("nan")}))
    with pytest.raises(ValueError, match="kl is not finite"):
        allocate.read_kl(tmp_path / "nan.json")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"class": "attn_q", "type": "Q8_0", "d_kl": float("nan")}]))
    with pytest.raises(ValueError, match="increment is not finite"):
        allocate.rows_from_manifest(rows)
    rows.write_text(json.dumps([{"class": "attn_q", "type": "Q8_0", "d_kl": 0.001}]))
    with pytest.raises(ValueError, match="baseline is not finite"):
        allocate.rows_from_manifest(rows, baseline=float("inf"))


def test_a_cell_below_the_floor_buys_nothing() -> None:
    plan = _plan()
    down = plan.classes["ffn_down"]
    # a budget that only ffn_down can meet, and only by taking Q4_K
    base, small = down.file_bytes["bf16"], down.file_bytes["Q4_K"]
    assert base is not None and small is not None
    budget = plan.file_total({c: BF16 for c in plan.classes}) - (base - small)
    noise = [allocate.Cell("ffn_down", "Q4_K", 1e-6)]
    got = allocate.allocate(plan, noise, budget, floor=1e-4, quantum=1)
    assert got.picks["ffn_down"] == "Q4_K" and got.d_kl == 0.0  # the reading bought no allocation cost
    got = allocate.allocate(plan, noise, budget, floor=0.0, quantum=1)
    assert got.picks["ffn_down"] == "Q4_K" and got.d_kl == pytest.approx(1e-6)


def test_manifest_rows_read_a_result_and_subtract_a_baseline(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"kl": 0.0100}))
    manifest = tmp_path / "rows.json"
    manifest.write_text(
        json.dumps(
            [
                {"class": "ffn_down", "type": "Q4_K", "out": str(run)},
                {"class": "attn_q", "type": "Q8_0", "d_kl": 0.0002},
            ]
        )
    )
    got = allocate.rows_from_manifest(manifest, baseline=0.0090)
    assert got[0].d_kl == pytest.approx(0.0010) and got[1].d_kl == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="needs d_kl, or out and a baseline"):
        allocate.rows_from_manifest(manifest)


def test_additivity_reports_the_relative_gap_and_the_sum() -> None:
    assert allocate.additivity(0.0104, 0.0100).within(0.15)
    assert not allocate.additivity(0.0120, 0.0100).within(0.05)
    assert allocate.additivity(0.0, 0.0).relative_error == 0.0
    picks = {"ffn_down": "Q4_K", "attn_q": "bf16"}
    assert allocate.summed(CELLS, picks) == pytest.approx(0.0040)
