"""A module whose shape no kernel of the row's format serves stays bf16 instead of refusing the plan.

Qwen3.5's Gated DeltaNet per-head projections (``in_proj_a`` / ``in_proj_b``, 32 rows on the 9B) have no
W4A8 kernel on Hopper: ``CutlassW4A8LinearKernel`` needs rows and columns that are multiples of 128. The
released W4A8 checkpoint keeps them bf16 and quantizes the rest, so the plan must do the same; only a
material share (or a module the graph does not already keep) is a refusal.
"""

import torch
from qda_quant import REGISTRY, Format, Target
from qda_quant.arch import MIN_QUANT_ROWS
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build_validation import (
    UNSERVABLE_SHARE,
    QuantWidth,
    kept_unservable,
    unservable_widths,
)
from qda_quant.engine.planning import compile_plan
from qda_quant.recipe import Recipe

from .engine_util import save_tiny_tokenizer, write_ids
from .tiny import model

TARGET = Target.parse("sm90")


def _widths(rows: tuple[int, ...], cols: int = 4096) -> list[QuantWidth]:
    return [QuantWidth(f"m{i}", r, cols) for i, r in enumerate(rows)]


def test_a_tiny_projection_with_no_w4a8_kernel_is_kept_and_a_16_bit_row_keeps_nothing():
    got = unservable_widths(_widths((32, 4096, 4096)), Format.parse("w4a8-sym-g128"), TARGET)
    assert got is not None and got.modules == ("m0",) and got.share < UNSERVABLE_SHARE
    assert "no kernel" in got.detail and "m0" in got.detail
    assert kept_unservable(got, Format.parse("w4a8-sym-g128")) == ("m0",)
    # Marlin (W4A16) serves the same shape, and a 16-bit row has no kernel rule to fail
    assert unservable_widths(_widths((32, 4096, 4096)), Format.parse("w4a16-sym-g128"), TARGET) is None


def test_r1_8b_shapes_are_unchanged():
    """Every width the released R1-8B W4A8 quantizes is a multiple of 128 and at least 128 rows wide."""
    widths = [
        QuantWidth(f"layer{i}.proj", rows, cols)
        for i, (rows, cols) in enumerate(
            ((4096, 4096), (1024, 4096), (512, 4096), (14336, 4096), (4096, 14336))
        )
    ]
    assert unservable_widths(widths, Format.parse("w4a8-sym-g128"), TARGET) is None


def test_a_material_share_is_not_kept_so_the_plan_refuses_the_format():
    """A fifth of the quantized parameters in a shape no kernel serves is a format the model should not use:
    nothing is kept here, and the plan's shape check refuses it with its own line."""
    widths = [QuantWidth("tiny", 32, 4096), QuantWidth("big", 4096, 128)]
    got = unservable_widths(widths, Format.parse("w4a8-sym-g128"), TARGET)
    assert got is not None and got.share > UNSERVABLE_SHARE
    assert "20.00%" in got.detail
    assert kept_unservable(got, Format.parse("w4a8-sym-g128")) == ()


def test_a_linear_under_the_minimum_rows_is_not_measured():
    """The plan keeps a Linear narrower than MIN_QUANT_ROWS bf16 anyway; prepare must not measure it."""
    widths = _widths((MIN_QUANT_ROWS - 1, 128, 128))
    assert unservable_widths(widths, Format.parse("w4a8-sym-g128"), TARGET) is None


def _qwen35(tmp):
    m = model("qwen3_5-heads32")
    m.save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    write_ids(tmp / "c.jsonl", [[3, 4, 5, 6]], sha)
    return str(tmp / "m"), FileSource(path=str(tmp / "c.jsonl"), samples=1, seqlen=4)


def _plan(tmp, keep=None):
    source, calib = _qwen35(tmp)
    return compile_plan(
        source,
        Recipe(steps=[REGISTRY["rtn"]()], calib=calib),
        Format.parse("w4a8-sym-g128"),
        TARGET,
        [torch.device("cpu")],
        keep=keep,
    )


def test_qwen3_5_9b_shapes_plan_w4a8_with_the_head_gates_bf16(tmp_path):
    """The released checkpoint's recipe: in_proj_a/b keep bf16 (the gdn_gates role), the rest is quantized."""
    plan = _plan(tmp_path)
    assert plan.fmt.name == "w4a8-sym-g128"
    quantized = {m for lg in plan.graph.layers for grp in lg.groups for m in grp.modules}
    gates = {"model.layers.0.linear_attn.in_proj_a", "model.layers.0.linear_attn.in_proj_b"}
    assert not gates & quantized


def test_without_the_role_the_plan_keeps_the_unservable_projection_and_says_so(tmp_path, monkeypatch):
    """With the role dropped, the same shapes reach the group and the new rule keeps them; plan prints it.

    The tiny model's two gates are a larger share of its parameters than a real 9B's 0.2 %, so the policy
    threshold is raised for this model only to exercise the keep path.
    """
    from qda_quant.engine import build_validation

    monkeypatch.setattr(build_validation, "UNSERVABLE_SHARE", 0.10)
    plan = _plan(tmp_path, keep=frozenset())
    kept = plan.graph.extra.get("kept_unservable") or ()
    assert {"model.layers.0.linear_attn.in_proj_a"} <= set(kept)
    assert "in_proj_a" in plan.graph.extra["kept_unservable_line"]
    quantized = {m for lg in plan.graph.layers for grp in lg.groups for m in grp.modules}
    assert not set(kept) & quantized
