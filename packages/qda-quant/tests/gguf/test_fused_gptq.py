"""The fused GGUF block must preserve FP32 offset arithmetic, zero scales, and packed build bytes."""

import json

import pytest
import torch
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from qda_quant.gguf.packing import q4k_bytes, q6k_bytes
from qda_quant.solve import solve
from qda_quant.solve.quant import Quantizer
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ
from qda_quant.target import Q4K, Q5K, Q6K, Target

from .test_export import WIDE, _source


def test_explicit_fused_grid_refuses_an_opt_out_or_unverified_grid():
    """Planning must reject an explicit fused request before calibration, including an inherited Q5 grid."""
    for fmt, grid in ((Q4K, "q4k"), (Q6K, "q6k"), (Q5K, "q5k")):
        f = Format.parse(fmt)
        plan = Target("llama_cpp").plan(f, "dense")
        for enabled in (False, True):
            step = GPTQ(grid=grid, impl="triton", fused_grid=enabled)
            if enabled and fmt != Q5K:
                step.check_grid(f, plan)
            else:
                with pytest.raises(Refused, match="no enabled fused formula"):
                    step.check_grid(f, plan)
            step.model_copy(update={"impl": "auto"}).check_grid(f, plan)


@pytest.mark.gpu
@pytest.mark.parametrize("fmt,grid,pack", [(Q4K, "q4k", q4k_bytes), (Q6K, "q6k", q6k_bytes)])
@pytest.mark.parametrize("actorder", ["none", "static"])
def test_fused_grid_preserves_codes_and_block_bytes(fmt, grid, pack, actorder):
    """Half ties, zero/negative scales, unaligned row tiles and lazy updates exercise the grid's exact math."""
    q = Quantizer.of(Format.parse(fmt), Target("llama_cpp").plan(Format.parse(fmt), "dense"), grid)
    if fmt == Q4K:
        scale = torch.tensor([63, 2, 0, 4, 8, 16, 32, 1], dtype=torch.float32)
        zero = torch.tensor([63, 2, 47, 4, 8, 16, 32, 1], dtype=torch.float32)
    else:
        scale = torch.tensor([127, -128, 0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64]) / 1024
        zero = None
    scale = scale.reshape(1, 1, -1).expand(2, 3, -1).contiguous().cuda()
    zero = None if zero is None else zero.reshape(1, 1, -1).expand_as(scale).contiguous().cuda()
    columns = torch.arange(256, device="cuda") // q.group(256)
    values = (torch.arange(256, device="cuda") % 65 - 32).float() + 0.5
    w = values * scale[..., columns]
    if zero is not None:
        w -= zero[..., columns]
    h = torch.eye(256, device="cuda").expand(2, -1, -1).clone()
    h += torch.ones_like(h) / 512
    h.diagonal(dim1=-2, dim2=-1).add_(torch.arange(256, device="cuda") / 256)
    solve.solve_offsets()
    old = solve.gptq(w, h.clone(), q, scale, zero, actorder=actorder, block=64, fused_grid=False)
    new = solve.gptq(w, h.clone(), q, scale, zero, actorder=actorder, block=64, fused_grid=True)
    records = solve.solve_offsets()
    assert [(r["impl"], r["fused_grid"]) for r in records] == [("torch", False), ("triton", True)]
    assert torch.equal(old.view(torch.int32), new.view(torch.int32))
    rows, original = torch.arange(6), w.cpu().reshape(6, 256)
    params = {"format": fmt, "scale": scale.cpu().reshape(6, -1)}
    if zero is not None:
        params["zero_point"] = zero.cpu().reshape(6, -1)
    assert pack({**params, "codes": old.reshape(6, 256)}, rows, original, "vector") == pack(
        {**params, "codes": new.reshape(6, 256)}, rows, original, "vector"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("fmt,grid,kind", [(Q4K, "q4k", "Q4_K"), (Q6K, "q6k", "Q6_K")])
def test_tiny_fused_grid_build_matches_opt_out(tmp_path, fmt, grid, kind):
    """A full layer pass must preserve stored bytes and record the user-selected default/opt-out path."""
    src, calib, template, _, _ = _source(tmp_path, {**WIDE, "head_dim": 64})
    cfg = BuildConfig(
        model=str(src),
        format=fmt,
        grid=grid,
        devices=["cuda:0"],
        steps=[{"gptq": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        exporter={
            "gguf": {
                "template": str(template),
                "tensor_types": [
                    {
                        "pattern": r"re:blk\.\d+\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight$",
                        "type": kind,
                    }
                ],
            }
        },
        out=str(tmp_path / "on"),
    )
    on = cfg.build(log=lambda _: None)
    off = BuildConfig.from_file(
        on / "resolved.yaml", overrides=[f"out={tmp_path / 'off'}", "steps.0.gptq.fused_grid=false"]
    ).build(log=lambda _: None)
    assert (on / "model-Q4_K_M.gguf").read_bytes() == (off / "model-Q4_K_M.gguf").read_bytes()
    for path, expected in ((on, True), (off, False)):
        entries = json.loads((path / "build.json").read_text())["solves"]["entries"]
        assert entries and all(e["fused_grid"] is expected for e in entries)
