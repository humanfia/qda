"""The ``q6k`` grid against llama.cpp's own Q6_K fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qda_quant.format import Format
from qda_quant.gguf import file, kernel
from qda_quant.gguf.q6k import GROUP, Q6K_FORMAT
from qda_quant.solve.grid import GRIDS
from qda_quant.step import Refused


def _plan():
    """A kernel plan `check_grid` can name in its message; the grids read the format and this scale dtype."""
    from types import SimpleNamespace

    return SimpleNamespace(kernel="test", scale_dtype="bf16")


def _ggml_values(x: torch.Tensor) -> np.ndarray:
    """What ``dequantize_row_q6_K`` would return for ggml's own fit of `x`."""
    d, scales, codes = kernel.q6k_fields(kernel.quantize_q6k(x, workers=1))
    fp16 = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)
    per_group = codes.reshape(-1, 16, GROUP).astype(np.float32) - 32
    return (fp16[:, None, None] * scales.astype(np.float32)[:, :, None] * per_group).reshape(x.shape)


def test_the_grid_reproduces_llama_cpps_own_q6_k_fit() -> None:
    """`value` on the grid's own scale and codes is ggml's dequantization, bit for bit.

    The grid reads its scales back out of the records ggml wrote, so this is the check that the reading, the
    sign convention and the group layout all line up: an off-by-one group or an unsigned code still produces
    plausible numbers, and only an exact comparison against ggml's own values rejects them.
    """
    grid = GRIDS["q6k"]
    fmt = Format.parse(Q6K_FORMAT)
    rng = np.random.default_rng(0)
    w = torch.from_numpy((rng.standard_normal((4, 512)) * 0.05).astype(np.float32))

    scale, zero, codes = grid.fit_codes(fmt, w)
    assert zero is None, "Q6_K is symmetric and has no zero point"
    assert scale.shape == (4, 512 // GROUP)
    assert codes.dtype is torch.int8 and int(codes.min()) >= -32 and int(codes.max()) <= 31

    wide = scale.repeat_interleave(GROUP, -1)
    assert np.array_equal(grid.value(codes.float(), wide, None).numpy(), _ggml_values(w))
    # and the grid's own coding of the weights is what ggml chose, so a solver starts from ggml's state
    assert torch.equal(grid.codes(w, wide, None, -32, 31).to(torch.int8), codes)


def test_the_grid_takes_only_the_q6_k_format_and_whole_groups() -> None:
    """A format with other bits or groups, and a row that is not whole groups, are refused not rounded."""
    grid = GRIDS["q6k"]
    assert grid.refusal(Format.parse(Q6K_FORMAT), None) is None
    assert "w6a16-sym-g16" in (grid.refusal(Format.parse("w4a16-asym-g32"), None) or "")
    with pytest.raises(Refused, match="whole number of 16-value groups"):
        grid.fit(Format.parse(Q6K_FORMAT), torch.zeros(1, 24))


def test_the_grid_keeps_scales_as_ggml_fitted_them(ggml_lib: Path) -> None:
    """`range_rule` is False, and the probe behind that: ggml's scales factor back exactly, a moved one does not.

    What a block can express is narrow but not a single point: the int8 factors are fixed by the ratios
    between the group scales, so rescaling a whole block stays representable while moving one group does not.
    This pins the measured ends -- ggml's own scales factor exactly, a per-group perturbation in none -- which
    is why a grid that fitted these scales group by group could not write them.
    """
    kernel.use_ggml(str(ggml_lib))
    try:
        grid = GRIDS["q6k"]
        assert grid.range_rule is False
        rng = np.random.default_rng(0)
        w = torch.from_numpy((rng.standard_normal((2, 1024)) * 0.05).astype(np.float32))
        scale, _ = grid.fit(Format.parse(Q6K_FORMAT), w)

        products = scale.reshape(-1, 16).numpy().astype(np.float32)
        for block, moved in ((products, False), (products * np.float32(1.001), True)):
            exact = 0
            for row in block:
                k = int(np.argmax(np.abs(row)))
                d = np.float16(row[k] / -128.0)
                if d == 0:
                    continue
                s = row / np.float32(d)
                exact += bool(np.all(s == np.rint(s)) and np.all(np.abs(s) <= 128))
            assert (exact == 0) if moved else (exact == len(block))
    finally:
        kernel.use_ggml(None)


def test_a_step_quantizes_each_unit_on_the_grid_that_serves_its_format() -> None:
    """A step names one grid; a unit the build stores in another format takes the grid that speaks it.

    The step's own grid is preferred wherever it serves the format, so a build with no overrides is unchanged.
    This is what lets one recipe store some tensors Q4_K and some Q6_K: the format is the unit's, and the
    grid follows it.
    """
    from qda_quant.steps.rtn import RTN

    step = RTN(grid="q4k")
    assert step.grid_for(Format.parse("w4a16-asym-g32"), None) == "q4k"
    assert step.grid_for(Format.parse(Q6K_FORMAT), None) == "q6k"
    assert RTN(grid="q6k").grid_for(Format.parse("w4a16-asym-g32"), None) == "q4k"


def test_a_format_no_grid_serves_is_refused_not_fitted_on_the_wrong_grid() -> None:
    """The guard the rule must keep: an unservable format refuses, it does not fall back to any grid.

    Measured before the rule existed, and the behaviour it has to preserve: a unit format the step's grid
    cannot store is refused by `check_grid`, never quantized on a grid whose records cannot hold it.
    """
    from qda_quant.steps.rtn import RTN

    unservable = Format.parse("w4a16-sym-g128")  # no gguf grid stores symmetric 4-bit in groups of 128
    step = RTN(grid="q4k")
    assert step.grid_for(unservable, None) == "q4k", "an unservable format keeps the step's own grid"
    with pytest.raises(Refused, match="Q4_K stores 4-bit asymmetric codes"):
        step.check_grid(unservable, _plan())


def test_a_plain_rtn_fit_round_trips_through_the_grid(ggml_lib: Path) -> None:
    """A deterministic unweighted control: the grid's codes and scales rebuild ggml's own Q6_K values.

    The evidence behind `range_rule = False` was sampled from importance-weighted checkpoints, where the
    scales are the harder case; this fits a tiny tensor with no importance at all, so the property is pinned
    on both kinds without depending on any build's output.
    """
    kernel.use_ggml(str(ggml_lib))
    try:
        grid = GRIDS["q6k"]
        rng = np.random.default_rng(7)
        w = torch.from_numpy((rng.standard_normal((3, 768)) * 0.1).astype(np.float32))
        scale, _, codes = grid.fit_codes(Format.parse(Q6K_FORMAT), w)
        wide = scale.repeat_interleave(GROUP, -1)
        assert np.array_equal(grid.value(codes.float(), wide, None).numpy(), _ggml_values(w))
    finally:
        kernel.use_ggml(None)


def test_a_build_with_a_q6_k_override_writes_solved_codes_end_to_end(tmp_path: Path, ggml_lib: Path) -> None:
    """One real build: `qda build` with an override, through the solver, the writer and the file it wrote.

    Every other test here checks a piece -- the grid picks `q6k`, the exporter admits the format, the writer
    packs records. Each passed while the feature was dead, four separate times, because nothing ran the whole
    path: the exporter refused the override, then the writer had no `formats` keyword, then the rounding
    shortcut skipped the layer pass. This is the test that would have caught all of them at once.

    What it pins: the tensor is Q6_K in the file, its bytes are the build's solution rather than the
    template's rounding, and the tensors around it are untouched.
    """

    from qda_quant.gguf import rule
    from qda_quant.gguf.tensor_types import resolve_types

    from .test_export import WIDE, _source

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    types = resolve_types(file.Template(template), [])
    promoted = next(n for n, t in types.items() if t == rule.Q6_K and n.endswith("ffn_down.weight"))
    module = f"model.layers.{promoted.split('.')[1]}.mlp.down_proj"

    def run(name: str, overrides: list[tuple[str, str]], step: object | None = None) -> Path:
        from qda_quant import REGISTRY, Recipe
        from qda_quant.calib.sources import FileSource
        from qda_quant.engine.build import build

        out = build(
            str(src),
            Recipe(
                steps=[step or REGISTRY["gptq"](grid="q4k", impl="torch")],
                calib=FileSource(path=str(calib), samples=1, seqlen=8),
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            format_overrides=overrides,
            exporter={
                "gguf": {
                    "llama_cpp": str(checkout),
                    "template": str(template),
                    "libggml": str(ggml_lib),
                }
            },
            log=lambda _: None,
        )
        return out / "model-Q4_K_M.gguf"

    plain = run("plain", [])
    solved = run("solved", [(module, Q6K_FORMAT)])

    a, b = file.Template(plain), file.Template(solved)
    assert b.tensors[promoted].tensor_type == rule.Q6_K, "the override changed the type the mix chose"
    assert b.raw(promoted) != a.raw(promoted), "the override wrote the template's rounding, not a solution"
    others = [n for n in a.tensors if n != promoted]
    assert all(a.raw(n) == b.raw(n) for n in others), "an override reached a tensor it does not name"

    record = json.loads((solved.parent / "build.json").read_text())["gguf"]["rounded_from_template"]
    assert promoted not in record, "a solved tensor is still recorded as rounded from the template"
    assert promoted in json.loads((plain.parent / "build.json").read_text())["gguf"]["rounded_from_template"]

    # Plain RTN is the path that took the rounding shortcut. Its bytes cannot show the difference -- RTN's
    # solution of an unchanged weight IS ggml's own fit, so an override writes the same records either way.
    # What the shortcut does change is whether the build goes through the layer pass at all, and so whether
    # the override is honoured or silently ignored; that is what the record reports and what this pins.
    from qda_quant.steps.rtn import RTN

    rtn = run("rtn", [(module, Q6K_FORMAT)], step=RTN(grid="q4k"))
    rtn_record = json.loads((rtn.parent / "build.json").read_text())["gguf"]["rounded_from_template"]
    assert promoted not in rtn_record, (
        "an RTN build with an override took the rounding shortcut: the tensor is recorded as written from "
        "the template, so the override was accepted and ignored"
    )


def test_the_weighted_fit_is_ggmls_own_imatrix_q6_k(ggml_lib: Path) -> None:
    """`fit_weighted` is what ``llama-quantize --imatrix`` fits for a Q6_K tensor, not merely something else.

    A weighted fit that simply differs from the plain one proves nothing -- any perturbation does that. The
    records ggml writes for the same weights and importances are the reference, and the grid's scales have to
    be the ones inside them.
    """
    from qda_quant.gguf.file import row_bytes, type_id

    kernel.use_ggml(str(ggml_lib))
    try:
        grid = GRIDS["q6k"]
        rng = np.random.default_rng(3)
        w = torch.from_numpy((rng.standard_normal((4, 512)) * 0.05).astype(np.float32))
        imp = torch.from_numpy((rng.random(512) * 4 + 0.05).astype(np.float32))

        scale, zero = grid.fit_weighted(w.reshape(1, 4, 512), imp.reshape(1, 512))
        assert zero is None, "Q6_K is symmetric"

        kind = type_id("Q6_K")
        records = kernel.quantize_weighted_chunk(kind, w, row_bytes(kind, 512), imp)
        d, scales, _ = kernel.q6k_fields(records)
        fp16 = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)
        want = (fp16[:, None] * scales.astype(np.float32)).reshape(4, 32)

        assert np.array_equal(scale.reshape(4, 32).numpy(), want), "not ggml's own weighted scales"
        plain, _ = grid.fit(Format.parse(Q6K_FORMAT), w)
        assert not np.array_equal(plain.numpy(), want), "the importances changed nothing"
    finally:
        kernel.use_ggml(None)


def test_the_imatrix_clip_serves_both_gguf_grids_and_refuses_the_rest() -> None:
    """A Q4_K_M mix with Q6_K promotions weighs tensors in both grids, so the clip must accept both.

    It stays refused everywhere else: the clip calls `fit_weighted`, which a grid without a weighted fit does
    not have, and refusing at plan time is what keeps that from being an AttributeError mid-build.
    """
    from qda_quant.gguf.q4k import IMATRIX
    from qda_quant.solve.grid import Grid
    from qda_quant.steps.gptq import GPTQ

    assert set(IMATRIX.only_grid) == {"q4k", "q5k", "q6k", "iq4xs", "iq4nl", "q3k"}
    for name in ("q4k", "q6k"):
        assert type(GRIDS[name]).fit_weighted is not Grid.fit_weighted, (
            f"{name} is declared on the clip but does not override the base's weighted fit"
        )
        GPTQ(grid=name, clip="imatrix").check_grid(Format.parse(GRIDS[name].stores), _plan())
    with pytest.raises(Refused, match="works on grid"):
        GPTQ(grid="absmax", clip="imatrix").check_grid(Format.parse("w4a16-asym-g32"), _plan())
    with pytest.raises(NotImplementedError, match="no importance-weighted fit"):
        GRIDS["absmax"].fit_weighted(torch.zeros(1, 32), torch.ones(32))


@pytest.mark.parametrize("case", ["one zero group", "every group zero", "no zero group"])
def test_a_weighted_fit_may_zero_a_group_the_unweighted_one_does_not(ggml_lib: Path, case: str) -> None:
    """An importance of zero across a group's columns is valid, and the writer must still write the tensor.

    The writer recovers a zero-scale group's unobservable fields from an unweighted fit of the template. That
    is the fit the build made only when it did not weigh its own; once an importance matrix is in play a group
    can be zero in the weighted fit and not in the unweighted one, and the reference then says nothing about
    this build's bytes. Refusing there rejects a valid matrix -- `llama-imatrix` reports zero mean squares for
    a column no calibration row exercised.
    """
    from qda_quant.gguf.export import q6k_bytes

    kernel.use_ggml(str(ggml_lib))
    try:
        grid = GRIDS["q6k"]
        torch.manual_seed(812)
        w = torch.randn(1, 1, 256) * 0.05
        imp = torch.ones(1, 256)
        if case == "one zero group":
            imp[:, 64:80] = 0
        elif case == "every group zero":
            imp[:] = 0

        scale, _ = grid.fit_weighted(w, imp)
        wide = scale.reshape(1, -1).repeat_interleave(GROUP, -1)
        codes = grid.codes(w.reshape(1, 256), wide, None, -32, 31)
        qp = {"format": Q6K_FORMAT, "codes": codes.to(torch.int8), "scale": scale.reshape(1, -1).float()}

        raw = q6k_bytes(qp, torch.arange(1), w.reshape(1, 256), "blk.0.ffn_down.weight", 1)
        assert len(raw) == 210, "a Q6_K super-block is 210 bytes"

        d, scales, _ = kernel.q6k_fields(raw)
        zeros = int((scale.reshape(-1) == 0).sum())
        assert zeros == {"one zero group": 1, "every group zero": 16, "no zero group": 0}[case]
        # Every group zero is ggml's underflow: it writes the block out as zeros, scale included.
        got = float(np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2")[0])
        assert (got == 0) is (case == "every group zero")
        # A zero-scale group decodes to zero whatever its codes are, so the values are right either way.
        fp16 = np.frombuffer(d.astype("<u2").tobytes(), dtype="<f2").astype(np.float32)
        value = fp16[:, None] * scales.astype(np.float32)
        assert np.array_equal((value == 0).reshape(-1), (scale.reshape(-1) == 0).numpy())
    finally:
        kernel.use_ggml(None)
