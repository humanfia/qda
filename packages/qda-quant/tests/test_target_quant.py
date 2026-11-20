from dataclasses import replace
from pathlib import Path

import pytest
import torch
from qda_quant import PRESETS, Format, Refused, Target
from qda_quant.format import Int8W8A8, W8A8Fp8
from qda_quant.output.serve_policy import allowed
from qda_quant.solve.quant import Quantizer, fp8_scale_parts, int8_token_parts, round_fp8
from qda_quant.target import TARGETS, UNVERIFIED_ON_HARDWARE, KernelPlan, Table


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_every_preset_has_a_plan(name, kind):
    """W8A8 has a dense plan only: QDA writes no FP8 expert bank, so no table plans one. NVFP4 W4A4 is
    planned on Blackwell only: sm90 refuses it."""
    target = Target.parse("sm103") if PRESETS[name].act == "nvfp4_block" else Target()
    if PRESETS[name].weight_type == "fp8" and kind == "moe":
        with pytest.raises(Refused, match="no FP8 expert row"):
            target.plan(PRESETS[name], kind)
        return
    plan = target.plan(PRESETS[name], kind)
    assert plan.kernel
    assert plan.zero_point == (not PRESETS[name].symmetric)
    assert plan.act_dtype == {"fp8_token": "fp8_e4m3", "nvfp4_block": "nvfp4"}.get(PRESETS[name].act, "bf16")


def test_refusals_say_why():
    with pytest.raises(Refused, match="0.33"):
        Target.parse("sm90@0.27.1").plan(Format.parse("w3a16-asym-g128"), "dense")
    Target.parse("sm90@0.33.0").plan(Format.parse("w3a16-asym-g128"), "dense")
    with pytest.raises(Refused, match="group 128"):
        Target().plan(Format.parse("w4a8-sym-g64"), "moe")
    with pytest.raises(Refused, match="symmetric weights only"):
        Target().plan(Format.parse("w4a8-asym-g128"), "dense")


@pytest.mark.parametrize(
    ("target", "capability", "want"),
    [
        (None, (9, 0), "sm90"),
        (None, (8, 0), "sm80"),
        (None, (8, 6), "sm80"),
        (None, (8, 9), "sm89"),
        (None, (10, 0), "sm100"),
        (None, (10, 3), "sm103"),
        (None, (12, 0), "sm120"),
        ("sm90", (8, 9), "sm90"),  # the target is where it is served, not where it is built
        ("a100", (9, 0), "sm80"),
        ("a100", (8, 6), "sm80"),
        ("llama_cpp", (8, 9), "llama_cpp"),  # llama.cpp serves the file on any card
    ],
)
def test_a_target_on_cards_resolves_when_one_table_covers_them(target, capability, want):
    """An unset target takes the table of the card, so an Ampere build no longer plans Hopper rows."""
    assert Target.on_cards(target, [("cuda:0", capability)]).arch == want
    assert Target.on_cards(target, []).arch == Target.parse(target).arch, "no card visible: unchanged"


@pytest.mark.parametrize("capability", [(7, 0), (12, 1)])
def test_an_unset_target_refuses_when_no_table_serves_the_cards(capability):
    """The limitation leads; the cross-card build is offered as a different action."""
    with pytest.raises(Refused) as e:
        Target.on_cards(None, [("cuda:0", capability)])
    text = str(e.value)
    assert f"no supported target serves compute capability {capability[0]}.{capability[1]} yet" in text
    assert "QDA supports sm120 (12.0) and sm103 (10.3) and sm100 (10.0) and sm90 (9.0) and sm89 (8.9)" in text
    assert (  # the cross-card build is the `fix:` line: the refusal of the machine names what to do instead
        e.value.fix
        == "build it here with --set target=sm120 (or sm103 or sm100 or sm90 or sm89 or sm80); that checkpoint is "
        "for that card, not this one"
    )
    assert e.value.code == 3  # the card, not the input


def test_cards_that_disagree_have_no_default_but_an_explicit_target_stands():
    """No one table covers both cards, so no default exists; a named serving target is the user's call."""
    cards = [("cuda:0", (9, 0)), ("cuda:1", (8, 0))]
    with pytest.raises(Refused, match="compute capability 9.0 and 8.0"):
        Target.on_cards(None, cards)
    assert Target.on_cards("sm90", cards) == Target.parse("sm90")


def test_an_explicit_target_on_other_cards_is_kept_and_noted():
    """The target is the serving target: a build on an 8.9 card may plan sm90, and the build log says so."""
    cards = [("cuda:0", (8, 9))]
    t = Target.on_cards("sm90", cards)
    assert t == Target.parse("sm90")
    note = t.on_other_cards(cards)
    assert note is not None and "sm90" in note and "cuda:0 is compute capability 8.9" in note
    assert Target.parse("sm90").on_other_cards([("cuda:0", (9, 0))]) is None, "the card it plans for: no note"


def test_a_cuda_device_with_no_usable_driver_is_no_card_information(monkeypatch):
    """Planning on a machine with no GPU for serving on one: keep the default, never ask the driver.

    `resolve_devices` sizes its list from `device_count`, so a host with no CUDA driver can still yield
    cuda-typed devices (a test, or a caller pinning card indices); asking for their capability raised on a
    GPU-less host. With no usable driver there is no card information, which is the
    CPU-only case: an unset target keeps the default and a named target is honoured.
    """
    from qda_quant.engine.build import _target_for
    from qda_quant.output.exporters import CompressedTensors

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    assert _target_for(CompressedTensors(), None, devs).arch == "sm90"
    assert _target_for(CompressedTensors(), "a100", devs).arch == "sm80"


@pytest.mark.parametrize("name", ["blackwell"])
def test_an_unknown_architecture_names_the_fix(name):
    """A name no table claims ("blackwell" is datacenter and consumer cards at once) is refused with what to
    pass, that no target serves it yet, and what can still be built there."""
    with pytest.raises(Refused) as e:
        Target.parse(name)
    text = str(e.value)
    assert "pass --set target=NAME" in text and "llama_cpp, sm100, sm103, sm120, sm80, sm89, sm90" in text
    assert "no target serves this GPU yet" in text
    assert (
        "you can build a checkpoint here to serve on a supported card with target sm120 (12.0) or sm103 (10.3) "
        "or sm100 (10.0) or sm90 (9.0) or sm89 (8.9) or sm80 (8.0/8.6)" in text
    )
    assert "or build for llama_cpp, whose table names no card" in text  # what can be built for this card


def test_a_name_no_one_uses_gets_no_pending_note():
    """Only the cards still waiting for a table get the pending note; a typo is told the known names."""
    with pytest.raises(Refused) as e:
        Target.parse("sm9O")
    assert "unknown architecture" in str(e.value) and "no target serves" not in str(e.value)


def test_a_non_numeric_vllm_version_is_refused_at_parse():
    """The version reaches `_version` at plan time; a malformed one must already be a target error."""
    with pytest.raises(Refused, match=r"target 'sm90@abc': vLLM version 'abc' is not numeric"):
        Target.parse("sm90@abc")


def test_dense_wna16_disables_only_machete_by_default():
    """Default sm90 serve: Machete off and nothing else pinned, so vLLM binds Humming (0.32+) or Marlin (before)."""
    machete_off = {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
    p = Target.parse("h200@0.27.1").plan(PRESETS["w4a16"], "dense")
    assert p.kernel == "MarlinLinearKernel" and p.serve_env == machete_off and p.serve_args == {}
    for fmt in (PRESETS["w4a16"], Format.parse("w4a16-sym-g128"), Format.parse("w4a16-sym-g32")):
        p33 = Target.parse("sm90@0.33.0").plan(fmt, "dense")
        assert p33.kernel == "HummingLinearKernel" and p33.serve_env == machete_off and p33.serve_args == {}
        assert not any("unverified" in n for n in p33.notes)
    marlin = Target.parse("h200:marlin").plan(PRESETS["w4a16"], "dense")
    assert marlin.kernel == "MarlinLinearKernel" and marlin.serve_args == {"linear_backend": "marlin"}
    assert marlin.serve_env == machete_off
    assert Target.parse("h200@0.32.0:marlin").plan(PRESETS["w4a16"], "dense").serve_env == {
        "VLLM_DISABLED_KERNELS": "HummingLinearKernel,MacheteLinearKernel"
    }
    with pytest.raises(Refused, match="backend"):
        Target.parse("h200:exllama").plan(PRESETS["w4a16"], "dense")
    with pytest.raises(Refused, match="0.32"):
        Target.parse("h200@0.27.1:humming").plan(PRESETS["w4a16"], "dense")
    # W3 is Humming's: its env must not disable Humming
    assert "Humming" not in str(Target.parse("sm90@0.33.0").plan(PRESETS["w3a16"], "dense").serve_env)


@pytest.mark.parametrize("name", ["h200:machete", "sm90@0.33.0:machete", "sm90@0.27.1:machete"])
def test_a_machete_target_is_refused_with_its_fix(name):
    rest = name.removesuffix(":machete")
    with pytest.raises(Refused, match=rf"^target '{name}': drop :machete; use target '{rest}' "):
        Target.parse(name)


@pytest.mark.parametrize("stored", ["sm90@0.33.0:machete", "h200:machete"])
def test_a_stored_machete_target_reads_as_plain_sm90(stored):
    """A checkpoint built for ``:machete`` opens (soup, scale_kd, capture read its target) as the plain table."""
    assert Target.stored(stored) == Target.parse(stored.removesuffix(":machete"))


def test_a_published_w3a16_serve_section_that_names_machete_is_read(tmp_path):
    """The published W3A16 checkpoints name Machete for their int4 head: the record reads, its line disables Machete.

    The kernels `qda-kl` then expects are this card's (`qda_kl.serve.card_kernels`, tests/test_card_kernels.py).
    """
    import json

    from qda_quant.output.serve_command import serve_command_line, serve_section
    from qda_quant.output.serve_policy import check

    serve = {"vllm": "0.33.0", "kernels": {"dense": ["HummingLinearKernel", "MacheteLinearKernel"]}}
    check(serve)  # the judge's intake checks a record as stored
    (tmp_path / "build.json").write_text(json.dumps({"target": "sm90@0.33.0", "serve": serve}))
    assert serve_section(tmp_path)["kernels"] == serve["kernels"]
    assert "MacheteLinearKernel" in serve_command_line(tmp_path)  # off: VLLM_DISABLED_KERNELS


@pytest.mark.parametrize(
    ("fmt", "kernel"),
    [("w4a16-sym-g128", "HummingIndexedExperts"), ("w4a16-asym-g128", "MarlinExperts"),
     ("w4a16-asym-g64", "MarlinExperts"), ("w4a16-asym-g32", "HummingIndexedExperts")],
)  # fmt: skip
def test_default_sm90_experts_are_the_ones_vllm_auto_binds(fmt, kernel):
    """Humming takes symmetric (and int4 g32) banks; vLLM 0.33 serves the other asymmetric ones on Marlin."""
    from qda_quant.target import sm90_moe_binds

    t = Target.parse("sm90@0.33.0")
    plan = t.plan(Format.parse(fmt), "moe")
    assert plan.kernel == kernel and plan.serve_args == {}
    assert sm90_moe_binds(t, Format.parse(fmt), (2048, 768), {}, {}) == kernel


def test_merge_serve():
    from qda_quant.target import merge_serve

    t = Target.parse("sm90@0.33.0")
    s = merge_serve(
        # a build stores the preset's routed experts symmetric (Format.symmetric_defaulted)
        {"dense": t.plan(PRESETS["w4a16"], "dense"), "moe": t.plan(Format.parse("w4a16-sym-g128"), "moe")},
        t.vllm,
    )
    assert s["kernels"] == {"dense": ["HummingLinearKernel"], "moe": ["HummingIndexedExperts"]}
    assert s["args"] == {} and s["vllm"] == "0.33.0"
    assert s["env"] == {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
    w3 = merge_serve({"moe": Target.parse("sm90@0.27.1").plan(PRESETS["w3a16"], "moe")}, "0.27.1")
    assert w3["plugins"] == ["qda_humming_moe"]


def test_newer_vllm_rows_are_marked_unverified():
    # w3a16 moe on 0.33.0 was observed on a card on 2026-11-09, so it is no longer unverified.
    # w3a16 moe on 0.32.0 stands in for a row that has not been: a version with no observed rows at all.
    p = Target.parse("h200@0.33.0").plan(PRESETS["w3a16"], "moe")
    assert p.kernel == "HummingIndexedExperts" and not any("unverified" in n for n in p.notes)
    assert p.act_order == ("none", "static")
    unobserved = Target.parse("h200@0.32.0").plan(PRESETS["w3a16"], "moe")
    assert any("unverified" in n for n in unobserved.notes)
    assert not any("unverified" in n for n in Target.parse("sm90@0.27.1").plan(PRESETS["w4a16"], "moe").notes)


@pytest.mark.parametrize("name", list(PRESETS))
def test_codes_in_range_and_fake_is_idempotent(name):
    fmt = PRESETS[name]
    target = Target.parse("sm103") if fmt.act == "nvfp4_block" else Target()
    q = Quantizer.of(fmt, target.plan(fmt, "dense"), "absmax")
    w = torch.randn(2, 16, 256)
    s, z = q.params(w)
    c = q.quantize(w, s, z)
    assert c.min() >= fmt.qmin and c.max() <= fmt.qmax
    if fmt.weight_type == "fp8":  # a code is the E4M3 value itself, not an integer
        assert torch.equal(c, c.to(torch.float8_e4m3fn).float())
    else:
        assert torch.equal(c, c.round())
    wq = q.fake(w, s, z)
    assert torch.equal(q.fake(wq, s, z), wq)


def test_fp8_grid_matches_vllm_convert_bf16_scales_to_fp8():
    s = torch.rand(4, 8) + 0.01
    f8, chan = fp8_scale_parts(s)
    # vLLM: per-row fp8 quant of the bf16 scales (x times 1/scale, as `QuantFP8`), then /8 to e4m3 and x8 into
    # the per-channel factor; `/ c` rounds differently from `* (1 / c)` in about 1 draw in 65
    sb = s.to(torch.bfloat16).float()
    c = sb.amax(-1, keepdim=True) / 448
    ref = ((sb * c.reciprocal()).to(torch.float8_e4m3fn).float() / 8).to(torch.float8_e4m3fn).float()
    assert torch.equal(f8, ref) and torch.allclose(chan, c * 8)


def test_int8_cut_rounds_as_the_kernel_does():
    """The cases a by-the-formula ``round(x / s)`` or a guarded zero row gets wrong (`test_vllm_gpu.py`: the kernel)."""
    x = torch.tensor(
        [
            [1.0, -0.9409448504447937, 0.0],  # x * (127 / absmax) rounds to -119, x / (absmax / 127) to -120
            # torch's `127.0 / absmax` (a reciprocal, not a division) rounds this one to -125
            [1.003999948501587, -0.9921416640281677, 0.0],
            [127.0, 0.5, 1.5],  # half to even
            [0.0, 0.0, 0.0],  # no floor: scale 0, codes 0
            [1e-40, -1e-40, 0.0],  # 127 / absmax overflows: saturated codes, and 0 * inf is code 0
        ]
    )
    codes, scale = int8_token_parts(x)
    want = [[127, -119, 0], [127, -126, 0], [127, 0, 2], [0, 0, 0], [127, -128, 0]]
    assert codes.tolist() == want and codes.dtype == scale.dtype == torch.float32
    assert torch.equal(scale, x.abs().amax(-1, keepdim=True) / torch.tensor(127.0))


def test_round_fp8_is_nearest_served_value():
    w = torch.randn(32, 128)
    f8, chan = fp8_scale_parts(w.abs().amax(-1, keepdim=True).expand(32, 1) / 7)
    f8 = f8.expand(32, 128)
    q, v = round_fp8(w, f8, chan, -8, 7)
    for d in range(-8, 8):
        other = (torch.full_like(w, d) * f8).to(torch.float8_e4m3fn).float() * chan
        assert ((w - v).abs() <= (w - other).abs() + 1e-7).all()


def test_plain_pytest_cannot_see_a_gpu():
    assert not torch.cuda.is_available()


def test_default_is_vllm_0_33_with_the_observed_serve_blocks():
    """The exact strings qda-kl compares against what vLLM 0.33.0 logs (observed on H200)."""
    from qda_quant.target import merge_serve

    t = Target()
    assert t.vllm == "0.33.0"
    sym = Format.parse("w4a16-sym-g128")
    off = {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
    moe = merge_serve({"dense": t.plan(sym, "dense"), "moe": t.plan(sym, "moe")}, t.vllm)
    assert moe == {"vllm": "0.33.0", "env": off, "args": {}, "plugins": [],
                   "kernels": {"dense": ["HummingLinearKernel"], "moe": ["HummingIndexedExperts"]}}  # fmt: skip
    marlin = Target.parse("sm90:marlin")
    pinned = merge_serve({"dense": marlin.plan(sym, "dense"), "moe": marlin.plan(sym, "moe")}, t.vllm)
    assert pinned == {"vllm": "0.33.0", "env": off, "args": {"linear_backend": "marlin", "moe_backend": "marlin"},
                      "plugins": [], "kernels": {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]}}  # fmt: skip
    w3 = merge_serve({"dense": t.plan(PRESETS["w3a16"], "dense")}, t.vllm)
    assert w3 == {
        "vllm": "0.33.0",
        "env": off,
        "args": {},
        "plugins": [],
        "kernels": {"dense": ["HummingLinearKernel"]},
    }
    assert t.plan(PRESETS["w3a16"], "moe").kernel == "HummingIndexedExperts"
    # w3a16 moe joined the observed set on 2026-11-09, so it belongs in the loop below now.
    for fmt, kind in (("w4a16-sym-g128", "moe"), ("w4a16", "moe"), ("w3a16", "dense"), ("w4a16", "dense"),
                      ("w4a8", "dense"), ("w4a8", "moe"), ("w3a16", "moe")):  # fmt: skip
        assert not any("unverified" in n for n in t.plan(Format.parse(fmt), kind).notes), (fmt, kind)


def test_w4a8_zero_group_next_to_a_large_group_gives_finite_codes():
    """An exact-zero group beside a large one in the same row: its FP8 group factor must not round to 0 (NaN codes)."""
    from qda_quant import Format, Target
    from qda_quant.solve.quant import Quantizer

    fmt = Format.parse("w4a8")
    q = Quantizer.of(fmt, Target().plan(fmt, "dense"), "absmax")
    assert q.fp8_scales
    g = q.group(512)
    for big in (0.05, 1.0, 8.0):
        w = torch.zeros(2, 512)
        w[:, g:] = torch.linspace(-big, big, 512 - g)
        s, z = q.params(w)
        codes = q.quantize(w, s, z)
        assert torch.isfinite(codes).all(), big
        assert torch.isfinite(q.fake(w, s, z)).all(), big
        assert torch.equal(q.fake(w, s, z)[:, :g], torch.zeros(2, g))  # the zero group stays zero
    # a row with no tiny group keeps its scales exactly
    w = torch.randn(4, 512)
    lo = torch.stack([c.amin(-1) for c in w.split(g, -1)], -1).clamp(max=0)
    hi = torch.stack([c.amax(-1) for c in w.split(g, -1)], -1).clamp(min=0)
    assert torch.equal(q.params(w)[0], torch.maximum(-lo, hi) / ((fmt.qmax - fmt.qmin) / 2))


def test_w4a8_floor_covers_tiny_rows_and_the_hmse_clip():
    """A row whose whole scale is tiny (the kernel clamps chan), and the hmse clip, also keep f8 > 0."""
    from qda_quant import Format, Target
    from qda_quant.clips import CLIPS
    from qda_quant.solve.quant import Quantizer

    fmt = Format.parse("w4a8")
    q = Quantizer.of(fmt, Target().plan(fmt, "dense"), "absmax")
    g = q.group(512)
    w = torch.zeros(2, 512)
    w[:, g:] = torch.linspace(-1e-5, 1e-5, 512 - g)  # row max scale below 1/512: chan is clamped
    w[:, :g] = 1e-9
    s, z = q.params(w)
    assert torch.isfinite(q.quantize(w, s, z)).all()
    w = torch.zeros(1, 2, 512)
    w[..., g:] = torch.linspace(-1.0, 1.0, 512 - g)
    s, z = CLIPS["hmse"].choose(q, w, torch.ones(1, 512))
    assert torch.isfinite(q.quantize(w, s, z)).all()


def test_head_binding_refuses_a_vllm_its_rule_was_not_read_from():
    """The sm90 binding rule replays one vLLM's kernel selection; another version's choice is refused, not guessed."""
    head = Format.parse("w4a16-sym-g64")
    assert Target.parse("sm90@0.33.0").binds(head, (256, 128), {}) == "HummingLinearKernel"
    with pytest.raises(Refused, match="read from vLLM 0.33.0"):
        Target.parse("sm90@0.34.0").binds(head, (256, 128), {})


@pytest.mark.parametrize(
    ("name", "kind", "kernel", "act"),
    [
        ("w8a8i", "dense", "CutlassInt8ScaledMMLinearKernel", "int8"),
    ],
)
def test_sm80_rows_bind_with_no_serve_settings_and_say_they_are_unverified(name, kind, kernel, act):
    """vLLM prefers Humming on capability 90 only, so a copy of the sm90 rows' settings would be wrong here."""
    assert Target.parse("a100") == Target.parse("sm86") == Target("sm80", "0.33.0")
    plan = Target.parse("a100").plan(Format.parse(name), kind)
    assert (plan.kernel, plan.act_dtype) == (kernel, act)
    assert not plan.serve_env and not plan.serve_args
    assert UNVERIFIED_ON_HARDWARE in plan.notes


def test_sm80_w4a16_moe_is_observed_on_a_card():
    """An A100 bound MarlinExperts for the preset W4A16 and scored it against a reference."""
    plan = Target.parse("a100").plan(Format.parse("w4a16"), "moe")
    assert (plan.kernel, plan.act_dtype) == ("MarlinExperts", "bf16")
    assert not plan.serve_env and not plan.serve_args
    assert UNVERIFIED_ON_HARDWARE not in plan.notes


def test_sm80_w4a16_dense_is_observed_on_a_card():
    """An A100 bound Marlin for the preset W4A16 and scored it, so this row no longer reads vLLM's source."""
    plan = Target.parse("a100").plan(Format.parse("w4a16"), "dense")
    assert (plan.kernel, plan.act_dtype) == ("MarlinLinearKernel", "bf16")
    assert not plan.serve_env and not plan.serve_args
    assert UNVERIFIED_ON_HARDWARE not in plan.notes


@pytest.mark.parametrize(
    ("kind", "kernel"), [("dense", "HummingLinearKernel"), ("moe", "HummingIndexedExperts")]
)
def test_sm80_serves_w3a16_through_ada_s_humming_row(kind, kernel):
    """Marlin has no int3 and Humming's minimum capability is 75, so Ampere takes the int3 row Ada plans."""
    fmt, a100 = Format.parse("w3a16-sym-g128"), Target.parse("a100")
    plan, ada = a100.plan(fmt, kind), Target.parse("l40s").plan(fmt, kind)
    assert (plan.kernel, plan.act_dtype, plan.zero_point) == (kernel, "bf16", False)
    assert plan.act_order == ada.act_order
    assert plan.notes == ada.notes
    assert not plan.serve_env and not plan.serve_args
    assert any("NVRTC" in n for n in plan.notes)
    assert UNVERIFIED_ON_HARDWARE not in plan.notes, "both rows ran on an A100 and an Ada card"
    if kind == "dense":  # 200 inputs straddle a group of 128, which Marlin refuses and Humming pads
        assert a100.binds(fmt, (256, 200), {}) == kernel
    else:
        assert a100.binds_moe(fmt, (2048, 768), {}) == kernel


@pytest.mark.parametrize(
    ("fmt", "why"),
    [
        (Format.parse("w4a8"), "FP8"),
        (Format.parse("w8a8-sym-g128"), "FP8"),  # what a bare `w8a8` spelled before 0.1.0 refused the name
        (Int8W8A8(symmetric=False), "symmetric per channel"),
        (Int8W8A8(group_size=128), "symmetric per channel"),
        (Format.parse("w8a16-sym-g128"), "int4 and int3 weight-only"),
        (Format.parse("w3a16-sym-g32"), "groups 64 and 128 only"),
    ],
)
def test_sm80_refuses_what_vllm_would_quietly_serve_some_other_way(fmt, why):
    """On Ampere vLLM degrades FP8 W8A8 to W8A16 and loads other INT8 schemes elsewhere; both are refused."""
    with pytest.raises(Refused, match=why):
        Target.parse("a100").plan(fmt, "dense")
    with pytest.raises(Refused, match=why):
        Target.parse("a100").binds(fmt, (256, 256), {})


def test_a_new_table_brings_its_own_gpu_names(monkeypatch):
    """A target needs no core edit: its aliases and what a scorer may apply come from its table, and two tables
    cannot claim one name."""

    def row(target, _fmt, _kind):  # noqa: ANN001, ANN202
        return KernelPlan("K", serve_args={"linear_backend": f"{target.backend}-{target.vllm}"})

    rows = Table(row, aliases=("newcard",), versions=("9.9.9",), backends=("newpin",))
    monkeypatch.setitem(TARGETS._entries, "sm1", rows)
    plan = Target.parse("newcard").plan(Format.parse("w4a16"), "dense")
    assert plan.target.arch == "sm1" and UNVERIFIED_ON_HARDWARE in plan.notes
    assert "newpin-9.9.9" in allowed.__wrapped__()["args"]["linear_backend"]
    monkeypatch.setitem(TARGETS._entries, "sm2", rows)
    with pytest.raises(Refused, match="both claim"):
        Target.parse("newcard")


def test_llama_cpp_serves_the_two_formats_a_gguf_holds_solved_codes_in() -> None:
    """The target declares what a build may *solve units in*, not every type the file may contain.

    A GGUF holds Q5_0, Q8_0, F32 and BF16 tensors too and llama.cpp reads them all, but those reach the file
    from the template: no grid writes them, so no unit is ever stored in them. Declaring them would promise a
    format no build can produce.
    """
    from qda_quant.target import Q4K, Q6K

    target = Target.parse("llama_cpp")
    for name, zero_point in ((Q4K, True), (Q6K, False)):
        plan = target.plan(Format.parse(name), "dense")
        assert plan.kernel == "llama.cpp"
        assert plan.zero_point is zero_point
        assert "static" in plan.act_order, "actorder static stores nothing, so the records are ordinary"

    for name in ("w4a16-sym-g128", "w3a16-sym-g64", "w8a16-sym-g32"):
        with pytest.raises(Refused, match="holds solved codes as"):
            target.plan(Format.parse(name), "dense")


def test_a_gguf_build_takes_its_own_target_and_refuses_one_a_config_names() -> None:
    """The GGUF output is read by llama.cpp, so it names its target; a config that also names one is refused.

    Silently ignoring the config's `target` would let a recipe say sm90 and get llama.cpp, and `qda plan`
    would print a target the build did not use.
    """
    from qda_quant.engine.build import _target_for
    from qda_quant.gguf.export import GGUF
    from qda_quant.output.exporters import CompressedTensors

    gguf = GGUF()
    assert _target_for(gguf, None, []).arch == "llama_cpp"
    assert _target_for(gguf, "llama_cpp", []).arch == "llama_cpp", (
        "naming its own target is not a contradiction"
    )
    with pytest.raises(Refused, match="remove `target` from the config"):
        _target_for(gguf, "sm90", [])

    other = CompressedTensors()
    assert _target_for(other, None, []).arch == "sm90", "an output without its own target keeps the default"
    assert _target_for(other, "sm90@0.27.1", []).vllm == "0.27.1"


def test_plan_and_build_resolve_the_same_target(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """One rule, both paths: a plan must never report a target the build would not use.

    `build` asks the exporter for its target; `plan` resolved the config's field directly, so a GGUF config
    planned as sm90 and built as llama_cpp -- the plan named kernels the file will never meet. This captures
    the target `BuildConfig.plan` actually hands the planner, so it protects that path and not a helper
    beside it: calling the rule directly would pass whether or not `plan` used it.
    """
    from qda_quant.config import BuildConfig
    from qda_quant.engine.build import _target_for
    from qda_quant.output.exporters import parse_exporter

    seen: list[Target] = []

    def capture(_source, _recipe, _fmt, target, *a, **k):  # noqa: ANN001, ANN202
        seen.append(target)
        raise RuntimeError("stop: the target is all this test needs")

    # `plan` does `from .engine.planning import compile_plan` inside the function, so the name is looked up
    # on the module at call time: patching the module attribute is what that import reads.
    import qda_quant.engine.planning as planning

    monkeypatch.setattr(planning, "compile_plan", capture)

    common = {
        "model": str(tmp_path),
        "format": "w4a16-asym-g32",
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 1}},
    }
    for exporter, want in (({"gguf": {}}, "llama_cpp"), ({"compressed_tensors": {}}, "sm90")):
        cfg = BuildConfig(**common, exporter=exporter)
        seen.clear()
        with pytest.raises(RuntimeError, match="stop: the target"):
            cfg.plan()
        assert seen, "compile_plan was not reached; this test is not exercising plan()"
        assert seen and seen[0].arch == want, f"{exporter}: plan passes {seen[0].arch}, not {want}"
        assert _target_for(parse_exporter(cfg.exporter), None, []).arch == want, "build disagrees with plan"

    named = BuildConfig(**common, exporter={"gguf": {}}, target="sm90")
    with pytest.raises(Refused, match="remove `target` from the config"):
        named.plan()


@pytest.mark.parametrize(
    ("name", "arch"),
    [
        ("l40s", "sm89"),
        ("l4", "sm89"),
        ("4090", "sm89"),
        ("ada", "sm89"),
        ("rtx6000", "sm120"),
        ("rtx_pro_6000", "sm120"),
        ("sm89", "sm89"),
        ("sm120", "sm120"),
    ],
)
def test_ada_and_rtx_pro_6000_plan_their_rows(name, arch):
    """Both cards serve W4A16 through Marlin and W3A16 through Humming; W4A8 is refused with the alternative."""
    t = Target.parse(name)
    assert t.arch == arch
    p = t.plan(PRESETS["w4a16"], "dense")
    assert p.kernel == "MarlinLinearKernel" and p.serve_args == {"linear_backend": "marlin"}
    assert p.zero_point is True
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in p.notes), "W4A16 ran on a card"
    p3 = t.plan(PRESETS["w3a16"], "dense")
    assert p3.kernel == "HummingLinearKernel" and not p3.zero_point
    assert any("NVRTC" in n for n in p3.notes)
    moe = t.plan(PRESETS["w4a16"], "moe")
    assert moe.kernel == "MarlinExperts" and moe.serve_args == {"moe_backend": "marlin"}
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in moe.notes), "the W4A16 MoE row ran on these cards"
    moe3 = t.plan(PRESETS["w3a16"], "moe")
    assert moe3.kernel == "HummingIndexedExperts"
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in moe3.notes), "the W3A16 MoE row ran on these cards"
    with pytest.raises(Refused, match="compute capability 90"):
        t.plan(PRESETS["w4a8"], "dense")
    with pytest.raises(Refused, match="use w4a16"):
        t.plan(PRESETS["w4a8"], "moe")
    with pytest.raises(Refused, match="groups 64 and 128"):
        t.plan(Format.parse("w3a16-sym-g32"), "dense")
    p8 = t.plan(Format.parse("w8a16-sym-g64"), "dense")  # the published W4A16s' 8-bit attention groups
    assert p8.kernel == "MarlinLinearKernel" and p8.serve_args == {"linear_backend": "marlin"}
    for fmt, kind in (("w8a16-sym-g64", "moe"), ("w8a16-asym-g64", "dense")):
        with pytest.raises(Refused, match="dense and symmetric"):
            t.plan(Format.parse(fmt), kind)


def test_w8a16_marlin_is_unverified_everywhere_but_where_it_ran():
    """The observed key carries the weight width: Marlin was seen at 4 bits on an L40S, so an L40S's (or a
    B200's) dense symmetric W8A16 keeps the unverified note; an RTX PRO 6000 ran it (Qwen3.5-4B W4A16's 8-bit
    groups, KL 0.010664), so sm120 does not."""
    w8, w4 = Format.parse("w8a16-sym-g64"), PRESETS["w4a16"]
    for name in ("l40s", "sm100"):
        t = Target.parse(name)
        assert UNVERIFIED_ON_HARDWARE in t.plan(w8, "dense").notes, name
        assert UNVERIFIED_ON_HARDWARE not in t.plan(w4, "dense").notes, name
        assert UNVERIFIED_ON_HARDWARE not in t.plan(Format.parse("w4a16-sym-g128"), "dense").notes, name
    t = Target.parse("rtx6000")
    assert t.plan(w8, "dense").kernel == "MarlinLinearKernel"
    assert UNVERIFIED_ON_HARDWARE not in t.plan(w8, "dense").notes
    assert UNVERIFIED_ON_HARDWARE not in t.plan(w4, "dense").notes


def test_a_gb300_ran_dense_marlin_at_4_bits_only():
    """A GB300 ran W4A16 through Marlin (Qwen3-8B, asymmetric and symmetric); no 8-bit Marlin layer has run on
    one, so a W8A16 dense plan on sm103 keeps the unverified note and W4A16 does not."""
    t = Target.parse("sm103")
    w8 = t.plan(Format.parse("w8a16-sym-g64"), "dense")
    assert w8.kernel == "MarlinLinearKernel" and UNVERIFIED_ON_HARDWARE in w8.notes
    for name in ("w4a16", "w4a16-sym-g128", "w4a16-sym-g64"):
        p = t.plan(Format.parse(name) if "-" in name else PRESETS[name], "dense")
        assert p.kernel == "MarlinLinearKernel" and UNVERIFIED_ON_HARDWARE not in p.notes, name


@pytest.mark.parametrize("arch", ["sm89", "sm100", "sm103", "sm120"])
def test_marlin_experts_were_observed_at_4_bits_only(arch):
    """Every MarlinExperts run on these cards was W4A16 (asymmetric). The tables refuse an 8-bit expert bank
    today, so the observed key is asked directly through the table with a row that plans MarlinExperts at any
    width: int4 asymmetric counts as observed, int8 asymmetric does not."""
    table = TARGETS[arch]
    with pytest.raises(Refused, match="dense and symmetric"):
        Target.parse(arch).plan(Format.parse("w8a16-asym-g64"), "moe")
    marlin = KernelPlan("MarlinExperts", zero_point=True)
    probe = replace(table, rows=lambda _t, _f, _k: marlin)
    target = Target.parse(arch)
    assert UNVERIFIED_ON_HARDWARE not in probe(target, Format.parse("w4a16-asym-g64"), "moe").notes
    assert UNVERIFIED_ON_HARDWARE in probe(target, Format.parse("w8a16-asym-g64"), "moe").notes


def test_sm90_marlin_was_observed_at_4_bits_only():
    """The H100/H200 Marlin runs (Qwen3-14B and Qwen3-30B-A3B W4A16, tiny asym MoE) were int4: ``:marlin``
    keeps the unverified note at W8A16, dense and experts, and drops it at W4A16."""
    t = Target.parse("sm90:marlin")
    for kind, kernel in (("dense", "MarlinLinearKernel"), ("moe", "MarlinExperts")):
        for sym in ("sym", "asym"):
            w8 = t.plan(Format.parse(f"w8a16-{sym}-g128"), kind)
            w4 = t.plan(Format.parse(f"w4a16-{sym}-g128"), kind)
            assert w8.kernel == w4.kernel == kernel
            assert any("unverified" in n for n in w8.notes), (kind, sym)
            assert not any("unverified" in n for n in w4.notes), (kind, sym)


def test_ada_bindings_are_marlin_and_refuse_a_group_straddle():
    t = Target.parse("l40s")
    assert t.binds(PRESETS["w4a16"], (4096, 4096), {}) == "MarlinLinearKernel"
    with pytest.raises(Refused, match="groups of 128"):
        t.binds(PRESETS["w4a16"], (4096, 300), {})
    assert t.binds_moe(PRESETS["w4a16"], (2048, 1408), {}) == "MarlinExperts"


@pytest.mark.parametrize("name", ["b200", "b100", "gb200", "sm100"])
def test_blackwell_datacenter_plans_its_rows(name):
    """B200/B100/GB200 serve W4A16 through Marlin and W3A16 through Humming (both measured on a B200); W4A8
    is refused naming W4A16."""
    t = Target.parse(name)
    assert t.arch == "sm100"
    p = t.plan(PRESETS["w4a16"], "dense")
    assert p.kernel == "MarlinLinearKernel" and p.serve_args == {"linear_backend": "marlin"}
    assert p.zero_point is True
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in p.notes), "W4A16 ran on a B200"
    p3 = t.plan(PRESETS["w3a16"], "dense")
    assert p3.kernel == "HummingLinearKernel" and not p3.zero_point
    moe = t.plan(PRESETS["w4a16"], "moe")
    assert moe.kernel == "MarlinExperts" and moe.serve_args == {"moe_backend": "marlin"}
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in moe.notes), "the W4A16 MoE row ran on a B200"
    moe3 = t.plan(PRESETS["w3a16"], "moe")
    assert moe3.kernel == "HummingIndexedExperts"
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in moe3.notes), "the W3A16 MoE row ran on a B200"
    with pytest.raises(Refused, match="compute capability 90"):
        t.plan(PRESETS["w4a8"], "dense")
    with pytest.raises(Refused, match="use w4a16"):
        t.plan(PRESETS["w4a8"], "moe")


def test_blackwell_datacenter_bindings_are_marlin_and_refuse_a_group_straddle():
    t = Target.parse("b200")
    assert t.binds(PRESETS["w4a16"], (4096, 4096), {}) == "MarlinLinearKernel"
    with pytest.raises(Refused, match="groups of 128"):
        t.binds(PRESETS["w4a16"], (4096, 300), {})
    assert t.binds_moe(PRESETS["w4a16"], (2048, 1408), {}) == "MarlinExperts"
    with pytest.raises(Refused, match="hidden"):
        t.binds_moe(PRESETS["w4a16"], (2000, 1408), {})


@pytest.mark.parametrize("arch", ["sm89", "sm100", "sm103", "sm120"])
@pytest.mark.parametrize(
    ("fmt", "args", "kernel"),
    [
        # vLLM 0.33.0 tries Marlin before Humming on every card but 9.0 (humming/priority.py): the row's kernel
        ("w4a16", {}, "MarlinExperts"),
        ("w3a16", {}, "HummingIndexedExperts"),
        # an explicit moe_backend is what vLLM binds (fused_moe/oracle/int_wna16.py)
        ("w4a16", {"moe_backend": "humming"}, "HummingIndexedExperts"),
        ("w4a16", {"moe_backend": "marlin"}, "MarlinExperts"),
    ],
)
def test_the_expert_rule_below_and_above_hopper_reads_moe_backend(arch, fmt, args, kernel):
    """An sm90 ``:humming`` build records ``moe_backend: humming``; served elsewhere, its experts bind Humming."""
    assert Target(arch).binds_moe(PRESETS[fmt], (2048, 768), {"args": args}) == kernel


@pytest.mark.parametrize("name", ["b300", "gb300", "sm103"])
def test_b300_and_gb300_plan_the_b200_rows_with_their_own_observations(name):
    """10.3 binds the same kernels as 10.0, so sm103 plans sm100's rows; only what a GB300 ran is observed."""
    t = Target.parse(name)
    assert t.arch == "sm103"
    assert Target.on_cards(None, [("cuda:0", (10, 3))]).arch == "sm103"
    for fmt in ("w4a16", "w3a16"):
        for kind in ("dense", "moe"):
            mine, b200 = t.plan(PRESETS[fmt], kind), Target.parse("b200").plan(PRESETS[fmt], kind)
            assert (mine.kernel, mine.zero_point, mine.serve_args) == (
                b200.kernel,
                b200.zero_point,
                b200.serve_args,
            )
            assert not any(UNVERIFIED_ON_HARDWARE in n for n in mine.notes), f"{fmt} {kind} ran on a GB300"
    assert any(UNVERIFIED_ON_HARDWARE in n for n in t.plan(Format.parse("w4a16-sym-g128"), "moe").notes)
    # a GB300 served NVFP4, dense and experts: both rows are its own observations
    for kind in ("dense", "moe"):
        assert not any(UNVERIFIED_ON_HARDWARE in n for n in t.plan(Format.parse("w4a4-nvfp4"), kind).notes)
    with pytest.raises(Refused, match="use w4a16"):
        t.plan(PRESETS["w4a8"], "dense")
    assert t.binds(PRESETS["w4a16"], (4096, 4096), {}) == "MarlinLinearKernel"
    assert t.binds_moe(PRESETS["w4a16"], (2048, 1408), {}) == "MarlinExperts"


@pytest.mark.parametrize("name", ["sm100", "b200", "b100", "gb200"])
def test_blackwell_serves_nvfp4_and_names_its_kernels(name):
    """The sm100 table's NVFP4 rows: FlashInfer's dense kernel and TRT-LLM's monolithic experts, both served
    on a B200."""
    t = Target.parse(name)
    assert t.arch == "sm100"
    nvfp4 = Format.parse("w4a4-nvfp4")
    dense, moe = t.plan(nvfp4, "dense"), t.plan(nvfp4, "moe")
    assert (dense.kernel, moe.kernel) == (
        "FlashInferCuteDslNvFp4LinearKernel",
        "TrtLlmNvFp4ExpertsMonolithic",
    )
    assert (dense.scale_dtype, dense.act_dtype, dense.zero_point) == ("nvfp4", "nvfp4", False)
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in dense.notes), "a B200 container served this row"
    assert not any(UNVERIFIED_ON_HARDWARE in n for n in moe.notes), "a B200 served the expert row"
    with pytest.raises(Refused, match="read from vLLM 0.33.0"):
        Target.parse(f"{name}@0.32.0").plan(nvfp4, "dense")
    assert t.binds(nvfp4, (4096, 4096), {}) == dense.kernel
    with pytest.raises(Refused, match="groups of 16"):
        t.binds(nvfp4, (4096, 4104), {})
    with pytest.raises(Refused, match="multiples of 16"):
        t.binds_moe(nvfp4, (2048, 520), {})


@pytest.mark.parametrize("name", ["sm120", "rtx6000", "rtx_pro_6000"])
def test_rtx_pro_6000_serves_nvfp4_through_flashinfer_cutlass(name):
    """12.0 binds FlashInfer's CUTLASS kernels for NVFP4 (CuTe-DSL and TRT-LLM admit 10.x only), both run on an
    RTX PRO 6000; experts need an intermediate in multiples of 64."""
    t = Target.parse(name)
    assert t.arch == "sm120"
    assert Target.on_cards(None, [("cuda:0", (12, 0))]).arch == "sm120"
    nvfp4 = Format.parse("w4a4-nvfp4")
    dense, moe = t.plan(nvfp4, "dense"), t.plan(nvfp4, "moe")
    assert (dense.kernel, moe.kernel) == ("FlashInferCutlassNvFp4LinearKernel", "FlashInferExperts")
    for plan in (dense, moe):
        assert (plan.scale_dtype, plan.act_dtype, plan.zero_point) == ("nvfp4", "nvfp4", False)
        assert not any(UNVERIFIED_ON_HARDWARE in n for n in plan.notes), plan.kernel
    with pytest.raises(Refused, match="read from vLLM 0.33.0"):
        Target.parse(f"{name}@0.32.0").plan(nvfp4, "dense")
    assert t.binds(nvfp4, (4096, 4096), {}) == dense.kernel
    with pytest.raises(Refused, match="groups of 16"):
        t.binds(nvfp4, (4096, 4104), {})
    assert t.binds_moe(nvfp4, (2048, 512), {}) == moe.kernel
    with pytest.raises(Refused, match=r"multiple of 64 on an RTX PRO 6000; use w4a16"):
        t.binds_moe(nvfp4, (2048, 528), {})
    with pytest.raises(Refused, match="multiples of 16"):
        t.binds_moe(nvfp4, (2048, 520), {})
    # the other formats keep Ada's rows
    assert t.plan(PRESETS["w4a16"], "dense") == Target.parse("l40s").plan(PRESETS["w4a16"], "dense")


def test_blackwell_names_the_modular_kernel_for_a_bank_that_routes_itself():
    """vLLM 0.33.0 routes a custom-router bank outside its fused experts kernel (Gemma 4's per-expert scale),
    so sm100's decomposed row is the modular kernel: same arithmetic, a different kernel name."""
    t = Target.parse("gb300")
    nvfp4 = Format.parse("w4a4-nvfp4")
    stock = t.plan(nvfp4, "moe")
    modular = t.plan(nvfp4, "moe", decomposed=True)
    assert stock.kernel == "TrtLlmNvFp4ExpertsMonolithic"
    assert modular.kernel == "TrtLlmNvFp4ExpertsModular"
    assert (modular.scale_dtype, modular.act_dtype, modular.zero_point) == ("nvfp4", "nvfp4", False)
    assert any("RoutingMethodType.Custom" in n for n in modular.notes)
    # a GB300 job bound it, so it is no longer marked unverified
    assert not any("unverified" in n for n in modular.notes)
    # the dense row and every table without a decomposed row are unchanged
    assert t.plan(nvfp4, "dense", decomposed=True) == t.plan(nvfp4, "dense")
    sm90 = Target(arch="sm90")
    assert sm90.plan(Format.parse("w4a8"), "moe", decomposed=True) == sm90.plan(Format.parse("w4a8"), "moe")
    # the binding rule reports the same kernel the plan names
    assert t.binds_moe(nvfp4, (2048, 512), {}) == stock.kernel
    assert t.binds_moe(nvfp4, (2048, 512), {}, decomposed=True) == modular.kernel
    with pytest.raises(Refused, match="multiples of 16"):
        t.binds_moe(nvfp4, (2048, 520), {}, decomposed=True)


def test_blackwell_serves_fp8_w8a8_dense_and_refuses_its_experts():
    """The sm100 FP8 row is CUTLASS for channel weights and token activations; experts name the unread methods."""
    t = Target.parse("gb300")
    fp8 = Format.parse("w8a8-fp8")
    plan = t.plan(fp8, "dense")
    assert plan.kernel == "CutlassFP8ScaledMMLinearKernel"
    assert (plan.scale_dtype, plan.act_dtype, plan.zero_point) == ("fp32", "fp8_e4m3", False)
    # a GB300 job logged "Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8" in vLLM 0.33.0
    assert not any("unverified" in n for n in plan.notes)
    with pytest.raises(Refused, match=r"no FP8 expert row.*Triton, DeepGEMM, FlashInfer and Marlin"):
        t.plan(fp8, "moe")
    with pytest.raises(Refused, match="dynamic per-token e4m3 activations"):
        t.plan(W8A8Fp8(group_size=32), "dense")
    # CUTLASS pads K and N to 16, so an FP8 weight needs no group of 16 in its input width
    assert t.binds(fp8, (4096, 4104), {}) == plan.kernel


def test_fp8_w8a8_dense_row_is_shared_by_every_card_with_fp8_tensor_cores():
    """8.9 and up plans the one `_FP8_W8A8` row; 8.6 and below is not W8A8 there, so it is refused."""
    fmt = Format.parse("w8a8-fp8")
    for name in ("sm89", "sm90", "sm100", "sm103", "sm120"):
        t = Target.parse(name)
        plan = t.plan(fmt, "dense")
        assert plan.kernel == "CutlassFP8ScaledMMLinearKernel"
        assert (plan.scale_dtype, plan.act_dtype, plan.zero_point) == ("fp32", "fp8_e4m3", False)
        assert any("unverified" in n for n in plan.notes) == (name not in ("sm103", "sm120")), name
        assert t.binds(fmt, (4096, 4104), {}) == plan.kernel
        with pytest.raises(Refused, match="no FP8 expert row"):
            t.plan(fmt, "moe")
    with pytest.raises(Refused, match="W8A16.*use w4a16"):
        Target.parse("a100").plan(fmt, "dense")


def test_nvfp4_has_no_row_before_blackwell():
    for name in ("sm90", "sm80"):
        with pytest.raises(Refused):
            Target.parse(name).plan(Format.parse("w4a4-nvfp4"), "dense")
