"""GSQ's stochastic gradients, code boundaries, frozen groups, and saved checkpoints."""

import copy

import pytest
import torch
from pydantic import ValidationError
from qda_quant import REGISTRY, Recipe
from qda_quant.arch import describe
from qda_quant.engine import build
from qda_quant.engine.runner import LayerIO, call, capture
from qda_quant.solve.quant import Quantizer
from qda_quant.step import Refused
from qda_quant.steps._gumbel import GumbelRounding, Lion, cosine_lr
from transformers import AutoModelForCausalLM

from ..ctx import contexts
from ..engine_util import reads_dense_int3, save_tiny_tokenizer, write_ids
from ..tiny import model


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("frozen", [True, False])
def test_gumbel_replay_matches_direct_autograd(symmetric, frozen):
    codes = torch.tensor([[-4, -3, -2, -1, 0, 1, 2, 3]], dtype=torch.float32)
    zero = None if symmetric else torch.tensor([[4, 3]], dtype=torch.float32)
    if not symmetric:
        codes += 4
    scales = torch.tensor([[0.25, 0.5]])
    if frozen:
        scales[0, 0] = 0
    rounding = GumbelRounding(
        codes,
        scales,
        zero,
        4,
        -4 if symmetric else 0,
        3 if symmetric else 7,
        dtype=torch.float32,
        logits_dtype=torch.float32,
    )
    reference = copy.deepcopy(rounding)
    torch.manual_seed(23)
    actual = rounding(0.7, 1.5)
    actual.square().sum().backward()
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(23)
    expected = reference.reference(0.7, 1.5)
    expected.square().sum().backward()
    assert torch.equal(rng_after, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(
        rounding.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=2e-6, atol=2e-6)


def test_gumbel_boundaries_and_zero_scale_groups():
    codes = torch.tensor([[-4, 3, 0, 2, -4, 3, 1, 0]], dtype=torch.float32)
    rounding = GumbelRounding(codes, torch.tensor([[0.0, 0.25]]), None, 4, -4, 3)
    initial = rounding.hard_codes().clone()
    optimizer = Lion([{"params": list(rounding.parameters()), "lr": 0.01}])
    for _ in range(3):
        optimizer.zero_grad()
        rounding(1.0, 100.0).float().sum().backward()
        optimizer.step()
    final = rounding.hard_codes()
    assert torch.equal(initial[:, :4], codes[:, :4])
    assert torch.equal(final[:, :4], codes[:, :4])
    assert rounding.scales[0, 0] == 0
    assert final.min() >= -4 and final.max() <= 3
    assert torch.isfinite(rounding.logits.grad).all()


def test_cosine_schedule_endpoints():
    assert cosine_lr(1.0, 0, 8, 0, 0.1) == 1.0
    assert cosine_lr(1.0, 7, 8, 0, 0.1) == 0.1
    assert cosine_lr(1.0, 0, 8, 2, 0.1) == 0.1
    assert cosine_lr(1.0, 2, 8, 2, 0.1) == 1.0


@pytest.mark.parametrize("settings", [{"epochs": 0}, {"temperature": (0, 1)}, {"betas": (0.9, 1)}])
def test_invalid_gsq_settings(settings):
    with pytest.raises(ValidationError):
        REGISTRY["gsq"](**settings)


def test_gsq_requires_initial_codes():
    ctx = contexts(model("qwen3"), "w3a16")[0]
    ctx.stats[("layer_io", ctx.graph.module)] = [(None, (), {}, None)]
    with pytest.raises(Refused, match="put gptq or rtn before gsq"):
        REGISTRY["gsq"](validation=0).apply_layer(ctx)


class _FakeComm:
    """Two ranks' collectives without a group: records each call and answers for the smallest shard.

    `broadcast_qparams` copies rank 0's parameters; this stands in for rank 0, so it only records the call.
    """

    world = 2

    def __init__(self, smallest: int) -> None:
        self.smallest = smallest
        self.calls: list = []

    def broadcast_qparams(self, qparams, skip_experts: bool = False) -> None:
        self.calls.append(("broadcast_qparams", skip_experts))

    def min_across(self, value: int) -> int:
        self.calls.append(("min_across", value))
        return self.smallest

    def average_grads(self, optimizer, only: set[int] | None = None) -> None:
        self.calls.append(("average_grads", only))


def _three_batch_context(batches: int = 3):
    """A one-layer context whose `layer_io` holds `batches` calibration batches, with RTN's codes."""
    source = model("qwen3")
    graph = describe(source, verify=False)
    tokens = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(3))
    slots = capture(source, graph, [tokens] * batches, torch.device("cpu"))
    ctx = contexts(source, "w3a16")[0]
    ctx.stats[("layer_io", ctx.graph.module)] = LayerIO(slots, 0, torch.device("cpu"))
    REGISTRY["rtn"]().apply_layer(ctx)
    return ctx


def test_gsq_refuses_validation_that_empties_the_smallest_shard():
    """A multi-card fit decides its schedule on the smallest shard, after the first collective.

    `engine.dist.shard` can leave one rank a batch short, so a rank that refused on its own shard before
    `broadcast_qparams` would leave the others waiting in that collective: `min_across` decides for all ranks.
    """
    ctx = _three_batch_context()
    fake = _FakeComm(smallest=-1)  # another rank's shard holds no training batch
    ctx.comm = fake
    with pytest.raises(Refused, match="smallest calibration shard"):
        REGISTRY["gsq"](validation=4, batches=2).apply_layer(ctx)
    assert fake.calls[0] == ("broadcast_qparams", False)  # the refusal came after the first collective
    # `validation` is global, so THIS rank holds out 4 // 2 = 2 of its 3 batches and has one left: the
    # refusal has to come from the smallest shard, not from this rank's own count
    assert fake.calls[-1] == ("min_across", 1)


def test_gsq_refuses_validation_on_one_card_before_the_schedule():
    ctx = _three_batch_context()
    assert ctx.comm is None
    with pytest.raises(Refused, match="leaves no training batches out of 3"):
        REGISTRY["gsq"](validation=3, batches=1).apply_layer(ctx)


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe"])
def test_gsq_trains_dense_and_fused_experts(kind):
    source = model(kind)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    graph = describe(source, verify=False)
    tokens = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(3))
    slots = capture(source, graph, [tokens], torch.device("cpu"))
    ctx = contexts(source, "w3a16")[0]
    with torch.no_grad():
        slots[0].out = call(ctx.layer, slots[0].h, *slots[0].calls[0])
    ctx.stats[("layer_io", ctx.graph.module)] = LayerIO(slots, 0, torch.device("cpu"))
    REGISTRY["rtn"]().apply_layer(ctx)
    before = {name: values["scale"].clone() for name, values in ctx.qparams.items()}
    REGISTRY["gsq"](epochs=2, batches=1, validation=0).apply_layer(ctx)
    assert any(not torch.equal(values["scale"], before[name]) for name, values in ctx.qparams.items())
    assert all(torch.isfinite(parameter).all() for parameter in ctx.layer.parameters())
    for name, values in ctx.qparams.items():
        quantizer = Quantizer.of(ctx.fmt, ctx.plan_moe if "experts" in name else ctx.plan_dense, "absmax")
        expected = quantizer.dequant(values["codes"], values["scale"], values.get("zero_point"))
        torch.testing.assert_close(
            source.get_parameter(name + ".weight") if "experts" not in name else source.get_parameter(name),
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("format_name", ["w3a16", "w4a16"])
def test_gsq_build_loads_reproducibly(tmp_path, format_name):
    if format_name == "w3a16" and not reads_dense_int3():
        pytest.skip("transformers reads through compressed-tensors, which before 0.20 packs int3 another way")
    source = model("qwen3")
    source.save_pretrained(tmp_path / "model")
    fingerprint = save_tiny_tokenizer(tmp_path / "model")
    generator = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "calib.jsonl",
        [torch.randint(0, 128, (16,), generator=generator).tolist() for _ in range(3)],
        fingerprint,
    )
    recipe = Recipe.from_dict(
        {
            "steps": [{"rtn": {}}, {"gsq": {"epochs": 2, "batches": 1, "validation": 1}}],
            "calib": {"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 3, "seqlen": 16}},
        }
    )
    assert "layer_io" in recipe.steps[1].needs
    assert Recipe.from_yaml(recipe.to_yaml()).steps[1] == recipe.steps[1]
    states = []
    for repeat in range(2):
        output = build(
            str(tmp_path / "model"),
            recipe,
            format=format_name,
            out_dir=tmp_path / str(repeat),
            devices=["cpu"],
            dtype=torch.float32,
            log=lambda message: None,
        )
        loaded = AutoModelForCausalLM.from_pretrained(output, dtype=torch.float32)
        states.append(loaded.state_dict())
        with torch.no_grad():
            logits = loaded(input_ids=torch.tensor([[1, 2, 3]])).logits
        assert torch.isfinite(logits).all()
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), name


@pytest.mark.parametrize("zeroed", [True, False])
def test_gumbel_starting_state_is_the_chained_formula(zeroed):
    """xy-2140: the constructor builds its five-shift state in place; the state is the chained form's, bit for bit.

    The in-place form holds one float32 five-shift array where the chained one held five at once (17.4 GiB to
    9.1 GiB of live tensors for a 32-expert bank unit), so this pins that the arithmetic and the noise draw
    did not move with it.
    """
    group, qmin, qmax, std, strength = 4, -4, 3, 0.01, 6.0
    torch.manual_seed(5)
    codes = torch.randint(-4, 4, (2, 6, 8)).float()
    scales = torch.rand(2, 6, 2) * 0.02 + 1e-3
    zero = torch.zeros(2, 6, 2) if zeroed else None
    torch.manual_seed(11)
    got = GumbelRounding(
        codes, scales, zero, group, qmin, qmax, std, strength, torch.bfloat16, torch.bfloat16
    )

    flat = codes.reshape(-1, codes.shape[-1]).float()
    grouped = scales.reshape(flat.shape[0], -1)
    offset = torch.zeros_like(grouped) if zero is None else zero.reshape_as(grouped).float()
    columns = offset.repeat_interleave(group, -1)
    initial = (flat - columns).to(torch.bfloat16)
    shifts = torch.arange(-2, 3, dtype=torch.bfloat16)
    candidates = initial[None] + shifts[:, None, None] + columns
    valid = (candidates >= qmin) & (candidates <= qmax)
    frozen = (grouped == 0).repeat_interleave(group, -1)
    valid &= ~frozen[None] | (shifts == 0)[:, None, None]
    logits = -0.5 * (torch.zeros_like(flat)[None] - shifts[:, None, None]).square()
    logits -= (logits * valid).sum(0, keepdim=True) / valid.sum(0, keepdim=True)
    torch.manual_seed(11)
    expected = (std * (torch.randn_like(logits) + logits * strength)).to(torch.bfloat16)

    assert torch.equal(got.logits, expected)
    assert torch.equal(got.initial, initial)
    assert torch.equal(got.valid, valid)


def test_gsq_prices_one_units_workspace_and_holds_the_layers_units_together():
    """A joint solve is one unit's state: `engine.residency._peak` sums the layer's (`joint_units`)."""
    from qda_quant.steps.gsq import GSQ

    assert GSQ.joint_units is True
    assert GSQ().solver_workspace(1856, 2688) == 80 * 1856 * 2688


def _nemotron_w8(tmp_path, steps, *, overrides):
    """Build a tiny Nemotron-H the way the 30B W3A16 recipe does: w3 body, w8 mamba/attention projections."""
    import transformers
    from qda_quant.calib.sources import FileSource

    from ..tiny import config as tiny_config

    cfg = tiny_config("nemotron_h")
    cfg._attn_implementation = cfg._experts_implementation = "eager"
    torch.manual_seed(0)
    source = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    source.save_pretrained(tmp_path / "model")
    fingerprint = save_tiny_tokenizer(tmp_path / "model")
    write_ids(tmp_path / "calib.jsonl", [[3, 4, 5, 6] * 8] * 8, fingerprint)
    recipe = Recipe(
        steps=[REGISTRY[name](**cfg_) for name, cfg_ in steps],
        calib=FileSource(path=str(tmp_path / "calib.jsonl"), samples=8, seqlen=32),
    )
    return build(
        str(tmp_path / "model"), recipe,
        format="w3a16-sym-g64", out_dir=tmp_path / f"out{len(steps)}{bool(overrides)}",
        devices=["cpu"], dtype=torch.bfloat16, format_overrides=overrides, log=lambda message: None,
    )  # fmt: skip


def _scales(output, *, bits):
    """Every stored group scale, split by the width its config group gives it."""
    import json

    from safetensors.torch import load_file

    stored = {}
    for shard in sorted(output.glob("*.safetensors")):
        stored.update(load_file(shard))
    groups = json.loads((output / "config.json").read_text())["quantization_config"]["config_groups"]
    named = {
        target
        for group in groups.values()
        if group["weights"]["num_bits"] == bits
        for target in group["targets"]
        if target != "Linear"
    }
    out = {}
    for key, value in stored.items():
        name = key.removesuffix(".weight_scale")
        if key.endswith(".weight_scale") and ((name in named) == (bits == 8)):
            out[name] = value.float()
    return out


W8_PROJECTIONS = r"re:^(backbone|model)\.layers\.\d+\.(mixer|self_attn)\.(in_proj|out_proj|[qkvo]_proj)$"


def test_gsq_keeps_an_8_bit_groups_scale_inside_its_grid(tmp_path):
    """An 8-bit group's scale is ~36x smaller than the 3-bit scale `scales_lr` was tuned on.

    Lion moves a scale by exactly +-lr however large it is, so at 8 bits the untreated step is a fifth of the
    scale and walks it through zero: the codes stay GPTQ's and the group's weights come back sign-flipped
    (measured on the 30B export: norm ratio 4.0-4.7, cosine 0.14-0.63, scale p95/p50 1e25). A negative
    scale is not a worse fit, it is outside the absmax grid the codes were solved in.
    """
    gptq = _nemotron_w8(tmp_path, [("gptq", {})], overrides=[(W8_PROJECTIONS, "w8a16-sym-g64")])
    gsq = _nemotron_w8(
        tmp_path,
        [("gptq", {}), ("gsq", {"epochs": 2, "batches": 2, "validation": 1})],
        overrides=[(W8_PROJECTIONS, "w8a16-sym-g64")],
    )
    before, after = _scales(gptq, bits=8), _scales(gsq, bits=8)
    assert before and set(before) == set(after)
    for name, scale in after.items():
        assert (scale > 0).all(), f"{name}: {int((scale <= 0).sum())} of {scale.numel()} scales left the grid"
        cosine = torch.nn.functional.cosine_similarity(scale.flatten(), before[name].flatten(), 0)
        assert cosine > 0.99, f"{name}: 8-bit scales moved to cos {cosine:.4f} of the GPTQ scales"
    banks = _scales(gsq, bits=3)
    assert banks, "the 3-bit banks must still be present"
    for name, scale in banks.items():
        assert (scale > 0).all(), name


def test_gsq_leaves_upstreams_own_bit_width_alone(tmp_path):
    """`scales_lr` is upstream's 3-bit rate: at 3 bits the step must be exactly what it was."""
    from qda_quant.format import Format
    from qda_quant.steps.gsq import GSQ, UPSTREAM_LEVELS

    step = GSQ()
    assert UPSTREAM_LEVELS == 7
    assert step._scales_lr(Format.parse("w3a16-sym-g64")) == step.scales_lr
    assert step._scales_lr(Format.parse("w8a16-sym-g64")) == pytest.approx(step.scales_lr * 7 / 255)
    assert step._scales_lr(Format.parse("w4a16-sym-g64")) == pytest.approx(step.scales_lr * 7 / 15)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_the_step_count_does_not_depend_on_how_many_cards_fit_the_layer(world):
    """Upstream's `batches` is the GLOBAL batch (``src/trainer.py:87``), so steps/epoch is world-invariant.

    A card's shard is 1/W of the calibration batches; dividing the batch by W too keeps
    ``per_epoch = train_count // batch`` the same on one card and on eight, and with it the LR, temperature
    and logit-scale schedules, which are all indexed by ``step/(total-1)``. Before this, eight cards took an
    eighth of the steps of one and silently ran a different (shorter) schedule.
    """
    from qda_quant.steps.gsq import GSQ

    step = GSQ(batches=16, epochs=3, validation=32)
    rows = 512  # calibration batches in the pool
    shard = rows // world  # engine.dist.shard: contiguous blocks
    # `validation` is global as well: holding the WHOLE count out of each shard would leave the step count
    # world-dependent (30 steps on one card, 16 on eight) even with the batch divided.
    train_count = shard - step.validation // world
    batch = min(step.batches // world, train_count)
    assert batch >= 1
    assert train_count // batch == 30, (world, shard, train_count, batch)
    assert (train_count // batch) * step.epochs == 90
