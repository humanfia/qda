"""``format_overrides``: rules that store chosen layer units in another format than the body's.

The plan decides and refuses before a weight loads; the checkpoint declares one config group per format and
stock transformers (the CPU proxy for vLLM's loader) serves every module with the scheme its group names.
"""

from __future__ import annotations

import json

import pytest
import torch
import transformers
import yaml
from qda_quant import REGISTRY, build
from qda_quant.calib.sources import FileSource
from qda_quant.config import BuildConfig
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.target import Target
from safetensors import safe_open

from .engine_util import save_tiny_tokenizer, tiny_qwen3, tiny_qwen3_moe, write_ids
from .goldens import check_golden, refusal


@pytest.fixture
def tiny(tmp_path):
    m = tiny_qwen3(layers=2)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(4)], sha
    )
    return tmp_path


def recipe(tmp):
    return Recipe(
        steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=16)
    )


def plan(tmp, *rules, keep=None):
    return compile_plan(
        str(tmp / "m"), recipe(tmp), Format.parse("w4a16-sym-g32"), Target(), [torch.device("cpu")],
        overrides=rules, **({} if keep is None else {"keep": keep}),
    )  # fmt: skip


def test_a_rule_names_one_or_many_units_and_the_body_keeps_the_rest(tiny):
    """A pattern is matched as `ignore` matches it: exact, or ``re:`` and a regex from the start."""
    p = plan(
        tiny, ("re:^model\\.layers\\.0\\.", "w8a16-sym-g32"), ("model.layers.1.self_attn.o_proj", "w8a16-sym-g32")
    )  # fmt: skip
    assert set(p.formats) == {
        "model.layers.0.self_attn.q_proj",  # keyed by the first linear of the fused q/k/v group
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.o_proj",
    }
    assert set(p.plans) == {("w8a16-sym-g32", "dense")}


def test_config_records_rules_in_canonical_form():
    base = {"model": "m", "format": "w4a16", "steps": [{"rtn": {}}],
            "calib": {"file": {"path": "c.jsonl", "samples": 4, "seqlen": 8}}, "out": "o"}  # fmt: skip
    cfg = BuildConfig.from_dict(
        {
            **base,
            "format_overrides": [
                {"pattern": "model.layers.0.self_attn.o_proj", "format": "w8a16"},
                ["re:^model\\.layers\\.1\\.", "w4a8"],
            ],
        }  # fmt: skip
    )
    assert cfg.format_overrides == [
        {"pattern": "model.layers.0.self_attn.o_proj", "format": "w8a16-sym-g128"},
        {"pattern": "re:^model\\.layers\\.1\\.", "format": "w4a8-sym-g128"},
    ]
    with pytest.raises(Refused, match="not a regex"):
        BuildConfig.from_dict({**base, "format_overrides": [{"pattern": "re:(", "format": "w8a16"}]})


def test_build_declares_the_formats_and_the_loader_serves_them(tiny):
    """The checkpoint says which modules are 8-bit, and stock transformers reads it that way."""
    out = build(
        str(tiny / "m"), recipe(tiny),
        format="w4a16-sym-g32", out_dir=tiny / "out", devices=["cpu"], dtype=torch.float32,
        format_overrides=[("model.layers.1.self_attn.o_proj", "w8a16-sym-g32")], log=lambda s: None,
    )  # fmt: skip
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert qc["config_groups"]["group_0"]["weights"]["num_bits"] == 4
    assert qc["config_groups"]["group_1"]["targets"] == ["model.layers.1.self_attn.o_proj"]
    assert qc["config_groups"]["group_1"]["weights"]["num_bits"] == 8
    assert yaml.safe_load((out / "resolved.yaml").read_text())["format_overrides"] == [
        {"pattern": "model.layers.1.self_attn.o_proj", "format": "w8a16-sym-g32"}
    ]
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(
        out / idx["model.layers.1.self_attn.o_proj.weight_packed"], "pt"
    ) as h:  # 64 codes at 8 bits
        assert h.get_tensor("model.layers.1.self_attn.o_proj.weight_packed").shape[-1] == 16
    with safe_open(out / idx["model.layers.1.mlp.down_proj.weight_packed"], "pt") as h:  # 96 codes at 4 bits
        assert h.get_tensor("model.layers.1.mlp.down_proj.weight_packed").shape[-1] == 12
    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    assert back.model.layers[1].self_attn.o_proj.quantization_scheme.weights.num_bits == 8
    assert back.model.layers[1].mlp.down_proj.quantization_scheme.weights.num_bits == 4
    source = transformers.AutoModelForCausalLM.from_pretrained(tiny / "m", dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = back(input_ids=ids).logits.float()
        assert torch.isfinite(logits).all()
        assert (logits - source(input_ids=ids).logits.float()).abs().max() < 0.5


@pytest.mark.parametrize(
    ("stored", "probed"),
    [
        # a wrapped text model: vLLM 0.33 and SGLang 0.5.23 rename the root but leave re: entries unmapped
        ("model.language_model.layers.3.mlp.experts", "language_model.model.layers.3.mlp.experts"),
        ("model.language_model.layers.3.mlp.experts", "model.layers.3.mlp.experts"),
        ("model.language_model.layers.3.mlp.experts", "model.language_model.layers.3.mlp.experts"),
        ("model.layers.3.mlp.experts", "model.layers.3.mlp.experts"),
        ("model.layers.3.block_sparse_moe.experts", "model.layers.3.block_sparse_moe.experts"),  # Mixtral
        ("mtp.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"),  # a source-only MTP bank
    ],
)
def test_an_expert_bank_entry_matches_every_name_an_engine_serves_it_as(stored, probed):
    """An expert bank's ``re:`` config entry claims the expert names each engine probes, and nothing else.

    The engines check a fused MoE through ``<bank>.0.{gate,up,down}_proj`` under their own module names, and
    neither maps a ``re:`` entry: an entry written from the checkpoint's wrapped name alone served the
    experts in the body's scheme. It must not reach another layer, another root (the MTP head), or
    the bank module itself.
    """
    from qda_quant.output.export import _experts_pattern, _ignore_matches

    pattern = _experts_pattern(stored)
    for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w3", "w2"):
        assert _ignore_matches(f"{probed}.0.{proj}", pattern), (probed, pattern)
    other = probed.replace(".3.", ".4.") if ".3." in probed else probed.replace(".0.", ".1.", 1)
    assert not _ignore_matches(f"{other}.0.gate_proj", pattern)
    assert not _ignore_matches(probed, pattern)
    root_of_other = "mtp.layers.0" if not stored.startswith("mtp.") else "model.layers.0"
    assert not _ignore_matches(f"{root_of_other}.mlp.experts.0.gate_proj", pattern)


@pytest.mark.parametrize("kind", ["mixtral", "minimax_m2"])
@pytest.mark.parametrize("stacked", [False, True], ids=["per-expert", "stacked"])
def test_an_expert_override_reaches_every_expert_under_each_engines_config_name(kind, stacked, tmp_path):
    """The override group claims each expert under the name every stock engine looks its scheme up as.

    Both engines load these experts as ``block_sparse_moe.experts.{e}.w1``/``w3``/``w2``, so either source
    layout is stored that way. SGLang 0.5.23 looks the scheme up under that name; vLLM 0.33 does too for
    Mixtral, but for MiniMax-M2 under ``mlp`` (its MoE layer's prefix). A group built from the in-memory
    ``mlp.experts``, or from the stored name alone, leaves one engine serving the experts in the body's scheme.
    """
    from qda_quant.output.export import _ignore_matches

    from .tiny import SMALL

    config = {"mixtral": transformers.MixtralConfig, "minimax_m2": transformers.MiniMaxM2Config}[kind](
        **SMALL, num_local_experts=4, num_experts_per_tok=2, bos_token_id=1, eos_token_id=2
    )
    config._attn_implementation = config._experts_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    m.save_pretrained(tmp_path / "m", save_original_format=not stacked)
    write_ids(tmp_path / "c.jsonl", [[3, 4, 5, 6] * 5] * 4, save_tiny_tokenizer(tmp_path / "m"))
    out = build(
        str(tmp_path / "m"), recipe(tmp_path),
        format="w4a16-sym-g32", out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.bfloat16,
        format_overrides=[(r"re:^model\.layers\.\d+\.mlp\.experts$", "w8a16-sym-g32")], log=lambda s: None,
    )  # fmt: skip
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    targets = [t for group in groups.values() for t in group["targets"] if t != "Linear"]
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    stored = {
        k.removesuffix(".weight_packed") for k in index if ".experts." in k and k.endswith(".weight_packed")
    }
    names = {
        f"model.layers.{i}.block_sparse_moe.experts.{e}.{p}"
        for i in range(2)
        for e in range(4)
        for p in ("w1", "w3", "w2")
    }
    assert stored == names
    probed = {  # the block name each engine reads quantization_config under (vLLM's minimax_m2.py:288, SGLang :853)
        "vllm": "mlp" if kind == "minimax_m2" else "block_sparse_moe",
        "sglang": "block_sparse_moe",
    }
    for engine, block in probed.items():
        for name in (n.replace(".block_sparse_moe.", f".{block}.") for n in names):
            assert any(_ignore_matches(name, t) for t in targets), (engine, name, targets)


def test_a_kv_shared_layers_unused_kv_follow_a_q_proj_override(tmp_path):
    """An override on a Gemma 4 KV-shared layer's q_proj also claims that layer's packed, never-read k/v.

    SGLang 0.5.23 builds one fused ``qkv_proj`` there and needs one scheme for its three shards, and vLLM 0.33
    matches each name: k/v stored in the override format must be in the override's group, not the body's.
    """
    from compressed_tensors.utils.match import match_name
    from safetensors.torch import load_file, save_file

    config = transformers.Gemma4Config(
        text_config=dict(
            hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, global_head_dim=32, num_global_key_value_heads=2,
            vocab_size=128, hidden_size_per_layer_input=0, num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, position_embedding_size=64,
        ),
        audio_config=None,
    )  # fmt: skip
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.Gemma4ForConditionalGeneration._from_config(config).to(torch.bfloat16).eval()
    m.save_pretrained(tmp_path / "m")
    stored = load_file(tmp_path / "m" / "model.safetensors")
    for i in (2, 3):  # the vendor checkpoint ships them; transformers builds none
        for p in "kv":
            stored[f"model.language_model.layers.{i}.self_attn.{p}_proj.weight"] = torch.randn(64, 128).to(
                torch.bfloat16
            )
    save_file(stored, tmp_path / "m" / "model.safetensors", metadata={"format": "pt"})
    write_ids(tmp_path / "c.jsonl", [list(range(16))] * 4, save_tiny_tokenizer(tmp_path / "m"))
    layer = "model.language_model.layers.2.self_attn"
    out = build(
        str(tmp_path / "m"), recipe(tmp_path),
        format="w4a16-sym-g32", out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.bfloat16,
        format_overrides=[(f"{layer}.q_proj", "w8a16-sym-g32")], log=lambda s: None,
    )  # fmt: skip
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    tensors = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}

    def bits(name):
        hits = [
            g for g in groups.values() if any(t != "Linear" and match_name(name, t) for t in g["targets"])
        ]
        return (hits or [g for g in groups.values() if "Linear" in g["targets"]])[0]["weights"]["num_bits"]

    for p in "qkv":
        name = f"{layer}.{p}_proj"
        assert bits(name) == 8, name
        assert tensors[f"{name}.weight_packed"].shape[-1] == 128 * 8 // 32  # stored in the override's 8 bits
    override = next(g for g in groups.values() if g["weights"]["num_bits"] == 8)
    assert f"{layer}.qkv_proj" in override["targets"]  # SGLang's class match precedes its shard lookup
    assert f"{layer.removeprefix('model.')}.qkv_proj" in override["targets"]
    assert bits(f"{layer}.qkv_proj") == 8
    assert {bits(f"model.language_model.layers.3.self_attn.{p}_proj") for p in "qkv"} == {4}
    assert "model.language_model.layers.3.self_attn.qkv_proj" not in override["targets"]


def test_an_expert_bank_override_names_the_projections_the_loader_probes(tmp_path):
    """Stock vLLM's MoE method asks a bank for ``.experts.0.{gate,up,down}_proj``, not for the bank.

    A group whose target ends at ``.experts`` misses those probes, so the engine serves the body's scheme
    for experts the checkpoint stored in the override format. The bank module itself must not be named
    either: transformers refuses to initialize a module that keeps its experts stacked.
    """
    from compressed_tensors.utils.match import match_name

    m = tiny_qwen3_moe(layers=2)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[3, 4, 5, 6] * 5] * 4, sha)
    out = build(
        str(tmp_path / "m"), recipe(tmp_path),
        format="w4a16-sym-g32", out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.float32,
        format_overrides=[(r"re:^model\.layers\.\d+\.mlp\.experts$", "w8a16-sym-g32")], log=lambda s: None,
    )  # fmt: skip
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    targets = next(
        group["targets"]
        for group in qc["config_groups"].values()
        if any(target.startswith("re:") and "experts" in target for target in group["targets"])
    )
    for layer in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.0.{proj}"
            assert any(match_name(name, target) for target in targets), (name, targets)
    assert not any(match_name("model.layers.0.mlp.experts", target) for target in targets)

    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.isfinite(back(input_ids=ids).logits).all()


def test_an_expert_override_records_only_the_kernels_the_checkpoint_uses(tmp_path):
    """A W3 body with W4 experts leaves no bank using the body's Humming MoE kernel.

    W4A16 experts bind Marlin on ``:marlin``, so recording the unused body kernel breaks ``check_kernels``. The
    formats must bind different MoE classes: W4A16 and W3A16 on a Humming target (the default) collapse to the
    same class and cannot expose the extra entry.
    """
    from qda_kl.serve import check_kernels

    target = Target.parse("sm90@0.33.0:marlin")
    body_fmt = Format.parse("w3a16-sym-g64")
    expert_fmt = Format.parse("w4a16-sym-g64")
    torch.manual_seed(0)
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,  # group 64 divides every projection
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    cfg._attn_implementation = "sdpa"
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    with torch.no_grad():
        for layer in m.model.layers:
            layer.mlp.experts.gate_up_proj.normal_(0, 0.1)
            layer.mlp.experts.down_proj.normal_(0, 0.1)
            layer.mlp.gate.weight.normal_(0, 0.5)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[3, 4, 5, 6] * 5] * 4, sha)
    out = build(
        str(tmp_path / "m"), recipe(tmp_path),
        format=body_fmt, out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.float32,
        format_overrides=[(r"re:^model\.layers\.\d+\.mlp\.experts$", expert_fmt.name)], log=lambda s: None,
        target=target,
    )  # fmt: skip
    serve = json.loads((out / "build.json").read_text())["serve"]
    body_moe = target.plan(body_fmt, "moe").kernel
    expert_moe = target.plan(expert_fmt, "moe").kernel
    assert body_moe != expert_moe  # keep this fixture capable of exposing an unused body entry
    assert serve["kernels"]["moe"] == [expert_moe]
    assert body_moe not in serve["kernels"]["moe"]
    # the scoring path's rule: the recorded set equals the kernels the stored formats bind
    bound = {
        "dense": [target.plan(body_fmt, "dense").kernel],
        "moe": [target.binds_moe(expert_fmt, (cfg.hidden_size, cfg.moe_intermediate_size), serve)],
    }
    check_kernels(serve["kernels"], bound)


def test_a_qwen3_5_w8_mixer_override_names_the_fused_inputs_sglang_builds(tmp_path):
    """Dense W8 overrides on a wrapped Qwen3.5's GDN inputs and attention reach SGLang's fused modules.

    SGLang 0.5.23 builds ``linear_attn.in_proj_qkvz`` and ``self_attn.qkv_proj`` as single modules and looks up
    each one's scheme by that fused name: a target naming the fused module wins; otherwise the body's ``Linear``
    group matches before the shard names are tried. The override group must name both, or the W8 shards are
    loaded into a W4 parameter and SGLang refuses the checkpoint.
    """
    from compressed_tensors.utils.match import match_name
    from safetensors.torch import load_file

    text = dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128, max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
    )  # fmt: skip
    config = transformers.Qwen3_5Config(
        text_config=text,
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=128,
            num_position_embeddings=16,
        ),
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
    )  # fmt: skip
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.Qwen3_5ForConditionalGeneration._from_config(config).to(torch.bfloat16).eval()
    m.save_pretrained(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [list(range(3, 19))] * 4, save_tiny_tokenizer(tmp_path / "m"))
    root = "model.language_model.layers"
    out = build(
        str(tmp_path / "m"), recipe(tmp_path),
        format="w4a16-sym-g32", out_dir=tmp_path / "out", devices=["cpu"], dtype=torch.bfloat16,
        format_overrides=[
            # the issue's rules, by the names `qda plan` prints for the wrapped model
            (r"re:^model\.layers\.\d+\.self_attn\.", "w8a16-sym-g32"),
            (r"re:^model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", "w8a16-sym-g32"),
        ],
        log=lambda s: None,
    )  # fmt: skip
    groups = json.loads((out / "config.json").read_text())["quantization_config"]["config_groups"]
    tensors = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}
    override = next(g for g in groups.values() if g["weights"]["num_bits"] == 8)

    def bits(name):
        """The scheme SGLang 0.5.23 takes for ``name``: a named target first, else the ``Linear`` group."""
        hits = [
            g for g in groups.values() if any(t != "Linear" and match_name(name, t) for t in g["targets"])
        ]
        return (hits or [g for g in groups.values() if "Linear" in g["targets"]])[0]["weights"]["num_bits"]

    for shard in ("in_proj_qkv", "in_proj_z", "out_proj"):
        assert tensors[f"{root}.0.linear_attn.{shard}.weight_packed"].shape[-1] == 128 * 8 // 32
    for p in "qkvo":
        assert tensors[f"{root}.1.self_attn.{p}_proj.weight_packed"].shape[-1] == 128 * 8 // 32
    # SGLang's wrapped tree, and the text-only root a language-model-only load uses
    for prefix in (root, "model.layers"):
        assert f"{prefix}.0.linear_attn.in_proj_qkvz" in override["targets"]
        assert f"{prefix}.1.self_attn.qkv_proj" in override["targets"]
        assert bits(f"{prefix}.0.linear_attn.in_proj_qkvz") == 8
        assert bits(f"{prefix}.1.self_attn.qkv_proj") == 8
    assert bits(f"{root}.0.mlp.gate_up_proj") == 4  # the body keeps the MLP


# ---------------------------------------------------------------- refusal wording: golden/errors_format_overrides.txt
O_PROJ = "model.layers.1.self_attn.o_proj"

# name -> (rules, keep); each plan refuses before a weight loads
REFUSALS = {
    # vLLM fuses q/k/v (and gate/up) into one linear with one scheme: half a group cannot be stored apart
    "part of a fused group": ([("model.layers.1.self_attn.q_proj", "w8a16")], None),
    "names no unit": ([("model.layers.7.self_attn.o_proj", "w8a16")], None),
    "names a unit ignore keeps": ([(O_PROJ, "w8a16")], frozenset({O_PROJ})),
    "two formats for one unit": ([(O_PROJ, "w8a16"), (O_PROJ, "w4a8")], None),
    "format the target cannot serve": ([(O_PROJ, "w2a16-sym-g32")], None),
}


def test_an_sm100_build_refuses_a_w4a16_override(tiny):
    """The sm100 rows serve w4a4-nvfp4 only and a build runs one activation scheme: a w4a16 override in an
    NVFP4 body is refused when the build is planned, before a weight loads."""
    with pytest.raises(Refused, match="quantizes activations as bf16, the body's nvfp4"):
        compile_plan(
            str(tiny / "m"), recipe(tiny), Format.parse("w4a4-nvfp4"), Target("sm100"),
            [torch.device("cpu")], overrides=[("model.layers.1.self_attn.o_proj", "w4a16-sym-g16")],
        )  # fmt: skip


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusals_keep_their_wording(tiny, name):
    """Each override refusal fires at plan time, with the exact text in the golden file."""
    rules, keep = REFUSALS[name]
    text = refusal(lambda: plan(tiny, *rules, keep=keep)).replace(str(tiny), "<tmp>")
    check_golden("format_overrides", name, text, list(REFUSALS))


def test_a_group_entry_reaches_the_bank_under_a_substituted_root():
    """vLLM substitutes NemotronH's root (backbone -> model) and keeps re: entries as written.

    A config-group entry must therefore identify the bank by its path below the root: the same expert
    under ``backbone.`` and under vLLM's ``model.`` (or no root) matches, and nothing else does.
    """
    from qda_quant.output.export import _experts_pattern, _ignore_matches

    stored = "backbone.layers.1.mixer.experts"
    group = _experts_pattern(stored, anchored=False)
    for root in ("backbone", "model", ""):
        prefix = f"{root}." if root else ""
        for proj in ("down_proj", "up_proj", "gate_proj"):
            assert _ignore_matches(f"{prefix}layers.1.mixer.experts.0.{proj}.weight", group)
    for name in (
        "model.layers.2.mixer.experts.0.down_proj.weight",  # another layer
        "model.layers.10.mixer.experts.0.down_proj.weight",  # the ".1" prefix trap
        "model.layers.1.mixer.experts",  # the bank module itself
        "model.layers.1.mixer.down_proj.weight",  # a dense module of the same layer
    ):
        assert not _ignore_matches(name, group), name
    anchored = _experts_pattern(stored)  # an ignore entry of a source-only bank keeps its own root
    assert _ignore_matches("backbone.layers.1.mixer.experts.0.down_proj.weight", anchored)
    assert not _ignore_matches("model.layers.1.mixer.experts.0.down_proj.weight", anchored)
