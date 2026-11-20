import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.arch import describe

from ..tiny import model


@pytest.mark.parametrize(
    "kind",
    ["qwen3", "qwen3-mha", "llama", "qwen3_moe", "qwen3_5", "deepseek_v2", "nemotron_h", "gpt_oss"],
)
def test_rotation_keeps_the_float_function(kind):
    m = model(kind, dtype=torch.float64)
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    before = {n: p.clone() for n, p in m.named_parameters()}
    graph = describe(m, verify=False)
    REGISTRY["rotate"]().apply_model(m, graph, None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()
    changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert {f"{graph.embed}.weight", f"{graph.lm_head}.weight"} <= set(changed)
    assert any("down_proj" in n for n in changed)


def test_rotation_keeps_the_float_function_when_ignore_keeps_a_writer_float():
    """`ignore` decides quantisation, not the basis: an ignored ``o_proj`` is rotated and stored as it is.

    fp32, not bf16 (ORDER-1745/1748): bf16's ~1.3e-3 floor would let a writer that was never rotated pass.
    """
    m = model("qwen3", dtype=torch.float32)
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    keep = frozenset({r"re:^model\.layers\.\d+\.self_attn\.o_proj"})
    graph = describe(m, verify=False, keep=keep)
    ignored = sorted(graph.extra["ignored"])
    assert ignored and all(n.endswith(".self_attn.o_proj") for n in ignored)
    before = {n: p.clone() for n, p in m.named_parameters()}
    REGISTRY["rotate"]().apply_model(m, graph, None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-4 * ref.abs().max()
    for n in ignored:  # rotated, in its own dtype, and out of every group (so never quantised)
        assert not torch.equal(m.get_parameter(f"{n}.weight"), before[f"{n}.weight"])
        assert m.get_parameter(f"{n}.weight").dtype == torch.float32


def test_rotation_keeps_the_float_function_when_ignore_keeps_a_bank_float():
    """A fused bank `ignore` keeps float writes the stream too, so it is rotated like any other writer."""
    m = model("qwen3_moe", dtype=torch.float32)
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    keep = frozenset({r"re:^model\.layers\.0\.mlp\.experts\."})
    graph = describe(m, verify=False, keep=keep)
    assert graph.extra["float_banks"] == ("model.layers.0.mlp.experts",)
    before = m.get_parameter("model.layers.0.mlp.experts.down_proj").clone()
    REGISTRY["rotate"]().apply_model(m, graph, None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-4 * ref.abs().max()
    rotated = m.get_parameter("model.layers.0.mlp.experts.down_proj")
    assert not torch.equal(rotated, before) and rotated.dtype == before.dtype


def test_value_heads_check_runs_grouped_mm_experts_on_the_reference_loop():
    """R2 checks itself on a fp64 layer copy; a loaded MoE runs grouped-mm experts, which have no fp64 kernel."""
    m = model("qwen3_moe")
    for module in m.modules():
        cfg = getattr(module, "config", None)
        if hasattr(cfg, "_experts_implementation"):
            cfg._experts_implementation = "grouped_mm"
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    v = "model.layers.0.self_attn.v_proj.weight"
    before = m.get_parameter(v).clone()
    REGISTRY["rotate"](residual=False).apply_model(m, describe(m, verify=False), None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert not torch.equal(m.get_parameter(v), before)
    assert (out - ref).abs().max() < 1e-4 * ref.abs().max()


def test_rotation_spreads_outliers():
    m = model("qwen3", dtype=torch.float64)
    with torch.no_grad():
        m.model.embed_tokens.weight[:, 3] *= 50  # one outlier channel in the residual stream
    kurt = lambda w: (w.pow(4).mean() / w.pow(2).mean() ** 2).item()  # noqa: E731
    before = kurt(m.model.embed_tokens.weight)
    REGISTRY["rotate"]().apply_model(m, describe(m, verify=False), None)
    assert kurt(m.model.embed_tokens.weight) < before / 5


def test_rotate_refuses_post_norm_blocks(tmp_path):
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.recipe import Recipe
    from qda_quant.step import Refused
    from qda_quant.target import Target

    model("olmo2").save_pretrained(tmp_path / "m")
    calib = {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 4, "seqlen": 16}}
    recipe = Recipe.from_dict({"steps": [{"rotate": {}}, {"gptq": {}}], "calib": calib})
    with pytest.raises(
        Refused, match=r"post-norm blocks \(model\.layers\.0\.post_attention_layernorm.*gptq alone"
    ):
        compile_plan(str(tmp_path / "m"), recipe, Format.parse("w4a16"), Target(), [torch.device("cpu")])


@pytest.mark.parametrize("kind", ["qwen3", "llama", "qwen3_moe", "qwen3_5"])
def test_pre_norm_layouts_pass_the_post_norm_check(kind):
    REGISTRY["rotate"]().check(describe(model(kind, meta=True), verify=False), None, None)


def _tiny_vlm(dtype=torch.float64):
    """A Qwen3.5 conditional (text + vision tower + merger), the release family's shape at tiny sizes."""
    import transformers

    text = transformers.Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=128,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    cfg = transformers.Qwen3_5Config(
        text_config=text,
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            out_hidden_size=128, num_position_embeddings=16,
        ),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Qwen3_5ForConditionalGeneration._from_config(cfg).eval().to(dtype)


def _image_prompt():
    t, h, w = 1, 4, 4
    n_img = t * h * w // 4  # spatial_merge_size 2
    pix = torch.randn(t * h * w, 3 * 2 * 16**2, generator=torch.Generator().manual_seed(1))
    ids = torch.cat(
        [torch.tensor([[7, 8, 122]]), torch.full((1, n_img), 120), torch.tensor([[123, 9, 10, 11]])], 1
    )
    return ids, pix, torch.tensor([[t, h, w]]), (ids == 120).int()


def test_image_embeddings_land_in_the_rotated_space():
    m = _tiny_vlm()
    ids, pix, grid, types = _image_prompt()
    kw = dict(pixel_values=pix.double(), image_grid_thw=grid, mm_token_type_ids=types)
    merger = m.get_parameter("model.visual.merger.linear_fc2.weight").clone()
    with torch.no_grad():
        ref = m(input_ids=ids, **kw).logits
        text = m(input_ids=ids).logits
        REGISTRY["rotate"]().apply_model(m, describe(m, verify=False), None)
        got = m(input_ids=ids, **kw).logits
    assert (ref - text).abs().max() > 1e-3  # the image reaches the logits
    assert (got - ref).abs().max() < 1e-5 * ref.abs().max()
    assert not torch.equal(m.get_parameter("model.visual.merger.linear_fc2.weight"), merger)


def test_a_declared_writer_a_later_step_transforms_refuses():
    from qda_quant.step import Refused

    m = _tiny_vlm()
    m.model.visual.merger.linear_fc2 = torch.nn.Sequential(m.model.visual.merger.linear_fc2, torch.nn.GELU())
    with pytest.raises(
        Refused,
        match=r"declares model\.visual\.merger\.linear_fc2 as a stream writer, but it is a Sequential",
    ):
        describe(m, verify=False)


def test_rotate_refuses_an_undeclared_module_outside_the_text_model():
    from qda_quant.step import Refused

    m = model("qwen3", dtype=torch.float64)
    m.mtp = torch.nn.Linear(m.config.hidden_size, m.config.hidden_size, dtype=torch.float64)
    with pytest.raises(
        Refused, match=r"mtp is outside the text model and no architecture adapter declares whether"
    ):
        REGISTRY["rotate"]().check(describe(m, verify=False), None, None)


def test_rotate_refuses_a_source_block_a_class_loads_without_a_checkpoint():
    nn = torch.nn
    from qda_quant.step import Refused

    m = _tiny_vlm()
    hidden = m.config.text_config.hidden_size
    m.mtp = nn.Module()
    m.mtp.fc = nn.Linear(2 * hidden, hidden, dtype=torch.float64)
    graph = describe(m, verify=False)
    assert "mtp" in graph.extra["stream"]["blocks"]
    with pytest.raises(Refused, match=r"mtp is a declared source block, but the model loads it.*gptq alone"):
        REGISTRY["rotate"]().apply_model(m, graph, None)


def test_the_writer_writes_the_rotated_merger(tmp_path):
    from types import SimpleNamespace

    from qda_quant.format import Format
    from qda_quant.output.export import Writer
    from qda_quant.target import Target
    from safetensors import safe_open

    m = _tiny_vlm(dtype=torch.float32)
    m.save_pretrained(tmp_path / "m")
    run = SimpleNamespace(tensors={}, info={})
    REGISTRY["rotate"]().apply_model(m, describe(m, verify=False), run)
    w = Writer(tmp_path / "out", tmp_path / "m", describe(m, verify=False), Format.parse("w4a16"), Target())
    w.add_source_tensors(run.tensors)
    out = w.finish(m.config, None, {"ok": True})
    key = "model.visual.merger.linear_fc2.weight"
    import json

    where = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(out / where[key], "pt") as h:
        written = h.get_tensor(key)
    assert torch.equal(written, m.get_parameter(key))
    with safe_open(tmp_path / "m" / "model.safetensors", "pt") as h:
        assert not torch.equal(h.get_tensor("model.visual.merger.linear_fc2.weight"), written)


def _planner(tmp_path, m, steps):
    """Save `m` with a tokenizer and calibration ids; returns a function that plans `steps` then gptq on it."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.recipe import Recipe
    from qda_quant.target import Target

    from ..engine_util import save_tiny_tokenizer, write_ids

    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    rows = [
        torch.randint(0, 128, (20,), generator=torch.Generator().manual_seed(0)).tolist() for _ in range(5)
    ]
    write_ids(tmp_path / "c.jsonl", rows, sha)
    recipe = Recipe.from_dict(
        {
            "steps": [*steps, {"gptq": {}}],
            "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )
    return lambda: compile_plan(
        str(tmp_path / "m"), recipe, Format.parse("w4a16"), Target(), [torch.device("cpu")]
    )


def test_rotate_refuses_a_source_with_weights_its_class_does_not_load(tmp_path):
    from qda_quant.step import Refused
    from safetensors.torch import save_file

    plan = _planner(tmp_path, model("qwen3"), [{"rotate": {}}])
    plan()  # the same checkpoint without a dropped head plans
    save_file({"mtp.fc.weight": torch.zeros(8, 8)}, tmp_path / "m" / "mtp.safetensors")
    with pytest.raises(Refused, match=r"weights under mtp that its model class does not load.*gptq alone"):
        plan()


def test_a_residual_permute_refuses_a_source_head_no_adapter_declares(tmp_path):
    """Permute does not check the source itself; without the shared guard the head reads the old order."""
    from qda_quant.step import Refused
    from safetensors.torch import save_file

    plan = _planner(tmp_path, model("qwen3"), [{"permute": {"intermediate": False}}])
    plan()
    save_file({"mtp.fc.weight": torch.zeros(8, 8)}, tmp_path / "m" / "mtp.safetensors")
    with pytest.raises(Refused, match=r"permute: it changes the residual basis, and the source has mtp"):
        plan()


def test_a_second_residual_rotate_refuses_over_source_transforms(tmp_path):
    """Each R1 rewrites the source merger from the checkpoint, so a second would drop the first's basis."""
    from qda_quant.step import Refused

    twice = [{"rotate": {"value_heads": False}}, {"rotate": {"seed": 1}}]
    _planner(tmp_path / "text", model("qwen3"), twice)()  # nothing outside the text model: they compose
    plan = _planner(tmp_path / "vlm", _tiny_vlm(dtype=torch.float32), twice)
    with pytest.raises(Refused, match=r"rotate: it changes the residual basis a second time"):
        plan()


def test_rotate_builds_a_text_checkpoint_of_a_family_that_declares_a_merger(tmp_path):
    """Qwen3.5's declaration names a vision merger the source may not ship; a text checkpoint has none."""
    from qda_quant import build
    from qda_quant.recipe import Recipe

    from ..engine_util import save_tiny_tokenizer, write_ids

    model("qwen3_5").save_pretrained(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[7, 8, 9, 10] * 5] * 5, save_tiny_tokenizer(tmp_path / "m"))
    calib = {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 4, "seqlen": 16}}
    recipe = Recipe.from_dict({"steps": [{"rotate": {}}, {"rtn": {}}], "calib": calib})
    build(str(tmp_path / "m"), recipe, format="w4a16", devices=["cpu"], out_dir=tmp_path / "out", log=print)
    assert (tmp_path / "out" / "build.json").exists()


@pytest.mark.parametrize("k_eq_v", [False, True])
def test_value_heads_follow_kv_shared_layers_across_head_widths(k_eq_v):
    """R2 checks one layer per class, which cannot see a layer that reads another layer's values.

    Gemma 4 has two head widths and KV-shared layers that project only queries. Without k_eq_v every value
    producer is rotated, so the shared layers' o_proj must be too; with it, the full-attention values come
    from k_proj, which R2 never rotates, so that width keeps no R2 while the sliding width still does.
    """
    from transformers import AutoModelForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
        vocab_size=128,
        max_position_embeddings=64,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=3,
        attention_k_eq_v=k_eq_v,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(config).eval().double()
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    o = {i: m.model.layers[i].self_attn.o_proj.weight.clone() for i in (4, 5)}  # shared: sliding, full
    REGISTRY["rotate"](residual=False).apply_model(m, describe(m, verify=False), None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()
    assert not torch.equal(m.model.layers[4].self_attn.o_proj.weight, o[4])
    assert torch.equal(m.model.layers[5].self_attn.o_proj.weight, o[5]) == k_eq_v


@pytest.mark.parametrize("kept", ["model.layers.1.self_attn.o_proj", "model.layers.4.self_attn.o_proj"])
def test_value_heads_rotate_an_ignored_o_proj_of_a_kv_shared_model(kept):
    """``ignore`` decides quantization, not the basis: R2 must rotate an o_proj the config keeps float.

    ``ignore`` takes the linear out of its group. On a KV-shared model the query-only layers' o_proj follows the
    rotation of the values they read, so leaving a kept o_proj out (of a producer, layer 1, or of a reader,
    layer 4) breaks the function while every other check still passes.
    """
    from transformers import AutoModelForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
        vocab_size=128,
        max_position_embeddings=64,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=3,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    before = m.get_submodule(kept).weight.clone()
    graph = describe(m, verify=False, keep=frozenset({kept}))
    assert kept in graph.extra["ignored"]
    REGISTRY["rotate"](residual=False).apply_model(m, graph, None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()
    assert not torch.equal(m.get_submodule(kept).weight, before)  # rotated, though kept float
