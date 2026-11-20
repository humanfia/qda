import json
import re
from types import SimpleNamespace

import pytest
import torch
import transformers
import yaml
from qda_quant import REGISTRY, Target
from qda_quant.arch import describe
from qda_quant.arch.adapters import ADAPTERS, DEEPSEEK_V4, FUSED_EXPERTS, FusedRuntime
from qda_quant.arch.derive import _serving_names, served_name
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from qda_quant.intake.probe import _weight_map
from qda_quant.intake.source_names import SourceNames
from qda_quant.output.export import Writer
from qda_quant.step import Refused
from safetensors import safe_open

from ..ctx import contexts
from ..engine_util import reads_dense_int3, save_tiny_tokenizer, writes
from ..tiny import model


def _build(kind, fmt, tmp_path, keep=()):
    src = tmp_path / "src"
    m = model(kind, dtype=torch.bfloat16)
    m.save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    graph = describe(m, verify=False)
    w = Writer(tmp_path / "out", src, graph, fmt, Target())
    for ctx in contexts(m, fmt):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        for name in tuple(ctx.qparams):
            if name.rpartition(".")[2] in keep:
                del ctx.qparams[name]
        w.add_layer(ctx)
    out = w.finish(m.config, None, {"ok": True})
    return src, m, out


def test_multimodal_keeps_checkpoint_names_and_vision_tower(tmp_path):
    """Gemma 3 stores ``language_model.model.*``, which transformers loads as ``model.language_model.*``.

    The text layers are packed under the checkpoint's names, and the vision tower is copied byte for byte, under
    the vendor layout both engines build (``vision_tower.vision_model.*``, which transformers 5 drops).
    """
    from qda_quant.arch.adapters.gemma3_vl import SERVING_NAMES

    src, _, out = _build("gemma3_vlm", "w4a16", tmp_path)
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert "language_model.model.layers.1.self_attn.q_proj.weight_packed" in idx
    assert not any(k.startswith("model.") for k in idx)
    with safe_open(src / "model.safetensors", "pt") as a:
        vision = sorted(k for k in a.keys() if k.startswith("vision_tower."))  # noqa: SIM118 (a safe_open handle)
    served = {k: served_name(k, SERVING_NAMES) for k in vision}
    assert vision and sorted(served.values()) == sorted(
        k for k in idx if k.startswith("vision_tower.")
    )  # unpacked
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    assert {
        v.removesuffix(".weight") for v in served.values() if v.endswith("self_attn.q_proj.weight")
    } <= set(ignore)
    with (
        safe_open(src / "model.safetensors", "pt") as a,
        safe_open(out / idx[served[vision[0]]], "pt") as b,
    ):
        assert all(torch.equal(a.get_tensor(k), b.get_tensor(served[k])) for k in vision)


def test_multimodal_names_never_pair_a_vision_module_with_a_text_tensor(tmp_path):
    """In a packed Gemma 3 checkpoint only text layers have ``weight_packed``; a suffix guess would hand layer
    0's to the vision tower's ``layers.0.self_attn.q_proj``. The load-time renamings are the whole map."""

    _, m, out = _build("gemma3_vlm", "w4a16", tmp_path)
    names = SourceNames(out)
    vision = next(
        n for n, _ in m.named_modules() if "vision_tower" in n and n.endswith("layers.0.self_attn.q_proj")
    )
    text = "model.language_model.layers.0.self_attn.q_proj"
    assert names.tensor(f"{vision}.weight_packed") is None
    assert (
        names.tensor(f"{vision}.weight")
        == vision.replace("model.vision_tower.", "vision_tower.vision_model.") + ".weight"
    )
    assert (
        names.tensor(f"{text}.weight_packed")
        == "language_model.model.layers.0.self_attn.q_proj.weight_packed"
    )


def test_gemma4_float_vision_names_cover_fusions_and_unfused_linears(tmp_path):
    """SGLang drops model. and fuses clippable projections; vLLM keeps the checkpoint names."""
    config = transformers.Gemma4Config(
        text_config=dict(
            hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, global_head_dim=32, num_global_key_value_heads=2,
            vocab_size=128, hidden_size_per_layer_input=0, layer_types=["full_attention"],
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, position_embedding_size=64,
        ),
        audio_config=None,
    )  # fmt: skip
    config._attn_implementation = "eager"
    m = transformers.Gemma4ForConditionalGeneration._from_config(config).to(torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    graph = describe(m, verify=False)
    writer = Writer(tmp_path / "out", src, graph, "w4a16-sym-g128", Target())
    for ctx in contexts(m, "w4a16-sym-g128", tokens=8):
        REGISTRY["rtn"]().apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(m.config, None, {})
    ignore = set(json.loads((out / "config.json").read_text())["quantization_config"]["ignore"])
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith(
            ("model.vision_tower.", "model.embed_vision.")
        ):
            assert {name, name.removeprefix("model.")} <= ignore
    for i in range(2):
        layer = f"vision_tower.encoder.layers.{i}"
        assert f"{layer}.self_attn.qkv_proj" in ignore
        assert {f"{layer}.self_attn.{part}_proj" for part in "qkv"} <= ignore
        assert f"{layer}.mlp.gate_up_proj" in ignore
        assert {f"{layer}.mlp.{part}_proj" for part in ("gate", "up")} <= ignore
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert any("language_model" in name and name.endswith("weight_packed") for name in index)
    with safe_open(src / "model.safetensors", "pt") as source:
        for name in source.keys():  # noqa: SIM118 (a safe_open handle)
            if "vision_tower" in name or "embed_vision" in name:
                with safe_open(out / index[name], "pt") as exported:
                    assert torch.equal(source.get_tensor(name), exported.get_tensor(name))


def test_gemma4_float_audio_tower_is_ignored_under_sglangs_tree(tmp_path):
    """SGLang 0.5.23 rebuilds the audio tower under its own names; vLLM keeps the checkpoint's.

    An audio linear SGLang does not find in ``ignore`` is built quantized, and its GLU wrapper fails before
    any weight loads (``clippable_linear.py:210``). The prefixes below are written out from SGLang's
    ``gemma4_mm.py``/``gemma4_audio.py``; the check is its ``should_ignore_layer`` (dotted suffixes match, a
    fused ``qkv_proj`` is looked up whole, else by its shards). The tower's bytes stay the source's.
    """
    from qda_quant.output.export import _ignore_matches

    config = transformers.Gemma4Config(
        text_config=dict(
            hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, global_head_dim=32, num_global_key_value_heads=2,
            vocab_size=128, hidden_size_per_layer_input=0, layer_types=["full_attention"],
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, position_embedding_size=64,
        ),
        audio_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, output_proj_dims=64),
    )  # fmt: skip
    config._attn_implementation = "eager"
    m = transformers.Gemma4ForConditionalGeneration._from_config(config).to(torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    writer = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-sym-g128", Target())
    for ctx in contexts(m, "w4a16-sym-g128", tokens=8):
        REGISTRY["rtn"]().apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(m.config, None, {})
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    fused = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}

    def sglang_ignores(name):
        proj = name.rpartition(".")[2]
        if proj in fused and name not in ignore:
            return all(any(_ignore_matches(name.replace(proj, s), e) for e in ignore) for s in fused[proj])
        return any(_ignore_matches(name, e) for e in ignore)

    tower = "model.audio_tower"
    served = [f"{tower}.subsample_conv_projection.input_proj_linear", f"{tower}.output_proj"]
    for i in range(2):
        layer = f"{tower}.conformer.{i}"
        served += [
            f"{layer}.{block}.ffw_layer_{k}.linear"
            for block in ("ffw_layer_start", "ffw_layer_end")
            for k in (1, 2)
        ]
        served += [
            f"{layer}.attention.attn.qkv_proj", f"{layer}.attention.attn.relative_position_embedding.pos_proj",
            f"{layer}.attention.post.linear", f"{layer}.lconv1d.linear_start.linear", f"{layer}.lconv1d.linear_end.linear",
        ]  # fmt: skip
    assert [name for name in served if not sglang_ignores(name)] == []
    for name, module in m.named_modules():  # vLLM matches the checkpoint's own names exactly
        if isinstance(module, torch.nn.Linear) and name.startswith(
            ("model.audio_tower.", "model.embed_audio.")
        ):
            assert name in ignore
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(src / "model.safetensors", "pt") as source:
        for name in source.keys():  # noqa: SIM118 (a safe_open handle)
            if "audio_tower" in name or "embed_audio" in name:
                with safe_open(out / index[name], "pt") as exported:
                    assert torch.equal(source.get_tensor(name), exported.get_tensor(name))


def test_gemma4_kv_shared_kv_take_q_scheme_and_nothing_else_changes(tmp_path):
    """A KV-shared layer's unused checkpoint k/v are packed in q_proj's format; every other tensor is unchanged.

    The model class builds only q_proj there, so the k/v weights reach the export as untouched source tensors.
    Kept float beside a packed q_proj, SGLang 0.5.23's fused ``qkv_proj`` has two schemes and is refused.
    """
    from qda_quant.output.export import _ignore_matches
    from safetensors.torch import load_file

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
    m = transformers.Gemma4ForConditionalGeneration._from_config(config).to(torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    shared = [f"model.language_model.layers.{i}.self_attn.{p}_proj" for i in (2, 3) for p in "kv"]
    from safetensors.torch import save_file

    stored = load_file(src / "model.safetensors")
    generator = torch.Generator().manual_seed(0)
    for (
        name
    ) in shared:  # the vendor checkpoint ships them although no engine reads them; transformers drops them
        stored[f"{name}.weight"] = (torch.randn(2 * 32, 128, generator=generator) * 0.02).to(torch.bfloat16)
    save_file(stored, src / "model.safetensors", metadata={"format": "pt"})

    def export(out, declared):  # each from a fresh load: RTN writes the served weights back into the model
        m = transformers.Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16).eval()
        graph = describe(m, verify=False)
        assert set(graph.extra["pack_like"]) == set(shared)
        if not declared:
            graph.extra.pop("pack_like")
        writer = Writer(tmp_path / out, src, graph, "w4a16-sym-g128", Target())
        for ctx in contexts(m, "w4a16-sym-g128", tokens=8):
            REGISTRY["rtn"]().apply_layer(ctx)
            writer.add_layer(ctx)
        done = writer.finish(m.config, None, {})
        tensors = {k: v for f in done.glob("*.safetensors") for k, v in load_file(f).items()}
        return tensors, json.loads((done / "config.json").read_text())["quantization_config"]["ignore"]

    before, ignore_before = export("float", declared=False)
    after, ignore_after = export("packed", declared=True)
    changed = {
        k
        for k in before.keys() | after.keys()
        if k not in before or k not in after or not torch.equal(before[k], after[k])
    }
    assert {k.rsplit(".", 1)[0] for k in changed} == set(shared)
    assert all(f"{name}.weight_packed" in after and f"{name}.weight" not in after for name in shared)
    assert set(ignore_before) - set(ignore_after) == set(shared) and set(ignore_after) <= set(ignore_before)
    for i in (2, 3):  # SGLang's fused qkv_proj: every shard has the scheme q_proj has
        assert not any(
            _ignore_matches(f"model.language_model.layers.{i}.self_attn.{p}_proj", e)
            for p in "qkv"
            for e in ignore_after
        )


@pytest.mark.parametrize("family", ["mistral3", "gemma3"])
def test_float_vision_tower_is_named_as_both_engines_build_it(family, tmp_path):
    """The bf16 vision tower is stored and ignored under the names stock vLLM 0.33 and SGLang 0.5.23 build.

    SGLang builds Pixtral and SigLIP attention as one ``qkv_proj`` and an output ``proj`` (and Pixtral's MLP as one
    ``gate_up_proj``); its LLaVA class declares no fused-module mapping, so those names must be in ``ignore``. Gemma
    3's tower is saved by transformers 5 without its ``vision_model`` level, which both engines build. The names
    below are written out from the engines' model files. Every vision tensor keeps its source bytes.
    """
    from qda_quant.output.export import _ignore_matches
    from safetensors.torch import load_file

    text = dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=64,
    )  # fmt: skip
    if family == "mistral3":
        config = transformers.Mistral3Config(
            text_config=dict(model_type="mistral", vocab_size=128, **text),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                               head_dim=16, patch_size=16, image_size=64),
            image_token_index=120,
        )  # fmt: skip
        layer = "vision_tower.transformer.layers.0"
        sglang = [f"{layer}.attention.qkv_proj", f"{layer}.attention.proj", f"{layer}.feed_forward.gate_up_proj",
                  f"{layer}.feed_forward.down_proj"]  # fmt: skip
        vllm = [f"{layer}.attention.{p}_proj" for p in "qkvo"] + [
            f"{layer}.feed_forward.{p}_proj" for p in ("gate", "up", "down")
        ]
        model_class = transformers.Mistral3ForConditionalGeneration
    else:
        config = transformers.Gemma3Config(
            text_config=dict(vocab_size=300, **text),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                               patch_size=14, image_size=28),
            mm_tokens_per_image=4, image_token_index=299,
        )  # fmt: skip
        layer = "vision_tower.vision_model.encoder.layers.0"
        sglang = [
            f"{layer}.self_attn.qkv_proj",
            f"{layer}.self_attn.proj",
            f"{layer}.mlp.fc1",
            f"{layer}.mlp.fc2",
        ]
        vllm = [f"{layer}.self_attn.{p}_proj" for p in ("q", "k", "v", "out")] + [
            f"{layer}.mlp.fc1",
            f"{layer}.mlp.fc2",
        ]
        model_class = transformers.Gemma3ForConditionalGeneration
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = model_class._from_config(config).to(torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    writer = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-sym-g128", Target())
    for ctx in contexts(m, "w4a16-sym-g128", tokens=8):
        REGISTRY["rtn"]().apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(m.config, None, {})
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    # SGLang: a literal (or dotted-suffix) match, no fused expansion for these classes' vision tower
    assert [n for n in sglang if not any(_ignore_matches(n, e) for e in ignore)] == []
    assert [n for n in vllm if n not in ignore] == []  # vLLM: the stored names, exactly
    from qda_quant.arch.adapters.gemma3_vl import SERVING_NAMES

    renames = SERVING_NAMES if family == "gemma3" else {}
    stored = load_file(src / "model.safetensors")
    written = load_file(out / "model-00001.safetensors")
    for key, value in stored.items():
        if "vision_tower" in key:
            assert torch.equal(written[served_name(key, renames)], value), key


def test_deepseek_v4_default_export_keeps_the_attention_inputs_float_under_both_fused_names(tmp_path):
    """Stock SGLang 0.5.23 fuses wq_a / wkv into ``self_attn.wqkv_a`` by default and its loader skips packed inputs
    there without an error, leaving the fused weight unwritten. The default export stores the pair float and
    names it as each engine builds it: SGLang's literal ``wqkv_a``, vLLM's ``fused_wqa_wkv`` and both shards."""
    from safetensors.torch import load_file, save_file

    cfg = model("deepseek_v4", meta=True).config
    cfg.mlp_layer_types = ["hash_moe", "hash_moe", "moe", "moe"]  # stock serving needs the hash layers first
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    stored = load_file(src / "model.safetensors")  # the real checkpoint omits the model prefix
    save_file(
        {k.removeprefix("model."): v for k, v in stored.items()},
        src / "model.safetensors",
        metadata={"format": "pt"},
    )
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    writer = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-sym-g128", Target())
    for ctx in contexts(m, "w4a16-sym-g128"):
        REGISTRY["rtn"]().apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(m.config, None, {})
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    ignore = set(json.loads((out / "config.json").read_text())["quantization_config"]["ignore"])
    for i in range(m.config.num_hidden_layers):
        for p in ("wq_a", "wkv"):
            assert (
                f"layers.{i}.attn.{p}.weight" in index and f"layers.{i}.attn.{p}.weight_packed" not in index
            )
            assert f"layers.{i}.attn.{p}" in ignore
        assert {f"model.layers.{i}.self_attn.wqkv_a", f"model.layers.{i}.attn.fused_wqa_wkv"} <= ignore
    assert any(k.endswith("attn.wq_b.weight_packed") for k in index)  # the rest of the attention stays packed


def _runtime_adapter(monkeypatch, declarations):
    """Register a qwen3 adapter whose serving engines declare `declarations`."""

    class RuntimeAdapter:
        def matches(self, model):
            return model.config.model_type == "qwen3"

        def adjust(self, model, graph):
            return graph

        def runtime_names(self, model):
            return declarations

    monkeypatch.setitem(ADAPTERS._entries, "test_runtime_names", RuntimeAdapter())


def _runtime_fused_experts(monkeypatch, declarations):
    """Give the built-in fused-experts adapter the test's runtime-name declarations."""
    monkeypatch.setattr(FUSED_EXPERTS, "runtime_names", lambda model: declarations, raising=False)


QWEN3_QKV = r"re:model\.layers\.\d+\.self_attn\.[qkv]_proj$"
FUSED_QKV = r"re:runtime\.layers\.\d+\.qkv_proj$"

# One row per alias case: the declarations, then the refusal the build must raise or the names it must write.
RUNTIME_CASES = {
    "dense-unquantized-modules": (
        {
            "sglang": {
                "model.engine_head": "lm_head",  # float: emitted
                "model.engine_q_proj": "model.layers.0.self_attn.q_proj",  # quantized: not emitted
                r"re:model\.layers\.\d+\.engine_norm": r"re:model\.layers\.\d+\.input_layernorm",
            },
            "other": {"model.other_embed": "model.embed_tokens"},
        },
        {
            "ignore_in": ("model.engine_head", "model.other_embed", r"re:model\.layers\.\d+\.engine_norm"),
            "ignore_out": ("model.engine_q_proj",),
        },
    ),
    # the runtime name is a packed checkpoint module, so every engine would skip it
    "runtime-name-is-packed": (
        {"sglang": {"model.layers.0.self_attn.q_proj": "model.embed_tokens"}},
        "quantized module",
    ),
    # a broad runtime pattern claims the packed modules of the checkpoint's own names
    "runtime-regex-claims-packed": (
        {"sglang": {r"re:model\.layers\.\d+\..*": "model.embed_tokens"}},
        "quantized module",
    ),
    # one engine builds the runtime module float, another packed: one list cannot serve both
    "engines-disagree": (
        {
            "a": {"model.engine_projection": "model.embed_tokens"},
            "b": {"model.engine_projection": "model.layers.0.self_attn.q_proj"},
        },
        "quantized module",
    ),
    # the second engine names its packed module with a regex that claims the first engine's entry
    "regex-names-packed": (
        {
            "a": {"model.engine_projection": "model.embed_tokens"},
            "b": {r"re:model\.engine_.*": "model.layers.0.self_attn.q_proj"},
        },
        "quantized module",
    ),
    # a plain entry also matches dotted suffixes in SGLang, so it claims the module the regex names
    "suffix-entry-claims-regex-name": (
        {
            "a": {"engine_projection": "model.embed_tokens"},
            "b": {r"re:model\.engine_projection$": "model.layers.0.self_attn.q_proj"},
        },
        "quantized module",
    ),
    # an engine may alias the float router of a fused MoE; every expert it builds stays packed
    "symmetric-moe": (
        {"sglang": {"model.layers.0.mlp.router": "model.layers.0.mlp.gate"}},
        {
            "patch": "fused",
            "model": "qwen3_moe",
            "ignore_in": ("model.layers.0.mlp.router", "model.layers.0.mlp.gate"),
            "no_experts": True,
        },
    ),
    # a fused bank is written per expert, so an alias naming one of them is a packed module
    "packed-moe-expert": (
        {"sglang": {"model.layers.0.mlp.experts.0.gate_proj": "model.embed_tokens"}},
        {"patch": "fused", "model": "qwen3_moe", "refused": "quantized module"},
    ),
    # SGLang expands a fused module unless its literal name is ignored, and a regex is not a literal name
    "fused-regex": (
        {"sglang": {FUSED_QKV: QWEN3_QKV}},
        {"keep": ("q_proj", "k_proj", "v_proj"), "refused": "fused runtime module"},
    ),
    # the fused pattern (vLLM matches it first) and every shard pattern (what SGLang probes) are written
    "fused-regex-with-shards": (
        {
            "sglang": {
                FUSED_QKV: FusedRuntime(
                    checkpoint=QWEN3_QKV,
                    shards=tuple(rf"re:runtime\.layers\.\d+\.{part}_proj$" for part in "qkv"),
                )
            }
        },
        {
            "keep": ("q_proj", "k_proj", "v_proj"),
            "ignore_in": (FUSED_QKV, *(rf"re:runtime\.layers\.\d+\.{part}_proj$" for part in "qkv")),
            "index_in": tuple(f"model.layers.{i}.self_attn.{p}_proj.weight" for i in range(2) for p in "qkv"),
            "index_out": tuple(
                f"model.layers.{i}.self_attn.{p}_proj.weight_packed" for i in range(2) for p in "qkv"
            ),
        },
    ),
    # a written shard is an ignore entry of its own, so it must not claim a packed weight
    "fused-shard-packed": (
        {
            "sglang": {
                FUSED_QKV: FusedRuntime(
                    checkpoint=QWEN3_QKV, shards=(r"re:model\.layers\.\d+\.self_attn\.[qkvo]_proj$",)
                )
            }
        },
        {"keep": ("q_proj", "k_proj", "v_proj"), "refused": "quantized module"},
    ),
}


@pytest.mark.parametrize(("declarations", "expected"), list(RUNTIME_CASES.values()), ids=list(RUNTIME_CASES))
def test_runtime_names(declarations, expected, monkeypatch, tmp_path):
    """A serving engine builds modules under its own names, so those names need ``ignore`` entries of their own.

    An entry appears only when its checkpoint side misses every quantized module: the engine must build that
    weight float, and naming a quantized weight would break its load. One ``ignore`` list serves every engine,
    so a declaration that would claim a packed weight, in any engine, is refused rather than written and
    silently missed.
    """
    expected = {"refused": expected} if isinstance(expected, str) else expected
    {"adapter": _runtime_adapter, "fused": _runtime_fused_experts}[expected.get("patch", "adapter")](
        monkeypatch, declarations
    )
    kind, keep = expected.get("model", "qwen3"), expected.get("keep", ())
    if refused := expected.get("refused"):
        with pytest.raises(Refused, match=refused):
            _build(kind, "w4a16-sym-g128", tmp_path, keep=keep)
        assert not (tmp_path / "out" / "config.json").exists()
        return
    _, _, out = _build(kind, "w4a16-sym-g128", tmp_path, keep=keep)
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    assert set(expected.get("ignore_in", ())) <= set(ignore)
    assert not set(expected.get("ignore_out", ())) & set(ignore)
    if expected.get("no_experts"):  # a fused bank is written per expert: its router is the only float module
        assert not any(".experts." in name for name in ignore)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(expected.get("index_in", ())) <= set(index)
    assert not set(expected.get("index_out", ())) & set(index)


@pytest.mark.parametrize("packed", [(), ("gate_proj", "up_proj")])
def test_deepseek_shared_runtime_name_never_hides_packed_shards(tmp_path, packed):
    """SGLang needs the literal shared gate/up alias only when both checkpoint shards stay float.

    The shared role is kept float by default, but a config can quantize it. An unconditional alias would
    make the loader look for float weights that the checkpoint no longer contains in the latter case.
    """
    src = tmp_path / "src"
    m = model("deepseek_v2", dtype=torch.bfloat16)
    m.save_pretrained(src, save_original_format=False)
    save_tiny_tokenizer(src)
    shared = next(n for n, _ in m.named_modules() if n.endswith(".shared_experts"))
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    out = BuildConfig(
        model=str(src),
        format="w4a16-sym-g64",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(tmp_path / "out"),
        **({"ignore": [shared + ".down_proj"]} if packed else {}),
    ).build()
    cfg = json.loads((out / "config.json").read_text())["quantization_config"]
    assert (shared + ".gate_up_proj" in cfg["ignore"]) == (not packed)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for projection in ("gate_proj", "up_proj"):
        key = f"{shared}.{projection}.weight" + ("_packed" if projection in packed else "")
        assert key in index
    defaults = ADAPTERS["deepseek_v2"].default_ignore(m)
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    for name in defaults:
        assert (name in resolved["ignore"]) == (not packed)
        assert (name + ".weight" in index) == (not packed)


@pytest.mark.parametrize("packed", [(), ("gate_proj", "up_proj")])
def test_qwen2_moe_shared_runtime_name_never_hides_packed_shards(tmp_path, packed):
    """SGLang builds the shared expert's gate/up as one fused module and probes that literal name.

    The shared role is kept float by default, so the fused name needs an ``ignore`` entry of its own. A
    config that quantizes the shared expert must not get the alias, or the loader would look for float
    weights the checkpoint no longer stores.
    """
    src = tmp_path / "src"
    m = model("qwen2_moe", dtype=torch.bfloat16)
    m.save_pretrained(src, save_original_format=False)
    save_tiny_tokenizer(src)
    shared = next(n for n, _ in m.named_modules() if n.endswith(".mlp.shared_expert"))
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    out = BuildConfig(
        model=str(src),
        format="w4a16-sym-g64",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(tmp_path / "out"),
        **({"ignore": [shared + ".down_proj"]} if packed else {}),
    ).build()
    cfg = json.loads((out / "config.json").read_text())["quantization_config"]
    assert (shared + ".gate_up_proj" in cfg["ignore"]) == (not packed)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for projection in ("gate_proj", "up_proj"):
        key = f"{shared}.{projection}.weight" + ("_packed" if projection in packed else "")
        assert key in index
    assert f"{shared}.down_proj.weight" in index  # the explicit ignore keeps only the down projection float


def test_runtime_names_refuse_a_fused_regex_with_native_checkpoint_names(monkeypatch, tmp_path):
    """Native DeepSeek source names differ from graph layer names; fused aliases must still be refused."""
    from safetensors.torch import load_file, save_file

    runtime = r"re:model\.layers\.\d+\.mlp\.shared_experts\.gate_up_proj$"
    checkpoint = r"re:layers\.\d+\.ffn\.shared_experts\.w[13]$"
    monkeypatch.setattr(
        DEEPSEEK_V4,
        "runtime_names",
        lambda model: {"sglang": {runtime: checkpoint}},
        raising=False,
    )
    src = tmp_path / "src"
    cfg = model("deepseek_v4", meta=True).config
    cfg.mlp_layer_types = ["hash_moe", "hash_moe", "moe", "moe"]
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    m.save_pretrained(src)
    tensors = load_file(src / "model.safetensors")
    save_file(
        {k.removeprefix("model."): v for k, v in tensors.items()},
        src / "model.safetensors",
        metadata={"format": "pt"},
    )
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    w = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-sym-g128", Target())
    for ctx in contexts(m, "w4a16-sym-g128"):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        for name in tuple(ctx.qparams):
            if ".shared_experts." in name and name.endswith((".gate_proj", ".up_proj")):
                del ctx.qparams[name]
        w.add_layer(ctx)
    with pytest.raises(Refused, match=r"layers\.0\.ffn\.shared_experts\.w1.*fused runtime module"):
        w.finish(m.config, None, {"ok": True})
    assert not (tmp_path / "out" / "config.json").exists()


@pytest.mark.parametrize("fmt", ["w4a16", "w4a16-sym-g128", "w4a8", "w3a16"])
def test_roundtrip_dense(fmt, tmp_path):
    _, served, out = _build("qwen3", fmt, tmp_path)
    serve = json.loads((out / "build.json").read_text())["serve"]
    assert set(serve["kernels"]) == {"dense"} and serve["vllm"] == "0.33.0"
    cfg = json.loads((out / "config.json").read_text())["quantization_config"]
    assert cfg["format"] == "pack-quantized" and "lm_head" in cfg["ignore"]
    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    sd = back.state_dict()
    for n, p in served.state_dict().items():
        if n in sd and sd[n].dtype == torch.bfloat16 and "rotary" not in n:
            if fmt == "w4a8" and "layers" in n and "norm" not in n:
                continue  # served W4A8 weight is the FP8 product; checked through codes in test_steps
            assert torch.equal(sd[n], p), n


def test_tied_head_stays_float_and_loads(tmp_path):
    """A tied head stores no ``lm_head.weight``, so the source-tensor scan alone drops it from ``ignore`` while the
    group over every Linear still claims it; transformers then loads ``lm_head`` as packed and fails. The output
    must name the tied head in ``ignore``, keep it tied and run."""
    from qda_quant import Recipe, build

    from ..engine_util import save_tiny_tokenizer, write_ids

    cfg = transformers.Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                                   vocab_size=128, max_position_embeddings=64, tie_word_embeddings=True)  # fmt: skip
    torch.manual_seed(0)
    ref = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    src = tmp_path / "src"
    ref.save_pretrained(src)
    assert "lm_head.weight" not in _weight_map(src)

    sha = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    g = torch.Generator().manual_seed(0)
    write_ids(calib, [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(4)], sha)
    out = build(
        str(src),
        Recipe.from_dict(
            {"steps": [{"rtn": {}}], "calib": {"file": {"path": str(calib), "samples": 4, "seqlen": 32}}}
        ),
        format="w4a16",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        log=lambda s: None,
    )
    assert "lm_head" in json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]

    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).eval()
    assert back.lm_head.weight is back.model.embed_tokens.weight  # loaded, not re-initialized
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        p, q = ref(input_ids=ids).logits.log_softmax(-1), back(input_ids=ids).logits.log_softmax(-1)
    assert (p.exp() * (p - q)).sum(-1).mean() < 0.5  # smoke bound only: loads and is not garbage


@pytest.mark.parametrize("kind", ["qwen2", "llama-bias"])
def test_linear_biases_are_written(kind, tmp_path):
    src, _, out = _build(kind, "w4a16-sym-g128", tmp_path)
    ref = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    biases = {n for n, _ in ref.named_parameters() if n.endswith(".bias")}
    assert biases
    written = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert biases <= set(written), sorted(biases - set(written))[:3]
    for n in _weight_map(src):  # every source tensor is written or replaced by its packed form
        assert n in written or n.removesuffix(".weight") + ".weight_packed" in written, n
    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    sd = back.state_dict()
    for n in biases:
        assert torch.equal(sd[n], ref.state_dict()[n]), n
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        p, lq = ref(input_ids=ids).logits.log_softmax(-1), back(input_ids=ids).logits.log_softmax(-1)
    assert (p.exp() * (p - lq)).sum(-1).mean() < 0.05


def test_moe_writes_per_expert_names(tmp_path):
    src, _, out = _build("qwen3_moe", "w4a16", tmp_path)
    src_keys = (
        set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"])
        if (src / "model.safetensors.index.json").exists()
        else None
    )
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.mlp.experts.3.gate_proj.weight_packed" in idx
    assert "model.layers.0.mlp.experts.3.down_proj.weight_zero_point" in idx
    assert not any(k.endswith("gate_up_proj") for k in idx)
    assert "model.layers.0.mlp.gate.weight" in idx  # the router stays bf16
    # asymmetric experts (they store zero points): vLLM 0.33 on sm90 serves them on Marlin
    assert json.loads((out / "build.json").read_text())["serve"]["kernels"]["moe"] == ["MarlinExperts"]
    del src_keys


def test_a_whole_float_routed_bank_is_copied_and_marked_serve_only(tmp_path):
    """`ignore` may keep a whole routed bank bf16: it is copied as stored, the rest is quantized.

    The tiny source stores its experts one module per expert (``experts.0.gate_proj``), which stock
    transformers loads with every routed expert at its random initialization, so the export records
    ``build.json["serve_only"]`` and `load_model` refuses it while the architecture alone stays constructible.
    """
    from qda_quant import FileSource, Recipe, build
    from qda_quant.intake.loading import load_model

    from ..engine_util import save_tiny_tokenizer, tiny_qwen3_moe, write_ids

    src = tmp_path / "src"
    tiny_qwen3_moe(layers=2).to(torch.bfloat16).save_pretrained(src, save_original_format=True)
    fingerprint = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], fingerprint)
    out = build(
        str(src),
        Recipe(steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-sym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        ignore=[r"re:^model\.layers\.0\.mlp\.experts\."],
        log=lambda s: None,
    )
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in idx  # float, as the source stores it
    assert "model.layers.1.mlp.experts.0.gate_proj.weight_packed" in idx  # the other bank is quantized
    from qda_quant.output.export import _ignore_matches

    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]

    def ignored(name):  # what vLLM and SGLang ask before they build a bank float
        return any(_ignore_matches(name, entry) for entry in ignore)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert ignored(f"model.layers.0.mlp.experts.0.{proj}")
        assert not any(ignored(f"model.layers.{i}.mlp.experts.0.{proj}") for i in (1,))
    mark = json.loads((out / "build.json").read_text())["serve_only"]
    assert mark["float_banks"] == ["model.layers.0.mlp.experts"]
    with pytest.raises(Refused, match="serve-only"):
        load_model(out)
    assert (
        load_model(out, weights=False).config.num_hidden_layers == 2
    )  # the architecture alone is constructible


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32], ids=["bf16-source", "f32-source"])
def test_a_fused_float_routed_bank_is_stored_per_expert(tmp_path, source_dtype):
    """`ignore` may keep a bank float whose source stores it fused: it is split, not copied fused.

    A stored ``...experts.gate_up_proj`` name flips stock SGLang 0.5.23's one ``is_fused_expert`` flag for the
    whole checkpoint, and every per-expert tensor stored after it is skipped with a warning (the 35B k5
    serving failure, 6,912 tensors). The split keeps the stored tensor's values and expert order, so an F32
    source stays F32 under a bf16 build instead of being sliced from the runtime copy.
    """
    from qda_quant import FileSource, Recipe, build
    from safetensors.torch import load_file, save_file

    from ..engine_util import save_tiny_tokenizer, write_ids

    src = tmp_path / "src"
    model("qwen3_5_moe", dtype=source_dtype).save_pretrained(src)
    # the published checkpoints store each bank fused; transformers writes the tiny model per expert
    tensors = load_file(str(src / "model.safetensors"))
    experts = {}
    for key, value in tensors.items():
        match = re.match(r"(.*\.mlp\.experts\.)(\d+)\.(gate|up|down)_proj\.weight$", key)
        if match is not None:
            experts.setdefault(match.group(1), {}).setdefault(int(match.group(2)), {})[match.group(3)] = value
    for prefix, bank in experts.items():
        order = sorted(bank)
        tensors[prefix + "gate_up_proj"] = torch.stack(
            [torch.cat((bank[index]["gate"], bank[index]["up"]), dim=0) for index in order]
        )
        tensors[prefix + "down_proj"] = torch.stack([bank[index]["down"] for index in order])
        for index in order:
            for projection in ("gate", "up", "down"):
                del tensors[f"{prefix}{index}.{projection}_proj.weight"]
    save_file(tensors, str(src / "model.safetensors"), metadata={"format": "pt"})
    source = dict(tensors)
    bank = "model.layers.0.mlp.experts"
    assert f"{bank}.gate_up_proj" in source and f"{bank}.down_proj" in source
    tokenizer_sha = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], tokenizer_sha)
    out = build(
        str(src),
        Recipe(steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-sym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        ignore=[r"re:^model\.layers\.0\.mlp\.experts\."],
        log=lambda s: None,
    )
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(key.endswith((".experts.gate_up_proj", ".experts.down_proj")) for key in index)
    fused, down = source[f"{bank}.gate_up_proj"], source[f"{bank}.down_proj"]
    for expert in range(fused.shape[0]):
        gate_up = fused[expert]
        gate = gate_up[: gate_up.shape[0] // 2]
        expected = {
            "gate_proj": gate,
            "up_proj": gate_up[gate_up.shape[0] // 2 :],
            "down_proj": down[expert],
        }
        for projection, value in expected.items():
            key = f"{bank}.{expert}.{projection}.weight"
            with safe_open(str(out / index[key]), framework="pt") as shard:
                got = shard.get_tensor(key)
                assert got.dtype == source_dtype, key
                assert torch.equal(got, value), key
    assert "model.layers.1.mlp.experts.0.gate_proj.weight_packed" in index  # the other bank is quantized
    mark = json.loads((out / "build.json").read_text())["serve_only"]
    assert mark["float_banks"] == [bank]


def test_a_source_prefixed_fused_float_bank_is_split_and_marked_serve_only(tmp_path):
    """A fused float bank stored under the engines' source container is split there and marked there.

    The Qwen3.5/3.6 VL models store a fused bank under ``model.language_model.`` while the graph names it
    ``model.layers.`` with no serving rename. The split writes the per-expert tensors under the source's
    container; `finish` must record ``serve_only`` under those written roots, or a checkpoint whose experts
    stock transformers leaves at random initialization would pass the safe-load refusal.
    """
    from safetensors.torch import save_file

    src = tmp_path / "src"
    src.mkdir()
    runtime = "model.layers.0.mlp.experts"
    source = "model.language_model.layers.0.mlp.experts"
    original = torch.arange(64, dtype=torch.bfloat16).reshape(2, 4, 8)
    save_file({f"{source}.gate_up_proj": original}, src / "model.safetensors")
    (src / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "hidden_size": 8, "num_hidden_layers": 1, "num_experts": 2})
    )
    graph = SimpleNamespace(
        extra={"float_banks": [runtime]}, adapter=None, layers=[], embed=None, lm_head=None
    )
    writer = Writer(tmp_path / "out", src, graph, Format.parse("w4a16-sym-g32"), Target())
    finished = False
    try:
        writer._push(writer._spread_float_bank(f"{runtime}.gate_up_proj", original))
        out = writer.finish(None, None, {})
        finished = True
    finally:
        if not finished:
            writer._q.put(None)
            writer._t.join()
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == {
        f"{source}.{e}.{proj}.weight" for e in range(2) for proj in ("gate_proj", "up_proj")
    }, sorted(index)
    mark = json.loads((out / "build.json").read_text())["serve_only"]
    assert mark["float_banks"] == [source]


def test_build_writes_fused_hybrid_experts(tmp_path):
    """Fused source banks become per-expert serving fields without changing solved weights."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant import FileSource, Recipe, build
    from qda_quant.steps.gptq import GPTQ
    from safetensors import safe_open

    cfg = transformers.Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.03)

    src = tmp_path / "hybrid"
    m.save_pretrained(src)
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(src / "model.safetensors"))
    experts = {}
    fused = {}
    for key, value in tensors.items():
        match = re.match(r"(.*\.mlp\.experts\.)(\d+)\.(gate|up|down)_proj\.weight$", key)
        if match is None:
            fused[key] = value
        else:
            experts.setdefault(match.group(1), {}).setdefault(int(match.group(2)), {})[match.group(3)] = value
    for prefix, bank in experts.items():
        order = sorted(bank)
        fused[prefix + "gate_up_proj"] = torch.stack(
            [torch.cat((bank[index]["gate"], bank[index]["up"]), dim=0) for index in order]
        )
        fused[prefix + "down_proj"] = torch.stack([bank[index]["down"] for index in order])
    save_file(fused, str(src / "model.safetensors"), metadata={"format": "pt"})
    assert not any(re.search(r"\.experts\.\d+\.", name) for name in fused)
    assert sum(name.endswith(".experts.gate_up_proj") for name in fused) == 2
    from ..engine_util import save_tiny_tokenizer, write_ids

    tokenizer_sha = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], tokenizer_sha)
    propagated = {}

    class CapturingWriter(Writer):
        def add_layer(self, ctx):
            prefix = ctx.graph.module + "."
            for name in ctx.qparams:
                if not name.startswith(prefix) or not name.endswith(
                    ("experts.gate_up_proj", "experts.down_proj")
                ):
                    continue
                rel = name[len(prefix) :]
                propagated[name] = ctx.layer.get_parameter(rel).detach().clone()
            super().add_layer(ctx)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = build(
            str(src),
            Recipe(
                steps=[GPTQ(impl="torch")],
                calib=FileSource(path=str(calib), samples=2, seqlen=8),
            ),
            format="w4a16-asym-g128",  # this test unpacks zero points; it names the symmetry it stores
            out_dir=tmp_path / "packed",
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter=writes(CapturingWriter),
        )
    finally:
        torch.set_num_threads(threads)

    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(propagated) == 4
    assert not any(name + "_packed" in index or name in index for name in propagated)
    for name, expected in propagated.items():
        projections = ("gate_proj", "up_proj") if name.endswith("gate_up_proj") else ("down_proj",)
        base = name.rpartition(".")[0]
        rows = expected.shape[1] // len(projections)
        for expert in range(expected.shape[0]):
            for projection, leaf in enumerate(projections):
                prefix = f"{base}.{expert}.{leaf}.weight_"
                with safe_open(out / index[prefix + "packed"], framework="pt") as shard:
                    shape = torch.Size(shard.get_tensor(prefix + "shape").tolist())
                    assert shape == (rows, expected.shape[2])
                    codes = unpack_from_int32(shard.get_tensor(prefix + "packed"), 4, shape).float()
                    scale = shard.get_tensor(prefix + "scale").float()
                    zero = unpack_from_int32(
                        shard.get_tensor(prefix + "zero_point"),
                        4,
                        torch.Size((rows, scale.shape[-1])),
                        packed_dim=0,
                    ).float()
                group_size = shape[-1] // scale.shape[-1]
                dequantized = (
                    (codes - zero.repeat_interleave(group_size, dim=-1))
                    * scale.repeat_interleave(group_size, dim=-1)
                ).to(expected.dtype)
                assert torch.equal(
                    dequantized, expected[expert, projection * rows : (projection + 1) * rows]
                ), (name, expert, leaf)


def test_gpt_oss_experts_are_stored_one_per_tensor_for_stock_vllm(tmp_path):
    """Regression against the silent-skip bug: vLLM's gpt-oss loader maps an INT4 expert only by
    ``...experts.<id>.<w13|w2>_*``; a fused ``gate_up_proj_packed`` reaches no branch and is dropped without an
    error, leaving every expert at its random initialization. Every expert tensor must be one the map accepts."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant import Format

    _, m, out = _build("gpt_oss", "w4a16-sym-g64", tmp_path)
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    per_expert = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)$")
    experts: dict[tuple[int, int], set[str]] = {}
    fused: set[str] = set()
    for key in idx:
        if ".mlp.experts." not in key:
            continue
        hit = per_expert.match(key)
        if hit is None:
            fused.add(key.rpartition(".")[2])
            assert key.endswith(("gate_up_proj_bias", "down_proj_bias")), key
            continue
        experts.setdefault((int(hit.group(1)), int(hit.group(2))), set()).add(hit.group(3))
    assert sorted(experts) == [
        (layer, expert)
        for layer in range(m.config.num_hidden_layers)
        for expert in range(m.config.num_local_experts)
    ]
    wanted = {"w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"}
    assert all(suffixes == wanted for suffixes in experts.values()), experts
    assert fused == {"gate_up_proj_bias", "down_proj_bias"}  # fused: vLLM maps the names itself
    f = Format.parse("w4a16-sym-g64")
    hidden, inter = m.config.hidden_size, m.config.intermediate_size
    with safe_open(out / idx["model.layers.0.mlp.experts.5.w13_weight_packed"], "pt") as h:
        packed = h.get_tensor("model.layers.0.mlp.experts.5.w13_weight_packed")
        scale = h.get_tensor("model.layers.0.mlp.experts.5.w13_weight_scale")
    assert packed.dtype == torch.int32 and scale.dtype == torch.bfloat16
    assert unpack_from_int32(packed, f.weight_bits, (2 * inter, hidden)).shape == (2 * inter, hidden)
    assert scale.shape == (2 * inter, hidden // f.group_size)


def test_gpt_oss_refuses_asymmetric_experts_at_plan_time(tmp_path):
    """Stock vLLM has no zero-point path for gpt-oss experts (its loader raises IndexError), so a plan must
    refuse before anything runs rather than write a checkpoint that cannot load. A bare spelling defaults the
    experts symmetric, so only an explicit ``-asym`` reaches that refusal."""
    from qda_quant import Format, Recipe
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.planning import compile_plan
    from qda_quant.step import Refused
    from qda_quant.target import Target

    from ..engine_util import tiny_gpt_oss

    src = tmp_path / "gpt-oss"
    tiny_gpt_oss(layers=1).save_pretrained(src)
    recipe = Recipe(steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(tmp_path / "calib.jsonl")))
    for fmt, reason in (("w4a16-asym-g64", "zero points"), ("w4a8", "WNA16")):
        with pytest.raises(Refused, match=reason):
            compile_plan(str(src), recipe, Format.parse(fmt), Target(), [torch.device("cpu")])
    p = compile_plan(str(src), recipe, Format.parse("w4a16"), Target(), [torch.device("cpu")])
    assert p.format_defaults == {"routed_experts": "w4a16-sym-g128"}


@pytest.mark.parametrize("fmt", ["w4a16-sym-g64", "w4a16-asym-g64"])
def test_gemma4_expert_layout_keeps_every_projection_and_its_values(fmt, tmp_path):
    """Both serving loaders map per-expert projections; SGLang misreads fused packed names as BF16."""
    from qda_quant.engine.packed import Packed

    config = transformers.Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size=128,
        max_position_embeddings=64,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        hidden_size_per_layer_input=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src)
    # Gemma 4 stores each bank fused ([E, 2I, D] and [E, D, I]), as its published checkpoints do; this test
    # covers that source, not a per-expert one.
    with safe_open(str(src / "model.safetensors"), framework="pt") as shard:
        source = set(shard.keys())
    assert {key for key in source if ".experts." in key} == {
        f"model.layers.{layer}.experts.{param}"
        for layer in range(config.num_hidden_layers)
        for param in ("gate_up_proj", "down_proj")
    }
    writer = Writer(tmp_path / "out", src, describe(m, verify=False), fmt, Target())
    for ctx in contexts(m, fmt, tokens=16):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(m.config, None, {})
    packed = Packed(out)
    fields = {"packed", "scale", "shape"}
    if not packed.fmt.symmetric:
        fields.add("zero_point")
    expected = {
        f"model.layers.{layer}.experts.{expert}.{projection}.weight_{field}"
        for layer in range(config.num_hidden_layers)
        for expert in range(config.num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for field in fields
    }
    assert {key for key in packed.where if ".experts." in key} == expected
    for layer in range(config.num_hidden_layers):
        bank = m.model.layers[layer].experts
        for expert in range(config.num_experts):
            weights = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                stem = f"model.layers.{layer}.experts.{expert}.{projection}"
                codes, zero = packed.codes(stem)
                weights.append(
                    packed.quantizer(stem, "moe").dequant(codes, packed.tensor(stem + ".weight_scale"), zero)
                )
            assert torch.equal(torch.cat(weights[:2]).to(bank.gate_up_proj.dtype), bank.gate_up_proj[expert])
            assert torch.equal(weights[2].to(bank.down_proj.dtype), bank.down_proj[expert])


@pytest.mark.parametrize("kind", ["deepseek_v4", "mixtral", "minimax_m2"])
@pytest.mark.parametrize("stacked", [False, True], ids=["per-expert", "stacked"])
def test_expert_source_conversion_preserves_projection_and_expert_order(kind, stacked, tmp_path):
    """Stock source conversions must map every packed expert to the weight the build propagated."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant import FileSource, Recipe, build
    from safetensors.torch import load_file, save_file

    from ..engine_util import save_tiny_tokenizer, write_ids
    from ..tiny import SMALL

    if kind == "deepseek_v4":
        cfg = model(kind, meta=True).config
        cfg.mlp_layer_types = ["hash_moe", "hash_moe", "moe", "moe"]
        m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    else:
        config_class = {"mixtral": transformers.MixtralConfig, "minimax_m2": transformers.MiniMaxM2Config}[
            kind
        ]
        cfg = config_class(
            **SMALL, num_local_experts=12, num_experts_per_tok=2, bos_token_id=1, eos_token_id=2
        )
        cfg._attn_implementation = "eager"
        cfg._experts_implementation = "eager"
        torch.manual_seed(0)
        m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    src = tmp_path / "src"
    m.save_pretrained(src, save_original_format=not stacked)
    if kind == "deepseek_v4":
        stored = load_file(src / "model.safetensors")
        save_file(
            {k.removeprefix("model."): v for k, v in stored.items()},
            src / "model.safetensors",
            metadata={"format": "pt"},
        )  # the real checkpoint omits the model prefix
    fingerprint = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], fingerprint)
    propagated = {}

    class CapturingWriter(Writer):
        def add_layer(self, ctx):
            for bank in ctx.graph.banks:
                for param in ("gate_up_proj", "down_proj"):
                    name = f"{bank.experts}.{param}"
                    propagated[name] = (
                        ctx.layer.get_parameter(name.removeprefix(ctx.graph.module + ".")).detach().clone()
                    )
            super().add_layer(ctx)

    out = build(
        str(src),
        Recipe(steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-sym-g128",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter=writes(CapturingWriter),
        log=lambda s: None,
    )
    written = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}
    assert propagated
    for name, weight in propagated.items():
        bank, _, param = name.rpartition(".")
        root = bank
        if kind == "deepseek_v4":
            root = root.removeprefix("model.")
            root = root.replace(".mlp.", ".ffn.")
        elif kind in ("mixtral", "minimax_m2") or not stacked:  # stock engines build the sparse block so
            root = root.replace(".mlp.", ".block_sparse_moe.")
        if param == "down_proj":
            projection = "w2"
            keys = [(f"{root}.{i}.{projection}.weight", weight[i]) for i in range(weight.shape[0])]
        else:
            half = weight.shape[1] // 2
            projections = ("w1", "w3")
            keys = [
                (f"{root}.{i}.{projection}.weight", weight[i, a:b])
                for i in range(weight.shape[0])
                for projection, a, b in ((projections[0], 0, half), (projections[1], half, weight.shape[1]))
            ]
        for key, expected in keys:
            stem = key.removesuffix(".weight")
            sep = ".weight_"
            shape = expected.shape
            packed = written[stem + sep + "packed"]
            codes = unpack_from_int32(packed, 4, shape).float()
            scale = written[stem + sep + "scale"].float()
            decoded = (codes * scale.repeat_interleave(128, -1)).to(torch.bfloat16)
            assert torch.equal(decoded, expected), key
    source = load_file(src / "model.safetensors")
    for key in source:
        source_key = key
        if kind == "deepseek_v4":
            from qda_quant.arch.adapters.deepseek_v4 import DEEPSEEK_V4

            key = served_name(key, DEEPSEEK_V4.serve_names(m))
        if kind in ("mixtral", "minimax_m2"):
            from qda_quant.arch.adapters.mixtral import MIXTRAL

            key = served_name(key, MIXTRAL.serve_names(m))
        assert (
            key in written
            or key.removesuffix(".weight") + ".weight_packed" in written
            or key + "_packed" in written
            or (
                stacked
                and source_key.endswith((".experts.gate_up_proj", ".experts.down_proj"))
                and key.rpartition(".")[0] + ".0.w2.weight_packed" in written
            )
        ), key


def test_served_name_moves_a_longest_subtree_path_once_and_composes_segment_rules():
    """One declaration carries whole-path moves and one-segment rules; longest path wins at each boundary."""
    renames = {
        "mlp": "block_sparse_moe",
        "gate_proj": "w1",
        "model.layers": "layers",
        "self_attn": "attn",
        "indexer.compressor": "compressor.indexer",
        "ape": "position_bias",
        "block_sparse_moe": "unscanned",  # a value is written once and never rescanned
    }
    assert served_name("model.layers.0.mlp.experts.0.gate_proj.weight", renames) == (
        "layers.0.block_sparse_moe.experts.0.w1.weight"
    )
    assert served_name("model.layers.0.self_attn.indexer.compressor.kv_proj.weight", renames) == (
        "layers.0.attn.compressor.indexer.kv_proj.weight"
    )
    assert served_name("model.ape", renames) == "model.position_bias"
    assert served_name("layers.0.mlp.gate.weight", renames) == "layers.0.block_sparse_moe.gate.weight"


def test_served_names_refuses_a_path_with_an_empty_segment():
    with pytest.raises(Refused, match="dotted paths of non-empty name segments"):
        _serving_names("demo", {"mlp": "block_sparse_moe.", "a.b": "c"})


@pytest.mark.parametrize("kind", ["mixtral", "minimax_m2"])
@pytest.mark.parametrize("stacked", [False, True], ids=["per-expert", "stacked"])
def test_sparse_block_export_names_the_block_as_the_stock_engines_build_it(kind, stacked, tmp_path):
    """Stock vLLM 0.33 and SGLang 0.5.23 build Mixtral's and MiniMax-M2's sparse block as ``block_sparse_moe``.

    A transformers-5 source names it ``mlp``; written under that name, vLLM refuses the checkpoint and
    SGLang silently skips every expert and router. The adapter declares the rename, so the packed experts,
    their scales and the float router all land under the name both engines probe.
    """
    from ..tiny import SMALL

    config_class = {"mixtral": transformers.MixtralConfig, "minimax_m2": transformers.MiniMaxM2Config}[kind]
    cfg = config_class(**SMALL, num_local_experts=4, num_experts_per_tok=2, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.manual_seed(0)
    source = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    src = tmp_path / "src"
    source.save_pretrained(src, save_original_format=not stacked)
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    fmt = "w4a16-sym-g128"
    writer = Writer(tmp_path / "out", src, describe(model, verify=False), fmt, Target())
    for ctx in contexts(model, fmt):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        writer.add_layer(ctx)
    out = writer.finish(model.config, None, {"ok": True})
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    block = "model.layers.0.block_sparse_moe"
    assert f"{block}.gate.weight" in index
    assert any(k.startswith(f"{block}.experts.") for k in index)
    assert not any(".mlp." in k for k in index)
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    assert f"{block}.gate" in ignore
    assert not any(".mlp." in name for name in ignore)
    experts = [k for k in index if k.startswith(f"{block}.experts.")]
    assert {k.split(".")[-2] for k in experts} == {"w1", "w2", "w3"}


def _vllm_w4a8_weight(codes: torch.Tensor, scale_bf16: torch.Tensor) -> torch.Tensor:
    """The CUTLASS W4A8 weight, written from vLLM 0.33.0's source without qda_quant code.

    quant_utils.py:931 convert_bf16_scales_to_fp8 + input_quant_fp8.py:202-221 QuantFP8 (dynamic, per row):
    x_max = |s|.amax(-1) (bf16) -> fp32; c = max(x_max / 448, 1 / (448 * 512)); f = e4m3(clamp(s * (1 / c)));
    fp8_scales = e4m3(f / 8); chan = c * 8. Mainloop: e4m3(q * fp8_scale), times chan after the MMA.
    """
    x_max = scale_bf16.abs().amax(dim=-1).unsqueeze(-1).to(torch.float32)
    c = (x_max / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    f = (scale_bf16.to(torch.float32) * c.reciprocal()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    fp8_scales = (f.float() / 8.0).to(torch.float8_e4m3fn)
    chan = c * 8.0
    g = codes.shape[-1] // scale_bf16.shape[-1]
    prod = (codes.float() * fp8_scales.float().repeat_interleave(g, dim=-1)).to(torch.float8_e4m3fn)
    return prod.float() * chan


def _wna16_weight(codes, scale_bf16, zero):
    """Marlin / Machete WNA16: (q - z) * s, dequantized in bf16."""
    g = codes.shape[-1] // scale_bf16.shape[-1]
    x = codes.float() - (zero.float().repeat_interleave(g, dim=-1) if zero is not None else 0)
    return (x * scale_bf16.float().repeat_interleave(g, dim=-1)).to(torch.bfloat16).float()


@pytest.mark.parametrize("fmt", ["w4a8", "w4a16", "w4a16-sym-g128"])
def test_stored_codes_reproduce_the_served_weight(fmt, tmp_path):
    """Unpack what the checkpoint stores and apply the serving kernel's operator, implemented here from vLLM's
    source (not through qda_quant): it must equal, bit for bit, the weight the build propagated."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant import Format
    from safetensors import safe_open

    _, served, out = _build("qwen3", fmt, tmp_path)
    f = Format.parse(fmt)
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    n = 0
    for k, file in idx.items():
        if not k.endswith(".weight_packed"):
            continue
        mod = k[: -len(".weight_packed")]
        with safe_open(out / file, "pt") as h:
            shape = torch.Size(h.get_tensor(mod + ".weight_shape").tolist())
            codes = unpack_from_int32(h.get_tensor(k), f.weight_bits, shape)  # signed, as CT stores them
            scale = h.get_tensor(mod + ".weight_scale")
            assert scale.dtype == torch.bfloat16
            zero = None
            if not f.symmetric:
                zp = h.get_tensor(mod + ".weight_zero_point")
                zero = unpack_from_int32(
                    zp, f.weight_bits, torch.Size((shape[0], scale.shape[1])), packed_dim=0
                )
        want = _vllm_w4a8_weight(codes, scale) if f.act == "fp8_token" else _wna16_weight(codes, scale, zero)
        w = served.get_submodule(mod).weight.float()
        if (
            f.act == "fp8_token"
        ):  # the propagated weight is stored in the param dtype; compare at that precision
            want = want.to(served.get_submodule(mod).weight.dtype).float()
        assert torch.equal(want, w), (mod, (want - w).abs().max().item())
        n += 1
    assert n == 14  # 7 linears x 2 layers


def test_rotated_model_round_trips_with_add_model(tmp_path):
    from qda_quant import FileSource, Recipe

    src = tmp_path / "src"
    m = model("qwen3", dtype=torch.bfloat16)
    m.save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    graph = describe(m, verify=False)
    REGISTRY["rotate"]().apply_model(m, graph, None)
    w = Writer(tmp_path / "out", src, graph, "w4a16", Target())
    for ctx in contexts(m, "w4a16"):
        REGISTRY["gptq"](impl="torch").apply_layer(ctx)
        w.add_layer(ctx)
    w.add_model(m, {})
    out = w.finish(m.config, Recipe(steps=[], calib=FileSource(path="x")), {})
    q = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    with torch.no_grad():
        p, lq = ref.log_softmax(-1), q(input_ids=ids).logits.log_softmax(-1)
    assert (p.exp() * (p - lq)).sum(-1).mean() < 0.5  # without add_model the stale embedding gives garbage


def test_add_model_unchanged_reads_the_source(tmp_path):
    """`changed=False` writes the same shard bytes: no model step, so the source's tensors are the model's."""
    src = tmp_path / "src"
    model("qwen3", dtype=torch.bfloat16).save_pretrained(src)
    shards = []
    for name, changed in (("model", True), ("source", False)):
        m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
        graph = describe(m, verify=False)
        w = Writer(tmp_path / name, src, graph, "w4a16", Target())
        for ctx in contexts(m, "w4a16"):
            REGISTRY["rtn"]().apply_layer(ctx)
            w.add_layer(ctx)
        w.add_model(m, {}, changed=changed)
        out = w.finish(m.config, None, {})
        shards.append([(p.name, p.read_bytes()) for p in sorted(out.glob("*.safetensors"))])
    assert shards[0] == shards[1]


def test_fp8_source_kept_weights_decode_and_untouched_parts_survive(tmp_path):
    """Rewritten and untouched encoded weights both decode; no source scale of either survives (I6)."""
    from qda_quant.intake.source import SOURCE_CODECS
    from safetensors.torch import load_file, save_file

    from ..intake.test_source import ref_blocks

    src = tmp_path / "src"
    m = model("qwen2_moe", dtype=torch.bfloat16)
    m.save_pretrained(src)
    stored = load_file(src / "model.safetensors")
    kept = "model.layers.0.mlp.shared_expert.down_proj.weight"
    rewritten = [
        kept,
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    expected = {}
    codec = SOURCE_CODECS["fp8_block"]
    for key in rewritten:
        weight = stored[key].float()
        rows, cols = weight.shape
        scale = torch.full((-(-rows // 128), -(-cols // 128)), 0.004123, dtype=torch.float32)
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
        codes = (weight / expanded).to(torch.float8_e4m3fn)
        expected[key] = ref_blocks(codes, scale, (128, 128)).to(torch.bfloat16)
        stored[key] = codes
        stored[codec.parts(key)[1]] = scale
        m.state_dict()[key].copy_(expected[key])  # the loader's decoded model, before any step
    mtp = "mtp.layers.0.proj.weight"
    stored[mtp] = stored[rewritten[1]].clone()
    stored[codec.parts(mtp)[1]] = stored[codec.parts(rewritten[1])[1]].clone()
    save_file(stored, src / "model.safetensors", metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (src / "config.json").write_text(json.dumps(cfg))

    graph = describe(m, verify=False)
    writer = Writer(tmp_path / "out", src, graph, "w4a16-sym-g64", Target())
    for ctx in contexts(m, "w4a16-sym-g64"):
        REGISTRY["rtn"]().apply_layer(ctx)
        assert kept.removesuffix(".weight") not in ctx.qparams
        writer.add_layer(ctx)
    writer.add_model(m, {}, changed=False)
    out = writer.finish(m.config, None, {})
    written = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}
    assert "model.layers.0.self_attn.q_proj.weight_packed" in written
    for key in [kept, "model.embed_tokens.weight", "lm_head.weight"]:
        assert written[key].dtype == torch.bfloat16
        assert torch.equal(written[key], expected[key]), key
    assert all(codec.parts(key)[1] not in written for key in rewritten)
    # no step touched the MTP weight, but it is FP8: it is written decoded, and its scale is gone
    assert written[mtp].dtype == torch.bfloat16
    assert torch.equal(written[mtp], expected[rewritten[1]])
    assert codec.parts(mtp)[1] not in written
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]
    assert kept.removesuffix(".weight") in ignore
    assert mtp.removesuffix(".weight") in ignore
    build = json.loads((out / "build.json").read_text())
    assert build["kept_decoded"] == {"dtype": "bfloat16", "modules": [mtp.removesuffix(".weight")]}


def test_actorder_is_recorded(tmp_path):
    from qda_quant import FileSource, Recipe
    from qda_quant.steps.gptq import GPTQ

    src = tmp_path / "src"
    model("qwen3", dtype=torch.bfloat16).save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    w = Writer(tmp_path / "o", src, describe(m, verify=False), "w4a16", Target())
    for steps, want in (([GPTQ()], "static"), ([GPTQ(actorder="none")], None)):
        cfg = w.quantization_config(Recipe(steps=steps, calib=FileSource(path="x")))
        assert cfg["config_groups"]["group_0"]["weights"]["actorder"] == want


@pytest.mark.parametrize(("body", "head"), [("w3a16", "w4a16-sym-g64"), ("w4a16", "w4a16")])
def test_head_in_its_own_format(body, head, tmp_path):
    if body == "w3a16" and not reads_dense_int3():
        pytest.skip("transformers reads through compressed-tensors, which before 0.20 packs int3 another way")
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from qda_quant import Format
    from safetensors import safe_open

    src = tmp_path / "src"
    m = model("qwen3", dtype=torch.bfloat16)
    m.save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    graph = describe(m, verify=False)
    run = SimpleNamespace(qparams={}, target=Target())  # the build's qparams dict and target (BuildRun)
    REGISTRY["head"](format=head).apply_model(m, graph, run)
    w = Writer(tmp_path / "out", src, graph, body, Target())
    for ctx in contexts(m, body):
        REGISTRY["rtn"]().apply_layer(ctx)
        w.add_layer(ctx)
    w.add_model(m, run.qparams)
    out = w.finish(m.config, None, {})
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    hf = Format.parse(head)
    assert qc["config_groups"]["group_head"]["targets"] == ["re:.*lm_head$"]
    assert qc["config_groups"]["group_head"]["weights"]["group_size"] == hf.group_size
    assert "lm_head" not in qc["ignore"]
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(
        out / idx["lm_head.weight_packed"], "pt"
    ) as h:  # independent (q - z) * s of the stored head
        shape = torch.Size(h.get_tensor("lm_head.weight_shape").tolist())
        q = unpack_from_int32(h.get_tensor("lm_head.weight_packed"), hf.weight_bits, shape).float()
        s = h.get_tensor("lm_head.weight_scale").float()
        if not hf.symmetric:
            zp = h.get_tensor("lm_head.weight_zero_point")
            q = q - unpack_from_int32(
                zp, hf.weight_bits, torch.Size((shape[0], s.shape[1])), packed_dim=0
            ).float().repeat_interleave(hf.group_size, -1)
    deq = (q * s.repeat_interleave(hf.group_size, -1)).to(torch.bfloat16)
    assert torch.equal(deq, m.lm_head.weight)
    # transformers keeps the head compressed (no .weight); its logits must equal the served model's
    back = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert (back(input_ids=ids).logits.float() - m(input_ids=ids).logits.float()).abs().max() < 2e-2


def test_shards_get_the_umask_mode(tmp_path):
    """safetensors' save_file creates 0600 files; the writer's shards get the mode every other file gets."""
    import os
    import stat

    mask = os.umask(0o022)
    try:
        _, _, out = _build("qwen3", "w4a16", tmp_path)
    finally:
        os.umask(mask)
    shards = sorted(out.glob("model-*.safetensors"))
    assert shards
    assert {stat.S_IMODE(f.stat().st_mode) for f in shards} == {0o644}
    assert stat.S_IMODE((out / "config.json").stat().st_mode) == 0o644
    assert not list(out.glob(".mode-probe-*"))  # the mode probe leaves nothing behind


def test_source_without_safetensors_is_a_clear_error(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    m = model("qwen3", dtype=torch.bfloat16)
    with pytest.raises(FileNotFoundError, match="no \\*.safetensors"):
        Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16", Target())


def test_processor_config_is_copied(tmp_path):
    """Gemma 4 keeps its image and audio settings in ``processor_config.json``; vLLM cannot start without them."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "processor_config.json").write_text("{}")
    _, _, out = _build("qwen3", "w4a16", tmp_path)
    assert (out / "processor_config.json").read_text() == "{}"


def test_export_keeps_every_source_config_field(tmp_path):
    """The exported config is the source's own JSON plus ``quantization_config``, not one rebuilt from the model.

    A Qwen3.5-MoE checkpoint's config carries ``full_attention_interval``, which transformers 5 does not
    declare (``layer_types`` implies the pattern). A config built from fields, as the tiny fixture was,
    loses it and SGLang then refuses the checkpoint; copying the source JSON keeps it.
    """
    src = tmp_path / "src"
    model("qwen3_5_moe", dtype=torch.bfloat16).save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    source = json.loads((src / "config.json").read_text())
    assert source["full_attention_interval"] == 2
    w = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-g64", Target())
    for ctx in contexts(m, "w4a16-g64"):
        REGISTRY["rtn"]().apply_layer(ctx)
        w.add_layer(ctx)
    out = w.finish(m.config, None, {})
    exported = json.loads((out / "config.json").read_text())
    assert exported.pop("quantization_config")["quant_method"] == "compressed-tensors"
    assert exported == source


def test_float_mtp_expert_bank_is_ignored_under_the_names_engines_probe(tmp_path):
    """An MoE MTP layer stores its routed experts as stacked float tensors the build never quantizes.

    vLLM and SGLang decide a fused MoE module is unquantized by matching ``<bank>.0.gate_proj`` (and up,
    down) against ``ignore``; a bank missing there is built as a packed MoE and fails to load its float
    weights under speculative decoding (Qwen3.6-35B-A3B). The quantized main banks must stay claimed.
    """
    from qda_quant.output.export import _ignore_matches
    from safetensors.torch import save_file

    src = tmp_path / "src"
    model("qwen3_5_moe", dtype=torch.bfloat16).save_pretrained(src)
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    d = m.config.hidden_size
    layer = m.model.layers[m.config.num_hidden_layers - 1]
    mtp = {f"mtp.layers.0.{k}": v.clone() for k, v in layer.state_dict().items()}
    assert {"mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"} <= set(mtp)
    save_file({**mtp, "mtp.fc.weight": torch.zeros(d, 2 * d, dtype=torch.bfloat16)}, src / "mtp.safetensors")
    w = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16-g64", Target())
    for ctx in contexts(m, "w4a16-g64"):
        REGISTRY["rtn"]().apply_layer(ctx)
        w.add_layer(ctx)
    out = w.finish(m.config, None, {})
    ignore = json.loads((out / "config.json").read_text())["quantization_config"]["ignore"]

    def ignored(name):
        return any(_ignore_matches(name, entry) for entry in ignore)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert ignored(f"mtp.layers.0.mlp.experts.0.{proj}")
        assert not any(ignored(f"model.layers.{i}.mlp.experts.0.{proj}") for i in range(len(m.model.layers)))


@pytest.mark.parametrize("name", ["added_tokens.json", "tokenizer.model"])
def test_sentencepiece_and_added_tokens_are_copied(name, tmp_path):
    """Regression against the old export, which copied these too: a sentencepiece tokenizer
    (``tokenizer.model``, Llama 2 / Mistral style) or extra tokens (``added_tokens.json``) live only there."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / name).write_bytes(b"\x00tok")
    _, _, out = _build("qwen3", "w4a16", tmp_path)
    assert (out / name).read_bytes() == b"\x00tok"


def test_float32_source_tensors_keep_their_bytes(tmp_path):
    """Regression against the old export, which copied every tensor it did not change from the source.

    Qwen3.5 / 3.6 / 3.8 store the Gated DeltaNet gated norms in float32; the model runs them in bf16. A
    tensor no step changed must be written as the source stores it, not rounded through the bf16 copy.
    """
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    model("qwen3_5", dtype=torch.bfloat16).save_pretrained(src)
    sd = load_file(src / "model.safetensors")
    g = torch.Generator().manual_seed(0)
    wide = sorted(k for k in sd if k.endswith("norm.weight"))
    for k in wide:  # off the bf16 grid
        sd[k] = torch.rand(sd[k].shape, generator=g, dtype=torch.float32) + 0.5
    save_file(sd, src / "model.safetensors", metadata={"format": "pt"})
    m = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    w = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16", Target())
    for ctx in contexts(m, "w4a16"):
        REGISTRY["rtn"]().apply_layer(ctx)
        w.add_layer(ctx)
    w.add_model(m, {})
    out = w.finish(m.config, None, {})
    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    changed = []
    for k in wide:
        with safe_open(out / idx[k], "pt") as h:
            t = h.get_tensor(k)
        if t.dtype != torch.float32 or not torch.equal(t, sd[k]):
            changed.append(k)
    assert not changed, changed[:3]


def test_export_config_hook_runs_once_on_a_copy_and_keeps_quantization_metadata(monkeypatch, tmp_path):
    """Adapter aliases must preserve source config and the writer's final quantization settings."""
    calls = []

    def normalize(config):
        calls.append(dict(config))
        config["serving_alias"] = "declared"
        config["quantization_config"] = {"format": "source-placeholder"}
        return config

    monkeypatch.setattr(FUSED_EXPERTS, "export_config", normalize, raising=False)
    src, _, out = _build("qwen3_moe", "w4a16-sym-g128", tmp_path)
    source = json.loads((src / "config.json").read_text())
    serving = json.loads((out / "config.json").read_text())
    assert calls == [source]
    assert "serving_alias" not in source
    assert serving["serving_alias"] == "declared"
    assert serving["quantization_config"]["format"] == "pack-quantized"


def test_mlp_override_names_the_fused_gate_up_proj(tmp_path):
    """SGLang resolves the fused ``gate_up_proj`` by its own name: a group naming only gate/up loads as the default.

    A Qwen3.5 config whose override group listed ``mlp.gate_proj``/``up_proj`` but not ``mlp.gate_up_proj`` loaded the
    g32 shards into a g64 module and asserted (9B r2); with the fused name added it loads.
    """
    m = model("qwen3_5", dtype=torch.bfloat16)
    src = tmp_path / "src"
    m.save_pretrained(src)
    graph = describe(m, verify=False)
    formats = {
        grp.modules[0]: Format.parse("w8a16-sym-g64")
        for grp in graph.layers[0].groups
        if grp.kind == "mlp_in"
    }
    w = Writer(tmp_path / "out", src, graph, Format.parse("w4a16-sym-g64"), Target(), formats=formats)
    w.quantized = {name for layer in graph.layers for grp in layer.groups for name in grp.modules}
    ((targets, _),) = w._override_groups()
    for root in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        assert f"{root}.0.mlp.gate_up_proj" in targets
        assert f"{root}.1.mlp.gate_up_proj" not in targets
    w._q.put(None)
    w._t.join()


@pytest.mark.parametrize("packed_source", [False, True])
def test_hybrid_attention_override_names_fused_inputs_and_refuses_split_schemes(tmp_path, packed_source):
    """A serving fusion needs one scheme even when the exporter receives a split override."""
    m = model("qwen3_5_moe", dtype=torch.bfloat16)
    src = tmp_path / "src"
    m.save_pretrained(src)
    graph = describe(m, verify=False)
    if packed_source:
        body = Writer(tmp_path / "body", src, graph, "w4a16-sym-g64", Target())
        for ctx in contexts(m, "w4a16-sym-g64"):
            REGISTRY["rtn"]().apply_layer(ctx)
            body.add_layer(ctx)
        src = body.finish(m.config, None, {})
    fmt = Format.parse("w4a16-sym-g64")
    override = Format.parse("w4a16-asym-g64")
    formats = {
        grp.modules[0]: override
        for layer in graph.layers
        for grp in layer.groups
        if ".linear_attn." in grp.modules[0] or ".self_attn." in grp.modules[0]
    }
    w = Writer(tmp_path / "out", src, graph, fmt, Target(), formats=formats)
    w.quantized = {name for layer in graph.layers for grp in layer.groups for name in grp.modules}
    groups = w._override_groups()
    assert len(groups) == 1
    targets = groups[0][0]
    for root in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        assert f"{root}.0.linear_attn.in_proj_qkvz" in targets
        assert f"{root}.1.self_attn.qkv_proj" in targets
    # A float z shard cannot share this packed runtime module's overridden scheme.
    z = next(name for name in w.quantized if name.endswith(".in_proj_z"))
    w.quantized.remove(z)
    with pytest.raises(Refused, match="fused runtime module.*different schemes"):
        w._override_groups()
    w._q.put(None)
    w._t.join()
