"""CSA2 carry and raw Engram storage on a tiny DeepSeek-V4.1 backbone."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch
from qda_quant.arch import describe
from qda_quant.arch.adapters.deepseek_v41 import DEEPSEEK_V41, SERVING_NAMES
from qda_quant.arch.adapters.deepseek_v41.config import DeepseekV41Config
from qda_quant.arch.adapters.deepseek_v41.numerics import hc_pre
from qda_quant.arch.adapters.deepseek_v41.prefill import SM120_REFUSAL, Sm90Prefill
from qda_quant.arch.adapters.prefill import PREFILL_NUMERICS
from qda_quant.arch.derive import served_name
from qda_quant.config import BuildConfig
from qda_quant.engine.execute import Timers
from qda_quant.engine.runner import Pass, capture
from qda_quant.step import LayerContext, Refused
from qda_quant.target import Target
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


@pytest.fixture
def sm90_card() -> None:
    """Skips off compute capability 9.0: vLLM 0.33 serves V4.1 through DeepGEMM and FlashMLA on Hopper (QDA's
    prefill numerics are its sm90 path); on an RTX PRO 6000 with QDA's refusal of V4.1 there; on an A100 its
    TileLang mHC fallback asserts on the tiny model's width before the engine starts. Read in the test: asking
    for the capability at collection initializes CUDA."""
    capability = torch.cuda.get_device_capability()
    if capability == (12, 0):
        pytest.skip(SM120_REFUSAL)
    if capability != (9, 0):
        pytest.skip("V4.1 in vLLM 0.33 is compared on its sm90 path (compute capability 9.0)")


def tiny_v41():
    DEEPSEEK_V41.register_models()
    config = DeepseekV41Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=128,
            moe_intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=128,
            o_lora_rank=64,
            o_groups=2,
            n_routed_experts=8,
            num_experts_per_tok=2,
            compress_ratios=[2, 2, 1, 1],
            kv_source_layer_ids=[0, 2],
            index_source_layer_ids=[0, 2, 3],
            index_n_heads=32,
            index_head_dim=128,
            index_topk=4,
            candidate_source_layer_id=2,
            candidate_topk_blocks=3,
            candidate_block_size=2,
            engram_layer_ids=[1],
            engram_num_embeddings=[64],
            engram_max_ngram_size=3,
            engram_n_heads=2,
            engram_vocab_size=11,
            engram_head_dim=32,
            engram_compressed_vocab_size=128,
            sliding_window=4,
            max_position_embeddings=64,
            rope_scaling={"rope_type": "default"},
        )
    )
    torch.manual_seed(5)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("hc_attn_fn", "hc_ffn_fn")):
                p.normal_(std=0.03)
            elif name.endswith(("hc_attn_base", "hc_ffn_base", "gate.bias")):
                p.normal_(std=0.1)
        model.model.layers[1].engram.embed_tokens.weight_scale.view(torch.uint8).random_(125, 130)
    DEEPSEEK_V41.configure_model(model, Target("sm90", "0.33.0"))
    return model


def save_tokenizer(path):
    backend = Tokenizer(
        models.WordLevel({f"w{i}": i for i in range(127)} | {"[UNK]": 127}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(path)


def test_v41_fused_down_hessians_match_native_per_expert_inputs():
    import copy

    from qda_quant.arch.adapters.deepseek_v41.model import Expert
    from qda_quant.engine.stats import LayerStats, observe_experts

    model = tiny_v41()
    layer = model.model.layers[0]
    cfg = copy.deepcopy(model.config.text_config)
    cfg.swiglu_limit = layer.ffn.experts.limit = 0.1
    graph = describe(model, verify=False)
    lg = graph.layers[0]
    x = (torch.randn(19, cfg.hidden_size) * 30).to(torch.bfloat16)
    weights, indices = layer.ffn.gate(x)
    stats = LayerStats(layer, lg, frozenset({"expert_hessian"}), torch.device("cpu"))
    bank_stats = stats.banks[lg.banks[0].router]
    expected = []
    with torch.no_grad():
        observe_experts(layer.ffn.experts, x, indices, weights, bank_stats)
        for e in range(cfg.n_routed_experts):
            expert = Expert(cfg).to(torch.bfloat16)
            gate, up = layer.ffn.experts.gate_up_proj[e].chunk(2)
            expert.w1.weight.copy_(gate)
            expert.w3.weight.copy_(up)
            expert.w2.weight.copy_(layer.ffn.experts.down_proj[e])
            seen = []
            handle = expert.w2.register_forward_pre_hook(
                lambda _m, args, seen=seen: seen.append(args[0].clone())
            )
            rows, picks = torch.where(indices == e)
            expert(x[rows], weights[rows, picks, None])
            h = seen.pop().float()
            expected.append(h.T @ h)
            expert(x)
            gate_up = torch.cat((expert.w1(x), expert.w3(x)), -1)
            torch.testing.assert_close(
                layer.ffn.experts.down_input(gate_up, None), seen.pop(), rtol=0, atol=0
            )
            handle.remove()
    actual = bank_stats.result()["hessian"]["down"]
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)


def test_v41_layer_pass_matches_whole_model_and_verifies_graph():
    model = tiny_v41()
    graph = describe(model, verify=True)
    assert graph.adapter == "deepseek_v41" and graph.verified and not graph.extra["dropped"]
    # wq_a / wkv stay float: wq_b and wo_b are the quantized attention groups
    assert all(len(layer.groups) == 2 and layer.banks[0].fused for layer in graph.layers)
    assert {f"{layer.module}.attn.{p}" for layer in graph.layers for p in ("wq_a", "wkv")} == set(
        graph.extra["kept_bf16"]["attention_input"]
    )
    # vLLM's native multimodal class matches the fused module literally (no packed mapping, no name mapper)
    vllm = graph.extra["runtime_names"]["vllm"]
    for i in range(len(graph.layers)):
        assert {
            f"language_model.model.layers.{i}.attn.fused_wqa_wkv",
            f"model.layers.{i}.attn.fused_wqa_wkv",
        } <= set(vllm)
    for layer in graph.layers:
        bank = model.get_submodule(layer.banks[0].experts)
        assert bank.gate_up_proj.shape == (8, 256, 128) and bank.down_proj.shape == (8, 128, 128)
    batches = [torch.arange(1, 10)[None], torch.arange(10, 1, -1)[None]]
    cpu = torch.device("cpu")
    slots = capture(model, graph, batches, cpu)
    with torch.no_grad():
        Pass(
            model,
            graph,
            [],
            cpu,
            make_ctx=lambda layer, lg, st: LayerContext(layer, lg, graph, None, None, None, st, device=cpu),
            timers=Timers(),
        ).run(slots)
        for ids, slot in zip(batches, slots, strict=True):
            logits = model.lm_head(model.model.norm(hc_pre(slot.h, slot.carry["carry"]["pre_mix"]))).float()
            torch.testing.assert_close(logits, model(ids).logits, rtol=0, atol=0)
    # Future target plugins select reference numerics without a model-config switch.
    DEEPSEEK_V41.configure_model(model, Target("reference", "0"))
    assert model.prefill_numerics == "deepseek_v41_reference"
    assert "prefill_codec" not in model.config.text_config.to_dict()


@pytest.mark.parametrize(
    "residency,layout,ignore",
    [
        ("resident", "hf", None),
        ("disk", "hf", None),
        ("resident", "native", None),
        ("disk", "native", None),
        ("disk", "native", []),
    ],
)
def test_v41_rtn_preserves_encoded_engram_and_records_target(tmp_path, residency, layout, ignore):
    from qda_quant.intake.source_names import SourceNames

    model = tiny_v41()
    source = tmp_path / "source"
    model.save_pretrained(source, save_original_format=layout == "native")
    if layout == "native":
        stored = load_file(source / "model.safetensors")
        assert {"embed.weight", "head.weight", "layers.1.engram.embed.scale"} <= stored.keys()
    save_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    out = tmp_path / "out"
    cfg = BuildConfig(
        model=str(source),
        format="w4a16-sym-g32",
        target="sm90@0.33.0",
        devices=["cpu"],
        residency=residency,
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(out),
    )
    if ignore is not None:
        cfg.ignore = ignore
    cfg.build()
    names = SourceNames(source)
    written = {k: v for shard in out.glob("*.safetensors") for k, v in load_file(shard).items()}

    def stored(name):
        """The exported key for model tensor `name`: the source's name, then the serving names."""
        return served_name(names.tensor(name), SERVING_NAMES)

    for leaf in ("embed_tokens.weight", "embed_tokens.weight_scale", "wkv.weight"):
        name = "model.layers.1.engram." + leaf
        torch.testing.assert_close(
            written[stored(name)].view(torch.uint8),
            model.state_dict()[name].view(torch.uint8),
            rtol=0,
            atol=0,
        )
    assert written[stored("model.layers.1.engram.embed_tokens.weight")].dtype == torch.float8_e4m3fn
    assert written[stored("model.layers.1.engram.embed_tokens.weight_scale")].dtype == torch.float8_e8m0fnu
    assert written[stored("model.layers.1.engram.wkv.weight")].dtype == torch.bfloat16
    quant = json.loads((out / "config.json").read_text())["quantization_config"]
    for i in range(model.config.text_config.num_hidden_layers):
        for projection in ("gate_up_proj", "down_proj"):
            # vLLM's VL text model keeps ``ffn``; SGLang remaps it to ``mlp`` and matches literally.
            for alias in (
                f"language_model.model.layers.{i}.ffn.shared_experts.{projection}",
                f"model.layers.{i}.mlp.shared_experts.{projection}",
            ):
                assert (alias in quant["ignore"]) == (ignore is None)
    # SGLang quantizes an indexer's ``wq_b`` unless its runtime name is ignored; the adapter
    # always keeps the indexer float, whatever roles the build keeps.
    assert "model.layers.0.self_attn.indexer.wq_b" in quant["ignore"]
    import yaml

    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["target"] == "sm90@0.33.0"
    assert resolved["provenance"]["plugins"]["deepseek_v41_sm90"]["group"] == "qda.prefill_numerics"


@pytest.mark.parametrize("layout", ["hf", "native"])
def test_v41_exports_the_names_stock_engines_load(tmp_path, layout):
    """A source in either layout is written under the native names vLLM 0.33 and SGLang 0.5.23 remap.

    The transformer-5 shadow root and fused routed banks are QDA-side shapes: stock vLLM's VL mapper turns
    ``head.weight`` into its own head but cannot undo ``lm_head.weight``, and both engines probe per-expert
    ``w1``/``w3``/``w2``. Engram's table is ``engram.embed`` (``scale``, not ``weight_scale``) natively.
    """
    model = tiny_v41()
    source = tmp_path / "source"
    model.save_pretrained(source, save_original_format=layout == "native")
    save_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    out = tmp_path / "out"
    BuildConfig(
        model=str(source),
        format="w4a16-sym-g128",
        target="sm90@0.33.0",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(out),
    ).build()
    keys = {k for shard in out.glob("*.safetensors") for k in load_file(shard)}
    assert {"embed.weight", "head.weight", "norm.weight", "layers.1.engram.embed.weight"} <= keys
    assert "layers.1.engram.embed.scale" in keys
    for projection in ("w1", "w2", "w3"):
        assert f"layers.0.ffn.experts.0.{projection}.weight_packed" in keys
    assert not any(k.startswith(("model.", "lm_head")) for k in keys)
    assert not any(k.endswith((".experts.gate_up_proj", ".experts.down_proj")) for k in keys)
    rope = json.loads((out / "config.json").read_text())["text_config"]["rope_parameters"]
    assert rope == {"rope_type": "default", "original_max_position_embeddings": 64}


def test_v41_disk_plan_fits_when_the_lookup_table_exceeds_card_memory(tmp_path, monkeypatch):
    from qda_quant.engine import residency

    model = tiny_v41()
    lookup = model.model.layers[1].engram.embed_tokens
    rows, width = 1 << 18, lookup.weight.shape[1]
    model.config.text_config.engram_num_embeddings = [rows]
    lookup.weight = torch.nn.Parameter(
        torch.zeros(rows, width, dtype=lookup.weight.dtype), requires_grad=False
    )
    lookup.weight_scale = torch.ones(rows, (width + 31) // 32, dtype=lookup.weight_scale.dtype)
    source = tmp_path / "source"
    model.save_pretrained(source, save_original_format=False)
    save_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4"}) + "\n")
    loaded = [sum(p.numel() for p in layer.parameters()) * 2 for layer in model.model.layers]
    kept = loaded.copy()
    kept[1] -= lookup.weight.numel() * 2
    free = (max(loaded) + max(kept)) // 2
    monkeypatch.setattr(residency, "free_bytes", lambda _device: free)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    plan = BuildConfig(
        model=str(source),
        format="w4a16-sym-g32",
        target="sm90@0.33.0",
        devices=["cuda:0"],
        residency="disk",
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 4}},
    ).plan()
    assert max(kept) < free < max(loaded)
    assert plan.host_parameter_bytes["model.layers.1"] == lookup.weight.numel() * 2
    assert plan.residency.weights == "disk"
    assert plan.residency.peak_gib == round(max(kept) / residency.GiB, 2)


def test_v41_rank_load_registers_models_in_a_fresh_process(tmp_path):
    source = tmp_path / "source"
    tiny_v41().save_pretrained(source, save_original_format=False)
    code = dedent("""
        import sys
        from types import SimpleNamespace
        import torch
        from transformers import AutoConfig
        from qda_quant.engine.stores import STREAM
        from qda_quant.target import Target

        try:
            AutoConfig.for_model("deepseek_v41")
        except ValueError:
            pass
        else:
            raise AssertionError("model registered at import time")
        job = SimpleNamespace(
            model=sys.argv[1], dtype=torch.bfloat16, weights=None,
            target=Target("sm90", "0.33.0"),
            plan=SimpleNamespace(store=STREAM),
        )
        model = job.plan.store.load(job, torch.device("cpu"))
        assert type(model).__name__ == "DeepseekV41ForCausalLM"
        assert model.prefill_numerics == "deepseek_v41_sm90"
        print("fresh rank load passed; no import-time registration")
    """)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_v41_plan_selects_extension_numerics_and_refuses_missing_or_ambiguous(tmp_path, monkeypatch):
    source = tmp_path / "source"
    tiny_v41().save_pretrained(source, save_original_format=False)
    save_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    cfg = BuildConfig(
        model=str(source),
        format="w4a16-sym-g32",
        target="sm90@0.33.1",
        devices=["cpu"],
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
    )
    with pytest.raises(Refused, match="no matching plugin"):
        cfg.plan()
    PREFILL_NUMERICS.items()  # load entry points before isolating the test's registrations
    monkeypatch.setattr(PREFILL_NUMERICS, "_entries", dict(PREFILL_NUMERICS._entries))

    class ExtensionPrefill(Sm90Prefill):
        def supports(self, target):
            return (target.arch, target.vllm) == ("sm90", "0.33.1")

    PREFILL_NUMERICS.register("extension_prefill", ExtensionPrefill())
    plan = cfg.plan()
    assert "extension_prefill" in cfg.plugins(adapters=(plan.graph.adapter,))
    PREFILL_NUMERICS.register("conflicting_prefill", ExtensionPrefill())
    with pytest.raises(Refused, match="multiple matching plugins:.*conflicting_prefill.*extension_prefill"):
        cfg.plan()


def test_v41_for_sm120_is_refused_at_plan_time(tmp_path, capsys):
    """`qda plan` and `qda build` of V4.1 for an RTX PRO 6000 stop in the plan with `SM120_REFUSAL`."""
    import yaml
    from qda_quant.cli import main

    source = tmp_path / "source"
    tiny_v41().save_pretrained(source, save_original_format=False)
    save_tokenizer(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    recipe = tmp_path / "v41.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "model": str(source),
                "format": "w4a16-sym-g32",
                "target": "rtx6000",
                "devices": ["cpu"],
                "steps": [{"rtn": {}}],
                "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
                "out": str(tmp_path / "out"),
            }
        )
    )
    for command in ("plan", "build"):
        assert main([command, str(recipe)]) == 2, command
        assert f"error: {SM120_REFUSAL}\n" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _native_expert_tensors(name, tensor):
    if name.endswith(".experts.gate_up_proj"):
        stem = name.removesuffix(".gate_up_proj")
        return {
            f"{stem}.{e}.{projection}.weight": weight.contiguous()
            for e, expert in enumerate(tensor)
            for projection, weight in zip(("w1", "w3"), expert.chunk(2), strict=True)
        }
    if name.endswith(".experts.down_proj"):
        stem = name.removesuffix(".down_proj")
        return {f"{stem}.{e}.w2.weight": weight.contiguous() for e, weight in enumerate(tensor)}
    return {name: tensor.contiguous()}


def _native_checkpoint(model, path):
    path.mkdir()
    save_tokenizer(path)
    cfg = model.config.to_dict()
    cfg["architectures"] = ["DeepseekV41ForCausalLM"]
    text = cfg["text_config"]
    text["rope_scaling"] = text.pop("rope_parameters")
    (path / "config.json").write_text(json.dumps(cfg))
    weights = {}
    for name, tensor in model.state_dict().items():
        name = name.replace("model.layers.", "layers.").replace("model.norm.", "norm.")
        name = name.replace("model.embed_tokens.", "embed.").replace("lm_head.", "head.")
        name = name.replace(".engram.embed_tokens.weight_scale", ".engram.embed.scale")
        name = name.replace(".engram.embed_tokens.weight", ".engram.embed.weight")
        if name.endswith(".engram.embed.scale"):
            tensor = tensor.view(torch.float8_e8m0fnu)
        weights.update(_native_expert_tensors(name, tensor))
    # vLLM's multimodal wrapper requires delimiter parameters even for text-only runs.
    for name in ("image_start", "image_end", "image_newline"):
        weights[name] = torch.zeros(model.config.text_config.hidden_size, dtype=torch.bfloat16)
    save_file(weights, str(path / "model.safetensors"))


def _record_logits(model):
    def hook(module, _args, out):
        module._qda_test_logits = module.compute_logits(out).detach().cpu()

    model.register_forward_hook(hook)


def _record_index_picks(model):
    """Keep each index-source layer's top-k picks; layers share one buffer that the next indexer overwrites."""
    model._qda_test_picks = {}

    def hook(module, args, _out):
        picks = module.topk_indices_buffer[: args[0].shape[0]]
        model._qda_test_picks[module.layer_id] = picks.detach().cpu().clone()

    for module in model.modules():
        if getattr(module, "is_index_source", False) and getattr(module, "topk_indices_buffer", None) is not None:
            module.register_forward_hook(hook)


# vLLM rounds index q and k to FP8 e4m3 (up to 2^-4 relative error each, so up to ~2^-3 per q.k product); it may
# swap two picks whose index scores are closer than this fraction of the k-th score. GH200, ids 1..9: one swap at 5%.
FP8_INDEX_TIE = 2**-3


def _forced_index_picks(monkeypatch, served):
    """Run our indexers on vLLM's picks; return (layer, position, relative gap) for each pick set that differed."""
    from qda_quant.arch.adapters.deepseek_v41.attention import Indexer

    forward, topk, ties = Indexer.forward, torch.Tensor.topk, []

    def forced(self, x, qr, latent, positions, ratio, carry, layer_id):
        seen = []

        def spy(scores, *args, **kwargs):
            seen.append(scores)
            return topk(scores, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.Tensor, "topk", spy)
            forward(self, x, qr, latent, positions, ratio, carry, layer_id)
        scores, ours = seen[-1][0].float(), carry["indices"][0]
        theirs = served[layer_id][: ours.shape[0], : ours.shape[1]].to(ours)
        for pos in range(ours.shape[0]):
            mine, other = set(ours[pos].tolist()) - {-1}, set(theirs[pos].tolist()) - {-1}
            if mine == other:
                continue
            row = scores[pos]
            kth = row[ours[pos][ours[pos] >= 0]].min()
            gap = float((row[sorted(mine - other)].max() - row[sorted(other - mine)].min()) / kth.abs())
            ties.append((layer_id, pos, gap))
        carry["indices"] = theirs.sort(-1).values.unsqueeze(0)

    monkeypatch.setattr(Indexer, "forward", forced)
    return ties


def _native_dspark_checkpoint(model, path):
    from qda_quant.arch.adapters.deepseek_v41.model import DecoderLayer

    cfg = model.config.text_config
    # FlashMLA decode kernels require aligned SWA and sparse-index widths.
    cfg.sliding_window = 128
    cfg.index_topk = 128
    cfg.candidate_topk_blocks = 65
    cfg.num_nextn_predict_layers = 3
    cfg.dspark_block_size = 3
    cfg.dspark_noise_token_id = cfg.vocab_size - 1
    cfg.dspark_target_layer_ids = list(range(cfg.num_hidden_layers - 3, cfg.num_hidden_layers))
    cfg.dspark_markov_rank = 32
    cfg.dspark_n_routed_experts = cfg.n_routed_experts
    cfg.dspark_num_experts_per_tok = cfg.num_experts_per_tok
    cfg.compress_ratios += [0] * cfg.num_nextn_predict_layers
    _native_checkpoint(model, path)
    weights = load_file(path / "model.safetensors")
    for i in range(cfg.num_nextn_predict_layers):
        layer = DecoderLayer(cfg, cfg.num_hidden_layers + i).to(dtype=torch.bfloat16)
        layer.apply(model._init_weights)
        for name, value in layer.state_dict().items():
            if name.startswith("hc_") or name.endswith(("attn_sink", "gate.bias")):
                value = value.float()
            weights.update(_native_expert_tensors(f"mtp.{i}.{name}", value))
    width, vocab, rank = cfg.hidden_size, cfg.vocab_size, cfg.dspark_markov_rank
    head = f"mtp.{cfg.num_nextn_predict_layers - 1}"
    weights.update(
        {
            "mtp.0.main_proj.weight": torch.randn(
                width, width * len(cfg.dspark_target_layer_ids), dtype=torch.bfloat16
            )
            * cfg.initializer_range,
            "mtp.0.main_norm.weight": torch.ones(width, dtype=torch.bfloat16),
            f"{head}.norm.weight": torch.ones(width, dtype=torch.bfloat16),
            f"{head}.markov_head.embed.weight": torch.randn(vocab, rank, dtype=torch.bfloat16)
            * cfg.initializer_range,
            f"{head}.markov_head.head.weight": torch.randn(vocab, rank, dtype=torch.bfloat16)
            * cfg.initializer_range,
            f"{head}.confidence_head.proj.weight": torch.randn(1, width + rank) * cfg.initializer_range,
        }
    )
    save_file(weights, str(path / "model.safetensors"))


@pytest.mark.gpu
def test_v41_gpu_lookup_keeps_disk_tables_on_host(tmp_path):
    from qda_quant.engine.disk import DISK

    model = tiny_v41()
    source = tmp_path / "native"
    _native_checkpoint(model, source)
    graph = describe(model)
    device = torch.device("cuda")
    job = SimpleNamespace(
        model=str(source),
        dtype=torch.bfloat16,
        target=Target("sm90", "0.33.0"),
        plan=SimpleNamespace(store=DISK, graph=graph),
    )
    loaded = job.plan.store.load(job, device)
    layer = loaded.model.layers[1]
    DISK.place(layer, graph.layers[1], device)
    try:
        lookup = layer.engram.embed_tokens
        assert lookup.weight.device.type == lookup.weight_scale.device.type == "cpu"
        assert layer.engram.wkv.weight.device.type == "cuda"
        rows = torch.tensor([0, 3, 3, 59], device=device)
        expected = model.model.layers[1].engram.embed_tokens(rows.cpu())
        actual = lookup(rows)
        assert actual.device.type == "cuda"
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        model.to(device)
        assert model.model.layers[1].engram.embed_tokens.weight.device.type == "cpu"
    finally:
        DISK.release(layer)


@pytest.mark.gpu
@pytest.mark.usefixtures("sm90_card")
def test_v41_prefill_logits_match_vllm_033(tmp_path, monkeypatch):
    vllm = pytest.importorskip("vllm")
    if not vllm.__version__.startswith("0.33."):
        pytest.skip("parity reference is vLLM 0.33")
    # Only trusted local functions cross IPC to install/read the raw-logit hook.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model = tiny_v41()
    source = tmp_path / "native"
    _native_checkpoint(model, source)
    llm = vllm.LLM(
        model=str(source),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        kv_cache_memory_bytes=64 << 20,
        enable_prefix_caching=False,
        swa_bounded_replay=False,
        limit_mm_per_prompt={"image": 0},
        kernel_config={"enable_jit_warmup": False},
    )
    llm.apply_model(_record_logits)
    llm.apply_model(_record_index_picks)
    ids = list(range(1, 10))
    llm.generate([{"prompt_token_ids": ids}], vllm.SamplingParams(temperature=0, max_tokens=1))
    (reference,) = llm.apply_model(lambda m: m._qda_test_logits)
    (picks,) = llm.apply_model(lambda m: m._qda_test_picks)
    assert sorted(picks) == model.config.text_config.index_source_layer_ids
    ties = _forced_index_picks(monkeypatch, picks)
    with torch.no_grad():
        actual = model(torch.tensor([ids])).logits[0]
    reference = reference[: len(ids)].float()
    torch.save({"actual": actual, "reference": reference, "ties": ties}, tmp_path / "prefill-logits.pt")
    delta = (actual - reference).abs()
    print(f"prefill logits: max_abs={delta.max().item():.6g}, mean_abs={delta.mean().item():.6g}; index ties {ties}")
    # Any other pick difference is an indexer bug; near-ties then run on vLLM's picks so the logits stay comparable.
    assert all(gap <= FP8_INDEX_TIE for *_, gap in ties), ties
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)


@pytest.mark.gpu
@pytest.mark.usefixtures("sm90_card")
def test_v41_w4a16_dspark_preserves_drafts_and_greedy_tokens(tmp_path):
    vllm = pytest.importorskip("vllm")
    model = tiny_v41()
    source, out = tmp_path / "native", tmp_path / "packed"
    _native_dspark_checkpoint(model, source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "w1 w2 w3 w4 w5 w6 w7 w8"}) + "\n")
    BuildConfig(
        model=str(source),
        format="w4a16-sym-g32",
        target="sm90@0.33.0",
        devices=["cpu"],
        residency="disk",
        steps=[{"rtn": {}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        out=str(out),
    ).build()
    stored = load_file(source / "model.safetensors")
    packed = {k: v for f in out.glob("*.safetensors") for k, v in load_file(f).items()}
    for name, value in stored.items():
        if name.startswith("mtp."):
            torch.testing.assert_close(packed[name], value, rtol=0, atol=0)
    config = json.loads((out / "config.json").read_text())
    config.pop("quantization_config")
    assert config == json.loads((source / "config.json").read_text())
    serving = dict(
        model=str(out),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        kv_cache_memory_bytes=64 << 20,
        enable_prefix_caching=False,
        swa_bounded_replay=False,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0},
        kernel_config={"enable_jit_warmup": False},
    )
    prompts = [{"prompt_token_ids": list(range(1, 10))}]
    sampling = vllm.SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    draft = vllm.LLM(**serving, speculative_config={"method": "dspark", "num_speculative_tokens": 2})
    try:
        speculative = draft.generate(prompts, sampling)[0].outputs[0].token_ids
        metrics = {m.name: m.value for m in draft.get_metrics() if hasattr(m, "value")}
        assert metrics["vllm:spec_decode_num_draft_tokens"] > 0
    finally:
        draft.llm_engine.engine_core.shutdown()
    ordinary = vllm.LLM(**serving)
    try:
        reference = ordinary.generate(prompts, sampling)[0].outputs[0].token_ids
    finally:
        ordinary.llm_engine.engine_core.shutdown()
    assert speculative == reference
    print(f"DSpark greedy tokens match; metrics: {metrics}")
