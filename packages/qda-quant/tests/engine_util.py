"""Tiny random models and hand-built graphs for the engine tests (until `arch.describe` lands)."""

from __future__ import annotations

import importlib.metadata
from typing import Any, ClassVar

import torch
from pydantic import PrivateAttr
from qda_quant.arch import ExpertBank, LayerGraph, LinearGroup, ModelGraph
from qda_quant.output.exporters import EXPORTERS, Exporter
from qda_quant.step import LayerContext, Step
from transformers import AutoModelForCausalLM, GptOssConfig, Qwen3Config, Qwen3MoeConfig


def tiny_qwen3(layers: int = 3, seed: int = 0, hidden: int = 64, inter: int = 96) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 16,
        num_key_value_heads=hidden // 32,
        head_dim=16,
        max_position_embeddings=256,
    )
    cfg._attn_implementation = "sdpa"
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()


def tiny_qwen3_moe(layers: int = 2, seed: int = 0, hidden: int = 128, inter: int = 32) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=96,
        moe_intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 16,
        num_key_value_heads=hidden // 32,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    cfg._attn_implementation = "sdpa"
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    with torch.no_grad():  # from_config leaves the 3-D expert params at their (empty) init on some versions
        for layer in m.model.layers:
            ex = layer.mlp.experts
            ex.gate_up_proj.normal_(0, 0.1)
            ex.down_proj.normal_(0, 0.1)
            layer.mlp.gate.weight.normal_(0, 0.5)
    return m


def tiny_gpt_oss(layers: int = 1, seed: int = 0, hidden: int = 128, inter: int = 128) -> torch.nn.Module:
    """gpt-oss stores its expert bank transposed ([E, in, out]), interleaved (gate0, up0, ...) and with biases."""
    torch.manual_seed(seed)
    cfg = GptOssConfig(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 16,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )
    cfg._attn_implementation = "eager"
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    with torch.no_grad():  # from_config leaves the expert parameters at their (empty) init
        for layer in m.model.layers:
            ex = layer.mlp.experts
            ex.gate_up_proj.normal_(0, 0.1)
            ex.down_proj.normal_(0, 0.1)
            ex.gate_up_proj_bias.normal_(0, 0.1)
            ex.down_proj_bias.normal_(0, 0.1)
            layer.mlp.router.weight.normal_(0, 0.5)
    return m


def graph_of(model: torch.nn.Module) -> ModelGraph:
    layers = []
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        groups = [
            LinearGroup(
                (f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"), "attn_in", norm=f"{p}.input_layernorm"
            ),
            LinearGroup((f"{a}.o_proj",), "attn_out", prev=f"{a}.v_proj"),
        ]
        banks = ()
        if hasattr(layer.mlp, "experts"):
            banks = (ExpertBank(f"{p}.mlp.experts", f"{p}.mlp.gate", layer.mlp.gate.top_k, fused=True),)
        else:
            m = f"{p}.mlp"
            groups += [
                LinearGroup(
                    (f"{m}.gate_proj", f"{m}.up_proj"), "mlp_in", norm=f"{p}.post_attention_layernorm"
                ),
                LinearGroup((f"{m}.down_proj",), "mlp_out", prev=f"{m}.up_proj"),
            ]
        layers.append(LayerGraph(i, p, tuple(groups), banks))
    return ModelGraph(model.config.model_type, tuple(layers), "model.embed_tokens", "lm_head", "model.norm")


def rtn_(w: torch.Tensor, bits: int = 4, group: int = 16) -> torch.Tensor:
    """Symmetric per-group RTN of a [..., out, in] weight, in place; returns it."""
    shp = w.shape
    g = w.reshape(*shp[:-1], shp[-1] // group, group)
    q = (1 << (bits - 1)) - 1
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    w.copy_((torch.round(g / s).clamp(-q - 1, q) * s).reshape(shp))
    return w


class StatDump(Step):
    """Saves every tensor statistic of each layer to ``{dump}/stats-{layer}.pt`` (rank 0 sees the merge)."""

    name: ClassVar[str] = "stat_dump"
    dump: str
    want: tuple[str, ...] = ("hessian", "expert_hessian", "act_absmax", "inputs")
    input_rows: int | None = 8192

    @property
    def needs(self):
        return frozenset(self.want)

    def apply_layer(self, ctx: LayerContext) -> None:
        import pathlib

        keep = {}
        for (n, k), v in ctx.stats.items():
            if n in ("layer_io", "outputs"):
                continue
            keep[f"{n}|{k}"] = (
                {a: b.cpu() for a, b in v.items()}
                if isinstance(v, dict)
                else (v.cpu() if isinstance(v, torch.Tensor) else v)
            )
        torch.save(keep, pathlib.Path(self.dump) / f"stats-{ctx.graph.index}.pt")


class TensorWriter:
    """Writer that saves each solved layer's tensors (for card-count parity of the solved weights)."""

    def __init__(self, out_dir, source, graph, fmt, target, *, dtype=None):  # noqa: ARG002
        import pathlib

        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def add_layer(self, ctx):
        torch.save(
            {k: v.cpu() for k, v in ctx.layer.state_dict().items()}, self.out / f"layer-{ctx.graph.index}.pt"
        )

    def finish(self, config, recipe, build_info):
        import json

        (self.out / "build.json").write_text(json.dumps(build_info))
        return self.out


def save_tiny_tokenizer(model_dir) -> str:
    """A 128-token WordLevel tokenizer (with a chat template) saved next to a tiny model; returns its fingerprint."""
    from qda_quant.engine.calib import fingerprint
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(127)} | {"[UNK]": 127}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}w1 {{ m['content'] }} w2 {% endfor %}"
    fast.save_pretrained(model_dir)
    return fingerprint(model_dir)


def write_ids(path, rows, tokenizer_sha) -> None:
    """A token-id calibration file with its tokenizer fingerprint header."""
    import json

    lines = [json.dumps({"tokenizer": tokenizer_sha})] + [json.dumps({"input_ids": r}) for r in rows]
    path.write_text("\n".join(lines))


def reads_dense_int3() -> bool:
    """Whether transformers can read an int3 checkpoint here: compressed-tensors before 0.20 packs 3 bits another way."""
    version = importlib.metadata.version("compressed-tensors").split(".")[:2]
    return tuple(int(x) for x in version) >= (0, 20)


class WriterExporter(Exporter):
    """A test's writer class as an exporter (`writes`); recorded as ``test_writer`` with no settings."""

    _writer: Any = PrivateAttr(default=None)

    def writer(self, out_dir, source, graph, fmt, target, *, dtype=torch.bfloat16):  # noqa: ANN001, ANN201
        return self._writer(out_dir, source, graph, fmt, target, dtype=dtype)


EXPORTERS.register("test_writer", WriterExporter)


def writes(writer_cls: Any) -> WriterExporter:
    """``build(..., exporter=writes(FakeWriter))``: the exporter that makes `writer_cls` the build's writer."""
    e = WriterExporter()
    e._writer = writer_cls
    return e
