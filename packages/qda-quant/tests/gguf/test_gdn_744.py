"""A GPTQ build of a gated-delta-net model must carry its solved codes through the converter's reorder.

The pinned converter reorders a GDN layer's value heads. On ``out_proj`` that reorder lands on the COLUMNS, so
the template is not a row permutation of the source and `row_order` refuses, which fails any build whose codes
reach the exporter. Plain ``rtn`` never gets here: it takes `GGUFWriter.writes_alone` and
quantizes every tensor from the template, which is why the byte-identity proofs for this
architecture did not cover it. A test for this has to run a method that solves.

The template is written here rather than converted, so the test needs no llama.cpp: it applies
the one transform under test (`v_head_columns`) and keeps every other tensor in the source's own
order, which the exporter matches by bytes.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import transformers
from qda_quant import REGISTRY, Recipe
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.gguf import file
from qda_quant.gguf.columns import template_columns
from qda_quant.gguf.export import GGUFWriter
from qda_quant.gguf.kernel import q4k_fields
from qda_quant.step import Step

from ..engine_util import save_tiny_tokenizer

#: Value heads outnumber key heads, which is the only case the converter reorders (and the 27B's: 16 and 48).
#: ``out_proj``'s rows are ``value heads x head dim`` and must be a whole 256-value super-block, or the rule
#: falls the tensor back to Q5_0, the layer pass skips it and the reorder is never exercised.
GDN = dict(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=256,
    tie_word_embeddings=False,
)


def v_head_columns(num_k_heads: int, num_v_heads: int, head_dim: int) -> np.ndarray:
    """``_LinearAttentionVReorderBase._reorder_v_heads`` as a column index map: grouped order -> tiled order."""
    per_k = num_v_heads // num_k_heads
    heads = [g * per_k + v for v in range(per_k) for g in range(num_k_heads)]
    return np.concatenate([np.arange(h * head_dim, (h + 1) * head_dim) for h in heads])


def _template(src: Path, out: Path, cfg, *, nextn: int = 0) -> None:  # noqa: ANN001
    """A bf16 GGUF of `src` with the converter's column reorder on ``out_proj`` and nothing else moved.

    `nextn` adds that many head blocks after the decoder layers, as the converter stores an MTP head.
    """
    import gguf
    from safetensors.torch import load_file

    cols = torch.from_numpy(
        v_head_columns(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    )
    gdn = {
        "in_proj_qkv": "attn_qkv",
        "in_proj_z": "attn_gate",
        "in_proj_a": "ssm_alpha",
        "in_proj_b": "ssm_beta",
        "out_proj": "ssm_out",
        "norm": "ssm_norm",
    }
    w = gguf.GGUFWriter(str(out), "qwen35")
    w.add_block_count(cfg.num_hidden_layers + nextn)
    if nextn:
        w.add_nextn_predict_layers(nextn)
    for key, t in load_file(str(src / "model.safetensors")).items():
        name = _gguf_name(key, gdn)
        if name is None:
            continue
        if key.endswith("linear_attn.out_proj.weight"):
            t = t.index_select(1, cols)  # the converter's reorder, on the input columns
        if key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
            t = t.float() + 1  # the converter's HF-to-ggml RMSNorm storage convention
        if t.dtype == torch.bfloat16:
            w.add_tensor(name, t.view(torch.int16).numpy(), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, t.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    for b in range(nextn):
        blk = cfg.num_hidden_layers + b
        w.add_tensor(f"blk.{blk}.attn_norm.weight", torch.ones(cfg.hidden_size, dtype=torch.float32).numpy())
        w.add_tensor(
            f"blk.{blk}.ffn_down.weight",
            torch.zeros(cfg.intermediate_size, cfg.hidden_size, dtype=torch.bfloat16)
            .view(torch.int16)
            .numpy(),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
        w.add_tensor(
            f"blk.{blk}.nextn.eh_proj.weight",
            torch.zeros(cfg.hidden_size, 2 * cfg.hidden_size, dtype=torch.bfloat16).view(torch.int16).numpy(),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _gguf_name(key: str, gdn: dict[str, str]) -> str | None:
    """The template name of a source tensor, or None for one a qwen35 GGUF does not hold."""
    flat = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if key in flat:
        return flat[key]
    if ".layers." not in key:
        return None
    bid, rest = key.split(".layers.", 1)[1].split(".", 1)
    for src_part, dst in gdn.items():
        if rest.startswith(f"linear_attn.{src_part}."):
            return f"blk.{bid}.{dst}.weight"
    plain = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
    }
    return f"blk.{bid}.{plain[rest]}" if rest in plain else None


def _dequantize_q4k(raw: bytes, rows: int, cols: int) -> torch.Tensor:
    """Q4_K records -> values, as ``dequantize_row_q4_K``."""
    d, dmin, sc, m, q = q4k_fields(raw)
    scale = torch.from_numpy(d.view("<f2").astype("<f4"))[:, None] * torch.from_numpy(sc).float()
    zero = torch.from_numpy(dmin.view("<f2").astype("<f4"))[:, None] * torch.from_numpy(m).float()
    v = scale.repeat_interleave(32, 1) * torch.from_numpy(q).float() - zero.repeat_interleave(32, 1)
    return v.reshape(rows, cols)


class ShiftOutProj(Step):
    """A test layer step: shifts layer 0's GDN output projection, as an offline rotation does. Module level: pickled."""

    name: ClassVar[str] = "shift_out_proj_test"
    preserves_quantized: ClassVar[bool] = True

    @torch.no_grad()
    def apply_layer(self, ctx) -> None:  # noqa: ANN001
        if ctx.graph.index == 0:
            ctx.layer.linear_attn.out_proj.weight.add_(0.125)


def _build(tmp_path: Path, step, out: str, **settings: object):  # noqa: ANN001
    cfg = transformers.Qwen3_5TextConfig(**GDN)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    src = tmp_path / f"m-{out}"
    model.save_pretrained(src)
    save_tiny_tokenizer(src)
    calib = tmp_path / f"calib-{out}.jsonl"
    calib.write_text('{"text": "the capital of france is paris and the sky is blue"}\n' * 4)
    template = tmp_path / f"template-{out}.gguf"
    _template(src, template, cfg)
    checkout = tmp_path / f"llama.cpp-{out}"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text("")
    import subprocess

    ident = ["-c", "user.email=t@e", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(checkout), *ident, "commit", "-qm", "x", "--allow-empty"], check=True)
    built = build(
        str(src),
        Recipe(
            steps=list(step) if isinstance(step, list) else [step],
            calib=FileSource(path=str(calib), samples=2, seqlen=16),
        ),
        format="w4a16-asym-g32",
        out_dir=tmp_path / out,
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), **settings}},
        log=lambda _: None,
    )
    return built / "model-Q4_K_M.gguf", src, cfg, model


def test_gptq_carries_solved_codes_through_the_value_head_reorder(tmp_path: Path, monkeypatch) -> None:
    """A solving build exports, and ``ssm_out`` holds its solution in the template's column order.

    Three things have to hold at once, and dropping any one of them is a way this has already gone wrong:
    the export must not refuse; the stored values must match the source under the converter's permutation (so
    the codes were placed in template order, not source order); and they must differ from what a plain
    round-to-nearest of the template gives, so a silent fall back to quantizing the template is caught.
    """
    from safetensors.torch import load_file

    solutions = {}
    add_layer = GGUFWriter.add_layer

    def capture(self, ctx):
        for module, qp in ctx.qparams.items():
            if module.endswith("linear_attn.out_proj"):
                solutions[self.gguf_name(module)] = (
                    qp["codes"].float() * qp["scale"].repeat_interleave(32, -1)
                    - qp["zero_point"].repeat_interleave(32, -1)
                ).cpu()
        return add_layer(self, ctx)

    monkeypatch.setattr(GGUFWriter, "add_layer", capture)
    gptq, src, cfg, model = _build(tmp_path, REGISTRY["gptq"](grid="q4k", impl="torch"), "out-gptq")
    rtn, _, _, _ = _build(tmp_path, REGISTRY["rtn"](grid="q4k"), "out-rtn")

    name = "blk.0.ssm_out.weight"
    tp = file.Template(gptq)
    assert tp.tensors[name].tensor_type == 12  # Q4_K: the layer pass owns it
    assert name not in json.loads((gptq.parent / "build.json").read_text())["gguf"]["rounded_from_template"]

    rows, cols = tp.tensors[name].shape[1], tp.tensors[name].shape[0]
    got = _dequantize_q4k(tp.raw(name), rows, cols)
    torch.testing.assert_close(got, solutions[name], rtol=0, atol=0)
    source = load_file(str(src / "model.safetensors"))["model.layers.0.linear_attn.out_proj.weight"].float()
    want = source.index_select(
        1,
        torch.from_numpy(
            v_head_columns(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        ),
    )

    err = (got - want).abs().mean()
    scale = want.abs().mean()
    assert err < 0.25 * scale, f"stored ssm_out is not the source under the reorder: {err=} {scale=}"
    assert (got - source).abs().mean() > err, "ssm_out is in the source's column order, not the template's"
    assert file.Template(rtn).raw(name) != tp.raw(name), "ssm_out fell back to quantizing the template"

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        native = model(ids, use_cache=False).logits.double()
        with ExitStack() as contexts:
            for layer in model.model.layers:
                contexts.enter_context(template_columns(layer))
            canonical = model(ids, use_cache=False).logits.double()
        restored = model(ids, use_cache=False).logits.double()
    logp, logq = native.log_softmax(-1), canonical.log_softmax(-1)
    assert (logp.exp() * (logp - logq)).sum(-1).mean() < 1e-6
    torch.testing.assert_close(native, restored, rtol=0, atol=0)


def _writer(tmp_path: Path, src: Path, template: Path):  # noqa: ANN001, ANN202
    from qda_quant.arch import describe
    from qda_quant.format import Format
    from qda_quant.target import Target

    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
    return GGUFWriter(
        tmp_path / "out-head",
        src,
        describe(model, verify=False),
        Format.parse("w4a16"),
        Target(),
        llama_cpp=None,
        template=template,
    )


def test_a_head_layer_is_named_in_the_files_own_nextn_block(tmp_path: Path) -> None:
    """``mtp.layers.<k>.<rest>`` names the block after the file's decoder blocks, whichever root it nests under.

    The 27B's checkpoint roots its text model at ``model.language_model``, which the converter's tensor map
    does not know (its keys are ``model.layers.<i>``), so a name built from the graph's own module root found
    nothing and the rotated build refused with "the template has no tensor for it". The file's own
    ``nextn_predict_layers`` says where its head starts; a file without it has no head block, and the callers
    must keep refusing rather than write a head's values into a decoder layer.
    """
    cfg = transformers.Qwen3_5TextConfig(**GDN)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    src = tmp_path / "m-head"
    model.save_pretrained(src)
    save_tiny_tokenizer(src)
    with_nextn = tmp_path / "with-nextn.gguf"
    _template(src, with_nextn, cfg, nextn=1)
    w = _writer(tmp_path, src, with_nextn)
    assert w._block_name("mtp.layers.0.input_layernorm.weight") == "blk.4.attn_norm.weight"
    assert w._block_name("mtp.layers.0.mlp.down_proj.weight") == "blk.4.ffn_down.weight"
    assert w._block_name("mtp.layers.1.input_layernorm.weight") is None  # the file has one head block
    assert w._block_name("mtp.fc.weight") == "blk.4.nextn.eh_proj.weight"
    without = tmp_path / "no-nextn.gguf"
    _template(src, without, cfg)
    w = _writer(tmp_path, src, without)
    assert w._block_name("mtp.layers.0.input_layernorm.weight") is None


def test_a_changed_output_projection_is_written_in_the_templates_column_order(tmp_path: Path) -> None:
    """A step that changes ``out_proj`` writes the build's value in the *template's* column order.

    The layer pass holds every GDN output projection in the converter's order (`template_columns`) while the
    writer runs, so this weight is already permuted and its *source* is the side a comparison must permute.
    Comparing the build against the source in source order refused an untouched 27B build outright ("the
    template is not a row permutation of the source weight"), and a changed one has to land in the column
    order the file's other GDN tensors are in. The recipe has no quantizer, as a rotation-only build has none,
    and ``ssm_out`` is stored F32 so the test reads the exact bytes the writer chose.
    """
    from safetensors.torch import load_file

    out, src, cfg, _ = _build(
        tmp_path,
        ShiftOutProj(),
        "out-shift",
        tensor_types=[{"pattern": "blk.0.ssm_out.weight", "type": "F32"}],
    )
    cols = torch.from_numpy(
        v_head_columns(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    )
    source = load_file(str(src / "model.safetensors"))["model.layers.0.linear_attn.out_proj.weight"]
    tp = file.Template(out)
    assert tp.tensors["blk.0.ssm_out.weight"].tensor_type == 0  # F32: no quantizer, so the bytes are exact
    want = source.index_select(1, cols) + 0.125  # permuted by the layer pass, then shifted by the step
    got = tp.values("blk.0.ssm_out.weight")
    torch.testing.assert_close(got, want.float(), rtol=0, atol=0)
    assert not torch.equal(got, (source + 0.125).float()), "ssm_out is in the source's column order"
    assert "blk.0.ssm_out.weight" in json.loads((out.parent / "build.json").read_text())["gguf"]["from_build"]
