"""The ``gguf`` exporter refuses codes a later step made stale.

A layer pass runs every step of the pass on a layer, then hands it to the writer (`engine.runner.Pass.run`), so a
second layer step can write a weight whose codes the quantize step already recorded. Steps write through ``.data``,
which the autograd version counter does not see; the writer's digest check (`export.check_unchanged`) does. The
template here is written straight from the checkpoint (a bf16 GGUF in the source's row order, the arrangement the
exporter matches rows against), so the test needs no llama.cpp.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
import transformers
from qda_quant import REGISTRY, Recipe
from qda_quant.arch import DEFAULT_KEEP, submodule
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.format import Format
from qda_quant.gguf import file, rule
from qda_quant.gguf.export import OUT_NAME
from qda_quant.gguf.tensor_types import TensorTypeRule
from qda_quant.step import Refused, Step
from qda_quant.steps.rtn import RTN

from ..engine_util import save_tiny_tokenizer, write_ids
from ..tiny import SMALL

#: Wide enough for the Q4_K_M rule to store ``q_proj`` as Q4_K (rows of 256 values); tiny.py's 128 falls back.
WIDE = {**SMALL, "hidden_size": 256, "intermediate_size": 512, "vocab_size": 256}


class Tamper(Step):
    """A buggy layer step: claims to keep quantized weights but scales layer 0's q_proj after RTN. Module level: pickled."""

    name: ClassVar[str] = "tamper_test"
    preserves_quantized: ClassVar[bool] = True

    def apply_layer(self, ctx) -> None:  # noqa: ANN001
        if ctx.graph.index == 0:
            submodule(ctx.layer, ctx.graph, ctx.graph.module + ".self_attn.q_proj").weight.data.mul_(2)


class ScaleRound(RTN):
    """An RTN plugin whose inherited solver follows its own change to a weight."""

    name: ClassVar[str] = "scale_round_test"

    @torch.no_grad()
    def apply_layer(self, ctx) -> None:
        ctx.layer.self_attn.q_proj.weight.mul_(2)
        super().apply_layer(ctx)


class Shift(Step):
    """A test model step: changes the embedding, the final norm and the head, as an offline rotation does."""

    name: ClassVar[str] = "shift_test"
    kind: ClassVar[str] = "model"

    @torch.no_grad()
    def apply_model(self, model, graph, run) -> None:  # noqa: ANN001, ARG002
        model.get_submodule(graph.embed).weight.add_(0.25)
        model.get_submodule(graph.final_norm).weight.add_(0.5)
        model.get_submodule(graph.lm_head).weight.mul_(0.5)


class NormShift(Step):
    """A test layer step: shifts layer 0's input norm, the tensor an AWQ fold or a rotation changes."""

    name: ClassVar[str] = "norm_shift_test"
    preserves_quantized: ClassVar[bool] = True  # a norm is not quantized, so no code goes stale

    @torch.no_grad()
    def apply_layer(self, ctx) -> None:  # noqa: ANN001
        if ctx.graph.index == 0:
            submodule(ctx.layer, ctx.graph, ctx.graph.module + ".input_layernorm").weight.add_(0.75)


def _bf16_template(
    src: Path,
    out: Path,
    n_layer: int,
    *,
    norms_plus_one: bool = False,
    norms_f32: bool = False,
    mutate: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
    mtp: bool = False,
) -> None:
    """Write a bf16 GGUF of the checkpoint `src` under llama.cpp's tensor names, rows in the source's order.

    The exporter matches each template row to the source row with the same bytes, so a template without the
    converter's q/k row permutes is still read, and a test needs no llama.cpp.

    Args:
        norms_plus_one: shift 1-D norm weights by +1, as llama.cpp's converter does.
        norms_f32: store the shifted norms F32, as llama.cpp does, instead of rounding the sum to bf16.
        mutate: apply a converter-like transform to a tensor before it is written (to make a basis the writer
            cannot name).
        mtp: add a head block after the decoder's, as ``convert_hf_to_gguf.py`` writes a model's MTP head.
    """
    import gguf
    from safetensors.torch import load_file

    weights = load_file(str(src / "model.safetensors"))
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, n_layer)
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_block_count(n_layer + 1 if mtp else n_layer)
    if mtp:
        w.add_uint32("llama.nextn_predict_layers", 1)
    for key, t in weights.items():
        name = names.get_name(key, try_suffixes=(".weight", ".bias"))
        if name is None:
            raise KeyError(f"{key}: no llama.cpp tensor")
        if norms_plus_one and name.endswith("norm.weight"):
            t = t.float().add(1.0) if norms_f32 else t.float().add(1.0).to(torch.bfloat16)
        if mutate is not None:
            t = mutate(name, t)
        if t.dtype == torch.bfloat16:  # numpy has no bf16; its bits go through int16
            w.add_tensor(name, t.view(torch.int16).numpy(), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, t.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    if mtp:
        hidden = weights["model.embed_tokens.weight"].shape[1]
        head = {
            f"blk.{n_layer}.nextn.eh_proj.weight": torch.full((2 * hidden, hidden), 0.5),
            f"blk.{n_layer}.nextn.enorm.weight": torch.ones(hidden),
            f"blk.{n_layer}.nextn.hnorm.weight": torch.ones(hidden),
            f"blk.{n_layer}.nextn.shared_head_norm.weight": torch.ones(hidden),
        }
        for name, t in head.items():
            w.add_tensor(name, t.numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_a_build_without_a_checkout_names_the_fix(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """The converter is llama.cpp's: no checkout and no template is refused, naming the one line that makes one."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.llama_cpp import ENV
    from qda_quant.step import Refused

    monkeypatch.delenv(ENV, raising=False)
    src, calib, _template, _checkout, _ = _source(tmp_path, WIDE)
    with pytest.raises(Refused, match="no llama.cpp checkout") as e:
        build(
            str(src),
            Recipe(
                steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / "out",
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {}},
            log=lambda _: None,
        )
    assert ENV in str(e.value) and "git clone" in str(e.value) and "template" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_the_environment_names_the_checkout(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """The exporter setting wins; unset, ``$QDA_LLAMA_CPP`` is the machine's checkout; neither is a refusal."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template
    from qda_quant.gguf.llama_cpp import ENV, locate

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    monkeypatch.setenv(ENV, str(checkout))
    assert locate(None) == checkout
    assert locate(str(tmp_path / "named")) == tmp_path / "named"  # a setting wins over the environment
    out = build(
        str(src),
        Recipe(steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"template": str(template)}},
        log=lambda _: None,
    )
    info = json.loads((out / "build.json").read_text())
    assert (
        info["gguf"]["llama_cpp"]
        == subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    assert Template(out / "model-Q4_K_M.gguf").arch == "llama"
    # a template alone needs no checkout at all
    monkeypatch.delenv(ENV)
    build(
        str(src),
        Recipe(steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out2",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"template": str(template)}},
        log=lambda _: None,
    )
    assert json.loads((tmp_path / "out2" / "build.json").read_text())["gguf"]["llama_cpp"] is None


def test_the_template_walks_a_file_as_the_gguf_reader_does(tmp_path: Path) -> None:
    """`file.Template` replaced ``gguf.GGUFReader`` for speed; every kind of KV pair must end where the reader says."""
    gguf = pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template

    path = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_block_count(2)
    w.add_custom_alignment(64)
    w.add_token_list(["a", "bb", "", "dddd"])
    w.add_array("test.ints", [1, 2, 3])
    w.add_array("test.floats", [0.5, 1.5])
    w.add_float32("test.f32", 0.25)
    w.add_bool("test.bool", True)
    w.add_int32("test.i32", -7)
    w.add_uint64("test.u64", 1 << 40)
    w.add_string("test.str", "value")
    bf16 = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
    w.add_tensor("blk.0.a.weight", bf16.view(torch.int16).numpy(), raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.b.weight", torch.ones(5).numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    ref, tp = gguf.GGUFReader(path), Template(path)
    pairs = {k: f for k, f in ref.fields.items() if not k.startswith("GGUF.")}
    assert list(tp.kv) == list(pairs)
    for k, f in pairs.items():
        start, _, _, end = tp.kv[k]
        assert (start, end) == (f.offset, f.offset + sum(p.nbytes for p in f.parts)), k
    assert (tp.arch, tp.n_layer, tp.alignment) == ("qwen3", 2, 64)
    assert (
        tp.number("test.i32") == -7 and tp.number("test.u64") == 1 << 40 and tp.string("test.str") == "value"
    )
    assert list(tp.tensors) == [t.name for t in ref.tensors]
    for t in ref.tensors:
        got = tp.tensors[t.name]
        assert got.shape == tuple(int(x) for x in t.shape) and got.tensor_type == int(t.tensor_type)
        assert got.data.tobytes() == t.data.tobytes()
    assert torch.equal(tp.values("blk.0.a.weight"), bf16.float())
    assert torch.equal(tp.values("blk.0.a.weight", 1, 2), bf16.float()[1:2])


def _source(tmp_path: Path, config: dict) -> tuple[Path, Path, Path, Path, int]:
    """A tiny checkpoint and what a build reads next to it: (source, calibration, template, checkout, layers)."""
    src = tmp_path / "m"
    cfg = transformers.LlamaConfig(**config)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    m.save_pretrained(src)
    tokenizer_sha = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], tokenizer_sha)

    template = tmp_path / "template.gguf"
    _bf16_template(src, template, cfg.num_hidden_layers)

    # The exporter records the checkout's commit; a template means the converter itself is never run.
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text("")
    ident = ["-c", "user.email=t@e", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(checkout), *ident, "commit", "-qm", "x", "--allow-empty"], check=True)
    return src, calib, template, checkout, cfg.num_hidden_layers


def test_the_mtp_head_is_written_unless_the_recipe_drops_it(tmp_path: Path) -> None:
    """The template's MTP/NextN head reaches the file by default; ``mtp: false`` writes the same file without it.

    Every other tensor must keep its bytes, so a file that drops the head is the one the build would have
    written from a ``--no-mtp`` conversion: same decoder, no head block, nothing re-solved.
    """
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template

    src, calib, _plain, checkout, layers = _source(tmp_path, WIDE)
    head = tmp_path / "head.gguf"
    _bf16_template(src, head, layers, mtp=True)
    recipe = Recipe(steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8))
    # One type for every matrix, so the file's types cannot depend on block_count the way the Q4_K_M rule's
    # attn_v/ffn_down schedule does: this test is about the head, not the mix.
    types = [{"pattern": "re:.*", "type": "Q8_0"}, {"pattern": "re:.*norm.*", "type": "F32"}]

    def run(name: str, **settings: bool) -> tuple[Path, dict]:
        out = build(
            str(src),
            recipe,
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={
                "gguf": {
                    "llama_cpp": str(checkout),
                    "template": str(head),
                    "tensor_types": types,
                    **settings,
                }
            },
            log=lambda _: None,
        )
        return out, json.loads((out / "build.json").read_text())

    kept_out, kept_info = run("kept")
    kept = Template(kept_out / "model-Q4_K_M.gguf")
    head_names = sorted(n for n in kept.tensors if n.startswith(f"blk.{layers}.nextn."))
    assert len(head_names) == 4
    assert kept.n_layer == layers + 1 and kept.number("llama.nextn_predict_layers") == 1

    dropped_out, dropped_info = run("dropped", mtp=False)
    dropped = Template(dropped_out / "model-Q4_K_M.gguf")
    assert dropped.n_layer == layers
    assert "llama.nextn_predict_layers" not in dropped.kv
    assert not [n for n in dropped.tensors if n.startswith(f"blk.{layers}.")]
    for name in dropped.tensors:
        assert dropped.raw(name) == kept.raw(name), name
    assert dropped_info["output_bytes"] < kept_info["output_bytes"]
    assert dropped_info["output_bytes"] == (dropped_out / "model-Q4_K_M.gguf").stat().st_size


def test_dropping_the_head_writes_what_a_headless_template_writes(tmp_path: Path) -> None:
    """``mtp: false`` reads the same view the writer does, so the default Q4_K_M rule decides the same types.

    The rule promotes the first and last ``attn_v`` to Q6_K and some ``ffn_down`` to Q5_K from the template's
    block count, so a head the file does not hold must not move them: the mix's derived formats and the stored
    types come from one view, or the export silently falls back to the template's values for every tensor whose
    type the two views disagree on. The file then has to equal the one built from a template converted without
    the head, byte for byte.
    """
    pytest.importorskip("gguf")
    src, calib, _plain, checkout, layers = _source(tmp_path, WIDE)
    head, plain = tmp_path / "head.gguf", tmp_path / "plain.gguf"
    _bf16_template(src, head, layers, mtp=True)
    _bf16_template(src, plain, layers)
    # A step that solves, so the build writes its own codes instead of the template-shortcut's copy: only then
    # can a type the two views disagree on show up as dropped codes.
    recipe = Recipe(steps=[REGISTRY["gptq"](grid="q4k", impl="torch")], calib=FileSource(path=str(calib), samples=1, seqlen=8))

    def run(name: str, template: Path, **settings: bool) -> tuple[bytes, dict]:
        out = build(
            str(src),
            recipe,
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), **settings}},
            log=lambda _: None,
        )
        return (out / OUT_NAME).read_bytes(), json.loads((out / "build.json").read_text())

    dropped, dropped_info = run("dropped", head, mtp=False)
    headless, headless_info = run("headless", plain)
    assert dropped == headless, "the dropped head moved the Q4_K_M mix away from the headless template's"
    # The Q4_K_M mix stores some tensors in a type the recipe's format does not solve (that is what the rule is
    # for); a head the file does not hold must not add to that set or take anything out of it.
    assert dropped_info["gguf"].get("codes_dropped") == headless_info["gguf"].get("codes_dropped")


def test_a_head_rule_is_refused_with_its_fix_when_the_recipe_drops_the_head(tmp_path: Path) -> None:
    """A rule for the dropped head's tensors still refuses, and the message names the fix rather than a typo.

    The published 27B Q4 recipe states a type for every head tensor. Turning ``mtp`` off on such a recipe has to
    fail -- a rule that names no tensor is a typo, and ignoring it would let real typos through -- but the fix is
    to delete those rules, not to look for a tensor the head already took away.
    """
    pytest.importorskip("gguf")
    from qda_quant.arch import describe
    from qda_quant.gguf.writer import GGUFWriter
    from qda_quant.target import Target

    src, _calib, _template, checkout, layers = _source(tmp_path, WIDE)
    head = tmp_path / "head.gguf"
    _bf16_template(src, head, layers, mtp=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)

    def refuse(pattern: str) -> str:
        with pytest.raises(Refused) as caught:
            GGUFWriter(
                tmp_path / "out",
                src,
                describe(model, verify=False),
                Format.parse("w4a16"),
                Target(),
                llama_cpp=str(checkout),
                template=str(head),
                mtp=False,
                tensor_types=[TensorTypeRule(pattern=pattern, type="F32")],
            )
        return str(caught.value)

    message = refuse(f"blk.{layers}.nextn.eh_proj.weight")
    assert message.count("\n") == 1, message
    assert "matches no tensor" in message
    assert "`mtp: false`" in message and "delete" in message
    typo = refuse("blk.0.no_such_tensor.weight")
    assert typo.count("\n") == 0 and "mtp: false" not in typo, typo


def test_a_changed_head_tensor_is_refused_when_the_recipe_drops_it(tmp_path: Path) -> None:
    """A step's change to a head tensor cannot be written into a file without the head: the export refuses."""
    pytest.importorskip("gguf")
    from qda_quant.arch import describe
    from qda_quant.gguf.writer import GGUFWriter
    from qda_quant.target import Target

    src, _calib, _template, checkout, layers = _source(tmp_path, WIDE)
    head = tmp_path / "head.gguf"
    _bf16_template(src, head, layers, mtp=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    writer = GGUFWriter(
        tmp_path / "out",
        src,
        describe(model, verify=False),
        Format.parse("w4a16"),
        Target(),
        llama_cpp=str(checkout),
        template=str(head),
        mtp=False,
    )
    with pytest.raises(Refused, match="the template has no tensor for it"):
        writer.add_source_tensors({"mtp.fc.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})


def test_rtn_writes_the_same_bytes_without_the_layer_pass(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """The fast path (the template, in parts of a few rows) against its reference (the layer pass)."""
    pytest.importorskip("gguf")
    # ffn_down's 384-value rows take the 32-value fallbacks; 256-value rows take Q4_K and Q6_K
    src, calib, template, checkout, layers = _source(tmp_path, {**WIDE, "intermediate_size": 384})

    def run(name: str, **settings: bool) -> tuple[bytes, dict]:
        out = build(
            str(src),
            Recipe(
                steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), **settings}},
            log=lambda _: None,
        )
        info = json.loads((out / "build.json").read_text())
        return (out / "model-Q4_K_M.gguf").read_bytes(), info

    reference, slow = run("slow", layer_pass=True)
    monkeypatch.setattr(
        "qda_quant.gguf.writer._PART_VALUES", 700
    )  # rows of 256 and 384: parts of 2 and 1 rows
    fast_bytes, fast = run("fast")

    assert [s["kind"] for s in slow["stages"]] == ["layers"]
    assert [s["kind"] for s in fast["stages"]] == ["export"]
    assert "load" not in fast["timings"]["phases"]
    from qda_quant.gguf.llama_cpp import PINNED

    checkout_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert fast["gguf"]["llama_cpp"] == checkout_commit  # the converter's commit, as read from the checkout
    assert fast["gguf"]["llama_cpp_pinned"] == PINNED  # and the commit the parity is stated against
    import gguf

    types = {t.tensor_type.name for t in gguf.GGUFReader(tmp_path / "fast" / "model-Q4_K_M.gguf").tensors}
    assert {"Q4_K", "Q6_K", "Q5_0", "Q8_0"} <= types, f"{layers} layers reach only {types}"
    assert fast_bytes == reference


def test_the_shipped_c_quantizer_writes_the_same_file_as_the_port(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """A whole tiny model, exported once by the numpy port and once by the shipped C library: the same bytes."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.kernel import use_ggml

    from .test_kernel import compile_quantizer

    lib = compile_quantizer(tmp_path)
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)

    def run(name: str) -> tuple[bytes, dict]:
        out = build(
            str(src),
            Recipe(
                steps=[REGISTRY["rtn"](grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), "layer_pass": True}},
            log=lambda _: None,
        )
        return (out / "model-Q4_K_M.gguf").read_bytes(), json.loads((out / "build.json").read_text())

    try:
        use_ggml(None)
        port, port_info = run("port")
        monkeypatch.setattr("qda_quant.gguf.kernel.bundled_path", lambda: lib)
        shipped, shipped_info = run("shipped")
    finally:
        use_ggml(None)
    assert shipped == port
    assert port_info["gguf"]["kernel"] == "numpy"
    assert shipped_info["gguf"]["kernel"].startswith("qda-quant-gguf")


def test_a_layer_step_that_changes_a_quantized_weight_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)

    recipe = Recipe(
        steps=[REGISTRY["rtn"](grid="q4k"), Tamper()],
        calib=FileSource(path=str(calib), samples=1, seqlen=8),
    )
    with pytest.raises(
        RuntimeError, match="self_attn.q_proj: the weight changed after its codes were recorded"
    ):
        build(
            str(src),
            recipe,
            format="w4a16-asym-g32",
            out_dir=tmp_path / "out",
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template)}},
            log=lambda s: None,
        )


@pytest.mark.parametrize(
    "steps",
    [[RTN(grid="q4k"), RTN(grid="q4k")], [ScaleRound(grid="q4k")]],
    ids=["two-rounding-steps", "rounding-subclass"],
)
def test_rounding_variants_take_the_layer_pass(tmp_path: Path, steps: list[Step]) -> None:
    """One template quantization cannot replace repeated rounding or a plugin's own quantization math.

    A second RTN re-quantizes the first step's dequantized weights; an RTN subclass may change the weights
    before fitting them. Both must take the layer pass and produce its reference bytes.
    """
    pytest.importorskip("gguf")
    src, calib, template, checkout, _ = _source(tmp_path, {**WIDE, "num_hidden_layers": 1})

    def run(name: str, **settings: bool) -> tuple[bytes, dict]:
        out = build(
            str(src),
            Recipe(
                steps=steps,
                calib=FileSource(path=str(calib), samples=1, seqlen=8),
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), **settings}},
            log=lambda _: None,
        )
        info = json.loads((out / "build.json").read_text())
        return (out / "model-Q4_K_M.gguf").read_bytes(), info

    reference, slow = run("slow", layer_pass=True)
    fast_bytes, fast = run("fast")

    assert [s["kind"] for s in slow["stages"]] == ["layers"]
    assert [s["kind"] for s in fast["stages"]] == ["layers"]  # the rounding shortcut is refused
    assert fast_bytes == reference


def test_imatrix_uses_the_checkout_library_when_a_wheel_ships_a_plain_kernel(
    tmp_path: Path, monkeypatch, ggml_lib: Path
) -> None:
    """A wheel's plain-only quantizer cannot hide the checkout's weighted fit from the documented recipe."""
    from qda_quant.gguf import kernel

    from .test_kernel import compile_quantizer

    shipped = compile_quantizer(tmp_path)
    monkeypatch.setattr(kernel, "bundled_path", lambda: shipped)
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    discovered = checkout / "build" / "bin" / "libggml-base.so"
    discovered.parent.mkdir(parents=True)
    discovered.symlink_to(ggml_lib)
    try:
        out = build(
            str(src),
            Recipe(
                steps=[REGISTRY["gptq"](grid="q4k", clip="imatrix", impl="torch")],
                calib=FileSource(path=str(calib), samples=1, seqlen=8),
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / "out",
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template)}},
            log=lambda _: None,
        )
        assert (out / "model-Q4_K_M.gguf").is_file()
        assert json.loads((out / "build.json").read_text())["gguf"]["kernel"].startswith("libggml")
    finally:
        kernel.use_ggml(None)


def test_a_solving_build_records_the_tensors_it_rounded_from_the_template(tmp_path: Path) -> None:
    """A build whose codes are dropped for a tensor says so, per tensor, in ``build.json``.

    `GGUFWriter._add` writes Q4_K only, so a Q6_K promotion is llama.cpp's round-to-nearest of the original
    weight even though the build solved it. Nothing in the file distinguishes the two, which is why the record
    exists; a reader comparing this build to one of llama.cpp's own needs to know which tensors are shared.
    """
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    out = build(
        str(src),
        Recipe(
            steps=[REGISTRY["gptq"](grid="q4k", impl="torch")],
            calib=FileSource(path=str(calib), samples=1, seqlen=8),
        ),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template)}},
        log=lambda _: None,
    )
    rounded = json.loads((out / "build.json").read_text())["gguf"]["rounded_from_template"]
    types = {n: t.tensor_type for n, t in file.Template(out / OUT_NAME).tensors.items()}

    dropped = {n for n, why in rounded.items() if "codes dropped" in why}
    assert dropped, "the mix stores some solved tensors outside Q4_K; none was recorded"
    assert all(types[n] != rule.Q4_K for n in dropped), "a Q4_K tensor was written from the template"
    assert all(file.type_name(types[n]) in rounded[n] for n in rounded), "a record does not name its type"
    # This fixture reaches both ways out of Q4_K, and the record has to cover both: a Q6_K promotion
    # (`use_more_bits`) and a Q5_0 fallback (rows that are not whole super-blocks).
    assert {types[n] for n in dropped} == {rule.Q6_K, rule.Q5_0}
    # A tensor the build never solved is rounded from the template too; the record says what the file was
    # written from, not only what the build lost.
    assert set(rounded) - dropped
    # A float conversion is not a rounding: a tensor the file stores as a float is neither recorded nor
    # counted. Both ends matter: with only the record filtered the warning read "3 of 1", and with
    # only F32 filtered its denominator counted every BF16 tensor the file copies unchanged.
    floats = {n for n, t in types.items() if t in (rule.F32, rule.F16, rule.BF16)}
    assert floats and not (floats & set(rounded))
    assert len(rounded) <= len(types) - len(floats)


def test_a_build_lists_every_tensor_whose_solved_codes_it_dropped(tmp_path: Path, caplog) -> None:
    """Solved Q4_K codes the mix stores another way are dropped LOUDLY, not refused (F3b).

    `rounded_from_template` leaves float tensors out (a float copy is not a rounding), so a solved tensor the mix
    stores as F16 used to leave no trace. `gguf.codes_dropped` lists every dropped tensor with its stored type,
    floats included, and one warning names the count and the remedy. The build itself still succeeds.
    """
    import logging

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    floated = "blk.0.attn_q.weight"
    with caplog.at_level(logging.WARNING, logger="qda_quant.gguf.writer"):
        out = build(
            str(src),
            Recipe(
                steps=[REGISTRY["gptq"](grid="q4k", impl="torch")],
                calib=FileSource(path=str(calib), samples=1, seqlen=8),
            ),
            format="w4a16-asym-g32",
            out_dir=tmp_path / "out",
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={
                "gguf": {
                    "llama_cpp": str(checkout),
                    "template": str(template),
                    "tensor_types": [{"pattern": floated, "type": "F16"}],
                }
            },
            log=lambda _: None,
        )
    record = json.loads((out / "build.json").read_text())["gguf"]
    types = {n: t.tensor_type for n, t in file.Template(out / OUT_NAME).tensors.items()}
    dropped = record["codes_dropped"]
    assert dropped[floated] == {"type": "F16", "module": "model.layers.0.self_attn.q_proj"}
    assert floated not in record["rounded_from_template"]  # the gap: a float drop is not a rounding
    # every rounded tensor the build solved is listed too, with the type the file really stores
    assert {n for n, why in record["rounded_from_template"].items() if "codes dropped" in why} <= set(dropped)
    assert all(file.type_name(types[n]) == d["type"] and types[n] != rule.Q4_K for n, d in dropped.items())
    assert {d["type"] for d in dropped.values()} >= {"F16", "Q6_K", "Q5_0"}
    warnings = [r.getMessage() for r in caplog.records if "codes_dropped" in r.getMessage()]
    assert len(warnings) == 1
    assert f"solved {len(dropped)} tensors" in warnings[0] and "tensor_types" in warnings[0]


def _imatrix_file(out: Path, template: Path, skip: set[str]) -> Path:
    """An importance matrix over `template`'s quantizable tensors, in llama-imatrix's layout.

    Written here rather than measured so the test needs no inference: the loader reads ``.in_sum2`` and
    ``.counts`` and divides, which is all that has to be exercised. `skip` leaves a tensor uncovered, as a real
    matrix does for ``token_embd``.
    """
    import gguf
    import numpy as np

    tp = file.Template(template)
    w = gguf.GGUFWriter(str(out), "imatrix")
    w.add_type("imatrix")
    rng = np.random.default_rng(0)
    for name, info in tp.tensors.items():
        if len(info.shape) < 2 or name in skip:
            continue
        width = info.shape[0]
        w.add_tensor(f"{name}.in_sum2", (rng.random(width) * 8 + 0.5).astype(np.float32))
        w.add_tensor(f"{name}.counts", np.array([4.0], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def test_an_importance_matrix_fits_only_the_tensors_written_from_the_template(
    tmp_path: Path, ggml_lib: Path
) -> None:
    """`imatrix` changes the tensors `finish` encodes, leaves the build's codes alone, and names what it skipped.

    The promoted and fallback tensors are the ones llama.cpp's own files fit with a matrix, so they are
    the ones this setting reaches. A tensor the matrix does not cover is fitted unweighted, as llama.cpp does,
    and recorded: `token_embd` has no matmul input to measure, so a real matrix never covers it.
    """
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    uncovered = "token_embd.weight"
    q8 = "blk.0.attn_q.weight"
    matrix = _imatrix_file(tmp_path / "m.imatrix.gguf", template, skip={uncovered})

    def run(name: str, **extra: object) -> tuple[bytes, dict]:
        out = build(
            str(src),
            Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={
                "gguf": {
                    "llama_cpp": str(checkout),
                    "template": str(template),
                    "libggml": str(ggml_lib),
                    "tensor_types": [{"pattern": q8, "type": "Q8_0"}],
                    **extra,
                }
            },
            log=lambda _: None,
        )
        return (out / OUT_NAME).read_bytes(), json.loads((out / "build.json").read_text())["gguf"]

    plain, plain_info = run("plain")
    weighted, info = run("weighted", imatrix=str(matrix))

    assert "imatrix" not in plain_info, "a build without the setting records one"
    assert info["imatrix"]["sha256"] and info["imatrix"]["template_sha256"]

    rounded = set(plain_info["rounded_from_template"])
    tp_plain, tp_weighted = (
        file.Template(tmp_path / "plain" / OUT_NAME),
        file.Template(tmp_path / "weighted" / OUT_NAME),
    )
    moved = {n for n in tp_plain.tensors if tp_plain.raw(n) != tp_weighted.raw(n)}
    assert moved, "the matrix changed nothing"
    assert moved <= rounded, "the matrix reached a tensor written from the build's own codes"
    assert uncovered not in moved, "a tensor the matrix does not cover was still weighted"
    assert tp_weighted.tensors[q8].tensor_type == rule.Q8_0
    assert tp_plain.raw(q8) == tp_weighted.raw(q8), "Q8_0 must ignore importance weights"
    assert "Q8_0 fit ignores the importance matrix" in info["rounded_from_template"][q8]
    assert "without the importance matrix" in info["rounded_from_template"][uncovered]
    assert plain != weighted


def test_cached_template_digest_keeps_the_file_and_provenance(
    tmp_path: Path, monkeypatch, ggml_lib: Path
) -> None:  # noqa: ANN001
    """Cold and warm cached exports match the uncached file and its full GGUF provenance."""
    from qda_quant.gguf import imatrix, writer
    from qda_quant.intake import fingerprint

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    matrix = _imatrix_file(tmp_path / "m.imatrix.gguf", template, skip={"token_embd.weight"})

    def run(name: str) -> tuple[str, dict]:
        out = build(
            str(src),
            Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={
                "gguf": {
                    "llama_cpp": str(checkout),
                    "template": str(template),
                    "libggml": str(ggml_lib),
                    "imatrix": str(matrix),
                    "layer_pass": True,
                }
            },
            log=lambda _: None,
        )
        return imatrix.sha256(out / OUT_NAME), json.loads((out / "build.json").read_text())["gguf"]

    with monkeypatch.context() as slow:
        slow.setattr(writer, "file_digest", imatrix.sha256)
        reference = run("reference")

    reads = []
    full_digest = fingerprint._sha256

    def counted(path: Path) -> str:
        if path == template.resolve():
            reads.append(path)
        return full_digest(path)

    monkeypatch.setattr(fingerprint, "_sha256", counted)
    assert run("cold") == reference
    assert reads == [template.resolve()]
    assert run("warm") == reference
    assert reads == [template.resolve()], "a shared template was read again on a cache hit"
    assert reference[1]["imatrix"]["sha256"] == imatrix.sha256(matrix)
    assert reference[1]["imatrix"]["template_sha256"] == imatrix.sha256(template)


def test_an_importance_matrix_is_refused_rather_than_ignored(tmp_path: Path, ggml_lib: Path) -> None:
    """Each way a matrix cannot be honoured refuses: the quiet alternative is a file that looks fitted.

    Without llama.cpp's library there is no weighted fit at all (the Python port has only the plain one), and a
    matrix measured on another model covers none of the tensors this build would fit with it.
    """
    from qda_quant.gguf import kernel

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    import gguf
    import numpy as np

    matrix = _imatrix_file(tmp_path / "m.imatrix.gguf", template, skip=set())
    foreign = tmp_path / "foreign.imatrix.gguf"  # names that belong to no tensor of this model
    w = gguf.GGUFWriter(str(foreign), "imatrix")
    w.add_type("imatrix")
    w.add_tensor("blk.99.attn_q.weight.in_sum2", np.ones(256, dtype=np.float32))
    w.add_tensor("blk.99.attn_q.weight.counts", np.array([4.0], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    def run(name: str, **extra: object) -> None:
        build(
            str(src),
            Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template), **extra}},
            log=lambda _: None,
        )

    with pytest.raises(Refused, match="covers none of"):
        run("foreign", libggml=str(ggml_lib), imatrix=str(foreign))
    kernel.use_ggml(None)
    try:
        with pytest.raises(Refused, match="needs llama.cpp's own fit"):
            run("nolib", imatrix=str(matrix))
    finally:
        kernel.use_ggml(None)


def test_an_importance_matrix_records_a_template_the_build_generated(
    tmp_path: Path, monkeypatch, ggml_lib: Path
) -> None:  # noqa: ANN001
    """The record hashes the template before `finish` deletes the one it made itself.

    Every other test here hands the exporter a `template`, which it keeps; a build that runs the converter
    owns the file and drops it at the end. Hashing it after that is a crash, not a wrong field, and only this
    path reaches it. The converter is stood in for: what is under test is the lifecycle.
    """
    import shutil

    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    matrix = _imatrix_file(tmp_path / "m.imatrix.gguf", template, skip=set())
    made: list[Path] = []

    def fake_convert(llama_cpp: object, source: object, out: object) -> Path:
        shutil.copyfile(template, out)
        made.append(Path(out))
        return Path(out)

    monkeypatch.setattr("qda_quant.gguf.writer.make_template", fake_convert)
    out = build(
        str(src),
        Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"llama_cpp": str(checkout), "libggml": str(ggml_lib), "imatrix": str(matrix)}},
        log=lambda _: None,
    )
    assert made and not made[0].exists(), "the generated template was kept"
    record = json.loads((out / "build.json").read_text())["gguf"]["imatrix"]
    assert record["template_sha256"] == hashlib.sha256(template.read_bytes()).hexdigest()


def test_an_expert_bank_takes_one_importance_vector_per_expert() -> None:
    """A bank's entry holds a vector per expert, and each expert's rows are weighed by its own.

    `llama-imatrix` measures every expert of a bank separately, so the entry for a `[cols, rows, experts]`
    tensor has `experts * cols` values, not one vector. Reading it as one, or as `rows * cols`, either refuses
    a valid matrix or weighs every expert by the first one's measurement.
    """
    from qda_quant.gguf import imatrix as imat

    ne = (512, 256, 4)  # a converted ffn_down_exps bank: 512 wide, 256 rows, 4 experts
    per_expert = np.concatenate([np.full(512, i + 1.0, dtype=np.float32) for i in range(4)])
    table = {"blk.0.ffn_down_exps.weight": per_expert}

    got = imat.column_weights(table, "blk.0.ffn_down_exps.weight", ne)
    assert got is not None and got.shape == (4, 512), "a per-expert entry is not read as one vector each"
    assert [float(got[i][0]) for i in range(4)] == [1.0, 2.0, 3.0, 4.0], "experts are mixed up"

    shared = {"blk.0.attn_q.weight": np.full(512, 7.0, dtype=np.float32)}
    plain = imat.column_weights(shared, "blk.0.attn_q.weight", (512, 256))
    assert plain is not None and plain.shape == (1, 512), "a plain entry is not one shared vector"

    # A single vector for a bank is refused, not broadcast: llama.cpp measures one per expert, so a file
    # holding one was measured for another model, and spreading it would weigh four experts by one of them.
    for size in (512, 512 * 256, 512 * 3, 7):
        with pytest.raises(Refused, match=r"expected 512 x 4 experts = 2048"):
            imat.column_weights(
                {"blk.0.ffn_down_exps.weight": np.ones(size, dtype=np.float32)},
                "blk.0.ffn_down_exps.weight",
                ne,
            )


def _ffn_down_module(template: Path, promoted: bool) -> str:
    """An ``mlp.down_proj`` whose GGUF tensor the mix does (or does not) store as Q6_K.

    `ffn_down` is its own unit, where q/k/v are solved as one fused linear and cannot be overridden singly --
    which is itself a constraint on what this opt-in can reach.
    """
    from qda_quant.gguf import rule
    from qda_quant.gguf.tensor_types import resolve_types

    types = resolve_types(file.Template(template), [])
    want = rule.Q6_K if promoted else rule.Q4_K
    name = next(n for n, t in types.items() if t == want and n.endswith("ffn_down.weight"))
    return f"model.layers.{name.split('.')[1]}.mlp.down_proj"


def test_the_gguf_exporter_declares_it_can_hold_units_stored_in_another_format(tmp_path: Path) -> None:
    """`check_exporter` refuses `plan.formats` unless the exporter declares it; the GGUF exporter must.

    A GGUF names a type per tensor, so it genuinely can say which modules are stored differently. Without the
    declaration every `format_overrides` build is refused before any solving, however well the rest works --
    a gap no test of the grid rule in isolation can see, since nothing in it reaches this check.

    This calls the real check on the real exporter. It does not run a build: a unit stored as Q6_K is still
    refused earlier, by the serving target, because the only target table is vLLM's and no vLLM kernel serves
    int6. That is tracked separately; when a llama.cpp target lands this should become an end-to-end build.
    """
    from types import SimpleNamespace

    from qda_quant.output.exporters import check_exporter, parse_exporter

    _, _, template, checkout, _ = _source(tmp_path, WIDE)
    exporter = parse_exporter({"gguf": {"llama_cpp": str(checkout), "template": str(template)}})
    assert type(exporter).supports_formats, "the class must declare it, not inherit a base's False"

    recipe = Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(tmp_path / "c.jsonl"), samples=1))
    plan = SimpleNamespace(
        formats={"model.layers.0.mlp.down_proj": Format.parse("w6a16-sym-g16")},
        graph=SimpleNamespace(layers=()),  # a dense override: no routed bank to store a projection of apart
        fmt=Format.parse("w4a16-asym-g32"),
        keep=DEFAULT_KEEP,
        staged=False,
        segments=(("layers", None),),
        float_stage=True,
    )
    check_exporter(exporter, plan, recipe)  # refuses on supports_formats=False; must not raise here


def test_an_override_takes_the_layer_pass_and_the_writer_accepts_the_format_map(tmp_path: Path) -> None:
    """Two ways a build with `format_overrides` escaped the checks that make overrides mean anything.

    The rounding shortcut (`writes_alone`) quantizes every tensor from the template without a layer pass, so a
    plain RTN build with an override wrote the mix's own type and never reached `_add` -- the override was
    accepted and silently did nothing. And once the exporter declares it holds several formats, the engine
    hands the writer a `formats` map, which this writer did not accept at all.
    """
    import inspect
    from types import SimpleNamespace

    from qda_quant.gguf.export import GGUF

    _, calib, template, checkout, _ = _source(tmp_path, WIDE)
    exporter = GGUF(llama_cpp=str(checkout), template=str(template))
    recipe = Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8))
    graph = SimpleNamespace(layers=[SimpleNamespace(banks=[])])

    assert exporter.writes_alone(SimpleNamespace(graph=graph, formats={}), recipe)
    assert not exporter.writes_alone(
        SimpleNamespace(graph=graph, formats={"model.layers.0.mlp.down_proj": object()}), recipe
    ), "a build with an override must take the layer pass, where its codes are written and checked"
    assert "formats" in inspect.signature(GGUF.writer).parameters, "the engine passes formats= to the writer"


def _stored(t: torch.Tensor, *, add: float = 0.0, plus: float = 0.0) -> torch.Tensor:
    """A bf16 tensor after a step's `add`, then the converter's `plus`, as the file must hold it (bf16, fp32-read)."""
    return ((t.float() + add).to(torch.bfloat16).float() + plus).to(torch.bfloat16).float()


def _converted(tmp_path: Path, src: Path, layers: int, **kw) -> Path:  # noqa: ANN003
    """A template written as llama.cpp's converter writes it: norms shifted by +1."""
    template = tmp_path / "converted.gguf"
    _bf16_template(src, template, layers, norms_plus_one=True, **kw)
    return template


def _build_with(
    tmp_path: Path, src: Path, calib: Path, template: Path, steps: list[Step], name: str, **exp
) -> Path:
    """Build `steps` against `template`; returns the output directory."""
    pytest.importorskip("gguf")
    return build(
        str(src),
        Recipe(steps=steps, calib=FileSource(path=str(calib), samples=1, seqlen=8)),
        format="w4a16-asym-g32",
        out_dir=tmp_path / name,
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"template": str(template), **exp}},
        log=lambda _: None,
    )


def test_a_changed_model_tensor_is_written_in_the_converters_basis(tmp_path: Path) -> None:
    """The embedding, final norm and head a model step changed are the build's values, in the converter's basis.

    A rotation changes exactly these three outside the layers; the template holds the *source's* values, with
    llama.cpp's ``norm.weight + 1``, so a changed norm must be written as ``bf16(build) + 1`` and not raw.
    """
    from safetensors.torch import load_file

    src, calib, _template, _checkout, layers = _source(tmp_path, WIDE)
    template = _converted(tmp_path, src, layers)
    out = _build_with(
        tmp_path,
        src,
        calib,
        template,
        [Shift(), REGISTRY["rtn"](grid="q4k")],
        "out",
        tensor_types=[  # read the two 2-D ones back exactly; the test is the basis, not the mix
            {"pattern": "token_embd.weight", "type": "F32"},
            {"pattern": "output.weight", "type": "F32"},
        ],
    )
    source = load_file(str(src / "model.safetensors"))
    got = file.Template(out / OUT_NAME)
    assert torch.equal(
        got.values("token_embd.weight"), _stored(source["model.embed_tokens.weight"], add=0.25)
    )
    assert torch.equal(
        got.values("output_norm.weight")[0], _stored(source["model.norm.weight"], add=0.5, plus=1.0)
    )
    assert torch.equal(got.values("output.weight"), _stored(source["lm_head.weight"]).mul(0.5))
    # a tensor no step changed keeps the template's bytes, offset and all
    assert torch.equal(
        got.values("blk.0.attn_norm.weight")[0],
        _stored(source["model.layers.0.input_layernorm.weight"], plus=1.0),
    )
    info = json.loads((out / "build.json").read_text())["gguf"]
    assert set(info["from_build"]) >= {"token_embd.weight", "output_norm.weight", "output.weight"}
    assert "+1" in info["from_build"]["output_norm.weight"]
    assert not set(info["rounded_from_template"]) & set(info["from_build"])


def test_a_changed_layer_norm_is_written_from_the_build(tmp_path: Path) -> None:
    """AWQ folds its scales into norms and an offline rotation rewrites them; the file must hold the folded value."""
    from safetensors.torch import load_file

    src, calib, _template, _checkout, layers = _source(tmp_path, WIDE)
    template = _converted(tmp_path, src, layers)
    out = _build_with(tmp_path, src, calib, template, [NormShift(), REGISTRY["rtn"](grid="q4k")], "out")
    source = load_file(str(src / "model.safetensors"))
    got = file.Template(out / OUT_NAME)
    layer0 = source["model.layers.0.input_layernorm.weight"]
    assert torch.equal(got.values("blk.0.attn_norm.weight")[0], _stored(layer0, add=0.75, plus=1.0))
    # another layer's norm is the template's, and every layer still has the build's quantized body
    layer1 = source["model.layers.1.input_layernorm.weight"]
    assert torch.equal(got.values("blk.1.attn_norm.weight")[0], _stored(layer1, plus=1.0))
    assert got.tensors["blk.0.attn_q.weight"].tensor_type == rule.Q4_K
    info = json.loads((out / "build.json").read_text())["gguf"]
    assert "blk.0.attn_norm.weight" in info["from_build"]
    assert "blk.1.attn_norm.weight" not in info["from_build"]


def test_an_f32_norm_is_written_in_the_converters_own_precision(tmp_path: Path) -> None:
    """A norm's ``+1`` is added in fp32 and stored F32: a bf16-rounded sum is a different tensor.

    The writer reproduces the template from the source to name the converter's offset, and writes the build's
    value the same way. Rounding either side to bf16 first breaks the identification (no offset matches) and,
    had it matched, would write a value the converter never produces.
    """
    from safetensors.torch import load_file

    src, calib, _template, _checkout, layers = _source(tmp_path, WIDE)
    template = _converted(tmp_path, src, layers, norms_f32=True)
    out = _build_with(tmp_path, src, calib, template, [NormShift(), REGISTRY["rtn"](grid="q4k")], "out")
    source = load_file(str(src / "model.safetensors"))
    got = file.Template(out / OUT_NAME)
    layer0 = source["model.layers.0.input_layernorm.weight"]
    want = (layer0.float() + 0.75).to(torch.bfloat16).float() + 1.0
    assert not torch.equal(want, want.to(torch.bfloat16).float()), "the fixture's +1 survives bf16 rounding"
    assert torch.equal(got.values("blk.0.attn_norm.weight")[0], want)
    layer1 = source["model.layers.1.input_layernorm.weight"]
    assert torch.equal(got.values("blk.1.attn_norm.weight")[0], layer1.float() + 1.0)
    assert "blk.0.attn_norm.weight" in json.loads((out / "build.json").read_text())["gguf"]["from_build"]


def test_a_basis_the_writer_cannot_name_is_refused_by_name(tmp_path: Path) -> None:
    """A converter that transforms values (``-exp(A_log)``) is not a basis: a changed tensor is refused, never copied.

    The template squares one norm, so neither a constant offset nor a row permutation relates it to the source;
    the writer must say so by name rather than write the build's value as if it were the same quantity.
    """
    src, calib, _template, _checkout, layers = _source(tmp_path, WIDE)
    template = _converted(
        tmp_path,
        src,
        layers,
        mutate=lambda name, t: t.float().pow(2).to(torch.bfloat16) if name == "blk.0.attn_norm.weight" else t,
    )
    with pytest.raises(Refused, match="blk.0.attn_norm.weight"):
        _build_with(tmp_path, src, calib, template, [NormShift(), REGISTRY["rtn"](grid="q4k")], "out")
