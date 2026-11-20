"""`build()` end to end on a tiny saved Qwen3 (CPU): plan, refusal, passes, writer calls, build.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.arch import submodule
from qda_quant.calib.sources import FileSource, VLMSource
from qda_quant.engine.build import build
from qda_quant.engine.planning import compile_plan, resolve_devices, segments
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.step import LayerContext, Refused, Step
from qda_quant.target import Target

from ..engine_util import (
    WriterExporter,
    rtn_,
    save_tiny_tokenizer,
    tiny_qwen3,
    tiny_qwen3_moe,
    write_ids,
    writes,
)
from ..goldens import check_golden, refusal


class TinyRTN(Step):
    needs: ClassVar[frozenset[str]] = frozenset({"hessian"})
    name: ClassVar[str] = "tiny_rtn"

    def apply_layer(self, ctx: LayerContext) -> None:
        for g in ctx.graph.groups:
            assert ctx.stat("hessian", g).dtype == torch.float32
            for m in g.modules:
                rtn_(submodule(ctx.layer, ctx.graph, m).weight.data)
                ctx.qparams[m] = {"done": torch.ones(1)}


class NoStatsRTN(Step):
    name: ClassVar[str] = "no_stats_rtn"

    def apply_layer(self, ctx: LayerContext) -> None:
        for group in ctx.graph.groups:
            for module in group.modules:
                rtn_(submodule(ctx.layer, ctx.graph, module).weight.data)
                ctx.qparams[module] = {"done": torch.ones(1)}


class NeverHere(Step):
    name: ClassVar[str] = "never_here"

    def check(self, graph, fmt, plan):
        raise Refused("not on this target")


class Marker(Step):
    kind: ClassVar[str] = "model"
    name: ClassVar[str] = "marker"
    calls: ClassVar[list] = []

    def apply_model(self, model, graph, run):
        Marker.calls.append(len(run.calib_ids))


class Toucher(Step):
    """A model step that changes a tensor outside the graph, as R1 does with a vision merger."""

    kind: ClassVar[str] = "model"
    name: ClassVar[str] = "toucher"

    def apply_model(self, model, graph, run):
        run.tensors["model.embed_tokens.weight"] = model.model.embed_tokens.weight.clone()


class FakeWriter:
    last: ClassVar[FakeWriter | None] = None

    def __init__(self, out_dir, source, graph, fmt, target, *, dtype=None):  # noqa: ARG002
        self.out = Path(out_dir)
        self.layers = []
        FakeWriter.last = self

    def add_layer(self, ctx):
        self.layers.append((ctx.graph.index, sorted(ctx.qparams)))

    def finish(self, config, recipe, build_info):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "build.json").write_text(json.dumps(build_info))
        return self.out


@pytest.fixture
def tiny(tmp_path):
    m = tiny_qwen3(layers=2)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (20,), generator=g).tolist() for _ in range(5)], sha
    )
    return tmp_path


def recipe(tmp, *steps):
    return Recipe(steps=list(steps), calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=16))


CPU = [torch.device("cpu")]


def test_segments():
    a, b, mk = TinyRTN(), TinyRTN(), Marker()
    assert [k for k, _ in segments([mk, a, b])] == ["model", "layers"]
    assert [k for k, _ in segments([a, mk])] == ["layers", "model", "layers"]


def test_plan_refuses_before_loading(tiny):
    with pytest.raises(Refused, match="not on this target"):
        compile_plan(str(tiny / "m"), recipe(tiny, NeverHere()), Format.parse("w4a16"), Target(), CPU)


def test_awq_and_rtn_never_hit_the_solve_offset_limit(tiny, monkeypatch):
    """The int32-offset limit only binds the GPTQ solve: AWQ + RTN apply no int32-addressed solve, so they still plan."""
    from qda_quant.solve import solve

    monkeypatch.setattr(solve, "INT32_ELEMENTS", 1)
    compile_plan(
        str(tiny / "m"),
        recipe(tiny, REGISTRY["awq"](), REGISTRY["rtn"]()),
        Format.parse("w4a16-sym-g32"),
        Target(),
        CPU,
    )


def test_shardable_needs_of_layer_steps_alone(tiny):
    """`build` asks this before it starts card processes: no statistics, one process; else the requested cards."""
    from qda_quant.engine.planning import shardable_needs
    from qda_quant.steps.gptq import GPTQ
    from qda_quant.steps.rtn import RTN

    def needs(*steps):
        return shardable_needs(str(tiny / "m"), recipe(tiny, *steps), Format.parse("w4a16-sym-g32"), Target())

    assert needs(RTN()) == frozenset()  # minmax reads the weights only
    assert needs(RTN(), RTN(clip="mse")) == frozenset()  # mse searches the weights' own error
    assert needs(RTN(clip="imse")) == frozenset({"hessian", "expert_hessian"})
    assert needs(TinyRTN(), GPTQ())  # a hessian: the calibration forward is what the cards share
    assert needs(Marker(), RTN()) is None  # a model step's weights fan out to the ranks
    assert needs(RTN(), Ckpt()) is None  # a checkpoint step runs its own stages


def test_plan_sizes_per_layer_inputs(tmp_path):
    """A model with per-layer embeddings keeps every calibration batch's [seq, layers, width] tensor on the
    card for the whole statistics pass, and the plan counts it on top of the flat call-arguments term.

    A Gemma 4 W4A16 build planned 46.9 GiB at 2,048 samples, passed, and died in the first GPTQ forward at
    79.17 GiB; at 1,024 samples (plan 31.9) it completed. The missing term was this tensor.
    """
    from qda_quant import REGISTRY
    from qda_quant.arch import describe
    from qda_quant.engine.planning import memory_terms, per_layer_inputs_bytes
    from qda_quant.intake.loading import load_model
    from transformers import AutoModelForCausalLM, Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size=128,
        max_position_embeddings=64,
        hidden_size_per_layer_input=16,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(tmp_path / "g")
    sha = save_tiny_tokenizer(tmp_path / "g")
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(8)]
    write_ids(tmp_path / "c.jsonl", rows, sha)
    fmt = Format.parse("w4a16-sym-pc")
    rec = Recipe(
        steps=[REGISTRY["gptq"](impl="torch")],
        calib=FileSource(path=str(tmp_path / "c.jsonl"), samples=8, seqlen=16),
    )
    plan = compile_plan(str(tmp_path / "g"), rec, fmt, Target(), CPU)
    assert "per-layer inputs" in dict(plan.residency.peak_terms)

    meta = load_model(str(tmp_path / "g"), weights=False, dtype=torch.bfloat16, target=Target())
    graph = describe(meta, verify=False)
    per_sample = per_layer_inputs_bytes(meta, len(graph.layers), 16, 2)
    assert per_sample == 16 * 2 * 16 * 2  # seq x layers x width x bf16
    terms, _ = memory_terms(
        meta, graph, rec.steps, samples=8, seqlen=16, batch_size=1, world=1, fmt=fmt, esz=2
    )
    got = {t.name: t.bytes for t in terms}
    assert got["per-layer inputs"] == 8 * per_sample
    assert got["call arguments"] == 8 * 5 * 2**20  # the flat term keeps its own value

    # The tensor is [batch, seq, layers, width], so two samples per forward retain exactly as much as two
    # forwards of one; the batch dimension multiplies the term instead of replacing the sample count.
    terms, _ = memory_terms(
        meta, graph, rec.steps, samples=8, seqlen=16, batch_size=2, world=1, fmt=fmt, esz=2
    )
    assert {t.name: t.bytes for t in terms}["per-layer inputs"] == 8 * per_sample
    # ...a partial last batch is not rounded up to a full one (5 samples in batches of 2)...
    terms, _ = memory_terms(
        meta, graph, rec.steps, samples=5, seqlen=16, batch_size=2, world=1, fmt=fmt, esz=2
    )
    assert {t.name: t.bytes for t in terms}["per-layer inputs"] == 5 * per_sample
    # ...and on two ranks a card counts only its own batches (4 batches over 2 ranks = 2 each = 4 samples).
    terms, _ = memory_terms(
        meta, graph, rec.steps, samples=8, seqlen=16, batch_size=2, world=2, fmt=fmt, esz=2
    )
    assert {t.name: t.bytes for t in terms}["per-layer inputs"] == 4 * per_sample

    qwen = tiny_qwen3(layers=2)  # no per-layer embeddings: no such term, and nothing to add per batch
    qwen.save_pretrained(tmp_path / "q")
    qmeta = load_model(str(tmp_path / "q"), weights=False, dtype=torch.bfloat16, target=Target())
    assert per_layer_inputs_bytes(qmeta, 2, 16, 2) == 0


def test_a_group_size_that_still_divides_plans(tiny):
    """A size the group-size refusal names as working still plans, for the body and for an expert bank."""
    tiny_qwen3_moe(layers=1).save_pretrained(tiny / "moe")
    rec = recipe(tiny, REGISTRY["rtn"]())
    body = compile_plan(str(tiny / "m"), rec, Format.parse("w4a16-sym-g32"), Target(), CPU)
    bank = compile_plan(str(tiny / "moe"), rec, Format.parse("w4a16-sym-g32"), Target(), CPU)
    assert body.fmt.group_size == 32 and bank.fmt.group_size == 32


@pytest.mark.parametrize("invalid", ["step", "fingerprint", "vocab"])
def test_plan_validates_before_cuda_memory_query(tiny, monkeypatch, invalid):
    def no_cuda(*args, **kwargs):
        pytest.fail("invalid plans must not initialize CUDA through a memory query")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_cuda)
    rec = recipe(tiny, NeverHere() if invalid == "step" else TinyRTN())
    if invalid == "fingerprint":
        (tiny / "c.jsonl").write_text(json.dumps({"input_ids": [1] * 20}) + "\n")
    elif invalid == "vocab":
        sha = json.loads((tiny / "c.jsonl").read_text().splitlines()[0])["tokenizer"]
        write_ids(tiny / "c.jsonl", [[1, 500] + [3] * 20] * 4, sha)
    with pytest.raises(
        Refused, match={"step": "not on this target", "fingerprint": "fingerprint", "vocab": "vocab"}[invalid]
    ):
        compile_plan(str(tiny / "m"), rec, Format.parse("w4a16"), Target(), [torch.device("cuda:0")])


def _output_build(tiny, out_dir, writer=FakeWriter, log=lambda s: None):
    return build(str(tiny / "m"), recipe(tiny, TinyRTN()), format="w4a16-sym-g128", out_dir=out_dir,
                 devices=["cpu"], dtype=torch.float32, exporter=writes(writer), log=log)  # fmt: skip


@pytest.mark.parametrize("given", ["out", "."])
def test_output_that_holds_anything_is_refused_and_kept(tiny, monkeypatch, given):
    """The refusal comes before any work and removes nothing, whichever way the path is written."""
    out = tiny / "out"
    out.mkdir()
    (out / "mine.txt").write_text("mine")
    monkeypatch.chdir(out if given == "." else tiny)
    FakeWriter.last = None
    with pytest.raises(Refused, match=str(out)):
        _output_build(tiny, given)
    assert FakeWriter.last is None and (out / "mine.txt").read_text() == "mine"
    assert not list(tiny.glob(".out.partial-*"))


@pytest.mark.parametrize("given", [".", "abs"])
def test_empty_current_directory_is_refused_and_stays_usable(tiny, monkeypatch, given):
    """Publishing would replace the directory under the shell, leaving it in a deleted one."""
    cwd = tiny / "here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    FakeWriter.last = None
    with pytest.raises(Refused, match="current directory"):
        _output_build(tiny, "." if given == "." else cwd)
    assert FakeWriter.last is None and Path().resolve() == cwd and Path("rel.txt").write_text("ok")
    assert not list(tiny.glob(".here.partial-*"))


@pytest.mark.parametrize("name", ["q" * 240, "q" + "é" * 119], ids=["ascii", "cut-inside-a-character"])
def test_output_name_near_the_filesystem_limit_builds(tiny, name):
    """The staging suffix must not push a valid name over the limit, or leave a path that cannot print."""
    lines = []
    _output_build(tiny, tiny / name, log=lines.append)
    assert (tiny / name / "build.json").is_file() and not list(tiny.glob(".*.partial-*"))
    "".join(lines).encode()


def test_output_name_over_the_filesystem_limit_fails_before_any_work(tiny):
    """Shortening the staging name must not move this failure to the rename, after all the work."""
    FakeWriter.last = None
    with pytest.raises(OSError, match="too long"):
        _output_build(tiny, tiny / ("q" * 256))
    assert FakeWriter.last is None and not list(tiny.glob(".*.partial-*"))


def test_failed_build_publishes_nothing_and_keeps_its_staging(tiny):
    """A failure leaves no directory at the output, and names the staging directory with the partial work."""

    class Failing(FakeWriter):
        def finish(self, config, recipe, build_info):
            (self.out / "model-00001.safetensors").write_text("partial")
            raise OSError("disk full")

    lines = []
    with pytest.raises(OSError, match="disk full"):
        _output_build(tiny, tiny / "out", Failing, lines.append)
    assert not (tiny / "out").exists()
    [left] = tiny.glob(".out.partial-*")
    assert (left / "model-00001.safetensors").read_text() == "partial"
    assert str(left) in lines[-1]


def test_output_that_appears_during_the_build_is_kept(tiny):
    """What someone put at the output during a build survives publishing; the checkpoint stays in staging."""

    class Racing(FakeWriter):
        def finish(self, config, recipe, build_info):
            (tiny / "out").mkdir()
            (tiny / "out" / "theirs.txt").write_text("theirs")
            return super().finish(config, recipe, build_info)

    with pytest.raises(Refused, match="finished checkpoint stays in"):
        _output_build(tiny, tiny / "out", Racing)
    assert [p.name for p in (tiny / "out").iterdir()] == ["theirs.txt"]
    [left] = tiny.glob(".out.partial-*")
    assert (left / "build.json").is_file() and (left / "resolved.yaml").is_file()


def test_a_build_from_a_downloaded_model_names_the_original_repo(tiny):
    """A prepared download records its repo in the model directory; any recipe built from it carries it.

    The agent's own recipes name no ``meta``, so the record in ``<out>/model`` is what keeps the serve name
    the original's (`Qwen3-8B-W4A16`) instead of the directory's (``model-W4A16``).
    """
    from qda_quant.config import SOURCE_RECORD
    from qda_quant.output.serve_command import served_name

    (tiny / "m" / SOURCE_RECORD).write_text(json.dumps({"repo": "Qwen/Qwen3-8B", "revision": "c0ffee"}))
    plain = recipe(tiny, Marker(), TinyRTN())
    assert plain.meta == {}  # an agent's recipe: no meta of its own
    out = build(
        str(tiny / "m"),
        plain,
        format="w4a16-sym-g128",
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    assert json.loads((out / "build.json").read_text())["source_repo"] == "Qwen/Qwen3-8B"
    assert served_name(out).startswith("Qwen3-8B-W4A16")

    # the recipe's own meta is the explicit override
    named = Recipe(
        steps=list(recipe(tiny, Marker(), TinyRTN()).steps),
        calib=FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16),
        meta={"source_repo": "Other/Model"},
    )
    out2 = build(
        str(tiny / "m"),
        named,
        format="w4a16-sym-g128",
        out_dir=tiny / "out2",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    assert json.loads((out2 / "build.json").read_text())["source_repo"] == "Other/Model"


def test_build_end_to_end(tiny):
    Marker.calls.clear()
    out = build(
        str(tiny / "m"),
        recipe(tiny, Marker(), TinyRTN()),
        format="w4a16-sym-g128",
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    info = json.loads((out / "build.json").read_text())
    assert Marker.calls == [4]
    assert [i for i, _ in FakeWriter.last.layers] == [0, 1]
    assert len(FakeWriter.last.layers[0][1]) == 7
    assert info["card_count"] == 1 and info["load_to_saved_s"] > 0
    assert {"load", "capture", "calib_forward", "steps", "propagate", "busy:calib"} <= set(
        info["timings"]["phases"]
    )
    # a recipe that solves nothing still records the empty offsets summary
    assert info["solves"] == {"limit": 2**31, "max_offset": 0, "max_factor_offset": 0, "entries": []}


def test_build_records_every_solve_offset(tiny):
    """A later offset audit is answered from build.json: each solve's E/R/C, strides and max offset."""
    out = build(
        str(tiny / "m"),
        recipe(tiny, REGISTRY["gptq"](impl="torch")),
        format="w4a16-sym-g32",
        out_dir=tiny / "gptq",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    record = json.loads((out / "build.json").read_text())["solves"]
    entries = record["entries"]
    assert record["limit"] == 2**31 and entries
    assert record["max_offset"] == max(e["max_offset"] for e in entries)
    assert record["max_factor_offset"] == max(e["u_max_offset"] for e in entries)
    sizes = [max(e["max_offset"], e["u_max_offset"]) for e in entries]
    assert sizes == sorted(sizes, reverse=True)  # largest first, by either path
    for e in entries:
        s0, s1 = e["strides"]
        assert e["max_offset"] == (e["experts"] - 1) * s0 + (e["rows"] - 1) * s1 + e["cols"] - 1
        assert e["max_offset"] < record["limit"] and e["impl"] == "torch" and e["label"]
        u0, u1, u2 = e["u_strides"]
        assert e["u_max_offset"] == (e["cols"] - 1) * u1 + (e["cols"] - 1) * u2
        assert e["u_max_offset"] < record["limit"]
    assert all(".layers." in e["label"] or "layers." in e["label"] for e in entries)


def test_build_releases_its_digest_buffer(tiny, monkeypatch):
    """The pinned copy buffer is the size of the largest weight, and a finished build does not keep it."""
    from qda_quant.intake import fingerprint

    monkeypatch.setattr(fingerprint, "_PINNED", [torch.empty(16, dtype=torch.uint8)])
    build(
        str(tiny / "m"),
        recipe(tiny, Marker(), TinyRTN()),
        format="w4a16-sym-g128",
        out_dir=tiny / "released",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    assert fingerprint._PINNED[0] is None


def test_build_refuses_tensors_its_writer_cannot_take(tiny):
    """A writer implementing the documented contract (add_layer, finish) publishes nothing it cannot write."""
    with pytest.raises(Refused) as refused:
        build(
            str(tiny / "m"),
            recipe(tiny, Toucher(), TinyRTN()),
            format="w4a16-sym-g128",
            out_dir=tiny / "out",
            devices=["cpu"],
            dtype=torch.float32,
            exporter=writes(FakeWriter),
            log=lambda s: None,
        )
    message = str(refused.value)
    assert "exporter 'test_writer'" in message and "model.embed_tokens.weight" in message
    assert "add_source_tensors" in message
    assert not (tiny / "out").exists()


def test_build_hands_tensors_outside_the_graph_to_the_writer(tiny):
    class Taker(FakeWriter):
        got = None

        def add_source_tensors(self, tensors):
            Taker.got = {k: v.clone() for k, v in tensors.items()}

    Taker.got = None
    build(
        str(tiny / "m"),
        recipe(tiny, Toucher(), TinyRTN()),
        format="w4a16-sym-g128",
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(Taker),
        log=lambda s: None,
    )
    assert Taker.got is not None and "model.embed_tokens.weight" in Taker.got


def _written_bytes(out: Path) -> int:
    """The checkpoint's size: every file the build wrote, but not the logs it writes beside them."""
    return sum(
        p.stat().st_size
        for p in out.iterdir()
        if p.is_file() and p.name not in ("build.json", "resolved.yaml")
    )


@pytest.mark.parametrize("head", [None, "w4a16-sym-g64"])
def test_plan_predicts_the_checkpoint_size(tiny, head):
    """Plan-time size of the checkpoint a build writes, within 2% (a packed head too, in its own format)."""
    from qda_quant import REGISTRY

    fmt = Format.parse("w4a16-asym-g32")
    steps = [REGISTRY["rtn"]()] + ([REGISTRY["head"](format=head)] if head else [])
    plan = compile_plan(str(tiny / "m"), recipe(tiny, *steps), fmt, Target(), CPU, dtype=torch.float32)
    out = build(
        str(tiny / "m"),
        recipe(tiny, *[REGISTRY["rtn"]()] + ([REGISTRY["head"](format=head)] if head else [])),
        format=fmt.name,
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        log=lambda s: None,
    )
    written = _written_bytes(out)
    assert abs(plan.output_bytes - written) / written < 0.02
    # the plan predicts; the record holds what the build actually wrote
    assert json.loads((out / "build.json").read_text())["output_bytes"] == written


def _config(tiny, steps, fmt="w4a16-asym-g32", out="out"):
    from qda_quant.config import BuildConfig

    return BuildConfig.from_dict(
        {
            "model": str(tiny / "m"),
            "format": fmt,
            "devices": ["cpu"],
            "steps": steps,
            "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
            "out": str(tiny / out),
        }
    )


def test_a_recipe_that_stores_no_codes_is_refused(tiny):
    """AWQ alone only folds scales, so the checkpoint would be bf16 under a W4A8 receipt.

    `qda plan` refuses before a weight loads, and the build publishes nothing: the receipt never claims a
    format the checkpoint does not store.
    """
    with pytest.raises(Refused, match="no step stores codes"):
        _config(tiny, [{"awq": {}}]).plan()
    with pytest.raises(Refused, match="no step stores codes"):
        _config(tiny, [{"awq": {}}]).build(log=lambda _: None)
    assert not (tiny / "out").exists()


def test_a_quantizing_step_after_awq_writes_and_records_the_packed_checkpoint(tiny):
    """The same recipe with `rtn` after `awq`: a packed W4A8 checkpoint, and its size as written."""
    out = _config(tiny, [{"awq": {}}, {"rtn": {}}]).build(log=lambda _: None)
    weights = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert sum(k.endswith(".weight_packed") for k in weights) > 0
    assert json.loads((out / "build.json").read_text())["output_bytes"] == _written_bytes(out)


def test_a_build_reports_an_abandoned_norm_fold_with_both_routes(tiny, monkeypatch):
    """A step reading norm relations gets the fp64 check; a fold it abandons is reported, naming both routes."""
    import dataclasses

    import qda_quant.arch

    real = qda_quant.arch.describe
    relation = "model.layers.0.input_layernorm -> model.layers.0.self_attn.q_proj"

    def with_dropped(model, *, verify=True, keep=qda_quant.arch.DEFAULT_KEEP):
        graph = real(model, verify=verify, keep=keep)
        if verify:  # the shape `verify.py` leaves on the graph when `_fold_ok` restores and abandons a fold
            graph = dataclasses.replace(graph, extra={**graph.extra, "dropped": [relation]})
        return graph

    monkeypatch.setattr(qda_quant.arch, "describe", with_dropped)
    lines: list[str] = []
    _config(tiny, [{"awq": {}}, {"rtn": {}}]).build(log=lines.append)
    text = "\n".join(lines)
    assert text.count(relation) == 1  # once per relation, once per build
    assert "qda: warning: norm fold abandoned" in text
    assert "qda.arch_adapters" in text and "open an issue" in text
    assert "numerically unaffected" in text
    assert "degrad" not in text and "corrupt" not in text

    lines.clear()  # a plain GPTQ recipe never verifies, so it has no fold to report
    _config(tiny, [{"gptq": {}}, {"rtn": {}}], out="out-gptq").build(log=lines.append)
    assert not any("norm fold" in line for line in lines)


def test_plan_predicts_a_decoded_untouched_weight(tiny):
    """An encoded weight no step touches (a draft's MTP layer) counts at its decoded size: within 2%."""
    from qda_quant import REGISTRY
    from qda_quant.intake.source import SOURCE_CODECS
    from safetensors.torch import load_file, save_file

    src = tiny / "m"
    stored = load_file(src / "model.safetensors")
    codec = SOURCE_CODECS["fp8_block"]

    def encode(key):
        w = stored[key].float()
        scale = torch.full((-(-w.shape[0] // 128), -(-w.shape[1] // 128)), 0.004123)
        grid = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        stored[key] = (w / grid).to(torch.float8_e4m3fn)
        stored[codec.parts(key)[1]] = scale

    for key in ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.self_attn.q_proj.weight"):
        encode(key)
    mtp = "mtp.layers.0.proj.weight"  # the model class holds no such layer
    stored[mtp] = 0.001 * torch.randn(32, 64, generator=torch.Generator().manual_seed(0))
    encode(mtp)
    save_file(stored, src / "model.safetensors", metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (src / "config.json").write_text(json.dumps(cfg))

    fmt = Format.parse("w4a16-sym-g32")
    plan = compile_plan(str(src), recipe(tiny, REGISTRY["rtn"]()), fmt, Target(), CPU, dtype=torch.float32)
    out = build(
        str(src),
        recipe(tiny, REGISTRY["rtn"]()),
        format=fmt.name,
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        log=lambda s: None,
    )
    written = _written_bytes(out)
    assert abs(plan.output_bytes - written) / written < 0.02
    info = json.loads((out / "build.json").read_text())
    assert info["kept_decoded"]["modules"] == [mtp.removesuffix(".weight")]


def test_plan_sizes_native_gpt_oss_experts_as_their_int4_export(tmp_path):
    """MXFP4 blocks/scales retire through the adapter layout, just like the float expert tensors."""
    from qda_quant import REGISTRY
    from safetensors.torch import load_file, save_file

    from ..engine_util import tiny_gpt_oss

    src = tmp_path / "m"
    tiny_gpt_oss().to(torch.bfloat16).save_pretrained(src)
    fp = save_tiny_tokenizer(src)
    write_ids(tmp_path / "calib.jsonl", [[1, 2, 3, 4, 5, 6, 7, 8]], fp)
    fmt = Format.parse("w4a16-sym-g64")
    float_plan = compile_plan(str(src), recipe(tmp_path, REGISTRY["rtn"]()), fmt, Target(), CPU)
    stored = load_file(src / "model.safetensors")
    for name in list(stored):
        if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            continue
        experts, columns, rows = stored.pop(name).shape
        stored[name + "_blocks"] = torch.zeros(experts, rows, columns // 32, 16, dtype=torch.uint8)
        stored[name + "_scales"] = torch.full((experts, rows, columns // 32), 127, dtype=torch.uint8)
    save_file(stored, src / "model.safetensors", metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "mxfp4"}
    (src / "config.json").write_text(json.dumps(cfg))
    native_plan = compile_plan(str(src), recipe(tmp_path, REGISTRY["rtn"]()), fmt, Target(), CPU)
    # The input encoding changes; the output packed weights and biases do not.
    assert abs(native_plan.output_bytes - float_plan.output_bytes) < 1024


@pytest.mark.parametrize("source_type", [FileSource, VLMSource])
def test_data_free_build_never_reads_calibration(tiny, monkeypatch, source_type):
    import importlib

    from qda_quant.engine import calib

    engine_execute = importlib.import_module("qda_quant.engine.execute")

    def fail(*args, **kwargs):
        pytest.fail("data-free build accessed calibration")

    monkeypatch.setattr(calib, "load_ids", fail)
    monkeypatch.setattr(calib, "load_with_inputs", fail)
    monkeypatch.setattr(calib, "batches", fail)
    monkeypatch.setattr(engine_execute, "_capture", fail)
    monkeypatch.setattr(source_type, "pending", fail)
    monkeypatch.setattr(source_type, "read", fail)
    source = source_type(**({"path": str(tiny / "missing.jsonl")} if source_type is FileSource else {}))
    rec = Recipe(steps=[NoStatsRTN()], calib=source)
    plan = compile_plan(str(tiny / "m"), rec, Format.parse("w4a16-sym-g128"), Target(), CPU)
    assert plan.calib_ids == []
    assert plan.calib_inputs is None
    assert plan.generate is None
    out = build(
        str(tiny / "m"),
        rec,
        format="w4a16-sym-g128",
        out_dir=tiny / "free-out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda _: None,
    )
    info = json.loads((out / "build.json").read_text())
    assert "capture" not in info["timings"]["phases"]
    assert "calib_forward" not in info["timings"]["phases"]
    assert "propagate" not in info["timings"]["phases"]


@pytest.mark.parametrize("method", ["guided", "fisher_trace"])
def test_token_weight_model_step_declares_calibration(tiny, method):
    from qda_quant import REGISTRY

    plan = compile_plan(
        str(tiny / "m"),
        recipe(tiny, REGISTRY[method](), REGISTRY["gptq"]()),
        Format.parse("w4a16-sym-g32"),
        Target(),
        CPU,
    )
    assert len(plan.calib_ids) == 4


@pytest.mark.parametrize(
    ("moe", "later"),
    [(False, ["rtn"]), (False, ["awq", "rtn"]), (False, ["autoround"]), (True, ["awq", "rtn"])],
)
def test_token_weights_no_later_step_reads_are_refused(tiny, moe, later):
    """The weights reach Hessians only; AWQ declares the expert Hessian but only rescales it for a later step."""
    from qda_quant import REGISTRY

    model = tiny / "m"
    if moe:
        tiny_qwen3_moe(layers=1).save_pretrained(model := tiny / "moe")
        save_tiny_tokenizer(model)  # the calibration ids' tokenizer
    steps = [REGISTRY["fisher_trace"](), *(REGISTRY[name]() for name in later)]
    with pytest.raises(Refused, match="no later step reads a token-weighted statistic"):
        compile_plan(str(model), recipe(tiny, *steps), Format.parse("w4a16-sym-g32"), Target(), CPU)
    control = recipe(tiny, *steps[:-1], REGISTRY["gptq"]())  # a later GPTQ reads the weighted Hessians
    compile_plan(str(model), control, Format.parse("w4a16-sym-g32"), Target(), CPU)


@pytest.mark.parametrize("residency", ["resident", "stream"])
def test_data_free_rtn_matches_reference_checkpoint(tiny, monkeypatch, residency):
    import hashlib
    import importlib

    from qda_quant import REGISTRY
    from qda_quant.engine import calib

    engine_build = importlib.import_module("qda_quant.engine.build")
    engine_execute = importlib.import_module("qda_quant.engine.execute")
    out = tiny / "equivalent"
    rec = recipe(tiny, REGISTRY["rtn"]())
    options = dict(
        format="w4a16-sym-g32", out_dir=out, devices=["cpu"], residency=residency, log=lambda _: None
    )

    def checksums():
        return {
            str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in out.rglob("*")
            if path.is_file() and path.name != "build.json"
        }

    build(str(tiny / "m"), rec, **options)
    fast = checksums()
    out.rename(tiny / "fast")
    compile_reference = engine_build.compile_plan
    pass_reference = engine_execute._layer_pass

    def reference_plan(*args, **kwargs):
        plan = compile_reference(*args, **kwargs)
        plan.calib_ids = calib.load_ids(rec.calib, str(tiny / "m"))
        return plan

    def reference_pass(run, steps, writer, graph, **kwargs):
        run.local_batches = calib.batches(run.calib_ids, run.batch_size)
        return pass_reference(run, steps, writer, graph, calibration=True)

    monkeypatch.setattr(engine_build, "compile_plan", reference_plan)
    monkeypatch.setattr(engine_execute, "_layer_pass", reference_pass)
    build(str(tiny / "m"), rec, **options)
    assert checksums() == fast
    reference_info = json.loads((out / "build.json").read_text())
    assert "calib_forward" in reference_info["timings"]["phases"]


@pytest.mark.parametrize("method", ["guided", "fisher_trace"])
@pytest.mark.parametrize("residency", ["resident", "stream"])
def test_token_weight_build_uses_one_anchor_forward_per_batch(tiny, monkeypatch, method, residency):
    """A token-weighted GPTQ build consumes engine batches once for its anchor calculation."""
    from qda_quant import REGISTRY
    from qda_quant.engine import calib, token_weights

    helper = getattr(token_weights, method)
    forwards = []
    backwards = []
    consumed = []
    engine_calls = {"load_ids": 0, "batches": 0}
    residency_calls = []

    def count_engine_call(name, original):
        def counted(*args, **kwargs):
            engine_calls[name] += 1
            return original(*args, **kwargs)

        return counted

    original_residency = token_weights.anchor_residency
    original_loss = token_weights._loss

    def checked_loss(*args, **kwargs):
        loss = original_loss(*args, **kwargs)
        loss.register_hook(lambda _gradient: backwards.append(1))
        return loss

    def checked_residency(*args, **kwargs):
        residency_calls.append(kwargs["store"].name)
        return original_residency(*args, **kwargs)

    for name in engine_calls:
        monkeypatch.setattr(calib, name, count_engine_call(name, getattr(calib, name)))
    monkeypatch.setattr(token_weights, "anchor_residency", checked_residency)
    monkeypatch.setattr(token_weights, "_loss", checked_loss)

    def checked(model, batches, *args, **kwargs):
        expected = list(batches)

        def counted_batches():
            for ids in expected:
                consumed.append(ids)
                yield ids

        handle = model.register_forward_pre_hook(lambda _module, _args: forwards.append(1))
        try:
            weights = helper(model, counted_batches(), *args, **kwargs)
        finally:
            handle.remove()
        assert len(consumed) == len(forwards) == len(expected)
        assert len(weights) == len(expected)
        return weights

    monkeypatch.setattr(token_weights, method, checked)
    out = build(
        str(tiny / "m"),
        recipe(tiny, REGISTRY[method](), REGISTRY["gptq"]()),
        format="w4a16-sym-g32",
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        residency=residency,
        exporter=writes(FakeWriter),
        log=lambda _: None,
    )
    assert len(forwards) == 4
    assert len(backwards) == (4 if method == "guided" else 0)
    assert engine_calls == {"load_ids": 1, "batches": 1}
    assert residency_calls == [residency]
    info = json.loads((out / "build.json").read_text())
    assert info["timings"]["phases"][f"model_step:{method}"] > 0


@pytest.mark.parametrize("method", ["guided", "fisher_trace"])
def test_token_weight_methods_refuse_multiple_cards_at_plan_time(tiny, monkeypatch, method):
    """Row-aligned weights are refused before calibration or residency work on multiple cards."""
    from qda_quant import REGISTRY
    from qda_quant.engine import calib

    def no_calibration(*args, **kwargs):
        pytest.fail("multi-card token-weight refusal must precede calibration loading")

    monkeypatch.setattr(calib, "load_ids", no_calibration)
    with pytest.raises(Refused, match="computes calibration-row weights and currently requires one card"):
        compile_plan(
            str(tiny / "m"),
            recipe(tiny, REGISTRY[method](), REGISTRY["gptq"]()),
            Format.parse("w4a16-sym-g32"),
            Target(),
            CPU * 2,
        )


@pytest.mark.parametrize(
    "method,alpha,keep_float", [("gptq", None, False), ("gptaq", 0.0, False), ("gptaq", 0.3, True)]
)
def test_build_allocates_float_stream_only_for_cross_hessian(tiny, monkeypatch, method, alpha, keep_float):
    """Only a nonzero refit's declared cross-Hessian need allocates the paired float inputs."""
    import importlib

    from qda_quant import REGISTRY

    engine = importlib.import_module("qda_quant.engine.execute")
    original = engine.capture
    captured = []

    def capture_checked(*args, **kwargs):
        slots = original(*args, **kwargs)
        assert kwargs["keep_float"] is keep_float
        assert all((slot.float_h is not None) is keep_float for slot in slots)
        captured.extend(slots)
        return slots

    monkeypatch.setattr(engine, "capture", capture_checked)
    step = REGISTRY[method](**({} if alpha is None else {"alpha": alpha}))
    build(
        str(tiny / "m"),
        recipe(tiny, step),
        format="w4a16-sym-g32",
        out_dir=tiny / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda _: None,
    )
    assert captured


def test_plan_budgets_both_gptaq_activation_streams(tiny, monkeypatch):
    """The residency budget doubles activation storage exactly when the recipe needs C."""
    from qda_quant import REGISTRY
    from qda_quant.engine import residency

    original = residency.plan
    sizes = []

    def checked(*args, **kwargs):
        sizes.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(residency, "plan", checked)
    for step in (REGISTRY["gptq"](), REGISTRY["gptaq"](alpha=0), REGISTRY["gptaq"]()):
        compile_plan(str(tiny / "m"), recipe(tiny, step), Format.parse("w4a16-sym-g32"), Target(), CPU)
    assert sizes[0] > 0
    assert sizes[0] == sizes[1]
    assert sizes[2] == 2 * sizes[0]


def test_plan_budgets_kept_layer_output(tiny, monkeypatch):
    """A step that keeps the layer outputs adds a retained stream, beside the offloadable ones."""
    from qda_quant import REGISTRY
    from qda_quant.engine import residency

    original = residency.plan
    sizes = []
    retained = []

    def checked(*args, **kwargs):
        sizes.append(args[2])
        retained.append(kwargs["retained_bytes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(residency, "plan", checked)
    for steps in (
        (REGISTRY["gptq"](),),
        (REGISTRY["autoround"](target="quant_stream"),),  # needs layer_io: the engine keeps s.out on the card
        (REGISTRY["autoround"](target="float_stream"),),  # needs layer_io_float: the float stream only
        (REGISTRY["gptaq"](alpha=0.5), REGISTRY["autoround"](target="quant_stream")),  # both streams
    ):
        compile_plan(str(tiny / "m"), recipe(tiny, *steps), Format.parse("w4a16-sym-g32"), Target(), CPU)
    base = sizes[0]
    assert base > 0
    assert (sizes[1], retained[1]) == (base, base)
    assert (sizes[2], retained[2]) == (2 * base, 0)
    assert (sizes[3], retained[3]) == (2 * base, base)


def test_plan_counts_the_inputs_a_parent_search_keeps():
    """Every retained token and the search's float32 batch are peak terms; a bounded sample is capped."""
    from qda_quant import REGISTRY
    from qda_quant.engine.planning import memory_terms

    from ..engine_util import graph_of

    model = tiny_qwen3(layers=1)
    graph = graph_of(model)
    groups = graph.layers[0].groups
    widths = [model.get_submodule(g.modules[0]).in_features for g in groups]
    attn_in = next(g for g in groups if g.kind == "attn_in")
    qkv = sum(model.get_submodule(m).out_features for m in attn_in.modules)
    hidden = model.config.hidden_size
    kw = dict(samples=128, seqlen=512, batch_size=2, world=1, fmt=Format.parse("w4a16-sym-g32"))
    parent, _ = memory_terms(model, graph, [REGISTRY["awq"]()], **kw)
    terms = {t.name: t.bytes for t in parent}
    assert terms["retained inputs"] == 128 * 512 * sum(widths) * 2  # every token, in the model dtype
    assert terms["search copy"] == 2 * 512 * max(widths) * 4  # one float32 batch of the widest group
    # The reference list and one candidate's list, each priced through its parent block (q/k/v and
    # gate/up), whose output is the residual's width -- not the concatenation of their own linears.
    assert terms["search outputs"] == 2 * 128 * 512 * hidden * 2
    assert qkv > hidden  # the parent path is the narrower one here, so the assertion above is a real choice
    # The search's copies of the group's weights (`w0` bf16 and `wcat`/`wn` float32) and, per candidate,
    # one linear's `ws` and its two float32 results plus the dtype copy; the widest group is gate+up.
    elems = {
        g.kind: [model.get_submodule(m).weight.numel() for m in g.modules]
        for g in groups
        if g.norm is not None or g.prev is not None
    }
    widest, widest_mod = max(map(sum, elems.values())), max(map(max, elems.values()))
    assert sum(elems["mlp_in"]) == widest  # gate+up is the widest group
    assert terms["search weights"] == widest * (2 + 8) + widest_mod * 14
    bounded, _ = memory_terms(model, graph, [REGISTRY["awq"](objective="linear_out")], **kw)
    terms = {t.name: t.bytes for t in bounded}
    assert terms["retained inputs"] == 8192 * sum(widths) * 2  # the 8192-row sample, not every token
    assert terms["search copy"] == 2 * 512 * max(widths) * 4
    # ...and q/k/v are then priced by their own linears: their concatenation is the widest output list.
    assert terms["search outputs"] == 2 * 8192 * qkv * 2


def test_residency_reserves_the_kept_output_against_the_weights(monkeypatch):
    """A kept output stays on the card, so it must not let a 60 GiB model go resident on an 80 GiB card."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80 * gib, 80 * gib))
    got = residency.plan(
        60 * gib, 1 * gib, 35 * gib, [torch.device("cuda:0")], retained_bytes=35 * gib, host_free=1024 * gib
    )
    assert got.weights == "stream"
    assert got.act_gib == 70.0
    assert got.peak_gib == 71.0  # one streamed layer plus the kept outputs and the hidden states


def test_residency_plans_disk_for_a_model_bigger_than_host_memory(monkeypatch):
    """`disk` holds one layer at a time, so a model larger than host memory still plans to it."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (140 * gib, 140 * gib))
    got = residency.plan(2000 * gib, 20 * gib, 0, [torch.device("cuda:0")], host_free=1024 * gib)
    assert got.weights == "disk"
    assert got.peak_gib == 20.0


def test_residency_solves_a_unit_larger_than_the_card_one_chunk_at_a_time(monkeypatch):
    """A solve that does not fit fills what the card leaves after the headroom, and counts its experts."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 * gib, 40 * gib))
    card = [torch.device("cuda:0")]
    chunked = residency.Solve(
        "solver workspace", experts=100, workspace=gib, factor_batch=16, buffers=2 * gib
    )
    got = residency.plan(10 * gib, 1 * gib, 0, card, solves=[chunked])
    # 40 GiB free, 10 GiB weights: 2 GiB buffers + 26 experts would fill the 2 GiB headroom's room (38 GiB), but
    # that plan's card peak (41.9 GiB with the CUDA context and allocator reserve) is above the card, so the plan
    # keeps its 3.9 GiB overhead free instead: 2 GiB buffers + 24 experts, card peak 39.7 GiB
    assert dict(got.peak_terms)["solver workspace, 24 of 100 at a time"] == 26.0
    assert got.peak_gib == 36.0
    assert got.card_gib <= 40.0
    whole = residency.Solve("solver workspace", experts=20, workspace=gib, factor_batch=16)
    got = residency.plan(10 * gib, 1 * gib, 0, card, solves=[whole])
    assert dict(got.peak_terms)["solver workspace"] == 20.0  # a unit that fits is solved in one call
    assert got.peak_gib == 30.0
    # another unit's codes stay on the card beside this one's chunk, and get a term of their own
    other = residency.Solve("solver workspace", experts=8, workspace=gib // 4, factor_batch=16, buffers=gib)
    got = residency.plan(10 * gib, 1 * gib, 0, card, solves=[chunked, other])
    terms = dict(got.peak_terms)
    assert terms["solver workspace, 23 of 100 at a time"] == 25.0  # the other unit's 1 GiB leaves 23 experts
    assert terms["write-back buffers"] == 1.0
    assert sum(terms.values()) == got.peak_gib == 36.0


def test_residency_refuses_a_chunk_whose_padded_factorization_does_not_fit(monkeypatch):
    """A chunk is sized by memory, but its factorization pads to 16 members, so a small chunk still needs 16."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * gib, 20 * gib))
    padded = residency.Solve("solver workspace", experts=100, workspace=2 * gib, factor_batch=16)
    with pytest.raises(Refused, match=r"cuda:0: the build needs 33\.0 GiB .*8 of 100 experts at a time"):
        residency.plan(10 * gib, 1 * gib, 0, [torch.device("cuda:0")], solves=[padded])


def test_residency_refuses_the_peak_of_the_least_free_card(monkeypatch):
    """A build that does not fit refuses at plan time, naming the card and the term that does not fit."""
    from qda_quant.engine import residency

    gib = 2**30
    free = {0: 80 * gib, 1: 24 * gib}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[device.index],) * 2)
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    with pytest.raises(
        Refused, match=r"cuda:1: the build needs 26\.0 GiB .*the solver workspace \(25\.0 GiB\)"
    ):
        residency.plan(60 * gib, 1 * gib, 0, devices, [residency.Term("solver workspace", 25 * gib)])


@pytest.mark.parametrize(
    ("free_gib", "model_gib", "layer_gib", "act_gib", "terms_gib", "workspace_gib"),
    [
        (77.7, 48.07, 1.6, 11.0, 13.7, 0.19),  # gemma-4-26b-a4b: a 64-expert bank chunked 26 at a time
        (78.7, 56.87, 1.19, 8.0, 9.88, 0.131),  # qwen3-30b-a3b: 29 at a time
    ],
)
def test_residency_leaves_headroom_before_keeping_the_whole_model(
    monkeypatch, free_gib, model_gib, layer_gib, act_gib, terms_gib, workspace_gib
):
    """The two plans that peaked within 0.1 GiB of free now pick stream, at half the peak."""
    from qda_quant.engine import residency

    gib = 2**30
    free = int(free_gib * gib)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, free))
    got = residency.plan(
        int(model_gib * gib),
        int(layer_gib * gib),
        int(act_gib * gib),
        [torch.device("cuda:0")],
        terms=[residency.Term("call arguments", int(terms_gib * gib))],
        solves=[residency.Solve("solver workspace", 64, int(workspace_gib * gib), 16)],
        host_free=1024 * gib,
    )
    assert got.weights == "stream"
    assert got.peak_gib < got.free_gib - 2


def test_residency_keeps_the_whole_model_with_ample_room(monkeypatch):
    """A plan whose peak leaves the headroom still keeps the model on the card."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80 * gib, 80 * gib))
    got = residency.plan(
        20 * gib,
        2 * gib,
        4 * gib,
        [torch.device("cuda:0")],
        terms=[residency.Term("call arguments", gib)],
        solves=[residency.Solve("solver workspace", 8, gib, 1)],
    )
    assert got.weights == "resident"
    assert got.peak_gib == 33.0  # weights 20, activations 4, call arguments 1, the whole solve 8


def test_residency_offloads_activations_before_refusing(monkeypatch):
    """Activations leave the card when they need the headroom, so the model still fits here."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10 * gib, 10 * gib))
    got = residency.plan(6 * gib, gib, int(7.5 * gib), [torch.device("cuda:0")], host_free=1024 * gib)
    assert (got.weights, got.activations) == ("resident", "host")
    assert got.peak_gib == 6.0


def test_residency_honours_an_explicit_store_and_warns_on_the_headroom(monkeypatch):
    """An explicit residency runs at the peak it had without the headroom, and warns."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (int(77.7 * gib),) * 2)
    with pytest.warns(UserWarning, match="headroom"):
        got = residency.plan(
            int(48.07 * gib),
            int(1.6 * gib),
            int(11.0 * gib),
            [torch.device("cuda:0")],
            terms=[residency.Term("call arguments", int(13.7 * gib))],
            solves=[residency.Solve("solver workspace", 64, int(0.19 * gib), 16)],
            force="resident",
            host_free=1024 * gib,
        )
    assert got.weights == "resident"
    assert got.peak_gib > got.free_gib - 2  # the resident plan uses the headroom


def test_gptaq_refuses_multiple_cards_before_calibration(tiny, monkeypatch):
    """Paired-stream calibration fails at plan time before loading calibration rows."""
    from qda_quant import REGISTRY
    from qda_quant.engine import calib

    def no_calibration(*args, **kwargs):
        pytest.fail("multi-card GPTAQ refusal must precede calibration loading")

    monkeypatch.setattr(calib, "load_ids", no_calibration)
    with pytest.raises(Refused, match="gptaq.*paired float-input calibration currently requires one card"):
        compile_plan(
            str(tiny / "m"),
            recipe(tiny, REGISTRY["gptaq"]()),
            Format.parse("w4a16-sym-g32"),
            Target(),
            CPU * 2,
        )


def test_calib_order(tiny):
    from qda_quant.engine.calib import load_ids

    rows = [json.loads(x)["input_ids"][:16] for x in (tiny / "c.jsonl").read_text().splitlines()[1:]]
    c0 = load_ids(FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16), tiny / "m")
    assert [r.tolist() for r in c0] == rows[:4]
    import random

    sh = rows.copy()
    random.Random(3).shuffle(sh)
    c3 = load_ids(FileSource(path=str(tiny / "c.jsonl"), samples=4, seqlen=16, seed=3), tiny / "m")
    assert [r.tolist() for r in c3] == sh[:4]


def test_calib_ids_need_matching_fingerprint(tiny):
    from qda_quant.engine.calib import load_ids

    rows = [[1, 2, 3, 4]] * 2
    (tiny / "bare.jsonl").write_text("\n".join(json.dumps({"input_ids": r}) for r in rows))
    with pytest.raises(Refused, match="no fingerprint"):
        load_ids(FileSource(path=str(tiny / "bare.jsonl"), samples=2, seqlen=4), tiny / "m")
    write_ids(tiny / "other.jsonl", rows, "0" * 64)
    with pytest.raises(Refused, match="token ids made by tokenizer 0000"):
        load_ids(FileSource(path=str(tiny / "other.jsonl"), samples=2, seqlen=4), tiny / "m")
    (tiny / "bare.jsonl.tokenizer").write_text(
        json.loads((tiny / "c.jsonl").read_text().splitlines()[0])["tokenizer"]
    )
    assert len(load_ids(FileSource(path=str(tiny / "bare.jsonl"), samples=2, seqlen=4), tiny / "m")) == 2


def test_calib_messages_cut(tiny):
    """Chat rows: template + tokenize; long rows truncated, short ones concatenated and cut."""
    from qda_quant.engine.calib import load_ids

    words = lambda n, k: " ".join(f"w{(i + k) % 120 + 3}" for i in range(n))  # noqa: E731
    convs = [
        [{"role": "user", "content": words(30, 0)}],
        [{"role": "user", "content": words(4, 1)}],
        [{"role": "user", "content": words(5, 2)}],
        [{"role": "user", "content": words(40, 3)}],
    ]
    (tiny / "msg.jsonl").write_text("\n".join(json.dumps({"messages": c}) for c in convs))
    got = [
        r.tolist()
        for r in load_ids(FileSource(path=str(tiny / "msg.jsonl"), samples=3, seqlen=8), tiny / "m")
    ]
    ids = lambda n, k: [1] + [(i + k) % 120 + 3 for i in range(n)] + [2]  # noqa: E731
    short = ids(4, 1) + ids(5, 2)  # 13 ids: one sample of 8, 5 left over
    assert got == [ids(30, 0)[:8], short[:8], ids(40, 3)[:8]]


def test_plan_refuses_unfingerprinted_ids(tiny):
    (tiny / "bare.jsonl").write_text(json.dumps({"input_ids": [1] * 20}) + "\n")
    r = Recipe(steps=[TinyRTN()], calib=FileSource(path=str(tiny / "bare.jsonl"), samples=1, seqlen=16))
    with pytest.raises(Refused, match="fingerprint"):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16"), Target(), CPU)


def test_resolve_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    assert resolve_devices([5]) == [torch.device("cuda:1")]
    assert resolve_devices([4, 5, 6]) == [torch.device(f"cuda:{i}") for i in range(3)]  # any N >= 1
    assert resolve_devices([0, 1]) == [torch.device("cuda:0"), torch.device("cuda:1")]  # none in the lease
    assert resolve_devices(["cpu"]) == CPU
    with pytest.raises(Refused, match="not visible"):
        resolve_devices([4, 5, 6, 7])  # four named, three visible


def test_build_json_covers_finish(tiny):
    import time as _t

    class SlowWriter(FakeWriter):
        def finish(self, config, recipe, build_info):
            _t.sleep(0.3)
            return super().finish(config, recipe, build_info)

    out = build(
        str(tiny / "m"),
        recipe(tiny, TinyRTN()),
        format="w4a16-sym-g128",
        out_dir=tiny / "o2",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(SlowWriter),
        log=lambda s: None,
    )
    info = json.loads((out / "build.json").read_text())
    assert info["load_to_saved_s"] >= info["timings"]["total"] + 0.3


def test_plan_refuses_negative_ids(tiny):
    sha = json.loads((tiny / "c.jsonl").read_text().splitlines()[0])["tokenizer"]
    write_ids(tiny / "neg.jsonl", [[1, -3] + [3] * 20], sha)
    r = Recipe(steps=[TinyRTN()], calib=FileSource(path=str(tiny / "neg.jsonl"), samples=1, seqlen=16))
    with pytest.raises(Refused, match=r"span \[-3"):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16"), Target(), CPU)


@pytest.mark.parametrize(
    "h",
    [
        {"dtype": "bf16"},
        {"dtype": "tf32", "buffer_rows": 0},
        {"dtype": "fp32"},
        {"dtype": "bf16", "buffer_rows": 0},
    ],
)
def test_hessian_setting_recorded_and_runs(tiny, h):
    from qda_quant.engine.stats import DEFAULT_HESSIAN

    out = build(
        str(tiny / "m"),
        recipe(tiny, TinyRTN()),
        format="w4a16-sym-g128",
        out_dir=tiny / f"h{len(h)}{h['dtype']}",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        hessian=h,
        log=lambda s: None,
    )
    want = {"dtype": h["dtype"], "buffer_rows": h.get("buffer_rows", DEFAULT_HESSIAN.buffer_rows)}
    assert json.loads((out / "build.json").read_text())["hessian"] == want


def test_hessian_default_recorded(tiny):
    from qda_quant.engine.stats import DEFAULT_HESSIAN

    out = build(
        str(tiny / "m"),
        recipe(tiny, TinyRTN()),
        format="w4a16",
        out_dir=tiny / "hd",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    got = json.loads((out / "build.json").read_text())["hessian"]
    assert got == {"dtype": DEFAULT_HESSIAN.dtype, "buffer_rows": DEFAULT_HESSIAN.buffer_rows}


def test_buffered_equals_per_sample_up_to_sum_order():
    """bf16 buffered (one GEMM per 8192 rows) and per sample (one GEMM per sample) give the same Hessian up to
    fp32 summation order (on CPU both are fp32 matmuls)."""
    from qda_quant.engine.stats import GroupStats

    x = torch.randn(6, 300, 32)
    a = GroupStats(32, torch.device("cpu"), frozenset({"hessian"}), flush_rows=8192)
    b = GroupStats(32, torch.device("cpu"), frozenset({"hessian"}), flush_rows=0)
    for t in x:
        a.add(t)
        b.add(t)
    torch.testing.assert_close(a.result()["hessian"], b.result()["hessian"], rtol=1e-5, atol=1e-4)


class Marker2(Marker):
    name: ClassVar[str] = "marker2"


class KeepMarker(Marker):
    name: ClassVar[str] = "keep_marker"
    preserves_quantized: ClassVar[bool] = True


def test_multicard_refuses_model_step_after_layers(tiny):
    """Even a step that keeps quantized weights may change calibration inputs (head with embed=True), and
    multi-card builds run model steps before the ranks start, so the order would change: refused."""
    r = recipe(tiny, TinyRTN(), KeepMarker())
    with pytest.raises(Refused, match="model step after a layer step"):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16"), Target(), CPU * 2)
    compile_plan(str(tiny / "m"), recipe(tiny, Marker(), TinyRTN()), Format.parse("w4a16"), Target(), CPU * 2)


def test_multicard_plans_awq_and_refuses_other_rerun_steps(tiny, monkeypatch):
    """AWQ's parent objective declares `multi_card_layer_io`, and on a dense layer it shares its
    calibration rows (`shares_rows`): every rank prices its own batches and keeps them, so the plan gathers
    nothing and prices one card's share of the rows. A step that needs `layer_io` without that flag
    (autoround on the quantized stream, which also wants the float-input copy) still needs one card."""
    from qda_quant import REGISTRY

    awq_rtn = recipe(tiny, REGISTRY["awq"](), REGISTRY["rtn"]())
    for devs in (CPU, CPU * 2):
        compile_plan(str(tiny / "m"), awq_rtn, Format.parse("w4a16-sym-g32"), Target(), devs)
    with pytest.raises(Refused, match="re-runs layers on the calibration set"):
        compile_plan(
            str(tiny / "m"), recipe(tiny, REGISTRY["autoround"](target="quant_stream")),
            Format.parse("w4a16-sym-g32"), Target(), CPU * 2,
        )  # fmt: skip
    # each rank keeps its own batches' inputs and layer outputs (`shares_rows`): the retained inputs are
    # planned on one card's shard, and nothing is gathered to rank 0 (the multi-card gather terms do not exist here)
    from qda_quant.engine import residency

    retained = []
    original = residency.plan

    def checked(*args, **kwargs):
        retained.append(kwargs["retained_bytes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(residency, "plan", checked)
    plan = compile_plan(str(tiny / "m"), awq_rtn, Format.parse("w4a16-sym-g32"), Target(), CPU * 2)
    assert retained == [4 * 16 * 64 * 2 // 2]  # samples, seqlen, hidden, model dtype: one card's shard
    gathered = {"gather pads", "gathered inputs", "gathered layer_io", "layer_io pad"}
    assert not gathered & set(dict(plan.residency.peak_terms))
    # a layer with routed expert banks keeps the gather: the bank search reads rank 0's summed statistics, so
    # the plan prices the copies rank 0 holds while it gathers (`shares_rows` is a dense-layer path only)
    tiny_qwen3_moe(layers=1).save_pretrained(tiny / "moe")
    save_tiny_tokenizer(tiny / "moe")  # the calibration rows' fingerprint is the same tokenizer
    terms = []

    def captured(*args, **kwargs):
        terms.append({t.name: t.bytes for t in args[4]})
        return original(*args, **kwargs)

    monkeypatch.setattr(residency, "plan", captured)
    compile_plan(str(tiny / "moe"), awq_rtn, Format.parse("w4a16-sym-g32"), Target(), CPU * 2)
    assert {"gathered layer_io", "layer_io pad"} <= set(terms[-1])


class SharingRTN(TinyRTN):
    """TinyRTN that declares `shares_rows`: the plan only (a card count is enough to place it)."""

    name: ClassVar[str] = "tiny_rtn_share"
    shares_rows: ClassVar[bool] = True


def test_plan_refuses_a_row_sharing_step_that_is_not_the_first_layer_step(tiny):
    """A step before it changes weights from rank 0's statistics alone, so the ranks would price different
    weights: refused at plan time, naming both steps; with the sharing step first the recipe plans."""
    first, shared = TinyRTN(), SharingRTN()
    with pytest.raises(
        Refused, match="'tiny_rtn_share' shares its calibration rows.*'tiny_rtn' runs before it"
    ):
        compile_plan(
            str(tiny / "m"), recipe(tiny, first, shared), Format.parse("w4a16-sym-g32"), Target(), CPU * 2
        )  # fmt: skip
    compile_plan(
        str(tiny / "m"), recipe(tiny, shared, first), Format.parse("w4a16-sym-g32"), Target(), CPU * 2
    )  # fmt: skip


def test_plan_refuses_fewer_calibration_batches_than_cards_for_a_row_sharing_step(tiny):
    """AWQ combines per-rank sums through the ranks' collective, so a card with no batch would skip the step
    its peers wait on: refused at plan time, naming the batches and the step, rather than hanging the build."""
    from qda_quant import REGISTRY

    r = Recipe(
        steps=[REGISTRY["awq"](), REGISTRY["rtn"]()],
        calib=FileSource(path=str(tiny / "c.jsonl"), samples=2, seqlen=16),
    )
    with pytest.raises(Refused, match=r"give 2 batches for 3 cards.*row-sharing step \(awq\)"):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16-sym-g32"), Target(), CPU * 3)
    compile_plan(str(tiny / "m"), r, Format.parse("w4a16-sym-g32"), Target(), CPU * 2)  # one batch a card


@pytest.mark.parametrize(
    ("samples", "seqlen", "batch_size", "cards", "rows"),
    [
        # Batch-share cases: 4 batches over 3 cards -> 2 batches (128 rows), not ceil(256 / 3) = 86;
        # 3 batches of 8 over 2 cards -> 2 batches (1024 rows), not ceil(1536 / 2) = 768;
        # and one batch of rows split over 2 cards, where the average and the share agree
        (4, 64, 1, 3, 128),
        (24, 64, 8, 2, 1024),
        (3, 16, 1, 2, 32),
    ],
)
def test_multicard_prices_a_search_on_the_largest_batch_share(
    tiny, monkeypatch, samples, seqlen, batch_size, cards, rows
):
    """`engine.dist.shard` gives a rank contiguous *batches*, so with an uneven batch count one rank
    runs a whole batch more than `rows // world`. A row-sharing AWQ prices that rank's share of the retained
    inputs and of the search's output lists, and gathers nothing: each rank keeps its own batches."""
    from qda_quant import REGISTRY
    from qda_quant.engine import residency

    header, *rest = (tiny / "c.jsonl").read_text().splitlines()
    (tiny / "many.jsonl").write_text("\n".join([header, *rest[:1] * samples]) + "\n")  # the plan reads ids
    r = Recipe(
        steps=[REGISTRY["awq"](), REGISTRY["rtn"]()],
        calib=FileSource(path=str(tiny / "many.jsonl"), samples=samples, seqlen=seqlen),
    )
    seen = []
    original = residency.plan

    def checked(*args, **kwargs):
        seen.append({t.name: t.bytes for t in args[4]})
        return original(*args, **kwargs)

    monkeypatch.setattr(residency, "plan", checked)
    devs = [torch.device("cpu") for _ in range(cards)]
    full_rows = samples * seqlen
    compile_plan(str(tiny / "m"), r, Format.parse("w4a16-sym-g32"), Target(), CPU, batch_size=batch_size)
    compile_plan(str(tiny / "m"), r, Format.parse("w4a16-sym-g32"), Target(), devs, batch_size=batch_size)
    # one card prices every row; its terms give the per-row cost, and the card plan prices the largest
    # share's rows (`rows`), not the average `full_rows // cards`
    one, card = seen[0], seen[-1]
    per_row = {name: one[name] // full_rows for name in ("retained inputs", "search outputs")}
    assert card["retained inputs"] == rows * per_row["retained inputs"]
    assert card["search outputs"] == rows * per_row["search outputs"]
    assert not {"gather pads", "gathered inputs", "gathered layer_io", "layer_io pad"} & set(card)


def test_multicard_plans_head_format_step(tiny):
    """The head's codes travel in the build's qparams dict, which the fan-out ships: no refusal on 2 cards."""
    from qda_quant import REGISTRY

    r = recipe(tiny, REGISTRY["head"](), TinyRTN())
    for devs in (CPU, CPU * 2):
        compile_plan(str(tiny / "m"), r, Format.parse("w4a16"), Target(), devs)


def test_plan_refuses_body_changing_model_step_after_layer_step(tiny):
    """[gptq, rotate] would export the pre-rotation codes onto the rotated body: refused at plan time
    on every card count, naming both steps."""
    from qda_quant import REGISTRY

    r = recipe(tiny, REGISTRY["gptq"](), REGISTRY["rotate"]())
    for devs in (CPU, CPU * 2):
        with pytest.raises(Refused, match="rotate changes weights gptq already quantized; put it before"):
            compile_plan(str(tiny / "m"), r, Format.parse("w4a16"), Target(), devs)


def test_plan_refuses_rerun_target_after_a_model_step_split_the_passes(tiny):
    """[gptq, head, gsq] would hand gsq the already-quantized layer's output as its target, not the float
    layer's: refused at plan time, naming both steps. With the model step last the recipe plans."""
    from qda_quant import REGISTRY

    gptq, head, gsq = REGISTRY["gptq"](), REGISTRY["head"](), REGISTRY["gsq"]()
    with pytest.raises(Refused, match="gsq re-runs layers after head split them into passes"):
        compile_plan(
            str(tiny / "m"), recipe(tiny, gptq, head, gsq), Format.parse("w4a16-sym-g32"), Target(), CPU
        )
    compile_plan(str(tiny / "m"), recipe(tiny, gptq, gsq, head), Format.parse("w4a16-sym-g32"), Target(), CPU)


class Ckpt(Step):
    kind: ClassVar[str] = "checkpoint"
    name: ClassVar[str] = "ckpt_test"

    def check_checkpoint(self, ctx) -> None:  # noqa: ANN001
        if ctx.fmt.name != "w4a16-sym-g32":
            raise Refused(f"ckpt_test: saw format {ctx.fmt.name}")
        raise Refused("ckpt_test: checked at plan time")


def test_plan_checks_checkpoint_steps_up_front(tiny):
    """`compile_plan` plans every stage: the layer steps' passes, and each checkpoint step checked with
    `check_checkpoint` (metadata only) before anything runs; segments never hold a checkpoint step."""
    from qda_quant import REGISTRY
    from qda_quant.engine.planning import check_order

    check_order([REGISTRY["gptq"](), Ckpt()])
    with pytest.raises(Refused, match="ckpt_test: checked at plan time"):
        compile_plan(
            str(tiny / "m"),
            recipe(tiny, REGISTRY["gptq"](), Ckpt()),
            Format.parse("w4a16-sym-g32"),
            Target(),
            CPU,
        )
    with pytest.raises(Refused, match="ckpt_test.*own stage"):
        segments([REGISTRY["gptq"](), Ckpt()])


def test_tensor_digest_detects_a_changed_weight():
    from qda_quant.intake.fingerprint import tensor_digest

    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 96, generator=g).to(torch.bfloat16)
    assert tensor_digest(w) == tensor_digest(w.clone())  # the bytes alone
    view = w[:, ::2]  # a plugin's stepped weight: strided, and hashed in its logical order
    assert not view.is_contiguous() and tensor_digest(view) == tensor_digest(view.contiguous())
    assert tensor_digest(w) != tensor_digest(w[:, torch.randperm(96, generator=g)])  # a column permutation
    assert tensor_digest(w) != tensor_digest(w * 2)
    changed = w.clone()
    changed[7, 13] += 1  # one value that a plain sum could miss
    assert tensor_digest(changed) != tensor_digest(w)


@pytest.mark.gpu
def test_tensor_digest_on_a_card_equals_the_host_bytes():
    """A card's tensor goes through the reusable pinned buffer: same digest as its host bytes, call after call."""
    from qda_quant.intake import fingerprint as fp

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(64, 96, generator=g, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(96, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    assert fp.tensor_digest(a) == fp.tensor_digest(a.cpu())
    big = fp._PINNED[0].numel()
    assert fp.tensor_digest(b) == fp.tensor_digest(b.cpu())  # b is smaller: the buffer did not shrink
    assert fp._PINNED[0].numel() == big == a.numel() * 2
    assert fp.tensor_digest(a) == fp.tensor_digest(a.cpu())  # still a's bytes, after b's call
    assert fp.tensor_digest(a[:, ::2]) == fp.tensor_digest(a[:, ::2].contiguous().cpu())  # a strided weight


def test_settle_drops_written_and_parks_kept_entries():
    from qda_quant.engine.runner import settle

    q = {
        "m.layers.0.a": {"codes": torch.ones(2), "format": "w4a16"},
        "m.layers.1.a": {"codes": torch.ones(2)},
    }
    settle(q, "m.layers.0", written=False)
    assert q["m.layers.0.a"]["codes"].device.type == "cpu" and q["m.layers.0.a"]["format"] == "w4a16"
    settle(q, "m.layers.0", written=True)
    assert list(q) == ["m.layers.1.a"]  # a prefix match on the layer, not "m.layers.1" for "m.layers.10"


def test_multicard_refuses_unpicklable_step(tiny):
    class Local(TinyRTN):
        name: ClassVar[str] = "local_rtn"

    with pytest.raises(Refused, match="cannot be pickled"):
        compile_plan(str(tiny / "m"), recipe(tiny, Local()), Format.parse("w4a16"), Target(), CPU * 2)


def test_hessian_fast_option(tiny):
    from qda_quant.engine.stats import DEFAULT_HESSIAN, FAST_HESSIAN

    assert DEFAULT_HESSIAN.buffer_rows == 0 and type(FAST_HESSIAN)("bf16", 8192) == FAST_HESSIAN
    out = build(
        str(tiny / "m"),
        recipe(tiny, TinyRTN()),
        format="w4a16",
        out_dir=tiny / "hf",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        hessian="fast",
        log=lambda s: None,
    )
    assert json.loads((out / "build.json").read_text())["hessian"] == {"dtype": "bf16", "buffer_rows": 8192}


def test_hub_repo_id_builds_from_its_snapshot(tiny, monkeypatch):
    """A repo id resolves once to its local snapshot, so the real writer finds the source shards."""
    import huggingface_hub
    from qda_quant import REGISTRY
    from qda_quant.engine.planning import SOURCE_FILES

    snap = tiny / "snapshot"  # wide enough for 128-column groups
    tiny_qwen3(layers=1, hidden=128, inter=256).save_pretrained(snap)
    save_tiny_tokenizer(snap)
    calls = []

    def fake_snapshot(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, tuple(allow_patterns)))
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = build(
        "org/tiny-qwen3",
        recipe(tiny, REGISTRY["rtn"]()),
        format="w4a16",
        out_dir=tiny / "hub",
        devices=["cpu"],
        dtype=torch.float32,
        log=lambda s: None,
    )
    assert calls == [("org/tiny-qwen3", None, SOURCE_FILES)]  # a bare id asks for the hub's main
    assert json.loads((out / "build.json").read_text())["source"] == str(snap)
    assert sorted(out.glob("model-*.safetensors"))


class AttentionProbe(Step):
    """Records which attention backends are enabled while the build runs its layer steps."""

    name: ClassVar[str] = "attention_probe_test"
    seen: ClassVar[list[dict[str, bool]]] = []

    def apply_layer(self, ctx: LayerContext) -> None:
        cuda = torch.backends.cuda
        AttentionProbe.seen.append(
            {
                "cudnn": cuda.cudnn_sdp_enabled(),
                "flash": cuda.flash_sdp_enabled(),
                "efficient": cuda.mem_efficient_sdp_enabled(),
                "math": cuda.math_sdp_enabled(),
            }
        )


def test_build_uses_default_attention_backends_whatever_the_caller_set(tiny):
    """vLLM disables cuDNN attention when it starts; a text build later in that process must not inherit that,
    or the calibration forward runs another kernel and the checkpoint changes. The caller's setting comes back
    after. Image calibration narrows the set instead (tests/test_calib_vlm.py)."""
    from qda_quant.engine.execute import ATTENTION_BACKENDS

    AttentionProbe.seen.clear()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        out = build(
            str(tiny / "m"),
            recipe(tiny, AttentionProbe(), TinyRTN()),
            format="w4a16-sym-g128",
            out_dir=tiny / "attn",
            devices=["cpu"],
            dtype=torch.float32,
            exporter=writes(FakeWriter),
            log=lambda s: None,
        )
        assert torch.backends.cuda.cudnn_sdp_enabled() is False  # restored
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)
    assert AttentionProbe.seen and all(all(s.values()) for s in AttentionProbe.seen)
    info = json.loads((out / "build.json").read_text())
    assert info["attention_backends"] == list(ATTENTION_BACKENDS)


def test_residency_holds_a_jointly_trained_layers_units_together(monkeypatch):
    """xy-2140: a step that trains every unit of the layer in one optimizer holds them all, and cuts none.

    `steps.gsq` builds every unit of a layer before it trains any and never cuts an expert-parallel bank
    (`_train`: `count = 1 << 40`), so the layer's peak is the sum of its units' state, not the largest one's.
    """
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (140 * gib, 140 * gib))
    card = [torch.device("cuda:0")]
    joint = [
        residency.Solve("solver workspace", experts=4, workspace=2 * gib, factor_batch=16, joint=True)
        for _ in range(2)
    ] + [residency.Solve("solver workspace", experts=1, workspace=2 * gib, factor_batch=1, joint=True)]
    got = residency.plan(10 * gib, 1 * gib, 0, card, solves=joint)
    assert dict(got.peak_terms)["solver workspace"] == 18.0  # 4 + 4 + 1 workspaces of 2 GiB, all held
    assert got.peak_gib == 28.0  # the weights plus every unit's workspace
    # a step that solves one unit at a time keeps its bound: the largest unit's workspace, beside the others
    single = [
        residency.Solve("solver workspace", experts=4, workspace=2 * gib, factor_batch=16) for _ in range(3)
    ]
    got = residency.plan(10 * gib, 1 * gib, 0, card, solves=single)
    # one unit's bound, not three: a 4-expert bank is under the 16-member factorization floor, so the plan
    # prices its padded factor (`solve.MIN_FACTOR_BATCH`), and only one unit's at a time
    assert dict(got.peak_terms)["solver workspace"] == 32.0


def test_residency_refuses_a_joint_unit_past_the_card_instead_of_chunking_it(monkeypatch):
    """xy-2140: the step never cuts a unit itself, so the plan must not price a chunk the run will not take."""
    from qda_quant.engine import residency

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 * gib, 40 * gib))
    joint = residency.Solve("solver workspace", experts=100, workspace=gib, factor_batch=16, joint=True)
    with pytest.raises(Refused, match=r"cuda:0: the build needs 110\.0 GiB .*the whole unit"):
        residency.plan(10 * gib, 1 * gib, 0, [torch.device("cuda:0")], solves=[joint])


def _refused_plan(tmp, steps, fmt, model="m"):
    return refusal(
        lambda: compile_plan(str(tmp / model), recipe(tmp, *steps), Format.parse(fmt), Target(), CPU)
    )


def _moe(tmp, inter=32):
    tiny_qwen3_moe(layers=1, inter=inter).save_pretrained(tmp / "moe")


def _offset_refusal(tmp, monkeypatch, model, limit, inter=32):
    """The smallest solve there is must fit the int32 offsets; the limit is shrunk to bite the tiny model."""
    from qda_quant.solve import solve

    if model == "moe":
        _moe(tmp, inter)
    monkeypatch.setattr(solve, "INT32_ELEMENTS", limit)
    return _refused_plan(tmp, [REGISTRY["gptq"]()], "w4a16-sym-g32", model)


def _grid3_refusal(tmp, monkeypatch, method, fmt, scale_dtype=None):
    """The grid3 clip is refused by the format or the step, before the calibration file is ever read."""
    # The step-level rows fix the target's scale grid; the format rows plan for real.
    if scale_dtype is not None:
        from qda_quant.target import KernelPlan

        monkeypatch.setattr(Target, "plan", lambda *args: KernelPlan("test", scale_dtype=scale_dtype))
    (tmp / "c.jsonl").unlink()
    return _refused_plan(tmp, [REGISTRY[method](clip="grid3")], fmt)


def _device_refusal(monkeypatch, visible, request):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    return refusal(lambda: resolve_devices(request))


def _ids_refusal(tmp, rows):
    sha = json.loads((tmp / "c.jsonl").read_text().splitlines()[0])["tokenizer"]
    write_ids(tmp / "big.jsonl", rows, sha)
    rec = Recipe(steps=[TinyRTN()], calib=FileSource(path=str(tmp / "big.jsonl"), samples=1, seqlen=16))
    return refusal(lambda: compile_plan(str(tmp / "m"), rec, Format.parse("w4a16"), Target(), CPU))


def _hessian_refusal(tmp, setting):
    return refusal(
        lambda: build(
            str(tmp / "m"),
            recipe(tmp, TinyRTN()),
            format="w4a16",
            out_dir=tmp / "x",
            devices=["cpu"],
            exporter=writes(FakeWriter),
            hessian=setting,
            log=lambda s: None,
        )
    )


RENDERERS = {
    "dense solve offsets": lambda t, m: _offset_refusal(t, m, "m", 1000),
    "expert matrix offsets": lambda t, m: _offset_refusal(t, m, "moe", 40_000, inter=512),
    "factor row offsets": lambda t, m: _offset_refusal(t, m, "moe", 150_000, inter=512),
    "group size width": lambda t, m: _refused_plan(t, [REGISTRY["rtn"]()], "w4a16"),
    "group size expert bank": lambda t, m: (
        _moe(t),
        _refused_plan(t, [REGISTRY["rtn"]()], "w4a16-sym-g64", "moe"),
    )[1],
    "group size head": lambda t, m: _refused_plan(
        t, [REGISTRY["rtn"](), REGISTRY["head"](format="w4a16-sym-g128", embed=True)], "w4a16-sym-g32"
    ),
    "grid3 formats w4a16": lambda t, m: _grid3_refusal(t, m, "gptq", "w4a16"),
    "grid3 formats g64": lambda t, m: _grid3_refusal(t, m, "gptq", "w4a8-sym-g64"),
    "grid3 formats asym g128": lambda t, m: _grid3_refusal(t, m, "gptq", "w4a8-asym-g128"),
    "grid3 without fp8 scales": lambda t, m: _grid3_refusal(t, m, "gptq", "w4a8", "bf16"),
    "grid3 without gptq": lambda t, m: _grid3_refusal(t, m, "rtn", "w4a8", "fp8_e4m3"),
    "ids beyond vocab": lambda t, m: _ids_refusal(t, [[1, 2, 500] + [3] * 20]),
    "mixed device request": lambda t, m: _device_refusal(m, "2,3,4,5", [0, 1, 2, 3]),
    "mixed device request one short": lambda t, m: _device_refusal(m, "1,2", [0, 1]),
    "repeated card physical": lambda t, m: _device_refusal(m, "4,5,6", [4, 4]),
    "repeated card logical": lambda t, m: _device_refusal(m, "4,5,6", [0, 0]),
    "hessian dtype": lambda t, m: _hessian_refusal(t, {"dtype": "fp16"}),
    "hessian buffer_rows": lambda t, m: _hessian_refusal(t, {"buffer_rows": -1}),
    "hessian unknown key": lambda t, m: _hessian_refusal(t, {"rows": 3}),
}


@pytest.mark.parametrize("name", list(RENDERERS))
def test_refusals_keep_their_wording(tiny, monkeypatch, name):
    """Each refusal fires, with the exact text in `golden/errors_engine_build.txt` (QDA_GOLDEN_UPDATE=1 rewrites)."""
    text = RENDERERS[name](tiny, monkeypatch).replace(str(tiny), "<tmp>")
    check_golden("engine_build", name, text, list(RENDERERS))


def test_an_export_alone_build_records_the_target_with_its_backend(tiny):
    """build.json's `target` reads back as the build's target (`target_name`) on both paths, backend and all."""

    class Alone(WriterExporter):
        def writes_alone(self, plan, recipe):  # noqa: ARG002
            return True

    exporter = Alone()
    exporter._writer = FakeWriter
    out = build(
        str(tiny / "m"),
        recipe(tiny, NoStatsRTN()),
        format="w4a16-sym-g128",
        target="sm90@0.33.0:marlin",
        out_dir=tiny / "alone",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=exporter,
        log=lambda s: None,
    )
    assert FakeWriter.last.layers == []  # no layer pass: the export-alone path wrote it
    assert json.loads((out / "build.json").read_text())["target"] == "sm90@0.33.0:marlin"


CUDA2 = [torch.device("cuda:0"), torch.device("cuda:1")]


def _one_card(tiny, given, *steps, devices=CUDA2):
    from qda_quant.engine.planning import one_card_devices

    said: list[str] = []
    got = one_card_devices(
        given,
        devices,
        str(tiny / "m"),
        recipe(tiny, *steps),
        Format.parse("w4a16-sym-g32"),
        Target(),
        said.append,
    )
    return got, said


@pytest.mark.parametrize("method", ["guided", "fisher_trace", "gptaq"])
def test_unset_devices_puts_a_one_card_step_on_the_first_card(tiny, monkeypatch, method):
    """`devices` not written (every visible card): a one-card step takes the first card and says so."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    got, said = _one_card(tiny, None, REGISTRY[method](), REGISTRY["gptq"]())
    assert got == CUDA2[:1]
    assert said == [f"step {method!r} runs on one card: devices = [0]"]


def test_one_card_note_names_the_card_as_the_lease_does(tiny, monkeypatch):
    """Under a lease the note names the physical card, the reading `devices` gives it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    _, said = _one_card(tiny, [], REGISTRY["guided"](), REGISTRY["gptq"]())
    assert said == ["step 'guided' runs on one card: devices = [2]"]


def test_a_model_step_after_a_layer_step_takes_one_card(tiny, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    got, said = _one_card(tiny, None, TinyRTN(), Marker())
    assert got == CUDA2[:1]
    assert said == [f"step {Marker.name!r} runs on one card: devices = [0]"]


def test_named_cards_refuse_a_one_card_step_with_the_fix(tiny):
    """Several cards the user named are kept as named: refused, with the one-card setting as the fix."""
    from qda_quant.exit import refusal

    with pytest.raises(Refused) as e:
        _one_card(tiny, [0, 1], REGISTRY["guided"](), REGISTRY["gptq"]())
    lines = refusal(str(e.value), e.value.fix).splitlines()
    assert len(lines) == 2
    assert "step 'guided' computes calibration-row weights; v0 builds it on 1 card only, not 2" in lines[0]
    assert lines[1].endswith("set devices: [0]")


def test_unset_devices_without_a_one_card_step_keeps_every_card(tiny):
    got, said = _one_card(tiny, None, REGISTRY["gptq"]())
    assert got == CUDA2
    assert said == []


def test_plan_reports_the_card_a_one_card_step_takes(tiny, monkeypatch, capsys):
    """`qda plan` makes the build's decision: the plan's cards are the ones the build uses."""
    from qda_quant.config import BuildConfig
    from qda_quant.engine import planning

    monkeypatch.setattr(planning, "resolve_devices", lambda devices: CPU * 2 if devices is None else CPU)
    cfg = BuildConfig(
        model=str(tiny / "m"),
        format="w4a16-sym-g32",
        steps=[{"guided": {}}, {"gptq": {}}],
        calib={"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 16}},
    )
    assert cfg.plan().devices == CPU
    assert "step 'guided' runs on one card: devices = [cpu]" in capsys.readouterr().out


def test_build_runs_a_one_card_step_on_one_card_when_devices_is_unset(tiny, monkeypatch):
    import importlib

    build_module = importlib.import_module("qda_quant.engine.build")

    monkeypatch.setattr(build_module, "resolve_devices", lambda devices: CPU * 2)
    monkeypatch.setattr(build_module, "_refuse_cpu_steps", lambda devices, recipe: None)
    said: list[str] = []
    out = build(
        str(tiny / "m"),
        recipe(tiny, REGISTRY["fisher_trace"](), REGISTRY["gptq"]()),
        format="w4a16-sym-g32",
        out_dir=tiny / "out",
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=said.append,
    )
    assert "step 'fisher_trace' runs on one card: devices = [cpu]" in said
    assert json.loads((out / "build.json").read_text())["card_count"] == 1
