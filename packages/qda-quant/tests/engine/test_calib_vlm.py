"""Image-text calibration: rows to samples with pixel inputs, and pixel inputs reaching the forward (CPU)."""

from __future__ import annotations

import io
from typing import ClassVar

import pytest
import torch
import transformers
from qda_quant.arch import describe
from qda_quant.calib.sources import Rows, VLMSource
from qda_quant.engine import calib
from qda_quant.engine.runner import capture
from qda_quant.step import Refused

from ..tiny import config

Image = pytest.importorskip("PIL.Image")

CPU = torch.device("cpu")
IMAGE = 299  # the tiny Gemma 3's image token; mm_tokens_per_image = 4


def _vlm():
    cfg = config("gemma3_vlm")
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


def _image_ids():
    return torch.tensor([[2, 5, 6, 297] + [IMAGE] * 4 + [298, 7, 8, 9]])


def test_pixel_inputs_reach_the_calibration_forward():
    """The first layer's input holds the vision tower's embeddings at the image tokens, and only there."""
    m = _vlm()
    graph = describe(m, verify=False)
    ids = _image_ids()
    tt = (ids == IMAGE).long()
    pixels = {"pixel_values": torch.randn(1, 3, 28, 28), "token_type_ids": tt}
    with_image = capture(m, graph, [ids], CPU, inputs=[pixels])[0].h[0]
    text_only = capture(m, graph, [ids], CPU)[0].h[0]
    image = (ids[0] == IMAGE).nonzero().flatten()
    text = (ids[0] != IMAGE).nonzero().flatten()
    assert torch.equal(with_image[text], text_only[text])
    assert not torch.allclose(with_image[image], text_only[image])


class _Processor:
    """Stands in for a model's processor: four image tokens per image, a token per word."""

    def apply_chat_template(self, messages, tokenize=False):
        assert not tokenize
        return " ".join(c.get("text", "<img>") for m in messages for c in m["content"])

    def __call__(self, text, images, return_tensors):
        assert len(images) == text.count("<img>")
        ids = [t for w in text.split() for t in ([IMAGE] * 4 if w == "<img>" else [5])]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids)),
            "pixel_values": torch.zeros(len(images), 3, 28, 28),
        }


class _Rows(VLMSource):
    def read(self, model_dir):
        img = Image.new("RGB", (28, 28))
        rows = [{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": w}]}],
                 "images": [img]} for w in ("a", "a b c d e f g h", "a b")]  # fmt: skip
        return Rows(rows, None)


class _ServingProcessor(_Processor):
    """A processor whose template validates serving: an assistant-last chat needs the continuation form."""

    def apply_chat_template(self, messages, tokenize=False, **kw):
        if messages[-1]["role"] == "assistant" and not kw.get("continue_final_message"):
            raise ValueError("Expected last role User or Tool for serving but got assistant")
        return super().apply_chat_template(messages, tokenize=tokenize)


def test_an_assistant_last_image_row_uses_the_processors_continuation_form(monkeypatch):
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _ServingProcessor())

    class AssistantRows(VLMSource):
        def read(self, model_dir):
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "a"}]}]  # fmt: skip
            return Rows([{"messages": msgs, "images": [Image.new("RGB", (28, 28))]}], None)

    samples = calib.load_samples(AssistantRows(samples=1, seqlen=8), "unused", images=True)
    assert [s.ids.tolist() for s in samples] == [[IMAGE] * 4 + [5, 5]]


def test_image_rows_give_one_sample_each_with_their_pixels(monkeypatch):
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    samples = calib.load_samples(_Rows(samples=2, seqlen=6), "unused", images=True)
    # the second row (12 tokens) is longer than seqlen and skipped
    assert [s.ids.tolist() for s in samples] == [[IMAGE] * 4 + [5], [IMAGE] * 4 + [5, 5]]
    assert all(set(s.inputs) == {"pixel_values"} for s in samples)
    ids, inputs = calib.load_with_inputs(_Rows(samples=2, seqlen=6), "unused")
    assert len(ids) == len(inputs) == 2
    with pytest.raises(Refused, match="1 image rows"):
        calib.load_samples(_Rows(samples=2, seqlen=5), "unused", images=True)


@pytest.mark.parametrize("residency", ["resident", "stream"])
def test_build_capture_keeps_planned_pixel_inputs(tmp_path, monkeypatch, residency):
    from qda_quant import REGISTRY, Format, Recipe, Target
    from qda_quant.engine.execute import BuildRun, Timers, _capture
    from qda_quant.engine.planning import compile_plan

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    model = _vlm()
    model.save_pretrained(tmp_path)
    recipe = Recipe(steps=[REGISTRY["gptq"]()], calib=_Rows(samples=2, seqlen=6))
    fmt, target = Format.parse("w4a16-sym-pc"), Target()
    plan = compile_plan(str(tmp_path), recipe, fmt, target, [CPU], torch.float32, residency=residency)
    run = BuildRun(model, plan.graph, recipe, fmt, target, plan, [CPU], plan.calib_ids, Timers([CPU]))
    with pytest.raises(Refused, match="image calibration"):
        run.batches()
    actual = _capture(run, recipe.steps)
    batches = calib.batches(plan.calib_ids)
    expected = capture(model, plan.graph, batches, CPU, inputs=plan.calib_inputs)
    text_only = capture(model, plan.graph, batches, CPU)
    assert len(actual) == len(expected) == 2
    assert all(torch.equal(got.h, want.h) for got, want in zip(actual, expected, strict=True))
    assert any(not torch.equal(got.h, text.h) for got, text in zip(actual, text_only, strict=True))


def test_image_calibration_runs_deterministic_attention_backends(tmp_path, monkeypatch):
    """Image samples have varying sequence lengths, where cuDNN attention picks one of two engines from process
    to process, so an image-calibration build's forwards run flash, memory-efficient and math attention.
    The caller's setting comes back after."""
    from qda_quant import REGISTRY, Format, Recipe, Target
    from qda_quant.engine.execute import BuildRun, Timers, _layer_pass
    from qda_quant.engine.planning import compile_plan
    from qda_quant.step import LayerContext, Step

    class Probe(Step):
        name: ClassVar[str] = "attention_probe_test"
        seen: ClassVar[list[bool]] = []

        def apply_layer(self, ctx: LayerContext) -> None:
            Probe.seen.append(torch.backends.cuda.cudnn_sdp_enabled())

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    model = _vlm()
    model.save_pretrained(tmp_path)
    recipe = Recipe(steps=[Probe(), REGISTRY["gptq"]()], calib=_Rows(samples=2, seqlen=6))
    fmt, target = Format.parse("w4a16-sym-pc"), Target()
    plan = compile_plan(str(tmp_path), recipe, fmt, target, [CPU], torch.float32)
    assert plan.calib_inputs is not None
    run = BuildRun(model, plan.graph, recipe, fmt, target, plan, [CPU], plan.calib_ids, Timers([CPU]))
    Probe.seen.clear()
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        _layer_pass(run, recipe.steps, None, plan.graph)
        assert torch.backends.cuda.cudnn_sdp_enabled() is True  # restored
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)
    assert Probe.seen and all(not s for s in Probe.seen)


def test_text_samples_have_no_inputs(tmp_path, monkeypatch):
    class Text(VLMSource):
        def read(self, model_dir):
            return Rows([{"input_ids": [1, 2, 3]}], "fp")

    monkeypatch.setattr(calib, "fingerprint", lambda _d: "fp")
    ids, inputs = calib.load_with_inputs(Text(samples=1, seqlen=8), tmp_path)
    assert [i.tolist() for i in ids] == [[1, 2, 3]] and inputs is None


def test_cauldron_rows_become_chat_messages_with_images():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    row = VLMSource()._row(
        {"images": [{"bytes": buf.getvalue()}], "texts": [{"user": "Q?", "assistant": "A."}]}
    )
    assert row["messages"][0]["content"] == [{"type": "image"}, {"type": "text", "text": "Q?"}]
    assert row["messages"][1]["content"] == [{"type": "text", "text": "A."}]
    assert row["images"][0].size == (8, 8)
    with pytest.raises(Refused, match="expected images"):
        VLMSource()._row({"images": [], "texts": []})


@pytest.mark.parametrize("step", ["scale_kd", "guided"])
def test_plan_refuses_steps_that_read_samples_without_their_pixels(tmp_path, monkeypatch, step):
    """scale_kd (its default data is the calibration rows) and token weights would see image tokens alone."""
    from qda_quant import REGISTRY, Format, Recipe, Target
    from qda_quant.engine.planning import compile_plan

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    _vlm().save_pretrained(tmp_path / "m")
    steps = (
        [REGISTRY["gptq"](), REGISTRY["scale_kd"](steps=1)]
        if step == "scale_kd"
        else [REGISTRY["guided"](), REGISTRY["gptq"]()]
    )
    recipe = Recipe(steps=steps, calib=_Rows(samples=2, seqlen=6))
    with pytest.raises(Refused, match="without (their pixels|the images)"):
        compile_plan(
            str(tmp_path / "m"), recipe, Format.parse("w4a16-sym-pc"), Target(), [CPU], torch.bfloat16
        )
    with pytest.raises(Refused, match="without their pixels; .*set the step's `data:`"):
        calib.load_samples(_Rows(samples=2, seqlen=6), "unused")


def test_scale_kd_with_its_own_data_plans_on_image_calibration(tmp_path, monkeypatch):
    """With image calibration, scale_kd fits its own text traces through `data:`; it never sees the pixels."""
    from qda_quant import REGISTRY, Format, Recipe, Target
    from qda_quant.engine.planning import compile_plan

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    _vlm().save_pretrained(tmp_path / "m")
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"prompt_ids": [5, 5], "completion_ids": [5, 5]}\n')
    recipe = Recipe(
        steps=[REGISTRY["gptq"](), REGISTRY["scale_kd"](steps=1, data=str(traces))],
        calib=_Rows(samples=2, seqlen=6),
    )
    plan = compile_plan(
        str(tmp_path / "m"), recipe, Format.parse("w4a16-sym-pc"), Target(), [CPU], torch.bfloat16
    )
    assert [s.name for s in plan.checkpoint_steps] == ["scale_kd"] and plan.calib_inputs is not None


def test_missing_pillow_is_refused_before_the_download(monkeypatch):
    import sys

    import huggingface_hub

    pytest.importorskip("pyarrow")
    monkeypatch.setitem(sys.modules, "PIL.Image", None)  # import fails as if Pillow were not installed
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *_a, **_k: pytest.fail("downloaded"))
    with pytest.raises(Refused, match=r'uv pip install -c constraints.txt "\./packages/qda-quant\[vlm\]"'):
        VLMSource().read("unused")


class _Text(VLMSource):
    """Text-id rows (the model's fingerprint), standing in for a text source in a mix."""

    images_source = False

    def read(self, model_dir):
        return Rows([{"input_ids": [1, 2, 3, i]} for i in range(10, 20)], "fp")


def _mix(monkeypatch, image_weight=1, text_weight=1, **kw):
    from qda_quant.calib.sources import MixSource

    parts = [(_Rows(seqlen=6), image_weight), (_Text(seqlen=8), text_weight)]
    monkeypatch.setattr(MixSource, "source_parts", lambda self: parts)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    monkeypatch.setattr(calib, "fingerprint", lambda _d: "fp")
    return MixSource(parts=[{"vlm": {}, "weight": 1}], **kw)


def test_a_mix_of_sources_interleaves_image_and_text_samples_deterministically(monkeypatch):
    src = _mix(monkeypatch, samples=4, mix_seed=3)
    ids, inputs = calib.load_with_inputs(src, "unused")
    assert len(ids) == 4 and sum(bool(i) for i in inputs) == 2  # two image samples, two text, by weight
    again, inputs2 = calib.load_with_inputs(src, "unused")
    assert [i.tolist() for i in ids] == [i.tolist() for i in again]
    assert [bool(i) for i in inputs] == [bool(i) for i in inputs2]
    other, _ = calib.load_with_inputs(_mix(monkeypatch, samples=4, mix_seed=4), "unused")
    assert sorted(map(tuple, (i.tolist() for i in other))) == sorted(map(tuple, (i.tolist() for i in ids)))
    with pytest.raises(
        Refused, match="without their pixels"
    ):  # a step reading the ids alone is still refused
        calib.load_samples(src, "unused")


def test_capture_takes_image_and_text_samples_in_one_pass():
    m = _vlm()
    graph = describe(m, verify=False)
    image = _image_ids()
    text = torch.tensor([[2, 5, 6, 7, 8]])
    pixels = {"pixel_values": torch.randn(1, 3, 28, 28), "token_type_ids": (image == IMAGE).long()}
    mixed = capture(m, graph, [image, text], CPU, inputs=[pixels, {}])
    assert torch.equal(mixed[1].h, capture(m, graph, [text], CPU)[0].h)
    assert torch.equal(mixed[0].h, capture(m, graph, [image], CPU, inputs=[pixels])[0].h)


@pytest.mark.parametrize(
    ("parts", "rows", "match"),
    [
        ([{"path": "a.jsonl"}, {"vlm": {}, "weight": 1}], 8, "either files"),
        ([{"path": "a.jsonl"}], None, "rows: required"),
        ([{"mix": {"parts": [{"path": "a.jsonl"}], "rows": 4}, "weight": 1}], None, "cannot be a part"),
    ],
)
def test_mix_parts_are_one_kind(parts, rows, match):
    from qda_quant.calib.sources import MixSource

    with pytest.raises(ValueError, match=match):
        MixSource(parts=parts, rows=rows)


def test_a_mix_of_sources_is_an_image_source_when_a_part_is():
    from qda_quant.calib.sources import MixSource

    mix = MixSource(parts=[{"vlm": {}, "weight": 1}, {"file": {"path": "a.jsonl"}, "weight": "1/2"}])
    (vlm, w1), (file, w2) = mix.source_parts()
    assert mix.images_source and vlm.name == "vlm" and file.name == "file" and (w1, w2) == (1, 0.5)
    assert not MixSource(parts=[{"path": "a.jsonl"}], rows=4).images_source
    recorded = mix.to_dict()["mix"]["parts"][
        0
    ]  # the recipe records every setting of a part, pinned revision too
    assert (
        recorded["weight"] == 1 and recorded["vlm"]["revision"] == vlm.revision and "file" in recorded["vlm"]
    )
    again = MixSource(**mix.to_dict()["mix"])
    assert again.to_dict() == mix.to_dict()


def test_heldout_guard_with_source_parts_is_refused_not_ignored(tmp_path):
    """The guard runs on rows of a mix of files; a mix of sources has no such step, so it refuses the setting."""
    from qda_quant.calib.sources import MixSource

    (tmp_path / "c.jsonl").write_text('{"text": "alpha beta gamma delta"}\n')
    (tmp_path / "eval.jsonl").write_text('{"text": "alpha beta gamma delta"}\n')
    with pytest.raises(ValueError, match="0 usable rows"):  # files: the guard drops the overlapping row
        MixSource(parts=[{"path": str(tmp_path / "c.jsonl")}], rows=1, heldout=str(tmp_path / "eval.jsonl"),
                  ngram=2, out=str(tmp_path / "m.jsonl")).read(tmp_path)  # fmt: skip
    part = {"file": {"path": str(tmp_path / "c.jsonl")}, "weight": 1}
    for setting in ({"heldout": str(tmp_path / "eval.jsonl"), "ngram": 2}, {"ngram": 2}, {"rows": 4},
                    {"out": "m.jsonl"}, {"seqlen": 16}, {"chunks": True}):  # fmt: skip
        with pytest.raises(ValueError, match=f"calib.mix.{next(iter(setting))}: does not apply"):
            MixSource(parts=[part], samples=1, **setting)
    recorded = MixSource(parts=[part], samples=1).to_dict()["mix"]  # every field, at its default
    assert MixSource(**recorded).to_dict()["mix"] == recorded


def test_a_short_image_calibration_names_the_settings_to_change(monkeypatch):
    """A row longer than seqlen is skipped whole, so the refusal names seqlen, rows and samples (xrev3 note (b))."""
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_a, **_k: _Processor())
    with pytest.raises(Refused) as caught:
        calib.load_samples(_Rows(samples=2, seqlen=5), "unused", images=True)
    said = str(caught.value)
    assert "raise calib.vlm.seqlen (now 5) (2 image rows are longer)" in said
    assert "raise calib.vlm.rows or lower calib.vlm.samples to 1" in said
