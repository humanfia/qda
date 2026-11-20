"""The build's data lint: another family's chat markup refuses; the model's own format is counted, never refused.

The fixtures copy the shapes of two real calibration files (placeholder words, ids of `r1_tokenizer`):

- ``calib_big.jsonl``: ``{"text", "domain"}`` rows, ChatML-rendered (with an empty think) beside R1-rendered ones;
- ``kd_fix.jsonl``: ``{"prompt_ids", "completion_ids", "domain"}`` traces, some flagged ``appended_eos``: a ChatML
  row stored as ids, then R1 rows cut mid-think or after the think with an EOS appended, cut with none, complete,
  and one without its BOS and with an empty think.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from qda_quant.calib.sources import FileSource
from qda_quant.calib.template import lint_rows
from qda_quant.engine.build import build
from qda_quant.engine.calib import fingerprint, load_samples
from qda_quant.recipe import Recipe
from qda_quant.step import Refused

from ..engine.test_build import FakeWriter, TinyRTN
from ..engine_util import tiny_qwen3, writes

FIXTURES = Path(__file__).parent / "fixtures" / "data_lint"
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
R1 = (
    BOS + "{% for m in messages %}{% if m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
    "{% else %}<｜Assistant｜>{{ m['content'] }}" + EOS + "{% endif %}{% endfor %}"
)
MARKERS = [BOS, EOS, "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "<|im_start|>", "<|im_end|>"]


def r1_tokenizer(path: Path | None = None):
    """A WordLevel tokenizer (100 words, then the markers: ids below 110) with R1's template, BOS and EOS."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.WordLevel({f"w{i}": i for i in range(100)} | {"[UNK]": 100}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.add_tokens(MARKERS)
    fast.bos_token, fast.eos_token = BOS, EOS
    fast.chat_template = R1
    if path is not None:
        fast.save_pretrained(path)
    return fast


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


def test_the_calib_big_shape_refuses_its_chatml_rows():
    found = lint_rows(rows("calib_big.jsonl"), r1_tokenizer(), "calib_big.jsonl")
    assert (found.foreign, found.own, found.first) == (2, 1, (0, "<|im_start|>")), found
    why = found.refusal("calib_big.jsonl")
    assert why is not None and "2 of 3 text rows" in why and "row 0" in why and "of chatml" in why, why


def test_the_kd_fix_shape_refuses_a_chatml_row_stored_as_ids(tmp_path):
    """Through the calibration loader: the ids carry the model's fingerprint, the decoded text is ChatML."""
    model = tmp_path / "m"
    r1_tokenizer(model)
    calib = tmp_path / "kd_fix.jsonl"
    calib.write_text((FIXTURES / "kd_fix.jsonl").read_text())
    (tmp_path / "kd_fix.jsonl.tokenizer").write_text(str(fingerprint(model)))
    with pytest.raises(
        Refused, match=r"kd_fix\.jsonl: 1 of 6 rows \(token ids decoded\) .* row 0, .*'<\|im_start\|>'"
    ):
        load_samples(FileSource(path=str(calib), samples=1, seqlen=4), model)


def test_own_format_rows_are_counted_not_refused():
    """The kd_fix R1 rows: unclosed think, empty think, no EOS, no BOS and the appended-EOS flag, each counted."""
    found = lint_rows(rows("kd_fix.jsonl")[1:], r1_tokenizer(), "kd_fix.jsonl")
    assert found.refusal("kd_fix.jsonl") is None
    got = {k: v for k, v in found.lint_record().items() if k != "file"}
    assert got == {
        "rows": 5,
        "token_rows": 5,
        "bad_ids": 0,
        "own": 5,
        "plain": 0,
        "foreign": 0,
        "empty_think": 1,
        "unclosed_think": 2,
        "no_eos": 1,
        "no_bos": 1,
        "appended_eos": 2,
    }


def test_the_chatml_rows_are_counted_by_the_same_defects():
    """A ChatML row on an R1 model: empty think, no R1 EOS at its end, no R1 BOS (and refused)."""
    found = lint_rows(rows("kd_fix.jsonl")[:1], r1_tokenizer(), "kd_fix.jsonl")
    assert (found.foreign, found.empty_think, found.no_eos, found.no_bos) == (1, 1, 1, 1)


def test_plain_text_and_messages_rows_are_not_counted():
    found = lint_rows(
        [{"text": "w1 w2 <think>"}, {"messages": [{"role": "user", "content": "w1"}]}], r1_tokenizer(), "x"
    )
    assert (found.plain, found.own, found.unclosed_think, found.no_eos, found.no_bos) == (1, 1, 0, 0, 0)


def test_a_build_logs_one_lint_block_and_records_each_file(tmp_path):
    m: Any = tiny_qwen3(layers=1)  # a transformers model: save_pretrained
    m.save_pretrained(tmp_path / "m")
    r1_tokenizer(tmp_path / "m")
    calib = tmp_path / "own.jsonl"
    calib.write_text("".join(json.dumps(r) + "\n" for r in rows("calib_big.jsonl")[2:] * 4))
    recipe = Recipe(steps=[TinyRTN()], calib=FileSource(path=str(calib), samples=1, seqlen=8))
    lines: list[str] = []
    build(
        str(tmp_path / "m"),
        recipe,
        format="w4a16-sym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lines.append,
    )
    info = json.loads((tmp_path / "out" / "build.json").read_text())
    (record,) = info["data_lint"]["files"]
    assert (record["file"], record["rows"], record["own"], record["unclosed_think"]) == (str(calib), 4, 4, 4)
    blocks = [line for line in lines if line.startswith("data lint")]
    assert len(blocks) == 1 and f"{calib}: 4 rows (0 token-id): own 4" in blocks[0], lines


def test_scale_kd_data_is_linted_as_calibration_is(tmp_path):
    """The KD trace file a step reads: its ChatML trace refuses; its R1 traces join the build's lint records."""
    from qda_quant.steps.scale_kd import ScaleKD

    teacher = tmp_path / "teacher"
    r1_tokenizer(teacher)
    recipe = Recipe(steps=[], calib=FileSource(path=str(tmp_path / "unused.jsonl")))
    step = ScaleKD.model_validate({"data": str(FIXTURES / "kd_fix.jsonl")})
    with pytest.raises(Refused, match=r"scale_kd\.data .*kd_fix\.jsonl: 1 of 6 rows \(token ids decoded\)"):
        step._lint(recipe, str(teacher), step.trace_rows())
    own = tmp_path / "kd_own.jsonl"
    own.write_text("".join(json.dumps(r) + "\n" for r in rows("kd_fix.jsonl")[1:]))
    step = ScaleKD.model_validate({"data": str(own)})
    assert len(step._lint(recipe, str(teacher), step.trace_rows())) == 5
    (record,) = recipe.calib.lint_records()
    assert (record["file"], record["unclosed_think"], record["appended_eos"]) == (
        f"scale_kd.data {own}",
        2,
        2,
    )


@pytest.mark.parametrize("ids", [[999999], [-1], None, [1, None]])
def test_ids_that_are_not_the_tokenizers_are_counted_not_decoded(ids):
    """A fast decode drops an unknown id silently, so such a row is not decoded (and a null is no TypeError);
    the engine's own id checks judge it against the model's vocabulary."""
    found = lint_rows([{"text": "w1"}, {"input_ids": ids}], r1_tokenizer(), "kd.jsonl")
    assert (found.bad_ids, found.refusal("kd.jsonl")) == (1, None)
    assert found.lint_record()["bad_ids"] == 1


def test_a_token_file_whose_tokenizer_cannot_load_builds_and_says_the_lint_did_not_run(tmp_path, monkeypatch):
    from qda_quant.engine import calib as engine_calib

    (tmp_path / "tokenizer.json").write_text("{}")
    path = tmp_path / "ids.jsonl"
    path.write_text(f'{{"tokenizer": "{fingerprint(tmp_path)}"}}\n{{"input_ids": [1, 2, 3]}}\n')
    source = FileSource(path=str(path), samples=1, seqlen=8)
    assert [s.ids.tolist() for s in engine_calib.load_samples(source, tmp_path)] == [[1, 2, 3]]
    (record,) = source.lint_records()
    assert record["check"].startswith("not run: the model's tokenizer cannot be loaded"), record


def _plan_model(path: Path) -> None:
    m: Any = tiny_qwen3(layers=1)  # a transformers model: save_pretrained
    m.save_pretrained(path)
    r1_tokenizer(path)


@pytest.mark.parametrize("bad", [999999, -1, None, 1.5])
def test_kd_out_of_vocab_ids_refuse_during_full_plan(tmp_path, bad):
    """scale_kd.data ids the model's vocabulary does not have refuse at plan time, naming the file, row and id."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.steps.rtn import RTN
    from qda_quant.steps.scale_kd import ScaleKD
    from qda_quant.target import Target

    root = tmp_path / "model"
    _plan_model(root)
    data = tmp_path / "traces.jsonl"
    rows = [
        {"prompt_ids": [1], "completion_ids": [2, 3, 4]},
        {"prompt_ids": [1], "completion_ids": [bad, 2, 3]},
    ]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    step = ScaleKD(data=str(data), teacher=str(root), steps=1, seqs_per_forward=1)
    recipe = Recipe(steps=[RTN(), step], calib=FileSource(path=str(tmp_path / "unused")))
    match = rf"scale_kd\.data {data}: row 1 \(counted from 0\) holds token ids outside the model's vocab .*completion_ids\[0\]"
    with pytest.raises(Refused, match=match):
        compile_plan(str(root), recipe, Format.parse("w4a16-sym-g32"), Target(), [torch.device("cpu")])


def test_a_mix_lints_every_raw_row_of_its_parts_not_only_those_it_picks(tmp_path, monkeypatch):
    """A part's ChatML row refuses though the mix picks one row; a clean mix records each part file it read."""
    from qda_quant.calib.sources import MixPart, MixSource

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    model = tmp_path / "model"
    r1_tokenizer(model)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "".join(json.dumps(r) + "\n" for r in [{"text": "w1 w2 w3 w4"}] * 3 + rows("calib_big.jsonl")[:1])
    )
    source = MixSource(parts=[MixPart(path=str(bad))], rows=1, samples=1, seqlen=2, mix_seed=1)
    with pytest.raises(Refused, match=r"bad\.jsonl: 1 of 4 text rows .*another model"):
        load_samples(source, model)
    own = tmp_path / "own.jsonl"
    own.write_text(
        "".join(json.dumps(r) + "\n" for r in [{"text": "w1 w2 w3 w4"}] * 3 + rows("calib_big.jsonl")[2:])
    )
    source = MixSource(parts=[MixPart(path=str(own))], rows=1, samples=1, seqlen=2, mix_seed=1)
    load_samples(source, model)
    by_file = {r["file"]: r for r in source.lint_records()}
    assert (by_file[str(own)]["rows"], by_file[str(own)]["own"], by_file[str(own)]["unclosed_think"]) == (
        4,
        1,
        1,
    )


def test_packed_kd_build_records_its_data_lint(tmp_path):
    """A ScaleKD build on a packed checkpoint writes its own data lint, not the source build's receipt."""
    from qda_quant.steps.rtn import RTN

    root = tmp_path / "model"
    _plan_model(root)
    data = tmp_path / "trace.jsonl"
    data.write_text(json.dumps({"prompt_ids": [1], "completion_ids": [2, 3, 4]}) + "\n")
    packed = build(
        str(root),
        Recipe(steps=[RTN()], calib=FileSource(path=str(tmp_path / "unused"))),
        format="w4a16-sym-g32",
        out_dir=tmp_path / "packed",
        devices=["cpu"],
        log=lambda _: None,
    )
    step = _kd_step(data, root)
    lines: list[str] = []
    output = build(
        str(packed),
        Recipe(steps=[step], calib=FileSource(path=str(tmp_path / "unused"))),
        out_dir=tmp_path / "trained",
        devices=["cpu"],
        log=lines.append,
    )
    assert len([line for line in lines if line.startswith("data lint")]) == 1, lines
    info = json.loads((Path(output) / "build.json").read_text())
    assert info["data_lint"]["files"][0]["file"] == f"scale_kd.data {data}"


def _kd_step(data: Path, teacher: Path) -> Any:
    from qda_quant.steps.scale_kd import ScaleKD

    return ScaleKD(
        data=str(data),
        teacher=str(teacher),
        steps=1,
        warmup=1,
        batch_tokens=8,
        max_len=8,
        head_chunk=16,
        seqs_per_forward=1,
        attn="eager",
    )


def test_float_kd_build_reads_trace_once_and_logs_once(tmp_path, monkeypatch):
    """RTN then ScaleKD: the trace file is read once (training uses the rows the plan checked), one lint block."""
    from qda_quant.steps.rtn import RTN

    root = tmp_path / "model"
    _plan_model(root)
    data = tmp_path / "trace.jsonl"
    data.write_text(json.dumps({"prompt_ids": [1], "completion_ids": [2, 3, 4]}) + "\n")
    original = Path.read_text
    reads = []

    def counted(path, *args, **kwargs):
        if path == data:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    lines: list[str] = []
    output = build(
        str(root),
        Recipe(steps=[RTN(), _kd_step(data, root)], calib=FileSource(path=str(tmp_path / "unused"))),
        format="w4a16-sym-g32",
        out_dir=tmp_path / "trained",
        devices=["cpu"],
        log=lines.append,
    )
    blocks = [line for line in lines if line.startswith("data lint")]
    info = json.loads((Path(output) / "build.json").read_text())
    assert (len(reads), len(blocks), "files" in info["data_lint"]) == (1, 1, True), lines


def test_a_kd_training_read_without_a_plan_still_lints(tmp_path):
    """Training reads the file only when no plan read it; that read is linted as the plan's would be."""
    from types import SimpleNamespace

    root = tmp_path / "model"
    r1_tokenizer(root)
    data = tmp_path / "trace.jsonl"
    data.write_text((FIXTURES / "kd_fix.jsonl").read_text().splitlines()[0] + "\n")
    recipe = Recipe(steps=[], calib=FileSource(path=str(tmp_path / "unused")))
    with pytest.raises(Refused, match="another model"):
        _kd_step(data, root)._rows(SimpleNamespace(recipe=recipe, log=lambda _: None), str(root))
