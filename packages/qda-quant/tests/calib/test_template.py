"""Calibration text rows in another model's chat template are refused (`qda_quant.calib.template`)."""

import json

import pytest
from qda_quant.calib.sources import FileSource, MixSource
from qda_quant.calib.template import check_rows
from qda_quant.engine.calib import load_samples
from qda_quant.step import Refused

R1 = (
    "<｜begin▁of▁sentence｜>{% for m in messages %}{% if m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
    "{% else %}<｜Assistant｜>{{ m['content'] }}<｜end▁of▁sentence｜>{% endif %}{% endfor %}"
)
CHATML = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
R1_ROW = {"text": "<｜User｜>w1 w2 w3<｜Assistant｜>w4 w5 w6 w7 w8<｜end▁of▁sentence｜>"}
CHATML_ROW = {"text": "<|im_start|>user\nw1 w2 w3<|im_end|>\n<|im_start|>assistant\nw4 w5 w6 w7 w8<|im_end|>"}
PLAIN_ROW = {"text": "w1 w2 w3 w4 w5 w6 w7 w8"}
MESSAGES_ROW = {
    "messages": [{"role": "user", "content": "w1 w2"}, {"role": "assistant", "content": "w3 w4 w5"}]
}


def tokenizer(path, template):
    """A tiny WordLevel tokenizer with `template` as its chat template, saved at `path` (a model directory)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.WordLevel({f"w{i}": i for i in range(127)} | {"[UNK]": 127}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.chat_template = template
    fast.save_pretrained(path)
    return fast


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_r1_rows_on_an_r1_tokenizer_are_its_own(tmp_path):
    found = check_rows([R1_ROW, R1_ROW], tokenizer(tmp_path / "r1", R1))
    assert (found.own, found.plain, found.foreign, found.produced) == (2, 0, 0, ["deepseek"])
    assert found.refusal("calib.jsonl") is None


def test_chatml_rows_on_an_r1_tokenizer_are_foreign(tmp_path):
    found = check_rows([PLAIN_ROW, R1_ROW, CHATML_ROW, CHATML_ROW], tokenizer(tmp_path / "r1", R1))
    assert (found.own, found.plain, found.foreign) == (1, 1, 2)
    why = found.refusal("calib.jsonl")
    assert why is not None
    for part in ("calib.jsonl", "2 of 4 text rows", "row 2", "'<|im_start|>' of chatml", "deepseek"):
        assert part in why, why
    assert why.endswith("write the rows with this model's chat template, or as messages rows")


def test_chatml_rows_on_a_qwen_tokenizer_are_its_own(tmp_path):
    found = check_rows([CHATML_ROW], tokenizer(tmp_path / "qwen", CHATML))
    assert (found.own, found.foreign, found.produced) == (1, 0, ["chatml"])


def test_plain_and_messages_rows_pass_without_a_tokenizer():
    found = check_rows([PLAIN_ROW, MESSAGES_ROW, {"input_ids": [1, 2]}], None)
    assert (found.own, found.plain, found.foreign) == (1, 1, 0)
    assert found.refusal("x") is None


def test_token_id_rows_are_not_classified_even_with_text_beside_their_ids():
    """The engine calibrates a token-id row on its ids, so text beside them is never checked (build and launch)."""
    found = check_rows([{"input_ids": [1, 2], "text": CHATML_ROW["text"]}, PLAIN_ROW], None)
    assert (found.own, found.plain, found.foreign) == (0, 1, 0)


def test_a_model_without_a_chat_template_has_no_own_markers(tmp_path):
    found = check_rows([CHATML_ROW, PLAIN_ROW], tokenizer(tmp_path / "base", None))
    assert (found.produced, found.foreign, found.plain) == ([], 1, 1)


def test_a_build_refuses_foreign_rows_and_records_the_counts(tmp_path):
    model = tmp_path / "r1"
    tokenizer(model, R1)
    bad = jsonl(tmp_path / "chatml.jsonl", [R1_ROW, CHATML_ROW])
    with pytest.raises(Refused, match=r"chatml\.jsonl: 1 of 2 text rows .* row 1, .*'<\|im_start\|>'"):
        load_samples(FileSource(path=str(bad), samples=1, seqlen=4), model)
    good = FileSource(
        path=str(jsonl(tmp_path / "r1.jsonl", [R1_ROW, PLAIN_ROW, MESSAGES_ROW])), samples=2, seqlen=4
    )
    assert len(load_samples(good, model)) == 2
    assert good.template_record() == {
        "check": "calibration rows vs the model's chat template",
        "produced": ["deepseek"],
        "own": 2,
        "plain": 1,
        "foreign": 0,
    }
    load_samples(good, model)  # a second read replaces the record
    assert (good.template_record() or {}).get("own") == 2


def test_a_mix_refuses_a_foreign_part_and_sums_the_parts(tmp_path):
    model = tmp_path / "r1"
    tokenizer(model, R1)
    own = {"file": {"path": str(jsonl(tmp_path / "a.jsonl", [R1_ROW] * 4)), "seqlen": 4}, "weight": 1}
    plain = {"file": {"path": str(jsonl(tmp_path / "b.jsonl", [PLAIN_ROW] * 4)), "seqlen": 4}, "weight": 1}
    mix = MixSource.model_validate({"parts": [own, plain], "samples": 2})
    load_samples(mix, model)
    assert {k: (mix.template_record() or {}).get(k) for k in ("own", "plain", "foreign")} == {
        "own": 4,
        "plain": 4,
        "foreign": 0,
    }
    load_samples(mix, model)  # a second read of the mix does not double the counts
    assert (mix.template_record() or {}).get("own") == 4
    foreign = {"file": {"path": str(jsonl(tmp_path / "c.jsonl", [CHATML_ROW] * 4)), "seqlen": 4}, "weight": 1}
    bad = MixSource.model_validate({"parts": [own, foreign], "samples": 2})
    with pytest.raises(Refused, match=r"c\.jsonl: 4 of 4 text rows"):
        load_samples(bad, model)


# ------------------------------------------------------------------------------------- data pools shipped to an agent

R1_TEXT, CHATML_TEXT = R1_ROW["text"], CHATML_ROW["text"]


def r1_ids(tmp_path):
    """An R1 tokenizer that also spells ChatML's markers as text tokens (a byte-level BPE decodes them back)."""
    from qda_quant.calib.template import FAMILIES

    fast = tokenizer(tmp_path / "r1", R1)
    fast.add_tokens([*FAMILIES["deepseek"], "<｜end▁of▁sentence｜>", *FAMILIES["chatml"]])
    return fast


def ids_row(fast, text, split=None):
    ids = fast.encode(text, add_special_tokens=False)
    if split is None:
        return {"input_ids": ids}
    return {"prompt_ids": ids[:split], "completion_ids": ids[split:]}


def test_a_pool_of_messages_plain_text_and_plain_token_rows_passes(tmp_path):
    from qda_quant.calib.template import check_file

    fast = r1_ids(tmp_path)
    plain = PLAIN_ROW["text"]
    rows = [
        PLAIN_ROW,
        MESSAGES_ROW,
        {"prompt": "w1 w2", "completion": "w3"},
        ids_row(fast, plain),
        ids_row(fast, plain, 3),
    ]
    found = check_file(jsonl(tmp_path / "set.jsonl", rows), lambda: fast)
    assert (found.plain, found.messages, found.rendered, found.undecodable) == (4, 1, 0, 0), found
    assert found.refusal("data set corpus") is None


@pytest.mark.parametrize("family", ["own", "foreign"])
@pytest.mark.parametrize("shape", ["text", "prompt", "input_ids", "prompt_ids"])
def test_a_rendered_row_in_a_pool_is_refused_whatever_its_family(tmp_path, shape, family):
    """A rendered row is refused as text, prompt + completion, or token ids; the model's own template included."""
    from qda_quant.calib.template import check_file

    fast = r1_ids(tmp_path)
    text = R1_TEXT if family == "own" else CHATML_TEXT
    bad = {
        "text": {"text": text},
        "prompt": {"prompt": text[:20], "completion": text[20:]},
        "input_ids": ids_row(fast, text),
        "prompt_ids": ids_row(fast, text, 4),
    }[shape]
    path = jsonl(tmp_path / "set.jsonl", [PLAIN_ROW, bad, MESSAGES_ROW, bad, bad, bad])
    found = check_file(path, lambda: fast)
    assert (found.rendered, found.rendered_at[0][0]) == (4, 2), found
    why = found.refusal("data set corpus")
    assert why is not None
    marker = "'<｜User｜>' of deepseek" if family == "own" else "'<|im_start|>' of chatml"
    for part in (
        f"data set corpus {path}",
        "of 6 rows, 4 rows hold pre-rendered chat markup",
        "lines 2, 4, 5;",
        marker,
    ):
        assert part in why, why
    assert why.endswith(
        "a pool shipped to the agent holds un-rendered text: messages rows, prompt fields or plain text"
    )


def test_token_rows_the_tokenizer_cannot_decode_are_refused(tmp_path):
    from qda_quant.calib.template import check_file

    fast = r1_ids(tmp_path)
    found = check_file(jsonl(tmp_path / "set.jsonl", [{"input_ids": ["x"]}, PLAIN_ROW]), lambda: fast)
    assert (found.undecodable, found.undecodable_at) == (1, [1])
    assert "1 token-id rows are not ids of this model's tokenizer" in str(found.refusal("data set corpus"))


def test_a_pool_without_token_rows_needs_no_tokenizer(tmp_path):
    from qda_quant.calib.template import check_file

    def no_tokenizer():
        raise AssertionError("the tokenizer was loaded")

    path = tmp_path / "set.jsonl"
    path.write_text('{"text": "w1"}\n\n[1, 2]\nnot json\n{"id": 3}\n' + json.dumps(MESSAGES_ROW) + "\n")
    found = check_file(path, no_tokenizer)
    assert (found.plain, found.messages, found.other, found.refusal("x")) == (1, 1, 3, None)
    found = check_file(jsonl(tmp_path / "r.jsonl", [CHATML_ROW]), no_tokenizer)
    assert found.rendered == 1
