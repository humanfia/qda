"""`qda calib render`: pool rows normalised to turns, then rendered in the model's own template and nothing else."""

import json
from collections import Counter

import pytest
from qda_quant.calib import render as R
from qda_quant.step import Refused


class ThinkingChatML:
    """A Qwen3-like template: reasoning in `<think>`, an empty think block on a final answer without one."""

    chat_template = "chatml-with-think"

    def apply_chat_template(self, messages, tokenize=False, enable_thinking=True, **kw):
        out = ""
        for m in messages:
            body = m["content"]
            if m["role"] == "assistant":
                r = m.get("reasoning_content")
                body = (
                    f"<think>\n{r}\n</think>\n\n{body}"
                    if r and enable_thinking
                    else f"<think>\n\n</think>\n\n{body}"
                )
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        return out


class Llama3:
    """A template with no thinking format: `reasoning_content` is ignored, as a Jinja template ignores it."""

    chat_template = "llama3"

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "".join(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>" for m in messages
        )


D1 = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nIt is 4.<|im_end|>\n"


def test_chatml_text_is_parsed_back_into_turns_and_its_empty_think_stripped():
    counts = Counter()
    got = R.normalise({"text": D1}, counts)
    assert got == {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."},
        ]
    }
    assert counts == {"empty think stripped": 1, "ChatML text parsed into turns": 1}


def test_an_inline_think_moves_into_reasoning_content():
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\nadd\n</think>\n\n4"},
        ]
    }
    counts = Counter()
    got = R.normalise(row, counts)
    assert got["messages"][1] == {"role": "assistant", "content": "4", "reasoning_content": "add"}
    assert counts == {"inline think moved to reasoning_content": 1}


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ({"messages": [{"role": "user", "content": "q"}]}, "no assistant answer"),
        (
            {"text": "<|start_header_id|>user<|end_header_id|>hi"},
            "pre-rendered in a template other than ChatML",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "<|im_start|>"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            "chat markup inside a message",
        ),
        ({"input_ids": [1, 2]}, "token ids or no row"),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a <think>x</think>"},
                ]
            },
            "think markup inside an answer",
        ),
    ],
)
def test_rows_that_cannot_be_normalised_are_left_out_by_reason(row, why):
    assert R.normalise(row, Counter()) == why


def test_reasoning_is_rendered_in_the_models_own_thinking_format():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "4", "reasoning_content": "add"},
            ]
        }
    ]
    out, record = R.render_rows(rows, ThinkingChatML(), "keep")
    assert out == [
        {
            "text": "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nadd\n</think>\n\n4<|im_end|>\n"
        }
    ]
    assert record["template"]["reasoning"] is True and record["turns with reasoning, rendered"] == 1


def test_off_renders_the_templates_thinking_off_form():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "4", "reasoning_content": "add"},
            ]
        }
    ]
    out, record = R.render_rows(rows, ThinkingChatML(), "off")
    assert "add" not in out[0]["text"] and "<think>\n\n</think>" in out[0]["text"]
    assert record["turns with reasoning, reasoning dropped"] == 1


def test_a_template_without_a_thinking_format_gets_no_foreign_think_marker():
    """d1's Qwen markup, parsed and rendered for a Llama-3 model, holds neither ChatML nor think markers."""
    counts = Counter()
    rows = [
        R.normalise({"text": D1}, counts),
        R.normalise(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<think>r</think>a"},
                ]
            },
            counts,
        ),
    ]
    out, record = R.render_rows(rows, Llama3(), "keep")
    text = "".join(r["text"] for r in out)
    assert "<think>" not in text and "</think>" not in text and "<|im_start|>" not in text
    assert record["template"]["reasoning"] is False and record["template"]["families"] == ["llama3"]
    assert record["turns with reasoning, reasoning dropped"] == 1


def test_a_render_holding_a_marker_the_template_never_produces_is_refused():
    class Leaky(Llama3):
        def apply_chat_template(self, messages, tokenize=False, **kw):
            text = super().apply_chat_template(messages)
            return text + ("<think></think>" if len(messages) > 2 else "")

    rows = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]
        }
    ]
    with pytest.raises(Refused, match="never produces"):
        R.render_rows(rows, Leaky(), "keep")


def test_render_mixes_the_named_files_by_weight_and_records_its_inputs(tmp_path):
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            + "\n"
            for i in range(6)
        )
    )
    d1 = tmp_path / "d1.jsonl"
    d1.write_text(
        "".join(json.dumps({"text": D1.replace("2+2", f"{i}+{i}")}) + "\n" for i in range(6))
        + '{"text": "<|im_start|>user\\nonly a prompt<|im_end|>\\n"}\n'
    )
    files, source = R.pool_files(str(tmp_path), ["chat", "d1"])
    assert source == {"dir": str(tmp_path.resolve())}
    weights = R.parse_names("chat=1,d1=2")
    rows, record = R.render(ThinkingChatML(), files, weights, rows=6, seed=0, thinking="keep")
    again, _ = R.render(ThinkingChatML(), files, weights, rows=6, seed=0, thinking="keep")
    assert rows == again and len(rows) == 6
    assert sum("q" in r["text"] and "a" in r["text"] and "+" not in r["text"] for r in rows) == 2
    assert record["left_out"] == {"chat": {}, "d1": {"no assistant answer": 1}}
    assert len(record["thinking"]["template"]["sha256"]) == 64
    assert "reasoning" in record and set(record["files"]) == {"chat", "d1"}


class Words(ThinkingChatML):
    """The thinking template, tokenized one id per whitespace-separated word (as `tok(text)` is called)."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}


def _chats(path, n):
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_render_keeps_as_many_rows_as_the_builds_samples_need(tmp_path):
    """256 rows pack to fewer than 128 samples here: render mixes more and keeps the fewest that give 128."""
    from qda_quant.engine.calib import cut

    files = {"chat": _chats(tmp_path / "chat.jsonl", 1000)}
    tok = Words()
    short, _ = R.render(tok, files, {"chat": 1}, rows=256, seed=0, thinking="keep")
    ids = lambda rows: [tok(r["text"])["input_ids"] for r in rows]  # noqa: E731
    assert len(cut(ids(short), 20)) < 128
    rows, record = R.render(tok, files, {"chat": 1}, samples=128, seqlen=20, seed=0, thinking="keep")
    assert len(cut(ids(rows), 20)) >= 128 > len(cut(ids(rows[:-1]), 20))  # the build's own packing
    assert record["samples"] == 128 and record["seqlen"] == 20 and record["rows"] == len(rows) > 256


def test_a_pool_too_small_for_the_samples_is_refused_at_render(tmp_path):
    files = {"chat": _chats(tmp_path / "chat.jsonl", 300)}
    with pytest.raises(Refused, match="128 samples of 20 tokens need more rows than the pools give"):
        R.render(Words(), files, {"chat": 1}, samples=128, seqlen=20, seed=0, thinking="keep")


def test_a_pool_too_small_for_the_rows_is_refused_with_its_reasons(tmp_path):
    files = {"chat": _chats(tmp_path / "chat.jsonl", 3)}
    with pytest.raises(Refused, match="rows left out of each file, by reason"):
        R.render(Words(), files, {"chat": 1}, rows=10, seed=0, thinking="keep")


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"rows": 4, "samples": 1}, "not both"),  # --samples without --seqlen was silently ignored
        ({"rows": 4, "samples": 1, "seqlen": 20}, "not both"),
        ({"rows": 0}, "at least 1"),
        ({"samples": 0, "seqlen": 20}, "at least 1"),
        ({"samples": 1, "seqlen": 0}, "at least 1"),
    ],
)
def test_a_row_count_that_is_ambiguous_or_zero_is_refused(tmp_path, kwargs, why):
    files = {"chat": _chats(tmp_path / "chat.jsonl", 10)}
    with pytest.raises(Refused, match=why):
        R.render(Words(), files, {"chat": 1}, seed=0, thinking="keep", **kwargs)
