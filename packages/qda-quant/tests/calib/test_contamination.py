"""The calibration contamination filter: drop reference-prompt matches, log them, never refuse.

The reference side is an index of sha256 digests only. A row is dropped when its PROMPT matches by id, by
normalized text or by a 20-word window under either normalization (the guard's or the regex normalization); an answer
body is never compared, so self-generated calibration is kept. A window that is generic scaffolding (a
problem's input/output section, a constraint template, a numeric example) is exempted and recorded, never
dropped; a copied problem statement is not scaffolding, whatever words it is made of.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from qda_quant.calib.contamination import (
    PromptIndex,
    filter_rows,
    prompt_texts,
)
from qda_quant.calib.sources import FileSource, SelfgenSource, parse_source
from qda_quant.engine.build import build
from qda_quant.recipe import Recipe
from qda_quant.step import Refused

from ..engine.test_build import FakeWriter, TinyRTN
from ..engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids, writes

REF_A = (
    "You are given an array of n integers. In one move you may increase any element by one. Your task is "
    "to calculate the minimum number of moves needed to make every element of the array equal."
)
REF_B = (
    "The second line of the test case contains n integers a_1, a_2, ..., a_n (1 <= a_i <= 10^9), where a_i "
    "is the i-th element of the array and every test case is independent of the others."
)
REF_C = "Explain how photosynthesis converts light energy into chemical energy stored in glucose molecules."

#: A synthetic passage in Thai, a script written without spaces, longer than the character window, and unrelated English padding to embed a copy in.
THAI = "ใต้สะพานหินเก่าริมทะเลสาบมีนกกระสาขาวมาพักอยู่เสมอยามเช้าผิวน้ำเป็นระลอกเล็กๆและเงาภูเขาไกลๆค่อยๆชัดขึ้น"
THAI_PADDING = (
    "Please read the following note about a lake carefully and then write one short calm sentence about "
    "the weather and the water near the hills before you classify the whole note as calm or stormy, and "
    "finally add one short remark about the colour of the sky in the early morning near the shore"
)

#: A synthetic reference prompt per script and an unrelated prompt in the same script. The copy tests change
#: whitespace only, so no window can hit and the normalized-text arm is the only one that can drop the row.
NON_LATIN = [
    pytest.param("หาผลรวมของสมาชิกทั้งหมด", "เรือประมงแล่นผ่านโขดหินกลับเข้าท่าเรือยามค่ำคืน", id="thai"),
    pytest.param(
        "सरणी में सभी तत्वों का योग ज्ञात कीजिए",
        "नदी के किनारे बच्चे पतंग उड़ा रहे हैं",
        id="devanagari",
    ),
    pytest.param(
        "احسب مجموع جميع العناصر في المصفوفة",
        "تفتح الأزهار في الحديقة مع بداية الربيع",
        id="arabic",
    ),
    pytest.param(
        "Найдите сумму всех элементов массива",
        "Дети играют в парке возле старого моста",
        id="cyrillic",
    ),
]

#: Scripts written with spaces between words; each passage is longer than one 20-word window.
SPACED = [
    pytest.param(
        "नदी के किनारे बच्चे पतंग उड़ा रहे हैं और हवा धीरे धीरे तेज़ हो रही है शाम को बारिश हो सकती है",
        id="devanagari",
    ),
    pytest.param(
        "الأطفال يلعبون في الحديقة قرب النافورة القديمة وتنمو الأزهار بين الأشجار العالية في الصباح الباكر "
        "بينما تغني الطيور فوق السور الحجري القديم",
        id="arabic",
    ),
    pytest.param(
        "Дети играют в парке возле старого моста а над рекой летают белые чайки и рыбаки чинят сети на "
        "берегу ранним утром",
        id="cyrillic",
    ),
]


def jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def index_file(tmp_path):
    return jsonl(
        tmp_path / "refs.jsonl",
        [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": REF_B}, {"id": "r3", "prompt": REF_C}],
    )


@pytest.fixture
def index(index_file):
    return PromptIndex.read(index_file)


def test_a_row_sharing_a_reference_window_is_flagged(index):
    hit = index.hit({"id": "x", "text": REF_A.replace("minimum number of moves", "smallest count of shifts")})
    assert hit is not None and hit["guard"] and hit["regex"]


def test_the_index_is_hashes_only(index, index_file):
    data = index.to_dict()
    for key in ("ids", "texts", "guard_windows", "regex_windows"):
        assert data[key] and all(len(v) == 64 and int(v, 16) >= 0 for v in data[key])
    assert "photosynthesis" not in json.dumps(data)
    saved = index.save(index_file.with_name("index.json"))
    assert "photosynthesis" not in saved.read_text() and "glucose" not in saved.read_text()


def test_drop_rules_id_normalized_text_and_a_20_word_window(index):
    rows = [
        {"id": "r1", "text": "a completely unrelated calibration passage about bicycles"},
        {"id": "clean", "text": "Alpha\nBeta   gamma"},
        {"id": "window", "text": "prefix words here " + " ".join(REF_A.split()[:20]) + " and a tail"},
        {"id": "kept", "text": "An unrelated calibration passage about marine biology and coral reefs."},
    ]
    index = PromptIndex.from_rows(
        [
            {"id": "r1", "prompt": "Alpha Beta Gamma"},
            {"id": "r2", "prompt": REF_B},
            {"id": "r3", "prompt": REF_A},
        ]
    )
    kept, dropped, exempted = filter_rows(rows, index)
    assert [r["id"] for r in kept] == ["kept"]
    assert [d["id"] for d in dropped] == ["r1", "clean", "window"]
    assert [d["match"] for d in dropped] == [["id"], ["text"], ["guard", "regex"]]
    assert exempted == []


def test_a_copied_problem_instruction_is_not_exempt(index):
    """An adversarial fixture: common words do not make a copied task scaffolding."""
    ref = PromptIndex.from_rows(
        [
            {
                "prompt": "Print the number of positive integers in the given array for which the value is the "
                "same as its index"
            }
        ]
    )
    row = {
        "id": "cand-1",
        "prompt": "Please solve this task: Print the number of positive integers in the "
        "given array for which the value is the same as its index",
    }
    kept, dropped, exempted = filter_rows([row], ref)
    assert kept == [] and exempted == [] and dropped[0]["id"] == "cand-1"


def test_generic_windows_are_exempted_and_recorded(index):
    rows = [
        {"id": "io", "text": "Some problem statement.\n\nInput\n\n" + REF_B + "\n\nOutput\n\nOne integer."},
        {"id": "io-ref", "text": REF_B},
    ]
    ref = PromptIndex.from_rows([{"id": "r", "prompt": REF_B}])
    kept, dropped, exempted = filter_rows(rows, ref)
    # the same text verbatim is the prompt itself, never exempted; only the window-only row is scaffolding
    assert [r["id"] for r in kept] == ["io"]
    assert [d["id"] for d in dropped] == ["io-ref"]
    assert dropped[0]["match"] == ["guard", "regex", "text"]  # an exact hit records every test that hit
    assert [e["id"] for e in exempted] == ["io"] and exempted[0]["rules"] == ["code-io"]


def test_recurrence_never_makes_a_window_generic():
    """Frequency is not genericity: a template in two references, or in many calibration rows, still drops."""
    template = (
        "write in exactly 3 paragraphs separated by a blank line and each paragraph should have a title "
        "wrapped in double angular brackets"
    )
    ref = PromptIndex.from_rows(
        [
            {"prompt": "Discuss marine policy. " + template},
            {"prompt": "Discuss bicycle design. " + template},
        ]
    )
    # two reference prompts share it: still reference content, so the copied row drops
    kept, dropped, exempted = filter_rows([{"prompt": "Discuss quantum computing. " + template}], ref)
    assert kept == [] and dropped and exempted == []
    # many calibration rows share it: a batch of suspect rows cannot certify itself clean
    ref = PromptIndex.from_rows([{"prompt": "Discuss marine policy. " + template}])
    rows = [{"prompt": f"Discuss topic {i}. " + template} for i in range(6)]
    kept, dropped, exempted = filter_rows(rows, ref)
    assert kept == [] and len(dropped) == 6 and exempted == []


def test_a_numeric_window_is_exempt_by_its_shape():
    numbers = " ".join(str(i) for i in range(1, 30))
    kept, dropped, exempted = filter_rows(
        [{"prompt": "0 " + numbers}], PromptIndex.from_rows([{"prompt": numbers}])
    )
    assert kept and dropped == [] and exempted[0]["rules"] == ["numeric"]


def test_answers_are_never_compared_both_selfgen_render_modes(index):
    passage = "amber birch cedar dolphin elm falcon glacier harbor iris juniper kestrel lantern"
    ref = PromptIndex.from_rows([{"prompt": passage}])
    messages = {
        "messages": [
            {"role": "user", "content": "Describe a blue bicycle."},
            {"role": "assistant", "content": passage},
        ]
    }
    rendered = {
        "text": "<|im_start|>user\nDescribe a blue bicycle.<|im_end|>\n<|im_start|>assistant\n"
        + passage
        + "<|im_end|>"
    }
    assert ref.hit(messages) is None
    assert ref.hit(rendered) is None
    assert prompt_texts(rendered) == ["Describe a blue bicycle."]


def test_a_selfgen_prompt_drop_takes_its_n_answer_rows(tmp_path, index):
    prompts = jsonl(
        tmp_path / "p.jsonl", [{"id": "p1", "prompt": "unrelated prompt one"}, {"id": "p2", "prompt": REF_C}]
    )
    src = SelfgenSource.model_validate(
        {"prompts": str(prompts), "n": 2, "contamination_index": str(index.save(tmp_path / "i.json"))}
    )
    rows = [{"id": f"p{j}", "text": f"answer {k}"} for j in (1, 2) for k in range(2)]
    kept = src.screen(rows)
    assert len(kept) == 2 and all(r["id"] == "p1" for r in kept)
    assert src.dropped == [{"row": 1, "id": "p2", "match": ["text"], "rows": 2}]
    assert src.contamination_record()["dropped"] == 2


def test_an_unattributable_selfgen_cache_is_refused(tmp_path, index):
    prompts = jsonl(tmp_path / "p.jsonl", [{"id": "p1", "prompt": "unrelated"}])
    src = SelfgenSource.model_validate(
        {"prompts": str(prompts), "n": 2, "contamination_index": str(index.save(tmp_path / "i.json"))}
    )
    with pytest.raises(Refused, match="cannot be attributed"):
        src.screen([{"text": "only one row"}])


def test_a_search_build_with_no_index_does_not_start(monkeypatch, tmp_path):
    monkeypatch.setenv("QDA_CONTAMINATION_REQUIRED", "1")
    monkeypatch.delenv("QDA_CONTAMINATION_INDEX", raising=False)
    with pytest.raises(Refused, match="infrastructure error, not the candidate"):
        FileSource(path=str(tmp_path / "c.jsonl")).index()


def test_a_recipe_cannot_name_its_own_index_in_a_search(monkeypatch, tmp_path, index_file):
    """The agent writes the recipe: its index may neither replace the host's nor stand in for a missing one."""
    own = PromptIndex.from_rows([{"id": "agent", "prompt": "anything at all"}]).save(tmp_path / "own.json")
    src = FileSource(path=str(tmp_path / "c.jsonl"), contamination_index=str(own))
    monkeypatch.setenv("QDA_CONTAMINATION_REQUIRED", "1")
    monkeypatch.delenv("QDA_CONTAMINATION_INDEX", raising=False)
    with pytest.raises(Refused, match="names its own contamination index"):
        src.index()
    monkeypatch.setenv("QDA_CONTAMINATION_INDEX", str(index_file))
    src = FileSource(path=str(tmp_path / "c.jsonl"), contamination_index=str(own))
    with pytest.raises(Refused, match="names its own contamination index"):
        src.index()
    # outside a search the same recipe index is allowed
    monkeypatch.delenv("QDA_CONTAMINATION_REQUIRED")
    monkeypatch.delenv("QDA_CONTAMINATION_INDEX")
    assert src.index().prompts == 1


def test_a_substantive_prompt_after_an_input_heading_still_drops():
    """Section placement is not itself generic I/O; the sentence must read as I/O format."""
    text = (
        "Read the task below.\n\nInput\n\n"
        "Print the number of positive integers in the given array for which the value is the same as its "
        "index and return the count in the only output line"
    )
    ref = PromptIndex.from_rows([{"prompt": text}])
    row = {"id": "copied", "text": "my preamble " + text}
    kept, dropped, exempted = filter_rows([row], ref)
    assert kept == [] and exempted == [] and dropped[0]["id"] == "copied"


def test_a_numeric_window_with_a_real_name_is_not_exempt():
    """Numeric means all numbers, not mostly numbers beside a constraint or a name."""
    numbers = " ".join(str(i) for i in range(1, 30))
    ref = PromptIndex.from_rows([{"prompt": "task Alpha " + numbers}])
    kept, dropped, exempted = filter_rows([{"text": "prefix task Alpha " + numbers}], ref)
    # a shared window holding a name is not numeric scaffolding, so the whole row drops
    assert kept == [] and dropped and exempted == []


@pytest.mark.parametrize(
    "body",
    [
        "{}",
        '{"kind": "qda-calibration-contamination", "window": 20, "chars": 32, "prompts": 1,',
        json.dumps(
            {
                "kind": "qda-calibration-contamination",
                "window": 20,
                "chars": 32,
                "prompts": 1,
                "ids": ["not-a-digest"],
                "texts": [],
                "guard_windows": [],
                "regex_windows": [],
            }
        ),
        json.dumps(
            {
                "kind": "qda-calibration-contamination",
                "window": 20,
                "chars": 32,
                "prompts": 1,
                "ids": [],
                "texts": [],
                "guard_windows": [],
                "regex_windows": [],
            }
        ),
    ],
)
def test_a_malformed_or_empty_index_fails_to_start(monkeypatch, tmp_path, body):
    p = tmp_path / "index.json"
    p.write_text(body)
    monkeypatch.setenv("QDA_CONTAMINATION_REQUIRED", "1")
    monkeypatch.setenv("QDA_CONTAMINATION_INDEX", str(p))
    with pytest.raises(Refused, match="contamination"):
        FileSource(path=str(tmp_path / "c.jsonl")).index()


def test_a_corpus_keeps_its_filter_in_its_config(tmp_path, index):
    source = jsonl(tmp_path / "corpus.jsonl", [{"id": "1", "prompt": "unrelated wording entirely"}])
    src = parse_source(
        {
            "corpus": {
                "file": {"path": str(source)},
                "contamination_index": str(index.save(tmp_path / "i.json")),
            }
        }
    )
    again = parse_source(src.to_dict())
    assert again.contamination_index == src.contamination_index


def test_a_sources_index_key_is_contamination_index(tmp_path):
    """The index path is `contamination_index`; the top-level `contamination` is the agent's scan."""
    steps = [{"rtn": {}}]
    with pytest.raises(Refused, match=r"calib\.file\.contamination: unknown key"):
        Recipe.from_dict({"steps": steps, "calib": {"file": {"path": "c.jsonl", "contamination": "i.json"}}})
    got = Recipe.from_dict(
        {"steps": steps, "calib": {"file": {"path": "c.jsonl", "contamination_index": "i.json"}}},
        base=tmp_path,
    )
    assert got.calib.contamination_index == str(tmp_path / "i.json")  # a path, resolved beside the recipe


def test_scale_kd_traces_are_screened_at_their_prompts(monkeypatch, tmp_path, index):
    from qda_quant.intake import tokenizer as tokenizer_module
    from qda_quant.steps.scale_kd import ScaleKD

    class Decoder:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return REF_C if ids == [7, 8, 9] else "an unrelated prompt"

    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda _d: Decoder())
    traces = jsonl(
        tmp_path / "kd.jsonl",
        [{"prompt_ids": [7, 8, 9], "completion_ids": [1, 2]}, {"prompt_ids": [1], "completion_ids": [2]}],
    )
    recipe = Recipe(
        steps=[],
        calib=FileSource(
            path=str(tmp_path / "c.jsonl"), contamination_index=str(index.save(tmp_path / "i.json"))
        ),
    )
    step = ScaleKD.model_validate({"data": str(traces)})
    kept = step._screen(recipe, "unused", step.trace_rows())
    assert [r["prompt_ids"] for r in kept] == [[1]]
    assert recipe.calib.dropped[0]["id"] == "row 0"


def test_a_build_records_the_drops_and_keeps_running(tmp_path, index_file):
    """End to end: 3 rows dropped by id / normalized text / window, 1 kept, logged in build.json."""
    m = tiny_qwen3(layers=2)
    m.save_pretrained(tmp_path / "m")
    save_tiny_tokenizer(tmp_path / "m")
    calib = jsonl(
        tmp_path / "c.jsonl",
        [
            {"id": "r1", "text": "a perfectly ordinary calibration sentence about bicycles and lanes"},
            {"id": "dup", "text": "alpha\nbeta gamma"},
            {"id": "win", "text": "lead words " + " ".join(REF_A.split()[:20]) + " trailing words"},
            {"id": "keep", "text": " ".join(["w3"] * 8 + ["w4"] * 8)},
        ],
    )
    refs = jsonl(
        tmp_path / "refs2.jsonl",
        [
            {"id": "r1", "prompt": "reference one"},
            {"id": "r2", "prompt": "Alpha Beta Gamma"},
            {"id": "r3", "prompt": REF_A},
        ],
    )
    index = PromptIndex.read(refs).save(tmp_path / "index.json")
    recipe = Recipe(
        steps=[TinyRTN()],
        calib=FileSource(path=str(calib), samples=1, seqlen=16, contamination_index=str(index)),
    )
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
    cal = info["calibration"]
    assert cal["rows"] == 4 and cal["kept"] == 1 and cal["dropped"] == 3
    assert [m["id"] for m in cal["matches"]] == ["r1", "dup", "win"]
    assert cal["compared"].startswith("prompts only")
    assert any("3 dropped" in line and "prompts only" in line for line in lines)


def test_a_build_without_a_reference_says_the_filter_did_not_run(tmp_path, monkeypatch):
    monkeypatch.delenv("QDA_CONTAMINATION_REQUIRED", raising=False)
    monkeypatch.delenv("QDA_CONTAMINATION_INDEX", raising=False)
    m = tiny_qwen3(layers=1)
    m.save_pretrained(tmp_path / "m")
    rows = [
        torch.randint(0, 128, (16,), generator=torch.Generator().manual_seed(0)).tolist() for _ in range(2)
    ]
    calib = tmp_path / "c.jsonl"
    write_ids(calib, rows, save_tiny_tokenizer(tmp_path / "m"))
    recipe = Recipe(steps=[TinyRTN()], calib=FileSource(path=str(calib), samples=1, seqlen=16))
    build(
        str(tmp_path / "m"),
        recipe,
        format="w4a16-sym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.float32,
        exporter=writes(FakeWriter),
        log=lambda s: None,
    )
    info = json.loads((tmp_path / "out" / "build.json").read_text())
    assert info["calibration"] == {"filter": "not run (no reference index)"}


def test_a_token_id_row_is_screened_by_its_decoded_prompt(tmp_path, index):
    """A pre-tokenized prompt is decoded and compared, not skipped."""
    m = tiny_qwen3(layers=1)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    ids = [3] * 8
    calib = tmp_path / "c.jsonl"
    write_ids(calib, [ids, [4] * 8], sha)
    ref = PromptIndex.from_rows([{"id": "r", "prompt": "w3 " * 8}]).save(tmp_path / "i.json")
    src = FileSource.model_validate(
        {"path": str(calib), "samples": 1, "seqlen": 8, "contamination_index": str(ref)}
    )

    class Tok:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return "w3 " * len(ids) if ids and ids[0] == 3 else "an unrelated prompt"

    src.screen([{"input_ids": ids}, {"input_ids": [4] * 8}], Tok())
    assert [d["row"] for d in src.dropped] == [0]


def test_a_null_or_assistant_only_reference_row_carries_no_text():
    """JSON null, a missing prompt and an assistant-only chat are no prompt text, not the string "None"."""
    assert prompt_texts({"prompt": None}) == [""]
    assert prompt_texts({"messages": [{"role": "assistant", "content": "an answer"}]}) == []
    assert prompt_texts({"text": None}) == [""]


def test_clean_text_beside_copied_token_ids_still_drops(index):
    """A row carrying both a text field and the token ids it trains on is screened as the tokens are read."""

    class Tok:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return REF_C if ids == [7, 8, 9] else "an unrelated prompt"

    rows = [
        {"id": "trace", "prompt": "a clean unrelated instruction", "prompt_ids": [7, 8, 9]},
        {"id": "input", "text": "a clean unrelated instruction", "input_ids": [7, 8, 9]},
        {"id": "clean", "prompt": "a clean unrelated instruction", "prompt_ids": [1, 2]},
    ]
    kept, dropped, _ = filter_rows(rows, index, Tok())
    assert [r["id"] for r in kept] == ["clean"]
    assert [d["id"] for d in dropped] == ["trace", "input"]


def test_a_copied_task_beside_a_constraint_phrase_drops():
    """A phrase that looks boilerplate does not exempt a window only one reference carries."""
    task = (
        "Print the number of positive integers in the given array for which the value is the same as its "
        "index and then your response must include the count and a short justification of each step"
    )
    ref = PromptIndex.from_rows([{"prompt": task}])
    kept, dropped, exempted = filter_rows([{"id": "copied", "text": "my preamble " + task}], ref)
    assert kept == [] and dropped and exempted == []


ZIRCON_TASK = (
    "Calculate zircon radiometric formation age using uranium lead measurements the input contains n "
    "integers specifying ratios and their decay constants"
)


def test_a_task_recurring_across_two_references_still_drops():
    """A substantive task shared by two reference prompts is still reference content, never scaffolding."""
    ref = PromptIndex.from_rows(
        [
            {"id": "r1", "prompt": "Task one. " + ZIRCON_TASK},
            {"id": "r2", "prompt": "Task two. " + ZIRCON_TASK},
        ]
    )
    kept, dropped, exempted = filter_rows([{"id": "copied", "text": "my preamble " + ZIRCON_TASK}], ref)
    assert kept == [] and dropped and exempted == []


def test_a_task_containing_an_io_clause_is_not_exempt():
    """The whole window must be I/O format: substantive words beside an I/O clause keep the row a hit."""
    text = "Read the task below.\n\nInput\n\n" + ZIRCON_TASK
    ref = PromptIndex.from_rows([{"id": "r", "prompt": text}])
    kept, dropped, exempted = filter_rows([{"id": "copied", "text": "my preamble " + text}], ref)
    assert kept == [] and dropped and exempted == []


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": "x"}],
        [{"id": "x", "prompt": "   "}],
        [{"id": "x", "prompt": "!!! ... ???"}],  # normalized text with no alphanumeric character
        [17],
    ],
)
def test_a_derived_index_without_prompt_text_refuses(tmp_path, rows):
    """PromptIndex.read -> from_rows validates like from_dict: empty, id-only or malformed cannot start."""
    p = jsonl(tmp_path / "refs.jsonl", rows)
    with pytest.raises(Refused, match="contamination reference"):
        PromptIndex.read(p)


def test_scale_kd_screens_the_rows_it_reads_not_the_selfgen_prompts(monkeypatch, tmp_path, index):
    """The KD intake filters its own traces, not the selfgen source's unrelated prompt file."""
    from qda_quant.intake import tokenizer as tokenizer_module
    from qda_quant.steps.scale_kd import ScaleKD

    class Decoder:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return REF_C if ids == [7, 8, 9] else "an unrelated trace prompt"

    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda _d: Decoder())
    prompts = jsonl(tmp_path / "p.jsonl", [{"id": "p1", "prompt": "unrelated selfgen prompt"}])
    calib = SelfgenSource.model_validate(
        {"prompts": str(prompts), "n": 1, "contamination_index": str(index.save(tmp_path / "i.json"))}
    )
    traces = jsonl(
        tmp_path / "kd.jsonl",
        [{"prompt_ids": [7, 8, 9], "completion_ids": [1, 2]}, {"prompt_ids": [1], "completion_ids": [2]}],
    )
    step = ScaleKD.model_validate({"data": str(traces)})
    kept = step._screen(Recipe(steps=[], calib=calib), "unused", step.trace_rows())
    # the copied trace prompt drops; the selfgen prompt file is never screened or counted
    assert [r["prompt_ids"] for r in kept] == [[1]]
    assert calib.dropped[0]["id"] == "row 0"


def test_a_short_reference_prompt_starts_with_a_text_hash_and_no_windows(tmp_path):
    """A six-word reference has no 20-word window; its normalized text still covers the id/hash arm."""
    p = jsonl(tmp_path / "refs.jsonl", [{"id": "r", "prompt": "Name the capital city of France"}])
    index = PromptIndex.read(p)
    assert index.prompts == 1 and index.texts
    assert not (index.guard or index.regex)  # too short for a window, and that is valid
    rows = [
        {"id": "c", "prompt": "NAME the capital city of FRANCE"},
        {"id": "d", "prompt": "an unrelated calibration row"},
    ]
    kept, dropped, _ = filter_rows(rows, index)
    assert [r["id"] for r in kept] == ["d"]
    assert dropped[0]["id"] == "c" and "text" in dropped[0]["match"]


#: The shape of a written index; each case below overrides only the counts and digests it is about.
WRITTEN_INDEX = {
    "kind": "qda-calibration-contamination",
    "window": 20,
    "chars": 32,
    "guard_windows": [],
    "regex_windows": [],
}
ID_DIGEST = hashlib.sha256(b"r").hexdigest()
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize(
    ("counts", "refusal"),
    [
        # Ids only: valid-shaped and 64-hex, but it matches nothing except a copied id, so no filter can start.
        pytest.param({"prompts": 1, "ids": [ID_DIGEST], "texts": []}, "no prompt-text hashes", id="ids-only"),
        # Empty: it would report a successful filter while filtering nothing.
        pytest.param({"prompts": 0, "ids": [], "texts": []}, "not a positive integer", id="zero-rows"),
        # The digest of the empty text is not a prompt: the non-empty-text rule rejects it.
        pytest.param(
            {"prompts": 1, "ids": [ID_DIGEST], "texts": [EMPTY_DIGEST]}, "empty text", id="empty-text-digest"
        ),
    ],
)
def test_a_written_index_that_cannot_filter_refuses(tmp_path, counts, refusal):
    """Refused at load: an index that cannot drop a copied prompt may not be loaded and reported as a filter."""
    p = tmp_path / "index.json"
    p.write_text(json.dumps(WRITTEN_INDEX | counts))
    with pytest.raises(Refused, match=refusal):
        PromptIndex.load(p)


def test_a_reference_index_records_its_eval_and_holdout_sources(tmp_path):
    """The written index names both prompt sets, and a holdout prompt copy is a hit like any reference."""
    d = tmp_path / "ref"
    d.mkdir()
    prompts = jsonl(
        tmp_path / "p.jsonl",
        [
            {"id": "r1", "prompt": REF_A},
            {"id": "r2", "prompt": REF_C},
            {"id": "h1", "prompt": REF_B, "split": "holdout"},
        ],
    )
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    index = PromptIndex.read_reference(d)
    assert sorted((s.role, s.rows) for s in index.sources) == [("eval", 2), ("holdout", 1)]
    assert index.prompts == 3
    assert index.hit({"id": "x", "prompt": REF_B}) is not None


def _parent_record(ref: Path, corpus: Path, *, rows: int | None = None, sha: str | None = None) -> None:
    """Write the selection record a subset reference carries: the corpus it was drawn from."""
    import hashlib

    (ref / "selection.json").write_text(
        json.dumps(
            {
                "corpus": {
                    "path": str(corpus),
                    "rows": len(corpus.read_text().splitlines()) if rows is None else rows,
                    "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest() if sha is None else sha,
                }
            }
        )
    )


def _subset_ref(root: Path, name: str, row: dict, corpus: Path) -> Path:
    d = root / name
    d.mkdir()
    prompts = jsonl(d / "prompts.jsonl", [row])
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    _parent_record(d, corpus)
    return d


def _corpus(tmp_path: Path, rows: list[dict]) -> Path:
    corpus = tmp_path / "corpus" / "klv2-corpus-v2.jsonl"
    corpus.parent.mkdir()
    return jsonl(corpus, rows)


def test_a_subset_reference_is_indexed_from_its_parent_corpus_eval_split(tmp_path):
    """A 96-prompt eval file leaves the rest of the corpus unfiltered: the record's full eval split is indexed."""
    corpus = _corpus(
        tmp_path,
        [
            {"id": "r1", "prompt": REF_A, "split": "eval"},
            {"id": "r2", "prompt": REF_C, "split": "eval"},
            {"id": "h1", "prompt": REF_B, "split": "holdout"},
        ],
    )
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    index = PromptIndex.read_reference(d)
    assert [(s.role, s.rows, s.base) for s in index.sources] == [("eval", 2, "selection")]
    assert Path(index.sources[0].path) == corpus
    assert index.hit({"prompt": REF_C}) is not None  # a row the subset file never held
    assert index.hit({"prompt": REF_B}) is None  # the holdout split is the holdout reference's role


def test_a_subset_holdout_reference_is_indexed_from_its_parent_corpus_holdout_split(tmp_path):
    corpus = _corpus(
        tmp_path,
        [
            {"id": "r1", "prompt": REF_A, "split": "eval"},
            {"id": "h1", "prompt": REF_B, "split": "holdout"},
            {"id": "h2", "prompt": REF_C, "split": "holdout"},
        ],
    )
    d = _subset_ref(tmp_path, "holdout", {"id": "h1", "prompt": REF_B}, corpus)
    index = PromptIndex.read_holdout(d)
    assert [(s.role, s.rows, s.base) for s in index.sources] == [("holdout", 2, "selection")]
    assert index.hit({"prompt": REF_C}) is not None and index.hit({"prompt": REF_A}) is None


def test_a_subset_record_whose_corpus_changed_refuses(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_A}])
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    corpus.write_text(corpus.read_text() + json.dumps({"id": "r2", "prompt": REF_C}) + "\n")
    with pytest.raises(Refused, match="the corpus changed"):
        PromptIndex.read_reference(d)


def test_a_malformed_selection_record_refuses_rather_than_keeping_the_subset(tmp_path):
    """An unreadable record cannot be skipped: skipping it would index the 96-row subset as the whole."""
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": REF_C}])
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    (d / "selection.json").write_text('{"corpus": ')
    with pytest.raises(Refused, match="selection record .*selection.json"):
        PromptIndex.read_reference(d)


def test_a_selection_record_whose_corpus_is_not_an_object_refuses(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": REF_C}])
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    (d / "selection.json").write_text(json.dumps({"corpus": "klv2-corpus-v2.jsonl"}))
    with pytest.raises(Refused, match="its corpus is str"):
        PromptIndex.read_reference(d)


def test_a_subset_reference_whose_rows_carry_no_ids_refuses(tmp_path):
    """Without ids no row can be shown to be in the parent, so the subset may not be indexed as the whole."""
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": REF_C}])
    d = _subset_ref(tmp_path, "eval", {"prompt": REF_A}, corpus)
    with pytest.raises(Refused, match="carry no id"):
        PromptIndex.read_reference(d)


def test_a_parent_row_with_the_same_id_but_another_prompt_refuses(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_B}, {"id": "r2", "prompt": REF_C}])
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    with pytest.raises(Refused, match="different prompt text"):
        PromptIndex.read_reference(d)


def test_a_parent_split_that_would_drop_the_references_own_prompt_refuses(tmp_path):
    """The parent labels the reference's own eval row holdout: the index may not omit what the run scores."""
    corpus = _corpus(
        tmp_path,
        [{"id": "r1", "prompt": REF_A, "split": "holdout"}, {"id": "r2", "prompt": REF_C, "split": "eval"}],
    )
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    with pytest.raises(Refused, match="leave a prompt the run itself scores out of the index"):
        PromptIndex.read_reference(d)


def test_a_parent_corpus_that_cannot_be_read_names_the_manifest_record_and_path(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": REF_C}])
    d = _subset_ref(tmp_path, "eval", {"id": "r1", "prompt": REF_A}, corpus)
    corpus.write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(Refused) as refused:
        PromptIndex.read_reference(d)
    message = str(refused.value)
    assert str(d / "manifest.json") in message
    assert str(d / "selection.json") in message
    assert str(corpus) in message


def test_a_separate_holdout_reference_is_unioned_with_the_eval_set(tmp_path):
    """Union(eval, holdout): a row copied from either set drops, and repeated short refs stay deduplicated."""
    eval_ref = PromptIndex.read_reference(
        jsonl(tmp_path / "e.jsonl", [{"id": "r1", "prompt": REF_A}, {"id": "r2", "prompt": "short one"}])
    )
    holdout_ref = PromptIndex.read_holdout(jsonl(tmp_path / "h.jsonl", [{"id": "h1", "prompt": REF_C}]))
    union = PromptIndex.union([eval_ref, holdout_ref])
    assert union.prompts == 3 and len(union.sources) == 2
    kept, dropped, _ = filter_rows(
        [{"id": "a", "prompt": REF_A}, {"id": "b", "prompt": REF_C}, {"id": "c", "prompt": REF_B}],
        union,
    )
    assert [r["id"] for r in kept] == ["c"] and {d["id"] for d in dropped} == {"a", "b"}


def test_a_recorded_source_digest_that_does_not_match_refuses(tmp_path):
    p = jsonl(tmp_path / "p.jsonl", [{"id": "r1", "prompt": REF_A}])
    d = tmp_path / "ref"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(p), "prompts_sha256": "0" * 64}}))
    with pytest.raises(Refused, match="does not match the file"):
        PromptIndex.read_reference(d)


def test_a_written_index_round_trips_its_sources_and_checks_their_sum(tmp_path):
    index = PromptIndex.read_holdout(jsonl(tmp_path / "h.jsonl", [{"id": "h1", "prompt": REF_C}]))
    p = index.save(tmp_path / "index.json")
    assert PromptIndex.load(p).same_content(index)
    d = json.loads(p.read_text())
    d["prompts"] = 5  # the union array and its source rows disagree
    p.write_text(json.dumps(d))
    with pytest.raises(Refused, match="sum of its source rows"):
        PromptIndex.load(p)
    d["prompts"] = 1
    d["sources"][0]["role"] = "somewhere"
    p.write_text(json.dumps(d))
    with pytest.raises(Refused, match="source role"):
        PromptIndex.load(p)


@pytest.mark.parametrize(("prompt", "clean"), NON_LATIN)
def test_a_copied_non_latin_prompt_matches_by_normalized_text(prompt, clean):
    """Whitespace and case only: the prompt is too short for a word window and not without-space, so text hits."""
    index = PromptIndex.from_rows([{"prompt": prompt}])
    assert index.hit({"prompt": "\n" + prompt.upper() + "  "}) == {"text": True}


@pytest.mark.parametrize(("prompt", "clean"), NON_LATIN)
def test_an_unrelated_non_latin_row_is_kept(prompt, clean):
    """ASCII-regex normalization is empty for these scripts; if normalization emptied the text, this row would hit."""
    index = PromptIndex.from_rows([{"prompt": prompt}])
    assert not index.regex
    kept, dropped, exempted = filter_rows([{"id": "clean", "prompt": clean}], index)
    assert [r["id"] for r in kept] == ["clean"] and dropped == [] and exempted == []


def test_a_thai_copy_at_the_character_window_length_drops():
    """Thai is written without spaces: a 32-character copy is found by a character window, not a word one."""
    index = PromptIndex.from_rows([{"prompt": THAI}])
    row = {"id": "copy", "prompt": THAI_PADDING + " " + THAI[:32]}
    hit = index.hit(row)
    assert hit is not None and hit["guard"] and all(v is None for v in hit["shared"].values())
    kept, dropped, exempted = filter_rows([row], index)
    assert kept == [] and exempted == [] and dropped[0]["match"] == ["guard"]


def test_a_thai_copy_shorter_than_the_character_window_is_not_found():
    """The character arm is the only one for Thai, and it starts at 32 characters: the documented limit."""
    index = PromptIndex.from_rows([{"prompt": THAI}])
    assert index.hit({"prompt": THAI_PADDING + " " + THAI[:31]}) is None


@pytest.mark.parametrize("passage", SPACED)
def test_a_copied_spaced_script_passage_is_found_by_the_word_window(passage):
    """Spaced scripts get word windows only; the ASCII regex arm is empty, so the guard window must catch it."""
    index = PromptIndex.from_rows([{"prompt": passage}])
    assert not index.regex
    words = passage.split()
    copy = {"id": "copy", "prompt": "Prepare the answer. " + " ".join(words[:20]) + " Now finish."}
    kept, dropped, exempted = filter_rows([copy], index)
    assert kept == [] and exempted == [] and dropped[0]["match"] == ["guard"]
    short = {"id": "short", "prompt": "Prepare the answer. " + " ".join(words[:19]) + " Now finish."}
    kept, dropped, exempted = filter_rows([short], index)
    assert [r["id"] for r in kept] == ["short"] and dropped == [] and exempted == []


def test_a_thai_answer_beside_an_unrelated_prompt_is_not_screened():
    """Only prompts are compared: an assistant body that carries the Thai passage keeps the row."""
    index = PromptIndex.from_rows([{"prompt": THAI}])
    row = {
        "messages": [
            {"role": "user", "content": "Explain what this passage means"},
            {"role": "assistant", "content": THAI},
        ]
    }
    assert index.hit(row) is None
