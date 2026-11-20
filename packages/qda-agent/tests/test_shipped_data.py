"""Data sets handed to the agent are checked against the model's chat template before the clock starts."""

import json
from pathlib import Path

import pytest
import testkit
import tests_support  # noqa: F401  (registers the tests' contamination checks)
from hmz.sdk import fakes
from qda_agent.config import RunConfig
from qda_agent.flow import CalibTemplateRefused, qda_search
from test_flow import write_config

R1_TEXT = "<｜User｜>w1 w2<｜Assistant｜>w3 w4<｜end▁of▁sentence｜>"
CHATML_TEXT = "<|im_start|>user\nw1 w2<|im_end|>\n<|im_start|>assistant\nw3 w4<|im_end|>"


def r1_model(path: Path):
    """A local model directory whose tokenizer has R1's chat template (and spells ChatML's markers as text)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.WordLevel({f"w{i}": i for i in range(8)} | {"[UNK]": 8}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.add_tokens(["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<|im_start|>", "<|im_end|>"])
    fast.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
        "{% else %}<｜Assistant｜>{{ m['content'] }}<｜end▁of▁sentence｜>{% endif %}{% endfor %}"
    )
    fast.save_pretrained(path)
    return fast


def jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def config(tmp_path: Path, sets: dict[str, Path], **extra) -> Path:
    model = tmp_path / "model"
    return write_config(
        tmp_path,
        model=str(model),
        toolset={
            "name": "qda",
            "components": [{"name": "data", "sets": {k: str(v) for k, v in sets.items()}}],
        },
        **extra,
    )


@pytest.mark.parametrize("family", ["own", "foreign"])
@pytest.mark.parametrize("shape", ["text", "ids"])
async def test_a_rendered_pool_refuses_the_launch_whatever_its_family(tmp_path, shape, family):
    """R1 or ChatML markup, as text or as token ids, in a pool shipped to an R1 model: the search does not start."""
    fast = r1_model(tmp_path / "model")
    text = R1_TEXT if family == "own" else CHATML_TEXT
    row = {"text": text} if shape == "text" else {"input_ids": fast.encode(text)}
    path = jsonl(tmp_path / "data" / "corpus.jsonl", [{"text": "w1 w2"}, row])
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    params = {"config": config(tmp_path, {"corpus": path}, seconds=0.2)}
    match = rf"data set corpus {path}: of 2 rows, 1 rows hold pre-rendered chat markup \(first at lines 2;"
    with pytest.raises(CalibTemplateRefused, match=match):
        await fakes.run_fake(qda_search, agents={"worker": worker}, params=params)
    assert worker.prompts == []


async def test_messages_and_plain_text_and_token_rows_start_the_search(tmp_path, capsys):
    fast = r1_model(tmp_path / "model")
    ids = fast.encode("w1 w2 w3 w4")
    rows = [
        {"messages": [{"role": "user", "content": "w1"}, {"role": "assistant", "content": "w2"}]},
        {"text": "w1 w2"},
        {"prompt": "w1", "completion": "w2"},
        {"prompt_ids": ids[:2], "completion_ids": ids[2:]},
        {"input_ids": ids},
    ]
    path = jsonl(tmp_path / "data" / "corpus.jsonl", rows)
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.01))
    params = {"config": config(tmp_path, {"corpus": path}, seconds=0.2)}
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params=params)
    assert out["ended"] == "deadline", out
    assert "data sets, pre-rendered chat markup: 1 JSONL files, 5 rows checked" in capsys.readouterr().err


def test_a_directory_set_names_its_one_bad_file_and_never_reads_other_files(tmp_path, capsys):
    r1_model(tmp_path / "model")
    root = tmp_path / "data" / "pool"
    for name in ("a.jsonl", "sub/c.jsonl"):
        jsonl(root / name, [{"text": "w1 w2"}])
    bad = jsonl(root / "b.jsonl", [{"text": "w1"}, {"text": R1_TEXT}])
    notes = jsonl(root / "notes.json", [{"text": CHATML_TEXT}])  # not JSONL: listed, never read
    notes.chmod(0)
    plain = jsonl(tmp_path / "data" / "plain.jsonl", [{"text": "w1"}])
    cfg = RunConfig.load(config(tmp_path, {"pool": root, "plain": plain}))
    with pytest.raises(CalibTemplateRefused, match=rf"data set pool {bad}: of 2 rows, 1 rows .*'<｜User｜>'"):
        cfg.toolset.check_shipped_data(cfg.model)
    bad.write_text(json.dumps({"text": "w2"}) + "\n")
    cfg.toolset.check_shipped_data(cfg.model)
    err = capsys.readouterr().err
    assert f"4 JSONL files, 4 rows checked; 1 files not checked (not JSONL): {notes}" in err, err


def test_plain_data_sets_need_no_tokenizer(tmp_path):
    """The config's model is not local: a set without token-id rows never loads it."""
    path = jsonl(tmp_path / "data" / "corpus.jsonl", [{"text": "w1"}, {"messages": []}])
    write = write_config(
        tmp_path, toolset={"name": "qda", "components": [{"name": "data", "sets": {"corpus": str(path)}}]}
    )
    cfg = RunConfig.load(write)
    cfg.toolset.check_shipped_data(cfg.model)


def _check(tmp_path: Path, sets: dict[str, Path]) -> None:
    cfg = RunConfig.load(config(tmp_path, sets))
    cfg.toolset.check_shipped_data(cfg.model)


def test_a_symlinked_directory_inside_a_set_is_followed(tmp_path):
    r1_model(tmp_path / "model")
    target = jsonl(tmp_path / "elsewhere" / "bad.jsonl", [{"text": CHATML_TEXT}])
    pool = tmp_path / "data" / "pool"
    jsonl(pool / "good.jsonl", [{"text": "w1"}])
    (pool / "linked").symlink_to(target.parent, target_is_directory=True)
    (pool / "loop").symlink_to(pool, target_is_directory=True)  # a cycle is walked once
    with pytest.raises(CalibTemplateRefused, match=r"linked/bad\.jsonl: of 1 rows, 1 rows hold pre-rendered"):
        _check(tmp_path, {"pool": pool})


@pytest.mark.parametrize("ids", [[999999], [-1], None, [1, None], "w1", [True]])
def test_ids_outside_the_vocabulary_or_not_integers_are_refused(tmp_path, ids):
    """A fast decode drops an unknown id silently, so every id is checked before decoding; nulls refuse cleanly."""
    r1_model(tmp_path / "model")
    path = jsonl(tmp_path / "data" / "pool.jsonl", [{"text": "w1"}, {"input_ids": ids}])
    with pytest.raises(
        CalibTemplateRefused, match=r"1 token-id rows are not ids of this model's tokenizer .*lines 2\)"
    ):
        _check(tmp_path, {"pool": path})


def test_a_file_reached_by_two_sets_or_links_is_read_once(tmp_path, monkeypatch, capsys):
    r1_model(tmp_path / "model")
    path = jsonl(tmp_path / "data" / "pool" / "plain.jsonl", [{"text": "w1"}])
    (tmp_path / "data" / "alias.jsonl").symlink_to(path)
    (tmp_path / "data" / "hard.jsonl").hardlink_to(path)
    opened = []
    original = Path.open

    def counted(self, *args, **kwargs):
        if self.is_file() and self.samefile(path):
            opened.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    sets = {"file": path, "directory": path.parent, "alias": tmp_path / "data" / "alias.jsonl"}
    _check(tmp_path, sets | {"hard": tmp_path / "data" / "hard.jsonl"})
    assert len(opened) == 1, opened
    assert "1 JSONL files, 1 rows checked" in capsys.readouterr().err


def test_ordinary_prose_with_a_chat_marker_still_refuses(tmp_path):
    """Fail closed: a pool row quoting a marker is refused, even in prose."""
    r1_model(tmp_path / "model")
    path = jsonl(tmp_path / "data" / "pool.jsonl", [{"text": "print('[INST]')"}])
    with pytest.raises(CalibTemplateRefused, match=r"'\[INST\]' of mistral"):
        _check(tmp_path, {"pool": path})
