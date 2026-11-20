"""``chunks: true``: every calibration row cut into consecutive ``seqlen`` pieces (old qda's rule); off by default."""

import hashlib
import json

import pytest
import torch
from qda_quant import Recipe
from qda_quant.calib.sources import FileSource
from qda_quant.engine import build
from qda_quant.engine.calib import load_samples, pieces
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ

from ..engine_util import save_tiny_tokenizer, tiny_qwen3


class WordTok:
    """Token id = the word's number (``w7`` -> 7)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(w[1:]) for w in text.split()]}


def text_file(path, lengths):
    """One text row per length, words numbered on from row to row, so every piece is recognizable."""
    rows, k = [], 0
    for n in lengths:
        rows.append({"text": " ".join(f"w{k + i}" for i in range(n))})
        k += n
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_pieces_keep_the_tail_and_the_order():
    assert pieces([list(range(10)), [10, 11]], 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11]]
    assert pieces([list(range(10)), [10, 11]], 4, n=2) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]  # stops at a row end


def test_long_rows_give_several_samples(tmp_path):
    path = text_file(tmp_path / "c.jsonl", [10, 3, 9])
    on = load_samples(FileSource(path=str(path), samples=7, seqlen=4, chunks=True), tmp_path, WordTok())
    assert [s.ids.tolist() for s in on] == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21],
    ]  # fmt: skip
    # off (the default): one truncated sample per long row, short rows concatenated
    off = load_samples(FileSource(path=str(path), samples=2, seqlen=4), tmp_path, WordTok())
    assert [s.ids.tolist() for s in off] == [[0, 1, 2, 3], [13, 14, 15, 16]]
    with pytest.raises(Refused, match="7 samples of 4, recipe asks for 8"):
        load_samples(FileSource(path=str(path), samples=8, seqlen=4, chunks=True), tmp_path, WordTok())


def test_a_short_calibration_names_the_setting_to_change(tmp_path):
    """11-14 first-user test: the default recipe's 125 samples were 122 for another tokenizer, and the refusal did
    not say which setting to change. It names the rows knob when the source has one, and the samples count."""
    from qda_quant.calib.sources import HFSource
    from qda_quant.engine.calib import _short_hint

    path = text_file(tmp_path / "c.jsonl", [10, 3, 9])
    with pytest.raises(
        Refused, match=r"recipe asks for 8; give it more rows or lower calib\.file\.samples to 7$"
    ):
        load_samples(FileSource(path=str(path), samples=8, seqlen=4, chunks=True), tmp_path, WordTok())
    hf = HFSource(dataset="d", revision="0" * 40, rows=512, samples=125)
    assert _short_hint(hf, 122) == "; raise calib.hf.rows (now 512) or lower calib.hf.samples to 122"
    assert _short_hint(hf, 0) == "; raise calib.hf.rows (now 512) or give it rows with text"


def test_chunks_refuse_traces(tmp_path):
    from qda_quant.engine.calib import fingerprint

    (tmp_path / "tokenizer.json").write_text("{}")
    rows = [{"tokenizer": fingerprint(tmp_path)}, {"prompt_ids": [1, 2], "completion_ids": [3, 4, 5, 6, 7]}]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(Refused, match="does not apply to traces"):
        load_samples(FileSource(path=str(path), samples=1, seqlen=4, chunks=True), tmp_path, WordTok())


def test_chunks_off_builds_the_same_bytes_as_a_config_without_the_key(tmp_path):
    """Existing configs do not move: the default and an explicit ``chunks: false`` write the same checkpoint."""
    tiny_qwen3(layers=2, hidden=128, inter=256).save_pretrained(tmp_path / "m")
    save_tiny_tokenizer(tmp_path / "m")
    text_file(tmp_path / "c.jsonl", [40, 7, 30, 50])
    out = {}
    for name, extra in (("default", {}), ("off", {"chunks": False})):
        r = Recipe.from_dict({"steps": [{"gptq": {}}], "calib": {"file": {"path": str(tmp_path / "c.jsonl"),
                              "samples": 3, "seqlen": 16, **extra}}})  # fmt: skip
        assert isinstance(r.steps[0], GPTQ) and r.calib.chunks is False
        ckpt = build(str(tmp_path / "m"), r, format="w4a16-sym-g128", out_dir=tmp_path / name, devices=["cpu"],
                     dtype=torch.float32, log=lambda s: None)  # fmt: skip
        out[name] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ckpt.glob("*.safetensors")}
    assert out["default"] and out["default"] == out["off"]


def test_a_short_part_of_a_mix_or_slice_names_the_enclosing_count(tmp_path):
    """A part gets its share of the enclosing samples; the part's own count is replaced, so the hint names the
    enclosing source (xrev3 11-14 note (b): `calib.file.samples` does not exist at the top level)."""
    from qda_quant.calib.sources import MixSource, SliceSource

    pool = tmp_path / "pool"
    pool.mkdir()
    text_file(pool / "a.jsonl", [4])
    text_file(pool / "b.jsonl", [4])

    sliced = SliceSource(pool=str(pool), names={"a": 1, "b": 1}, samples=4, seqlen=4)
    with pytest.raises(Refused) as caught:
        load_samples(sliced, tmp_path, WordTok())
    said = str(caught.value)
    assert "lower calib.slice.samples (now 4)" in said and "calib.file.samples" not in said

    mixed = MixSource(
        parts=[
            {"file": {"path": str(pool / "a.jsonl"), "seqlen": 4}, "weight": 1},
            {"file": {"path": str(pool / "b.jsonl"), "seqlen": 4}, "weight": 1},
        ],
        samples=4,
    )
    with pytest.raises(Refused) as caught:
        load_samples(mixed, tmp_path, WordTok())
    said = str(caught.value)
    assert "lower calib.mix.samples (now 4)" in said and "calib.file.samples" not in said
