"""Calibration loader on real calibration files and tokenizers.

Needs ``--calib``, ``--calib-ids`` and ``--model-dir qwen3-8b=PATH`` (plus ``--model-dir qwen3.8-27b=PATH`` for
the fingerprint check); each test skips when an option it needs is not given (see conftest.py).
"""

from __future__ import annotations

import json

import pytest
from qda_quant.calib.sources import FileSource
from qda_quant.engine.calib import fingerprint, load_ids
from qda_quant.step import Refused

from ..engine_util import write_ids


def test_messages_reproduce_calib128_on_q8b(calib_file, calib_ids_file, real_model):
    q8b = real_model("qwen3-8b")
    got = load_ids(FileSource(path=str(calib_file), samples=6, seqlen=2048), q8b)
    ref = [json.loads(x)["input_ids"] for x in calib_ids_file.read_text().splitlines()[:6]]
    assert [r.tolist() for r in got] == ref


def test_qwen3_ids_refused_for_qwen38(calib_ids_file, real_model, tmp_path):
    """Qwen3.8's vocab is larger, so a vocab check alone would pass Qwen3 ids; the fingerprint refuses them."""
    q8b, q38 = real_model("qwen3-8b"), real_model("qwen3.8-27b")
    rows = [json.loads(x)["input_ids"] for x in calib_ids_file.read_text().splitlines()[:2]]
    write_ids(tmp_path / "q3.jsonl", rows, fingerprint(q8b))
    c = FileSource(path=str(tmp_path / "q3.jsonl"), samples=2, seqlen=2048)
    assert len(load_ids(c, q8b)) == 2
    with pytest.raises(Refused, match="token ids made by tokenizer"):
        load_ids(c, q38)
