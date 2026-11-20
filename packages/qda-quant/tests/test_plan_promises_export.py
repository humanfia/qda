"""A green plan is a promise: a format plans green exactly when the exporter can write it.

`qda plan` and the build ask the same exporter check (`output.exporters.check_exporter`). A format the exporter
cannot write must be refused there, not after a whole calibration run. For every registered format (a format
registered later included) the test measures:

* green: `compile_plan` and `check_exporter(CompressedTensors())` pass, as `qda plan` asks;
* written: a CPU build with that check switched off finishes and writes `build.json`, so the measure does not
  depend on the check it tests.

and asserts green == written for each, which over the registry is the set equality {green} == {written}: a
format planned green but unwritable fails, and so does one refused but writable.

compressed_tensors x w8a8i used to plan green and be refused only when `ct_config` wrote the config, at the end
of the build; it is now refused at plan, naming what can be built.
"""

import json
import sys

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.engine.planning import compile_plan
from qda_quant.format import FORMATS, Format
from qda_quant.output.exporters import CompressedTensors, check_exporter
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.target import Target

from .engine_util import save_tiny_tokenizer, tiny_qwen3


def _pairs() -> list:
    return [pytest.param(name, id=f"compressed_tensors-{name}") for name, _ in FORMATS.items()]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("plan-promise")
    tiny_qwen3(layers=1, hidden=128, inter=256).save_pretrained(d / "m")
    save_tiny_tokenizer(d / "m")
    rows = [{"text": " ".join(f"w{(i * 7 + k) % 90 + 3}" for k in range(40))} for i in range(4)]
    (d / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


@pytest.fixture(autouse=True)
def _no_reference_index(monkeypatch):
    monkeypatch.delenv("QDA_CONTAMINATION_REQUIRED", raising=False)
    monkeypatch.delenv("QDA_CONTAMINATION_INDEX", raising=False)


def _recipe(d):
    return Recipe(steps=[REGISTRY["rtn"]()], calib=FileSource(path=str(d / "c.jsonl"), samples=2, seqlen=16))


def _green(d, name: str) -> bool:
    r = _recipe(d)
    try:
        plan = compile_plan(str(d / "m"), r, Format.parse(name), Target(), [torch.device("cpu")])
        check_exporter(CompressedTensors(), plan, r)
    except Refused:
        return False
    return True


def _written(d, name: str, monkeypatch, out_root) -> bool:
    engine_build = sys.modules["qda_quant.engine.build"]  # the package re-exports the function `build`
    monkeypatch.setattr(engine_build, "check_exporter", lambda *a, **k: None)
    out = out_root / f"out-{name}"  # a fresh dir per call: a build refuses to write over an existing one
    try:
        build(str(d / "m"), _recipe(d), format=name, out_dir=out, devices=["cpu"], dtype=torch.float32,
              log=lambda s: None)  # fmt: skip
    except Refused:
        return False
    return (out / "build.json").is_file()


@pytest.mark.parametrize("name", _pairs())
def test_a_format_plans_green_exactly_when_the_exporter_writes_it(tiny, name, monkeypatch, tmp_path):
    green, written = _green(tiny, name), _written(tiny, name, monkeypatch, tmp_path)
    assert green == written, (
        f"{name}: planned {'green' if green else 'refused'}, {'written' if written else 'not written'}"
    )


def test_the_registry_has_a_written_format(tiny, monkeypatch, tmp_path):
    """The equality must not hold vacuously: at least one registered format is green and written."""
    assert any(_green(tiny, n) and _written(tiny, n, monkeypatch, tmp_path) for n, _ in FORMATS.items())


def test_w8a8i_is_refused_at_plan_naming_what_can_be_built(tiny):
    r = _recipe(tiny)
    plan = compile_plan(str(tiny / "m"), r, Format.parse("w8a8i"), Target(), [torch.device("cpu")])
    with pytest.raises(Refused, match=r"w8a8i-sym-pc: INT8 activations .* cannot be written yet") as e:
        check_exporter(CompressedTensors(), plan, r)
    can = [n for n, f in FORMATS.items() if f.act != "int8_token"]
    assert can and all(n in str(e.value) for n in can) and "w8a8i" not in str(e.value).split("writes")[1]
