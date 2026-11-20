"""``QDA_OUTPUT_ROOTS``: a relative output goes under the first root, one outside every root is refused."""

import pytest
from qda_quant.engine.planning import check_output
from qda_quant.output.roots import ENV, output_path
from qda_quant.step import Refused


def test_unset_keeps_the_named_path(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert output_path("out") == tmp_path / "out"


def test_relative_goes_under_the_first_root_and_outside_is_refused(tmp_path, monkeypatch):
    a, b, elsewhere = tmp_path / "a", tmp_path / "b", tmp_path / "shared"
    for d in (a, b, elsewhere):
        d.mkdir()
    monkeypatch.setenv(ENV, f"{a}:{b}")
    monkeypatch.chdir(elsewhere)
    assert output_path("ckpt") == a / "ckpt"
    assert output_path(b / "x") == b / "x"
    assert check_output("ckpt") == a / "ckpt"  # the build's own check applies the same rule
    for bad in (elsewhere / "ckpt", a / ".." / "shared" / "ckpt"):
        with pytest.raises(Refused, match=ENV):
            output_path(bad)
        with pytest.raises(Refused, match=ENV):
            check_output(bad)
