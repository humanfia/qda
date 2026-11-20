"""docs/verify.md and docs/add-scorer.md: doctests pass, configs validate, commands, flags and plugin group exist."""

import doctest
import re
import shlex
import sys
from pathlib import Path

import pytest
from qda_kl import cli
from qda_kl.config import ScoreConfig

DOCS = Path(__file__).resolve().parents[3] / "docs"
PAGE = DOCS / "verify.md"
SCORER_PAGE = DOCS / "add-scorer.md"


def blocks(lang: str, page: Path = PAGE) -> list[str]:
    """The bodies of a page's fenced ``lang`` blocks."""
    return re.findall(rf"^```{lang}\n(.*?)^```", page.read_text(), re.S | re.M)


def test_doctests_pass():
    failed, tried = doctest.testfile(
        str(SCORER_PAGE), module_relative=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert tried > 0 and failed == 0
    assert "vllm" not in sys.modules


@pytest.mark.parametrize("text", blocks("yaml"))
def test_yaml_blocks_validate(text):
    """A block naming ``checkpoints:`` is a `qda eval` config, one naming ``original:`` is a `qda-kl gencheck`
    config; any other is a `qda-kl score` config."""
    from qda_kl.eval.config import EvalConfig
    from qda_kl.gencheck import GencheckConfig

    if re.search(r"^checkpoints:", text, re.M):
        EvalConfig.from_yaml(text)
    elif re.search(r"^original:", text, re.M):
        GencheckConfig.from_yaml(text)
    else:
        ScoreConfig.from_yaml(text)


def test_shell_commands_exist(capsys):
    lines = [shlex.split(x) for b in blocks("bash") for x in b.splitlines() if x.startswith("qda-kl ")]
    assert {"ref", "score"} <= {argv[1] for argv in lines}
    for argv in lines:
        with pytest.raises(SystemExit):
            cli.main([argv[1], "--help"])
        usage = capsys.readouterr().out
        for flag in (a.split("=")[0] for a in argv if a.startswith("--")):
            assert flag in usage, f"{flag} is not an option of qda-kl {argv[1]}"


def test_scorer_guide_registers_through_the_hook():
    """The guide links the one hook page and registers its scorer in `SCORERS` under the class's `name`."""
    text = SCORER_PAGE.read_text()
    assert "extend.md#the-registration-hook" in text
    assert 'SCORERS.register("constant", "my_scorers:Constant")' in text and 'name = "constant"' in text


def test_scorer_guide_config_scores_with_its_reference(tmp_path, monkeypatch):
    """The guide's config runs its scorer as written once its ``reference`` path exists, and is refused before."""
    import json

    import yaml
    from pydantic import BaseModel
    from qda_kl import scorers
    from qda_kl.scorers import KLScorer, Outcome
    from qda_quant.registry import Registry

    class Result(BaseModel):
        kl: float

    class Constant(KLScorer):
        name = "constant"
        value: float = 0.5

        def run(self, checkpoint, reference, *, subset=None):
            r = Result(kl=self.value)
            return Outcome(readings=[r], result=r)

    monkeypatch.setattr(scorers, "SCORERS", Registry("scorers"))
    scorers.SCORERS.register("constant", Constant)
    monkeypatch.chdir(tmp_path)
    ((text,),) = [blocks("yaml", SCORER_PAGE)]
    cfg = yaml.safe_load(text)
    assert set(cfg) == {"checkpoint", "reference", "out", "scorer"}
    (tmp_path / "score.yaml").write_text(text)
    assert cli.main(["score", "score.yaml"]) == 2
    (tmp_path / cfg["reference"]).mkdir()
    assert cli.main(["score", "score.yaml"]) == 0
    assert json.loads((tmp_path / cfg["out"] / "score.json").read_text()) == {"kl": 0.25}
