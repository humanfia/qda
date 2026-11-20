"""``qda-kl score CONFIG.yaml``: one config, scorers by key, resolved.yaml repeats the run."""

import json
import re

import pytest
import yaml
from qda_kl import cli, score, scorers
from qda_kl.config import ScoreConfig, check_checkpoint
from qda_kl.scorers import SCORERS, TopKScorer
from qda_quant import registry
from qda_quant.registry import Registry
from qda_quant.step import Refused
from test_cli import fake_result, scorer, sequence_scorer


def config(tmp_path, **scorer_cfg):
    out = {
        "checkpoint": str(tmp_path / "ckpt"),
        "reference": str(tmp_path / "ref"),
        "out": str(tmp_path / "run"),
    }
    if scorer_cfg:
        out["scorer"] = {"topk": scorer_cfg}
    (tmp_path / "ref").mkdir(exist_ok=True)
    p = tmp_path / "score.yaml"
    p.write_text(yaml.safe_dump(out))
    return p


def test_a_model_is_pinned_by_its_directory_or_the_revision_it_names():
    """A local model directory is signed; a hub id carries its revision; a bare hub id pins nothing."""
    from qda_kl.config import source_identity

    assert source_identity("org/name@abc123") == {
        "snapshot": None,
        "revision": "abc123",
        "signature": None,
    }
    assert source_identity("org/name") == {"snapshot": None, "revision": None, "signature": None}
    assert source_identity("/no/such/directory") == {
        "snapshot": None,
        "revision": None,
        "signature": None,
    }


def test_topk_is_the_default_and_registered_like_a_plugin():
    assert "topk" in SCORERS and SCORERS["topk"] is TopKScorer
    assert SCORERS.provenance("topk")[0] == "qda-kl"
    cfg = ScoreConfig(checkpoint="c", reference="r", out="o")
    assert cfg.to_dict()["scorer"] == {"topk": TopKScorer().model_dump(mode="json")}


def test_success_writes_the_result_and_resolved_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))
    assert cli.main(["score", str(config(tmp_path))]) == 0
    run = tmp_path / "run"
    assert json.loads((run / "score.json").read_text())["kl"] == 0.02
    resolved = yaml.safe_load((run / "resolved.yaml").read_text())
    assert resolved["scorer"]["topk"]["repeat"] == 1 and resolved["scorer"]["topk"]["check"] is True
    prov = resolved["provenance"]
    assert (
        set(prov)
        == {
            "qda_commit",
            "source_snapshot",
            "source_revision",
            "source_signature",
            "cards",
            "not_recorded",
            "plugins",
            "sources",
        }
        and prov["qda_commit"]
    )
    # Both models the run read are named; a path that is not a model directory is recorded, not signed.
    assert prov["sources"] == {
        "checkpoint": {"snapshot": None, "revision": None, "signature": None},
        "reference": {"snapshot": None, "revision": None, "signature": None},
    }
    assert (
        prov["plugins"]["topk"]["group"] == "qda_kl.scorers"
        and prov["plugins"]["topk"]["package"] == "qda-kl"
    )


def test_a_rerun_from_resolved_yaml_is_the_same_run(tmp_path, monkeypatch):
    seen = []

    class Recording:
        def __init__(self, ckpt, ref, **kw):
            seen.append((ckpt, str(ref), kw["settings"].model_dump(), kw["allow_vllm_mismatch"], kw["tp"]))

        def score(self, **kw):
            seen.append(kw)
            return fake_result()

    monkeypatch.setattr(score, "Scorer", Recording)
    p = config(tmp_path, repeat=1, group=2, tp=2, serve_env={"A": "1"}, expect_kernels={"dense": "K"})
    assert cli.main(["score", str(p)]) == 0
    assert seen[0][-1] == 2  # the scorer's tp reaches the engine
    assert seen[0][2]["nccl_nvls"] is False  # an omitted key is resolved, reaches the engine and is recorded
    assert (
        yaml.safe_load((tmp_path / "run" / "resolved.yaml").read_text())["scorer"]["topk"]["nccl_nvls"]
        is False
    )
    first, seen[:] = list(seen), []
    run2 = tmp_path / "run2"
    assert cli.main(["score", str(tmp_path / "run" / "resolved.yaml"), "--set", f"out={run2}"]) == 0
    assert seen == first
    assert (run2 / "score.json").read_text() == (tmp_path / "run" / "score.json").read_text()
    r1, r2 = (yaml.safe_load((d / "resolved.yaml").read_text()) for d in (tmp_path / "run", run2))
    assert {**r1, "out": None} == {**r2, "out": None}


def test_automatic_cache_workspace_is_recorded_and_replayed(tmp_path, monkeypatch):
    """The engine's selected cache must reach resolved.yaml; recording only the input env loses it."""
    from qda_kl import serve

    cache = str(tmp_path / "local")
    monkeypatch.setattr(serve, "cache_env", lambda env: {**env, "FLASHINFER_WORKSPACE_BASE": cache})
    seen = []

    class Recording:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs["settings"].env)

        def score(self, **kwargs):
            return fake_result()

    monkeypatch.setattr(score, "Scorer", Recording)
    assert cli.main(["score", str(config(tmp_path))]) == 0
    resolved = tmp_path / "run" / "resolved.yaml"
    assert yaml.safe_load(resolved.read_text())["scorer"]["topk"]["serve_env"] == seen[0]
    assert cli.main(["score", str(resolved), "--set", f"out={tmp_path / 'again'}"]) == 0
    off = {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
    assert seen == [{"NCCL_NVLS_ENABLE": "0", **off, "FLASHINFER_WORKSPACE_BASE": cache}] * 2


def test_overrides_reach_the_scorer(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result()] * 3))
    assert cli.main(["score", str(config(tmp_path)), "--set", "scorer.topk.repeat=3"]) == 0
    resolved = yaml.safe_load((tmp_path / "run" / "resolved.yaml").read_text())
    assert resolved["scorer"]["topk"]["repeat"] == 3
    assert json.loads((tmp_path / "run" / "score.json").read_text())["repeats"]["kl"] == [0.02] * 3


def test_refusal_leaves_no_result_and_no_resolved(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "score.json").write_text("old")
    (run / "resolved.yaml").write_text("old")
    monkeypatch.setattr(score, "Scorer", scorer(error=Refused("no answer tokens scored")))
    assert cli.main(["score", str(config(tmp_path))]) == 2
    assert not (run / "score.json").exists() and not (run / "resolved.yaml").exists()


def test_an_engine_crash_leaves_no_result_and_no_resolved(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "score.json").write_text("old")
    (run / "resolved.yaml").write_text("old")
    monkeypatch.setattr(score, "Scorer", scorer(error=RuntimeError("out of CUDA memory")))
    assert cli.main(["score", str(config(tmp_path))]) == 1
    assert not (run / "score.json").exists() and not (run / "resolved.yaml").exists()


def test_nondeterminism_is_refused_through_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result(), fake_result(kl=0.03)]))
    assert cli.main(["score", str(config(tmp_path, repeat=2))]) == 2
    assert not (tmp_path / "run" / "score.json").exists()
    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result(), fake_result(kl=0.03)]))
    assert cli.main(["score", str(config(tmp_path, repeat=2, allow_nondeterministic=True))]) == 0


def test_missing_reference_is_refused_through_the_config(tmp_path, capsys):
    (tmp_path / "ckpt").mkdir()  # the checkpoint is there: this run is about the reference
    assert cli.main(["score", str(config(tmp_path))]) == 2
    assert "not a qda-kl reference" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_a_missing_local_reference_is_refused_at_config_load(tmp_path, capsys, monkeypatch):
    """Any scorer is handed only a reference path that exists; a missing one is two lines naming it."""
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))
    p = config(tmp_path)
    (tmp_path / "ref").rmdir()
    assert cli.main(["score", str(p)]) == 2
    error, fix = capsys.readouterr().err.splitlines()
    assert error == f"error: {p}: reference: no reference at {tmp_path / 'ref'}"
    assert fix.startswith("fix:") and "`qda-kl ref --model ORIGINAL --prompts PROMPTS --out " in fix
    assert not (tmp_path / "run").exists()
    (tmp_path / "ref").mkdir()
    assert cli.main(["score", str(p)]) == 0


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"checkpoint": "c", "reference": "r"}, "out: missing required key"),
        ({"checkpoint": "c", "reference": "r", "out": "o", "extra": 1}, "extra: unknown key"),
        (
            {"checkpoint": "c", "reference": "r", "out": "o", "scorer": {"nope": {}}},
            "scorer.nope: unknown scorer",
        ),
        (
            {"checkpoint": "c", "reference": "r", "out": "o", "scorer": {"topk": {"repeats": 2}}},
            "scorer.topk.repeats: unknown key",
        ),
    ],
)
def test_bad_configs_are_refused_naming_the_key(tmp_path, capsys, body, match):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body))
    assert cli.main(["score", str(p)]) == 2
    assert match in capsys.readouterr().err


def test_set_needs_a_config(tmp_path, capsys):
    assert cli.main(["score", str(tmp_path), "--set", "a=b"]) == 2
    assert "--set needs a config file" in capsys.readouterr().err


# ---------------------------------------------------------------- a third-party scorer, installed for real


PLUGIN = """
from pydantic import BaseModel
from qda_kl.scorers import KLScorer, Outcome


class Reading(BaseModel):
    kl: float


class Constant(KLScorer):
    name = "constant"
    value: float = 0.5

    def run(self, checkpoint, reference):
        return Outcome([Reading(kl=self.value)], Reading(kl=self.value))


def register():
    from qda_kl.scorers import SCORERS
    SCORERS.register("constant", Constant)
"""


def test_a_third_party_scorer_loads_through_its_entry_point(tmp_path, monkeypatch):
    dist = tmp_path / "site" / "qda_kl_constant-0.3.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: qda-kl-constant\nVersion: 0.3.0\n")
    (dist / "entry_points.txt").write_text("[qda.plugins]\nconstant = qda_kl_constant:register\n")
    (tmp_path / "site" / "qda_kl_constant.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    fresh = Registry("scorers", dict(SCORERS._entries), group="qda_kl.scorers")
    monkeypatch.setattr(scorers, "SCORERS", fresh)
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    assert list(fresh) == ["constant", "topk"]
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "c",
                "reference": str(tmp_path),
                "out": str(tmp_path / "run"),
                "scorer": {"constant": {"value": 0.25}},
            }
        )
    )
    assert cli.main(["score", str(p)]) == 0
    assert json.loads((tmp_path / "run" / "score.json").read_text()) == {"kl": 0.25}
    prov = yaml.safe_load((tmp_path / "run" / "resolved.yaml").read_text())["provenance"]["plugins"]
    assert prov == {"constant": {"group": "qda_kl.scorers", "package": "qda-kl-constant", "version": "0.3.0"}}


def test_an_invalid_rerun_removes_the_old_result(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))
    assert cli.main(["score", str(config(tmp_path))]) == 0
    run = tmp_path / "run"
    assert (run / "score.json").exists() and (run / "resolved.yaml").exists()
    assert cli.main(["score", str(run / "resolved.yaml"), "--set", "scorer.topk.repeat=0"]) == 2
    assert not (run / "score.json").exists() and not (run / "resolved.yaml").exists()


@pytest.mark.parametrize("bad", ["scorer.topk.repeat.bad=1", "out.x=1", "nonsense"])
def test_a_rerun_with_an_override_that_does_not_apply_removes_the_old_result(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))
    assert cli.main(["score", str(config(tmp_path))]) == 0
    run = tmp_path / "run"
    resolved = (run / "resolved.yaml").read_text()
    (tmp_path / "again.yaml").write_text(resolved)
    assert cli.main(["score", str(tmp_path / "again.yaml"), "--set", bad]) == 2
    assert not (run / "score.json").exists() and not (run / "resolved.yaml").exists()


def test_reading_a_score_config_imports_no_build_code():
    import subprocess
    import sys

    code = (
        "import sys, qda_kl.config; print(sorted(m for m in sys.modules if m.startswith('qda_quant.engine')))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_a_missing_local_checkpoint_names_the_build_not_a_hub_typo(tmp_path):
    """First-user 11-13 step 5: after a refused build the scorer read the output name as a Hub id."""
    missing = str(tmp_path / "nous-llama-3.1-8b-w4a16")
    with pytest.raises(Refused, match=re.escape(f"no checkpoint at {missing} (did the build fail?)")):
        check_checkpoint(missing)
    check_checkpoint(str(tmp_path))  # a directory is what the build writes; whether it is a model is not this
    check_checkpoint("org/not-downloaded-yet")  # a Hub id is the engine's to resolve
    check_checkpoint("org/not-downloaded-yet@v1")


def test_a_bare_name_that_is_not_a_directory_says_how_a_hub_model_is_named(tmp_path, monkeypatch):
    """A one-word name is read as a local output; a Hub model of the old one-word kind is told its org/name form."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        Refused, match=re.escape("no checkpoint at gpt2 (did the build fail?; a Hub model is named org/name)")
    ):
        check_checkpoint("gpt2")


def test_a_missing_local_checkpoint_is_refused_before_the_engine(tmp_path):
    """The refusal comes before the reference is read or an engine reserves a card."""
    with pytest.raises(Refused, match="did the build fail"):
        score.Scorer(str(tmp_path / "gone-w4a16"), str(tmp_path / "ref"))
