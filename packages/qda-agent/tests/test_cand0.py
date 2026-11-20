"""`qda-agent cand0`: one step builds candidate 0 and writes the run config entry.

CPU only: the build itself is a stand-in `qda` that writes the metadata a real build writes
(`build.json` with its measured `seconds`, and `config.json`), so no card is needed here.
"""

import sys
from pathlib import Path

import pytest
import yaml
from qda_agent import cand0, cli
from qda_agent.config import RunConfig
from tests_support import filled

CONFIGS = Path(__file__).parents[1] / "configs"


def config(tmp_path: Path, *overrides: str) -> cand0.Inputs:
    """What candidate 0 reads of the shipped search config, filled under `tmp_path`, with its own run dir."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    return cand0.Inputs.read(path, [f"run_dir={tmp_path / 'run'}", *overrides])


def fake_qda(tmp_path: Path, seconds: float = 42.5) -> Path:
    """A stand-in `qda build` that writes the checkpoint metadata and exits 0."""
    script = tmp_path / "fake-qda"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, yaml\n"
        "recipe = yaml.safe_load(open(sys.argv[2]))\n"
        "out = recipe['out']\n"
        "os.makedirs(out, exist_ok=True)\n"
        f"open(os.path.join(out, 'build.json'), 'w').write(json.dumps({{'load_to_saved_s': {seconds}}}))\n"
        "open(os.path.join(out, 'config.json'), 'w').write('{}')\n"
    )
    script.chmod(0o755)
    return script


def test_the_recipe_names_the_method_and_its_defaults(tmp_path):
    """No tuned value lives in qda-agent: each method's step is empty, its defaults are qda-quant's."""
    cfg = config(tmp_path)
    gptq = cand0.candidate0_recipe(cfg, method="gptq", out=tmp_path / "c0")
    assert gptq["steps"] == [{"gptq": {}}]
    awq = cand0.candidate0_recipe(cfg, method="awq", out=tmp_path / "c0")
    assert awq["steps"] == [{"awq": {}}]
    both = cand0.candidate0_recipe(cfg, method="gptq", gsq=True, out=tmp_path / "c0")
    assert both["steps"] == [{"gptq": {}}, {"gsq": {}}]  # quantize first, then refine the codes
    assert both["model"] == cfg.model and both["format"] == cfg.format
    assert both["calib"]["file"]["path"] == str(cfg.calib)
    assert "residency" not in both  # the build's own default, not a choice made here


def test_awq_with_gsq_is_refused_because_awq_writes_no_codes(tmp_path):
    """qda-quant's awq only folds scales ("a later `rtn` or `gptq` step to quantize"), so gsq has nothing."""
    with pytest.raises(ValueError, match="nothing to refine"):
        cand0.candidate0_recipe(config(tmp_path), method="awq", gsq=True, out=tmp_path / "c0")


def test_qda_quants_own_check_refuses_gsq_before_a_quantizing_step():
    """The order above is not cosmetic: the step's own order check refuses gsq with no codes to refine."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from qda_quant.steps.gsq import _initial

    class NoCodes:
        qparams: dict = {}

    with pytest.raises(Exception, match="put gptq or rtn before gsq"):
        _initial(NoCodes(), SimpleNamespace(part=None, names="down_proj"), "q")


def test_an_unknown_method_is_refused(tmp_path):
    with pytest.raises(ValueError, match="method"):
        cand0.candidate0_recipe(config(tmp_path), method="rotate", out=tmp_path / "c0")


def test_the_build_time_is_the_one_the_build_recorded(tmp_path):
    """`candidate0_build_s` is the build's own time (build.json), not the whole command's wall clock."""
    script = fake_qda(tmp_path)
    cfg = config(tmp_path, f"toolset.qda=[{script}]")
    built = cand0.build_candidate0(cfg, method="gptq", out=tmp_path / "c0")
    assert built.build_seconds == 42.5
    assert built.wall_seconds >= 0
    assert built.recipe.is_file() and built.recipe.parent == tmp_path  # beside the out, repeatable


def test_an_existing_out_is_refused(tmp_path):
    script = fake_qda(tmp_path)
    cfg = config(tmp_path, f"toolset.qda=[{script}]")
    (tmp_path / "c0").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        cand0.build_candidate0(cfg, method="gptq", out=tmp_path / "c0")


def test_a_config_with_no_build_interpreter_is_refused(tmp_path):
    cfg = config(tmp_path, "toolset.python=null", "toolset.qda=null")
    with pytest.raises(ValueError, match="toolset.python"):
        cand0.build_program(cfg)


def test_the_entry_puts_candidate_0_first_and_fills_the_build_time(tmp_path):
    """The entry is `baselines[0]` plus the measured seconds; the KL is then read at host start."""
    cfg = config(tmp_path)
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=42.5,
        wall_seconds=48.0,
        method="gptq",
        gsq=False,
    )
    changes = cand0.entry(cfg, built)
    assert changes["baselines"][0] == str(tmp_path / "c0")
    assert changes["baselines"][1:] == [str(p) for p in cfg.raw["baselines"]]
    assert changes["judge"] == {"candidate0_build_s": 42.5, "select_candidate0": True}


def test_a_build_json_with_only_phase_timings_is_read_too(tmp_path):
    """`timings.total` (the engine phases) is the fallback when `load_to_saved_s` is absent."""
    out = tmp_path / "c0"
    out.mkdir()
    (out / "build.json").write_text('{"timings": {"total": 7.25}}')
    assert cand0._recorded_seconds(out) == 7.25


def test_an_unrecorded_build_time_falls_back_to_the_wall_clock(tmp_path):
    cfg = config(tmp_path)
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=None,
        wall_seconds=48.25,
        method="gptq",
        gsq=False,
    )
    assert cand0.entry(cfg, built)["judge"]["candidate0_build_s"] == 48.25


def test_a_named_kl_keeps_select_candidate_0_off(tmp_path):
    """A config that already describes candidate 0 keeps its own label; nothing is silently switched."""
    cfg = config(tmp_path, "judge.candidate0_kl=0.1")
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=42.5,
        wall_seconds=48.0,
        method="gptq",
        gsq=False,
    )
    assert "select_candidate0" not in cand0.entry(cfg, built)["judge"]


def test_labeled_starts_keep_select_candidate_0_off(tmp_path):
    """With judge.starts the labels are the run's own; the entry adds no switch of its own."""
    plain = config(tmp_path)
    baseline = plain.raw["baselines"][0]
    cfg = config(tmp_path, f"judge.starts=[{{name: gptq, path: {baseline}, origin: own}}]")
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=42.5,
        wall_seconds=48.0,
        method="gptq",
        gsq=False,
    )
    assert "select_candidate0" not in cand0.entry(cfg, built)["judge"]


def test_the_default_target_is_a_new_file_beside_the_config(tmp_path):
    """The user's file is never rewritten by default."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    before = path.read_text()
    cfg = cand0.Inputs.read(path, [f"run_dir={tmp_path / 'run'}"])
    (tmp_path / "c0").mkdir()
    (tmp_path / "c0" / "config.json").write_text("{}")
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=9.5,
        wall_seconds=12.0,
        method="awq",
        gsq=False,
    )
    target = cand0.write_config(path, raw=yaml.safe_load(before), changes=cand0.entry(cfg, built))
    assert target == path.with_name("search.candidate0.yaml") and target.is_file()
    assert path.read_text() == before  # untouched
    written = RunConfig.load(target)
    assert written.baselines[0] == tmp_path / "c0" and written.judge.candidate0_build_s == 9.5


@pytest.mark.parametrize("level", ["none", "strict"])
def test_the_entry_loads_at_both_levels(tmp_path, level):
    """The generated config is a run config: it loads the same at the product default and the experiment wall."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    cfg = cand0.Inputs.read(path, [f"run_dir={tmp_path / 'run'}", f"isolation.level={level}"])
    (tmp_path / "c0").mkdir()
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=5.0,
        wall_seconds=6.0,
        method="gptq",
        gsq=False,
    )
    target = cand0.write_config(path, raw=yaml.safe_load(path.read_text()), changes=cand0.entry(cfg, built))
    written = RunConfig.load(target, [f"isolation.level={level}"])
    assert written.isolation.level == level and written.judge.candidate0_build_s == 5.0


def test_in_place_refuses_a_commented_config_unless_forced(tmp_path):
    path = tmp_path / "commented.yaml"
    path.write_text("# a comment\nmodel: x\n")
    with pytest.raises(ValueError, match="comments"):
        cand0.write_config(path, raw={"model": "x"}, changes={"baselines": []}, in_place=True)
    target = cand0.write_config(
        path, raw={"model": "x"}, changes={"baselines": []}, in_place=True, force=True
    )
    assert target == path and "baselines" in path.read_text()


def test_the_written_config_carries_the_effective_overrides(tmp_path):
    """The generated file is the run config as the run would see it: every --set applied, not a re-read."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    other = tmp_path / "cfg" / "other-model"
    other.mkdir()
    (other / "config.json").write_text("{}")
    code = cli.main(
        ["cand0", str(path), "--recipe-only", "--set", f"model={other}", "--set", "isolation.level=none"]
    )
    assert code == 0
    cfg = cand0.Inputs.read(path, [f"model={other}", "isolation.level=none", f"run_dir={tmp_path / 'run'}"])
    (tmp_path / "c0").mkdir()
    built = cand0.Candidate0(
        out=tmp_path / "c0",
        recipe=tmp_path / "c0.recipe.yaml",
        build_seconds=5.0,
        wall_seconds=6.0,
        method="gptq",
        gsq=False,
    )
    target = cand0.write_config(path, raw=cfg.raw, changes=cand0.entry(cfg, built))
    written = RunConfig.load(target)
    assert written.model == str(other) and written.isolation.level == "none"


def test_recipe_only_refuses_like_the_build_path(tmp_path, capsys):
    """`--recipe-only --method awq --gsq` is the same clean refusal (message, exit 2), not a traceback."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    assert cli.main(["cand0", str(path), "--recipe-only", "--method", "awq", "--gsq"]) == 2
    captured = capsys.readouterr()
    assert "nothing to refine" in captured.err and captured.out == ""


def test_the_cli_prints_the_recipe_without_a_card(tmp_path, capsys):
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    code = cli.main(["cand0", str(path), "--recipe-only", "--method", "awq"])
    assert code == 0
    recipe = yaml.safe_load(capsys.readouterr().out)
    assert recipe["steps"] == [{"awq": {}}]
    assert str(recipe["model"]).endswith("original-bf16-model")  # the config's model, as given


def test_the_command_refuses_a_config_that_labels_its_starts(tmp_path, capsys):
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    baseline = tmp_path / "cfg" / "baselines" / "gptq"
    code = cli.main(
        ["cand0", str(path), "--set", f"judge.starts=[{{name: gptq, path: {baseline}, origin: own}}]"]
    )
    assert code == 2
    assert "judge.starts" in capsys.readouterr().err


def test_a_fresh_config_builds_candidate_0_before_its_other_inputs_exist(tmp_path, capsys):
    """The shipped config with only the build's keys filled: baselines[0], format_reference and the KL
    reference are still placeholders, and the generated file fills the first two with candidate 0."""
    script = fake_qda(tmp_path)
    model, calib = tmp_path / "model", tmp_path / "calib.jsonl"
    model.mkdir()
    calib.write_text("")
    path = tmp_path / "search.yaml"
    path.write_text((CONFIGS / "search.yaml").read_text())
    sets = [f"model={model}", f"calib={calib}", f"toolset.qda=[{script}]"]
    code = cli.main(
        ["cand0", str(path), "--out", str(tmp_path / "c0"), *[a for s in sets for a in ("--set", s)]]
    )
    assert code == 0, capsys.readouterr().err
    written = yaml.safe_load((tmp_path / "search.candidate0.yaml").read_text())
    assert written["baselines"] == [str(tmp_path / "c0")]
    assert written["judge"]["format_reference"] == str(tmp_path / "c0")
    assert written["judge"]["candidate0_build_s"] == 42.5 and written["judge"]["select_candidate0"] is True
    out = capsys.readouterr().out
    assert "before the run, fix:" in out and "judge.kl_ref" in out and "baselines" not in out.split("fix:")[1]


def test_a_build_child_that_refuses_the_machine_is_reported_as_3(tmp_path, capsys):
    """`qda build` exits 3 when the card cannot hold it, or the toolchain is missing: cand0 says the same.

    The child is a `qda` command, so its status is the answer a caller acts on (`qda_quant.exit`); the
    wrapper passes it through instead of calling every child failure an input error.
    """
    script = tmp_path / "refusing-qda"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('error: 20.0 GiB needed, 8.0 GiB free\\n')\n"
        "sys.exit(3)\n"
    )
    script.chmod(0o755)
    model, calib = tmp_path / "model", tmp_path / "calib.jsonl"
    model.mkdir()
    calib.write_text("")
    path = tmp_path / "search.yaml"
    path.write_text((CONFIGS / "search.yaml").read_text())
    sets = [f"model={model}", f"calib={calib}", f"toolset.qda=[{script}]"]
    code = cli.main(
        ["cand0", str(path), "--out", str(tmp_path / "c0"), *[a for s in sets for a in ("--set", s)]]
    )
    assert code == 3
    assert capsys.readouterr().err.startswith("error: ")


def test_a_placeholder_in_a_key_the_build_reads_is_still_refused(tmp_path, capsys):
    path = tmp_path / "search.yaml"
    path.write_text((CONFIGS / "search.yaml").read_text())
    assert cli.main(["cand0", str(path), "--recipe-only"]) == 2
    err = capsys.readouterr().err
    assert "fill in model" in err and "kl_ref" not in err and "baselines" not in err


def test_the_gguf_method_is_the_measured_q4km_recipe(tmp_path):
    recipe = cand0.candidate0_recipe(config(tmp_path), method="gguf-q4km", out=tmp_path / "c0")
    assert recipe["format"] == "w4a16-asym-g32" and recipe["grid"] == "q4k"
    assert recipe["steps"] == [{"gptq": {"clip": "imatrix", "damp": 0.1}}] and recipe["exporter"] == {
        "gguf": {}
    }
    with pytest.raises(ValueError, match="refuses gsq"):
        cand0.candidate0_recipe(config(tmp_path), method="gguf-q4km", gsq=True, out=tmp_path / "c0")


def test_a_measured_type_map_carries_the_libggml_its_q5k_needs(tmp_path):
    lib = tmp_path / "libggml-base.so"
    types = [{"pattern": "blk.0.ffn_down.weight", "type": "Q5_K"}]
    recipe = cand0.candidate0_recipe(
        config(tmp_path), method="gguf-q4km", out=tmp_path / "c0", tensor_types=types, libggml=lib
    )
    exporter = recipe["exporter"]["gguf"]
    assert exporter["tensor_types"] == types and exporter["libggml"] == str(lib)
    # no type map: the default Q4_K_M mix writes no libggml key, as a run without --sensitivity did
    plain = cand0.candidate0_recipe(config(tmp_path), method="gguf-q4km", out=tmp_path / "c0")
    assert "libggml" not in plain["exporter"]["gguf"]


def test_cards_are_taken_as_cuda_visible_devices_or_as_words():
    assert cand0.parse_cards(["0,1,2,3"]) == cand0.parse_cards(["0", "1", "2", "3"]) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="card indices"):
        cand0.parse_cards(["0,x"])
