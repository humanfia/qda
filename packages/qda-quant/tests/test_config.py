"""One config file per build: `BuildConfig`, ``qda build CONFIG.yaml`` and the ``resolved.yaml`` it writes (CPU)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from qda_quant import Recipe
from qda_quant.cli import main
from qda_quant.config import BuildConfig
from qda_quant.engine import build
from qda_quant.engine.planning import compile_plan, local_source
from qda_quant.engine.stats import hessian_setting
from qda_quant.format import FORMATS
from qda_quant.intake.fingerprint import model_signature
from qda_quant.output.record import RESOLVED
from qda_quant.step import Refused
from safetensors import safe_open

from .engine.test_build import FakeWriter, TinyRTN
from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids, writes
from .goldens import check_golden, refusal


def test_config_import_defers_execution_and_transformers():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from qda_quant.config import BuildConfig; "
            "assert not {'qda_quant.engine.runner', 'transformers'} & sys.modules.keys(); "
            "assert not any(name.startswith('qda_quant.steps.') for name in sys.modules)",
        ],
        check=True,
    )


def test_only_the_gguf_exporter_needs_the_gguf_extra(tiny):
    """A config that does not write a GGUF must not import the optional ``gguf`` package, and one that does
    must say so by name: config validation and plan for W4A16/W3A16/W4A8 with ``gguf`` unimportable."""
    import subprocess
    import sys

    script = """
import json
import sys

class BlockGGUF:
    def find_spec(self, name, path=None, target=None):
        if name == "gguf" or name.startswith("gguf."):
            raise ImportError("gguf is not installed")
        return None

sys.meta_path.insert(0, BlockGGUF())
from qda_quant.config import BuildConfig
from qda_quant.step import Refused

base = json.loads(sys.argv[1])
for fmt in ("w4a16", "w3a16", "w4a8"):
    BuildConfig(**{**base, "format": fmt, "out": base["out"] + "-" + fmt}).plan()
try:
    BuildConfig(**{**base, "format": "w4a16-asym-g32", "exporter": "gguf"}).plan()
except Refused as e:
    assert "./packages/qda-quant[hf,gguf]" in str(e), e
else:
    raise SystemExit("a GGUF build planned without the gguf extra")
assert "gguf" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, json.dumps(config(tiny))], check=True)


@pytest.fixture
def tiny(tmp_path):
    """A saved tiny Qwen3 and a fingerprinted token-id calibration file."""
    tiny_qwen3(layers=2, hidden=128, inter=256).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(6)], sha
    )
    return tmp_path


def config(tmp: Path, **over) -> dict:
    """A build config for the tiny model: a model step and a layer step, so the build runs two stages."""
    return {
        "model": str(tmp / "m"),
        "format": "w4a16",
        "devices": ["cpu"],
        "steps": [{"rotate": {}}, {"gptq": {"damp": 0.02}}],
        "calib": {"file": {"path": str(tmp / "c.jsonl"), "samples": 6, "seqlen": 32}},
        "out": str(tmp / "out"),
        **over,
    }


def write(path: Path, d: dict) -> Path:
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def tensors(ckpt: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118  (safe_open has no __iter__)
    return out


def test_rebuild_from_resolved_is_bit_identical(tiny, capsys):
    assert main(["build", str(write(tiny / "build.yaml", config(tiny)))]) == 0
    first = tiny / "out"
    resolved = yaml.safe_load((first / RESOLVED).read_text())

    # every key, every default filled in, the plugins' provenance recorded
    assert list(resolved) == [
        "model", "format", "format_overrides", "target", "devices", "hessian", "batch_size", "residency",
        "nccl_nvls", "dtype", "grid", "steps", "calib", "out", "exporter", "keep_intermediate",
        "ignore", "provenance",
    ]  # fmt: skip
    assert resolved["exporter"] == {"compressed_tensors": {}}
    assert resolved["format"] == "w4a16-asym-g128" and resolved["target"] == "sm90@0.33.0"
    assert resolved["hessian"] == {"dtype": "bf16", "buffer_rows": 0}
    gptq = resolved["steps"][1]["gptq"]
    assert resolved["grid"] == "absmax"  # a step's own grid is recorded only where it differs
    fields = set(type(Recipe.from_dict(resolved).steps[1]).model_fields)
    assert gptq["damp"] == 0.02 and set(gptq) == fields - {"grid"}
    assert resolved["calib"] == {
        "file": {
            "samples": 6,
            "seqlen": 32,
            "seed": 0,
            "chunks": False,
            "contamination_index": None,
            "path": str(tiny / "c.jsonl"),
        }
    }
    prov = resolved["provenance"]
    assert prov["source_snapshot"] is None and prov["source_revision"] is None and prov["qda_commit"]
    assert prov["cards"] == 1 and prov["not_recorded"] == []
    assert set(prov["plugins"]) == {
        "rotate",
        "gptq",
        "minmax",
        "file",
        "sm90",
        "absmax",
        "resident",
        "compressed_tensors",
        "w4a16",
    }  # the planner chose resident; w4a16 is the format's registered key
    assert prov["plugins"]["absmax"]["group"] == "qda.grids"
    assert prov["plugins"]["file"]["group"] == "qda.calib"
    package, version = FORMATS.provenance(
        "w4a16"
    )  # the entry point's distribution, or the source tree's module
    assert prov["plugins"]["w4a16"] == {"group": "qda.formats", "package": package, "version": version}
    assert (
        prov["plugins"]["gptq"]["group"] == "qda.steps" and prov["plugins"]["sm90"]["group"] == "qda.targets"
    )
    assert not (first / "recipe.yaml").exists()  # resolved.yaml holds the recipe

    stages = json.loads((first / "build.json").read_text())["stages"]
    assert [(s["kind"], s["steps"]) for s in stages] == [("model", ["rotate"]), ("layers", ["gptq"])]

    second = tiny / "again"
    assert main(["build", str(first / RESOLVED), "--set", f"out={second}"]) == 0
    a, b = tensors(first), tensors(second)
    assert a.keys() == b.keys() and a
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    again = yaml.safe_load((second / RESOLVED).read_text())
    assert {**again, "out": None, "provenance": None} == {**resolved, "out": None, "provenance": None}
    stdout = capsys.readouterr().out
    lines = stdout.splitlines()
    assert any(line.startswith(f"built {second} in ") for line in lines)
    # the last thing a build prints is how to serve it, on the kernel the plan named
    assert lines[-2] == "serve it with:"
    assert f" vllm serve {second} " in lines[-1] and "VLLM_DISABLED_KERNELS=MacheteLinearKernel" in lines[-1]
    assert "--linear-backend" not in lines[-1]  # sm90 lets vLLM pick with Machete off
    assert stdout.count(str(second)) == 2  # the build line and the serve command; staging stays out


@pytest.mark.parametrize(
    ("calib", "where"),
    [
        ({"file": {"path": "x"}}, "file.path"),
        ({"selfgen": {"prompts": "x"}}, "selfgen.prompts"),
        ({"slice": {"pool": "x", "names": "n"}}, "slice.pool"),
        ({"corpus": {"file": {"path": "x"}}}, "corpus.file.path"),
        ({"mix": {"parts": [{"path": "x"}], "rows": 2}}, "mix.parts.0.path"),
        ({"mix": {"parts": [{"path": "x"}], "rows": 2, "heldout": "x"}}, "mix.heldout"),
        ({"mix": {"parts": [{"file": {"path": "x"}}], "samples": 2}}, "mix.parts.0.file.path"),
    ],
)
def test_a_config_resolves_a_source_path_from_its_own_directory(tiny, calib, where):
    """A source declares its path settings; the file resolves them beside itself, nested sources included."""
    got = BuildConfig.from_dict(config(tiny, calib=calib), base=tiny).calib
    for key in where.split("."):
        got = got[int(key)] if key.isdigit() else got[key]
    assert got == str(tiny / "x")


def test_a_config_resolves_a_step_paths_from_its_own_directory(tiny):
    """A step declares its path and model settings (`scale_kd.data`, `.teacher`), beside the config file."""
    cfg = BuildConfig.from_dict(
        config(tiny, model="m", steps=[{"scale_kd": {"teacher": "m", "data": "traces.jsonl"}}]), base=tiny
    )
    kd = cfg.steps[0]["scale_kd"]
    assert kd["teacher"] == str(tiny / "m") and kd["data"] == str(tiny / "traces.jsonl")


def test_a_step_model_that_is_not_beside_the_config_stays_a_hub_id(tiny):
    """A teacher that names no directory beside the file is a hub id: the build reads it from the hub."""
    cfg = BuildConfig.from_dict(
        config(tiny, model="m", steps=[{"scale_kd": {"teacher": "org/teacher"}}]), base=tiny
    )
    assert cfg.steps[0]["scale_kd"]["teacher"] == "org/teacher"


def test_home_paths_plan_and_replay_from_another_directory(tiny, monkeypatch):
    """Home inputs retain their source across cwd changes and resolved.yaml replay."""
    monkeypatch.setenv("HOME", str(tiny))
    path = write(
        tiny / "build.yaml",
        config(
            tiny,
            model="~/m",
            steps=[{"rtn": {}}],
            calib={
                "file": {"path": "~/c.jsonl", "samples": 6, "seqlen": 32},
            },
        ),
    )
    elsewhere = tiny / "elsewhere"
    elsewhere.mkdir()
    for cwd in (tiny / "m", elsewhere):
        monkeypatch.chdir(cwd)
        cfg = BuildConfig.from_file(path)
        cfg.plan()
        resolved = cfg.write_resolved(elsewhere)
        record = yaml.safe_load(resolved.read_text())
        assert record["model"] == str(tiny / "m")
        assert record["calib"]["file"]["path"] == str(tiny / "c.jsonl")
        replay = BuildConfig.from_file(resolved)
        replay.plan()
        assert replay.model_dump(exclude={"provenance"}) == cfg.model_dump(exclude={"provenance"})


def test_a_hub_model_does_not_plan_a_same_named_cwd_directory(tiny, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    path = write(tiny / "build.yaml", config(tiny, model="org/name", out=None))

    def uncached(*args, **kwargs):
        raise LocalEntryNotFoundError("Hub model not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", uncached)
    for name, layers in (("A", 1), ("B", 2)):
        cwd = tiny / name
        tiny_qwen3(layers=layers, hidden=128, inter=256).save_pretrained(cwd / "org" / "name")
        monkeypatch.chdir(cwd)
        assert main(["plan", str(path)]) == 2
        with pytest.raises(Refused, match="Hub model not cached"):
            BuildConfig.from_file(path).plan()


def test_declared_plugin_path_defaults_read_beside_the_config(tiny, monkeypatch):
    from typing import ClassVar

    from qda_quant.calib.sources import SOURCES, FileSource
    from qda_quant.step import REGISTRY, Step

    class DefaultSource(FileSource):
        name: ClassVar[str] = "test_default_source"
        path: str = "default.jsonl"

    class DefaultStep(Step):
        name: ClassVar[str] = "test_default_step"
        paths: ClassVar[tuple[str, ...]] = ("data",)
        data: str = "default.jsonl"

    monkeypatch.setitem(SOURCES._entries, DefaultSource.name, DefaultSource)
    monkeypatch.setitem(REGISTRY._entries, DefaultStep.name, DefaultStep)
    (tiny / "default.jsonl").write_text('{"text": "config data"}\n')
    (tiny / "m" / "default.jsonl").write_text('{"text": "cwd data"}\n')
    path = write(
        tiny / "build.yaml", config(tiny, steps=[{DefaultStep.name: {}}], calib={DefaultSource.name: {}})
    )
    monkeypatch.chdir(tiny / "m")
    cfg = BuildConfig.from_file(path)
    assert cfg.recipe.calib.read("unused").rows == [{"text": "config data"}]
    assert Path(cfg.recipe.steps[0].data).read_text() == (tiny / "default.jsonl").read_text()


def test_a_mix_config_reproduces_from_another_directory(tiny, monkeypatch):
    """A mix's relative parts resolve beside the file, and resolved.yaml rebuilds elsewhere, byte for byte."""
    monkeypatch.setenv("QDA_CACHE", str(tiny / "cache"))
    sha = save_tiny_tokenizer(tiny / "m")
    g = torch.Generator().manual_seed(1)
    for name in ("a", "b"):
        ids = [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(4)]
        write_ids(tiny / f"{name}.jsonl", ids, sha)
    cfg = config(tiny, model="m", calib={"mix": {
        "parts": [{"path": "a.jsonl"}, {"path": "b.jsonl"}], "rows": 6, "samples": 6, "seqlen": 16,
    }})  # fmt: skip
    monkeypatch.chdir(tiny / "m")  # neither the model nor a part is relative to the working directory
    assert main(["build", str(write(tiny / "build.yaml", cfg))]) == 0
    first = yaml.safe_load((tiny / "out" / RESOLVED).read_text())
    assert first["model"] == str(tiny / "m")
    assert [p["path"] for p in first["calib"]["mix"]["parts"]] == [
        str(tiny / f"{n}.jsonl") for n in ("a", "b")
    ]

    (tiny / "elsewhere").mkdir()
    monkeypatch.chdir(tiny / "elsewhere")
    second = tiny / "again"
    assert main(["build", str(tiny / "out" / RESOLVED), "--set", f"out={second}"]) == 0
    a, b = tensors(tiny / "out"), tensors(second)
    assert a.keys() == b.keys() and a
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def test_a_local_model_is_signed_in_provenance(tiny):
    """resolved.yaml names the model directory's content, so a rebuild can tell its weights changed."""
    assert main(["build", str(write(tiny / "build.yaml", config(tiny)))]) == 0
    prov = yaml.safe_load((tiny / "out" / RESOLVED).read_text())["provenance"]
    assert prov["source_signature"] == model_signature(tiny / "m")


def test_the_model_signature_reads_a_shard_header_not_its_values(tmp_path):
    """The signature is cheap and ignores edited weight bytes; another shape is another tensor set."""
    from safetensors.torch import save_file

    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text("{}")
    save_file({"w": torch.zeros(4)}, m / "model.safetensors")
    original = model_signature(m)
    save_file({"w": torch.ones(4)}, m / "model.safetensors")  # same header, other values
    assert model_signature(m) == original
    save_file({"w": torch.zeros(2, 2)}, m / "model.safetensors")  # another shape, same 16 bytes
    assert model_signature(m) != original


def test_python_api_writes_the_same_resolved_config(tiny):
    """`qda_quant.build` is the one build path: called directly, it records what the config route records."""
    cfg = BuildConfig.from_dict(config(tiny))
    cfg.plan()  # the config route records the planned config: every grid named
    recipe = Recipe.from_dict({"steps": cfg.steps, "calib": cfg.calib})
    out = build(
        str(tiny / "m"), recipe, format="w4a16", out_dir=tiny / "out", devices=["cpu"], log=lambda s: None
    )
    resolved = yaml.safe_load((out / RESOLVED).read_text())
    del resolved["provenance"]
    assert resolved == cfg.model_dump(exclude={"provenance", "meta", "drafts", "layer_cache"})


def test_a_relative_path_object_records_a_local_model_for_replay(tiny, monkeypatch):
    """A public API Path keeps its local meaning after serialization and a change of working directory."""
    from qda_quant import build as public_build

    recipe = BuildConfig.from_dict(config(tiny, steps=[{"rtn": {}}])).recipe
    monkeypatch.chdir(tiny)
    out = public_build(Path("m"), recipe, format="w4a16", devices=["cpu"], out_dir=tiny / "out")
    resolved = yaml.safe_load((out / RESOLVED).read_text())
    assert resolved["model"] == str(tiny / "m")
    assert resolved["provenance"]["source_snapshot"] is None
    elsewhere = tiny / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["plan", str(out / RESOLVED), "--set", f"out={elsewhere / 'out'}"]) == 0


def test_public_build_names_the_engine_options():
    """`qda_quant.build` lists the options `engine.build.build` lists, so the public signature cannot drift."""
    import inspect

    import qda_quant

    public = list(inspect.signature(qda_quant.build).parameters)
    engine = list(inspect.signature(build).parameters)
    assert public == engine


def test_a_build_whose_serve_section_has_no_command_warns_instead_of_a_name_error(tmp_path, capsys):
    """`_print_serve` caught `Refused` without importing it (F821): every unrenderable serve section
    raised `NameError` instead of the warning it was written to print."""
    from qda_quant.cli import _print_serve

    out = tmp_path / "ckpt"
    out.mkdir()
    (out / "build.json").write_text(json.dumps({"serve": {"args": {"not_a_setting": 1}}}))
    _print_serve(out)  # reaches the refusal path: an unplanned serve section is not rendered
    assert capsys.readouterr().err.startswith("qda: warning: no serve command: ")


def test_cli_reports_a_bad_config_without_a_traceback(tiny, capsys):
    """A refused config, a refused ``--set`` and a model that is not there each exit 2 with the reason on
    stderr and no traceback, from ``plan`` and ``build``. The wording of each is pinned in the golden file."""
    d = config(tiny)
    del d["format"]
    path = write(tiny / "b.yaml", d)
    missing = write(tiny / "m.yaml", config(tiny, model=str(tiny / "no-such-model")))
    for argv, says in [
        (["build", str(path)], "format: missing required key"),
        (
            ["build", str(write(tiny / "s.yaml", config(tiny))), "--set", "steps.3.damp=1"],
            "--set steps.3.damp=1:",
        ),
        (["plan", str(missing)], "no such directory"),
        (["build", str(missing)], "no such directory"),
    ]:
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert says in err and "Traceback" not in err, argv
    # the Python API still refuses a build with no out; only `qda build` gives a default (next test)
    with pytest.raises(Refused, match="missing required key 'out'"):
        BuildConfig.from_dict(config(tiny, out=None)).build()


def test_a_missing_or_file_model_path_is_refused_without_asking_the_hub(tiny, monkeypatch, capsys):
    """A model path that is not there (a multi-stage recipe's later stage before its earlier stage is built) or
    is a file is refused in two lines about the path; the hub is never asked, so no "not a hub repo" tail."""
    import huggingface_hub

    def no_hub(*a, **k):
        raise AssertionError("a local path reached the hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_hub)
    monkeypatch.chdir(tiny)
    for model, says, fix in [
        ("./start", "/start': no such directory", "build that recipe first"),
        (str(tiny / "c.jsonl"), "a file, not a model directory", "config.json and the *.safetensors shards"),
    ]:
        assert main(["plan", str(write(tiny / "p.yaml", config(tiny, model=model)))]) == 2, model
        err = capsys.readouterr().err
        assert says in err and "fix:" in err and fix in err, err
        assert "hub repo" not in err and "Traceback" not in err, err


def test_a_model_directory_without_safetensors_is_refused_by_build_in_two_lines(tiny, capsys):
    """A local model directory with config and tokenizer but no *.safetensors (a partial download, or a
    pytorch_model.bin-only model) is refused by ``qda build`` before any stage, not by a traceback mid-build."""
    import shutil

    for name, extra, says in [
        ("nw", None, "no *.safetensors weights"),
        ("bin", "pytorch_model.bin", "holds pytorch_model"),
    ]:
        src = tiny / name
        shutil.copytree(tiny / "m", src, ignore=shutil.ignore_patterns("*.safetensors*"))
        if extra:
            (src / extra).write_bytes(b"")
        argv = [
            "build",
            str(write(tiny / f"{name}.yaml", config(tiny, model=str(src), out=str(tiny / f"o-{name}")))),
        ]
        assert main(argv) == 2, name
        err = capsys.readouterr().err
        assert says in err and "fix:" in err and "safetensors" in err and "Traceback" not in err, err
        assert not (tiny / f"o-{name}").exists(), "nothing is built"


def test_plan_refuses_gguf_tensor_types_without_the_gguf_extra(tiny, monkeypatch, capsys):
    """``tensor_types`` rules are checked while the config is read, before the exporter's own refusal: without
    ``gguf`` that check is a short refusal naming the install line, not a traceback."""
    import sys

    monkeypatch.setitem(sys.modules, "gguf", None)
    rules = [{"pattern": "output.weight", "type": "Q6_K"}]
    path = write(
        tiny / "g.yaml",
        config(tiny, format="w4a16-asym-g32", exporter={"gguf": {"tensor_types": rules}}),
    )
    assert main(["plan", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and len(err.splitlines()) <= 2, err
    assert 'uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]"' in err, err  # README's line
    assert "Traceback" not in err, err


def _no_out_recipe(tiny: Path) -> Path:
    """A config with no out in a subdirectory of its own, as a user's recipe tree has them."""
    (tiny / "recipes").mkdir(exist_ok=True)
    return write(tiny / "recipes" / "my-recipe.yaml", config(tiny, out=None, steps=[{"rtn": {}}]))


def test_build_with_no_out_writes_next_to_the_user_not_the_recipe(tiny, monkeypatch, capsys):
    """`qda build tree/x.yaml` with no out writes ./x/ in the CURRENT directory (never tree/x/), says so
    on its first line, and records the path absolute in resolved.yaml."""
    monkeypatch.delenv("QDA_OUTPUT_ROOTS", raising=False)
    recipe = _no_out_recipe(tiny)
    work = tiny / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert main(["build", str(Path("..") / "recipes" / "my-recipe.yaml")]) == 0
    out = work / "my-recipe"
    assert capsys.readouterr().out.splitlines()[0].startswith(f"out {out} ")
    assert (out / RESOLVED).is_file() and not (recipe.parent / "my-recipe").exists()
    assert yaml.safe_load((out / RESOLVED).read_text())["out"] == str(out)  # absolute, not "my-recipe"


def test_build_with_no_out_goes_in_the_first_output_root(tiny, monkeypatch):
    first, second = tiny / "root-a", tiny / "root-b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("QDA_OUTPUT_ROOTS", f"{first}{os.pathsep}{second}")
    recipe = _no_out_recipe(tiny)
    monkeypatch.chdir(tiny)
    assert main(["build", str(recipe)]) == 0
    assert (first / "my-recipe" / RESOLVED).is_file() and not (tiny / "my-recipe").exists()
    assert yaml.safe_load((first / "my-recipe" / RESOLVED).read_text())["out"] == str(first / "my-recipe")


def test_build_with_no_out_refuses_a_default_that_holds_something(tiny, monkeypatch, capsys):
    """A ./<stem>/ already there and not empty (an earlier build, or a same-named recipe from elsewhere) is
    refused as any out is: the default never replaces what is there."""
    monkeypatch.delenv("QDA_OUTPUT_ROOTS", raising=False)
    recipe = _no_out_recipe(tiny)
    monkeypatch.chdir(tiny)
    (tiny / "my-recipe").mkdir()
    (tiny / "my-recipe" / "keep.txt").write_text("mine")
    assert main(["build", str(recipe)]) == 2
    assert "exists and is not an empty directory" in capsys.readouterr().err
    assert (tiny / "my-recipe" / "keep.txt").read_text() == "mine"


def test_overrides_and_canonical_values(tiny):
    cfg = BuildConfig.from_dict(
        config(tiny, target="h200", hessian="fast"),
        overrides=["steps.1.damp=0.003", "calib.file.samples=4", "batch_size=2", "devices=[0, 1]"],
    )
    assert cfg.recipe.steps[1].damp == 0.003 and cfg.recipe.calib.samples == 4
    assert cfg.batch_size == 2 and cfg.devices == [0, 1]
    assert cfg.target == "sm90@0.33.0" and cfg.hessian == {"dtype": "bf16", "buffer_rows": 8192}
    # the full form reads back to itself
    assert BuildConfig.from_dict(cfg.model_dump()).model_dump() == cfg.model_dump()


def test_provenance_records_the_nested_corpus_plugin(tiny):
    """A corpus calibration source records the corpus plugin, not only the source that reads it."""
    d = config(tiny, calib={"corpus": {"file": {"path": str(tiny / "c.jsonl")}}})
    plugins = BuildConfig.from_dict(d).resolved()["provenance"]["plugins"]
    assert plugins["corpus"]["group"] == "qda.calib"
    assert plugins["file"]["group"] == "qda.corpora"


def test_provenance_records_an_unregistered_format_spelling(tiny):
    """A built-in grammar spelling no registry entry holds is recorded under its canonical name, group null."""
    plugins = BuildConfig.from_dict(config(tiny, format="w4a16-sym-g64")).resolved()["provenance"]["plugins"]
    assert plugins["w4a16-sym-g64"]["group"] is None  # not a registered entry: no entry-point provenance
    assert plugins["w4a16-sym-g64"]["package"]  # still named by the package of its module


def test_provenance_is_ignored_on_read(tiny):
    d = config(tiny, provenance={"qda_commit": "abc", "anything": 1})
    assert BuildConfig.from_dict(d).model_dump(exclude={"provenance"}) == BuildConfig.from_dict(
        config(tiny)
    ).model_dump(exclude={"provenance"})


def test_plan_needs_no_out(tiny, capsys):
    assert main(["plan", str(write(tiny / "p.yaml", config(tiny, out=None)))]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "format w4a16-asym-g128",
        "layers 2  passes ['model', 'layers']",
        "dense HummingLinearKernel, bf16 scales, bf16 activations",
        "residency weights resident, activations device, model 0.0 GiB, free cpu, card peak (nvidia-smi) 0.9 GiB: "
        "tensors 0.0 + CUDA context 0.8 + allocator reserve 0.0; tensor peak 0.0 GiB: call arguments 0.0",
        "checkpoint 0.0 GiB to write (predicted)",
    ]


def test_the_plan_warns_about_the_slow_gdn_path(monkeypatch):
    """GDN layers whose kernels would take transformers' torch path: the plan says so once.

    No pinned modeling module is available here, so the check falls back to the availability flags (the
    binding-level decision is covered in `test_fast_kernels`).
    """
    from types import SimpleNamespace

    import qda_quant.checks as checks
    from qda_quant.cli import _slow_gdn

    gdn = SimpleNamespace(layers=[SimpleNamespace(groups=[SimpleNamespace(kind="linear_attn_in")])])
    dense = SimpleNamespace(layers=[SimpleNamespace(groups=[SimpleNamespace(kind="attn_in")])])
    monkeypatch.setattr(checks, "_modeling_modules", lambda: [])
    monkeypatch.setattr("transformers.utils.import_utils.is_flash_linear_attention_available", lambda: False)
    assert "flash-linear-attention" in _slow_gdn(gdn)
    assert _slow_gdn(dense) is None  # nothing to warn about without a GDN layer

    monkeypatch.setattr("transformers.utils.import_utils.is_flash_linear_attention_available", lambda: True)
    monkeypatch.setattr("qda_quant.arch.derive_trace.fla_causal_conv", lambda: object())
    assert _slow_gdn(gdn) is None


def test_unregistered_step_is_recorded_without_a_group(tiny):
    """A step class defined in a script still gets a record; reading it back needs the step registered."""
    recipe = Recipe(steps=[TinyRTN()], calib=BuildConfig.from_dict(config(tiny)).recipe.calib)
    out = build(str(tiny / "m"), recipe, format="w4a16", out_dir=tiny / "o", devices=["cpu"],
                exporter=writes(FakeWriter), log=lambda s: None)  # fmt: skip
    plugins = yaml.safe_load((out / RESOLVED).read_text())["provenance"]["plugins"]
    assert plugins["tiny_rtn"]["group"] is None
    assert plugins["test_writer"]["group"] == "qda.exporters"


# ---------------------------------------------------------------- error messages and recording edges


def test_an_override_can_use_the_key_an_error_prints(tiny):
    """N4: ``steps.1.gptq.damp`` (the error's form) and ``steps.1.damp`` reach the same setting; nested
    mappings outside a step list are walked key by key."""
    named = BuildConfig.from_dict(config(tiny), ["steps.1.gptq.damp=0.005"])
    short = BuildConfig.from_dict(config(tiny), ["steps.1.damp=0.005"])
    assert named.recipe.steps[1].damp == short.recipe.steps[1].damp == 0.005
    assert named.model_dump() == short.model_dump()
    null = BuildConfig.from_dict(config(tiny, steps=[{"rotate": None}]), ["steps.0.rotate.seed=5"])
    assert null.recipe.steps[0].seed == 5
    meta = BuildConfig.from_dict(config(tiny, meta={"a": {"b": 1}}), ["meta.a.b=2", "meta.c.d=3"])
    assert meta.meta == {"a": {"b": 2}, "c": {"d": 3}}


def test_a_config_dtype_is_recorded_and_reaches_the_rotation(tiny, capsys):
    """``dtype: fp32`` keeps the rotation's full precision into gptq: a config key, recorded, and not the bf16
    checkpoint bytes. The Python API keeps a dtype no config names out of ``resolved.yaml``, naming it."""
    assert main(["build", str(write(tiny / "bf16.yaml", config(tiny, out=str(tiny / "o-bf16"))))]) == 0
    fp32 = tiny / "fp32.yaml"
    fp32.write_text(yaml.safe_dump(config(tiny, out=str(tiny / "o-fp32"), dtype="fp32")))
    assert main(["build", str(fp32)]) == 0
    bf16 = yaml.safe_load((tiny / "o-bf16" / RESOLVED).read_text())
    full = yaml.safe_load((tiny / "o-fp32" / RESOLVED).read_text())
    assert bf16["dtype"] == "bf16" and full["dtype"] == "fp32"
    assert bf16["provenance"]["not_recorded"] == full["provenance"]["not_recorded"] == []
    # build.json names the loaded dtype beside hessian's own: a rebuild can tell the arms apart
    assert json.loads((tiny / "o-bf16" / "build.json").read_text())["load_dtype"] == "bfloat16"
    assert json.loads((tiny / "o-fp32" / "build.json").read_text())["load_dtype"] == "float32"
    # the load dtype reaches the load: the same recipe quantized from fp32 does not store the bf16 scales
    a, b = tensors(tiny / "o-bf16"), tensors(tiny / "o-fp32")
    scales = [k for k in a if k.endswith(".weight_scale")]
    assert scales and any(not torch.equal(a[k], b[k]) for k in scales)
    # the Python API: torch.float32 is a config key and is recorded; a dtype no config names says so instead
    recipe = BuildConfig.from_dict(config(tiny, steps=[{"gptq": {}}])).recipe
    out = build(str(tiny / "m"), recipe, format="w4a16", devices=["cpu"], out_dir=tiny / "api-fp32",
                dtype=torch.float32, log=lambda s: None)  # fmt: skip
    prov = yaml.safe_load((out / RESOLVED).read_text())
    assert prov["dtype"] == "fp32" and prov["provenance"]["not_recorded"] == []
    out = build(str(tiny / "m"), recipe, format="w4a16", devices=["cpu"], out_dir=tiny / "api-fp64",
                dtype=torch.float64, log=lambda s: None)  # fmt: skip
    prov = yaml.safe_load((out / RESOLVED).read_text())
    assert prov["dtype"] == "bf16" and prov["provenance"]["not_recorded"] == ["dtype"]


def test_provenance_records_cards_and_the_hub_revision(tiny, monkeypatch):
    """N6: the card count and a hub id's resolved commit are recorded; ``org/name@rev`` pins the revision."""
    import huggingface_hub

    snapshot = tiny / "hub" / "models--org--tiny" / "snapshots" / "0123abcd"
    snapshot.parent.mkdir(parents=True)
    (tiny / "m").rename(snapshot)
    asked = []

    patterns = []

    def fake_download(repo, revision=None, allow_patterns=None):
        asked.append((repo, revision))
        patterns.append(allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert local_source("org/tiny@v1") == str(snapshot) and asked == [("org/tiny", "v1")]
    # a consumer that serves the checkpoint instead of planning a build wants every file
    assert local_source("org/tiny", weights=None) == str(snapshot) and patterns[-1] is None
    # planning a hub id must not fetch weights: a model can be terabytes
    BuildConfig.from_dict(config(tiny, model="org/tiny")).plan()
    assert "*.safetensors" not in patterns[-1]
    out = BuildConfig.from_dict(config(tiny, model="org/tiny")).build(log=lambda s: None)
    resolved = yaml.safe_load((out / RESOLVED).read_text())
    assert resolved["model"] == "org/tiny" and asked[-1] == ("org/tiny", None)
    prov = resolved["provenance"]
    assert prov["source_snapshot"] == str(snapshot) and prov["source_revision"] == "0123abcd"
    assert prov["source_signature"] is None  # a hub id is pinned by its revision, not signed
    assert prov["cards"] == 1 and prov["not_recorded"] == []


def test_a_gated_hub_repo_is_refused_with_the_licence_and_the_token(monkeypatch):
    """meta-llama/* is gated: with no token the hub answers 401, and the refusal must say what to do about it."""
    import httpx2
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    def gated(repo, **_kw):
        response = httpx2.Response(
            401, request=httpx2.Request("GET", f"https://huggingface.co/api/models/{repo}")
        )
        raise GatedRepoError("401 Client Error: Cannot access gated repo", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    with pytest.raises(Refused) as caught:
        local_source("meta-llama/Llama-3.1-8B-Instruct")
    said = str(caught.value)
    assert "meta-llama/Llama-3.1-8B-Instruct" in said and "gated" in said
    assert "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct" in said
    assert "HF_TOKEN" in said and "hf auth login" in said
    assert "401 Client Error" in said  # the hub's own first line stays, after the fix


def test_a_build_from_a_git_checkout_records_its_commit(tiny):
    """A release build runs from a qda checkout: its resolved.yaml must name the commit, never ``unknown``."""
    import re
    import subprocess

    import qda_quant

    pkg = Path(qda_quant.__file__).resolve().parent
    tracked = subprocess.run(
        ["git", "-C", str(pkg), "ls-files", "--error-unmatch", "__init__.py"],
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("this qda_quant is not in a git checkout (an archive or an installed wheel)")
    assert main(["build", str(write(tiny / "build.yaml", config(tiny)))]) == 0
    commit = yaml.safe_load((tiny / "out" / RESOLVED).read_text())["provenance"]["qda_commit"]
    head = subprocess.run(
        ["git", "-C", str(pkg), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit), commit
    assert commit.removesuffix("-dirty") == head


@pytest.mark.parametrize("explicit", [None, [], ["model.layers.1.mlp.down_proj"]])
def test_adapter_default_ignore_is_resolved_and_overridable(tiny, monkeypatch, explicit):
    """Omitted ignore uses adapter defaults in planning, builds and replay; explicit lists replace them."""
    from qda_quant.arch.adapters import ADAPTERS

    name = "model.layers.0.mlp.down_proj"

    class Defaults:
        def matches(self, model):
            return model.config.model_type == "qwen3"

        def adjust(self, model, graph):
            return graph

        def default_ignore(self, model):
            return (name,)

    monkeypatch.setitem(ADAPTERS._entries, "test_defaults", Defaults())
    data = config(tiny, steps=[{"rtn": {}}])
    if explicit is not None:
        data["ignore"] = explicit
    planned = BuildConfig.from_dict(data)
    expected = {"gdn_gates", "shared_experts", name} if explicit is None else set(explicit)
    assert planned.plan().keep == expected
    assert planned.ignore == (sorted(expected) if explicit is None else explicit)
    out = BuildConfig.from_dict(data).build()
    resolved = yaml.safe_load((out / RESOLVED).read_text())
    assert set(resolved["ignore"]) == expected
    assert BuildConfig.from_dict({**resolved, "out": None}).plan().keep == expected
    assert (name + ".weight" in tensors(out)) == (name in expected)


def test_ignore_roles(tiny):
    """Both roles by default (the paper's coverage); ``[]`` quantizes them; a word that is no role is refused
    (golden row ``ignore a word that is no role``)."""
    assert BuildConfig.from_dict(config(tiny)).ignore == ["gdn_gates", "shared_experts"]
    cfg = BuildConfig.from_dict(config(tiny, ignore=[]))
    assert cfg.ignore == [] and cfg.plan().keep == frozenset()


def test_ignore_keeps_named_linears_float(tiny, capsys):
    """An exact name and a ``re:`` pattern stay float and equal to the source, the exported ``ignore`` lists
    exactly them (so transformers loads them float), and resolved.yaml rebuilds the same checkpoint."""
    from transformers import AutoModelForCausalLM

    ignore = ["model.layers.0.mlp.down_proj", r"re:model\.layers\.1\.self_attn\.[qkv]_proj$"]
    kept = ["model.layers.0.mlp.down_proj", *(f"model.layers.1.self_attn.{p}_proj" for p in "qkv")]
    path = write(tiny / "build.yaml", config(tiny, steps=[{"gptq": {}}], ignore=ignore))
    assert main(["plan", str(path)]) == 0
    assert f"kept bf16 ({ignore[1]}, ignore): 3 weights, e.g. {kept[1]}" in capsys.readouterr().out
    assert main(["build", str(path)]) == 0
    out = tiny / "out"
    t, src = tensors(out), tensors(tiny / "m")
    for k in kept:
        w = t[f"{k}.weight"]
        assert w.dtype == src[f"{k}.weight"].dtype and torch.equal(w, src[f"{k}.weight"]), k
        assert not any(n.startswith(f"{k}.weight_") for n in t), k
    assert "model.layers.0.self_attn.q_proj.weight_packed" in t
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert set(qc["ignore"]) == {*kept, "lm_head", "model.embed_tokens"}
    loaded = AutoModelForCausalLM.from_pretrained(out)
    assert torch.equal(loaded.model.layers[0].mlp.down_proj.weight, src[f"{kept[0]}.weight"])
    assert yaml.safe_load((out / RESOLVED).read_text())["ignore"] == sorted(ignore)
    assert main(["build", str(out / RESOLVED), "--set", f"out={tiny / 'again'}"]) == 0
    again = tensors(tiny / "again")
    assert again.keys() == t.keys() and all(torch.equal(t[k], again[k]) for k in t)


def test_ignore_keeps_a_residual_writer_float_and_rotate_rotates_it_anyway(tiny, capsys):
    """`ignore` decides quantisation, not the basis (ORDER-1748): the plan passes instead of refusing.

    The writer 'ignore' keeps is rotated with the stream and stored in its own dtype, never quantised, and what
    that store costs is in ``build.json`` beside the rest of the run's record.
    """
    ignore = ["model.layers.0.mlp.down_proj"]
    path = write(tiny / "build.yaml", config(tiny, steps=[{"rotate": {}}, {"gptq": {}}], ignore=ignore))
    assert main(["plan", str(path)]) == 0
    assert main(["build", str(path)]) == 0
    capsys.readouterr()
    t, src = tensors(tiny / "out"), tensors(tiny / "m")
    key = f"{ignore[0]}.weight"
    assert t[key].dtype == src[key].dtype  # stored as it is
    assert not torch.equal(t[key], src[key])  # but rotated
    assert not any(n.startswith(key + "_") for n in t)  # and never quantised
    (record,) = json.loads((tiny / "out" / "build.json").read_text())["ignored_writers"]["modules"].items()
    assert record[0] == ignore[0] and record[1] >= 0.0


def test_gguf_refuses_a_non_default_ignore(tiny):
    """A Q4_K_M file takes its tensor types from llama.cpp's rule, which would write an ignored module quantized."""
    from qda_quant.arch import DEFAULT_KEEP
    from qda_quant.format import Format
    from qda_quant.gguf.export import GGUF
    from qda_quant.output.exporters import check_exporter
    from qda_quant.target import Target

    def refusal(keep: frozenset[str]) -> str:
        recipe = Recipe.from_dict({"steps": [{"rtn": {"grid": "q4k"}}], "calib": config(tiny)["calib"]})
        fmt, cpu = Format.parse("w4a16-asym-g32"), [torch.device("cpu")]
        plan = compile_plan(str(tiny / "m"), recipe, fmt, Target(), cpu, keep=keep)
        with pytest.raises(Refused) as e:
            check_exporter(GGUF(llama_cpp=str(tiny / "no-llama.cpp")), plan, recipe)
        return str(e.value)

    assert "ignore" not in refusal(DEFAULT_KEEP)  # the default gets as far as the llama.cpp checkout
    why = refusal(DEFAULT_KEEP | {"model.layers.0.mlp.down_proj"})
    assert "'model.layers.0.mlp.down_proj'" in why and "llama.cpp's rule" in why


def test_a_build_without_drafts_records_no_drafts_key(tiny):
    """The key is left out of resolved.yaml when empty, so a build that names no draft writes the same record."""
    cfg = BuildConfig.from_dict(config(tiny))
    assert "drafts" not in cfg.resolved() and cfg.drafts == []
    assert BuildConfig.from_dict(config(tiny, drafts=["z-lab/Fake-DFlash"])).resolved()["drafts"] == [
        "z-lab/Fake-DFlash"
    ]


def test_a_build_writes_each_rotated_draft_beside_its_checkpoint(tiny, monkeypatch):
    """After the checkpoint is written, each draft is rotated into ``<out>-<draft name>``; none is refused late."""
    import importlib

    engine_build = importlib.import_module(
        "qda_quant.engine.build"
    )  # the package re-exports a function by that name
    from qda_quant.output import rotate_draft

    calls = []
    monkeypatch.setattr(engine_build, "build", lambda *a, out_dir, **kw: Path(out_dir))
    monkeypatch.setattr(rotate_draft, "rotate", lambda ckpt, draft, out: calls.append((ckpt, draft, out)))
    drafts = ["z-lab/Fake-DFlash@abc", str(tiny / "local-dspark")]
    cfg = BuildConfig.from_dict(config(tiny, drafts=drafts))
    out = cfg.build()
    assert calls == [
        (out, drafts[0], tiny / "out-Fake-DFlash"),
        (out, drafts[1], tiny / "out-local-dspark"),
    ]
    (tiny / "out-local-dspark").mkdir()
    with pytest.raises(Refused, match="out-local-dspark exists"):
        cfg.build()


# ---------------------------------------------------------------- refusal wording: golden/errors_config.txt


def _without(tmp: Path, key: str) -> dict:
    d = config(tmp)
    del d[key]
    return d


def _override(tmp: Path, override: str) -> BuildConfig:
    return BuildConfig.from_file(write(tmp / "b.yaml", config(tmp, hessian="fast")), [override])


def _ignore(tmp: Path, ignore: list[str], steps: list | None = None):
    return BuildConfig.from_dict(config(tmp, steps=steps or [{"gptq": {}}], ignore=ignore)).plan()


REFUSALS = {
    "missing key": lambda t: BuildConfig.from_dict(_without(t, "model")),
    "unknown key": lambda t: BuildConfig.from_dict({**config(t), "bacth_size": 2}),
    "unknown step": lambda t: BuildConfig.from_dict(config(t, steps=[{"gptqq": {}}])),
    "unknown step key": lambda t: BuildConfig.from_dict(config(t, steps=[{"gptq": {"dampp": 0.1}}])),
    "step entry with two names": lambda t: BuildConfig.from_dict(
        config(t, steps=[{"gptq": {}, "rotate": {}}])
    ),
    "step entry empty": lambda t: BuildConfig.from_dict(config(t, steps=[{}])),
    "calib unknown key": lambda t: BuildConfig.from_dict(
        config(t, calib={"file": {"path": "c.jsonl", "sampels": 4}})
    ),
    "calib missing key": lambda t: BuildConfig.from_dict(config(t, calib={"file": {"samples": 4}})),
    "hessian dtype": lambda t: BuildConfig.from_dict(config(t, hessian={"dtype": "fp16"})),
    "hessian name": lambda t: BuildConfig.from_dict(config(t, hessian="fastest")),
    "hessian name (python)": lambda t: hessian_setting("fastest"),
    "hessian type (python build)": lambda t: build(
        str(t / "m"),
        BuildConfig.from_dict(config(t)).recipe,
        format="w4a16",
        out_dir=t / "o",
        devices=["cpu"],
        hessian=3,
        log=lambda s: None,
    ),  # fmt: skip
    "no such config file": lambda t: BuildConfig.from_file(t / "nope.yaml"),
    "out missing at build": lambda t: BuildConfig.from_dict(config(t, out=None)).build(),
    "model not found": lambda t: BuildConfig.from_dict(config(t, model=str(t / "no-such-model"))).plan(),
    "set index past the end": lambda t: _override(t, "steps.3.damp=1"),
    "set index not a number": lambda t: _override(t, "steps.x.damp=1"),
    "set into a string": lambda t: _override(t, "model.x=1"),
    "set into a hessian name": lambda t: _override(t, "hessian.dtype=fp32"),
    "set without a value": lambda t: _override(t, "batch_size"),
    "set with bad yaml": lambda t: _override(t, "batch_size=[1,"),
    "ignore a word that is no role": lambda t: _ignore(t, ["routers"]),
    "ignore matches no linear": lambda t: _ignore(t, ["model.layers.0.mlp.down"]),
    # vLLM serves q/k/v as one fused linear with one scheme
    "ignore part of a fused linear": lambda t: _ignore(t, ["model.layers.0.self_attn.q_proj"]),
    # permute moves channels, so an ignored writer would be left in the old order
    "ignore a writer permute moves": lambda t: _ignore(
        t, [r"re:.*mlp\.(gate|up)_proj$"], [{"permute": {}}, {"gptq": {}}]
    ),
    "drafts without rotate": lambda t: BuildConfig.from_dict(config(t, steps=[{"gptq": {}}], drafts=["z/D"])),
    "drafts without residual rotate": lambda t: BuildConfig.from_dict(
        config(t, steps=[{"rotate": {"residual": False}}, {"gptq": {}}], drafts=["z/D"])
    ),
}


THIRD_PARTY_TAIL = {"set with bad yaml": "--set batch_size=[1,:"}


def test_a_top_level_seed_is_an_unknown_key(tiny):
    """The build config has no top-level `seed` (it had one legal value); a step's seed stays."""
    with pytest.raises(Refused, match=r"(?m)^  seed: unknown key$"):
        BuildConfig.from_dict({**config(tiny), "seed": 0})
    rotated = BuildConfig.from_dict(config(tiny, steps=[{"rotate": {"seed": 5}}]))
    assert "seed" not in rotated.recipe.to_dict() and rotated.recipe.steps[0].seed == 5


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusals_keep_their_wording(tiny, name):
    """Each config refusal fires, with the exact text in ``golden/errors_config.txt``."""
    text = refusal(lambda: REFUSALS[name](tiny)).replace(str(tiny), "<tmp>")
    if name in THIRD_PARTY_TAIL:  # pin our words, not the hub client's or PyYAML's
        text = text[: text.index(THIRD_PARTY_TAIL[name]) + len(THIRD_PARTY_TAIL[name])]
    check_golden("config", name, text, list(REFUSALS))


def test_quickstart_config_plans(tiny):
    """``configs/quickstart.yaml`` plans as shipped, with only the model and the calibration swapped for the tiny ones.

    Its steps, format, residency and every other key are kept, so the default recipe's step / format / residency
    combination is planned, not just parsed. The real ``hf`` source needs the Hub and is not read here.
    """
    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "quickstart.yaml").read_text()
    )
    assert "model" in shipped and "hf" in shipped["calib"]
    d = {
        **shipped,
        "model": str(tiny / "m"),
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 6, "seqlen": 32}},
        "devices": ["cpu"],
        "out": str(tiny / "out"),
    }
    plan = BuildConfig.from_dict(d).plan()
    assert plan.fmt.name == shipped["format"]
    assert plan.residency.weights == shipped["residency"]
    # tiny_qwen3 is untied, so the shipped `rotate: {when: accepted}` rotates here: every shipped step is planned
    assert [s.name for _, steps in plan.segments for s in steps] == [next(iter(s)) for s in shipped["steps"]]
    assert plan.steps_skipped == []


def test_quickstart_config_plans_at_w4a4(tiny):
    """The quickstart recipe with only ``format: w4a4`` (and a Blackwell target) plans: its ``hmse`` runs as ``fp4hmse``."""
    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "quickstart.yaml").read_text()
    )
    d = {
        **shipped,
        "format": "w4a4",
        "target": "sm100",
        "model": str(tiny / "m"),
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 6, "seqlen": 32}},
        "devices": ["cpu"],
        "out": str(tiny / "out"),
    }
    cfg = BuildConfig.from_dict(d)
    assert cfg.steps[-1] == {"gptq": {**cfg.steps[-1]["gptq"], "clip": "fp4hmse"}}
    assert cfg.plan().fmt.name == "w4a4-nvfp4"


@pytest.mark.parametrize(
    ("fmt", "step", "clip"),
    [
        ("w4a4", {"gptq": {"clip": "hmse"}}, "fp4hmse"),
        ("w4a4", {"gptq": {"clip": "mse"}}, "fp4hmse"),
        ("w4a4", {"gptq": {}}, "fp4hmse"),
        ("w4a4", {"rtn": None}, "fp4hmse"),
        ("w4a4", {"gptq": {"clip": "minmax"}}, "minmax"),
        ("w4a4", {"rtn": {"clip": "fp4mse"}}, "fp4mse"),
        ("w4a16", {"gptq": {"clip": "hmse"}}, "hmse"),
        ("w4a16", {"gptq": {}}, "minmax"),
    ],
)
def test_w4a4_default_clip(tiny, fmt, step, clip):
    """At ``w4a4`` a quant step that names no clip or a range search runs ``fp4hmse``; other clips and formats keep theirs."""
    d = {
        "model": str(tiny / "m"),
        "format": fmt,
        "steps": [step],
        "calib": {"file": {"path": str(tiny / "c.jsonl")}},
    }
    cfg = BuildConfig.from_dict(d)
    assert cfg.recipe.steps[0].clip == clip
    assert cfg.steps[0][next(iter(step))]["clip"] == clip


def test_w4a4_default_clip_leaves_token_weights(tiny):
    """`guided`'s ``clip`` caps token weights, not a group scale: ``w4a4`` leaves it as set."""
    d = {
        "model": str(tiny / "m"),
        "format": "w4a4",
        "steps": [{"guided": {}}, {"gptq": {}}],
        "calib": {"file": {"path": str(tiny / "c.jsonl")}},
    }
    guided, gptq = BuildConfig.from_dict(d).recipe.steps
    assert (guided.clip, gptq.clip) == (50.0, "fp4hmse")


@pytest.fixture
def tiny_tied(tmp_path):
    """The `tiny` model with its output head tied to the input embedding (the layout `rotate` refuses)."""
    m = tiny_qwen3(layers=2, hidden=128, inter=256)
    m.config.tie_word_embeddings = True
    m.lm_head.weight = m.model.embed_tokens.weight
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(6)], sha
    )
    return tmp_path


ACCEPTED = [{"rotate": {"residual": True, "when": "accepted"}}, {"gptq": {"damp": 0.02}}]


def test_rotate_when_accepted_skips_a_refused_rotation_at_plan_time(tiny_tied, capsys):
    """Tied embeddings: ``when: always`` refuses as before; ``when: accepted`` plans without the step, prints one
    line, and the build records the refusal under ``steps_skipped`` with the same layers as a recipe without it."""
    with pytest.raises(Refused, match="tied input/output embeddings"):
        BuildConfig.from_dict(config(tiny_tied, steps=[{"rotate": {}}, {"gptq": {"damp": 0.02}}])).plan()
    plan = BuildConfig.from_dict(config(tiny_tied, steps=ACCEPTED)).plan()
    assert plan.steps_skipped == [
        {"step": "rotate", "reason": "rotate: tied input/output embeddings; untie them first"}
    ]
    without = BuildConfig.from_dict(config(tiny_tied, steps=ACCEPTED[1:], out=str(tiny_tied / "o2"))).plan()
    names = [(k, [s.name for s in v]) for k, v in plan.segments]
    assert names == [("layers", ["gptq"])] == [(k, [s.name for s in v]) for k, v in without.segments]
    assert plan.residency.as_dict() == without.residency.as_dict()
    assert main(["plan", str(write(tiny_tied / "a.yaml", config(tiny_tied, steps=ACCEPTED)))]) == 0
    assert (
        "step rotate: skipped (when: accepted): rotate: tied input/output embeddings"
        in capsys.readouterr().out
    )
    out = BuildConfig.from_dict(config(tiny_tied, steps=ACCEPTED)).build()
    info = json.loads((out / "build.json").read_text())
    assert info["steps_skipped"] == plan.steps_skipped
    assert [(s["kind"], s["steps"]) for s in info["stages"]] == [("layers", ["gptq"])]
    resolved = yaml.safe_load((out / RESOLVED).read_text())
    assert resolved["steps"][0] == {
        "rotate": {"residual": True, "value_heads": True, "seed": 0, "when": "accepted"}
    }


def test_rotate_when_accepted_rotates_an_untied_model(tiny):
    """Untied: ``when: accepted`` rotates exactly like the default, and records no skip."""
    plan = BuildConfig.from_dict(config(tiny, steps=ACCEPTED)).plan()
    assert plan.steps_skipped == []
    assert [(k, [s.name for s in v]) for k, v in plan.segments] == [
        ("model", ["rotate"]),
        ("layers", ["gptq"]),
    ]
    out = BuildConfig.from_dict(config(tiny, steps=ACCEPTED)).build()
    assert "steps_skipped" not in json.loads((out / "build.json").read_text())


def test_rotate_default_when_leaves_the_record_unchanged(tiny):
    """``when`` at its default is not written: a recipe without it resolves to the same text as before."""
    r = Recipe.from_dict({"steps": [{"rotate": {}}], "calib": {"file": {"path": "c.jsonl"}}})
    assert r.to_dict(full=True)["steps"] == [{"rotate": {"residual": True, "value_heads": True, "seed": 0}}]
    assert r.to_dict()["steps"] == [{"rotate": {}}]
    with pytest.raises(Refused):
        Recipe.from_dict(
            {"steps": [{"rotate": {"when": "sometimes"}}], "calib": {"file": {"path": "c.jsonl"}}}
        )


def test_plan_prices_the_dtype_the_build_loads(tiny, monkeypatch):
    """`qda plan` passes the config's dtype to the planner, as `build` does: an fp32 plan prices fp32 bytes."""
    import torch
    from qda_quant.engine import planning

    seen: list = []
    real = planning.compile_plan

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dtype"))
        return real(*args, **kwargs)

    monkeypatch.setattr(planning, "compile_plan", spy)
    BuildConfig.from_dict(config(tiny, out=None, dtype="fp32")).plan()
    BuildConfig.from_dict(config(tiny, out=None)).plan()
    assert seen == [torch.float32, torch.bfloat16]
