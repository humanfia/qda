"""``qda pipeline``: a staged spec plans every stage, hashes its inputs and outputs, and resumes.

The chain runs on a tiny Qwen3.5 model: the recipe's own steps, a token-id calibration file and a trace
file, so a chain that refuses or resumes here refuses or resumes on a real model for the same reasons.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from importlib import import_module
from pathlib import Path

import pytest
import torch
import yaml
from qda_quant import cli, pipeline
from qda_quant.pipeline.plan import config, expand, with_data_dir, with_root
from qda_quant.pipeline.record import path_sha
from qda_quant.pipeline.run import _watched
from qda_quant.step import Refused

from . import tiny
from .engine_util import save_tiny_tokenizer, write_ids

runner = import_module("qda_quant.pipeline.run")


def ids(rows: int) -> str:
    """A trace file of `rows` traces the scale_kd step can read."""
    row = json.dumps({"prompt_ids": [1, 2, 3], "completion_ids": [4, 5, 6]})
    return "".join(row + "\n" for _ in range(rows))


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Path:
    """A tiny Qwen3.5 model with a tokenizer, its calibration file, a trace file and a two-stage recipe."""
    d = tmp_path_factory.mktemp("pipeline-world")
    model = d / "model"
    tiny.model("qwen3_5").save_pretrained(model)
    sha = save_tiny_tokenizer(model)
    g = torch.Generator().manual_seed(0)
    write_ids(d / "calib.jsonl", [torch.randint(0, 127, (32,), generator=g).tolist() for _ in range(4)], sha)
    (d / "traces.jsonl").write_text(ids(2))
    (d / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(model),
                "format": "w4a16-asym-g128",
                "steps": [
                    {"rotate": {"seed": 35}},
                    {"gptq": {"actorder": "static", "damp": 0.01}},
                    {
                        "scale_kd": {
                            "data": str(d / "traces.jsonl"),
                            "steps": 2,
                            "batch_tokens": 64,
                            "max_len": 16,
                        }
                    },
                ],
                "calib": {"file": {"path": str(d / "calib.jsonl"), "samples": 4, "seqlen": 32}},
            }
        )
    )
    return d


def spec(world: Path, *, name: str = "run", stages=None, data=None, out: str | None = None) -> Path:
    """A pipeline spec beside the tiny recipe: the two stages partition its three steps."""
    doc = {
        "name": "tiny",
        "recipe": "recipe.yaml",
        "out": out or name,
        "data": data if data is not None else {"traces": "traces.jsonl", "calib": "calib.jsonl"},
        "stages": stages
        or [
            {"name": "quant", "steps": "0:2"},
            {"name": "kd", "steps": "2:3", "model": "previous"},
        ],
    }
    p = world / f"{name}.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


CPU_RECIPE = "cpu-recipe.yaml"


def cpu_recipe(world: Path) -> str:
    """Write and name a one-step recipe that runs on the CPU, for chains that run under a card reservation.

    A chain's build stages name `devices: [cpu]`; on a host that really has the reserved cards, a card step
    on the CPU is refused by design. The chains below test the scheduler, not the quantizer, so their recipe
    has one step that needs no card (`rtn`), and one build stage runs it.
    """
    (world / CPU_RECIPE).write_text(
        yaml.safe_dump(
            {
                "model": str(world / "model"),
                "format": "w4a16-asym-g128",
                "steps": [{"rtn": {}}],
                "calib": {"file": {"path": str(world / "calib.jsonl"), "samples": 4, "seqlen": 32}},
            }
        )
    )
    return CPU_RECIPE


def test_plan_runs_every_stage_and_builds_nothing(world, capsys):
    p = pipeline.load(spec(world, name="plan"))
    assert pipeline.plan(p) == 0
    out = capsys.readouterr().out
    assert "PLANNED" in out
    assert "stage quant: build" in out and "stage kd: build" in out
    assert not (world / "plan").exists(), "plan wrote the pipeline's directory"


def test_a_missing_input_is_named(world, capsys):
    p = spec(world, name="missing", data={"traces": "no-such-traces.jsonl"})
    assert pipeline.plan(pipeline.load(p)) == 2
    said = capsys.readouterr()
    assert said.out == "", "a chain that cannot start prints no plan"
    assert said.err.startswith("error: pipeline tiny: not planned: ")  # the contract's refusal, on stderr
    assert "missing input" in said.err and "no-such-traces.jsonl" in said.err


def test_a_pin_that_does_not_match_is_refused(world):
    p = pipeline.load(spec(world, name="pin", data={"traces": {"path": "traces.jsonl", "sha256": "0" * 64}}))
    with pytest.raises(Refused, match="pin does not match"):
        p.check_data()


def test_a_pipeline_follows_the_output_roots_rule(world, capsys, monkeypatch):
    """``QDA_OUTPUT_ROOTS`` confines a pipeline the way it confines `qda build`: relative goes in, outside is refused."""
    monkeypatch.setenv("QDA_OUTPUT_ROOTS", "/var/tmp")
    with pytest.raises(Refused, match="is outside QDA_OUTPUT_ROOTS"):
        pipeline.load(spec(world, name="disk-pool", out="/srv/pool/qda-builds/x"))
    # --out is the other way of naming the root, and follows the same rule in both forms
    p = pipeline.load(spec(world, name="disk-moved", out="qda-builds/moved"))
    with pytest.raises(Refused, match="is outside QDA_OUTPUT_ROOTS"):
        with_root(p, Path("/srv/pool/qda-builds/moved"))
    assert with_root(p, Path("qda-builds/moved2")).root == Path("/var/tmp/qda-builds/moved2")
    # check_out is the guard for a root that moved after the rule did: it refuses before any work, and plan says so
    loose = pipeline.load(spec(world, name="disk-loose", out="qda-builds/loose"))
    monkeypatch.setenv("QDA_OUTPUT_ROOTS", "/tmp")
    with pytest.raises(Refused, match="is outside QDA_OUTPUT_ROOTS"):
        loose.check_out()
    assert pipeline.plan(loose) == 2
    assert "QDA_OUTPUT_ROOTS" in capsys.readouterr().err
    monkeypatch.setenv("QDA_OUTPUT_ROOTS", "/var/tmp")
    # a relative out goes under the first root, not beside the spec
    rel = pipeline.load(spec(world, name="disk-rel", out="qda-builds/rel"))
    assert rel.root == Path("/var/tmp/qda-builds/rel"), rel.root
    rel.check_out()
    # unset, the rule does not apply: relative paths are beside the spec, as before
    monkeypatch.delenv("QDA_OUTPUT_ROOTS")
    assert pipeline.load(spec(world, name="disk-off", out="qda-builds/off")).root == world / "qda-builds/off"


def test_the_stages_must_run_every_step_once_in_order(world):
    gap = spec(world, name="gap", stages=[{"name": "quant", "steps": "0:1"}, {"name": "kd", "steps": "2:3"}])
    with pytest.raises(Refused, match="missing \\[1\\]"):
        pipeline.load(gap)
    twice = spec(
        world, name="twice", stages=[{"name": "quant", "steps": "0:3"}, {"name": "kd", "steps": "2:3"}]
    )
    with pytest.raises(Refused, match="repeated \\[2\\]"):
        pipeline.load(twice)


def test_a_run_builds_the_chain_and_records_every_hash(world):
    p = pipeline.load(spec(world, name="built"))
    assert pipeline.run(p) == 0
    record = json.loads((p.root / pipeline.RECORD).read_text())
    assert sorted(record["stages"]) == ["kd", "quant"]
    for name in ("quant", "kd"):
        stage = record["stages"][name]
        assert Path(stage["out"], "build.json").is_file(), name
        assert stage["key"]["inputs"], f"{name}: no input hashed"
        assert stage["output"]["sha256"] == path_sha(Path(stage["out"]))["sha256"]
    assert (Path(record["stages"]["kd"]["out"]) / "build.json").is_file()
    # the second stage read the first stage's checkpoint, and its record says so
    assert record["stages"]["quant"]["out"] in json.dumps(record["stages"]["kd"]["key"])


def test_a_second_run_skips_every_recorded_stage(world, capsys):
    p = pipeline.load(spec(world, name="built"))
    before = json.loads((p.root / pipeline.RECORD).read_text())
    assert pipeline.run(p) == 0
    after = json.loads((p.root / pipeline.RECORD).read_text())
    assert capsys.readouterr().out.count("skipped") == 2
    assert [h["finished"] for h in after["history"]] == [h["finished"] for h in before["history"]]


def test_a_changed_input_is_refused_naming_it(world):
    p = pipeline.load(spec(world, name="built"))
    traces = world / "traces.jsonl"
    was = traces.read_text()
    try:
        traces.write_text(was + ids(1))
        with pytest.raises(Refused, match="inputs changed.*traces.jsonl"):
            pipeline.run(p)
    finally:
        traces.write_text(was)


def test_a_changed_output_is_refused(world):
    p = pipeline.load(spec(world, name="built"))
    record = json.loads((p.root / pipeline.RECORD).read_text())
    planted = Path(record["stages"]["quant"]["out"]) / "planted.txt"
    planted.write_text("someone else's\n")
    try:
        with pytest.raises(Refused, match="changed since the pipeline wrote it"):
            pipeline.run(p)
    finally:
        planted.unlink()


def test_a_step_card_limit_is_refused_by_the_stage_plan(world, capsys):
    """AutoRound fits the float stream's target, which one card alone keeps: the stage's own ``qda plan``
    refuses the build before anything is spent. GSQ plans on two cards (it gathers `layer_io` and its
    targets), so the single-card limit is autoround's `layer_io_float`."""
    (world / "autoround.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(world / "model"),
                "format": "w4a16-asym-g128",
                "steps": [{"rtn": {}}, {"autoround": {"iters": 1}}],
                "calib": {"file": {"path": str(world / "calib.jsonl"), "samples": 4, "seqlen": 32}},
            }
        )
    )
    doc = {
        "recipe": "autoround.yaml",
        "out": "autoround-out",
        "stages": [{"name": "autoround", "steps": "0:2", "set": {"devices": ["cpu", "cpu"]}}],
    }
    p = world / "autoround.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    assert pipeline.plan(pipeline.load(p)) == 2
    said = capsys.readouterr().out
    assert "1 card only" in said and "autoround" in said
    # GSQ takes the second card too: it declares `multi_card_layer_io`, so the plan accepts two
    (world / "gsq.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(world / "model"),
                "format": "w4a16-asym-g128",
                "steps": [{"rtn": {}}, {"gsq": {"epochs": 1, "batches": 1, "validation": 1}}],
                "calib": {"file": {"path": str(world / "calib.jsonl"), "samples": 4, "seqlen": 32}},
            }
        )
    )
    (world / "gsq.spec.yaml").write_text(
        yaml.safe_dump(
            {
                "recipe": "gsq.yaml",
                "out": "gsq-out",
                "stages": [{"name": "gsq", "steps": "0:2", "set": {"devices": ["cpu", "cpu"]}}],
            }
        )
    )
    assert pipeline.plan(pipeline.load(world / "gsq.spec.yaml")) == 0


def test_a_command_stage_writes_a_data_entry_a_later_stage_reads(world, capsys):
    """The chain's writer slot: a command writes the traces; the next stage reads the file it wrote."""
    produced = world / "written-traces.jsonl"
    produced.unlink(missing_ok=True)
    doc = {
        "recipe": "recipe.yaml",
        "out": "with-command",
        "data": {
            "written": {"path": "written-traces.jsonl", "produced_by": "write-traces"},
            "calib": "calib.jsonl",
        },
        "stages": [
            {"name": "quant", "steps": "0:2"},
            {
                "name": "write-traces",
                "out": "written-traces.jsonl",
                "command": [
                    sys.executable,
                    "-c",
                    "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
                    str(world / "traces.jsonl"),
                    "{out}",
                ],
            },
            {"name": "kd", "steps": "2:3", "set": {"steps.0.data": str(produced)}},
        ],
    }
    p = world / "with-command.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    chain = pipeline.load(p)
    assert pipeline.plan(chain) == 0, capsys.readouterr().out
    assert pipeline.run(chain) == 0
    assert produced.is_file()
    record = json.loads((pipeline.load(p).root / pipeline.RECORD).read_text())
    assert record["data"]["written"]["sha256"] == path_sha(produced)["sha256"]
    assert str(produced) in json.dumps(record["stages"]["kd"]["key"]["inputs"])


def test_a_writer_placed_after_its_reader_is_refused(world):
    doc = {
        "recipe": "recipe.yaml",
        "out": "writer-late",
        "data": {"written": {"path": "written-traces.jsonl", "produced_by": "write-traces"}},
        "stages": [
            {"name": "quant", "steps": "0:2"},
            {"name": "kd", "steps": "2:3", "set": {"steps.0.data": str(world / "written-traces.jsonl")}},
            {"name": "write-traces", "command": ["true"]},
        ],
    }
    p = world / "writer-late.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(Refused, match="writes after it"):
        pipeline.load(p)


def writer_spec(world: Path, name: str = "from-empty") -> Path:
    """A chain whose first stage writes a data file it declares and no ``out``, and whose last stage reads it."""
    doc = {
        "recipe": "recipe.yaml",
        "out": name,
        "data": {
            "calib": "calib.jsonl",
            "written": {"path": "produced/written.jsonl", "produced_by": "write"},
        },
        "stages": [
            {
                "name": "write",
                "command": [
                    sys.executable,
                    "-c",
                    "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
                    "traces.jsonl",
                    "{data:written}",
                ],
            },
            {"name": "quant", "steps": "0:2"},
            {"name": "kd", "steps": "2:3", "set": {"steps.0.data": "{data:written}"}},
        ],
    }
    p = world / f"{name}.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_a_command_stage_with_no_out_writes_the_data_it_declares(world, capsys):
    """The from-empty shape: the writer's outputs are its data entries, hashed, and the build stage reads one."""
    written = world / "produced" / "written.jsonl"
    written.unlink(missing_ok=True)
    (world / "produced").mkdir(exist_ok=True)
    p = pipeline.load(writer_spec(world))
    assert p.stages[0].out is None, "a stage that only writes data names no out"
    assert pipeline.plan(p) == 0, capsys.readouterr().out
    assert pipeline.run(p) == 0
    record = json.loads((p.root / pipeline.RECORD).read_text())
    assert record["stages"]["write"]["out"] is None
    assert record["stages"]["write"]["outputs"][str(written)]["sha256"] == path_sha(written)["sha256"]
    assert str(written) in record["stages"]["kd"]["key"]["inputs"], (
        "the reader did not hash what the writer wrote"
    )
    capsys.readouterr()
    assert pipeline.run(pipeline.load(writer_spec(world))) == 0
    assert capsys.readouterr().out.count("skipped") == 3


def test_a_stage_that_needs_a_card_this_host_lacks_is_3(world, capsys, monkeypatch):
    """`qda pipeline plan` reports the stage's own status: a card the process cannot see is the machine."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # this process was given no card at all
    doc = {
        "recipe": "recipe.yaml",
        "out": "no-card",
        "stages": [{"name": "quant", "steps": "0:3", "set": {"devices": [0]}}],
    }
    p = world / "no-card.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    assert pipeline.plan(pipeline.load(p)) == 3
    said = capsys.readouterr()
    assert "stage quant: build: REFUSED: " in said.out and "is not visible" in said.out  # the stage's report
    # the refusal itself is the contract's, on stderr, naming the stage
    assert said.err.startswith("error: pipeline no-card.spec: 1 stage(s) do not plan: stage quant: ")
    assert "is not visible" in said.err and "PLANNED" not in said.out


def test_a_produced_file_with_no_record_is_refused(world, capsys):
    """A file already at a writer's path is someone else's: the chain does not adopt it and call the stage done."""
    (world / "produced").mkdir(exist_ok=True)
    written = world / "produced" / "written.jsonl"
    written.write_text(ids(1))
    try:
        p = pipeline.load(writer_spec(world, "already-there"))
        assert pipeline.plan(p) == 2
        assert "holds something and no record matches it" in capsys.readouterr().out
    finally:
        written.unlink(missing_ok=True)


def test_the_data_dir_option_moves_every_relative_entry(world):
    """``--data-dir`` runs one spec twice: over the data on disk, then from the empty directory its stages fill."""
    doc = {
        "recipe": "recipe.yaml",
        "out": "data-dir",
        "data_dir": "data",
        "data": {"calib": "calib.jsonl", "traces": "traces.jsonl"},
        "stages": [{"name": "quant", "steps": "0:2"}, {"name": "kd", "steps": "2:3", "model": "previous"}],
    }
    p = world / "data-dir.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    chain = pipeline.load(p)
    assert chain.data["calib"].path == world / "data" / "calib.jsonl"
    moved = with_data_dir(chain, Path("/var/tmp/qda-data/x"))
    assert moved.data["traces"].path == Path("/var/tmp/qda-data/x/traces.jsonl")
    assert moved.data_dir == Path("/var/tmp/qda-data/x")


def test_a_set_value_takes_data_placeholders(world):
    """A build stage reads a produced file by entry, and ``{dir:}``/``{stem:}`` name the path a writer writes beside."""
    other = world / "other-traces.jsonl"
    other.write_text(ids(3))
    chain = pipeline.load(spec(world, name="set-fill", data={"other": "other-traces.jsonl"}))
    stage = pipeline.Stage(
        "kd", "build", None, world / "set-fill-out", (2, 3), {"steps.0.data": "{data:other}"}
    )
    cfg = config(chain, stage)
    assert cfg.model_dump(mode="json")["steps"][0]["scale_kd"]["data"] == str(other)
    writer = pipeline.Stage("w", "command", None, None, command=["x", "{stem:t}", "{dir:t}"])
    p = pipeline.load(spec(world, name="stem", data={"t": "sub/dir/name.jsonl"}))
    assert expand(p, writer)[1:] == [str(world / "sub/dir/name"), str(world / "sub/dir")]
    # the PREFIX of the trace writer, whose own -traces / -answers ending the entry's name already carries
    for entry, prefix in (("a/r1-traces.jsonl", "a/r1"), ("own-answers.jsonl", "own"), ("x-traces", "x")):
        q = pipeline.load(spec(world, name="stem-" + prefix.replace("/", "-"), data={"t": entry}))
        assert expand(q, writer)[1] == str(world / prefix)
    with pytest.raises(Refused, match="is not filled here"):
        pipeline.load(
            spec(
                world,
                name="bad-set",
                stages=[
                    {"name": "quant", "steps": "0:2"},
                    {"name": "kd", "steps": "2:3", "set": {"devices": "{model}"}},
                ],
            )
        )


def test_model_recipe_names_the_recipes_own_model(world):
    """A trace writer runs on the original model without the spec repeating the recipe's model again."""
    doc = {
        "recipe": "recipe.yaml",
        "out": "recipe-model",
        "stages": [
            {"name": "own", "model": "recipe", "command": ["{python}", "-c", "pass", "{model}"]},
            {"name": "quant", "steps": "0:2"},
            {"name": "kd", "steps": "2:3", "model": "previous"},
        ],
    }
    p = world / "recipe-model.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    chain = pipeline.load(p)
    assert expand(chain, chain.stages[0])[-1] == str(world / "model")


def test_devices_come_from_the_reservation_and_nowhere_else(monkeypatch):
    """`--devices` allocates the cards a reservation gave this process; an unreserved card is never started."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(Refused, match="never reads the node's card table"):
        pipeline.cards_of("auto")
    with pytest.raises(Refused, match="names no card"):
        pipeline.cards_of("4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert pipeline.cards_of("auto") == ["4", "5"]
    assert pipeline.cards_of("5") == ["5"]
    with pytest.raises(Refused, match="never starts a card it was not given"):
        pipeline.cards_of("6")
    assert pipeline.cards_of(None) is None


def test_independent_stages_run_at_once_on_their_own_cards(world, monkeypatch):
    """The graph: two writers with no input between them start together, each on the card it was given."""
    one, two = world / "side-one.jsonl", world / "side-two.jsonl"
    for path in (one, two):
        path.unlink(missing_ok=True)
    recipe = cpu_recipe(world)
    doc = {
        "recipe": recipe,
        "out": "parallel",
        "data": {
            "one": {"path": "side-one.jsonl", "produced_by": "one"},
            "two": {"path": "side-two.jsonl", "produced_by": "two"},
            "calib": "calib.jsonl",
        },
        "stages": [
            {
                "name": "one",
                "command": [
                    sys.executable,
                    "-c",
                    "import os, sys, time; time.sleep(2); open(sys.argv[1], 'w').write("
                    "os.environ['CUDA_VISIBLE_DEVICES'])",
                    "{data:one}",
                ],
            },
            {
                "name": "two",
                "command": [
                    sys.executable,
                    "-c",
                    "import os, sys, time; time.sleep(2); open(sys.argv[1], 'w').write("
                    "os.environ['CUDA_VISIBLE_DEVICES'])",
                    "{data:two}",
                ],
            },
            {"name": "quant", "steps": "0:1", "set": {"devices": ["cpu"]}},
        ],
    }
    p = world / "parallel.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,8")
    assert pipeline.run(pipeline.load(p), "auto") == 0
    assert one.read_text() != two.read_text(), "both writers ran on one card"
    assert {one.read_text(), two.read_text()} == {"7", "8"}
    record = json.loads((pipeline.load(p).root / pipeline.RECORD).read_text())
    assert record["stages"]["one"]["cards"] == [one.read_text()]

    # the two 2-second writers finished in the same second: they overlapped rather than queueing
    def at(name: str) -> float:
        return datetime.strptime(record["stages"][name]["finished"], "%Y-%m-%dT%H:%M:%SZ").timestamp()

    assert abs(at("one") - at("two")) <= 1, "the independent writers did not run at the same time"


def test_a_build_stage_runs_on_its_reserved_card_and_resumes(world, monkeypatch, capsys):
    """A build stage is its own `qda build` process with its card in the environment, and it still resumes."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "11")
    p = pipeline.load(writer_spec(world, "device-build"))
    assert pipeline.run(p, "auto") == 0
    record = json.loads((p.root / pipeline.RECORD).read_text())
    assert record["stages"]["quant"]["cards"] == ["11"]
    assert Path(record["stages"]["quant"]["log"]).is_file()
    capsys.readouterr()
    assert pipeline.run(pipeline.load(writer_spec(world, "device-build")), "auto") == 0
    assert capsys.readouterr().out.count("skipped") == 3


def relay_spec(world: Path, name: str, *, cpu_only: bool = False) -> Path:
    """A chain on one card: ``first`` writes A, ``second`` and ``third`` each copy A, then the build(s).

    ``first`` and ``second`` take a second each, so the scheduler makes several passes while one runs, and
    ``third`` and ``quant`` wait for the card another stage holds. `cpu_only` takes the CPU-only recipe and
    its single build when the chain must run a build under a real card reservation.
    """
    for x in "abc":
        (world / f"{name}-{x}.jsonl").unlink(missing_ok=True)
    copy = (
        "import shutil, sys, time; time.sleep(float(sys.argv[1])); shutil.copyfile(sys.argv[2], sys.argv[3])"
    )
    builds = (
        [{"name": "quant", "steps": "0:1", "set": {"devices": ["cpu"]}}]
        if cpu_only
        else [
            {"name": "quant", "steps": "0:2", "set": {"devices": ["cpu"]}},
            {"name": "kd", "steps": "2:3", "set": {"devices": ["cpu"], "steps.0.data": "{data:c}"}},
        ]
    )
    doc = {
        "recipe": CPU_RECIPE if cpu_only else "recipe.yaml",
        "out": name,
        "data": {
            x: {"path": f"{name}-{x}.jsonl", "produced_by": s}
            for x, s in zip("abc", ("first", "second", "third"), strict=True)
        },
        "stages": [
            {"name": "first", "command": [sys.executable, "-c", copy, "1", "traces.jsonl", "{data:a}"]},
            {"name": "second", "command": [sys.executable, "-c", copy, "1", "{data:a}", "{data:b}"]},
            {"name": "third", "command": [sys.executable, "-c", copy, "0", "{data:a}", "{data:c}"]},
            *builds,
        ],
    }
    p = world / f"{name}.spec.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_a_finished_stage_is_recorded_first_and_each_stage_resolves_once(world, monkeypatch):
    """A stage that exited is recorded before another stage's paths are resolved, and each resolves them once.

    Resolving stats every path a config names, and on 2026-11-10 one such stat on NFS held the scheduler 3 h
    while a finished build sat unreaped and unrecorded. Here every resolution first checks that a stage whose
    output is on disk -- so it has exited a moment later -- is already recorded. One chain checks both: its
    build costs a CPU build.
    """
    cpu_recipe(world)  # the one-step CPU recipe the chain builds under the card reservation
    p = pipeline.load(relay_spec(world, "reap-first", cpu_only=True))
    wrote = {"first": world / "reap-first-a.jsonl", "second": world / "reap-first-b.jsonl"}
    calls = {"_deps": Counter(), "_ready": Counter()}

    def checked(fn):
        real = getattr(runner, fn)

        def call(chain, s, *rest):
            for name, path in wrote.items():
                if path.exists():
                    time.sleep(0.3)  # the writer has exited by now
                    record = p.root / pipeline.RECORD
                    stages = json.loads(record.read_text())["stages"] if record.exists() else {}
                    assert name in stages, f"{s.name}'s paths resolved while {name} had exited unrecorded"
            calls[fn][s.name] += 1
            return real(chain, s, *rest)

        return call

    monkeypatch.setattr(runner, "_deps", checked("_deps"))
    monkeypatch.setattr(runner, "_ready", checked("_ready"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert pipeline.run(p, "auto") == 0
    names = [s.name for s in p.stages]
    assert calls["_deps"] == dict.fromkeys(names, 1)
    # `run` plans every stage once before the graph starts; `third` and `quant` then wait for the card
    assert calls["_ready"] == dict.fromkeys(names, 2)


def test_a_slow_path_resolution_says_so_in_the_log(tmp_path, monkeypatch, capsys):
    """A stat that does not return shows in the pipeline log, with its stage and its path, while it hangs."""
    stuck = tmp_path / "stuck" / "model"
    real = os.path.realpath

    def slow(path, *args, **kwargs):
        time.sleep(0.4)
        return real(path, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(runner, "SLOW_RESOLVE_S", 0.05)
        m.setattr(os.path, "realpath", slow)
        got = _watched("stage 'kd-r2'", stuck.resolve)
    assert got == stuck.resolve()
    out = capsys.readouterr().out
    assert "stage 'kd-r2': resolving its paths has not returned after" in out
    assert f"(at {stuck})" in out


def test_each_stage_records_its_commit_and_a_resume_onto_other_code_says_so(world, capsys, monkeypatch):
    """A chain resumed onto newer code must not claim one version: each stage names the commit that ran it."""
    from qda_quant.output.record import qda_commit

    p = pipeline.load(spec(world, name="commits"))
    assert pipeline.run(p) == 0
    record = json.loads((p.root / pipeline.RECORD).read_text())
    assert {s["commit"] for s in record["stages"].values()} == {qda_commit()}
    assert all(h["commit"] == qda_commit() for h in record["history"])
    capsys.readouterr()
    monkeypatch.setattr(runner, "qda_commit", lambda: "f" * 40)
    assert pipeline.run(p) == 0
    out = capsys.readouterr().out
    assert out.count(f"(it ran on commit {qda_commit()}; this run is {'f' * 40})") == 2


def test_a_writer_into_a_data_directory_resumes_after_the_directory_fills(world, capsys):
    """``--out {dir:NAME}`` names where a stage writes; later files beside its output must not change its key.

    The 9B spec's `public` stage writes into the data directory every later stage fills, so hashing the directory
    refused to resume the from-empty run at its first stage. A stage that writes into a data directory keys on
    its own output, not on the files a later stage adds beside it.
    """
    doc = {
        "recipe": "recipe.yaml",
        "out": "dir-writer",
        "data": {"written": {"path": "dw/written.jsonl", "produced_by": "write"}},
        "stages": [
            {
                "name": "write",
                "command": [
                    sys.executable,
                    "-c",
                    "import pathlib, sys; pathlib.Path(sys.argv[1], 'written.jsonl').write_text('1')",
                    "{dir:written}",
                ],
            },
            {"name": "quant", "steps": "0:2"},
            {"name": "kd", "steps": "2:3"},
        ],
    }
    spec_path = world / "dir-writer.spec.yaml"
    spec_path.write_text(yaml.safe_dump(doc))
    (world / "dw").mkdir(exist_ok=True)
    assert pipeline.run(pipeline.load(spec_path)) == 0
    record = json.loads((pipeline.load(spec_path).root / pipeline.RECORD).read_text())
    assert str(world / "dw") not in record["stages"]["write"]["key"]["inputs"]
    (world / "dw" / "later-stage.jsonl").write_text("written beside it by another stage")
    capsys.readouterr()
    assert pipeline.run(pipeline.load(spec_path)) == 0
    assert capsys.readouterr().out.count("skipped") == 3
    # a record from before the fix hashed the directory: it still resumes
    rec_path = pipeline.load(spec_path).root / pipeline.RECORD
    old = json.loads(rec_path.read_text())
    old["stages"]["write"]["key"]["inputs"][str(world / "dw")] = "0" * 64
    rec_path.write_text(json.dumps(old))
    assert pipeline.run(pipeline.load(spec_path)) == 0
    assert capsys.readouterr().out.count("skipped") == 3


def test_pipeline_has_one_name_per_action(capsys):
    """`qda pipeline build` is the only spelling: `run` was a second name for it and is gone."""
    with pytest.raises(SystemExit) as e:
        cli.main(["pipeline", "run", "spec.yaml"])
    assert e.value.code == 2
    assert "invalid choice: 'run'" in capsys.readouterr().err
