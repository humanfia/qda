"""Prepare decides candidate 0's group size from the model's widths, before any step runs.

CPU only: the models are safetensors headers (name and shape), as a plan reads them; no card is asked.
"""

import json
import struct

import pytest
from qda_agent import cli
from qda_agent import prepare as P
from qda_agent.group_size import target_for as read_target  # the autouse `hopper` fixture replaces it
from qda_quant.target import Target


def header_model(root, tensors: dict) -> "object":
    """A model directory whose `model.safetensors` header holds `tensors` (name -> shape); no data bytes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "fake", "architectures": ["Fake"]}))
    header = {n: {"dtype": "BF16", "shape": list(s), "data_offsets": [0, 0]} for n, s in tensors.items()}
    raw = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return root


@pytest.fixture(autouse=True)
def hopper(monkeypatch):
    """No card is read in a plan test: Hopper's table is the one an H100 build plans."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(P.group_size, "target_for", lambda cards: Target.parse("sm90"))


def test_nemotrons_prepare_picks_group_64_before_any_step(tmp_path, capsys):
    m = header_model(
        tmp_path / "nemotron",
        {
            "model.layers.1.mixer.down_proj.weight": (4480, 15680),
            "model.layers.0.mixer.in_proj.weight": (22656, 4480),
        },
    )
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    out = capsys.readouterr().out
    assert p.fmt.name == "w4a16-asym-g64"
    assert p.build_inputs().format == "w4a16-g64"  # candidate 0's recipe
    assert "w4a16: group size 64 (model.layers.1.mixer.down_proj: 15680 is not a multiple of 128)" in out
    assert not (tmp_path / "out").exists()  # a plan writes nothing


def test_deepseek_v2_lites_10944_takes_group_64_too(tmp_path):
    m = header_model(tmp_path / "dsv2lite", {"model.layers.0.mlp.down_proj.weight": (2048, 10944)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    assert p.fmt.name == "w4a16-asym-g64"


def test_widths_that_divide_128_keep_the_default(tmp_path, capsys):
    m = header_model(tmp_path / "qwen", {"model.layers.0.mlp.down_proj.weight": (4096, 14336)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    assert p.fmt.name == "w4a16-asym-g128"
    assert "group size" not in capsys.readouterr().out


def test_the_recipe_and_the_run_config_read_one_decision(tmp_path):
    m = header_model(tmp_path / "nemotron", {"model.layers.0.mixer.down_proj": (4480, 15680)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    (p.cand0).mkdir(parents=True)
    (p.cand0 / "build.json").write_text("{}")
    p.state["cand0"] = {"build_s": 1.0, "source": p.source_identity() or "local"}
    assert P.search_config(p)["format"] == "w4a16-g64"
    assert p.recipe()["format"] == "w4a16-g64"


def test_the_choice_and_its_trigger_are_recorded(tmp_path):
    m = header_model(tmp_path / "nemotron", {"model.layers.0.mixer.down_proj": (4480, 15680)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    p.resolve_format(save=True)
    record = json.loads((tmp_path / "out" / "prepare.json").read_text())["format_choice"]
    assert record["format"] == "w4a16-asym-g64" and record["group_size"] == 64
    assert record["trigger"] == {
        "module": "model.layers.0.mixer.down_proj",
        "width": 15680,
        "reason": "is not a multiple of 128",
    }
    assert 64 in record["feasible_group_sizes"] and 128 not in record["feasible_group_sizes"]


def test_per_channel_is_refused_with_the_format_that_would_work(tmp_path):
    m = header_model(tmp_path / "odd", {"model.layers.0.mlp.down_proj.weight": (1024, 48)})
    with pytest.raises(P.Refusal, match="pass --format w4a16-pc"):
        P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])


def test_an_explicit_group_is_taken_as_given(tmp_path):
    m = header_model(tmp_path / "qwen", {"model.layers.0.mlp.down_proj.weight": (4096, 14336)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16-g64", cards=[0])
    assert p.fmt.name == "w4a16-asym-g64"
    refused = header_model(tmp_path / "nem", {"model.layers.0.mixer.down_proj": (4480, 15680)})
    with pytest.raises(P.Refusal, match="pass --format"):
        P.Prepare(str(refused), tmp_path / "out2", format="w4a16-g128", cards=[0])


def test_the_dry_run_prints_the_group_and_runs_nothing(tmp_path, capsys):
    m = header_model(tmp_path / "nemotron", {"model.layers.0.mixer.down_proj": (4480, 15680)})
    code = cli.main(["prepare", "--model", str(m), "--out", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and not (tmp_path / "out").exists()
    assert "(w4a16-g64, cards" in out
    assert "w4a16: group size 64 (model.layers.0.mixer.down_proj: 15680" in out


def test_a_second_resolve_does_not_repeat_the_line(tmp_path, capsys):
    """A real run resolves twice (the Hub's metadata, then the downloaded headers) and prints one line."""
    m = header_model(tmp_path / "nemotron", {"model.layers.0.mixer.down_proj.weight": (4480, 15680)})
    p = P.Prepare(str(m), tmp_path / "out", format="w4a16", cards=[0])
    assert capsys.readouterr().out.count("group size 64") == 1
    p.resolve_format(save=False)
    assert capsys.readouterr().out == ""


def test_the_target_is_the_drivers_capability_with_no_cuda_context(monkeypatch):
    """`target_for` asks nvidia-smi (`compute_cap`), so a CPU process beside the cards names their table.

    The judge prices builds in the run's CPU process: torch there would open a CUDA context on every card of
    the machine, and on a GB300 its guess of "no card" is sm90, which serves no W4A4.
    """
    from types import SimpleNamespace

    from qda_agent.isolation import devices

    card = SimpleNamespace(bus="0008:06:00.0", ordinal=0, minor=0, uuid="GPU-1")
    monkeypatch.setattr(devices, "smi_capabilities", lambda proc=None: [(card, (10, 3))])
    monkeypatch.setattr(devices, "resolve_cards", lambda cards, proc=None, visible=None: [card])
    monkeypatch.setattr(P.group_size, "_torch_target", lambda cards: pytest.fail("torch was asked"))
    assert read_target([0]).arch == "sm103"


def test_no_card_listed_falls_back_to_the_tables_default(monkeypatch):
    from qda_agent.isolation import devices

    monkeypatch.setattr(devices, "smi_capabilities", lambda proc=None: None)
    asked = []
    monkeypatch.setattr(P.group_size, "_torch_target", lambda cards: asked.append(cards) or Target())
    assert read_target([0]).arch == "sm90"
    assert asked == [[0]], "a machine whose driver cannot be read falls back to torch"


def test_with_target_adds_only_where_a_target_may_go():
    from qda_agent.group_size import with_target

    t = Target.parse("sm103")
    assert with_target({}, t) == {"target": "sm103@0.33.0"}
    assert with_target({"target": "sm90"}, t) == {"target": "sm90"}, "a recipe's own target wins"
    assert with_target({"exporter": "gguf"}, t) == {"exporter": "gguf"}, "llama.cpp decides its own"


def test_the_judges_target_never_opens_cuda_and_is_none_when_the_driver_cannot_tell(monkeypatch):
    """`run_target`: no nvidia-smi answer, or cards no one table covers, leave the recipe on `Target()`."""
    from types import SimpleNamespace

    from qda_agent.group_size import run_target
    from qda_agent.isolation import devices

    monkeypatch.setattr(P.group_size, "_torch_target", lambda cards: pytest.fail("the judge asked torch"))
    monkeypatch.setattr(devices, "smi_capabilities", lambda proc=None: None)
    assert run_target([0]) is None
    a = SimpleNamespace(bus="0000:01:00.0", ordinal=0, minor=0, uuid="GPU-a")
    b = SimpleNamespace(bus="0000:02:00.0", ordinal=1, minor=1, uuid="GPU-b")
    monkeypatch.setattr(devices, "smi_capabilities", lambda proc=None: [(a, (9, 0)), (b, (10, 0))])
    monkeypatch.setattr(devices, "resolve_cards", lambda cards, proc=None, visible=None: [a, b][: len(cards)])
    assert run_target([0, 1]) is None, "an sm90 + sm100 mix: no one table, so no target is added"
    assert run_target([0]).arch == "sm90"
