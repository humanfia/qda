"""`qda inspect` without `--target`: the card this process sees, else the target build.json records, else the default.

An NVFP4 W4A4 checkpoint is planned on sm100 and refused on the default sm90, which has no NVFP4 row. A recorded
target this version cannot parse is a note and the default table, never a refusal with no report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from qda_quant.cli import main
from qda_quant.intake import report
from qda_quant.intake.report import _text, inspect
from safetensors.torch import save_file

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with no card: a test that needs one patches the capability `report.local_target` reads."""
    monkeypatch.setattr(report, "_card_capability", lambda: None)


def _checkpoint(path: Path, record: dict | None = None, quantized: bool = True) -> Path:
    """One Linear and the files vLLM needs beside it; `record` is written to build.json when given."""
    path.mkdir()
    down = "model.layers.0.mlp.down_proj.weight"
    tensors = {"model.embed_tokens.weight": torch.zeros((8, 8), dtype=torch.bfloat16)}
    if quantized:
        tensors.update(
            {
                f"{down}_packed": torch.zeros((8, 16), dtype=torch.uint8),
                f"{down}_scale": torch.zeros((8, 1), dtype=torch.float8_e4m3fn),
                f"{down}_global_scale": torch.zeros((1,), dtype=torch.float32),
            }
        )
    else:
        tensors[down] = torch.zeros((8, 8), dtype=torch.bfloat16)
    save_file(tensors, str(path / "model.safetensors"))
    cfg: dict = {"architectures": ["X"]}
    if quantized:
        group = {
            "targets": ["Linear"],
            "format": "nvfp4-pack-quantized",
            "weights": {
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
                "strategy": "group",
                "symmetric": True,
            },
            "input_activations": None,
        }
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": group},
        }
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text('{"chat_template": "x"}')
    if record is not None:
        (path / "build.json").write_text(json.dumps(record))
    return path


def test_the_local_target_is_the_table_of_the_card_this_process_sees(monkeypatch: pytest.MonkeyPatch):
    """`qda plan`'s default on one card: capability 10.3 is sm103, which plans sm100's rows."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 3))
    assert report.local_target().arch == "sm103"


def test_a_capability_no_table_covers_is_no_card(monkeypatch: pytest.MonkeyPatch):
    """A card this version cannot plan for leaves the record and the default to decide, as no card does."""
    monkeypatch.setattr(report, "_card_capability", lambda: (7, 0))
    assert report.local_target() is None


def test_a_visible_card_is_planned_and_the_recorded_target_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without --target the card here is judged, whatever the checkpoint was built for."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 0))
    d = _checkpoint(tmp_path / "nvfp4", {"target": "sm90@0.33.0"})
    rep = inspect(str(d))
    assert rep["target"] == "sm100@0.33.0"
    assert rep["reasons"] == [] and rep["kernels"] == inspect(str(d), "sm100")["kernels"]
    assert rep["notes"] == ["the checkpoint was built for sm90@0.33.0 (build.json); this card is sm100"]
    assert "  note: the checkpoint was built for sm90@0.33.0 (build.json); this card is sm100" in (
        _text(rep).splitlines()
    )


def test_a_visible_card_with_the_recorded_target_adds_no_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A checkpoint built for the card it is inspected on is checked against build.json's target verbatim."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 0))
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", {"target": "sm100@0.33.0"})))
    assert rep["target"] == "sm100@0.33.0" and rep["reasons"] == [] and rep["notes"] == []


def test_a_record_on_another_vllm_version_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The card is the same but the rows were read on another vLLM: the note still names the record's target."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 0))
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", {"target": "sm100@0.27.1"})))
    assert rep["target"] == "sm100@0.33.0"
    assert rep["notes"] == ["the checkpoint was built for sm100@0.27.1 (build.json); this card is sm100"]


def test_an_explicit_target_wins_over_the_card_and_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--target` names the card to check for, whatever this card and the build recorded."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 0))
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", {"target": "sm100@0.33.0"})), "sm90")
    assert rep["target"] == "sm90@0.33.0" and rep["reasons"]
    assert not any("build.json" in n or "card" in n for n in rep["notes"])


def test_no_card_plans_the_recorded_target(tmp_path: Path):
    """On a host with no card the checkpoint is planned on the card it was built for, not the default."""
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", {"target": "sm100@0.33.0"})))
    assert rep["target"] == "sm100@0.33.0" and rep["reasons"] == []
    assert rep["notes"] == ["target sm100@0.33.0, as build.json records it"]
    assert "card" not in rep


@pytest.mark.parametrize(
    "record", [None, {}, {"format": "w4a4-nvfp4"}, {"target": 100}, {"target": ["sm90"]}]
)
def test_no_card_and_no_recorded_target_name_takes_the_default(tmp_path: Path, record: dict | None):
    """Another tool's checkpoint, or a record whose target is absent or not a name, keeps today's default."""
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", record)))
    assert rep["target"] == "sm90@0.33.0"
    assert not any(n.startswith("target ") for n in rep["notes"])


def test_a_recorded_target_this_version_cannot_parse_reports_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A newer QDA's or another tool's spelling is a note, not an exit 2 with no report."""
    d = _checkpoint(tmp_path / "bf16", {"target": "cuda"}, quantized=False)
    assert main(["inspect", str(d)]) == 0
    out, err = capsys.readouterr()
    assert (
        "  note: build.json records target 'cuda', which this version cannot plan against; "
        "checked against the default sm90@0.33.0" in out.splitlines()
    )
    assert "servable as is on sm90@0.33.0" in out and err == ""


def test_a_record_this_version_cannot_parse_is_no_note_when_a_card_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The report is about the card here, so a record this version cannot read changes nothing in it."""
    monkeypatch.setattr(report, "_card_capability", lambda: (9, 0))
    rep = inspect(str(_checkpoint(tmp_path / "bf16", {"target": "cuda"}, quantized=False)))
    assert rep["target"] == "sm90@0.33.0" and rep["notes"] == []


def test_a_recorded_kernel_suffix_this_version_drops_stays_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """`:machete` pins a kernel today and may be refused once Machete is gone: either way a report, exit 0."""
    d = _checkpoint(tmp_path / "bf16", {"target": "sm90@0.33.0:machete"}, quantized=False)
    assert main(["inspect", str(d)]) == 0
    out, err = capsys.readouterr()
    assert "build.json" in out and "servable as is on sm90@0.33.0" in out and err == ""


def test_the_serve_check_says_which_card_the_engine_loads_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The verdict may be about the --target's card while --serve loaded the engine here, on this card."""
    monkeypatch.setattr(report, "_card_capability", lambda: (10, 0))
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", {"target": "sm100@0.33.0"})), "sm90")
    rep["serve"] = {"ok": True, "error": None, "unconsumed": [], "dropped": []}
    assert "  serve check on this card (sm100): vLLM read every tensor" in _text(rep).splitlines()


def test_the_serve_check_names_no_card_when_this_process_sees_none(tmp_path: Path):
    """No card here: the serve check ran nowhere, so its line does not claim a card."""
    rep = inspect(str(_checkpoint(tmp_path / "bf16", {"target": "sm90@0.33.0"}, quantized=False)))
    rep["serve"] = {"ok": True, "error": None, "unconsumed": [], "dropped": []}
    assert "  serve check: vLLM read every tensor" in _text(rep).splitlines()


def test_the_published_checkpoint_is_planned_on_the_card_its_record_names(tmp_path: Path):
    """The real build.json of qda-lab/QDA-Qwen3.5-9B-W4A4 (copied, no weights) plans its NVFP4 groups on sm100."""
    record = json.loads((DATA / "qda-qwen3.5-9b-w4a4-build.json").read_text())
    rep = inspect(str(_checkpoint(tmp_path / "nvfp4", record)))
    assert rep["target"] == "sm100@0.33.0" and rep["reasons"] == []
    assert "FlashInferCuteDslNvFp4LinearKernel" in rep["kernels"]["group_0/dense"]
    assert "  note: target sm100@0.33.0, as build.json records it" in _text(rep).splitlines()


def test_a_machete_record_is_read_like_the_no_card_branch_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`:machete` is dropped on the same sm90 table, so a card here that matches it has no note to make."""
    monkeypatch.setattr(report, "_card_capability", lambda: (9, 0))
    rep = inspect(str(_checkpoint(tmp_path / "w4a16", {"target": "sm90@0.33.0:machete"})))
    assert rep["target"] == "sm90@0.33.0" and rep["notes"] == []


def test_a_w4a4_build_inspected_on_an_sm90_card_says_it_was_built_for_sm100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The published W4A4 record has no sm90 row: exit 2, and the note says the build was for sm100."""
    monkeypatch.setattr(report, "_card_capability", lambda: (9, 0))
    record = json.loads((DATA / "qda-qwen3.5-9b-w4a4-build.json").read_text())
    assert record["format"] == "w4a4-nvfp4" and record["target"] == "sm100@0.33.0"
    d = _checkpoint(tmp_path / "nvfp4", record)
    assert main(["inspect", str(d)]) == 2
    out, _ = capsys.readouterr()
    assert (
        "  note: the checkpoint was built for sm100@0.33.0 (build.json); this card is sm90"
        in out.splitlines()
    )
    assert "not servable as is on sm90@0.33.0" in out


def test_a_record_with_another_backend_names_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same card, a recorded backend pin: the note shows the pin, so the two sides do not read the same."""
    monkeypatch.setattr(report, "_card_capability", lambda: (9, 0))
    rep = inspect(str(_checkpoint(tmp_path / "w4a16", {"target": "sm90@0.33.0:marlin"})))
    assert rep["notes"] == ["the checkpoint was built for sm90@0.33.0:marlin (build.json); this card is sm90"]
