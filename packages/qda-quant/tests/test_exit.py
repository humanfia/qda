"""The exit-status contract: 2 is the input, 3 is the machine, 130 is an interruption, and a refusal says why.

`qda_quant.exit` holds the codes; `Refused` carries the one its raise site knows (`code=`) and the fix that
site knows (`fix=`), and the CLI prints both as the contract's two lines. The doc page for users is
`docs/troubleshooting.md`'s exit-status table, which `test_docs.py` checks against these codes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from qda_quant.cli import main
from qda_quant.config import BuildConfig
from qda_quant.exit import ENVIRONMENT, ERROR, INTERRUPTED, USAGE
from qda_quant.step import Refused
from transformers import PreTrainedModel

from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    """A saved tiny Qwen3, a fingerprinted calibration file and a config that plans on a CPU."""
    cast(PreTrainedModel, tiny_qwen3(layers=2, hidden=128, inter=256)).save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(4)], sha
    )
    (tmp_path / "build.yaml").write_text(
        f"model: {tmp_path / 'm'}\n"
        "format: w4a16\n"
        "devices: [cpu]\n"
        "steps: [{rtn: {}}]\n"
        f"calib: {{file: {{path: {tmp_path / 'c.jsonl'}, samples: 4, seqlen: 16}}}}\n"
        f"out: {tmp_path / 'out'}\n"
    )
    return tmp_path


def test_a_refused_input_is_2_and_prints_the_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A checkpoint this command cannot read is the input, and the first line says which."""
    path = tmp_path / "not-here"  # beside tmp_path, so the id is not looked up on the hub
    assert main(["inspect", str(path)]) == USAGE
    assert capsys.readouterr().err == (
        f"error: {path}: not a checkpoint directory, a .gguf file or an HF repo id\n"
    )


def test_a_checkpoint_that_does_not_serve_is_2_with_the_report_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`qda inspect` prints its report, then refuses: the verdict is the contract's line, not a bare 2."""
    cast(PreTrainedModel, tiny_qwen3(layers=1)).save_pretrained(tmp_path / "m")  # no tokenizer files
    capsys.readouterr()  # the writer's progress bar
    assert main(["inspect", str(tmp_path / "m")]) == USAGE
    said = capsys.readouterr()
    assert "not servable as is on" in said.out and "missing: tokenizer" in said.out
    (line,) = said.err.splitlines()
    assert line.startswith(f"error: inspect: {tmp_path / 'm'}: not servable as is on ")
    assert "missing tokenizer" in line


def test_a_machine_that_cannot_hold_it_is_3(
    tiny: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A card this process cannot see is the machine: the same command works once the host gives it the card."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # this process was given no card at all
    assert main(["plan", str(tiny / "build.yaml"), "--set", "devices=[0]"]) == ENVIRONMENT
    err = capsys.readouterr().err
    assert err.startswith("error: cuda:0 is not visible: the process sees 0 card(s)")
    assert "fix:   " in err  # the change that unblocks it, not just the symptom


def test_a_card_too_full_to_read_is_3_in_two_lines(
    tiny: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A card another job has filled cannot even report its free memory; the plan names it, not a traceback."""

    def full(device: object = None) -> tuple[int, int]:
        raise torch.AcceleratorError("CUDA error: out of memory\nCUDA kernel errors might be reported later")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))  # no context needed
    monkeypatch.setattr(torch.cuda, "mem_get_info", full)
    assert main(["plan", str(tiny / "build.yaml"), "--set", "devices=[0]"]) == ENVIRONMENT
    err = capsys.readouterr().err
    assert "Traceback" not in err
    error, fix = err.splitlines()  # the error's first line only
    assert error.startswith("error: cuda:0 ") and "CUDA error: out of memory" in error
    assert fix.startswith("fix:   ") and 'CUDA_VISIBLE_DEVICES=""' in fix
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)  # the fix's CPU plan: no card to read
    assert main(["plan", str(tiny / "build.yaml")]) == 0


def test_the_code_a_refusal_carries_is_the_code_the_command_exits_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raise site knows whether it is the input or the machine; the command only prints it."""

    def refuse(config: str, sets: object = ()) -> BuildConfig:
        raise Refused("a card that serves w4a8 is Hopper", "--format w4a16", code=ENVIRONMENT)

    monkeypatch.setattr(BuildConfig, "from_file", refuse)
    assert main(["plan", "run.yaml"]) == ENVIRONMENT


def test_an_unexpected_error_keeps_python_s_own_status(tmp_path: Path) -> None:
    """A crash is not a verdict on the input: the command leaves Python's own status for an uncaught error."""
    junk = tmp_path / "not-a.gguf"
    junk.write_text("not a gguf at all\n")  # its header is read as a struct, which is no `Refused`
    done = subprocess.run(
        [sys.executable, "-m", "qda_quant.cli", "sensitivity", "plan", str(junk)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == ERROR
    assert "Traceback (most recent call last)" in done.stderr  # a crash is the one case that prints frames


def test_ctrl_c_is_130_and_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C unwinds like any error and leaves the contract's status, with one line rather than frames."""

    def interrupt(config: str, sets: object = ()) -> BuildConfig:
        raise KeyboardInterrupt

    monkeypatch.setattr(BuildConfig, "from_file", interrupt)
    assert main(["plan", "run.yaml"]) == INTERRUPTED
    assert capsys.readouterr().err == "interrupted\n"
