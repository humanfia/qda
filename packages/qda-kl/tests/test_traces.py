"""`qda traces` refuses what the user gave before it reads a model: the contract's line, and status 2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from qda_kl import traces
from qda_quant.exit import ENVIRONMENT
from qda_quant.step import Refused


def _traces(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``qda traces`` the way the shell does, so the status is the process's, not a return value."""
    return subprocess.run(
        [sys.executable, "-m", "qda_kl.traces", "org/model", *args], capture_output=True, text=True
    )


@pytest.mark.parametrize("make", [False, True], ids=["missing", "directory"])
def test_a_prompt_file_that_cannot_be_read_is_a_refusal(tmp_path: Path, make: bool) -> None:
    """The path the user gave is the input: no traceback, and the status a script can act on."""
    prompts = tmp_path / "prompts.jsonl"
    if make:
        prompts.mkdir()  # a directory in its place: readable as a path, not as a file
    done = _traces(str(prompts), str(tmp_path / "p"), "--max-tokens", "1", "--seed", "17")
    assert done.returncode == 2
    assert done.stderr.startswith(f"error: {prompts}: ")
    assert "Traceback" not in done.stderr


def test_a_rows_slice_that_is_not_a_slice_is_a_refusal(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    done = _traces(str(prompts), str(tmp_path / "p"), "--max-tokens", "1", "--seed", "17", "--rows", "abc")
    assert done.returncode == 2
    assert done.stderr.startswith("error: --rows 'abc': ") and "--rows START:STOP" in done.stderr


def test_a_missing_vllm_is_the_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine is the `vllm` extra: without it the command refuses as the machine, with the install line."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    monkeypatch.setitem(sys.modules, "vllm", None)  # `from vllm import ...` fails, as without the extra
    with pytest.raises(Refused) as refused:
        traces.write("org/model", prompts, str(tmp_path / "p"), 1, 17)
    assert refused.value.code == ENVIRONMENT
    assert refused.value.fix == 'uv pip install -c constraints.txt "./packages/qda-kl[vllm]"'
