"""The llama.cpp checkout a GGUF build converts with: located by setting or environment, read at its commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from qda_quant.gguf.llama_cpp import (
    ENV,
    PINNED,
    REPO,
    checkout_command,
    commit_of,
    converter,
    locate,
)
from qda_quant.step import Refused


def _checkout(root: Path) -> Path:
    """A checkout-shaped directory at a commit of its own (the converter itself is not run)."""
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("")
    ident = ["-c", "user.email=t@e", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), *ident, "commit", "-qm", "x", "--allow-empty"], check=True)
    return root


def test_the_setting_wins_over_the_environment(tmp_path: Path, monkeypatch) -> None:
    """``llama_cpp`` is a build's own checkout; ``$QDA_LLAMA_CPP`` is the machine's; neither is no checkout."""
    monkeypatch.delenv(ENV, raising=False)
    assert locate(None) is None
    monkeypatch.setenv(ENV, str(tmp_path))
    assert locate(None) == tmp_path
    assert locate(tmp_path / "named") == tmp_path / "named"


def test_the_converter_and_its_commit_are_read_from_the_checkout(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "llama.cpp")
    assert converter(root) == root / "convert_hf_to_gguf.py"
    expected = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert commit_of(root) == expected


@pytest.mark.parametrize("head", ["branch", "packed", "detached", "worktree"])
def test_with_no_git_the_commit_is_read_from_the_checkout_files(tmp_path, monkeypatch, head) -> None:
    """Compute nodes may have no git; the commit still reads from ``.git`` (HEAD, loose or packed refs)."""
    root = _checkout(tmp_path / "llama.cpp")
    git = ["git", "-C", str(root)]
    expected = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head == "packed":
        subprocess.run([*git, "pack-refs", "--all"], check=True)
    elif head == "detached":
        subprocess.run([*git, "checkout", "-q", "--detach"], check=True)
    elif head == "worktree":
        subprocess.run([*git, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "w"], check=True)
        root = tmp_path / "wt"
        (root / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    assert commit_of(root) == expected


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Refused, match="no convert_hf_to_gguf.py"):
        converter(tmp_path)
    with pytest.raises(Refused, match="no convert_hf_to_gguf.py"):
        commit_of(tmp_path)
    root = tmp_path / "loose"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("")
    with pytest.raises(Refused, match="cannot read the llama.cpp commit"):
        commit_of(root)


def test_the_fix_is_one_line_at_the_pinned_commit() -> None:
    line = checkout_command()
    assert REPO in line and PINNED in line and "\n" not in line
    assert len(PINNED) == 40
