"""The llama.cpp checkout a GGUF build converts with: one pinned commit, named by the config or the environment.

``convert_hf_to_gguf.py`` is llama.cpp's own script; run as a script it imports the ``gguf-py`` beside it, so the
converter and the GGUF writer that fills its template come from one checkout at one commit. qda does not vendor
either: ``exporter: {gguf: {llama_cpp: PATH}}`` names a checkout, or ``QDA_LLAMA_CPP`` does for a machine, and
nothing else is needed (conversion is pure Python, no C build). The pinned commit is the one the exporter's byte
parity and its quality numbers are stated against; a build with another commit still runs and records that commit,
and ``qda doctor`` names the difference.

A build that is given ``template:`` never runs the converter, so it needs no checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..step import Refused

#: The llama.cpp commit the exporter's parity and quality numbers are stated against.
PINNED = "9e2a04a439edc49f08c5acdafefae19dfa25a658"
#: Where the checkout comes from.
REPO = "https://github.com/ggml-org/llama.cpp"
#: The environment variable that names the checkout when ``llama_cpp`` is not set.
ENV = "QDA_LLAMA_CPP"

__all__ = ["ENV", "PINNED", "REPO", "checkout_command", "commit_of", "converter", "locate", "model_file"]


def locate(llama_cpp: str | Path | None) -> Path | None:
    """The checkout to convert with: the ``llama_cpp`` setting, else ``$QDA_LLAMA_CPP``, else None."""
    named = llama_cpp or os.environ.get(ENV)
    return Path(named).expanduser() if named else None


def converter(root: str | Path) -> Path:
    """The checkout's ``convert_hf_to_gguf.py``.

    Raises:
        Refused: `root` is not a llama.cpp checkout.
    """
    script = Path(root) / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise Refused(f"gguf: {root} is not a llama.cpp checkout (no convert_hf_to_gguf.py)")
    return script


def commit_of(root: str | Path) -> str:
    """The commit of the llama.cpp checkout.

    Raises:
        Refused: not a checkout with ``convert_hf_to_gguf.py``, or no readable ``git`` history.
    """
    converter(root)
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError as e:  # no git on this machine (a compute node): read the checkout's own files
        if commit := _head_from_files(Path(root)):
            return commit
        raise Refused(f"gguf: cannot run git for {root}: {e}") from e
    if r.returncode:  # git's stderr can be several lines; the doctor line is one
        why = (r.stderr.strip().splitlines() or [f"git exited {r.returncode}"])[0]
        raise Refused(f"gguf: cannot read the llama.cpp commit of {root}: {why}")
    return r.stdout.strip()


def _head_from_files(root: Path) -> str | None:
    """The commit ``HEAD`` names, read from the checkout's ``.git`` files without git, or None."""
    git = root / ".git"
    try:
        if git.is_file():  # a worktree or submodule: ``gitdir: PATH``
            git = (root / git.read_text().partition("gitdir:")[2].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        common = (git / (git / "commondir").read_text().strip()) if (git / "commondir").is_file() else git
        if not head.startswith("ref:"):
            return head if _is_sha(head) else None
        ref = head[4:].strip()
        for d in (git, common):
            if (d / ref).is_file():
                sha = (d / ref).read_text().strip()
                return sha if _is_sha(sha) else None
        for line in (common / "packed-refs").read_text().splitlines():
            sha, _, name = line.partition(" ")
            if name == ref and _is_sha(sha):
                return sha
    except OSError:
        return None
    return None


def _is_sha(text: str) -> bool:
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def checkout_command() -> str:
    """The one line that makes a checkout at the pinned commit, in the current directory."""
    return f"git clone {REPO} llama.cpp && git -C llama.cpp checkout {PINNED}"


def model_file(checkpoint: str | Path) -> Path:
    """The single GGUF file a llama.cpp checkpoint stands for.

    A file names itself, a frozen suite directory stands for its one ``.gguf``; a directory holding
    zero or more than one ``.gguf`` is refused, the message naming the count.
    """
    path = Path(checkpoint)
    if not path.is_dir():
        return path
    files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".gguf")
    if len(files) != 1:
        raise Refused(f"the checkpoint holds {len(files)} .gguf files; a llama.cpp checkpoint is exactly one")
    return files[0]
