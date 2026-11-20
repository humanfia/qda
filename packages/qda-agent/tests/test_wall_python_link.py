"""The agent wall starts a venv made by `uv venv --managed-python` (install.md).

Such a venv's `bin/python` runs `~/.local/share/uv/python/cpython-3.12-<platform>/bin/python3.12`, where the
`cpython-3.12-...` directory is a link to the patch release's. The wall shows the patch release's directory
(`sys.base_prefix`) and hides the rest of the home, so it must make that link again, or every turn at
`isolation.level: sandbox` or `strict` stops with `bwrap: execvp .../bin/python: No such file or directory`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from qda_agent.isolation import wall


def uv_venv(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A home with uv's managed Python (a version link to the patch release) and a venv outside it."""
    home = tmp_path / "home"
    base = home / ".local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3.12").write_text("")
    minor = base.parent / "cpython-3.12-linux-x86_64-gnu"
    minor.symlink_to(base)
    venv = tmp_path / "work/.venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(minor / "bin/python3.12")
    return home, minor, venv / "bin/python"


def test_the_version_link_of_a_uv_python_is_made_again(tmp_path):
    home, minor, exe = uv_venv(tmp_path)
    links = wall.interpreter_links(str(exe), home, str(minor))
    real_minor = Path(os.path.realpath(minor.parent)) / minor.name
    assert links == {str(real_minor): str(minor.readlink())}


def test_an_interpreter_with_no_link_under_the_home_needs_none(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    exe = tmp_path / "opt/python/bin/python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert wall.interpreter_links(str(exe), home, str(tmp_path / "opt/python")) == {}


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")
def test_the_venv_python_starts_in_an_empty_root_with_the_link(tmp_path):
    """The wall's own argv (`bwrap_argv(minimal=True)`) with the link starts the venv's `bin/python`."""
    home, _minor, exe = uv_venv(tmp_path)
    base = Path(os.path.realpath(_minor))
    (base / "bin/python3.12").unlink()
    (base / "bin/python3.12").symlink_to("/bin/true")
    read = ["/usr", "/bin", "/lib", "/lib64", str(base), str(exe.parent.parent)]
    argv = wall.bwrap_argv(
        read=[p for p in read if Path(p).exists()],
        write=[],
        hide=[],
        minimal=True,
        hide_early=[str(home)],
        links=wall.interpreter_links(str(exe), home, str(_minor)),
    )
    probe = subprocess.run([*argv, str(exe)], capture_output=True, text=True, check=False)
    if "Operation not permitted" in probe.stderr or "No permissions" in probe.stderr:
        pytest.skip(f"bwrap cannot start here: {probe.stderr.strip()}")
    assert probe.returncode == 0, probe.stderr


def venv_in_linked_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """uv's managed Python in the home, and the venv in `~/proj/.venv` where `~/proj -> /scratch/proj`."""
    home, minor, _ = uv_venv(tmp_path)
    scratch = tmp_path / "scratch/proj"
    (scratch / ".venv/bin").mkdir(parents=True)
    (scratch / ".venv/bin/python").symlink_to(minor / "bin/python3.12")
    (home / "proj").symlink_to(scratch)
    return home, minor, home / "proj", home / "proj/.venv/bin/python"


def test_a_link_to_a_project_directory_is_not_made_again(tmp_path):
    home, minor, _proj, exe = venv_in_linked_project(tmp_path)
    real_minor = Path(os.path.realpath(minor.parent)) / minor.name
    assert wall.interpreter_links(str(exe), home, str(minor)) == {str(real_minor): str(minor.readlink())}


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")
def test_the_venv_python_starts_from_a_linked_project_directory(tmp_path):
    """With the venv bound at `~/proj/.venv`, a made-again `~/proj` link stops bwrap; the wall's links start it."""
    home, minor, proj, exe = venv_in_linked_project(tmp_path)
    base = Path(os.path.realpath(minor))
    (base / "bin/python3.12").unlink()
    (base / "bin/python3.12").symlink_to("/bin/true")
    read = ["/usr", "/bin", "/lib", "/lib64", str(base), str(exe.parent.parent)]

    def run(links: dict[str, str]) -> subprocess.CompletedProcess[str]:
        argv = wall.bwrap_argv(
            read=[p for p in read if Path(p).exists()],
            write=[],
            hide=[],
            minimal=True,
            hide_early=[str(home)],
            links=links,
        )
        return subprocess.run([*argv, str(exe)], capture_output=True, text=True, check=False)

    links = wall.interpreter_links(str(exe), home, str(minor))
    probe = run(links)
    if "Operation not permitted" in probe.stderr or "No permissions" in probe.stderr:
        pytest.skip(f"bwrap cannot start here: {probe.stderr.strip()}")
    assert probe.returncode == 0, probe.stderr
    real_proj = str(Path(os.path.realpath(home)) / proj.name)
    every = run({**links, real_proj: str(proj.readlink())})
    assert every.returncode != 0 and "Can't mkdir" in every.stderr, every.stderr


def test_a_link_to_the_directory_that_holds_the_python_is_made_again(tmp_path):
    """`~/pythons -> /scratch/pythons` with the interpreter in `/scratch/pythons/cpython-3.12`."""
    home = tmp_path / "home"
    home.mkdir()
    base = tmp_path / "scratch/pythons/cpython-3.12"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3.12").write_text("")
    (home / "pythons").symlink_to(base.parent)
    venv = tmp_path / "work/.venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(home / "pythons/cpython-3.12/bin/python3.12")
    links = wall.interpreter_links(str(venv / "bin/python"), home, str(base))
    assert links == {str(Path(os.path.realpath(home)) / "pythons"): str(base.parent)}
