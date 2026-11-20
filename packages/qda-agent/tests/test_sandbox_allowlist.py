"""The empty-root sandbox: the host-root argv unchanged, and only pinned mounts inside the allowlist."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from qda_agent.livecodebench_sandbox import SYSTEM_TREES, interpreter, interpreter_trees, sandbox_argv
from qda_agent.sched import sandbox
from qda_agent.sched.sandbox import Pinned, bwrap_allowlist_argv, pin_sources

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="the sandbox is bubblewrap")


def _has(argv: list[str], part: list[str]) -> bool:
    """Whether ``part`` appears in ``argv`` as consecutive arguments."""
    return any(argv[i : i + len(part)] == part for i in range(len(argv)))


def test_an_existing_callers_bwrap_argv_is_unchanged(tmp_path, monkeypatch):
    """The literal argv of a host-root job: the shared bind helper may not move a byte of it."""
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))  # the host's /proc form is another test's
    monkeypatch.setattr(sandbox, "SYSTEM", ("/usr",))
    monkeypatch.setattr(sandbox, "python_roots", lambda: [Path("/py/lib")])
    work, mine = tmp_path / "w", tmp_path / "mine"
    work.mkdir()
    mine.mkdir()
    argv = sandbox.bwrap_argv(
        ["true"],
        str(work),
        writable=[work],
        readable=[Path("/data")],
        devices=[Path("/dev/nvidia0")],
        env={"A": "b", "PATH": "/bin"},
        binds={work / "x": mine},
    )
    assert argv == [
        "bwrap",
        "--die-with-parent",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/py/lib",
        "/py/lib",
        "--ro-bind",
        "/data",
        "/data",
        "--bind",
        str(work),
        str(work),
        "--ro-bind",
        str(mine),
        str(work / "x"),
        "--dev-bind",
        "/dev/nvidia0",
        "/dev/nvidia0",
        "--clearenv",
        "--setenv",
        "A",
        "b",
        "--setenv",
        "PATH",
        "/bin",
        "--chdir",
        str(work),
        "--",
        "true",
    ]


def test_an_existing_callers_pinned_bwrap_argv_is_unchanged(tmp_path, monkeypatch):
    """The literal argv of a production job: pinned sources by their descriptor numbers, a path bind for it."""
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))  # the host's /proc form is another test's
    monkeypatch.setattr(sandbox, "SYSTEM", ("/usr",))
    monkeypatch.setattr(sandbox, "python_roots", lambda: [Path("/py/lib")])
    work = tmp_path / "w"
    work.mkdir()
    argv = sandbox.bwrap_argv(
        ["true"],
        str(work),
        writable=[work],
        readable=[Path("/data")],
        pinned={
            Path("/py/lib"): Pinned(Path("/py/lib"), 901),
            Path("/data"): Pinned(Path("/data"), 900),
        },
        devices=[Path("/dev/nvidia0")],
        env={"A": "b", "PATH": "/bin"},
    )
    assert argv == [
        "bwrap",
        "--die-with-parent",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind-fd",
        "901",
        "/py/lib",
        "--ro-bind-fd",
        "900",
        "/data",
        "--bind",
        str(work),
        str(work),
        "--dev-bind",
        "/dev/nvidia0",
        "/dev/nvidia0",
        "--clearenv",
        "--setenv",
        "A",
        "b",
        "--setenv",
        "PATH",
        "/bin",
        "--chdir",
        str(work),
        "--",
        "true",
    ]


def test_the_allowlist_sandbox_mounts_only_pinned_sources(tmp_path):
    work, numpy, grader = tmp_path / "case-1", tmp_path / "numpy", tmp_path / "grader"
    for path in (work, numpy, grader):
        path.mkdir()
    mounts = {
        Path("/usr"): Path("/usr"),
        Path("/deps/numpy"): numpy,
        Path("/grader"): grader,
        Path("/scratch"): work,
    }
    pinned = pin_sources([], mounts)
    try:
        argv = bwrap_allowlist_argv(
            ["true"],
            "/scratch",
            mounts=mounts,
            writable=[Path("/scratch")],
            pinned=pinned,
            env={"PATH": "/usr/bin"},
            status_fd=5,
            shm_bytes=1 << 20,
        )
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
    assert argv[:9] == [
        "bwrap",
        "--die-with-parent",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--tmpfs",
        "/",
    ]
    assert argv[9:12] == ["--ro-bind-fd", str(pinned[Path("/usr")].fd), "/usr"]
    path_binds = [(f, argv[i + 1]) for i, f in enumerate(argv[:-1]) if f in ("--ro-bind", "--bind")]
    # Every source the caller names comes from a descriptor; the grader names its /proc mode, and the default
    # is a fresh procfs (`proc: private`), so no host path is bound and no user namespace is asked for.
    assert path_binds == []
    assert "--unshare-user" not in argv
    assert _has(argv, ["--bind-fd", str(pinned[Path("/scratch")].fd), "/scratch"])
    assert _has(
        argv,
        ["--dev", "/dev", "--remount-ro", "/dev", "--size", str(1 << 20), "--tmpfs", "/dev/shm"],
    )
    assert _has(argv, ["--json-status-fd", "5"])
    assert _has(argv, ["--clearenv", "--setenv", "PATH", "/usr/bin"])
    assert argv.index("--remount-ro") > argv.index("--bind-fd")  # the writable mount survives the root
    assert argv[-4:] == ["--chdir", "/scratch", "--", "true"]


def test_the_graders_host_readonly_mode_is_the_hosts_proc(tmp_path):
    """`proc: host-readonly` (the grader's fallback where a fresh procfs is refused) is the same helper's answer."""
    work = tmp_path / "scratch"
    work.mkdir()
    mounts = {Path("/scratch"): work}
    pinned = pin_sources([], mounts)
    try:
        argv = bwrap_allowlist_argv(
            ["true"],
            "/scratch",
            mounts=mounts,
            writable=[Path("/scratch")],
            pinned=pinned,
            env={"PATH": "/usr/bin"},
            proc="host-readonly",
        )
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
    path_binds = [(f, argv[i + 1]) for i, f in enumerate(argv[:-1]) if f in ("--ro-bind", "--bind")]
    assert path_binds == [("--ro-bind", "/proc")]
    assert "--unshare-user" not in argv
    with pytest.raises(ValueError, match="proc"):
        bwrap_allowlist_argv(
            ["true"],
            "/scratch",
            mounts=mounts,
            writable=[Path("/scratch")],
            pinned={},
            env={},
            proc="host",
        )


def test_an_unpinned_mount_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not opened by pin_sources"):
        bwrap_allowlist_argv(
            ["true"],
            "/scratch",
            mounts={Path("/usr"): Path("/usr")},
            writable=[],
            pinned={},
            env={"PATH": "/usr/bin"},
        )


def test_proc_is_private_by_default_and_host_readonly_binds_it_readonly(tmp_path):
    work = tmp_path / "case"
    work.mkdir()
    mounts = {Path("/scratch"): work}
    pinned = pin_sources([], mounts)
    try:
        common = {"mounts": mounts, "writable": [Path("/scratch")], "pinned": pinned, "env": {}}
        default = bwrap_allowlist_argv(["true"], "/scratch", **common)
        host = bwrap_allowlist_argv(["true"], "/scratch", proc="host-readonly", **common)
        with pytest.raises(ValueError, match="not 'private' or 'host-readonly'"):
            bwrap_allowlist_argv(["true"], "/scratch", proc="shared", **common)
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
    assert _has(default, ["--proc", "/proc"])
    assert _has(host, ["--ro-bind", "/proc", "/proc"])  # read-only: the host tree stays out of reach
    assert "--proc" not in host


def test_a_writable_destination_that_is_not_mounted_is_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    mounts = {Path("/scratch"): source}
    pinned = pin_sources([], mounts)
    try:
        with pytest.raises(ValueError, match="writable but not in mounts"):
            bwrap_allowlist_argv(
                ["true"],
                "/scratch",
                mounts=mounts,
                writable=[Path("/scratch"), Path("/elsewhere")],
                pinned=pinned,
                env={},
            )
    finally:
        for pin in pinned.values():
            os.close(pin.fd)


def test_a_case_directory_that_resolves_into_the_run_directory_is_refused(tmp_path):
    run = tmp_path / "run"
    (run / "judge" / "case").mkdir(parents=True)
    scratch = run / "out" / "livecodebench" / "scratch"
    scratch.mkdir(parents=True)
    link = scratch / "case-link"
    link.symlink_to(run / "judge")
    for where in (link, run / "judge" / "case"):
        with pytest.raises(ValueError, match="resolves into run_dir"):
            sandbox_argv(where, 3, run_dir=run, host_written=[scratch])
    where = scratch / "case-ok"
    where.mkdir()
    _, pinned = sandbox_argv(where, 3, run_dir=run, host_written=[scratch])
    for pin in pinned.values():
        os.close(pin.fd)


@needs_bwrap
def test_the_scratch_stays_writable_and_the_root_does_not(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    mounts = {Path(tree): Path(tree) for tree in ("/usr", "/lib", "/lib64") if Path(tree).exists()}
    mounts.update({tree: tree for tree in interpreter_trees()})  # a Python installed outside /usr
    mounts[Path("/scratch")] = scratch
    code = (
        "import os\n"
        "ok = os.access('/scratch', os.W_OK)\n"
        "for path in ('/x', '/dev/x'):\n"
        "    try:\n"
        "        open(path, 'w').write('x')\n"
        "    except OSError:\n"
        "        pass\n"
        "    else:\n"
        "        ok = False\n"
        "open('/dev/shm/x', 'w').write('x')\n"
        "print('yes' if ok else 'no')\n"
    )
    pinned = pin_sources([], mounts)
    try:
        argv = bwrap_allowlist_argv(
            [str(interpreter()), "-s", "-S", "-c", code],
            "/scratch",
            mounts=mounts,
            writable=[Path("/scratch")],
            pinned=pinned,
            env={"PATH": "/usr/bin"},
            shm_bytes=1 << 20,
        )
        done = subprocess.run(
            argv, capture_output=True, text=True, pass_fds=tuple(p.fd for p in pinned.values())
        )
    finally:
        for pin in pinned.values():
            os.close(pin.fd)
    assert done.stdout.strip() == "yes", done.stderr


def _managed_python(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A Python installed the way uv's ``--managed-python`` lays it out under a home directory.

    Returns the minor-version link the venv's ``home`` names, the installation it points to, and the home.
    """
    home = tmp_path / "home"
    real = home / ".local" / "share" / "uv" / "python" / "cpython-3.12.16-linux-x86_64-gnu"
    (real / "bin").mkdir(parents=True)
    (real / "lib" / "python3.12").mkdir(parents=True)
    (real / "bin" / "python3.12").write_text("")
    link = real.parent / "cpython-3.12-linux-x86_64-gnu"
    link.symlink_to(real.name)
    return link, real, home


def _pinned_argv(where: Path) -> tuple[list[str], dict[Path, Pinned]]:
    argv, pinned = sandbox_argv(where, 3)
    for pin in pinned.values():
        os.close(pin.fd)
    return argv, pinned


def test_an_interpreter_outside_the_system_trees_is_mounted_read_only(tmp_path, monkeypatch):
    """A managed Python (uv, pyenv, conda) is exec'd from its own installation, so that is mounted, read-only."""
    link, real, home = _managed_python(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "_base_executable", str(link / "bin" / "python3.12"))
    monkeypatch.setattr(sys, "base_prefix", str(link))
    where = tmp_path / "case"
    where.mkdir()
    argv, pinned = _pinned_argv(where)
    assert {link, real} <= set(pinned)
    for tree in (link, real):
        assert _has(argv, ["--ro-bind-fd", str(pinned[tree].fd), str(tree)])
        assert not _has(argv, ["--bind-fd", str(pinned[tree].fd), str(tree)])
    # The installation only: the home directory around it, its parent and the venv are not mounted.
    assert home not in pinned and real.parent not in pinned and Path(sys.prefix) not in pinned
    assert str(link / "bin" / "python3.12") in argv


def test_a_system_interpreter_adds_no_mount(tmp_path, monkeypatch):
    """Under ``/usr`` (or spelled ``/bin``, a link into it) the system trees already hold the interpreter."""
    where = tmp_path / "case"
    where.mkdir()
    system = {Path(tree) for tree in SYSTEM_TREES if Path(tree).exists()}
    for executable in ("/usr/bin/python3.12", "/usr/local/bin/python3.12", "/bin/python3.12"):
        monkeypatch.setattr(sys, "_base_executable", executable)
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        _, pinned = _pinned_argv(where)
        assert {p for p in pinned if not str(p).startswith(("/deps/", "/grader", "/scratch"))} == system


def test_an_interpreter_installed_as_the_home_directory_is_refused(tmp_path, monkeypatch):
    """A Python in ``~/bin`` has the home directory as its installation: mounting it would expose all of home."""
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "python3.12").write_text("")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "_base_executable", str(home / "bin" / "python3.12"))
    monkeypatch.setattr(sys, "base_prefix", str(home))
    where = tmp_path / "case"
    where.mkdir()
    with pytest.raises(RuntimeError, match="is the home directory or contains it"):
        sandbox_argv(where, 3)
