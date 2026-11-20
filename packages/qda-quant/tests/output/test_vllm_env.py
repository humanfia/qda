"""The engine environment: where its caches are, and the host compiler and libraries it starts with."""

import errno
import os
import stat
from pathlib import Path

import pytest
from qda_quant.exit import ENVIRONMENT
from qda_quant.output import vllm_env
from qda_quant.step import Refused


def test_the_caches_default_to_one_node_local_directory_per_user(tmp_path, monkeypatch):
    """Unset, every engine cache is a directory of `cache_root`: a local disk, outside any run directory."""
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vllm_env.getpass, "getuser", lambda: "someone")
    assert vllm_env.cache_root() == Path("/tmp/qda_someone")
    root = tmp_path / "qda_someone"  # the same shape, made under tmp_path so the test leaves /tmp alone
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    resolved = vllm_env.cache_env({"OTHER": "kept"})
    assert resolved == {
        "OTHER": "kept",
        "FLASHINFER_WORKSPACE_BASE": str(root / "flashinfer"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
    }


def test_a_cache_the_user_named_is_kept_wherever_it_is(tmp_path, monkeypatch):
    """One variable set in the environment and one passed in: both are the user's choice, neither is replaced."""
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "qda_someone"
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "exported"))
    resolved = vllm_env.cache_env({"VLLM_CACHE_ROOT": str(tmp_path / "passed")})
    assert resolved["TRITON_CACHE_DIR"] == str(tmp_path / "exported")
    assert resolved["VLLM_CACHE_ROOT"] == str(tmp_path / "passed")
    assert resolved["FLASHINFER_WORKSPACE_BASE"] == str(root / "flashinfer")


def default_root(tmp_path, monkeypatch, mode: int = 0o700) -> Path:
    """A `cache_root` under `tmp_path`, with every cache variable unset so `cache_env` falls back to it."""
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "qda_someone"
    root.mkdir()
    root.chmod(mode)
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    return root


def test_a_default_root_another_user_made_is_refused(tmp_path, monkeypatch):
    """`/tmp` is shared: a root another user made first would feed the engine build output it loads and runs."""
    root = default_root(tmp_path, monkeypatch)
    monkeypatch.setattr(vllm_env.os, "getuid", lambda: root.stat().st_uid + 1)
    problem = vllm_env.cache_root_problem(root)
    assert problem is not None and "another user" in problem
    with pytest.raises(Refused) as refused:
        vllm_env.cache_env({})
    assert str(root) in str(refused.value)
    assert refused.value.fix == vllm_env.CACHE_FIX
    assert refused.value.code == ENVIRONMENT


def test_a_default_root_writable_by_group_or_others_is_refused(tmp_path, monkeypatch):
    """A group or other-writable root is writable by another user too, whatever it is owned by."""
    root = default_root(tmp_path, monkeypatch, mode=0o770)
    problem = vllm_env.cache_root_problem(root)
    assert problem is not None and "group or others" in problem
    with pytest.raises(Refused):
        vllm_env.cache_env({})


def test_a_default_root_that_is_a_symbolic_link_is_refused(tmp_path, monkeypatch):
    """A link is followed wherever it points; the directory the engine compiles into must be this user's own."""
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    root = tmp_path / "qda_someone"
    root.symlink_to(tmp_path / "elsewhere")
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    problem = vllm_env.cache_root_problem(root)
    assert problem is not None and "symbolic link" in problem
    with pytest.raises(Refused):
        vllm_env.cache_env({})


def test_this_users_private_root_and_a_root_not_made_yet_are_fine(tmp_path, monkeypatch):
    """0700 and this user's: the default; missing: the first engine start makes it as this user."""
    root = default_root(tmp_path, monkeypatch)
    assert vllm_env.cache_root_problem(root) is None
    assert vllm_env.cache_env({})["VLLM_CACHE_ROOT"] == str(root / "vllm")
    assert vllm_env.cache_root_problem(tmp_path / "not_made_yet") is None


def _inherited_setgid(parent: Path) -> int:
    """The setgid bit a directory made under `parent` takes from it (shared scratch); no other extra bit is allowed."""
    return stat.S_ISGID if parent.stat().st_mode & stat.S_ISGID else 0


def test_a_new_default_root_is_made_private_under_a_permissive_umask(tmp_path, monkeypatch):
    """The root itself is 0700: made as a parent with `mkdir(parents=True)` it would be 0755 under umask 022."""
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "qda_someone"
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    previous = os.umask(0o022)
    try:
        vllm_env.cache_env({})
    finally:
        os.umask(previous)
    private = 0o700 | _inherited_setgid(tmp_path)
    assert stat.S_IMODE(root.stat().st_mode) == private
    for leaf in vllm_env.CACHE_DIRS.values():
        assert stat.S_IMODE((root / leaf).stat().st_mode) == private


def test_make_caches_makes_a_new_default_root_private_under_a_permissive_umask(tmp_path, monkeypatch):
    """`make_caches` makes the default root through the same check-and-make call as the resolver."""
    root = tmp_path / "qda_someone"
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    env = {name: str(root / leaf) for name, leaf in vllm_env.CACHE_DIRS.items()}
    previous = os.umask(0o022)
    try:
        vllm_env.make_caches(env)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700 | _inherited_setgid(tmp_path)
    for leaf in vllm_env.CACHE_DIRS.values():
        assert (root / leaf).is_dir()


def test_make_caches_refuses_a_default_root_another_user_made(tmp_path, monkeypatch):
    """The check is in the call that makes the root, so `make_caches` alone writes nothing under a foreign root."""
    root = tmp_path / "qda_someone"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    monkeypatch.setattr(vllm_env.os, "getuid", lambda: root.stat().st_uid + 1)
    env = {name: str(root / leaf) for name, leaf in vllm_env.CACHE_DIRS.items()}
    with pytest.raises(Refused) as refused:
        vllm_env.make_caches(env)
    assert "another user" in str(refused.value)


def test_a_root_is_not_checked_when_the_user_named_every_cache(tmp_path, monkeypatch):
    """The variables the user set are used as given, and the default root is never looked at for them."""
    root = default_root(tmp_path, monkeypatch, mode=0o777)
    monkeypatch.setattr(vllm_env.os, "getuid", lambda: root.stat().st_uid + 1)
    named = {name: str(tmp_path / leaf) for name, leaf in vllm_env.CACHE_DIRS.items()}
    for name, path in named.items():
        monkeypatch.setenv(name, path)
    resolved = vllm_env.cache_env({})
    assert {name: resolved[name] for name in vllm_env.CACHE_DIRS} == named


def test_the_root_qda_makes_is_private_under_a_group_writable_umask(tmp_path, monkeypatch):
    """Under umask 0002 the parents `mkdir(parents=True)` made were 0775, and the engine start refused them."""
    root = tmp_path / "qda_user"
    monkeypatch.setattr(vllm_env, "cache_root", lambda: root)
    for name in vllm_env.CACHE_DIRS:
        monkeypatch.delenv(name, raising=False)
    old = os.umask(0o002)
    try:
        assert vllm_env.cache_problem(root / "flashinfer") is None  # doctor's probe of one leaf
        vllm_env.make_caches(vllm_env.cache_env({}))  # the engine start
    finally:
        os.umask(old)
    assert root.stat().st_mode & 0o777 == 0o700
    assert all((root / leaf).stat().st_mode & 0o777 == 0o700 for leaf in vllm_env.CACHE_DIRS.values())
    assert vllm_env.cache_root_problem(root) is None


def test_a_cache_that_cannot_be_locked_is_a_problem_and_a_busy_lock_is_not(tmp_path, monkeypatch):
    """What doctor asks of the cache: the directory can be made and a file in it locked. Contention is fine."""
    assert vllm_env.cache_problem(tmp_path / "fresh") is None
    assert (tmp_path / "fresh").stat().st_mode & 0o777 == 0o700

    def busy(fd, operation):
        raise BlockingIOError(errno.EAGAIN, "already locked")

    monkeypatch.setattr(vllm_env.fcntl, "flock", busy)
    assert vllm_env.cache_problem(tmp_path / "contended") is None

    def broken(fd, operation):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(vllm_env.fcntl, "flock", broken)
    problem = vllm_env.cache_problem(tmp_path / "network")
    assert problem is not None and str(tmp_path / "network") in problem and "Input/output" in problem


def test_the_engine_environment_carries_the_caches(tmp_path, monkeypatch):
    """`vllm_environ` is where an in-process engine gets them, and they are restored after the block."""
    monkeypatch.setattr(vllm_env, "refuse_serve", lambda: None, raising=False)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "exported"))
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    import qda_quant.checks as checks

    monkeypatch.setattr(checks, "refuse_serve", lambda: None)
    with vllm_env.vllm_environ({}):
        assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "exported")
        assert os.environ["VLLM_CACHE_ROOT"] == str(vllm_env.cache_root() / "vllm")
    assert "VLLM_CACHE_ROOT" not in os.environ


#: What nvcc printed when its host compiler could not find the system headers (the block-scaled FP8 case).
FATAL = "/usr/include/features-time64.h:20:10: fatal error: bits/wordsize.h: No such file or directory"


def executable(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def host_compiler(directory: Path, works: bool) -> Path:
    """A ``g++`` that preprocesses (``works``) or fails like the module-loaded gcc did."""
    return executable(directory, "g++", "exit 0\n" if works else f"echo '{FATAL}' >&2\nexit 1\n")


def fake_nvcc(tmp_path: Path) -> Path:
    """An ``nvcc`` that preprocesses as nvcc does.

    ``NVCC_PREPEND_FLAGS`` come before its arguments and ``NVCC_APPEND_FLAGS`` after; the last ``-ccbin`` wins;
    with none, ``NVCC_CCBIN``, then the g++ on ``PATH``.
    """
    return executable(
        tmp_path / "cuda" / "bin",
        "nvcc",
        f"""
set -- ${{NVCC_PREPEND_FLAGS:-}} "$@" ${{NVCC_APPEND_FLAGS:-}}
cxx=""
last=""
for arg in "$@"; do
  case "$last" in -ccbin|--compiler-bindir) cxx="$arg" ;; esac
  case "$arg" in -ccbin=*|--compiler-bindir=*) cxx="${{arg#*=}}" ;; esac
  last="$arg"
done
if [ -z "$cxx" ]; then cxx="$NVCC_CCBIN"; fi
if [ -z "$cxx" ]; then cxx=$(command -v g++); fi
case "$cxx" in
  */good/*) exit 0 ;;
esac
echo "{FATAL}" >&2
exit 1
""",
    )


@pytest.fixture
def host_engine(monkeypatch, tmp_path):
    """The engine environment without the FlashInfer toolchain check, which these tests do not exercise."""
    monkeypatch.setattr(vllm_env, "cache_env", lambda env=None: dict(env or {}))
    monkeypatch.delenv("NVCC_CCBIN", raising=False)
    fake_nvcc(tmp_path)
    return {"CUDA_HOME": str(tmp_path / "cuda")}


def test_the_broken_host_compiler_is_replaced_by_the_configured_one(host_engine, tmp_path):
    bad = host_compiler(tmp_path / "bad", works=False)
    good = host_compiler(tmp_path / "good", works=True)
    host_engine |= {"PATH": f"{bad.parent}:{good.parent}", "CXX": str(good)}
    with vllm_env.vllm_environ(host_engine):
        assert os.environ["NVCC_CCBIN"] == str(good)
    assert "NVCC_CCBIN" not in os.environ


def test_a_working_host_compiler_is_left_alone(host_engine, tmp_path):
    good = host_compiler(tmp_path / "good", works=True)
    host_engine["PATH"] = str(good.parent)
    with vllm_env.vllm_environ(host_engine):
        assert "NVCC_CCBIN" not in os.environ


def test_an_explicit_working_ccbin_is_left_alone(host_engine, tmp_path):
    good = host_compiler(tmp_path / "good", works=True)
    host_engine |= {"PATH": "", "NVCC_CCBIN": str(good)}
    with vllm_env.vllm_environ(host_engine):
        assert os.environ["NVCC_CCBIN"] == str(good)


def test_an_explicit_working_ccbin_in_prepend_flags_passes(host_engine, tmp_path):
    good = host_compiler(tmp_path / "good", works=True)
    host_engine |= {"PATH": "", "NVCC_PREPEND_FLAGS": f"-ccbin {good}"}
    with vllm_env.vllm_environ(host_engine):
        assert "NVCC_CCBIN" not in os.environ


@pytest.mark.parametrize(
    "variable, flag",
    [("NVCC_CCBIN", "")]
    + [
        (variable, flag)
        for variable in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
        for flag in ("-ccbin ", "--compiler-bindir ", "-ccbin=", "--compiler-bindir=")
    ],
)
def test_an_explicit_broken_ccbin_is_refused_not_replaced(host_engine, tmp_path, variable, flag):
    """Appending a compiler option the engine will not use can pass a probe on a broken engine."""
    bad = host_compiler(tmp_path / "bad", works=False)
    good = host_compiler(tmp_path / "good", works=True)
    host_engine["PATH"] = str(good.parent)
    host_engine[variable] = f"{flag}{bad}"
    with (
        pytest.raises(Refused, match="host compiler nvcc was given.*bits/wordsize.h"),
        vllm_env.vllm_environ(host_engine),
    ):
        pass


def test_refusal_names_the_missing_header_and_what_to_set(host_engine, tmp_path, monkeypatch):
    bad = host_compiler(tmp_path / "bad", works=False)
    monkeypatch.setattr(vllm_env, "HOST_COMPILERS", ())
    host_engine |= {"PATH": str(bad.parent), "CXX": str(bad)}
    with pytest.raises(Refused, match="bits/wordsize.h.*NVCC_CCBIN"), vllm_env.vllm_environ(host_engine):
        pass


def test_the_vens_nvidia_libraries_go_on_the_loader_path_once(tmp_path, monkeypatch):
    """Humming's NVRTC child process needs the venv's nvidia/*/lib, and an explicit path is kept."""
    site = tmp_path / "site-packages"
    (site / "nvidia" / "cu13" / "lib").mkdir(parents=True)
    (site / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
    paths = {"purelib": str(site), "platlib": str(site)}
    monkeypatch.setattr(vllm_env.sysconfig, "get_paths", lambda: paths)
    libs = [str(site / "nvidia" / "cu13" / "lib"), str(site / "nvidia" / "cudnn" / "lib")]
    assert vllm_env.nvidia_lib_path().split(os.pathsep) == libs
    explicit = f"/opt/cuda/lib64{os.pathsep}{libs[1]}"
    merged = vllm_env.loader_env({"LD_LIBRARY_PATH": explicit, "OTHER": "kept"})
    assert merged["LD_LIBRARY_PATH"].split(os.pathsep) == [*libs, "/opt/cuda/lib64"]
    assert merged["OTHER"] == "kept"
    again = vllm_env.loader_env({"LD_LIBRARY_PATH": merged["LD_LIBRARY_PATH"]})
    assert again["LD_LIBRARY_PATH"] == merged["LD_LIBRARY_PATH"], "a second pass adds nothing"


def test_no_nvidia_package_leaves_the_environment_alone(tmp_path, monkeypatch):
    paths = {"purelib": str(tmp_path / "empty"), "platlib": str(tmp_path / "empty")}
    monkeypatch.setattr(vllm_env.sysconfig, "get_paths", lambda: paths)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert vllm_env.nvidia_lib_path() == ""
    assert vllm_env.loader_env({"A": "b"}) == {"A": "b"}


def test_the_engine_environment_carries_the_nvidia_directory(tmp_path, monkeypatch):
    """`vllm_environ` is the one place the engine's environment is built; the child inherits it."""
    monkeypatch.setattr(vllm_env, "tool_path", lambda: "/bin")
    monkeypatch.setattr(vllm_env, "nvidia_lib_path", lambda: str(tmp_path / "lib"))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with vllm_env.vllm_environ({"OTHER": "1"}):
        assert os.environ["LD_LIBRARY_PATH"] == str(tmp_path / "lib")
        assert os.environ["OTHER"] == "1"
    assert "LD_LIBRARY_PATH" not in os.environ


def test_the_mount_point_of_a_path_is_the_longest_one_above_it():
    mountinfo = "\n".join(
        [
            "22 1 0:21 / / rw,relatime - overlay overlay rw",
            "36 22 0:32 / /shared rw - nfs4 nfssrv:/home rw",
            "37 36 0:33 / /shared/work/02 rw - ext4 /dev/sdb1 rw",
            r"38 22 0:34 / /mnt/a\040b rw - nfs nfssrv:/x rw",
        ]
    )
    assert vllm_env.fstype_from_mountinfo("/shared/work/02/user", mountinfo) == "ext4"
    assert vllm_env.fstype_from_mountinfo("/shared/other", mountinfo) == "nfs4"
    assert vllm_env.fstype_from_mountinfo("/var/tmp", mountinfo) == "overlay"
    assert vllm_env.fstype_from_mountinfo("/mnt/a b/c", mountinfo) == "nfs"
    assert vllm_env.fstype_from_mountinfo("/shared", "") is None


def test_cache_dir_is_the_env_then_this_process_then_the_root_and_makes_nothing(tmp_path, monkeypatch):
    """`cache_dir` is the rule `cache_env` applies, with no side effect: the passed env wins, then this process's
    variable, then the leaf under `cache_root`, which it does not create."""
    monkeypatch.setattr(vllm_env, "cache_root", lambda: tmp_path / "root")
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    assert vllm_env.cache_dir("VLLM_CACHE_ROOT") == tmp_path / "root" / "vllm"
    assert not (tmp_path / "root").exists()
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "shell"))
    assert vllm_env.cache_dir("VLLM_CACHE_ROOT") == tmp_path / "shell"
    assert (
        vllm_env.cache_dir("VLLM_CACHE_ROOT", {"VLLM_CACHE_ROOT": str(tmp_path / "passed")})
        == tmp_path / "passed"
    )


@pytest.mark.parametrize(
    ("prebuilt", "nvcc", "cxx", "cannot"),
    [
        (True, False, False, False),
        (False, True, True, False),
        (False, False, True, True),
        (False, True, False, True),
    ],
)
def test_flashinfer_cannot_build_without_its_prebuilt_kernels_and_nvcc_or_cxx(
    tmp_path, monkeypatch, prebuilt, nvcc, cxx, cannot
):
    """An RTX PRO 6000 without nvcc died in FlashInfer's JIT (`Could not find nvcc`) once FlashInfer's prebuilt
    kernels were hidden; with them it needs neither nvcc nor c++."""
    find_spec = vllm_env.importlib.util.find_spec
    installed = {"flashinfer": True, "flashinfer_jit_cache": prebuilt}

    def spec(name, *args):
        return (object() if installed[name] else None) if name in installed else find_spec(name, *args)

    monkeypatch.setattr(vllm_env.importlib.util, "find_spec", spec)
    (tmp_path / "bin").mkdir()
    if nvcc:
        executable(tmp_path / "cuda" / "bin", "nvcc", "exit 0\n")
    if cxx:
        executable(tmp_path / "bin", "c++", "exit 0\n")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert vllm_env.flashinfer_cannot_build() is cannot
