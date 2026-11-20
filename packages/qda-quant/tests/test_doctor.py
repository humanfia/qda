"""`qda doctor`: the ``qda.checks`` registry, the OK/MISS/skip/---- lines with their fix, the tail, the exit status.

The checks are faked: nothing here reads a GPU, a tool's version, or the machine's ``PATH``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from qda_quant import checks, doctor
from qda_quant.doctor import Check, Context, Info, main, report, run
from qda_quant.output import vllm_env
from qda_quant.registry import Registry
from qda_quant.step import Refused
from qda_quant.target import Target

#: The real driver comparison, kept before `_no_driver_query` replaces it for every test of this module.
REAL_DRIVER_PROBLEM = checks.cuda_driver_problem


@pytest.fixture(autouse=True)
def _no_driver_query(monkeypatch):
    """The GPU line's tests fake ``nvidia-smi`` through ``subprocess.run``; the CUDA driver query (a child
    process with this host's libcuda) is tested on its own below, so the rest see a driver that fits torch."""
    monkeypatch.setattr(checks, "cuda_driver_problem", lambda **k: None)


def fake_checks(monkeypatch, checks: dict) -> Registry:
    """Use the supplied checks without installed extensions."""
    from .test_plugins import fake_plugins

    fake_plugins(monkeypatch, lambda: None)
    fresh = Registry("checks", checks)
    monkeypatch.setattr(doctor, "CHECKS", fresh)
    return fresh


def args(**kw) -> argparse.Namespace:
    """What `main` reads from the parsed ``qda doctor`` options."""
    return argparse.Namespace(**{"search": None, "set": [], "run_dir": None, "report": False, **kw})


def test_a_miss_names_the_one_command_that_fixes_it():
    assert Check("GPU", True, "1 × H100, driver 550.54").line() == "OK   GPU: 1 × H100, driver 550.54"
    assert (
        Check("ninja", False, "not found", "pip install ninja").line()
        == "MISS ninja: not found (fix: pip install ninja)"
    )
    later = Check("GDN kernels", False, "needs a model", unchecked=True, where="qda build of a GDN model")
    assert later.line() == "---- GDN kernels: needs a model; checked by qda build of a GDN model"
    assert not later.missed, "a line only the command decides is never a MISS"
    assert Check("search", True, "not checked", skipped=True).line() == "skip search: not checked"
    assert Info("glibc", "glibc 2.35").line() == "glibc: glibc 2.35"
    assert Check("x", True).line() == "OK   x"


def test_run_asks_every_plugin_in_name_order(monkeypatch):
    """Each plugin's lines are kept in its own order; a plugin that raises is one MISS naming its key."""
    fake_checks(
        monkeypatch,
        {
            "second": lambda ctx: [Check("b", True, "fine")],
            "first": lambda ctx: [Check("a", True, "fine")],
            "broken": lambda ctx: (_ for _ in ()).throw(RuntimeError("no service")),
        },
    )
    lines = run(Context())
    assert [c.name for c in lines] == ["broken", "a", "b"]
    assert lines[0].name == "broken" and not lines[0].ok
    assert lines[0].detail == "the check raised RuntimeError: no service"


def test_a_search_asked_for_without_qda_agent_is_a_required_miss(monkeypatch, capsys):
    """`--search` needs the ``search`` plugin; a machine without qda-agent fails rather than passing silently."""
    fake_checks(monkeypatch, {"environment": lambda ctx: [Check("GPU", True, "available")]})
    assert main(args(search=Path("search.yaml"))) == 3  # a missing plugin is the machine, not the input
    out = capsys.readouterr().out
    assert (
        "MISS search: search.yaml: qda-agent is not installed (fix: uv pip install -c constraints.txt ./packages/qda-agent)"
        in out
    )
    assert out.endswith("checked 2, 1 MISS, 0 cannot be checked in advance\n")


PASS_A = [Check("a", True, "fine")]  # every registry below passes this one check first
SKIP_B = [Check("b", True, "off", skipped=True)]
MISS_C = [Check("c", False, "missing", "install it")]
LATER_B = [Check("b", False, "needs a model", unchecked=True, where="qda build")]
MISS_GPU = [Check("GPU", False, "nvidia-smi: not found", "install the NVIDIA driver")]
LATER_GGUF = [Check("GGUF converter", False, "no checkout", unchecked=True, where="a GGUF build")]


def _doctor(monkeypatch, capsys, registry: dict) -> tuple[int, str]:
    """Run `main` over `registry`; return its exit status and the report it printed."""
    fake_checks(monkeypatch, registry)
    code = main(args())
    return code, capsys.readouterr().out


MISS_CASES = [
    # (the registry, the report `main` prints, its exit status)
    (
        {"one": lambda ctx: [*PASS_A, *SKIP_B], "two": lambda ctx: MISS_C},
        "OK   a: fine\nskip b: off\nMISS c: missing (fix: install it)\n"
        "checked 2, 1 MISS, 0 cannot be checked in advance\n",
        3,
    ),
    # a line only the command decides (a model's kernels, a GGUF build's converter) is K, never a MISS
    (
        {"a-checked": lambda ctx: PASS_A, "b-later": lambda ctx: LATER_B},
        "OK   a: fine\n---- b: needs a model; checked by qda build\n"
        "checked 1, 0 MISS, 1 cannot be checked in advance\n",
        0,
    ),
    (
        {"ok": lambda ctx: PASS_A, "gpu": lambda ctx: MISS_GPU, "later": lambda ctx: LATER_GGUF},
        "MISS GPU: nvidia-smi: not found (fix: install the NVIDIA driver)\n"
        "---- GGUF converter: no checkout; checked by a GGUF build\nOK   a: fine\n"
        "checked 2, 1 MISS, 1 cannot be checked in advance\n",
        3,
    ),
]
MISS_IDS = ["a-miss-fails", "a-later-line-is-not-a-miss", "a-miss-and-a-later-line"]


@pytest.mark.parametrize(("registry", "out", "code"), MISS_CASES, ids=MISS_IDS)
def test_miss_lines_and_exit_status(monkeypatch, capsys, registry, out, code):
    """A miss prints its line and fix and exits 3; the tail says what was checked, never that all passed."""
    all_pass = {"ok": lambda ctx: PASS_A}
    assert _doctor(monkeypatch, capsys, all_pass) == (
        0,
        "OK   a: fine\nchecked 1, 0 MISS, 0 cannot be checked in advance\n",
    )
    assert _doctor(monkeypatch, capsys, registry) == (code, out)  # `main` prints the report `run` produces
    assert report(run(Context())) == out
    assert "passed" not in report([Check("a", True)])


def test_a_config_that_does_not_load_exits_2_and_report_mode_exits_0(monkeypatch, capsys):
    """A miss of the input (`USAGE`) exits 2; ``--report`` prints facts only and exits 0 whatever they are."""
    from qda_quant.exit import USAGE

    fake_checks(
        monkeypatch, {"x": lambda ctx: [Check("run config", False, "bad yaml", "edit it", code=USAGE)]}
    )
    assert main(args()) == 2
    capsys.readouterr()
    fake_checks(monkeypatch, {"x": lambda ctx: [Info("glibc", "glibc 2.28, below the vLLM extra's 2.31")]})
    assert main(args(report=True)) == 0
    assert capsys.readouterr().out == "glibc: glibc 2.28, below the vLLM extra's 2.31\n"


def test_a_skipped_check_alone_passes(monkeypatch):
    fake_checks(monkeypatch, {"later": lambda ctx: [Check("x", False, "not this machine", skipped=True)]})
    assert main(args()) == 0


def test_a_package_adds_a_check_through_an_entry_point(monkeypatch):
    """A third-party ``qda.checks`` entry point is a plugin like any other: it runs and keeps its provenance."""
    from .test_plugins import fake_plugins

    def from_package(ctx):  # noqa: ARG001
        return [Check("mine", True, "from a package")]

    fresh = fake_checks(monkeypatch, {})
    fake_plugins(monkeypatch, lambda: fresh.register("mine", from_package))
    assert [c.name for c in run(Context())] == ["mine"]


def test_a_missing_tool_is_a_miss_with_the_command_that_installs_it(monkeypatch):
    """A tool no command in scope needs is out of scope (a skip naming its install); one a build needs is a MISS."""

    monkeypatch.setattr(checks.shutil, "which", lambda *a, **k: None)
    tools = checks.tools_info()
    assert tools.name == "tools" and "ninja not found" in tools.detail  # a fact for --report, not a verdict

    def missing(dist):
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(checks.md, "version", missing)
    vllm = checks.check_vllm()
    assert vllm.skipped and 'uv pip install -c constraints.txt "./packages/qda-kl[vllm]"' in vllm.detail

    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    gpu = checks.check_gpu()
    assert gpu.missed and gpu.fix == "install the NVIDIA driver"


def test_the_compiler_check_builds_tritons_driver_and_names_one_that_works(monkeypatch, tmp_path):
    """A broken `CC` is a miss with its first error and a `PATH` compiler that works; an empty `CC` is used
    as Triton uses it (the empty override, which fails), not silently replaced; no compiler is a miss."""

    monkeypatch.setattr(checks, "_triton_driver", lambda *_: (tmp_path / "driver.c", []))
    monkeypatch.setattr(checks, "_nvcc", lambda *_: None)
    failed = "fatal error: bits/libc-header-start.h: No such file or directory"
    denied = "PermissionError: [Errno 13] Permission denied: ''"

    def fake_build(cc, *args):  # noqa: ARG001
        if cc == "/usr/bin/gcc":
            return None
        return denied if cc == "" else failed

    monkeypatch.setattr(checks, "_build", fake_build)
    monkeypatch.setattr(checks, "_on_path", lambda: ["/opt/spack/bin/gcc", "/usr/bin/gcc"])

    monkeypatch.setenv("CC", "/opt/spack/bin/gcc")
    missed = checks.check_compiler()
    assert missed.detail == f"/opt/spack/bin/gcc cannot build Triton's CUDA driver: {failed}"
    assert missed.fix == "CC=/usr/bin/gcc"

    monkeypatch.setenv("CC", "")
    empty = checks.check_compiler()
    assert empty.detail == f"CC='' cannot build Triton's CUDA driver: {denied}"
    assert empty.fix == "CC=/usr/bin/gcc"

    monkeypatch.setenv("CC", "/usr/bin/gcc")
    assert checks.check_compiler() == Check("C compiler", True, "/usr/bin/gcc")

    monkeypatch.delenv("CC")
    monkeypatch.setattr(checks.shutil, "which", lambda name: None)
    assert checks.check_compiler().fix == "install gcc"


def test_the_compiler_line_misses_when_flashinfer_has_no_cxx(tmp_path, monkeypatch):
    """FlashInfer JIT-compiles with ``CXX``, else ``c++``; a machine with only ``g++-13`` passes the Triton
    build (gcc) but every engine start stops at FlashInfer. The line names the versioned compiler as the fix,
    and passes once ``c++`` or ``CXX`` resolves or the prebuilt kernels are installed."""
    real = checks.importlib.util.find_spec
    jit_cache = {"on": False}

    def find_spec(name, *a):
        if name == "flashinfer":
            return object()
        if name == "flashinfer_jit_cache":
            return object() if jit_cache["on"] else None
        return real(name, *a)

    monkeypatch.setattr(checks.importlib.util, "find_spec", find_spec)
    gxx = tmp_path / "g++-13"
    gxx.write_text("#!/bin/sh\n")
    gxx.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CXX", raising=False)
    missed = checks.check_flashinfer_cxx()
    assert missed is not None and missed.missed and missed.name == "C compiler"
    assert missed.detail.startswith("c++ not found: FlashInfer compiles its kernels with it")
    assert missed.fix == f"CXX={gxx}; or run `{checks.DOWNLOAD_KERNELS}` once for prebuilt kernels"

    monkeypatch.setenv("CXX", str(gxx))
    assert checks.check_flashinfer_cxx() is None
    monkeypatch.delenv("CXX")
    (tmp_path / "c++").symlink_to(gxx)
    assert checks.check_flashinfer_cxx() is None
    (tmp_path / "c++").unlink()
    jit_cache["on"] = True
    assert checks.check_flashinfer_cxx() is None


def test_doctor_misses_when_flashinfer_has_no_nvcc_and_no_prebuilt_kernels(tmp_path, monkeypatch):
    """A GH200 node with no CUDA toolkit: `qda doctor` printed 0 MISS, then `vllm serve` died building
    FlashInfer's sampler (`Could not find nvcc`). The line names the download that fixes it, and goes once
    nvcc resolves or the prebuilt kernels are installed."""
    real = checks.importlib.util.find_spec
    jit_cache = {"on": False}

    def find_spec(name, *a):
        if name == "flashinfer":
            return object()
        if name == "flashinfer_jit_cache":
            return object() if jit_cache["on"] else None
        return real(name, *a)

    monkeypatch.setattr(checks.importlib.util, "find_spec", find_spec)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(checks, "_nvcc", lambda env: None)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    missed = checks.check_flashinfer_nvcc()
    assert missed is not None and missed.missed and missed.name == "FlashInfer kernels"
    assert f"`{checks.DOWNLOAD_KERNELS}`" in missed.fix
    monkeypatch.setattr(checks, "_serving", lambda: True)
    assert "FlashInfer kernels" in [c.name for c in checks.environment(Context()) if isinstance(c, Check)]
    search = checks.environment(Context(search=tmp_path / "search.yaml"))
    assert "FlashInfer kernels" not in [c.name for c in search if isinstance(c, Check)]

    monkeypatch.setattr(checks, "_nvcc", lambda env: "/usr/local/cuda/bin/nvcc")
    assert checks.check_flashinfer_nvcc() is None
    monkeypatch.setattr(checks, "_nvcc", lambda env: None)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    assert checks.check_flashinfer_nvcc() is None
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER")
    jit_cache["on"] = True
    assert checks.check_flashinfer_nvcc() is None


def test_every_kernels_fix_names_this_venvs_vllm_download_kernels():
    """One doctor run showed `vllm download-kernels` on one line and `python -m flashinfer download-kernels` on
    another; a bare name can resolve to another venv's. Every fix names this interpreter's `vllm`."""
    vllm = Path(sys.executable).with_name("vllm")
    module = f"{sys.executable} -m vllm.entrypoints.cli.main download-kernels"
    assert checks.DOWNLOAD_KERNELS == (f"{vllm} download-kernels" if vllm.exists() else module)
    nvcc = next(fix for pattern, fix, _ in vllm_env.ENGINE_FIXES if pattern.search("Could not find nvcc"))
    assert f"`{checks.DOWNLOAD_KERNELS}`" in nvcc


@pytest.mark.parametrize(
    ("cache", "flashinfer", "matches"),
    [
        ("0.7.2.post1+cu129", "0.7.2.post1", True),
        ("0.7.2.post1", "0.7.2.post1", True),
        ("0.7.1+cu129", "0.7.2.post1", False),
        ("0.7.2+cu129", "0.7.2.post1", False),
    ],
)
def test_kernels_of_another_flashinfer_version_count_as_missing(monkeypatch, cache, flashinfer, matches):
    """FlashInfer ignores a `flashinfer_jit_cache` of another version unless FLASHINFER_DISABLE_VERSION_CHECK is
    set, and JIT-compiles (`flashinfer.jit.env`): an upgraded flashinfer with the old kernels still needs nvcc."""
    problem = None if matches else f"prebuilt FlashInfer kernels {cache} are not for flashinfer {flashinfer}"
    real = vllm_env.importlib.util.find_spec
    monkeypatch.setattr(
        vllm_env.importlib.util,
        "find_spec",
        lambda name, *a: object() if name in ("flashinfer", "flashinfer_jit_cache") else real(name, *a),
    )
    versions = {"flashinfer-jit-cache": cache, "flashinfer-python": flashinfer}
    monkeypatch.setattr(vllm_env.md, "version", versions.__getitem__)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    assert vllm_env.flashinfer_kernels_problem() is None
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK")
    assert vllm_env.flashinfer_kernels_problem() == problem
    monkeypatch.setattr(checks, "_nvcc", lambda env: None)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    missed = checks.check_flashinfer_nvcc()
    assert (missed is None) is (problem is None)
    if problem:
        assert missed.detail.startswith(f"no nvcc and {problem}: ")


def test_the_glibc_line_names_the_vllm_extras_floor(monkeypatch):
    """Below glibc 2.31 the vLLM extra's llguidance wheel cannot install. No command refuses on it, so it is
    ``--report``'s fact, with the floor and the wheel named."""
    monkeypatch.setattr(checks.os, "confstr", lambda name: "glibc 2.28")
    low = checks.glibc_info()
    assert low.name == "glibc" and "glibc 2.28" in low.detail and "2.31" in low.detail
    assert "cp39-abi3" in low.detail and "manylinux_2_31" in low.detail and "Rust" in low.detail

    monkeypatch.setattr(checks.os, "confstr", lambda name: "glibc 2.31")
    assert checks.glibc_info() == Info("glibc", "glibc 2.31")

    def not_glibc(name):
        raise ValueError(name)

    monkeypatch.setattr(checks.os, "confstr", not_glibc)
    assert checks.glibc_info().detail == "no GNU libc version (not a glibc host)"


#: What the module-loaded g++ printed when nvcc preprocessed with it.
FATAL = "/usr/include/features-time64.h:20:10: fatal error: bits/wordsize.h: No such file or directory"


def executable(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def host_compiler(directory: Path, works: bool) -> Path:
    """A ``g++`` that preprocesses (``works``) or fails on the system headers, as the spack build did."""
    return executable(directory, "g++", "exit 0\n" if works else f"echo '{FATAL}' >&2\nexit 1\n")


@pytest.fixture
def host_compiler_probe(monkeypatch, tmp_path):
    """nvcc runs the fake at ``tmp_path/cuda/bin``; Triton has no driver here, so only the nvcc probe runs."""
    monkeypatch.setattr(checks, "_triton_driver", lambda *_: None)
    monkeypatch.setattr(vllm_env, "HOST_COMPILERS", ())
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    for name in ("NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    return executable(
        tmp_path / "cuda" / "bin",
        "nvcc",
        f"""
cxx=""
last=""
for arg in "$@"; do
  if [ "$last" = "-ccbin" ]; then cxx="$arg"; fi
  last="$arg"
done
for flags in "$NVCC_PREPEND_FLAGS" "$NVCC_APPEND_FLAGS"; do
  for arg in $flags; do
    if [ "$last" = "-ccbin" ]; then cxx="$arg"; fi
    last="$arg"
  done
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


def test_a_host_compiler_the_engine_selects_is_not_a_miss(host_compiler_probe, monkeypatch, tmp_path):
    """nvcc's default g++ is the spack build, but the engine selects CXX, so the doctor passes."""
    good = host_compiler(tmp_path / "good", works=True)
    monkeypatch.setenv("PATH", str(host_compiler(tmp_path / "bad", works=False).parent))
    monkeypatch.setenv("CXX", str(good))
    monkeypatch.setenv("CC", str(good))
    assert checks.check_compiler() == Check("C compiler", True, str(good))


def test_a_host_compiler_nvcc_cannot_preprocess_is_a_miss(host_compiler_probe, monkeypatch, tmp_path):
    """No compiler the engine would try preprocesses, so the engine refuses: the miss says why and the fix."""
    bad = host_compiler(tmp_path / "bad", works=False)
    monkeypatch.setenv("PATH", str(bad.parent))
    monkeypatch.setenv("CXX", str(bad))
    monkeypatch.setenv("CC", str(bad))
    missed = checks.check_compiler()
    assert not missed.ok and missed.detail.endswith(f": {FATAL}")
    assert "any host compiler the engine would try" in missed.detail
    assert missed.fix == "install a g++ that finds the system headers, or set NVCC_CCBIN to one"


def test_a_broken_compiler_named_for_nvcc_is_a_miss_that_does_not_replace_it(
    host_compiler_probe, monkeypatch, tmp_path
):
    """An explicit NVCC_CCBIN is the caller's choice: the engine refuses it rather than selecting another."""
    bad = host_compiler(tmp_path / "bad", works=False)
    good = host_compiler(tmp_path / "good", works=True)
    monkeypatch.setenv("PATH", str(bad.parent))
    monkeypatch.setenv("CXX", str(good))
    monkeypatch.setenv("CC", str(bad))
    monkeypatch.setenv("NVCC_CCBIN", str(bad))
    missed = checks.check_compiler()
    assert not missed.ok
    assert missed.detail == f"the host compiler nvcc is given ({bad}) cannot preprocess a CUDA file: {FATAL}"
    assert missed.fix == "remove -ccbin/NVCC_CCBIN or point it at a compiler that finds the system headers"


def test_a_ccbin_in_nvcc_prepend_flags_is_probed_as_the_engine_runs_it(
    host_compiler_probe, monkeypatch, tmp_path
):
    """``-ccbin`` in ``NVCC_PREPEND_FLAGS`` is nvcc's host compiler too, and is not replaced either."""
    bad = host_compiler(tmp_path / "bad", works=False)
    monkeypatch.setenv("PATH", str(host_compiler(tmp_path / "good", works=True).parent))
    monkeypatch.setenv("CC", str(bad))
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", f"-ccbin {bad}")
    missed = checks.check_compiler()
    assert not missed.ok and f"({bad})" in missed.detail and missed.detail.endswith(f": {FATAL}")
    assert missed.fix.startswith("remove -ccbin")


def test_an_explicit_working_nvcc_ccbin_passes(host_compiler_probe, monkeypatch, tmp_path):
    good = host_compiler(tmp_path / "good", works=True)
    monkeypatch.setenv("PATH", str(host_compiler(tmp_path / "bad", works=False).parent))
    monkeypatch.setenv("CC", str(good))
    monkeypatch.setenv("NVCC_CCBIN", str(good))
    assert checks.check_compiler() == Check("C compiler", True, str(good))


def test_a_probe_that_outlasts_the_budget_is_not_a_checked_ok(monkeypatch, tmp_path):
    """A probe that outlasts the budget must not be a checked OK: OK means the engine would start."""

    executable(
        tmp_path / "cuda" / "bin",
        "nvcc",
        "/bin/sleep 2.6\necho 'fatal error: bits/wordsize.h' >&2\nexit 1\n",
    )
    monkeypatch.setattr(checks, "_triton_driver", lambda *_: None)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setenv("CC", "/usr/bin/gcc")
    monkeypatch.setenv("CXX", "/usr/bin/g++")
    for name in ("NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    started = time.monotonic()
    got = checks.check_compiler()
    assert got.skipped or not got.ok, "doctor reports a checked OK without a successful nvcc probe"
    assert "budget was spent" in got.detail
    assert time.monotonic() - started < checks._BUDGET_S + 2
    with pytest.raises(Refused):  # the same environment: the engine refuses every compiler it tries
        vllm_env.host_compiler_env(dict(os.environ))


def test_a_slow_compiler_times_out_with_its_time(monkeypatch, tmp_path):
    """An attempt that outlasts its share of the budget names the time it had instead of waiting for it."""

    slow = tmp_path / "cc"
    slow.write_text("#!/bin/sh\nsleep 30\n")
    slow.chmod(0o755)
    monkeypatch.setattr(checks, "_ATTEMPT_S", 0.2)
    started = time.monotonic()
    why = checks._build(str(slow), (tmp_path / "driver.c", []), tmp_path / "driver.so", started + 5)
    assert why is not None and why.startswith("timed out after ")
    assert time.monotonic() - started < 2


def test_cuda_discovery_takes_tritons_libcuda_path_as_triton_does(monkeypatch, tmp_path):
    """Triton links its driver against ``TRITON_LIBCUDA_PATH`` alone when it is set, without ``ldconfig``."""
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", str(tmp_path))
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: pytest.fail("ldconfig ran"))
    assert checks._libcuda_dirs(time.monotonic() + 5) == [str(tmp_path)]


def test_a_slow_cuda_discovery_names_its_own_timeout(monkeypatch):
    """``ldconfig`` gets its own share; when it outlasts it, the miss names discovery, not a compiler."""

    monkeypatch.setattr(checks, "_DISCOVERY_S", 0.1)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    asked = []

    def slow(argv, **kwargs):
        asked.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(checks.subprocess, "run", slow)
    with pytest.raises(checks._DiscoveryTimeout, match="ldconfig"):
        checks._libcuda_dirs(time.monotonic() + 5)
    assert 0 < asked[0] <= 0.1

    def timed_out(*_):
        raise checks._DiscoveryTimeout("CUDA library discovery (/sbin/ldconfig -p) timed out after 1.0 s")

    monkeypatch.setattr(checks, "_triton_driver", timed_out)
    monkeypatch.setenv("CC", "/usr/bin/gcc")
    miss = checks.check_compiler()
    assert "CUDA library discovery" in miss.detail and "timed out after 1.0 s" in miss.detail
    assert "compiler was not tried" in miss.detail
    assert "ldconfig" in miss.fix and "install a gcc" not in miss.fix


def test_a_verified_library_path_settles_discovery_without_ldconfig(monkeypatch, tmp_path):
    """The fix's ``LD_LIBRARY_PATH`` remedy works: a directory holding the library is used, ``ldconfig`` skipped."""

    cuda = tmp_path / "cuda"
    cuda.mkdir()
    (cuda / "libcuda.so.1").write_text("")
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([str(cuda), "/nonexistent"]))

    def unasked(*_):  # noqa: ANN002
        raise AssertionError("ldconfig runs although LD_LIBRARY_PATH already holds libcuda.so.1")

    monkeypatch.setattr(checks.subprocess, "run", unasked)
    assert checks._libcuda_dirs(time.monotonic() + 5) == [str(cuda), "/nonexistent"]


def _shell_value(fix: str) -> str:
    """What the printed fix sets ``SGLANG_JIT_CACHE_DIR`` to when a shell runs it."""
    out = subprocess.run(
        ["bash", "-c", fix + '; printf %s "$SGLANG_JIT_CACHE_DIR"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_a_symlinked_sglang_cache_root_is_a_report_fact(monkeypatch, tmp_path):
    """SGLang's JIT cache only hits when its root is a real path. A symlinked root recompiles (slow, never
    refused), so ``--report`` says which form it is and names the variable, with no path of this machine."""

    real = tmp_path / "real cache"
    real.mkdir()
    link = tmp_path / "linked cache"
    link.symlink_to(real)
    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(link))
    linked = checks.sglang_cache_info()
    assert linked is not None and "symlinked" in linked.detail and "SGLANG_JIT_CACHE_DIR" in linked.detail
    assert str(tmp_path) not in linked.detail

    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(real))
    assert checks.sglang_cache_info() == Info("SGLang JIT cache", "a real path")

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda name: None)
    assert checks.sglang_cache_info() is None


def test_a_gpu_and_a_vllm_of_the_checked_version_pass(monkeypatch):
    two = "0, GPU-aaaa, NVIDIA H200, 580.105.08, 0, [N/A]\n1, GPU-bbbb, NVIDIA H200, 580.105.08, 0, [N/A]\n"
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, two, ""))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu = checks.check_gpu()
    assert gpu.ok and gpu.detail == "2 × NVIDIA H200, driver 580.105.08"

    wanted = Target().vllm
    monkeypatch.setattr(checks.md, "version", lambda dist: wanted)
    assert checks.check_vllm().ok
    other = ".".join([*wanted.split(".")[:2], "999"])  # the same minor series is checked
    monkeypatch.setattr(checks.md, "version", lambda dist: other)
    assert checks.check_vllm().ok
    major = f"{int(wanted.split('.')[0]) + 1}.0.0"
    monkeypatch.setattr(checks.md, "version", lambda dist: major)
    other_major = checks.check_vllm()  # no command refuses another vLLM: said, with the pin, and not a MISS
    assert not other_major.missed and wanted in other_major.detail
    assert f'pip install "vllm=={wanted}"' in other_major.detail


VISIBLE = "CUDA_VISIBLE_DEVICES"
EIGHT = "".join(f"{i}, GPU-{i:04d}, NVIDIA H100 80GB HBM3, 550.54.15, 0, [N/A]\n" for i in range(8))


def test_the_gpu_check_honours_cuda_visible_devices_and_says_so(monkeypatch):
    """``nvidia-smi`` lists every card, but CUDA gives the build only the named ones."""
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, EIGHT, "")
    )
    monkeypatch.delenv(VISIBLE, raising=False)
    assert checks.check_gpu().detail == "8 × NVIDIA H100 80GB HBM3, driver 550.54.15"

    monkeypatch.setenv(VISIBLE, "7")
    one = checks.check_gpu()
    assert one.ok and one.detail == "1 × NVIDIA H100 80GB HBM3, driver 550.54.15 (CUDA_VISIBLE_DEVICES=7)"

    monkeypatch.setenv(VISIBLE, "0, 2")  # CUDA takes the entries in their own order
    two = checks.check_gpu()
    assert two.detail == "2 × NVIDIA H100 80GB HBM3, driver 550.54.15 (CUDA_VISIBLE_DEVICES=0, 2)"

    monkeypatch.setenv(VISIBLE, "GPU-0003")  # a UUID, as docker passes it; a prefix is enough
    assert checks.check_gpu().detail.startswith("1 × ") and "GPU-0003" in checks.check_gpu().detail


def test_a_card_with_an_uncorrectable_ecc_error_is_a_miss_naming_it(monkeypatch):
    """GPU 4 of eight had a fatal ECC fault and every CUDA call on it failed, yet the line said OK."""
    ecc = EIGHT.replace(
        "4, GPU-0004, NVIDIA H100 80GB HBM3, 550.54.15, 0,",
        "4, GPU-0004, NVIDIA H100 80GB HBM3, 550.54.15, 2,",
    )
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, ecc, ""))
    monkeypatch.delenv(VISIBLE, raising=False)
    miss = checks.card_health_problem()
    assert miss is not None and miss.text == "card 4: 2 uncorrectable ECC error(s) since the driver loaded"
    assert VISIBLE in (miss.fix or "")
    assert checks.check_gpu().ok, "the GPU line lists the cards; the `card health` guard judges them"
    with pytest.raises(Refused) as refused:  # the build refuses on the same function, with the same words
        checks.refuse_build([4])
    assert str(refused.value) == miss.text and refused.value.code == 3 and refused.value.fix == miss.fix
    checks.refuse_build([0, 1])  # a build on the healthy cards is not refused
    monkeypatch.setenv(VISIBLE, "0,1,2,3")  # a build that cannot see the card does not need it
    assert checks.card_health_problem() is None
    monkeypatch.delenv(VISIBLE)
    lines = {c.name: c for c in checks.environment(Context()) if isinstance(c, Check)}
    assert lines["card health"].missed and lines["card health"].detail == miss.text
    # --search: the run's cards are the search's own `card health` line, so the build's is left out
    assert "card health" not in [c.name for c in checks.environment(Context(search=Path("run.yaml")))]


@pytest.mark.parametrize(
    "health,why",
    [
        ("0, Yes", "retired memory pages are pending"),
        (
            "[GPU requires reset], [N/A]",
            "reads ecc.errors.uncorrected.volatile.total as [GPU requires reset]",
        ),
        ("0, No", None),
        ("[N/A], [N/A]", None),  # no ECC on this card: nothing to report
        (
            "[Insufficient Permissions], [Insufficient Permissions]",
            None,  # a shared node's counters are root's: unknown, not a fault
        ),
    ],
)
def test_card_fault_reads_each_health_field(health, why):
    got = checks.card_fault(health.split(","))
    assert got == why if why is None else why in got


def test_a_permission_limited_health_counter_refuses_no_card(monkeypatch):
    """On a shared node nvidia-smi reports the counters as [Insufficient Permissions]: unknown, not a fault."""
    limited = EIGHT.replace(", 0, [N/A]", ", [Insufficient Permissions], [Insufficient Permissions]")
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, limited, "")
    )
    monkeypatch.delenv(VISIBLE, raising=False)
    gpu = checks.check_gpu()
    assert gpu.ok and gpu.detail.startswith("8 × ")
    assert checks.card_health_problem() is None


def test_a_card_nvidia_smi_cannot_read_and_a_missing_nvidia_smi_are_misses(monkeypatch):
    unread = EIGHT.replace(
        "4, GPU-0004, NVIDIA H100 80GB HBM3, 550.54.15, 0, [N/A]",
        "Unable to determine the device handle for GPU4: 0000:9A:00.0: Unknown Error",
    )
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, unread, "")
    )
    monkeypatch.delenv(VISIBLE, raising=False)
    listed = checks.check_gpu()
    assert listed.ok and "GPU4" in listed.detail and "Unknown Error" in listed.detail
    miss = checks.card_health_problem()  # a card nvidia-smi cannot read is the build's refusal
    assert miss is not None and "GPU4" in miss.text and "Unknown Error" in miss.text

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(checks.subprocess, "run", missing)
    gone = checks.check_gpu()
    assert (
        not gone.ok
        and gone.detail == "nvidia-smi: FileNotFoundError"
        and gone.fix == "install the NVIDIA driver"
    )


def test_a_cuda_visible_devices_that_names_no_card_is_a_miss(monkeypatch):
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, EIGHT, "")
    )
    monkeypatch.setenv(VISIBLE, "9")
    missing = checks.check_gpu()
    assert not missing.ok and "no card has index 9" in missing.detail
    assert missing.detail.startswith("CUDA_VISIBLE_DEVICES='9'") and "unset it" in missing.fix
    monkeypatch.setenv(VISIBLE, "")  # an empty value hides every card from CUDA
    empty = checks.check_gpu()
    assert not empty.ok and "sees no card" in empty.detail


def test_the_engine_cache_line_names_the_directory_and_its_miss_names_the_variables(monkeypatch, tmp_path):
    """One line for the engines' caches: where they are, and -- when a lock fails there -- what to export."""
    for name in checks.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        checks, "cache_env", lambda env=None: {n: str(tmp_path / "local") for n in checks.CACHE_ENV}
    )
    got = checks.check_engine_cache()
    assert got.ok and str(tmp_path / "local") in got.detail and "lockable" in got.detail

    monkeypatch.setattr(checks, "cache_problem", lambda base: f"{base}: [Errno 5] Input/output error")
    got = checks.check_engine_cache()
    assert got.missed and "Input/output error" in got.detail
    for name in checks.CACHE_ENV:
        assert name in got.fix, "the fix names every variable the user moves them with"

    # a cache the user moved is checked where they moved it, and the line says which variable named it
    monkeypatch.setattr(checks, "cache_problem", lambda base: None)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "exported"))
    assert "TRITON_CACHE_DIR" in checks.check_engine_cache().detail


def test_the_engine_cache_line_misses_a_default_root_writable_by_others(monkeypatch, tmp_path):
    """The same check the engine start makes: MISS when the default root is not this user's private directory."""
    for name in checks.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "qda_someone"
    root.mkdir()
    root.chmod(0o770)
    monkeypatch.setattr(checks, "cache_root", lambda: root)
    got = checks.check_engine_cache()
    assert got.missed and str(root) in got.detail and "writable by group or others" in got.detail
    assert got.fix == checks.CACHE_FIX


def test_the_engine_cache_line_reports_a_root_that_cannot_be_made(monkeypatch, tmp_path):
    """The resolver refuses a root it cannot make; doctor reports that refusal instead of tracing back."""
    for name in checks.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(checks, "cache_root", lambda: tmp_path / "qda_someone")  # not made: no mode problem

    def cannot_make() -> Path:
        raise Refused("/tmp/qda_someone: [Errno 28] No space left on device", checks.CACHE_FIX)

    monkeypatch.setattr(vllm_env, "ensure_cache_root", cannot_make)
    got = checks.check_engine_cache()
    assert got.missed and "No space left on device" in got.detail
    assert got.fix == checks.CACHE_FIX


def test_torch_cuda_reads_torch_version_py_without_importing_torch(monkeypatch, tmp_path):
    (tmp_path / "version.py").write_text(
        "from typing import Optional\n__version__ = '2.14.0+cu130'\ncuda: Optional[str] = '13.0'\n"
    )
    spec = argparse.Namespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda name: spec)
    assert checks.torch_cuda() == ("2.14.0+cu130", "13.0")

    (tmp_path / "version.py").write_text("__version__ = '2.14.0+cpu'\ncuda: Optional[str] = None\n")
    assert checks.torch_cuda() is None  # a CPU build has no CUDA to compare
    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda name: None)
    assert checks.torch_cuda() is None


@pytest.mark.parametrize(
    "rc,out,err,want",
    [
        (0, "12080 \n", "", (12080, "")),
        (0, "13000 CUDA_ERROR_NO_DEVICE\n", "", (13000, "CUDA_ERROR_NO_DEVICE")),
        (
            0,
            "noload libcuda.so.1: cannot open shared object file\n",
            "",
            (None, "libcuda.so.1 does not load: "),
        ),
        (
            1,
            "",
            "Traceback ...\nOSError: boom\n",
            (None, "not checked: the CUDA driver query failed: OSError: boom"),
        ),
    ],
)
def test_cuda_driver_reads_the_childs_answer(monkeypatch, rc, out, err, want):
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, out, err)
    )
    version, why = checks.cuda_driver()
    assert version == want[0] and why.startswith(want[1])


def test_the_driver_query_runs_in_a_child_without_importing_torch(monkeypatch):
    """The doctor process never initializes CUDA itself: the query is a child of this Python, isolated."""
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "13000 \n", "")

    monkeypatch.setattr(checks.subprocess, "run", run)
    checks.cuda_driver()
    assert seen[0][:3] == [checks.sys.executable, "-I", "-c"] and "import torch" not in seen[0][3]
    assert "cuDriverGetVersion" in seen[0][3] and "cuInit" in seen[0][3]


@pytest.mark.parametrize(
    "driver,detail,fix",
    [
        # the A100 on a 570 driver: nvidia-smi lists every card, torch's first CUDA call fails
        (
            (12080, ""),
            "the driver supports CUDA 12.8, but torch 2.14.0+cu130 is built for CUDA 13.0",
            "install an NVIDIA driver 580 or newer; or, on a data-center card, put NVIDIA's cuda-compat-13-0",
        ),
        ((13000, ""), None, None),  # a 580 driver, or 570 with the forward-compat libraries loaded
        ((13020, ""), None, None),  # a newer minor of the same major runs it too
        (
            (13000, "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE"),
            "CUDA does not start: cuInit returned CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
            "this card does not take forward-compat libraries: take them off LD_LIBRARY_PATH",
        ),
        (
            (13000, "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
            "CUDA does not start: cuInit returned CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
            "the libcuda on LD_LIBRARY_PATH does not match the kernel driver",
        ),
        (
            (None, "libcuda.so.1 does not load: x"),
            "libcuda.so.1 does not load: x",
            "install the NVIDIA driver",
        ),
    ],
)
def test_the_driver_problem_compares_libcudas_cuda_with_torchs(monkeypatch, driver, detail, fix):
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: driver)
    got = REAL_DRIVER_PROBLEM()
    if detail is None:
        assert got is None
    else:
        assert got is not None and got[0] == detail and got[1].startswith(fix)


def test_a_cpu_torch_is_never_a_driver_problem(monkeypatch):
    monkeypatch.setattr(checks, "torch_cuda", lambda: None)
    monkeypatch.setattr(checks, "cuda_driver", lambda: (12080, ""))
    assert REAL_DRIVER_PROBLEM() is None


def test_a_driver_too_old_for_torch_is_a_miss_not_an_ok(monkeypatch):
    """torch 2.14 (CUDA 13) on a 570 driver: nvidia-smi listed the cards and the line said OK, exit 0, and the
    first build stopped on "The NVIDIA driver on your system is too old". The `CUDA driver` guard misses with
    the fix, and the build and an engine start refuse with the same words."""
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, EIGHT, "")
    )
    monkeypatch.delenv(VISIBLE, raising=False)
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (12080, ""))
    monkeypatch.setattr(checks, "cuda_driver_problem", REAL_DRIVER_PROBLEM)
    assert checks.check_gpu().ok, "the GPU line lists the cards; the `CUDA driver` guard judges the driver"
    lines = {c.name: c for c in checks.environment(Context(search=Path("x.yaml")))}
    miss = lines["CUDA driver"]
    assert miss.missed and miss.fix.startswith("install an NVIDIA driver 580 or newer")
    assert miss.detail == "the driver supports CUDA 12.8, but torch 2.14.0+cu130 is built for CUDA 13.0"
    for refuse in (lambda: checks.refuse_build([0]), checks.refuse_serve):  # --search too: same driver
        with pytest.raises(Refused) as refused:
            refuse()
        assert (str(refused.value), refused.value.fix, refused.value.code) == (miss.detail, miss.fix, 3)


def test_a_driver_with_no_verdict_is_a_skip_that_says_why(monkeypatch):
    """No libcuda, or a query that did not answer: the build and an engine start leave it to torch, so doctor's
    line is a skip naming why, never an OK and never a refusal the commands do not make."""
    monkeypatch.setattr(
        checks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, EIGHT, "")
    )
    monkeypatch.delenv(VISIBLE, raising=False)
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (None, "libcuda.so.1 does not load: x"))
    monkeypatch.setattr(checks, "cuda_driver_problem", REAL_DRIVER_PROBLEM)
    line = {c.name: c for c in checks.environment(Context())}["CUDA driver"]
    assert line.skipped and line.detail == "libcuda.so.1 does not load: x"
    checks.refuse_build([0])
    checks.refuse_serve()
    monkeypatch.setattr(checks, "torch_cuda", lambda: None)
    assert {c.name: c for c in checks.environment(Context())}["CUDA driver"].detail == "torch is a CPU build"


def test_a_driver_query_with_no_answer_is_a_miss_that_says_not_checked(monkeypatch):
    """The doctor's rule: never an OK without the verdict; a query that did not answer says so."""
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (None, "not checked: the CUDA driver query failed: x"))
    got = REAL_DRIVER_PROBLEM()
    assert got is not None and got[0] == "not checked: the CUDA driver query failed: x"
    assert "rerun" in got[1]
    assert REAL_DRIVER_PROBLEM(verdict_only=True) is None  # the build leaves it to torch


def test_a_build_leaves_a_missing_libcuda_to_torch(monkeypatch):
    """A CPU host hands a build fake CUDA devices in tests: no libcuda at all is not the driver-too-old refusal."""
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (None, "libcuda.so.1 does not load: x"))
    assert REAL_DRIVER_PROBLEM(verdict_only=True) is None
    monkeypatch.setattr(checks, "cuda_driver", lambda: (12080, ""))
    assert REAL_DRIVER_PROBLEM(verdict_only=True) is not None
