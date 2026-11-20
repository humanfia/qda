"""The build's pre-flight: Triton's CUDA driver must compile on this machine before a card build starts.

A build JIT-compiles Triton's driver at its first kernel, so a Python installed without its
development headers dies inside Triton after the model load, at ``Python.h: No such file or directory``, with
no fix. The doctor already runs that compile; the build now runs it too, before any work, and a miss names
the package to install. The detection is the real compile against the running interpreter's include dir, so
a missing header, an empty ``/usr/include/pythonX.Y``, or Debian's missing multiarch ``pyconfig.h`` all fail
it (a file-existence test would pass the empty directory).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
from qda_quant import build, checks
from qda_quant.engine.build import _refuse_unbuildable_triton
from qda_quant.exit import ENVIRONMENT
from qda_quant.recipe import Recipe
from qda_quant.step import Refused


def skip_without_triton() -> None:
    """Every case here compiles Triton's driver source; a machine without it has nothing to compile."""
    spec = importlib.util.find_spec("triton")
    backend = (
        Path(next(iter(spec.submodule_search_locations)), "backends", "nvidia")
        if spec is not None and spec.submodule_search_locations
        else None
    )
    if backend is None or not (backend / "driver.c").is_file():
        pytest.skip("Triton's CUDA driver source is not installed here")


def empty_include(tmp_path, monkeypatch) -> str:
    """An include dir with no Python headers, and no ``CPATH`` through which the compiler could find any."""
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path / "include"
    directory.mkdir()
    return str(directory)


def dev_packages() -> tuple[str, str]:
    """The Debian and RHEL package names of this interpreter's headers, as the fix names them."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"python{version}-dev", f"python{version}-devel"


@pytest.mark.usefixtures("python_headers")
def test_the_running_interpreters_headers_build_tritons_driver():
    """This machine's own interpreter, Triton's own driver source and flags: the compile a build would run.

    The link needs a CUDA driver library; a machine without one cannot complete the compile, which says
    nothing about its Python headers, so it skips rather than fails.
    """
    skip_without_triton()
    check = checks.check_triton_build()
    if not check.ok and "libcuda.so.1" in check.detail:
        pytest.skip("no libcuda.so.1 to link Triton's driver against")
    if not check.ok and "Python.h" in check.detail:  # the refusal itself is the next test's
        pytest.skip("this interpreter has no Python.h (a system Python without its dev package)")
    assert check.ok and not check.skipped, check


def test_a_missing_python_header_refuses_and_names_the_dev_package(tmp_path, monkeypatch):
    """No header in the include dir is a real compile failure; the fix is the headers' package."""
    skip_without_triton()
    check = checks.check_triton_build(include=empty_include(tmp_path, monkeypatch))
    assert not check.ok
    assert "Python.h" in check.detail
    debian, rhel = dev_packages()
    assert debian in check.fix and rhel in check.fix and "conda" in check.fix
    assert "--managed-python" in check.fix  # the fix that needs no root


def test_a_card_build_refuses_before_it_reads_the_model(tmp_path, monkeypatch):
    """The boundary runs that check: a missing header is a Refused naming the fix, before the model is read;
    a CPU build compiles no driver, so it is not checked."""
    skip_without_triton()
    # the driver's own refusal is tested below
    monkeypatch.setattr(checks, "cuda_driver_problem", lambda **k: None)
    monkeypatch.setattr(checks, "_python_include", lambda: empty_include(tmp_path, monkeypatch))
    recipe = Recipe.from_yaml("steps: []\ncalib: {file: {path: rows.jsonl}}\n")
    with pytest.raises(Refused) as refused:
        build(
            "qwen/not-read",
            recipe,
            out_dir=tmp_path / "out",
            devices=[torch.device("cuda")],
            log=lambda _: None,
        )
    assert "Python.h" in str(refused.value)
    assert dev_packages()[0] in refused.value.fix and "fix:" not in str(refused.value)
    # a cached helper can load without a fresh compile: the refusal says so instead of claiming a failure,
    # and still says uncached launchers need the headers
    assert "cached native helper" in str(refused.value) and "uncached launchers" in str(refused.value)
    assert (
        refused.value.code == ENVIRONMENT
    )  # a compiler this host lacks: install it, do not change the command
    _refuse_unbuildable_triton([torch.device("cpu")], lambda _: None)


def test_a_cuda_build_refuses_a_driver_too_old_for_torch_before_any_work(monkeypatch):
    """torch for CUDA 13 on a CUDA 12 driver: the build downloaded the model, then stopped on a traceback. The
    build's `CUDA driver` guard (`checks.refuse_build`) refuses it before any work."""
    asked = []

    def problem(*, verdict_only=False):
        asked.append(verdict_only)
        return ("the driver supports CUDA 12.8, but torch 2.14.0+cu130 is built for CUDA 13.0", "upgrade it")

    monkeypatch.setattr(checks, "cuda_driver_problem", problem)
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "cuda_driver", lambda: (12080, ""))
    monkeypatch.setattr(checks, "card_health_problem", lambda ordinals=None: None)
    checks.refuse_build([])  # a CPU build does not ask the driver
    assert not asked
    with pytest.raises(Refused) as refused:
        checks.refuse_build([0])
    assert str(refused.value).startswith("the driver supports CUDA 12.8")
    assert refused.value.fix == "upgrade it" and refused.value.code == ENVIRONMENT
    assert asked == [True]  # no libcuda at all is left to torch: CPU hosts hand the build fake CUDA devices

    monkeypatch.setattr(checks, "cuda_driver_problem", lambda **k: None)
    checks.refuse_build([0])


def engine_with_a_card(tmp_path, monkeypatch) -> None:
    """An engine start on a machine with a CUDA torch and a card, whose driver guard holds."""
    monkeypatch.setattr(checks, "refuse_serve", lambda: None)
    monkeypatch.setattr(checks, "torch_cuda", lambda: ("2.14.0+cu130", "13.0"))
    monkeypatch.setattr(checks, "_no_cards_to_judge", lambda ctx: None)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))  # no driver built in it yet


def test_an_engine_start_without_python_headers_refuses_before_the_engine(tmp_path, monkeypatch):
    """A venv on a system Python without ``Python.h`` passed the engine start, which then died in Triton's
    driver compile after the model load; `vllm_environ` now runs the build's compile first."""
    skip_without_triton()
    from qda_quant.output import vllm_env

    engine_with_a_card(tmp_path, monkeypatch)
    monkeypatch.setattr(checks, "_python_include", lambda: empty_include(tmp_path, monkeypatch))
    entered = []
    with pytest.raises(Refused) as refused, vllm_env.vllm_environ({}):
        entered.append(True)
    assert not entered
    assert "Python.h" in str(refused.value) and "--managed-python" in refused.value.fix
    assert "fix:" not in str(refused.value) and refused.value.code == ENVIRONMENT


def test_an_engine_start_without_a_c_compiler_refuses_before_the_engine(tmp_path, monkeypatch):
    """No compiler on ``PATH``: vLLM's inductor stopped 112 s in on Qwen3-0.6B; the start refuses first."""
    skip_without_triton()
    from qda_quant.output import vllm_env

    engine_with_a_card(tmp_path, monkeypatch)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no gcc, clang or cc
    monkeypatch.setattr(vllm_env, "tool_path", lambda: str(tmp_path))
    with pytest.raises(Refused) as refused, vllm_env.vllm_environ({}):
        pass
    assert str(refused.value).startswith("C compiler: not found") and refused.value.fix == "install gcc"


def test_a_driver_triton_already_built_in_the_engines_cache_is_not_compiled_again(tmp_path, monkeypatch):
    """A warm ``TRITON_CACHE_DIR`` holds the driver, which loads with no compiler and no headers: on Qwen3-0.6B
    a reference built so with neither on the machine. The engine start does not refuse it."""
    skip_without_triton()
    import sysconfig

    from qda_quant.output import vllm_env
    from triton.backends.nvidia import driver
    from triton.runtime import build as triton_build

    engine_with_a_card(tmp_path, monkeypatch)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no compiler: only the cache can make the start pass
    monkeypatch.setattr(vllm_env, "tool_path", lambda: str(tmp_path))
    with pytest.raises(Refused):
        checks.refuse_engine_compiler()
    try:
        config = {
            "language": ["c"],
            "library_dirs": driver.library_dirs(),
            "include_dirs": driver.include_dirs,
            "libraries": driver.libraries,
            "ccflags": None,
        }
    except Exception:  # noqa: BLE001
        pytest.skip("no libcuda.so.1: Triton cannot name its driver's build here")
    source = Path(driver.dirname, "driver.c").read_bytes()
    name = "cuda_utils" + sysconfig.get_config_var("EXT_SUFFIX")
    triton_build._get_cache_manager(source, config).put(b"built", name, binary=True)
    checks.refuse_engine_compiler()
    with vllm_env.vllm_environ({}):
        pass


@pytest.mark.parametrize(
    "case",
    [
        "no card",
        "cpu torch",
        "timed out",
    ],
)
def test_an_engine_start_is_not_refused_without_a_verdict(case, tmp_path, monkeypatch):
    """No card or a CPU torch: no driver to build. A compile that timed out twice says nothing about the machine."""
    from qda_quant.doctor import Check

    engine_with_a_card(tmp_path, monkeypatch)
    budgets = []

    def check(deadline=None, include=None, budget=None):
        budgets.append(budget)
        detail = "gcc cannot build Triton's CUDA driver: timed out after 2.5 s"
        return Check("C compiler", False, detail, "x", timed_out=True)

    monkeypatch.setattr(checks, "check_triton_build", check)
    if case == "no card":
        monkeypatch.setattr(checks, "_no_cards_to_judge", lambda ctx: "no card to judge")
    elif case == "cpu torch":
        monkeypatch.setattr(checks, "torch_cuda", lambda: None)
    logged = []
    checks.refuse_engine_compiler(logged.append)
    assert budgets == ([None, checks._RETRY_BUDGET_S] if case == "timed out" else [])
    assert len(logged) == (1 if case == "timed out" else 0)


def slow_cc(tmp_path, monkeypatch, script: str) -> None:
    """``CC`` is a shell script, the only compiler on ``PATH``; the check's budgets are cut to a fraction."""
    sleep = shutil.which("sleep")
    cc = tmp_path / "slow-cc"
    cc.write_text(f"#!/bin/sh\n{script.replace('sleep', sleep)}\n")
    cc.chmod(0o755)
    monkeypatch.setenv("CC", str(cc))
    monkeypatch.setenv("PATH", str(tmp_path))  # no other gcc, clang or cc to try
    (tmp_path / "libcuda.so.1").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))  # no ldconfig
    monkeypatch.setattr(checks, "_BUDGET_S", 0.5)  # an attempt gets 0.25 s
    monkeypatch.setattr(checks, "_RETRY_BUDGET_S", 3.0)  # the retry's attempt gets 1.5 s


@pytest.mark.parametrize(
    ("script", "outcome"),
    [
        ("sleep 0.7", "builds"),
        ("sleep 30", "no verdict"),
        ("sleep 0.7; echo 'cc1: error: /net/include: Connection timed out' >&2; exit 1", "refused"),
    ],
)
def test_a_compiler_that_outlasts_the_check_is_checked_again_with_more_time(
    tmp_path, monkeypatch, script, outcome
):
    """A gcc on a network filesystem took over 2.5 s on a busy node, and the build was refused though that gcc
    built the driver the minute before. A timeout runs the check once more with `_RETRY_BUDGET_S`: a slow
    compiler then builds, a hung one is no verdict (one line, the build goes on), and a compile error is refused
    even when its text says "timed out"."""
    skip_without_triton()
    slow_cc(tmp_path, monkeypatch, script)
    logged = []
    started = time.monotonic()
    if outcome == "refused":
        with pytest.raises(Refused) as refused:
            _refuse_unbuildable_triton([torch.device("cuda")], logged.append)
        assert "Connection timed out" in str(refused.value) and refused.value.code == ENVIRONMENT
        assert refused.value.fix == "install a gcc that can build Triton's driver" and logged == []
    else:
        _refuse_unbuildable_triton([torch.device("cuda")], logged.append)
        if outcome == "builds":
            assert logged == [f"preflight: Triton's CUDA driver builds with {tmp_path / 'slow-cc'}"]
        else:
            assert (
                len(logged) == 1 and "did not finish in 3 s" in logged[0] and "timed out after" in logged[0]
            )
    assert time.monotonic() - started < 0.5 + 3.0 + 1.0
