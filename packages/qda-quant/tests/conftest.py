"""Shared pytest setup of the qda-quant tests.

Options for the tests that need real models, data or a serving venv (each such test skips with the option it
needs named in the reason when that option is not given):

* ``--model-dir NAME=PATH`` (repeatable): a local Hugging Face model directory. Names used: ``qwen3-8b``
  (Qwen3-8B), ``qwen3-0.6b`` (Qwen3-0.6B), ``r1-llama-8b`` (DeepSeek-R1-Distill-Llama-8B), ``qwen3-30b-a3b``
  (Qwen3-30B-A3B), ``qwen3.5-9b`` (Qwen3.5-9B), ``qwen3.8-27b`` (Qwen3.8-27B), ``qwen3.6-35b-a3b``
  (Qwen3.6-35B-A3B).
* ``--calib PATH``: a chat calibration JSONL (``{"messages": [...]}`` rows).
* ``--calib-ids PATH``: the first rows of ``--calib`` already tokenized for ``qwen3-8b`` (``{"input_ids": ...}``
  rows, no header), the reference its rendering must reproduce.
* ``--vllm-python PATH``: the python of a venv with vLLM 0.33.0 (test_vllm_gpu.py).
* ``--ggml-lib PATH``: a built ``libggml-base`` from a llama.cpp checkout (gguf/test_kernel.py compares it with
  the port).
* ``--serve-case NAME=QUANT_DIR:BASE_DIR`` (repeatable) and ``--ref-python PATH``: intake/test_serve_humming.py.

CUDA is hidden from plain ``pytest`` by the repository's root ``conftest.py``.
"""

from __future__ import annotations

import gc
import sys
import sysconfig
from pathlib import Path

import pytest

# Multi-card tests send steps and writers defined in this test package to the card processes by pickle, as
# ``tests.<module>.<name>``. Under ``--import-mode=importlib`` (the root config) the package directory is not on
# sys.path, so a spawned card process could not import ``tests``. Spawned processes copy the parent's sys.path.
_PKG = str(Path(__file__).resolve().parents[1])
if _PKG not in sys.path:
    sys.path.append(_PKG)


def pytest_addoption(parser):
    """The options listed in the module docstring."""
    parser.addoption(
        "--model-dir", action="append", default=[], help="NAME=PATH of a local model (repeatable)"
    )
    parser.addoption("--calib", default=None, help="chat calibration JSONL ({'messages': ...} rows)")
    parser.addoption("--calib-ids", default=None, help="the first --calib rows tokenized for qwen3-8b")
    parser.addoption("--vllm-python", default=None, help="python of a venv with vLLM 0.33.0")
    parser.addoption("--ggml-lib", default=None, help="a built libggml-base from a llama.cpp checkout")
    parser.addoption("--serve-case", action="append", default=[], help="NAME=QUANT_DIR:BASE_DIR")
    parser.addoption(
        "--ref-python", default=None, help="python of a venv with transformers + compressed-tensors"
    )


def _model_dirs(config) -> dict[str, Path]:
    out = {}
    for spec in config.getoption("--model-dir"):
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise pytest.UsageError(f"--model-dir {spec!r}: expected NAME=PATH")
        out[name] = Path(path)
    return out


def pytest_configure(config):
    for name, path in _model_dirs(config).items():
        if not path.is_dir():
            raise pytest.UsageError(f"--model-dir {name}={path}: not a directory")
    for opt in ("--calib", "--calib-ids", "--vllm-python", "--ggml-lib"):
        value = config.getoption(opt)
        if value is not None and not Path(value).is_file():
            raise pytest.UsageError(f"{opt} {value}: not a file")


@pytest.fixture(autouse=True)
def _release_cards():
    """After each test, hand the cards' cached blocks back to the driver.

    torch's allocator keeps a finished test's blocks reserved in this process, and a later test that starts
    vLLM (in a child process or in this one) sizes its engine from the driver's free memory, so one GPU run
    would refuse to start an engine after a test that loaded a real model. Never initializes CUDA itself.
    """
    yield
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def real_model(request):
    """``real_model(NAME)``: the ``--model-dir NAME=PATH`` directory, or a skip when it is not given."""
    dirs = _model_dirs(request.config)

    def get(name: str) -> Path:
        if name not in dirs:
            pytest.skip(f"needs --model-dir {name}=PATH")
        return dirs[name]

    return get


def _file_option(request, opt: str) -> Path:
    value = request.config.getoption(opt)
    if value is None:
        pytest.skip(f"needs {opt} PATH")
    return Path(value)


@pytest.fixture
def cpu_build(monkeypatch) -> None:
    """Lets a GPU test build on ``devices=["cpu"]``, its CPU reference or a CPU-trained checkpoint.

    A run that selects GPU tests holds cards, and a build on a job with cards refuses card steps on CPU
    (`engine.build._refuse_cpu_steps`); these tests put them there on purpose.
    """
    import importlib

    build_module = importlib.import_module("qda_quant.engine.build")
    monkeypatch.setattr(build_module, "_refuse_cpu_steps", lambda devices, recipe: None)


@pytest.fixture
def calib_file(request) -> Path:
    """``--calib``: a chat calibration JSONL; skips when it is not given."""
    return _file_option(request, "--calib")


@pytest.fixture
def calib_ids_file(request) -> Path:
    """``--calib-ids``: the first ``--calib`` rows tokenized for qwen3-8b; skips when it is not given."""
    return _file_option(request, "--calib-ids")


@pytest.fixture
def vllm_python(request) -> Path:
    """``--vllm-python``: the python of a venv with vLLM 0.33.0; skips when it is not given."""
    return _file_option(request, "--vllm-python")


@pytest.fixture
def ggml_lib(request) -> Path:
    """``--ggml-lib``: a built ``libggml-base``; skips when it is not given."""
    return _file_option(request, "--ggml-lib")


@pytest.fixture(autouse=True)
def _release_cuda_cache(request):
    """Hand a GPU test's cached blocks back to the card when it ends. Without this the pytest process keeps
    whatever the heaviest test touched (test_solve's int32-boundary tests cache ~129 GiB) and a vLLM child
    spawned by a later test finds the card full."""
    yield
    if request.node.get_closest_marker("gpu") is None or "torch" not in sys.modules:
        return
    import gc

    import torch

    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture
def python_headers():
    """Skips when this interpreter has no ``Python.h``: a build's compiler check refuses before anything else."""
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        pytest.skip(f"no Python.h in {include}: make the venv from a uv-managed Python")


@pytest.fixture
def triton_driver(python_headers):
    """Skips where Triton's CUDA driver cannot link: a build that names cards compiles it before anything else,
    and the link needs ``libcuda.so.1``, which a host without an NVIDIA driver (a CPU CI runner) does not have."""
    from qda_quant import checks

    check = checks.check_triton_build()
    if not check.ok and "libcuda.so.1" in check.detail:
        pytest.skip("no libcuda.so.1 to link Triton's driver against (no NVIDIA driver on this host)")
