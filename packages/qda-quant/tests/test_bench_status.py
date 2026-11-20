"""The speed-bench row's status names why a cell did not finish.

The matrix reads a row's ``status`` to tell a cell the planner refused from one that ran out of memory and one
that failed in the environment; a single ``unsupported`` for all three hides the difference. The driver is a
script, so it is loaded by path here rather than imported from an installed package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from qda_quant.step import Refused

BENCH_CELL = Path(__file__).resolve().parents[3] / "bench" / "bench_cell.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("bench_cell_under_test", BENCH_CELL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_planner_refusal_is_unsupported(driver):
    status, error = driver.failure(Refused("step 'awq' re-runs layers on the calibration set"))
    assert status == "unsupported"
    assert "re-runs layers" in error


def test_a_cuda_oom_is_oom(driver):
    status, error = driver.failure(
        torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20 MiB.")
    )
    assert status == "oom"
    assert error.startswith("CUDA out of memory")


def test_any_other_failure_is_failed(driver):
    status, error = driver.failure(RuntimeError("checkpoint writer failed"))
    assert status == "failed"
    assert error == "RuntimeError: checkpoint writer failed"
