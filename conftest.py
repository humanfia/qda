"""Pytest setup shared by the three packages' tests.

Plain ``pytest`` never reaches a GPU: CUDA is hidden unless the run selects GPU tests (``-m gpu``), and such a
run must be started with ``CUDA_VISIBLE_DEVICES`` already set to the cards it may use, e.g.
``CUDA_VISIBLE_DEVICES=0 pytest -m gpu``. This root file is loaded before any package's tests are imported, so
a module that asks ``torch.cuda`` at import time sees the hidden cards too.
"""

from __future__ import annotations

import os


def _selects_gpu(markexpr: str) -> bool:
    m = " ".join(markexpr.split())
    return "gpu" in m and "not gpu" not in m


def pytest_configure(config):
    if not _selects_gpu(config.option.markexpr or ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # read by torch at CUDA init, which no test has reached yet


def pytest_collection_modifyitems(config, items):
    if _selects_gpu(config.option.markexpr or "") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError(
            "GPU tests need CUDA_VISIBLE_DEVICES set to the cards they may use, e.g. CUDA_VISIBLE_DEVICES=0"
        )
