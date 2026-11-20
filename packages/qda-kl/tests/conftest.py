import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # test_cli.py imports write_ref from test_topk.py

import pytest


@pytest.fixture(autouse=True)
def engine_compiler_holds(monkeypatch):
    """The engines here are stand-ins; the engine start's compile (`refuse_engine_compiler`) is qda-quant's test."""
    monkeypatch.setattr("qda_quant.checks.refuse_engine_compiler", lambda: None)
