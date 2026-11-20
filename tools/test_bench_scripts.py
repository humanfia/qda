"""Repo-level checks: the bench scripts the docs tell a user to run still import."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gguf_to_float_imports() -> None:
    """``docs/formats.md`` scores a GGUF through this script; a moved import must fail here, not at run time."""
    pytest.importorskip("gguf")
    path = ROOT / "bench" / "gguf_to_float.py"
    spec = importlib.util.spec_from_file_location("bench_gguf_to_float", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # ImportError when qda_quant moves what the script imports
    assert hasattr(module, "main")
