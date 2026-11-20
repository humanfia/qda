"""Staged plans with scale_kd on real model configs (meta device, no weights, no GPU).

Needs ``--model-dir NAME=PATH`` for ``qwen3-8b``, ``qwen3.5-9b`` and ``qwen3.8-27b`` (see conftest.py); each case
skips when its model is not given. Tiny-model tests cannot catch what these do: there the teacher and the student
are one model, and a multimodal checkpoint's text-only skeleton has no ``architectures`` of its own.
"""

from __future__ import annotations

import json

import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.calib.sources import FileSource
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.recipe import Recipe
from qda_quant.step import Refused
from qda_quant.target import Target

CPU = [torch.device("cpu")]


def _recipe(tmp_path, *steps) -> Recipe:
    path = tmp_path / "calib.jsonl"
    path.write_text(
        "".join(json.dumps({"text": f"Row {i}: the quick brown fox jumps. " * 40}) + "\n" for i in range(4))
    )
    return Recipe(steps=list(steps), calib=FileSource(path=str(path), samples=4, seqlen=64))


@pytest.mark.parametrize("name", ["qwen3-8b", "qwen3.5-9b", "qwen3.8-27b"])
def test_staged_plan_on_a_real_config(name, real_model, tmp_path):
    """[gptq, scale_kd, scale_kd] (the fixed recipe's shape) plans, the teacher being the source itself."""
    src = str(real_model(name))
    kd = REGISTRY["scale_kd"](steps=1)
    plan = compile_plan(
        src, _recipe(tmp_path, REGISTRY["gptq"](), kd, kd), Format.parse("w4a16"), Target(), CPU
    )
    assert [s.name for s in plan.checkpoint_steps] == ["scale_kd", "scale_kd"]


@pytest.mark.parametrize("student, teacher", [("qwen3.5-9b", "qwen3-8b"), ("qwen3-8b", "qwen3.8-27b")])
def test_staged_plan_refuses_another_family_as_teacher(student, teacher, real_model, tmp_path):
    """A teacher of another model class is refused at plan time, naming both."""
    src, tdir = str(real_model(student)), str(real_model(teacher))
    kd = REGISTRY["scale_kd"](steps=1, teacher=tdir)
    with pytest.raises(Refused, match=r"scale_kd: teacher .* is .*, the student"):
        compile_plan(src, _recipe(tmp_path, REGISTRY["gptq"](), kd), Format.parse("w4a16"), Target(), CPU)


def test_teacher_as_a_hub_repo_id(real_model, tmp_path, monkeypatch):
    """A recipe may name the teacher by hub id (``org/name@rev``); it resolves through the HF cache like the source."""
    import importlib

    engine_planning = importlib.import_module("qda_quant.engine.planning")
    src = str(real_model("qwen3.5-9b"))
    monkeypatch.setattr(
        engine_planning, "local_source", lambda m, weights=True: src if str(m).startswith("Qwen/") else str(m)
    )
    kd = REGISTRY["scale_kd"](steps=1, teacher="Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    plan = compile_plan(src, _recipe(tmp_path, REGISTRY["gptq"](), kd), Format.parse("w4a16"), Target(), CPU)
    assert [s.name for s in plan.checkpoint_steps] == ["scale_kd"]
