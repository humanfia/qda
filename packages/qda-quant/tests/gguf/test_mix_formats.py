"""The Q4_K_M mix's own stored types decide the solve formats, before the solve runs.

A plain GGUF recipe names no stored types, so every unit was solved at the row's 4-bit format while the mix
stored some of them as Q6_K: the export then dropped exactly those codes and wrote the template's Q6_K
``llama-quantize`` values instead. `qda_quant.gguf.candidates.mix_formats` reads the mix, and the plan binds
the units it promotes before the solve, so a recipe that names nothing keeps its codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
from qda_quant import REGISTRY, Recipe
from qda_quant.arch import describe
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.engine.planning import compile_plan
from qda_quant.format import Format
from qda_quant.gguf import candidates, file
from qda_quant.gguf.export import GGUF
from qda_quant.target import Target

from ..tiny import SMALL
from .test_export import _source
from .test_gdn_744 import GDN
from .test_gdn_744 import _template as _gdn_template

#: A 32-layer Llama wide enough for the Q4_K_M rule's Q6_K promotions (rows of 256 values).
WIDE32 = {
    **SMALL,
    "hidden_size": 256,
    "intermediate_size": 512,
    "vocab_size": 256,
    "num_hidden_layers": 32,
    # every row a whole 256-value super-block: q/k/v/o and the FFN, so no tensor falls back to Q5_0
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
}

#: Qwen3.8-27B's shape of a Q4_K_M file: 64 blocks (gated delta net, full attention every fourth), half of
#: whose ``ffn_down`` llama.cpp promotes to Q6_K.
QWEN38 = {
    **GDN,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 64,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
}


def _recipe(calib: Path) -> Recipe:
    return Recipe(
        steps=[REGISTRY["gptq"](grid="q4k", impl="torch")],
        calib=FileSource(path=str(calib), samples=1, seqlen=8),
    )


def test_a_qwen38_shape_solves_the_mixs_q6k_tensors_in_q6k(tmp_path: Path) -> None:
    """Qwen3.8-27B's 64-block mix promotes 32 ``ffn_down``; each is solved there, not at the row's 4-bit grid."""
    src = tmp_path / "m"
    cfg = transformers.Qwen3_5TextConfig(**QWEN38)
    cfg._attn_implementation = "eager"
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=None).eval()
    model.save_pretrained(src)
    template = tmp_path / "t.gguf"
    _gdn_template(src, template, cfg)

    rules, unmet = candidates.mix_formats(file.Template(template), describe(model, verify=False))
    down = {r["pattern"] for r in rules if r["pattern"].endswith(".mlp.down_proj")}
    assert len(down) == 32, sorted(
        down
    )  # the mix's counter promotes every other block, as llama-quantize does
    assert {r["format"] for r in rules if r["pattern"] in down} == {"w6a16-sym-g16"}
    # the fused units are the ones no one format covers: the mix stores part of each at Q6_K (the issue's
    # stated limit), so they are named as unmet rather than silently solved at the wrong grid
    assert unmet, "a 64-block qwen35 mix stores part of its fused qkv units at Q6_K"
    assert all(len(kinds) > 1 for kinds in unmet.values()), unmet
    assert not [uid for uid in unmet if "mlp.down_proj" in uid], unmet


def test_the_plan_binds_the_mixs_q6k_units_before_the_solve(tmp_path: Path) -> None:
    """A recipe naming no stored types still plans the mix's Q6_K units at ``w6a16-sym-g16``."""
    src, calib, template, checkout, _n = _source(tmp_path, WIDE32)
    plan = compile_plan(
        str(src),
        _recipe(calib),
        Format.parse("w4a16-asym-g32"),
        Target.parse("llama_cpp"),  # the output the exporter writes is served by llama.cpp
        [torch.device("cpu")],
        exporter=GGUF(llama_cpp=str(checkout), template=str(template)),
    )
    bound = {k for k, f in plan.formats.items() if f.name == "w6a16-sym-g16"}
    assert bound, "the mix promotes tensors; the plan bound none"
    assert bound == set(plan.stored_formats), (sorted(bound), sorted(plan.stored_formats))
    assert all(k.endswith((".mlp.down_proj", ".self_attn.v_proj")) for k in bound)
    assert set(plan.stored_formats.values()) == {"w6a16-sym-g16"}


def test_a_plain_recipe_keeps_the_mixs_q6k_codes(tmp_path: Path) -> None:
    """Only the output head still loses its codes: no promoted layer tensor of a plain 32-layer build is dropped."""
    src, calib, template, checkout, _n = _source(tmp_path, WIDE32)
    out = build(
        str(src),
        _recipe(calib),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        exporter={"gguf": {"llama_cpp": str(checkout), "template": str(template)}},
        log=lambda _: None,
    )
    record = json.loads((out / "build.json").read_text())["gguf"]
    dropped = record["codes_dropped"]
    # every unit `candidates.unit_tensors` maps is solved where the mix stores it: no down projection is
    # dropped any more. The remaining drops are the fused q/k/v tensors, which the template stores as one
    # ``attn_qkv.weight`` -- no unit of the graph owns that tensor, so no override can name it (the issue's
    # stated limit: a fused q/k/v unit needs one format, and the mix gives it two).
    assert dropped, "the mix promotes the fused qkv; its v half is still dropped"
    assert all("down_proj" not in d["module"] for d in dropped.values()), dropped
    assert all("self_attn" in d["module"] for d in dropped.values()), dropped
    assert not [
        n for n, why in record["rounded_from_template"].items() if "down_proj" in n and "codes dropped" in why
    ]
