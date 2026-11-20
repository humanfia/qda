"""A GGUF may solve a routed bank's down projections apart from its gate/up; a fused-bank output may not.

llama.cpp stores ``ffn_gate_exps``, ``ffn_up_exps`` and ``ffn_down_exps`` as three tensors, each with its own
type, so a build can spend its extra bits on down alone (the allocation the 35B vendor census shows). vLLM and
SGLang load the bank as one fused module with one scheme, so the same rule is refused for their outputs rather
than written and mis-served.
"""

from __future__ import annotations

import json

import pytest
from qda_quant.config import BuildConfig
from qda_quant.gguf import rule
from qda_quant.step import Refused

from .test_experts import _source

#: The down parameter of the tiny model's one routed bank: the key a rule names to store it apart.
DOWN = "model.layers.0.mlp.experts.down_proj"
Q6K, Q4K, W8 = "w6a16-sym-g16", "w4a16-asym-g32", "w8a16-asym-g32"
#: gate/up stay Q4_K and down goes to Q6_K: what the vendor census spends its extra routed bytes on.
TYPES = [
    {"pattern": r"re:blk\.\d+\.(attn_q|attn_k|attn_v|attn_output)\.weight$", "type": "Q4_K"},
    {"pattern": r"re:blk\.\d+\.ffn_(gate|up)_exps\.weight$", "type": "Q4_K"},
    {"pattern": r"re:blk\.\d+\.ffn_down_exps\.weight$", "type": "Q6_K"},
]


def _tiny(tmp_path, monkeypatch):
    """The tiny MoE checkpoint and template, with the mix taking each tensor's type from the rules below.

    `qwen3moe` is not one of the architectures the Q4_K_M rule is ported for, and which types that rule would
    choose is not what these tests are about: every type here is named explicitly by `tensor_types`.
    """
    monkeypatch.setattr(
        rule, "q4_k_m_types", lambda arch, n, tensors: {name: kind for name, _, kind in tensors}
    )
    return _source(tmp_path)


def _config(tmp_path, src, calib, template, out, *, overrides=(), exporter=None, types=TYPES, grid="q4k"):
    """A tiny MoE build whose routed down bank is stored Q6_K and whose gate/up stay Q4_K."""
    return BuildConfig(
        model=str(src),
        format=Q4K,
        grid=grid,
        devices=["cpu"],
        steps=[{"gptq": {"impl": "torch"}}],
        calib={"file": {"path": str(calib), "samples": 1, "seqlen": 8}},
        format_overrides=[{"pattern": p, "format": f} for p, f in overrides],
        exporter=exporter or {"gguf": {"template": str(template), "tensor_types": list(types)}},
        out=str(tmp_path / out),
    )


def test_a_routed_banks_down_projections_carry_their_own_solved_codes(tmp_path, monkeypatch):
    """The down tensor holds the build's Q6_K solution while gate/up hold Q4_K codes of the same build.

    Three things could each leave this feature a no-op and still write a loadable file: the planner could store
    the whole bank in one format, the writer could drop the down codes and round the template instead, or it
    could keep solving down as Q4_K. The bytes tell them apart -- a solved tensor differs from llama.cpp's own
    fit of the original weight, which is what the plain build writes and what `rounded_from_template` reports.
    """
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import Template

    src, calib, template = _tiny(tmp_path, monkeypatch)
    plain = _config(tmp_path, src, calib, template, "plain").build(log=lambda _: None)
    split = _config(tmp_path, src, calib, template, "split", overrides=[(DOWN, Q6K)]).build(
        log=lambda _: None
    )

    a, b = Template(plain / "model-Q4_K_M.gguf"), Template(split / "model-Q4_K_M.gguf")
    down = "blk.0.ffn_down_exps.weight"
    assert b.tensors[down].tensor_type == rule.Q6_K, "the down bank is not stored Q6_K"
    assert all(b.tensors[f"blk.0.ffn_{p}_exps.weight"].tensor_type == rule.Q4_K for p in ("gate", "up"))
    assert b.raw(down) != a.raw(down), "the down bank holds the template's rounding, not the build's solution"
    assert all(a.raw(n) == b.raw(n) for n in a.tensors if n != down), (
        "storing the down projections apart changed a tensor outside them"
    )

    rounded = json.loads((split / "build.json").read_text())["gguf"]["rounded_from_template"]
    assert down not in rounded, "the solved down bank is still recorded as rounded from the template"
    assert down in json.loads((plain / "build.json").read_text())["gguf"]["rounded_from_template"]


def test_the_planned_bytes_are_the_bytes_a_split_bank_build_writes(tmp_path, monkeypatch):
    """The plan sizes the file before it is built, with one bank's projections in two types.

    The payload is read back with upstream's own reader, so this cannot pass by repeating qda's arithmetic.
    """
    gguf = pytest.importorskip("gguf")

    src, calib, template = _tiny(tmp_path, monkeypatch)
    cfg = _config(tmp_path, src, calib, template, "out", overrides=[(DOWN, Q6K)])
    plan = cfg.plan()
    assert plan.output_bytes_exact is True
    out = cfg.build(log=lambda _: None) / "model-Q4_K_M.gguf"
    assert plan.output_bytes == out.stat().st_size
    assert plan.weight_bytes == sum(int(t.n_bytes) for t in gguf.GGUFReader(out).tensors)


def test_routed_gate_up_and_down_can_store_three_distinct_types(tmp_path, monkeypatch, ggml_lib):
    """A fused solve may use separate stored gate/up types while down retains its own Q6_K solution."""
    from qda_quant.gguf.file import Template

    src, calib, template = _tiny(tmp_path, monkeypatch)
    types = [
        *TYPES,
        {"pattern": "blk.0.ffn_gate_exps.weight", "type": "Q5_K"},
        {"pattern": "blk.0.ffn_up_exps.weight", "type": "IQ4_XS"},
    ]
    cfg = _config(
        tmp_path,
        src,
        calib,
        template,
        "split",
        overrides=[(DOWN, Q6K)],
        exporter={"gguf": {"template": str(template), "tensor_types": types, "libggml": str(ggml_lib)}},
    )
    out = cfg.build(log=lambda _: None)
    stored = Template(out / "model-Q4_K_M.gguf")
    names = [f"blk.0.ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
    assert [stored.tensors[n].tensor_type for n in names] == [rule.Q5_K, rule.IQ4_XS, rule.Q6_K]
    rounded = json.loads((out / "build.json").read_text())["gguf"]["rounded_from_template"]
    assert all("codes dropped" in rounded[n] for n in names[:2])
    assert names[2] not in rounded


def test_an_output_that_loads_a_bank_fused_refuses_a_split_bank(tmp_path, monkeypatch):
    """compressed-tensors serves one scheme per bank, so a rule it cannot express must be refused.

    The format here is one vLLM does serve, so the refusal can only come from the split itself: a down-only
    rule in a format no vLLM kernel serves is already refused by the target, which would hide this.
    """
    src, calib, template = _tiny(tmp_path, monkeypatch)
    cfg = _config(
        tmp_path,
        src,
        calib,
        template,
        "ct",
        overrides=[(DOWN, W8)],
        exporter="compressed_tensors",
        grid="absmax",  # the GGUF grids store one record layout each; W8A16 is solved on the plain grid
    )
    with pytest.raises(Refused, match="one fused module with one scheme"):
        cfg.plan()


def test_a_rule_naming_the_whole_bank_still_stores_it_in_one_format(tmp_path, monkeypatch):
    """The down projections are a unit a rule may name, not one that every bank rule now splits off.

    A pattern matching the bank matches its down parameter too; if that counted as splitting the bank, every
    whole-bank override that exists today would start refusing for vLLM and SGLang.
    """
    src, calib, template = _tiny(tmp_path, monkeypatch)
    plan = _config(
        tmp_path,
        src,
        calib,
        template,
        "whole",
        overrides=[("re:model.layers.0.mlp.experts", W8)],
        exporter="compressed_tensors",
        grid="absmax",
    ).plan()
    assert [f.name for f in plan.formats.values()] == [W8], "the bank is not stored in one format"
    assert DOWN not in plan.formats, "a whole-bank rule recorded its down projections as stored apart"
