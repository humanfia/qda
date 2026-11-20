"""A class plan must state the bytes a class costs at each type, and the buckets that are never allocated."""

from __future__ import annotations

import numpy as np
import pytest
from qda_quant.gguf import classes as sensitivity
from qda_quant.gguf import file, rule

BF16 = rule.BF16
_H = 128  #: hidden width of the tiny model; 256 is one Q4_K/Q6_K super-block
_N = 3  #: blocks 0 and 1 are text layers; block 2 is the MTP block (``nextn.*``)

#: A tiny gated-delta-net hybrid: one full-attention and one linear-attention block, an MTP block, and a keep.
ROWS = [
    ("token_embd.weight", (256, _H), BF16),
    ("output.weight", (256, _H), BF16),
    ("blk.0.attn_norm.weight", (_H,), BF16),
    ("blk.0.attn_q.weight", (256, _H), BF16),
    ("blk.0.attn_k.weight", (128, _H), BF16),
    ("blk.0.attn_v.weight", (128, _H), BF16),
    ("blk.0.attn_output.weight", (_H, 256), BF16),
    ("blk.0.ffn_gate.weight", (256, _H), BF16),
    ("blk.0.ffn_up.weight", (256, _H), BF16),
    ("blk.0.ffn_down.weight", (_H, 256), BF16),
    ("blk.1.attn_qkv.weight", (512, _H), BF16),
    ("blk.1.attn_gate.weight", (256, _H), BF16),
    ("blk.1.ssm_out.weight", (_H, 512), BF16),
    ("blk.1.ssm_alpha.weight", (_H, 4), BF16),
    ("blk.1.ssm_beta.weight", (_H, 4), BF16),
    ("blk.1.ffn_gate.weight", (256, _H), BF16),
    ("blk.1.ffn_up.weight", (256, _H), BF16),
    ("blk.1.ffn_down.weight", (_H, 256), BF16),
    ("blk.2.nextn.eh_proj.weight", (256, _H), BF16),
    ("blk.2.nextn.embed_tokens.weight", (256, _H), BF16),
    ("blk.2.attn_q.weight", (256, _H), BF16),
    ("blk.2.ffn_down.weight", (_H, 256), BF16),
]


def _plan(rows=ROWS):
    return sensitivity.plan_from_tensors("qwen35", _N, rows)


def test_every_class_has_its_tensors_and_no_mtp_tensor() -> None:
    p = _plan()
    assert list(p.classes) == list(sensitivity.CLASSES)  # all 14 are present in this fixture
    assert p.classes["attn_q"].tensors == 1  # the MTP block's attn_q is not the class's
    assert p.classes["ffn_down"].tensors == 2  # blocks 0 and 1
    assert p.classes["attn_qkv"].params == 512 * _H
    assert p.classes["ssm_alpha"].params == 4 * _H
    assert p.excluded["mtp"].tensors == 2  # blk.2.attn_q, blk.2.ffn_down
    assert p.excluded["mtp_eh"].tensors == 2  # eh_proj, embed_tokens (both quantized)
    assert p.excluded["keeps"].tensors == 1  # attn_norm


def test_bytes_follow_the_rule_and_the_grid() -> None:
    p = _plan()
    c = p.classes["attn_qkv"]
    assert c.bf16_bytes == 2 * c.params
    assert c.bytes["Q4_K"] == file.row_bytes(rule.Q4_K, 512) * (512 * _H // 512)
    assert c.bytes["Q8_0"] == file.row_bytes(rule.Q8_0, 512) * (512 * _H // 512)
    # one width that is a Q8_0 block but not a Q4_K super-block: no Q4_K/Q6_K cell, an honest Q8_0 one
    s = p.classes["ssm_alpha"]
    assert s.bytes["Q4_K"] is None and s.bytes["Q6_K"] is None and s.bytes["Q5_K"] is None
    assert s.bytes["Q8_0"] == 4 * _H * 34 // 32
    # the rule: an untied output is Q6_K, the embedding Q4_K
    assert p.classes["output"].rule_bytes == 256 * _H * 210 // 256
    assert p.classes["token_embd"].rule_bytes == 256 * _H * 144 // 256
    assert p.total() == sum(c.rule_bytes for c in p.classes.values())
    assert p.total({c: "bf16" for c in p.classes}) == sum(c.bf16_bytes for c in p.classes.values())


def test_a_quantized_tensor_in_no_class_is_refused() -> None:
    rows = [*ROWS, ("blk.0.experts.weight", (256, _H), BF16)]
    with pytest.raises(ValueError, match="in no class"):
        _plan(rows)


def test_the_plan_is_deterministic() -> None:
    assert _plan() == _plan()


def test_a_template_file_plans_like_its_rows(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_block_count(_N)
    for name, ne, _ in ROWS:
        # gguf stores ne[0] = the numpy array's last axis, so the array is the reverse of `ne`
        data = np.zeros(tuple(reversed(ne)), dtype=np.int16)  # bf16 rows; values are not read
        w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    got = sensitivity.plan(path)
    assert got.arch == "qwen35" and got.n_layer == _N
    assert got.classes == _plan().classes
    # the plan's file size is the writer's exact layout, header and alignment included
    types = rule.q4_k_m_types(got.arch, got.n_layer, ROWS)
    layout = file.Layout(file.Template(path), types)
    assert got.header_bytes == len(layout.head)
    assert got.file_total() == layout.total
    assert got.total() + sum(c.rule_bytes for c in got.excluded.values()) == layout.payload


def test_total_refuses_a_key_that_names_no_class() -> None:
    with pytest.raises(ValueError, match="not an allocatable class"):
        _plan().total({"attn_typo": "Q8_0"})
    # the fixed buckets are never an allocation
    with pytest.raises(ValueError, match="not an allocatable class"):
        _plan().total({"mtp": "Q8_0"})


def test_importing_the_module_does_not_import_torch() -> None:
    import subprocess
    import sys

    code = "import sys; import qda_quant.gguf.classes; assert 'torch' not in sys.modules"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def _write_template(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_block_count(_N)
    for name, ne, _ in ROWS:
        data = np.zeros(tuple(reversed(ne)), dtype=np.int16)
        w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_a_class_allocation_expands_to_exact_tensor_rules(tmp_path) -> None:
    path = _write_template(tmp_path)
    rules = sensitivity.expand(path, {"attn_q": "Q6_K", "ffn_down": "Q5_K", "token_embd": "bf16"})
    named = {r["pattern"]: r["type"] for r in rules}
    assert named["blk.0.attn_q.weight"] == "Q6_K"
    assert named["blk.0.ffn_down.weight"] == "Q5_K" and named["blk.1.ffn_down.weight"] == "Q5_K"
    assert named["token_embd.weight"] == "BF16"
    assert "blk.2.attn_q.weight" not in named  # the MTP block is never allocated
    assert "blk.2.nextn.eh_proj.weight" not in named
    assert not any(name.endswith("attn_norm.weight") for name in named)  # the rule keeps it float
    assert all(set(r) == {"pattern", "type"} for r in rules)
    with pytest.raises(ValueError, match="no tensor of this class"):
        sensitivity.expand(path, {"attn_typo": "Q4_K"})


def test_the_cli_states_the_rule_sizes_and_expands_the_allocation(tmp_path, capsys) -> None:
    import json

    from qda_quant import cli

    path = _write_template(tmp_path)
    assert cli.main(["sensitivity", "plan", str(path), "--json"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    totals = lines[-1]["totals"]
    planned = sensitivity.plan(path)
    assert totals["rule_file_bytes"] == planned.file_total()
    assert totals["bf16_file_bytes"] == planned.file_total({c: "bf16" for c in planned.classes})
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"picks": {"attn_q": "Q4_K"}}))
    assert cli.main(["sensitivity", "types", str(path), str(picks)]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert {r["pattern"] for r in rules} == {"blk.0.attn_q.weight"}
    assert all(r["type"] == "Q4_K" for r in rules)


def test_the_cli_takes_a_budgeted_table_result_as_a_types_input(tmp_path, capsys) -> None:
    import json

    from qda_quant import cli

    path = _write_template(tmp_path)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"class": "attn_q", "type": "Q4_K", "d_kl": 0.001}]))
    assert cli.main(["sensitivity", "table", str(path), str(rows), "--json"]) == 0
    # the smallest file the measured cells allow: a budget at it fits, bf16's larger file does not
    budget = json.loads(capsys.readouterr().out)[0]["file_bytes"]
    assert cli.main(["sensitivity", "table", str(path), str(rows), "--budget-bytes", str(budget), "--json"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)[0]["picks"]["attn_q"] == "Q4_K"
    allocation = tmp_path / "allocation.json"
    allocation.write_text(printed)  # what prepare writes: the table's own output, verbatim
    assert cli.main(["sensitivity", "types", str(path), str(allocation)]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert {"pattern": "blk.0.attn_q.weight", "type": "Q4_K"} in rules


def test_the_cli_refuses_a_frontier_as_a_single_allocation(tmp_path, capsys) -> None:
    import json

    from qda_quant import cli

    path = _write_template(tmp_path)
    frontier = tmp_path / "frontier.json"
    frontier.write_text(json.dumps([{"picks": {"attn_q": "Q4_K"}}, {"picks": {"attn_q": "Q6_K"}}]))
    assert cli.main(["sensitivity", "types", str(path), str(frontier)]) == 2
    err = capsys.readouterr().err
    assert "2 allocations" in err and "name --budget-bytes" in err
