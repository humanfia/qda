"""The lead's acceptance pair, through the whole judge: a per-layer mixed W8A16/W4A16 checkpoint is admitted
under the task's bpw budget, and one over it is refused with its bpw printed.

The judge is configured as a real search is (anchor, format reference and menu, budget), not with
`allow_any_format`: the candidate passes `files_problem` (anchor-identical model files and config), the
menu's engine facts (`format_problem`) and `judge.budget_bpw`, in that order, before it is queued.
"""

import json
from pathlib import Path

import pytest
from qda_agent.judge.checks.budget import bpw
from safetensors_kit import write_safetensors
from test_judge import Rig

CONFIG = {"model_type": "qwen3", "hidden_size": 128, "num_hidden_layers": 2}
L0, L1 = "model.layers.0.self_attn", "model.layers.1.self_attn"
MODULES = (f"{L0}.q_proj", f"{L0}.k_proj", f"{L1}.q_proj", f"{L1}.k_proj")


def group(bits: int, targets: list[str]) -> dict:
    """One config group: `targets` stored at `bits` a weight, group 128, symmetric."""
    return {
        "targets": targets,
        "weights": {"num_bits": bits, "group_size": 128, "strategy": "group", "type": "int"},
    }


def qconfig(*groups: dict) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {f"g{i}": g for i, g in enumerate(groups)},
        "ignore": ["model.embed_tokens"],
    }


def model_dir(d: Path, tensors: dict, quant: dict | None = None) -> Path:
    """A tiny checkpoint: the same config (with a quantization config when quantized) and tokenizer."""
    d.mkdir(parents=True)
    cfg = {**CONFIG, "quantization_config": quant} if quant else dict(CONFIG)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return d


def bf16(modules) -> dict:
    return {f"{m}.weight": ("BF16", (128, 128), None) for m in modules}


def packed(module: str, bits: int) -> dict:
    """A module's stored parts: int`bits` group-128 codes, their bf16 scales and the shape metadata."""
    return {
        f"{module}.weight_packed": ("I32", (128, 128 * bits // 32), None),
        f"{module}.weight_scale": ("BF16", (128, 1), None),
        f"{module}.weight_shape": ("I64", (2,), None),
    }


@pytest.fixture
def search(tmp_path):
    """The original (anchor) and a uniform W4A16 reference that also serves as candidate 0."""
    anchor = model_dir(tmp_path / "anchor", bf16(MODULES))
    tensors = {}
    for m in MODULES:
        tensors.update(packed(m, 4))
    ref = model_dir(tmp_path / "ref", tensors, qconfig(group(4, ["Linear"])))
    return Rig(
        tmp_path,
        anchor=anchor,
        baselines=[ref],
        format_reference=ref,
        format_menu=["bf16", "w8a16", "w4a16"],
        budget_bpw=6.5,
    ), anchor


def test_a_per_layer_mixed_w8a16_w4a16_candidate_is_admitted_under_budget(tmp_path, search):
    r, anchor = search
    # layer 0 at 4 bits, layer 1 at 8: the menu admits both formats in one checkpoint, per layer.
    tensors = {
        **packed(MODULES[0], 4),
        **packed(MODULES[1], 4),
        **packed(MODULES[2], 8),
        **packed(MODULES[3], 8),
    }
    quant = qconfig(group(4, [r"re:model\.layers\.0\..*"]), group(8, [r"re:model\.layers\.1\..*"]))
    mixed = model_dir(r.work / "mixed", tensors, quant)
    assert bpw(mixed, anchor) == pytest.approx((2 * (4 + 16 / 128) + 2 * (8 + 16 / 128)) / 4)
    rep = r.call("handin", "mixed")
    assert rep.exit == 0, rep.stderr
    assert r.host.handins()[0].valid is None  # admitted: its serving check is queued


def test_a_candidate_over_budget_is_refused_with_its_bpw(tmp_path, search):
    r, anchor = search
    # layer 0 at 4 bits, layer 1 kept bf16 (16 bits a weight): 10.0625 bpw, over the budget.
    tensors = {**packed(MODULES[0], 4), **packed(MODULES[1], 4), **bf16(MODULES[2:])}
    quant = qconfig(group(4, [r"re:model\.layers\.0\..*"]))
    heavy = model_dir(r.work / "heavy", tensors, quant)
    assert bpw(heavy, anchor) == pytest.approx((2 * (4 + 16 / 128) + 2 * 16) / 4)
    rep = r.call("handin", "heavy")
    assert rep.exit == 2
    assert "10.062 bits per quantized weight" in rep.stderr and "6.5 bpw" in rep.stderr
    assert r.host.handins() == [] and not any(r.be.launched)
