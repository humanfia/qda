"""Every step of `qda_quant.steps` at the sm100 row's format, ``w4a4-nvfp4``.

The NVFP4 design note needs one matrix: each step either builds on a tiny model and writes a checkpoint whose
packed weights compressed-tensors' NVFP4 decoder reads, or `compile_plan` refuses it with a reason that names
NVFP4. A step whose outcome changes fails here, instead of crashing a build later. The checkpoint steps build
too: a dense packed NVFP4 checkpoint is read back (`Packed`, `load_student`).
"""

import json
import shutil
from pathlib import Path

import pytest
import torch
from qda_quant import Recipe, build
from qda_quant.step import Refused
from qda_quant.target import Target
from safetensors import safe_open

from ..engine_util import save_tiny_tokenizer, write_ids
from ..tiny import model

FORMAT = "w4a4-nvfp4"
TARGET = Target.parse("sm100")

#: steps that build at NVFP4, with the quantizing step a folding or weighting step needs to store codes
BUILD = {
    "gptq": [{"gptq": {"actorder": "static", "damp": 0.01}}],
    "gptq-fp4hmse": [{"gptq": {"actorder": "static", "damp": 0.01, "clip": "fp4hmse"}}],
    "rtn-fp4mse": [{"rtn": {"clip": "fp4mse"}}],
    "gptaq": [{"gptaq": {"alpha": 0.3}}],
    "rotate": [{"rotate": {}}, {"rtn": {}}],
    "permute": [{"permute": {}}, {"rtn": {}}],
    "guided": [{"guided": {}}, {"gptq": {}}],
    "fisher_trace": [{"fisher_trace": {}}, {"gptq": {}}],
}

#: steps that build on a packed NVFP4 checkpoint: `Packed` decodes the weights and `load_student` serves them,
#: dense or per-expert in a routed bank (``gate_up_proj`` / ``down_proj`` cut per 16 under the bank's scales).
#: `(steps, the packed base, its model kind)`.
BUILD_PACKED = {
    "soup": (lambda base: [{"soup": {"with": [str(base)]}}], "dense", "qwen3"),
    "scale_kd": (
        lambda base: [{"scale_kd": {"steps": 1, "batch_tokens": 64, "head_chunk": 16}}],
        "dense",
        "qwen3",
    ),
    "scale_kd-bank": (
        lambda base: [
            {"scale_kd": {"steps": 1, "batch_tokens": 64, "head_chunk": 16, "trainable": ["norms"]}}
        ],
        "bank",
        "qwen3_moe",
    ),
}

#: steps that cannot run at NVFP4 yet: the plan refuses, and the reason names NVFP4. `(steps, the source is the
#: packed base, the base model)`.
REFUSED = {
    "autoround": (lambda base: [{"autoround": {}}], False, "qwen3"),
    "gsq": (lambda base: [{"rtn": {}}, {"gsq": {}}], False, "qwen3"),
    "head": (lambda base: [{"head": {"format": FORMAT}}, {"rtn": {}}], False, "qwen3"),
}


def _recipe(d: Path, steps: list[dict], kind: str = "qwen3") -> Recipe:
    m = model(kind)
    m.save_pretrained(d / "m")
    sha = save_tiny_tokenizer(d / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(d / "c.jsonl", [torch.randint(0, 128, (16,), generator=g).tolist() for _ in range(2)], sha)
    return Recipe.from_dict(
        {"steps": steps, "calib": {"file": {"path": str(d / "c.jsonl"), "samples": 2, "seqlen": 16}}}
    )


def _build(d: Path, steps: list[dict], source: str | None = None, kind: str = "qwen3") -> Path:
    recipe = _recipe(d, steps, kind)
    return build(
        source or str(d / "m"), recipe, format=FORMAT, out_dir=d / "out", devices=["cpu"],
        dtype=torch.float32, target=TARGET, log=lambda _: None,
    )  # fmt: skip


def _tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(d.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            out |= {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    return out


def _decoded(out: Path) -> int:
    """Decode every packed weight `out` stores with compressed-tensors' NVFP4 decoder; returns how many."""
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme("NVFP4", ["Linear"])
    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["format"] == "nvfp4-pack-quantized"
    assert "NvFp4" in str(json.loads((out / "build.json").read_text())["serve"]["kernels"])
    n = 0
    for shard in sorted(out.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if not key.endswith(".weight_packed"):
                    continue
                stem = key.removesuffix(".weight_packed")
                packed = f.get_tensor(key)
                back = NVFP4PackedCompressor.decompress(
                    {
                        "weight_packed": packed,
                        "weight_scale": f.get_tensor(stem + ".weight_scale"),
                        "weight_global_scale": f.get_tensor(stem + ".weight_global_scale"),
                    },
                    scheme,
                )["weight"]
                assert back.shape == (packed.shape[0], 2 * packed.shape[1])
                assert torch.isfinite(back).all()
                assert f.get_tensor(stem + ".input_global_scale").numel() == 1
                n += 1
    return n


@pytest.fixture(scope="module")
def nvfp4_base(tmp_path_factory) -> Path:
    """A packed NVFP4 checkpoint of the tiny model: the source the checkpoint steps read back."""
    return _build(tmp_path_factory.mktemp("nvfp4-base"), [{"rtn": {}}])


@pytest.fixture(scope="module")
def nvfp4_bank_base(tmp_path_factory) -> Path:
    """A packed NVFP4 checkpoint of the tiny MoE model: its experts are stored one Linear per expert."""
    return _build(tmp_path_factory.mktemp("nvfp4-bank-base"), [{"rtn": {}}], kind="qwen3_moe")


def test_a_stage_on_a_checkpoint_with_no_record_writes_the_format_and_target_it_planned(nvfp4_base, tmp_path):
    """`Packed.write` copies the source's record; a checkpoint another tool wrote has none, so the output must
    record what the stage planned for."""
    foreign, other = tmp_path / "foreign", tmp_path / "other"
    shutil.copytree(nvfp4_base, foreign)
    (foreign / "build.json").unlink()
    shutil.copytree(foreign, other)
    out = _build(tmp_path, [{"soup": {"with": [str(other)]}}], str(foreign))
    rec = json.loads((out / "build.json").read_text())
    assert (rec["format"], rec["source"], rec["target"]) == (FORMAT, str(foreign), "sm100@0.33.0")


@pytest.mark.parametrize("name", [*BUILD, *BUILD_PACKED, *REFUSED])
def test_step_at_nvfp4(name, tmp_path, nvfp4_base, nvfp4_bank_base):
    if name in BUILD:
        assert _decoded(_build(tmp_path, BUILD[name])) > 0
        return
    if name in BUILD_PACKED:
        steps, which, kind = BUILD_PACKED[name]
        base = nvfp4_bank_base if which == "bank" else nvfp4_base
        out = _build(tmp_path, steps(base), str(base), kind)
        assert _decoded(out) > 0
        if name == "scale_kd-bank":  # with the scales frozen the served bank is stored back as it was read
            a, b = _tensors(base), _tensors(out)
            assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a if ".experts." in k)
        return
    steps, packed, kind = REFUSED[name]
    base = nvfp4_bank_base if kind == "qwen3_moe" else nvfp4_base
    with pytest.raises(Refused, match="(?i)nvfp4"):
        _build(tmp_path, steps(base), str(base) if packed else None, kind)
