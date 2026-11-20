"""`qda-kl` checks the kernels vLLM bound against the plan of the card it runs on.

A checkpoint's ``build.json`` records the kernels its build target's table planned. On a card whose table is
another, vLLM binds that card's plan under the build's engine arguments, so the expected kernels are re-planned
from the checkpoint's config groups with this card's table (the real tables, `qda_quant.target`).

``published/`` holds the ``build.json`` and ``config.json`` (no weights) of two qda-lab checkpoints built for
sm90, at the revisions read on an RTX PRO 6000 (sm120, vLLM 0.33.0) on 2026-11-19, where vLLM bound:

- QDA-Qwen3.6-35B-A3B-W3A16 @ 5d3e6d90: dense HummingLinearKernel, MarlinLinearKernel; moe HummingIndexedExperts
- QDA-Qwen3.6-35B-A3B-W4A16 @ 1e90af66: dense MarlinLinearKernel; moe MarlinExperts
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from qda_kl import score
from qda_kl.scorers import TopKScorer
from qda_kl.serve import card_kernels, check_kernels
from qda_quant.step import Refused
from test_score_nojit import tiny_reference

H100, L40S, PRO6000 = [("cuda:0", (9, 0))], [("cuda:0", (8, 9))], [("cuda:0", (12, 0))]
HUMMING, MACHETE, MARLIN = "HummingLinearKernel", "MacheteLinearKernel", "MarlinLinearKernel"
HUMMING_EXPERTS, MARLIN_EXPERTS = "HummingIndexedExperts", "MarlinExperts"
PUBLISHED = Path(__file__).parent / "published"
W3A16, W4A16 = "QDA-Qwen3.6-35B-A3B-W3A16", "QDA-Qwen3.6-35B-A3B-W4A16"
#: what vLLM 0.33.0 bound for them on an RTX PRO 6000
ON_SM120 = {
    W3A16: {"dense": {HUMMING, MARLIN}, "moe": {HUMMING_EXPERTS}},
    W4A16: {"dense": {MARLIN}, "moe": {MARLIN_EXPERTS}},
}


def published(path, name, **build):
    """A copy of a published checkpoint's ``build.json`` and ``config.json``; ``build`` replaces build.json keys."""
    shutil.copytree(PUBLISHED / name, path, dirs_exist_ok=True)
    record = json.loads((path / "build.json").read_text())
    (path / "build.json").write_text(json.dumps({**record, **build}))
    return path


def recorded(path):
    return {
        kind: set(names)
        for kind, names in json.loads((path / "build.json").read_text())["serve"]["kernels"].items()
    }


def checkpoint(path, target, fmt, kernels, groups, *, env=None, args=None, vocab=248320, hidden=4096):
    """A directory with a ``build.json`` and ``config.json`` groups, for a case no published checkpoint is."""
    path.mkdir(parents=True, exist_ok=True)
    serve = {"vllm": "0.33.0", "env": env or {}, "args": args or {}, "plugins": [], "kernels": kernels}
    (path / "build.json").write_text(json.dumps({"format": fmt, "target": target, "serve": serve}))
    qc = {"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups": groups}
    config = {"quantization_config": qc, "text_config": {"vocab_size": vocab, "hidden_size": hidden}}
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("name", [W3A16, W4A16])
def test_a_published_sm90_checkpoint_on_an_rtx_pro_6000_expects_what_vllm_bound_there(tmp_path, name):
    """The W3A16's int4 head binds Machete on an H100 and Marlin here; its int3 experts keep Humming."""
    path = published(tmp_path, name)
    bound = {kind: sorted(names) for kind, names in ON_SM120[name].items()}
    if name == W3A16:
        with pytest.raises(Refused, match="did not bind"):
            check_kernels(recorded(path), bound)  # build.json's kernels: refused before
    assert card_kernels(path, recorded(path), PRO6000) == (ON_SM120[name], "sm120")
    check_kernels(ON_SM120[name], bound)


def test_the_experts_expect_the_kernel_the_builds_moe_backend_selects_on_this_card(tmp_path):
    """The engine starts with build.json's arguments: ``moe_backend: humming`` binds Humming experts on sm120 too."""
    path = published(
        tmp_path,
        W4A16,
        target="sm90@0.33.0:humming",
        serve={
            "vllm": "0.33.0",
            "env": {"VLLM_DISABLED_KERNELS": MACHETE},
            "args": {"moe_backend": "humming"},
            "plugins": [],
            "kernels": {"dense": [HUMMING], "moe": [HUMMING_EXPERTS]},
        },
    )
    assert card_kernels(path, recorded(path), PRO6000) == (
        {"dense": {MARLIN}, "moe": {HUMMING_EXPERTS}},
        "sm120",
    )


def test_on_the_build_card_the_recorded_kernels_are_checked(tmp_path):
    path = published(tmp_path, W4A16, target="sm90@0.33.0:marlin")
    assert card_kernels(path, recorded(path), H100) == (recorded(path), None)


def test_a_published_record_on_the_old_marlin_default_is_read_as_vllm_picks(tmp_path):
    """The published W4A16 pins Marlin by its serve args, not its target: read unpinned, as its card line serves.

    vLLM 0.33 then binds Humming for the body and keeps MarlinExperts for its asymmetric experts.
    """
    path = published(tmp_path, W4A16)
    assert card_kernels(path, recorded(path), H100) == ({"dense": {HUMMING}, "moe": {MARLIN_EXPERTS}}, "sm90")
    assert card_kernels(path, recorded(path), PRO6000) == (ON_SM120[W4A16], "sm120")


def test_a_recorded_machete_head_expects_humming_on_the_build_card(tmp_path):
    """The published W3A16 records Machete for its int4 head; every engine has Machete off, so vLLM binds Humming."""
    path = published(tmp_path, W3A16)
    assert card_kernels(path, recorded(path), H100) == ({"dense": {HUMMING}, "moe": {HUMMING_EXPERTS}}, "sm90")


def test_the_build_cards_backend_pin_is_kept(tmp_path):
    """Without the ``:humming`` pin, sm90's table would plan Marlin for this body."""
    path = published(
        tmp_path,
        W4A16,
        target="sm90@0.33.0:humming",
        serve={
            "vllm": "0.33.0",
            "env": {"VLLM_DISABLED_KERNELS": MACHETE},
            "args": {"moe_backend": "humming"},
            "plugins": [],
            "kernels": {"dense": [HUMMING], "moe": [HUMMING_EXPERTS]},
        },
    )
    assert card_kernels(path, recorded(path), H100) == (recorded(path), None)


def test_a_recorded_kernel_this_cards_table_no_longer_binds_is_replanned_on_the_build_card(tmp_path):
    """A vocabulary of 151937 rows is no multiple of 128, so Machete cannot take the head: it binds Humming."""
    path = published(tmp_path, W3A16)
    config = json.loads((path / "config.json").read_text())
    config["text_config"]["vocab_size"] = 151937
    (path / "config.json").write_text(json.dumps(config))
    assert card_kernels(path, recorded(path), H100) == (
        {"dense": {HUMMING}, "moe": {HUMMING_EXPERTS}},
        "sm90",
    )


def int_group(targets, bits, symmetric, group_size=64):
    """A ``pack-quantized`` weight-only config group."""
    w = {"num_bits": bits, "type": "int", "group_size": group_size, "strategy": "group"}
    return {"targets": targets, "weights": {**w, "symmetric": symmetric}}


def test_8bit_groups_beside_a_4bit_body_expect_marlin_on_an_rtx_pro_6000(tmp_path):
    """The published Qwen3.5-4B W4A16 keeps some attention layers in 8 bits; vLLM bound Marlin for all."""
    groups = {
        "group_0": int_group(["Linear"], 4, False),
        "group_1": int_group(["model.language_model.layers.3.self_attn.q_proj"], 8, True),
    }
    env = {"VLLM_DISABLED_KERNELS": MACHETE}
    path = checkpoint(tmp_path, "sm90@0.33.0", "w4a16-asym-g64", {"dense": [HUMMING]}, groups, env=env)
    assert card_kernels(path, recorded(path), PRO6000) == ({"dense": {MARLIN}}, "sm120")


def test_a_body_group_that_stores_only_experts_plans_no_dense_kernel(tmp_path):
    """The published Nemotron-3-Nano W3A16: its Mamba, attention and shared-expert Linears are in 8-bit and
    4-bit groups, and its 3-bit body group stores the routed experts alone. On an RTX PRO 6000 vLLM bound
    dense MarlinLinearKernel and HummingIndexedExperts; a shared expert's name claims no routed expert."""
    dense_8 = ["backbone.layers.0.mixer.in_proj", "backbone.layers.1.mixer.shared_experts.up_proj"]
    dense_4 = ["backbone.layers.3.mixer.shared_experts.up_proj"]
    groups = {
        "group_0": int_group(["Linear"], 3, True),
        "group_1": int_group(dense_8, 8, True),
        "group_2": int_group(dense_4, 4, True),
    }
    kernels = {"dense": [HUMMING], "moe": [HUMMING_EXPERTS]}
    path = checkpoint(tmp_path, "sm90@0.33.0", "w3a16-sym-g64", kernels, groups, hidden=2688)
    config = json.loads((path / "config.json").read_text())
    config["text_config"]["moe_intermediate_size"] = 1856
    (path / "config.json").write_text(json.dumps(config))
    stems = [*dense_8, *dense_4, "backbone.layers.1.mixer.experts.0.up_proj"]
    index = {"weight_map": {f"{s}.weight_packed": "model-00001.safetensors" for s in stems}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    expected = {"dense": {MARLIN}, "moe": {HUMMING_EXPERTS}}
    assert card_kernels(path, recorded(path), PRO6000) == (expected, "sm120")


#: an FP8 W8A8 body (``float-quantized``, per channel, dynamic per-token activations) and an int4 head
FP8_INT4_HEAD = {
    "group_0": {
        "format": "float-quantized",
        "targets": ["Linear"],
        "weights": {
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "group_size": None,
            "symmetric": True,
        },
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
    },
    "group_head": {
        "targets": ["re:.*lm_head$"],
        "weights": {"num_bits": 4, "type": "int", "group_size": 64, "strategy": "group", "symmetric": True},
    },
}


def test_a_dense_fp8_body_has_no_expert_row_and_is_still_replanned(tmp_path):
    """FP8 W8A8 has no expert row on any table, and a dense model's record names no ``moe``: only dense is planned."""
    kernels = {"dense": {"CutlassFP8ScaledMMLinearKernel", MACHETE}}
    path = checkpoint(tmp_path, "sm90@0.33.0", "w8a8-fp8", {"dense": sorted(kernels["dense"])}, FP8_INT4_HEAD)
    config = json.loads((path / "config.json").read_text())
    qc = config["quantization_config"]  # as a float-quantized build stores it, beside its index
    qc["format"], qc["config_groups"]["group_head"]["format"] = "float-quantized", "pack-quantized"
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    assert card_kernels(path, kernels, PRO6000) == (
        {"dense": {"CutlassFP8ScaledMMLinearKernel", MARLIN}},
        "sm120",
    )


def test_linear_backend_marlin_on_the_build_card_keeps_the_record(tmp_path):
    """``linear_backend: marlin`` binds Marlin for the body and for the head in its own format."""
    path = published(tmp_path, W4A16, target="sm90@0.33.0:marlin")
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"]["config_groups"]["group_head"] = FP8_INT4_HEAD["group_head"]
    (path / "config.json").write_text(json.dumps(config))
    assert json.loads((path / "build.json").read_text())["serve"]["args"]["linear_backend"] == "marlin"
    assert card_kernels(path, recorded(path), H100) == (recorded(path), None)


#: qda-lab's W4A4 body group
NVFP4 = {
    "group_0": {
        "format": "nvfp4-pack-quantized",
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4,
            "type": "float",
            "group_size": 16,
            "strategy": "tensor_group",
            "symmetric": True,
        },
        "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"},
    }
}


def read(tmp_path, monkeypatch, here, bound):
    """`qda-kl kl`'s scorer on the checkpoint at `tmp_path`, on cards `here` where vLLM binds `bound`."""
    ref, engine = tiny_reference(tmp_path)
    engine.collective_rpc = lambda _: [bound]
    monkeypatch.setattr(score, "cards", lambda tp: here)
    monkeypatch.setattr(score, "open_engine", lambda *a, **k: engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    monkeypatch.setattr(
        score, "teacher_force", lambda *a, **k: {"a": dict.fromkeys(("kl", "flip", "tgt"), np.zeros(1))}
    )
    return TopKScorer().run(str(tmp_path), ref).result


def test_a_format_this_cards_table_refuses_keeps_the_recorded_kernels_and_their_refusal(
    tmp_path, monkeypatch
):
    """An L40S has no NVFP4 row: a W4A4 built for a B200 is checked against build.json and refused as before."""
    checkpoint(
        tmp_path, "sm100@0.33.0", "w4a4-nvfp4", {"dense": ["FlashInferCuteDslNvFp4LinearKernel"]}, NVFP4
    )
    with pytest.raises(Refused, match="expected \\['FlashInferCuteDslNvFp4LinearKernel'\\]"):
        read(
            tmp_path, monkeypatch, L40S, {"dense": ["MarlinNvFp4LinearKernel"]}
        )  # vLLM's FP4 fallback below 10.0


def test_a_checkpoint_that_records_no_kernels_has_nothing_to_replan(tmp_path):
    """The published W4A4 checkpoints record ``kernels: {}``: nothing is checked on any card."""
    path = checkpoint(tmp_path, "sm100@0.33.0", "w4a4-nvfp4", {}, NVFP4)
    assert card_kernels(path, {}, PRO6000) == ({}, None)


def test_the_reading_records_the_kernels_it_checked_and_their_table(tmp_path, monkeypatch):
    """`qda-kl kl` on an RTX PRO 6000 scores the published sm90 W3A16 and records sm120's plan as checked."""
    published(tmp_path, W3A16)
    # the config's quantization groups and sizes only: its model type would name another tokenizer class than
    # the one-word stand-in the reading is scored with
    config = json.loads((tmp_path / "config.json").read_text())
    sizes = {k: config["text_config"][k] for k in ("vocab_size", "hidden_size", "moe_intermediate_size")}
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": config["quantization_config"], "text_config": sizes})
    )
    result = read(tmp_path, monkeypatch, PRO6000, {kind: sorted(v) for kind, v in ON_SM120[W3A16].items()})
    assert result.kernels == {"dense": [HUMMING, MARLIN], "moe": [HUMMING_EXPERTS]}
    assert result.serve.kernels == ON_SM120[W3A16] and result.kernels_from == "sm120"
