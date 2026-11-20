"""The format gate and baseline identity (paper §4.1 admissible recipes), ported from the earlier checker.

Fixtures are real (tiny) safetensors checkpoints. The differential suite against the earlier checker is
`test_judge_parity.py`.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from qda_agent.judge import JudgeConfig, JudgeHost
from qda_agent.judge.checks import (
    DEFAULT_MENU,
    CheckpointError,
    files_problem,
    format_key,
    format_problem,
    loaded,
    tensor_fingerprint,
)
from qda_agent.rpc import Request
from qda_agent.sched import FakeBackend, FakeClock, SchedConfig
from safetensors_kit import CONFIG, QCONFIG, anchor_dir, packed_tensors, quant_dir, write_safetensors


def _ref_dir(root):
    """An eval+holdout reference: the judge host derives its calibration index from configured references."""
    d = Path(root) / "calib-ref"
    d.mkdir(parents=True, exist_ok=True)
    prompts = d / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "r1", "prompt": "a reference prompt used by the judge tests only"})
        + "\n"
        + json.dumps(
            {"id": "h1", "prompt": "a holdout prompt used by the judge tests only", "split": "holdout"}
        )
        + "\n"
    )
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return d


def _index_file(root):
    """A prebuilt index equal to the canonical union of `_ref_dir`'s eval and holdout sets."""
    from qda_quant.calib.contamination import PromptIndex

    return PromptIndex.read_reference(_ref_dir(root)).save(Path(root) / "contamination-index.json")


@pytest.fixture
def cell(tmp_path):
    return anchor_dir(tmp_path / "anchor"), quant_dir(tmp_path / "ref", weights=b"ref")


def test_valid_same_format_checkpoint_is_accepted(tmp_path, cell):
    anchor, ref = cell
    assert files_problem(quant_dir(tmp_path / "mine", weights=b"mine"), anchor, ref) is None


def _weights(bits: int = 4, group_size: int = 128) -> dict:
    """A compressed-tensors group's `weights`: int codes, one scale per `group_size`."""
    return {"num_bits": bits, "group_size": group_size, "strategy": "group", "type": "int", "symmetric": True}


def _leaves(bits) -> dict:
    """One attention's q/k/v at one format: an int (g128, bf16 activations) or a (bits, [group_size, act])."""
    return dict.fromkeys(("q_proj", "k_proj", "v_proj"), bits)


def _attn_ckpt(d, spec: dict) -> Path:
    """A checkpoint storing each layer's q/k/v packed at the format `spec` gives; a Linear fallback stays W4."""
    d.mkdir(parents=True)
    tensors = {"model.embed_tokens.weight": ("BF16", (16, 8), b"\x01\x02")}
    groups = {"group_0": {"targets": ["Linear"], "weights": _weights()}}
    for layer, leaves in spec.items():
        for leaf, fmt in leaves.items():
            parts = (fmt,) if isinstance(fmt, int) else fmt
            bits, group_size, act = parts[0], parts[1] if len(parts) > 1 else 128, len(parts) > 2
            module = f"model.layers.{layer}.self_attn.{leaf}"
            tensors[f"{module}.weight_packed"] = ("I32", (8, 1), b"\x01\0\0\0")
            tensors[f"{module}.weight_scale"] = ("BF16", (8, 1), b"\x03\x04")
            group = {"targets": [module], "weights": _weights(bits, group_size)}
            if act:
                group["input_activations"] = {"num_bits": 8, "type": "float", "dynamic": True}
            groups[f"g_{layer}_{leaf}"] = group
    (d / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": groups,
                    "ignore": ["model.embed_tokens"],
                },
            }
        )
    )
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return d


def test_a_per_layer_mixed_w8a16_w4a16_checkpoint_is_admitted(tmp_path, cell):
    """The menu is per module: one layer may be W8 and the next W4, as long as no fused unit mixes them."""
    anchor, ref = cell
    d = _attn_ckpt(tmp_path / "mixed", {0: _leaves(8), 1: _leaves(4)})
    assert files_problem(d, anchor, ref) is None
    assert format_problem(d) is None


def test_q_and_k_in_different_formats_is_refused_naming_the_module(tmp_path, cell):
    anchor, ref = cell
    d = _attn_ckpt(tmp_path / "split", {0: {**_leaves(8), "v_proj": 4}})
    why = files_problem(d, anchor, ref)
    assert why is not None and why.startswith("format: ")
    assert "model.layers.0.self_attn.v_proj is w4a16" in why
    assert "fused q/k/v" in why


def test_a_format_outside_the_menu_is_refused(tmp_path, cell):
    anchor, ref = cell
    d = _attn_ckpt(tmp_path / "w3", {0: _leaves(3)})
    assert files_problem(d, anchor, ref) == (
        "format: model.layers.0.self_attn.k_proj is w3a16, outside judge.format_menu (bf16, w8a16, w4a16)"
    )
    assert files_problem(d, anchor, ref, format_menu=[*DEFAULT_MENU, "w3a16"]) is None  # the W3 tier


@pytest.mark.parametrize(
    "bits,high,group", [(4, 8, 64), (4, 8, 128), (3, 4, 64), (3, 4, 128), (3, 8, 64), (3, 8, 128)]
)
def test_the_group_is_one_per_checkpoint_and_survives_config_resolution(tmp_path, cell, bits, high, group):
    from qda_agent.config import RunConfig

    anchor, ref = cell
    cfg = RunConfig(
        contamination="none",
        model="tiny/model",
        format=f"w{bits}a16-sym-g{group}",
        run_dir=tmp_path / "run",
        budget_gpu_hours=1,
        cards=[0],
        judge={"kl_cmd": ["score"]},
    )
    recorded = cfg.resolved()["judge"]["format_menu"]
    menu = JudgeConfig.model_validate(cfg.judge_config()).format_menu
    assert recorded == menu
    good = _attn_ckpt(tmp_path / "good", {0: _leaves((bits, group)), 1: _leaves((high, group))})
    assert files_problem(good, anchor, ref, format_menu=menu) is None
    changed = 64 if group == 128 else 128
    bad = _attn_ckpt(tmp_path / "bad", {0: _leaves((bits, group)), 1: _leaves((bits, changed))})
    assert "outside judge.format_menu" in files_problem(bad, anchor, ref, format_menu=menu)
    # a menu may name other groups; a checkpoint still takes one group across its quantized modules
    both = RunConfig(
        contamination="none",
        model="tiny/model",
        format=f"w{bits}a16-sym-g{group}",
        run_dir=tmp_path / "both",
        budget_gpu_hours=1,
        cards=[0],
        judge={
            "kl_cmd": ["score"],
            "format_menu": ["bf16", f"w{bits}a16-g{group}", f"w{bits}a16-sym-g{changed}"],
        },
    ).judge.format_menu
    assert both == ["bf16", f"w{bits}a16-sym-g{group}", f"w{bits}a16-sym-g{changed}"]
    assert "one checkpoint takes one group" in files_problem(bad, anchor, ref, format_menu=both)


def test_fused_units_cannot_mix_group_sizes(tmp_path, cell):
    anchor, ref = cell
    split = _attn_ckpt(tmp_path / "split", {0: {**_leaves((4, 128)), "v_proj": (4, 64)}})
    why = files_problem(split, anchor, ref, format_menu=["w4a16-g64", "w4a16-g128"])
    assert why is not None and "fused q/k/v" in why


def test_a_layer_has_one_activation_format(tmp_path, cell):
    anchor, ref = cell
    d = _attn_ckpt(tmp_path / "act", {0: {**_leaves(8), "o_proj": (8, 128, True)}})
    why = files_problem(d, anchor, ref, format_menu=[*DEFAULT_MENU, "w8a8-sym-g128"])
    assert why is not None and "model.layers.0 mixes two activation formats" in why


def test_a_routed_expert_bank_takes_one_format(tmp_path, cell):
    """A bank's fused gate_up projection is one stored tensor: its leaves may not be declared apart."""
    anchor, ref = cell
    d = tmp_path / "bank"
    d.mkdir()
    bank = "model.layers.0.mlp.experts"
    groups = {
        "gate": {"targets": [f"re:^{bank}\\.0\\.gate_proj$"], "weights": _weights(8)},
        "up": {"targets": [f"re:^{bank}\\.0\\.up_proj$"], "weights": _weights(4)},
    }
    (d / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": groups,
                    "ignore": ["model.embed_tokens"],
                },
            }
        )
    )
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(
        d / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (16, 8), b"\x01\x02"),
            f"{bank}.gate_up_proj_packed": ("I32", (2, 8, 1), b"\x01\0\0\0"),
            f"{bank}.gate_up_proj_scale": ("BF16", (2, 8, 1), b"\x03\x04"),
        },
    )
    why = files_problem(d, anchor, ref)
    assert why is not None and "gate_up_proj" in why and "one stored tensor" in why


def test_an_autoawq_configs_global_bits_are_its_modules_format(tmp_path, cell):
    """autoawq writes no config_groups: its `bits` must still put its packed modules on the menu."""
    anchor, ref = cell
    for bits, admitted in ((4, True), (3, False)):
        d = tmp_path / f"awq-{bits}"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    **CONFIG,
                    "quantization_config": {
                        "quant_method": "awq",
                        "bits": bits,
                        "group_size": 128,
                        "zero_point": True,
                        "version": "gemm",
                    },
                }
            )
        )
        (d / "tokenizer.json").write_text("{}")
        write_safetensors(
            d / "model.safetensors",
            {
                "model.embed_tokens.weight": ("BF16", (16, 8), None),
                "model.layers.0.self_attn.q_proj.qweight": ("I32", (8, 1), None),
                "model.layers.0.self_attn.q_proj.scales": ("F16", (1, 8), None),
                "model.layers.0.self_attn.q_proj.qzeros": ("I32", (1, 1), None),
            },
        )
        why = files_problem(d, anchor, ref)
        assert (why is None) == admitted, why
        if why:
            assert "model.layers.0.self_attn.q_proj is w3a16" in why


def _head_case(d: Path, head: str, stored: str, tied: bool = False, nested: bool = False) -> Path:
    """A one-layer W4 checkpoint whose `head` is stored `w8` (packed, own group), `packed` (ignored) or a dtype."""
    groups = {"body": {"targets": ["Linear"], "weights": _weights(4)}}
    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": ("I32", (8, 1), None),
        "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (8, 1), None),
        "model.embed_tokens.weight": ("BF16", (16, 8), None),
    }
    if head.rpartition(".")[2] == "embed_tokens":
        del tensors["model.embed_tokens.weight"]
    if stored in ("w8", "packed"):
        tensors[f"{head}.weight_packed"] = ("I32", (16, 1), None)
        tensors[f"{head}.weight_scale"] = ("BF16", (16, 1), None)
        if stored == "w8":
            groups["head"] = {"targets": [head], "weights": _weights(8)}
    else:
        tensors[f"{head}.weight"] = (stored, (16, 8), None)
    cfg = {
        **CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": groups,
            "ignore": ["model.embed_tokens"],
        },
    }
    if tied:
        cfg["text_config" if nested else "tie_word_embeddings"] = (
            {"tie_word_embeddings": True} if nested else True
        )
    if stored in ("packed", "BF16", "F16", "F32"):
        cfg["quantization_config"]["ignore"] = [head]
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return d


@pytest.mark.parametrize(
    ("head", "stored", "tied", "nested", "admitted"),
    [
        ("lm_head", "w8", False, False, False),
        ("model.lm_head", "w8", False, False, False),
        ("lm_head", "BF16", False, False, True),
        ("lm_head", "F16", False, False, True),
        ("lm_head", "F32", False, False, True),
        ("lm_head", "packed", False, False, False),  # an ignore entry claims a packed head float
        ("model.embed_tokens", "w8", True, False, False),
        ("model.embed_tokens", "F16", True, False, True),  # a tied head is 16-bit too
        ("model.embed_tokens", "BF16", True, False, True),
        ("model.embed_tokens", "F32", True, False, True),
        ("model.language_model.embed_tokens", "w8", True, True, False),  # the flag under text_config
        ("model.embed_tokens", "BF16", False, False, True),  # untied: the embedding is not the head
    ],
)
def test_the_output_head_stays_unquantized(tmp_path, head, stored, tied, nested, admitted):
    d = _head_case(tmp_path / "head", head, stored, tied, nested)
    why = format_problem(d)
    if admitted:
        assert why is None
    else:
        assert why is not None and "lm_head is stored" in why
        assert "the task keeps it 16-bit (BF16 or F16)" in why


@pytest.mark.parametrize("head,tied", [("lm_head", False), ("model.embed_tokens", True)])
@pytest.mark.parametrize(
    "source_dtype,head_dtype,admitted",
    [("F32", "F32", True), ("BF16", "F16", False), ("BF16", "F32", False), ("F32", "BF16", False)],
)
def test_the_head_preserves_the_source_dtype(tmp_path, head, tied, source_dtype, head_dtype, admitted):
    d = _head_case(tmp_path / "head", head, head_dtype, tied=tied)
    anchor = anchor_dir(tmp_path / "source")
    (anchor / "config.json").write_text(json.dumps({**CONFIG, "tie_word_embeddings": tied}))
    cfg = json.loads((d / "config.json").read_text())
    cfg["tie_word_embeddings"] = tied
    (d / "config.json").write_text(json.dumps(cfg))
    write_safetensors(anchor / "model.safetensors", {f"{head}.weight": (source_dtype, (16, 8), None)})
    why = files_problem(d, anchor=anchor)
    assert (why is None) == admitted, why
    if why:
        assert why == f"format: lm_head is stored {head_dtype}; the task keeps it 16-bit (BF16 or F16)"


@pytest.mark.parametrize("head_dtype,admitted", [("F32", True), ("BF16", False)])
def test_a_tied_head_uses_the_source_embedding_dtype(tmp_path, head_dtype, admitted):
    d = _head_case(tmp_path / "head", "lm_head", head_dtype, tied=True)
    write_safetensors(d / "model.safetensors", {"lm_head.weight": (head_dtype, (16, 8), None)})
    anchor = anchor_dir(tmp_path / "source")
    write_safetensors(anchor / "model.safetensors", {"model.embed_tokens.weight": ("F32", (16, 8), None)})
    why = format_problem(d, anchor=anchor)
    assert (why is None) == admitted, why


@pytest.mark.parametrize("head,tied", [("lm_head", False), ("model.embed_tokens", True)])
@pytest.mark.parametrize(
    "suffix", ["weight_packed", "weight_scale", "weight_zero_point", "qweight", "scales", "qzeros"]
)
def test_the_head_refuses_quantization_tensors_even_with_a_float_weight(tmp_path, head, tied, suffix):
    d = _head_case(tmp_path / "head", head, "BF16", tied=tied)
    write_safetensors(
        d / "model.safetensors",
        {f"{head}.weight": ("BF16", (16, 8), None), f"{head}.{suffix}": ("I32", (16, 1), None)},
    )
    why = format_problem(d)
    assert why is not None and "the task keeps it 16-bit (BF16 or F16)" in why


def test_a_bf16_lm_head_is_admitted_as_a_candidate(tmp_path, cell):
    anchor, ref = cell
    assert files_problem(_head_case(tmp_path / "head", "lm_head", "BF16"), anchor, ref) is None


def test_the_builtin_awq_and_gptq_baselines_pass_the_head_rule(tmp_path):
    gptq = _head_case(tmp_path / "gptq", "lm_head", "BF16")
    cfg = json.loads((gptq / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "gptq",
        "format": "pack-quantized",
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": _weights()}},
        "ignore": ["lm_head"],
    }
    (gptq / "config.json").write_text(json.dumps(cfg))
    awq = tmp_path / "awq"
    awq.mkdir()
    (awq / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": True,
                    "modules_to_not_convert": ["lm_head"],
                    "version": "gemm",
                },
            }
        )
    )
    (awq / "tokenizer.json").write_text("{}")
    write_safetensors(
        awq / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (16, 8), None),
            "model.layers.0.self_attn.q_proj.qweight": ("I32", (8, 1), None),
            "model.layers.0.self_attn.q_proj.scales": ("F16", (1, 8), None),
            "model.layers.0.self_attn.q_proj.qzeros": ("I32", (1, 1), None),
            "lm_head.weight": ("BF16", (16, 8), None),
        },
    )
    assert format_problem(gptq) is None
    assert format_problem(awq) is None


def test_changed_architecture_is_refused_naming_the_field(tmp_path, cell):
    anchor, ref = cell
    d = quant_dir(tmp_path / "wide")
    (d / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": 16, "quantization_config": QCONFIG}))
    assert files_problem(d, anchor, ref) == "config.json differs from the anchor's: hidden_size"
    # a nested field is named by its path
    (d / "config.json").write_text(
        json.dumps({**CONFIG, "text_config": {"num_hidden_layers": 2}, "quantization_config": QCONFIG})
    )
    (anchor / "config.json").write_text(json.dumps({**CONFIG, "text_config": {"num_hidden_layers": 1}}))
    assert (
        files_problem(d, anchor, ref)
        == "config.json differs from the anchor's: text_config.num_hidden_layers"
    )
    (anchor / "config.json").write_text(json.dumps(CONFIG))


def test_a_dtype_or_bookkeeping_difference_is_admitted_at_any_depth(tmp_path, cell):
    """A checkpoint saved in fp16 (its own measured config) or with its own name is still the anchor's model."""
    anchor, ref = cell
    d = quant_dir(tmp_path / "fp16")
    (d / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "dtype": "float16",
                "torch_dtype": "float16",
                "name_or_path": "vendor/Qwen-Tiny",
                "_name_or_path": "vendor/Qwen-Tiny",
                "transformers_version": "9",
                "text_config": {"dtype": "float16", "hidden_size": 8},
                "vision_config": {"depth": 1, "torch_dtype": "float16"},
                "quantization_config": QCONFIG,
            }
        )
    )
    (anchor / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "text_config": {"dtype": "bfloat16", "hidden_size": 8},
                "vision_config": {"depth": 1},
            }
        )
    )
    assert files_problem(d, anchor, ref) is None


def test_free_keys_inside_a_list_of_subconfigs_are_ignored(tmp_path, cell):
    """A list of sub-configs recurses too: a dtype inside one is free, an architecture field is not."""
    anchor, ref = cell
    d = quant_dir(tmp_path / "subconfigs")
    (d / "config.json").write_text(
        json.dumps(
            {
                **CONFIG,
                "subconfigs": [{"hidden_size": 8, "dtype": "float16"}],
                "quantization_config": QCONFIG,
            }
        )
    )
    (anchor / "config.json").write_text(
        json.dumps({**CONFIG, "subconfigs": [{"hidden_size": 8, "dtype": "bfloat16"}]})
    )
    assert files_problem(d, anchor, ref) is None
    (anchor / "config.json").write_text(
        json.dumps({**CONFIG, "subconfigs": [{"hidden_size": 4, "dtype": "bfloat16"}]})
    )
    assert files_problem(d, anchor, ref) == "config.json differs from the anchor's: subconfigs.0.hidden_size"
    (anchor / "config.json").write_text(json.dumps(CONFIG))


def test_structure_and_anchor_files(tmp_path, cell):
    anchor, ref = cell
    d = quant_dir(tmp_path / "idx")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "config.json"}}))
    assert "not a shard file" in files_problem(d, anchor, ref)
    bad = quant_dir(tmp_path / "bad")
    (bad / "model.safetensors").write_bytes(b"not safetensors")
    assert "not a safetensors file" in files_problem(bad, anchor, ref)
    (anchor / "vocab.json").write_text('{"a": 0}')
    v = quant_dir(tmp_path / "vocab")
    assert "vocab.json differs" in files_problem(v, anchor, ref)  # the anchor has it: so must the checkpoint
    (v / "vocab.json").write_text('{"a": 0}')
    assert files_problem(v, anchor, ref) is None  # byte-identical: accepted
    (anchor / "additional_chat_templates").mkdir()
    (anchor / "additional_chat_templates" / "tool_use.jinja").write_text("T")
    assert "additional_chat_templates/ differs" in files_problem(v, anchor, ref)
    shutil.copytree(anchor / "additional_chat_templates", v / "additional_chat_templates")
    assert files_problem(v, anchor, ref) is None
    (v / "added_tokens.json").write_text("{}")  # the anchor has none
    assert "added_tokens.json differs" in files_problem(v, anchor, ref)
    (v / "added_tokens.json").unlink()
    (v / ".gitattributes").write_text("*")
    assert "hidden files" in files_problem(v, anchor, ref)


def test_candidate_0_is_checked_by_its_files_not_its_tensor_inventory(tmp_path):
    """The baseline's own directory carries build logs; those are still not served, but its tensors are its own."""
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = quant_dir(tmp_path / "base0", weights=b"base0")
    (base0 / "build.log").write_text("logged")
    assert files_problem(base0, anchor, ref) == (
        "build.log: not served (weights are safetensors; any other file must be the anchor's own, byte for byte)"
    )
    # rebuilt weights that differ from the reference's are admitted on their format alone
    assert files_problem(_attn_ckpt(tmp_path / "mine", {0: _leaves(4)}), anchor, ref) is None


def test_fingerprint_ignores_shard_names_sharding_and_metadata(tmp_path):
    base = quant_dir(tmp_path / "base")
    renamed = tmp_path / "renamed"
    shutil.copytree(base, renamed)
    (renamed / "model.safetensors").rename(renamed / "other.safetensors")
    resharded = tmp_path / "resharded"
    resharded.mkdir()
    shutil.copy(base / "config.json", resharded)
    t = list(packed_tensors().items())
    write_safetensors(resharded / "a.safetensors", dict(t[:2]), metadata={"format": "pt"})
    write_safetensors(resharded / "b.safetensors", dict(t[2:]))
    fp = tensor_fingerprint(base)
    assert tensor_fingerprint(renamed) == fp == tensor_fingerprint(resharded)
    assert tensor_fingerprint(quant_dir(tmp_path / "other", weights=b"w2")) != fp
    # unscored tensors (an MTP head) do not make a baseline another checkpoint
    mtp = quant_dir(
        tmp_path / "mtp", tensors={**packed_tensors(), "mtp.fc.weight": ("BF16", (2,), b"\x01\x02")}
    )
    assert tensor_fingerprint(mtp) == fp


def test_the_reads_of_a_span_agree_with_and_without_numpy():
    """The near-copy tolerance must not change where numpy is missing (bf16 has no struct type)."""
    from qda_agent.judge.checks.tensors import _norms, _norms_slow

    for dtype, a, b in (
        ("BF16", b"\x00\x3f" * 4, b"\x80\x3f" * 4),  # 0.5, 0.5, 0.5, 0.5 against 1.0, 1.0, 1.0, 1.0
        ("U8", bytes([1, 2, 3]), bytes([1, 2, 4])),
        ("I32", b"\x01\x00\x00\x00\x02\x00\x00\x00", b"\x01\x00\x00\x00\x03\x00\x00\x00"),
    ):
        assert _norms(a, b, dtype) == pytest.approx(_norms_slow(a, b, dtype))


def test_renamed_baseline_is_refused(tmp_path, cell):
    anchor, ref = cell
    base = quant_dir(tmp_path / "base")
    run = tmp_path / "run"
    renamed = run / "work" / "renamed"
    shutil.copytree(base, renamed)
    (renamed / "model.safetensors").rename(renamed / "renamed.safetensors")
    clock = FakeClock()
    host = JudgeHost(
        run,
        SchedConfig(cards=[4], budget_s=3600, fake_gpus=True),
        JudgeConfig(
            contamination="none",
            anchor=anchor,
            format_reference=ref,
            baselines=[base],
            kl_cmd=["score"],
            check_cmd=None,
            kl_ref=str(_ref_dir(run.parent)),
            kl_holdout_ref=str(_ref_dir(run.parent)),
        ),
        clock,
        FakeBackend(clock),
        autotick=False,
    )
    reply = host.handle(Request("handin", ["renamed"], str(run / "work")))
    assert "a baseline checkpoint as-is" in reply.stdout + reply.stderr
    assert host.final().rule == "handin"


def test_a_qda_writer_shaped_checkpoint_is_accepted(tmp_path, cell):
    # qda_quant's Writer copies the anchor's tokenizer / template / processor files next to the weights.
    anchor, ref = cell
    files = {
        "vocab.json": '{"a": 0}',
        "merges.txt": "a b\n",
        "chat_template.json": "{}",
        "preprocessor_config.json": "{}",
    }
    for name, text in files.items():
        (anchor / name).write_text(text)
    d = quant_dir(tmp_path / "built", weights=b"built")
    for name in files:
        shutil.copy(anchor / name, d / name)
    (d / "recipe.yaml").write_text("steps: []\n")
    (d / "build.json").write_text("{}")
    assert files_problem(d, anchor, ref) is None
    (d / "merges.txt").write_text("edited\n")
    assert "merges.txt differs" in files_problem(d, anchor, ref)


def test_a_config_built_checkpoint_carrying_resolved_yaml_is_accepted(tmp_path, cell):
    # `qda build CONFIG.yaml` writes resolved.yaml (no recipe.yaml) next to the weights.
    anchor, ref = cell
    d = quant_dir(tmp_path / "built", weights=b"built")
    resolved = "model: m\nformat: w4a16\nsteps: [{gptq: {}}]\ncalib: {file: {path: c.jsonl}}\n"
    (d / "resolved.yaml").write_text(resolved)
    (d / "build.json").write_text("{}")
    assert files_problem(d, anchor, ref) is None


def test_inert_globs_match_top_level_files_only(tmp_path, cell):
    # fnmatch's * crosses "/": LICENSE* must not accept a directory of weights named LICENSE.
    anchor, ref = cell
    d = quant_dir(tmp_path / "lic", weights=b"lic")
    (d / "LICENSE.txt").write_text("MIT\n")
    assert files_problem(d, anchor, ref) is None
    (d / "LICENSE").mkdir()
    write_safetensors(d / "LICENSE" / "model.safetensors", {"x": ("BF16", (2,), b"\x01\x02")})
    assert "LICENSE/model.safetensors: not served" in files_problem(d, anchor, ref)


def test_scale_kd_state_is_inert_at_its_one_path(tmp_path, cell):
    # scale_kd keeps its trained deltas in qda_state/scales.safetensors, which vLLM does not load.
    anchor, ref = cell
    d = quant_dir(tmp_path / "kd", weights=b"kd")
    (d / "qda_state").mkdir()
    write_safetensors(d / "qda_state" / "scales.safetensors", {"delta": ("BF16", (2,), b"\x01\x02")})
    assert files_problem(d, anchor, ref) is None
    (d / "qda_state" / "other.safetensors").write_bytes((d / "qda_state" / "scales.safetensors").read_bytes())
    assert "qda_state/other.safetensors: not served" in files_problem(d, anchor, ref)
    (d / "qda_state" / "other.safetensors").unlink()
    (d / "sub").mkdir()  # nor the same name anywhere else
    (d / "qda_state" / "scales.safetensors").rename(d / "sub" / "scales.safetensors")
    assert "sub/scales.safetensors: not served" in files_problem(d, anchor, ref)


def test_an_unindexed_extra_shard_is_refused(tmp_path, cell):
    anchor, ref = cell
    d = quant_dir(tmp_path / "indexed")
    names = {k: "model.safetensors" for k in packed_tensors()}
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    assert files_problem(d, anchor, ref) is None
    write_safetensors(d / "zz.safetensors", {"extra.weight": ("BF16", (2,), b"\x01\x02")})
    assert "zz.safetensors: a shard the index does not name" in files_problem(d, anchor, ref)
    base = quant_dir(tmp_path / "base")
    plus = tmp_path / "plus"
    shutil.copytree(base, plus)
    write_safetensors(plus / "zz.safetensors", {"extra.weight": ("BF16", (2,), b"\x01\x02")})
    assert tensor_fingerprint(plus) != tensor_fingerprint(base)  # a baseline plus side tensors is another one
    assert files_problem(plus, anchor, ref) is None  # a side tensor is not a format problem


def test_an_indexed_tensor_no_shard_holds_is_refused(tmp_path, cell):
    """The index and the shards' headers must name the same tensors: a name only the index holds would be
    silently left out of the fingerprint."""
    anchor, ref = cell
    d = quant_dir(tmp_path / "ghost")
    names = {k: "model.safetensors" for k in packed_tensors()}
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    assert files_problem(d, anchor, ref) is None
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {**names, "missing.weight": "model.safetensors"}})
    )
    with pytest.raises(CheckpointError, match="missing.weight"):
        loaded(d)
    assert "missing.weight" in files_problem(d, anchor, ref)


def test_a_checkpoint_over_the_size_cap_is_refused(tmp_path, cell):
    anchor, ref = cell
    run = tmp_path / "run"
    quant_dir(
        run / "work" / "big",
        tensors={
            **packed_tensors(),
            "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (8, 4096), None),
        },
    )
    host = _host(run, anchor=anchor, format_reference=ref, max_copy_bytes=4096)
    reply = host.handle(Request("handin", ["big"], str(run / "work")))
    assert reply.exit == 2 and "more than the cap" in reply.stderr


def _host(run, **judge):
    clock = FakeClock()
    return JudgeHost(
        run,
        SchedConfig(cards=[4], budget_s=3600, fake_gpus=True),
        JudgeConfig(
            contamination="none",
            kl_cmd=["score"],
            check_cmd=None,
            kl_ref=str(_ref_dir(run.parent)),
            kl_holdout_ref=str(_ref_dir(run.parent)),
            **judge,
        ),
        clock,
        FakeBackend(clock),
        autotick=False,
    )


def test_a_search_needs_no_format_reference(tmp_path, monkeypatch):
    """The menu and the engine's facts gate the format: a reference checkpoint is optional now."""
    anchor = anchor_dir(tmp_path / "anchor")
    host = _host(tmp_path / "run", anchor=anchor)  # no format_reference, no allow_any_format

    def evaluator():
        raise RuntimeError("passed the start gate")

    monkeypatch.setattr(host, "install_evaluator", evaluator)
    with pytest.raises(RuntimeError, match="passed the start gate"):
        host.start()
    # a candidate is still gated by the menu, reference or not
    d = _attn_ckpt(tmp_path / "w3", {0: _leaves(3)})
    assert files_problem(d, anchor) == (
        "format: model.layers.0.self_attn.k_proj is w3a16, outside judge.format_menu (bf16, w8a16, w4a16)"
    )


def test_candidate_0_passes_the_start_gate_with_its_own_extra_files(tmp_path, monkeypatch):
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = quant_dir(tmp_path / "base0", weights=b"base0")
    (base0 / "build.log").write_text("logged")  # its own file: the start check ignores it
    host = _host(tmp_path / "run", anchor=anchor, format_reference=ref, baselines=[base0])

    def evaluator():
        raise RuntimeError("passed the start gate")

    monkeypatch.setattr(host, "install_evaluator", evaluator)
    with pytest.raises(RuntimeError, match="passed the start gate"):
        host.start()


def test_candidate_0_outside_the_menu_is_refused_before_start(tmp_path):
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = _attn_ckpt(tmp_path / "base0", {0: _leaves(3)})
    host = _host(tmp_path / "run", anchor=anchor, format_reference=ref, baselines=[base0])
    with pytest.raises(ValueError, match=r"\(candidate 0\): format: .*w3a16"):
        host.start()
    assert (host.server is None) and not host.sched.jobs


def test_candidate_0_with_an_edited_tokenizer_is_refused_before_start(tmp_path):
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = quant_dir(tmp_path / "base0")
    (base0 / "tokenizer.json").write_text('{"mine": 1}')
    host = _host(tmp_path / "run", anchor=anchor, format_reference=ref, baselines=[base0])
    with pytest.raises(ValueError, match=r"\(candidate 0\): tokenizer.json differs"):
        host.start()


def test_an_unreadable_first_baseline_is_refused_before_start(tmp_path):
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    bad = tmp_path / "bad"
    bad.mkdir()
    shutil.copy(anchor / "tokenizer.json", bad)
    (bad / "config.json").write_text("not JSON")
    host = _host(tmp_path / "run", anchor=anchor, format_reference=ref, baselines=[bad])
    with pytest.raises(ValueError, match=r"\(candidate 0\): no readable config.json"):
        host.start()
    assert host.server is None


def _malformed(d, quantization_config):
    """Rewrite `d`'s config with a quantization config that is not a layout."""
    cfg = json.loads((d / "config.json").read_text())
    cfg["quantization_config"] = quantization_config
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_a_malformed_alternative_layout_is_refused_before_start(tmp_path):
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = _malformed(quant_dir(tmp_path / "base0"), ["invalid"])
    host = _host(tmp_path / "run", anchor=anchor, format_reference=ref, baselines=[base0])
    with pytest.raises(
        ValueError, match=r"\(candidate 0\): format: quantization_config is not a layout"
    ) as err:
        host.start()
    assert "quantization_config is not a layout" in str(err.value)
    assert host.server is None and not host.sched.jobs


def test_a_malformed_alternative_source_is_a_refusal_at_admission_too(tmp_path):
    """A request on a host whose start was refused (the issue's probe keeps going) does not raise either."""
    anchor = anchor_dir(tmp_path / "anchor")
    ref = quant_dir(tmp_path / "ref")
    base0 = _malformed(quant_dir(tmp_path / "base0"), ["invalid"])
    run = tmp_path / "run"
    host = _host(run, anchor=anchor, format_reference=ref, baselines=[base0])
    _malformed(_attn_ckpt(run / "work" / "mine", {0: _leaves(4)}), ["invalid"])
    reply = host.handle(Request("kl", ["mine"], str(run / "work")))
    assert reply.exit == 2 and "quantization_config is not a layout" in reply.stderr


MALFORMED_QUANTIZATION = [
    (["invalid"], "list"),
    ("invalid", "string"),
    ({"config_groups": ["invalid"]}, "config-groups-list"),
    ({"config_groups": {"group_0": {"weights": "invalid"}}}, "weights-string"),
]


@pytest.mark.parametrize(
    "quantization_config", [q for q, _ in MALFORMED_QUANTIZATION], ids=[i for _, i in MALFORMED_QUANTIZATION]
)
def test_a_malformed_quantization_config_is_refused_not_raised(tmp_path, cell, quantization_config):
    """An agent can write any `quantization_config`; admission names it instead of raising from `_groups`."""
    anchor, ref = cell
    run = tmp_path / "run"
    d = _malformed(quant_dir(run / "work" / "mine"), quantization_config)
    cfg = json.loads((d / "config.json").read_text())
    why = files_problem(d, anchor, ref)
    assert why and why.startswith("config.json: quantization_config is not a layout"), why
    assert format_problem(d, cfg=cfg).startswith("format: quantization_config is not a layout")
    reply = _host(run, anchor=anchor, format_reference=ref).handle(
        Request("handin", ["mine"], str(run / "work"))
    )
    assert reply.exit == 2 and "quantization_config is not a layout" in reply.stderr


@pytest.mark.parametrize("verb", ["kl", "handin"])
def test_a_malformed_quantization_config_is_a_refusal_and_the_host_serves_on(tmp_path, cell, verb):
    anchor, ref = cell
    run = tmp_path / "run"
    host = _host(run, anchor=anchor, format_reference=ref)
    _malformed(quant_dir(run / "work" / "bad"), ["invalid"])
    quant_dir(run / "work" / "good")
    reply = host.handle(Request(verb, ["bad"], str(run / "work")))
    assert reply.exit == 2 and "quantization_config is not a layout" in reply.stderr
    assert host.entries == [] and not host.sched.jobs  # nothing was frozen or queued
    assert host.handle(Request("handin", ["good"], str(run / "work"))).exit == 0


def test_more_than_4096_entries_are_refused(tmp_path, cell):
    anchor, ref = cell
    run = tmp_path / "run"
    d = quant_dir(run / "work" / "many")
    for i in range(4100):
        (d / f"LICENSE-{i}").write_text("")
    reply = _host(run, anchor=anchor, format_reference=ref).handle(
        Request("handin", ["many"], str(run / "work"))
    )
    assert reply.exit == 2 and "more than 4096 files" in reply.stderr


def test_size_cap_is_twice_the_anchor_plus_a_gib_and_an_empty_anchor_is_refused(tmp_path, cell):
    from qda_agent.judge.freeze import tree_size

    anchor, ref = cell
    assert _host(tmp_path / "r1", anchor=anchor, format_reference=ref).max_bytes == 2 * tree_size(anchor) + (
        1 << 30
    )
    assert _host(tmp_path / "r3", format_reference=ref).max_bytes == 200 << 30  # no anchor, no setting
    assert (  # an explicit cap replaces both
        _host(tmp_path / "r4", anchor=anchor, format_reference=ref, max_copy_bytes=4096).max_bytes == 4096
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="has no files"):
        _host(tmp_path / "r2", anchor=empty, format_reference=ref)
    with pytest.raises(ValueError, match="has no files"):  # an explicit cap does not hide a broken anchor
        _host(tmp_path / "r5", anchor=empty, format_reference=ref, max_copy_bytes=4096)


def hub_snapshot(root, anchor):
    """A Hugging Face hub cache holding `anchor`: snapshots/<rev>/<file> -> ../../blobs/<hash>."""
    repo = root / "models--org--tiny"
    (repo / "blobs").mkdir(parents=True)
    snap = repo / "snapshots" / "0123abc"
    for f in sorted(p for p in anchor.rglob("*") if p.is_file()):
        blob = repo / "blobs" / hashlib.sha256(f.read_bytes()).hexdigest()
        shutil.copyfile(f, blob)
        link = snap / f.relative_to(anchor)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snap


def test_a_hub_snapshot_anchor_counts_its_blobs_and_a_served_link_out_is_refused(tmp_path, cell):
    from qda_agent.judge.freeze import model_size, tree_size

    anchor, ref = cell
    snap = hub_snapshot(tmp_path / "hub", anchor)
    assert tree_size(snap) == 0  # frozen copies never follow links
    assert model_size(snap) == tree_size(anchor)
    host = _host(tmp_path / "r1", anchor=snap, format_reference=ref)
    assert host.max_bytes == 2 * tree_size(anchor) + (1 << 30)
    secret = tmp_path / "secret.txt"
    secret.write_text("not the model's")
    (snap / "extra.txt").symlink_to(
        secret
    )  # a release's other file: never read, so neither sized nor checked
    assert _host(tmp_path / "r2", anchor=snap, format_reference=ref).max_bytes == host.max_bytes
    (snap / "config.json").unlink()
    (snap / "config.json").symlink_to(secret)  # a served file's link, out of the repository
    with pytest.raises(ValueError, match=r"judge\.anchor .*not a regular file inside the model"):
        _host(tmp_path / "r3", anchor=snap, format_reference=ref)


def _w3_head_ckpt(d: Path, head: tuple[int, int] | None, *, stored_group: int | None = None, body_group=128):
    """A W3 g128 checkpoint (hidden 128) whose lm_head is packed `head` = (bits, group) or bf16 (None).

    `stored_group` writes the head's scales for another group than it declares (a declaration that lies).
    """
    hidden, vocab = 128, 16
    groups = {"body": {"targets": ["Linear"], "weights": _weights(3, body_group)}}
    tensors = {
        "model.embed_tokens.weight": ("BF16", (vocab, hidden), None),
        "model.layers.0.self_attn.q_proj.weight_packed": ("I32", (hidden, hidden * 3 // 32), None),
        "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (hidden, hidden // body_group), None),
    }
    ignore = ["model.embed_tokens"]
    if head is None:
        tensors["lm_head.weight"] = ("BF16", (vocab, hidden), None)
        ignore.append("lm_head")
    else:
        bits, group = head
        tensors["lm_head.weight_packed"] = ("I32", (vocab, hidden * bits // 32), None)
        tensors["lm_head.weight_scale"] = ("BF16", (vocab, hidden // (stored_group or group)), None)
        groups["head"] = {"targets": ["lm_head"], "weights": _weights(bits, group)}
    cfg = {**CONFIG, "hidden_size": hidden}
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": groups,
        "ignore": ignore,
    }
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return d


W3_MENU = ["bf16", "w3a16-sym-g128", "w4a16-sym-g128"]


@pytest.mark.parametrize(
    ("head", "stored_group", "body_group", "head_format", "why"),
    [
        ((4, 64), None, 128, "w4a16-sym-g64", None),  # the paper's W3A16 head
        ((4, 64), None, 128, None, "the task keeps it 16-bit"),  # no head_format: the 16-bit rule
        (None, None, 128, "w4a16-sym-g64", "lm_head is declared bf16"),  # the head must be the declared one
        ((4, 128), None, 128, "w4a16-sym-g64", "lm_head is declared w4a16 (g128, sym)"),
        ((4, 64), 128, 128, "w4a16-sym-g64", "lm_head is stored 4-bit g128 (read from its tensors)"),
        ((4, 64), None, 64, "w4a16-sym-g64", "outside judge.format_menu"),  # the exemption is lm_head's only
    ],
)
def test_head_format_fixes_lm_head_only_and_reads_it_back(
    tmp_path, head, stored_group, body_group, head_format, why
):
    """Publishing an already-built paper W3A16 checkpoint: its INT4 g64 head is checked from its tensors."""
    d = _w3_head_ckpt(tmp_path / "w3", head, stored_group=stored_group, body_group=body_group)
    got = format_problem(d, W3_MENU, head_format=head_format)
    if why is None:
        assert got is None
    else:
        assert got is not None and why in got, got


def test_judge_head_pins_an_exception_models_released_head(tmp_path):
    """An exception row names the release's head format; the 16-bit default refuses it."""
    d = _w3_head_ckpt(tmp_path / "w3", (4, 64))
    assert format_problem(d, W3_MENU) is not None  # 16-bit by default
    assert format_problem(d, W3_MENU, head="w4a16-sym-g64") is None
    assert (
        format_problem(d, W3_MENU, head="w4a16-sym-g128") is not None
    )  # the release's own group, not another


#: A Blackwell row's menu: bf16, NVFP4 W4A4 and FP8 W8A8.
MIXED_MENU = ["bf16", "w4a4-nvfp4", "w8a8-fp8"]
_LANG = "model.language_model"
#: The config groups a mixed-precision NVFP4+FP8 winner declares, from its `config.json`.
_NVFP4_W = {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "tensor_group", "group_size": 16}
_NVFP4_A = {**_NVFP4_W, "dynamic": "local"}
_FP8_W = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "group_size": None}
_FP8_A = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "token",
    "group_size": None,
    "dynamic": True,
}


def _nvfp4(module: str, rows: int, cols: int) -> dict:
    """An NVFP4 module's stored parts: E2M1 codes two a U8 byte, an E4M3 scale per 16, two fp32 globals."""
    return {
        f"{module}.weight_packed": ("U8", (rows, cols // 2), None),
        f"{module}.weight_scale": ("F8_E4M3", (rows, cols // 16), None),
        f"{module}.weight_global_scale": ("F32", (1,), None),
        f"{module}.input_global_scale": ("F32", (1,), None),
    }


def _fp8(module: str, rows: int, cols: int) -> dict:
    """An FP8 W8A8 module's stored parts: the E4M3 values themselves, one F32 scale an output row."""
    return {
        f"{module}.weight": ("F8_E4M3", (rows, cols), None),
        f"{module}.weight_scale": ("F32", (rows, 1), None),
    }


def _mixed_pair(root: Path, *, mixed: bool = True, conflict: bool = False) -> tuple[Path, Path]:
    """A real-shaped mixed-precision checkpoint and the bf16 original it was quantized from.

    The config and header layout are the ones a q36-n1-class winner stores: NVFP4 expert projections
    (`U8` `[out, in/2]` codes, an `F8_E4M3` scale `[out, in/16]`, both fp32 globals), FP8 W8A8 attention and
    output head (`F8_E4M3` `[out, in]` under an `F32` `[out, 1]` scale), a bf16 embedding and MoE router, and
    `format: mixed-precision` with one config group a family. `conflict` stores expert 1's up_proj in FP8
    beside its NVFP4 gate_proj, a fused unit the engine serves with one kernel.
    """
    hidden, vocab = 128, 256
    attn = f"{_LANG}.layers.0.linear_attn"
    anchor, d = root / "anchor", root / "mixed"
    anchor.mkdir(parents=True)
    d.mkdir(parents=True)
    original = {
        f"{_LANG}.embed_tokens.weight": ("BF16", (vocab, hidden), None),
        f"{attn}.in_proj_qkv.weight": ("BF16", (192, hidden), None),
        f"{attn}.out_proj.weight": ("BF16", (hidden, hidden), None),
        f"{_LANG}.layers.0.mlp.gate.weight": ("BF16", (8, hidden), None),
        "lm_head.weight": ("BF16", (vocab, hidden), None),
    }
    tensors = {
        f"{_LANG}.embed_tokens.weight": ("BF16", (vocab, hidden), None),
        f"{_LANG}.layers.0.mlp.gate.weight": ("BF16", (8, hidden), None),
        **_fp8(f"{attn}.in_proj_qkv", 192, hidden),
        **_fp8(f"{attn}.out_proj", hidden, hidden),
        **_fp8("lm_head", vocab, hidden),
    }
    for i in range(2):
        bank = f"{_LANG}.layers.0.mlp.experts.{i}"
        for leaf, (rows, cols) in (
            ("gate_proj", (64, hidden)),
            ("up_proj", (64, hidden)),
            ("down_proj", (hidden, 64)),
        ):
            original[f"{bank}.{leaf}.weight"] = ("BF16", (rows, cols), None)
            fp8_leaf = conflict and i == 1 and leaf == "up_proj"
            tensors.update(
                _fp8(f"{bank}.{leaf}", rows, cols) if fp8_leaf else _nvfp4(f"{bank}.{leaf}", rows, cols)
            )
    fp8 = {
        "weights": _FP8_W,
        "input_activations": _FP8_A,
        "output_activations": None,
        "format": "float-quantized",
    }
    groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": _NVFP4_W,
            "input_activations": _NVFP4_A,
            "output_activations": None,
            "format": "nvfp4-pack-quantized",
        },
        "group_head": {"targets": ["re:.*lm_head$"], **fp8},
        "group_1": {"targets": [f"{attn}.in_proj_qkv", f"{attn}.out_proj"], **fp8},
    }
    if conflict:
        groups["group_2"] = {"targets": [f"{_LANG}.layers.0.mlp.experts.1.up_proj"], **fp8}
    cfg = {
        **CONFIG,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "quantization_config": {
            **({"format": "mixed-precision"} if mixed else {}),
            "quant_method": "compressed-tensors",
            "config_groups": groups,
            "ignore": [
                f"{_LANG}.embed_tokens",
                f"{_LANG}.layers.0.mlp.gate",
                f"{attn}.in_proj_a",
                f"{attn}.in_proj_b",
                f"{attn}.in_proj_ba",
                f"{_LANG}.layers.0.mlp.shared_expert.gate_proj",
                f"{_LANG}.layers.0.mlp.shared_expert.up_proj",
                f"{_LANG}.layers.0.mlp.shared_expert.down_proj",
                f"{_LANG}.layers.0.mlp.shared_expert_gate",
            ],
        },
    }
    (anchor / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": hidden}))
    (anchor / "tokenizer.json").write_text("{}")
    write_safetensors(anchor / "model.safetensors", original)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(d / "model.safetensors", tensors)
    return anchor, d


def _fp8_bits(rows: int, cols: int) -> int:
    """An FP8 module's stored bits: E4M3 values and an F32 scale a row."""
    return 8 * (rows * cols + 4 * rows)


def _nvfp4_bits(rows: int, cols: int) -> int:
    """An NVFP4 module's stored bits: codes two a byte, an E4M3 scale per 16 values, two fp32 globals."""
    return 8 * (rows * cols // 2 + rows * cols // 16 + 8)


def test_the_fp8_family_has_a_key_of_its_own():
    """`w8a8-fp8` is not the INT8 `w8a8` family, and NVFP4's key is its activation width's `w4a4`."""
    from qda_agent.judge.checks.formats import Stored

    assert format_key("w8a8-fp8") == "w8a8-fp8"
    assert format_key("w8a8i") == "w8a8"  # the INT8 W8A8 family
    assert format_key("w4a4-nvfp4") == "w4a4"
    fp8 = Stored(8, -1, True, "float", "channel", (8, "float", True))
    assert fp8.family == "w8a8-fp8" and str(fp8) == "w8a8-fp8 (channel, sym)"


def test_a_real_shaped_nvfp4_fp8_mixed_checkpoint_is_admitted_and_priced(tmp_path):
    """One group a family: NVFP4 expert projections beside FP8 attention under `format: mixed-precision`."""
    from qda_agent.judge.checks import budget

    anchor, d = _mixed_pair(tmp_path)
    assert format_problem(d, MIXED_MENU, anchor=anchor, head="w8a8-fp8", floor=False) is None
    weights = 192 * 128 + 128 * 128 + 2 * (2 * 64 * 128 + 128 * 64)
    bits = _fp8_bits(192, 128) + _fp8_bits(128, 128) + 2 * (2 * _nvfp4_bits(64, 128) + _nvfp4_bits(128, 64))
    # the two experts and the attention; no router, and the FP8 head (256 x 128) is outside the body
    assert budget.denominator(anchor) == weights
    assert budget.bpw(d, anchor) == pytest.approx(bits / weights)
    # without the declaration the same tensors break the layer's one activation format
    plain = _mixed_pair(tmp_path / "plain", mixed=False)[1]
    why = format_problem(plain, MIXED_MENU, head="w8a8-fp8", floor=False)
    assert why is not None and "mixes two activation formats" in why
    # the default 16-bit rule still refuses the FP8 head
    why = format_problem(d, MIXED_MENU, anchor=anchor, floor=False)
    assert why is not None and "the task keeps it 16-bit" in why


def test_a_mixed_checkpoint_still_needs_each_fused_unit_at_one_format(tmp_path):
    """The mixed exception lifts the layer's activation rule, not the kernel's: one bank, one format."""
    _, d = _mixed_pair(tmp_path, conflict=True)
    why = format_problem(d, MIXED_MENU, head="w8a8-fp8")
    assert why is not None and "routed expert bank" in why and "experts.1.up_proj is w8a8-fp8" in why


def _fp8_head_ckpt(
    d: Path, *, weight: tuple[str, tuple[int, ...]] = ("F8_E4M3", (16, 8)), scale: tuple[int, ...] = (16, 1)
) -> Path:
    """A W4 body whose lm_head is declared FP8 W8A8 and stored `weight` under an F32 scale of `scale` shape."""
    groups = {
        "body": {"targets": ["Linear"], "weights": _weights(4)},
        "head": {"targets": ["re:.*lm_head$"], "weights": _FP8_W, "input_activations": _FP8_A},
    }
    cfg = {
        **CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": groups,
            "ignore": ["model.embed_tokens"],
        },
    }
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    write_safetensors(
        d / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (16, 8), None),
            "model.layers.0.self_attn.q_proj.weight_packed": ("I32", (8, 1), None),
            "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (8, 1), None),
            "lm_head.weight": (*weight, None),
            "lm_head.weight_scale": ("F32", scale, None),
        },
    )
    return d


@pytest.mark.parametrize(
    ("weight", "scale", "why"),
    [
        (("F8_E4M3", (16, 8)), (16, 1), None),
        (("BF16", (16, 8)), (16, 1), "not an E4M3 weight under an fp32 scale per output row"),
        (("F8_E4M3", (16, 8)), (8, 2), "not an E4M3 weight under an fp32 scale per output row"),
    ],
)
def test_an_fp8_output_head_is_read_back_from_its_tensors(tmp_path, weight, scale, why):
    """`judge.head: w8a8-fp8` trusts the declaration only as far as the stored weight and scale agree."""
    d = _fp8_head_ckpt(tmp_path / "fp8head", weight=weight, scale=scale)
    got = format_problem(d, ["bf16", "w4a16", "w8a8-fp8"], head="w8a8-fp8")
    if why is None:
        assert got is None
    else:
        assert got is not None and why in got


def _symmetric(d: Path, modules: dict[str, bool]) -> Path:
    """Rewrite the stored symmetry of the named modules' groups in `d`'s config.json."""
    cfg = json.loads((d / "config.json").read_text())
    for g in cfg["quantization_config"]["config_groups"].values():
        for module, sym in modules.items():
            if module in g["targets"]:
                g["weights"]["symmetric"] = sym
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _row(tmp_path, fmt, menu=None):
    from qda_agent.config import RunConfig

    judge = {"kl_cmd": ["score"], **({"format_menu": menu} if menu is not None else {})}
    run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    return RunConfig(
        contamination="none",
        model="tiny/model",
        format=fmt,
        run_dir=run,
        budget_gpu_hours=1,
        cards=[0],
        judge=judge,
    )


def test_symmetry_is_in_the_menu_key_and_bare_entries_bind_to_the_row(tmp_path, cell):
    """The gate keys on (family, group, symmetric): a bare entry pins the row's symmetry; a menu may offer both."""
    anchor, ref = cell
    pinned = _row(tmp_path, "w4a16-sym-g128").judge.format_menu
    assert pinned == ["bf16", "w8a16-sym-g128", "w4a16-sym-g128"]
    layers = {0: _leaves(4), 1: _leaves(4)}
    sym = _attn_ckpt(tmp_path / "sym", layers)
    asym = _symmetric(
        _attn_ckpt(tmp_path / "asym", layers),
        {f"model.layers.{i}.self_attn.{x}_proj": False for i in (0, 1) for x in "qkv"},
    )
    assert files_problem(sym, anchor, ref, format_menu=pinned) is None
    assert "(g128, asym), outside judge.format_menu" in files_problem(asym, anchor, ref, format_menu=pinned)
    offered = _row(  # only a row that names no symmetry may offer both for its own family
        tmp_path, "w4a16-g128", ["bf16", "w4a16-sym-g128", "w4a16-asym-g128", "w8a16"]
    ).judge.format_menu
    assert offered == ["bf16", "w4a16-sym-g128", "w4a16-asym-g128", "w8a16-sym-g128"]
    assert files_problem(asym, anchor, ref, format_menu=offered) is None
    mixed = _symmetric(
        _attn_ckpt(tmp_path / "mixed", layers), {f"model.layers.1.self_attn.{x}_proj": False for x in "qkv"}
    )
    assert "mix symmetries" in files_problem(mixed, anchor, ref, format_menu=offered)


def test_promoted_formats_are_symmetric_only_and_an_unspelled_row_keeps_both(tmp_path):
    """A wider menu entry named asymmetric is refused; a row spelling no symmetry admits both for its family."""
    with pytest.raises(ValueError, match="promoted modules are stored symmetric"):
        _row(tmp_path, "w4a16-asym-g64", ["bf16", "w4a16", "w8a16-asym"])
    assert _row(tmp_path, "w4a16-asym-g64").judge.format_menu == ["bf16", "w8a16-sym-g64", "w4a16-asym-g64"]
    assert _row(tmp_path, "w4a16-g64").judge.format_menu == ["bf16", "w8a16-sym-g64", "w4a16-g64"]


@pytest.mark.parametrize("shared", ["shared_expert", "shared_experts"])
def test_routed_and_shared_experts_keep_their_symmetry_apart_from_dense(shared):
    """A bare row stores routed banks and the shared experts beside them symmetric, and dense modules asymmetric."""
    from qda_agent.judge.checks.formats import Stored, _row_problem

    def at(sym):
        return Stored(4, 64, sym, "int", "group", None)

    layer = "model.layers.0"
    formats = {
        f"{layer}.self_attn.q_proj": at(False),
        f"{layer}.mlp.experts.0.gate_proj": at(True),
        f"{layer}.mlp.experts.gate_up_proj": at(True),
        f"{layer}.mlp.{shared}.gate_proj": at(True),
        f"{layer}.mlp.{shared}.down_proj": at(True),
    }
    assert _row_problem(formats) is None
    assert "expert w4a16 modules mix symmetries" in _row_problem(
        {**formats, f"{layer}.mlp.{shared}.up_proj": at(False)}
    )
    assert "dense w4a16 modules mix symmetries" in _row_problem(
        {**formats, f"{layer}.self_attn.k_proj": at(True)}
    )


def test_a_menu_offers_every_family_at_every_group_it_offers(tmp_path):
    """One checkpoint takes one group, so a family missing at an offered group is refused at config time."""
    with pytest.raises(ValueError, match="w8a16 is offered at g128 but not at g64"):
        _row(tmp_path, "w4a16-asym-g128", ["bf16", "w8a16", "w4a16", "w4a16-asym-g64"])
    with pytest.raises(ValueError, match="w8a16 is offered at g64 but not at g128"):
        _row(tmp_path, "w4a16-asym-g64", ["bf16", "w8a16", "w4a16", "w4a16-asym-g128"])
    both = _row(tmp_path, "w4a16-asym-g128", ["bf16", "w8a16", "w4a16", "w4a16-asym-g64", "w8a16-g64"])
    assert both.judge.format_menu == [
        "bf16",
        "w8a16-sym-g128",
        "w4a16-asym-g128",
        "w4a16-asym-g64",
        "w8a16-sym-g64",
    ]
    assert _row(tmp_path, "w3a16-sym-g128").judge.format_menu == [
        "bf16",
        "w3a16-sym-g128",
        "w4a16-sym-g128",
        "w8a16-sym-g128",
    ]


def test_a_spelled_row_fixes_its_own_familys_symmetry_in_the_menu(tmp_path):
    """An own-family entry naming the other symmetry would let the agent build the whole checkpoint at it."""
    with pytest.raises(ValueError, match="w4a16-sym-g64 is sym and the row's w4a16-asym-g128 is asym"):
        _row(tmp_path, "w4a16-asym-g128", ["bf16", "w8a16", "w4a16", "w8a16-sym-g64", "w4a16-sym-g64"])
    with pytest.raises(ValueError, match="w3a16-asym-g128 is asym and the row's w3a16-sym-g128 is sym"):
        _row(tmp_path, "w3a16-sym-g128", ["bf16", "w3a16", "w3a16-asym-g128", "w4a16", "w8a16"])
    fixed = _row(tmp_path, "w4a16-asym-g128", ["bf16", "w8a16", "w4a16", "w8a16-sym-g64", "w4a16-asym-g64"])
    assert fixed.judge.format_menu == [
        "bf16",
        "w8a16-sym-g128",
        "w4a16-asym-g128",
        "w8a16-sym-g64",
        "w4a16-asym-g64",
    ]
    # Exempt: the promoted family is sym by rule, and a row naming no symmetry keeps both for its family.
    assert (
        _row(tmp_path, "w4a16-asym-g128", ["bf16", "w8a16-sym-g128", "w4a16"]).judge.format_menu[1]
        == "w8a16-sym-g128"
    )
    assert (
        "w4a16-sym-g128"
        in _row(tmp_path, "w4a16-g128", ["bf16", "w8a16", "w4a16-sym-g128"]).judge.format_menu
    )


def _wide_model(
    root: Path, name: str, *, bits: int | None = None, packed: bool = False, ignore=None, gate: bool = False
) -> Path:
    """A one-layer model whose only decoder Linear is 64 x 64 (a scalar gate is smaller than 32 rows)."""
    from safetensors_kit import write_safetensors

    d = root / name
    d.mkdir()
    cfg = {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 1}
    if bits is not None:
        q: dict = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": bits,
                        "group_size": 64,
                        "strategy": "group",
                        "type": "int",
                        "symmetric": True,
                    },
                    "targets": ["Linear"],
                }
            },
        }
        if ignore is not None:
            q["ignore"] = ignore
        cfg["quantization_config"] = q
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text("{}")
    tensors = {
        "model.embed_tokens.weight": ("BF16", (64, 64), b"\x01\x02"),
        "model.layers.0.mlp.down_proj.weight": ("BF16", (64, 64), b"\x05\x06"),
    }
    if gate:  # a scalar gate the planner keeps bf16 (`out_features < 32`)
        tensors["model.layers.0.mlp.shared_expert_gate.weight"] = ("BF16", (1, 64), b"\x09\x0a")
    if packed:
        del tensors["model.layers.0.mlp.down_proj.weight"]
        tensors["model.layers.0.mlp.down_proj.weight_packed"] = ("I32", (64, 16), b"\x07\x08")
        tensors["model.layers.0.mlp.down_proj.weight_scale"] = ("BF16", (1, 64), b"\x03\x04")
        tensors["model.layers.0.mlp.down_proj.weight_shape"] = (
            "I32",
            (2,),
            b"\x40\x00\x00\x00\x40\x00\x00\x00",
        )
    write_safetensors(d / "model.safetensors", tensors)
    return d


_MENU = ["bf16", "w3a16", "w4a16", "w8a16-sym-g64"]


def test_without_a_budget_an_all_bf16_build_is_refused_as_no_quantization(tmp_path):
    """The bpw gate used to be the only check that noticed an unquantized 'candidate'. The floor holds
    with `judge.budget_bpw` unset: every decoder Linear the row implies must be stored quantized."""
    anchor = _wide_model(tmp_path, "anchor")
    body = _wide_model(tmp_path, "all-bf16", bits=8, ignore=["re:.*"])
    why = files_problem(body, anchor, None, format_menu=_MENU)
    assert why is not None and "0.0% of the decoder-Linear weights" in why
    assert "4096 of 4096 would stay bf16" in why
    # a candidate 0 (`smoke`) is the row's baseline, not a candidate: the floor is not applied to it
    assert files_problem(body, anchor, None, format_menu=_MENU, floor=False) is None


def test_a_quantized_build_passes_the_floor_without_a_budget(tmp_path):
    anchor = _wide_model(tmp_path, "anchor")
    qt = _wide_model(tmp_path, "w8", bits=8, packed=True)
    assert files_problem(qt, anchor, None, format_menu=_MENU) is None


def test_a_module_a_build_may_keep_bf16_does_not_count_against_the_floor(tmp_path):
    """A scalar gate (`out_features < 32`) is kept bf16 by the planner (`qda_quant.arch`), so it is not one of
    the rows the floor counts: a quantized body beside it passes."""
    anchor = _wide_model(tmp_path, "anchor", gate=True)
    qt = _wide_model(tmp_path, "w8", bits=8, packed=True, gate=True)
    assert files_problem(qt, anchor, None, format_menu=_MENU) is None


def test_a_handin_of_the_original_is_refused_even_when_no_baseline_holds_it(tmp_path):
    """The anchor is a copy reference too, so handing in the original is refused even with no baseline
    configured (it used to be compared only with `baselines` / `judge.format_reference`)."""
    from test_judge import Rig

    anchor = quant_dir(tmp_path / "anchor", weights=b"model")  # quantized: near_copy can pair tensors
    r = Rig(tmp_path / "rig", anchor=str(anchor), baselines=[])
    shutil.copytree(anchor, r.work / "out" / "copy")
    rep = r.call("handin", "out/copy")
    assert rep.exit == 0 and "a baseline checkpoint as-is" in rep.stdout  # admitted, then refused
    assert [e.valid for e in r.host.entries if e.id == "h1"] == [False]
